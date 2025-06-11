"""
ddp_light.py

轻量级DDP训练策略，专门用于训练参数很少的VLMCogACT模型。
由于大部分参数（Qwen-72B和action_model）都被冻结，只需要对很少的可训练参数进行简单的数据并行。
"""

import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.overwatch import initialize_overwatch

from vla.cogvla import VLMCogACT
from training.strategies.base_strategy_cogact import TrainingStrategy

# 初始化Overwatch =>> 包装`logging.Logger`
overwatch = initialize_overwatch(__name__)


class DDPLightStrategy(TrainingStrategy):
    """轻量级DDP策略，专为A100 GPU优化，移除混合精度训练"""

    def __init__(
        self,
        vlm: VLMCogACT,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        worker_init_fn=None,
        # 简化参数，移除混合精度相关
        accumulate_grad_batches: int = 1,
        offload_optimizer: bool = False,
        **kwargs,
    ) -> None:
        # 移除混合精度相关参数
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=False,
            enable_mixed_precision_training=False,  # 强制禁用
            reduce_in_full_precision=False,
            mixed_precision_dtype=torch.float32,  # 不使用，但保持兼容
            worker_init_fn=worker_init_fn,
            **kwargs,
        )
        self.accumulate_grad_batches = accumulate_grad_batches
        self.offload_optimizer = offload_optimizer
        self.effective_batch_size = global_batch_size * accumulate_grad_batches

        # 移除GradScaler相关代码
        self.grad_scaler = None

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """设置DDP训练环境，专为A100优化"""

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.vlm.parameters())

        # 按模块统计参数
        cross_attn_params = sum(p.numel() for p in self.vlm.cross_attention.parameters())
        action_model_params = sum(p.numel() for p in self.vlm.action_model.parameters() if p.requires_grad)

        overwatch.info(
            f"VLMCogACT参数分布:\n"
            f"  总参数: {total_params:,}\n"
            f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)\n"
            f"  交叉注意力: {cross_attn_params:,}\n"
            f"  动作模型(可训练): {action_model_params:,}"
        )

        if trainable_params == 0:
            raise ValueError("VLMCogACT模型中没有找到可训练参数!")

        # 智能设备分配
        device = torch.cuda.current_device()

        # 检查GPU内存使用情况
        self._log_memory_usage("训练开始前")

        # A100优化的DDP配置
        ddp_kwargs = {
            "device_ids": [device],
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "static_graph": True,
            "bucket_cap_mb": 100,  # A100可以使用更大的bucket
        }

        # 优化的DDP包装策略 - A100专用配置
        # 1. 交叉注意力模块 - 小而重要，放在主设备
        if hasattr(self.vlm, "cross_attention"):
            self.vlm.cross_attention = self.vlm.cross_attention.to(device)
            self.vlm.cross_attention = DDP(self.vlm.cross_attention, **ddp_kwargs)

        # 2. 认知投影器 - 如果存在且可训练
        if hasattr(self.vlm, "cognition_projector") and not isinstance(self.vlm.cognition_projector, nn.Identity):
            trainable_projector = any(p.requires_grad for p in self.vlm.cognition_projector.parameters())
            if trainable_projector:
                self.vlm.cognition_projector = self.vlm.cognition_projector.to(device)
                self.vlm.cognition_projector = DDP(self.vlm.cognition_projector, **ddp_kwargs)

        # 3. 动作模型 - 大部分可训练参数，需要特殊处理
        if hasattr(self.vlm, "action_model"):
            action_trainable = any(p.requires_grad for p in self.vlm.action_model.parameters())
            if action_trainable:
                self.vlm.action_model = self.vlm.action_model.to(device)
                self.vlm.action_model = DDP(self.vlm.action_model, **ddp_kwargs)

        overwatch.info("已将VLMCogACT的可训练模块包装为A100优化的DDP")
        self._log_memory_usage("DDP包装后")

        # 创建优化器和学习率调度器
        n_train_examples = math.ceil(n_train_examples / self.effective_batch_size) * self.effective_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.effective_batch_size
        else:
            num_training_steps = self.max_steps

        # 创建参数组 - 更细粒度的权重衰减控制
        decay, no_decay = [], []
        for name, param in self.vlm.named_parameters():
            if not param.requires_grad:
                continue

            # 更精确的no_decay规则
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "bn" in name.lower()
                or "ln" in name.lower()
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        # 不同模块使用不同的学习率
        param_groups = [
            {"params": decay, "weight_decay": self.weight_decay, "lr": self.learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": self.learning_rate},
        ]

        # A100优化的AdamW配置
        self.optimizer = AdamW(
            param_groups, 
            eps=1e-8, 
            betas=(0.9, 0.95),
            fused=True  # A100支持fused AdamW
        )

        # 学习率调度器
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
        elif self.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f"不支持的学习率调度类型: {self.lr_scheduler_type}")

        # 记录详细的训练设置
        overwatch.info(
            "DDPLight策略 =>> A100优化训练设置完成:\n"
            f"         |-> 全局批次大小 = {self.global_batch_size}\n"
            f"         |-> 有效批次大小 = {self.effective_batch_size}\n"
            f"         |-> 每设备批次大小 = {self.per_device_batch_size}\n"
            f"         |-> 梯度累积步数 = {self.accumulate_grad_batches}\n"
            f"         |-> 分布式世界大小 = {overwatch.world_size()}\n"
            f"         |-> 学习率 = {self.learning_rate}\n"
            f"         |-> 权重衰减 = {self.weight_decay}\n"
            f"         |-> 学习率调度类型 = {self.lr_scheduler_type}\n"
            f"         |-> 混合精度训练 = 禁用 (A100使用FP32)\n"
            f"         |-> 优化器卸载 = {self.offload_optimizer}\n"
            f"         |-> 数据集大小 = {n_train_examples} 样本\n"
            f"         |-> 最大步数 = {num_training_steps}\n"
        )

    def _log_memory_usage(self, stage: str) -> None:
        """记录GPU内存使用情况"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                overwatch.info(f"{stage} - GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    def clip_grad_norm(self) -> None:
        """简化的梯度裁剪，移除混合精度支持"""
        trainable_params = [p for p in self.vlm.parameters() if p.requires_grad and p.grad is not None]

        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.max_grad_norm)

    def optimizer_step(self) -> None:
        """简化的优化器步骤，移除混合精度支持"""
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """简化版检查点保存，移除混合精度相关代码"""

        model_state_dicts = {}

        try:
            # 交叉注意力模块
            if hasattr(self.vlm, "cross_attention"):
                if isinstance(self.vlm.cross_attention, DDP):
                    model_state_dicts["cross_attention"] = self.vlm.cross_attention.module.state_dict()
                else:
                    model_state_dicts["cross_attention"] = self.vlm.cross_attention.state_dict()

            # 认知投影器
            if hasattr(self.vlm, "cognition_projector"):
                if isinstance(self.vlm.cognition_projector, DDP):
                    model_state_dicts["cognition_projector"] = self.vlm.cognition_projector.module.state_dict()
                elif not isinstance(self.vlm.cognition_projector, nn.Identity):
                    model_state_dicts["cognition_projector"] = self.vlm.cognition_projector.state_dict()

            # 动作模型
            if hasattr(self.vlm, "action_model"):
                if isinstance(self.vlm.action_model, DDP):
                    model_state_dicts["action_model"] = self.vlm.action_model.module.state_dict()
                else:
                    model_state_dicts["action_model"] = self.vlm.action_model.state_dict()

            # EMA状态（如果使用）
            if hasattr(self.vlm, "ema_diffusion") and self.vlm.use_ema:
                model_state_dicts["ema_diffusion"] = self.vlm.ema_diffusion.state_dict()

        except RuntimeError as e:
            overwatch.error(f"保存模型状态时出错: {e}")
            return

        if overwatch.is_rank_zero():
            checkpoint_dir = run_dir / "checkpoints"
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

            try:
                torch.save({"model": model_state_dicts}, checkpoint_path)

                # 保存优化器状态（移除grad_scaler）
                if not only_trainable:
                    optimizer_path = checkpoint_path.with_suffix(".optimizer")
                    optimizer_checkpoint = {
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": {
                            "epoch": epoch,
                            "global_step": global_step,
                            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                        },
                    }
                    torch.save(optimizer_checkpoint, optimizer_path)

                overwatch.info(f"保存检查点到 {checkpoint_path}")
                self._log_memory_usage("保存检查点后")

            except Exception as e:
                overwatch.error(f"保存检查点失败: {e}")

        dist.barrier()
