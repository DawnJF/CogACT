"""

使用HuggingFace Trainer简化的训练策略，专用于冻结大部分参数的VLMCogACT模型。
自动处理多GPU训练，无需手动管理DDP。
"""

from pathlib import Path
from typing import Optional, Callable
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import IterableDataset

from prismatic.overwatch import initialize_overwatch
from vla.cogvla import VLMCogACT
from training.metrics import VLAMetrics
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

overwatch = initialize_overwatch(__name__)


class VLATrainer(Trainer):
    """自定义Trainer用于VLA训练，添加EMA支持和自定义指标"""

    def __init__(self, *args, **kwargs):
        self.metrics = kwargs.pop("metrics", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # 自定义损失计算

        loss, outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            state=inputs["state"],
            actions=inputs["actions"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            action_masks=inputs.get("action_masks", None),
            labels=inputs.get("labels", None),
            output_hidden_states=True,
        )

        # 记录指标
        if self.metrics is not None:
            self.metrics.commit(loss=loss)
            if self.state.global_step % self.args.logging_steps == 0:
                lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.args.learning_rate
                epoch = (self.state.global_step + 1) // (
                    len(self.train_dataset) // self.args.per_device_train_batch_size
                )
                self.metrics.commit(update_step_time=True, global_step=self.state.global_step, epoch=epoch, lr=lr)
                self.metrics.push()

        return (loss, outputs) if return_outputs else loss


class TrainerStrategy:
    """使用HF Trainer的简化训练策略"""

    def __init__(
        self,
        vlm: VLMCogACT,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        offload_optimizer: bool = False,
        **kwargs,
    ) -> None:
        self.vlm = vlm
        self.device_id = device_id
        self.epochs = epochs
        self.max_steps = max_steps
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.worker_init_fn = worker_init_fn
        self.offload_optimizer = offload_optimizer

        # 计算梯度累积步数
        self.grad_accumulation_steps = self.global_batch_size // (
            self.per_device_batch_size * max(1, torch.cuda.device_count())
        )

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.vlm.parameters())
        overwatch.info(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")

        if trainable_params == 0:
            overwatch.warning("没有可训练参数，将以纯推理模式运行")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """使用HF Trainer的设置"""
        pass  # 所有设置将在run_training中完成

    def run_vla_training(
        self,
        run_dir,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        action_model: bool = False,
    ) -> None:
        """使用HF Trainer的主训练循环"""

        overwatch.info(f"CUDA available: {torch.cuda.is_available()}")
        overwatch.info(f"Device count: {torch.cuda.device_count()}")
        if torch.distributed.is_initialized():
            overwatch.info(f"Distributed world size: {torch.distributed.get_world_size()}")

        # 计算训练总步数
        if self.max_steps is None:
            num_examples = len(vla_dataset)
            num_update_steps_per_epoch = num_examples // self.global_batch_size
            num_training_steps = num_update_steps_per_epoch * self.epochs
        else:
            num_training_steps = self.max_steps

        warmup_steps = int(num_training_steps * self.warmup_ratio) if self.warmup_ratio > 0 else 0

        # 训练参数配置
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.grad_accumulation_steps,
            learning_rate=self.learning_rate,
            # weight_decay=self.weight_decay,
            # max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.epochs,
            # max_steps=num_training_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_steps=warmup_steps,
            logging_steps=5,
            save_steps=save_interval,
            save_total_limit=3,
            dataloader_num_workers=0,
            report_to=[],  # 禁用wandb等报告
            optim="adamw_torch",
            bf16=True,
        )

        # 创建自定义Trainer
        trainer = VLATrainer(
            model=self.vlm,
            args=training_args,
            train_dataset=vla_dataset,
            data_collator=collator,
            metrics=metrics,
        )

        # 开始训练
        trainer.train()

        # 保存最终模型
        if save_full_model:
            trainer.save_model(str(run_dir / "final_model"))
