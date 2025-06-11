"""

只考虑VLMCogACT模型的训练脚本
6-8张卡来训练VLMCogACT模型
合理分配72B qwen模型和action_model的参数

"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
import wandb

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from training import VLAMetrics
from training.strategies.ddp_light import DDPLightStrategy
from conf import VLAConfig, VLARegistry
from vla import load, load_vla, init_vla
from vla.cogvla import VLMCogACT

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    run_root_dir: Path = Path("runs")

    # Resume Run Parameters
    cogact_pretrained_checkpoint: Optional[Union[str, Path]] = None

    # Run Arguments
    run_id: Optional[str] = None
    run_id_note: Optional[str] = None
    save_interval: int = 2500
    image_aug: bool = False
    seed: int = 42

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    wandb_project: str = ""
    wandb_entity: str = ""
    
    # VLMCogACT相关配置
    use_vlm_cogact: bool = True  # 默认使用VLMCogACT
    qwen_model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    cross_attn_heads: int = 8
    cognition_token_size: int = 4096
    max_new_tokens: int = 50
    
    # 训练相关参数（简化）
    repeated_diffusion_steps: int = 8
    load_all_data_for_training: bool = True
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_model_type: str = 'DiT-B'
    use_ema: bool = False
    action_dim: int = 7
    
    # 新增：A100优化配置，移除混合精度相关
    model_sharding: bool = True  # 启用模型分片
    qwen_device_map: str = "auto"  # Qwen模型设备映射策略
    trainable_device: int = 0  # 可训练模块的主设备
    gradient_checkpointing: bool = True  # 启用梯度检查点以节省内存
    offload_optimizer: bool = False  # 是否卸载优化器状态到CPU
    max_memory_per_gpu: str = "70GB"  # A100 80GB，保留10GB
    
    # 训练优化配置
    accumulate_grad_batches: int = 1  # 梯度累积步数
    compile_model: bool = False  # 是否使用torch.compile (需要PyTorch 2.0+)

    def __post_init__(self) -> None:
        """A100优化的初始化配置"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

        # 计算有效批次大小，考虑梯度累积
        self.effective_batch_size = self.global_batch_size * self.accumulate_grad_batches
        
        # 根据GPU数量调整设备映射策略
        if self.model_sharding and overwatch.world_size() >= 6:
            # 6-8张A100：智能分配Qwen模型到前几张卡，可训练模块到后几张卡
            num_layers = 80  # Qwen2.5-72B层数
            qwen_gpus = overwatch.world_size() - 2  # 前N-2张卡给Qwen
            
            self.qwen_device_map = {}
            for i in range(num_layers):
                gpu_id = i % qwen_gpus
                self.qwen_device_map[f"model.layers.{i}"] = f"cuda:{gpu_id}"
            
            # 其他组件分配
            self.qwen_device_map.update({
                "model.embed_tokens": "cuda:0",
                "model.norm": f"cuda:{qwen_gpus-1}",
                "lm_head": f"cuda:{qwen_gpus-1}",
            })
            
            # 可训练模块使用最后两张卡
            self.trainable_device = overwatch.world_size() - 2
        else:
            # 少于6卡时使用自动设备映射
            self.qwen_device_map = "auto"
            self.trainable_device = 0

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("CogACT-VLA Training :: A100优化训练")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # A100专用配置
    if cfg.gradient_checkpointing:
        torch.backends.cudnn.benchmark = True  # A100性能优化
        torch.backends.cudnn.deterministic = False

    # 打印A100信息
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if "A100" in props.name:
            overwatch.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f}GB")
        else:
            overwatch.warning(f"GPU {i}: {props.name} - 非A100 GPU，性能可能不佳")

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"
    if cfg.model_sharding:
        cfg.run_id += "--sharded"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    dist.barrier()
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")

    # 从CogAct中加载ActionMode的参数
    if cfg.cogact_pretrained_checkpoint is not None:
        vla = init_vla(
            cfg.cogact_pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.past_action_window_size,
            use_ema=cfg.use_ema,
        )
    else:
        # 直接创建新模型
        from vla.cogvla import VLMCogACT

        vla = VLMCogACT(
            qwen_model_path=cfg.qwen_model_path,
            action_model_type=cfg.action_model_type,
            token_size=cfg.cognition_token_size,
            action_dim=cfg.action_dim,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.past_action_window_size,
            use_ema=cfg.use_ema,
            cross_attn_heads=cfg.cross_attn_heads,
            # A100优化参数
            qwen_device_map=cfg.qwen_device_map,
            trainable_device=cfg.trainable_device,
            max_memory_per_gpu=cfg.max_memory_per_gpu,
        )

    # 启用梯度检查点（如果配置）
    if cfg.gradient_checkpointing:
        if hasattr(vla, "enable_gradient_checkpointing"):
            vla.enable_gradient_checkpointing()

    # A100优化的编译
    if cfg.compile_model and hasattr(torch, "compile"):
        overwatch.info("A100优化编译可训练模块")
        vla.cross_attention = torch.compile(
            vla.cross_attention, 
            mode="max-autotune",  # A100最大优化
            fullgraph=True
        )
        vla.action_model = torch.compile(
            vla.action_model, 
            mode="max-autotune",
            fullgraph=True
        )

    # [Validate] Model should be in Full Precision for trainable parts!
    for name, param in vla.named_parameters():
        if param.requires_grad:
            assert param.dtype == torch.float32, f"Trainable parameter {name} not in full precision: {param.dtype}"

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    qwen_params = sum(p.numel() for p in vla.qwen_model.parameters())
    action_params = sum(p.numel() for p in vla.action_model.parameters())

    overwatch.info(
        f"模型参数统计 (百万参数):\n"
        f"  总参数: {num_params / 10**6:.1f}M\n"
        f"  可训练参数: {num_trainable_params / 10**6:.1f}M\n"
        f"  Qwen模型: {qwen_params / 10**6:.1f}M (冻结)\n"
        f"  Action模型: {action_params / 10**6:.1f}M\n"
        f"  可训练比例: {100 * num_trainable_params / num_params:.2f}%"
    )

    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, _, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    dist.barrier()
    # 创建训练策略
    if cfg.use_vlm_cogact:
        overwatch.info("A100优化的VLMCogACT ddp-light训练策略")
        train_strategy = DDPLightStrategy(
            vlm=vla,
            device_id=device_id,
            stage="",
            epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            global_batch_size=cfg.global_batch_size,
            per_device_batch_size=cfg.per_device_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            worker_init_fn=worker_init_fn,
            # A100优化参数
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            offload_optimizer=cfg.offload_optimizer,
        )

    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        metrics,
        save_interval=cfg.save_interval,
        action_model=True,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
