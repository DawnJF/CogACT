"""

使用Qwen2.5-VL-72B作为视觉语言模型的CogACT实现
考虑两个情况：
1. 多卡训练
2. 少量卡推理
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from action_model.action_model import ActionModel
from action_model.models import DiT

# 添加缺失的logging导入
import logging

overwatch = logging.getLogger(__name__)

# HuggingFace默认 / LLaMa-2 IGNORE_INDEX (用于标签)
IGNORE_INDEX = -100


class CrossAttention(nn.Module):
    """交叉注意力模块，用于从序列隐状态生成认知特征"""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 使用PyTorch内置的多头注意力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 可学习的查询向量
        self.learnable_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, seq_len, hidden_dim] 从Qwen获得的隐状态序列
            attention_mask: [B, seq_len] 注意力掩码
        Returns:
            cognition_features: [B, 1, hidden_dim] 认知特征
        """
        B, seq_len, _ = hidden_states.shape

        # 扩展可学习查询向量
        query = self.learnable_query.expand(B, -1, -1)  # [B, 1, hidden_dim]

        # key_padding_mask为True表示需要忽略的位置
        key_padding_mask = ~attention_mask.bool()
        # 使用MultiheadAttention进行交叉注意力计算
        # query: [B, 1, hidden_dim], key&value: [B, seq_len, hidden_dim]
        cognition_features, _ = self.multihead_attn(
            query=query, key=hidden_states, value=hidden_states, key_padding_mask=key_padding_mask
        )  # 输出: [B, 1, hidden_dim]

        return cognition_features


class VLMCogACT(nn.Module):
    def __init__(
        self,
        qwen_model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        action_model_type: str = "DiT-B",
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        cross_attn_heads: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()

        # 初始化Qwen2.5-VL-72B模型，使用优化的设备映射
        overwatch.info(f"加载Qwen模型: {qwen_model_path}")

        try:
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_model_path,
                torch_dtype="auto",
            )

            self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

            overwatch.info("Qwen模型加载成功")

        except Exception as e:
            overwatch.error(f"Qwen模型加载失败: {e}")
            raise

        # 获取Qwen的隐藏维度
        qwen_hidden_dim = self.qwen_model.config.hidden_size

        # 交叉注意力模块用于生成认知特征 - 放在指定的可训练设备上
        self.cross_attention = CrossAttention(qwen_hidden_dim, cross_attn_heads)

        # 如果token_size与qwen_hidden_dim不匹配，添加投影层
        if token_size != qwen_hidden_dim:
            self.cognition_projector = nn.Linear(qwen_hidden_dim, token_size)
        else:
            self.cognition_projector = nn.Identity()

        # 动作模型 - 也放在可训练设备上
        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        # self.use_ema = use_ema

        self.all_module_keys = ["action_model", "cross_attention", "cognition_projector"]

        # 可训练模块键
        self._trainable_module_keys = ["action_model", "cross_attention", "cognition_projector"]
        self.norm_stats = norm_stats

        # 设置默认图像分辨率 (Qwen2.5-VL通常使用448x448)
        self._default_image_resolution = (3, 448, 448)  # TODO

        # 记录内存使用情况
        self._log_model_memory_usage()

    def freeze_qwen(self):
        # 完全冻结Qwen参数
        self.qwen_model.requires_grad_(False)
        self.qwen_model.eval()

    def freeze_action_model(self):
        """冻结动作模型的参数"""
        self.action_model.requires_grad_(False)
        self.action_model.eval()

    def _log_model_memory_usage(self):
        """记录模型内存使用情况"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                if allocated > 0 or reserved > 0:
                    overwatch.info(f"GPU {i} 内存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    @property
    def trainable_module_keys(self) -> List[str]:
        """返回可训练模块的键列表"""
        return self._trainable_module_keys

    @property
    def device(self):
        """返回可训练模块的设备"""
        return next(self.cross_attention.parameters()).device

    def extract_qwen_hidden_states(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask,
        image_grid_thw,
        max_new_tokens: int = 100,
    ):
        """
        从Qwen模型中提取自回归生成的隐状态

        """
        with torch.no_grad():

            # (['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
            # outputs, attention_mask = loop_forward(
            #     self.qwen_model, self.tokenizer, input_ids, attention_mask, pixel_values, image_grid_thw
            # )
            outputs = self.qwen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            # 方法一：所有Token的最后一层hidden_states（包括输入和生成）(B, total_seq_len, hidden_dim)
            all_last_layer = torch.cat([hs[-1] for hs in outputs.hidden_states], dim=1)
            all_mask = (outputs.sequences != self.tokenizer.pad_token_id).long()  # (B, total_seq_len)

            # 方法二：仅新生成Token的最后一层hidden_states (B, T, hidden_dim)
            new_tokens_last_layer = torch.stack([hs[-1][:, -1, :] for hs in outputs.hidden_states], dim=1)
            new_sequences = outputs.sequences[:, input_ids.shape[1] :]  # (B, T)
            new_mask = (new_sequences != self.tokenizer.pad_token_id).long()  # (B, T)

            return new_tokens_last_layer, new_mask, outputs.sequences

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks=None,
        max_new_tokens: int = 500,
    ) -> Tuple:
        """通过VLM运行前向传播，返回CausalLMOutputWithPast实例（包含损失）"""

        # 从Qwen提取隐状态
        hidden_states, attention_mask, _sequences = self.extract_qwen_hidden_states(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
        )

        # hidden_states: [B, total_seq_len, hidden_dim]
        # 使用交叉注意力生成认知特征
        cognition_features = self.cross_attention(hidden_states, attention_mask)

        # 投影到所需维度
        cognition_features = self.cognition_projector(cognition_features)  # [B, 1, token_size]

        # 处理动作数据
        actions_history = actions[:, 0 : self.past_action_window_size, :]
        actions_future = actions[:, -(self.future_action_window_size + 1) :, :]

        # 重复动作数据用于扩散步骤
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1)

        # 动作模型前向传播并计算损失
        loss = self.action_model.loss(actions_repeated, cognition_features_repeated)

        # 创建兼容的输出对象
        output = CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
        )

        return loss, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回FSDP包装策略，仅包装可训练模块"""
        # 为可训练模块创建FSDP包装策略
        # 由于Qwen模型已被冻结，我们只需要包装可训练的组件
        cogact_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={
                # 动作模型相关
                DiT,
                # 注意力和投影模块
                nn.MultiheadAttention,
                nn.Linear,
                # 自定义模块
                CrossAttention,
                ActionModel,
            },
        )

        # 返回单一策略，因为我们只有可训练的模块需要包装
        # Qwen模型被冻结，不需要FSDP包装
        return cogact_fsdp_wrapping_policy

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        qwen_model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = "DiT-B",
        use_ema: bool = False,
        norm_stats=None,
        # 新增设备管理参数
        qwen_device_map: Union[str, Dict] = "auto",
        max_memory_per_gpu: str = "22GB",
        **kwargs,
    ) -> VLMCogACT:
        """从预训练检查点加载CogACT模型"""

        # 初始化CogACT模型，传递设备管理参数
        cogact = VLMCogACT(
            qwen_model_path=qwen_model_path,
            token_size=4096,
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            action_model_type=action_model_type,
            use_ema=use_ema,
            norm_stats=norm_stats,
            qwen_device_map=qwen_device_map,
            max_memory_per_gpu=max_memory_per_gpu,
            **kwargs,
        )

        # 从检查点加载权重
        if pretrained_checkpoint and pretrained_checkpoint.exists():
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

            # 加载动作模型权重
            if "action_model" in model_state_dict:
                cogact.action_model.load_state_dict(model_state_dict["action_model"])  # #
                overwatch.info("加载动作模型权重成功")

                if "ema_diffusion" in model_state_dict and use_ema:
                    cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
                elif use_ema:
                    cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])

            # 加载交叉注意力权重
            if "cross_attention" in model_state_dict:
                cogact.cross_attention.load_state_dict(model_state_dict["cross_attention"])
                overwatch.info("加载交叉注意力权重成功")

            # 加载认知投影器权重
            if "cognition_projector" in model_state_dict:
                cogact.cognition_projector.load_state_dict(model_state_dict["cognition_projector"])
                overwatch.info("加载认知投影器权重成功")
        else:
            overwatch.warning("预训练检查点不存在，初始化新模型")

        return cogact

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        max_new_tokens: int = 50,
        **kwargs: str,
    ) -> np.ndarray:
        """
        VLA推理的核心函数；将输入图像和任务指令映射到连续动作
        """

        # 构建Qwen提示
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"你应该执行什么动作来{instruction}？请详细描述机器人应该如何完成这个任务。",
                    },
                ],
            }
        ]

        # 处理输入
        text_input = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_processor(images=[image], text=[text_input], return_tensors="pt")

        # 确保输入在正确的设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 从Qwen提取隐状态
        hidden_states, generation_attention_mask = self.extract_qwen_hidden_states(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
        )

        # 生成认知特征
        cognition_features = self.cross_attention(hidden_states, generation_attention_mask)  # [1, 1, hidden_dim]
        cognition_features = self.cognition_projector(cognition_features)  # [1, 1, token_size]

        # 准备动作预测
        using_cfg = cfg_scale > 1.0
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = 1

        cognition_features = cognition_features.to(model_dtype)

        # 采样随机噪声
        noise = torch.randn(
            B, self.future_action_window_size + 1, self.action_model.in_channels, device=cognition_features.device
        ).to(model_dtype)

        # 设置分类器引导
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0).expand(B, 1, -1)
            z = torch.cat([cognition_features, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM采样
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
                eta=0.0,
            )
        else:
            # DDPM采样
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        normalized_actions = samples[0].cpu().numpy()

        # 反归一化动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @torch.inference_mode()
    def predict_action_batch(
        self,
        image: List[Image],
        instruction: List[str],
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        max_new_tokens: int = 50,
        **kwargs: str,
    ) -> np.ndarray:
        """
        批量VLA推理的核心函数；将输入图像和任务指令映射到连续动作
        该函数用于仿真器中的批量推理

        @param image: PIL图像列表，每个图像为 [height, width, 3]
        @param instruction: 任务指令字符串列表
        @param unnorm_key: 可选的数据集名称，用于检索反归一化统计信息
        @param cfg_scale: 分类器自由引导(CFG)的缩放因子；如果 == 1.0，则禁用CFG
        @param use_ddim: 使用DDIM采样而不是DDPM采样
        @param num_ddim_steps: DDIM采样使用的步数
        @param max_new_tokens: 每个样本生成的最大token数

        @return 反归一化的（连续）动作向量 --> 末端执行器增量
        """

        B = len(image)
        input_ids_list = []
        pixel_values_list = []

        # 为每个样本构建Qwen提示
        for i in range(B):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image[i]},
                        {
                            "type": "text",
                            "text": f"你应该执行什么动作来{instruction[i]}？请详细描述机器人应该如何完成这个任务。",
                        },
                    ],
                }
            ]

            # 处理单个输入
            text_input = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processed_input = self.qwen_processor(images=[image[i]], text=[text_input], return_tensors="pt")

            input_ids_list.append(processed_input["input_ids"].squeeze(0))
            pixel_values_list.append(processed_input["pixel_values"].squeeze(0))

        # 对input_ids进行填充处理
        max_length = max(ids.shape[0] for ids in input_ids_list)
        pad_token_id = self.qwen_processor.tokenizer.pad_token_id

        padded_input_ids = []
        attention_masks = []

        for ids in input_ids_list:
            # 右侧填充
            padding_length = max_length - ids.shape[0]
            if padding_length > 0:
                padded_ids = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=ids.dtype)])
                attention_mask = torch.cat(
                    [torch.ones(ids.shape[0], dtype=torch.bool), torch.zeros(padding_length, dtype=torch.bool)]
                )
            else:
                padded_ids = ids
                attention_mask = torch.ones(ids.shape[0], dtype=torch.bool)

            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

        # 堆叠为批次张量
        input_ids = torch.stack(padded_input_ids).to(self.device)
        attention_mask = torch.stack(attention_masks).to(self.device)
        pixel_values = torch.stack(pixel_values_list).to(self.device)

        # 从Qwen提取隐状态
        hidden_states, generation_attention_mask = self.extract_qwen_hidden_states(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens
        )

        # 使用交叉注意力生成认知特征
        cognition_features = self.cross_attention(hidden_states, generation_attention_mask)  # [B, 1, hidden_dim]
        cognition_features = self.cognition_projector(cognition_features)  # [B, 1, token_size]

        # 验证认知特征的形状
        assert cognition_features.shape[0] == B, f"批次大小必须为{B}用于动作预测"

        using_cfg = cfg_scale > 1.0
        model_dtype = next(self.action_model.net.parameters()).dtype
        cognition_features = cognition_features.to(model_dtype)

        # 采样随机噪声
        noise = torch.randn(
            B, self.future_action_window_size + 1, self.action_model.in_channels, device=cognition_features.device
        ).to(model_dtype)

        # 设置分类器自由引导
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0).expand(B, 1, -1)
            z = torch.cat([cognition_features, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM采样
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
                eta=0.0,
            )
        else:
            # DDPM采样
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # 移除空类别样本
        normalized_actions = samples.cpu().numpy()

        # 反归一化动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def get_image_transform(self):
        """获取图像预处理变换，使用Qwen自己的处理方式"""

        # return QwenImageTransform(self.qwen_processor, self.default_image_resolution)
        return QwenInputTransform(self.qwen_processor)

    def get_tokenizer(self):
        """返回Qwen模型的tokenizer"""
        return self.tokenizer

    @property
    def prompt_builder_fn(self):
        """返回构建提示的函数"""

        def build_prompt(conversations):
            """为VLM构建提示的函数

            Args:
                conversations: 对话数据，可以是字符串或对话列表

            Returns:
                构建好的消息列表或处理后的文本
            """
            if isinstance(conversations, str):
                # 简单字符串格式
                return conversations
            elif isinstance(conversations, list):
                # 处理对话列表格式
                messages = []
                for conv in conversations:
                    if isinstance(conv, dict):
                        messages.append(conv)
                    else:
                        messages.append({"role": "user", "content": str(conv)})
                return messages
            else:
                # 其他格式转为字符串
                return str(conversations)

        return build_prompt

    @property
    def default_image_resolution(self):
        """返回默认图像分辨率"""
        return self._default_image_resolution


class QwenInputTransform:
    """Qwen输入转换器，用于将图像和文本转换为模型输入格式"""

    def __init__(self, qwen_processor: AutoProcessor):
        self.processor = qwen_processor

    def __call__(self, image, lang) -> Dict[str, torch.Tensor]:
        """
        self.image_transform(img, lang=lang)
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"你要控制机械臂末端运动并完成任务，具体的任务是：'{lang}'，请详细说明接下来2s的运动细节，包括方向和轨迹，回答要求直接简洁明了。"
                        ),
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [Image.fromarray(image)]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return inputs


def loop_forward(model, tokenizer, input_ids, attention_mask, pixel_values, image_grid_thw, max_new_tokens=150):
    """
    一个稳健的、自定义的自回归生成循环。

    该实现遵循以下原则：
    1. 始终处理完整批次，以避免维度不匹配的错误。
    2. 使用 `past_key_values` (KV Cache) 来加速生成。
    3. 当一个序列生成 EOS 标记后，会继续为其填充 PAD 标记，直到所有序列完成或达到最大长度。
    """
    # 检查并获取 pad_token_id，如果不存在，则使用 eos_token_id
    # 这是必要的，因为我们需要一个 token 来填充已完成的序列
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 获取 input_ids 和 attention_mask

    # 提取额外参数 (例如 vision-language 模型中的 pixel_values)
    extra_inputs = {"image_grid_thw": image_grid_thw, "pixel_values": pixel_values}

    # 克隆 input_ids，用于存储完整的生成序列
    generated_ids = input_ids.clone()

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)  # 处理padding位置

    # 初始化 past_key_values 和一个标志，用于表示每个样本是否已完成
    past_key_values = None
    done = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

    # 初始化收集所有隐状态的列表
    all_hidden_states = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            # --- 模型输入准备 ---
            if past_key_values is not None:
                # 如果有 KV Cache，下一次的输入只需要最后一个 token
                model_inputs = {"input_ids": generated_ids[:, -1].unsqueeze(-1)}
                current_position_ids = position_ids[:, -1].unsqueeze(-1)
                outputs = model(
                    **model_inputs,
                    attention_mask=attention_mask,
                    position_ids=current_position_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                # 第一次迭代，输入是完整的 prompt
                model_inputs = {"input_ids": generated_ids}
                outputs = model(
                    **model_inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                    **extra_inputs,
                )

            # 收集当前步骤的隐状态 (最后一层)
            current_hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
            all_hidden_states.append(current_hidden_states)

            # --- Token 解码 ---
            # 获取下一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]
            # 使用贪心解码获取下一个 token 的 id
            next_token_ids = torch.argmax(next_token_logits, dim=-1)

            # --- 状态更新 ---
            # 1. 确定哪些序列在当前步骤刚刚完成
            #    只有那些之前未完成 `(~done)` 且新 token 是 EOS 的序列，才是“刚刚完成”
            newly_done = (next_token_ids == tokenizer.eos_token_id) & (~done)

            # 2. 更新 `done` 标志
            #    将刚刚完成的序列加入到 `done` 集合中
            done |= newly_done

            # 3. 决定要添加到序列末尾的 token
            #    - 对于已经完成的序列 (`done` 为 True)，添加 pad_token
            #    - 对于尚未完成的序列 (`done` 为 False)，添加新生成的 token
            tokens_to_add = torch.where(done, tokenizer.pad_token_id, next_token_ids)

            # 4. 将新 token 拼接到 `generated_ids`
            generated_ids = torch.cat([generated_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            # 更新位置ID (添加新位置)
            position_ids = torch.cat(
                [position_ids, position_ids[:, -1:] + 1],
                dim=-1,  # 新位置 = 最后位置 + 1
            )

            # 5. 更新 attention_mask
            #    - 对于未完成的序列，其 attention_mask 长度应加 1 (补 True/1)
            #    - 对于已完成的序列，其 attention_mask 长度也应加 1，但用 0 填充
            #      这样可以确保张量形状一致，同时让模型忽略填充部分
            attention_mask = torch.cat([attention_mask, (~done).long().unsqueeze(-1)], dim=-1)

            # 6. 更新 past_key_values 供下一次迭代使用
            past_key_values = outputs.past_key_values

            # --- 提前终止 ---
            # 如果批次中的所有序列都已完成，则提前结束循环
            if done.all():
                break

    return all_hidden_states, attention_mask
