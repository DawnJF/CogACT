"""

使用Qwen2.5-VL-72B作为视觉语言模型的CogACT实现
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
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

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

        # 处理注意力掩码 - MultiheadAttention期望的掩码格式
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask为True表示有效位置，key_padding_mask为True表示需要忽略的位置
            key_padding_mask = ~attention_mask  # [B, seq_len]

        # 使用MultiheadAttention进行交叉注意力计算
        # query: [B, 1, hidden_dim], key&value: [B, seq_len, hidden_dim]
        cognition_features, _ = self.multihead_attn(
            query=query, key=hidden_states, value=hidden_states, key_padding_mask=key_padding_mask
        )  # 输出: [B, 1, hidden_dim]

        return cognition_features


class CogACT(nn.Module):
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

        # 初始化Qwen2.5-VL-72B模型
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)

        # 完全冻结Qwen参数
        self.qwen_model.requires_grad_(False)
        self.qwen_model.eval()

        # 获取Qwen的隐藏维度
        qwen_hidden_dim = self.qwen_model.config.hidden_size

        # 交叉注意力模块用于生成认知特征
        self.cross_attention = CrossAttention(qwen_hidden_dim, cross_attn_heads)

        # 如果token_size与qwen_hidden_dim不匹配，添加投影层
        if token_size != qwen_hidden_dim:
            self.cognition_projector = nn.Linear(qwen_hidden_dim, token_size)
        else:
            self.cognition_projector = nn.Identity()

        # 动作模型
        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema

        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ["action_model", "ema_diffusion", "cross_attention", "cognition_projector"]
        else:
            self.all_module_keys = ["action_model", "cross_attention", "cognition_projector"]

        # 可训练模块键
        self._trainable_module_keys = ["action_model", "cross_attention", "cognition_projector"]
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> List[str]:
        """返回可训练模块的键列表"""
        return self._trainable_module_keys

    @property
    def device(self):
        """返回模型设备"""
        # 修复device属性 - 使用cross_attention的device作为参考
        return next(self.cross_attention.parameters()).device

    def extract_qwen_hidden_states(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从Qwen模型中提取自回归生成的隐状态

        Args:
            pixel_values: 图像张量
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数

        Returns:
            hidden_states: [B, total_seq_len, hidden_dim] 所有生成token的最后一层隐状态
            generation_attention_mask: [B, total_seq_len] 生成序列的注意力掩码
        """
        with torch.no_grad():
            # 确保输入在正确的设备上
            input_ids = input_ids.to(self.qwen_model.device)
            pixel_values = pixel_values.to(self.qwen_model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.qwen_model.device)

            # 使用Qwen生成，获取隐状态
            outputs = self.qwen_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,  # 确定性生成
                pad_token_id=self.qwen_processor.tokenizer.eos_token_id,
            )

            # 提取所有生成步骤的最后一层隐状态
            batch_size = input_ids.shape[0]
            hidden_dim = self.qwen_model.config.hidden_size

            # 收集所有时间步的最后一层隐状态
            all_hidden_states = []

            # 添加初始序列的隐状态（从第一个生成步骤获取）
            if len(outputs.hidden_states) > 0:
                # 第一个生成步骤包含完整的输入序列隐状态
                first_step_hidden = outputs.hidden_states[0][-1]  # 最后一层
                all_hidden_states.append(first_step_hidden)

                # 添加后续生成的token隐状态
                for step_idx in range(1, len(outputs.hidden_states)):
                    step_hidden = outputs.hidden_states[step_idx][-1]  # 最后一层
                    # 只取新生成的token（最后一个）
                    new_token_hidden = step_hidden[:, -1:, :]  # [B, 1, hidden_dim]
                    all_hidden_states.append(new_token_hidden)

            # 拼接所有隐状态
            if all_hidden_states:
                hidden_states = torch.cat(all_hidden_states, dim=1)  # [B, total_seq_len, hidden_dim]
            else:
                # 后备方案：如果没有隐状态，创建零张量
                total_seq_len = input_ids.shape[1] + max_new_tokens
                hidden_states = torch.zeros(
                    batch_size, total_seq_len, hidden_dim, device=self.qwen_model.device, dtype=torch.bfloat16
                )

            # 创建对应的注意力掩码
            total_seq_len = hidden_states.shape[1]
            generation_attention_mask = torch.ones(
                batch_size, total_seq_len, device=self.qwen_model.device, dtype=torch.bool
            )

            return hidden_states, generation_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks=None,
        max_new_tokens: int = 50,
    ) -> Tuple:
        """通过VLM运行前向传播，返回CausalLMOutputWithPast实例（包含损失）"""

        # 从Qwen提取隐状态
        hidden_states, generation_attention_mask = self.extract_qwen_hidden_states(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens
        )

        # 使用交叉注意力生成认知特征
        cognition_features = self.cross_attention(hidden_states, generation_attention_mask)  # [B, 1, hidden_dim]

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
        **kwargs,
    ) -> CogACT:
        """从预训练检查点加载CogACT模型"""

        # 初始化CogACT模型
        cogact = CogACT(
            qwen_model_path=qwen_model_path,
            token_size=4096,  # 可以根据需要调整
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            action_model_type=action_model_type,
            use_ema=use_ema,
            norm_stats=norm_stats,
            **kwargs,  # 传递额外参数
        )

        # 从检查点加载权重
        if pretrained_checkpoint and pretrained_checkpoint.exists():
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

            # 加载动作模型权重
            if "action_model" in model_state_dict:
                cogact.action_model.load_state_dict(model_state_dict["action_model"])
                if "ema_diffusion" in model_state_dict and use_ema:
                    cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
                elif use_ema:
                    cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])

            # 加载交叉注意力权重
            if "cross_attention" in model_state_dict:
                cogact.cross_attention.load_state_dict(model_state_dict["cross_attention"])

            # 加载认知投影器权重
            if "cognition_projector" in model_state_dict:
                cogact.cognition_projector.load_state_dict(model_state_dict["cognition_projector"])
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
