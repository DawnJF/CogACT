#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=f3ff179b6f827f5e96753a72451d069bd58bd413
# export WANDB_MODE=offline
# export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# A100优化环境变量
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# PyTorch优化
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=32
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 训练配置
WORLD_SIZE=6
MASTER_PORT=29500
BATCH_SIZE=4  # 每张卡的batch size
GLOBAL_BATCH_SIZE=24  # 6 * 4
ACCUMULATE_GRAD=1
LEARNING_RATE=1e-4
EPOCHS=10
MAX_STEPS=50000

# 数据配置
DATA_ROOT="datasets/open-x-embodiment"
RUN_ROOT="runs"
SAVE_INTERVAL=2500

# 模型配置
QWEN_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
ACTION_MODEL_TYPE="DiT-B"
COGNITION_TOKEN_SIZE=4096
CROSS_ATTN_HEADS=8
MAX_NEW_TOKENS=50

# VLA配置ID (根据你的配置调整)
VLA_TYPE="EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS"

# Wandb配置
WANDB_PROJECT="cogact-vla-training"
WANDB_ENTITY="your_wandb_entity"  # 替换为你的wandb实体名

# 运行标识
RUN_ID="cogact-vlm-6xa100-bs${BATCH_SIZE}-lr${LEARNING_RATE}-$(date +%Y%m%d_%H%M%S)"
RUN_NOTE="6xA100-optimized-fp32-training"

echo "======================================"
echo "CogACT VLM Training Script"
echo "======================================"
echo "World Size: ${WORLD_SIZE}"
echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "Per Device Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Steps: ${MAX_STEPS}"
echo "Run ID: ${RUN_ID}"
echo "======================================"

# 确保数据目录存在
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Error: Data directory ${DATA_ROOT} does not exist!"
    echo "Please download the Open-X-Embodiment dataset first."
    exit 1
fi

# 创建运行目录
mkdir -p "${RUN_ROOT}"

# 启动分布式训练
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${WORLD_SIZE} \
    --master_port=${MASTER_PORT} \
    train_vlm.py \
    --vla.type ${VLA_TYPE} \
    --data_root_dir ${DATA_ROOT} \
    --run_root_dir ${RUN_ROOT} \
    --run_id ${RUN_ID} \
    --run_id_note ${RUN_NOTE} \
    --save_interval ${SAVE_INTERVAL} \
    --seed 42 \
    --trackers jsonl wandb \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_entity ${WANDB_ENTITY} \
    --use_vlm_cogact true \
    --qwen_model_path ${QWEN_MODEL} \
    --cross_attn_heads ${CROSS_ATTN_HEADS} \
    --cognition_token_size ${COGNITION_TOKEN_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --repeated_diffusion_steps 8 \
    --load_all_data_for_training true \
    --future_action_window_size 15 \
    --past_action_window_size 0 \
    --action_model_type ${ACTION_MODEL_TYPE} \
    --use_ema false \
    --action_dim 7 \
    --model_sharding true \
    --qwen_device_map auto \
    --trainable_device 4 \
    --gradient_checkpointing true \
    --offload_optimizer false \
    --max_memory_per_gpu 70GB \
    --accumulate_grad_batches ${ACCUMULATE_GRAD} \
    --compile_model false \
    --image_aug false \
    --vla.expected_world_size ${WORLD_SIZE} \
    --vla.global_batch_size ${GLOBAL_BATCH_SIZE} \
    --vla.per_device_batch_size ${BATCH_SIZE} \
    --vla.learning_rate ${LEARNING_RATE} \
    --vla.weight_decay 0.01 \
    --vla.max_grad_norm 1.0 \
    --vla.lr_scheduler_type "linear-warmup+cosine-decay" \
    --vla.warmup_ratio 0.1 \
    --vla.epochs ${EPOCHS} \
    --vla.max_steps ${MAX_STEPS} \
    --vla.shuffle_buffer_size 100000

echo "Training completed!"
echo "Check results in: ${RUN_ROOT}/${RUN_ID}"
