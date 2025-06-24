#!/bin/bash

cd /liujinxin/code/CogACT_mjf

# 设置环境变量
export WANDB_API_KEY=f3ff179b6f827f5e96753a72451d069bd58bd413
# export WANDB_MODE=offline

# export CUDA_VISIBLE_DEVICES=4,5,6,7
WORLD_SIZE=8

PROJECT_FOLDER="CogVLA-libero-32b-v9"

BATCH_SIZE=6
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * WORLD_SIZE))

COGACT_CHECKPOINT="/liujinxin/code/CogACT_mjf/logs/libero--image_aug/checkpoints/step-070000-epoch-190-loss=0.0065.pt"
QWEN_MODEL="/liujinxin/model/Qwen/Qwen2.5-VL-32B-Instruct"

# 启动分布式训练
/ssdwork/liujinxin/miniconda3/envs/cogact_mjf/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${WORLD_SIZE} \
    scripts/train_vlm.py \
    --cogact_pretrained_checkpoint ${COGACT_CHECKPOINT} \
    --qwen_model_path ${QWEN_MODEL} \
    --deepspeed scripts/zero3.json \
    --vla.type prism-dinosiglip-224px+oxe+diffusion \
    --vla.data_mix libero_10_no_noops \
    --data_root_dir /liujinxin/code/rlds_dataset_builder/tensorflow_datasets/datasets--openvla--modified_libero_rlds \
    --vla.per_device_batch_size ${BATCH_SIZE} \
    --vla.learning_rate 4e-4 \
    --run_root_dir ./logs \
    --run_id ${PROJECT_FOLDER} \
    --image_aug False \
    --wandb_project ${PROJECT_FOLDER} \
    --wandb_entity mjf \
    --prompt_v v9

echo "Training completed!"
