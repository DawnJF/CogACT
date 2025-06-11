#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=f3ff179b6f827f5e96753a72451d069bd58bd413
# export WANDB_MODE=offline
# export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# 执行 torchrun 命令
/ssdwork/liujinxin/miniconda3/envs/cogact_mjf/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 6 \
    "${PWD}/scripts/train.py" \
    --pretrained_checkpoint "/liujinxin/code/CogACT/models/CogACT-Base/checkpoints/CogACT-Base.pt" \
    --vla.type "prism-dinosiglip-224px+oxe+diffusion" \
    --vla.data_mix "libero_spatial_no_noops" \
    --vla.expected_world_size 6 \
    --vla.global_batch_size 144 \
    --vla.per_device_batch_size 24 \
    --vla.learning_rate 2e-5 \
    --data_root_dir "/liujinxin/code/rlds_dataset_builder/tensorflow_datasets/datasets--openvla--modified_libero_rlds" \
    --run_root_dir "./logs" \
    --run_id "libero" \
    --image_aug "True" \
    --wandb_project "CogACT-libero" \
    --wandb_entity "yijiulanpishu" \
    --save_interval 5000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 15 \
    --action_model_type "DiT-B" \
    --is_resume "False"
