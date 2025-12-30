#!/bin/bash

# 默认配置
MODE="real"
DATASET="XR_1_DATASET_SAMPLE_No_Ego4d" # 默认数据集

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            MODE="debug"
            shift
            ;;
        --real)
            MODE="real"
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$MODE" == "debug" ]; then
    echo "Starting in DEBUG mode with dataset: $DATASET..."
    accelerate launch --num_processes 1 lerobot/scripts/train_humanoid_multigpu.py \
        --policy.type=xr1_stage2 \
        --policy.stage1_supervised=false \
        --policy.stage2_pretrained_path=../pretrained/XR-1-Stage2 \
        --policy.action_chunk_size=50 \
        --policy.n_action_steps=50 \
        --policy.freeze_vision_encoder=true \
        --policy.freeze_language_encoder=true \
        --policy.train_expert_only=false \
        --policy.optimizer_lr=1e-4 \
        --policy.dataset_stats_generate=true \
        --dataset.select_dataset=$DATASET \
        --batch_size=16 \
        --save_freq=20 \
        --output_dir=./debug_output/ 
else
    echo "Starting in REAL TRAINING mode with dataset: $DATASET..."
    accelerate launch --num_processes 1 lerobot/scripts/train_humanoid_multigpu.py \
        --policy.type=xr1_stage2 \
        --policy.stage1_supervised=false \
        --policy.stage2_pretrained_path=../pretrained/XR-1-Stage2 \
        --policy.action_chunk_size=50 \
        --policy.n_action_steps=50 \
        --policy.freeze_vision_encoder=true \
        --policy.freeze_language_encoder=true \
        --policy.train_expert_only=false \
        --policy.optimizer_lr=5e-5 \
        --policy.scheduler_warmup_steps=1_000 \
        --policy.scheduler_decay_steps=50_000 \
        --policy.dataset_stats_generate=true \
        --dataset.image_transforms.enable=false \
        --dataset.select_dataset=$DATASET \
        --num_workers=12 \
        --batch_size=20 \
        --steps=50_000 \
        --save_freq=50_000 \
        --output_dir=../save_xr1/xr1_stage3/xr1_stage3_finetune \
        --wandb.enable=true \
        --wandb.project=xr1_stage3 \
        --job_name=xr1_stage3_finetune
fi