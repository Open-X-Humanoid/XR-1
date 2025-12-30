#!/bin/bash

accelerate launch --num_processes 1 lerobot/scripts/train_humanoid_multigpu.py \
    --policy.type=xr1_stage2 \
    --policy.stage1_supervised=true \
    --policy.stage1_pretrained_path=../pretrained/XR-1-Stage1-UVMC \
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
    --dataset.image_transforms.enable=true \
    --dataset.select_dataset=XR_1_DATASET_SAMPLE_No_Ego4d \
    --num_workers=8 \
    --batch_size=12 \
    --steps=50_000 \
    --save_freq=50_000 \
    --output_dir=../save_xr1/xr1_stage2/xr1_stage2_finetune \
    --wandb.enable=true \
    --wandb.project=xr1_stage2 \
    --job_name=xr1_stage2_finetune
   








