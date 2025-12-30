#!/bin/bash

accelerate launch --num_processes 1 lerobot/scripts/train_humanoid_multigpu.py \
    --policy.type=xr1_stage1 \
    --policy.stage1_pretrained_path=../pretrained/XR-1-Stage1-UVMC \
    --policy.image_interval_step=50 \
    --policy.action_chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.codebook_k_size=256 \
    --policy.codebook_embed_dim=256 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=5_000 \
    --policy.scheduler_decay_steps=300_000 \
    --dataset.image_transforms.enable=true \
    --dataset.select_dataset=XR_1_DATASET_SAMPLE \
    --num_workers=8 \
    --batch_size=12 \
    --steps=300_000 \
    --save_freq=300_000 \
    --output_dir=../save_xr1/xr1_stage1/xr1_stage1_finetune \
    --wandb.enable=true \
    --wandb.project=xr1_stage1 \
    --job_name=xr1_stage1_finetune
   












