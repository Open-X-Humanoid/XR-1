#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset_humanoid,make_dataset_humanoid_heterogeneous
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy,make_policy_multidata,make_policy_heterogeneous
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

# multi-gpu add
import accelerate
from accelerate import DistributedDataParallelKwargs as DDPK

def update_policy(
    step: int,
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler | None = None,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    acc=None
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    
    # accelerate autocast
    with acc.autocast():
        loss, output_dict = policy.forward(batch)

    # Add NaN check
    if torch.isnan(loss):
        acc.print(colored(f"Warning: NaN loss detected at step {step}. Skipping update.", "red"))
        optimizer.zero_grad()
        # You might want to return early or continue
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)    

    # Accelerate backward
    acc.backward(loss)

    grad_norm = None

    # 只有在梯度同步（积累够了）时才进行更新
    if acc.sync_gradients:
        grad_norm = acc.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
        )
        
        with lock if lock is not None else nullcontext():
            optimizer.step()
        
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if has_method(policy, "update"):
            policy.update()

    train_metrics.loss = loss.item()
    if grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
        
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    print("train last update 20251211 xr1")
    kwargs = DDPK(find_unused_parameters=True)
    acc = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[kwargs]) # "bf16" "no"
    # acc = accelerate.Accelerator(kwargs_handlers=[kwargs])
    print(f"mixed_precision: {acc.mixed_precision}")
    cfg.validate(acc.is_main_process)
    # [acc.print(pformat(cfg.to_dict()))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print("cfg.wandb.enable:", cfg.wandb.enable)
    print("cfg.wandb.project:", cfg.wandb.project)
    print("acc.is_main_process:", acc.is_main_process)
    print("rank:",rank)
    print("world_size:",world_size)

    if cfg.wandb.enable and cfg.wandb.project and acc.is_main_process and rank==0:
        wandb_logger = WandBLogger(cfg)
        logging.info(colored("wandb start with main process", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
        
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available

    device = acc.device
    acc.print(f"Using Accelerate device: {device}")

    # device = get_safe_torch_device(cfg.device, log=True) # no need in accelerate
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    acc.print("Creating dataset")
    if cfg.policy.heterogeneous:
        dataset = make_dataset_humanoid_heterogeneous(cfg)
    else:
        dataset = make_dataset_humanoid(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        acc.print("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    acc.print("Creating policy")
    if hasattr(dataset, 'meta'):  
        policy = make_policy(
            cfg=cfg.policy,
            device=device,
            ds_meta=dataset.meta,
        )
    else:
        if cfg.policy.heterogeneous:
            policy = make_policy_heterogeneous(
                cfg=cfg.policy,
                features=dataset.template_features,
                device=device,
            )
        else:
            policy = make_policy_multidata(
                cfg=cfg.policy,
                device=device,
                features=dataset.merged_features,
                stats=dataset.stats
            )
        

    acc.print("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler = GradScaler(device, enabled=cfg.use_amp) # no need in accelerate

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    acc.print(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        acc.print(f"{cfg.env.task=}")
    acc.print(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    acc.print(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    acc.print(f"{dataset.num_episodes=}")
    acc.print(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    acc.print(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # print(torch.cuda.memory_allocated() / 1024 / 1024, "MB before prepare")
    policy,optimizer,dataloader = acc.prepare(policy, optimizer, dataloader)
    # print(torch.cuda.memory_allocated() / 1024 / 1024, "MB after prepare")
    dl_iter = cycle(dataloader)
    
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    acc.print("Start offline training on a fixed dataset")
    while step < cfg.steps:
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # for key in batch:
        #     if isinstance(batch[key], torch.Tensor):
        #         batch[key] = batch[key].to(device, non_blocking=True)
        with acc.accumulate(policy):
            train_tracker, output_dict = update_policy(
                step,
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                # grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
                acc=acc,
            )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        if acc.sync_gradients:
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                acc.print(train_tracker)
                if wandb_logger:
                    wandb_log_dict = {**train_tracker.to_dict(), **output_dict}
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step and acc.is_main_process and rank==0:
                acc.print(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_policy = policy.module if hasattr(policy, "module") else policy
                save_checkpoint(checkpoint_dir, step, cfg, save_policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                acc.print(f"Eval policy at step {step}")
                with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                acc.print(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    acc.print("End of training")


if __name__ == "__main__":
    init_logging()
    train()
