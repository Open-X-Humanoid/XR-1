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
import logging,os,math,json
import numpy as np
from pprint import pformat
from pathlib import Path
import torch,pickle

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.Humanoid_lerobot_dataset import (
    HumanoidLeRobotDataset,
    HumanoidMultiLeRobotDataset,
    HumanoidMultiRobotLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

from data.train_robot_script import section_dataset

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

def list_to_ndarray(obj):
    if isinstance(obj, list):
        # 判断是不是纯数字的 list（可转为 ndarray）
        if obj and all(isinstance(x, (int, float, bool)) for x in obj):
            return np.array(obj)
        # 判断是不是嵌套 list（如矩阵/多维数组）
        if obj and all(isinstance(x, list) and all(isinstance(y, (int, float, bool)) for y in x) for x in obj):
            return np.array(obj)
        # 递归处理更深层次
        return [list_to_ndarray(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: list_to_ndarray(v) for k, v in obj.items()}
    else:
        return obj

def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key.startswith("action") and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, cfg.dataset.root,local_files_only=cfg.dataset.local_files_only,revision=cfg.dataset.revision)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
	    revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            local_files_only=cfg.dataset.local_files_only,
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index , indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

def make_dataset_humanoid(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, f"{cfg.dataset.root}/{cfg.dataset.repo_id}",local_files_only=cfg.dataset.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = HumanoidLeRobotDataset(
            cfg.dataset.repo_id,
            f"{cfg.dataset.root}/{cfg.dataset.repo_id}",
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            local_files_only=cfg.dataset.local_files_only,
        )
        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    else:
        ALL_TASK_CONFIG = section_dataset(cfg.dataset.select_dataset,cfg.dataset.repo_ids)
        cfg.dataset.root = ALL_TASK_CONFIG['multi_task'][0]['root']

        ds_meta = [  
            LeRobotDatasetMetadata(repo_id, f"{cfg.dataset.root}/{repo_id}")  
            for repo_id in cfg.dataset.repo_ids  
        ]  

        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta[0]) # 暂时默认fps不同数据集是一样的
        if cfg.policy.split_dataset:
                rank = int(os.environ.get('RANK', 0))
                world_size = int(os.environ.get('WORLD_SIZE', 1))
                total_episodes={}
                rank_total_episodes_num=0
                for repo_id in  cfg.dataset.repo_ids:
                    ds_meta = LeRobotDatasetMetadata(repo_id, f"{cfg.dataset.root}/{repo_id}") 
                    if ds_meta.total_episodes == 0:
                        raise ValueError(f"Dataset {repo_id} has no episodes")
                    all_indices = list(range(ds_meta.total_episodes))
                    num_per_rank = math.ceil(ds_meta.total_episodes / world_size)
                    expanded = np.resize(all_indices, num_per_rank * world_size)
                    indices_for_this_rank = expanded[rank::world_size].tolist()
                    rank_total_episodes_num += len(indices_for_this_rank)
                    total_episodes[repo_id] = indices_for_this_rank   
                print(f"Rank {rank} Total episode in world_size {world_size}: {rank_total_episodes_num}")
                dataset = HumanoidMultiLeRobotDataset(
                    cfg.dataset.repo_ids,
                    root=cfg.dataset.root,
                    episodes=total_episodes,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms
                )
                try:
                    with open(f"{cfg.dataset.root}/dataset_stats.json", "r") as f:
                        dataset_stats_dict = json.load(f)
                    dataset_stats = list_to_ndarray(dataset_stats_dict['dataset_stats'])  
                    dataset.stats = dataset_stats
                    print(f"Read Prepared dataset_stats.json from {cfg.dataset.root}/dataset_stats.json")
                except:
                    raise ValueError(f"No dataset_stats.json in {cfg.dataset.root}, Please generate first")
        else:             
            if cfg.policy.small_sample:
                # 列表20 小样本      
                total_episodes = {}
                for repo_id in cfg.dataset.repo_ids:
                    total_episodes[repo_id] = list(range(cfg.policy.small_sample_rate))
                dataset = HumanoidMultiLeRobotDataset(
                    cfg.dataset.repo_ids,
                    root=cfg.dataset.root,
                    delta_timestamps=delta_timestamps,
                    episodes=total_episodes,
                    image_transforms=image_transforms
                )
            else:
                dataset = HumanoidMultiLeRobotDataset(
                    cfg.dataset.repo_ids,
                    root=cfg.dataset.root,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms
                )
    return dataset


def make_dataset_humanoid_heterogeneous(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    ALL_TASK_CONFIG = section_dataset(cfg.dataset.select_dataset,cfg.dataset.repo_ids)
    if cfg.policy.dataset_stats_generate:
        save_policy_path = f"{cfg.output_dir}/dataset_stats"
    else:
        save_policy_path = None
    
    multi_dataset = HumanoidMultiRobotLeRobotDataset(ALL_TASK_CONFIG, cfg.policy,pretrained_policy_path=save_policy_path, image_transforms=image_transforms)
    multi_dataset.num_frames = 0
    multi_dataset.num_episodes = 0
    for dataset in multi_dataset.multirobot_datasets:
        multi_dataset.num_frames += dataset['dataset'].num_frames
        multi_dataset.num_episodes += dataset['dataset'].num_episodes


    return multi_dataset