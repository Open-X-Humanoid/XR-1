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
import logging

from pathlib import Path
from typing import Callable

import datasets
import numpy as np

import torch
import torch.utils
from datasets import load_dataset
import torch.nn.functional as F 

from torchvision.transforms import v2
from lerobot.common.constants import HF_LEROBOT_HOME

from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    hf_transform_to_torch,
    MULTIROBOT_TEMPLATE_FEATURES, 
    MULTIROBOT_TEMPLATE_FEATURES_WITH_PAD,
    SINGLE_ROBOT_ONE_CAMERA_TEMPLATE_FEATURES,
    SINGLE_ROBOT_ONE_CAMERA_TEMPLATE_FEATURES_WITH_PAD
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
)
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.configs.types import FeatureType
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.datasets.utils import dataset_to_policy_features_humanoid
# For maintainers, see lerobot/common/datasets/push_dataset_to_hub/CODEBASE_VERSION.md
CODEBASE_VERSION = "v2.1"

from lerobot.common.datasets.compute_stats import aggregate_humanoid_stats, compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from lerobot.common.datasets.ego4d_dataset import Ego4DDataset
import os,psutil,math,json

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(i) for i in obj]
    else:
        return obj
    
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

class HumanoidLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        print(f"repo_id: {repo_id}")
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            if self.episodes is not None:
                try:
                    ep_idx_order = self.episodes.index(ep_idx)
                    query_indices, padding = self._get_query_indices(idx, ep_idx_order)
                except ValueError:
                    print(f"{ep_idx} is not in self.episodes")
            else:
                query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
                item = {**video_frames, **item}
            except Exception as e:
                print(f"Error querying videos: {e}")
                print(f"Problematic query_timestamps: {query_timestamps}")
                print(f"Episode index (ep_idx): {ep_idx}")
                # Continue with original item without video frames
                pass

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                # visualize original image
                # import cv2
                # image_show = item[cam][0].permute(1, 2, 0).numpy() * 255
                # image_show = cv2.cvtColor(image_show.astype(np.uint8), cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f"./visualization/image_{cam}.png", image_show)
                item[cam] = self.image_transforms(item[cam])

                # visualize transformed image
                # import cv2
                # image_show = item[cam][0].permute(1, 2, 0).numpy() * 255
                # image_show = cv2.cvtColor(image_show.astype(np.uint8), cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f"./visualization/image_{cam}.png", image_show)


        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item 
    
    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            cache_dir = str(self.root / ".cache")
            # hf_dataset = load_dataset("parquet", data_dir=path, cache_dir=cache_dir,split="train")
            hf_dataset = load_dataset(path, cache_dir=cache_dir, split="train")
        else:
            rank = int(os.environ.get('RANK', 0))
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            path = str(self.root / "data")
            cache_dir = str(self.root / ".cache" / f"rank_{rank}")
            hf_dataset = load_dataset(path, cache_dir=cache_dir, data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)

        return hf_dataset

class HumanoidMultiLeRobotDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else {repo_id: 1e-4 for repo_id in repo_ids}
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            HumanoidLeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
            
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_humanoid_stats([dataset.meta.stats for dataset in self._datasets])
        self.merged_features = {}  
        for dataset in self._datasets:  
            self.merged_features.update(dataset.features)  
        # depth 数据删除
        depth_keys = [key for key in self.merged_features if 'depth' in key]
        for key in depth_keys:
            # self.disabled_features.add(key)
            del self.merged_features[key]
    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)
    @property
    def robot_type(self) -> str:
        return self._datasets[0].meta.info["robot_type"]
    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )


class HumanoidMultiRobotLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self,
        ALL_TASK_CONFIG,
        config,
        samples_per_epoch=200000000,
        pretrained_policy_path=None,
        heterogenous=True,
        generate_dataset_stats=False,
        random_item=True,
        image_transforms: Callable | None = None,
    ):
        super().__init__()
        self.random_item = random_item
        self.Pretrain_all_TASK_DIR = ALL_TASK_CONFIG['multi_task']
        self.Pretrain_sample_weights = ALL_TASK_CONFIG['sample_weights']
        # for batch sampling
        if config.split_dataset:
            ego4d_indices = [
                i for i, task in enumerate(ALL_TASK_CONFIG['multi_task']) 
                if task['dataset_type'] == 'ego4d'
            ]
            if len(ego4d_indices) == 1:
                world_size = int(os.environ.get('WORLD_SIZE', 1))
                self.Pretrain_sample_weights[ego4d_indices[0]] = [1]*(len(self.Pretrain_sample_weights[ego4d_indices[0]]) // world_size)

        self.sample_rate = self.calculate_sample_rate()
        
        self.policy_config = config
        self.multirobot_datasets = []

        for task_config in self.Pretrain_all_TASK_DIR:
            if task_config['dataset_type'] == 'ego4d':
                print("NOW READING DATASET_MACHINE_TYPE: ", task_config['dataset_type'])
                root = task_config['root']  
                repo_ids = task_config['repo_ids']
                n_obs_steps = config.n_obs_steps
                dataset = Ego4DDataset(json_path=f"{root}/{repo_ids}",interval=25,split_dataset=config.split_dataset,downsample_factor=config.downsample_factor)  

                dataset_features = {
                    'dataset': dataset,
                    'dataset_stats': None,
                    'num_tasks': None,
                    'n_obs_steps': n_obs_steps,
                    'robot_type': task_config['dataset_type'],
                    'dataset_type':task_config['dataset_type'],
                    'dataset_machine_type': root.split('/')[-1],
                    'normalize_inputs': None,
                    'normalize_targets': None,
                    'unnormalize_outputs': None
                }
                self.multirobot_datasets.append(dataset_features)
            else:
                repo_ids = task_config['repo_ids'][:] 
                root = task_config['root']
                if task_config['dataset_type'] == 'oxe':
                    root_path = f"{root}/{repo_ids[0]}"
                else:
                    root_path = root
                print("NOW READING DATASET_MACHINE_TYPE: ", root_path.split('/')[-1]) 

                ds_meta = LeRobotDatasetMetadata(repo_ids[0], f"{root}/{repo_ids[0]}") 
                n_obs_steps = config.n_obs_steps
                action_chunk_size = config.action_chunk_size
                image_interval_steps = config.image_interval_steps
                delta_timestamps = self.resolve_delta_timestamps(n_obs_steps, image_interval_steps, action_chunk_size, ds_meta, action_sample_factor=config.action_sample_factor)
                if config.split_dataset:
                    rank = int(os.environ.get('RANK', 0))
                    world_size = int(os.environ.get('WORLD_SIZE', 1))
                    total_episodes={}
                    rank_total_episodes_num=0
                    for repo_id in repo_ids:
                        ds_meta = LeRobotDatasetMetadata(repo_id, f"{root}/{repo_id}") 
                        if ds_meta.total_episodes == 0:
                            raise ValueError(f"Dataset {repo_id} has no episodes")
                        all_indices = list(range(ds_meta.total_episodes))
                        num_per_rank = math.ceil(ds_meta.total_episodes / world_size)
                        expanded = np.resize(all_indices, num_per_rank * world_size)
                        indices_for_this_rank = expanded[rank::world_size].tolist()
                        rank_total_episodes_num += len(indices_for_this_rank)
                        total_episodes[repo_id] = indices_for_this_rank   

                        rank_total_episodes_num = len(indices_for_this_rank[::config.downsample_factor])    
                        total_episodes[repo_id] = indices_for_this_rank[::config.downsample_factor]

                    print(f"Rank {rank} Total episode in world_size {world_size}: {rank_total_episodes_num}")
                    dataset = HumanoidMultiLeRobotDataset(repo_ids, root=root, episodes=total_episodes,delta_timestamps=delta_timestamps,image_transforms=image_transforms)
                    try:
                        with open(f"{root_path}/dataset_stats.json", "r") as f:
                            dataset_stats_dict = json.load(f)
                        dataset_stats = list_to_ndarray(dataset_stats_dict['dataset_stats'])  
                        dataset.stats = dataset_stats
                        print(f"Read Prepared dataset_stats.json from {root_path}/dataset_stats.json")
                    except:
                        raise ValueError(f"No dataset_stats.json in {root_path}, Please generate first")
                else:
                    if config.small_sample:
                        total_episodes={}                 
                        for repo_id in repo_ids:
                            if task_config['dataset_type'] in ['humanoid_station', 'robomind']:
                                ds_meta = LeRobotDatasetMetadata(repo_id, f"{root}/{repo_id}") 
                                episode_len = min(config.small_sample_rate, ds_meta.total_episodes)
                                total_episodes[repo_id] = list(range(episode_len))
                            else:
                                total_episodes[repo_id] = list(range(config.small_sample_rate))

                        dataset = HumanoidMultiLeRobotDataset(repo_ids, root=root,delta_timestamps=delta_timestamps,episodes=total_episodes, image_transforms=image_transforms)
                    else:
                        dataset = HumanoidMultiLeRobotDataset(repo_ids, root=root,delta_timestamps=delta_timestamps, image_transforms=image_transforms)
                    # daset normalization init
                    dataset_stats = dataset.stats

                # save dataset_stats.json for pretrian & deployment
                '''save dataset_stats.json to specific saving check point path'''
                if pretrained_policy_path is not None:
                    dataset_stats_dict={}
                    dataset_stats_dict['dataset_name']=root_path.split('/')[-1]
                    dataset_stats_dict['root']=root_path
                    dataset_stats_list = ndarray_to_list(dataset.stats)
                    dataset_stats_dict['dataset_stats']=dataset_stats_list
                    for repo_id in repo_ids:
                        os.makedirs(f"{pretrained_policy_path}/{repo_id}", exist_ok=True)
                        with open(f"{pretrained_policy_path}/{repo_id}/dataset_stats.json", "w") as f:
                            json.dump(dataset_stats_dict, f)
                        print(f"save dataset_stats.json to {pretrained_policy_path}/{repo_id}/dataset_stats.json")
                        
                '''save json to specific root path, ready for split dataset only generate for pretrain stage'''
                if generate_dataset_stats:
                    dataset_stats_dict={}
                    dataset_stats_dict['dataset_name']=root_path.split('/')[-1]
                    dataset_stats_dict['root']=root_path
                    dataset_stats_list = ndarray_to_list(dataset.stats)
                    dataset_stats_dict['dataset_stats']=dataset_stats_list
                    with open(f"{root_path}/dataset_stats.json", "w") as f:
                        json.dump(dataset_stats_dict, f)
                    print(f"save dataset_stats.json to {root_path}/dataset_stats.json")
                    with open(f"{root_path}/dataset_stats.json", "r") as f:
                        dataset_stats_dict = json.load(f)
                    dataset_stats = list_to_ndarray(dataset_stats_dict['dataset_stats']) 
                    print(f"load dataset_stats.json from {root_path}/dataset_stats.json")
                
                features = dataset_to_policy_features_humanoid(dataset.merged_features)
                output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
                input_features = {key: ft for key, ft in features.items() if key not in output_features}
                normalize_inputs = Normalize(input_features, config.normalization_mapping, dataset_stats)
                normalize_targets = Normalize(output_features, config.normalization_mapping, dataset_stats)
                unnormalize_outputs = Unnormalize(output_features, config.normalization_mapping, dataset_stats)

                dataset_features = {
                    'dataset': dataset,
                    'dataset_stats': dataset_stats,
                    'num_tasks': len(repo_ids),
                    'n_obs_steps': n_obs_steps, # observation steps
                    'robot_type': dataset.robot_type,
                    'dataset_type':task_config['dataset_type'], # dataset type robomind, humanoid_station, oxe, ego4d
                    'dataset_machine_type': root_path.split('/')[-1],
                    'normalize_inputs': normalize_inputs,
                    'normalize_targets': normalize_targets,
                    'unnormalize_outputs': unnormalize_outputs
                }
                self.multirobot_datasets.append(dataset_features)

        if len(self.multirobot_datasets) == 1:
            if 'tienkung' in self.multirobot_datasets[0]['robot_type']:
                self.heterogeneous = False
                if config.n_obs_steps > 1:
                    self.template_features = SINGLE_ROBOT_ONE_CAMERA_TEMPLATE_FEATURES_WITH_PAD
                else:
                    self.template_features = SINGLE_ROBOT_ONE_CAMERA_TEMPLATE_FEATURES
                print("Only one dataset, is tienkung, Heterogeneous: False")
            else:
                if config.n_obs_steps > 1:
                    self.template_features = MULTIROBOT_TEMPLATE_FEATURES_WITH_PAD
                else:
                    self.template_features = MULTIROBOT_TEMPLATE_FEATURES
                self.heterogeneous = True
                print("One datasets, But Heterogeneous: True")
        else:
            if config.n_obs_steps > 1:
                self.template_features = MULTIROBOT_TEMPLATE_FEATURES_WITH_PAD
            else:
                self.template_features = MULTIROBOT_TEMPLATE_FEATURES
            self.heterogeneous = True
            print("Multiple datasets, Heterogeneous: True")

        print('Dataset Number: ', len(self.multirobot_datasets))

        self.samples_per_epoch = samples_per_epoch
        self.epoch_indices = {}

    def get_dataset_stats(self):
        return self.multirobot_datasets[0]['dataset_stats']

    def resolve_delta_timestamps( self, n_obs_steps, image_interval_steps, action_chunk_size, ds_meta: LeRobotDatasetMetadata,action_sample_factor=1) -> dict[str, list] | None:
        action_delta_indices = list(range(action_chunk_size))
        observation_delta_indices = list(range(0, n_obs_steps * image_interval_steps, image_interval_steps))[:n_obs_steps]
        delta_timestamps = {}
        for key in ds_meta.features:
            if key.startswith("action"):
                delta_timestamps[key] = [i * action_sample_factor / ds_meta.fps for i in action_delta_indices]
            if key.startswith("observation."):
                delta_timestamps[key] = [i * action_sample_factor / ds_meta.fps for i in observation_delta_indices]

        if len(delta_timestamps) == 0:
            delta_timestamps = None

        return delta_timestamps
    def calculate_sample_rate(self):
        # calculate the total sum of each sublist
        total_weights = [sum(weights) for weights in self.Pretrain_sample_weights]
        # calculate the total sum of all weights
        total_sum = sum(total_weights)
        # calculate the ratio of each sublist
        sample_rate = [weight / total_sum for weight in total_weights]
        return sample_rate
    def robomind_prepare_item_to_common_format(self, item):
        item_new = item
        return item_new
    def oxe_prepare_item_to_common_format(self, item):
        item_new = item
        return item_new

    def ego4d_prepare_item_to_common_format(self, item):
        item_new = item
        item_new["task"] = "human object interaction"
        item_new["observation.state"] = torch.empty((self.policy_config.n_obs_steps, 0), dtype=torch.float32)
        item_new['observation.state_is_pad'] = torch.zeros((self.policy_config.n_obs_steps), dtype=torch.bool)
        item_new["action"] = torch.empty((self.policy_config.action_chunk_size, 0), dtype=torch.float32)
        item_new['action_is_pad'] = torch.zeros((self.policy_config.action_chunk_size), dtype=torch.bool)
        return item_new

    def pad_vector(self, vector, new_dim):
        if vector.shape[-1] == new_dim:
            return vector
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector
        return new_vector
    # create mask: 1 represents the original value, 0 represents the padding value
    def pad_vector_mask(self,vector, new_dim):
        if vector.shape[-1] == new_dim:
            mask = torch.zeros(*vector.shape, dtype=torch.bool, device=vector.device)
            mask[..., :vector.shape[-1]] = True
            return vector, mask
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector

        # create mask: 1 represents the original value, 0 represents the padding value
        mask = torch.zeros(*shape, dtype=torch.bool, device=vector.device)
        mask[..., :current_dim] = True

        return new_vector, mask
    def __len__(self):
        if self.random_item:
            return self.samples_per_epoch # directly define the number of iterations, let it randomly extract, index is not used
        else:
            print(f"total num frames: {self.multirobot_datasets[0]['dataset'].num_frames}")
            return self.multirobot_datasets[0]['dataset'].num_frames


    def humanoid_station_prepare_item_to_common_format(self, item):
        observation_images = {}
        observation_images_is_pad = {}
        item_new = {}
        bgr_flag = False

        # task
        item_new["task"] = item["task"]
        # image
        for key in item.keys():
            if 'observation' in key and 'image' in key and 'is_pad' not in key:
                observation_images[key] = item[key]
            if 'observation' in key and 'image' in key and 'is_pad' in key:
                observation_images_is_pad[key] = item[key]
            if key.startswith('observation.state'):
                item_new[key] = item[key]
            if key.startswith('action'):
                item_new[key] = item[key]

        # state & action 
        if item['robot_type'] in ['dual_arm_franka', 'dual_arm_agx']:
            ctl_elem_key = ['arm_joint_position'] # hand_joint_position is in the arm_joint_position

        elif item['robot_type'] in ['dual_arm_ur', 'dual_arm_tien_kung2', 'single_arm_ur']:
            ctl_elem_key = ['arm_joint_position','hand_joint_position'] # hand_joint_position is not in the arm_joint_position
        else:
            raise ValueError(f"Unsupported robot type: {item['robot_type']}")

        wrist_index = 0
        index = 0
        for key, value in observation_images.items():

            if bgr_flag:
                value = value[:, [2, 1, 0], :, :]  # exchange channel to bgr

            if 'wrist' in key:
                item_new[f'observation.images.image_wrist_{wrist_index}'] = value
                wrist_index += 1
            else:
                item_new[f'observation.images.image_{index}'] = value
                index += 1
        wrist_index = 0
        index = 0
        for key, value in observation_images_is_pad.items():
            if 'wrist' in key:
                item_new[f'observation.images.image_wrist_{wrist_index}_is_pad'] = value
                wrist_index += 1
            else:
                item_new[f'observation.images.image_{index}_is_pad'] = value
                index += 1
        if isinstance(ctl_elem_key, list):
            item_new['robot_type'] = item['robot_type']
            state_list = []
            action_list = []
            for key in ctl_elem_key:
                if f'observation.state.{key}' in item_new:
                    if len(item_new[f'observation.state.{key}'].shape) == 1:
                        item_new[f'observation.state.{key}'] = item_new[f'observation.state.{key}'].unsqueeze(-1)
                    state_list.append(item_new[f'observation.state.{key}'])
                if f'action.{key}' in item_new:
                    if len(item_new[f'action.{key}'].shape) == 1:
                        item_new[f'action.{key}'] = item_new[f'action.{key}'].unsqueeze(-1)
                    action_list.append(item_new[f'action.{key}'])

            item_new['observation.state'] = torch.cat(state_list, dim=-1)
            item_new['observation.state_is_pad'] = item_new['observation.state.arm_joint_position_is_pad']
            item_new['action'] = torch.cat(action_list, dim=-1)
            item_new['action_is_pad'] = item_new['action.arm_joint_position_is_pad']

        return item_new

    def delete_depth_key(self, item):
        depth_keys = [key for key in item if 'depth' in key]
        for key in depth_keys:
            del item[key]
    def delete_not_for_train_key(self, item_new):
        for key in list(item_new.keys()):
            if key not in self.template_features.keys(): 
                if key != 'task':
                    del item_new[key]
    def pad_vector_to_template(self, item_new, picked_dataset):
        for key, feature in self.template_features.items():
            if key not in item_new:
                dtype = feature['dtype']
                if dtype == 'video':
                    shape = feature['shape']
                    if  picked_dataset['n_obs_steps'] == 2:
                        seq_len = picked_dataset['n_obs_steps']
                        new_shape = (seq_len, *shape)
                        default_value = torch.zeros(new_shape, dtype=torch.float32)
                    else:
                        default_value = torch.zeros(shape, dtype=torch.float32)
                elif dtype == 'bool':
                    shape = feature['shape']
                    new_shape = (picked_dataset['n_obs_steps'], *shape[1:])
                    default_value = torch.ones(new_shape, dtype=torch.bool)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
                item_new[key] = default_value
            elif key == 'action' or key == 'observation.state':
                target_shape = feature['shape']
                item_new[key], mask = self.pad_vector_mask(item_new[key], target_shape[0])
                item_new[key+'_mask'] = mask
        return item_new

    def pad_vector_to_single_dataset(self, item_new):
        for key, feature in self.template_features.items():
            if key == 'action' or key == 'observation.state':
                target_shape = feature['shape']
                item_new[key], mask = self.pad_vector_mask(item_new[key], target_shape[0])
                item_new[key+'_mask'] = mask
        return item_new


    def resize_with_pad(self, img, width, height, pad_value=-1):

        cur_height, cur_width = img.shape[-2:] # last 2 dimensions

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  
            resized_img = F.interpolate(
                img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )
            resized_img = resized_img.squeeze(0)
        else:
            resized_img = F.interpolate(
                img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

    def resize_images(self, item):
        for key in item.keys():
            if 'observation' in key and 'image' in key and 'is_pad' not in key:
                item[key] = self.resize_with_pad(item[key], *self.policy_config.resize_imgs_with_padding, pad_value=0)


    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ind = np.random.choice(list(range(len(self.multirobot_datasets))), p=self.sample_rate)
        picked_dataset = self.multirobot_datasets[ind]
        # randomly select a episode corresponding frame
        if picked_dataset['dataset_type'] == 'ego4d':
            random_index = np.random.randint(0, picked_dataset['dataset'].num_episodes)
        else:
            if self.random_item:
                # maintain independent sampling sequence for each dataset
                if ind not in self.epoch_indices or len(self.epoch_indices[ind]) == 0:
                    # when the index of a dataset is used up, generate a shuffled sequence again
                    self.epoch_indices[ind] = np.random.permutation(picked_dataset['dataset'].num_frames)
                # take out an index from the current sequence and delete it
                random_index = int(self.epoch_indices[ind][-1])  # convert to Python int type
                self.epoch_indices[ind] = self.epoch_indices[ind][:-1]
            else:
                random_index = idx % picked_dataset['dataset'].num_frames
            # random_index = idx % picked_dataset['dataset'].num_frames
            # random_index = idx % picked_dataset['dataset'].num_frames

        item = picked_dataset['dataset'][random_index]
        item['robot_type'] = picked_dataset['robot_type']
        # to remove the bug, delete the end_effector提前删去end_effector
        keys = list(item.keys())
        for key in keys:
            if "end_effector" in key:
                del item[key]
        if self.heterogeneous:        
            # normalization
            item = picked_dataset['normalize_inputs'](item) if picked_dataset['normalize_inputs'] is not None else item
            item = picked_dataset['normalize_targets'](item) if picked_dataset['normalize_targets'] is not None else item
            # align all images to template
            # 1、删去depth 相关的key
            self.delete_depth_key(item)
            #  resize all images to the size specified by the policy
            self.resize_images(item) 
            # 2、IMPORTANT: convert item to common format
            if picked_dataset['dataset_type'] == 'humanoid_station':
                item_new = self.humanoid_station_prepare_item_to_common_format(item)
            elif picked_dataset['dataset_type'] == 'robomind':
                item_new = self.robomind_prepare_item_to_common_format(item)
            elif picked_dataset['dataset_type'] == 'oxe':
                item_new = self.oxe_prepare_item_to_common_format(item)
            elif picked_dataset['dataset_type'] == 'ego4d':
                item_new = self.ego4d_prepare_item_to_common_format(item)
            else:
                raise ValueError(f"Unsupported dataset type: {picked_dataset['dataset_type']}")
            # 3、delete keys not in feature but keep is_pad keys
            self.delete_not_for_train_key(item_new)
            
            # 4、align with template, missing keys
            item_new = self.pad_vector_to_template(item_new, picked_dataset)
        else:
            # normalization
            item = picked_dataset['normalize_inputs'](item) if picked_dataset['normalize_inputs'] is not None else item
            item = picked_dataset['normalize_targets'](item) if picked_dataset['normalize_targets'] is not None else item
            # align all images to template
            # 1、delete keys related to depth
            self.delete_depth_key(item)
            # resize all images to the size specified by the policy
            self.resize_images(item) 
            # 2、IMPORTANT: convert item to common format
            if picked_dataset['dataset_type'] == 'humanoid_station':
                item_new = self.humanoid_station_prepare_item_to_common_format(item)
            else:
                raise ValueError(f"Single type dataset, Unsupported dataset type: {picked_dataset['dataset_type']}")
            # 3、align with single dataset, missing keys
            item_new = self.pad_vector_to_single_dataset(item_new)
        return item_new
    
