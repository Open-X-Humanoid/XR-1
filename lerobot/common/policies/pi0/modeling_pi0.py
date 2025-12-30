#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math,json
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype
from deploy.real_robot.root_path import REAL_ROBOT_DEPLOY_ROOT_PATH
from lerobot.configs.types import FeatureType,PolicyFeature
def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        ) 

        if self.config.real_robot_dev:
            self.language_tokenizer = AutoTokenizer.from_pretrained(f"{REAL_ROBOT_DEPLOY_ROOT_PATH}/PretrainModel/paligemma-3b-pt-224")
        else:
            self.language_tokenizer = AutoTokenizer.from_pretrained("/media/jushen/bamboo-fan/dev/PretrainModel/paligemma-3b-pt-224")

        self.model = PI0FlowMatching(config)

        self.reset(action_horizon=self.config.n_action_steps)

    def list_to_ndarray(self,obj):
        if isinstance(obj, list):
            # 判断是不是纯数字的 list（可转为 ndarray）
            if obj and all(isinstance(x, (int, float, bool)) for x in obj):
                return np.array(obj)
            # 判断是不是嵌套 list（如矩阵/多维数组）
            if obj and all(isinstance(x, list) and all(isinstance(y, (int, float, bool)) for y in x) for x in obj):
                return np.array(obj)
            # 递归处理更深层次
            return [self.list_to_ndarray(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.list_to_ndarray(v) for k, v in obj.items()}
        else:
            return obj    

    def init_dataset_stats(self,device,robot_type,path=None):
        self.robot_type = robot_type
        if path is None:
            with open(f"{self.config.pretrained_path}/dataset_stats.json", "r") as f:
                dataset_stats_dict = json.load(f)
            print(f"load dataset_stats.json from {self.config.pretrained_path}/dataset_stats.json")
        else:
            with open(f"{path}/dataset_stats.json", "r") as f:
                dataset_stats_dict = json.load(f)
            print(f"load dataset_stats.json from {path}/dataset_stats.json")
        self.dataset_stats = dataset_stats_dict['dataset_stats']
        dataset_stats = self.list_to_ndarray(dataset_stats_dict['dataset_stats']) 
        if 'franka' in robot_type:
            self.config.action_feature.shape = dataset_stats['action.arm_joint_position']['mean'].shape
            self.config.robot_state_feature.shape = dataset_stats['observation.state.arm_joint_position']['mean'].shape
            output_features = {'action.arm_joint_position':self.config.action_feature}
            input_features={'observation.state.arm_joint_position':self.config.robot_state_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif 'ur' in robot_type:
            self.config.action_feature.shape = (dataset_stats['action.arm_joint_position']['mean'].shape[0] + dataset_stats['action.hand_joint_position']['mean'].shape[0],)
            action_arm_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.arm_joint_position']['mean'].shape)
            action_hand_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.hand_joint_position']['mean'].shape)
            state_arm_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.arm_joint_position']['mean'].shape)
            state_hand_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.hand_joint_position']['mean'].shape)

            output_features = {'action.arm_joint_position':action_arm_feature,'action.hand_joint_position':action_hand_feature}
            input_features={'observation.state.arm_joint_position':state_arm_feature,'observation.state.hand_joint_position':state_hand_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif 'tienkung' in robot_type:
            # print(f"dataset_stats: {dataset_stats}")
            self.config.action_feature.shape = (dataset_stats['action.arm_joint_position']['mean'].shape[0] + dataset_stats['action.hand_joint_position']['mean'].shape[0],)
            action_arm_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.arm_joint_position']['mean'].shape)
            action_hand_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.hand_joint_position']['mean'].shape)
            state_arm_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.arm_joint_position']['mean'].shape)
            state_hand_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.hand_joint_position']['mean'].shape)

            output_features = {'action.arm_joint_position':action_arm_feature,'action.hand_joint_position':action_hand_feature}
            input_features={'observation.state.arm_joint_position':state_arm_feature,'observation.state.hand_joint_position':state_hand_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif robot_type == 'agilex_dual':

            self.config.action_feature.shape = dataset_stats['action.arm_joint_position']['mean'].shape # + dataset_stats['action.chassis_state_twist']['mean'].shape[0]
            self.config.robot_state_feature.shape = dataset_stats['observation.state.arm_joint_position']['mean'].shape
            
            action_arm_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.arm_joint_position']['mean'].shape)
            # action_chassis_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.chassis_state_twist']['mean'].shape)
            state_arm_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.arm_joint_position']['mean'].shape)
            state_chasis_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.chassis_state_twist']['mean'].shape)

            output_features = {'action.arm_joint_position':action_arm_feature} #，'action.chassis_state_twist':action_chassis_feature}
            input_features={'observation.state.arm_joint_position':state_arm_feature,'observation.state.chassis_state_twist':state_chasis_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif robot_type == 'ark_lift_dual':
            self.config.action_feature.shape = (dataset_stats['action.arm_joint_position']['mean'].shape[0] + dataset_stats['action.chassis_state_coor']['mean'].shape[0],)
            action_arm_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.arm_joint_position']['mean'].shape)
            action_chassis_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.chassis_state_coor']['mean'].shape)
            state_arm_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.arm_joint_position']['mean'].shape)
            state_chassis_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.chassis_state_coor']['mean'].shape)

            output_features = {'action.arm_joint_position':action_arm_feature,'action.chassis_state_coor':action_chassis_feature}
            input_features={'observation.state.arm_joint_position':state_arm_feature,'observation.state.chassis_state_coor':state_chassis_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif 'agilex_1' in robot_type:
            self.config.action_feature.shape = dataset_stats['action.arm_joint_position']['mean'].shape
            self.config.robot_state_feature.shape = dataset_stats['observation.state.arm_joint_position']['mean'].shape
            output_features = {'action.arm_joint_position':self.config.action_feature}
            input_features={'observation.state.arm_joint_position':self.config.robot_state_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif 'agilex_new_chassis' in robot_type:
            self.config.action_feature.shape = (dataset_stats['action.arm_joint_position']['mean'].shape[0] + dataset_stats['action.hand_joint_position']['mean'].shape[0] + dataset_stats['action.chassis_twist']['mean'].shape[0],)
            
            action_arm_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.arm_joint_position']['mean'].shape)
            action_hand_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.hand_joint_position']['mean'].shape)
            action_chassis_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.chassis_twist']['mean'].shape)
            
            state_arm_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.arm_joint_position']['mean'].shape)
            state_hand_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.hand_joint_position']['mean'].shape)
            state_chassis_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.chassis_twist']['mean'].shape)

            output_features = {'action.arm_joint_position':action_arm_feature,'action.hand_joint_position':action_hand_feature,'action.chassis_twist':action_chassis_feature}
            input_features={'observation.state.arm_joint_position':state_arm_feature,'observation.state.hand_joint_position':state_hand_feature,'observation.state.chassis_twist':state_chassis_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True
        elif 'tianyi_mob' in robot_type:
            action_arm_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.arm_joint_position']['mean'].shape)
            action_hand_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.hand_joint_position']['mean'].shape)
            action_chassis_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action.chassis_twist']['mean'].shape)
            
            state_arm_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.arm_joint_position']['mean'].shape)
            state_hand_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.hand_joint_position']['mean'].shape)
            state_chassis_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state.chassis_twist']['mean'].shape)

            output_features = {'action.arm_joint_position':action_arm_feature,'action.hand_joint_position':action_hand_feature,'action.chassis_twist':action_chassis_feature}
            input_features={'observation.state.arm_joint_position':state_arm_feature,'observation.state.hand_joint_position':state_hand_feature,'observation.state.chassis_twist':state_chassis_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True

        elif 'widowx' in robot_type:
            self.config.action_feature.shape = (dataset_stats['action']['mean'].shape[0],)
            action_feature = PolicyFeature(type=FeatureType.ACTION, shape=dataset_stats['action']['mean'].shape)
            state_feature = PolicyFeature(type=FeatureType.STATE, shape=dataset_stats['observation.state']['mean'].shape)
    
            output_features = {'action':action_feature}
            input_features={'observation.state':state_feature}
            self.unnormalize_outputs = Unnormalize(output_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalize_inputs = Normalize(input_features, self.config.normalization_mapping, dataset_stats).to(device)
            self.normalization = True            
        else:
            raise ValueError(f"Invalid robot type: {robot_type}")

    def reset(self, action_horizon: int, ensemble: bool = True, 
              exp_weight: float = 0.05, action_execute_steps: int = 1, 
              sampled_action_factor: int = 1):
        """This should be called whenever the environment is reset."""
        self.ensemble = ensemble
        if self.ensemble:
            self.exp_weight = exp_weight
            self.action_execute_steps=action_execute_steps
        self.sampled_action_factor = sampled_action_factor
        self.action_horizon = action_horizon
        self.action_history = deque([], maxlen=math.ceil(action_horizon / self.sampled_action_factor))
        
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], action_horizon: int | None = None,noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.normalization:
            if 'franka' in self.robot_type:
                batch['observation.state'] = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
            elif 'ur' in self.robot_type:
                norm_state_arm  = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_hand = self.normalize_inputs({"observation.state.hand_joint_position": batch['observation.state.hand_joint_position']})["observation.state.hand_joint_position"]
                batch['observation.state'] = torch.cat([norm_state_arm,norm_state_hand],dim=-1)
            elif 'tienkung' in self.robot_type:
                norm_state_arm  = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_hand = self.normalize_inputs({"observation.state.hand_joint_position": batch['observation.state.hand_joint_position']})["observation.state.hand_joint_position"]
                batch['observation.state'] = torch.cat([norm_state_arm,norm_state_hand],dim=-1)
            elif self.robot_type == 'agilex_dual':
                norm_state_arm = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_chasis = self.normalize_inputs({"observation.state.chassis_state_twist": batch['observation.state.chassis_state_twist']})['observation.state.chassis_state_twist']
                norm_state_chasis = norm_state_chasis[:, [0,5]]
                batch['observation.state'] = torch.cat([norm_state_arm, norm_state_chasis], dim=-1)
            elif self.robot_type == 'ark_lift_dual':
                norm_state_arm = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_chasis = self.normalize_inputs({"observation.state.chassis_state_coor": batch['observation.state.chassis_state_coor']})['observation.state.chassis_state_coor']
                batch['observation.state'] = torch.cat([norm_state_arm, norm_state_chasis], dim=-1)
            elif 'agilex_1' in self.robot_type:
                batch['observation.state'] = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
            elif 'agilex_new_chassis' in self.robot_type:
                norm_state_arm = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_hand = self.normalize_inputs({"observation.state.hand_joint_position": batch['observation.state.hand_joint_position']})["observation.state.hand_joint_position"]
                norm_state_chasis = self.normalize_inputs({"observation.state.chassis_twist": batch['observation.state.chassis_twist']})['observation.state.chassis_twist']
                batch['observation.state'] = torch.cat([norm_state_arm, norm_state_hand, norm_state_chasis], dim=-1)
            elif 'tianyi_mob' in self.robot_type:
                norm_state_arm = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_hand = self.normalize_inputs({"observation.state.hand_joint_position": batch['observation.state.hand_joint_position']})["observation.state.hand_joint_position"]
                norm_state_chasis = self.normalize_inputs({"observation.state.chassis_twist": batch['observation.state.chassis_twist']})['observation.state.chassis_twist']
                batch['observation.state'] = torch.cat([norm_state_arm, norm_state_hand, norm_state_chasis], dim=-1)
            elif 'widowx' in self.robot_type:
                batch['observation.state'] = self.normalize_inputs({"observation.state": batch['observation.state']})["observation.state"]     
            else:
                raise ValueError(f"Invalid robot type: {self.robot_type} in normalization")


        # batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        # actions = self.unnormalize_outputs({"action": actions})["action"]

        if self.normalization:
            if 'franka' in self.robot_type:
                actions = self.unnormalize_outputs({"action.arm_joint_position": actions})["action.arm_joint_position"]
            elif 'ur_dual' in self.robot_type:
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:12]})["action.arm_joint_position"]
                actions_hand = self.unnormalize_outputs({"action.hand_joint_position": actions[:,:,12:]})["action.hand_joint_position"]
                actions = torch.cat([actions_arm,actions_hand],dim=-1)
            elif 'ur_single' in self.robot_type:
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:6]})["action.arm_joint_position"]
                actions_hand = self.unnormalize_outputs({"action.hand_joint_position": actions[:,:,6:]})["action.hand_joint_position"]
                actions = torch.cat([actions_arm,actions_hand],dim=-1)
            elif 'tienkung' in self.robot_type:
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:14]})["action.arm_joint_position"]
                actions_hand = self.unnormalize_outputs({"action.hand_joint_position": actions[:,:,14:]})["action.hand_joint_position"]
                actions = torch.cat([actions_arm,actions_hand],dim=-1)
            elif self.robot_type == 'agilex_dual':
                actions = self.unnormalize_outputs({"action.arm_joint_position": actions})["action.arm_joint_position"]
            elif self.robot_type == 'ark_lift_dual':
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:14]})["action.arm_joint_position"]
                actions_chasis = self.unnormalize_outputs({"action.chassis_state_coor": actions[:,:,14:]})["action.chassis_state_coor"]
                actions = torch.cat([actions_arm,actions_chasis],dim=-1)
            elif 'agilex_1' in self.robot_type:
                actions = self.unnormalize_outputs({"action.arm_joint_position": actions})["action.arm_joint_position"]
            elif 'agilex_new_chassis' in self.robot_type:
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:12]})["action.arm_joint_position"]
                actions_hand = self.unnormalize_outputs({"action.hand_joint_position": actions[:,:,12:14]})["action.hand_joint_position"]
                actions_chasis = self.unnormalize_outputs({"action.chassis_twist": actions[:,:,14:]})["action.chassis_twist"]
                actions = torch.cat([actions_arm,actions_hand,actions_chasis],dim=-1)
            elif 'tianyi_mob' in self.robot_type:
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:12]})["action.arm_joint_position"]
                actions_hand = self.unnormalize_outputs({"action.hand_joint_position": actions[:,:,12:14]})["action.hand_joint_position"]
                actions_chasis = self.unnormalize_outputs({"action.chassis_twist": actions[:,:,14:]})["action.chassis_twist"]
                actions = torch.cat([actions_arm,actions_hand,actions_chasis],dim=-1)
            elif 'widowx' in self.robot_type:
                actions = self.unnormalize_outputs({"action": actions})["action"]
            else:
                raise ValueError(f"Invalid robot type: {self.robot_type} in normalization")
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)

        for key in list(batch.keys()):
            if key.startswith('observation.images') and len(batch[key].shape) == 5:
                batch[key] = batch[key][:, 0, :, :, :]  # 结果形状: [batch_size, channels, height, width]
                assert len(batch[key].shape) == 4
            elif key.startswith('observation.state') and len(batch[key].shape) == 3:
                batch[key] = batch[key][:, 0, :]
                assert len(batch[key].shape) == 2

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        # actions = self.prepare_action(batch)
        actions, batch['action_mask'] = self.prepare_action_mask(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        '''action cut off''' 
        # if actions_is_pad is not None:
        #     in_episode_bound = ~actions_is_pad
        #     action_mask = in_episode_bound.unsqueeze(-1) * batch['action_mask']
        # losses = losses[action_mask]  
        '''action original''' 
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [
            key for key in self.config.image_features if key not in batch
        ]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def pad_vector_mask(self, vector, new_dim):
        """Can be (batch_size x sequence_length x features_dimension)
        or (batch_size x features_dimension)
        """
        if vector.shape[-1] == new_dim:
            mask = torch.zeros(*vector.shape, dtype=torch.bool, device=vector.device)
            mask[..., :vector.shape[-1]] = True
            return vector, mask
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector

        # 创建mask: 1表示原始值，0表示填充值
        mask = torch.zeros(*shape, dtype=torch.bool, device=vector.device)
        mask[..., :current_dim] = True

        return new_vector, mask

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def prepare_action_mask(self, batch):
        actions,action_mask = self.pad_vector_mask(batch[ACTION], self.config.max_action_dim)
        return actions,action_mask

class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        paligemma_with_export_config.real_robot_dev = self.config.real_robot_dev
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)
       
        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
