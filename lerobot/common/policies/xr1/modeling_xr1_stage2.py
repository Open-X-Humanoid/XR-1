
import math,random
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.xr1.configuration_xr1_stage2 import Xr1Stage2Config
from lerobot.common.policies.xr1.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy

from lerobot.common.policies.xr1.cross_stage_function.cross_stage import (
    Stage1SupervisedModel,
)
from lerobot.common.policies.xr1.cross_stage_function.common_func import *
from lerobot.common.policies.xr1.cross_stage_function.cross_stage_module import PerceiverResampler
import json
import numpy as np
from lerobot.configs.types import FeatureType,PolicyFeature
from deploy.real_robot.root_path import REAL_ROBOT_DEPLOY_ROOT_PATH
from lerobot.common.datasets.utils import MULTIROBOT_TEMPLATE_FEATURES_WITH_PAD
class Xr1Stage2Policy(PreTrainedPolicy):
    config_class = Xr1Stage2Config
    name = "xr1_stage2"

    def __init__(
        self,
        config: Xr1Stage2Config,
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
        if self.config.real_robot_dev:
            self.language_tokenizer = AutoTokenizer.from_pretrained(f"{REAL_ROBOT_DEPLOY_ROOT_PATH}/PretrainModel/paligemma-3b-pt-224")
        else:
            self.language_tokenizer = AutoTokenizer.from_pretrained("../pretrained/paligemma-3b-pt-224")
        if self.config.stage1_supervised:
            self.stage1_model = Stage1SupervisedModel(config)  # stage1 suppervised model
            self.stage2_model = Stage2Model(config) # stage2 model
            if self.config.action_branch:
                # ACTION WEIGHT INITIALIZATION
                self.init_action_weights_from_stage1()
        else:
            self.stage2_model = Stage2Model(config)
        self.normalization = False


        self.reset(action_horizon=self.config.n_action_steps)

    def init_action_weights_from_stage1(self):
        try:
            # action expert init parameters by stage1 action decoder
            self.stage2_model.paligemma_with_expert.gemma_expert.load_state_dict(
                self.stage1_model.action_decoder.motion_token_expert.gemma_expert.state_dict(),strict=True)
            
            self.stage2_model.state_proj.load_state_dict(
                self.stage1_model.action_decoder.state_proj.state_dict(),strict=True)
            self.stage2_model.action_in_proj.load_state_dict(
                self.stage1_model.action_decoder.action_in_proj.state_dict(),strict=True)
            self.stage2_model.action_out_proj.load_state_dict(
                self.stage1_model.action_decoder.action_out_proj.state_dict(),strict=True)
            self.stage2_model.action_time_mlp_in.load_state_dict(
                self.stage1_model.action_decoder.action_time_mlp_in.state_dict(),strict=True)
            self.stage2_model.action_time_mlp_out.load_state_dict(
                self.stage1_model.action_decoder.action_time_mlp_out.state_dict(),strict=True)
            
            del self.stage1_model.action_decoder
            self.stage1_model.action_decoder = None
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during init_action_weights_from_stage1: {e}")
        
    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:

        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)
        batch_train = batch
        batch_size = batch_train[OBS_ROBOT].shape[0]
        device = batch_train[OBS_ROBOT].device
        if self.config.stage1_supervised:
            token_seq_len = self.config.action_latent_token_num
            if self.config.action_branch and self.config.vision_branch:
                # supervised signal action latent
                action_latent_token = self.stage1_model.get_action_latent_token(batch_train)
                action_latent_token_mask = torch.ones((action_latent_token.shape[0], action_latent_token.shape[1]), 
                                            dtype=torch.bool,
                                            device=action_latent_token.device)
                # supervised signal image latent
                images, img_masks = self.prepare_images_stage1(batch_train)
                
                motion_latent_token = []
                motion_latent_token_mask = []

                for i in range(0, len(images), 2):
                    if i + 1 < len(images):
                        image_mask = img_masks[i]
                        if not image_mask.any():  # if mask is all False
                            motion_latent_token.append(torch.zeros(batch_size,token_seq_len,self.config.codebook_embed_dim,device=device))
                            motion_latent_token_mask.append(image_mask.unsqueeze(1).expand(-1, token_seq_len))
                            continue  # skip this iteration
                        cond_hidden_states = self.stage1_model.vision_tower(images[i]).last_hidden_state
                        target_hidden_states = self.stage1_model.vision_tower(images[i + 1]).last_hidden_state
                        motion_latent_token.append(self.stage1_model.get_motion_latent_token(cond_hidden_states,target_hidden_states))
                        motion_latent_token_mask.append(image_mask.unsqueeze(1).expand(-1, token_seq_len).bool())        
                motion_latent_token = torch.cat(motion_latent_token,dim=1)
                motion_latent_token_mask = torch.cat(motion_latent_token_mask,dim=1)

                # stage1's supervised signal
                latent_token = torch.cat([action_latent_token,motion_latent_token],dim=1)
                latent_token_mask = torch.cat([action_latent_token_mask,motion_latent_token_mask],dim=1)

            elif self.config.action_branch:
                # supervised signal action latent
                action_latent_token = self.stage1_model.get_action_latent_token(batch_train)
                action_latent_token_mask = torch.ones((action_latent_token.shape[0], action_latent_token.shape[1]), 
                                            dtype=torch.bool,
                                            device=action_latent_token.device)
                # stage1's supervised signal
                latent_token = action_latent_token
                latent_token_mask = action_latent_token_mask
            elif self.config.vision_branch:
                # supervised signal image latent
                images, img_masks = self.prepare_images_stage1(batch_train)
                
                motion_latent_token = []
                motion_latent_token_mask = []

                for i in range(0, len(images), 2):
                    if i + 1 < len(images):
                        image_mask = img_masks[i]
                        if not image_mask.any():  # if mask is all False
                            motion_latent_token.append(torch.zeros(batch_size,token_seq_len,self.config.codebook_embed_dim,device=device))
                            motion_latent_token_mask.append(image_mask.unsqueeze(1).expand(-1, token_seq_len))
                            continue  # skip this iteration
                        cond_hidden_states = self.stage1_model.vision_tower(images[i]).last_hidden_state
                        target_hidden_states = self.stage1_model.vision_tower(images[i + 1]).last_hidden_state
                        motion_latent_token.append(self.stage1_model.get_motion_latent_token(cond_hidden_states,target_hidden_states))
                        motion_latent_token_mask.append(image_mask.unsqueeze(1).expand(-1, token_seq_len).bool())        
                motion_latent_token = torch.cat(motion_latent_token,dim=1)
                motion_latent_token_mask = torch.cat(motion_latent_token_mask,dim=1)
                # stage1's supervised signal
                latent_token = motion_latent_token
                latent_token_mask = motion_latent_token_mask
            else:
                raise ValueError(f"Invalid branch configuration: {self.config.action_branch} and {self.config.vision_branch}")
        else:
            latent_token = None
            latent_token_mask = None

        # the input images and state are the same as pi0
        for key in list(batch_train.keys()):
            if key.startswith('observation.images') and len(batch_train[key].shape) == 5:
                batch_train[key] = batch_train[key][:, 0, :, :, :]  # result shape: [batch_size, channels, height, width]
                assert len(batch_train[key].shape) == 4
            elif key.startswith('observation.state') and len(batch_train[key].shape) == 3:
                batch_train[key] = batch_train[key][:, 0, :]
                assert len(batch_train[key].shape) == 2
        images, img_masks = self.prepare_images(batch_train) # 6 view list b*3*224*224
        lang_tokens, lang_masks = self.prepare_language(batch_train)

        state = self.prepare_state(batch_train)

        actions, batch_train['action_mask'] = self.prepare_action_mask(batch_train)
        actions = actions[:,:self.config.n_action_steps,:] 
        actions_is_pad = batch_train.get("action_is_pad")[:,:self.config.n_action_steps] 


        stage2_losses_dict, recons_actions_loss = self.stage2_model.forward(
            images, img_masks, lang_tokens, lang_masks, 
            latent_token, latent_token_mask, 
            state, actions,
            noise, time)
        
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            if self.config.cut_off_action:
                action_mask = in_episode_bound.unsqueeze(-1) * batch_train['action_mask'][:,:self.config.n_action_steps,:]
            else:
                action_mask = in_episode_bound  # dimension preserved 32
           
        stage2_losses_dict['recons_actions_loss'] = recons_actions_loss[action_mask].mean() 
        
        losses = stage2_losses_dict['latent_action_loss'] + stage2_losses_dict['image_latent_loss'] \
                    + stage2_losses_dict['recons_actions_loss']

        stage2_losses_dict = {k: v.item() for k, v in stage2_losses_dict.items()}
        return losses, stage2_losses_dict
    
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
    
    def list_to_ndarray(self,obj):
        if isinstance(obj, list):
            # check if the list is pure numbers (can be converted to ndarray)
            if obj and all(isinstance(x, (int, float, bool)) for x in obj):
                return np.array(obj)
            # check if the list is nested list (like matrix/multi-dimensional array)
            if obj and all(isinstance(x, list) and all(isinstance(y, (int, float, bool)) for y in x) for x in obj):
                return np.array(obj)
            # recursively process deeper levels
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
        
        elif 'dual_arm_tien_kung2' in robot_type:
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
        else:
            raise ValueError(f"Invalid robot type: {robot_type}")

    
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], action_horizon: int | None = None,noise: Tensor | None = None) -> Tensor:
        self.eval()

        if self.normalization:
            if 'franka' in self.robot_type:
                batch['observation.state'] = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
            elif 'dual_arm_tien_kung2' in self.robot_type:
                norm_state_arm  = self.normalize_inputs({"observation.state.arm_joint_position": batch['observation.state.arm_joint_position']})["observation.state.arm_joint_position"]
                norm_state_hand = self.normalize_inputs({"observation.state.hand_joint_position": batch['observation.state.hand_joint_position']})["observation.state.hand_joint_position"]
                batch['observation.state'] = torch.cat([norm_state_arm, norm_state_hand],dim=-1)
            else:
                raise ValueError(f"Invalid robot type: {self.robot_type} in normalization")
        images, img_masks = self.prepare_all_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        state = self.prepare_state(batch)

        if self.config.stage1_supervised:
            actions,latent_token_pred,latent_token_mask = self.stage2_model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            if self.config.stage2_latent_image_token_check:
                self.check_stage2_latent_image_token_reconstruction(batch,latent_token_pred)
            
            # 计算flops
            # flops_info = self.stage2_model.count_sample_actions_flops(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        else:
            actions = self.stage2_model.sample_actions_no_latent_token(images, img_masks, lang_tokens, lang_masks, state, noise=noise)


        original_action_dim = self.config.action_feature.shape[0]
        if action_horizon is None:
            actions = actions[:, :, :original_action_dim]
            action_horizon = actions.shape[1]
        else:
            action_output_size = actions.shape[1]
            action_horizon = min(action_horizon,action_output_size)
            actions = actions[:, :action_horizon, :original_action_dim]

        if self.normalization:
            if 'franka' in self.robot_type:
                actions = self.unnormalize_outputs({"action.arm_joint_position": actions})["action.arm_joint_position"]
            elif 'dual_arm_tien_kung2' in self.robot_type:
                actions_arm = self.unnormalize_outputs({"action.arm_joint_position": actions[:,:,:14]})["action.arm_joint_position"]
                actions_hand = self.unnormalize_outputs({"action.hand_joint_position": actions[:,:,14:]})["action.hand_joint_position"]
                actions = torch.cat([actions_arm,actions_hand],dim=-1)
            else:
                raise ValueError(f"Invalid robot type: {self.robot_type} in normalization")
        sub_actions = actions[:,::self.sampled_action_factor,:]
        if self.ensemble:
            self.action_history.append(sub_actions[0])
            max_history_len = math.ceil(self.action_horizon / self.sampled_action_factor)
            num_actions = len(self.action_history)
            num_actions = min(num_actions, max_history_len-self.action_execute_steps) # 确保取值正常
            # select the predicted action for the current step from the history of action chunk predictions

            curr_action_preds = torch.stack(
                [
                    pred_actions[i:i+self.action_execute_steps]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.action_history #生成倒叙 3,2,1,0
                    )
                ]
            ) # 把当前状态下的其余action的值都获取到

            # more recent predictions get exponentially *less* weight than older predictions
            weights = torch.exp(-self.exp_weight * torch.arange(num_actions)).to(curr_action_preds.device)
            weights = weights / weights.sum()
            
            # compute the weighted average across all predictions for this timestep
            pred_action = torch.sum(weights[:, None,None] * curr_action_preds, axis=0)
            pred_action = pred_action.unsqueeze(0) # bs *sqe * dim
        else: 
            pred_action = sub_actions # bs *sqe * dim
        self._action_queue.extend(pred_action.transpose(0, 1)) # bs * sqe * dim

        return self._action_queue


        
    def prepare_batch_train(self, batch):
        batch_train = {}
        # task
        batch_train["task"] = batch["task"]
        # image
        # 保留observation.rgb_images开头的字段
        for key in batch.keys():
            if key.startswith('observation.rgb_images'):
                batch_train[key] = batch[key]
            if key.startswith('observation.state'):
                batch_train[key] = batch[key]
            if key.startswith('action'):
                batch_train[key] = batch[key]

        # state 进一步合并
        if 'observation.state.arm_joint_position' in batch_train  \
        and 'observation.state.hand_joint_position' in batch_train:
            arm_joint = batch_train['observation.state.arm_joint_position'] 
            hand_joint = batch_train['observation.state.hand_joint_position'] 
            
            batch_size = arm_joint.shape[0]
            seq_len = arm_joint.shape[1]
            
            arm_flat = arm_joint.reshape(batch_size, seq_len, -1)  
            hand_flat = hand_joint.reshape(batch_size, seq_len, 1)
            
            batch_train['observation.state'] = torch.cat([arm_flat, hand_flat], dim=2)  
        
        # action
        if 'action.arm_joint_position' in batch_train  \
        and 'action.hand_joint_position' in batch_train:
            arm_joint = batch_train['action.arm_joint_position'] 
            hand_joint = batch_train['action.hand_joint_position'] 
            batch_size = arm_joint.shape[0]
            seq_len = arm_joint.shape[1]
            arm_flat = arm_joint.reshape(batch_size, seq_len, -1)  # Preserves batch and sequence dims
            hand_flat = hand_joint.reshape(batch_size, seq_len, 1)  # Makes it [40, 50, 1]
            
            batch_train['action'] = torch.cat([arm_flat, hand_flat], dim=2) 

        return batch_train
    def prepare_images_stage1(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            bsize = img.shape[0]
            device = img.device
            his_img_num = img.shape[1]
            for img_his_idx in range(his_img_num):
                his_img = img[:, img_his_idx]  # [B, C, H, W]
                mask = torch.tensor([not torch.all(his_img[i] == 0) for i in range(bsize)], 
                                  dtype=torch.bool, 
                                  device=device)
                
                if self.config.resize_imgs_with_padding is not None:
                    his_img = resize_with_pad(his_img, *self.config.resize_imgs_with_padding, pad_value=0)
                
                # Normalize from range [0,1] to [-1,1] as expected by siglip
                his_img = his_img * 2.0 - 1.0
                
                images.append(his_img)
                img_masks.append(mask)



        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(images[0]) * -1
            mask = torch.zeros_like(img_masks[0])
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
    
    def prepare_all_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        
        for key in self.config.image_features.keys():
            if key in present_img_keys:
                img = batch[key]
                bsize = img.shape[0]
                device = img.device
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                
                mask = torch.tensor([not torch.all(img[i] == 0) for i in range(bsize)], 
                        dtype=torch.bool, 
                        device=device) # mask = True 则是有效的
                # Normalize from range [0,1] to [-1,1] as expacted by siglip
                img = img * 2.0 - 1.0
                images.append(img)
                img_masks.append(mask)
            elif key in missing_img_keys:
                img = torch.ones_like(img) * -1
                mask = torch.zeros_like(mask)
                images.append(img)
                img_masks.append(mask)
            else:
                raise ValueError(f"Image feature {key} is not in the batch. (batch: {batch.keys()}) (image_features:{self.config.image_features})")
        return images, img_masks


    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            bsize = img.shape[0]
            device = img.device
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            
            mask = torch.tensor([not torch.all(img[i] == 0) for i in range(bsize)], 
                    dtype=torch.bool, 
                    device=device) # mask = True 则是有效的
            
            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0
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
            truncation=True,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks
    
    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state
    def prepare_action_mask(self, batch):
        if batch[ACTION].shape[-1] == self.config.max_action_dim:
            actions = batch[ACTION]
            ACTION_MASK = "action_mask"
            action_mask = batch[ACTION_MASK]
        else:
            actions,action_mask = pad_vector_mask(batch[ACTION], self.config.max_action_dim)
        return actions,action_mask
    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    # check the latent action whether is correct
    def check_stage2_latent_image_token_reconstruction(self, batch: dict[str, Tensor],pred_latent_action: Tensor) -> Tensor:
        
        self.eval()
        batch_train = batch
        latent_token_num = self.config.action_latent_token_num
        for key in list(batch_train.keys()):
            if key.startswith('observation.images') and len(batch_train[key].shape) == 5:
                batch_train[key] = batch_train[key][:, 0, :, :, :]  # 结果形状: [batch_size, channels, height, width]
                assert len(batch_train[key].shape) == 4 
            elif key.startswith('observation.state') and len(batch_train[key].shape) == 3:
                batch_train[key] = batch_train[key][:, 0, :]
                assert len(batch_train[key].shape) == 2
        images, img_masks = self.prepare_all_images(batch_train)
        img_masks_tensor = torch.cat(img_masks,dim=0)


        latent_motion_tokens_up = self.stage1_model.vq_up_resampler(pred_latent_action[:,latent_token_num:])
        for i,mask in enumerate(img_masks_tensor):
            if not mask:
                continue
            source_pixel_values = images[i]

            recons_pixel_values = self.stage1_model.decoder(
                cond_input=source_pixel_values,
                latent_motion_tokens=latent_motion_tokens_up[:,latent_token_num*i:latent_token_num*(i+1)]
            )

            # visualization 
            import cv2
            import numpy as np
            # source image 
            source_pixel_values_vis = (source_pixel_values+1.0) / 2.0
            image = source_pixel_values_vis[0].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)  
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  
            cv2.imwrite("./visualization/source_pixel_values{}.png".format(i), image)

            recons_pixel_values_vis = (recons_pixel_values+1.0) / 2.0
            recons_pixel_values_vis = torch.clamp(recons_pixel_values_vis, 0.0, 1.0)
            image = recons_pixel_values_vis[0].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)  
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  
            cv2.imwrite("./visualization/recons_pixel_values{}.png".format(i), image)

        return recons_pixel_values    

       



class Stage2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            freeze_language_encoder=self.config.freeze_language_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        paligemma_with_export_config.real_robot_dev = self.config.real_robot_dev
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)
        if self.config.stage1_supervised:
            # 根据模板确定图片数量
            image_num = 0
            for key, feature in self.config.input_features.items():
                if feature.type is FeatureType.VISUAL:
                    image_num = image_num + 1
   
            self.latent_queries = nn.Embedding(1, paligemma_with_export_config.paligemma_config.hidden_size)
            if self.config.action_branch and self.config.vision_branch:
                self.latent_action_queries = nn.Embedding(self.config.action_latent_token_num, paligemma_with_export_config.paligemma_config.hidden_size) 
                self.latent_action_queries.weight.data.fill_(0) 
                self.latent_img_queries = nn.Embedding(self.config.action_latent_token_num*image_num, paligemma_with_export_config.paligemma_config.hidden_size)
                self.latent_img_queries.weight.data.fill_(0) 
            elif self.config.action_branch and not self.config.vision_branch:     
                self.latent_action_queries = nn.Embedding(self.config.action_latent_token_num, paligemma_with_export_config.paligemma_config.hidden_size) 
                self.latent_action_queries.weight.data.fill_(0) 
            elif self.config.vision_branch and not self.config.action_branch:
                self.latent_img_queries = nn.Embedding(self.config.action_latent_token_num*image_num, paligemma_with_export_config.paligemma_config.hidden_size)
                self.latent_img_queries.weight.data.fill_(0) 
            else:
                raise ValueError(f"Invalid branch configuration: {self.config.action_branch} and {self.config.vision_branch}")

            self.latent_token_out_proj = nn.Linear(paligemma_with_export_config.paligemma_config.hidden_size, self.config.codebook_embed_dim)


        # resampler
        self.resampler = self.config.resampler
        if self.resampler:
            self.perceiver_resampler = PerceiverResampler(
                dim=self.config.resampler_dim,
                depth=self.config.resampler_depth,
                dim_head=self.config.resampler_dim_head,
                heads=self.config.resampler_heads,
                num_latents=self.config.resampler_num_latents,
                num_media_embeds=self.config.resampler_num_media_embeds)    
            
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

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, latent_token, latent_token_mask, state, actions,
        noise=None, time=None
    ) -> Tensor:
        if self.config.stage1_supervised:
            latent_token_num = latent_token.shape[1]
            action_latent_token_num = self.config.action_latent_token_num


        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions # ground truth noise


        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        ) # prefix_att_masks 参与cross transformer 计算 目前设置都为[0],说明不做mask
        # add learnable token
        if self.config.stage1_supervised:
            learnable_token, learnable_token_pad_mask, learnable_token_att_masks = self.embed_learnable_token(latent_token_mask)
            prefix_learnable_embs = torch.cat([prefix_embs, learnable_token], dim=1)
            prefix_learnable_pad_masks = torch.cat([prefix_pad_masks, learnable_token_pad_mask], dim=1)
            prefix_learnable_att_masks = torch.cat([prefix_att_masks, learnable_token_att_masks], dim=1)
        else:
            prefix_learnable_embs = prefix_embs
            prefix_learnable_pad_masks = prefix_pad_masks
            prefix_learnable_att_masks = prefix_att_masks

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_learnable_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_learnable_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_learnable_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )            
        loss_dict = {}
        # for latent token
        if self.config.stage1_supervised:
            latent_token_out = prefix_out[:,-latent_token_num:]
            latent_token_out = latent_token_out.to(dtype=torch.float32)
            latent_token_out = self.latent_token_out_proj(latent_token_out)
            latent_token_losses = F.mse_loss(latent_token[:,:latent_token_num], latent_token_out, reduction="none")
            if self.config.action_branch and self.config.vision_branch:
                loss_dict["latent_action_loss"] = latent_token_losses[:,:action_latent_token_num][latent_token_mask[:,:action_latent_token_num]].mean()
                loss_dict["image_latent_loss"] = latent_token_losses[:,action_latent_token_num:][latent_token_mask[:,action_latent_token_num:]].mean()
            elif self.config.action_branch and not self.config.vision_branch:
                loss_dict["latent_action_loss"] = latent_token_losses[:,:][latent_token_mask[:,:]].mean()
                loss_dict["image_latent_loss"] = torch.tensor(0.0).to(prefix_out.device)
            elif self.config.vision_branch and not self.config.action_branch:
                loss_dict["latent_action_loss"] = torch.tensor(0.0).to(prefix_out.device)
                loss_dict["image_latent_loss"] = latent_token_losses[:,:][latent_token_mask[:,:]].mean()
            else:
                raise ValueError(f"Invalid branch configuration: {self.config.action_branch} and {self.config.vision_branch}")
        else:
            loss_dict["latent_action_loss"] = torch.tensor(0.0).to(prefix_out.device)
            loss_dict["image_latent_loss"] = torch.tensor(0.0).to(prefix_out.device)

        # for action expert
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        actions_losses = F.mse_loss(u_t, v_t, reduction="none")


        return loss_dict,actions_losses
    
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
    
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
       
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        img_masks_tensor = torch.cat(img_masks,dim=0).unsqueeze(1).expand(-1,self.config.action_latent_token_num)
        img_masks_tensor = img_masks_tensor.reshape(1, -1)
        
        if self.config.action_branch and self.config.vision_branch:
            action_masks_tensor = torch.ones(bsize, self.config.action_latent_token_num, dtype=torch.bool, device=device)
            latent_token_mask = torch.cat([action_masks_tensor,img_masks_tensor],dim=1)
        elif self.config.action_branch:
            action_masks_tensor = torch.ones(bsize, self.config.action_latent_token_num, dtype=torch.bool, device=device)
            latent_token_mask = action_masks_tensor
        elif self.config.vision_branch:
            latent_token_mask = img_masks_tensor
        else:
            raise ValueError(f"Invalid branch configuration: {self.config.action_branch} and {self.config.vision_branch}")

        # latent_token_mask = torch.ones(bsize, self.config.action_latent_token_num*7, dtype=torch.bool, device=device)
        # latent_token_mask[:, 4*self.config.action_latent_token_num:5*self.config.action_latent_token_num] = False
        # latent_token_mask[:, 6*self.config.action_latent_token_num:7*self.config.action_latent_token_num] = False
        learnable_token, learnable_token_pad_mask, learnable_token_att_masks = self.embed_learnable_token(latent_token_mask)
        
        # for training free motion/vision ablation
        # learnable_token[:, :self.config.action_latent_token_num, :] = 0.0 # no motion
        # learnable_token[:, self.config.action_latent_token_num:, :] = 0.0 # no vision
        
        prefix_embs = torch.cat([prefix_embs, learnable_token], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, learnable_token_pad_mask], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, learnable_token_att_masks], dim=1)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        (prefix_out,suffix_out), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        latent_token_num = latent_token_mask.shape[1]
        latent_token_out = prefix_out[:,-latent_token_num:].to(dtype=torch.float32)
        latent_token_out = self.latent_token_out_proj(latent_token_out)
        # latent_token_out = latent_token_out[latent_token_mask].unsqueeze(0) 

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

        return x_t,latent_token_out,latent_token_mask
    
    def sample_actions_no_latent_token(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
       
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
    
    def count_sample_actions_flops(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        """
        计算sample_actions函数的FLOPs，包含flow matching去噪过程
        """
        flops = 0
        bsize = state.shape[0]
        
        # 获取配置参数
        vision_config = self.paligemma_with_expert.config.paligemma_config.vision_config
        text_config = self.paligemma_with_expert.config.paligemma_config.text_config
        gemma_config = self.paligemma_with_expert.config.gemma_expert_config
        
        # 基础配置
        vision_hidden_size = vision_config.hidden_size  # 1152
        text_hidden_size = text_config.hidden_size  # 2048
        expert_hidden_size = gemma_config.hidden_size  # 1024
        expert_intermediate_size = gemma_config.intermediate_size  # 4096
        expert_num_attention_heads = gemma_config.num_attention_heads  # 8
        expert_num_layers = gemma_config.num_hidden_layers  # 18
        
        # 1. 初始化阶段的FLOPs
        # embed_prefix的FLOPs (只需计算一次)
        prefix_flops = 0
        for img in images:
            img_size = img.shape[-1]  # 224
            num_patches = (img_size // vision_config.patch_size) ** 2
            # Vision encoder FLOPs
            prefix_flops += bsize * (
                # Patch embedding
                img_size * img_size * 3 * vision_hidden_size +
                # Vision transformer layers
                vision_config.num_hidden_layers * num_patches * (
                    3 * vision_hidden_size * vision_hidden_size +
                    vision_config.num_attention_heads * num_patches * (vision_hidden_size // vision_config.num_attention_heads)
                )
            )
        
        # Language encoder FLOPs (只需计算一次)
        seq_length = lang_tokens.shape[1]
        prefix_flops += bsize * seq_length * text_config.num_hidden_layers * (
            3 * text_hidden_size * text_hidden_size +
            text_config.num_attention_heads * seq_length * (text_hidden_size // text_config.num_attention_heads)
        )
        
        flops += prefix_flops
        
        # 2. Flow Matching去噪过程的FLOPs
        # 计算每个去噪步骤的FLOPs
        denoising_steps = self.config.num_steps  # 通常是10步
        per_step_flops = 0
        
        # 每个去噪步骤中的操作
        action_dim = self.config.max_action_dim
        proj_width = self.config.proj_width
        
        # denoise_step中的操作
        per_step_flops += bsize * (
            # state projection
            state.shape[-1] * proj_width +
            # action projection
            action_dim * proj_width +
            # time embedding and MLP
            2 * proj_width * proj_width +
            # action_time_mlp operations
            2 * proj_width * proj_width
        )
        
        # Expert model forward pass
        num_patches = sum((img.shape[-1] // vision_config.patch_size) ** 2 for img in images)
        total_seq_length = num_patches + seq_length + self.config.n_action_steps
        
        per_step_flops += bsize * total_seq_length * expert_num_layers * (
            # Multi-head attention
            3 * expert_hidden_size * expert_hidden_size +
            expert_num_attention_heads * total_seq_length * (expert_hidden_size // expert_num_attention_heads) +
            # FFN
            expert_hidden_size * expert_intermediate_size * 2
        )
        
        # Euler step
        per_step_flops += bsize * action_dim * 2  # x_t += dt * v_t
        
        # 总的去噪FLOPs
        denoising_flops = per_step_flops * denoising_steps
        flops += denoising_flops
        
        results = {
            'total_flops': flops,
            'gflops': flops / 1e9,
            'flops_per_sample': flops / bsize,
            'breakdown': {
                'initialization': prefix_flops,
                'denoising': denoising_flops,
                'per_denoising_step': per_step_flops,
                'details': {
                    'vision_encoder': prefix_flops * 0.4,
                    'text_encoder': prefix_flops * 0.2,
                    'expert_model_per_step': per_step_flops * 0.7,
                    'projections_and_mlp_per_step': per_step_flops * 0.3
                }
            }
        }
        
        # 打印详细信息
        print(f"\nDetailed FLOPs Analysis:")
        print(f"Total GFLOPs: {results['gflops']:.2f}")
        print(f"Initialization GFLOPs: {results['breakdown']['initialization']/1e9:.2f}")
        print(f"Total Denoising GFLOPs: {results['breakdown']['denoising']/1e9:.2f}")
        print(f"Per Denoising Step GFLOPs: {results['breakdown']['per_denoising_step']/1e9:.2f}")
        print(f"\nDenoising steps: {denoising_steps}")
        
        return results
    
    
    def embed_learnable_token(self, latent_token_mask):
        bsize = latent_token_mask.shape[0]
        latent_queries = self.latent_queries.weight  
        if self.config.action_branch and self.config.vision_branch:
            latent_action_queries = self.latent_action_queries.weight + latent_queries 
            latent_img_queries = self.latent_img_queries.weight + latent_queries 
            learnable_token = torch.cat([latent_action_queries,latent_img_queries],dim=0)
        elif self.config.action_branch and not self.config.vision_branch:
            latent_action_queries = self.latent_action_queries.weight + latent_queries 
            learnable_token = latent_action_queries
        elif self.config.vision_branch and not self.config.action_branch:
            latent_img_queries = self.latent_img_queries.weight + latent_queries 
            learnable_token = latent_img_queries
        else:
            raise ValueError(f"Invalid branch configuration: {self.config.action_branch} and {self.config.vision_branch}")
        

        learnable_token = learnable_token.expand(bsize, -1, -1)
        _, learnable_token_num = learnable_token.shape[:2]
        learnable_token_pad_mask = latent_token_mask[:, :learnable_token_num]
        learnable_token_att_masks = [1] + ([0] * (learnable_token_num - 1))
        learnable_token_att_masks = torch.tensor(learnable_token_att_masks, dtype=torch.bool, device=latent_token_mask.device)
        learnable_token_att_masks = learnable_token_att_masks[None, :].expand(bsize, len(learnable_token_att_masks))
        return learnable_token, learnable_token_pad_mask, learnable_token_att_masks
    
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
        device = lang_tokens.device
        device_bsize = lang_tokens.shape[0]
        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):

            img_emb = self.paligemma_with_expert.embed_image(img)
            

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            if self.resampler:
                img_emb = self.perceiver_resampler(img_emb)

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
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []
        bsize = noisy_actions.shape[0]
        dtype = noisy_actions.dtype
        device = noisy_actions.device

        # Embed state
        state_emb = self.state_proj(state)
        if len(state_emb.shape) == 2:
            embs.append(state_emb[:, None, :])
        else:
            embs.append(state_emb)
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
        # update only action expert ,language image and state token attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks


    
    def _load_pretrained_weights(self, pretrained_path):
        import os
        model_file = os.path.join(pretrained_path, "model.safetensors")
        if os.path.exists(model_file):
            try:
                from safetensors.torch import load_file
                latent_action_weights = load_file(model_file)
                
                # get the state dictionary of the current LAM model
                latent_action_state_dict = self.state_dict()
                # create a new state dictionary for updating
                new_state_dict = {}
                
                # align keys and update weights
                for key in latent_action_state_dict.keys():
                    # build full key
                    full_key = f'model.{key}'
                    if full_key in latent_action_weights:
                        # add the pretrained weights to the new state dictionary
                        new_state_dict[key] = latent_action_weights[full_key]
                    else:
                        # keep the original weights
                        new_state_dict[key] = latent_action_state_dict[key]
                        
                # check which keys are missing in the pretrained weights
                missing_keys = [k for k in latent_action_state_dict.keys() if f'model.{k}' not in latent_action_weights]

                # print missing keys
                if missing_keys:
                    print(f"Missing keys in latent action pretrained weights: {missing_keys}")
                    print(f"Total missing keys: {len(missing_keys)}/{len(latent_action_state_dict)}")
                else:
                    print("All latent action keys found in pretrained weights!")
                
                # load weights - use the new state dictionary instead of the original latent_action_weights
                self.load_state_dict(new_state_dict, strict=False)
                print(f"Successfully loaded latent action weights from {pretrained_path}")
            except Exception as e:
                print(f"Error loading latent action weights: {e}")
        else:
            print(f"latent action pretrained model not found at {model_file}")




