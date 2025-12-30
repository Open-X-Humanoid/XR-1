import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.functional import gaussian_blur
from torch import Tensor, nn

import numpy as np
import math

from lerobot.common.policies.normalize import Normalize,Unnormalize
from lerobot.common.policies.xr1.configuration_xr1_stage1 import Xr1Stage1Config
from lerobot.common.policies.pretrained import PreTrainedPolicy
from typing import Optional
from transformers.models.auto import AutoModel
from transformers.models.vit.modeling_vit import ViTConfig

import lpips,random
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.xr1.cross_stage_function.cross_stage import motion_token_action_decoder_expert
from lerobot.common.policies.xr1.cross_stage_function.cross_stage_module import LMDViTModel,MFormer,VectorQuantizer2,ResidualTemporalBlock,LatentMotionDecoder
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

def pad_vector_mask(vector, new_dim):
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

    # create mask: 1 for original value, 0 for padding value
    mask = torch.zeros(*shape, dtype=torch.bool, device=vector.device)
    mask[..., :current_dim] = True

    return new_vector, mask

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




class Xr1Stage1Policy(PreTrainedPolicy):

    config_class = Xr1Stage1Config
    name = "xr1_stage1"

    def __init__(
        self,
        config: Xr1Stage1Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        hidden_state_decoder=None,
        commit_loss_w=1.,
        recon_loss_w=1.,
        recon_action_loss_w=1.,
        recon_hidden_loss_w=1.,
        perceptual_loss_w=1.,
        use_abs_recons_loss=False,
        kl_loss_w=1.
    ):

        super().__init__(config)
        # config.validate_features()
        self.config = config
        self.random_image_flag = False 
        
        # preprocess
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
            )  

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_action_loss_w = recon_action_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.kl_loss_w = kl_loss_w * 10
        self.use_abs_recons_loss = use_abs_recons_loss


        self.vision_branch = self.config.vision_branch
        if self.vision_branch:
            # pretrained vision encoder siglip same from pi0
            self.vision_tower = AutoModel.from_pretrained('../pretrained/siglip-so400m-patch14-224').vision_model
            if self.config.freeze_vision_encoder:
                self.vision_tower.eval() # freeze vision encoder
                for params in self.vision_tower.parameters():
                    params.requires_grad = False
            
            # mformer
            self.motion_token_num = self.config.action_latent_token_num
            self.mformer = MFormer(
                    config=ViTConfig(
                    query_num=self.motion_token_num,
                    input_hidden_size=1152,
                    num_patches=256,  # include the [CLS] token
                    attention_probs_dropout_prob=0.0,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.0,
                    hidden_size=768,
                    initializer_range=0.02,
                    intermediate_size=3072,
                    layer_norm_eps=1e-12,
                    model_type="vit",
                    num_attention_heads=12,
                    num_hidden_layers=4,
                    qkv_bias=True,
                    action_token_num=0
                )
            )
            # codebook
            mformer_hidden_size = self.config.mformer_hidden_size
            decoder_hidden_size = self.config.decoder_hidden_size
            codebook_embed_dim = self.config.codebook_embed_dim
            codebook_k_size = self.config.codebook_k_size

            self.vq_down_resampler = nn.Sequential(
                nn.Linear(mformer_hidden_size, decoder_hidden_size),
                nn.Tanh(),
                nn.Linear(decoder_hidden_size, codebook_embed_dim)
            )
        
            self.vq_up_resampler = nn.Sequential(
                nn.Linear(codebook_embed_dim, codebook_embed_dim),
                nn.Tanh(),
                nn.Linear(codebook_embed_dim, decoder_hidden_size)
            )


            self.decoder = LatentMotionDecoder(
                config=ViTConfig(
                    query_num=self.motion_token_num,
                    attention_probs_dropout_prob=0.0,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.0,
                    hidden_size=768,
                    image_size=224,
                    initializer_range=0.02,
                    intermediate_size=3072,
                    layer_norm_eps=1e-12,
                    model_type="vit",
                    num_attention_heads=12,
                    num_channels=3,
                    num_hidden_layers=12,
                    patch_size=16,
                    qkv_bias=True,
                    encoder_stride=16,
                    num_patches=196,
                )
            )
            self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()


        self.action_branch = self.config.action_branch
        if self.action_branch:
            encoder_dim = 512
            encoder_heads = 8
            attn_pdrop = 0.1
            encoder_layers = 8
            downsample_factor = 4
            use_causal_encoder = True
            self.use_causal_encoder = use_causal_encoder
            assert int(np.log2(downsample_factor)) == np.log2(downsample_factor), 'downsample_factor must be a power of 2'
            strides = [2] * int(np.log2(downsample_factor)) + [1]
            kernel_sizes = [5] + [3] * int(np.log2(downsample_factor))
            self.action_proj = nn.Linear(self.config.max_action_dim, encoder_dim)
            self.conv_block = ResidualTemporalBlock(
                encoder_dim, encoder_dim, kernel_size=kernel_sizes, 
                stride=strides, causal=use_causal_encoder)
            self.action_encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, 
                                                        nhead=encoder_heads, 
                                                        dim_feedforward=4*encoder_dim, 
                                                        dropout=attn_pdrop, 
                                                        activation='gelu', 
                                                        batch_first=True, 
                                                        norm_first=True)
            self.action_encoder =  nn.TransformerEncoder(self.action_encoder_layer, 
                                        num_layers=encoder_layers,
                                        enable_nested_tensor=False)
            
            self.add_positional_emb = Summer(PositionalEncoding1D(encoder_dim))
            self.action_encoder_proj = nn.Linear(encoder_dim, self.config.codebook_embed_dim)
            
            

            # total_params = sum(p.numel() for p in self.parameters()) - total_params_last
            # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) - trainable_params_last
            # print(f"Latent action Encoder Total parameters: {total_params/1e6:.2f}M")
            # print(f"Latent action Encoder Trainable parameters: {trainable_params/1e6:.2f}M")
            # total_params_last = sum(p.numel() for p in self.parameters())
            # trainable_params_last = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            
            
            
            # action_decoder
            self.action_decoder = motion_token_action_decoder_expert(config) 
            # total_params = sum(p.numel() for p in self.action_decoder.motion_token_expert.parameters())
            # trainable_params = sum(p.numel() for p in self.action_decoder.motion_token_expert.parameters() if p.requires_grad) 
            # print(f"Latent action Decoder Total parameters: {total_params/1e6:.2f}M")
            # print(f"Latent action Decoder Trainable parameters: {trainable_params/1e6:.2f}M")

            # total_params = sum(p.numel() for p in self.parameters()) - total_params_last
            # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) - trainable_params_last 
            # print(f"Latent action Decoder Total parameters: {total_params/1e6:.2f}M")
            # print(f"Latent action Decoder Trainable parameters: {trainable_params/1e6:.2f}M")
            # total_params_last = sum(p.numel() for p in self.parameters())
            # trainable_params_last = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if self.action_branch and not self.vision_branch:
            self.action_vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
        elif self.vision_branch and not self.action_branch:
            self.vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
        elif self.action_branch and self.vision_branch:
            if self.config.combine_codebook:
                self.combine_vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size, e_dim=self.config.codebook_embed_dim, beta=0.25,remap=None,sane_index_shape=True)
            else:
                self.vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
                self.action_vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
        else:
            raise ValueError(f"Invalid branch configuration: {self.action_branch} and {self.vision_branch}")

        if self.config.stage1_pretrained_path is not None:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        import os
        if hasattr(self.config, "stage1_pretrained_path") and self.config.stage1_pretrained_path:
            stage1_pretrained_path = self.config.stage1_pretrained_path
            model_file = os.path.join(stage1_pretrained_path, "model.safetensors")
            if os.path.exists(model_file):
                try:
                    import os
                    from safetensors.torch import load_file
                    stage1_pretrained_weights = load_file(model_file)
                    stage1_sup_state_dict = self.state_dict()
                    missing_keys = [k for k in stage1_sup_state_dict.keys() if k not in stage1_pretrained_weights.keys()]
                    
                    # print missing keys
                    if missing_keys:
                        print(f"Missing keys in stage1  pretrained weights: {missing_keys}")
                        print(f"Total missing keys: {len(missing_keys)}/{len(stage1_pretrained_weights.keys())}")
                    else:
                        print("All stage1 keys found in pretrained weights!")
                    # filter out keys with shape mismatch
                    filtered_weights = {}
                    skipped_keys = []
                    for k, v in stage1_pretrained_weights.items():
                        if k in stage1_sup_state_dict and stage1_sup_state_dict[k].shape == v.shape:
                            filtered_weights[k] = v
                        else:
                            print(f"Skipped key: {k} due to shape mismatch: {stage1_sup_state_dict[k].shape} != {v.shape}")
                            skipped_keys.append(k)

                    if skipped_keys:
                        print(f"Skipped keys in stage1 pretrained weights due to shape mismatch: {skipped_keys}")
                    # load weights
                    self.load_state_dict(filtered_weights, strict=False)
                    print(f"Successfully loaded stage1 weights from {stage1_pretrained_path}")
                except Exception as e:
                    print(f"Error loading stage1 weights: {e}")
            else:
                print(f"stage1 pretrained model not found at {model_file}")
        # self.to_bfloat16()
    def action_encode(self, act, obs_emb=None):
        x = self.action_proj(act)
        x = self.conv_block(x)
        B, H, D = x.shape
        
        if obs_emb is not None:
            x = torch.cat([obs_emb, x], dim=1)
        x = self.add_positional_emb(x)

        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.action_encoder(x, mask=mask, is_causal=True)
        else:
            x = self.action_encoder(x)

        x = x[:, -H:]

        return x

    def to_bfloat16(self):
        for name,param in self.named_parameters():
            param.data = param.data.to(dtype=torch.bfloat16)

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
    
    def prepare_batch_train(self, batch):
        batch_train = {}
        # task
        batch_train["task"] = batch["task"]
        # image
        # keep fields starting with observation.rgb_images
        for key in batch.keys():
            if key.startswith('observation.rgb_images'):
                batch_train[key] = batch[key]
            if key.startswith('observation.state'):
                batch_train[key] = batch[key]
            if key.startswith('action'):
                batch_train[key] = batch[key]

        # state further merge
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
            arm_flat = arm_joint.reshape(batch_size, seq_len, -1)  # preserve batch and sequence dims
            hand_flat = hand_joint.reshape(batch_size, seq_len, 1)  # Makes it [40, 50, 1]
            
            batch_train['action'] = torch.cat([arm_flat, hand_flat], dim=2) 

        return batch_train
    def kl_loss(self, action_quant, quant):
        # softmax the last dimension to ensure it is a probability distribution
        action_quant = F.softmax(action_quant, dim=-1)
        quant = F.softmax(quant, dim=-1)
        
        # calculate KL divergence
        kl_div = F.kl_div(quant.log(), action_quant, reduction='batchmean')
        
        return kl_div
    
    def action_vision_dual_branch(self, batch_train, no_ego4d_mask, noise=None, time=None):
        state = self.prepare_state(batch_train)
        actions,batch_train['action_mask'] = self.prepare_action_mask(batch_train) 
        actions_is_pad = batch_train.get("action_is_pad")


        emb_actions = self.action_encode(actions)
        emb_actions = self.action_encoder_proj(emb_actions)
        if self.config.combine_codebook:
            action_quant, action_indices, action_commit_loss = self.combine_vector_quantizer(emb_actions)
        else:
            action_quant, action_indices, action_commit_loss = self.action_vector_quantizer(emb_actions)
        if action_quant.shape[1] != self.config.action_latent_token_num:
            action_quant = F.interpolate(
                        action_quant.transpose(1, 2),  # (bs, dim, sqe)
                        size=self.config.action_latent_token_num, 
                        mode='linear'
                    ).transpose(1, 2)  # (bs, action_latent_token_num, dim)
        # action decoder
        recons_actions_loss = self.action_decoder.forward(action_quant, state[:,:1,:], actions, noise, time)
        
        total_loss = 0.0
        all_outputs = {
            "loss": 0.0,
            "commit_loss": 0.0,
            "recons_image_loss": 0.0,
            "recons_hidden_loss": 0.0,
            "perceptual_loss": 0.0,
            "active_code_num": 0.0,
            # action
            "action_commit_loss": 0.0,
            "recons_actions_loss": 0.0,
            "action_active_code_num": 0.0,
            "kl_loss": 0.0,
            # "siml2_loss": 0.0

        }
        pair_count = 0 

        images, img_masks, images_wrist, img_masks_wrist = self.prepare_images(batch_train) 

        # wrist 同一个
        for i in range(0, len(images_wrist), 2):
            if i + 1 < len(images_wrist):
                image_mask = img_masks_wrist[i]
                if not image_mask.any():  # if mask is all False
                    continue  # skip this iteration
                cond_hidden_states = self.vision_tower(images_wrist[i]).last_hidden_state
                target_hidden_states = self.vision_tower(images_wrist[i + 1]).last_hidden_state

                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                if self.config.combine_codebook:
                    quant, indices, commit_loss = self.combine_vector_quantizer(latent_motion_tokens_down)
                else:
                    quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                commit_loss = commit_loss.mean()
                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                # only use the corresponding image
                source_pixel_values = images_wrist[i]

                recons_pixel_values = self.decoder(
                    cond_input=source_pixel_values,
                    latent_motion_tokens=latent_motion_tokens_up
                )
                # Compute loss
                outputs = {
                    "loss": torch.zeros_like(commit_loss),
                    "commit_loss": commit_loss,
                    "recons_image_loss": torch.zeros_like(commit_loss),
                    "recons_hidden_loss": torch.zeros_like(commit_loss),
                    "perceptual_loss": torch.zeros_like(commit_loss),
                    "recons_actions_loss": torch.zeros_like(commit_loss),
                    "action_commit_loss": torch.zeros_like(commit_loss),
                    "kl_loss": torch.zeros_like(commit_loss),
                    # "siml2_loss": torch.zeros_like(commit_loss),
                }
                # similarity
                image_ego4d_mask_combine = image_mask & no_ego4d_mask 
                if self.config.combine_codebook:
                    kl_loss = self.kl_loss(emb_actions[image_ego4d_mask_combine], latent_motion_tokens_down[image_ego4d_mask_combine]) if image_ego4d_mask_combine.any() else torch.zeros_like(commit_loss)
                else: 
                    kl_loss = self.kl_loss(action_quant[image_ego4d_mask_combine], quant[image_ego4d_mask_combine]) if image_ego4d_mask_combine.any() else torch.zeros_like(commit_loss)
                
                outputs["kl_loss"] = kl_loss
                # siml2_loss = F.mse_loss(action_quant[image_ego4d_mask_combine], quant[image_ego4d_mask_combine]) if image_ego4d_mask_combine.any() else torch.zeros_like(commit_loss)
                # outputs["siml2_loss"] = siml2_loss
                
                # action only loss
                if actions_is_pad is not None:
                    in_episode_bound = ~actions_is_pad
                    action_mask = in_episode_bound.unsqueeze(-1) * batch_train['action_mask']
                recons_actions_loss_mean = recons_actions_loss[action_mask].mean() if action_mask.any() else torch.zeros_like(commit_loss)
                action_commit_loss_mean = action_commit_loss[no_ego4d_mask].mean() if no_ego4d_mask.any() else torch.zeros_like(commit_loss)
                outputs["recons_actions_loss"] = recons_actions_loss_mean
                outputs["action_commit_loss"] = action_commit_loss_mean
                ###############image only loss############
                target_pixel_values = images_wrist[i + 1]
                recons_image_loss = F.mse_loss(target_pixel_values[image_mask], recons_pixel_values[image_mask])
                outputs["recons_image_loss"] = recons_image_loss

                if self.perceptual_loss_w > 0:
                    with torch.no_grad():
                        perceptual_loss = self.loss_fn_lpips.forward(
                            target_pixel_values, recons_pixel_values, normalize=True)
                        perceptual_loss = perceptual_loss[image_mask].mean()
                else:
                    perceptual_loss = torch.zeros_like(recons_image_loss)
                outputs["perceptual_loss"] = perceptual_loss

                pair_loss =  self.commit_loss_w * outputs["commit_loss"]+ self.commit_loss_w * outputs["action_commit_loss"] \
                                + self.recon_loss_w * outputs["recons_image_loss"] + self.recon_action_loss_w * outputs["recons_actions_loss"] \
                                + self.perceptual_loss_w * outputs["perceptual_loss"] \
                                + self.kl_loss_w * outputs["kl_loss"] 
                                # + outputs["siml2_loss"]
                

                outputs["loss"] = pair_loss

                # 累加每个配对的损失
                total_loss += pair_loss
                pair_count += 1
                
                # 累积各种损失度量
                for key in all_outputs:
                    if key in outputs:
                        all_outputs[key] += outputs[key].detach() if torch.is_tensor(outputs[key]) else outputs[key]
                
                # 计算活跃码本数量
                active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(pair_loss.device)
                active_code_num_action = torch.tensor(torch.unique(action_indices).shape[0]).float().to(pair_loss.device)
                all_outputs["active_code_num"] += active_code_num
                all_outputs["action_active_code_num"] += active_code_num_action

        cond_images_indices = list(range(0, len(images), 2))
        # invariance of motion tokens from different angles
        for i in range(0, len(images), 2):
            if i + 1 < len(images):
                image_mask = img_masks[i]
                if self.random_image_flag:
                    random_cond_idx = random.choice(cond_images_indices)
                    image_mask_random = img_masks[random_cond_idx]
                    if not image_mask.any() and not image_mask_random.any():  # if mask is all False
                        continue  # skip this iteration
                    image_mask_combine = image_mask_random & image_mask
                    image_valid = images[i]
                    image_valid_next = images[i + 1]
                    source_pixel_values = images[random_cond_idx]
                    target_pixel_values = images[random_cond_idx+1]
                else:
                    if not image_mask.any(): 
                        continue 
                    # only use the corresponding image
                    image_mask_combine = image_mask
                    image_valid = images[i]
                    image_valid_next = images[i + 1]
                    source_pixel_values = image_valid
                    target_pixel_values = image_valid_next

                cond_hidden_states = self.vision_tower(image_valid).last_hidden_state
                target_hidden_states = self.vision_tower(image_valid_next).last_hidden_state
                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                if self.config.combine_codebook:
                    quant, indices, commit_loss = self.combine_vector_quantizer(latent_motion_tokens_down)
                else:
                    quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                commit_loss = commit_loss.mean()

                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                recons_pixel_values = self.decoder(
                    cond_input=source_pixel_values,
                    latent_motion_tokens=latent_motion_tokens_up
                )
                # Compute loss
                outputs = {
                    "loss": torch.zeros_like(commit_loss),
                    "commit_loss": commit_loss,
                    "recons_image_loss": torch.zeros_like(commit_loss),
                    "recons_hidden_loss": torch.zeros_like(commit_loss),
                    "perceptual_loss": torch.zeros_like(commit_loss),
                    "recons_actions_loss": torch.zeros_like(commit_loss),
                    "action_commit_loss": torch.zeros_like(commit_loss),
                    "kl_loss": torch.zeros_like(commit_loss),
                    # "siml2_loss": torch.zeros_like(commit_loss),
                }
                # similarity
                image_ego4d_mask_combine = image_mask_combine & no_ego4d_mask
                kl_loss = self.kl_loss(action_quant[image_ego4d_mask_combine], quant[image_ego4d_mask_combine]) if image_ego4d_mask_combine.any() else torch.zeros_like(commit_loss)
                outputs["kl_loss"] = kl_loss
                # siml2_loss = F.mse_loss(action_quant[image_ego4d_mask_combine], quant[image_ego4d_mask_combine]) if image_ego4d_mask_combine.any() else torch.zeros_like(commit_loss)
                # outputs["siml2_loss"] = siml2_loss
                
                # action only loss
                if actions_is_pad is not None:
                    in_episode_bound = ~actions_is_pad
                    action_mask = in_episode_bound.unsqueeze(-1) * batch_train['action_mask']
                recons_actions_loss_mean = recons_actions_loss[action_mask].mean() if action_mask.any() else torch.zeros_like(commit_loss)
                action_commit_loss_mean = action_commit_loss[no_ego4d_mask].mean() if no_ego4d_mask.any() else torch.zeros_like(commit_loss)
                outputs["recons_actions_loss"] = recons_actions_loss_mean
                outputs["action_commit_loss"] = action_commit_loss_mean
                

                # get motion region masks
                motion_masks = self.get_motion_regions_blur(source_pixel_values[image_mask_combine],target_pixel_values[image_mask_combine])
                pixel_loss = F.mse_loss(target_pixel_values[image_mask_combine] * motion_masks, recons_pixel_values[image_mask_combine] * motion_masks, reduction='none')
                valid_pixels = torch.sum(motion_masks > 0).float()
                motion_recons_image_loss = torch.sum(pixel_loss) / valid_pixels if valid_pixels > 0 else 0.0

                global_recons_image_loss = F.mse_loss(target_pixel_values[image_mask_combine], recons_pixel_values[image_mask_combine])
                lambda_motion = 0.9  # weight of motion region loss
                recons_image_loss = lambda_motion * motion_recons_image_loss + (1 - lambda_motion) * global_recons_image_loss

                outputs["recons_image_loss"] = recons_image_loss

                if self.perceptual_loss_w > 0:
                    with torch.no_grad():
                        perceptual_loss = self.loss_fn_lpips.forward(
                            target_pixel_values, recons_pixel_values, normalize=True)
                        perceptual_loss = perceptual_loss[image_mask_combine].mean()
                else:
                    perceptual_loss = torch.zeros_like(recons_image_loss)
                outputs["perceptual_loss"] = perceptual_loss

                pair_loss =  self.commit_loss_w * outputs["commit_loss"] + self.commit_loss_w * outputs["action_commit_loss"] \
                                + self.recon_loss_w * outputs["recons_image_loss"] + self.recon_action_loss_w * outputs["recons_actions_loss"] \
                                + self.perceptual_loss_w * outputs["perceptual_loss"] \
                                + self.kl_loss_w * outputs["kl_loss"] 
                                # + outputs["siml2_loss"]
                
                outputs["loss"] = pair_loss

                # accumulate loss of each pair
                total_loss += pair_loss
                pair_count += 1
                
                # accumulate various loss metrics
                for key in all_outputs:
                    if key in outputs:
                        all_outputs[key] += outputs[key].detach() if torch.is_tensor(outputs[key]) else outputs[key]
                
                # calculate active codebook number
                active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(pair_loss.device)
                active_code_num_action = torch.tensor(torch.unique(action_indices).shape[0]).float().to(pair_loss.device)
                all_outputs["active_code_num"] += active_code_num
                all_outputs["action_active_code_num"] += active_code_num_action
                # if there is a pair processed, calculate average loss
        if pair_count > 0:
            total_loss = total_loss / pair_count
            # calculate average of all metrics
            for key in all_outputs:
                all_outputs[key] = (all_outputs[key] / pair_count).detach().cpu().item()  
        
        return total_loss, all_outputs

    def action_only_branch(self, batch_train, no_ego4d_mask, noise=None, time=None):
        state = self.prepare_state(batch_train)
        actions,batch_train['action_mask'] = self.prepare_action_mask(batch_train) 
        actions_is_pad = batch_train.get("action_is_pad")

        emb_actions = self.action_encode(actions)
        emb_actions = self.action_encoder_proj(emb_actions)
        action_quant, action_indices, action_commit_loss = self.action_vector_quantizer(emb_actions)

        if action_quant.shape[1] != self.config.action_latent_token_num:
            action_quant = F.interpolate(
                        action_quant.transpose(1, 2),  # (bs, dim, sqe)
                        size=self.config.action_latent_token_num, 
                        mode='linear'
                    ).transpose(1, 2)  # (bs, action_latent_token_num, dim)
        # action decoder
        recons_actions_loss = self.action_decoder.forward(action_quant, state[:,:1,:], actions, noise, time)
        outputs = {
            "loss": torch.zeros_like(action_commit_loss),
            "recons_actions_loss": torch.zeros_like(action_commit_loss),
            "action_commit_loss": torch.zeros_like(action_commit_loss),

        }
        # action only loss
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_mask = in_episode_bound.unsqueeze(-1) * batch_train['action_mask']
        recons_actions_loss_mean = recons_actions_loss[action_mask].mean() if action_mask.any() else torch.zeros_like(action_commit_loss)
        action_commit_loss_mean = action_commit_loss[no_ego4d_mask].mean() if no_ego4d_mask.any() else torch.zeros_like(action_commit_loss)
        outputs["recons_actions_loss"] = recons_actions_loss_mean
        outputs["action_commit_loss"] = action_commit_loss_mean
        pair_loss =  self.commit_loss_w * outputs["action_commit_loss"] \
                    + self.recon_action_loss_w * outputs["recons_actions_loss"] 
        outputs["loss"] = pair_loss

        total_loss = 0.0
        all_outputs = {
            "loss": 0.0,
            "action_commit_loss": 0.0,
            "recons_actions_loss": 0.0,
            "action_active_code_num": 0.0,
        }

        # accumulate loss of each pair
        total_loss = pair_loss
        for key in all_outputs:
            if key in outputs:
                all_outputs[key] += outputs[key].detach() if torch.is_tensor(outputs[key]) else outputs[key]
        
        active_code_num_action = torch.tensor(torch.unique(action_indices).shape[0]).float().to(pair_loss.device)
        all_outputs["action_active_code_num"] += active_code_num_action

        for key in all_outputs:
            all_outputs[key] = all_outputs[key].detach().cpu().item() 

        return total_loss, all_outputs

    def vision_only_branch(self, batch_train, no_ego4d_mask, noise=None, time=None):

        total_loss = 0.0
        all_outputs = {
            "loss": 0.0,
            "commit_loss": 0.0,
            "recons_image_loss": 0.0,
            "recons_hidden_loss": 0.0,
            "perceptual_loss": 0.0,
            "active_code_num": 0.0,
        }
        pair_count = 0 


        images, img_masks, images_wrist, img_masks_wrist = self.prepare_images(batch_train) 
        # wrist 同一个
        for i in range(0, len(images_wrist), 2):
            if i + 1 < len(images_wrist):
                image_mask = img_masks_wrist[i]
                if not image_mask.any():  # if mask is all False
                    continue  # 跳过这次迭代
                cond_hidden_states = self.vision_tower(images_wrist[i]).last_hidden_state
                target_hidden_states = self.vision_tower(images_wrist[i + 1]).last_hidden_state

                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                commit_loss = commit_loss.mean()
                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                # 只用对应的图像
                source_pixel_values = images_wrist[i]

                recons_pixel_values = self.decoder(
                    cond_input=source_pixel_values,
                    latent_motion_tokens=latent_motion_tokens_up
                )
                # Compute loss
                outputs = {
                    "loss": torch.zeros_like(commit_loss),
                    "commit_loss": commit_loss,
                    "recons_image_loss": torch.zeros_like(commit_loss),
                    "recons_hidden_loss": torch.zeros_like(commit_loss),
                    "perceptual_loss": torch.zeros_like(commit_loss),
                }

                ###############image only loss############
                target_pixel_values = images_wrist[i + 1]
                recons_image_loss = F.mse_loss(target_pixel_values[image_mask], recons_pixel_values[image_mask])
                outputs["recons_image_loss"] = recons_image_loss

                if self.perceptual_loss_w > 0:
                    with torch.no_grad():
                        perceptual_loss = self.loss_fn_lpips.forward(
                            target_pixel_values, recons_pixel_values, normalize=True)
                        perceptual_loss = perceptual_loss[image_mask].mean()
                else:
                    perceptual_loss = torch.zeros_like(recons_image_loss)
                outputs["perceptual_loss"] = perceptual_loss

                pair_loss =  self.commit_loss_w * outputs["commit_loss"] \
                                + self.recon_loss_w * outputs["recons_image_loss"] \
                                + self.perceptual_loss_w * outputs["perceptual_loss"] 
                
                outputs["loss"] = pair_loss
                # 累加每个配对的损失
                total_loss += pair_loss
                pair_count += 1
                
                # 累积各种损失度量
                for key in all_outputs:
                    if key in outputs:
                        all_outputs[key] += outputs[key].detach() if torch.is_tensor(outputs[key]) else outputs[key]
                
                # 计算活跃码本数量
                active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(pair_loss.device)
                all_outputs["active_code_num"] += active_code_num

        cond_images_indices = list(range(0, len(images), 2))
        # 不同角度的motion token不变性
        for i in range(0, len(images), 2):
            if i + 1 < len(images):
                image_mask = img_masks[i]
                if self.random_image_flag:
                    random_cond_idx = random.choice(cond_images_indices)
                    image_mask_random = img_masks[random_cond_idx]
                    if not image_mask.any() and not image_mask_random.any():  # 如果mask全是False
                        continue  # 跳过这次迭代
                    image_mask_combine = image_mask_random & image_mask
                    image_valid = images[i]
                    image_valid_next = images[i + 1]
                    source_pixel_values = images[random_cond_idx]
                    target_pixel_values = images[random_cond_idx+1]
                else:
                    if not image_mask.any(): 
                        continue 
                    # 只用对应的图像
                    image_mask_combine = image_mask
                    image_valid = images[i]
                    image_valid_next = images[i + 1]
                    source_pixel_values = image_valid
                    target_pixel_values = image_valid_next

                cond_hidden_states = self.vision_tower(image_valid).last_hidden_state
                target_hidden_states = self.vision_tower(image_valid_next).last_hidden_state
                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                commit_loss = commit_loss.mean()

                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                recons_pixel_values = self.decoder(
                    cond_input=source_pixel_values,
                    latent_motion_tokens=latent_motion_tokens_up
                )
                # Compute loss
                outputs = {
                    "loss": torch.zeros_like(commit_loss),
                    "commit_loss": commit_loss,
                    "recons_image_loss": torch.zeros_like(commit_loss),
                    "recons_hidden_loss": torch.zeros_like(commit_loss),
                    "perceptual_loss": torch.zeros_like(commit_loss),
                }

                # 获取运动区域掩码
                motion_masks = self.get_motion_regions_blur(source_pixel_values[image_mask_combine],target_pixel_values[image_mask_combine])
                pixel_loss = F.mse_loss(target_pixel_values[image_mask_combine] * motion_masks, recons_pixel_values[image_mask_combine] * motion_masks, reduction='none')
                valid_pixels = torch.sum(motion_masks > 0).float()
                motion_recons_image_loss = torch.sum(pixel_loss) / valid_pixels if valid_pixels > 0 else 0.0

                global_recons_image_loss = F.mse_loss(target_pixel_values[image_mask_combine], recons_pixel_values[image_mask_combine])
                lambda_motion = 0.9  # 运动区域损失的权重
                recons_image_loss = lambda_motion * motion_recons_image_loss + (1 - lambda_motion) * global_recons_image_loss

                outputs["recons_image_loss"] = recons_image_loss

                if self.perceptual_loss_w > 0:
                    with torch.no_grad():
                        perceptual_loss = self.loss_fn_lpips.forward(
                            target_pixel_values, recons_pixel_values, normalize=True)
                        perceptual_loss = perceptual_loss[image_mask_combine].mean()
                else:
                    perceptual_loss = torch.zeros_like(recons_image_loss)
                outputs["perceptual_loss"] = perceptual_loss

                pair_loss =  self.commit_loss_w * outputs["commit_loss"]  \
                                + self.recon_loss_w * outputs["recons_image_loss"] \
                                + self.perceptual_loss_w * outputs["perceptual_loss"] 
                
                outputs["loss"] = pair_loss

                # 累加每个配对的损失
                total_loss += pair_loss
                pair_count += 1
                
                # 累积各种损失度量
                for key in all_outputs:
                    if key in outputs:
                        all_outputs[key] += outputs[key].detach() if torch.is_tensor(outputs[key]) else outputs[key]
                
                # 计算活跃码本数量
                active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(pair_loss.device)         
                all_outputs["active_code_num"] += active_code_num

                # 如果有配对被处理，计算平均损失
        if pair_count > 0:
            total_loss = total_loss / pair_count
            # 计算所有度量的平均值
            for key in all_outputs:
                all_outputs[key] = (all_outputs[key] / pair_count).detach().cpu().item()  
        
        return total_loss, all_outputs


    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:

        batch_train = batch
        no_ego4d_mask =  torch.tensor([robot_type != "ego4d" for robot_type in batch['robot_type']], dtype=torch.bool).to(batch_train['observation.state'].device)
    
        if self.action_branch and self.vision_branch:
            total_loss, all_outputs = self.action_vision_dual_branch(batch_train, no_ego4d_mask, noise, time)

        elif self.action_branch and not self.vision_branch:
            total_loss, all_outputs = self.action_only_branch(batch_train, no_ego4d_mask, noise, time)
        elif self.vision_branch and not self.action_branch:
            total_loss, all_outputs = self.vision_only_branch(batch_train, no_ego4d_mask, noise, time)
        else:
            raise ValueError(f"Invalid branch configuration: {self.action_branch} and {self.vision_branch}")


        return total_loss, all_outputs

         

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        images_wrist = []
        img_masks_wrist = []

        # present_img_keys = ['observation.images.image_0', 'observation.images.image_1', 'observation.images.image_2',]
        # wrist_img_keys = ['observation.images.image_wrist_0']
        # missing_img_keys = []
        present_img_keys = [key for key in self.config.image_features if key in batch and 'wrist' not in key]
        wrist_img_keys = [key for key in self.config.image_features if 'wrist' in key]
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

        for key in wrist_img_keys:
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
                
                images_wrist.append(his_img)
                img_masks_wrist.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(images[0]) * -1
            mask = torch.zeros_like(img_masks[0])
            images.append(img)
            img_masks.append(mask)

        return images, img_masks, images_wrist, img_masks_wrist
    
    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """This should be called whenever the environment is reset."""
        pass

    def get_motion_regions_blur(self, source_frame, target_frame, kernel_size=5, sigma=1.0, percentile=80):
        """使用高斯模糊减少噪声影响"""
        # 计算帧差
        frame_diff = torch.abs(target_frame - source_frame)
        
        # 转为单通道
        if frame_diff.shape[1] > 1:
            frame_diff = frame_diff.mean(dim=1, keepdim=True)
        
        # 应用高斯模糊
        padding = kernel_size // 2
        blurred_diff = gaussian_blur(frame_diff, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        
        # 计算阈值并生成掩码
        motion_masks = []
        for i in range(blurred_diff.shape[0]):
            diff = blurred_diff[i]
            threshold = torch.quantile(diff.flatten(), percentile/100.0)
            mask = (diff > threshold).float()
            motion_masks.append(mask)
        
        return torch.stack(motion_masks, dim=0)
    def get_motion_regions_simple(self, source_frame, target_frame, threshold_percentile=85):
        """使用简单帧差法获取运动区域掩码"""
        # 计算帧差
        frame_diff = torch.abs(target_frame - source_frame)
        
        # 如果是RGB图像，将通道合并
        if frame_diff.shape[1] > 1:
            frame_diff = frame_diff.mean(dim=1, keepdim=True)
        
        # 使用百分位数阈值
        motion_masks = []
        for i in range(frame_diff.shape[0]):
            diff = frame_diff[i]
            threshold = torch.quantile(diff.flatten(), threshold_percentile/100.0)
            mask = (diff > threshold).float()
            motion_masks.append(mask)
        
        return torch.stack(motion_masks, dim=0)
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], action_horizon: int | None = None,noise: Tensor | None = None) -> Tensor:
        pass

    @torch.no_grad
    def inference(self, batch: dict[str, Tensor], noise=None, time=None) -> Tensor:

        self.eval()
        if not self.config.heterogeneous:
            # preprocess
            batch = self.normalize_inputs(batch)
            batch = self.normalize_targets(batch)
            batch_train = self.prepare_batch_train(batch)
        else:
            batch_train = batch

        if self.action_branch:
            if not self.config.heterogeneous:
                state = self.prepare_state(batch_train)
                actions,action_mask = self.prepare_action_mask(batch_train) 
                actions_is_pad = batch_train.get("action.arm_joint_position_is_pad")    
            else:
                state = batch_train['observation.state']
                actions,action_mask = self.prepare_action_mask(batch_train) 
                actions_is_pad = batch_train.get("action_is_pad")
            emb_actions = self.action_encode(actions)
            emb_actions = self.action_encoder_proj(emb_actions)
            if self.config.combine_codebook:
                action_quant, action_indices, action_commit_loss = self.combine_vector_quantizer(emb_actions)
            else:
                action_quant, action_indices, action_commit_loss = self.action_vector_quantizer(emb_actions)
            # action decoder
            recons_actions_loss = self.action_decoder.forward(action_quant, state[:,:1,:], actions, noise, time)

        images, img_masks, images_wrist, img_masks_wrist= self.prepare_images(batch_train) 
        source_pixel_values = {}
        target_pixel_values = {}
        recons_pixel_values = {}
        recons_image_loss = {}
        for i in range(0, len(images_wrist), 2):
            if i + 1 < len(images_wrist):
                image_mask = img_masks_wrist[i]
                if not image_mask.any():  # 如果mask全是False
                    continue  # 跳过这次迭代
                cond_hidden_states = self.vision_tower(images_wrist[i]).last_hidden_state
                target_hidden_states = self.vision_tower(images_wrist[i + 1]).last_hidden_state
                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                if self.config.combine_codebook:
                    quant, indices, commit_loss = self.combine_vector_quantizer(latent_motion_tokens_down)
                else:
                    quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                
                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                # only use the corresponding image
                source_pixel_values[f"image_wrist_{i//2}"] = images_wrist[i]
                
                recons_pixel_values[f"image_wrist_{i//2}"] = self.decoder(
                    cond_input=source_pixel_values[f"image_wrist_{i//2}"],
                    latent_motion_tokens=latent_motion_tokens_up
                )
                target_pixel_values[f"image_wrist_{i//2}"] = images_wrist[i + 1]
                recons_image_loss[f"image_wrist_{i//2}"] = F.mse_loss(target_pixel_values[f"image_wrist_{i//2}"][image_mask], recons_pixel_values[f"image_wrist_{i//2}"][image_mask])
        
        for i in range(0, len(images), 2):
            if i + 1 < len(images):
                image_mask = img_masks[i]
                if not image_mask.any():  # if mask is all False
                    continue  # skip this iteration
                cond_hidden_states = self.vision_tower(images[i]).last_hidden_state
                target_hidden_states = self.vision_tower(images[i + 1]).last_hidden_state
                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                if self.config.combine_codebook:
                    quant, indices, commit_loss = self.combine_vector_quantizer(latent_motion_tokens_down)
                else:
                    quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                
                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                # only use the corresponding image
                source_pixel_values[f"image_{i//2}"] = images[i]
                
                recons_pixel_values[f"image_{i//2}"] = self.decoder(
                    cond_input=source_pixel_values[f"image_{i//2}"],
                    latent_motion_tokens=latent_motion_tokens_up
                )
                target_pixel_values[f"image_{i//2}"] = images[i+1]
                recons_image_loss[f"image_{i//2}"] = F.mse_loss(target_pixel_values[f"image_{i//2}"][image_mask], recons_pixel_values[f"image_{i//2}"][image_mask])

        
        return source_pixel_values, target_pixel_values, recons_pixel_values, recons_image_loss, recons_actions_loss
    
    @torch.no_grad
    def inference_vis(self, batch: dict[str, Tensor], noise=None, time=None) -> Tensor:

        self.eval()
        codebook_features = {
            "action": {},
            "images_0": {},
            "images_1": {},
            "images_2": {},
            "images_3": {},
            "images_wrist_0": {},
            "images_wrist_1": {},          
        }
        if not self.config.heterogeneous:
            # preprocess
            batch = self.normalize_inputs(batch)
            batch = self.normalize_targets(batch)
            batch_train = self.prepare_batch_train(batch)
        else:
            batch_train = batch


        if not self.config.heterogeneous:
            state = self.prepare_state(batch_train)
            actions,action_mask = self.prepare_action_mask(batch_train) 
            actions_is_pad = batch_train.get("action.arm_joint_position_is_pad")    
        else:
            state = batch_train['observation.state']
            actions,action_mask = self.prepare_action_mask(batch_train) 
            actions_is_pad = batch_train.get("action_is_pad")
        emb_actions = self.action_encode(actions)
        emb_actions = self.action_encoder_proj(emb_actions)
        if self.config.combine_codebook:
            action_quant, action_indices, action_commit_loss = self.combine_vector_quantizer(emb_actions)
        else:
            action_quant, action_indices, action_commit_loss = self.action_vector_quantizer(emb_actions)
            codebook_features["action"]["indices"] = action_indices
            codebook_features["action"]["features"] = action_quant
        # action decoder
        recons_actions_loss = self.action_decoder.forward(action_quant, state[:,:1,:], actions, noise, time)

        images, img_masks, images_wrist, img_masks_wrist= self.prepare_images(batch_train) 
        source_pixel_values = {}
        target_pixel_values = {}
        recons_pixel_values = {}
        recons_image_loss = {}
        for i in range(0, len(images_wrist), 2):
            if i + 1 < len(images_wrist):
                image_mask = img_masks_wrist[i]
                if not image_mask.any():  # if mask is all False
                    continue  # skip this iteration
                cond_hidden_states = self.vision_tower(images_wrist[i]).last_hidden_state
                target_hidden_states = self.vision_tower(images_wrist[i + 1]).last_hidden_state
                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                if self.config.combine_codebook:
                    quant, indices, commit_loss = self.combine_vector_quantizer(latent_motion_tokens_down)
                else:
                    quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                    codebook_features[f"images_wrist_{i//2}"]["indices"] = indices
                    codebook_features[f"images_wrist_{i//2}"]["features"] = quant
                
                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                # only use the corresponding image
                source_pixel_values[f"image_wrist_{i//2}"] = images_wrist[i]
                
                recons_pixel_values[f"image_wrist_{i//2}"] = self.decoder(
                    cond_input=source_pixel_values[f"image_wrist_{i//2}"],
                    latent_motion_tokens=latent_motion_tokens_up
                )
                target_pixel_values[f"image_wrist_{i//2}"] = images_wrist[i + 1]
                recons_image_loss[f"image_wrist_{i//2}"] = F.mse_loss(target_pixel_values[f"image_wrist_{i//2}"][image_mask], recons_pixel_values[f"image_wrist_{i//2}"][image_mask])
        
        for i in range(0, len(images), 2):
            if i + 1 < len(images):
                image_mask = img_masks[i]
                if not image_mask.any():  # if mask is all False
                    continue  # skip this iteration
                cond_hidden_states = self.vision_tower(images[i]).last_hidden_state
                target_hidden_states = self.vision_tower(images[i + 1]).last_hidden_state
                query_num = self.mformer.query_num 
                latent_motion_tokens = self.mformer(
                    cond_hidden_states=cond_hidden_states,
                    target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
                latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
                if self.config.combine_codebook:
                    quant, indices, commit_loss = self.combine_vector_quantizer(latent_motion_tokens_down)
                else:
                    quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
                    codebook_features[f"images_{i//2}"]["indices"] = indices
                    codebook_features[f"images_{i//2}"]["features"] = quant
                
                # image decoder
                latent_motion_tokens_up = self.vq_up_resampler(quant)

                # only use the corresponding image
                source_pixel_values[f"image_{i//2}"] = images[i]
                
                recons_pixel_values[f"image_{i//2}"] = self.decoder(
                    cond_input=source_pixel_values[f"image_{i//2}"],
                    latent_motion_tokens=latent_motion_tokens_up
                )
                target_pixel_values[f"image_{i//2}"] = images[i+1]
                recons_image_loss[f"image_{i//2}"] = F.mse_loss(target_pixel_values[f"image_{i//2}"][image_mask], recons_pixel_values[f"image_{i//2}"][image_mask])

        return source_pixel_values, target_pixel_values, recons_pixel_values, codebook_features
     

    # check the latent action whether is correct
    def inference_latent_action(self, batch: dict[str, Tensor],pred_latent_action: Tensor, latent_action: Tensor) -> Tensor:

        self.eval()
        images, img_masks = self.prepare_images(batch) 

        i = 0
        # latent_action_random = torch.randn(latent_action.shape).to(latent_action.device)
        # latent_motion_tokens_up = self.vq_up_resampler(latent_action_random)
        latent_action = latent_action.to(torch.float32)
        latent_motion_tokens_up = self.vq_up_resampler(pred_latent_action)


        # only use the corresponding image
        source_pixel_values = images[i]

        recons_pixel_values = self.decoder(
            cond_input=source_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )

        target_pixel_values = images[i+1]

        # get motion region masks
        motion_masks = self.get_motion_regions_blur(source_pixel_values,target_pixel_values)
        pixel_loss = F.mse_loss(target_pixel_values * motion_masks, recons_pixel_values * motion_masks, reduction='none')
        valid_pixels = torch.sum(motion_masks > 0).float()
        motion_recons_image_loss = torch.sum(pixel_loss) / valid_pixels if valid_pixels > 0 else 0.0

        global_recons_image_loss = F.mse_loss(target_pixel_values, recons_pixel_values)
        lambda_motion = 0.8  # weight of motion region loss
        recons_image_loss = lambda_motion * motion_recons_image_loss + (1 - lambda_motion) * global_recons_image_loss

        # visualization 
        import cv2
        # source image 
        source_pixel_values_vis = (source_pixel_values+1.0) / 2.0
        image = source_pixel_values_vis[0].permute(1, 2, 0).cpu().numpy()  # convert to (H, W, C)  
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  
        cv2.imwrite("./source_pixel_values.png", image)

        # motion mask
        mask = motion_masks[0, 0].cpu().numpy()  # extract mask [H, W]
        masked_image = image.copy()
        # set the region where the mask is 0 to 0 in the image
        masked_image[mask < 0.5] = 0  # assume the mask is binary, otherwise can adjust the threshold
        cv2.imwrite("./masked_pixel_values.png", masked_image)

        taget_pixel_values_vis = (target_pixel_values+1.0) / 2.0
        image = taget_pixel_values_vis[0].permute(1, 2, 0).cpu().numpy()  # convert to (H, W, C)  
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  
        cv2.imwrite("./target_pixel_values.png", image)

        recons_pixel_values_vis = (recons_pixel_values+1.0) / 2.0
        recons_pixel_values_vis = torch.clamp(recons_pixel_values_vis, 0.0, 1.0)
        image = recons_pixel_values_vis[0].permute(1, 2, 0).cpu().numpy()  # convert to (H, W, C)  
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  
        cv2.imwrite("./recons_pixel_values.png", image)

        return recons_image_loss, recons_pixel_values
    