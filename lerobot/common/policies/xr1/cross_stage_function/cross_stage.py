
import torch
import torch.nn.functional as F  # noqa: N812

from torch import Tensor, nn
import numpy as np



from lerobot.common.policies.pretrained import PreTrainedPolicy
from transformers.models.vit.modeling_vit import (
    ViTPatchEmbeddings,
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from transformers.models.auto import AutoModel
from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.pi0.flex_attention import flex_attention_forward
from lerobot.common.policies.xr1.cross_stage_function.common_func import *

from lerobot.common.policies.xr1.cross_stage_function.cross_stage_module import MFormer,VectorQuantizer2,ResidualTemporalBlock,LatentMotionDecoder
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from lerobot.common.policies.xr1.configuration_xr1_stage2 import Xr1Stage2Config


from deploy.real_robot.root_path import REAL_ROBOT_DEPLOY_ROOT_PATH

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


class Stage1SupervisedModel(PreTrainedPolicy):
    config_class = Xr1Stage2Config
    name = "stage1_supervised_model"
    def __init__(
        self,
        config: Xr1Stage2Config,
    ):
        super().__init__(config)
        self.config = config
        
        if self.config.vision_branch:
            if self.config.real_robot_dev:
                self.vision_tower = AutoModel.from_pretrained(f'{REAL_ROBOT_DEPLOY_ROOT_PATH}/PretrainModel/siglip/models--google--siglip-so400m-patch14-224/snapshots/d04cf29fca7b6374f74d8bea1969314492266b5e').vision_model
            else:
                self.vision_tower = AutoModel.from_pretrained('../pretrained/siglip-so400m-patch14-224').vision_model
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
    
            # stage2 latent image token check
            self.stage2_latent_image_token_check = config.stage2_latent_image_token_check
            if self.stage2_latent_image_token_check:
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
        if self.config.action_branch:
            # action latent feature
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
            
            self.action_decoder = motion_token_action_decoder_expert(config)

        if self.config.action_branch and not self.config.vision_branch:
            self.action_vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
        elif self.config.vision_branch and not self.config.action_branch:
            self.vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
        elif self.config.action_branch and self.config.vision_branch:
            if self.config.combine_codebook:
                self.combine_vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size, e_dim=self.config.codebook_embed_dim, beta=0.25,remap=None,sane_index_shape=True)
            else:
                self.vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
                self.action_vector_quantizer = VectorQuantizer2(n_e=self.config.codebook_k_size,e_dim=self.config.codebook_embed_dim,beta=0.25,remap=None,sane_index_shape=True)
        else:
            raise ValueError(f"Invalid branch configuration: {self.config.action_branch} and {self.config.vision_branch}")
        # load pretrained weights when initializing
        self._load_pretrained_weights()
        self.set_requires_grad()
        # self.set_action_decoder_requires_grad()

    def to_bfloat16(self):
        params_to_change_dtype = [
            "mformer",
            "vq_down_resampler",
            "vector_quantizer",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)
    def set_requires_grad(self):
        # freeze all layers
        self.eval()
        for params in self.parameters():
            params.requires_grad = False
    def set_action_decoder_requires_grad(self):
        for params in self.action_decoder.parameters():
            params.requires_grad = True
    def prepare_action_mask(self, batch):
        actions,action_mask = pad_vector_mask(batch[ACTION], self.config.max_action_dim)
        return actions,action_mask
    
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
    
    def get_action_latent_token(self, batch: dict[str, Tensor]):
        actions,action_mask = self.prepare_action_mask(batch) 
        actions_is_pad = batch.get("action_is_pad")
        emb_actions = self.action_encode(actions)
        emb_actions = self.action_encoder_proj(emb_actions)
        if self.config.combine_codebook:
            action_quant, _, _ = self.combine_vector_quantizer(emb_actions)
        else:   
            action_quant, _, _ = self.action_vector_quantizer(emb_actions)
        # interpolate to the same length as the action latent token num
        if action_quant.shape[1] != self.config.action_latent_token_num:
            action_latent_token = F.interpolate(
                        action_quant.transpose(1, 2),  # (bs, dim, sqe)
                        size=self.config.action_latent_token_num, 
                        mode='linear'
                    ).transpose(1, 2)  # (bs, action_latent_token_num, dim)
        else:
            action_latent_token = action_quant
        return action_latent_token
    
    def get_motion_latent_token(self, cond_hidden_states,target_hidden_states):
        query_num = self.mformer.query_num 
        latent_motion_tokens = self.mformer(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num] # bs*8*768
        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)

        if self.config.combine_codebook:
            quant, indices, _ = self.combine_vector_quantizer(latent_motion_tokens_down)
        else:   
            quant, indices, _ = self.vector_quantizer(latent_motion_tokens_down)
        
        return quant


    def forward(self, batch: dict[str, Tensor]):
        pass
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
                    
                    # load weights
                    self.load_state_dict(stage1_pretrained_weights, strict=False)
                    print(f"Successfully loaded stage1 weights from {stage1_pretrained_path}")
                except Exception as e:
                    print(f"Error loading stage1 weights: {e}")
            else:
                print(f"stage1 pretrained model not found at {model_file}")

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """This should be called whenever the environment is reset."""
        pass
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], action_horizon: int | None = None,noise: Tensor | None = None) -> Tensor:
        pass               

class MotionTokenWithExpertConfig(PretrainedConfig):
    model_type = "MotionTokenWithExpertModel"
    sub_configs = {"gemma_expert_config": AutoConfig}

    def __init__(
        self,
        gemma_expert_config: dict | None = None,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.attention_implementation = attention_implementation

        self.gemma_expert_config = CONFIG_MAPPING["gemma"](
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=2,
            eos_token_id=1,
            head_dim=256,
            hidden_act="gelu_pytorch_tanh",
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=4096,
            max_position_embeddings=8192,
            model_type="gemma",
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            pad_token_id=0,
            rms_norm_eps=1e-06,
            rope_theta=10000.0,
            torch_dtype="float32",
            transformers_version="4.49.0",
            use_cache=True,
            vocab_size=257152,
        )

        super().__init__(**kwargs)

class MotionTokenWithExpertModel(PreTrainedModel):
    config_class = MotionTokenWithExpertConfig
    def __init__(self, config: MotionTokenWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()


    def to_bfloat16_like_physical_intelligence(self):
        params_to_change_dtype = [
            "gemma_expert.model.layers",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        models = [self.gemma_expert.model]

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.gemma_expert.config.num_hidden_layers
        head_dim = self.gemma_expert.config.head_dim
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_output = att_output.to(dtype=torch.bfloat16)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.gemma_expert_config.num_attention_heads
        num_key_value_heads = self.config.gemma_expert_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output



class motion_token_action_decoder_expert(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        motiontoken_with_export_config = MotionTokenWithExpertConfig()
        self.motion_token_expert = MotionTokenWithExpertModel(motiontoken_with_export_config)
       
        self.motion_token_proj = nn.Linear(self.config.codebook_embed_dim, self.config.proj_width)

        # self.vlm_prefix_proj = nn.Linear(2048, self.config.proj_width)
    
        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = True

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


    def embed_suffix(self, latent_actions, state, noisy_actions, timestep,vlm_prefix_out=None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device
        # vlm_prefix_out = None
        # vlm prefix token
        if vlm_prefix_out is not None:
            vlm_prefix_out = self.vlm_prefix_proj(vlm_prefix_out)
            vlm_prefix_out = vlm_prefix_out.to(dtype=torch.bfloat16)
            embs.append(vlm_prefix_out)
            num_vlm_prefix_embs = vlm_prefix_out.shape[1]
            vlm_prefix_mask = torch.ones(bsize, num_vlm_prefix_embs, dtype=torch.bool, device=vlm_prefix_out.device)
            pad_masks.append(vlm_prefix_mask)
            att_masks += [0] * num_vlm_prefix_embs

        # motion token
        latent_actions_emb = self.motion_token_proj(latent_actions)
        latent_actions_emb = latent_actions_emb.to(dtype=torch.bfloat16)
        embs.append(latent_actions_emb)
        num_latent_embs = latent_actions_emb.shape[1]
        latent_action_mask = torch.ones(bsize, num_latent_embs, dtype=torch.bool, device=latent_actions_emb.device)
        pad_masks.append(latent_action_mask)
        att_masks += [0] * num_latent_embs
        
        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        if len(state_emb.shape) == 2:
            embs.append(state_emb[:, None, :])
        else:
            embs.append(state_emb)
        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [0]

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

    def forward(
        self, latent_actions, state, actions, vlm_prefix_out=None,noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions


        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(latent_actions, state, x_t, time,vlm_prefix_out)

        pad_masks = suffix_pad_masks
        att_masks = suffix_att_masks

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        suffix_out, _ = self.motion_token_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[0][:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, latent_actions, state, vlm_prefix_out=None, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                latent_actions,
                state,
                x_t,
                expanded_time,
                vlm_prefix_out,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        latent_actions,
        state,
        x_t,
        timestep,
        vlm_prefix_out=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(latent_actions,state, x_t, timestep,vlm_prefix_out)


        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = suffix_att_2d_masks

        position_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.motion_token_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = outputs_embeds[0]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t