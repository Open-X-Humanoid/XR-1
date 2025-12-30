
import torch,math
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.functional import gaussian_blur
from torch import nn, einsum 
 
from einops_exts import rearrange_many

import numpy as np
from einops import rearrange,repeat
from einops.layers.torch import Rearrange

from transformers.models.vit.modeling_vit import (
    ViTPatchEmbeddings,
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=4, causal=True, no_pad=False):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride, no_pad=no_pad)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[i], 
                stride[i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)
    

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer or "closest"
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def setup_remap(self, remap, unknown_index):
        self.remap = remap
        self.register_buffer("used", torch.tensor(np.load(self.remap)))
        self.re_embed = self.used.shape[0]
        self.unknown_index = unknown_index  # "random" or "extra" or integer or "closest"
        if self.unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices.")


    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        bz = z.shape[0]
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        if self.remap is not None and self.unknown_index == "closest":
            embedding_weight = self.embedding(self.used)
        else:
            embedding_weight = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            # loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
            #        torch.mean((z_q - z.detach()) ** 2)
            commitment_loss = (z_q.detach() - z) ** 2
            codebook_loss = (z_q - z.detach()) ** 2
            loss = self.beta * commitment_loss + codebook_loss
        else:
            # loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
            #        torch.mean((z_q - z.detach()) ** 2)
            commitment_loss = (z_q.detach() - z) ** 2
            codebook_loss = (z_q - z.detach()) ** 2
            loss = commitment_loss + self.beta * codebook_loss
        # commitment loss: (z_q.detach() - z) ** 2, encourage encoder output to be close to the codebook vectors
        # codebook loss: (z_q - z.detach()) ** 2, encourage codebook vectors to be close to the encoder output

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.reshape(bz, -1, z_q.shape[-1])
        if self.remap is not None and self.unknown_index != "closest":
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[:-1])
        # z_q: quantized result
        # min_encoding_indices: quantized indices
        # loss: quantized loss
        return z_q, min_encoding_indices, loss

    def get_codebook_entry(self, indices):
        if self.remap is not None and self.unknown_index != "closest":
            indices = indices.reshape(indices.shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        if self.remap is not None and self.unknown_index == "closest":
            z_q = self.embedding(self.used)[indices]
        else:
            z_q = self.embedding(indices)

        return z_q
    

class MFormerEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.action_token_num = config.action_token_num
        self.query_num = query_num
        self.latent_motion_token = nn.Parameter(torch.zeros(1, query_num, config.hidden_size))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.projection = nn.Linear(config.input_hidden_size, config.hidden_size, bias=True)
        self.position_embeddings = nn.Parameter(torch.randn(1, config.num_patches*2 + 1 + query_num + self.action_token_num, config.hidden_size))
        self.token_type_embeddings = nn.Parameter(torch.randn(2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        if hasattr(config, "legacy"):
            self.legacy = config.legacy
        else:
            self.legacy = True

    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, per_seq_length = target_hidden_states.shape[:2]

        cond_embeddings = self.projection(cond_hidden_states)

        latent_motion_tokens = self.latent_motion_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1) # 前后2帧分割标记符
        cond_embeddings = torch.cat((latent_motion_tokens, cond_embeddings, sep_tokens), dim=1)

        target_embeddings = self.projection(target_hidden_states)
        embeddings = torch.cat((cond_embeddings, target_embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        # add token type encoding to each token
        cond_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, per_seq_length + self.query_num + 1 + self.action_token_num, -1)
        if self.legacy:
            target_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, per_seq_length, -1)
        else:
            target_token_type_embeddings = self.token_type_embeddings[1].expand(batch_size, per_seq_length, -1)
        token_type_embeddings = torch.cat((cond_token_type_embeddings, target_token_type_embeddings), dim=1)
        embeddings = embeddings + token_type_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MFormer(ViTPreTrainedModel):
    def __init__(self,
                 config: ViTConfig,
                 add_pooling_layer: bool = True):

        super().__init__(config)
        self.config = config
        self.query_num = config.query_num
        self.embeddings = MFormerEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MFormerEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.token_type_embeddings.data = nn.init.trunc_normal_(
                module.token_type_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.token_type_embeddings.dtype)

            module.latent_motion_token.data = nn.init.trunc_normal_(
                module.latent_motion_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.latent_motion_token.dtype)

            module.sep_token.data = nn.init.trunc_normal_(
                module.sep_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.sep_token.dtype)

    # def get_input_embeddings(self):
    #     return self.embeddings.projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
class LMDViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.query_num = query_num

        is_io_hidden_states = getattr(config, 'is_io_hidden_states', False)
        if is_io_hidden_states:
            self.patch_embeddings = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        else:
            self.patch_embeddings = ViTPatchEmbeddings(config)

        query_fusion_mode = getattr(config, 'query_fusion_mode', 'add')
        assert query_fusion_mode in ['add', 'concat']
        self.query_fusion_mode = query_fusion_mode
        self.query_pooling_layer = nn.Linear(query_num * config.hidden_size, config.hidden_size) if query_fusion_mode == 'add' else None
        # print(f"query_fusion_mode: {query_fusion_mode}")

        use_mask_token = getattr(config, 'use_mask_token', False)
        self.use_mask_token = use_mask_token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # print(f"use_mask_token: {use_mask_token}")

        seq_len = config.num_patches*(1+use_mask_token) + (query_fusion_mode=='concat') * query_num
        self.position_embeddings = nn.Parameter(torch.randn(1, seq_len, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        latent_motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        # insert [MASK] tokens to inputs
        if self.use_mask_token:
            num_patches = self.config.num_patches
            mask_tokens = self.mask_token.expand(batch_size, num_patches, -1)
            embeddings = torch.cat((embeddings, mask_tokens), dim=1)

        if self.query_fusion_mode == 'add':
            # add the projected motion token to the embedded patch tokens
            latent_motion_tokens = latent_motion_tokens.reshape(batch_size, 1, -1)
            latent_motion_tokens = self.query_pooling_layer(latent_motion_tokens)
            embeddings = embeddings + latent_motion_tokens
        elif self.query_fusion_mode == 'concat':
            # concatenate latent motion tokens with the embedded patch tokens
            embeddings = torch.cat((latent_motion_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class LMDViTModel(ViTPreTrainedModel):
    def __init__(self,
                 config: ViTConfig,
                 add_pooling_layer: bool = True):

        super().__init__(config)
        self.config = config
        is_io_hidden_states = getattr(config, 'is_io_hidden_states', False)
        self.is_io_hidden_states = is_io_hidden_states

        self.embeddings = LMDViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LMDViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        latent_motion_tokens: Optional[torch.Tensor] = None, # change lam_tokens to latent_motion_tokens
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        if self.is_io_hidden_states:
            expected_dtype = self.embeddings.patch_embeddings.weight.dtype
        else:
            expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values=pixel_values,
            latent_motion_tokens=latent_motion_tokens
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class LatentMotionDecoder(nn.Module):
    def __init__(self,
                 config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = LMDViTModel(config, add_pooling_layer=False)

        is_io_hidden_states = getattr(config, 'is_io_hidden_states', False)
        self.is_io_hidden_states = is_io_hidden_states

        if is_io_hidden_states:
            self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

            self.decoder.weight.data = nn.init.trunc_normal_(
                self.decoder.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(self.decoder.weight.dtype)
            self.decoder.bias.data.zero_()

        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=config.hidden_size,
                    out_channels=config.encoder_stride**2 * config.num_channels,
                    kernel_size=1,
                ),
                nn.PixelShuffle(config.encoder_stride),
            )

            self.decoder[0].weight.data = nn.init.trunc_normal_(
                self.decoder[0].weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(self.decoder[0].weight.dtype)


    def forward(
        self,
        cond_input: Optional[torch.Tensor] = None,
        latent_motion_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        outputs = self.transformer(
            pixel_values=cond_input,
            latent_motion_tokens=latent_motion_tokens
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, -self.config.num_patches:]
        batch_size, sequence_length, num_channels = sequence_output.shape

        if not self.is_io_hidden_states:
            height = width = math.floor(sequence_length**0.5)
            sequence_output = sequence_output.permute(0, 2, 1).contiguous().reshape(batch_size, num_channels, height, width)

        reconstructed_output = self.decoder(sequence_output)

        return reconstructed_output



def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)
    
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.norm(latents)
        
        # Convert back from bmnd to bnd format
        if times == 1:  # If we expanded from bnd to b1nd
            latents = rearrange(latents, 'b 1 n d -> b n d')

        return latents