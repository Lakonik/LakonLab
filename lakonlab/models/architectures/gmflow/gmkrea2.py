from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import init_empty_weights
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Transformer2DModel, Krea2RMSNorm, Krea2TimestepEmbedding, Krea2TextFusion, Krea2TextProjection,
    Krea2RotaryPosEmbed, Krea2TransformerBlock)
from diffusers.configuration_utils import register_to_config
from diffusers.utils import apply_lora_scale
from peft import LoraConfig
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..utils import flex_freeze
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class GMKrea2FinalNorm(nn.Module):
    """Krea2FinalLayer without its output projection, keeping the pretrained key names."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = (temb + self.scale_shift_table).chunk(2, dim=1)
        return (1.0 + scale) * self.norm(hidden_states) + shift


class _GMKrea2Transformer2DModel(Krea2Transformer2DModel):

    _no_split_modules = ['Krea2TransformerBlock', 'Krea2TextFusionBlock', 'GMKrea2FinalNorm']

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            logweights_channels=1,
            in_channels: int = 64,
            num_layers: int = 28,
            attention_head_dim: int = 128,
            num_attention_heads: int = 48,
            num_key_value_heads: int = 12,
            intermediate_size: int = 16384,
            timestep_embed_dim: int = 256,
            text_hidden_dim: int = 2560,
            num_text_layers: int = 12,
            text_num_attention_heads: int = 20,
            text_num_key_value_heads: int = 20,
            text_intermediate_size: int = 6912,
            num_layerwise_text_blocks: int = 2,
            num_refiner_text_blocks: int = 2,
            axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
            rope_theta: float = 1000.0,
            norm_eps: float = 1e-5):
        super(Krea2Transformer2DModel, self).__init__()

        self.num_gaussians = num_gaussians
        self.logweights_channels = logweights_channels

        hidden_size = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(
                f'sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim={attention_head_dim}')

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, eps=norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.final_layer = GMKrea2FinalNorm(hidden_size, eps=norm_eps)
        self.proj_out_means = nn.Linear(hidden_size, self.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(hidden_size, self.num_gaussians * self.logweights_channels)
        self.constant_logstd = constant_logstd

        if self.constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = hidden_size
            logstd_layers = []
            for _ in range(gm_num_logstd_layers - 1):
                logstd_layers.extend([
                    nn.SiLU(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.proj_out_logstds = nn.Sequential(
                *logstd_layers,
                nn.SiLU(),
                nn.Linear(in_dim, 1))

    def init_weights(self):
        # Output layers
        constant_init(self.proj_out_means.to_empty(device='cpu'), val=0)
        rand_noise = torch.randn((self.num_gaussians * self.out_channels // self.logweights_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[:, None].expand(-1, self.logweights_channels).flatten())
        constant_init(self.proj_out_logweights.to_empty(device='cpu'), val=0)
        if self.constant_logstd is None:
            # logstd layers
            for m in self.proj_out_logstds:
                if isinstance(m, nn.Linear):
                    xavier_init(m.to_empty(device='cpu'), distribution='uniform')
            constant_init(self.proj_out_logstds[-1], val=0)

    @apply_lora_scale('attention_kwargs')
    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            position_ids: torch.Tensor,
            encoder_attention_mask: torch.Tensor | None = None,
            attention_kwargs: dict[str, Any] | None = None):
        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate='tanh'))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb_mod, image_rotary_emb, attention_mask)
            else:
                hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]

        # Output layers
        hidden_states = self.final_layer(hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        out_means = self.proj_out_means(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels).log_softmax(dim=-2)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(temb.detach()).reshape(bs, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1), float(self.constant_logstd))

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMKrea2Transformer2DModel(_GMKrea2Transformer2DModel):

    # Keep a uniform parameter dtype for FSDP1 flat-parameter sharding, as in Wan.
    _keep_in_fp32_modules = None

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            checkpointing=True,
            compile_forward=False,
            compile_kwargs=dict(
                mode='default',
                fullgraph=True,
                dynamic=False),
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        assert self.patch_size * self.patch_size == self.logweights_channels

        self.init_weights(pretrained, pretrained_adapter)

        if autocast_dtype is not None:
            assert torch_dtype == 'float32'
        self.autocast_dtype = autocast_dtype

        self.use_lora = use_lora
        self.lora_target_modules = lora_target_modules
        self.lora_rank = lora_rank
        if self.use_lora:
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights='gaussian',
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
            )
            self.add_adapter(transformer_lora_config)

        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

        if checkpointing:
            self.enable_gradient_checkpointing()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained=None, pretrained_adapter=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            if ('final_layer.linear.weight' in state_dict
                    and state_dict['final_layer.linear.weight'].size(0) == self.out_channels):
                state_dict['proj_out_means.weight'] = state_dict['final_layer.linear.weight'][None].expand(
                    self.num_gaussians, -1, -1).reshape(self.num_gaussians * self.out_channels, -1)
                del state_dict['final_layer.linear.weight']
            if ('final_layer.linear.bias' in state_dict
                    and state_dict['final_layer.linear.bias'].size(0) == self.out_channels):
                state_dict['proj_out_means.bias'] = state_dict['final_layer.linear.bias'][None].expand(
                    self.num_gaussians, -1).reshape(self.num_gaussians * self.out_channels)
                p2 = self.patch_size * self.patch_size
                rand_noise = torch.randn(
                    (self.num_gaussians * self.out_channels // p2),
                    dtype=state_dict['proj_out_means.bias'].dtype,
                    device=state_dict['proj_out_means.bias'].device) * 0.05
                state_dict['proj_out_means.bias'] += rand_noise[:, None].expand(-1, p2).flatten()
                del state_dict['final_layer.linear.bias']
            if (self.constant_logstd is None
                    and 'proj_out_means.weight' in state_dict
                    and 'proj_out_means.bias' in state_dict):
                self.proj_out_logstds[-1].bias.data = torch.full_like(
                    self.proj_out_logstds[-1].bias.data, np.log(0.05))  # reduce the initial logstd
            if pretrained_adapter is not None:
                adapter_state_dict = _load_cached_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                lora_state_dict = dict()
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k] = v
                    else:
                        state_dict[k] = v
                load_full_state_dict(self, state_dict, logger=logger, assign=True)
                if len(lora_state_dict) > 0:
                    self.load_lora_adapter(lora_state_dict, prefix=None)
                    self.fuse_lora(safe_fusing=True)
                    self.unload_lora()
                    self.requires_grad_(True)  # PEFT freezes the weights after fusing
            else:
                load_full_state_dict(self, state_dict, logger=logger, assign=True)

    def patchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        return latents

    def unpatchify(self, gm):
        if self.patch_size > 1:
            bs, k, c, h, w = gm['means'].size()
            gm['means'] = gm['means'].reshape(
                bs, k, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
            gm['logweights'] = gm['logweights'].reshape(
                bs, k, 1, self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, 1, h * self.patch_size, w * self.patch_size)
        return gm

    def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        hidden_states = hidden_states.reshape(bs, c, h * w).permute(0, 2, 1).contiguous()
        text_ids = torch.zeros(encoder_hidden_states.size(1), 3, device=hidden_states.device)
        image_ids = torch.zeros(h, w, 3, device=hidden_states.device)
        image_ids[..., 1] = torch.arange(h, device=hidden_states.device)[:, None]
        image_ids[..., 2] = torch.arange(w, device=hidden_states.device)[None, :]
        position_ids = torch.cat([text_ids, image_ids.reshape(h * w, 3)], dim=0)

        output = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_hidden_states_mask,
            timestep=timestep,
            position_ids=position_ids,
            **kwargs)

        output['means'] = output['means'].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.out_channels, h, w)
        output['logweights'] = output['logweights'].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.logweights_channels, h, w)
        output['logstds'] = output['logstds'].unsqueeze(-1)  # (bs, 1, 1, 1, 1)
        return self.unpatchify(output)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
            return forward_fn(
                hidden_states=hidden_states.to(dtype),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                **kwargs)
