import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from diffusers.utils import apply_lora_scale
from diffusers.models import ModelMixin  # noqa: F401
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanRotaryPosEmbed
from diffusers.configuration_utils import register_to_config
from peft import LoraConfig
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from ..utils import flex_freeze
from ..diffusers.wan import (
    WanTimeTextImageEmbeddingFix, FP32LayerNorm, WanTransformerBlockFix, _WanTransformer3DModelMod)
from .gm_output import GMFlowModelOutput
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


class _GMWanTransformer3DModel(_WanTransformer3DModelMod):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            patch_size: Tuple[int] = (1, 2, 2),
            num_attention_heads: int = 40,
            attention_head_dim: int = 128,
            in_channels: int = 16,
            out_channels: int = 16,
            text_dim: int = 4096,
            freq_dim: int = 256,
            ffn_dim: int = 13824,
            num_layers: int = 40,
            cross_attn_norm: bool = True,
            qk_norm: Optional[str] = "rms_norm_across_heads",
            eps: float = 1e-6,
            image_dim: Optional[int] = None,
            added_kv_proj_dim: Optional[int] = None,
            rope_max_seq_len: int = 1024,
            pos_embed_seq_len: Optional[int] = None,
            causal_attention: bool = False,
            causal_num_frames_per_block: int = 1):
        super(WanTransformer3DModel, self).__init__()

        self._validate_causal_config(patch_size, causal_attention, causal_num_frames_per_block)

        self.num_gaussians = num_gaussians

        inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbeddingFix(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockFix(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        patch_numel = math.prod(patch_size)
        self.proj_out_means = nn.Linear(
            inner_dim, patch_numel * self.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(
            inner_dim, patch_numel * self.num_gaussians)
        self.constant_logstd = constant_logstd

        if self.constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = inner_dim
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

        self.gradient_checkpointing = False

        self._causal_attention_mask_cache: dict[tuple[Any, ...], Any] = {}
        if self.causal_attention:
            self._enable_causal_attention_backend()

    def init_weights(self):
        # Output layers
        constant_init(self.proj_out_means.to_empty(device='cpu'), val=0)
        rand_noise = torch.randn((self.num_gaussians * self.out_channels)) * 0.1
        self.proj_out_means.bias.data.copy_(rand_noise[None, :].expand(
            math.prod(self.config.patch_size), -1).flatten())
        constant_init(self.proj_out_logweights.to_empty(device='cpu'), val=0)
        if self.constant_logstd is None:
            # logstd layers
            for m in self.proj_out_logstds:
                if isinstance(m, nn.Linear):
                    xavier_init(m.to_empty(device='cpu'), distribution='uniform')
            constant_init(self.proj_out_logstds[-1], val=0)

    @apply_lora_scale("attention_kwargs")
    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: torch.Tensor | None = None,
            attention_kwargs: dict[str, Any] | None = None):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        self_attention_mask = self._get_causal_attention_mask(
            post_patch_num_frames=post_patch_num_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).unbind(dim=2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        num_gaussians = self.num_gaussians
        out_means = self.proj_out_means(hidden_states).reshape(
            batch_size,
            post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, num_gaussians * self.out_channels
        ).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch_size, num_gaussians, self.out_channels,
            post_patch_num_frames * p_t, post_patch_height * p_h, post_patch_width * p_w)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            batch_size,
            post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, num_gaussians
        ).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch_size, num_gaussians, 1,
            post_patch_num_frames * p_t, post_patch_height * p_h, post_patch_width * p_w)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(
                temb.detach()).reshape(batch_size, 1, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((batch_size, 1, 1, 1, 1, 1), float(self.constant_logstd))

        return GMFlowModelOutput(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)


@MODULES.register_module()
class GMWanTransformer3DModel(_GMWanTransformer3DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.init_weights(pretrained)

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

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            patch_numel = math.prod(self.config.patch_size)
            ori_out_channels = patch_numel * self.out_channels
            if 'proj_out.weight' in state_dict:
                if state_dict['proj_out.weight'].size(0) == ori_out_channels:
                    state_dict['proj_out_means.weight'] = state_dict['proj_out.weight'].reshape(
                        patch_numel, 1, self.out_channels, -1
                    ).expand(-1, self.num_gaussians, -1, -1).reshape(
                        self.num_gaussians * ori_out_channels, -1)
                    del state_dict['proj_out.weight']
            if 'proj_out.bias' in state_dict:
                if state_dict['proj_out.bias'].size(0) == ori_out_channels:
                    state_dict['proj_out_means.bias'] = state_dict['proj_out.bias'].reshape(
                        patch_numel, 1, self.out_channels
                    ).expand(-1, self.num_gaussians, -1).reshape(
                        self.num_gaussians * ori_out_channels)
                    rand_noise = torch.randn(
                        (self.num_gaussians * self.out_channels),
                        dtype=state_dict['proj_out_means.bias'].dtype,
                        device=state_dict['proj_out_means.bias'].device) * 0.05
                    state_dict['proj_out_means.bias'] += rand_noise[None, :].expand(patch_numel, -1).flatten()
                    del state_dict['proj_out.bias']
            if (self.constant_logstd is None
                    and 'proj_out_means.weight' in state_dict
                    and 'proj_out_means.bias' in state_dict):
                self.proj_out_logstds[-1].bias.data[:1] = math.log(0.05)  # reduce the initial logstd
            load_full_state_dict(self, state_dict, logger=logger, assign=True)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            return super().forward(
                hidden_states=hidden_states.to(dtype),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_image=None,  # CLIP image conditioning is not implemented
                **kwargs)
