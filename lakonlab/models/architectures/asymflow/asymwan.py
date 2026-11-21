import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from diffusers.models import ModelMixin  # noqa: F401
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanRotaryPosEmbed
from diffusers.configuration_utils import register_to_config
from diffusers.utils import apply_lora_scale
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze, fp32_module_forward, cast_dtype
from lakonlab.ops import quantize_frozen_linears
from ..diffusers.attention import set_attention_backends
from ..diffusers.wan import (
    WanTimeTextImageEmbeddingFix, FP32LayerNorm, WanTransformerBlockFix, _WanTransformer3DModelMod)
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from .common import AsymFlowMixin


class _AsymWanTransformer3DModel(AsymFlowMixin, _WanTransformer3DModelMod):

    # _keep_in_fp32_modules is also used by diffusers in quantization and module dtype detection
    # (modules in this list are excluded from dtype detection); buffers only — learnable
    # parameters must not be kept in fp32 as mixed dtypes break FSDP1 flat-parameter sharding
    _keep_in_fp32_modules = ['proj_buffer', 'first_frame_proj_buffer']

    @register_to_config
    def __init__(
            self,
            patch_size: Tuple[int, ...] = (4, 16, 16),
            first_frame_patch_size: Tuple[int, ...] = (1, 16, 16),
            in_channels: int = 3,
            out_channels: int | None = None,
            basis_rank: int = 64,
            input_bottleneck: bool = False,
            secondary_temb_start_layer: int | None = None,
            num_attention_heads: int = 40,
            attention_head_dim: int = 128,
            text_dim: int = 4096,
            freq_dim: int = 256,
            ffn_dim: int = 13824,
            num_layers: int = 40,
            cross_attn_norm: bool = True,
            qk_norm: Optional[str] = 'rms_norm_across_heads',
            eps: float = 1e-6,
            image_dim: Optional[int] = None,
            added_kv_proj_dim: Optional[int] = None,
            rope_max_seq_len: int = 1024,
            pos_embed_seq_len: Optional[int] = None,
            sigma_min: float = 1e-4,
            num_timesteps=1000):
        super(WanTransformer3DModel, self).__init__()

        patch_size = tuple(patch_size)
        first_frame_patch_size = tuple(first_frame_patch_size)
        assert first_frame_patch_size[1:] == patch_size[1:], \
            'first_frame_patch_size must match patch_size spatially.'

        self.patch_size = patch_size
        self.first_frame_patch_size = first_frame_patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        inner_dim = num_attention_heads * attention_head_dim

        # 1. Position embedding; evaluated directly on the token grid (see forward), hence
        # patch size (1, 1, 1). The token grid matches the pretrained Wan latent grid.
        self.rope = WanRotaryPosEmbed(attention_head_dim, (1, 1, 1), rope_max_seq_len)

        # 2. Patch embeddings (the first latent frame always has its own input projection);
        # with input_bottleneck the embeddings consume the proj_buffer-projected tokens
        # unpatchified back to the input resolution
        self.patch_embedding = nn.Conv3d(
            basis_rank // math.prod(patch_size) if input_bottleneck else in_channels,
            inner_dim, kernel_size=patch_size, stride=patch_size)
        self.first_frame_patch_embedding = nn.Conv3d(
            basis_rank // math.prod(first_frame_patch_size) if input_bottleneck else in_channels,
            inner_dim, kernel_size=first_frame_patch_size, stride=first_frame_patch_size)

        # 3. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbeddingFix(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.secondary_temb_start_layer = secondary_temb_start_layer
        if secondary_temb_start_layer is not None:
            assert 0 <= secondary_temb_start_layer < num_layers
            self.secondary_time_embedder = TimestepEmbedding(
                in_channels=freq_dim, time_embed_dim=inner_dim)
            self.secondary_time_proj = nn.Linear(inner_dim, inner_dim * 6)

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockFix(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projections (the first latent frame always has its own output projection)
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.proj_out = nn.Linear(inner_dim, self.out_channels * math.prod(patch_size))
        self.first_frame_proj_out = nn.Linear(
            inner_dim, self.out_channels * math.prod(first_frame_patch_size))

        # 6. AsymFlow attributes and buffers
        self.basis_rank = basis_rank
        self.input_bottleneck = input_bottleneck
        self.sigma_min = sigma_min
        self.num_timesteps = num_timesteps
        # Calibration is unsupported
        self.init_asymflow_buffers(
            in_channels * math.prod(patch_size), basis_rank, use_calibration=False)
        # additional first-frame projection buffer
        first_frame_patch_dim = in_channels * math.prod(first_frame_patch_size)
        assert first_frame_patch_dim >= basis_rank
        self.register_buffer(
            'first_frame_proj_buffer', torch.eye(first_frame_patch_dim, basis_rank))
        self.first_frame_proj_buffer.lakonlab_no_shard = True  # avoid FSDP sharding

        self.gradient_checkpointing = False

    @staticmethod
    def patchify(latents, patch_size, pack_channels=True):
        bs, c, t, h, w = latents.size()
        p_t, p_h, p_w = patch_size
        latents = latents.reshape(
            bs, c, t // p_t, p_t, h // p_h, p_h, w // p_w, p_w)
        if pack_channels:  # patch dim before channel dim (Wan unpatchify convention)
            latents = latents.permute(
                0, 3, 5, 7, 1, 2, 4, 6
            ).reshape(bs, p_t * p_h * p_w * c, t // p_t, h // p_h, w // p_w)
        else:  # channel dim before patch dim (for consistency with other models)
            latents = latents.permute(
                0, 1, 3, 5, 7, 2, 4, 6
            ).reshape(bs, c, p_t * p_h * p_w, t // p_t, h // p_h, w // p_w)
        return latents

    @staticmethod
    def unpatchify(latents, patch_size, packed_channels=True):
        p_t, p_h, p_w = patch_size
        p = p_t * p_h * p_w
        if packed_channels:
            bs, c, t, h, w = latents.size()
            latents = latents.reshape(
                bs, p_t, p_h, p_w, c // p, t, h, w
            ).permute(
                0, 4, 5, 1, 6, 2, 7, 3
            ).reshape(
                bs, c // p, t * p_t, h * p_h, w * p_w)
        else:
            bs, c, _, t, h, w = latents.size()
            latents = latents.reshape(
                bs, c, p_t, p_h, p_w, t, h, w
            ).permute(
                0, 1, 5, 2, 6, 3, 7, 4
            ).reshape(
                bs, c, t * p_t, h * p_h, w * p_w)
        return latents

    @staticmethod
    def pack(latents):
        bs, c, t, h, w = latents.shape
        latents = latents.reshape(bs, c, t * h * w).permute(0, 2, 1)
        return latents

    @staticmethod
    def unpack(latents, t, h, w):
        bs, _, c = latents.shape
        latents = latents.permute(0, 2, 1).reshape(bs, c, t, h, w)
        return latents

    @apply_lora_scale('attention_kwargs')
    def forward(
            self,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: torch.Tensor | None = None,
            attention_kwargs: dict[str, Any] | None = None,
            cache_mode: Optional[str] = None,
            cache_config: Optional[dict] = None):
        bs, _, num_frames, height, width = x_t.size()
        p_t, p_h, p_w = self.patch_size
        ff_p_t = self.first_frame_patch_size[0]
        assert (num_frames - ff_p_t) % p_t == 0 and height % p_h == 0 and width % p_w == 0
        post_patch_num_frames = 1 + (num_frames - ff_p_t) // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        x_t_first = x_t[:, :, :ff_p_t]
        x_t_rest = x_t[:, :, ff_p_t:]
        x_t_first_packed = self.pack(self.patchify(x_t_first, self.first_frame_patch_size))
        x_t_rest_packed = self.pack(self.patchify(x_t_rest, self.patch_size))
        num_first_tokens = x_t_first_packed.size(1)

        # dummy calibration step, calibration is not used in AsymWan
        # timestep shape is (bs, ), Wan 2.2 TI2V per-token timestep is not supported
        calibration = self.asymflow_calibration(timestep, bs, 3)

        if cache_mode is not None:
            cache_after_block = cache_config['cache_after_block']
            if self.secondary_temb_start_layer is not None:
                assert cache_after_block < self.secondary_temb_start_layer

        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-bs:]
            encoder_hidden_states = self._cache_storage.pop('encoder_hidden_states')[-bs:]
            rotary_emb = self._cache_storage.pop('rotary_emb')
            temb = self._cache_storage.pop('temb')[-bs:]
            timestep_proj = self._cache_storage.pop('timestep_proj')[-bs:]

            start_block = cache_after_block + 1

        else:
            if self.input_bottleneck:
                x_t_first = self.unpatchify(self.unpack(
                    x_t_first_packed @ self.first_frame_proj_buffer.to(x_t.dtype),
                    1, post_patch_height, post_patch_width
                ), self.first_frame_patch_size)
                x_t_rest = self.unpatchify(self.unpack(
                    x_t_rest_packed @ self.proj_buffer.to(x_t.dtype),
                    post_patch_num_frames - 1, post_patch_height, post_patch_width
                ), self.patch_size)
            hidden_states = self.first_frame_patch_embedding(x_t_first)
            if x_t_rest.size(2) > 0:  # single-frame (image) inputs have no rest segment
                hidden_states = torch.cat([hidden_states, self.patch_embedding(x_t_rest)], dim=2)

            # rope was built with patch size (1, 1, 1), evaluating positions on the token grid
            rotary_emb = self.rope(hidden_states)
            hidden_states = self.pack(hidden_states)

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                calibration.timestep, encoder_hidden_states, encoder_hidden_states_image
            )
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

            if encoder_hidden_states_image is not None:
                encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

            start_block = 0

        for index_block in range(start_block, len(self.blocks)):
            # Update temb and timestep_proj when we reach the specified block
            if index_block == self.secondary_temb_start_layer:
                temb = self.secondary_time_embedder(
                    self.condition_embedder.timesteps_proj(calibration.timestep).type_as(hidden_states))
                timestep_proj = self.secondary_time_proj(
                    self.condition_embedder.act_fn(temb)).unflatten(1, (6, -1))

            block = self.blocks[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, None)
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, None)

            # Save cache after the specified block
            if cache_mode == 'save' and index_block == cache_after_block:
                self._cache_storage['hidden_states'] = hidden_states
                self._cache_storage['encoder_hidden_states'] = encoder_hidden_states
                self._cache_storage['rotary_emb'] = rotary_emb
                self._cache_storage['temb'] = temb
                self._cache_storage['timestep_proj'] = timestep_proj

        # Output norm & projections
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states.float()) * (1 + scale) + shift

        u_a_first_packed = fp32_module_forward(
            self.first_frame_proj_out, hidden_states[:, :num_first_tokens])
        u_a_rest_packed = fp32_module_forward(
            self.proj_out, hidden_states[:, num_first_tokens:])

        output_first_packed = self.asymflow_velocity(
            u_a_first_packed, x_t_first_packed, calibration,
            proj_buffer=self.first_frame_proj_buffer)
        output_rest_packed = self.asymflow_velocity(
            u_a_rest_packed, x_t_rest_packed, calibration)

        output_first = self.unpatchify(
            self.unpack(output_first_packed, 1, post_patch_height, post_patch_width),
            self.first_frame_patch_size)
        output_rest = self.unpatchify(
            self.unpack(output_rest_packed, post_patch_num_frames - 1, post_patch_height, post_patch_width),
            self.patch_size)

        return torch.cat([output_first, output_rest], dim=2)


@MODULES.register_module()
class AsymWanTransformer3DModel(_AsymWanTransformer3DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_linear_proj=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            freeze_quant_config=None,
            attention_backend_config=None,
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

        self.init_weights(pretrained, pretrained_linear_proj)

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
            cast_dtype(self, getattr(torch, torch_dtype), keep_fp32_keys=self._keep_in_fp32_modules)

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

        if freeze_quant_config is not None:
            # the 8-bit kernels engage only for fp16/bf16 activations, which the frozen
            # trunk produces only when its weights are natively 16-bit (not fp32 + autocast)
            assert torch_dtype in ('bfloat16', 'float16'), \
                'freeze_quant_config requires torch_dtype bfloat16 or float16'
            # the output heads run through fp32_module_forward and must stay unquantized
            freeze_quant_config['exclude_keys'] = (
                ['self.proj_out', 'self.first_frame_proj_out'] + list(freeze_quant_config.get('exclude_keys', [])))
            quantize_frozen_linears(self, **freeze_quant_config)

        if attention_backend_config is not None:
            set_attention_backends(self, **attention_backend_config)

        if checkpointing:
            self.enable_gradient_checkpointing()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    @staticmethod
    def _project_patch_embed_weight(patch_embed_weight, in_proj_weight, patch_size, channels):
        # Rewrite the Conv3d patch embedding as explicit patchify(p_t, p_h, p_w, c) + linear,
        # apply the pretrained projector in that basis, then convert the linear weight back
        # to conv storage (c, p_t, p_h, p_w).
        linear_weight = patch_embed_weight.permute(0, 2, 3, 4, 1).reshape(patch_embed_weight.size(0), -1)
        expected_shape = (linear_weight.size(1), channels * math.prod(patch_size))
        if tuple(in_proj_weight.shape) != expected_shape:
            raise RuntimeError(
                f'in_proj shape mismatch: expected {expected_shape}, got {tuple(in_proj_weight.shape)}. '
                'The pretrained linear projector likely does not match the latent/pixel patch sizes for this model.'
            )
        return (linear_weight @ in_proj_weight).reshape(
            patch_embed_weight.size(0), *patch_size, channels
        ).permute(
            0, 4, 1, 2, 3
        )

    def init_weights(
            self,
            pretrained=None,
            pretrained_linear_proj=None):
        if pretrained is None:
            return

        logger = get_root_logger()
        checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if self.secondary_temb_start_layer is not None:
            secondary_temb_state_dict = dict()
            for k, v in state_dict.items():
                if k.startswith('condition_embedder.time'):
                    secondary_k = 'secondary_' + k.removeprefix('condition_embedder.')
                    if secondary_k not in state_dict:
                        secondary_temb_state_dict[secondary_k] = v.clone()
            state_dict.update(secondary_temb_state_dict)

        if pretrained_linear_proj is not None:
            linear_proj_state_dict = _load_cached_checkpoint(
                pretrained_linear_proj, map_location='cpu', logger=logger)
            dtype = state_dict['patch_embedding.weight'].dtype
            proj_mat_key = 'proj_mat_p{}x{}x{}'.format(*self.patch_size)
            if proj_mat_key in linear_proj_state_dict:
                proj_mat = linear_proj_state_dict[proj_mat_key]  # (patch_dim, basis_rank)
            else:
                proj_mat = self.proj_buffer  # padded identity init
            first_frame_proj_mat_key = 'first_frame_proj_mat_p{}x{}x{}'.format(*self.first_frame_patch_size)
            if first_frame_proj_mat_key in linear_proj_state_dict:
                # (first_frame_patch_dim, basis_rank)
                first_frame_proj_mat = linear_proj_state_dict[first_frame_proj_mat_key]
            else:
                first_frame_proj_mat = self.first_frame_proj_buffer  # padded identity init
            if 'patch_embedding.weight' in state_dict:
                pe_weight = state_dict['patch_embedding.weight']  # (inner_dim, z_dim, lp_t, lp_h, lp_w)
                if self.input_bottleneck:
                    # the runtime proj_buffer projection replaces the pretrained projector
                    in_proj_weight = first_frame_in_proj_weight = torch.eye(self.basis_rank, dtype=dtype)
                else:
                    in_proj_weight = proj_mat.T.to(dtype)
                    first_frame_in_proj_weight = first_frame_proj_mat.T.to(dtype)
                state_dict['patch_embedding.weight'] = self._project_patch_embed_weight(
                    pe_weight,
                    in_proj_weight,
                    self.patch_size,
                    self.patch_embedding.in_channels)
                state_dict['first_frame_patch_embedding.weight'] = self._project_patch_embed_weight(
                    pe_weight,
                    first_frame_in_proj_weight,
                    self.first_frame_patch_size,
                    self.first_frame_patch_embedding.in_channels)
                state_dict['first_frame_patch_embedding.bias'] = state_dict['patch_embedding.bias'].clone()
            if 'proj_out.weight' in state_dict and 'proj_out.bias' in state_dict:
                out_weight = state_dict['proj_out.weight']  # pretrained, (basis_rank, inner_dim)
                out_bias = state_dict['proj_out.bias']  # pretrained, (basis_rank, )
                out_proj_weight = proj_mat.to(dtype)  # (patch_dim, basis_rank)
                state_dict['proj_out.weight'] = out_proj_weight @ out_weight
                state_dict['proj_out.bias'] = (
                    out_proj_weight @ out_bias.unsqueeze(-1)).squeeze(-1)
                first_frame_out_proj_weight = first_frame_proj_mat.to(dtype)  # (first_frame_patch_dim, basis_rank)
                state_dict['first_frame_proj_out.weight'] = first_frame_out_proj_weight @ out_weight
                state_dict['first_frame_proj_out.bias'] = (
                    first_frame_out_proj_weight @ out_bias.unsqueeze(-1)).squeeze(-1)
                orthogonal_lift_key = 'orthogonal_lift_p{}x{}x{}'.format(*self.patch_size)
                if orthogonal_lift_key in linear_proj_state_dict:
                    state_dict['proj_out.weight'] = (
                        state_dict['proj_out.weight']
                        - linear_proj_state_dict[orthogonal_lift_key].to(dtype))
                orthogonal_lift_bias_key = 'orthogonal_lift_bias_p{}x{}x{}'.format(*self.patch_size)
                if orthogonal_lift_bias_key in linear_proj_state_dict:
                    state_dict['proj_out.bias'] = (
                        state_dict['proj_out.bias']
                        - linear_proj_state_dict[orthogonal_lift_bias_key].to(dtype))
                first_frame_orthogonal_lift_key = 'first_frame_orthogonal_lift_p{}x{}x{}'.format(
                    *self.first_frame_patch_size)
                if first_frame_orthogonal_lift_key in linear_proj_state_dict:
                    state_dict['first_frame_proj_out.weight'] = (
                        state_dict['first_frame_proj_out.weight']
                        - linear_proj_state_dict[first_frame_orthogonal_lift_key].to(dtype))
                first_frame_orthogonal_lift_bias_key = 'first_frame_orthogonal_lift_bias_p{}x{}x{}'.format(
                    *self.first_frame_patch_size)
                if first_frame_orthogonal_lift_bias_key in linear_proj_state_dict:
                    state_dict['first_frame_proj_out.bias'] = (
                        state_dict['first_frame_proj_out.bias']
                        - linear_proj_state_dict[first_frame_orthogonal_lift_bias_key].to(dtype))
            state_dict['proj_buffer'] = proj_mat
            state_dict['first_frame_proj_buffer'] = first_frame_proj_mat

        load_full_state_dict(self, state_dict, logger=logger, assign=True)

    def _forward_impl(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
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
            forward_fn = self._forward_impl if self._compiled_forward is None else self._compiled_forward
            return forward_fn(
                x_t=hidden_states.to(dtype),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_image=None,  # CLIP image conditioning is not implemented
                **kwargs)
