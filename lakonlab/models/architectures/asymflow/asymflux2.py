# Copyright (c) 2026 Hansheng Chen

from typing import Any, Optional, List

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from diffusers.models import ModelMixin  # noqa: F401
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel, Flux2PosEmbed, Flux2TransformerBlock, Flux2SingleTransformerBlock,
    Flux2TimestepGuidanceEmbeddings, Flux2Modulation)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.configuration_utils import register_to_config
from diffusers.utils import apply_lora_scale
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze, fp32_module_forward, cast_dtype
from lakonlab.ops import quantize_frozen_linears
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from .common import AsymFlowMixin


class _AsymFlux2Transformer2DModel(AsymFlowMixin, Flux2Transformer2DModel):

    # _keep_in_fp32_modules is also used by diffusers in quantization and module dtype detection
    # (modules in this list are excluded from dtype detection)
    _keep_in_fp32_modules = ['proj_buffer', 'scale_buffer']
    _cache_storage = dict()

    def get_cache_hidden_states(self):
        hidden_states = self._cache_storage['hidden_states']
        encoder_hidden_states = self._cache_storage.get('encoder_hidden_states', None)
        if encoder_hidden_states is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        return hidden_states

    def clear_cache(self):
        self._cache_storage.clear()

    @register_to_config
    def __init__(
            self,
            patch_size=16,
            in_channels: int = 3,
            out_channels: int | None = None,
            basis_rank: int = 128,
            input_bottleneck: bool = False,
            num_layers: int = 8,
            num_single_layers: int = 48,
            secondary_temb_start_single_layer: int | None = None,
            attention_head_dim: int = 128,
            num_attention_heads: int = 48,
            joint_attention_dim: int = 15360,
            timestep_guidance_channels: int = 256,
            mlp_ratio: float = 3.0,
            axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
            rope_theta: int = 2000,
            eps: float = 1e-6,
            sigma_min: float = 1e-4,
            num_timesteps=1,
            use_calibration: bool = True,
            guidance_embeds: bool = True):
        super(Flux2Transformer2DModel, self).__init__()

        self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        assert self.out_channels % self.in_channels == 0, "out_channels must be a multiple of in_channels"

        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        self.secondary_temb_start_single_layer = secondary_temb_start_single_layer
        if secondary_temb_start_single_layer is not None:
            assert 0 <= secondary_temb_start_single_layer < num_single_layers
            self.secondary_time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
                in_channels=timestep_guidance_channels,
                embedding_dim=self.inner_dim,
                bias=False,
                guidance_embeds=guidance_embeds,
            )
            self.secondary_single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(
            basis_rank if input_bottleneck else in_channels * (patch_size ** 2),
            self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(
            self.inner_dim, self.out_channels * (patch_size ** 2), bias=False
        )

        # 8. AsymFlow attributes and buffers
        self.basis_rank = basis_rank
        self.input_bottleneck = input_bottleneck
        self.sigma_min = sigma_min
        self.num_timesteps = num_timesteps
        self.init_asymflow_buffers(
            self.in_channels * (patch_size ** 2), self.basis_rank, use_calibration=use_calibration)

        self.gradient_checkpointing = False

    @staticmethod
    def patchify(latents, patch_size, pack_channels=True):
        bs, c, h, w = latents.size()
        latents = latents.reshape(
            bs, c, h // patch_size, patch_size, w // patch_size, patch_size
        ).permute(
            0, 1, 3, 5, 2, 4
        )
        if pack_channels:
            latents = latents.reshape(
                bs, c * patch_size * patch_size, h // patch_size, w // patch_size)
        else:
            latents = latents.reshape(
                bs, c, patch_size * patch_size, h // patch_size, w // patch_size)
        return latents

    @staticmethod
    def unpatchify(latents, patch_size, packed_channels=True):
        if packed_channels:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c // (patch_size * patch_size), patch_size, patch_size, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c // (patch_size * patch_size), h * patch_size, w * patch_size)
        else:
            bs, c, _, h, w = latents.size()
            latents = latents.reshape(
                bs, c, patch_size, patch_size, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c, h * patch_size, w * patch_size)
        return latents

    @staticmethod
    def pack(latents):
        bs, c, h, w = latents.shape
        latents = latents.reshape(bs, c, h * w).permute(0, 2, 1)
        return latents

    @staticmethod
    def unpack(latents, h, w):
        bs, _, c = latents.shape
        latents = latents.permute(0, 2, 1).reshape(bs, c, h, w)
        return latents

    @staticmethod
    def _prepare_latent_ids(latents):
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids.to(device=latents.device)

    @staticmethod
    def _prepare_condition_latent_ids(
            image_latents: List[torch.Tensor],
            scale: int = 10):
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            _, _, h, w = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(h), torch.arange(w), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0).expand(image_latents[0].size(0), -1, -1)

        return image_latent_ids.to(device=image_latents[0].device)

    def _get_rotary_emb(self, img_ids, txt_ids):
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
            img_ids = img_ids.float()
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
            return (
                torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
            self,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: List[torch.Tensor] | None = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: dict[str, Any] | None = None,
            cache_mode: Optional[str] = None,
            cache_config: Optional[dict] = None):
        x_t = self.patchify(x_t, self.patch_size)
        img_ids = self._prepare_latent_ids(x_t)

        bs, _, h, w = x_t.size()
        x_t_packed = self.pack(x_t)
        num_x_tokens = x_t_packed.size(1)
        packed_ndim = x_t_packed.dim()
        num_txt_tokens = encoder_hidden_states.shape[1]

        # scale and timestep calibration
        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)

        cache_after_block = None
        cache_after_single_block = None
        if cache_mode is not None:
            cache_after_block = cache_config.get('cache_after_block', None)
            cache_after_single_block = cache_config.get('cache_after_single_block', None)
            assert (cache_after_block is None) != (cache_after_single_block is None)
            if self.secondary_temb_start_single_layer is not None and cache_after_single_block is not None:
                assert cache_after_single_block < self.secondary_temb_start_single_layer

        load_dual_block_cache = cache_mode == 'load' and cache_after_block is not None
        load_single_block_cache = cache_mode == 'load' and cache_after_single_block is not None

        if cache_mode == 'load':
            hidden_states = self._cache_storage.pop('hidden_states')[-bs:]
            temb = self._cache_storage.pop('temb')[-bs:]
            single_stream_mod = self._cache_storage.pop('single_stream_mod')[-bs:]
            concat_rotary_emb = self._cache_storage.pop('concat_rotary_emb')
            if load_dual_block_cache:
                encoder_hidden_states = self._cache_storage.pop('encoder_hidden_states')[-bs:]
                double_stream_mod_img = self._cache_storage.pop('double_stream_mod_img')[-bs:]
                double_stream_mod_txt = self._cache_storage.pop('double_stream_mod_txt')[-bs:]

        else:
            if self.use_calibration:
                hidden_states = x_t_packed * calibration.k.to(x_t_packed.dtype)
            else:
                hidden_states = x_t_packed

            input_img_ids = img_ids
            if condition_latents is not None:
                condition_hidden_states = [self.patchify(z, self.patch_size) for z in condition_latents]
                condition_latent_ids = self._prepare_condition_latent_ids(condition_hidden_states)
                if self.use_calibration:
                    condition_hidden_states = [
                        self.pack(z) / calibration.s.to(z.dtype) for z in condition_hidden_states]
                else:
                    condition_hidden_states = [self.pack(z) for z in condition_hidden_states]
                hidden_states = torch.cat([hidden_states] + condition_hidden_states, dim=1)
                input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

            if self.input_bottleneck:
                hidden_states = hidden_states @ self.proj_buffer.to(hidden_states.dtype)
            hidden_states = self.x_embedder(hidden_states)

            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
            temb = self.time_guidance_embed(calibration.timestep.to(hidden_states.dtype) * 1000, guidance)

            double_stream_mod_img = self.double_stream_modulation_img(temb)
            double_stream_mod_txt = self.double_stream_modulation_txt(temb)
            single_stream_mod = self.single_stream_modulation(temb)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
            concat_rotary_emb = self._get_rotary_emb(input_img_ids, txt_ids)

        start_block = 0
        if load_dual_block_cache:
            start_block = cache_after_block + 1

        if not load_single_block_cache:
            for index_block in range(start_block, len(self.transformer_blocks)):
                block = self.transformer_blocks[index_block]
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        double_stream_mod_img,
                        double_stream_mod_txt,
                        concat_rotary_emb,
                        joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb_mod_img=double_stream_mod_img,
                        temb_mod_txt=double_stream_mod_txt,
                        image_rotary_emb=concat_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if cache_mode == 'save' and index_block == cache_after_block:
                    self._cache_storage['hidden_states'] = hidden_states
                    self._cache_storage['temb'] = temb
                    self._cache_storage['single_stream_mod'] = single_stream_mod
                    self._cache_storage['concat_rotary_emb'] = concat_rotary_emb
                    self._cache_storage['encoder_hidden_states'] = encoder_hidden_states
                    self._cache_storage['double_stream_mod_img'] = double_stream_mod_img
                    self._cache_storage['double_stream_mod_txt'] = double_stream_mod_txt

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        start_block = 0
        if load_single_block_cache:
            start_block = cache_after_single_block + 1

        for index_block in range(start_block, len(self.single_transformer_blocks)):
            # Update temb and single_stream_mod when we reach the specified block
            if index_block == self.secondary_temb_start_single_layer:
                temb = self.secondary_time_guidance_embed(calibration.timestep.to(hidden_states.dtype) * 1000, guidance)
                single_stream_mod = self.secondary_single_stream_modulation(temb)

            block = self.single_transformer_blocks[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # Save cache after the specified block
            if cache_mode == 'save' and index_block == cache_after_single_block:
                self._cache_storage['hidden_states'] = hidden_states
                self._cache_storage['temb'] = temb
                self._cache_storage['single_stream_mod'] = single_stream_mod
                self._cache_storage['concat_rotary_emb'] = concat_rotary_emb

        hidden_states = hidden_states[:, num_txt_tokens:num_txt_tokens + num_x_tokens]
        hidden_states = fp32_module_forward(self.norm_out, hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        p = self.patch_size
        p2 = p ** 2

        u_a_packed = fp32_module_forward(self.proj_out, hidden_states)

        output_packed = self.asymflow_velocity(
            u_a_packed.reshape(bs, seq_len * (self.out_channels // self.in_channels), self.in_channels * p2),
            x_t_packed.reshape(bs, seq_len, 1, self.in_channels * p2).expand(
                -1, -1, self.out_channels // self.in_channels, -1
            ).reshape(bs, seq_len * (self.out_channels // self.in_channels), self.in_channels * p2),
            calibration
        ).reshape(bs, seq_len, self.out_channels * p2)

        output = self.unpack(output_packed, h, w)
        output = self.unpatchify(output, self.patch_size)

        return output


@MODULES.register_module()
class AsymFlux2Transformer2DModel(_AsymFlux2Transformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            pretrained_adapter_scale=1.0,
            pretrained_linear_proj=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            freeze_quant_config=None,
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)

        self.init_weights(pretrained, pretrained_adapter, pretrained_adapter_scale, pretrained_linear_proj)

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
            # norm_out and proj_out run through fp32_module_forward and must stay unquantized
            freeze_quant_config['exclude_keys'] = (
                ['self.norm_out', 'self.proj_out'] + list(freeze_quant_config.get('exclude_keys', [])))
            quantize_frozen_linears(self, **freeze_quant_config)

        if checkpointing:
            self.enable_gradient_checkpointing()

    def init_weights(
            self,
            pretrained=None,
            pretrained_adapter=None,
            pretrained_adapter_scale=1.0,
            pretrained_linear_proj=None):
        if pretrained is None:
            return

        logger = get_root_logger()
        checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if self.secondary_temb_start_single_layer is not None:
            secondary_temb_state_dict = dict()
            for k, v in state_dict.items():
                if k.startswith(('time_guidance_embed', 'single_stream_modulation')):
                    secondary_k = 'secondary_' + k
                    if secondary_k not in state_dict:
                        secondary_temb_state_dict[secondary_k] = v.clone()
            state_dict.update(secondary_temb_state_dict)

        if pretrained_linear_proj is not None:
            linear_proj_state_dict = _load_cached_checkpoint(
                pretrained_linear_proj, map_location='cpu', logger=logger)
            dtype = state_dict['x_embedder.weight'].dtype
            p = self.patch_size
            proj_mat_key = f'proj_mat_p{p}'
            if proj_mat_key in linear_proj_state_dict:
                proj_mat = linear_proj_state_dict[proj_mat_key]  # (in_channels * p2, basis_rank)
            else:
                proj_mat = self.proj_buffer  # padded identity init
            if 'x_embedder.weight' in state_dict and not self.input_bottleneck:
                in_proj_weight = proj_mat.T.to(dtype)  # (basis_rank, in_channels * p2)
                state_dict['x_embedder.weight'] = state_dict['x_embedder.weight'] @ in_proj_weight
            if 'proj_out.weight' in state_dict:
                out_proj_weight = proj_mat.to(dtype)  # (in_channels * p2, basis_rank)
                state_dict['proj_out.weight'] = out_proj_weight @ state_dict['proj_out.weight']
                if f'orthogonal_lift_p{p}' in linear_proj_state_dict:
                    state_dict['proj_out.weight'] = (
                        state_dict['proj_out.weight']
                        - linear_proj_state_dict[f'orthogonal_lift_p{p}'].to(dtype))
            state_dict['proj_buffer'] = proj_mat
            if f'scale_p{p}' in linear_proj_state_dict:
                state_dict['scale_buffer'] = linear_proj_state_dict[f'scale_p{p}']

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
                self.fuse_lora(lora_scale=pretrained_adapter_scale, safe_fusing=True)
                self.unload_lora()
                self.requires_grad_(True)  # PEFT freezes the weights after fusing
        else:
            load_full_state_dict(self, state_dict, logger=logger, assign=True)

    @staticmethod
    def _prepare_text_ids(
            x: torch.Tensor,  # (B, L, D) or (L, D)
            t_coord: Optional[torch.Tensor] = None):
        """
        Copied from Diffusers
        """
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids).to(device=x.device)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            condition_latents: torch.Tensor | None = None,  # our wrapper supports only single condition latent for now
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        txt_ids = self._prepare_text_ids(encoder_hidden_states)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                x_t=hidden_states.to(dtype),
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                condition_latents=[condition_latents.to(dtype)] if condition_latents is not None else None,
                txt_ids=txt_ids,
                **kwargs)

        return output
