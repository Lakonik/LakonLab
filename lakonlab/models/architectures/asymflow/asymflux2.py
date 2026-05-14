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
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from .common import AsymFlowMixin


class _AsymFlux2Transformer2DModel(AsymFlowMixin, Flux2Transformer2DModel):

    @register_to_config
    def __init__(
            self,
            patch_size=16,
            in_channels: int = 3,
            base_rank: int = 128,
            num_layers: int = 8,
            num_single_layers: int = 48,
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
            guidance_embeds: bool = True):
        super(Flux2Transformer2DModel, self).__init__()

        self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_channels = in_channels

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

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels * (patch_size ** 2), self.inner_dim, bias=False)
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
        self.base_rank = base_rank
        self.sigma_min = sigma_min
        self.num_timesteps = num_timesteps
        self.init_asymflow_buffers(self.in_channels * (patch_size ** 2), self.base_rank)

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
            joint_attention_kwargs: dict[str, Any] | None = None):
        x_t = self.patchify(x_t, self.patch_size)
        img_ids = self._prepare_latent_ids(x_t)

        bs, _, h, w = x_t.size()
        x_t_packed = self.pack(x_t)
        num_x_tokens = x_t_packed.size(1)
        packed_ndim = x_t_packed.dim()

        # scale and timestep calibration
        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)
        hidden_states = x_t_packed * calibration.k.to(x_t_packed.dtype)

        input_img_ids = img_ids
        if condition_latents is not None:
            condition_hidden_states = [self.patchify(z, self.patch_size) for z in condition_latents]
            condition_latent_ids = self._prepare_condition_latent_ids(condition_hidden_states)
            condition_hidden_states = [self.pack(z) / calibration.s for z in condition_hidden_states]
            hidden_states = torch.cat([hidden_states] + condition_hidden_states, dim=1)
            input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

        hidden_states = self.x_embedder(hidden_states)

        num_txt_tokens = encoder_hidden_states.shape[1]

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(calibration.timestep.to(hidden_states.dtype) * 1000, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        concat_rotary_emb = self._get_rotary_emb(input_img_ids, txt_ids)

        for block in self.transformer_blocks:
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

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
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

        hidden_states = hidden_states[:, num_txt_tokens:num_txt_tokens + num_x_tokens]
        hidden_states = self.norm_out(hidden_states, temb)

        u_a_packed = self.proj_out(hidden_states)

        output_packed = self.asymflow_velocity(u_a_packed, x_t_packed, calibration)

        output = self.unpack(output_packed.to(hidden_states.dtype), h, w)
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
            pretrained_linear_proj=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(*args, **kwargs)

        self.init_weights(pretrained, pretrained_adapter, pretrained_linear_proj)

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

    def init_weights(
            self,
            pretrained=None,
            pretrained_adapter=None,
            pretrained_linear_proj=None):
        if pretrained is None:
            return

        logger = get_root_logger()
        checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if pretrained_linear_proj is not None:
            linear_proj_state_dict = _load_cached_checkpoint(
                pretrained_linear_proj, map_location='cpu', logger=logger)
            dtype = state_dict['x_embedder.weight'].dtype
            p = self.patch_size
            proj_mat = linear_proj_state_dict[f'proj_mat_p{p}']  # (in_channels * p2, base_rank)
            if 'x_embedder.weight' in state_dict:
                in_proj_weight = proj_mat.T.to(dtype)  # (base_rank, in_channels * p2)
                state_dict['x_embedder.weight'] = state_dict['x_embedder.weight'] @ in_proj_weight
            if 'proj_out.weight' in state_dict:
                out_proj_weight = proj_mat.to(dtype)  # (in_channels * p2, base_rank)
                state_dict['proj_out.weight'] = out_proj_weight @ state_dict['proj_out.weight']
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
                self.fuse_lora()
                self.unload_lora()
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
                x_t=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                condition_latents=[condition_latents.to(dtype)] if condition_latents is not None else None,
                txt_ids=txt_ids,
                **kwargs)

        return output
