from typing import Optional, Dict, Any

import numpy as np
import torch

from accelerate import init_empty_weights
from diffusers.models import FluxTransformer2DModel as _FluxTransformer2DModel
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import load_checkpoint, _load_cached_checkpoint


@MODULES.register_module()
class FluxTransformer2DModel(_FluxTransformer2DModel):

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_lora=None,
            pretrained_lora_scale=1.0,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        with init_empty_weights():
            super().__init__(patch_size=1, *args, **kwargs)
        self.patch_size = patch_size

        self.init_weights(pretrained, pretrained_lora, pretrained_lora_scale)

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

    def init_weights(self, pretrained=None, pretrained_lora=None, pretrained_lora_scale=1.0):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(
                self, pretrained,
                map_location='cpu', strict=False, logger=logger, assign=True, use_cache=True)
            if pretrained_lora is not None:
                if not isinstance(pretrained_lora, (list, tuple)):
                    assert isinstance(pretrained_lora, str)
                    pretrained_lora = [pretrained_lora]
                if not isinstance(pretrained_lora_scale, (list, tuple)):
                    assert isinstance(pretrained_lora_scale, (int, float))
                    pretrained_lora_scale = [pretrained_lora_scale]
                for pretrained_lora_single, pretrained_lora_scale_single in zip(pretrained_lora, pretrained_lora_scale):
                    lora_state_dict = _load_cached_checkpoint(
                        pretrained_lora_single, map_location='cpu', logger=logger)
                    self.load_lora_adapter(lora_state_dict)
                    self.fuse_lora(lora_scale=pretrained_lora_scale_single)
                    self.unload_lora()

    @staticmethod
    def _prepare_latent_image_ids(height, width, device, dtype):
        """
        Copied from Diffusers
        """
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

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

    def unpatchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(
                bs, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
        return latents

    def _forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            controlnet_blocks_repeat: bool = False):
        """
        Fixed the image_rotary_emb dtype issue in Diffusers for FSDP2
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            assert joint_attention_kwargs is None or joint_attention_kwargs.get('scale', None) is None

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        image_rotary_emb = tuple([x.to(hidden_states.dtype) for x in image_rotary_emb])

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return output

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            mask: Optional[torch.Tensor] = None,
            masked_image_latents: Optional[torch.Tensor] = None,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        dtype = hidden_states.dtype
        device = hidden_states.device
        hidden_states = hidden_states.reshape(bs, c, h * w).permute(0, 2, 1)
        img_ids = self._prepare_latent_image_ids(
            h, w, device, dtype)
        txt_ids = img_ids.new_zeros((encoder_hidden_states.shape[-2], 3))

        #  Flux fill
        if mask is not None and masked_image_latents is not None:
            hidden_states = torch.cat(
                (hidden_states, masked_image_latents.to(dtype=dtype), mask.to(dtype=dtype)), dim=-1)

        output = self._forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            pooled_projections=pooled_projections.to(dtype),
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs)

        output = output.permute(0, 2, 1).reshape(bs, self.out_channels, h, w)
        return self.unpatchify(output)
