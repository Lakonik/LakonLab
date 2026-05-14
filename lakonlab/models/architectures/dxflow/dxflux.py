from typing import Optional

import torch

from accelerate import init_empty_weights
from diffusers.models import FluxTransformer2DModel
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


@MODULES.register_module()
class DXFluxTransformer2DModel(FluxTransformer2DModel):

    def __init__(
            self,
            n_grid=16,
            patch_size=2,
            in_channels=64,
            out_channels=None,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        out_channels = out_channels or in_channels
        with init_empty_weights():
            super().__init__(
                patch_size=1, in_channels=in_channels, out_channels=n_grid * out_channels, **kwargs)
        self.n_grid = n_grid
        self.patch_size = patch_size

        self.init_weights(pretrained, pretrained_adapter)

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

    def init_weights(self, pretrained=None, pretrained_adapter=None):
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            if 'proj_out.weight' in state_dict and \
                    state_dict['proj_out.weight'].size(0) == self.out_channels // self.n_grid:
                state_dict['proj_out.weight'] = state_dict['proj_out.weight'][None].expand(
                    self.n_grid, -1, -1).reshape(self.out_channels, -1)
            if 'proj_out.bias' in state_dict and \
                    state_dict['proj_out.bias'].size(0) == self.out_channels // self.n_grid:
                state_dict['proj_out.bias'] = state_dict['proj_out.bias'][None].expand(
                    self.n_grid, -1).reshape(self.out_channels)
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
            bs, k, c, h, w = latents.size()
            latents = latents.reshape(
                bs, k, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
        return latents

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

        output = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            pooled_projections=pooled_projections.to(dtype),
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
            **kwargs)[0]

        output = output.permute(0, 2, 1).reshape(bs, self.n_grid, self.out_channels // self.n_grid, h, w)
        return self.unpatchify(output)
