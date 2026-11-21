# Copyright (c) 2026 Hansheng Chen

import torch

from accelerate import init_empty_weights
from peft import LoraConfig

from ...builder import MODULES
from ..diffusers.qwen import _QwenImageTransformer2DModelFix
from ..utils import flex_freeze
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger
from .common import AsymFlowMixin


class _AsymQwenImageTransformer2DModel(AsymFlowMixin, _QwenImageTransformer2DModelFix):

    def __init__(
            self,
            *args,
            patch_size=16,
            in_channels=3,
            basis_rank=64,
            num_timesteps=1,
            sigma_min=1e-4,
            use_calibration: bool = True,
            **kwargs):
        patch_dim = in_channels * (patch_size ** 2)
        super().__init__(
            *args,
            patch_size=1,
            in_channels=patch_dim,
            out_channels=patch_dim,
            **kwargs)
        self.register_to_config(
            patch_size=patch_size,
            in_channels=in_channels,
            basis_rank=basis_rank,
            num_timesteps=num_timesteps,
            sigma_min=sigma_min,
            use_calibration=use_calibration,
            **kwargs)

        self.patch_size = patch_size
        self.basis_rank = basis_rank
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.init_asymflow_buffers(patch_dim, basis_rank, use_calibration=use_calibration)

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

    def forward(
            self,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        x_t = self.patchify(x_t, self.patch_size)
        bs, _, h, w = x_t.size()
        x_t_packed = self.pack(x_t)
        packed_ndim = x_t_packed.dim()

        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)
        if self.use_calibration:
            hidden_states = x_t_packed * calibration.k.to(x_t_packed.dtype)
        else:
            hidden_states = x_t_packed

        img_shapes = [[(1, h, w)]]
        if encoder_hidden_states_mask is not None:
            keep_mask = encoder_hidden_states_mask.any(dim=0)
            encoder_hidden_states = encoder_hidden_states[:, keep_mask]
            encoder_hidden_states_mask = encoder_hidden_states_mask[:, keep_mask]

        u_a_packed = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=calibration.timestep,
            img_shapes=img_shapes,
            fp32_final_layer=True,
            **kwargs)

        output_packed = self.asymflow_velocity(u_a_packed, x_t_packed, calibration)

        output = self.unpack(output_packed, h, w)
        output = self.unpatchify(output, self.patch_size)

        return output


@MODULES.register_module()
class AsymQwenImageTransformer2DModel(_AsymQwenImageTransformer2DModel):

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
            freeze_exclude_autocast_dtype=None,
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
            dtype = state_dict['img_in.weight'].dtype
            p = self.patch_size
            proj_mat = linear_proj_state_dict[f'proj_mat_p{p}']  # (in_channels * p2, basis_rank)
            if 'img_in.weight' in state_dict:
                in_proj_weight = proj_mat.T.to(dtype)  # (basis_rank, in_channels * p2)
                state_dict['img_in.weight'] = state_dict['img_in.weight'] @ in_proj_weight
            if 'proj_out.weight' in state_dict and 'proj_out.bias' in state_dict:
                out_proj_weight = proj_mat.to(dtype)  # (in_channels * p2, basis_rank)
                state_dict['proj_out.bias'] = (
                    out_proj_weight @ state_dict['proj_out.bias'].unsqueeze(-1)
                ).squeeze(-1)
                state_dict['proj_out.weight'] = out_proj_weight @ state_dict['proj_out.weight']
                if f'orthogonal_lift_p{p}' in linear_proj_state_dict:
                    state_dict['proj_out.weight'] = (
                        state_dict['proj_out.weight']
                        - linear_proj_state_dict[f'orthogonal_lift_p{p}'].to(dtype))
                if f'orthogonal_lift_bias_p{p}' in linear_proj_state_dict:
                    state_dict['proj_out.bias'] = (
                        state_dict['proj_out.bias']
                        - linear_proj_state_dict[f'orthogonal_lift_bias_p{p}'].to(dtype))
            state_dict['proj_buffer'] = proj_mat
            if f'scale_p{p}' in linear_proj_state_dict:
                state_dict['scale_buffer'] = linear_proj_state_dict[f'scale_p{p}']

        load_full_state_dict(self, state_dict, logger=logger, assign=True)

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

    def forward(
            self,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = x_t.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            return super().forward(
                x_t=x_t,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                **kwargs)
