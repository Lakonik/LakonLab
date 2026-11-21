from typing import Optional

import torch

from accelerate import init_empty_weights
from peft import LoraConfig

from ...builder import MODULES
from ..utils import flex_freeze
from ..asymflow.asymflux2 import _AsymFlux2Transformer2DModel
from lakonlab.utils import get_root_logger
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict


@MODULES.register_module()
class DXAsymFlux2Transformer2DModel(_AsymFlux2Transformer2DModel):

    def __init__(
            self,
            n_grid=1,
            p_order=1,
            in_channels=3,
            out_channels=None,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            pretrained_adapter_scale=1.0,
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
        assert n_grid > 0 and p_order > 0
        assert n_grid == 1 or p_order == 1, \
            "Only one of n_grid and p_order can be greater than 1."

        out_channels = out_channels or in_channels
        with init_empty_weights():
            super().__init__(
                in_channels=in_channels, out_channels=n_grid * p_order * out_channels, **kwargs)
        self.n_grid = n_grid
        self.p_order = p_order

        self.init_weights(
            pretrained,
            pretrained_adapter,
            pretrained_adapter_scale)

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

    def init_weights(self, pretrained=None, pretrained_adapter=None, pretrained_adapter_scale=1.0):
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            c_p2 = self.out_channels * self.patch_size ** 2

            if pretrained_adapter is not None:
                adapter_state_dict = _load_cached_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                lora_state_dict = dict()
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k] = v
                    else:
                        state_dict[k] = v

                # expand the output channels
                if 'proj_out.weight' in state_dict and \
                        state_dict['proj_out.weight'].size(0) == c_p2 // self.n_grid:
                    state_dict['proj_out.weight'] = state_dict['proj_out.weight'][None].expand(
                        self.n_grid * self.p_order, -1, -1).reshape(c_p2, -1)

                load_full_state_dict(self, state_dict, logger=logger, assign=True)
                if len(lora_state_dict) > 0:
                    self.load_lora_adapter(lora_state_dict, prefix=None)
                    self.fuse_lora(lora_scale=pretrained_adapter_scale, safe_fusing=True)
                    self.unload_lora()
                    self.requires_grad_(True)  # PEFT freezes the weights after fusing
            else:
                # expand the output channels
                if 'proj_out.weight' in state_dict and \
                        state_dict['proj_out.weight'].size(0) == c_p2 // self.n_grid:
                    state_dict['proj_out.weight'] = state_dict['proj_out.weight'][None].expand(
                        self.n_grid * self.p_order, -1, -1).reshape(c_p2, -1)

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
            condition_latents: Optional[torch.Tensor] = None,
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

        bs, c, h, w = output.size()
        extra_dim = max(self.n_grid, self.p_order)
        output = output.reshape(bs, extra_dim, c // extra_dim, h, w)
        return output
