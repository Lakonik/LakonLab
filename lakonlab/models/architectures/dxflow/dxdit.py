from typing import Optional

import torch
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict

from ...builder import MODULES
from ..diffusers.dit import _DiTTransformer2DModelMod
from ..utils import flex_freeze
from lakonlab.utils import get_root_logger


@MODULES.register_module()
class DXDiTTransformer2DModel(_DiTTransformer2DModelMod):

    def __init__(
            self,
            *args,
            n_grid=16,
            in_channels=4,
            out_channels=None,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            **kwargs):
        out_channels = out_channels or in_channels
        super().__init__(
            *args, in_channels=in_channels, out_channels=n_grid * out_channels, **kwargs)
        self.n_grid = n_grid

        self.init_weights(pretrained)

        if autocast_dtype is not None:
            assert torch_dtype == 'float32'
        self.autocast_dtype = autocast_dtype

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
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # expand the output channels
            p2 = self.config.patch_size * self.config.patch_size
            ori_out_channels = p2 * self.out_channels // self.n_grid
            if 'proj_out_2.weight' in state_dict:
                # if this is GMDiT V1 model with 1 Gaussian
                if state_dict['proj_out_2.weight'].size(0) == p2 * (
                        self.out_channels // self.n_grid + 1):
                    state_dict['proj_out_2.weight'] = state_dict['proj_out_2.weight'].reshape(
                        p2, self.out_channels // self.n_grid + 1, -1
                    )[:, :-1].reshape(ori_out_channels, -1)
                if state_dict['proj_out_2.weight'].size(0) == ori_out_channels:
                    state_dict['proj_out_2.weight'] = state_dict['proj_out_2.weight'].reshape(
                        p2, 1, self.out_channels // self.n_grid, -1
                    ).expand(-1, self.n_grid, -1, -1).reshape(
                        self.n_grid * ori_out_channels, -1)
            if 'proj_out_2.bias' in state_dict:
                # if this is GMDiT V1 model with 1 Gaussian
                if state_dict['proj_out_2.bias'].size(0) == p2 * (
                        self.out_channels // self.n_grid + 1):
                    state_dict['proj_out_2.bias'] = state_dict['proj_out_2.bias'].reshape(
                        p2, self.out_channels // self.n_grid + 1
                    )[:, :-1].reshape(ori_out_channels)
                if state_dict['proj_out_2.bias'].size(0) == ori_out_channels:
                    state_dict['proj_out_2.bias'] = state_dict['proj_out_2.bias'].reshape(
                        p2, 1, self.out_channels // self.n_grid
                    ).expand(-1, self.n_grid, -1).reshape(
                        self.n_grid * ori_out_channels)
            load_state_dict(self, state_dict, logger=logger)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = hidden_states.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                hidden_states.to(dtype),
                timestep=timestep,
                class_labels=class_labels,
                **kwargs)
        bs, _, h, w = output.shape
        output = output.reshape(bs, self.n_grid, self.out_channels // self.n_grid, h, w)
        return output
