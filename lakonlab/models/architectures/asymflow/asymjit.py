# Copyright (c) 2026 Hansheng Chen

import torch

from ...builder import MODULES
from ..utils import flex_freeze
from ..lightningdit.jit import _JiT
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger
from .common import AsymFlowMixin


@MODULES.register_module()
class AsymJiT(AsymFlowMixin, _JiT):
    def __init__(
            self,
            *args,
            patch_size: int = 16,
            in_channels: int = 3,
            base_rank: int = 32,
            sigma_min: float = 1e-4,
            num_timesteps=1,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_linear_proj=None,
            torch_dtype='float32',
            compile_forward=False,
            compile_kwargs=dict(
                mode='reduce-overhead',
                fullgraph=True,
                dynamic=False),
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            **kwargs):
        super().__init__(
            *args,
            patch_size=patch_size,
            in_channels=in_channels,
            **kwargs)

        self.base_rank = base_rank
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.init_asymflow_buffers(in_channels * (patch_size ** 2), base_rank)

        self.init_weights(pretrained, pretrained_linear_proj)

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
                exclude_autocast_dtype=freeze_exclude_autocast_dtype,
            )

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    def init_weights(self, pretrained=None, pretrained_linear_proj=None):
        logger = get_root_logger()

        if pretrained is not None:
            state_dict = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)

            if pretrained_linear_proj is not None:
                raise NotImplementedError

            load_full_state_dict(self, state_dict, logger=logger)

        else:
            super().init_weights()

            if pretrained_linear_proj is not None:
                linear_proj_state_dict = _load_cached_checkpoint(
                    pretrained_linear_proj, map_location='cpu', logger=logger)
                p = self.patch_size
                proj_mat = linear_proj_state_dict[f'proj_mat_p{p}']
                self.proj_buffer.copy_(proj_mat)
                self._init_projected_bottleneck(proj_mat)
                if f'scale_p{p}' in linear_proj_state_dict:
                    scale = linear_proj_state_dict[f'scale_p{p}']
                    self.scale_buffer.copy_(scale)

    def _init_projected_bottleneck(self, proj_mat):
        # JiT's first conv is the actual patch-to-bottleneck projection.  Keep
        # the original scratch init, but reserve the first base_rank channels
        # for the low-rank basis used by the analytic output term.
        p = self.patch_size
        patch_dim = self.in_channels * (p ** 2)
        expected_shape = (patch_dim, self.base_rank)
        if tuple(proj_mat.shape) != expected_shape:
            raise RuntimeError(
                f'proj_mat shape mismatch: expected {expected_shape}, got {tuple(proj_mat.shape)}.')
        if self.x_embedder.proj1.out_channels < self.base_rank:
            raise RuntimeError(
                f'JiT bottleneck_dim must be >= base_rank, got '
                f'{self.x_embedder.proj1.out_channels} and {self.base_rank}.')

        proj1_weight = self.x_embedder.proj1.weight
        proj1_basis = proj_mat.T.reshape(
            self.base_rank, p, p, self.in_channels
        ).permute(
            0, 3, 1, 2
        )
        # match xavier init gain
        init_gain = (2 * patch_dim / (patch_dim + self.x_embedder.proj1.out_channels)) ** 0.5
        proj1_basis = proj1_basis.mul(init_gain).to(dtype=proj1_weight.dtype, device=proj1_weight.device)

        proj1_weight.data[:self.base_rank].copy_(proj1_basis)

    @staticmethod
    def patchify(latents, patch_size, pack_channels=True):
        bs, c, h, w = latents.shape
        latents = latents.reshape(
            bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        if pack_channels:  # patch dim before channel dim (ImageNet DiT convention)
            latents = latents.permute(
                0, 3, 5, 1, 2, 4
            ).reshape(bs, patch_size * patch_size * c, h // patch_size, w // patch_size)
        else:  # channel dim before patch dim (for consistency with other models)
            latents = latents.permute(
                0, 1, 3, 5, 2, 4
            ).reshape(bs, c, patch_size * patch_size, h // patch_size, w // patch_size)
        return latents

    @staticmethod
    def unpatchify(latents, patch_size, packed_channels=True):
        if packed_channels:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, patch_size, patch_size, c // (patch_size * patch_size), h, w
            ).permute(
                0, 3, 4, 1, 5, 2
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

    def _forward_impl(self, x_t, timestep, class_labels=None, **kwargs):
        bs, _, h, w = x_t.size()
        ndim = x_t.dim()
        x_t_packed = self.pack(self.patchify(x_t, self.patch_size))
        packed_ndim = x_t_packed.dim()

        # scale and timestep calibration
        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)
        hidden_states = x_t * calibration.k.reshape(bs, *((ndim - 1) * [1])).to(x_t.dtype)

        u_a_packed = super().forward(
            hidden_states,
            calibration.timestep,
            class_labels=class_labels,
            unpatchify_output=False,
            **kwargs)

        output_packed = self.asymflow_velocity(u_a_packed, x_t_packed, calibration)

        output = self.unpatchify(self.unpack(
            output_packed.to(x_t.dtype),
            h // self.patch_size, w // self.patch_size
        ), self.patch_size)
        return output

    def forward(self, x_t, timestep, class_labels=None, **kwargs):
        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = x_t.dtype
        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            if self._compiled_forward is not None:
                return self._compiled_forward(x_t.to(dtype), timestep, class_labels=class_labels, **kwargs)
            return self._forward_impl(x_t.to(dtype), timestep, class_labels=class_labels, **kwargs)
