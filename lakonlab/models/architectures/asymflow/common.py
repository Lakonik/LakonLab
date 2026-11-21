# Copyright (c) 2026 Hansheng Chen

from typing import NamedTuple

import torch
import torch.nn.functional as F


class AsymFlowCalibration(NamedTuple):
    s: torch.Tensor | None
    k: torch.Tensor | None
    timestep: torch.Tensor
    sigma: torch.Tensor


class AsymFlowMixin:

    train_sigma_min = 1e-6

    def init_asymflow_buffers(self, patch_dim: int, basis_rank: int, use_calibration: bool = True):
        assert patch_dim >= basis_rank
        self.use_calibration = use_calibration
        self.register_buffer('proj_buffer', torch.eye(patch_dim, basis_rank))  # (patch_dim, basis_rank)
        self.proj_buffer.lakonlab_no_shard = True  # avoid FSDP sharding
        if use_calibration:
            self.register_buffer('scale_buffer', torch.tensor(1.0))
            self.scale_buffer.lakonlab_no_shard = True  # avoid FSDP sharding

    def asymflow_calibration(self, timestep, batch_size: int, ndim: int):
        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
            timestep = timestep.float()
            sigma = timestep / self.num_timesteps
            if self.use_calibration:
                s = self.scale_buffer.float()
                k = 1 / (s + (1 - s) * sigma)
                cal_timestep = timestep * k
                k = k.reshape(batch_size, *((ndim - 1) * [1]))
                return AsymFlowCalibration(
                    s=s,
                    k=k,
                    timestep=cal_timestep,
                    sigma=sigma.expand(batch_size).reshape(batch_size, *((ndim - 1) * [1])).float(),
                )
            else:
                return AsymFlowCalibration(
                    s=None,
                    k=None,
                    timestep=timestep,
                    sigma=sigma.expand(batch_size).reshape(batch_size, *((ndim - 1) * [1])).float(),
                )

    @staticmethod
    def orthogonal_decomposition(full_rank_state, proj_buffer):
        subspace = full_rank_state @ proj_buffer @ proj_buffer.T
        complement = full_rank_state - subspace
        return subspace, complement

    def asymflow_velocity(
            self,
            u_a_packed,
            x_t_packed,
            calibration: AsymFlowCalibration,
            proj_buffer=None):
        if proj_buffer is None:
            proj_buffer = self.proj_buffer
        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
            sigma_min = self.train_sigma_min if self.training else self.sigma_min
            u_a_packed = u_a_packed.float()
            x_t_packed = x_t_packed.float()
            proj_buffer = proj_buffer.float()
            # orthogonal decomposition
            u_a_subspace, u_a_complement = self.orthogonal_decomposition(u_a_packed, proj_buffer)
            x_t_subspace, x_t_complement = self.orthogonal_decomposition(x_t_packed, proj_buffer)
            sigma_clamped = calibration.sigma.clamp(min=sigma_min)
            if self.use_calibration:
                sk = calibration.s * calibration.k
                u_subspace = sk * u_a_subspace + (1 - sk) / sigma_clamped * x_t_subspace
                # orthogonal complement
                u_complement = (x_t_complement + calibration.s * u_a_complement) / sigma_clamped
            else:
                u_subspace = u_a_subspace
                # orthogonal complement
                u_complement = (x_t_complement + u_a_complement) / sigma_clamped
            # full velocity
            return u_subspace + u_complement
