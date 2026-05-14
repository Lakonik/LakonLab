# Copyright (c) 2026 Hansheng Chen

from abc import ABCMeta, abstractmethod
import torch


class BasePolicy(metaclass=ABCMeta):

    @abstractmethod
    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        pass

    @abstractmethod
    def detach(self):
        pass

    def integrate(
            self,
            x_t_start: torch.Tensor,  # (B, C, *, H, W)
            sigma_t_start: torch.Tensor,  # (B, 1, *, 1, 1)
            raw_t_start: torch.Tensor,  # (B, )
            raw_t_end: torch.Tensor,  # (B, )
            timestep_sampler,
            seq_len=None,
            total_substeps: int = 128):
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()
        raw_t_start = raw_t_start.reshape(num_batches, *((ndim - 1) * [1]))
        raw_t_end = raw_t_end.reshape(num_batches, *((ndim - 1) * [1]))

        delta_raw_t = raw_t_start - raw_t_end
        num_substeps = (delta_raw_t * total_substeps).ceil().to(torch.long).clamp(min=1)
        substep_size = delta_raw_t / num_substeps
        max_num_substeps = num_substeps.max()

        raw_t = raw_t_start
        sigma_t = sigma_t_start
        x_t = x_t_start

        for substep_id in range(max_num_substeps.item()):
            u = self.pi(x_t, sigma_t)

            raw_t_minus = (raw_t - substep_size).clamp(min=0)
            sigma_t_minus = timestep_sampler.warp_t(raw_t_minus, seq_len=seq_len)
            x_t_minus = x_t + u * (sigma_t_minus - sigma_t)

            active_mask = num_substeps > substep_id
            x_t = torch.where(active_mask, x_t_minus, x_t)
            sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
            raw_t = torch.where(active_mask, raw_t_minus, raw_t)

        x_t_end = x_t
        sigma_t_end = sigma_t
        t_end = sigma_t_end.flatten() * timestep_sampler.num_timesteps
        return x_t_end, sigma_t_end, t_end
