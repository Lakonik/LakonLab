# Copyright (c) 2026 Hansheng Chen

import torch
from .base import BasePolicy


class DXPolicy(BasePolicy):
    """DX policy. The number of grid points N is inferred from the denoising output.

    Note: segment_size and shift are intrinsic parameters of the DX policy. For elastic inference (i.e., changing
    the number of function evaluations or noise schedule at test time), these parameters should be kept unchanged.

    Args:
        denoising_output (torch.Tensor): The output of the denoising model. Shape (B, N, C, H, W) or (B, N, C, T, H, W).
        x_t_src (torch.Tensor): The initial noisy sample. Shape (B, C, H, W) or (B, C, T, H, W).
        sigma_t_src (torch.Tensor): The initial noise level. Shape (B,).
        segment_size (float): The size of each DX policy time segment. Defaults to 1.0.
        shift (float): The shift parameter for the DX policy noise schedule. Defaults to 1.0.
        mode (str): Either 'grid' or 'polynomial' mode for calculating x_0. Defaults to 'grid'.
        eps (float): A small value to avoid numerical issues. Defaults to 1e-4.
    """

    def __init__(
            self,
            denoising_output: torch.Tensor,
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            segment_size: float = 1.0,
            shift: float = 1.0,
            mode: str = 'grid',
            eps: float = 1e-4):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps

        assert mode in ['grid', 'polynomial']
        self.mode = mode

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        self.raw_t_dst = (self.raw_t_src - segment_size).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)

        self.denoising_output_x_0 = self._u_to_x_0(
            denoising_output, self.x_t_src, self.sigma_t_src)

    def _unwarp_t(self, sigma_t):
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(denoising_output, x_t, sigma_t):
        x_0 = x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output
        return x_0

    @staticmethod
    def _interpolate(x, t):
        """
        Args:
            x (torch.Tensor): (B, N, *)
            t (torch.Tensor): (B, *) in [0, 1]

        Returns:
            torch.Tensor: (B, *)
        """
        n = x.size(1)
        if n < 2:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        t0t1 = torch.stack([t0, t1], dim=1)  # (B, 2, *)
        x0x1 = torch.gather(x, dim=1, index=t0t1.expand(-1, -1, *x.shape[2:]))
        x_interp = (t1 - t) * x0x1[:, 0] + (t - t0) * x0x1[:, 1]
        return x_interp

    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        if self.mode == 'grid':
            x_0 = self._interpolate(
                self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        elif self.mode == 'polynomial':
            p_order = self.denoising_output_x_0.size(1)
            diff_t = self.raw_t_src - raw_t  # (B, 1, 1, 1)
            basis = torch.stack(
                [diff_t ** i for i in range(p_order)], dim=1)  # (B, N, 1, 1, 1)
            x_0 = torch.sum(basis * self.denoising_output_x_0, dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        u = (x_t - x_0) / sigma_t.clamp(min=self.eps)
        return u

    def copy(self):
        new_policy = DXPolicy.__new__(DXPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.ndim = self.ndim
        new_policy.shift = self.shift
        new_policy.eps = self.eps
        new_policy.mode = self.mode
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.raw_t_src = self.raw_t_src
        new_policy.raw_t_dst = self.raw_t_dst
        new_policy.segment_size = self.segment_size
        new_policy.denoising_output_x_0 = self.denoising_output_x_0
        return new_policy

    def detach_(self):
        self.denoising_output_x_0 = self.denoising_output_x_0.detach()
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()
