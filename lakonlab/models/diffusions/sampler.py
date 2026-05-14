# Copyright (c) 2026 Hansheng Chen

import numpy as np
import torch

from ..builder import MODULES


@MODULES.register_module()
class ContinuousTimeStepSampler:
    def __init__(
            self,
            num_timesteps,
            shift=1.0,
            logit_normal_enable=False,
            logit_normal_mean=0.0,
            logit_normal_std=1.0,
            use_dynamic_shifting=False,
            base_seq_len=256,
            max_seq_len=4096,
            base_logshift=0.5,
            max_logshift=1.15):
        self.num_timesteps = num_timesteps
        self.shift = shift
        self.logit_normal_enable = logit_normal_enable
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_seq_len = base_seq_len
        self.max_seq_len = max_seq_len
        self.base_logshift = base_logshift
        self.max_logshift = max_logshift

    def get_shift(self, seq_len=None):
        if self.use_dynamic_shifting and seq_len is not None:
            m = (self.max_logshift - self.base_logshift) / (self.max_seq_len - self.base_seq_len)
            logshift = (seq_len - self.base_seq_len) * m + self.base_logshift
            if isinstance(logshift, torch.Tensor):
                shift = torch.exp(logshift)
            else:
                shift = np.exp(logshift)
        else:
            shift = self.shift
        return shift

    def warp_t(self, raw_t, seq_len=None):
        shift = self.get_shift(seq_len=seq_len)
        return shift * raw_t / (1 + (shift - 1) * raw_t)

    def unwarp_t(self, sigma_t, seq_len=None):
        shift = self.get_shift(seq_len=seq_len)
        return sigma_t / (shift + (1 - shift) * sigma_t)

    def sample(self, batch_size, warp_t=True, scale_t=True, seq_len=None,
               raw_t_range=None, device=None):
        if self.logit_normal_enable:
            if raw_t_range is not None:
                assert raw_t_range[0] == 0.0 and raw_t_range[1] == 1.0
            t = torch.sigmoid(
                self.logit_normal_mean + self.logit_normal_std * torch.randn(
                    (batch_size, ), dtype=torch.float, device=device))
        else:
            if raw_t_range is not None:
                assert isinstance(raw_t_range, (tuple, list)) and len(raw_t_range) == 2
                t = torch.rand(
                    (batch_size, ), dtype=torch.float, device=device
                ) * (raw_t_range[0] - raw_t_range[1]) + raw_t_range[1]
            else:
                t = 1 - torch.rand((batch_size, ), dtype=torch.float, device=device)
        if warp_t:
            t = self.warp_t(t, seq_len=seq_len)
        if scale_t:
            t = t * self.num_timesteps
        return t

    def __call__(self, batch_size, **kwargs):
        return self.sample(batch_size, **kwargs)
