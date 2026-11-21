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
            dynamic_shifting_type='exp',
            base_seq_len=256,
            max_seq_len=4096,
            base_logshift=0.5,
            max_logshift=1.15,
            raw_t_bleed=(0.0, 0.0)):
        self.num_timesteps = num_timesteps
        self.shift = shift
        self.logit_normal_enable = logit_normal_enable
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.use_dynamic_shifting = use_dynamic_shifting
        self.dynamic_shifting_type = dynamic_shifting_type
        self.base_seq_len = base_seq_len
        self.max_seq_len = max_seq_len
        self.base_logshift = base_logshift
        self.max_logshift = max_logshift
        self.raw_t_bleed = raw_t_bleed

    def get_shift(self, seq_len=None):
        if self.use_dynamic_shifting and seq_len is not None:
            if self.dynamic_shifting_type == 'exp':
                m = (self.max_logshift - self.base_logshift
                     ) / (self.max_seq_len - self.base_seq_len)
                logshift = (seq_len - self.base_seq_len) * m + self.base_logshift
                if isinstance(logshift, torch.Tensor):
                    shift = torch.exp(logshift)
                else:
                    shift = np.exp(logshift)
            elif self.dynamic_shifting_type == 'sqrt':
                max_shift = np.exp(self.max_logshift)
                base_shift = np.exp(self.base_logshift)
                sqrt_max_seq_len = np.sqrt(self.max_seq_len)
                sqrt_base_seq_len = np.sqrt(self.base_seq_len)
                m = (max_shift - base_shift) / (sqrt_max_seq_len - sqrt_base_seq_len)
                shift = (np.sqrt(seq_len) - sqrt_base_seq_len) * m + base_shift
            else:
                raise ValueError(f'Unsupported dynamic_shifting_type [{self.dynamic_shifting_type}].')
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
        raw_t_min, raw_t_max = raw_t_range or (0.0, 1.0)

        if self.logit_normal_enable:
            if raw_t_min != 0.0 or raw_t_max != 1.0:
                raise NotImplementedError('Custom raw_t_range is not yet supported for logit-normal sampling.')
            t = torch.sigmoid(
                self.logit_normal_mean + self.logit_normal_std * torch.randn(
                    (batch_size, ), dtype=torch.float, device=device))
        else:
            t = torch.rand(
                (batch_size, ), dtype=torch.float, device=device
            ) * (raw_t_min - raw_t_max) + raw_t_max

        if any(self.raw_t_bleed):
            bleed_min, bleed_max = self.raw_t_bleed
            t = (t - raw_t_min) * (
                1 + (bleed_min + bleed_max) / (raw_t_max - raw_t_min)
            ) + raw_t_min - bleed_min
            t = t.clamp(raw_t_min, raw_t_max)

        if warp_t:
            t = self.warp_t(t, seq_len=seq_len)
        if scale_t:
            t = t * self.num_timesteps
        return t

    def __call__(self, batch_size, **kwargs):
        return self.sample(batch_size, **kwargs)
