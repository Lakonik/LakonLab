# Copyright (c) 2026 Hansheng Chen

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMapSDESchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMapSDEScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            h: Union[float, str] = 0.0,
            use_fp64: bool = False,
            shift: float = 1.0,
            use_dynamic_shifting=False,
            dynamic_shifting_type='exp',
            base_seq_len=256,
            max_seq_len=4096,
            base_logshift=0.5,
            max_logshift=1.15,
            final_step_size_scale=1.0,
            max_raw_t=1.0,
            min_raw_t=0.0):
        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_train_timesteps, dtype=np.float32, endpoint=False))
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def get_shift(self, seq_len=None):
        if self.config.use_dynamic_shifting and seq_len is not None:
            if self.config.dynamic_shifting_type == 'exp':
                m = (self.config.max_logshift - self.config.base_logshift
                     ) / (self.config.max_seq_len - self.config.base_seq_len)
                logshift = (seq_len - self.config.base_seq_len) * m + self.config.base_logshift
                if isinstance(logshift, torch.Tensor):
                    shift = torch.exp(logshift)
                else:
                    shift = np.exp(logshift)
            elif self.config.dynamic_shifting_type == 'sqrt':
                max_shift = np.exp(self.config.max_logshift)
                base_shift = np.exp(self.config.base_logshift)
                sqrt_max_seq_len = np.sqrt(self.config.max_seq_len)
                sqrt_base_seq_len = np.sqrt(self.config.base_seq_len)
                m = (max_shift - base_shift) / (sqrt_max_seq_len - sqrt_base_seq_len)
                shift = (np.sqrt(seq_len) - sqrt_base_seq_len) * m + base_shift
            else:
                raise ValueError(f'Unsupported dynamic_shifting_type [{self.config.dynamic_shifting_type}].')
        else:
            shift = self.config.shift
        return shift

    def warp_t(self, raw_t, seq_len=None):
        shift = self.get_shift(seq_len=seq_len)
        return shift * raw_t / (1 + (shift - 1) * raw_t)

    def unwarp_t(self, sigma_t, seq_len=None):
        shift = self.get_shift(seq_len=seq_len)
        return sigma_t / (shift + (1 - shift) * sigma_t)

    def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            sigmas: Optional[List[float]] = None,
            seq_len=None,
            device=None):
        if sigmas is None:
            assert num_inference_steps is not None, 'Either num_inference_steps or sigmas must be provided.'
            self.num_inference_steps = num_inference_steps
            sigmas = np.linspace(
                self.config.max_raw_t,
                (self.config.max_raw_t - self.config.min_raw_t) * self.config.final_step_size_scale / (
                    num_inference_steps - 1 + self.config.final_step_size_scale) + self.config.min_raw_t,
                num_inference_steps, dtype=np.float32)
        else:
            if num_inference_steps is not None:
                assert len(sigmas) == num_inference_steps
            self.num_inference_steps = len(sigmas)
            sigmas = np.array(sigmas, dtype=np.float32)

        sigmas = torch.from_numpy(sigmas).to(device).clamp(min=0)
        sigmas = self.warp_t(sigmas, seq_len=seq_len)

        self.timesteps = sigmas * self.config.num_train_timesteps
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        sigmas_dst, m = self.calculate_sigmas_dst(self.sigmas)
        self.timesteps_dst = sigmas_dst * self.config.num_train_timesteps
        self.m_vals = m

        self._step_index = None
        self._begin_index = None

    def calculate_sigmas_dst(self, sigmas, eps=1e-6):
        alphas = 1 - sigmas

        sigmas_src = sigmas[:-1]
        sigmas_to = sigmas[1:]
        alphas_src = alphas[:-1]
        alphas_to = alphas[1:]

        if self.config.h == 'inf':
            m = torch.zeros_like(sigmas_src)
        elif self.config.h == 0.0:
            m = torch.ones_like(sigmas_src)
        else:
            assert self.config.h > 0.0
            h2 = self.config.h * self.config.h
            m = (sigmas_to * alphas_src / (sigmas_src * alphas_to).clamp(min=eps)) ** h2

        sigmas_to_mul_m = sigmas_to * m
        sigmas_dst = sigmas_to_mul_m / (alphas_to + sigmas_to_mul_m).clamp(min=eps)

        return sigmas_dst, m

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True) -> Union[FlowMapSDESchedulerOutput, Tuple]:

        if isinstance(timestep, int) \
                or isinstance(timestep, torch.IntTensor) \
                or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    'Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to'
                    ' `EulerDiscreteScheduler.step()` is not supported. Make sure to pass'
                    ' one of the `scheduler.timesteps` as a timestep.'
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        ori_dtype = model_output.dtype
        solver_dtype = torch.float64 if self.config.use_fp64 else torch.float32
        model_output = model_output.to(solver_dtype)  # x_t_dst

        sigma_to = self.sigmas[self.step_index + 1].to(solver_dtype)
        alpha_to = 1 - sigma_to
        m = self.m_vals[self.step_index].to(solver_dtype)

        noise = randn_tensor(
            model_output.shape, dtype=solver_dtype, device=model_output.device, generator=generator)

        prev_sample = (alpha_to + sigma_to * m) * model_output + sigma_to * (1 - m.square()).clamp(min=0).sqrt() * noise

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMapSDESchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
