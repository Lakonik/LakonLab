# Copyright (c) 2026 Hansheng Chen

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowEulerODESchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowEulerODEScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            use_fp64: bool = False,
            shift: float = 1.0,
            use_dynamic_shifting=False,
            dynamic_shifting_type='exp',
            base_seq_len=256,
            max_seq_len=4096,
            base_logshift=0.5,
            max_logshift=1.15,
            terminal_sigma=None,
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

    def stretch_to_terminal(self, sigma):
        one_minus_sigma = 1 - sigma
        stretched_sigma = 1 - (one_minus_sigma * (1 - self.config.terminal_sigma) / one_minus_sigma[-1])
        return stretched_sigma

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
                self.config.max_raw_t, self.config.min_raw_t,
                num_inference_steps, dtype=np.float32, endpoint=False)
        else:
            if num_inference_steps is not None:
                assert len(sigmas) == num_inference_steps
            self.num_inference_steps = len(sigmas)
            sigmas = np.array(sigmas, dtype=np.float32)

        sigmas = torch.from_numpy(sigmas)
        shift = self.get_shift(seq_len=seq_len)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.terminal_sigma is not None:
            sigmas = self.stretch_to_terminal(sigmas)

        self.timesteps = (sigmas * self.config.num_train_timesteps).to(device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

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
            return_dict: bool = True,
            prediction_type='u',
            eps=1e-6) -> Union[FlowEulerODESchedulerOutput, Tuple]:
        assert prediction_type in ['u', 'x0']

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
        sample = sample.to(solver_dtype)
        model_output = model_output.to(solver_dtype)

        sigma = self.sigmas[self.step_index].to(solver_dtype)
        sigma_to = self.sigmas[self.step_index + 1].to(solver_dtype)

        if prediction_type == 'u':
            derivative = model_output
        else:
            derivative = (sample - model_output) / sigma

        dt = sigma_to - sigma
        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowEulerODESchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
