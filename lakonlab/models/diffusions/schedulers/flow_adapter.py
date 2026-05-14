# Copyright (c) 2026 Hansheng Chen

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import diffusers

from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin


@dataclass
class FlowWrapperSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowAdapterScheduler(SchedulerMixin, ConfigMixin):

    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            use_dynamic_shifting=False,
            dynamic_shifting_type='exp',
            base_seq_len=256,
            max_seq_len=4096,
            base_logshift=0.5,
            max_logshift=1.15,
            terminal_sigma=None,
            base_scheduler='UniPCMultistep',
            eps=1e-4,
            **kwargs):

        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_train_timesteps, dtype=np.float32, endpoint=False))
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps
        alphas = 1 - self.sigmas

        base_scheduler_class = getattr(diffusers.schedulers, base_scheduler + 'Scheduler', None)

        if base_scheduler_class is None:
            raise AttributeError(f'Cannot find base_scheduler [{base_scheduler}].')
        if base_scheduler in ['EulerDiscrete', 'EulerAncestralDiscrete']:
            assert kwargs.get('prediction_type', 'epsilon') == 'epsilon'
            kwargs['prediction_type'] = 'epsilon'
            self.scales = ((alphas ** 2 + self.sigmas ** 2) / (
                    1 + (self.sigmas / alphas.clamp(min=self.config.eps)) ** 2)).sqrt()
        elif base_scheduler in [
                'UniPCMultistep', 'DPMSolverSinglestep', 'DPMSolverMultistep', 'DEISMultistep', 'SASolver']:
            self.scales = torch.ones_like(alphas)
            assert kwargs.get('prediction_type', 'flow_prediction') == 'flow_prediction'
            kwargs['prediction_type'] = 'flow_prediction'
            kwargs['use_flow_sigmas'] = True
        else:
            raise AttributeError(f'Unsupported base_scheduler [{base_scheduler}].')

        signatures = inspect.signature(base_scheduler_class).parameters.keys()
        if 'final_sigmas_type' in signatures:
            kwargs['final_sigmas_type'] = 'zero'
        if 'lower_order_final' in signatures:
            kwargs['lower_order_final'] = True

        self.base_scheduler = base_scheduler_class(
            num_train_timesteps=num_train_timesteps,
            **kwargs)
        self.base_scheduler.timesteps = self.timesteps
        if self.config.base_scheduler in ['EulerDiscrete', 'EulerAncestralDiscrete']:
            self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)
        elif self.config.base_scheduler in [
                'UniPCMultistep', 'DPMSolverSinglestep', 'DPMSolverMultistep', 'DEISMultistep', 'SASolver']:
            self.base_scheduler.sigmas = self.sigmas
        else:
            raise AttributeError(f'Unsupported base_scheduler [{self.config.base_scheduler}].')

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
            sigmas = np.linspace(1, 0, num_inference_steps, dtype=np.float32, endpoint=False)
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
        if self.config.base_scheduler in ['DEISMultistep', 'SASolver']:
            self.sigmas = torch.cat(
                [sigmas, torch.tensor([self.config.eps], dtype=torch.float32, device=sigmas.device)])
        else:
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        alphas = 1 - self.sigmas

        self.base_scheduler.set_timesteps(num_inference_steps, device=device)

        self.base_scheduler.timesteps = self.timesteps
        if self.config.base_scheduler in ['EulerDiscrete', 'EulerAncestralDiscrete']:
            self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)
            self.scales = ((alphas ** 2 + self.sigmas ** 2) / (
                    1 + (self.sigmas / alphas.clamp(min=self.config.eps)) ** 2)).sqrt()
        elif self.config.base_scheduler in [
                'UniPCMultistep', 'DPMSolverSinglestep', 'DPMSolverMultistep', 'DEISMultistep', 'SASolver']:
            self.base_scheduler.sigmas = self.sigmas.clamp(max=1 - self.config.eps)
            self.scales = torch.ones_like(alphas)
        else:
            raise AttributeError(f'Unsupported base_scheduler [{self.config.base_scheduler}].')

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
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
            eps=1e-6) -> Union[FlowWrapperSchedulerOutput, Tuple]:
        assert prediction_type in ['u', 'x0']

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        ori_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        alpha = 1 - sigma
        scale = self.scales[self.step_index]
        next_scale = self.scales[self.step_index + 1]

        if hasattr(self.base_scheduler, 'is_scale_input_called'):
            self.base_scheduler.is_scale_input_called = True
        kwargs = dict(return_dict=False)
        if generator is not None:
            kwargs.update(generator=generator)

        if self.config.base_scheduler in [
                'UniPCMultistep', 'DPMSolverSinglestep', 'DPMSolverMultistep', 'DEISMultistep', 'SASolver']:
            if prediction_type == 'u':
                model_output = model_output
            else:
                model_output = (sample - model_output) / sigma.clamp(min=eps)
        else:  # to epsilon
            if prediction_type == 'u':
                model_output = sample + alpha * model_output
            else:
                model_output = (sample - alpha * model_output) / sigma.clamp(min=eps)
        prev_sample = self.base_scheduler.step(
            model_output,
            timestep,
            sample / scale,
            **kwargs
        )[0] * next_scale

        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowWrapperSchedulerOutput(prev_sample=prev_sample)
