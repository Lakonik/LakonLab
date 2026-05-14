# Copyright (c) 2026 Hansheng Chen

import ast
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_H_EXPR_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}
_H_EXPR_UNARY_OPS = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}
_H_EXPR_FUNCS = {
    'abs': abs,
    'clip': lambda x, min=None, max=None: min if x < min else max if x > max else x,
    'clamp': lambda x, min=None, max=None: min if x < min else max if x > max else x,
    'cos': math.cos,
    'exp': math.exp,
    'log': math.log,
    'max': max,
    'maximum': max,
    'min': min,
    'minimum': min,
    'pow': pow,
    'sin': math.sin,
    'sqrt': math.sqrt,
    'tan': math.tan,
}
_H_EXPR_CONSTS = {
    'e': math.e,
    'inf': math.inf,
    'pi': math.pi,
}


def _compile_h_expression(expr):
    parsed = ast.parse(expr, mode='eval')

    def _eval(node, context):
        if isinstance(node, ast.Expression):
            return _eval(node.body, context)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            if node.id in _H_EXPR_CONSTS:
                return _H_EXPR_CONSTS[node.id]
            raise ValueError(f'Unknown h expression symbol [{node.id}].')
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _H_EXPR_BIN_OPS:
                raise ValueError(f'Unsupported h expression binary op [{op_type.__name__}].')
            return _H_EXPR_BIN_OPS[op_type](_eval(node.left, context), _eval(node.right, context))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _H_EXPR_UNARY_OPS:
                raise ValueError(f'Unsupported h expression unary op [{op_type.__name__}].')
            return _H_EXPR_UNARY_OPS[op_type](_eval(node.operand, context))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError('Only direct function calls are supported in h expressions.')
            func_name = node.func.id
            if func_name not in _H_EXPR_FUNCS:
                raise ValueError(f'Unsupported h expression function [{func_name}].')
            args = [_eval(arg, context) for arg in node.args]
            kwargs = {kw.arg: _eval(kw.value, context) for kw in node.keywords}
            return _H_EXPR_FUNCS[func_name](*args, **kwargs)
        raise ValueError(f'Unsupported h expression node [{type(node).__name__}].')

    return lambda sigma: float(_eval(parsed, dict(sigma=float(sigma))))


@dataclass
class FlowSDESchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowSDEScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            h: Union[float, str] = 1.0,
            solver_type: str = 'canonical',
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
        self._h_expr_fn = None
        if isinstance(h, str):
            self._h_expr_fn = _compile_h_expression(h)
        assert solver_type in ['canonical', 'euler-maruyama']

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

    def _get_h(self, sigma):
        if self._h_expr_fn is None:
            return self.config.h
        return self._h_expr_fn(float(sigma))

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            prediction_type='u',
            eps=1e-6) -> Union[FlowSDESchedulerOutput, Tuple]:
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

        if float(sigma_to) == 0.0:  # only affects euler-maruyama, canonical does not have noise when sigma_to is zero
            noise = torch.zeros_like(model_output)
        else:
            noise = randn_tensor(
                model_output.shape, dtype=solver_dtype, device=model_output.device, generator=generator)

        h = self._get_h(sigma)

        if self.config.solver_type == 'canonical':
            alpha = 1 - sigma
            alpha_to = 1 - sigma_to
            if prediction_type == 'u':
                x0 = sample - sigma * model_output
                epsilon = sample + alpha * model_output
            else:
                x0 = model_output
                epsilon = (sample - alpha * x0) / sigma.clamp(min=eps)
            if math.isinf(h):
                m = torch.zeros_like(sigma)
            elif h == 0.0:
                m = torch.ones_like(sigma)
            else:
                assert h > 0.0
                h2 = h * h
                m = (sigma_to * alpha / (sigma * alpha_to).clamp(min=eps)) ** h2
            prev_sample = alpha_to * x0 + sigma_to * (m * epsilon + (1 - m.square()).clamp(min=0).sqrt() * noise)

        elif self.config.solver_type == 'euler-maruyama':
            dt = sigma_to - sigma
            alpha = 1 - sigma
            if prediction_type == 'u':
                u = model_output
            else:
                u = (sample - model_output) / sigma.clamp(min=eps)
            h2 = h * h
            dt_h2_over_alpha = torch.where(alpha < eps, dt, dt * h2 / alpha)
            prev_sample = \
                (1 + dt_h2_over_alpha) * sample + (1 + h2) * dt * u \
                + (-2 * dt_h2_over_alpha * sigma).clamp(min=0).sqrt() * noise

        else:
            raise ValueError(f'Unsupported solver type [{self.config.solver_type}].')

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowSDESchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
