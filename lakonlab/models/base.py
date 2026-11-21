# Copyright (c) 2026 Hansheng Chen

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FSDPModule

from lakonlab.utils import fsdp2_clip_grad, reduce_mean
from lakonlab.parallel import maybe_ddp_no_sync


def chunk_list(input_list, chunks):
    """
    Splits a list into a specified number of chunks, similar to torch.chunk.

    Args:
        input_list (list): The list to be chunked.
        chunks (int): The desired number of chunks.

    Returns:
        list: A list of sub-lists (chunks).
    """
    list_len = len(input_list)
    assert list_len % chunks == 0
    chunk_size = list_len // chunks

    result_chunks = []

    for i in range(chunks):
        result_chunks.append(input_list[i * chunk_size:(i + 1) * chunk_size])

    return result_chunks


def chunk_data_dict(data, chunks):
    data_splits = [dict() for _ in range(chunks)]
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            assert v.size(0) % chunks == 0
            v_splits = torch.chunk(v, chunks, dim=0)
        elif isinstance(v, list):
            v_splits = chunk_list(v, chunks)
        elif isinstance(v, dict):
            v_splits = chunk_data_dict(v, chunks)
        else:
            raise TypeError(
                f'Unsupported data type {type(v)} for gradient accumulation. '
                'Only torch.Tensor and list are supported.')
        for grad_step_id in range(chunks):
            data_splits[grad_step_id][k] = v_splits[grad_step_id]
    return data_splits


def guess_bs(data):
    for v in data.values():
        if isinstance(v, torch.Tensor):
            return v.size(0)
        elif isinstance(v, list):
            return len(v)
        elif isinstance(v, dict):
            bs = guess_bs(v)
            if bs is not None:
                return bs
    return None


def get_grad_accum_steps(train_cfg, batch_size):
    grad_accum_num_steps = train_cfg.get('grad_accum_num_steps', None)
    grad_accum_batch_size = train_cfg.get('grad_accum_batch_size', None)
    if grad_accum_num_steps is not None and grad_accum_batch_size is not None:
        raise ValueError('Only one of grad_accum_num_steps and grad_accum_batch_size can be set.')

    if batch_size is None:
        return 1

    if grad_accum_num_steps is not None:
        grad_accum_num_steps = int(grad_accum_num_steps)
        assert grad_accum_num_steps > 0, '`grad_accum_num_steps` must be positive.'
        assert batch_size % grad_accum_num_steps == 0, \
            f'Data batch size {batch_size} is not divisible by `grad_accum_num_steps` {grad_accum_num_steps}.'
        return grad_accum_num_steps

    if grad_accum_batch_size is not None:
        grad_accum_batch_size = max(min(grad_accum_batch_size, batch_size), 1)
        assert batch_size % grad_accum_batch_size == 0, \
            f'Data batch size {batch_size} is not divisible by `grad_accum_batch_size` {grad_accum_batch_size}.'
        return batch_size // grad_accum_batch_size

    return 1


@dataclass(kw_only=True, slots=True)
class MultistepState:
    """Typed state threaded through multistep training (see `train_fwd_bwd`).

    Static plan fields are sealed after construction; everything else is
    dynamic. Models with extra per-step carries declare them as typed fields
    on a subclass (slots make undeclared/typo'd fields fail loudly).

    Static plan (sealed):
        num_steps: Fixed number of steps, or None for model-driven dynamic
            termination via `terminate`. Declaring it enables
            gradient-accumulation DDP no_sync on non-final steps.
        detachable: Whether each step's loss is backwarded (and detached)
            immediately. Non-detachable losses are accumulated and backwarded
            once after the loop, which requires an armed (synced) forward, so
            no_sync is never used in that case.

    Dynamic:
        step_id: Current step index; owned and advanced by `train_fwd_bwd`.
        terminate: Only for dynamic-termination models (num_steps=None), which
            set it on their final step. Fixed-step models must not touch it
            (enforced); their termination is derived from num_steps.
        fakes / reals: Complete per-step discriminator inputs. Their structure
            is defined by the receiving discriminator; `train_fwd_bwd` passes
            them through, detaches tensor leaves as needed, and clears them
            after each step. None disables GAN losses for that step.

    FSDP2 treats this dataclass as one pytree leaf. Tensors whose backward must
    arm wrapper hooks must therefore not be stored here.
    """
    num_steps: int | None = None
    detachable: bool = True
    step_id: int = 0
    terminate: bool = False
    fakes: Any = None
    reals: Any = None
    # init=True keeps dataclasses.replace() (used by FSDP1 traversal) valid.
    _sealed: bool = field(default=False, repr=False)

    _STATIC_FIELDS = ('num_steps', 'detachable')

    def __post_init__(self):
        if self.num_steps is not None and self.terminate:
            raise ValueError(
                '`terminate` is only for dynamic-termination models '
                '(num_steps=None); fixed-step termination is derived from '
                '`num_steps`.')
        self._sealed = True

    @property
    def finished(self):
        if self.num_steps is not None:
            return self.step_id >= self.num_steps
        return self.terminate

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            if name in self._STATIC_FIELDS:
                raise AttributeError(
                    f'MultistepState.{name} is a static plan field and cannot '
                    'be modified after initialization.')
            if name == 'terminate' and self.num_steps is not None:
                raise AttributeError(
                    '`terminate` is only for dynamic-termination models '
                    '(num_steps=None); fixed-step termination is derived from '
                    '`num_steps`.')
        # zero-arg super() is broken inside slots=True dataclasses (the
        # decorator rebuilds the class, orphaning the __class__ closure)
        object.__setattr__(self, name, value)


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base class for all models in the training framework. Optionally supports:
    - Gradient accumulation
    - Gradient clipping
    """

    def step_optimizer(self, optimizer, loss_scaler, running_status=None):
        log_this_step = (
            running_status is None or
            running_status['iteration'] % self.train_cfg.get('log_interval', 1) == 0)
        log_vars = dict()
        for k, v in optimizer.items():
            grad_clip = self.train_cfg.get(k + '_grad_clip', 0.0)
            grad_clip_begin_iter = self.train_cfg.get(k + '_grad_clip_begin_iter', 0)
            grad_clip_skip_ratio = self.train_cfg.get(k + '_grad_clip_skip_ratio', 0.0)
            grad_clip_check_finite = self.train_cfg.get(k + '_grad_clip_check_finite', True)
            skip_step = False
            if loss_scaler is not None:
                loss_scaler.unscale_(v)
            if grad_clip > 0.0 and running_status['iteration'] >= grad_clip_begin_iter:
                m = getattr(self, k)
                if isinstance(m, FullyShardedDataParallel):  # FSDP1
                    grad_norm = m.clip_grad_norm_(grad_clip)
                elif FSDPModule is not None and isinstance(m, FSDPModule):  # FSDP2
                    grad_norm = fsdp2_clip_grad(m, grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
                if (grad_clip_check_finite and torch.logical_or(grad_norm.isnan(), grad_norm.isinf()).item()) or (
                        grad_clip_skip_ratio > 0 and grad_norm > grad_clip * grad_clip_skip_ratio):
                    grad_norm = float('nan')
                    v.zero_grad()
                    skip_step = True
                if log_this_step:
                    log_vars.update({k + '_grad_norm': float(grad_norm)})
            if not skip_step:
                if loss_scaler is None:
                    v.step()
                else:
                    loss_scaler.step(v)
        return log_vars

    @abstractmethod
    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        """Training forward/backward inside a single gradient accumulation minibatch.
        """

    def train_grad_accum(
            self, train_minibatch_func, data_splits, optimizer, grad_accum_steps, loss_scaler=None, running_status=None):
        log_this_step = (
            running_status is None or
            running_status['iteration'] % self.train_cfg.get('log_interval', 1) == 0)
        log_vars = dict()
        optimizer_modules = [getattr(self, k, None) for k in optimizer.keys()]

        bs = 0
        for grad_step_id in range(grad_accum_steps):
            with maybe_ddp_no_sync(
                    optimizer_modules, enabled=(grad_step_id < grad_accum_steps - 1)):
                log_vars_single, bs_single = train_minibatch_func(
                    data_splits[grad_step_id], loss_scaler, running_status)
            if log_this_step:
                for k, v in log_vars_single.items():
                    if isinstance(v, torch.Tensor):
                        assert not v.requires_grad
                    if k in log_vars:
                        log_vars[k] = log_vars[k] + v
                    else:
                        log_vars[k] = v
            bs += bs_single

        if grad_accum_steps > 1:
            norm_factor = 1 / grad_accum_steps
            log_vars = {k: v * norm_factor for k, v in log_vars.items()}
            for v in optimizer.values():
                for group in v.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            p.grad._values().mul_(norm_factor)
                        else:
                            p.grad.mul_(norm_factor)

        log_vars_optim = self.step_optimizer(optimizer, loss_scaler, running_status)
        log_vars.update(log_vars_optim)

        if self.train_cfg.get('reduce_log', False):
            tensor_keys = sorted(
                k for k, v in log_vars.items() if isinstance(v, torch.Tensor))
            if tensor_keys:
                values = reduce_mean(torch.stack([
                    log_vars[k].reshape(()) for k in tensor_keys]))
                log_vars.update(zip(tensor_keys, values))

        log_vars = {k: float(v) for k, v in log_vars.items()}

        return log_vars, bs

    def train_step(self, data, optimizer, loss_scaler=None, running_status=None):
        for v in optimizer.values():
            v.zero_grad()

        _bs = guess_bs(data)
        grad_accum_steps = get_grad_accum_steps(self.train_cfg, _bs)
        data_splits = chunk_data_dict(data, grad_accum_steps)

        log_vars, bs = self.train_grad_accum(
            self.train_minibatch,
            data_splits,
            optimizer,
            grad_accum_steps,
            loss_scaler=loss_scaler,
            running_status=running_status)
        if _bs is not None:
            assert bs == _bs

        outputs_dict = dict(log_vars=log_vars, num_samples=bs)

        return outputs_dict
