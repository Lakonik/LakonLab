# Copyright (c) 2026 Hansheng Chen

import contextlib
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel
try:
    from torch.distributed.fsdp import FSDPModule
except:
    FSDPModule = None

from lakonlab.utils import kai_zhang_clip_grad


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


@contextlib.contextmanager
def maybe_ddp_no_sync(modules, enabled=True):
    if not enabled:
        yield
        return

    seen = set()
    with contextlib.ExitStack() as stack:
        for module in modules:
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            if isinstance(module, DistributedDataParallel):
                stack.enter_context(module.no_sync())
        yield


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base class for all models in the training framework. Optionally supports:
    - Gradient accumulation
    - Gradient clipping
    """

    def step_optimizer(self, optimizer, loss_scaler, running_status):
        log_vars = dict()
        for k, v in optimizer.items():
            grad_clip = self.train_cfg.get(k + '_grad_clip', 0.0)
            grad_clip_begin_iter = self.train_cfg.get(k + '_grad_clip_begin_iter', 0)
            grad_clip_skip_ratio = self.train_cfg.get(k + '_grad_clip_skip_ratio', 0.0)
            skip_step = False
            if loss_scaler is not None:
                loss_scaler.unscale_(v)
            if grad_clip > 0.0 and running_status['iteration'] >= grad_clip_begin_iter:
                m = getattr(self, k)
                if isinstance(m, FullyShardedDataParallel):  # FSDP1
                    grad_norm = m.clip_grad_norm_(grad_clip)
                elif FSDPModule is not None and isinstance(m, FSDPModule):  # FSDP2
                    grad_norm = kai_zhang_clip_grad(m, grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
                if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()).item() or (
                        grad_clip_skip_ratio > 0 and grad_norm > grad_clip * grad_clip_skip_ratio):
                    grad_norm = float('nan')
                    v.zero_grad()
                    skip_step = True
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
        log_vars = dict()
        optimizer_modules = [getattr(self, k, None) for k in optimizer.keys()]

        bs = 0
        for grad_step_id in range(grad_accum_steps):
            with maybe_ddp_no_sync(
                    optimizer_modules, enabled=(grad_step_id < grad_accum_steps - 1)):
                log_vars_single, bs_single = train_minibatch_func(
                    data_splits[grad_step_id], loss_scaler, running_status)
            for k, v in log_vars_single.items():
                if k in log_vars:
                    log_vars[k] += float(v)
                else:
                    log_vars[k] = float(v)
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

        return log_vars, bs

    def train_step(self, data, optimizer, loss_scaler=None, running_status=None):
        for v in optimizer.values():
            v.zero_grad()

        _bs = guess_bs(data)
        grad_accum_batch_size = self.train_cfg.get('grad_accum_batch_size', None)
        if grad_accum_batch_size is not None and _bs is not None:
            grad_accum_batch_size = max(min(grad_accum_batch_size, _bs), 1)
            assert _bs % grad_accum_batch_size == 0, \
                f'Data batch size {_bs} is not divisible by `grad_accum_batch_size` {grad_accum_batch_size}.'
            grad_accum_steps = _bs // grad_accum_batch_size
        else:
            grad_accum_steps = 1
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
