# Copyright (c) 2026 Hansheng Chen

import gc
import contextlib
import functools
import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from functools import partial

import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor
try:
    from torch.distributed.fsdp import FSDPModule
except:
    FSDPModule = None

from mmcv.parallel import is_module_wrapper
from peft.tuners.lora import LoraLayer


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if is_module_wrapper(obj):
            obj = obj.module
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if is_module_wrapper(pre):
        pre = pre.module
    return setattr(pre, post, val)


def rhasattr(obj, attr):
    return rgetattr(obj, attr, None) is not None


def rdelattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if is_module_wrapper(pre):
        pre = pre.module
    return delattr(pre, post)


class module_requires_grad:
    def __init__(self, module, requires_grad=True):
        self.module = module
        self.requires_grad = requires_grad
        self.prev = []

    def __enter__(self):
        self.prev = []
        for p in self.module.parameters():
            self.prev.append(p.requires_grad)
            p.requires_grad = self.requires_grad
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for p, r in zip(self.module.parameters(), self.prev):
            p.requires_grad = r


class module_eval:
    def __init__(self, module):
        self.module = module
        self.prev = None

    def __enter__(self):
        self.prev = self.module.training
        self.module.train(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.train(self.prev)


def all_frozen(modules):
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                return False
    return True


def tie_untrained_root_params(tgt_module, src_module):
    for key, val in src_module._parameters.items():
        tgt_param = tgt_module._parameters.get(key, None)
        if (val is not None) \
                and (tgt_param is not None) \
                and (not val.requires_grad) \
                and (not tgt_param.requires_grad):
            val.lakonlab_no_shard = True  # hint for FSDP to not shard this parameter
            tgt_module._parameters[key] = val


def tie_untrained_submodules(tgt_module, src_module, tie_tgt_lora_base_layer=False):
    for key, src_submodule in src_module._modules.items():
        if key in tgt_module._modules:
            if (tie_tgt_lora_base_layer
                    and isinstance(tgt_module._modules[key], LoraLayer)
                    and not isinstance(src_submodule, LoraLayer)):
                if all_frozen((tgt_module._modules[key]._modules['base_layer'], src_submodule)) and not (
                        getattr(tgt_module._modules[key]._modules['base_layer'], 'lakonlab_no_tie', False)
                        or getattr(src_submodule, 'lakonlab_no_tie', False)):
                    tgt_module._modules[key]._modules['base_layer'] = src_submodule
                else:
                    tie_untrained_submodules(
                        tgt_module._modules[key]._modules['base_layer'], src_submodule, tie_tgt_lora_base_layer)
                    tie_untrained_root_params(
                        tgt_module._modules[key]._modules['base_layer'], src_submodule)
            else:
                if all_frozen((tgt_module._modules[key], src_submodule)) and not (
                        getattr(tgt_module._modules[key], 'lakonlab_no_tie', False)
                        or getattr(src_submodule, 'lakonlab_no_tie', False)):
                    tgt_module._modules[key] = src_submodule
                else:
                    tie_untrained_submodules(
                        tgt_module._modules[key], src_submodule, tie_tgt_lora_base_layer)
                    tie_untrained_root_params(
                        tgt_module._modules[key], src_submodule)


def _shares_storage(a, b):
    if a is b:
        return True
    if a.is_meta or b.is_meta:
        return False
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False
    return a.data_ptr() == b.data_ptr()


def untie_all_parameters(tgt_module, src_module, recursive=True, untie_tgt_lora_base_layer=False):
    for key, val in src_module._parameters.items():
        if val is not None \
                and tgt_module._parameters.get(key, None) is not None \
                and _shares_storage(tgt_module._parameters[key], val):
            tgt_param = tgt_module._parameters[key]
            tgt_module._parameters[key] = torch.nn.Parameter(
                tgt_param.detach().clone(), requires_grad=tgt_param.requires_grad)
    for key, val in src_module._buffers.items():
        if val is not None \
                and tgt_module._buffers.get(key, None) is not None \
                and _shares_storage(tgt_module._buffers[key], val):
            tgt_module._buffers[key] = tgt_module._buffers[key].clone()
    if recursive:
        for key, val in src_module._modules.items():
            if key in tgt_module._modules:
                if untie_tgt_lora_base_layer \
                        and isinstance(tgt_module._modules[key], LoraLayer) \
                        and not isinstance(val, LoraLayer):
                    untie_all_parameters(
                        tgt_module._modules[key]._modules['base_layer'], val,
                        recursive=True, untie_tgt_lora_base_layer=untie_tgt_lora_base_layer)
                else:
                    untie_all_parameters(
                        tgt_module._modules[key], val,
                        recursive=True, untie_tgt_lora_base_layer=untie_tgt_lora_base_layer)


def clone_params(tgt_module, src_module, recursive=True):
    """Clone parameters and buffers from src_module to tgt_module (sharing the same structure).
    Tied parameters/buffers are not cloned. Used for EMA model initialization.
    """
    for key, val in src_module._parameters.items():
        if (val is not None) \
                and (val is not tgt_module._parameters[key]):
            tgt_module._parameters[key] = torch.nn.Parameter(
                val.detach().clone(), requires_grad=val.requires_grad)
    for key, val in src_module._buffers.items():
        if val is not tgt_module._buffers[key]:
            tgt_module._buffers[key] = val.clone()
    if recursive:
        for key, val in src_module._modules.items():
            clone_params(
                tgt_module._modules[key], val, recursive)


@torch.no_grad()
def kai_zhang_clip_grad(model, max_norm: float) -> torch.Tensor:
    """
    https://github.com/Kai-46/minFM/blob/385568691b021aa4888269dafa671900daf17cf6/utils/clip_grad.py#L9
    """
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    shard_size, replicate_factor = 1, world_size
    if FSDPModule is not None and isinstance(model, FSDPModule):
        shard_size = model._get_fsdp_state()._fsdp_param_group.mesh_info.shard_mesh_size
        replicate_factor = world_size // shard_size

    # Separate DTensor and non-DTensor parameters
    all_param_grads = []
    dtensor_param_grads = []
    regular_param_grads = []

    for p in model.parameters():
        if (not p.requires_grad) or (p.grad is None):
            continue

        if isinstance(p.grad.data, DTensor):
            local_p_grad = p.grad.data.to_local()
            dtensor_param_grads.append(local_p_grad.ravel())
        else:
            local_p_grad = p.grad.data
            regular_param_grads.append(local_p_grad.ravel())

        all_param_grads.append(local_p_grad)

    if not all_param_grads:
        return torch.tensor(0.0)

    # Compute local square sum for each group separately
    local_sq_sum = torch.tensor(0.0, device=all_param_grads[0].device)

    if dtensor_param_grads:
        dtensor_sq_sum = (torch.cat(dtensor_param_grads, dim=0) ** 2).float().sum()
        local_sq_sum = local_sq_sum + dtensor_sq_sum

    if regular_param_grads:
        regular_sq_sum = (torch.cat(regular_param_grads, dim=0) ** 2).float().sum()
        local_sq_sum = local_sq_sum + regular_sq_sum / shard_size

    # Single all-reduce operation
    global_sq_sum = local_sq_sum.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_sq_sum, op=dist.ReduceOp.SUM)
    global_sq_sum = global_sq_sum / replicate_factor

    total_norm = global_sq_sum.sqrt()

    # Only apply clipping when exceeding threshold
    if total_norm.item() > max_norm:
        clip_factor = max_norm / total_norm.item()
        torch._foreach_mul_(all_param_grads, clip_factor)

    return total_norm


def materialize_meta_states(module, device=None):
    if device is None:
        device = torch.get_default_device()
    assert device != torch.device('meta'), 'Please specify a non-meta device.'
    for mod in module.modules():
        for name, p in list(mod._parameters.items()):
            if p is not None and p.is_meta:
                new = torch.empty_like(p, device=device)
                new = torch.nn.Parameter(new, requires_grad=p.requires_grad)
                mod._parameters[name] = new
        for name, b in list(mod._buffers.items()):
            if b is not None and b.is_meta:
                new = torch.empty_like(b, device=device)
                mod._buffers[name] = new
    return module


@contextlib.contextmanager
def gc_context(enable=False):
    prev_enabled = gc.isenabled()
    if enable:
        gc.enable()
    else:
        gc.disable()
    try:
        yield
    finally:
        if prev_enabled:
            gc.enable()
        else:
            gc.disable()


def first_tensor_device(x: Any) -> Optional[torch.device]:
    """
    Walk a nested structure (dict/list/tuple/etc.) and return the device of the
    first torch.Tensor found, or None if no tensor exists.
    """
    if isinstance(x, torch.Tensor):
        return x.device

    if isinstance(x, Mapping):
        for v in x.values():
            d = first_tensor_device(v)
            if d is not None:
                return d

    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        for v in x:
            d = first_tensor_device(v)
            if d is not None:
                return d

    return None


def get_module_object(path):
    module_path, attribute = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attribute)
