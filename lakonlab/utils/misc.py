# Copyright (c) 2026 Hansheng Chen

import gc
import contextlib
import functools
import importlib
from collections.abc import Mapping, Sequence
from itertools import repeat
from typing import Any, Optional
from functools import partial

import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor

from mmcv.parallel import is_module_wrapper


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
    """Set ``requires_grad`` on a module's parameters, restoring it on exit.

    ``requires_grad`` is either a single value for all parameters or a
    per-parameter sequence in ``module.parameters()`` order, e.g. a state
    captured from an earlier forward pass.

    Restores both the original and currently exposed parameter objects when
    a wrapper such as FSDP2 swaps them during the context.
    """

    def __init__(self, module, requires_grad=True):
        self.module = module
        self.requires_grad = requires_grad
        self.prev = []

    def __enter__(self):
        self.prev = []
        if isinstance(self.requires_grad, bool):
            requires_grad = repeat(self.requires_grad)
        else:
            requires_grad = self.requires_grad
        for p, r in zip(self.module.parameters(), requires_grad):
            self.prev.append((p, p.requires_grad))
            p.requires_grad = r
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for p, (prev_p, r) in zip(self.module.parameters(), self.prev):
            prev_p.requires_grad = r
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


def all_shapes_equal(tgt_module, src_module):
    tgt_tensors = dict(tgt_module.named_parameters())
    tgt_tensors.update(tgt_module.named_buffers())
    src_tensors = dict(src_module.named_parameters())
    src_tensors.update(src_module.named_buffers())
    if tgt_tensors.keys() != src_tensors.keys():
        return False
    return all(tgt_tensors[k].shape == src_tensors[k].shape for k in tgt_tensors)


def tie_untrained_root_params(tgt_module, src_module):
    for key, val in src_module._parameters.items():
        tgt_param = tgt_module._parameters.get(key, None)
        if (val is not None) \
                and (tgt_param is not None) \
                and (not val.requires_grad) \
                and (not tgt_param.requires_grad) \
                and tgt_param.shape == val.shape:
            val.lakonlab_no_shard = True  # hint for FSDP to not shard this parameter
            tgt_module._parameters[key] = val


def tie_untrained_submodules(tgt_module, src_module, tie_tgt_lora_base_layer=False):
    from peft.tuners.lora import LoraLayer  # peft imports bitsandbytes, which initializes CUDA
    # A protected descendant also prevents sharing its entire parent container.
    for key, src_submodule in src_module._modules.items():
        if key in tgt_module._modules:
            if (tie_tgt_lora_base_layer
                    and isinstance(tgt_module._modules[key], LoraLayer)
                    and not isinstance(src_submodule, LoraLayer)):
                if all_frozen((tgt_module._modules[key]._modules['base_layer'], src_submodule)) and not any(
                        getattr(m, 'lakonlab_no_tie', False)
                        for module in (tgt_module._modules[key]._modules['base_layer'], src_submodule)
                        for m in module.modules()) \
                        and all_shapes_equal(tgt_module._modules[key]._modules['base_layer'], src_submodule):
                    tgt_module._modules[key]._modules['base_layer'] = src_submodule
                else:
                    tie_untrained_submodules(
                        tgt_module._modules[key]._modules['base_layer'], src_submodule, tie_tgt_lora_base_layer)
                    tie_untrained_root_params(
                        tgt_module._modules[key]._modules['base_layer'], src_submodule)
            else:
                if all_frozen((tgt_module._modules[key], src_submodule)) and not any(
                        getattr(m, 'lakonlab_no_tie', False)
                        for module in (tgt_module._modules[key], src_submodule)
                        for m in module.modules()) \
                        and all_shapes_equal(tgt_module._modules[key], src_submodule):
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


def untie_all_parameters(
        tgt_module, src_module, recursive=True, untie_tgt_lora_base_layer=False,
        preserve_tied=False):
    """Separate aliased storage, optionally preserving shared modules and parameter objects.

    Buffers in independent modules are untied even when they are the same tensor object.
    """
    from peft.tuners.lora import LoraLayer  # peft imports bitsandbytes, which initializes CUDA
    if preserve_tied and tgt_module is src_module:
        return
    for key, val in src_module._parameters.items():
        if val is not None \
                and tgt_module._parameters.get(key, None) is not None \
                and not (preserve_tied and tgt_module._parameters[key] is val) \
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
                        recursive=True, untie_tgt_lora_base_layer=untie_tgt_lora_base_layer,
                        preserve_tied=preserve_tied)
                else:
                    untie_all_parameters(
                        tgt_module._modules[key], val,
                        recursive=True, untie_tgt_lora_base_layer=untie_tgt_lora_base_layer,
                        preserve_tied=preserve_tied)


def clone_params(tgt_module, src_module, recursive=True):
    """Clone parameters and buffers from src_module to tgt_module (sharing the same structure).
    Tied parameters/buffers are not cloned. Used for EMA model initialization.
    """
    for key, val in src_module._parameters.items():
        tgt_param = tgt_module._parameters[key]
        if (val is not None) and (val is not tgt_param):
            tgt_module._parameters[key] = torch.nn.Parameter(
                val.detach().to(dtype=tgt_param.dtype, copy=True), requires_grad=tgt_param.requires_grad)
    for key, val in src_module._buffers.items():
        tgt_buffer = tgt_module._buffers[key]
        if val is not tgt_buffer:
            tgt_module._buffers[key] = val.to(dtype=tgt_buffer.dtype, copy=True)
    if recursive:
        for key, val in src_module._modules.items():
            clone_params(
                tgt_module._modules[key], val, recursive)


@torch.no_grad()
def fsdp2_clip_grad(model, max_norm: float) -> torch.Tensor:
    """Clip the global L2 norm of FSDP2 DTensor gradients.

    All trainable parameters must be DTensors whose meshes span the default
    process group. Gradient placements may be replicated or sharded, but not
    partial. All ranks must call this function collectively with the same
    parameter set.

    Returns a plain, fully reduced FP32 tensor that is identical on every rank.
    Performs no host-device synchronization; clipping uses an on-device clamped
    coefficient like ``torch.nn.utils.clip_grad_norm_``.
    """
    world_size = dist.get_world_size()
    local_param_grads = []
    replica_counts = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if not isinstance(p, DTensor):
            raise RuntimeError(
                'fsdp2_clip_grad found a trainable non-DTensor parameter '
                'inside an FSDP2 module; FSDP2 does not synchronize gradients '
                'for ignored parameters')
        if p.grad is None:
            continue
        if not isinstance(p.grad, DTensor):
            raise RuntimeError(
                'fsdp2_clip_grad requires every trainable DTensor '
                'parameter to have a DTensor gradient')

        grad = p.grad
        mesh_size = grad.device_mesh.size()
        if mesh_size != world_size:
            raise RuntimeError(
                'fsdp2_clip_grad requires every DTensor mesh to span the '
                f'default process group, but got mesh size {mesh_size} and '
                f'world size {world_size}')

        replica_count = mesh_size
        for mesh_dim_size, placement in zip(grad.device_mesh.shape, grad.placements):
            if placement.is_partial():
                raise RuntimeError(
                    'fsdp2_clip_grad does not support Partial gradient placements')
            if placement.is_shard():
                replica_count //= mesh_dim_size
            elif not placement.is_replicate():
                # catches e.g. _StridedShard (FSDP2 + TP), whose is_shard() is
                # False on torch 2.10 and which would otherwise be silently
                # miscounted as replication
                raise RuntimeError(
                    'fsdp2_clip_grad only supports replicated or sharded '
                    f'gradient placements, but got {placement}')

        local_param_grads.append(grad.to_local())
        replica_counts.append(replica_count)

    if not local_param_grads:
        # grad presence is rank-uniform under FSDP2 (a rank-divergent backward
        # would have hung its own reduce-scatter before reaching this point),
        # so returning early without the all-reduce is collective-safe
        return torch.zeros((), dtype=torch.float32)

    local_norms = torch.stack(torch._foreach_norm(local_param_grads, 2.0)).float()
    replica_counts_tensor = torch.tensor(
        replica_counts, dtype=torch.float32, device=local_param_grads[0].device)
    local_sq_sum = (local_norms.square() / replica_counts_tensor).sum()

    dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
    total_norm = local_sq_sum.sqrt()

    # compute the clamped coefficient on-device and multiply unconditionally,
    # matching torch.nn.utils.clip_grad_norm_, so that clipping itself never
    # forces a host-device sync (callers may disable grad_clip_check_finite
    # precisely to avoid per-step syncs)
    clip_factor = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    torch._foreach_mul_(local_param_grads, clip_factor)

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


_warmup = False


def is_warmup():
    """True while the runner is compiling graph variants rather than really training.

    A module whose input shapes are not fully determined by the data it is given -- a text encoder
    padding to the longest prompt in the batch, say -- reads this in its own `forward` and runs the
    shapes it would otherwise compile lazily, stalling one rank at a time mid-run. Reading it in
    `forward` is what keeps it correct: the module is entered through whatever wrapper holds it,
    with its parameters gathered, at the batch size the caller really passed.
    """
    return _warmup


@contextlib.contextmanager
def warmup_mode(enable=True):
    global _warmup
    prev_enabled = _warmup
    _warmup = enable
    try:
        yield
    finally:
        _warmup = prev_enabled


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


def onload(method):
    """Decorate a module method (e.g. a VAE's encode/decode) to run a CPU-offloaded module (FSDP ``offload_keys``)
    on the device of its first tensor argument, moving it back to CPU afterwards.

    A no-op for modules that are not on CPU.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if next(self.parameters()).device.type != 'cpu':
            return method(self, *args, **kwargs)
        self.to(first_tensor_device((args, kwargs)))
        try:
            return method(self, *args, **kwargs)
        finally:
            self.to('cpu')
    return wrapper


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
