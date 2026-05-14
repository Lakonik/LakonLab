# Copyright (c) 2026 Hansheng Chen

from collections import abc
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.distributed.utils import _p_assert
from torch.distributed.fsdp._runtime_utils import (
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
    _post_forward,
    _post_forward_reshard
)
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.parallel import MODULE_WRAPPERS

from lakonlab.utils import get_module_object


MODULE_WRAPPERS.register_module(
    name='FSDP', module=FullyShardedDataParallel)


def _clone_if_reqgrad(x, memo):
    # Avoid cloning the same tensor multiple times across nested structures
    if isinstance(x, torch.Tensor) and x.requires_grad:
        xid = id(x)
        y = memo.get(xid)
        if y is None:
            # clone keeps graph connectivity but breaks storage identity
            # (good for breaking shared-input identity across blocks)
            y = x.clone()
            memo[xid] = y
        return y
    return x


def _map_structure(obj, memo):
    # Fast path for tensors/leaf types
    y = _clone_if_reqgrad(obj, memo)
    if y is not obj:
        return y

    # Containers
    if isinstance(obj, (list, tuple)):
        seq = [_map_structure(v, memo) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    if isinstance(obj, dict):
        return {k: _map_structure(v, memo) for k, v in obj.items()}
    if isinstance(obj, abc.Mapping):  # other mappings
        return type(obj)((k, _map_structure(v, memo)) for k, v in obj.items())
    if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(_map_structure(v, memo) for v in obj)

    # Namedtuple support
    if hasattr(obj, "_fields") and hasattr(obj, "_asdict"):
        return type(obj)(**{k: _map_structure(v, memo) for k, v in obj._asdict().items()})

    # Everything else untouched
    return obj


def clone_grad_inputs(*args, **kwargs):
    """
    Args: *args, **kwargs (any nested structure)
    Returns:
        new_args, new_kwargs with every tensor that has requires_grad=True cloned.
        Cloning preserves dtype/device/grad requirement and graph connectivity,
        but breaks storage identity so FSDP's multi-grad hook won't wait on
        a shared input used by other blocks.
    """
    memo = {}
    new_args = tuple(_map_structure(a, memo) for a in args)
    new_kwargs = {k: _map_structure(v, memo) for k, v in kwargs.items()}
    return new_args, new_kwargs


class FullyShardedDataParallelFix(FullyShardedDataParallel):

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        handle = self._handle
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            # =================================================
            # clone the input tensors that require grad if the module is frozen
            flat_param = handle.flat_param
            already_registered = hasattr(flat_param, "_post_backward_hook_handle")
            if not already_registered and not flat_param.requires_grad:
                args, kwargs = clone_grad_inputs(*args, **kwargs)
            # =================================================
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_unshard,
                self._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )


def tie_fsdp_modules(tgt_module, src_module, recursive=True):
    if isinstance(src_module, FullyShardedDataParallel) and isinstance(tgt_module, FullyShardedDataParallel):

        old_forward = tgt_module.forward

        def new_forward(*args, **kwargs):
            handle = src_module._handle
            args, kwargs = _root_pre_forward(src_module, src_module, args, kwargs)
            unused = None
            # =================================================
            # clone the input tensors that require grad if the module is frozen
            flat_param = handle.flat_param
            already_registered = hasattr(flat_param, "_post_backward_hook_handle")
            if not already_registered and not flat_param.requires_grad:
                args, kwargs = clone_grad_inputs(*args, **kwargs)
            # =================================================
            args, kwargs = _pre_forward(
                src_module,
                handle,
                _pre_forward_unshard,
                src_module._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == src_module.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{src_module.compute_device} but got {handle.flat_param.device}",
                )
            output = old_forward(*args, **kwargs)
            return _post_forward(
                src_module, handle, _post_forward_reshard, src_module, unused, output
            )

        tgt_module.forward = new_forward

    if recursive:
        for key, val in src_module._modules.items():
            if key in tgt_module._modules:
                tie_fsdp_modules(tgt_module._modules[key], val, recursive)


@MODULE_WRAPPERS.register_module()
class FSDPWrapper(nn.Module):
    
    def __init__(
            self,
            module,
            device_id,
            wrap_frozen_modules=False,
            ignore_frozen_parameters=False,
            param_dtype='bfloat16',
            reduce_dtype='float32',
            buffer_dtype='bfloat16',
            fsdp_modules=None,
            exclude_keys=(),
            tie_key_mappings=None,
            use_orig_params=True,
            hybrid_sharding=True,
            **kwargs):
        super().__init__()
        self.module = module
        if fsdp_modules is not None:
            assert isinstance(fsdp_modules, (list, tuple))
            fsdp_modules = tuple([get_module_object(m) for m in fsdp_modules])
        else:
            fsdp_modules = ()
        fsdp_kwargs = kwargs
        if hybrid_sharding:
            global_world_size = dist.get_world_size()
            num_devices_per_node = torch.cuda.device_count()
            mesh = dist.init_device_mesh(
                'cuda',
                (global_world_size // num_devices_per_node, num_devices_per_node),
                mesh_dim_names=('replicate', 'shard'))
            fsdp_kwargs.update(
                device_mesh=mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD)
        else:
            fsdp_kwargs.update(
                sharding_strategy=ShardingStrategy.FULL_SHARD)
        fsdp_kwargs.update(
            use_orig_params=use_orig_params,
            mixed_precision=MixedPrecision(
                param_dtype=getattr(torch, param_dtype),
                reduce_dtype=getattr(torch, reduce_dtype),
                buffer_dtype=getattr(torch, buffer_dtype),
                cast_root_forward_inputs=False))
        self.to_fsdp(
            device_id,
            wrap_frozen_modules,
            ignore_frozen_parameters,
            exclude_keys,
            tie_key_mappings,
            fsdp_modules=fsdp_modules,
            **fsdp_kwargs)

    def to_fsdp(
            self, device_id, wrap_frozen_modules=False, ignore_frozen_parameters=False,
            exclude_keys=(), tie_key_mappings=None, fsdp_modules=(), **kwargs):
        for name, module in self.module._modules.items():
            if name in exclude_keys or next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                if wrap_frozen_modules:
                    fsdp_kwargs = kwargs.copy()
                    ignored_states = []
                    for p in module.parameters():
                        if getattr(p, 'lakonlab_no_shard', False):
                            p.data = p.data.cuda()
                            ignored_states.append(p)
                    fsdp_kwargs.update(
                        use_orig_params=False,
                        ignored_states=ignored_states,
                        auto_wrap_policy=CustomPolicy(
                            lambda m: isinstance(m, fsdp_modules) and next(m.parameters(), None) is not None))
                    module = FullyShardedDataParallelFix(
                        module,
                        device_id=device_id,
                        **fsdp_kwargs)
                else:
                    module = module.cuda()
            else:
                fsdp_kwargs = kwargs.copy()
                ignored_states = []
                for p in module.parameters():
                    if getattr(p, 'lakonlab_no_shard', False) or (ignore_frozen_parameters and not p.requires_grad):
                        p.data = p.data.cuda()
                        ignored_states.append(p)
                fsdp_kwargs.update(
                    ignored_states=ignored_states,
                    auto_wrap_policy=CustomPolicy(
                        lambda m: isinstance(m, fsdp_modules) and next(m.parameters(), None) is not None))
                module = FullyShardedDataParallelFix(
                    module,
                    device_id=device_id,
                    **fsdp_kwargs)
            self.module._modules[name] = module

        if tie_key_mappings is not None:
            # parse tie_key_mappings in the format ('teacher->diffusion', 'teacher->diffusion_ema')
            src_keys = []
            tgt_keys = []
            for mapping in tie_key_mappings:
                src_key, tgt_key = mapping.split('->')
                src_keys.append(src_key)
                tgt_keys.append(tgt_key)
            # detect chained tie_key_mappings
            chained_keys = sorted(set(src_keys) & set(tgt_keys))
            if len(chained_keys) > 0:
                raise ValueError(
                    'Chained `tie_key_mappings` are not supported in FSDPWrapper. '
                    f'Found keys used as both source and target: {chained_keys}. '
                    'Please tie all targets directly from a canonical source, e.g. '
                    '`teacher->diffusion`, `teacher->diffusion_ema`.')
            # tie the pre-forward and post-forward calls
            for src_key, tgt_key in zip(src_keys, tgt_keys):
                tie_fsdp_modules(
                    self.module._modules[tgt_key], self.module._modules[src_key])

    def scatter(self, inputs, kwargs, device_ids):
        """Scatter function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
            device_ids (int): Device id.
        """
        return scatter_kwargs(inputs, kwargs, device_ids)

    def forward(self, *inputs, **kwargs):
        """Forward function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output
