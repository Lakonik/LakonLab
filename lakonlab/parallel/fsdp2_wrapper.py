# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.parallel import MODULE_WRAPPERS
from mmcv.utils import digit_version

from lakonlab.utils import get_module_object
from .ddp_wrapper import MMDistributedDataParallelFix


def _patch_fsdp2_mp_dtypes():
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    # Backport PyTorch 2.14's fix for pytorch/pytorch#186998:
    # https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py
    def _init_mp_dtypes(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[FSDPParam] = [
            p for p in self.fsdp_params if p.sharded_param.requires_grad
        ]
        if trainable_params:
            params_for_dtype = trainable_params
        else:
            params_for_dtype = [
                p for p in self.fsdp_params if p.orig_dtype.is_floating_point
            ]
        orig_dtypes = {p.orig_dtype for p in params_for_dtype}
        reduce_dtypes = {p.reduce_dtype for p in params_for_dtype}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            # Models may have no grad params
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            # This can be relaxed if we issue one reduce-scatter per reduce
            # dtype (but we would need a way for users to specify multiple
            # reduce dtypes)
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        dtype_sets_are_uniform = len(orig_dtypes) == 1 and len(reduce_dtypes) == 1
        self._orig_dtype = next(iter(orig_dtypes)) if dtype_sets_are_uniform else None
        self._reduce_dtype = (
            next(iter(reduce_dtypes)) if dtype_sets_are_uniform else None
        )

    FSDPParamGroup._init_mp_dtypes = _init_mp_dtypes


if digit_version(torch.__version__)[:2] < (2, 14):
    _patch_fsdp2_mp_dtypes()


def _reduce_param_grad(param):
    # DDP-style replication for trainable no-shard params: FSDP skips gradient reduction
    # for ignored states, so average the gradient over all ranks manually (fires after
    # each grad accumulation; already-averaged partial sums are fixed points, so this is
    # correct under gradient accumulation)
    if param.grad is not None:
        dist.all_reduce(param.grad.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)


@MODULE_WRAPPERS.register_module()
class FSDP2Wrapper(nn.Module):

    def __init__(
            self,
            module,
            wrap_frozen_modules=False,
            ignore_frozen_parameters=False,
            param_dtype='bfloat16',
            reduce_dtype='float32',
            fsdp_modules=None,
            exclude_keys=(),
            offload_keys=(),
            ddp_keys=(),
            ddp_kwargs=dict(),
            hybrid_sharding=True,
            **kwargs):
        super().__init__()
        self.module = module
        fsdp_kwargs = kwargs
        self.param_dtype = getattr(torch, param_dtype)
        self.reduce_dtype = getattr(torch, reduce_dtype)
        if hybrid_sharding:
            global_world_size = dist.get_world_size()
            num_devices_per_node = torch.cuda.device_count()
            mesh = dist.init_device_mesh(
                'cuda',
                (global_world_size // num_devices_per_node, num_devices_per_node),
                mesh_dim_names=('replicate', 'shard'))
            fsdp_kwargs.update(mesh=mesh)
        if fsdp_modules is not None:
            assert isinstance(fsdp_modules, (list, tuple))
            fsdp_modules = tuple([get_module_object(m) for m in fsdp_modules])
        else:
            fsdp_modules = ()
        self.to_fsdp(
            wrap_frozen_modules,
            ignore_frozen_parameters,
            exclude_keys,
            offload_keys=offload_keys,
            ddp_keys=ddp_keys,
            ddp_kwargs=ddp_kwargs,
            fsdp_modules=fsdp_modules,
            **fsdp_kwargs)

    def to_fsdp(self,
                wrap_frozen_modules=False,
                ignore_frozen_parameters=False,
                exclude_keys=(),
                offload_keys=(),
                ddp_keys=(),
                ddp_kwargs=dict(),
                fsdp_modules=(),
                **kwargs):
        for name, module in self.module._modules.items():
            if name in ddp_keys:
                module = MMDistributedDataParallelFix(
                    module.cuda(), device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=True, **ddp_kwargs)
                module.broadcast_buffers = False  # https://github.com/pytorch/pytorch/issues/177514
            elif name in exclude_keys or next(module.parameters(), None) is None:
                # excluded modules in offload_keys stay on CPU; their @onload methods move them in
                module = module if name in offload_keys else module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                if wrap_frozen_modules:
                    ignored_params = set()  # fully_shard needs a set (a list would compare tensors with ==)
                    for p in module.parameters():
                        if getattr(p, 'lakonlab_no_shard', False):
                            p.data = p.data.cuda()
                            ignored_params.add(p)
                    for submodule in module.modules():
                        if isinstance(submodule, fsdp_modules) and next(submodule.parameters(), None) is not None:
                            fsdp_kwargs = kwargs.copy()
                            fsdp_kwargs.update(
                                mp_policy=MixedPrecisionPolicy(
                                    param_dtype=self.param_dtype,
                                    reduce_dtype=self.reduce_dtype))
                            if len(ignored_params) > 0:  # requires torch >= 2.7
                                fsdp_kwargs.update(ignored_params=ignored_params)
                            fully_shard(submodule, **fsdp_kwargs)
                    fsdp_kwargs = kwargs.copy()
                    fsdp_kwargs.update(
                        mp_policy=MixedPrecisionPolicy(
                            param_dtype=self.param_dtype,
                            reduce_dtype=self.reduce_dtype,
                            cast_forward_inputs=False))
                    if len(ignored_params) > 0:  # requires torch >= 2.7
                        fsdp_kwargs.update(ignored_params=ignored_params)
                    fully_shard(module, **fsdp_kwargs)
                else:
                    module = module.cuda()
            else:
                ignored_params = set()  # fully_shard needs a set (a list would compare tensors with ==)
                for p in module.parameters():
                    if getattr(p, 'lakonlab_no_shard', False) or (ignore_frozen_parameters and not p.requires_grad):
                        p.data = p.data.cuda()
                        ignored_params.add(p)
                        if p.requires_grad:  # trainable params with lakonlab_no_shard
                            p.register_post_accumulate_grad_hook(_reduce_param_grad)
                for submodule in module.modules():
                    if isinstance(submodule, fsdp_modules) and next(submodule.parameters(), None) is not None:
                        fsdp_kwargs = kwargs.copy()
                        fsdp_kwargs.update(
                            mp_policy=MixedPrecisionPolicy(
                                param_dtype=self.param_dtype,
                                reduce_dtype=self.reduce_dtype))
                        if len(ignored_params) > 0:  # requires torch >= 2.7
                            fsdp_kwargs.update(ignored_params=ignored_params)
                        fully_shard(submodule, **fsdp_kwargs)
                fsdp_kwargs = kwargs.copy()
                fsdp_kwargs.update(
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=self.param_dtype,
                        reduce_dtype=self.reduce_dtype,
                        cast_forward_inputs=False))
                if len(ignored_params) > 0:  # requires torch >= 2.7
                    fsdp_kwargs.update(ignored_params=ignored_params)
                fully_shard(module, **fsdp_kwargs)
            self.module._modules[name] = module

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
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output
