# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
except:
    pass
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.parallel import MODULE_WRAPPERS

from lakonlab.utils import get_module_object


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
            fsdp_modules=fsdp_modules,
            **fsdp_kwargs)

    def to_fsdp(self,
                wrap_frozen_modules=False,
                ignore_frozen_parameters=False,
                exclude_keys=(),
                fsdp_modules=(),
                **kwargs):
        for name, module in self.module._modules.items():
            if name in exclude_keys or next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                if wrap_frozen_modules:
                    ignored_params = []
                    for p in module.parameters():
                        if getattr(p, 'lakonlab_no_shard', False):
                            p.data = p.data.cuda()
                            ignored_params.append(p)
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
                ignored_params = []
                for p in module.parameters():
                    if getattr(p, 'lakonlab_no_shard', False) or (ignore_frozen_parameters and not p.requires_grad):
                        p.data = p.data.cuda()
                        ignored_params.append(p)
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
