# Copyright (c) 2026 Hansheng Chen

import itertools

import torch
from torch.func import functional_call
from lakonlab.utils import rgetattr, get_root_logger


def cast_dtype(module, dtype, keep_fp32_keys=None):
    if keep_fp32_keys is None:
        keep_fp32_keys = []

    for name, t in itertools.chain(module.named_parameters(), module.named_buffers()):
        if not t.is_floating_point():
            continue
        keep_fp32 = False
        for keep_key in keep_fp32_keys:
            if keep_key.startswith('self.'):  # use full name matching (a tensor or a parent module)
                if keep_key[5:] == name or name.startswith(keep_key[5:] + '.'):
                    keep_fp32 = True
                    break
            elif keep_key in name:  # use partial name matching
                keep_fp32 = True
                break
        t.data = t.data.to(dtype=torch.float32 if keep_fp32 else dtype)


def autocast_patch(module, dtype=None, enabled=True):

    def make_new_forward(old_forward, dtype, enabled):
        def new_forward(*args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled):
                result = old_forward(*args, **kwargs)
            return result

        return new_forward

    module.forward = make_new_forward(module.forward, dtype, enabled)


def _to_fp32(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.float()
    if isinstance(x, tuple):
        return tuple(_to_fp32(v) for v in x)
    if isinstance(x, list):
        return [_to_fp32(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_fp32(v) for k, v in x.items()}
    return x


def fp32_module_forward(module, *args, enabled=True, **kwargs):
    """Run a module in fp32 without changing its stored parameter dtype."""
    if not enabled:
        return module(*args, **kwargs)

    params_and_buffers = {
        name: _to_fp32(tensor)
        for name, tensor in tuple(module.named_parameters()) + tuple(module.named_buffers())
    }
    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
        return functional_call(module, params_and_buffers, _to_fp32(args), _to_fp32(kwargs))


def flex_freeze(module, exclude_keys=None, exclude_fp32=True, exclude_autocast_dtype=None):
    module.requires_grad_(False)

    if exclude_keys is not None and len(exclude_keys) > 0:

        logger = get_root_logger()

        # find modules
        excluded_module_keys = set()
        exclude_modules_names = []
        for name, _ in module.named_modules():
            for exclude_key in exclude_keys:
                if exclude_key.startswith('self.'):  # use full name matching
                    if exclude_key[5:] == name:
                        exclude_modules_names.append(name)
                        excluded_module_keys.add(exclude_key)
                        break
                elif exclude_key in name:  # use partial name matching
                    exclude_modules_names.append(name)
                    excluded_module_keys.add(exclude_key)
                    break

        for name in exclude_modules_names:
            m = rgetattr(module, name)
            if exclude_fp32:
                m.to(torch.float32)
                if exclude_autocast_dtype is not None:
                    autocast_patch(m, dtype=getattr(torch, exclude_autocast_dtype))
            m.requires_grad_(True)

        if exclude_fp32 and len(exclude_modules_names) > 0 and hasattr(module, '_keep_in_fp32_modules'):
            # register the fp32 modules for diffusers dtype detection (get_parameter_dtype)
            module._keep_in_fp32_modules = (
                list(module._keep_in_fp32_modules or []) + exclude_modules_names)

        exclude_keys = set(exclude_keys) - excluded_module_keys

        if len(exclude_keys) > 0:
            # find parameters
            excluded_parameter_keys = set()
            exclude_parameters_names = []
            for name, _ in module.named_parameters():
                for exclude_key in exclude_keys:
                    if exclude_key.startswith('self.'):  # use full name matching
                        if exclude_key[5:] == name:
                            exclude_parameters_names.append(name)
                            excluded_parameter_keys.add(exclude_key)
                            break
                    elif exclude_key in name:  # use partial name matching
                        exclude_parameters_names.append(name)
                        excluded_parameter_keys.add(exclude_key)
                        break

            for name in exclude_parameters_names:
                p = rgetattr(module, name)
                if exclude_fp32:
                    logger.warning(
                        f'Parameter autocast patching is not supported yet. '
                        f'Please ensure that parameter {name} is used in fp32 context.')
                    p.data = p.data.to(torch.float32)
                p.requires_grad_(True)

            if exclude_fp32 and len(exclude_parameters_names) > 0 and hasattr(module, '_keep_in_fp32_modules'):
                # register the fp32 parameters for diffusers dtype detection (get_parameter_dtype)
                module._keep_in_fp32_modules = (
                    list(module._keep_in_fp32_modules or []) + exclude_parameters_names)

            exclude_keys = exclude_keys - excluded_parameter_keys

            if len(exclude_keys) > 0:
                logger.warning(f'Exclusion keys not found: {exclude_keys}')


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError('The input module should contain parameters.')

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()

    return torch.device('cpu')
