# Copyright (c) 2026 Hansheng Chen

import torch
from lakonlab.utils import rgetattr, get_root_logger


def autocast_patch(module, dtype=None, enabled=True):

    def make_new_forward(old_forward, dtype, enabled):
        def new_forward(*args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled):
                result = old_forward(*args, **kwargs)
            return result

        return new_forward

    module.forward = make_new_forward(module.forward, dtype, enabled)


def flex_freeze(module, exclude_keys=None, exclude_fp32=True, exclude_autocast_dtype='float32'):
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
                autocast_patch(m, dtype=getattr(torch, exclude_autocast_dtype))
            m.requires_grad_(True)

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
