import torch
import mmcv

from . import MMDistributedDataParallelFix, DistributedDataParallelWrapper, FSDPWrapper, FSDP2Wrapper
from lakonlab.utils import get_module_object


def prepare_module_wrapper(cfg):
    module_wrapper = cfg.get('module_wrapper', None)
    if module_wrapper is not None and module_wrapper.lower() in ['fsdp', 'fsdp2']:
        fsdp_kwargs = cfg.get('fsdp_kwargs', {})
        fsdp_modules = fsdp_kwargs.get('fsdp_modules', None)
        if fsdp_modules is not None:
            for module_name in fsdp_modules:
                module_class = get_module_object(module_name)
                module_class.lakonlab_no_tie = True
    return None


def apply_module_wrapper(model, module_wrapper, cfg):
    if module_wrapper is None:
        model = MMDistributedDataParallelFix(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=True,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
        model.broadcast_buffers = False  # https://github.com/pytorch/pytorch/issues/177514
    elif module_wrapper.lower() == 'ddp':
        mmcv.print_log('Use DDP Wrapper.', 'lakonlab')
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    elif module_wrapper.lower() == 'fsdp':
        mmcv.print_log('Use FSDP Wrapper.', 'lakonlab')
        fsdp_kwargs = cfg.get('fsdp_kwargs', {})
        model = FSDPWrapper(
            model,
            device_id=torch.cuda.current_device(),
            **fsdp_kwargs)
    elif module_wrapper.lower() == 'fsdp2':
        mmcv.print_log('Use FSDP2 Wrapper.', 'lakonlab')
        fsdp_kwargs = cfg.get('fsdp_kwargs', {})
        model = FSDP2Wrapper(
            model,
            **fsdp_kwargs)
    else:
        raise ValueError(f'Unsupported module wrapper: {module_wrapper}.')
    return model
