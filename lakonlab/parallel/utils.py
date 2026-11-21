import contextlib
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel
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
    ddp_kwargs = cfg.get('ddp_kwargs', {}).copy()
    ddp_kwargs.setdefault(
        'find_unused_parameters',
        cfg.get('find_unused_parameters', False))
    fsdp_kwargs = cfg.get('fsdp_kwargs', {})
    if module_wrapper is None:
        model = MMDistributedDataParallelFix(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=True,
            **ddp_kwargs)
        model.broadcast_buffers = False  # https://github.com/pytorch/pytorch/issues/177514
    elif module_wrapper.lower() == 'ddp':
        mmcv.print_log('Use DDP Wrapper.', 'lakonlab')
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            **ddp_kwargs)
    elif module_wrapper.lower() == 'fsdp':
        mmcv.print_log('Use FSDP Wrapper.', 'lakonlab')
        model = FSDPWrapper(
            model,
            device_id=torch.cuda.current_device(),
            **fsdp_kwargs)
    elif module_wrapper.lower() == 'fsdp2':
        mmcv.print_log('Use FSDP2 Wrapper.', 'lakonlab')
        model = FSDP2Wrapper(
            model,
            **fsdp_kwargs)
    else:
        raise ValueError(f'Unsupported module wrapper: {module_wrapper}.')
    return model


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
                if module.static_graph:
                    # entering no_sync on a static-graph DDP module corrupts its
                    # one-shot graph recording (crashes with an internal reducer
                    # assert), so fall back to syncing gradients every step
                    warnings.warn(
                        'maybe_ddp_no_sync: skipping no_sync for a DDP module with '
                        'static_graph=True (incompatible with no_sync). Gradients '
                        'will be all-reduced every step; disable static_graph in '
                        'ddp_kwargs to enable gradient-accumulation no_sync.',
                        RuntimeWarning)
                    continue
                stack.enter_context(module.no_sync())
        yield
