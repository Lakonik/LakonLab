import torch
import mmcv.parallel._functions
from torch.nn.parallel._functions import _get_stream


def _get_stream_wrapper(dev):
    if isinstance(dev, int):
        if dev == -1:
            return None
        dev = torch.device("cuda", dev)
    return _get_stream(dev)


mmcv.parallel._functions._get_stream = _get_stream_wrapper


from .ddp_wrapper import DistributedDataParallelWrapper, MMDistributedDataParallelFix
from .fsdp_wrapper import FSDPWrapper
from .fsdp2_wrapper import FSDP2Wrapper
from .utils import apply_module_wrapper

__all__ = [
    'MMDistributedDataParallelFix', 'DistributedDataParallelWrapper', 'FSDPWrapper', 'FSDP2Wrapper',
    'apply_module_wrapper']
