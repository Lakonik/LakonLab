import inspect
import bitsandbytes

from typing import List
from mmcv.runner import build_optimizer
from lakonlab.utils import rgetattr

from mmcv.runner.optimizer.builder import OPTIMIZERS

from .default_constructor import DefaultOptimizerConstructor  # noqa: F401


def register_bitsandbytes_optimizers() -> List:
    bitsandbytes_optimizers = []
    for module_name in dir(bitsandbytes.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(bitsandbytes.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, bitsandbytes.optim.optimizer.Optimizer2State) \
                and module_name not in OPTIMIZERS.module_dict:
            OPTIMIZERS.register_module(module=_optim)
            bitsandbytes_optimizers.append(module_name)
    return bitsandbytes_optimizers


BNB_OPTIMIZERS = register_bitsandbytes_optimizers()


def build_optimizers(model, cfgs):
    """Modified from MMGeneration
    """
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizer
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            is_dict_of_dict = False
    if is_dict_of_dict:
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = rgetattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    return build_optimizer(model, cfgs)
