import torch
import mmcv
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model
from lakonlab.runner.hooks.ema_hook import get_ori_key


def init_model(
        config, checkpoint=None, device='cuda:0', cfg_options=None,
        ema_only=True, torch_dtype=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = build_model(
        config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)

    if ema_only:
        module_keys = []
        for hook in config.get('custom_hooks', []):
            if hook['type'] == 'ExponentialMovingAverageHook':
                if isinstance(hook['module_keys'], str):
                    module_keys.append(hook['module_keys'])
                else:
                    module_keys.extend(hook['module_keys'])
        for key in module_keys:
            ori_key = get_ori_key(key)
            del model._modules[ori_key]

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    model._cfg = config  # save the config in the model for convenience

    if torch_dtype is not None:
        for m in model.modules():
            if hasattr(m, 'autocast_dtype'):
                setattr(m, 'autocast_dtype', None)
        model.to(dtype=getattr(torch, torch_dtype))

    model.to(device)
    model.eval()

    return model
