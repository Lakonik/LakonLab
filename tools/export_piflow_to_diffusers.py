import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

import os
import argparse
import json
from collections import OrderedDict

from safetensors.torch import save_file
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from mmcv import Config
from mmcv.runner import _load_checkpoint

from lakonlab.models import build_module
from lakonlab.runner.checkpoint import exists_ckpt
from lakonlab.utils import rgetattr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a LakonLab pi-Flow checkpoint to diffusers safetensors format.')
    parser.add_argument(
        'config', help='Config file path')
    parser.add_argument(
        '--ckpt',
        help='Checkpoint file. If not specified, the latest checkpoint will be used.')
    parser.add_argument(
        '--out-dir',
        help='Output directory where the configuration json and model safetensors will be saved. If not specified, '
        '`checkpoints/<MODEL_NAME>` will be used with `<MODEL_NAME>` inferred from the config file.')
    parser.add_argument(
        '--non-ema',
        action='store_true',
        help='If specified, the non-EMA weights will be exported instead of the EMA weights.')
    return parser.parse_args()


def save_config(model, save_directory, class_name_override=None):
    if os.path.isfile(save_directory):
        raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)

    json_string = model.to_json_string()
    config_dict = json.loads(json_string)
    if '_class_name' in config_dict and class_name_override is not None:
        config_dict['_class_name'] = class_name_override
        json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    output_config_file = os.path.join(save_directory, model.config_name)
    with open(output_config_file, "w", encoding="utf-8") as writer:
        writer.write(json_string)


def main():
    args = parse_args()
    config_path = args.config
    ckpt = args.ckpt
    out_dir = args.out_dir
    non_ema = args.non_ema

    cfg = Config.fromfile(config_path)
    policy_type = rgetattr(cfg, 'model.diffusion.policy_type', None)
    if policy_type is None:
        raise ValueError('Unsupported configuration: policy_type not found in model.diffusion')

    if out_dir is None:
        if hasattr(cfg, 'name'):
            model_name = cfg.name
        else:
            model_name = os.path.splitext(os.path.basename(config_path))[0]
        out_dir = os.path.join('checkpoints', model_name)
        print(f'Output directory not specified. Using {out_dir}')

    # disable unused modules for export
    if hasattr(cfg.model.diffusion.denoising, 'use_lora'):
        cfg.model.diffusion.denoising.use_lora = False
    if hasattr(cfg.model.diffusion.denoising, 'pretrained'):
        cfg.model.diffusion.denoising.pretrained = None
    if hasattr(cfg.model.diffusion.denoising, 'freeze_exclude'):
        # suppress warnings
        cfg.model.diffusion.denoising.freeze_exclude = None

    model = build_module(cfg.model.diffusion.denoising)
    save_config(model, out_dir, class_name_override=cfg.model.diffusion.denoising.type)
    print(f'Saved model config to {out_dir}')

    if ckpt is None:
        if exists_ckpt(cfg.resume_from):
            ckpt = cfg.resume_from
        elif exists_ckpt(cfg.load_from):
            ckpt = cfg.load_from
        else:
            raise ValueError('No checkpoint specified and no valid checkpoint found in config.')

    print(f'Loading checkpoint from {ckpt}')
    checkpoint = _load_checkpoint(ckpt, map_location='cpu')
    state_dict = checkpoint['state_dict']

    if non_ema:
        prefix = 'diffusion.denoising.'
    else:
        prefix = 'diffusion_ema.denoising.'

    out_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            new_k = new_k.replace(
                'lora_A.default.weight', 'lora_A.weight'
            ).replace(
                'lora_B.default.weight', 'lora_B.weight')
            out_dict[new_k] = v

    policy_config = cfg.model.diffusion.get('policy_kwargs', dict())
    policy_config.update(type=policy_type)
    policy_config_json = json.dumps(policy_config)
    save_file(
        out_dict,
        os.path.join(out_dir, SAFETENSORS_WEIGHTS_NAME),
        metadata=dict(
            policy_config=policy_config_json
        ))
    print(f'Saved model safetensors to {out_dir} with policy config: {policy_config_json}')


if __name__ == '__main__':
    main()
