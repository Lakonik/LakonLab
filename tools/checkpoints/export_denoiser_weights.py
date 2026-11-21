# Copyright (c) 2026 Hansheng Chen

import warnings

warnings.filterwarnings(
    'ignore',
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    category=UserWarning)

import argparse
import os
import tempfile

from accelerate import init_empty_weights
from mmcv import Config
from safetensors.torch import save_file

from lakonlab.models import build_module
from lakonlab.runner.checkpoint import _load_checkpoint, exists_ckpt
from lakonlab.utils.io_utils import FileClient


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export the EMA denoiser of a training checkpoint to a safetensors file that the denoiser '
                    'loads through `pretrained` or `pretrained_adapter`. LoRA keys use the standard adapter format. '
                    'Keys are checked against the current denoiser: its parameters '
                    'and persistent buffers are kept, so a checkpoint from an older revision loads without '
                    'unexpected keys; non-persistent buffers are excluded, as load_state_dict rejects them.')
    parser.add_argument('config', help='Training config of the checkpoint')
    parser.add_argument('--ckpt', help='Checkpoint file; defaults to the config\'s resume_from or load_from.')
    parser.add_argument('--out', required=True, help='Output .safetensors path, local or s3://')
    parser.add_argument('--non-ema', action='store_true', help='Export the raw weights instead of the EMA.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    ckpt = args.ckpt
    if ckpt is None:
        ckpt = cfg.resume_from if exists_ckpt(cfg.resume_from) else cfg.load_from
        assert exists_ckpt(ckpt), 'No checkpoint specified and none found in the config.'

    cfg.model.diffusion.denoising.pretrained = None
    with init_empty_weights():
        model = build_module(cfg.model.diffusion.denoising)
    keep = {name for name, _ in model.named_parameters()}
    for module_name, module in model.named_modules():
        keep.update(f'{module_name}.{name}' if module_name else name
                    for name in module._buffers if name not in module._non_persistent_buffers_set)

    prefix = 'diffusion.denoising.' if args.non_ema else 'diffusion_ema.denoising.'
    print(f'Loading {ckpt}')
    state_dict = _load_checkpoint(ckpt, map_location='cpu')['state_dict']
    found = {k[len(prefix):]: v.contiguous() for k, v in state_dict.items() if k.startswith(prefix)}
    dropped = sorted(set(found) - keep)
    missing = sorted(keep - set(found))
    out_dict = {
        k.replace('lora_A.default.weight', 'lora_A.weight').replace(
            'lora_B.default.weight', 'lora_B.weight'): found[k] for k in sorted(keep & set(found))}
    print(f'{len(out_dict)} tensors kept, {sum(v.numel() for v in out_dict.values()) / 1e6:.1f}M elements, '
          f'dtypes {sorted({str(v.dtype) for v in out_dict.values()})}')
    print('dropped:', dropped if dropped else 'none')
    if missing:
        print('WARNING missing from the checkpoint:', missing)

    if args.out.startswith('s3://'):
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as f:
            save_file(out_dict, f.name)
            FileClient.infer_client(uri=args.out).put(open(f.name, 'rb').read(), args.out)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_file(out_dict, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
