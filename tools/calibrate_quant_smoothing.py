# Copyright (c) 2026 Hansheng Chen

import warnings

warnings.filterwarnings(
    'ignore',
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    category=UserWarning)

import argparse
import copy
import os
import tempfile
from copy import deepcopy

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from safetensors.torch import save_file

from lakonlab.datasets import build_dataset, build_dataloader
from lakonlab.models import build_model
from lakonlab.utils.io_utils import FileClient


def parse_args():
    parser = argparse.ArgumentParser(
        description='Record the per-input-channel activation absmax of every linear layer of a config\'s denoiser '
                    'while sampling a few items of one of its data splits with its first method, for SmoothQuant '
                    'smoothing (`freeze_quant_config=dict(type="int8_smoothquant", smooth_stats=<out>)`). The config\'s own quantization is '
                    'disabled for the pass, so the statistics describe the unquantized model.')
    parser.add_argument('config')
    parser.add_argument('--out', required=True, help='Output .safetensors path, local or s3://')
    parser.add_argument('--data', default='test', help='Key in cfg.data to sample from')
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--batch-size', type=int, help='Defaults to the split\'s dataloader samples_per_gpu.')
    parser.add_argument('--workers', type=int, help='Defaults to the config workers_per_gpu.')
    parser.add_argument('--seed', type=int, default=2021)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.diffusion.denoising.pop('freeze_quant_config', None)
    for module_cfg in (cfg.model.diffusion.denoising, cfg.model.get('text_encoder', {})):
        if 'compile_forward' in module_cfg:  # only override an existing option
            module_cfg.compile_forward = False
    method = next(iter(cfg.methods.values()))
    torch.manual_seed(args.seed)  # stochastic samplers draw from the global rng

    dataset = build_dataset(deepcopy(cfg.data[args.data]))
    batch_size = args.batch_size or cfg.data.get(f'{args.data}_dataloader', cfg.data.get('val_dataloader', {})).get('samples_per_gpu', 1)
    workers = args.workers if args.workers is not None else cfg.data.get('workers_per_gpu', 1)
    dataloader = build_dataloader(dataset, batch_size, workers, shuffle=True, seed=args.seed, persistent_workers=False)

    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).eval()
    denoiser = model.diffusion_ema.denoising if cfg.model.get('diffusion_use_ema') else model.diffusion.denoising
    model = MMDataParallel(model.cuda(), device_ids=[0])

    stats = dict()

    def record(name):
        def hook(module, inputs):
            absmax = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).abs().amax(dim=0).float()
            stats[name] = absmax if name not in stats else torch.maximum(stats[name], absmax)
        return hook

    handles = [m.register_forward_pre_hook(record(name))
               for name, m in denoiser.named_modules() if m.__class__ is nn.Linear]

    num_seen = 0
    for data in dataloader:
        outputs = model.val_step(data, test_cfg_override=copy.deepcopy(method))  # every step and the guidance batch
        num_seen += outputs['num_samples']  # bucketed datasets may override the batch size
        print(f'{min(num_seen, args.num_samples)}/{args.num_samples} samples', flush=True)
        if num_seen >= args.num_samples:
            break
    for handle in handles:
        handle.remove()

    stats = {k: v.cpu().contiguous() for k, v in stats.items()}
    print(f'{len(stats)} layers; activation absmax range '
          f'{min(v.min().item() for v in stats.values()):.3g} .. {max(v.max().item() for v in stats.values()):.3g}')
    if args.out.startswith('s3://'):
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as f:
            save_file(stats, f.name)
            FileClient.infer_client(uri=args.out).put(open(f.name, 'rb').read(), args.out)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_file(stats, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
