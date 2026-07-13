# Copyright (c) 2026 Hansheng Chen

import argparse
from copy import deepcopy

import torch
from mmcv import Config, DictAction
from tqdm import tqdm

from lakonlab.datasets import build_dataset, build_dataloader
from lakonlab.models import build_module
from lakonlab.runner.checkpoint import write_checkpoint_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a patch PCA subspace for AsymFlow models using the configured encoder and DiT patch convention (channel last).')
    parser.add_argument('config', help='AsymFlow ImageNet training config file.')
    parser.add_argument('--out', help='Output checkpoint path. Defaults to `pretrained_linear_proj` in the config.')
    parser.add_argument('--num-images', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, help='Batch size. Defaults to the config train dataloader.')
    parser.add_argument('--workers', type=int, help='DataLoader workers. Defaults to the config workers_per_gpu.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, using key=value pairs.')
    return parser.parse_args()


def patchify_dit(latents, patch_size: int):
    bs, c, h, w = latents.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f'Latent size {(h, w)} is not divisible by patch size {patch_size}.')
    return latents.reshape(
        bs, c, h // patch_size, patch_size, w // patch_size, patch_size
    ).permute(
        0, 3, 5, 1, 2, 4
    ).reshape(
        bs, patch_size * patch_size * c, h // patch_size, w // patch_size)


def compute_latent_gram(cfg, args, patch_size, device):
    dataset_cfg = deepcopy(cfg.data.train)
    dataset = build_dataset(dataset_cfg)
    batch_size = args.batch_size or cfg.data.train_dataloader.get('samples_per_gpu', 1)
    workers = args.workers if args.workers is not None else cfg.data.get('workers_per_gpu', 1)
    dataloader_kwargs = dict(
        dist=True,
        shuffle=True,
        seed=args.seed,
        persistent_workers=workers > 0,
    )
    if workers > 0 and cfg.data.get('prefetch_factor', None) is not None:
        dataloader_kwargs['prefetch_factor'] = cfg.data.prefetch_factor
    dataloader = build_dataloader(
        dataset,
        batch_size,
        workers,
        **dataloader_kwargs,
    )
    encoder = build_module(cfg.model.vae).to(device).eval()
    if hasattr(encoder, 'dtype'):
        encoder_dtype = encoder.dtype
    else:
        encoder_dtype = next(encoder.parameters()).dtype

    latent_gram = None
    num_seen = 0
    progress = tqdm(total=min(args.num_images, len(dataset)), disable=args.no_progress)

    for data in dataloader:
        if num_seen >= args.num_images:
            break

        remaining = args.num_images - num_seen

        if 'latents' in data:
            latents = data['latents'].to(device=device, dtype=torch.float32)
        elif 'images' in data:
            images = data['images'].to(device=device, dtype=torch.float32)
            latents = encoder.encode((images * 2 - 1).to(encoder_dtype)).float()
        else:
            raise KeyError(
                'Dataset batch contains neither `latents` nor `images`; cannot build PCA samples.')

        if latents.size(0) > remaining:
            latents = latents[:remaining]

        patches = patchify_dit(latents, patch_size)
        patches = patches.permute(0, 2, 3, 1).reshape(-1, patches.size(1)).double()

        if latent_gram is None:
            dim = patches.size(1)
            latent_gram = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        latent_gram += patches.T @ patches
        num_seen += latents.size(0)
        progress.update(latents.size(0))
    progress.close()

    if latent_gram is None:
        raise RuntimeError('No images were processed.')
    return latent_gram


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    denoising_cfg = cfg.model.diffusion.denoising
    patch_size = denoising_cfg.patch_size
    rank = denoising_cfg.base_rank

    out = args.out
    if out is None:
        out = denoising_cfg.get(
            'pretrained_linear_proj', 'checkpoints/asymflow_subspace_pca_dit.pth')

    torch.set_grad_enabled(False)
    latent_gram = compute_latent_gram(cfg, args, patch_size, device)
    dim = latent_gram.size(0)
    if rank > dim:
        raise ValueError(f'Rank {rank} exceeds patch dimension {dim}.')

    _, evecs = torch.linalg.eigh(latent_gram)
    proj_mat = evecs[:, -rank:].contiguous().float().cpu()

    write_checkpoint_to_file({f'proj_mat_p{patch_size}': proj_mat}, out)
    print(f'Saved rank-{rank} patch-{patch_size} PCA subspace to {out}')


if __name__ == '__main__':
    main()
