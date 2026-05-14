# Copyright (c) 2026 Hansheng Chen

import argparse
from copy import deepcopy

import numpy as np
import torch
from mmcv import Config, DictAction
from PIL import Image
from tqdm import tqdm

from lakonlab.datasets import build_dataset, build_dataloader
from lakonlab.models import build_module
from lakonlab.runner.checkpoint import write_checkpoint_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a Procrustes latent-to-pixel subspace for pixel AsymFlow finetuning.')
    parser.add_argument('config', help='AsymFLUX.2 klein training config file.')
    parser.add_argument('--out', help='Output checkpoint path. Defaults to `pretrained_linear_proj` in the config.')
    parser.add_argument('--num-images', type=int, default=1000)
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


def patchify_channel_first(x, patch_size: int):
    bs, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f'Image size {(h, w)} is not divisible by patch size {patch_size}.')
    return x.reshape(
        bs, c, h // patch_size, patch_size, w // patch_size, patch_size
    ).permute(
        0, 1, 3, 5, 2, 4
    ).reshape(
        bs, c * patch_size * patch_size, h // patch_size, w // patch_size)


def fit_scale(pixel_gram: torch.Tensor, proj_mat: torch.Tensor, latent_norm_sq: torch.Tensor):
    projected_pixel_norm_sq = torch.trace(proj_mat.T @ pixel_gram @ proj_mat)
    return (projected_pixel_norm_sq / latent_norm_sq.clamp(min=1e-12)).clamp(min=1e-12).sqrt()


def resize_images_if_needed(images: torch.Tensor, target_size):
    target_h, target_w = target_size
    if images.shape[-2:] == (target_h, target_w):
        return images

    resized_images = []
    for image in images:
        image_np = (image.permute(1, 2, 0).clamp(0, 1) * 255.0).round().to(
            torch.uint8).cpu().numpy()
        resized = Image.fromarray(image_np).resize(
            (target_w, target_h), resample=Image.LANCZOS)
        resized_images.append(torch.from_numpy(np.asarray(resized)).permute(2, 0, 1))
    return torch.stack(resized_images, dim=0).to(
        device=images.device, dtype=images.dtype) / 255.0


def build_token_pairs(encoder, latents, images, latent_patch_size, patch_size):
    latent_patches = patchify_channel_first(latents, latent_patch_size)
    h, w = latent_patches.shape[-2:]
    latent_tokens = latent_patches.permute(0, 2, 3, 1).reshape(
        -1, latent_patches.size(1)).double()

    images = resize_images_if_needed(images, (h * patch_size, w * patch_size))
    pixel_patches = encoder.encode(images * 2 - 1).float()
    pixel_patches = patchify_channel_first(pixel_patches, patch_size)
    pixel_tokens = pixel_patches.permute(0, 2, 3, 1).reshape(
        -1, pixel_patches.size(1)).double()
    return latent_tokens, pixel_tokens


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
    latent_patch_size = cfg.model.diffusion.latent_patch_size

    out = args.out
    if out is None:
        out = denoising_cfg.get(
            'pretrained_linear_proj', 'checkpoints/asymflow_subspace_procrustes.pth')

    torch.set_grad_enabled(False)

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

    cross_gram = None
    pixel_gram = None
    latent_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    num_seen = 0

    progress = tqdm(total=min(args.num_images, len(dataset)), disable=args.no_progress)
    for data in dataloader:
        if num_seen >= args.num_images:
            break

        latents = data['latents'].to(device=device, dtype=torch.float32)
        images = data['images'].to(device=device, dtype=torch.float32)
        remaining = args.num_images - num_seen
        if latents.size(0) > remaining:
            latents = latents[:remaining]
            images = images[:remaining]

        latent_tokens, pixel_tokens = build_token_pairs(
            encoder, latents, images, latent_patch_size, patch_size)

        if cross_gram is None:
            d_latent = latent_tokens.size(1)
            d_pixel = pixel_tokens.size(1)
            if d_latent > d_pixel:
                raise ValueError(
                    f'Latent patch dimension {d_latent} exceeds pixel patch dimension {d_pixel}.')
            cross_gram = torch.zeros((d_latent, d_pixel), dtype=torch.float64, device=device)
            pixel_gram = torch.zeros((d_pixel, d_pixel), dtype=torch.float64, device=device)

        cross_gram += latent_tokens.T @ pixel_tokens
        pixel_gram += pixel_tokens.T @ pixel_tokens
        latent_norm_sq += (latent_tokens * latent_tokens).sum()
        num_seen += latents.size(0)
        progress.update(latents.size(0))
    progress.close()

    if cross_gram is None:
        raise RuntimeError('No samples were processed.')

    u, _, vh = torch.linalg.svd(cross_gram, full_matrices=False)
    rank = cross_gram.size(0)
    proj_mat = (vh[:rank].T @ u[:, :rank].T).contiguous()
    scale = fit_scale(pixel_gram, proj_mat, latent_norm_sq)

    write_checkpoint_to_file({
        f'proj_mat_p{patch_size}': proj_mat.float().cpu(),
        f'scale_p{patch_size}': scale.float().cpu(),
    }, out)
    print(f'Saved patch-{patch_size} Procrustes subspace to {out}')


if __name__ == '__main__':
    main()
