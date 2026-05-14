import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

import os
import itertools
import argparse
import multiprocessing as mp
from io import BytesIO

import torch
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import mmcv
from mmcv.runner import get_dist_info, init_dist
from mmcv.fileio import FileClient

from lakonlab.datasets import ImageNet, build_dataloader
from lakonlab.models import PretrainedVAEEncoder
from lakonlab.runner.utils import set_random_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cache the image latents for the ImageNet dataset using SD-VAE.')
    parser.add_argument('--data-root', type=str, default='data/imagenet/train')
    parser.add_argument('--datalist-path', type=str, default='data/imagenet/train.txt')
    parser.add_argument('--out-data-root', type=str, default='data/imagenet/train_cache')
    parser.add_argument('--out-datalist-path', type=str, default='data/imagenet/train_cache.txt')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per GPU')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()

    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    init_dist('pytorch')
    rank, ws = get_dist_info()

    if args.seed is not None:
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=True)

    dataset = ImageNet(
        data_root=args.data_root,
        datalist_path=args.datalist_path,
        image_size=args.image_size)

    dataloader = build_dataloader(
        dataset, args.batch_size, 8,
        persistent_workers=True, prefetch_factor=max(1, args.batch_size // 4), dist=True, shuffle=False)

    encoder = PretrainedVAEEncoder(
        model_name_or_path='stabilityai/sd-vae-ft-ema', torch_dtype=args.dtype).eval().cuda()

    root_file_client = FileClient.infer_client(uri=args.out_data_root)
    datalist_file_client = FileClient.infer_client(uri=args.out_datalist_path)

    torch.set_grad_enabled(False)

    if rank == 0:
        pbar = mmcv.ProgressBar(len(dataset))

    for data in dataloader:
        images = data['images'].to(dtype=getattr(torch, args.dtype)).cuda()
        labels = data['labels']
        paths = data['paths']

        latents = encoder(images * 2 - 1)

        for latent, label, path in zip(latents, labels, list(itertools.chain.from_iterable(paths.data))):
            out_path = root_file_client.join_path(
                os.path.dirname(path),
                os.path.splitext(os.path.basename(path))[0] + '.pth'
            )
            torch_data = dict(x=latent.cpu(), y=label.cpu())
            bytesio = BytesIO()
            torch.save(torch_data, bytesio)
            root_file_client.put(
                bytesio.getvalue(),
                root_file_client.join_path(args.out_data_root, out_path))

        if rank == 0:
            pbar.update(args.batch_size * ws)

    dist.barrier()

    if rank == 0:
        lines = []
        for label, path in zip(dataset.all_labels, dataset.all_paths):
            out_path = root_file_client.join_path(
                os.path.dirname(path),
                os.path.splitext(os.path.basename(path))[0] + '.pth'
            )
            lines.append(f'{out_path} {label:d}\n')
        datalist_file_client.put_text(''.join(lines), args.out_datalist_path)
        print('Done!')
