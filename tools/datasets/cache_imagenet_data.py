# Example command (SD-VAE):
# torchrun --nnodes=1 --nproc_per_node=8 tools/datasets/cache_imagenet_data.py configs/gmflow/gmdit_k8_imagenet_8gpus.py --random-flip --launcher pytorch --diff_seed

from lakonlab.utils.env import apply_default_env
apply_default_env()

import os
import argparse
import copy
import itertools
import multiprocessing as mp
import pickle
import tarfile
import tempfile
import tqdm
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from io import BytesIO

import torch
import torch.distributed as dist

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist

from lakonlab.models import build_module
from lakonlab.utils import get_root_logger, setup_global_config, setup_multi_processes
from lakonlab.utils.io_utils import S3Backend, FileClient
from lakonlab.datasets import build_dataset, build_dataloader
from lakonlab.runner.utils import set_random_seed


def normalize_moments(config, mean, std):
    if config.latents_mean is not None and config.latents_std is not None:
        latents_mean = torch.tensor(
            config.latents_mean, device=mean.device, dtype=mean.dtype)[:, None, None]
        latents_std = torch.tensor(
            config.latents_std, device=mean.device, dtype=mean.dtype)[:, None, None]
        return (mean - latents_mean) / latents_std, std / latents_std
    else:
        scaling_factor = config.scaling_factor
        shift_factor = config.shift_factor
        if scaling_factor is None:
            scaling_factor = 1.0
        if shift_factor is None:
            shift_factor = 0.0
        return (mean - shift_factor) * scaling_factor, std * scaling_factor


def save_cache(data, cache_path):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and '://' not in cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    file_client = FileClient.infer_client(uri=cache_path)
    bytesio = BytesIO()
    pickle.dump(data, bytesio, protocol=pickle.HIGHEST_PROTOCOL)
    file_client.put(bytesio.getvalue(), cache_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cache ImageNet latents from the VAE/encoder configured in a model config.')
    parser.add_argument('config', help='file path to the config containing model.vae')
    parser.add_argument(
        '--data',
        type=str,
        default='train',
        help='cfg.data split whose data_root/datalist_path define the cache target')
    parser.add_argument('--data-root', type=str, default='data/imagenet/train')
    parser.add_argument('--datalist-path', type=str, default='data/imagenet/train.txt')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per GPU')
    parser.add_argument('--workers-per-gpu', type=int, help='dataloader workers per GPU')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        help='override cfg.model.vae.torch_dtype')
    parser.add_argument(
        '--cache-std',
        action='store_true',
        help='store x_mean/x_std instead of a single sampled/encoded x tensor')
    parser.add_argument(
        '--cache-image-bytes',
        action='store_true',
        help='store raw image bytes in each cache entry')
    parser.add_argument(
        '--random-flip',
        action='store_true',
        help='whether to apply random horizontal flip while caching')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='whether to set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_multi_processes(cfg)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    setup_global_config(dict(cfg.get('global_config', dict())))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
        world_size = 1
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE', 1)
    num_cpus = mp.cpu_count() // int(local_world_size)
    max_save_cache_workers = max(num_cpus - 2, 1)

    logger = get_root_logger(log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    # build vae
    if hasattr(cfg.model, 'vae_2'):
        vae_cfg = copy.deepcopy(cfg.model.vae_2)
    else:
        vae_cfg = copy.deepcopy(cfg.model.vae)
    if args.dtype is not None:
        vae_cfg['torch_dtype'] = args.dtype
    vae = build_module(vae_cfg).requires_grad_(False).eval().cuda()
    if hasattr(vae, 'dtype'):
        vae_dtype = vae.dtype
    else:
        vae_dtype = next(vae.parameters()).dtype
    device = torch.device('cuda')

    # build dataset and dataloader
    dataset = cfg.data[args.data]
    out_data_root = dataset.data_root
    out_datalist_path = dataset.datalist_path
    for key in ['start_ind', 'end_ind', 'repeat', 'tar_extract_dir']:
        dataset.pop(key, None)
    dataset.update(
        data_root=args.data_root,
        datalist_path=args.datalist_path,
        return_image_bytes=args.cache_image_bytes,
        random_flip=args.random_flip,
        test_mode=False)
    dataset = build_dataset(dataset)

    workers_per_gpu = args.workers_per_gpu
    if workers_per_gpu is None:
        workers_per_gpu = cfg.data.get('workers_per_gpu', 8)
    batch_size = args.batch_size
    dataloader = build_dataloader(
        dataset,
        batch_size,
        workers_per_gpu,
        persistent_workers=workers_per_gpu > 0,
        prefetch_factor=max(1, batch_size // workers_per_gpu * 2) if workers_per_gpu > 0 else None,
        shuffle=False)

    # prepare output paths
    tar_output = out_data_root.lower().endswith(('.tar', '.tar.gz', '.tgz'))
    if tar_output:
        tar_name = os.path.basename(out_data_root)
        for suffix in ('.tar.gz', '.tgz', '.tar'):
            if tar_name.endswith(suffix):
                tar_name = tar_name[:-len(suffix)]
                break
        cache_data_root = os.path.join(tempfile.gettempdir(), tar_name)
    else:
        cache_data_root = out_data_root

    root_file_client = FileClient.infer_client(uri=cache_data_root)
    datalist_file_client = FileClient.infer_client(uri=out_datalist_path)
    cache_paths = {
        path: root_file_client.join_path(
            os.path.dirname(path),
            os.path.splitext(os.path.basename(path))[0] + '.pkl')
        for path in dataset.all_paths
    }

    torch.set_grad_enabled(False)

    mp_ctx = mp.get_context('spawn')
    proc_pool = ProcessPoolExecutor(
        max_workers=min(max_save_cache_workers, args.batch_size * 2),
        mp_context=mp_ctx)
    cap = max_save_cache_workers * 2
    pending = set()

    if rank == 0:
        mmcv.print_log(
            f'Cache output: data_root={out_data_root}, datalist_path={out_datalist_path}',
            logger=logger)
        mmcv.print_log(
            f'Cache format: cache_std={args.cache_std}, cache_image_bytes={args.cache_image_bytes}, random_flip={args.random_flip}',
            logger=logger)

    pbar = tqdm.tqdm(total=len(dataset), disable=rank != 0)

    for data in dataloader:
        images = data['images'].to(device=device, dtype=vae_dtype, non_blocking=True)
        labels = data['labels']
        ids = list(itertools.chain.from_iterable(data['ids'].data))
        paths = list(itertools.chain.from_iterable(data['paths'].data))
        image_bytes = (
            list(itertools.chain.from_iterable(data['image_bytes'].data))
            if args.cache_image_bytes else None)

        if args.cache_std:
            posterior = vae.vae.encode(images * 2 - 1).latent_dist
            means, stds = normalize_moments(vae.vae.config, posterior.mean, posterior.std)
        else:
            latents = vae.encode(images * 2 - 1).float()

        for batch_id, (data_id, label, path) in enumerate(zip(ids, labels, paths)):
            out_path = cache_paths[path]
            if args.cache_std:
                torch_data = dict(
                    x_mean=means[batch_id].cpu(),
                    x_std=stds[batch_id].cpu(),
                    y=label.clone())
            else:
                torch_data = dict(
                    x=latents[batch_id].cpu(),
                    y=label.clone())
            if args.cache_image_bytes:
                torch_data.update(
                    image_bytes=image_bytes[batch_id])

            if len(pending) >= cap:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()

            pending.add(proc_pool.submit(
                save_cache,
                torch_data,
                root_file_client.join_path(cache_data_root, out_path)))

        pbar.update(batch_size * world_size)

    if pending:
        done, pending = wait(pending)
        for f in done:
            f.result()
    proc_pool.shutdown(wait=True)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        if tar_output:
            mmcv.print_log(
                'Creating tar archive...', logger=logger)
            mode = 'w:gz' if out_data_root.lower().endswith(('.tar.gz', '.tgz')) else 'w'
            if out_data_root.startswith('s3://'):
                with tempfile.NamedTemporaryFile(suffix=os.path.basename(out_data_root), delete=False) as f:
                    local_tar_path = f.name
            elif '://' in out_data_root:
                raise NotImplementedError('Tar cache output only supports local filesystem paths and s3:// paths.')
            else:
                tar_parent = os.path.dirname(out_data_root)
                if tar_parent:
                    os.makedirs(tar_parent, exist_ok=True)
                local_tar_path = out_data_root
            try:
                with tarfile.open(local_tar_path, mode=mode) as tar:
                    for out_path in cache_paths.values():
                        tar.add(os.path.join(cache_data_root, out_path), arcname=out_path)
                if out_data_root.startswith('s3://'):
                    S3Backend().upload_file(local_tar_path, out_data_root)
            finally:
                if out_data_root.startswith('s3://'):
                    os.remove(local_tar_path)

        lines = []
        for label, path in zip(dataset.all_labels, dataset.all_paths):
            lines.append(f'{cache_paths[path]} {label:d}\n')
        datalist_file_client.put_text(''.join(lines), out_datalist_path)
        mmcv.print_log('Done!', logger=logger)
