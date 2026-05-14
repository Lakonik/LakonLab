import platform
import random
import warnings
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version

from .samplers import DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')


def unique_dataloaders(data_loaders):
    """Yield dataloader objects once while preserving order."""
    seen = set()
    for data_loader in data_loaders:
        key = id(data_loader)
        if key in seen:
            continue
        seen.add(key)
        yield data_loader


def build_dataset(cfg, default_args=None):
    """Build dataset.

    Args:
        cfg (dict): Config for the dataset.
        default_args (dict | None, optional): Default arguments.
            Defaults to None.

    Returns:
        Object: Dataset for sampling data batch.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     persistent_workers=False,
                     sampler=None,
                     **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        assert sampler is None, 'sampler is not supported in distributed mode'
        sampler = DistributedSampler(
            dataset,
            world_size,
            rank,
            shuffle=shuffle,
            samples_per_gpu=samples_per_gpu,
            seed=seed)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (digit_version(TORCH_VERSION) >= digit_version('1.7.0')
            and TORCH_VERSION != 'parrots'):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
