# Copyright (c) 2026 Hansheng Chen

import numpy as np
import torch

from torch.utils.data import DistributedSampler as _DistributedSampler
from lakonlab.runner import sync_random_seed


def reverse_index_map(bucket_ids):
    bucket_map = dict()
    for data_id, bucket_id in enumerate(bucket_ids):
        if bucket_id not in bucket_map:
            bucket_map[bucket_id] = []
        bucket_map[bucket_id].append(data_id)
    return bucket_map


class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 samples_per_gpu=1,
                 seed=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)

        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu

        self.bucket_map = self.total_size_bucketwise = None
        if hasattr(dataset, 'bucket_ids'):
            self._init_bucket_sampler(dataset)
        else:
            self._init_sampler(dataset)

        self.seed = sync_random_seed(seed)
        self.skip_iter = 0

    def _init_sampler(self, dataset):
        data_len = len(dataset)
        # to avoid padding bug when meeting too small dataset
        if data_len < self.num_replicas * self.samples_per_gpu:
            raise ValueError(
                'You may use too small dataset and our distributed '
                'sampler cannot pad your dataset correctly. Please '
                'use fewer GPUs or smaller batch sizes per GPU.')

        num_batches = int(np.ceil(data_len / self.num_replicas / self.samples_per_gpu))
        self.num_samples = num_batches * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def _init_bucket_sampler(self, dataset):
        self.bucket_map = reverse_index_map(dataset.bucket_ids)
        self.bucket_map = dict(sorted(self.bucket_map.items()))  # sort by bucket_id

        data_len = 0
        self.total_size_bucketwise = {}

        for bucket_id, data_indices in self.bucket_map.items():
            _data_len = len(data_indices)
            if _data_len < self.samples_per_gpu:
                raise ValueError(
                    'You may use too small dataset and our distributed '
                    'sampler cannot pad your dataset correctly. Please '
                    'use smaller batch sizes per GPU.')

            _total_num_batches = int(np.ceil(_data_len / self.samples_per_gpu))
            _total_size = _total_num_batches * self.samples_per_gpu

            data_len += _total_size
            self.total_size_bucketwise[bucket_id] = _total_size

        if data_len < self.num_replicas * self.samples_per_gpu:
            raise ValueError(
                'You may use too small dataset and our distributed '
                'sampler cannot pad your dataset correctly. Please '
                'use fewer GPUs or smaller batch sizes per GPU.')

        num_batches = int(np.ceil(data_len / self.num_replicas / self.samples_per_gpu))
        self.num_samples = num_batches * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def update_sampler(self, dataset, samples_per_gpu=None):
        self.dataset = dataset
        if samples_per_gpu is not None:
            self.samples_per_gpu = samples_per_gpu
        self.bucket_map = self.total_size_bucketwise = None
        if hasattr(dataset, 'bucket_ids'):
            self._init_bucket_sampler(dataset)
        else:
            self._init_sampler(dataset)

    def set_iter(self, iteration):
        num_batches = self.num_samples // self.samples_per_gpu
        self.skip_iter = iteration % num_batches

    def __iter__(self):
        if self.bucket_map is None:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = torch.arange(len(self.dataset)).tolist()
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]

        else:  # guarantees that batch samples are from the same bucket
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
            else:
                g = None
            indices = []
            for bucket_id, data_indices in self.bucket_map.items():
                data_indices = torch.tensor(data_indices)
                if g is not None:
                    data_indices = data_indices[torch.randperm(len(data_indices), generator=g)]
                pad = self.total_size_bucketwise[bucket_id] - data_indices.numel()
                if pad:
                    data_indices = torch.cat([data_indices, data_indices[:pad]], dim=0)
                assert data_indices.numel() == self.total_size_bucketwise[bucket_id]
                _total_num_batches = self.total_size_bucketwise[bucket_id] // self.samples_per_gpu
                _num_batches = _total_num_batches // self.num_replicas
                _total_leftover_batches = _total_num_batches % self.num_replicas
                # data_indices_a: evenly split batches for full round-robins across replicas
                # data_indices_b: the leftover partial round-robin
                data_indices_a = data_indices[:(_num_batches * self.num_replicas * self.samples_per_gpu)].reshape(
                    _num_batches, self.samples_per_gpu, self.num_replicas
                ).permute(0, 2, 1).reshape(
                    _num_batches * self.num_replicas, self.samples_per_gpu)
                data_indices_b = data_indices[(_num_batches * self.num_replicas * self.samples_per_gpu):].reshape(
                    self.samples_per_gpu, _total_leftover_batches
                ).permute(1, 0)
                indices.extend([data_indices_a, data_indices_b])
            indices = torch.cat(indices, dim=0)  # (total_num_batches, samples_per_gpu)
            if g is not None:
                indices = indices[torch.randperm(indices.size(0), generator=g)]
            total_num_batches = self.total_size // self.samples_per_gpu
            pad = total_num_batches - indices.size(0)
            if pad:
                indices = torch.cat([indices, indices[:pad]], dim=0)
            assert indices.numel() == self.total_size
            indices = indices[self.rank:total_num_batches:self.num_replicas].flatten().tolist()

        assert len(indices) == self.num_samples
        skip_len = self.skip_iter * self.samples_per_gpu
        assert skip_len < self.num_samples
        indices = indices[skip_len:]
        self.skip_iter = 0

        return iter(indices)
