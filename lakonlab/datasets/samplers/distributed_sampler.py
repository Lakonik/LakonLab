# Copyright (c) 2026 Hansheng Chen

import numpy as np
import torch

from torch.utils.data import DistributedSampler as _DistributedSampler
from lakonlab.runner import sync_random_seed


def reverse_index_map(bucket_ids):
    """bucket id -> the dataset indices in it, as an int32 array.

    Arrays rather than lists of Python ints: a dataset weighted up by `repeat` can hold a billion
    samples, where boxed ints or int64 indices cost GBs per rank. A single bucket covering the
    whole dataset is left as None, so `__iter__` can shuffle without materializing an arange and
    gathering through it.
    """
    bucket_ids = np.asarray(bucket_ids)
    unique = np.unique(bucket_ids)
    if len(unique) == 1:
        return {int(unique[0]): None}
    return {int(bucket_id): np.flatnonzero(bucket_ids == bucket_id).astype(np.int32) for bucket_id in unique}


def bucket_indices(data_indices, num_samples, generator=None):
    """The bucket's dataset indices, shuffled when a generator is given.

    `data_indices` is None when the bucket is the whole dataset, where a permutation is already
    the answer: gathering an arange through it would give the same order for twice the memory.
    """
    if generator is None:
        return (torch.arange(num_samples, dtype=torch.int32) if data_indices is None
                else torch.from_numpy(data_indices))
    permutation = torch.randperm(num_samples, generator=generator, dtype=torch.int32)
    return permutation if data_indices is None else torch.from_numpy(data_indices)[permutation]


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
        self.bucket_sizes = {
            bucket_id: len(dataset) if indices is None else len(indices)
            for bucket_id, indices in self.bucket_map.items()}

        data_len = 0
        self.total_size_bucketwise = {}

        for bucket_id, data_indices in self.bucket_map.items():
            _data_len = self.bucket_sizes[bucket_id]
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
                indices = torch.randperm(len(self.dataset), generator=g, dtype=torch.int32)
            else:
                indices = torch.arange(len(self.dataset), dtype=torch.int32)
            # add extra samples to make it evenly divisible
            pad = self.total_size - indices.numel()
            if pad:
                indices = torch.cat([indices, indices[:pad]], dim=0)
            assert indices.numel() == self.total_size
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas].numpy()

        else:  # guarantees that batch samples are from the same bucket
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
            else:
                g = None
            indices = []
            for bucket_id, data_indices in self.bucket_map.items():
                data_indices = bucket_indices(data_indices, self.bucket_sizes[bucket_id], generator=g)
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
                indices.extend(part for part in (data_indices_a, data_indices_b) if part.numel())
                del data_indices
            # (total_num_batches, samples_per_gpu)
            indices = indices[0] if len(indices) == 1 else torch.cat(indices, dim=0)
            total_num_batches = self.total_size // self.samples_per_gpu
            # shuffle and pad the batch order, then gather this rank's batches alone: gathering
            # the whole matrix would copy every index, which is GBs at a billion samples
            order = torch.arange(indices.size(0), dtype=torch.int32) if g is None else torch.randperm(
                indices.size(0), generator=g, dtype=torch.int32)
            pad = total_num_batches - order.numel()
            if pad:
                order = torch.cat([order, order[:pad]], dim=0)
            assert order.numel() == total_num_batches
            indices = indices[order[self.rank:total_num_batches:self.num_replicas]].flatten().numpy()

        assert len(indices) == self.num_samples
        skip_len = self.skip_iter * self.samples_per_gpu
        assert skip_len < self.num_samples
        indices = indices[skip_len:]
        self.skip_iter = 0

        # boxed one at a time as the loader asks for them: `tolist()` up front would cost GBs per
        # rank, while a numpy index breaks anything a dataset feeds an int-only API
        return map(int, indices)


class DistributedBucketBatchSampler:
    """Distributed batch sampler with per-bucket local batch sizes."""

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            seed=None):
        if not hasattr(dataset, 'bucket_ids'):
            raise ValueError('DistributedBucketBatchSampler requires dataset.bucket_ids.')
        if not hasattr(dataset, 'bucket_samples_per_gpu'):
            raise ValueError('DistributedBucketBatchSampler requires dataset.bucket_samples_per_gpu.')

        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = sync_random_seed(seed)
        self.epoch = 0
        self.skip_iter = 0

        self.bucket_samples_per_gpu = dict(dataset.bucket_samples_per_gpu)
        self._init_batch_sampler(dataset)

    def _init_batch_sampler(self, dataset):
        self.bucket_map = reverse_index_map(dataset.bucket_ids)
        self.bucket_map = dict(sorted(self.bucket_map.items()))
        self.bucket_sizes = {
            bucket_id: len(dataset) if indices is None else len(indices)
            for bucket_id, indices in self.bucket_map.items()}

        if set(self.bucket_samples_per_gpu) != set(self.bucket_map):
            raise ValueError('bucket_samples_per_gpu keys must match bucket ids.')

        self.total_num_batches_bucketwise = {}
        total_num_batches = 0
        for bucket_id, data_indices in self.bucket_map.items():
            samples_per_gpu = self.bucket_samples_per_gpu[bucket_id]
            _data_len = self.bucket_sizes[bucket_id]
            if _data_len < samples_per_gpu:
                raise ValueError(
                    'You may use too small dataset and our distributed '
                    'sampler cannot pad your dataset correctly. Please '
                    'use smaller per-bucket samples_per_gpu.')

            _total_num_batches = int(np.ceil(_data_len / samples_per_gpu))
            self.total_num_batches_bucketwise[bucket_id] = _total_num_batches
            total_num_batches += _total_num_batches

        if total_num_batches < self.num_replicas:
            raise ValueError(
                'You may use too small dataset and our distributed '
                'batch sampler cannot pad your dataset correctly. Please '
                'use fewer GPUs or smaller per-bucket samples_per_gpu.')

        self.num_batches = int(np.ceil(total_num_batches / self.num_replicas))
        self.total_num_batches = self.num_batches * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_iter(self, iteration):
        self.skip_iter = iteration % self.num_batches

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
        else:
            g = None

        batches = []
        for bucket_id, data_indices in self.bucket_map.items():
            samples_per_gpu = self.bucket_samples_per_gpu[bucket_id]
            data_indices = bucket_indices(data_indices, self.bucket_sizes[bucket_id], generator=g)

            total_num_batches = self.total_num_batches_bucketwise[bucket_id]
            total_size = total_num_batches * samples_per_gpu
            pad = total_size - data_indices.numel()
            if pad:
                data_indices = torch.cat([data_indices, data_indices[:pad]], dim=0)
            assert data_indices.numel() == total_size

            _num_batches = total_num_batches // self.num_replicas
            _total_leftover_batches = total_num_batches % self.num_replicas
            split = _num_batches * self.num_replicas * samples_per_gpu
            data_indices_a = data_indices[:split].reshape(
                _num_batches, samples_per_gpu, self.num_replicas
            ).permute(0, 2, 1).reshape(
                _num_batches * self.num_replicas, samples_per_gpu)
            data_indices_b = data_indices[split:].reshape(
                samples_per_gpu, _total_leftover_batches).permute(1, 0)
            batches.extend(part.numpy() for part in (data_indices_a, data_indices_b) if part.numel())
            del data_indices

        # buckets have their own samples_per_gpu, so the batch matrices cannot be concatenated;
        # locate this rank's batches using cumulative matrix sizes
        matrix_ends = np.cumsum([len(matrix) for matrix in batches])

        order = np.arange(matrix_ends[-1], dtype=np.int32) if g is None else torch.randperm(
            int(matrix_ends[-1]), generator=g, dtype=torch.int32).numpy()
        pad = self.total_num_batches - len(order)
        if pad:
            order = np.concatenate([order, order[:pad]])
        assert len(order) == self.total_num_batches

        order = order[self.rank:self.total_num_batches:self.num_replicas]
        assert len(order) == self.num_batches

        skip_len = self.skip_iter
        assert skip_len < self.num_batches
        order = order[skip_len:]
        self.skip_iter = 0

        matrix_ids = np.searchsorted(matrix_ends, order, side='right').astype(np.int32)
        row_ids = (order - np.concatenate([[0], matrix_ends[:-1]])[matrix_ids]).astype(np.int32)
        for matrix_id, matrix in enumerate(batches):
            selected = np.flatnonzero(matrix_ids == matrix_id)
            batches[matrix_id] = matrix[row_ids[selected]]
            row_ids[selected] = np.arange(len(selected), dtype=np.int32)
        return (batches[matrix_id][row_id].tolist() for matrix_id, row_id in zip(matrix_ids, row_ids))

    def __len__(self):
        return self.num_batches
