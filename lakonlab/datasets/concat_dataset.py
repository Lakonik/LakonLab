# Copyright (c) 2026 Hansheng Chen

import numpy as np

from torch.utils.data import Dataset
from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class ConcatDataset(Dataset):
    """Concatenation of datasets, optionally sharing their buckets.

    Args:
        datasets (list[dict]): Datasets to concatenate.
        global_bucket_ids (bool): If True, a bucket id means the same thing in every dataset, so
            equal ids merge into one bucket and a batch may mix datasets -- which is what several
            sources of the same resolution want. The default offsets each dataset's ids instead,
            keeping every dataset's buckets to itself. Datasets sharing an id must agree on its
            `bucket_samples_per_gpu`.
    """

    def __init__(
            self,
            datasets,
            global_bucket_ids=False):
        super().__init__()

        self.datasets = [build_dataset(ds) for ds in datasets]
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

        has_bucket_ids = [hasattr(ds, 'bucket_ids') for ds in self.datasets]
        if any(has_bucket_ids):
            if not all(has_bucket_ids):
                raise ValueError(
                    'If you use bucketized sampling, all datasets should have '
                    'bucket_ids attribute.')
            bucket_ids = []
            has_bucket_samples_per_gpu = [
                hasattr(ds, 'bucket_samples_per_gpu') for ds in self.datasets]
            if any(has_bucket_samples_per_gpu):
                if not all(has_bucket_samples_per_gpu):
                    raise ValueError(
                        'If you use bucket-wise samples_per_gpu, all datasets should have '
                        'bucket_samples_per_gpu attribute.')
                self.bucket_samples_per_gpu = {}
            base_bucket_id = 0
            for ds in self.datasets:
                bucket_ids.append(np.asarray(ds.bucket_ids, dtype=np.int16) + base_bucket_id)
                if hasattr(self, 'bucket_samples_per_gpu'):
                    for bucket_id, samples_per_gpu in ds.bucket_samples_per_gpu.items():
                        bucket_id += base_bucket_id
                        if self.bucket_samples_per_gpu.setdefault(bucket_id, samples_per_gpu) != samples_per_gpu:
                            raise ValueError(
                                f'Datasets sharing bucket {bucket_id} disagree on '
                                f'samples_per_gpu: {self.bucket_samples_per_gpu[bucket_id]} and '
                                f'{samples_per_gpu}.')
                if not global_bucket_ids:
                    base_bucket_id += int(np.max(ds.bucket_ids)) + 1
            self.bucket_ids = np.concatenate(bucket_ids)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
