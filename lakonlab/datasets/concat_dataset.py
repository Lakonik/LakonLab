# Copyright (c) 2026 Hansheng Chen

import numpy as np

from torch.utils.data import Dataset
from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class ConcatDataset(Dataset):
    def __init__(
            self,
            datasets):
        super().__init__()

        self.datasets = [build_dataset(ds) for ds in datasets]
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

        has_bucket_ids = [hasattr(ds, 'bucket_ids') for ds in self.datasets]
        if any(has_bucket_ids):
            if not all(has_bucket_ids):
                raise ValueError(
                    'If you use bucketized sampling, all datasets should have '
                    'bucket_ids attribute.')
            self.bucket_ids = []
            base_bucket_id = 0
            for ds in self.datasets:
                self.bucket_ids.extend([base_bucket_id + b for b in ds.bucket_ids])
                base_bucket_id += max(ds.bucket_ids) + 1

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
