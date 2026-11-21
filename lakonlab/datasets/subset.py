# Copyright (c) 2026 Hansheng Chen

import numpy as np

from torch.utils.data import Dataset
from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class Subset(Dataset):
    def __init__(
            self,
            dataset,
            indices):
        super().__init__()

        self.dataset = dataset if isinstance(dataset, Dataset) else build_dataset(dataset)
        self.indices = np.asarray(indices, dtype=np.int64)

        if hasattr(self.dataset, 'bucket_ids'):
            # arrays rather than lists of Python ints, which cost GBs per rank on a large dataset
            self.bucket_ids = np.asarray(self.dataset.bucket_ids, dtype=np.int16)[self.indices]
            if hasattr(self.dataset, 'bucket_samples_per_gpu'):
                self.bucket_samples_per_gpu = {
                    bucket_id: self.dataset.bucket_samples_per_gpu[bucket_id]
                    for bucket_id in np.unique(self.bucket_ids).tolist()
                }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]
