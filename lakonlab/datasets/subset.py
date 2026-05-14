# Copyright (c) 2026 Hansheng Chen

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
        self.indices = indices

        if hasattr(self.dataset, 'bucket_ids'):
            self.bucket_ids = [self.dataset.bucket_ids[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
