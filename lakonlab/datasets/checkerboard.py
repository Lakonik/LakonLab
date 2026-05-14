# Copyright (c) 2026 Hansheng Chen

import torch

from torch.utils.data import Dataset
from .builder import DATASETS


@DATASETS.register_module()
class CheckerboardData(Dataset):
    def __init__(
            self,
            n_rc=4,
            n_samples=1e8,
            thickness=1.0,
            scale=1,
            shift=[0.0, 0.0],
            rotation=0.0,
            test_mode=False):
        super().__init__()
        self.n_rc = n_rc
        self.n_samples = int(n_samples)
        self.thickness = thickness
        self.scale = scale
        self.shift = torch.tensor(shift, dtype=torch.float32)
        self.rotation = rotation
        white_squares = [(i, j) for i in range(n_rc) for j in range(n_rc) if (i + j) % 2 == 0]
        self.white_squares = torch.tensor(white_squares, dtype=torch.float32)
        self.n_squares = len(white_squares)
        self.samples = self.draw_samples(self.n_samples)

    def draw_samples(self, n_samples):
        chosen_indices = torch.randint(0, self.n_squares, size=(n_samples, ))
        chosen_squares = self.white_squares[chosen_indices]
        square_samples = torch.rand(n_samples, 2, dtype=torch.float32)
        if self.thickness < 1:
            square_samples = square_samples - 0.5
            square_samples_r = square_samples.square().sum(dim=-1, keepdims=True)
            square_samples_angle = torch.atan2(square_samples[:, 1], square_samples[:, 0]).unsqueeze(-1)
            max_r = torch.minimum(
                0.5 / square_samples_angle.cos().abs().clamp(min=1e-6),
                0.5 / square_samples_angle.sin().abs().clamp(min=1e-6)).square()
            square_samples_r_scaled = max_r - (max_r - square_samples_r) * self.thickness ** 0.5
            square_samples *= (square_samples_r_scaled / square_samples_r).sqrt()
            square_samples = square_samples + 0.5
        samples = (chosen_squares + square_samples) * (2 / self.n_rc) - 1
        if self.rotation != 0.0:
            angle = torch.tensor(self.rotation, dtype=torch.float32) * torch.pi / 180
            rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                            [torch.sin(angle), torch.cos(angle)]])
            samples = samples @ rotation_matrix
        return samples * self.scale + self.shift

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        data = dict(x=self.samples[idx])
        return data
