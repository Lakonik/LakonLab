# Copyright (c) 2026 Hansheng Chen

import torch

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from ...builder import MODULES


@MODULES.register_module()
class OklabColorEncoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        use_affine_norm=True,
        mean=(0.5, 0.0, 0.0),
        std=0.21,
    ):
        super().__init__()
        self.use_affine_norm = use_affine_norm
        self.register_buffer('lrgb_to_lms', torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ], dtype=torch.float32))
        self.register_buffer('lms_to_oklab', torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660]
        ], dtype=torch.float32))
        self.register_buffer('oklab_to_lms', torch.linalg.inv(self.lms_to_oklab))
        self.register_buffer('lms_to_lrgb', torch.linalg.inv(self.lrgb_to_lms))
        if self.use_affine_norm:
            self.register_buffer('affine_mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer('affine_std', torch.tensor(std, dtype=torch.float32))

    @property
    def dtype(self):
        return self.lrgb_to_lms.dtype

    @staticmethod
    def srgb_to_lrgb(srgb):
        a = 0.055
        return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

    @staticmethod
    def lrgb_to_srgb(lrgb):
        lrgb = lrgb.clamp(min=0)
        a = 0.055
        return torch.where(lrgb <= 0.0031308, lrgb * 12.92, (1 + a) * (lrgb ** (1 / 2.4)) - a)

    def lrgb_to_oklab(self, lrgb):
        """
        Args:
            lrgb (torch.Tensor): Linear RGB, shape (N, 3, *)
        """
        lms = torch.einsum('ij,bj...->bi...', self.lrgb_to_lms, lrgb).clamp(min=0)
        oklab = torch.einsum('ij,bj...->bi...', self.lms_to_oklab, lms.pow(1/3))
        return oklab

    def oklab_to_lrgb(self, oklab):
        """
        Args:
            oklab (torch.Tensor): Oklab, shape (N, 3, *)
        """
        lms = torch.einsum('ij,bj...->bi...', self.oklab_to_lms, oklab).pow(3)
        lrgb = torch.einsum('ij,bj...->bi...', self.lms_to_lrgb, lms)
        return lrgb.clamp(0, 1)

    def encode(self, img):
        rgb = img / 2 + 0.5
        lrgb = self.srgb_to_lrgb(rgb)
        oklab = self.lrgb_to_oklab(lrgb)
        if self.use_affine_norm:
            n_dim = img.dim() - 2
            mean = self.affine_mean.reshape(-1, *([1] * n_dim))
            std = self.affine_std.reshape(-1, *([1] * n_dim))
            oklab = (oklab - mean) / std
        return oklab

    def decode(self, oklab):
        if self.use_affine_norm:
            n_dim = oklab.dim() - 2
            mean = self.affine_mean.reshape(-1, *([1] * n_dim))
            std = self.affine_std.reshape(-1, *([1] * n_dim))
            oklab = oklab * std + mean
        lrgb = self.oklab_to_lrgb(oklab)
        rgb = self.lrgb_to_srgb(lrgb)
        img = rgb * 2 - 1
        return img


@MODULES.register_module()
class RGBColorEncoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        use_affine_norm=False,
        mean=(0.0, 0.0, 0.0),
        std=1.0,
    ):
        super().__init__()
        self.use_affine_norm = use_affine_norm
        if self.use_affine_norm:
            self.register_buffer('affine_mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer('affine_std', torch.tensor(std, dtype=torch.float32))

    @property
    def dtype(self):
        if self.use_affine_norm:
            return self.affine_mean.dtype
        else:
            return torch.float32

    def encode(self, img):
        if self.use_affine_norm:
            n_dim = img.dim() - 2
            mean = self.affine_mean.reshape(-1, *([1] * n_dim))
            std = self.affine_std.reshape(-1, *([1] * n_dim))
            img = (img - mean) / std
        return img

    def decode(self, img):
        if self.use_affine_norm:
            n_dim = img.dim() - 2
            mean = self.affine_mean.reshape(-1, *([1] * n_dim))
            std = self.affine_std.reshape(-1, *([1] * n_dim))
            img = img * std + mean
        return img.clamp(min=-1, max=1)
