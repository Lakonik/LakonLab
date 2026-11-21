# Adapted from https://github.com/autonomousvision/stylegan-t/blob/main/training/diffaug.py

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES


@MODULES.register_module()
class DiffAug(nn.Module):
    """Differentiable discriminator augmentation from StyleGAN-T.

    Applies random translation, color jitter (brightness/saturation/contrast)
    and cutout (zero-masking a random rectangle, not cropping) to (B, C, H, W)
    images or (B, C, T, H, W) videos (per-sample random parameters shared
    across frames). Each transform is enabled for the whole batch with its own
    probability, with per-sample random parameters. The defaults reproduce the
    StyleGAN-T/RAE recipe (translation + color jitter always on, no cutout).
    All transforms are differentiable w.r.t. the input, so generator gradients
    flow through the augmentation.

    Args:
        translation_prob (float): Per-batch probability of random translation.
        translation_ratio (float): Maximum translation as a fraction of the
            image size.
        color_prob (float): Per-batch probability of color jitter.
        cutout_prob (float): Per-batch probability of cutout.
        cutout_ratio (float): Cutout size as a fraction of the image size.
    """

    def __init__(self,
                 translation_prob=1.0,
                 translation_ratio=0.125,
                 color_prob=1.0,
                 cutout_prob=0.0,
                 cutout_ratio=0.2):
        super().__init__()
        self.translation_prob = translation_prob
        self.translation_ratio = translation_ratio
        self.color_prob = color_prob
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio

    def forward(self, x):
        x = x.float()

        apply_translation = random.random() < self.translation_prob
        apply_color = random.random() < self.color_prob
        apply_cutout = random.random() < self.cutout_prob
        if not (apply_translation or apply_color or apply_cutout):
            return x

        ori_shape = x.shape
        b, h, w = x.size(0), x.size(-2), x.size(-1)
        device = x.device

        if apply_translation:
            rand_translation = torch.rand(2, b, 1, 1, device=device)  # broadcasts against the grids
            delta_h = round(h * self.translation_ratio)
            delta_w = round(w * self.translation_ratio)
            translation_h = rand_translation[0].mul(2 * delta_h + 1).floor().long() - delta_h
            translation_w = rand_translation[1].mul(2 * delta_w + 1).floor().long() - delta_w
            grid_b, grid_h, grid_w = torch.meshgrid(
                torch.arange(b, dtype=torch.long, device=device),
                torch.arange(h, dtype=torch.long, device=device),
                torch.arange(w, dtype=torch.long, device=device),
                indexing='ij')
            grid_h = (grid_h + translation_h + 1).clamp(0, h + 1)
            grid_w = (grid_w + translation_w + 1).clamp(0, w + 1)
            x_pad = F.pad(x.reshape(b, -1, h, w), [1, 1, 1, 1])
            x = x_pad.permute(0, 2, 3, 1)[grid_b, grid_h, grid_w].permute(
                0, 3, 1, 2).reshape(ori_shape).contiguous()

        if apply_color:
            rand_color = torch.rand((3, b) + (1, ) * (x.dim() - 1), device=device)  # broadcasts against x
            x = x + (rand_color[0] - 0.5)  # brightness
            x_mean = x.mean(dim=1, keepdim=True)
            x = (x - x_mean) * (rand_color[1] * 2) + x_mean  # saturation
            x_mean = x.mean(dim=tuple(range(1, x.dim())), keepdim=True)
            x = (x - x_mean) * (rand_color[2] + 0.5) + x_mean  # contrast

        if apply_cutout:
            rand_cutout = torch.rand(2, b, 1, 1, device=device)  # broadcasts against the grids
            cutout_h = round(h * self.cutout_ratio)
            cutout_w = round(w * self.cutout_ratio)
            offset_h = rand_cutout[0].mul(h + (1 - cutout_h % 2)).floor().long()
            offset_w = rand_cutout[1].mul(w + (1 - cutout_w % 2)).floor().long()
            grid_b, grid_h, grid_w = torch.meshgrid(
                torch.arange(b, dtype=torch.long, device=device),
                torch.arange(cutout_h, dtype=torch.long, device=device),
                torch.arange(cutout_w, dtype=torch.long, device=device),
                indexing='ij')
            grid_h = (grid_h + offset_h - cutout_h // 2).clamp(min=0, max=h - 1)
            grid_w = (grid_w + offset_w - cutout_w // 2).clamp(min=0, max=w - 1)
            mask = x.new_ones(b, h, w)
            mask[grid_b, grid_h, grid_w] = 0
            x = x * mask.view((b, ) + (1, ) * (x.dim() - 3) + (h, w))

        return x
