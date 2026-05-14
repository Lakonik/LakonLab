"""Oklab colorspace transform, faithful port of LakonLab's `OklabColorEncoder`
with the affine-normalization parameters used by AsymFLUX.2 klein:

    mean = (0.56, 0.0, 0.01),  std = 0.16

ComfyUI tensor conventions:
    IMAGE  -> (B, H, W, 3) in sRGB [0, 1]
    LATENT -> (B, C, H, W) in floats, no fixed range

AsymFLUX expects 3-channel pixel-space "latents" in Oklab after the affine
normalization. There is no learned decoder; encode/decode are deterministic.
"""

import torch

try:
    from .key_map import OKLAB_AFFINE_MEAN, OKLAB_AFFINE_STD
except ImportError:
    from key_map import OKLAB_AFFINE_MEAN, OKLAB_AFFINE_STD


# Standard Oklab matrices (Bjorn Ottosson). All in float32 for accuracy.
_LRGB_TO_LMS = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=torch.float32)

_LMS_TO_OKLAB = torch.tensor([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=torch.float32)


def _matmul_chw(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 matrix M to a (B, 3, H, W) tensor along the channel dim."""
    # einsum keeps device/dtype of x; cast M to match for speed but keep f32 for fidelity.
    return torch.einsum('ij,bj...->bi...', M.to(x.device, x.dtype), x)


def srgb_to_lrgb(srgb: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)


def lrgb_to_srgb(lrgb: torch.Tensor) -> torch.Tensor:
    lrgb = lrgb.clamp(min=0)
    a = 0.055
    return torch.where(lrgb <= 0.0031308, lrgb * 12.92, (1 + a) * (lrgb ** (1 / 2.4)) - a)


def lrgb_to_oklab(lrgb: torch.Tensor) -> torch.Tensor:
    lms = _matmul_chw(_LRGB_TO_LMS, lrgb).clamp(min=0)
    return _matmul_chw(_LMS_TO_OKLAB, lms.pow(1 / 3))


def oklab_to_lrgb(oklab: torch.Tensor) -> torch.Tensor:
    oklab_to_lms = torch.linalg.inv(_LMS_TO_OKLAB).to(oklab.device, oklab.dtype)
    lms_to_lrgb = torch.linalg.inv(_LRGB_TO_LMS).to(oklab.device, oklab.dtype)
    lms = torch.einsum('ij,bj...->bi...', oklab_to_lms, oklab).pow(3)
    return torch.einsum('ij,bj...->bi...', lms_to_lrgb, lms).clamp(0, 1)


def _affine_vec(device, dtype) -> tuple:
    mean = torch.tensor(OKLAB_AFFINE_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([OKLAB_AFFINE_STD] * 3, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def encode_image_to_oklab_latent(image_bhwc_srgb: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE (B, H, W, 3) sRGB [0,1] -> Oklab "latent" (B, 3, H, W).

    Uses the same arithmetic as LakonLab's OklabColorEncoder.encode().
    """
    # to (B, 3, H, W) f32
    x = image_bhwc_srgb.permute(0, 3, 1, 2).contiguous().to(torch.float32)
    lrgb = srgb_to_lrgb(x)
    oklab = lrgb_to_oklab(lrgb)
    mean, std = _affine_vec(oklab.device, oklab.dtype)
    return (oklab - mean) / std


def decode_oklab_latent_to_image(oklab_latent_bchw: torch.Tensor) -> torch.Tensor:
    """Oklab "latent" (B, 3, H, W) -> ComfyUI IMAGE (B, H, W, 3) sRGB [0,1]."""
    x = oklab_latent_bchw.to(torch.float32)
    mean, std = _affine_vec(x.device, x.dtype)
    oklab = x * std + mean
    lrgb = oklab_to_lrgb(oklab)
    srgb = lrgb_to_srgb(lrgb)
    return srgb.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
