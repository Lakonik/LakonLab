"""Port of LakonLab's `FlowAdapterScheduler` sigma computation for AsymFLUX.2.

We only port the sigma-schedule generation. ComfyUI's UniPC sampler will
consume the resulting `sigmas` tensor and run the actual denoising loop.

Defaults below come from `demo/example_asymflux2_klein_pipeline.py`:

    shift=17.0, use_dynamic_shifting=True,
    base_seq_len=1024**2, max_seq_len=2048**2,
    base_logshift=ln(17.0), max_logshift=ln(34.0),
    dynamic_shifting_type='sqrt',
    base_scheduler='UniPCMultistep'

`seq_len` here is the value passed to LakonLab's `set_timesteps`, which the
pipeline computes as `latents.shape[2:].numel()` -- i.e. pixel count of the
target image (height * width), not patch count.
"""

import math
import numpy as np
import torch

DEFAULT_SHIFT_BASE = 17.0
DEFAULT_SHIFT_MAX = 34.0
DEFAULT_BASE_SEQ_LEN = 1024 ** 2
DEFAULT_MAX_SEQ_LEN = 2048 ** 2
DEFAULT_EPS = 1e-6


def compute_shift(
        image_pixel_count: int,
        base_shift: float = DEFAULT_SHIFT_BASE,
        max_shift: float = DEFAULT_SHIFT_MAX,
        base_seq_len: int = DEFAULT_BASE_SEQ_LEN,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> float:
    """Sqrt-interpolated shift between (base_shift, max_shift) over sqrt(seq_len)."""
    sqrt_base = math.sqrt(base_seq_len)
    sqrt_max = math.sqrt(max_seq_len)
    m = (max_shift - base_shift) / (sqrt_max - sqrt_base)
    return (math.sqrt(image_pixel_count) - sqrt_base) * m + base_shift


def compute_sigmas(
        num_inference_steps: int,
        image_pixel_count: int,
        base_shift: float = DEFAULT_SHIFT_BASE,
        max_shift: float = DEFAULT_SHIFT_MAX,
        base_seq_len: int = DEFAULT_BASE_SEQ_LEN,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Returns a `torch.float32` 1-D tensor of length `num_inference_steps + 1`,
    ending in 0.0, ready to hand to ComfyUI's UniPC sampler.
    """
    shift = compute_shift(image_pixel_count, base_shift, max_shift,
                          base_seq_len, max_seq_len)

    sigmas = np.linspace(1.0, 0.0, num_inference_steps, endpoint=False, dtype=np.float32)
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

    # UniPCMultistep path clamps sigmas to <= 1 - eps.
    sigmas = np.clip(sigmas, a_min=None, a_max=1.0 - eps)
    sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)], axis=0)
    return torch.from_numpy(sigmas)
