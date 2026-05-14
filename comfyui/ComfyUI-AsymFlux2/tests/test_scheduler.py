"""Verify the sqrt-shift sigma schedule matches LakonLab's FlowAdapterScheduler
math exactly.

We replicate the reference math from
`lakonlab/models/diffusions/schedulers/flow_adapter.py` and compare to our
port. No torch needed for the reference; we just check our function's
torch-Tensor output against the numpy reference.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

# Avoid importing scheduler at module level so we can skip if torch is missing.
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _reference_sigmas(num_steps, image_pixel_count,
                      base_shift=17.0, max_shift=34.0,
                      base_seq_len=1024 ** 2, max_seq_len=2048 ** 2,
                      eps=1e-6):
    """LakonLab's exact arithmetic, copied from flow_adapter.py."""
    sqrt_base = np.sqrt(base_seq_len)
    sqrt_max = np.sqrt(max_seq_len)
    m = (max_shift - base_shift) / (sqrt_max - sqrt_base)
    shift = (np.sqrt(image_pixel_count) - sqrt_base) * m + base_shift

    sigmas = np.linspace(1.0, 0.0, num_steps, endpoint=False, dtype=np.float32)
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sigmas = np.clip(sigmas, a_min=None, a_max=1.0 - eps)
    sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)], axis=0)
    return sigmas


def test_shift_at_base_returns_base():
    # If image_pixel_count == base_seq_len, shift should equal base_shift exactly.
    if not HAVE_TORCH:
        return
    from scheduler import compute_shift
    s = compute_shift(image_pixel_count=1024 ** 2)
    assert abs(s - 17.0) < 1e-9, s


def test_shift_at_max_returns_max():
    if not HAVE_TORCH:
        return
    from scheduler import compute_shift
    s = compute_shift(image_pixel_count=2048 ** 2)
    assert abs(s - 34.0) < 1e-9, s


def test_sigmas_match_reference():
    if not HAVE_TORCH:
        print('[skip] torch not available')
        return
    from scheduler import compute_sigmas
    for steps, pixels in [(38, 960 * 1280), (28, 1024 * 1024), (50, 2048 * 2048)]:
        ours = compute_sigmas(steps, pixels).cpu().numpy()
        ref = _reference_sigmas(steps, pixels)
        assert ours.shape == ref.shape, (ours.shape, ref.shape)
        assert np.allclose(ours, ref, atol=1e-6, rtol=1e-6), \
            f'max diff = {np.abs(ours - ref).max()} for steps={steps}, pixels={pixels}'


def test_sigmas_are_monotone_decreasing():
    if not HAVE_TORCH:
        return
    from scheduler import compute_sigmas
    s = compute_sigmas(38, 960 * 1280).cpu().numpy()
    diffs = np.diff(s)
    assert (diffs <= 0).all(), f'non-monotone at indices {np.where(diffs > 0)[0]}'
    assert s[0] < 1.0, s[0]      # clamped below 1
    assert s[-1] == 0.0, s[-1]


def test_default_38_step_schedule_matches_demo_example():
    # Pure-python reproduction of what the README demo runs.
    if not HAVE_TORCH:
        return
    from scheduler import compute_sigmas
    sigmas = compute_sigmas(38, 960 * 1280)
    assert sigmas.shape[0] == 39, sigmas.shape  # 38 steps + final 0
    # First sigma is shifted but clamped to <=1-1e-6 (compared in float32).
    expected_clamp = float(np.float32(1.0 - 1e-6))
    assert abs(sigmas[0].item() - expected_clamp) < 1e-9, sigmas[0].item()


if __name__ == '__main__':
    funcs = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failures = 0
    for fn in funcs:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except AssertionError as e:
            failures += 1
            print(f'FAIL  {fn.__name__}: {e}')
        except Exception as e:
            failures += 1
            print(f'ERROR {fn.__name__}: {type(e).__name__}: {e}')
    if failures:
        sys.exit(1)
    print(f'\nAll {len(funcs)} tests passed.')
