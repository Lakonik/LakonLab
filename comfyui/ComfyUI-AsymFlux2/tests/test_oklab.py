"""Oklab encode/decode sanity tests.

We don't have a reference image to byte-compare against, so we instead check:
  (a) encode -> decode round-trips back to the original image within sRGB
      precision (gamma curve introduces ~1/255 noise);
  (b) encode of grey-ish images puts most energy near affine_mean / std=1;
  (c) shape contract matches ComfyUI's (B, H, W, 3) image convention.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    print('[skip] torch not installed')
    sys.exit(0)

from oklab import (
    encode_image_to_oklab_latent,
    decode_oklab_latent_to_image,
    srgb_to_lrgb, lrgb_to_srgb,
    lrgb_to_oklab, oklab_to_lrgb,
)


def test_srgb_lrgb_roundtrip():
    x = torch.linspace(0, 1, 256)
    back = lrgb_to_srgb(srgb_to_lrgb(x))
    assert (back - x).abs().max() < 1e-5


def test_lrgb_oklab_roundtrip():
    x = torch.rand(1, 3, 8, 8)
    back = oklab_to_lrgb(lrgb_to_oklab(x))
    assert (back - x).abs().max() < 1e-3, (back - x).abs().max().item()


def test_image_latent_image_roundtrip():
    # Simulate a ComfyUI IMAGE tensor.
    torch.manual_seed(0)
    img = torch.rand(2, 16, 16, 3)
    latent = encode_image_to_oklab_latent(img)
    assert latent.shape == (2, 3, 16, 16), latent.shape
    out = decode_oklab_latent_to_image(latent)
    assert out.shape == (2, 16, 16, 3), out.shape
    err = (out - img).abs().max().item()
    # Some precision loss from gamma + matrix inversion, but should be < 1/255.
    assert err < 0.01, f'max round-trip error = {err}'


def test_mid_grey_normalizes_near_zero_a_b():
    # A mid-grey image should have oklab a, b near 0 (chromatic axes).
    img = torch.full((1, 4, 4, 3), 0.5)
    latent = encode_image_to_oklab_latent(img)
    a, b = latent[0, 1], latent[0, 2]
    # After (oklab - mean(0.56,0,0.01)) / std(0.16), a-channel should be near 0,
    # b-channel near -0.01/0.16 ≈ -0.0625.
    assert a.abs().max() < 0.05, a.abs().max().item()
    assert (b + 0.0625).abs().max() < 0.05, b.abs().max().item()


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
            import traceback
            print(f'ERROR {fn.__name__}:')
            traceback.print_exc()
    if failures:
        sys.exit(1)
    print(f'\nAll {len(funcs)} tests passed.')
