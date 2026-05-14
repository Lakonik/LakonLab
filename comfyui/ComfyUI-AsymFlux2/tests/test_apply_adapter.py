"""Unit-tests for the adapter-loader surgery primitives.

We don't build a real-sized FLUX stub here (FLUX.2's ~5 B params don't fit in
the 15 GB sandbox). Instead we exercise each surgery primitive
(_fuse_lora_into_weight, module replacement, buffer registration, weight
overwrite) against a tiny synthetic module that has the same hierarchy as
a real FLUX block.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print('[skip] torch not installed')
    sys.exit(0)

from adapter_loader import (
    apply_adapter,
    _fuse_lora_into_weight,
    _get_submodule,
    _set_submodule,
)


# ---------------------------------------------------------------------------
# 1. LoRA fusion math
# ---------------------------------------------------------------------------

def test_fuse_lora_matches_manual_matmul_in_fp32():
    """Verify the fusion math directly in fp32 storage so the test isn't gated
    on bf16's ~0.4% storage precision."""
    torch.manual_seed(0)
    linear = nn.Linear(64, 128, bias=False)  # fp32
    base = linear.weight.detach().clone()
    A = torch.randn(8, 64)
    B = torch.randn(128, 8)

    _fuse_lora_into_weight(linear.weight, A, B, scaling=1.0)

    expected = base + (B @ A)
    got = linear.weight.detach()
    err = (expected - got).abs().max().item()
    assert err < 1e-5, f'max error after fuse = {err}'


def test_fuse_lora_scaling():
    torch.manual_seed(1)
    linear = nn.Linear(32, 64, bias=False).to(torch.bfloat16)
    base = linear.weight.detach().clone().to(torch.float32)
    A = torch.randn(4, 32, dtype=torch.float16)
    B = torch.randn(64, 4, dtype=torch.float16)

    _fuse_lora_into_weight(linear.weight, A, B, scaling=0.5)

    expected = base + 0.5 * (B.to(torch.float32) @ A.to(torch.float32))
    got = linear.weight.detach().to(torch.float32)
    assert (expected - got).abs().max().item() < 0.05


def test_fuse_lora_shape_validation():
    linear = nn.Linear(64, 128, bias=False).to(torch.bfloat16)
    A_wrong = torch.randn(8, 32)  # in_features mismatch
    B = torch.randn(128, 8)
    try:
        _fuse_lora_into_weight(linear.weight, A_wrong, B, scaling=1.0)
    except AssertionError:
        return
    raise AssertionError('expected AssertionError on incompatible lora_A shape')


# ---------------------------------------------------------------------------
# 2. Submodule walk / replacement
# ---------------------------------------------------------------------------

def test_get_and_set_submodule_nested():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Inner(), Inner()])
            self.head = nn.Sequential(nn.SiLU(), nn.Linear(4, 4))

    m = Outer()

    # Get
    assert _get_submodule(m, 'blocks.1.lin') is m.blocks[1].lin
    assert _get_submodule(m, 'head.1') is m.head[1]

    # Set
    new_lin = nn.Linear(8, 8)
    _set_submodule(m, 'blocks.0.lin', new_lin)
    assert m.blocks[0].lin is new_lin


# ---------------------------------------------------------------------------
# 3. apply_adapter end-to-end on a *one-block* miniature
# ---------------------------------------------------------------------------

class _TinyFlux(nn.Module):
    """One double block, one single block, time_in, img_in, final_layer.
    All dims scaled down. This stays well under 100 MB."""

    def __init__(self):
        super().__init__()
        H = 64
        MH = 96  # mlp_hidden

        # img_in: starts at "latent-FLUX dims" (32 in) and will be replaced with (768, H).
        self.img_in = nn.Linear(32, H, bias=False)

        # time_in
        class TI(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_layer = nn.Linear(256, H, bias=False)
                self.out_layer = nn.Linear(H, H, bias=False)
        self.time_in = TI()

        # one double block (i=0)
        class DB(nn.Module):
            def __init__(self):
                super().__init__()
                self.img_mlp = nn.Sequential(
                    nn.Linear(H, MH * 2, bias=False), nn.SiLU(),
                    nn.Linear(MH, H, bias=False))
                self.txt_mlp = nn.Sequential(
                    nn.Linear(H, MH * 2, bias=False), nn.SiLU(),
                    nn.Linear(MH, H, bias=False))
        self.double_blocks = nn.ModuleList([DB()])

        # one single block (i=0)
        class SB(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear2 = nn.Linear(H + MH, H, bias=False)
        self.single_blocks = nn.ModuleList([SB()])

        # final_layer
        class FL(nn.Module):
            def __init__(self):
                super().__init__()
                # adaLN_modulation.1 ends at 2*H = 128 (matches FLUX layout)
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(H, 2 * H, bias=False))
                self.linear = nn.Linear(H, 32, bias=False)
        self.final_layer = FL()


def _mini_adapter_for_one_block():
    """Build a state_dict targeting just block 0 + time_in + I/O, with sizes
    matching _TinyFlux. NOTE: x_embedder/proj_out/norm_out/proj_buffer have
    shapes hardcoded in key_map -- so we MUST honor those exact shapes."""
    torch.manual_seed(2)
    H = 64
    MH = 96
    R = 4  # lora rank for the test

    sd = {}

    # 5 full-overrides at the canonical AsymFLUX shapes:
    #   x_embedder: (4096, 768)  -- but our tiny model uses H=64; so the
    #   replacement target shape needs to match what key_map declares. The
    #   replaced Linear ends up with weight (768, 4096), not interacting with
    #   the rest of our tiny block.
    sd['x_embedder.weight']      = torch.randn(4096, 768, dtype=torch.float16)
    sd['proj_out.weight']        = torch.randn(768, 4096, dtype=torch.float16)
    sd['norm_out.linear.weight'] = torch.randn(2 * 4096, 4096, dtype=torch.float16)
    sd['proj_buffer']            = torch.randn(768, 128, dtype=torch.bfloat16)
    sd['scale_buffer']           = torch.tensor(0.5, dtype=torch.bfloat16)
    return sd


def test_unknown_key_strict_raises():
    model = _TinyFlux().to(torch.bfloat16)
    sd = _mini_adapter_for_one_block()
    sd['something.bogus.weight'] = torch.zeros(1)
    try:
        apply_adapter(model, sd, compute_dtype=torch.bfloat16, strict=True)
    except KeyError:
        return
    raise AssertionError('expected KeyError for bogus key')


def test_module_replacement_is_size_768():
    """The full-override shapes are baked into key_map; verify the replaced
    submodules end up at the canonical FLUX.2-pixel dims."""
    # We can't apply the full adapter to _TinyFlux because the
    # norm_out.linear.weight (8192, 4096) won't match _TinyFlux's
    # adaLN_modulation[1] of shape (128, 64). So instead build a model whose
    # adaLN_modulation matches the real shape and skip the block LoRAs.
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_in = nn.Linear(32, 4096, bias=False)
            class FL(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.adaLN_modulation = nn.Sequential(
                        nn.SiLU(), nn.Linear(4096, 8192, bias=False))
                    self.linear = nn.Linear(4096, 32, bias=False)
            self.final_layer = FL()
    model = M().to(torch.bfloat16)

    sd = {
        'x_embedder.weight':      torch.randn(4096, 768, dtype=torch.float16),
        'proj_out.weight':        torch.randn(768, 4096, dtype=torch.float16),
        'norm_out.linear.weight': torch.randn(8192, 4096, dtype=torch.float16),
        'proj_buffer':            torch.randn(768, 128, dtype=torch.bfloat16),
        'scale_buffer':           torch.tensor(0.5, dtype=torch.bfloat16),
    }

    # Loose strict because we're not providing the 116 LoRA halves.
    manifest = apply_adapter(model, sd, compute_dtype=torch.bfloat16, strict=False)
    assert 'img_in' in manifest['replaced_modules']
    assert 'final_layer.linear' in manifest['replaced_modules']
    assert model.img_in.in_features == 768
    assert model.img_in.out_features == 4096
    assert model.final_layer.linear.in_features == 4096
    assert model.final_layer.linear.out_features == 768
    assert 'proj_buffer' in manifest['registered_buffers']
    assert 'scale_buffer' in manifest['registered_buffers']
    assert model.proj_buffer.shape == (768, 128)
    assert model.scale_buffer.shape == ()


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
