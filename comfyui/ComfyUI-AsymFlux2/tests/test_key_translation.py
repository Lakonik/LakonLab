"""Validate the key-translation map against the known 121-key adapter dump.

The reference key list below was produced by inspecting the real file
`Lakonik/AsymFLUX.2-klein-9B/diffusion_pytorch_model.safetensors`.

This test imports only `key_map` (no torch, no diffusers, no ComfyUI) and runs
in well under a second. If a future adapter release adds or removes keys, this
will flag it immediately.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import key_map


REFERENCE_KEYS = """
norm_out.linear.weight
proj_buffer
proj_out.weight
scale_buffer
x_embedder.weight
time_guidance_embed.timestep_embedder.linear_1.lora_A.weight
time_guidance_embed.timestep_embedder.linear_1.lora_B.weight
time_guidance_embed.timestep_embedder.linear_2.lora_A.weight
time_guidance_embed.timestep_embedder.linear_2.lora_B.weight
""".strip().splitlines()

# Programmatically add the 116 block-level LoRA keys.
for i in range(24):
    REFERENCE_KEYS.append(f'single_transformer_blocks.{i}.attn.to_out.lora_A.weight')
    REFERENCE_KEYS.append(f'single_transformer_blocks.{i}.attn.to_out.lora_B.weight')
for i in range(8):
    for sub in ('ff.linear_in', 'ff.linear_out',
                'ff_context.linear_in', 'ff_context.linear_out'):
        REFERENCE_KEYS.append(f'transformer_blocks.{i}.{sub}.lora_A.weight')
        REFERENCE_KEYS.append(f'transformer_blocks.{i}.{sub}.lora_B.weight')


def test_reference_set_is_121_keys():
    assert len(REFERENCE_KEYS) == 121, f'expected 121, got {len(REFERENCE_KEYS)}'
    assert len(set(REFERENCE_KEYS)) == 121, 'duplicates in reference set'


def test_expected_keys_matches_reference():
    expected = key_map.expected_adapter_keys()
    ref = set(REFERENCE_KEYS)
    missing = ref - expected
    extra = expected - ref
    assert not missing, f'translation map is missing real keys: {sorted(missing)}'
    assert not extra,   f'translation map has phantom keys: {sorted(extra)}'


def test_translate_classifies_every_key():
    # Use placeholder tensor stand-ins (just strings); translate() doesn't touch them.
    fake_state = {k: f'<tensor:{k}>' for k in REFERENCE_KEYS}
    out = key_map.translate(fake_state)
    assert out.unknown_keys == [], f'unknown: {out.unknown_keys}'
    assert len(out.weight_overrides) == 1, out.weight_overrides
    assert len(out.module_replacements) == 2, out.module_replacements
    assert len(out.buffers) == 2, out.buffers
    assert len(out.lora_pairs) == 58, len(out.lora_pairs)
    # Every LoRA target must have BOTH A and B halves.
    for target, halves in out.lora_pairs.items():
        assert set(halves.keys()) == {'A', 'B'}, (target, halves)


def test_lora_targets_are_unique():
    assert len(set(key_map.LORA_MAP.values())) == len(key_map.LORA_MAP), \
        'two adapter prefixes map to the same BFL module'


def test_module_replacement_shapes():
    # img_in: hidden x (in_channels * patch^2) = 4096 x 768
    # final_layer.linear: (out_channels * patch^2) x hidden = 768 x 4096
    img_in_shape = key_map.FULL_OVERRIDES['x_embedder.weight'][2]
    fl_shape = key_map.FULL_OVERRIDES['proj_out.weight'][2]
    assert img_in_shape == (4096, 768), img_in_shape
    assert fl_shape == (768, 4096), fl_shape


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
    if failures:
        sys.exit(1)
    print(f'\nAll {len(funcs)} tests passed.')
