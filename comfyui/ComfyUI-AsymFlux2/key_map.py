"""Translation table between Lakonik AsymFLUX.2 adapter keys (diffusers naming)
and ComfyUI's FLUX.2 module names (BFL/original naming).

Source of truth: the 121-key dump produced from
`Lakonik/AsymFLUX.2-klein-9B/diffusion_pytorch_model.safetensors`.

The adapter contains:
  - 5 full-weight tensors that REPLACE base-model state (some require module
    surgery because their shapes differ from the base FLUX.2 latent model).
  - 116 LoRA tensors (rank 256, alpha 256 -> scaling 1.0) targeting 58 modules:
      * double_blocks {0..7}: img_mlp.0, img_mlp.2, txt_mlp.0, txt_mlp.2  (32 tgts)
      * single_blocks {0..23}: linear2                                    (24 tgts)
      * time_in: in_layer, out_layer                                       ( 2 tgts)
"""

from dataclasses import dataclass
from typing import Optional

NUM_DOUBLE_BLOCKS = 8
NUM_SINGLE_BLOCKS = 24

LORA_RANK = 256
LORA_ALPHA = 256
LORA_SCALING = LORA_ALPHA / LORA_RANK  # = 1.0

OKLAB_AFFINE_MEAN = (0.56, 0.0, 0.01)
OKLAB_AFFINE_STD = 0.16

ASYMFLUX_PATCH_SIZE = 16
ASYMFLUX_IN_CHANNELS = 3
ASYMFLUX_OUT_CHANNELS = 3
ASYMFLUX_HIDDEN = 4096
ASYMFLUX_BASE_RANK = 128  # rank of the asymmetric PCA subspace (proj_buffer width)


# Full-weight overrides. The dict value is a callable that returns the BFL-side
# action: ('weight', bfl_key) overwrites a parameter in-place;
# ('module_replace', bfl_module, expected_shape) replaces a Linear submodule;
# ('buffer', name) registers a new top-level buffer.
FULL_OVERRIDES = {
    # diffusers key                  : action tuple
    'x_embedder.weight':              ('module_replace', 'img_in',
                                       (ASYMFLUX_HIDDEN, ASYMFLUX_IN_CHANNELS * ASYMFLUX_PATCH_SIZE ** 2)),
    'proj_out.weight':                ('module_replace', 'final_layer.linear',
                                       (ASYMFLUX_OUT_CHANNELS * ASYMFLUX_PATCH_SIZE ** 2, ASYMFLUX_HIDDEN)),
    'norm_out.linear.weight':         ('weight', 'final_layer.adaLN_modulation.1.weight'),
    'proj_buffer':                    ('buffer', 'proj_buffer'),
    'scale_buffer':                   ('buffer', 'scale_buffer'),
}


# LoRA module remapping. Each entry maps an adapter prefix (without
# `.lora_A.weight` / `.lora_B.weight` suffix) to the equivalent ComfyUI module
# path whose `.weight` should receive the fused delta.
def _build_lora_map() -> dict:
    m = {}
    # Double-stream blocks: ff -> img_mlp, ff_context -> txt_mlp.
    # The MLP is nn.Sequential[Linear, SiLUActivation, Linear], so the two
    # Linears are submodules .0 and .2.
    for i in range(NUM_DOUBLE_BLOCKS):
        m[f'transformer_blocks.{i}.ff.linear_in']         = f'double_blocks.{i}.img_mlp.0'
        m[f'transformer_blocks.{i}.ff.linear_out']        = f'double_blocks.{i}.img_mlp.2'
        m[f'transformer_blocks.{i}.ff_context.linear_in'] = f'double_blocks.{i}.txt_mlp.0'
        m[f'transformer_blocks.{i}.ff_context.linear_out']= f'double_blocks.{i}.txt_mlp.2'

    # Single-stream blocks: diffusers `attn.to_out` corresponds to ComfyUI's
    # fused `linear2` (which carries both attention-out and MLP-out, hence
    # input dim hidden + mlp_hidden = 4096 + 12288 = 16384).
    for i in range(NUM_SINGLE_BLOCKS):
        m[f'single_transformer_blocks.{i}.attn.to_out']   = f'single_blocks.{i}.linear2'

    # Combined time/guidance embedder: linear_1 -> in_layer, linear_2 -> out_layer.
    m['time_guidance_embed.timestep_embedder.linear_1']   = 'time_in.in_layer'
    m['time_guidance_embed.timestep_embedder.linear_2']   = 'time_in.out_layer'

    return m


LORA_MAP = _build_lora_map()


@dataclass
class TranslatedAdapter:
    # bfl_param_path -> tensor (full-weight overwrite)
    weight_overrides: dict
    # bfl_module_path -> (expected_shape, tensor) (module replacement)
    module_replacements: dict
    # buffer_name -> tensor (register on transformer)
    buffers: dict
    # bfl_module_path -> {'A': tensor, 'B': tensor}  (fuse into module.weight)
    lora_pairs: dict
    # any keys we didn't recognize
    unknown_keys: list


def _split_lora_key(adapter_key: str) -> Optional[tuple]:
    """Return (prefix, 'A' | 'B') or None if the key isn't a LoRA tensor."""
    if adapter_key.endswith('.lora_A.weight'):
        return adapter_key[:-len('.lora_A.weight')], 'A'
    if adapter_key.endswith('.lora_B.weight'):
        return adapter_key[:-len('.lora_B.weight')], 'B'
    return None


def translate(adapter_state_dict: dict) -> TranslatedAdapter:
    """Classify every adapter key into an override, buffer, replacement or LoRA pair.

    `adapter_state_dict` is a {str: Tensor-like}. Tensors are not modified here;
    the loader applies them downstream.
    """
    weight_overrides = {}
    module_replacements = {}
    buffers = {}
    lora_pairs: dict = {}
    unknown_keys = []

    for key, tensor in adapter_state_dict.items():
        lora = _split_lora_key(key)
        if lora is not None:
            prefix, slot = lora
            bfl_target = LORA_MAP.get(prefix)
            if bfl_target is None:
                unknown_keys.append(key)
                continue
            lora_pairs.setdefault(bfl_target, {})[slot] = tensor
            continue

        action = FULL_OVERRIDES.get(key)
        if action is None:
            unknown_keys.append(key)
            continue

        kind = action[0]
        if kind == 'weight':
            weight_overrides[action[1]] = tensor
        elif kind == 'module_replace':
            module_replacements[action[1]] = (action[2], tensor)
        elif kind == 'buffer':
            buffers[action[1]] = tensor
        else:
            unknown_keys.append(key)

    return TranslatedAdapter(
        weight_overrides=weight_overrides,
        module_replacements=module_replacements,
        buffers=buffers,
        lora_pairs=lora_pairs,
        unknown_keys=unknown_keys,
    )


def expected_adapter_keys() -> set:
    """The full set of keys we expect to see in a valid AsymFLUX.2-klein-9B adapter.

    Useful for sanity-checking a new release: any key in the adapter not in
    this set is unknown; any key in this set not in the adapter is missing.
    """
    keys = set(FULL_OVERRIDES.keys())
    for prefix in LORA_MAP:
        keys.add(f'{prefix}.lora_A.weight')
        keys.add(f'{prefix}.lora_B.weight')
    return keys
