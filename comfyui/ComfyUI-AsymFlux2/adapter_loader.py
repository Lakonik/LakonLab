"""Apply a translated AsymFLUX.2 adapter onto a stock ComfyUI FLUX.2 transformer.

The transformer is the `Flux` nn.Module from `comfy/ldm/flux/model.py`. After
this function runs:

  - `img_in`               has been replaced by `nn.Linear(768, hidden)` (no bias).
  - `final_layer.linear`   has been replaced by `nn.Linear(hidden, 768)` (no bias).
  - `final_layer.adaLN_modulation.1.weight` has been overwritten.
  - Two buffers (`proj_buffer`, `scale_buffer`) are registered on the transformer.
  - 58 weight tensors across `double_blocks`/`single_blocks`/`time_in` have a
    rank-256 LoRA delta fused into them.

Numerics: the delta is computed in float32, then cast back to the existing
parameter's dtype to match how PEFT does `fuse_lora` and how LakonLab loads
the adapter.
"""

import torch
import torch.nn as nn

try:
    from .key_map import (
        translate, LORA_SCALING,
        ASYMFLUX_HIDDEN, ASYMFLUX_IN_CHANNELS,
        ASYMFLUX_OUT_CHANNELS, ASYMFLUX_PATCH_SIZE,
    )
except ImportError:  # standalone-import path (tests, ad-hoc usage)
    from key_map import (
        translate, LORA_SCALING,
        ASYMFLUX_HIDDEN, ASYMFLUX_IN_CHANNELS,
        ASYMFLUX_OUT_CHANNELS, ASYMFLUX_PATCH_SIZE,
    )


def _get_submodule(root: nn.Module, dotted_path: str) -> nn.Module:
    obj = root
    for part in dotted_path.split('.'):
        obj = getattr(obj, part)
    return obj


def _set_submodule(root: nn.Module, dotted_path: str, value: nn.Module) -> None:
    parts = dotted_path.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _get_param(root: nn.Module, dotted_path: str) -> torch.nn.Parameter:
    # dotted_path is like "final_layer.adaLN_modulation.1.weight" -> walk to
    # the `weight` Parameter on the leaf module.
    *module_parts, leaf = dotted_path.split('.')
    mod = root
    for part in module_parts:
        mod = getattr(mod, part)
    return getattr(mod, leaf)


def _fuse_lora_into_weight(
        weight: torch.nn.Parameter,
        lora_A: torch.Tensor,  # (rank, in_features)
        lora_B: torch.Tensor,  # (out_features, rank)
        scaling: float,
) -> None:
    """In-place: weight += (B @ A) * scaling, computed in float32."""
    out_features, in_features = weight.shape
    assert lora_A.shape == (lora_B.shape[1], in_features), \
        f'lora_A {tuple(lora_A.shape)} incompatible with weight {tuple(weight.shape)}'
    assert lora_B.shape == (out_features, lora_A.shape[0]), \
        f'lora_B {tuple(lora_B.shape)} incompatible with weight {tuple(weight.shape)}'

    target_dtype = weight.dtype
    target_device = weight.device

    delta = (lora_B.to(torch.float32, copy=False)
             @ lora_A.to(torch.float32, copy=False)) * scaling
    with torch.no_grad():
        weight.add_(delta.to(dtype=target_dtype, device=target_device))


def apply_adapter(
        transformer: nn.Module,
        adapter_state_dict: dict,
        compute_dtype: torch.dtype = torch.bfloat16,
        strict: bool = True,
) -> dict:
    """Mutate `transformer` in place. Returns a manifest describing what changed.

    Args:
        transformer: the inner `Flux` nn.Module (NOT the ModelPatcher wrapper).
        adapter_state_dict: {key: Tensor} from the safetensors file.
        compute_dtype: dtype used for the new replacement modules.
        strict: if True, raise on any unknown adapter key or missing translation
            target. If False, log and skip.

    Returns:
        manifest: a dict with keys 'replaced_modules', 'overwrote_weights',
        'registered_buffers', 'fused_loras', 'skipped'.
    """
    plan = translate(adapter_state_dict)

    if plan.unknown_keys:
        msg = f'Unknown adapter keys ({len(plan.unknown_keys)}): {plan.unknown_keys[:5]}...'
        if strict:
            raise KeyError(msg)
        print(f'[asymflux2] warning: {msg}')

    manifest = {
        'replaced_modules': [],
        'overwrote_weights': [],
        'registered_buffers': [],
        'fused_loras': [],
        'skipped': [],
    }

    # 1. Replace modules with shape-changing surgery.
    for bfl_path, (expected_shape, tensor) in plan.module_replacements.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f'{bfl_path}: adapter tensor shape {tuple(tensor.shape)} != '
                f'expected {expected_shape}')
        out_features, in_features = expected_shape
        try:
            existing = _get_submodule(transformer, bfl_path)
        except AttributeError:
            if strict:
                raise AttributeError(f'transformer has no submodule "{bfl_path}"')
            manifest['skipped'].append(bfl_path)
            continue

        device = next(existing.parameters()).device if any(True for _ in existing.parameters()) else 'cpu'
        new_linear = nn.Linear(in_features, out_features, bias=False,
                               device=device, dtype=compute_dtype)
        with torch.no_grad():
            new_linear.weight.copy_(tensor.to(dtype=compute_dtype, device=device))
        _set_submodule(transformer, bfl_path, new_linear)
        manifest['replaced_modules'].append(bfl_path)

    # 2. Overwrite individual weight tensors in place.
    for bfl_param_path, tensor in plan.weight_overrides.items():
        try:
            param = _get_param(transformer, bfl_param_path)
        except AttributeError:
            if strict:
                raise AttributeError(f'transformer has no parameter "{bfl_param_path}"')
            manifest['skipped'].append(bfl_param_path)
            continue
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(
                f'{bfl_param_path}: base shape {tuple(param.shape)} != '
                f'adapter shape {tuple(tensor.shape)}')
        with torch.no_grad():
            param.copy_(tensor.to(dtype=param.dtype, device=param.device))
        manifest['overwrote_weights'].append(bfl_param_path)

    # 3. Register new buffers.
    device = next(transformer.parameters()).device
    for name, tensor in plan.buffers.items():
        transformer.register_buffer(
            name, tensor.to(device=device).clone(), persistent=False)
        manifest['registered_buffers'].append(name)

    # 4. Fuse LoRA deltas.
    for bfl_module_path, halves in plan.lora_pairs.items():
        if set(halves.keys()) != {'A', 'B'}:
            if strict:
                raise KeyError(f'{bfl_module_path}: incomplete LoRA pair {list(halves)}')
            manifest['skipped'].append(bfl_module_path)
            continue
        try:
            module = _get_submodule(transformer, bfl_module_path)
        except AttributeError:
            if strict:
                raise AttributeError(f'transformer has no submodule "{bfl_module_path}"')
            manifest['skipped'].append(bfl_module_path)
            continue

        weight = module.weight
        try:
            _fuse_lora_into_weight(weight, halves['A'], halves['B'], LORA_SCALING)
        except AssertionError as exc:
            raise ValueError(f'{bfl_module_path}: {exc}') from None
        manifest['fused_loras'].append(bfl_module_path)

    return manifest


def asymflux2_model_config() -> dict:
    """The architecture parameters a ComfyUI FluxParams must be set to in order
    for img_in / final_layer.linear / patch grid to match the adapter."""
    return {
        'patch_size':   ASYMFLUX_PATCH_SIZE,
        'in_channels':  ASYMFLUX_IN_CHANNELS,
        'out_channels': ASYMFLUX_OUT_CHANNELS,
        'hidden_size':  ASYMFLUX_HIDDEN,
    }
