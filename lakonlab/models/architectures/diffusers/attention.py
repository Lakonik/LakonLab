# Copyright (c) 2026 Hansheng Chen

import torch
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    _check_attention_backend_requirements,
    _maybe_download_kernel_for_backend,
)
from lakonlab.ops import attention as attention_ops
from lakonlab.utils import get_root_logger


# Backends whose kernels cannot produce gradients; checked by set_attention_backends.
_NO_BACKWARD_BACKENDS = {b for b in AttentionBackendName if 'sage' in b.value}


if attention_ops.flash_attn_interface is not None:
    @_AttentionBackendRegistry.register(
        AttentionBackendName._FLASH_3,
        constraints=_AttentionBackendRegistry._constraints[AttentionBackendName._FLASH_3],
    )
    def _flash_attention_3(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            scale: float | None = None,
            is_causal: bool = False,
            return_lse: bool = False,
            _parallel_config=None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError('`attn_mask` is not supported for flash-attn 3.')
        out, lse = attention_ops._flash_attn_3_forward_op(query, key, value, scale, is_causal)
        # native (B, H, S) -> public (B, S, H), as a view and only when requested
        return (out, lse.permute(0, 2, 1)) if return_lse else out


if attention_ops.sageattn is not None:
    _diffusers_sage_attention = _AttentionBackendRegistry._backends[AttentionBackendName.SAGE]


    @_AttentionBackendRegistry.register(
        AttentionBackendName.SAGE,
        constraints=_AttentionBackendRegistry._constraints[AttentionBackendName.SAGE],
    )
    def _sage_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            is_causal: bool = False,
            scale: float | None = None,
            return_lse: bool = False,
            _parallel_config=None,
    ) -> torch.Tensor:
        if _parallel_config is not None:
            return _diffusers_sage_attention(
                query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale, return_lse=return_lse,
                _parallel_config=_parallel_config)
        if attn_mask is not None:
            raise ValueError('`attn_mask` is not supported for sage attention')
        out, lse = attention_ops._sage_attn_op(query, key, value, scale, is_causal, return_lse)
        return (out, lse) if return_lse else out


if attention_ops.flash_attn_interface is not None:
    @_AttentionBackendRegistry.register(
        AttentionBackendName._FLASH_VARLEN_3,
        constraints=_AttentionBackendRegistry._constraints[AttentionBackendName._FLASH_VARLEN_3],
    )
    def _flash_varlen_attention_3(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            scale: float | None = None,
            is_causal: bool = False,
            return_lse: bool = False,
            *,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            _parallel_config=None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError('`attn_mask` is not supported for flash-attn 3 varlen.')
        out, lse = attention_ops._flash_attn_varlen_3_forward_op(
            query.flatten(0, 1), key.flatten(0, 1), value.flatten(0, 1),
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, is_causal)
        out = out.unflatten(0, query.shape[:2])
        return (out, lse) if return_lse else out


def _no_backward_patch(module, backend):
    """Raise on grad-enabled use of an attention module whose backend has no backward.

    Without this, a residual connection around the attention keeps its grad_fn and
    backward silently drops the attention-path gradients instead of erroring.
    """

    def make_new_forward(old_forward, backend):
        def new_forward(*args, **kwargs):
            if torch.is_grad_enabled():
                input_requires_grad = any(
                    isinstance(t, torch.Tensor) and t.requires_grad
                    for t in list(args) + list(kwargs.values()))
                parameter_requires_grad = any(p.requires_grad for p in module.parameters())
                if input_requires_grad or parameter_requires_grad:
                    raise RuntimeError(
                        f'Attention backend {backend} does not support backward; this module must '
                        f'only run without gradients (e.g. under torch.no_grad())')
            return old_forward(*args, **kwargs)

        return new_forward

    module.forward = make_new_forward(module.forward, backend)


def default_attention_backend(kind='dense', device=None):
    """Select the default attention backend from the installed kernels and the GPU of ``device``.

    FlashAttention 3 is used on SM90 only, its documented target, and SageAttention on every
    architecture except SM100/SM103, for which it has no kernel.

    Args:
        kind (str): 'dense' for long sequences (SageAttention, then FlashAttention 3, then SDPA),
            'short' for short query sequences, which SageAttention handles inefficiently
            (FlashAttention 3, then SDPA), or 'varlen' for modules that accept variable-length
            sequences (variable-length FlashAttention 3, then SDPA). Defaults to 'dense'.
        device (torch.device | int | None): CUDA device. Defaults to the current device.

    Returns:
        str: A diffusers AttentionBackendName value.
    """
    major = torch.cuda.get_device_capability(device)[0]
    for backend in dict(dense=('sage', '_flash_3'), short=('_flash_3',), varlen=('_flash_varlen_3',))[kind]:
        if (major != 10 if backend == 'sage' else major == 9):
            try:
                check_attention_backend(backend)
                return backend
            except RuntimeError:
                pass
    return 'native'


def check_attention_backend(backend):
    """Raise a RuntimeError if a diffusers attention backend is not usable.

    Besides diffusers' package and version requirements, the SageAttention and FlashAttention 3
    kernels that this module wraps must have imported. Kernels of hub backends are downloaded.

    Args:
        backend (str): A diffusers AttentionBackendName value.
    """
    resolved = AttentionBackendName(backend.lower())
    _check_attention_backend_requirements(resolved)
    kernel = dict(
        sage=attention_ops.sageattn, _flash_3=attention_ops.flash_attn_interface,
        _flash_varlen_3=attention_ops.flash_attn_interface).get(resolved.value, True)
    if kernel is None:
        raise RuntimeError(f'Attention backend {resolved.value!r} is installed but failed to import.')
    _maybe_download_kernel_for_backend(resolved)


def set_attention_backends(module, backend=None, overrides=None):
    """Set diffusers attention backends per attention module.

    Whether a block may use a backend without backward support (e.g. SageAttention)
    depends on the training topology, not on the module itself, so the assignment is
    fully config-driven. Note that fully-frozen submodules are shared between the
    teacher, student and EMA models via weight tying, and the teacher's modules are
    the ones that survive tying — set the same option on both model configs.

    Args:
        module (nn.Module): Root module whose attention submodules are configured.
        backend (str | None): Backend applied to every attention module (see diffusers
            AttentionBackendName for valid names, e.g. '_flash_3', '_native_cudnn',
            'sage'). '_flash_3' supports backward through the
            autograd formula registered in this file, as does '_flash_varlen_3' with
            prepared offsets passed through attention_kwargs. SageAttention backends have no
            backward; their modules are patched to raise if run with gradients enabled
            on a grad-requiring input or with trainable parameters. None configures only
            the modules matched by ``overrides``. Defaults to None.
        overrides (dict[str, str] | None): Maps module names to backends, overriding
            ``backend``, with the same matching semantics as flex_freeze
            ('self.'-prefixed keys match full module names, other keys match
            substrings; the first matching key wins). Defaults to None.
    """
    if overrides is None:
        overrides = {}
    logger = get_root_logger()

    # validate availability once per distinct backend
    for name in set(([] if backend is None else [backend]) + list(overrides.values())):
        check_attention_backend(name)

    matched_keys = set()
    for module_name, m in module.named_modules():
        if not isinstance(m, AttentionModuleMixin):
            continue
        target = backend
        for key, key_backend in overrides.items():
            if key.startswith('self.'):  # use full name matching (an attention module or a parent)
                if key[5:] == module_name or module_name.startswith(key[5:] + '.'):
                    target = key_backend
                    matched_keys.add(key)
                    break
            elif key in module_name:  # use partial name matching
                target = key_backend
                matched_keys.add(key)
                break
        if target is None:
            continue
        m.set_attention_backend(target)
        if AttentionBackendName(target.lower()) in _NO_BACKWARD_BACKENDS:
            _no_backward_patch(m, target)

    unmatched_keys = set(overrides) - matched_keys
    if len(unmatched_keys) > 0:
        logger.warning(f'Attention backend override keys not found: {unmatched_keys}')
    logger.info(f'Set attention backends (default={backend}, {len(overrides)} override keys)')
