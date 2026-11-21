# Copyright (c) 2026 Hansheng Chen

import torch

try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None


try:
    from sageattention import sageattn
except ImportError:
    sageattn = None


from .sol_attn import get_sol_attn_backend, sol_attn

SOL_COMPILE_BUCKET_SIZE = 4096


@torch.library.custom_op('lakonlab::sol_attn', mutates_args=())
def sol_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefix: torch.Tensor,
        tau: float,
        thresh_type: str,
        kv_splits: int,
        prefix_backend: str,
) -> torch.Tensor:
    """Forward-only Sol attention on BF16 ``(B, T, H, 128)`` tensors.

    ``prefix`` is a CPU scalar: these rows remain an exact KV sink for every query and get dense queries from
    ``prefix_backend`` (``sdpa``, ``flash3`` or ``sage``). Zero KV splits selects automatically.
    """
    if prefix.device.type != 'cpu':
        raise ValueError('Sol prefix length must be a CPU scalar tensor.')
    prefix = int(prefix)
    backend = get_sol_attn_backend(q.device)
    if backend not in ('cute_sm90', 'cute_sm100', 'cute_sm120'):
        raise RuntimeError('Sol attention requires CuTe on SM90/SM100/SM103/SM120; install lakonlab[sol].')
    if kv_splits == 0:
        kv_splits = 4 if backend == 'cute_sm90' and q.size(1) >= 65536 else 1
    scale = q.size(-1) ** -0.5
    if backend == 'cute_sm120':
        if kv_splits != 1:
            raise ValueError('SM120 BSA uses one KV split.')
        from .sol_attn.bsa import sol_bsa
        capacity = (q.size(1) + SOL_COMPILE_BUCKET_SIZE - 1) // SOL_COMPILE_BUCKET_SIZE * SOL_COMPILE_BUCKET_SIZE
        output = sol_bsa(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, prefix=prefix, capacity=capacity)
    else:
        output = sol_attn(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type,
                          kv_splits=kv_splits, sink_start=0, sink_tokens=prefix,
                          compile_bucket_size=SOL_COMPILE_BUCKET_SIZE)
    if prefix:
        if prefix_backend == 'sdpa':
            dense = torch.nn.functional.scaled_dot_product_attention(
                q[:, :prefix].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale).transpose(1, 2)
        elif prefix_backend == 'flash3':
            dense = flash_attn_interface.flash_attn_func(q[:, :prefix], k, v, softmax_scale=scale, causal=False)
        elif prefix_backend == 'sage':
            dense = sageattn(q[:, :prefix], k, v, tensor_layout='NHD', is_causal=False, sm_scale=scale)
        else:
            raise ValueError(f'Unsupported Sol prefix attention backend: {prefix_backend}')
        output[:, :prefix] = dense
    return output


@sol_attention.register_fake
def _(q, k, v, prefix, tau, thresh_type, kv_splits, prefix_backend):
    if torch.cuda.get_device_capability(q.device) in ((9, 0), (12, 0)):
        output = q.new_empty(q.shape)
    else:
        capacity = (q.size(1) + SOL_COMPILE_BUCKET_SIZE - 1) // SOL_COMPILE_BUCKET_SIZE * SOL_COMPILE_BUCKET_SIZE
        output = q.new_empty_strided(q.shape, (capacity * q.size(2) * q.size(3), q.size(2) * q.size(3), q.size(3), 1))
    return output


if flash_attn_interface is not None:
    # Functional custom ops expose FA3's forward/backward to export and keep the fake LSE layout truthful.
    @torch.library.custom_op('lakonlab::flash_attn_3_forward', mutates_args=())
    def _flash_attn_3_forward_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            softmax_scale: float | None,
            causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = flash_attn_interface.flash_attn_func(
            q=q, k=k, v=v, softmax_scale=softmax_scale, causal=causal,
            window_size=(-1, -1), return_attn_probs=True)
        out, lse, *_ = result
        # LSE stays in FA3's native contiguous (B, H, S) layout (no copies); it is
        # permuted to diffusers' public (B, S, H) layout only in the dispatch wrapper
        return out, lse


    @_flash_attn_3_forward_op.register_fake
    def _(q, k, v, softmax_scale, causal):
        lse = q.new_empty((q.size(0), q.size(2), q.size(1)), dtype=torch.float32)
        return q.new_empty(q.shape), lse


    @torch.library.custom_op('lakonlab::flash_attn_3_backward', mutates_args=())
    def _flash_attn_3_backward_op(
            grad_out: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            out: torch.Tensor,
            lse: torch.Tensor,
            softmax_scale: float,
            causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The legacy FA3 bwd operator mutates and aliases explicit dq/dk/dv buffers,
        # which AOTAutograd cannot functionalize. Keep that call opaque and expose
        # only the fresh, non-aliased gradient tensors from this custom op.
        grad_q, grad_k, grad_v = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # Both FA3 signatures default to an unrestricted attention window.
        flash_attn_interface._flash_attn_backward(
            grad_out.contiguous(), q, k, v, out, lse,
            None, None,  # cu_seqlens_q, cu_seqlens_k
            None, None,  # sequed_q, sequed_k
            None, None,  # max_seqlen_q, max_seqlen_k
            grad_q, grad_k, grad_v,
            softmax_scale, causal)
        return grad_q, grad_k, grad_v


    @_flash_attn_3_backward_op.register_fake
    def _(grad_out, q, k, v, out, lse, softmax_scale, causal):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


    def _flash_attn_3_setup_context(ctx, inputs, output):
        q, k, v, softmax_scale, causal = inputs
        # The LSE output is discarded by the regular diffusers attention path. Without
        # this, autograd materializes its missing gradient as a zero tensor and the
        # explicit unsupported-LSE check in backward rejects every ordinary call.
        ctx.set_materialize_grads(False)
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
        ctx.causal = causal


    def _flash_attn_3_backward(ctx, grad_out, grad_lse):
        if grad_lse is not None:
            raise NotImplementedError('Gradients w.r.t. the FlashAttention-3 LSE output are not supported')
        q, k, v, out, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_attn_3_backward_op(
            grad_out, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return grad_q, grad_k, grad_v, None, None


    torch.library.register_autograd(
        'lakonlab::flash_attn_3_forward',
        _flash_attn_3_backward, setup_context=_flash_attn_3_setup_context)


if sageattn is not None:
    # Sage calls torch.cuda.set_device, which Dynamo cannot trace, and its Triton quantization reads raw data
    # pointers, which export cannot trace, so it stays a forward-only opaque custom op.
    @torch.library.custom_op('lakonlab::sage_attn', mutates_args=())
    def _sage_attn_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            sm_scale: float | None,
            is_causal: bool,
            return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = sageattn(
            q=q, k=k, v=v, tensor_layout='NHD', is_causal=is_causal, sm_scale=sm_scale,
            return_lse=return_lse)
        if return_lse:
            out, lse = result  # LSE in the kernel's native (B, H, S) fp32 layout
        else:
            out, lse = result, q.new_empty(0, dtype=torch.float32)
        return out, lse


    @_sage_attn_op.register_fake
    def _(q, k, v, sm_scale, is_causal, return_lse):
        lse = q.new_empty((q.size(0), q.size(2), q.size(1)) if return_lse else 0, dtype=torch.float32)
        out = q.new_empty((*q.shape[:-1], 64 if q.size(-1) <= 64 else 128))
        return out[..., :q.size(-1)], lse


if flash_attn_interface is not None:
    @torch.library.custom_op('lakonlab::flash_attn_varlen_3_forward', mutates_args=())
    def _flash_attn_varlen_3_forward_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            softmax_scale: float | None,
            causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse, *_ = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale, causal=causal, return_attn_probs=True)
        return out, lse


    @_flash_attn_varlen_3_forward_op.register_fake
    def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        lse = q.new_empty((q.size(1), q.size(0)), dtype=torch.float32)
        return q.new_empty(q.shape), lse


    @torch.library.custom_op('lakonlab::flash_attn_varlen_3_backward', mutates_args=())
    def _flash_attn_varlen_3_backward_op(
            grad_out: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            out: torch.Tensor,
            lse: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            softmax_scale: float,
            causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_q, grad_k, grad_v = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        flash_attn_interface._flash_attn_backward(
            grad_out.contiguous(), q, k, v, out, lse,
            cu_seqlens_q, cu_seqlens_k,
            None, None,  # sequed_q, sequed_k
            max_seqlen_q, max_seqlen_k,
            grad_q, grad_k, grad_v,
            softmax_scale, causal)
        return grad_q, grad_k, grad_v


    @_flash_attn_varlen_3_backward_op.register_fake
    def _(grad_out, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k,
          max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


    def _flash_attn_varlen_3_setup_context(ctx, inputs, output):
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal = inputs
        ctx.set_materialize_grads(False)
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q, ctx.max_seqlen_k = max_seqlen_q, max_seqlen_k
        ctx.softmax_scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
        ctx.causal = causal


    def _flash_attn_varlen_3_backward(ctx, grad_out, grad_lse):
        if grad_lse is not None:
            raise NotImplementedError('Gradients w.r.t. the FlashAttention-3 LSE output are not supported')
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_attn_varlen_3_backward_op(
            grad_out, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.softmax_scale, ctx.causal)
        return grad_q, grad_k, grad_v, None, None, None, None, None, None


    torch.library.register_autograd(
        'lakonlab::flash_attn_varlen_3_forward',
        _flash_attn_varlen_3_backward, setup_context=_flash_attn_varlen_3_setup_context)
