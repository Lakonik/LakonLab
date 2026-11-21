# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _silu_mul_kernel(x_ptr, hidden, stride_row, GATE_FIRST: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hidden
        first_ptr = x_ptr + row * stride_row + offs
        first = tl.load(first_ptr, mask=mask).to(tl.float32)
        second = tl.load(first_ptr + hidden, mask=mask).to(tl.float32)
        if GATE_FIRST:
            out = first * tl.sigmoid(first) * second
        else:
            out = second * tl.sigmoid(second) * first
        tl.store(first_ptr, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op('lakonlab::silu_mul_', mutates_args=('x',))
def silu_mul_(x: torch.Tensor, gate_first: bool = True) -> None:
    """In-place SwiGLU product: overwrite the first half of the last dim with silu(gate) * up, where gate is the
    first half (or the second with gate_first=False) and up the other. As a mutating custom op it stays in place in
    compiled graphs, which otherwise copy the whole tensor for an in-place write into a slice. Not differentiable."""
    hidden = x.size(-1) // 2
    first, second = x[..., :hidden], x[..., hidden:]
    if triton is None or not x.is_cuda:
        F.silu(first, inplace=True).mul_(second) if gate_first else first.mul_(F.silu(second))
        return
    x_2d = x.view(-1, x.size(-1))
    _silu_mul_kernel[(x_2d.size(0), triton.cdiv(hidden, 1024))](
        x_2d, hidden, x_2d.stride(0), GATE_FIRST=gate_first, BLOCK=1024)


@silu_mul_.register_fake
def _(x, gate_first=True):
    return None


def swiglu_linear(linear, x, gate_first=True):
    """``linear(silu(gate) * up)`` without grad for a contiguous SwiGLU projection x = [gate, up], or [up, gate]
    with ``gate_first=False``. Linears with a ``forward_swiglu`` (fp8 linears and in-place LoRA layers) compute the
    product inside their activation quantization; others read the product silu_mul_ writes over the first half."""
    if hasattr(linear, 'forward_swiglu'):
        return linear.forward_swiglu(x, gate_first)
    silu_mul_(x, gate_first=gate_first)
    return linear(x[..., :x.size(-1) // 2])
