# Copyright (c) 2026 Hansheng Chen
# INT8 ConvRot adapted from Comfy Kitchen (copyright 2025 Comfy Org).
# Source: https://github.com/Comfy-Org/comfy-kitchen
# Apache 2.0: https://github.com/Comfy-Org/comfy-kitchen/blob/main/LICENSE

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin

from lakonlab.utils import rgetattr, get_root_logger
from .._extension import get_backend
from ..activation import silu_mul_

FP8_DTYPE = torch.float8_e4m3fn
FP8_GRAD_DTYPE = torch.float8_e5m2
FP8_MAX = 448.0
FP8_BLOCK_SIZE = 128  # the only blockwise recipe torch._scaled_mm offers

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # no fallback — quantize_frozen_linears raises without triton
    triton = None

if triton is not None:

    # the fp8 kernels take QMAX and SCALE_FLOOR as constexprs: torch.compile would pass float arguments as fp32 or fp64
    # depending on the torch version, and libdevice.div_rn takes neither mix
    @triton.jit
    def _quantize_fp8_rowwise_kernel(
            x_ptr, q_ptr, scale_ptr, col_scale_ptr, k, stride_xm, stride_qm,
            QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr, HAS_COL_SCALE: tl.constexpr, BLOCK_K: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * stride_xm
        q_row = q_ptr + row * stride_qm
        offs = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            mask = k0 + offs < k
            v = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_COL_SCALE:
                v *= tl.load(col_scale_ptr + k0 + offs, mask=mask, other=0.0)
            acc = tl.maximum(acc, tl.abs(v))
        # IEEE division (triton's default fp32 division is approximate). The scale is
        # rounded to the input dtype (the stored fp32 scale IS the rounded value, so
        # quantization and dequantization cannot mismatch) and the margin in qmax
        # keeps the rounded scale from overflowing the fp8 range
        s = libdevice.div_rn(tl.max(acc, axis=0), QMAX)
        s = s.to(x_ptr.dtype.element_ty).to(tl.float32)
        s = tl.maximum(s, SCALE_FLOOR)
        for k0 in range(0, k, BLOCK_K):
            mask = k0 + offs < k
            v = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_COL_SCALE:
                v *= tl.load(col_scale_ptr + k0 + offs, mask=mask, other=0.0)
            v = libdevice.div_rn(v, s)
            tl.store(q_row + k0 + offs, v.to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row, s)


    @triton.jit
    def _silu_mul(x_row, cols, k, mask, GATE_FIRST: tl.constexpr):
        first = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(x_row + k + cols, mask=mask, other=0.0).to(tl.float32)
        if GATE_FIRST:
            return first * tl.sigmoid(first) * second
        return second * tl.sigmoid(second) * first


    @triton.jit
    def _quantize_fp8_rowwise_swiglu_kernel(
            x_ptr, q_ptr, scale_ptr, k, stride_xm, stride_qm, QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr,
            GATE_FIRST: tl.constexpr, STORE_PRODUCT: tl.constexpr, BLOCK_K: tl.constexpr):
        # _quantize_fp8_rowwise_kernel on silu(gate) * up of the row's halves, rounded to the input dtype as silu_mul_
        # leaves it; with STORE_PRODUCT the product is also written over the first half and quantized from there
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * stride_xm
        q_row = q_ptr + row * stride_qm
        offs = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            mask = k0 + offs < k
            v = _silu_mul(x_row, k0 + offs, k, mask, GATE_FIRST).to(x_ptr.dtype.element_ty)
            if STORE_PRODUCT:
                tl.store(x_row + k0 + offs, v, mask=mask)
            acc = tl.maximum(acc, tl.abs(v.to(tl.float32)))
        s = libdevice.div_rn(tl.max(acc, axis=0), QMAX)
        s = s.to(x_ptr.dtype.element_ty).to(tl.float32)
        s = tl.maximum(s, SCALE_FLOOR)
        tl.debug_barrier()
        for k0 in range(0, k, BLOCK_K):
            mask = k0 + offs < k
            if STORE_PRODUCT:
                v = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                v = _silu_mul(x_row, k0 + offs, k, mask, GATE_FIRST).to(x_ptr.dtype.element_ty).to(tl.float32)
            v = libdevice.div_rn(v, s)
            tl.store(q_row + k0 + offs, v.to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row, s)


    @triton.jit
    def _quantize_fp8_blockwise_kernel(
            x_ptr, q_ptr, scale_ptr, m, k, stride_xm,
            QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        # one (BLOCK_M, BLOCK_K) tile per program: a single pass gives both the
        # block amax and the cast; scales land in the (K // BLOCK_K, M) layout
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        blk = tl.program_id(1)
        cols = blk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (rows[:, None] < m) & (cols[None, :] < k)
        v = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        s = libdevice.div_rn(tl.max(tl.abs(v), axis=1), QMAX)
        s = tl.maximum(s.to(x_ptr.dtype.element_ty).to(tl.float32), SCALE_FLOOR)
        tl.store(q_ptr + rows[:, None].to(tl.int64) * k + cols[None, :],
                 libdevice.div_rn(v, s[:, None]).to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + blk * m + rows, s, mask=rows < m)


    @triton.jit
    def _convrot_stage(v, STRIDE: tl.constexpr, BLOCK_K: tl.constexpr):
        v = v.reshape(BLOCK_K // (4 * STRIDE), 4, STRIDE).trans(0, 2, 1)
        even, odd = tl.split(v.reshape(BLOCK_K // (4 * STRIDE), STRIDE, 2, 2))
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        y0 = 0.5 * (x0 + x1 + x2 - x3)
        y1 = 0.5 * (x0 + x1 - x2 + x3)
        y2 = 0.5 * (x0 - x1 + x2 + x3)
        y3 = 0.5 * (-x0 + x1 + x2 + x3)
        v = tl.join(tl.join(y0, y2), tl.join(y1, y3))
        return v.reshape(BLOCK_K // (4 * STRIDE), STRIDE, 4).trans(0, 2, 1).reshape(BLOCK_K)


    @triton.jit
    def _quantize_int8_rowwise_kernel(
            x_ptr, q_ptr, scale_ptr, col_scale_ptr, k, MAX_VALUE: tl.constexpr,
            MIN_VALUE: tl.constexpr, STAGES: tl.constexpr, HAS_COL_SCALE: tl.constexpr,
            BLOCK_K: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_K)
        v = tl.load(x_ptr + row * k + offs, mask=offs < k, other=0.0).to(tl.float32)
        if HAS_COL_SCALE:  # SmoothQuant: x / s, the weight holds W * s
            v *= tl.load(col_scale_ptr + offs, mask=offs < k, other=0.0)
        # ConvRot's regular H4 (Theorem 3.3, Eq. 9). Radix-4 stages keep
        # rotation and quantization in one launch; STAGES == 0 is plain
        # per-token INT8.
        for stage in tl.static_range(STAGES):
            v = _convrot_stage(v, 1 << (2 * stage), BLOCK_K)
        # Upstream's smaller groups round the rotation before taking absmax.
        if STAGES < 4:
            v = v.to(x_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.maximum(tl.minimum(tl.max(tl.abs(v), axis=0), MAX_VALUE) * (1.0 / 127.0), 1e-30)
        # Match the input-dtype division/rounding in ComfyKitchen's CUDA path.
        # Group 256 takes absmax before the dtype cast. Protect zero fp16 rows.
        scale_t = scale.to(x_ptr.dtype.element_ty).to(tl.float32)
        scale_t = tl.where(scale_t == 0, MIN_VALUE, scale_t)
        v = libdevice.div_rn(v.to(x_ptr.dtype.element_ty).to(tl.float32), scale_t)
        v = v.to(x_ptr.dtype.element_ty).to(tl.float32)
        q = tl.clamp(libdevice.rint(v), -128.0, 127.0).to(tl.int8)
        tl.store(q_ptr + row * k + offs, q, mask=offs < k)
        tl.store(scale_ptr + row, scale)


    @triton.autotune(
        configs=[
            triton.Config(dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64), num_stages=3, num_warps=8),
            triton.Config(dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=32), num_stages=4, num_warps=4),
            triton.Config(dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32), num_stages=4, num_warps=4),
            triton.Config(dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32), num_stages=4, num_warps=4),
            triton.Config(dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32), num_stages=4, num_warps=4),
            triton.Config(dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=32), num_stages=4, num_warps=4),
        ],
        key=['m', 'n', 'k'])
    @triton.jit
    def _int8_mm_kernel(
            a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, bias_ptr, m, n, k,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            HAS_BIAS: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m, num_pid_n = tl.cdiv(m, BLOCK_M), tl.cdiv(n, BLOCK_N)
        group_id = pid // (8 * num_pid_n)
        first_pid_m = group_id * 8
        group_size_m = tl.minimum(num_pid_m - first_pid_m, 8)
        pid_m = first_pid_m + pid % group_size_m
        pid_n = (pid % (8 * num_pid_n)) // group_size_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None].to(tl.int64) * k + offs_k[None, :]
        b_ptrs = b_ptr + cols[None, :].to(tl.int64) * k + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(tl.cdiv(k, BLOCK_K)):
            a = tl.load(a_ptrs, mask=(rows[:, None] < m) & (offs_k[None, :] < k - k0 * BLOCK_K), other=0)
            b = tl.load(b_ptrs, mask=(cols[None, :] < n) & (offs_k[:, None] < k - k0 * BLOCK_K), other=0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        a_scale = tl.load(a_scale_ptr + rows, mask=rows < m, other=0)
        b_scale = tl.load(b_scale_ptr + cols, mask=cols < n, other=0)
        out = acc.to(tl.float32) * (a_scale[:, None] * b_scale[None, :])
        if HAS_BIAS:
            out += tl.load(bias_ptr + cols, mask=cols < n, other=0)[None, :]
        tl.store(c_ptr + rows[:, None].to(tl.int64) * n + cols[None, :], out,
                 mask=(rows[:, None] < m) & (cols[None, :] < n))


def quantize_fp8_channelwise(x):
    """Quantize a 2D weight to fp8 with one scale per output channel (row)."""
    scale = (x.abs().amax(dim=1).float() / FP8_MAX).clamp(min=1e-30)
    q = x.float().div(scale[:, None]).clamp_(-FP8_MAX, FP8_MAX)
    return q.to(FP8_DTYPE), scale


# a triton op: torch.compile and AOTInductor launch the triton kernel itself (AOTInductor packages it) rather than
# calling back into Python, which also spares the first call the kernel's Python-side load
@torch.library.triton_op('lakonlab::quantize_fp8_rowwise', mutates_args=())
def _quantize_fp8_rowwise(
        x: torch.Tensor, col_scale: torch.Tensor | None, dtype_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, dtype_name)
    m, k = x.shape
    q = torch.empty_like(x, dtype=dtype)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    # the clamp floor (fp16 -> 2^-24, bf16 -> 2^-126) must be a NORMAL fp32 value —
    # a subnormal floor is effectively flushed to zero in the kernel, quantizing
    # all-zero rows to 0/0 = NaN — and at least the input dtype's smallest
    # subnormal, which bounds v / floor below qmax when the scale underflows
    scale_floor = max(torch.finfo(x.dtype).tiny * torch.finfo(x.dtype).eps, torch.finfo(torch.float32).tiny)
    torch.library.wrap_triton(_quantize_fp8_rowwise_kernel)[(m,)](
        x, q, scale, x if col_scale is None else col_scale, k, x.stride(0), q.stride(0),
        QMAX=torch.finfo(dtype).max * (1.0 - 2.0 ** -8), SCALE_FLOOR=scale_floor,
        HAS_COL_SCALE=col_scale is not None, BLOCK_K=1024, num_warps=4)
    return q, scale


def quantize_fp8_rowwise(x, col_scale=None, dtype=FP8_DTYPE):
    """Dynamically quantize activations to fp8 with one scale per row (token).

    A single fused triton kernel (amax + scale + cast in one launch). If
    ``col_scale`` (an fp32 vector of size ``x.size(-1)``) is given, ``x`` is
    multiplied by it columnwise before quantization, inside the same kernel.
    """
    # the kernel takes a row stride, so row-strided views such as a SwiGLU gate half are not copied
    return _quantize_fp8_rowwise(
        x if x.stride(-1) == 1 else x.contiguous(), col_scale, str(dtype).removeprefix('torch.'))


def _launch_quantize_fp8_rowwise_swiglu(x, gate_first, dtype_name, store_product):
    dtype = getattr(torch, dtype_name)
    m, k = x.size(0), x.size(1) // 2
    q = x.new_empty((m, k), dtype=dtype)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    scale_floor = max(torch.finfo(x.dtype).tiny * torch.finfo(x.dtype).eps, torch.finfo(torch.float32).tiny)
    torch.library.wrap_triton(_quantize_fp8_rowwise_swiglu_kernel)[(m,)](
        x, q, scale, k, x.stride(0), q.stride(0), QMAX=torch.finfo(dtype).max * (1.0 - 2.0 ** -8),
        SCALE_FLOOR=scale_floor, GATE_FIRST=gate_first, STORE_PRODUCT=store_product, BLOCK_K=1024, num_warps=4)
    return q, scale


@torch.library.triton_op('lakonlab::quantize_fp8_rowwise_swiglu', mutates_args=())
def _quantize_fp8_rowwise_swiglu(
        x: torch.Tensor, gate_first: bool, dtype_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_quantize_fp8_rowwise_swiglu(x, gate_first, dtype_name, False)


@torch.library.triton_op('lakonlab::quantize_fp8_rowwise_swiglu_', mutates_args=('x',))
def _quantize_fp8_rowwise_swiglu_(
        x: torch.Tensor, gate_first: bool, dtype_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_quantize_fp8_rowwise_swiglu(x, gate_first, dtype_name, True)


def quantize_fp8_rowwise_swiglu(x, gate_first=True, store_product=False, dtype=FP8_DTYPE):
    """quantize_fp8_rowwise of the SwiGLU product silu(gate) * up of a (m, 2k) projection x = [gate, up], or
    [up, gate] with ``gate_first=False``, in one kernel that does not write the 16-bit product. With
    ``store_product=True`` the product is also written in place over the first half of x, as silu_mul_ does."""
    assert x.ndim == 2 and x.stride(-1) == 1 and x.size(1) % 2 == 0
    fn = _quantize_fp8_rowwise_swiglu_ if store_product else _quantize_fp8_rowwise_swiglu
    return fn(x, gate_first, str(dtype).removeprefix('torch.'))


def dequantize_fp8(q, scale, dtype, block_size=None):
    if block_size:  # (K // block_size, N) blockwise scales
        scale = scale.T.repeat_interleave(block_size, dim=1)[:, :q.size(1)]
        return q.to(dtype) * scale.to(dtype)
    return q.to(dtype) * scale.unsqueeze(-1).to(dtype)


@torch.library.triton_op('lakonlab::quantize_fp8_blockwise', mutates_args=())
def _quantize_fp8_blockwise(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    q = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty((-(-k // block_size), m), dtype=torch.float32, device=x.device)
    scale_floor = max(torch.finfo(x.dtype).tiny * torch.finfo(x.dtype).eps, torch.finfo(torch.float32).tiny)
    torch.library.wrap_triton(_quantize_fp8_blockwise_kernel)[(-(-m // 32), scale.size(0))](
        x, q, scale, m, k, x.stride(0), QMAX=FP8_MAX * (1.0 - 2.0 ** -8), SCALE_FLOOR=scale_floor,
        BLOCK_M=32, BLOCK_K=block_size, num_warps=4)
    return q, scale


def quantize_fp8_blockwise(x, block_size=FP8_BLOCK_SIZE):
    """Quantize a 2D tensor to fp8 with one scale per (row, ``block_size``-wide slice).

    Returns the scales as ``(ceil(K / block_size), rows)``, which is the
    ``scale_b`` layout of ``torch._scaled_mm``'s 1x128 recipe for weights and
    whose transpose is its ``scale_a`` layout for activations. CUDA inputs use
    the fused triton kernel; other devices use an equivalent torch fallback
    (same qmax margin and scale rounding), so weights quantized at conversion
    time match the kernel.
    """
    if x.is_cuda:
        return _quantize_fp8_blockwise(x.contiguous(), block_size)
    m, k = x.shape
    x_pad = F.pad(x.float(), (0, -k % block_size)).unflatten(1, (-1, block_size))
    scale_floor = max(torch.finfo(x.dtype).tiny * torch.finfo(x.dtype).eps, torch.finfo(torch.float32).tiny)
    scale = (x_pad.abs().amax(dim=2) / (FP8_MAX * (1.0 - 2.0 ** -8))).to(x.dtype).float().clamp(min=scale_floor)
    q = x_pad.div(scale[..., None]).flatten(1)[:, :k].to(FP8_DTYPE)
    return q, scale.T.contiguous()


def _fp8_linear_forward(x, qweight, qscale, bias):
    x_8, x_scale = quantize_fp8_rowwise(x)
    # scales and bias are applied in the GEMM epilogue (rowwise recipe), so the
    # output is written once. use_fast_accum stays disabled: ~4% end-to-end
    # speedup but doubled the model output error on real inputs
    return torch._scaled_mm(
        x_8, qweight.T, scale_a=x_scale, scale_b=qscale.unsqueeze(0), bias=bias, out_dtype=x.dtype)


def _fp8_blockwise_linear_forward(x, qweight, qscale, bias, block_size):
    m = x.size(0)
    x = F.pad(x, (0, 0, 0, -m % 4)) if m % 4 else x  # cuBLAS requires M % 4 == 0 for the 1x128 recipe
    x_8, x_scale = quantize_fp8_blockwise(x, block_size)
    return torch._scaled_mm(
        x_8, qweight.T, scale_a=x_scale.T, scale_b=qscale, bias=bias, out_dtype=x.dtype)[:m]


# the grad path is a custom op with registered autograd rather than an
# autograd.Function, whose backward can be lost under torch.compile/AOTAutograd;
# weight gradients are never produced
@torch.library.custom_op('lakonlab::fp8_linear', mutates_args=())
def _fp8_linear(
        x: torch.Tensor, qweight: torch.Tensor, qweight_t: torch.Tensor | None,
        qscale: torch.Tensor, qscale_ones: torch.Tensor | None,
        bias: torch.Tensor | None, quantize_backward: bool, block_size: int | None) -> torch.Tensor:
    if block_size:
        return _fp8_blockwise_linear_forward(x, qweight, qscale, bias, block_size)
    return _fp8_linear_forward(x, qweight, qscale, bias)


@_fp8_linear.register_fake
def _(x, qweight, qweight_t, qscale, qscale_ones, bias, quantize_backward, block_size):
    return x.new_empty((x.shape[0], qweight.shape[0]))


def _fp8_linear_setup_context(ctx, inputs, output):
    _, qweight, qweight_t, qscale, qscale_ones, _, quantize_backward, block_size = inputs
    ctx.save_for_backward(qweight, qweight_t, qscale, qscale_ones)
    ctx.quantize_backward = quantize_backward
    ctx.block_size = block_size


def _fp8_linear_backward(ctx, grad_out):
    qweight, qweight_t, qscale, qscale_ones = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    if not ctx.quantize_backward:  # also the blockwise path: dequantized weights
        grad_in = grad_out @ dequantize_fp8(qweight, qscale, grad_out.dtype, ctx.block_size)
    else:
        # the per-channel weight scale lies on this GEMM's contraction dim, so it is
        # folded into grad_out inside the quantization kernel; qweight_t provides the
        # column-major rhs required by _scaled_mm (per-call transposes instead
        # measured 27-48% of this backward)
        g_8, g_scale = quantize_fp8_rowwise(grad_out, col_scale=qscale, dtype=FP8_GRAD_DTYPE)
        grad_in = torch._scaled_mm(
            g_8, qweight_t.T, scale_a=g_scale, scale_b=qscale_ones, out_dtype=grad_out.dtype)
    return grad_in, None, None, None, None, None, None, None


torch.library.register_autograd(
    'lakonlab::fp8_linear', _fp8_linear_backward, setup_context=_fp8_linear_setup_context)


class FrozenFP8Linear(nn.Linear):
    """Frozen linear layer with fp8 weights and kernels (W8A8).

    Uses channelwise (per-output-feature) weight scaling and dynamic rowwise
    (per-token) activation scaling, or with ``block_size`` set, 1 x block_size
    blockwise scaling of both (the DeepSeek-V3 recipe; the same error size
    but spread evenly within each token, so per-token error patterns are
    weaker). With ``quantize_backward=True`` the rowwise backward
    (input-gradient) GEMM uses E5M2 gradients and a transposed E4M3 weight copy
    created at conversion time; otherwise the backward matmul runs on the
    dequantized weight. Weight gradients are not supported.
    """

    def forward(self, x):
        x_2d = x.reshape(-1, x.size(-1))
        if not x_2d.is_cuda or x_2d.dtype not in (torch.bfloat16, torch.float16):
            # CPU tensors and non-16-bit inputs (e.g. fp32 timestep embeddings)
            # take the dequantized path, keeping their precision
            out = F.linear(
                x_2d, dequantize_fp8(self.qweight, self.qscale, x.dtype, self.block_size),
                None if self.bias is None else self.bias.to(x.dtype))
        elif torch.is_grad_enabled() and x_2d.requires_grad:
            out = _fp8_linear(
                x_2d, self.qweight, self.qweight_t, self.qscale,
                self.qscale_ones, self.bias, self.quantize_backward, self.block_size)
        elif self.block_size:
            out = _fp8_blockwise_linear_forward(x_2d, self.qweight, self.qscale, self.bias, self.block_size)
        else:
            # no input gradients needed: skip the autograd op
            out = _fp8_linear_forward(x_2d, self.qweight, self.qscale, self.bias)
        return out.reshape(*x.shape[:-1], out.size(-1))

    def forward_swiglu(self, x, gate_first=True, store_product=False):
        """``self(silu(gate) * up)`` without grad for a contiguous SwiGLU projection x = [gate, up], or [up, gate]
        with ``gate_first=False``. The rowwise recipe computes the product inside its activation quantization;
        with ``store_product=True`` it also writes the product over the first half of x. Other inputs take the
        product silu_mul_ writes there."""
        x_2d = x.view(-1, x.size(-1))
        if not x_2d.is_cuda or x_2d.dtype not in (torch.bfloat16, torch.float16) or self.block_size:
            silu_mul_(x, gate_first=gate_first)
            return self(x[..., :x.size(-1) // 2])
        x_8, x_scale = quantize_fp8_rowwise_swiglu(x_2d, gate_first, store_product)
        out = torch._scaled_mm(
            x_8, self.qweight.T, scale_a=x_scale, scale_b=self.qscale.unsqueeze(0), bias=self.bias, out_dtype=x.dtype)
        return out.reshape(*x.shape[:-1], out.size(-1))

    def extra_repr(self):
        return super().extra_repr() + f', quantize_backward={self.quantize_backward}, block_size={self.block_size}'

    @classmethod
    def convert(cls, linear, quantize_backward=False, use_meta=False, block_size=None):
        """Convert a frozen nn.Linear into a FrozenFP8Linear in place."""
        assert type(linear) is nn.Linear
        assert not linear.weight.requires_grad and (linear.bias is None or not linear.bias.requires_grad)
        assert not (block_size and quantize_backward), 'blockwise fp8 has no fp8 backward'
        if use_meta or linear.weight.is_meta:
            q = torch.empty_like(linear.weight, dtype=FP8_DTYPE, device='meta')
            scale = torch.empty(
                (-(-linear.in_features // block_size), linear.out_features) if block_size else linear.out_features,
                dtype=torch.float32, device='meta')
        elif block_size:
            q, scale = quantize_fp8_blockwise(linear.weight.data, block_size)
        else:
            q, scale = quantize_fp8_channelwise(linear.weight.data)
        del linear.weight
        linear.__class__ = cls
        linear.quantize_backward = quantize_backward
        linear.block_size = block_size
        linear.qweight = nn.Parameter(q, requires_grad=False)
        # created at conversion so the parameter set is final before DDP/FSDP
        # wrapping, checkpointing and compile
        linear.register_parameter(
            'qweight_t', nn.Parameter(q.T.contiguous(), requires_grad=False) if quantize_backward else None)
        linear.qscale = nn.Parameter(scale, requires_grad=False)
        # (1, in_features) fp32 ones for the backward GEMM's scale_b; per-layer
        # rather than a global cache, which would leak real tensors into
        # torch.compile's FakeTensor tracing
        linear.register_parameter(
            'qscale_ones',
            nn.Parameter(q.new_ones((1, linear.in_features), dtype=torch.float32), requires_grad=False)
            if quantize_backward else None)
        return linear


def _rotate_convrot(x, group_size=256):
    """Apply a symmetric orthogonal regular Hadamard transform in float32."""
    v = x.float()
    for stage in range((group_size.bit_length() - 1) // 2):
        stride = 4 ** stage
        x0, x1, x2, x3 = v.reshape(*x.shape[:-1], -1, 4, stride).unbind(-2)
        v = torch.stack((
            x0 + x1 + x2 - x3, x0 + x1 - x2 + x3,
            x0 - x1 + x2 + x3, -x0 + x1 + x2 + x3
        ), dim=-2).reshape(x.shape) * 0.5
    return v


def quantize_int8_channelwise(x, group_size=1):
    """Quantize a 2D weight to INT8 with one scale per output channel.

    ``group_size`` > 1 applies the ConvRot rotation over groups of that size
    first; 1 is plain per-channel INT8.
    """
    if x.is_cuda:
        return quantize_int8_rowwise(x, group_size)
    v = _rotate_convrot(x, group_size)
    if group_size < 256:
        v = v.to(x.dtype).float()
    scale = (v.abs().amax(dim=-1).clamp(max=torch.finfo(x.dtype).max) * (1.0 / 127.0)).clamp(min=1e-30)
    scale_t = scale.to(x.dtype)
    scale_t = torch.where(scale_t == 0, torch.finfo(x.dtype).tiny, scale_t)
    q = (v.to(x.dtype) / scale_t[:, None]).round().clamp(-128, 127).to(torch.int8)
    return q, scale


@torch.library.triton_op('lakonlab::quantize_int8_rowwise', mutates_args=())
def _quantize_int8_rowwise(
        x: torch.Tensor, group_size: int = 1, col_scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(m, device=x.device, dtype=torch.float32)
    if m:
        torch.library.wrap_triton(_quantize_int8_rowwise_kernel)[(m,)](
            x, q, scale, x if col_scale is None else col_scale, k, torch.finfo(x.dtype).max, torch.finfo(x.dtype).tiny,
            STAGES=(group_size.bit_length() - 1) // 2, HAS_COL_SCALE=col_scale is not None,
            BLOCK_K=triton.next_power_of_2(k), num_warps=4, enable_fp_fusion=False)
    return q, scale


def quantize_int8_rowwise(x, group_size=1, col_scale=None):
    """Quantize activations to INT8 with one scale per row (token).

    ``group_size`` > 1 applies the ConvRot rotation over groups of that size
    first. If ``col_scale`` (an fp32 vector of size ``x.size(-1)``) is given,
    ``x`` is multiplied by it columnwise before quantization, inside the kernel.
    """
    return _quantize_int8_rowwise(x.contiguous(), group_size, col_scale)


def dequantize_int8(q, scale, dtype, group_size=1):
    return _rotate_convrot(q.float() * scale[:, None], group_size).to(dtype)


def _int8_linear_forward(x, qweight, qscale, bias, group_size=1, inv_smooth=None):
    x_8, x_scale = quantize_int8_rowwise(x, group_size, inv_smooth)
    m, k = x.shape
    n = qweight.shape[0]
    out = x.new_empty((m, n))
    if (m >= 256 and k % 16 == 0 and n % 8 == 0 and x.dtype == torch.bfloat16
            and (bias is None or bias.dtype == torch.bfloat16)
            and torch.cuda.get_device_capability(x.device) == (9, 0)):
        backend = get_backend(__package__, optional=True)
        if backend is not None:
            backend.int8_mm(x_8, qweight, x_scale, qscale, bias, out)
            return out
    if m:
        grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(n, meta['BLOCK_N']),)
        _int8_mm_kernel[grid](
            x_8, qweight, out, x_scale, qscale, x if bias is None else bias, m, n, k,
            HAS_BIAS=bias is not None, enable_fp_fusion=False)
    return out


@torch.library.custom_op('lakonlab::int8_linear', mutates_args=())
def _int8_linear(
        x: torch.Tensor, qweight: torch.Tensor, qscale: torch.Tensor,
        bias: torch.Tensor | None, group_size: int = 1, inv_smooth: torch.Tensor | None = None) -> torch.Tensor:
    return _int8_linear_forward(x, qweight, qscale, bias, group_size, inv_smooth)


@_int8_linear.register_fake
def _(x, qweight, qscale, bias, group_size=1, inv_smooth=None):
    return x.new_empty((x.shape[0], qweight.shape[0]))


class FrozenINT8Linear(nn.Linear):
    """Frozen INT8 linear with per-token and per-output-channel scales (W8A8).

    CUDA fp16/bf16 inference uses the fused quantization kernel and the INT8
    GEMM. Other inputs use the dequantized weight. This layer is
    inference-only; input gradients and weight gradients are not supported.

    ``convrot_groupsize`` > 1 rotates groups of input channels with ConvRot's
    regular Hadamard transform before quantization (fused in the kernel), 1 is
    plain per-token INT8. With ``smooth`` (SmoothQuant, Xiao et al. 2023), a
    per-input-channel factor ``s`` migrates activation outliers into the
    weight: the stored weight is ``W * s`` and the input is quantized as
    ``x / s`` inside the kernel, so ``y = (x / s) (W s)^T`` is unchanged in
    exact arithmetic while each token's dynamic range shrinks.
    """

    def forward(self, x):
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('FrozenINT8Linear is inference-only; input gradients are not supported')
        x_2d = x.reshape(-1, x.size(-1))
        if not x_2d.is_cuda or x_2d.dtype not in (torch.bfloat16, torch.float16):
            weight = dequantize_int8(self.qweight, self.qscale, x.dtype, self.convrot_groupsize)
            if self.inv_smooth is not None:
                weight = weight * self.inv_smooth.to(x.dtype)  # (W s) / s
            out = F.linear(x_2d, weight, None if self.bias is None else self.bias.to(x.dtype))
        else:
            out = _int8_linear(x_2d, self.qweight, self.qscale, self.bias, self.convrot_groupsize, self.inv_smooth)
        return out.reshape(*x.shape[:-1], out.size(-1))

    def extra_repr(self):
        return super().extra_repr() + f', convrot_groupsize={self.convrot_groupsize}, smooth={self.inv_smooth is not None}'

    @classmethod
    def convert(cls, linear, use_meta=False, convrot=False, smooth=None):
        """Convert a frozen nn.Linear into a FrozenINT8Linear in place.

        Args:
            convrot (bool): Rotate with the largest ConvRot group (256, 64 or 16)
                dividing ``in_features``.
            smooth (torch.Tensor | None): Per-input-channel factor ``s`` of shape
                ``(in_features,)``; see the class docstring.
        """
        assert type(linear) is nn.Linear
        assert not linear.weight.requires_grad and (linear.bias is None or not linear.bias.requires_grad)
        convrot_groupsize = next(size for size in (256, 64, 16) if linear.in_features % size == 0) if convrot else 1
        if use_meta or linear.weight.is_meta:
            q = torch.empty_like(linear.weight, dtype=torch.int8, device='meta')
            scale = torch.empty(linear.out_features, dtype=torch.float32, device='meta')
            inv_smooth = None if smooth is None else torch.empty(linear.in_features, dtype=torch.float32, device='meta')
        elif smooth is None:
            q, scale = quantize_int8_channelwise(linear.weight.data, convrot_groupsize)
            inv_smooth = None
        else:
            smooth = smooth.to(device=linear.weight.device, dtype=torch.float32)
            q, scale = quantize_int8_channelwise(
                (linear.weight.data.float() * smooth).to(linear.weight.dtype), convrot_groupsize)
            inv_smooth = 1 / smooth  # the kernel multiplies columns by it
        del linear.weight
        linear.__class__ = cls
        linear.convrot_groupsize = convrot_groupsize
        linear.qweight = nn.Parameter(q, requires_grad=False)
        linear.qscale = nn.Parameter(scale, requires_grad=False)
        linear.register_parameter(
            'inv_smooth', None if inv_smooth is None else nn.Parameter(inv_smooth.contiguous(), requires_grad=False))
        return linear


def smooth_factors(act_absmax, weight, alpha=0.5):
    """SmoothQuant's per-input-channel factor ``s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)``.

    ``alpha`` in [0, 1] sets how much of the quantization difficulty migrates from the
    activation to the weight: 1 equalizes the activation channels' maxima, 0 equalizes the
    weight columns' maxima instead.
    """
    act_absmax = act_absmax.to(device=weight.device, dtype=torch.float32).clamp(min=1e-5)
    weight_absmax = weight.float().abs().amax(dim=0).clamp(min=1e-5)
    return (act_absmax ** alpha / weight_absmax ** (1 - alpha)).clamp(min=1e-5)


def _set_reported_dtype(module, dtype):
    """Preserve the compute dtype, including modules with a read-only dtype property."""
    try:
        module.dtype = dtype
    except AttributeError:
        cls = type(module)
        module.__class__ = type(cls.__name__, (cls,), dict(
            __module__=cls.__module__, __qualname__=cls.__qualname__,
            dtype=property(lambda self: self._reported_dtype,
                           lambda self, value: setattr(self, '_reported_dtype', value))))
        module.dtype = dtype


def quantize_frozen_linears(
        module, quantize_backward=False, exclude_keys=None, use_meta=False,
        type='fp8', smooth_stats=None):
    """Convert all fully-frozen nn.Linear submodules to W8A8 kernels.

    The default fp8 mode requires CUDA compute capability 8.9 or above —
    ``torch._scaled_mm`` raises on older GPUs. Call after freezing (e.g.
    flex_freeze) and dtype casting, and before DDP/FSDP wrapping, optimizer
    construction and torch.compile; only linears whose parameters all have
    requires_grad=False are converted. Quantized tensors are stored as
    non-trainable parameters (not buffers), so DDP buffer broadcasting, EMA
    syncing and trainable-only checkpoints skip them. With CUDA available,
    CPU-resident linears are quantized on the current CUDA device one layer at
    a time and returned to the CPU, so only one layer is held in VRAM. Do not call
    ``.to(dtype=...)`` on a converted model — fp8 is a floating dtype, so a
    generic cast would clobber the quantized weights (fails loudly at the
    next forward).

    Args:
        module (nn.Module): Root module whose frozen linears are converted in place.
        quantize_backward (bool): 'fp8' only: run the input-gradient GEMM in fp8 using
            a transposed weight copy (doubling quantized-weight memory) instead of
            on dequantized weights. Defaults to False.
        exclude_keys (list[str] | None): Names of modules to keep unquantized, with
            the same matching semantics as flex_freeze ('self.'-prefixed keys match
            full module names, other keys match substrings). Model wrappers prepend
            their fp32 output heads to this list. Defaults to None.
        use_meta (bool): Convert to structure-only placeholders on the meta
            device, skipping the quantization compute and memory.
        type (str): Quantization method. INT8 can be faster than FP8 on RTX 40/50-series
            GPUs. RTX 30-series supports INT8 but has no native FP8 support.
            - 'fp8' (default): Rowwise/channelwise scales. Supports input gradients.
            - 'fp8_blockwise': 1x128 blockwise scales on activations and weights
              (DeepSeek-V3 recipe). Supports input gradients on dequantized weights.
              Requires torch >= 2.9 built against CUDA 12.9+ and a GPU with blockwise
              ``_scaled_mm`` support (Hopper only, as of torch 2.11).
            - 'int8': Per-token INT8, inference-only.
            - 'int8_smoothquant': 'int8' with SmoothQuant channel smoothing from
              ``smooth_stats``, inference-only.
            - 'int8_convrot': INT8 with fused Hadamard rotation, inference-only.
              Near-bf16 quality, but substantial rotation overhead. Recommended for
              speed only on RTX 30/40/50-series GPUs.
        smooth_stats (str | dict | None): 'int8_smoothquant' only. Per-layer activation
            absmax over the input channels, keyed by module name relative to ``module``,
            as written by ``tools/calibrate_quant_smoothing.py`` (a checkpoint path or the
            loaded dict). Layers absent from it are quantized without smoothing.
    """
    if type not in ('fp8', 'fp8_blockwise', 'int8', 'int8_smoothquant', 'int8_convrot'):
        raise ValueError(f'Unknown quantization type: {type}')
    if quantize_backward and type != 'fp8':
        raise ValueError('quantize_backward is only supported with type=fp8')
    if (smooth_stats is not None) != (type == 'int8_smoothquant'):
        raise ValueError('smooth_stats is required by, and only by, type=int8_smoothquant')
    if isinstance(smooth_stats, str):
        from lakonlab.runner.checkpoint import _load_checkpoint
        smooth_stats = _load_checkpoint(smooth_stats, map_location='cpu')
    if triton is None:
        raise ImportError(
            'quantize_frozen_linears requires triton for the fused W8A8 kernels, which is '
            'unavailable in this environment (triton is bundled with PyTorch on Linux; there '
            'is no macOS support).')
    if exclude_keys is None:
        exclude_keys = []

    logger = get_root_logger()

    # dtype-based input casting must not see the fp8 parameter dtype: record every module's reported
    # dtype before quantization and restore it afterwards (see below)
    dtypes = {}
    for name, m in module.named_modules():
        if isinstance(m, ModelMixin):
            continue
        if hasattr(m, 'dtype'):
            dtypes[name] = m.dtype
        elif m is module:
            dtypes[name] = next(
                (t.dtype for t in itertools.chain(m.parameters(), m.buffers()) if t.is_floating_point()), None)

    convert_names = []
    for name, m in module.named_modules():
        if m.__class__ is not nn.Linear or m.weight.requires_grad or (
                m.bias is not None and m.bias.requires_grad):
            continue
        if type.startswith('fp8') and (m.in_features % 16 != 0 or m.out_features % 16 != 0):
            logger.warning(f'Skipping fp8 quantization of {name}: features not divisible by 16')
            continue
        if type.startswith('int8') and m.in_features % (16 if type == 'int8_convrot' else 4) != 0:
            logger.warning(f'Skipping {type} quantization of {name}: input width not divisible')
            continue
        excluded = False
        for exclude_key in exclude_keys:
            if exclude_key.startswith('self.'):  # use full name matching (a linear or a parent module)
                if exclude_key[5:] == name or name.startswith(exclude_key[5:] + '.'):
                    excluded = True
                    break
            elif exclude_key in name:  # use partial name matching
                excluded = True
                break
        if not excluded:
            convert_names.append(name)

    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else None
    num_smoothed = 0
    for name in convert_names:
        linear = rgetattr(module, name)
        quantize_on_device = device is not None and linear.weight.device.type == 'cpu' and not use_meta
        if quantize_on_device:  # only the weight moves; the bias keeps its own device
            linear.weight.data = linear.weight.data.to(device)
        if type.startswith('fp8'):
            FrozenFP8Linear.convert(
                linear, quantize_backward=quantize_backward, use_meta=use_meta,
                block_size=FP8_BLOCK_SIZE if type == 'fp8_blockwise' else None)
        else:
            smooth = None
            if smooth_stats is not None and name in smooth_stats:
                smooth = smooth_factors(smooth_stats[name], linear.weight)
                num_smoothed += 1
            FrozenINT8Linear.convert(linear, use_meta=use_meta, convrot=type == 'int8_convrot', smooth=smooth)
        if quantize_on_device:  # return the tensors quantized from the weight
            for param_name, param in linear.named_parameters():
                if param_name != 'bias':
                    param.data = param.data.cpu()
    if smooth_stats is not None and num_smoothed < len(convert_names):
        logger.warning(f'{len(convert_names) - num_smoothed} of {len(convert_names)} quantized layers have no '
                       f'smoothing statistics and are quantized without smoothing')

    if len(convert_names) > 0:
        for name, m in module.named_modules():
            if isinstance(m, ModelMixin):  # diffusers dtype detection (get_parameter_dtype) skips these names
                m._keep_in_fp32_modules = list(m._keep_in_fp32_modules or []) + ['qweight', 'qscale']
            elif dtypes.get(name) is not None:
                _set_reported_dtype(m, dtypes[name])

    logger.info(f'Quantized {len(convert_names)} frozen linear layers to {type}')
