"""Sol routing with the SM120 BF16 block-sparse attention kernel."""

import math

import torch
import triton
import triton.language as tl

from .bsa_metadata import route_and_compact
from .bsa_residual import merge_sol_residual
from .preprocess import prepare


_compiled = {}


@triton.jit(do_not_specialize=['tokens', 'stride_token', 'stride_head'])
def _pool_query(q, pooled, tokens, stride_token, stride_head, HEADS: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * 64 + tl.arange(0, 64)
    channels = tl.arange(0, 128)
    values = tl.load(q + rows[:, None] * stride_token + head * stride_head + channels[None, :],
                     mask=rows[:, None] < tokens, other=0).to(tl.float32)
    mean = tl.sum(values, axis=0) / tl.minimum(64, tokens - block * 64)
    tl.store(pooled + (block * HEADS + head) * 128 + channels, mean)


def _kernel_args(q, k, v, output, lse, indices, counts, sizes):
    from cutlass.cute.runtime import from_dlpack

    tensors = (q.permute(1, 3, 2, 0), k.permute(1, 3, 2, 0), v.permute(3, 1, 2, 0),
               output.permute(1, 3, 2, 0), lse.permute(2, 1, 0), indices.permute(3, 2, 1, 0),
               counts.permute(2, 1, 0), sizes)
    modes = ((0,), (0,), (1,), (0,), (0,), (0, 1), (0,), (0,))
    args = []
    for tensor, dynamic in zip(tensors, modes):
        descriptor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True)
        for mode in dynamic:
            descriptor = descriptor.mark_compact_shape_dynamic(mode=mode)
        args.append(descriptor)
    return args


@torch.no_grad()
def sol_bsa(q, k, v, *, scale, tau, thresh_type, prefix, capacity):
    from . import interface

    interface._validate_inputs(q, k, v, thresh_type, sink_tokens=prefix, sink_start=0)
    from .sm120.bsa_kernel import BlockSparseAttnForwardSm120Blk64
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda

    if q.shape[0] != 1:
        raise ValueError('SM120 Sol BSA requires batch 1.')
    tokens, heads = q.shape[1:3]
    if any(t.stride()[1:] != (heads * 128, 128, 1) for t in (q, k, v)):
        raise ValueError('SM120 Sol BSA requires contiguous token and head dimensions.')
    blocks = triton.cdiv(tokens, 64)
    if interface._aot_runtime is not None:
        interface._aot_runtime.validate(q, capacity, tau, thresh_type)
    kc, vc, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type,
                                capacity_tokens=capacity)
    kc, vc, threshold = kc[:, :blocks], vc[:, :blocks], threshold[:, :blocks]
    pooled = torch.empty((1, blocks, heads, 128), device=q.device, dtype=torch.float32)
    _pool_query[(blocks, heads)](q, pooled, tokens, q.stride(1), q.stride(2), heads)
    scores = torch.einsum('bqhd,bkhd->bqkh', pooled, kc.float())
    route, indices, counts = route_and_compact(
        scores, threshold, scale_log2=scale * math.log2(math.e),
        sink_first=0, sink_last=triton.cdiv(prefix, 64), metadata_blocks=blocks)
    del pooled, scores, threshold
    sizes = (tokens - torch.arange(blocks, device=q.device, dtype=torch.int32) * 64).clamp(max=64)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((1, heads, tokens), dtype=torch.float32, device=q.device)
    args = _kernel_args(q, k, v, output, lse, indices, counts, sizes)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    if interface._aot_runtime is not None:
        kernel = interface._aot_runtime.kernel(q.device, 1)
    else:
        # Sequence lengths and tensor strides are runtime values in these descriptors.
        key = (q.device.index, tuple(t.stride()[1:] for t in (q, k, v)))
        kernel = _compiled.get(key)
        if kernel is None:
            operator = BlockSparseAttnForwardSm120Blk64(has_block_sizes=True, has_block_nums=True, block_sizes_mode=1)
            kernel = cute.compile(operator, *args[:7], blocks, args[7], scale, stream=stream,
                                  options='--enable-tvm-ffi')
            _compiled[key] = kernel
    kernel(*args[:7], blocks, args[7], scale, stream=stream)
    return merge_sol_residual(q, kc, vc, route, output.transpose(1, 2), lse)
