"""Hopper kernel recipe with runtime sequence-length masking."""

import cutlass

from .fwd import SolAttnSplitForwardSm90
from .mainloop import SolAttnMainloopSm90


def make_kernel(kv_splits: int):
    kernel = SolAttnMainloopSm90 if kv_splits == 1 else SolAttnSplitForwardSm90

    return kernel(
        cutlass.BFloat16,
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=1,
        is_causal=False,
        is_local=False,
        pack_gqa=False,
        tile_m=64,
        tile_n=64,
        num_stages=1,
        num_threads=128,
        # Exact streams mask the last KV block using the logical sequence length.
        sol_attn_exact_mask_seqlen_last_only=True,
        sol_attn_num_splits=kv_splits,
    )


__all__ = ["make_kernel"]
