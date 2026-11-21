# Sol-Attn source

This implementation vendors and adapts Sol-Attn from NVIDIA's Sol-H3 runtime.
https://github.com/NVlabs/Sana/tree/781e8341fb3b29d3aea5d50265772e9619340b66/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn

The import namespace is `lakonlab.ops.sol_attn`. Upstream third-party notices
and the FlashAttention license are retained alongside the source.

The SM90 implementation receives the logical sequence length at runtime for
routing, query scheduling, exact-attention masks, and split-KV combination.
Its kernel recipe uses runtime route-group counts and last-block masking.
SM90 compilation is shared across padded capacities with the same batch,
head count, KV split count and inner QKV strides. SM100 retains the upstream
capacity-based compile key.
SM90 reads QKV directly at their real lengths; only routing-summary buffers
use the compile bucket's capacity. SM100 retains padded QKV descriptors.
Triton preprocessing keeps logical lengths unspecialized, so aligned and odd
lengths reuse the same bucket's compiled variants.
The FlashAttention utility selects the NVVM `fmax` calling convention from
its binding signature, independently of the CUDA toolkit version.

## Offline SM90 inference

On an SM90 GPU with the serving environment installed, export the default H3 bundle:

```bash
PYTHONPATH=. python tools/export_sol_attention.py /path/to/sol-sm90
PYTHONPATH=. python demo/gradio_pih3.py --aot --sol-aot /path/to/sol-sm90
```

The bundle contains exported CuTe shared libraries for KV split counts 1 and 4,
and a relocatable Triton binary/autotuning cache for routing preprocessing and,
when SageAttention is installed, Sage's query/key quantization. The default export covers batch 1, BF16,
56 heads of dimension 128, and 4096-token capacities through 131072 tokens.
The exporter accepts the routing settings and maximum capacity as arguments.
The source, dependency versions, GPU architecture and routing settings must
match when loading. The Triton cache is installed during CPU configuration;
CUDA libraries are loaded lazily in each GPU process. Transformer block AoT
packages are managed separately by the demo's `--aot` option.

The exported CuTe functions use the current PyTorch stream. The tested Sage
SM90 extension does not preserve nondefault-stream parity even standalone, so
on a nondefault stream every Sage-backed attention is affected: the dense
blocks as well as the Sol prefix rows. Use non-Sage backends for both there.

## SM120 BSA inference

The SM120 path used by `lakonlab.ops.attention.sol_attention` computes selected
blocks with cuDNN Frontend's BF16 BSA kernel and merges Sol's omitted-block
centroid correction. Query-prefix rows are evaluated by the configured dense
backend. QKV use batch 1, dimension 128 and contiguous token/head dimensions.

The BSA kernel keeps the head geometry static and marks sequence dimensions
and their dependent strides dynamic. Routing compaction and residual merging
receive logical lengths at runtime. Routing summaries use 4096-token capacities.
The offline exporter emits one SM120 BSA shared library and the Triton cache
for every exported capacity; no QKV padding is required.

```bash
PYTHONPATH=. python tools/export_sol_attention.py /path/to/sol-sm120
PYTHONPATH=. python demo/gradio_pih3.py --aot --sol-aot /path/to/sol-sm120
```

The BSA kernel and scheduler come from NVIDIA cuDNN Frontend commit
`29106622617bfd9031a53099a6fbbc5e74a474e9`; the route-compaction and residual
helpers come from NVlabs/Sana commit `46d06010be2b422d070b4cd804e5c69bb583c899`.
