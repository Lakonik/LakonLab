# Third-party notices

This package vendors and adapts NVIDIA Sol-Engine source released under Apache-2.0.
The license is included in `LICENSE.sana`. The upstream revisions are listed in `UPSTREAM.md`.

The files under `sol_attn/_vendor/flash_attn/cute/` and portions of the SM90
and SM100 design scaffold derive from the FlashAttention project. Its
BSD-3-Clause license is included at
`sol_attn/sm100/LICENSE.flash-attention`.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, PyTorch,
and Triton. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.

The SM120 warp-MMA/TMA execution skeleton and online-softmax helpers are
adapted from NVIDIA cuDNN Frontend's block-sparse-attention reference at commit
`74785165de2da954a2c879a5e3e6f95411c2292d`. That source is licensed under the
Apache License 2.0; adapted files retain the corresponding SPDX header.
The SM90 split-KV combine kernel also comes from the cuDNN BSA reference and
retains the FlashAttention authors’ copyright notice.

The SM120 BSA kernel and scheduler are adapted from NVIDIA cuDNN Frontend
commit `29106622617bfd9031a53099a6fbbc5e74a474e9`, under Apache-2.0.
The license is included in `sm120/LICENSE.cudnn`. The Sol route-compaction
and omitted-block residual helpers derive from NVlabs/Sana commit
`46d06010be2b422d070b4cd804e5c69bb583c899`.
