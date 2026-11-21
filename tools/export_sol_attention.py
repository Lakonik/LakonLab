# Copyright (c) 2026 Hansheng Chen

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
import triton
import cutlass.cute as cute

from lakonlab.models.architectures.diffusers.attention import default_attention_backend
from lakonlab.ops import attention as attention_ops
from lakonlab.ops.attention import SOL_COMPILE_BUCKET_SIZE
from lakonlab.ops.sol_attn import interface
from lakonlab.ops.sol_attn.aot import source_hash, versions
from lakonlab.ops.sol_attn.preprocess import _reduce_kv_kernel, prepare


def parse_args():
    parser = argparse.ArgumentParser(description='Export Sol libraries and Triton binaries for SM90 or SM120 serving.')
    parser.add_argument('output', type=Path)
    parser.add_argument('--heads', type=int, default=56)
    parser.add_argument('--max-tokens', type=int, default=2 ** 17)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--thresh-type', choices=['exact', 'diag'], default='exact')
    parser.add_argument('--kv-splits', nargs='+', type=int, choices=[1, 2, 4], default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    arch = torch.cuda.get_device_capability()
    if arch not in ((9, 0), (12, 0)):
        raise ValueError('This exporter supports SM90 and SM120.')
    if args.kv_splits is None:
        args.kv_splits = [1, 4] if arch == (9, 0) else [1]
    if arch == (12, 0) and args.kv_splits != [1]:
        raise ValueError('SM120 BSA uses one KV split.')
    if args.max_tokens % SOL_COMPILE_BUCKET_SIZE:
        raise ValueError('--max-tokens must be a multiple of the Sol compile bucket size.')
    args.output.mkdir(parents=True, exist_ok=True)
    capacities = list(range(SOL_COMPILE_BUCKET_SIZE, args.max_tokens + 1, SOL_COMPILE_BUCKET_SIZE))
    kernels, symbols = {}, {}
    interface._compiled.clear()
    if arch == (12, 0):
        from lakonlab.ops.sol_attn.bsa import _compiled, sol_bsa
        _compiled.clear()
    _reduce_kv_kernel.cache_results = True
    with tempfile.TemporaryDirectory(dir=args.output) as temporary, triton.knobs.cache.scope():
        triton.knobs.cache.dir = temporary
        # The logical length stays dynamic; routing buffers specialize on their bucket capacity.
        for capacity in capacities:
            qkv = torch.zeros((1, capacity - 1, args.heads, 128), device='cuda', dtype=torch.bfloat16)
            if arch == (9, 0):
                prepare(qkv, qkv, qkv, tau=args.tau, scale=128 ** -0.5,
                        thresh_type=args.thresh_type, capacity_tokens=capacity)
            else:
                sol_bsa(qkv, qkv, qkv, scale=128 ** -0.5, tau=args.tau,
                        thresh_type=args.thresh_type, prefix=0, capacity=capacity)
            del qkv
            print(f'Routing: {capacity} tokens', flush=True)

        for splits in args.kv_splits:
            tokens = (splits - 1) * SOL_COMPILE_BUCKET_SIZE + 513
            qkv = torch.zeros((1, tokens, args.heads, 128), device='cuda', dtype=torch.bfloat16)
            if arch == (9, 0):
                interface.sol_attn(
                    qkv, qkv, qkv, tau=args.tau, thresh_type=args.thresh_type, kv_splits=splits,
                    compile_bucket_size=SOL_COMPILE_BUCKET_SIZE)
                key = next(key for key in interface._compiled if key[4] == splits)
                compiled = interface._compiled[key]
                symbol = f'sol_sm90_s{splits}'
            else:
                if len(_compiled) != 1:
                    raise RuntimeError('SM120 BSA did not reuse one compiled kernel across capacities.')
                compiled = next(iter(_compiled.values()))
                symbol = 'sol_sm120_bsa'
            obj = Path(temporary) / f'{symbol}.o'
            library = args.output / f'{symbol}.so'
            compiled.export_to_c(str(obj), function_name=symbol)
            subprocess.run(['gcc', '-shared', '-o', str(library), str(obj),
                            *cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)], check=True)
            kernels[str(splits)] = library.name
            symbols[str(splits)] = symbol
            del qkv
            print(f'CuTe: {symbol}', flush=True)

        if default_attention_backend('dense') == 'sage':  # Sage passes the app's checks
            # Sage's query/key quantizers specialize on singleton lengths and alignment.
            for query_tokens, key_tokens in ((1, 1), (16, 16), (65, 65), (65, 385), (65, 512)):
                q = torch.zeros((1, query_tokens, args.heads, 128), device='cuda', dtype=torch.bfloat16)
                kv = torch.zeros((1, key_tokens, args.heads, 128), device='cuda', dtype=torch.bfloat16)
                attention_ops.sageattn(q, kv, kv, tensor_layout='NHD', is_causal=False)
        torch.cuda.synchronize()
        cache = args.output / 'triton_cache'
        cache.mkdir(exist_ok=True)
        for source in Path(temporary).iterdir():
            if source.is_dir():
                shutil.copytree(source, cache / source.name, dirs_exist_ok=True)

    manifest = dict(source_hash=source_hash(), versions=versions(), arch=list(arch), heads=args.heads,
                    tau=args.tau, thresh_type=args.thresh_type, capacities=capacities, kernels=kernels, symbols=symbols)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Exported {args.output}', flush=True)


if __name__ == '__main__':
    main()
