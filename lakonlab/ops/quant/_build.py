import os
from pathlib import Path


def get_build_config():
    src = Path(__file__).resolve().parent / 'src'
    cutlass = os.environ.get('CUTLASS_PATH')
    if not cutlass or not (Path(cutlass) / 'include/cutlass/cutlass.h').is_file():
        raise RuntimeError('Set CUTLASS_PATH to a CUTLASS source checkout.')
    cutlass = Path(cutlass).resolve()
    return dict(
        sources=[str(src / name) for name in ('int8_mm.cu', 'bindings.cpp')],
        include_dirs=[str(cutlass / 'include'), str(cutlass / 'tools/util/include')],
        extra_compile_args=dict(
            cxx=['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],
            nvcc=[
                '-O3', '-std=c++17', '--fmad=false', '--expt-relaxed-constexpr',
                '-gencode=arch=compute_90a,code=sm_90a',
                '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
            ]))
