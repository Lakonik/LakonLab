import os
from pathlib import Path


def get_build_config():
    src = Path(__file__).resolve().parent / 'src'
    return dict(
        sources=[str(src / name) for name in ('gmflow_ops.cu', 'bindings.cpp')],
        extra_compile_args=dict(
            cxx=['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],
            nvcc=[
                '-O3', '-std=c++17',
                '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
            ]))
