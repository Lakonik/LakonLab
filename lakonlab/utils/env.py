import importlib
import os

import torch
from mmcv.utils import digit_version

DEFAULT_ENV = {
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}

BASE_GLOBAL_CONFIG = {
    # dynamo: effectively unlimited recompiles (bucketed shapes), static
    # shapes by default
    'torch._dynamo.config.recompile_limit': 10000,
    'torch._dynamo.config.accumulated_recompile_limit': 10000,
    'torch._dynamo.config.cache_size_limit': 10000,
    'torch._dynamo.config.accumulated_cache_size_limit': 10000,
    'torch._dynamo.config.assume_static_by_default': True,
    'torch._dynamo.config.automatic_dynamic_shapes': True,
    'torch._dynamo.config.force_parameter_static_shapes': True,
    'torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint': False,
    # Optional checkpointed SDPA cache workaround (pytorch/pytorch#185562).
    # Disables CUDA-graph-safe RNG handling.
    # 'torch._functorch.config.graphsafe_rng_functionalization': False,
    # Stale reshape strides after padding cause silent corruption
    # (pytorch/pytorch#194281, fixed in 2.15 by #192575).
    'torch._inductor.config.comprehensive_padding': digit_version(torch.__version__)[:2] >= (2, 15),
    # Mix-order reductions drop writes to shared gradient-buffer views
    # (pytorch/pytorch#193061, fixed in 2.15 by #193103).
    'torch._inductor.config.triton.mix_order_reduction': digit_version(torch.__version__)[:2] >= (2, 15),
    # inductor: the slow cudagraph pool assertion snapshots storage liveness
    # before its internal gc.collect(), then compares against allocator state
    # after collection, which can report already-dead storages as phantom
    # leaks (observed on torch 2.11).
    'torch._inductor.config.triton.slow_path_cudagraph_asserts': False,
    # backends: TF32 matmuls/convs, cudnn autotuning
    'torch.backends.cuda.matmul.allow_tf32': True,
    'torch.backends.cudnn.allow_tf32': True,
    'torch.backends.cudnn.benchmark': True,
}

_MISSING = object()


def apply_default_env(overrides=None):
    env = dict(DEFAULT_ENV, **(overrides or {}))
    for key, value in env.items():
        os.environ.setdefault(key, value)


def setup_global_config(global_config=None):
    from .misc import rgetattr, rsetattr

    _config = BASE_GLOBAL_CONFIG.copy()
    _config.update(global_config or {})

    for key, value in _config.items():
        root, _, attr = key.partition('.')
        module = importlib.import_module(root)
        if rgetattr(module, attr, _MISSING) is _MISSING:
            raise KeyError(f'Unknown option: {key}')
        rsetattr(module, attr, value)


# Modified from https://github.com/open-mmlab/mmgeneration
def setup_multi_processes(cfg):
    import warnings
    import platform
    import multiprocessing as mp

    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method(mp_start_method)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


# Modified from https://github.com/open-mmlab/mmgeneration
def collect_env():
    """Collect the information of the running environments."""
    import os.path as osp
    import subprocess
    import sys
    from collections import defaultdict

    import mmcv
    import torch
    import torchvision
    from mmcv.utils import get_build_config, get_git_hash

    import lakonlab

    env_info = dict()
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        if mmcv.__version__ < '1.3.11':
            from mmcv.utils.parrots_wrapper import CUDA_HOME
        else:
            from mmcv.utils.parrots_wrapper import _get_cuda_home
            CUDA_HOME = _get_cuda_home()

        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for devname, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = devname

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    env_info['TorchVision'] = torchvision.__version__

    env_info['MMCV'] = mmcv.__version__
    env_info['LakonLab'] = f'{ lakonlab.__version__ }+{get_git_hash()[:7]}'
    try:
        from mmcv.ops import get_compiler_version, get_compiling_cuda_version
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV CUDA Compiler'] = get_compiling_cuda_version()
    except ImportError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV CUDA Compiler'] = 'n/a'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
