"""Offline artifacts and process-local execution for Sol attention."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

import torch


def source_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob('*.py'), root.parent / 'attention.py']):
        digest.update(str(path.relative_to(root.parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions():
    return dict(python=platform.python_version(), **{name: _version(name) for name in (
        'torch', 'triton', 'nvidia-cutlass-dsl', 'cuda-bindings', 'apache-tvm-ffi', 'sageattention')})


class SolAttentionAOT:
    """Exported CuTe libraries and precompiled Triton kernels for SM90/SM120 inference.

    Configuration is CPU-only. Loaded CUDA functions belong to the calling GPU process.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest['source_hash'] != source_hash() or self.manifest['versions'] != versions():
            raise ValueError('Sol AoT artifacts do not match the installed source and dependency versions.')
        self._context = None
        self._kernels = {}

        # Triton's cache groups store absolute paths to their compiled children.
        import triton
        cache = Path(triton.knobs.cache.dir)
        for source in (self.path / 'triton_cache').iterdir():
            target = cache / source.name
            target.mkdir(parents=True, exist_ok=True)
            for file in source.iterdir():
                if file.name.startswith('__grp__'):
                    data = json.loads(file.read_text())
                    data['child_paths'] = {name: str(target / name) for name in data['child_paths']}
                    (target / file.name).write_text(json.dumps(data))
                elif not (target / file.name).exists():
                    shutil.copyfile(file, target / file.name)
        from .preprocess import _reduce_kv_kernel
        _reduce_kv_kernel.cache_results = True

    def _check_context(self, device):
        context = (os.getpid(), device.index)
        if context != self._context:
            if list(torch.cuda.get_device_capability(device)) != self.manifest['arch']:
                raise ValueError('Sol AoT artifacts were built for a different GPU architecture.')
            self._kernels.clear()
            self._context = context

    def validate(self, q, capacity, tau, thresh_type):
        self._check_context(q.device)
        if (q.size(0), q.size(2), q.size(3), q.dtype) != (1, self.manifest['heads'], 128, torch.bfloat16):
            raise ValueError('Sol AoT artifacts require batch 1, BF16 and the exported head dimensions.')
        if (tau, thresh_type) != (self.manifest['tau'], self.manifest['thresh_type']):
            raise ValueError('Sol AoT artifacts were built with different routing settings.')
        if capacity not in self.manifest['capacities']:
            raise ValueError(f'Sol token capacity {capacity} was not exported in this bundle.')

    def kernel(self, device, splits):
        self._check_context(device)
        if splits not in self._kernels:
            if str(splits) not in self.manifest['kernels']:
                raise ValueError(f'Sol AoT artifacts contain KV splits {sorted(map(int, self.manifest["kernels"]))}, '
                                 f'not {splits}; export it or choose another --sol-kv-splits.')
            import cutlass.cute as cute
            module = cute.runtime.load_module(
                str(self.path / self.manifest['kernels'][str(splits)]), enable_tvm_ffi=True)
            function = getattr(module, self.manifest.get('symbols', {}).get(str(splits), f'sol_sm90_s{splits}'))

            def run(*args, stream=None):
                return function(*args, stream)

            self._kernels[splits] = (module, run)
        return self._kernels[splits][1]
