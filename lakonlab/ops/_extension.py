from functools import lru_cache
from importlib import import_module

from lakonlab.utils import get_root_logger


@lru_cache(None)
def get_backend(package, optional=False):
    name = f'{package}._C'
    try:
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise

        from torch.utils.cpp_extension import load

        config = import_module(f'{package}._build').get_build_config()
        return load(
            name='_' + package.replace('.', '_'),
            sources=config['sources'],
            extra_include_paths=config.get('include_dirs', []),
            extra_cflags=config['extra_compile_args']['cxx'],
            extra_cuda_cflags=config['extra_compile_args']['nvcc'])
    except Exception:
        if not optional:
            raise
        logger = get_root_logger()
        logger.warning(f'{name} could not load or build; using the fallback implementation.')
        logger.debug(f'{name} load/build failure', exc_info=True)
        return None
