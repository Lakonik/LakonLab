"""Set BUILD_EXT=quant,gmflow_ops to build CUDA extensions."""

import os

from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'lakonlab/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        list[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    ext_modules = []
    cmdclass = {}
    build_extensions = os.environ.get('BUILD_EXT', '')
    if build_extensions:
        import runpy
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        cmdclass['build_ext'] = BuildExtension
        for name in build_extensions.split(','):
            name = name.strip()
            if name not in ('quant', 'gmflow_ops'):
                raise ValueError(f'Unknown CUDA extension: {name}')
            config = runpy.run_path(f'lakonlab/ops/{name}/_build.py')['get_build_config']()
            config['sources'] = [os.path.relpath(source) for source in config['sources']]
            ext_modules.append(CUDAExtension(name=f'lakonlab.ops.{name}._C', **config))

    setup(
        name='lakonlab',
        version=get_version(),
        description='LakonLab',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Hansheng Chen',
        author_email='hanshengchen97@gmail.com',
        packages=find_packages(exclude=('configs', 'data', 'tools', 'demo', 'work_dirs')),
        license='Apache-2.0',
        include_package_data=True,
        package_data={
            'lakonlab.ops.gmflow_ops': ['src/*.cu', 'src/*.cpp', 'src/*.h'],
            'lakonlab.ops.quant': ['src/*.cu', 'src/*.cpp', 'src/*.h'],
            'lakonlab.ops.sol_attn': [
                'UPSTREAM.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE.sana',
                'sm100/LICENSE.flash-attention', 'sm120/LICENSE.cudnn'],
        },
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        install_requires=parse_requirements('requirements.txt'),
        extras_require={
            'sol': ['nvidia-cutlass-dsl>=4.5', 'cuda-bindings>=12.8', 'apache-tvm-ffi>=0.1.15'],
            'gmuon': [
                'gram-newton-schulz @ git+https://github.com/nanovisionx/gmuon.git@'
                '4863cc2446f09a54d937c579d4e775c125f5ab2e'],
        },
        zip_safe=False)
