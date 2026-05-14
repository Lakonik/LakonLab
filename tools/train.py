# Modified from https://github.com/open-mmlab/mmgeneration

import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

import argparse
import copy
import multiprocessing as mp
import os
import os.path as osp
import platform
import time
import re
from copy import deepcopy

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import cv2
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

import lakonlab.evaluation  # noqa: F401
from lakonlab.models import build_model
from lakonlab.datasets import build_dataset, build_dataloader, unique_dataloaders
from lakonlab.apis import train_model
from lakonlab.runner.checkpoint import clear_checkpoint_cache
from lakonlab.runner.utils import set_random_seed
from lakonlab.utils import get_root_logger, collect_env
from lakonlab.parallel.utils import prepare_module_wrapper
from lakonlab import __version__

cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi', 'flyte'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher != 'flyte':  # flyte launcher already sets up the distributed environment
            init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    try:
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
    except Exception as e:
        logger.warning(f'Error in collecting environment info: {e}')
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)
    cfg.seed = args.seed
    cfg.diff_seed = args.diff_seed
    cfg.deterministic = args.deterministic
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # build datasets and dataloaders before building the model
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.val.pipeline
        datasets.append(build_dataset(val_dataset))

    # default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        seed=cfg.seed)

    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items()
        if k not in [
            'train', 'train_dataloader', 'val_dataloader', 'test_dataloader'
        ] and not re.fullmatch(r'(val|test)\d*', k)
    })

    # The specific datalaoder settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', dict())}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in datasets]

    if not args.no_validate and cfg.get('evaluation', None) is not None:
        assert isinstance(cfg.evaluation, list)
        _evaluation = []
        eval_dataloaders = dict()
        for eval_cfg_ in cfg.evaluation:
            if eval_cfg_.data in eval_dataloaders:
                val_dataloader = eval_dataloaders[eval_cfg_.data]
            else:
                val_dataset = build_dataset(cfg.data[eval_cfg_.data])
                val_loader_cfg = {
                    **loader_cfg, 'shuffle': False,
                    **cfg.data.get('val_dataloader', dict())
                }
                val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
                eval_dataloaders[eval_cfg_.data] = val_dataloader
            eval_cfg = deepcopy(eval_cfg_)
            eval_cfg.update(dict(dist=distributed, dataloader=val_dataloader))
            _evaluation.append(eval_cfg)
        cfg.evaluation = _evaluation

    # warm up dataloader workers
    for data_loader in unique_dataloaders(
            data_loaders + [eval_cfg.dataloader for eval_cfg in cfg.evaluation]):
        if (getattr(data_loader, 'num_workers', 0) > 0
                and getattr(data_loader, 'persistent_workers', False)):
            _ = iter(data_loader)  # spawns workers early, no data consumed

    # build the model after building the dataloaders
    if distributed:
        prepare_module_wrapper(cfg)

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    clear_checkpoint_cache()

    if cfg.checkpoint_config is not None:
        # save lakonlab version, config file content and class names in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(lakonlab_version=__version__ + get_git_hash()[:7])

    train_model(
        model,
        data_loaders,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
