import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

import os
import sys
import argparse
import multiprocessing as mp
import platform
import re
from copy import deepcopy

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import cv2
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.fileio import FileClient

from lakonlab.models import build_model
from lakonlab.utils import get_root_logger
from lakonlab.evaluation import build_metric
from lakonlab.evaluation.eval_hooks import evaluate
from lakonlab.datasets import build_dataset, build_dataloader, unique_dataloaders
from lakonlab.parallel.utils import prepare_module_wrapper, apply_module_wrapper
from lakonlab.runner.checkpoint import exists_ckpt, clear_checkpoint_cache
from lakonlab.runner.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test and eval a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--ckpt', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi', 'flyte'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed testing)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed testing)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
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
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='whether to skip evaluation when visualization HTML already exists')
    parser.add_argument(
        '--reuse-viz',
        action='store_true',
        help='whether to bypass generation and reuse existing visualization images for evaluation')
    parser.add_argument(
        '--timer',
        action='store_true',
        help='whether to enable timers')
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

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        if args.launcher != 'flyte':  # flyte launcher already sets up the distributed environment
            init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    log_path = None
    ckpt_name = None
    if args.ckpt is not None:
        dirname = os.path.dirname(args.ckpt)
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        log_name = ckpt_name + '_eval_log' + '.txt'
        if not args.ckpt.startswith(('http://', 'https://', 'huggingface://', 's3://')):
            log_path = os.path.join(dirname, log_name)
        else:
            os.makedirs(cfg.work_dir, exist_ok=True)
            log_path = os.path.join(cfg.work_dir, log_name)
    if log_path is None:
        os.makedirs(cfg.work_dir, exist_ok=True)
        log_path = os.path.join(cfg.work_dir, 'eval_log.txt')

    logger = get_root_logger(
        log_file=log_path, log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    # build datasets and dataloaders before building the model
    # The default loader config
    loader_cfg = dict(
        workers_per_gpu=cfg.data.get('val_workers_per_gpu', cfg.data.workers_per_gpu),
        num_gpus=len(cfg.gpu_ids),
        shuffle=False)

    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items()
        if k not in [
            'train', 'train_dataloader', 'val_dataloader', 'test_dataloader'
        ] and not re.fullmatch(r'(val|test)\d*', k)
    })

    # The specific datalaoder settings
    _evaluation = []
    eval_dataloaders = dict()
    for eval_cfg_ in cfg.evaluation:
        if args.data is not None:
            if eval_cfg_.data not in args.data:
                continue
        if eval_cfg_.data in eval_dataloaders:
            test_data_loader = eval_dataloaders[eval_cfg_.data]
        else:
            test_dataset = build_dataset(cfg.data[eval_cfg_.data])
            test_loader_cfg = {
                **loader_cfg,
                **cfg.data.get('test_dataloader', dict())
            }
            test_data_loader = build_dataloader(test_dataset, **test_loader_cfg)
            eval_dataloaders[eval_cfg_.data] = test_data_loader
        eval_cfg = deepcopy(eval_cfg_)
        eval_cfg.update(dataloader=test_data_loader)
        _evaluation.append(eval_cfg)
    cfg.evaluation = _evaluation

    # warm up dataloader workers
    for data_loader in unique_dataloaders(
            [eval_cfg.dataloader for eval_cfg in cfg.evaluation]):
        if (getattr(data_loader, 'num_workers', 0) > 0
                and getattr(data_loader, 'persistent_workers', False)):
            _ = iter(data_loader)  # spawns workers early, no data consumed

    # build the model and load checkpoint
    if distributed:
        prepare_module_wrapper(cfg)

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.requires_grad_(False).eval()
    clear_checkpoint_cache()

    if distributed:
        module_wrapper = cfg.get('module_wrapper', 'ddp')
        model = apply_module_wrapper(model, module_wrapper, cfg)
    else:
        model = MMDataParallel(model, device_ids=[0])

    if args.ckpt is not None:
        load_checkpoint(model, args.ckpt, map_location='cpu')
    elif exists_ckpt(cfg.get('resume_from', None)):
        load_checkpoint(model, cfg.resume_from, map_location='cpu')
    elif exists_ckpt(cfg.get('load_from', None)):
        load_checkpoint(model, cfg.load_from, map_location='cpu')

    for eval_cfg in cfg.evaluation:
        viz_dir = eval_cfg.get('viz_dir', None)
        if viz_dir is not None:
            if ckpt_name is not None:
                viz_dir = os.path.join(viz_dir, ckpt_name)  # add ckpt name
            else:
                viz_dir = os.path.join(viz_dir, 'default')
            file_client = FileClient.infer_client(uri=viz_dir)
            html_path = file_client.join_path(
                os.path.dirname(viz_dir), os.path.basename(viz_dir) + '.html')
            if args.skip_existing:
                # Keep distributed ranks on the same eval/skip branch.
                skip_existing = file_client.exists(html_path) if rank == 0 else False
                if distributed:
                    skip_existing_list = [skip_existing]
                    torch.distributed.broadcast_object_list(skip_existing_list, src=0)
                    skip_existing = skip_existing_list[0]
                if skip_existing:
                    if rank == 0:
                        logger.info(f'Skip existing evaluation: {html_path}')
                    continue

        metrics = eval_cfg['metrics']
        if isinstance(metrics, dict):
            metrics = [metrics]
        metrics = [build_metric(metric) for metric in metrics]
        for metric in metrics:
            metric.prepare()

        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}, '
                        f'use_rank_shift: {args.diff_seed}')
            set_random_seed(
                args.seed,
                deterministic=args.deterministic,
                use_rank_shift=args.diff_seed)

        log_vars = evaluate(
            model, eval_cfg.dataloader, metrics=metrics,
            feed_batch_size=eval_cfg.get('feed_batch_size', 32),
            viz_dir=viz_dir,
            viz_num=eval_cfg.get('viz_num', None),
            sample_kwargs=eval_cfg.get('sample_kwargs', dict()),
            enable_timers=args.timer,
            reuse_viz=args.reuse_viz)

        if rank == 0:
            sys.stdout.write('\n')
            prefix = eval_cfg.get('prefix', '')
            for metric in metrics:
                with torch.no_grad():
                    metric.summary()
                for name, val in metric._result_dict.items():
                    prefix_name = prefix + '_' + name if len(prefix) > 0 else name
                    mmcv.print_log(f'{eval_cfg.data}_{prefix_name} = {val}', 'lakonlab')
            for name, val in log_vars.items():
                prefix_name = prefix + '_' + name if len(prefix) > 0 else name
                mmcv.print_log(f'{eval_cfg.data}_{prefix_name} = {val}', 'lakonlab')

    return


if __name__ == '__main__':
    main()
