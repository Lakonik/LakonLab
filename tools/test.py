import os
import sys
import argparse

from lakonlab.utils.env import apply_default_env
apply_default_env()

import re
from copy import deepcopy

import torch

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from lakonlab.models import build_model
from lakonlab.utils import get_root_logger
from lakonlab.evaluation import build_metric
from lakonlab.evaluation.eval_hooks import evaluate
from lakonlab.datasets import build_dataset, build_dataloader, shared_parse_cache, unique_dataloaders
from lakonlab.parallel.utils import prepare_module_wrapper, apply_module_wrapper
from lakonlab.runner.checkpoint import exists_ckpt, clear_checkpoint_cache
from lakonlab.utils import setup_global_config, setup_multi_processes
from lakonlab.utils.io_utils import FileClient
from lakonlab.runner.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test and eval a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--ckpt', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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
    setup_global_config(dict(cfg.get('global_config', dict())))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()

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
    # datasets sharing metadata or datalists parse each file once; the cache is dropped before workers fork
    with shared_parse_cache():
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
            file_client = FileClient.infer_client(uri=viz_dir)
            if ckpt_name is not None:
                viz_dir = file_client.join_path(viz_dir, ckpt_name)  # add ckpt name
            else:
                viz_dir = file_client.join_path(viz_dir, 'default')
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
            fps=eval_cfg.get('fps', 16),
            sample_rate=eval_cfg.get('sample_rate', None),
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

    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
