import os
import os.path as osp
import argparse

from lakonlab.utils.env import apply_default_env
apply_default_env()

import time
import re
from copy import deepcopy

import torch

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

import lakonlab.evaluation  # noqa: F401
from lakonlab.models import build_model
from lakonlab.datasets import build_dataset, build_dataloader, shared_parse_cache, unique_dataloaders
from lakonlab.apis import train_model
from lakonlab.runner.checkpoint import clear_checkpoint_cache
from lakonlab.runner.utils import set_random_seed
from lakonlab.utils import get_root_logger, collect_env, setup_global_config, setup_multi_processes
from lakonlab.parallel.utils import prepare_module_wrapper
from lakonlab import __version__


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
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()

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
    # datasets sharing metadata or datalists parse each file once; the cache is dropped before workers fork
    with shared_parse_cache():
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.val.pipeline
            datasets.append(build_dataset(val_dataset))

        # default loader config
        loader_cfg = dict(
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

    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
