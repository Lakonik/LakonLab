# Copyright (c) 2026 Hansheng Chen

import sys
import os
import math
import numpy as np
import warnings
from bisect import bisect_right
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import torch
import mmcv
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmcv.fileio import FileClient
from . import build_metric
from lakonlab.models.architectures.utils import get_module_device
from lakonlab.runner.timer import default_timers
from lakonlab.utils import gc_context
from lakonlab.utils.io_utils import save_image, save_video, load_images_parallel
from lakonlab.ui.media_viewer import write_html

default_timers.add_timer('total time')


def flatten_list(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)  # recurse into sub-list
        else:
            yield item


def evaluate(model, dataloader, metrics=None,
             feed_batch_size=32, viz_dir=None, viz_num=None, sample_kwargs=dict(),
             fps=16, enable_timers=False, reuse_viz=False):
    has_metrics = metrics is not None and len(metrics) > 0
    if has_metrics:
        for metric in metrics:
            if hasattr(metric, 'load_to_gpu'):
                metric.load_to_gpu()

    if enable_timers:
        default_timers.enable_all()

    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_fakes = len(dataloader.dataset)

    if viz_dir is not None:
        html_entries = []
        saved_data_ids = set()
        file_client = FileClient.infer_client(uri=viz_dir)
        executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 4)
    else:
        html_entries = saved_data_ids = file_client = executor = None

    if rank == 0:
        mmcv.print_log(
            f'Generate {max_num_fakes} fake samples for evaluation', 'lakonlab')
        pbar = mmcv.ProgressBar(max_num_fakes)

    log_vars = dict()
    batch_size_list = []

    for i, data in enumerate(dataloader):
        can_reuse = False
        loaded_imgs_tensor = None
        loaded_html_entries = None

        if viz_dir is not None and reuse_viz:
            batch_names = list(flatten_list(data['name'].data))
            ids = list(flatten_list(data['ids'].data))

            reuse_candidates = []
            all_png_exist = True

            for name, data_id in zip(batch_names, ids):
                filename = f'{data_id:09d}.png'
                filepath = file_client.join_path(viz_dir, filename)
                if file_client.isfile(filepath):
                    reuse_candidates.append((data_id, name, filename, filepath))
                else:
                    all_png_exist = False
                    break

            if all_png_exist:
                filepaths = []
                loaded_html_entries = []
                for (data_id, name, filename, filepath) in reuse_candidates:
                    filepaths.append(filepath)
                    rel_filepath = file_client.join_path(os.path.basename(viz_dir), filename)
                    loaded_html_entries.append((data_id, rel_filepath, name))
                try:
                    loaded_imgs_tensor = torch.from_numpy(
                        np.stack(load_images_parallel(filepaths, file_client), axis=0)
                    ).permute(0, 3, 1, 2).float() / 255.0
                    can_reuse = True
                except Exception as e:
                    can_reuse = False
                    loaded_imgs_tensor = None
                    loaded_html_entries = None

        if can_reuse:
            outputs_dict = dict(
                pred_imgs=loaded_imgs_tensor,  # [0,1] float32
                num_samples=loaded_imgs_tensor.size(0))
            html_entries.extend(loaded_html_entries)

        else:
            # fall back to generation
            sample_kwargs_ = deepcopy(sample_kwargs)

            with default_timers['total time']:
                outputs_dict = model.val_step(data, show_pbar=rank == 0, **sample_kwargs_)

            if viz_dir is not None:
                batch_names = list(flatten_list(data['name'].data))
                for batch_id, data_id in enumerate(flatten_list(data['ids'].data)):
                    if (viz_num is not None and data_id >= viz_num) or (data_id in saved_data_ids):
                        continue
                    name = batch_names[batch_id]

                    image_viz = (outputs_dict['pred_imgs'][batch_id] * 255).round().to(torch.uint8)
                    if image_viz.dim() == 3:  # image
                        image_viz = image_viz.permute(1, 2, 0).cpu().numpy()
                        filename = f'{data_id:09d}.png'
                        executor.submit(
                            save_image,
                            image_viz, file_client.join_path(viz_dir, filename), file_client)
                    elif image_viz.dim() == 4:  # video
                        image_viz = image_viz.permute(1, 2, 3, 0).cpu().numpy()  # (t, h, w, c)
                        filename = f'{data_id:09d}.mp4'
                        executor.submit(
                            save_video,
                            image_viz, file_client.join_path(viz_dir, filename), file_client, fps)
                    else:
                        raise ValueError(f'Unsupported image dimension: {image_viz.dim()}')
                    rel_filepath = file_client.join_path(os.path.basename(viz_dir), filename)
                    html_entries.append((data_id, rel_filepath, name))
                    saved_data_ids.add(data_id)

        if 'log_vars' in outputs_dict:
            for k, v in outputs_dict['log_vars'].items():
                if k in log_vars:
                    log_vars[k].append(outputs_dict['log_vars'][k])
                else:
                    log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if has_metrics:
            pred_imgs = outputs_dict['pred_imgs'].split(feed_batch_size, dim=0)
            real_imgs = None
            if 'images' in data:
                real_imgs = data['images']
            elif 'target_imgs' in outputs_dict:
                real_imgs = outputs_dict['target_imgs']
            if real_imgs is not None:
                real_imgs = real_imgs.split(feed_batch_size, dim=0)
            requires_prompt = False
            for metric in metrics:
                requires_prompt |= getattr(metric, 'requires_prompt', False)
            if requires_prompt:
                prompts = list(flatten_list(data['name'].data))  # list of prompts
                prompts = [prompts[i:i + feed_batch_size] for i in range(0, len(prompts), feed_batch_size)]
            for metric in metrics:
                for batch_id, batch_imgs in enumerate(pred_imgs):
                    if getattr(metric, 'requires_prompt', False):
                        metric.feed(
                            dict(imgs=batch_imgs * 2 - 1, prompts=prompts[batch_id]), 'fakes')
                        if real_imgs is not None:
                            metric.feed(
                                dict(imgs=real_imgs[batch_id] * 2 - 1, prompts=prompts[batch_id]), 'reals')
                    else:
                        metric.feed(batch_imgs * 2 - 1, 'fakes')
                        if real_imgs is not None:
                            metric.feed(real_imgs[batch_id] * 2 - 1, 'reals')

        if rank == 0:
            pbar.update(total_batch_size)

    if ws > 1:
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        torch.distributed.all_reduce(batch_size_sum, op=torch.distributed.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            torch.distributed.all_reduce(weigted_values_sum, op=torch.distributed.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)

    if viz_dir is not None:
        if ws > 1:
            gathered = [None for _ in range(ws)]
            torch.distributed.all_gather_object(gathered, html_entries)
            if rank == 0:
                html_entries = [e for sub in gathered for e in (sub or [])]
        if rank == 0:
            unique_entries = dict()
            for entry in html_entries:
                data_id = entry[0]
                if data_id not in unique_entries:
                    unique_entries[data_id] = entry
            html_entries = list(unique_entries.values())
            html_entries.sort(key=lambda item: item[0])
            html_path = file_client.join_path(os.path.dirname(viz_dir), os.path.basename(viz_dir) + '.html')
            write_html(html_path, html_entries, file_client)
        executor.shutdown(wait=True)

    if enable_timers:
        default_timers.print_all()
        default_timers.reset_all()
        default_timers.disable_all()

    return log_vars


@HOOKS.register_module(force=True)
class GenerativeEvalHook(Hook):

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -math.inf, 'less': math.inf}
    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'is']
    less_keys = ['loss', 'fid', 'kid']
    _supported_best_metrics = ['fid', 'kid', 'is']

    def __init__(self,
                 dataloader,
                 interval=1,
                 dist=True,
                 metrics=None,
                 sample_kwargs=None,
                 save_best_ckpt=True,
                 best_metric='fid',
                 data='',
                 viz_dir=None,
                 feed_batch_size=32,
                 viz_num=None,
                 clear_reals=False,
                 prefix='',
                 metric_cpu_offload=False):
        if metrics is None:
            metrics = []

        self.dataloader = dataloader
        self.dist = dist
        self.sample_kwargs = sample_kwargs if sample_kwargs else dict()
        self.save_best_ckpt = save_best_ckpt
        self.best_metric = best_metric

        if isinstance(interval, int):
            self.interval = interval
        elif isinstance(interval, dict):
            if 'milestones' not in interval or 'interval' not in interval:
                raise KeyError(
                    '`milestones` and `interval` must exist in interval dict '
                    'if you want to use the dynamic interval evaluation '
                    f'strategy. But receive [{[k for k in interval.keys()]}] '
                    'in the interval dict.')

            self.milestones = interval['milestones']
            self.interval = interval['interval']
            # check if length of interval match with the milestones
            if len(self.interval) != len(self.milestones) + 1:
                raise ValueError(
                    f'Length of `interval`(={len(self.interval)}) cannot '
                    f'match length of `milestones`(={len(self.milestones)}).')

            # check if milestones is in order
            for idx in range(len(self.milestones) - 1):
                former, latter = self.milestones[idx], self.milestones[idx + 1]
                if former >= latter:
                    raise ValueError(
                        'Elements in `milestones` should in ascending order.')
        else:
            raise TypeError('`interval` only support `int` or `dict`,'
                            f'recieve {type(self.interval)} instead.')

        if isinstance(best_metric, str):
            self.best_metric = [self.best_metric]

        if self.save_best_ckpt:
            not_supported = set(self.best_metric) - set(
                self._supported_best_metrics)
            assert len(not_supported) == 0, (
                f'{not_supported} is not supported for saving best ckpt')

        self.metrics = build_metric(metrics)

        if isinstance(metrics, dict):
            self.metrics = [self.metrics]

        for metric in self.metrics:
            metric.prepare()

        # add support for saving best ckpt
        if self.save_best_ckpt:
            self.rule = {}
            self.compare_func = {}
            self._curr_best_score = {}
            self._curr_best_ckpt_path = {}
            for name in self.best_metric:
                if name in self.greater_keys:
                    self.rule[name] = 'greater'
                else:
                    self.rule[name] = 'less'
                self.compare_func[name] = self.rule_map[self.rule[name]]
                self._curr_best_score[name] = self.init_value_map[
                    self.rule[name]]
                self._curr_best_ckpt_path[name] = None

        self.data = data
        self.viz_dir = viz_dir
        self.file_client = FileClient.infer_client(
            uri=viz_dir) if viz_dir is not None else None
        self.feed_batch_size = feed_batch_size
        self.viz_num = viz_num
        self.clear_reals = clear_reals
        self.prefix = prefix
        self.metric_cpu_offload = metric_cpu_offload

    def get_current_interval(self, runner):
        """Get current evaluation interval.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if isinstance(self.interval, int):
            return self.interval
        else:
            curr_iter = runner.iter + 1
            index = bisect_right(self.milestones, curr_iter)
            return self.interval[index]

    def before_run(self, runner):
        """The behavior before running.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if self.save_best_ckpt is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating an empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())

    @torch.no_grad()
    def after_train_iter(self, runner):
        with gc_context(enable=True):
            interval = self.get_current_interval(runner)
            if not self.every_n_iters(runner, interval):
                return

            torch.cuda.empty_cache()

            runner.model.eval()
            rank, ws = get_dist_info()

            if self.viz_dir is not None:
                viz_dir = self.file_client.join_path(self.viz_dir, str(runner.iter + 1))
                if rank == 0:
                    if self.file_client.exists(viz_dir):
                        for name in self.file_client.list_dir_or_file(viz_dir):
                            self.file_client.remove(self.file_client.join_path(viz_dir, name))
                if ws > 1:
                    torch.distributed.barrier()
            else:
                viz_dir = None
            log_vars = evaluate(
                runner.model, self.dataloader, self.metrics, self.feed_batch_size,
                viz_dir, self.viz_num, self.sample_kwargs)

            if len(runner.log_buffer.output) == 0:
                runner.log_buffer.clear()

            # a dirty walkround to change the line at the end of pbar
            if rank == 0:
                sys.stdout.write('\n')
                for metric in self.metrics:
                    metric.summary()
                    for name, val in metric._result_dict.items():
                        prefix_name = self.prefix + '_' + name if len(self.prefix) > 0 else name
                        runner.log_buffer.output[self.data + '_' + prefix_name] = val
                        # record best metric and save the best ckpt
                        if self.save_best_ckpt and name in self.best_metric:
                            self._save_best_ckpt(runner, val, name)
                for name, val in log_vars.items():
                    prefix_name = self.prefix + '_' + name if len(self.prefix) > 0 else name
                    # print(self.data + '_' + prefix_name + ' = {}'.format(val))
                    runner.log_buffer.output[self.data + '_' + prefix_name] = val
                    # record best metric and save the best ckpt
                    if self.save_best_ckpt and name in self.best_metric:
                        self._save_best_ckpt(runner, val, name)
                runner.log_buffer.ready = True

            runner.model.train()

            for metric in self.metrics:
                metric.clear(clear_reals=self.clear_reals)
                if self.metric_cpu_offload:
                    if hasattr(metric, 'offload_to_cpu'):
                        metric.offload_to_cpu()

            torch.cuda.empty_cache()

    def _save_best_ckpt(self, runner, new_score, metric_name):
        """Save checkpoint with best metric score.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            new_score (float): New metric score.
            metric_name (str): Name of metric.
        """
        curr_iter = f'iter_{runner.iter + 1}'

        if self.compare_func[metric_name](new_score,
                                          self._curr_best_score[metric_name]):
            best_ckpt_name = f'best_{metric_name}_{curr_iter}.pth'
            runner.meta['hook_msgs'][f'best_score_{metric_name}'] = new_score

            if self._curr_best_ckpt_path[metric_name] and os.path.isfile(
                    self._curr_best_ckpt_path[metric_name]):
                os.remove(self._curr_best_ckpt_path[metric_name])

            self._curr_best_ckpt_path[metric_name] = os.path.join(
                runner.work_dir, best_ckpt_name)
            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.meta['hook_msgs'][
                f'best_ckpt_{metric_name}'] = self._curr_best_ckpt_path[
                    metric_name]

            self._curr_best_score[metric_name] = new_score
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(f'Best {metric_name} is {new_score:0.4f} '
                               f'at {curr_iter}.')
