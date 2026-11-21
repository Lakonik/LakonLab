import copy
import os.path as osp
import warnings
from typing import Optional
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import allreduce_params, get_dist_info
from lakonlab.parallel import FSDP2Wrapper, FSDPWrapper
from lakonlab.utils.io_utils import FileClient


@HOOKS.register_module(force=True)
class CheckpointHook(Hook):
    """mmcv's CheckpointHook with `must_save_interval`, checkpoints that outlive
    `max_keep_ckpts`, asynchronous saving, FSDP support, and lakonlab's `FileClient`.
    """

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 out_dir: Optional[str] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 sync_buffer: bool = False,
                 file_client_args: Optional[dict] = None,
                 must_save_interval: int = 1000000000,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args
        self.must_save_interval = must_save_interval

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args, self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir} by '
                           f'{self.file_client.name}.')

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    'create_symlink is set as True by the user but is changed'
                    'to be False because creating symbolic link is not '
                    f'allowed in {self.file_client.name}')
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.every_n_epochs(runner, self.interval) \
                or self.every_n_epochs(runner, self.must_save_interval) \
                or (self.save_last and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner, asynchronous=not self.is_last_epoch(runner))

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.every_n_iters(runner, self.interval) \
                or self.every_n_iters(runner, self.must_save_interval) \
                or (self.save_last and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner, asynchronous=not self.is_last_iter(runner))

    def _save_checkpoint(self, runner, asynchronous=False):
        rank, _ = get_dist_info()
        # in FSDP we need to call save_checkpoint on all ranks
        if rank == 0 or isinstance(runner.model, (FSDPWrapper, FSDP2Wrapper)):
            if self.max_keep_ckpts > 0:
                # remove other checkpoints
                cur_progress = copy.deepcopy(runner.epoch) if self.by_epoch else copy.deepcopy(runner.iter)

                def after_save_hook():
                    if self.by_epoch:
                        name = 'epoch_{}.pth'
                        current_ckpt = cur_progress + 1
                    else:
                        name = 'iter_{}.pth'
                        current_ckpt = cur_progress + 1
                    redundant_ckpts = [i for i in range(
                        current_ckpt - self.max_keep_ckpts * self.interval, 0,
                        -self.interval) if i % self.must_save_interval != 0]
                    filename_tmpl = self.args.get('filename_tmpl', name)
                    for _step in redundant_ckpts:
                        ckpt_path = self.file_client.join_path(
                            self.out_dir, filename_tmpl.format(_step))
                        if self.file_client.isfile(ckpt_path):
                            self.file_client.remove(ckpt_path)
                        else:
                            break

            else:
                after_save_hook = None

            runner.save_checkpoint(
                self.out_dir,
                save_optimizer=self.save_optimizer,
                after_save_hook=after_save_hook,
                asynchronous=asynchronous,
                **self.args)

        if rank == 0:
            if runner.meta is not None:
                if self.by_epoch:
                    cur_ckpt_filename = self.args.get(
                        'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
                else:
                    cur_ckpt_filename = self.args.get(
                        'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
                runner.meta.setdefault('hook_msgs', dict())
                runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                    self.out_dir, cur_ckpt_filename)
