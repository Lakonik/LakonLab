import os
import math
import itertools
import torch

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveStatsHook(Hook):

    def __init__(self,
                 save_stats_interval=-1):
        self.save_stats_interval = save_stats_interval

    def save_stats(self, named_params, work_dir, iterate, rank):
        if self.save_stats_interval > 0 and (iterate % self.save_stats_interval) == 0:

            grad_dir = os.path.join(work_dir, 'grad')
            os.makedirs(grad_dir, exist_ok=True)
            outfile_path = os.path.join(
                grad_dir, 'iter_{:06d}_{:d}.txt'.format(iterate, rank))
            outfile = open(outfile_path, 'w')
            outfile.write('\n{:>12} {:>12} {:>12}    {}\n'.format(
                'grad_rms', 'std', 'mean', 'name'))
            for name, param in named_params:
                if param.grad is not None:
                    grad_rms = param.grad.detach().square().mean().sqrt()
                else:
                    grad_rms = math.nan
                std, mean = torch.std_mean(param.detach().float())
                outfile.write('{:>12.6f} {:>12.6f} {:>12.6f}    {}\n'.format(
                    grad_rms, std, mean, name))
            outfile.close()

    def after_train_iter(self, runner):
        self.save_stats(itertools.chain(runner.model.named_parameters(), runner.model.named_buffers()),
                        runner.work_dir, runner.iter, runner.rank)
