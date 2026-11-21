# Copyright (c) 2026 Hansheng Chen

import os.path as osp

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger import WandbLoggerHook as _WandbLoggerHook


@HOOKS.register_module(force=True)
class WandbLoggerHook(_WandbLoggerHook):
    """mmcv's WandbLoggerHook, with the config file uploaded to the run's Files tab."""

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        config_path = osp.join(self.wandb.run.dir, runner.meta['exp_name'])
        with open(config_path, 'w') as f:
            f.write(runner.meta['config'])
        self.wandb.save(config_path, policy='now')
