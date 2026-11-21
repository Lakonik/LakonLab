from typing import Optional

from mmcv.runner.hooks import HOOKS, LrUpdaterHook
from mmcv.runner.hooks.lr_updater import annealing_linear


@HOOKS.register_module()
class FlatLinearAnnealingLrUpdaterHook(LrUpdaterHook):
    """Flat + Linear annealing LR scheduler.

    Args:
        start_step (int): When to start annealing the learning rate.
        end_step (int): When to finish annealing the learning rate.
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
            self,
            start_step: int,
            end_step: int,
            min_lr: Optional[float] = None,
            min_lr_ratio: Optional[float] = None,
            **kwargs) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        if start_step < 0 or not isinstance(start_step, int):
            raise ValueError(
                'expected non-negative int start_step, but '
                f'got {start_step}')
        if end_step < start_step or not isinstance(end_step, int):
            raise ValueError(
                'expected int no less than start_step end_step, but '
                f'got {end_step}')
        self.start_step = start_step
        self.end_step = end_step
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        max_progress = max(self.end_step - self.start_step, 1)
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore

        if progress < self.start_step:
            return base_lr
        if progress >= self.end_step:
            return target_lr
        return annealing_linear(
            base_lr, target_lr, (progress - self.start_step) / max_progress)
