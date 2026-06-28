from mmcv.runner.hooks.logger import LoggerHook


def _after_train_iter(self, runner) -> None:
    if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
        runner.log_buffer.average()
    elif not self.by_epoch and self.every_n_iters(runner, self.interval):
        runner.log_buffer.average()
    elif self.end_of_epoch(runner) and not self.ignore_last:
        runner.log_buffer.average()

    if runner.log_buffer.ready:
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear()


def _after_train_epoch(self, runner) -> None:
    if runner.log_buffer.ready:
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear()


LoggerHook.after_train_iter = _after_train_iter
LoggerHook.after_train_epoch = _after_train_epoch

from .text import TextLoggerHook

__all__ = ['LoggerHook', 'TextLoggerHook']
