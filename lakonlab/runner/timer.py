# modified from
# https://github.com/tjiiv-cprg/EPro-PnP/blob/42412220b641aef9e8943ceba516b3175631d370/EPro-PnP-Det/epropnp_det/utils/timer.py

"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import numpy as np
import torch
import mmcv
from mmcv import Timer
from lakonlab.utils import reduce_mean, get_root_logger


class IterTimer:
    def __init__(self, name='time', sync=True, enabled=True):
        self.name = name
        self.times = []
        self.timer = Timer(start=False)
        self.sync = sync
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer.start()
        return self

    def __exit__(self, type, value, traceback):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer_record()
        self.timer._is_running = False

    def timer_start(self):
        self.timer.start()

    def timer_record(self):
        self.times.append(self.timer.since_last_check())

    def print_time(self):
        if not self.enabled:
            return
        logger = get_root_logger()
        avg_time = np.average(self.times)
        avg_time = reduce_mean(torch.tensor(avg_time).cuda()).item()
        mmcv.print_log(f'Average {self.name} = {avg_time:.4f}', logger=logger)

    def reset(self):
        self.times = []


class IterTimers(dict):
    def __init__(self, *args, **kwargs):
        super(IterTimers, self).__init__(*args, **kwargs)

    def disable_all(self):
        for timer in self.values():
            timer.enabled = False

    def enable_all(self):
        for timer in self.values():
            timer.enabled = True

    def add_timer(self, name='time', sync=True, enabled=False):
        self[name] = IterTimer(
            name, sync=sync, enabled=enabled)

    def print_all(self):
        for timer in self.values():
            timer.print_time()

    def reset_all(self):
        for timer in self.values():
            timer.reset()


default_timers = IterTimers()
