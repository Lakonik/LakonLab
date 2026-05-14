# Copyright (c) 2026 Hansheng Chen

import mmcv
import torch

from copy import deepcopy
try:
    from torch.distributed.fsdp import FSDPModule
except:
    FSDPModule = None
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from lakonlab.utils import rgetattr, rhasattr


def get_ori_key(key):
    ori_key = key.split('.')
    if ori_key[0].endswith('_ema'):
        ori_key[0] = ori_key[0][:-4]
    elif ori_key[0].endswith('_ema2'):
        ori_key[0] = ori_key[0][:-5]
    else:
        raise ValueError(
            f'Invalid module key {key}, it should be in the format of '
            '<module_name>_ema or <module_name>_ema2, but got {ori_key[0]}')
    ori_key = '.'.join(ori_key)
    return ori_key


@HOOKS.register_module(force=True)
class ExponentialMovingAverageHook(Hook):

    _registered_interp_funcs = ['lerp']
    _registered_momentum_updaters = ['rampup', 'fixed', 'karras']

    def __init__(self,
                 module_keys,
                 trainable_only=True,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0,
                 momentum_policy='fixed',
                 momentum_cfg=None):
        super().__init__()
        self.trainable_only = trainable_only
        # check args
        assert interp_mode in self._registered_interp_funcs, (
            'Supported '
            f'interpolation functions are {self._registered_interp_funcs}, '
            f'but got {interp_mode}')

        assert momentum_policy in self._registered_momentum_updaters, (
            'Supported momentum policy are'
            f'{self._registered_momentum_updaters},'
            f' but got {momentum_policy}')

        assert isinstance(module_keys, str) or mmcv.is_tuple_of(
            module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            module_name = k.split('.')[0]
            assert module_name.endswith('_ema') or module_name.endswith('_ema2')
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = getattr(self, interp_mode)

        self.momentum_cfg = dict() if momentum_cfg is None else deepcopy(
            momentum_cfg)
        self.momentum_policy = momentum_policy
        if momentum_policy != 'fixed':
            assert hasattr(
                self, momentum_policy
            ), f'Currently, we do not support {self.momentum_policy} for EMA.'
            self.momentum_updater = getattr(self, momentum_policy)

    @staticmethod
    def lerp(a, b, momentum=0.999, momentum_nontrainable=0., trainable=True):
        """Does a linear interpolation of two parameters/ buffers.

        Args:
            a (torch.Tensor): Interpolation start point, refer to orig state.
            b (torch.Tensor): Interpolation end point, refer to ema state.
            momentum (float, optional): The weight for the interpolation
                formula. Defaults to 0.999.
            momentum_nontrainable (float, optional): The weight for the
                interpolation formula used for nontrainable parameters.
                Defaults to 0..
            trainable (bool, optional): Whether input parameters is trainable.
                If set to False, momentum_nontrainable will be used.
                Defaults to True.

        Returns:
            torch.Tensor: Interpolation result.
        """
        m = momentum if trainable else momentum_nontrainable
        return a + (b - a) * m

    @staticmethod
    def rampup(runner, ema_kimg=10, ema_rampup=0.05, batch_size=4, eps=1e-8):
        """Ramp up ema momentum.

        Ref: https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/training/training_loop.py#L300-L308 # noqa

        Args:
            runner (_type_): _description_
            ema_kimg (int, optional): Half-life of the exponential moving
                average of generator weights. Defaults to 10.
            ema_rampup (float, optional): EMA ramp-up coefficient.If set to
                None, then rampup will be disabled. Defaults to 0.05.
            batch_size (int, optional): Total batch size for one training
                iteration. Defaults to 4.
            eps (float, optional): Epsiolon to avoid ``batch_size`` divided by
                zero. Defaults to 1e-8.

        Returns:
            dict: Updated momentum.
        """
        cur_nimg = (runner.iter + 1) * batch_size
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5**(batch_size / max(ema_nimg, eps))
        return dict(momentum=ema_beta)

    def karras(self, runner, gamma=7.0, max_momentum=1.0):
        t = max(runner.iter + 1 - self.start_iter, 1)
        ema_beta = min((1 - 1 / t) ** (gamma + 1), max_momentum)
        return dict(momentum=ema_beta)

    def every_n_iters(self, runner, n):
        if runner.iter < self.start_iter:
            return True
        return (runner.iter + 1 - self.start_iter) % n == 0 if n > 0 else False

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        with torch.no_grad():
            model = runner.model.module if is_module_wrapper(
                runner.model) else runner.model

            # update momentum
            _interp_cfg = deepcopy(self.interp_cfg)
            if self.momentum_policy != 'fixed':
                _updated_args = self.momentum_updater(runner, **self.momentum_cfg)
                _interp_cfg.update(_updated_args)

            for key in self.module_keys:
                net = rgetattr(model, get_ori_key(key))
                ema = rgetattr(model, key)
                if FSDPModule is not None and isinstance(net, FSDPModule):  # Root parameters in EMA are unsharded after inference
                    net_is_sharded = net._get_fsdp_state()._fsdp_param_group.is_sharded
                    ema_is_sharded = ema._get_fsdp_state()._fsdp_param_group.is_sharded
                    if net_is_sharded and not ema_is_sharded:
                        ema.reshard()

                for p_net, p_ema in zip(net.parameters(), ema.parameters()):
                    if self.trainable_only and not p_net.requires_grad:
                        continue
                    if runner.iter < self.start_iter:
                        p_ema.data.copy_(p_net.data)
                    else:
                        p_ema.data.copy_(self.interp_func(
                            p_net, p_ema, trainable=p_net.requires_grad, **_interp_cfg))

                for b_net, b_ema in zip(net.buffers(), ema.buffers()):
                    b_ema.data.copy_(b_net.data)

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not rhasattr(model, k):
                raise RuntimeError(
                    f'Cannot find {k} network for EMA hook.')
