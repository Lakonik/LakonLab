from abc import abstractmethod
from copy import deepcopy
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import mmcv

from ..builder import MODULES
from .pixelwise_loss import gaussian_nll_loss, mse_loss, gaussian_mixture_nll_loss, _reduction_modes
from .utils import reduce_loss


class DiffusionLoss(nn.Module):

    def __init__(self,
                 rescale_mode=None,
                 rescale_cfg=None,
                 log_cfgs=None,
                 weight=None,
                 sampler=None,
                 reduction='mean',
                 loss_name=None):
        super().__init__()

        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.reduction = reduction
        self._loss_name = loss_name

        self.log_fn_list = []

        log_cfgs_ = deepcopy(log_cfgs)
        if log_cfgs_ is not None:
            if not isinstance(log_cfgs_, list):
                log_cfgs_ = [log_cfgs_]
            assert mmcv.is_list_of(log_cfgs_, dict)
            for log_cfg_ in log_cfgs_:
                log_type = log_cfg_.pop('type')
                log_collect_fn = f'{log_type}_log_collect'
                assert hasattr(self, log_collect_fn)
                log_collect_fn = getattr(self, log_collect_fn)

                log_cfg_.setdefault('prefix_name', 'loss')
                assert log_cfg_['prefix_name'].startswith('loss')
                log_cfg_.setdefault('reduction', reduction)

                self.log_fn_list.append(partial(log_collect_fn, **log_cfg_))
        self.log_vars = dict()

        # handle rescale mode
        if not rescale_mode:
            self.rescale_fn = lambda loss, t: loss
        else:
            rescale_fn_name = f'{rescale_mode}_rescale'
            assert hasattr(self, rescale_fn_name)
            if rescale_mode == 'timestep_weight':
                if sampler is not None and hasattr(sampler, 'weight'):
                    weight = sampler.weight
                else:
                    assert weight is not None and isinstance(
                        weight, torch.Tensor), (
                            '\'weight\' or a \'sampler\' contains weight '
                            'attribute is must be \'torch.Tensor\' for '
                            '\'timestep_weight\' rescale_mode.')

                mmcv.print_log(
                    'Apply \'timestep_weight\' rescale_mode for '
                    f'{self._loss_name}. Please make sure the passed weight '
                    'can be updated by external functions.', 'lakonlab')

                rescale_cfg = dict(weight=weight)
            self.rescale_fn = partial(
                getattr(self, rescale_fn_name), **rescale_cfg)

    @staticmethod
    def constant_rescale(loss, timesteps, scale):
        """Rescale losses at all timesteps with a constant factor.

        Args:
            loss (torch.Tensor): Losses to rescale.
            timesteps (torch.Tensor): Timesteps of each loss items.
            scale (int): Rescale factor.

        Returns:
            torch.Tensor: Rescaled losses.
        """

        return loss * scale

    @staticmethod
    def timestep_weight_rescale(loss, timesteps, weight, scale=1):
        """Rescale losses corresponding to timestep.

        Args:
            loss (torch.Tensor): Losses to rescale.
            timesteps (torch.Tensor): Timesteps of each loss items.
            weight (torch.Tensor): Weight corresponding to each timestep.
            scale (int): Rescale factor.

        Returns:
            torch.Tensor: Rescaled losses.
        """

        return loss * weight[timesteps] * scale

    @torch.no_grad()
    def collect_log(self, loss, timesteps):
        """Collect logs.

        Args:
            loss (torch.Tensor): Losses to collect.
            timesteps (torch.Tensor): Timesteps of each loss items.
        """
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self,
                             loss,
                             timesteps,
                             total_timesteps,
                             prefix_name,
                             reduction='mean'):
        """Collect loss logs by quartile timesteps.

        Args:
            loss (torch.Tensor): Loss value of each input. Each loss tensor
                should be shape as [bz, ]
            timesteps (torch.Tensor): Timesteps corresponding to each loss.
                Each loss tensor should be shape as [bz, ].
            total_timesteps (int): Total timesteps of diffusion process.
            prefix_name (str): Prefix want to show in logs.
            reduction (str, optional): Specifies the reduction to apply to the
                output losses. Defaults to `mean`.

        Returns:
            dict: Collected log variables.
        """
        quartile = (timesteps / total_timesteps * 4)
        quartile = quartile.type(torch.LongTensor)

        log_vars = dict()

        for idx in range(4):
            if not (quartile == idx).any():
                loss_quartile = torch.zeros((1, ))
            else:
                loss_quartile = reduce_loss(loss[quartile == idx], reduction)
            log_vars[f'{prefix_name}_quartile_{idx}'] = loss_quartile.item()

        return log_vars

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``mse_loss``.
        """
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        # update log_vars of this class
        self.collect_log(loss, timesteps=timesteps)

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)

    @abstractmethod
    def _forward_loss(self, output_dict):
        """Forward function for loss calculation. This method should be
        implemented by each subclasses.

        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """

        raise NotImplementedError(
            '\'self._forward_loss\' must be implemented.')

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODULES.register_module()
class DiffusionMSELoss(DiffusionLoss):
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 sampler=None,
                 weight=None,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_mse'):
        super().__init__(rescale_mode=rescale_mode,
                         rescale_cfg=rescale_cfg,
                         log_cfgs=log_cfgs,
                         weight=weight,
                         sampler=sampler,
                         reduction=reduction,
                         loss_name=loss_name)

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.loss_fn = partial(mse_loss, reduction='flatmean')

    def _forward_loss(self, outputs_dict):
        """Forward function for loss calculation.
        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict) * 0.5
        return loss


@MODULES.register_module()
class DiffusionNLLLoss(DiffusionLoss):
    _default_data_info = dict(pred='u_t_pred', target='u_t', logstd='logstd')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_nll'):
        super().__init__(
            rescale_mode=rescale_mode,
            rescale_cfg=rescale_cfg,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1, ), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1, ), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1, ), dtype=torch.long))

    @torch.no_grad()
    def collect_log(self, loss, var, timesteps):
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_v = [torch.zeros_like(var) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_v, var)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            var = torch.cat(placeholder_v, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, var, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self,
                             loss,
                             var,
                             timesteps,
                             total_timesteps,
                             prefix_name,
                             reduction='mean',
                             momentum=0.1):
        quartile = (timesteps / total_timesteps * 4)
        quartile = quartile.to(torch.long).clamp(min=0, max=3)

        log_vars = dict()

        for idx in range(4):
            quartile_mask = quartile == idx
            quartile_count = torch.count_nonzero(quartile_mask).reshape(1)
            if quartile_count > 0:
                loss_quartile = reduce_loss(loss[quartile_mask], reduction).reshape(1)
                var_quartile = reduce_loss(var[quartile_mask], reduction).reshape(1)

                cur_weight = 1 - torch.exp(-momentum * quartile_count)
                getattr(self, f'count_quartile_{idx}').add_(quartile_count)
                total_weight = 1 - torch.exp(-momentum * getattr(self, f'count_quartile_{idx}'))
                cur_weight /= total_weight.clamp(min=1e-4)
                getattr(self, f'loss_quartile_{idx}').mul_(1 - cur_weight).add_(loss_quartile * cur_weight)
                getattr(self, f'var_quartile_{idx}').mul_(1 - cur_weight).add_(var_quartile * cur_weight)

            log_vars[f'{prefix_name}_quartile_{idx}'] = getattr(self, f'loss_quartile_{idx}').item()
            log_vars[f'{prefix_name}_var_quartile_{idx}'] = getattr(self, f'var_quartile_{idx}').item()

        return log_vars

    def _forward_loss(self, outputs_dict):
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict)
        return loss

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        with torch.no_grad():
            var = torch.exp(output_dict['logstd'] * 2)  # (bs, *)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']]  # (bs, *)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class GMFlowNLLLoss(DiffusionNLLLoss):
    _default_data_info = dict(
        pred_means='means',
        target='u_t',
        pred_logstds='logstds',
        pred_logweights='logweights')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_nll'):
        super().__init__(
            rescale_mode=rescale_mode,
            rescale_cfg=rescale_cfg,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_mixture_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1,), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1,), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1,), dtype=torch.long))

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        with torch.no_grad():
            weights = output_dict['logweights'].exp()
            mean = (weights * output_dict['means']).sum(-4, keepdim=True)  # (bs, *, 1, c, h, w)
            var = (weights * ((output_dict['means'] - mean).square()
                              + (output_dict['logstds'] * 2).exp())).sum(-4)  # (bs, *, c, h, w)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']].unsqueeze(-3)  # (bs, *, 1, h, w)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)
