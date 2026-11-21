import warnings
from typing import Dict, Union

import torch
import torch.nn as nn

from torch.nn import GroupNorm, LayerNorm

from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor as _DefaultOptimizerConstructor
from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg
from mmcv.utils.ext_loader import check_ops_exist


@OPTIMIZER_BUILDERS.register_module(force=True)
class DefaultOptimizerConstructor(_DefaultOptimizerConstructor):
    """MMCV default optimizer constructor with merged paramwise groups.

    Difference from MMCV: per-parameter lr / weight_decay rules are preserved,
    but params with identical effective optimizer options are merged into one
    PyTorch param group instead of one param group per parameter.
    """

    @staticmethod
    def _group_key(param_group: Dict) -> tuple:
        return tuple(sorted(
            (key, value) for key, value in param_group.items()
            if key != 'params'))

    def _add_param_group(self, params: Dict, param_group: Dict) -> None:
        # Difference from MMCV: params is a dict keyed by effective optimizer
        # options instead of MMCV's list[dict]. MMCV appends the fully computed
        # param_group directly; here matching option keys share one group.
        group_key = self._group_key(param_group)
        if group_key not in params:
            params[group_key] = dict(
                (key, value) for key, value in param_group.items()
                if key != 'params')
            params[group_key]['params'] = []
        params[group_key]['params'].extend(param_group['params'])

    def _is_in(self, param_group: Dict, params: Dict) -> bool:
        # Difference from MMCV: params is the same option-keyed dict used by
        # _add_param_group, not MMCV's list[dict], so duplicate detection scans
        # params.values().
        param = set(param_group['params'])
        param_set = set()
        for group in params.values():
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(
            self,
            params: Dict,
            module: nn.Module,
            prefix: str = '',
            is_dcn_module: Union[int, float, None] = None) -> None:
        """Add all parameters of module to the params dict.

        The parameters of the given module will be added to grouped param
        buckets, with specific rules defined by paramwise_cfg.

        Args:
            params (dict): A dict of param groups keyed by optimizer options,
                it will be modified in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                self._add_param_group(params, param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group[
                            'weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult

            self._add_param_group(params, param_group)

        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        # Difference from MMCV: params is an option-keyed dict while add_params
        # runs, then it is converted back to the standard list[dict] format
        # expected by PyTorch optimizers.
        params = {}
        self.add_params(params, model)
        optimizer_cfg['params'] = list(params.values())

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
