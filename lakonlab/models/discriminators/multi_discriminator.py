# Copyright (c) 2026 Hansheng Chen

from copy import deepcopy

import torch.nn as nn

from ..builder import MODULES, build_module


@MODULES.register_module()
class MultiDiscriminator(nn.Module):
    """Route per-modality inputs through independent discriminator wrappers.

    Each fakes/reals input maps modality names to complete child inputs,
    passed through unchanged. Each child defines its own input structure.
    Child wrappers own adversarial and feature-matching losses; generator
    calls receive the corresponding real input as well as the fake input.
    """

    def __init__(self, modalities, train_cfg=None, test_cfg=None):
        super().__init__()
        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()
        train_cfg = self.train_cfg.copy()
        modality_train_cfg = train_cfg.pop('modalities', {})
        test_cfg = self.test_cfg.copy()
        modality_test_cfg = test_cfg.pop('modalities', {})
        self.discriminators = nn.ModuleDict({
            name: build_module(config, default_args=dict(
                train_cfg=dict(train_cfg, **modality_train_cfg.get(name, {})),
                test_cfg=dict(test_cfg, **modality_test_cfg.get(name, {}))))
            for name, config in modalities.items()})

    def forward(self, return_d_loss=False, return_g_loss=False, reals=None, fakes=None, **kwargs):
        assert not (return_d_loss and return_g_loss), 'Cannot return both d_loss and g_loss at the same time.'
        loss, log_vars, predictions = 0, {}, {}
        for name, modality_fakes in fakes.items():
            modality_output = self.discriminators[name](
                return_d_loss=return_d_loss, return_g_loss=return_g_loss,
                reals=None if reals is None else reals[name], fakes=modality_fakes, **kwargs)
            if return_d_loss or return_g_loss:
                modality_loss, modality_log_vars = modality_output
                loss = loss + modality_loss
                log_vars.update({f'{key}_{name}': value for key, value in modality_log_vars.items()})
            else:
                predictions[name] = modality_output
        return (loss, log_vars) if return_d_loss or return_g_loss else predictions
