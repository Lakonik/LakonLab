# Copyright (c) 2026 Hansheng Chen

from copy import deepcopy

import torch
from accelerate import init_empty_weights

from .builder import MODELS, build_module
from .base import BaseModel
from lakonlab.utils import clone_params, tie_untrained_submodules, untie_all_parameters


@MODELS.register_module()
class Diffusion2D(BaseModel):

    def __init__(self,
                 diffusion=dict(type='GMFlow'),
                 diffusion_use_ema=False,
                 tie_ema=True,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        diffusion = deepcopy(diffusion)
        diffusion.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            if inference_only:
                self.diffusion_ema = self.diffusion
            else:
                diffusion_ema = deepcopy(diffusion)
                if isinstance(diffusion_ema.get('denoising', None), dict):
                    diffusion_ema['denoising'].pop('pretrained', None)
                with init_empty_weights():
                    self.diffusion_ema = build_module(diffusion_ema)
                if tie_ema:
                    tie_untrained_submodules(self.diffusion_ema, self.diffusion)
                else:
                    untie_all_parameters(self.diffusion_ema, self.diffusion)
                clone_params(self.diffusion_ema, self.diffusion)

        self.train_cfg = dict() if train_cfg is None else deepcopy(train_cfg)
        self.test_cfg = dict() if test_cfg is None else deepcopy(test_cfg)

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs = data['x'].size(0)

        loss, log_vars = self.diffusion(
            data['x'].reshape(bs, 2, 1, 1),
            return_loss=True)

        loss.backward() if loss_scaler is None else loss_scaler.scale(loss).backward()

        return log_vars, bs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        bs = data['x'].size(0)
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            if 'noise' in data:
                noise = data['noise'].reshape(bs, 2, 1, 1)
            else:
                noise = torch.randn((bs, 2, 1, 1), device=data['x'].device)
            x_out = diffusion(
                noise=noise,
                test_cfg_override=test_cfg_override)

            return dict(num_samples=bs, pred_x=x_out)
