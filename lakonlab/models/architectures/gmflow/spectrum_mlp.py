import math

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, xavier_init

from ...builder import MODULES
from lakonlab.utils import get_root_logger


class _SpectrumMLP(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            base_size=(4, 32, 32),  # (c, h, w)
            layers=[64, 8]):
        super().__init__()
        assert len(base_size) == 3
        mlp = []
        in_chn = 2
        for i, out_chn in enumerate(layers):
            mlp.append(nn.Linear(in_chn, out_chn))
            mlp.append(nn.SiLU())
            in_chn = out_chn
        mlp.append(nn.Linear(in_chn, base_size[0] * base_size[1] * base_size[2]))
        self.mlp = nn.Sequential(*mlp)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        constant_init(self.mlp[-1], val=0)

    def forward(self, gaussian_output):

        ori_dtype = gaussian_output['mean'].dtype
        spectral_mlp_dtype = self.dtype

        output_stats = torch.stack(
            [gaussian_output['var'].mean(dim=(-3, -2, -1)),
             gaussian_output['mean'].var(dim=(-2, -1)).mean(dim=-1)],
            dim=-1).to(spectral_mlp_dtype)

        c, base_h, base_w = self.config.base_size
        h, w = gaussian_output['mean'].shape[-2:]
        batch_shape = output_stats.shape[:-1]
        bs = batch_shape.numel()

        output_stats = output_stats.reshape(bs, 2)
        spectrum = self.mlp(output_stats).reshape(bs, c, base_h, base_w)

        if h != base_h or w != base_w:
            assert h <= base_h and w <= base_w
            h1 = (h + 1) // 2
            h2 = h - h1
            w1 = (w + 1) // 2
            w2 = w - w1
            spectrum = torch.cat(
                [torch.cat([spectrum[..., :h1, :w1], spectrum[..., :h1, -w2:]], dim=-1),
                 torch.cat([spectrum[..., -h2:, :w1], spectrum[..., -h2:, -w2:]], dim=-1)], dim=-2)

        log_var = spectrum.flatten(-2).log_softmax(dim=-1) + math.log(h * w)
        log_var = log_var.reshape(*batch_shape, c, h, w)

        return log_var.to(ori_dtype)


@MODULES.register_module()
class SpectrumMLP(_SpectrumMLP):

    def __init__(
            self,
            *args,
            pretrained=None,
            torch_dtype='float32',
            **kwargs):

        super().__init__(*args, **kwargs)

        self.init_weights(pretrained)
        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
