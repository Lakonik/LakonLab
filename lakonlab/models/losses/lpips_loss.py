import torch
import torch.nn as nn
import lpips
from ..builder import MODULES
from .utils import weighted_loss


@weighted_loss
def lpips_loss(pred, target, model):
    return model(pred, target)   # (bs, 1, h, w) if spatial else (bs, 1, 1, 1)


# Global caches for model loading
_lpips_cache = {}


def load_lpips(torch_dtype, device):
    cache_key = f'{torch_dtype}_{device}'
    if cache_key in _lpips_cache:
        return _lpips_cache[cache_key]

    model = lpips.LPIPS(net='vgg', eval_mode=True, pnet_tune=False)
    model = model.to(dtype=getattr(torch, torch_dtype), device=device)
    _lpips_cache[cache_key] = model
    return model


@MODULES.register_module()
class LPIPSLoss(nn.Module):

    def __init__(self,
                 spatial=True,
                 torch_dtype='bfloat16',
                 device='cuda',
                 loss_weight=1.0,
                 reduction='mean'):
        super().__init__()
        self.spatial = spatial
        self.lpips = [load_lpips(torch_dtype, device)]
        self.dtype = getattr(torch, torch_dtype)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None):
        dtype = pred.dtype
        model = self.lpips[0]
        model.spatial = self.spatial
        return lpips_loss(
            pred.to(self.dtype), target.to(self.dtype), model=model,
            weight=weight, reduction=self.reduction, avg_factor=avg_factor
        ).to(dtype) * self.loss_weight

    def _apply(self, *args, **kwargs):
        self.lpips[0]._apply(*args, **kwargs)
