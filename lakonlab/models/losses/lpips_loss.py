import torch
import torch.nn as nn
import lpips
from ..builder import MODULES
from .utils import weighted_loss


@weighted_loss
def lpips_loss(pred, target, lpips_fn):
    return lpips_fn(pred, target)   # (bs, 1, h, w) if spatial else (bs, 1, 1, 1)


# Global caches for model loading
_lpips_cache = {}


def load_lpips(torch_dtype, device):
    cache_key = f'{torch_dtype}_{device}'
    if cache_key in _lpips_cache:
        return _lpips_cache[cache_key]

    model = lpips.LPIPS(net='vgg', eval_mode=True, pnet_tune=False)
    # pnet_tune only freezes the trunk; the lin probes ship requires_grad=True, which
    # would retain activations for grads no optimizer owns
    model.requires_grad_(False)
    model = model.to(dtype=getattr(torch, torch_dtype), device=device)
    _lpips_cache[cache_key] = model
    return model


@MODULES.register_module()
class LPIPSLoss(nn.Module):
    """LPIPS for images (B, C, H, W) or videos (B, C, T, H, W).

    Video frames and optional weights (B, 1, T, H, W) are flattened into
    the batch dimension before checkpoint chunking and loss reduction.
    """

    def __init__(self,
                 spatial=False,
                 torch_dtype='bfloat16',
                 device='cuda',
                 loss_weight=1.0,
                 reduction='mean',
                 checkpoint_chunk_size=None,
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='default',
                     fullgraph=True,
                     dynamic=False)):
        super().__init__()
        self.spatial = spatial
        self.lpips = [load_lpips(torch_dtype, device)]
        self.dtype = getattr(torch, torch_dtype)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self.lpips[0].forward, **compile_kwargs)

    def chunked_lpips_fn(self, pred, target):
        lpips_fn = self.lpips[0].forward if self._compiled_forward is None else self._compiled_forward
        cs = self.checkpoint_chunk_size
        if cs is None or not (torch.is_grad_enabled() and pred.requires_grad):
            return lpips_fn(pred, target)
        else:
            return torch.cat([
                torch.utils.checkpoint.checkpoint(
                    lpips_fn,
                    pred[i:i + cs],
                    target[i:i + cs],
                    use_reentrant=False)
                for i in range(0, pred.size(0), cs)])

    def forward(self, pred, target, weight=None, avg_factor=None):
        if pred.ndim == 5:
            b, c, t, h, w = pred.shape
            pred = pred.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            target = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            if weight is not None:
                weight = weight.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        dtype = pred.dtype
        self.lpips[0].spatial = self.spatial
        return lpips_loss(
            pred.to(self.dtype), target.to(self.dtype), lpips_fn=self.chunked_lpips_fn,
            weight=weight, reduction=self.reduction, avg_factor=avg_factor
        ).to(dtype) * self.loss_weight

    def _apply(self, *args, **kwargs):
        self.lpips[0]._apply(*args, **kwargs)
