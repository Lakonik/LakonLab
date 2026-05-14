import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from ..builder import MODULES


@MODULES.register_module()
class REPALoss(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=None,
                 cache_config=None,
                 loss_weight=0.5):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.cache_config = deepcopy(cache_config) if cache_config is not None else None
        self.loss_weight = loss_weight
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, pred_features, target_features):
        pred_features = self.projector(pred_features.to(self.dtype))
        pred_features = F.normalize(pred_features.float(), dim=-1)
        target_features = F.normalize(target_features.float(), dim=-1)
        if pred_features.shape[:2] != target_features.shape[:2]:
            raise ValueError(
                'Predicted and target feature tokens must match in batch and token dimensions, got '
                f'{pred_features.shape} vs {target_features.shape}.')
        return -self.loss_weight * (pred_features * target_features).sum(dim=-1).mean()
