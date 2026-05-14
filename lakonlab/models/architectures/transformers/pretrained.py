import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ...builder import MODULES
from lakonlab.utils.io_utils import hf_model_loader


@MODULES.register_module()
class PretrainedDinoV2(nn.Module):
    def __init__(self,
                 model_name_or_path='facebook/dinov2-base',
                 image_size=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = hf_model_loader(AutoModel, model_name_or_path, **kwargs)
        self.image_size = image_size
        self.freeze = freeze
        self.eval_mode = eval_mode
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def embed_dim(self):
        return self.model.config.hidden_size

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def preprocess(self, images):
        images = images.float()
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).clamp(min=0, max=1)
        images = (images - self.mean) / self.std
        return images

    def forward(self, images):
        pixel_values = self.preprocess(images).to(self.dtype)
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        hidden_states = outputs.last_hidden_state
        num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        return hidden_states[:, 1 + num_register_tokens:]
