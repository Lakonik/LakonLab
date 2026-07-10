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
                 compile_forward=False,
                 compile_kwargs=dict(
                     mode='reduce-overhead',
                     fullgraph=True,
                     dynamic=False),
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
        self.resize_position_embeddings()
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()
        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def embed_dim(self):
        return self.model.config.hidden_size

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def resize_position_embeddings(self):
        position_embeddings = self.model.embeddings.position_embeddings
        num_positions = position_embeddings.shape[1] - 1
        new_size = self.image_size // self.model.config.patch_size
        if num_positions == new_size * new_size:
            return

        class_pos_embed = position_embeddings[:, :1]
        patch_pos_embed = position_embeddings[:, 1:]
        old_size = int(num_positions ** 0.5)
        dim = patch_pos_embed.shape[-1]
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = patch_pos_embed.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(),
            size=(new_size, new_size),
            mode='bicubic',
            antialias=True  # REPA uses antialiasing in timm.layers.pos_embed.resample_abs_pos_embed
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        self.model.embeddings.position_embeddings = nn.Parameter(
            torch.cat((class_pos_embed, patch_pos_embed), dim=1))

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

    def _forward_impl(self, images):
        pixel_values = self.preprocess(images).to(self.dtype)
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        hidden_states = outputs.last_hidden_state
        num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        return hidden_states[:, 1 + num_register_tokens:]

    def forward(self, images):
        if self._compiled_forward is not None:
            return self._compiled_forward(images).clone()
        return self._forward_impl(images)
