import re

import torch
import torch.nn as nn
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from diffusers.models.autoencoders.autoencoder_rae import RAEDecoder, ViTMAELayer

from ...builder import MODULES, build_module
from ..utils import flex_freeze
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger


def _convert_legacy_decoder_state_dict(state_dict):
    replacements = [
        ('.attention.attention.query.', '.attention.to_q.'),
        ('.attention.attention.key.', '.attention.to_k.'),
        ('.attention.attention.value.', '.attention.to_v.'),
        ('.attention.output.dense.', '.attention.to_out.0.'),
    ]
    converted = {}
    for k, v in state_dict.items():
        for old, new in replacements:
            if old in k:
                k = k.replace(old, new)
                break
        converted[k] = v
    return converted


@MODULES.register_module()
class RAEv2Decoder(RAEDecoder):
    """diffusers RAEDecoder decoding (B, C, H, W) latent maps to images.

    Takes latents normalized with the RAEv2 latent statistics when
    ``pretrained_latent_stat`` is given, and returns images in [-1, 1].
    Defaults reproduce the RAEv2 ImageNet-256 ViT-XL decoder; load the
    released weights with e.g.
    ``pretrained='huggingface://nyu-visionx/RAEv2-models/stage1/imagenet/dinov3l-k7/decoder.pt'``
    (legacy RAE-repo attention keys are converted automatically).
    """

    def __init__(
            self,
            hidden_size=1024,
            decoder_hidden_size=1152,
            decoder_num_hidden_layers=28,
            decoder_num_attention_heads=16,
            decoder_intermediate_size=4096,
            num_patches=256,
            patch_size=16,
            num_channels=3,
            image_size=256,
            qkv_bias=True,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            hidden_act='gelu',
            pretrained=None,
            pretrained_latent_stat=None,
            eps=1e-5,
            freeze=False,
            freeze_exclude=[],
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype=None,
            eval_mode=False,
            torch_dtype='float32',
            compile_forward=False,
            compile_kwargs=dict(
                mode='reduce-overhead',
                fullgraph=True,
                dynamic=False)):
        super(RAEDecoder, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_patches = num_patches
        self.eps = eps

        self.decoder_embed = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        grid_size = int(num_patches ** 0.5)
        # RAEv2 uses plain integer grid coordinates for positional embeddings,
        # so we set base_size=grid_size to avoid scaling
        pos_embed = get_2d_sincos_pos_embed(
            decoder_hidden_size, grid_size, cls_token=True, extra_tokens=1,
            base_size=grid_size, output_type='pt')
        # Register persistent buffer since RAEv2 learns positional embeddings
        self.register_buffer('decoder_pos_embed', pos_embed.unsqueeze(0).float(), persistent=True)

        self.decoder_layers = nn.ModuleList([
            ViTMAELayer(
                hidden_size=decoder_hidden_size,
                num_attention_heads=decoder_num_attention_heads,
                intermediate_size=decoder_intermediate_size,
                qkv_bias=qkv_bias,
                layer_norm_eps=layer_norm_eps,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_act=hidden_act,
            ) for _ in range(decoder_num_hidden_layers)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_hidden_size, eps=layer_norm_eps)
        self.decoder_pred = nn.Linear(decoder_hidden_size, patch_size ** 2 * num_channels, bias=True)
        self.gradient_checkpointing = False

        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

        self.do_normalization = pretrained_latent_stat is not None
        self.init_weights(pretrained, pretrained_latent_stat)

        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

        self.eval_mode = eval_mode
        if self.eval_mode:
            self.eval()

        self._compiled_forward = None
        if compile_forward:
            self._compiled_forward = torch.compile(self._forward_impl, **compile_kwargs)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def init_weights(self, pretrained=None, pretrained_latent_stat=None):
        logger = get_root_logger()
        if pretrained is not None:
            checkpoint = _load_cached_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {re.sub(r'^module\.', '', k): v for k, v in state_dict.items()}
            if any('.attention.attention.query.' in k for k in state_dict):
                state_dict = _convert_legacy_decoder_state_dict(state_dict)
            load_full_state_dict(self, state_dict, logger=logger)
        if pretrained_latent_stat is not None:
            stats = _load_cached_checkpoint(pretrained_latent_stat, map_location='cpu', logger=logger)
            self.register_buffer('latent_mean', stats['mean'].float(), persistent=False)
            self.register_buffer('latent_var', stats['var'].float(), persistent=False)

    def _forward_impl(self, hidden_states):
        if self.do_normalization:
            hidden_states = hidden_states * torch.sqrt(self.latent_var + self.eps) + self.latent_mean
        b, c, h, w = hidden_states.shape
        tokens = hidden_states.view(b, c, h * w).transpose(1, 2)
        logits = super().forward(tokens, return_dict=False)[0]
        decoded = self.unpatchify(logits).clamp(0, 1)
        return decoded * 2 - 1

    def decode(self, hidden_states):
        if self._compiled_forward is not None:
            return self._compiled_forward(hidden_states).clone()
        return self._forward_impl(hidden_states)

    def forward(self, hidden_states):
        return self.decode(hidden_states)


@MODULES.register_module()
class RAEv2(nn.Module):
    """RAEv2 autoencoder: a thin composite of a frozen pretrained encoder and
    an ``RAEv2Decoder``. Both submodules handle the latent normalization
    themselves (pass ``pretrained_latent_stat`` to each), so either can also
    be used standalone."""

    def __init__(self,
                 encoder=dict(type='PretrainedRAEv2Encoder'),
                 decoder=dict(type='RAEv2Decoder')):
        super().__init__()
        self.encoder = build_module(encoder) if encoder is not None else None
        self.decoder = build_module(decoder)

    def encode(self, img):
        return self.encoder(img)

    def decode(self, z):
        return self.decoder(z)
