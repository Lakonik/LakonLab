# Copyright (c) 2026 Hansheng Chen

import torch
import torch.utils.checkpoint
from torch import nn

from ...builder import MODULES
from .model import (
    BLOCK, DIM, FRAMES_TO_TRIM, NORM_EPS, NUM_LAYERS, NUM_REGISTER_TOKENS, OUT_CH, PATCH, PATCH_T,
    T_SHIFT, WINDOW, SolH3Block, _global_rows, _padded, _window_rows)
from lakonlab.runner.checkpoint import _load_cached_checkpoint, load_full_state_dict
from lakonlab.utils import get_root_logger


@MODULES.register_module()
class SolH3DWDHead(nn.Module):
    """The last ``num_layers`` Sol-H3-VAE decoder layers, final norm and pixel projection.

    A full-rank linear expansion maps every token onto 16-pixel Sol tokens,
    one per latent frame, at any spatial source patch that is a multiple of
    16. The whole latent sequence is refined at once with native Sol
    attention: shifted 5-latent x 16x16-token windows plus per-block global
    tokens, which start as the pretrained registers and a zero class token.
    Latent frame 0 must start a native 5-latent block.

    Output is Sol's native ImageNet-normalized RGB, (B, F, 3, 16H, 16W),
    using separate one-frame and four-frame projections; map it to
    pixels with an affine RGB color codec. Use ``linear_head=False`` on the
    DWD head; Sol predicts pixels from features with its pretrained projection.
    """

    patch_size = 1

    def __init__(
            self,
            in_channels,
            src_patch_size,
            first_frame_src_patch_size,
            first_frame_period,
            residual=False,
            pixel_channels=OUT_CH,
            num_layers=8,
            pretrained=None,
            checkpointing=False):
        super().__init__()
        assert not residual, 'SolH3DWDHead requires linear_head=False on the DWD head.'
        _, src_h, src_w = src_patch_size
        assert tuple(src_patch_size) == (PATCH_T, src_h, src_w) and src_h % PATCH == 0 and src_w % PATCH == 0, \
            f'tokens must cover one latent frame and whole Sol tokens, got src_patch_size {src_patch_size}.'
        assert first_frame_src_patch_size is not None \
            and tuple(first_frame_src_patch_size) == (PATCH_T - FRAMES_TO_TRIM, src_h, src_w) \
            and first_frame_period == BLOCK and pixel_channels == OUT_CH, \
            'the pixel geometry must follow Sol\'s frame trimming at every 5-latent block.'
        self.expansion = (src_h // PATCH, src_w // PATCH)
        self.pretrained = pretrained
        self.checkpointing = checkpointing
        self.feature_expand = nn.Linear(in_channels, DIM * self.expansion[0] * self.expansion[1])
        self.register_tokens = nn.Parameter(torch.empty(1, NUM_REGISTER_TOKENS, DIM))
        self.blocks = nn.ModuleList([SolH3Block() for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(DIM, eps=NORM_EPS)
        self.proj_out = nn.Linear(DIM, OUT_CH * PATCH_T * PATCH * PATCH)
        self.first_frame_proj_out = nn.Linear(DIM, OUT_CH * (PATCH_T - FRAMES_TO_TRIM) * PATCH * PATCH)

    def init_weights(self):
        if self.pretrained is not None:
            self.feature_expand.to_empty(device='cpu').reset_parameters()
            logger = get_root_logger()
            checkpoint = _load_cached_checkpoint(self.pretrained, map_location='cpu', logger=logger)
            source_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            block_offset = NUM_LAYERS - len(self.blocks)
            state_dict = {
                k: source_state_dict[k] for k in self.state_dict()
                if not k.startswith(('blocks.', 'feature_expand.', 'first_frame_proj_out.'))}
            for key in self.first_frame_proj_out.state_dict():
                value = source_state_dict[f'proj_out.{key}']
                state_dict[f'first_frame_proj_out.{key}'] = value.view(
                    OUT_CH, PATCH_T, PATCH, PATCH, *value.shape[1:])[:, FRAMES_TO_TRIM:].reshape(
                        -1, *value.shape[1:])
            for index, block in enumerate(self.blocks):
                for key in block.state_dict():
                    state_dict[f'blocks.{index}.{key}'] = source_state_dict[
                        f'blocks.{block_offset + index}.{key}']
            load_full_state_dict(self, state_dict, logger=logger, assign=True)

    def forward(self, pixels, features, token_grid_size, global_tokens=None):
        """Decode (B, t * h * w, in_channels) features on a (t, h, w) token grid; pixels must be None.

        ``global_tokens`` (B, ceil(t / 5), 5, D) overrides the initial global tokens, e.g. with the native
        Sol activations.
        """
        B = features.size(0)
        t, h, w = token_grid_size
        e_h, e_w = self.expansion
        x = self.feature_expand(features.to(self.register_tokens.dtype))  # feature_expand may be quantized
        x = x.view(B, t, h, w, e_h, e_w, DIM).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, t, h * e_h, w * e_w, DIM)
        return self.decode(x, global_tokens)

    def decode(self, x, global_tokens=None):
        """Run the Sol layers on (B, T, H, W, D) activations; returns (B, F, 3, 16H, 16W)."""
        B, T, H, W, _ = x.shape
        if global_tokens is None:
            global_tokens = torch.cat([self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, DIM)], 1)
            global_tokens = global_tokens.unsqueeze(1).expand(B, (T + BLOCK - 1) // BLOCK, -1, -1).contiguous()
        g = global_tokens.to(x.dtype)
        for block, args in self.block_args(x.shape, g.shape, x.device, x.dtype):
            if self.checkpointing and torch.is_grad_enabled():
                x, g = torch.utils.checkpoint.checkpoint(block, x, g, *args, use_reentrant=False)
            else:
                x, g = block(x, g, *args)
        x = self.norm_out(x).to(self.proj_out.weight.dtype)
        num_chunks, tail = divmod(T, BLOCK)
        if num_chunks:
            out = self._project_pixels(x[:, :num_chunks * BLOCK].unflatten(1, (num_chunks, BLOCK))).flatten(1, 2)
        if tail:
            end = self._project_pixels(x[:, num_chunks * BLOCK:].unsqueeze(1)).flatten(1, 2)
            out = torch.cat((out, end), dim=1) if num_chunks else end
        return out

    def block_args(self, x_shape, g_shape, device, dtype=torch.bfloat16):
        """Yield each Sol layer and its arguments after (x, g), for (B, T, H, W, D) tokens and (B, nb, S, D) global
        tokens of these shapes and dtype; the attention indices are built once and shared by the layers."""
        B, T, H, W, _ = x_shape
        nb, S = g_shape[1:3]
        global_indices = {}
        window_indices = {}
        for index, block in enumerate(self.blocks, NUM_LAYERS - len(self.blocks)):
            padded = _padded(block.attn, device, dtype)
            if padded not in global_indices:
                global_indices[padded] = _global_rows(B, T, H, W, nb, S, device, padded=padded)
            shifted = index % 2 == 1
            if (padded, shifted) not in window_indices:
                s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
                window_indices[padded, shifted] = _window_rows(B, T, H, W, nb, S, s, st, device, padded=padded)
            yield block, (shifted, global_indices[padded], window_indices[padded, shifted])

    def _project_pixels(self, x):
        """Project chunks (B, n, t, H, W, D) into (B, n, F, 3, 16H, 16W)."""
        B, n, t, H, W, _ = x.shape
        first = self.first_frame_proj_out(x[:, :, :1])
        first = first.view(B, n, H, W, OUT_CH, PATCH_T - FRAMES_TO_TRIM, PATCH, PATCH).permute(
            0, 1, 5, 4, 2, 6, 3, 7).reshape(B, n, PATCH_T - FRAMES_TO_TRIM, OUT_CH, H * PATCH, W * PATCH)
        if t > 1:
            rest = self.proj_out(x[:, :, 1:])
            rest = rest.view(B, n, t - 1, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(
                0, 1, 2, 6, 5, 3, 7, 4, 8).reshape(B, n, (t - 1) * PATCH_T, OUT_CH, H * PATCH, W * PATCH)
            first = torch.cat((first, rest), dim=2)
        return first
