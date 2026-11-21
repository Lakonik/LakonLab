# Copyright (c) 2026 Hansheng Chen

import bisect
import math

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, xavier_init

from ..builder import MODULES, build_module
from .utils import fp32_module_forward


def _unpack_and_repatch(tokens, token_grid_size, src_patch_size, tgt_patch_size=1):
    bs, _, channels = tokens.shape
    output_channels = channels // math.prod(src_patch_size)
    src_h, src_w = src_patch_size[-2:]
    ratio_h = src_h // tgt_patch_size
    ratio_w = src_w // tgt_patch_size
    tgt_channels = output_channels * tgt_patch_size ** 2

    if len(token_grid_size) == 2:
        h, w = token_grid_size
        return tokens.reshape(
            bs, h, w, ratio_h, tgt_patch_size, ratio_w, tgt_patch_size, output_channels
        ).permute(
            0, 7, 4, 6, 1, 3, 2, 5
        ).reshape(
            bs, tgt_channels, h * ratio_h, w * ratio_w)
    else:
        t, h, w = token_grid_size
        src_t = src_patch_size[0]
        return tokens.reshape(
            bs, t, h, w, src_t, ratio_h, tgt_patch_size,
            ratio_w, tgt_patch_size, output_channels
        ).permute(
            0, 1, 4, 9, 6, 8, 2, 5, 3, 7
        ).reshape(
            bs, t * src_t, tgt_channels, h * ratio_h, w * ratio_w)


def _is_first_frame(frame_id, first_frame_period):
    """Token frames at multiples of ``first_frame_period`` are first frames; only frame 0 without a period."""
    return frame_id % first_frame_period == 0 if first_frame_period else frame_id == 0


def _first_frame_layout(frame_ids, first_frame_period, first_t, src_t):
    """Split token frames into first and remaining frames.

    Returns the indices of both groups into ``frame_ids``, and the pixel frame order that
    interleaves the concatenated [first, remaining] outputs of first_t and src_t frames per token frame.
    """
    first = [i for i, frame_id in enumerate(frame_ids) if _is_first_frame(frame_id, first_frame_period)]
    rest = [i for i in range(len(frame_ids)) if i not in first]
    order = []
    for i in range(len(frame_ids)):
        if i in first:
            order.extend(range(first.index(i) * first_t, (first.index(i) + 1) * first_t))
        else:
            start = len(first) * first_t + rest.index(i) * src_t
            order.extend(range(start, start + src_t))
    return first, rest, order


class ConvRefinerResidualBlock(nn.Module):
    """Residual block on (n, c, h, w) frames."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def init_weights(self):
        kaiming_init(self.conv1)
        kaiming_init(self.conv2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        y += x
        return y


@MODULES.register_module()
class ConvRefiner(nn.Module):
    """Feature-conditioned spatial refinement for a linear pixel head.

    With ``residual``, the refinement is added to the linear head's pixels,
    which are also embedded as input; otherwise it predicts pixels from the
    features alone. ``checkpoint_chunk_size`` sets the nominal number of images or video frames
    per checkpoint: whole clips when they fit, otherwise frame windows of one
    clip. Each frame is refined independently. The feature map is
    expanded inside each chunk, so only the low-rank token features are kept
    for backward. Recomputing chunks costs one extra conv pass. Unlike a frozen
    feature extractor, these weights train, so the gate cannot test the input's
    ``requires_grad``: the activations are retained for the parameter grads
    either way.
    """

    def __init__(
            self,
            in_channels,
            src_patch_size,
            first_frame_src_patch_size=None,
            first_frame_period=None,
            residual=True,
            pixel_channels=3,
            projection_channels=128,
            channels=32,
            num_res_blocks=2,
            patch_size=2,
            checkpoint_chunk_size=None):
        super().__init__()
        if isinstance(src_patch_size, int):
            src_patch_size = (src_patch_size,) * 2
        src_patch_size = (1,) * (3 - len(src_patch_size)) + tuple(src_patch_size)  # images are one-frame clips
        self.pixel_channels = pixel_channels
        self.patch_size = patch_size
        self.first_frame_period = first_frame_period
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.feature_expansion_shape = self._get_feature_expansion_shape(src_patch_size)
        if first_frame_src_patch_size is not None:
            self.first_frame_feature_expansion_shape = self._get_feature_expansion_shape(
                first_frame_src_patch_size)
        else:
            self.first_frame_feature_expansion_shape = None

        self.feature_scale = (in_channels / projection_channels) ** 0.5
        self.feature_down = nn.Linear(in_channels, projection_channels, bias=False)
        self.feature_expand = nn.Linear(
            projection_channels,
            channels * math.prod(self.feature_expansion_shape),
            bias=False)
        if self.first_frame_feature_expansion_shape is not None:
            self.first_frame_feature_expand = nn.Linear(
                projection_channels,
                channels * math.prod(self.first_frame_feature_expansion_shape),
                bias=False)
        else:
            self.first_frame_feature_expand = None

        packed_pixel_channels = pixel_channels * patch_size ** 2
        self.pixel_embed = nn.Conv2d(packed_pixel_channels, channels, kernel_size=1) if residual else None
        self.act = nn.SiLU()
        self.res_blocks = nn.ModuleList([
            ConvRefinerResidualBlock(channels)
            for _ in range(num_res_blocks)
        ])
        self.out = nn.Conv2d(channels, packed_pixel_channels, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def init_weights(self):
        self.feature_down.to_empty(device='cpu')
        nn.init.kaiming_uniform_(self.feature_down.weight, a=math.sqrt(5))
        feature_bound = 1 / math.sqrt(self.feature_down.in_features)
        self.feature_expand.to_empty(device='cpu')
        nn.init.uniform_(self.feature_expand.weight, -feature_bound, feature_bound)
        if self.first_frame_feature_expand is not None:
            self.first_frame_feature_expand.to_empty(device='cpu')
            nn.init.uniform_(
                self.first_frame_feature_expand.weight, -feature_bound, feature_bound)
        if self.pixel_embed is not None:
            xavier_init(self.pixel_embed.to_empty(device='cpu'), distribution='uniform')
        for block in self.res_blocks:
            block.to_empty(device='cpu')
            block.init_weights()
        # zero output: with a residual, the refinement starts as an exact identity on the linear head's pixels
        constant_init(self.out.to_empty(device='cpu'), val=0)

    def _get_feature_expansion_shape(self, src_patch_size):
        *temporal_expansion, src_h, src_w = src_patch_size
        if src_h % self.patch_size != 0 or src_w % self.patch_size != 0:
            raise ValueError(
                'src_patch_size spatial dimensions must be divisible by patch_size, '
                f'got {src_patch_size} and {self.patch_size}.')
        return *temporal_expansion, \
            src_h // self.patch_size, src_w // self.patch_size

    def _get_frame_starts(self, t):
        """First pixel frame of each of t token frames, plus the total."""
        src_t = self.feature_expansion_shape[0]
        if self.first_frame_feature_expand is None:
            first_t = src_t
        else:
            first_t = self.first_frame_feature_expansion_shape[0]
        frame_starts = [0]
        for i in range(t):
            frame_starts.append(frame_starts[-1] + (
                first_t if _is_first_frame(i, self.first_frame_period) else src_t))
        return frame_starts

    def _expand_features(self, features, frame_starts, start, end, h, w):
        """Expand (n, t, h * w, P) low-rank token features into the (n, end - start, c, H, W)
        feature map of pixel frames [start, end)."""
        tok_start = bisect.bisect_right(frame_starts, start) - 1
        tok_end = bisect.bisect_left(frame_starts, end)
        features = features[:, tok_start:tok_end]
        if self.first_frame_feature_expand is None:
            feature_map = _unpack_and_repatch(
                self.feature_expand(features.flatten(1, 2)), (features.size(1), h, w),
                self.feature_expansion_shape)
        else:
            first, rest, order = _first_frame_layout(
                range(tok_start, tok_end), self.first_frame_period,
                self.first_frame_feature_expansion_shape[0], self.feature_expansion_shape[0])
            # both expansions always run, even on zero token frames, so each gets grads on every chunk
            feature_map = torch.cat([
                _unpack_and_repatch(
                    self.first_frame_feature_expand(features[:, first].flatten(1, 2)), (len(first), h, w),
                    self.first_frame_feature_expansion_shape),
                _unpack_and_repatch(
                    self.feature_expand(features[:, rest].flatten(1, 2)), (len(rest), h, w),
                    self.feature_expansion_shape),
            ], dim=1)[:, order]
        offset = start - frame_starts[tok_start]
        return feature_map[:, offset:offset + end - start]

    def refine(self, packed_pixels, features, frame_starts, start, end, h, w):
        """Refine frames [start, end) of (n, num_frames, c, H, W) clips (None without a residual) given
        their (n, t, h * w, P) low-rank token features."""
        feature_map = self._expand_features(features, frame_starts, start, end, h, w)
        n, t = feature_map.shape[:2]
        if packed_pixels is None:
            x = feature_map.flatten(0, 1)
        else:
            packed_pixels = packed_pixels[:, start:end]
            x = self.pixel_embed(packed_pixels.flatten(0, 1))
            x += feature_map.flatten(0, 1)
        del feature_map
        x = self.act(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.out(x).unflatten(0, (n, t))
        return x if packed_pixels is None else packed_pixels + x

    def forward(self, pixels, features, token_grid_size):
        """Refine (bs, c, H, W) images or (bs, num_frames, c, H, W) videos of linear-head pixels
        packed at ``patch_size`` (None without a residual), conditioned on (bs, num_tokens, in_channels)
        features on a (h, w) or (t, h, w) token grid."""
        if len(token_grid_size) == 2:
            pixels = None if pixels is None else pixels.unsqueeze(1)
            return self.forward(pixels, features, (1, *token_grid_size)).squeeze(1)
        t, h, w = token_grid_size
        bs = features.size(0)
        frame_starts = self._get_frame_starts(t)
        num_frames = frame_starts[-1]
        features = (self.feature_down(features) * self.feature_scale).unflatten(1, (t, h * w))
        cs = self.checkpoint_chunk_size
        if cs is None or not torch.is_grad_enabled():
            pixels = self.refine(pixels, features, frame_starts, 0, num_frames, h, w)
        else:
            n = max(cs // num_frames, 1)
            pixels = torch.cat([
                torch.utils.checkpoint.checkpoint(
                    self.refine, None if pixels is None else pixels[i:i + n], features[i:i + n],
                    frame_starts, j, min(j + cs, num_frames), h, w, use_reentrant=False).flatten(0, 1)
                for i in range(0, bs, n) for j in range(0, num_frames, cs)
            ]).unflatten(0, (bs, num_frames))
        return self.pixel_shuffle(pixels)


class DWDHeadMixin:
    """Pixel heads for Decoding Without Decoder (DWD).

    Adds linear heads that expand the denoiser's post-norm, post-modulation
    final-layer features directly into pixels. The host supplies
    features from its native output norm or a separate DWD output norm.
    An optional refiner receives the linear-head pixels (None with
    ``linear_head=False``), the features and the token grid, expands the
    features itself, and adds its output to the linear-head pixels when they
    exist.

    The pixel geometry describes every token frame: with
    ``first_frame_pixel_patch_size``, token frames at multiples of
    ``first_frame_period`` (only frame 0 when None) cover their own (shorter)
    pixel patch and use the first-frame head, and all frames are interleaved
    into one full-length sequence.

    Host contract (generic across LakonLab denoisers):

    - The inner model calls ``init_dwd_head`` at construction and supports
      ``return_features=True``, returning the 2-tuple ``(native_output,
      features)`` with its diffusers-convention output completely unchanged.
    - The outer wrapper supports ``return_pixels=True``: it invokes
      ``forward_dwd_head`` on the returned features and returns
      ``(native_output, pixels)``. Per-token pixel outputs always store
      expansion dimensions before channels, independently of the host model's
      patch convention.
    - The host's own ``init_weights`` calls ``super().init_weights()`` before
      loading any checkpoint, which initializes whatever this mixin owns, and
      is a no-op for a host that never constructed a DWD head.
    """

    def init_dwd_head(
            self,
            inner_dim,
            pixel_patch_size,
            first_frame_pixel_patch_size=None,
            first_frame_period=None,
            pixel_channels=3,
            fp32_dwd_head=False,
            pixel_refiner=None,
            linear_head=True):
        if isinstance(pixel_patch_size, int):
            pixel_patch_size = (pixel_patch_size,) * 2
        self.pixel_channels = pixel_channels
        self.pixel_patch_size = tuple(pixel_patch_size)
        self.first_frame_period = first_frame_period
        assert linear_head or pixel_refiner is not None, 'linear_head=False requires a pixel_refiner.'
        if first_frame_pixel_patch_size is not None:
            first_frame_pixel_patch_size = tuple(first_frame_pixel_patch_size)
            assert len(self.pixel_patch_size) == 3, \
                'first_frame_pixel_patch_size is only supported for video.'
            assert first_frame_pixel_patch_size[1:] == self.pixel_patch_size[1:], \
                'first_frame_pixel_patch_size must match pixel_patch_size spatially.'
        self.first_frame_pixel_patch_size = first_frame_pixel_patch_size
        self.dwd_linear_head = None
        self.first_frame_dwd_linear_head = None
        if linear_head:
            self.dwd_linear_head = nn.Linear(inner_dim, pixel_channels * math.prod(self.pixel_patch_size))
            self.dwd_linear_head.lakonlab_no_muon = True
            if first_frame_pixel_patch_size is not None:
                self.first_frame_dwd_linear_head = nn.Linear(
                    inner_dim, pixel_channels * math.prod(first_frame_pixel_patch_size))
                self.first_frame_dwd_linear_head.lakonlab_no_muon = True
        self.fp32_dwd_head = fp32_dwd_head
        if pixel_refiner is None:
            self.dwd_pixel_refiner = None
        else:
            self.dwd_pixel_refiner = build_module(
                pixel_refiner,
                default_args=dict(
                    in_channels=inner_dim,
                    pixel_channels=self.pixel_channels,
                    src_patch_size=self.pixel_patch_size,
                    first_frame_src_patch_size=self.first_frame_pixel_patch_size,
                    first_frame_period=self.first_frame_period,
                    residual=linear_head))
        # backward compatibility for legacy keys
        self.register_load_state_dict_pre_hook(self._load_dwd_head_state_dict_pre_hook)

    def _load_dwd_head_state_dict_pre_hook(
            self, module, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        for old_key in list(state_dict):
            if old_key.startswith(prefix):
                new_key = old_key.replace(
                    'pixel_upsampler', 'dwd_pixel_refiner')
                new_key = new_key.replace(
                    'dwd_pixel_refiner.pixel_proj',
                    'dwd_pixel_refiner.pixel_embed')
                new_key = new_key.replace(
                    'first_frame_pixel_proj_out', 'first_frame_dwd_linear_head')
                new_key = new_key.replace('pixel_proj_out', 'dwd_linear_head')
                if new_key == old_key:
                    continue
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                state_dict.pop(old_key)

    def init_weights(self):
        if not hasattr(self, 'dwd_pixel_refiner'):  # no DWD head
            return
        for linear_head in (self.dwd_linear_head, self.first_frame_dwd_linear_head):
            if linear_head is not None:
                xavier_init(linear_head.to_empty(device='cpu'), distribution='uniform')
        if self.dwd_pixel_refiner is not None:
            self.dwd_pixel_refiner.init_weights()

    def _project_pixels(self, linear_head, features, token_grid_size, pixel_patch_size):
        pixels = fp32_module_forward(
            linear_head, features, enabled=self.fp32_dwd_head)
        tgt_patch_size = (
            self.dwd_pixel_refiner.patch_size if self.dwd_pixel_refiner is not None else 1)
        return _unpack_and_repatch(
            pixels, token_grid_size, pixel_patch_size, tgt_patch_size)

    def forward_dwd_head(self, features, token_grid_size):
        """Project final-layer features directly to pixel x_0.

        Args:
            features (torch.Tensor): (bs, num_tokens, inner_dim) final-layer tokens.
            token_grid_size (tuple[int]): Token grid size, (t, h, w) for video
                models or (h, w) for image models.

        Returns:
            torch.Tensor: Pixels, (bs, c, T, H, W) or (bs, c, H, W).
        """
        if self.dwd_linear_head is None:
            pixels = None
        elif self.first_frame_dwd_linear_head is not None:
            t, h, w = token_grid_size
            first, rest, order = _first_frame_layout(
                range(t), self.first_frame_period,
                self.first_frame_pixel_patch_size[0], self.pixel_patch_size[0])
            frames = features.unflatten(1, (t, h * w))
            pixels = torch.cat([
                self._project_pixels(
                    self.first_frame_dwd_linear_head,
                    frames[:, first].flatten(1, 2),
                    (len(first), h, w),
                    self.first_frame_pixel_patch_size),
                self._project_pixels(
                    self.dwd_linear_head,
                    frames[:, rest].flatten(1, 2),
                    (len(rest), h, w),
                    self.pixel_patch_size),
            ], dim=1)[:, order]
        else:
            pixels = self._project_pixels(
                self.dwd_linear_head, features,
                token_grid_size, self.pixel_patch_size)
        if self.dwd_pixel_refiner is not None:
            pixels = self.dwd_pixel_refiner(pixels, features, token_grid_size)
        if len(token_grid_size) == 3:
            pixels = pixels.transpose(1, 2)
        return pixels
