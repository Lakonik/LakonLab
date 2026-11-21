# Copyright (c) 2026 Hansheng Chen
# Masking adapted from UC Merced Vision and Learning Lab (copyright 2018).
# Source: https://github.com/phoenix104104/fast_blind_video_consistency
# MIT license: https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/LICENSE
# Squared RGB warping error follows VideoGigaGAN, Eq. (3).

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info, load_checkpoint
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from lakonlab.utils.io_utils import download_from_url
from .builder import METRICS
from .metrics import Metric


_raft_cache = {}


def load_raft():
    if 'C_T_V1' in _raft_cache:
        return _raft_cache['C_T_V1']

    weights = Raft_Large_Weights.C_T_V1
    model = raft_large(weights=None)
    checkpoint = download_from_url(weights.url, hash_prefix='22a6c225')
    load_checkpoint(
        model, checkpoint, map_location='cpu', strict=True)
    model.requires_grad_(False)
    model.eval()
    _raft_cache['C_T_V1'] = model
    return model


def _flow_warp(image, flow):
    """Sample image at pixel coordinates displaced by flow (x, y)."""
    height, width = flow.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing='ij')
    x = x + flow[:, 0]
    y = y + flow[:, 1]
    grid = torch.stack(
        (2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1), dim=-1)
    return F.grid_sample(
        image, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _pair_warping_error(first, second, forward_flow, backward_flow):
    """Return masked squared RGB error and coverage for each [0, 1] frame pair.

    Mask thresholds follow Lai et al.'s released scorer:
    https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/utils.py
    RGB channels are summed per valid pixel, as in VideoGigaGAN, Eq. (3):
    https://videogigagan.github.io/assets/paper.pdf
    """
    warped_second = _flow_warp(second, forward_flow)
    warped_backward = _flow_warp(backward_flow, forward_flow)
    forward_magnitude = forward_flow.square().sum(dim=1)
    backward_magnitude = warped_backward.square().sum(dim=1)
    inconsistent = (forward_flow + warped_backward).square().sum(dim=1) > (
        0.01 * (forward_magnitude + backward_magnitude) + 0.5)
    dx = (forward_flow[:, :, :, :-1] - forward_flow[:, :, :, 1:]).square().sum(dim=1)
    dy = (forward_flow[:, :, :-1, :] - forward_flow[:, :, 1:, :]).square().sum(dim=1)
    gradient_magnitude = F.pad(dx, (0, 1)) + F.pad(dy, (0, 0, 0, 1))
    motion_boundary = gradient_magnitude > 0.01 * forward_magnitude + 0.002
    valid = ~inconsistent & ~motion_boundary
    pixel_error = (first - warped_second).square().sum(dim=1)
    # Like the original scorer, an entirely masked pair contributes zero error.
    # Report coverage as well so that zero is not mistaken for reliable alignment.
    error = (pixel_error * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    return torch.stack((error, valid.float().mean(dim=(1, 2))), dim=1)


@METRICS.register_module()
class WarpingError(Metric):
    """Reference-free temporal warping error; lower is better.

    Accepts RGB videos (B, C, T, H, W) in LakonLab's metric range [0, 1].
    Scores every adjacent pair at native resolution, averaging pairs within each
    video, then videos equally. Uses the original RAFT Things weights
    (torchvision C_T_V1), 20 refinement iterations, and FP32 on generated frames.
    RAFT receives [-1, 1]; squared RGB error uses [0, 1] and sums channels per
    valid pixel. Masking follows Lai et al.; the error follows VideoGigaGAN,
    Eq. (3). Flow is symmetrically replicate-padded to multiples of eight and
    cropped back before scoring; padded dimensions must be at least 128 pixels.

    Args:
        num_images (int | None): Number of videos to evaluate, or None for all.
    """
    name = 'WarpingError'

    def __init__(self, num_images=None):
        super().__init__(num_images)
        self.device = 'cpu'
        self.model = load_raft()

    def prepare(self):
        self.scores = []

    def _estimate_flow(self, first, second):
        height, width = first.shape[-2:]
        pad_height = -height % 8
        pad_width = -width % 8
        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
        first = F.pad(first, padding, mode='replicate')
        second = F.pad(second, padding, mode='replicate')
        try:
            flow = self.model(
                first, second, num_flow_updates=20)[-1]
        finally:
            # These tensors are not buffers, so .cpu() does not release them.
            self.model.corr_block.corr_pyramid.clear()
        return flow[:, :, padding[2]:padding[2] + height, padding[0]:padding[0] + width].contiguous()

    def _score_video(self, video):
        scores = []
        # Process one pair/direction at a time to bound RAFT correlation memory.
        first = video[:, 0].unsqueeze(0).to(
            device=self.device, dtype=torch.float32).clamp(0, 1) * 2 - 1
        for frame_index in range(1, video.shape[1]):
            second = video[:, frame_index].unsqueeze(0).to(
                device=self.device, dtype=torch.float32).clamp(0, 1) * 2 - 1
            forward_flow = self._estimate_flow(first, second)
            backward_flow = self._estimate_flow(second, first)
            scores.append(_pair_warping_error(
                first / 2 + 0.5, second / 2 + 0.5, forward_flow, backward_flow))
            first = second
        return torch.cat(scores, dim=0).mean(dim=0)

    @torch.no_grad()
    def feed_op(self, batch, mode):
        if (batch.ndim != 5 or batch.shape[1] != 3
                or batch.shape[2] < 2 or min(batch.shape[-2:]) < 2):
            raise ValueError(
                'WarpingError expects (B, 3, T, H, W) videos with T, H, W >= 2, '
                f'got {tuple(batch.shape)}.')
        with torch.autocast(device_type=self.device, enabled=False):
            scores = torch.stack([self._score_video(video) for video in batch])

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.empty_like(scores) for _ in range(ws)]
            dist.all_gather(placeholder, scores)
            scores = torch.stack(placeholder, dim=1).reshape(
                scores.size(0) * ws, *scores.shape[1:]
            )

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.scores.append(scores.float().cpu())

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            if isinstance(batch, dict):
                batch_size = len(list(batch.values())[0])
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = {k: v[:end] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                batch_to_feed = batch[:end]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        scores = torch.cat(self.scores, dim=0)
        if self.num_images is not None:
            assert scores.shape[0] >= self.num_images
            scores = scores[:self.num_images]
        error, valid_fraction = scores.mean(dim=0).tolist()
        self._result_dict = dict(
            warping_error=error, warping_valid_fraction=valid_fraction)
        self._result_str = f'WarpingError: {error:.6f} (Valid: {valid_fraction:.4f})'
        return error

    def clear_fake_data(self):
        self.scores = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()

    def load_to_gpu(self):
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = 'cuda'

    def offload_to_cpu(self):
        self.model.cpu()
        self.device = 'cpu'
