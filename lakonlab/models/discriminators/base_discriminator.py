from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from ..builder import MODULES, build_module


@MODULES.register_module()
class BaseDiscriminator(nn.Module):
    """Discriminator wrapper providing GAN losses and input processing.

    Args:
        network (dict): Config of the discriminator network.
        gan_loss (dict): Config of the GAN loss.
        feature_matching_loss (dict | None): Optional config of the
            generator-side feature matching loss. The network must support
            return_features=True, returning (scores, features) instead of
            scores alone. When configured, forward requests this output for
            both prediction and loss calls.
        aug (dict | None): Optional config of a differentiable augmentation
            module (e.g. ``dict(type='DiffAug')``) intrinsic to discriminator
            prediction. It remains active in eval mode; callers requiring
            unaugmented predictions must configure ``aug=None``. Separate
            predictions receive independent random draws.
        crop_size (int | tuple[int, int] | None): Optional maximum (H, W) size
            of random crops taken before augmentation/resizing, clamped per
            axis to the input size. An int means a square size. For
            (B, C, T, H, W) videos the crop window is shared
            across frames. Like ``aug``, separate predictions receive
            independent random crops. None disables cropping.
        crop_margin (int): Pixels of slack the crops must leave along each axis,
            which is what lets them jitter. Per axis the count is the largest n
            with ``n * crop + crop_margin <= size``, so a bigger input is
            covered by more crops rather than by the same crops at a coarser
            spacing. Within an axis the n starts are placed uniformly over the
            valid non-overlapping configurations (sorted uniform starts plus
            cumulative crop offsets), and vertical rows draw their horizontal
            starts independently. The batch dim is expanded to ``B * n_h * n_w``.
        num_frames (int | None): Optional maximum length of random temporal
            crops of (B, C, T, H, W) videos, taken before the spatial crops.
            Crops span ``min(T, num_frames)`` frames, and per sample the count
            is the largest n that fits in T, placed like the spatial crops, so
            the batch dim is further expanded by n. None disables temporal
            cropping.
        resize_size (int | tuple[int, int] | None): Optional (H, W) size the
            (augmented) input is spatially resized to before the network, with
            plain bilinear interpolation (no antialias, following the
            StyleGAN-T/RAE discriminator recipe). An int means a square size.
            Supports (B, C, H, W) images and (B, C, T, H, W) videos (per-frame
            spatial resize). None disables resizing.
    """

    def __init__(
            self,
            network=None,
            gan_loss=None,
            feature_matching_loss=None,
            aug=None,
            crop_size=None,
            crop_margin=0,
            num_frames=None,
            resize_size=None,
            train_cfg=None,
            test_cfg=None):
        super().__init__()
        self.network = build_module(network)
        self.gan_loss = build_module(gan_loss)
        self.feature_matching_loss = build_module(feature_matching_loss) if feature_matching_loss is not None else None
        self.aug = build_module(aug) if aug is not None else None
        if crop_size is not None and not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.crop_margin = crop_margin
        self.num_frames = num_frames
        if resize_size is not None and not isinstance(resize_size, (tuple, list)):
            resize_size = (resize_size, resize_size)
        self.resize_size = tuple(resize_size) if resize_size is not None else None
        self.train_cfg = deepcopy(train_cfg) if train_cfg else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg else dict()

    def d_loss(self, disc_pred_fakes, disc_pred_reals):
        if self.feature_matching_loss is not None:
            disc_pred_fakes, _ = disc_pred_fakes
            disc_pred_reals, _ = disc_pred_reals

        if getattr(self.gan_loss, 'gan_type', None) == 'relativistic':
            loss_disc = self.gan_loss(
                disc_pred_fakes, target_is_real=False, is_disc=True, relativistic_input=disc_pred_reals)
            log_vars = dict(loss_disc=loss_disc.detach())

        else:
            loss_disc_real = self.gan_loss(disc_pred_reals, target_is_real=True, is_disc=True)
            loss_disc_fake = self.gan_loss(disc_pred_fakes, target_is_real=False, is_disc=True)
            loss_disc = loss_disc_real + loss_disc_fake
            log_vars = dict(
                loss_disc_real=loss_disc_real.detach(),
                loss_disc_fake=loss_disc_fake.detach(),
                loss_disc=loss_disc.detach())

        return loss_disc, log_vars

    def g_loss(self, disc_pred_fakes, disc_pred_reals):
        if self.feature_matching_loss is not None:
            disc_pred_fakes, features_fakes = disc_pred_fakes
            disc_pred_reals, features_reals = disc_pred_reals

        if getattr(self.gan_loss, 'gan_type', None) == 'relativistic':
            loss_gen = self.gan_loss(
                disc_pred_fakes, target_is_real=True, is_disc=False, relativistic_input=disc_pred_reals)
        else:
            loss_gen = self.gan_loss(disc_pred_fakes, target_is_real=True, is_disc=False)
        log_vars = dict(loss_gen=loss_gen.detach())
        loss = loss_gen

        if self.feature_matching_loss is not None:
            loss_gen_feature = self.feature_matching_loss(features_fakes, features_reals)
            log_vars.update(loss_gen_feature=loss_gen_feature.detach())
            loss = loss + loss_gen_feature

        return loss, log_vars

    @staticmethod
    def rand_nonoverlap_starts(n, size, crop_size):
        assert n * crop_size <= size, \
            f'Cannot place {n} non-overlapping crops of size {crop_size} in {size}.'
        return ((torch.sort(torch.rand(n)).values * (size - n * crop_size + 1)).long()
                + torch.arange(n) * crop_size).tolist()

    def random_temporal_crop(self, x):
        t = x.size(2)
        crop_t = min(t, self.num_frames)
        n_t = t // crop_t
        return torch.stack([
            sample[:, t0:t0 + crop_t]
            for sample in x for t0 in self.rand_nonoverlap_starts(n_t, t, crop_t)])

    def random_crop(self, x):
        h, w = x.shape[-2:]
        crop_h, crop_w = min(h, self.crop_size[0]), min(w, self.crop_size[1])
        # as many crops per axis as fit while leaving crop_margin pixels to jitter over
        n_h = max(1, (h - self.crop_margin) // crop_h)
        n_w = max(1, (w - self.crop_margin) // crop_w)
        crops = []
        for sample in x:
            for y0 in self.rand_nonoverlap_starts(n_h, h, crop_h):
                for x0 in self.rand_nonoverlap_starts(n_w, w, crop_w):
                    crops.append(sample[..., y0:y0 + crop_h, x0:x0 + crop_w])
        return torch.stack(crops)

    def pred(self, x, *args, **kwargs):
        if self.num_frames is not None and x.dim() == 5:
            x = self.random_temporal_crop(x)

        if self.crop_size is not None:
            x = self.random_crop(x)

        if self.aug is not None:
            x = self.aug(x)

        if self.resize_size is not None and x.shape[-2:] != self.resize_size:
            x = x.float()
            if x.dim() == 5:  # (B, C, T, H, W): fold time into channels
                b, c, t = x.shape[:3]
                x = F.interpolate(
                    x.reshape(b, c * t, *x.shape[-2:]),
                    size=self.resize_size, mode='bilinear', align_corners=False
                ).reshape(b, c, t, *self.resize_size)
            else:
                x = F.interpolate(
                    x, size=self.resize_size, mode='bilinear', align_corners=False)

        ori_dtype = x.dtype
        if hasattr(self.network, 'dtype'):
            network_dtype = self.network.dtype
        else:
            network_dtype = next(self.network.parameters()).dtype
        x = x.to(network_dtype)

        return tree_map(
            lambda value: value.to(ori_dtype) if isinstance(value, torch.Tensor) else value,
            self.network(x, *args, **kwargs))

    def forward(
            self, return_d_loss=False, return_g_loss=False, reals=None, fakes=None, **kwargs):
        assert not (return_d_loss and return_g_loss), "Cannot return both d_loss and g_loss at the same time."
        use_features = self.feature_matching_loss is not None

        fakes_kwargs = fakes.get('kwargs', dict()).copy()
        fakes_kwargs.update(kwargs)
        if use_features:
            fakes_kwargs['return_features'] = True
        disc_pred_fakes = self.pred(*fakes['args'], **fakes_kwargs)

        disc_pred_reals = None
        if return_d_loss or (return_g_loss and (
                use_features or getattr(self.gan_loss, 'gan_type', None) == 'relativistic')):
            real_kwargs = reals.get('kwargs', dict()).copy()
            real_kwargs.update(kwargs)
            if use_features:
                real_kwargs['return_features'] = True
            disc_pred_reals = self.pred(*reals['args'], **real_kwargs)

        if return_d_loss:
            return self.d_loss(disc_pred_fakes, disc_pred_reals)
        elif return_g_loss:
            return self.g_loss(disc_pred_fakes, disc_pred_reals)
        else:
            return disc_pred_fakes
