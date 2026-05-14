# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn.functional as F

from ..builder import MODULES, build_module
from . import GaussianFlow
from lakonlab.models.architectures.utils import get_module_device
from lakonlab.utils import module_eval


def calc_shifted_signal_ratio(sigma, shift):
    alpha = 1 - sigma
    alpha_sq = alpha.square()
    return alpha_sq / (alpha_sq + (shift * sigma).square())


@MODULES.register_module()
class AsymFlowVR(GaussianFlow):

    def __init__(
            self,
            *args,
            latent_patch_size=2,
            mse_loss_weight=1.0,
            perceptual_loss=None,
            loss_shift=None,
            **kwargs):
        super().__init__(*args, **kwargs)
        assert self.denoising_mean_mode.upper() == 'U'
        self.latent_patch_size = latent_patch_size
        self.mse_loss_weight = mse_loss_weight
        self.perceptual_loss = build_module(perceptual_loss) if perceptual_loss is not None else None
        self.loss_shift = loss_shift

    def forward_train(
            self,
            x_0,  # full rank data
            latents_2,  # low rank data in latent space
            teacher=None,  # reference low rank model (just reusing distillation model signature)
            teacher_kwargs=dict(),
            vae=None,  # full rank decoder
            **kwargs):
        device = get_module_device(self)
        dtype = x_0.dtype

        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()  # h * w or t * h * w
        ndim = x_0.dim()

        eps = self.train_cfg.get('eps', 1e-4)
        max_raw_t = self.train_cfg.get('max_raw_t', 1.0)
        min_raw_t = self.train_cfg.get('min_raw_t', 0.0)

        sigma = self.timestep_sampler(
            num_batches,
            seq_len=seq_len,
            raw_t_range=(min_raw_t, max_raw_t),
            scale_t=False,
            device=device
        ).reshape(num_batches, *((ndim - 1) * [1]))
        t = self.num_timesteps * sigma.flatten()
        sigma_clamped = sigma.clamp(min=self.sigma_min)

        # AsymFlowVR data preparation
        proj_mat = self.denoising.proj_buffer.to(dtype=dtype)  # (full_rank, base_rank)
        s = self.denoising.scale_buffer.to(dtype=dtype)

        latents_patchified = self.denoising.patchify(latents_2, self.latent_patch_size)
        latents_patchified_shape = latents_patchified.shape[2:]  # (h, w) or (t, h, w)
        # (bs, n, base_rank)
        latents_packed = self.denoising.pack(latents_patchified)

        x_0_low_rank_packed = latents_packed @ (proj_mat.T * s)  # (bs, n, full_rank)
        x_0_low_rank = self.denoising.unpatchify(
            self.denoising.unpack(x_0_low_rank_packed, *latents_patchified_shape),
            self.denoising.patch_size
        )

        noise = torch.randn_like(x_0)
        x_t, _, _ = self.sample_forward_diffusion(x_0, t, noise)
        x_t_low_rank, _, _ = self.sample_forward_diffusion(x_0_low_rank, t, noise)

        # reference low-rank forward
        with torch.no_grad(), module_eval(teacher):
            ref_u_low_rank = teacher(return_u=True, x_t=x_t_low_rank, t=t, **teacher_kwargs)
            ref_x_0_low_rank = self.u_to_x_0(ref_u_low_rank, x_t_low_rank, sigma=sigma)

        # full-rank forward
        pred_u = self.pred(x_t, t, **kwargs)
        pred_x_0 = self.u_to_x_0(pred_u, x_t, sigma=sigma)

        # adaptive residual weighting
        low_rank_diff = self.denoising.patchify(
            x_0_low_rank - ref_x_0_low_rank, self.denoising.patch_size, pack_channels=False
        )  # (bs, c, patch_numel, *)
        full_rank_diff = self.denoising.patchify(
            x_0 - pred_x_0.detach(), self.denoising.patch_size, pack_channels=False
        )  # (bs, c, patch_numel, *)

        num = (full_rank_diff * low_rank_diff).mean(dim=2, keepdim=True)
        den = low_rank_diff.square().mean(dim=2, keepdim=True).clamp(min=eps)
        res_coef = (num / den).clamp_(0.0, 1.0)

        if self.loss_shift is None:
            shifted_signal_ratio = 0.0
        else:
            shifted_signal_ratio = calc_shifted_signal_ratio(sigma, self.loss_shift)

        tgt_x_0 = x_0 - (1 - shifted_signal_ratio) * self.denoising.unpatchify(
            res_coef * low_rank_diff, self.denoising.patch_size, packed_channels=False)
        mse_loss = F.mse_loss(pred_x_0, tgt_x_0, reduction='none')
        mse_loss = 0.5 * self.mse_loss_weight * (mse_loss / sigma_clamped.square()).mean()  # velocity-weighted MSE loss

        loss = mse_loss
        log_vars = dict(
            loss_diffusion=float(mse_loss),
            res_coef=float(res_coef.mean())
        )

        if self.perceptual_loss is not None:
            if hasattr(vae, 'dtype'):
                vae_dtype = vae.dtype
            else:
                vae_dtype = next(vae.parameters()).dtype
            pred_image = vae.decode(pred_x_0.to(vae_dtype))
            tgt_image = vae.decode(x_0.to(vae_dtype))

            res_weight = self.denoising.unpatchify(
                res_coef.expand_as(low_rank_diff).square().mean(dim=1, keepdim=True).sqrt(),  # (bs, 1, patch_numel, *)
                self.denoising.patch_size,
                packed_channels=False)
            time_weight = shifted_signal_ratio / sigma_clamped.square()

            perceptual_loss = self.perceptual_loss(
                pred_image, tgt_image, weight=res_weight * time_weight)
            loss = loss + perceptual_loss
            log_vars.update(loss_perceptual=float(perceptual_loss))

        return loss, log_vars

    def forward(
            self,
            x_0=None,
            latents_2=None,
            return_loss=False,
            return_u=False,
            return_denoising_output=False,
            return_x_0=False,
            **kwargs):
        if return_loss:
            return self.forward_train(x_0, latents_2, **kwargs)
        elif return_u:
            return self.forward_u(**kwargs)
        elif return_denoising_output:
            return self.pred(**kwargs)
        elif return_x_0:
            return self.forward_x_0(**kwargs)
        else:
            return self.forward_test(x_0, **kwargs)
