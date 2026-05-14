# Copyright (c) 2026 Hansheng Chen

import os
import argparse
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from lakonlab.models.diffusions.piflow_policies import GMFlowPolicy
from lakonlab.models.architectures import (
    QwenImageTransformer2DModel, PretrainedVAEQwenImage, PretrainedQwenImageTextEncoder)
from lakonlab.models import GaussianFlow


EPS = 1e-4
TOTAL_SUBSTEPS = 128
PRINT_EVERY = 10
SAVE_EVERY = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description='A minimal 1-NFE pi-Flow imitation distillation trainer that overfits the teacher (Qwen-Image) '
                    'behavior on a fixed initial noise using a static GMFlow policy.')
    parser.add_argument(
        '--prompt',
        type=str,
        default='Photo of a coffee shop entrance featuring a chalkboard sign reading "π-Qwen Coffee 😊 $2 per cup," with a neon '
                'light beside it displaying "π-通义千问". Next to it hangs a poster showing a beautiful Chinese woman, '
                'and beneath the poster is written "e≈2.71828-18284-59045-23536-02874-71352".',
        help='text prompt')
    parser.add_argument(
        '--cfg', type=float, default=4.0, help='teacher classifier-free guidance scale')
    parser.add_argument(
        '--seed', type=int, default='42', help='random seed')
    parser.add_argument(
        '-k', type=int, default=32, help='number of Gasussian components')
    parser.add_argument(
        '--num-iters', type=int, default=5000, help='number of iterations')
    parser.add_argument(
        '--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument(
        '--out', type=str, default='viz/piflow_qwen_toymodel/output.png', help='output file path')
    parser.add_argument(
        '--h', type=int, default=768, help='image height')
    parser.add_argument(
        '--w', type=int, default=1360, help='image width')
    parser.add_argument(
        '--num-intermediates', type=int, default=2, help='number of intermediate samples')
    args = parser.parse_args()
    return args


class StaticGMM(nn.Module):
    """A toy model that outputs a static GM, ignoring the input x_t_src and t_src. In practice, a real model should
    take x_t and t as input and output a dynamic GM that varies with x_t_src and t_src.
    """

    def __init__(self, init_u, num_gaussians=8):
        super().__init__()
        self.latent_size = init_u.shape[1:]
        self.num_gaussians = num_gaussians
        self.means = nn.Parameter(
            init_u.repeat(1, num_gaussians, 1, 1, 1)
            + torch.randn(1, num_gaussians, *self.latent_size, device=init_u.device) * 0.5)
        self.logstds = nn.Parameter(torch.full((1, 1, 1, 1, 1), fill_value=np.log(0.05)))
        self.logweight_logits = nn.Parameter(torch.zeros(1, num_gaussians, 1, *self.latent_size[1:]))

    def forward(self, x_t_src, t_src):
        assert (t_src == 1).all(), 'This toy model only supports 1-NFE sampling, thus t_src == 1.'
        assert x_t_src.size(0) == 1, 'This toy model only supports batch size 1.'
        assert x_t_src.shape[1:] == self.latent_size, \
            f'Expected input shape (1, {self.latent_size}), got {x_t_src.shape}.'
        # this toy model assumes the input is fixed, so we ignore x_t_src and t_src and return the static GM
        return dict(
            means=self.means,
            logstds=self.logstds,
            logweights=self.logweight_logits.log_softmax(dim=1)
        )


def policy_rollout(
        x_t_start: torch.Tensor,  # (B, C, *, H, W)
        raw_t_start: torch.Tensor,  # (B, )
        raw_t_end: torch.Tensor,  # (B, )
        policy,
        warp_t_fun):

    ndim = x_t_start.dim()
    raw_t_start = raw_t_start.reshape(*(ndim * [1]))
    raw_t_end = raw_t_end.reshape(*(ndim * [1]))

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * TOTAL_SUBSTEPS).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps

    raw_t = raw_t_start
    sigma_t = warp_t_fun(raw_t)
    x_t = x_t_start

    for substep_id in range(num_substeps.item()):
        u = policy.pi(x_t, sigma_t)

        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = warp_t_fun(raw_t_minus)
        x_t_minus = x_t + u * (sigma_t_minus - sigma_t)

        x_t = x_t_minus
        sigma_t = sigma_t_minus
        raw_t = raw_t_minus

    x_t_end = x_t
    sigma_t_end = sigma_t
    return x_t_end, sigma_t_end.flatten()


def main():
    args = parse_args()
    prompt = args.prompt
    num_gaussians = args.k
    num_iters = args.num_iters
    lr = args.lr
    out_path = args.out
    guidance_scale = args.cfg
    num_intermediates = args.num_intermediates

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_path_noext, out_ext = os.path.splitext(out_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'

    text_encoder = PretrainedQwenImageTextEncoder(
        model_name_or_path='Qwen/Qwen-Image',
        torch_dtype=dtype,
        max_sequence_length=512,
        pad_seq_len=512,
    ).to(device)

    prompt_embed_kwargs = text_encoder(prompt)
    if guidance_scale > 1.0:
        empty_prompt_embed_kwargs = text_encoder('')
        for k in prompt_embed_kwargs:
            prompt_embed_kwargs[k] = torch.cat([
                empty_prompt_embed_kwargs[k],
                prompt_embed_kwargs[k]], dim=0)

    del text_encoder
    torch.cuda.empty_cache()

    vae = PretrainedVAEQwenImage(
        model_name_or_path='Qwen/Qwen-Image',
        subfolder='vae',
        torch_dtype=dtype).to(device)
    vae_scale_factor = 8
    vae_latent_size = (16, args.h // vae_scale_factor, args.w // vae_scale_factor)

    teacher = GaussianFlow(
        denoising=QwenImageTransformer2DModel(
            patch_size=2,
            freeze=True,
            pretrained='huggingface://Qwen/Qwen-Image/transformer/diffusion_pytorch_model.safetensors.index.json',
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
            torch_dtype=dtype),
        num_timesteps=1,
        denoising_mean_mode='U',
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=3.2,
            logit_normal_enable=False)).eval().to(device)

    # get initial noise
    torch.manual_seed(args.seed)
    x_t_src = torch.randn((1, *vae_latent_size), device=device)
    t_src = torch.ones(1, device=device)

    # initialize student using the u of teacher
    u = teacher.forward(
        return_u=True, x_t=x_t_src, t=t_src, guidance_scale=guidance_scale, **prompt_embed_kwargs)
    student = StaticGMM(
        init_u=u,
        num_gaussians=num_gaussians).to(device)

    # start training
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    loss_list = []

    for i in range(1, num_iters + 1):
        optimizer.zero_grad()

        denoising_output = student(x_t_src, t_src)
        policy = GMFlowPolicy(denoising_output, x_t_src, t_src)
        detached_policy = policy.detach()

        loss = 0
        intermediate_t_samples = torch.rand(num_intermediates, device=device).clamp(min=EPS)
        for raw_t in intermediate_t_samples:
            x_t, t = policy_rollout(
                x_t_start=x_t_src,
                raw_t_start=t_src,
                raw_t_end=raw_t,
                policy=detached_policy,
                warp_t_fun=teacher.timestep_sampler.warp_t)
            pred_u = policy.pi(x_t, t)
            teacher_u = teacher.forward(
                return_u=True, x_t=x_t, t=t, guidance_scale=guidance_scale, **prompt_embed_kwargs)
            loss += F.mse_loss(pred_u, teacher_u) / num_intermediates

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if i % PRINT_EVERY == 0 or i == num_iters:
            print(f'Iter {i:04d}/{num_iters:04d}, loss: {np.mean(loss_list):.6f}')
            loss_list = []

        if i % SAVE_EVERY == 0 or i == num_iters:
            with torch.no_grad():
                x_0, _ = policy_rollout(
                    x_t_start=x_t_src,
                    raw_t_start=t_src,
                    raw_t_end=torch.zeros(1, device=device),
                    policy=policy,
                    warp_t_fun=teacher.timestep_sampler.warp_t)
                image = ((vae.decode(x_0.to(getattr(torch, dtype))) / 2 + 0.5).clamp(0, 1) * 255).round().to(
                    dtype=torch.uint8, device='cpu').squeeze(0).permute(1, 2, 0).numpy()
                Image.fromarray(image).save(f'{out_path_noext}.iter{i:04d}{out_ext}')
                print(f'Image saved to {out_path_noext}.iter{i:04d}{out_ext}')


if __name__ == '__main__':
    main()
