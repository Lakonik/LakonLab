# Copyright (c) 2026 Hansheng Chen

import os
import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from lakonlab.models import GMFlowMLP2DDenoiser, GMFlow
from lakonlab.datasets import CheckerboardData


D = 2
EPS = 1e-4
T_SCALE = 1000
N_TEST_SAMPLES = 1e6


def parse_args():
    parser = argparse.ArgumentParser(
        description='A minimal GMFlow trainer using the 2D checkerboard dataset (without transition loss and EMA).')
    parser.add_argument('-k', type=int, default=32, help='number of Gasussian components')
    parser.add_argument('--batch-size', type=int, default=4096, help='batch size')
    parser.add_argument('--num-iters', type=int, default=50000, help='number of iterations')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--nfe', type=int, default=8, help='number of sampling steps')
    parser.add_argument('--out', type=str, default='gmflow_toymodel.png', help='output file path')
    args = parser.parse_args()
    return args


def gm_kl_loss(gm, sample, eps=1e-4):
    """
    Gaussian mixture KL divergence loss (without constant terms), a.k.a. GM NLL loss.

    Args:
        gm (dict):
            means (torch.Tensor): (bs, num_gaussians, D)
            logstds (torch.Tensor): (bs, 1, 1)
            logweights (torch.Tensor): (bs, num_gaussians, 1)
        sample (torch.Tensor): (bs, D)

    Returns:
        torch.Tensor: (bs, )
    """
    means = gm['means']
    logstds = gm['logstds']
    logweights = gm['logweights']

    inverse_stds = torch.exp(-logstds).clamp(max=1 / eps)
    diff_weighted = (sample.unsqueeze(-2) - means) * inverse_stds  # (bs, num_gaussians, D)
    gaussian_ll = (-0.5 * diff_weighted.square() - logstds).sum(dim=-1)  # (bs, num_gaussians)
    gm_nll = -torch.logsumexp(gaussian_ll + logweights.squeeze(-1), dim=-1)  # (bs, )
    return gm_nll


def main():
    args = parse_args()
    num_gaussians = args.k
    batch_size = args.batch_size
    num_iters = args.num_iters
    lr = args.lr
    num_steps = args.nfe
    out_path = args.out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    denoiser = GMFlowMLP2DDenoiser(num_gaussians=num_gaussians).to(device)
    dataset = CheckerboardData(scale=4)
    sample_batches = dataset.samples.to(device).split(batch_size, dim=0)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)

    for i in range(num_iters):
        x_0 = sample_batches[i % len(sample_batches)]
        optimizer.zero_grad()

        t = torch.rand(x_0.size(0), device=device).clamp(min=EPS)
        noise = torch.randn_like(x_0)

        sigma = t
        alpha = 1 - sigma
        x_t = alpha.unsqueeze(-1) * x_0 + sigma.unsqueeze(-1) * noise
        u = noise - x_0  # equal to (x_t - x_0) / sigma

        u_gm = denoiser(x_t, t * T_SCALE)
        loss = gm_kl_loss(u_gm, u)

        loss.mean().backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f'Iter {i}, loss: {loss.mean().item()}')

    print('Training finished. Starting inference...')

    torch.set_grad_enabled(False)

    model = GMFlow(
        denoising=denoiser,
        num_timesteps=T_SCALE,
        test_cfg=dict(  # use 2nd-order GM-SDE solver
            output_mode='sample',
            sampler='FlowSDE',
            num_timesteps=num_steps,
            order=2)
    ).eval()

    samples = []
    for _ in tqdm.tqdm(range(int(N_TEST_SAMPLES // batch_size))):
        noise = torch.randn((batch_size, D, 1, 1), device=device)
        samples.append(model.forward_test(noise=noise).reshape(batch_size, D).cpu().numpy())
    samples = np.concatenate(samples, axis=0)

    histo, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=200, range=[[-4.2, 4.2], [-4.2, 4.2]])
    histo_image = (histo.T[::-1] / 160).clip(0, 1)
    histo_image = cm.viridis(histo_image)
    histo_image = np.round(histo_image * 255).clip(min=0, max=255).astype(np.uint8)

    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.imsave(out_path, histo_image)

    print(f'Sample histogram saved to {out_path}.')


if __name__ == '__main__':
    main()
