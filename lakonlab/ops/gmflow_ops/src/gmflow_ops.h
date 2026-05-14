// Copyright (c) 2026 Hansheng Chen

#pragma once

#include <stdint.h>
#include <torch/torch.h>

void gm1d_inverse_cdf(
    torch::Tensor means,
    torch::Tensor logstds,
    torch::Tensor logweights,
    torch::Tensor gm_weights,
    torch::Tensor scaled_cdfs,
    torch::Tensor samples, // already initialized
    int n_steps,
    float eps,
    float max_step_size
);
