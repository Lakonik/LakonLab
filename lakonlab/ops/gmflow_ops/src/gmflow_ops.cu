// Copyright (c) 2026 Hansheng Chen

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif

// Device function to compute pdf and cdf for a single sample.
__device__ __forceinline__ void gm1d_pdf_cdf_single(
    const float sample_val,
    const float inv_std,
    const float logstd_val,
    const float* __restrict__ means_ptr,
    const float* __restrict__ weights_ptr,
    const float* __restrict__ logweights_ptr,
    int num_gaussians,
    int HW,
    float& pdf_val,
    float& cdf_val
) {
    const float norm_const = 1.0f / sqrtf(2.0f * (float)M_PI);

    pdf_val = 0.0f;
    cdf_val = 0.0f;

    const float neg_logstd_val = -logstd_val;
    const float inv_sqrt2 = M_SQRT1_2;
    const float neg_half = -0.5f;

    for (int g = 0; g < num_gaussians; g++) {
        int gm_idx = g * HW;
        float mean_val = means_ptr[gm_idx];
        float w_val = weights_ptr[gm_idx];
        float lw_val = logweights_ptr[gm_idx];

        float diff = (sample_val - mean_val) * inv_std;
        float diff_sq = diff * diff;
        float logprob = fmaf(neg_half, diff_sq, lw_val + neg_logstd_val);

        float prob = __expf(logprob);
        pdf_val += prob;

        float erf_arg = diff * inv_sqrt2;
        float cdf_g = w_val * erff(erf_arg);
        cdf_val += cdf_g;
    }

    pdf_val *= norm_const;
}

// Kernel to perform the Newton-Raphson iterations entirely on the device.
// means: (B, num_gaussians, H, W)
// logstds: (B, 1, 1, 1)
// weights or logweights: (B, num_gaussians, H, W)
// samples: (B, n_samples, H, W)
// scaled_cdfs: (B, n_samples, H, W)
__global__ void gm1d_inverse_cdf_cuda_kernel(
    const float* __restrict__ means,
    const float* __restrict__ logstds,
    const float* __restrict__ logweights,
    const float* __restrict__ gm_weights,
    const float* __restrict__ scaled_cdfs,
    float* __restrict__ samples, // in-place update
    int B,
    int num_gaussians,
    int HW,
    int n_samples,
    int n_steps,
    float eps,
    float max_step_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_samples * HW;
    if (idx >= total) return;

    int hw_idx = idx % HW;
    int b_idx = idx / (HW * n_samples);
    int gm_idx = b_idx * num_gaussians * HW + hw_idx;

    float sample_val = samples[idx];
    float sc = scaled_cdfs[idx];
    float logstd_val = logstds[b_idx];
    float inv_std = __expf(-logstd_val);
    float clamp_range = max_step_size / inv_std;

    const float* cur_means_ptr = means + gm_idx;
    const float* cur_logweights_ptr = logweights + gm_idx;
    const float* cur_weights_ptr = gm_weights + gm_idx;

    for (int step = 0; step < n_steps; step++) {
        // Compute pdf and cdf at current sample:
        float pdf_val, cdf_val;
        gm1d_pdf_cdf_single(
            sample_val, inv_std, logstd_val,
            cur_means_ptr, cur_weights_ptr, cur_logweights_ptr,
            num_gaussians, HW, pdf_val, cdf_val
        );
        float denom = fmaxf(pdf_val, eps);
        float delta = 0.5f * (cdf_val - sc) / denom;
        // clamp delta
        if (delta > clamp_range) delta = clamp_range;
        if (delta < -clamp_range) delta = -clamp_range;
        sample_val = sample_val - delta;
    }

    samples[idx] = sample_val;
}

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
) {
    // Check device and dtypes (float32)
    TORCH_CHECK(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK(logstds.is_cuda(), "logstds must be CUDA");
    TORCH_CHECK(logweights.is_cuda(), "logweights must be CUDA");
    TORCH_CHECK(scaled_cdfs.is_cuda(), "scaled_cdfs must be CUDA");
    TORCH_CHECK(samples.is_cuda(), "samples must be CUDA");
    TORCH_CHECK(means.dtype() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(logstds.dtype() == torch::kFloat32, "logstds must be float32");
    TORCH_CHECK(logweights.dtype() == torch::kFloat32, "logweights must be float32");
    TORCH_CHECK(scaled_cdfs.dtype() == torch::kFloat32, "scaled_cdfs must be float32");
    TORCH_CHECK(samples.dtype() == torch::kFloat32, "samples must be float32");

    // Extract shapes
    // means: (B, num_gaussians, H, W)
    // logstds: (B, 1, 1, 1)
    // scaled_cdfs: (B, n_samples, H, W)
    // samples: (B, n_samples, H, W)
    int B = means.size(0);
    int num_gaussians = means.size(1);
    int H = means.size(-2);
    int W = means.size(-1);
    int n_samples = scaled_cdfs.size(1);
    int HW = H * W;

    TORCH_CHECK(samples.size(0) == B, "B must match");
    TORCH_CHECK(samples.size(1) == n_samples, "n_samples must match");
    TORCH_CHECK(samples.size(2) == H && samples.size(3) == W, "H,W must match");

    int threads = 256;
    int total = B * n_samples * HW;
    int blocks = (total + threads - 1) / threads;

    gm1d_inverse_cdf_cuda_kernel<<<blocks, threads>>>(
        means.data_ptr<float>(),
        logstds.data_ptr<float>(),
        logweights.data_ptr<float>(),
        gm_weights.data_ptr<float>(),
        scaled_cdfs.data_ptr<float>(),
        samples.data_ptr<float>(),
        B, num_gaussians, HW, n_samples,
        n_steps,
        eps,
        max_step_size
    );
    cudaDeviceSynchronize();
}
