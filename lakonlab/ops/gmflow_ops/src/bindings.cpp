// Copyright (c) 2026 Hansheng Chen

#include <torch/extension.h>

#include "gmflow_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gm1d_inverse_cdf", &gm1d_inverse_cdf, "gm1d_inverse_cdf (CUDA)");
}
