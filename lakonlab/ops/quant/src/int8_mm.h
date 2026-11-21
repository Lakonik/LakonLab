#pragma once

#include <ATen/ATen.h>
#include <optional>

void int8_mm(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& x_scale, const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias, at::Tensor& out);
