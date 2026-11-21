#include "int8_mm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

void int8_mm(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& x_scale, const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias, at::Tensor& out) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && weight.dim() == 2,
                "Expected CUDA matrices");
    for (const auto& tensor : {x, weight, x_scale, weight_scale, out}) {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                    "Expected contiguous tensors on the same CUDA device");
    }
    TORCH_CHECK(x.scalar_type() == at::kChar && weight.scalar_type() == at::kChar,
                "Expected INT8 activations and weights");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat && weight_scale.scalar_type() == at::kFloat,
                "Expected FP32 scales");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "Expected BF16 output");
    const int m = x.size(0), n = weight.size(0), k = x.size(1);
    TORCH_CHECK(m > 0 && n > 0 && k > 0 && k % 16 == 0 && n % 8 == 0,
                "Expected nonempty matrices with K divisible by 16 and N divisible by 8");
    TORCH_CHECK(weight.size(1) == k && x_scale.numel() == m && weight_scale.numel() == n,
                "Incompatible weights or scales");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == n,
                "Incompatible output shape");
    if (bias) {
        TORCH_CHECK(bias->device() == x.device() && bias->is_contiguous()
                    && bias->scalar_type() == at::kBFloat16 && bias->numel() == n,
                    "Expected a contiguous BF16 bias on the same device with N elements");
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0, "Expected an SM90 GPU");
    const auto stream = at::cuda::getCurrentCUDAStream();
    using Tile = Shape<_128, _256, _128>;
    using Cluster = Shape<_2, _1, _1>;
    using D = cutlass::bfloat16_t;
    using StrideD = Stride<int64_t, _1, _0>;
    using AS = fusion::Sm90ColBroadcast<0, Tile, float, float, Stride<_1, _0, _0>>;
    using BS = fusion::Sm90RowBroadcast<0, Tile, float, float, Stride<_0, _1, _0>>;
    using Bias = fusion::Sm90RowBroadcast<0, Tile, D, D, Stride<_0, _1, _0>>;
    using Mul = fusion::Sm90Compute<cutlass::multiplies, float, float,
                                    cutlass::FloatRoundStyle::round_to_nearest>;
    using Scales = fusion::Sm90EVT<Mul, AS, BS>;
    using Scaled = fusion::Sm90EVT<Mul, fusion::Sm90AccFetch, Scales>;
    using Add = fusion::Sm90Compute<cutlass::plus, D, float,
                                    cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT = fusion::Sm90EVT<Add, Scaled, Bias>;
    using ES = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using KS = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        int32_t, float, void, StrideD, 8, D, StrideD, 8,
        ES, EVT>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        int8_t, cutlass::layout::RowMajor, 16,
        int8_t, cutlass::layout::ColumnMajor, 16,
        int32_t, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        KS>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
        Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    auto da = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(m, k, 1));
    auto db = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(n, k, 1));
    auto dd = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
    typename AS::Arguments asa{x_scale.data_ptr<float>()};
    typename BS::Arguments bsa{weight_scale.data_ptr<float>()};
    // Sm90RowBroadcast treats a null bias pointer as zero.
    typename Bias::Arguments ba{bias ? reinterpret_cast<const D*>(bias->data_ptr<at::BFloat16>()) : nullptr};
    typename Scales::Arguments scales{asa, bsa, {}};
    typename Scaled::Arguments scaled{{}, scales, {}};
    typename EVT::Arguments evt{scaled, ba, {}};
    cutlass::KernelHardwareInfo hw{};
    hw.device_id = x.get_device();
    hw.sm_count = properties->multiProcessorCount;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1}, {x.data_ptr<int8_t>(), da, weight.data_ptr<int8_t>(), db},
        {evt, nullptr, dd, reinterpret_cast<D*>(out.data_ptr<at::BFloat16>()), dd}, hw};
    Gemm gemm;
    auto status = gemm.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
    auto workspace = at::empty({static_cast<int64_t>(gemm.get_workspace_size(args))},
                               x.options().dtype(at::kByte));
    status = gemm.run(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}
