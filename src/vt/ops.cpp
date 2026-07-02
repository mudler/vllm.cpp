// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/ops.h"

#include <array>

namespace vt {

namespace {
using OpTable =
    std::array<std::array<void*, kNumDeviceTypes>, static_cast<size_t>(OpId::kCount)>;
OpTable& Table() {
  static OpTable table{};
  return table;
}

bool IsFloat(DType d) { return d == DType::kF32 || d == DType::kF16 || d == DType::kBF16; }
}  // namespace

void RegisterOp(OpId op, DeviceType device, void* fn) {
  Table()[static_cast<size_t>(op)][static_cast<size_t>(device)] = fn;
}

void* GetOp(OpId op, DeviceType device) {
  void* fn = Table()[static_cast<size_t>(op)][static_cast<size_t>(device)];
  VT_CHECK(fn != nullptr, std::string("no kernel for op ") +
                              std::to_string(static_cast<int>(op)) + " on device type " +
                              std::to_string(static_cast<int>(device)));
  return fn;
}

using MatmulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);

void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2, "matmul: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[0], "matmul: inner dims mismatch");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[1],
           "matmul: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && out.dtype == DType::kF32,
           "matmul: float inputs and f32 output required");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "matmul: contiguous tensors required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul: device mismatch");
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmul, q.device.type))(q, out, a, b);
}

}  // namespace vt
