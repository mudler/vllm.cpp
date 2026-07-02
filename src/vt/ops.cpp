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

using RmsNormFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const RmsNormArgs&, Tensor*);

void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 1, "rmsnorm: x/out rank-2, w rank-1");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1], "rmsnorm: shape mismatch");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && out.dtype == DType::kF32,
           "rmsnorm: float in, f32 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "rmsnorm: contiguous required");
  if (residual != nullptr) {
    VT_CHECK(residual->dtype == DType::kF32 && residual->rank == 2 &&
                 residual->shape[0] == x.shape[0] && residual->shape[1] == x.shape[1] &&
                 residual->IsContiguous() && residual->device == x.device,
             "rmsnorm: residual must be f32 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "rmsnorm: device mismatch (x/out/weight/queue)");
  reinterpret_cast<RmsNormFn>(GetOp(OpId::kRmsNorm, q.device.type))(q, out, x, weight, args,
                                                                    residual);
}

using SiluAndMulFn = void (*)(Queue&, Tensor&, const Tensor&);
using EmbeddingFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);

void SiluAndMul(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "silu_and_mul: rank-2 required");
  VT_CHECK(x.shape[1] % 2 == 0, "silu_and_mul: inner dim must be even");
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1] / 2,
           "silu_and_mul: output shape mismatch");
  VT_CHECK(IsFloat(x.dtype) && out.dtype == DType::kF32, "silu_and_mul: float in, f32 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "silu_and_mul: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "silu_and_mul: device mismatch");
  reinterpret_cast<SiluAndMulFn>(GetOp(OpId::kSiluAndMul, q.device.type))(q, out, x);
}

void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids) {
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2, "embedding: bad ranks");
  VT_CHECK(out.shape[0] == ids.shape[0] && out.shape[1] == table.shape[1],
           "embedding: output shape mismatch");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64, "embedding: ids i32/i64");
  VT_CHECK(IsFloat(table.dtype) && out.dtype == DType::kF32, "embedding: float table, f32 out");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "embedding: contiguous required");
  VT_CHECK(table.device == out.device && ids.device == table.device && table.device == q.device,
           "embedding: device mismatch (table/out/ids/queue)");
  reinterpret_cast<EmbeddingFn>(GetOp(OpId::kEmbedding, q.device.type))(q, out, table, ids);
}

}  // namespace vt
