// Shared validation/dispatch for the compiled Gemma fusion bindings.
#include "vt/compiled_gemma.h"

#include <cmath>
#include <limits>

namespace vt {
namespace {
size_t CheckRow(const Queue& q, const Tensor& t, int64_t rows, int64_t width) {
  VT_CHECK(rows >= 0 && width > 0 && t.rank == 2 && t.shape[0] == rows && t.shape[1] == width &&
               t.dtype == DType::kBF16 && t.device == q.device,
           "compiled_gemma: expected contiguous BF16 [T,H] on queue device");
  VT_CHECK(static_cast<uint64_t>(width) <= std::numeric_limits<size_t>::max() / 2 &&
               static_cast<uint64_t>(rows) <= std::numeric_limits<size_t>::max() / 2 / width,
           "compiled_gemma: row storage overflow");
  // Validate the product before IsContiguous() multiplies the dimensions.
  VT_CHECK(t.IsContiguous(), "compiled_gemma: expected contiguous BF16 [T,H] on queue device");
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(width) * 2;
  const auto address = reinterpret_cast<uintptr_t>(t.data);
  VT_CHECK(bytes == 0 || (address && address % 2 == 0 &&
                          address <= std::numeric_limits<uintptr_t>::max() - bytes),
           "compiled_gemma: invalid row storage");
  return bytes;
}
void CheckWeight(const Queue& q, const Tensor& w, int64_t width) {
  VT_CHECK(w.rank == 1 && w.shape[0] == width && w.stride[0] == 1 && w.dtype == DType::kBF16 &&
               w.device == q.device && w.data && reinterpret_cast<uintptr_t>(w.data) % 2 == 0 &&
               reinterpret_cast<uintptr_t>(w.data) <=
                   std::numeric_limits<uintptr_t>::max() - static_cast<size_t>(width) * 2,
           "compiled_gemma: expected contiguous BF16 gamma [H] on queue device");
}
bool Overlap(const Tensor& a, size_t na, const Tensor& b, size_t nb) {
  const auto x = reinterpret_cast<uintptr_t>(a.data), y = reinterpret_cast<uintptr_t>(b.data);
  return na && nb && (x <= y ? y - x < na : x - y < nb);
}
void Epsilon(float eps) {
  VT_CHECK(std::isfinite(eps) && eps >= 0.f,
           "compiled_gemma: epsilon must be finite and nonnegative");
}
}  // namespace

void FusedChain(Queue& q, Tensor& out, const Tensor& input, const Tensor& weight,
                const ScaledRmsNormArgs& args, Tensor& residual) {
  VT_CHECK(input.rank == 2, "compiled_gemma: scaled input must be rank two");
  const auto rows = input.shape[0], width = input.shape[1];
  const size_t bytes = CheckRow(q, input, rows, width);
  CheckRow(q, out, rows, width);
  CheckRow(q, residual, rows, width);
  CheckWeight(q, weight, width);
  Epsilon(args.eps);
  VT_CHECK(std::isfinite(args.scale) && args.scale > 0.f,
           "compiled_gemma: scale must be finite and positive");
  VT_CHECK(!Overlap(out, bytes, input, bytes) && !Overlap(out, bytes, residual, bytes) &&
               !Overlap(residual, bytes, input, bytes) &&
               !Overlap(out, bytes, weight, static_cast<size_t>(width) * 2) &&
               !Overlap(residual, bytes, weight, static_cast<size_t>(width) * 2),
           "compiled_gemma: scaled norm outputs overlap inputs");
  if (rows == 0) return;
  reinterpret_cast<ScaledRmsNormFn>(GetOp(OpId::kScaledRmsNorm, q.device.type))(
      q, out, input, weight, args, residual);
}

void FusedChain(Queue& q, Tensor& out, const SandwichNormInputs& in, float eps, Tensor* residual) {
  VT_CHECK(in.a.rank == 2, "compiled_gemma: sandwich input must be rank two");
  const auto rows = in.a.shape[0], width = in.a.shape[1];
  const size_t bytes = CheckRow(q, in.a, rows, width);
  CheckRow(q, in.base, rows, width);
  CheckRow(q, out, rows, width);
  CheckWeight(q, in.a_weight, width);
  CheckWeight(q, in.weight, width);
  Epsilon(eps);
  VT_CHECK((in.delta != nullptr) == (in.delta_weight != nullptr),
           "compiled_gemma: delta requires its normalization weight");
  if (in.delta) {
    CheckRow(q, *in.delta, rows, width);
    CheckWeight(q, *in.delta_weight, width);
  }
  if (residual) CheckRow(q, *residual, rows, width);
  for (const Tensor* input :
       {&in.a, &in.base, in.delta, &in.a_weight, &in.weight, in.delta_weight}) {
    if (!input) continue;
    const size_t input_bytes = input->rank == 1 ? static_cast<size_t>(width) * 2 : bytes;
    VT_CHECK(!Overlap(out, bytes, *input, input_bytes),
             "compiled_gemma: normalized output overlaps input");
    if (residual)
      VT_CHECK(!Overlap(*residual, bytes, *input, input_bytes) ||
                   (input == &in.base && residual->data == in.base.data),
               "compiled_gemma: residual may alias only the complete base tensor");
  }
  VT_CHECK(!residual || !Overlap(out, bytes, *residual, bytes), "compiled_gemma: outputs overlap");
  if (rows == 0) return;
  reinterpret_cast<SandwichRmsNormFn>(GetOp(OpId::kSandwichRmsNorm, q.device.type))(q, out, in, eps,
                                                                                    residual);
}

void CompiledGeluErfMul(Queue& q, Tensor& out, const Tensor& packed) {
  VT_CHECK(packed.rank == 2 && packed.shape[1] > 0 && packed.shape[1] % 2 == 0,
           "compiled_gelu_erf_mul: expected packed [T,2H]");
  const auto rows = packed.shape[0], width = packed.shape[1] / 2;
  const size_t input_bytes = CheckRow(q, packed, rows, width * 2),
               output_bytes = CheckRow(q, out, rows, width);
  VT_CHECK(!Overlap(packed, input_bytes, out, output_bytes),
           "compiled_gelu_erf_mul: output overlaps input");
  if (rows == 0) return;
  reinterpret_cast<GeluAndMulFn>(GetOp(OpId::kCompiledGeluErfMul, q.device.type))(q, out, packed);
}
}  // namespace vt
