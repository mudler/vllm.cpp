// CPU compute-in-quant GEMM (`OpId::kMatmulBTQuant`) — the skeleton and its
// GENERIC COMPOSITE fallback (QUANT-GGUF-CIQ-GEMM work row G1).
//
// Structure mirrors llama.cpp @ 237ad9b96
// `ggml/src/ggml-cpu/ggml-cpu.c:1245-1443` (`ggml_compute_forward_mul_mat`):
// src0 is the [N,K] block-quantized weight, src1 the f32/bf16 activation, and
// the output is produced one row-dot at a time. The two pieces upstream puts
// in front of that dot — quantizing src1 into `wdata` with the weight type's
// `vec_dot_type` (:1313-1349) and the per-type integer `vec_dot` (:1426-1433)
// — are work rows G2 and G3. Until BOTH exist for a type, this kernel takes
// the composite path: decode the weight row to f32 through the traits table's
// `to_float` and take the plain f32 dot. That is the same arithmetic the
// current dequant-to-bf16 loader path performs, just without materializing the
// whole tensor, so it is a correct (if unaccelerated) executor for every block
// dtype and it is exactly the reference the ported MUL_MAT NMSE tests measure
// the quantized path against.
//
// Parallelism partitions OUTPUT ROWS only and each output keeps its own
// sequential K reduction, so results are bit-identical to single-thread by
// construction (same rule as every other kernel in cpu_ops.cpp).
#include <vector>

#include "vt/quant.h"
#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadActF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default:
      VT_CHECK(false, "matmul_bt_quant: unsupported activation dtype");
      return 0.0f;
  }
}

void StoreOutF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "matmul_bt_quant: unsupported output dtype");
  }
}

// Generic composite: decode weight row j once, dot it against every activation
// row. Chunking by WEIGHT ROWS (ggml's nr0) keeps one decode per row instead of
// one per output element, which is what makes the fallback usable as the unit
// oracle at model shapes.
void ComposeChunk(Tensor& out, const Tensor& a, const Tensor& b,
                  ToFloatFn to_float, int64_t j0, int64_t j1) {
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = b.shape[0];
  const int64_t a_rs = a.stride[0];
  const size_t row_bytes = RowSizeBytes(b.dtype, k);
  const uint8_t* blocks = b.Ptr<const uint8_t>();

  std::vector<float> w(static_cast<size_t>(k));
  for (int64_t j = j0; j < j1; ++j) {
    to_float(blocks + static_cast<size_t>(j) * row_bytes, w.data(), k);
    for (int64_t i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += LoadActF32(a, i * a_rs + p) * w[static_cast<size_t>(p)];
      }
      StoreOutF32(out, i * n + j, acc);
    }
  }
}

void MatmulBTQuantKernel(Queue& q, Tensor& out, const Tensor& a,
                         const Tensor& b) {
  (void)q;
  const QuantTypeTraits& traits = QuantTraits(b.dtype);
  // G2/G3 flip this on per type; when they do, the quantize-src1-once + integer
  // vec_dot path from ggml-cpu.c:1313-1443 replaces the composite below.
  VT_CHECK(!HasQuantDotKernel(b.dtype),
           "matmul_bt_quant: a vec_dot kernel is registered for this dtype but "
           "the quantized GEMM path is not wired yet (CIQ work rows G2/G3)");
  VT_CHECK(traits.to_float != nullptr,
           "matmul_bt_quant: no to_float decoder for this weight dtype");

  ParallelForRows(CurrentThreadpool(), b.shape[0],
                  [&](int64_t j0, int64_t j1) {
                    ComposeChunk(out, a, b, traits.to_float, j0, j1);
                  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantKernel)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cpu
