// CPU compute-in-quant GEMM (`OpId::kMatmulBTQuant`) — QUANT-GGUF-CIQ-GEMM
// work rows G1 (skeleton + composite fallback) and G3 (the quantized path).
//
// Structure mirrors llama.cpp @ 237ad9b96
// `ggml/src/ggml-cpu/ggml-cpu.c:1245-1443` (`ggml_compute_forward_mul_mat`):
// src0 is the [N,K] block-quantized weight, src1 the f32/bf16 activation, and
// the output is produced one row-dot at a time.
//
// TWO PATHS, selected by `HasQuantDotKernel(b.dtype)`:
//
//  1. QUANTIZED (the point of the track) — for the six executable GGUF weight
//     encodings. Mirrors upstream exactly: quantize src1 ONCE into a scratch
//     `wdata` buffer using the weight type's `vec_dot_type`
//     (:1313-1349), then run one integer `vec_dot` per output element
//     (:1426-1433). The weight blocks are never expanded.
//
//  2. GENERIC COMPOSITE (G1's fallback) — for any block dtype without a
//     `vec_dot` (today only Q8_K, which is activation-only). Decodes the
//     weight row to f32 via the traits table's `to_float` and takes the plain
//     f32 dot. It stays in the tree permanently because it is the INDEPENDENT
//     reference the ported MUL_MAT NMSE tests measure path 1 against — a
//     different decode (the loader's `dequantize_row_*`) reaching the same
//     mathematical answer, so a block-decode bug in a `vec_dot` cannot hide.
//
// DETERMINISM (project rule: no atomicAdd-style nondeterminism). Both paths
// partition OUTPUT ROWS only; every output element keeps its own sequential K
// reduction in a fixed order, and the activation scratch is written once per
// row before any dot reads it. Results are therefore bit-identical run to run
// and independent of thread count — asserted directly in
// tests/vt/test_ops_quant_dot.cpp.
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

// Quantized path — ggml-cpu.c:1313-1349 (src1 -> wdata) + :1155-1243 / :1426
// (one vec_dot per output). `act` holds the M quantized activation rows, laid
// out contiguously at `act_row_bytes` stride exactly like upstream's wdata.
void QuantChunk(Tensor& out, const std::vector<uint8_t>& act,
                size_t act_row_bytes, const Tensor& b, VecDotFn vec_dot,
                int64_t k, int64_t j0, int64_t j1) {
  const int64_t m = out.shape[0];
  const int64_t n = b.shape[0];
  const size_t w_row_bytes = RowSizeBytes(b.dtype, k);
  const uint8_t* w = b.Ptr<const uint8_t>();

  for (int64_t j = j0; j < j1; ++j) {
    const uint8_t* w_row = w + static_cast<size_t>(j) * w_row_bytes;
    for (int64_t i = 0; i < m; ++i) {
      const uint8_t* a_row = act.data() + static_cast<size_t>(i) * act_row_bytes;
      float acc = 0.0f;
      // nrc == 1: the generic tier dots exactly one row pair, so the row
      // strides bs/bx/by are inert (upstream passes 0 the same way outside its
      // mmla path). G6's nrows==2 kernels are what give them meaning.
      vec_dot(static_cast<int>(k), &acc, /*bs=*/0, w_row, /*bx=*/0, a_row,
              /*by=*/0, /*nrc=*/1);
      StoreOutF32(out, i * n + j, acc);
    }
  }
}

void MatmulBTQuantKernel(Queue& q, Tensor& out, const Tensor& a,
                         const Tensor& b) {
  (void)q;
  const QuantTypeTraits& traits = QuantTraits(b.dtype);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];

  if (HasQuantDotKernel(b.dtype)) {
    // `QuantActRowBytes` throws unless k is a whole number of ACTIVATION
    // blocks (256 for the K-quants, 32 otherwise); RowSizeBytes on the weight
    // side enforces the same for its own block size. A ragged K therefore
    // fails loudly here instead of mis-striding scratch.
    const size_t act_row_bytes = QuantActRowBytes(b.dtype, k);
    const FromFloatFn from_float = QuantTraits(traits.vec_dot_type).from_float;

    // Widen src1 to f32 and quantize it once, mirroring ggml-cpu.c:1313-1349.
    // ggml's src1 is already f32; ours may be bf16/f16, so the widen is the
    // one extra step — it is exact (both widen losslessly into f32).
    std::vector<uint8_t> act(QuantActScratchBytes(b.dtype, m, k));
    std::vector<float> row(static_cast<size_t>(k));
    const int64_t a_rs = a.stride[0];
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t p = 0; p < k; ++p) {
        row[static_cast<size_t>(p)] = LoadActF32(a, i * a_rs + p);
      }
      from_float(row.data(), act.data() + static_cast<size_t>(i) * act_row_bytes,
                 k);
    }

    ParallelForRows(CurrentThreadpool(), b.shape[0],
                    [&](int64_t j0, int64_t j1) {
                      QuantChunk(out, act, act_row_bytes, b, traits.vec_dot, k,
                                 j0, j1);
                    });
    return;
  }

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
