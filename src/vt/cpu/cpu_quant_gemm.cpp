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
#include <cmath>
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

// mmla 2x2-tiled path — cpu_quant_dot_arm.cpp kernels + ggml-cpu.c:1233-1239
// tile convention. Parallelized over WEIGHT-ROW PAIRS so each thread owns whole
// pairs (0,1),(2,3),… : the (weight-row, activation-row) pairing is therefore
// GLOBAL and independent of how the pairs are chunked across threads, which is
// what keeps the result bit-identical run-to-run AND across thread counts (the
// project determinism contract). One `vec_dot2(nrc=2)` writes a 2x2 tile:
// s[0]=(w_j,a_i), s[1]=(w_{j+1},a_i), s[bs]=(w_j,a_{i+1}), s[bs+1]=(w_{j+1},a_{i+1}).
// Requires m and n even; the caller guards that (else the whole GEMM takes the
// nrc==1 QuantChunk, exactly as ggml drops to num_rows_per_vec_dot=1).
void QuantChunkMmla(Tensor& out, const std::vector<uint8_t>& act,
                    size_t act_row_bytes, const Tensor& b, VecDotFn vec_dot2,
                    int64_t k, int64_t jp0, int64_t jp1) {
  const int64_t m = out.shape[0];
  const int64_t n = b.shape[0];
  const size_t w_row_bytes = RowSizeBytes(b.dtype, k);
  const uint8_t* w = b.Ptr<const uint8_t>();

  for (int64_t jp = jp0; jp < jp1; ++jp) {
    const int64_t j = 2 * jp;
    const uint8_t* w_row = w + static_cast<size_t>(j) * w_row_bytes;
    for (int64_t i = 0; i + 1 < m; i += 2) {
      const uint8_t* a_row = act.data() + static_cast<size_t>(i) * act_row_bytes;
      float tmp[4];
      vec_dot2(static_cast<int>(k), tmp, /*bs=*/2, w_row, /*bx=*/w_row_bytes,
               a_row, /*by=*/act_row_bytes, /*nrc=*/2);
      StoreOutF32(out, i * n + j, tmp[0]);
      StoreOutF32(out, i * n + j + 1, tmp[1]);
      StoreOutF32(out, (i + 1) * n + j, tmp[2]);
      StoreOutF32(out, (i + 1) * n + j + 1, tmp[3]);
    }
  }
}

void MatmulBTQuantKernel(Queue& q, Tensor& out, const Tensor& a,
                         const Tensor& b) {
  (void)q;
  const QuantTypeTraits& traits = QuantTraits(b.dtype);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];

  // CIQ G7: a weight the loader REPACKED into the i8mm interleave takes the
  // repack gemm/gemv (bit-identical to the paths below). The flag is only ever
  // set on an i8mm-capable process (QuantRepackEligible), so QuantRepackMatmul's
  // kernels are guaranteed live here.
  if (b.repacked) {
    QuantRepackMatmul(out, a, b);
    return;
  }

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

    // Arm i8mm mmla tier (G6): a 2x2 register-tiled GEMM at prefill shapes
    // (M even, N even). Decode (M=1) and any odd dim take the portable nrc==1
    // path below — the same "num_rows_per_vec_dot falls to 1" guard ggml uses
    // (ggml-cpu.c:1432). Null off i8mm hardware / when VT_CPU_QUANT_MMLA=0 /
    // for q3_K,q5_K (no upstream mmla).
    const int64_t n = b.shape[0];
    const VecDotFn mmla = QuantMmlaVecDot(b.dtype);
    if (mmla != nullptr && m % 2 == 0 && n % 2 == 0) {
      ParallelForRows(CurrentThreadpool(), n / 2, [&](int64_t jp0, int64_t jp1) {
        QuantChunkMmla(out, act, act_row_bytes, b, mmla, k, jp0, jp1);
      });
      return;
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

// GROUPED keep-quant GEMM (kMatmulBTQuantGrouped). out[P,N], act[P,K],
// weight[E*N,K] block-quant, expert_ids[P] i32. Runs the SAME kMatmulBTQuant
// kernel once per group over the [expert_ids[p]*N, +N) weight row-slice — so it is
// BYTE-IDENTICAL to the per-expert kMatmulBTQuant path the DeepSeek-V4 forward
// used before. The CUDA provider fuses this into one launch; this CPU provider is
// the correctness reference (and the tiny-model doctest's grouped path).
//
// DTYPE CONTRACT (P0 fix, 2026-08-06). `vt::MatmulBTQuantGrouped` accepts ANY
// float activation and an f32/bf16 output (ops.cpp:220-221), and the CUDA
// provider honours all three activation dtypes (cuda_quant_dot.cu:1868-1871).
// The row views below must therefore be addressed through `act.dtype` /
// `out.dtype`, NOT assumed f32: this kernel used to advance a `float*` by
// `act.stride[0]` and declare the row `kF32` regardless, so a bf16/f16
// activation was BOTH mis-strode (2x too far per row) and mis-decoded (two
// bf16 lanes read as one f32). Every caller and test happened to pass f32/f32
// until qwen3_5's grouped MoE (`KqGrouped`, bf16 activations, commit b4f5610a)
// became the first non-f32 caller, which is why the CUDA gate was byte-exact
// while CPU decoded token-0 garbage. Gated per dtype in
// tests/vt/test_ops_quant_dot.cpp.
//
// The `repacked`/`q8_0_aligned` layout markers are carried onto the slice for
// the same reason `KqResidentSlice` (qwen3_5.cpp) inherits them: dropping
// `repacked` makes `kMatmulBTQuant` read i8mm-interleaved bytes as plain q8_0,
// which is the CIQ-G7 all-zero-token failure mode (state.md 2026-07-23).
// `Tensor::Slice` throws on a repacked weight, so the slice is built by hand.
void MatmulBTQuantGroupedKernel(Queue& q, Tensor& out, const Tensor& act,
                                const Tensor& weight, const Tensor& expert_ids) {
  const int64_t P = out.shape[0];
  const int64_t N = out.shape[1];
  const int64_t K = act.shape[1];
  const int64_t act_rows = act.shape[0];
  const bool bcast = (act_rows == 1 && P > 1);  // broadcast a shared hidden across experts
  // Prefill gather: activation is [T, H], output is [P, H] where P = T * top_k.
  // Each output row p reads activation row p / top_k. Infer top_k from shapes.
  // Mirrors the Vulkan shader's gather_k parameter (vulkan_ops.cpp:2283-2286).
  const bool gather = (!bcast && act_rows > 1 && act_rows < P &&
                        P % act_rows == 0);
  const int64_t gather_k = gather ? P / act_rows : 1;
  const size_t row_bytes = RowSizeBytes(weight.dtype, K);
  const size_t act_elem = SizeOf(act.dtype);
  const size_t out_elem = SizeOf(out.dtype);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  for (int64_t p = 0; p < P; ++p) {
    const int64_t e = eids[p];
    const int64_t act_row_idx = bcast ? 0 : (p / gather_k);
    Tensor a_row = Tensor::Contiguous(
        static_cast<uint8_t*>(act.data) +
            static_cast<size_t>(act_row_idx) *
                static_cast<size_t>(act.stride[0]) * act_elem,
        act.dtype, act.device, {1, K});
    Tensor o_row = Tensor::Contiguous(
        static_cast<uint8_t*>(out.data) +
            static_cast<size_t>(p) * static_cast<size_t>(N) * out_elem,
        out.dtype, out.device, {1, N});
    Tensor w{};
    w.data = static_cast<uint8_t*>(weight.data) + static_cast<size_t>(e) * N * row_bytes;
    w.dtype = weight.dtype;
    w.device = weight.device;
    w.repacked = weight.repacked;
    w.q8_0_aligned = weight.q8_0_aligned;
    w.rank = 2;
    w.shape[0] = N;
    w.shape[1] = K;
    w.stride[0] = K;
    w.stride[1] = 1;
    MatmulBTQuantKernel(q, o_row, a_row, w);
  }
}

// SHARED fused MoE gate+up+SwiGLU keep-quant (kMoeGateUpSwiGLUGrouped) — the CPU
// GOLDEN. This is the Tier-0 BYTE-EXACT COMPOSITE the CUDA fused kernel is defined
// against: two grouped keep-quant GEMMs (gate, up) into f32 temporaries via the
// SAME MatmulBTQuantGroupedKernel above, then the elementwise clamped-SwiGLU
//   gate = min(g, limit); up = clamp(u, -limit, limit); out = gate·sigmoid(gate)·up
// exactly matching QuantDotGemmGroupedFusedSwiGLUKernel (α=1, β=0). Because the
// grouped GEMM already folds the weight FinalFactor into g/u, no extra scale is
// applied here. limit=+inf → plain silu(g)·u (standard SwiGLU MLP).
void MoeGateUpSwiGLUGroupedKernel(Queue& q, Tensor& out, const Tensor& act,
                                  const Tensor& gate_w, const Tensor& up_w,
                                  const Tensor& expert_ids, float limit) {
  const int64_t P = out.shape[0];
  const int64_t N = out.shape[1];
  std::vector<float> g(static_cast<size_t>(P) * N);
  std::vector<float> u(static_cast<size_t>(P) * N);
  Tensor gt = Tensor::Contiguous(g.data(), DType::kF32, out.device, {P, N});
  Tensor ut = Tensor::Contiguous(u.data(), DType::kF32, out.device, {P, N});
  MatmulBTQuantGroupedKernel(q, gt, act, gate_w, expert_ids);
  MatmulBTQuantGroupedKernel(q, ut, act, up_w, expert_ids);
  for (size_t i = 0; i < g.size(); ++i) {
    const float gate = std::fmin(g[i], limit);
    const float up = std::fmin(std::fmax(u[i], -limit), limit);
    StoreOutF32(out, static_cast<int64_t>(i),
                gate * (1.0F / (1.0F + std::exp(-gate))) * up);
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantKernel)));
    RegisterOp(
        OpId::kMatmulBTQuantGrouped, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<MatmulBTQuantGroupedFn>(&MatmulBTQuantGroupedKernel)));
    RegisterOp(OpId::kMoeGateUpSwiGLUGrouped, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<MoeGateUpSwiGLUGroupedFn>(&MoeGateUpSwiGLUGroupedKernel)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cpu
