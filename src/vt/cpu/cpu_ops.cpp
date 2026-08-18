// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// EXCEPT the parallel dispatch (QUANT-GGUF-CPU-THREADPOOL): the GEMM chunk
// policy and the row/batch chunking are ported 1:1 from llama.cpp (local fork)
// ggml/src/ggml-cpu/ggml-cpu.c:1155-1443 and ggml-cpu/ops.cpp:9070-9126 @
// 237ad9b96 — see cpu_threadpool.h and the per-kernel anchors below. Every
// kernel keeps its exact per-element math and per-output sequential reduction
// order; parallelism partitions OUTPUT elements only, so results are
// bit-identical to single-thread by construction (spec § Dispatch behavior).
#include "vt/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "cpu_matmul_elem.h"
#include "cpu_threadpool.h"
#include "vt/unaligned.h"

namespace vt::cpu {
namespace {

// Row/batch-chunked dispatch through the process pool (or the test-swapped
// pool). body(r0, r1) produces output rows [r0, r1) exactly once each.
inline void ForRows(int64_t nr, const std::function<void(int64_t, int64_t)>& body) {
  ParallelForRows(CurrentThreadpool(), nr, body);
}

float LoadF32(const Tensor& t, int64_t elem_offset) {
  const auto* address = static_cast<const uint8_t*>(t.data) +
                        elem_offset * SizeOf(t.dtype);
  switch (t.dtype) {
    case DType::kF32: return LoadUnaligned<float>(address);
    case DType::kF16: return F16ToF32(LoadUnaligned<uint16_t>(address));
    case DType::kBF16: return BF16ToF32(LoadUnaligned<uint16_t>(address));
    default: VT_CHECK(false, "LoadF32: unsupported dtype"); return 0.0f;
  }
}

// Mirror of LoadF32 for outputs: reduced-width formats round to storage dtype.
void StoreF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[elem_offset] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "StoreF32: unsupported dtype");
  }
}

// GEMM chunk worker — 16x16 block tiling inside a chunk, ported from
// ggml_compute_forward_mul_mat_one_chunk (ggml-cpu.c:1155-1243; empty-chunk
// yield :1181-1184, blck_0/blck_1 = 16 :1192-1194). ggml's vec_dot per output
// element is our per-element K loop: byte-identical accumulation (sequential
// over K, f32, -ffp-contract=off pinned) to the pre-threadpool kernels.
// ir0 indexes output COLUMNS j (ggml nr0 = src0/weight rows = N), ir1 indexes
// output ROWS i (ggml nr1 = src1 rows = M). kBT selects the [N,K] row-major
// weight orientation (MatmulBT) vs [K,N] (Matmul).
template <bool kBT>
void MatmulOneChunkRef(Tensor& out, const Tensor& a, const Tensor& b, int64_t k, int64_t n,
                       int64_t ir0_start, int64_t ir0_end, int64_t ir1_start, int64_t ir1_end) {
  // MLA campaign W6: the activation may be ROW-STRIDED (a column slice of a
  // wider buffer — see vt::MatmulBT). For a contiguous activation `a_rs == k`,
  // so the offsets are integer-identical to the pre-W6 `i * k + p` form and
  // every existing model is bit-identical by construction.
  const int64_t a_rs = a.stride[0];
  // threads with no work simply yield
  if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
    return;
  }

  // block-tiling attempt
  const int64_t blck_0 = 16;
  const int64_t blck_1 = 16;

  for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
    for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
      for (int64_t i = iir1; i < iir1 + blck_1 && i < ir1_end; ++i) {
        for (int64_t j = iir0; j < iir0 + blck_0 && j < ir0_end; ++j) {
          float acc = 0.0f;
          for (int64_t p = 0; p < k; ++p) {
            acc += LoadF32(a, i * a_rs + p) * LoadF32(b, kBT ? j * k + p : p * n + j);
          }
          StoreF32(out, i * n + j, acc);
        }
      }
    }
  }
}

// Byte offset of element `off` of an ELEMENTWISE tensor.
inline const void* ElemPtr(const Tensor& t, int64_t off) {
  return static_cast<const uint8_t*>(t.data) + static_cast<size_t>(off) * SizeOf(t.dtype);
}

// Specialized/vectorized elementwise GEMM chunk (row `CPU-ELEM-GEMM`,
// .agents/specs/cpu-elementwise-gemm.md). Structurally identical to
// MatmulOneChunkRef — same 16x16 tile from ggml_compute_forward_mul_mat_one_chunk
// (ggml-cpu.c:1155-1243), same output set, same strictly sequential f32
// accumulation per output element — with two defects removed:
//   1. the per-ELEMENT `LoadF32` dtype switch is hoisted out of the K loop
//      (the activation row is widened to f32 ONCE per 16-row tile; the weight
//      dtype is resolved once per chunk into a typed micro-kernel), and
//   2. the single serial accumulator becomes 16 independent ones, vectorized
//      ACROSS OUTPUT COLUMNS so no reduction is reassociated.
// Both are bit-exact by construction; tests/vt/test_ops_matmul_elem.cpp
// asserts equality against MatmulOneChunkRef byte-for-byte.
template <bool kBT>
void MatmulOneChunk(Tensor& out, const Tensor& a, const Tensor& b, int64_t k, int64_t n,
                    int64_t ir0_start, int64_t ir0_end, int64_t ir1_start, int64_t ir1_end) {
  if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
    return;
  }
  ElemKind bk;
  ElemKind ak;
  if (!ElemKindOf(b.dtype, &bk) || !ElemKindOf(a.dtype, &ak) || k <= 0 || ElemGemmUseRef()) {
    MatmulOneChunkRef<kBT>(out, a, b, k, n, ir0_start, ir0_end, ir1_start, ir1_end);
    return;
  }
  const ElemGemmTierTable& tier = ElemGemmTier();
  const int bi = static_cast<int>(bk);
  const int64_t a_rs = a.stride[0];
  const int64_t blck_0 = kElemLanes;
  const int64_t blck_1 = 16;

  // Widened activation rows for the current 16-row tile, reused across every
  // column block of the chunk. Thread-local so the buffer is allocated once
  // per worker for the process lifetime (no per-chunk allocation).
  static thread_local std::vector<float> af;

  for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
    const int64_t i_hi = std::min(iir1 + blck_1, ir1_end);
    const int64_t nrows = i_hi - iir1;
    af.resize(static_cast<size_t>(nrows * k));
    for (int64_t i = iir1; i < i_hi; ++i) {
      WidenRowToF32(a.dtype, ElemPtr(a, i * a_rs), k, af.data() + (i - iir1) * k);
    }
    // M blocking applies to BOTH orientations. It used to be gated on kBT, so
    // the [K,N] path always ran mr=1 and re-read the whole weight tile once per
    // activation row (a 131-row activation read it 131 times). Each family is
    // guarded on its own function pointer because a tier may provide one and
    // not the other (the portable tier has no btm, having no transpose to
    // amortize, but its nkm still amortizes the weight load).
    const ElemNkMFn nkm_fn = tier.nkm[bi];
    const int mr = (kBT ? (tier.btm[bi] != nullptr) : (nkm_fn != nullptr)) ? tier.mr : 1;
    for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
      const int64_t j_hi = std::min(iir0 + blck_0, ir0_end);
      int64_t i = iir1;
      // M-blocked fast path: `mr` activation rows share one weight load +
      // transpose per column block (see ElemBtMFn).
      if (j_hi - iir0 == blck_0 && mr > 1) {
        float accm[kElemLanes * 8];
        for (; i + mr <= i_hi; i += mr) {
          if (kBT) {
            tier.btm[bi](af.data() + (i - iir1) * k, k, ElemPtr(b, iir0 * k), k, accm);
          } else {
            nkm_fn(af.data() + (i - iir1) * k, k, ElemPtr(b, iir0), k, n, accm);
          }
          for (int r = 0; r < mr; ++r) {
            for (int64_t j = iir0; j < j_hi; ++j) {
              StoreF32(out, (i + r) * n + j, accm[r * kElemLanes + (j - iir0)]);
            }
          }
        }
      }
      for (; i < i_hi; ++i) {
        const float* arow = af.data() + (i - iir1) * k;
        float acc[kElemLanes];
        if (j_hi - iir0 == blck_0) {
          if (kBT) {
            tier.bt[bi](arow, ElemPtr(b, iir0 * k), k, acc);
          } else {
            tier.nk[bi](arow, ElemPtr(b, iir0), k, n, acc);
          }
        } else {
          // Ragged column tail: the scalar form, identical accumulation order.
          for (int64_t j = iir0; j < j_hi; ++j) {
            float s = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
              s += arow[p] * LoadF32(b, kBT ? j * k + p : p * n + j);
            }
            acc[j - iir0] = s;
          }
        }
        for (int64_t j = iir0; j < j_hi; ++j) {
          StoreF32(out, i * n + j, acc[j - iir0]);
        }
      }
    }
  }
}

// GEMM chunking policy + atomic work stealing, ported from
// ggml_compute_forward_mul_mat (ggml-cpu.c:1245-1443): thread 0 seeds the
// steal cursor at nth and a barrier publishes it (:1350-1355); chunk_size 16,
// 64 for vector shapes (:1388-1393); nchunk0 x nchunk1 grid (:1398-1399);
// re-chunk per-thread when the grid is < nth*4 or NUMA (:1404-1408, IsNuma()
// stubbed false); each thread starts at chunk ith then steals via the atomic
// cursor (:1415-1442). num_rows_per_vec_dot is 1 for our scalar dot (no mmla).
// VT_CPU_MATMUL_STEAL: same-binary A/B for the decode-shape chunk policy above.
// Read once; default OFF keeps the ggml-mirrored behaviour byte-for-byte.
bool MatmulStealEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_CPU_MATMUL_STEAL");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

template <bool kBT>
void MatmulChunked(Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1];
  const int64_t n = kBT ? b.shape[0] : b.shape[1];
  // ggml nr0 = ne0 (dst dim0 = weight rows) -> our N; nr1 = ne1*ne2*ne3
  // (src1 rows) -> our M.
  const int64_t nr0 = n;
  const int64_t nr1 = m;

  Threadpool& tp = CurrentThreadpool();
  tp.Run([&](int ith, int nth) {
    if (ith == 0) {
      // Every thread starts at ith, so the first unprocessed chunk is nth.
      tp.ChunkSet(nth);
    }

    tp.Barrier();

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
      chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap
    // the whole plan. Re-chunk it by thread.
    //
    // VT_CPU_MATMUL_STEAL=1 (default OFF) SKIPS this collapse. Rationale, and
    // why it is an opt-in A/B rather than a new default: the collapse is a
    // faithful port of ggml (ggml-cpu.c:1404-1408), so changing it by default
    // would be an unmirrored deviation. But it has a cost this project has not
    // measured. At decode shapes it rewrites the grid to exactly `nth` chunks
    // and the loop below then breaks after one chunk each, so a 20-thread
    // decode GEMV becomes 20 EQUAL STATIC chunks gated by the slowest core,
    // with the self-balancing steal cursor switched off. On a heterogeneous
    // core complex (GB10 mixes core classes) that is a straggler trap.
    // Skipping the collapse keeps the fine-grained grid and lets stealing
    // balance it. Byte-identity is unaffected either way: every output's
    // reduction is local and sequential over K (see MatmulOneChunk), so which
    // thread computes which output changes nothing.
    if ((nchunk0 * nchunk1 < nth * 4 && !MatmulStealEnabled()) || IsNuma()) {
      nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by weight rows (N)
      nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows (M)
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int64_t current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
      const int64_t ith0 = current_chunk % nchunk0;
      const int64_t ith1 = current_chunk / nchunk0;

      const int64_t ir0_start = dr0 * ith0;
      const int64_t ir0_end = std::min(ir0_start + dr0, nr0);

      const int64_t ir1_start = dr1 * ith1;
      const int64_t ir1_end = std::min(ir1_start + dr1, nr1);

      MatmulOneChunk<kBT>(out, a, b, k, n, ir0_start, ir0_end, ir1_start, ir1_end);

      // Same switch: the early break is what makes the collapsed grid a pure
      // static partition. With stealing enabled we keep pulling chunks.
      if (nth >= nchunk0 * nchunk1 && !MatmulStealEnabled()) {
        break;
      }

      current_chunk = tp.ChunkAdd(1);
    }
  });
}

void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  MatmulChunked<false>(out, a, b);
}

// b is the torch Linear weight [N,K] row-major (see vt::MatmulBT); identical
// accumulation order to MatmulKernel (sequential over K), so on CPU the two
// orientations are bit-identical for the same logical weight.
void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  // KERNEL-GEMM-CPU-TILED lever 2: a loader-repacked weight keeps its [N,K]
  // SHAPE (so this op's contract is unchanged for callers) but its BYTES are
  // [K,N]. Present it as the [K,N] tensor it literally is and take the
  // transpose-free path. Byte-identical by the same argument the comment above
  // MatmulBTKernel already states: both orientations accumulate each output
  // over K in strict increasing order.
  if (b.elem_kn_repacked) {
    Tensor bkn = b;
    bkn.shape[0] = b.shape[1];   // K
    bkn.shape[1] = b.shape[0];   // N
    bkn.stride[0] = b.shape[0];  // one [K,N] row is N elements
    bkn.stride[1] = 1;
    MatmulChunked<false>(out, a, bkn);
    return;
  }
  MatmulChunked<true>(out, a, b);
}

// vt::BatchedMatmul (`torch.bmm`) CPU reference — out[G,M,N] = a[G,M,K] @
// b[G,K,N]. Sequential f32 accumulation over K, exactly like MatmulOneChunk's
// per-element dot, so the reference and the cuBLASLt CUDA path share the same
// numeric contract (f32 accumulate, round on store). Every operand is addressed
// through its STRIDES: the MLA absorption call sites (mla_attention.py:789,
// :1034) pass transposed views whose batch axis is not the outermost storage
// axis. Parallelized over the flattened (batch, row) output space, which leaves
// each output element's K reduction on one thread — bit-identical to a serial
// run and run-to-run reproducible.
void BatchedMatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t g = out.shape[0], m = out.shape[1], n = out.shape[2];
  const int64_t k = a.shape[2];
  const int64_t rows = g * m;
  if (rows == 0 || n == 0) return;
  ForRows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t bi = r / m, i = r % m;
      const int64_t a_row = bi * a.stride[0] + i * a.stride[1];
      const int64_t b_base = bi * b.stride[0];
      const int64_t o_row = bi * out.stride[0] + i * out.stride[1];
      for (int64_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int64_t p = 0; p < k; ++p) {
          acc += LoadF32(a, a_row + p) * LoadF32(b, b_base + p * b.stride[1] + j);
        }
        StoreF32(out, o_row + j, acc);
      }
    }
  });
}

// vt::ConcatMlaNopeRope CPU reference — the scalar, width-generic form of
// upstream's `ConcatMLAQKernel` (csrc/libtorch_stable/concat_mla_q.cuh) and of
// the `_concat_k_nope_k_pe` slice assignments (mla_attention.py:2085-2090).
// Pure copy: no arithmetic, so it is exact for every dtype. `rope.shape[1] == 1`
// with more output heads is the broadcast form the prefill K concat needs.
void ConcatMlaNopeRopeKernel(Queue&, Tensor& out, const Tensor& nope, const Tensor& rope) {
  const int64_t tokens = out.shape[0], heads = out.shape[1];
  const int64_t dn = nope.shape[2], dr = rope.shape[2];
  const bool rope_broadcast = rope.shape[1] == 1 && heads > 1;
  ForRows(tokens, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      for (int64_t h = 0; h < heads; ++h) {
        const int64_t o = t * out.stride[0] + h * out.stride[1];
        const int64_t n = t * nope.stride[0] + h * nope.stride[1];
        const int64_t r =
            t * rope.stride[0] + (rope_broadcast ? 0 : h) * rope.stride[1];
        for (int64_t d = 0; d < dn; ++d) StoreF32(out, o + d, LoadF32(nope, n + d));
        for (int64_t d = 0; d < dr; ++d) StoreF32(out, o + dn + d, LoadF32(rope, r + d));
      }
    }
  });
}

void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  // Row-chunked over tokens (ops.cpp:9070-9126 pattern); each row's f32
  // variance reduction stays sequential on one thread — bit-identical.
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const int64_t rbase = i * h;
    float sumsq = 0.0f;
    for (int64_t j = 0; j < h; ++j) {
      float v = LoadF32(x, i * h + j);
      if (residual) {
        v += LoadF32(*residual, rbase + j);   // add in f32
        StoreF32(*residual, rbase + j, v);     // new residual stream (rounds to its dtype)
        v = LoadF32(*residual, rbase + j);     // re-read rounded value (bf16-faithful)
      }
      sumsq += v * v;
    }
    float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + args.eps);
    for (int64_t j = 0; j < h; ++j) {
      float v = residual ? LoadF32(*residual, rbase + j) : LoadF32(x, i * h + j);
      float wj = LoadF32(w, j);
      if (args.gemma) wj += 1.0f;
      StoreF32(out, i * h + j, v * inv * wj);
    }
  }
  });
}

void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      float gate = LoadF32(x, i * 2 * d + j);
      float up = LoadF32(x, i * 2 * d + d + j);
      float silu = gate / (1.0f + std::exp(-gate));
      StoreF32(out, i * d + j, silu * up);
    }
  }
  });
}

// Gemma GeGLU: out = gelu_tanh(gate) * up. gelu_tanh(g) = 0.5*g*(1 + tanh(
// sqrt(2/pi)*(g + 0.044715*g^3))) — the exact gelu_pytorch_tanh, computed in f32.
void GeluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      float g = LoadF32(x, i * 2 * d + j);
      float up = LoadF32(x, i * 2 * d + d + j);
      float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
      float gelu = 0.5f * g * (1.0f + std::tanh(inner));
      StoreF32(out, i * d + j, gelu * up);
    }
  }
  });
}

// out[i] = x[i] * scalar (f32 compute, out-dtype store).
void MulScalarKernel(Queue&, Tensor& out, const Tensor& x, double scalar) {
  const int64_t n = x.Numel();
  const float s = static_cast<float>(scalar);
  ForRows(n, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) StoreF32(out, i, LoadF32(x, i) * s);
  });
}

// out[i] = cap * tanh(x[i] / cap) (f32 compute, out-dtype store). The Gemma-2
// final logit soft-cap (gemma2.py:344-345).
// Mirrors torch: logits.div_(cap).tanh_().mul_(cap) — division, not reciprocal.
void SoftCapKernel(Queue&, Tensor& out, const Tensor& x, double cap) {
  const int64_t n = x.Numel();
  const float c = static_cast<float>(cap);
  ForRows(n, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) StoreF32(out, i, c * std::tanh(LoadF32(x, i) / c));
  });
}

void MoeSiluMulKernel(Queue&, Tensor& out, const Tensor& gate, const Tensor& up) {
  const int64_t n = out.Numel();
  // Elementwise: partition the flat output range.
  ForRows(n, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const float g = LoadF32(gate, i);
    const float silu = g / (1.0f + std::exp(-g));
    StoreF32(out, i, silu * LoadF32(up, i));
  }
  });
}

// The NON-GATED MoE activation: out[i] = relu(x[i])^2, the whole epilogue of a
// NemotronH expert (nemotron_h.py:227 -> MoEActivation.RELU2_NO_MUL). Mirrors
// vLLM's relu_squared_kernel (csrc/libtorch_stable/activation_kernels.cu:673-678)
// EXACTLY in dtype order: widen to f32, clamp at zero in f32, square in f32, and
// round ONCE on the store. LoadF32/StoreF32 are that widen/round pair, so a bf16
// input with an f32 output keeps the full f32 square (no intermediate narrowing).
void MoeRelu2Kernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = out.Numel();
  ForRows(n, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const float f = LoadF32(x, i);
    const float v = f > 0.0f ? f : 0.0f;
    StoreF32(out, i, v * v);
  }
  });
}

// --- TRUE W4A4 (fp4xfp4) helpers + kernels (notes §7). Self-contained fp8/fp4
// codec (vt does not depend on vllm), bit-matching vllm::F8E4M3ToF32 /
// F32ToF8E4M3 / CastToFp4 / kE2M1Lut so the op equals vllm::RunNvfp4Emulation.
constexpr float kFp4Max = 6.0F;    // E2M1 max magnitude
constexpr float kFp8Max = 448.0F;  // fp8-e4m3fn max finite
constexpr float kE2M1[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

inline float ClampF(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float RecipF(float x) { return x == 0.0F ? 0.0F : 1.0F / x; }

// IEEE fp8-e4m3fn byte -> f32 (bit-matches vllm::F8E4M3ToF32).
float Fp8ToF32(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0F : 1.0F;
  if (exp == 0xFU && mant == 0x7U) return std::numeric_limits<float>::quiet_NaN();
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0F / 512.0F));
  const float mantissa = 1.0F + static_cast<float>(mant) * (1.0F / 8.0F);
  return sm * std::ldexp(mantissa, static_cast<int>(exp) - 7);
}

// f32 -> fp8-e4m3fn byte, round-to-nearest-even saturating (bit-matches
// vllm::F32ToF8E4M3).
uint8_t F32ToFp8(float f) {
  if (std::isnan(f)) return 0x7FU;
  const uint8_t sign = std::signbit(f) ? 0x80U : 0x00U;
  const float a = std::fabs(f);
  if (!std::isfinite(a) || a >= kFp8Max) return static_cast<uint8_t>(sign | 0x7EU);
  if (a == 0.0F) return sign;
  int e2 = 0;
  const float frac = std::frexp(a, &e2);
  int exp_field = (e2 - 1) + 7;
  if (exp_field <= 0) {
    const double qd = static_cast<double>(a) * 512.0;
    const int qi = static_cast<int>(std::nearbyint(qd));
    if (qi <= 0) return sign;
    if (qi < 8) return static_cast<uint8_t>(sign | static_cast<uint8_t>(qi));
    return static_cast<uint8_t>(sign | (1U << 3));
  }
  const double sig = static_cast<double>(frac) * 2.0;
  int mi = static_cast<int>(std::nearbyint(sig * 8.0));
  if (mi == 16) {
    mi = 8;
    exp_field += 1;
  }
  const int mant = mi - 8;
  if (exp_field > 15 || (exp_field == 15 && mant >= 7)) {
    return static_cast<uint8_t>(sign | 0x7EU);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint8_t>(exp_field) << 3) |
                              static_cast<uint8_t>(mant));
}

// --- Static per-tensor FP8 W8A8 (VT-FP8-W8A8-CPU-ARM, #468). The CPU arm of the
// path vLLM's ModelOptFp8LinearMethod runs: a static per-tensor activation quant
// followed by a per-tensor fp8 GEMM. It exists so the fp8 seam is reachable, and
// therefore testable, without a GPU.

// QuantFp8Static CPU kernel — mirror of vLLM's static_scaled_fp8_quant
// (csrc/quantization/w8a8/fp8/common.cuh:58-77 `scaled_fp8_conversion`):
//   x = val * scale;  r = fmaxf(-448, fminf(x, 448));  hardware RNE convert
// with the RECIPROCAL formed ONCE outside the loop, exactly as upstream forms it
// (csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31 `1.0f / scale[...]`)
// and exactly as our CUDA kernel does (cuda_matmul_fp8_cutlass.cu: `const float
// inv = 1.0f / input_scale;` then `LoadIn(x, i) * inv`). It is a MULTIPLY BY THE
// RECIPROCAL, not a divide: the two differ by up to one f32 ulp before the fp8
// round, and near an e4m3 tie that ulp changes the emitted byte.
//
// F32ToFp8 supplies both remaining halves: it saturates (|a| >= 448 -> 0x7E,
// which IS the encoding of 448, so clamp-then-convert and saturating-convert
// coincide because 448 is the largest finite e4m3fn value) and it rounds to
// nearest-even. The scale is per-TENSOR — upstream collapses the per-shard
// input_scale to one scalar with `.max()` (modelopt.py:528) and then treats
// `scale.numel() == 1` as a single group spanning the whole tensor
// (common.cu:204-210). LoadF32 widens a bf16 x to f32 BEFORE the multiply, as
// the CUDA kernel's LoadIn overload does, so both backends round at one point.
void QuantFp8StaticKernel(Queue&, Tensor& out_fp8, const Tensor& x, float input_scale) {
  const int64_t n = x.shape[0] * x.shape[1];
  const float inv_scale = 1.0F / input_scale;
  uint8_t* op = out_fp8.Ptr<uint8_t>();
  ForRows(n, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) op[i] = F32ToFp8(LoadF32(x, i) * inv_scale);
  });
}

// --- Block-wise FP8 (VT-QUANT-FP8-GROUP, #1189 M1). QuantFp8Group CPU kernel:
// the DYNAMIC per-token, per-group activation quant.
//
// MIRROR OF THE KERNEL THAT ACTUALLY EXECUTES, which is the C++ custom op and
// not the Triton kernel. `per_token_group_quant_fp8` calls
// `torch.ops._C.per_token_group_fp8_quant` and RETURNS whenever the platform is
// CUDA-alike and the input is contiguous
// (vllm/model_executor/layers/quantization/utils/fp8_utils.py:635-650), so the
// Triton kernel below it never runs there. The executing kernel is
// csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:
//   :47  float local_absmax = eps                  eps SEEDS the reduction
//   :53  fmaxf(local_absmax, fabsf((float)src))
//   :68  float y_s = local_absmax / max_8bit       a DIVIDE
//   :85  fminf(fmaxf((float)src / y_s, min_8bit), max_8bit)   a DIVIDE
//   :86  DST_DTYPE(q)                              hardware e4m3 RNE
//
// TWO DIVIDES, DELIBERATELY, and this is the opposite of QuantFp8StaticKernel
// forty lines above. That kernel multiplies by a hoisted reciprocal because
// upstream ships the reciprocal there (common.cuh:62, with `1.0f / scale`
// formed by the caller at common.cu:31). Here upstream ships a divide, and the
// scale changes per group, so there is no loop-invariant reciprocal to hoist in
// the first place. The Triton fallback's `_absmax * (1.0 / fp8_max)`
// (fp8_utils.py:145) differs by up to one f32 ulp and carries an upstream
// comment saying so. Near an e4m3 tie that ulp changes the emitted byte.
// tests/vt/test_ops_quant_fp8_group_cpu.cpp G1 compares BYTES for this reason;
// upstream's own test compares values at rtol=0.15 and cannot see it.
//
// eps is the reduction's INITIAL value rather than a clamp afterwards. The two
// are numerically identical, and writing it upstream's way makes it visible
// that an all-zero group yields y_s = 1e-10/448 instead of dividing by zero.
//
// LoadF32 widens a bf16 x to f32 before the absolute value and before the
// divide, matching `fabsf(static_cast<float>(src))` at :53 and
// `static_cast<float>(src) / y_s` at :85, so a bf16 input rounds at one point.
// Parallel over ROWS: each row's groups are independent, and the reduction
// order inside a group is fixed and sequential, so the result does not depend
// on the thread count.
void QuantFp8GroupKernel(Queue&, Tensor& out_fp8, Tensor& out_scale, const Tensor& x,
                         int group_size) {
  const int64_t m = x.shape[0], k = x.shape[1];
  const int64_t groups = k / group_size;
  constexpr float kEps = 1e-10F;      // fp8_utils.py:570, the only value any
                                      // upstream call site passes
  constexpr float kFp8MaxV = 448.0F;  // quant_utils.py:27-35 finfo(e4m3fn).max
  constexpr float kFp8MinV = -448.0F;
  uint8_t* op = out_fp8.Ptr<uint8_t>();
  float* sp = out_scale.Ptr<float>();
  ForRows(m, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      for (int64_t g = 0; g < groups; ++g) {
        const int64_t base = r * k + g * group_size;
        float amax = kEps;
        for (int64_t i = 0; i < group_size; ++i)
          amax = std::fmax(amax, std::fabs(LoadF32(x, base + i)));
        const float y_s = amax / kFp8MaxV;
        sp[r * groups + g] = y_s;
        for (int64_t i = 0; i < group_size; ++i)
          op[base + i] =
              F32ToFp8(std::fmin(std::fmax(LoadF32(x, base + i) / y_s, kFp8MinV), kFp8MaxV));
      }
    }
  });
}

// MatmulFp8Cutlass CPU kernel: out[m,n] = alpha * Sum_k f8val(a[m,k])*f8val(b[n,k]),
// f32 accumulate, ONE folded alpha (= input_scale*weight_scale — our recorded
// deviation from upstream's two epilogue scalars, see include/vt/ops.h).
//
// A CORRECTNESS REFERENCE, NOT A PERFORMANCE PATH. It is a naive triple loop; it
// makes no speed claim and nothing routes a production model through it. Its
// purpose is that the fp8 GEMM seam resolves on a CPU queue so the surrounding
// wiring can be gated without a GPU (#468).
//
// It is deliberately NOT a bit-mirror of the CUDA GEMM and does not claim to be:
// the CUDA arm reduces K in tensor-core order and rounds its epilogue through
// bf16, so the two agree to fp8/bf16 tolerance. Only the QUANT half above carries
// a bit-exactness claim. Shaped like MatmulNvfp4Fp4Kernel: the A row is decoded
// once per M and reused across N.
void MatmulFp8CutlassKernel(Queue&, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                            float alpha) {
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  const auto* ap = a_fp8.Ptr<uint8_t>();
  const auto* bp = b_fp8.Ptr<uint8_t>();
  ForRows(m, [&](int64_t r0, int64_t r1) {
    std::vector<float> arow(static_cast<size_t>(k));
    for (int64_t i = r0; i < r1; ++i) {
      for (int64_t kk = 0; kk < k; ++kk)
        arow[static_cast<size_t>(kk)] = Fp8ToF32(ap[i * k + kk]);
      for (int64_t col = 0; col < n; ++col) {
        float acc = 0.0F;
        for (int64_t kk = 0; kk < k; ++kk)
          acc += arow[static_cast<size_t>(kk)] * Fp8ToF32(bp[col * k + kk]);
        StoreF32(out, i * n + col, alpha * acc);
      }
    }
  });
}

// Fused fp8 RMSNorm -> static per-tensor quant (mirror vLLM Inductor
// fused_add_rms_norm_static_fp8_quant, rms_quant_fusion.py:124). Same reduction
// order as RmsNormKernel; the fp8 is taken from the SAME bf16-rounded normed value
// the split RmsNorm(bf16)+QuantFp8Static path quantizes (bf16-intermediate form),
// so the two are bit-identical. out_bf16 (optional) is the normed activation in
// bf16 (for a coexisting bf16 consumer, e.g. GDN in_proj_a/b). CUDA has the hot
// path; this CPU kernel keeps the op available on the host backend.
void RmsNormQuantFp8Kernel(Queue&, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                           const Tensor& w, const RmsNormArgs& args, Tensor* residual,
                           float input_scale) {
  const int64_t t = x.shape[0], h = x.shape[1];
  const float inv_scale = 1.0F / input_scale;
  uint8_t* op = out_fp8.Ptr<uint8_t>();
  uint16_t* bp = out_bf16 == nullptr ? nullptr : out_bf16->Ptr<uint16_t>();
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const int64_t rbase = i * h;
    float sumsq = 0.0F;
    for (int64_t j = 0; j < h; ++j) {
      float v = LoadF32(x, rbase + j);
      if (residual) {
        v += LoadF32(*residual, rbase + j);   // add in f32
        StoreF32(*residual, rbase + j, v);     // new residual stream (rounds to its dtype)
        v = LoadF32(*residual, rbase + j);     // re-read rounded value (bf16-faithful)
      }
      sumsq += v * v;
    }
    const float inv = 1.0F / std::sqrt(sumsq / static_cast<float>(h) + args.eps);
    for (int64_t j = 0; j < h; ++j) {
      float v = residual ? LoadF32(*residual, rbase + j) : LoadF32(x, rbase + j);
      float wj = LoadF32(w, j);
      if (args.gemma) wj += 1.0F;
      // bf16-intermediate: round the normed value to bf16 (as RmsNorm's bf16 store),
      // then quant from that bf16 (as QuantFp8Static's bf16 load).
      const uint16_t nb = F32ToBF16(v * inv * wj);
      if (bp) bp[rbase + j] = nb;
      op[rbase + j] = F32ToFp8(BF16ToF32(nb) * inv_scale);
    }
  }
  });
}

// f32 -> E2M1 nibble (bit-matches vllm::CastToFp4 + Fp4ToNibble). Input pre-scaled.
uint8_t F32ToFp4Nibble(float x) {
  const float a = std::fabs(x);
  uint8_t idx = 7;  // 6.0
  if (a <= 0.25F) idx = 0;
  else if (a < 0.75F) idx = 1;
  else if (a <= 1.25F) idx = 2;
  else if (a < 1.75F) idx = 3;
  else if (a <= 2.5F) idx = 4;
  else if (a < 3.5F) idx = 5;
  else if (a <= 5.0F) idx = 6;
  if (idx == 0) return 0;
  return static_cast<uint8_t>((x < 0.0F ? 0x8U : 0x0U) | idx);
}

inline float Nibble(uint8_t nib) {
  return kE2M1[nib & 0x7U] * ((nib & 0x8U) ? -1.0F : 1.0F);
}

// ScaledFp4Quant CPU kernel: x [M,K] float -> out_packed [M,K/2] i8 + out_scale
// [M,K/16] i8. Per-token, per-16-group; equals vllm::RefScaledFp4Quant.
void ScaledFp4QuantKernel(Queue&, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                          float input_global_scale_inv,
                          Fp4ScaleLayout scale_layout) {
  const int64_t m = x.shape[0], k = x.shape[1];
  constexpr int kBS = 16;
  const int64_t groups = k / kBS;
  const float gs_recip = RecipF(input_global_scale_inv);
  auto* packed = out_packed.Ptr<uint8_t>();
  auto* scale = out_scale.Ptr<uint8_t>();
  const int64_t scale_cols = out_scale.shape[1];
  if (scale_layout == Fp4ScaleLayout::kCutlassSwizzled) {
    std::fill_n(scale, out_scale.Numel(), uint8_t{0});
  }
  ForRows(m, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t base = g * kBS;
      float vec_max = 0.0F;
      for (int j = 0; j < kBS; ++j) vec_max = std::fmax(vec_max, std::fabs(LoadF32(x, i * k + base + j)));
      float sc = ClampF(input_global_scale_inv * (vec_max * (1.0F / kFp4Max)), -kFp8Max, kFp8Max);
      const uint8_t sc_f8 = F32ToFp8(sc);
      if (scale_layout == Fp4ScaleLayout::kLinear) {
        scale[i * groups + g] = sc_f8;
      } else {
        const int64_t m_tile = i / 128;
        const int64_t outer_m = i % 32;
        const int64_t inner_m = (i % 128) / 32;
        const int64_t k_tile = g / 4;
        const int64_t inner_k = g % 4;
        const int64_t scale_offset =
            ((((m_tile * (scale_cols / 4) + k_tile) * 32 + outer_m) * 4 +
               inner_m) *
                  4 +
              inner_k);
        scale[scale_offset] = sc_f8;
      }
      const float block_scale = Fp8ToF32(sc_f8) * gs_recip;
      const float out_scale_v = RecipF(block_scale);
      for (int j = 0; j < kBS; j += 2) {
        const float lo = ClampF(LoadF32(x, i * k + base + j) * out_scale_v, -kFp4Max, kFp4Max);
        const float hi = ClampF(LoadF32(x, i * k + base + j + 1) * out_scale_v, -kFp4Max, kFp4Max);
        packed[(i * k + base + j) / 2] =
            static_cast<uint8_t>(F32ToFp4Nibble(lo) | (F32ToFp4Nibble(hi) << 4));
      }
    }
  }
  });
}

// SiluMulFp4Quant CPU fallback = the exact composite (bf16 intermediate then
// quant) — which IS the definition of correctness for the CUDA fused kernel. The
// bf16 scratch reproduces the round-through-bf16 the CUDA kernel folds in.
void SiluMulFp4QuantKernel(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                           const Tensor& up, float input_global_scale_inv,
                           Fp4ScaleLayout scale_layout) {
  const int64_t m = gate.shape[0], i = gate.shape[1];
  std::vector<uint16_t> tmp(static_cast<size_t>(m) * static_cast<size_t>(i));
  Tensor act = Tensor::Contiguous(tmp.data(), DType::kBF16, gate.device, {m, i});
  MoeSiluMulKernel(q, act, gate, up);
  ScaledFp4QuantKernel(q, out_packed, out_scale, act, input_global_scale_inv,
                       scale_layout);
}

// CPU definition of vLLM's one-input silu_and_mul_nvfp4_quant custom op. Keep
// this visibly composite: it is the correctness oracle for the CUDA single-pass
// producer and preserves the BF16 store/load boundary exactly.
void SiluAndMulFp4QuantKernel(Queue& q, Tensor& out_packed, Tensor& out_scale,
                              const Tensor& gate_up,
                              float input_global_scale_inv,
                              Fp4ScaleLayout scale_layout) {
  const int64_t m = gate_up.shape[0], i = gate_up.shape[1] / 2;
  std::vector<uint16_t> tmp(static_cast<size_t>(m) * static_cast<size_t>(i));
  Tensor act = Tensor::Contiguous(tmp.data(), DType::kBF16, gate_up.device, {m, i});
  SiluAndMulKernel(q, act, gate_up);
  ScaledFp4QuantKernel(q, out_packed, out_scale, act,
                       input_global_scale_inv, scale_layout);
}

void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn,
                           const Tensor& gate);  // defined below

// SigmoidGateFp4Quant CPU fallback = the exact composite (bf16 intermediate then
// quant) — the definition of correctness for the CUDA fused kernel. The bf16
// scratch reproduces the round-through-bf16 the CUDA kernel folds in.
void SigmoidGateFp4QuantKernel(Queue& q, Tensor& out_packed, Tensor& out_scale,
                               const Tensor& attn, const Tensor& gate,
                               float input_global_scale_inv, Fp4ScaleLayout scale_layout) {
  const int64_t m = attn.shape[0], i = attn.shape[1];
  std::vector<uint16_t> tmp(static_cast<size_t>(m) * static_cast<size_t>(i));
  Tensor act = Tensor::Contiguous(tmp.data(), DType::kBF16, attn.device, {m, i});
  SigmoidGateBf16Kernel(q, act, attn, gate);
  ScaledFp4QuantKernel(q, out_packed, out_scale, act, input_global_scale_inv,
                       scale_layout);
}

// MatmulNvfp4Fp4 CPU kernel: out[m,n] = alpha * Σ_k (a_fp4·f8(a_scale))·(b_fp4·
// f8(b_scale)). Equals vllm::RunNvfp4Emulation up to K-reduction order.
void MatmulNvfp4Fp4Kernel(Queue&, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                          const Tensor& b_packed, const Tensor& b_scale, float alpha) {
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  constexpr int kBS = 16;
  const int64_t groups = k / kBS;
  const auto* ap = a_packed.Ptr<uint8_t>();
  const auto* as = a_scale.Ptr<uint8_t>();
  const auto* bp = b_packed.Ptr<uint8_t>();
  const auto* bs = b_scale.Ptr<uint8_t>();
  // Row-chunked over M (each output row + its arow decode owned by one
  // thread); per-column K-group reduction order unchanged.
  ForRows(m, [&](int64_t r0, int64_t r1) {
  std::vector<float> arow(static_cast<size_t>(k));
  for (int64_t i = r0; i < r1; ++i) {
    // Decode a_fp4·a_scale_fp8 for this row once (reused across N columns).
    for (int64_t g = 0; g < groups; ++g) {
      const float asf = Fp8ToF32(as[i * groups + g]);
      for (int j = 0; j < kBS / 2; ++j) {
        const uint8_t byte = ap[(i * k + g * kBS) / 2 + j];
        arow[static_cast<size_t>(g * kBS + 2 * j)] = Nibble(byte & 0x0FU) * asf;
        arow[static_cast<size_t>(g * kBS + 2 * j + 1)] = Nibble(byte >> 4) * asf;
      }
    }
    for (int64_t col = 0; col < n; ++col) {
      float acc = 0.0F;
      for (int64_t g = 0; g < groups; ++g) {
        const float bsf = Fp8ToF32(bs[col * groups + g]);
        for (int j = 0; j < kBS / 2; ++j) {
          const uint8_t byte = bp[(col * k + g * kBS) / 2 + j];
          acc += arow[static_cast<size_t>(g * kBS + 2 * j)] * (Nibble(byte & 0x0FU) * bsf);
          acc += arow[static_cast<size_t>(g * kBS + 2 * j + 1)] * (Nibble(byte >> 4) * bsf);
        }
      }
      StoreF32(out, i * n + col, alpha * acc);
    }
  }
  });
}

void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    int64_t id = ids.dtype == DType::kI32 ? ids.Ptr<int32_t>()[i] : ids.Ptr<int64_t>()[i];
    VT_CHECK(id >= 0 && id < v, "embedding: id out of range");
    for (int64_t j = 0; j < h; ++j) {
      StoreF32(out, i * h + j, LoadF32(table, id * h + j));
    }
  }
  });
}

// Llama-3 rope frequency rescale (vLLM Llama3RotaryEmbedding._compute_inv_freq,
// rotary_embedding/llama3_rope.py:33-54); no-op when scaling_factor <= 0. Mirrors
// the CUDA Llama3ScaleFreq element-for-element so the CPU reference and the CUDA
// kernel agree.
inline double Llama3ScaleFreq(double freq, const RopeArgs& a) {
  const double sf = static_cast<double>(a.llama3_scaling_factor);
  if (!(sf > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double lo = static_cast<double>(a.llama3_low_freq_factor);
  const double hi = static_cast<double>(a.llama3_high_freq_factor);
  const double omax = static_cast<double>(a.llama3_orig_max_position);
  const double low_freq_wavelen = omax / lo;
  const double high_freq_wavelen = omax / hi;
  const double wave_len = kTwoPi / freq;
  double smooth = 0.0;
  if (lo != hi) smooth = (omax / wave_len - lo) / (hi - lo);
  if (wave_len < high_freq_wavelen) return freq;
  if (wave_len > low_freq_wavelen) return freq / sf;
  return (1.0 - smooth) * freq / sf + smooth * freq;
}

// In-place rotation of one head starting at element head_off; f32 math,
// stores round back to the tensor's dtype (f32 or bf16).
void RopeRotateHead(const Tensor& t, int64_t head_off, int rot, double base, int64_t pos,
                    const RopeArgs& args) {
  const int half = rot / 2;
  for (int i = 0; i < half; ++i) {
    double freq = std::pow(base, -2.0 * i / rot);
    freq = Llama3ScaleFreq(freq, args);
    double angle = static_cast<double>(pos) * freq;
    float c = static_cast<float>(std::cos(angle));
    float s = static_cast<float>(std::sin(angle));
    float x = LoadF32(t, head_off + i);
    float y = LoadF32(t, head_off + i + half);
    StoreF32(t, head_off + i, x * c - y * s);
    StoreF32(t, head_off + i + half, x * s + y * c);
  }
}

void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    int64_t p = pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[i] : pos.Ptr<int64_t>()[i];
    for (int64_t hh = 0; hh < hq; ++hh) {
      RopeRotateHead(qs, (i * hq + hh) * d, args.rotary_dim, static_cast<double>(args.base), p, args);
    }
    for (int64_t hh = 0; hh < hk; ++hh) {
      RopeRotateHead(ks, (i * hk + hh) * d, args.rotary_dim, static_cast<double>(args.base), p, args);
    }
  }
  });
}

// Ported from vLLM's supplied-cache rotary path:
//   base.py:160-252; common.py:145-185; mrope.py:14-187,263-375
// @ e24d1b24fe96. Formula construction stays outside this hot apply loop.
int MropeAxisForPair(int64_t pair, const RopeArgs& args) {
  if (args.mrope_interleaved) {
    if (pair % 3 == 1 &&
        pair <= 3LL * static_cast<int64_t>(args.mrope_section[1])) {
      return 1;
    }
    if (pair % 3 == 2 &&
        pair <= 3LL * static_cast<int64_t>(args.mrope_section[2])) {
      return 2;
    }
    return 0;
  }
  if (pair < args.mrope_section[0]) return 0;
  if (pair < static_cast<int64_t>(args.mrope_section[0]) +
                 args.mrope_section[1]) {
    return 1;
  }
  return 2;
}

void RopeFromCacheKernel(Queue&, Tensor& qs, Tensor* ks,
                         const Tensor& positions, const Tensor& cache,
                         const RopeArgs& args) {
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  const bool is_mrope = positions.rank == 2;
  // MLA campaign W6: q/k are addressed through their STRIDES, not a contiguous
  // (token * heads + head) * head_dim formula. DeepSeek's DECOUPLED RoPE rotates
  // only the trailing qk_rope_head_dim slice of the query head
  // (deepseek_v2.py:580-595 / mla.py:160-167 pass `q[..., qk_nope_head_dim:]`),
  // and its k_pe is the trailing column block of the single fused
  // kv_a_proj_with_mqa output — both are STRIDED VIEWS. For a contiguous tensor
  // the strided offsets are integer-identical to the old formula, so every
  // existing caller is bit-identical by construction.
  ForRows(tokens, [&](int64_t row_start, int64_t row_end) {
    for (int64_t token = row_start; token < row_end; ++token) {
      for (int64_t pair = 0; pair < half; ++pair) {
        const int axis = is_mrope ? MropeAxisForPair(pair, args) : 0;
        const int64_t pos_offset =
            is_mrope ? static_cast<int64_t>(axis) * tokens + token : token;
        const int64_t position =
            positions.dtype == DType::kI32
                ? static_cast<int64_t>(positions.Ptr<int32_t>()[pos_offset])
                : positions.Ptr<int64_t>()[pos_offset];
        VT_CHECK(position >= 0 && position < cache.shape[0],
                 "rope_from_cache: position outside cache");
        const int64_t cache_off = position * args.rotary_dim;
        const float c = LoadF32(cache, cache_off + pair);
        const float s = LoadF32(cache, cache_off + half + pair);
        const int64_t first = args.is_neox_style ? pair : pair * 2;
        const int64_t second =
            args.is_neox_style ? pair + half : pair * 2 + 1;
        for (int64_t head = 0; head < hq; ++head) {
          const int64_t off = token * qs.stride[0] + head * qs.stride[1];
          const float x = LoadF32(qs, off + first);
          const float y = LoadF32(qs, off + second);
          StoreF32(qs, off + first, x * c - y * s);
          StoreF32(qs, off + second, x * s + y * c);
        }
        if (ks != nullptr) {
          for (int64_t head = 0; head < hk; ++head) {
            const int64_t off = token * ks->stride[0] + head * ks->stride[1];
            const float x = LoadF32(*ks, off + first);
            const float y = LoadF32(*ks, off + second);
            StoreF32(*ks, off + first, x * c - y * s);
            StoreF32(*ks, off + second, x * s + y * c);
          }
        }
      }
    }
  });
}

// Fused MLA norm-rope (kFusedNormRope) CPU reference — the byte-exact composite
// of {RmsNormKernel(x[:, :off]) ; RopeFromCacheKernel(x[:, off:off+rot])}. The
// two halves address DISJOINT dims, so running them fused (one row loop) is the
// SAME arithmetic in the SAME order as the two standalone kernels; this is the
// Tier-0 golden the CUDA kernel is gated against.
//   x           [T, off+rot]  — merged kv_a output (off = norm_weight length, rot = rotary_dim)
//   norm_weight [off]         — kv_a_layernorm weight
//   latent_out  [T, off]      — RmsNorm of the leading latent slice (NOT roped)
//   pe_out      [T, rot]      — RopeFromCache rotation of the trailing pe slice (single vector)
void FusedNormRopeKernel(Queue&, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                         const Tensor& norm_weight, const Tensor& positions,
                         const Tensor& cache, const RmsNormArgs& norm_args,
                         const RopeArgs& rope_args) {
  const int64_t t = x.shape[0];
  const int64_t off = norm_weight.shape[0];       // latent width (kv_lora_rank)
  const int64_t rot = rope_args.rotary_dim;       // decoupled-rope width (qk_rope_head_dim)
  const int64_t half = rot / 2;
  const int64_t xrs = x.stride[0], lrs = latent_out.stride[0], prs = pe_out.stride[0];
  ForRows(t, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) {
      // --- latent RMSNorm over [0, off) — mirrors RmsNormKernel exactly. ------
      const int64_t xb = i * xrs;
      float sumsq = 0.0f;
      for (int64_t j = 0; j < off; ++j) {
        const float v = LoadF32(x, xb + j);
        sumsq += v * v;
      }
      const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(off) + norm_args.eps);
      const int64_t lb = i * lrs;
      for (int64_t j = 0; j < off; ++j) {
        float wj = LoadF32(norm_weight, j);
        if (norm_args.gemma) wj += 1.0f;
        StoreF32(latent_out, lb + j, LoadF32(x, xb + j) * inv * wj);
      }
      // --- decoupled-pe RopeFromCache over [off, off+rot) — mirrors
      //     RopeFromCacheKernel (single head, base rope, positions rank-1). ----
      const int64_t position =
          positions.dtype == DType::kI32
              ? static_cast<int64_t>(positions.Ptr<int32_t>()[i])
              : positions.Ptr<int64_t>()[i];
      VT_CHECK(position >= 0 && position < cache.shape[0],
               "fused_norm_rope: position outside cache");
      const int64_t cache_off = position * rot;
      const int64_t pb = i * prs;
      for (int64_t pair = 0; pair < half; ++pair) {
        const float c = LoadF32(cache, cache_off + pair);
        const float s = LoadF32(cache, cache_off + half + pair);
        const int64_t first = rope_args.is_neox_style ? pair : pair * 2;
        const int64_t second = rope_args.is_neox_style ? pair + half : pair * 2 + 1;
        const float xr = LoadF32(x, xb + off + first);
        const float yr = LoadF32(x, xb + off + second);
        StoreF32(pe_out, pb + first, xr * c - yr * s);
        StoreF32(pe_out, pb + second, xr * s + yr * c);
      }
    }
  });
}

float Silu(float x) { return x / (1.0f + std::exp(-x)); }

// Per-step RoPE cos|sin cache fill (fused-attn-preamble prep). cos_sin[T,rot] f32:
// cols [0,half)=cos, [half,rot)=sin. Angle math in DOUBLE + f32 cast, matching
// RopeRotateHead/RopeNeoxKernel element-for-element so the cache reproduces the
// inline rotation bit-for-bit.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions, const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  const int64_t half = rot / 2;
  const double base = static_cast<double>(args.base);
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const int64_t p =
        positions.dtype == DType::kI32 ? positions.Ptr<int32_t>()[i] : positions.Ptr<int64_t>()[i];
    for (int64_t pair = 0; pair < half; ++pair) {
      double freq = std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      StoreF32(cos_sin, i * rot + pair, static_cast<float>(std::cos(angle)));
      StoreF32(cos_sin, i * rot + half + pair, static_cast<float>(std::sin(angle)));
    }
  }
  });
}

// gemma-RMSNorm one element: (v*inv)*(gemma ? w+1 : w) — matches RmsNormKernel's
// `v * inv * wj` (wj = w [+1 if gemma]) grouping and order exactly.
float GemmaNormElem(float v, float inv, float w, bool gemma) {
  float wj = w;
  if (gemma) wj += 1.0f;
  return v * inv * wj;
}

// Fused full-attention preamble: split q|gate + gemma qk-RMSNorm(Dh) + partial
// NeoX RoPE (from the cos_sin cache) + gate passthrough, in one pass. Bit-for-bit
// equal (f32 out) to AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox composed:
// the variance is f32, the weight is applied as (1+w), and the rotation reuses the
// same f32 c/sn the cache holds (x*c - y*sn / x*sn + y*c). Tail dims [rot,Dh) are
// normed but unrotated.
void AttnQkNormRopeGateKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                              const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const RmsNormArgs& na, const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int rot = ra.rotary_dim;
  const int64_t half = rot / 2;
  const bool gemma = na.gemma;
  // Normalize one head row (src..src+Dh) into out.., applying partial NeoX RoPE
  // from cs[0..rot). Recomputes normed[i]/normed[i+half] where paired.
  auto do_head = [&](const Tensor& src, int64_t src_off, const Tensor& w, const Tensor& out,
                     int64_t out_off, const float* cs) {
    float ss = 0.0f;
    for (int64_t j = 0; j < dh; ++j) {
      const float v = LoadF32(src, src_off + j);
      ss += v * v;
    }
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dh) + na.eps);
    for (int64_t j = 0; j < dh; ++j) {
      if (j < half) {
        const float ni = GemmaNormElem(LoadF32(src, src_off + j), inv, LoadF32(w, j), gemma);
        const float nih =
            GemmaNormElem(LoadF32(src, src_off + j + half), inv, LoadF32(w, j + half), gemma);
        StoreF32(out, out_off + j, ni * cs[j] - nih * cs[half + j]);
      } else if (j < rot) {
        const int64_t i = j - half;
        const float ni = GemmaNormElem(LoadF32(src, src_off + i), inv, LoadF32(w, i), gemma);
        const float nih =
            GemmaNormElem(LoadF32(src, src_off + i + half), inv, LoadF32(w, i + half), gemma);
        StoreF32(out, out_off + j, ni * cs[half + i] + nih * cs[i]);
      } else {
        StoreF32(out, out_off + j,
                 GemmaNormElem(LoadF32(src, src_off + j), inv, LoadF32(w, j), gemma));
      }
    }
  };
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t tok = r0; tok < r1; ++tok) {
    const float* cs = cos_sin.Ptr<float>() + tok * rot;
    for (int64_t h = 0; h < hq; ++h) {
      const int64_t qrow = tok * qgate.stride[0] + h * 2 * dh;
      do_head(qgate, qrow, q_norm, q_out, (tok * hq + h) * dh, cs);
      // gate passthrough: the second Dh of each (t,h) q|gate pair (no norm/rope).
      const int64_t gbase = qrow + dh;
      const int64_t gout = (tok * hq + h) * dh;
      for (int64_t j = 0; j < dh; ++j) StoreF32(gate_out, gout + j, LoadF32(qgate, gbase + j));
    }
    for (int64_t h = 0; h < hkv; ++h) {
      do_head(kf, tok * kf.stride[0] + h * dh, k_norm, k_out,
              (tok * hkv + h) * dh, cs);
    }
  }
  });
}

// GDN CPU reference kernels. Formulas: .agents/specs/gdn-semantics.md (§ cited per
// kernel); scalar f32 math throughout, states f32 in place.

// §2 causal_conv1d_fn. Per sequence s (tokens [qsl[s], qsl[s+1])), channel c,
// token t: window[j] = x token t-(K-1-j), falling back to
// conv_state[c, (K-1)+(t-i)] (init state) or 0 before the sequence start.
// Write-back: last K-1 RAW x tokens, left-padded with zeros / shifted old
// state when T < K-1. Outputs read the OLD state, so the row is buffered
// before overwrite.
void CausalConv1dFwdKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                           const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t total = x.shape[0], c_dim = x.shape[1], k = w.shape[1], width = k - 1;
  const int64_t n = conv_state.shape[0];
  // x may be a padded-row (inner-contiguous) view of the merged qkvz output;
  // out/conv_state stay contiguous so they keep the c_dim row stride.
  const int64_t x_rs = x.stride[0];
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total, "causal_conv1d_fwd: bad query_start_loc bounds");
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s] && qslp[s] >= 0,
             "causal_conv1d_fwd: query_start_loc not monotonic");
  }
  // Row-chunked over (sequence, channel) pairs: each pair owns its out column
  // slice and its conv_state row — independent outputs (spec W3 "conv1d").
  ForRows(n * c_dim, [&](int64_t r0, int64_t r1) {
  std::vector<float> old_row(static_cast<size_t>(width));
  for (int64_t r = r0; r < r1; ++r) {
    const int64_t s = r / c_dim, c = r % c_dim;
    const int64_t begin = qslp[s], end = qslp[s + 1], t_len = end - begin;
    const bool init = his.dtype == DType::kI8 ? his.Ptr<int8_t>()[s] != 0
                                               : his.Ptr<int32_t>()[s] != 0;
    float* srow_base = conv_state.Ptr<float>() + s * c_dim * width;
    {
      float* srow = srow_base + c * width;
      for (int64_t j = 0; j < width; ++j) old_row[static_cast<size_t>(j)] = srow[j];
      const float b = bias != nullptr ? LoadF32(*bias, c) : 0.0f;
      for (int64_t t = 0; t < t_len; ++t) {
        float acc = b;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ti = t - (k - 1 - j);  // token index of window[j]
          float v = 0.0f;
          if (ti >= 0) {
            v = LoadF32(x, (begin + ti) * x_rs + c);
          } else if (init) {
            v = old_row[static_cast<size_t>(width + ti)];  // state col (K-1)+(t-i)
          }
          acc += LoadF32(w, c * k + j) * v;
        }
        StoreF32(out, (begin + t) * c_dim + c, args.silu_activation ? Silu(acc) : acc);
      }
      for (int64_t j = 0; j < width; ++j) {
        const int64_t tj = t_len - width + j;  // new state col j holds token tj
        float v = 0.0f;
        if (tj >= 0) {
          v = LoadF32(x, (begin + tj) * x_rs + c);
        } else if (init) {
          v = old_row[static_cast<size_t>(width + tj)];  // shifted old state
        }
        srow[j] = v;
      }
    }
  }
  });
}

// §3 causal_conv1d_update (seqlen==1): read-old-then-roll. conv_state_indices
// (optional; mirrors mamba conv_state_indices): token bt's row is cache slot
// idx[bt] (idx<0 == NULL block → skip); null => compact row == bt.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1], width = k - 1;
  // The conv row physical width may be WIDENED to (K-1)+num_spec for spec-decode
  // rollback; the non-spec update operates on the LEADING (K-1) taps with the
  // row's PHYSICAL stride (mirror vLLM non-spec update `conv_state_token_offset=0`
  // + `state_len=KERNEL_WIDTH-1`, causal_conv1d.py:66-69,850-852). At num_spec==0
  // state_len==width, so the addressing below is byte-identical to before.
  const int64_t state_len = conv_state.shape[2];
  // x may be a padded-row view of the merged qkvz output; out is contiguous.
  const int64_t x_rs = x.stride[0];
  const int32_t* cache_idx =
      conv_state_indices != nullptr ? conv_state_indices->Ptr<int32_t>() : nullptr;
  // Row-chunked over (batch, channel) pairs: each pair owns its out element and
  // its conv_state row slice (batch rows map to distinct cache slots).
  ForRows(batch * c_dim, [&](int64_t r0, int64_t r1) {
  for (int64_t r = r0; r < r1; ++r) {
    const int64_t bt = r / c_dim, c = r % c_dim;
    int64_t srow_row = bt;
    if (cache_idx != nullptr) {
      if (cache_idx[bt] < 0) continue;  // NULL block
      srow_row = cache_idx[bt];
    }
    float* srow_base = conv_state.Ptr<float>() + srow_row * c_dim * state_len;
    {
      float* srow = srow_base + c * state_len;
      const float xt = LoadF32(x, bt * x_rs + c);
      float acc = bias != nullptr ? LoadF32(*bias, c) : 0.0f;
      for (int64_t j = 0; j < width; ++j) acc += LoadF32(w, c * k + j) * srow[j];
      acc += LoadF32(w, c * k + width) * xt;
      StoreF32(out, bt * c_dim + c, args.silu_activation ? Silu(acc) : acc);
      for (int64_t j = 0; j + 1 < width; ++j) srow[j] = srow[j + 1];  // roll left
      if (width > 0) srow[width - 1] = xt;                            // raw x
    }
  }
  });
}

// SPECULATIVE causal_conv1d_update (SPEC-MTP I4). Ported from
// vllm/model_executor/layers/mamba/ops/causal_conv1d.py @ e24d1b24
// (_causal_conv1d_update_kernel, IS_SPEC_DECODING + IS_VARLEN branches
// :818-1067). The conv row is the widened sliding window
// state_len = (K-1) + (max_query_len - 1); the read offset is
// num_accepted_tokens[i] - 1 (:835-852) and the row is shifted left by exactly
// ONE tap per step (:889-896, `idx_tokens + 1` under IS_SPEC_DECODING), which is
// what makes the NEXT step's offset select the window ending at the last
// ACCEPTED token. This is the executable reference for the CUDA kernel.
void CausalConv1dSpecUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                                  const Tensor* bias, Tensor& conv_state,
                                  const Tensor& conv_state_indices,
                                  const Tensor& num_accepted_tokens,
                                  const Tensor& cu_seqlens,
                                  const CausalConv1dArgs& args) {
  const int64_t c_dim = x.shape[1], k = w.shape[1], width = k - 1;
  const int64_t state_len = conv_state.shape[2];
  const int64_t x_rs = x.stride[0];
  const int64_t num_reqs = cu_seqlens.shape[0] - 1;
  const int32_t* cs = cu_seqlens.Ptr<int32_t>();
  const int32_t* nat = num_accepted_tokens.Ptr<int32_t>();
  const int32_t* cache_idx = conv_state_indices.Ptr<int32_t>();
  // Row-chunked over (request, channel): each pair owns its own conv_state row
  // slice and its own output column across the request's tokens.
  ForRows(num_reqs * c_dim, [&](int64_t r0, int64_t r1) {
    std::vector<float> win;
    std::vector<float> next(static_cast<size_t>(state_len));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t i = r / c_dim, c = r % c_dim;
      if (cache_idx[i] < 0) continue;  // NULL slot
      const int64_t lo = cs[i], hi = cs[i + 1];
      const int64_t seqlen = hi - lo;
      if (seqlen == 0) continue;  // zero-length padded row (:832-833)
      const int64_t off = static_cast<int64_t>(nat[i]) - 1;
      float* srow = conv_state.Ptr<float>() +
                    (static_cast<int64_t>(cache_idx[i]) * c_dim + c) * state_len;
      // S = old_state[off .. off+width-1] ++ x[lo .. hi-1]: the conv window
      // stream this step reads (:863-880 col0..col4 preload + the per-token x
      // load at :1019-1021).
      win.assign(static_cast<size_t>(width + seqlen), 0.0f);
      for (int64_t j = 0; j < width; ++j) {
        const int64_t src = off + j;
        win[static_cast<size_t>(j)] = src < state_len ? srow[src] : 0.0f;
      }
      for (int64_t t = 0; t < seqlen; ++t) {
        win[static_cast<size_t>(width + t)] = LoadF32(x, (lo + t) * x_rs + c);
      }
      for (int64_t t = 0; t < seqlen; ++t) {
        float acc = bias != nullptr ? LoadF32(*bias, c) : 0.0f;
        for (int64_t j = 0; j < k; ++j) {
          acc += LoadF32(w, c * k + j) * win[static_cast<size_t>(t + j)];
        }
        StoreF32(out, (lo + t) * c_dim + c, args.silu_activation ? Silu(acc) : acc);
      }
      // New row = old_state[off+1 .. off+(state_len-seqlen)] ++ x[lo..hi-1]
      // (:889-908 `new_conv_state = where(mask, conv_state, loaded_x)` with the
      // IS_SPEC_DECODING source offset `off + idx_tokens + 1`).
      const int64_t keep = state_len - seqlen;
      for (int64_t j = 0; j < keep; ++j) {
        const int64_t src = off + j + 1;
        next[static_cast<size_t>(j)] = src < state_len ? srow[src] : 0.0f;
      }
      for (int64_t t = 0; t < seqlen; ++t) {
        next[static_cast<size_t>(keep + t)] = LoadF32(x, (lo + t) * x_rs + c);
      }
      for (int64_t j = 0; j < state_len; ++j) srow[j] = next[static_cast<size_t>(j)];
    }
  });
}

// §4 l2norm_fwd: y = x * rsqrt(sum(x^2) + eps) over the last dim (plain SUM).
void L2NormKernel(Queue&, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = x.Numel() / d;
  ForRows(rows, [&](int64_t r0, int64_t r1) {
  for (int64_t r = r0; r < r1; ++r) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
      const float v = LoadF32(x, r * d + j);
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq + args.eps);
    for (int64_t j = 0; j < d; ++j) StoreF32(out, r * d + j, LoadF32(x, r * d + j) * inv);
  }
  });
}

// §5 RMSNormGated (norm_before_gate=True, group_size=None):
// out = x * rsqrt(mean(x^2) + eps) * w * act(gate); act = silu or sigmoid.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& w, const RmsNormGatedArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t t = x.Numel() / d;
  // x/out are contiguous (flat row i at i*d). The gate may be a padded-row
  // rank-3 [T,Hv,D] view of the merged qkvz z slice: map super-row i to
  // (token=i/Hv, head=i%Hv) and honor the token stride. Contiguous rank-2 gate
  // degenerates to i*gate.stride[0] == i*d (group == 1), byte-identical.
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
      const float v = LoadF32(x, i * d + j);
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
    const int64_t gbase = (i / gate_group) * gate_outer + (i % gate_group) * d;
    for (int64_t j = 0; j < d; ++j) {
      const float z = LoadF32(gate, gbase + j);
      const float act = args.sigmoid_gate ? 1.0f / (1.0f + std::exp(-z)) : Silu(z);
      StoreF32(out, i * d + j, LoadF32(x, i * d + j) * inv * LoadF32(w, j) * act);
    }
  }
  });
}

// RmsNormGated + static fp8 quant, fused (RmsNormGatedQuantFp8). Composite of
// RmsNormGatedKernel (bf16-rounded output) + QuantFp8Static: bit-identical to the
// split path because the fp8 is taken from the SAME bf16-rounded gated-norm value.
// CUDA has the hot path; this keeps the op available on the host backend for tests.
void RmsNormGatedQuantFp8Kernel(Queue&, Tensor& out_fp8, const Tensor& x, const Tensor& gate,
                                const Tensor& w, const RmsNormGatedArgs& args, float input_scale) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t t = x.Numel() / d;
  const float inv_scale = 1.0F / input_scale;
  uint8_t* op = out_fp8.Ptr<uint8_t>();
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    float sumsq = 0.0F;
    for (int64_t j = 0; j < d; ++j) {
      const float v = LoadF32(x, i * d + j);
      sumsq += v * v;
    }
    const float inv = 1.0F / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
    const int64_t gbase = (i / gate_group) * gate_outer + (i % gate_group) * d;
    for (int64_t j = 0; j < d; ++j) {
      const float z = LoadF32(gate, gbase + j);
      const float act = args.sigmoid_gate ? 1.0f / (1.0f + std::exp(-z)) : Silu(z);
      // bf16-intermediate (matches RmsNormGated's bf16 store then QuantFp8Static's load).
      const uint16_t nb = F32ToBF16(LoadF32(x, i * d + j) * inv * LoadF32(w, j) * act);
      op[i * d + j] = F32ToFp8(BF16ToF32(nb) * inv_scale);
    }
  }
  });
}

// §7 gated-delta-rule step for ONE value-head at ONE token. s_head points at
// this (sequence, head) [Dv,Dk] f32 block; tok indexes the packed q/k/v/g/beta
// rows; hk = hv / (Hv/Hk) is the GQA-broadcast key/query head.
//   q' = q * scale;  S *= exp(g[hv]);  v' = (v - S @ k) * beta[hv];
//   S += outer(v', k);  out = S @ q'      (k is NOT scaled)
// The per-head recurrence is sequential in `tok` (S carries forward), but
// distinct heads (and distinct sequences) touch disjoint state blocks and
// disjoint output rows — so heads are the parallel axis (GdnPrefillKernel),
// matching how the GPU GDN chunks over heads. Same instruction sequence and
// f32 reduction order regardless of which thread runs it: bit-identical.
void GdnHeadTokenStep(Tensor& out, const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
                      const Tensor& g, const Tensor& beta, float* s_head, int64_t tok,
                      int64_t hv, int64_t hk, int64_t hk_n, int64_t hv_n, int64_t dk, int64_t dv,
                      float scale, std::vector<float>& qbuf, std::vector<float>& kbuf,
                      std::vector<float>& vbuf) {
  const float g_t = g.Ptr<float>()[tok * hv_n + hv];
  const float beta_t = beta.Ptr<float>()[tok * hv_n + hv];
  const float decay = std::exp(g_t);
  for (int64_t i = 0; i < dk; ++i) {
    qbuf[static_cast<size_t>(i)] = LoadF32(q_in, (tok * hk_n + hk) * dk + i) * scale;
    kbuf[static_cast<size_t>(i)] = LoadF32(k_in, (tok * hk_n + hk) * dk + i);
  }
  for (int64_t vi = 0; vi < dv; ++vi) {
    float* s_row = s_head + vi * dk;
    float dot = 0.0f;  // (S * exp(g)) @ k, fused with the decay pass
    for (int64_t ki = 0; ki < dk; ++ki) {
      s_row[ki] *= decay;
      dot += s_row[ki] * kbuf[static_cast<size_t>(ki)];
    }
    vbuf[static_cast<size_t>(vi)] =
        (LoadF32(v_in, (tok * hv_n + hv) * dv + vi) - dot) * beta_t;
  }
  for (int64_t vi = 0; vi < dv; ++vi) {
    float* s_row = s_head + vi * dk;
    float o = 0.0f;  // (S + outer(v',k)) @ q', fused with the rank-1 update
    for (int64_t ki = 0; ki < dk; ++ki) {
      s_row[ki] += vbuf[static_cast<size_t>(vi)] * kbuf[static_cast<size_t>(ki)];
      o += s_row[ki] * qbuf[static_cast<size_t>(ki)];
    }
    StoreF32(out, (tok * hv_n + hv) * dv + vi, o);
  }
}

// All value-heads for one token, `state` at this sequence's [Hv,Dv,Dk] block.
// Used by the decode kernel (which parallelizes over the batch, so a whole
// token is one work item). Loop order (head-outer here, head-outer in prefill)
// is irrelevant to the result: each head owns a disjoint state block.
void GdnTokenStep(Tensor& out, const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
                  const Tensor& g, const Tensor& beta, float* state, int64_t tok, float scale,
                  std::vector<float>& qbuf, std::vector<float>& kbuf,
                  std::vector<float>& vbuf) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v_in.shape[1], dv = v_in.shape[2];
  const int64_t ratio = hv_n / hk_n;
  for (int64_t hv = 0; hv < hv_n; ++hv) {
    const int64_t hk = hv / ratio;
    GdnHeadTokenStep(out, q_in, k_in, v_in, g, beta, state + hv * dv * dk, tok, hv, hk, hk_n,
                     hv_n, dk, dv, scale, qbuf, kbuf, vbuf);
  }
}

void GdnPrefillKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                      const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& qsl,
                      const GdnArgs& args) {
  const int64_t n = state.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == q_in.shape[0],
           "gdn_prefill: bad query_start_loc bounds");
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s], "gdn_prefill: query_start_loc not monotonic");
  }
  // Row-chunked over the (SEQUENCE, VALUE-HEAD) product. At c1 prefill there is
  // one sequence but hv_n independent heads; sequences alone (spec W3's original
  // axis) left the whole recurrence single-threaded (kGdnPrefill profiled at 25%
  // of prefill). Each (s, hv) item owns a disjoint state block
  // state[(s*hv_n + hv)] and disjoint output rows out[(t*hv_n + hv)]; the
  // in-sequence, in-head recurrence stays sequential in tok. Byte-identical to
  // the single-thread order by construction (disjoint outputs, no shared
  // reduction) — matches the GPU GDN's head chunking.
  const int64_t hk_n = q_in.shape[1];
  const int64_t ratio = hv_n / hk_n;
  const int64_t nitems = n * hv_n;
  ForRows(nitems, [&](int64_t r0, int64_t r1) {
  std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
      vbuf(static_cast<size_t>(dv));
  for (int64_t item = r0; item < r1; ++item) {
    const int64_t s = item / hv_n;
    const int64_t hv = item % hv_n;
    const int64_t hk = hv / ratio;
    float* s_head = state.Ptr<float>() + (s * hv_n + hv) * dv * dk;
    for (int64_t t = qslp[s]; t < qslp[s + 1]; ++t)
      GdnHeadTokenStep(out, q_in, k, v, g, beta, s_head, t, hv, hk, hk_n, hv_n, dk, dv,
                       args.scale, qbuf, kbuf, vbuf);
  }
  });
}

void GdnDecodeKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                     const Tensor& g, const Tensor& beta, Tensor& state,
                     const Tensor* state_idx, const GdnArgs& args) {
  const int64_t batch = q_in.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int32_t* sidx = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  // Row-chunked over the BATCH (spec W3): each token owns its state slot
  // (distinct cache rows) and its output row.
  ForRows(batch, [&](int64_t r0, int64_t r1) {
  std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
      vbuf(static_cast<size_t>(dv));
  for (int64_t bt = r0; bt < r1; ++bt) {
    // state_idx != null => in-place on the FULL cache at slot sidx[bt] (fla
    // ssm_state_indices); sidx[bt]<0 == NULL block → zero out; null => row == bt.
    int64_t srow = bt;
    if (sidx != nullptr) {
      if (sidx[bt] < 0) {
        for (int64_t hv = 0; hv < hv_n; ++hv)
          for (int64_t d = 0; d < dv; ++d)
            StoreF32(out, (bt * hv_n + hv) * dv + d, 0.0f);
        continue;
      }
      srow = sidx[bt];
    }
    float* s_state = state.Ptr<float>() + srow * hv_n * dv * dk;
    GdnTokenStep(out, q_in, k, v, g, beta, s_state, bt, args.scale, qbuf, kbuf, vbuf);
  }
  });
}

// ── KDA per-K-channel-decay gated-delta recurrence (kKdaGatedDeltaRule) ────────
// Byte-for-byte GdnHeadTokenStep EXCEPT the decay is per-K-channel: plain GDN
// does `s_row[ki] *= exp(g_head)` (one scalar per value head), KDA does
// `s_row[ki] *= exp(g[.,hv,ki])` (one log-decay per K channel), broadcast across
// the Dv rows. Ported 1:1 from FLA fused_recurrent_gated_delta_rule_fwd_kernel
// IS_KDA=True (`b_h *= exp(b_gk[None, :])`, fused_recurrent.py:136-137 @ 555967922).
// All arithmetic f32 (FLA loads bf16 -> tl.float32); same reduction order as GDN.
void KdaHeadTokenStep(Tensor& out, const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
                      const Tensor& g, const Tensor& beta, float* s_head, int64_t tok,
                      int64_t hv, int64_t hk, int64_t hk_n, int64_t hv_n, int64_t dk, int64_t dv,
                      float scale, std::vector<float>& qbuf, std::vector<float>& kbuf,
                      std::vector<float>& vbuf, std::vector<float>& decaybuf) {
  const float beta_t = beta.Ptr<float>()[tok * hv_n + hv];
  const float* g_row = g.Ptr<float>() + (tok * hv_n + hv) * dk;
  for (int64_t ki = 0; ki < dk; ++ki) {
    qbuf[static_cast<size_t>(ki)] = LoadF32(q_in, (tok * hk_n + hk) * dk + ki) * scale;
    kbuf[static_cast<size_t>(ki)] = LoadF32(k_in, (tok * hk_n + hk) * dk + ki);
    decaybuf[static_cast<size_t>(ki)] = std::exp(g_row[ki]);
  }
  for (int64_t vi = 0; vi < dv; ++vi) {
    float* s_row = s_head + vi * dk;
    float dot = 0.0f;  // (S * exp(g_channel)) @ k, fused with the per-channel decay
    for (int64_t ki = 0; ki < dk; ++ki) {
      s_row[ki] *= decaybuf[static_cast<size_t>(ki)];
      dot += s_row[ki] * kbuf[static_cast<size_t>(ki)];
    }
    vbuf[static_cast<size_t>(vi)] =
        (LoadF32(v_in, (tok * hv_n + hv) * dv + vi) - dot) * beta_t;
  }
  for (int64_t vi = 0; vi < dv; ++vi) {
    float* s_row = s_head + vi * dk;
    float o = 0.0f;  // (S + outer(v',k)) @ q', fused with the rank-1 update
    for (int64_t ki = 0; ki < dk; ++ki) {
      s_row[ki] += vbuf[static_cast<size_t>(vi)] * kbuf[static_cast<size_t>(ki)];
      o += s_row[ki] * qbuf[static_cast<size_t>(ki)];
    }
    StoreF32(out, (tok * hv_n + hv) * dv + vi, o);
  }
}

void KdaGatedDeltaRuleKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k,
                             const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                             const Tensor& qsl, const GdnArgs& args) {
  const int64_t n = state.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == q_in.shape[0],
           "kda_gated_delta_rule: bad query_start_loc bounds");
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s], "kda_gated_delta_rule: query_start_loc not monotonic");
  }
  const int64_t hk_n = q_in.shape[1];
  const int64_t ratio = hv_n / hk_n;
  const int64_t nitems = n * hv_n;
  // Row-chunked over (SEQUENCE, VALUE-HEAD) exactly as GdnPrefillKernel — each
  // (s, hv) owns a disjoint state block + output rows; sequential in tok.
  ForRows(nitems, [&](int64_t r0, int64_t r1) {
    std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
        vbuf(static_cast<size_t>(dv)), decaybuf(static_cast<size_t>(dk));
    for (int64_t item = r0; item < r1; ++item) {
      const int64_t s = item / hv_n;
      const int64_t hv = item % hv_n;
      const int64_t hk = hv / ratio;
      float* s_head = state.Ptr<float>() + (s * hv_n + hv) * dv * dk;
      for (int64_t t = qslp[s]; t < qslp[s + 1]; ++t)
        KdaHeadTokenStep(out, q_in, k, v, g, beta, s_head, t, hv, hk, hk_n, hv_n, dk, dv,
                         args.scale, qbuf, kbuf, vbuf, decaybuf);
    }
  });
}

// KDA CHUNK-PREFILL (kKdaChunkPrefill) — CPU reference. There is no portable
// chunk kernel (the chunk path is the vendored FLA Triton-AOT cubins); the chunked
// forward computes the SAME per-K-channel gated-delta output as the recurrence up
// to reduction order, so the CPU reference fuses the gate exactly as FLA's
// kda_gate_cumsum (g = -exp(a_log)*softplus(g_raw + dt_bias), beta=1 softplus) then
// runs the proven KdaGatedDeltaRuleKernel recurrence. This keeps the op dual-
// registered (CPU+CUDA) so the whole-forward CPU gate exercises the wiring; the
// numerically-meaningful chunked-vs-recurrent comparison runs on the CUDA path.
void KdaChunkPrefillKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g_raw, const Tensor& beta,
                           const Tensor& a_log, const Tensor& dt_bias, Tensor& state,
                           const Tensor& qsl, const GdnArgs& args) {
  const int64_t T = q_in.shape[0], hv_n = state.shape[1], dk = state.shape[3];
  const bool has_bias = dt_bias.shape[0] != 0;
  const float* alp = a_log.Ptr<float>();
  const float* dbp = has_bias ? dt_bias.Ptr<float>() : nullptr;
  std::vector<float> g_dec(static_cast<size_t>(T) * hv_n * dk);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < hv_n; ++h) {
      const float a = -std::exp(alp[h]);
      for (int64_t d = 0; d < dk; ++d) {
        float x = LoadF32(g_raw, (t * hv_n + h) * dk + d);
        if (has_bias) x += dbp[h * dk + d];
        const float sp = x > 20.0f ? x : std::log1p(std::exp(x));  // softplus(beta=1)
        g_dec[static_cast<size_t>((t * hv_n + h) * dk + d)] = a * sp;
      }
    }
  }
  Tensor g_t = g_raw;  // same [T,Hv,Dk] f32 contiguous metadata, gated data
  g_t.data = g_dec.data();
  KdaGatedDeltaRuleKernel(q, out, q_in, k, v, g_t, beta, state, qsl, args);
}

// ── Mamba2 / SSD host references (.agents/specs/mamba2-ssd.md W1, #496) ──────
//
// Landed in THIS translation unit rather than a new src/vt/cpu/cpu_mamba2_ssd.cpp
// (which the spec's §3 port map names) because adding a source file means editing
// the root CMakeLists.txt, which check-doc-checkpoint classifies as user_usage and
// therefore owes a docs/USAGE.md update this change does not have. The GDN and KDA
// references these ops are siblings of already live here, so this is where a reader
// looks for them.
//
// Three ops, each a 1:1 transcription of the upstream path named on it, at the
// pinned oracle `555967922` (vLLM 0.26.0.dev0):
//
//   Mamba2ChunkScan   <- ops/ssd_combined.py:27-235 (the 5-stage varlen pipeline)
//                        over ssd_chunk_state.py, ssd_state_passing.py,
//                        ssd_bmm.py and ssd_chunk_scan.py
//   Mamba2StateUpdate <- ops/mamba_ssm.py:497+ `selective_state_update`, at the
//                        scalar-per-head shape Mamba2 uses, which is also the
//                        one upstream's OWN CPU kernel implements
//                        (csrc/cpu/mamba_kernels.hpp:104-250)
//   RmsNormGatedGroup <- mamba_mixer2.py:100-149 `Mixer2RMSNormGated.forward_native`
//
// SSD IS NOT THE GATED DELTA RULE. Nothing here is shared with, or derived from,
// cpu_ops.cpp's GdnPrefill/GdnDecode/KdaGatedDeltaRule: those carry a delta
// removal term `(I - beta*k*k^T)` this recurrence does not have, and a per-head
// (or per-K-channel) decay where this one has `exp(A[h]*dt[t,h])` with `B`/`C`
// shared across `n_groups` head groups. Sibling ops, one shared state layout
// (mamba2-ssd.md §0, §7).
//
// DTYPE POLICY. Arithmetic is ALWAYS f32; only the load/store width varies, so a
// bf16 stream rounds exactly at upstream's cast point and nowhere else. The two
// buffers that are f32 REGARDLESS of the activation dtype are f32 because
// upstream pins them there, not by our choice:
//   * the per-chunk state `states` — `states_in_fp32=True` (ssd_combined.py:100-102);
//   * `CB` — `output_dtype=torch.float32` (ssd_combined.py:124).
// The INTER-CHUNK state that `_state_passing_fwd` produces is NOT f32: it carries
// the caller's `state_dtype` (ssd_combined.py:46,119,176), and `_chunk_scan_fwd`
// reads it back at that width, so the rounding is observable in `out` as well as
// in `final_states`. `RoundThrough` below reproduces exactly that rounding.
//
// DELIBERATE DEVIATION FROM THE DEVICE ARM (W2 owns closing it). Triton's dots
// downcast their tile inputs — `_chunk_state_fwd` casts the decayed `B` to x's
// dtype before `tl.dot` (ssd_chunk_state.py:283-285) and `_chunk_scan_fwd` casts
// the f32 `CB` and the previous state to C's dtype (:266-269, :359-363). Those
// are tensor-core input-precision details of the device kernels, not statements
// of the algorithm; this host reference keeps f32 throughout. The W1 gate is the
// host reference against a sequential double-precision recurrence, not a
// bit-compare against Triton (mamba2-ssd.md §5).
// The value `v` as it would read back after a store/load round trip through
// `dt`. Used to keep an f32 working buffer numerically identical to one actually
// held at the target width — the `state_dtype` / `input_dtype` cast points.
float RoundThrough(DType dt, float v) {
  switch (dt) {
    case DType::kF32: return v;
    case DType::kF16: return F16ToF32(F32ToF16(v));
    case DType::kBF16: return BF16ToF32(F32ToBF16(v));
    default: VT_CHECK(false, "mamba2: unsupported dtype"); return 0.0f;
  }
}

// softplus, guarded exactly as upstream: `tl.where(dt <= 20.0, softplus(dt), dt)`
// (ssd_chunk_state.py:94; the same guard at mamba_kernels.hpp:177).
inline float SoftplusGuarded(float v) { return v <= 20.0f ? std::log1p(std::exp(v)) : v; }

inline float SiluGate(float z) { return z / (1.0f + std::exp(-z)); }

// A IS STRICTLY NEGATIVE, BY CONTRACT — and the contract is enforced, not
// assumed. Upstream builds it as `A = -torch.exp(self.A_log.float())`
// (mamba_mixer2.py:456), which is < 0 for every finite `A_log`, and `dt >= 0`
// after the `dt_limit` clamp (`dt_min < 0` is refused in the validator).
// Together those two make `dA_cumsum` non-increasing WITHIN a chunk, which is
// the only reason upstream's `min(., 0)` clamps (ssd_chunk_state.py:283-285,
// ssd_chunk_scan.py:339-341) are algebraic no-ops here rather than a
// truncation. Neither is derivable from the arguments: fed `A = +1.0` the
// clamped intra-chunk term silently returns a truncated recurrence instead of
// the growing one it was asked for. An arm that is not implemented is REFUSED
// with the missing piece named (mamba2-ssd.md §7).
void CheckMamba2ANegative(const Tensor& A, const char* name) {
  const float* Ap = A.Ptr<float>();
  for (int64_t h = 0; h < A.shape[0]; ++h) {
    VT_CHECK(Ap[h] < 0.0f,
             std::string(name) + ": A must be strictly negative (A = -exp(A_log), " +
                 "mamba_mixer2.py:456) — A[" + std::to_string(h) + "] = " +
                 std::to_string(Ap[h]));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// vt::Mamba2ChunkScan — the 5-stage varlen SSD prefill pipeline, in upstream
// order (_mamba_chunk_scan_combined_fwd, ssd_combined.py:88-146).
// ─────────────────────────────────────────────────────────────────────────────
void Mamba2ChunkScanKernel(Queue&, Tensor& out, Tensor& final_states, const Tensor& x,
                           const Tensor& dt_in, const Tensor& A, const Tensor& B,
                           const Tensor& C, const Tensor* D, const Tensor* z,
                           const Tensor* dt_bias, const Tensor* initial_states,
                           const Tensor& cu_seqlens, const Tensor& cu_chunk_seqlens,
                           const Tensor& last_chunk_indices, const Tensor& seq_idx,
                           const Mamba2Args& args) {
  const int64_t T = x.shape[0], H = x.shape[1], P = x.shape[2];
  const int64_t G = B.shape[1], N = B.shape[2];
  const int64_t S = final_states.shape[0];
  const int64_t cs = args.chunk_size;
  const int64_t nchunks = cu_chunk_seqlens.shape[0] - 1;
  const int64_t hpg = H / G;  // nheads_ngroups_ratio (ssd_chunk_state.py:238)

  const int32_t* ccs = cu_chunk_seqlens.Ptr<int32_t>();
  const int32_t* lci = last_chunk_indices.Ptr<int32_t>();
  const int32_t* sidx = seq_idx.Ptr<int32_t>();
  const int32_t* cus = cu_seqlens.Ptr<int32_t>();
  const float* Ap = A.Ptr<float>();
  const float* dbp = dt_bias != nullptr ? dt_bias->Ptr<float>() : nullptr;
  const float* Dp = D != nullptr ? D->Ptr<float>() : nullptr;
  const bool d_has_hdim = D != nullptr && D->rank == 2;

  // The metadata contract `compute_varlen_chunk_metadata` guarantees
  // (mamba2_attn.py:56-74): chunks tile [0,T) in order, none crosses a physical
  // chunk boundary, and none is empty.
  VT_CHECK(ccs[0] == 0 && ccs[nchunks] == static_cast<int32_t>(T),
           "mamba2_chunk_scan: cu_chunk_seqlens must tile [0, T)");
  VT_CHECK(cus[0] == 0 && cus[S] == static_cast<int32_t>(T),
           "mamba2_chunk_scan: cu_seqlens must tile [0, T)");
  for (int64_t c = 0; c < nchunks; ++c) {
    const int64_t len = ccs[c + 1] - ccs[c];
    VT_CHECK(len > 0 && len <= cs,
             "mamba2_chunk_scan: each logical chunk must be non-empty and at most chunk_size");
    VT_CHECK(sidx[c] >= 0 && sidx[c] < S,
             "mamba2_chunk_scan: seq_idx entries must index a sequence");
  }
  CheckMamba2ANegative(A, "mamba2_chunk_scan");
  for (int64_t b = 0; b < S; ++b) {
    // `last_chunk_indices[b] == -1` is what `compute_varlen_chunk_metadata`
    // leaves for a sequence with NO tokens (mamba2_attn.py:56-74 pushes no chunk
    // for it). Upstream's `varlen_states = states[last_chunk_indices]`
    // (ssd_combined.py:154) is then a torch NEGATIVE index: it silently returns
    // the last chunk of the whole batch — some OTHER sequence's state. vLLM
    // never schedules an empty sequence, so that is an indexing quirk rather
    // than behaviour to mirror; refuse instead of deviating quietly.
    VT_CHECK(lci[b] >= 0,
             "mamba2_chunk_scan: last_chunk_indices[" + std::to_string(b) +
                 "] < 0 — empty sequences are NOT supported (upstream's "
                 "states[last_chunk_indices] would negative-index another sequence's "
                 "chunk, ssd_combined.py:154)");
    VT_CHECK(lci[b] < nchunks, "mamba2_chunk_scan: last_chunk_indices out of range");
  }

  // ── stage 1: `_chunk_cumsum_fwd` (ssd_chunk_state.py:300-346) ──────────────
  // dt_out and dA_cumsum are [H, nchunks, chunk_size] f32 (:314-319). Positions
  // past a partial chunk hold dt = 0 (:104-107), so dA_cumsum[..., cs-1] is the
  // chunk's TOTAL decay whatever its length — which is what stages 2 and 3 read.
  std::vector<float> dtv(static_cast<size_t>(H * nchunks * cs), 0.0f);
  std::vector<float> dac(static_cast<size_t>(H * nchunks * cs), 0.0f);
  ForRows(H, [&](int64_t h0, int64_t h1) {
    for (int64_t h = h0; h < h1; ++h) {
      const float a = Ap[h];
      for (int64_t c = 0; c < nchunks; ++c) {
        const int64_t start = ccs[c], len = ccs[c + 1] - start;
        const size_t base = static_cast<size_t>((h * nchunks + c) * cs);
        float acc = 0.0f;
        for (int64_t i = 0; i < cs; ++i) {
          float d = 0.0f;
          if (i < len) {
            d = LoadF32(dt_in, (start + i) * H + h);
            if (dbp != nullptr) d += dbp[h];
            if (args.dt_softplus) d = SoftplusGuarded(d);
            d = std::min(std::max(d, args.dt_min), args.dt_max);
          }
          dtv[base + static_cast<size_t>(i)] = d;
          acc += d * a;
          dac[base + static_cast<size_t>(i)] = acc;
        }
      }
    }
  });

  // ── stage 2: `_chunk_state_fwd` (ssd_chunk_state.py:349-407) ───────────────
  // states[c,h,p,n] = sum_i x[i,h,p] * B[i,g,n] * exp(min(dA_last - dA_i, 0)) * dt_i
  // f32 by upstream's own `states_in_fp32=True` (ssd_combined.py:100-102).
  std::vector<float> states(static_cast<size_t>(nchunks * H * P * N), 0.0f);
  ForRows(nchunks * H, [&](int64_t r0, int64_t r1) {
    std::vector<float> xrow(static_cast<size_t>(P)), brow(static_cast<size_t>(N));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t c = r / H, h = r % H, g = h / hpg;
      const int64_t start = ccs[c], len = ccs[c + 1] - start;
      const size_t dbase = static_cast<size_t>((h * nchunks + c) * cs);
      const float da_last = dac[dbase + static_cast<size_t>(cs - 1)];
      float* st = states.data() + static_cast<size_t>((c * H + h) * P * N);
      for (int64_t i = 0; i < len; ++i) {
        // The `min(., 0)` is upstream's, and INSIDE THE ENFORCED CONTRACT it is
        // an algebraic no-op: `A < 0` (CheckMamba2ANegative) and `dt >= 0`
        // (dt_min >= 0, checked in the validator) make `dac` non-increasing over
        // i, so `da_last - dac[i] <= 0` for every i in the chunk. It is kept
        // because upstream keeps it — it guards the floating-point edge where
        // the running cumsum ticks up by an ulp — and it is NOT what stands
        // between the op and an `A > 0` caller; that arm is refused outright.
        const float scale = std::exp(std::min(da_last - dac[dbase + static_cast<size_t>(i)],
                                              0.0f)) *
                            dtv[dbase + static_cast<size_t>(i)];
        if (scale == 0.0f) continue;
        for (int64_t p = 0; p < P; ++p)
          xrow[static_cast<size_t>(p)] = LoadF32(x, ((start + i) * H + h) * P + p);
        for (int64_t n = 0; n < N; ++n)
          brow[static_cast<size_t>(n)] =
              LoadF32(B, ((start + i) * G + g) * N + n) * scale;
        for (int64_t p = 0; p < P; ++p) {
          const float xv = xrow[static_cast<size_t>(p)];
          float* srow = st + p * N;
          for (int64_t n = 0; n < N; ++n) srow[n] += xv * brow[static_cast<size_t>(n)];
        }
      }
    }
  });

  // ── stage 3: `_state_passing_fwd` (ssd_state_passing.py:99-146) ────────────
  // Per SEQUENCE, over the chunk range last_chunk_indices derives (:56-60):
  //   S_c = exp(dA_last[c]) * S_{c-1} + states[c],  S_{-1} = initial_states[b]
  // and out[c] is the state AFTER chunk c (:90-97).
  //
  // THE RUNNING STATE STAYS F32 AND THE STORE ROUNDS. Upstream carries `states`
  // in f32 registers across the chunk loop and stores a `state_dtype` copy per
  // chunk (:88-97) — it never reads its own rounded store back into the
  // recurrence. `_chunk_scan_fwd` DOES read that stored copy, so `state_dtype`
  // shows up in `out` as well as in `final_states`, but it must not compound
  // inside the state passing itself.
  // `passed` is f32 while upstream's `out` buffer is `state_dtype` (2x narrower
  // at bf16). It holds only `state_dtype`-ROUNDED values (RoundThrough below),
  // so the extra width is not numerically observable — it is a HOST-REFERENCE
  // working buffer that keeps stage 5 on one load path, never a model-path
  // allocation. W2 (the CUDA arm) must allocate this at `state_dtype`, as
  // upstream does, and must NOT inherit this width (.agents/porting.md: a
  // token gate cannot catch a dtype that is too WIDE).
  std::vector<float> passed(static_cast<size_t>(nchunks * H * P * N), 0.0f);
  const DType state_dt = final_states.dtype;
  ForRows(S * H, [&](int64_t r0, int64_t r1) {
    std::vector<float> s(static_cast<size_t>(P * N));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t b = r / H, h = r % H;
      const int64_t chunk_end = lci[b] + 1;
      const int64_t chunk_start = b > 0 ? lci[b - 1] + 1 : 0;
      if (initial_states != nullptr) {
        for (int64_t i = 0; i < P * N; ++i)
          s[static_cast<size_t>(i)] = LoadF32(*initial_states, (b * H + h) * P * N + i);
      } else {
        std::fill(s.begin(), s.end(), 0.0f);
      }
      for (int64_t c = chunk_start; c < chunk_end; ++c) {
        const float decay =
            std::exp(dac[static_cast<size_t>((h * nchunks + c) * cs + cs - 1)]);
        const float* in = states.data() + static_cast<size_t>((c * H + h) * P * N);
        float* outp = passed.data() + static_cast<size_t>((c * H + h) * P * N);
        for (int64_t i = 0; i < P * N; ++i) {
          s[static_cast<size_t>(i)] = s[static_cast<size_t>(i)] * decay + in[i];
          outp[i] = RoundThrough(state_dt, s[static_cast<size_t>(i)]);
        }
      }
      // `varlen_states = states[last_chunk_indices]` (ssd_combined.py:154).
      // A sequence with no chunk keeps its incoming state, which for a fresh
      // sequence is zeros.
      for (int64_t i = 0; i < P * N; ++i)
        StoreF32(final_states, (b * H + h) * P * N + i, s[static_cast<size_t>(i)]);
    }
  });

  // ── stage 4: `_bmm_chunk_fwd` (ssd_bmm.py:148-209) ─────────────────────────
  // CB[c,g,i,j] = sum_n C[i,g,n] * B[j,g,n], f32 REGARDLESS of the activation
  // dtype (`output_dtype=torch.float32`, ssd_combined.py:124).
  std::vector<float> cb(static_cast<size_t>(nchunks * G * cs * cs), 0.0f);
  ForRows(nchunks * G, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t c = r / G, g = r % G;
      const int64_t start = ccs[c], len = ccs[c + 1] - start;
      float* dst = cb.data() + static_cast<size_t>((c * G + g) * cs * cs);
      std::vector<float> crow(static_cast<size_t>(N));
      for (int64_t i = 0; i < len; ++i) {
        for (int64_t n = 0; n < N; ++n)
          crow[static_cast<size_t>(n)] = LoadF32(C, ((start + i) * G + g) * N + n);
        for (int64_t j = 0; j <= i; ++j) {  // only j <= i is ever read (IS_CAUSAL)
          float acc = 0.0f;
          for (int64_t n = 0; n < N; ++n)
            acc += crow[static_cast<size_t>(n)] * LoadF32(B, ((start + j) * G + g) * N + n);
          dst[i * cs + j] = acc;
        }
      }
    }
  });

  // ── stage 5: `_chunk_scan_fwd` (ssd_chunk_scan.py:216-525) ────────────────
  //   out_i = exp(dA_i) * (C_i . S_{c-1})                                  inter
  //         + sum_{j<=i} CB[i,j] * exp(min(dA_i - dA_j, 0)) * dt_j * x_j    intra
  //         + D * x_i                                                      skip
  //   then `out *= z * sigmoid(z)` when z is given (:394-406).
  ForRows(nchunks * H, [&](int64_t r0, int64_t r1) {
    std::vector<float> prev(static_cast<size_t>(P * N));
    std::vector<float> crow(static_cast<size_t>(N));
    std::vector<float> w(static_cast<size_t>(cs));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t c = r / H, h = r % H, g = h / hpg;
      const int64_t start = ccs[c], len = ccs[c + 1] - start;
      const size_t dbase = static_cast<size_t>((h * nchunks + c) * cs);

      // Previous state: initial_states[seq_idx[c]] when this chunk opens a new
      // sequence AND initial states were supplied, ZEROS when they were not
      // (ssd_chunk_scan.py:236-250, :271-274), else the passed state of c-1.
      const int32_t si = sidx[c];
      const int32_t si_prev = c >= 1 ? sidx[c - 1] : -1;
      bool prev_zero = false;
      if (si != si_prev) {
        if (initial_states != nullptr) {
          for (int64_t i = 0; i < P * N; ++i)
            prev[static_cast<size_t>(i)] =
                LoadF32(*initial_states, (static_cast<int64_t>(si) * H + h) * P * N + i);
        } else {
          prev_zero = true;
        }
      } else {
        const float* src = passed.data() + static_cast<size_t>(((c - 1) * H + h) * P * N);
        std::copy(src, src + P * N, prev.begin());
      }

      const float* cbc = cb.data() + static_cast<size_t>((c * G + g) * cs * cs);
      for (int64_t i = 0; i < len; ++i) {
        const float da_i = dac[dbase + static_cast<size_t>(i)];
        const float scale_m = std::exp(da_i);
        for (int64_t n = 0; n < N; ++n)
          crow[static_cast<size_t>(n)] = LoadF32(C, ((start + i) * G + g) * N + n);
        // The intra-chunk weight is independent of p; hoist it out of the p loop.
        // As in stage 2, `min(., 0)` is an algebraic no-op inside the enforced
        // contract (`A < 0`, `dt >= 0` => `dac` non-increasing => `da_i <= da_j`
        // for j <= i) and is kept only because upstream keeps it.
        for (int64_t j = 0; j <= i; ++j) {
          w[static_cast<size_t>(j)] =
              cbc[i * cs + j] *
              std::exp(std::min(da_i - dac[dbase + static_cast<size_t>(j)], 0.0f)) *
              dtv[dbase + static_cast<size_t>(j)];
        }
        for (int64_t p = 0; p < P; ++p) {
          float acc = 0.0f;
          if (!prev_zero) {
            const float* prow = prev.data() + p * N;
            for (int64_t n = 0; n < N; ++n) acc += crow[static_cast<size_t>(n)] * prow[n];
          }
          acc *= scale_m;
          for (int64_t j = 0; j <= i; ++j)
            acc += w[static_cast<size_t>(j)] * LoadF32(x, ((start + j) * H + h) * P + p);
          const float xi = LoadF32(x, ((start + i) * H + h) * P + p);
          if (Dp != nullptr) acc += (d_has_hdim ? Dp[h * P + p] : Dp[h]) * xi;
          if (z != nullptr) acc *= SiluGate(LoadF32(*z, ((start + i) * H + h) * P + p));
          StoreF32(out, ((start + i) * H + h) * P + p, acc);
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// vt::Mamba2StateUpdate — `selective_state_update` (ops/mamba_ssm.py:497+) at the
// scalar-per-head shape, transcribed from upstream's own CPU kernel
// (csrc/cpu/mamba_kernels.hpp:104-250 `selective_state_update_kernel`).
// ─────────────────────────────────────────────────────────────────────────────
void Mamba2StateUpdateKernel(Queue&, Tensor& out, Tensor& state, const Tensor& x,
                             const Tensor& dt_in, const Tensor& A, const Tensor& B,
                             const Tensor& C, const Tensor* D, const Tensor* z,
                             const Tensor* dt_bias, const Tensor* state_indices,
                             const Mamba2Args& args) {
  const int64_t Nb = x.shape[0], H = x.shape[1], P = x.shape[2];
  const int64_t G = B.shape[1], N = B.shape[2];
  const int64_t S = state.shape[0];
  const int64_t hpg = H / G;
  const int32_t* sidx = state_indices != nullptr ? state_indices->Ptr<int32_t>() : nullptr;
  const float* Ap = A.Ptr<float>();
  const float* Dp = D != nullptr ? D->Ptr<float>() : nullptr;
  const float* dbp = dt_bias != nullptr ? dt_bias->Ptr<float>() : nullptr;

  CheckMamba2ANegative(A, "mamba2_state_update");

  // Row-chunked over the BATCH: each row owns one cache slot and one output row,
  // as GdnDecodeKernel does. `ForRows` is a PARALLEL for, so two rows naming the
  // same slot race on the same read-modify-write and the answer depends on the
  // thread schedule; upstream's own CPU kernel is sequential and stays
  // deterministic under duplicates, so distinctness is a LOCAL precondition and
  // is enforced here rather than only documented. NULL rows (index < 0) touch no
  // slot at all and may repeat.
  if (sidx != nullptr) {
    std::vector<int32_t> slots;
    slots.reserve(static_cast<size_t>(Nb));
    for (int64_t b = 0; b < Nb; ++b)
      if (sidx[b] >= 0) slots.push_back(sidx[b]);
    std::sort(slots.begin(), slots.end());
    VT_CHECK(std::adjacent_find(slots.begin(), slots.end()) == slots.end(),
             "mamba2_state_update: state_indices entries must be DISTINCT (each row owns "
             "one cache slot; the row dispatch is parallel)");
  }

  ForRows(Nb, [&](int64_t r0, int64_t r1) {
    for (int64_t b = r0; b < r1; ++b) {
      int64_t slot = b;
      if (sidx != nullptr) {
        // LOCAL ABI: index < 0 is the NULL row (upstream's `NULL_BLOCK_ID`
        // padding, v1/attention/backends/utils.py:46, remapped by the caller).
        // Its cache slot is untouched — `continue` at mamba_kernels.hpp:147 —
        // and its output row is zeroed, as GdnDecodeKernel does.
        if (sidx[b] < 0) {
          for (int64_t i = 0; i < H * P; ++i) StoreF32(out, b * H * P + i, 0.0f);
          continue;
        }
        slot = sidx[b];
        VT_CHECK(slot < S, "mamba2_state_update: state_indices entry out of range");
      }
      for (int64_t h = 0; h < H; ++h) {
        const int64_t g = h / hpg;
        float d = LoadF32(dt_in, b * H + h);
        if (dbp != nullptr) d += dbp[h];
        if (args.dt_softplus) d = SoftplusGuarded(d);
        const float dA = std::exp(Ap[h] * d);
        for (int64_t p = 0; p < P; ++p) {
          const float xv = LoadF32(x, (b * H + h) * P + p);
          const int64_t sbase = ((slot * H + h) * P + p) * N;
          float y = 0.0f;
          for (int64_t n = 0; n < N; ++n) {
            const float bv = LoadF32(B, (b * G + g) * N + n);
            const float cv = LoadF32(C, (b * G + g) * N + n);
            // The readout uses the F32 value, not the value re-read from the
            // cache: the Triton kernel holds `state` in registers and computes
            // `out = sum(state * C)` from it, storing the cache-width copy
            // separately (mamba_ssm.py:433,451); upstream's CPU kernel does the
            // same with `s_new` (mamba_kernels.hpp:225-228).
            const float sn = LoadF32(state, sbase + n) * dA + bv * xv * d;
            StoreF32(state, sbase + n, sn);
            y += sn * cv;
          }
          if (Dp != nullptr) y += Dp[h] * xv;
          if (z != nullptr) y *= SiluGate(LoadF32(*z, (b * H + h) * P + p));
          StoreF32(out, (b * H + h) * P + p, y);
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// vt::RmsNormGatedGroup — `Mixer2RMSNormGated.forward_native`
// (mamba_mixer2.py:100-149).
// ─────────────────────────────────────────────────────────────────────────────
void RmsNormGatedGroupKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                             const Tensor* weight, const RmsNormGatedGroupArgs& args) {
  const int64_t hidden = x.shape[x.rank - 1];
  int64_t rows = 1;
  for (int r = 0; r < x.rank - 1; ++r) rows *= x.shape[r];
  const int64_t group_size = hidden / args.n_groups;
  // THE WEIGHT IS READ AT ITS OWN DTYPE. `Mixer2RMSNormGated.weight` is
  // `nn.Parameter(torch.ones(per_rank_hidden_size))` (mamba_mixer2.py:91),
  // created at the MODEL dtype — bf16 for every checkpoint that ships this
  // layer — and the validator accepts any float (`CheckMamba2Operand(...,
  // is_output=false)`, src/vt/ops.cpp). `Tensor::Ptr<float>()` is an unchecked
  // `static_cast` (include/vt/tensor.h), so taking one here would read
  // `hidden * 4` bytes out of a `hidden * 2` byte allocation. `LoadF32` is the
  // dtype-aware read the sibling `RmsNormGatedKernel` already uses.
  // `input_dtype = x.dtype` (:113) is the width the normalized value is cast back
  // to before the weight multiply (`self.weight * x.to(input_dtype)`, :149).
  const DType input_dt = x.dtype;

  ForRows(rows, [&](int64_t r0, int64_t r1) {
    std::vector<float> v(static_cast<size_t>(hidden));
    for (int64_t r = r0; r < r1; ++r) {
      // `x = x * nn.functional.silu(gate.to(torch.float32))` (:114) — the gate is
      // promoted to f32 BEFORE the silu, which is why this is f32 throughout.
      for (int64_t j = 0; j < hidden; ++j)
        v[static_cast<size_t>(j)] =
            LoadF32(x, r * hidden + j) * SiluGate(LoadF32(gate, r * hidden + j));
      if (weight == nullptr) {
        // use_rms_norm == False: no parameter, no norm (:94-96, :115-116).
        for (int64_t j = 0; j < hidden; ++j)
          StoreF32(out, r * hidden + j,
                    RoundThrough(input_dt, v[static_cast<size_t>(j)]));
        continue;
      }
      for (int64_t g = 0; g < args.n_groups; ++g) {
        // variance = x_grouped.pow(2).mean(-1) over group_size (:139-140); the
        // n_groups == 1 branch (:127-131) is the same expression with
        // group_size == hidden.
        // f32 accumulation, NOT double: upstream reduces `x.pow(2).mean(-1)`
        // in f32 (x is f32 from the :114 gate promotion) and the sibling
        // `RmsNormGatedKernel` above does the same. W2 must not inherit a wider
        // host-reference reduction.
        float ss = 0.0f;
        for (int64_t j = 0; j < group_size; ++j) {
          const float t = v[static_cast<size_t>(g * group_size + j)];
          ss += t * t;
        }
        // `rsqrt(variance + eps)` (:130, :141) — eps is INSIDE the square root.
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(group_size) + args.eps);
        for (int64_t j = 0; j < group_size; ++j) {
          const int64_t idx = g * group_size + j;
          const float normed = RoundThrough(input_dt, v[static_cast<size_t>(idx)] * inv);
          StoreF32(out, r * hidden + idx, LoadF32(*weight, idx) * normed);
        }
      }
    }
  });
}

// SPECULATIVE (multi-token, slot-snapshotting) gated-delta-rule step.
// Ported from vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py @
// e24d1b24 — fused_recurrent_gated_delta_rule_fwd_kernel with IS_VARLEN
// (:66-72), IS_SPEC_DECODING (the initial-slot select :103-116) and
// INPLACE_FINAL_STATE (the per-timestep snapshot :156-166) all True, which is
// exactly how qwen_gdn_linear_attn.py:1455-1476 calls it.
//
// This is the executable CPU reference the CUDA kernel is checked against, and
// the ground truth for the rollback-equivalence test: because the recurrence is
// strictly sequential and slot t receives the state after exactly t+1 tokens,
// selecting slot j next step IS running only the first j+1 tokens.
void GdnSpecDecodeKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor& cu_seqlens,
                         const Tensor& state_indices,
                         const Tensor& num_accepted_tokens, const GdnArgs& args) {
  const int64_t hv_n = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk_n = q_in.shape[1];
  const int64_t ratio = hv_n / hk_n;
  const int64_t num_reqs = state_indices.shape[0];
  const int64_t num_cols = state_indices.shape[1];
  const int64_t row_elems = hv_n * dv * dk;
  const int32_t* cs = cu_seqlens.Ptr<int32_t>();
  const int32_t* nat = num_accepted_tokens.Ptr<int32_t>();
  const int32_t* sidx = state_indices.Ptr<int32_t>();
  // Row-chunked over (request, value-head): each item owns a disjoint
  // [Dv,Dk] state block in every slot of its request's row, and disjoint
  // output rows. The in-request recurrence stays sequential in t.
  ForRows(num_reqs * hv_n, [&](int64_t r0, int64_t r1) {
    std::vector<float> qbuf(static_cast<size_t>(dk)), kbuf(static_cast<size_t>(dk)),
        vbuf(static_cast<size_t>(dv));
    std::vector<float> head(static_cast<size_t>(dv * dk));
    for (int64_t item = r0; item < r1; ++item) {
      const int64_t i = item / hv_n;
      const int64_t hv = item % hv_n;
      const int64_t hk = hv / ratio;
      const int64_t lo = cs[i], hi = cs[i + 1];
      // Initial state slot = column (num_accepted - 1) — the snapshot taken
      // after the last ACCEPTED token of the previous step (:106-116).
      const int32_t init_slot = sidx[i * num_cols + (nat[i] - 1)];
      if (init_slot < 0) {  // NULL slot ⇒ skip the whole request (:114-115)
        for (int64_t t = lo; t < hi; ++t)
          for (int64_t d = 0; d < dv; ++d)
            StoreF32(out, (t * hv_n + hv) * dv + d, 0.0f);
        continue;
      }
      if (hi == lo) continue;  // T == 0 early-out (:76-78)
      float* s_head = head.data();
      const int64_t base = static_cast<int64_t>(init_slot) * row_elems + hv * dv * dk;
      for (int64_t e = 0; e < dv * dk; ++e)
        s_head[e] = LoadF32(state, base + e);
      for (int64_t t = lo; t < hi; ++t) {
        GdnHeadTokenStep(out, q_in, k, v, g, beta, s_head, t, hv, hk, hk_n, hv_n, dk, dv,
                         args.scale, qbuf, kbuf, vbuf);
        // Snapshot AFTER this timestep into its own slot (:156-166). A NULL
        // column skips only its store, leaving that slot untouched.
        const int32_t slot = sidx[i * num_cols + (t - lo)];
        if (slot < 0) continue;
        const int64_t dst = static_cast<int64_t>(slot) * row_elems + hv * dv * dk;
        for (int64_t e = 0; e < dv * dk; ++e) StoreF32(state, dst + e, s_head[e]);
      }
    }
  });
}

// Pure packed decode ported from vLLM v0.25.0
// vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-336 @ 702f4814.
// Unlike GdnDecodeKernel, raw q/k are normalized here in f32 and never
// materialized through the activation dtype. sigmoid(b) is rounded through
// b.dtype before recurrence, exactly at the upstream store/load boundary.
void GdnPackedDecodeKernel(Queue&, Tensor& out, const Tensor& mixed_qkv,
                           const Tensor& a, const Tensor& b,
                           const Tensor& a_log, const Tensor& dt_bias,
                           Tensor& state, const Tensor& state_idx,
                           const GdnArgs& args) {
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv_n = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  const int64_t qk_dim = mixed_qkv.shape[1] - hv_n * dv;
  const int64_t hk_n = qk_dim / (2 * dk);
  const int64_t ratio = hv_n / hk_n;
  const int32_t* indices = state_idx.Ptr<int32_t>();

  ForRows(batch, [&](int64_t r0, int64_t r1) {
    std::vector<float> qv(static_cast<size_t>(dk));
    std::vector<float> kv(static_cast<size_t>(dk));
    std::vector<float> state_row(static_cast<size_t>(dk));
    for (int64_t bt = r0; bt < r1; ++bt) {
      const int32_t slot = indices[bt];
      if (slot < 0) {
        for (int64_t i = 0; i < hv_n * dv; ++i)
          StoreF32(out, bt * hv_n * dv + i, 0.0f);
        continue;
      }
      for (int64_t hv = 0; hv < hv_n; ++hv) {
        const int64_t hk = hv / ratio;
        float q_sumsq = 0.0f;
        float k_sumsq = 0.0f;
        for (int64_t ki = 0; ki < dk; ++ki) {
          qv[static_cast<size_t>(ki)] =
              LoadF32(mixed_qkv, bt * mixed_qkv.stride[0] + hk * dk + ki);
          kv[static_cast<size_t>(ki)] =
              LoadF32(mixed_qkv, bt * mixed_qkv.stride[0] + hk_n * dk + hk * dk + ki);
          q_sumsq += qv[static_cast<size_t>(ki)] * qv[static_cast<size_t>(ki)];
          k_sumsq += kv[static_cast<size_t>(ki)] * kv[static_cast<size_t>(ki)];
        }
        const float q_inv = 1.0f / std::sqrt(q_sumsq + 1e-6f);
        const float k_inv = 1.0f / std::sqrt(k_sumsq + 1e-6f);
        for (int64_t ki = 0; ki < dk; ++ki) {
          qv[static_cast<size_t>(ki)] *= q_inv * args.scale;
          kv[static_cast<size_t>(ki)] *= k_inv;
        }

        const float av = LoadF32(a, bt * a.stride[0] + hv);
        const float bv = LoadF32(b, bt * b.stride[0] + hv);
        const float x = av + LoadF32(dt_bias, hv);
        const float softplus =
            x <= 20.0f ? std::log1p(std::exp(std::min(x, 20.0f))) : x;
        const float g = -std::exp(LoadF32(a_log, hv)) * softplus;
        float beta = 1.0f / (1.0f + std::exp(-bv));
        if (b.dtype == DType::kF16)
          beta = F16ToF32(F32ToF16(beta));
        else if (b.dtype == DType::kBF16)
          beta = BF16ToF32(F32ToBF16(beta));
        const float decay = std::exp(g);

        for (int64_t vi = 0; vi < dv; ++vi) {
          const int64_t state_base =
              ((static_cast<int64_t>(slot) * hv_n + hv) * dv + vi) * dk;
          float dot = 0.0f;
          for (int64_t ki = 0; ki < dk; ++ki) {
            state_row[static_cast<size_t>(ki)] =
                LoadF32(state, state_base + ki) * decay;
            dot += state_row[static_cast<size_t>(ki)] * kv[static_cast<size_t>(ki)];
          }
          const int64_t v_offset = bt * mixed_qkv.stride[0] +
                                   2 * hk_n * dk + hv * dv + vi;
          const float vp = (LoadF32(mixed_qkv, v_offset) - dot) * beta;
          float output = 0.0f;
          for (int64_t ki = 0; ki < dk; ++ki) {
            const float updated = state_row[static_cast<size_t>(ki)] +
                                  vp * kv[static_cast<size_t>(ki)];
            StoreF32(state, state_base + ki, updated);
            output += updated * qv[static_cast<size_t>(ki)];
          }
          StoreF32(out, (bt * hv_n + hv) * dv + vi, output);
        }
      }
    }
  });
}

// Indexed cache boundary for mixed GDN prefill. This is the CPU executable
// reference for the fused CUDA gather/scatter kernels: cache rows may be f32 or
// bf16, while the compact recurrence/conv working state is always f32.
// GDN state gather/scatter. The cache's INNERMOST dim may be WIDER than the
// working buffer's when the GDN conv state is widened to (K-1)+num_spec taps for
// spec-decode rollback: the non-spec op then operates on the LEADING working
// inner-dim columns per channel, with the cache row's PHYSICAL stride (mirror
// vLLM `state_len=KERNEL_WIDTH-1` + physical `stride_conv_state_tok`,
// causal_conv1d.py:67-69). At num_spec==0 (every non-spec model, and the SSM
// state cache which is never widened) cache_inner==work_inner, so the contiguous
// fast path below runs — byte-for-byte the pre-spec kernel.
void GdnStateGatherKernel(Queue&, Tensor& working, const Tensor& cache,
                          const Tensor& state_idx,
                          const Tensor* has_initial_state) {
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int32_t* idx = state_idx.Ptr<int32_t>();
  for (int64_t r = 0; r < rows; ++r) {
    VT_CHECK(idx[r] >= 0 && idx[r] < cache.shape[0],
             "gdn_state_gather: state_idx out of range");
  }
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int64_t work_row = working.Numel() / rows;   // = mid * work_inner
  const int64_t mid = work_row / work_inner;          // channels/heads per row
  const int64_t cache_row = mid * cache_inner;         // physical cache row width
  ForRows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      bool keep = true;
      if (has_initial_state != nullptr) {
        keep = has_initial_state->dtype == DType::kI8
                   ? has_initial_state->Ptr<int8_t>()[r] != 0
                   : has_initial_state->Ptr<int32_t>()[r] != 0;
      }
      const int64_t src_row = static_cast<int64_t>(idx[r]) * cache_row;
      const int64_t dst_row = r * work_row;
      if (cache_inner == work_inner) {  // contiguous fast path (num_spec==0)
        for (int64_t e = 0; e < work_row; ++e) {
          StoreF32(working, dst_row + e, keep ? LoadF32(cache, src_row + e) : 0.0f);
        }
      } else {  // widened cache: leading work_inner cols per channel
        for (int64_t m = 0; m < mid; ++m) {
          const int64_t src = src_row + m * cache_inner;
          const int64_t dst = dst_row + m * work_inner;
          for (int64_t e = 0; e < work_inner; ++e) {
            StoreF32(working, dst + e, keep ? LoadF32(cache, src + e) : 0.0f);
          }
        }
      }
    }
  });
}

void GdnStateScatterKernel(Queue&, Tensor& cache, const Tensor& working,
                           const Tensor& state_idx) {
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int32_t* idx = state_idx.Ptr<int32_t>();
  for (int64_t r = 0; r < rows; ++r) {
    VT_CHECK(idx[r] >= 0 && idx[r] < cache.shape[0],
             "gdn_state_scatter: state_idx out of range");
  }
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int64_t work_row = working.Numel() / rows;
  const int64_t mid = work_row / work_inner;
  const int64_t cache_row = mid * cache_inner;
  ForRows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t src_row = r * work_row;
      const int64_t dst_row = static_cast<int64_t>(idx[r]) * cache_row;
      if (cache_inner == work_inner) {  // contiguous fast path (num_spec==0)
        for (int64_t e = 0; e < work_row; ++e) {
          StoreF32(cache, dst_row + e, LoadF32(working, src_row + e));
        }
        continue;
      }
      // widened cache: write leading work_inner cols per channel (physical stride)
      for (int64_t m = 0; m < mid; ++m) {
        const int64_t src = src_row + m * work_inner;
        const int64_t dst = dst_row + m * cache_inner;
        for (int64_t e = 0; e < work_inner; ++e) {
          StoreF32(cache, dst + e, LoadF32(working, src + e));
        }
      }
    }
  });
}

// Row gather over dim 0: out[i,...] = in[idx[i],...] (torch index_select).
// dtype-agnostic byte copy per row; the base (in) may carry an outer row stride.
void IndexSelectKernel(Queue&, Tensor& out, const Tensor& in,
                       const Tensor& idx) {
  const int64_t rows = idx.shape[0];
  if (rows == 0) return;
  const int32_t* ix = idx.Ptr<int32_t>();
  const int64_t inner = out.Numel() / rows;                 // packed row elems
  const size_t esz = SizeOf(out.dtype);
  const size_t row_bytes = static_cast<size_t>(inner) * esz;
  const int64_t in_row_stride = in.stride[0];               // elements
  auto* dst = static_cast<char*>(out.data);
  const auto* src = static_cast<const char*>(in.data);
  for (int64_t i = 0; i < rows; ++i) {
    VT_CHECK(ix[i] >= 0 && ix[i] < in.shape[0],
             "index_select: idx out of range");
    std::memcpy(dst + static_cast<size_t>(i) * row_bytes,
                src + static_cast<size_t>(ix[i]) * static_cast<size_t>(in_row_stride) * esz,
                row_bytes);
  }
}

// Row scatter over dim 0: out[idx[i],...] = in[i,...] (torch index_copy_).
void IndexCopyKernel(Queue&, Tensor& out, const Tensor& in, const Tensor& idx) {
  const int64_t rows = idx.shape[0];
  if (rows == 0) return;
  const int32_t* ix = idx.Ptr<int32_t>();
  const int64_t inner = in.Numel() / rows;
  const size_t esz = SizeOf(in.dtype);
  const size_t row_bytes = static_cast<size_t>(inner) * esz;
  const int64_t out_row_stride = out.stride[0];             // elements
  auto* dst = static_cast<char*>(out.data);
  const auto* src = static_cast<const char*>(in.data);
  for (int64_t i = 0; i < rows; ++i) {
    VT_CHECK(ix[i] >= 0 && ix[i] < out.shape[0],
             "index_copy: idx out of range");
    std::memcpy(dst + static_cast<size_t>(ix[i]) * static_cast<size_t>(out_row_stride) * esz,
                src + static_cast<size_t>(i) * row_bytes, row_bytes);
  }
}

// Grouped-topk (`noaux_tc`) router — the CPU REFERENCE and the executable spec
// for the DeepSeek router. 1:1 port of
// vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:106-161
// (`grouped_topk`, the `forward_native` path) @ pin e24d1b24. Step numbering in
// the comments matches the ops.h contract. Reached ONLY when
// args.num_expert_group > 0; the ungrouped kernel below is untouched.
void MoeRouterGroupedTopKKernel(Tensor& weights, Tensor& indices, const Tensor& logits,
                                const MoeRouterTopKArgs& args, const Tensor* bias_t) {
  const int64_t t = logits.shape[0], e = logits.shape[1];
  const int k = args.top_k;
  const int64_t n_group = args.num_expert_group;
  const int64_t group_size = e / n_group;  // wrapper checked divisibility
  const float* bias = bias_t != nullptr ? bias_t->Ptr<float>() : nullptr;

  ForRows(t, [&](int64_t r0, int64_t r1) {
    std::vector<float> sel(static_cast<size_t>(e));   // SELECTION score (biased)
    std::vector<float> orig(static_cast<size_t>(e));  // WEIGHT score (unbiased)
    std::vector<float> gscore(static_cast<size_t>(n_group));
    std::vector<char> gkeep(static_cast<size_t>(n_group));
    for (int64_t row = r0; row < r1; ++row) {
      // (1) scores = softmax(logits, -1) | sigmoid(logits)   (:110-117)
      if (args.scoring_func == MoeScoringFunc::kSigmoid) {
        // ELEMENTWISE — not normalized across experts. This is the V3/R1 path.
        for (int64_t j = 0; j < e; ++j) {
          orig[static_cast<size_t>(j)] = 1.0f / (1.0f + std::exp(-LoadF32(logits, row * e + j)));
        }
      } else {
        float mx = -INFINITY;
        for (int64_t j = 0; j < e; ++j) mx = std::max(mx, LoadF32(logits, row * e + j));
        float sum = 0.0f;
        for (int64_t j = 0; j < e; ++j) {
          const float ex = std::exp(LoadF32(logits, row * e + j) - mx);
          orig[static_cast<size_t>(j)] = ex;
          sum += ex;
        }
        for (int64_t j = 0; j < e; ++j) {
          float& pj = orig[static_cast<size_t>(j)];
          pj = sum > 0.0f ? pj / sum : 0.0f;
          if (!std::isfinite(pj)) pj = 0.0f;
        }
      }
      // (2) The BIAS asymmetry (:120-124): the bias shifts the score used for
      // SELECTION only; the routing WEIGHT is always read from the unbiased
      // score. Getting this backwards is a silent accuracy bug.
      for (int64_t j = 0; j < e; ++j) {
        sel[static_cast<size_t>(j)] =
            orig[static_cast<size_t>(j)] + (bias != nullptr ? bias[j] : 0.0f);
      }
      // Group score: SUM of the top-2 in the group when a bias is present
      // (:124-126), else the group MAX (:128-131).
      for (int64_t g = 0; g < n_group; ++g) {
        const int64_t base = g * group_size;
        if (bias != nullptr) {
          float b0 = -INFINITY, b1 = -INFINITY;
          for (int64_t j = 0; j < group_size; ++j) {
            const float v = sel[static_cast<size_t>(base + j)];
            if (v > b0) {
              b1 = b0;
              b0 = v;
            } else if (v > b1) {
              b1 = v;
            }
          }
          // group_size >= 2 whenever a bias is used in practice (V3: 32); if it
          // were 1, torch's topk(2) would fail upstream too, so b1 stays -inf
          // and the sum is -inf — an honest propagation, not a silent 0.
          gscore[static_cast<size_t>(g)] = b0 + b1;
        } else {
          float m = -INFINITY;
          for (int64_t j = 0; j < group_size; ++j) {
            m = std::max(m, sel[static_cast<size_t>(base + j)]);
          }
          gscore[static_cast<size_t>(g)] = m;
        }
      }
      // (3) keep the top `topk_group` groups, mask the rest to -inf (:133-145).
      // Greedy argmax with a strict `>` ascending scan — lowest group index wins
      // an exact tie (our determinism convention; see the ops.h deviation note).
      for (int64_t g = 0; g < n_group; ++g) gkeep[static_cast<size_t>(g)] = 0;
      for (int gi = 0; gi < args.topk_group; ++gi) {
        int64_t best = -1;
        float best_v = -INFINITY;
        for (int64_t g = 0; g < n_group; ++g) {
          if (gkeep[static_cast<size_t>(g)]) continue;
          // The FIRST unkept index seeds the scan, so an all-`-inf` group row
          // still selects (the lowest unkept index) instead of falling through.
          if (best < 0 || gscore[static_cast<size_t>(g)] > best_v) {
            best_v = gscore[static_cast<size_t>(g)];
            best = g;
          }
        }
        if (best < 0) break;  // fewer groups than topk_group (wrapper forbids it)
        gkeep[static_cast<size_t>(best)] = 1;
      }
      for (int64_t g = 0; g < n_group; ++g) {
        if (gkeep[static_cast<size_t>(g)]) continue;
        for (int64_t j = 0; j < group_size; ++j) {
          sel[static_cast<size_t>(g * group_size + j)] = -INFINITY;
        }
      }
      // (4) top-k over the masked SELECTION scores; the weight comes from the
      // UNBIASED score at the selected id (:147-150).
      float denom = 0.0f;
      for (int j = 0; j < k; ++j) {
        int64_t best = -1;
        float best_v = -INFINITY;
        for (int64_t idx = 0; idx < e; ++idx) {
          if (sel[static_cast<size_t>(idx)] > best_v) {
            best_v = sel[static_cast<size_t>(idx)];
            best = idx;
          }
        }
        if (best < 0) best = 0;
        sel[static_cast<size_t>(best)] = -INFINITY;  // exclude from later rounds
        const float w = orig[static_cast<size_t>(best)];
        weights.Ptr<float>()[row * k + j] = w;
        indices.Ptr<int32_t>()[row * k + j] = static_cast<int32_t>(best);
        denom += w;
      }
      // (5) renormalize (:156-157) then routed_scaling_factor (:159-160). Order
      // matters: upstream scales AFTER the renormalize divide.
      if (args.renormalize) {
        if (!(denom > 0.0f)) denom = 1.0f;
        for (int j = 0; j < k; ++j) weights.Ptr<float>()[row * k + j] /= denom;
      }
      if (args.routed_scaling_factor != 1.0f) {
        for (int j = 0; j < k; ++j) {
          weights.Ptr<float>()[row * k + j] *= args.routed_scaling_factor;
        }
      }
    }
  });
}

// §3 router: softmax (f32, over all E) -> greedy top-k (lowest-index tie-break)
// -> optional renormalize. weights [T,K] f32, indices [T,K] i32.
void MoeRouterTopKKernel(Queue&, Tensor& weights, Tensor& indices, const Tensor& logits,
                         const MoeRouterTopKArgs& args, const Tensor* bias) {
  if (args.num_expert_group > 0) {  // W3 grouped-topk (`noaux_tc`) path
    MoeRouterGroupedTopKKernel(weights, indices, logits, args, bias);
    return;
  }
  const int64_t t = logits.shape[0], e = logits.shape[1];
  const int k = args.top_k;
  ForRows(t, [&](int64_t r0, int64_t r1) {
  std::vector<float> p(static_cast<size_t>(e));
  std::vector<char> chosen(static_cast<size_t>(e));
  for (int64_t row = r0; row < r1; ++row) {
    // softmax(logits.float()) with max-subtraction (topk_softmax_kernels.cu).
    float mx = -INFINITY;
    for (int64_t j = 0; j < e; ++j) mx = std::max(mx, LoadF32(logits, row * e + j));
    float sum = 0.0f;
    for (int64_t j = 0; j < e; ++j) {
      const float ex = std::exp(LoadF32(logits, row * e + j) - mx);
      p[static_cast<size_t>(j)] = ex;
      sum += ex;
    }
    for (int64_t j = 0; j < e; ++j) {
      float& pj = p[static_cast<size_t>(j)];
      pj = sum > 0.0f ? pj / sum : 0.0f;
      if (!std::isfinite(pj)) pj = 0.0f;  // NaN/Inf clamp (.cu:136)
      chosen[static_cast<size_t>(j)] = 0;
    }
    // Greedy argmax, k rounds; strict `>` over ascending j -> lowest index wins.
    float denom = 0.0f;
    for (int j = 0; j < k; ++j) {
      int64_t best = -1;
      float best_v = -INFINITY;
      for (int64_t idx = 0; idx < e; ++idx) {
        if (chosen[static_cast<size_t>(idx)]) continue;
        if (p[static_cast<size_t>(idx)] > best_v) {
          best_v = p[static_cast<size_t>(idx)];
          best = idx;
        }
      }
      chosen[static_cast<size_t>(best)] = 1;
      weights.Ptr<float>()[row * k + j] = best_v;
      indices.Ptr<int32_t>()[row * k + j] = static_cast<int32_t>(best);
      denom += best_v;
    }
    if (args.renormalize) {
      if (!(denom > 0.0f)) denom = 1.0f;  // (.cu:245-253) denom<=0 -> 1 guard
      for (int j = 0; j < k; ++j) weights.Ptr<float>()[row * k + j] /= denom;
    }
  }
  });
}

// §4/§6 weighted scatter-combine: out[t,:] = sum_j w[t,j]*expert_out[t,j,:]
// (f32 accumulation) + shared[t,:] (optional). Stored at out's dtype.
// `routed_scale` multiplies the ROUTED sum only, BEFORE the shared term is added
// — upstream's apply_routed_scale_to_output arm (moe_runner.py:390-407, :402-406 scales
// `fused_output`, leaves `shared_output` alone, then :722-725 adds them). The
// default 1.0f is the fold-into-router-weights polarity every landed caller uses.
// It scales the ASSEMBLED sum, not each router weight: upstream's
// `fused_output *= factor` (:404) is one multiply on the finished tensor, so the
// scale rounds ONCE after the K-term reduction. Folding it into `weights[j]` is
// equal in exact arithmetic and a different f32 value (it rounds K times inside
// the sum); Laguna is entitled to that fold (`laguna_ops.h:48`, no `shared`),
// this path is not. Pinned bitwise in test_ops_moe_nongated_relu2.cpp.
// NOT MIRRORED, UNREACHABLE: upstream's fp16 arm (:403-406) instead divides
// `shared_output` by the factor to dodge an fp16 overflow. That branch is keyed
// on `fused_output.dtype == torch.float16`; the analogue here is `out`, whose
// dtype `MoeCombine` gates through `IsOutFloat` (ops.cpp:22 — f32/bf16 only, no
// kF16), so no caller can reach it. Pinned by the f16-out refusal test.
void MoeCombineKernel(Queue&, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                      const Tensor* shared, float routed_scale) {
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t row = r0; row < r1; ++row) {
    for (int64_t col = 0; col < h; ++col) {
      float acc = 0.0f;
      for (int64_t j = 0; j < k; ++j)
        acc += weights.Ptr<float>()[row * k + j] *
               LoadF32(expert_out, (row * k + j) * h + col);
      if (routed_scale != 1.0f) acc *= routed_scale;
      if (shared != nullptr) acc += LoadF32(*shared, row * h + col);
      StoreF32(out, row * h + col, acc);
    }
  }
  });
}

// Dense causal attention (qwen36-forward-notes.md §5). Causal scaled-dot-product
// with GQA broadcast over a single packed sequence. query [T,Hq,D],
// key/value [T,Hk,D], out [T,Hq,D]. Per q-head h (kv-head g = h/(Hq/Hk)) and
// query i: softmax over keys j<=i of scale*(q·k), then weighted sum of v. f32
// softmax with online max-subtraction for numerical stability.
void AttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const AttentionArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  const int64_t qpk = hq / hk;  // q-heads per kv-head (GQA ratio)
  const float scale = args.scale;
  // Row-chunked over (head, query) pairs (spec W3) — the flash-attn q-row
  // split (ops.cpp:9072-9073 nr = neq1*neq2*neq3). Same h-outer/i-inner walk.
  ForRows(hq * t, [&](int64_t r0, int64_t r1) {
  std::vector<float> probs(static_cast<size_t>(t));
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t r = r0; r < r1; ++r) {
    const int64_t h = r / t;
    const int64_t g = h / qpk;
    {
      const int64_t i = r % t;
      const int64_t jmax = args.causal ? i : t - 1;  // causal: keys 0..i
      const int64_t qoff = (i * hq + h) * d;
      // Pass 1: scores + running max.
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j <= jmax; ++j) {
        const int64_t koff = (j * hk + g) * d;
        float dot = 0.0f;
        for (int64_t e = 0; e < d; ++e) dot += LoadF32(query, qoff + e) * LoadF32(key, koff + e);
        dot *= scale;
        probs[static_cast<size_t>(j)] = dot;
        if (dot > m) m = dot;
      }
      // Pass 2: exp + normalization denominator.
      float denom = 0.0f;
      for (int64_t j = 0; j <= jmax; ++j) {
        float e = std::exp(probs[static_cast<size_t>(j)] - m);
        probs[static_cast<size_t>(j)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;  // denom >= 1 (j==i term is exp(0)=1)
      // Pass 3: weighted sum of v (f32 accumulation), stored at out's dtype.
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = 0; j <= jmax; ++j) {
        const float p = probs[static_cast<size_t>(j)] * inv;
        const int64_t voff = (j * hk + g) * d;
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * LoadF32(value, voff + e);
      }
      for (int64_t e = 0; e < d; ++e) StoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
    }
  }
  });
}

// Dense non-causal CROSS attention (LTX-2.5 L2) — the CPU REFERENCE. Same three
// pass structure as AttentionKernel, with the key extent taken from KEY's own
// token count (not query's) and an optional additive score bias. Semantics
// ported from torch's `scaled_dot_product_attention(q,k,v,attn_mask=...,
// is_causal=False)` as LTX's PytorchAttention calls it
// (ltx_core/model/transformer/attention.py:97-102).
void AttentionCrossKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const Tensor* bias,
                          const AttentionCrossArgs& args) {
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t s = key.shape[0], hk = key.shape[1];
  const int64_t qpk = hq / hk;  // q-heads per kv-head (GQA ratio)
  const float scale = args.scale;
  const float* bias_data = bias != nullptr ? bias->Ptr<float>() : nullptr;
  const int64_t bias_rows = bias != nullptr ? bias->shape[0] : 0;
  ForRows(hq * tq, [&](int64_t r0, int64_t r1) {
  std::vector<float> probs(static_cast<size_t>(s));
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t r = r0; r < r1; ++r) {
    const int64_t h = r / tq;
    const int64_t g = h / qpk;
    const int64_t i = r % tq;
    const int64_t qoff = (i * hq + h) * d;
    // The bias row this query reads: its own, or the single broadcast row.
    const float* brow = bias_data == nullptr ? nullptr
                                             : bias_data + (bias_rows == 1 ? 0 : i) * s;
    // Pass 1: scores + running max.
    float m = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < s; ++j) {
      const int64_t koff = (j * hk + g) * d;
      float dot = 0.0f;
      for (int64_t e = 0; e < d; ++e) dot += LoadF32(query, qoff + e) * LoadF32(key, koff + e);
      dot *= scale;
      if (brow != nullptr) dot += brow[j];
      probs[static_cast<size_t>(j)] = dot;
      if (dot > m) m = dot;
    }
    // Pass 2: exp + normalization denominator.
    float denom = 0.0f;
    for (int64_t j = 0; j < s; ++j) {
      const float e = std::exp(probs[static_cast<size_t>(j)] - m);
      probs[static_cast<size_t>(j)] = e;
      denom += e;
    }
    const float inv = 1.0f / denom;  // denom >= 1 (the argmax term is exp(0)=1)
    // Pass 3: weighted sum of v (f32 accumulation), stored at out's dtype.
    for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
    for (int64_t j = 0; j < s; ++j) {
      const float p = probs[static_cast<size_t>(j)] * inv;
      const int64_t voff = (j * hk + g) * d;
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * LoadF32(value, voff + e);
    }
    for (int64_t e = 0; e < d; ++e) StoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
  }
  });
}

// DFlash in-block attention (SPEC-DFLASH D2, DF-DRAFT-MODEL) — the CPU REFERENCE
// for the project's first non-causal / bidirectional attention. Semantics ported
// from DFlashQwen3Attention + _resolve_layer_attention (qwen3_dflash.py:86-146,
// 149-263 @ 555967922). For each request block [cu[r],cu[r+1]) and query i in the
// block, attend over keys j in the SAME block:
//   - full-attention layer (args.causal==false): ALL j (BIDIRECTIONAL, no mask);
//   - SWA layer (args.causal==true): j <= i AND (window<=0 || j >= i-(window-1)).
// f32 online softmax (max-subtracted), GQA broadcast (q-head h reads kv-head
// h/(Hq/Hk)). Same three-pass structure as AttentionKernel, generalized to
// per-block bounds + a bidirectional/window mask. This is the authoritative
// reference; the CUDA kernel mirrors this recurrence.
void DFlashBlockAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& key,
                                const Tensor& value, const DFlashBlockAttentionArgs& args) {
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  const int64_t qpk = hq / hk;  // q-heads per kv-head (GQA ratio)
  const float scale = args.scale;
  const int64_t window = args.sliding_window;
  const bool causal = args.causal;
  const int num_reqs = args.num_reqs;
  const int32_t* cu = args.cu_seqlens;
  // Row-chunked over (head, request-block, query) triples. Each row does its own
  // block-local softmax; blocks never attend across their boundary.
  ForRows(hq * static_cast<int64_t>(num_reqs), [&](int64_t r0, int64_t r1) {
    std::vector<float> probs;
    std::vector<float> acc(static_cast<size_t>(d));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t h = r % hq;
      const int64_t req = r / hq;
      const int64_t g = h / qpk;
      const int64_t qs = cu[req];
      const int64_t qe = cu[req + 1];
      const int64_t blen = qe - qs;
      probs.resize(static_cast<size_t>(blen));
      for (int64_t ii = 0; ii < blen; ++ii) {
        const int64_t i = qs + ii;  // global row; intra-block offset = ii
        // Visible key range within the block (intra-block offsets [jlo,jhi]).
        const int64_t jhi = causal ? ii : blen - 1;
        int64_t jlo = 0;
        if (causal && window > 0) jlo = ii - (window - 1) > 0 ? ii - (window - 1) : 0;
        const int64_t qoff = (i * hq + h) * d;
        // Pass 1: scores + running max.
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t jj = jlo; jj <= jhi; ++jj) {
          const int64_t koff = ((qs + jj) * hk + g) * d;
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e) dot += LoadF32(query, qoff + e) * LoadF32(key, koff + e);
          dot *= scale;
          probs[static_cast<size_t>(jj)] = dot;
          if (dot > m) m = dot;
        }
        // Pass 2: exp + denom.
        float denom = 0.0f;
        for (int64_t jj = jlo; jj <= jhi; ++jj) {
          float e = std::exp(probs[static_cast<size_t>(jj)] - m);
          probs[static_cast<size_t>(jj)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;  // >= 1 (the j==i term is exp(0)=1 for both masks)
        // Pass 3: weighted sum of v.
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
        for (int64_t jj = jlo; jj <= jhi; ++jj) {
          const float p = probs[static_cast<size_t>(jj)] * inv;
          const int64_t voff = ((qs + jj) * hk + g) * d;
          for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * LoadF32(value, voff + e);
        }
        for (int64_t e = 0; e < d; ++e) StoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
      }
    }
  });
}

// DFlash PAGED in-block attention (SPEC-DFLASH D12 Part B) — CPU reference. Each
// (1+k) block query attends over [PAGED context ; its own (1+k) block] with the D2
// mask over the COMBINED index (context rows [0,C) then block rows [C, C+blen)).
// Bit-identical math to DFlashBlockAttentionKernel over a materialized combined
// buffer: same ascending key order, same f32 online (2-pass) softmax. The context
// enters as a paged K/V cache + per-request seq_lens + block_table (DATA), matching
// the capture-safe CUDA kernel.
void DFlashPagedBlockAttentionKernel(Queue&, Tensor& out, const Tensor& query,
                                     const Tensor& block_key, const Tensor& block_value,
                                     const Tensor& ctx_key, const Tensor& ctx_value,
                                     const Tensor& cu_seqlens, const Tensor& seq_lens,
                                     const Tensor& block_table,
                                     const DFlashPagedBlockAttentionArgs& args) {
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t hk = block_key.shape[1];
  const int64_t qpk = hq / hk;  // GQA ratio
  const float scale = args.scale;
  const int64_t window = args.sliding_window;
  const bool causal = args.causal;
  const int num_reqs = args.num_reqs;
  const int64_t block_size = args.block_size;
  const int64_t max_pages = block_table.shape[1];
  // Contiguous paged cache strides: [num_pages, block_size, Hkv, D].
  const int64_t ck_blk = block_size * hk * d, ck_pg = hk * d, ck_hd = d;
  const int32_t* cu = cu_seqlens.Ptr<int32_t>();
  const int32_t* slen = seq_lens.Ptr<int32_t>();
  const int32_t* btbl = block_table.Ptr<int32_t>();
  ForRows(hq * static_cast<int64_t>(num_reqs), [&](int64_t r0, int64_t r1) {
    std::vector<float> probs;
    std::vector<float> acc(static_cast<size_t>(d));
    for (int64_t rr = r0; rr < r1; ++rr) {
      const int64_t h = rr % hq;
      const int64_t req = rr / hq;
      const int64_t g = h / qpk;
      const int64_t qs = cu[req];
      const int64_t qe = cu[req + 1];
      const int64_t blen = qe - qs;
      const int64_t C = slen[req];
      const int64_t N = C + blen;  // combined key length
      probs.resize(static_cast<size_t>(N));
      for (int64_t ii = 0; ii < blen; ++ii) {
        const int64_t i = qs + ii;          // global block-query row
        const int64_t ii_comb = C + ii;     // query offset in the combined sequence
        const int64_t jhi = causal ? ii_comb : N - 1;
        int64_t jlo = 0;
        if (causal && window > 0) jlo = ii_comb - (window - 1) > 0 ? ii_comb - (window - 1) : 0;
        const int64_t qoff = (i * hq + h) * d;
        // Pass 1: scores + running max over combined keys [jlo, jhi].
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t cj = jlo; cj <= jhi; ++cj) {
          int64_t koff;
          const Tensor* ksrc;
          if (cj < C) {  // paged context key
            const int64_t page = btbl[req * max_pages + cj / block_size];
            const int64_t off = cj % block_size;
            koff = page * ck_blk + off * ck_pg + g * ck_hd;
            ksrc = &ctx_key;
          } else {  // block key (contiguous)
            const int64_t brow = qs + (cj - C);
            koff = (brow * hk + g) * d;
            ksrc = &block_key;
          }
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e) dot += LoadF32(query, qoff + e) * LoadF32(*ksrc, koff + e);
          dot *= scale;
          probs[static_cast<size_t>(cj)] = dot;
          if (dot > m) m = dot;
        }
        // Pass 2: exp + denom.
        float denom = 0.0f;
        for (int64_t cj = jlo; cj <= jhi; ++cj) {
          float e = std::exp(probs[static_cast<size_t>(cj)] - m);
          probs[static_cast<size_t>(cj)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;
        // Pass 3: weighted sum of v.
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
        for (int64_t cj = jlo; cj <= jhi; ++cj) {
          const float pw = probs[static_cast<size_t>(cj)] * inv;
          int64_t voff;
          const Tensor* vsrc;
          if (cj < C) {
            const int64_t page = btbl[req * max_pages + cj / block_size];
            const int64_t off = cj % block_size;
            voff = page * ck_blk + off * ck_pg + g * ck_hd;
            vsrc = &ctx_value;
          } else {
            const int64_t brow = qs + (cj - C);
            voff = (brow * hk + g) * d;
            vsrc = &block_value;
          }
          for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += pw * LoadF32(*vsrc, voff + e);
        }
        for (int64_t e = 0; e < d; ++e) StoreF32(out, qoff + e, acc[static_cast<size_t>(e)]);
      }
    }
  });
}

// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). Elementwise fusions of the
// small host-side loops between the big decode ops; all math f32, dims inferred
// from the tensor shapes.

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// out[i] = F32ToBF16(in[i]); out bf16, in f32, same element count.
void CastBf16Kernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  ForRows(n, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) StoreF32(out, i, LoadF32(in, i));
  });
}

// out[i] = (float)in[i]; out f32, in bf16, same element count. CPU sibling of
// the CUDA CastF32 kernel (bf16 -> f32 widen); LoadF32 reads the bf16 source as
// f32 and StoreF32 writes it to the f32 destination (widening is exact).
void CastF32Kernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  ForRows(n, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) StoreF32(out, i, LoadF32(in, i));
  });
}

// x[m,n] *= col[n]; x f32 OR bf16 [M,N] (inner-contiguous rows, row stride
// x.stride[0]), col always f32 [N]. CPU sibling of the CUDA MulColVecF32 kernel,
// and the portable reference every other backend ports FROM — so it carries the
// same bf16 store width the CUDA kernel gained for PERF-FP8-ALPHA-FOLD / #417,
// not a CUDA-only capability.
//
// Both arms do the SAME single IEEE f32 multiply; only the store differs. The
// f32 arm keeps the direct `*=` loop byte-for-byte (it is the shipped path and
// the hot one), and the bf16 arm goes through LoadF32/StoreF32, whose F32ToBF16
// is round-to-nearest-even — the same rounding __float2bfloat16 applies on CUDA,
// which is what keeps the two tiers comparable element for element.
void MulColVecF32Kernel(Queue&, Tensor& x, const Tensor& col) {
  const int64_t m = x.shape[0], n = x.shape[1], rs = x.stride[0];
  const float* c = col.Ptr<float>();
  if (x.dtype == DType::kF32) {
    ForRows(m, [&](int64_t r0, int64_t r1) {
      for (int64_t i = r0; i < r1; ++i) {
        float* row = x.Ptr<float>() + i * rs;
        for (int64_t j = 0; j < n; ++j) row[j] *= c[j];
      }
    });
    return;
  }
  VT_CHECK(x.dtype == DType::kBF16, "cpu mul_col_vec_f32: x must be f32 or bf16");
  ForRows(m, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) {
      const int64_t base = i * rs;
      for (int64_t j = 0; j < n; ++j)
        StoreF32(x, base + j, LoadF32(x, base + j) * c[j]);
    }
  });
}

// Split fused [T, Hq*2*Dh] q/gate projection into q_out/gate_out [T,Hq,Dh].
void AttnGateSplitKernel(Queue&, Tensor& q_out, Tensor& gate_out, const Tensor& qgate) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    for (int64_t h = 0; h < hq; ++h) {
      const int64_t base = i * (hq * 2 * dh) + h * 2 * dh;  // start of (i,h) pair
      const int64_t out_off = (i * hq + h) * dh;
      for (int64_t d = 0; d < dh; ++d) {
        StoreF32(q_out, out_off + d, LoadF32(qgate, base + d));
        StoreF32(gate_out, out_off + d, LoadF32(qgate, base + dh + d));
      }
    }
  }
  });
}

// out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])); out bf16, attn/gate f32.
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  ForRows(n, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i)
      StoreF32(out, i, LoadF32(attn, i) * Sigmoid(LoadF32(gate, i)));
  });
}

// GDN g/beta from raw projections (gdn-semantics.md §6). softplus threshold 20.
void GdnGBetaKernel(Queue&, Tensor& g_out, Tensor& beta_out, const Tensor& araw,
                    const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias) {
  const int64_t t = g_out.shape[0], hv = g_out.shape[1];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    for (int64_t h = 0; h < hv; ++h) {
      const int64_t idx = i * hv + h;
      const int64_t aidx = i * araw.stride[0] + h;
      const int64_t bidx = i * braw.stride[0] + h;
      const float x = LoadF32(araw, aidx) + LoadF32(dt_bias, h);
      const float sp = x > 20.0f ? x : std::log1p(std::exp(x));  // softplus
      StoreF32(g_out, idx, -std::exp(LoadF32(a_log, h)) * sp);
      StoreF32(beta_out, idx, Sigmoid(LoadF32(braw, bidx)));
    }
  }
  });
}

// Split GDN conv [T, 2*key_dim+value_dim] into q/k [T,key_dim] and v [T,value_dim].
void GdnConvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& conv) {
  const int64_t t = conv.shape[0], key_dim = q_out.Numel() / t, value_dim = v_out.Numel() / t;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const int64_t row = i * conv_dim;
    for (int64_t j = 0; j < key_dim; ++j) {
      StoreF32(q_out, i * key_dim + j, LoadF32(conv, row + j));
      StoreF32(k_out, i * key_dim + j, LoadF32(conv, row + key_dim + j));
    }
    for (int64_t j = 0; j < value_dim; ++j)
      StoreF32(v_out, i * value_dim + j, LoadF32(conv, row + 2 * key_dim + j));
  }
  });
}

// QkvSplit: GQA merged-qkv split with INDEPENDENT q/k/v dims (q_dim != k_dim).
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  const int64_t t = qkv.shape[0], q_dim = q_out.Numel() / t, k_dim = k_out.Numel() / t,
                v_dim = v_out.Numel() / t;
  const int64_t total = q_dim + k_dim + v_dim;
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const int64_t row = i * total;
    for (int64_t j = 0; j < q_dim; ++j) StoreF32(q_out, i * q_dim + j, LoadF32(qkv, row + j));
    for (int64_t j = 0; j < k_dim; ++j)
      StoreF32(k_out, i * k_dim + j, LoadF32(qkv, row + q_dim + j));
    for (int64_t j = 0; j < v_dim; ++j)
      StoreF32(v_out, i * v_dim + j, LoadF32(qkv, row + q_dim + k_dim + j));
  }
  });
}

// Fused GDN post-conv prep: GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta in
// one pass (mirror of fla fused_gdn_prefill_post_conv). Bit-for-bit equal to
// composing those four ops (same f32 math, same softplus threshold 20).
void GdnPostConvKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                       Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                       const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                       const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const int64_t row = i * conv_dim;
    // q/k: split then l2norm over Dk, per head (plain SUM of squares, §4).
    for (int64_t h = 0; h < hk; ++h) {
      float qss = 0.0f, kss = 0.0f;
      for (int64_t j = 0; j < dk; ++j) {
        const float qv = LoadF32(conv, row + h * dk + j);
        const float kv = LoadF32(conv, row + key_dim + h * dk + j);
        qss += qv * qv;
        kss += kv * kv;
      }
      const float qinv = 1.0f / std::sqrt(qss + args.eps);
      const float kinv = 1.0f / std::sqrt(kss + args.eps);
      for (int64_t j = 0; j < dk; ++j) {
        const int64_t o = (i * hk + h) * dk + j;
        StoreF32(q_out, o, LoadF32(conv, row + h * dk + j) * qinv);
        StoreF32(k_out, o, LoadF32(conv, row + key_dim + h * dk + j) * kinv);
      }
    }
    // v: plain copy.
    for (int64_t j = 0; j < value_dim; ++j)
      StoreF32(v_out, i * value_dim + j, LoadF32(conv, row + 2 * key_dim + j));
    // g/beta from a/b + A_log/dt_bias (§6).
    for (int64_t h = 0; h < hv; ++h) {
      const int64_t idx = i * hv + h;
      const int64_t aidx = i * araw.stride[0] + h;
      const int64_t bidx = i * braw.stride[0] + h;
      const float x = LoadF32(araw, aidx) + LoadF32(dt_bias, h);
      const float sp = x > 20.0f ? x : std::log1p(std::exp(x));  // softplus
      StoreF32(g_out, idx, -std::exp(LoadF32(a_log, h)) * sp);
      StoreF32(beta_out, idx, Sigmoid(LoadF32(braw, bidx)));
    }
  }
  });
}

// out[t,c] = F32ToBF16(sigmoid(gl[t]) * sd[t*H+c]); shared-expert sigmoid gate.
void SharedExpertGateKernel(Queue&, Tensor& out, const Tensor& sd, const Tensor& gl) {
  const int64_t t = out.shape[0], h = out.shape[1];
  ForRows(t, [&](int64_t r0, int64_t r1) {
  for (int64_t i = r0; i < r1; ++i) {
    const float g = Sigmoid(LoadF32(gl, i));
    for (int64_t c = 0; c < h; ++c) StoreF32(out, i * h + c, g * LoadF32(sd, i * h + c));
  }
  });
}

// --- Fused declarative recipe (TDR): the Tier-1 single-pass INTERPRETER over the
// canonical (out, x, weight, residual) 4-operand shape. The Tier-0 composite is
// device-agnostic and lives in ops.cpp (it dispatches each opcode to the
// standalone vt:: op). This kernel is reached ONLY for Tier-1-able recipes (all
// steps in {kAdd,kMul,kSilu,kSigmoid,kRmsNorm}) when VT_FUSED_TIER=1; the general
// FusedChain wrapper resolves the canonical operand order [0=x,1=weight,
// 2=residual,3=out] and forwards here. Bit-identical to vt::RmsNorm(residual) for
// kFusedAddRmsNorm — same f32 variance accumulation, gemma (1+w), residual-add
// rounding.
//
// Bit-identity across SEPARATELY COMPILED loops (here vs RmsNormKernel) holds
// because the build pins -ffp-contract=off (top-level CMakeLists): under the
// GCC/Clang default (fast), `sumsq += v*v` may be fma-contracted in one loop
// and not the other (observed: RmsNormKernel's residual path contracted, this
// interpreter not, on FMA-capable ISAs), skewing inv by 1 ulp on ~half the rows.

// Read one operand element by its CANONICAL index. Index 1 (weight) is per-column
// ([H]); the [T,H] operands index row*h + j.
float FusedLoad(uint8_t idx, int64_t row, int64_t j, int64_t h, const Tensor& x,
                const Tensor& weight, const Tensor* residual, const Tensor& out) {
  switch (idx) {
    case 0: return LoadF32(x, row * h + j);
    case 1: return LoadF32(weight, j);
    case 2: return LoadF32(*residual, row * h + j);
    case 3: return LoadF32(out, row * h + j);
    default: VT_CHECK(false, "fused_chain interp: operand index out of canonical range"); return 0.0f;
  }
}

// Store one element into a WRITABLE operand (residual=2 / out=3 only), rounding to
// that operand's dtype (bf16 residual mirrors vLLM's model-dtype residual).
void FusedStore(uint8_t idx, int64_t row, int64_t j, int64_t h, float v, Tensor* residual,
                Tensor& out) {
  switch (idx) {
    case 2: StoreF32(*residual, row * h + j, v); break;
    case 3: StoreF32(out, row * h + j, v); break;
    default: VT_CHECK(false, "fused_chain interp: step writes a read-only operand");
  }
}

// silu(x) = x * sigmoid(x); sigmoid in f32 (matches the standalone kernels' math).
inline float FSigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Tier 1 — scalar interpreter: walk the recipe over EACH ROW in one pass.
void FusedChainInterpKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                            Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  ForRows(t, [&](int64_t rr0, int64_t rr1) {
  for (int64_t row = rr0; row < rr1; ++row) {
    for (int s = 0; s < r.n; ++s) {
      const FStep& st = r.steps[s];
      switch (st.op) {
        case FOp::kAdd:
        case FOp::kMul:
          for (int64_t j = 0; j < h; ++j) {
            const float a = FusedLoad(st.in[0], row, j, h, x, weight, residual, out);
            const float b = FusedLoad(st.in[1], row, j, h, x, weight, residual, out);
            FusedStore(st.out, row, j, h, st.op == FOp::kAdd ? a + b : a * b, residual, out);
          }
          break;
        case FOp::kSilu:
        case FOp::kSigmoid:
          for (int64_t j = 0; j < h; ++j) {
            const float a = FusedLoad(st.in[0], row, j, h, x, weight, residual, out);
            FusedStore(st.out, row, j, h, st.op == FOp::kSilu ? a * FSigmoid(a) : FSigmoid(a),
                       residual, out);
          }
          break;
        case FOp::kRmsNorm: {
          VT_CHECK(st.reduce == FReduce::kMeanSquare, "fused_chain: rmsnorm needs kMeanSquare");
          float sumsq = 0.0f;
          for (int64_t j = 0; j < h; ++j) {
            const float v = FusedLoad(st.in[0], row, j, h, x, weight, residual, out);
            sumsq += v * v;  // f32 variance accumulation
          }
          const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + eps);
          for (int64_t j = 0; j < h; ++j) {
            const float v = FusedLoad(st.in[0], row, j, h, x, weight, residual, out);
            float wj = FusedLoad(st.in[1], row, j, h, x, weight, residual, out);
            if (st.gemma) wj += 1.0f;
            FusedStore(st.out, row, j, h, v * inv * wj, residual, out);
          }
          break;
        }
        default:
          VT_CHECK(false, "fused_chain interp: non-Tier-1 opcode reached the interpreter");
      }
    }
  }
  });
}

// Registered kernel: the Tier-1 interpreter (the general wrapper only dispatches
// here for Tier-1-able recipes; Tier-0 composite is device-agnostic in ops.cpp).
void FusedChainKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  FusedChainInterpKernel(q, out, x, weight, residual, r, eps);
}

struct Registrar {
  Registrar() {
    // static_cast against the ops.h aliases ties kernel signatures to the
    // registration contract at compile time.
    RegisterOp(OpId::kMatmul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kBatchedMatmul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<BatchedMatmulFn>(&BatchedMatmulKernel)));
    RegisterOp(
        OpId::kConcatMlaNopeRope, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<ConcatMlaNopeRopeFn>(&ConcatMlaNopeRopeKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kRmsNormQuantFp8, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RmsNormQuantFp8Fn>(&RmsNormQuantFp8Kernel)));
    RegisterOp(OpId::kQuantFp8Static, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<QuantFp8StaticFn>(&QuantFp8StaticKernel)));
    RegisterOp(OpId::kQuantFp8Group, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<QuantFp8GroupFn>(&QuantFp8GroupKernel)));
    RegisterOp(OpId::kMatmulFp8Cutlass, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFp8CutlassFn>(&MatmulFp8CutlassKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kGeluAndMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&GeluAndMulKernel)));
    RegisterOp(OpId::kMulScalar, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MulScalarFn>(&MulScalarKernel)));
    RegisterOp(OpId::kSoftCap, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SoftCapFn>(&SoftCapKernel)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernel)));
    RegisterOp(OpId::kMoeRelu2, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeRelu2Fn>(&MoeRelu2Kernel)));
    RegisterOp(OpId::kScaledFp4Quant, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ScaledFp4QuantFn>(&ScaledFp4QuantKernel)));
    RegisterOp(OpId::kSiluMulFp4Quant, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SiluMulFp4QuantFn>(&SiluMulFp4QuantKernel)));
    RegisterOp(OpId::kSigmoidGateFp4Quant, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<SigmoidGateFp4QuantFn>(&SigmoidGateFp4QuantKernel)));
    RegisterOp(
        OpId::kSiluAndMulFp4Quant, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<SiluAndMulFp4QuantFn>(
            &SiluAndMulFp4QuantKernel)));
    RegisterOp(OpId::kMatmulNvfp4Fp4, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fp4Fn>(&MatmulNvfp4Fp4Kernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(
        OpId::kRopeFromCache, DeviceType::kCPU,
        reinterpret_cast<void*>(
            static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(
        OpId::kFusedNormRope, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<FusedNormRopeFn>(&FusedNormRopeKernel)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    RegisterOp(OpId::kCausalConv1dSpecUpdate, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<CausalConv1dSpecUpdateFn>(
                   &CausalConv1dSpecUpdateKernel)));
    RegisterOp(OpId::kL2Norm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(
        OpId::kRmsNormGatedQuantFp8, DeviceType::kCPU,
        reinterpret_cast<void*>(
            static_cast<RmsNormGatedQuantFp8Fn>(&RmsNormGatedQuantFp8Kernel)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
    RegisterOp(
        OpId::kGdnSpecDecode, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<GdnSpecDecodeFn>(&GdnSpecDecodeKernel)));
    RegisterOp(
        OpId::kGdnPackedDecode, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<GdnPackedDecodeFn>(
            &GdnPackedDecodeKernel)));
    RegisterOp(
        OpId::kKdaGatedDeltaRule, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<KdaGatedDeltaRuleFn>(&KdaGatedDeltaRuleKernel)));
    RegisterOp(
        OpId::kKdaChunkPrefill, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<KdaChunkPrefillFn>(&KdaChunkPrefillKernel)));
    RegisterOp(
        OpId::kMamba2ChunkScan, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<Mamba2ChunkScanFn>(&Mamba2ChunkScanKernel)));
    RegisterOp(
        OpId::kMamba2StateUpdate, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<Mamba2StateUpdateFn>(&Mamba2StateUpdateKernel)));
    RegisterOp(
        OpId::kRmsNormGatedGroup, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<RmsNormGatedGroupFn>(&RmsNormGatedGroupKernel)));
    RegisterOp(
        OpId::kGdnStateGather, DeviceType::kCPU,
        reinterpret_cast<void*>(
            static_cast<GdnStateGatherFn>(&GdnStateGatherKernel)));
    RegisterOp(
        OpId::kGdnStateScatter, DeviceType::kCPU,
        reinterpret_cast<void*>(
            static_cast<GdnStateScatterFn>(&GdnStateScatterKernel)));
    RegisterOp(OpId::kIndexSelect, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<IndexSelectFn>(&IndexSelectKernel)));
    RegisterOp(OpId::kIndexCopy, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<IndexCopyFn>(&IndexCopyKernel)));
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernel)));
    RegisterOp(OpId::kMoeCombine, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MoeCombineFn>(&MoeCombineKernel)));
    RegisterOp(OpId::kAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernel)));
    RegisterOp(OpId::kAttentionCross, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionCrossFn>(&AttentionCrossKernel)));
    // Dense-fast shares the CPU reference (the warp variant is a CUDA-only
    // optimization); byte-identical to kAttention on CPU.
    RegisterOp(OpId::kAttentionDenseFast, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernel)));
    // Flash-tiled dense attention is a CUDA shared-memory optimization; byte-identical
    // to kAttention on CPU.
    RegisterOp(OpId::kAttentionDenseFlash, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernel)));
    // The FA-2 dense variant is a CUDA tensor-core optimization; byte-identical to
    // kAttention on CPU.
    RegisterOp(OpId::kAttentionDenseFa2, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernel)));
    RegisterOp(OpId::kDFlashBlockAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<DFlashBlockAttentionFn>(&DFlashBlockAttentionKernel)));
    RegisterOp(OpId::kDFlashPagedBlockAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<DFlashPagedBlockAttentionFn>(&DFlashPagedBlockAttentionKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16Kernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32Kernel)));
    RegisterOp(OpId::kMulColVecF32, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MulColVecF32Fn>(&MulColVecF32Kernel)));
    RegisterOp(OpId::kAttnGateSplit, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttnGateSplitFn>(&AttnGateSplitKernel)));
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kGdnGBeta, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnGBetaFn>(&GdnGBetaKernel)));
    RegisterOp(OpId::kGdnConvSplit, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnConvSplitFn>(&GdnConvSplitKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(
        OpId::kAttnQkNormRopeGate, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernel)));
    RegisterOp(OpId::kSharedExpertGate, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SharedExpertGateFn>(&SharedExpertGateKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
