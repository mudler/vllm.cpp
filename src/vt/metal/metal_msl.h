// Metal backend — the embedded Metal Shading Language source (BACKEND-METAL-MLX,
// W0 skeleton). vllm.cpp original: vLLM has no Metal kernels to mirror, so the
// PER-ELEMENT MATH of every kernel here is ported 1:1 from our own CPU reference
// kernels (which are themselves the vLLM-parity goldens), and only the DISPATCH
// SHAPE is ported from llama.cpp's Metal backend.
//
// Math ported FROM (line-for-line correspondence noted per kernel below):
//   src/vt/cpu/cpu_ops.cpp:225-250  RmsNormKernel        -> vt_rms_norm
//   src/vt/cpu/cpu_ops.cpp:252-264  SiluAndMulKernel     -> vt_silu_and_mul
//   src/vt/cpu/cpu_ops.cpp:1436-1451 CastBf16/CastF32Kernel -> vt_cast
//   src/vt/cpu/cpu_ops.cpp:1649-1695 FusedChainInterpKernel -> vt_fused_chain
//   src/vt/cpu/cpu_layernorm.cpp:49-73 LayerNormKernel   -> vt_layer_norm
//   src/vt/cpu/cpu_layernorm.cpp:75-85 ReluKernel        -> vt_relu
//   src/vt/cpu/cpu_layernorm.cpp:87-99 AddKernel         -> vt_add
//   src/vt/cpu/cpu_ops.cpp MatmulKernel/MatmulBTKernel  -> vt_matmul
//   src/vt/dtype.cpp:224-233        BF16<->F32 codec     -> vt_bf16_to_f32 /
//                                                           vt_f32_to_bf16
//
// Dispatch shape ported FROM llama.cpp `ggml/src/ggml-metal/` @ 237ad9b96:
//   * flat elementwise ops dispatch one thread per element
//     (`ggml_metal_op_bin` / `kernel_add`);
//   * row-reducing ops dispatch ONE THREADGROUP PER ROW with a threadgroup-memory
//     tree reduction (`ggml_metal_op_norm` / `kernel_rms_norm`).
//
// NUMERICS. The reduction ORDER necessarily differs from the CPU reference: the
// CPU tier's reproducibility comes from a fixed SEQUENTIAL accumulation
// (src/vt/cpu/cpu_quant_dot.cpp:22-28, deliberate) and no GPU tree reduction
// preserves it. Per the spike § Gates the bar is therefore NMSE <= 5e-4 for
// reducing ops, NOT bit-exactness. Non-reducing, non-arithmetic paths (Copy /
// Memset / a same-dtype cast) ARE bit-exact and are gated as such.
// The library is compiled with MTLMathModeSafe (see metal_context.mm) so `exp`,
// `sqrt` and the arithmetic below keep IEEE semantics rather than the Metal
// default fast-math relaxations.
#ifndef VT_METAL_METAL_MSL_H_
#define VT_METAL_METAL_MSL_H_

namespace vt::metal {

// clang-format off
inline constexpr const char* kMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Storage dtype codes. These mirror the three FLOAT entries of vt::DType and are
// translated host-side by DtypeCode() in metal_ops.mm — the shader never sees
// vt::DType itself.
#define VT_DT_F32  0u
#define VT_DT_F16  1u
#define VT_DT_BF16 2u

// --- bf16 codec, bit-identical to src/vt/dtype.cpp:224-233 -------------------
// BF16ToF32 is a pure 16-bit left shift; F32ToBF16 is round-to-nearest-EVEN with
// the NaN-quieting special case. Reproduced exactly so a bf16 round-trip through
// a Metal kernel rounds the same way the CPU reference does.
inline float vt_bf16_to_f32(ushort b) {
  return as_type<float>(uint(b) << 16);
}
inline ushort vt_f32_to_bf16(float f) {
  uint u = as_type<uint>(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
    return ushort((u >> 16) | 0x0040u);   // nan: keep quiet, truncate
  }
  uint rounding = 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
  return ushort((u + rounding) >> 16);
}

// --- dtype-erased element access, mirroring cpu_ops.cpp:27-43 LoadF32/StoreF32.
// Reduced-width outputs round ON STORE, exactly once, like the CPU reference.
inline float vt_load(device const uchar* base, uint dt, ulong idx) {
  if (dt == VT_DT_F32)  { return ((device const float*)base)[idx]; }
  if (dt == VT_DT_F16)  { return float(((device const half*)base)[idx]); }
  return vt_bf16_to_f32(((device const ushort*)base)[idx]);
}
inline void vt_store(device uchar* base, uint dt, ulong idx, float v) {
  if (dt == VT_DT_F32)  { ((device float*)base)[idx] = v; return; }
  if (dt == VT_DT_F16)  { ((device half*)base)[idx] = half(v); return; }
  ((device ushort*)base)[idx] = vt_f32_to_bf16(v);
}

// silu/sigmoid in f32, matching cpu_ops.cpp:1646 FSigmoid and the `gate / (1 +
// exp(-gate))` spelling of SiluAndMulKernel.
inline float vt_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Threadgroup tree reduction over `tg` lanes. `tg` is always a power of two
// (host-side ChooseThreadgroupSize guarantees it), so the halving loop is exact.
//
// The LEADING barrier makes this safe to call MORE THAN ONCE per kernel, which
// vt_layer_norm does (mean, then variance) and vt_fused_chain can do (one per
// kRmsNorm step). Without it there is a real race: every thread reads smem[0] at
// the end of one call, and a thread that races ahead into the next call would
// overwrite smem[0] while a slower thread is still reading the previous result.
// It is NOT redundant with the trailing barrier inside the loop, which only
// orders the reduction itself.
inline float vt_tg_sum(threadgroup float* smem, uint tid, uint tg, float v) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  smem[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg / 2u; s > 0u; s >>= 1u) {
    if (tid < s) { smem[tid] += smem[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return smem[0];
}

// Same tree shape, `max` instead of `+`. Used by the paged-attention softmax to
// find the running row maximum. Same leading-barrier rule, same reason: it is
// called once per key CHUNK inside a loop, so consecutive calls must not race.
inline float vt_tg_max(threadgroup float* smem, uint tid, uint tg, float v) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  smem[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg / 2u; s > 0u; s >>= 1u) {
    if (tid < s) { smem[tid] = max(smem[tid], smem[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return smem[0];
}

// ===========================================================================
// Flat elementwise kernels — one thread per output element.
// ===========================================================================

struct VtElemParams {
  uint  n;        // total elements
  uint  d;        // last-dim extent (for the rank-1 bias broadcast)
  uint  a_dt;
  uint  b_dt;
  uint  out_dt;
  uint  bcast;    // 1 => b is rank-1 [d] and is indexed by (i % d)
};

// cpu_layernorm.cpp:87-99 AddKernel, including the nn.Linear bias row-broadcast.
kernel void vt_add(device const uchar* a   [[buffer(0)]],
                   device const uchar* b   [[buffer(1)]],
                   device uchar*       out [[buffer(2)]],
                   constant VtElemParams& p [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) { return; }
  ulong bi = p.bcast != 0u ? ulong(gid % p.d) : ulong(gid);
  vt_store(out, p.out_dt, ulong(gid), vt_load(a, p.a_dt, ulong(gid)) + vt_load(b, p.b_dt, bi));
}

// cpu_layernorm.cpp:75-85 ReluKernel.
kernel void vt_relu(device const uchar* x   [[buffer(0)]],
                    device uchar*       out [[buffer(1)]],
                    constant VtElemParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) { return; }
  float v = vt_load(x, p.a_dt, ulong(gid));
  vt_store(out, p.out_dt, ulong(gid), v > 0.0f ? v : 0.0f);
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — both are the identity
// through LoadF32/StoreF32, i.e. a dtype-converting copy. When src and dst dtype
// are equal this is a bit-exact copy (gated as such).
kernel void vt_cast(device const uchar* x   [[buffer(0)]],
                    device uchar*       out [[buffer(1)]],
                    constant VtElemParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) { return; }
  vt_store(out, p.out_dt, ulong(gid), vt_load(x, p.a_dt, ulong(gid)));
}

struct VtSiluMulParams {
  uint t;
  uint d;        // HALF the input inner dim: x is [t, 2*d], out is [t, d]
  uint x_dt;
  uint out_dt;
};

// cpu_ops.cpp:252-264 SiluAndMulKernel: out[i,j] = silu(x[i,j]) * x[i,d+j].
kernel void vt_silu_and_mul(device const uchar* x   [[buffer(0)]],
                            device uchar*       out [[buffer(1)]],
                            constant VtSiluMulParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= p.t * p.d) { return; }
  uint i = gid / p.d;
  uint j = gid % p.d;
  ulong base = ulong(i) * ulong(2u * p.d);
  float gate = vt_load(x, p.x_dt, base + ulong(j));
  float up   = vt_load(x, p.x_dt, base + ulong(p.d) + ulong(j));
  vt_store(out, p.out_dt, ulong(i) * ulong(p.d) + ulong(j), gate * vt_sigmoid(gate) * up);
}

// ===========================================================================
// Row-reducing kernels — one THREADGROUP per row (llama.cpp kernel_rms_norm
// shape). VT_TG_MAX matches the M4's maxTotalThreadsPerThreadgroup (1024); the
// threadgroup scratch is 4 KiB, well inside the 32 KiB limit.
// ===========================================================================

#define VT_TG_MAX 1024

struct VtRmsParams {
  uint  t;
  uint  h;
  uint  x_dt;
  uint  w_dt;
  uint  out_dt;
  uint  res_dt;
  uint  has_res;
  uint  gemma;
  uint  tg;
  float eps;
};

// cpu_ops.cpp:225-250 RmsNormKernel. The residual idiom is reproduced EXACTLY,
// including the deliberate store-then-RE-READ that makes a bf16 residual stream
// bit-faithful (cpu_ops.cpp:235-237): add in f32, round into the residual's own
// dtype, then read the ROUNDED value back for both the variance and the scale.
kernel void vt_rms_norm(device const uchar* x   [[buffer(0)]],
                        device const uchar* w   [[buffer(1)]],
                        device uchar*       out [[buffer(2)]],
                        device uchar*       res [[buffer(3)]],
                        constant VtRmsParams& p [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  ulong rbase = ulong(row) * ulong(p.h);

  float partial = 0.0f;
  for (uint j = tid; j < p.h; j += p.tg) {
    float v = vt_load(x, p.x_dt, rbase + ulong(j));
    if (p.has_res != 0u) {
      v += vt_load(res, p.res_dt, rbase + ulong(j));
      vt_store(res, p.res_dt, rbase + ulong(j), v);
      v = vt_load(res, p.res_dt, rbase + ulong(j));
    }
    partial += v * v;
  }
  if (p.has_res != 0u) { threadgroup_barrier(mem_flags::mem_device); }
  float sumsq = vt_tg_sum(smem, tid, p.tg, partial);
  float inv = 1.0f / sqrt(sumsq / float(p.h) + p.eps);

  for (uint j = tid; j < p.h; j += p.tg) {
    float v = p.has_res != 0u ? vt_load(res, p.res_dt, rbase + ulong(j))
                              : vt_load(x, p.x_dt, rbase + ulong(j));
    float wj = vt_load(w, p.w_dt, ulong(j));
    if (p.gemma != 0u) { wj += 1.0f; }
    vt_store(out, p.out_dt, rbase + ulong(j), v * inv * wj);
  }
}

struct VtLayerNormParams {
  uint  rows;
  uint  d;
  uint  x_dt;
  uint  w_dt;
  uint  b_dt;
  uint  out_dt;
  uint  has_w;
  uint  has_b;
  uint  tg;
  float eps;
};

// cpu_layernorm.cpp:49-73 LayerNormKernel — the two-pass numerically stable form
// (mean, then squared deviations ABOUT that mean), BIASED (1/N) variance, f32
// accumulation, one rounding on store.
kernel void vt_layer_norm(device const uchar* x   [[buffer(0)]],
                          device const uchar* w   [[buffer(1)]],
                          device const uchar* b   [[buffer(2)]],
                          device uchar*       out [[buffer(3)]],
                          constant VtLayerNormParams& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  ulong base = ulong(row) * ulong(p.d);

  float psum = 0.0f;
  for (uint i = tid; i < p.d; i += p.tg) { psum += vt_load(x, p.x_dt, base + ulong(i)); }
  float mean = vt_tg_sum(smem, tid, p.tg, psum) / float(p.d);

  float psq = 0.0f;
  for (uint i = tid; i < p.d; i += p.tg) {
    float dv = vt_load(x, p.x_dt, base + ulong(i)) - mean;
    psq += dv * dv;
  }
  float rstd = 1.0f / sqrt(vt_tg_sum(smem, tid, p.tg, psq) / float(p.d) + p.eps);

  for (uint i = tid; i < p.d; i += p.tg) {
    float v = (vt_load(x, p.x_dt, base + ulong(i)) - mean) * rstd;
    if (p.has_w != 0u) { v *= vt_load(w, p.w_dt, ulong(i)); }
    if (p.has_b != 0u) { v += vt_load(b, p.b_dt, ulong(i)); }
    vt_store(out, p.out_dt, base + ulong(i), v);
  }
}

// ===========================================================================
// The Tier-1 FusedChain interpreter — the ONE registration that inherits the
// whole portable fusion catalog for this backend (include/vt/fused_recipe.h).
// Structural mirror of cpu_ops.cpp:1649-1695 FusedChainInterpKernel, including
// its canonical operand indexing (cpu_ops.cpp:1621-1643 FusedLoad/FusedStore):
//   0 = x [t,h]   1 = weight [h]   2 = residual [t,h]   3 = out [t,h]
// with 2 and 3 the only WRITABLE slots. Opcodes mirror vt::FOp; only the
// Tier-1-able subset {kAdd,kMul,kSilu,kSigmoid,kRmsNorm} can reach here (the
// device-agnostic Tier-0 composite in src/vt/ops.cpp handles everything else, so
// this backend inherits those recipes too without a second kernel).
// ===========================================================================

#define VT_FOP_ADD      0u
#define VT_FOP_MUL      1u
#define VT_FOP_SILU     2u
#define VT_FOP_SIGMOID  3u
#define VT_FOP_RMSNORM  4u

struct VtFStep {
  uint op;
  uint out;
  uint in0;
  uint in1;
  uint gemma;
  uint pad;
};

// ===========================================================================
// Dense GEMM — the NATIVE MSL provider for kMatmul and kMatmulBT.
//
// This is the DEFAULT on Metal. It exists so the MLX provider
// (src/vt/metal/metal_mlx_provider.mm) is a CONFIGURATION rather than the only
// way to get a GEMM: the provider seam's whole premise is that two providers of
// one op coexist and can be A/B'd against each other, which requires ours to be
// there. It also gives the correctness gate its middle column
// (MLX vs native MSL vs the CPU oracle).
//
// Math ported FROM src/vt/cpu/cpu_ops.cpp MatmulKernel / MatmulBTKernel: f32
// accumulation over K regardless of storage dtype, one rounding on store.
// Dispatch shape is the classic threadgroup-tiled GEMM (llama.cpp
// ggml-metal `kernel_mul_mm` uses the same 2-D tile-per-threadgroup structure
// over simdgroup_matrix; we stay on plain threadgroup memory because the W0
// build compiles MSL at RUNTIME with no offline `metal`, and a portable tile
// loop is what the CPU-reference math transcribes to directly).
//
// ORIENTATION. One kernel serves both ops via `bt`:
//   bt == 0 (kMatmul):   B is [K,N] row-major, element (kk,col) at kk*N + col
//   bt == 1 (kMatmulBT): B is [N,K] row-major, element (col,kk) at col*K + kk
// `lda` is the ACTIVATION's row stride in elements — kMatmulBT explicitly admits
// a row-strided activation (src/vt/ops.cpp MatmulBT: `a.stride[0] >= a.shape[1]`),
// so it cannot be assumed equal to K.
#define VT_GEMM_TILE 16u

struct VtGemmParams {
  uint m;
  uint n;
  uint k;
  uint lda;     // activation row stride, in ELEMENTS
  uint a_dt;
  uint b_dt;
  uint out_dt;
  uint bt;      // 0 => b is [K,N]; 1 => b is [N,K] (the torch Linear orientation)
};

kernel void vt_matmul(device const uchar* a   [[buffer(0)]],
                      device const uchar* b   [[buffer(1)]],
                      device uchar*       out [[buffer(2)]],
                      constant VtGemmParams& p [[buffer(3)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint2 lid  [[thread_position_in_threadgroup]]) {
  threadgroup float as[VT_GEMM_TILE][VT_GEMM_TILE];
  threadgroup float bs[VT_GEMM_TILE][VT_GEMM_TILE];

  const uint row = tgid.y * VT_GEMM_TILE + lid.y;
  const uint col = tgid.x * VT_GEMM_TILE + lid.x;
  const uint ntiles = (p.k + VT_GEMM_TILE - 1u) / VT_GEMM_TILE;

  float acc = 0.0f;
  for (uint t = 0u; t < ntiles; ++t) {
    const uint ka = t * VT_GEMM_TILE + lid.x;
    as[lid.y][lid.x] =
        (row < p.m && ka < p.k) ? vt_load(a, p.a_dt, ulong(row) * ulong(p.lda) + ulong(ka)) : 0.0f;

    const uint kb = t * VT_GEMM_TILE + lid.y;
    float bv = 0.0f;
    if (col < p.n && kb < p.k) {
      bv = (p.bt != 0u) ? vt_load(b, p.b_dt, ulong(col) * ulong(p.k) + ulong(kb))
                        : vt_load(b, p.b_dt, ulong(kb) * ulong(p.n) + ulong(col));
    }
    bs[lid.y][lid.x] = bv;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < VT_GEMM_TILE; ++i) { acc += as[lid.y][i] * bs[i][lid.x]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < p.m && col < p.n) {
    vt_store(out, p.out_dt, ulong(row) * ulong(p.n) + ulong(col), acc);
  }
}

struct VtFcParams {
  uint  t;
  uint  h;
  uint  nsteps;
  uint  x_dt;
  uint  w_dt;
  uint  res_dt;
  uint  out_dt;
  uint  tg;
  float eps;
};

inline float vt_fc_load(uint idx, ulong rbase, uint j,
                        device const uchar* x, uint x_dt,
                        device const uchar* w, uint w_dt,
                        device const uchar* res, uint res_dt,
                        device const uchar* out, uint out_dt) {
  if (idx == 0u) { return vt_load(x, x_dt, rbase + ulong(j)); }
  if (idx == 1u) { return vt_load(w, w_dt, ulong(j)); }
  if (idx == 2u) { return vt_load(res, res_dt, rbase + ulong(j)); }
  return vt_load(out, out_dt, rbase + ulong(j));
}

inline void vt_fc_store(uint idx, ulong rbase, uint j, float v,
                        device uchar* res, uint res_dt,
                        device uchar* out, uint out_dt) {
  if (idx == 2u) { vt_store(res, res_dt, rbase + ulong(j), v); return; }
  vt_store(out, out_dt, rbase + ulong(j), v);
}

kernel void vt_fused_chain(device const uchar*  x     [[buffer(0)]],
                           device const uchar*  w     [[buffer(1)]],
                           device uchar*        res   [[buffer(2)]],
                           device uchar*        out   [[buffer(3)]],
                           device const VtFStep* steps [[buffer(4)]],
                           constant VtFcParams& p     [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  ulong rbase = ulong(row) * ulong(p.h);

  for (uint s = 0u; s < p.nsteps; ++s) {
    VtFStep st = steps[s];
    if (st.op == VT_FOP_ADD || st.op == VT_FOP_MUL) {
      for (uint j = tid; j < p.h; j += p.tg) {
        float a = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        float b = vt_fc_load(st.in1, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        vt_fc_store(st.out, rbase, j, st.op == VT_FOP_ADD ? a + b : a * b,
                    res, p.res_dt, out, p.out_dt);
      }
    } else if (st.op == VT_FOP_SILU || st.op == VT_FOP_SIGMOID) {
      for (uint j = tid; j < p.h; j += p.tg) {
        float a = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        float sg = vt_sigmoid(a);
        vt_fc_store(st.out, rbase, j, st.op == VT_FOP_SILU ? a * sg : sg,
                    res, p.res_dt, out, p.out_dt);
      }
    } else {  // VT_FOP_RMSNORM (the host validated reduce == kMeanSquare)
      float partial = 0.0f;
      for (uint j = tid; j < p.h; j += p.tg) {
        float v = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        partial += v * v;   // f32 variance accumulation
      }
      float inv = 1.0f / sqrt(vt_tg_sum(smem, tid, p.tg, partial) / float(p.h) + p.eps);
      for (uint j = tid; j < p.h; j += p.tg) {
        float v = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        float wj = vt_fc_load(st.in1, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        if (st.gemma != 0u) { wj += 1.0f; }
        vt_fc_store(st.out, rbase, j, v * inv * wj, res, p.res_dt, out, p.out_dt);
      }
    }
    // A later step may read what this step wrote (the interpreter is a chain),
    // and a step's own two phases race across lanes, so every step boundary is a
    // device-memory barrier within the row's threadgroup.
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

// ===========================================================================
// M3a — the five kernels OPT-125m (`OPTForCausalLM`) needs beyond the W0 set.
// Math ported 1:1 from our own CPU reference kernels, which are the vLLM-parity
// goldens:
//   src/vt/cpu/cpu_ops.cpp:531-543        EmbeddingKernel       -> vt_embedding
//   src/vt/cpu/cpu_ops.cpp:1529-1543      QkvSplitKernel        -> vt_qkv_split
//   src/vt/cpu/cpu_cache.cpp:33-72        ReshapeAndCacheKernel -> vt_reshape_and_cache
//   src/vt/cpu/cpu_paged_attn.cpp:51-131  PagedAttentionKernel  -> vt_paged_attention
//   src/vt/cpu/cpu_sample.cpp:40-57       GreedyArgmaxKernel    -> vt_greedy_argmax
// ===========================================================================

struct VtEmbedParams {
  uint rows;     // T
  uint h;        // embedding width
  uint vocab;    // table rows, for the bounds check
  uint id_i64;   // 1 => ids are i64, 0 => i32
  uint tab_dt;
  uint out_dt;
};

// cpu_ops.cpp:531-543 EmbeddingKernel — a pure row gather through LoadF32/
// StoreF32. One thread per output ELEMENT. An out-of-range id cannot throw from
// a shader, so the row is written as zeros and the host-side wrapper's own
// validation remains the place the error surfaces (the CPU reference VT_CHECKs;
// vt::Embedding in src/vt/ops.cpp validates shapes, and OPT's ids come from the
// tokenizer, so an out-of-range id is a corrupted-checkpoint case, not a
// reachable input).
kernel void vt_embedding(device const uchar* table [[buffer(0)]],
                         device const uchar* ids   [[buffer(1)]],
                         device uchar*       out   [[buffer(2)]],
                         constant VtEmbedParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows * p.h) { return; }
  const uint i = gid / p.h;
  const uint j = gid % p.h;
  long id = p.id_i64 != 0u ? ((device const long*)ids)[i] : long(((device const int*)ids)[i]);
  if (id < 0 || id >= long(p.vocab)) {
    vt_store(out, p.out_dt, ulong(gid), 0.0f);
    return;
  }
  vt_store(out, p.out_dt, ulong(gid), vt_load(table, p.tab_dt, ulong(id) * ulong(p.h) + ulong(j)));
}

struct VtQkvSplitParams {
  uint t;
  uint q_dim;
  uint k_dim;
  uint v_dim;
  uint in_dt;
  uint q_dt;
  uint k_dt;
  uint v_dt;
};

// cpu_ops.cpp:1529-1543 QkvSplitKernel — the merged [T, q+k+v] row cut into three
// DENSE per-shard buffers. One thread per merged element; the column decides
// which output it belongs to.
kernel void vt_qkv_split(device const uchar* qkv [[buffer(0)]],
                         device uchar*       q   [[buffer(1)]],
                         device uchar*       k   [[buffer(2)]],
                         device uchar*       v   [[buffer(3)]],
                         constant VtQkvSplitParams& p [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint total = p.q_dim + p.k_dim + p.v_dim;
  if (gid >= p.t * total) { return; }
  const uint i = gid / total;
  const uint c = gid % total;
  const float val = vt_load(qkv, p.in_dt, ulong(gid));
  if (c < p.q_dim) {
    vt_store(q, p.q_dt, ulong(i) * ulong(p.q_dim) + ulong(c), val);
  } else if (c < p.q_dim + p.k_dim) {
    vt_store(k, p.k_dt, ulong(i) * ulong(p.k_dim) + ulong(c - p.q_dim), val);
  } else {
    vt_store(v, p.v_dt, ulong(i) * ulong(p.v_dim) + ulong(c - p.q_dim - p.k_dim), val);
  }
}

// 8-byte members FIRST, then an even count of 4-byte members: neither MSL nor the
// host struct in metal_ops.mm can insert interior padding, so the two layouts
// coincide by construction (metal_ops.mm static_asserts the offsets).
struct VtCacheParams {
  ulong k_blk_stride; // all strides in ELEMENTS, from the tensors themselves
  ulong k_pg_stride;
  ulong v_blk_stride;
  ulong v_pg_stride;
  ulong k_tok_stride;
  ulong v_tok_stride;
  uint  num_slots;
  uint  n_elems;      // num_kv_heads * head_size — one token's NHD page
  uint  block_size;
  uint  esz;          // element size in BYTES (2 = bf16/f16, 4 = f32)
};

// cpu_cache.cpp:33-72 ReshapeAndCacheKernel. Upstream (and our CPU reference) is
// a `memcpy` per token, so this copies RAW ELEMENTS rather than routing through
// LoadF32/StoreF32 — the cache write is BIT-EXACT on every backend, which is what
// the gate claims for pure copy/layout ops. `slot < 0` is the padded-token skip
// the upstream kernel documents.
kernel void vt_reshape_and_cache(device const uchar* k    [[buffer(0)]],
                                 device const uchar* v    [[buffer(1)]],
                                 device uchar*       kc   [[buffer(2)]],
                                 device uchar*       vc   [[buffer(3)]],
                                 device const long*  slots [[buffer(4)]],
                                 constant VtCacheParams& p [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num_slots * p.n_elems) { return; }
  const uint t = gid / p.n_elems;
  const uint e = gid % p.n_elems;
  const long slot = slots[t];
  if (slot < 0) { return; }
  const ulong block = ulong(slot) / ulong(p.block_size);
  const ulong off = ulong(slot) % ulong(p.block_size);
  const ulong kd = block * p.k_blk_stride + off * p.k_pg_stride + ulong(e);
  const ulong vd = block * p.v_blk_stride + off * p.v_pg_stride + ulong(e);
  const ulong ks = ulong(t) * p.k_tok_stride + ulong(e);
  const ulong vs = ulong(t) * p.v_tok_stride + ulong(e);
  if (p.esz == 4u) {
    ((device uint*)kc)[kd] = ((device const uint*)k)[ks];
    ((device uint*)vc)[vd] = ((device const uint*)v)[vs];
  } else {
    ((device ushort*)kc)[kd] = ((device const ushort*)k)[ks];
    ((device ushort*)vc)[vd] = ((device const ushort*)v)[vs];
  }
}

// Paged causal/full GQA attention over the NHD paged cache. Tile widths: the key
// CHUNK bounds the threadgroup score buffer, VT_PA_MAXD bounds the accumulator.
// Both are asserted host-side, so an unsupported head_size is a loud vt:: error
// rather than a silent wrong answer.
#define VT_PA_CHUNK 256u
#define VT_PA_MAXD  256u

// Same layout discipline as VtCacheParams: 8-byte members first, then 4-byte.
struct VtPagedAttnParams {
  ulong kc_blk; ulong kc_pg; ulong kc_hd;
  ulong vc_blk; ulong vc_pg; ulong vc_hd;
  uint  num_reqs;
  uint  hq;
  uint  d;
  uint  qpk;         // q-heads per kv-head (the GQA ratio)
  uint  block_size;
  uint  causal;
  uint  tg;
  int   window_left;
  int   window_right;
  int   bt_row;
  int   bt_col;
  uint  q_dt; uint kc_dt; uint vc_dt; uint out_dt;
  float scale;
};

// cpu_paged_attn.cpp:51-131 PagedAttentionKernel. The CPU reference is a THREE-
// PASS max-subtracted softmax that materializes the whole score row; on a GPU
// that row can be arbitrarily long, so this is the algebraically identical
// ONLINE (flash) form: keys are consumed in chunks of VT_PA_CHUNK with a running
// (max, denominator) pair and a rescaled accumulator. Same f32 accumulation, same
// single rounding on store. The reduction ORDER differs from the CPU reference by
// construction — per the spike § Gates the bar for reducing ops is NMSE <= 5e-4,
// not bit-exactness, and no bit-exactness is claimed here.
//
// One THREADGROUP per (query token, q-head). The owning request is found by
// scanning query_start_loc in-kernel rather than uploading a per-token request
// index: num_reqs is tiny, the scan is uniform across the threadgroup, and it
// keeps the op's device-side inputs exactly the ones the vt:: signature already
// passes.
kernel void vt_paged_attention(device const uchar* q     [[buffer(0)]],
                               device const uchar* kc    [[buffer(1)]],
                               device const uchar* vc    [[buffer(2)]],
                               device const int*   btab  [[buffer(3)]],
                               device const int*   slens [[buffer(4)]],
                               device const int*   qsl   [[buffer(5)]],
                               device uchar*       out   [[buffer(6)]],
                               constant VtPagedAttnParams& p [[buffer(7)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               // MSL requires every position attribute in one
                               // signature to have the SAME dimensionality, so
                               // the thread index is uint2 with an unused .y even
                               // though the threadgroup is 1-D.
                               uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float smem[VT_TG_MAX];
  threadgroup float scores[VT_PA_CHUNK];
  threadgroup float acc[VT_PA_MAXD];
  threadgroup float sh_m;
  threadgroup float sh_l;

  const uint h = tgid.x;
  const uint t = tgid.y;

  // Locate the request owning global query index `t`: qsl[r] <= t < qsl[r+1].
  // UNIFORM across the threadgroup, so the early-outs below never split a
  // barrier.
  int r = -1;
  for (uint i = 0u; i < p.num_reqs; ++i) {
    if (int(t) >= qsl[i] && int(t) < qsl[i + 1u]) { r = int(i); break; }
  }
  if (r < 0) { return; }
  const int q0 = qsl[r];
  const int query_len = qsl[r + 1] - q0;
  const int seqlen = slens[r];
  const int context = seqlen - query_len;   // past positions before this chunk
  const int pos = context + (int(t) - q0);  // absolute position of this token

  int jmin = p.window_left >= 0 ? max(0, pos - p.window_left) : 0;
  int jmax = p.causal != 0u ? pos : seqlen - 1;
  if (p.window_right >= 0) { jmax = min(jmax, pos + p.window_right); }
  jmax = min(jmax, seqlen - 1);
  if (jmax < jmin) { return; }  // cpu_paged_attn.cpp `continue`: out left untouched

  const uint g = h / p.qpk;
  const ulong qoff = (ulong(t) * ulong(p.hq) + ulong(h)) * ulong(p.d);

  for (uint e = tid; e < p.d; e += p.tg) { acc[e] = 0.0f; }
  if (tid == 0u) { sh_m = -INFINITY; sh_l = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j0 = jmin; j0 <= jmax; j0 += int(VT_PA_CHUNK)) {
    const int jc = min(int(VT_PA_CHUNK), jmax - j0 + 1);

    // Scores for this chunk (scaled), one thread per key.
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
      const int j = j0 + idx;
      const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
      const int off = j % int(p.block_size);
      const ulong kbase = ulong(blk) * p.kc_blk + ulong(off) * p.kc_pg + ulong(g) * p.kc_hd;
      float dot = 0.0f;
      for (uint e = 0u; e < p.d; ++e) {
        dot += vt_load(q, p.q_dt, qoff + ulong(e)) * vt_load(kc, p.kc_dt, kbase + ulong(e));
      }
      scores[idx] = dot * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Running max, then rescale the accumulator and the denominator.
    float pm = -INFINITY;
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) { pm = max(pm, scores[idx]); }
    const float mchunk = vt_tg_max(smem, tid, p.tg, pm);
    const float mold = sh_m;
    const float mnew = max(mold, mchunk);
    const float corr = mold == -INFINITY ? 0.0f : exp(mold - mnew);
    for (uint e = tid; e < p.d; e += p.tg) { acc[e] *= corr; }

    for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
      scores[idx] = exp(scores[idx] - mnew);
    }
    float ps = 0.0f;
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) { ps += scores[idx]; }
    // vt_tg_sum's LEADING barrier is what makes every thread's exp() above
    // visible to the V accumulation below, which reads the WHOLE chunk.
    const float lchunk = vt_tg_sum(smem, tid, p.tg, ps);

    // Weighted V accumulation, parallel over the head dimension so each thread
    // owns its own acc[] slots.
    for (uint e = tid; e < p.d; e += p.tg) {
      float s = 0.0f;
      for (int idx = 0; idx < jc; ++idx) {
        const int j = j0 + idx;
        const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
        const int off = j % int(p.block_size);
        const ulong vbase = ulong(blk) * p.vc_blk + ulong(off) * p.vc_pg + ulong(g) * p.vc_hd;
        s += scores[idx] * vt_load(vc, p.vc_dt, vbase + ulong(e));
      }
      acc[e] += s;
    }
    if (tid == 0u) { sh_l = sh_l * corr + lchunk; sh_m = mnew; }
    // Orders this chunk's acc/scores reads before the next chunk overwrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / sh_l;  // every valid window has >= 1 key
  for (uint e = tid; e < p.d; e += p.tg) {
    vt_store(out, p.out_dt, qoff + ulong(e), acc[e] * inv);
  }
}

struct VtArgmaxParams {
  uint n;
  uint v;
  uint tg;
};

// cpu_sample.cpp:40-57 GreedyArgmaxKernel. Upstream `greedy_sample` is
// argmax(dim=-1) and our CPU reference uses a STRICT `>` so the FIRST (lowest-
// index) maximum wins, bit-exact vs torch.argmax. A tree reduction must
// reproduce that tie rule explicitly, so the combine is "greater value, or equal
// value at a lower index" — which makes the result independent of the reduction
// order and therefore BIT-EXACT vs the CPU reference, not merely close.
kernel void vt_greedy_argmax(device const float* logits [[buffer(0)]],
                             device long*        ids    [[buffer(1)]],
                             constant VtArgmaxParams& p [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float bv[VT_TG_MAX];
  threadgroup uint  bi[VT_TG_MAX];
  const ulong base = ulong(row) * ulong(p.v);

  float best_v = -INFINITY;
  uint best_i = 0xFFFFFFFFu;
  for (uint j = tid; j < p.v; j += p.tg) {
    const float x = logits[base + ulong(j)];
    if (x > best_v || (x == best_v && j < best_i)) { best_v = x; best_i = j; }
  }
  bv[tid] = best_v;
  bi[tid] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = p.tg / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      const float ov = bv[tid + s];
      const uint oi = bi[tid + s];
      if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) { ids[row] = long(bi[0] == 0xFFFFFFFFu ? 0u : bi[0]); }
}

// ===========================================================================
// RoPE (NeoX) — the Qwen3-dense default (deterministic) rotation, work row M3b.
// ===========================================================================
// cpu_ops.cpp:636-665 RopeRotateHead / RopeNeoxKernel. In-place rotation of the
// leading `rot` dims of every (token, head) row of q [T,Hq,D] and k [T,Hk,D],
// NeoX split (pair `i` with `i+half`). One thread per (row, pair), where a row is
// one (token, head) in q (rows [0, T*Hq)) then one in k (rows [T*Hq, T*Hq+T*Hk)).
// Non-reducing and per-element, so no thread aliases another's pair.
//
// NUMERICS. The CPU/CUDA reference computes freq/angle and cos/sin in DOUBLE and
// casts to f32; Metal has no double, so this uses f32 `pow`/`precise::cos`/
// `precise::sin`. The rotation arithmetic (x*c - y*s / x*s + y*c) is otherwise the
// identical f32 form with a single bf16 rounding on store. Per the spike § Gates
// the bar is therefore NMSE <= 5e-4 vs the CPU oracle, NOT bit-exactness — the
// same posture as vt_paged_attention. For the short-prompt greedy gate the angles
// stay small enough that the bf16-rounded result matches the reference at all but
// bf16 near-ties (measured on the M4).
struct VtRopeParams {
  uint  t;        // tokens
  uint  hq;       // query heads
  uint  hk;       // key heads
  uint  d;        // head_dim
  uint  rot;      // rotary_dim (even, <= d)
  uint  rhalf;    // rot / 2
  uint  q_dt;
  uint  k_dt;
  uint  pos_i64;  // 1 => positions are i64, else i32
  float base;
};

kernel void vt_rope_neox(device uchar* q             [[buffer(0)]],
                         device uchar* k             [[buffer(1)]],
                         device const uchar* positions [[buffer(2)]],
                         constant VtRopeParams& p    [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint pair = gid % p.rhalf;
  const uint row  = gid / p.rhalf;
  const uint qrows = p.t * p.hq;
  device uchar* buf;
  uint dt;
  uint token;
  ulong head_off;
  if (row < qrows) {
    buf = q; dt = p.q_dt; token = row / p.hq; head_off = ulong(row) * p.d;
  } else {
    const uint r2 = row - qrows;
    buf = k; dt = p.k_dt; token = r2 / p.hk; head_off = ulong(r2) * p.d;
  }
  const long pos = p.pos_i64
      ? ((device const long*)positions)[token]
      : (long)((device const int*)positions)[token];
  const float freq  = pow(p.base, -2.0f * float(pair) / float(p.rot));
  const float angle = float(pos) * freq;
  const float c = precise::cos(angle);
  const float s = precise::sin(angle);
  const float x = vt_load(buf, dt, head_off + pair);
  const float y = vt_load(buf, dt, head_off + pair + p.rhalf);
  vt_store(buf, dt, head_off + pair,          x * c - y * s);
  vt_store(buf, dt, head_off + pair + p.rhalf, x * s + y * c);
}

// cpu_ops.cpp:751-768 RopeCosSinCacheKernel. Fill cos_sin[T, rot]: cols [0,half)
// = cos, [half,rot) = sin. One thread per (token, pair). Same f32-transcendental
// deviation and NMSE bar as vt_rope_neox above. (Built once per step by the dense
// attention preamble; consumed only by the opt-in RopeFromCache path, so on the
// deterministic default path its output is unused — but the op must exist or the
// engine's GetOp throws.)
struct VtRopeCacheParams {
  uint  t;
  uint  rot;
  uint  rhalf;
  uint  out_dt;
  uint  pos_i64;
  float base;
};

kernel void vt_rope_cos_sin_cache(device uchar* cos_sin          [[buffer(0)]],
                                  device const uchar* positions  [[buffer(1)]],
                                  constant VtRopeCacheParams& p  [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  const uint pair  = gid % p.rhalf;
  const uint token = gid / p.rhalf;
  const long pos = p.pos_i64
      ? ((device const long*)positions)[token]
      : (long)((device const int*)positions)[token];
  const float freq  = pow(p.base, -2.0f * float(pair) / float(p.rot));
  const float angle = float(pos) * freq;
  vt_store(cos_sin, p.out_dt, ulong(token) * p.rot + pair,           precise::cos(angle));
  vt_store(cos_sin, p.out_dt, ulong(token) * p.rot + p.rhalf + pair, precise::sin(angle));
}

// ===========================================================================
// RoPE from a precomputed cos|sin cache — the Qwen3-dense DEFAULT rotation
// (VT_QWEN3_ROPE_CACHE defaults ON), work row M3b.
// ===========================================================================
// cpu_ops.cpp:690-742 RopeFromCacheKernel. Rotate the leading `rot` dims of every
// (token, head) row of q [T,Hq,D] and k [T,Hk,D] using cos|sin READ from a
// cos_sin[P, rot] cache (cols [0,half)=cos, [half,rot)=sin), indexed by
// positions[token]. STRIDE-DRIVEN (q/k need only a unit-stride innermost dim).
// One thread per (row, pair). No transcendentals in the kernel — the c/s are the
// SAME cached values the CPU reference reads, and the rotation (x*c - y*s /
// x*s + y*c) is the identical f32 arithmetic with a single bf16 rounding on store,
// so this is bit-exact to the CPU oracle (a non-reducing per-element op). This is
// the op the correctness gate actually exercises; kRopeNeox above serves the
// VT_QWEN3_ROPE_CACHE=0 opt-out. Rank-1 positions only (MRoPE is guarded off in
// the host wrapper — Qwen3-dense never uses it).
struct VtRopeApplyParams {
  ulong q_s0;    // q element stride over tokens
  ulong q_s1;    // q element stride over heads
  ulong k_s0;
  ulong k_s1;
  uint  t;
  uint  hq;
  uint  hk;
  uint  rot;
  uint  rhalf;
  uint  is_neox;
  uint  q_dt;
  uint  k_dt;
  uint  cache_dt;
  uint  pos_i64;
  uint  has_k;
  uint  pad;
};

kernel void vt_rope_from_cache(device uchar* q               [[buffer(0)]],
                               device uchar* k               [[buffer(1)]],
                               device const uchar* positions [[buffer(2)]],
                               device const uchar* cache     [[buffer(3)]],
                               constant VtRopeApplyParams& p [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
  const uint pair = gid % p.rhalf;
  const uint row  = gid / p.rhalf;
  const uint qrows = p.t * p.hq;
  device uchar* buf;
  uint dt;
  ulong s0, s1;
  uint token, head;
  if (row < qrows) {
    buf = q; dt = p.q_dt; s0 = p.q_s0; s1 = p.q_s1; token = row / p.hq; head = row % p.hq;
  } else {
    if (p.has_k == 0u) return;
    const uint r2 = row - qrows;
    buf = k; dt = p.k_dt; s0 = p.k_s0; s1 = p.k_s1; token = r2 / p.hk; head = r2 % p.hk;
  }
  const long pos = p.pos_i64
      ? ((device const long*)positions)[token]
      : (long)((device const int*)positions)[token];
  const ulong coff = ulong(pos) * p.rot;
  const float c = vt_load(cache, p.cache_dt, coff + pair);
  const float s = vt_load(cache, p.cache_dt, coff + p.rhalf + pair);
  const uint first  = p.is_neox ? pair : pair * 2u;
  const uint second = p.is_neox ? pair + p.rhalf : pair * 2u + 1u;
  const ulong off = ulong(token) * s0 + ulong(head) * s1;
  const float x = vt_load(buf, dt, off + first);
  const float y = vt_load(buf, dt, off + second);
  vt_store(buf, dt, off + first,  x * c - y * s);
  vt_store(buf, dt, off + second, x * s + y * c);
}
)MSL";
// clang-format on

}  // namespace vt::metal

#endif  // VT_METAL_METAL_MSL_H_
