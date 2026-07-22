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
)MSL";
// clang-format on

}  // namespace vt::metal

#endif  // VT_METAL_METAL_MSL_H_
