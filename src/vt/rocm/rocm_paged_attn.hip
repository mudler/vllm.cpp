// ROCm paged attention — port of cpu_paged_attn.cpp algebra + CUDA device-side
// request lookup (cuda_paged_attn.cu). Online softmax; NHD KV via strides.
// Supports bf16/f32 query x bf16/f32 KV (mixed allowed).
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
// rocWMMA only on gfx12 (R9700). Fat builds also emit gfx1036 — must not include there.
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
#define VT_ROCWMMA_OK 1
#endif

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/backend.h"
#include "vt/rocm/rocm_arch.h"

#include <array>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

// Per-device, once: launch SharedK WMMA only when this device's gcnArchName is
// gfx1200/gfx1201 (prefix). Not per-build, not getenv. Device kernel body
// still uses VT_ROCWMMA_OK (#785).
bool PrefillSharedKWmmaHostOk() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0) return false;
  struct Slot {
    std::once_flag once;
    bool ok = false;
  };
  static std::array<Slot, 64> slots;
  if (static_cast<unsigned>(dev) >= slots.size()) return false;
  std::call_once(slots[static_cast<unsigned>(dev)].once, [dev] {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
      slots[static_cast<unsigned>(dev)].ok = false;
      return;
    }
    slots[static_cast<unsigned>(dev)].ok = GcnArchNameIsGfx12PrefillWmma(prop.gcnArchName);
  });
  return slots[static_cast<unsigned>(dev)].ok;
}

// Optional correctness path: run the CPU paged-attn oracle on host copies.
// VT_ROCM_ATTN_CPU_REF=1 — isolates whether custom HIP attn is the decode bug.
bool CpuRefEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_ATTN_CPU_REF");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

void PagedAttnCpuRef(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                     const Tensor& v_cache, const Tensor& block_table,
                     const Tensor& seq_lens, const Tensor& query_start_loc,
                     const PagedAttentionArgs& args) {
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  Check(hipStreamSynchronize(st), "cpu_ref sync in");

  // Contiguous host packs (engine KV is a strided unbind view — cannot Bytes()-memcpy).
  auto pack4 = [&](const Tensor& t) {
    VT_CHECK(t.rank == 4, "cpu_ref pack4");
    const int64_t n0 = t.shape[0], n1 = t.shape[1], n2 = t.shape[2], n3 = t.shape[3];
    const size_t es = vt::SizeOf(t.dtype);
    std::vector<uint8_t> buf(static_cast<size_t>(n0 * n1 * n2 * n3) * es);
    // element-wise D2H via one big buffer read of parent is hard; use hipMemcpy2D-ish loop
    // Simple: copy full bounding box if contiguous innermost
    VT_CHECK(t.stride[3] == 1, "cpu_ref need unit inner stride");
    for (int64_t i0 = 0; i0 < n0; ++i0)
      for (int64_t i1 = 0; i1 < n1; ++i1)
        for (int64_t i2 = 0; i2 < n2; ++i2) {
          const int64_t src_off = i0 * t.stride[0] + i1 * t.stride[1] + i2 * t.stride[2];
          const int64_t dst_off = ((i0 * n1 + i1) * n2 + i2) * n3;
          Check(hipMemcpy(buf.data() + static_cast<size_t>(dst_off) * es,
                          static_cast<const char*>(t.data) + static_cast<size_t>(src_off) * es,
                          static_cast<size_t>(n3) * es, hipMemcpyDeviceToHost),
                "pack4 d2h");
        }
    return buf;
  };
  auto pack_contig = [&](const Tensor& t) {
    std::vector<uint8_t> buf(t.Bytes());
    Check(hipMemcpy(buf.data(), t.data, t.Bytes(), hipMemcpyDeviceToHost), "d2h");
    return buf;
  };

  auto h_q = pack_contig(query);
  auto h_k = pack4(k_cache);
  auto h_v = pack4(v_cache);
  auto h_bt = pack_contig(block_table);
  auto h_sl = pack_contig(seq_lens);
  auto h_qsl = pack_contig(query_start_loc);
  std::vector<uint8_t> h_out(out.Bytes(), 0);

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue cq = cpu.CreateQueue();
  const vt::Device cd{vt::DeviceType::kCPU, 0};

  Tensor tq = Tensor::Contiguous(h_q.data(), query.dtype, cd,
                                 {query.shape[0], query.shape[1], query.shape[2]});
  Tensor tk = Tensor::Contiguous(h_k.data(), k_cache.dtype, cd,
                                 {k_cache.shape[0], k_cache.shape[1], k_cache.shape[2],
                                  k_cache.shape[3]});
  Tensor tv = Tensor::Contiguous(h_v.data(), v_cache.dtype, cd,
                                 {v_cache.shape[0], v_cache.shape[1], v_cache.shape[2],
                                  v_cache.shape[3]});
  Tensor tbt = Tensor::Contiguous(h_bt.data(), block_table.dtype, cd,
                                  {block_table.shape[0], block_table.shape[1]});
  Tensor tsl = Tensor::Contiguous(h_sl.data(), seq_lens.dtype, cd, {seq_lens.shape[0]});
  Tensor tqsl =
      Tensor::Contiguous(h_qsl.data(), query_start_loc.dtype, cd, {query_start_loc.shape[0]});
  Tensor to = Tensor::Contiguous(h_out.data(), out.dtype, cd,
                                 {out.shape[0], out.shape[1], out.shape[2]});
  vt::PagedAttention(cq, to, tq, tk, tv, tbt, tsl, tqsl, args);
  cpu.DestroyQueue(cq);

  Check(hipMemcpy(out.data, h_out.data(), out.Bytes(), hipMemcpyHostToDevice), "h2d out");
  Check(hipStreamSynchronize(st), "cpu_ref sync out");
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

__device__ inline float Softcap(float s, float cap) {
  return cap > 0.f ? cap * tanhf(s / cap) : s;
}

// gfx1201: exp2 is the native path; expf often lowers slower. Used in online softmax.
__device__ inline float FastExp(float x) {
  return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);  // x * log2(e)
}

// One block per (query_token, q_head). Online softmax matches CUDA.
// Fast-path: num_reqs==1 skips linear request scan (Hermes/single-seq decode).
// Vectorized x4 Q·K / V when head_dim % 4 == 0 (Gemma-4 Dh is).
template <typename TQ, typename TKV, typename TO>
__global__ void PagedAttnOnline(TO* out, const TQ* query, const TKV* k_cache,
                                const TKV* v_cache, const int32_t* block_table,
                                const int32_t* seq_lens, const int32_t* query_start_loc,
                                int64_t num_reqs, int64_t hq, int64_t num_kv_heads, int64_t d,
                                int64_t block_size, int64_t bt_row, int64_t bt_col,
                                int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                int64_t vc_pg, int64_t vc_hd, float scale, float softcap,
                                bool causal, int window_left, int window_right) {
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (h >= hq) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);

  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;
  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;
  const bool vec4 = (d % 4) == 0;

  extern __shared__ float smem[];
  float* acc = smem;
  float* red = smem + d;
  __shared__ float s_score, s_m, s_l;

  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.f;
  if (threadIdx.x == 0) {
    s_m = -INFINITY;
    s_l = 0.f;
  }
  __syncthreads();

  if (jmax >= jmin) {
    for (int64_t j = jmin; j <= jmax; ++j) {
      const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
      const int64_t off = j % block_size;
      const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
      float part = 0.f;
      if (vec4) {
        const int64_t nvec = d >> 2;
        for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
          const int64_t e = v << 2;
          part += Ld(query, qoff + e) * Ld(k_cache, kbase + e);
          part += Ld(query, qoff + e + 1) * Ld(k_cache, kbase + e + 1);
          part += Ld(query, qoff + e + 2) * Ld(k_cache, kbase + e + 2);
          part += Ld(query, qoff + e + 3) * Ld(k_cache, kbase + e + 3);
        }
      } else {
        for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
          part += Ld(query, qoff + e) * Ld(k_cache, kbase + e);
      }
      red[threadIdx.x] = part;
      __syncthreads();
      for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
      }
      if (threadIdx.x == 0) s_score = Softcap(red[0] * scale, softcap);
      __syncthreads();

      const float s = s_score;
      const float m_new = fmaxf(s_m, s);
      const float corr = expf(s_m - m_new);
      const float pw = expf(s - m_new);
      const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
      if (vec4) {
        const int64_t nvec = d >> 2;
        for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
          const int64_t e = v << 2;
          acc[e] = acc[e] * corr + pw * Ld(v_cache, vbase + e);
          acc[e + 1] = acc[e + 1] * corr + pw * Ld(v_cache, vbase + e + 1);
          acc[e + 2] = acc[e + 2] * corr + pw * Ld(v_cache, vbase + e + 2);
          acc[e + 3] = acc[e + 3] * corr + pw * Ld(v_cache, vbase + e + 3);
        }
      } else {
        for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
          acc[e] = acc[e] * corr + pw * Ld(v_cache, vbase + e);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        s_l = s_l * corr + pw;
        s_m = m_new;
      }
      __syncthreads();
    }
  }

  const float inv = (s_l > 0.f) ? (1.f / s_l) : 0.f;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) St(out, qoff + e, acc[e] * inv);
}

// ===========================================================================
// DECODE-oriented path (ported from cuda_paged_attn DecodeOpt). d==256 bf16.
// 8 warps stride KV; warp-shuffle Q·K; register online softmax; no per-key
// block sync. Grid (total_q, hq), block 256.
// ===========================================================================
constexpr int kDecWarps = 8;
constexpr int kDecEpl = 8;  // 32 lanes * 8 = 256 head_dim

// EPL elems/lane: 4 → d=128 (Qwen3/Llama-class GQA), 8 → d=256, 16 → d=512
// (32 lanes). d=128 (#488, ROCM-DECODE-ATTN-D128): the kernel body below is
// already generic in EPL; only the 4/8/16-element vectorized load/store and
// the host dispatch needed a new case, not a new kernel.
template <int EPL>
__device__ inline void LoadRowEplBf16(const __hip_bfloat16* p, int64_t base, int lane, float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
  if constexpr (EPL == 4) {
    const uint2* u = reinterpret_cast<const uint2*>(p + base);
    const uint2 w = u[lane];
    const __hip_bfloat16* h = reinterpret_cast<const __hip_bfloat16*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = __bfloat162float(h[i]);
  } else {
    const uint4* u = reinterpret_cast<const uint4*>(p + base);
    if constexpr (EPL == 8) {
      const uint4 w = u[lane];
      const __hip_bfloat16* h = reinterpret_cast<const __hip_bfloat16*>(&w);
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] = __bfloat162float(h[i]);
    } else {
      const uint4 w0 = u[lane];
      const uint4 w1 = u[lane + 32];
      const __hip_bfloat16* h0 = reinterpret_cast<const __hip_bfloat16*>(&w0);
      const __hip_bfloat16* h1 = reinterpret_cast<const __hip_bfloat16*>(&w1);
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] = __bfloat162float(h0[i]);
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i + 8] = __bfloat162float(h1[i]);
    }
  }
}

template <int EPL>
__device__ inline void StoreRowEplBf16(__hip_bfloat16* p, int64_t base, int lane, const float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
  if constexpr (EPL == 4) {
    __hip_bfloat16 tmp[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) tmp[i] = __float2bfloat16(r[i]);
    reinterpret_cast<uint2*>(p + base)[lane] = *reinterpret_cast<const uint2*>(tmp);
  } else if constexpr (EPL == 8) {
    __hip_bfloat16 tmp[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) tmp[i] = __float2bfloat16(r[i]);
    reinterpret_cast<uint4*>(p + base)[lane] = *reinterpret_cast<const uint4*>(tmp);
  } else {
    __hip_bfloat16 t0[8], t1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) t0[i] = __float2bfloat16(r[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) t1[i] = __float2bfloat16(r[i + 8]);
    uint4* u = reinterpret_cast<uint4*>(p + base);
    u[lane] = *reinterpret_cast<const uint4*>(t0);
    u[lane + 32] = *reinterpret_cast<const uint4*>(t1);
  }
}

__device__ inline void LoadRow8Bf16(const __hip_bfloat16* p, int64_t base, int lane, float r[8]) {
  LoadRowEplBf16<8>(p, base, lane, r);
}
__device__ inline void StoreRow8Bf16(__hip_bfloat16* p, int64_t base, int lane, const float r[8]) {
  StoreRowEplBf16<8>(p, base, lane, r);
}

// BF16 Q/K/V/Out; EPL=4 → d=128, EPL=8 → d=256, EPL=16 → d=512. No per-key block sync
// (decode+prefill). The d=128 case is opt-in behind VT_ATTN_DECODE_D128 (default OFF).
template <int EPL>
__global__ void PagedAttnDecodeOptBf16T(__hip_bfloat16* out, const __hip_bfloat16* query,
                                        const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                        const int32_t* block_table, const int32_t* seq_lens,
                                        const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                        int64_t num_kv_heads, int64_t d, int64_t block_size,
                                        int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                        int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                        float scale, float softcap, bool causal, int window_left,
                                        int window_right) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (h >= hq || d != d_expect) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);

  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;

  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;

  float q_reg[kEpl];
  LoadRowEplBf16<kEpl>(query, qoff, lane, q_reg);

  float m = -INFINITY, lsum = 0.f;
  float o_reg[kEpl];
#pragma unroll
  for (int i = 0; i < kEpl; ++i) o_reg[i] = 0.f;

  for (int64_t j = jmin + warp; j <= jmax; j += kDecWarps) {
    const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
    const int64_t off = j % block_size;
    float k_reg[kEpl];
    LoadRowEplBf16<kEpl>(k_cache, blk * kc_blk + off * kc_pg + g * kc_hd, lane, k_reg);
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) dot += q_reg[i] * k_reg[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
    dot = __shfl(dot, 0);

    const float s = Softcap(dot * scale, softcap);
    const float m_new = fmaxf(m, s);
    const float corr = FastExp(m - m_new);
    const float pw = FastExp(s - m_new);
    float v_reg[kEpl];
    LoadRowEplBf16<kEpl>(v_cache, blk * vc_blk + off * vc_pg + g * vc_hd, lane, v_reg);
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[i] = o_reg[i] * corr + pw * v_reg[i];
    lsum = lsum * corr + pw;
    m = m_new;
  }

  extern __shared__ float smem_dec[];
  float* o_sh = smem_dec;
  float* m_sh = o_sh + kDecWarps * d;
  float* l_sh = m_sh + kDecWarps;
#pragma unroll
  for (int i = 0; i < kEpl; ++i) o_sh[warp * d + lane * kEpl + i] = o_reg[i];
  if (lane == 0) {
    m_sh[warp] = m;
    l_sh[warp] = lsum;
  }
  __syncthreads();

  if (warp == 0) {
    float gm = -INFINITY;
#pragma unroll
    for (int w = 0; w < kDecWarps; ++w) gm = fmaxf(gm, m_sh[w]);
    float gl = 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = 0.f;
#pragma unroll
    for (int w = 0; w < kDecWarps; ++w) {
      const float sc = FastExp(m_sh[w] - gm);
      gl += l_sh[w] * sc;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] += sc * o_sh[w * d + lane * kEpl + i];
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] *= inv;
    StoreRowEplBf16<kEpl>(out, qoff, lane, acc);
  }
}

// GQA-fused decode: one CTA per KV head; load K/V once, attend QG q-heads.
// Gemma-4: hq=16, kv=8 → QG=2. EPL=4→d=128, EPL=8→d=256, EPL=16→d=512.
// grid=(total_q, num_kv_heads). The d=128 case is opt-in behind VT_ATTN_DECODE_D128 (default OFF).
template <int QG, int EPL = 8, int NWARPS = kDecWarps>
__global__ void PagedAttnDecodeGqaBf16(__hip_bfloat16* out, const __hip_bfloat16* query,
                                       const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                       const int32_t* block_table, const int32_t* seq_lens,
                                       const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                       int64_t num_kv_heads, int64_t d, int64_t block_size,
                                       int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                       int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                       float scale, float softcap, bool causal, int window_left,
                                       int window_right) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (g >= num_kv_heads || d != d_expect) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);
  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;

  const int64_t qg_total = hq / num_kv_heads;
  // blockIdx.z splits a large GQA group into QG-sized tiles (e.g. global QG=8 → two×4).
  const int64_t h0 = g * qg_total + static_cast<int64_t>(blockIdx.z) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;
  float q_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh)
    LoadRowEplBf16<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);

  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }

  for (int64_t j = jmin + warp; j <= jmax; j += NWARPS) {
    const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
    const int64_t off = j % block_size;
    float k_reg[kEpl];
    LoadRowEplBf16<kEpl>(k_cache, blk * kc_blk + off * kc_pg + g * kc_hd, lane, k_reg);

    float s_h[QG];
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
      s_h[hh] = Softcap(__shfl(dot, 0) * scale, softcap);
    }

    float v_reg[kEpl];
    LoadRowEplBf16<kEpl>(v_cache, blk * vc_blk + off * vc_pg + g * vc_hd, lane, v_reg);
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float m_new = fmaxf(m[hh], s_h[hh]);
      const float corr = FastExp(m[hh] - m_new);
      const float pw = FastExp(s_h[hh] - m_new);
#pragma unroll
      for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
      lsum[hh] = lsum[hh] * corr + pw;
      m[hh] = m_new;
    }
  }

  extern __shared__ float smem_gqa[];
  float* o_sh = smem_gqa;
  float* m_sh = o_sh + static_cast<size_t>(NWARPS) * QG * d;
  float* l_sh = m_sh + static_cast<size_t>(NWARPS) * QG;
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    float* dst = o_sh + (static_cast<size_t>(warp) * QG + hh) * d + lane * kEpl;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) dst[i] = o_reg[hh][i];
    if (lane == 0) {
      m_sh[warp * QG + hh] = m[hh];
      l_sh[warp * QG + hh] = lsum[hh];
    }
  }
  __syncthreads();

  for (int hh = warp; hh < QG; hh += NWARPS) {
    float gm = -INFINITY;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) gm = fmaxf(gm, m_sh[w * QG + hh]);
    float gl = 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = 0.f;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      const float sc = FastExp(m_sh[w * QG + hh] - gm);
      gl += l_sh[w * QG + hh] * sc;
      const float* src = o_sh + (static_cast<size_t>(w) * QG + hh) * d + lane * kEpl;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] += sc * src[i];
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] *= inv;
    StoreRowEplBf16<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
  }
}


// SGLang-style flash prefill GQA (steal base 2026-08-10):
//   BLOCK_M queries × BLOCK_N keys, Q+K tiles in smem, online softmax.
//   HIP default tiles from extend_attention: BLOCK_M=64, BLOCK_N=64.
// grid=(ceil(total_q/BM), num_kv_heads). One CTA = one Q-tile × one KV head (QG q-heads).
// SGLang-style tile flash: BM×BN, Q+K in smem, o/m/l in registers (no huge o smem).
// HIP steal defaults ~64×64; we use BM=16 BN=64 @d=256 and BM=8 BN=64 @d=512 to fit LDS.
template <int QG, int EPL, int BM, int BN>
__global__ void PagedAttnPrefillFlashTile(__hip_bfloat16* out, const __hip_bfloat16* query,
                                          const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                          const int32_t* block_table, const int32_t* seq_lens,
                                          const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                          int64_t num_kv_heads, int64_t d, int64_t block_size,
                                          int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                          int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                          float scale, float softcap, bool causal, int window_left,
                                          int window_right) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  // Max rows this CTA tracks = BM * QG; each warp owns a strided subset.
  constexpr int kRows = BM * QG;
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  if (g >= num_kv_heads || d != d_expect) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int nrows = nq * QG;

  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  __shared__ int64_t s_jmin, s_jmax;
  for (int qq = tid; qq < BM; qq += static_cast<int>(blockDim.x)) {
    if (qq >= nq) {
      s_jmin_q[qq] = seqlen;
      s_jmax_q[qq] = -1;
      continue;
    }
    const int64_t t = t_base + qq;
    const int64_t pos = context + (t - q0);
    int64_t jn = 0;
    if (window_left >= 0) {
      jn = pos - window_left;
      if (jn < 0) jn = 0;
    }
    int64_t jx = causal ? pos : (seqlen - 1);
    if (window_right >= 0) {
      const int64_t jr = pos + window_right;
      if (jr < jx) jx = jr;
    }
    if (jx > seqlen - 1) jx = seqlen - 1;
    s_jmin_q[qq] = jn;
    s_jmax_q[qq] = jx;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int qq = 0; qq < nq; ++qq) {
      if (s_jmin_q[qq] < tn) tn = s_jmin_q[qq];
      if (s_jmax_q[qq] > tx) tx = s_jmax_q[qq];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  const int64_t h0 = g * QG;
  // smem: Q[BM*QG*d] bf16 + KV[BN*d] bf16  (d=256 BM=16 BN=64 → 16KB+32KB)
  extern __shared__ __hip_bfloat16 dyn_bf[];
  __hip_bfloat16* q_sm = dyn_bf;
  __hip_bfloat16* kv_sm = q_sm + static_cast<size_t>(BM) * QG * d;
  __shared__ int64_t s_blk[BN], s_off[BN];
  // Scores: QG planes of BM×BN so rocWMMA can store_matrix_sync with ld=BN
  // (consecutive qq for fixed hh). No per-warp tile_s[8] — that overflowed LDS.
  __shared__ float s_sc[QG * BM * BN];

  // Load Q tile; zero-pad to full BM so WMMA never OOB on partial last tiles.
  for (int e = tid; e < BM * QG * static_cast<int>(d); e += static_cast<int>(blockDim.x))
    q_sm[e] = __hip_bfloat16{};
  __syncthreads();
  for (int e = tid; e < nq * QG * static_cast<int>(d); e += static_cast<int>(blockDim.x)) {
    const int dd = e % static_cast<int>(d);
    const int tmp = e / static_cast<int>(d);
    const int hh = tmp % QG;
    const int qq = tmp / QG;
    q_sm[(static_cast<size_t>(qq) * QG + hh) * d + dd] =
        query[((t_base + qq) * hq + (h0 + hh)) * d + dd];
  }
  __syncthreads();

  // Per-row online state in registers for rows owned by this warp.
  // Cap: BM=16 QG=2 → 32 rows, 8 warps → 4 rows/warp.
  constexpr int kMaxRowsPerWarp = (kRows + 7) / 8;  // assume >=8 warps
  float m_r[kMaxRowsPerWarp];
  float l_r[kMaxRowsPerWarp];
  float o_r[kMaxRowsPerWarp][kEpl];
  int row_ids[kMaxRowsPerWarp];
  int n_own = 0;
  for (int row = warp; row < nrows && n_own < kMaxRowsPerWarp; row += nwarps) {
    row_ids[n_own] = row;
    m_r[n_own] = -INFINITY;
    l_r[n_own] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_r[n_own][i] = 0.f;
    ++n_own;
  }

  for (int64_t j0 = s_jmin; j0 <= s_jmax; j0 += BN) {
    const int n_k = static_cast<int>((s_jmax - j0 + 1) < BN ? (s_jmax - j0 + 1) : BN);
    const int bdx = static_cast<int>(blockDim.x);
    for (int kk = tid; kk < n_k; kk += bdx) {
      const int64_t j = j0 + kk;
      s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
      s_off[kk] = j % block_size;
    }
    __syncthreads();
    for (int e = tid; e < BN * static_cast<int>(d); e += bdx) kv_sm[e] = __hip_bfloat16{};
    __syncthreads();
    for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = k_cache[s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd];
    }
    __syncthreads();

    // QK: concurrent multi-wave rocWMMA → s_sc[hh][qq][kk] (ld=BN).
    // Direct store (no tile_s[8]) keeps LDS under 64KB at BM=16 BN=64 d=256.
    constexpr bool kUseWmma =
        (BM % 16 == 0) && (BN % 16 == 0) && (d_expect % 16 == 0) && (BM >= 16) && (BN >= 16);
#if defined(VT_ROCWMMA_OK)
    if constexpr (kUseWmma) {
      constexpr int WM = 16, WN = 16, WK = 16;
      using namespace rocwmma;
      const int tiles_m = BM / WM;
      const int tiles_n = BN / WN;
      const int ntiles_h = tiles_m * tiles_n;
      const int ntiles = ntiles_h * QG;
      const int step = (nwarps < 8) ? nwarps : 8;
      if (warp < 8) {
        for (int t = warp; t < ntiles; t += step) {
          const int hh = t / ntiles_h;
          const int t2 = t - hh * ntiles_h;
          const int tm = t2 / tiles_n;
          const int tn = t2 - tm * tiles_n;
          const int m0 = tm * WM;
          const int n0 = tn * WN;
          fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> b_frag;
          fragment<accumulator, WM, WN, WK, float> acc;
          fill_fragment(acc, 0.f);
          const bfloat16_t* Qbase =
              reinterpret_cast<const bfloat16_t*>(q_sm + static_cast<size_t>(hh) * d);
          const long lda = static_cast<long>(QG) * static_cast<long>(d);
          const bfloat16_t* Kbase =
              reinterpret_cast<const bfloat16_t*>(kv_sm + static_cast<size_t>(n0) * d);
          for (int k = 0; k < d_expect; k += WK) {
            load_matrix_sync(a_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
            load_matrix_sync(b_frag, Kbase + k, d);
            mma_sync(acc, a_frag, b_frag, acc);
          }
          // Plane hh: rows = qq, cols = kk, leading dim BN
          float* sc_base = s_sc + static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(m0) * BN + n0;
          store_matrix_sync(sc_base, acc, BN, mem_row_major);
        }
      }
      __syncthreads();
      // Mask/softcap applied in online loop (no extra full-matrix pass).
    } else
#endif
    {
      for (int own = 0; own < n_own; ++own) {
        const int row = row_ids[own];
        const int qq = row / QG;
        const int hh = row - qq * QG;
#pragma unroll 1
        for (int kk = 0; kk < BN; ++kk) {
          float s = -INFINITY;
          if (kk < n_k) {
            const int64_t j = j0 + kk;
            if (j >= s_jmin_q[qq] && j <= s_jmax_q[qq]) {
              float dot = 0.f;
#pragma unroll
              for (int i = 0; i < kEpl; ++i) {
                const int dd = lane * kEpl + i;
                if (dd < d) {
                  dot += __bfloat162float(q_sm[(static_cast<size_t>(qq) * QG + hh) * d + dd]) *
                         __bfloat162float(kv_sm[static_cast<size_t>(kk) * d + dd]);
                }
              }
#pragma unroll
              for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
              s = Softcap(__shfl(dot, 0) * scale, softcap);
            }
          }
          if (lane == 0)
            s_sc[static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN + kk] = s;
        }
      }
      __syncthreads();
    }


    // V into kv_sm
    for (int e = tid; e < n_k * static_cast<int>(d); e += static_cast<int>(blockDim.x)) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = v_cache[s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd];
    }
    __syncthreads();

    // Tile-softmax + PV (FA2-style): one rowmax, then one exp pass — fewer
    // exp/renorm than per-key online. WMMA QK scores stay in s_sc.
    for (int own = 0; own < n_own; ++own) {
      const int row = row_ids[own];
      const int qq = row / QG;
      const int hh = row - qq * QG;
      float* rp = s_sc + static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN;

      float m_tile = -INFINITY;
#pragma unroll 1
      for (int kk = 0; kk < BN; ++kk) {
        if (kk >= n_k) break;
        float s;
        if constexpr (kUseWmma) {
          const int64_t j = j0 + kk;
          if (!(j >= s_jmin_q[qq] && j <= s_jmax_q[qq])) {
            rp[kk] = -INFINITY;
            continue;
          }
          s = Softcap(rp[kk] * scale, softcap);
        } else {
          s = rp[kk];
        }
        rp[kk] = s;
        m_tile = fmaxf(m_tile, s);
      }

      if (!(m_tile > -INFINITY)) continue;
      const float m_new = fmaxf(m_r[own], m_tile);
      const float e_old = (m_r[own] > -INFINITY) ? expf(m_r[own] - m_new) : 0.f;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) o_r[own][i] *= e_old;
      l_r[own] *= e_old;

      float l_add = 0.f;
#pragma unroll 1
      for (int kk = 0; kk < BN; ++kk) {
        if (kk >= n_k) break;
        const float s = rp[kk];
        if (!(s > -INFINITY)) continue;
        const float p = expf(s - m_new);
        l_add += p;
#pragma unroll
        for (int i = 0; i < kEpl; ++i) {
          const int dd = lane * kEpl + i;
          const float vv = (dd < d) ? __bfloat162float(kv_sm[static_cast<size_t>(kk) * d + dd]) : 0.f;
          o_r[own][i] += p * vv;
        }
      }
      l_r[own] += l_add;
      m_r[own] = m_new;
    }
    __syncthreads();
  }

  for (int own = 0; own < n_own; ++own) {
    const int row = row_ids[own];
    const int qq = row / QG;
    const int hh = row - qq * QG;
    const float inv = (l_r[own] > 0.f) ? (1.f / l_r[own]) : 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = o_r[own][i] * inv;
    StoreRowEplBf16<kEpl>(out, ((t_base + qq) * hq + (h0 + hh)) * d, lane, acc);
  }
}


// 1-wave/CTA flash prefill with rocWMMA QK (gfx1201-safe).
// grid=(ceil(total_q/BM), num_kv_heads), block=32.
// Parallelism = many CTAs; never multi-wave mma_sync (hangs on gfx1201).
template <int QG, int EPL, int BM, int BN>
__global__ void PagedAttnPrefillWmmaWave(__hip_bfloat16* out, const __hip_bfloat16* query,
                                         const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                         const int32_t* block_table, const int32_t* seq_lens,
                                         const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                         int64_t num_kv_heads, int64_t d, int64_t block_size,
                                         int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                         int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                         float scale, float softcap, bool causal, int window_left,
                                         int window_right) {
#if !defined(VT_ROCWMMA_OK)
  (void)out; (void)query; (void)k_cache; (void)v_cache; (void)block_table; (void)seq_lens;
  (void)query_start_loc; (void)num_reqs; (void)hq; (void)num_kv_heads; (void)d; (void)block_size;
  (void)bt_row; (void)bt_col; (void)kc_blk; (void)kc_pg; (void)kc_hd; (void)vc_blk; (void)vc_pg;
  (void)vc_hd; (void)scale; (void)softcap; (void)causal; (void)window_left; (void)window_right;
  return;
#else
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  constexpr int WM = 16, WN = 16, WK = 16;
  static_assert(BM % WM == 0 && BN % WN == 0, "WMMA tile alignment");
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int tid = lane;  // 1 wave
  if (g >= num_kv_heads || d != d_expect || threadIdx.x >= 32) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t h0 = g * QG;

  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  __shared__ int64_t s_jmin, s_jmax;
  for (int qq = tid; qq < BM; qq += 32) {
    if (qq >= nq) {
      s_jmin_q[qq] = seqlen;
      s_jmax_q[qq] = -1;
      continue;
    }
    const int64_t pos = context + ((t_base + qq) - q0);
    int64_t jn = 0;
    if (window_left >= 0) {
      jn = pos - window_left;
      if (jn < 0) jn = 0;
    }
    int64_t jx = causal ? pos : (seqlen - 1);
    if (window_right >= 0) {
      const int64_t jr = pos + window_right;
      if (jr < jx) jx = jr;
    }
    if (jx > seqlen - 1) jx = seqlen - 1;
    s_jmin_q[qq] = jn;
    s_jmax_q[qq] = jx;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int qq = 0; qq < nq; ++qq) {
      if (s_jmin_q[qq] < tn) tn = s_jmin_q[qq];
      if (s_jmax_q[qq] > tx) tx = s_jmax_q[qq];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  // smem: Q[BM*QG*d] + KV[BN*d]
  extern __shared__ __hip_bfloat16 dyn[];
  __hip_bfloat16* q_sm = dyn;
  __hip_bfloat16* kv_sm = dyn + static_cast<size_t>(BM) * QG * d;
  __shared__ int64_t s_blk[BN], s_off[BN];
  __shared__ float s_score[BM * QG * BN];
  __shared__ float tile_s[WM * WN];

  // zero + load Q
  for (int e = tid; e < BM * QG * static_cast<int>(d); e += 32) q_sm[e] = __hip_bfloat16{};
  __syncthreads();
  for (int e = tid; e < nq * QG * static_cast<int>(d); e += 32) {
    const int dd = e % static_cast<int>(d);
    const int tmp = e / static_cast<int>(d);
    const int hh = tmp % QG;
    const int qq = tmp / QG;
    q_sm[(static_cast<size_t>(qq) * QG + hh) * d + dd] =
        query[((t_base + qq) * hq + (h0 + hh)) * d + dd];
  }
  __syncthreads();

  // online state in regs: nrows = nq*QG <= BM*QG
  constexpr int kMaxRows = BM * QG;
  float m_r[kMaxRows];
  float l_r[kMaxRows];
  float o_r[kMaxRows][kEpl];
  const int nrows = nq * QG;
  for (int row = 0; row < kMaxRows; ++row) {
    m_r[row] = -INFINITY;
    l_r[row] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_r[row][i] = 0.f;
  }

  using namespace rocwmma;
  for (int64_t j0 = s_jmin; j0 <= s_jmax; j0 += BN) {
    const int n_k = static_cast<int>((s_jmax - j0 + 1) < BN ? (s_jmax - j0 + 1) : BN);
    // 1-wave CTA: must stride — tid<n_k only covers 32 slots when BN=64 (GPU hang).
    for (int kk = tid; kk < n_k; kk += 32) {
      const int64_t j = j0 + kk;
      s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
      s_off[kk] = j % block_size;
    }
    __syncthreads();
    for (int e = tid; e < BN * static_cast<int>(d); e += 32) kv_sm[e] = __hip_bfloat16{};
    __syncthreads();
    for (int e = tid; e < n_k * static_cast<int>(d); e += 32) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = k_cache[s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd];
    }
    __syncthreads();

    // WMMA QK — single wave, serial 16x16 tiles (safe on gfx1201)
    for (int hh = 0; hh < QG; ++hh) {
      for (int m0 = 0; m0 < BM; m0 += WM) {
        for (int n0 = 0; n0 < BN; n0 += WN) {
          fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> b_frag;
          fragment<accumulator, WM, WN, WK, float> acc;
          fill_fragment(acc, 0.f);
          const bfloat16_t* Qbase =
              reinterpret_cast<const bfloat16_t*>(q_sm + static_cast<size_t>(hh) * d);
          const long lda = static_cast<long>(QG) * static_cast<long>(d);
          const bfloat16_t* Kbase =
              reinterpret_cast<const bfloat16_t*>(kv_sm + static_cast<size_t>(n0) * d);
          for (int k = 0; k < d_expect; k += WK) {
            load_matrix_sync(a_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
            load_matrix_sync(b_frag, Kbase + k, d);
            mma_sync(acc, a_frag, b_frag, acc);
          }
          store_matrix_sync(tile_s, acc, WN, mem_row_major);
          if (lane == 0) {
            for (int rm = 0; rm < WM; ++rm) {
              for (int rn = 0; rn < WN; ++rn) {
                const int qq = m0 + rm;
                const int kk = n0 + rn;
                if (qq >= nq || kk >= n_k) continue;
                float s = tile_s[rm * WN + rn];
                const int64_t j = j0 + kk;
                if (!(j >= s_jmin_q[qq] && j <= s_jmax_q[qq])) s = -INFINITY;
                else s = Softcap(s * scale, softcap);
                s_score[(qq * QG + hh) * BN + kk] = s;
              }
            }
          }
        }
      }
    }
    __syncthreads();

    // V load
    for (int e = tid; e < n_k * static_cast<int>(d); e += 32) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = v_cache[s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd];
    }
    __syncthreads();

    // Online softmax + V; each lane owns d strip via kEpl
    for (int row = 0; row < nrows; ++row) {
#pragma unroll 1
      for (int kk = 0; kk < BN; ++kk) {
        if (kk >= n_k) break;
        const float s = s_score[row * BN + kk];
        if (!(s > -INFINITY)) continue;
        const float m_new = fmaxf(m_r[row], s);
        const float e_old = (m_r[row] > -INFINITY) ? expf(m_r[row] - m_new) : 0.f;
        const float e_s = expf(s - m_new);
        l_r[row] = l_r[row] * e_old + e_s;
#pragma unroll
        for (int i = 0; i < kEpl; ++i) {
          const int dd = lane * kEpl + i;
          const float vv = (dd < d) ? __bfloat162float(kv_sm[static_cast<size_t>(kk) * d + dd]) : 0.f;
          o_r[row][i] = o_r[row][i] * e_old + e_s * vv;
        }
        m_r[row] = m_new;
      }
    }
    __syncthreads();
  }

  for (int row = 0; row < nrows; ++row) {
    const int qq = row / QG;
    const int hh = row - qq * QG;
    const float inv = (l_r[row] > 0.f) ? (1.f / l_r[row]) : 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = o_r[row][i] * inv;
    StoreRowEplBf16<kEpl>(out, ((t_base + qq) * hq + (h0 + hh)) * d, lane, acc);
  }
#endif  // VT_ROCWMMA_OK
}

// Shared-K prefill (scoreless — SGLang direction, gfx1201-safe):
//   BM queries × 1 warp each, QG heads in registers (DecodeGqa-style).
//   K/V streamed in BN tiles into smem once per tile; online softmax in regs.
//   No score matrix, no multi-wave mma. Grid=(ceil(total_q/BM), num_kv_heads[, z]).
//   blockIdx.z tiles large GQA groups (global qg=8 → z=4 with QG=2).
template <int QG, int EPL, int BM, int BN>
__global__ void PagedAttnPrefillSharedK(__hip_bfloat16* out, const __hip_bfloat16* query,
                                        const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                        const int32_t* block_table, const int32_t* seq_lens,
                                        const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                        int64_t num_kv_heads, int64_t d, int64_t block_size,
                                        int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                        int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                        float scale, float softcap, bool causal, int window_left,
                                        int window_right) {
  static_assert(BM >= 1 && BM <= 32, "BM warps");
  static_assert(BN <= 64, "BN smem/regs");
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bdx = static_cast<int>(blockDim.x);
  if (g >= num_kv_heads || d != d_expect || warp >= BM) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t qg_total = hq / num_kv_heads;
  const int64_t h0 = g * qg_total + static_cast<int64_t>(blockIdx.z) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;

  // Per-warp query row
  const int qq = warp;
  const bool live = qq < nq;
  int64_t jmin = seqlen, jmax = -1;
  float q_reg[QG][kEpl];
  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }
  if (live) {
    const int64_t t = t_base + qq;
    const int64_t pos = context + (t - q0);
    jmin = 0;
    if (window_left >= 0) {
      jmin = pos - window_left;
      if (jmin < 0) jmin = 0;
    }
    jmax = causal ? pos : (seqlen - 1);
    if (window_right >= 0) {
      const int64_t jr = pos + window_right;
      if (jr < jmax) jmax = jr;
    }
    if (jmax > seqlen - 1) jmax = seqlen - 1;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh)
      LoadRowEplBf16<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);
  }

  __shared__ int64_t s_jmin, s_jmax;
  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  if (lane == 0) {
    s_jmin_q[qq] = live ? jmin : seqlen;
    s_jmax_q[qq] = live ? jmax : -1;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int i = 0; i < nq; ++i) {
      if (s_jmin_q[i] < tn) tn = s_jmin_q[i];
      if (s_jmax_q[i] > tx) tx = s_jmax_q[i];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  // smem: K[BN*d] + V[BN*d] — fused QK+online V, no score matrix / no SUB sync tax.
  // d=256 BN=32 → 32KB; d=512 BN=16 → 32KB. (BN=64×2 would be 64KB edge.)
  extern __shared__ __hip_bfloat16 dyn_kv[];
  __hip_bfloat16* k_sm = dyn_kv;
  __hip_bfloat16* v_sm = dyn_kv + static_cast<size_t>(BN) * static_cast<size_t>(d_expect);
  __shared__ int64_t s_blk[BN], s_off[BN];

  for (int64_t j0 = s_jmin; j0 <= s_jmax; j0 += BN) {
    const int n_k = static_cast<int>((s_jmax - j0 + 1) < BN ? (s_jmax - j0 + 1) : BN);
    for (int kk = tid; kk < n_k; kk += bdx) {
      const int64_t j = j0 + kk;
      s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
      s_off[kk] = j % block_size;
    }
    __syncthreads();
    // Prefetch K and V tiles together (vectorized uint4 = 8 bf16 when d%8==0).
    if ((static_cast<int>(d) & 7) == 0) {
      const int d8 = static_cast<int>(d) >> 3;
      const int nvec = n_k * d8;
      for (int e = tid; e < nvec; e += bdx) {
        const int kk = e / d8;
        const int v = e - kk * d8;
        const int dd = v << 3;
        const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
        const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
        const size_t dst = (static_cast<size_t>(kk) * static_cast<size_t>(d) + dd) >> 3;
        reinterpret_cast<uint4*>(k_sm)[dst] = *reinterpret_cast<const uint4*>(k_cache + bo);
        reinterpret_cast<uint4*>(v_sm)[dst] = *reinterpret_cast<const uint4*>(v_cache + vo);
      }
    } else {
      for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
        const int kk = e / static_cast<int>(d);
        const int dd = e - kk * static_cast<int>(d);
        const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
        const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
        k_sm[static_cast<size_t>(kk) * d + dd] = k_cache[bo];
        v_sm[static_cast<size_t>(kk) * d + dd] = v_cache[vo];
      }
    }
    __syncthreads();

    // Fused QK + online V (scoreless, one pass — no s_h[BN] regs).
    if (live) {
#pragma unroll 1
      for (int kk = 0; kk < n_k; ++kk) {
        const int64_t j = j0 + kk;
        if (!(j >= jmin && j <= jmax)) continue;
        float k_reg[kEpl];
        LoadRowEplBf16<kEpl>(k_sm, static_cast<int64_t>(kk) * d, lane, k_reg);
        float v_reg[kEpl];
        LoadRowEplBf16<kEpl>(v_sm, static_cast<int64_t>(kk) * d, lane, v_reg);
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) {
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
          const float s = Softcap(__shfl(dot, 0) * scale, softcap);
          const float m_new = fmaxf(m[hh], s);
          const float corr = (m[hh] > -INFINITY) ? FastExp(m[hh] - m_new) : 0.f;
          const float pw = FastExp(s - m_new);
#pragma unroll
          for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
          lsum[hh] = lsum[hh] * corr + pw;
          m[hh] = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (live) {
    const int64_t t = t_base + qq;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float inv = (lsum[hh] > 0.f) ? (1.f / lsum[hh]) : 0.f;
      float acc[kEpl];
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] = o_reg[hh][i] * inv;
      StoreRowEplBf16<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
    }
  }
}

// SharedK + rocWMMA QK (score matrix only for one BN tile; online softmax/V in regs).
// Dyn smem: Q[BM*QG*d] + KV[BN*d]. Static: s_sc[QG*BM*BN].
// d=256 BM16 BN64 ≈56KB; d=512 BM16 BN16 ≈50KB. Env: VT_ATTN_PREFILL_SHAREDK_WMMA=1.
// kQGlobal=true: Q from global (lda=hq*d), dyn smem = K+V only — enables d=512 WMMA under 64KB LDS.
// kQGlobal=false: Q tile in smem (d=256 path). Partial tiles (nq<BM) use scalar QK when kQGlobal.
template <int QG, int EPL, int BM, int BN, bool kQGlobal>
__global__ void PagedAttnPrefillSharedKWmma(__hip_bfloat16* out, const __hip_bfloat16* query,
                                            const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                            const int32_t* block_table, const int32_t* seq_lens,
                                            const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                            int64_t num_kv_heads, int64_t d, int64_t block_size,
                                            int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                            int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                            float scale, float softcap, bool causal, int window_left,
                                            int window_right) {
#if !defined(VT_ROCWMMA_OK)
  (void)out;
  (void)query;
  (void)k_cache;
  (void)v_cache;
  (void)block_table;
  (void)seq_lens;
  (void)query_start_loc;
  (void)num_reqs;
  (void)hq;
  (void)num_kv_heads;
  (void)d;
  (void)block_size;
  (void)bt_row;
  (void)bt_col;
  (void)kc_blk;
  (void)kc_pg;
  (void)kc_hd;
  (void)vc_blk;
  (void)vc_pg;
  (void)vc_hd;
  (void)scale;
  (void)softcap;
  (void)causal;
  (void)window_left;
  (void)window_right;
#else
  static_assert(BM % 16 == 0 && BN % 16 == 0, "WMMA tile");
  static_assert(BM >= 16 && BM <= 32 && BN <= 64, "bounds");
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  constexpr int WM = 16, WN = 16, WK = 16;
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bdx = static_cast<int>(blockDim.x);
  const int nwarps = bdx >> 5;
  if (g >= num_kv_heads || d != d_expect) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t qg_total = hq / num_kv_heads;
  const int64_t h0 = g * qg_total + static_cast<int64_t>(blockIdx.z) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;

  const int qq = warp;  // may be >= BM if block oversized; only BM warps own rows
  const bool live = (qq < nq) && (qq < BM);

  __shared__ int64_t s_jmin, s_jmax;
  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  if (lane == 0 && qq < BM) {
    if (live) {
      const int64_t t = t_base + qq;
      const int64_t pos = context + (t - q0);
      int64_t jn = 0;
      if (window_left >= 0) {
        jn = pos - window_left;
        if (jn < 0) jn = 0;
      }
      int64_t jx = causal ? pos : (seqlen - 1);
      if (window_right >= 0) {
        const int64_t jr = pos + window_right;
        if (jr < jx) jx = jr;
      }
      if (jx > seqlen - 1) jx = seqlen - 1;
      s_jmin_q[qq] = jn;
      s_jmax_q[qq] = jx;
    } else {
      s_jmin_q[qq] = seqlen;
      s_jmax_q[qq] = -1;
    }
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int i = 0; i < nq; ++i) {
      if (s_jmin_q[i] < tn) tn = s_jmin_q[i];
      if (s_jmax_q[i] > tx) tx = s_jmax_q[i];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  extern __shared__ __hip_bfloat16 dyn_bf[];
  __hip_bfloat16* q_sm = nullptr;
  __hip_bfloat16* k_sm = nullptr;
  __hip_bfloat16* v_sm = nullptr;
  if constexpr (kQGlobal) {
    // Dyn = K+V only (d=512: BN16 → 32KB).
    k_sm = dyn_bf;
    v_sm = k_sm + static_cast<size_t>(BN) * static_cast<size_t>(d_expect);
  } else {
    q_sm = dyn_bf;
    k_sm = q_sm + static_cast<size_t>(BM) * QG * static_cast<size_t>(d_expect);
    v_sm = k_sm + static_cast<size_t>(BN) * static_cast<size_t>(d_expect);
  }
  __shared__ int64_t s_blk[BN], s_off[BN];
  __shared__ float s_sc[QG * BM * BN];

  // Q: smem tile (d=256) or per-warp regs for scalar tail (d=512 global).
  float q_reg[QG][kEpl];
  if constexpr (!kQGlobal) {
    for (int e = tid; e < BM * QG * static_cast<int>(d); e += bdx) q_sm[e] = __hip_bfloat16{};
    __syncthreads();
    for (int e = tid; e < nq * QG * static_cast<int>(d); e += bdx) {
      const int dd = e % static_cast<int>(d);
      const int tmp = e / static_cast<int>(d);
      const int hh = tmp % QG;
      const int qi = tmp / QG;
      q_sm[(static_cast<size_t>(qi) * QG + hh) * d + dd] =
          query[((t_base + qi) * hq + (h0 + hh)) * d + dd];
    }
    __syncthreads();
  } else if (live) {
    const int64_t t = t_base + qq;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh)
      LoadRowEplBf16<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);
  }

  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }

  using namespace rocwmma;
  // Full BM tiles can WMMA from global Q without OOB; partial last tile uses scalar QK.
  const bool use_wmma_qk = !kQGlobal || (nq == BM);
  for (int64_t j0 = s_jmin; j0 <= s_jmax; j0 += BN) {
    const int n_k = static_cast<int>((s_jmax - j0 + 1) < BN ? (s_jmax - j0 + 1) : BN);
    for (int kk = tid; kk < n_k; kk += bdx) {
      const int64_t j = j0 + kk;
      s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
      s_off[kk] = j % block_size;
    }
    __syncthreads();
    // Prefetch K+V together (vectorized when d%8==0).
    if ((static_cast<int>(d) & 7) == 0) {
      const int d8 = static_cast<int>(d) >> 3;
      const int nvec_all = BN * d8;
      for (int e = tid; e < nvec_all; e += bdx) {
        reinterpret_cast<uint4*>(k_sm)[static_cast<size_t>(e)] = uint4{0, 0, 0, 0};
        reinterpret_cast<uint4*>(v_sm)[static_cast<size_t>(e)] = uint4{0, 0, 0, 0};
      }
      __syncthreads();
      const int nvec = n_k * d8;
      for (int e = tid; e < nvec; e += bdx) {
        const int kk = e / d8;
        const int v = e - kk * d8;
        const int dd = v << 3;
        const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
        const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
        const size_t dst = static_cast<size_t>(e);
        reinterpret_cast<uint4*>(k_sm)[dst] = *reinterpret_cast<const uint4*>(k_cache + bo);
        reinterpret_cast<uint4*>(v_sm)[dst] = *reinterpret_cast<const uint4*>(v_cache + vo);
      }
    } else {
      for (int e = tid; e < BN * static_cast<int>(d); e += bdx) {
        k_sm[e] = __hip_bfloat16{};
        v_sm[e] = __hip_bfloat16{};
      }
      __syncthreads();
      for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
        const int kk = e / static_cast<int>(d);
        const int dd = e - kk * static_cast<int>(d);
        k_sm[e] = k_cache[s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd];
        v_sm[e] = v_cache[s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd];
      }
    }
    __syncthreads();

    if (use_wmma_qk) {
      // WMMA QK → s_sc[hh][qq][kk]
      constexpr int tiles_m = BM / WM;
      constexpr int tiles_n = BN / WN;
      constexpr int ntiles_h = tiles_m * tiles_n;
      constexpr int ntiles = ntiles_h * QG;
      const int step = nwarps;
      for (int t = warp; t < ntiles; t += step) {
        const int hh = t / ntiles_h;
        const int t2 = t - hh * ntiles_h;
        const int tm = t2 / tiles_n;
        const int tn = t2 - tm * tiles_n;
        const int m0 = tm * WM;
        const int n0 = tn * WN;
        fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> a_frag;
        fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> b_frag;
        fragment<accumulator, WM, WN, WK, float> acc;
        fill_fragment(acc, 0.f);
        const bfloat16_t* Kbase =
            reinterpret_cast<const bfloat16_t*>(k_sm + static_cast<size_t>(n0) * d);
        if constexpr (kQGlobal) {
          // query layout [t][hq][d]; fixed head h0+hh, row stride = hq*d
          const bfloat16_t* Qbase = reinterpret_cast<const bfloat16_t*>(
              query + ((t_base + m0) * hq + (h0 + hh)) * d);
          const long lda = static_cast<long>(hq) * static_cast<long>(d);
          for (int k = 0; k < d_expect; k += WK) {
            load_matrix_sync(a_frag, Qbase + k, lda);
            load_matrix_sync(b_frag, Kbase + k, d);
            mma_sync(acc, a_frag, b_frag, acc);
          }
        } else {
          const bfloat16_t* Qbase =
              reinterpret_cast<const bfloat16_t*>(q_sm + static_cast<size_t>(hh) * d);
          const long lda = static_cast<long>(QG) * static_cast<long>(d);
          for (int k = 0; k < d_expect; k += WK) {
            load_matrix_sync(a_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
            load_matrix_sync(b_frag, Kbase + k, d);
            mma_sync(acc, a_frag, b_frag, acc);
          }
        }
        float* sc_base = s_sc + static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(m0) * BN + n0;
        store_matrix_sync(sc_base, acc, BN, mem_row_major);
      }
      __syncthreads();
    } else if (live) {
      // Partial tile + global Q: scalar QK into s_sc for this row (no OOB WMMA).
#pragma unroll 1
      for (int kk = 0; kk < n_k; ++kk) {
        float k_reg[kEpl];
        LoadRowEplBf16<kEpl>(k_sm, static_cast<int64_t>(kk) * d, lane, k_reg);
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) {
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
          if (lane == 0)
            s_sc[static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN + kk] =
                __shfl(dot, 0);
        }
      }
      __syncthreads();
    } else {
      __syncthreads();
    }

    // Online softmax + V from v_sm (no K reload, no V global round-trip).
    if (live) {
      const int64_t jmin = s_jmin_q[qq];
      const int64_t jmax = s_jmax_q[qq];
#pragma unroll 1
      for (int kk = 0; kk < n_k; ++kk) {
        const int64_t j = j0 + kk;
        float v_reg[kEpl];
        bool loaded = false;
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) {
          float s = -INFINITY;
          if (j >= jmin && j <= jmax) {
            s = Softcap(s_sc[static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN + kk] *
                            scale,
                        softcap);
          }
          if (!(s > -INFINITY)) continue;
          if (!loaded) {
            LoadRowEplBf16<kEpl>(v_sm, static_cast<int64_t>(kk) * d, lane, v_reg);
            loaded = true;
          }
          const float m_new = fmaxf(m[hh], s);
          const float corr = (m[hh] > -INFINITY) ? FastExp(m[hh] - m_new) : 0.f;
          const float pw = FastExp(s - m_new);
#pragma unroll
          for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
          lsum[hh] = lsum[hh] * corr + pw;
          m[hh] = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (live) {
    const int64_t t = t_base + qq;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float inv = (lsum[hh] > 0.f) ? (1.f / lsum[hh]) : 0.f;
      float acc[kEpl];
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] = o_reg[hh][i] * inv;
      StoreRowEplBf16<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
    }
  }
#endif  // VT_ROCWMMA_OK
}

}  // namespace

void PagedAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args) {
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto,
           "rocm paged_attention: fp8 KV not implemented");
  VT_CHECK(args.scale > 0.f, "rocm paged_attention: scale must be > 0");
  if (CpuRefEnabled()) {
    PagedAttnCpuRef(q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc,
                    args);
    return;
  }
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  VT_CHECK(hq % num_kv_heads == 0, "rocm paged_attention: GQA ratio");
  if (total_q == 0) return;

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  const int window_left =
      args.window_size.has_value() ? static_cast<int>(args.window_size->left) : -1;
  const int window_right =
      args.window_size.has_value() ? static_cast<int>(args.window_size->right) : -1;
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  static const bool decode_opt = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_OPT");
    if (e && e[0] == '0') return false;
    return true;
  }();
  static const bool decode_gqa = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_GQA");
    if (e && e[0] == '0') return false;
    return true;  // default ON when decode_opt
  }();

  // head_dim-128 arm (#382) -- the Qwen3-dense / Llama / Mistral head size.
  // Same kernels as d==256/512, EPL=4 instead of 8/16; before this, bf16 decode
  // at d==128 fell all the way to the generic PagedAttnOnline (#488 measured
  // that fallback at 41.1us/call against vLLM's 5.10us on gfx1200).
  //
  // DEFAULT OFF, opt in with VT_ATTN_DECODE_D128=1 -- the SAME flag, default and
  // reason as the merged CUDA arm (cuda_paged_attn.cu, DecodeD128Enabled, #382 /
  // PR #425). It is correctness-complete but NOT byte-exact against the kernel it
  // replaces: warp-strided online softmax reduces the KV sequence in a different
  // ORDER than PagedAttnOnline's per-tile loop, so a greedy anchor can move at an
  // exact bf16 tie. Shipping OFF keeps every existing golden byte-identical. The
  // flip owes the near-tie razor + distributional gate + regen under the ratified-
  // tie rule, on both backends, and is a separate change.
  static const bool decode_d128 = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_D128");
    return e != nullptr && e[0] == '1';
  }();
  const bool bf16_decode_opt =
      decode_opt && ((d == 128 && decode_d128) || d == 256 || d == 512) &&
      query.dtype == DType::kBF16 &&
      k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16 && out.dtype == DType::kBF16 &&
      kc_hd % 8 == 0;

  if (bf16_decode_opt) {
    const int64_t qg = hq / num_kv_heads;
    // GQA fuse: sliding QG=2. Global QG=8: tile as DecodeGqa QG=2 × z=4
    // (proven kernel; halves K/V reloads vs DecodeOpt's 8×). QG=4/8 CTAs lost A/B.
    if (decode_gqa &&
        ((qg == 2 && (d == 128 || d == 256 || d == 512)) || (qg == 8 && d == 512))) {
      // Prefill SharedK (scoreless). Default ON for long single-req prefill (lab win
      // ~1.09× @11k vs DecodeGqa-z4). Set VT_ATTN_PREFILL_FLASH_SHAREDK=0 to disable.
      // SHAREDK_WMMA=1: rocWMMA QK + online V (lab; keep only if beats scalar SharedK).
      static const bool prefill_sharedk = [] {
        const char* e = std::getenv("VT_ATTN_PREFILL_FLASH_SHAREDK");
        if (!e) return true;  // default ON
        return e[0] != '0';
      }();
      static const bool sharedk_wmma = [] {
        // Default ON: d=256 Q-smem WMMA only. d=512 stays scalar SharedK
        // until a separately gated kernel repair (shipping f58b spills).
        // =0 forces scalar SharedK for A/B.
        const char* e = std::getenv("VT_ATTN_PREFILL_SHAREDK_WMMA");
        if (!e) return true;
        return e[0] != '0';
      }();
      if (prefill_sharedk && total_q >= 64 && num_reqs == 1) {
        auto smem_kv = [&](int bn) -> size_t {
          // K[bn*d] + V[bn*d] fused scoreless
          return sizeof(__hip_bfloat16) * static_cast<size_t>(2 * bn) * static_cast<size_t>(d);
        };
        auto smem_qkv = [&](int bm, int bn) -> size_t {
          // Q[bm*QG*d] + K[bn*d] + V[bn*d]  (QG=2 packed as bm*2*d)
          return sizeof(__hip_bfloat16) *
                 (static_cast<size_t>(bm) * 2 * static_cast<size_t>(d) +
                  static_cast<size_t>(2 * bn) * static_cast<size_t>(d));
        };
        const unsigned z_groups = (qg == 8 && d == 512) ? 4u : 1u;
        // Host must not launch WMMA stubs on non-gfx1200/1201 devices
        // (kernel body is empty there → silent garbage out). Runtime prefix
        // match on hipDeviceProp_t.gcnArchName; VT_ROCWMMA_OK is device-only.
        if (sharedk_wmma && PrefillSharedKWmmaHostOk()) {
          if (d == 256) {
            // Q16K+K16K+V16K=48KB dyn + s_sc~4KB
            constexpr int BM = 16, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
            dim3 block(static_cast<unsigned>(BM * 32));
            PagedAttnPrefillSharedKWmma<2, 8, BM, BN, false>
                <<<grid, block, smem_qkv(BM, BN), st>>>(
                    out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(),
                    k_cache.Ptr<__hip_bfloat16>(), v_cache.Ptr<__hip_bfloat16>(),
                    block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                    query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size,
                    bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale,
                    args.logits_soft_cap, args.causal, window_left, window_right);
            Check(hipGetLastError(), "paged_attention prefill_sharedk_wmma d256 launch");
            return;
          }
        }
        if (d == 256) {
          constexpr int BM = 32, BN = 32;  // K+V 32KB; BM32 ~+2% vs BM16 @11k
          const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
          dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
          dim3 block(static_cast<unsigned>(BM * 32));
          PagedAttnPrefillSharedK<2, 8, BM, BN><<<grid, block, smem_kv(BN), st>>>(
              out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
              v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
              query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
              bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
              args.causal, window_left, window_right);
        } else if (d == 512) {
          constexpr int BM = 16, BN = 16;  // K+V dual 32KB; was BN32 K-only overwrite
          const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
          dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
          dim3 block(static_cast<unsigned>(BM * 32));
          PagedAttnPrefillSharedK<2, 16, BM, BN><<<grid, block, smem_kv(BN), st>>>(
              out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
              v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
              query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
              bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
              args.causal, window_left, window_right);
        } else {
          goto flash_fallback;
        }
        Check(hipGetLastError(), "paged_attention prefill_sharedk launch");
        return;
      }
      // Prefill flash (opt-in). Default OFF — DecodeGqa remains prod.
      // FLASH=1 → PrefillFlashTile + concurrent multi-wave WMMA QK (lab 2026-08-10).
      // FLASH_WMMA=1 → 1-wave/CTA WMMA only.
      // Flash tiles are QG=2-shaped — sliding only (quality fail on global 2026-08-10).
      static const bool prefill_flash = [] {
        const char* e = std::getenv("VT_ATTN_PREFILL_FLASH");
        return e && e[0] == '1';
      }();
      if (prefill_flash && qg == 2 && total_q >= 64 && num_reqs == 1) {
        static const bool flash_wmma1 = [] {
          const char* e = std::getenv("VT_ATTN_PREFILL_FLASH_WMMA");
          return e && e[0] == '1';
        }();
        auto smem_qk = [&](int bm, int bn) -> size_t {
          return sizeof(__hip_bfloat16) *
                 (static_cast<size_t>(bm) * 2 * static_cast<size_t>(d) +
                  static_cast<size_t>(bn) * static_cast<size_t>(d));
        };
        if (flash_wmma1) {
          dim3 block(32);
          if (d == 256) {
            constexpr int BM = 16, BN = 64;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillWmmaWave<2, 8, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else if (d == 512) {
            constexpr int BM = 16, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillWmmaWave<2, 16, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else {
            goto flash_fallback;
          }
          Check(hipGetLastError(), "paged_attention prefill_wmma_wave launch");
          return;
        }
        // Default FLASH=1: concurrent WMMA direct-to-s_sc; BM=32 packs more Q/CTA
        {
          dim3 block(256);
          if (d == 256) {
            constexpr int BM = 32, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillFlashTile<2, 8, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else if (d == 512) {
            constexpr int BM = 16, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillFlashTile<2, 16, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else {
            goto flash_fallback;
          }
          Check(hipGetLastError(), "paged_attention prefill_flash_wmma_mw launch");
          return;
        }
      }
      flash_fallback:
      // DecodeGqa QG=2: sliding z=1; global qg=8 → z=4 head tiles (same kernel/regs).
      const int nwarps = kDecWarps;
      const size_t smem =
          sizeof(float) *
          (static_cast<size_t>(nwarps) * 2 * static_cast<size_t>(d) +
           2 * static_cast<size_t>(nwarps) * 2);
      const unsigned z_groups = (qg == 8 && d == 512) ? 4u : 1u;
      dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(num_kv_heads), z_groups);
      dim3 block(static_cast<unsigned>(nwarps * 32));
      if (d == 128) {
        PagedAttnDecodeGqaBf16<2, 4, 8><<<grid, block, smem, st>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right);
      } else if (d == 256) {
        PagedAttnDecodeGqaBf16<2, 8, 8><<<grid, block, smem, st>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right);
      } else {
        PagedAttnDecodeGqaBf16<2, 16, 8><<<grid, block, smem, st>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right);
      }
      Check(hipGetLastError(), "paged_attention decode_gqa launch");
      return;
    }
    const size_t smem =
        sizeof(float) * (static_cast<size_t>(kDecWarps) * static_cast<size_t>(d) +
                         2 * static_cast<size_t>(kDecWarps));
    dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));
    if (d == 128) {
      PagedAttnDecodeOptBf16T<4><<<grid, kDecWarps * 32, smem, st>>>(
          out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
          v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
          query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col,
          kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
          window_left, window_right);
    } else if (d == 256) {
      PagedAttnDecodeOptBf16T<8><<<grid, kDecWarps * 32, smem, st>>>(
          out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
          v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
          query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col,
          kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
          window_left, window_right);
    } else {
      PagedAttnDecodeOptBf16T<16><<<grid, kDecWarps * 32, smem, st>>>(
          out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
          v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
          query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col,
          kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
          window_left, window_right);
    }
    Check(hipGetLastError(), "paged_attention decode_opt launch");
    return;
  }

  // threads: reduce over head_dim; need power-of-2 <= 256 and <= d*something
  unsigned threads = 64;
  while (threads < 256u && threads < static_cast<unsigned>(d)) threads *= 2;
  if (threads > 256u) threads = 256;
  const size_t smem = sizeof(float) * (static_cast<size_t>(d) + threads);
  dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));

  auto launch = [&](auto q_tag, auto kv_tag, auto o_tag) {
    using TQ = decltype(q_tag);
    using TKV = decltype(kv_tag);
    using TO = decltype(o_tag);
    PagedAttnOnline<TQ, TKV, TO><<<grid, threads, smem, st>>>(
        out.Ptr<TO>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
        block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(),
        num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,
        vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal, window_left,
        window_right);
  };

  if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
      out.dtype == DType::kBF16) {
    launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
  } else if (query.dtype == DType::kF32 && k_cache.dtype == DType::kF32 &&
             out.dtype == DType::kF32) {
    launch(float{}, float{}, float{});
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kF32) {
    launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kF32 &&
             out.dtype == DType::kBF16) {
    launch(__hip_bfloat16{}, float{}, __hip_bfloat16{});
  } else if (query.dtype == DType::kF32 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kF32) {
    launch(float{}, __hip_bfloat16{}, float{});
  } else {
    VT_CHECK(false, "rocm paged_attention dtype");
  }
  Check(hipGetLastError(), "paged_attention launch");
}

}  // namespace vt::rocm
