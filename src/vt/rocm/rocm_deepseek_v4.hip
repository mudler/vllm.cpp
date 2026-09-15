// DeepSeek-V4-Flash — the ROCm/HIP MHC kernel family.  A 1:1 port of the CUDA
// launchers in cuda_deepseek_v4.cu (lines 1077-1537, env-var helpers 1440-1488,
// and the device kernels 143-587).  HIP is source-compatible with CUDA for
// __global__/__device__/__shared__/<<<>>>/__syncthreads, so the kernel bodies
// are identical; only the runtime API (cuda*→hip*) and the include change.
//
// Scope: O34 registers kDeepseekV4Mhc on kROCM.  The DSA/Compressor/MoE
// families stay CUDA-only until their owed rows land.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vt/ops.h"

namespace vllm::deepseek_v4 {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;
using vt::Tensor;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("vt rocm deepseek_v4: ") + what + ": " +
                             hipGetErrorString(e));
}
hipStream_t AsStream(Queue& q) { return static_cast<hipStream_t>(q.handle); }

struct Dev {
  void* p = nullptr;
  explicit Dev(size_t bytes) { Check(hipMalloc(&p, bytes ? bytes : 1), "hipMalloc"); }
  ~Dev() {
    if (p) (void)hipFree(p);
  }
  Dev(Dev&& o) noexcept : p(o.p) { o.p = nullptr; }
  Dev& operator=(Dev&&) = delete;
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

template <class T>
Dev Upload(const std::vector<T>& v, hipStream_t s) {
  Dev d(v.size() * sizeof(T));
  if (!v.empty())
    Check(hipMemcpyAsync(d.p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice, s),
          "H2D");
  return d;
}
template <class T>
void Download(std::vector<T>& v, void* dp, hipStream_t s) {
  if (!v.empty())
    Check(hipMemcpyAsync(v.data(), dp, v.size() * sizeof(T), hipMemcpyDeviceToHost, s),
          "D2H");
}
// Raw-pointer variants for the in-place launchers on discrete GPUs: upload host
// inputs to device memory, launch, then download outputs back to the host
// pointers.  On CUDA unified memory (GB10) the kernel can dereference host
// pointers directly, but ROCm discrete GPUs cannot, so the in-place glue must
// round-trip through device allocations just like the vector launchers above.
static Dev UploadRaw(const void* host, size_t bytes, hipStream_t s) {
  Dev d(bytes);
  if (bytes) Check(hipMemcpyAsync(d.p, host, bytes, hipMemcpyHostToDevice, s), "H2D raw");
  return d;
}
static void DownloadRaw(void* host, const void* dev, size_t bytes, hipStream_t s) {
  if (bytes) Check(hipMemcpyAsync(host, dev, bytes, hipMemcpyDeviceToHost, s), "D2H raw");
}

// ── device math helpers (bit-faithful to the host references) ─────────────────
__device__ inline float Sig(float x) { return 1.0f / (1.0f + expf(-x)); }

// ============================================================================
// (1) MHC family
// ============================================================================
__device__ void SinkhornInplace(const float* logits, float* m, int hc, int iters, float eps) {
  for (int j = 0; j < hc; ++j) {
    float rmax = logits[j * hc];
    for (int k = 1; k < hc; ++k) rmax = fmaxf(rmax, logits[j * hc + k]);
    float rsum = 0.0f;
    for (int k = 0; k < hc; ++k) {
      const float e = expf(logits[j * hc + k] - rmax);
      m[j * hc + k] = e;
      rsum += e;
    }
    for (int k = 0; k < hc; ++k) m[j * hc + k] = m[j * hc + k] / rsum + eps;
  }
  for (int k = 0; k < hc; ++k) {
    float c = 0.0f;
    for (int j = 0; j < hc; ++j) c += m[j * hc + k];
    const float den = c + eps;
    for (int j = 0; j < hc; ++j) m[j * hc + k] /= den;
  }
  for (int it = 0; it < iters - 1; ++it) {
    for (int j = 0; j < hc; ++j) {
      float r = 0.0f;
      for (int k = 0; k < hc; ++k) r += m[j * hc + k];
      const float den = r + eps;
      for (int k = 0; k < hc; ++k) m[j * hc + k] /= den;
    }
    for (int k = 0; k < hc; ++k) {
      float c = 0.0f;
      for (int j = 0; j < hc; ++j) c += m[j * hc + k];
      const float den = c + eps;
      for (int j = 0; j < hc; ++j) m[j * hc + k] /= den;
    }
  }
}

__global__ void SinkhornKernel(const float* logits, float* out, int hc, int iters, float eps) {
  float m[256];
  SinkhornInplace(logits, m, hc, iters, eps);
  for (int i = 0; i < hc * hc; ++i) out[i] = m[i];
}

__device__ inline void SinkhornInplace4(const float* __restrict__ logits, float* m, int iters,
                                        float eps) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float rmax = logits[j * 4];
#pragma unroll
    for (int k = 1; k < 4; ++k) rmax = fmaxf(rmax, logits[j * 4 + k]);
    float rsum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float e = expf(logits[j * 4 + k] - rmax);
      m[j * 4 + k] = e;
      rsum += e;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) m[j * 4 + k] = m[j * 4 + k] / rsum + eps;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float c = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) c += m[j * 4 + k];
    const float den = c + eps;
#pragma unroll
    for (int j = 0; j < 4; ++j) m[j * 4 + k] /= den;
  }
  for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float r = 0.0f;
#pragma unroll
      for (int k = 0; k < 4; ++k) r += m[j * 4 + k];
      const float den = r + eps;
#pragma unroll
      for (int k = 0; k < 4; ++k) m[j * 4 + k] /= den;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float c = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) c += m[j * 4 + k];
      const float den = c + eps;
#pragma unroll
      for (int j = 0; j < 4; ++j) m[j * 4 + k] /= den;
    }
  }
}

__global__ void MhcPreKernel(const float* residual, const float* fn, const float* scale,
                             const float* base, int hc, int hidden, float rms_eps,
                             float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult,
                             int iters, const float* norm_weight, int has_norm, float norm_eps,
                             float* mixes, float* pre_out, float* post_out, float* comb_out,
                             float* layer_out) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  double sqrsum = 0.0;
  for (int i = 0; i < flat; ++i) {
    const double r = residual[i];
    sqrsum += r * r;
  }
  for (int o = 0; o < hc3; ++o) {
    float acc = 0.0f;
    const int frow = o * flat;
    for (int i = 0; i < flat; ++i) acc += residual[i] * fn[frow + i];
    mixes[o] = acc;
  }
  const float rms =
      1.0f / sqrtf(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);
  for (int o = 0; o < hc3; ++o) mixes[o] *= rms;
  for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
  for (int j = 0; j < hc; ++j)
    post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
  float cl[256];
  for (int j = 0; j < hc; ++j)
    for (int k = 0; k < hc; ++k) {
      const int idx = j * hc + k;
      cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
    }
  float m[256];
  SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
  for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  for (int h = 0; h < hidden; ++h) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  if (has_norm) {
    double ss = 0.0;
    for (int h = 0; h < hidden; ++h) {
      const double v = layer_out[h];
      ss += v * v;
    }
    const float r =
        1.0f / sqrtf(static_cast<float>(ss / static_cast<double>(hidden)) + norm_eps);
    for (int h = 0; h < hidden; ++h) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

__global__ void MhcPreDotsKernel(const float* __restrict__ residual,
                                 const float* __restrict__ fn, int flat, float* __restrict__ mixes) {
  const int o = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ double red[];
  const int frow = o * flat;
  double la = 0.0;
  for (int i = tid; i < flat; i += nt) la += static_cast<double>(residual[i]) * fn[frow + i];
  red[tid] = la;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) mixes[o] = static_cast<float>(red[0]);
}

__global__ void MhcPreFinishKernel(const float* __restrict__ residual,
                                   const float* __restrict__ scale, const float* __restrict__ base,
                                   int hc, int hidden, float rms_eps, float hc_pre_eps,
                                   float hc_sinkhorn_eps, float hc_post_mult, int iters,
                                   const float* __restrict__ norm_weight, int has_norm,
                                   float norm_eps, float* mixes, float* pre_out, float* post_out,
                                   float* comb_out, float* layer_out) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ double red[];
  auto block_reduce = [&](double v) -> double {
    red[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const double r = red[0];
    __syncthreads();
    return r;
  };
  double ls = 0.0;
  for (int i = tid; i < flat; i += nt) { const double r = residual[i]; ls += r * r; }
  const double sqrsum = block_reduce(ls);
  const float rms = 1.0f / sqrtf(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);
  for (int o = tid; o < hc3; o += nt) mixes[o] = mixes[o] * rms;
  __syncthreads();
  if (tid == 0) {
    for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
    for (int j = 0; j < hc; ++j)
      post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
    float cl[256], m[256];
    for (int j = 0; j < hc; ++j)
      for (int k = 0; k < hc; ++k) {
        const int idx = j * hc + k;
        cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
      }
    SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
    for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  }
  __syncthreads();
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  __syncthreads();
  if (has_norm) {
    double ss = 0.0;
    for (int h = tid; h < hidden; h += nt) { const double v = layer_out[h]; ss += v * v; }
    const double ssr = block_reduce(ss);
    const float r =
        1.0f / sqrtf(static_cast<float>(ssr / static_cast<double>(hidden)) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

__global__ void MhcPreDotsFloatKernel(const float* __restrict__ residual,
                                      const float* __restrict__ fn, int flat,
                                      float* __restrict__ mixes, int fold_sqrsum, int hc3) {
  const int o = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float redf[];
  const int frow = o * flat;
  float la = 0.0f, sq = 0.0f;
  for (int i = tid; i < flat; i += nt) {
    const float r = residual[i];
    la += r * fn[frow + i];
    if (fold_sqrsum && o == 0) sq += r * r;
  }
  redf[tid] = la;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
  if (tid == 0) mixes[o] = redf[0];
  if (fold_sqrsum && o == 0) {
    __syncthreads();
    redf[tid] = sq;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    if (tid == 0) mixes[hc3] = redf[0];
  }
}

__global__ void MhcPreFinishFloatKernel(const float* __restrict__ residual,
                                        const float* __restrict__ scale,
                                        const float* __restrict__ base, int hc, int hidden,
                                        float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                        float hc_post_mult, int iters,
                                        const float* __restrict__ norm_weight, int has_norm,
                                        float norm_eps, float* mixes, float* pre_out,
                                        float* post_out, float* comb_out, float* layer_out,
                                        int fold_sqrsum) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float redf[];
  auto block_reduce = [&](float v) -> float {
    redf[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    const float r = redf[0];
    __syncthreads();
    return r;
  };
  float sqrsum;
  if (fold_sqrsum) {
    sqrsum = mixes[hc3];
  } else {
    float ls = 0.0f;
    for (int i = tid; i < flat; i += nt) { const float r = residual[i]; ls += r * r; }
    sqrsum = block_reduce(ls);
  }
  const float rms = rsqrtf(sqrsum / static_cast<float>(flat) + rms_eps);
  for (int o = tid; o < hc3; o += nt) mixes[o] = mixes[o] * rms;
  __syncthreads();
  if (tid == 0) {
    for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
    for (int j = 0; j < hc; ++j)
      post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
    float cl[256], m[256];
    for (int j = 0; j < hc; ++j)
      for (int k = 0; k < hc; ++k) {
        const int idx = j * hc + k;
        cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
      }
    SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
    for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  }
  __syncthreads();
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  __syncthreads();
  if (has_norm) {
    float ss = 0.0f;
    for (int h = tid; h < hidden; h += nt) { const float v = layer_out[h]; ss += v * v; }
    const float ssr = block_reduce(ss);
    const float r = rsqrtf(ssr / static_cast<float>(hidden) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

__global__ void MhcPreFinishFloatKernel4(const float* __restrict__ residual,
                                         const float* __restrict__ scale,
                                         const float* __restrict__ base, int hc, int hidden,
                                         float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                         float hc_post_mult, int iters,
                                         const float* __restrict__ norm_weight, int has_norm,
                                         float norm_eps, float* mixes, float* pre_out,
                                         float* post_out, float* comb_out, float* layer_out,
                                         int fold_sqrsum) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float redf[];
  auto block_reduce = [&](float v) -> float {
    redf[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) redf[tid] += redf[tid + s]; __syncthreads(); }
    const float r = redf[0];
    __syncthreads();
    return r;
  };
  float sqrsum;
  if (fold_sqrsum) {
    sqrsum = mixes[hc3];
  } else {
    float ls = 0.0f;
    for (int i = tid; i < flat; i += nt) { const float r = residual[i]; ls += r * r; }
    sqrsum = block_reduce(ls);
  }
  const float rms = rsqrtf(sqrsum / static_cast<float>(flat) + rms_eps);
  for (int o = tid; o < hc3; o += nt) mixes[o] = mixes[o] * rms;
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      post_out[j] = Sig(mixes[4 + j] * scale[1] + base[4 + j]) * hc_post_mult;
    float cl[16], m[16];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int idx = j * 4 + k;
        cl[idx] = mixes[8 + idx] * scale[2] + base[8 + idx];
      }
    SinkhornInplace4(cl, m, iters, hc_sinkhorn_eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) comb_out[i] = m[i];
  }
  __syncthreads();
  float ss = 0.0f;
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
    ss += acc * acc;
  }
  if (has_norm) {
    const float ssr = block_reduce(ss);
    const float r = rsqrtf(ssr / static_cast<float>(hidden) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

__global__ void MhcPostKernel(const float* x, const float* residual, const float* post_mix,
                              const float* comb, int hc, int hidden, float* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * hidden) return;
  const int j = idx / hidden, h = idx % hidden;
  float mixed = 0.0f;
  for (int i = 0; i < hc; ++i) mixed += comb[i * hc + j] * residual[i * hidden + h];
  out[idx] = mixed + post_mix[j] * x[h];
}

__global__ void HcHeadKernel(const float* __restrict__ x, const float* __restrict__ fn, float scale,
                             const float* __restrict__ base, int hc, int hidden, float rms_eps,
                             float hc_eps, float* __restrict__ out) {
  const int flat = hc * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ double red[];
  __shared__ float pre_sh[256];
  __shared__ float rms_sh;
  auto block_reduce = [&](double v) -> double {
    red[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const double r = red[0];
    __syncthreads();
    return r;
  };
  double ls = 0.0;
  for (int p = tid; p < flat; p += nt) { const double v = x[p]; ls += v * v; }
  const double ss = block_reduce(ls);
  if (tid == 0) rms_sh = 1.0f / sqrtf(static_cast<float>(ss / static_cast<double>(flat)) + rms_eps);
  __syncthreads();
  const float r = rms_sh;
  for (int m = 0; m < hc; ++m) {
    double la = 0.0;
    const int frow = m * flat;
    for (int p = tid; p < flat; p += nt) la += static_cast<double>(x[p] * r) * fn[frow + p];
    const double acc = block_reduce(la);
    if (tid == 0) pre_sh[m] = Sig(static_cast<float>(acc) * scale + base[m]) + hc_eps;
  }
  __syncthreads();
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int m = 0; m < hc; ++m) acc += pre_sh[m] * x[m * hidden + h];
    out[h] = acc;
  }
}

unsigned Grid(int64_t n, int block) {
  return static_cast<unsigned>((n + block - 1) / block);
}

// ── launchers (host-vector wrappers; upload -> kernel -> download) ────────────
std::vector<float> MhcSinkhornLaunch(Queue& q, const std::vector<float>& logits, int64_t hc,
                                     int64_t iters, float eps) {
  hipStream_t s = AsStream(q);
  Dev dl = Upload(logits, s);
  std::vector<float> out(static_cast<size_t>(hc * hc));
  Dev dout(out.size() * sizeof(float));
  SinkhornKernel<<<1, 1, 0, s>>>(static_cast<const float*>(dl.p), static_cast<float*>(dout.p),
                                 static_cast<int>(hc), static_cast<int>(iters), eps);
  Download(out, dout.p, s);
  Check(hipStreamSynchronize(s), "sync sinkhorn");
  return out;
}

MhcPreResult MhcPreLaunch(Queue& q, const std::vector<float>& residual,
                          const std::vector<float>& fn, const std::vector<float>& scale,
                          const std::vector<float>& base, int64_t hc, int64_t hidden,
                          float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                          float hc_post_mult, int64_t iters,
                          const std::vector<float>& norm_weight, float norm_eps) {
  hipStream_t s = AsStream(q);
  const int hc3 = static_cast<int>((2 + hc) * hc);
  Dev dr = Upload(residual, s), df = Upload(fn, s), ds = Upload(scale, s), db = Upload(base, s);
  const bool has_norm = !norm_weight.empty();
  std::vector<float> nw = has_norm ? norm_weight : std::vector<float>(1, 0.0f);
  Dev dnw = Upload(nw, s);
  Dev dmix(static_cast<size_t>(hc3) * sizeof(float));
  MhcPreResult out;
  out.pre_mix.resize(static_cast<size_t>(hc));
  out.post_mix.resize(static_cast<size_t>(hc));
  out.comb_mix.resize(static_cast<size_t>(hc * hc));
  out.layer_input.resize(static_cast<size_t>(hidden));
  Dev dpre(out.pre_mix.size() * sizeof(float)), dpost(out.post_mix.size() * sizeof(float));
  Dev dcomb(out.comb_mix.size() * sizeof(float)), dlin(out.layer_input.size() * sizeof(float));
  MhcPreKernel<<<1, 1, 0, s>>>(
      static_cast<const float*>(dr.p), static_cast<const float*>(df.p),
      static_cast<const float*>(ds.p), static_cast<const float*>(db.p), static_cast<int>(hc),
      static_cast<int>(hidden), rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult,
      static_cast<int>(iters), static_cast<const float*>(dnw.p), has_norm ? 1 : 0, norm_eps,
      static_cast<float*>(dmix.p), static_cast<float*>(dpre.p), static_cast<float*>(dpost.p),
      static_cast<float*>(dcomb.p), static_cast<float*>(dlin.p));
  Download(out.pre_mix, dpre.p, s);
  Download(out.post_mix, dpost.p, s);
  Download(out.comb_mix, dcomb.p, s);
  Download(out.layer_input, dlin.p, s);
  Check(hipStreamSynchronize(s), "sync mhc_pre");
  return out;
}

std::vector<float> MhcPostLaunch(Queue& q, const std::vector<float>& x,
                                 const std::vector<float>& residual,
                                 const std::vector<float>& post_mix,
                                 const std::vector<float>& comb, int64_t hc, int64_t hidden) {
  hipStream_t s = AsStream(q);
  Dev dx = Upload(x, s), dr = Upload(residual, s), dp = Upload(post_mix, s), dc = Upload(comb, s);
  std::vector<float> out(static_cast<size_t>(hc * hidden));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  MhcPostKernel<<<Grid(hc * hidden, block), block, 0, s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(dr.p),
      static_cast<const float*>(dp.p), static_cast<const float*>(dc.p), static_cast<int>(hc),
      static_cast<int>(hidden), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(hipStreamSynchronize(s), "sync mhc_post");
  return out;
}

std::vector<float> HcHeadLaunch(Queue& q, const std::vector<float>& x,
                                const std::vector<float>& fn, float scale,
                                const std::vector<float>& base, int64_t hc, int64_t hidden,
                                float rms_eps, float hc_eps) {
  hipStream_t s = AsStream(q);
  Dev dx = Upload(x, s), df = Upload(fn, s), db = Upload(base, s);
  std::vector<float> out(static_cast<size_t>(hidden));
  Dev dout(out.size() * sizeof(float));
  HcHeadKernel<<<1, 256, 256 * sizeof(double), s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(df.p), scale,
      static_cast<const float*>(db.p), static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
      hc_eps, static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(hipStreamSynchronize(s), "sync hc_head");
  return out;
}

// ── in-place MHC glue (upload → kernel → download for discrete GPUs) ──────────
void MhcPostInPlaceLaunch(Queue& q, float* out, const float* x, const float* residual,
                          const float* post_mix, const float* comb, int64_t hc,
                          int64_t hidden) {
  if (hc == 0 || hidden == 0) return;
  hipStream_t s = AsStream(q);
  Dev dx = UploadRaw(x, static_cast<size_t>(hidden) * sizeof(float), s);
  Dev dr = UploadRaw(residual, static_cast<size_t>(hc * hidden) * sizeof(float), s);
  Dev dpm = UploadRaw(post_mix, static_cast<size_t>(hc) * sizeof(float), s);
  Dev dc = UploadRaw(comb, static_cast<size_t>(hc * hc) * sizeof(float), s);
  Dev dout(static_cast<size_t>(hc * hidden) * sizeof(float));
  const int block = 128;
  MhcPostKernel<<<Grid(hc * hidden, block), block, 0, s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(dr.p),
      static_cast<const float*>(dpm.p), static_cast<const float*>(dc.p),
      static_cast<int>(hc), static_cast<int>(hidden), static_cast<float*>(dout.p));
  DownloadRaw(out, dout.p, static_cast<size_t>(hc * hidden) * sizeof(float), s);
  Check(hipStreamSynchronize(s), "sync mhc_post_ip");
}

void HcHeadInPlaceLaunch(Queue& q, float* out, const float* x, const float* fn, float scale,
                         const float* base, int64_t hc, int64_t hidden, float rms_eps,
                         float hc_eps) {
  if (hidden == 0) return;
  hipStream_t s = AsStream(q);
  Dev dx = UploadRaw(x, static_cast<size_t>(hc * hidden) * sizeof(float), s);
  Dev dfn = UploadRaw(fn, static_cast<size_t>(hc * hc * hidden) * sizeof(float), s);
  Dev db = UploadRaw(base, static_cast<size_t>(hc) * sizeof(float), s);
  Dev dout(static_cast<size_t>(hidden) * sizeof(float));
  HcHeadKernel<<<1, 256, 256 * sizeof(double), s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(dfn.p), scale,
      static_cast<const float*>(db.p), static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
      hc_eps, static_cast<float*>(dout.p));
  DownloadRaw(out, dout.p, static_cast<size_t>(hidden) * sizeof(float), s);
  Check(hipStreamSynchronize(s), "sync hc_head_ip");
}

static bool MhcFusedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_MHC_FUSED");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

static unsigned MhcFinishBlock() {
  static const unsigned b = [] {
    const char* e = std::getenv("VT_V4_MHC_LEAN");
    if (e == nullptr) return 1024u;
    const int v = std::atoi(e);
    if (v == 0) return 256u;
    if (v == 1) return 1024u;
    if (v == 512 || v == 1024) return static_cast<unsigned>(v);
    return 1024u;
  }();
  return b;
}

static inline bool MhcLeanOn() { return MhcFinishBlock() != 256u; }

static bool MhcSink4On() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_MHC_SINK4");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

void MhcPreInPlaceLaunch(Queue& q, float* pre_mix, float* post_mix, float* comb_mix,
                         float* layer_input, float* mix_scratch, const float* residual,
                         const float* fn, const float* scale, const float* base, int64_t hc,
                         int64_t hidden, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                         float hc_post_mult, int64_t iters, const float* norm_weight,
                         bool has_norm, float norm_eps) {
  (void)mix_scratch;
  if (hidden == 0) return;
  hipStream_t s = AsStream(q);
  const unsigned block = 256;
  const int hc3 = static_cast<int>((2 + hc) * hc);
  const int flat = static_cast<int>(hc * hidden);
  Dev dr = UploadRaw(residual, static_cast<size_t>(flat) * sizeof(float), s);
  Dev dfn = UploadRaw(fn, static_cast<size_t>(hc3) * flat * sizeof(float), s);
  Dev ds = UploadRaw(scale, 3 * sizeof(float), s);
  Dev db = UploadRaw(base, static_cast<size_t>(hc3) * sizeof(float), s);
  std::vector<float> nw = has_norm
      ? std::vector<float>(norm_weight, norm_weight + hidden)
      : std::vector<float>(1, 0.0f);
  Dev dnw = Upload(nw, s);
  Dev dmix(static_cast<size_t>(hc3 + 1) * sizeof(float));
  Dev dpre(static_cast<size_t>(hc) * sizeof(float));
  Dev dpost(static_cast<size_t>(hc) * sizeof(float));
  Dev dcomb(static_cast<size_t>(hc * hc) * sizeof(float));
  Dev dlin(static_cast<size_t>(hidden) * sizeof(float));
  if (MhcFusedEnabled()) {
    const unsigned shmemf = block * sizeof(float);
    const int fold = MhcLeanOn() ? 1 : 0;
    MhcPreDotsFloatKernel<<<static_cast<unsigned>(hc3), block, shmemf, s>>>(
        static_cast<const float*>(dr.p), static_cast<const float*>(dfn.p), flat,
        static_cast<float*>(dmix.p), fold, hc3);
    const unsigned fblock = MhcFinishBlock();
    if (MhcSink4On() && hc == 4) {
      MhcPreFinishFloatKernel4<<<1, fblock, fblock * sizeof(float), s>>>(
          static_cast<const float*>(dr.p), static_cast<const float*>(ds.p),
          static_cast<const float*>(db.p), static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
          hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters),
          static_cast<const float*>(dnw.p), has_norm ? 1 : 0, norm_eps,
          static_cast<float*>(dmix.p), static_cast<float*>(dpre.p),
          static_cast<float*>(dpost.p), static_cast<float*>(dcomb.p),
          static_cast<float*>(dlin.p), fold);
    } else {
      MhcPreFinishFloatKernel<<<1, fblock, fblock * sizeof(float), s>>>(
          static_cast<const float*>(dr.p), static_cast<const float*>(ds.p),
          static_cast<const float*>(db.p), static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
          hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters),
          static_cast<const float*>(dnw.p), has_norm ? 1 : 0, norm_eps,
          static_cast<float*>(dmix.p), static_cast<float*>(dpre.p),
          static_cast<float*>(dpost.p), static_cast<float*>(dcomb.p),
          static_cast<float*>(dlin.p), fold);
    }
    Check(hipGetLastError(), "mhc_pre_ip launch (fused)");
  } else {
    const unsigned shmem = block * sizeof(double);
    MhcPreDotsKernel<<<static_cast<unsigned>(hc3), block, shmem, s>>>(
        static_cast<const float*>(dr.p), static_cast<const float*>(dfn.p), flat,
        static_cast<float*>(dmix.p));
    MhcPreFinishKernel<<<1, block, shmem, s>>>(
        static_cast<const float*>(dr.p), static_cast<const float*>(ds.p),
        static_cast<const float*>(db.p), static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters),
        static_cast<const float*>(dnw.p), has_norm ? 1 : 0, norm_eps,
        static_cast<float*>(dmix.p), static_cast<float*>(dpre.p), static_cast<float*>(dpost.p),
        static_cast<float*>(dcomb.p), static_cast<float*>(dlin.p));
    Check(hipGetLastError(), "mhc_pre_ip launch");
  }
  DownloadRaw(pre_mix, dpre.p, static_cast<size_t>(hc) * sizeof(float), s);
  DownloadRaw(post_mix, dpost.p, static_cast<size_t>(hc) * sizeof(float), s);
  DownloadRaw(comb_mix, dcomb.p, static_cast<size_t>(hc * hc) * sizeof(float), s);
  DownloadRaw(layer_input, dlin.p, static_cast<size_t>(hidden) * sizeof(float), s);
  Check(hipStreamSynchronize(s), "sync mhc_pre_ip");
}

// ── registration ─────────────────────────────────────────────────────────────
const MhcDeviceKernels kMhc = {&MhcSinkhornLaunch, &MhcPreLaunch, &MhcPostLaunch, &HcHeadLaunch,
                               &MhcPostInPlaceLaunch, &HcHeadInPlaceLaunch, &MhcPreInPlaceLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDeepseekV4Mhc, DeviceType::kROCM,
               const_cast<void*>(static_cast<const void*>(&kMhc)));
  }
} registrar;

}  // namespace
}  // namespace vllm::deepseek_v4
