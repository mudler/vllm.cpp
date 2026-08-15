// CUDA half of the LTX-2.5 DiT device-forward glue table (vt::OpId::kLtx2).
//
// The CPU sibling (src/vt/cpu/cpu_ltx2.cpp) is the reference; each kernel here is
// the same expression in the same arithmetic order, one thread per output
// element, so the two are bit-identical (no reductions appear in any of them
// except the per-(row, head) broadcast in gate_heads, which is a read).
//
// DTYPE: arithmetic is ALWAYS f32; only the load/store width varies, so a bf16
// stream rounds ONCE on store via __float2bfloat16 (round-to-nearest-even,
// identical to the host F32ToBF16 and to the CPU reference) -- the same
// discipline as cuda_glue.cu / cuda_layernorm.cu / cuda_minimax_h3.cu.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/ltx2_kernels.h"
#include "vt/ops.h"

namespace vllm::ltx2 {
namespace {

using vt::DType;
using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 1 ? 1 : (blocks < 65535 ? blocks : 65535));
}

__device__ inline float Load(const void* p, int64_t i, bool bf16) {
  return bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i])
              : static_cast<const float*>(p)[i];
}

__device__ inline void Store(void* p, int64_t i, float v, bool bf16) {
  if (bf16) {
    static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(v);
  } else {
    static_cast<float*>(p)[i] = v;
  }
}

// --- ada_value -------------------------------------------------------------
__global__ void AdaValueK(void* out, const float* table, const void* modulation, int64_t rows,
                          int64_t width, int64_t num_params, int64_t table_row, int64_t mod_index,
                          bool bf16) {
  const int64_t n = rows * width;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t r = i / width, c = i % width;
    const int64_t m = r * num_params * width + mod_index * width + c;
    Store(out, i, table[table_row * width + c] + Load(modulation, m, bf16), bf16);
  }
}

void AdaValueCuda(Queue& q, void* out, const float* table, const void* modulation, int64_t rows,
                  int64_t width, int64_t num_params, int64_t table_row, int64_t mod_index,
                  DType dtype) {
  const int64_t n = rows * width;
  if (n <= 0) return;
  AdaValueK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(out, table, modulation, rows, width, num_params,
                                                    table_row, mod_index,
                                                    dtype == DType::kBF16);
  Check(cudaGetLastError(), "ltx2 ada_value launch");
}

// --- modulate --------------------------------------------------------------
__global__ void ModulateK(void* x, const void* scale, const void* shift, int64_t rows,
                          int64_t width, int64_t src_row_stride, bool bf16, bool src_bf16) {
  const int64_t n = rows * width;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t r = i / width, c = i % width;
    const int64_t src = r * src_row_stride + c;
    Store(x, i,
          Load(x, i, bf16) * (1.0f + Load(scale, src, src_bf16)) + Load(shift, src, src_bf16),
          bf16);
  }
}

void ModulateCuda(Queue& q, void* x, const void* scale, const void* shift, int64_t rows,
                  int64_t width, int64_t src_row_stride, DType dtype, DType src_dtype) {
  const int64_t n = rows * width;
  if (n <= 0) return;
  ModulateK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, scale, shift, rows, width, src_row_stride,
                                                    dtype == DType::kBF16,
                                                    src_dtype == DType::kBF16);
  Check(cudaGetLastError(), "ltx2 modulate launch");
}

// --- add_gated -------------------------------------------------------------
__global__ void AddGatedK(void* dst, const void* src, const void* gate, int64_t rows,
                          int64_t width, int64_t rows_per_gate_row, bool bf16) {
  const int64_t n = rows * width;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t r = i / width, c = i % width;
    const int64_t g = (r / rows_per_gate_row) * width + c;
    Store(dst, i, Load(dst, i, bf16) + Load(src, i, bf16) * Load(gate, g, bf16), bf16);
  }
}

void AddGatedCuda(Queue& q, void* dst, const void* src, const void* gate, int64_t rows,
                  int64_t width, int64_t rows_per_gate_row, DType dtype) {
  const int64_t n = rows * width;
  if (n <= 0) return;
  AddGatedK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(dst, src, gate, rows, width,
                                                    rows_per_gate_row, dtype == DType::kBF16);
  Check(cudaGetLastError(), "ltx2 add_gated launch");
}

// --- gate_heads ------------------------------------------------------------
__global__ void GateHeadsK(void* attn, const void* logits, int64_t rows, int64_t heads,
                           int64_t dim_head, bool bf16) {
  const int64_t inner = heads * dim_head;
  const int64_t n = rows * inner;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t r = i / inner;
    const int64_t h = (i % inner) / dim_head;
    const float gate = 2.0f / (1.0f + __expf(-Load(logits, r * heads + h, bf16)));
    Store(attn, i, Load(attn, i, bf16) * gate, bf16);
  }
}

void GateHeadsCuda(Queue& q, void* attn, const void* logits, int64_t rows, int64_t heads,
                   int64_t dim_head, DType dtype) {
  const int64_t n = rows * heads * dim_head;
  if (n <= 0) return;
  GateHeadsK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(attn, logits, rows, heads, dim_head,
                                                     dtype == DType::kBF16);
  Check(cudaGetLastError(), "ltx2 gate_heads launch");
}

// --- rope ------------------------------------------------------------------
// One thread per ROTATED PAIR, in both layouts, so no thread writes a channel
// another thread reads -- the in-place halves rotation (lo, hi) is not safe to
// split across threads any other way.
__global__ void RopeInterleavedK(void* x, const float* cos, const float* sin, int64_t rows,
                                 int64_t dim, bool bf16) {
  const int64_t pairs = dim / 2;
  const int64_t n = rows * pairs;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t r = i / pairs;
    const int64_t c = (i % pairs) * 2;
    const int64_t off = r * dim + c;
    const float a = Load(x, off, bf16), b = Load(x, off + 1, bf16);
    Store(x, off, a * cos[off] + (-b) * sin[off], bf16);
    Store(x, off + 1, b * cos[off + 1] + a * sin[off + 1], bf16);
  }
}

__global__ void RopeSplitK(void* x, const float* cos, const float* sin, int64_t batch,
                           int64_t tokens, int64_t dim, int64_t heads, int64_t per_head,
                           bool bf16) {
  const int64_t head_dim = dim / heads;
  const int64_t n = batch * tokens * heads * per_head;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t r = i % per_head;
    const int64_t h = (i / per_head) % heads;
    const int64_t t = (i / (per_head * heads)) % tokens;
    const int64_t b = i / (per_head * heads * tokens);
    const int64_t v = (b * tokens + t) * dim + h * head_dim;
    const int64_t base = ((b * heads + h) * tokens + t) * per_head;
    const float c = cos[base + r], s = sin[base + r];
    const float lo = Load(x, v + r, bf16), hi = Load(x, v + per_head + r, bf16);
    Store(x, v + r, lo * c - hi * s, bf16);
    Store(x, v + per_head + r, hi * c + lo * s, bf16);
  }
}

void RopeCuda(Queue& q, void* x, const float* cos, const float* sin, int64_t batch,
              int64_t tokens, int64_t dim, int64_t heads, int64_t per_head, bool interleaved,
              DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  if (interleaved) {
    const int64_t n = batch * tokens * (dim / 2);
    if (n <= 0) return;
    RopeInterleavedK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, cos, sin, batch * tokens, dim,
                                                             bf16);
  } else {
    const int64_t n = batch * tokens * heads * per_head;
    if (n <= 0) return;
    RopeSplitK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, cos, sin, batch, tokens, dim, heads,
                                                       per_head, bf16);
  }
  Check(cudaGetLastError(), "ltx2 rope launch");
}

// --- output_modulate -------------------------------------------------------
__global__ void OutputModulateK(void* x, const float* table, const void* embedded, int64_t rows,
                                int64_t width, bool bf16) {
  const int64_t n = rows * width;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t c = i % width;
    const float e = Load(embedded, i, bf16);
    const float shift_v = table[c] + e;
    const float scale_v = table[width + c] + e;
    Store(x, i, Load(x, i, bf16) * (1.0f + scale_v) + shift_v, bf16);
  }
}

void OutputModulateCuda(Queue& q, void* x, const float* table, const void* embedded, int64_t rows,
                        int64_t width, DType dtype) {
  const int64_t n = rows * width;
  if (n <= 0) return;
  OutputModulateK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, table, embedded, rows, width,
                                                          dtype == DType::kBF16);
  Check(cudaGetLastError(), "ltx2 output_modulate launch");
}

// --- silu ------------------------------------------------------------------
// __expf, not expf: the CPU reference uses std::exp and this kernel is held to
// the goldens, not to the CPU kernel byte-for-byte. Every other kernel in this
// table IS bit-identical to its CPU twin; this one and gate_heads are the two
// that carry a transcendental, and they say so here rather than in a claim.
__global__ void SiluK(void* x, int64_t n, bool bf16) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float v = Load(x, i, bf16);
    Store(x, i, v / (1.0f + __expf(-v)), bf16);
  }
}

void SiluCuda(Queue& q, void* x, int64_t n, DType dtype) {
  if (n <= 0) return;
  SiluK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, n, dtype == DType::kBF16);
  Check(cudaGetLastError(), "ltx2 silu launch");
}

const Ltx2DeviceKernels kKernels{
    &AdaValueCuda, &ModulateCuda,       &AddGatedCuda, &GateHeadsCuda,
    &RopeCuda,     &OutputModulateCuda, &SiluCuda,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLtx2, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vllm::ltx2
