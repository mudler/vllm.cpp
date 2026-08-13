// CPU half of the LTX-2.5 DiT device-forward glue table (vt::OpId::kLtx2).
//
// Seven small ops the shared vt:: surface does not cover; see
// include/vllm/model_executor/models/ltx2_kernels.h for why the table is this
// short (everything else in the DiT forward reuses tuned shared ops).
//
// Each kernel is a 1:1 transcription of the host reference it stands in for
// (ltx2_dit.cpp: AdaValue / AdaZero's affine tail / PostSelfAttention's
// accumulate / AddGatedBroadcast / _process_output's affine; ltx2.cpp:
// Ltx2ApplyRotaryEmb / the gated-attention scaling / Silu), in the SAME
// arithmetic order. The CUDA sibling lives in src/vt/cuda/cuda_ltx2.cu.
//
// DTYPE: arithmetic is ALWAYS f32; only the load/store width varies. A bf16
// stream therefore rounds on STORE, which is exactly upstream's cast point -- no
// separate rounding pass, and no way for a fused store to drift from it. The
// `float*` parameters are the two f32 islands ltx2_kernels.h names: the
// scale-shift tables (stored F32 by the checkpoint itself) and the RoPE cos/sin
// (whose last-ulp precision is the whole point of Ltx2FreqGrid).
//
// Registering this on kCPU is what lets the whole device-forward code path be
// covered by CPU CI, so a GPU is needed to gate the KERNELS, not the port's
// structure. Same decision, same reason, as cpu_minimax_h3.cpp:15-17.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vllm/model_executor/models/ltx2_kernels.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vt::cpu {
namespace {

// bf16 is stored as the high 16 bits of the f32 pattern; round-to-nearest-even on
// store, matching torch's `.to(bfloat16)` and cpu_minimax_h3.cpp's StoreBf16.
inline float LoadBf16(const void* p, int64_t i) {
  const uint32_t bits = static_cast<uint32_t>(static_cast<const uint16_t*>(p)[i]) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline void StoreBf16(void* p, int64_t i, float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u) {  // inf/nan pass through
    bits &= 0xFFFF0000u;
  } else {
    const uint32_t lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
  }
  static_cast<uint16_t*>(p)[i] = static_cast<uint16_t>(bits >> 16);
}

inline float Load(const void* p, int64_t i, bool bf16) {
  return bf16 ? LoadBf16(p, i) : static_cast<const float*>(p)[i];
}

inline void Store(void* p, int64_t i, float v, bool bf16) {
  if (bf16) {
    StoreBf16(p, i, v);
  } else {
    static_cast<float*>(p)[i] = v;
  }
}

// AdaValue (ltx2_dit.cpp:69-80, get_ada_values transformer.py:191-200).
void Ltx2AdaValue(Queue&, void* out, const float* table, const void* modulation, int64_t rows,
                  int64_t width, int64_t num_params, int64_t table_row, int64_t mod_index,
                  DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  const float* t = table + table_row * width;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t m = r * num_params * width + mod_index * width;
    for (int64_t c = 0; c < width; ++c) {
      Store(out, r * width + c, t[c] + Load(modulation, m + c, bf16), bf16);
    }
  }
}

// The affine tail of AdaZero (ltx2_dit.cpp:98, PytorchAdaZeroFunction ops.py:50-58)
// and of ModulateContext (ltx2_dit.cpp:128). `src_row_stride == 0` broadcasts the
// static [2, dim] prompt table's single row over every token.
void Ltx2Modulate(Queue&, void* x, const void* scale, const void* shift, int64_t rows,
                  int64_t width, int64_t src_row_stride, DType dtype, DType src_dtype) {
  const bool bf16 = dtype == DType::kBF16;
  const bool src_bf16 = src_dtype == DType::kBF16;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t src = r * src_row_stride;
    for (int64_t c = 0; c < width; ++c) {
      const float v = Load(x, r * width + c, bf16) * (1.0f + Load(scale, src + c, src_bf16)) +
                      Load(shift, src + c, src_bf16);
      Store(x, r * width + c, v, bf16);
    }
  }
}

// The gated residual accumulate: PostSelfAttention's `x += y * gate`
// (ltx2_dit.cpp:110), the text cross-attention gate (:201), the feed-forward gate
// (:390) and AddGatedBroadcast (:142) are the same expression at different
// `rows_per_gate_row`.
void Ltx2AddGated(Queue&, void* dst, const void* src, const void* gate, int64_t rows,
                  int64_t width, int64_t rows_per_gate_row, DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t g = (r / rows_per_gate_row) * width;
    for (int64_t c = 0; c < width; ++c) {
      const float v = Load(dst, r * width + c, bf16) +
                      Load(src, r * width + c, bf16) * Load(gate, g + c, bf16);
      Store(dst, r * width + c, v, bf16);
    }
  }
}

// PytorchGatedAttention (ops.py:94-106) as ltx2.cpp:884-890 applies it.
void Ltx2GateHeads(Queue&, void* attn, const void* logits, int64_t rows, int64_t heads,
                   int64_t dim_head, DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  const int64_t inner = heads * dim_head;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t h = 0; h < heads; ++h) {
      const float gate = 2.0f / (1.0f + std::exp(-Load(logits, r * heads + h, bf16)));
      const int64_t base = r * inner + h * dim_head;
      for (int64_t e = 0; e < dim_head; ++e) {
        Store(attn, base + e, Load(attn, base + e, bf16) * gate, bf16);
      }
    }
  }
}

// Ltx2ApplyRotaryEmb (ltx2.cpp:670-714, apply_rotary_emb rope.py:16-84).
void Ltx2Rope(Queue&, void* x, const float* cos, const float* sin, int64_t batch, int64_t tokens,
              int64_t dim, int64_t heads, int64_t per_head, bool interleaved, DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  if (interleaved) {
    // apply_interleaved_rotary_emb (rope.py:30-40): rotate the (even, odd) pairs.
    for (int64_t r = 0; r < batch * tokens; ++r) {
      const int64_t row = r * dim;
      const int64_t off = r * dim;
      for (int64_t c = 0; c < dim; c += 2) {
        const float a = Load(x, row + c, bf16), b = Load(x, row + c + 1, bf16);
        Store(x, row + c, a * cos[off + c] + (-b) * sin[off + c], bf16);
        Store(x, row + c + 1, b * cos[off + c + 1] + a * sin[off + c + 1], bf16);
      }
    }
    return;
  }
  // apply_split_rotary_emb (rope.py:43-84): within one head the channels split
  // into HALVES -- channel r pairs with per_head + r, not with its neighbour.
  const int64_t head_dim = dim / heads;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t t = 0; t < tokens; ++t) {
      const int64_t row = (b * tokens + t) * dim;
      for (int64_t h = 0; h < heads; ++h) {
        const int64_t v = row + h * head_dim;
        const int64_t base = ((b * heads + h) * tokens + t) * per_head;
        for (int64_t r = 0; r < per_head; ++r) {
          const float c = cos[base + r], s = sin[base + r];
          const float lo = Load(x, v + r, bf16), hi = Load(x, v + per_head + r, bf16);
          Store(x, v + r, lo * c - hi * s, bf16);
          Store(x, v + per_head + r, hi * c + lo * s, bf16);
        }
      }
    }
  }
}

// _process_output's affine (ltx2_dit.cpp:539-547, model.py:482-488). Upstream
// forms `table + embedded` FIRST and only then applies the affine; folding the
// two additions the other way round would round differently.
void Ltx2OutputModulate(Queue&, void* x, const float* table, const void* embedded, int64_t rows,
                        int64_t width, DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  const float* shift = table;
  const float* scale = table + width;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < width; ++c) {
      const float e = Load(embedded, r * width + c, bf16);
      const float shift_v = shift[c] + e;
      const float scale_v = scale[c] + e;
      Store(x, r * width + c, Load(x, r * width + c, bf16) * (1.0f + scale_v) + shift_v, bf16);
    }
  }
}

// Matches ltx2.cpp:67 `Silu` EXACTLY (x / (1 + exp(-x))); the algebraically
// equivalent x * sigmoid(x) is NOT bit-identical, and this path is gated against
// that host reference.
void Ltx2Silu(Queue&, void* x, int64_t n, DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  for (int64_t i = 0; i < n; ++i) {
    const float v = Load(x, i, bf16);
    Store(x, i, v / (1.0f + std::exp(-v)), bf16);
  }
}

const vllm::ltx2::Ltx2DeviceKernels kKernels{
    &Ltx2AdaValue, &Ltx2Modulate,        &Ltx2AddGated, &Ltx2GateHeads,
    &Ltx2Rope,     &Ltx2OutputModulate,  &Ltx2Silu,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLtx2, DeviceType::kCPU,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
