// LTX-2.5 DiT — the DEVICE-FORWARD KERNEL SEAM (vt::OpId::kLtx2), phase L8.
//
// Row: MODEL-DIFFUSION-LTX25. Spec: .agents/specs/ltx-2-5.md phase L8. Issue #435.
//
// This header is DELIBERATELY THIN, and mirrors minimax_h3_device.h for the same
// reason: `src/vt/cuda/cuda_ltx2.cu` includes it, and a backend kernel TU must not
// have to compile the model's own headers. `ltx2_device.h` — which declares the
// forward — pulls in `ltx2.h` and with it `nlohmann/json.hpp`, so the two are
// separated rather than nvcc being asked to parse a JSON library. `ltx2_device.h`
// includes this one; nothing here includes anything but `vt/`.
//
// ─── WHY THE TABLE IS SO SHORT ───────────────────────────────────────────────
//
// Almost the whole LTX-2.5 DiT forward is already covered by tuned SHARED vt::
// ops, and the device port reuses them rather than growing a private kernel set:
//
//   Linear                   -> vt::MatmulBT + vt::Add (rank-1 row-broadcast bias)
//   rms_norm (no weight)     -> vt::RmsNorm with an all-ones weight
//   RMSNorm(inner_dim)       -> vt::RmsNorm (q_norm / k_norm)
//   LayerNorm(affine=False)  -> vt::LayerNorm (weight = bias = nullptr)
//   gelu(approximate="tanh") -> vt::GeluTanh
//   self-attention           -> vt::Attention(causal=false)
//   cross / biased attention -> vt::AttentionCross
//   dtype boundary           -> vt::CastBf16 / vt::CastF32
//
// That leaves exactly SEVEN ops the shared surface does not provide, and every
// one of them is a transcription of a named host helper in ltx2_dit.cpp /
// ltx2.cpp, in the same arithmetic order.
//
// SEAM: one OpProvider entry whose payload is a static kernels-struct of typed
// launchers, mirroring the kMiniMaxH3 precedent. Registered for BOTH kCPU
// (src/vt/cpu/cpu_ltx2.cpp) and kCUDA (src/vt/cuda/cuda_ltx2.cu) — so the device
// forward's STRUCTURE is covered by CPU CI and a GPU is needed to gate the
// KERNELS, not the port.
#pragma once

#include <cstdint>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm {

namespace ltx2 {

// The seven ops the shared vt:: surface does not express. Every one names the
// host helper it stands in for and the upstream line that helper came from.
//
// DTYPE: arithmetic is ALWAYS f32; only the load/store width varies. A bf16
// stream therefore rounds on STORE, which is exactly where upstream's cast falls
// — there is no separate rounding pass, and no way for a fused store to drift
// from it. `dtype` below is always the STREAM dtype (kF32 or kBF16) of the
// buffers it applies to; the `float*` parameters are the f32 islands named in
// this header's DTYPE note (the scale-shift tables and the RoPE cos/sin).
struct Ltx2DeviceKernels {
  // `AdaValue` (ltx2_dit.cpp:69-80, get_ada_values transformer.py:191-200):
  //   out[r, c] = table[table_row, c] + modulation[r, mod_index, c]
  // `modulation` is the flat [rows, num_params * width] AdaLN projection, so the
  // slice for row r lives at r*num_params*width + mod_index*width. Passing
  // `num_params` explicitly is deliberate: reading it as anything else silently
  // mixes the self-attention, feed-forward and cross-attention groups, which
  // renders confidently and wrongly.
  //
  // `table_row` and `mod_index` differ for the AV cross GATE, where upstream
  // hands in `scale_shift_table[4:]` (one row) against a one-parameter timestep
  // (transformer.py:214-215): table row 4, modulation index 0.
  void (*ada_value)(vt::Queue&, void* out, const float* table, const void* modulation,
                    int64_t rows, int64_t width, int64_t num_params, int64_t table_row,
                    int64_t mod_index, vt::DType dtype);

  // The affine tail of `AdaZero` (ltx2_dit.cpp:89-101, PytorchAdaZeroFunction
  // ops.py:50-58) and of `ModulateContext` (ltx2_dit.cpp:120-131,
  // apply_cross_attention_adaln transformer.py:420-447), IN PLACE:
  //   x[r, c] = x[r, c] * (1 + scale[r, c]) + shift[r, c]
  // `src_row_stride` is the ROW stride of scale/shift in elements. It is `width`
  // for the per-token AdaLN values and ZERO for the static [2, dim] prompt table,
  // whose single row broadcasts over every token. Passing it explicitly is what
  // lets one kernel serve both without a materialized broadcast.
  //
  // `src_dtype` is SEPARATE from `dtype` and that is the load-bearing part. The
  // AdaLN values arrive at the stream dtype (they come out of `ada_value`), but
  // the prompt table is F32 — the checkpoint stores it F32 and this port keeps it
  // F32. One dtype for both would either narrow the table or force a
  // materialized widening of `x`; reading an F32 table as bf16 would halve every
  // stride and produce a plausible-looking result from the wrong memory.
  void (*modulate)(vt::Queue&, void* x, const void* scale, const void* shift, int64_t rows,
                   int64_t width, int64_t src_row_stride, vt::DType dtype, vt::DType src_dtype);

  // The gated residual accumulate, IN PLACE:
  //   dst[r, c] += src[r, c] * gate[r / rows_per_gate_row, c]
  // Four host sites collapse onto this one kernel: the `x + y * gate` half of
  // `PostSelfAttention` (ltx2_dit.cpp:104-114, ops.py:72-82), the text
  // cross-attention gate (ltx2_dit.cpp:197-202, transformer.py:250-251), the
  // feed-forward gate (ltx2_dit.cpp:387-392, transformer.py:413-415), and
  // `AddGatedBroadcast` (ltx2_dit.cpp:135-145, transformer.py:355-364/:386-395).
  // The first three pass `rows_per_gate_row = 1`; the AV cross gate carries a
  // SINGLE token row per batch element and passes the stream's token count.
  void (*add_gated)(vt::Queue&, void* dst, const void* src, const void* gate, int64_t rows,
                    int64_t width, int64_t rows_per_gate_row, vt::DType dtype);

  // `PytorchGatedAttention` (ops.py:94-106) as ltx2.cpp:880-891 runs it, IN PLACE:
  //   attn[r, h * dim_head + e] *= 2 / (1 + exp(-logits[r, h]))
  // Applied to the attention output BEFORE `to_out` (attention.py:576-579) and
  // driven by the RAW attention input, never by the attention output — gating
  // after `to_out` would be a different model.
  void (*gate_heads)(vt::Queue&, void* attn, const void* logits, int64_t rows, int64_t heads,
                     int64_t dim_head, vt::DType dtype);

  // `Ltx2ApplyRotaryEmb` (ltx2.cpp:670-714, apply_rotary_emb rope.py:16-84) over
  // `x` [batch, tokens, dim] IN PLACE.
  //
  // SPLIT (`interleaved = false`, LTX-2.5's setting): cos/sin are
  // [batch, heads, tokens, per_head] and within one head the channels split into
  // HALVES — channel r pairs with channel per_head + r, NOT with its neighbour
  // (rope.py:67, d=2). `per_head` must be head_dim / 2.
  //
  // INTERLEAVED (`interleaved = true`, upstream's documented legacy mode):
  // cos/sin are [batch, tokens, dim] and the (even, odd) pairs rotate
  // (rope.py:30-40). `per_head` is ignored.
  void (*rope)(vt::Queue&, void* x, const float* cos, const float* sin, int64_t batch,
               int64_t tokens, int64_t dim, int64_t heads, int64_t per_head, bool interleaved,
               vt::DType dtype);

  // `_process_output`'s affine (ltx2_dit.cpp:534-547, model.py:482-488), IN PLACE:
  //   shift_v = table[0, c] + embedded[r, c]
  //   scale_v = table[1, c] + embedded[r, c]
  //   x[r, c] = x[r, c] * (1 + scale_v) + shift_v
  // Upstream forms `scale_shift_values = table + embedded` FIRST and only then
  // applies the affine; folding the two additions the other way round would round
  // differently, so the kernel keeps upstream's order.
  void (*output_modulate)(vt::Queue&, void* x, const float* table, const void* embedded,
                          int64_t rows, int64_t width, vt::DType dtype);

  // Plain ELEMENTWISE SiLU in place: x[i] = x[i] / (1 + exp(-x[i])). The shared op
  // set has SiluAndMul / MoeSiluMul (both GATED forms) but no ungated SiLU, which
  // `Ltx2AdaLayerNormSingle` needs twice (ltx2.cpp:766, :773 — TimestepEmbedding
  // .forward timestep_embedding.py:84-96 and AdaLayerNormSingle.forward
  // adaln.py:44-45).
  //
  // Matches ltx2.cpp's `Silu` EXACTLY (x / (1 + exp(-x))); the algebraically
  // equivalent x * sigmoid(x) is NOT bit-identical, and this path is gated
  // against that host reference.
  void (*silu)(vt::Queue&, void* x, int64_t n, vt::DType dtype);
};

// Resolver. Throws when nothing is registered for (kLtx2, device) — which cannot
// happen for kCPU/kCUDA in a normal build, but keeps the failure explicit rather
// than a null dereference on an unexpected backend.
const Ltx2DeviceKernels* Ltx2Device(vt::DeviceType device);
// True iff the table is registered for `device` (guards the device forward).
bool Ltx2DeviceKernelsAvailable(vt::DeviceType device);

}  // namespace ltx2

}  // namespace vllm
