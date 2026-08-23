// LTX-2.5 DiT — the DEVICE-RESIDENT forward (phase L8).
//
// `Ltx2DitForward` (ltx2_dit.cpp) is the portable reference: it computes into
// host `std::vector<float>` buffers and only reaches for `vt::` at the two big
// ops, so on a CUDA device it would hand HOST pointers to device kernels. This
// file runs the SAME graph with every activation in device memory, so the block
// stack — and, above it, the denoise loop — never round-trips through the host.
//
// ─── WHAT IS PORTED, AND FROM WHERE ──────────────────────────────────────────
// Structure is 1:1 with the reference; each helper below names the reference
// helper it replaces. The port REUSES tuned shared ops wherever one exists, which
// is why only seven LTX-specific kernels were added (ltx2_device.h):
//
//   Linear                    -> vt::MatmulBT + vt::Add (rank-1 row-broadcast bias)
//   RmsNormRows (no weight)   -> vt::RmsNorm with an all-ones weight
//   RmsNormRows (q/k norm)    -> vt::RmsNorm
//   LayerNormRows             -> vt::LayerNorm (weight = bias = nullptr)
//   GeluTanh                  -> vt::GeluTanh
//   self-attention            -> vt::AttentionDenseFa2(causal=false)
//   cross / biased attention  -> vt::AttentionCross
//   AdaValue / AdaZero affine -> kLtx2 glue table
//   PostSelfAttention / gates -> kLtx2 glue table (add_gated)
//   Ltx2ApplyRotaryEmb        -> kLtx2 glue table (rope)
//   gated attention           -> kLtx2 glue table (gate_heads)
//   _process_output affine    -> kLtx2 glue table (output_modulate)
//   Silu (ungated)            -> kLtx2 glue table
//
// ─── WHAT STAYS ON THE HOST, AND WHY EACH ONE IS RIGHT THERE ─────────────────
// Three things, all of them O(tokens) or smaller and none of them per-layer:
//
//   * The RoPE frequency tables (`Ltx2PrecomputeFreqsCis`). Built ONCE per
//     forward from the fp64 position grid, then uploaded; MiniMax-H3 builds its
//     own rope cache the same way (minimax_h3_device.cpp:193-210). The fp64
//     ladder is the point — see Ltx2FreqGrid (ltx2.h:253-260).
//   * The two mask preparations (`Ltx2PrepareContextMask`,
//     `Ltx2PrepareSelfAttentionMask`). One pass over the mask, once per forward.
//   * The 256-channel sinusoidal timestep projection inside the AdaLN single.
//     One [count, 256] table per stream per forward.
//
// Everything else — every GEMM, every norm, every elementwise pass over an
// activation — is a device kernel. This is stated precisely because "it ran on
// the GPU" is the claim this phase exists to make true.
//
// ─── NUMERICS: CLOSE, NOT BIT-IDENTICAL ──────────────────────────────────────
// This path is NOT bit-identical to the CPU reference and does not claim to be.
// The divergence is in the SHARED ops, not the LTX kernels: vt::RmsNorm and
// vt::LayerNorm reduce in f32 where RmsNormRows / LayerNormRows accumulate in
// double, the CUDA cross-attention uses an online-softmax recurrence where the
// CPU one uses an explicit three-pass, and MatmulBT uses its own accumulation
// order. f32 is what upstream torch does, so the device path is arguably the
// closer mirror. It is gated against the SAME upstream goldens as the CPU
// forward, at a tolerance that is stated and justified in the test.
#include "vllm/model_executor/models/ltx2_device.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using vt::DType;
using vt::Tensor;

// ---------------------------------------------------------------------------
// Small device utilities
// ---------------------------------------------------------------------------

const ltx2::Ltx2DeviceKernels* Glue(const Dev& d) {
  VT_CHECK(ltx2::Ltx2DeviceKernelsAvailable(d.q.device.type),
           "ltx2: no device glue table registered for this backend (vt::OpId::kLtx2). The CPU "
           "and CUDA tables live in src/vt/cpu/cpu_ltx2.cpp and src/vt/cuda/cuda_ltx2.cu");
  return ltx2::Ltx2Device(d.q.device.type);
}

// bf16 round-to-nearest-even, identical to vt::F32ToBF16 and to the CPU glue's
// StoreBf16, so a host-prepared buffer rounds exactly where a kernel store would.
uint16_t RoundBf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u) {
    bits &= 0xFFFF0000u;
  } else {
    const uint32_t lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
  }
  return static_cast<uint16_t>(bits >> 16);
}

float WidenBf16(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Upload an f32 host buffer at the STREAM dtype, rounding on the way in.
DBuf UploadStream(Dev d, DType s, const float* host, const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t e : shape) n *= e;
  if (s == DType::kF32) return DBuf(d, DType::kF32, shape, host);
  std::vector<uint16_t> narrowed(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) narrowed[static_cast<size_t>(i)] = RoundBf16(host[i]);
  return DBuf(d, DType::kBF16, shape, narrowed.data());
}

std::vector<float> DownloadF32(Dev d, DBuf& buf, int64_t n) {
  std::vector<float> out(static_cast<size_t>(n));
  if (buf.t().dtype == DType::kF32) {
    buf.Download(d, out.data());
    return out;
  }
  std::vector<uint16_t> raw(static_cast<size_t>(n));
  buf.Download(d, raw.data());
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = WidenBf16(raw[static_cast<size_t>(i)]);
  return out;
}

// A row-offset view into a [rows, ...] device buffer. Used for the per-batch
// attention slices, which is the only place this forward indexes a buffer.
Tensor RowsFrom(const Tensor& base, int64_t first_row, int64_t row_elems,
                const std::vector<int64_t>& shape) {
  auto* p = static_cast<std::byte*>(base.data) +
            static_cast<size_t>(first_row * row_elems) * vt::SizeOf(base.dtype);
  return MakeTensor(p, base.dtype, base.device, shape);
}

// The weightless `rms_norm(x, eps)` (ops.py:58) has no upstream weight at all, but
// vt::RmsNorm takes one. An all-ones vector is that identity: multiplying by 1.0f
// is exact, so this is a signature adaptation and not a numerical one.
//
// THE DTYPE IS NOT FREE, and a real GPU is what said so. The CPU RmsNorm kernel
// reads its weight through a dtype-aware `LoadF32`, so an f32 ones vector against
// a bf16 stream is fine there and every CPU-backend case passed with one. The
// CUDA kernel instead REQUIRES `w.dtype == x.dtype` (cuda_ops.cu:452) and throws.
// So the identity is built at the STREAM dtype: 1.0f is exactly representable in
// bf16, which is what makes that a dtype change and not a precision one. Cached
// per (dtype, width) — the forward needs at most two widths.
class OnesCache {
 public:
  explicit OnesCache(Dev d) : d_(d) {}
  const Tensor& Get(DType dt, int64_t width) {
    const auto key = std::make_pair(static_cast<int>(dt), width);
    auto it = bufs_.find(key);
    if (it == bufs_.end()) {
      if (dt == DType::kBF16) {
        // 0x3F80 is the bf16 pattern of 1.0f — the top 16 bits of 0x3F800000.
        const std::vector<uint16_t> ones(static_cast<size_t>(width), 0x3F80);
        it = bufs_.emplace(key, DBuf(d_, DType::kBF16, {width}, ones.data())).first;
      } else {
        const std::vector<float> ones(static_cast<size_t>(width), 1.0f);
        it = bufs_.emplace(key, DBuf(d_, DType::kF32, {width}, ones.data())).first;
      }
    }
    return it->second.t();
  }

 private:
  Dev d_;
  std::map<std::pair<int, int64_t>, DBuf> bufs_;
};

// The one bundle every helper below needs, so the call sites read like the
// reference's rather than carrying six parameters each.
struct Ctx {
  Dev d;
  DType s;  // the STREAM dtype
  const ltx2::Ltx2DeviceKernels* k;
  const Ltx2DitParams* p;
  OnesCache* ones;
};

// vt::MatmulBT + optional rank-1 bias — the device twin of ltx2.cpp's `Linear`.
// `weight` is [out_features, in_features], torch's own nn.Linear layout, so
// y = x @ W^T + b reads straight off the module.
void LinearDev(Ctx& c, const Tensor& in, int64_t rows, int64_t in_features,
               const Ltx2LinearWeight& w, Tensor& out) {
  VT_CHECK(w.weight.rank == 2 && w.weight.shape[1] == in_features,
           "ltx2 device linear: weight shape does not match input width");
  VT_CHECK(w.weight.dtype == c.s,
           "ltx2 device linear: the weight dtype must equal the stream dtype. Staging f32 "
           "weights under a bf16 stream (or the reverse) compares a DIFFERENT MODEL, not a "
           "different dtype policy — see Ltx2StageDitWeightsToDevice");
  Tensor a = Reshape(in, {rows, in_features});
  Tensor o = Reshape(out, {rows, w.weight.shape[0]});
  vt::MatmulBT(c.d.q, o, a, w.weight);
  if (w.bias.data != nullptr) {
    vt::Add(c.d.q, o, o, w.bias);  // rank-1 row-broadcast == a nn.Linear bias term
  }
}

// `RmsNormRows` with no weight (ltx2_dit.cpp:29-38, ops.py:58).
void RmsNormNoWeight(Ctx& c, Tensor& out, const Tensor& x, int64_t rows, int64_t width) {
  vt::RmsNormArgs args;
  args.eps = static_cast<float>(c.p->norm_eps);
  Tensor o = Reshape(out, {rows, width});
  Tensor i = Reshape(x, {rows, width});
  vt::RmsNorm(c.d.q, o, i, c.ones->Get(c.s, width), args);
}

// `RmsNormRows` WITH an elementwise weight (ltx2.cpp:51-65; torch.nn.RMSNorm, the
// q_norm / k_norm at attention.py:505-506, applied over the FULL inner width).
void RmsNormWeighted(Ctx& c, Tensor& out, const Tensor& x, const Tensor& weight, int64_t rows,
                     int64_t width) {
  vt::RmsNormArgs args;
  args.eps = static_cast<float>(c.p->norm_eps);
  Tensor o = Reshape(out, {rows, width});
  Tensor i = Reshape(x, {rows, width});
  vt::RmsNorm(c.d.q, o, i, weight, args);
}

// Every glue kernel that takes a table takes it as `const float*`. A bf16 tensor
// handed to one of those is not a type error, a shape error, or a device error —
// it is a silent reinterpretation that halves every stride and reads the wrong
// half of the wrong element. That is exactly the defect this port hit, and this
// assertion is the guard that would have named it in one line: the `_a2v_ca_*`
// tables fell outside a SUFFIX-matching staging predicate, were staged bf16, and
// turned the audio<->video gate into 2.85e32 while the f32 arm — where no
// mismatch is possible — stayed green at 1e-7.
void CheckTableF32(const Tensor& t, const char* what) {
  VT_CHECK(t.dtype == DType::kF32,
           std::string("ltx2 device: '") + what +
               "' must be F32. The scale-shift tables are stored F32 by the checkpoint and "
               "are read through a `const float*` by the kLtx2 glue; a bf16 table would be "
               "reinterpreted rather than converted. See Ltx2DitTensorIsTable.");
}

// `AdaValue` (ltx2_dit.cpp:69-86).
DBuf AdaValueDev(Ctx& c, const Tensor& table, const Tensor& modulation, int64_t rows,
                 int64_t width, int64_t num_params, int64_t table_row, int64_t mod_index) {
  CheckTableF32(table, "an AdaLN scale-shift table");
  DBuf out(c.d, c.s, {rows, width});
  c.k->ada_value(c.d.q, out.t().data, table.Ptr<float>(), modulation.data, rows, width, num_params,
                 table_row, mod_index, c.s);
  return out;
}

DBuf AdaValueDev(Ctx& c, const Tensor& table, const Tensor& modulation, int64_t rows,
                 int64_t width, int64_t num_params, int64_t index) {
  return AdaValueDev(c, table, modulation, rows, width, num_params, index, index);
}

// `AdaZero` (ltx2_dit.cpp:89-101): rms_norm(x) * (1 + scale) + shift.
DBuf AdaZeroDev(Ctx& c, const Tensor& x, const DBuf& scale, const DBuf& shift, int64_t rows,
                int64_t width) {
  DBuf out(c.d, c.s, {rows, width});
  RmsNormNoWeight(c, out.t(), x, rows, width);
  // scale/shift come out of `ada_value` at the stream dtype, so src_dtype == c.s.
  c.k->modulate(c.d.q, out.t().data, scale.t().data, shift.t().data, rows, width, width, c.s, c.s);
  return out;
}

// ---------------------------------------------------------------------------
// RoPE tables, uploaded
// ---------------------------------------------------------------------------

struct DevFreqs {
  std::optional<DBuf> cos, sin;
  int64_t per_head = 0;
  bool interleaved = false;
  bool valid = false;
};

DevFreqs UploadFreqs(Dev d, const Ltx2FreqsCis& pe, Ltx2RopeType rope_type) {
  DevFreqs out;
  if (pe.cos.empty()) return out;
  const int64_t n = static_cast<int64_t>(pe.cos.size());
  // f32 by design: the two frequency ladders differ only in the last f32 ulps of
  // every angle and the audio ladder multiplies those up by four orders of
  // magnitude before RoPE takes their cosine (ltx2.h:253-260). Narrowing the
  // TABLE would discard exactly the bits that distinguish them.
  out.cos.emplace(d, DType::kF32, std::vector<int64_t>{n}, pe.cos.data());
  out.sin.emplace(d, DType::kF32, std::vector<int64_t>{n}, pe.sin.data());
  out.interleaved = rope_type == Ltx2RopeType::kInterleaved;
  out.per_head = out.interleaved ? 0 : pe.shape[3];
  out.valid = true;
  return out;
}

void ApplyRopeDev(Ctx& c, Tensor& x, int64_t batch, int64_t tokens, int64_t dim, int64_t heads,
                  const DevFreqs& pe) {
  if (!pe.valid) return;
  c.k->rope(c.d.q, x.data, pe.cos->t().Ptr<float>(), pe.sin->t().Ptr<float>(), batch, tokens, dim,
            heads, pe.per_head, pe.interleaved, c.s);
}

// ---------------------------------------------------------------------------
// Attention (the device twin of ltx2.cpp's Ltx2Attention)
// ---------------------------------------------------------------------------

struct AttnArgsDev {
  int64_t batch = 1;
  int64_t tokens = 0;          // Tq
  int64_t context_tokens = 0;  // S
  int64_t query_dim = 0;
  int64_t context_dim = 0;
  int64_t heads = 0;
  int64_t dim_head = 0;
  const DevFreqs* pe = nullptr;
  const DevFreqs* k_pe = nullptr;
  const Tensor* bias = nullptr;  // f32 [batch * bias_rows, S]
  int64_t bias_rows = 0;
  // `all_perturbed` (attention.py:557, `use_attention = not all_perturbed`). The
  // device twin of `Ltx2AttentionArgs::all_perturbed` (ltx2.h:460-476), with the
  // same meaning and the same port boundary: the PARTIAL blend at
  // `attention.py:573` is not ported and is degenerate at the one batch size this
  // port runs.
  bool all_perturbed = false;
};

// attention.py:576-579 — everything the ordinary path and the STG-perturbed path
// share, which is the per-head gate and `to_out`. Factored out rather than
// duplicated for the reason `ltx2.cpp:825-829` gives on the host arm: a perturbed
// pass that skipped `to_out` returns a tensor of the right shape in the wrong
// width-space, the block would add it to the residual, and it would render.
DBuf AttentionEpilogueDev(Ctx& c, const Ltx2AttentionWeights& w, const Tensor& x, DBuf attn,
                          const AttnArgsDev& a) {
  const int64_t batch = a.batch, tq = a.tokens;
  const int64_t heads = a.heads, dim_head = a.dim_head;
  const int64_t inner = heads * dim_head;

  // PytorchGatedAttention (ops.py:94-106), applied to the attention output BEFORE
  // `to_out` (attention.py:576-579) and driven by the RAW input `x`.
  if (w.to_gate_logits.weight.data != nullptr) {
    DBuf logits(c.d, c.s, {batch * tq, heads});
    LinearDev(c, x, batch * tq, a.query_dim, w.to_gate_logits, logits.t());
    c.k->gate_heads(c.d.q, attn.t().data, logits.t().data, batch * tq, heads, dim_head, c.s);
  }

  DBuf out(c.d, c.s, {batch * tq, a.query_dim});
  LinearDev(c, attn.t(), batch * tq, inner, w.to_out, out.t());
  return out;
}

// Attention.forward (attention.py:520-579). `context` is null for self-attention
// (upstream's `context = x if context is None else context`).
DBuf AttentionDev(Ctx& c, const Ltx2AttentionWeights& w, const Tensor& x, const Tensor* context,
                  const AttnArgsDev& a) {
  const int64_t batch = a.batch, tq = a.tokens;
  const int64_t heads = a.heads, dim_head = a.dim_head;
  const int64_t inner = heads * dim_head;
  const Tensor& ctx = context != nullptr ? *context : x;
  const int64_t s = context != nullptr ? a.context_tokens : tq;
  const int64_t ctx_dim = context != nullptr ? a.context_dim : a.query_dim;

  // attention.py:559-565: v first, then q and k.
  DBuf v(c.d, c.s, {batch * s, inner});
  LinearDev(c, ctx, batch * s, ctx_dim, w.to_v, v.t());

  // attention.py:557 / :561-562 — `use_attention = not all_perturbed`, and when
  // it is false `out = v`. The STG arm computes `to_v` and NOTHING else of the
  // attention: no `to_q`, no `to_k`, no q/k RMSNorm, no RoPE, no scores. Written
  // as an early exit rather than a chain of `if (!perturbed)` guards, exactly as
  // the host arm is (ltx2.cpp:873-893), so the skipped work is visibly skipped —
  // a guarded form that still projected q and threw the result away would be
  // numerically identical and would hide the whole point of the perturbation,
  // which is that the query/key path does not run.
  if (a.all_perturbed) {
    // The host arm's own refusal (ltx2.cpp:880-886), mirrored rather than
    // dropped. `Ltx2DitPerturbation` carries no cross-attention SELF-perturbation
    // type: the two CROSS types are `cross_attn_skip_all` booleans that skip the
    // whole branch (transformer.py:335, :367), not a rule applied inside it.
    VT_CHECK(context == nullptr,
             "ltx2 attention (device): `all_perturbed` is upstream's SELF-attention STG "
             "perturbation (guidance/perturbations.py:8-16 names SKIP_VIDEO_SELF_ATTN and "
             "SKIP_AUDIO_SELF_ATTN). The CROSS-attention perturbations reach a whole-branch "
             "guard rather than this rule, so a cross call carrying this flag is refused "
             "rather than served the self-attention substitution");
    // `s == tq` here, because a self-attention call is the only one that reaches
    // this branch, so `v` already has the [batch * tq, inner] shape the epilogue
    // reads. Asserted rather than assumed: a future cross caller that slipped
    // past the check above would otherwise run `to_out` over the wrong row count.
    VT_CHECK(s == tq, "ltx2 attention (device): a perturbed pass must be square");
    return AttentionEpilogueDev(c, w, x, std::move(v), a);
  }

  DBuf qb(c.d, c.s, {batch * tq, inner});
  LinearDev(c, x, batch * tq, a.query_dim, w.to_q, qb.t());

  // PytorchPreAttention (ops.py:22-37): the q/k RMSNorm runs over the FULL inner
  // width, before the head split, and RoPE follows it.
  DBuf qn(c.d, c.s, {batch * tq, inner});
  RmsNormWeighted(c, qn.t(), qb.t(), w.q_norm, batch * tq, inner);
  DBuf kb(c.d, c.s, {batch * s, inner});
  LinearDev(c, ctx, batch * s, ctx_dim, w.to_k, kb.t());
  DBuf kn(c.d, c.s, {batch * s, inner});
  RmsNormWeighted(c, kn.t(), kb.t(), w.k_norm, batch * s, inner);

  if (a.pe != nullptr && a.pe->valid) {
    ApplyRopeDev(c, qn.t(), batch, tq, inner, heads, *a.pe);
    const DevFreqs& kpe = (a.k_pe != nullptr && a.k_pe->valid) ? *a.k_pe : *a.pe;
    ApplyRopeDev(c, kn.t(), batch, s, inner, heads, kpe);
  }

  DBuf attn(c.d, c.s, {batch * tq, inner});
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim_head)));
  for (int64_t b = 0; b < batch; ++b) {
    Tensor tq_t = RowsFrom(qn.t(), b * tq, inner, {tq, heads, dim_head});
    Tensor tk_t = RowsFrom(kn.t(), b * s, inner, {s, heads, dim_head});
    Tensor tv_t = RowsFrom(v.t(), b * s, inner, {s, heads, dim_head});
    Tensor to_t = RowsFrom(attn.t(), b * tq, inner, {tq, heads, dim_head});
    // Route on what the call MEANS, never on what its numbers happen to be — the
    // dispatch rule ltx2.cpp:857-867 established and the reason it is stated
    // there. `context == nullptr` is upstream's own self-attention marker.
    if (context == nullptr && a.bias == nullptr) {
      vt::AttentionArgs args;
      args.scale = scale;
      args.causal = false;
      // The FAST dense op, not `vt::Attention` (#1549). `vt::Attention` is frozen
      // on `vt::cuda::AttentionKernel`, which its own header calls
      // "Correctness-grade (M0.9)": one 256-thread block per (query, head), a
      // 256-wide shared-memory tree reduction per key, and no K/V tiling, so K
      // and V are re-read from global once per (query, head). At 768x448/49f the
      // video stream is 2352 tokens x 32 heads = 75,264 blocks each looping 2352
      // keys, x 48 layers, and ONE DiT forward measured 47.84 s on GB10. That
      // freeze is deliberate and correct — it is what keeps text decode
      // byte-identical (cuda_ops.cu:3119-3121) — but it means the fast kernels
      // are SEPARATE OPS a caller opts into BY NAME, with no shape routing and no
      // fallback notice. A model that never opts in gets correct output at
      // roughly 500x the cost and nothing anywhere says so.
      //
      // BOTH LTX STREAMS FIT THIS OP AT THE PRODUCTION DTYPE, and it is worth
      // writing the arithmetic down rather than citing the advertised bound. The
      // flash kernel's K/V tile is `2 * kFlashBc(64) * head_dim * sizeof(Tin)`
      // and it never opts into more than the 49,152 B of dynamic shared memory
      // every CUDA architecture guarantees. At the stream dtype this model
      // actually renders in — bf16, per the VT_CHECK in
      // `Ltx2StreamDitToDevice` — that is 32,768 B for the video stream's
      // head_dim 128 and 16,384 B for the audio stream's 64. Both fit, so no
      // cap-raise is needed here and #1549 does not make one; the op's
      // ADVERTISED head_dim domain is a separate defect owned by #1578.
      //
      // The f32 arm is the L2 parity reference, not a serving path, and at
      // head_dim 128 its tile is 65,536 B and does NOT fit. It is exercised at
      // the fixture's reduced dimensions, where it does fit; at production
      // geometry it now REFUSES rather than running slowly. It refuses LOUDLY in
      // either world -- today a throw from `Check(cudaGetLastError(),
      // "attention-dense-flash launch")` at cuda_ops.cu:3352, which names the op
      // but not the head_dim, and once #1578 lands a VT_CHECK that names the
      // head_dim too. Never silently, which is why this is a disclosure and not
      // a blocker. Recorded under `## Owed` in
      // .agents/specs/ltx25-dit-attn-flash.md and tracked by #1612.
      //
      // The square-problem contract holds here by construction: this branch is
      // entered only when `context == nullptr`, which is exactly when `s == tq`
      // above.
      //
      // NOT bit-identical to `vt::Attention` on CUDA, at either fast rung and for
      // two DIFFERENT reasons. The flash-tiled kernel groups the head_dim partial
      // sums across 32 lanes instead of a 256-thread block, so the same f32
      // online softmax associates differently. FA-2 goes further: `mma.sync`
      // reassociates BOTH the QK^T and the P.V reductions, and it exponentiates
      // with `exp2f` on a log2-scaled score where the scalar kernels use `expf` on
      // a linearly-scaled one. Every rung IS byte-identical on CPU, where all
      // three ops resolve to the same registered kernel. The binding gates are the
      // CUDA host-vs-device parity case in test_ltx2_device.cpp and, for the
      // FA-2-versus-flash deviation at this model's own geometry, the head_dim-128
      // cases in tests/vt/test_ops_attention_dense_fa2.cpp.
      //
      // THE DEFAULT IS NOW `vt::AttentionDenseFa2` (#1551), and the rung above
      // `AttentionDenseFlash` is the point of it. `AttentionDenseFlash` removed
      // the redundant global K/V traffic, but it is still a SCALAR recurrence:
      // one warp per query walking the keys through a dependent online-softmax
      // chain, with no tensor core anywhere in it. `AttentionDenseFa2` runs the
      // vendored FlashAttention-2 forward, whose QK^T and P.V are `mma.sync`
      // block reductions — the kernel vLLM itself dispatches for a dense
      // non-causal self-attention. Until #1551 that op refused every head dim but
      // 64, so LTX's head_dim-128 video stream could not reach it at all.
      //
      // CALLING IT UNCONDITIONALLY IS SAFE BY THE OP'S OWN CONTRACT, and that is
      // why there is no shape test here. `vt::AttentionDenseFa2` is TOTAL: its
      // fast path is bf16 / head_dim {64,128} / non-causal / MHA with the
      // vendored kernels compiled, and every other shape falls through to
      // `AttentionDenseFlash` BIT-exactly (cuda_ops.cu, and the four fall-through
      // cases in tests/vt/test_ops_attention_dense_fa2.cpp). Both LTX streams are
      // inside the fast path at the production dtype — video 32 heads x 128,
      // audio 32 heads x 64, bf16, non-causal, h_k == h — and the f32 parity arm
      // and the CPU backend take the fall-through and are unchanged by this. A
      // shape test written here would be a SECOND copy of the op's domain, and
      // the copy is what goes stale when the instantiation set moves.
      //
      // THE A/B KNOB IS THREE-WAY, because a two-arm knob cannot measure a
      // three-rung ladder. Each value selects a DIFFERENT vt:: op, so each arm
      // states which rung it ran in its own `VT_OP_PROVIDER_STATS=1` log rather
      // than being inferred from its timing — which is the quantity under
      // measurement and cannot also be the evidence:
      //
      //   unset (default)  vt::AttentionDenseFa2      kAttentionDenseFa2
      //   "flash"          vt::AttentionDenseFlash    kAttentionDenseFlash
      //   "0"              vt::Attention              kAttention
      //
      // `=0` keeps exactly the meaning #1549 gave it and docs/ENVIRONMENT.md
      // records, so the 47.84 s denominator remains reachable from this binary.
      // "flash" is the arm #1551's ratio is taken against. Read FRESH rather than
      // cached, so a test can flip it inside one process.
      //
      // EVERY ARM IS AN EXACT MATCH AND A FOURTH VALUE IS REFUSED (#1751). This
      // read used to test the naive arm with `arm[0] == '0'` — a PREFIX — while
      // the flash arm used `strcmp`, and everything matching neither fell into a
      // bare `else` that ran the FA-2 default in silence. Two consequences, and
      // both of them make a measurement say the wrong thing rather than fail:
      // `0x`, `07` and `0flash` selected the naive rung, and `falsh`, `FLASH`,
      // `flash ` with a trailing space, `naive`, `1` or an empty value selected
      // the DEFAULT. `flash` is the DENOMINATOR of #1551's 2.74x, so a mistyped
      // denominator ran the NUMERATOR's kernel a second time and yielded ~1.00x
      // — which is also exactly what "no speedup" looks like, so the number could
      // not report its own failure. A knob whose whole purpose is to say which
      // rung ran must not answer a question it was not asked, so an unrecognised
      // value is REFUSED BY NAME, in the shape AGENTS.md requires of an
      // unimplemented arm.
      //
      // AN EMPTY VALUE IS REFUSED, not treated as unset, and that is the one
      // choice here that is not forced. `export VLLM_LTX2_DIT_FLASH_ATTN=$ARM`
      // with an unset `ARM` produces exactly this, and it is the case where the
      // operator most believes they selected an arm. Silently defaulting is what
      // this whole change exists to remove.
      //
      // UNSET STILL MEANS FA-2, and that is deliberate rather than inherited:
      // it is the shipped default and the only serving arm (docs/ENVIRONMENT.md),
      // so requiring the variable would make every production render — none of
      // which sets it — refuse.
      const char* arm = std::getenv("VLLM_LTX2_DIT_FLASH_ATTN");
      if (arm == nullptr) {
        vt::AttentionDenseFa2(c.d.q, to_t, tq_t, tk_t, tv_t, args);
      } else if (std::strcmp(arm, "0") == 0) {
        // VT-ATTN-NAIVE: the naive arm of a same-binary A/B, not a serving path.
        // The default is `vt::AttentionDenseFa2` in the unset branch above, and
        // this call exists so every arm of the 47.84 s / 7.680 s / FA-2
        // measurement runs from ONE build — the shape `VT_FA2_DENSE`
        // (cuda_ops.cu) already takes. Deleting it would not remove a naive call
        // from production; it would remove the control that proves the fast one
        // is what runs.
        vt::Attention(c.d.q, to_t, tq_t, tk_t, tv_t, args);
      } else if (std::strcmp(arm, "flash") == 0) {
        // The #1549 default, kept as the DENOMINATOR arm of #1551's ratio. Not a
        // serving path either: it is the rung this change claims to beat, and a
        // claim measured against a rung nobody can still run is not measurable.
        vt::AttentionDenseFlash(c.d.q, to_t, tq_t, tk_t, tv_t, args);
      } else {
        throw std::invalid_argument(
            std::string("VLLM_LTX2_DIT_FLASH_ATTN=\"") + arm +
            "\" is not a rung of the LTX-2.5 DiT attention A/B. It accepts exactly three "
            "values: unset selects vt::AttentionDenseFa2, the shipped default and the only "
            "serving arm; \"flash\" selects vt::AttentionDenseFlash, the denominator arm; "
            "\"0\" selects vt::Attention, the naive arm. REFUSED rather than defaulted, "
            "because a value that quietly falls back runs one rung under another rung's "
            "name and makes a same-binary A/B measure the same kernel twice.");
      }
    } else {
      vt::AttentionCrossArgs args;
      args.scale = scale;
      Tensor bias;
      if (a.bias != nullptr) {
        bias = RowsFrom(*a.bias, b * a.bias_rows, s, {a.bias_rows, s});
      }
      vt::AttentionCross(c.d.q, to_t, tq_t, tk_t, tv_t, a.bias != nullptr ? &bias : nullptr, args);
    }
  }

  return AttentionEpilogueDev(c, w, x, std::move(attn), a);
}

// ---------------------------------------------------------------------------
// Leaf bricks
// ---------------------------------------------------------------------------

// Ltx2FeedForward (ltx2.cpp:782-789): net.0.proj -> gelu(tanh) -> net.2.
DBuf FeedForwardDev(Ctx& c, const Ltx2FeedForwardWeights& w, const Tensor& x, int64_t rows,
                    int64_t dim, int64_t inner) {
  DBuf hidden(c.d, c.s, {rows, inner});
  LinearDev(c, x, rows, dim, w.proj_in, hidden.t());
  Tensor h = Reshape(hidden.t(), {rows, inner});
  vt::GeluTanh(c.d.q, h, h);
  DBuf out(c.d, c.s, {rows, dim});
  LinearDev(c, hidden.t(), rows, inner, w.proj_out, out.t());
  return out;
}

struct AdalnOutDev {
  std::optional<DBuf> modulation;
  std::optional<DBuf> embedded;
};

// Ltx2AdaLayerNormSingle (ltx2.cpp:745-778). The 256-channel sinusoidal
// projection is built on the HOST (one [count, 256] table per stream per
// forward); everything after it is a device GEMM or a device elementwise pass.
AdalnOutDev AdaLayerNormSingleDev(Ctx& c, const Ltx2AdaLayerNormSingleWeights& w,
                                  const float* timesteps, int64_t count, int64_t dim) {
  // get_timestep_embedding (timestep_embedding.py:6-54) with num_channels=256,
  // flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000, so
  // the row is [cos(...) | sin(...)].
  const int64_t kChannels = 256;
  const int64_t half = kChannels / 2;
  std::vector<float> proj(static_cast<size_t>(count * kChannels));
  for (int64_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(-std::log(10000.0)) * static_cast<float>(i) / static_cast<float>(half);
    const float freq = std::exp(exponent);
    for (int64_t r = 0; r < count; ++r) {
      const float v = timesteps[r] * freq;
      proj[static_cast<size_t>(r * kChannels + i)] = std::cos(v);
      proj[static_cast<size_t>(r * kChannels + half + i)] = std::sin(v);
    }
  }
  DBuf d_proj = UploadStream(c.d, c.s, proj.data(), {count, kChannels});

  // TimestepEmbedding.forward (timestep_embedding.py:84-96): linear_1 -> SiLU -> linear_2.
  DBuf hidden(c.d, c.s, {count, dim});
  LinearDev(c, d_proj.t(), count, kChannels, w.linear_1, hidden.t());
  c.k->silu(c.d.q, hidden.t().data, count * dim, c.s);
  AdalnOutDev out;
  out.embedded.emplace(c.d, c.s, std::vector<int64_t>{count, dim});
  LinearDev(c, hidden.t(), count, dim, w.linear_2, out.embedded->t());

  // AdaLayerNormSingle.forward (adaln.py:44-45): linear(silu(embedded_timestep)).
  DBuf activated(c.d, c.s, {count, dim});
  c.d.b.Copy(c.d.q, activated.ptr(), out.embedded->ptr(), activated.bytes());
  c.k->silu(c.d.q, activated.t().data, count * dim, c.s);
  const int64_t coefficient_dim = w.linear.weight.shape[0];
  out.modulation.emplace(c.d, c.s, std::vector<int64_t>{count, coefficient_dim});
  LinearDev(c, activated.t(), count, dim, w.linear, out.modulation->t());
  return out;
}

// ---------------------------------------------------------------------------
// One BasicAVTransformerBlock (transformer.py:254-417)
// ---------------------------------------------------------------------------

struct BlockArgsDev {
  int64_t batch = 1;
  int64_t video_tokens = 0, audio_tokens = 0;
  int64_t video_context_tokens = 0, audio_context_tokens = 0;
  bool video_enabled = true, audio_enabled = true;
  const Tensor* video_timestep_modulation = nullptr;
  const Tensor* audio_timestep_modulation = nullptr;
  // The prompt-side AdaLN modulation, [batch, 2 * width] — nullptr is upstream's
  // `prompt_timestep is None` (transformer.py:442), i.e. the flag is off.
  const Tensor* video_prompt_modulation = nullptr;
  const Tensor* audio_prompt_modulation = nullptr;
  const Tensor* video_cross_scale_shift = nullptr;
  const Tensor* video_cross_gate = nullptr;
  const Tensor* audio_cross_scale_shift = nullptr;
  const Tensor* audio_cross_gate = nullptr;
  const Tensor* video_context = nullptr;
  const Tensor* audio_context = nullptr;
  const Tensor* video_context_bias = nullptr;
  const Tensor* audio_context_bias = nullptr;
  const Tensor* video_self_bias = nullptr;
  int64_t video_self_bias_rows = 0;
  const Tensor* audio_self_bias = nullptr;
  int64_t audio_self_bias_rows = 0;
  const DevFreqs* video_pe = nullptr;
  const DevFreqs* audio_pe = nullptr;
  const DevFreqs* video_cross_pe = nullptr;
  const DevFreqs* audio_cross_pe = nullptr;
  // This block's four perturbation flags, the device twin of `Ltx2BlockArgs`'s
  // (ltx2.h:653-675). `BlockPerturbationsProcessor` (transformer_args.py:99-118)
  // resolves them per block from the batched config; at `batch == 1`
  // `all_in_batch` is the flag itself.
  bool video_self_attn_perturbed = false;
  bool audio_self_attn_perturbed = false;
  // `cross_attn_skip_all` on the VIDEO args, i.e. SKIP_A2V_CROSS_ATTN. The flag
  // rides on the stream being WRITTEN (model.py:442-458), so this one gates the
  // audio->video direction.
  bool video_cross_attn_skip_all = false;
  bool audio_cross_attn_skip_all = false;
};

// One stream's text cross-attention (transformer.py:223-252 + :420-447), the
// device twin of ltx2_dit.cpp's TextCrossAttention.
void TextCrossAttentionDev(Ctx& c, const Ltx2AttentionWeights& attn, const Tensor& sst,
                           const Tensor& prompt_table, const Tensor& modulation,
                           const Tensor* prompt_modulation,
                           const Tensor& x_normed, const Tensor& context,
                           const Tensor* context_bias, int64_t batch, int64_t tokens,
                           int64_t context_tokens, int64_t width, int64_t heads,
                           int64_t dim_head, Tensor& x) {
  const int64_t coefficient = c.p->adaln_embedding_coefficient();
  VT_CHECK(c.p->cross_attention_adaln,
           "ltx2: cross_attention_adaln=false is upstream's plain cross-attention path "
           "(transformer.py:252); LTX-2.5 sets it true and only that arm is gated");
  const int64_t rows = batch * tokens;
  // slice(6, 9) — shift_q, scale_q, gate. NOT the self-attention slice(0, 3) and
  // NOT the feed-forward slice(3, 6) (transformer.py:240).
  DBuf shift_q = AdaValueDev(c, sst, modulation, rows, width, coefficient, 6);
  DBuf scale_q = AdaValueDev(c, sst, modulation, rows, width, coefficient, 7);
  DBuf gate = AdaValueDev(c, sst, modulation, rows, width, coefficient, 8);

  DBuf attn_input(c.d, c.s, {rows, width});
  c.d.b.Copy(c.d.q, attn_input.ptr(), x_normed.data, attn_input.bytes());
  c.k->modulate(c.d.q, attn_input.t().data, scale_q.t().data, shift_q.t().data, rows, width, width,
                c.s, c.s);

  // apply_cross_attention_adaln (transformer.py:420-447): the STATIC [2, dim]
  // per-block table (:441), plus the prompt-side AdaLN row when the flag is on
  // (:442-443).
  VT_CHECK(prompt_table.dtype == DType::kF32,
           "ltx2 device: the prompt scale-shift table is F32 in the checkpoint and is read as "
           "F32 here; a narrowed table would be the dtype rule applied backwards");
  DBuf encoder(c.d, c.s, {batch * context_tokens, width});
  c.d.b.Copy(c.d.q, encoder.ptr(), context.data, encoder.bytes());
  if (prompt_modulation == nullptr) {
    // No timestep term at all. `src_row_stride = 0` is the broadcast of the
    // table's single row over every token, and the table is read at F32 while the
    // stream stays at c.s — which is the whole reason `modulate` carries a
    // separate src_dtype.
    auto* table = prompt_table.Ptr<float>();
    c.k->modulate(c.d.q, encoder.t().data, table + width, table, batch * context_tokens, width, 0,
                  c.s, DType::kF32);
  } else {
    // `kv_modulation = table[None, None] + prompt_timestep.reshape(B, 1, 2, -1)`
    // (:441-443) is EXACTLY `ada_value`'s `table[row] + modulation[r, row]` over
    // a 2-parameter modulation with one row per BATCH element, so the sum is
    // formed here the way every other table+modulation sum in this file is —
    // before `(1 + scale)` applies, which is the order upstream rounds in.
    DBuf shift_kv = AdaValueDev(c, prompt_table, *prompt_modulation, batch, width,
                                /*num_params=*/2, /*index=*/0);
    DBuf scale_kv = AdaValueDev(c, prompt_table, *prompt_modulation, batch, width,
                                /*num_params=*/2, /*index=*/1);
    // `modulate`'s `src_row_stride` is a single stride, so it can broadcast ONE
    // row over every token (stride 0) or give every row its own (stride width) —
    // but not "row b for this batch element's context_tokens rows". The batch
    // loop supplies that offset rather than widening the kernel's contract for a
    // dimension that is 1 or 2 in every shipped call.
    const int64_t elem = static_cast<int64_t>(vt::SizeOf(c.s));
    for (int64_t b = 0; b < batch; ++b) {
      void* dst = static_cast<uint8_t*>(encoder.ptr()) + b * context_tokens * width * elem;
      const void* sc = static_cast<const uint8_t*>(scale_kv.t().data) + b * width * elem;
      const void* sh = static_cast<const uint8_t*>(shift_kv.t().data) + b * width * elem;
      c.k->modulate(c.d.q, dst, sc, sh, context_tokens, width, 0, c.s, c.s);
    }
  }

  AttnArgsDev a;
  a.batch = batch;
  a.tokens = tokens;
  a.context_tokens = context_tokens;
  a.query_dim = width;
  a.context_dim = width;
  a.heads = heads;
  a.dim_head = dim_head;
  a.bias = context_bias;
  a.bias_rows = context_bias != nullptr ? 1 : 0;
  DBuf out = AttentionDev(c, attn, attn_input.t(), &encoder.t(), a);
  c.k->add_gated(c.d.q, x.data, out.t().data, gate.t().data, rows, width, 1, c.s);
}

void BlockForwardDev(Ctx& c, const Ltx2BlockWeights& w, const BlockArgsDev& args, Tensor* video_x,
                     Tensor* audio_x) {
  const int64_t batch = args.batch;
  const int64_t dim = c.p->inner_dim();
  const int64_t adim = c.p->audio_inner_dim();
  const int64_t tv = args.video_tokens;
  const int64_t ta = args.audio_tokens;
  const int64_t coefficient = c.p->adaln_embedding_coefficient();

  // transformer.py:265-269.
  const bool run_vx = args.video_enabled && video_x != nullptr && tv > 0;
  const bool run_ax = args.audio_enabled && audio_x != nullptr && ta > 0;
  const bool run_a2v = run_vx && audio_x != nullptr && ta > 0;
  const bool run_v2a = run_ax && video_x != nullptr && tv > 0;

  std::optional<DBuf> vx_normed, ax_normed;

  if (run_vx) {
    const int64_t rows = batch * tv;
    // slice(0, 3) — shift, scale, gate for the self-attention (transformer.py:272-273).
    DBuf shift = AdaValueDev(c, w.scale_shift_table, *args.video_timestep_modulation, rows, dim,
                             coefficient, 0);
    DBuf scale = AdaValueDev(c, w.scale_shift_table, *args.video_timestep_modulation, rows, dim,
                             coefficient, 1);
    DBuf gate = AdaValueDev(c, w.scale_shift_table, *args.video_timestep_modulation, rows, dim,
                            coefficient, 2);
    DBuf norm_vx = AdaZeroDev(c, *video_x, scale, shift, rows, dim);

    AttnArgsDev a;
    a.batch = batch;
    a.tokens = tv;
    a.context_tokens = tv;
    a.query_dim = dim;
    a.context_dim = dim;
    a.heads = c.p->num_attention_heads;
    a.dim_head = c.p->attention_head_dim;
    a.pe = args.video_pe;
    a.bias = args.video_self_bias;
    a.bias_rows = args.video_self_bias_rows;
    a.all_perturbed = args.video_self_attn_perturbed;
    DBuf msa = AttentionDev(c, w.attn1, norm_vx.t(), nullptr, a);

    // PytorchPostSAFunction (ops.py:72-82): x + y * gate, then rms_norm of that sum.
    c.k->add_gated(c.d.q, video_x->data, msa.t().data, gate.t().data, rows, dim, 1, c.s);
    vx_normed.emplace(c.d, c.s, std::vector<int64_t>{rows, dim});
    RmsNormNoWeight(c, vx_normed->t(), *video_x, rows, dim);

    TextCrossAttentionDev(c, w.attn2, w.scale_shift_table, w.prompt_scale_shift_table,
                          *args.video_timestep_modulation, args.video_prompt_modulation,
                          vx_normed->t(), *args.video_context,
                          args.video_context_bias, batch, tv, args.video_context_tokens, dim,
                          c.p->num_attention_heads, c.p->attention_head_dim, *video_x);
  }

  if (run_ax) {
    const int64_t rows = batch * ta;
    DBuf shift = AdaValueDev(c, w.audio_scale_shift_table, *args.audio_timestep_modulation, rows,
                             adim, coefficient, 0);
    DBuf scale = AdaValueDev(c, w.audio_scale_shift_table, *args.audio_timestep_modulation, rows,
                             adim, coefficient, 1);
    DBuf gate = AdaValueDev(c, w.audio_scale_shift_table, *args.audio_timestep_modulation, rows,
                            adim, coefficient, 2);
    DBuf norm_ax = AdaZeroDev(c, *audio_x, scale, shift, rows, adim);

    AttnArgsDev a;
    a.batch = batch;
    a.tokens = ta;
    a.context_tokens = ta;
    a.query_dim = adim;
    a.context_dim = adim;
    a.heads = c.p->audio_num_attention_heads;
    a.dim_head = c.p->audio_attention_head_dim;
    a.pe = args.audio_pe;
    a.bias = args.audio_self_bias;
    a.bias_rows = args.audio_self_bias_rows;
    a.all_perturbed = args.audio_self_attn_perturbed;
    DBuf msa = AttentionDev(c, w.audio_attn1, norm_ax.t(), nullptr, a);

    c.k->add_gated(c.d.q, audio_x->data, msa.t().data, gate.t().data, rows, adim, 1, c.s);
    ax_normed.emplace(c.d, c.s, std::vector<int64_t>{rows, adim});
    RmsNormNoWeight(c, ax_normed->t(), *audio_x, rows, adim);

    TextCrossAttentionDev(c, w.audio_attn2, w.audio_scale_shift_table,
                          w.audio_prompt_scale_shift_table, *args.audio_timestep_modulation,
                          args.audio_prompt_modulation,
                          ax_normed->t(), *args.audio_context, args.audio_context_bias, batch, ta,
                          args.audio_context_tokens, adim, c.p->audio_num_attention_heads,
                          c.p->audio_attention_head_dim, *audio_x);
  }

  // Audio <-> video cross attention (transformer.py:329-397). Both directions
  // read the PRE-cross snapshots so the order of the two does not bias the result.
  if (run_a2v || run_v2a) {
    std::optional<DBuf> vx_pre, ax_pre;
    if (video_x != nullptr) {
      vx_pre.emplace(c.d, c.s, std::vector<int64_t>{batch * tv, dim});
      c.d.b.Copy(c.d.q, vx_pre->ptr(), video_x->data, vx_pre->bytes());
    }
    if (audio_x != nullptr) {
      ax_pre.emplace(c.d, c.s, std::vector<int64_t>{batch * ta, adim});
      c.d.b.Copy(c.d.q, ax_pre->ptr(), audio_x->data, ax_pre->bytes());
    }

    // get_av_ca_ada_values (transformer.py:202-221): SCALE comes first and SHIFT
    // second — the opposite order from the self-attention slice above.
    auto av_scale = [&](const Tensor& table, const Tensor& ss, int64_t tokens, int64_t width,
                        int64_t first) {
      return AdaValueDev(c, table, ss, batch * tokens, width, 4, first);
    };
    // The gate is row 4 of the same table, driven by the CROSS modality's sigma
    // and carrying a SINGLE token row broadcast over the sequence.
    auto av_gate = [&](const Tensor& table, const Tensor& gate_ts, int64_t width) {
      return AdaValueDev(c, table, gate_ts, batch, width, /*num_params=*/1, /*table_row=*/4,
                         /*mod_index=*/0);
    };

    // `if run_a2v and not video.cross_attn_skip_all` (transformer.py:335). The
    // guard is INSIDE the `run_a2v || run_v2a` block, exactly as upstream's is, so
    // a pass that skips one direction still took the `vx_pre` / `ax_pre` snapshot
    // above and the surviving direction reads the PRE-cross state. Hoisting it
    // into the outer condition would be equivalent only while both directions are
    // always skipped together, which is true of the one caller today and is not a
    // property of the flag.
    if (run_a2v && !args.video_cross_attn_skip_all) {
      DBuf scale_v = av_scale(w.scale_shift_table_a2v_ca_video, *args.video_cross_scale_shift, tv,
                              dim, 0);
      DBuf shift_v = av_scale(w.scale_shift_table_a2v_ca_video, *args.video_cross_scale_shift, tv,
                              dim, 1);
      DBuf gate = av_gate(w.scale_shift_table_a2v_ca_video, *args.video_cross_gate, dim);
      DBuf vq = AdaZeroDev(c, vx_pre->t(), scale_v, shift_v, batch * tv, dim);
      DBuf scale_a = av_scale(w.scale_shift_table_a2v_ca_audio, *args.audio_cross_scale_shift, ta,
                              adim, 0);
      DBuf shift_a = av_scale(w.scale_shift_table_a2v_ca_audio, *args.audio_cross_scale_shift, ta,
                              adim, 1);
      DBuf akv = AdaZeroDev(c, ax_pre->t(), scale_a, shift_a, batch * ta, adim);

      AttnArgsDev a;
      a.batch = batch;
      a.tokens = tv;
      a.context_tokens = ta;
      a.query_dim = dim;    // Q from the VIDEO stream
      a.context_dim = adim;  // K/V from the AUDIO stream
      a.heads = c.p->audio_num_attention_heads;
      a.dim_head = c.p->audio_attention_head_dim;
      a.pe = args.video_cross_pe;
      a.k_pe = args.audio_cross_pe;
      DBuf out = AttentionDev(c, w.audio_to_video_attn, vq.t(), &akv.t(), a);
      c.k->add_gated(c.d.q, video_x->data, out.t().data, gate.t().data, batch * tv, dim, tv, c.s);
    }

    // `if run_v2a and not audio.cross_attn_skip_all` (transformer.py:367).
    if (run_v2a && !args.audio_cross_attn_skip_all) {
      DBuf scale_a = av_scale(w.scale_shift_table_a2v_ca_audio, *args.audio_cross_scale_shift, ta,
                              adim, 2);
      DBuf shift_a = av_scale(w.scale_shift_table_a2v_ca_audio, *args.audio_cross_scale_shift, ta,
                              adim, 3);
      DBuf gate = av_gate(w.scale_shift_table_a2v_ca_audio, *args.audio_cross_gate, adim);
      DBuf aq = AdaZeroDev(c, ax_pre->t(), scale_a, shift_a, batch * ta, adim);
      DBuf scale_v = av_scale(w.scale_shift_table_a2v_ca_video, *args.video_cross_scale_shift, tv,
                              dim, 2);
      DBuf shift_v = av_scale(w.scale_shift_table_a2v_ca_video, *args.video_cross_scale_shift, tv,
                              dim, 3);
      DBuf vkv = AdaZeroDev(c, vx_pre->t(), scale_v, shift_v, batch * tv, dim);

      AttnArgsDev a;
      a.batch = batch;
      a.tokens = ta;
      a.context_tokens = tv;
      a.query_dim = adim;   // Q from the AUDIO stream
      a.context_dim = dim;  // K/V from the VIDEO stream
      a.heads = c.p->audio_num_attention_heads;
      a.dim_head = c.p->audio_attention_head_dim;
      a.pe = args.audio_cross_pe;
      a.k_pe = args.video_cross_pe;
      DBuf out = AttentionDev(c, w.video_to_audio_attn, aq.t(), &vkv.t(), a);
      c.k->add_gated(c.d.q, audio_x->data, out.t().data, gate.t().data, batch * ta, adim, ta, c.s);
    }
  }

  // Feed-forward, slice(3, 6) (transformer.py:399-415).
  if (run_vx) {
    const int64_t rows = batch * tv;
    DBuf shift = AdaValueDev(c, w.scale_shift_table, *args.video_timestep_modulation, rows, dim,
                             coefficient, 3);
    DBuf scale = AdaValueDev(c, w.scale_shift_table, *args.video_timestep_modulation, rows, dim,
                             coefficient, 4);
    DBuf gate = AdaValueDev(c, w.scale_shift_table, *args.video_timestep_modulation, rows, dim,
                            coefficient, 5);
    DBuf scaled = AdaZeroDev(c, *video_x, scale, shift, rows, dim);
    DBuf ff = FeedForwardDev(c, w.ff, scaled.t(), rows, dim, 4 * dim);
    c.k->add_gated(c.d.q, video_x->data, ff.t().data, gate.t().data, rows, dim, 1, c.s);
  }

  if (run_ax) {
    const int64_t rows = batch * ta;
    DBuf shift = AdaValueDev(c, w.audio_scale_shift_table, *args.audio_timestep_modulation, rows,
                             adim, coefficient, 3);
    DBuf scale = AdaValueDev(c, w.audio_scale_shift_table, *args.audio_timestep_modulation, rows,
                             adim, coefficient, 4);
    DBuf gate = AdaValueDev(c, w.audio_scale_shift_table, *args.audio_timestep_modulation, rows,
                            adim, coefficient, 5);
    DBuf scaled = AdaZeroDev(c, *audio_x, scale, shift, rows, adim);
    DBuf ff = FeedForwardDev(c, w.audio_ff, scaled.t(), rows, adim, 4 * adim);
    c.k->add_gated(c.d.q, audio_x->data, ff.t().data, gate.t().data, rows, adim, 1, c.s);
  }
}

// ---------------------------------------------------------------------------
// LTXModel.forward (model.py:492-538)
// ---------------------------------------------------------------------------

// One stream's prepared inputs, all device-resident (TransformerArgs,
// transformer_args.py:46-70). The device twin of ltx2_dit.cpp's PreparedStream.
struct PreparedStreamDev {
  std::optional<DBuf> x;
  std::optional<DBuf> modulation;
  std::optional<DBuf> embedded;
  std::optional<DBuf> context;
  std::optional<DBuf> context_bias;  // f32
  std::optional<DBuf> self_bias;     // f32
  int64_t self_bias_rows = 0;
  DevFreqs pe, cross_pe;
  std::optional<DBuf> cross_scale_shift;
  std::optional<DBuf> cross_gate;
  // [batch, 2 * width] — empty when use_prompt_adaln_single is false.
  std::optional<DBuf> prompt_modulation;
};

// _prepare_timestep (transformer_args.py:173-186) + AdaLayerNormSingle.
AdalnOutDev PrepareTimestepDev(Ctx& c, const Ltx2AdaLayerNormSingleWeights& adaln,
                               const float* timesteps, int64_t count, int64_t width,
                               int64_t multiplier) {
  std::vector<float> scaled(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    scaled[static_cast<size_t>(i)] = timesteps[i] * static_cast<float>(multiplier);
  }
  return AdaLayerNormSingleDev(c, adaln, scaled.data(), count, width);
}

PreparedStreamDev PrepareStreamDev(Ctx& c, const Ltx2LinearWeight& patchify,
                                   const Ltx2AdaLayerNormSingleWeights& adaln,
                                   const Ltx2AdaLayerNormSingleWeights& cross_scale_shift_adaln,
                                   const Ltx2AdaLayerNormSingleWeights& cross_gate_adaln,
                                   const Ltx2AdaLayerNormSingleWeights* prompt_adaln,
                                   const Tensor* keyframes_embedding,
                                   const Ltx2ModalityInput& m, int64_t width, int64_t in_channels,
                                   int64_t n_pos_dims, const std::vector<int64_t>& max_pos,
                                   int64_t heads, const Ltx2ModalityInput* cross,
                                   int64_t context_dim) {
  PreparedStreamDev out;
  const int64_t rows = m.batch * m.tokens;

  // transformer_args.py:268 — x = patchify_proj(latent).
  DBuf latent = UploadStream(c.d, c.s, m.latent, {rows, in_channels});
  out.x.emplace(c.d, c.s, std::vector<int64_t>{rows, width});
  LinearDev(c, latent.t(), rows, in_channels, patchify, out.x->t());

  // transformer_args.py:269 — `apply_keyframes_absolute_embedding`, in the same
  // place and with the same polarity the host forward uses. Not a second rule:
  // the two paths differ only in where the bytes live.
  VT_CHECK(m.keyframes_mask == nullptr || keyframes_embedding != nullptr,
           "ltx2: a keyframes_mask was supplied for a stream that carries no "
           "keyframes_abs_pos_embedding (model.py:333 builds the AUDIO preprocessor with no "
           "keyframes_embedding_provider; :314 gives the video one). Refusing rather than "
           "rendering without a trained term.");
  if (keyframes_embedding != nullptr && m.keyframes_mask != nullptr) {
    // Same pairing the host arm asserts: a declared flag and an unbound view
    // cannot disagree, because together they ARE
    // `supports_keyframes_abs_pos_embedding` (model.py:166-173). `CheckResident`
    // deliberately no-ops on an unbound view, so it does not cover this.
    VT_CHECK(keyframes_embedding->data != nullptr,
             "ltx2: use_keyframes_abs_pos_embedding is set but the staged weight view is unbound");
    VT_CHECK(keyframes_embedding->rank == 2 && keyframes_embedding->shape[0] == 1 &&
                 keyframes_embedding->shape[1] == width,
             "ltx2: keyframes_abs_pos_embedding must be [1, stream width] (model.py:217-219)");
    // `mask * embedding` cast to `hidden_states.dtype` (transformer_args.py:42-43)
    // — so the ADDEND is at the stream dtype, not wider. The staged tensor already
    // is, because `Ltx2StreamDitToDevice` stages the whole contract at one dtype;
    // asserted rather than assumed because vt's CUDA elementwise kernels require
    // matching dtypes and would throw from inside a kernel instead of by name.
    VT_CHECK(keyframes_embedding->dtype == out.x->t().dtype,
             "ltx2: keyframes_abs_pos_embedding is staged at a different dtype from the video "
             "stream; upstream casts the embedding to hidden_states.dtype and never widens it");
    // vt::Add's ROW-BROADCAST shape is rank-1 [D] (ops.h:1887-1889), which is
    // exactly `mask * embedding` over a run of MARKED rows. Walking contiguous
    // runs keeps the general mask exact while the first-frame rule — a prefix
    // (tools.py:194-195) — costs a single call.
    const Tensor bias =
        MakeTensor(keyframes_embedding->data, keyframes_embedding->dtype,
                   keyframes_embedding->device, {width});
    int64_t r = 0;
    while (r < rows) {
      if (!(m.keyframes_mask[r] > 0.0F)) {
        ++r;
        continue;
      }
      int64_t end = r;
      while (end < rows && m.keyframes_mask[end] > 0.0F) ++end;
      Tensor run = RowsFrom(out.x->t(), r, width, {end - r, width});
      vt::Add(c.d.q, run, run, bias);
      r = end;
    }
  }

  AdalnOutDev ada =
      PrepareTimestepDev(c, adaln, m.timesteps, rows, width, c.p->timestep_scale_multiplier);
  out.modulation = std::move(ada.modulation);
  out.embedded = std::move(ada.embedded);

  // transformer_args.py:274-277 — the PROMPT-side AdaLN runs on this modality's
  // own SIGMA, [batch], not on its per-token `timesteps`.
  if (prompt_adaln != nullptr) {
    VT_CHECK(m.sigma != nullptr,
             "ltx2: use_prompt_adaln_single=true needs this modality's sigma "
             "(transformer_args.py:274-277); it drives the prompt-side AdaLN MLP whose output is "
             "added to the cross-attention K/V modulation, and a missing sigma would silently "
             "fall back to the static table");
    AdalnOutDev prompt = PrepareTimestepDev(c, *prompt_adaln, m.sigma, m.batch, width,
                                            c.p->timestep_scale_multiplier);
    out.prompt_modulation = std::move(prompt.modulation);
  }

  if (m.context != nullptr && m.context_tokens > 0) {
    out.context = UploadStream(c.d, c.s, m.context, {m.batch * m.context_tokens, context_dim});
  }
  if (m.context_mask != nullptr) {
    // f32 by contract: vt::AttentionCross takes an f32 additive bias, and the
    // `(mask - 1) * finfo(f32).max` this produces is a magnitude a bf16 store
    // would not survive intact.
    const std::vector<float> bias = Ltx2PrepareContextMask(m.context_mask, m.batch,
                                                           m.context_tokens);
    out.context_bias.emplace(c.d, DType::kF32,
                             std::vector<int64_t>{m.batch, m.context_tokens}, bias.data());
  }
  if (m.attention_mask != nullptr) {
    VT_CHECK(m.attention_mask_rows == 1 || m.attention_mask_rows == m.tokens,
             "ltx2: attention_mask rows must be 1 (key-only) or the token count");
    const std::vector<float> bias = Ltx2PrepareSelfAttentionMask(
        m.attention_mask, m.batch * m.attention_mask_rows * m.tokens);
    out.self_bias.emplace(
        c.d, DType::kF32,
        std::vector<int64_t>{m.batch * m.attention_mask_rows, m.tokens}, bias.data());
    out.self_bias_rows = m.attention_mask_rows;
  }

  const Ltx2FreqsCis pe = Ltx2PrecomputeFreqsCis(
      m.positions, m.batch, m.tokens, n_pos_dims, n_pos_dims, c.p->use_middle_indices_grid, width,
      max_pos, c.p->positional_embedding_theta, heads, c.p->rope_type,
      c.p->double_precision_rope);
  out.pe = UploadFreqs(c.d, pe, c.p->rope_type);

  if (cross != nullptr) {
    VT_CHECK(cross->sigma != nullptr,
             "ltx2: the cross modality must supply sigma (transformer_args.py:373-379)");
    // transformer_args.py:364-371 — the cross RoPE is built from the TIME axis
    // only, at audio_cross_attention_dim, and ALWAYS with the middle-indices grid
    // regardless of the model's own flag.
    const Ltx2FreqsCis cpe = Ltx2PrecomputeFreqsCis(
        m.positions, m.batch, m.tokens, /*n_pos_dims=*/1, n_pos_dims, true,
        c.p->audio_cross_attention_dim, {c.p->cross_pe_max_pos()},
        c.p->positional_embedding_theta, heads, c.p->rope_type, c.p->double_precision_rope);
    out.cross_pe = UploadFreqs(c.d, cpe, c.p->rope_type);

    // _prepare_cross_attention_timestep (transformer_args.py:388-411).
    AdalnOutDev css = PrepareTimestepDev(c, cross_scale_shift_adaln, m.timesteps, rows, width,
                                         c.p->timestep_scale_multiplier);
    out.cross_scale_shift = std::move(css.modulation);
    const float factor = static_cast<float>(c.p->av_ca_timestep_scale_multiplier) /
                         static_cast<float>(c.p->timestep_scale_multiplier);
    std::vector<float> gate_ts(static_cast<size_t>(m.batch));
    for (int64_t b = 0; b < m.batch; ++b) {
      gate_ts[static_cast<size_t>(b)] =
          cross->sigma[b] * static_cast<float>(c.p->timestep_scale_multiplier) * factor;
    }
    AdalnOutDev gate = AdaLayerNormSingleDev(c, cross_gate_adaln, gate_ts.data(), m.batch, width);
    out.cross_gate = std::move(gate.modulation);
  }
  return out;
}

// _process_output (model.py:472-490).
std::vector<float> ProcessOutputDev(Ctx& c, const Tensor& table, const Ltx2LinearWeight& proj,
                                    const Tensor& x, const DBuf& embedded, int64_t rows,
                                    int64_t width, int64_t out_channels) {
  CheckTableF32(table, "the output scale-shift table");
  DBuf normed(c.d, c.s, {rows, width});
  vt::LayerNormArgs args;
  args.eps = static_cast<float>(c.p->norm_eps);
  Tensor o = Reshape(normed.t(), {rows, width});
  Tensor i = Reshape(x, {rows, width});
  // torch.nn.LayerNorm(elementwise_affine=False) (model.py:231, :261).
  vt::LayerNorm(c.d.q, o, i, nullptr, nullptr, args);
  c.k->output_modulate(c.d.q, normed.t().data, table.Ptr<float>(), embedded.t().data, rows, width,
                       c.s);
  DBuf out(c.d, c.s, {rows, out_channels});
  LinearDev(c, normed.t(), rows, width, proj, out.t());
  return DownloadF32(c.d, out, rows * out_channels);
}

// Every weight the forward will read must live on the queue's device. A host view
// handed to a device GEMM does not throw on GB10 — unified memory addresses it —
// it just runs the model at host bandwidth while every gate stays green. So the
// binding is checked once, by name, before a single kernel launches.
void CheckResident(const Tensor& t, vt::Device dev, const char* what) {
  if (t.data == nullptr) return;
  VT_CHECK(t.device.type == dev.type && t.device.index == dev.index,
           std::string("ltx2 device forward: '") + what +
               "' is not resident on the queue's device. Stage the DiT with "
               "Ltx2StreamDitToDevice or Ltx2StageDitWeightsToDevice; a host view would run "
               "at host bandwidth on unified memory and silently pass every gate.");
}

void CheckLinearResident(const Ltx2LinearWeight& w, vt::Device dev, const char* what) {
  CheckResident(w.weight, dev, what);
  CheckResident(w.bias, dev, what);
}

void CheckAttnResident(const Ltx2AttentionWeights& w, vt::Device dev, const char* what) {
  CheckLinearResident(w.to_q, dev, what);
  CheckLinearResident(w.to_k, dev, what);
  CheckLinearResident(w.to_v, dev, what);
  CheckLinearResident(w.to_gate_logits, dev, what);
  CheckLinearResident(w.to_out, dev, what);
  CheckResident(w.q_norm, dev, what);
  CheckResident(w.k_norm, dev, what);
}

void CheckAdalnResident(const Ltx2AdaLayerNormSingleWeights& w, vt::Device dev, const char* what) {
  CheckLinearResident(w.linear_1, dev, what);
  CheckLinearResident(w.linear_2, dev, what);
  CheckLinearResident(w.linear, dev, what);
}

void CheckWeightsResident(const Ltx2DitWeights& w, vt::Device dev) {
  CheckLinearResident(w.patchify_proj, dev, "patchify_proj");
  CheckLinearResident(w.proj_out, dev, "proj_out");
  CheckLinearResident(w.audio_patchify_proj, dev, "audio_patchify_proj");
  CheckLinearResident(w.audio_proj_out, dev, "audio_proj_out");
  CheckAdalnResident(w.adaln_single, dev, "adaln_single");
  CheckAdalnResident(w.audio_adaln_single, dev, "audio_adaln_single");
  // Bound only when use_prompt_adaln_single is on; `CheckResident` no-ops on an
  // unbound view, so this needs no flag and cannot go stale against one.
  CheckAdalnResident(w.prompt_adaln_single, dev, "prompt_adaln_single");
  CheckAdalnResident(w.audio_prompt_adaln_single, dev, "audio_prompt_adaln_single");
  CheckAdalnResident(w.av_ca_video_scale_shift, dev, "av_ca_video_scale_shift");
  CheckAdalnResident(w.av_ca_audio_scale_shift, dev, "av_ca_audio_scale_shift");
  CheckAdalnResident(w.av_ca_a2v_gate, dev, "av_ca_a2v_gate");
  CheckAdalnResident(w.av_ca_v2a_gate, dev, "av_ca_v2a_gate");
  CheckResident(w.scale_shift_table, dev, "scale_shift_table");
  CheckResident(w.audio_scale_shift_table, dev, "audio_scale_shift_table");
  // Bound only when use_keyframes_abs_pos_embedding resolves on; `CheckResident`
  // no-ops on an unbound view, so this needs no flag either.
  CheckResident(w.keyframes_abs_pos_embedding, dev, "keyframes_abs_pos_embedding");
  for (size_t i = 0; i < w.blocks.size(); ++i) {
    const Ltx2BlockWeights& b = w.blocks[i];
    const std::string tag = "blocks." + std::to_string(i);
    CheckAttnResident(b.attn1, dev, tag.c_str());
    CheckAttnResident(b.attn2, dev, tag.c_str());
    CheckAttnResident(b.audio_attn1, dev, tag.c_str());
    CheckAttnResident(b.audio_attn2, dev, tag.c_str());
    CheckAttnResident(b.audio_to_video_attn, dev, tag.c_str());
    CheckAttnResident(b.video_to_audio_attn, dev, tag.c_str());
    CheckLinearResident(b.ff.proj_in, dev, tag.c_str());
    CheckLinearResident(b.ff.proj_out, dev, tag.c_str());
    CheckLinearResident(b.audio_ff.proj_in, dev, tag.c_str());
    CheckLinearResident(b.audio_ff.proj_out, dev, tag.c_str());
    CheckResident(b.scale_shift_table, dev, tag.c_str());
    CheckResident(b.audio_scale_shift_table, dev, tag.c_str());
    CheckResident(b.prompt_scale_shift_table, dev, tag.c_str());
    CheckResident(b.audio_prompt_scale_shift_table, dev, tag.c_str());
    CheckResident(b.scale_shift_table_a2v_ca_video, dev, tag.c_str());
    CheckResident(b.scale_shift_table_a2v_ca_audio, dev, tag.c_str());
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

bool Ltx2DitTensorIsTable(const std::string& name) {
  // The `scale_shift_table` family, ALL SIX of it (ltx2.cpp:268-269, :294-300):
  //
  //     scale_shift_table                        audio_scale_shift_table
  //     <block>.scale_shift_table                <block>.audio_scale_shift_table
  //     <block>.prompt_scale_shift_table         <block>.audio_prompt_scale_shift_table
  //     <block>.scale_shift_table_a2v_ca_video   <block>.scale_shift_table_a2v_ca_audio
  //
  // These are the tensors the CHECKPOINT stores F32 (ltx2_loader.h:64-66), so they
  // stay F32 on every arm — narrowing a tensor the file itself widened would be
  // the dtype rule applied backwards — and, decisively, they are the ones the
  // glue kernels read through a `const float*` parameter.
  //
  // MEASURED, not assumed. An earlier version of this predicate matched a
  // SUFFIX, which silently excluded the two `_a2v_ca_*` tables because their name
  // does not END in `scale_shift_table`. They were then staged bf16 and read as
  // f32 by `ada_value`, which made the audio<->video cross gate 2.85e32 and the
  // whole video stream 6.89e30 after one block — while the f32 arm, where the
  // mismatch cannot arise, stayed green at 1e-7. A SUBSTRING match is what the
  // name family actually is.
  return name.find("scale_shift_table") != std::string::npos;
}

Ltx2DitDeviceWeights Ltx2StageDitWeightsToDevice(vt::Queue& queue, const Ltx2DitParams& params,
                                                 const std::map<std::string, vt::Tensor>& host,
                                                 vt::DType stream_dtype) {
  VT_CHECK(stream_dtype == vt::DType::kF32 || stream_dtype == vt::DType::kBF16,
           "ltx2: the device stream dtype is bf16 (production) or f32 (the L2 parity arm)");
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Ltx2DitDeviceWeights out;
  for (const auto& kv : host) {
    const vt::Tensor& t = kv.second;
    const bool table = Ltx2DitTensorIsTable(kv.first);
    const vt::DType want = table ? vt::DType::kF32 : stream_dtype;
    int64_t numel = 1;
    for (int r = 0; r < t.rank; ++r) numel *= t.shape[r];

    std::vector<uint8_t> staged(static_cast<size_t>(numel) * vt::SizeOf(want));
    if (t.dtype == want) {
      std::memcpy(staged.data(), t.data, staged.size());
    } else if (want == vt::DType::kBF16 && t.dtype == vt::DType::kF32) {
      auto* dst = reinterpret_cast<uint16_t*>(staged.data());
      const auto* src = t.Ptr<float>();
      for (int64_t i = 0; i < numel; ++i) dst[i] = RoundBf16(src[i]);
    } else if (want == vt::DType::kF32 && t.dtype == vt::DType::kBF16) {
      auto* dst = reinterpret_cast<float*>(staged.data());
      const auto* src = static_cast<const uint16_t*>(t.data);
      for (int64_t i = 0; i < numel; ++i) dst[i] = WidenBf16(src[i]);
    } else {
      VT_CHECK(false, "ltx2 staging: '" + kv.first + "' is neither f32 nor bf16");
    }

    void* dev = backend.Alloc(staged.size());
    backend.Copy(queue, dev, staged.data(), staged.size());
    backend.Synchronize(queue);  // `staged` dies at the end of this iteration
    out.storage.emplace_back(dev, [&backend](void* p) { backend.Free(p); });

    vt::Tensor view;
    view.data = dev;
    view.dtype = want;
    view.device = queue.device;
    view.rank = t.rank;
    int64_t acc = 1;
    for (int r = t.rank - 1; r >= 0; --r) {
      view.shape[r] = t.shape[r];
      view.stride[r] = acc;
      acc *= view.shape[r];
    }
    out.views[kv.first] = view;
  }
  out.weights = BindLtx2DitWeights(params, out.views);
  return out;
}

// ---------------------------------------------------------------------------
// The forward
// ---------------------------------------------------------------------------

Ltx2DitOutputs Ltx2DitForwardDevice(vt::Queue& queue, const Ltx2DitParams& params,
                                    const Ltx2DitWeights& weights,
                                    const Ltx2ModalityInput* video,
                                    const Ltx2ModalityInput* audio, vt::DType compute_dtype,
                                    Ltx2PromptKvCache* cache,
                                    const Ltx2DitPerturbation* perturbations) {
  VT_CHECK(compute_dtype == vt::DType::kF32 || compute_dtype == vt::DType::kBF16,
           "ltx2: the device forward computes in bf16 (the production stream, which is what "
           "Ltx2StreamDitToDevice puts on the device) or f32 (the L2 parity arm)");
  // LTX-2.5 is an LTXModelType.AudioVideo checkpoint; the VideoOnly / AudioOnly
  // types build a DIFFERENT parameter set, so they are refused by name here
  // exactly as the host forward refuses them.
  VT_CHECK(video != nullptr && audio != nullptr,
           "ltx2: the AudioVideo model type only; LTXModelType.VideoOnly and "
           "LTXModelType.AudioOnly carry a different weight contract and are not ported");
  VT_CHECK(cache == nullptr,
           "ltx2: the prompt K/V cache is not ported to the device forward yet and is REFUSED "
           "rather than ignored — an ignored cache would recompute correctly and quietly lose "
           "the reuse, which is exactly the divergence that is found a phase later. Pass "
           "nullptr, or use Ltx2DitForward for the cached host path.");

  const int64_t dim = params.inner_dim();
  const int64_t adim = params.audio_inner_dim();
  VT_CHECK(params.cross_attention_dim == dim,
           "ltx2: cross_attention_dim must equal the video stream width");
  // The audio half of the same invariant. `audio_cross_attention_dim` serves TWO
  // roles upstream (ltx2.h:96-99) — `audio_attn2`'s context width and the width
  // the audio<->video cross positional embedding is built at — and both are the
  // audio stream width. The host forward leaves this implicit; the device path
  // sizes a real allocation from it, so it is checked rather than assumed.
  VT_CHECK(params.audio_cross_attention_dim == adim,
           "ltx2: audio_cross_attention_dim must equal the audio stream width");
  VT_CHECK(static_cast<int64_t>(weights.blocks.size()) == params.num_layers,
           "ltx2: the bound block count does not match num_layers");
  // `perturbations` (model.py:493), checked exactly as the host forward checks it
  // (ltx2_dit.cpp:837-842). A vector that is not exactly `num_layers` long is
  // REFUSED rather than indexed defensively: a config built for another layer
  // count would otherwise perturb a prefix of the blocks and leave the rest
  // alone, which is a legal-looking STG pass over the wrong blocks and renders.
  if (perturbations != nullptr) {
    for (const std::vector<uint8_t>* v :
         {&perturbations->video_self_attn, &perturbations->audio_self_attn}) {
      VT_CHECK(v->empty() || static_cast<int64_t>(v->size()) == params.num_layers,
               "ltx2: a perturbation vector is neither empty nor one entry per block");
    }
  }
  CheckWeightsResident(weights, queue.device);

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  OnesCache ones(d);
  Ctx c{d, compute_dtype, Glue(d), &params, &ones};

  // model.py:222-226 / :252-256 — the module exists only when BOTH flags hold.
  const bool prompt_adaln = params.cross_attention_adaln && params.use_prompt_adaln_single;
  PreparedStreamDev vs = PrepareStreamDev(
      c, weights.patchify_proj, weights.adaln_single, weights.av_ca_video_scale_shift,
      weights.av_ca_a2v_gate, prompt_adaln ? &weights.prompt_adaln_single : nullptr,
      // model.py:314 — only the VIDEO preprocessor gets the provider.
      params.use_keyframes_abs_pos_embedding ? &weights.keyframes_abs_pos_embedding : nullptr,
      *video, dim,
      params.in_channels, 3,
      params.positional_embedding_max_pos, params.num_attention_heads, audio,
      params.cross_attention_dim);
  PreparedStreamDev as = PrepareStreamDev(
      c, weights.audio_patchify_proj, weights.audio_adaln_single, weights.av_ca_audio_scale_shift,
      weights.av_ca_v2a_gate, prompt_adaln ? &weights.audio_prompt_adaln_single : nullptr,
      // model.py:333 — the audio preprocessor is built with NO provider.
      /*keyframes_embedding=*/nullptr,
      *audio,
      adim, params.audio_in_channels, 1,
      params.audio_positional_embedding_max_pos, params.audio_num_attention_heads, video,
      params.audio_cross_attention_dim);

  for (int64_t i = 0; i < params.num_layers; ++i) {
    BlockArgsDev a;
    a.batch = video->batch;
    a.video_tokens = video->tokens;
    a.audio_tokens = audio->tokens;
    a.video_context_tokens = video->context_tokens;
    a.audio_context_tokens = audio->context_tokens;
    a.video_enabled = video->enabled;
    a.audio_enabled = audio->enabled;
    a.video_timestep_modulation = &vs.modulation->t();
    a.audio_timestep_modulation = &as.modulation->t();
    a.video_prompt_modulation = vs.prompt_modulation ? &vs.prompt_modulation->t() : nullptr;
    a.audio_prompt_modulation = as.prompt_modulation ? &as.prompt_modulation->t() : nullptr;
    a.video_cross_scale_shift = vs.cross_scale_shift ? &vs.cross_scale_shift->t() : nullptr;
    a.video_cross_gate = vs.cross_gate ? &vs.cross_gate->t() : nullptr;
    a.audio_cross_scale_shift = as.cross_scale_shift ? &as.cross_scale_shift->t() : nullptr;
    a.audio_cross_gate = as.cross_gate ? &as.cross_gate->t() : nullptr;
    a.video_context = vs.context ? &vs.context->t() : nullptr;
    a.audio_context = as.context ? &as.context->t() : nullptr;
    a.video_context_bias = vs.context_bias ? &vs.context_bias->t() : nullptr;
    a.audio_context_bias = as.context_bias ? &as.context_bias->t() : nullptr;
    a.video_self_bias = vs.self_bias ? &vs.self_bias->t() : nullptr;
    a.video_self_bias_rows = vs.self_bias_rows;
    a.audio_self_bias = as.self_bias ? &as.self_bias->t() : nullptr;
    a.audio_self_bias_rows = as.self_bias_rows;
    a.video_pe = &vs.pe;
    a.audio_pe = &as.pe;
    a.video_cross_pe = &vs.cross_pe;
    a.audio_cross_pe = &as.cross_pe;
    // `_process_transformer_blocks` (model.py:442-458): the SELF types are per
    // block and the CROSS types ride whole. An EMPTY vector is "no block", which
    // is what `PerturbationConfig.empty()` reaches the forward as.
    if (perturbations != nullptr) {
      a.video_self_attn_perturbed = !perturbations->video_self_attn.empty() &&
                                    perturbations->video_self_attn[static_cast<size_t>(i)] != 0;
      a.audio_self_attn_perturbed = !perturbations->audio_self_attn.empty() &&
                                    perturbations->audio_self_attn[static_cast<size_t>(i)] != 0;
      a.video_cross_attn_skip_all = perturbations->video_cross_attn_skip_all;
      a.audio_cross_attn_skip_all = perturbations->audio_cross_attn_skip_all;
    }
    BlockForwardDev(c, weights.blocks[static_cast<size_t>(i)], a, &vs.x->t(), &as.x->t());
  }

  Ltx2DitOutputs out;
  out.video = ProcessOutputDev(c, weights.scale_shift_table, weights.proj_out, vs.x->t(),
                               *vs.embedded, video->batch * video->tokens, dim,
                               params.out_channels);
  out.audio = ProcessOutputDev(c, weights.audio_scale_shift_table, weights.audio_proj_out,
                               as.x->t(), *as.embedded, audio->batch * audio->tokens, adim,
                               params.audio_out_channels);
  return out;
}

}  // namespace vllm
