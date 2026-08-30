// MODEL-MM-QWEN4-EXP W5b-5 — the QSA decoder-layer block as ONE production
// composition, gated against the lane-pinned oracle's own
// `Qwen4ExpTextAttention.forward`.
//
// Issue #2211, wave issue #2031, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS FILE GATES THAT `test_qwen4_exp_qsa_device.cpp` DOES NOT ──────
// That file gates the four `vt::` ops from the INDEXER's own inputs, and it
// composes them in a TEST HELPER (`RunIndexer`). The spec's `## Owed` says why
// that is not enough: "W5b must write this recipe again where no test helper is
// watching, and two of the four have no gate that would catch a wrong value
// there". This file gates the composition that now lives under `src/`, plus the
// outer block the helper never had — the projections, the per-head q/k norms,
// the RoPE, the sigmoid OUTPUT GATE and `o_proj`.
//
// ─── THE VALUE GATE ON THE LOGITS ────────────────────────────────────────────
// `n_head_scale` and `softmax_scale` are GLOBAL POSITIVE rescales of every
// score, so no selection can move when either is wrong — spec mutation M26
// measures that survival, and `DsaIndexerLogitsArgs` states it in its own
// comment. The first case below therefore compares `Qwen4ExpQsaIndex`'s logits
// BY VALUE against the oracle's own pre-top-k `scores` tensor, captured by
// intercepting `torch.Tensor.topk` inside the unmodified oracle
// (`fixtures/gen_qwen4_exp_qsa_block_goldens.py`). A selection gate here would
// be an instrument nobody wired up.
//
// ─── THE MASK-SHAPED-CONSUMER GATE ───────────────────────────────────────────
// A sparse mask over a dense cache agrees with a gather VALUE FOR VALUE
// (`exp(-inf - m)` is exactly +0), so the golden comparison below cannot see
// one. The NaN-poison case runs the WHOLE BLOCK over a cache whose unselected
// rows are not numbers, which a mask multiplies into `0.0f * NaN`. The
// fetch-level property — that the bytes were never READ — is discharged one
// layer down by the `mprotect(PROT_NONE)` probe in
// `test_qwen4_exp_qsa_device.cpp`, because this block's ONLY consumer call is
// that op; the mutation that says so is in the spec's W5b-5 table.
//
// ─── TOLERANCES, AND WHY THERE ARE TWO ───────────────────────────────────────
// The oracle runs bf16, which is the model dtype vLLM resolves and which this
// tree's `vt::` output-gate ops store unconditionally. So:
//   * the LOGITS are f32 on both sides — the oracle scores through an explicit
//     `.float()` and `vt::DsaIndexerLogits` writes f32 — and carry a TIGHT
//     relative bound.
//   * the BLOCK OUTPUT is bf16 and carries a relative bound sized to the bf16
//     quantum. That is strong enough for every structural property the mutation
//     table exercises and too weak for an epsilon, which is stated rather than
//     implied.
// Both bounds are RELATIVE. W3 measured why an absolute one is wrong at model
// width: an exact-double evaluation of the oracle's own algorithm already
// exceeds 1e-5 absolute there, so an absolute bound tests the accumulator.
#include "vllm/model_executor/models/qwen4_exp_qsa_block.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // dense_attn::ResidentWeight
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "fixtures/qwen4_exp_qsa_block_goldens.inc"  // NOLINT — golden literals

namespace g = qwen4_exp_qsa_block_goldens;

using vllm::OwnedTensor;
using vllm::Qwen4ExpParams;
using vllm::Qwen4ExpQsaCaches;
using vllm::Qwen4ExpQsaWeights;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= t.shape[i];
  }
  return t;
}

// A golden float array into an `OwnedTensor` at `dt`. The goldens are bf16
// values PRINTED as f32 literals (the oracle module runs bf16), so the bf16
// conversion below is exact and a bf16 weight holds the oracle's own bytes.
OwnedTensor OwnedFrom(DType dt, const std::vector<int64_t>& shape, const float* src) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  // Raw torch `nn.Linear` [N, K] order, which is what `vt::MatmulBT` consumes
  // and what the GGUF loader hands the forward.
  t.nk = t.rank == 2;
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(src[i]);
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = src[i];
  }
  return t;
}

std::vector<uint16_t> Bf16Of(const float* src, int64_t n) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = vt::F32ToBF16(src[i]);
  return v;
}

std::vector<float> F32Of(const uint16_t* src, int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  return v;
}

// The released tiny geometry the fixture was captured at.
Qwen4ExpParams GoldenParams() {
  Qwen4ExpParams p;
  p.hidden_size = g::kHiddenSize;
  p.num_hidden_layers = 4;
  p.rms_norm_eps = g::kRmsNormEps;
  p.num_attention_heads = g::kNumAttentionHeads;
  p.num_key_value_heads = g::kNumKeyValueHeads;
  p.head_dim = g::kHeadDim;
  p.rotary_dim = g::kRotaryDim;
  p.qsa.n_heads = g::kIndexNHeads;
  p.qsa.kv_heads = g::kIndexKvHeads;
  p.qsa.head_dim = g::kIndexHeadDim;
  p.qsa.budget = g::kTokenBudget;
  p.qsa.compress_ratio = g::kCompressRatio;
  return p;
}

// The oracle's `index_qk_proj` is ONE `[(n_heads + kv_heads) * D, H]` weight;
// the converter SPLITS it at `n_heads * D` and the loader carries the halves.
// Slicing the golden the same way is a SHAPE operation on a row-major matrix,
// not arithmetic, and it is what makes the two halves reproduce the whole.
Qwen4ExpQsaWeights GoldenWeights(DType dt) {
  const int64_t H = g::kHiddenSize, HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads;
  const int64_t DH = g::kHeadDim, IH = g::kIndexNHeads, ID = g::kIndexHeadDim;
  Qwen4ExpQsaWeights w;
  w.q_proj = OwnedFrom(dt, {HQ * DH * 2, H}, g::kQProj);
  w.k_proj = OwnedFrom(dt, {HKV * DH, H}, g::kKProj);
  w.v_proj = OwnedFrom(dt, {HKV * DH, H}, g::kVProj);
  w.o_proj = OwnedFrom(dt, {H, HQ * DH}, g::kOProj);
  w.q_norm = OwnedFrom(dt, {DH}, g::kQNormW);
  w.k_norm = OwnedFrom(dt, {DH}, g::kKNormW);
  w.idx_q_proj = OwnedFrom(dt, {IH * ID, H}, g::kIdxQkProj);
  w.idx_k_proj = OwnedFrom(dt, {ID, H}, g::kIdxQkProj + IH * ID * H);
  w.idx_q_norm = OwnedFrom(dt, {ID}, g::kIdxQNormW);
  w.idx_k_norm = OwnedFrom(dt, {ID}, g::kIdxKNormW);
  return w;
}

// One golden case: everything that varies with the sequence length.
struct Case {
  const char* name;
  int64_t seq;
  const float* hidden;
  const float* cos;
  const float* sin;
  const float* scores;
  const int32_t* score_offsets;
  const int32_t* selected;
  const float* out;
  const float* idx_q_post;  // the ORACLE's roped indexer query [seq, IH, ID]
  const float* idx_k_raw;   // the ORACLE's raw indexer keys    [seq, ID]
};

const Case kSubBudget{"sub_budget",         g::kSubBudgetSeq,          g::kSubBudgetHidden,
                      g::kSubBudgetCos,     g::kSubBudgetSin,          g::kSubBudgetScores,
                      g::kSubBudgetScoreOffsets, g::kSubBudgetSelected, g::kSubBudgetOut,
                      g::kSubBudgetIdxQPost, g::kSubBudgetIdxKRaw};
const Case kOverBudget{"over_budget",        g::kOverBudgetSeq,          g::kOverBudgetHidden,
                       g::kOverBudgetCos,    g::kOverBudgetSin,          g::kOverBudgetScores,
                       g::kOverBudgetScoreOffsets, g::kOverBudgetSelected, g::kOverBudgetOut,
                       g::kOverBudgetIdxQPost, g::kOverBudgetIdxKRaw};

// The two RoPE layouts one set of angles has to be handed in, and the reason is
// in the block header: `vt::RopeFromCache` reads a PACKED `[P, rot]` cache whose
// columns are `[cos(rot/2) | sin(rot/2)]`, while `vt::Qwen4ExpQsaCompress` reads
// the two FULL `[P, rot]` tables separately and in f32. Upstream's `emb =
// cat(freqs, freqs)` makes the second half of each row a copy of the first, so
// the packed cache is built from the leading halves and the two layouts describe
// the same angles by construction.
struct RopeTables {
  std::vector<uint16_t> packed;  // [seq, rot] bf16, cos|sin
  std::vector<float> cos;        // [seq, rot] f32
  std::vector<float> sin;        // [seq, rot] f32
};

RopeTables BuildRope(const Case& c) {
  const int64_t rot = g::kRotaryDim, half = rot / 2;
  RopeTables r;
  r.cos.assign(c.cos, c.cos + c.seq * rot);
  r.sin.assign(c.sin, c.sin + c.seq * rot);
  r.packed.resize(static_cast<size_t>(c.seq * rot));
  for (int64_t p = 0; p < c.seq; ++p) {
    for (int64_t j = 0; j < half; ++j) {
      r.packed[static_cast<size_t>(p * rot + j)] = vt::F32ToBF16(r.cos[static_cast<size_t>(p * rot + j)]);
      r.packed[static_cast<size_t>(p * rot + half + j)] =
          vt::F32ToBF16(r.sin[static_cast<size_t>(p * rot + j)]);
    }
  }
  return r;
}

// The block's caches, sized to `max_kv` and owned by the caller.
struct Caches {
  std::vector<uint16_t> key, value, index_key;
  Qwen4ExpQsaCaches t;
  Caches(int64_t max_kv, int64_t hkv, int64_t dh, int64_t idx_d)
      : key(static_cast<size_t>(max_kv * hkv * dh), 0),
        value(static_cast<size_t>(max_kv * hkv * dh), 0),
        index_key(static_cast<size_t>(max_kv * idx_d), 0) {
    t.key = MakeT(key.data(), DType::kBF16, {max_kv, hkv, dh});
    t.value = MakeT(value.data(), DType::kBF16, {max_kv, hkv, dh});
    t.index_key = MakeT(index_key.data(), DType::kBF16, {max_kv, idx_d});
  }
};

// Expand `block_ids` into the token set the consumer attends: every selected
// block `b` as tokens [CR*b, CR*b + CR), plus the ALWAYS-attended ragged tail.
// This mirrors what the op does with ADDRESSES; here it is a comparison target
// for the oracle's own selected sets and nothing computes with it.
std::vector<int32_t> ExpandSelection(const int32_t* ids, int64_t topk, int64_t kv_len,
                                     int64_t cr, int64_t width) {
  std::vector<int32_t> out;
  for (int64_t j = 0; j < topk; ++j) {
    if (ids[j] < 0) break;
    for (int64_t i = 0; i < cr; ++i)
      out.push_back(static_cast<int32_t>(ids[j] * cr + i));
  }
  for (int64_t p = (kv_len / cr) * cr; p < kv_len; ++p) out.push_back(static_cast<int32_t>(p));
  out.resize(static_cast<size_t>(width), -1);
  return out;
}

// Relative difference against the golden, guarded so a golden of zero does not
// divide. `scale` is the golden array's own max magnitude, which is what makes
// the bound relative to the SIGNAL rather than to each element.
double MaxRelDiff(const std::vector<float>& got, const float* want, int64_t n) {
  double scale = 0.0;
  for (int64_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
  if (scale == 0.0) scale = 1.0;
  double worst = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = std::fabs(static_cast<double>(got[static_cast<size_t>(i)]) -
                               static_cast<double>(want[i]));
    worst = std::max(worst, d / scale);
  }
  return worst;
}

// One prefill call of the block over a whole case, from an empty cache.
struct BlockRun {
  std::vector<float> out;      // [seq, H] widened from bf16
  std::vector<int32_t> ids;    // [seq, block_topk]
  std::vector<float> logits;   // [seq, nb]
  int64_t nb = 0;
  int64_t keys_visited = 0;
};

BlockRun RunCase(const Case& c, const Qwen4ExpQsaWeights& w, const Qwen4ExpParams& p,
                 Caches& caches, bool tap_logits) {
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  const int64_t H = p.hidden_size, rot = p.rotary_dim;
  RopeTables rope = BuildRope(c);

  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  Tensor t_hidden = MakeT(hidden.data(), DType::kBF16, {c.seq, H});
  Tensor t_pos = MakeT(positions.data(), DType::kI32, {c.seq});
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});

  BlockRun r;
  r.nb = c.seq / p.qsa.compress_ratio;
  vllm::Qwen4ExpQsaBlockOutput o = vllm::RunQwen4ExpQsaBlock(
      d, w, p, t_hidden, t_pos, t_cs, t_cos, t_sin, caches.t, /*past_len=*/0, &r.keys_visited);
  r.out = F32Of(o.tensor.Ptr<uint16_t>(), c.seq * H);

  // The selection and the logits come from a SECOND call to the composed
  // indexer over the caches the block just filled. That is the same production
  // function the block used, on the same inputs, so it observes the block's own
  // selection rather than a reconstruction of it — and it is how the logits tap
  // is read without putting a debug parameter on the block itself.
  if (tap_logits) {
    // Re-derive the indexer query exactly as the block does. Everything here is
    // a `vt::` call; nothing reimplements the block's arithmetic.
    const int64_t IH = p.qsa.n_heads, ID = p.qsa.head_dim;
    std::vector<uint16_t> qi(static_cast<size_t>(c.seq * IH * ID));
    Tensor t_qi = MakeT(qi.data(), DType::kBF16, {c.seq, IH * ID});
    vt::MatmulBT(q, t_qi, t_hidden, vllm::dense_attn::ResidentWeight(d, w.idx_q_proj, {IH * ID, H}));
    Tensor flat = MakeT(qi.data(), DType::kBF16, {c.seq * IH, ID});
    vt::RmsNorm(q, flat, flat, vllm::dense_attn::ResidentWeight(d, w.idx_q_norm, {ID}),
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), /*gemma=*/true});
    Tensor t_q3 = MakeT(qi.data(), DType::kBF16, {c.seq, IH, ID});
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(rot);
    ra.is_neox_style = true;
    vt::RopeFromCache(q, t_q3, nullptr, t_pos, t_cs, ra);

    std::vector<int32_t> kv_lens(static_cast<size_t>(c.seq));
    for (int64_t t = 0; t < c.seq; ++t)
      kv_lens[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
    Tensor t_len = MakeT(kv_lens.data(), DType::kI32, {c.seq});

    r.logits.assign(static_cast<size_t>(c.seq * r.nb), 0.0f);
    Tensor t_lg = MakeT(r.logits.data(), DType::kF32, {c.seq, r.nb});
    vllm::Qwen4ExpQsaSelection sel = vllm::Qwen4ExpQsaIndex(
        d, p.qsa, static_cast<float>(p.rms_norm_eps), t_q3, caches.t.index_key,
        vllm::dense_attn::ResidentWeight(d, w.idx_k_norm, {ID}), t_cos, t_sin, t_len, c.seq,
        /*round_intermediates_to_bf16=*/true, &t_lg);
    const int64_t topk = p.qsa.block_topk();
    r.ids.assign(sel.block_ids.Ptr<int32_t>(), sel.block_ids.Ptr<int32_t>() + c.seq * topk);
  }
  return r;
}

}  // namespace

// ── 1. THE VALUE GATE ON THE LOGITS ─────────────────────────────────────────

TEST_CASE("qwen4_exp qsa block: the composed indexer's LOGITS match the oracle BY VALUE") {
  // THE CASE THE SPEC'S `## Owed` ASKS FOR, and the only one in this tree that
  // can answer it. Two of the four settings the composition depends on —
  // `n_head_scale == 1` and `softmax_scale == index_head_dim ** -0.5` — are
  // positive GLOBAL rescales, so NO selection can move when either is wrong;
  // spec mutation M26 measures that survival and `DsaIndexerLogitsArgs` states
  // it in its own comment. `k...Scores` is the oracle's own pre-top-k score
  // tensor, recorded by intercepting `torch.Tensor.topk` inside the unmodified
  // `Qwen4ExpTextQSAIndexer.forward`.
  //
  // THE INPUTS ARE THE ORACLE'S OWN, AND THAT IS WHAT MAKES THE BOUND MEAN
  // SOMETHING. Fed this port's bf16 projection and bf16 RoPE, the same
  // comparison inherits a bf16 ulp and lands at ~2e-3 relative — measured, not
  // feared; it is the case below. Fed `k...IdxQPost` and `k...IdxKRaw`, the only
  // difference left is the reassociation the spec names (upstream divides AFTER
  // the head sum, the op's fold multiplies BEFORE it), which is at most a few
  // f32 ulps. So this case gates the CONSTANTS and the case below gates the
  // INPUTS, and neither pretends to be the other.
  constexpr double kLogitsTol = 1e-6;
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const int64_t IH = p.qsa.n_heads, ID = p.qsa.head_dim;
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a bare char* as a bool
    RopeTables rope = BuildRope(*c);
    Caches caches(c->seq, p.num_key_value_heads, p.head_dim, ID);
    // The side cache is loaded with the ORACLE's raw indexer keys.
    for (int64_t i = 0; i < c->seq * ID; ++i)
      caches.index_key[static_cast<size_t>(i)] = vt::F32ToBF16(c->idx_k_raw[i]);
    std::vector<uint16_t> qp = Bf16Of(c->idx_q_post, c->seq * IH * ID);
    Tensor t_q3 = MakeT(qp.data(), DType::kBF16, {c->seq, IH, ID});
    Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c->seq, g::kRotaryDim});
    Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c->seq, g::kRotaryDim});
    std::vector<int32_t> lens(static_cast<size_t>(c->seq));
    for (int64_t t = 0; t < c->seq; ++t) lens[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
    Tensor t_len = MakeT(lens.data(), DType::kI32, {c->seq});

    const int64_t nb = c->seq / p.qsa.compress_ratio;
    std::vector<float> logits(static_cast<size_t>(c->seq * nb), 0.0f);
    Tensor t_lg = MakeT(logits.data(), DType::kF32, {c->seq, nb});
    vllm::Qwen4ExpQsaIndex(d, p.qsa, static_cast<float>(p.rms_norm_eps), t_q3, caches.t.index_key,
                           vllm::dense_attn::ResidentWeight(d, w.idx_k_norm, {ID}), t_cos, t_sin,
                           t_len, c->seq, /*round_intermediates_to_bf16=*/true, &t_lg);

    // The oracle only reaches `.topk` for a query that saw at least one complete
    // block, so the first CR-1 queries contribute nothing and the offsets say so.
    const int64_t first = p.qsa.compress_ratio - 1;
    REQUIRE(c->score_offsets[0] == 0);
    double worst = 0.0, scale = 0.0;
    int64_t compared = 0;
    for (int64_t t = first; t < c->seq; ++t) {
      const int64_t k = t - first;
      const int64_t lo = c->score_offsets[k], hi = c->score_offsets[k + 1];
      // The oracle scores `(t + 1) / CR` blocks for query t — the COMPLETE
      // VISIBLE ones. If this count disagreed, setting 4 (the scoring window)
      // would be wrong and every value below would be compared against the
      // wrong block. Mutation M-W4 in the spec's table is the paired red.
      REQUIRE(hi - lo == (t + 1) / p.qsa.compress_ratio);
      for (int64_t b = 0; b < hi - lo; ++b) {
        const double want = c->scores[lo + b];
        const double got = logits[static_cast<size_t>(t * nb + b)];
        scale = std::max(scale, std::fabs(want));
        worst = std::max(worst, std::fabs(got - want));
        ++compared;
      }
    }
    // A comparison that compared nothing is the "a gate that never fired is not
    // a gate" failure, so the count is REQUIREd rather than assumed.
    REQUIRE(compared > 0);
    REQUIRE(scale > 0.0);
    INFO("compared ", compared, " logits, max abs ", worst, " over scale ", scale);
    CHECK(worst / scale < kLogitsTol);
  }
}

TEST_CASE("qwen4_exp qsa block: the block's OWN indexer inputs are the oracle's") {
  // THE COMPANION THE CASE ABOVE NEEDS. That one hands the composition the
  // oracle's own roped query and raw keys, so on its own it says nothing about
  // whether the BLOCK produces those. This one says it: the block is run, and
  // the side cache it filled plus the indexer query it would have built are
  // compared against `k...IdxKRaw` and `k...IdxQPost`.
  //
  // The bound is bf16-sized and that is the measurement, not a concession: the
  // projections are bf16 GEMMs and `vt::RopeFromCache` computes each rotated
  // pair in f32 and stores once where upstream multiplies and adds in bf16, so
  // one ulp is the floor. A tighter bound here would fail the correct code.
  constexpr double kBf16Tol = 1e-2;
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const int64_t H = p.hidden_size, rot = p.rotary_dim, IH = p.qsa.n_heads, ID = p.qsa.head_dim;
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));
    Caches caches(c->seq, p.num_key_value_heads, p.head_dim, ID);
    RopeTables rope = BuildRope(*c);
    std::vector<uint16_t> hidden = Bf16Of(c->hidden, c->seq * H);
    std::vector<int32_t> positions(static_cast<size_t>(c->seq));
    for (int64_t t = 0; t < c->seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    Tensor t_h = MakeT(hidden.data(), DType::kBF16, {c->seq, H});
    Tensor t_p = MakeT(positions.data(), DType::kI32, {c->seq});
    Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c->seq, rot});
    Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c->seq, rot});
    Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c->seq, rot});
    vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, t_cos, t_sin, caches.t, 0);

    // The side cache holds the RAW indexer keys — un-normed, un-roped. A block
    // that normed or roped them before the store would double-apply both inside
    // the compressor, and it would still produce plausible output.
    const std::vector<float> got_k = F32Of(caches.index_key.data(), c->seq * ID);
    const double rel_k = MaxRelDiff(got_k, c->idx_k_raw, c->seq * ID);
    INFO("raw indexer keys, max relative difference ", rel_k);
    CHECK(rel_k < kBf16Tol);

    // And the roped, q-layernormed indexer query, rebuilt through the same
    // `vt::` calls the block makes.
    std::vector<uint16_t> qi(static_cast<size_t>(c->seq * IH * ID));
    Tensor t_qi = MakeT(qi.data(), DType::kBF16, {c->seq, IH * ID});
    vt::MatmulBT(q, t_qi, t_h, vllm::dense_attn::ResidentWeight(d, w.idx_q_proj, {IH * ID, H}));
    Tensor flat = MakeT(qi.data(), DType::kBF16, {c->seq * IH, ID});
    vt::RmsNorm(q, flat, flat, vllm::dense_attn::ResidentWeight(d, w.idx_q_norm, {ID}),
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), /*gemma=*/true});
    Tensor t_q3 = MakeT(qi.data(), DType::kBF16, {c->seq, IH, ID});
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(rot);
    ra.is_neox_style = true;
    vt::RopeFromCache(q, t_q3, nullptr, t_p, t_cs, ra);
    const std::vector<float> got_q = F32Of(qi.data(), c->seq * IH * ID);
    const double rel_q = MaxRelDiff(got_q, c->idx_q_post, c->seq * IH * ID);
    INFO("roped indexer query, max relative difference ", rel_q);
    CHECK(rel_q < kBf16Tol);
  }
}

// ── 2. THE SELECTION ────────────────────────────────────────────────────────

TEST_CASE("qwen4_exp qsa block: the composed indexer selects the oracle's token sets") {
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const int64_t topk = p.qsa.block_topk();
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));
    Caches caches(c->seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    const BlockRun r = RunCase(*c, w, p, caches, /*tap_logits=*/true);
    for (int64_t t = 0; t < c->seq; ++t) {
      CAPTURE(t);
      const std::vector<int32_t> got = ExpandSelection(
          r.ids.data() + t * topk, topk, t + 1, p.qsa.compress_ratio, g::kIndexWidth);
      for (int64_t j = 0; j < g::kIndexWidth; ++j) {
        CAPTURE(j);
        CHECK(got[static_cast<size_t>(j)] == c->selected[t * g::kIndexWidth + j]);
      }
    }
  }
}

// ── 3. THE BLOCK, END TO END ────────────────────────────────────────────────

TEST_CASE("qwen4_exp qsa block: the block output matches Qwen4ExpTextAttention.forward") {
  // The bound is RELATIVE and sized to bf16. The oracle runs bf16 and so does
  // this block, but the two do not round at the same instants: upstream's RoPE
  // multiplies and adds in bf16 while `vt::RopeFromCache` computes the pair in
  // f32 and stores once, and its `torch.sigmoid` narrows before the multiply
  // where `vt::SigmoidGateBf16` narrows after. Each such difference is one bf16
  // ulp, ~0.4%, and they compound across eleven steps. 3% of the signal's own
  // magnitude is a bound every structural mutation in the W5b-5 table clears by
  // more than an order of magnitude, and it is NOT a bound that would see an
  // epsilon — which is why the epsilon-shaped properties are gated on the f32
  // logits above and not here.
  constexpr double kOutTol = 3e-2;
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));
    Caches caches(c->seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    const BlockRun r = RunCase(*c, w, p, caches, /*tap_logits=*/false);
    REQUIRE(static_cast<int64_t>(r.out.size()) == c->seq * p.hidden_size);
    const double rel = MaxRelDiff(r.out, c->out, c->seq * p.hidden_size);
    INFO("max relative difference ", rel);
    CHECK(rel < kOutTol);
    for (float v : r.out) REQUIRE(std::isfinite(v));
  }
}

TEST_CASE("qwen4_exp qsa block: a DECODE step continues a prefilled cache") {
  // The same case run as prefill(seq-1) then one decode token, against the same
  // golden. Nothing else in this file exercises `past_len > 0`, and the cache
  // slot arithmetic is where an off-by-one lives: a decode that wrote its k/v to
  // the wrong row would still produce finite, plausible output.
  constexpr double kOutTol = 3e-2;
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const Case& c = kOverBudget;
  const int64_t H = p.hidden_size, rot = p.rotary_dim;
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
  RopeTables rope = BuildRope(c);
  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});

  {  // prefill of the first seq-1 tokens
    Tensor t_h = MakeT(hidden.data(), DType::kBF16, {c.seq - 1, H});
    Tensor t_p = MakeT(positions.data(), DType::kI32, {c.seq - 1});
    vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, t_cos, t_sin, caches.t, /*past_len=*/0);
  }
  int64_t visited = 0;
  vllm::Qwen4ExpQsaBlockOutput o;
  {  // one decode token
    Tensor t_h = MakeT(hidden.data() + (c.seq - 1) * H, DType::kBF16, {1, H});
    Tensor t_p = MakeT(positions.data() + (c.seq - 1), DType::kI32, {1});
    o = vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, t_cos, t_sin, caches.t,
                                  /*past_len=*/c.seq - 1, &visited);
  }
  const std::vector<float> got = F32Of(o.tensor.Ptr<uint16_t>(), H);
  const double rel = MaxRelDiff(got, c.out + (c.seq - 1) * H, H);
  INFO("decode-step max relative difference ", rel, ", keys_visited ", visited);
  CHECK(rel < kOutTol);
  CHECK(visited > 0);

  // THE SIDE CACHE AFTER THE SPLIT, ROW BY ROW. This is the repair for a
  // mutation that SURVIVED the first battery: writing the indexer key at row 0
  // instead of row `past_len`. The prefill cases cannot see it — `past_len` is 0
  // there, so the two spellings are the same expression — and the block output
  // alone did not either, because one wrong pooled key moves the selection by
  // one block out of five and the result stays inside a bf16-sized bound. A
  // cache compared row for row against the oracle's own raw keys does see it,
  // and it is the observable that says WHERE the write landed rather than what
  // it was worth.
  const int64_t ID = p.qsa.head_dim;
  const std::vector<float> cached = F32Of(caches.index_key.data(), c.seq * ID);
  for (int64_t r = 0; r < c.seq; ++r) {
    CAPTURE(r);
    const double rel_row = MaxRelDiff(
        std::vector<float>(cached.begin() + r * ID, cached.begin() + (r + 1) * ID),
        c.idx_k_raw + r * ID, ID);
    CHECK(rel_row < kOutTol);
  }
}

// ── 4. THE MASK-SHAPED-CONSUMER GATE, THROUGH THE BLOCK ─────────────────────

TEST_CASE("qwen4_exp qsa block: the block's consumer is a GATHER, not a mask") {
  // The golden comparison above cannot see a mask-shaped consumer, and that is
  // not a weakness of this fixture: a sparse mask over a dense cache agrees with
  // a gather VALUE FOR VALUE, because `exp(-inf - m)` is exactly +0 and adding
  // an exact zero changes no accumulator. What separates them is an observable
  // OF THE WALK.
  //
  // The block is run twice over the same inputs. The second run's key and value
  // caches have every row the block's own selection does NOT name replaced by
  // NaN, for ONE query token — the last, whose complement is therefore well
  // defined. A gather never addresses those rows and is bit-identical. A mask
  // reads every value row and accumulates `w * v` with `w == 0.0f`, and
  // `0.0f * NaN` is NaN in IEEE-754, so its output is NaN in every lane.
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const Case& c = kOverBudget;  // the case where top-k actually discards blocks
  const int64_t H = p.hidden_size, rot = p.rotary_dim, HKV = p.num_key_value_heads;
  const int64_t DH = p.head_dim, CR = p.qsa.compress_ratio, topk = p.qsa.block_topk();
  const int64_t qi = c.seq - 1, kv = c.seq;
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  RopeTables rope = BuildRope(c);
  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});
  Tensor t_h1 = MakeT(hidden.data() + qi * H, DType::kBF16, {1, H});
  Tensor t_p1 = MakeT(positions.data() + qi, DType::kI32, {1});

  // Pass 1: prefill the cache with the first `kv-1` tokens, then run the last
  // token alone and record BOTH its output and the selection it made.
  Caches clean(kv, HKV, DH, p.qsa.head_dim);
  {
    Tensor t_h = MakeT(hidden.data(), DType::kBF16, {qi, H});
    Tensor t_p = MakeT(positions.data(), DType::kI32, {qi});
    vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, t_cos, t_sin, clean.t, 0);
  }
  int64_t visited_clean = 0;
  vllm::Qwen4ExpQsaBlockOutput o_clean = vllm::RunQwen4ExpQsaBlock(
      d, w, p, t_h1, t_p1, t_cs, t_cos, t_sin, clean.t, qi, &visited_clean);
  const std::vector<float> want = F32Of(o_clean.tensor.Ptr<uint16_t>(), H);

  // The selection this token made, read back through the same production
  // indexer over the now-complete cache.
  std::vector<int32_t> sel_ids;
  {
    const int64_t IH = p.qsa.n_heads, ID = p.qsa.head_dim;
    std::vector<uint16_t> qi_buf(static_cast<size_t>(IH * ID));
    Tensor t_qi = MakeT(qi_buf.data(), DType::kBF16, {1, IH * ID});
    vt::MatmulBT(q, t_qi, t_h1, vllm::dense_attn::ResidentWeight(d, w.idx_q_proj, {IH * ID, H}));
    Tensor flat = MakeT(qi_buf.data(), DType::kBF16, {IH, ID});
    vt::RmsNorm(q, flat, flat, vllm::dense_attn::ResidentWeight(d, w.idx_q_norm, {ID}),
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), /*gemma=*/true});
    Tensor t_q3 = MakeT(qi_buf.data(), DType::kBF16, {1, IH, ID});
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(rot);
    ra.is_neox_style = true;
    vt::RopeFromCache(q, t_q3, nullptr, t_p1, t_cs, ra);
    std::vector<int32_t> lens(1, static_cast<int32_t>(kv));
    Tensor t_len = MakeT(lens.data(), DType::kI32, {1});
    vllm::Qwen4ExpQsaSelection s = vllm::Qwen4ExpQsaIndex(
        d, p.qsa, static_cast<float>(p.rms_norm_eps), t_q3, clean.t.index_key,
        vllm::dense_attn::ResidentWeight(d, w.idx_k_norm, {ID}), t_cos, t_sin, t_len, kv, true);
    sel_ids.assign(s.block_ids.Ptr<int32_t>(), s.block_ids.Ptr<int32_t>() + topk);
  }
  const std::vector<int32_t> attended =
      ExpandSelection(sel_ids.data(), topk, kv, CR, g::kIndexWidth);
  std::vector<bool> keep(static_cast<size_t>(kv), false);
  int64_t n_sel = 0;
  for (int32_t v : attended) {
    if (v < 0) continue;
    keep[static_cast<size_t>(v)] = true;
    ++n_sel;
  }
  // A poison set that is empty makes the case vacuous. 23 cached, fewer attended.
  REQUIRE(n_sel > 0);
  REQUIRE(n_sel < kv);

  // Pass 2: the identical run over a cache whose unselected rows are NaN. The
  // caches are rebuilt from scratch and re-prefilled so the block writes exactly
  // what it wrote before, and only THEN poisoned.
  Caches poisoned(kv, HKV, DH, p.qsa.head_dim);
  {
    Tensor t_h = MakeT(hidden.data(), DType::kBF16, {qi, H});
    Tensor t_p = MakeT(positions.data(), DType::kI32, {qi});
    vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, t_cos, t_sin, poisoned.t, 0);
  }
  const uint16_t nan_bf16 = vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN());
  int64_t n_poisoned = 0;
  for (int64_t rrow = 0; rrow < qi; ++rrow) {  // row qi is written by the call below
    if (keep[static_cast<size_t>(rrow)]) continue;
    ++n_poisoned;
    for (int64_t j = 0; j < HKV * DH; ++j) {
      poisoned.key[static_cast<size_t>(rrow * HKV * DH + j)] = nan_bf16;
      poisoned.value[static_cast<size_t>(rrow * HKV * DH + j)] = nan_bf16;
    }
  }
  REQUIRE(n_poisoned > 0);
  INFO("poisoned ", n_poisoned, " of ", kv, " cached rows; ", n_sel, " attended");

  int64_t visited_poisoned = 0;
  vllm::Qwen4ExpQsaBlockOutput o_poisoned = vllm::RunQwen4ExpQsaBlock(
      d, w, p, t_h1, t_p1, t_cs, t_cos, t_sin, poisoned.t, qi, &visited_poisoned);
  const std::vector<float> got = F32Of(o_poisoned.tensor.Ptr<uint16_t>(), H);

  // Finiteness alone would pass a body that read the poison and threw the row
  // away; the bit-equality says the poison never entered the arithmetic.
  CHECK(visited_poisoned == visited_clean);
  for (int64_t i = 0; i < H; ++i) {
    CAPTURE(i);
    CHECK(std::isfinite(got[static_cast<size_t>(i)]));
    CHECK(got[static_cast<size_t>(i)] == want[static_cast<size_t>(i)]);
  }
}

// ── 5. THE RELEASED CONFIG, PAST 2048 TOKENS OF CONTEXT ─────────────────────

TEST_CASE("qwen4_exp qsa block: the released config past 2048 tokens is genuinely sparse") {
  // THE GATE THE SPEC DEMANDS OF ANY QSA CLAIM. At or below `indexer_budget`
  // every candidate is selected, so every read-count assertion is trivially true
  // and a dense body passes it — the spec measures exactly that at kv_len 2051.
  // A QSA gate that never crosses 2048 is not a weaker gate; it is not a gate.
  //
  // The INDEXER values are the released ones (budget 2048, compress_ratio 4);
  // the model width is not, because the property under test is the relationship
  // between the context and the budget and a 2560-wide fixture would only make
  // it slower to observe.
  Qwen4ExpParams p = GoldenParams();
  p.qsa.budget = 2048;
  p.qsa.compress_ratio = 4;
  REQUIRE(p.qsa.block_topk() == 512);
  const int64_t H = p.hidden_size, rot = p.rotary_dim, HKV = p.num_key_value_heads;
  const int64_t DH = p.head_dim, HQ = p.num_attention_heads;
  const int64_t kv = 3002;  // past the budget, and NOT a multiple of 4: a ragged tail
  REQUIRE(kv % p.qsa.compress_ratio != 0);
  const int64_t complete = kv / p.qsa.compress_ratio;
  REQUIRE(p.qsa.block_topk() < complete);  // genuinely sparse: 512 of 750 blocks

  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  Caches caches(kv, HKV, DH, p.qsa.head_dim);

  // The cache is filled by a deterministic pseudo-random draw rather than by a
  // prefill of 3001 tokens: what is under test is the CONSUMER's read pattern at
  // this context, and 3001 block calls would only add minutes.
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>(static_cast<double>(s >> 40) / (1 << 24) - 0.5);
  };
  for (int64_t i = 0; i < kv * HKV * DH; ++i) {
    caches.key[static_cast<size_t>(i)] = vt::F32ToBF16(next());
    caches.value[static_cast<size_t>(i)] = vt::F32ToBF16(next());
  }
  for (int64_t i = 0; i < (kv - 1) * p.qsa.head_dim; ++i)
    caches.index_key[static_cast<size_t>(i)] = vt::F32ToBF16(next());

  std::vector<uint16_t> hidden(static_cast<size_t>(H));
  for (int64_t i = 0; i < H; ++i) hidden[static_cast<size_t>(i)] = vt::F32ToBF16(next());
  std::vector<int32_t> pos(1, static_cast<int32_t>(kv - 1));
  std::vector<float> cos(static_cast<size_t>(kv * rot)), sin(static_cast<size_t>(kv * rot));
  std::vector<uint16_t> packed(static_cast<size_t>(kv * rot));
  for (int64_t pp = 0; pp < kv; ++pp) {
    for (int64_t j = 0; j < rot / 2; ++j) {
      const double th = static_cast<double>(pp) / std::pow(10000.0, 2.0 * j / rot);
      const float cv = static_cast<float>(std::cos(th)), sv = static_cast<float>(std::sin(th));
      cos[static_cast<size_t>(pp * rot + j)] = cv;
      cos[static_cast<size_t>(pp * rot + rot / 2 + j)] = cv;
      sin[static_cast<size_t>(pp * rot + j)] = sv;
      sin[static_cast<size_t>(pp * rot + rot / 2 + j)] = sv;
      packed[static_cast<size_t>(pp * rot + j)] = vt::F32ToBF16(cv);
      packed[static_cast<size_t>(pp * rot + rot / 2 + j)] = vt::F32ToBF16(sv);
    }
  }

  Tensor t_h = MakeT(hidden.data(), DType::kBF16, {1, H});
  Tensor t_p = MakeT(pos.data(), DType::kI32, {1});
  Tensor t_cs = MakeT(packed.data(), DType::kBF16, {kv, rot});
  Tensor t_cos = MakeT(cos.data(), DType::kF32, {kv, rot});
  Tensor t_sin = MakeT(sin.data(), DType::kF32, {kv, rot});

  int64_t visited = -1;
  vllm::Qwen4ExpQsaBlockOutput o = vllm::RunQwen4ExpQsaBlock(
      d, w, p, t_h, t_p, t_cs, t_cos, t_sin, caches.t, /*past_len=*/kv - 1, &visited);

  // 512 blocks * 4 rows + the 2-token ragged tail = 2050 rows, each read once
  // per query head per softmax pass. Re-derived here on purpose: a single-pass
  // online-softmax rewrite legitimately halves it, and that constant is where
  // the change would have to be argued rather than silently absorbed.
  constexpr int64_t kReadsPerRowPerHead = 2;
  const int64_t attended = p.qsa.block_topk() * p.qsa.compress_ratio + (kv - complete * 4);
  const int64_t want_reads = attended * HQ * kReadsPerRowPerHead;
  const int64_t dense_reads = kv * HQ * kReadsPerRowPerHead;
  INFO("keys_visited ", visited, " want ", want_reads, " dense ", dense_reads);
  CHECK(visited == want_reads);
  CHECK(visited < dense_reads);
  const std::vector<float> out = F32Of(o.tensor.Ptr<uint16_t>(), H);
  for (float v : out) CHECK(std::isfinite(v));
}

// ── 5b. THE PAGED CONSUMER (W5d-3, #2249 item 2) ────────────────────────────

namespace {

// The paged K/V the ENGINE allocates, laid out as the runner lays it out: the
// FlashAttention buffer `[num_pages, 2, kv_block_size, num_kv_heads, head_dim]`
// that `dense_attn::KvSlice` unbinds into the two rank-4 K and V views.
//
// EVERY ELEMENT STARTS AS NaN, and that is the instrument rather than hygiene.
// A correct read addresses exactly the rows this step's slot mapping wrote; any
// other row — an unnamed physical page, or the unused tail of the last named one
// — is not a number, so a mis-paged read cannot come back plausible. It is the
// same discriminator the gather-vs-mask case one section up uses, doing a second
// job: there `0.0f * NaN` convicts a mask, here it convicts a wrong ADDRESS.
struct PagedCaches {
  std::vector<uint16_t> buf;        // the whole flash cache, NaN-filled
  std::vector<uint16_t> index_key;  // the indexer side cache, STILL CONTIGUOUS
  std::vector<int32_t> table;       // [1, pages] logical page -> physical page
  std::vector<int64_t> slots;       // [T] i64 destination slot per new token
  vllm::Qwen4ExpQsaPagedCaches t;

  PagedCaches(int64_t num_pages, int64_t page, int64_t hkv, int64_t dh, int64_t idx_d,
              int64_t max_kv, const std::vector<int32_t>& block_table)
      : buf(static_cast<size_t>(num_pages * 2 * page * hkv * dh), vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN())),
        index_key(static_cast<size_t>(max_kv * idx_d), 0),
        table(block_table) {
    t.kv.data = buf.data();
    t.kv.dtype = DType::kBF16;
    t.kv.num_blocks = num_pages;
    t.kv.block_size = page;
    t.kv.num_kv_heads = hkv;
    t.kv.head_size = dh;
    t.block_table = MakeT(table.data(), DType::kI32,
                          {1, static_cast<int64_t>(table.size())});
    t.index_key = MakeT(index_key.data(), DType::kBF16, {max_kv, idx_d});
  }

  // The runner's own slot arithmetic: `block * block_size + offset`, for the T
  // tokens that land at logical positions [past_len, past_len + T).
  void SetSlots(int64_t past_len, int64_t T, const std::vector<int32_t>& read_table) {
    const int64_t page = t.kv.block_size;
    slots.resize(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) {
      const int64_t pos = past_len + i;
      slots[static_cast<size_t>(i)] =
          static_cast<int64_t>(read_table[static_cast<size_t>(pos / page)]) * page + pos % page;
    }
    t.slot_mapping = MakeT(slots.data(), DType::kI64, {T});
  }
};

}  // namespace

TEST_CASE("qwen4_exp qsa block: the PAGED consumer serves the cache the engine allocates") {
  // THE GAP THIS CLOSES, in #2249's own words: "`Qwen4ExpQsaCaches` is contiguous
  // `[max_kv, ...]`; `MakeQwen4ExpKVCache` publishes PAGED specs. The block landed
  // by W5b-5 reads the contiguous form, so nothing can serve from the cache the
  // engine actually allocates."
  //
  // THE BLOCK TABLE IS DELIBERATELY NOT THE IDENTITY, AND IT NAMES MORE THAN ONE
  // PAGE. Under `logical i -> physical i` a paged read and a contiguous read
  // return the same answer for every input, so an identity table would make this
  // case prove nothing at all — it would pass over a body that ignored the table.
  // `{5, 3, 7}` shares no fixed point with `{0, 1, 2}`, so the three pages an
  // identity-reading body would touch are exactly the three this one never
  // writes, and they stay NaN.
  //
  // THE LAST PAGE IS PARTIAL. 23 tokens over pages of 8 fill the third page's
  // rows 0..6 and leave row 7 NaN, so a body that reads a full page past the
  // visible length reads a NaN rather than a stale-but-finite value.
  constexpr double kOutTol = 3e-2;
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const Case& c = kOverBudget;  // 23 tokens: over budget, so the gather is sparse
  const int64_t H = p.hidden_size, rot = p.rotary_dim;
  const int64_t Hkv = p.num_key_value_heads, Dh = p.head_dim, ID = p.qsa.head_dim;
  const int64_t kPage = 8;       // a multiple of compress_ratio, as the KV spec requires
  const int64_t kNumPages = 8;   // more physical pages than the sequence needs
  const std::vector<int32_t> kPermuted{5, 3, 7};
  const std::vector<int32_t> kIdentity{0, 1, 2};
  REQUIRE(c.seq == 23);
  REQUIRE(kPage % p.qsa.compress_ratio == 0);

  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  RopeTables rope = BuildRope(c);
  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  Tensor t_h = MakeT(hidden.data(), DType::kBF16, {c.seq, H});
  Tensor t_p = MakeT(positions.data(), DType::kI32, {c.seq});
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});

  // The paged run, over a permuted table.
  PagedCaches paged(kNumPages, kPage, Hkv, Dh, ID, c.seq, kPermuted);
  paged.SetSlots(/*past_len=*/0, c.seq, kPermuted);
  int64_t paged_visited = 0;
  vllm::Qwen4ExpQsaBlockOutput po = vllm::RunQwen4ExpQsaBlockPaged(
      d, w, p, t_h, t_p, t_cs, t_cos, t_sin, paged.t, /*past_len=*/0, &paged_visited);
  const std::vector<float> got = F32Of(po.tensor.Ptr<uint16_t>(), c.seq * H);

  // 1. AGAINST THE ORACLE. The expectation is `Qwen4ExpTextAttention.forward`'s
  //    own output at the lane pin, the same golden the contiguous case answers
  //    to — an independently computed one, not a value read back from anything
  //    under test here.
  // FINITENESS FIRST, AND THE ORDER IS NOT COSMETIC. `MaxRelDiff` folds with
  // `std::max`, and `std::max(x, NaN)` returns `x` — so a run that comes back
  // ALL NaN reports a relative difference of exactly 0 and sails through the
  // bound below. That is a tolerance absorbing the defect in its purest form,
  // and it was MEASURED here rather than feared: with the paged address
  // resolution disarmed, every one of the 1472 outputs was NaN and `rel` still
  // printed 0. The finiteness loop is what convicts, and the bound is what says
  // the finite answer is the ORACLE's.
  for (float v : got) CHECK(std::isfinite(v));
  const double rel = MaxRelDiff(got, c.out, c.seq * H);
  INFO("paged block max relative difference vs the oracle ", rel);
  CHECK(rel < kOutTol);

  // 2. AGAINST THE CONTIGUOUS ARM, BIT FOR BIT, WITH NO TOLERANCE. Paging moves
  //    WHERE a row lives and nothing else: the same logical rows are visited in
  //    the same ascending order and reduced in the same f32 order, so the bf16
  //    stores must be EQUAL, not close. A tolerance here would absorb exactly the
  //    class of defect this case exists to find — a read one row or one page off
  //    lands inside a bf16-sized bound often enough to pass one.
  Caches contig(c.seq, Hkv, Dh, ID);
  int64_t contig_visited = 0;
  vllm::Qwen4ExpQsaBlockOutput co = vllm::RunQwen4ExpQsaBlock(
      d, w, p, t_h, t_p, t_cs, t_cos, t_sin, contig.t, /*past_len=*/0, &contig_visited);
  const uint16_t* pbits = po.tensor.Ptr<uint16_t>();
  const uint16_t* cbits = co.tensor.Ptr<uint16_t>();
  int64_t differing = 0;
  for (int64_t i = 0; i < c.seq * H; ++i) differing += (pbits[i] != cbits[i]) ? 1 : 0;
  INFO("paged vs contiguous differing bf16 words ", differing, " of ", c.seq * H);
  CHECK(differing == 0);
  // The same rows, therefore the same count of key-row reads. `keys_visited` is
  // counted AT THE READ (see the op's contract), so this says the paged walk did
  // the same amount of work and not merely that it agreed.
  CHECK(paged_visited == contig_visited);
  CHECK(paged_visited > 0);

  // 3. THE CONTROL: THE BLOCK TABLE IS ACTUALLY CONSULTED. Same inputs, same
  //    writes — the slot mapping still stores at the permuted pages — but the
  //    table handed to the READ is the identity. If the consumer ignored the
  //    table, or resolved a physical page any other way, this run would agree
  //    with the one above. It reads three never-written pages instead, so it
  //    comes back NaN, and the case fails if it does not.
  PagedCaches misread(kNumPages, kPage, Hkv, Dh, ID, c.seq, kIdentity);
  misread.SetSlots(/*past_len=*/0, c.seq, kPermuted);  // write permuted, read identity
  vllm::Qwen4ExpQsaBlockOutput mo = vllm::RunQwen4ExpQsaBlockPaged(
      d, w, p, t_h, t_p, t_cs, t_cos, t_sin, misread.t, /*past_len=*/0);
  const std::vector<float> mis = F32Of(mo.tensor.Ptr<uint16_t>(), c.seq * H);
  int64_t nan_rows = 0;
  for (int64_t t = 0; t < c.seq; ++t) {
    bool row_nan = false;
    for (int64_t j = 0; j < H; ++j)
      row_nan = row_nan || std::isnan(mis[static_cast<size_t>(t * H + j)]);
    nan_rows += row_nan ? 1 : 0;
  }
  INFO("identity-table control: NaN rows ", nan_rows, " of ", c.seq);
  CHECK(nan_rows == c.seq);
}

TEST_CASE("qwen4_exp qsa block: a PAGED decode step lands in the right page row") {
  // `past_len > 0` over a paged cache is where two off-by-ones meet: the slot the
  // new K/V is STORED at and the page the consumer READS the prefix from. The
  // prefill case above cannot see either — every token is written in one call
  // from position 0 — and the golden alone would not either, because a decode
  // that wrote one row off still produces finite, plausible output. The NaN fill
  // is what turns "plausible" into "not a number": row 7 of the last page is the
  // only row the 23-token sequence leaves unwritten, and it is exactly the row a
  // partial-final-page defect reaches for.
  constexpr double kOutTol = 3e-2;
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const Case& c = kOverBudget;
  const int64_t H = p.hidden_size, rot = p.rotary_dim;
  const int64_t Hkv = p.num_key_value_heads, Dh = p.head_dim, ID = p.qsa.head_dim;
  const int64_t kPage = 8;
  const std::vector<int32_t> kPermuted{5, 3, 7};
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  RopeTables rope = BuildRope(c);
  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});

  PagedCaches paged(8, kPage, Hkv, Dh, ID, c.seq, kPermuted);
  {  // prefill of the first seq-1 tokens — which STOPS mid-page, at row 6 of the
     // third page, so the decode token below is the one that fills row 6..
     // (22 tokens: pages 0 and 1 full, page 2 rows 0..5)
    paged.SetSlots(/*past_len=*/0, c.seq - 1, kPermuted);
    Tensor t_hh = MakeT(hidden.data(), DType::kBF16, {c.seq - 1, H});
    Tensor t_pp = MakeT(positions.data(), DType::kI32, {c.seq - 1});
    vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_hh, t_pp, t_cs, t_cos, t_sin, paged.t,
                                   /*past_len=*/0);
  }
  int64_t visited = 0;
  vllm::Qwen4ExpQsaBlockOutput o;
  {  // one decode token
    paged.SetSlots(/*past_len=*/c.seq - 1, 1, kPermuted);
    Tensor t_hh = MakeT(hidden.data() + (c.seq - 1) * H, DType::kBF16, {1, H});
    Tensor t_pp = MakeT(positions.data() + (c.seq - 1), DType::kI32, {1});
    o = vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_hh, t_pp, t_cs, t_cos, t_sin, paged.t,
                                       /*past_len=*/c.seq - 1, &visited);
  }
  const std::vector<float> got = F32Of(o.tensor.Ptr<uint16_t>(), H);
  // Finiteness before the bound, for the reason the prefill case above states:
  // `MaxRelDiff` cannot see a NaN.
  for (float v : got) CHECK(std::isfinite(v));
  const double rel = MaxRelDiff(got, c.out + (c.seq - 1) * H, H);
  INFO("paged decode-step max relative difference ", rel, ", keys_visited ", visited);
  CHECK(rel < kOutTol);
  CHECK(visited > 0);
}

TEST_CASE("qwen4_exp qsa block: the PAGED arm refuses by name") {
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const Case& c = kSubBudget;
  const int64_t H = p.hidden_size, rot = p.rotary_dim;
  const int64_t Hkv = p.num_key_value_heads, Dh = p.head_dim, ID = p.qsa.head_dim;
  const std::vector<int32_t> table{2, 0, 1};
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  RopeTables rope = BuildRope(c);
  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  Tensor t_h = MakeT(hidden.data(), DType::kBF16, {c.seq, H});
  Tensor t_p = MakeT(positions.data(), DType::kI32, {c.seq});
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});

  SUBCASE("an fp8 paged cache, which the QSA consumer has no dequantising read for") {
    PagedCaches pc(4, 8, Hkv, Dh, ID, c.seq, table);
    pc.SetSlots(0, c.seq, table);
    pc.t.kv.dtype = DType::kI8;
    pc.t.kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
    CHECK_THROWS_WITH_AS(vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_h, t_p, t_cs, t_cos, t_sin,
                                                        pc.t, /*past_len=*/0),
                         doctest::Contains("fp8"), std::exception);
  }
  SUBCASE("a KV page size the compress ratio does not divide") {
    PagedCaches pc(4, 6, Hkv, Dh, ID, c.seq, table);
    pc.SetSlots(0, c.seq, table);
    CHECK_THROWS_WITH_AS(vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_h, t_p, t_cs, t_cos, t_sin,
                                                        pc.t, /*past_len=*/0),
                         doctest::Contains("multiple of"), std::exception);
  }
  SUBCASE("a multi-request block table, which this block cannot serve yet") {
    PagedCaches pc(4, 8, Hkv, Dh, ID, c.seq, table);
    pc.SetSlots(0, c.seq, table);
    pc.t.block_table = MakeT(pc.table.data(), DType::kI32, {3, 1});
    CHECK_THROWS_WITH_AS(vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_h, t_p, t_cs, t_cos, t_sin,
                                                        pc.t, /*past_len=*/0),
                         doctest::Contains("ONE sequence"), std::exception);
  }
  SUBCASE("a block table naming fewer tokens than the sequence holds") {
    std::vector<int32_t> one{2};
    PagedCaches pc(4, 8, Hkv, Dh, ID, c.seq, one);
    pc.SetSlots(0, c.seq, std::vector<int32_t>{2, 2});
    CHECK_THROWS_WITH_AS(vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_h, t_p, t_cs, t_cos, t_sin,
                                                        pc.t, /*past_len=*/0),
                         doctest::Contains("fewer tokens than kv_len"), std::exception);
  }
  SUBCASE("a slot mapping of the wrong length") {
    PagedCaches pc(4, 8, Hkv, Dh, ID, c.seq, table);
    pc.SetSlots(0, c.seq - 1, table);
    CHECK_THROWS_WITH_AS(vllm::RunQwen4ExpQsaBlockPaged(d, w, p, t_h, t_p, t_cs, t_cos, t_sin,
                                                        pc.t, /*past_len=*/0),
                         doctest::Contains("slot_mapping"), std::exception);
  }
  SUBCASE("a kv_block_size with no page table, at the op") {
    // The two travel together or the paged read silently becomes a contiguous
    // one over a strided view — wrong rows, no message.
    std::vector<uint16_t> kv(static_cast<size_t>(c.seq * Hkv * Dh), 0);
    std::vector<uint16_t> ob(static_cast<size_t>(c.seq * p.num_attention_heads * Dh), 0);
    std::vector<int32_t> ids(static_cast<size_t>(c.seq * p.qsa.block_topk()), -1);
    std::vector<int32_t> lens(static_cast<size_t>(c.seq), 1);
    Tensor t_k = MakeT(kv.data(), DType::kBF16, {c.seq, Hkv, Dh});
    Tensor t_o = MakeT(ob.data(), DType::kBF16, {c.seq, p.num_attention_heads, Dh});
    Tensor t_q = MakeT(ob.data(), DType::kBF16, {c.seq, p.num_attention_heads, Dh});
    Tensor t_i = MakeT(ids.data(), DType::kI32, {c.seq, p.qsa.block_topk()});
    Tensor t_l = MakeT(lens.data(), DType::kI32, {c.seq});
    vt::Qwen4ExpQsaAttnArgs a;
    a.scale = 1.0f;
    a.compress_ratio = p.qsa.compress_ratio;
    a.kv_block_size = 8;  // set, with no table
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaGatherAttention(q, t_o, t_q, t_k, t_k, t_i, t_l, a),
        doctest::Contains("TOGETHER"), std::exception);
  }
}

// ── 6. REFUSALS ─────────────────────────────────────────────────────────────

TEST_CASE("qwen4_exp qsa block: refuses by name rather than computing something else") {
  const Qwen4ExpParams p = GoldenParams();
  const Qwen4ExpQsaWeights w = GoldenWeights(DType::kBF16);
  const Case& c = kSubBudget;
  const int64_t H = p.hidden_size, rot = p.rotary_dim;
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  RopeTables rope = BuildRope(c);
  std::vector<uint16_t> hidden = Bf16Of(c.hidden, c.seq * H);
  std::vector<float> hidden_f32(c.hidden, c.hidden + c.seq * H);
  std::vector<int32_t> positions(static_cast<size_t>(c.seq));
  for (int64_t t = 0; t < c.seq; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  Tensor t_h = MakeT(hidden.data(), DType::kBF16, {c.seq, H});
  Tensor t_p = MakeT(positions.data(), DType::kI32, {c.seq});
  Tensor t_cs = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
  Tensor t_cos = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
  Tensor t_sin = MakeT(rope.sin.data(), DType::kF32, {c.seq, rot});

  SUBCASE("an f32 hidden, because every vt:: output gate stores bf16") {
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    Tensor bad = MakeT(hidden_f32.data(), DType::kF32, {c.seq, H});
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, p, bad, t_p, t_cs, t_cos, t_sin, caches.t, 0),
        doctest::Contains("hidden must be bf16"), std::exception);
  }
  SUBCASE("an f32 cos_sin cache, which vt::RopeFromCache could not read") {
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    Tensor bad = MakeT(rope.cos.data(), DType::kF32, {c.seq, rot});
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, bad, t_cos, t_sin, caches.t, 0),
        doctest::Contains("PACKED cos|sin cache"), std::exception);
  }
  // ─── THE TWO ROPE LAYOUTS MUST DESCRIBE THE SAME ANGLES ──────────────────
  // The block takes one set of angles TWICE, in the two layouts its two ops were
  // ported to read. Nothing in the type system forces a caller to build both
  // from one table, and a layer loop that does not diverges SILENTLY: the query
  // would be roped with one set of angles and the pooled indexer keys with
  // another, and every value stays finite.
  //
  // These two subcases are what makes the header's cross-check claim executable.
  // `BuildRope` above derives `packed` FROM `cos`/`sin`, so the two agree BY
  // CONSTRUCTION in every other case in this file — agreement by construction is
  // not an assertion, and before the cross-check landed both of these ran the
  // whole block to a finite answer and threw nothing.
  SUBCASE("two rope layouts built from DIFFERENT tables") {
    // The perturbed row is `c.seq - 1`, which NOTHING ELSE IN THE BLOCK READS:
    // `vt::RopeFromCache` reads the PACKED cache, not this one, and the
    // compressor reads only BLOCK-START rows — multiples of `compress_ratio`,
    // and 10 is not one at seq 11. So this divergence is invisible to every
    // value gate in this file, which is exactly why it needs its own.
    REQUIRE((c.seq - 1) % p.qsa.compress_ratio != 0);
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    std::vector<float> skewed = rope.cos;
    skewed[static_cast<size_t>((c.seq - 1) * rot)] += 0.5f;
    Tensor bad = MakeT(skewed.data(), DType::kF32, {c.seq, rot});
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, bad, t_sin, caches.t, 0),
        doctest::Contains("same angles"), std::exception);
  }
  SUBCASE("rope tables of DIFFERENT lengths") {
    // The two layouts are documented as `[P, rotary_dim]` for one `P`, and
    // nothing checked it. It is a refusal in its own right — a packed cache and
    // a full table of different heights cannot have come from one build — and it
    // is also the precondition of the row sample above, which indexes `cos` at
    // rows it takes from `cos_sin`.
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    Tensor bad = MakeT(rope.cos.data(), DType::kF32, {c.seq - 1, rot});
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, bad, t_sin, caches.t, 0),
        doctest::Contains("same number of rows"), std::exception);
  }
  SUBCASE("bf16 cos/sin for the compressor, which requires f32") {
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    Tensor bad = MakeT(rope.packed.data(), DType::kBF16, {c.seq, rot});
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, bad, t_sin, caches.t, 0),
        doctest::Contains("FULL tables"), std::exception);
  }
  SUBCASE("a cache too short for the new tokens") {
    Caches caches(c.seq - 1, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, p, t_h, t_p, t_cs, t_cos, t_sin, caches.t, 0),
        doctest::Contains("do not fit"), std::exception);
  }
  SUBCASE("a rotary_dim wider than the indexer head, which upstream rejects") {
    Qwen4ExpParams bad = p;
    bad.rotary_dim = g::kIndexHeadDim + 2;
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, bad, t_h, t_p, t_cs, t_cos, t_sin, caches.t, 0),
        doctest::Contains("indexer_head_dim"), std::exception);
  }
  SUBCASE("num_attention_heads not a multiple of num_key_value_heads") {
    Qwen4ExpParams bad = p;
    bad.num_key_value_heads = 3;
    Caches caches(c.seq, 3, p.head_dim, p.qsa.head_dim);
    CHECK_THROWS_WITH_AS(
        vllm::RunQwen4ExpQsaBlock(d, w, bad, t_h, t_p, t_cs, t_cos, t_sin, caches.t, 0),
        doctest::Contains("multiple of num_key_value_heads"), std::exception);
  }
  SUBCASE("a logits tap of the wrong shape") {
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    std::vector<uint16_t> qi(static_cast<size_t>(c.seq * p.qsa.n_heads * p.qsa.head_dim), 0);
    Tensor t_q3 = MakeT(qi.data(), DType::kBF16, {c.seq, p.qsa.n_heads, p.qsa.head_dim});
    std::vector<int32_t> lens(static_cast<size_t>(c.seq));
    for (int64_t t = 0; t < c.seq; ++t) lens[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
    Tensor t_len = MakeT(lens.data(), DType::kI32, {c.seq});
    std::vector<float> lg(static_cast<size_t>(c.seq), 0.0f);
    Tensor bad = MakeT(lg.data(), DType::kF32, {c.seq, 1});
    CHECK_THROWS_WITH_AS(
        vllm::Qwen4ExpQsaIndex(d, p.qsa, static_cast<float>(p.rms_norm_eps), t_q3, caches.t.index_key,
                               vllm::dense_attn::ResidentWeight(d, w.idx_k_norm,
                                                                {p.qsa.head_dim}),
                               t_cos, t_sin, t_len, c.seq, true, &bad),
        doctest::Contains("logits tap"), std::exception);
  }
  SUBCASE("indexer_kv_heads != 1, which upstream requires") {
    Qwen4ExpParams bad = p;
    bad.qsa.kv_heads = 2;
    Caches caches(c.seq, p.num_key_value_heads, p.head_dim, p.qsa.head_dim);
    std::vector<uint16_t> qi(static_cast<size_t>(c.seq * p.qsa.n_heads * p.qsa.head_dim), 0);
    Tensor t_q3 = MakeT(qi.data(), DType::kBF16, {c.seq, p.qsa.n_heads, p.qsa.head_dim});
    std::vector<int32_t> lens(static_cast<size_t>(c.seq));
    for (int64_t t = 0; t < c.seq; ++t) lens[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
    Tensor t_len = MakeT(lens.data(), DType::kI32, {c.seq});
    CHECK_THROWS_WITH_AS(
        vllm::Qwen4ExpQsaIndex(d, bad.qsa, static_cast<float>(p.rms_norm_eps), t_q3, caches.t.index_key,
                               vllm::dense_attn::ResidentWeight(d, w.idx_k_norm,
                                                                {p.qsa.head_dim}),
                               t_cos, t_sin, t_len, c.seq, true),
        doctest::Contains("indexer_kv_heads == 1"), std::exception);
  }
}
