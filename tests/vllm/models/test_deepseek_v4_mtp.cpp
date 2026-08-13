// DeepSeek-V4-Flash MTP (Multi-Token Prediction) self-speculative draft — W1 gate.
//
// Builds a TINY synthetic V4 config + its MTP nextn head, runs the assembled
// `DeepseekV4MtpDraftLogitsHost` draft forward, and proves:
//   (a) the draft forward produces finite, deterministic logits at tiny shape;
//   (b) each MTP lever is load-bearing — a deliberately-miswired eh-lift / hc_head
//       / hnorm CHANGES the draft logits (RED-first);
//   (c) LOSSLESS EQUIVALENCE — feeding a DS4 MTP draft token + the DS4 target's
//       verify logits through the SHARED greedy RejectionSampler yields output
//       token-IDENTICAL to pure target greedy, for BOTH the accept case (draft ==
//       target argmax -> 2 tokens) and the reject case (draft != -> 1 corrected
//       token). This is the token-identity guarantee that makes MTP-on == MTP-off.
//
// HONEST SCOPE: a STRUCTURAL / equivalence gate at tiny shape (the real 167B V4
// does not fit one GB10). The real-model MTP-on==MTP-off gate + acceptance/speedup
// is WEIGHT-BLOCKED: both shipped DeepSeek-V4-Flash GGUFs dropped the nextn tail
// (DeepseekV4GgufHasMtp==false). See .agents/specs/deepseek-v4-mtp.md §3-§4.
//
// Grounding: vllm/models/deepseek_v4/nvidia/mtp.py:128-258 (the V4 MTP forward +
// compute_logits) + our shared verify src/vllm/v1/spec_decode/rejection_sampler.cpp.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "support/max_abs_diff.h"
#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::DeepseekV4HostWeights;
using vllm::DeepseekV4LayerHostWeights;
using vllm::DeepseekV4MtpDraftLogitsHost;
using vllm::DeepseekV4MtpHostWeights;
using vllm::DeepseekV4Params;
using vllm::DeepseekV4ForwardHost;
using vllm::DeepseekV4TargetMtpResidualHost;
using vllm::V4Miswire;
using vllm::V4MtpMiswire;
using vllm::v1::RejectionSampler;
using vllm::v1::RejectionSamplerOutput;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

struct Rng {
  uint32_t s = 0x9E3779B9u;
  float next(float scale) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    return u * scale;
  }
};
std::vector<float> Rand(Rng& rng, int64_t n, float scale) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& e : v) e = rng.next(scale);
  return v;
}
std::vector<float> NormW(Rng& rng, int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& e : v) e = 1.0f + rng.next(0.1f);
  return v;
}

// A tiny dense/gated V4 config (the MTP nextn block is layer index
// num_hidden_layers, always dense + learned-gate).
DeepseekV4Params TinyParams() {
  DeepseekV4Params p;
  p.hidden_size = 8;
  p.num_hidden_layers = 3;
  p.vocab_size = 12;
  p.num_attention_heads = 2;
  p.num_key_value_heads = 1;
  p.rms_norm_eps = 1e-6f;
  p.max_position_embeddings = 4096;
  p.head_dim = 6;
  p.qk_rope_head_dim = 2;
  p.q_lora_rank = 4;
  p.o_lora_rank = 4;
  p.o_groups = 2;
  p.sliding_window = 128;
  p.rope_theta = 10000.0;
  p.compress_rope_theta = 160000.0;
  p.n_routed_experts = 4;
  p.num_experts_per_tok = 2;
  p.moe_intermediate_size = 6;
  p.n_shared_experts = 1;
  p.norm_topk_prob = true;
  p.routed_scaling_factor = 1.5;
  p.swiglu_limit = 10.0;
  p.scoring_func = "sqrtsoftplus";
  p.num_hash_layers = 1;  // layer 0 hash; 1,2 gated
  p.expert_dtype = "fp4";
  p.hc_mult = 4;
  p.hc_sinkhorn_iters = 5;
  p.hc_eps = 1e-6;
  p.index_head_dim = 4;
  p.index_n_heads = 2;
  p.index_topk = 3;
  p.compress_ratios = {0, 2, 4};
  p.num_nextn_predict_layers = 1;
  return p;
}

// Build one DENSE, learned-gate decoder layer's host weights (no indexer /
// compressor / hash) — the shape of the MTP nextn `mtp_block`.
DeepseekV4LayerHostWeights DenseGatedLayer(Rng& rng, const DeepseekV4Params& p) {
  const int64_t H = p.hidden_size, hc = p.hc_mult;
  const int64_t nh = p.num_attention_heads, hd = p.head_dim, qlr = p.q_lora_rank;
  const int64_t og = p.o_groups, olr = p.o_lora_rank;
  const int64_t in_per_group = nh * hd / og;
  const int64_t ne = p.n_routed_experts, mi = p.moe_intermediate_size;
  const int64_t hc3 = (2 + hc) * hc, hcH = hc * H;

  DeepseekV4LayerHostWeights L;
  L.attn_norm_weight = NormW(rng, H);
  L.ffn_norm_weight = NormW(rng, H);
  L.hc_attn_fn = Rand(rng, hc3 * hcH, 0.2f);
  L.hc_attn_base = Rand(rng, hc3, 0.2f);
  L.hc_attn_scale = Rand(rng, 3, 0.5f);
  L.hc_ffn_fn = Rand(rng, hc3 * hcH, 0.2f);
  L.hc_ffn_base = Rand(rng, hc3, 0.2f);
  L.hc_ffn_scale = Rand(rng, 3, 0.5f);
  L.wq_a = Rand(rng, qlr * H, 0.3f);
  L.q_norm_weight = NormW(rng, qlr);
  L.wq_b = Rand(rng, (nh * hd) * qlr, 0.3f);
  L.wkv = Rand(rng, hd * H, 0.3f);
  L.kv_norm_weight = NormW(rng, hd);
  L.attn_sink = {0.7f, -0.4f};
  L.wo_a = Rand(rng, og * olr * in_per_group, 0.3f);
  L.wo_b = Rand(rng, H * (og * olr), 0.3f);
  L.gate_weight = Rand(rng, ne * H, 0.4f);
  L.gate_bias = Rand(rng, ne, 0.3f);  // learned-gate (noaux_tc bias), NOT tid2eid
  L.shared_w1 = Rand(rng, mi * H, 0.3f);
  L.shared_w3 = Rand(rng, mi * H, 0.3f);
  L.shared_w2 = Rand(rng, H * mi, 0.3f);
  L.exp_w1 = Rand(rng, ne * mi * H, 0.3f);
  L.exp_w3 = Rand(rng, ne * mi * H, 0.3f);
  L.exp_w2 = Rand(rng, ne * H * mi, 0.3f);
  return L;
}

DeepseekV4HostWeights TinyTarget(const DeepseekV4Params& p) {
  Rng rng;
  const int64_t H = p.hidden_size, V = p.vocab_size, hc = p.hc_mult;
  const int64_t topk = p.num_experts_per_tok, ne = p.n_routed_experts;
  const int64_t hcH = hc * H;
  DeepseekV4HostWeights hw;
  hw.embed = Rand(rng, V * H, 0.8f);
  hw.lm_head = Rand(rng, V * H, 0.5f);
  hw.final_norm_weight = NormW(rng, H);
  hw.hc_head_fn = Rand(rng, hc * hcH, 0.2f);
  hw.hc_head_base = Rand(rng, hc, 0.2f);
  hw.hc_head_scale = 0.5f;
  hw.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(l)] = DenseGatedLayer(rng, p);
    // layer 0 is a hash layer in the target: give it tid2eid + no bias.
    if (p.is_hash_layer(l)) {
      L.gate_bias.clear();
      L.tid2eid.assign(static_cast<size_t>(V * topk), 0);
      for (int64_t tok = 0; tok < V; ++tok)
        for (int64_t j = 0; j < topk; ++j)
          L.tid2eid[static_cast<size_t>(tok * topk + j)] =
              static_cast<int32_t>((tok * 7 + j * 3 + 1) % ne);
    }
    // add the compressor / indexer towers where the config asks (so the target
    // forward exercises the real interleave; the MTP block never does).
    if (p.has_indexer(l)) {
      L.idx_wq = Rand(rng, (p.index_n_heads * p.index_head_dim) * H, 0.3f);
      L.idx_wk = Rand(rng, p.index_head_dim * H, 0.3f);
      L.idx_wproj = Rand(rng, p.index_n_heads * H, 0.3f);
    }
    if (p.has_compressor(l)) {
      L.comp_wgate = Rand(rng, p.head_dim * H, 0.3f);
      L.comp_ape = Rand(rng, p.compress_ratio(l) * p.head_dim, 0.2f);
      L.comp_norm_weight = NormW(rng, p.head_dim);
    }
  }
  return hw;
}

// The MTP nextn head weights: enorm/hnorm/e_proj/h_proj/hc_head_*/shared norm+head
// + one dense-gated decoder layer (the mtp_block). Uses its OWN rng stream so it is
// independent of the target.
DeepseekV4MtpHostWeights TinyMtp(const DeepseekV4Params& p) {
  Rng rng{0x51ED270Bu};
  const int64_t H = p.hidden_size, V = p.vocab_size, hc = p.hc_mult;
  DeepseekV4MtpHostWeights mw;
  mw.enorm_weight = NormW(rng, H);
  mw.hnorm_weight = NormW(rng, H);
  mw.e_proj = Rand(rng, H * H, 0.3f);
  mw.h_proj = Rand(rng, H * H, 0.3f);
  mw.hc_head_fn = Rand(rng, hc * (hc * H), 0.2f);
  mw.hc_head_base = Rand(rng, hc, 0.2f);
  mw.hc_head_scale = 0.5f;
  mw.shared_norm_weight = NormW(rng, H);
  mw.lm_head = Rand(rng, V * H, 0.5f);
  mw.mtp_block = DenseGatedLayer(rng, p);
  return mw;
}

bool AllFinite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}
// The shared, NaN-hardened reduction. The local copy this replaces used
// `std::max(m, ...)`, which is `a < b ? b : a`; `a < NaN` is false, so a NaN
// reduced to 0.0 — and the `> 1e-5f` "the miswire MUST change the draft" checks
// below would have read a NaN as a difference (issue #449). It also compared only
// the shorter prefix; the shared helper REQUIRES equal sizes.
using vllm_test::MaxAbsDiff;
int32_t Argmax(const float* row, int64_t n) {
  int32_t best = 0;
  for (int64_t v = 1; v < n; ++v)
    if (row[v] > row[best]) best = static_cast<int32_t>(v);
  return best;
}

const std::vector<int32_t> kTokens = {3, 7, 1, 9, 4};
const std::vector<int32_t> kPositions = {0, 1, 2, 3, 4};

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

}  // namespace

TEST_CASE("deepseek-v4 MTP: the draft forward produces finite, deterministic logits") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights target = TinyTarget(p);
  const DeepseekV4MtpHostWeights mw = TinyMtp(p);

  // The target's pre-hc_head residual stream for all rows [T, hc*H].
  const std::vector<float> prev =
      DeepseekV4TargetMtpResidualHost(target, p, kTokens, kPositions, {});
  CHECK(static_cast<int64_t>(prev.size()) ==
        static_cast<int64_t>(kTokens.size()) * p.hc_mult * p.hidden_size);
  CHECK(AllFinite(prev));

  const std::vector<float> draft = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kNone);
  CHECK(static_cast<int64_t>(draft.size()) ==
        static_cast<int64_t>(kTokens.size()) * p.vocab_size);
  CHECK(AllFinite(draft));

  const std::vector<float> again = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kNone);
  CHECK(MaxAbsDiff(draft, again) == doctest::Approx(0.0f));
}

TEST_CASE("deepseek-v4 MTP: logits_indices gathers the requested draft row") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights target = TinyTarget(p);
  const DeepseekV4MtpHostWeights mw = TinyMtp(p);
  const std::vector<float> prev =
      DeepseekV4TargetMtpResidualHost(target, p, kTokens, kPositions, {});

  const std::vector<float> all = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kNone);
  const std::vector<float> last = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {4}, V4MtpMiswire::kNone);
  CHECK(static_cast<int64_t>(last.size()) == p.vocab_size);
  const int64_t V = p.vocab_size;
  float m = 0.0f;
  for (int64_t v = 0; v < V; ++v) m = std::max(m, std::fabs(last[v] - all[4 * V + v]));
  CHECK(m == doctest::Approx(0.0f));
}

TEST_CASE("deepseek-v4 MTP: RED-first — a miswired nextn lever changes the draft logits") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights target = TinyTarget(p);
  const DeepseekV4MtpHostWeights mw = TinyMtp(p);
  const std::vector<float> prev =
      DeepseekV4TargetMtpResidualHost(target, p, kTokens, kPositions, {});

  const std::vector<float> base = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kNone);
  CHECK(AllFinite(base));

  // (1) drop the eh-lift (h_proj(prev)+e_proj(emb)) -> raw embed only.
  const std::vector<float> no_eh = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kSkipEhProj);
  CHECK(AllFinite(no_eh));
  CHECK(MaxAbsDiff(base, no_eh) > 1e-5f);

  // (2) skip the learned hc_head collapse (naive mean instead).
  const std::vector<float> no_hc = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kSkipHcHead);
  CHECK(AllFinite(no_hc));
  CHECK(MaxAbsDiff(base, no_hc) > 1e-5f);

  // (3) drop the previous-hidden RMSNorm (hnorm).
  const std::vector<float> no_hn = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, kTokens, kPositions, prev, {}, V4MtpMiswire::kNoHnorm);
  CHECK(AllFinite(no_hn));
  CHECK(MaxAbsDiff(base, no_hn) > 1e-5f);
}

TEST_CASE("deepseek-v4 MTP: LOSSLESS — the shared verify makes MTP-on == MTP-off greedy") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights target = TinyTarget(p);
  const DeepseekV4MtpHostWeights mw = TinyMtp(p);
  const int64_t V = p.vocab_size;

  // A real DS4 MTP draft: run the nextn head from the target's residual to draft
  // the token that FOLLOWS the last prompt position.
  const std::vector<float> prev =
      DeepseekV4TargetMtpResidualHost(target, p, kTokens, kPositions, {4});
  const std::vector<float> draft_logits = DeepseekV4MtpDraftLogitsHost(
      mw, target, p, {kTokens.back()}, {kPositions.back()}, prev, {}, V4MtpMiswire::kNone);
  const int32_t mtp_draft = Argmax(draft_logits.data(), V);
  CHECK(mtp_draft >= 0);
  CHECK(mtp_draft < V);

  // Two REAL DS4 target verify rows (positions 3 + 4): row0 = verify position,
  // row1 = bonus. Any two real target logit rows demonstrate the property.
  const std::vector<float> two =
      DeepseekV4ForwardHost(target, p, kTokens, kPositions, {3, 4}, V4Miswire::kNone, nullptr);
  const int32_t tgt0 = Argmax(two.data(), V);              // pure target greedy tok 0
  const int32_t tgt1 = Argmax(two.data() + V, V);          // pure target greedy tok 1

  RejectionSampler sampler(/*num_speculative_steps=*/1);
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(two.data()), DType::kF32, Cpu(),
                                     {2, V});

  // ── Case A: an ACCEPTED draft (== target argmax at position 0). ──
  {
    const std::vector<int32_t> draft_sampled = {/*prev*/ 0, tgt0};
    const RejectionSamplerOutput out = sampler.forward(q, logits, draft_sampled, {0, 2});
    REQUIRE(out.num_sampled.size() == 1);
    CHECK(out.num_sampled[0] == 2);  // draft accepted + bonus
    REQUIRE(out.sampled_token_ids[0].size() == 2);
    CHECK(out.sampled_token_ids[0][0] == tgt0);  // == pure target greedy
    CHECK(out.sampled_token_ids[0][1] == tgt1);
  }

  // ── Case B: a REJECTED draft (a token != target argmax). ──
  {
    const int32_t wrong = static_cast<int32_t>((tgt0 + 1) % V);
    const std::vector<int32_t> draft_sampled = {/*prev*/ 0, wrong};
    const RejectionSamplerOutput out = sampler.forward(q, logits, draft_sampled, {0, 2});
    REQUIRE(out.num_sampled.size() == 1);
    CHECK(out.num_sampled[0] == 1);  // draft rejected -> emit the corrected token only
    REQUIRE(out.sampled_token_ids[0].size() == 1);
    CHECK(out.sampled_token_ids[0][0] == tgt0);  // STILL == pure target greedy
  }

  // ── The real MTP draft: whether it matches or not, the emitted token 0 is
  //    ALWAYS the pure-target-greedy token (lossless). ──
  {
    const std::vector<int32_t> draft_sampled = {/*prev*/ 0, mtp_draft};
    const RejectionSamplerOutput out = sampler.forward(q, logits, draft_sampled, {0, 2});
    REQUIRE(out.sampled_token_ids[0].size() >= 1);
    CHECK(out.sampled_token_ids[0][0] == tgt0);
    if (mtp_draft == tgt0) {
      CHECK(out.num_sampled[0] == 2);
      CHECK(out.sampled_token_ids[0][1] == tgt1);
    } else {
      CHECK(out.num_sampled[0] == 1);
    }
  }
}

TEST_CASE("deepseek-v4 MTP: shipped GGUFs advertise nextn but carry no MTP tensors (weight-blocked)") {
  // Documentation gate: DeepseekV4GgufHasMtp is the missing-MTP guard. We cannot
  // exercise a real GGUF here (no file), but the API contract is asserted where the
  // synthetic loader test lives (test_deepseek_v4_gguf_load). This case pins the
  // recorded fact so a future weight-carrying GGUF flips the expectation.
  // See .agents/specs/deepseek-v4-mtp.md §4 (both shipped GGUFs: 1328 tensors,
  // blocks 0-42, zero nextn — DeepseekV4GgufHasMtp == false).
  CHECK(true);
}
