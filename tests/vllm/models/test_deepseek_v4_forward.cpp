// DeepSeek-V4-Flash W7 — the DeepseekV4Model::Forward ASSEMBLY structural gate.
// Builds a TINY synthetic V4 config (a few layers / small hidden / few experts /
// hc_mult=4 exercising the hash-vs-gated split, the MHC interleave, the DSA sparse
// path, the compressor KV) and runs the assembled `DeepseekV4ForwardHost` forward.
//
// HONEST SCOPE: this is a STRUCTURAL / composition gate — it asserts the interleave
// RAN as designed (the [T,hc,H] MHC stream, hash layers route by tid, the DSA
// Lightning-Indexer selects + the compressor pools, shapes/finiteness hold) and
// that deliberately-miswired interleaves change the output (RED-first). It is NOT a
// real-checkpoint token gate — the fixed-config 167B V4 (156.7 GiB) does not fit
// one GB10, so the strict token gate is the multi-Spark W8 residual. See
// .agents/specs/deepseek-v4-flash.md §W7.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "support/max_abs_diff.h"

using vllm::DeepseekV4ForwardHost;
using vllm::DeepseekV4HostWeights;
using vllm::DeepseekV4LayerHostWeights;
using vllm::DeepseekV4Params;
using vllm::V4ForwardTrace;
using vllm::V4Miswire;

namespace {

// Deterministic small-magnitude weight generator so activations stay O(1) and the
// fp8_ds_mla / clamped-SwiGLU / Sinkhorn compositions round-trip meaningfully.
struct Rng {
  uint32_t s = 0x243F6A88u;
  float next(float scale) {
    s = s * 1664525u + 1013904223u;
    // map to [-1, 1) then scale
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    return u * scale;
  }
};

std::vector<float> Rand(Rng& rng, int64_t n, float scale) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& e : v) e = rng.next(scale);
  return v;
}
// Norm weights hover around 1.0 (so RMSNorm outputs are O(1)).
std::vector<float> NormW(Rng& rng, int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& e : v) e = 1.0f + rng.next(0.1f);
  return v;
}

// A tiny V4 config that exercises every W3-W6 primitive path.
DeepseekV4Params TinyParams() {
  DeepseekV4Params p;
  p.hidden_size = 8;
  p.num_hidden_layers = 4;
  p.vocab_size = 12;
  p.num_attention_heads = 2;
  p.num_key_value_heads = 1;
  p.rms_norm_eps = 1e-6f;
  p.max_position_embeddings = 4096;
  p.head_dim = 6;          // 4 NoPE + 2 RoPE (mirrors the 448+64=512 split at tiny width)
  p.qk_rope_head_dim = 2;
  p.q_lora_rank = 4;
  p.o_lora_rank = 4;
  p.o_groups = 2;          // heads_per_group = 1, in_per_group = 6
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
  p.num_hash_layers = 2;   // layers 0,1 hash-routed; 2,3 gated
  p.expert_dtype = "fp4";
  p.hc_mult = 4;
  p.hc_sinkhorn_iters = 5;
  p.hc_eps = 1e-6;
  p.index_head_dim = 4;
  p.index_n_heads = 2;
  p.index_topk = 3;
  // layer 0: neither; layer 1: indexer+compressor (cr 4); layer 2: compressor (cr 2);
  // layer 3: indexer+compressor (cr 4).
  p.compress_ratios = {0, 4, 2, 4};
  return p;
}

DeepseekV4HostWeights TinyWeights(const DeepseekV4Params& p) {
  Rng rng;
  const int64_t H = p.hidden_size, V = p.vocab_size, hc = p.hc_mult;
  const int64_t nh = p.num_attention_heads, hd = p.head_dim, qlr = p.q_lora_rank;
  const int64_t og = p.o_groups, olr = p.o_lora_rank;
  const int64_t in_per_group = nh * hd / og;
  const int64_t ne = p.n_routed_experts, topk = p.num_experts_per_tok, mi = p.moe_intermediate_size;
  const int64_t inh = p.index_n_heads, ihd = p.index_head_dim;
  const int64_t hc3 = (2 + hc) * hc, hcH = hc * H;

  DeepseekV4HostWeights hw;
  hw.embed = Rand(rng, V * H, 0.8f);
  hw.lm_head = Rand(rng, V * H, 0.5f);
  hw.final_norm_weight = NormW(rng, H);
  hw.hc_head_fn = Rand(rng, hc * hcH, 0.2f);
  hw.hc_head_base = Rand(rng, hc, 0.2f);
  hw.hc_head_scale = 0.5f;

  hw.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(l)];
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
    L.attn_sink = {0.7f, -0.4f};  // non-trivial per-head sinks (kNoAttnSink must differ)
    L.wo_a = Rand(rng, og * olr * in_per_group, 0.3f);
    L.wo_b = Rand(rng, H * (og * olr), 0.3f);

    if (p.has_indexer(l)) {
      L.idx_wq = Rand(rng, (inh * ihd) * H, 0.3f);
      L.idx_wk = Rand(rng, ihd * H, 0.3f);
      L.idx_wproj = Rand(rng, inh * H, 0.3f);
    }
    if (p.has_compressor(l)) {
      const int64_t cr = p.compress_ratio(l);
      L.comp_wgate = Rand(rng, hd * H, 0.3f);
      L.comp_ape = Rand(rng, cr * hd, 0.2f);
      L.comp_norm_weight = NormW(rng, hd);
    }

    L.gate_weight = Rand(rng, ne * H, 0.4f);
    if (p.is_hash_layer(l)) {
      // tid2eid [vocab, topk]: a deterministic token-id -> expert table that is
      // INDEPENDENT of the gating logits, so the hash route reliably differs from
      // the learned top-k (the kAllLayersGated RED-first lever).
      L.tid2eid.assign(static_cast<size_t>(V * topk), 0);
      for (int64_t tok = 0; tok < V; ++tok)
        for (int64_t j = 0; j < topk; ++j)
          L.tid2eid[static_cast<size_t>(tok * topk + j)] =
              static_cast<int32_t>((tok * 7 + j * 3 + 1) % ne);
    } else {
      L.gate_bias = Rand(rng, ne, 0.3f);
    }

    L.shared_w1 = Rand(rng, mi * H, 0.3f);
    L.shared_w3 = Rand(rng, mi * H, 0.3f);
    L.shared_w2 = Rand(rng, H * mi, 0.3f);
    L.exp_w1 = Rand(rng, ne * mi * H, 0.3f);
    L.exp_w3 = Rand(rng, ne * mi * H, 0.3f);
    L.exp_w2 = Rand(rng, ne * H * mi, 0.3f);
  }
  return hw;
}

bool AllFinite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}
// The shared, NaN-hardened reduction. The local copy this replaces used
// `std::max(m, ...)`, which is `a < b ? b : a`; `a < NaN` is false, so a NaN
// reduced to 0.0 — and the `> 1e-5f` "the miswire MUST change the output" checks
// below would have read a NaN as a difference (issue #449). It also compared only
// the shorter prefix; the shared helper REQUIRES equal sizes.
using vllm_test::MaxAbsDiff;

const std::vector<int32_t> kTokens = {3, 7, 1, 9, 4};
const std::vector<int32_t> kPositions = {0, 1, 2, 3, 4};

}  // namespace

TEST_CASE("deepseek-v4 W7: the assembled forward produces finite logits end-to-end") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights hw = TinyWeights(p);

  V4ForwardTrace trace;
  const std::vector<float> logits =
      DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, &trace);

  // shape: [T, vocab] flattened (all rows, logits_indices empty).
  CHECK(static_cast<int64_t>(logits.size()) ==
        static_cast<int64_t>(kTokens.size()) * p.vocab_size);
  CHECK(AllFinite(logits));

  // determinism: same inputs -> byte-identical outputs.
  const std::vector<float> again =
      DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, nullptr);
  CHECK(MaxAbsDiff(logits, again) == doctest::Approx(0.0f));
}

TEST_CASE("deepseek-v4 W7: the MHC [T,hc,H] stream + per-layer interleave RAN") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights hw = TinyWeights(p);
  V4ForwardTrace tr;
  (void)DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, &tr);

  // The residual manifold is [T, hc_mult, H] (the V4 hyper-connection topology).
  CHECK(tr.hc_mult == 4);
  CHECK(tr.hidden == 8);
  CHECK(tr.num_tokens == 5);
  CHECK(tr.residual_stream_elems == 5 * 4 * 8);  // T*hc*H
  // Every layer's attn + ffn MHC sub-blocks ran (trace vectors sized per layer).
  CHECK(static_cast<int64_t>(tr.layer_is_hash.size()) == p.num_hidden_layers);
}

TEST_CASE("deepseek-v4 W7: hash layers route by tid, gated layers by learned top-k") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights hw = TinyWeights(p);
  V4ForwardTrace tr;
  (void)DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, &tr);

  // num_hash_layers=2 -> layers 0,1 config-hash; 2,3 learned-gated.
  CHECK(tr.layer_is_hash == std::vector<int>{1, 1, 0, 0});
  // The router took the hash (tid2eid) branch exactly on the hash layers.
  CHECK(tr.layer_hash_routed == std::vector<int>{1, 1, 0, 0});
}

TEST_CASE("deepseek-v4 W7: the DSA sparse path selects + the compressor pools") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights hw = TinyWeights(p);
  V4ForwardTrace tr;
  (void)DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, &tr);

  // compress_ratios {0,4,2,4}: indexer on layers 1,3 (ratio==4); compressor on 1,2,3.
  CHECK(tr.layer_is_indexer == std::vector<int>{0, 1, 0, 1});
  CHECK(tr.layer_compressor_ran == std::vector<int>{0, 1, 1, 1});
  // The Lightning-Indexer top-k (index_topk=3) selected 3 keys for the last query
  // (5 causal candidates > 3 -> full top-k), on each indexer layer.
  CHECK(tr.layer_indexer_selected[1] == 3);
  CHECK(tr.layer_indexer_selected[3] == 3);
  // Non-indexer layers ran the dense causal path (indexer trace stays 0).
  CHECK(tr.layer_indexer_selected[0] == 0);
  CHECK(tr.layer_indexer_selected[2] == 0);
}

TEST_CASE("deepseek-v4 W7: logits_indices gathers the requested rows") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights hw = TinyWeights(p);

  const std::vector<float> all =
      DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, nullptr);
  const std::vector<float> last =
      DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {4}, V4Miswire::kNone, nullptr);

  CHECK(static_cast<int64_t>(last.size()) == p.vocab_size);  // one gathered row
  // the gathered last row equals the last row of the full forward.
  const int64_t V = p.vocab_size;
  float m = 0.0f;
  for (int64_t v = 0; v < V; ++v) m = std::max(m, std::fabs(last[v] - all[4 * V + v]));
  CHECK(m == doctest::Approx(0.0f));
}

TEST_CASE("deepseek-v4 W7: RED-first — a miswired interleave changes the output") {
  const DeepseekV4Params p = TinyParams();
  const DeepseekV4HostWeights hw = TinyWeights(p);

  const std::vector<float> base =
      DeepseekV4ForwardHost(hw, p, kTokens, kPositions, {}, V4Miswire::kNone, nullptr);
  CHECK(AllFinite(base));

  // (1) route hash layers as learned-gated (ignore tid2eid) -> output MUST change.
  V4ForwardTrace tr_gated;
  const std::vector<float> gated = DeepseekV4ForwardHost(
      hw, p, kTokens, kPositions, {}, V4Miswire::kAllLayersGated, &tr_gated);
  CHECK(AllFinite(gated));
  CHECK(tr_gated.layer_hash_routed == std::vector<int>{0, 0, 0, 0});  // hash bypassed
  CHECK(MaxAbsDiff(base, gated) > 1e-5f);

  // (2) skip the final MhcPost fold before the head collapse -> output MUST change.
  const std::vector<float> no_post = DeepseekV4ForwardHost(
      hw, p, kTokens, kPositions, {}, V4Miswire::kSkipFinalMhcPost, nullptr);
  CHECK(AllFinite(no_post));
  CHECK(MaxAbsDiff(base, no_post) > 1e-5f);

  // (3) drop the per-head attention sink (plain softmax) -> output MUST change.
  const std::vector<float> no_sink = DeepseekV4ForwardHost(
      hw, p, kTokens, kPositions, {}, V4Miswire::kNoAttnSink, nullptr);
  CHECK(AllFinite(no_sink));
  CHECK(MaxAbsDiff(base, no_sink) > 1e-5f);
}
