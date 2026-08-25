// MODEL-DSV4-EXL3 W2 — the EXL3 routed-expert tower is REACHED from a production
// entry point, and computes the function its dequantized equivalent does.
//
// WHAT THIS EXISTS FOR. W1b landed a loader that coalesces a TP4 trellis tower
// into TP1 and then NOTHING consumed it: a forward over an EXL3 load refused
// through a `has_host_weights` guard that did not even name the row. AGENTS.md's
// "Nothing lands dead" says a capability is what a production entry point can
// reach at its own merge commit, and that a unit test constructing the type by
// hand proves the class works, never that anything reaches it. So this drives
// `vllm::DeepseekV4Model::Forward` — the function `deepseek_v4_registry.cpp`
// routes `ModelRegistry::Forward` to — over a `DeepseekV4Weights` carrying a real
// EXL3 tower, and asserts two things a mutation can tell apart:
//
//   1. EQUIVALENCE. The EXL3 arm's logits match a DENSE forward whose expert
//      weights are `vt::Exl3DequantLinear` of the SAME trellis. That is the
//      algebraic identity the format rests on (`exl3.py:183-214` vs `:227-237`):
//      the two Hadamards may ride the activations or the weights.
//   2. DISCRIMINATION. The EXL3 arm's logits are FAR from a dense forward over
//      the fixture's own unrelated random expert weights. The EXL3 weights
//      struct is attached to a host tower whose `exp_w*` are those unrelated
//      weights, so deleting the `has_exl3_weights` dispatch in
//      `DeepseekV4Model::Forward` makes (1) and (2) BOTH fail — which is the
//      reachability mutation, and why the fixture is built this way rather than
//      the convenient way.
//
// The bound in (1) is stated, not tuned. Each EXL3 expert call rounds through
// fp16 on the way in and out (`.agents/specs/model-dsv4-exl3.md` `## W2 design`
// §2: fp16 is upstream's own output dtype), which is ~4.9e-4 relative each; the
// three chained calls of one expert (w1, w3 -> SwiGLU -> w2) give ~1.5e-3, the
// MoE output is a weighted sum over topk of those plus an IDENTICAL shared
// expert, and each layer's RMSNorm renormalizes rather than amplifies. 2.0e-2
// relative RMS is more than ten times that estimate and still two orders below
// what (2) measures, so the two checks cannot both be satisfied by an arm that
// ran the wrong weights.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::DeepseekV4Exl3Expert;
using vllm::DeepseekV4Exl3Linear;
using vllm::DeepseekV4HostWeights;
using vllm::DeepseekV4LayerHostWeights;
using vllm::DeepseekV4Params;
using vllm::DeepseekV4Weights;

namespace {

struct Rng {
  uint32_t s = 0x243F6A88u;
  float next(float scale) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    return u * scale;
  }
  uint16_t bits16() {
    s = s * 1664525u + 1013904223u;
    return static_cast<uint16_t>(s >> 13);
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

// The tiny structural config of tests/vllm/models/test_deepseek_v4_forward.cpp,
// WIDENED to hidden_size = moe_intermediate_size = 128. That is not cosmetic:
// an EXL3 linear was Hadamard-128 transformed on BOTH sides at quantization time
// (`exl3_lib/quantize.py:15`), so 128 is the smallest width the format admits.
DeepseekV4Params TinyParams() {
  DeepseekV4Params p;
  p.hidden_size = 128;
  p.num_hidden_layers = 2;
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
  p.moe_intermediate_size = 128;
  p.n_shared_experts = 1;
  p.norm_topk_prob = true;
  p.routed_scaling_factor = 1.5;
  p.swiglu_limit = 10.0;
  p.scoring_func = "sqrtsoftplus";
  p.num_hash_layers = 1;
  p.expert_dtype = "fp4";
  p.hc_mult = 4;
  p.hc_sinkhorn_iters = 5;
  p.hc_eps = 1e-6;
  p.index_head_dim = 4;
  p.index_n_heads = 2;
  p.index_topk = 3;
  p.compress_ratios = {0, 4};
  return p;
}

DeepseekV4HostWeights TinyHost(const DeepseekV4Params& p) {
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
    L.attn_sink = {0.7f, -0.4f};
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
    // The routed dense experts are UNRELATED to the trellis below. That is the
    // discrimination lever: an EXL3 forward that fell back to the dense arm
    // would compute with these instead.
    L.exp_w1 = Rand(rng, ne * mi * H, 0.3f);
    L.exp_w3 = Rand(rng, ne * mi * H, 0.3f);
    L.exp_w2 = Rand(rng, ne * H * mi, 0.3f);
  }
  return hw;
}

// One synthetic EXL3 linear: a pseudo-random BIT STREAM for the trellis (every
// 16-bit codeword decodes to a valid fp16 pair under the MCG codebook, so a
// random stream is a valid quantized weight) and sign+scale vectors shaped like
// the DeepSeek-V4 artifact's, which carry a real per-channel scale rather than
// bare signs.
DeepseekV4Exl3Linear MakeLinear(Rng& rng, int64_t k, int64_t n, int bits) {
  DeepseekV4Exl3Linear lin;
  lin.in_features = k;
  lin.out_features = n;
  lin.bits = bits;
  lin.mcg = 1;
  lin.trellis.resize(static_cast<size_t>(k / 16 * n / 16 * 16 * bits));
  for (auto& w : lin.trellis) w = rng.bits16();
  lin.suh.resize(static_cast<size_t>(k));
  for (auto& s : lin.suh) s = vt::F32ToF16(rng.next(1.0f) >= 0.0f ? 0.5f : -0.5f);
  lin.svh.resize(static_cast<size_t>(n));
  for (auto& s : lin.svh) s = vt::F32ToF16(rng.next(1.0f) >= 0.0f ? 0.5f : -0.5f);
  return lin;
}

// The dequantized equivalent of `lin`, written into the host tower's row-major
// [out, in] layout. `Exl3DequantLinear` produces [in, out] (k rows, n columns),
// which is the transpose of what `MoeBlock`'s host arm indexes.
void DequantInto(const DeepseekV4Exl3Linear& lin, float* dst) {
  const int64_t k = lin.in_features, n = lin.out_features;
  std::vector<float> w(static_cast<size_t>(k * n));
  vt::Exl3DequantLinear(lin.trellis.data(), lin.suh.data(), lin.svh.data(), k, n, lin.bits,
                        w.data());
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < k; ++i) dst[j * k + i] = w[static_cast<size_t>(i * n + j)];
}

double RelRms(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

bool AllFinite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return !v.empty();
}

struct QueueGuard {
  vt::Backend& b;
  vt::Queue q;
  QueueGuard() : b(vt::GetBackend(vt::DeviceType::kCPU)), q(b.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

const std::vector<int32_t> kTokens = {3, 7, 1};
const std::vector<int32_t> kPositions = {0, 1, 2};

}  // namespace

TEST_CASE("dsv4 exl3 W2: the trellis tower is REACHED from DeepseekV4Model::Forward") {
  const DeepseekV4Params p = TinyParams();
  const int kBits = 3;  // the 3.0bpw artifact's width
  const int64_t H = p.hidden_size, mi = p.moe_intermediate_size;
  const int64_t ne = p.n_routed_experts;

  DeepseekV4Weights w;
  w.params = p;
  w.host = TinyHost(p);
  w.has_host_weights = true;

  // The EXL3 routed-expert tower, and the DENSE tower that is its dequantized
  // equivalent. Both come from the SAME trellis bits.
  Rng trng;
  trng.s = 0x51ED270Bu;
  DeepseekV4HostWeights deq = w.host;
  w.exl3.tp = 1;
  w.exl3.bits = kBits;
  w.exl3.codebook = "mcg";
  w.exl3.version = "rank-sliced-deepseek-v4-v1";
  w.exl3.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    auto& layer = w.exl3.layers[static_cast<size_t>(l)];
    layer.experts.resize(static_cast<size_t>(ne));
    for (int64_t e = 0; e < ne; ++e) {
      DeepseekV4Exl3Expert& xe = layer.experts[static_cast<size_t>(e)];
      xe.w1 = MakeLinear(trng, H, mi, kBits);   // gate: [H] -> [mi]
      xe.w3 = MakeLinear(trng, H, mi, kBits);   // up:   [H] -> [mi]
      xe.w2 = MakeLinear(trng, mi, H, kBits);   // down: [mi] -> [H]
      DeepseekV4LayerHostWeights& DL = deq.layers[static_cast<size_t>(l)];
      DequantInto(xe.w1, &DL.exp_w1[static_cast<size_t>(e * mi * H)]);
      DequantInto(xe.w3, &DL.exp_w3[static_cast<size_t>(e * mi * H)]);
      DequantInto(xe.w2, &DL.exp_w2[static_cast<size_t>(e * H * mi)]);
    }
  }
  w.has_exl3_weights = true;

  QueueGuard g;
  const vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;

  // (a) the EXL3 arm, through the production entry point.
  const std::vector<float> exl3_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, w, g.q, {});
  REQUIRE(AllFinite(exl3_logits));
  CHECK(static_cast<int64_t>(exl3_logits.size()) ==
        static_cast<int64_t>(kTokens.size()) * p.vocab_size);

  // (b) the DEQUANTIZED-weight dense arm: same weights, other basis.
  DeepseekV4Weights wd;
  wd.params = p;
  wd.host = deq;
  wd.has_host_weights = true;
  const std::vector<float> deq_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wd, g.q, {});
  REQUIRE(AllFinite(deq_logits));

  // (c) the UNRELATED dense arm: the fixture's own random expert weights, which
  //     are what a forward that missed the EXL3 dispatch would use.
  DeepseekV4Weights wr;
  wr.params = p;
  wr.host = w.host;
  wr.has_host_weights = true;
  const std::vector<float> rand_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wr, g.q, {});
  REQUIRE(AllFinite(rand_logits));

  const double equiv = RelRms(exl3_logits, deq_logits);
  const double discrim = RelRms(exl3_logits, rand_logits);
  MESSAGE("exl3 vs dequantized-dense rel_rms=", equiv,
          "   exl3 vs unrelated-dense rel_rms=", discrim);
  // (1) EQUIVALENCE, at the bound this file's header derives.
  CHECK(equiv <= 2.0e-2);
  // (2) DISCRIMINATION: two orders above it. Deleting the `has_exl3_weights`
  //     dispatch in DeepseekV4Model::Forward makes exl3_logits == rand_logits,
  //     so this goes to 0 and (1) blows up at the same time.
  CHECK(discrim > 1.0e-1);
}

TEST_CASE("dsv4 exl3 W2: a forward with no non-expert tower refuses BY NAME") {
  // The trellis tower loads long before the `carried-*` FP8 tensors are
  // materialized (MODEL-DSV4-EXL3 W1c owns those), and until then the refusal
  // must name THIS row rather than the generic host-tower message, which is what
  // the row's `## Owed` recorded as missing.
  DeepseekV4Weights w;
  w.params = TinyParams();
  w.has_exl3_weights = true;
  w.has_host_weights = false;
  w.exl3.bits = 3;

  QueueGuard g;
  const vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;
  std::string msg;
  try {
    (void)vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, w, g.q, {});
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  CHECK(msg.find("MODEL-DSV4-EXL3") != std::string::npos);
  CHECK(msg.find("W1c") != std::string::npos);
  CHECK(msg.find("EXL3") != std::string::npos);
}
