// GLM-5.3-Flash W5b-2 gate — the decoder layer, the mHC stream threading, the
// assembled `Glm5NextTextModel::Forward` and the KV binding (#2241, row
// MODEL-MM-glm5-next-glm5-next-for-conditional-generation,
// `.agents/specs/glm5-next-flash.md` §W5b-2).
//
// THE ORACLE IS RUN, NOT TRANSCRIBED. Every golden in
// `fixtures/glm5_next_layer_goldens.inc` is the return value of an unmodified
// `transformers` v5.16.1 module — `Glm5NextTextDecoderLayer.forward`
// (`modeling_glm5_next.py:1279-1329`), the `Glm5NextTextModel` layer loop
// (`:1477-1493`) and its `DynamicCache` continuation — captured by
// `fixtures/gen_glm5_next_layer_goldens.py`, which asserts the module file's
// sha256 `2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b`
// before it runs. vLLM registers no `glm5_next` at any revision, so under
// AGENTS.md "When vLLM has no implementation" transformers is the reference for
// this surface; W0 (#2096) recorded the lane revision.
//
// EVERY GOLDEN THE FIXTURE EMITS IS CONSUMED BY AN ASSERTION BELOW, and this
// file says so because it already cost this row: W3's review found a captured
// `kIndexScores` golden that nothing read, which let two real scale defects pass
// 1602 of 1602 assertions. The weight arrays are INPUTS, the per-layer streams
// and the final hidden state are OUTPUTS, and the three scalar margins
// (`kStreamSeparation`, `kEarlyCollapseGap`, `kCachedVsOneShotGap`) are each
// read by a case that would be vacuous without them.
//
// ─── WHAT THIS FILE IS FOR ───────────────────────────────────────────────────
//
// The `[T, hc_mult, hidden]` manifold. A port that threads `[T, hidden]` and
// collapses the residual streams early RUNS: finite activations, the right
// shapes, fluent tokens, and every sublayer gate on this row still green,
// because the collapsed stream is exactly what the sublayers consume. There is
// no end-to-end token gate for this model on this fleet (spec §Gates) that would
// catch it downstream either. So it is caught HERE, three ways that do not
// overlap:
//
//   * the per-layer `[B, S, 4, H]` streams are asserted ELEMENTWISE, so a port
//     with one stream cannot produce them;
//   * `kStreamSeparation` is the oracle's own minimum pairwise distance between
//     the four streams — 6.47 on this fixture — so the elementwise assertion is
//     shown to be DISCRIMINATING and not four copies of one value;
//   * `kEarlyCollapseFinal` is the DECOY: the same oracle modules with the
//     manifold collapsed to its mean and re-broadcast after every layer, which
//     is precisely what the wrong port computes. The gate asserts ours matches
//     the real final state and DIFFERS from the decoy by the oracle's own
//     measured gap.
#include "vllm/model_executor/models/glm5_next_layer.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

#include "glm5_next_layer_goldens.inc"

namespace g = glm5_next_layer_goldens;
namespace gn = vllm::glm5_next;

namespace {

// Our f32 host reductions against the oracle's f32 torch reductions. Only the
// summation ORDER differs, and the stack is five layers deep, so the tolerance
// covers accumulated rounding over that depth and nothing else. It is three
// orders of magnitude below `kStreamSeparation` (6.47) and below
// `kEarlyCollapseGap` (2.40), which is what stops it from being a tolerance
// wide enough to admit the defects this file exists to catch.
constexpr float kLayerTol = 2e-4f;
// The assembled five-layer forward accumulates further through the final
// collapse and norm.
constexpr float kModelTol = 5e-4f;

vt::Queue CpuQueue() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

template <size_t N>
std::vector<float> Vec(const float (&a)[N]) {
  return std::vector<float>(a, a + N);
}

template <size_t N>
std::vector<uint8_t> VecU8(const uint8_t (&a)[N]) {
  return std::vector<uint8_t>(a, a + N);
}

// `Glm5NextParams` for the fixture, built field by field and then CHECKED
// against the schedule the reference's own `__post_init__` resolved. The check
// is the point: `layer_types` is rewritten in there (`full_attention` becomes
// `deepseek_sparse_attention`), so a hand-built schedule that agreed with this
// generator's constants and disagreed with the resolved config would gate the
// generator instead of the model.
vllm::Glm5NextParams FixtureParams() {
  vllm::Glm5NextParams p;
  p.hidden_size = g::kHidden;
  p.num_hidden_layers = g::kNumLayers;
  p.intermediate_size = g::kIntermediate;
  p.vocab_size = g::kVocab;
  p.num_attention_heads = g::kNHeads;
  p.num_key_value_heads = g::kNHeads;
  p.rms_norm_eps = g::kRmsNormEps;
  p.swiglu_limit = g::kSwigluLimit;

  for (int64_t i = 0; i < g::kNumLayers; ++i) {
    const std::string lt = g::kLayerTypes[i];
    p.layer_types.push_back(lt == "linear_attention"
                                ? vllm::Glm5NextLayerKind::kLinearAttention
                                : vllm::Glm5NextLayerKind::kDeepseekSparseAttention);
    p.mlp_layer_types.push_back(std::string(g::kMlpTypes[i]) == "dense"
                                    ? vllm::Glm5NextMlpKind::kDense
                                    : vllm::Glm5NextMlpKind::kSparse);
    p.indexer_types.push_back(std::string(g::kIndexerTypes[i]) == "full"
                                  ? vllm::Glm5NextIndexerKind::kFull
                                  : vllm::Glm5NextIndexerKind::kShared);
  }

  p.mla.q_lora_rank = g::kQLora;
  p.mla.kv_lora_rank = g::kKvLora;
  p.mla.qk_nope_head_dim = g::kQkNope;
  p.mla.qk_rope_head_dim = 0;
  p.mla.v_head_dim = g::kVHead;
  p.mla.qk_head_dim = g::kQkNope;

  p.indexer.head_dim = g::kIdxHeadDim;
  p.indexer.n_heads = g::kIdxNHeads;
  p.indexer.topk = g::kIndexTopk;
  p.indexer.kpool = g::kIndexKpool;
  p.indexer.kpool_always_select_tail = true;

  p.kda.num_heads = g::kLinHeads;
  p.kda.head_dim = g::kLinHeadDim;
  p.kda.conv_kernel_dim = g::kLinConvK;
  p.kda.lower_bound = g::kGateLowerBound;

  p.mhc.mult = g::kHcMult;
  p.mhc.sinkhorn_iters = g::kHcSinkhornIters;
  p.mhc.eps = g::kHcEps;

  p.moe.n_routed_experts = g::kNRouted;
  p.moe.n_shared_experts = g::kNShared;
  p.moe.num_experts_per_tok = g::kTopK;
  p.moe.moe_intermediate_size = g::kMoeIntermediate;
  p.moe.n_group = 1;
  p.moe.topk_group = 1;
  p.moe.routed_scaling_factor = g::kRoutedScaling;
  p.moe.norm_topk_prob = true;
  return p;
}

// The weight tower, five layers, straight out of the goldens. Written out in
// full rather than looped because the four arms carry different tensors and a
// loop over them would need the same switch anyway — and because a reader
// checking layer 2 against `kL2*` can do it by eye.
gn::DecoderLayerWeights L0() {
  gn::DecoderLayerWeights w;
  w.attn_kind = vllm::Glm5NextLayerKind::kLinearAttention;
  w.mlp_kind = vllm::Glm5NextMlpKind::kDense;
  w.input_layernorm = Vec(g::kL0InputNorm);
  w.post_attention_layernorm = Vec(g::kL0PostAttnNorm);
  w.attn_hc = {Vec(g::kL0AttnHcFn), Vec(g::kL0AttnHcBase), Vec(g::kL0AttnHcScale)};
  w.ffn_hc = {Vec(g::kL0FfnHcFn), Vec(g::kL0FfnHcBase), Vec(g::kL0FfnHcScale)};
  w.kda.q_proj = Vec(g::kL0KdaQProj);
  w.kda.k_proj = Vec(g::kL0KdaKProj);
  w.kda.v_proj = Vec(g::kL0KdaVProj);
  w.kda.q_conv1d = Vec(g::kL0KdaQConv);
  w.kda.k_conv1d = Vec(g::kL0KdaKConv);
  w.kda.v_conv1d = Vec(g::kL0KdaVConv);
  w.kda.f_a_proj = Vec(g::kL0KdaFAProj);
  w.kda.f_b_proj = Vec(g::kL0KdaFBProj);
  w.kda.dt_bias = Vec(g::kL0KdaDtBias);
  w.kda.a_log = Vec(g::kL0KdaALog);
  w.kda.b_proj = Vec(g::kL0KdaBProj);
  w.kda.g_a_proj = Vec(g::kL0KdaGAProj);
  w.kda.g_b_proj = Vec(g::kL0KdaGBProj);
  w.kda.o_norm = Vec(g::kL0KdaONorm);
  w.kda.o_proj = Vec(g::kL0KdaOProj);
  w.dense_mlp = {Vec(g::kL0MlpGate), Vec(g::kL0MlpUp), Vec(g::kL0MlpDown)};
  return w;
}

gn::DecoderLayerWeights L1() {
  gn::DecoderLayerWeights w;
  w.attn_kind = vllm::Glm5NextLayerKind::kDeepseekSparseAttention;
  w.mlp_kind = vllm::Glm5NextMlpKind::kDense;
  w.input_layernorm = Vec(g::kL1InputNorm);
  w.post_attention_layernorm = Vec(g::kL1PostAttnNorm);
  w.attn_hc = {Vec(g::kL1AttnHcFn), Vec(g::kL1AttnHcBase), Vec(g::kL1AttnHcScale)};
  w.ffn_hc = {Vec(g::kL1FfnHcFn), Vec(g::kL1FfnHcBase), Vec(g::kL1FfnHcScale)};
  w.dsa.mla.q_a_proj = Vec(g::kL1MlaQAProj);
  w.dsa.mla.q_a_layernorm = Vec(g::kL1MlaQANorm);
  w.dsa.mla.q_b_proj = Vec(g::kL1MlaQBProj);
  w.dsa.mla.kv_a_proj_with_mqa = Vec(g::kL1MlaKvAProj);
  w.dsa.mla.kv_a_layernorm = Vec(g::kL1MlaKvANorm);
  w.dsa.mla.k_b_proj = Vec(g::kL1MlaKB);
  w.dsa.mla.v_b_proj = Vec(g::kL1MlaVB);
  w.dsa.mla.o_proj = Vec(g::kL1MlaOProj);
  w.dsa.idx_wq_b = Vec(g::kL1IdxWqB);
  w.dsa.idx_wk = Vec(g::kL1IdxWk);
  w.dsa.idx_k_norm_weight = Vec(g::kL1IdxKNormWeight);
  w.dsa.idx_k_norm_bias = Vec(g::kL1IdxKNormBias);
  w.dsa.idx_weights_proj = Vec(g::kL1IdxWeightsProj);
  w.dsa.idx_kpool_ape = Vec(g::kL1IdxKpoolApe);
  w.dsa.idx_kpool_gate = Vec(g::kL1IdxKpoolGate);
  w.dense_mlp = {Vec(g::kL1MlpGate), Vec(g::kL1MlpUp), Vec(g::kL1MlpDown)};
  return w;
}

gn::MoeLayerWeights MoeL2() {
  gn::MoeLayerWeights m;
  m.router_weight = Vec(g::kL2MoeRouter);
  m.e_score_correction_bias = Vec(g::kL2MoeRouterBias);
  m.expert_gate_up = Vec(g::kL2MoeExpertGateUp);
  m.expert_down = Vec(g::kL2MoeExpertDown);
  m.shared = {Vec(g::kL2MoeSharedGate), Vec(g::kL2MoeSharedUp),
              Vec(g::kL2MoeSharedDown)};
  return m;
}

gn::DecoderLayerWeights L2() {
  gn::DecoderLayerWeights w;
  w.attn_kind = vllm::Glm5NextLayerKind::kDeepseekSparseAttention;
  w.mlp_kind = vllm::Glm5NextMlpKind::kSparse;
  w.input_layernorm = Vec(g::kL2InputNorm);
  w.post_attention_layernorm = Vec(g::kL2PostAttnNorm);
  w.attn_hc = {Vec(g::kL2AttnHcFn), Vec(g::kL2AttnHcBase), Vec(g::kL2AttnHcScale)};
  w.ffn_hc = {Vec(g::kL2FfnHcFn), Vec(g::kL2FfnHcBase), Vec(g::kL2FfnHcScale)};
  w.dsa.mla.q_a_proj = Vec(g::kL2MlaQAProj);
  w.dsa.mla.q_a_layernorm = Vec(g::kL2MlaQANorm);
  w.dsa.mla.q_b_proj = Vec(g::kL2MlaQBProj);
  w.dsa.mla.kv_a_proj_with_mqa = Vec(g::kL2MlaKvAProj);
  w.dsa.mla.kv_a_layernorm = Vec(g::kL2MlaKvANorm);
  w.dsa.mla.k_b_proj = Vec(g::kL2MlaKB);
  w.dsa.mla.v_b_proj = Vec(g::kL2MlaVB);
  w.dsa.mla.o_proj = Vec(g::kL2MlaOProj);
  // NO indexer storage: layer 2 is `shared`, and the reference builds no
  // `Glm5NextTextIndexer` for it at all (`:1131`). The generator emits no
  // `kL2Idx*` array for exactly that reason.
  w.moe = MoeL2();
  return w;
}

gn::DecoderLayerWeights L3() {
  gn::DecoderLayerWeights w;
  w.attn_kind = vllm::Glm5NextLayerKind::kLinearAttention;
  w.mlp_kind = vllm::Glm5NextMlpKind::kSparse;
  w.input_layernorm = Vec(g::kL3InputNorm);
  w.post_attention_layernorm = Vec(g::kL3PostAttnNorm);
  w.attn_hc = {Vec(g::kL3AttnHcFn), Vec(g::kL3AttnHcBase), Vec(g::kL3AttnHcScale)};
  w.ffn_hc = {Vec(g::kL3FfnHcFn), Vec(g::kL3FfnHcBase), Vec(g::kL3FfnHcScale)};
  w.kda.q_proj = Vec(g::kL3KdaQProj);
  w.kda.k_proj = Vec(g::kL3KdaKProj);
  w.kda.v_proj = Vec(g::kL3KdaVProj);
  w.kda.q_conv1d = Vec(g::kL3KdaQConv);
  w.kda.k_conv1d = Vec(g::kL3KdaKConv);
  w.kda.v_conv1d = Vec(g::kL3KdaVConv);
  w.kda.f_a_proj = Vec(g::kL3KdaFAProj);
  w.kda.f_b_proj = Vec(g::kL3KdaFBProj);
  w.kda.dt_bias = Vec(g::kL3KdaDtBias);
  w.kda.a_log = Vec(g::kL3KdaALog);
  w.kda.b_proj = Vec(g::kL3KdaBProj);
  w.kda.g_a_proj = Vec(g::kL3KdaGAProj);
  w.kda.g_b_proj = Vec(g::kL3KdaGBProj);
  w.kda.o_norm = Vec(g::kL3KdaONorm);
  w.kda.o_proj = Vec(g::kL3KdaOProj);
  w.moe.router_weight = Vec(g::kL3MoeRouter);
  w.moe.e_score_correction_bias = Vec(g::kL3MoeRouterBias);
  w.moe.expert_gate_up = Vec(g::kL3MoeExpertGateUp);
  w.moe.expert_down = Vec(g::kL3MoeExpertDown);
  w.moe.shared = {Vec(g::kL3MoeSharedGate), Vec(g::kL3MoeSharedUp),
                  Vec(g::kL3MoeSharedDown)};
  return w;
}

gn::DecoderLayerWeights L4() {
  gn::DecoderLayerWeights w;
  w.attn_kind = vllm::Glm5NextLayerKind::kDeepseekSparseAttention;
  w.mlp_kind = vllm::Glm5NextMlpKind::kSparse;
  w.input_layernorm = Vec(g::kL4InputNorm);
  w.post_attention_layernorm = Vec(g::kL4PostAttnNorm);
  w.attn_hc = {Vec(g::kL4AttnHcFn), Vec(g::kL4AttnHcBase), Vec(g::kL4AttnHcScale)};
  w.ffn_hc = {Vec(g::kL4FfnHcFn), Vec(g::kL4FfnHcBase), Vec(g::kL4FfnHcScale)};
  w.dsa.mla.q_a_proj = Vec(g::kL4MlaQAProj);
  w.dsa.mla.q_a_layernorm = Vec(g::kL4MlaQANorm);
  w.dsa.mla.q_b_proj = Vec(g::kL4MlaQBProj);
  w.dsa.mla.kv_a_proj_with_mqa = Vec(g::kL4MlaKvAProj);
  w.dsa.mla.kv_a_layernorm = Vec(g::kL4MlaKvANorm);
  w.dsa.mla.k_b_proj = Vec(g::kL4MlaKB);
  w.dsa.mla.v_b_proj = Vec(g::kL4MlaVB);
  w.dsa.mla.o_proj = Vec(g::kL4MlaOProj);
  w.dsa.idx_wq_b = Vec(g::kL4IdxWqB);
  w.dsa.idx_wk = Vec(g::kL4IdxWk);
  w.dsa.idx_k_norm_weight = Vec(g::kL4IdxKNormWeight);
  w.dsa.idx_k_norm_bias = Vec(g::kL4IdxKNormBias);
  w.dsa.idx_weights_proj = Vec(g::kL4IdxWeightsProj);
  w.dsa.idx_kpool_ape = Vec(g::kL4IdxKpoolApe);
  w.dsa.idx_kpool_gate = Vec(g::kL4IdxKpoolGate);
  w.moe.router_weight = Vec(g::kL4MoeRouter);
  w.moe.e_score_correction_bias = Vec(g::kL4MoeRouterBias);
  w.moe.expert_gate_up = Vec(g::kL4MoeExpertGateUp);
  w.moe.expert_down = Vec(g::kL4MoeExpertDown);
  w.moe.shared = {Vec(g::kL4MoeSharedGate), Vec(g::kL4MoeSharedUp),
                  Vec(g::kL4MoeSharedDown)};
  return w;
}

gn::TextModelWeights FixtureModel() {
  gn::TextModelWeights m;
  m.params = FixtureParams();
  m.layers = {L0(), L1(), L2(), L3(), L4()};
  m.norm = Vec(g::kFinalNorm);
  return m;
}

// The maximum absolute difference, and the index it happened at, so a failure
// says WHERE.
//
// **A NON-FINITE value is an INFINITE gap, and this is load-bearing rather than
// defensive.** The first version of this helper was the obvious
// `if (d > gp.max_abs)`, and a mutation that truncated the attention's key range
// under a filled cache SURVIVED the whole suite with `max_abs == 0` — because
// its output was all NaN, `NaN - want` is NaN, and `NaN > x` is FALSE for every
// x, so the maximum never moved off its initial zero. An all-NaN forward
// therefore read as a PERFECT match on every gap assertion in this file. The
// mutation was fine; the instrument was blind, which is the failure class
// `.agents/verification.md` calls a broken instrument failing toward a code
// verdict. `non_finite` is reported separately from the gap so a reader can tell
// "wrong number" from "not a number".
struct Gap {
  float max_abs = 0.0F;
  size_t at = 0;
  size_t non_finite = 0;
};

Gap MaxGap(const std::vector<float>& got, const float* want, size_t n) {
  Gap gp;
  for (size_t i = 0; i < n && i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
      if (gp.non_finite == 0) gp.at = i;
      ++gp.non_finite;
      gp.max_abs = std::numeric_limits<float>::infinity();
      continue;
    }
    const float d = std::fabs(got[i] - want[i]);
    if (d > gp.max_abs) {
      gp.max_abs = d;
      gp.at = i;
    }
  }
  return gp;
}

// The minimum pairwise distance between the `hc_mult` streams of a
// `[B, S, hc, H]` manifold. Zero for a port that broadcasts one stream.
// NaN-blind for the same reason `MaxGap` was: `std::max(m, NaN)` returns `m`
// and `std::min(worst, NaN)` returns `worst`, so an all-NaN manifold would score
// `+inf` here and sail past `sep > 1.0F`. A non-finite element makes the
// separation ZERO, which is the value the assertions treat as a failure.
float MinStreamSeparation(const std::vector<float>& streams, int64_t tokens,
                          int64_t hc, int64_t hidden) {
  for (float v : streams) {
    if (!std::isfinite(v)) return 0.0F;
  }
  float worst = std::numeric_limits<float>::infinity();
  for (int64_t a = 0; a < hc; ++a) {
    for (int64_t b = a + 1; b < hc; ++b) {
      float m = 0.0F;
      for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < hidden; ++h) {
          m = std::max(m, std::fabs(streams[(t * hc + a) * hidden + h] -
                                    streams[(t * hc + b) * hidden + h]));
        }
      }
      worst = std::min(worst, m);
    }
  }
  return worst;
}

// A window of the fixture's `[B, S, *]` inputs, so the cached case can drive
// [0, PREFILL) and then [PREFILL, S).
std::vector<float> EmbedWindow(int64_t from, int64_t to) {
  std::vector<float> out;
  for (int64_t b = 0; b < g::kBatch; ++b) {
    for (int64_t s = from; s < to; ++s) {
      const float* src = g::kInputsEmbeds + (b * g::kSeqLen + s) * g::kHidden;
      out.insert(out.end(), src, src + g::kHidden);
    }
  }
  return out;
}

std::vector<uint8_t> MaskWindow(int64_t from, int64_t to) {
  std::vector<uint8_t> out;
  for (int64_t b = 0; b < g::kBatch; ++b) {
    for (int64_t s = from; s < to; ++s) {
      out.push_back(g::kMask[b * g::kSeqLen + s]);
    }
  }
  return out;
}

}  // namespace

// ─── the schedule the reference RESOLVED ─────────────────────────────────────
//
// Not decoration. `__post_init__` rewrites every `full_attention` entry into
// `deepseek_sparse_attention`, so a port that took the config string at face
// value would wire a dense attention block on a layer that runs the DSA
// indexer. The generator emits the strings the config object carried AFTER that
// rewrite, and these cases assert our enums round-trip to them.
TEST_CASE("glm5_next W5b-2: the fixture schedule is upstream's resolved one") {
  const vllm::Glm5NextParams p = FixtureParams();
  REQUIRE(static_cast<int64_t>(p.layer_types.size()) == g::kNumLayers);
  for (int64_t i = 0; i < g::kNumLayers; ++i) {
    CHECK(std::string(vllm::Glm5NextLayerKindName(p.layer_types[i])) ==
          std::string(g::kLayerTypes[i]));
    CHECK(std::string(vllm::Glm5NextMlpKindName(p.mlp_layer_types[i])) ==
          std::string(g::kMlpTypes[i]));
    CHECK(std::string(vllm::Glm5NextIndexerKindName(p.indexer_types[i])) ==
          std::string(g::kIndexerTypes[i]));
  }
  // All four control-flow combinations `:1261-1272` selects between are present,
  // plus the `shared` indexer arm. A schedule missing one of them would make
  // every case below gate the arm that ran and nothing about the selection.
  bool kda_dense = false, dsa_dense = false, dsa_sparse = false, kda_sparse = false;
  bool has_shared = false;
  for (int64_t i = 0; i < g::kNumLayers; ++i) {
    const bool kda = p.layer_types[i] == vllm::Glm5NextLayerKind::kLinearAttention;
    const bool dense = p.mlp_layer_types[i] == vllm::Glm5NextMlpKind::kDense;
    kda_dense |= kda && dense;
    kda_sparse |= kda && !dense;
    dsa_dense |= !kda && dense;
    dsa_sparse |= !kda && !dense;
    has_shared |= p.indexer_types[i] == vllm::Glm5NextIndexerKind::kShared;
  }
  CHECK(kda_dense);
  CHECK(kda_sparse);
  CHECK(dsa_dense);
  CHECK(dsa_sparse);
  CHECK(has_shared);
  // 4 * 16 on this fixture, 4 * 4096 = 16384 on the published checkpoint. The
  // manifold is this wide through the WHOLE stack.
  CHECK(p.residual_stream_width() == g::kHcMult * g::kHidden);
}

// ─── the manifold expansion ──────────────────────────────────────────────────
TEST_CASE("glm5_next W5b-2: the embedding expands into hc_mult identical streams") {
  const std::vector<float> streams = gn::ExpandToHiddenStreams(
      Vec(g::kInputsEmbeds), g::kBatch, g::kSeqLen, g::kHcMult, g::kHidden);
  REQUIRE(static_cast<int64_t>(streams.size()) ==
          g::kBatch * g::kSeqLen * g::kHcMult * g::kHidden);
  // `:1477` is an `expand`, so every stream starts as a COPY. They diverge from
  // the first layer's `post`/`comb` fold, which is what the next case measures.
  for (int64_t t = 0; t < g::kBatch * g::kSeqLen; ++t) {
    for (int64_t m = 0; m < g::kHcMult; ++m) {
      for (int64_t h = 0; h < g::kHidden; ++h) {
        REQUIRE(streams[(t * g::kHcMult + m) * g::kHidden + h] ==
                g::kInputsEmbeds[t * g::kHidden + h]);
      }
    }
  }
  CHECK(MinStreamSeparation(streams, g::kBatch * g::kSeqLen, g::kHcMult,
                            g::kHidden) == 0.0F);
}

// ─── the layer loop, every arm, against the oracle ───────────────────────────
TEST_CASE("glm5_next W5b-2: each decoder layer reproduces the reference manifold") {
  const gn::TextModelWeights model = FixtureModel();
  vt::Queue q = CpuQueue();
  const std::vector<uint8_t> mask = VecU8(g::kMask);
  std::vector<float> streams = gn::ExpandToHiddenStreams(
      Vec(g::kInputsEmbeds), g::kBatch, g::kSeqLen, g::kHcMult, g::kHidden);
  std::vector<int32_t> topk;
  int64_t topk_width = 0;

  const float* const layer_goldens[] = {g::kL0Streams, g::kL1Streams,
                                        g::kL2Streams, g::kL3Streams,
                                        g::kL4Streams};
  const bool propagates[] = {g::kL0Propagates, g::kL1Propagates, g::kL2Propagates,
                             g::kL3Propagates, g::kL4Propagates};

  for (int64_t i = 0; i < g::kNumLayers; ++i) {
    CAPTURE(i);
    gn::DecoderLayerResult r = gn::DecoderLayerForward(
        model.params, i, model.layers[static_cast<size_t>(i)], streams, mask,
        topk.empty() ? nullptr : &topk, topk_width, g::kBatch, g::kSeqLen,
        /*cache=*/nullptr, q);
    const size_t n = static_cast<size_t>(g::kBatch * g::kSeqLen * g::kHcMult *
                                         g::kHidden);
    REQUIRE(r.hidden_streams.size() == n);
    const Gap gp = MaxGap(r.hidden_streams, layer_goldens[i], n);
    INFO("layer ", i, " max |ours - oracle| = ", gp.max_abs, " at ", gp.at,
         ", non-finite values: ", gp.non_finite);
    CHECK(gp.non_finite == 0);
    CHECK(gp.max_abs < kLayerTol);

    // `topk_indices if self.next_skip_topk else None` (`:1216`), and `None` for
    // every KDA layer (`:1297`). Only layer 1 propagates on this schedule,
    // because layer 2 is the `shared` one.
    CHECK(r.topk_indices.empty() == !propagates[i]);
    streams = std::move(r.hidden_streams);
    topk = std::move(r.topk_indices);
    topk_width = topk.empty() ? 0 : r.topk_width;
  }

  // THE MANIFOLD, measured rather than asserted from prose. The oracle's own
  // minimum pairwise stream distance is 6.47 on this fixture; a port that
  // broadcast one stream into four would score zero, and the elementwise
  // assertions above would then be four copies of one value.
  const float sep = MinStreamSeparation(streams, g::kBatch * g::kSeqLen,
                                        g::kHcMult, g::kHidden);
  INFO("min pairwise stream separation: ours ", sep, ", oracle ",
       g::kStreamSeparation);
  CHECK(sep > 1.0F);
  CHECK(std::fabs(sep - g::kStreamSeparation) < kLayerTol);
}

// ─── the assembled forward, and the EARLY-COLLAPSE decoy ─────────────────────
TEST_CASE("glm5_next W5b-2: the assembled forward matches the oracle and NOT the "
          "early-collapse port") {
  const gn::TextModelWeights model = FixtureModel();
  vt::Queue q = CpuQueue();
  const std::vector<float> out = gn::TextModelForward(
      model, Vec(g::kInputsEmbeds), VecU8(g::kMask), g::kBatch, g::kSeqLen,
      /*caches=*/nullptr, q);
  const size_t n = static_cast<size_t>(g::kBatch * g::kSeqLen * g::kHidden);
  REQUIRE(out.size() == n);

  const Gap right = MaxGap(out, g::kFinalHidden, n);
  INFO("vs the reference final hidden state: ", right.max_abs, " at ", right.at,
       ", non-finite values: ", right.non_finite);
  CHECK(right.non_finite == 0);
  CHECK(right.max_abs < kModelTol);

  // THE DECOY. `kEarlyCollapseFinal` came out of the SAME oracle modules with
  // the manifold collapsed to its mean and re-broadcast after every layer —
  // exactly what a port that threads `[T, hidden]` computes. It is finite, it is
  // the right shape, and it is wrong. A gate that only checked "finite and
  // plausible" would pass it.
  const Gap decoy = MaxGap(out, g::kEarlyCollapseFinal, n);
  INFO("vs the early-collapse decoy: ", decoy.max_abs,
       " (the oracle's own gap between the two is ", g::kEarlyCollapseGap, ")");
  CHECK(decoy.max_abs > 0.5F * g::kEarlyCollapseGap);
  // ...and the decoy is a real alternative rather than noise: the two oracle
  // outputs are 2.40 apart, four orders of magnitude above the tolerance the
  // first assertion uses.
  CHECK(g::kEarlyCollapseGap > 1000.0F * kModelTol);
}

// ─── the MTP block is NOT a layer ────────────────────────────────────────────
TEST_CASE("glm5_next W5b-2: a tower with one layer too many is refused by name") {
  gn::TextModelWeights model = FixtureModel();
  vt::Queue q = CpuQueue();
  // `blk.45` on the published artifact is the multi-token-prediction head and
  // the reference DISCARDS it (`_keys_to_ignore_on_load_unexpected`). Building
  // it as a 46th decoder layer produces a fluent wrong model that no gate on
  // this fleet could detect, so the count is CHECKED.
  model.layers.push_back(L4());
  bool threw = false;
  try {
    gn::TextModelForward(model, Vec(g::kInputsEmbeds), VecU8(g::kMask), g::kBatch,
                         g::kSeqLen, nullptr, q);
  } catch (const std::exception& e) {
    threw = true;
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("6 decoder layers") != std::string::npos);
    CHECK(msg.find("multi-token-prediction") != std::string::npos);
  }
  CHECK(threw);

  // ...and one too FEW is refused too, because "45 built from 0..44" and "45
  // built from 1..45" have the same count and only the second is caught by the
  // loader's own `mtp_block_tensors_dropped`.
  model.layers.pop_back();
  model.layers.pop_back();
  threw = false;
  try {
    gn::TextModelForward(model, Vec(g::kInputsEmbeds), VecU8(g::kMask), g::kBatch,
                         g::kSeqLen, nullptr, q);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

// ─── a WRONG schedule builds a DIFFERENT model ───────────────────────────────
TEST_CASE("glm5_next W5b-2: the schedule decides the arm, and a disagreement is "
          "refused") {
  gn::TextModelWeights model = FixtureModel();
  vt::Queue q = CpuQueue();

  SUBCASE("attention arm") {
    // Layer 1 is `deepseek_sparse_attention`. Claiming it is KDA is the failure
    // this refusal exists for: a KDA arm run on a DSA layer produces finite
    // activations of the right shape and fluent text.
    model.params.layer_types[1] = vllm::Glm5NextLayerKind::kLinearAttention;
    bool threw = false;
    try {
      gn::TextModelForward(model, Vec(g::kInputsEmbeds), VecU8(g::kMask),
                           g::kBatch, g::kSeqLen, nullptr, q);
    } catch (const std::exception& e) {
      threw = true;
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("layer_types[1]") != std::string::npos);
      CHECK(msg.find("linear_attention") != std::string::npos);
    }
    CHECK(threw);
  }

  SUBCASE("feed-forward arm") {
    // Layer 2 is `sparse`. Claiming it is dense would run a 288-expert layer's
    // shared expert alone and drop the routing entirely.
    model.params.mlp_layer_types[2] = vllm::Glm5NextMlpKind::kDense;
    bool threw = false;
    try {
      gn::TextModelForward(model, Vec(g::kInputsEmbeds), VecU8(g::kMask),
                           g::kBatch, g::kSeqLen, nullptr, q);
    } catch (const std::exception& e) {
      threw = true;
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("mlp_layer_types[2]") != std::string::npos);
    }
    CHECK(threw);
  }
}

// ─── the `prev_topk_indices` thread ──────────────────────────────────────────
TEST_CASE("glm5_next W5b-2: a shared layer consumes the previous full layer's "
          "selection and refuses without one") {
  const gn::TextModelWeights model = FixtureModel();
  vt::Queue q = CpuQueue();
  const std::vector<uint8_t> mask = VecU8(g::kMask);
  std::vector<float> streams = gn::ExpandToHiddenStreams(
      Vec(g::kInputsEmbeds), g::kBatch, g::kSeqLen, g::kHcMult, g::kHidden);

  // Layers 0 and 1, to get the real selection layer 1 propagates.
  gn::DecoderLayerResult r0 = gn::DecoderLayerForward(
      model.params, 0, model.layers[0], streams, mask, nullptr, 0, g::kBatch,
      g::kSeqLen, nullptr, q);
  REQUIRE(r0.topk_indices.empty());  // a KDA layer WIPES the thread (`:1297`)
  gn::DecoderLayerResult r1 = gn::DecoderLayerForward(
      model.params, 1, model.layers[1], r0.hidden_streams, mask, nullptr, 0,
      g::kBatch, g::kSeqLen, nullptr, q);
  REQUIRE_FALSE(r1.topk_indices.empty());
  CHECK(r1.topk_width == g::kL1TopkWidth);
  REQUIRE(r1.topk_indices.size() ==
          static_cast<size_t>(g::kBatch * g::kSeqLen * g::kL1TopkWidth));
  // The selection ITSELF, index for index, against what the oracle's layer 1
  // returned. Top-k error is BIMODAL, so this is an exact comparison and not a
  // tolerance: a wrong selection carries plausible values.
  size_t differing = 0;
  for (size_t i = 0; i < r1.topk_indices.size(); ++i) {
    if (r1.topk_indices[i] != g::kL1Topk[i]) ++differing;
  }
  INFO("selected indices differing from the oracle: ", differing, " of ",
       r1.topk_indices.size());
  CHECK(differing == 0);

  // ...and layer 2, the `shared` one, REFUSES without it, in upstream's own
  // words (`:1189-1190`).
  bool threw = false;
  try {
    gn::DecoderLayerForward(model.params, 2, model.layers[2], r1.hidden_streams,
                            mask, nullptr, 0, g::kBatch, g::kSeqLen, nullptr, q);
  } catch (const std::exception& e) {
    threw = true;
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("Shared DSA layers require top-k indices from a previous "
                   "full indexer layer.") != std::string::npos);
  }
  CHECK(threw);

  // With it, layer 2 reproduces the oracle — which is the positive half, and
  // without it the refusal above would only prove that null is rejected.
  gn::DecoderLayerResult r2 = gn::DecoderLayerForward(
      model.params, 2, model.layers[2], r1.hidden_streams, mask,
      &r1.topk_indices, r1.topk_width, g::kBatch, g::kSeqLen, nullptr, q);
  const size_t n =
      static_cast<size_t>(g::kBatch * g::kSeqLen * g::kHcMult * g::kHidden);
  const Gap gp = MaxGap(r2.hidden_streams, g::kL2Streams, n);
  INFO("shared layer 2 max |ours - oracle| = ", gp.max_abs,
       ", non-finite values: ", gp.non_finite);
  CHECK(gp.non_finite == 0);
  CHECK(gp.max_abs < kLayerTol);
  // It consumed the selection and did NOT propagate it further (`:1216`: a
  // `shared` layer's `next_skip_topk` is false by construction).
  CHECK(r2.topk_indices.empty());
}

// ═════════════════════════════════════════════════════════════════════════════
//  THE KV BINDING
// ═════════════════════════════════════════════════════════════════════════════

// The widths the cache actually stores against the widths `MakeGlm5NextKVCache`
// PUBLISHES, reached through the production `make_kv_cache` hook on the
// PUBLISHED config. Without this the two are independent opinions: the runner
// would allocate one geometry and the layer would fill another.
TEST_CASE("glm5_next W5b-2: the cache the layer fills is the one the KV spec "
          "publishes") {
  std::ifstream in(std::string(GLM5_NEXT_CKPT_FIXTURE_DIR) + "/config.json");
  REQUIRE(in.good());
  std::stringstream ss;
  ss << in.rdbuf();
  const nlohmann::json doc = nlohmann::json::parse(ss.str());
  // The PUBLISHED `config.json`, byte for byte, so the widths below are the
  // checkpoint's 512 / 257 / [24576, 4] and not a fixture's.
  const vllm::HfConfig cfg = vllm::ParseHfConfig(
      doc, std::string(GLM5_NEXT_CKPT_FIXTURE_DIR) + "/config.json");
  // REACHABILITY. `ModelRegistry::Resolve` is a production entry point -- it is
  // what `LoadedEngine::FromModelDir` calls to pick the factory -- and
  // `make_kv_cache` is the hook the engine invokes on the resolved
  // registration. Nothing here names `MakeGlm5NextKVCache`.
  const std::vector<std::string> archs = {"Glm5NextForConditionalGeneration"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(archs);
  REQUIRE(reg.factory->make_kv_cache != nullptr);
  const vllm::v1::KVCacheConfig kv = reg.factory->make_kv_cache(cfg, 16, 8);
  REQUIRE(kv.kv_cache_groups.size() == 3);

  const vllm::Glm5NextParams p = vllm::ParseGlm5NextParams(cfg);
  const auto* mla = dynamic_cast<const vllm::v1::MLAAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(mla != nullptr);
  // Group 0's row is what `DsaCache::k_pass` stores per token: the LATENT, not
  // the expanded K/V upstream caches. 512 on the published checkpoint.
  CHECK(mla->head_size == p.mla.kv_lora_rank + p.mla.qk_rope_head_dim);

  const auto* mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(mamba != nullptr);
  REQUIRE(mamba->shapes.size() == 2);
  // Group 1's two states are what `Glm5NextKdaCache` holds: ONE grouped
  // `[q; k; v]` conv state `conv_kernel_dim` columns wide, and the f32
  // `[heads, head_dim, head_dim]` recurrent state.
  CHECK(mamba->shapes[0] ==
        std::vector<int64_t>{3 * p.kda.num_heads * p.kda.head_dim,
                             p.kda.conv_kernel_dim});
  CHECK(mamba->shapes[1] ==
        std::vector<int64_t>{p.kda.num_heads, p.kda.head_dim, p.kda.head_dim});

  const auto* idx = dynamic_cast<const vllm::v1::MLAAttentionSpec*>(
      kv.kv_cache_groups[2].kv_cache_spec.get());
  REQUIRE(idx != nullptr);
  // Group 2's row is what `DsaCache::indexer_packed` stores per token:
  // `concat[k, gate_scores, valid]`, 257 on the published checkpoint.
  CHECK(idx->head_size == 2 * p.indexer.head_dim + 1);
}

TEST_CASE("glm5_next W5b-2: a second step reuses the cache instead of "
          "recomputing") {
  const gn::TextModelWeights model = FixtureModel();
  vt::Queue q = CpuQueue();
  std::vector<gn::LayerCache> caches(static_cast<size_t>(g::kNumLayers));

  const std::vector<float> pre_out = gn::TextModelForward(
      model, EmbedWindow(0, g::kPrefill), MaskWindow(0, g::kPrefill), g::kBatch,
      g::kPrefill, &caches, q);
  const size_t pre_n = static_cast<size_t>(g::kBatch * g::kPrefill * g::kHidden);
  REQUIRE(pre_out.size() == pre_n);
  const Gap pre_gap = MaxGap(pre_out, g::kCachedPrefillHidden, pre_n);
  INFO("prefill max |ours - oracle| = ", pre_gap.max_abs,
       ", non-finite values: ", pre_gap.non_finite);
  CHECK(pre_gap.non_finite == 0);
  CHECK(pre_gap.max_abs < kModelTol);

  // THE CACHE IS FILLED, and this is checked BEFORE the continuation, because a
  // write-only cache that the continuation ignores still produces the right
  // answer by recomputing — and would then pass the value assertion below.
  for (int64_t i = 0; i < g::kNumLayers; ++i) {
    CAPTURE(i);
    const gn::LayerCache& c = caches[static_cast<size_t>(i)];
    if (model.params.layer_types[i] == vllm::Glm5NextLayerKind::kLinearAttention) {
      // Group 1: one recurrent state per SEQUENCE, never one shared across the
      // batch — the delta rule carries history and sharing it mixes requests.
      REQUIRE(static_cast<int64_t>(c.kda.size()) == g::kBatch);
      for (int64_t b = 0; b < g::kBatch; ++b) {
        CHECK(static_cast<int64_t>(c.kda[static_cast<size_t>(b)].conv_state.size()) ==
              3 * g::kLinHeads * g::kLinHeadDim * g::kLinConvK);
        CHECK(static_cast<int64_t>(
                  c.kda[static_cast<size_t>(b)].recurrent_state.size()) ==
              g::kLinHeads * g::kLinHeadDim * g::kLinHeadDim);
      }
      CHECK(c.dsa.cached_len == 0);
    } else {
      // Group 0, the MLA latent: `kv_lora_rank` per token per request.
      CHECK(c.dsa.cached_len == g::kPrefill);
      CHECK(static_cast<int64_t>(c.dsa.k_pass.size()) ==
            g::kBatch * g::kPrefill * g::kKvLora);
      if (model.params.indexer_types[i] == vllm::Glm5NextIndexerKind::kFull) {
        // Group 2, the indexer side cache: `2 * index_head_dim + 1` per token.
        CHECK(static_cast<int64_t>(c.dsa.indexer_packed.size()) ==
              g::kBatch * g::kPrefill * (2 * g::kIdxHeadDim + 1));
      } else {
        // A `shared` layer builds no indexer at all (`:1131`), so
        // `update_indexer` is never reached and its side cache stays EMPTY.
        CHECK(c.dsa.indexer_packed.empty());
      }
      CHECK(c.kda.empty());
    }
  }

  // The continuation. Its mask is the TAIL of the batch mask, which is what the
  // reference hands the layers: `create_recurrent_attention_mask` "trims the
  // mask to the trailing `inputs_embeds.shape[1]` positions"
  // (`masking_utils.py:1490-1494` @ v5.16.1), so a generator that passes the
  // whole `[B, 12]` mask and a C++ caller that passes the trailing `[B, 4]` are
  // handing the layer the SAME tensor.
  const std::vector<float> cont_out = gn::TextModelForward(
      model, EmbedWindow(g::kPrefill, g::kSeqLen),
      MaskWindow(g::kPrefill, g::kSeqLen), g::kBatch, g::kCont, &caches, q);
  const size_t cont_n = static_cast<size_t>(g::kBatch * g::kCont * g::kHidden);
  REQUIRE(cont_out.size() == cont_n);
  const Gap cont_gap = MaxGap(cont_out, g::kCachedContHidden, cont_n);
  INFO("continuation max |ours - oracle| = ", cont_gap.max_abs,
       ", non-finite values: ", cont_gap.non_finite);
  CHECK(cont_gap.non_finite == 0);
  CHECK(cont_gap.max_abs < kModelTol);

  // The cache GREW, so the second step attended over 12 keys and not 4.
  for (int64_t i = 0; i < g::kNumLayers; ++i) {
    if (model.params.layer_types[i] == vllm::Glm5NextLayerKind::kLinearAttention) {
      continue;
    }
    CAPTURE(i);
    CHECK(caches[static_cast<size_t>(i)].dsa.cached_len == g::kSeqLen);
    CHECK(static_cast<int64_t>(caches[static_cast<size_t>(i)].dsa.k_pass.size()) ==
          g::kBatch * g::kSeqLen * g::kKvLora);
  }

  // ...and the continuation agrees with the one-shot tail, which is the property
  // a cache exists for. The tolerance is the oracle's OWN measured gap between
  // its cached and uncached runs, widened by our reduction-order slack, so it is
  // a number rather than a hope.
  const std::vector<float> one_shot = gn::TextModelForward(
      model, Vec(g::kInputsEmbeds), VecU8(g::kMask), g::kBatch, g::kSeqLen,
      nullptr, q);
  float tail = 0.0F;
  for (int64_t b = 0; b < g::kBatch; ++b) {
    for (int64_t s = 0; s < g::kCont; ++s) {
      for (int64_t h = 0; h < g::kHidden; ++h) {
        const float a = cont_out[(b * g::kCont + s) * g::kHidden + h];
        const float c =
            one_shot[(b * g::kSeqLen + g::kPrefill + s) * g::kHidden + h];
        // Non-finite is an infinite gap here too; see `MaxGap`.
        tail = (!std::isfinite(a) || !std::isfinite(c))
                   ? std::numeric_limits<float>::infinity()
                   : std::max(tail, std::fabs(a - c));
      }
    }
  }
  INFO("cached tail vs one-shot tail: ours ", tail, ", the oracle's own ",
       g::kCachedVsOneShotGap);
  CHECK(tail < kModelTol);
  // The oracle's own two runs agree to 3.4e-06, so this identity is a property
  // of the model and not an artifact of a loose tolerance.
  CHECK(g::kCachedVsOneShotGap < kModelTol);
}

// ─── the latent cache is EQUIVALENT to caching the expanded K/V ──────────────
//
// The spec's "MLA: cache the latent, do not cache what the reference caches"
// decided this and said the equivalence was to be PROVED rather than asserted.
// `ExpandKv` is token-wise — and it is NoPE that makes that true, because a
// positive `qk_rope_head_dim` would put a position-dependent slice alongside the
// latent — so expanding a concatenated history equals concatenating the
// expansions. That is what makes our 512-wide group-0 row reproduce upstream's
// 32,768-wide one.
TEST_CASE("glm5_next W5b-2: expanding a concatenated latent equals concatenating "
          "the expansions") {
  const vllm::Glm5NextParams p = FixtureParams();
  const gn::MlaDims d = gn::MlaDimsFrom(p);
  gn::MlaWeights w;
  w.q_a_proj = Vec(g::kL1MlaQAProj);
  w.q_a_layernorm = Vec(g::kL1MlaQANorm);
  w.q_b_proj = Vec(g::kL1MlaQBProj);
  w.kv_a_proj_with_mqa = Vec(g::kL1MlaKvAProj);
  w.kv_a_layernorm = Vec(g::kL1MlaKvANorm);
  w.k_b_proj = Vec(g::kL1MlaKB);
  w.v_b_proj = Vec(g::kL1MlaVB);
  w.o_proj = Vec(g::kL1MlaOProj);

  // One batch row's latent, from real values rather than a synthetic ramp.
  const std::vector<float> hidden(
      g::kInputsEmbeds, g::kInputsEmbeds + g::kSeqLen * g::kHidden);
  const std::vector<float> latent = gn::CompressKv(d, w, hidden, 1, g::kSeqLen);
  const gn::ExpandedKv whole = gn::ExpandKv(d, w, latent, 1, g::kSeqLen);

  const int64_t cut = g::kPrefill;
  const std::vector<float> head(latent.begin(), latent.begin() + cut * g::kKvLora);
  const std::vector<float> tail(latent.begin() + cut * g::kKvLora, latent.end());
  const gn::ExpandedKv a = gn::ExpandKv(d, w, head, 1, cut);
  const gn::ExpandedKv b = gn::ExpandKv(d, w, tail, 1, g::kSeqLen - cut);

  // Head-major: `[1, num_heads, seq_len, qk_head_dim]`, so the concatenation is
  // per head and not a flat append — which is exactly the splice `AppendRows`
  // has to get right on the batch axis for the same reason.
  float worst = 0.0F;
  for (int64_t h = 0; h < d.num_heads; ++h) {
    for (int64_t s = 0; s < g::kSeqLen; ++s) {
      for (int64_t c = 0; c < d.qk_head_dim(); ++c) {
        const float want =
            whole.key_states[(h * g::kSeqLen + s) * d.qk_head_dim() + c];
        const float got =
            s < cut ? a.key_states[(h * cut + s) * d.qk_head_dim() + c]
                    : b.key_states[(h * (g::kSeqLen - cut) + (s - cut)) *
                                       d.qk_head_dim() + c];
        worst = std::max(worst, std::fabs(want - got));
      }
    }
  }
  INFO("expand(a ++ b) vs expand(a) ++ expand(b): ", worst);
  // EXACT, not within a tolerance: the two computations run the same dot
  // products over the same values in the same order.
  CHECK(worst == 0.0F);
}
