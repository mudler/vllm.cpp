// Qwen4-Exp W5f — `Qwen4ExpTextModel::Forward` against the ORACLE, end to end.
//
// Row MODEL-MM-QWEN4-EXP, issues #2031 and #2336, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS SUITE CAN SEE THAT NO EARLIER ONE COULD ───────────────────────
// Eleven waves of this row gated ONE thing each: `RunGdnBlockPaged`,
// `RunQwen4ExpQsaBlockPaged`, `RunQwen4ExpMoeBlock`, `RunQwen4ExpPleBlock`, the
// two hyper-connection ops. Every one of them is green and NONE of them can see:
//
//   * the LAYER ORDER — which index runs Gated DeltaNet and which runs Qwen
//     Sparse Attention;
//   * the hyper-connection ARITHMETIC BETWEEN the blocks — that `hyper_input` is
//     the RAW stream, that the write-back is rank-1 onto it, and that there are
//     TWO sites per layer and not one;
//   * where the PLE layer SITS — first in its own decoder layer, on the hc-wide
//     stream, and added to it;
//   * the `repeat(1, 1, hc_count)` WIDEN at the top and the `use_combine=False`
//     mixer at the bottom, with NO final RMSNorm between the mixer and the head.
//
// A composition of correct parts in the wrong order is exactly the defect class
// a per-block gate is blind to, and it produces fluent wrong text rather than a
// crash. The golden below is an observation of `Qwen4ExpTextModel.forward`
// itself, so it sees all four.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// transformers **5.16.0**, this row's accepted lane pin
// (`.agents/oracles/transformers.md`), sha256
// `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`, generated
// by `scripts/gen-qwen4-exp-forward-goldens.py`, which ASSERTS that sha256
// against the file it imported before it observes anything. vLLM registers no
// `qwen4_exp` at the pinned revision, so there is no primary oracle to mirror.
//
// ─── THE TOLERANCE, AND WHY IT IS NOT 1e-5 ───────────────────────────────────
// The oracle runs the whole tower in f32; this tree runs the model path at the
// dtype the model resolves, which is bf16, exactly as AGENTS.md "Inherit vLLM
// defaults" requires. Four layers of bf16 rounding is the residual this gate
// measures, and it is NOT a tolerance chosen to make the case pass: the WEIGHTS
// are bf16-exact by construction (the generator asserts a bf16 round trip per
// tensor), so the only rounding is in the activations. The bound is stated as a
// constant below, the MEASURED residual is printed, and every mutation in the
// spec's record separates by more than an order of magnitude above it — which is
// the property that makes the bound a gate rather than a mute switch.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen4_exp_forward.h"
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "support/qwen4_exp_gguf_fixture.h"

#include "../../support/max_abs_diff.h"

namespace {

#include "qwen4_exp_forward_goldens.inc"

using vllm::OwnedTensor;
using vllm::Qwen4ExpLayerKind;
using vllm::Qwen4ExpParams;
using vllm::Qwen4ExpWeights;
using vt::DType;

constexpr int64_t kStream = kFwdHcCount * kFwdHiddenSize;
constexpr int64_t kNgramHeads = (kFwdNgramSize - 1) * kFwdHeadsPerNgram;

// THE BOUND. See the header note: bf16 activations against an f32 oracle over
// four layers. Reported as a measured value by the case itself.
constexpr float kTol = 3.0e-2F;

vt::Queue CpuQ() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

OwnedTensor Owned(DType dt, const std::vector<int64_t>& shape, bool nk = false) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  t.nk = nk;
  t.bytes.assign(static_cast<size_t>(Numel(shape)) * vt::SizeOf(dt), 0);
  return t;
}

// Every weight below arrives from the golden in the ORACLE's own orientation and
// in the RAW HuggingFace gamma parameterization, which is precisely what the
// loader leaves in `Qwen4ExpWeights` (#2218). Nothing is transposed or folded
// here: a fixture that pre-chewed either would be gating itself.
//
// `nk` DEFAULTS TO TRUE BECAUSE `LoadMatmul` PRODUCES TRUE, unconditionally
// (`qwen4_exp_weights.cpp`: every arm — keep-quant blocks, f16 and the bf16
// expansion — passes `nk = true`). The flag is not decoration: consumers that
// branch on it read the SAME BYTES in two orientations. `BorrowWhole` in the MoE
// adapter preserves the source's flag and hands it to `MatmulF32D`, so a fixture
// that built the shared expert's projections `nk = false` fed the seam a [K, N]
// weight where a [N, K] one was meant — the same element count, no shape error,
// and a wrong answer. That was the first defect this gate found, and it was in
// the fixture rather than in the loop, which is why the default is stated here
// once instead of at thirty call sites.
OwnedTensor Bf16(const float* src, const std::vector<int64_t>& shape,
                 bool nk = true) {
  OwnedTensor t = Owned(DType::kBF16, shape, nk);
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  const int64_t n = Numel(shape);
  for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(src[i]);
  return t;
}

OwnedTensor F32(const float* src, const std::vector<int64_t>& shape) {
  OwnedTensor t = Owned(DType::kF32, shape);
  std::memcpy(t.bytes.data(), src, t.bytes.size());
  return t;
}

Qwen4ExpParams MakeParams() {
  Qwen4ExpParams p;
  p.hidden_size = kFwdHiddenSize;
  p.num_hidden_layers = kFwdLayers;
  p.vocab_size = kFwdVocabSize;
  p.rms_norm_eps = kFwdRmsNormEps;
  p.hc_count = kFwdHcCount;
  p.hc_lowrank = kFwdHcLowrank;
  p.num_experts = kFwdNumExperts;
  p.num_experts_per_tok = kFwdNumExpertsPerTok;
  p.moe_intermediate_size = kFwdMoeIntermediateSize;
  p.shared_expert_intermediate_size = kFwdSharedExpertIntermediateSize;
  p.linear_num_key_heads = kFwdLinearNumKeyHeads;
  p.linear_num_value_heads = kFwdLinearNumValueHeads;
  p.linear_key_head_dim = kFwdLinearKeyHeadDim;
  p.linear_value_head_dim = kFwdLinearValueHeadDim;
  p.linear_conv_kernel_dim = kFwdLinearConvKernelDim;
  p.eos_token_id = kFwdEosTokenId;
  p.num_attention_heads = kFwdNumAttentionHeads;
  p.num_key_value_heads = kFwdNumKeyValueHeads;
  p.head_dim = kFwdHeadDim;
  p.rotary_dim = kFwdRotaryDim;
  p.qsa.n_heads = kFwdIndexerNHeads;
  p.qsa.kv_heads = kFwdIndexerKvHeads;
  p.qsa.head_dim = kFwdIndexerHeadDim;
  p.qsa.budget = kFwdIndexerBudget;
  p.qsa.compress_ratio = kFwdIndexerCompressRatio;
  p.ple.embed_dim = kFwdPleEmbedDim;
  p.ple.conv_kernel_size = kFwdPleConvKernelSize;
  p.ple.ngram_size = kFwdNgramSize;
  p.ple.heads_per_ngram = kFwdHeadsPerNgram;
  p.ple.ngram_vocab_size_base = kFwdNgramVocabBase;
  p.ple.make_ngram_vocab_size_divisible_by = kFwdNgramVocabDivisor;
  p.ple.seed = kFwdSeed;
  p.ple.layer_ids_zero_based = {kFwdPleLayerZeroBased};
  // THE LAYER SCHEDULE, spelled out rather than synthesized from an interval,
  // because it is the thing under test. `full_attention_interval` 4 over 4
  // layers gives 3 linear then 1 sparse, which is the released pattern at its
  // shortest, and the golden's own config declares exactly this list.
  p.layer_types = {Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kQwenSparseAttention};
  return p;
}

vllm::HfConfig MakeHfConfig(const Qwen4ExpParams& p) {
  vllm::HfConfig c;
  c.model_type = "qwen4_exp_text";
  c.hidden_size = p.hidden_size;
  c.num_hidden_layers = p.num_hidden_layers;
  c.rms_norm_eps = p.rms_norm_eps;
  c.num_experts = p.num_experts;
  c.linear_num_key_heads = p.linear_num_key_heads;
  c.linear_num_value_heads = p.linear_num_value_heads;
  c.linear_key_head_dim = p.linear_key_head_dim;
  c.linear_value_head_dim = p.linear_value_head_dim;
  c.linear_conv_kernel_dim = p.linear_conv_kernel_dim;
  c.rotary_dim = p.rotary_dim;
  c.rope_theta = kFwdRopeTheta;
  c.rope_parameters.rope_theta = kFwdRopeTheta;
  // THE VALUE THE CHECKPOINT ASKED FOR, and the reason `Qwen4ExpGdnHfConfig`
  // carries it rather than deriving it: `Qwen4ExpParams` has no field for it.
  c.output_gate_type = kFwdOutputGateType;
  c.mamba_ssm_dtype = "float32";
  return c;
}

struct HcArrays {
  const float* norm;
  const float* down;
  const float* up;
  const float* inject;  // nullptr on the terminal mixer
};

void FillHc(vllm::Qwen4ExpGatedResidualWeights& g, const HcArrays& a,
            const Qwen4ExpParams& p) {
  g.hc_norm = Bf16(a.norm, {kStream});
  g.down = Bf16(a.down, {p.hc_lowrank, kStream});
  g.up = Bf16(a.up, {kStream, p.hc_lowrank});
  g.has_inject = a.inject != nullptr;
  if (g.has_inject) g.inject = Bf16(a.inject, {p.hc_count, kStream});
}

struct GdnArrays {
  const float *qkv, *z, *b, *a, *conv, *alog, *dt, *norm, *out;
};

void FillGdn(vllm::Qwen4ExpGdnWeights& g, const GdnArrays& s,
             const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t num_v = p.linear_num_value_heads;
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = num_v * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  // `nk = true` — the `[N, K]` orientation both GGUF loaders produce
  // (`MakeGdnProj` under `GgufLoadPolicy::gdn_expand_nk`). Passing these as
  // `nk = false` would be read transposed with no shape error.
  g.in_proj_qkv = Bf16(s.qkv, {conv_dim, H}, /*nk=*/true);
  g.in_proj_z = Bf16(s.z, {value_dim, H}, /*nk=*/true);
  g.in_proj_b = Bf16(s.b, {num_v, H}, /*nk=*/true);
  g.in_proj_a = Bf16(s.a, {num_v, H}, /*nk=*/true);
  g.conv1d = Bf16(s.conv, {conv_dim, p.linear_conv_kernel_dim});
  g.a_log = F32(s.alog, {num_v});
  g.dt_bias = F32(s.dt, {num_v});
  // THE ONE GAMMA WITH NO FOLD ANYWHERE: `Qwen4ExpTextRMSNormGated.weight` is
  // ones-init and its forward multiplies by `weight` directly, which is why
  // ggml-org/llama.cpp#27742 excludes exactly this tensor and the loader does
  // not unshift it.
  g.norm_weight = Bf16(s.norm, {p.linear_value_head_dim});
  g.out_proj = Bf16(s.out, {H, value_dim}, /*nk=*/true);
}

struct MoeArrays {
  const float *router, *gate_exps, *up_exps, *down_exps, *shared_gate,
      *shared_gate_proj, *shared_up_proj, *shared_down_proj;
};

void FillMoe(vllm::Qwen4ExpMoeWeights& m, const MoeArrays& s,
             const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t I = p.moe_intermediate_size;
  const int64_t S = p.shared_expert_intermediate_size;
  // f32 BY THE LOADER'S OWN CHOICE (`qwen4_exp_weights.cpp:437-447`), and
  // `Qwen4ExpMoeBlockWeights` refuses either of them at any other dtype.
  m.router = F32(s.router, {E, H});
  m.shared_gate = F32(s.shared_gate, {H});
  m.gate_exps = Bf16(s.gate_exps, {E, I, H});
  m.up_exps = Bf16(s.up_exps, {E, I, H});
  m.down_exps = Bf16(s.down_exps, {E, H, I});
  m.shared_gate_proj = Bf16(s.shared_gate_proj, {S, H});
  m.shared_up_proj = Bf16(s.shared_up_proj, {S, H});
  m.shared_down_proj = Bf16(s.shared_down_proj, {H, S});
}

Qwen4ExpWeights MakeWeights(const Qwen4ExpParams& p) {
  Qwen4ExpWeights w;
  w.params = p;
  w.embed_tokens = Bf16(kFwdEmbedTokens, {p.vocab_size, p.hidden_size});
  w.tied_word_embeddings = true;
  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));

  const HcArrays hc[kFwdLayers][2] = {
      {{kFwdL0AttnHcNorm, kFwdL0AttnHcDown, kFwdL0AttnHcUp, kFwdL0AttnHcInject},
       {kFwdL0MlpHcNorm, kFwdL0MlpHcDown, kFwdL0MlpHcUp, kFwdL0MlpHcInject}},
      {{kFwdL1AttnHcNorm, kFwdL1AttnHcDown, kFwdL1AttnHcUp, kFwdL1AttnHcInject},
       {kFwdL1MlpHcNorm, kFwdL1MlpHcDown, kFwdL1MlpHcUp, kFwdL1MlpHcInject}},
      {{kFwdL2AttnHcNorm, kFwdL2AttnHcDown, kFwdL2AttnHcUp, kFwdL2AttnHcInject},
       {kFwdL2MlpHcNorm, kFwdL2MlpHcDown, kFwdL2MlpHcUp, kFwdL2MlpHcInject}},
      {{kFwdL3AttnHcNorm, kFwdL3AttnHcDown, kFwdL3AttnHcUp, kFwdL3AttnHcInject},
       {kFwdL3MlpHcNorm, kFwdL3MlpHcDown, kFwdL3MlpHcUp, kFwdL3MlpHcInject}},
  };
  const GdnArrays gdn[3] = {
      {kFwdL0GdnInProjQkv, kFwdL0GdnInProjZ, kFwdL0GdnInProjB, kFwdL0GdnInProjA,
       kFwdL0GdnConv1d, kFwdL0GdnALog, kFwdL0GdnDtBias, kFwdL0GdnNorm,
       kFwdL0GdnOutProj},
      {kFwdL1GdnInProjQkv, kFwdL1GdnInProjZ, kFwdL1GdnInProjB, kFwdL1GdnInProjA,
       kFwdL1GdnConv1d, kFwdL1GdnALog, kFwdL1GdnDtBias, kFwdL1GdnNorm,
       kFwdL1GdnOutProj},
      {kFwdL2GdnInProjQkv, kFwdL2GdnInProjZ, kFwdL2GdnInProjB, kFwdL2GdnInProjA,
       kFwdL2GdnConv1d, kFwdL2GdnALog, kFwdL2GdnDtBias, kFwdL2GdnNorm,
       kFwdL2GdnOutProj},
  };
  const MoeArrays moe[kFwdLayers] = {
      {kFwdL0MoeRouter, kFwdL0MoeGateExps, kFwdL0MoeUpExps, kFwdL0MoeDownExps,
       kFwdL0MoeSharedGate, kFwdL0MoeSharedGateProj, kFwdL0MoeSharedUpProj,
       kFwdL0MoeSharedDownProj},
      {kFwdL1MoeRouter, kFwdL1MoeGateExps, kFwdL1MoeUpExps, kFwdL1MoeDownExps,
       kFwdL1MoeSharedGate, kFwdL1MoeSharedGateProj, kFwdL1MoeSharedUpProj,
       kFwdL1MoeSharedDownProj},
      {kFwdL2MoeRouter, kFwdL2MoeGateExps, kFwdL2MoeUpExps, kFwdL2MoeDownExps,
       kFwdL2MoeSharedGate, kFwdL2MoeSharedGateProj, kFwdL2MoeSharedUpProj,
       kFwdL2MoeSharedDownProj},
      {kFwdL3MoeRouter, kFwdL3MoeGateExps, kFwdL3MoeUpExps, kFwdL3MoeDownExps,
       kFwdL3MoeSharedGate, kFwdL3MoeSharedGateProj, kFwdL3MoeSharedUpProj,
       kFwdL3MoeSharedDownProj},
  };

  int64_t gi = 0;
  for (int64_t il = 0; il < p.num_hidden_layers; ++il) {
    vllm::Qwen4ExpLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    lw.is_linear_attention =
        p.layer_types[static_cast<size_t>(il)] == Qwen4ExpLayerKind::kLinearAttention;
    FillHc(lw.attn_hc, hc[il][0], p);
    FillHc(lw.mlp_hc, hc[il][1], p);
    FillMoe(lw.moe, moe[il], p);
    if (lw.is_linear_attention) {
      FillGdn(lw.gdn, gdn[gi++], p);
    } else {
      const int64_t H = p.hidden_size;
      lw.qsa.q_proj = Bf16(kFwdL3QsaQProj, {p.num_attention_heads * p.head_dim * 2, H});
      lw.qsa.k_proj = Bf16(kFwdL3QsaKProj, {p.num_key_value_heads * p.head_dim, H});
      lw.qsa.v_proj = Bf16(kFwdL3QsaVProj, {p.num_key_value_heads * p.head_dim, H});
      lw.qsa.o_proj = Bf16(kFwdL3QsaOProj, {H, p.num_attention_heads * p.head_dim});
      lw.qsa.q_norm = Bf16(kFwdL3QsaQNorm, {p.head_dim});
      lw.qsa.k_norm = Bf16(kFwdL3QsaKNorm, {p.head_dim});
      lw.qsa.idx_q_proj = Bf16(kFwdL3QsaIdxQProj, {p.qsa.n_heads * p.qsa.head_dim, H});
      lw.qsa.idx_k_proj = Bf16(kFwdL3QsaIdxKProj, {p.qsa.kv_heads * p.qsa.head_dim, H});
      lw.qsa.idx_q_norm = Bf16(kFwdL3QsaIdxQNorm, {p.qsa.head_dim});
      lw.qsa.idx_k_norm = Bf16(kFwdL3QsaIdxKNorm, {p.qsa.head_dim});
    }
    if (il == kFwdPleLayerZeroBased) {
      lw.has_ple = true;
      lw.ple.key_proj = Bf16(kFwdL1PleKeyProj, {kStream, p.ple.embed_dim});
      lw.ple.value_proj = Bf16(kFwdL1PleValueProj, {p.hidden_size, p.ple.embed_dim});
      lw.ple.norm_key = Bf16(kFwdL1PleNormKey, {kStream});
      lw.ple.norm_query = Bf16(kFwdL1PleNormQuery, {kStream});
      lw.ple.norm_conv = Bf16(kFwdL1PleNormConv, {kStream});
      lw.ple.conv1d = Bf16(kFwdL1PleConv1d, {kStream, p.ple.conv_kernel_size});
      w.ngram_table =
          Bf16(kFwdL1PleNgramTable, {kFwdNgramPaddedVocab, kFwdNgramHeadDim});
    }
  }
  FillHc(w.mixer, {kFwdMixerNorm, kFwdMixerDown, kFwdMixerUp, nullptr}, p);
  return w;
}

std::vector<float> Download(vllm::dense_attn::Dev d, const vt::Tensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  std::vector<float> out(static_cast<size_t>(n));
  if (t.dtype == DType::kBF16) {
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    d.b.Copy(d.q, raw.data(), t.data, raw.size() * sizeof(uint16_t));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] = vt::BF16ToF32(raw[static_cast<size_t>(i)]);
  } else {
    REQUIRE(t.dtype == DType::kF32);
    d.b.Copy(d.q, out.data(), t.data, out.size() * sizeof(float));
  }
  return out;
}


}  // namespace

TEST_CASE(
    "qwen4_exp layer loop: Qwen4ExpTextModel::Forward matches the transformers "
    "5.16.0 oracle end to end") {
  // ─── THE FIXTURE'S OWN CONDITIONING, ASSERTED BEFORE ANYTHING RUNS ───────
  // A top-k router is a DISCRETE SELECTION and its error is BIMODAL: either the
  // two sides pick the same experts and the residual is bf16-sized, or they pick
  // different ones and it is O(1). No tolerance can straddle that. This suite is
  // gateable only because the fixture's worst router margin is far above the
  // hidden-state residual the bf16 model path introduces, and the generator
  // measures that margin on the same run that produced the golden.
  //
  // THE FLOOR IS NOT DECORATION. The first draft of this fixture had a margin of
  // 0.0164 against a residual of 0.0152, layer 0 flipped an expert, and the
  // end-to-end residual read 0.466 — a fixture artefact that reads exactly like a
  // broken layer loop. Regenerating the golden onto another draw can put it back
  // there, and this REQUIRE is what makes that loud instead of silent.
  REQUIRE_MESSAGE(kFwdMinRouterMargin > 0.25,
                  "the fixture's worst router margin is "
                      << kFwdMinRouterMargin
                      << ", which is too close to the bf16 hidden-state residual "
                         "for a top-k selection to be stable; regenerate with "
                         "scripts/gen-qwen4-exp-forward-goldens.py and pick a "
                         "better-conditioned SALT / ROUTER_SCALE rather than "
                         "widening the tolerance");

  const Qwen4ExpParams p = MakeParams();
  const vllm::HfConfig config = MakeHfConfig(p);
  Qwen4ExpWeights w = MakeWeights(p);

  // THE GDN OUTPUT GATE IS SIGMOID HERE AND THE SHARED READER DEFAULTS TO SILU.
  // Asserted rather than assumed, because it is the one field `Qwen4ExpParams`
  // cannot carry and the whole reason `Qwen4ExpGdnHfConfig` takes a source
  // config: the two activations differ on 36 of 48 layers of the released model
  // and no shape check can tell them apart (#489).
  REQUIRE(std::string(kFwdOutputGateType) == "sigmoid");
  REQUIRE(vllm::Qwen4ExpGdnHfConfig(p, config).output_gate_type == "sigmoid");

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t T = kFwdSeqLen;
  std::vector<int32_t> token_ids(static_cast<size_t>(T));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    token_ids[static_cast<size_t>(t)] = static_cast<int32_t>(kFwdInputIds[t]);
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  // ── the step metadata, exactly as the runner builds it for a prefill ──
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  // ── the caches ──
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = p.linear_num_value_heads * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = p.linear_conv_kernel_dim - 1;
  const int64_t ssm_row =
      p.linear_num_value_heads * p.linear_value_head_dim * p.linear_key_head_dim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  vllm::Qwen4ExpForwardCaches caches;
  caches.gdn.resize(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(d, DType::kF32,
                       std::vector<int64_t>{1, p.linear_num_value_heads,
                                            p.linear_value_head_dim,
                                            p.linear_key_head_dim},
                       ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    caches.gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    caches.gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(T * p.num_key_value_heads *
                                               p.head_dim * 2),
                           0);
  std::vector<uint16_t> index_key(
      static_cast<size_t>(T * p.qsa.head_dim), 0);
  std::vector<int32_t> bt{0};
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) slots[static_cast<size_t>(t)] = t;
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, 1, T, p.num_key_value_heads, p.head_dim},
                              kv.data());
  // W5i: the indexer side cache is PAGED, `[num_pages, block_size, D]`. One page
  // of `T` rows here, matching the single-page K/V beside it.
  vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {1, T, p.qsa.head_dim},
                               index_key.data());
  vllm::dense_attn::DBuf bt_b(d, DType::kI32, {1, 1}, bt.data());
  vllm::dense_attn::DBuf slot_b(d, DType::kI64, {T}, slots.data());
  caches.qsa.resize(1);
  caches.qsa[0].kv.data = kv_b.t().data;
  caches.qsa[0].kv.dtype = DType::kBF16;
  caches.qsa[0].kv.num_blocks = 1;
  caches.qsa[0].kv.block_size = T;
  caches.qsa[0].kv.num_kv_heads = p.num_key_value_heads;
  caches.qsa[0].kv.head_size = p.head_dim;
  caches.qsa[0].block_table = bt_b.t();
  caches.qsa[0].slot_mapping = slot_b.t();
  caches.qsa[0].index_key = idx_b.t();
  caches.qsa[0].index_block_table = bt_b.t();

  const int64_t state_len = p.ple.short_conv_state_len();
  std::vector<float> ple_conv(static_cast<size_t>(kStream * state_len), 0.0F);
  std::vector<int64_t> ple_tok(static_cast<size_t>(p.ple.ngram_size - 1), 0);
  // THE RING CARRIES THE STREAM DTYPE (W5k, #2031). It was f32 against a bf16
  // loop, which upstream cannot produce: each cache slot is allocated with the
  // dtype of the tensor that first reaches it (`cache_utils.py:1019-1023`), and
  // the tensor reaching this one is `hidden_states`
  // (`modeling_qwen4_exp.py:1157-1159`). THE GOLDEN DOES NOT MOVE, and that is a
  // property rather than luck: on a `past_len == 0` call the block zeroes the ring
  // on entry and the conv only WRITES it at the end, so nothing reads a ring value
  // and its dtype cannot reach a single output. It would have reached the answer at
  // the first continuing step, which is what W5k makes possible.
  vllm::dense_attn::DBuf ple_conv_b(d, vllm::kQwen4ExpStreamDType,
                                    {1, kStream, state_len}, ple_conv.data());
  caches.ple.resize(1);
  caches.ple[0].conv_state = ple_conv_b.t();
  caches.ple[0].tokens = vllm::dense_attn::MakeTensor(
      ple_tok.data(), DType::kI64, vt::Device{vt::DeviceType::kCPU, 0},
      {1, p.ple.ngram_size - 1});
  caches.ple[0].state_row = 0;

  // ── the n-gram LAYOUT, checked against the oracle's own buffers ──
  // The C++ side DERIVES the prime chain and the offsets; the golden carries
  // what upstream derived. Comparing them is what stops this fixture from
  // agreeing with itself about a 320-million-row table's addressing.
  {
    const vllm::qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
    REQUIRE(static_cast<int64_t>(layout.head_vocab_sizes.size()) == kNgramHeads);
    for (int64_t i = 0; i < kNgramHeads; ++i) {
      CHECK(layout.head_vocab_sizes[static_cast<size_t>(i)] ==
            kFwdNgramHeadVocabSizes[i]);
      CHECK(layout.head_offsets[static_cast<size_t>(i)] == kFwdNgramHeadOffsets[i]);
    }
    CHECK(layout.padded_vocab_size == kFwdNgramPaddedVocab);
    REQUIRE(layout.layer_multipliers.size() == 3);
    for (int i = 0; i < 3; ++i)
      CHECK(layout.layer_multipliers[static_cast<size_t>(i)] ==
            kFwdNgramLayerMultipliers[i]);
  }

  const vllm::Qwen4ExpTextModelOutput out = vllm::Qwen4ExpTextModelForward(
      d, w, config, token_ids, positions, am, gm, caches, /*past_len=*/0);

  REQUIRE(out.storage != nullptr);
  REQUIRE(out.tensor.rank == 2);
  REQUIRE(out.tensor.shape[0] == T);
  REQUIRE(out.tensor.shape[1] == p.hidden_size);

  const std::vector<float> got = Download(d, out.tensor);
  REQUIRE(got.size() == static_cast<size_t>(T * p.hidden_size));

  // FINITENESS BEFORE ANY TOLERANCE (#2272, #449). `vllm_test::MaxAbsDiff`
  // returns +infinity on a non-finite operand rather than the NaN-blind zero the
  // two obvious `std::max` folds return, and it raises on the offending index —
  // but an explicit scan first says WHICH failure a red is.
  for (size_t i = 0; i < got.size(); ++i) {
    REQUIRE_MESSAGE(std::isfinite(got[i]),
                    "the layer loop emitted a non-finite value at index " << i);
  }

  const double worst = vllm_test::MaxAbsDiff(got, kFwdExpectedHidden,
                                             got.size());
  MESSAGE("layer loop vs transformers 5.16.0: max|diff| = " << worst
          << " against a bound of " << kTol);
  CHECK(worst < kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// REACHABILITY (AGENTS.md "Nothing lands dead", `.agents/reachability.md`).
//
// The case above drives `Qwen4ExpTextModelForward` DIRECTLY, which proves the
// arithmetic and proves nothing about whether anything reaches it. This one
// enters through a PRODUCTION entry point — `ModelRegistry::Forward` — on a
// model produced by `ModelRegistry::Load` from a real `qwen4exp` GGUF, and it is
// the first time this architecture has run a forward from one.
//
// WHAT IT CAN AND CANNOT BE. It cannot be a token gate: the synthetic fixture's
// weights are a deterministic ramp, not a checkpoint, so there is no reference
// token stream. It cannot run through `GPUModelRunner` either — that path sets
// `multi_kv`, which `ModelRegistry::Forward` refuses by name for every model, a
// refusal #2353 established must NOT be lifted yet. What it IS: proof that the
// registered hook is entered, opens its handle, assembles the caches, runs all
// four block seams over 4 layers, returns finite logits of the right shape, and
// that `vt::GreedyArgmax` turns them into a token id.
// The mutation that reds it is deleting the `Qwen4ExpTextModelForward` call
// site, which is exactly the mutation `.agents/reachability.md` step 5 asks for
// and which the three preceding waves could only record as VACUOUS.
TEST_CASE(
    "qwen4_exp layer loop: ModelRegistry::Forward reaches it on a loaded "
    "qwen4exp GGUF") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    // Every id in range and the LAST one EOS, so the n-gram hash sees a segment
    // boundary rather than a uniform ramp.
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // The two POSITIONAL channels, sized from the fixture's own geometry: three
  // Gated DeltaNet layers and one Qwen Sparse Attention layer.
  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(
      static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, 1, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids,     pos,      am, gm, attn_kv,
                             gdn,     config,   q,  logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // ─── WHAT THIS CASE ASSERTS: A COMPLETED FORWARD AND A SAMPLED TOKEN ──────
  //
  // W5f could only get to layer 1's PLE block, because W5e-2's
  // `Qwen4ExpPleLayout` derived the n-gram head vocabulary from a DEFAULTED
  // `ngram_vocab_size_base` and refused when the file's stated sizes disagreed.
  // A `qwen4exp` GGUF never states that base — llama.cpp #27742's converter
  // writes the RESOLVED arrays instead — so the comparison was against a
  // default, and it held for exactly one file in the world: the released
  // checkpoint, whose base really is 20,000,000. W5g makes the stated set the
  // authority for the layout, which is what `NgramTableRows` and
  // `Qwen4ExpPleParams`'s own field comment already said it was, and narrows
  // the cross-check to sources that state the base. See
  // `.agents/specs/qwen4-exp-flash-next.md` `## The PLE layout's two sources`.
  //
  // SO THE FORWARD NOW COMPLETES. `ModelRegistry::Forward` enters the hook, the
  // hook opens its handle, assembles the caches, runs all four block seams over
  // 4 layers, gathers the requested row and multiplies it by the lm_head, and
  // returns `[1, vocab]` f32 logits. `vt::GreedyArgmax` — the on-device sampler
  // AGENTS.md routes decode through — then turns them into a token id. That is
  // the FIRST token this architecture has ever produced in this tree.
  //
  // WHAT IT IS NOT. It is not a token GATE. The fixture's weights are a
  // deterministic ramp, not a checkpoint, so no reference token stream exists
  // and the id below is not compared against one. The ARITHMETIC of the whole
  // tower is gated by the golden case above against transformers 5.16.0.
  //
  // FINITENESS IS ASSERTED BEFORE ANYTHING ELSE, and it is not decoration.
  // `MaxAbsDiff`-style folds over `std::max` return the non-NaN operand, so an
  // all-NaN logit row reads as a perfect match to any tolerance and an argmax
  // over it still returns an index in range. Every element is checked.
  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  CHECK(fl.rows == 1);
  CHECK(fl.vocab == kVocab);
  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  REQUIRE(fl.device_tensor.dtype == DType::kF32);

  std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
  d.b.Copy(q, host.data(), fl.device_tensor.data, host.size() * sizeof(float));
  d.b.Synchronize(q);

  int finite = 0;
  for (float v : host) finite += std::isfinite(v) ? 1 : 0;
  REQUIRE(finite == static_cast<int>(host.size()));
  // A CONSTANT row would satisfy every check above and mean the tower
  // contributed nothing — the lm_head times a zero hidden is finite, in range,
  // and argmaxes to 0. So the row must actually VARY.
  float lo = host[0];
  float hi = host[0];
  for (float v : host) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  CHECK(hi > lo);

  // ─── THE TOKEN ────────────────────────────────────────────────────────────
  // `vt::GreedyArgmax` is the production greedy sampler (`vt/ops.h`), the same
  // primitive `GPUModelRunner` samples with, and it is driven here on the
  // device tensor the forward returned rather than on a host copy.
  std::vector<int64_t> tok(1, -1);
  {
    vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
    vt::Tensor tt = tok_b.t();
    vt::GreedyArgmax(q, tt, fl.device_tensor);
    tok_b.Download(d, tok.data());
  }
  MESSAGE("qwen4_exp sampled token id: " << tok[0] << " of " << fl.vocab
                                         << " (logit range [" << lo << ", "
                                         << hi << "])");
  CHECK(tok[0] >= 0);
  CHECK(tok[0] < fl.vocab);
  // The argmax must be the row's own maximum, so the sampler is reading THESE
  // logits and not an uninitialised buffer.
  CHECK(host[static_cast<size_t>(tok[0])] == hi);
  // And the old refusals must be gone: neither the pre-W5f unconditional one
  // nor W5f's PLE layout stop can still be reachable on this input.
  CHECK(finite > 0);

  // ─── AND THE FORWARD DEPENDS ON ITS INPUT ─────────────────────────────────
  //
  // EVERY ASSERTION ABOVE IS SATISFIED BY A HOOK THAT NEVER READS `token_ids`.
  // MUT-REACH measured exactly that: with the whole four-layer tower deleted,
  // 90 of this case's assertions still held and only `CHECK(hi > lo)` went red.
  // But `hi > lo` says the row is not CONSTANT, and `lm_head` times any
  // non-constant hidden clears it — including a hidden derived from nothing but
  // the shapes. A forward that ignored the prompt entirely would still pass
  // every check above, so "a token came out" is not yet "a token came out of
  // THIS prompt".
  //
  // A SECOND PROMPT IS THE MISSING HALF, and it is the sibling row's own shape:
  // `test_glm5_next_forward.cpp` runs a second `Step` and asserts the logits
  // moved. THE CACHES ARE REBUILT RATHER THAN REUSED. The first forward WRITES
  // the paged KV and both recurrent states, so a second call on the same
  // buffers could differ because the cache was dirty rather than because the
  // tokens changed — which a `token_ids`-ignoring hook would also produce. Fresh
  // zeroed buffers leave the prompt as the only input that moved.
  std::vector<int32_t> ids2(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    // A DIFFERENT prompt, still every id in range and still EOS-terminated, so
    // the n-gram hash sees the same segment structure and a different history.
    ids2[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : (t + 1) * 4 + 1);
    REQUIRE(ids2[static_cast<size_t>(t)] < static_cast<int32_t>(kVocab));
  }
  REQUIRE(ids2 != ids);

  std::vector<std::vector<float>> ssm2(3), conv2(3);
  std::vector<vllm::dense_attn::DBuf> ssm2_b, conv2_b;
  std::vector<vllm::GdnStateCache> gdn2(3);
  ssm2_b.reserve(3);
  conv2_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm2[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv2[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm2_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm2[i].data());
    conv2_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, conv_dim, conv_len},
                         conv2[i].data());
    gdn2[static_cast<size_t>(i)].ssm_state = ssm2_b.back().t();
    gdn2[static_cast<size_t>(i)].conv_state = conv2_b.back().t();
  }
  std::vector<uint16_t> kv2(
      static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv2_b(d, DType::kBF16,
                               {2, 1, T, kKvHeads, kHeadDim}, kv2.data());
  std::vector<vllm::PagedKvCache> attn_kv2(1);
  attn_kv2[0] = attn_kv[0];
  attn_kv2[0].data = kv2_b.t().data;

  vllm::ModelForwardInput in_p2{ids2, pos,    am, gm, attn_kv2,
                                gdn2, config, q,  logits_indices};
  in_p2.num_reqs = 1;
  in_p2.gdn_state_slots = 1;

  vllm::ForwardLogits fl2;
  REQUIRE_NOTHROW(fl2 = vllm::ModelRegistry::Forward(*model, in_p2));
  REQUIRE(fl2.on_device());
  REQUIRE(fl2.device_tensor.data != nullptr);
  REQUIRE(fl2.rows == fl.rows);
  REQUIRE(fl2.vocab == fl.vocab);
  std::vector<float> host2(static_cast<size_t>(fl2.rows * fl2.vocab), 0.0F);
  d.b.Copy(q, host2.data(), fl2.device_tensor.data, host2.size() * sizeof(float));
  d.b.Synchronize(q);
  // `MaxAbsDiff` RAISES on a non-finite operand rather than folding it away, so
  // this is also the second row's finiteness check.
  const double moved = vllm_test::MaxAbsDiff(host, host2);
  MESSAGE("qwen4_exp second-prompt logit movement: " << moved);
  CHECK(moved > 0.0);

  // ─── THE TWO REFUSALS THIS HOOK ADVERTISES, GATED BY THEIR MESSAGE ────────
  //
  // A BARE `CHECK_THROWS` HERE WOULD BE A MUTE SWITCH, and it was one until this
  // repair. Every input below reaches the loop if its refusal is deleted, and
  // the loop then throws anyway at the SAME PLE layout cross-check the golden
  // reach above lands on. An unrelated exception satisfied the assertion, so
  // deleting either `VT_CHECK` in `qwen4_exp_registry.cpp` left this suite at
  // 2 cases / 76 assertions SUCCESS and the scaffold suite at 12 / 294.
  //
  // So each refusal is asserted TWO-SIDED on its own MESSAGE: the bytes that
  // identify it are present, AND the PLE message that means the input got past
  // it is absent. Deleting either `VT_CHECK` flips both halves of its pair.
  auto refusal_message = [&](const vllm::ModelForwardInput& bad,
                             const char* what) {
    std::string msg;
    try {
      (void)vllm::ModelRegistry::Forward(*model, bad);
      FAIL(what);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  // `past_len != 0` means the QSA indexer side cache and the PLE conv ring and
  // n-gram history would have had to persist across a step, and no channel
  // carries them. The refusal must fire BEFORE the loop runs.
  vllm::v1::CommonAttentionMetadata am2 = am;
  am2.seq_lens.assign(1, static_cast<int32_t>(T + 1));
  vllm::ModelForwardInput in2{ids,     pos,      am2, gm, attn_kv,
                              gdn,     config,   q,   logits_indices};
  in2.num_reqs = 1;
  in2.gdn_state_slots = 1;
  const std::string past_len_msg = refusal_message(
      in2,
      "ModelRegistry::Forward returned on a step at past_len 1 with NO by-name "
      "cache index; the scratch-arm refusal is gone");
  INFO("the past_len refusal said: ", past_len_msg);
  CHECK(past_len_msg.find("Qwen4ExpForConditionalGeneration") !=
        std::string::npos);
  // W5k: THE REFUSAL IS NOW ABOUT THIS ARM, NOT ABOUT THE MODEL. It used to say
  // the hook served a "SINGLE-SHOT PREFILL" because the PLE states could not
  // persist at all; they persist now, in the engine's own recurrent group, and a
  // continuing step on the BY-NAME arm returns a token (see the by-name case).
  // What is refused here is the POSITIONAL arm specifically: nothing published
  // the PLE states, so the hook allocates them per call, and a per-call buffer is
  // zeroed on entry — every step would re-seed the history with EOS and decode as
  // though each token were the first.
  CHECK(past_len_msg.find("NO by-name cache index") != std::string::npos);
  // The word that named the OLD limit must be GONE, not merely joined by a new
  // one: a message that still claims a single-shot-prefill hook would send the
  // next reader to rebuild what this wave landed.
  CHECK(past_len_msg.find("SINGLE-SHOT") == std::string::npos);
  // The VALUE, not only the word: the hook reports the past_len it was handed.
  CHECK(past_len_msg.find("at past_len 1") != std::string::npos);
  // It stopped at the boundary and never entered the loop, so the PLE layout
  // cross-check the golden reach above lands on cannot be what threw.
  CHECK(past_len_msg.find("qwen4_exp ple layout") == std::string::npos);

  // `num_reqs != 1` is the OTHER refusal, and nothing drove it at all before
  // this repair. `RunQwen4ExpQsaBlockPaged` takes a block_table of i32
  // [1, max_pages], so a ragged multi-request batch has no plumbing here.
  vllm::ModelForwardInput in3{ids,     pos,      am, gm, attn_kv,
                              gdn,     config,   q,  logits_indices};
  in3.num_reqs = 2;
  in3.gdn_state_slots = 1;
  const std::string num_reqs_msg = refusal_message(
      in3,
      "ModelRegistry::Forward returned on a step carrying num_reqs 2; the "
      "one-sequence-per-call refusal is gone");
  INFO("the num_reqs refusal said: ", num_reqs_msg);
  CHECK(num_reqs_msg.find("Qwen4ExpForConditionalGeneration") !=
        std::string::npos);
  CHECK(num_reqs_msg.find("ONE sequence per call") != std::string::npos);
  // The VALUE again: the count the step actually carried.
  CHECK(num_reqs_msg.find("the step carries 2") != std::string::npos);
  CHECK(num_reqs_msg.find("qwen4_exp ple layout") == std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// W5j — THE BY-NAME CACHE CHANNEL, THROUGH THE PRODUCTION ENTRY POINT.
//
// The case above enters `ModelRegistry::Forward` with `multi_kv == nullptr`,
// which is the shape no engine step has: `qwen4_exp` publishes THREE KV cache
// groups, so `GPUModelRunner` sets `multi_cache_topology_` and hands the forward
// a `MultiKvCacheIndex`. Three things stood between that step and a token, and
// this case drives all three.
//
//   1. `ModelRegistry::Forward` refused ANY non-null `multi_kv`. It now refuses
//      only a topology reaching a forward whose `ModelFactory::consumes_multi_kv`
//      is false, and this architecture sets it. A green run below IS the proof
//      that the guard let it past; every refusal asserted further down also
//      asserts the ENGINE's message is ABSENT, so a case that stopped at the
//      guard cannot read as this hook refusing.
//
//   2. `attn_kv` ARRIVES AT 2 x n_qsa, not n_qsa. The runner allocates one paged
//      buffer per (attention group x layer) and this model publishes TWO
//      attention groups over the same QSA layers — group 0's K/V and group 2's
//      indexer side cache. The hook's old `attn_kv.size() == n_qsa` assertion
//      would have turned the lifted guard into a DIFFERENT refusal rather than a
//      token, which is why the channel is resolved by NAME here.
//
//   3. W5i's production path used a per-call scratch behind an IDENTITY block
//      table, so the permutation it gates was never exercised by an allocator.
//      Group 2's own buffer and its own gathered table reach the block now.
//
// ─── HOW THE PAGING IS GATED, AND WHY NOT BY VALUE ───────────────────────────
// W5i measured the trap: 9 of 23 rows landed in the wrong physical page and BOTH
// the paged-vs-contiguous value diff (0 of 1472 words) and the oracle bound
// stayed green, because the store and the read share ONE translation and a
// consistently wrong one returns the right answer. So the gate here is
// STRUCTURAL and comparative:
//
//   * the group-2 buffer is POISONED with a sentinel before the step, and after
//     it exactly the rows the group's OWN table names for logical [0, T) differ
//     from the sentinel — every other row is byte-for-byte the sentinel still;
//   * the SAME step is run twice with two DIFFERENT group-2 tables. The logits
//     must be BIT-IDENTICAL (paging is transparent to the arithmetic) and the
//     written row sets must DIFFER (paging is not transparent to the buffer).
//     A hook that ignored the table would pass the first half and fail this one;
//   * the written set is asserted to be neither the IDENTITY set nor group 0's
//     translation, which are the two wrong tables physically present in the step.
//
// The tables are NON-IDENTITY and FIXED-POINT-FREE over the pages actually used,
// and the final page is PARTIAL (T = 6 over a page size of 4), so a translation
// that dropped the row-within-page term or clamped the last page is separated.
TEST_CASE(
    "qwen4_exp: ModelRegistry::Forward serves the BY-NAME multi-cache channel") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  // The fixture's topology: layers 0,1,2 linear_attention and layer 3
  // qwen_sparse_attention, so ONE QSA layer and THREE recurrent ones.
  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t T = 6;
  // `RunQwen4ExpQsaBlockPaged` requires the K/V page size to be a multiple of
  // `indexer_compress_ratio`, so 4 is the smallest legal page here — and it
  // leaves T = 6 spanning two pages with a PARTIAL final one.
  constexpr int64_t kPage = 4;
  static_assert(kPage % kCompressRatio == 0, "the QSA page must hold whole compress blocks");
  constexpr int64_t kCols = 2;   // pages named per sequence
  constexpr int64_t kBlocks0 = 3;  // group 0's pool
  constexpr int64_t kBlocks2 = 4;  // group 2's pool — a DIFFERENT size on purpose

  // THE THREE PERMUTATIONS, all distinct over the two pages this step uses and
  // none of them the identity or a fixed point there.
  const std::vector<int32_t> bt0{2, 0};        // group 0's map
  const std::vector<int32_t> bt2_a{1, 3};      // group 2's map, run A
  const std::vector<int32_t> bt2_b{2, 0};      // group 2's map, run B — same
                                               // pages as group 0's, in group
                                               // 2's OWN pool
  const std::vector<int32_t> bt_recurrent{0};  // the recurrent group's row

  // The physical row a logical position lands on under a given table.
  const auto rows_of = [](const std::vector<int32_t>& table, int64_t n) {
    std::vector<int64_t> r(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t)
      r[static_cast<size_t>(t)] =
          static_cast<int64_t>(table[static_cast<size_t>(t / kPage)]) * kPage + t % kPage;
    return r;
  };

  std::vector<int32_t> ids(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
  for (int32_t v : ids) REQUIRE(v < static_cast<int32_t>(kVocab));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  // The bf16 sentinel the group-2 buffer is poisoned with: 0x3F80 is 1.0f.
  // A row the block wrote is a raw indexer key and is not plausibly sixteen
  // consecutive exact 1.0s, and a row it did not write is EXACTLY this.
  constexpr uint16_t kPoison = 0x3F80;
  const int64_t kIdxRows = kBlocks2 * kPage;

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // ─── ONE BY-NAME STEP, ASSEMBLED THE WAY THE RUNNER ASSEMBLES ONE ─────────
  // Every buffer is FRESH per call: the first step writes the paged K/V, both
  // recurrent states and the indexer pages, so reusing them would let a second
  // run differ because the caches were dirty rather than because the input was.
  struct StepResult {
    std::vector<float> logits;
    std::vector<uint16_t> index_cache;
    std::vector<uint16_t> kv_cache;
    int64_t vocab = 0;
    int64_t token = -1;
  };
  // `corrupt` names ONE way the channel can be malformed; 0 is a well-formed
  // step. When `err` is non-null the call is expected to refuse and the message
  // is returned through it instead of the step's results.
  enum Corrupt { kNone = 0, kWrongName = 1, kWrongKind = 2, kNoGroupTable = 3 };
  const auto run_step = [&](const std::vector<int32_t>& idx_table,
                            const std::vector<int32_t>& prompt,
                            int corrupt = kNone, std::string* err = nullptr) {
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.block_table_num_cols = static_cast<int>(kCols);
    am.block_table_tensor = bt0;
    am.seq_lens.assign(1, static_cast<int32_t>(T));
    am.query_start_loc = {0, static_cast<int32_t>(T)};
    am.slot_mapping.resize(static_cast<size_t>(T));
    {
      const std::vector<int64_t> r0 = rows_of(bt0, T);
      for (int64_t t = 0; t < T; ++t)
        am.slot_mapping[static_cast<size_t>(t)] =
            static_cast<int32_t>(r0[static_cast<size_t>(t)]);
    }

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_prefills = 1;
    gm.num_prefill_tokens = static_cast<int>(T);
    gm.num_actual_tokens = static_cast<int>(T);
    gm.has_initial_state = std::vector<uint8_t>{0};
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_state_indices = std::vector<int32_t>{0};
    gm.prefill_has_initial_state = std::vector<uint8_t>{0};
    {
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    }

    const int64_t key_dim = kNumKHeads * kLinHeadDim;
    const int64_t value_dim = kNumVHeads * kLinHeadDim;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    const int64_t conv_len = kConvKernel - 1;
    const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

    std::vector<std::vector<float>> ssm(kGdnLayers), conv(kGdnLayers);
    std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
    std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
    ssm_b.reserve(kGdnLayers);
    conv_b.reserve(kGdnLayers);
    for (int i = 0; i < kGdnLayers; ++i) {
      ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
      conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
      ssm_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                         ssm[i].data());
      conv_b.emplace_back(d, DType::kF32,
                          std::vector<int64_t>{1, conv_dim, conv_len}, conv[i].data());
      gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
      gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
    }

    // GROUP 0: [2, num_blocks, block_size, num_kv_heads, head_size].
    std::vector<uint16_t> kv(
        static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
    vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                                {2, kBlocks0, kPage, kKvHeads, kHeadDim}, kv.data());
    // GROUP 2: the fused MLA page, [num_blocks, block_size, indexer_head_dim],
    // POISONED so an unwritten row is distinguishable from a written zero.
    std::vector<uint16_t> idx(static_cast<size_t>(kIdxRows * kIdxHeadDim), kPoison);
    vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                                 idx.data());

    std::vector<vllm::PagedKvCache> attn_kv(2);
    attn_kv[0].data = kv_b.t().data;
    attn_kv[0].dtype = DType::kBF16;
    attn_kv[0].num_blocks = kBlocks0;
    attn_kv[0].block_size = kPage;
    attn_kv[0].num_kv_heads = kKvHeads;
    attn_kv[0].head_size = kHeadDim;
    attn_kv[1].data = idx_b.t().data;
    attn_kv[1].dtype = DType::kBF16;
    attn_kv[1].num_blocks = kBlocks2;
    attn_kv[1].block_size = kPage;
    // `indexer_kv_heads` is 1 — upstream requires it — so the MLA page is one
    // vector of `indexer_head_dim` per token slot.
    attn_kv[1].num_kv_heads = kIdxKvHeads;
    attn_kv[1].head_size = kIdxHeadDim;

    // ─── THE CHANNEL, IN THE RUNNER'S OWN PUBLICATION ORDER ────────────────
    // One pass over the groups in the order `MakeQwen4ExpKVCache` published
    // them, emitting each group's names in the group's own order — which is
    // exactly `runner.cpp`'s by-name index build. The recurrent group sits
    // BETWEEN the two attention groups, so a flat index is NOT a payload slot.
    std::vector<std::string> names;
    std::vector<int32_t> group_ids, layer_indices, payload_slots;
    std::vector<uint8_t> payload_kinds;
    names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
    group_ids.push_back(0);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(0);
    for (int i = 0; i < kGdnLayers; ++i) {
      names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
      group_ids.push_back(1);
      layer_indices.push_back(i);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
      payload_slots.push_back(i);
    }
    names.push_back("model.layers." + std::to_string(kQsaLayer) +
                    ".self_attn.indexer.k_cache");
    group_ids.push_back(2);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(1);

    std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, idx_table};
    std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                    static_cast<int32_t>(kCols)};

    // ─── THE THREE MALFORMED CHANNELS ─────────────────────────────────────
    // Each is a shape the ENGINE cannot judge — it does not know which names
    // this model expects — and each returns a wrong buffer with no shape error
    // if the hook trusts it.
    if (corrupt == kWrongName) {
      // The publisher and the consumer disagree about where the side cache is.
      names.back() = "model.layers." + std::to_string(kQsaLayer) +
                     ".self_attn.indexer.key_cache";
    } else if (corrupt == kWrongKind) {
      // A recurrent group classified as paged: `gdn_state[0]` would be read out
      // of `attn_kv` instead.
      payload_kinds[1] = static_cast<uint8_t>(vllm::KvCachePayload::kPaged);
    } else if (corrupt == kNoGroupTable) {
      // Group 2 allocated and its page map never gathered — the exact state
      // W5c-2 exists to remove, and the one that leaves a cache unreadable.
      group_tables[2].clear();
    }

    vllm::MultiKvCacheIndex mk;
    mk.layer_names = &names;
    mk.group_ids = &group_ids;
    mk.layer_indices = &layer_indices;
    mk.payload_kinds = &payload_kinds;
    mk.payload_slots = &payload_slots;
    mk.group_block_tables = &group_tables;
    mk.group_block_table_cols = &group_cols;
    if (corrupt == kNone) {
      REQUIRE(mk.size() == names.size());
      REQUIRE(mk.num_paged() == 2);
      REQUIRE(mk.num_recurrent() == kGdnLayers);
      REQUIRE(mk.num_published_groups() == 3);
      REQUIRE(mk.num_group_block_tables() == 3);
    }

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{prompt,  pos,    am, gm, attn_kv,
                               gdn,     config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;

    StepResult out;
    vllm::ForwardLogits fl;
    if (err != nullptr) {
      try {
        (void)vllm::ModelRegistry::Forward(*model, in);
        *err = "<the step RETURNED; the refusal is gone>";
      } catch (const std::exception& e) {
        *err = e.what();
      }
      return out;
    }
    // The guard is what this must get past: `ModelRegistry::Forward` refused
    // every non-null `multi_kv` before W5j.
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
    REQUIRE(fl.on_device());
    REQUIRE(fl.device_tensor.data != nullptr);
    REQUIRE(fl.device_tensor.dtype == DType::kF32);
    REQUIRE(fl.rows == 1);
    out.vocab = fl.vocab;
    out.logits.assign(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
    d.b.Copy(q, out.logits.data(), fl.device_tensor.data,
             out.logits.size() * sizeof(float));
    d.b.Synchronize(q);
    out.index_cache.assign(idx.size(), 0);
    idx_b.Download(d, out.index_cache.data());
    out.kv_cache.assign(kv.size(), 0);
    kv_b.Download(d, out.kv_cache.data());
    // THE TOKEN, sampled on the device the forward returned, by the same
    // `vt::GreedyArgmax` `GPUModelRunner` samples with. AGENTS.md routes decode
    // through on-device sampling and this is that primitive, not a host argmax.
    {
      std::vector<int64_t> t1(1, -1);
      vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
      vt::Tensor tt = tok_b.t();
      vt::GreedyArgmax(q, tt, fl.device_tensor);
      tok_b.Download(d, t1.data());
      out.token = t1[0];
    }
    return out;
  };

  // ─── RUN A ────────────────────────────────────────────────────────────────
  StepResult a;
  REQUIRE_NOTHROW(a = run_step(bt2_a, ids));
  CHECK(a.vocab == kVocab);

  int finite = 0;
  for (float v : a.logits) finite += std::isfinite(v) ? 1 : 0;
  REQUIRE(finite == static_cast<int>(a.logits.size()));
  float lo = a.logits[0];
  float hi = a.logits[0];
  for (float v : a.logits) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  // Not a constant row: the lm_head times a zero hidden would be finite, in
  // range and argmax to 0.
  CHECK(hi > lo);

  MESSAGE("qwen4_exp by-name step: sampled token id "
          << a.token << " of " << a.vocab << " (logit range [" << lo << ", "
          << hi << "])");
  CHECK(a.token >= 0);
  CHECK(a.token < a.vocab);
  // The argmax must be THIS row's maximum, so the sampler read these logits and
  // not an uninitialised buffer.
  CHECK(a.logits[static_cast<size_t>(a.token)] == hi);

  // ─── THE STRUCTURAL ROW-SET ASSERTION ─────────────────────────────────────
  // Exactly the rows group 2's OWN table names for logical [0, T) were written,
  // and nothing else was touched. A value gate cannot see this; W5i measured a
  // 9-of-23 mis-page that left every value check green.
  const auto written_rows = [&](const std::vector<uint16_t>& cache) {
    std::vector<int64_t> w;
    for (int64_t r = 0; r < kIdxRows; ++r) {
      bool touched = false;
      for (int64_t c = 0; c < kIdxHeadDim; ++c)
        if (cache[static_cast<size_t>(r * kIdxHeadDim + c)] != kPoison) touched = true;
      if (touched) w.push_back(r);
    }
    return w;
  };
  std::vector<int64_t> expect_a = rows_of(bt2_a, T);
  std::sort(expect_a.begin(), expect_a.end());
  const std::vector<int64_t> got_a = written_rows(a.index_cache);
  {
    std::string s;
    for (int64_t r : got_a) s += std::to_string(r) + " ";
    INFO("group-2 rows written under table {1,3}: ", s);
    CHECK(got_a == expect_a);
  }
  // And it is NEITHER of the two wrong maps physically present in the step.
  std::vector<int64_t> identity(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) identity[static_cast<size_t>(t)] = t;
  CHECK(got_a != identity);
  std::vector<int64_t> via_group0 = rows_of(bt0, T);
  std::sort(via_group0.begin(), via_group0.end());
  CHECK(got_a != via_group0);

  // ─── RUN B: THE SAME STEP THROUGH A DIFFERENT GROUP-2 MAP ─────────────────
  // Paging is transparent to the ARITHMETIC and not to the BUFFER. A hook that
  // ignored the group's table would pass the first half of this pair and fail
  // the second.
  StepResult b;
  REQUIRE_NOTHROW(b = run_step(bt2_b, ids));
  REQUIRE(b.logits.size() == a.logits.size());
  const double drift = vllm_test::MaxAbsDiff(a.logits, b.logits);
  MESSAGE("qwen4_exp by-name: logit drift across two group-2 page maps = " << drift);
  CHECK(drift == 0.0);
  std::vector<int64_t> expect_b = rows_of(bt2_b, T);
  std::sort(expect_b.begin(), expect_b.end());
  const std::vector<int64_t> got_b = written_rows(b.index_cache);
  CHECK(got_b == expect_b);
  CHECK(got_b != got_a);

  // ─── AND THE STEP STILL DEPENDS ON ITS PROMPT ─────────────────────────────
  // Every assertion above is satisfied by a hook that never reads `token_ids`:
  // MUT-REACH on this row measured 90 of 91 assertions surviving the deletion of
  // the whole tower.
  std::vector<int32_t> ids2(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    ids2[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : (t + 1) * 2 + 1);
  for (int32_t v : ids2) REQUIRE(v < static_cast<int32_t>(kVocab));
  REQUIRE(ids2 != ids);
  StepResult c;
  REQUIRE_NOTHROW(c = run_step(bt2_a, ids2));
  const double moved = vllm_test::MaxAbsDiff(a.logits, c.logits);
  MESSAGE("qwen4_exp by-name second-prompt logit movement: " << moved);
  CHECK(moved > 0.0);
  // The indexer keys are the prompt's own, so the CONTENT of the written rows
  // moves too while the row SET does not.
  CHECK(written_rows(c.index_cache) == expect_a);
  // ─── WHAT THIS FIXTURE CANNOT GATE, MEASURED RATHER THAN ASSUMED ─────────
  // The CONTENT of neither cache moves with the prompt here, and the count
  // below is printed so the next reader does not write the assertion that
  // reddened on this exact line. It is a DYNAMIC-RANGE property of the fixture,
  // not a defect: the four-layer ramp puts the layer-3 activations near 2^18,
  // where one bf16 ULP is ~1024, and the two prompts move the logits by 31.84
  // out of 95090 — 0.03%, an order of magnitude under one ULP at that exponent.
  // The paged K/V, whose store is `vt::ReshapeAndCache` and is gated by every
  // other model in this tree, is prompt-invariant here for the same reason and
  // by the same count, which is what separates "the fixture saturates" from
  // "the indexer writes one row T times".
  //
  // So the ROW SET is what gates the paging on this fixture, exactly as
  // `.agents/reachability.md` and W5i's measurement require, and the CONTENT is
  // gated at the block instead (`test_qwen4_exp_qsa_block.cpp`, the W5i case,
  // whose values are chosen to separate). Recorded in the spec's `## Owed` as a
  // fixture the row may want to rescale.
  {
    int idx_moved = 0;
    for (size_t i = 0; i < a.index_cache.size(); ++i)
      idx_moved += a.index_cache[i] != c.index_cache[i] ? 1 : 0;
    int kv_moved = 0;
    for (size_t i = 0; i < a.kv_cache.size(); ++i)
      kv_moved += a.kv_cache[i] != c.kv_cache[i] ? 1 : 0;
    MESSAGE("qwen4_exp by-name: across two prompts, "
            << idx_moved << " of " << a.index_cache.size()
            << " indexer-cache words moved and " << kv_moved << " of "
            << a.kv_cache.size() << " paged K/V words moved");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// W5j — WHAT THE LIFTED GUARD STILL REFUSES.
//
// `ModelRegistry::Forward` no longer stops this architecture's multi-cache
// topology, so the model is now the only thing that can judge the CONTENT of the
// channel: the engine does not know which layer names `qwen4_exp` expects, which
// payload kind each is, or which group's block table has to have been gathered.
// Three ways the channel can lie, each refused by name at the hook's boundary.
//
// A BARE `CHECK_THROWS` HERE WOULD BE A MUTE SWITCH — that exact defect was
// found on this row, where the only assertion on a refusal was satisfied by a
// DIFFERENT exception thrown earlier. So each is asserted TWO-SIDED on its
// message: the bytes that identify THIS refusal are present, and the bytes of
// two OTHER refusals are absent —
//
//   * `qwen4_exp ple layout`, which is where an input that got past the boundary
//     and into the loop lands;
//   * `does not consume a cache set keyed by layer name`, the ENGINE's guard. A
//     case that stopped there proves nothing about the hook, and its absence is
//     what makes the green run above a lift and not a coincidence.
TEST_CASE(
    "qwen4_exp: the by-name channel is refused by name when it cannot be "
    "consumed") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t T = 6;
  constexpr int64_t kPage = 4;
  constexpr int64_t kCols = 2;
  constexpr int64_t kBlocks0 = 3;
  constexpr int64_t kBlocks2 = 4;
  const std::vector<int32_t> bt0{2, 0};
  const std::vector<int32_t> bt2{1, 3};
  const std::vector<int32_t> bt_recurrent{0};

  std::vector<int32_t> ids(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  enum Corrupt { kWrongName = 1, kWrongKind = 2, kNoGroupTable = 3 };
  const auto refusal = [&](int corrupt) {
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.block_table_num_cols = static_cast<int>(kCols);
    am.block_table_tensor = bt0;
    am.seq_lens.assign(1, static_cast<int32_t>(T));
    am.query_start_loc = {0, static_cast<int32_t>(T)};
    am.slot_mapping.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(
          static_cast<int64_t>(bt0[static_cast<size_t>(t / kPage)]) * kPage + t % kPage);

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_prefills = 1;
    gm.num_prefill_tokens = static_cast<int>(T);
    gm.num_actual_tokens = static_cast<int>(T);
    gm.has_initial_state = std::vector<uint8_t>{0};
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_state_indices = std::vector<int32_t>{0};
    gm.prefill_has_initial_state = std::vector<uint8_t>{0};
    {
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    }

    const int64_t key_dim = kNumKHeads * kLinHeadDim;
    const int64_t value_dim = kNumVHeads * kLinHeadDim;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    const int64_t conv_len = kConvKernel - 1;
    const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;
    std::vector<std::vector<float>> ssm(kGdnLayers), conv(kGdnLayers);
    std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
    std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
    ssm_b.reserve(kGdnLayers);
    conv_b.reserve(kGdnLayers);
    for (int i = 0; i < kGdnLayers; ++i) {
      ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
      conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
      ssm_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                         ssm[i].data());
      conv_b.emplace_back(d, DType::kF32,
                          std::vector<int64_t>{1, conv_dim, conv_len}, conv[i].data());
      gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
      gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
    }

    std::vector<uint16_t> kv(
        static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
    vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                                {2, kBlocks0, kPage, kKvHeads, kHeadDim}, kv.data());
    std::vector<uint16_t> idx(
        static_cast<size_t>(kBlocks2 * kPage * kIdxHeadDim), 0);
    vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                                 idx.data());
    std::vector<vllm::PagedKvCache> attn_kv(2);
    attn_kv[0].data = kv_b.t().data;
    attn_kv[0].dtype = DType::kBF16;
    attn_kv[0].num_blocks = kBlocks0;
    attn_kv[0].block_size = kPage;
    attn_kv[0].num_kv_heads = kKvHeads;
    attn_kv[0].head_size = kHeadDim;
    attn_kv[1].data = idx_b.t().data;
    attn_kv[1].dtype = DType::kBF16;
    attn_kv[1].num_blocks = kBlocks2;
    attn_kv[1].block_size = kPage;
    attn_kv[1].num_kv_heads = kIdxKvHeads;
    attn_kv[1].head_size = kIdxHeadDim;

    std::vector<std::string> names;
    std::vector<int32_t> group_ids, layer_indices, payload_slots;
    std::vector<uint8_t> payload_kinds;
    names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
    group_ids.push_back(0);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(0);
    for (int i = 0; i < kGdnLayers; ++i) {
      names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
      group_ids.push_back(1);
      layer_indices.push_back(i);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
      payload_slots.push_back(i);
    }
    names.push_back("model.layers." + std::to_string(kQsaLayer) +
                    ".self_attn.indexer.k_cache");
    group_ids.push_back(2);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(1);
    std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, bt2};
    std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                    static_cast<int32_t>(kCols)};

    if (corrupt == kWrongName) {
      names.back() = "model.layers." + std::to_string(kQsaLayer) +
                     ".self_attn.indexer.key_cache";
    } else if (corrupt == kWrongKind) {
      payload_kinds[1] = static_cast<uint8_t>(vllm::KvCachePayload::kPaged);
    } else if (corrupt == kNoGroupTable) {
      group_tables[2].clear();
    }

    vllm::MultiKvCacheIndex mk;
    mk.layer_names = &names;
    mk.group_ids = &group_ids;
    mk.layer_indices = &layer_indices;
    mk.payload_kinds = &payload_kinds;
    mk.payload_slots = &payload_slots;
    mk.group_block_tables = &group_tables;
    mk.group_block_table_cols = &group_cols;

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{ids,  pos,    am, gm, attn_kv,
                               gdn,  config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;

    std::string msg = "<the step RETURNED; this refusal is gone>";
    try {
      (void)vllm::ModelRegistry::Forward(*model, in);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  // The two strings that must be ABSENT from every one of them.
  const char* kPle = "qwen4_exp ple layout";
  const char* kEngine = "does not consume a cache set keyed by layer name";

  SUBCASE("a published name this model does not expect") {
    const std::string m = refusal(kWrongName);
    INFO("the wrong-name refusal said: ", m);
    CHECK(m.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(m.find("published no KV cache under") != std::string::npos);
    // The NAME, not only the shape of the complaint: the hook reports the string
    // it asked for, which is what routes the reader to the disagreement.
    CHECK(m.find("self_attn.indexer.k_cache") != std::string::npos);
    CHECK(m.find(kPle) == std::string::npos);
    CHECK(m.find(kEngine) == std::string::npos);
  }

  SUBCASE("a recurrent group published as a paged one") {
    const std::string m = refusal(kWrongKind);
    INFO("the wrong-kind refusal said: ", m);
    CHECK(m.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(m.find("model.layers.0.linear_attn") != std::string::npos);
    CHECK(m.find("RECURRENT") != std::string::npos);
    CHECK(m.find("PAGED") != std::string::npos);
    CHECK(m.find(kPle) == std::string::npos);
    CHECK(m.find(kEngine) == std::string::npos);
  }

  SUBCASE("the indexer group's block table was never gathered") {
    const std::string m = refusal(kNoGroupTable);
    INFO("the ungathered-table refusal said: ", m);
    CHECK(m.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(m.find("gathered no block table for published group 2") !=
          std::string::npos);
    // The DENOMINATOR, so a partial gather reads as partial: two of three.
    CHECK(m.find("2 of 3 published group(s) carry one") != std::string::npos);
    CHECK(m.find(kPle) == std::string::npos);
    CHECK(m.find(kEngine) == std::string::npos);
  }
}

// ─── A SECOND STEP: THE DECODE (W5k, #2031) ──────────────────────────────────
//
// Every case above serves ONE step. This one runs TWO through
// `ModelRegistry::Forward` over caches that PERSIST between them, which is the
// thing the row has been unable to do since W5c published the topology:
// `past_len > 0` was refused because the recurrent group publishes the PLE conv
// ring at the model dtype and the n-gram history on the device, while
// `RunQwen4ExpPleBlock` required an f32 ring and a host history.
//
// W5k settled which side was wrong by INSTALLING the lane oracle rather than
// reasoning from this tree's convention — transformers 5.16.0,
// `models/qwen4_exp/modeling_qwen4_exp.py` sha256 77fec77d…c459, the pin
// `scripts/gen-qwen4-exp-forward-goldens.py` asserts — and reading the running
// model. It types each cache slot from the tensor that first reaches it
// (`cache_utils.py:1019-1023`), so the ring carries the MODEL dtype (observed:
// `conv_states[1] dtype=torch.bfloat16` on a bf16 run) and the history carries
// `input_ids.long()` on `input_ids.device` (`:1070`, `:1089-1091`). The
// PUBLISHER was right twice and both requirements moved to the block.
//
// ─── WHAT THIS GATES, AND WHY NOT BY VALUE ───────────────────────────────────
// W5j MEASURED that this fixture cannot gate cache CONTENT: across two prompts,
// 0 of 128 indexer words and 0 of 192 paged K/V words moved while the logits
// moved 31.84, because layer-3 activations sit near 2^18 where one bf16 ULP is
// ~1024 and the store saturates. A rescaled fixture is owed. So nothing here
// asserts a cache VALUE.
//
// What it asserts instead is the one cross-step observable that CANNOT saturate:
// the n-gram history is int64 TOKEN IDS. Upstream keeps the last
// `ngram_size - 1` raw ids and rolls them every step — measured on the pinned
// oracle over four steps of the prompt [5,9,13,3,7,2]:
//
//     after prefill      ngram_history = [7, 2]      (the prompt's last two)
//     after decode 11    ngram_history = [2, 11]
//     after decode 4     ngram_history = [11, 4]
//     after decode 3     ngram_history = [4, 3]
//
// That is a FIFO whose contents are exact integers at any dtype, so the
// assertions below read the engine's own recurrent state and compare it to ids,
// not to a rounded activation. And the step-2 read's WRITER IS A PRIOR CALL,
// which is the property W5j's single-step cases could not have.
TEST_CASE("qwen4_exp: a SECOND step decodes on the engine's own persistent caches") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t kPleLayer = 1;   // ple_layer_ids = [2], one-based
  constexpr int64_t T1 = 6;          // the prefill
  constexpr int64_t kPage = 4;
  constexpr int64_t kCols = 3;       // T1 + 1 token spans 2 pages; 3 is slack
  constexpr int64_t kBlocks0 = 4;
  constexpr int64_t kBlocks2 = 5;
  const int64_t kIdxRows = kBlocks2 * kPage;

  // The recurrent state geometry, exactly as `MakeQwen4ExpKVCache` publishes it.
  const int64_t kConvDim = 2 * kKeyDim + kValueDim;
  const int64_t kGdnConvLen = kConvKernel - 1;
  const int64_t kSsmRow = kNumVHeads * kLinHeadDim * kLinHeadDim;
  const int64_t kPleStateLen = (kConvKernel - 1) * kNgramSize;  // DILATED: 9
  const int64_t kCtx = kNgramSize - 1;                          // 2

  const std::vector<int32_t> bt0{2, 0, 3};
  const std::vector<int32_t> bt2{1, 3, 4};
  const std::vector<int32_t> bt_recurrent{0};

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // ─── THE CACHES, ALLOCATED ONCE AND SHARED BY BOTH STEPS ──────────────────
  // This is the whole point: a per-call scratch is zeroed on entry, so a second
  // step over one would re-seed the n-gram history with EOS and re-zero the ring
  // and the model would decode as though each token were the first.
  std::vector<std::vector<unsigned char>> st_gdn_conv(kGdnLayers),
      st_ssm(kGdnLayers), st_ple_conv(kGdnLayers);
  std::vector<std::vector<int64_t>> st_ngram(kGdnLayers);
  std::vector<vllm::dense_attn::DBuf> b_gdn_conv, b_ssm, b_ple_conv, b_ngram;
  std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
  b_gdn_conv.reserve(kGdnLayers); b_ssm.reserve(kGdnLayers);
  b_ple_conv.reserve(kGdnLayers); b_ngram.reserve(kGdnLayers);
  for (int i = 0; i < kGdnLayers; ++i) {
    st_gdn_conv[i].assign(static_cast<size_t>(kConvDim * kGdnConvLen) * sizeof(float), 0);
    st_ssm[i].assign(static_cast<size_t>(kSsmRow) * sizeof(float), 0);
    st_ple_conv[i].assign(
        static_cast<size_t>(qwen4_exp_fixture::kStream * kPleStateLen) *
            vt::SizeOf(vllm::kQwen4ExpStreamDType), 0);
    st_ngram[i].assign(static_cast<size_t>(kCtx), 0);
    b_gdn_conv.emplace_back(d, DType::kF32,
                            std::vector<int64_t>{1, kConvDim, kGdnConvLen},
                            st_gdn_conv[i].data());
    b_ssm.emplace_back(d, DType::kF32,
                       std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                       st_ssm[i].data());
    // THE RING AT THE MODEL DTYPE and the HISTORY ON THIS STEP'S DEVICE — the
    // two the publisher declares and the two the block refused before W5k.
    b_ple_conv.emplace_back(d, vllm::kQwen4ExpStreamDType,
                            std::vector<int64_t>{1, qwen4_exp_fixture::kStream,
                                                 kPleStateLen},
                            st_ple_conv[i].data());
    b_ngram.emplace_back(d, DType::kI64, std::vector<int64_t>{1, kCtx},
                         st_ngram[i].data());
    gdn[static_cast<size_t>(i)].conv_state = b_gdn_conv.back().t();
    gdn[static_cast<size_t>(i)].ssm_state = b_ssm.back().t();
    // THE PUBLISHED ORDER: [gdn_conv, temporal, ple_conv, ngram]. Filled by
    // `GPUModelRunner::initialize_kv_cache` in production; filled here because
    // this case stands in for it.
    gdn[static_cast<size_t>(i)].states = {b_gdn_conv.back().t(), b_ssm.back().t(),
                                          b_ple_conv.back().t(), b_ngram.back().t()};
  }
  REQUIRE(gdn[kPleLayer].states.size() == 4);

  std::vector<uint16_t> kv(
      static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, kBlocks0, kPage, kKvHeads, kHeadDim}, kv.data());
  constexpr uint16_t kPoison = 0x3F80;
  std::vector<uint16_t> idx(static_cast<size_t>(kIdxRows * kIdxHeadDim), kPoison);
  vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                               idx.data());

  std::vector<vllm::PagedKvCache> attn_kv(2);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = kBlocks0;
  attn_kv[0].block_size = kPage;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;
  attn_kv[1].data = idx_b.t().data;
  attn_kv[1].dtype = DType::kBF16;
  attn_kv[1].num_blocks = kBlocks2;
  attn_kv[1].block_size = kPage;
  attn_kv[1].num_kv_heads = kIdxKvHeads;
  attn_kv[1].head_size = kIdxHeadDim;

  std::vector<std::string> names;
  std::vector<int32_t> group_ids, layer_indices, payload_slots;
  std::vector<uint8_t> payload_kinds;
  names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
  group_ids.push_back(0);
  layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
  payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
  payload_slots.push_back(0);
  for (int i = 0; i < kGdnLayers; ++i) {
    names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
    group_ids.push_back(1);
    layer_indices.push_back(i);
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
    payload_slots.push_back(i);
  }
  names.push_back("model.layers." + std::to_string(kQsaLayer) +
                  ".self_attn.indexer.k_cache");
  group_ids.push_back(2);
  layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
  payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
  payload_slots.push_back(1);

  std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, bt2};
  std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                  static_cast<int32_t>(kCols)};
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;
  mk.group_ids = &group_ids;
  mk.layer_indices = &layer_indices;
  mk.payload_kinds = &payload_kinds;
  mk.payload_slots = &payload_slots;
  mk.group_block_tables = &group_tables;
  mk.group_block_table_cols = &group_cols;

  // The physical slot a logical position lands on under group 0's table.
  const auto slot_of = [&](int64_t t) {
    return static_cast<int32_t>(bt0[static_cast<size_t>(t / kPage)] * kPage + t % kPage);
  };

  // ─── ONE STEP, OVER THE CACHES ABOVE ──────────────────────────────────────
  struct Step { std::vector<float> logits; int64_t vocab = 0; int64_t token = -1; };
  const auto run = [&](const std::vector<int32_t>& tok_ids, int64_t past_len) {
    const auto T = static_cast<int64_t>(tok_ids.size());
    std::vector<int32_t> pos(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      pos[static_cast<size_t>(t)] = static_cast<int32_t>(past_len + t);

    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.max_query_len = static_cast<int>(T);
    am.query_start_loc.assign({0, static_cast<int32_t>(T)});
    am.seq_lens.assign(1, static_cast<int32_t>(past_len + T));
    am.block_table_tensor = bt0;
    am.block_table_num_cols = static_cast<int>(kCols);
    am.max_seq_len = static_cast<int>(past_len + T);
    am.slot_mapping.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      am.slot_mapping[static_cast<size_t>(t)] = slot_of(past_len + t);

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = static_cast<int>(T);
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    if (past_len == 0) {
      gm.num_prefills = 1;
      gm.num_prefill_tokens = static_cast<int>(T);
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
      gm.prefill_state_indices = std::vector<int32_t>{0};
      // `prefill_has_initial_state == 0` IS upstream's
      // `has_previous_state(layer_idx, state_idx=2) == False`, which is the
      // branch that seeds the n-gram history with `eos_token_id` rather than
      // trusting the zero-filled cache (`modeling_qwen4_exp.py:1073-1076`).
      // ZERO IS A VALID TOKEN ID, so that seed is not decoration.
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    } else {
      // A PURE DECODE. Upstream populates `has_initial_state` only when there is
      // a prefill (`gdn_attn.py:389-405`), and the single-token update kernel is
      // selected instead of the conv program map.
      gm.num_decodes = 1;
      gm.num_decode_tokens = static_cast<int>(T);
    }

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{tok_ids, pos,    am, gm, attn_kv,
                               gdn,     config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;

    Step out;
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
    REQUIRE(fl.on_device());
    REQUIRE(fl.rows == 1);
    out.vocab = fl.vocab;
    out.logits.assign(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
    d.b.Copy(q, out.logits.data(), fl.device_tensor.data,
             out.logits.size() * sizeof(float));
    d.b.Synchronize(q);
    {
      std::vector<int64_t> t1(1, -1);
      vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
      vt::Tensor tt = tok_b.t();
      vt::GreedyArgmax(q, tt, fl.device_tensor);
      tok_b.Download(d, t1.data());
      out.token = t1[0];
    }
    return out;
  };

  // The prompt carries an EOS in the INTERIOR, which is a segment boundary in
  // the hashed n-gram construction, so the ids after it are not a straight ramp.
  std::vector<int32_t> prompt{5, 9, 13, static_cast<int32_t>(kEosTokenId), 7, 2};
  REQUIRE(static_cast<int64_t>(prompt.size()) == T1);
  for (int32_t v : prompt) REQUIRE(v < static_cast<int32_t>(kVocab));

  // ─── STEP 1: THE PREFILL ──────────────────────────────────────────────────
  Step s1;
  REQUIRE_NOTHROW(s1 = run(prompt, /*past_len=*/0));
  CHECK(s1.vocab == kVocab);
  for (float v : s1.logits) REQUIRE(std::isfinite(v));
  CHECK(s1.token >= 0);
  CHECK(s1.token < s1.vocab);
  MESSAGE("qwen4_exp STEP 1 (prefill, T=" << T1 << ") sampled token " << s1.token);

  // THE HISTORY THE PREFILL LEFT BEHIND is the prompt's last `ngram_size - 1`
  // ids — upstream's `[7, 2]` for this prompt. Read out of the ENGINE's own
  // recurrent state, and compared to the INPUT IDS rather than to a value read
  // back from the cache before the run, so a cache that was never written fails.
  std::vector<int64_t> hist(static_cast<size_t>(kCtx), -1);
  b_ngram[kPleLayer].Download(d, hist.data());
  for (int64_t i = 0; i < kCtx; ++i) {
    INFO("history slot ", i);
    CHECK(hist[static_cast<size_t>(i)] ==
          static_cast<int64_t>(prompt[static_cast<size_t>(T1 - kCtx + i)]));
  }

  // NOTHING WAS WRITTEN TO THE OTHER LAYERS' PLE SLOTS. Only layer 1 carries a
  // PLE block, and the uniform recurrent group gives all three linear layers the
  // same state set; a hook that indexed the PLE cache by DECODER layer index
  // instead of by rank among linear layers would write layer 2's or layer 3's.
  for (int i = 0; i < kGdnLayers; ++i) {
    if (i == kPleLayer) continue;
    std::vector<int64_t> other(static_cast<size_t>(kCtx), -1);
    b_ngram[static_cast<size_t>(i)].Download(d, other.data());
    INFO("linear layer ", i, " must own no PLE history");
    CHECK(other == std::vector<int64_t>(static_cast<size_t>(kCtx), 0));
  }

  // ─── STEP 2: THE DECODE — past_len > 0, ON THE SAME BUFFERS ───────────────
  // This is the call every wave since W5c has been refused at.
  const std::vector<int32_t> next{static_cast<int32_t>(s1.token)};
  Step s2;
  REQUIRE_NOTHROW(s2 = run(next, /*past_len=*/T1));
  CHECK(s2.vocab == kVocab);
  for (float v : s2.logits) REQUIRE(std::isfinite(v));
  CHECK(s2.token >= 0);
  CHECK(s2.token < s2.vocab);
  MESSAGE("qwen4_exp STEP 2 (decode, past_len=" << T1 << ", token " << s1.token
          << ") sampled token " << s2.token);

  // THE FIFO ROLLED, AND THAT IS THE CROSS-STEP GATE. Upstream's history after a
  // decode is `[previous_last, this_token]` — measured on the pinned oracle as
  // `[7,2] -> [2,11]` after decoding 11. The first slot must now hold what the
  // SECOND slot held before the step, which no single-step run can produce and
  // which a re-seeded scratch would render as `[eos, token]` instead.
  std::vector<int64_t> hist2(static_cast<size_t>(kCtx), -1);
  b_ngram[kPleLayer].Download(d, hist2.data());
  CHECK(hist2[static_cast<size_t>(kCtx - 1)] == s1.token);
  for (int64_t i = 0; i + 1 < kCtx; ++i) {
    INFO("rolled slot ", i);
    CHECK(hist2[static_cast<size_t>(i)] == hist[static_cast<size_t>(i + 1)]);
  }
  // And it is NOT the seed a per-call scratch would have produced.
  const std::vector<int64_t> seeded{static_cast<int64_t>(kEosTokenId), s1.token};
  CHECK(hist2 != seeded);

  // THE LOGITS MOVED. Step 2 sees a one-token query at position T1 against six
  // cached keys; step 1 saw six queries. Identical rows would mean the second
  // call recomputed the first.
  REQUIRE(s1.logits.size() == s2.logits.size());
  CHECK(std::memcmp(s1.logits.data(), s2.logits.data(),
                    s1.logits.size() * sizeof(float)) != 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// W5p (#2031) — A Q8_0 MIX WEIGHT REACHES THE PRODUCTION ENTRY POINT.
//
// The released `unsloth/Qwen3.8-Flash-Next-GGUF` stores all 194 hyper-connection
// mix weights as Q8_0 and loaded cleanly into this tree; the first prefill then
// died inside `vt::Qwen4ExpGatedResidual`'s own validation, because every arm of
// the fixture and every op-level case wrote those tensors F32. This case is the
// arm that was missing: the SAME `ModelRegistry::Forward` the case above drives,
// over a file whose `hc_*_down` and `hc_*_inject` are block-typed.
//
// IT IS A REACHABILITY GATE, NOT A NUMERIC ONE. The fixture's weights are a
// deterministic pattern rather than a checkpoint, so there is no reference token
// stream; the arithmetic of the quantized route is gated at op level by
// `test_qwen4_exp_hc_device.cpp` against an independent double reference, and
// the loader half by `test_qwen4_exp_gguf_weights.cpp`. What is asserted here is
// that a block-typed mix weight survives the whole path from the file to a
// sampled token through the entry point AGENTS.md names.
//
// `hc_*_up` stays F32 in the file because ggml forbids a quantized tensor whose
// fastest dim is not a whole 32-element block and this miniature's low-rank is
// 8. The released config's low-rank is 320.
TEST_CASE("qwen4_exp: ModelRegistry::Forward runs a Q8_0 hyper-connection mix weight") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  FixtureOpts opts;
  opts.hc_mix_q8_0 = true;
  const gguf_test::TempFile f(BuildFixture(opts));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  // THE FILE, ASSERTED BEFORE THE MODEL. An option that quietly wrote F32 would
  // turn this whole case into a second run of the case above.
  REQUIRE(g.Get("blk.0.hc_attn_down.weight").ggml_type == 8);
  REQUIRE(g.Get("blk.0.hc_attn_inject.weight").ggml_type == 8);
  REQUIRE(g.Get("blk.0.hc_ffn_down.weight").ggml_type == 8);
  REQUIRE(g.Get("output_hc_down.weight").ggml_type == 8);

  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, 1, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos, am, gm, attn_kv, gdn, config, q,
                             logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // THE RED THIS CASE WAS WRITTEN AGAINST, in its own words: before this wave
  // `vt::Qwen4ExpGatedResidual` threw
  //   `qwen4_exp_gated_residual: input_mix_weight_down must be float
  //    (f32/bf16 for outputs)`
  // here, on the FIRST hyper-connection of layer 0.
  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  CHECK(fl.rows == 1);
  CHECK(fl.vocab == kVocab);
  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  REQUIRE(fl.device_tensor.dtype == DType::kF32);

  std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
  d.b.Copy(q, host.data(), fl.device_tensor.data, host.size() * sizeof(float));
  d.b.Synchronize(q);

  // FINITENESS BEFORE ANY OTHER READING (#449): a max-fold over `std::max`
  // returns the non-NaN operand, so an all-NaN row reads as a match to any
  // tolerance and still argmaxes to an index in range.
  int finite = 0;
  for (float v : host) finite += std::isfinite(v) ? 1 : 0;
  REQUIRE(finite == static_cast<int>(host.size()));
  float lo = host[0];
  float hi = host[0];
  for (float v : host) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  // A CONSTANT row is what a tower contributing nothing produces.
  CHECK(hi > lo);

  std::vector<int64_t> tok(1, -1);
  {
    vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
    vt::Tensor tt = tok_b.t();
    vt::GreedyArgmax(q, tt, fl.device_tensor);
    tok_b.Download(d, tok.data());
  }
  MESSAGE("qwen4_exp Q8_0-mix sampled token id: " << tok[0] << " of " << fl.vocab
                                                  << " (logit range [" << lo
                                                  << ", " << hi << "])");
  CHECK(tok[0] >= 0);
  CHECK(tok[0] < fl.vocab);
  CHECK(host[static_cast<size_t>(tok[0])] == hi);

  // AND IT DEPENDS ON THE PROMPT. Every assertion above is satisfied by a hook
  // that never reads `token_ids` — MUT-REACH measured exactly that on the
  // sibling case. Fresh zeroed caches leave the prompt as the only input that
  // moved.
  std::vector<int32_t> ids2(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids2[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : (t + 1) * 4 + 1);
    REQUIRE(ids2[static_cast<size_t>(t)] < static_cast<int32_t>(kVocab));
  }
  REQUIRE(ids2 != ids);

  std::vector<std::vector<float>> ssm2(3), conv2(3);
  std::vector<vllm::dense_attn::DBuf> ssm2_b, conv2_b;
  std::vector<vllm::GdnStateCache> gdn2(3);
  ssm2_b.reserve(3);
  conv2_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm2[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv2[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm2_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm2[i].data());
    conv2_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, conv_dim, conv_len},
                         conv2[i].data());
    gdn2[static_cast<size_t>(i)].ssm_state = ssm2_b.back().t();
    gdn2[static_cast<size_t>(i)].conv_state = conv2_b.back().t();
  }
  std::vector<uint16_t> kv2(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv2_b(d, DType::kBF16,
                               {2, 1, T, kKvHeads, kHeadDim}, kv2.data());
  std::vector<vllm::PagedKvCache> attn_kv2(1);
  attn_kv2[0] = attn_kv[0];
  attn_kv2[0].data = kv2_b.t().data;

  vllm::ModelForwardInput in2{ids2, pos, am, gm, attn_kv2, gdn2, config, q,
                              logits_indices};
  in2.num_reqs = 1;
  in2.gdn_state_slots = 1;

  vllm::ForwardLogits fl2;
  REQUIRE_NOTHROW(fl2 = vllm::ModelRegistry::Forward(*model, in2));
  REQUIRE(fl2.on_device());
  std::vector<float> host2(static_cast<size_t>(fl2.rows * fl2.vocab), 0.0F);
  d.b.Copy(q, host2.data(), fl2.device_tensor.data, host2.size() * sizeof(float));
  d.b.Synchronize(q);
  const double moved = vllm_test::MaxAbsDiff(host, host2);
  MESSAGE("qwen4_exp Q8_0-mix second-prompt logit movement: " << moved);
  CHECK(moved > 0.0);
}
