// MODEL-DSV4-EXL3 W1c/W2 — a LOADED EXL3 checkpoint reaches the forward, and the
// trellis tower computes the function its dequantized equivalent does.
//
// WHAT CHANGED, AND WHY IT IS THE POINT. W2 claimed the loaded tower was
// REACHABLE and gated the claim on a `DeepseekV4Weights` this suite built BY
// HAND, setting `has_host_weights = true` itself at five sites. The loader never
// set that flag and never wrote `host`, so `has_exl3_weights && has_host_weights`
// could not come out of a load at all: an end-to-end `vllm-server` probe over a
// real rank-sliced checkpoint loaded, printed its residency line, and then killed
// the engine on the first completion with the `kHostPending` refusal. Zero tokens
// (#1923). Every mutation the W2 reviews ran was therefore evaluated on a struct
// no loader can produce — `.agents/reachability.md`'s documented failure, in its
// exact shape.
//
// So this suite no longer constructs weights. It writes a hermetic rank-sliced
// EXL3 checkpoint to disk (`dsv4_exl3_fixture.h`, shared with the loader suite),
// loads it through `vllm::LoadDeepseekV4ForCausalLMWeights` — the entry
// `deepseek_v4_registry.cpp` routes `ModelRegistry::Load` to — and runs
// `vllm::DeepseekV4Model::Forward` over the result. Deleting the loader's
// carried-tower materialization, or its `has_host_weights = true`, reds every
// case below.
//
// It then asserts two things a mutation can tell apart:
//
//   1. EQUIVALENCE. The EXL3 arm's logits match a DENSE forward whose expert
//      weights are `vt::Exl3DequantLinear` of the SAME trellis, over the SAME
//      loaded carried tower. That is the algebraic identity the format rests on
//      (`exl3.py:183-214` vs `:227-237`): the two Hadamards may ride the
//      activations or the weights.
//   2. DISCRIMINATION. The EXL3 arm's logits are FAR from a dense forward over
//      unrelated random expert weights, which is what an arm that missed the
//      EXL3 dispatch would compute.
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
#include <cstdlib>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

#include "dsv4_exl3_fixture.h"

using dsv4_exl3_fixture::BuildFixture;
using dsv4_exl3_fixture::FixtureOptions;
using vllm::DeepseekV4Exl3Linear;
using vllm::DeepseekV4HostWeights;
using vllm::DeepseekV4LayerHostWeights;
using vllm::DeepseekV4Weights;

namespace {

// The shape of the model the forward fixture describes: two layers so the MoE
// runs more than once, layer 0 hash-routed and layer 1 carrying the DSA
// compressor, so the load has to materialize every carried family the host
// forward reads. `topk = 2` over the fixture's two routed experts keeps both
// live on every token.
//
// WHY LAYER 1 IS `cr == 128` AND NOT `cr == 4` (#1970). The loader derives the
// compressor width from `coff = 1 + (compress_ratio == 4)`
// (`vllm/models/deepseek_v4/compressor.py:247-248`), which at `cr == 4` is 2 —
// so a `cr == 4` layer the host forward can RUN would have to be written at the
// collapsed, undoubled width, and that is a checkpoint upstream cannot emit and
// this loader refuses. At `cr == 128` `coff` is 1, the derived width and the
// collapsed one are the same value, and these cases keep driving an EXL3-loaded
// compressor layer end to end through the production forward.
//
// WHAT THAT COSTS, stated rather than implied: no case here runs an EXL3-loaded
// INDEXER forward, because the indexer exists only at `cr == 4`
// (`attention.py:274`) where the real artifact's geometry is the one the forward
// refuses. The indexer maths itself stays gated at the synthetic geometry by
// `test_deepseek_v4_forward.cpp` and `test_deepseek_v4_dsa.cpp`, which do not
// load through this arm.
FixtureOptions ForwardFixtureOptions() {
  FixtureOptions opt;
  opt.layers = 2;
  opt.num_hash_layers = 1;
  opt.topk = 2;
  opt.compress_ratios = {0, 128};
  opt.index_n_heads = 2;
  opt.index_head_dim = 4;
  opt.index_topk = 3;
  return opt;
}

// The same model with layer 1 at `cr == 4` and the DSA family written at the
// width the REAL DeepSeek-V4-Flash artifact stores: `coff` is 2 there, so the
// compressor + indexer families are doubled and `indexer.wq_b`'s K is
// `q_lora_rank`. This LOADS and the forward REFUSES it.
FixtureOptions RealDsaFixtureOptions() {
  FixtureOptions opt = ForwardFixtureOptions();
  opt.compress_ratios = {0, 4};
  opt.real_dsa_geometry = true;
  return opt;
}

struct Rng {
  uint32_t s = 0x243F6A88u;
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

// A copy of the LOADED carried tower with a dense routed-expert tower attached:
// either the dequantized trellis (`from_trellis`) or unrelated random weights.
// `has_exl3_weights` is left FALSE on the copy, so the same production entry
// point takes its dense arm over the identical non-expert weights — the only
// difference between the arms is where the routed experts came from.
DeepseekV4Weights DenseCopy(const DeepseekV4Weights& src, bool from_trellis) {
  DeepseekV4Weights out;
  out.params = src.params;
  out.host = src.host;
  out.has_host_weights = src.has_host_weights;
  const int64_t H = src.params.hidden_size;
  const int64_t mi = src.params.moe_intermediate_size;
  const int64_t ne = src.params.n_routed_experts;
  Rng rng;
  rng.s = 0x7F4A7C15u;
  for (int64_t l = 0; l < src.params.num_hidden_layers; ++l) {
    DeepseekV4LayerHostWeights& L = out.host.layers[static_cast<size_t>(l)];
    if (!from_trellis) {
      L.exp_w1 = Rand(rng, ne * mi * H, 0.3f);
      L.exp_w3 = Rand(rng, ne * mi * H, 0.3f);
      L.exp_w2 = Rand(rng, ne * H * mi, 0.3f);
      continue;
    }
    L.exp_w1.assign(static_cast<size_t>(ne * mi * H), 0.0f);
    L.exp_w3.assign(static_cast<size_t>(ne * mi * H), 0.0f);
    L.exp_w2.assign(static_cast<size_t>(ne * H * mi), 0.0f);
    const auto& experts = src.exl3.layers[static_cast<size_t>(l)].experts;
    for (int64_t e = 0; e < ne; ++e) {
      const vllm::DeepseekV4Exl3Expert& xe = experts[static_cast<size_t>(e)];
      DequantInto(xe.w1, &L.exp_w1[static_cast<size_t>(e * mi * H)]);
      DequantInto(xe.w3, &L.exp_w3[static_cast<size_t>(e * mi * H)]);
      DequantInto(xe.w2, &L.exp_w2[static_cast<size_t>(e * H * mi)]);
    }
  }
  return out;
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

// One line of `RequireDsaGeometryOrRefuse`'s mismatch list, rendered exactly as
// the refusal renders it (`deepseek_v4.cpp`, the `want` lambda).
//
// WHY THE COUNTS ARE ASSERTED BOUND TO A TENSOR NAME AND NOT ON THEIR OWN. At
// this fixture's dimensions several of the counts COINCIDE: `compress_ratio *
// head_dim`, `index_n_heads * index_head_dim * hidden_size` and `2 *
// index_head_dim * hidden_size` are all 2048, and `head_dim - 1` and
// `index_n_heads * hidden_size - 1` are both 511. A bare `Mentions(msg,
// "2048")` therefore does not say which tensor reported it, and the round-2
// fresh review measured that directly: a mutation that made `compressor.ape`'s
// expected count wrong left both suites GREEN, because another tensor's line
// still carried the number. Binding the count to the name closes that.
std::string MismatchLine(const char* tensor, const char* indexed_as, int64_t indexed,
                         int64_t carried) {
  return "attn." + std::string(tensor) + ": this forward indexes it as " + indexed_as +
         " = " + std::to_string(indexed) + " elements, the checkpoint carries " +
         std::to_string(carried);
}

const std::vector<int32_t> kTokens = {3, 7, 1};
const std::vector<int32_t> kPositions = {0, 1, 2};

}  // namespace

TEST_CASE("dsv4 exl3 W1c: a LOADED checkpoint reaches the forward and emits logits") {
  auto f = BuildFixture(ForwardFixtureOptions());
  const DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  // THE DEFECT #1923 NAMES, stated as an assertion. The loader sets BOTH flags
  // on the same arm; before W1c it set only the first, and the forward below
  // could not run at all.
  REQUIRE(w.has_exl3_weights);
  REQUIRE(w.has_host_weights);
  // ...and the tower it set the flag for is actually populated. A flag set
  // beside an empty tower is the "fake the flag" failure the row's dispatch
  // forbade, and it is what a mutation that deletes the materialization but
  // keeps the assignment would produce.
  REQUIRE(w.host.layers.size() == static_cast<size_t>(w.params.num_hidden_layers));
  CHECK(!w.host.embed.empty());
  CHECK(!w.host.lm_head.empty());
  CHECK(!w.host.layers[0].wq_a.empty());
  CHECK(!w.host.layers[0].shared_w1.empty());
  // The routed experts are the TRELLIS tower and nothing else, so the host
  // routed slots stay empty by design (`## W1c design` W1c-2).
  CHECK(w.host.layers[0].exp_w1.empty());

  QueueGuard g;
  const vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;

  // (a) the EXL3 arm, over a LOADED checkpoint, through the production entry.
  const std::vector<float> exl3_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, w, g.q, {});
  REQUIRE(AllFinite(exl3_logits));
  CHECK(static_cast<int64_t>(exl3_logits.size()) ==
        static_cast<int64_t>(kTokens.size()) * w.params.vocab_size);

  // (b) the DEQUANTIZED-weight dense arm: same trellis bits, other basis, and
  //     the SAME loaded carried tower.
  const DeepseekV4Weights wd = DenseCopy(w, /*from_trellis=*/true);
  const std::vector<float> deq_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wd, g.q, {});
  REQUIRE(AllFinite(deq_logits));

  // (c) the UNRELATED dense arm: what a forward that missed the EXL3 dispatch
  //     would compute if it had any host experts at all.
  const DeepseekV4Weights wr = DenseCopy(w, /*from_trellis=*/false);
  const std::vector<float> rand_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wr, g.q, {});
  REQUIRE(AllFinite(rand_logits));

  const double equiv = RelRms(exl3_logits, deq_logits);
  const double discrim = RelRms(exl3_logits, rand_logits);
  MESSAGE("exl3 vs dequantized-dense rel_rms=", equiv,
          "   exl3 vs unrelated-dense rel_rms=", discrim);
  CHECK(equiv <= 2.0e-2);
  CHECK(discrim > 1.0e-1);
}

TEST_CASE("dsv4 exl3 W1c: the generic host-tower refusal is the one that is REACHABLE") {
  // #1923's second finding, settled. The EXL3-specific `has_host_weights`
  // refusal that used to sit in `DeepseekV4ForwardExl3` named this row and was
  // unreachable on the default path — the runner's default `gather` routes to
  // `ForwardDevice`, whose generic check fires first — and W1c makes the state
  // it guarded unreachable from ANY load, because the one arm that sets
  // `has_exl3_weights` now sets `has_host_weights` in the same function. It is
  // deleted rather than decorated (`## W1c design` W1c-5).
  //
  // The refusal that IS reachable is the generic one, and this is the load that
  // reaches it: the DENSE DeepSeek-V4 safetensors arm still only ACCOUNTS for
  // its tensors (the standing W2b residual of `deepseek-v4-flash.md`), so it
  // returns `has_host_weights == false` and the forward refuses BY NAME. That
  // is the exact shape the EXL3 arm was in before this wave.
  FixtureOptions opt;
  opt.quant_method = "fp8";        // NOT exl3: the pre-existing dense arm
  opt.dense_routed_experts = true; // dense NVFP4 experts, no rank shards
  auto f = BuildFixture(opt);
  const DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(!w.has_exl3_weights);
  REQUIRE(!w.has_host_weights);

  QueueGuard g;
  const vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;
  const std::string msg = dsv4_exl3_fixture::ThrowMessage([&] {
    (void)vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, w, g.q, {});
  });
  CAPTURE(msg);
  CHECK(dsv4_exl3_fixture::Mentions(msg, "host-float weight tower"));
}

TEST_CASE("dsv4 exl3 W2d: VT_DSV4_EXL3_FUSED_MOE parses like the row's other knob") {
  // The parse is factored into the header precisely so it is gateable without
  // mutating the environment (house shape: `AsyncRunnerFlagIsOn`). This is the
  // NARROWER rule the row already uses for `VT_DSV4_EXL3_HOST_BUDGET`, not the
  // general flag rule, and pinning it here is what stops `false` or `off` from
  // silently leaving the fused arm on while a bisecting operator believes it is
  // measuring the loop.
  using vllm::Dsv4Exl3FusedMoeFlagIsOn;
  CHECK(Dsv4Exl3FusedMoeFlagIsOn(nullptr));  // unset: the fused arm
  CHECK(Dsv4Exl3FusedMoeFlagIsOn("1"));
  CHECK(Dsv4Exl3FusedMoeFlagIsOn(""));
  CHECK(Dsv4Exl3FusedMoeFlagIsOn("on"));
  CHECK(Dsv4Exl3FusedMoeFlagIsOn("false"));  // NOT a disable, and the doc says so
  CHECK(Dsv4Exl3FusedMoeFlagIsOn(" 0"));     // leading space, not a '0' first char
  CHECK(Dsv4Exl3FusedMoeFlagIsOn("10"));
  CHECK_FALSE(Dsv4Exl3FusedMoeFlagIsOn("0"));
  CHECK_FALSE(Dsv4Exl3FusedMoeFlagIsOn("00"));
  CHECK_FALSE(Dsv4Exl3FusedMoeFlagIsOn("0abc"));
}

TEST_CASE("dsv4 exl3 W2d: the FUSED MoE op is what the LOADED forward dispatches") {
  // WHY A COUNTER AND NOT A NUMBER COMPARISON. The fused arm and the per-expert
  // loop compute the same algebra, so deleting the fused call site leaves the
  // LOGITS right — the loop picks the work up, which is what makes it a genuine
  // tail path rather than dead code. A value gate therefore cannot see the
  // dispatch at all. `OpProviderStats::selections` can: it is the positive
  // signal `include/vt/op_provider.h` exists for, and deleting the
  // `Exl3FusedMoePass` call in `MoeBlock` takes `kExl3MoeMlp` to zero and
  // `kExl3Gemm` to 36 in the same run.
  auto f = BuildFixture(ForwardFixtureOptions());
  const DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);
  REQUIRE(w.has_host_weights);

  QueueGuard g;
  const vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;

  vt::EnableOpProviderCallStats(true);
  vt::ResetOpProviderStats(vt::OpId::kExl3MoeMlp, vt::DeviceType::kCPU);
  vt::ResetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU);
  const std::vector<float> exl3_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, w, g.q, {});
  const unsigned long long fused =
      vt::GetOpProviderStats(vt::OpId::kExl3MoeMlp, vt::DeviceType::kCPU).selections;
  const unsigned long long per_expert =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;
  vt::EnableOpProviderCallStats(false);
  REQUIRE(AllFinite(exl3_logits));

  // The suite is registered TWICE in ctest, once plain and once with
  // `VT_DSV4_EXL3_FUSED_MOE=0`, so both arms are gated by the same case and the
  // flag's rollback is measured rather than asserted. The predicate is the one
  // the production getter uses.
  const bool fused_arm = vllm::Dsv4Exl3FusedMoeFlagIsOn(std::getenv("VT_DSV4_EXL3_FUSED_MOE"));
  if (fused_arm) {
    // ONE call per MoE layer, and the per-expert GEMM is not reached AT ALL:
    // every expert here holds at most 6 assignments, far under the 128-row cut,
    // so the fused arm takes all of them.
    CHECK(fused == 2);
    CHECK(per_expert == 0);
  } else {
    // The rollback: 3 tokens x 2 experts x 3 projections x 2 layers.
    CHECK(fused == 0);
    CHECK(per_expert == 36);
  }

  // And whichever arm ran, the answer still tracks the dequantized-weight dense
  // tower at the bound this file's header derives, so the rollback is a rollback
  // and not a different model.
  const DeepseekV4Weights wd = DenseCopy(w, /*from_trellis=*/true);
  const std::vector<float> deq_logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wd, g.q, {});
  REQUIRE(AllFinite(deq_logits));
  const double equiv = RelRms(exl3_logits, deq_logits);
  MESSAGE("arm=", std::string(fused_arm ? "fused" : "loop"), "  fused_calls=", fused,
          "  per_expert_calls=", per_expert, "  vs dequantized-dense rel_rms=", equiv);
  CHECK(equiv <= 2.0e-2);
}

TEST_CASE("dsv4 exl3 #1970: the REAL DSA geometry LOADS and the FORWARD refuses by name") {
  // OPTION C of `.agents/specs/dsv4-dsa-geometry.md` (#1961), gated end to end:
  // the loader materializes every DSA tensor at the width the REAL
  // DeepSeek-V4-Flash artifact stores, and `AttentionBlock` refuses BY NAME
  // instead of indexing one at a width it does not have.
  //
  // WHY IT ENTERS THROUGH THE LOADER AND THE MODEL, AND NOT THROUGH THE TYPE.
  // `.agents/reachability.md`: a test that constructs `DeepseekV4Weights` by
  // hand proves the check works and NEVER that anything reaches it. This row has
  // already paid for that once — W2's reachability claim was gated on a struct
  // no loader could produce, and the real `vllm-server` load then generated zero
  // tokens (#1923). So this drives `vllm::LoadDeepseekV4ForCausalLMWeights`, the
  // entry `deepseek_v4_registry.cpp` routes `ModelRegistry::Load` to, and
  // `vllm::DeepseekV4Model::Forward`, what `ForwardDeepseekV4ForCausalLM` calls.
  auto f = BuildFixture(RealDsaFixtureOptions());

  // 1. IT LOADS. Before #1970 `RequireShape` refused four tensors here, and on
  //    the real artifact that is 41 of 43 layers — so the EXL3 tower at real
  //    scale, MoE, MTP and W2 residency were all unreachable behind a geometry
  //    none of them use.
  const DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);
  REQUIRE(w.has_host_weights);

  // ...and it materialized the REAL widths, not the collapsed ones. A loader
  // that "accepted" by silently truncating to `head_dim` would pass the refusal
  // check below for the wrong reason, so the widths are asserted directly.
  const int64_t hd = w.params.head_dim;
  const int64_t H = w.params.hidden_size;
  const int64_t cr = w.params.compress_ratio(1);
  const int64_t inh = w.params.index_n_heads;
  const int64_t ihd = w.params.index_head_dim;
  REQUIRE(cr == 4);
  const DeepseekV4LayerHostWeights& L1 = w.host.layers[1];
  CHECK(L1.comp_ape.size() == static_cast<size_t>(cr * 2 * hd));
  CHECK(L1.comp_wgate.size() == static_cast<size_t>(2 * hd * H));
  CHECK(L1.idx_wk.size() == static_cast<size_t>(2 * ihd * H));
  CHECK(L1.idx_wq.size() == static_cast<size_t>(inh * ihd * w.params.q_lora_rank));
  // The two upstream does NOT widen (`compressor.py:288` is
  // `RMSNorm(self.head_dim, self.rms_norm_eps)`, the file's only `RMSNorm(`;
  // `weights_proj` is `[n_head, hidden_size]`).
  CHECK(L1.comp_norm_weight.size() == static_cast<size_t>(hd));
  CHECK(L1.idx_wproj.size() == static_cast<size_t>(inh * H));

  // 2. AND THE FORWARD REFUSES BY NAME. Be exact about what that is worth:
  //    `Gemm`'s host arm is a `MatVec` whose size assertion is UNCONDITIONAL
  //    (`deepseek_v4.cpp:413`, a `VT_CHECK` throw and not an `assert`), so
  //    without this the wide `comp_wgate` does NOT emit a plausible wrong
  //    number — it throws `vt: MatVec weight size mismatch at
  //    deepseek_v4.cpp:413`, naming no tensor, no layer and nothing missing.
  //    What this case gates is the DIAGNOSTIC: that the refusal arrives instead,
  //    and carries the layer, every mismatched tensor, both counts and the
  //    missing capability. The `!msg.empty()` assertion below therefore does not
  //    on its own separate the two, which is why every `Mentions` check that
  //    follows it is part of the gate and not decoration.
  QueueGuard g;
  const vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;
  const std::string msg = dsv4_exl3_fixture::ThrowMessage([&] {
    (void)vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, w, g.q, {});
  });
  CAPTURE(msg);
  REQUIRE(!msg.empty());  // a forward that RETURNS here mis-indexed and did not say so

  // The AFFECTED LAYER, by number.
  CHECK(dsv4_exl3_fixture::Mentions(msg, "layer 1"));

  // EVERY tensor whose width it cannot index, each with both counts. All four
  // are named in ONE message on purpose: a refusal that stopped at the first
  // mismatch would make the other three checks unfalsifiable, because deleting
  // any one of them would still red on the first.
  CHECK(dsv4_exl3_fixture::Mentions(msg, "compressor.ape"));
  CHECK(dsv4_exl3_fixture::Mentions(msg, "compressor.wgate.weight"));
  CHECK(dsv4_exl3_fixture::Mentions(msg, "indexer.compressor.wkv.weight"));
  CHECK(dsv4_exl3_fixture::Mentions(msg, "indexer.wq_b"));
  // Both counts, BOUND to the tensor that reported them. Asserting the numbers
  // on their own does not gate this: several coincide at these dimensions (see
  // `MismatchLine`), so a wrong count on one tensor is still found on another's
  // line. `indexer.wq_b` gets its counts here too — it had none before, and its
  // indexed count is one of the colliding 2048s.
  CHECK(dsv4_exl3_fixture::Mentions(
      msg, MismatchLine("compressor.ape", "[compress_ratio, head_dim]", cr * hd,
                        cr * 2 * hd)));
  CHECK(dsv4_exl3_fixture::Mentions(
      msg, MismatchLine("compressor.wgate.weight", "[head_dim, hidden_size]", hd * H,
                        2 * hd * H)));
  CHECK(dsv4_exl3_fixture::Mentions(
      msg, MismatchLine("indexer.compressor.wkv.weight", "[index_head_dim, hidden_size]",
                        ihd * H, 2 * ihd * H)));
  CHECK(dsv4_exl3_fixture::Mentions(
      msg, MismatchLine("indexer.wq_b", "[index_n_heads*index_head_dim, hidden_size]",
                        inh * ihd * H, inh * ihd * w.params.q_lora_rank)));

  // The MISSING CAPABILITY, named — the AGENTS.md rule this case exists for.
  CHECK(dsv4_exl3_fixture::Mentions(msg, "coff"));
  CHECK(dsv4_exl3_fixture::Mentions(msg, "compressor.py:247-248"));
  CHECK(dsv4_exl3_fixture::Mentions(msg, "q_lora_rank"));
  CHECK(dsv4_exl3_fixture::Mentions(msg, "1970"));

  // 3. AND A COMPRESSOR LAYER THE FORWARD CAN INDEX STILL RUNS. The refusal is
  //    keyed on the width the forward indexes, not on the weight SOURCE — which
  //    is the mistake `dsa_dense = (be.gguf != nullptr)` makes (#1964). Without
  //    this case a check that simply refused every EXL3 compressor layer would
  //    pass everything above. `ForwardFixtureOptions()` is `cr == 128`, where
  //    `coff` is 1 and the loaded width IS the indexed one.
  auto fc = BuildFixture(ForwardFixtureOptions());
  const DeepseekV4Weights wc =
      vllm::LoadDeepseekV4ForCausalLMWeights(fc->shards, fc->config);
  const std::vector<float> logits =
      vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wc, g.q, {});
  CHECK(AllFinite(logits));

  // 4. THE TWO CHECKS THE LOADER MAKES UNREACHABLE ARE STILL FALSIFIABLE.
  //    `comp_norm_weight` and `idx_wproj` are the two DSA slots upstream does
  //    NOT widen (`compressor.py:288`; `weights_proj` is `[n_head, hidden_size]`),
  //    so the loader requires them at exactly the width the forward indexes and
  //    NO checkpoint can reach their checks. Deleting either one leaves both
  //    suites green, which would leave "all six are load-bearing" as an
  //    impression the gate does not support.
  //
  //    So they are gated by MUTATING a tower the production loader produced,
  //    rather than by a checkpoint. Be exact about what that proves and what it
  //    does not: it proves the two checks FIRE and are named in the message, and
  //    it proves nothing about reachability, because the state it constructs is
  //    one the loader cannot emit. They are defensive checks against a future
  //    loader/forward disagreement, and this is the falsifiability half only.
  //
  //    They are mutated ONE AT A TIME, in separate forwards. Mutating both at
  //    once and asserting on one message cannot separate them here: `head_dim -
  //    1` and `index_n_heads * hidden_size - 1` are the SAME number at this
  //    fixture (511), so the two count assertions were one assertion written
  //    twice. One tensor per message makes each count unambiguous.
  auto RefusalAfter = [&](void (*mutate)(DeepseekV4LayerHostWeights&)) {
    DeepseekV4Weights wm = w;  // the REAL-geometry tower, which already refuses
    mutate(wm.host.layers[1]);
    const std::string m = dsv4_exl3_fixture::ThrowMessage([&] {
      (void)vllm::DeepseekV4Model::Forward(kTokens, kPositions, meta, kv, wm, g.q, {});
    });
    REQUIRE(!m.empty());
    return m;
  };

  const std::string nmsg =
      RefusalAfter([](DeepseekV4LayerHostWeights& L) { L.comp_norm_weight.pop_back(); });
  CAPTURE(nmsg);
  CHECK(dsv4_exl3_fixture::Mentions(nmsg, "compressor.norm.weight"));
  CHECK(dsv4_exl3_fixture::Mentions(
      nmsg, MismatchLine("compressor.norm.weight", "[head_dim]", hd, hd - 1)));

  const std::string pmsg =
      RefusalAfter([](DeepseekV4LayerHostWeights& L) { L.idx_wproj.pop_back(); });
  CAPTURE(pmsg);
  CHECK(dsv4_exl3_fixture::Mentions(pmsg, "weights_proj.weight"));
  CHECK(dsv4_exl3_fixture::Mentions(
      pmsg, MismatchLine("indexer.weights_proj.weight", "[index_n_heads, hidden_size]",
                         inh * H, inh * H - 1)));
  // ...and each mutation names ONLY its own tensor, which is what makes the two
  // checks separately falsifiable rather than jointly.
  CHECK(!dsv4_exl3_fixture::Mentions(nmsg, "weights_proj.weight"));
  CHECK(!dsv4_exl3_fixture::Mentions(pmsg, "compressor.norm.weight"));
}
