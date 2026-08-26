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
// compressor + Lightning-Indexer, so the load has to materialize every carried
// family the host forward reads. `topk = 2` over the fixture's two routed
// experts keeps both live on every token.
FixtureOptions ForwardFixtureOptions() {
  FixtureOptions opt;
  opt.layers = 2;
  opt.num_hash_layers = 1;
  opt.topk = 2;
  opt.compress_ratios = {0, 4};
  opt.index_n_heads = 2;
  opt.index_head_dim = 4;
  opt.index_topk = 3;
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
