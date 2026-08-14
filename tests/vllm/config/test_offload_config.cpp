// ENG-WEIGHT-OFFLOAD W0 (#797) — the weight-offload config surface.
//
// Ported from (test-porting protocol .agents/porting.md), all @ 555967922:
//   * vllm/config/offload.py:12 — OffloadBackend Literal["auto","uva","prefetch"].
//   * vllm/config/offload.py:23,34-44 — cpu_offload_gb / cpu_offload_params, and
//     the docstring at :39-43 that is the AUTHORITY on segment matching:
//     for "mlp.experts.w2_weight", "experts" and "experts.w2_weight" match;
//     "expert" and "w2" do NOT.
//   * vllm/config/offload.py:54,57,62,66,70-76 — the prefetch grouping fields and
//     the worked example at :57 (group_size=8, num_in_group=2 offloads layers
//     6,7,14,15,22,23,...).
//   * vllm/config/offload.py:96-136 — validate_offload_config: the TWO hard
//     errors (:100-106, :107-112) and the THREE warnings (:120-135).
//   * vllm/model_executor/offloader/base.py:139-149 — create_offloader's "auto"
//     selection ORDER: prefetch if offload_group_size > 0, elif uva if
//     cpu_offload_gb > 0, else NoopOffloader.
//   * vllm/model_executor/offloader/uva.py:91-93 — the match as implemented,
//     `f".{param}." in f".{name}."`.
//
// Upstream has NO unit test for this config: its coverage is the end-to-end
// tests/basic_correctness/test_cpu_offload.py and tests/quantization/
// test_cpu_offload.py, both of which need a GPU and a real checkpoint and both
// of which land in W2/W3, not here. These cases are therefore AUTHORED against
// the upstream source, exactly as tests/vllm/config/test_multimodal_config.cpp
// records for the mm limits — they pin the pure-config behaviour those e2e
// suites depend on. SKIPPED from upstream, with reason: compute_hash (:138) —
// we have no config-hash surface.
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>

#include "vllm/config/offload.h"

using vllm::OffloadBackend;
using vllm::OffloadConfig;
using vllm::UVAOffloadConfig;

namespace {

OffloadConfig WithUva(double gb) {
  OffloadConfig c;
  c.uva.cpu_offload_gb = gb;
  return c;
}

OffloadConfig WithPrefetch(int64_t group, int64_t in_group, int64_t step) {
  OffloadConfig c;
  c.prefetch.offload_group_size = group;
  c.prefetch.offload_num_in_group = in_group;
  c.prefetch.offload_prefetch_step = step;
  return c;
}

}  // namespace

TEST_CASE("offload backend enum round-trips the three upstream names") {
  // offload.py:12 — exactly three, and nothing else parses.
  CHECK(vllm::parse_offload_backend("auto") == OffloadBackend::kAuto);
  CHECK(vllm::parse_offload_backend("uva") == OffloadBackend::kUva);
  CHECK(vllm::parse_offload_backend("prefetch") == OffloadBackend::kPrefetch);
  CHECK(std::string(vllm::offload_backend_str(OffloadBackend::kAuto)) == "auto");
  CHECK(std::string(vllm::offload_backend_str(OffloadBackend::kUva)) == "uva");
  CHECK(std::string(vllm::offload_backend_str(OffloadBackend::kPrefetch)) ==
        "prefetch");
  // Not a backend upstream knows. "disk" in particular is ENG-EXPERT-STREAM's
  // idea, not this row's, and must not silently parse here.
  CHECK(vllm::parse_offload_backend("disk") == std::nullopt);
  CHECK(vllm::parse_offload_backend("") == std::nullopt);
  CHECK(vllm::parse_offload_backend("UVA") == std::nullopt);
}

TEST_CASE("segment matching is dot-anchored, per offload.py:39-43") {
  const std::string name = "mlp.experts.w2_weight";
  // Upstream's own worked examples, verbatim.
  CHECK(UVAOffloadConfig::MatchesSegment(name, "experts"));
  CHECK(UVAOffloadConfig::MatchesSegment(name, "experts.w2_weight"));
  CHECK_FALSE(UVAOffloadConfig::MatchesSegment(name, "expert"));
  CHECK_FALSE(UVAOffloadConfig::MatchesSegment(name, "w2"));

  // The distinction the docstring exists for (:43): a trailing segment must not
  // match a LONGER segment that merely starts with it.
  CHECK(UVAOffloadConfig::MatchesSegment(name, "w2_weight"));
  CHECK_FALSE(
      UVAOffloadConfig::MatchesSegment("mlp.experts.w2_weight_scale", "w2_weight"));
  CHECK(UVAOffloadConfig::MatchesSegment("mlp.experts.w2_weight_scale",
                                         "w2_weight_scale"));

  // Leading and whole-name segments.
  CHECK(UVAOffloadConfig::MatchesSegment(name, "mlp"));
  CHECK(UVAOffloadConfig::MatchesSegment(name, "mlp.experts.w2_weight"));
  CHECK_FALSE(UVAOffloadConfig::MatchesSegment(name, "ml"));
}

TEST_CASE("empty cpu_offload_params targets EVERYTHING (offload.py:36-38)") {
  UVAOffloadConfig u;
  CHECK(u.cpu_offload_params.empty());
  CHECK(u.Targets("mlp.experts.w2_weight"));
  CHECK(u.Targets("self_attn.q_proj.weight"));
  CHECK(u.Targets("anything"));

  // Non-empty => ONLY matching parameters (":Unmatched parameters are not
  // offloaded", :36).
  u.cpu_offload_params = {"experts"};
  CHECK(u.Targets("mlp.experts.w2_weight"));
  CHECK_FALSE(u.Targets("self_attn.q_proj.weight"));
}

TEST_CASE("cpu_offload_gb converts to bytes as int(gb * 1024**3)") {
  // offloader/base.py:155.
  CHECK(WithUva(0.0).uva.cpu_offload_max_bytes() == 0);
  CHECK(WithUva(1.0).uva.cpu_offload_max_bytes() == 1073741824LL);
  CHECK(WithUva(10.0).uva.cpu_offload_max_bytes() == 10737418240LL);
  // Truncation, not rounding — int() in Python truncates toward zero.
  CHECK(WithUva(0.5).uva.cpu_offload_max_bytes() == 536870912LL);
}

TEST_CASE("prefetch grouping reproduces the worked example at offload.py:57") {
  // group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
  vllm::PrefetchOffloadConfig p;
  p.offload_group_size = 8;
  p.offload_num_in_group = 2;

  // Count them rather than spot-check one: a per-group off-by-one passes a
  // single assertion and fails the set.
  int offloaded = 0;
  for (int64_t i = 0; i < 24; ++i) {
    const bool want = (i % 8) >= 6;
    CHECK_MESSAGE(p.OffloadsLayer(i) == want, "layer ", i);
    if (want) ++offloaded;
  }
  CHECK(offloaded == 6);  // 6,7,14,15,22,23

  // Disabled (the default) offloads nothing at all.
  vllm::PrefetchOffloadConfig off;
  CHECK(off.offload_group_size == 0);
  for (int64_t i = 0; i < 24; ++i) CHECK_FALSE(off.OffloadsLayer(i));
}

TEST_CASE("validate: num_in_group > group_size is a hard error (:100-106)") {
  OffloadConfig c = WithPrefetch(4, 5, 1);
  CHECK_THROWS_AS(c.Validate(), std::invalid_argument);

  // Equal is legal (the bound is `>`), and so is less.
  OffloadConfig ok = WithPrefetch(4, 4, 1);
  CHECK_NOTHROW(ok.Validate());
  OffloadConfig ok2 = WithPrefetch(8, 2, 1);
  CHECK_NOTHROW(ok2.Validate());
}

TEST_CASE("validate: prefetch_step < 1 is a hard error when enabled (:107-112)") {
  OffloadConfig c = WithPrefetch(8, 2, 0);
  CHECK_THROWS_AS(c.Validate(), std::invalid_argument);

  // The SAME zero step is fine when prefetch is NOT enabled — upstream guards
  // the whole block on `backend == "prefetch" or offload_group_size > 0`
  // (:99), so a disabled prefetch config with step 0 must pass.
  OffloadConfig disabled;
  disabled.prefetch.offload_prefetch_step = 0;
  CHECK(disabled.prefetch.offload_group_size == 0);
  CHECK_NOTHROW(disabled.Validate());

  // ...but an EXPLICIT prefetch backend enables the block even with group_size
  // still 0 (:99 is an `or`), so the step check bites there too.
  OffloadConfig explicit_backend;
  explicit_backend.offload_backend = OffloadBackend::kPrefetch;
  explicit_backend.prefetch.offload_prefetch_step = 0;
  CHECK_THROWS_AS(explicit_backend.Validate(), std::invalid_argument);
}

TEST_CASE("validate: the pydantic field bounds are enforced too") {
  // ge=0 on cpu_offload_gb (:23).
  OffloadConfig neg = WithUva(-1.0);
  CHECK_THROWS_AS(neg.Validate(), std::invalid_argument);
  // ge=0 on offload_group_size (:54).
  OffloadConfig g;
  g.prefetch.offload_group_size = -1;
  CHECK_THROWS_AS(g.Validate(), std::invalid_argument);
  // ge=1 on offload_num_in_group (:62).
  OffloadConfig n = WithPrefetch(8, 0, 1);
  CHECK_THROWS_AS(n.Validate(), std::invalid_argument);
  // ge=0 on offload_prefetch_step (:66) — negative is out of bounds even when
  // prefetch is disabled, because that bound is on the FIELD, not the validator.
  OffloadConfig s;
  s.prefetch.offload_prefetch_step = -1;
  CHECK_THROWS_AS(s.Validate(), std::invalid_argument);
}

TEST_CASE("validate: the three backend/field mismatches WARN and never throw") {
  // :120-126 uva backend with prefetch fields set.
  OffloadConfig a = WithPrefetch(8, 2, 1);
  a.offload_backend = OffloadBackend::kUva;
  CHECK_NOTHROW(a.Validate());
  CHECK(a.warnings.size() == 1);

  // :127-133 prefetch backend with uva fields set.
  OffloadConfig b = WithUva(1.0);
  b.offload_backend = OffloadBackend::kPrefetch;
  b.prefetch.offload_group_size = 8;
  b.prefetch.offload_num_in_group = 2;
  CHECK_NOTHROW(b.Validate());
  CHECK(b.warnings.size() == 1);

  // :134-135 auto with BOTH set — and the message's claim is load-bearing:
  // "Prefetch backend will be selected."
  OffloadConfig c = WithPrefetch(8, 2, 1);
  c.uva.cpu_offload_gb = 1.0;
  CHECK(c.offload_backend == OffloadBackend::kAuto);
  CHECK_NOTHROW(c.Validate());
  CHECK(c.warnings.size() == 1);
  CHECK(c.ResolvedBackend() == OffloadBackend::kPrefetch);

  // A clean config warns about nothing, and Validate is idempotent — calling it
  // twice must not accumulate duplicate warnings.
  OffloadConfig clean = WithUva(1.0);
  clean.Validate();
  CHECK(clean.warnings.empty());
  clean.Validate();
  CHECK(clean.warnings.empty());
  a.Validate();
  CHECK(a.warnings.size() == 1);
}

TEST_CASE("auto selection order mirrors create_offloader (base.py:139-149)") {
  // Nothing configured => NO offloading. This is the inert default and the one
  // that must leave the engine on its existing path.
  OffloadConfig none;
  CHECK(none.ResolvedBackend() == std::nullopt);
  CHECK_FALSE(none.is_offloading_enabled());

  // prefetch wins when group_size > 0, even with a uva budget also set.
  OffloadConfig both = WithPrefetch(8, 2, 1);
  both.uva.cpu_offload_gb = 4.0;
  CHECK(both.ResolvedBackend() == OffloadBackend::kPrefetch);

  // uva only when group_size == 0 and gb > 0.
  CHECK(WithUva(1.0).ResolvedBackend() == OffloadBackend::kUva);

  // An explicit backend SELECTS that backend even at a zero budget — upstream
  // constructs the offloader either way (base.py:151-159). But it moves nothing,
  // so `is_offloading_enabled` must still be false. These are two different
  // questions and conflating them would let a zero-budget explicit backend read
  // as "offloading is on".
  OffloadConfig explicit_uva;
  explicit_uva.offload_backend = OffloadBackend::kUva;
  CHECK(explicit_uva.ResolvedBackend() == OffloadBackend::kUva);
  CHECK_FALSE(explicit_uva.is_offloading_enabled());

  OffloadConfig explicit_prefetch;
  explicit_prefetch.offload_backend = OffloadBackend::kPrefetch;
  CHECK(explicit_prefetch.ResolvedBackend() == OffloadBackend::kPrefetch);
  CHECK_FALSE(explicit_prefetch.is_offloading_enabled());

  // ...and a budget-bearing explicit backend does enable it.
  OffloadConfig live = WithUva(2.0);
  live.offload_backend = OffloadBackend::kUva;
  CHECK(live.is_offloading_enabled());
}

TEST_CASE("json parsing mirrors the kv_transfer_config precedent") {
  // Empty/blank == default == inert, exactly as `kv_transfer_config` empty means
  // the byte-identical default engine.
  CHECK_FALSE(vllm::parse_offload_config_json("").is_offloading_enabled());
  CHECK_FALSE(vllm::parse_offload_config_json("   ").is_offloading_enabled());
  CHECK_FALSE(vllm::parse_offload_config_json("{}").is_offloading_enabled());

  OffloadConfig c = vllm::parse_offload_config_json(R"({
    "offload_backend": "prefetch",
    "prefetch": {"offload_group_size": 8, "offload_num_in_group": 2,
                 "offload_prefetch_step": 4, "offload_params": ["experts"]}
  })");
  CHECK(c.offload_backend == OffloadBackend::kPrefetch);
  CHECK(c.prefetch.offload_group_size == 8);
  CHECK(c.prefetch.offload_num_in_group == 2);
  CHECK(c.prefetch.offload_prefetch_step == 4);
  CHECK(c.prefetch.offload_params.count("experts") == 1);

  OffloadConfig u = vllm::parse_offload_config_json(
      R"({"uva": {"cpu_offload_gb": 10, "cpu_offload_params": ["experts","mlp"]}})");
  CHECK(u.offload_backend == OffloadBackend::kAuto);
  CHECK(u.uva.cpu_offload_gb == doctest::Approx(10.0));
  CHECK(u.uva.cpu_offload_params.size() == 2);
  CHECK(u.ResolvedBackend() == OffloadBackend::kUva);

  // Malformed, unknown backend, and wrong-typed fields are all refused — a
  // silently-ignored typo in an offload budget is a memory bug the user cannot
  // see.
  CHECK_THROWS_AS(vllm::parse_offload_config_json("{not json"),
                  std::invalid_argument);
  CHECK_THROWS_AS(vllm::parse_offload_config_json(R"({"offload_backend":"disk"})"),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      vllm::parse_offload_config_json(R"({"uva":{"cpu_offload_gb":"ten"}})"),
      std::invalid_argument);
  CHECK_THROWS_AS(vllm::parse_offload_config_json(R"({"uva": 5})"),
                  std::invalid_argument);
}
