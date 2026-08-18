// ENG-WEIGHT-OFFLOAD W1 (#797) — the offloader seam.
//
// Ported from (test-porting protocol .agents/porting.md), all @ 555967922:
//   * vllm/model_executor/offloader/base.py:94-103 — NoopOffloader returns the
//     modules unchanged.
//   * vllm/model_executor/offloader/base.py:106-125 — the module-global
//     `_instance`, defaulted to NoopOffloader, with get_offloader/set_offloader.
//   * vllm/model_executor/offloader/base.py:126-162 — create_offloader, whose
//     "auto" order is prefetch when offload_group_size > 0, elif uva when
//     cpu_offload_gb > 0, else NoopOffloader.
//
// Upstream has NO unit test for the seam: its coverage is the end-to-end
// tests/basic_correctness/test_cpu_offload.py, which needs a GPU and a real
// checkpoint and belongs to W2/W3. These cases are AUTHORED against the
// upstream source, as tests/vllm/config/test_offload_config.cpp records for the
// config surface.
//
// The load-bearing property here is INERTNESS: the default instance must be the
// no-op, so an engine that never installs a backend behaves exactly as it does
// today. A seam that defaulted to anything else would change every model path.
#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "vllm/config/offload.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/weight_offload_policy.h"
#include "vllm/model_executor/weight_offloader.h"
#include "vt/tensor.h"

using vllm::CreateWeightOffloader;
using vllm::GetWeightOffloader;
using vllm::NoopWeightOffloader;
using vllm::UvaWeightOffloader;
using vllm::WeightOffloadPolicy;
using vllm::OffloadBackend;
using vllm::OffloadConfig;
using vllm::SetWeightOffloader;
using vllm::WeightOffloader;
using vllm::WeightOffloadDecision;
using vllm::WeightOffloaderChoice;

namespace {

// A probe that records whether the seam reached it. It never moves a weight;
// it exists to prove the install point and the read point are connected.
class CountingOffloader final : public WeightOffloader {
 public:
  WeightOffloadDecision ConsiderWeight(const std::string&, int64_t bytes) override {
    ++considered;
    bytes_seen += bytes;
    return WeightOffloadDecision::kOffload;
  }
  void OnModelPrepared(vllm::LoadedModel&) override { ++prepared; }
  void SyncPrevOnload() override { ++sync_prev; }
  void JoinAfterForward() override { ++join_after; }
  const char* name() const override { return "CountingOffloader"; }
  bool moves_weights() const override { return true; }

  int64_t offloaded_bytes() const override { return bytes_seen; }
  int64_t weights_considered() const override { return considered; }

  int prepared = 0;
  int considered = 0;
  int64_t bytes_seen = 0;
  int sync_prev = 0;
  int join_after = 0;
};

// Restores whatever instance was installed, so one case cannot leak a backend
// into the next. The global is process-wide, exactly as upstream's is.
struct OffloaderGuard {
  std::unique_ptr<WeightOffloader> saved;
  explicit OffloaderGuard(std::unique_ptr<WeightOffloader> next)
      : saved(SetWeightOffloader(std::move(next))) {}
  ~OffloaderGuard() { SetWeightOffloader(std::move(saved)); }
};

}  // namespace

TEST_CASE("the default offloader is the no-op, and it moves no weight") {
  // base.py:106-107. This is the inertness property the whole row rests on.
  WeightOffloader& d = GetWeightOffloader();
  CHECK(std::string(d.name()) == "NoopWeightOffloader");
  CHECK_FALSE(d.moves_weights());
}

TEST_CASE("set and get round-trip, and a null argument restores the no-op") {
  // base.py:113-125.
  auto probe = std::make_unique<CountingOffloader>();
  CountingOffloader* raw = probe.get();
  {
    OffloaderGuard g(std::move(probe));
    CHECK(&GetWeightOffloader() == raw);
    CHECK(std::string(GetWeightOffloader().name()) == "CountingOffloader");
    CHECK(GetWeightOffloader().moves_weights());

    // A null install must not leave the seam empty: it restores the no-op, so
    // GetWeightOffloader() can never return a dangling reference.
    SetWeightOffloader(nullptr);
    CHECK(std::string(GetWeightOffloader().name()) == "NoopWeightOffloader");
    CHECK_FALSE(GetWeightOffloader().moves_weights());
  }
  // The guard restored the previous instance.
  CHECK(std::string(GetWeightOffloader().name()) == "NoopWeightOffloader");
}

TEST_CASE("the no-op accepts every hook without effect") {
  // base.py:68-85: post_init, sync_prev_onload and join_after_forward are
  // defaulted no-ops upstream, so the graph seam can call them unconditionally.
  NoopWeightOffloader noop;
  CHECK_NOTHROW(noop.SyncPrevOnload());
  CHECK_NOTHROW(noop.JoinAfterForward());
  CHECK_FALSE(noop.moves_weights());
}

TEST_CASE("the factory returns the no-op for an unconfigured config") {
  // base.py:145-149: "auto" with both budgets at zero yields NoopOffloader.
  OffloadConfig none;
  WeightOffloaderChoice c = CreateWeightOffloader(none);
  REQUIRE(c.offloader != nullptr);
  CHECK(std::string(c.offloader->name()) == "NoopWeightOffloader");
  CHECK_FALSE(c.offloader->moves_weights());
  // Nothing was requested, so nothing is pending.
  CHECK(c.selected_backend_pending.empty());
}

TEST_CASE("the factory NAMES only a backend it cannot build") {
  // UPDATED BY W2b, and the change is deliberate rather than a widened scope:
  // W1 built NEITHER backend, so this case asserted that `uva` was reported as
  // pending. W2b builds the UVA arm, so `uva` is no longer pending and only
  // `prefetch` is. The assertion is kept and its expectation moved; deleting it
  // would have hidden the very distinction the engine's two messages rest on.
  //
  // "auto" with BOTH set still resolves to prefetch (base.py:141-144), so it is
  // still reported as unbuilt even though the UVA arm now exists.
  struct Case {
    OffloadConfig cfg;
    const char* expect_name;
    const char* expect_pending;
  };
  std::vector<Case> cases;

  OffloadConfig uva;
  uva.uva.cpu_offload_gb = 4.0;
  cases.push_back({uva, "UvaWeightOffloader", ""});

  OffloadConfig prefetch;
  prefetch.prefetch.offload_group_size = 8;
  prefetch.prefetch.offload_num_in_group = 2;
  cases.push_back({prefetch, "NoopWeightOffloader", "prefetch"});

  OffloadConfig both;
  both.uva.cpu_offload_gb = 4.0;
  both.prefetch.offload_group_size = 8;
  both.prefetch.offload_num_in_group = 2;
  cases.push_back({both, "NoopWeightOffloader", "prefetch"});

  for (Case& c : cases) {
    c.cfg.Validate();
    WeightOffloaderChoice choice = CreateWeightOffloader(c.cfg);
    REQUIRE(choice.offloader != nullptr);
    CHECK_MESSAGE(std::string(choice.offloader->name()) == c.expect_name,
                  c.expect_name);
    CHECK_MESSAGE(choice.selected_backend_pending == c.expect_pending,
                  c.expect_name);
  }
}

TEST_CASE("an explicit backend at a zero budget is NOT reported as pending") {
  // `is_offloading_enabled()` is false when the selected backend has no budget,
  // so nothing would move and there is nothing to warn about. Reporting it
  // would train the operator to ignore the message.
  OffloadConfig explicit_uva;
  explicit_uva.offload_backend = OffloadBackend::kUva;
  CHECK(explicit_uva.ResolvedBackend() == OffloadBackend::kUva);
  CHECK_FALSE(explicit_uva.is_offloading_enabled());

  WeightOffloaderChoice c = CreateWeightOffloader(explicit_uva);
  REQUIRE(c.offloader != nullptr);
  CHECK(std::string(c.offloader->name()) == "NoopWeightOffloader");
  CHECK(c.selected_backend_pending.empty());
}

TEST_CASE("ModelRegistry::Prepare reaches the installed offloader") {
  // THE WIRING TEST. Mutation testing showed the other cases in this file all
  // stayed green when the `GetWeightOffloader().OnModelPrepared(model)` call was
  // deleted from ModelRegistry::Prepare, because none of them drives a model.
  // This case closes that gap: it is the only proof that the install point and
  // the wrap site are connected.
  //
  // Upstream's equivalent connection is get_offloader() inside make_layers
  // (model_executor/models/utils.py:824). We have no make_layers, so Prepare is
  // the analogue (see weight_offloader.h for the recorded deviation).
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::MakeQwen3_5DenseLoadedModel(vllm::Qwen3_5DenseWeights{});
  REQUIRE(model != nullptr);

  auto probe = std::make_unique<CountingOffloader>();
  CountingOffloader* raw = probe.get();
  OffloaderGuard g(std::move(probe));
  CHECK(raw->prepared == 0);

  vllm::HfConfig config;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_NOTHROW(vllm::ModelRegistry::Prepare(*model, config, q));

  // Exactly once per Prepare, not zero and not twice.
  CHECK(raw->prepared == 1);
  CHECK_NOTHROW(vllm::ModelRegistry::Prepare(*model, config, q));
  CHECK(raw->prepared == 2);
}

TEST_CASE("the no-op keeps every weight resident, whatever it is asked") {
  // The inertness property, now expressed on the DECISION seam rather than on a
  // hook nobody implements. A loader that consults the default offloader gets
  // the same answer for every weight, so the existing path is unchanged by
  // construction and not by a flag the caller must remember to check.
  NoopWeightOffloader noop;
  const char* names[] = {"mlp.experts.w2_weight", "self_attn.q_proj.weight",
                         "model.embed_tokens.weight", ""};
  for (const char* n : names) {
    CHECK(noop.ConsiderWeight(n, 1 << 20) ==
          WeightOffloadDecision::kBudgetExhausted);
  }
  CHECK(noop.offloaded_bytes() == 0);
  CHECK_FALSE(noop.moves_weights());
}

TEST_CASE("the factory builds the UVA arm, and it answers under its budget") {
  // W2b. Upstream constructs UVAOffloader with
  // int(cpu_offload_gb * 1024**3) and the targeting set (base.py:151-159).
  OffloadConfig cfg;
  cfg.uva.cpu_offload_gb = 1.0;
  cfg.uva.cpu_offload_params = {"experts"};
  cfg.Validate();

  WeightOffloaderChoice c = CreateWeightOffloader(cfg);
  REQUIRE(c.offloader != nullptr);
  CHECK(std::string(c.offloader->name()) == "UvaWeightOffloader");
  CHECK(c.offloader->moves_weights());
  // It is BUILT, so it is not reported as a backend this build lacks. That
  // distinction is what the engine's two different messages rest on.
  CHECK(c.selected_backend_pending.empty());

  // It answers through the policy it owns: targeted weights are offloaded,
  // untargeted ones are not, and the running total advances only for the first.
  CHECK(c.offloader->ConsiderWeight("mlp.experts.w2_weight", 4096) ==
        WeightOffloadDecision::kOffload);
  CHECK(c.offloader->offloaded_bytes() == 4096);
  CHECK(c.offloader->ConsiderWeight("self_attn.q_proj.weight", 4096) ==
        WeightOffloadDecision::kNotTargeted);
  CHECK(c.offloader->offloaded_bytes() == 4096);
}

TEST_CASE("the UVA arm stops at its budget and reports the total") {
  UvaWeightOffloader off(WeightOffloadPolicy(100, {}));
  CHECK(off.moves_weights());
  CHECK(off.max_bytes() == 100);
  CHECK(off.ConsiderWeight("a", 60) == WeightOffloadDecision::kOffload);
  CHECK(off.ConsiderWeight("b", 60) == WeightOffloadDecision::kOffload);
  CHECK(off.offloaded_bytes() == 120);
  // Spent. Everything after stays resident, which is what bounds the host
  // memory the feature is allowed to consume.
  CHECK(off.ConsiderWeight("c", 1) == WeightOffloadDecision::kBudgetExhausted);
  CHECK(off.offloaded_bytes() == 120);
}

TEST_CASE("a zero-budget UVA arm reports that it moves nothing") {
  // `moves_weights()` is what the engine uses to decide whether to warn, so a
  // zero budget must answer false or the operator gets a message about an
  // offloader that was never going to move anything.
  UvaWeightOffloader off(WeightOffloadPolicy(0, {}));
  CHECK_FALSE(off.moves_weights());
  CHECK(off.ConsiderWeight("a", 1) == WeightOffloadDecision::kBudgetExhausted);
  CHECK(off.offloaded_bytes() == 0);
}

TEST_CASE("prefetch is still reported as a backend this build lacks") {
  OffloadConfig cfg;
  cfg.prefetch.offload_group_size = 8;
  cfg.prefetch.offload_num_in_group = 2;
  cfg.Validate();
  WeightOffloaderChoice c = CreateWeightOffloader(cfg);
  REQUIRE(c.offloader != nullptr);
  CHECK(std::string(c.offloader->name()) == "NoopWeightOffloader");
  CHECK(c.selected_backend_pending == "prefetch");
}

TEST_CASE("the totality guard refuses an offload the model cannot honour") {
  // THE POINT OF THE GUARD. There is no single upload seam in this tree, so a
  // model whose loader was never wired would accept a budget and keep every
  // weight on the device with no error anywhere. That is a memory bug the
  // operator cannot see, and the guard converts it into a refusal that names
  // the architecture.
  OffloadConfig on;
  on.uva.cpu_offload_gb = 4.0;
  on.Validate();

  // Unsupported model + configured offload => refused.
  CHECK_THROWS_AS(
      vllm::RefuseUnsupportedWeightOffload(on, "Qwen3MoeForCausalLM", false),
      std::invalid_argument);

  // Supported model => allowed through.
  CHECK_NOTHROW(
      vllm::RefuseUnsupportedWeightOffload(on, "Qwen3MoeForCausalLM", true));

  // No offload configured => never refused, whatever the model declares. This
  // is what keeps every existing engine path untouched.
  OffloadConfig off;
  CHECK_NOTHROW(vllm::RefuseUnsupportedWeightOffload(off, "AnyArch", false));
  CHECK_NOTHROW(vllm::RefuseUnsupportedWeightOffload(off, "AnyArch", true));

  // A backend selected at a ZERO budget would move nothing, so it is not
  // refused either. Refusing it would train the operator to ignore the message.
  OffloadConfig zero;
  zero.offload_backend = OffloadBackend::kUva;
  zero.Validate();
  CHECK_FALSE(zero.is_offloading_enabled());
  CHECK_NOTHROW(vllm::RefuseUnsupportedWeightOffload(zero, "AnyArch", false));
}

TEST_CASE("the refusal message names the architecture and the reason") {
  // An operator reading a log needs the model name, not just "unsupported".
  OffloadConfig on;
  on.uva.cpu_offload_gb = 1.0;
  on.Validate();
  try {
    vllm::RefuseUnsupportedWeightOffload(on, "DeepseekV4ForCausalLM", false);
    FAIL("expected a refusal");
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    CHECK(msg.find("DeepseekV4ForCausalLM") != std::string::npos);
    CHECK(msg.find("free nothing") != std::string::npos);
  }
}

TEST_CASE("the second guard catches a model that declares support and never asks") {
  // A model can declare the capability and then never call ConsiderWeight. The
  // first guard cannot see that: the run simply offloads nothing. ZERO is the
  // only count that proves the defect.
  UvaWeightOffloader off(WeightOffloadPolicy(1024, {}));
  CHECK(off.moves_weights());
  CHECK(off.weights_considered() == 0);
  CHECK_THROWS_AS(
      vllm::VerifyWeightOffloadWasConsulted(off, "SomeArch", true),
      std::invalid_argument);

  // One question is enough to prove the loader consulted the seam. Whether it
  // asked about EVERY weight is that loader's own test to make.
  off.ConsiderWeight("mlp.experts.w2_weight", 8);
  CHECK(off.weights_considered() == 1);
  CHECK_NOTHROW(vllm::VerifyWeightOffloadWasConsulted(off, "SomeArch", true));

  // A model that does NOT declare support is already refused by the first
  // guard, so this one stays silent rather than reporting the same fault twice.
  UvaWeightOffloader other(WeightOffloadPolicy(1024, {}));
  CHECK_NOTHROW(vllm::VerifyWeightOffloadWasConsulted(other, "SomeArch", false));

  // An offloader that moves nothing is never a defect.
  NoopWeightOffloader noop;
  CHECK_NOTHROW(vllm::VerifyWeightOffloadWasConsulted(noop, "SomeArch", true));
}

TEST_CASE("no model in the tree declares weight-offload support yet") {
  // The default is false, and that default is the mechanism. This case pins it:
  // when the first loader is wired, this assertion fails and whoever wired it
  // must come back here and say which model changed. That is the record the
  // capability is supposed to leave.
  int supporting = 0;
  for (const vllm::ModelRegistration& r : vllm::ModelRegistry::Registrations()) {
    if (r.factory != nullptr && r.factory->supports_weight_offload) ++supporting;
  }
  CHECK_MESSAGE(supporting == 0,
                "a model now declares supports_weight_offload; update this "
                "count and the spec's W2c note");
}
