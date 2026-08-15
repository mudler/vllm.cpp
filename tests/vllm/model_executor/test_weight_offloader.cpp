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
#include <string>

#include "vllm/config/offload.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/weight_offloader.h"
#include "vt/tensor.h"

using vllm::CreateWeightOffloader;
using vllm::GetWeightOffloader;
using vllm::NoopWeightOffloader;
using vllm::OffloadBackend;
using vllm::OffloadConfig;
using vllm::SetWeightOffloader;
using vllm::WeightOffloader;
using vllm::WeightOffloaderChoice;

namespace {

// A probe that records whether the seam reached it. It never moves a weight;
// it exists to prove the install point and the read point are connected.
class CountingOffloader final : public WeightOffloader {
 public:
  void PrepareModel(vllm::LoadedModel&) override { ++prepared; }
  void PostInit() override { ++post_init; }
  void SyncPrevOnload() override { ++sync_prev; }
  void JoinAfterForward() override { ++join_after; }
  const char* name() const override { return "CountingOffloader"; }
  bool moves_weights() const override { return true; }

  int prepared = 0;
  int post_init = 0;
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
  CHECK_NOTHROW(noop.PostInit());
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

TEST_CASE("the factory NAMES a requested backend it cannot honour yet") {
  // W1 implements no backend. The contract under test is that a config which
  // WOULD offload is reported rather than silently ignored: a cpu_offload_gb
  // that frees nothing without a word is a memory bug the operator cannot see.
  struct Case {
    OffloadConfig cfg;
    const char* expect;
  };
  std::vector<Case> cases;

  OffloadConfig uva;
  uva.uva.cpu_offload_gb = 4.0;
  cases.push_back({uva, "uva"});

  OffloadConfig prefetch;
  prefetch.prefetch.offload_group_size = 8;
  prefetch.prefetch.offload_num_in_group = 2;
  cases.push_back({prefetch, "prefetch"});

  // "auto" with BOTH set resolves to prefetch (base.py:141-144), so the name
  // reported must be prefetch and not uva.
  OffloadConfig both;
  both.uva.cpu_offload_gb = 4.0;
  both.prefetch.offload_group_size = 8;
  both.prefetch.offload_num_in_group = 2;
  cases.push_back({both, "prefetch"});

  for (Case& c : cases) {
    c.cfg.Validate();
    WeightOffloaderChoice choice = CreateWeightOffloader(c.cfg);
    REQUIRE(choice.offloader != nullptr);
    // Still the no-op: W1 changes no behaviour.
    CHECK_MESSAGE(std::string(choice.offloader->name()) == "NoopWeightOffloader",
                  c.expect);
    CHECK_MESSAGE(choice.selected_backend_pending == c.expect, c.expect);
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
  // stayed green when the `GetWeightOffloader().PrepareModel(model)` call was
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
