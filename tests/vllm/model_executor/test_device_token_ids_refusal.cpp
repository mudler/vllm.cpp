// ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710) — the refusal predicate, both polarities,
// and the guard entered through the production seam.
//
// `ModelForwardInput::device_token_ids` was ADVISORY. When it is non-null the
// step's identifiers live in that device buffer and the host `token_ids` vector
// is stale for decode rows; a forward that ignored it embedded a host array the
// runner never wrote, and since `token_ids_cpu` is zero-initialised it decoded
// from TOKEN ID 0 at every step after the first — at rc=0, with fluent wrong
// output. Five architectures were caught this way one at a time, by hardware runs
// rather than by any gate (#1305, #2496, #2544). This row makes the sixth
// impossible to add silently, by giving the channel the shape its `multi_kv`
// sibling already has: a capability the model declares, and a refusal.
//
// WHY BOTH A PURE-PREDICATE FILE AND A SEAM CASE. The predicate is gateable
// without a model, a runner or a registry, which is the same argument
// `test_multi_kv_refusal.cpp` makes for its sibling — so every polarity can be
// pinned directly. But a predicate that nothing calls is a function, not a guard:
// AGENTS.md `## Nothing lands dead` requires the smallest failing test to enter
// through a production entry point, and `ModelRegistry::Forward` is the one here.
// So the last two cases construct a registration and go in through it.
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
// `model_registry.h` forward-declares these; the fixture below holds them BY
// VALUE and `ModelRegistry::Forward` returns one, so the test needs the real
// definitions rather than the declarations.
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, PagedKvCache, GdnStateCache
#include "vllm/transformers_utils/hf_config.h"   // HfConfig
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vt/backend.h"

namespace {

// The three inputs, named, so the polarity of each assertion is readable.
constexpr bool kClaimed = true;
constexpr bool kUnclaimed = false;
constexpr bool kStale = true;
constexpr bool kFresh = false;

}  // namespace

TEST_CASE("the refusal fires on the DISAGREEMENT, not on the arrival") {
  const int32_t ids[2] = {7, 11};

  // THE DEFECT'S SHAPE: the mirror is live, a row was spliced, and the forward
  // does not read the device buffer. This is the case that has cost five
  // architectures, and before this row it was not a refusal at all.
  CHECK(vllm::DeviceTokenIdsRefusalApplies(ids, kStale, kUnclaimed));

  // THE SIDE THAT, IF WRONG, MAKES THE CAPABILITY UNREACHABLE while every other
  // gate stays green: a forward that HAS wired itself to the device identifiers
  // must proceed. Sixteen registrations depend on this answer.
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(ids, kStale, kClaimed));

  // THE TERM THAT KEEPS A CORRECT MODEL ALIVE, and the reason this predicate is
  // not simply its sibling's `ptr != nullptr && !claimed`. The runner sets the
  // pointer on EVERY step once the mirror is engaged, including steps whose rows
  // are all prefill, where the combine splices nothing and the host vector is
  // perfectly good. `ForwardLlamaModelEmbedding` is a pooling forward: it reads
  // host identifiers, never reads the pointer, and serves only prefill-only
  // requests. A guard that refused here would take a working path away, which is
  // a worse outcome than the staleness this exists to stop.
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(ids, kFresh, kUnclaimed));
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(ids, kFresh, kClaimed));

  // NO MIRROR, EITHER WAY. `device_token_ids` is null on every non-mirror path,
  // which is every CPU step and every run with VT_ASYNC_DEVICE_MIRROR=0. A guard
  // that fired here would break every forward in the tree at once. The `stale`
  // flag is meaningless without the pointer and must not be able to refuse on its
  // own.
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(nullptr, kStale, kUnclaimed));
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(nullptr, kStale, kClaimed));
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(nullptr, kFresh, kUnclaimed));
  CHECK_FALSE(vllm::DeviceTokenIdsRefusalApplies(nullptr, kFresh, kClaimed));
}

namespace {

// A registration whose forward records that it ran. The guard fires BEFORE
// dispatch, so "did the forward run" is exactly "did the guard let the step
// through" — which is the observable a reachability case needs.
bool g_forward_ran = false;

vllm::ForwardLogits RecordingForward(vllm::LoadedModel& model,
                                     const vllm::ModelForwardInput& input) {
  (void)model;
  (void)input;
  g_forward_ran = true;
  return {};
}

class FakeLoadedModel : public vllm::LoadedModel {
 public:
  explicit FakeLoadedModel(const vllm::ModelRegistration& reg)
      : vllm::LoadedModel(reg) {}
};

// Held by reference inside `ModelForwardInput`, so they outlive each call.
struct StepFixture {
  std::vector<int32_t> token_ids{7, 11};
  std::vector<int32_t> positions{0, 1};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vllm::HfConfig config{};
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<int32_t> logits_indices;

  vllm::ModelForwardInput Make() {
    vllm::ModelForwardInput in{token_ids, positions, attn_meta,
                               gdn_meta,  attn_kv,   gdn_state,
                               config,    queue,     logits_indices};
    // TWO requests, MIXED: this is the granularity case at the seam. A guard
    // that consulted a per-request disposition rather than the step's own answer
    // would find a prefill row here and let the step through.
    in.num_reqs = 2;
    return in;
  }
};

}  // namespace

TEST_CASE("ModelRegistry::Forward REFUSES a stale step for an unclaimed forward") {
  vllm::ModelFactory factory{};
  factory.forward = &RecordingForward;
  factory.consumes_device_token_ids = false;  // the default, and the mechanism
  const vllm::ModelRegistration reg{"FakeUnclaimedForCausalLM", &factory, {}};
  FakeLoadedModel model(reg);

  StepFixture fx;
  const int32_t device_ids[2] = {7, 11};

  // ARM 1 — THE ROUTING. With no mirror the step reaches the forward. Without
  // this arm a guard that refused unconditionally would look correct.
  {
    vllm::ModelForwardInput in = fx.Make();
    REQUIRE(in.device_token_ids == nullptr);
    REQUIRE_FALSE(in.host_token_ids_stale);
    g_forward_ran = false;
    vllm::ModelRegistry::Forward(model, in);
    CHECK(g_forward_ran);
  }

  // ARM 2 — THE MIRROR IS LIVE BUT NOTHING IS STALE (the all-prefill step, and
  // every step a pooling model serves). The forward still runs.
  {
    vllm::ModelForwardInput in = fx.Make();
    in.device_token_ids = device_ids;
    in.host_token_ids_stale = false;
    g_forward_ran = false;
    vllm::ModelRegistry::Forward(model, in);
    CHECK(g_forward_ran);
  }

  // ARM 3 — THE DEFECT. A stale step reaches a forward that cannot read the
  // device identifiers, and the engine refuses BEFORE dispatch rather than
  // handing it a host array the runner never wrote.
  {
    vllm::ModelForwardInput in = fx.Make();
    in.device_token_ids = device_ids;
    in.host_token_ids_stale = true;
    g_forward_ran = false;

    std::string message;
    try {
      vllm::ModelRegistry::Forward(model, in);
      FAIL("ModelRegistry::Forward accepted a stale step for a forward that "
           "does not read device_token_ids");
    } catch (const std::exception& e) {
      message = e.what();
    }

    // It refused BEFORE dispatch. This is the assertion that separates a guard
    // from a diagnostic: a message printed after the model already embedded
    // stale identifiers would not have prevented anything.
    CHECK_FALSE(g_forward_ran);

    // It names the arriving ARCHITECTURE, computed from the handle rather than
    // enumerated, so the message routes the reader to the row that owes the
    // consuming forward.
    CHECK(message.find("FakeUnclaimedForCausalLM") != std::string::npos);
    // And it names the capability whose absence caused the refusal, so the fix
    // is in the message rather than in a spec the reader has to find.
    CHECK(message.find("consumes_device_token_ids") != std::string::npos);
  }
}

TEST_CASE("ModelRegistry::Forward ADMITS a stale step for a claimed forward") {
  // The direction nothing else asserts. If the guard refused here, all sixteen
  // registrations that read the device identifiers would stop serving the async
  // default arm — loudly, but only on a CUDA box, and every CPU gate in this tree
  // would stay green.
  vllm::ModelFactory factory{};
  factory.forward = &RecordingForward;
  factory.consumes_device_token_ids = true;
  const vllm::ModelRegistration reg{"FakeClaimedForCausalLM", &factory, {}};
  FakeLoadedModel model(reg);

  StepFixture fx;
  const int32_t device_ids[2] = {7, 11};

  vllm::ModelForwardInput in = fx.Make();
  in.device_token_ids = device_ids;
  in.host_token_ids_stale = true;
  g_forward_ran = false;
  vllm::ModelRegistry::Forward(model, in);
  CHECK(g_forward_ran);
}
