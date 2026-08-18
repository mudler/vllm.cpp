// The registry type-confusion CLASS gate — issue #847, spec
// `.agents/specs/registry-downcast-sweep.md`.
//
// Every registered `prepare`/`forward` is handed a type-erased `LoadedModel&`
// and has to open it as its own concrete model. Doing that with a bare
// `static_cast` down the hierarchy is a PROMISE the compiler is entitled to act
// on: on an object whose dynamic type is not that model, every member call
// through the resulting reference is undefined behaviour. UBSan's vptr check
// reports it as "member call on address ... which does not point to an object of
// type 'X'", and `-fno-sanitize-recover=all` aborts the process. #775 caught
// exactly one of these — the NemotronH forward — because a doctest stub happened
// to reach it. The other 34 were latent by that same luck.
//
// This suite is the class gate the sweep owes. It does NOT try to be one test
// per site; it is one test per SHAPE, because the sweep's failure modes are
// shape-shaped, not site-shaped:
//
//   * a non-const `auto& x = static_cast<T&>(model)` forward   (Qwen3 dense)
//   * a `const auto& x = ...` forward, 14 of the 34 sites                (StableLM)
//   * an ALIAS FAMILY: two registered architectures sharing ONE factory, so
//     the entry point has no single registered name of its own. The refusal
//     must name the FAMILY PRIMARY (whose `load_weights` produces the model
//     type) and, separately, the architecture the PASSED model's registration
//     claims                                       (Olmo3/Olmo2, InternLM3/Llama)
//   * a `prepare` entry point rather than a forward   (Qwen3.5 MoE)
//   * the one site in the class that was NOT `(const )?auto& x = cast(model);`
//     but an INLINE member call on the cast expression itself
//     (`qwen3_vl_registry.cpp:105`), which is the site a mechanical
//     search-and-replace over the 33 uniform ones would miss   (Qwen3-VL)
//
// EVERY case here enters through the production seam — `reg.factory->forward` /
// `reg.factory->prepare`, which is exactly what `ModelRegistry::Forward` calls
// (`model_registry.cpp`). None of them constructs the concrete model by hand,
// because a test that does that proves the class works and never that anything
// reaches it.
//
// CPU-only, no checkpoint, no GPU: the refusal happens before any weight is
// touched, which is the entire point of moving the check to the top of the entry
// point.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
// The type-erased seam only forward-declares the types a CALL through it needs.
// `nemotron_h_registry.cpp:28` and `test_nemotron_h_scaffold.cpp` reach for the
// same header for the same reason.
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, *KvCache
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

using vllm::HfConfig;
using vllm::LoadedModel;
using vllm::ModelForwardInput;
using vllm::ModelRegistration;
using vllm::RegistrationFor;

namespace {

// A FOREIGN `LoadedModel`: complete, well-formed, and simply not any concrete
// model's type. This is the shape a caller produces by resolving one
// registration and handing `factory->forward` a model that some other path
// produced — which is what nothing in the type system prevents and what
// `ModelRegistry::Forward` cannot check on its callers' behalf.
//
// It carries a REAL registration (the entry point's own), so
// `model.registration().architecture` is right and only the OBJECT is wrong.
// That is deliberately the informative case rather than a strawman: a foreign
// object with a foreign registration would also be caught by a much weaker
// check that merely compared architecture strings, and such a check is
// explicitly rejected by the spec because it establishes what the registration
// CLAIMS rather than what the object IS.
class ForeignLoadedModel final : public LoadedModel {
 public:
  explicit ForeignLoadedModel(const ModelRegistration& registration)
      : LoadedModel(registration) {}
};

// One reusable, entirely empty forward input. Every entry point under test
// refuses before it reads a single field of it, and proving that is half of
// what these cases assert: a refusal that needed a valid config or a live queue
// would not be a refusal at the top of the entry point.
struct EmptyForwardInput {
  std::vector<int32_t> token_ids{0};
  std::vector<int32_t> positions{0};
  std::vector<int32_t> logits_indices{0};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  HfConfig config{};
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  ModelForwardInput Get() {
    return ModelForwardInput{.token_ids = token_ids,
                             .positions = positions,
                             .attn_meta = attn_meta,
                             .gdn_meta = gdn_meta,
                             .attn_kv = attn_kv,
                             .gdn_state = gdn_state,
                             .config = config,
                             .queue = queue,
                             .logits_indices = logits_indices,
                             .num_reqs = 1};
  }
};

// `registered` is the architecture whose registration the caller resolved.
// `refuser` is the architecture the entry point refuses UNDER — the family
// primary, which for a single-architecture registry is the same string and for
// an alias family is the architecture whose `load_weights` produces the model
// type the entry point opens.
void CheckForwardRefusesByName(const char* registered, const char* refuser) {
  const ModelRegistration& reg = RegistrationFor(registered);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->forward != nullptr);
  ForeignLoadedModel foreign(reg);
  EmptyForwardInput in;
  const ModelForwardInput input = in.Get();

  // Names the entry point that refused, so a mismatch reported from one of the
  // 35 near-identical registry entry points can be told from another's.
  CHECK_THROWS_WITH_AS(reg.factory->forward(foreign, input),
                       doctest::Contains(refuser), std::runtime_error);
  // ...and names what actually went wrong, rather than blaming the weights the
  // real object would have failed on next.
  CHECK_THROWS_WITH_AS(reg.factory->forward(foreign, input),
                       doctest::Contains("was not produced by"),
                       std::runtime_error);
  // ...and separately reports the architecture the PASSED model's registration
  // claims. On an alias family these two strings DIFFER, which is the whole
  // reason both are in the message.
  CHECK_THROWS_WITH_AS(reg.factory->forward(foreign, input),
                       doctest::Contains(registered), std::runtime_error);
}

void CheckPrepareRefusesByName(const char* registered, const char* refuser) {
  const ModelRegistration& reg = RegistrationFor(registered);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->prepare != nullptr);
  ForeignLoadedModel foreign(reg);
  const HfConfig config{};
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  CHECK_THROWS_WITH_AS(reg.factory->prepare(foreign, config, queue),
                       doctest::Contains(refuser), std::runtime_error);
  CHECK_THROWS_WITH_AS(reg.factory->prepare(foreign, config, queue),
                       doctest::Contains("was not produced by"),
                       std::runtime_error);
}

}  // namespace

// SHAPE 1 — the plain non-const forward, one registered architecture.
// `qwen3_dense.cpp:87`, `auto& qwen = static_cast<Qwen3DenseLoadedModel&>(...)`.
TEST_CASE("registry downcast: a NON-CONST forward REFUSES a foreign model") {
  CheckForwardRefusesByName("Qwen3ForCausalLM", "Qwen3ForCausalLM");
}

// SHAPE 2 — the `const auto&` forward. 14 of the 34 sites bind the cast to a
// `const auto&`. These are NOT the sites that select `ModelAs`'s const overload,
// which is worth stating because #847 assumed the opposite and called the sweep
// non-mechanical over it: `model` is a non-const `LoadedModel&` at every one of
// the 34, so overload resolution picks the NON-const `ModelAs` and its
// `Model&` result simply binds to the `const auto&`. The const overload has no
// caller in the tree at all (spec §4). What this case therefore proves is the
// thing that could actually break: that the returned reference is a real
// reference to the checked object rather than a copy or a dangling temporary,
// and that the site still refuses by name. `stablelm_registry.cpp:69`.
TEST_CASE("registry downcast: a CONST-bound forward REFUSES a foreign model") {
  CheckForwardRefusesByName("StableLmForCausalLM", "StableLmForCausalLM");
}

// SHAPE 3 — an ALIAS FAMILY, const arm. `Olmo2ForCausalLM` and
// `Olmo3ForCausalLM` are two REGISTER_VLLM_MODEL lines against ONE
// `kOlmo2Factory`, so `ForwardOlmo2ForCausalLM` has no single registered name.
// The sweep's decision: refuse under the FAMILY PRIMARY. Asserting BOTH strings
// is what makes that decision executable — the message must name Olmo2 as the
// refuser and Olmo3 as what the passed model's registration claimed.
// `olmo2_registry.cpp:74`.
TEST_CASE("registry downcast: an ALIAS arch refuses under the FAMILY PRIMARY") {
  CheckForwardRefusesByName("Olmo3ForCausalLM", "Olmo2ForCausalLM");
}

// SHAPE 3b — the same alias decision on a NON-const site, and on a family whose
// two architectures are not near-spellings of each other: `InternLM3ForCausalLM`
// is an alias-only registration against `kLlamaFactory`.
// `llama_registry.cpp:103`.
TEST_CASE("registry downcast: a non-const ALIAS forward names both archs") {
  CheckForwardRefusesByName("InternLM3ForCausalLM", "LlamaForCausalLM");
}

// SHAPE 4 — a `prepare` entry point, and its `forward` sibling in the same TU.
// `prepare` runs once at load; the cast there is the same promise, and it is
// made BEFORE any weight is materialized, so nothing downstream would catch it.
// `qwen3_5_moe.cpp:91` (prepare) and `:97` (forward), reached through the
// text-only alias registration so the alias arm is covered on `prepare` too.
TEST_CASE("registry downcast: a PREPARE entry point REFUSES a foreign model") {
  CheckPrepareRefusesByName("Qwen3_5MoeForCausalLM",
                            "Qwen3_5MoeForConditionalGeneration");
  CheckForwardRefusesByName("Qwen3_5MoeForCausalLM",
                            "Qwen3_5MoeForConditionalGeneration");
}

// SHAPE 5 — THE ODD SITE. Every other site in the class is
// `(const )?auto& x = static_cast<T&>(model);`. This one is an inline member
// call on the cast expression itself:
//
//     static_cast<Qwen3VLLoadedModel&>(model).CosSinCache(queue, config);
//
// It is the site a uniform search-and-replace over the other 33 would step past,
// and it is also the site where the UB is most immediate: there is no
// intervening binding at all, the member call happens ON the cast expression.
// `qwen3_vl_registry.cpp:105`.
TEST_CASE("registry downcast: the INLINE-call prepare site REFUSES too") {
  CheckPrepareRefusesByName("Qwen3VLForConditionalGeneration",
                            "Qwen3VLForConditionalGeneration");
}
