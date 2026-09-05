// Out-of-core plugin fixture for the ENG-PLUGIN-SYSTEM extensibility proof.
//
// Mirrors tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/__init__.py
// (@ 555967922): a plugin whose `register()` calls
// `ModelRegistry.register_model("MyOPTForCausalLM", ...)`. Here the C++ analog
// registers a TOY model architecture through the PUBLIC `vllm::RegisterModel`
// seam and publishes its `register()` callback through the general-plugin seam
// via REGISTER_VLLM_GENERAL_PLUGIN — proving the engine picks up a new arch from
// an out-of-core translation unit with ZERO edit to any engine-core file.
//
// This TU is compiled ONLY into the test_plugin_system executable, NOT the
// vllm::vllm library, so the toy arch never pollutes the shared model registry
// the other suites count (test_model_registry.cpp requires exactly 28 archs).
#include <memory>
#include <stdexcept>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits (complete type)
#include "vllm/plugins/plugins.h"

namespace {

// Sentinels the test reads to prove the plugin ran exactly when expected.
int g_toy_register_calls = 0;
bool g_broken_plugin_ran = false;

constexpr std::string_view kToyArch = "ToyPluginForCausalLM";

void ToyParseConfig(const vllm::HfConfig& /*config*/) {}

// Minimal concrete model so the factory's load_weights is a real (non-null)
// entry, matching the "complete factory" contract other archs satisfy.
class ToyLoadedModel : public vllm::LoadedModel {
 public:
  explicit ToyLoadedModel(const vllm::ModelRegistration& registration)
      : vllm::LoadedModel(registration) {}
};

std::unique_ptr<vllm::LoadedModel> ToyLoadWeights(
    const vllm::ModelRegistration& registration, const vllm::HfConfig& /*config*/,
    const vllm::ModelSource& /*source*/) {
  return std::make_unique<ToyLoadedModel>(registration);
}

void ToyPrepare(vllm::LoadedModel& /*model*/, const vllm::HfConfig& /*config*/,
                vt::Queue& /*queue*/) {}

vllm::ForwardLogits ToyForward(vllm::LoadedModel& /*model*/,
                               const vllm::ModelForwardInput& /*input*/) {
  return vllm::ForwardLogits{};
}

vllm::v1::KVCacheConfig ToyMakeKVCache(const vllm::HfConfig& /*config*/,
                                       int /*block_size*/, int /*num_blocks*/) {
  return vllm::v1::KVCacheConfig{};
}

// DESIGNATED, like the 36 other `ModelFactory` initializers in the tree (every
// one of them under src/vllm/model_executor/models/). This site used
// POSITIONAL `/*field=*/value` comments and #2379 inserted three function
// pointers (`encode_mm`, `embed_mm`, `mrope_prompt_positions`) ahead of
// `is_dense_model`, so `true` landed in `encode_mm` and every CPU CI lane failed
// to compile. It was caught only because `bool` does not convert to a function
// pointer; two adjacent same-typed fields would have MISASSIGNED SILENTLY.
constexpr vllm::ModelFactory kToyFactory{
    .parse_config = ToyParseConfig,
    .load_weights = ToyLoadWeights,
    .prepare = ToyPrepare,
    .forward = ToyForward,
    .make_kv_cache = ToyMakeKVCache,
    .is_dense_model = true,
};

constexpr vllm::ModelInfo kToyInfo{
    /*is_text_generation_model=*/true,
};

// The plugin's register() callback. Idempotent (mirror of the dummy plugin's
// `if "MyOPTForCausalLM" not in get_supported_archs()` guard): register the toy
// arch only once even if invoked twice.
void RegisterToyModel() {
  ++g_toy_register_calls;
  for (const vllm::ModelRegistration& r : vllm::ModelRegistry::Registrations()) {
    if (r.architecture == kToyArch) return;  // already registered
  }
  vllm::RegisterModel(
      vllm::ModelRegistration{kToyArch, &kToyFactory, kToyInfo});
}

// A second general plugin that throws — proves LoadGeneralPlugins isolates a
// broken plugin (logs + skips) without aborting the good one or the engine.
void RegisterBrokenPlugin() {
  g_broken_plugin_ran = true;
  throw std::runtime_error("toy broken plugin");
}

}  // namespace

// Accessors for the test (defined in this TU so the sentinels stay TU-local).
namespace toy_plugin {
int ToyRegisterCalls() { return g_toy_register_calls; }
bool BrokenPluginRan() { return g_broken_plugin_ran; }
std::string_view ToyArch() { return kToyArch; }
}  // namespace toy_plugin

// Self-register both plugins from this out-of-core TU. Registration order across
// TUs is unspecified, but resolution is order-independent. Names are the
// VLLM_PLUGINS allowlist keys.
REGISTER_VLLM_GENERAL_PLUGIN(toy_model, "register_toy_model", RegisterToyModel)
REGISTER_VLLM_GENERAL_PLUGIN(toy_broken, "register_broken_plugin",
                             RegisterBrokenPlugin)
