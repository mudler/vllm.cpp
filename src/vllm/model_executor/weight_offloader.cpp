// Ported from: vllm/model_executor/offloader/base.py:106-162 @ 555967922.
#include "vllm/model_executor/weight_offloader.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace vllm {
namespace {

// Upstream: the module-global `_instance`, defaulted to NoopOffloader
// (offloader/base.py:106-107). A function-local static gives us the same
// "default is the no-op" property with defined initialisation order.
std::unique_ptr<WeightOffloader>& InstanceSlot() {
  static std::unique_ptr<WeightOffloader> instance =
      std::make_unique<NoopWeightOffloader>();
  return instance;
}

}  // namespace

WeightOffloader& GetWeightOffloader() {
  // NO self-healing fallback here, deliberately. The slot is non-null by two
  // constructions: the static initialiser installs the no-op, and
  // SetWeightOffloader replaces a null argument with the no-op. A `if (slot ==
  // nullptr) slot = make_unique<Noop>()` line here would be dead code, and
  // mutation testing proved it is worse than dead: it MASKED both invariants,
  // so a mutant that nulled the initialiser and a mutant that let a null
  // install empty the slot both still passed. The getter must not repair what
  // the tests exist to detect.
  return *InstanceSlot();
}

std::unique_ptr<WeightOffloader> SetWeightOffloader(
    std::unique_ptr<WeightOffloader> instance) {
  std::unique_ptr<WeightOffloader>& slot = InstanceSlot();
  std::unique_ptr<WeightOffloader> previous = std::move(slot);
  slot = instance != nullptr ? std::move(instance)
                             : std::make_unique<NoopWeightOffloader>();
  return previous;
}

WeightOffloaderChoice CreateWeightOffloader(const OffloadConfig& config) {
  WeightOffloaderChoice choice;
  choice.offloader = std::make_unique<NoopWeightOffloader>();

  // Upstream resolves the backend first (base.py:139-149) and only then
  // constructs. We reuse the config's own resolver so the selection ORDER lives
  // in exactly one place: under "auto" it is prefetch when
  // offload_group_size > 0, else uva when cpu_offload_gb > 0, else nothing.
  if (!config.is_offloading_enabled()) return choice;

  const std::optional<OffloadBackend> backend = config.ResolvedBackend();
  if (!backend.has_value()) return choice;

  // W2b: the UVA arm exists. It answers the decision under its byte budget; the
  // loader that asks is W2c, so today it is installed and never consulted.
  if (*backend == OffloadBackend::kUva) {
    choice.offloader = std::make_unique<UvaWeightOffloader>(
        WeightOffloadPolicy::FromConfig(config));
    return choice;
  }

  // Prefetch is W5. Record which backend was asked for so the caller can report
  // it once. Returning the no-op silently would make a configured budget look
  // honoured.
  choice.selected_backend_pending = offload_backend_str(*backend);
  return choice;
}

void RefuseUnsupportedWeightOffload(const OffloadConfig& config,
                                    std::string_view architecture,
                                    bool supports_weight_offload) {
  if (!config.is_offloading_enabled()) return;
  if (supports_weight_offload) return;
  throw std::invalid_argument(
      "weight offload is configured but architecture \"" +
      std::string(architecture) +
      "\" does not support it: its loader does not consult the weight "
      "offloader, so every weight would stay on the device and the budget "
      "would free nothing (ENG-WEIGHT-OFFLOAD W2c). Remove --offload-config, "
      "or wire this model's loader.");
}

void VerifyWeightOffloadWasConsulted(const WeightOffloader& offloader,
                                     std::string_view architecture,
                                     bool supports_weight_offload) {
  if (!offloader.moves_weights()) return;
  if (!supports_weight_offload) return;
  if (offloader.weights_considered() > 0) return;
  throw std::invalid_argument(
      "architecture \"" + std::string(architecture) +
      "\" declares supports_weight_offload but its loader asked the weight "
      "offloader about ZERO weights, so the configured budget freed nothing. "
      "This is a defect in that loader, not a configuration error.");
}

}  // namespace vllm
