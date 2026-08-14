// Ported from: vllm/model_executor/offloader/base.py:106-162 @ 555967922.
#include "vllm/model_executor/weight_offloader.h"

#include <memory>
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

  // W1: neither backend exists yet. Record which one was asked for so the
  // caller can report it once. Returning the no-op silently would make a
  // configured budget look honoured.
  choice.selected_backend_pending = offload_backend_str(*backend);
  return choice;
}

}  // namespace vllm
