// `ENG-HYBRID-PLACEMENT` W2 (issue #2023) — the resolved answer to "which device
// runs this tensor".
//
// WHAT THIS IS NOT. It is not a sharding concept. `TensorParallel` / `ShardRange`
// (models/tensor_parallel.h) already say WHICH SLICE of a tensor a rank owns;
// this says WHICH DEVICE a group runs on. The two are orthogonal axes and a group
// can carry both, which is why nothing here touches a dimension, a rank or a
// world size. The spec makes a review against that header W2's gate rather than a
// suggestion, because the easy failure is to grow a second sharding concept here.
//
// THE MECHANISM IS llama.cpp's, at pin `b10451`. Compute follows the tensor's
// buffer there, so a tensor-name pattern mapped to a buffer type IS the placement
// decision; there is no separate compute dispatch. We keep the same shape with a
// `vt::DeviceType` where upstream has a `ggml_backend_buffer_type_t`.
//
// TWO SEMANTICS THAT A RE-IMPLEMENTATION GETS WRONG BY DEFAULT, both transcribed
// from `src/llama-model-loader.cpp:1178-1184`:
//
//   1. The scan is FIRST-MATCH-WINS, front to back, and stops at the first hit.
//      Order is therefore part of the operator's input, not an implementation
//      detail, and a later broad pattern cannot override an earlier narrow one.
//      Never sort the list.
//   2. The match is `regex_search`, NOT a full match. That is what makes an
//      unanchored `\.ffn_up_exps` reach every layer while an anchored `blk\.7\.`
//      one reaches exactly one, and it is the whole reason `-ncmoe N` can be
//      expressed as N anchored patterns.
//
// COMPILED ONCE. The regexes are built when the placement is, not per tensor. A
// `std::regex` constructed inside the name loop would put regex compilation on the
// load path once per tensor per override, which for a 40-layer `-ncmoe` over a
// few thousand tensors is six figures of compilations.
#ifndef VLLM_MODEL_EXECUTOR_DEVICE_PLACEMENT_H_
#define VLLM_MODEL_EXECUTOR_DEVICE_PLACEMENT_H_

#include <cstddef>
#include <regex>
#include <string>
#include <vector>

#include "vllm/config/weight_residency.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace vllm {

class DevicePlacement {
 public:
  // The inert placement: every tensor runs on `engine_device`. This is what a
  // load with no placement configured builds, and `IsTrivial()` is true of it.
  explicit DevicePlacement(vt::DeviceType engine_device = vt::DeviceType::kCPU)
      : engine_device_(engine_device) {}

  // Build from a resolved, desugared override list — the output of
  // `ResolvePlacementOverrides()`, which has already applied
  // environment-over-config precedence and expanded `cpu_moe` / `n_cpu_moe`.
  //
  // THROWS `std::invalid_argument` on a pattern that does not compile or a device
  // name that is not one. Both are refused by W1's parser at startup, so reaching
  // either here means a caller built a list by hand; failing loudly beats a
  // placement that silently drops an entry the operator asked for.
  static DevicePlacement FromOverrides(
      const std::vector<PlacementOverride>& overrides,
      vt::DeviceType engine_device);

  // The device this tensor runs on. First-match-wins over the override list, then
  // the engine's own device. Pure and allocation-free apart from the regex match.
  vt::DeviceType DeviceFor(const std::string& tensor_name) const;

  // True when NOTHING is placed away from the engine's device, so the engine must
  // take its existing single-device path with no added indirection. Two ways to
  // be trivial and both count: no overrides at all, and overrides that every one
  // name the engine's own device. The second is not a curiosity — `cpu_moe` on a
  // CPU engine is exactly that, and it is what a user gets for pasting a llama.cpp
  // command line at a CPU build.
  bool IsTrivial() const { return trivial_; }

  size_t override_count() const { return compiled_.size(); }
  vt::DeviceType engine_device() const { return engine_device_; }

  // One line for the install log: what was resolved, and how many tensors it can
  // reach. Empty when trivial, because a line about a placement that changes
  // nothing is noise an operator learns to skip past.
  std::string Describe() const;

 private:
  struct Compiled {
    std::regex re;
    std::string pattern;
    vt::DeviceType device;
  };

  std::vector<Compiled> compiled_;
  vt::DeviceType engine_device_ = vt::DeviceType::kCPU;
  bool trivial_ = true;
};

// The per-LAYER decision, resolved once from a `DevicePlacement` at model build.
//
// WHY A SECOND TYPE. `DevicePlacement` answers by tensor NAME, which is the form
// llama.cpp's mechanism takes and the form an operator writes. The forward does
// not have tensor names: it has a layer index and a weight struct. Resolving the
// names to a per-layer answer once, at build, is what the spec means by
// "resolved ONCE at model build into a per-tensor-group device assignment. The
// forward reads that assignment; it never re-decides per step."
//
// THE NAMES ARE llama.cpp's GGUF SPELLING, because that is what the patterns are
// written against — `blk.<l>.ffn_{up,down,gate}_exps.weight`. An operator who
// pastes a `-ncmoe 40` command line is writing patterns for those names, so
// asking any other spelling would silently match nothing.
class MoePlacementPlan {
 public:
  MoePlacementPlan() = default;

  // Ask the placement about every layer's routed-expert tensors.
  //
  // THROWS `std::invalid_argument` on a PARTIAL placement — one layer whose
  // gate, up and down tensors do not all resolve to the same device. That is a
  // legal thing to write with `-ot` and a thing this row does not implement: the
  // MoE block runs one grouped GEMM over the three, so splitting them across
  // devices is not a scheduling decision but a different kernel. Refusing by name
  // is what `AGENTS.md` requires of an unimplemented arm, and the alternative —
  // picking one of the three — would place weights the operator did not ask to
  // place and say nothing.
  static MoePlacementPlan Resolve(const DevicePlacement& placement,
                                  int64_t num_hidden_layers);

  // The device layer `l`'s routed experts run on. Out-of-range answers the
  // engine device, so a caller that asks about a layer a model does not have
  // gets the inert answer rather than an exception on the decode path.
  vt::DeviceType DeviceForLayer(int64_t l) const;

  // True when at least one layer runs its experts somewhere other than the
  // engine device. The forward reads this to decide whether any of the placement
  // machinery runs at all, so an unplaced model pays nothing.
  bool PlacesAnything() const { return placed_ > 0; }

  int64_t placed_layer_count() const { return placed_; }
  vt::DeviceType engine_device() const { return engine_device_; }

  // One line for the install log, empty when nothing is placed.
  std::string Describe() const;

 private:
  std::vector<vt::DeviceType> per_layer_;
  vt::DeviceType engine_device_ = vt::DeviceType::kCPU;
  int64_t placed_ = 0;
};

// The three routed-expert tensor names llama.cpp's GGUF export uses for a layer,
// which are what a `-cmoe` / `-ncmoe` / `-ot` pattern is written against. Exposed
// so a test asserts the same strings the resolver asks about rather than a copy
// of them.
std::vector<std::string> RoutedExpertTensorNames(int64_t layer);

// ── The placement device's queue, and the active plan ────────────────────────

// The queue a placed group runs on. One per device type, created on FIRST USE and
// reused for the process's life, because a queue created per layer per token
// would dominate the round trip it exists to serve.
//
// NOT destroyed. `Backend::DestroyQueue` is a no-op on CPU, which is the only
// placement target this row ships, and a process-lifetime queue on an
// accelerator would leak a stream — so this refuses a target whose backend needs
// its queues destroyed rather than leaking one quietly. When that target becomes
// real it needs an owner with a lifetime, not a wider static.
vt::Queue& PlacementQueue(vt::DeviceType device);

// The plan the forward reads, installed once at model build. A process-global for
// the same reason `ActiveWeightResidencyConfig` is one: the forward is reached
// through several model entry points that do not share a config parameter, and
// threading one through every architecture's signature to serve a feature most
// loads do not use would be the wider change.
void SetActiveMoePlacementPlan(const MoePlacementPlan& plan);
const MoePlacementPlan& ActiveMoePlacementPlan();
void ResetActiveMoePlacementPlanForTesting();

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_DEVICE_PLACEMENT_H_
