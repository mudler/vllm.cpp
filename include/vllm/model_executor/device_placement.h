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

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_DEVICE_PLACEMENT_H_
