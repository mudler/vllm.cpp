// Ported from: vllm/model_executor/offloader/uva.py:64-107 @ 555967922
//              (UVAOffloader._maybe_offload_to_cpu).
//
// Scope (ENG-WEIGHT-OFFLOAD W2a, spec .agents/specs/weight-offload-uva.md,
// issue #797): the offload DECISION only. Given a weight's canonical name and
// its size, this answers whether that weight is offloaded, and it keeps the
// running byte budget. It moves nothing and it knows nothing about a device.
//
// WHY THE DECISION IS SPLIT FROM THE MOVE, and why the move is NOT in W1's
// `PrepareModel` hook:
//
//   Upstream offloads by walking a constructed model
//   (`module.named_parameters()`), which is safe there because PyTorch builds
//   modules on the meta device and materialises them during load, so
//   `wrap_modules` runs BEFORE the real allocation.
//
//   Our loaders materialise a weight directly. By the time a `LoadedModel`
//   exists, the device copy is already allocated, so offloading at that point
//   would allocate and then move back. That pays the exact peak the feature
//   exists to avoid, and on a memory-constrained device it can fail before it
//   helps.
//
//   So the decision must be asked DURING loading, next to the residency policy
//   that already exists there (`GgufKeepQuantPolicy::Route`, which returns a
//   `GgufResidency` per tensor). This header is the piece both the loader-side
//   application and any future model-side application need, so it lands first
//   and alone.
//
// KNOWN COST, recorded rather than discovered later: upstream matches DOTTED
// PARAMETER names such as `mlp.experts.w2_weight`. Loader-side names differ by
// format: GGUF uses `blk.0.ffn_gate_exps`, safetensors uses
// `model.layers.0.mlp.experts...`. A caller must therefore pass a CANONICAL
// name, or upstream's targeting semantics do not transfer. This class does not
// canonicalise; it documents the requirement and the caller owns it.
#ifndef VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOAD_POLICY_H_
#define VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOAD_POLICY_H_

#include <cstdint>
#include <set>
#include <string>

#include "vllm/config/offload.h"

namespace vllm {

// The three outcomes upstream's loop produces, named so a caller cannot confuse
// them. The distinction is load-bearing: uva.py:80-84 BREAKS on an exhausted
// budget, and uva.py:94-95 CONTINUES on an untargeted parameter.
enum class WeightOffloadDecision {
  // Offload this weight. The caller keeps it off the device.
  kOffload,
  // `cpu_offload_params` is non-empty and this name does not match it. Upstream
  // `continue`s: no budget is consumed and later weights are still considered.
  kNotTargeted,
  // The byte budget is spent. Upstream `break`s out of the module and every
  // later module early-returns, so no further weight is offloaded.
  kBudgetExhausted,
};

// Mirrors the budget and targeting half of UVAOffloader. Stateful: the running
// total advances only on kOffload.
class WeightOffloadPolicy {
 public:
  WeightOffloadPolicy(int64_t max_bytes, std::set<std::string> target_segments);

  // Builds the policy for `config`'s resolved backend. A config that offloads
  // nothing yields a policy whose budget is zero, so every Consider returns
  // kBudgetExhausted and the caller's behaviour is unchanged.
  static WeightOffloadPolicy FromConfig(const OffloadConfig& config);

  // The decision for one weight. `canonical_name` must be the dotted parameter
  // name (see the header note on canonicalisation). `bytes` is the weight's
  // size; upstream counts `numel * element_size` AFTER the move (uva.py:107),
  // which for our purposes is the same count either side.
  //
  // Advances the running total by `bytes` on kOffload, and only then.
  WeightOffloadDecision Consider(const std::string& canonical_name,
                                 int64_t bytes);

  int64_t offloaded_bytes() const { return offloaded_bytes_; }
  int64_t max_bytes() const { return max_bytes_; }

  // True when the budget is spent. uva.py:74-75 early-returns a whole module on
  // this, so a caller can skip work rather than call Consider per weight.
  bool exhausted() const { return offloaded_bytes_ >= max_bytes_; }

  // True when this policy can ever offload. False for a zero budget, which
  // lets a caller take its existing path with no per-weight cost at all.
  bool active() const { return max_bytes_ > 0; }

 private:
  int64_t max_bytes_ = 0;
  int64_t offloaded_bytes_ = 0;
  std::set<std::string> target_segments_;
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOAD_POLICY_H_
