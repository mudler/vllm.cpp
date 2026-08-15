// Ported from: vllm/model_executor/offloader/base.py @ 555967922
//              (BaseOffloader:46-92; NoopOffloader:94-103;
//               _instance/get_offloader/set_offloader:106-125;
//               create_offloader:126-162).
//
// Scope (ENG-WEIGHT-OFFLOAD W1, spec .agents/specs/weight-offload-uva.md,
// issue #797): the offloader SEAM only. This file defines the interface, the
// no-op default, the process-global instance, and the factory that reads an
// `OffloadConfig`. No backend moves a weight. The UVA backend is W2 and the
// prefetch backend is W5.
//
// Upstream provenance note: vLLM's own base.py header records that it was
// adapted from sgl-project/sglang `python/sglang/srt/utils/offloader.py`. The
// lineage is SGLang to vLLM to us. vLLM is the mirror source.
//
// DEVIATION, and the reason W1 needed a decision (recorded here because the
// spec's W1 row assumed a seam this tree does not have):
//
//   Upstream installs the offloader in ONE place, `make_layers`
//   (model_executor/models/utils.py:816,824), which every model calls, so a
//   single `get_offloader().wrap_modules(...)` covers every architecture.
//
//   We have NO `make_layers`. Twenty-eight model headers each declare their own
//   `std::vector<...LayerWeights> layers`, so there is no per-layer chokepoint
//   to wrap. What we do have is a type-erased model seam every production load
//   goes through: `ModelRegistry::Load` and `ModelRegistry::Prepare`
//   (models/model_registry.h:366). `Prepare` is where a model materialises its
//   resident weights, which is the decision an offloader must influence, so
//   `Prepare` is our analogue of the wrap site.
//
//   The GLOBAL instance is kept exactly as upstream has it, and for the same
//   reason: it lets the engine choose a backend at construction without
//   threading an `OffloadConfig` through `GPUModelRunner`'s constructors. The
//   engine sets it (mirroring gpu_model_runner.py:939) and `Prepare` reads it
//   (mirroring models/utils.py:824).
#ifndef VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOADER_H_
#define VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOADER_H_

#include <memory>
#include <string>
#include <string_view>

#include "vllm/config/offload.h"
#include "vllm/model_executor/weight_offload_policy.h"

namespace vllm {

class LoadedModel;

// Upstream: class BaseOffloader (offloader/base.py:46-92).
class WeightOffloader {
 public:
  virtual ~WeightOffloader() = default;

  // THE DECISION SEAM. A loader asks this for each weight it is about to
  // materialise, and keeps the weight off the device when the answer is
  // kOffload. This is the one place that answers "is this weight offloaded",
  // and it is asked DURING loading rather than after, because a materialised
  // weight has already paid the allocation the feature exists to avoid.
  //
  // `canonical_name` must be the dotted parameter name upstream matches against
  // (`mlp.experts.w2_weight`), not a format-specific tensor name. See
  // weight_offload_policy.h for why the caller owns that conversion.
  //
  // Upstream has no equivalent call because it rewrites parameter storage in
  // `wrap_modules` after construction; the recorded reason we cannot is in the
  // spec's port map.
  virtual WeightOffloadDecision ConsiderWeight(const std::string& canonical_name,
                                               int64_t bytes) = 0;

  // Upstream: post_init (base.py:68-76). Called once after the model is built
  // and prepared, so a backend can finalise storage, report a total, or start a
  // first prefetch. This REPLACES the earlier `PrepareModel` hook, which was a
  // port of `wrap_modules` and became a lie the moment the decision moved into
  // the loaders: nothing could implement it meaningfully, and it duplicated
  // this one.
  virtual void OnModelPrepared(LoadedModel& model) { (void)model; }

  // Upstream: sync_prev_onload / join_after_forward (base.py:79-85). vLLM calls
  // these around CUDA-graph capture and replay
  // (compilation/cuda_graph.py:310,324,359 and
  // compilation/breakable_cudagraph.py:379,387,421). They are declared now so
  // the graph seam has a name to call; the no-op makes them free.
  virtual void SyncPrevOnload() {}
  virtual void JoinAfterForward() {}

  // Name for logs and tests. Upstream logs `type(instance).__name__`
  // (base.py:118-125).
  virtual const char* name() const = 0;

  // Bytes this offloader has kept off the device so far. Upstream reports the
  // same total once at the end of wrap_modules (uva.py:57-61).
  virtual int64_t offloaded_bytes() const = 0;

  // How many weights this offloader was ASKED about. It is the evidence that a
  // loader which declares `supports_weight_offload` actually consulted the
  // seam. A model can declare the capability and then never ask, and without
  // this counter that lie is invisible: the run simply offloads nothing.
  virtual int64_t weights_considered() const = 0;

  virtual bool moves_weights() const = 0;
};

// Upstream: class NoopOffloader (offloader/base.py:94-103). Returns the model
// unchanged, which is the current engine path.
class NoopWeightOffloader final : public WeightOffloader {
 public:
  // Every weight stays resident, so the engine's existing path is unchanged BY
  // CONSTRUCTION rather than by a flag a caller has to remember to check.
  WeightOffloadDecision ConsiderWeight(const std::string&, int64_t) override {
    ++considered_;
    return WeightOffloadDecision::kBudgetExhausted;
  }
  const char* name() const override { return "NoopWeightOffloader"; }
  int64_t offloaded_bytes() const override { return 0; }
  int64_t weights_considered() const override { return considered_; }
  bool moves_weights() const override { return false; }

 private:
  int64_t considered_ = 0;
};

// Upstream: class UVAOffloader (offloader/uva.py:21-137). The concrete arm that
// keeps a weight off the device under a byte budget.
//
// WHAT THIS CLASS DOES AND DOES NOT DO. It answers the DECISION, delegating to
// the `WeightOffloadPolicy` it owns, and it counts the bytes it has approved.
// It does not copy, pin, or map anything: the caller that asked keeps the
// weight in host memory instead of materialising it on the device, so the move
// is an ABSENCE of work rather than work. That is the whole reason the decision
// had to move into the loaders (see weight_offload_policy.h).
//
// The pinned-host-copy and device-view halves of upstream's UVA arm
// (uva.py:97-105) have no equivalent yet and are NOT silently skipped: they
// belong to the loader-side application, which owns the buffer.
class UvaWeightOffloader final : public WeightOffloader {
 public:
  explicit UvaWeightOffloader(WeightOffloadPolicy policy)
      : policy_(std::move(policy)) {}

  WeightOffloadDecision ConsiderWeight(const std::string& canonical_name,
                                       int64_t bytes) override {
    ++considered_;
    return policy_.Consider(canonical_name, bytes);
  }
  const char* name() const override { return "UvaWeightOffloader"; }
  int64_t offloaded_bytes() const override { return policy_.offloaded_bytes(); }
  int64_t weights_considered() const override { return considered_; }
  bool moves_weights() const override { return policy_.active(); }

  // Upstream logs the total once when wrap_modules finishes (uva.py:57-61). We
  // have no single finish point, so the engine reports it here after the model
  // is prepared.
  int64_t max_bytes() const { return policy_.max_bytes(); }

 private:
  WeightOffloadPolicy policy_;
  int64_t considered_ = 0;
};

// Upstream: get_offloader / set_offloader and the module-global `_instance`
// (offloader/base.py:106-125). The default is the no-op, so an engine that
// never calls the setter behaves exactly as it does today.
WeightOffloader& GetWeightOffloader();

// Installs `instance`. A null pointer restores the no-op default, which is what
// makes the seam safe to reset between tests. Returns the previous instance so
// a caller can restore it.
std::unique_ptr<WeightOffloader> SetWeightOffloader(
    std::unique_ptr<WeightOffloader> instance);

// Upstream: create_offloader (offloader/base.py:126-162). Resolves the config's
// backend and returns the offloader for it.
//
// W1 STATE, stated plainly because silence here is a memory bug the user cannot
// see: no backend is implemented yet, so this returns the no-op for EVERY
// config. When the config selects a backend that would move weights, the
// returned `selected_backend_pending` is set to that backend's name so the
// caller can say so once, rather than accepting a `cpu_offload_gb` and doing
// nothing without a word. W2 gives `uva` a real offloader and W5 gives
// `prefetch` one.
struct WeightOffloaderChoice {
  std::unique_ptr<WeightOffloader> offloader;
  // Empty when nothing was requested. Otherwise the backend name the config
  // resolved to but which this build cannot yet honour.
  std::string selected_backend_pending;
};
WeightOffloaderChoice CreateWeightOffloader(const OffloadConfig& config);

// THE TOTALITY GUARD. Throws std::invalid_argument when `config` would offload
// a weight but `supports_weight_offload` is false for the resolved model.
//
// This exists because there is no single upload seam to enforce the obligation
// structurally: each model allocates its own device buffers. Without the guard,
// a model nobody wired accepts a budget and keeps every weight on the device,
// with no error anywhere. Refusing names the architecture, which is what turns
// an invisible memory bug into a message the operator can act on.
//
// Mirrors the project rule that an unimplemented arm is refused with a message
// naming the missing piece, rather than silently doing nothing.
void RefuseUnsupportedWeightOffload(const OffloadConfig& config,
                                    std::string_view architecture,
                                    bool supports_weight_offload);

// THE SECOND HALF OF THE GUARD, checked after the model is loaded. A model can
// declare `supports_weight_offload` and then never call `ConsiderWeight`, and
// the first guard cannot see that lie: the run just offloads nothing. Throws
// std::invalid_argument when the offloader would move weights, the model
// declared support, and it was asked about ZERO weights.
//
// Zero is the only value that proves a defect. Any other count means the loader
// consulted the seam, and whether it asked about every weight is the loader's
// own test to make.
void VerifyWeightOffloadWasConsulted(const WeightOffloader& offloader,
                                     std::string_view architecture,
                                     bool supports_weight_offload);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOADER_H_
