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

#include "vllm/config/offload.h"

namespace vllm {

class LoadedModel;

// Upstream: class BaseOffloader (offloader/base.py:46-92).
class WeightOffloader {
 public:
  virtual ~WeightOffloader() = default;

  // The wrap-site analogue. Upstream's `wrap_modules` receives the layer
  // modules and rewrites each parameter's storage; ours receives the loaded
  // model at the point it materialises resident weights. A backend that
  // offloads acts here. The no-op returns without touching the model.
  virtual void PrepareModel(LoadedModel& model) = 0;

  // Upstream: post_init (base.py:68-76). Called after model construction
  // completes so a backend can finalise storage or start a first prefetch.
  virtual void PostInit() {}

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

  // True when this offloader can move a weight. False for the no-op and for a
  // backend that is selected but not yet implemented. The engine reports the
  // difference rather than leaving a configured budget silently inert.
  virtual bool moves_weights() const = 0;
};

// Upstream: class NoopOffloader (offloader/base.py:94-103). Returns the model
// unchanged, which is the current engine path.
class NoopWeightOffloader final : public WeightOffloader {
 public:
  void PrepareModel(LoadedModel&) override {}
  const char* name() const override { return "NoopWeightOffloader"; }
  bool moves_weights() const override { return false; }
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

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_WEIGHT_OFFLOADER_H_
