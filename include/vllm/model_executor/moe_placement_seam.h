// `ENG-HYBRID-PLACEMENT` W3c — the ONE seam every architecture routes its
// routed-expert compute through, so placement is wired once instead of per model.
//
// WHY THIS EXISTS. W3b wired placement into `qwen3_moe.cpp` by hand. That is the
// parallel path `AGENTS.md` forbids: a second architecture would have copied the
// round trip, a third would have copied it again, and the first one to copy it
// slightly wrong would have been a silent divergence. This file is the shared
// seam that rule asks for.
//
// WHAT MAKES IT POSSIBLE. Every MoE block in this tree already has the same
// shape, which is a fact about the code rather than a convention this file
// invents:
//
//   MoeBlock              (Dev, MoeBlockWeights,     HfConfig,        dh, T) -> DBuf
//   NemotronHMoeBlockDevice(Dev, NemotronHMoeWeights, NemotronHParams, dh, T) -> DBuf
//
// Same `Dev`, same `[T,H]` bf16 hidden state in, same owning `[T,H]` out. Only
// the weight and parameter types differ, and a template closes over those at the
// call site. So the seam needs no common weights interface and no virtual
// dispatch — the thing that varies is exactly the thing a lambda captures.
//
// THE CALL SHAPE, one line per architecture:
//
//   DBuf out = RunMoePlaced(d, layer_index, dh, T, H,
//                           [&](Dev p, const Tensor& h) {
//                             return ItsOwnMoeBlock(p, w, params, h, T);
//                           });
//
// INERTNESS IS BY CONSTRUCTION, not by a flag. When the plan places nothing —
// which is every load that configured no placement — this calls `body(engine,
// dh)` and returns its result. No copy, no allocation, no branch beyond one
// comparison, and the value is the value the unplaced path produced.
#ifndef VLLM_MODEL_EXECUTOR_MOE_PLACEMENT_SEAM_H_
#define VLLM_MODEL_EXECUTOR_MOE_PLACEMENT_SEAM_H_

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm {

// The owning result of a routed-expert block: a `[T,H]` bf16 view plus the
// shared_ptr that returns its pool block when the last reference drops. This is
// the shape `MoeBlockOutput` already has and the shape a `DBuf` converts to with
// `ReleaseShared`, so every architecture reaches it in one line without a common
// weights interface or a virtual call.
struct MoePlacedOutput {
  vt::Tensor tensor;
  std::shared_ptr<void> storage;
};

// Run one layer's routed-expert block, on the placement device when this layer
// is placed and on the engine's own device otherwise.
//
// `body` is `(dense_attn::Dev, const vt::Tensor&) -> MoePlacedOutput`, and it
// receives the device it must compute on. It must NOT capture the engine `Dev`
// and use that instead: the whole point is that the block runs where it is told.
template <class Body>
MoePlacedOutput RunMoePlaced(dense_attn::Dev engine, int64_t layer_index,
                             const vt::Tensor& dh, int64_t T, int64_t H,
                             Body&& body) {
  const MoePlacementPlan& plan = ActiveMoePlacementPlan();
  const vt::DeviceType engine_device = engine.q.device.type;
  const vt::DeviceType placed_on = plan.PlacesAnything()
                                       ? plan.DeviceForLayer(layer_index)
                                       : engine_device;
  if (placed_on == engine_device) {
    // THE UNPLACED PATH, and it is the existing call with nothing around it: no
    // copy, no allocation, and the value the architecture already produced.
    return body(engine, dh);
  }

  const size_t bytes = static_cast<size_t>(T) * static_cast<size_t>(H) *
                       vt::SizeOf(vt::DType::kBF16);

  // DOWN. `Synchronize` is required, not defensive: the placed backend is about
  // to read these bytes and knows nothing about the engine's stream.
  std::vector<uint8_t> staging(bytes);
  engine.b.Copy(engine.q, staging.data(), dh.data, bytes);
  engine.b.Synchronize(engine.q);

  // ACROSS. `ResidentWeight` aliases host weight bytes for a CPU `Dev`, so a
  // layer whose block only ever runs here is never staged to the accelerator —
  // no upload happens, rather than one happening and being ignored. That is what
  // makes this free the device memory instead of only moving the arithmetic.
  vt::Queue& placed_queue = PlacementQueue(placed_on);
  dense_attn::Dev placed{vt::GetBackend(placed_queue.device.type), placed_queue};
  dense_attn::DBuf placed_in(placed, vt::DType::kBF16, {T, H}, staging.data());
  MoePlacedOutput placed_out = body(placed, placed_in.t());

  // BACK UP, into a buffer from the ENGINE's pool, so the composing forward owns
  // the result exactly as it owns an unplaced block's output.
  placed.b.Copy(placed.q, staging.data(), placed_out.tensor.data, bytes);
  placed.b.Synchronize(placed.q);
  dense_attn::DBuf back(engine, vt::DType::kBF16, {T, H}, staging.data());

  MoePlacedOutput r;
  r.tensor = back.t();
  r.storage = back.ReleaseShared();
  return r;
}

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MOE_PLACEMENT_SEAM_H_
