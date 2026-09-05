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
#include <stdexcept>
#include <utility>
#include <vector>

#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm {

// Run one layer's routed-expert block, on the placement device when this layer
// is placed and on the engine's own device otherwise.
//
// `body` is `(dense_attn::Dev, const vt::Tensor&) -> dense_attn::DBuf`, and it
// receives the device it must compute on. It must NOT capture the engine `Dev`
// and use that instead: the whole point is that the block runs where it is told.
//
// DBuf IS THE CURRENCY because it is what every MoE block in this tree already
// returns — `MoeBlock`, `NemotronHMoeBlockDevice` and their siblings. An earlier
// shape traded in a tensor-plus-storage pair and made every DBuf-returning call
// site need a conversion that does not exist; inverting it made the wiring one
// line per architecture, which was the point.
//
// ONE GLUE, ONE SPELLING. An earlier shape templated this on `Dev`/`DBuf`
// because `qwen3_5.cpp` kept private copies in an anonymous namespace, which
// gave anything it returned internal linkage. That file now uses the shared glue
// (`ENG-QWEN35-SHARED-GLUE`), so the template parameters bought nothing and are
// gone. What remains are two SHAPES, and the difference between them is a real
// one about ownership rather than about types.
// WHY A `placeable` FLAG RATHER THAN A CHECK INSIDE THE SEAM. The seam is
// templated over a lambda and never sees the weights, so it cannot inspect them.
// The architecture can: it knows whether ITS experts are in a form a placement
// would help or hurt. Passing that in keeps the seam generic and puts the
// knowledge where it lives.
//
// THE CASE THIS EXISTS FOR, and it is a regression this file already shipped
// once. An fp4-RESIDENT expert arm builds its device residents EAGERLY at load,
// so placing it uploads every expert and then computes on the host across the
// bus — strictly worse than not placing. W3b refused that by name; the W3c
// refactor moved every architecture onto this seam and did NOT carry the refusal
// across, leaving the old helper dead and the live path unguarded. A TOKEN GATE
// CANNOT SEE THAT DEFECT: the tokens stay right while the path moves twice the
// bytes. It REFUSES rather than silently running unplaced, because quietly not
// doing what the operator asked is the invisible-fallback shape this tree rejects.
template <class Body>
dense_attn::DBuf RunMoePlaced(dense_attn::Dev engine, int64_t layer_index,
                              const vt::Tensor& dh, int64_t T, int64_t H,
                              Body&& body, bool placeable = true,
                              const char* unplaceable_reason = nullptr) {
  const MoePlacementPlan& plan = ActiveMoePlacementPlan();
  const vt::DeviceType engine_device = engine.q.device.type;
  const vt::DeviceType placed_on = plan.PlacesAnything()
                                       ? plan.DeviceForLayer(layer_index)
                                       : engine_device;
  if (placed_on != engine_device && !placeable) {
    throw std::invalid_argument(
        std::string("device placement: this layer's routed experts cannot be "
                    "placed") +
        (unplaceable_reason ? std::string(" - ") + unplaceable_reason : "") +
        ". Refusing rather than placing them anyway, because that would be "
        "slower than not placing and a token gate would not show it.");
  }
  if (placed_on == engine_device) {
    // THE UNPLACED PATH, and it is the existing call with nothing around it: no
    // copy, no allocation, and the value the architecture already produced.
    dense_attn::DBuf out = body(engine, dh);
    // Inert unless VT_PLACEMENT_DUMP_MOE is set; this is the gate's reference arm.
    MaybeDumpMoeBlockOutput(layer_index, engine.b, engine.q, out.t().data, T * H,
                            /*data_is_host=*/false, out.t().dtype);
    return out;
  }

  // THE BLOCK'S OWN DTYPE, never an assumed one. This was hardcoded to bf16 in
  // six places and never compared against `dh`, while `kimi_linear_device.cpp`
  // hands the seam an f32 `[T,H]` buffer -- so the placed branch copied HALF the
  // bytes and reinterpreted f32 as bf16, silently. It could not fire before the
  // plan was installed (#2382), because `placed_on` always equalled the engine
  // device and this whole branch was dead; installing the plan is what made it
  // reachable, so it is fixed in the same change.
  const vt::DType dt = dh.dtype;
  const size_t bytes =
      static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(dt);

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
  dense_attn::DBuf placed_in(placed, dt, {T, H}, staging.data());
  dense_attn::DBuf placed_out = body(placed, placed_in.t());

  // BACK UP, into a buffer from the ENGINE's pool, so the composing forward owns
  // the result exactly as it owns an unplaced block's output.
  // The OUTPUT's dtype need not equal the input's, so size this copy by what the
  // body actually produced rather than by what went in.
  const vt::DType out_dt = placed_out.t().dtype;
  const size_t out_bytes =
      static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(out_dt);
  staging.resize(out_bytes);
  placed.b.Copy(placed.q, staging.data(), placed_out.t().data, out_bytes);
  placed.b.Synchronize(placed.q);
  // The measured arm. `staging` is already host memory, so this costs no copy.
  MaybeDumpMoeBlockOutput(layer_index, engine.b, engine.q, staging.data(), T * H,
                          /*data_is_host=*/true, out_dt);
  return dense_attn::DBuf(engine, out_dt, {T, H}, staging.data());
}

// The owning-PAIR variant, for a caller whose MoE block lives in another
// translation unit.
//
// IT EXISTS BECAUSE `DBuf` IS MOVE-ONLY AND POOL-OWNING, so a block whose
// implementation lives in another translation unit hands back the owning PAIR
// (`MoeBlockOutput`: a `vt::Tensor` plus the `shared_ptr` that returns its pool
// block) rather than the buffer itself. `RunMoeBlock` has that shape for exactly
// this reason. An in-TU caller trades in `DBuf`, an out-of-TU caller trades in
// pairs, and both reach the same transfer rather than each growing one.
struct MoePlacedOutput {
  vt::Tensor tensor;
  std::shared_ptr<void> storage;
};

template <class Body>
MoePlacedOutput RunMoePlacedPair(dense_attn::Dev engine, int64_t layer_index,
                                 const vt::Tensor& dh, int64_t T, int64_t H,
                                 Body&& body, bool placeable = true,
                                 const char* unplaceable_reason = nullptr) {
  const MoePlacementPlan& plan = ActiveMoePlacementPlan();
  const vt::DeviceType engine_device = engine.q.device.type;
  const vt::DeviceType placed_on = plan.PlacesAnything()
                                       ? plan.DeviceForLayer(layer_index)
                                       : engine_device;
  if (placed_on != engine_device && !placeable) {
    throw std::invalid_argument(
        std::string("device placement: this layer's routed experts cannot be "
                    "placed") +
        (unplaceable_reason ? std::string(" - ") + unplaceable_reason : "") +
        ". Refusing rather than placing them anyway, because that would be "
        "slower than not placing and a token gate would not show it.");
  }
  if (placed_on == engine_device) {
    MoePlacedOutput out = body(engine, dh);
    MaybeDumpMoeBlockOutput(layer_index, engine.b, engine.q, out.tensor.data,
                            T * H, /*data_is_host=*/false, out.tensor.dtype);
    return out;
  }

  // THE BLOCK'S OWN DTYPE, never an assumed one. This was hardcoded to bf16 in
  // six places and never compared against `dh`, while `kimi_linear_device.cpp`
  // hands the seam an f32 `[T,H]` buffer -- so the placed branch copied HALF the
  // bytes and reinterpreted f32 as bf16, silently. It could not fire before the
  // plan was installed (#2382), because `placed_on` always equalled the engine
  // device and this whole branch was dead; installing the plan is what made it
  // reachable, so it is fixed in the same change.
  const vt::DType dt = dh.dtype;
  const size_t bytes =
      static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(dt);
  std::vector<uint8_t> staging(bytes);
  engine.b.Copy(engine.q, staging.data(), dh.data, bytes);
  engine.b.Synchronize(engine.q);

  vt::Queue& placed_queue = PlacementQueue(placed_on);
  dense_attn::Dev placed{vt::GetBackend(placed_queue.device.type), placed_queue};
  dense_attn::DBuf placed_in(placed, dt, {T, H}, staging.data());
  MoePlacedOutput placed_out = body(placed, placed_in.t());

  const vt::DType out_dt = placed_out.tensor.dtype;
  const size_t out_bytes =
      static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(out_dt);
  staging.resize(out_bytes);
  placed.b.Copy(placed.q, staging.data(), placed_out.tensor.data, out_bytes);
  placed.b.Synchronize(placed.q);
  MaybeDumpMoeBlockOutput(layer_index, engine.b, engine.q, staging.data(), T * H,
                          /*data_is_host=*/true, out_dt);
  dense_attn::DBuf back(engine, out_dt, {T, H}, staging.data());

  MoePlacedOutput r;
  r.tensor = back.t();
  r.storage = back.ReleaseShared();
  return r;
}

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MOE_PLACEMENT_SEAM_H_
