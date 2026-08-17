// vllm.cpp original. Pinned vLLM has no GGUF load format, so there is no
// upstream counterpart to mirror; the closest upstream idea is the startup
// memory profile (`vllm/v1/worker/gpu/model_runner.py:504,647`), which measures
// the NON-weight footprint and is a different question from the one here.
//
// ENG-EXPERT-STREAM, issue #1123: a load-time answer to "can this device
// actually hold this checkpoint's weights?".
//
// The failure this exists to remove: `Qwen3.8-2.4T-A95B UD-Q1_0` (369.96 GiB)
// reached a serving state on `--device cuda` on a 119.631 GiB GB10 after 26
// minutes and then died on the FIRST forward with `vt cuda: cudaMalloc: out of
// memory`. The load succeeds because a keep-quant expert tower is BORROWED from
// the GGUF mapping and costs zero anonymous bytes; the forward dies because a
// weight-staging platform copies each borrowed tower into device memory
// (`ResidentWeight`, `qwen3_5.cpp:1011`, `d.b.Alloc(w.bytes.size())` on a
// STACKED `[E*N,K]` tower: 276 towers of 1,275,068,416 bytes plus 3 of
// 2,818,572,288, so 360,374,599,680 bytes = 335.62 GiB in total).
//
// See `.agents/specs/expert-streaming.md`, section "`--device cuda` loads for 26
// minutes and then dies", for the measurement this is keyed on.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace vllm {

// A bound on the device-resident weight bytes a weight-staging platform must
// allocate for this GGUF, built to be WRONG LOW rather than wrong high.
//
// Per tensor the bound is `min(gguf_bytes, elems * model_dtype_bytes)`: a weight
// the loader keeps quantized is staged verbatim (`gguf_bytes`), a weight it
// expands is staged at the model dtype (`elems * 2` for bf16), and which of the
// two happens is a per-tensor loader policy this header deliberately does not
// try to predict. Taking the minimum makes the per-tensor term a true lower
// bound on that tensor's staged size.
//
// Both directions of error are named here rather than claimed away, because a
// bound whose error direction is unstated is not a bound.
//
//   * It can over-count, by including a tensor the loader never stages. The one
//     such class in this tree is the MTP / `nextn` block, which is loaded only
//     when a speculator is configured. Measured on the target checkpoint, that
//     is block 92: 20 tensors, 8,940,488,704 of 397,245,341,184 bytes, so
//     2.2506 %.
//   * It under-counts by everything that is NOT a weight: KV cache,
//     activations, the device scratch pool and the driver context. That term is
//     far larger than 2.2506 %.
//
// So the two errors point in opposite directions and the under-count dominates.
// The refusal built on this therefore fires only well above the budget, never on
// a marginal case, and the marginal cases stay owed to the startup memory
// profile (`KV-WARMUP-PROFILE`) rather than being covered by an invented
// headroom fraction here.
struct GgufStagedFootprint {
  // The bound, in bytes.
  size_t lower_bound_bytes = 0;
  // How many tensor records went into it. A caller that reports a footprint
  // without this cannot say how many things it examined.
  size_t tensor_count = 0;
  // The largest single tensor, which is the largest single contiguous
  // allocation the staging path will ask the driver for. An aggregate that fits
  // is not the same as a contiguous block that fits.
  size_t largest_tensor_bytes = 0;
  std::string largest_tensor_name;
};

// `model_dtype_bytes` is the resolved model dtype's size (2 for bf16, which is
// what every GGUF path here loads at). vLLM resolves ONE model dtype and every
// layer inherits it, so one value is the faithful shape.
GgufStagedFootprint GgufStagedWeightFootprint(const GgufFile& gguf,
                                              size_t model_dtype_bytes = 2);

// The budget to compare a footprint against, in bytes, or 0 for UNKNOWN.
//
// `device_memory_total_bytes` is the platform's own probe
// (`ResidencyPolicy::device_memory_total_bytes`), which is 0 on every platform
// that does not probe one. `VT_DEVICE_WEIGHT_BUDGET_BYTES` overrides it, for an
// operator whose pool is smaller than the probe reports because something else
// lives in it, and for an operator who wants to attempt the load anyway. A
// value of 0 in the environment means "unknown", i.e. disables the check, and
// an unparseable value is ignored rather than treated as 0, because silently
// disabling a guard on a typo is the failure shape this tree refuses.
//
// TOTAL rather than FREE on purpose: `free` at load time carries the page cache
// and whatever else the box is doing, which would make the verdict a function
// of contention. `total` is a device property.
size_t DeviceWeightBudgetBytes(size_t device_memory_total_bytes);

// The verdict. `refuse == false` with a zero budget means "not decided",
// which is NOT the same as "it fits" — see the comment on the budget above.
struct DeviceWeightFit {
  bool refuse = false;
  size_t needed_bytes = 0;
  size_t budget_bytes = 0;
  // Empty unless `refuse`. Names the device, both numbers, the missing
  // capability and what to do instead.
  std::string message;
};

// The predicate, in one place so one description exists:
//
//   refuse  <=>  needs_weight_staging  AND  budget != 0  AND  needed > budget
//
// Keyed on the MEASURED condition, never on "CUDA + GGUF" and never on an
// architecture name, so a GGUF that genuinely fits the pool still loads.
// Strictly greater than: a checkpoint that exactly equals the budget is not
// refused here, because the bound is a lower bound and the equality case is not
// evidence of anything.
DeviceWeightFit CheckDeviceWeightFit(const GgufFile& gguf,
                                     std::string_view device_name,
                                     bool needs_weight_staging,
                                     size_t budget_bytes,
                                     size_t model_dtype_bytes = 2);

}  // namespace vllm
