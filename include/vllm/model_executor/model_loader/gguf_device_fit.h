// vllm.cpp original. Pinned vLLM (555967922) has no GGUF load format — the whole
// tree carries two incidental mentions of the word and no loader — so there is no
// upstream counterpart to mirror.
//
// The closest upstream idea is the startup memory profile, and it answers a
// different question. `GPUWorker.determine_available_memory`
// (`vllm/v1/worker/gpu_worker.py:451-495`) runs `memory_profiling` around
// `profile_run` (`vllm/v1/worker/gpu/model_runner.py:682`) and passes the weight
// bytes IN as a known quantity, `weights_memory=model_memory_usage`
// (`gpu_worker.py:493`). That quantity is recorded AFTER the load has finished
// (`gpu/model_runner.py:315`). So upstream measures what is left once the weights
// are resident; it never asks whether they will fit, because by then it has paid
// for them. This file asks the question upstream does not.
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

// A lower bound on the device-resident bytes a weight-staging platform must
// allocate to stage EVERY TENSOR IN THIS FILE.
//
// Read that scope literally, because it is where the one over-count comes from: it
// is NOT a lower bound on what a particular load stages, since a load may stage a
// subset of the file. The per-tensor term is exact-or-low; the SET is exact-or-high.
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
//   * It can OVER-count, by including a tensor the loader never stages. The one
//     such class in this tree is the MTP / `nextn` block, which is attached only
//     when a speculator is configured
//     (`model_loader.cpp`, the `speculative_config->method == "mtp"` guard).
//     Measured on the target checkpoint, that is block 92: 20 tensors,
//     8,940,488,704 of 397,245,341,184 bytes, so 2.2506 %.
//   * It UNDER-counts by everything that is NOT a weight: KV cache,
//     activations, the device scratch pool and the driver context. That term is
//     far larger than 2.2506 %.
//
// The two errors are on DIFFERENT quantities and do not cancel, so neither
// rescues the other. In particular the refusal CAN over-refuse: a budget in
// [what a default load stages, what this counts) rejects a weight set that fits.
// On the target checkpoint that window is 8.33 GiB wide on a 369.96 GiB
// checkpoint, `VT_DEVICE_WEIGHT_BUDGET_BYTES` is the operator's way out of it,
// and `test_gguf_device_fit` pins the direction rather than leaving it described
// (issue #1136). Closing it means teaching the bound which tensors THIS load will
// stage, which is load policy and not a property of the file, so it is owed and
// not invented here. The under-count is owed to the startup memory profile
// (`KV-WARMUP-PROFILE`) for the same reason: an invented headroom fraction here
// would be the guess this bound exists to avoid.
struct GgufStagedFootprint {
  // The bound, in bytes. The name is accurate for what it measures — the sum of
  // per-tensor lower bounds over the file's whole tensor table — and it is NOT a
  // lower bound on one load's staging, for the reason above.
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
// that does not probe one. Two things override it, for an operator whose pool is
// smaller than the probe reports because something else lives in it, and for an
// operator who wants to attempt the load anyway:
// `--offload-config '{"vllm_cpp":{"device_fit":{"weight_budget_bytes":N}}}'`, and
// `VT_DEVICE_WEIGHT_BUDGET_BYTES`, which beats the config. A value of 0 from
// either means "unknown", i.e. disables the check, and an unparseable
// ENVIRONMENT value is ignored rather than treated as 0, because silently
// disabling a guard on a typo is the failure shape this tree refuses; a
// malformed CONFIG value cannot get this far, because the parser refuses it at
// startup.
//
// THE RULE ITSELF LIVES IN `vllm/config/weight_residency.h`
// (`ResolveDeviceWeightBudgetBytes`), and this function is a delegation to it
// (issue #1127). That keeps one reader for the variable, which is what stops the
// install-time override announcement from drifting away from what the resolver
// does with the value.
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
// Strictly greater than: a checkpoint whose footprint exactly equals the budget is
// not refused here. The footprint is approximate in both directions, so equality is
// not evidence of anything, and the tie goes to attempting the load.
DeviceWeightFit CheckDeviceWeightFit(const GgufFile& gguf,
                                     std::string_view device_name,
                                     bool needs_weight_staging,
                                     size_t budget_bytes,
                                     size_t model_dtype_bytes = 2);

}  // namespace vllm
