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

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
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
// ENG-EXPERT-STREAM-DEVICE W0d (issue #1124). The one input that tells the bound
// above "these tensors are NOT staged, and this arena is what the device pays
// instead".
//
// WHY THIS AND NOT A GENERAL PER-TENSOR STAGING POLICY. The header above records
// that closing its over-count "means teaching the bound which tensors THIS load
// will stage, which is load policy and not a property of the file, so it is owed
// and not invented here" (issue #1136). That judgement stands, and this is not
// that: it is not a predicate, not a callback and not a policy object. It is one
// literal name suffix and one byte count, both known at load, describing a lane
// that is either on or off for the whole file. A caller cannot express "stage
// this tensor and not that one" through it.
//
// EMPTY MEANS OFF, and off must be BYTE-IDENTICAL to the pre-W0d bound — same
// sum, same tensor count, same largest tensor, same message. That is a gate
// (`test_gguf_device_fit`), because a load-time refusal that quietly moved for
// every CPU user would be a far worse defect than the one this input removes.
struct StreamedExpertLane {
  // The tensor-name SUFFIX the slot lane serves. On a llama.cpp MoE export the
  // stacked expert towers are `blk.<n>.ffn_{gate,up,down}_exps.weight`, so
  // `_exps.weight` names exactly the set `KqExpertSlice` streams and nothing
  // else. Empty == the lane is off and this whole struct is inert.
  std::string_view tensor_name_suffix{};
  // The slot arena's resident bytes — `slots * slot_bytes`, which is what
  // `HostExpertSlotStore` allocates up front and holds for the process. Added to
  // the bound ONLY when at least one tensor actually matched the suffix, because
  // a file with no expert towers builds no store and would otherwise be charged
  // for an arena that never exists.
  size_t arena_bytes = 0;
};

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
  // W0d. How many tensors the lane took out of the sum, and how many bytes they
  // would otherwise have contributed. Both are 0 with the lane off. They are
  // reported rather than merely subtracted because a caller that says "75.35 GiB"
  // without saying "and 335.62 GiB across 279 towers is served elsewhere" has not
  // said how many things it examined.
  size_t streamed_tensor_count = 0;
  size_t streamed_bytes = 0;
  // The arena term actually added (0 when the lane is off, and 0 when it is on
  // but nothing matched).
  size_t arena_bytes = 0;
};

// The per-expert SLICE size of a stacked expert tower, which is the slot size
// the lane needs for it: `nbytes / n_expert`.
//
// A llama.cpp MoE export stores a tower as a 3-D tensor whose last `ne`
// dimension is the expert count — `[n_embd, n_ff_exp, n_expert]` — so the
// divisor is read off the tensor itself and no metadata key, architecture prefix
// or config lookup is involved. `GgufTensorInfo::shape` is the REVERSE of `ne`
// order (`gguf_reader.cpp:443`), so in this tree the expert count is `shape[0]`;
// the implementation says so at the line that divides, because reading the other
// end silently yields `n_embd` and a slot 16x too small. Measured on
// `Qwen3.8-2.4T-A95B UD-Q1_0`: 1,275,068,416 / 512 = 2,490,368 bytes, which is
// exactly the slice `KqExpertSlice` takes.
//
// Returns the largest such slice over every tensor whose name ends in `suffix`,
// or 0 when none matches. A matching tensor that is not 3-D contributes its whole
// size, which over-states rather than under-states the slot it would need.
//
// IT SCANS THE FILE, AND A DEFAULT LOAD STREAMS A SUBSET OF IT, so the answer
// can be larger than the slot the store actually builds. That is the same
// over-count direction the footprint above documents, on the same tensors: on
// `Qwen3.8-2.4T-A95B UD-Q1_0` the largest `*_exps` slice in the FILE is an MTP
// `nextn`-block Q2_K tower at 2,818,572,288 / 512 = 5,505,024 bytes, while the
// IQ1_XXXS towers a default load streams need 2,490,368. The arena term is then
// 2.21x the one `Qwen35ExpertStream::Reserve` will settle on. Narrowing it means
// teaching this function which tensors THIS load will reach, which is load
// policy and not a property of the file — the shape issue #1136 declines to
// invent — so the error is named here and left in the safe direction.
size_t GgufLargestExpertSliceBytes(const GgufFile& gguf,
                                   std::string_view tensor_name_suffix);

// ENG-EXPERT-STREAM-DEVICE W0d repair, issue #1378. The RESIDENCY ROUTE term of
// the lane condition: will the towers the lane proposes to serve actually reach
// the slot seam, or will they be staged like any other weight?
//
// The four terms the loader asked before this one -- the platform stages, the
// platform can read host slots, the factory declares `streams_routed_experts`,
// streaming is requested -- are properties of the DEVICE and of the
// ARCHITECTURE. None of them is a property of THIS FILE under THIS POLICY.
// `KqExpertSlice`, the seam that serves a slice out of the host slot store, is
// reached only from the `expert_*_kq` arm, and `LoadExpertsOrNvfp4`
// (`qwen3_5_gguf_weights.cpp`) routes the same
// `blk.<n>.ffn_{gate,up,down}_exps.weight` tensors to two OTHER arms as well:
// `kNvfp4Fp4` fills `expert_*_fp4` and `kExpandBf16` fills `expert_*`, and both
// of those reach `MoeBlockFusedMarlinCuda` / `MoeBlockFusedCuda` /
// `MoeBlockBf16Cuda`, which stage every tower through `ResidentWeight`.
//
// So with `VT_GGUF_KEEP_QUANT=0` -- a documented, supported opt-out
// (`gguf_keep_quant.cpp`, `FromEnv`) -- or with an NVFP4 GGUF, all four terms held,
// the lane turned on, every tower left the bound, and the #1123 refusal was
// DELETED on a load that stages every one of them. The `expert_streamed`
// tripwire in `ResidentWeight` does not catch it either, because a tower on
// those arms is never claimed. What is left is the 26-minute load and the
// `cudaMalloc: out of memory` first forward that refusal exists to prevent.
//
// Keyed on the DECLARED property, exactly as the architecture term is: the
// residency `RouteGgufTensor` gives each tower under the policy this process
// resolved, never a list of encodings or of file names.
//
// THE ACCEPTED SET IS `kKeepQuant` OR `kKeepF16`, and that is read off the code
// rather than chosen: `LoadExpertsOrNvfp4` dispatches everything that is neither
// fp4 nor expand into `LoadExpertsStackedKq`, whose own `VT_CHECK` names exactly
// those two, and `KqExpertSlice` is dtype-agnostic (it sizes a slice with
// `vt::RowSizeBytes`, which serves a block dtype and an f16 alike). So both keep
// residencies fill `expert_*_kq` and both reach the lane.
//
// ALL OR NOTHING, in the safe direction. `StreamedExpertLane` is deliberately one
// suffix and one byte count for the whole file -- its own comment says a caller
// cannot express "stage this tensor and not that one" through it -- so this
// returns true only when EVERY matching tensor reaches the lane, and false the
// moment one of them would be staged. A file that mixes the two therefore keeps
// the WHOLE bound: it can over-refuse, which `VT_DEVICE_WEIGHT_BUDGET_BYTES`
// releases, and it can never under-refuse, which nothing releases.
//
// False when NOTHING matches the suffix, because "every element of the empty set
// reaches the lane" is true and useless: a file with no expert towers has no lane
// to turn on.
bool GgufExpertTowersReachSlotLane(const GgufFile& gguf,
                                   std::string_view tensor_name_suffix,
                                   const GgufLoadPolicy& policy);

// `model_dtype_bytes` is the resolved model dtype's size (2 for bf16, which is
// what every GGUF path here loads at). vLLM resolves ONE model dtype and every
// layer inherits it, so one value is the faithful shape.
GgufStagedFootprint GgufStagedWeightFootprint(const GgufFile& gguf,
                                              size_t model_dtype_bytes = 2,
                                              const StreamedExpertLane& lane = {});

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
// `lane` (W0d) changes only what `needed` COUNTS, never the shape of the test:
// with the lane on, the towers it serves are not staged and its arena is, so
// `needed` is the remainder plus the arena.
//
// THE NUMBER THIS CHARGES, on `Qwen3.8-2.4T-A95B UD-Q1_0` at the 8000 slots the
// row measures with, stated as what the code computes rather than as what the
// store will settle on, because the two differ and the charged one is what
// decides the load:
//
//   34.34 GiB non-expert remainder
//   + 8000 x 5,505,024 B = 44,040,192,000 B = 41.02 GiB arena
//   = 80,910,933,504 B = 75.35 GiB, against a 119.631 GiB pool.
//
// The arena term is `slots * ResolveExpertStreamSlotBytes(GgufLargestExpertSliceBytes(...))`,
// and that function's own doc above says what its answer is on this checkpoint:
// the largest `*_exps` slice in the FILE, which is an MTP `nextn`-block Q2_K
// tower at 2,818,572,288 / 512 = 5,505,024 B, not the 2,490,368 B IQ1_XXXS slice
// a default load actually streams. So the charged arena is 2.2105x the 18.55 GiB
// `Qwen35ExpertStream::Reserve` will allocate, and the charged total is 75.35 GiB
// where the resident total will be 52.89 GiB. That is the SAME over-count, in the
// same safe direction, as the file-versus-load scope error the footprint doc
// above names (issue #1136) — it can over-refuse, never under-refuse — and it is
// written here as the charged figure because a reader deciding whether a
// checkpoint fits reads this line and not the resolver.
//
// The over-count on the remainder is unchanged and still includes the MTP block
// (issue #1136), which is the difference between that 34.34 GiB and the 26.01
// GiB a default load actually stages.
//
// Keyed on the MEASURED condition, never on "CUDA + GGUF", so a GGUF that
// genuinely fits the pool still loads. `lane` is the ONE architecture-dependent
// input, and it is not resolved here: the loader fills it only for a model whose
// factory declares `streams_routed_experts`, because the `_exps.weight` suffix is
// written by MoE families this tree does not stream and dropping THEIR towers
// would delete a correct refusal. This function still decides nothing from a
// name; it is told, by the one caller that knows.
// Strictly greater than: a checkpoint whose footprint exactly equals the budget is
// not refused here. The footprint is approximate in both directions, so equality is
// not evidence of anything, and the tie goes to attempting the load.
DeviceWeightFit CheckDeviceWeightFit(const GgufFile& gguf,
                                     std::string_view device_name,
                                     bool needs_weight_staging,
                                     size_t budget_bytes,
                                     size_t model_dtype_bytes = 2,
                                     const StreamedExpertLane& lane = {});

}  // namespace vllm
