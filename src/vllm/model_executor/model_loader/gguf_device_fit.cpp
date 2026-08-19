// ENG-EXPERT-STREAM, issue #1123. See the header for what this decides and why.
#include "vllm/model_executor/model_loader/gguf_device_fit.h"

#include <string>

#include "vllm/config/weight_residency.h"

namespace vllm {
namespace {

// Bytes -> "N.NN GiB", so a refusal reads as a size rather than as 19 digits.
// Both the raw byte count and the GiB appear in the message: the first is what a
// reader can grep for in the code, the second is what an operator compares
// against the box.
std::string Gib(size_t bytes) {
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  std::string s = std::to_string(gib);
  const size_t dot = s.find('.');
  if (dot != std::string::npos && s.size() > dot + 3) s.resize(dot + 3);
  return s + " GiB";
}

// W0d. Does `name` end in `suffix`? An empty suffix matches NOTHING, which is
// what makes `StreamedExpertLane{}` inert: the natural reading of "ends with the
// empty string" is "always", and that would exclude every tensor in the file the
// moment a caller forgot to set the field.
bool NameHasSuffix(const std::string& name, std::string_view suffix) {
  if (suffix.empty() || suffix.size() > name.size()) return false;
  return name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The staged size of one tensor: `min(gguf_bytes, elems * model_dtype_bytes)`.
// See the header for why the minimum is the defensible term.
size_t StagedBytes(const GgufTensorInfo& t, size_t model_dtype_bytes) {
  size_t elems = 1;
  for (const int64_t d : t.shape) {
    if (d <= 0) {  // A malformed dim cannot be reasoned about; contribute the
      elems = 0;   // on-disk size alone rather than a bogus expanded size.
      break;
    }
    elems *= static_cast<size_t>(d);
  }
  // The expanded size, when it is knowable. `elems == 0` means the shape was
  // unusable, and then the on-disk size is the only defensible term.
  const size_t expanded = elems == 0 ? t.nbytes : elems * model_dtype_bytes;
  return expanded < t.nbytes ? expanded : t.nbytes;
}

}  // namespace

size_t GgufLargestExpertSliceBytes(const GgufFile& gguf,
                                   std::string_view tensor_name_suffix) {
  size_t largest = 0;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    if (!NameHasSuffix(t.name, tensor_name_suffix)) continue;
    // `shape[0]`, NOT `shape.back()`, and the difference is measured rather than
    // assumed: `gguf_reader.cpp:443` pushes `ggml_dims[d-1]`, so this vector is
    // the REVERSE of the file's `ne` order. A tower the file writes as
    // `[n_embd, n_ff_exp, n_expert]` therefore arrives as
    // `[n_expert, n_ff_exp, n_embd]`, and the expert count is the first entry.
    // Reading the last one gives `n_embd` — 8192 on the target checkpoint
    // instead of 512, a slot 16x too small.
    //
    // A tensor of another rank is not a stacked tower; charging its whole size
    // over-states the slot rather than under-stating it.
    const size_t experts = t.shape.size() == 3 && t.shape[0] > 0
                               ? static_cast<size_t>(t.shape[0])
                               : 1;
    const size_t slice = t.nbytes / experts;
    if (slice > largest) largest = slice;
  }
  return largest;
}

bool GgufExpertTowersReachSlotLane(const GgufFile& gguf,
                                   std::string_view tensor_name_suffix,
                                   const GgufLoadPolicy& policy) {
  bool matched = false;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    if (!NameHasSuffix(t.name, tensor_name_suffix)) continue;
    matched = true;
    // The SAME call the model's own loader makes, on the SAME tensor, with the
    // SAME role, through the SAME function: `LoadExpertsOrNvfp4` peeks
    // `kStackedExpertWeight` and dispatches on the answer. Asking it the same way
    // is what stops the bound and the forward from disagreeing about a file.
    const GgufResidency r =
        PeekRoute(policy, t, GgufTensorRole::kStackedExpertWeight);
    if (r != GgufResidency::kKeepQuant && r != GgufResidency::kKeepF16) {
      return false;
    }
  }
  return matched;
}

GgufStagedFootprint GgufStagedWeightFootprint(const GgufFile& gguf,
                                              size_t model_dtype_bytes,
                                              const StreamedExpertLane& lane) {
  GgufStagedFootprint out;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    // W0d: a tensor the slot lane serves is never staged, so it contributes
    // nothing to the bound and cannot be the largest single allocation either.
    // It is still COUNTED, in its own two fields, so the caller can say what was
    // left out and how much of it there was.
    if (NameHasSuffix(t.name, lane.tensor_name_suffix)) {
      ++out.streamed_tensor_count;
      out.streamed_bytes += StagedBytes(t, model_dtype_bytes);
      continue;
    }
    const size_t staged = StagedBytes(t, model_dtype_bytes);
    out.lower_bound_bytes += staged;
    ++out.tensor_count;
    if (staged > out.largest_tensor_bytes) {
      out.largest_tensor_bytes = staged;
      out.largest_tensor_name = t.name;
    }
  }
  // The arena is charged only when the lane actually has towers to serve. A file
  // with no expert towers builds no slot store at all (`Qwen35ExpertStream::Get`
  // is reached only from the expert-slice seam), so charging one would invent
  // bytes nothing allocates.
  if (out.streamed_tensor_count > 0) {
    out.arena_bytes = lane.arena_bytes;
    out.lower_bound_bytes += lane.arena_bytes;
  }
  return out;
}

size_t DeviceWeightBudgetBytes(size_t device_memory_total_bytes) {
  // THE PRECEDENCE LIVES IN ONE PLACE, and since #1127 that place is
  // `vllm/config/weight_residency.h`: `VT_DEVICE_WEIGHT_BUDGET_BYTES` >
  // `vllm_cpp.device_fit.weight_budget_bytes` > this probe. The environment
  // grammar is unchanged — decimal digits only, so a malformed value is IGNORED
  // rather than read as 0, which would silently disable the guard on a typo —
  // and what it now falls through to is the config rather than straight to the
  // probe. A run with no config therefore resolves byte-for-byte as it did
  // before the key existed.
  //
  // This function keeps its name and its callers. Moving the rule rather than
  // the entry point is what lets `ResolveDeviceWeightBudgetBytes` be the SOLE
  // reader of the variable, which is the contract every knob in that header has
  // and the reason `DescribeEnvOverrides` cannot drift from the resolver.
  return ResolveDeviceWeightBudgetBytes(device_memory_total_bytes);
}

DeviceWeightFit CheckDeviceWeightFit(const GgufFile& gguf,
                                     std::string_view device_name,
                                     bool needs_weight_staging,
                                     size_t budget_bytes,
                                     size_t model_dtype_bytes,
                                     const StreamedExpertLane& lane) {
  DeviceWeightFit fit;
  fit.budget_bytes = budget_bytes;
  // A platform that does not stage weights reads them where they already are, so
  // a borrowed tower costs it nothing and there is nothing to compare. This is
  // the branch every CPU load takes, and it must be free of any behaviour
  // change: no footprint is even computed.
  if (!needs_weight_staging) return fit;
  // 0 == UNKNOWN, and unknown is not a verdict. Refusing a load because nothing
  // reported a budget would break every device whose budget nothing probes.
  if (budget_bytes == 0) return fit;

  const GgufStagedFootprint fp =
      GgufStagedWeightFootprint(gguf, model_dtype_bytes, lane);
  fit.needed_bytes = fp.lower_bound_bytes;
  if (fp.lower_bound_bytes <= budget_bytes) return fit;

  fit.refuse = true;
  fit.message =
      "device '" + std::string(device_name) +
      "' cannot serve this GGUF: staging its weights needs at least " +
      std::to_string(fp.lower_bound_bytes) + " bytes (" +
      Gib(fp.lower_bound_bytes) + ") of device memory across " +
      std::to_string(fp.tensor_count) + " tensors, the largest single "
      "allocation being " + std::to_string(fp.largest_tensor_bytes) + " bytes (" +
      Gib(fp.largest_tensor_bytes) + ", '" + fp.largest_tensor_name +
      "'), and this device's memory pool is " + std::to_string(budget_bytes) +
      " bytes (" + Gib(budget_bytes) + "). THE MISSING PART: the "
      "larger-than-memory lane that makes a checkpoint like this fit is "
      "HOST-ONLY. The GGUF mapping is borrowed in place on the CPU path and "
      "costs no resident bytes, while a weight-staging device copies every "
      "expert tower into device memory; there is no device-side expert slot "
      "store and no device streaming lane (ENG-EXPERT-STREAM, issues #1123 and "
      "#1124). Use device=cpu, which serves this checkpoint today, or a "
      "checkpoint that fits the pool. This is refused at LOAD on purpose: "
      "before this check the load succeeded and the FIRST forward died with "
      "'vt cuda: cudaMalloc: out of memory'. Setting "
      "VT_DEVICE_WEIGHT_BUDGET_BYTES higher (or to 0), or "
      "--offload-config '{\"vllm_cpp\":{\"device_fit\":"
      "{\"weight_budget_bytes\":0}}}', suppresses this refusal "
      "and restores that late failure; it does not make the model fit.";
  // W0d. With the lane ON the message above is misleading in its most important
  // sentence — the device streaming lane is exactly what IS running — so the
  // correction is appended rather than left to be read as a stale claim. The
  // lane-OFF message is untouched, byte for byte, because every CPU and discrete
  // user reads that one and it is still true for them.
  if (fp.streamed_tensor_count > 0) {
    fit.message +=
        " NOTE: the expert-stream lane IS active for this load, so the "
        "sentence above about there being no device streaming lane does not "
        "apply. " + std::to_string(fp.streamed_tensor_count) +
        " expert towers (" + std::to_string(fp.streamed_bytes) + " bytes, " +
        Gib(fp.streamed_bytes) + ") were EXCLUDED from the figure above "
        "because the lane serves their slices from host slot storage, and the "
        "slot arena's " + std::to_string(fp.arena_bytes) + " bytes (" +
        Gib(fp.arena_bytes) + ") were counted instead. What does not fit is "
        "therefore the NON-expert remainder plus that arena; lowering "
        "VT_MOE_EXPERT_STREAM_SLOTS shrinks the arena, and nothing shrinks the "
        "remainder.";
  }
  return fit;
}

}  // namespace vllm
