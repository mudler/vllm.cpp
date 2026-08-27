// Exposed entry point for the file-static Gated DeltaNet (GDN) linear-attention
// block that lives in the anonymous namespace of
// src/vllm/model_executor/models/qwen3_5.cpp (`GdnBlockPaged` @ the Qwen3.5/3.6
// paged decoder layer, over vt::CausalConv1d*/vt::GdnDecode/vt::GdnSpecDecode).
//
// The Qwen3.5/3.6 forward (`RunLayerPaged` / the dense paged layer) calls
// `GdnBlockPaged` internally, but a NEW hybrid architecture — Qwen4-Exp
// (`Qwen4ExpLayerKind::kLinearAttention`, 36 of its 48 layers, row
// MODEL-MM-QWEN4-EXP W5b) — must reuse the SAME block from a DIFFERENT
// translation unit. `GdnBlockPaged` takes the qwen3_5.cpp-internal `Dev` and
// `StepDevInputs` types and returns its internal pooled `DBuf` (each TU defines
// its own), so it cannot be called cross-TU directly.
//
// This header therefore mirrors the `RunMoeBlock` precedent next door
// (qwen3_5_moe_block.h), which exists for the identical reason: a thin public
// wrapper over primitive vt:: types. `RunGdnBlockPaged` builds the internal
// `Dev`, calls `GdnBlockPaged`, and hands back the block's device buffer as an
// owning `GdnBlockOutput` whose deleter returns the pool block to the shared
// DevicePool (the same `WrapDeviceLogits` release pattern `MoeBlockOutput`
// uses). The Qwen3.5/3.6 path is UNTOUCHED and byte-identical — this only ADDS
// a second caller; there is exactly one GDN implementation and both callers run
// it.
//
// WHY THE STEP INPUTS ARE A SEPARATE HANDLE. `StepDevInputs` is the per-step
// device upload of positions / slot_mapping / block_table / seq_lens /
// query_start_loc / the GDN state-index tensors, and qwen3_5.cpp builds it ONCE
// PER STEP and shares it across every layer precisely so those uploads do not
// repeat per layer ("ONE upload per input, per step", qwen3_5.cpp). A wrapper
// that rebuilt it inside each block call would reinstate exactly the per-layer
// upload that was removed, 36 times per step on this architecture. So the build
// is its own public call returning an opaque owning handle (the same
// `shared_ptr<void>` idiom `MoeBlockOutput::storage` uses to keep an internal
// type out of this header), and the composing forward keeps it alive for the
// whole layer loop, exactly as `RunLayerPaged`'s caller does.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"          // GdnStateCache
#include "vllm/model_executor/models/qwen3_5_weights.h"  // GdnLayerWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// Owning device-resident output of one GDN layer: a [T,H] tensor view at the
// block's output dtype plus the shared_ptr that returns its pool block to the
// DevicePool when the last reference drops (the block outlives the wrapper by
// design so the composing forward can keep computing over `tensor`).
struct GdnBlockOutput {
  vt::Tensor tensor;              // [T,H] device view at GdnOutDType()
  std::shared_ptr<void> storage;  // owns the pool block (Pool().Put on release)
};

// Opaque owning handle over the per-step device upload the GDN block reads
// (`StepDevInputs`, internal to qwen3_5.cpp). Build ONE per step and hand the
// same handle to every layer's `RunGdnBlockPaged`; it must outlive the last
// call that reads it.
struct GdnStepInputs {
  std::shared_ptr<void> impl;  // owns a qwen3_5.cpp StepDevInputs
};

// Upload one step's positions and attention/GDN metadata to `queue`'s device.
// This is the production `BuildStepDevInputs`, unchanged: same validation, same
// buffers, same order. `gdn_state_slots` is the GDN state cache's slot count
// (`state.ssm_state.shape[0]`), against which the state indices are validated.
GdnStepInputs BuildGdnStepInputs(vt::Queue& queue,
                                 const std::vector<int32_t>& positions,
                                 const v1::CommonAttentionMetadata& attn_meta,
                                 const v1::GDNAttentionMetadata& gdn_meta,
                                 int64_t gdn_state_slots);

// Run one Qwen3.5/3.6 GDN linear-attention layer over a device-resident hidden
// `dh` [T,H] and return the [T,H] output as an owning device buffer. `state`'s
// SSM and conv tensors are read AND written in place, exactly as in the
// Qwen3.5/3.6 forward. `dh_fp8` is the optional fp8 view of the same hidden that
// the fp8 input-projection tower consumes (nullptr on every other arm). Which
// arm runs — decode, prefill, spec-decode, packed, fp8 — is decided inside the
// block by `gdn_meta` and the weight set, identically for both callers.
GdnBlockOutput RunGdnBlockPaged(vt::Queue& queue, const GdnLayerWeights& weights,
                                const HfConfig& config, const vt::Tensor& dh,
                                const GdnStepInputs& step,
                                const v1::GDNAttentionMetadata& gdn_meta,
                                const GdnStateCache& state, int64_t T,
                                const vt::Tensor* dh_fp8 = nullptr);

}  // namespace vllm
