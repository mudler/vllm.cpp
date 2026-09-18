// LTX-2.5 LoRA — thin aliases onto the shared DiT LoRA seam.
//
// The types and fusion arithmetic are model-agnostic and live in
// `dit_lora.h` (row ROAD-V1-DIT-LORA). This header provides the `Ltx2Lora*`
// type aliases so existing includes compile unchanged. LTX-2.5's ComfyUI
// prefix is `{"diffusion_model."}` (LTXV_LORA_COMFY_RENAMING_MAP, sd_ops.py:136).
//
// The free functions are NOT aliased here. Call sites use the `Dit*` names
// directly and pass the model's prefix set. The fusion output is byte-identical
// to the pre-generalization implementation — the arithmetic is unchanged.
#pragma once

#include "vllm/model_executor/models/dit_lora.h"

namespace vllm {

using Ltx2LoraSpec = DitLoraSpec;
using Ltx2LoraFactorPair = DitLoraFactorPair;
using Ltx2LoraAdapter = DitLoraAdapter;
using Ltx2LoraReferenceFactors = DitLoraReferenceFactors;

}  // namespace vllm
