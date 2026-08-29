// Exposed entry point for the interleaved-mRoPE cos|sin table builder that
// lived as a file-static in src/vllm/model_executor/models/qwen3_5.cpp
// (`BuildMropeCosSinHost`, the M3-b VL greedy driver's per-token cache).
//
// The Qwen3.5/3.6 VL image and video drivers call it internally, but a SECOND
// architecture now needs the SAME tables from a DIFFERENT translation unit:
// Qwen4-Exp's Qwen Sparse Attention half of the layer loop
// (`Qwen4ExpLayerKind::kQwenSparseAttention`, row MODEL-MM-QWEN4-EXP W5d,
// issue #2249 item 5). `static` gives the definition INTERNAL LINKAGE, so no
// other TU can name it and the QSA block would have had to grow a second copy
// of the axis selection and the angle math. Two implementations of one table
// is exactly the parallel hand-written path AGENTS.md `## Shared seams`
// forbids, and mRoPE is the kind of arithmetic where a second copy diverges
// silently: a wrong axis still produces plausible tokens.
//
// This mirrors the `RunGdnBlockPaged` (qwen3_5_gdn_block.h, #2110) and
// `RunMoeBlock` (qwen3_5_moe_block.h) seams already opened for this row, and
// takes the SIMPLER of the two available shapes. Those two needed a thin
// wrapper because they take `StepDevInputs` / `MoeBlockWeights`-adjacent types
// that qwen3_5.cpp declares privately. This one needs NO wrapper at all: every
// type in the signature is already public (`std::vector`, `int64_t`,
// `vllm::HfConfig`), so the definition keeps its exact body and its exact place
// in qwen3_5.cpp and only loses the `static` keyword. The bytes of the
// computation are unchanged, which is the property the W5d-2 value gate
// (tests/vllm/models/test_qwen3_5_mrope.cpp) pins against the values the
// file-static produced at base SHA 94de63ff5.
//
// UPSTREAM. The interleaved 3-section axis selection mirrors vLLM's
// `_triton_qwen2vl_mrope_forward` masks
// (vllm/model_executor/layers/rotary_embedding/mrope.py:60-63 at the pinned
// 5559679229), whose `apply_interleaved_rope` (:190-198) states the same layout
// as a tensor rewrite; the chunked branch mirrors the same function's `else`
// arm (:66-70), and the per-pair frequency mirrors `RotaryEmbeddingBase`'s
// inv_freq (`base ** (-2 * pair / rotary_dim)`).
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Build the per-token mRoPE cos|sin cache [T, config.rotary_dim] (host f32)
// from the 3-D positions `positions3` [3,T] (axis-major: t rows, then h, then
// w) and `config.rope_parameters`. Row `i` holds `rotary_dim/2` cosines
// followed by `rotary_dim/2` sines, which is the layout
// `vt::RopeCosSinCacheKernel` writes and the fused `AttnQkNormRopeGate` reads,
// so the same buffer serves the 1-D text path and the 3-D visual path.
//
// `config.rope_parameters.mrope_interleaved` selects the axis layout:
// interleaved [THTHWHTHW...TT] when true, chunked [TTT...HHH...WWW] when false.
// `mrope_section` must hold exactly 3 entries summing to `rotary_dim/2`.
// No Llama3 frequency scaling — mrope's rope_type is identity.
std::vector<float> BuildMropeCosSinHost(const std::vector<int32_t>& positions3,
                                        int64_t T, const HfConfig& config);

}  // namespace vllm
