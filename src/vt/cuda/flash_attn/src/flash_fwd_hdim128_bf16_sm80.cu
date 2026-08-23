// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
//
// vllm.cpp: ADDED for the LTX-2.5 DiT self-attention (#1551). It is the SECOND
// non-split instantiation in this directory, after the head_dim-64 Whisper encoder
// one, and it is the same one-line ask for a different (Headdim, entry-point) pair:
//
//  1. head_dim 128 — LTX-2.5's video stream is 32 heads x 128
//     (include/vllm/model_executor/models/ltx2.h:124-125). It is one of upstream's
//     own head dims and the generic launcher already exists
//     (flash_fwd_launch_template.h:224-259 `run_mha_fwd_hdim128`); the head_dim-128
//     SPLIT-KV instantiation has shipped here since the Qwen3-dense decode path, so
//     the kernel traits, the `mma.sync` bodies and the softmax are all already
//     compiled and exercised at this width. Only the explicit NON-SPLIT
//     instantiation was missing.
//  2. the NON-SPLIT `run_mha_fwd_` entry (upstream's `mha_fwd`) rather than
//     `run_mha_fwd_splitkv_dispatch`. LTX's DiT self-attention is dense, non-paged,
//     one request, q/k/v the same length — upstream's plain batch forward. vLLM
//     reaches the same forward for a dense encoder-style attention
//     (vllm/model_executor/models/whisper.py WhisperEncoderAttention:255 ->
//     forward:298-317 -> flash_attn_varlen_func, causal=False).
//
// SHARED MEMORY, which is the one thing that differs from the hd-64 file and the
// reason head_dim 128 is worth stating rather than assuming. On a non-sm8x target
// `run_mha_fwd_hdim128` selects `Flash_fwd_kernel_traits<128, 128, 64, 4, false,
// false, T>`, whose kSmemSize is over CUDA's default 48 KiB launch cap. That is NOT
// the #1544 problem: `run_flash_fwd` raises the cap itself with
// `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
// before the launch (flash_fwd_launch_template.h:85-88), which is exactly the opt-in
// `LaunchAttentionDenseFlash` does not make and is bounded by instead. So the
// vendored path carries its own opt-in and needs nothing added here.
//
// NOTHING in the vendored template changed — this file only asks the existing
// generic `run_mha_fwd_hdim128<T, Is_causal>` for one more (Headdim, entry-point)
// pair, so the compiled 128 / 192 / 256 split-KV kernels every text and MLA path
// calls are byte-identical. Non-causal only: LTX's DiT self-attention is
// bidirectional and no causal non-split hd-128 caller exists, so the causal
// instantiation is deliberately not built (it is one line away if one appears, and
// `LaunchDenseFA2Bf16` refuses causal by name until then).
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
