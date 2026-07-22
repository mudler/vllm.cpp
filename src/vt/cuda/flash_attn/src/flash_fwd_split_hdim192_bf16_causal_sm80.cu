// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
//
// vllm.cpp: ADDED at MLA campaign W5 — see the non-causal sibling
// flash_fwd_split_hdim192_bf16_sm80.cu for the full rationale. The CAUSAL arm is
// what `run_prefill_new_tokens` needs
// (vllm/v1/attention/backends/mla/prefill/flash_attn.py:223 `causal=True`); the
// non-causal arm is what `run_prefill_context_chunk` needs (`:246`, "Context is
// unmasked").
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
