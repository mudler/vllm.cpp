// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
//
// vllm.cpp: ADDED at MLA campaign W5. head_dim 192 is the MLA PREFILL QK width
// (qk_nope_head_dim 128 + qk_rope_head_dim 64) for every DeepSeek variant; V is
// zero-padded 128 -> 192 exactly as upstream does
// (vllm/v1/attention/backends/mla/prefill/flash_attn.py:164-168), so the kernel
// itself stays a SYMMETRIC head_dim instantiation. Head dim 192 is one of the
// upstream FA-2 head dims (flash_fwd_launch_template.h:259-274
// `run_mha_fwd_hdim192`); only the explicit instantiation was missing here.
// NOTHING in the vendored template changed — this file only asks the existing
// generic `run_mha_fwd_splitkv_dispatch<T, Headdim, Is_causal>` for one more
// Headdim, so the compiled 128 / 256 kernels the 27B / 35B / Qwen3-dense paths
// call are byte-identical.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
