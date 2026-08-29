// Exposed entry point for the file-static bf16/NVFP4 sparse-MoE block that lives
// in the anonymous namespace of src/vllm/model_executor/models/qwen3_5.cpp
// (`MoeBlock` @ the Qwen3.6-35B forward, over `ExpertMlp` / `SharedExpert` /
// `vt::MoeRouterTopK` / `vt::MoeCombine`).
//
// The Qwen3.6-35B forward (`Qwen3_5Model::Forward`) calls `MoeBlock` internally,
// but a NEW full-attention MoE — Qwen3-Coder `Qwen3MoeForCausalLM` (qwen3_moe.cpp,
// W3) — must reuse the SAME block from a DIFFERENT translation unit. `MoeBlock`
// has INTERNAL LINKAGE (it sits in that anonymous namespace), so it cannot be
// called cross-TU directly. THIS SENTENCE USED TO GIVE A DIFFERENT REASON —
// that `MoeBlock` "takes/returns the qwen3_5.cpp-internal `Dev`/`DBuf` types
// (which each TU defines privately)" — and that reason was true when this
// header was written (`7c96dd9fc`) and stopped being true at `f730eb11c`
// (ENG-HYBRID-PLACEMENT, #2046), which replaced qwen3_5.cpp's private copies
// with `vllm::dense_attn::Dev` / `DBuf` from the public `dense_device_glue.h`.
// Corrected here because `qwen3_5_gdn_block.h` copied the stale claim from it.
// This header
// exposes a thin public wrapper (`RunMoeBlock`) over primitive vt:: types: it
// builds the internal `Dev`, calls `MoeBlock`, and hands back the combined
// device buffer as an owning `MoeBlockOutput` whose deleter returns the pool
// block to the shared DevicePool (exactly the `WrapDeviceLogits` release
// pattern). The 35B path is UNTOUCHED and byte-identical — this only ADDS a
// second caller. See .agents/specs/sweep-qwen3-coder-30b.md §3b SEAM GAP #2.
#pragma once

#include <cstdint>
#include <memory>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // MoeBlockWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// Owning device-resident output of the sparse-MoE block: a [T,H] bf16 tensor
// view plus the shared_ptr that returns its pool block to the DevicePool when
// the last reference drops (the block outlives the wrapper by design so the
// composing forward can keep computing over `tensor`).
struct MoeBlockOutput {
  vt::Tensor tensor;              // [T,H] bf16 device view
  std::shared_ptr<void> storage;  // owns the pool block (Pool().Put on release)
};

// Run the Qwen3 sparse-MoE block over a device-resident hidden `dh` [T,H] bf16
// and return the combined [T,H] bf16 output as an owning device buffer. The
// weight set decides the compute path exactly as the internal `MoeBlock`:
// fp4-resident experts (35B real checkpoint, CUDA) take the fused grouped-GEMM
// path; bf16 experts (Qwen3-Coder, synthetic, GGUF) take the per-expert
// reference path. `config.shared_expert_intermediate_size == 0` (Coder) skips
// the shared expert (SEAM GAP #3); `> 0` (the 35B) runs it unchanged.
MoeBlockOutput RunMoeBlock(vt::Queue& queue, const MoeBlockWeights& weights,
                           const HfConfig& config, const vt::Tensor& dh, int64_t T);


}  // namespace vllm
