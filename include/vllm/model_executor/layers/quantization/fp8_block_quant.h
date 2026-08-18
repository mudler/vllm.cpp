// Block-wise (fine-grained) FP8 detection and its named refusal.
//
// UPSTREAM (ported FROM, ground-every-impl rule), pinned vLLM
// `5559679229bc961848b121ccdeaa8fa5d79bec98`:
//   vllm/model_executor/layers/quantization/fp8.py:161
//     Fp8Config.from_config — reads `weight_block_size` out of the checkpoint's
//     quantization config. Absent means None, which means per-tensor.
//   vllm/model_executor/layers/quantization/fp8.py:115-132
//     Fp8Config.__init__ — validates it: an fp8-serialized checkpoint, exactly
//     2 dimensions, and a dynamic activation scheme.
//   vllm/model_executor/layers/quantization/fp8.py:297-298
//     Fp8LinearMethod — `self.block_quant = self.weight_block_size is not None`
//     is the whole dispatch, and this tree has no arm to dispatch TO.
//   vllm/model_executor/layers/quantization/fp8.py:378-379, :511
//     the block scale registers as `weight_scale_inv`, not `weight_scale`, and
//     the name is strictly conditional on block quant.
//
// WHY THIS FILE EXISTS. `include/.../quantization/fp8.h` mirrors the PER-TENSOR
// arm and says so on its first line. A block-wise checkpoint used to enter that
// arm anyway, because the dense loader branches on the weight dtype alone
// (`qwen3_5_dense_weights.cpp:479`) and the block-wise weight really is
// `F8_E4M3`. The load then asked for `<proj>.weight_scale`
// (`qwen3_5_weights.cpp:458`), which a block-wise checkpoint does not have, and
// died on `tensor not found`. That sentence is wrong about the world: the
// checkpoint is complete, and it is this tree that is missing an arm. Issue
// #1166, spec `.agents/specs/fp8-blockwise-refusal.md`.
//
// SCOPE. Detect and refuse by name. Reading `weight_scale_inv`, applying a
// 128x128 block scale, and the dynamic per-token activation quant upstream
// pairs with it are OWED, not done, and the refusal names the issue that owes
// them.
#pragma once

#include <string>
#include <vector>

namespace vllm {

struct HfConfig;

// The `weight_block_size` a checkpoint declares, empty when it declares none.
//
// Mirrors `Fp8Config.from_config`: the key is read from `quantization_config`,
// and absent, null, or empty all mean per-tensor. Both spellings are read, the
// top level and the `text_config` nesting a multimodal wrapper can use, because
// the wrapper shape is exactly the one in play on `Qwen3_5ForConditionalGeneration`.
std::vector<int64_t> Fp8WeightBlockSizeOf(const HfConfig& config);

// Refuses a block-wise FP8 checkpoint by name, or returns when the checkpoint
// is not block-wise.
//
// Throws `std::runtime_error`, the type every other load refusal in this tree
// throws, so the C API surfaces it as `VLLM_ERR_MODEL_LOAD` unchanged.
void RefuseUnsupportedFp8BlockQuant(const HfConfig& config);

}  // namespace vllm
