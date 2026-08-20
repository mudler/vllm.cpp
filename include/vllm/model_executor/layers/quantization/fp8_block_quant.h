// Block-wise (fine-grained) FP8: the quantization-config reader, the supported
// shape, and the named refusals for everything else.
//
// UPSTREAM (ported FROM, ground-every-impl rule), pinned vLLM
// `5559679229bc961848b121ccdeaa8fa5d79bec98`:
//   vllm/model_executor/layers/quantization/fp8.py:157-172
//     Fp8Config.from_config — reads `weight_block_size`, `activation_scheme`,
//     `ignored_layers`, and `modules_to_not_convert` as the fallback for the
//     ignore list. Absent means None, which means per-tensor.
//   vllm/model_executor/layers/quantization/fp8.py:115-131
//     Fp8Config.__init__ — validates it: an fp8-serialized checkpoint, exactly
//     2 dimensions, and a dynamic activation scheme. Each of those three is
//     mirrored below as a refusal.
//   vllm/model_executor/layers/quantization/fp8.py:297-298
//     Fp8LinearMethod — `self.block_quant = self.weight_block_size is not None`
//     is the whole dispatch.
//   vllm/model_executor/layers/quantization/fp8.py:378-379, :511
//     the block scale registers as `weight_scale_inv`, not `weight_scale`, and
//     the name is strictly conditional on block quant.
//   vllm/model_executor/layers/quantization/utils/quant_utils.py:510-524,568-569
//     is_layer_skipped — the DEFAULT match is `prefix_full_match`, i.e. exact
//     membership of the module prefix in the ignore list, not a substring test.
//
// HISTORY. `469f38395` (#1166) refused the whole scheme by name here, because
// the dense loader branches on the weight dtype alone and a block-wise weight
// really is `F8_E4M3`: the projection entered the per-tensor arm, asked for
// `<proj>.weight_scale`, and died on `tensor not found` — a sentence that is
// wrong about the world, since the checkpoint is complete and it is this tree
// that lacked an arm. MODEL-FP8-BLOCK-WEIGHT (#1189 M3, spec
// `.agents/specs/model-fp8-block-weight.md`) narrows that refusal: a
// `[128, 128]` `dynamic` checkpoint now LOADS, and only the shapes and schemes
// nothing here can execute are still refused.
//
// SCOPE. Reading the config and refusing what this file does not cover. The
// loader rung lives in `qwen3_5_dense_weights.cpp` and the weight in
// `models/qwen3_5_weights.h`. The linear method landed in #1189 milestone M4
// (`281b4bc76`), so `PrepareQwen3_5Dense` no longer refuses a loaded-but-unread
// block weight: it refuses a device with no block-scaled GEMM.
#pragma once

#include "vt/device.h"  // vt::DeviceType

#include <string>
#include <vector>

namespace vllm {

struct HfConfig;

// The block geometry and ignore list a checkpoint declares, once, validated.
//
// `block_quant` false means the checkpoint declares no `weight_block_size` and
// every other field is unset — the per-tensor world, byte-identical to before
// this row.
struct Fp8BlockQuantConfig {
  bool block_quant = false;
  int64_t block_n = 0;
  int64_t block_k = 0;
  // `dynamic` whenever `block_quant` is true; the reader refuses anything else.
  std::string activation_scheme;
  // `modules_to_not_convert`, or `ignored_layers` when the checkpoint spells it
  // that way. `Qwen/Qwen3.8-27B-FP8` ships ~400 entries here, which is why the
  // loader reads this list rather than inferring exclusion from a dtype probe.
  std::vector<std::string> modules_to_not_convert;

  // Exact-membership test on the MODULE prefix — the tensor name with its
  // trailing `.weight` removed. Mirrors `is_layer_skipped`'s default
  // `prefix_full_match` (`quant_utils.py:517-518,524,568-569`). Upstream first
  // rewrites the list into vLLM module naming (`fp8.py:151-153`); we match in
  // CHECKPOINT naming, which is what this loader has, and the two coincide for
  // every entry that names a real checkpoint module.
  bool ExcludesModule(const std::string& module_prefix) const;
};

// The `weight_block_size` a checkpoint declares, empty when it declares none.
//
// Mirrors `Fp8Config.from_config`: the key is read from `quantization_config`,
// and absent, null, or empty all mean per-tensor. Both spellings are read, the
// top level and the `text_config` nesting a multimodal wrapper can use, because
// the wrapper shape is exactly the one in play on `Qwen3_5ForConditionalGeneration`.
std::vector<int64_t> Fp8WeightBlockSizeOf(const HfConfig& config);

// Reads and VALIDATES the block-quant config, or returns a default-constructed
// value when the checkpoint is not block-wise.
//
// Throws `std::runtime_error` — the type every other load refusal in this tree
// throws, so the C API surfaces it as `VLLM_ERR_MODEL_LOAD` unchanged — for a
// `quant_method` that is not fp8, a `weight_block_size` that is not exactly two
// dimensions, an `activation_scheme` other than `dynamic`, and a block shape
// other than 128x128. The first three mirror upstream's own `ValueError`s
// (`fp8.py:115-131`); the fourth is OUR limit and says so, because #1189's
// kernel and its CPU reference are both 128x128 and a `[64, 128]` checkpoint
// would otherwise load into a weight nothing can execute.
Fp8BlockQuantConfig ReadFp8BlockQuantConfig(const HfConfig& config);

// The pre-load gate, called from `ModelRegistry::Load`. Reads the config for its
// refusals and discards the result; the loader reads it again where it needs the
// geometry. Sited on the registry rather than per loader because
// `weight_block_size` is a property of the checkpoint's quantization config and
// not of one architecture.
void RefuseUnsupportedFp8BlockQuant(const HfConfig& config);

// The M4/M5 seam, narrowing M3's. The forward READS a block-wise weight now
// (`layers::Fp8BlockLinearMethod`, quantization/fp8_block.h), so what is left
// to refuse is a DEVICE with no block-scaled GEMM: `vt::MatmulFp8BlockScaled`
// is a CPU correctness reference and the mainloop-scaled CUTLASS kernel is
// milestone M5. `proj` is the projection that carries the weight, so the
// message names one instead of the class, and `device` is the one that cannot
// run it, so the reader is told what to change rather than what is missing.
[[noreturn]] void RefuseUnrunnableFp8BlockWeight(const std::string& proj,
                                                 vt::DeviceType device);

}  // namespace vllm
