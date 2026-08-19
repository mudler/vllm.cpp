// DFlash draft loading from a `dflash`-arch GGUF (`SPEC-DFLASH-GGUF`).
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_

#include <cstdint>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Build the draft's HfConfig from a `dflash` GGUF's metadata, the GGUF
// counterpart of MakeDflashDraftConfig's config.json read.
//
// Two conventions this undoes, BOTH invisible to shape checks:
//   * `dflash.target_layers` is stored +1-offset by llama.cpp's converter, so
//     the rebuilt `dflash_config.target_layer_ids` subtracts one;
//   * the mask token arrives on the STANDARD `tokenizer.ggml.mask_token_id`
//     key, not a dflash-specific one.
// `vocab_size` is left 0: a DFlash draft carries no vocab key and no
// embed/lm_head tensors because it SHARES the target's.
//
// SPEC-DFLASH2 W1 (#1314): it also carries `dflash.attention.causal` into
// `raw["is_causal"]` when the file declares it. That KV is the GGUF spelling of
// the HF top-level `is_causal`, and `ResolveQwen3DFlashAttnModes` resolves it
// ahead of the pattern-derived rule. The published DFlash2 GGUF sets it false
// beside an ALL-TRUE sliding-window pattern, so without the read every layer
// runs causal. A DFlash1 GGUF declares no such key and is unchanged.
HfConfig MakeDflashGgufConfig(const GgufFile& gguf);

// SPEC-DFLASH2 W1 (#1314): whether a `dflash`-arch GGUF is a DFlash2 drafter.
//
// The architecture string CANNOT answer this. `z-lab/Qwen3.8-27B-DFlash2-GGUF`
// @ `57ab3265056d4024870b0621cfc2c127537020ed` writes
// `general.architecture = "dflash"`, byte-identical to a DFlash1 drafter, and a
// GGUF carries no `architectures` array for the config.json-keyed classification
// to read. So the discriminator is the DFlash2-only metadata:
// `dflash.conv_kernel_size`, `dflash.selector_rank` and `dflash.selector_top_k`.
// Verified against both published artifacts on 2026-08-19 -- the DFlash2 file
// carries all three, and `muse-glimmer-30b-gguf/dflash-kquant.gguf` carries none.
//
// `matched_key` receives the first key that answered, so the refusal can name
// what identified the file. It may be null.
bool IsDflash2Gguf(const GgufFile& gguf, std::string* matched_key = nullptr);

// Load the draft's weights from the same file. Norms are read RAW - the
// `DFlashModel` converter class does NOT inherit the Qwen3Next `(w + 1)` norm
// shift, so unlike the trunk and the MTP head these must not be un-shifted.
// embed_tokens / lm_head are NOT read here; they come from the target.
Qwen3DFlashWeights LoadQwen3DFlashFromGguf(const GgufFile& gguf,
                                           const HfConfig& config,
                                           int64_t num_taps,
                                           int32_t mask_token_id);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_
