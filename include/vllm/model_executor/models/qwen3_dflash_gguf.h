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
//
// SPEC-DFLASH2 W5 (#1314): for a DFlash2 file it also carries the four geometry
// keys the two mechanisms need -- `conv_kernel_size`, `conv_group_size`,
// `selector_rank`, `selector_top_k` -- and any of the three output scalars the
// file declares, all under the MEASURED `dflash.<hf key>` spelling. All four
// geometry keys are REQUIRED once the file is classified DFlash2, because
// guessing one sizes the projection or the codebooks wrong, and that is
// acceptance-only and token-invisible.
HfConfig MakeDflashGgufConfig(const GgufFile& gguf);

// The draft GGUF's own vocabulary, taken from `tokenizer.ggml.tokens`.
//
// SPEC-DFLASH2 W5 (#1314). A DFlash draft declares no vocab key -- it shares the
// target's embedding and head -- so `MakeDflashGgufConfig` leaves
// `HfConfig::vocab_size` 0 on purpose. The candidate selector's codebooks are
// `[vocab, rank]` and are checked at load, before the target is anywhere in
// scope, so the check needs a number from the draft file itself. The tokenizer
// array is used rather than the codebook extent because a check that read its
// own expectation off the tensor under test would pass for a transposed or
// truncated codebook. Throws by name when the key is absent.
int64_t DflashGgufTokenizerVocab(const GgufFile& gguf);

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
//
// SPEC-DFLASH2 W5 (#1314): for a DFlash2 file it also resolves the four conv
// tensors and the selector's three, under the GGUF names llama.cpp writes them
// with (`attn_conv_base`, `attn_conv_proj.weight`, `ffn_conv_base`,
// `ffn_conv_proj.weight`, `selector_hidden.weight`,
// `selector_predecessor.weight`, `selector_successor.weight`). Every tensor is
// DEQUANTIZED to bf16 on the way in, which is this lane's design and not a
// fallback -- see the file comment on the .cpp for why, and
// tests/vllm/models/test_qwen3_dflash2_gguf.cpp for the lower bound that a token
// gate cannot supply because of it.
Qwen3DFlashWeights LoadQwen3DFlashFromGguf(const GgufFile& gguf,
                                           const HfConfig& config,
                                           int64_t num_taps,
                                           int32_t mask_token_id);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_
