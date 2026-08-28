// `glm5next` GGUF support and the HF -> GGUF tensor name map for GLM-5.3-Flash
// (`Glm5NextForConditionalGeneration`, HF `model_type: glm5_next`).
//
// WHY THIS IS ITS OWN TRANSLATION UNIT, and not another key on
// `vllm::HfConfigFromGguf`. That function asserts its own three architectures
// by name and reports every failure as "qwen3_5 gguf: ...". Reusing it for a
// fifth family is exactly the defect #809 removed: a refusal that names a model
// the user never asked about, in a translation unit that owes none of the work.
// A GGUF family owns its config builder; the dispatch table in
// `entrypoints/model_loader.cpp` owns the mapping.
//
// WHERE THE FILE COMES FROM. No upstream tool can write one: llama.cpp
// implements no `glm5_next` at our pin `b10451` or at its `master`, and
// `gguf.quants.Q2_K` upstream has `dequantize_blocks` and no
// `quantize_blocks`, so upstream Python cannot even produce a k-quant. W7a
// (#2011) authored `scripts/convert-glm5-next-gguf.py` for exactly this reason.
// **The metadata key spellings and the tensor names below are that converter's,
// and they are an interop contract between two files in this repository.** The
// spellings themselves are llama.cpp's at `b10451` wherever `b10451` has one —
// `class KDA` with `{arch}.kda.head_dim` / `{arch}.kda.gate_lower_bound`
// (`gguf-py/gguf/constants.py:262-264`), the `ssm_conv1d_q/k/v`, `ssm_f_a/f_b`,
// `ssm_g_a/g_b`, `ssm_beta`, `ssm_a`, `ssm_dt`, `ssm_norm` KDA tensor names
// (`src/llama-arch.cpp:465-479`), and the indexer names including
// `indexer_compressor_ape` / `indexer_compressor_gate`
// (`:626-636`). The three `attention.indexer.kpool*` keys have no upstream
// spelling at any revision because no upstream implements this indexer; they
// are OURS, namespaced under the indexer they belong to.
//
// llama.cpp is a STRUCTURAL reference for the container here and is never the
// mirror source. The behaviour source is transformers `v5.16.1`.
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The `general.architecture` value this family is written and read under. One
// spelling, one definition, shared by the reader and by the dispatch table.
inline constexpr const char* kGlm5NextGgufArch = "glm5next";

// True when `gguf` carries `general.architecture == kGlm5NextGgufArch`.
bool IsGlm5NextGguf(const GgufFile& gguf);

// The GGUF metadata -> `HfConfig` builder for this family.
//
// It does NOT build a second notion of what a `glm5_next` config is. It reads
// the container and synthesizes an HF-shaped `text_config` / `vision_config`
// under the SAME key spellings `zai-org/GLM-5.3-Flash`'s `config.json` uses, so
// the result descends through `ParseGlm5NextParams` — one validation surface
// for both sources rather than two that can drift apart on a model with no
// reachable token gate. Throws by name on a missing required key.
HfConfig Glm5NextHfConfigFromGguf(const GgufFile& gguf);

// ---------------------------------------------------------------------------
// The tensor name map.
//
// `hf` is the module path the real checkpoint's `model.safetensors.index.json`
// carries with `model.language_model.` stripped (the GGUF block namespace is
// flat); `gguf` is the `blk.N.` suffix, or the whole name for a model-level
// tensor. Enumerated per layer KIND because the kinds do not overlap: a KDA
// layer carries none of the MLA tensors and vice versa, which is what makes the
// inventory able to CONTRADICT a wrong per-layer schedule.
struct Glm5NextTensorName {
  const char* hf;
  const char* gguf;
};

// Present on every layer regardless of kind, including the flat mHC parameters.
// The checkpoint stores those FLAT on the layer (`hc_attn_fn`, not
// `attn_hc.fn`) and carries NO `hc_head.*` at any layer, which is what
// independently settles the unweighted-mean head collapse: there is nothing to
// weight with.
std::vector<Glm5NextTensorName> Glm5NextCommonTensorMap();
// KDA (`linear_attention`) layers. The checkpoint stores THREE separate
// depthwise convs where the reference uses one grouped conv over the
// concatenated [q; k; v] channel axis.
std::vector<Glm5NextTensorName> Glm5NextKdaTensorMap();
// DSA (`deepseek_sparse_attention`) layers: NoPE MLA plus the indexer, whose
// `index_kpool_compress_{ape,gate}` parameters are this model's net-new k-pool
// compression stage.
std::vector<Glm5NextTensorName> Glm5NextDsaTensorMap();
std::vector<Glm5NextTensorName> Glm5NextDenseMlpTensorMap();
// The non-expert half of a sparse layer. The 288 routed experts are STACKED
// into one 3-D tensor per projection (`ffn_{gate,up,down}_exps.weight`), which
// is llama.cpp's convention and our reader's `kStackedExpertWeight` role, so
// they are not in this map.
std::vector<Glm5NextTensorName> Glm5NextSparseMlpTensorMap();
std::vector<Glm5NextTensorName> Glm5NextStackedExpertTensorMap();
std::vector<Glm5NextTensorName> Glm5NextVisionTensorMap();
std::vector<Glm5NextTensorName> Glm5NextVisionBlockTensorMap();

// Every GGUF tensor name a COMPLETE artifact of `params` carries, generated
// from the per-layer topology rather than transcribed. The structural
// enumeration W1 owes: it is what lets a reader ask what this file should hold
// without having the file.
std::vector<std::string> Glm5NextExpectedGgufTensors(
    const Glm5NextParams& params);

}  // namespace vllm
