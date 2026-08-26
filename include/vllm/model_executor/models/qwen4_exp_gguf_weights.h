// `qwen4exp` GGUF support — Qwen3.8-Flash-Next
// (`Qwen4ExpForConditionalGeneration`, HF `model_type: qwen4_exp`).
//
// WHY THIS IS ITS OWN TRANSLATION UNIT, and not a fourth key on
// `vllm::HfConfigFromGguf`. That function asserts its own three architectures
// by name and reports every failure as "qwen3_5 gguf: ...". Reusing it for a
// fourth family is exactly the defect #809 removed: a refusal that names a
// model the user never asked about, in a translation unit that owes none of the
// work. A GGUF family owns its config builder; the dispatch table in
// `entrypoints/model_loader.cpp` owns the mapping.
//
// WHY THE GGUF ARM IS THE PATH AND NOT A FOLLOW-UP. Every safetensors artifact
// of this model is larger than any device this project owns: bf16 ~360 GB,
// official FP8 ~180 GB, NVFP4 ~128 GB, against ~119.6 GiB usable on GB10. The
// UD-IQ1_S GGUF is 67.56 GiB in three shards. See `.agents/specs/
// qwen4-exp-flash-next.md` and issue #1989.
//
// KEY NAMES ARE AN INTEROP CONTRACT, AND THEY ARE NOT SETTLED. Two competing
// llama.cpp pull requests disagree on the `ple.*` spellings and on whether the
// n-gram table is model-level or per-layer. This TU follows **#27742**, which
// is the layout the SHIPPED file uses: model-level `per_layer_token_embd.weight`
// and `blk.N.ple_{key,value,norm_key,norm_query,norm_conv,conv1d}`. Read live
// from `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ1_S`, 2026-08-26; the frozen
// tensor table is `tests/vllm/models/qwen4_exp_gguf_manifest.inc`. If upstream
// renames these before merging, this file changes and the manifest is regrown —
// there is no compatibility shim for a name that never shipped.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// llama.cpp's `general.architecture` value for this family, as written by
// #27742 and as carried by the shipped file. One spelling, one definition.
inline constexpr const char* kQwen4ExpGgufArch = "qwen4exp";

// True when `gguf` carries `general.architecture == kQwen4ExpGgufArch`.
bool IsQwen4ExpGguf(const GgufFile& gguf);

// The GGUF metadata -> `HfConfig` builder for this family. Throws by name on a
// missing required key. Every value it produces is READ FROM THE FILE; the two
// exceptions are annotated at their assignment and are facts the GGUF container
// has no key for.
HfConfig Qwen4ExpHfConfigFromGguf(const GgufFile& gguf);

}  // namespace vllm
