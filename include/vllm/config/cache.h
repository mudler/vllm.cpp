// Ported from: vllm/config/cache.py @ 555967922 (CacheDType:19-36,
//               cache_dtype:76, calculate_kv_scales:111) plus the two resolvers
//               that turn a checkpoint's declaration into that string:
//               vllm/utils/torch_utils.py:64-67 (MODELOPT_TO_VLLM_KV_CACHE_
//               DTYPE_MAP), :310-362 (get_kv_cache_quant_algo_string) and
//               :374-392 (resolve_kv_cache_dtype_string).
//
// This is the CONFIG half of `KV-FP8` W3: how `--kv-cache-dtype` and the
// checkpoint's own `kv_cache_quant_algo` combine into ONE resolved CacheDType
// string, before anything sizes a block or writes a byte. `include/vllm/v1/
// kv_cache_dtype.h` (W1) then turns that string into a storage dtype and an fp8
// interpretation; this file decides WHICH string it is handed.
//
// THE ORDER IS UPSTREAM'S AND IT MATTERS. `resolve_kv_cache_dtype_string`
// (`torch_utils.py:374-392`) returns an explicit user value UNCHANGED and only
// consults the checkpoint when the user said "auto". So `--kv-cache-dtype
// bfloat16` on an FP8-declaring checkpoint serves bf16 — the operator wins —
// and `--kv-cache-dtype fp8` on a checkpoint that declares nothing is likewise
// honoured. `attention.py:279-290` re-applies the same precedence defensively
// and says so in its own comment ("an explicit choice (e.g. bfloat16) must
// win").
//
// WHAT A DECLARATION IS AND IS NOT. `Declared()` answers "did the checkpoint ask
// for a quantized KV cache", and it is a DIFFERENT question from "did the
// checkpoint ship k/v scales". A checkpoint can answer yes to the first and no
// to the second; see
// `include/vllm/model_executor/layers/quantization/kv_cache.h` for why the two
// must not collapse into one fallthrough.
//
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` — the #1574 campaign
// subject — is NOT that checkpoint, although its legacy `hf_quant_config.json`
// reads like it: its `config.json:quantization_config` declares no
// `kv_cache_*` key, and the inline document is the one that is read
// (`transformers_utils/config.py:751-761`). It therefore answers NO to the
// first question on both engines, and an fp8 KV run of it needs
// `--kv-cache-dtype fp8` typed explicitly on each side. MEASURED from the live
// artifact 2026-08-22.
#ifndef VLLM_CONFIG_CACHE_H_
#define VLLM_CONFIG_CACHE_H_

#include <optional>
#include <string>

namespace vllm {

// vllm/config/cache.py:76 — "Data type for kv cache storage. If 'auto', will use
// model data type." The default every surface starts from.
inline constexpr const char* kDefaultCacheDType = "auto";

// The outcome of resolving `--kv-cache-dtype` against the checkpoint.
struct ResolvedCacheDTypeString {
  // The CacheDType string to hand to `vllm::v1::ParseCacheDType`.
  std::string cache_dtype = kDefaultCacheDType;
  // True when `cache_dtype` came from the CHECKPOINT's `kv_cache_quant_algo`
  // rather than from the caller — i.e. the caller said "auto" and the checkpoint
  // declared a KV quantization algorithm (`torch_utils.py:381-390`). False both
  // when the caller named a dtype and when nothing declared one, because those
  // are different facts from this one and the callers below need to tell them
  // apart.
  bool declared_by_checkpoint = false;
};

// `torch_utils.py:310-362` get_kv_cache_quant_algo_string. `quant_config_json`
// is the raw text of the checkpoint's `hf_quant_config.json` (or the
// `quantization_config` object out of `config.json`); an empty string means the
// checkpoint has neither. Returns nullopt when no algorithm is declared, and
// "auto" when one is declared in a shape this port does not recognize —
// upstream's own safe fallback, which it reaches with a warning rather than by
// guessing.
//
// Only `quant_method` values that start with "modelopt" are read, exactly as
// upstream (`:319`): the compressed-tensors `kv_cache_scheme` route is a
// different surface (`attention.py:283-290`) and is owed, not silently folded
// in here.
std::optional<std::string> GetKvCacheQuantAlgoString(
    const std::string& quant_config_json);

// `torch_utils.py:374-392` resolve_kv_cache_dtype_string. `requested` is what
// the operator typed (or `kDefaultCacheDType`); `quant_config_json` is as above.
ResolvedCacheDTypeString ResolveKvCacheDTypeString(
    const std::string& requested, const std::string& quant_config_json);

// Read the `quantization_config` object inside `config.json` from a model
// directory, falling back to a standalone `hf_quant_config.json`. That order is
// upstream's (`transformers_utils/config.py:751-761`) and it is the inline
// document first: ModelOpt writes it into `config.json` from 0.31.0 on, and the
// separate file is what 0.29.0 and before wrote. Returns "" when the directory
// holds neither — which is not an error: most checkpoints declare no
// quantization at all.
std::string ReadQuantConfigJson(const std::string& model_dir);

}  // namespace vllm

#endif  // VLLM_CONFIG_CACHE_H_
