// Ported from: vllm/utils/torch_utils.py @ 555967922 — :64-67
//               MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP, :310-362
//               get_kv_cache_quant_algo_string, :374-392
//               resolve_kv_cache_dtype_string. See include/vllm/config/cache.h.
#include "vllm/config/cache.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace vllm {

namespace {

using nlohmann::json;

std::string Lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

// torch_utils.py:64-67 MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP. Two entries, and the
// second one is deliberately kept: `nvfp4` resolves to the string `nvfp4`, which
// `vllm::v1::ParseCacheDType` then REFUSES by name (KV-NVFP4-TURBO owns it).
// Dropping it here would turn a declared-and-unimplemented KV format into
// "nothing declared", which is the silent-default failure this whole path is
// built to avoid.
std::optional<std::string> MapModeloptKvAlgo(const std::string& algo_lower) {
  if (algo_lower == "fp8") return std::string("fp8_e4m3");
  if (algo_lower == "nvfp4") return std::string("nvfp4");
  return std::nullopt;
}

// torch_utils.py:329-346 — the DICT spelling of `kv_cache_scheme`.
std::optional<std::string> KvAlgoFromObject(const json& kv_algo) {
  const bool dynamic_false = kv_algo.contains("dynamic") &&
                             kv_algo["dynamic"].is_boolean() &&
                             !kv_algo["dynamic"].get<bool>();
  // Written as a statement rather than a conditional expression: GCC 13 at -O2
  // cannot see through the inlined `std::optional` ternary and reports
  // `-Wmaybe-uninitialized` on the `*num_bits` reads below, which is -Werror in
  // this tree's Release and RelWithDebInfo builds.
  std::optional<int> num_bits;
  if (kv_algo.contains("num_bits") && kv_algo["num_bits"].is_number_integer()) {
    num_bits = kv_algo["num_bits"].get<int>();
  }
  const std::string type = kv_algo.contains("type") && kv_algo["type"].is_string()
                               ? kv_algo["type"].get<std::string>()
                               : std::string();
  if (dynamic_false && num_bits.has_value() && *num_bits == 8 && type == "float") {
    return std::string("fp8");
  }
  if (num_bits.has_value() && *num_bits == 4 && type == "float") {
    return std::string("nvfp4");
  }
  return std::nullopt;
}

}  // namespace

std::optional<std::string> GetKvCacheQuantAlgoString(
    const std::string& quant_config_json) {
  if (quant_config_json.empty()) return std::nullopt;
  json cfg;
  try {
    cfg = json::parse(quant_config_json);
  } catch (const json::exception&) {
    // A malformed quantization config is not this resolver's error to raise —
    // the weight loader reports it with far more context. "Nothing declared" is
    // the honest answer here.
    return std::nullopt;
  }
  if (!cfg.is_object()) return std::nullopt;

  // torch_utils.py:319 — only modelopt configs carry `kv_cache_quant_algo`.
  const std::string quant_method =
      cfg.contains("quant_method") && cfg["quant_method"].is_string()
          ? cfg["quant_method"].get<std::string>()
          : std::string();
  const json& inner = (cfg.contains("quantization") && cfg["quantization"].is_object())
                          ? cfg["quantization"]
                          : cfg;
  // THE MARKER IS TWO UPSTREAM FUNCTIONS, NOT ONE, and reading only the first
  // gets this wrong. `get_kv_cache_quant_algo_string` tests
  // `quant_cfg.get("quant_method", "").startswith("modelopt")` at the TOP level
  // (`torch_utils.py:319`) — which a ModelOpt 0.29.0-and-before
  // `hf_quant_config.json` does not carry, because that file nests the algorithm
  // under `"quantization"` and names the producer at the top
  // (`{"producer":{"name":"modelopt"},"quantization":{...}}`). Upstream STILL
  // resolves it, because `ModelArchConfigConvertorBase.
  // _normalize_quantization_config`
  // (`vllm/transformers_utils/model_arch_config_convertor.py:208-247`) runs
  // FIRST, at `ModelConfig.__post_init__` (`vllm/config/model.py:577` ->
  // `get_model_arch_config`), and INJECTS `quant_cfg["quant_method"]` when
  // `producer["name"] == "modelopt"` OR the nested document carries
  // `modelopt_quant_config` (`:216-235`); it mutates the very dict
  // `hf_config.quantization_config` names, so the test at `:319` sees the
  // injected marker. `:238-246` then lower-cases it, which is what `Lower()`
  // mirrors here.
  //
  // MEASURED 2026-08-22 by running BOTH pinned functions (extracted with `ast`
  // from the files at `555967922`, not transcribed) over the live documents:
  // `nvidia/Llama-3.3-70B-Instruct-FP8`'s producer-only `hf_quant_config.json`
  // answers `None` before normalization and `'fp8_e4m3'` after it. Accepting
  // the producer name here is that two-step chain collapsed into one function,
  // and it is a MIRROR rather than a widening.
  //
  // ONE DIFFERENCE REMAINS and is recorded rather than copied: the injector
  // RAISES `ValueError: Unknown ModelOpt quant algo: <algo>` (`:235`) when the
  // producer is modelopt and the nested `quant_algo` is neither FP8-family nor
  // NVFP4 — which is what `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`'s legacy file
  // (`MIXED_PRECISION`) gets, and we answer `fp8_e4m3` for it instead. That
  // refusal is a WEIGHT-half validation living in a config convertor this port
  // does not have, and importing it into the KV resolver would refuse a
  // checkpoint whose weights we load; the spec's `## Owed` names it. It is
  // unreachable for that checkpoint anyway, because its `config.json` carries an
  // inline `quantization_config` and that document wins.
  const std::string producer =
      cfg.contains("producer") && cfg["producer"].is_object() &&
              cfg["producer"].contains("name") && cfg["producer"]["name"].is_string()
          ? cfg["producer"]["name"].get<std::string>()
          : std::string();
  // THE INJECTOR'S NESTED DOCUMENT IS NOT THE READER'S. `:217` is
  // `quant_cfg.get("quantization", {})` and falls back to an EMPTY object,
  // while `torch_utils.py:321` is `quant_cfg.get("quantization", quant_cfg)` and
  // falls back to the whole document. Two different fallbacks in two different
  // functions, so two variables here: `modelopt_inner` decides whether a marker
  // is injected, `inner` (above) decides where the KV key is looked up. Reading
  // the legacy key out of `inner` made a TOP-LEVEL `modelopt_quant_config` with
  // no `quantization` object a marker, which upstream never treats as one.
  static const json kEmptyObject = json::object();
  const json& modelopt_inner =
      (cfg.contains("quantization") && cfg["quantization"].is_object())
          ? cfg["quantization"]
          : kEmptyObject;
  // `_normalize_quantization_config:218-220` — the legacy nested shape, which
  // names no producer and is recognised by the key alone.
  const bool legacy_modelopt = modelopt_inner.contains("modelopt_quant_config");
  const auto starts_with_modelopt = [](const std::string& s) {
    return s.rfind("modelopt", 0) == 0;
  };
  // AND THE INJECTION IS GUARDED. `:224` is `if quant_algo is not None`, read
  // out of that same nested document, and only then does `:225-235` write a
  // marker: `modelopt` for {FP8, FP8_PER_CHANNEL_PER_TOKEN, FP8_PB_WO},
  // `modelopt_fp4` for NVFP4, and a `ValueError` for anything else. A modelopt
  // producer that declares NO `quant_algo` therefore gets no marker at all, and
  // `torch_utils.py:319` answers `None` for it.
  //
  // The two outcomes for a `quant_algo` that IS present are "inject" and
  // "raise", and the paragraph above records why this port takes the raise's arm
  // rather than importing it. So the mirrored test is exactly
  // `quant_algo is not None`, and what it newly refuses are the three shapes
  // that used to resolve to `fp8_e4m3` here while upstream answered `None`: a
  // `producer.name` of `modelopt` with no `quant_algo`, a
  // `quantization.modelopt_quant_config` with no `quant_algo`, and a top-level
  // `modelopt_quant_config` with no `quantization` key.
  const bool has_quant_algo = modelopt_inner.contains("quant_algo") &&
                              !modelopt_inner["quant_algo"].is_null();
  // `quant_method` is prefix-matched and case-folded because upstream lower-cases
  // it before testing (`:238-246` then `torch_utils.py:319`); the producer name
  // is neither, because `:222` is a raw `==` against the literal and nothing
  // normalises it first. Same file, two different tests, mirrored separately.
  const bool injected_marker =
      (producer == "modelopt" || legacy_modelopt) && has_quant_algo;
  if (!starts_with_modelopt(Lower(quant_method)) && !injected_marker) {
    return std::nullopt;
  }

  // torch_utils.py:322-328 — the four spellings, in upstream's own order.
  const json* kv_algo = nullptr;
  const json* const candidates[] = {&inner, &cfg};
  for (const json* obj : candidates) {
    for (const char* key : {"kv_cache_scheme", "kv_cache_quant_algo"}) {
      if (obj->contains(key) && !(*obj)[key].is_null()) {
        kv_algo = &(*obj)[key];
        break;
      }
    }
    if (kv_algo != nullptr) break;
  }
  // Upstream's order is scheme(inner), scheme(outer), algo(inner), algo(outer);
  // the loop above is scheme(inner), algo(inner), scheme(outer), algo(outer).
  // They differ only for a config that carries an inner `kv_cache_quant_algo`
  // AND an outer `kv_cache_scheme`, which no shipped checkpoint does — recorded
  // rather than silently equated.
  if (kv_algo == nullptr) return std::nullopt;

  if (kv_algo->is_object()) {
    const std::optional<std::string> named = KvAlgoFromObject(*kv_algo);
    if (!named.has_value()) {
      std::cerr << "vllm.cpp: WARNING unknown kv_cache_quant_algo object in the "
                   "model quantization config; falling back to 'auto' "
                   "(torch_utils.py:339-346)\n";
      return std::string("auto");
    }
    const std::optional<std::string> mapped = MapModeloptKvAlgo(*named);
    return mapped.has_value() ? mapped : std::optional<std::string>("auto");
  }
  if (kv_algo->is_string()) {
    const std::string algo_lower = Lower(kv_algo->get<std::string>());
    const std::optional<std::string> mapped = MapModeloptKvAlgo(algo_lower);
    if (mapped.has_value()) return mapped;
    std::cerr << "vllm.cpp: WARNING unknown kv_cache_quant_algo '"
              << kv_algo->get<std::string>()
              << "' in the model quantization config (supported: fp8, nvfp4); "
                 "falling back to 'auto' (torch_utils.py:351-361)\n";
    return std::string("auto");
  }
  return std::nullopt;
}

ResolvedCacheDTypeString ResolveKvCacheDTypeString(
    const std::string& requested, const std::string& quant_config_json) {
  ResolvedCacheDTypeString out;
  // torch_utils.py:380-381 — an explicit choice is returned UNCHANGED and the
  // checkpoint is never consulted. The operator outranks the checkpoint.
  if (!requested.empty() && requested != kDefaultCacheDType) {
    out.cache_dtype = requested;
    out.declared_by_checkpoint = false;
    return out;
  }
  const std::optional<std::string> declared =
      GetKvCacheQuantAlgoString(quant_config_json);
  if (declared.has_value() && *declared != kDefaultCacheDType) {
    out.cache_dtype = *declared;
    out.declared_by_checkpoint = true;
    return out;
  }
  // torch_utils.py:391-392 — nothing declared, or declared in a shape that fell
  // back to "auto". Either way the model dtype wins downstream.
  out.cache_dtype = kDefaultCacheDType;
  out.declared_by_checkpoint = false;
  return out;
}

std::string ReadQuantConfigJson(const std::string& model_dir) {
  namespace fs = std::filesystem;
  if (model_dir.empty()) return std::string();
  std::error_code ec;
  const fs::path dir(model_dir);
  if (!fs::is_directory(dir, ec)) return std::string();

  // UPSTREAM ORDER, and it is the CURRENT file first. `config.py:751-761`:
  //
  //   quantization_config = config_dict.get("quantization_config", None)
  //   if quantization_config is None and file_or_path_exists(
  //           model, "hf_quant_config.json", revision):
  //       quantization_config = get_hf_file_to_dict("hf_quant_config.json", ...)
  //
  // carrying upstream's own two comments: "ModelOpt 0.31.0 and after saves the
  // quantization config in the model config file", and the separate file is
  // "ModelOpt 0.29.0 and before". So the legacy file is a FALLBACK and is never
  // opened when `config.json` carries the inline document. That matters because
  // checkpoints get re-quantized in place: a repository that grew an inline
  // `quantization_config` and kept its old `hf_quant_config.json` beside it
  // resolves to what the CURRENT file says rather than to the stale one. W3
  // shipped this pair in the other order and nothing gated it; the case that
  // does now is G10 in tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp.
  const fs::path config = dir / "config.json";
  if (fs::is_regular_file(config, ec)) {
    std::ifstream in(config, std::ios::binary);
    if (in) {
      try {
        nlohmann::json doc = nlohmann::json::parse(in);
        if (doc.is_object() && doc.contains("quantization_config") &&
            doc["quantization_config"].is_object()) {
          return doc["quantization_config"].dump();
        }
      } catch (const nlohmann::json::exception&) {
        // A malformed `config.json` is not this resolver's error to raise, and
        // it is not a reason to reach for the legacy file either: upstream
        // parses this file once and would have failed there. "Nothing declared"
        // is the honest answer; the loader reports the parse with its own
        // context.
        return std::string();
      }
    }
  }
  const fs::path hf_quant = dir / "hf_quant_config.json";
  if (fs::is_regular_file(hf_quant, ec)) {
    std::ifstream in(hf_quant, std::ios::binary);
    if (in) {
      std::ostringstream ss;
      ss << in.rdbuf();
      return ss.str();
    }
  }
  return std::string();
}

}  // namespace vllm
