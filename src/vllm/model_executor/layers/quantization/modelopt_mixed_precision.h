// ModelOpt `MIXED_PRECISION`: resolving a quantization ALGORITHM PER MODULE.
//
// UPSTREAM (ported FROM, ground-every-impl rule) @ pinned oracle
// 5559679229bc961848b121ccdeaa8fa5d79bec98 (vLLM 0.26.0.dev0):
//   vllm/model_executor/layers/quantization/modelopt.py:2279-2410
//     class ModelOptMixedPrecisionConfig, override_quantization_method,
//     _from_config (where `quantized_layers` is read and group_size is seeded)
//   vllm/model_executor/layers/quantization/modelopt.py:2412-2487
//     _resolve_quant_algo — the FIVE resolution strategies, in order
//   vllm/model_executor/layers/quantization/modelopt.py:2489-2505
//     _quantized_layer_prefix_candidates
//   vllm/model_executor/layers/quantization/modelopt.py:282-367
//     ModelOptQuantConfigBase.from_config (both config shapes)
//   vllm/model_executor/layers/quantization/modelopt.py:145-181
//     ModelOptQuantConfigBase.is_layer_excluded
//   vllm/model_executor/layers/quantization/utils/quant_utils.py:510-572
//     is_layer_skipped
//
// WHY THIS EXISTS. Every quantized checkpoint we load so far names ONE scheme
// for the whole model, so `quantization_config` could be read ad-hoc where it
// was needed (`kimi_k3_weights.cpp:171`, `deepseek_v2_weights.cpp:365`) and the
// scheme handed to every layer. `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-
// NVFP4` breaks that assumption inside a single file: 5935 of its 5981
// `quantized_layers` entries are NVFP4 W4A16 with `group_size 16` (routed
// experts, shared experts, `lm_head`), 46 are FP8 W8A8 static (the Mamba2
// `in_proj`/`out_proj` ONLY), and a 72-entry `ignore` list leaves embeddings,
// conv1d, the MoE gates and the attention tower in bf16.
//
// The repo NAME says NVFP4, and reading it as uniformly NVFP4 is precisely the
// failure a token gate CANNOT SEE: dequantizing an FP8 projection to bf16, or
// treating a bf16 projection as quantized, is still numerically fine, so the
// tokens match, the goldens pass, and we move the wrong bytes forever. Hence
// this header, and hence the refusal policy below.
//
// SCOPE. This is a pure config capability: parse + resolve, no kernels, no
// weights, no forward path. Row MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
// W1, issue #517, spec `.agents/specs/nemotron-h-model.md`.
//
// WHO CONSUMES IT. `LoadQwen3_5Dense`
// (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`), the loader
// every consumer of a Qwen3.5-family safetensors checkpoint arrives through.
// It reads this config once for the whole checkpoint and refuses a declared
// algorithm it has no loader for, or a declaration the shipped tensor names
// contradict — `Refusal` and `RefusalForQuantizationConfig` at the bottom of
// this file. `QUANT-QWEN38-27B-NVFP4-ARM` W5, issue #821, campaign #1574, spec
// `.agents/specs/qwen38-27b-quant-arms.md`. Until that landed this header was
// reachable ONLY from its own two unit tests, which is what `AGENTS.md`
// §"Nothing lands dead" calls measuring a class rather than a capability; the
// sentence in divergence 2 below that said so is kept, corrected, rather than
// deleted, because the state it describes was real for the rows that read it.
//
// THREE DELIBERATE DIVERGENCES FROM UPSTREAM, each recorded here rather than
// discovered later:
//
//  1. AN UNRESOLVED ALGORITHM IS REFUSED BY NAME. Upstream's consumer
//     (`get_quant_method`, modelopt.py:2508-2560) has branches for FP8, NVFP4,
//     W4A16_NVFP4 and MXFP8 and falls through to `UnquantizedLinearMethod()`
//     for anything else — including `FP8_PB_WO` and
//     `FP8_PER_CHANNEL_PER_TOKEN`, which are real entries of its own
//     `QUANT_ALGOS` list. Silently dequantizing a quantized layer is invisible
//     to a token gate, so `Resolve` throws and names the algorithm instead.
//     A prefix that is simply ABSENT from `quantized_layers` is NOT this case:
//     upstream leaves it unquantized and so do we, because "not listed" is the
//     checkpoint saying bf16, not an unrepresentable scheme.
//
//  2. HEADER-ONLY, AND UNDER `src/` RATHER THAN `include/`. The resolver is
//     string and JSON logic with no device or kernel dependency, so it needs no
//     translation unit — matching `base_config.h` and
//     `compressed_tensors/schemes/nvfp4.h`, which are header-only too. That
//     also keeps the change clear of the root `CMakeLists.txt`, which currently
//     reds `check-doc-checkpoint` when a source file is added (issue #515).
//
//     Its two sibling headers live under `include/vllm/...`; this one does not,
//     for a reason worth stating rather than leaving to look like an accident.
//     `check-doc-checkpoint` classifies the whole `include/vllm/` prefix as a
//     USER-FACING surface and requires `docs/USAGE.md` to move with it — the
//     list's own comment calls it "user-facing configuration/build/install
//     entrypoints". When this was written nothing consumed the resolver: it was
//     not on the `include/vllm.h` ABI, no loader called it, and no command, C
//     API key, config key or install step changed because of it, which is
//     exactly what AGENTS.md says `docs/USAGE.md` tracks. Putting an internal
//     header there would have forced a `docs/USAGE.md` edit that documented
//     nothing. `src/vllm/model_executor/models/*.h` is the established
//     precedent for an internal header.
//
//     IT IS CONSUMED NOW, AND IT STAYS HERE. `LoadQwen3_5Dense` is a `src/`
//     translation unit, so the consumer needs no `include/vllm/` surface, and
//     the sibling that answers the same question for compressed-tensors
//     (`compressed_tensors/compressed_tensors_config.h`) is consumed by the
//     same file from the same place. Promoting one of two peers would say the
//     two formats sit at different levels of the tree, which they do not.
//     `docs/USAGE.md` still moves with this change, for the reason AGENTS.md
//     §"Say which weights, and from where" gives rather than for a path
//     classification: a second checkpoint became loadable and a reader cannot
//     infer which bytes to feed it.
//
//  3. `Parse` IS TEMPLATED ON THE JSON TYPE so it accepts
//     `nlohmann::ordered_json`. Upstream iterates a Python dict, whose order is
//     INSERTION order; plain `nlohmann::json` sorts object keys
//     lexicographically, which would silently change which entry the
//     group_size seeding (modelopt.py:2360-2372) and the prefix scans
//     (:2450, :2458) return. Callers that care about order fidelity parse with
//     `ordered_json`; both types compile.
#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace layers {
namespace modelopt {

// The quantization algorithms the MIXED_PRECISION consumer implements — the
// exact four `get_quant_method` has branches for (modelopt.py:2526-2560).
// Anything else in the checkpoint is refused by name; see divergence 1 above.
enum class QuantAlgo {
  kUnquantized,  // not listed, or on the `ignore` list: bf16/f16 as loaded
  kFp8,          // "FP8" — per-tensor weight scale + static input scale
  kNvfp4,        // "NVFP4" — W4A4
  kW4A16Nvfp4,   // "W4A16_NVFP4" — 4-bit weights, bf16/f16 activations
  kMxfp8,        // "MXFP8"
};

// Which of the strategies produced the answer. Upstream returns only the algo
// string; we carry the strategy because a resolver that reaches the right
// answer by the wrong route is a latent defect (a `.experts` container that
// resolves by an accidental substring, say), and because it is the only way a
// test can prove each of the five paths is actually exercised.
enum class Resolution {
  kExcluded,       // matched the `ignore` list — checked FIRST, wins outright
  kDirect,         // strategy 1: modelopt.py:2424-2427
  kPacked,         // strategy 2: modelopt.py:2429-2447 (packed_modules_mapping)
  kPrefix,         // strategy 3: modelopt.py:2449-2453
  kExpertsParent,  // strategy 4: modelopt.py:2455-2461 (".experts" -> parent)
  kFusedShards,    // strategy 5: modelopt.py:2463-2486 (qkv / gate_up names)
  kUnlisted,       // no strategy matched — unquantized, and NOT an error
};

inline const char* QuantAlgoName(QuantAlgo a) {
  switch (a) {
    case QuantAlgo::kFp8:
      return "FP8";
    case QuantAlgo::kNvfp4:
      return "NVFP4";
    case QuantAlgo::kW4A16Nvfp4:
      return "W4A16_NVFP4";
    case QuantAlgo::kMxfp8:
      return "MXFP8";
    case QuantAlgo::kUnquantized:
      break;
  }
  return "UNQUANTIZED";
}

struct ModuleQuant {
  QuantAlgo algo = QuantAlgo::kUnquantized;
  Resolution how = Resolution::kUnlisted;
  // The NVFP4-family block size. ZERO for FP8/MXFP8/unquantized, which are not
  // group-quantized — deliberately not a "default 16" that would read as a real
  // group size on a scheme that has none.
  int group_size = 0;

  bool Quantized() const { return algo != QuantAlgo::kUnquantized; }
};

// vLLM's per-model `packed_modules_mapping`: the fused module name a model
// builds (`qkv_proj`) to the checkpoint shard names it is fused FROM
// (`q_proj`, `k_proj`, `v_proj`). Registered by the model, so it is empty until
// a model supplies one — strategies 2 and the fused arm of the exclusion check
// are inert without it, exactly as upstream.
using PackedModulesMapping = std::map<std::string, std::vector<std::string>>;

namespace detail {

inline std::string Upper(std::string s) {
  for (char& c : s) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return s;
}

inline bool StartsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

inline bool EndsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Python `s.rsplit(".", 1)[-1]` / `s.split(".")[-1]`: the whole string when
// there is no separator (which is how the bare "lm_head" key resolves).
inline std::string LastSegment(const std::string& s) {
  const std::size_t dot = s.rfind('.');
  return dot == std::string::npos ? s : s.substr(dot + 1);
}

// Python `s.rsplit(".", 1)[0]`: likewise the whole string when there is no dot.
inline std::string ParentSegment(const std::string& s) {
  const std::size_t dot = s.rfind('.');
  return dot == std::string::npos ? s : s.substr(0, dot);
}

// Python `str.replace(old, new)` — ALL occurrences, which is what
// quant_utils.py:545 relies on.
inline std::string ReplaceAll(std::string s, const std::string& from,
                              const std::string& to) {
  if (from.empty()) return s;
  std::size_t at = 0;
  while ((at = s.find(from, at)) != std::string::npos) {
    s.replace(at, from.size(), to);
    at += to.size();
  }
  return s;
}

// One `fnmatch` token against one character; `*np` lands after the token.
inline bool MatchOneToken(char c, const std::string& pat, std::size_t p,
                          std::size_t* np) {
  if (pat[p] == '?') {
    *np = p + 1;
    return true;
  }
  if (pat[p] == '[') {
    std::size_t i = p + 1;
    bool negate = false;
    if (i < pat.size() && pat[i] == '!') {
      negate = true;
      ++i;
    }
    const std::size_t first = i;
    bool matched = false;
    while (i < pat.size()) {
      if (pat[i] == ']' && i > first) break;
      if (i + 2 < pat.size() && pat[i + 1] == '-' && pat[i + 2] != ']') {
        if (c >= pat[i] && c <= pat[i + 2]) matched = true;
        i += 3;
      } else {
        if (c == pat[i]) matched = true;
        ++i;
      }
    }
    if (i >= pat.size()) {  // unterminated class: Python treats '[' as literal
      *np = p + 1;
      return c == '[';
    }
    *np = i + 1;
    return negate ? !matched : matched;
  }
  *np = p + 1;
  return c == pat[p];
}

// Python `fnmatch.fnmatch` semantics (`*`, `?`, `[seq]`, `[!seq]`), as used by
// modelopt.py:177-179. `*` spans separators because the upstream call has no
// FNM_PATHNAME equivalent, and module paths are dot-separated anyway. ModelOpt
// itself only ever emits `module_path*` (the real checkpoint's one wildcard
// entry is `mtp*`); the rest of the syntax is mirrored so a hand-edited or
// future config cannot be silently mis-read.
//
// ONE KNOWN DIVERGENCE CLASS, measured and deliberately left alone. Swept
// differentially against CPython 3.12 `fnmatch.fnmatchcase`:
//
//   names <=3 chars over {a,b,.,0} x patterns <=5 over {a,.,?,*,[,],!,-}
//     3,183,165 pairs, 0 mismatches
//   names <=2 chars over {a,b,.,0} x patterns <=6 over the same alphabet
//     6,291,453 pairs, 108 mismatches spanning 27 distinct patterns
//
// The 27 is arithmetic, not a tally: every one of the 27 patterns translates to
// `(?s:.)\Z`, which matches a name of length ONE and nothing else, so each
// contributes exactly the 4 one-character names over {a,b,.,0} and 27 x 4 =
// 108. A count of 20 would be impossible on its face (it was in this comment
// until 2026-08-13; the ceiling is 4 per pattern, so 108 needs 27 of them).
//
// All 108 are ONE class and all in the same direction (CPython True, this
// False): a bracket whose contents reduce to a bare `!` once CPython's
// `translate` has DROPPED a reversed range. All 27 have the shape `[X-Y!]`
// with X > Y; `[?-.!]` is the worked example — `?`(0x3f) down to `.`(0x2e) is
// empty, so `stuff` becomes `"!"`, and translate's "negated empty class matches
// any character" rule compiles the whole pattern to `(?s:.)\Z`. Matching that
// means reproducing translate's REWRITING, not its matching: a
// bracket-by-bracket matcher cannot see it.
//
// Not fixed on purpose. It needs a reversed range AND a trailing `!` inside one
// class; ModelOpt emits no bracket expressions at all (`mtp*` is the real
// checkpoint's only wildcard), so nothing in the reachable input space can
// touch it, and the risk of rewriting a matcher to chase a `translate` quirk
// exceeds the risk of leaving it recorded here.
inline bool FnMatch(const std::string& name, const std::string& pat) {
  std::size_t n = 0, p = 0, star = std::string::npos, retry = 0;
  while (n < name.size()) {
    if (p < pat.size() && pat[p] == '*') {
      star = p++;
      retry = n;
      continue;
    }
    std::size_t next = p;
    if (p < pat.size() && MatchOneToken(name[n], pat, p, &next)) {
      ++n;
      p = next;
      continue;
    }
    if (star != std::string::npos) {
      p = star + 1;
      n = ++retry;
      continue;
    }
    return false;
  }
  while (p < pat.size() && pat[p] == '*') ++p;
  return p == pat.size();
}

}  // namespace detail

// ModelOpt MIXED_PRECISION quantization config: the `quantized_layers` map, the
// `ignore` list, and the resolved config-level group size.
class MixedPrecisionConfig {
 public:
  // SELECTION, not parsing: modelopt.py:2333-2339 override_quantization_method
  // + :245-263 _extract_modelopt_quant_algo. It answers "is this config MINE to
  // claim?", so it requires the config to NAME modelopt in `quant_method`
  // before its quant_algo is read at all — that is how upstream tells a
  // ModelOpt `quantization_config` apart from a compressed-tensors one sitting
  // in the same field of the same config.json. `Parse` must NOT share this
  // precondition; see the comment on it.
  template <class Json>
  static bool IsMixedPrecision(const Json& cfg) {
    if (!cfg.is_object()) return false;
    return ExtractQuantAlgo(cfg) == "MIXED_PRECISION";
  }

  // modelopt.py:282-367 from_config + :2349-2410 _from_config.
  //
  // PARSING is deliberately NOT gated on `quant_method`. `from_config` never
  // looks at it: it dispatches on the SHAPE (`"quantization" in config`,
  // :283-318) and reads `quant_algo` out of whichever shape that was. The
  // distinction is not academic — the driver checkpoint's own
  // `hf_quant_config.json`
  // ($CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4, repo
  // nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 @29f2d174) has exactly
  // two top-level keys, `producer` and `quantization`, and names no
  // `quant_method` anywhere in the file. Borrowing the selection hook's
  // precondition here refused that file outright while upstream parsed it and
  // resolved all 5981 entries.
  template <class Json>
  static MixedPrecisionConfig Parse(const Json& cfg) {
    if (!cfg.is_object()) {
      throw std::invalid_argument(
          "modelopt: quantization config must be a JSON object");
    }
    if (ShapeQuantAlgo(cfg) != "MIXED_PRECISION") {
      throw std::invalid_argument(
          "modelopt: not a MIXED_PRECISION config (quant_algo=\"" +
          ShapeQuantAlgo(cfg) + "\")");
    }

    // from_config picks BOTH shapes apart: the nested `{"quantization": {...}}`
    // of hf_quant_config.json and the flat compressed-tensors-style
    // quantization_config of config.json (modelopt.py:283-318).
    const bool nested = cfg.contains("quantization") && cfg["quantization"].is_object();
    const Json& section = nested ? cfg["quantization"] : cfg;

    MixedPrecisionConfig out;

    // exclude_modules: "exclude_modules" nested, "ignore" flat.
    const char* exclude_key = nested ? "exclude_modules" : "ignore";
    if (section.contains(exclude_key)) {
      const Json& ex = section[exclude_key];
      if (!ex.is_array()) {
        throw std::invalid_argument(std::string("modelopt: ") + exclude_key +
                                    " must be a list");
      }
      for (const auto& e : ex) out.exclude_modules_.push_back(e.template get<std::string>());
    }

    // kv cache: a plain algo string nested, a `kv_cache_scheme` dict flat
    // (modelopt.py:294, :306-314).
    if (nested) {
      if (section.contains("kv_cache_quant_algo") &&
          section["kv_cache_quant_algo"].is_string()) {
        out.kv_cache_quant_algo_ =
            detail::Upper(section["kv_cache_quant_algo"].template get<std::string>());
      }
    } else if (section.contains("kv_cache_scheme") &&
               section["kv_cache_scheme"].is_object()) {
      const Json& kv = section["kv_cache_scheme"];
      const bool is_fp8 = kv.contains("type") && kv["type"].is_string() &&
                          kv["type"].template get<std::string>() == "float" &&
                          kv.contains("num_bits") &&
                          kv["num_bits"].is_number_integer() &&
                          kv["num_bits"].template get<int>() == 8;
      if (is_fp8) out.kv_cache_quant_algo_ = "FP8";
    }

    // quantized_layers lives beside quant_algo in whichever shape this is
    // (modelopt.py:2357-2363).
    if (!section.contains("quantized_layers") ||
        !section["quantized_layers"].is_object() ||
        section["quantized_layers"].empty()) {
      throw std::invalid_argument(
          "modelopt: MIXED_PRECISION quant_algo requires a non-empty "
          "'quantized_layers' mapping in the quantization config");
    }
    const Json& layers = section["quantized_layers"];
    out.entries_.reserve(layers.size());
    for (auto it = layers.begin(); it != layers.end(); ++it) {
      const Json& info = it.value();
      if (!info.is_object() || !info.contains("quant_algo") ||
          !info["quant_algo"].is_string()) {
        throw std::invalid_argument(
            "modelopt: quantized_layers entry \"" + it.key() +
            "\" has no string 'quant_algo'");
      }
      Entry e;
      e.name = it.key();
      e.algo = detail::Upper(info["quant_algo"].template get<std::string>());
      if (info.contains("group_size") && info["group_size"].is_number_integer()) {
        e.group_size = info["group_size"].template get<int>();
        e.has_group_size = true;
      }
      out.index_.emplace(e.name, out.entries_.size());
      out.entries_.push_back(std::move(e));
    }

    // group_size: an explicit config-level value wins; otherwise it is SEEDED
    // from the first NVFP4-family entry, defaulting to 16 (modelopt.py:2365-
    // 2377). Upstream then builds ONE nvfp4 config from that single value, so
    // it — not the per-entry field — is what every NVFP4 module gets.
    if (section.contains("group_size") && !section["group_size"].is_null()) {
      const Json& gs = section["group_size"];
      if (gs.is_number_integer()) {
        out.group_size_ = gs.template get<int>();
      } else if (gs.is_string()) {
        try {
          out.group_size_ = std::stoi(gs.template get<std::string>());
        } catch (const std::exception&) {
          throw std::invalid_argument("modelopt: group_size must be an integer");
        }
      } else {
        throw std::invalid_argument("modelopt: group_size must be an integer");
      }
    } else {
      out.group_size_ = 16;
      for (const Entry& e : out.entries_) {
        if (e.algo == "NVFP4" || e.algo == "W4A16_NVFP4") {
          out.group_size_ = e.has_group_size ? e.group_size : 16;
          break;
        }
      }
    }
    return out;
  }

  // The owning model's fused-module mapping. Empty until a model registers one,
  // which is exactly upstream's state for a model that declares none.
  void SetPackedModulesMapping(PackedModulesMapping mapping) {
    packed_modules_mapping_ = std::move(mapping);
  }
  const PackedModulesMapping& packed_modules_mapping() const {
    return packed_modules_mapping_;
  }

  int group_size() const { return group_size_; }
  const std::string& kv_cache_quant_algo() const { return kv_cache_quant_algo_; }
  std::size_t num_quantized_layers() const { return entries_.size(); }
  const std::vector<std::string>& exclude_modules() const {
    return exclude_modules_;
  }

  // What scheme does the module at `prefix` use?
  //
  // Order matters and mirrors get_quant_method (modelopt.py:2508-2524): the
  // `ignore` list is consulted BEFORE `quantized_layers`, so an entry present in
  // both is unquantized. Reversing that would quantize a layer the producer
  // explicitly excluded.
  //
  // Throws std::invalid_argument when the shards of one fused module disagree
  // (upstream ValueError), and std::runtime_error when the resolved algorithm
  // is one this consumer does not implement (divergence 1 above).
  ModuleQuant Resolve(const std::string& prefix) const {
    if (IsLayerExcluded(prefix)) {
      return ModuleQuant{QuantAlgo::kUnquantized, Resolution::kExcluded, 0};
    }
    Resolution how = Resolution::kUnlisted;
    const std::optional<std::string> algo = ResolveQuantAlgoString(prefix, &how);
    if (!algo.has_value()) {
      return ModuleQuant{QuantAlgo::kUnquantized, Resolution::kUnlisted, 0};
    }
    const QuantAlgo a = ToQuantAlgo(*algo, prefix);
    const int gs =
        (a == QuantAlgo::kNvfp4 || a == QuantAlgo::kW4A16Nvfp4) ? group_size_ : 0;
    return ModuleQuant{a, how, gs};
  }

  // modelopt.py:145-181. Exact match (with fused unfusing), then the legacy
  // substring rule kept for pre-0.39 ModelOpt exports, then wildcards.
  bool IsLayerExcluded(const std::string& prefix) const {
    if (exclude_modules_.empty()) return false;
    if (IsLayerSkipped(prefix)) return true;

    static const std::string kLangPrefix = "language_model.";
    for (const std::string& excluded : exclude_modules_) {
      if (excluded == prefix) continue;  // handled by the exact pass above
      if (prefix.find(excluded) != std::string::npos) return true;
      // The `language_model.` clause is mirrored verbatim from
      // modelopt.py:170-172 but is provably redundant: `removeprefix` returns a
      // SUFFIX of `prefix`, so anything found in the stripped string is already
      // found in the full one and the line above has returned. It is kept so
      // the two implementations read alike and a future upstream edit here is
      // easy to follow — it cannot be covered by a test, because no input can
      // reach it first.
      if (detail::StartsWith(prefix, kLangPrefix) &&
          prefix.substr(kLangPrefix.size()).find(excluded) != std::string::npos) {
        return true;
      }
    }
    for (const std::string& pattern : exclude_modules_) {
      if (detail::FnMatch(prefix, pattern)) return true;
    }
    return false;
  }

 private:
  struct Entry {
    std::string name;
    std::string algo;  // stored UPPER-cased, as every upstream return does
    int group_size = 0;
    bool has_group_size = false;
  };

  template <class Json>
  static std::string StringField(const Json& cfg, const char* key) {
    if (!cfg.contains(key) || !cfg[key].is_string()) return std::string();
    return cfg[key].template get<std::string>();
  }

  // The `quant_algo` of whichever config SHAPE this is — the shape dispatch of
  // from_config, modelopt.py:283-318, and like it blind to `quant_method`:
  // nested `{"quantization": {...}}` for hf_quant_config.json, flat for a
  // config.json `quantization_config`.
  template <class Json>
  static std::string ShapeQuantAlgo(const Json& cfg) {
    if (cfg.contains("quantization")) {
      if (!cfg["quantization"].is_object()) return std::string();
      return detail::Upper(StringField(cfg["quantization"], "quant_algo"));
    }
    return detail::Upper(StringField(cfg, "quant_algo"));
  }

  // modelopt.py:245-263 _extract_modelopt_quant_algo — the SELECTION hook, and
  // the ONLY place the `quant_method` precondition belongs.
  template <class Json>
  static std::string ExtractQuantAlgo(const Json& cfg) {
    std::string method = StringField(cfg, "quant_method");
    for (char& c : method) {
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    if (!detail::StartsWith(method, "modelopt")) return std::string();
    return ShapeQuantAlgo(cfg);
  }

  const Entry* Find(const std::string& name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &entries_[it->second];
  }

  // modelopt.py:2489-2505 _quantized_layer_prefix_candidates, order-preserving
  // and de-duplicated exactly as `dict.fromkeys` does.
  static std::vector<std::string> PrefixCandidates(const std::string& prefix) {
    std::vector<std::string> out;
    out.push_back(prefix);
    if (detail::EndsWith(prefix, ".lm_head")) out.push_back("lm_head");

    static const std::string kA = "language_model.model.";
    static const std::string kB = "model.language_model.";
    if (detail::StartsWith(prefix, kA)) {
      out.push_back(kB + prefix.substr(kA.size()));
    } else if (detail::StartsWith(prefix, kB)) {
      out.push_back(kA + prefix.substr(kB.size()));
    }

    std::vector<std::string> deduped;
    for (const std::string& c : out) {
      bool seen = false;
      for (const std::string& d : deduped) seen = seen || d == c;
      if (!seen) deduped.push_back(c);
    }
    return deduped;
  }

  // quant_utils.py:510-572 is_layer_skipped, with skip_with_substr=False.
  bool IsLayerSkipped(const std::string& prefix) const {
    const std::string proj = detail::LastSegment(prefix);
    const auto fused = packed_modules_mapping_.find(proj);

    if (fused != packed_modules_mapping_.end()) {
      // A checkpoint may list the FUSED name directly; honour that first
      // (quant_utils.py:540-541).
      for (const std::string& e : exclude_modules_) {
        if (e == prefix) return true;
      }
      bool have = false;
      bool skipped = false;
      for (const std::string& shard : fused->second) {
        const std::string shard_prefix = detail::ReplaceAll(prefix, proj, shard);
        bool shard_skipped = false;
        for (const std::string& e : exclude_modules_) {
          if (e == shard_prefix) {
            shard_skipped = true;
            break;
          }
        }
        if (!have) {
          have = true;
          skipped = shard_skipped;
        } else if (shard_skipped != skipped) {
          throw std::invalid_argument(
              "modelopt: detected some but not all shards of " + prefix +
              " are quantized; all shards of a fused layer must have the same "
              "precision");
        }
      }
      if (!have) {
        throw std::invalid_argument(
            "modelopt: packed_modules_mapping entry for \"" + proj +
            "\" is empty, so the exclusion of " + prefix + " is undecidable");
      }
      return skipped;
    }

    // quant_utils.py:559-565. Note the direction, which is the opposite of
    // every other rule here: `prefix in layer_name` — the IGNORE ENTRY must
    // contain the PREFIX. ModelOpt lists experts one index at a time while a
    // FusedMoE layer is ONE module spanning all of them, so naming any single
    // expert child leaves the whole container unquantized.
    if (prefix.find("experts") != std::string::npos) {
      for (const std::string& e : exclude_modules_) {
        // Upstream's `filter(lambda l: "experts" in l, ...)`, mirrored though
        // provably redundant: the guard above says `prefix` contains "experts",
        // and the next line only returns for an `e` that contains `prefix`, so
        // any `e` that could return already contains "experts". Kept so the two
        // implementations read alike; no input can make it change the answer.
        if (e.find("experts") == std::string::npos) continue;
        if (e.find(prefix) != std::string::npos) return true;
      }
      return false;
    }

    for (const std::string& e : exclude_modules_) {
      if (e == prefix) return true;
    }
    return false;
  }

  // modelopt.py:2412-2487 _resolve_quant_algo, all five strategies in order.
  std::optional<std::string> ResolveQuantAlgoString(const std::string& prefix,
                                                    Resolution* how) const {
    const std::vector<std::string> candidates = PrefixCandidates(prefix);

    // 1. Direct lookup (:2424-2427).
    for (const std::string& c : candidates) {
      if (const Entry* e = Find(c)) {
        *how = Resolution::kDirect;
        return e->algo;
      }
    }

    const std::string proj = detail::LastSegment(prefix);

    // 2. Packed / fused lookup: unfuse via packed_modules_mapping (:2429-2447).
    // Note the algo set spans ALL base candidates here — unlike strategy 5,
    // which rebuilds it per candidate. Mirrored as written upstream.
    const auto fused = packed_modules_mapping_.find(proj);
    if (!packed_modules_mapping_.empty() &&
        fused != packed_modules_mapping_.end()) {
      std::set<std::string> algos;
      const std::string base = detail::ParentSegment(prefix);
      for (const std::string& bc : PrefixCandidates(base)) {
        for (const std::string& shard : fused->second) {
          if (const Entry* e = Find(bc + "." + shard)) algos.insert(e->algo);
        }
      }
      if (algos.size() == 1) {
        *how = Resolution::kPacked;
        return *algos.begin();
      }
      if (algos.size() > 1) throw MixedShardError(prefix, algos);
    }

    // 3. Prefix lookup, for a parent module such as a routed-expert container
    // (:2449-2453). Returns the FIRST child in map order.
    for (const std::string& c : candidates) {
      const std::string child_prefix = c + ".";
      for (const Entry& e : entries_) {
        if (detail::StartsWith(e.name, child_prefix)) {
          *how = Resolution::kPrefix;
          return e.algo;
        }
      }
    }

    // 4. The FusedMoE ".experts" special case (:2455-2461): the layer prefix is
    // "...moe.experts" while ModelOpt lists "...moe.up_proj" / "...moe.down_proj".
    static const std::string kExperts = ".experts";
    if (detail::EndsWith(prefix, kExperts)) {
      const std::string parent_prefix =
          prefix.substr(0, prefix.size() - kExperts.size()) + ".";
      for (const Entry& e : entries_) {
        if (detail::StartsWith(e.name, parent_prefix)) {
          *how = Resolution::kExpertsParent;
          return e.algo;
        }
      }
    }

    // 5. Fused-projection fallback for configs that list shard names where vLLM
    // uses a packed name, with no packed_modules_mapping registered (:2463-2486).
    static const std::map<std::string, std::vector<std::string>> kFusedShards = {
        {"qkv_proj", {"q_proj", "k_proj", "v_proj"}},
        {"gate_up_proj", {"gate_proj", "up_proj"}},
    };
    const auto shards = kFusedShards.find(proj);
    if (shards != kFusedShards.end()) {
      for (const std::string& c : candidates) {
        const std::string parent_prefix = detail::ParentSegment(c) + ".";
        std::set<std::string> algos;
        for (const std::string& shard : shards->second) {
          if (const Entry* e = Find(parent_prefix + shard)) algos.insert(e->algo);
        }
        if (algos.size() == 1) {
          *how = Resolution::kFusedShards;
          return *algos.begin();
        }
        if (algos.size() > 1) throw MixedShardError(prefix, algos);
      }
    }

    return std::nullopt;
  }

  static std::invalid_argument MixedShardError(const std::string& prefix,
                                               const std::set<std::string>& algos) {
    std::string joined;
    for (const std::string& a : algos) {
      if (!joined.empty()) joined += ", ";
      joined += a;
    }
    return std::invalid_argument("modelopt: mixed quant_algo within fused layer " +
                                 prefix + ": {" + joined +
                                 "}. All shards must use the same quantization.");
  }

  // Divergence 1: refuse by name rather than fall through to unquantized.
  static QuantAlgo ToQuantAlgo(const std::string& algo, const std::string& prefix) {
    if (algo == "FP8") return QuantAlgo::kFp8;
    if (algo == "NVFP4") return QuantAlgo::kNvfp4;
    if (algo == "W4A16_NVFP4") return QuantAlgo::kW4A16Nvfp4;
    if (algo == "MXFP8") return QuantAlgo::kMxfp8;

    // These ARE ModelOpt algorithms (modelopt.py:105-120) — they are simply not
    // implemented by this consumer, and the distinction is worth saying out loud
    // in the message so a reader knows whether to port a scheme or distrust the
    // checkpoint.
    const bool known = algo == "FP8_PER_CHANNEL_PER_TOKEN" || algo == "FP8_PB_WO" ||
                       algo == "MIXED_PRECISION";
    throw std::runtime_error(
        std::string("modelopt MIXED_PRECISION: ") +
        (known ? "quant_algo \"" + algo +
                     "\" is a recognized ModelOpt algorithm that is not "
                     "implemented here"
               : "unknown quant_algo \"" + algo + "\"") +
        " for module \"" + prefix +
        "\". Refusing rather than loading it unquantized: a silent dequantization "
        "is numerically correct and therefore invisible to a token gate.");
  }

  std::vector<Entry> entries_;                        // map order preserved
  std::unordered_map<std::string, std::size_t> index_;
  std::vector<std::string> exclude_modules_;
  PackedModulesMapping packed_modules_mapping_;
  std::string kv_cache_quant_algo_;
  int group_size_ = 16;
};

// ── from a TENSOR name to the MODULE name `quantized_layers` is written against
//
// `quantized_layers` and `exclude_modules` name MODULES
// (`...self_attn.q_proj`); a checkpoint ships OPERANDS
// (`...self_attn.q_proj.weight_scale`). Resolving an operand name against the
// map matches nothing and reads as "unquantized", which is the silent-dequant
// direction, so the split belongs to the resolver rather than to each caller.
//
// A SECOND suffix list, deliberately, and not a widening of the
// compressed-tensors one in
// `compressed_tensors/compressed_tensors_config.h`. The two formats spell the
// same operands differently: ModelOpt writes `weight` + `weight_scale` +
// `weight_scale_2` where compressed-tensors writes `weight_packed` +
// `weight_scale` + `weight_global_scale`, and ModelOpt's `input_scale` has no
// compressed-tensors counterpart on the checkpoints that resolver was verified
// against. That resolver's list carries a "verified complete against
// unsloth/Qwen3.8-27B-NVFP4" claim; editing it to carry names that artifact
// does not ship would make the claim false without measuring anything. One
// list per format is the same argument the tree already recorded for keeping
// one resolver per format.
//
// LONGEST SUFFIX FIRST: `<proj>.weight_global_scale` must not be split on
// `.weight`, which would leave `<proj>.weight_global` and resolve nothing.
//
// Verified complete against both ModelOpt `MIXED_PRECISION` checkpoints this
// tree pins: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` (2001 index
// names: 937 `weight`, 401 `weight_scale`, 208 `input_scale`, 193
// `weight_scale_2`, 166 `bias`, 48 `A_log`, 48 `dt_bias`) and
// `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160` (2194 names: 937 `weight`, 401
// `weight_scale`, 401 `input_scale`, 193 `weight_scale_2`, 166 `bias`, 48
// `A_log`, 48 `dt_bias`). The compressed-tensors spellings and the two
// KV-cache scales are carried too, so a name this list has never seen is a
// genuinely new family rather than a format it already knows.
inline const std::vector<std::string>& OperandSuffixes() {
  static const std::vector<std::string>* s = new std::vector<std::string>{
      ".weight_global_scale", ".input_global_scale", ".weight_packed",
      ".weight_scale_2",      ".weight_scale",      ".input_scale",
      ".weight",              ".bias",              ".A_log",
      ".dt_bias",             ".k_scale",           ".v_scale"};
  return *s;
}

// True when `name` ends in a known operand suffix; `module` and `suffix` are
// then its two halves (`suffix` without the leading dot). False means this
// resolver has never seen the family, and the caller must NOT read that as
// "unquantized". `Refusal` is the one caller, and it REFUSES such a name rather
// than skipping it, because a name that belongs to no module is cross-checked
// in neither direction — which is the silent state this whole file exists to
// close.
inline bool SplitOperand(const std::string& name, std::string* module,
                         std::string* suffix) {
  for (const std::string& s : OperandSuffixes()) {
    if (name.size() > s.size() &&
        name.compare(name.size() - s.size(), s.size(), s) == 0) {
      *module = name.substr(0, name.size() - s.size());
      *suffix = s.substr(1);
      return true;
    }
  }
  return false;
}

// The two operands that belong to the KV cache rather than to any Linear.
// `kv_cache_quant_algo` is a SIBLING of `quantized_layers`
// (modelopt.py:294, :306-314) and no path in THIS loader reads a checkpoint
// `k_scale`/`v_scale` tensor — `KV-FP8` (#1593) landed the CUDA fp8 KV store
// itself, and feeding it from a checkpoint's scales is a different job; see
// the refusal note below for why that is named as owed rather than refused
// here. `ModuleOperands::Add` is the caller: a KV scale is
// RECORDED against its module and then deliberately kept out of
// `AnyQuantOperand`, so the decision is one branch a test can mutate rather
// than a skip that changes no verdict.
inline bool IsKvCacheScaleSuffix(const std::string& suffix) {
  return suffix == "k_scale" || suffix == "v_scale";
}

// Which operands one module actually SHIPS, by name. Name-only on purpose: the
// loader's own routing probes are name-only too
// (`IsNvfp4Projection` = `has(<proj>.weight_packed) || has(<proj>.weight_scale_2)`),
// so a name-only cross-check compares like with like. A dtype check would
// answer a different question than the one that decides the arm.
struct ModuleOperands {
  bool weight = false;
  bool weight_packed = false;
  bool weight_scale = false;
  bool weight_scale_2 = false;
  bool weight_global_scale = false;
  bool input_scale = false;
  bool input_global_scale = false;
  // Recorded, and deliberately NOT part of `AnyQuantOperand` — see below.
  bool k_scale = false;
  bool v_scale = false;

  void Add(const std::string& suffix) {
    if (suffix == "weight") weight = true;
    else if (suffix == "weight_packed") weight_packed = true;
    else if (suffix == "weight_scale") weight_scale = true;
    else if (suffix == "weight_scale_2") weight_scale_2 = true;
    else if (suffix == "weight_global_scale") weight_global_scale = true;
    else if (suffix == "input_scale") input_scale = true;
    else if (suffix == "input_global_scale") input_global_scale = true;
    else if (IsKvCacheScaleSuffix(suffix)) {
      (suffix == "k_scale" ? k_scale : v_scale) = true;
    }
  }

  // A Linear ships one of these. A module carrying only `A_log`/`dt_bias`/
  // `bias` is not a Linear and is not what `quantized_layers` names, so it is
  // not cross-checked — see the note on `Refusal` for why that matters.
  bool WeightBearing() const { return weight || weight_packed; }

  // An operand whose presence means the WEIGHTS of this module are quantized.
  //
  // `k_scale` and `v_scale` are recorded above and are absent from this
  // disjunction ON PURPOSE, because they quantize the KV CACHE and not the
  // module's weights: `kv_cache_quant_algo` is a SIBLING of `quantized_layers`
  // (modelopt.py:294, :306-314). A bf16 attention tower that ships a KV scale
  // is therefore a checkpoint that `quantized_layers` correctly does not name,
  // and refusing it for "shipping a quantized spelling" would refuse
  // `nvidia/Qwen3.6-27B-NVFP4`'s successor the moment one ships the scales its
  // `kv_cache_scheme` already declares. This is the executable half of the
  // "THE KV CACHE IS NOT REFUSED HERE" decision argued on `Refusal`; adding
  // either flag to this disjunction turns that case red.
  bool AnyQuantOperand() const {
    return weight_packed || weight_scale || weight_scale_2 ||
           weight_global_scale || input_scale || input_global_scale;
  }

  // The ModelOpt NVFP4 spelling this tree's `LoadNvfp4AnyNaming` reads.
  bool ModeloptNvfp4() const { return weight && weight_scale && weight_scale_2; }
  // The compressed-tensors NVFP4 spelling the same function also reads.
  bool CtNvfp4() const {
    return weight_packed && weight_scale && weight_global_scale;
  }
  // The per-tensor static FP8 spelling this tree's `LoadFp8Raw` reads: a
  // weight, ONE weight scale and ONE static input scale to fold into `alpha`.
  bool StaticFp8() const {
    return weight && weight_scale && input_scale && !weight_scale_2 &&
           !weight_packed;
  }

  std::string Spelling() const {
    std::string out;
    const auto add = [&out](const char* s) {
      if (!out.empty()) out += " + ";
      out += s;
    };
    if (weight) add("weight");
    if (weight_packed) add("weight_packed");
    if (weight_scale) add("weight_scale");
    if (weight_scale_2) add("weight_scale_2");
    if (weight_global_scale) add("weight_global_scale");
    if (input_scale) add("input_scale");
    if (input_global_scale) add("input_global_scale");
    if (k_scale) add("k_scale");
    if (v_scale) add("v_scale");
    return out.empty() ? std::string("<no operand>") : out;
  }
};

namespace detail {

inline std::string JoinModules(const std::set<std::string>& modules,
                               std::size_t cap) {
  std::string out;
  std::size_t shown = 0;
  for (const std::string& m : modules) {
    if (shown == cap) break;
    if (shown != 0) out += ", ";
    out += m;
    ++shown;
  }
  if (modules.size() > shown)
    out += " (and " + std::to_string(modules.size() - shown) + " more)";
  return out;
}

}  // namespace detail

// The refusal a loader owes a ModelOpt `MIXED_PRECISION` checkpoint whose
// DECLARED algorithm this build cannot execute, or whose declared algorithm and
// SHIPPED operand names disagree. "" means every weight-bearing module's
// declaration and spelling agree and every declared algorithm has a loader.
//
// WHY THIS IS THE CHECK, AND NOT A ROUTER. This tree routes each projection by
// probing which tensor NAMES are present
// (`qwen3_5_dense_weights.cpp` `IsNvfp4Projection`, then an `F8_E4M3` dtype
// probe, then bf16). That probe cannot be wrong about the BYTES, and it can be
// wrong about the CHECKPOINT: a module the producer declared FP8 that ships an
// NVFP4 spelling routes to the NVFP4 arm, and a module declared unquantized
// that ships scales routes to a quantized arm. Both produce finite, plausible
// numbers and matching tokens while moving the wrong bytes, which is the one
// defect class `AGENTS.md` §"Inherit vLLM defaults" says a token gate cannot
// see. Refusing on disagreement makes the probe's answer CHECKED against the
// producer's declaration without changing which arm a checkpoint takes, so no
// artifact that loads today loads differently.
//
// ONLY WEIGHT-BEARING MODULES ARE CROSS-CHECKED. `quantized_layers` names
// Linears. Splitting operand names also produces container names that own no
// Linear weight — `...layers.<i>.linear_attn`, which ships `A_log` and
// `dt_bias` — and upstream's strategy-3 prefix scan (modelopt.py:2449-2453)
// resolves such a container to its first quantized CHILD's algorithm. That is
// correct upstream, where the container IS the quantized layer, and it is not a
// statement about the two scalars the container ships. Cross-checking it would
// refuse both artifacts this was verified against for a disagreement neither
// has.
//
// THE KV CACHE IS NOT REFUSED HERE, and that is a decision rather than an
// omission. `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160` — a gate model this tree
// loads and measures today — declares `kv_cache_scheme` 8-bit float static and
// ships ZERO `k_scale`/`v_scale` tensors, and
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` declares `kv_cache_quant_algo: "FP8"`
// in `hf_quant_config.json`. Since `KV-FP8` W3 (#1593) that file DOES have a
// production reader — `vllm::ReadQuantConfigJson` (`src/vllm/config/cache.cpp`
// :207), called from `LoadedEngine::FromModelDir` (`model_loader.cpp:1988`) —
// but only as the LEGACY FALLBACK behind `config.json`'s inline
// `quantization_config` (`config.py:751-761`), and only to resolve the KV cache
// dtype. This checkpoint ships that inline document, so its legacy file is
// never opened and the declaration reaches nothing.
// A refusal here would refuse a checkpoint that loads today. The FP8 KV
// arm is owned by `KV-FP8` (issue #1593) and is listed under `## Owed` in
// `.agents/specs/qwen38-27b-quant-arms.md`. `ModuleOperands` RECORDS a KV scale
// and leaves it out of `AnyQuantOperand`, which is where that decision is
// executable; this function no longer skips the suffix, because a skip that
// changes no verdict is a claim no test can read.
//
// A TENSOR NAME WHOSE FAMILY THE SPLITTER HAS NEVER SEEN IS REFUSED, not
// skipped. `SplitOperand` documents that `false` means "this resolver has never
// seen the family, and the caller must NOT read that as unquantized", and a
// `continue` here read it as exactly that: the name belonged to no module, so
// NOTHING cross-checked it in either direction and the checkpoint loaded
// silently. `.weight_scale_inv` is the case that makes this concrete — it is
// the block-wise FP8 spelling `IsFp8BlockProjection` reads two rungs above the
// call site in `qwen3_5_dense_weights.cpp`, so a ModelOpt checkpoint could ship
// it, take the block-FP8 arm, and never be cross-checked at all. Both artifacts
// this resolver is verified against classify EVERY shipped name (2001 of 2001
// and 2194 of 2194), so this refusal cannot fire on either, and the suffix list
// above carries the compressed-tensors spellings for the same reason.
inline std::string Refusal(const MixedPrecisionConfig& c,
                           const std::vector<std::string>& tensor_names) {
  std::map<std::string, ModuleOperands> shipped;
  std::set<std::string> unclassified;
  for (const std::string& name : tensor_names) {
    std::string module;
    std::string suffix;
    if (!SplitOperand(name, &module, &suffix)) {
      unclassified.insert(name);
      continue;
    }
    shipped[module].Add(suffix);
  }

  // Grouped by (declared algorithm, reason): on a real artifact one missing
  // piece is shared by hundreds of modules, and 208 sentences about it is not a
  // better message than one.
  std::map<std::pair<std::string, std::string>, std::set<std::string>> refused;
  for (const auto& entry : shipped) {
    const std::string& module = entry.first;
    const ModuleOperands& ops = entry.second;
    if (!ops.WeightBearing()) continue;

    // `Resolve` itself throws for an algorithm string this consumer does not
    // implement (divergence 1 above), which names the module and the algorithm
    // and is exactly the sentence this function would otherwise build.
    const ModuleQuant q = c.Resolve(module);
    const std::string algo = QuantAlgoName(q.algo);
    std::string reason;
    switch (q.algo) {
      case QuantAlgo::kUnquantized:
        if (ops.AnyQuantOperand()) {
          reason =
              "the config does not quantize them, and they ship a QUANTIZED "
              "spelling (" + ops.Spelling() +
              "). Loading them quantized overrules the producer's declaration; "
              "loading them bf16 ignores tensors the file carries";
        }
        break;
      case QuantAlgo::kFp8:
        if (!ops.StaticFp8()) {
          reason =
              "this build reads a per-tensor STATIC FP8 spelling (weight + "
              "weight_scale + input_scale), and they ship " + ops.Spelling();
        }
        break;
      case QuantAlgo::kNvfp4:
      case QuantAlgo::kW4A16Nvfp4:
        if (!ops.ModeloptNvfp4() && !ops.CtNvfp4()) {
          reason =
              "this build reads an NVFP4 spelling (weight + weight_scale + "
              "weight_scale_2, or weight_packed + weight_scale + "
              "weight_global_scale), and they ship " + ops.Spelling();
        }
        break;
      case QuantAlgo::kMxfp8:
        reason =
            "this build has no MXFP8 loader. Refusing rather than reading them "
            "as FP8 or as bf16: both are numerically plausible and therefore "
            "invisible to a token gate";
        break;
    }
    if (!reason.empty()) refused[{algo, reason}].insert(module);
  }
  if (refused.empty() && unclassified.empty()) return std::string();

  std::string out =
      "modelopt MIXED_PRECISION quantization_config declares " +
      std::to_string(c.num_quantized_layers()) +
      " quantized layer(s), and the shipped tensors do not agree with it.";
  if (!unclassified.empty()) {
    out += "\n  " + std::to_string(unclassified.size()) +
           " shipped tensor name(s) end in an operand family this resolver has "
           "never seen, so no module owns them and NOTHING cross-checked them "
           "in either direction.";
    out += "\n  unclassified: " + detail::JoinModules(unclassified, 8);
  }
  for (const auto& kv : refused) {
    out += "\n  " + std::to_string(kv.second.size()) +
           " module(s) resolve to quant_algo \"" + kv.first.first + "\": " +
           kv.first.second + ".";
    out += "\n  refused: " + detail::JoinModules(kv.second, 8);
  }
  return out;
}

// The whole-config convenience a loader calls with the document's
// `quantization_config` and its shipped tensor names. A config that is not a
// ModelOpt `MIXED_PRECISION` one answers "" and reads nothing, so a caller that
// adds this leaves every other checkpoint byte-identical.
//
// The `quant_method` precondition is `IsMixedPrecision`'s and not `Parse`'s,
// for the reason argued on both: a loader holding a `config.json` has to tell a
// ModelOpt `quantization_config` apart from a compressed-tensors one sitting in
// the same field, and `quant_method` is how upstream does it
// (modelopt.py:2333-2339).
template <class Json>
inline std::string RefusalForQuantizationConfig(
    const Json& quant, const std::vector<std::string>& tensor_names) {
  if (!MixedPrecisionConfig::IsMixedPrecision(quant)) return std::string();
  return Refusal(MixedPrecisionConfig::Parse(quant), tensor_names);
}

}  // namespace modelopt
}  // namespace layers
}  // namespace vllm
