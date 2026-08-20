// compressed-tensors `config_groups`: resolving a quantization SCHEME PER
// MODULE, from the config rather than from a tensor-dtype probe.
//
// UPSTREAM (ported FROM, ground-every-impl rule) @ the pinned parity oracle
// `5559679229bc961848b121ccdeaa8fa5d79bec98`:
//   vllm/model_executor/layers/quantization/compressed_tensors/utils.py:50-102
//     should_ignore_layer — the `ignore` list is consulted FIRST and wins
//     outright.
//   .../compressed_tensors/utils.py:105-110   check_equal_or_regex_match
//   .../compressed_tensors/utils.py:113-152   find_matched_target
//   .../compressed_tensors/utils.py:155-172   _find_first_match — FIRST match in
//     iteration order, which is why `targets` order is load-bearing below.
//   .../compressed_tensors/utils.py:175-193   _is_equal_or_regex_match — a
//     target is a Python `re.match` when it starts with `re:`, and an EXACT
//     string comparison otherwise. Not a substring test.
//   .../compressed_tensors/compressed_tensors.py:230-265   from_config
//   .../compressed_tensors/compressed_tensors.py:300-369
//     _quantization_scheme_map_from_config — flattens config_groups into
//     {target -> {weights, input_activations, format}} in DICT INSERTION ORDER.
//   .../compressed_tensors/compressed_tensors.py:404-421  _is_nvfp4_format
//   .../compressed_tensors/compressed_tensors.py:526-560  _is_fp8_w8a8
//   .../compressed_tensors/compressed_tensors.py:722-743
//     _get_scheme_from_parts — the NVFP4 branch, including the `input_quant is
//     None` => W4A16 split.
//   .../compressed_tensors/compressed_tensors.py:810-836  the W8A8 FP8 branch
//   .../compressed_tensors/compressed_tensors.py:877-959  get_scheme /
//     get_scheme_dict — ignore first, then first-matching target.
//   .../compressed_tensors/compressed_tensors.py:1034-1073
//     validate_kv_cache_scheme
//   .../compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:53-56,127-139,152-165
//     the weight STRATEGY (`tensor` vs `channel`) picks the scale parameter
//     type and the post-load path BEFORE a byte is read, and `input_scale` is
//     registered ONLY when the input scheme is static.
//
// WHY THIS EXISTS, AND WHY IT IS NOT `modelopt_mixed_precision.h`.
//
// `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` is a
// compressed-tensors checkpoint whose `format` is `"mixed-precision"`: layers
// 0-55's MLP is NVFP4 W4A4 and layers 56-63's MLP is FP8 W8A8, under the SAME
// module names. Which arm a projection takes is decided by a REGEX OVER LAYER
// INDICES, and no per-projection dtype probe can recover that: it is only in
// the config. Everything this tree reads today is a probe.
//
// `modelopt_mixed_precision.h` resolves a DIFFERENT upstream format that shares
// the English word "mixed". That one is ModelOpt's `MIXED_PRECISION`
// (`quant_method: "modelopt"`, a `quantized_layers` map of module prefix ->
// algorithm string, resolved by five PREFIX strategies —
// `modelopt.py:2412-2505`). This one is compressed-tensors
// (`quant_method: "compressed-tensors"`, `config_groups` with regex `targets`
// and a separate `ignore` list, resolved by ORDERED FIRST REGEX MATCH). The two
// share no key, no matching rule and no upstream module: vLLM keeps them in
// `modelopt.py` and `compressed_tensors/` respectively, and reading either
// config with the other's resolver silently mis-resolves every module.
//
// `.agents/specs/qwen38-27b-quant-arms.md` §"Stop conditions" says to adopt or
// extend `modelopt_mixed_precision.h` or record ONE EXACT TRACKED EXCEPTION, and
// says that finding it unfit is a legitimate outcome. This is that exception,
// and its argument is the paragraph above: the tree does not end up with two
// resolvers for one format, it ends up with one resolver per format, which is
// upstream's own structure and the structure `.agents/porting.md` requires us to
// mirror. `QuantizationConfigOf` below is shared with `fp8_block_quant.cpp`
// rather than copied, because THAT lookup really is one function.
//
// SCOPE. Reading the config and answering "which scheme does this module take".
// No kernels, no weights, no forward. Header-only and under `src/` for the
// reason `modelopt_mixed_precision.h` records: it is string and JSON logic with
// no device dependency, and `include/vllm/` is a user-facing surface with its
// own public-document obligation that an internal resolver does not earn.
//
// Row `QUANT-QWEN38-27B-NVFP4-ARM`, issue
// https://github.com/mudler/vllm.cpp/issues/821, spec
// `.agents/specs/qwen38-27b-quant-arms.md`.
#pragma once

#include <algorithm>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace layers {
namespace compressed_tensors {

// `quantization_config`, top level first and then the `text_config` nesting.
// ONE implementation: `fp8_block_quant.cpp` calls this rather than carrying its
// own copy, because two copies of a config lookup are two answers to "where
// does the quantization config live" and the wrapper shape
// (`Qwen3_5ForConditionalGeneration`) is exactly where they would differ.
inline const nlohmann::json* QuantizationConfigOf(const nlohmann::json& raw) {
  if (!raw.is_object()) return nullptr;
  const auto top = raw.find("quantization_config");
  if (top != raw.end() && top->is_object()) return &*top;
  const auto text = raw.find("text_config");
  if (text == raw.end() || !text->is_object()) return nullptr;
  const auto nested = text->find("quantization_config");
  if (nested != text->end() && nested->is_object()) return &*nested;
  return nullptr;
}

// One `weights` / `input_activations` block of a config group, as
// `QuantizationArgs` carries it upstream. `present` distinguishes "the config
// declared no input activations" — which is what selects W4A16 at
// `compressed_tensors.py:735-736` — from "declared, and every field defaulted".
struct QuantArgs {
  bool present = false;
  int num_bits = 0;
  std::string type;      // "float" | "int"
  std::string strategy;  // "tensor" | "channel" | "token" | "group" |
                         // "tensor_group" | "block"
  // `dynamic` is TRI-STATE in the released configs and upstream reads it as
  // such: `false` (static), `true` (recompute per forward), or the STRING
  // "local" that `nvfp4-pack-quantized` uses for its per-16 activation scale.
  // A parser that folds the string into `true` cannot tell a per-token FP8
  // activation from an NVFP4 one, and those take different kernels.
  bool dynamic = false;
  std::string dynamic_str;
  int group_size = 0;
  bool symmetric = true;
  std::string actorder;
};

// `kv_cache_scheme` — a SIBLING of the config groups, not one of them
// (`compressed_tensors.py:262`, validated at `:1034-1073`). The 27B artifact
// declares one and ships `<layer>.self_attn.{k,v}_scale` for it.
struct KvCacheScheme {
  bool present = false;
  int num_bits = 0;
  std::string type;
  std::string strategy;
  bool dynamic = false;
  bool symmetric = true;
};

// Which scheme a module resolved to, and — when this build cannot execute it —
// WHY, by name. `kUnsupported` is never silently downgraded to `kUnquantized`:
// dequantizing a quantized projection is numerically plausible, matches tokens,
// and moves twice the bytes, which is the one defect class a token gate cannot
// see (`AGENTS.md` §"Inherit vLLM defaults").
enum class SchemeKind {
  kUnquantized,    // on the `ignore` list, or no target matched
  kFp8W8A8Tensor,  // per-tensor weight scale + static per-tensor input scale
  kNvfp4W4A4,      // tensor_group g16 weights AND activations
  kNvfp4W4A16,     // tensor_group g16 weights, no input activations
  kUnsupported,    // a real scheme this build has no loader or kernel for
};

struct ModuleScheme {
  bool ignored = false;  // matched the `ignore` list
  bool matched = false;  // matched a `targets` entry
  std::string group;     // "group_0" / "group_1" — the config_groups key
  std::string target;    // the entry that matched, verbatim
  std::string format;    // the group's `format`, else the global one
  QuantArgs weights;
  QuantArgs input_activations;
  SchemeKind kind = SchemeKind::kUnquantized;
  // Non-empty exactly when `kind == kUnsupported`: the missing piece, named.
  std::string unsupported_reason;

  bool Quantized() const { return kind != SchemeKind::kUnquantized; }
};

namespace detail {

inline std::string StringField(const nlohmann::json& o, const char* key) {
  const auto it = o.find(key);
  if (it == o.end() || !it->is_string()) return std::string();
  return it->get<std::string>();
}

inline int IntField(const nlohmann::json& o, const char* key, int fallback) {
  const auto it = o.find(key);
  if (it == o.end() || !it->is_number_integer()) return fallback;
  return it->get<int>();
}

inline QuantArgs ParseQuantArgs(const nlohmann::json& o) {
  QuantArgs a;
  if (!o.is_object()) return a;
  a.present = true;
  a.num_bits = IntField(o, "num_bits", 0);
  a.type = StringField(o, "type");
  a.strategy = StringField(o, "strategy");
  a.group_size = IntField(o, "group_size", 0);
  a.actorder = StringField(o, "actorder");
  const auto sym = o.find("symmetric");
  a.symmetric = sym == o.end() || !sym->is_boolean() || sym->get<bool>();
  const auto dyn = o.find("dynamic");
  if (dyn != o.end()) {
    if (dyn->is_boolean()) a.dynamic = dyn->get<bool>();
    if (dyn->is_string()) {
      a.dynamic_str = dyn->get<std::string>();
      a.dynamic = true;
    }
  }
  return a;
}

// Python `re.match`: anchored at position 0, NOT required to reach the end.
// `std::regex_constants::match_continuous` is exactly that, and using
// `regex_match` here instead would silently drop every unanchored target.
inline bool RegexMatchesAtStart(const std::string& value, const std::regex& re) {
  std::smatch m;
  return std::regex_search(value, m, re, std::regex_constants::match_continuous);
}

}  // namespace detail

// The compressed-tensors quantization config a checkpoint declares, parsed once.
//
// `declared()` false means this checkpoint is not compressed-tensors (or
// declares no `config_groups`), and every `Resolve` answers `kUnquantized` —
// which leaves every existing routing decision in this tree byte-identical.
class Config {
 public:
  Config() = default;

  static Config FromQuantizationConfig(const nlohmann::json& quant) {
    Config c;
    if (!quant.is_object()) return c;
    c.quant_method_ = detail::StringField(quant, "quant_method");
    c.format_ = detail::StringField(quant, "format");
    if (c.quant_method_ != "compressed-tensors") return c;

    const auto groups = quant.find("config_groups");
    if (groups != quant.end() && groups->is_object()) {
      // INSERTION ORDER, and it decides the answer. `group_0` targets
      // `re:.*layers\.(56|...|63)\.mlp\.(gate|up|down)_proj$` and `group_1`
      // targets `re:.*mlp\.(gate|up|down)_proj$`; both match layer 60's
      // `gate_proj`, and upstream takes the FIRST
      // (`utils.py:169-172`). nlohmann's default `json` sorts object keys,
      // and "group_0" < "group_1" lexicographically, so the order this loop
      // sees is the order the file declares for THIS config. A checkpoint
      // whose group keys do not sort into declaration order would need
      // `ordered_json`, which is why the divergence is stated here rather
      // than assumed away.
      for (auto it = groups->begin(); it != groups->end(); ++it) {
        const nlohmann::json& g = it.value();
        if (!g.is_object()) continue;
        const auto targets = g.find("targets");
        if (targets == g.end() || !targets->is_array()) continue;
        // `from_config` (compressed_tensors.py:230-246) DROPS a group whose
        // targets are exactly one `*Attention` entry, because attention
        // quantization on its own is coupled to the KV-cache scheme rather
        // than to a linear method. Mirrored so a checkpoint carrying such a
        // group resolves here the way it resolves upstream, instead of being
        // refused for a scheme upstream never asked us to execute. This
        // artifact declares no such group; the branch is a mirror, not a
        // response to it.
        if (targets->size() == 1 && targets->front().is_string()) {
          const std::string only = targets->front().get<std::string>();
          const std::string kAttn = "Attention";
          if (only.size() >= kAttn.size() &&
              only.compare(only.size() - kAttn.size(), kAttn.size(), kAttn) ==
                  0) {
            continue;
          }
        }
        Group parsed;
        parsed.name = it.key();
        parsed.format = detail::StringField(g, "format");
        const auto w = g.find("weights");
        if (w != g.end()) parsed.weights = detail::ParseQuantArgs(*w);
        const auto ia = g.find("input_activations");
        if (ia != g.end() && ia->is_object())
          parsed.input_activations = detail::ParseQuantArgs(*ia);
        for (const nlohmann::json& t : *targets) {
          if (!t.is_string()) continue;
          parsed.targets.push_back(t.get<std::string>());
        }
        c.groups_.push_back(std::move(parsed));
      }
    }

    const auto ignore = quant.find("ignore");
    if (ignore != quant.end() && ignore->is_array()) {
      for (const nlohmann::json& e : *ignore) {
        if (!e.is_string()) continue;
        const std::string s = e.get<std::string>();
        // A literal goes in a set: 302 of this checkpoint's 303 entries are
        // literals, `any()` makes their order irrelevant
        // (`utils.py:105-110`), and compiling 302 regexes to compare strings
        // would be a per-load cost for nothing.
        if (s.rfind("re:", 0) == 0) {
          c.ignore_regexes_.emplace_back(s.substr(3),
                                         std::regex::ECMAScript | std::regex::optimize);
        } else {
          c.ignore_literals_.insert(s);
        }
        c.ignore_count_ += 1;
      }
    }

    const auto kv = quant.find("kv_cache_scheme");
    if (kv != quant.end() && kv->is_object()) {
      c.kv_.present = true;
      c.kv_.num_bits = detail::IntField(*kv, "num_bits", 0);
      c.kv_.type = detail::StringField(*kv, "type");
      c.kv_.strategy = detail::StringField(*kv, "strategy");
      const auto dyn = kv->find("dynamic");
      c.kv_.dynamic = dyn != kv->end() && dyn->is_boolean() && dyn->get<bool>();
      const auto sym = kv->find("symmetric");
      c.kv_.symmetric =
          sym == kv->end() || !sym->is_boolean() || sym->get<bool>();
    }

    // Compile the targets ONCE, in declaration order.
    for (Group& g : c.groups_) {
      for (const std::string& t : g.targets) {
        if (t.rfind("re:", 0) == 0) {
          g.compiled.emplace_back(t.substr(3),
                                  std::regex::ECMAScript | std::regex::optimize);
          g.is_regex.push_back(true);
        } else {
          g.compiled.emplace_back();
          g.is_regex.push_back(false);
        }
      }
    }
    c.declared_ = !c.groups_.empty();
    return c;
  }

  // The whole top-level `config.json` document; the group lives at the top
  // level or under `text_config`, and a multimodal wrapper uses the latter.
  static Config FromHfConfigRaw(const nlohmann::json& raw) {
    const nlohmann::json* quant = QuantizationConfigOf(raw);
    if (quant == nullptr) return Config();
    return FromQuantizationConfig(*quant);
  }

  bool declared() const { return declared_; }
  const std::string& format() const { return format_; }
  const std::string& quant_method() const { return quant_method_; }
  const KvCacheScheme& kv_cache_scheme() const { return kv_; }
  size_t ignore_count() const { return ignore_count_; }
  size_t group_count() const { return groups_.size(); }

  // `should_ignore_layer` (utils.py:50-102) with an empty `fused_mapping`: this
  // loader matches on CHECKPOINT module names, which are already unfused, so
  // the fused arm upstream takes for `qkv_proj`/`gate_up_proj` has nothing to
  // do here. Naming that rather than leaving the omission to be found.
  bool Ignored(const std::string& module) const {
    if (ignore_literals_.count(module) != 0) return true;
    for (const std::regex& re : ignore_regexes_)
      if (detail::RegexMatchesAtStart(module, re)) return true;
    return false;
  }

  ModuleScheme Resolve(const std::string& module) const {
    ModuleScheme s;
    if (!declared_) return s;
    if (Ignored(module)) {
      s.ignored = true;
      return s;
    }
    for (const Group& g : groups_) {
      for (size_t i = 0; i < g.targets.size(); ++i) {
        const bool hit = g.is_regex[i]
                             ? detail::RegexMatchesAtStart(module, g.compiled[i])
                             : g.targets[i] == module;
        if (!hit) continue;
        s.matched = true;
        s.group = g.name;
        s.target = g.targets[i];
        s.format = g.format.empty() ? format_ : g.format;
        s.weights = g.weights;
        s.input_activations = g.input_activations;
        Classify(s);
        return s;
      }
    }
    return s;
  }

 private:
  struct Group {
    std::string name;
    std::string format;
    std::vector<std::string> targets;
    std::vector<std::regex> compiled;
    std::vector<bool> is_regex;
    QuantArgs weights;
    QuantArgs input_activations;
  };

  // `_is_nvfp4_format` (compressed_tensors.py:404-421), verbatim.
  static bool IsNvfp4(const QuantArgs& a) {
    return a.present && a.strategy == "tensor_group" && a.type == "float" &&
           a.num_bits == 4 && a.group_size == 16 && a.symmetric;
  }

  // `_get_scheme_from_parts` (compressed_tensors.py:722-836), narrowed to the
  // arms this tree HAS, with everything else refused BY NAME rather than
  // falling through to an unquantized read.
  static void Classify(ModuleScheme& s) {
    const QuantArgs& w = s.weights;
    const QuantArgs& a = s.input_activations;
    if (IsNvfp4(w)) {
      if (!a.present) {
        s.kind = SchemeKind::kNvfp4W4A16;
        return;
      }
      if (!IsNvfp4(a)) {
        s.kind = SchemeKind::kUnsupported;
        s.unsupported_reason =
            "NVFP4 weights whose input activations are not NVFP4, which "
            "upstream refuses outright (compressed_tensors.py:738-742)";
        return;
      }
      s.kind = SchemeKind::kNvfp4W4A4;
      return;
    }
    if (w.type == "float" && w.num_bits == 8) {
      // The ONE FP8 arm this tree loads: a per-TENSOR weight scale beside a
      // static per-TENSOR input scale, folded once at load into
      // `Fp8Weight::alpha` (`qwen3_5_weights.cpp` LoadFp8Raw). Upstream
      // registers `input_scale` only when the input scheme is static
      // (`compressed_tensors_w8a8_fp8.py:137-139`), and picks the scale
      // parameter type from the weight STRATEGY (`:127-133`).
      if (w.strategy == "tensor" && a.present && !a.dynamic &&
          a.strategy == "tensor") {
        s.kind = SchemeKind::kFp8W8A8Tensor;
        return;
      }
      s.kind = SchemeKind::kUnsupported;
      std::string missing;
      if (w.strategy == "channel") {
        missing =
            "a PER-OUTPUT-CHANNEL weight scale (weights.strategy \"channel\": "
            "<proj>.weight_scale ships [out, 1], not the one element this "
            "build's per-tensor Fp8Weight holds)";
      } else if (w.strategy == "block") {
        missing =
            "a BLOCK weight scale declared through compressed-tensors "
            "(weights.strategy \"block\"); this build reads a block scale only "
            "from an fp8 quant_method checkpoint's weight_scale_inv";
      } else if (!w.strategy.empty() && w.strategy != "tensor") {
        missing = "weights.strategy \"" + w.strategy + "\"";
      }
      if (a.present && a.dynamic) {
        const std::string act =
            "DYNAMIC activation quantization (input_activations.dynamic " +
            (s.input_activations.dynamic_str.empty()
                 ? std::string("true")
                 : "\"" + s.input_activations.dynamic_str + "\"") +
            ", strategy \"" + a.strategy +
            "\"): the activation scale is computed per forward and the "
            "checkpoint therefore ships NO <proj>.input_scale, while this "
            "build folds a STATIC input_scale into the GEMM at load";
        missing = missing.empty() ? act : missing + " AND " + act;
      }
      if (missing.empty()) {
        missing =
            "an FP8 scheme with no static per-tensor input scale to fold";
      }
      s.unsupported_reason = missing;
      return;
    }
    if (w.present) {
      s.kind = SchemeKind::kUnsupported;
      s.unsupported_reason = "weights " + std::to_string(w.num_bits) +
                             "-bit type \"" + w.type + "\" strategy \"" +
                             w.strategy + "\"";
    }
  }

  bool declared_ = false;
  std::string quant_method_;
  std::string format_;
  std::vector<Group> groups_;
  std::unordered_set<std::string> ignore_literals_;
  std::vector<std::regex> ignore_regexes_;
  size_t ignore_count_ = 0;
  KvCacheScheme kv_;
};


// ── from a TENSOR name to the MODULE name the config is written against ─────
//
// `targets` and `ignore` name MODULES (`...self_attn.q_proj`); a checkpoint
// ships OPERANDS (`...self_attn.q_proj.weight_scale`). Resolving an operand
// name against `targets` matches nothing and reads as "unquantized", which is
// the silent-dequant direction, so the split is part of the resolver rather
// than of each caller.
//
// LONGEST SUFFIX FIRST: stripping `.weight` from `<proj>.weight_global_scale`
// would leave `<proj>.weight_global` and resolve nothing. Verified complete
// against `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d...`, whose 1968 index names
// carry exactly these ten trailing components (769 `weight`, 401
// `weight_scale`, 168 each of `weight_packed` / `weight_global_scale` /
// `input_global_scale`, 166 `bias`, 48 each of `A_log` / `dt_bias`, 16 each of
// `k_scale` / `v_scale`).
inline const std::vector<std::string>& OperandSuffixes() {
  static const std::vector<std::string>* s = new std::vector<std::string>{
      ".weight_global_scale", ".input_global_scale", ".weight_packed",
      ".weight_scale",        ".weight",            ".bias",
      ".A_log",               ".dt_bias",           ".k_scale",
      ".v_scale"};
  return *s;
}

// True when `name` ends in a known operand suffix; `module` and `suffix` are
// then its two halves (`suffix` without the leading dot). False means this
// resolver has never seen the family and the caller must NOT treat it as
// unquantized.
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

// True for the two operands that belong to the SIBLING `kv_cache_scheme`
// rather than to any config group (`compressed_tensors.py:262`).
inline bool IsKvCacheScaleSuffix(const std::string& suffix) {
  return suffix == "k_scale" || suffix == "v_scale";
}

namespace detail {

inline std::string JoinModules(const std::set<std::string>& modules,
                               size_t cap) {
  std::string out;
  size_t shown = 0;
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

// The refusal a loader owes a checkpoint whose declared scheme this build
// cannot execute. Empty string means "nothing here is unrepresentable"; every
// non-compressed-tensors checkpoint answers empty, so a caller that adds this
// leaves every existing checkpoint byte-identical.
//
// WHY A REFUSAL AND NOT A FALLBACK. Loading a `channel`-strategy FP8 projection
// as bf16, or reading a per-channel scale as its first element, produces
// numbers that are plausible and tokens that match. `AGENTS.md` §"Inherit vLLM
// defaults" says a token gate cannot see a dtype that is too wide, so the only
// safe answer to a scheme with no loader is to name the missing piece and stop.
//
// The message groups the refused modules by (group, reason) rather than naming
// the first one, because on the real 27B artifact 233 modules share ONE missing
// piece and 233 sentences about it is not a better message than one.
inline std::string UnsupportedSchemeRefusal(
    const Config& c, const std::vector<std::string>& tensor_names) {
  if (!c.declared()) return std::string();
  std::map<std::pair<std::string, std::string>, std::set<std::string>> refused;
  std::map<std::pair<std::string, std::string>, std::string> formats;
  std::set<std::string> seen;
  for (const std::string& name : tensor_names) {
    std::string module;
    std::string suffix;
    if (!SplitOperand(name, &module, &suffix)) continue;
    if (IsKvCacheScaleSuffix(suffix)) continue;
    if (!seen.insert(module).second) continue;
    const ModuleScheme s = c.Resolve(module);
    if (s.kind != SchemeKind::kUnsupported) continue;
    const auto key = std::make_pair(s.group, s.unsupported_reason);
    refused[key].insert(module);
    formats[key] = s.format;
  }
  if (refused.empty()) return std::string();

  std::string out = "compressed-tensors quantization_config (quant_method \"" +
                    c.quant_method() + "\", format \"" + c.format() +
                    "\") declares a scheme this build has no loader for.";
  for (const auto& kv : refused) {
    out += "\n  " + std::to_string(kv.second.size()) +
           " module(s) resolve to config_groups \"" + kv.first.first +
           "\" (format \"" + formats[kv.first] + "\"), which needs " +
           kv.first.second + ".";
    out += "\n  refused: " + detail::JoinModules(kv.second, 8);
  }
  return out;
}

// `kv_cache_scheme` is declared and nothing in this build reads it. Silence
// about a KV-cache quantization scheme is the same token-invisible defect as a
// silent dequant: the tokens match while the cache holds the wrong precision.
inline std::string KvCacheSchemeRefusal(
    const Config& c, const std::vector<std::string>& tensor_names) {
  const KvCacheScheme& kv = c.kv_cache_scheme();
  if (!kv.present) return std::string();
  size_t k = 0;
  size_t v = 0;
  for (const std::string& name : tensor_names) {
    std::string module;
    std::string suffix;
    if (!SplitOperand(name, &module, &suffix)) continue;
    if (suffix == "k_scale") ++k;
    if (suffix == "v_scale") ++v;
  }
  return "compressed-tensors kv_cache_scheme: the checkpoint declares " +
         std::to_string(kv.num_bits) + "-bit " + kv.type + " " + kv.strategy +
         (kv.dynamic ? " dynamic" : " static") +
         " KV-cache quantization and ships " + std::to_string(k) +
         " k_scale and " + std::to_string(v) +
         " v_scale tensors for it. This build reads neither, and it has no "
         "quantized KV cache to apply them to, so honouring the scheme is the "
         "missing piece.";
}

// Both refusals, scheme first. Returns "" when the checkpoint is fully
// representable here.
inline std::string Refusal(const Config& c,
                           const std::vector<std::string>& tensor_names) {
  const std::string scheme = UnsupportedSchemeRefusal(c, tensor_names);
  if (!scheme.empty()) return scheme;
  return KvCacheSchemeRefusal(c, tensor_names);
}

// The whole-config convenience a loader calls: parse the document's
// `quantization_config` (top level or under `text_config`) and answer the
// refusal. A document that declares none answers "".
inline std::string RefusalForHfConfigRaw(
    const nlohmann::json& raw, const std::vector<std::string>& tensor_names) {
  return Refusal(Config::FromHfConfigRaw(raw), tensor_names);
}

}  // namespace compressed_tensors
}  // namespace layers
}  // namespace vllm
