// DFlash draft weight loader (SPEC-DFLASH D2, DF-DRAFT-MODEL). Ported from
// DFlashQwen3Model.load_weights + hf_to_vllm_mapper (qwen3_dflash.py:347-356,
// 657-661, 772-855 @ 555967922). All draft tensors are BF16.
//
// On-disk names: vLLM's load_weights prepends "model." to every tensor except
// lm_head (:787-788) and maps q/k/v_proj -> qkv_proj, gate/up_proj ->
// gate_up_proj (the stacked mapper, :349-355). We consume the RAW checkpoint
// names (the mapper's job) and concatenate q|k|v and gate|up ourselves. The exact
// on-disk key spelling (bare vs "model."-prefixed) is confirmed against the
// checkpoint's dumped key list at the D2 capture step (scripts/spec/
// d2_dflash_draft_ref.py); the resolver here tries the bare name first, then a
// "model."-prefixed fallback, matching both conventions.
#include "vllm/model_executor/models/qwen3_dflash.h"

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

OwnedTensor MakeOwned(vt::DType dtype, const std::vector<int64_t>& shape) {
  OwnedTensor out;
  out.dtype = dtype;
  out.rank = static_cast<int>(shape.size());
  VT_CHECK(out.rank <= vt::kMaxRank, "qwen3_dflash: rank exceeds kMaxRank");
  int64_t numel = 1;
  for (int i = 0; i < out.rank; ++i) {
    out.shape[i] = shape[static_cast<size_t>(i)];
    numel *= out.shape[i];
  }
  out.bytes.resize(static_cast<size_t>(numel) * vt::SizeOf(dtype));
  return out;
}

OwnedTensor LoadBf16Direct(const TensorResolver& get, const std::string& name) {
  const StTensor& tensor = get(name);
  VT_CHECK(tensor.dtype == "BF16", "qwen3_dflash: expected BF16 for " + name);
  OwnedTensor out = MakeOwned(vt::DType::kBF16, tensor.shape);
  VT_CHECK(tensor.nbytes == out.bytes.size(), "qwen3_dflash: byte-size mismatch for " + name);
  std::memcpy(out.bytes.data(), tensor.data, tensor.nbytes);
  return out;
}

OwnedTensor LoadBf16RawNK(const TensorResolver& get, const std::string& name) {
  OwnedTensor out = LoadBf16Direct(get, name);
  VT_CHECK(out.rank == 2, "qwen3_dflash: expected a 2-D Linear weight for " + name);
  out.nk = true;
  return out;
}

// Load `name` if the checkpoint ships it, else return an EMPTY OwnedTensor. The
// z-lab DFlash draft ships neither embed_tokens nor lm_head (confirmed against the
// on-disk key dump, 58 tensors: fc/hidden_norm/norm + 5 layers only) — the draft
// SHARES the target model's embed_tokens + lm_head, exactly as vLLM's loader skips
// them (qwen3_dflash.py:787-806 `skip_substrs.append("embed_tokens")` +
// lm_head untied-but-shared). The caller supplies them from the resolved target.
OwnedTensor TryLoadBf16(const TensorResolver& get, const std::string& name, bool nk) {
  try {
    return nk ? LoadBf16RawNK(get, name) : LoadBf16Direct(get, name);
  } catch (const std::runtime_error&) {
    return OwnedTensor{};
  }
}

// Concatenate several [N_i, K] BF16 raw-NK matrices along their output rows,
// preserving order (vLLM's QKV / gate_up stacked mapping). Sets nk=true.
OwnedTensor ConcatRawNK(const TensorResolver& get, const std::vector<std::string>& names,
                        const std::string& what) {
  std::vector<OwnedTensor> parts;
  int64_t total_n = 0, k = -1;
  parts.reserve(names.size());
  for (const std::string& n : names) {
    OwnedTensor t = LoadBf16Direct(get, n);
    VT_CHECK(t.rank == 2, "qwen3_dflash: expected 2-D for " + n);
    if (k < 0) k = t.shape[1];
    VT_CHECK(t.shape[1] == k, "qwen3_dflash: K mismatch concatenating " + what);
    total_n += t.shape[0];
    parts.push_back(std::move(t));
  }
  OwnedTensor out = MakeOwned(vt::DType::kBF16, {total_n, k});
  out.nk = true;
  size_t off = 0;
  for (const OwnedTensor& p : parts) {
    std::memcpy(out.bytes.data() + off, p.bytes.data(), p.bytes.size());
    off += p.bytes.size();
  }
  return out;
}

// The RoPE base a DFlash draft takes when NEITHER spelling declares one, which
// is upstream's own `set_default_rope_theta(config, default_theta=1000000)`
// (qwen3_dflash.py:304 @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`). Inert on every published draft:
// all four DFlash1 configs declare the flat key and both DFlash2 configs declare
// the nested one. It exists so that this builder answers what upstream answers
// rather than throwing where upstream defaults.
constexpr double kDflashDefaultRopeTheta = 1000000.0;

// DeclaredCausal: upstream's `is not None` test followed by its `bool(...)`
// coercion, for either spelling of the explicit causality (#1366).
//
// Upstream reads the key off a HuggingFace config object and writes
// `if is_causal is not None: return bool(is_causal)`, with the same shape one
// arm down for `dflash_config.causal` (qwen3_dflash.py:58-67 @
// vllm-project/vllm#52816 head `19c9351904df4c63042671bc67a866ca48dc7d6f`). So a
// config that spells the value `0` or `1` is honoured there, and a port that
// demanded a JSON boolean dropped it in SILENCE -- back onto the legacy rule,
// with nothing raised and only acceptance moving.
//
// The bound on the coercion is the OTHER container, not Python. `MakeDflashGgufConfig`
// reads the GGUF spelling `dflash.attention.causal` through `KvI64`, which takes
// every integer width and bool and names its own error on anything else
// (qwen3_dflash_gguf.cpp). This is that rule, so the two containers now answer
// identically: an absent key and a JSON null fall through, a boolean or a number
// is honoured, and a type neither container can coerce is refused BY NAME rather
// than dropped. Returns whether the key was declared; writes the resolved value
// only when it was.
bool DeclaredCausal(const nlohmann::json& obj, const char* key, bool* out) {
  if (!obj.is_object() || !obj.contains(key)) return false;
  const nlohmann::json& v = obj.at(key);
  if (v.is_null()) return false;
  VT_CHECK(v.is_boolean() || v.is_number(),
           std::string("qwen3_dflash: config key \"") + key +
               "\" must be a boolean or a number (upstream coerces it with "
               "bool(), and the GGUF arm reads the same value as an integer)");
  *out = v.is_boolean() ? v.get<bool>() : (v.get<double>() != 0.0);
  return true;
}

}  // namespace

std::vector<Qwen3DFlashLayerAttnMode> ResolveQwen3DFlashAttnModes(const HfConfig& config) {
  // Mirror _resolve_layer_attention + _dflash_layer_causal. WHICH REVISION each
  // half is read at, because the two differ and an unqualified anchor hides that:
  // `_resolve_layer_attention` is `qwen3_dflash.py:86-146` at the parity pin
  // `555967922` and `:109-169` at vllm-project/vllm#52816 head
  // `19c9351904df4c63042671bc67a866ca48dc7d6f`, and its body is the SAME at both
  // -- the PR moves the function down the file and does not edit it.
  // `_dflash_layer_causal` is `:58-64` at the pin and `:58-67` at the PR head, and
  // that one IS edited: the head adds the top-level `is_causal` arm this function
  // resolves first. So the sliding-window half below mirrors the pin, the
  // causality half mirrors the PR head, and both anchors are BEYOND-PIN-qualified
  // rather than pointing at one revision that carries only half the rule.
  // dflash_config overrides live in config.raw["dflash_config"].
  //
  // SPEC-DFLASH2 W1 (#1314), BEYOND-PIN. `_dflash_layer_causal` resolves an
  // EXPLICIT top-level `is_causal` before it falls back to anything else
  // (qwen3_dflash.py:58-67 @ vllm-project/vllm#52816 head
  // `19c9351904df4c63042671bc67a866ca48dc7d6f`; the pinned 555967922 form at
  // :58-64 has only the two legacy arms). So the order is exactly:
  //
  //   1. top-level `is_causal`, if DECLARED    -> every layer takes bool(it)
  //   2. `dflash_config.causal`, if DECLARED   -> every layer takes bool(it)
  //   3. the legacy rule  -> causal iff `layer_types[i]` IS `sliding_attention`
  //
  // Arm 3 reads the DECLARED `layer_types` and not the resolved layer type, so
  // `dflash_config.use_swa` does not reach it; see the comment on the loop body
  // and #1366. Arms 1 and 2 test PRESENCE and then coerce, as upstream's
  // `bool(...)` does; see `DeclaredCausal` above.
  //
  // The order is the whole change. `z-lab/Qwen3.8-27B-DFlash2` declares all five
  // layers `sliding_attention` AND `is_causal false`; under the legacy rule alone
  // every layer runs CAUSAL, the draft still emits plausible tokens, a token gate
  // against our own output sees nothing, and only ACCEPTANCE falls -- which the
  // lossless verify hides (.agents/specs/dflash2-spec-decode.md D4).
  //
  // No published DFlash1 checkpoint declares `is_causal`, so arm 1 never fires
  // for one and their resolution is unchanged, which is what upstream does in the
  // same commit.
  //
  // The WINDOW is a separate answer and none of this touches it: upstream returns
  // `(sliding_window, causal)` as two independent resolutions, and a
  // non-causal SWA layer still attends within its window.
  static const std::string kSliding = "sliding_attention";
  const nlohmann::json empty = nlohmann::json::object();
  const nlohmann::json& dflash =
      (config.raw.is_object() && config.raw.contains("dflash_config") &&
       config.raw.at("dflash_config").is_object())
          ? config.raw.at("dflash_config")
          : empty;
  const bool use_swa = dflash.value("use_swa", false);
  // `||` is the precedence, and it short-circuits exactly where upstream returns:
  // a declared `is_causal` answers and `dflash_config.causal` is never consulted.
  bool explicit_causal = false;
  const bool has_explicit_causal = DeclaredCausal(config.raw, "is_causal", &explicit_causal) ||
                                   DeclaredCausal(dflash, "causal", &explicit_causal);

  const std::vector<std::string>& lt = config.layer_types;
  int64_t num_sliding = 0;
  for (const std::string& s : lt) num_sliding += (s == kSliding) ? 1 : 0;
  const bool any_sliding = num_sliding > 0;

  std::vector<Qwen3DFlashLayerAttnMode> modes;
  modes.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    bool is_sliding;
    if (lt.empty() || (use_swa && !any_sliding)) {
      is_sliding = use_swa;
    } else {
      is_sliding = lt[static_cast<size_t>(i)] == kSliding;
    }
    // The legacy fallback reads the RAW `layer_types`, and NOT the `is_sliding`
    // resolved just above (#1366). The two differ whenever `dflash_config.use_swa`
    // is set, because `use_swa` forces SWA onto every layer -- an absent
    // `layer_types` and an all-full one alike -- while upstream's fallback is
    // `bool(layer_types) and layer_types[i] == _SLIDING_ATTENTION`
    // (qwen3_dflash.py:66-67 @ the PR head), which such a config fails. Upstream's
    // own `_resolve_layer_attention` docstring table states it as a row --
    // `layer_types=None` + `use_swa=True` -> causal False -- and names
    // `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` as the published checkpoint of that
    // shape. `use_swa` moves the WINDOW, never the causality. Reading `is_sliding`
    // here ran every layer of such a DFlash1 draft CAUSAL against upstream's
    // non-causal, which is acceptance-only and invisible to a token gate because
    // the verify is lossless.
    const bool legacy_causal =
        static_cast<size_t>(i) < lt.size() && lt[static_cast<size_t>(i)] == kSliding;
    Qwen3DFlashLayerAttnMode m;
    m.causal = has_explicit_causal ? explicit_causal : legacy_causal;
    if (is_sliding) {
      int64_t win = 0;
      if (dflash.contains("swa_window_size") && dflash.at("swa_window_size").is_number())
        win = dflash.at("swa_window_size").get<int64_t>();
      else if (config.sliding_window.has_value())
        win = config.sliding_window.value();
      VT_CHECK(win > 0,
               "qwen3_dflash: sliding attention needs dflash_config.swa_window_size "
               "or top-level sliding_window");
      m.sliding_window = win;
    } else {
      m.sliding_window = 0;
    }
    modes.push_back(m);
  }
  return modes;
}

HfConfig MakeQwen3DFlashDraftConfig(const nlohmann::json& c) {
  HfConfig cfg;
  cfg.hidden_size = c.at("hidden_size").get<int64_t>();
  cfg.num_attention_heads = c.at("num_attention_heads").get<int64_t>();
  cfg.num_key_value_heads = c.at("num_key_value_heads").get<int64_t>();
  cfg.head_dim = c.at("head_dim").get<int64_t>();
  cfg.rotary_dim = cfg.head_dim;
  // ROPE THETA, in BOTH spellings (SPEC-DFLASH2 W2, spec `## Owed` O3, #1314).
  // `transformers` 5 moved the RoPE settings under `rope_parameters`, and BOTH
  // published DFlash2 drafts nest it there and declare NO top-level `rope_theta`
  // -- so the flat `c.at("rope_theta")` threw before any DFlash2 mechanism could
  // be reached. Upstream reads one resolved `config.rope_parameters`
  // (`qwen3_dflash.py:340` @ vllm-project/vllm#52816 head
  // `19c9351904df4c63042671bc67a866ca48dc7d6f`) after
  // `set_default_rope_theta(config, default_theta=1000000)` (`:304`), which is
  // where the fallback value below comes from. This is a FALLBACK and not a
  // replacement: every published DFlash1 draft carries the flat spelling and
  // must keep taking it, which is why the flat key is tested FIRST.
  cfg.rope_theta = kDflashDefaultRopeTheta;
  if (c.contains("rope_theta") && c.at("rope_theta").is_number()) {
    cfg.rope_theta = c.at("rope_theta").get<double>();
  } else if (c.contains("rope_parameters") && c.at("rope_parameters").is_object() &&
             c.at("rope_parameters").contains("rope_theta") &&
             c.at("rope_parameters").at("rope_theta").is_number()) {
    cfg.rope_theta = c.at("rope_parameters").at("rope_theta").get<double>();
  }
  cfg.intermediate_size = c.at("intermediate_size").get<int64_t>();
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.num_hidden_layers = c.at("num_hidden_layers").get<int64_t>();
  cfg.rms_norm_eps = c.at("rms_norm_eps").get<double>();
  cfg.sliding_window = c.at("sliding_window").get<int64_t>();
  // LAYER TYPES are OPTIONAL (spec `## Owed` O4, #1314, #1366). Upstream reads
  // `getattr(config, "layer_types", None)` (`qwen3_dflash.py:134`, and `:66` in
  // `_dflash_layer_causal`, @ that head), so an absent key is upstream's `None`
  // and an EMPTY vector here -- which is exactly the state
  // `ResolveQwen3DFlashAttnModes` already treats as "no declared layer types",
  // taking `dflash_config.use_swa` for the window and NON-causal for the
  // causality. `c.at("layer_types")` instead threw
  // `[json.exception.out_of_range.403]` on the only published draft of that
  // shape (`XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`), which is why #1366's
  // `use_swa` repair was UNREACHED at its own merge commit.
  if (c.contains("layer_types") && c.at("layer_types").is_array()) {
    cfg.layer_types = c.at("layer_types").get<std::vector<std::string>>();
  }
  cfg.raw = nlohmann::json::object();
  cfg.raw["dflash_config"] = c.at("dflash_config");
  const nlohmann::json& dflash_cfg = cfg.raw.at("dflash_config");
  // ATTENTION SINK BIAS is REFUSED BY NAME (spec `## Owed` O4, #1314).
  //
  // Upstream reads `dflash_config.attention_sink_bias`, falling back to a
  // top-level `add_swa_attention_sink_bias`, and when it is truthy it allocates a
  // per-head `attention_sink_bias` parameter and passes it into its `Attention`
  // as `sinks=` (`qwen3_dflash.py:309-313` and `:240-257` @ that head). This lane
  // has NO attention sink of any kind: `vt::DFlashBlockAttention` and its paged
  // sibling compute a plain max-subtracted softmax with no extra denominator
  // term, and the loader has no name for the tensor.
  //
  // Refusing rather than ignoring is the whole point of this arm. The key sits on
  // `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`, which is ALSO the only published draft
  // that declares no `layer_types` -- so before the fallback above, that
  // checkpoint died on a loud `key 'layer_types' not found`, and the fallback
  // alone would have converted that loud failure into a draft that loads with the
  // sinks silently absent. A missing sink moves acceptance and nothing else: the
  // verify is lossless, so the emitted tokens are still the target's and no token
  // gate can see it. That is the exact defect class this row exists to remove.
  //
  // A FALSY value is upstream's own default and is not refused, because upstream
  // then creates no sink parameter at all and the two engines agree.
  const auto sink_declared = [](const nlohmann::json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key)) return false;
    const nlohmann::json& v = obj.at(key);
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0.0;
    return !v.is_null();
  };
  VT_CHECK(!sink_declared(dflash_cfg, "attention_sink_bias") &&
               !sink_declared(c, "add_swa_attention_sink_bias"),
           "qwen3_dflash: this draft declares a per-head attention sink "
           "(dflash_config.attention_sink_bias / add_swa_attention_sink_bias), and "
           "this engine has no attention sink: vt::DFlashBlockAttention computes a "
           "plain softmax with no sink term. Upstream allocates the parameter and "
           "passes it as Attention(sinks=...) "
           "(vllm/model_executor/models/qwen3_dflash.py:240-257,309-313 @ "
           "vllm-project/vllm#52816 head 19c9351904df4c63042671bc67a866ca48dc7d6f). "
           "Loading without it would succeed and draft worse tokens with no visible "
           "symptom, because the verify is lossless and only acceptance falls. Owed "
           "by row SPEC-DFLASH2 (.agents/specs/dflash2-spec-decode.md `## Owed` O4), "
           "issue #1314 (https://github.com/mudler/vllm.cpp/issues/1314).");
  // BLOCK SIZE, in BOTH spellings (spec `## Owed` O3). The DFlash1 drafts declare
  // it at the top level; both published DFlash2 drafts declare it ONLY as
  // `dflash_config.block_size`, so the flat read threw on them too. Upstream
  // never reads a top-level `block_size` in this file at all -- the conv's block
  // is `1 + speculative_config.num_speculative_tokens`
  // (`qwen3_dflash2.py` `DFlash2Qwen3DecoderLayer.__init__` @ that head) and the
  // checkpoint key only supplies that value's DEFAULT, which this engine's loader
  // resolves from the CLI. The key is carried for the DFlash1 callers that read
  // it and is not the conv's authority.
  if (c.contains("block_size")) {
    cfg.raw["block_size"] = c.at("block_size");
  } else if (dflash_cfg.contains("block_size")) {
    cfg.raw["block_size"] = dflash_cfg.at("block_size");
  }
  // SPEC-DFLASH2 W1 (#1314): the top-level attention semantics, which
  // ResolveQwen3DFlashAttnModes resolves ahead of every legacy arm. Upstream gets
  // this key for free by reading it off a HuggingFace config object
  // (`getattr(config, "is_causal", None)`, qwen3_dflash.py:60 @ the PR head);
  // this builder copies named keys, so a key it drops is a key the resolution can
  // never see. Optional, and absent from every DFlash1 checkpoint.
  // Carried whenever it is DECLARED, in whatever scalar the checkpoint spells it
  // (#1366). The builder plumbs and the resolution decides: gating the carry on a
  // type would put a second, narrower predicate in front of `DeclaredCausal` and
  // reintroduce exactly the silent drop that key exists to prevent.
  if (c.contains("is_causal") && !c.at("is_causal").is_null()) {
    cfg.raw["is_causal"] = c.at("is_causal");
  }
  return cfg;
}

Qwen3DFlashWeights LoadQwen3DFlash(const TensorResolver& get, const HfConfig& config,
                                   int64_t num_taps, int32_t mask_token_id) {
  VT_CHECK(config.hidden_size > 0 && config.num_hidden_layers > 0,
           "qwen3_dflash: invalid config dims");
  VT_CHECK(num_taps > 0, "qwen3_dflash: num_taps (len(target_layer_ids)) must be > 0");

  Qwen3DFlashWeights out;
  out.num_taps = num_taps;
  out.mask_token_id = mask_token_id;
  out.draft_vocab_size = config.vocab_size;

  // SPEC-DFLASH2 W2 (#1314): the grouped-convolution geometry, read off the
  // draft's own dflash_config. A DFlash1 checkpoint declares NONE of these keys
  // and carries no conv tensor, so `conv_taps` stays 0, `IsDflash2()` is false,
  // and everything below this point loads exactly as before -- which is the
  // inertness the DFlash1 gates assert.
  {
    const nlohmann::json empty = nlohmann::json::object();
    const nlohmann::json& dflash =
        (config.raw.is_object() && config.raw.contains("dflash_config") &&
         config.raw.at("dflash_config").is_object())
            ? config.raw.at("dflash_config")
            : empty;
    const bool has_taps =
        dflash.contains("conv_kernel_size") && dflash.at("conv_kernel_size").is_number();
    const bool has_group =
        dflash.contains("conv_group_size") && dflash.at("conv_group_size").is_number();
    // Both or neither. A checkpoint declaring one alone is not a shape this port
    // knows how to size, and guessing the other would size the projection wrong
    // and be invisible: the draft would still emit the target's tokens.
    VT_CHECK(has_taps == has_group,
             "qwen3_dflash: dflash_config declares only one of conv_kernel_size / "
             "conv_group_size; a DFlash2 draft declares both (SPEC-DFLASH2, #1314)");
    if (has_taps) {
      out.conv_taps = dflash.at("conv_kernel_size").get<int64_t>();
      out.conv_group_size = dflash.at("conv_group_size").get<int64_t>();
      VT_CHECK(out.conv_taps > 0 && out.conv_group_size > 0,
               "qwen3_dflash: conv_kernel_size and conv_group_size must be > 0");
      // Upstream refuses a group size that does not divide hidden
      // (`DFlashGroupedConv.__init__` @ the PR head), with the same polarity.
      VT_CHECK(config.hidden_size % out.conv_group_size == 0,
               "qwen3_dflash: conv_group_size must divide hidden_size");
      // The DEFAULT query block, from the checkpoint. The loader overwrites it
      // with `1 + k` once the resolved speculative config is known; see the field
      // comment on Qwen3DFlashWeights::conv_block_size.
      if (config.raw.contains("block_size") && config.raw.at("block_size").is_number()) {
        out.conv_block_size = config.raw.at("block_size").get<int64_t>();
      }
    }
    // SPEC-DFLASH2 W3 (#1314): the CANDIDATE SELECTOR's geometry and the two
    // output scalars, read off the same dflash_config. A DFlash2 checkpoint
    // carries both mechanisms, so these are required whenever the conv keys are
    // present rather than optional beside them.
    if (has_taps) {
      VT_CHECK(dflash.contains("selector_rank") && dflash.at("selector_rank").is_number() &&
                   dflash.contains("selector_top_k") &&
                   dflash.at("selector_top_k").is_number(),
               "qwen3_dflash: this draft declares the DFlash2 convolution keys but no "
               "selector_rank/selector_top_k; a DFlash2 draft declares both mechanisms "
               "(SPEC-DFLASH2 W3, #1314)");
      out.candidate_selector.rank = dflash.at("selector_rank").get<int64_t>();
      out.candidate_selector.top_k = dflash.at("selector_top_k").get<int64_t>();
      VT_CHECK(out.candidate_selector.rank > 0 && out.candidate_selector.top_k > 0,
               "qwen3_dflash: selector_rank and selector_top_k must be > 0");
      VT_CHECK(out.candidate_selector.top_k <= config.vocab_size,
               "qwen3_dflash: selector_top_k must not exceed the vocabulary");
      // `float(draft_config.get("output_multiplier", 1.0))` and
      // `float(draft_config.get("final_logit_softcapping") or 0.0)`, then
      // disabled unless > 0 -- upstream's own defaults
      // (`DFlash2Qwen3ForCausalLM.__init__` @ the PR head). Both are ABSENT from
      // `z-lab/Qwen3.8-27B-DFlash2` and PRESENT on
      // `z-lab/Muse-Glimmer-30B-DFlash2` (#1327), so a gate built from the 27B
      // draft alone measures the default path only.
      if (dflash.contains("output_multiplier") && dflash.at("output_multiplier").is_number())
        out.candidate_selector.output_multiplier =
            static_cast<float>(dflash.at("output_multiplier").get<double>());
      if (dflash.contains("final_logit_softcapping") &&
          dflash.at("final_logit_softcapping").is_number()) {
        const float cap =
            static_cast<float>(dflash.at("final_logit_softcapping").get<double>());
        out.candidate_selector.final_logit_softcapping = cap > 0.0f ? cap : 0.0f;
      }
      // INPUT EMBEDDING SCALE is REFUSED BY NAME when it is declared and is not
      // upstream's default. Upstream applies it inside `embed_input_ids`
      // (`DFlash2Qwen3Model.embed_input_ids` @ the PR head), which is a third
      // call site in each of this engine's three layer bodies; NEITHER published
      // DFlash2 draft declares the key, so implementing it would land three
      // unreachable call sites, and IGNORING it would run a quietly different
      // model on the first checkpoint that sets it -- acceptance-only and
      // token-invisible, because the verify is lossless. The polarity is the one
      // W2 set for `dflash_config.attention_sink_bias` (spec `## Owed` O4b): a
      // value equal to upstream's default is upstream's own no-op and is not
      // refused. Owed: `## Owed` O9.
      if (dflash.contains("input_embedding_scale") &&
          dflash.at("input_embedding_scale").is_number()) {
        const double scale = dflash.at("input_embedding_scale").get<double>();
        VT_CHECK(scale == 1.0,
                 "qwen3_dflash: this draft declares dflash_config.input_embedding_scale "
                 "!= 1.0, and this engine does not apply it. Upstream scales the draft's "
                 "token embedding by it (DFlash2Qwen3Model.embed_input_ids, "
                 "vllm/model_executor/models/qwen3_dflash2.py @ "
                 "vllm-project/vllm#52816 head "
                 "66e5414c6d75a8529473d977f7458c140bbab8a0). Loading without it would "
                 "succeed and draft worse tokens with no visible symptom, because the "
                 "verify is lossless and only acceptance falls. Owed by row SPEC-DFLASH2 "
                 "(.agents/specs/dflash2-spec-decode.md `## Owed` O9), issue #1314 "
                 "(https://github.com/mudler/vllm.cpp/issues/1314).");
      }
    }
  }

  // embed_tokens + lm_head are SHARED from the target (the draft ckpt omits them,
  // see TryLoadBf16); load if present, else leave empty for the caller to fill.
  out.embed_tokens = TryLoadBf16(get, "embed_tokens.weight", /*nk=*/false);
  out.fc = LoadBf16RawNK(get, "fc.weight");
  out.hidden_norm = LoadBf16Direct(get, "hidden_norm.weight");
  out.final_norm = LoadBf16Direct(get, "norm.weight");
  out.lm_head = TryLoadBf16(get, "lm_head.weight", /*nk=*/true);

  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(config);
  out.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    const std::string base = "layers." + std::to_string(i) + ".";
    const std::string attn = base + "self_attn.";
    const std::string mlp = base + "mlp.";
    Qwen3DFlashLayerWeights layer;
    layer.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
    layer.post_attention_layernorm = LoadBf16Direct(get, base + "post_attention_layernorm.weight");
    layer.qkv_proj = ConcatRawNK(
        get, {attn + "q_proj.weight", attn + "k_proj.weight", attn + "v_proj.weight"}, "qkv");
    layer.o_proj = LoadBf16RawNK(get, attn + "o_proj.weight");
    layer.q_norm = LoadBf16Direct(get, attn + "q_norm.weight");
    layer.k_norm = LoadBf16Direct(get, attn + "k_norm.weight");
    layer.gate_up_proj =
        ConcatRawNK(get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"}, "gate_up");
    layer.down_proj = LoadBf16RawNK(get, mlp + "down_proj.weight");
    layer.attn_mode = modes[static_cast<size_t>(i)];
    // SPEC-DFLASH2 W2 (#1314): the two grouped convolutions, under the exact
    // names the published checkpoint stores them under. Loaded only for a DFlash2
    // draft; a DFlash1 checkpoint has no such tensor and asking for one would
    // throw "tensor not found" on every existing drafter.
    if (out.IsDflash2()) {
      const int64_t groups = config.hidden_size / out.conv_group_size;
      for (const char* which : {"attention_conv.", "mlp_conv."}) {
        const std::string cp = base + which;
        Qwen3DFlashConvWeights conv;
        conv.base_kernel = LoadBf16Direct(get, cp + "base_kernel");
        conv.kernel_projection = LoadBf16RawNK(get, cp + "kernel_projection.weight");
        // SHAPES, asserted rather than assumed. `base_kernel` is
        // [SIDES=2, taps, H] -- dim 0 is prepare/finish and NOT a tap, and on the
        // published 27B draft both are 2, so nothing but this check separates a
        // correct load from a transposed one. `kernel_projection` is
        // [2*taps*num_groups, H]: one projection of the sublayer input carrying
        // BOTH sides' deltas.
        VT_CHECK(conv.base_kernel.rank == 3 && conv.base_kernel.shape[0] == 2 &&
                     conv.base_kernel.shape[1] == out.conv_taps &&
                     conv.base_kernel.shape[2] == config.hidden_size,
                 "qwen3_dflash: " + cp + "base_kernel must be [2, conv_kernel_size, H]");
        VT_CHECK(conv.kernel_projection.rank == 2 &&
                     conv.kernel_projection.shape[0] == 2 * out.conv_taps * groups &&
                     conv.kernel_projection.shape[1] == config.hidden_size,
                 "qwen3_dflash: " + cp +
                     "kernel_projection.weight must be [2*conv_kernel_size*num_groups, H]");
        if (std::string(which) == "attention_conv.") {
          layer.attention_conv = std::move(conv);
        } else {
          layer.mlp_conv = std::move(conv);
        }
      }
    }
    out.layers.push_back(std::move(layer));
  }

  // SPEC-DFLASH2 W3 (#1314): the candidate selector's three tensors, under the
  // exact names the published checkpoint stores them under. Loaded only for a
  // DFlash2 draft; a DFlash1 checkpoint has none of them and asking would throw
  // "tensor not found" on every existing drafter.
  if (out.IsDflash2()) {
    Dflash2SelectorWeights& sel = out.candidate_selector;
    const int64_t rank = sel.rank;
    sel.hidden_projection =
        LoadBf16RawNK(get, "candidate_selector.hidden_projection.weight");
    sel.predecessor_codebook = LoadBf16Direct(get, "candidate_selector.predecessor_codebook");
    sel.successor_codebook = LoadBf16Direct(get, "candidate_selector.successor_codebook");
    // SHAPES, asserted rather than assumed. `hidden_projection` is a
    // ReplicatedLinear [rank <- H] and the codebooks are [vocab, rank]; on the
    // published 27B draft rank 256 differs from every other axis, so a
    // transposed load would be caught here rather than by a wrong answer.
    VT_CHECK(sel.hidden_projection.rank == 2 && sel.hidden_projection.shape[0] == rank &&
                 sel.hidden_projection.shape[1] == config.hidden_size,
             "qwen3_dflash: candidate_selector.hidden_projection.weight must be [rank, H]");
    for (const OwnedTensor* book : {&sel.predecessor_codebook, &sel.successor_codebook})
      VT_CHECK(book->rank == 2 && book->shape[0] == config.vocab_size &&
                   book->shape[1] == rank,
               "qwen3_dflash: candidate_selector.{predecessor,successor}_codebook must be "
               "[vocab, rank]");
  }

  // fc input width validation: [H, H*num_taps].
  VT_CHECK(out.fc.shape[0] == config.hidden_size &&
               out.fc.shape[1] == config.hidden_size * num_taps,
           "qwen3_dflash: fc.weight must be [H, H*num_taps]");
  return out;
}

Qwen3DFlashWeights LoadQwen3DFlash(const std::vector<SafetensorsFile>& shards,
                                   const HfConfig& config, int64_t num_taps,
                                   int32_t mask_token_id) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  // Resolver: try the bare checkpoint name, then a "model."-prefixed fallback
  // (vLLM adds "model." at load; a checkpoint may or may not ship it pre-added).
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    std::string key = name;
    if (it == where.end()) {
      key = "model." + name;
      it = where.find(key);
    }
    VT_CHECK(it != where.end(), "qwen3_dflash: tensor not found: " + name);
    return it->second->Get(key);
  };
  return LoadQwen3DFlash(get, config, num_taps, mask_token_id);
}

}  // namespace vllm
