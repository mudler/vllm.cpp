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
  cfg.rope_theta = c.at("rope_theta").get<double>();
  cfg.intermediate_size = c.at("intermediate_size").get<int64_t>();
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.num_hidden_layers = c.at("num_hidden_layers").get<int64_t>();
  cfg.rms_norm_eps = c.at("rms_norm_eps").get<double>();
  cfg.sliding_window = c.at("sliding_window").get<int64_t>();
  cfg.layer_types = c.at("layer_types").get<std::vector<std::string>>();
  cfg.raw = nlohmann::json::object();
  cfg.raw["dflash_config"] = c.at("dflash_config");
  cfg.raw["block_size"] = c.at("block_size");
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
    out.layers.push_back(std::move(layer));
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
