// vllm.cpp ORIGINAL — the `qwen4exp` GGUF config builder. See the header for
// why this family owns its own translation unit and which llama.cpp pull
// request its key names follow.
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"

#include <string>
#include <vector>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

const std::string kPrefix = "qwen4exp.";

int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("qwen4_exp gguf: key " + key +
                               " is not an integer");
  }
}

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen4_exp gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) : dflt;
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen4_exp gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

// A required integer ARRAY key, read out as int64. Arrays carry the per-layer
// schedules this architecture is built out of, and reading one wrong splits the
// model into layers of the wrong kind, so a missing or wrong-typed array fails
// here rather than defaulting.
std::vector<int64_t> ReqIntArray(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen4_exp gguf: missing metadata key " + key);
  VT_CHECK(v->TypeId() == kGgufArray,
           "qwen4_exp gguf: key " + key + " must be an array");
  std::vector<int64_t> out;
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    out.push_back(KvInt(e, key));
  }
  return out;
}

std::vector<int64_t> OptIntArray(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) return {};
  std::vector<int64_t> out;
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    out.push_back(KvInt(e, key));
  }
  return out;
}

}  // namespace

bool IsQwen4ExpGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kQwen4ExpGgufArch;
}

HfConfig Qwen4ExpHfConfigFromGguf(const GgufFile& gguf) {
  VT_CHECK(IsQwen4ExpGguf(gguf),
           "qwen4_exp gguf: general.architecture must be '" +
               std::string(kQwen4ExpGgufArch) + "'");
  const std::string p = kPrefix;

  HfConfig c;
  // The HF `model_type` and architecture string, verbatim from the released
  // `Qwen/Qwen3.8-Flash-Next` config.json (read 2026-08-26). They are NOT
  // derived from the GGUF architecture key: llama.cpp's `qwen4exp` is a family
  // key, not a model class, and the registry keys on the HF class.
  c.model_type = "qwen4_exp";
  c.architectures = {"Qwen4ExpForConditionalGeneration"};

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim = ReqInt(gguf, p + "attention.key_length");
  c.max_position_embeddings = ReqInt(gguf, p + "context_length");
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.rope_theta = ReqFloat(gguf, p + "rope.freq_base");
  // `rope.dimension_count` IS `partial_rotary_factor * head_dim` already
  // resolved by the converter: 0.25 * 256 == 64 on the released checkpoint.
  c.rotary_dim = ReqInt(gguf, p + "rope.dimension_count");

  // MoE. `expert_feed_forward_length` (640) is what makes every routed expert
  // row indivisible by 256 and therefore un-encodable as any K-quant — the
  // reason the shipped file reaches for IQ4_NL and IQ1_S/IQ2_XXS instead.
  c.num_experts = ReqInt(gguf, p + "expert_count");
  c.num_experts_per_tok = ReqInt(gguf, p + "expert_used_count");
  c.moe_intermediate_size = ReqInt(gguf, p + "expert_feed_forward_length");
  c.shared_expert_intermediate_size =
      OptInt(gguf, p + "expert_shared_feed_forward_length",
             c.moe_intermediate_size);
  c.intermediate_size = c.moe_intermediate_size;

  // Gated Delta Net. llama.cpp's converter reuses the Mamba key namespace for
  // the linear-attention geometry, so the names read `ssm.*` while the values
  // are GDN's — the same mapping `qwen3next` already uses in this tree.
  c.linear_num_key_heads = ReqInt(gguf, p + "ssm.group_count");
  c.linear_num_value_heads = ReqInt(gguf, p + "ssm.time_step_rank");
  c.linear_key_head_dim = ReqInt(gguf, p + "ssm.state_size");
  c.linear_value_head_dim = c.linear_key_head_dim;
  c.linear_conv_kernel_dim = ReqInt(gguf, p + "ssm.conv_kernel");
  // `ssm.inner_size` is the value stream width and must equal
  // num_value_heads * value_head_dim; a file where it does not is malformed and
  // would silently size every GDN buffer wrong.
  const int64_t inner = OptInt(gguf, p + "ssm.inner_size", 0);
  if (inner != 0) {
    VT_CHECK(inner == c.linear_num_value_heads * c.linear_value_head_dim,
             "qwen4_exp gguf: ssm.inner_size disagrees with "
             "ssm.time_step_rank * ssm.state_size");
  }
  // Not in the GGUF container at ALL, and not inferable from it: the released
  // config.json says `output_gate_type: "sigmoid"`, where this tree's default
  // (and Qwen3.5's value) is "silu". Left at the default it would be a silent
  // wrong activation on the GDN output gate, which no shape check can see.
  c.output_gate_type = "sigmoid";
  c.mamba_ssm_dtype = "float32";  // config.json `mamba_ssm_dtype`

  // The layer schedule. `full_attention_interval` 4 gives the documented
  // 3 x linear_attention -> 1 x qwen_sparse_attention pattern, and the file
  // corroborates it twice over: `attention.compress_ratios` is non-zero on
  // exactly those layers, and `blk.N.indexer.*` exists on exactly those layers.
  const int64_t interval = ReqInt(gguf, p + "full_attention_interval");
  VT_CHECK(interval > 0, "qwen4_exp gguf: full_attention_interval must be > 0");
  c.layer_types.reserve(static_cast<size_t>(c.num_hidden_layers));
  for (int64_t il = 0; il < c.num_hidden_layers; ++il) {
    const bool full = ((il + 1) % interval) == 0;
    c.layer_types.emplace_back(full ? "full_attention" : "linear_attention");
  }
  const std::vector<int64_t> ratios =
      OptIntArray(gguf, p + "attention.compress_ratios");
  if (!ratios.empty()) {
    VT_CHECK(static_cast<int64_t>(ratios.size()) == c.num_hidden_layers,
             "qwen4_exp gguf: attention.compress_ratios length must equal "
             "block_count");
    for (int64_t il = 0; il < c.num_hidden_layers; ++il) {
      const bool full = c.layer_types[static_cast<size_t>(il)] ==
                        "full_attention";
      VT_CHECK((ratios[static_cast<size_t>(il)] != 0) == full,
               "qwen4_exp gguf: attention.compress_ratios disagrees with the "
               "full_attention_interval schedule");
    }
  }

  // Interleaved mRoPE. `rope.dimension_sections` is llama.cpp's 4-slot array
  // and HF's `mrope_section` is 3 long, so the trailing zero is dropped rather
  // than carried: a 4-element section list would make the sum check fail
  // downstream for a reason that has nothing to do with the model.
  std::vector<int64_t> sections =
      OptIntArray(gguf, p + "rope.dimension_sections");
  while (!sections.empty() && sections.back() == 0) sections.pop_back();
  if (!sections.empty()) {
    int64_t sum = 0;
    for (int64_t s : sections) sum += s;
    VT_CHECK(sum * 2 == c.rotary_dim,
             "qwen4_exp gguf: rope.dimension_sections must sum to "
             "rope.dimension_count / 2");
    c.rope_parameters.mrope_section.clear();
    for (int64_t s : sections) {
      c.rope_parameters.mrope_section.push_back(s);
    }
    c.rope_parameters.mrope_interleaved = true;
    c.rope_parameters.rope_theta = c.rope_theta;
    c.has_rope_parameters = true;
  }

  // vocab_size: no metadata key, so read the gather table's own leading dim,
  // exactly as the qwen3_5 and muse-glimmer builders do.
  c.vocab_size = OptInt(gguf, p + "vocab_size", 0);
  if (c.vocab_size == 0) {
    c.vocab_size = gguf.Get("token_embd.weight").shape[0];
  }
  c.torch_dtype = "bfloat16";

  // --- the architecture-specific numbers, under HF's OWN key names ----------
  //
  // These are carried in `raw` rather than typed, because no other model in
  // this tree has them and the fields would be dead everywhere else. The names
  // are the released `Qwen/Qwen3.8-Flash-Next` config.json's, read 2026-08-26,
  // NOT names invented here: an implementer of the model waves reads the same
  // spellings whether the config came from a GGUF or from a config.json.
  nlohmann::json raw = nlohmann::json::object();
  raw["model_type"] = c.model_type;
  raw["architectures"] = c.architectures;
  nlohmann::json text = nlohmann::json::object();
  text["hc_count"] = ReqInt(gguf, p + "hyper_connection.count");
  text["hc_lowrank"] = ReqInt(gguf, p + "hyper_connection.low_rank");
  text["indexer_n_heads"] = ReqInt(gguf, p + "attention.indexer.head_count");
  text["indexer_head_dim"] = ReqInt(gguf, p + "attention.indexer.key_length");
  text["indexer_budget"] = ReqInt(gguf, p + "attention.indexer.top_k");
  // NOT IN THE CONTAINER, and 1 is not a guess: the indexer is an MQA with one
  // key head by construction (`indexer_kv_heads` in the released config, and
  // upstream's validator refuses any other value), which is why llama.cpp
  // writes no key for it. Added by W5 (#2031, issue #2064) because its absence
  // made every real `qwen4exp` file UNPARSEABLE: the QSA group is
  // all-or-nothing in `ParseQwen4ExpParams`, so four-of-five present is refused
  // with "QSA config is missing required fields: indexer_kv_heads" — a refusal
  // no one had seen, because W6a's gate builds the config and never parses it
  // and W1's gate parses a config.json and never builds one from a file.
  text["indexer_kv_heads"] = 1;
  text["full_attention_interval"] = interval;
  text["ngram_size"] = ReqInt(gguf, p + "ple.ngram_size");
  text["heads_per_ngram"] = ReqInt(gguf, p + "ple.heads_per_ngram");
  text["ple_conv_kernel_size"] = ReqInt(gguf, p + "ple.conv_kernel");
  // The PLE table's per-head row width. HF states the TOTAL (`ple_embed_dim`
  // 2560); the GGUF states the per-head slice (160), and 2560 == 160 * 16 heads.
  // Both are recorded so neither wave has to reconstruct the other.
  const int64_t ple_row = ReqInt(gguf, p + "embedding_length_per_layer_input");
  text["ple_embed_dim_per_head"] = ple_row;
  // W5 (#2031) also emits the TOTAL under HF's own key. Without it
  // `ParseQwen4ExpParams` falls back to its documented default of `hidden_size`
  // (`ple_embed_dim` is set in upstream's `__post_init__`), which happens to be
  // right on the released checkpoint — 2560 either way — and would be silently
  // wrong on any config where the two differ. A default that is correct only by
  // coincidence is the shape of defect this row keeps finding.
  const int64_t ngram_heads_gguf =
      (ReqInt(gguf, p + "ple.ngram_size") - 1) *
      ReqInt(gguf, p + "ple.heads_per_ngram");
  VT_CHECK(ngram_heads_gguf > 0,
           "qwen4_exp gguf: ple.ngram_size and ple.heads_per_ngram give a "
           "non-positive n-gram head count");
  text["ple_embed_dim"] = ple_row * ngram_heads_gguf;
  // `indexer_compress_ratio`: the file states it per LAYER; HF states the one
  // value. Take the first non-zero and require the rest to agree, so a file
  // with a mixed schedule is a loud failure rather than a silent first-wins.
  if (!ratios.empty()) {
    int64_t cr = 0;
    for (int64_t r : ratios) {
      if (r == 0) continue;
      if (cr == 0) cr = r;
      VT_CHECK(r == cr,
               "qwen4_exp gguf: attention.compress_ratios is not uniform "
               "across the sparse layers");
    }
    if (cr != 0) text["indexer_compress_ratio"] = cr;
  }
  // The n-gram head table. `head_offsets` and `head_vocab_sizes` have no HF
  // counterpart at all: HF derives them from `ngram_vocab_size_base` and
  // `make_ngram_vocab_size_divisible_by`, while the GGUF writes the resolved
  // arrays. The resolved arrays are the authority here, because they are what
  // the shipped tensor was built against.
  text["ple_head_offsets"] = OptIntArray(gguf, p + "ple.head_offsets");
  text["ple_head_vocab_sizes"] = OptIntArray(gguf, p + "ple.head_vocab_sizes");
  text["ple_layer_multipliers"] = OptIntArray(gguf, p + "ple.layer_multipliers");
  // `ple.layers` is recorded under its GGUF name AND mapped onto HF's
  // `ple_layer_ids`. W6a left the mapping open on the ground that "nothing in
  // either file says which end the offset is on"; W5 (#2031, issue #2064) read
  // the converter and it says so in one line —
  //
  //     ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
  //
  // (`conversion/qwen4exp.py`, `set_gguf_parameters`, at llama.cpp #27742 head
  // `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`). The GGUF key is therefore
  // ZERO-based and HF's is one-based, which is exactly the [1] vs [2] the two
  // files show and why the tensors live at `blk.1.*`.
  //
  // WHY THE MAPPING IS NOT COSMETIC. `ParseQwen4ExpParams` resolves the PLE
  // layer set from `ple_layer_ids` and from nothing else. Left unmapped, a
  // GGUF-derived config parsed with an EMPTY PLE set: no PLE layer, no n-gram
  // table, and `number_of_conv_states()` reporting 1 on a model that needs 3 —
  // all without a single refusal, because "this config has no PLE" is a legal
  // state. The GGUF-only spelling stays beside it so a reader of either file
  // can see both conventions at once.
  const std::vector<int64_t> gguf_ple_layers =
      ReqIntArray(gguf, p + "ple.layers");
  text["gguf_ple_layers"] = gguf_ple_layers;
  std::vector<int64_t> hf_ple_layer_ids;
  hf_ple_layer_ids.reserve(gguf_ple_layers.size());
  for (int64_t zero_based : gguf_ple_layers) {
    VT_CHECK(zero_based >= 0 && zero_based < c.num_hidden_layers,
             "qwen4_exp gguf: ple.layers holds " + std::to_string(zero_based) +
                 ", which is outside the zero-based layer range [0, " +
                 std::to_string(c.num_hidden_layers) + ")");
    hf_ple_layer_ids.push_back(zero_based + 1);
  }
  text["ple_layer_ids"] = hf_ple_layer_ids;
  // EOS, likewise: the PLE branch of `ParseQwen4ExpParams` refuses a config
  // that names a PLE layer without one, and it is the id the n-gram hash reads
  // on the first token of every sequence.
  //
  // RECORDED DIVERGENCE, not a silent inheritance. The converter resolves this
  // key as `int(eos[-1])` — the LAST element of the HF list — where
  // `Qwen4ExpTextModel.forward` takes element [0]. We follow the ALGORITHM
  // oracle wherever a config.json is available; here the container is the only
  // source, so this value is what the file says, and the spec's `## Owed`
  // carries the disagreement.
  {
    const GgufValue* eos = gguf.FindKv(p + "ple.eos_token_id");
    if (eos != nullptr) text["eos_token_id"] = KvInt(*eos, "ple.eos_token_id");
  }
  raw["text_config"] = text;
  c.raw = std::move(raw);

  return c;
}

}  // namespace vllm
