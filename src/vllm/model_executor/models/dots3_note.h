// dots3-note (`Dots3NoteForCausalLM`) — W1 config + registry brick.
// Issue #699, spec `.agents/specs/dots3-note.md` (§4 is the load-bearing part).
//
// W1 SCOPE, STATED SO NOBODY HAS TO INFER IT. This TU makes the architecture
// RESOLVE, makes its config PARSE with the four defaults the released
// `config.json` does NOT carry, and makes the on-disk name map accountable
// against the real shard index. It does NOT forward: `Dots3NoteForward` REFUSES
// BY NAME, naming the brick that owes the maths (W3-W10). A forward that
// returned zeros would emit a plausible garbage token, which is the one failure
// mode this row cannot afford — see §6.4 of the spec: this model has NO oracle
// on any hardware we own, so there is no token gate downstream to catch it.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// BEYOND-PIN. `dots3_note` does NOT exist at our parity pin `555967922`
// (0.26.0.dev0); it is on vLLM `main` only, added by
// https://github.com/vllm-project/vllm/pull/51255. Every anchor below was read
// at upstream `origin/main` = `c205726108df54bb6fbf15b19e725a4a3add2b18`
// (2026-08-23), and the deviation is recorded in
// `.agents/porting-inventory.md` §9. Paths are relative to that checkout.
//
//   OURS                        <-  UPSTREAM
//   Dots3NoteParams (the four
//     absent defaults)          <-  `vllm/transformers_utils/configs/dots3_note.py`
//                                   ::Dots3NoteConfig.__init__ (:12-25)
//   Dots3NoteParams::full       <-  `vllm/models/dots3_note/nvidia/model.py`
//                                   ::Dots3NoteFullAttention (:219),
//                                   its __init__ (:222-308)
//   Dots3NoteParams::swa        <-  `vllm/models/dots3_note/nvidia/model.py`
//                                   ::Dots3NoteSlidingAttention (:329),
//                                   its __init__ (:332-460)
//   *_lora_scale                <-  `model.py`::Dots3NoteFullAttention (:303-307)
//                                   and ::Dots3NoteSlidingAttention (:438-443)
//   indexer_rope_is_neox        <-  `vllm/model_executor/models/deepseek_v2.py`
//                                   ::DeepseekV2MLAAttention.__init__ (:1144-1149)
//   physical_latent_row()       <-  `model.py`::Dots3NoteFullAttention (:283)
//                                   + ::Dots3NotePaddedMLAAttention (:204-217)
//   EnumerateDots3NoteTensors   <-  `nvidia/multimodal.py`::Dots3NoteForCausalLM
//                                   .hf_to_vllm_mapper (:53-62) + the released
//                                   `model.safetensors.index.json`
//   Dots3NoteForward            <-  NOT PORTED. Refuses by name (W3-W10).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/mla_attention.h"    // MlaBlockDims / the seam
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Which of the TWO attention geometries a backbone layer runs. `layer_types` in
// the released config.json is 46 entries of exactly these two spellings
// (`full_attention` / `sliding_attention`), consumed by
// `model.py`::Dots3NoteDecoderLayer.__init__ (:504-509).
enum class Dots3NoteLayerKind { kFullAttention, kSlidingAttention };

// One of dots3-note's two MLA geometries. The full layers and the sliding
// layers differ in head count, TWO of the three LoRA ranks, the NoPE width, the
// RoPE theta, the window and whether a DSA indexer exists — so they get one
// struct each rather than a pile of `swa_`-prefixed scalars on the parent.
struct Dots3NoteAttnParams {
  int64_t num_attention_heads = 0;
  int64_t q_lora_rank = 0;
  int64_t kv_lora_rank = 0;
  int64_t qk_nope_head_dim = 0;
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;
  // The full layers carry the model-level `rope_theta` (8e7); the sliding
  // layers carry their OWN `swa_rope_theta` (5e4), model.py:404-408.
  double rope_theta = 0.0;
  // BOTH geometries drive the MLA rope in GPT-J (interleaved, adjacent-pair)
  // layout: the sliding layers pass `is_neox_style=False` explicitly
  // (model.py:404-408) and the full layers inherit the same value from
  // `deepseek_v2.py`::DeepseekV2MLAAttention (:1093-1098). See §4 of the spec.
  //
  // THE DEFAULT IS DELIBERATELY THE WRONG VALUE (review finding F9). Neither
  // geometry is ever NeoX, so a default of `false` would equal the production
  // value and deleting the two resolution assignments in
  // `ParseDots3NoteParams` would leave every layout assertion green — a field
  // no gate can prove is being set. `true` here means an unresolved field
  // fails loudly instead.
  bool rope_is_neox_style = true;
  // 0 == full attention. 513 on the sliding layers (`sliding_window_size`,
  // passed to MLAAttention at model.py:456).
  int64_t sliding_window = 0;
  // "headwise" on both classes in the released checkpoint. The non-headwise
  // branch multiplies the whole [num_heads*v_head_dim] output by one gate
  // instead of one gate per head (model.py:191-201).
  std::string attention_gate_type;
  // Only the full layers run the DSA lightning indexer; the sliding layers set
  // `self.indexer = None` / `is_sparse = False` (model.py:430-432).
  bool has_indexer = false;
  // `apply_mla_qkv_lora_rescale` => sqrt(hidden_size / rank), applied AFTER the
  // respective layernorm (model.py:154, :160). 1.0 when the flag is false.
  double q_lora_scale = 1.0;
  double kv_lora_scale = 1.0;

  int64_t qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // The LOGICAL MLA latent row this geometry reads: 576 full, 1088 sliding.
  int64_t latent_row() const { return kv_lora_rank + qk_rope_head_dim; }
};

// Every dots3-note config field W1 resolves, read ONCE from the HfConfig. Most
// keys are not typed on HfConfig, so they come from `config.raw`; the four §4
// defaults are not in the released `config.json` at all and come from
// `Dots3NoteConfig.__init__`.
struct Dots3NoteParams {
  // --- shared geometry (released `dots-studio/dots3-note-prev`) ---
  int64_t hidden_size = 0;              // 5120
  int64_t num_hidden_layers = 0;        // 46
  int64_t vocab_size = 0;               // 152064
  int64_t intermediate_size = 0;        // 13824 (the dense layers' MLP)
  double rms_norm_eps = 1e-5;
  int64_t max_position_embeddings = 0;  // 524288
  bool tie_word_embeddings = false;

  // 46 entries, 13 full / 33 sliding.
  std::vector<Dots3NoteLayerKind> layer_types;

  // --- MoE (`model.py`::Dots3NoteMoE :76, DeepseekV2MoE's router) ---
  int64_t n_routed_experts = 0;      // 256
  int64_t num_experts_per_tok = 0;   // 8
  int64_t moe_intermediate_size = 0; // 1536
  int64_t n_shared_experts = 0;      // 1
  int64_t first_k_dense_replace = 0; // 1
  int64_t moe_layer_freq = 1;
  bool norm_topk_prob = true;
  double routed_scaling_factor = 1.0;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "noaux_tc";
  // ★ §4 TRAP 1 (`n_group` AND `topk_group`, which §4 states as ONE item)
  // — ABSENT from the released config.json.
  // `Dots3NoteConfig.__init__` sets BOTH to 1 rather than inheriting
  // DeepseekV3Config's 8 / 4 (transformers `configuration_deepseek_v3.py`
  // :168-169). Upstream's own comment: "Do not inherit DeepSeek-V3's
  // 8-group/4-group router defaults: Note was trained with an ungrouped (1/1)
  // noaux_tc router ... A different grouping changes the selected experts at
  // every MoE layer." Numerically silent when wrong — the model still emits
  // plausible text, and this row has no oracle to catch it (spec §6.4).
  int64_t n_group = 0;
  int64_t topk_group = 0;

  // --- DSA lightning indexer (full layers only) ---
  int64_t index_n_heads = 0;   // 64
  int64_t index_head_dim = 0;  // 128
  int64_t index_topk = 0;      // 2048
  // ★ §4 TRAP 2 — ABSENT from the released config.json, defaulted True by
  // `Dots3NoteConfig.__init__` (:22). The value is consumed as
  // `is_neox_style = not indexer_rope_interleave`
  // (`deepseek_v2.py`::DeepseekV2MLAAttention :1148), so True means the indexer
  // RoPE rotates ADJACENT (GPT-J) pairs. DeepSeek-V3.2 — and therefore the
  // indexer we already ported — leaves the flag absent and gets split-half
  // NeoX. Wrong value => a different set of learned coordinates is rotated,
  // with no shape change and no error.
  bool indexer_rope_interleave = false;
  // Derived, and the form the layer actually consumes.
  bool indexer_rope_is_neox_style() const { return !indexer_rope_interleave; }

  // ★ §4 TRAP 3 — ABSENT from the released config.json, defaulted 1 by
  // `Dots3NoteConfig.__init__` (:24). DeepseekV3Config has no such field at
  // all, so an absent key would read 0 and the whole nextn tail would be
  // silently unclaimed. The released shard index carries EXACTLY ONE nextn
  // layer (`model.layers.46.*`, 18 tensors), which is the checkpoint agreeing.
  int64_t num_nextn_predict_layers = 0;

  // ★ §4 TRAP 5 — PRESENT in the config.json (true), but it differs from what
  // our DeepSeek MLA assumes: it scales q_c and kv_c_normed by
  // sqrt(hidden_size / rank) AFTER their layernorms (model.py:154, :160). The
  // resolved scalars live on `full`/`swa` below.
  bool apply_mla_qkv_lora_rescale = false;

  // ★ §4 TRAP 6 lives on these two: the sliding geometry's own RoPE theta
  // (5e4 vs the full layers' 8e7) and the `is_neox_style=False` both share.
  Dots3NoteAttnParams full;
  Dots3NoteAttnParams swa;

  // The PHYSICAL MLA cache row both classes must share. Upstream pads the full
  // layers' 576-wide logical row up to the sliding layers' 1088
  // (`model.py`::Dots3NoteFullAttention :283 passes
  // `physical_head_size=swa_kv_lora_rank + swa_qk_rope_head_dim` into
  // `Dots3NotePaddedMLAAttention`, whose `get_kv_cache_spec` reports it :211).
  int64_t physical_latent_row() const { return swa.latent_row(); }

  Dots3NoteLayerKind kind_of(int64_t layer) const {
    return layer_types.at(static_cast<size_t>(layer));
  }
  bool is_moe_layer(int64_t layer) const {
    return n_routed_experts > 0 && layer >= first_k_dense_replace &&
           moe_layer_freq > 0 && layer % moe_layer_freq == 0;
  }
};

// Resolve + validate. Throws (std::runtime_error via VT_CHECK) naming the key
// on anything this port cannot represent.
Dots3NoteParams ParseDots3NoteParams(const HfConfig& config);

// The registry's `parse_config` hook: the resolve IS the validation.
void ParseDots3NoteConfig(const HfConfig& config);

// One on-disk tensor and the named consumer that will read it.
struct Dots3NoteTensor {
  std::string name;      // exactly as the checkpoint ships it
  std::string consumer;  // stable tag: what will read it
};

// The on-disk weight name map of the LANGUAGE tower, by the names the
// checkpoint SHIPS — `q_a_proj` and `kv_a_proj_with_mqa` SEPARATE (upstream
// fuses them into `fused_qkv_a_proj` at load time via
// `DeepSeekV2FusedQkvAProjLinear`, so the module-level view hides the two real
// tensors), routed experts unstacked, `model.` prefix. Ordered: root, backbone
// layers ascending, then the nextn tail.
//
// `layers` selects WHICH backbone layers to enumerate. Pass every backbone
// layer for a production load; a slice is for a focused test. This function
// claims the LANGUAGE tower only — the two tower files are named deferrals,
// see `Dots3NoteDeferredTowers()` below.
std::vector<Dots3NoteTensor> EnumerateDots3NoteTensors(
    const Dots3NoteParams& params, const std::vector<int64_t>& layers,
    bool include_root, bool include_nextn);

// Convenience: every backbone layer, the root tensors and the nextn tail. Over
// the released checkpoint this is exactly 35381 names.
std::vector<Dots3NoteTensor> EnumerateDots3NoteTensors(
    const Dots3NoteParams& params);

// A tower the released checkpoint ships that this port does NOT load yet, and
// the brick that owes it. This table is the difference between a DEFERRAL and a
// SILENT DROP: a tensor matched here is refused a language-tower consumer on
// purpose, by a record that names the brick, the file it ships in and what it
// is. A tensor matched by nothing lands in `unaccounted` and refuses the load.
//
// The counters alone could not carry that meaning. Folding the towers into the
// language count leaves every "100% accounted" assertion green while 2625
// weights go unloaded — the exact mutation the W1 review found unguarded
// (#1805, M15), and W2 is the scale at which it would have mattered.
struct Dots3NoteDeferredTower {
  const char* prefix;  // the on-disk name prefix, e.g. "vision_encoder."
  const char* file;    // the ONE shard file every one of them ships in
  const char* brick;   // the phase of `.agents/specs/dots3-note.md` §7 that owes it
  const char* what;    // what it is, for the message a reader gets
};

// The complete deferral table, in the order a report should print it.
const std::vector<Dots3NoteDeferredTower>& Dots3NoteDeferredTowers();

// The deferral that claims `name`, or nullptr when no deferral does. A nullptr
// for a name the language map does not claim either is an UNACCOUNTED tensor.
const Dots3NoteDeferredTower* Dots3NoteDeferralFor(const std::string& name);

// What an accounting pass over a checkpoint's tensor NAMES found. Every name on
// disk lands in exactly one bucket, and `unaccounted` must be empty: a tensor
// nobody claims is a silently dropped weight, which reads as zeros and renders.
// The two tower buckets are NAMED deferrals (W6 vision, W7 audio), not silence.
//
// Over the whole released `dots-studio/dots3-note-prev` index the three buckets
// are 35381 / 2195 / 430 = 38006. Assert them BY NUMBER: "nothing was left
// over" is also true of a classifier that claims the towers as language.
struct Dots3NoteAccounting {
  int64_t language = 0;  // claimed by a named language-tower consumer
  int64_t vision = 0;    // `vision_encoder.*`, deferred to W6
  int64_t audio = 0;     // `audio_encoder.*`, deferred to W7
  std::vector<std::string> unaccounted;  // on disk, claimed by nobody
  std::vector<std::string> missing;      // enumerated, not on disk
  std::vector<std::string> duplicated;   // enumerated more than once
  int64_t total() const { return language + vision + audio; }
};

// Classify every name in `present` against what `params` says the language
// tower ships. `expected_layers` is the backbone layer set to enumerate; pass
// every layer for a production load, or a slice for the W1 gate.
Dots3NoteAccounting AccountDots3NoteTensors(
    const Dots3NoteParams& params, const std::vector<std::string>& present,
    const std::vector<int64_t>& expected_layers);

// The W1 loaded-model payload. It carries the resolved params and the
// accounting result, and NOTHING else: W2 owns the materialization, so
// `materialized` is false on every object this brick can produce and the
// forward refuses on it BY NAME. It is a real, complete object of the type the
// factory returns — not a look-alike — so a test reaches the refusal without
// downcasting a fabricated stub, which is undefined behaviour and was exactly
// the defect UBSan caught on the NemotronH row (#730/#784, #775).
// ─── W4a: the FULL-attention layer on the decode path ────────────────────────
//
// The DEVICE-resident weights of ONE full-attention layer, in the shapes the
// shared MLA seam consumes. Names on the left are `mla::MlaBlockWeights`
// fields; the checkpoint tensors behind them are named in
// `dots3_note_device.cpp` beside the load.
//
// The DSA indexer's five tensors are DELIBERATELY ABSENT. They are accounted
// for by name at load (`EnumerateDots3NoteTensors`), and the forward refuses by
// name for any sequence long enough for the top-k to prune — see
// `Dots3NoteDeviceRefusal`. Materializing weights nothing reads is how a port
// grows a silent unused arm.
struct Dots3NoteMlaLayerWeights {
  OwnedTensor fused_qkv_a_proj;  // {q_a_proj, kv_a_proj_with_mqa} merged, raw-NK
  OwnedTensor q_a_layernorm;     // [q_lora_rank]
  OwnedTensor q_b_proj;          // [heads*qk_head_dim, q_lora_rank] raw-NK
  OwnedTensor kv_a_layernorm;    // [kv_lora_rank]
  OwnedTensor kv_b_proj;         // [heads*(qk_nope+v), kv_lora_rank] raw-NK
  OwnedTensor w_uk_t;            // absorbed [heads, qk_nope, kv_lora_rank]
  OwnedTensor w_uv;              // absorbed [heads, kv_lora_rank, v]
  OwnedTensor o_proj;            // [hidden, heads*v] raw-NK
  // dots3-note ONLY (model.py:299-301 / :292-298).
  OwnedTensor k_rope_only_layernorm;  // [qk_rope_head_dim]
  OwnedTensor g_proj;                 // [heads, hidden] raw-NK
};

// `Dots3NoteMLP` — the DENSE (pre-`first_k_dense_replace`) MLP, gate/up merged
// for the shared `layers::MlpGateUpMethodBase` seam.
struct Dots3NoteDenseMlp {
  OwnedTensor gate_up_proj;  // [2*intermediate, hidden] raw-NK
  OwnedTensor down_proj;     // [hidden, intermediate] raw-NK
};

struct Dots3NoteLayerDeviceWeights {
  OwnedTensor input_layernorm;           // [hidden]
  OwnedTensor post_attention_layernorm;  // [hidden]
  Dots3NoteMlaLayerWeights attn;
  Dots3NoteDenseMlp mlp;
};

// The materialized language tower, present ONLY for a config the device forward
// can actually run end to end (see `Dots3NoteDeviceRefusal`). `present == false`
// is the normal state for the released checkpoint and is not an error: W1/W2's
// accounting still runs, and the forward refuses BY NAME.
struct Dots3NoteDeviceWeights {
  bool present = false;
  mla::MlaBlockDims mla{};
  OwnedTensor embed_tokens;       // [vocab, hidden] (embed lookup; NOT transposed)
  OwnedTensor final_norm;         // [hidden]
  OwnedTensor lm_head;            // [hidden, vocab] Matmul-B; EMPTY when tied
  OwnedTensor rope_cos_sin_cache;  // [max_position_embeddings, qk_rope_head_dim]
  std::vector<Dots3NoteLayerDeviceWeights> layers;
};

// The W1 loaded-model payload. It carries the resolved params and the
// accounting result, and NOTHING else: W2 owns the materialization, so
// `materialized` is false on every object this brick can produce and the
// forward refuses on it BY NAME. It is a real, complete object of the type the
// factory returns — not a look-alike — so a test reaches the refusal without
// downcasting a fabricated stub, which is undefined behaviour and was exactly
// the defect UBSan caught on the NemotronH row (#730/#784, #775).
//
// W4a adds `device`: for a config whose every layer is FULL attention with a
// DENSE MLP the loader also materializes the tower, and `materialized` becomes
// true. For anything else — the released checkpoint included — nothing changes.
struct Dots3NoteWeights {
  Dots3NoteParams params;
  Dots3NoteAccounting accounting;
  bool materialized = false;
  Dots3NoteDeviceWeights device;
};

// Why the DEVICE forward cannot run `params`, or "" when it can. The message
// names ONE thing — the first unrepresentable feature in brick order — and the
// brick that owes it, so a reader is told what to build rather than that
// something is missing.
//
// W4a covers exactly one shape: every layer `full_attention` with a DENSE MLP,
// and no sequence long enough for the DSA top-k to prune. Everything else is
// W4b (sliding window), W5 (MoE) or a later brick.
std::string Dots3NoteDeviceRefusal(const Dots3NoteParams& params);

// The longest sequence the full-attention device path may serve. Upstream's
// full layers are SPARSE: the lightning indexer picks `index_topk` keys per
// query (model.py:171 -> deepseek_v2.py::Indexer). The shared MLA seam computes
// DENSE attention, and at `context + query <= index_topk` the top-k selects
// EVERY causal candidate, so the two answers coincide exactly. Past that they
// do not, and W3's gate measured what the difference costs: a wrong selection
// moved the layer output by 0.39. The forward therefore refuses BY NAME rather
// than quietly serving dense attention on a sparse model.
int64_t Dots3NoteDenseEquivalentMaxSeqLen(const Dots3NoteParams& params);

// Read every tensor the full-attention language tower needs out of `shards` and
// build the device-side forms (the fused A-projection, the absorbed kv_b_proj,
// the rope cache). REFUSES BY NAME on the first tensor whose shape disagrees
// with the config. Only called when `Dots3NoteDeviceRefusal` is empty.
Dots3NoteDeviceWeights MaterializeDots3NoteDevice(
    const std::vector<SafetensorsFile>& shards, const Dots3NoteParams& params);

// The `mla::MlaBlockDims` a dots3-note FULL-attention layer runs, including the
// two §4-trap-5 LoRA rescales. Exported so the gate can drive the same struct
// the forward builds instead of typing one by hand.
mla::MlaBlockDims Dots3NoteFullAttnMlaDims(const Dots3NoteParams& params);

// W1 loader: resolves the config, accounts for 100% of the checkpoint's tensors
// (refusing BY NAME on the first unclaimed or missing one), and returns an
// UNMATERIALIZED model. It never returns a half-built tower and never silently
// drops a weight.
Dots3NoteWeights LoadDots3NoteWeights(const std::vector<SafetensorsFile>& shards,
                                      const HfConfig& config);

// The language tower's decode entry point. **DEFINED IN
// `dots3_note_device.cpp` since W4a**, not in `dots3_note.cpp`, because it is
// no longer a refusal: it decodes a config whose every layer is
// `full_attention` with a DENSE MLP, and refuses everything else BY NAME with
// the brick that owes it. The signature is unchanged from the one W1 wrote to
// match `KimiK3Model::ForwardDevice`.
//
// The classification is earned by the BODY, not by anything written here.
// `scripts/check-runner-routing-consistency.py` reads that body: W1's
// `VT_CHECK(false, ...)` made it REFUSE, and W4a's `WrapDeviceLogits` makes it
// DEVICE — the device-resident-logits seam, which is where a model with a real
// forward belongs. Neither state is the silently-exempt NONE bucket, and the
// transition between them is a fact about the body rather than about this
// comment.
//
// Deliberately no `[[noreturn]]`, and it now matters less because the function
// returns. `ForwardLogits` is not `void`, MSVC answers
// that with C4646, and `/W4 /WX` turns the warning into C2220 -- which failed the
// entire Windows compile of `main` while every POSIX lane stayed green (#1829).
// KimiK3 never carried the attribute either. `check-windows-portability.py`
// refuses the shape now, so this cannot come back unseen.
class Dots3NoteModel {
 public:
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Dots3NoteWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices);
};

// The MLA KV topology. Both classes share ONE physical row of
// `swa_kv_lora_rank + swa_qk_rope_head_dim` = 1088 (see
// `physical_latent_row()`); the full layers read their logical 576 out of it.
v1::KVCacheConfig MakeDots3NoteKVCache(const HfConfig& config, int block_size,
                                       int num_blocks);

}  // namespace vllm
