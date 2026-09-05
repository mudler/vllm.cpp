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

#include "vllm/model_executor/models/dots3_note_audio.h"
#include "vllm/model_executor/models/dots3_note_vision.h"  // W6a: the DENSE ViT arm
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

  // The `quantization_config` block. ABSENT from the released bf16
  // `dots-studio/dots3-note-prev` (verified against the committed fixture) and
  // PRESENT on the `-fp8` sibling as `{"quant_method": "fp8", "fmt": "e4m3",
  // "activation_scheme": "dynamic", "weight_block_size": [128, 128]}`.
  //
  // Upstream reads exactly this: `_padded_mlp_size` (model.py:63-73) takes
  // `getattr(quant_config, "weight_block_size", None)`, and
  // `Dots3NoteModel._pad_dense_mlp_weight` (model.py:598-618) does the same.
  // We read it to REFUSE by name rather than to pad: the fp8 arm is W9, and
  // `dense_loaders::MaterializeBf16Source` handles a per-tensor or
  // per-output-ROW `<name>_scale` and nothing else, so a blockwise
  // `weight_scale_inv` would otherwise surface as a bare "tensor not found".
  std::string quant_method;                // "" when the block is absent
  std::vector<int64_t> weight_block_size;  // empty when the key is absent
  bool has_blockwise_quant() const { return !weight_block_size.empty(); }

  // `intermediate_size=config.moe_intermediate_size * num_shared_experts`
  // (model.py:103-107, through `_padded_mlp_size`, which is the identity at
  // TP=1 with no `weight_block_size`). NOT `intermediate_size`: the released
  // checkpoint's `mlp.shared_experts.gate_proj.weight` is [1536, 5120] and the
  // dense layers' `mlp.gate_proj.weight` is [13824, 5120], so reading the wrong
  // one is an 9x-too-wide MLP that the shape check refuses BY NAME.
  int64_t shared_intermediate_size() const {
    return moe_intermediate_size * n_shared_experts;
  }

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
  // The nextn (MTP) tail: `model.layers.{num_hidden_layers + i}.*` and
  // `model.mtp.*`, deferred to W10 (W5c, #2176). It is a SEPARATE bucket rather
  // than part of `language` for the reason the tower buckets exist: the
  // language forward never reads these tensors, and folding them into the
  // language count leaves "100% accounted" green over 19 weights nobody loads.
  // They stay ENUMERATED, so `missing` still refuses a checkpoint that claims a
  // nextn layer and does not ship it — only the bucket moved.
  int64_t nextn = 0;
  int64_t vision = 0;    // `vision_encoder.*`, deferred to W6
  int64_t audio = 0;     // `audio_encoder.*`, deferred to W7
  std::vector<std::string> unaccounted;  // on disk, claimed by nobody
  std::vector<std::string> missing;      // enumerated, not on disk
  std::vector<std::string> duplicated;   // enumerated more than once
  int64_t total() const { return language + nextn + vision + audio; }
};

// Is `name` part of the nextn (MTP) tail this port defers to W10? (W5c, #2176.)
//
// A PREDICATE rather than a `Dots3NoteDeferredTowers()` row, and the reason is
// the data rather than a preference: a tower's prefix is a literal, while the
// nextn tail's is CONFIG-DERIVED — `model.layers.{num_hidden_layers + i}.` for
// `i` in `[0, num_nextn_predict_layers)`, plus the flat `model.mtp.`. A static
// table cannot spell the first one.
//
// Upstream skips exactly these names when it loads the MAIN model:
// `get_spec_layer_idx_from_weight_name` (`utils.py:542`, matching
// `model.layers.{base+i}.` or `layers.{base+i}.` at `:559`), consulted at
// `deepseek_v2.py:1618-1620`; and `if name.startswith("mtp."): continue` inside
// `Dots3NoteModel._adapt_weights` (`model.py:624`). Both at `bc2d63e650`.
bool Dots3NoteIsNextnTensor(const Dots3NoteParams& params, const std::string& name);

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
  // The DSA "Lightning Indexer" (W4b-3c, #699), `Indexer.__init__`
  // deepseek_v2.py:691-708 @ `bc2d63e650`. FULL-attention layers only: upstream
  // gives a sliding layer no indexer at all (`self.indexer = None` /
  // `is_sparse = False`, model.py:432-434), so these stay EMPTY on the 33
  // sliding layers and `MlaBlockDims::has_indexer()` is false for them.
  OwnedTensor indexer_wq_b;           // [index_n_heads*index_head_dim, q_lora_rank] raw-NK
  OwnedTensor indexer_wk;             // [index_head_dim, hidden] raw-NK
  OwnedTensor indexer_weights_proj;   // [index_n_heads, hidden] raw-NK
  OwnedTensor indexer_k_norm_weight;  // [index_head_dim]
  OwnedTensor indexer_k_norm_bias;    // [index_head_dim]  — a LayerNorm, not an RmsNorm
};

// `Dots3NoteMLP` — the DENSE (pre-`first_k_dense_replace`) MLP, gate/up merged
// for the shared `layers::MlpGateUpMethodBase` seam.
struct Dots3NoteDenseMlp {
  OwnedTensor gate_up_proj;  // [2*intermediate, hidden] raw-NK
  OwnedTensor down_proj;     // [hidden, intermediate] raw-NK
};

// `Dots3NoteMoE` (model.py:76) — `DeepseekV2MoE`'s router and routed experts
// with the shared expert LIFTED OUT of the base and added unfused
// (model.py:87-99 sets `n_shared_experts` to None on the routed config, so
// `DeepseekV2MoE.__init__` takes its `shared_experts = None` branch at
// deepseek_v2.py:354-355; model.py:125-127 does the add). Numerically that is
// the same function the base computes when it owns the shared expert, which is
// why `vt::MoeCombine`'s optional `shared` term expresses it exactly.
//
// The released `dots-studio/dots3-note-prev` ships every one of these BF16
// except `e_score_correction_bias`, which is F32 — the ONLY dtype exception in
// the MoE block, and F32 upstream too (deepseek_v2.py:322-324).
struct Dots3NoteMoeWeights {
  OwnedTensor router_gate;  // [H, E] Matmul-B (from the [E, H] `mlp.gate.weight`)
  // `e_score_correction_bias` [E] F32 — built only for `topk_method ==
  // "noaux_tc"` (deepseek_v2.py:321-326), which `ParseDots3NoteParams` already
  // requires for this architecture, so it is never empty here.
  OwnedTensor e_score_correction_bias;
  std::vector<OwnedTensor> expert_gate;  // E x [H, moe_intermediate_size] Matmul-B
  std::vector<OwnedTensor> expert_up;    // E x [H, moe_intermediate_size] Matmul-B
  std::vector<OwnedTensor> expert_down;  // E x [moe_intermediate_size, H] Matmul-B
  // The ONE shared expert, at `moe_intermediate_size * n_shared_experts`
  // (`Dots3NoteParams::shared_intermediate_size`), gate/up merged for the
  // shared `layers::MlpGateUpMethodBase` seam exactly as the dense MLP is.
  Dots3NoteDenseMlp shared;
  // The device-resident per-expert pointer arrays the grouped bf16 MoE GEMM
  // reads, built ONCE on first device-MoE use and owned BY THIS BLOCK (issue
  // #237's `ResidentSlot`, qwen3_5_weights.h, which this header already
  // includes). Opaque here on purpose: the resident type is a CUDA
  // implementation detail of `dots3_note_device.cpp`, exactly as
  // `MoeBlockWeights` keeps qwen3_5.cpp's out of its own header.
  //
  // KEYED ON THE SLOT, NEVER ON AN ADDRESS. W5 first shipped this state as a
  // process-lifetime `static std::map<const Dots3NoteMoeWeights*, ...>`. An
  // address is only a valid identity while the object lives: destroy one engine
  // and build another in the same process, and the allocator can hand the new
  // weights the old ones' address. The new block then finds an entry already
  // marked ready and every routed expert GEMM reads the PREVIOUS engine's
  // device pointers. Because those buffers are deliberately never freed, there
  // is no crash and no error — the second model silently answers from the
  // first's experts. That is issue #237, and `tests/vllm/models/
  // test_moe_resident_lifetime.cpp` pins the invariant for this block too.
  ResidentSlot resident_moe;
};

struct Dots3NoteLayerDeviceWeights {
  // WHICH of the two attention geometries this layer runs — `config.layer_types
  // [layer_idx] == "sliding_attention"` selects `Dots3NoteSlidingAttention`
  // over `Dots3NoteFullAttention` upstream (`model.py:501-505` @
  // `bc2d63e650`). Every tensor in `attn` is shaped by this, so it is stored
  // beside them rather than re-read from the params at each use (W4b-2, #699).
  Dots3NoteLayerKind kind = Dots3NoteLayerKind::kFullAttention;
  OwnedTensor input_layernorm;           // [hidden]
  OwnedTensor post_attention_layernorm;  // [hidden]
  Dots3NoteMlaLayerWeights attn;
  // WHICH MLP this layer runs — `is_moe` at model.py:514-519, i.e.
  // `layer_idx < num_hidden_layers and n_routed_experts is not None and
  // layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0`,
  // which is `Dots3NoteParams::is_moe_layer`. Stored beside the weights rather
  // than re-derived at each use, for the same reason `kind` is (W4b-2): on a
  // MoE layer `mlp.gate_proj.weight` DOES NOT EXIST on disk, so the choice is
  // structural and not a preference. (W5, #699.)
  bool is_moe = false;
  Dots3NoteDenseMlp mlp;      // populated iff !is_moe
  Dots3NoteMoeWeights moe;    // populated iff  is_moe
};

// The materialized language tower, present ONLY for a config the device forward
// can actually run end to end (see `Dots3NoteDeviceRefusal`). `present == false`
// is the normal state for the released checkpoint and is not an error: W1/W2's
// accounting still runs, and the forward refuses BY NAME.
struct Dots3NoteDeviceWeights {
  bool present = false;
  // The FULL-attention geometry (13 of 46 layers).
  mla::MlaBlockDims mla{};
  // The SLIDING geometry (33 of 46 layers) — a DIFFERENT head count, latent
  // rank, NoPE width, rope theta and softmax scale, plus the 513 window. Not a
  // parameterisation of the one above; see the table in
  // `.agents/specs/dots3-note.md` §4.7. `sliding_window == 0` here means the
  // config has no sliding layer and the struct is unused. (W4b-2, #699.)
  mla::MlaBlockDims swa_mla{};
  OwnedTensor embed_tokens;       // [vocab, hidden] (embed lookup; NOT transposed)
  OwnedTensor final_norm;         // [hidden]
  OwnedTensor lm_head;            // [hidden, vocab] Matmul-B; EMPTY when tied
  // TWO rope caches, because the two geometries carry DIFFERENT thetas — 8e7
  // on the full layers and `swa_rope_theta` 5e4 on the sliding ones
  // (`model.py:401-409` @ `bc2d63e650`). Sharing one would be numerically
  // silent, which is why they are separate tensors and not one with a flag.
  // Each is EMPTY when no layer of that kind exists.
  OwnedTensor rope_cos_sin_cache;      // [max_position_embeddings, qk_rope_head_dim]
  OwnedTensor swa_rope_cos_sin_cache;  // [max_position_embeddings, swa_qk_rope_head_dim]
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

  // W6a (#2512) — the DENSE vision tower. `vision.present` is TRUE only when
  // the config HAS a `vision_config` AND `Dots3NoteVisionRefusal` accepted it,
  // which for the RELEASED `dots-studio/dots3-note-prev` it does NOT: 17 of its
  // 42 vision blocks are pyramid MoE and belong to W6b. `vision_refusal` keeps
  // that message so the encoder hook can report WHY rather than only that the
  // tower is absent — a refusal a caller can only see as a missing pointer is
  // one nobody can act on.
  //
  // The two are kept BESIDE the language tower rather than folded into
  // `device`, because they have different lifetimes: a config can have a
  // runnable language tower and a refused vision tower (that is the released
  // checkpoint) and the reverse is a defect, not a state.
  Dots3NoteVisionParams vision_params;
  Dots3NoteVisionWeights vision;
  std::string vision_refusal;

  // W7a (#2703) — the AUDIO tower, on exactly the same polarity and for the
  // same reason. `audio.present` is TRUE only when the config HAS an
  // `audio_config` AND `Dots3NoteAudioRefusal` accepted it, which for the
  // RELEASED `dots-studio/dots3-note-prev` it DOES: all 430
  // `audio_encoder.*` tensors are BF16 and every `audio_config` key it sets is
  // an arm W7a computes. `audio_refusal` keeps the message for the encoder
  // hook to report verbatim.
  Dots3NoteAudioParams audio_params;
  Dots3NoteAudioWeights audio;
  std::string audio_refusal;
};

// Why the DEVICE forward cannot run `params`, or "" when it can. The message
// names ONE thing — the first unrepresentable feature in brick order — and the
// brick that owes it, so a reader is told what to build rather than that
// something is missing.
//
// W4a/W4b cover both attention geometries — full and sliding-window — with a
// DENSE MLP, and since W4b-3c a long SINGLE-SHOT prefill is served with the DSA
// selection rather than refused. **W5 added the MoE layer and W5c removed the
// nextn branch, so this function now returns "" for the RELEASED
// `dots-studio/dots3-note-prev` config.** What it still refuses is a
// BLOCKWISE-QUANTIZED checkpoint, which is W9: the `-fp8` sibling carries
// `quantization_config.weight_block_size = [128, 128]` and a `weight_scale_inv`
// per projection, while `dense_loaders::MaterializeBf16Source` reads a
// per-tensor or per-output-ROW `<name>_scale` and would otherwise fail with a
// bare "tensor not found".
//
// The vision and audio towers (W6/W7) and the nextn tail (W10) are NAMED
// DEFERRALS in the accounting rather than refusals here — the towers through
// `Dots3NoteDeferredTowers()`, the nextn tail through
// `Dots3NoteIsNextnTensor`. The nextn branch used to refuse, which was
// STRICTER than upstream: vLLM drops those weights from the main model
// (utils.py:542 -> deepseek_v2.py:1618-1620; model.py:624). See #2176.
//
// NOTE what this function does NOT decide. The `seq_len > index_topk` question
// is a property of the STEP, not of the config, so it lives in the forward
// (`Dots3NoteModel::ForwardDevice`) and not here: a request with cached context
// past `index_topk` needs the indexer's own key cache, which is
// `KV-DSV4-MULTICACHE`'s (#1925).
std::string Dots3NoteDeviceRefusal(const Dots3NoteParams& params);

// ─── THE GGUF SIDE OF THIS ARCHITECTURE (W9a, #2882) ─────────────────────────
// The architecture string llama.cpp writes into `general.architecture` for this
// model: `LLM_ARCH_DOTS3NOTE -> "dots3note"`, `ggml-org/llama.cpp`
// `src/llama-arch.cpp:114` at `master`. It arrived in that tree on 2026-08-21
// (`5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c`, "model: add dots3-note", #27060,
// with the converter at `conversion/dots3.py`), and its vision+audio half on
// 2026-08-22 (`54ee5ee643f29abba6852903ddfdb688c2361b5b`, "mtmd: support
// dots3-note vision+audio", #27524). This header owns the key, not the
// entrypoint — exactly as `nemotron_h.h` owns `kNemotronHGgufArch`.
//
// It is the LLAMA.CPP spelling and not ours, deliberately: what this constant
// has to match is the byte string in a file a user passes, and nothing else
// produces one.
inline constexpr const char* kDots3NoteGgufArch = "dots3note";

// True when this file's `general.architecture` is dots3-note's.
//
// WHY THIS EXISTS AT ALL. The registry factory has refused `Kind::kGguf` by
// name since W1, and no real artifact could ever reach it:
// `src/vllm/entrypoints/model_loader.cpp::FromModelDir` resolves a GGUF's
// config through `src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch`
// long before the same function builds a `ModelSource` through
// `ModelSource::FromGguf`, so a `dots3note` file died at the build-level
// default naming neither this model, nor the row, nor the brick.
//
// Those two citations are SYMBOLS and not lines on purpose, and the first draft
// of this comment got it wrong in the way #1143 describes: it wrote
// `model_loader.cpp:2668` and `:3069`, and the four lines this very slice
// inserts at `:1289` pushed both call sites down to 2672 and 3073, so the
// anchors were stale at their own merge commit and landed a reader in an
// unrelated row's prose. `scripts/check-symbol-anchors.py` could not have
// caught it -- its own docstring says it does not verify LINE citations.
// This predicate is what the dispatch tests to route it to the message below,
// and it is `IsNemotronHGguf`'s shape for the same reason (#809, #2882).
bool IsDots3NoteGguf(const GgufFile& gguf);

// The ONE owner of the W9 GGUF refusal text. Both doors a GGUF can arrive at —
// the entrypoint's architecture dispatch and `LoadDots3NoteForCausalLM`'s
// source-kind guard — throw THIS string, so whichever one a user reaches, the
// message names this model, this arm, this row and the spec section that owes
// it. A second spelling of the same refusal is how the two drift.
std::string Dots3NoteGgufRefusal();

// The longest sequence for which DENSE attention IS upstream's answer, i.e.
// `index_topk`. Upstream's full layers are SPARSE: the lightning indexer picks
// `index_topk` keys per query (model.py:171 -> deepseek_v2.py::Indexer), and at
// `context + query <= index_topk` the top-k selects EVERY causal candidate, so
// the sparse and dense answers coincide exactly. Past that they do not, and
// W3's gate measured what the difference costs: a wrong selection moved the
// layer output by 0.39.
//
// THIS IS NO LONGER A CEILING ON WHAT THE PATH SERVES, and the name predates
// that (W4b-3c, #699). It is the threshold upstream's own metadata builder
// turns on — `use_dense_mha = prefill_max_seq_len <= self.topk_tokens`
// (`vllm/model_executor/layers/attention/sparse_mla_attention.py:296-299` @
// `bc2d63e650`), consumed at `mla_attention.py:829-851`. At or below it the
// step keeps the dense MHA prefill; above it `BuildDots3NoteSparseStep`
// promotes the whole step to per-token MQA with the DSA selection. What the
// forward still refuses BY NAME is a request with CACHED CONTEXT past this
// bound, because the indexer's key for a token comes from that token's own
// hidden state and a resumed step has none for its context — that needs the
// indexer's own KV cache, which is `KV-DSV4-MULTICACHE` (#1925).
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

// The `mla::MlaBlockDims` a dots3-note SLIDING layer runs
// (`model.py`::Dots3NoteSlidingAttention.__init__ :341-460 @ `bc2d63e650`):
// `swa_*` geometry throughout, the softmax scale is a PLAIN `qk_head_dim**-0.5`
// with no YaRN and no mscale (`:446`), the rope is `rope_type="default"` at
// `swa_rope_theta` (`:401-409`), and `sliding_window` carries
// `config.sliding_window_size` (`:457`). Exported for the same reason the full
// one is: the gate drives the struct the forward builds, never one it typed.
mla::MlaBlockDims Dots3NoteSlidingAttnMlaDims(const Dots3NoteParams& params);

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
  // `mm` is `ModelForwardInput::mm` (W6a, #2512). When it carries
  // `inputs_embeds` the forward SKIPS the embedding lookup and starts from
  // those rows, which is upstream's `inputs_embeds is not None` arm
  // (`deepseek_v2.py`'s `DeepseekV2Model.forward`, mirrored for dots3 at
  // `nvidia/model.py`) and the ONLY way the vision rows the runner merged can
  // reach the residual stream. NULL for every text step, so the text path is
  // byte-identical.
  //
  // dots3-note is NOT an M-RoPE model — upstream's `Dots3NoteForCausalLM` is
  // `SupportsMultiModal, SupportsPP` and not `SupportsMRoPE`
  // (`nvidia/multimodal.py:49` @ `9035151d6`) — so `mm->positions3` is never
  // read and `kDots3NoteFactory` leaves `mrope_prompt_positions` null. The
  // runner then hands the ordinary 1-D `positions`, which is what this forward
  // has always consumed.
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Dots3NoteWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices,
      const MultiModalForwardInput* mm = nullptr);
};

// The MLA KV topology. Both classes share ONE physical row of
// `swa_kv_lora_rank + swa_qk_rope_head_dim` = 1088 (see
// `physical_latent_row()`); the full layers read their logical 576 out of it.
v1::KVCacheConfig MakeDots3NoteKVCache(const HfConfig& config, int block_size,
                                       int num_blocks);

}  // namespace vllm
