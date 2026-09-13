// DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`) — the W1 config surface.
//
// SCOPE. This header is W1 of `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`
// (ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP): it makes the architecture RESOLVE and
// makes its config PARSE and VALIDATE. It does NOT make the model load and it
// does NOT make it forward. The loader, the forward, the KV-cache spec and the
// GGUF container each refuse BY NAME, naming the missing part and the wave that
// owes it (`.agents/specs/deepseek-v4-1-flash.md` `## Work breakdown`).
//
// WHY A NEW FILE RATHER THAN A WIDENING OF `deepseek_v4.h`. Upstream FORKED:
// `vllm/models/deepseek_v4_1/` is a sibling package of `vllm/models/deepseek_v4/`
// rather than a parameterization of it, and the two disagree on values the V4
// parser hard-validates. `ParseDeepseekV4Params` would reject this checkpoint
// outright — its `compress_ratios` are drawn from {0,1,2} where V4's are drawn
// from {0,4,128}, and V4's `has_indexer(layer)` is `compress_ratio == 4`, which
// is never true here. Widening the V4 parser to accept both would put V4's
// landed gates at risk for no gain, so V4.1 gets its own parser, its own
// registration TU and its own refusals. AGENTS.md: "Add new files for new
// hardware, architectures, and models. Mirror the vLLM file structure."
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream revision: vLLM `e77daef89e`. This is AHEAD OF our parity pin, and no
// pin was advanced by this wave. `DeepseekV41ForCausalLM` does not exist at the
// pin; the architecture landed on vLLM `main` in September 2026. Registration
// claims NO gate against the oracle, which is the same standing this tree
// already gives `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM` — both sit at
// `PARTIAL` as an explicit "Ahead-of-pin forward port … REGISTERED, NOT
// RUN-GATED" (#490). What a pin advance gates is the DENOMINATOR, and nothing
// here produces one.
//
//   OURS                      <-  UPSTREAM
//   DeepseekV41Params         <-  transformers_utils/configs/deepseek_v41.py
//                                 (the whole class: it is a FLATTENING shim)
//   ParseDeepseekV41Params    <-  the same file, plus the validation that lives
//                                 in models/deepseek_v4_1/attention.py:250-283
//                                 and models/deepseek_v4_1/compressor.py:192-210
//   the ONE registered string <-  models/registry.py:379
//                                 "DeepseekV41ForCausalLM" ->
//                                 vllm.models.deepseek_v4_1
//
// TWO UPSTREAM FACTS THIS HEADER RECORDS RATHER THAN SMOOTHS OVER.
//
// 1. `DeepseekV41Config.__init__` RAISES NOTHING. It is purely a flattening
//    shim: it lifts `text_config` onto the top level, re-exposes `vision_config`
//    under `vision_*` names, and lifts `expert_dtype` out of
//    `quantization_config`. Every validation mirrored below therefore comes from
//    the MODEL code that consumes the config, not from the config class, and
//    each one cites the line it came from. A validation with no upstream line
//    beside it is ours, and says so.
// 2. `len(compress_ratios)` is 43 while `num_hidden_layers` is 40. That is NOT
//    a malformed config and a length-equality check would REJECT the real
//    published artifact. The extra three entries are the
//    `num_nextn_predict_layers == 3` DSpark/MTP layers, and upstream reads the
//    list with an explicit bounds guard — `if compress_ratios is not None and
//    layer_id < len(compress_ratios)`, else 0, commented "MTP layers past the
//    configured list are pure sliding-window" (attention.py:251-256). We mirror
//    the guard and deliberately do NOT add the length check.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

class GgufFile;

// Every DeepSeek-V4.1-Flash config field W1 resolves, read ONCE from the
// HfConfig. Values in the trailing comments are the REAL published
// `deepseek-ai/DeepSeek-V4.1-Flash` config.json at revision
// `dba1be0a40aa45a94ad051997016db3960a90277`, which is checked in byte-for-byte
// at `tests/vllm/models/fixtures/deepseek_v4_1/config.json` (sha256
// 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879) so a silent
// re-quantization upstream under an unchanged name cannot move what we gated on.
//
// THE NESTING IS THE POINT. V4's keys sit at the top level of its config.json;
// V4.1's sit under `text_config` (and the tower under `vision_config`), which is
// exactly what `DeepseekV41Config` exists to flatten. `ParseHfConfig` already
// resolves `text_config` for the fields IT types (mirroring upstream
// `PretrainedConfig.get_text_config()`), but the DeepSeek-specific keys are not
// typed there and are read out of `config.raw["text_config"]` here.
struct DeepseekV41Params {
  // --- shared geometry ---
  int64_t hidden_size = 0;          // 5120
  int64_t num_hidden_layers = 0;    // 40
  int64_t vocab_size = 0;           // 129280
  int64_t num_attention_heads = 0;  // 64
  int64_t num_key_value_heads = 0;  // 1 (MLA: single latent)
  // 1e-20, and that is the published value rather than a typo for 1e-2. It is
  // carried as double and narrowed at use. `float` would REPRESENT it fine --
  // `float(1e-20)` is `0x1e3ce508`, a normal float, eighteen orders of
  // magnitude above f32's smallest normal 1.175e-38 -- but it would not
  // round-trip the JSON literal: `float(1e-20)` is 9.99999968e-21, a relative
  // error of 3.2e-8. `double` holds the literal the checkpoint ships exactly,
  // which is what lets the gate compare against it EXACTLY instead of picking
  // a tolerance at a magnitude where any relative epsilon is arbitrary.
  double rms_norm_eps = 1e-6;
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;   // 1048576
  int64_t num_nextn_predict_layers = 0;  // 3 (DSpark/MTP tail)

  // --- 512-wide MLA, unchanged in geometry from V4 (448 NoPE + 64 RoPE) ---
  int64_t head_dim = 0;          // 512
  int64_t qk_rope_head_dim = 0;  // 64
  int64_t q_lora_rank = 0;       // 1280 (V4 Flash: 1024 — this one MOVED)
  int64_t o_lora_rank = 0;       // 1024
  int64_t o_groups = 0;          // 8
  int64_t sliding_window = 0;    // 128
  double rope_theta = 10000.0;
  double compress_rope_theta = 160000.0;  // dual theta for compressed layers
  // YaRN, read from `text_config.rope_scaling`. Upstream pops this key before
  // `super().__init__` and restores it onto `rope_parameters` afterwards,
  // because `rope_scaling` is a PROPERTY in transformers v5 and the base
  // constructor re-standardizes it (deepseek_v41.py:29-44). We have no such
  // property to dodge, so we read the same values directly; the comment records
  // why upstream's dance exists rather than imitating a workaround we do not
  // need.
  std::string rope_type = "yarn";
  double rope_scale_factor = 1.0;  // 16
  int64_t rope_orig_ctx = 0;       // 65536
  double rope_beta_fast = 32.0;
  double rope_beta_slow = 1.0;

  // --- MoE ---
  int64_t n_routed_experts = 0;       // 384 (V4 Flash: 256)
  int64_t num_experts_per_tok = 0;    // 6
  int64_t moe_intermediate_size = 0;  // 2304
  int64_t n_shared_experts = 0;       // 1
  bool norm_topk_prob = true;
  double routed_scaling_factor = 1.0;  // 1.5
  double swiglu_limit = 0.0;           // 10.0 (clamped SwiGLU)
  std::string scoring_func = "sqrtsoftplus";
  std::string topk_method = "noaux_tc";
  // "fp4" | "fp8", lifted out of `quantization_config` by upstream's config
  // class (deepseek_v41.py:48-50) rather than read from the text config.
  std::string expert_dtype;  // "fp4"
  // V4.1 has NO `num_hash_layers`. V4 shipped 3 hash-gated layers whose
  // `ffn.gate.tid2eid` replaced the learned gate; this checkpoint carries no
  // such key, so the field does not exist here rather than defaulting to 0 and
  // inviting a reader to look for the tensors.

  // --- Manifold Hyper-Connections ---
  int64_t hc_mult = 0;            // 4
  int64_t hc_sinkhorn_iters = 0;  // 20
  double hc_eps = 1e-6;

  // --- DSA lightning indexer + the ratio-1/2 compressors ---
  int64_t index_head_dim = 0;  // 128
  int64_t index_n_heads = 0;   // 32 (V4 Flash: 64 — this one MOVED)
  int64_t index_topk = 0;      // 512
  // 43 entries against 40 layers; see the header note. Values drawn from
  // {0,1,2}, NOT V4's {0,4,128}.
  std::vector<int64_t> compress_ratios;
  // NEW vs V4: the compressed layers read their KV and their index state from
  // an earlier SOURCE layer rather than computing it per layer.
  std::vector<int64_t> kv_source_layer_ids;     // [2,8,14,20]
  std::vector<int64_t> index_source_layer_ids;  // [2,8,14,20,24,28,32,36]
  int64_t candidate_source_layer_id = -1;       // 20
  int64_t candidate_topk_blocks = 0;            // 2048
  int64_t candidate_block_size = 0;             // 8

  // --- Engram (NEW; no V4 counterpart) ---
  std::vector<int64_t> engram_layer_ids;      // [1,14]
  std::vector<int64_t> engram_num_embeddings; // [384006168, 384016682]
  int64_t engram_max_ngram_size = 0;          // 4
  int64_t engram_vocab_size = 0;              // 16000000
  int64_t engram_n_heads = 0;                 // 8
  int64_t engram_head_dim = 0;                // 256
  int64_t engram_pad_token_id = 0;            // 2
  int64_t engram_compressed_vocab_size = 0;   // 99092

  // --- DSpark block drafter (NEW; V4 had a separate DSparkDraftModel) ---
  int64_t dspark_block_size = 0;             // 5
  int64_t dspark_noise_token_id = 0;         // 128799
  std::vector<int64_t> dspark_target_layer_ids;  // [37,38,39]
  int64_t dspark_markov_rank = 0;            // 256
  int64_t dspark_n_routed_experts = 0;       // 128
  int64_t dspark_num_experts_per_tok = 0;    // 3

  // --- the vision_* lift (deepseek_v41.py:52-62) ---
  // Upstream re-exposes `vision_config` under these names because the model code
  // reads flat attributes. The DEFAULTS below are upstream's own `.get(...)`
  // fallbacks, not ours, and they differ from "absent": `vision_dim` falls back
  // to 1024 even when there is no tower at all.
  int64_t vision_n_layers = 0;          // 32
  int64_t vision_dim = 1024;            // 1024
  int64_t vision_n_heads = 16;          // 16
  int64_t vision_inter_dim = 2816;      // 2816
  int64_t vision_patch_size = 14;       // 14
  double vision_rope_theta = 10000.0;   // 10000
  int64_t vision_downsample_ratio = 3;  // 3
  int64_t vision_max_n_token = 1024;    // 1024
  int64_t vision_min_pixels = 295936;   // 295936
  // `vision_config.max_wh_ratio` is literally `null` in the published file, and
  // upstream's `.get("max_wh_ratio")` yields None with no fallback. An optional
  // is the only faithful carrier: 0.0 would be a ratio, and a ratio of zero is
  // not what "absent" means.
  std::optional<double> vision_max_wh_ratio;

  // --- the three mm-prefix flags upstream DERIVES (deepseek_v41.py:66-76) ---
  // All three are functions of `vision_n_layers > 0` and none is read from the
  // file. They are carried because the mm-prefix plumbing reads them off the
  // config, and because the modulus is the one value that DIFFERS from V4.
  bool is_mm_prefix_lm = false;
  bool mm_prefix_clamp_sliding_window = false;
  // 2 for V4.1 against V4's 4 (deepseek_v4.py:51), because V4.1's compressors
  // are ratio-2 where V4's were ratio-4. Verified on both upstream files.
  int64_t mm_prefix_span_leading_pad_modulus = 0;

  // --- quantization_config, carried verbatim ---
  std::string quant_method;              // "fp8"
  std::string quant_activation_scheme;   // "dynamic"
  std::string quant_scale_fmt;           // "ue8m0"
  std::vector<int64_t> weight_block_size;  // [32,32] (V4: [128,128])

  // Upstream's bounds guard, not a lookup: a layer past the end of the list is
  // 0 ("pure sliding-window"), never out-of-range (attention.py:251-256).
  int64_t compress_ratio(int64_t layer) const {
    return (layer >= 0 && static_cast<size_t>(layer) < compress_ratios.size())
               ? compress_ratios[static_cast<size_t>(layer)]
               : 0;
  }
  bool has_compressor(int64_t layer) const { return compress_ratio(layer) != 0; }
  // `is_backbone = layer_id < config.num_hidden_layers`
  // (`deepseek_v4_1/attention.py:273` — the V4.1 package, not V4's file of the
  // same name, whose `:274` is the unrelated `compress_ratio == 4` branch).
  bool is_backbone_layer(int64_t layer) const {
    return layer >= 0 && layer < num_hidden_layers;
  }
};

// Resolve + validate `DeepseekV41Params` from an HfConfig. Pure/host — it is
// unit-testable without a checkpoint, which is what makes W1 gateable at all on
// a fleet where no published V4.1 artifact fits any device we own. Throws with a
// precise message on a missing required field or a value this wave cannot
// represent; each message names the field.
DeepseekV41Params ParseDeepseekV41Params(const HfConfig& config);

// The registry's config hook. The resolve IS the validation.
void ParseDeepseekV41Config(const HfConfig& config);

// The GGUF-side half of the arch entry points, in `IsNemotronHGguf` /
// `IsDots3NoteGguf` shape (#809, #2882): a KNOWN architecture whose GGUF arm is
// OWED refuses with ITS OWN message, so the reader lands on the row that owes
// the work instead of on a generic "unrecognized architecture".
//
// THE SPELLING IS DERIVED AND PROVISIONAL, AND THIS COMMENT IS WHERE THAT IS
// RECORDED. No tool anywhere writes a DeepSeek-V4.1 GGUF today: no converter in
// this tree emits one, and the architecture is days old. `deepseek41` is the
// spelling produced by the same rule that maps this project's `deepseek_v4` ->
// `deepseek4` arm. W8 owns the converter and the loader arm and may settle on a
// different string; until it does, this predicate makes the refusal reachable
// rather than leaving the path to be discovered later. It is a REFUSAL and not
// a capability, so a wrong guess costs a reader nothing.
inline constexpr const char* kDeepseekV41GgufArch = "deepseek41";
bool IsDeepseekV41Gguf(const GgufFile& gguf);

// The ONE owner of the W8 GGUF refusal text. Both doors a GGUF can arrive at —
// the entrypoint's architecture dispatch and the loader hook's source-kind
// guard — throw THIS string, so whichever one a user reaches names this model,
// this arm and the wave that owes it. A second spelling is how the two drift.
std::string DeepseekV41GgufRefusal();

}  // namespace vllm
