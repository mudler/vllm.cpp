// DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`) — the ADDITIVE model TU skeleton
// for the DeepSeek-V4 bring-up campaign (`CLAIM-DEEPSEEK-V4-IMPL`, W1/W2). This
// header defines the config parse + the checkpoint-verified weight layout + the
// forward/KV-cache seams; the genuinely-NEW primitives (Manifold Hyper-
// Connections, the DSA Lightning-Indexer + Compressor, the 512-wide MLA geometry
// with grouped output-LoRA, sqrtsoftplus/hash MoE) are STUBBED with precise
// `TODO(W3-W8)` port markers and a `VT_CHECK(false, ...)` forward so the TU
// BUILDS but a forward loudly reports the pending brick — never a silent wrong
// answer. The full forward + the strict gate are NAMED residuals (W3-W8), see
// `.agents/specs/deepseek-v4-flash.md` §5.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922 / 0.26.0.dev0) ──
//   OURS                             <-  UPSTREAM (all under vllm/models/deepseek_v4/)
//   DeepseekV4Params                 <-  transformers_utils/configs/deepseek_v4.py
//                                        + the shipped nvidia/DeepSeek-V4-Flash-NVFP4
//                                        config.json (arch scalars)
//   ParseDeepseekV4Params            <-  nvidia/model.py:512-757 (DeepseekV4MoE),
//                                        attention.py:122-315 (DeepseekV4Attention),
//                                        nvidia/model.py:794-957 (DecoderLayer + MHC)
//   LoadDeepseekV4ForCausalLMWeights <-  nvidia/model.py:1150-1310 (load_weights),
//                                        :1316-1360 (_make_deepseek_v4_weights_mapper)
//   MakeDeepseekV4KVCache            <-  attention.py:626 (get_kv_cache_spec) — STUB
//   DeepseekV4Model::Forward         <-  nvidia/model.py:1080-1148 — STUB (W3-W8)
//
// ─── LOADER VERIFIED vs the REAL checkpoint header (2026-07-28, no download) ────
// `nvidia/DeepSeek-V4-Flash-NVFP4` model.safetensors.index.json (135,235 tensors,
// 46 shards) + shard-2 safetensors header via HTTP range. Confirmed name map +
// dtypes/shapes (see deepseek_v4_weights.cpp for the full table). HW-FIT REVERSAL
// recorded there: index `total_size` = 168,266,793,544 B = **156.7 GiB**, so this
// NVFP4 checkpoint does NOT fit ONE GB10's 119 GiB unified pool — the W0 spike's
// "~83 GiB fits" estimate was wrong (only the 256 routed experts are NVFP4; the
// MLA/shared-expert linears are FP8 block, + NVFP4 double-scale overhead). W1
// (single-GB10 oracle run) is therefore MEMORY-INFEASIBLE, not merely disk-blocked.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;
class GgufFile;
struct GgufLoadPolicy;

// Every DeepSeek-V4-Flash config field the loader/forward consume, resolved ONCE
// from the HfConfig. DeepSeek keys (`n_routed_experts`, `hc_mult`, `o_groups`,
// `compress_ratios`, ...) are NOT on the typed HfConfig struct, so most are read
// from `config.raw`. Values below are the shipped
// `nvidia/DeepSeek-V4-Flash-NVFP4` config.json (VERIFIED 2026-07-28).
struct DeepseekV4Params {
  // --- shared geometry ---
  int64_t hidden_size = 0;          // 4096
  int64_t num_hidden_layers = 0;    // 43
  int64_t vocab_size = 0;           // 129280
  int64_t num_attention_heads = 0;  // 64
  int64_t num_key_value_heads = 0;  // 1 (MLA: single latent)
  float rms_norm_eps = 1e-6f;       // 1e-6
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;  // 1048576
  int64_t num_nextn_predict_layers = 0;  // 1 (MTP tail; loader SKIPS `mtp.*`)

  // --- 512-wide MLA (attention.py) — NEW geometry vs V2/V3 (448 NoPE + 64 RoPE) ---
  int64_t head_dim = 0;          // 512  (= 448 v/nope + 64 rope, wq_b out = heads*512)
  int64_t qk_rope_head_dim = 0;  // 64
  int64_t q_lora_rank = 0;       // 1024
  int64_t o_lora_rank = 0;       // 1024  (grouped OUTPUT LoRA — NEW, no V2/V3 analogue)
  int64_t o_groups = 0;          // 8
  int64_t sliding_window = 0;    // 128  (per-head attention sinks + SWA — NEW)
  double rope_theta = 10000.0;
  double compress_rope_theta = 160000.0;  // dual theta for compressed layers
  // YaRN (compressed layers only): freq_scale = 1/rope_scale_factor interpolation
  // with the beta_fast/beta_slow correction-dim ramp (ds4 rope_tail_ext_inplace;
  // net mscale == 1). Dense layers (compress_ratio==0) use plain theta, scale 1.
  double rope_scale_factor = 16.0;
  int64_t rope_orig_ctx = 65536;
  double rope_beta_fast = 32.0;
  double rope_beta_slow = 1.0;

  // --- MoE (nvidia/model.py:512-757) ---
  int64_t n_routed_experts = 0;      // 256
  int64_t num_experts_per_tok = 0;   // 6
  int64_t moe_intermediate_size = 0; // 2048
  int64_t n_shared_experts = 0;      // 1 (FP8 block, NOT NVFP4)
  bool norm_topk_prob = true;
  double routed_scaling_factor = 1.0;  // 1.5
  double swiglu_limit = 0.0;           // 10.0 (clamped SwiGLU — NEW)
  // "sqrtsoftplus" (NEW — not sigmoid/softmax) + noaux_tc e_score_correction_bias.
  std::string scoring_func = "sqrtsoftplus";
  std::string topk_method = "noaux_tc";
  // First `num_hash_layers` layers replace the learned gate with a `tid2eid`
  // token-id -> expert-id HASH lookup (NEW). Verified: layers 0,1,2 carry
  // `ffn.gate.tid2eid` and NO `ffn.gate.bias`.
  int64_t num_hash_layers = 0;  // 3
  // "fp4" (MXFP4/NVFP4 experts) | "fp8" (block experts). NVFP4 vehicle = "fp4".
  std::string expert_dtype = "fp4";

  // --- Manifold/Markov Hyper-Connections (MHC) — NEW topology, no eager ref ---
  int64_t hc_mult = 0;           // 4  (residual stream -> [T, hc_mult, H])
  int64_t hc_sinkhorn_iters = 0; // 20
  double hc_eps = 1e-6;

  // --- DSA Lightning-Indexer + Compressor (attention.py:689-857) — NEW ---
  int64_t index_head_dim = 0;  // 128
  int64_t index_n_heads = 0;   // 64
  int64_t index_topk = 0;      // 512
  // Per-layer compress ratio: 4 => indexed (has DSA indexer + compressor@4),
  // 128 => compressor@128 only, 0 => neither (verified: layers 0,1 and the last
  // carry 0). Length == num_hidden_layers.
  std::vector<int64_t> compress_ratios;

  bool is_hash_layer(int64_t layer) const { return layer < num_hash_layers; }
  int64_t compress_ratio(int64_t layer) const {
    return (layer >= 0 && static_cast<size_t>(layer) < compress_ratios.size())
               ? compress_ratios[static_cast<size_t>(layer)]
               : 0;
  }
  bool has_compressor(int64_t layer) const { return compress_ratio(layer) != 0; }
  bool has_indexer(int64_t layer) const { return compress_ratio(layer) == 4; }
};

// Resolve + validate DeepseekV4Params from a HfConfig. Pure/host — unit-testable
// without a checkpoint (config-descent gate). Throws with a precise message on a
// missing required field or a value this bring-up cannot represent.
DeepseekV4Params ParseDeepseekV4Params(const HfConfig& config);

// ─── W7 host-float weight tower (the tiny-config CPU forward assembly) ────────
// The W7 forward composes the landed host-reference primitives on the portable
// CPU path at a SMALL synthetic config (the fixed-config 167B does not fit ONE
// GB10, so the real e2e run is the multi-Spark W8 gate — see this header top).
// These structs hold the plain-float weights the CPU forward consumes; they are
// populated by the STRUCTURAL unit gate (test_deepseek_v4_forward.cpp) at a tiny
// shape, NOT by the real checkpoint loader — the FP8-block + NVFP4 tower
// MATERIALIZATION into this layout is the named W2b residual. All tensors row-major
// fp32 unless noted.
struct DeepseekV4LayerHostWeights {
  // MHC mixing (nvidia/model.py:820-865): hc_attn/hc_ffn fn [(2+hc)*hc, hc*H],
  // base [(2+hc)*hc], scale [3]; the attn/ffn RMSNorms folded into the pre-mix.
  std::vector<float> attn_norm_weight;  // [H]
  std::vector<float> ffn_norm_weight;   // [H]
  std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;
  std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  // 512-wide MLA (attention.py): q down/up, kv down, per-branch RMSNorms,
  // per-head attention sink, grouped OUTPUT-LoRA wo_a (bmm) + wo_b.
  std::vector<float> wq_a;            // [q_lora_rank, H]
  std::vector<float> q_norm_weight;   // [q_lora_rank]
  std::vector<float> wq_b;            // [n_heads*head_dim, q_lora_rank]
  std::vector<float> wkv;             // [head_dim, H]
  std::vector<float> kv_norm_weight;  // [head_dim]
  std::vector<float> attn_sink;       // [n_heads]
  std::vector<float> wo_a;            // [n_groups, o_lora_rank, in_per_group]
  std::vector<float> wo_b;            // [H, n_groups*o_lora_rank]
  // DSA Lightning-Indexer (indexer layers only; empty otherwise).
  std::vector<float> idx_wq;     // [index_n_heads*index_head_dim, H]
  std::vector<float> idx_wk;     // [index_head_dim, H]
  std::vector<float> idx_wproj;  // [index_n_heads, H]
  // DSA compressor (compressor layers only; empty otherwise).
  std::vector<float> comp_wgate;        // [head_dim, H]  (produces the pool score)
  std::vector<float> comp_ape;          // [compress_ratio, head_dim]
  std::vector<float> comp_norm_weight;  // [head_dim]
  // MoE router: learned gate + (non-hash) noaux_tc bias OR (hash) tid2eid table.
  std::vector<float> gate_weight;  // [n_routed_experts, H]
  std::vector<float> gate_bias;    // [n_routed_experts]  (non-hash layers)
  std::vector<int32_t> tid2eid;    // [vocab, num_experts_per_tok] (hash layers)
  // Shared + routed experts (clamped SwiGLU). Routed stored flat over experts.
  std::vector<float> shared_w1, shared_w3;  // [moe_inter, H]
  std::vector<float> shared_w2;             // [H, moe_inter]
  std::vector<float> exp_w1, exp_w3;        // [n_experts, moe_inter, H]
  std::vector<float> exp_w2;                // [n_experts, H, moe_inter]
};
struct DeepseekV4HostWeights {
  std::vector<float> embed;              // [vocab, H]
  std::vector<float> lm_head;            // [vocab, H]  (untied)
  std::vector<float> final_norm_weight;  // [H]
  std::vector<float> hc_head_fn;         // [hc, hc*H]
  std::vector<float> hc_head_base;       // [hc]
  float hc_head_scale = 0.0f;            // scalar (hc_head_scale[0] upstream)
  std::vector<DeepseekV4LayerHostWeights> layers;
};

// ─── W2b GGUF keep-quant weight tower (the `deepseek4` GGUF materialization) ───
// The single-Spark `unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS` (~91 GiB, the only
// build that fits ONE GB10's 119 GiB unified pool) materializes here. Every GGUF
// `blk.N.*` tensor routes via the LANDED name-map (scripts/check-dsv4-gguf-namemap.py,
// EXACT 1328/1328 coverage) into the slot below. The MW/SEW roles (the MLA linears,
// the router gate, the shared + the 256 routed experts, lm_head) KEEP their ~2-3-bit
// blocks COMPRESSED via `OwnGgufQuantBlocks` (the keep-quant memory enabler:
// dequant-to-bf16 would need ~316 GiB and OOM-reboot the box); the small V/ET/HASH
// tensors (norms, MHC/DSA mixing, sinks, the `tid2eid` hash table, embed) dequant to
// f32/bf16 exactly as our other GGUF loaders do. Keeping the block dtype is the point:
// `OwnedTensor::dtype` stays a `vt::IsBlockQuant` type and the CPU `kMatmulBTQuant`
// GEMM (CIQ) consumes it in place. An empty OwnedTensor means "this layer has no such
// slot" (a non-indexer/non-compressor/gated-vs-hash layer).
struct DeepseekV4GgufLayerWeights {
  // 512-wide MLA (MW keep-quant capable) + its norms/sink (V, dequant f32).
  OwnedTensor wq_a, wq_b, wkv, wo_a, wo_b;
  OwnedTensor attn_norm, q_a_norm, kv_a_norm, attn_sink, ffn_norm;
  // MHC per-layer mixing (V, dequant f32).
  OwnedTensor hc_attn_base, hc_attn_fn, hc_attn_scale;
  OwnedTensor hc_ffn_base, hc_ffn_fn, hc_ffn_scale;
  // MoE: router gate (MW) + the 256 routed experts + shared expert (SEW/MW
  // keep-quant), plus the per-layer router side-table (HASH `tid2eid` on the
  // first num_hash_layers layers, else the noaux_tc `e_score_bias`, both V).
  OwnedTensor moe_gate, moe_gate_exps, moe_up_exps, moe_down_exps;
  OwnedTensor shared_gate, shared_up, shared_down;
  OwnedTensor tid2eid, e_score_bias;
  // DSA compressor (compressor layers only) + Lightning-Indexer (indexer layers).
  OwnedTensor comp_ape, comp_wgate, comp_wkv, comp_norm;
  OwnedTensor idx_wq_b, idx_proj;
  OwnedTensor idx_comp_ape, idx_comp_wgate, idx_comp_wkv, idx_comp_norm;
  bool is_hash = false, has_compressor = false, has_indexer = false;
};
struct DeepseekV4GgufWeights {
  OwnedTensor embed, lm_head, final_norm;               // ET / MW / V
  OwnedTensor hc_head_base, hc_head_fn, hc_head_scale;  // V (final head collapse)
  std::vector<DeepseekV4GgufLayerWeights> layers;
};

// Whole DeepSeek-V4 weights. W1/W2 SCAFFOLDING: carries the resolved params + the
// loader's accounting result. The safetensors loader still ACCOUNTS for every
// checkpoint tensor without uploading the FP8/NVFP4 towers (a named residual). The
// GGUF loader (W2b) MATERIALIZES the keep-quant `deepseek4` tower into `gguf` and
// dequants the tiny-config CPU composition tower into `host`. The W7 CPU forward
// runs off `host` when `has_host_weights` is set.
// ─── MODEL-DSV4-EXL3 W1b: the rank-sliced EXL3 routed-expert tower ───────────
//
// The SparkInfer artifact `0xSero/deepseek-v4-flash-0731-spark` re-quantizes ONLY
// the routed experts, to EXL3 trellis (exllamav3 @ 2398c056, MIT — vLLM ships no
// EXL3 at the parity pin, so exllamav3 is this format's secondary oracle). Its
// `config.json` declares `quantization_config.quant_method == "exl3"` +
// `version == "rank-sliced-deepseek-v4-v1"`, and the expert tensors arrive
// PRE-SLICED across four tensor-parallel ranks under
// `layers.{L}.ffn.experts.{E}.{w1|w2|w3}.rank{r}.{trellis|suh|svh|mcg}`.
// Everything else ships as the un-requantized `deepseek_v4_fp8` source tensors in
// `carried-*.safetensors`, which the existing name-map accounts for unchanged.
//
// The loader coalesces TP4 -> TP1 at load. That inverts upstream's own
// `LinearEXL3.tp_import_split` (`modules/quant/exl3.py:296-313`): an OUT split
// took `svh[first:last]` + `trellis[:, first//16:last//16]` (w1, w3), an IN split
// took `suh[first:last]` + `trellis[first//16:last//16, :]` (w2). Both inverses
// are pure concatenation and LOSSLESS — 16x16 trellis tiles are independent and
// every rank boundary is a multiple of the Hadamard block size 128
// (`exl3_lib/quantize.py:15`).
struct DeepseekV4Exl3Linear {
  int64_t in_features = 0;   // k
  int64_t out_features = 0;  // n
  int bits = 0;              // K (3 for the 3.0bpw artifact)
  // The four stored tensors (`exl3.py:20-91`). There are NO scales — `exl3.py:38`
  // asserts "scale is no longer used"; `suh`/`svh` carry the per-channel scale.
  std::vector<uint16_t> trellis;  // [k/16, n/16, 16*bits], the int16 bit patterns
  std::vector<uint16_t> suh;      // [k], the fp16 bit patterns
  std::vector<uint16_t> svh;      // [n], the fp16 bit patterns
  int32_t mcg = 0;                // codebook marker; never read at inference
                                  // (`exl3_lib/quantize.py:1414-1424`)
  int64_t Bytes() const {
    return static_cast<int64_t>((trellis.size() + suh.size() + svh.size()) *
                                sizeof(uint16_t));
  }
};
struct DeepseekV4Exl3Expert {
  DeepseekV4Exl3Linear w1, w2, w3;
};
struct DeepseekV4Exl3LayerWeights {
  std::vector<DeepseekV4Exl3Expert> experts;
};
struct DeepseekV4Exl3Weights {
  int tp = 0;            // the source artifact's tensor-parallel width
  int bits = 0;          // K
  std::string codebook;  // "mcg" — the only codebook this arm decodes
  std::string version;   // "rank-sliced-deepseek-v4-v1"
  std::vector<DeepseekV4Exl3LayerWeights> layers;
  // MTP tail tensors skipped, mirroring vLLM's own DeepSeek-V4 loader
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474). COUNTED
  // rather than dropped silently: on the real artifact these are 3985 NVFP4
  // draft-head tensors, and a reader must be able to see that the loader met
  // them and chose to skip them.
  int64_t skipped_mtp_tensors = 0;
};

struct DeepseekV4Weights {
  DeepseekV4Params params{};
  // Accounting from the verified name-map pass (W2 gate): how many checkpoint
  // tensors the loader recognized / consumed. Diagnostic (and the gate assertion).
  int64_t accounted_tensors = 0;
  // W7 host-float tower for the tiny-config CPU forward composition. Populated by
  // the GGUF loader (dequant bridge) or the structural gate. `Forward` VT_CHECKs it.
  DeepseekV4HostWeights host{};
  bool has_host_weights = false;
  // W2b materialized keep-quant `deepseek4` GGUF tower (blocks stay COMPRESSED for
  // the MW/SEW roles). Non-empty only on the GGUF load path.
  DeepseekV4GgufWeights gguf{};
  bool has_gguf_weights = false;
  // MODEL-DSV4-EXL3 W1b: the TP1-coalesced EXL3 routed-expert tower. Non-empty
  // only when `quantization_config.quant_method == "exl3"`. EXECUTION is NOT in
  // this wave: nothing consumes the tower yet, so a forward over an EXL3 load
  // still refuses through the existing `has_host_weights` guard. The
  // dequant-to-bf16 fallback arm is MODEL-DSV4-EXL3 W1c and the trellis GEMM is
  // W2; both are listed under that spec's `## Owed`.
  DeepseekV4Exl3Weights exl3{};
  bool has_exl3_weights = false;
};

// Host bytes the coalesced EXL3 tower holds. The point of the number is the
// NEGATIVE it proves: the four per-rank slices are consumed and dropped, so this
// equals the TP1 tensor sizes and never four copies of them.
int64_t DeepseekV4Exl3ResidentBytes(const DeepseekV4Weights& weights);

// Price the coalesced tower from INSIDE the load, refuse one the host cannot
// hold, and report the figure once the last layer is in. Called per layer by
// `LoadDeepseekV4ForCausalLMWeights`'s EXL3 arm, and the only production reader
// of `DeepseekV4Exl3ResidentBytes`.
//
// `host_available_bytes` is a PARAMETER, not a query made inside: 0 means
// unknown and never refuses, and injecting it is what makes the refusal gateable
// without a machine of a chosen size — the same shape `check_enough_state_memory`
// (`vllm/v1/core/kv_cache_utils.h`) uses for the recurrent-state budget.
// Returns the bytes the tower holds so far.
int64_t ReportDeepseekV4Exl3Residency(const DeepseekV4Weights& weights,
                                      int64_t layers_done, int64_t layers_total,
                                      int64_t host_available_bytes);

// Pure predicate for the `VT_DSV4_EXL3_HOST_BUDGET` contract: the EXL3 arm's
// host-residency refusal is ON by default and OFF only when the environment
// value is present AND its first character is '0'. nullptr (unset) and every
// non-'0'-leading value are ON. This is the house default-ON / '0'-rollback
// shape (`AsyncRunnerFlagIsOn`, `vllm/v1/worker/gpu/async_runner_flag.h`), and
// the parse is factored out of the getenv call so it is unit-testable without
// touching the environment.
//
// WHY AN OVERRIDE EXISTS AT ALL. The budget the refusal measures against is
// `/proc/meminfo` MemAvailable, which is an ESTIMATE that is wrong in both
// directions — it ignores swap and under-counts some reclaimables, and inside a
// container it reports the HOST's figure rather than the cgroup's. The caveats
// are recorded in full at the read site in `deepseek_v4_weights.cpp`
// (`LoadDeepseekV4Exl3`). `VT_DSV4_EXL3_HOST_BUDGET=0` hands the reporter an
// UNKNOWN budget (0), which never refuses, so a developer who knows the estimate
// is wrong proceeds in the SAME binary instead of rebuilding one.
inline bool Dsv4Exl3HostBudgetFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

// Resolve DeepseekV4Params directly from a `deepseek4`-arch GGUF's KV metadata
// (block_count, hash_layer_count, expert_count, key/value_length, q_lora_rank,
// output_group_count, sinkhorn_iterations, indexer head/key/top_k, compress_ratios,
// ...). The GGUF vehicle carries no config.json, so this is the config half of the
// GGUF loader (the safetensors path uses ParseDeepseekV4Params off HfConfig.raw).
// Throws with a precise message on a missing/unrepresentable field.
DeepseekV4Params DeepseekV4ParamsFromGguf(const GgufFile& gguf);

// Build a full HfConfig from a `deepseek4`-arch GGUF's KV metadata. This is the
// TOP-LEVEL GGUF dispatch arm for DeepSeek-V4: it maps llama.cpp's
// `general.architecture == "deepseek4"` onto the registered vLLM model class
// `DeepseekV4ForCausalLM` (so `ModelRegistry::Resolve` routes the file into the
// DeepSeek-V4 factory) and republishes the resolved geometry into the typed
// HfConfig fields + `config.raw` (the scalars the registry parse hook
// `ParseDeepseekV4Config` validates). `LoadedEngine::FromModelDir` calls this for
// a deepseek4 GGUF exactly as it calls `HfConfigFromGguf` for the qwen families;
// the weight loader (`LoadDeepseekV4FromGguf`) still re-derives its params from
// the GGUF KV directly (it ignores `config`). Throws (via DeepseekV4ParamsFromGguf)
// on a non-deepseek4 file or a missing/unrepresentable field.
HfConfig DeepseekV4HfConfigFromGguf(const GgufFile& gguf);

// Materialize a `deepseek4` GGUF (`unsloth/DeepSeek-V4-Flash-GGUF`) into a
// DeepseekV4Weights. Routes EVERY GGUF tensor through GgufLoadPolicy with the
// name-map role (scripts/check-dsv4-gguf-namemap.py): MW/SEW stay keep-quant blocks,
// V/ET/HASH dequant. Accounts for every tensor (throws on a missing expected tensor
// or a leftover file tensor the map does not cover) and also builds the `host`
// tiny-config CPU composition tower (dequant bridge) so a loaded model can Forward.
// `policy` null → GgufLoadPolicy::FromEnv (keep-quant ON wherever the CPU quant GEMM
// is registered); tests pass an explicit keep-quant policy. `config` is accepted for
// the registry signature; the params are resolved from the GGUF KV directly.
DeepseekV4Weights LoadDeepseekV4FromGguf(const GgufFile& gguf, const HfConfig& config,
                                        const GgufLoadPolicy* policy = nullptr);

// ─── W2C memory accounting (the memory-bound gate) ───────────────────────────
// Resident bytes of the SMALL f32 `host` tower (norms/embed/mixing/hash/ape/sink
// + the int32 hash table). After W2C this does NOT include the big MLA/MoE/lm_head
// weights — those are keep-quant in `gguf`. A test asserts this stays far below
// the keep-quant tower + the f32-expanded projection (a rebuild of the ~1 TiB f32
// tower would blow it up).
int64_t DeepseekV4HostResidentBytes(const DeepseekV4Weights& w);
// Resident bytes of the keep-quant `gguf` tower (compressed blocks + the small
// V/ET dequant OwnedTensors it also carries).
int64_t DeepseekV4GgufResidentBytes(const DeepseekV4Weights& w);

// W7 structural-gate knobs. A deliberately-miswired interleave MUST change the
// output (RED-first) — the structural gate proves each lever is load-bearing.
enum class V4Miswire {
  kNone,              // the faithful interleave
  kAllLayersGated,    // route hash layers by learned top-k (ignore tid2eid)
  kSkipFinalMhcPost,  // skip the final residual fold before the head collapse
  kNoAttnSink,        // drop the per-head attention sink (plain softmax)
};

// Structural facts the W7 forward records for the composition gate (proves the
// interleave RAN as designed at tiny shape — not a numerical parity claim).
struct V4ForwardTrace {
  int64_t hc_mult = 0, hidden = 0, num_tokens = 0;
  int64_t residual_stream_elems = 0;  // == num_tokens*hc*hidden (proves [T,hc,H])
  std::vector<int> layer_is_hash;          // per layer: config says hash-routed
  std::vector<int> layer_hash_routed;      // per layer: the router took the hash branch
  std::vector<int> layer_is_indexer;       // per layer: DSA Lightning-Indexer present
  std::vector<int> layer_indexer_selected; // per layer: #keys the last query selected
  std::vector<int> layer_compressor_ran;   // per layer: the DSA compressor pooled KV
};

// The W7 tiny-config CPU forward: compose the landed host primitives (MHC pre/post
// + Sinkhorn, DSA indexer select + compressor + fp8_ds_mla KV, 512-wide MLA with
// sinks + grouped output-LoRA, sqrtsoftplus/hash MoE with clamped SwiGLU) into an
// end-to-end logits producer. Returns flat row-major logits for the `logits_indices`
// rows (all rows if empty). `miswire` deliberately breaks the interleave for the
// RED-first structural gate; `trace` (optional) records structural facts.
// Grounding: vllm/models/deepseek_v4/nvidia/model.py:1080-1148 (DeepseekV4Model.forward)
// + :866-957 (DeepseekV4DecoderLayer.forward).
std::vector<float> DeepseekV4ForwardHost(
    const DeepseekV4HostWeights& hw, const DeepseekV4Params& p,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& logits_indices = {},
    V4Miswire miswire = V4Miswire::kNone, V4ForwardTrace* trace = nullptr);

// W2C — the GGUF keep-quant forward. Runs the SAME composition as
// DeepseekV4ForwardHost but the big 512-wide MLA linears + the 256 routed/shared
// expert GEMMs + lm_head CONSUME the COMPRESSED `weights.gguf` keep-quant blocks
// in place via vt::MatmulBT -> the CPU kMatmulBTQuant CIQ GEMM (NO per-layer f32
// expansion — the ~1 TiB f32 tower is never built). The small non-GEMM tensors
// (norms, sinks, MHC/DSA mixing, ape, the hash table, embed) still come from the
// SMALL f32 `weights.host` tower the loader dequants, exactly as our other GGUF
// models keep them (qwen3_5_gguf_weights.cpp). Requires a queue (the CPU quant
// GEMM consumer). This is the memory ENABLER for the single-Spark `UD-IQ2_XXS`
// (~91 GiB) vehicle. Grounding: model.py:1080-1148 + vt/ops.cpp:134-201.
std::vector<float> DeepseekV4ForwardGguf(
    const DeepseekV4Weights& weights, vt::Queue& queue,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& logits_indices = {},
    V4Miswire miswire = V4Miswire::kNone, V4ForwardTrace* trace = nullptr);

// The MLA compressed-latent KV cache for INCREMENTAL decode (ForwardDevice
// campaign, Stage 1). For the real dense-MLA run (num_key_value_heads=1, no
// compressor/indexer) the only cross-token state is the per-layer `deck` latent
// [head_dim] fed to attention as key=value. This caches it per layer so decode
// processes ONE new token against cached KV instead of re-running the whole
// forward over the growing context. Mirror of ds4 `ds4_layer_cache.raw_kv`
// (ds4.c:12089/12351), cached at f32 to stay bit-exact to our full-recompute path.
struct DeepseekV4KvCache {
  std::vector<std::vector<float>> deck;  // [layer] -> flat [len * head_dim]
  int64_t len = 0;                       // cached token count (same for all layers)
  int64_t head_dim = 0;
  // Brick D: opaque persistent decode-CUDA-graph state (the fixed-capacity KV +
  // persistent scratch + captured graph), lazily built on the first graphed decode
  // step and reused across steps. void so this public header stays decoupled from
  // the CUDA-only graph type; the deleter (set in deepseek_v4.cpp) frees it.
  std::shared_ptr<void> decode_graph;
  void Reset(int64_t num_layers, int64_t head_dim_) {
    deck.assign(static_cast<size_t>(num_layers), {});
    len = 0;
    head_dim = head_dim_;
    decode_graph.reset();
  }
};

// Incremental-decode variant of DeepseekV4ForwardGguf. On the FIRST call (prefill)
// pass all prompt tokens with cache.len==0; each later call passes ONE new token
// with positions={cache.len}. The forward appends each token's per-layer `deck` to
// the cache and attends the new query over the full cached KV — token-IDENTICAL to
// the full-recompute path (a pure equivalence: same tokens, ~ctx x fewer FLOPs).
// `logits_indices` are LOCAL indices into the tokens passed THIS call.
std::vector<float> DeepseekV4ForwardGgufCached(
    const DeepseekV4Weights& weights, vt::Queue& queue, DeepseekV4KvCache& cache,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& logits_indices = {});

// Stage-2 decode-step profiling (inert unless env `VT_V4_PROF` is set). Split the
// forward's device wall time into GEMM-dispatch vs stream-drain; host glue is the
// remainder (step_time - gemm - sync). Reset before a step, read after.
void DeepseekV4ProfReset();
double DeepseekV4ProfGemmSeconds();
double DeepseekV4ProfSyncSeconds();

// Coherence-debug #188 Phase-2 discriminator: per routed expert, compare each
// projection's keep-quant kMatmulBTQuant output (A) vs a dequant-then-GEMM oracle
// (B) from the SAME blocks. Splits vec_dot/decode numerics (A!=B) from slice
// offset (A==B but over-scaled). Diagnostic (prints to stderr).
void DeepseekV4ExpertProbe(const DeepseekV4Weights& weights, vt::Queue& queue,
                           int64_t layer, const std::vector<int64_t>& experts);

// Per-head RMS-normalization of the MLA query: each of `n_head` contiguous
// `head_dim`-wide sub-vectors of `q` is scaled by 1/sqrt(mean(x^2) + eps) — NO
// learnable weight. DeepSeek-V4 applies this to q AFTER wq_b and BEFORE RoPE
// (ds4 `head_rms_norm_inplace` / `layer_q_projection_normed_one`; the KV latent
// does NOT get it — only its `attn_kv_a_norm`). q is laid out [n_head*head_dim]
// (for a batch, pass n_head = tokens*heads). #188 coherence fix.
void DeepseekV4QHeadRmsNormInplace(std::vector<float>& q, int64_t n_head,
                                   int64_t head_dim, float eps);

// Load `DeepseekV4ForCausalLM` safetensors into DeepseekV4Weights. Encodes the
// checkpoint name-map VERIFIED against the real header (deepseek_v4_weights.cpp)
// and performs the W2 accounting pass (throws on a missing expected tensor). The
// device materialization of the quantized towers is a NAMED W2b residual.
DeepseekV4Weights LoadDeepseekV4ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The DeepSeek-V4 forward. STUB (W3-W8): composes the 512-wide MLA block + DSA
// indexer/compressor + MHC hyper-connections + sqrtsoftplus/hash MoE, none of
// which are ported yet — both entrypoints VT_CHECK(false, ...) so a forward
// LOUDLY reports the pending brick. See `.agents/specs/deepseek-v4-flash.md` §5.
class DeepseekV4Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});
};

// ─── MTP (Multi-Token Prediction) self-speculative draft head ────────────────
// DeepSeek-V4 ships a native MTP head (`num_nextn_predict_layers = 1`): ONE extra
// decoder layer predicting the NEXT token from the current post-layer hidden, so
// the engine can self-speculate (draft k=1/step, the full model verifies via the
// shared greedy RejectionSampler — LOSSLESS by construction: MTP-on greedy output
// is token-IDENTICAL to MTP-off). Ground truth = the V4-SPECIFIC upstream module
// `vllm/models/deepseek_v4/nvidia/mtp.py` (DeepSeekV4MultiTokenPredictorLayer),
// which differs from V3's fused `eh_proj` (deepseek_mtp.py): V4 keeps SEPARATE
// e_proj/h_proj, an MHC-aware `mtp_block` decoder layer, and its own hc_head
// collapse in compute_logits. See `.agents/specs/deepseek-v4-mtp.md`.

// The checkpoint-owned tiny-config CPU MTP tower (the W1 host oracle; the real
// FP8/keep-quant materialization is a named residual, exactly like the main
// `host` tower). All tensors row-major f32 unless noted. The `mtp_block` reuses
// the main model's DeepseekV4LayerHostWeights layout (attn + MoE + MHC per-layer
// mixing). embed_tokens is SHARED with the target (not stored here).
struct DeepseekV4MtpHostWeights {
  std::vector<float> enorm_weight;   // [H]  RMSNorm on the embed  (nvidia/mtp.py:86)
  std::vector<float> hnorm_weight;   // [H]  RMSNorm on prev hidden (:87)
  std::vector<float> e_proj;         // [H,H] ReplicatedLinear, no bias (:90)
  std::vector<float> h_proj;         // [H,H] ReplicatedLinear, no bias (:99)
  // The MTP head's OWN hc_head collapse (mirrors the main model's hc_head_*).
  std::vector<float> hc_head_fn;     // [hc, hc*H]
  std::vector<float> hc_head_base;   // [hc]
  float hc_head_scale = 0.0f;        // scalar
  // shared_head: its OWN final norm + lm_head (checkpoint-owned).
  std::vector<float> shared_norm_weight;  // [H]
  std::vector<float> lm_head;             // [V,H]
  // The one V4 decoder layer (attn + MoE + per-layer MHC mixing).
  DeepseekV4LayerHostWeights mtp_block{};
};

// RED-first structural knobs for the MTP draft gate: a deliberately-broken lever
// MUST change the draft logits (each is load-bearing).
enum class V4MtpMiswire {
  kNone,          // the faithful nextn forward
  kSkipEhProj,    // feed the raw embed instead of eh-lift(embed, prev_hidden)
  kSkipHcHead,    // skip the hc_head collapse before the shared norm+lm_head
  kNoHnorm,       // drop the prev-hidden RMSNorm (hnorm)
};

// The W1 tiny-config MTP DRAFT FORWARD: nextn layer + compute_logits, 1:1 with
// nvidia/mtp.py:128-258, reusing the DS4 host composition helpers (AttentionBlock,
// MoeBlock, MhcPre/Post, HcHeadCollapse). `previous_hidden` is the target's
// PRE-hc_head MHC residual stream flat [T, hc*H] (NOT the post-final-norm hidden).
// `embed` is the SHARED target embed table [V,H]. Returns draft logits for the
// `logits_indices` rows (all rows if empty), flat row-major [rows, V].
std::vector<float> DeepseekV4MtpDraftLogitsHost(
    const DeepseekV4MtpHostWeights& mw, const DeepseekV4HostWeights& target,
    const DeepseekV4Params& p, const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions, const std::vector<float>& previous_hidden,
    const std::vector<int32_t>& logits_indices = {},
    V4MtpMiswire miswire = V4MtpMiswire::kNone);

// The TARGET's pre-hc_head MHC residual stream [T, hc*H] — the state MTP consumes
// as `previous_hidden`. This is the host oracle's final MhcPost output before the
// hc_head collapse (deepseek_v4.cpp::ForwardComposeImpl:1737-1747). Exposed so the
// self-speculation driver (and the lossless gate) can stash it per step.
std::vector<float> DeepseekV4TargetMtpResidualHost(
    const DeepseekV4HostWeights& hw, const DeepseekV4Params& p,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& logits_indices = {});

// Whether a `deepseek4` GGUF actually carries the MTP/nextn tail tensors (block
// index == block_count). Mirrors vLLM's missing-MTP-layer guard (nvidia/mtp.py
// load_weights raises if the checkpoint was quantized WITHOUT the mtp.* layers).
// BOTH shipped DeepSeek-V4-Flash GGUFs return FALSE (the converter dropped nextn,
// though the KV advertises nextn_predict_layers=1) — the engine then cleanly falls
// back to MTP-off. See `.agents/specs/deepseek-v4-mtp.md` §4.
bool DeepseekV4GgufHasMtp(const GgufFile& gguf);

// Per-family config hook (registry `parse_config`): resolves + validates
// DeepseekV4Params and throws on anything unsupported.
void ParseDeepseekV4Config(const HfConfig& config);

// KV-cache spec builder. STUB (W3): V4 uses the fp8_ds_mla UE8M0 576B-paged latent
// KV (attention.py:89, :626) + the DSA indexer/compressor caches — a NEW geometry
// not yet representable. Emits a placeholder MLA spec sized to the compressed
// latent so the arch RESOLVES; the true multi-cache topology is a named W3 residual.
v1::KVCacheConfig MakeDeepseekV4KVCache(const HfConfig& config, int block_size,
                                        int num_blocks);

}  // namespace vllm
