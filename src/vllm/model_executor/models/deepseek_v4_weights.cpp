// DeepSeek-V4-Flash config resolution + checkpoint name-map (W1/W2 scaffolding).
// The header carries the full `file:line`-on-both-sides port map; this TU
// implements `ParseDeepseekV4Params` (config-descent, unit-testable) and the
// `LoadDeepseekV4ForCausalLMWeights` ACCOUNTING pass over the checkpoint's
// verified name-map. Heavy tensor MATERIALIZATION (FP8-block MLA linears, NVFP4
// grouped experts, MHC/DSA towers) is the named W2b/W3-W6 residual.
//
// ─── NAME MAP VERIFIED vs the REAL header (2026-07-28, HTTP range, NO download) ─
// `nvidia/DeepSeek-V4-Flash-NVFP4` model.safetensors.index.json (135,235 tensors,
// 46 shards, total_size 168,266,793,544 B = 156.7 GiB) + shard-2 header. The
// checkpoint uses a FLAT `layers.N.` prefix (no `model.`/mm wrapper); vLLM's
// `_make_deepseek_v4_weights_mapper` (nvidia/model.py:1316) re-prefixes to
// `model.layers.` and fuses wq_a|wkv / compressor.wkv|wgate + w1|w3 -> gate_up.
// We keep the checkpoint's own flat names. VERIFIED dtypes/shapes (layer 0):
//
//   MODEL LEVEL
//     embed.weight               BF16 [129280,4096]      head.weight BF16 [129280,4096]
//     norm.weight                BF16 [4096]
//     hc_head_{base,fn,scale}    F32  (final MHC head collapse, nvidia/model.py:1137)
//
//   PER LAYER N (0..42) — FLAT `layers.N.`
//     attn_norm.weight / ffn_norm.weight            BF16 [4096]
//     hc_attn_{base[24],fn[24,16384],scale[3]}      F32   (MHC: 24=(2+hc_mult)*hc_mult,
//     hc_ffn_{base,fn,scale}                        F32    16384=hc_mult*H) — NEW, no eager ref
//     -- 512-wide MLA (attention.py), FP8-block E4M3 weight + E8M0 [.,.] block scale --
//     attn.wq_a.{weight[1024,4096],scale[8,32]}     q down-proj (q_lora_rank=1024)
//     attn.wq_b.{weight[32768,1024],scale}          q up-proj (64 heads * 512 head_dim)
//     attn.wkv.{weight[512,4096],scale[4,32]}       kv down (fused w/ wq_a upstream)
//     attn.wo_a.{weight[8192,4096],scale}           OUTPUT-LoRA down (o_groups*o_lora=8*1024) NEW
//     attn.wo_b.{weight[4096,8192],scale}           OUTPUT-LoRA up
//     attn.q_norm.weight[1024] / attn.kv_norm.weight[512]   BF16 RMSNorm
//     attn.attn_sink[64]                            F32   per-head attention sink — NEW
//     -- DSA compressor, layers where compress_ratio!=0 (41 layers) --   NEW
//     attn.compressor.{ape, norm.weight, wgate.weight, wkv.weight}
//     -- DSA Lightning-Indexer, layers where compress_ratio==4 (21 layers) --  NEW
//     attn.indexer.compressor.{ape,norm.weight,wgate.weight,wkv.weight}
//     attn.indexer.weights_proj.weight / attn.indexer.wq_b.{weight,scale}
//     -- MoE (nvidia/model.py:512) --
//     ffn.gate.weight[256,4096]  BF16
//     ffn.gate.tid2eid           HASH layers 0,1,2 only (num_hash_layers=3) — NEW
//     ffn.gate.bias              non-hash layers (noaux_tc e_score_correction_bias)
//     ffn.shared_experts.w{1,2,3}.{weight,scale}    FP8-block E4M3 + E8M0 (NOT NVFP4)
//     ffn.experts.E.w{1,2,3}.{weight, weight_scale, weight_scale_2, input_scale}
//                                                   NVFP4: U8-packed [I,H/2] + group-16
//                                                   E4M3 weight_scale + F32 scalar scale_2/input
//   MTP TAIL (`mtp.*`, num_nextn_predict_layers=1) — SKIPPED by the loader,
//   exactly as vLLM's AutoWeightsLoader(skip_substrs=["mtp."]) (nvidia/model.py:1474).
//
// ─── HW-FIT REVERSAL (recorded) ─────────────────────────────────────────────
// total_size = 156.7 GiB. Only the 256 routed experts are NVFP4 (4-bit); the MLA
// + shared-expert linears are FP8 block (`exclude_modules: *.attn.*,
// *.ffn.shared_experts.*, head, mtp.*`), and NVFP4 carries a double scale
// (weight_scale + weight_scale_2) + input_scale per weight. So this "NVFP4"
// checkpoint is ~the same size as the native fp4 (148.7 GiB) and does NOT fit ONE
// GB10's 119 GiB unified pool. The W0 spike's "~83 GiB fits" was a bad estimate.
// W1 (single-GB10 oracle run) is MEMORY-INFEASIBLE — needs multi-node TP / offload.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
#include "vllm/v1/core/kv_cache_utils.h"  // host_available_memory_bytes
#include "vt/dtype.h"

namespace vllm {
namespace {

// --- raw config.json readers (DeepSeek V4 keys are not typed on HfConfig) ---
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_string()) ? f->get<std::string>() : fallback;
}

}  // namespace

DeepseekV4Params ParseDeepseekV4Params(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  DeepseekV4Params p;

  // --- shared geometry (typed fields fall back to raw) ---
  p.hidden_size = config.hidden_size > 0 ? config.hidden_size
                                         : RawInt(raw, "hidden_size", 0);
  p.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(raw, "num_hidden_layers", 0);
  p.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(raw, "vocab_size", 0);
  p.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : RawInt(raw, "num_attention_heads", 0);
  p.num_key_value_heads = RawInt(raw, "num_key_value_heads", 1);
  p.rms_norm_eps = static_cast<float>(RawDouble(raw, "rms_norm_eps", 1e-6));
  p.tie_word_embeddings = RawBool(raw, "tie_word_embeddings", false);
  p.max_position_embeddings = RawInt(raw, "max_position_embeddings", 0);
  p.num_nextn_predict_layers = RawInt(raw, "num_nextn_predict_layers", 0);

  // --- 512-wide MLA geometry (NEW) ---
  p.head_dim =
      config.head_dim > 0 ? config.head_dim : RawInt(raw, "head_dim", 0);
  p.qk_rope_head_dim = RawInt(raw, "qk_rope_head_dim", 64);
  p.q_lora_rank = RawInt(raw, "q_lora_rank", 0);
  p.o_lora_rank = RawInt(raw, "o_lora_rank", 0);
  p.o_groups = RawInt(raw, "o_groups", 0);
  p.sliding_window = RawInt(raw, "sliding_window", 0);
  p.rope_theta = RawDouble(raw, "rope_theta", 10000.0);
  p.compress_rope_theta = RawDouble(raw, "compress_rope_theta", 160000.0);

  // --- MoE ---
  p.n_routed_experts = RawInt(raw, "n_routed_experts", 0);
  p.num_experts_per_tok = RawInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = RawInt(raw, "moe_intermediate_size", 0);
  p.n_shared_experts = RawInt(raw, "n_shared_experts", 0);
  p.norm_topk_prob = RawBool(raw, "norm_topk_prob", true);
  p.routed_scaling_factor = RawDouble(raw, "routed_scaling_factor", 1.0);
  p.swiglu_limit = RawDouble(raw, "swiglu_limit", 0.0);
  p.scoring_func = RawString(raw, "scoring_func", "sqrtsoftplus");
  p.topk_method = RawString(raw, "topk_method", "noaux_tc");
  p.num_hash_layers = RawInt(raw, "num_hash_layers", 0);
  p.expert_dtype = RawString(raw, "expert_dtype", "fp4");

  // --- MHC ---
  p.hc_mult = RawInt(raw, "hc_mult", 0);
  p.hc_sinkhorn_iters = RawInt(raw, "hc_sinkhorn_iters", 0);
  p.hc_eps = RawDouble(raw, "hc_eps", 1e-6);

  // --- DSA ---
  p.index_head_dim = RawInt(raw, "index_head_dim", 0);
  p.index_n_heads = RawInt(raw, "index_n_heads", 0);
  p.index_topk = RawInt(raw, "index_topk", 0);
  if (const nlohmann::json* cr = Field(raw, "compress_ratios");
      cr != nullptr && cr->is_array()) {
    for (const auto& v : *cr)
      p.compress_ratios.push_back(v.is_number() ? v.get<int64_t>() : 0);
  }

  // --- validation (throw with a precise message on anything unrepresentable) ---
  VT_CHECK(p.hidden_size > 0, "deepseek-v4: hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0,
           "deepseek-v4: num_hidden_layers must be positive");
  VT_CHECK(p.n_routed_experts > 0,
           "deepseek-v4: n_routed_experts must be positive (this is a MoE arch)");
  VT_CHECK(p.hc_mult > 0,
           "deepseek-v4: hc_mult must be positive — Manifold Hyper-Connections "
           "are structural to V4 (config.raw.hc_mult missing?)");
  VT_CHECK(p.head_dim == 512,
           "deepseek-v4: only the 512-wide MLA geometry (448 NoPE + 64 RoPE) is "
           "scoped; got head_dim=" + std::to_string(p.head_dim));
  VT_CHECK(p.scoring_func == "sqrtsoftplus",
           "deepseek-v4: only scoring_func='sqrtsoftplus' is scoped; got '" +
               p.scoring_func + "'");
  VT_CHECK(p.expert_dtype == "fp4" || p.expert_dtype == "fp8",
           "deepseek-v4: expert_dtype must be 'fp4' (NVFP4/MXFP4) or 'fp8'; got '" +
               p.expert_dtype + "'");
  return p;
}

void ParseDeepseekV4Config(const HfConfig& config) {
  // The resolve itself IS the validation (throws on every unsupported field).
  (void)ParseDeepseekV4Params(config);
}

// ════════════════════════════════════════════════════════════════════════════
// MODEL-DSV4-EXL3 W1b — the rank-sliced EXL3 loader arm.
//
// Detected off `quantization_config.quant_method == "exl3"` +
// `version == "rank-sliced-deepseek-v4-v1"` (the SparkInfer artifact
// `0xSero/deepseek-v4-flash-0731-spark`). Two tensor populations arrive:
//
//   EXL3   `layers.{L}.ffn.experts.{E}.{w1|w2|w3}.rank{r}.{trellis|suh|svh|mcg}`
//          — the routed experts, PRE-SLICED across `tp` ranks. Coalesced back to
//          TP1 here by inverting `LinearEXL3.tp_import_split`
//          (exllamav3 @ 2398c056, `modules/quant/exl3.py:296-313`).
//   CARRIED everything else, by the SAME names the FP8/BF16 checkpoint uses.
//          Accounted through the identical name-map the non-EXL3 arm walks —
//          minus the routed-expert block EXL3 replaced. Materializing those
//          towers is the pre-existing W2b residual of deepseek-v4-flash.md, not
//          something this row changes.
//
// REFUSE BY NAME is the rule for everything else: an unknown schema version, a
// codebook other than mcg, a missing rank tensor, a slice on the wrong axis, a
// carried tensor no arm routes. Nothing is improvised.
//
// RESIDENCY, named honestly: the coalesced tower is COPIED into host owner
// buffers. That is right for the fixture and for W2's byte-parity gate, and it
// is ~100 GB on the real 216-expert artifact — real-checkpoint residency
// (borrow / device-resident / per-layer streaming) is owed to W2 and is listed
// under `.agents/specs/model-dsv4-exl3.md` `## Owed`.
namespace {

constexpr const char* kExl3Row = "MODEL-DSV4-EXL3";

const nlohmann::json* QuantConfig(const HfConfig& config) {
  return Field(config.raw, "quantization_config");
}

bool IsExl3Checkpoint(const HfConfig& config) {
  const nlohmann::json* qc = QuantConfig(config);
  if (qc == nullptr || !qc->is_object()) return false;
  return RawString(*qc, "quant_method", "") == "exl3";
}

// Process-cached read of `VT_DSV4_EXL3_HOST_BUDGET` (default ON; a '0'-leading
// value disables the host-residency refusal). The parse itself lives in the
// header as `Dsv4Exl3HostBudgetFlagIsOn` so it is unit-testable without touching
// the environment; only the one getenv is here, read once per process the way
// every other `VT_*` knob on this model is.
bool Dsv4Exl3HostBudgetEnabled() {
  static const bool on =
      Dsv4Exl3HostBudgetFlagIsOn(std::getenv("VT_DSV4_EXL3_HOST_BUDGET"));
  return on;
}

// One tensor, wherever it lives among the shards.
using StIndex = std::unordered_map<std::string, const StTensor*>;

StIndex IndexShards(const std::vector<SafetensorsFile>& shards) {
  StIndex index;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) index.emplace(name, &shard.Get(name));
  return index;
}

const StTensor* RequireTensor(const StIndex& index, const std::string& name) {
  const auto it = index.find(name);
  VT_CHECK(it != index.end(),
           std::string("deepseek-v4 exl3 loader: expected checkpoint tensor missing: ") +
               name + " (" + kExl3Row + " W1b)");
  return it->second;
}

void RequireDtype(const StTensor& t, const char* want, const std::string& name) {
  VT_CHECK(t.dtype == want,
           std::string("deepseek-v4 exl3 loader: ") + name + " must be " + want +
               ", got " + t.dtype + " (" + kExl3Row + " W1b)");
}

std::vector<uint16_t> ReadU16(const StTensor& t, const std::string& name) {
  VT_CHECK(t.nbytes % sizeof(uint16_t) == 0,
           std::string("deepseek-v4 exl3 loader: ") + name + " is not 16-bit aligned");
  std::vector<uint16_t> v(t.nbytes / sizeof(uint16_t));
  if (!v.empty()) std::memcpy(v.data(), t.data, t.nbytes);
  return v;
}

// The four tensors of ONE rank slice of ONE EXL3 linear.
struct Exl3RankSlice {
  std::vector<uint16_t> trellis, suh, svh;
  int64_t tiles_k = 0, tiles_n = 0;
  int32_t mcg = 0;
};

Exl3RankSlice ReadRankSlice(const StIndex& index, const std::string& base, int bits,
                            int64_t* consumed) {
  Exl3RankSlice s;
  const StTensor& tr = *RequireTensor(index, base + ".trellis");
  const StTensor& su = *RequireTensor(index, base + ".suh");
  const StTensor& sv = *RequireTensor(index, base + ".svh");
  const StTensor& mc = *RequireTensor(index, base + ".mcg");
  *consumed += 4;
  RequireDtype(tr, "I16", base + ".trellis");
  RequireDtype(su, "F16", base + ".suh");
  RequireDtype(sv, "F16", base + ".svh");
  RequireDtype(mc, "I32", base + ".mcg");
  VT_CHECK(tr.shape.size() == 3,
           std::string("deepseek-v4 exl3 loader: ") + base +
               ".trellis must be 3-D [k/16, n/16, 16*bits] (exl3.py:47), got rank " +
               std::to_string(tr.shape.size()) + " (" + kExl3Row + " W1b)");
  VT_CHECK(tr.shape[2] == 16LL * bits,
           std::string("deepseek-v4 exl3 loader: ") + base + ".trellis last dim is " +
               std::to_string(tr.shape[2]) + ", which is not 16*bits for bits=" +
               std::to_string(bits) + " (" + kExl3Row + " W1b)");
  VT_CHECK(su.shape.size() == 1 && sv.shape.size() == 1,
           std::string("deepseek-v4 exl3 loader: ") + base +
               " suh/svh must be 1-D (exl3.py:48-49) (" + kExl3Row + " W1b)");
  VT_CHECK(su.shape[0] == tr.shape[0] * 16 && sv.shape[0] == tr.shape[1] * 16,
           std::string("deepseek-v4 exl3 loader: ") + base +
               " suh/svh lengths do not match the trellis tile grid (suh=" +
               std::to_string(su.shape[0]) + " svh=" + std::to_string(sv.shape[0]) +
               " trellis=[" + std::to_string(tr.shape[0]) + "," +
               std::to_string(tr.shape[1]) + "]) (" + kExl3Row + " W1b)");
  s.tiles_k = tr.shape[0];
  s.tiles_n = tr.shape[1];
  s.trellis = ReadU16(tr, base + ".trellis");
  s.suh = ReadU16(su, base + ".suh");
  s.svh = ReadU16(sv, base + ".svh");
  VT_CHECK(mc.nbytes == sizeof(int32_t),
           std::string("deepseek-v4 exl3 loader: ") + base +
               ".mcg must be one int32 (" + kExl3Row + " W1b)");
  std::memcpy(&s.mcg, mc.data, sizeof(int32_t));
  return s;
}

// Invert `tp_import_split` (exl3.py:296-313). `split_out` is the w1/w3 case: the
// ranks each hold the WHOLE suh and a slice of svh + trellis dim 1. The w2 case
// is the mirror: whole svh, sliced suh + trellis dim 0.
DeepseekV4Exl3Linear CoalesceExl3Linear(const StIndex& index, const std::string& base,
                                        int tp, int bits, bool split_out,
                                        int64_t want_in, int64_t want_out,
                                        int64_t* consumed) {
  std::vector<Exl3RankSlice> ranks;
  ranks.reserve(static_cast<size_t>(tp));
  for (int r = 0; r < tp; ++r)
    ranks.push_back(ReadRankSlice(index, base + ".rank" + std::to_string(r), bits,
                                  consumed));

  const int64_t words = 16LL * bits;
  DeepseekV4Exl3Linear lin;
  lin.bits = bits;
  lin.mcg = ranks[0].mcg;

  // The INVARIANT side must be byte-identical on every rank — each rank received
  // the whole vector, so a difference means the slicing axis is not what the
  // schema says it is.
  const auto require_invariant = [&](const std::vector<uint16_t>& a,
                                     const std::vector<uint16_t>& b, int r,
                                     const char* which) {
    VT_CHECK(a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(),
                                                               a.size() * sizeof(uint16_t)) == 0),
             std::string("deepseek-v4 exl3 loader: ") + base + " " + which +
                 " differs on rank" + std::to_string(r) +
                 " but this projection is declared " +
                 (split_out ? "OUT-split" : "IN-split") +
                 ", where it is replicated whole (exl3.py:296-313) — the slice axis "
                 "does not match the schema (" + kExl3Row + " W1b)");
  };

  if (split_out) {
    lin.suh = ranks[0].suh;
    int64_t tiles_n = 0;
    for (int r = 0; r < tp; ++r) {
      VT_CHECK(ranks[r].tiles_k == ranks[0].tiles_k,
               std::string("deepseek-v4 exl3 loader: ") + base +
                   " rank" + std::to_string(r) +
                   " disagrees on the IN tile count (" + kExl3Row + " W1b)");
      require_invariant(ranks[0].suh, ranks[r].suh, r, "suh");
      tiles_n += ranks[r].tiles_n;
    }
    const int64_t tiles_k = ranks[0].tiles_k;
    lin.in_features = tiles_k * 16;
    lin.out_features = tiles_n * 16;
    lin.trellis.assign(static_cast<size_t>(tiles_k * tiles_n * words), 0);
    lin.svh.reserve(static_cast<size_t>(lin.out_features));
    int64_t off_n = 0;
    for (int r = 0; r < tp; ++r) {
      const int64_t rtn = ranks[r].tiles_n;
      for (int64_t i = 0; i < tiles_k; ++i) {
        std::memcpy(&lin.trellis[static_cast<size_t>((i * tiles_n + off_n) * words)],
                    &ranks[r].trellis[static_cast<size_t>(i * rtn * words)],
                    static_cast<size_t>(rtn * words) * sizeof(uint16_t));
      }
      lin.svh.insert(lin.svh.end(), ranks[r].svh.begin(), ranks[r].svh.end());
      off_n += rtn;
    }
  } else {
    lin.svh = ranks[0].svh;
    int64_t tiles_k = 0;
    for (int r = 0; r < tp; ++r) {
      VT_CHECK(ranks[r].tiles_n == ranks[0].tiles_n,
               std::string("deepseek-v4 exl3 loader: ") + base +
                   " rank" + std::to_string(r) +
                   " disagrees on the OUT tile count (" + kExl3Row + " W1b)");
      require_invariant(ranks[0].svh, ranks[r].svh, r, "svh");
      tiles_k += ranks[r].tiles_k;
    }
    const int64_t tiles_n = ranks[0].tiles_n;
    lin.in_features = tiles_k * 16;
    lin.out_features = tiles_n * 16;
    lin.trellis.reserve(static_cast<size_t>(tiles_k * tiles_n * words));
    lin.suh.reserve(static_cast<size_t>(lin.in_features));
    for (int r = 0; r < tp; ++r) {
      // trellis dim 0 is outermost, so an IN split concatenates verbatim.
      lin.trellis.insert(lin.trellis.end(), ranks[r].trellis.begin(),
                         ranks[r].trellis.end());
      lin.suh.insert(lin.suh.end(), ranks[r].suh.begin(), ranks[r].suh.end());
    }
  }

  VT_CHECK(lin.in_features == want_in && lin.out_features == want_out,
           std::string("deepseek-v4 exl3 loader: ") + base + " coalesces to [" +
               std::to_string(lin.in_features) + ", " +
               std::to_string(lin.out_features) + "] but the config says [" +
               std::to_string(want_in) + ", " + std::to_string(want_out) +
               "] — the rank slices do not reassemble this projection (" + kExl3Row +
               " W1b)");
  // Both sides were Hadamard-128 transformed at quantization time
  // (exl3_lib/quantize.py:15), so a reassembly that is not 128-divisible cannot
  // be dequantized at all.
  VT_CHECK(lin.in_features % 128 == 0 && lin.out_features % 128 == 0,
           std::string("deepseek-v4 exl3 loader: ") + base +
               " reassembles to features that are not multiples of the Hadamard "
               "block size 128 (" + kExl3Row + " W1b)");
  return lin;
}

}  // namespace

int64_t DeepseekV4Exl3ResidentBytes(const DeepseekV4Weights& weights) {
  int64_t bytes = 0;
  for (const DeepseekV4Exl3LayerWeights& l : weights.exl3.layers)
    for (const DeepseekV4Exl3Expert& e : l.experts)
      bytes += e.w1.Bytes() + e.w2.Bytes() + e.w3.Bytes();
  return bytes;
}

int64_t ReportDeepseekV4Exl3Residency(const DeepseekV4Weights& weights,
                                      int64_t layers_done, int64_t layers_total,
                                      int64_t host_available_bytes) {
  const int64_t tower = DeepseekV4Exl3ResidentBytes(weights);
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const auto gib = [](int64_t b) { return static_cast<double>(b) / kGiB; };

  // The projection is EXACT for this schema rather than an extrapolation: every
  // MoE layer of the rank-sliced artifact carries the same `n_routed_experts` x
  // 3 projections at the same two shapes, so one loaded layer prices all of
  // them. It is checked per layer so the refusal lands at the FIRST layer that
  // cannot fit rather than after the whole tower is committed.
  const int64_t projected =
      layers_done > 0 ? tower / layers_done * layers_total : 0;

  // An UNKNOWN budget never refuses. `host_available_memory_bytes()` returns 0
  // when /proc/meminfo is unreadable, and an unknown budget must not become a
  // false refusal — the polarity `check_enough_state_memory` keeps for the
  // recurrent-state budget (`host_available_memory_bytes` /
  // `check_enough_state_memory` in `vllm/v1/core/kv_cache_utils.cpp`, issue
  // #371). `VT_DSV4_EXL3_HOST_BUDGET=0` reaches this same branch on purpose.
  VT_CHECK(
      host_available_bytes <= 0 || projected <= host_available_bytes,
      std::string("deepseek-v4 exl3 loader: the coalesced EXL3 tower does not "
                  "fit host memory. Layer ") +
          std::to_string(layers_done) + " of " + std::to_string(layers_total) +
          " already holds " + std::to_string(gib(tower)) +
          " GiB, which prices the whole tower at " + std::to_string(gib(projected)) +
          " GiB against MemAvailable " + std::to_string(gib(host_available_bytes)) +
          " GiB. This arm COPIES the TP1-coalesced tower into host owner buffers; "
          "a device-resident / per-layer-streaming destination is owed to " +
          kExl3Row +
          " W2 (see `.agents/specs/model-dsv4-exl3.md` `## Owed`). Refusing "
          "before the allocation takes the box down. MemAvailable is an ESTIMATE "
          "that ignores swap and, inside a container, reports the HOST pool "
          "rather than the cgroup limit; set VT_DSV4_EXL3_HOST_BUDGET=0 to "
          "proceed anyway in this same binary.");

  // Reported once, when the tower is complete. Residency is the one number a
  // reader cannot get any other way on this arm: the real artifact's trellis
  // alone is ~83.5 GiB, on a unified-memory box where an over-commit reboots
  // the machine instead of failing.
  if (layers_done >= layers_total) {
    if (host_available_bytes > 0) {
      std::fprintf(stderr,
                   "[vt load] dsv4-exl3: coalesced TP1 tower resident_bytes=%lld "
                   "(%.3f GiB) over %lld layers, tp%d->tp1, %d-bit trellis; host "
                   "MemAvailable=%.3f GiB\n",
                   static_cast<long long>(tower), gib(tower),
                   static_cast<long long>(layers_total), weights.exl3.tp,
                   weights.exl3.bits, gib(host_available_bytes));
    } else {
      std::fprintf(stderr,
                   "[vt load] dsv4-exl3: coalesced TP1 tower resident_bytes=%lld "
                   "(%.3f GiB) over %lld layers, tp%d->tp1, %d-bit trellis; host "
                   "MemAvailable unknown (/proc/meminfo unreadable, or the "
                   "refusal disabled by VT_DSV4_EXL3_HOST_BUDGET=0), so nothing "
                   "was refused\n",
                   static_cast<long long>(tower), gib(tower),
                   static_cast<long long>(layers_total), weights.exl3.tp,
                   weights.exl3.bits);
    }
  }
  return tower;
}

namespace {

DeepseekV4Weights LoadDeepseekV4Exl3(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config,
                                     const DeepseekV4Params& p) {
  const nlohmann::json& qc = *QuantConfig(config);
  const std::string version = RawString(qc, "version", "");
  VT_CHECK(version == "rank-sliced-deepseek-v4-v1",
           std::string("deepseek-v4 exl3 loader: unsupported quantization_config."
                       "version '") + version +
               "'; only 'rank-sliced-deepseek-v4-v1' is implemented (" + kExl3Row +
               " W1b). A new schema needs its own row.");
  const std::string codebook = RawString(qc, "codebook", "");
  VT_CHECK(codebook == "mcg",
           std::string("deepseek-v4 exl3 loader: unsupported EXL3 codebook '") +
               codebook +
               "'; only 'mcg' (cb=1, codebook.cuh:67-75) is decoded. The mul1 (cb=2) "
               "and cb=0 codebooks are owed to " + kExl3Row + " W2.");
  const double bits_raw = RawDouble(qc, "bits", 0.0);
  const int bits = static_cast<int>(bits_raw);
  VT_CHECK(bits >= 1 && bits <= 8 && static_cast<double>(bits) == bits_raw,
           std::string("deepseek-v4 exl3 loader: quantization_config.bits must be a "
                       "whole number in [1, 8]; got ") +
               std::to_string(bits_raw) + " (" + kExl3Row + " W1b)");

  // `tp` lives beside the tensor schema, not in quantization_config.
  const nlohmann::json* tail = Field(config.raw, "hybrid_tr3_tail");
  const int declared_tp =
      (tail != nullptr && tail->is_object())
          ? static_cast<int>(RawInt(*tail, "tp", 0))
          : 0;
  VT_CHECK(declared_tp >= 1,
           std::string("deepseek-v4 exl3 loader: hybrid_tr3_tail.tp is missing or not "
                       "positive — the rank-sliced schema cannot be reassembled "
                       "without the tensor-parallel width (") + kExl3Row + " W1b)");

  const StIndex index = IndexShards(shards);

  DeepseekV4Weights w;
  w.params = p;
  w.exl3.tp = declared_tp;
  w.exl3.bits = bits;
  w.exl3.codebook = codebook;
  w.exl3.version = version;
  w.has_exl3_weights = true;

  int64_t accounted = 0;
  int64_t skipped_mtp = 0;

  // ── the CARRIED half: the same name-map the non-EXL3 arm walks, minus the
  //    routed-expert block EXL3 replaced. ─────────────────────────────────────
  std::unordered_set<std::string> routed;
  const auto require = [&](const std::string& name) {
    (void)RequireTensor(index, name);
    routed.insert(name);
    ++accounted;
  };
  require("embed.weight");
  require("norm.weight");
  if (!p.tie_word_embeddings) require("head.weight");
  for (const char* s : {"hc_head_base", "hc_head_fn", "hc_head_scale"}) require(s);
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    require(b + "attn_norm.weight");
    require(b + "ffn_norm.weight");
    for (const char* h : {"hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base",
                          "hc_ffn_fn", "hc_ffn_scale"})
      require(b + h);
    const std::string a = b + "attn.";
    for (const char* wn : {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}) {
      require(a + wn + ".weight");
      require(a + wn + ".scale");
    }
    require(a + "q_norm.weight");
    require(a + "kv_norm.weight");
    require(a + "attn_sink");
    if (p.has_compressor(l))
      for (const char* c : {"ape", "norm.weight", "wgate.weight", "wkv.weight"})
        require(a + "compressor." + c);
    if (p.has_indexer(l)) {
      for (const char* c : {"ape", "norm.weight", "wgate.weight", "wkv.weight"})
        require(a + "indexer.compressor." + c);
      require(a + "indexer.weights_proj.weight");
      require(a + "indexer.wq_b.weight");
      require(a + "indexer.wq_b.scale");
    }
    const std::string f = b + "ffn.";
    require(f + "gate.weight");
    if (p.is_hash_layer(l))
      require(f + "gate.tid2eid");
    else
      require(f + "gate.bias");
    for (const char* wn : {"w1", "w2", "w3"}) {
      require(f + "shared_experts." + wn + ".weight");
      require(f + "shared_experts." + wn + ".scale");
    }
  }

  // ── the EXL3 half: coalesce every routed expert back to TP1. ───────────────
  // The budget is read ONCE, before the first copy: MemAvailable falls as this
  // loop allocates, so re-reading it mid-load would compare the tower against a
  // pool the tower itself has already drained.
  //
  // WHAT THIS NUMBER IS AND IS NOT. It is `/proc/meminfo` MemAvailable, the
  // kernel's own estimate of what can be handed out without swapping. It is an
  // ESTIMATE, and it is wrong in BOTH directions here:
  //   (a) it ignores swap and under-counts some reclaimable pages, so a tower
  //       this host could in fact have held can still be refused; and
  //   (b) inside a container it reports the HOST's figure, not the cgroup's
  //       limit, so a memory-capped container gets NO protection from this
  //       refusal while the logged budget names a pool the process cannot draw
  //       on.
  // Neither is fixable from inside this loader — a cgroup-aware budget is its
  // own row — so the refusal ships with a same-binary escape hatch:
  // `VT_DSV4_EXL3_HOST_BUDGET=0` hands the reporter an UNKNOWN budget (0), which
  // never refuses. Default is the refusal ENABLED, because on the unified-memory
  // box this arm targets an over-commit reboots the machine rather than failing.
  const int64_t host_available =
      Dsv4Exl3HostBudgetEnabled() ? vllm::v1::host_available_memory_bytes() : 0;
  w.exl3.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4Exl3LayerWeights& lw = w.exl3.layers[static_cast<size_t>(l)];
    lw.experts.resize(static_cast<size_t>(p.n_routed_experts));
    for (int64_t e = 0; e < p.n_routed_experts; ++e) {
      const std::string base = "layers." + std::to_string(l) + ".ffn.experts." +
                               std::to_string(e) + ".";
      DeepseekV4Exl3Expert& ex = lw.experts[static_cast<size_t>(e)];
      int64_t consumed = 0;
      // w1 and w3 are the gate/up projections [moe_inter, hidden] — OUT-split.
      // w2 is the down projection [hidden, moe_inter] — IN-split.
      ex.w1 = CoalesceExl3Linear(index, base + "w1", declared_tp, bits,
                                 /*split_out=*/true, p.hidden_size,
                                 p.moe_intermediate_size, &consumed);
      ex.w3 = CoalesceExl3Linear(index, base + "w3", declared_tp, bits,
                                 /*split_out=*/true, p.hidden_size,
                                 p.moe_intermediate_size, &consumed);
      ex.w2 = CoalesceExl3Linear(index, base + "w2", declared_tp, bits,
                                 /*split_out=*/false, p.moe_intermediate_size,
                                 p.hidden_size, &consumed);
      accounted += consumed;
      for (int r = 0; r < declared_tp; ++r)
        for (const char* proj : {"w1", "w2", "w3"})
          for (const char* suf : {".trellis", ".suh", ".svh", ".mcg"})
            routed.insert(base + proj + ".rank" + std::to_string(r) + suf);
    }
    // Price what has been committed, refuse a tower this host cannot hold, and
    // report the figure once the last layer is in. This is the only production
    // reader of `DeepseekV4Exl3ResidentBytes`.
    (void)ReportDeepseekV4Exl3Residency(w, l + 1, p.num_hidden_layers,
                                        host_available);
  }

  // ── totality: every checkpoint tensor is routed or explicitly skipped. ─────
  // vLLM's DeepSeek-V4 loader skips the MTP tail wholesale
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474) and so do
  // we; anything else left over is a layout this arm does not implement and is
  // REFUSED BY NAME rather than silently ignored.
  for (const auto& [name, tensor] : index) {
    (void)tensor;
    if (routed.count(name) != 0) continue;
    if (name.rfind("mtp.", 0) == 0) {
      // MEASURED on the complete 190-file artifact (2026-08-24): the three MTP
      // layers carry `mtp.{L}.ffn.experts.{0..215}.{w1,w2,w3}.weight` as I8
      // [2048, 2048] + `.scale` F8_E8M0 [2048, 128] — NVFP4 e2m1 packed
      // two-per-byte with ue8m0 block scales, i.e. the config's
      // `packed_e2m1_fp4_with_ue8m0_scales`. The MTP experts were NOT
      // requantized to EXL3; only the main model's were. Their presence is why
      // the upstream repo can run a K5 speculative draft at all, and reaching
      // them is a later row's work.
      ++skipped_mtp;
      continue;
    }
    VT_CHECK(false,
             std::string("deepseek-v4 exl3 loader: checkpoint tensor no arm routes: ") +
                 name +
                 " — this loader implements the rank-sliced EXL3 experts + the "
                 "carried deepseek_v4_fp8 name-map only. Refusing rather than "
                 "improvising a layout (" + kExl3Row + " W1b).");
  }

  w.exl3.skipped_mtp_tensors = skipped_mtp;
  w.accounted_tensors = accounted;
  return w;
}

}  // namespace

DeepseekV4Weights LoadDeepseekV4ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  const DeepseekV4Params p = ParseDeepseekV4Params(config);

  // MODEL-DSV4-EXL3 W1b: the rank-sliced EXL3 artifact takes its own arm. Its
  // routed experts are trellis-quantized and pre-split across `tp` ranks, so the
  // routed-expert block of the name-map below does not apply to it.
  if (IsExl3Checkpoint(config)) return LoadDeepseekV4Exl3(shards, config, p);

  // The full checkpoint name set (for the W2 accounting pass).
  std::unordered_set<std::string> have;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) have.insert(name);

  int64_t accounted = 0;
  const auto require = [&](const std::string& name) {
    VT_CHECK(have.count(name) != 0,
             "deepseek-v4 loader: expected checkpoint tensor missing: " + name);
    ++accounted;
  };

  // NVFP4 experts carry a double weight-scale (weight_scale + weight_scale_2) +
  // an input_scale; FP8-block experts carry a single `.scale`. Verified: the
  // `expert_dtype=fp4` NVFP4 vehicle uses the former.
  const std::vector<std::string> expert_suffixes =
      (p.expert_dtype == "fp4")
          ? std::vector<std::string>{".weight", ".weight_scale",
                                     ".weight_scale_2", ".input_scale"}
          : std::vector<std::string>{".weight", ".scale"};

  // --- model level ---
  require("embed.weight");
  require("norm.weight");
  if (!p.tie_word_embeddings) require("head.weight");
  for (const char* s : {"hc_head_base", "hc_head_fn", "hc_head_scale"}) require(s);

  // --- per layer (flat `layers.N.`) ---
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    require(b + "attn_norm.weight");
    require(b + "ffn_norm.weight");
    for (const char* h : {"hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                          "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"})
      require(b + h);

    // 512-wide MLA (FP8-block weight + block scale).
    const std::string a = b + "attn.";
    for (const char* w : {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}) {
      require(a + w + ".weight");
      require(a + w + ".scale");
    }
    require(a + "q_norm.weight");
    require(a + "kv_norm.weight");
    require(a + "attn_sink");

    // DSA compressor (compress_ratio != 0) + Lightning-Indexer (== 4).
    if (p.has_compressor(l)) {
      for (const char* c : {"ape", "norm.weight", "wgate.weight", "wkv.weight"})
        require(a + "compressor." + c);
    }
    if (p.has_indexer(l)) {
      for (const char* c : {"ape", "norm.weight", "wgate.weight", "wkv.weight"})
        require(a + "indexer.compressor." + c);
      require(a + "indexer.weights_proj.weight");
      require(a + "indexer.wq_b.weight");
      require(a + "indexer.wq_b.scale");
    }

    // MoE gate: hash layers carry `tid2eid` (no bias); others carry the
    // noaux_tc `bias` (e_score_correction_bias).
    const std::string f = b + "ffn.";
    require(f + "gate.weight");
    if (p.is_hash_layer(l))
      require(f + "gate.tid2eid");
    else
      require(f + "gate.bias");

    // Shared expert (FP8-block).
    for (const char* w : {"w1", "w2", "w3"}) {
      require(f + "shared_experts." + w + ".weight");
      require(f + "shared_experts." + w + ".scale");
    }

    // 256 routed experts (NVFP4).
    for (int64_t e = 0; e < p.n_routed_experts; ++e) {
      const std::string ep = f + "experts." + std::to_string(e) + ".";
      for (const char* w : {"w1", "w2", "w3"})
        for (const std::string& suf : expert_suffixes) require(ep + w + suf);
    }
  }

  // TODO(W2b): materialize the accounted towers into device OwnedTensors —
  //   * FP8-block MLA linears (wq_a/wq_b/wkv/wo_a/wo_b) + E8M0 block scales:
  //     reuse the fp8 block loaders + cuda_scaled_mm_c3x_sm100.
  //   * NVFP4 grouped experts (U8 w13/w2 + group-16 E4M3 weight_scale + F32
  //     weight_scale_2/input_scale): reuse src/vt/cuda/cuda_matmul_nvfp4_sm100.cu
  //     + the nvfp4 tactics for the FusedMoE-fallback path (MegaMoE is SM100-only,
  //     nvidia/model.py:307 — GB10 uses the fallback).
  //   * MHC mixing matrices (hc_*_{base,fn,scale}) + DSA indexer/compressor
  //     towers: NEW primitives, ported at W3-W6 (see deepseek-v4-flash.md §5).

  DeepseekV4Weights w;
  w.params = p;
  w.accounted_tensors = accounted;
  return w;
}

// ════════════════════════════════════════════════════════════════════════════
// W2b — the `deepseek4` GGUF keep-quant TOWER materialization.
//
// Wires the LANDED `blk.N.*` -> V4 name map (scripts/check-dsv4-gguf-namemap.py,
// EXACT 1328/1328 coverage) + the LANDED keep-quant vec_dot (IQ2_XXS/IQ3_XXS/Q2_K,
// CIQ) into the DeepseekV4 weight towers, so `unsloth/DeepSeek-V4-Flash-GGUF
// UD-IQ2_XXS` (~91 GiB, the only single-GB10-fitting build) LOADS. MW/SEW roles
// KEEP their ~2-3-bit blocks COMPRESSED (OwnGgufQuantBlocks, the memory enabler —
// dequant-to-bf16 would need ~316 GiB and OOM-reboot the unified pool); the small
// V/ET/HASH tensors dequant to f32/bf16 exactly as our other GGUF loaders do
// (qwen3_5_gguf_weights.cpp is the structural mirror). The load ALSO dequants the
// tiny-config CPU composition tower (`host`) so a loaded model can Forward.
//
// HONEST 3-state: the tiny-synthetic load->forward here is DERIVED + BUILD-VERIFIED
// (structural, tiny shape, test_deepseek_v4_gguf_load.cpp). The REAL 91 GiB
// checkpoint load + generate stays W8-final (operational: download + DGX) — the
// name-map coverage vs the real 1328-tensor manifest is gated separately
// (check-dsv4-gguf-namemap.py rc=0).
namespace {

// --- GGUF KV metadata readers (mirror qwen3_5_gguf_weights.cpp KvInt/KvFloat) ---
int64_t GgKvInt(const GgufValue& v, const std::string& key) {
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
      throw std::runtime_error("deepseek-v4 gguf: key " + key + " is not an integer");
  }
}
double GgKvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(GgKvInt(v, key));
}
int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "deepseek-v4 gguf: missing metadata key " + key);
  return GgKvInt(*v, key);
}
int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? GgKvInt(*v, key) : dflt;
}
double OptFloat(const GgufFile& g, const std::string& key, double dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? GgKvFloat(*v, key) : dflt;
}

std::string Blk(int64_t l, const std::string& suffix) {
  return "blk." + std::to_string(l) + "." + suffix;
}

bool HasGgufTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors())
    if (t.name == name) return true;
  return false;
}

// --- host-tower dequant bridge (torch [out,in] row-major f32) ----------------
// Reads the RAW ggml bytes (Q8_0/IQ2_XXS/IQ3_XXS/F32 are self-contained — no
// sidecar scale), independent of the keep-quant OwnedTensor residency.
std::vector<float> DqRowF32(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo& t = g.Get(name);
  int64_t numel = 1;
  for (int64_t d : t.shape) numel *= d;
  return DequantGgufRowToF32(t.ggml_type, t.data, numel);
}

// --- tower materializers (route once, account, keep-quant or expand) ---------
OwnedTensor MakeBf16Owned(const std::vector<uint16_t>& dq,
                          const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  VT_CHECK(static_cast<int64_t>(dq.size()) == n, "deepseek-v4 gguf: bf16 size mismatch");
  o.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  std::memcpy(o.bytes.data(), dq.data(), o.bytes.size());
  o.nk = nk;
  return o;
}
OwnedTensor MakeF32Owned(const std::vector<float>& dq,
                         const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = vt::DType::kF32;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  VT_CHECK(static_cast<int64_t>(dq.size()) == n, "deepseek-v4 gguf: f32 size mismatch");
  o.bytes.resize(static_cast<size_t>(n) * sizeof(float));
  std::memcpy(o.bytes.data(), dq.data(), o.bytes.size());
  return o;
}

// Brick 4 (DeepSeek-V4 last-mile): opt-in CUDA coalesced-Q8_0 repack at load
// (the MLA q/kv/o-LoRA/shared-expert/lm_head Q8_0 tower — ~6.1 GiB, deinterleaved
// off the mmap into owned aligned buffers with the source file pages dropped, so
// net-resident is ~flat). Default OFF (`VT_V4_Q8_0_ALIGN=1` to enable). CUDA-only:
// the aligned layout is consumed only by the CUDA Q8_0 GEMM.
inline bool EnvQ8_0CudaAlign() {
  const char* v = std::getenv("VT_V4_Q8_0_ALIGN");
  return v != nullptr && !(std::strcmp(v, "") == 0 || std::strcmp(v, "0") == 0 ||
                           std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}

// A per-load routing context: the file, the residency policy, and the set of
// consumed tensor names (the totality/accounting contract — every routed tensor
// is recorded, and a leftover unroutes-to-fail at the end).
struct V4GgufCtx {
  const GgufFile& g;
  const GgufLoadPolicy& pol;
  std::unordered_set<std::string> consumed;

  const GgufTensorInfo& Take(const std::string& name) {
    const GgufTensorInfo& t = g.Get(name);  // throws (unmapped/missing -> FAIL)
    consumed.insert(name);
    return t;
  }
  // 2-D [out,in] matmul weight: keep its blocks (MW keep-quant) else expand bf16,
  // both in the file's own [N,K] order (nk=true) — GGUF disk order IS MatmulBT.
  // `allow_align`: gate the Brick-4 CUDA coalesced-Q8_0 repack for THIS tensor.
  // Must be false for a weight that is later read via a block ROW-SLICE (wo_a, the
  // per-o-group GemmRowSliceInto) — the aligned layout is per-tensor deinterleaved,
  // so a row-offset sub-pointer would address the wrong section. Full-tensor GEMM
  // weights (wq_a/wq_b/wkv/wo_b/shared_*/lm_head) align safely.
  OwnedTensor Mw(const std::string& name, bool allow_align = true) {
    const GgufTensorInfo& t = Take(name);
    VT_CHECK(t.shape.size() == 2, "deepseek-v4 gguf: expected 2-D MW " + name);
    const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
    if (r != GgufResidency::kExpandBf16) {
      // mmap-VIEW the blocks when the policy allows (borrow in place out of the
      // file's read-only mapping, refcounted) instead of COPYING them into an owned
      // buffer — the ~91 GiB keep-quant tower then shares the file's page cache
      // rather than doubling it (measured on the real model: ~116 GiB copy -> the
      // mmap-resident image). Mirrors qwen3_5_gguf_weights.cpp:214 (MmapSrc).
      return OwnGgufQuantBlocks(t, t.shape[0], t.shape[1], /*row_offset=*/0,
                                pol.mmap_residency ? &g : nullptr, pol.quant_repack,
                                /*cuda_align=*/allow_align && EnvQ8_0CudaAlign());
    }
    return MakeBf16Owned(DequantGgufRowToBf16(t.ggml_type, t.data, t.shape[0] * t.shape[1]),
                         {t.shape[0], t.shape[1]}, /*nk=*/true);
  }
  // Stacked [E,out,in] expert weight: KEEP the whole block slab COMPRESSED (each
  // expert = out whole rows = whole blocks, so E*out rows is one contiguous keep),
  // else expand to bf16 [E*out,in].
  OwnedTensor Sew(const std::string& name, int64_t experts) {
    const GgufTensorInfo& t = Take(name);
    VT_CHECK(t.shape.size() == 3 && t.shape[0] == experts,
             "deepseek-v4 gguf: expected [E,out,in] expert tensor " + name);
    const int64_t rows = t.shape[0] * t.shape[1];  // E*out
    const int64_t k = t.shape[2];                  // in
    const GgufResidency r = pol.Route(t, GgufTensorRole::kStackedExpertWeight);
    if (r != GgufResidency::kExpandBf16) {
      // mmap-VIEW when allowed (see Mw) — the 256 routed-expert slabs are the bulk
      // of the ~91 GiB, so borrowing them in place is the dominant memory win.
      return OwnGgufQuantBlocks(t, rows, k, /*row_offset=*/0,
                                pol.mmap_residency ? &g : nullptr, pol.quant_repack);
    }
    return MakeBf16Owned(DequantGgufRowToBf16(t.ggml_type, t.data, rows * k),
                         {rows, k}, /*nk=*/true);
  }
  // A value/table tensor whose bytes are rewritten (norm/bias/scale/sink/table/
  // embed) — NEVER keep-quant. Asserts the policy agrees (totality) then dequants
  // to f32 in the file's torch shape.
  OwnedTensor Vec(const std::string& name, GgufTensorRole role) {
    const GgufTensorInfo& t = Take(name);
    VT_CHECK(pol.Route(t, role) == GgufResidency::kExpandBf16,
             std::string("deepseek-v4 gguf: a ") + Name(role) +
                 " tensor must not keep quant blocks: " + name);
    return MakeF32Owned(DqRowF32(g, name), t.shape);
  }
};

// Fill the tiny-config CPU composition host field for one flat weight.
std::vector<float> HostVec(const GgufFile& g, const std::string& name) {
  return DqRowF32(g, name);
}

}  // namespace

DeepseekV4Params DeepseekV4ParamsFromGguf(const GgufFile& g) {
  const GgufValue* arch = g.FindKv("general.architecture");
  VT_CHECK(arch != nullptr && arch->TypeId() == kGgufString,
           "deepseek-v4 gguf: general.architecture must be a string");
  VT_CHECK(std::get<std::string>(arch->v) == "deepseek4",
           "deepseek-v4 gguf: expected general.architecture 'deepseek4', got '" +
               std::get<std::string>(arch->v) + "'");
  const std::string p = "deepseek4.";

  DeepseekV4Params d;
  d.hidden_size = ReqInt(g, p + "embedding_length");
  d.num_hidden_layers = ReqInt(g, p + "block_count");
  d.num_attention_heads = ReqInt(g, p + "attention.head_count");
  d.num_key_value_heads = OptInt(g, p + "attention.head_count_kv", 1);
  d.head_dim = ReqInt(g, p + "attention.key_length");
  d.qk_rope_head_dim = OptInt(g, p + "rope.dimension_count", 64);
  d.q_lora_rank = ReqInt(g, p + "attention.q_lora_rank");
  d.o_lora_rank = ReqInt(g, p + "attention.output_lora_rank");
  d.o_groups = ReqInt(g, p + "attention.output_group_count");
  d.sliding_window = OptInt(g, p + "attention.sliding_window", 0);
  d.rope_theta = OptFloat(g, p + "rope.freq_base", 10000.0);
  d.compress_rope_theta = OptFloat(g, p + "attention.compress_rope_freq_base", 160000.0);
  d.rope_scale_factor = OptFloat(g, p + "rope.scaling.factor", 16.0);
  d.rope_orig_ctx = OptInt(g, p + "rope.scaling.original_context_length", 65536);
  d.rope_beta_fast = OptFloat(g, p + "rope.scaling.yarn_beta_fast", 32.0);
  d.rope_beta_slow = OptFloat(g, p + "rope.scaling.yarn_beta_slow", 1.0);
  d.rms_norm_eps =
      static_cast<float>(OptFloat(g, p + "attention.layer_norm_rms_epsilon", 1e-6));
  d.max_position_embeddings = OptInt(g, p + "context_length", 0);
  d.num_nextn_predict_layers = OptInt(g, p + "nextn_predict_layers", 0);

  // MoE.
  d.n_routed_experts = ReqInt(g, p + "expert_count");
  d.num_experts_per_tok = ReqInt(g, p + "expert_used_count");
  d.n_shared_experts = OptInt(g, p + "expert_shared_count", 0);
  d.moe_intermediate_size = ReqInt(g, p + "expert_feed_forward_length");
  d.num_hash_layers = OptInt(g, p + "hash_layer_count", 0);
  d.routed_scaling_factor = OptFloat(g, p + "expert_weights_scale", 1.0);
  // Clamped-SwiGLU limit. llama.cpp converters emit EITHER a scalar
  // `deepseek4.swiglu_clamp` (our tiny-synthetic gate) OR a per-layer f32 array
  // `deepseek4.swiglu_clamp_exp` (the real DeepSeek-V4 GGUFs — antirez's
  // q2-imatrix + unsloth). The real arrays are uniform (all 10.0), so the first
  // element is the faithful scalar. A missing/zero limit would zero every expert
  // (ClampedSwiGLU: min(gate,0)*...*clamp(up,0,0)=0), so read the array form too.
  d.swiglu_limit = OptFloat(g, p + "swiglu_clamp", -1.0);
  if (d.swiglu_limit < 0.0) {
    if (const GgufValue* sc = g.FindKv(p + "swiglu_clamp_exp");
        sc != nullptr && sc->TypeId() == kGgufArray &&
        !std::get<GgufArray>(sc->v).elems.empty()) {
      d.swiglu_limit =
          GgKvFloat(std::get<GgufArray>(sc->v).elems[0], p + "swiglu_clamp_exp");
    } else {
      d.swiglu_limit = 0.0;
    }
  }
  d.norm_topk_prob = OptInt(g, p + "norm_topk_prob", 1) != 0;
  // The `deepseek4` arch is fixed sqrtsoftplus/noaux_tc (expert_gating_func==4);
  // the forward selects those unconditionally, so we do not re-validate the enum.
  d.scoring_func = "sqrtsoftplus";
  d.topk_method = "noaux_tc";
  d.expert_dtype = "fp4";

  // MHC.
  d.hc_mult = ReqInt(g, p + "hyper_connection.count");
  d.hc_sinkhorn_iters = OptInt(g, p + "hyper_connection.sinkhorn_iterations", 20);
  d.hc_eps = OptFloat(g, p + "hyper_connection.epsilon", 1e-6);

  // DSA.
  d.index_head_dim = OptInt(g, p + "attention.indexer.key_length", 0);
  d.index_n_heads = OptInt(g, p + "attention.indexer.head_count", 0);
  d.index_topk = OptInt(g, p + "attention.indexer.top_k", 0);
  if (const GgufValue* cr = g.FindKv(p + "attention.compress_ratios");
      cr != nullptr && cr->TypeId() == kGgufArray) {
    for (const GgufValue& e : std::get<GgufArray>(cr->v).elems)
      d.compress_ratios.push_back(GgKvInt(e, "compress_ratios"));
  }

  // vocab: prefer the token_embd leading (out) dim, else the kv.
  const GgufValue* vk = g.FindKv(p + "vocab_size");
  d.vocab_size = vk != nullptr ? GgKvInt(*vk, p + "vocab_size")
                               : g.Get("token_embd.weight").shape[0];

  // Minimal self-consistency (the GGUF is the source of geometry truth — the
  // strict head_dim==512/sqrtsoftplus assertions live in ParseDeepseekV4Params
  // for the safetensors path; a tiny synthetic GGUF uses a small head_dim).
  VT_CHECK(d.hidden_size > 0 && d.num_hidden_layers > 0 && d.head_dim > 0,
           "deepseek-v4 gguf: degenerate geometry");
  VT_CHECK(d.n_routed_experts > 0, "deepseek-v4 gguf: n_routed_experts must be > 0");
  VT_CHECK(d.hc_mult > 0, "deepseek-v4 gguf: hc_mult must be > 0 (MHC is structural)");
  // The real DeepSeek-V4 GGUFs carry compress_ratios of length block_count + the
  // MTP/nextn layers (antirez's q2-imatrix: 44 = 43 main + 1 nextn), so accept
  // >= block_count and keep only the main-layer prefix (the topology helpers
  // is_hash_layer/has_indexer/has_compressor index [0, block_count)). A tiny
  // synthetic file has exactly block_count entries (no truncation).
  VT_CHECK(static_cast<int64_t>(d.compress_ratios.size()) >= d.num_hidden_layers,
           "deepseek-v4 gguf: compress_ratios length must be >= block_count");
  if (static_cast<int64_t>(d.compress_ratios.size()) > d.num_hidden_layers)
    d.compress_ratios.resize(static_cast<size_t>(d.num_hidden_layers));
  return d;
}

bool DeepseekV4GgufHasMtp(const GgufFile& g) {
  // The MTP/nextn tail lives at block index == block_count (llama.cpp `blk.{N}.*`
  // for the nextn layer) or under an explicit `nextn`/`mtp` name. vLLM raises when
  // the checkpoint advertises nextn_predict_layers but was quantized WITHOUT these
  // tensors (nvidia/mtp.py load_weights). Detect their PRESENCE so the engine can
  // fall back to MTP-off cleanly. NOTE (verified 2026-07-30, .agents/specs/
  // deepseek-v4-mtp.md §4): BOTH shipped DeepSeek-V4-Flash GGUFs return false —
  // the converter dropped nextn though the KV sets nextn_predict_layers=1.
  const int64_t nextn = OptInt(g, "deepseek4.nextn_predict_layers", 0);
  if (nextn <= 0) return false;
  const int64_t block_count = ReqInt(g, "deepseek4.block_count");
  const std::string nextn_prefix = "blk." + std::to_string(block_count) + ".";
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name.rfind(nextn_prefix, 0) == 0) return true;
    if (t.name.find("nextn") != std::string::npos) return true;
    if (t.name.find(".mtp") != std::string::npos || t.name.rfind("mtp.", 0) == 0)
      return true;
  }
  return false;
}

HfConfig DeepseekV4HfConfigFromGguf(const GgufFile& g) {
  // The GGUF is the source of geometry truth; resolve it ONCE (throws on a
  // non-deepseek4 arch or a missing required key) and republish it.
  const DeepseekV4Params p = DeepseekV4ParamsFromGguf(g);

  HfConfig c;
  // Keep llama.cpp's GGUF family key in model_type, but map `architectures` onto
  // the registered vLLM model class so ModelRegistry::Resolve routes a deepseek4
  // file into the DeepSeek-V4 factory — the same trick HfConfigFromGguf uses to
  // map the qwen35* keys onto the Qwen3.5 wrappers.
  c.model_type = "deepseek4";
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = p.hidden_size;
  c.num_hidden_layers = p.num_hidden_layers;
  c.vocab_size = p.vocab_size;
  c.num_attention_heads = p.num_attention_heads;
  c.num_key_value_heads = p.num_key_value_heads;
  c.head_dim = p.head_dim;
  c.rms_norm_eps = p.rms_norm_eps;
  c.rope_theta = p.rope_theta;
  c.max_position_embeddings = p.max_position_embeddings;
  c.torch_dtype = "bfloat16";

  // Republish the DeepSeek-V4 scalars the registry parse hook
  // (ParseDeepseekV4Config -> ParseDeepseekV4Params) reads from config.raw, so
  // that hook validates the SAME geometry this GGUF describes. The weight loader
  // itself re-derives from the GGUF KV (LoadDeepseekV4FromGguf ignores config).
  nlohmann::json& raw = c.raw = nlohmann::json::object();
  raw["hidden_size"] = p.hidden_size;
  raw["num_hidden_layers"] = p.num_hidden_layers;
  raw["vocab_size"] = p.vocab_size;
  raw["num_attention_heads"] = p.num_attention_heads;
  raw["num_key_value_heads"] = p.num_key_value_heads;
  raw["head_dim"] = p.head_dim;
  raw["qk_rope_head_dim"] = p.qk_rope_head_dim;
  raw["q_lora_rank"] = p.q_lora_rank;
  raw["o_lora_rank"] = p.o_lora_rank;
  raw["o_groups"] = p.o_groups;
  raw["sliding_window"] = p.sliding_window;
  raw["rope_theta"] = p.rope_theta;
  raw["compress_rope_theta"] = p.compress_rope_theta;
  raw["rms_norm_eps"] = p.rms_norm_eps;
  raw["max_position_embeddings"] = p.max_position_embeddings;
  raw["num_nextn_predict_layers"] = p.num_nextn_predict_layers;
  raw["n_routed_experts"] = p.n_routed_experts;
  raw["num_experts_per_tok"] = p.num_experts_per_tok;
  raw["moe_intermediate_size"] = p.moe_intermediate_size;
  raw["n_shared_experts"] = p.n_shared_experts;
  raw["norm_topk_prob"] = p.norm_topk_prob;
  raw["routed_scaling_factor"] = p.routed_scaling_factor;
  raw["swiglu_limit"] = p.swiglu_limit;
  raw["scoring_func"] = p.scoring_func;
  raw["topk_method"] = p.topk_method;
  raw["num_hash_layers"] = p.num_hash_layers;
  raw["expert_dtype"] = p.expert_dtype;
  raw["hc_mult"] = p.hc_mult;
  raw["hc_sinkhorn_iters"] = p.hc_sinkhorn_iters;
  raw["hc_eps"] = p.hc_eps;
  raw["index_head_dim"] = p.index_head_dim;
  raw["index_n_heads"] = p.index_n_heads;
  raw["index_topk"] = p.index_topk;
  raw["compress_ratios"] = p.compress_ratios;
  return c;
}

DeepseekV4Weights LoadDeepseekV4FromGguf(const GgufFile& g, const HfConfig& config,
                                        const GgufLoadPolicy* policy) {
  (void)config;  // params resolved from the GGUF KV (self-describing vehicle)
  const GgufLoadPolicy env = GgufLoadPolicy::FromEnv();
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env;
  const DeepseekV4Params p = DeepseekV4ParamsFromGguf(g);

  DeepseekV4Weights w;
  w.params = p;
  V4GgufCtx ctx{g, pol, {}};

  const int64_t H = p.hidden_size;
  const int64_t nh = p.num_attention_heads;
  const int64_t hd = p.head_dim;
  const int64_t qlr = p.q_lora_rank;
  const int64_t ne = p.n_routed_experts;
  const int64_t topk = p.num_experts_per_tok;
  const int64_t mi = p.moe_intermediate_size;
  const int64_t hc = p.hc_mult;
  const int64_t inh = p.index_n_heads;
  const int64_t ihd = p.index_head_dim;
  const bool tied = !HasGgufTensor(g, "output.weight");

  // ── model-level tower slots ─────────────────────────────────────────────
  DeepseekV4GgufWeights& tw = w.gguf;
  tw.embed = ctx.Vec("token_embd.weight", GgufTensorRole::kEmbeddingTable);
  tw.lm_head = tied ? ctx.Vec("token_embd.weight", GgufTensorRole::kEmbeddingTable)
                    : ctx.Mw("output.weight");
  tw.final_norm = ctx.Vec("output_norm.weight", GgufTensorRole::kVector);
  tw.hc_head_base = ctx.Vec("output_hc_base.weight", GgufTensorRole::kVector);
  tw.hc_head_fn = ctx.Vec("output_hc_fn.weight", GgufTensorRole::kVector);
  tw.hc_head_scale = ctx.Vec("output_hc_scale.weight", GgufTensorRole::kVector);

  // ── SMALL host composition tower (W2C: dequant bridge for the NON-GEMM tensors
  //    ONLY). The big MLA/MoE/lm_head weights are NOT f32-expanded here — the
  //    forward (DeepseekV4ForwardGguf) consumes them keep-quant from `w.gguf`. So
  //    `host` holds only: embed (a gather), the RMSNorms, the attention sink, the
  //    MHC/DSA mixing/ape/scale, the hash table / noaux_tc bias, and the small
  //    indexer weights_proj. This is the ~1 TiB -> ~small memory fix (the routed
  //    experts alone are ~277B params; f32 would OOM the 119 GiB unified pool). ─
  DeepseekV4HostWeights& hw = w.host;
  hw.embed = HostVec(g, "token_embd.weight");  // gather table (V/f32), NOT a GEMM
  // hw.lm_head deliberately LEFT EMPTY — the final projection reads the keep-quant
  // `tw.lm_head` block via the forward's Gemm (tied => an f32 OwnedTensor).
  hw.final_norm_weight = HostVec(g, "output_norm.weight");
  hw.hc_head_fn = HostVec(g, "output_hc_fn.weight");
  hw.hc_head_base = HostVec(g, "output_hc_base.weight");
  const std::vector<float> hhs = HostVec(g, "output_hc_scale.weight");
  hw.hc_head_scale = hhs.empty() ? 0.0f : hhs[0];
  hw.layers.resize(static_cast<size_t>(p.num_hidden_layers));

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4GgufLayerWeights lw;
    DeepseekV4LayerHostWeights& hl = hw.layers[static_cast<size_t>(l)];
    lw.is_hash = p.is_hash_layer(l);
    lw.has_compressor = p.has_compressor(l);
    lw.has_indexer = p.has_indexer(l);

    // 512-wide MLA (MW keep-quant) + norms/sink (V).
    lw.wq_a = ctx.Mw(Blk(l, "attn_q_a.weight"));
    lw.wq_b = ctx.Mw(Blk(l, "attn_q_b.weight"));
    lw.wkv = ctx.Mw(Blk(l, "attn_kv.weight"));
    lw.wo_a = ctx.Mw(Blk(l, "attn_output_a.weight"), /*allow_align=*/false);  // row-sliced (o-groups)
    lw.wo_b = ctx.Mw(Blk(l, "attn_output_b.weight"));
    lw.attn_norm = ctx.Vec(Blk(l, "attn_norm.weight"), GgufTensorRole::kVector);
    lw.q_a_norm = ctx.Vec(Blk(l, "attn_q_a_norm.weight"), GgufTensorRole::kVector);
    lw.kv_a_norm = ctx.Vec(Blk(l, "attn_kv_a_norm.weight"), GgufTensorRole::kVector);
    lw.attn_sink = ctx.Vec(Blk(l, "attn_sinks.weight"), GgufTensorRole::kVector);
    lw.ffn_norm = ctx.Vec(Blk(l, "ffn_norm.weight"), GgufTensorRole::kVector);

    // MHC per-layer mixing (V).
    lw.hc_attn_base = ctx.Vec(Blk(l, "hc_attn_base.weight"), GgufTensorRole::kVector);
    lw.hc_attn_fn = ctx.Vec(Blk(l, "hc_attn_fn.weight"), GgufTensorRole::kVector);
    lw.hc_attn_scale = ctx.Vec(Blk(l, "hc_attn_scale.weight"), GgufTensorRole::kVector);
    lw.hc_ffn_base = ctx.Vec(Blk(l, "hc_ffn_base.weight"), GgufTensorRole::kVector);
    lw.hc_ffn_fn = ctx.Vec(Blk(l, "hc_ffn_fn.weight"), GgufTensorRole::kVector);
    lw.hc_ffn_scale = ctx.Vec(Blk(l, "hc_ffn_scale.weight"), GgufTensorRole::kVector);

    // MoE: router gate (MW), 256 routed experts + shared (SEW/MW keep-quant),
    // the hash `tid2eid` (hash layers) or the noaux_tc `exp_probs_b` bias.
    lw.moe_gate = ctx.Mw(Blk(l, "ffn_gate_inp.weight"));
    lw.moe_gate_exps = ctx.Sew(Blk(l, "ffn_gate_exps.weight"), ne);
    lw.moe_up_exps = ctx.Sew(Blk(l, "ffn_up_exps.weight"), ne);
    lw.moe_down_exps = ctx.Sew(Blk(l, "ffn_down_exps.weight"), ne);
    lw.shared_gate = ctx.Mw(Blk(l, "ffn_gate_shexp.weight"));
    lw.shared_up = ctx.Mw(Blk(l, "ffn_up_shexp.weight"));
    lw.shared_down = ctx.Mw(Blk(l, "ffn_down_shexp.weight"));
    if (lw.is_hash) {
      lw.tid2eid = ctx.Vec(Blk(l, "ffn_gate_tid2eid.weight"), GgufTensorRole::kVector);
    } else {
      lw.e_score_bias = ctx.Vec(Blk(l, "exp_probs_b.bias"), GgufTensorRole::kVector);
    }

    // DSA compressor (compress_ratio != 0) + Lightning-Indexer (== 4).
    if (lw.has_compressor) {
      lw.comp_ape = ctx.Vec(Blk(l, "attn_compressor_ape.weight"), GgufTensorRole::kVector);
      lw.comp_wgate = ctx.Mw(Blk(l, "attn_compressor_gate.weight"));
      lw.comp_wkv = ctx.Mw(Blk(l, "attn_compressor_kv.weight"));
      lw.comp_norm = ctx.Vec(Blk(l, "attn_compressor_norm.weight"), GgufTensorRole::kVector);
    }
    if (lw.has_indexer) {
      lw.idx_wq_b = ctx.Mw(Blk(l, "indexer.attn_q_b.weight"));
      lw.idx_proj = ctx.Vec(Blk(l, "indexer.proj.weight"), GgufTensorRole::kVector);
      lw.idx_comp_ape =
          ctx.Vec(Blk(l, "indexer_compressor_ape.weight"), GgufTensorRole::kVector);
      lw.idx_comp_wgate = ctx.Mw(Blk(l, "indexer_compressor_gate.weight"));
      lw.idx_comp_wkv = ctx.Mw(Blk(l, "indexer_compressor_kv.weight"));
      lw.idx_comp_norm =
          ctx.Vec(Blk(l, "indexer_compressor_norm.weight"), GgufTensorRole::kVector);
    }

    // ── SMALL host bridge for THIS layer (W2C: only the NON-GEMM slots the
    //    forward reads as f32; the big MLA/MoE GEMM weights below are consumed
    //    keep-quant from `lw` and are LEFT EMPTY in `hl`). ────────────────────
    hl.attn_norm_weight = HostVec(g, Blk(l, "attn_norm.weight"));
    hl.ffn_norm_weight = HostVec(g, Blk(l, "ffn_norm.weight"));
    hl.hc_attn_fn = HostVec(g, Blk(l, "hc_attn_fn.weight"));
    hl.hc_attn_base = HostVec(g, Blk(l, "hc_attn_base.weight"));
    hl.hc_attn_scale = HostVec(g, Blk(l, "hc_attn_scale.weight"));
    hl.hc_ffn_fn = HostVec(g, Blk(l, "hc_ffn_fn.weight"));
    hl.hc_ffn_base = HostVec(g, Blk(l, "hc_ffn_base.weight"));
    hl.hc_ffn_scale = HostVec(g, Blk(l, "hc_ffn_scale.weight"));
    hl.q_norm_weight = HostVec(g, Blk(l, "attn_q_a_norm.weight"));
    hl.kv_norm_weight = HostVec(g, Blk(l, "attn_kv_a_norm.weight"));
    hl.attn_sink = HostVec(g, Blk(l, "attn_sinks.weight"));
    // wq_a/wq_b/wkv/wo_a/wo_b, gate, shared+routed experts: KEEP-QUANT (in `lw`),
    // NOT f32-expanded here — the memory fix.
    if (lw.is_hash) {
      const std::vector<float> t2e = HostVec(g, Blk(l, "ffn_gate_tid2eid.weight"));
      hl.tid2eid.resize(t2e.size());
      for (size_t i = 0; i < t2e.size(); ++i)
        hl.tid2eid[i] = static_cast<int32_t>(std::lround(t2e[i]));
    } else {
      hl.gate_bias = HostVec(g, Blk(l, "exp_probs_b.bias"));
    }
    if (lw.has_compressor) {
      // comp_wgate is keep-quant (in `lw`); only ape/norm are f32 (small V).
      hl.comp_ape = HostVec(g, Blk(l, "attn_compressor_ape.weight"));
      hl.comp_norm_weight = HostVec(g, Blk(l, "attn_compressor_norm.weight"));
    }
    if (lw.has_indexer) {
      // idx_wq (indexer.attn_q_b) + idx_wk (indexer_compressor_kv) are keep-quant
      // (in `lw`); only the small weights_proj stays f32 (documented tiny-config
      // structural bridge, deepseek_v4.cpp).
      hl.idx_wproj = HostVec(g, Blk(l, "indexer.proj.weight"));
    }

    tw.layers.push_back(std::move(lw));
  }

  // ── the ACCOUNTING gate: every file tensor must have been routed exactly once
  //    (none unmapped, none leftover) — the C++ half of the name-map contract
  //    (scripts/check-dsv4-gguf-namemap.py gates the real 1328-tensor manifest). ─
  for (const GgufTensorInfo& t : g.Tensors()) {
    VT_CHECK(ctx.consumed.count(t.name) != 0,
             "deepseek-v4 gguf loader: LEFTOVER tensor not covered by the blk.N.* "
             "name map: " + t.name);
  }
  // ── W2C MEMORY-BOUND ASSERTION: the big MLA/MoE/lm_head weights must NOT be
  //    f32-expanded into the host tower. This is the crux of the fix — a
  //    regression that re-adds a HostVec for any big weight (rebuilding the ~1 TiB
  //    f32 tower that OOM-reboots the 119 GiB pool) fails LOUDLY here at load. ──
  VT_CHECK(w.host.lm_head.empty(),
           "deepseek-v4 gguf: lm_head must stay keep-quant, not f32-expanded");
  for (const DeepseekV4LayerHostWeights& hl : w.host.layers) {
    VT_CHECK(hl.wq_a.empty() && hl.wq_b.empty() && hl.wkv.empty() &&
                 hl.wo_a.empty() && hl.wo_b.empty() && hl.gate_weight.empty() &&
                 hl.shared_w1.empty() && hl.shared_w3.empty() && hl.shared_w2.empty() &&
                 hl.exp_w1.empty() && hl.exp_w3.empty() && hl.exp_w2.empty() &&
                 hl.comp_wgate.empty() && hl.idx_wq.empty() && hl.idx_wk.empty(),
             "deepseek-v4 gguf: a big MLA/MoE weight was f32-expanded into the host "
             "tower — the keep-quant forward must consume it from `gguf` (W2C)");
  }
  w.accounted_tensors = static_cast<int64_t>(ctx.consumed.size());
  w.has_gguf_weights = true;
  w.has_host_weights = true;
  // Silence unused in a non-indexer/degenerate tiny config.
  (void)nh; (void)hd; (void)qlr; (void)topk; (void)mi; (void)hc; (void)inh; (void)ihd; (void)H;
  return w;
}

// ─── W2C memory accounting ───────────────────────────────────────────────────
namespace {
int64_t HostBytes(const DeepseekV4HostWeights& hw) {
  auto vf = [](const std::vector<float>& v) {
    return static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(float));
  };
  auto vi = [](const std::vector<int32_t>& v) {
    return static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(int32_t));
  };
  int64_t b = vf(hw.embed) + vf(hw.lm_head) + vf(hw.final_norm_weight) +
              vf(hw.hc_head_fn) + vf(hw.hc_head_base) + static_cast<int64_t>(sizeof(float));
  for (const DeepseekV4LayerHostWeights& hl : hw.layers) {
    b += vf(hl.attn_norm_weight) + vf(hl.ffn_norm_weight) + vf(hl.hc_attn_fn) +
         vf(hl.hc_attn_base) + vf(hl.hc_attn_scale) + vf(hl.hc_ffn_fn) +
         vf(hl.hc_ffn_base) + vf(hl.hc_ffn_scale) + vf(hl.wq_a) + vf(hl.q_norm_weight) +
         vf(hl.wq_b) + vf(hl.wkv) + vf(hl.kv_norm_weight) + vf(hl.attn_sink) +
         vf(hl.wo_a) + vf(hl.wo_b) + vf(hl.idx_wq) + vf(hl.idx_wk) + vf(hl.idx_wproj) +
         vf(hl.comp_wgate) + vf(hl.comp_ape) + vf(hl.comp_norm_weight) +
         vf(hl.gate_weight) + vf(hl.gate_bias) + vi(hl.tid2eid) + vf(hl.shared_w1) +
         vf(hl.shared_w3) + vf(hl.shared_w2) + vf(hl.exp_w1) + vf(hl.exp_w3) +
         vf(hl.exp_w2);
  }
  return b;
}
int64_t OwnedBytesOf(const OwnedTensor& t) {
  return static_cast<int64_t>(t.bytes.size());
}
int64_t GgufBytes(const DeepseekV4GgufWeights& gw) {
  int64_t b = OwnedBytesOf(gw.embed) + OwnedBytesOf(gw.lm_head) +
              OwnedBytesOf(gw.final_norm) + OwnedBytesOf(gw.hc_head_base) +
              OwnedBytesOf(gw.hc_head_fn) + OwnedBytesOf(gw.hc_head_scale);
  for (const DeepseekV4GgufLayerWeights& l : gw.layers) {
    for (const OwnedTensor* t :
         {&l.wq_a, &l.wq_b, &l.wkv, &l.wo_a, &l.wo_b, &l.attn_norm, &l.q_a_norm,
          &l.kv_a_norm, &l.attn_sink, &l.ffn_norm, &l.hc_attn_base, &l.hc_attn_fn,
          &l.hc_attn_scale, &l.hc_ffn_base, &l.hc_ffn_fn, &l.hc_ffn_scale, &l.moe_gate,
          &l.moe_gate_exps, &l.moe_up_exps, &l.moe_down_exps, &l.shared_gate,
          &l.shared_up, &l.shared_down, &l.tid2eid, &l.e_score_bias, &l.comp_ape,
          &l.comp_wgate, &l.comp_wkv, &l.comp_norm, &l.idx_wq_b, &l.idx_proj,
          &l.idx_comp_ape, &l.idx_comp_wgate, &l.idx_comp_wkv, &l.idx_comp_norm}) {
      b += OwnedBytesOf(*t);
    }
  }
  return b;
}
}  // namespace

int64_t DeepseekV4HostResidentBytes(const DeepseekV4Weights& w) {
  return HostBytes(w.host);
}
int64_t DeepseekV4GgufResidentBytes(const DeepseekV4Weights& w) {
  return GgufBytes(w.gguf);
}

}  // namespace vllm
