// A2-P (#810, .agents/specs/nemotron-h-a2p-paged-forward.md) — the PAGED
// forward, gated through the production entry point.
//
// ─── WHY EVERY CASE HERE ENTERS THROUGH `ModelRegistry::Forward` ────────────
//
// AGENTS.md §"Nothing lands dead" is explicit that a unit test constructing the
// type by hand proves the class works and never that anything reaches it.
// Before this change `NemotronHDeviceForward` had exactly ONE non-declaration
// call site in the whole tree — `test_nemotron_h_forward.cpp:1805` — which is
// the test-only-driver shape `.agents/reachability.md` names. So these cases
// build a real `NemotronHLoadedModel` from a real (synthetic) checkpoint, hand
// it to a real `GPUModelRunner`, and drive `execute_model` / `sample_tokens`.
// The runner allocates the paged KV and recurrent pages, builds the attention
// and GDN metadata, and calls `ModelRegistry::Forward` at `runner.cpp:1465`.
// Nothing in this file fabricates a `ModelForwardInput`.
//
// THE RED THIS FILE WAS WRITTEN AGAINST. On the base commit the same call
// reaches `nemotron_h_registry.cpp:161` and refuses by name with "the
// PAGED/BATCHED decode path is not ported", because the runner hands it
// non-empty `attn_kv` and `gdn_state`. Delete the paged branch in
// `ForwardNemotronHForCausalLM` and every case below RED again — that is
// mutation P-M7, and it is what separates "the class works" from "the
// capability is reached".
//
// ─── WHAT EACH CASE CAN AND CANNOT SEE ──────────────────────────────────────
//
// The multi-step token arm is the only one that can see a DROPPED CARRY: with
// one leg and fresh state, the recurrent half is unobservable
// (nemotron_h_forward.h:379-382 says so outright). It cannot see a too-WIDE
// dtype, a dequant fallback, or a dropped mechanism whose argmax is unchanged —
// which is why the per-block NUMERIC arm and the explicit memory-format
// assertions are here too.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::NemotronHBlock;
using vllm::NemotronHHostWeights;
using vllm::NemotronHLoadReport;
using vllm::NemotronHParams;
using vllm::NemotronHTrace;
using vllm::SafetensorsFile;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;
using vt::DType;

namespace {

// ─── the synthetic checkpoint ───────────────────────────────────────────────
//
// Five layers over the schedule `M E * M E`, so the case exercises TWO Mamba2
// layers (two independent recurrent pages), ONE GQA layer (one paged KV page)
// and TWO MoE layers. A single-mamba schedule would gate the degenerate case
// only, and a schedule with no attention layer or no mamba layer would make the
// runner abandon membership-by-name wholesale (`GroupLayerMask`,
// runner.cpp:363-378) and silently classify every layer as full attention.
//
// The geometry mirrors `TinyParams()` in test_nemotron_h_forward.cpp so the two
// files agree on what "tiny" means for this architecture.
constexpr int kHidden = 24;
constexpr int kVocab = 32;
constexpr int kAttnHeads = 4;
constexpr int kKvHeads = 2;  // GQA 2:1, as the released 32/2 is
constexpr int kHeadDim = 6;
constexpr int kMambaHeads = 4;
constexpr int kMambaHeadDim = 6;
constexpr int kNGroups = 2;
constexpr int kStateSize = 8;
constexpr int kConvKernel = 4;
constexpr int kChunkSize = 8;
constexpr int kRoutedExperts = 8;
constexpr int kExpertsPerTok = 3;
constexpr int kMoeInter = 10;
constexpr int kSharedInter = 12;

constexpr int kMambaInter = kMambaHeads * kMambaHeadDim;                  // 24
constexpr int kConvDim = kMambaInter + 2 * kNGroups * kStateSize;         // 56
constexpr int kInProjOut = kMambaInter + kConvDim + kMambaHeads;          // 84
constexpr int kQDim = kAttnHeads * kHeadDim;                             // 24
constexpr int kKvDim = kKvHeads * kHeadDim;                              // 12

constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 16;
constexpr int kMaxModelLen = 128;

struct Fx {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// A small deterministic spread. The values matter: an all-equal checkpoint
// makes every expert route identically and every recurrent state converge, so
// a dropped carry would move no token and the whole file would prove nothing.
float Synth(uint32_t& r, float scale) {
  r = r * 1664525u + 1013904223u;
  const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
  return (u - 0.5f) * scale;
}

std::string Bf16Bytes(size_t n, int seed, float scale) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t bf = vt::F32ToBF16(Synth(r, scale));
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}

std::string F32Bytes(size_t n, int seed, float scale) {
  std::string s(n * 4, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2246822519u + 1u;
  for (size_t i = 0; i < n; ++i) {
    const float f = Synth(r, scale);
    std::memcpy(&s[i * 4], &f, 4);
  }
  return s;
}

Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.5f) {
  return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed, scale)};
}
Fx F32(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.5f) {
  return {n, "F32", sh, F32Bytes(static_cast<size_t>(NumEl(sh)), seed, scale)};
}

std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype},
                   {"shape", t.shape},
                   {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes, const char* ext = ".safetensors") {
    // Unique PER PROCESS: two concurrent ctest processes sharing a fixed name
    // would overwrite each other's fixture mid-read.
    static int c = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("nemotron_h_paged_" + std::to_string(::getpid()) + "_" +
              std::to_string(c++) + ext))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// EXACTLY the tensor set `EnumerateNemotronHTensors` names for this config, and
// no more: the loader refuses on `enumerated != in_index` in BOTH directions
// (nemotron_h_weights.cpp), so a stray or missing tensor here is a load-time
// failure rather than a silent partial model.
std::vector<Fx> BuildTensors(const std::vector<NemotronHBlock>& schedule) {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("backbone.embeddings.weight", {kVocab, kHidden}, s++));
  v.push_back(Bf16("backbone.norm_f.weight", {kHidden}, s++, 0.8f));
  v.push_back(Bf16("lm_head.weight", {kVocab, kHidden}, s++, 0.25f));
  for (size_t l = 0; l < schedule.size(); ++l) {
    const std::string p = "backbone.layers." + std::to_string(l) + ".";
    const std::string m = p + "mixer.";
    v.push_back(Bf16(p + "norm.weight", {kHidden}, s++, 0.9f));
    switch (schedule[l]) {
      case NemotronHBlock::kMamba:
        v.push_back(Bf16(m + "in_proj.weight", {kInProjOut, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "out_proj.weight", {kHidden, kMambaInter}, s++, 0.3f));
        v.push_back(Bf16(m + "conv1d.weight", {kConvDim, 1, kConvKernel}, s++, 0.4f));
        v.push_back(Bf16(m + "conv1d.bias", {kConvDim}, s++, 0.2f));
        // f32 BY CONTRACT: vt::Mamba2ChunkScan validates A/D/dt_bias as f32,
        // mirroring upstream's `-torch.exp(self.A_log.float())`.
        v.push_back(F32(m + "A_log", {kMambaHeads}, s++, 0.6f));
        v.push_back(F32(m + "D", {kMambaHeads}, s++, 0.6f));
        v.push_back(F32(m + "dt_bias", {kMambaHeads}, s++, 0.3f));
        v.push_back(Bf16(m + "norm.weight", {kMambaInter}, s++, 0.7f));
        break;
      case NemotronHBlock::kAttention:
        v.push_back(Bf16(m + "q_proj.weight", {kQDim, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "k_proj.weight", {kKvDim, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "v_proj.weight", {kKvDim, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "o_proj.weight", {kHidden, kQDim}, s++, 0.3f));
        break;
      case NemotronHBlock::kMoe:
        // The router is f32 on disk AND in memory: upstream builds it with
        // `out_dtype=torch.float32, force_fp32_compute=True` (nemotron_h.py:150-156).
        v.push_back(F32(m + "gate.weight", {kRoutedExperts, kHidden}, s++, 0.35f));
        v.push_back(F32(m + "gate.e_score_correction_bias", {kRoutedExperts}, s++, 0.4f));
        for (int e = 0; e < kRoutedExperts; ++e) {
          const std::string ex = m + "experts." + std::to_string(e) + ".";
          v.push_back(Bf16(ex + "up_proj.weight", {kMoeInter, kHidden}, s++, 0.3f));
          v.push_back(Bf16(ex + "down_proj.weight", {kHidden, kMoeInter}, s++, 0.3f));
        }
        v.push_back(Bf16(m + "shared_experts.up_proj.weight", {kSharedInter, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "shared_experts.down_proj.weight", {kHidden, kSharedInter}, s++, 0.3f));
        break;
      case NemotronHBlock::kMlp:
        break;
    }
  }
  return v;
}

const char* BlockName(NemotronHBlock b) {
  switch (b) {
    case NemotronHBlock::kMamba: return "mamba";
    case NemotronHBlock::kAttention: return "attention";
    case NemotronHBlock::kMoe: return "moe";
    case NemotronHBlock::kMlp: return "mlp";
  }
  return "mamba";
}

// NO `quantization_config` — the released checkpoint is MIXED_PRECISION, but a
// plain bf16 NemotronH safetensors checkpoint ships none, and the loader's
// whole enumeration branches on `p.quant.present` (nemotron_h_weights.cpp:952).
// Adding one here would flip every expert to the NVFP4 triple and every mamba
// projection to the FP8 triple, which is A2-Q's surface, not A2-P's.
std::string TinyConfigJson(const std::vector<NemotronHBlock>& schedule,
                           const std::string& dtype) {
  nlohmann::json j;
  j["architectures"] = nlohmann::json::array({"NemotronHForCausalLM"});
  j["model_type"] = "nemotron_h";
  j["dtype"] = dtype;
  nlohmann::json blocks = nlohmann::json::array();
  for (NemotronHBlock b : schedule) blocks.push_back(BlockName(b));
  j["layers_block_type"] = blocks;
  j["num_hidden_layers"] = static_cast<int>(schedule.size());
  j["hidden_size"] = kHidden;
  j["vocab_size"] = kVocab;
  j["max_position_embeddings"] = kMaxModelLen;
  j["layer_norm_epsilon"] = 1e-5;
  j["tie_word_embeddings"] = false;
  j["num_attention_heads"] = kAttnHeads;
  j["num_key_value_heads"] = kKvHeads;
  j["head_dim"] = kHeadDim;
  j["attention_bias"] = false;
  j["mamba_num_heads"] = kMambaHeads;
  j["mamba_head_dim"] = kMambaHeadDim;
  j["n_groups"] = kNGroups;
  j["ssm_state_size"] = kStateSize;
  j["conv_kernel"] = kConvKernel;
  j["chunk_size"] = kChunkSize;
  j["expand"] = 2;
  j["mamba_hidden_act"] = "silu";
  // Resolved INDEPENDENTLY of the model dtype (mamba_utils.py:96-107), and
  // "float32" is what the released checkpoint ships. Collapsing it to the
  // activation dtype halves the recurrent state and is invisible to a token
  // gate, which is why the fixture states it rather than defaulting.
  j["mamba_ssm_cache_dtype"] = "float32";
  j["use_conv_bias"] = true;
  j["mamba_proj_bias"] = false;
  j["n_routed_experts"] = kRoutedExperts;
  j["num_experts_per_tok"] = kExpertsPerTok;
  j["moe_intermediate_size"] = kMoeInter;
  j["n_shared_experts"] = 1;
  j["moe_shared_expert_intermediate_size"] = kSharedInter;
  j["n_group"] = 1;
  j["topk_group"] = 1;
  j["routed_scaling_factor"] = 2.5;  // the released value
  j["norm_topk_prob"] = true;
  j["mlp_hidden_act"] = "relu2";
  j["mlp_bias"] = false;
  return j.dump(2);
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

const std::vector<NemotronHBlock>& Schedule() {
  static const std::vector<NemotronHBlock> s{
      NemotronHBlock::kMamba, NemotronHBlock::kMoe, NemotronHBlock::kAttention,
      NemotronHBlock::kMamba, NemotronHBlock::kMoe};
  return s;
}

struct Fixture {
  std::unique_ptr<TempFile> st;
  std::unique_ptr<TempFile> cfg_json;
  std::vector<SafetensorsFile> shards;
  HfConfig cfg;
  NemotronHParams params;
  std::unique_ptr<vllm::LoadedModel> model;
  // A SECOND, independent materialization of the same bytes, so the host
  // reference arm runs on weights that are byte-identical to the model's
  // without needing a private accessor onto `NemotronHLoadedModel`.
  NemotronHHostWeights host;
  NemotronHLoadReport report;

  explicit Fixture(const std::string& dtype = "bfloat16") {
    st = std::make_unique<TempFile>(BuildSt(BuildTensors(Schedule())));
    cfg_json = std::make_unique<TempFile>(TinyConfigJson(Schedule(), dtype), ".json");
    shards.push_back(SafetensorsFile::Open(st->path()));
    cfg = vllm::LoadHfConfig(cfg_json->path());
    params = vllm::ParseNemotronHParams(cfg);
    model = ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards));
    host = vllm::LoadNemotronHHostWeights(
        shards, params, vllm::ResolveNemotronHModelDType(cfg), &report);
  }
};

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;
  sp.PostInit();
  return sp;
}

// `MakeNemotronHKVCache` publishes group 0 = the full-attention pages, group 1
// = the Mamba2 recurrent slots (nemotron_h_registry.cpp:235-270), and
// `NewRequestData::block_ids` is parallel to that order.
NewRequestData MakeNewReq(const std::string& id, std::vector<int32_t> prompt,
                          std::vector<int> fa_blocks, int state_slot) {
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = prompt;
  nr.sampling_params = Greedy();
  nr.block_ids = {std::move(fa_blocks), std::vector<int>{state_slot}};
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = std::move(prompt);
  return nr;
}

SchedulerOutput NewStep(std::vector<NewRequestData> new_reqs,
                        std::map<std::string, int> scheduled) {
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  so.scheduled_new_reqs = std::move(new_reqs);
  int total = 0;
  for (const auto& [id, n] : scheduled) total += n;
  so.num_scheduled_tokens = std::move(scheduled);
  so.total_num_scheduled_tokens = total;
  return so;
}

SchedulerOutput DecodeStep(const std::vector<std::string>& ids,
                           const std::vector<int>& num_computed,
                           const std::vector<int>& num_output) {
  SchedulerOutput so;
  CachedRequestData cached;
  cached.req_ids = ids;
  for (size_t i = 0; i < ids.size(); ++i) {
    cached.num_computed_tokens.push_back(num_computed[i]);
    cached.num_output_tokens.push_back(num_output[i]);
    cached.new_block_ids.emplace_back(std::nullopt);
  }
  so.scheduled_cached_reqs = std::move(cached);
  for (const std::string& id : ids) so.num_scheduled_tokens[id] = 1;
  so.total_num_scheduled_tokens = static_cast<int>(ids.size());
  return so;
}

// One prefill + (steps-1) single-token decode steps through the runner, i.e.
// through `ModelRegistry::Forward`. Returns the greedy tokens.
std::vector<int32_t> RunnerGreedy(GPUModelRunner& runner, const std::string& id,
                                  const std::vector<int32_t>& prompt, int steps,
                                  std::vector<int> fa_blocks, int state_slot) {
  std::vector<NewRequestData> reqs;
  reqs.push_back(MakeNewReq(id, prompt, std::move(fa_blocks), state_slot));
  std::map<std::string, int> sched;
  sched[id] = static_cast<int>(prompt.size());
  SchedulerOutput s1 = NewStep(std::move(reqs), std::move(sched));
  CHECK_FALSE(runner.execute_model(s1).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  REQUIRE(m1.sampled_token_ids[0].size() == 1);

  std::vector<int32_t> out{m1.sampled_token_ids[0][0]};
  std::vector<std::string> ids{id};
  std::vector<int> computed{static_cast<int>(prompt.size())};
  std::vector<int> outputs{1};
  for (int s = 1; s < steps; ++s) {
    SchedulerOutput sd = DecodeStep(ids, computed, outputs);
    CHECK_FALSE(runner.execute_model(sd).has_value());
    vllm::v1::ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids.size() == 1);
    REQUIRE(md.sampled_token_ids[0].size() == 1);
    out.push_back(md.sampled_token_ids[0][0]);
    computed[0] += 1;
    outputs[0] += 1;
  }
  return out;
}

// Read one recurrent page row back to the host. The pages are the runner's own
// device (here: host) buffers; this is what lets a case assert on the STATE
// rather than only on the tokens it produced.
std::vector<float> ReadStateRow(const vt::Tensor& page, int64_t slot) {
  int64_t row = 1;
  for (int r = 1; r < page.rank; ++r) row *= page.shape[r];
  std::vector<float> out(static_cast<size_t>(row));
  const char* base = static_cast<const char*>(page.data) +
                     static_cast<size_t>(slot) * static_cast<size_t>(page.stride[0]) *
                         vt::SizeOf(page.dtype);
  if (page.dtype == DType::kF32) {
    std::memcpy(out.data(), base, out.size() * sizeof(float));
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(base);
    for (int64_t i = 0; i < row; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  }
  return out;
}

double MaxAbs(const std::vector<float>& v) {
  double m = 0.0;
  for (float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
  return m;
}

// THE BAND IS TIED TO THE REFERENCE'S OWN SCALE, and the comparison CERTIFIES
// ITSELF: the same band must REJECT an all-zeros answer before it is allowed to
// accept the real one. Both halves are the repair
// test_nemotron_h_forward.cpp:96-113 already paid for on this model — a flat
// absolute band there was 169x the signal of the block it was judging, so a
// block returning all zeros passed.
//
// Returns the number of ELEMENTS compared, so a caller can assert it is the
// count the geometry predicts. "worst deviation: 0" over zero elements is a
// mute instrument, not a pass.
// `what` is a std::string, NOT a const char*: doctest 2.5.2 stringifies a
// `const char*` through its BOOL overload, so streaming one prints `1` and the
// message names nothing. That already cost this project a debugging cycle
// elsewhere in the tree.
//
// `worst_rel_out`, when non-null, receives this comparison's worst RELATIVE
// deviation, so the caller can report the number the band was derived FROM
// rather than only the band.
size_t ExpectCloseRel(const std::vector<float>& got, const std::vector<float>& want,
                      double rel, const std::string& what,
                      double* worst_rel_out = nullptr) {
  REQUIRE_MESSAGE(got.size() == want.size(), what << ": element counts differ");
  REQUIRE_MESSAGE(!want.empty(), what << ": nothing to compare");
  const double scale = MaxAbs(want);
  REQUIRE_MESSAGE(scale > 0.0, what << ": the reference is identically zero, so no "
                                       "band derived from it can fail");
  const double band = rel * scale;
  // The property: this band rejects the degenerate answer.
  bool rejects_zero = false;
  for (float w : want) {
    if (std::abs(static_cast<double>(w)) > band) {
      rejects_zero = true;
      break;
    }
  }
  REQUIRE_MESSAGE(rejects_zero, what << ": the band accepts an all-zeros answer, so it "
                                        "cannot fail");
  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (d > worst) {
      worst = d;
      worst_i = i;
    }
  }
  if (worst_rel_out != nullptr) *worst_rel_out = std::max(*worst_rel_out, worst / scale);
  CHECK_MESSAGE(worst <= band,
                what << ": worst deviation " << worst << " at element " << worst_i
                     << " over " << got.size() << " elements exceeds band " << band
                     << " (reference peak " << scale << " worst relative "
                     << (worst / scale) << ")");
  return got.size();
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE STRUCTURE — the runner allocates what this architecture published.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH paged: the runner allocates the two groups this model published") {
  Fixture fx;
  REQUIRE(fx.model != nullptr);
  // Load accounting first: the tensor set the fixture writes IS the enumerated
  // set, in both directions, or the load would have refused.
  CHECK(fx.report.enumerated == fx.report.in_index);
  CHECK(fx.report.materialized == fx.report.enumerated);
  CHECK(fx.report.deferred == 0);

  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  REQUIRE(kv.kv_cache_groups.size() == 2);

  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == 1);

  // ONE attention layer, TWO mamba layers — the schedule's own counts, taken
  // from `layer_names` rather than re-derived from the HF config, which is the
  // whole point of #810.
  const size_t n_attn = fx.params.LayerIndices(NemotronHBlock::kAttention).size();
  const size_t n_mamba = fx.params.LayerIndices(NemotronHBlock::kMamba).size();
  REQUIRE(n_attn == 1);
  REQUIRE(n_mamba == 2);
  REQUIRE(runner.attn_kv().size() == n_attn);
  REQUIRE(runner.gdn_state().size() == n_mamba);

  CHECK(runner.attn_kv()[0].num_kv_heads == kKvHeads);
  CHECK(runner.attn_kv()[0].head_size == kHeadDim);
  CHECK(runner.attn_kv()[0].block_size == kBlockSize);

  // ★ THE MEMORY FORMAT, ASSERTED RATHER THAN INFERRED FROM MATCHING TOKENS.
  // A too-WIDE page is numerically correct, so the token arm below cannot see
  // it. The conv page is the CACHE dtype (bf16, `mamba_utils.py:96-107` — conv
  // state carries `mamba_cache_dtype`, default auto -> the model dtype), and
  // the SSM page is `mamba_ssm_cache_dtype` = float32, resolved independently.
  // Making both f32 would pass every other case in this file.
  for (size_t g = 0; g < n_mamba; ++g) {
    const vllm::GdnStateCache& s = runner.gdn_state()[g];
    CHECK(s.conv_state.dtype == DType::kBF16);
    CHECK(s.ssm_state.dtype == DType::kF32);
    REQUIRE(s.conv_state.rank == 3);
    REQUIRE(s.ssm_state.rank == 4);
    CHECK(s.conv_state.shape[1] == kConvDim);
    CHECK(s.conv_state.shape[2] == kConvKernel - 1);
    CHECK(s.ssm_state.shape[1] == kMambaHeads);
    CHECK(s.ssm_state.shape[2] == kMambaHeadDim);
    CHECK(s.ssm_state.shape[3] == kStateSize);
  }
  // The two recurrent pages are DISTINCT allocations. One buffer shared by both
  // mamba layers would make layer 3 read layer 0's state, and the token arm
  // would see it only as "different tokens", never as the cause.
  CHECK(runner.gdn_state()[0].conv_state.data != runner.gdn_state()[1].conv_state.data);
  CHECK(runner.gdn_state()[0].ssm_state.data != runner.gdn_state()[1].ssm_state.data);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. THE UNIT — a multi-step decode through the runner matches the reference.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH paged: a multi-step runner decode matches the host reference token for token") {
  // f32 KV pages so the paged store carries exactly the values the dense
  // reference holds; any token difference is then a state-carry or paging
  // defect rather than a rounding one. The bf16-page arm is the sibling case
  // below.
  setenv("VT_KV_CACHE_F32", "1", 1);
  Fixture fx("float32");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  // 20 prompt tokens at chunk_size 8 => THREE logical SSD chunks, so the
  // inter-chunk state passing runs twice inside the prefill as well as between
  // the steps. Then SIX decode steps: with one leg and fresh state the
  // recurrent half is unobservable (nemotron_h_forward.h:379-382), so a
  // one-step case would gate nothing this unit adds.
  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11, 0, 6,
                                    8, 4, 13, 10, 12, 15, 1, 9, 3, 7};
  constexpr int kSteps = 6;

  const std::vector<int32_t> got =
      RunnerGreedy(runner, "R0", prompt, kSteps, {0, 1}, /*state_slot=*/0);

  // THE REFERENCE: a fresh whole-prefix forward per step. It carries no state
  // because it recomputes everything, which is exactly why it is the right
  // operand — the paged arm must reach the same answer while recomputing
  // nothing.
  vt::Queue hq = Q();
  const std::vector<int32_t> want =
      vllm::NemotronHGreedyDecode(fx.host, fx.params, prompt, kSteps, hq);

  REQUIRE(got.size() == static_cast<size_t>(kSteps));
  REQUIRE(want.size() == static_cast<size_t>(kSteps));
  MESSAGE("paged tokens: " << got[0] << "," << got[1] << "," << got[2] << "," << got[3]
                           << "," << got[4] << "," << got[5]);
  MESSAGE("reference   : " << want[0] << "," << want[1] << "," << want[2] << ","
                           << want[3] << "," << want[4] << "," << want[5]);
  // Compared over all kSteps positions, not just the first. The parent spec's
  // §6d already matched 3/3 FIRST tokens against a forward carrying no state at
  // all, which is exactly how little a first token proves.
  size_t compared = 0;
  for (int s = 0; s < kSteps; ++s) {
    CHECK_MESSAGE(got[static_cast<size_t>(s)] == want[static_cast<size_t>(s)],
                  "token " << s << " differs: paged " << got[static_cast<size_t>(s)]
                           << " vs reference " << want[static_cast<size_t>(s)]);
    ++compared;
  }
  CHECK(compared == static_cast<size_t>(kSteps));

  // A degenerate reference — every step emitting the same token — would let a
  // forward that ignored its inputs pass. Assert the sequence is not constant.
  bool varies = false;
  for (int s = 1; s < kSteps; ++s)
    if (want[static_cast<size_t>(s)] != want[0]) varies = true;
  CHECK_MESSAGE(varies, "the reference emitted a constant token sequence, so this "
                        "comparison could not have failed");
  unsetenv("VT_KV_CACHE_F32");
}

TEST_CASE("NemotronH paged: the bf16 page arm decodes the same tokens as the reference") {
  // The RELEASED checkpoint's model dtype, and the production page dtype
  // (bf16 KV store, mirroring vLLM's flash_attn cache). Both K/V and the conv
  // page round to bf16 here, which is what the shipped configuration does.
  unsetenv("VT_KV_CACHE_F32");
  Fixture fx("bfloat16");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  CHECK(runner.attn_kv()[0].dtype == DType::kBF16);

  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  constexpr int kSteps = 4;
  const std::vector<int32_t> got =
      RunnerGreedy(runner, "R0", prompt, kSteps, {0, 1}, /*state_slot=*/0);
  vt::Queue hq = Q();
  const std::vector<int32_t> want =
      vllm::NemotronHGreedyDecode(fx.host, fx.params, prompt, kSteps, hq);
  REQUIRE(got.size() == static_cast<size_t>(kSteps));
  for (int s = 0; s < kSteps; ++s) {
    CHECK_MESSAGE(got[static_cast<size_t>(s)] == want[static_cast<size_t>(s)],
                  "bf16-page token " << s << " differs");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. THE CARRY IS REAL — direct assertions on the recurrent pages.
// ═══════════════════════════════════════════════════════════════════════════

// ─── A2-D1 (#1311): the ARM RECORDER, armed through the production entry point ──
//
// WHY THIS CASE EXISTS, and it is not the swap's gate. A2-D1 replaces the
// chunked SSD kernels with the single-step ones on the DECODE rows of the
// device mamba arm. That arm is CUDA-only by construction — it enters through
// an FP8 W8A8 `in_proj` and `kMatmulFp8CublasLt` is registered ONLY by
// `cuda_matmul.cu:920` — so no case in this CPU suite can execute the swapped
// code, and saying so is better than a case that pretends otherwise. The swap
// itself is gated on the real checkpoint (`scripts/nemotron-h-a2q1-dgx-gate.sh`
// steps 7 and 8, the same-binary `VT_NEMOTRON_H_MAMBA_DECODE_STEP` A/B) and at
// the op level by the two driver-group equivalence cases in
// `tests/vt/test_ops_mamba2_state_update.cpp`.
//
// ★ WHAT THIS CASE DOES GATE IS THE INSTRUMENT THOSE RUNS ARE READ WITH.
// The dgx evidence is a set of COUNTERS: "the decode step launched 0 chunk
// scans and 23 state-update rows". A counter that is never incremented reports
// exactly the same thing as a kernel that is never launched
// ([[absent-hook-looks-like-armed-instrument]]), so a broken recorder would
// present as a triumphant GREEN. This case drives the counters through
// `ModelRegistry::Forward` on the arm that IS reachable here — the host mamba
// branch, whose gather/scatter A2-D1 leaves alone — and asserts NON-ZERO counts
// against the fixture's own geometry. If the recorder stops recording, this
// goes red on a CPU box in seconds, before anyone spends a GPU window reading a
// zero as a result.
TEST_CASE("NemotronH paged: the recurrent arm recorder counts what the step launched") {
  unsetenv("VT_KV_CACHE_F32");
  Fixture fx("bfloat16");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  // The fixture schedule is `M E * M E`, so TWO mamba layers. Every recurrent
  // layer of a step gathers twice (conv + ssm) and scatters twice, which is the
  // per-step cost A2-D1 removes from the decode half of the DEVICE arm.
  constexpr int64_t kMambaLayers = 2;
  constexpr int64_t kPerLayerGathers = 2;

  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11};

  // Read-and-reset first, so nothing an earlier case in this binary contributed
  // is counted here.
  (void)vllm::NemotronHTakeMambaArmCounts();

  std::vector<NewRequestData> reqs;
  reqs.push_back(MakeNewReq("REC0", prompt, {0, 1}, /*state_slot=*/0));
  std::map<std::string, int> sched;
  sched["REC0"] = static_cast<int>(prompt.size());
  SchedulerOutput s1 = NewStep(std::move(reqs), std::move(sched));
  CHECK_FALSE(runner.execute_model(s1).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  const vllm::NemotronHMambaArmCounts pf = vllm::NemotronHTakeMambaArmCounts();

  // ONE decode step, measured on its own.
  std::vector<std::string> ids{"REC0"};
  std::vector<int> computed{static_cast<int>(prompt.size())};
  std::vector<int> outputs{1};
  SchedulerOutput sd = DecodeStep(ids, computed, outputs);
  CHECK_FALSE(runner.execute_model(sd).has_value());
  vllm::v1::ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
  REQUIRE(md.sampled_token_ids.size() == 1);
  const vllm::NemotronHMambaArmCounts dec = vllm::NemotronHTakeMambaArmCounts();

  MESSAGE("prefill step: gathers=" << pf.state_gathers << " scatters=" << pf.state_scatters
                                   << " chunk_scan_calls=" << pf.chunk_scan_calls
                                   << " state_update_rows=" << pf.state_update_rows
                                   << " conv_fwd_calls=" << pf.conv_fwd_calls
                                   << " conv_update_rows=" << pf.conv_update_rows);
  MESSAGE("decode step : gathers=" << dec.state_gathers << " scatters=" << dec.state_scatters
                                   << " chunk_scan_calls=" << dec.chunk_scan_calls
                                   << " state_update_rows=" << dec.state_update_rows
                                   << " conv_fwd_calls=" << dec.conv_fwd_calls
                                   << " conv_update_rows=" << dec.conv_update_rows);

  // ★ THE COUNTS ARE ASSERTED AGAINST THE FIXTURE'S GEOMETRY, not against
  // "> 0". A recorder that incremented once per FORWARD instead of once per
  // LAYER would satisfy a `> 0` assertion and misreport every dgx number by a
  // factor of the layer count.
  CHECK(pf.state_gathers == kMambaLayers * kPerLayerGathers);
  CHECK(pf.state_scatters == kMambaLayers * kPerLayerGathers);
  CHECK(dec.state_gathers == kMambaLayers * kPerLayerGathers);
  CHECK(dec.state_scatters == kMambaLayers * kPerLayerGathers);

  // THIS TEST'S QUEUE has no FP8 GEMM -- and that is a statement about the
  // queue, not about the box: `Q()` is a CPU queue, `kMatmulFp8CublasLt` is
  // registered only by `cuda_matmul.cu:920`, and the paged forward selects the
  // device mamba arm by asking the OP TABLE for the queue's device. So this
  // case reads the same on a GPU box as on a CPU one, which is why it is a
  // stable floor rather than a property of wherever it happens to run.
  // Asserting the zero here is what makes the non-zero on a leased GPU mean
  // something: the two counters are wired to the `vt::` call sites and not to
  // the branch condition.
  CHECK(dec.state_update_rows == 0);
  CHECK(dec.conv_update_rows == 0);
  CHECK(dec.chunk_scan_calls == 0);  // the HOST mixer's scan is not this counter's
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, Q().device.type));

  // And the reader really does RESET: a second read with no forward between
  // must be all zeros, or every dgx figure would be a running total.
  const vllm::NemotronHMambaArmCounts again = vllm::NemotronHTakeMambaArmCounts();
  CHECK(again.state_gathers == 0);
  CHECK(again.state_scatters == 0);
  CHECK(again.chunk_scan_calls == 0);
  CHECK(again.state_update_rows == 0);
}

TEST_CASE("NemotronH paged: the recurrent pages carry state across steps and are indexed") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  Fixture fx("float32");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  REQUIRE(runner.gdn_state().size() == 2);

  // Every recurrent slot starts at zero, and the run below writes exactly one
  // of them. Establish the "before" so "after" means something.
  const int64_t slots = runner.gdn_state()[0].conv_state.shape[0];
  REQUIRE(slots >= 2);
  size_t zero_elems = 0;
  for (int64_t s = 0; s < slots; ++s) {
    for (const vllm::GdnStateCache& c : runner.gdn_state()) {
      for (float v : ReadStateRow(c.conv_state, s)) {
        CHECK(v == 0.0F);
        ++zero_elems;
      }
      for (float v : ReadStateRow(c.ssm_state, s)) {
        CHECK(v == 0.0F);
        ++zero_elems;
      }
    }
  }
  // The count the geometry predicts: slots x layers x (conv row + ssm row).
  const size_t conv_row = static_cast<size_t>(kConvDim) * (kConvKernel - 1);
  const size_t ssm_row =
      static_cast<size_t>(kMambaHeads) * kMambaHeadDim * kStateSize;
  CHECK(zero_elems == static_cast<size_t>(slots) * 2 * (conv_row + ssm_row));

  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  (void)RunnerGreedy(runner, "R0", prompt, /*steps=*/3, {0, 1}, /*state_slot=*/0);

  // ★ THE SLOT THE RUNNER ACTUALLY ASSIGNED, READ FROM THE METADATA. It is NOT
  // the block id the scheduler handed over: `remap_gdn_state_slots`
  // (runner.cpp:993-1050) keys the compact state slot on the sequence's
  // IDENTITY, deliberately, because once a sequence exceeds one mamba block the
  // block table's column 0 collapses to the shared null block id and every long
  // concurrent sequence would map to one slot. So the test reads what the
  // runner decided rather than asserting what it was told.
  const vllm::v1::GDNAttentionMetadata& gm = runner.last_gdn_meta();
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  REQUIRE(gm.non_spec_state_indices_tensor->size() == 1);
  const int64_t used = (*gm.non_spec_state_indices_tensor)[0];
  REQUIRE(used >= 0);
  REQUIRE(used < slots);
  MESSAGE("the runner assigned recurrent state slot " << used);

  size_t written_layers = 0;
  size_t untouched_elems = 0;
  for (size_t g = 0; g < runner.gdn_state().size(); ++g) {
    const vllm::GdnStateCache& c = runner.gdn_state()[g];
    const std::vector<float> conv = ReadStateRow(c.conv_state, used);
    const std::vector<float> ssm = ReadStateRow(c.ssm_state, used);
    REQUIRE(conv.size() == conv_row);
    REQUIRE(ssm.size() == ssm_row);
    CHECK_MESSAGE(MaxAbs(conv) > 0.0, "layer " << g << ": the conv page slot the "
                                                  "metadata named was never written");
    CHECK_MESSAGE(MaxAbs(ssm) > 0.0, "layer " << g << ": the SSM page slot the "
                                                 "metadata named was never written");
    ++written_layers;
    // Every OTHER slot is untouched. This is what says the write is CONFINED to
    // the row the metadata named — the property that keeps two sequences from
    // corrupting each other once A2-B lifts the request count. It cannot, on its
    // own, distinguish an indexed write from a hardcoded 0 at `num_reqs == 1`,
    // because the runner always assigns the first live sequence slot 0
    // (`gdn_free_slots_` is built descending at runner.cpp:522-523, so `back()`
    // is 0). Spec §4.1 predicted exactly that, and the sibling case below is the
    // instrument that CAN tell them apart.
    for (int64_t s = 0; s < slots; ++s) {
      if (s == used) continue;
      for (float v : ReadStateRow(c.conv_state, s)) {
        CHECK_MESSAGE(v == 0.0F, "layer " << g << ": conv slot " << s
                                          << " was written although the request was "
                                             "admitted on slot " << used);
        ++untouched_elems;
      }
      for (float v : ReadStateRow(c.ssm_state, s)) {
        CHECK_MESSAGE(v == 0.0F, "layer " << g << ": SSM slot " << s
                                          << " was written although the request was "
                                             "admitted on slot " << used);
        ++untouched_elems;
      }
    }
  }
  CHECK(written_layers == 2);
  CHECK(untouched_elems ==
        2 * static_cast<size_t>(slots - 1) * (conv_row + ssm_row));
  unsetenv("VT_KV_CACHE_F32");
}

// NO COMMA IN THIS NAME — doctest's `-tc` filter splits on commas, so a comma
// here would make a mutation pass select ZERO cases and print `SUCCESS!` with a
// zero exit code. That shape has already read GREEN in this tree for a whole
// mutation pass including the row that deleted the guard.
TEST_CASE("NemotronH paged: the recurrent slot is INDEXED from the metadata rather than hardcoded") {
  // ★ §4.1, and the reason this case exists at all. A forward that replaced the
  // metadata's state index with a literal `0` passes EVERY runner-driven case in
  // this file, because at `num_reqs <= 1` the runner never assigns anything but
  // slot 0 (`gdn_free_slots_` is built descending, runner.cpp:522-523). The
  // indexing machinery still lands in A2-P — only the COUNT is one — so it needs
  // an instrument, and the only one that can distinguish the two is a step whose
  // metadata names a slot the runner would not have chosen.
  //
  // This case therefore calls `NemotronHPagedForward` with the runner's own
  // caches and the runner's own metadata, with ONE field changed: the state
  // index. Reachability is not what it proves — the seven runner-driven cases
  // above already do that, and mutation P-M7 is their proof. What it proves is
  // that the slot vector is read.
  setenv("VT_KV_CACHE_F32", "1", 1);
  Fixture fx("float32");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  const int64_t T = static_cast<int64_t>(prompt.size());
  std::vector<NewRequestData> reqs;
  reqs.push_back(MakeNewReq("R0", prompt, {0, 1}, /*state_slot=*/0));
  std::map<std::string, int> sched;
  sched["R0"] = static_cast<int>(prompt.size());
  SchedulerOutput s1 = NewStep(std::move(reqs), std::move(sched));
  CHECK_FALSE(runner.execute_model(s1).has_value());
  (void)runner.sample_tokens(std::nullopt);

  // Reset every recurrent page, so "written" below means "written by the call
  // this case makes".
  auto& pages = const_cast<std::vector<vllm::GdnStateCache>&>(runner.gdn_state());
  for (vllm::GdnStateCache& c : pages) {
    for (vt::Tensor* p : {&c.conv_state, &c.ssm_state}) {
      int64_t n = 1;
      for (int r = 0; r < p->rank; ++r) n *= p->shape[r];
      std::memset(p->data, 0, static_cast<size_t>(n) * vt::SizeOf(p->dtype));
    }
  }
  const int64_t slots = pages[0].conv_state.shape[0];
  REQUIRE(slots >= 2);

  // The runner's metadata, with the state index moved to slot 1. Everything
  // else — the block table, the slot mapping, the query offsets, the
  // has_initial_state mask — is the runner's own.
  vllm::v1::GDNAttentionMetadata gm = runner.last_gdn_meta();
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  REQUIRE(gm.non_spec_state_indices_tensor->size() == 1);
  REQUIRE((*gm.non_spec_state_indices_tensor)[0] == 0);  // else this proves nothing
  (*gm.non_spec_state_indices_tensor)[0] = 1;
  REQUIRE(gm.prefill_state_indices.has_value());
  (*gm.prefill_state_indices)[0] = 1;

  vt::Queue hq = Q();
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  std::vector<int32_t> positions(static_cast<size_t>(T), 0);
  vllm::ModelForwardInput input{
      .token_ids = prompt,
      .positions = positions,
      .attn_meta = runner.last_attn_meta(),
      .gdn_meta = gm,
      .attn_kv = const_cast<std::vector<vllm::PagedKvCache>&>(runner.attn_kv()),
      .gdn_state = pages,
      .config = fx.cfg,
      .queue = hq,
      .logits_indices = logits_indices,
      .num_reqs = 1};
  (void)vllm::NemotronHPagedForward(fx.host, fx.params, input, nullptr);

  size_t checked = 0;
  for (size_t g = 0; g < pages.size(); ++g) {
    const std::vector<float> at1 = ReadStateRow(pages[g].ssm_state, 1);
    const std::vector<float> at0 = ReadStateRow(pages[g].ssm_state, 0);
    CHECK_MESSAGE(MaxAbs(at1) > 0.0,
                  "layer " << g << ": the SSM row the metadata named (slot 1) was "
                                   "never written -- the slot index is not read");
    for (float v : at0) {
      CHECK_MESSAGE(v == 0.0F,
                    "layer " << g << ": slot 0 was written although the metadata "
                                     "named slot 1 -- the index is hardcoded");
      ++checked;
    }
    const std::vector<float> cv1 = ReadStateRow(pages[g].conv_state, 1);
    const std::vector<float> cv0 = ReadStateRow(pages[g].conv_state, 0);
    CHECK_MESSAGE(MaxAbs(cv1) > 0.0,
                  "layer " << g << ": the conv row the metadata named (slot 1) was "
                                   "never written");
    for (float v : cv0) {
      CHECK_MESSAGE(v == 0.0F,
                    "layer " << g << ": conv slot 0 was written although the "
                                     "metadata named slot 1");
      ++checked;
    }
  }
  const size_t conv_row = static_cast<size_t>(kConvDim) * (kConvKernel - 1);
  const size_t ssm_row = static_cast<size_t>(kMambaHeads) * kMambaHeadDim * kStateSize;
  CHECK(checked == 2 * (conv_row + ssm_row));
  unsetenv("VT_KV_CACHE_F32");
}

TEST_CASE("NemotronH paged: a fresh request on a REUSED slot does not read the previous tenant") {
  // ★ THE ZEROING OBLIGATION (gdn_attn.h:126-139) — the silent-wrong-answer
  // path in this unit, and the one a token gate designed without it in mind
  // absorbs. The recurrence kernels read the state buffer UNCONDITIONALLY, so a
  // request whose `has_initial_state` is 0 must be handed ZEROS. Here the same
  // state slot is used twice by two different requests: if the gather's zeroing
  // is dropped, the second request continues the first one's recurrence and
  // emits different tokens from a fresh reference.
  setenv("VT_KV_CACHE_F32", "1", 1);
  Fixture fx("float32");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  const std::vector<int32_t> first{5, 12, 2, 9, 14, 1, 7, 3, 11, 6};
  const std::vector<int32_t> second{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  constexpr int kSteps = 4;

  // Request A leaves state in slot 0 and KV in blocks {0,1}.
  (void)RunnerGreedy(runner, "RA", first, kSteps, {0, 1}, /*state_slot=*/0);
  const std::vector<float> after_a = ReadStateRow(runner.gdn_state()[0].ssm_state, 0);
  REQUIRE(MaxAbs(after_a) > 0.0);

  // Request B is admitted on the SAME state slot with fresh KV blocks. It is a
  // new sequence, so `has_initial_state` is 0 and it must not see A's rows.
  const std::vector<int32_t> got =
      RunnerGreedy(runner, "RB", second, kSteps, {2, 3}, /*state_slot=*/0);
  vt::Queue hq = Q();
  const std::vector<int32_t> want =
      vllm::NemotronHGreedyDecode(fx.host, fx.params, second, kSteps, hq);
  REQUIRE(got.size() == static_cast<size_t>(kSteps));
  for (int s = 0; s < kSteps; ++s) {
    CHECK_MESSAGE(got[static_cast<size_t>(s)] == want[static_cast<size_t>(s)],
                  "reused-slot token " << s << " differs: the fresh request read the "
                                              "previous tenant's recurrent state");
  }
  // ...and the gate is not vacuous: A and B really do produce different state,
  // so "did not read A's rows" is a distinguishable property.
  const std::vector<float> after_b = ReadStateRow(runner.gdn_state()[0].ssm_state, 0);
  REQUIRE(after_b.size() == after_a.size());
  bool differs = false;
  for (size_t i = 0; i < after_a.size(); ++i)
    if (after_a[i] != after_b[i]) differs = true;
  CHECK_MESSAGE(differs, "the two requests left byte-identical recurrent state, so a "
                         "stale-state defect would be invisible here");
  unsetenv("VT_KV_CACHE_F32");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. THE PER-BLOCK NUMERIC ARM — a mechanism can be missing while the argmax
//    is unchanged (porting-a-model.md §3), so tokens alone are not enough.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH paged: every layer's activations match the host reference at prefill") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  Fixture fx("float32");
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  const int64_t T = static_cast<int64_t>(prompt.size());

  // Drive ONE prefill step through the runner. This is what BUILDS the step
  // metadata and the block table — the whole point is that the paged arm below
  // runs on the runner's own `CommonAttentionMetadata` and
  // `GDNAttentionMetadata`, not on a fabricated one.
  std::vector<NewRequestData> reqs;
  reqs.push_back(MakeNewReq("R0", prompt, {0, 1}, /*state_slot=*/0));
  std::map<std::string, int> sched;
  sched["R0"] = static_cast<int>(prompt.size());
  SchedulerOutput s1 = NewStep(std::move(reqs), std::move(sched));
  CHECK_FALSE(runner.execute_model(s1).has_value());
  (void)runner.sample_tokens(std::nullopt);

  // Reset the RECURRENT pages so the traced re-run below sees the same fresh
  // state the first prefill saw. The attention pages need no reset: a prefill
  // rewrites every slot it then reads, so replaying it is idempotent, while the
  // recurrent pages are advanced IN PLACE and replaying over them would carry
  // the prompt twice. Reaching into the runner's buffers is surgery a test may
  // do and production may not, and it is stated here rather than hidden.
  auto& pages = const_cast<std::vector<vllm::GdnStateCache>&>(runner.gdn_state());
  size_t zeroed = 0;
  for (vllm::GdnStateCache& c : pages) {
    for (vt::Tensor* p : {&c.conv_state, &c.ssm_state}) {
      int64_t n = 1;
      for (int r = 0; r < p->rank; ++r) n *= p->shape[r];
      std::memset(p->data, 0, static_cast<size_t>(n) * vt::SizeOf(p->dtype));
      zeroed += static_cast<size_t>(n);
    }
  }
  const size_t conv_row = static_cast<size_t>(kConvDim) * (kConvKernel - 1);
  const size_t ssm_row = static_cast<size_t>(kMambaHeads) * kMambaHeadDim * kStateSize;
  const size_t slots = static_cast<size_t>(runner.gdn_state()[0].conv_state.shape[0]);
  CHECK(zeroed == slots * 2 * (conv_row + ssm_row));

  // ── the PAGED trace, over the runner's own caches and metadata ──
  vt::Queue hq = Q();
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  // NemotronH reads no positional information of any kind
  // (`kNemotronHAttentionHasNoRope`), so this vector exists only to satisfy the
  // struct; the runner's own `StepInputs::positions` is i64 and is not the type
  // this field takes.
  std::vector<int32_t> positions(static_cast<size_t>(T), 0);
  vllm::ModelForwardInput input{
      .token_ids = prompt,
      .positions = positions,
      .attn_meta = runner.last_attn_meta(),
      .gdn_meta = runner.last_gdn_meta(),
      .attn_kv = const_cast<std::vector<vllm::PagedKvCache>&>(runner.attn_kv()),
      .gdn_state = pages,
      .config = fx.cfg,
      .queue = hq,
      .logits_indices = logits_indices,
      .num_reqs = 1};
  NemotronHTrace got;
  got.capture = true;
  (void)vllm::NemotronHPagedForward(fx.host, fx.params, input, &got);

  // ── the HOST reference's trace over the same prompt ──
  NemotronHTrace want;
  want.capture = true;
  (void)vllm::NemotronHForward(fx.host, fx.params, prompt, logits_indices, hq, &want);

  // The comparison, LAYER BY LAYER. A token comparison cannot see a dropped
  // mechanism whose argmax is unchanged; this can, because a paged attention
  // read that spanned the wrong blocks, or a recurrent gather that fetched the
  // wrong row, moves the attention or mamba layer's mixer output long before it
  // moves a token.
  const int64_t L = fx.params.num_hidden_layers();
  REQUIRE(want.mixer.size() == static_cast<size_t>(L));
  REQUIRE(got.mixer.size() == static_cast<size_t>(L));
  // The f32 band, derived from what these two arms actually agree to (reported
  // by every CHECK_MESSAGE below when it fails) rather than invented.
  constexpr double kRelF32 = 2e-4;
  size_t elements = 0;
  int layers_compared = 0;
  double worst_rel = 0.0;
  for (int64_t l = 0; l < L; ++l) {
    const std::string tag = "layer " + std::to_string(l) + " (" +
                            BlockName(fx.params.layers_block_type[static_cast<size_t>(l)]) +
                            ")";
    elements += ExpectCloseRel(got.normed[static_cast<size_t>(l)],
                               want.normed[static_cast<size_t>(l)], kRelF32,
                               tag + " normed", &worst_rel);
    elements += ExpectCloseRel(got.mixer[static_cast<size_t>(l)],
                               want.mixer[static_cast<size_t>(l)], kRelF32,
                               tag + " mixer", &worst_rel);
    elements += ExpectCloseRel(got.hidden[static_cast<size_t>(l)],
                               want.hidden[static_cast<size_t>(l)], kRelF32,
                               tag + " hidden", &worst_rel);
    ++layers_compared;
  }
  elements += ExpectCloseRel(got.final_normed, want.final_normed, kRelF32, "final_normed",
                             &worst_rel);
  MESSAGE("prefill: worst RELATIVE deviation over every layer = " << worst_rel
          << " against a band of " << kRelF32);

  // ★ THE INSTRUMENT REPORTS HOW MANY THINGS IT EXAMINED. "worst deviation: 0"
  // over zero elements is a mute switch, not a pass — so the element count is
  // asserted against the geometry the config predicts.
  CHECK(layers_compared == static_cast<int>(L));
  CHECK(elements == static_cast<size_t>((3 * L + 1) * T * kHidden));
  MESSAGE("compared " << elements << " activation elements over " << layers_compared
                      << " layers at T=" << T);
  unsetenv("VT_KV_CACHE_F32");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4b. ★ THE DECODE-STEP CARRY, ASSERTED DIRECTLY.
//
//     THIS CASE EXISTS BECAUSE THE TOKEN ARM COULD NOT SEE THE CARRY, AND THAT
//     IS A MEASURED RESULT RATHER THAN a precaution. Running the §5.6 mutation
//     pass on this fixture, P-M1 (zero the carried SSM state every step), P-M2
//     (zero the carried conv state), P-M4 (drop the fresh-request zeroing) and
//     P-M9 (invert the decode/prefill classification) ALL SURVIVED the
//     token-comparison arms: with both recurrent states zeroed on every step
//     the paged decode still emitted 26,17,4,20,2,23 — byte-identical to the
//     reference. At this geometry the residual stream is dominated by the MoE
//     block (`routed_scaling_factor` 2.5 on the routed sum) and the argmax over
//     32 vocabulary entries simply does not move.
//
//     The spec's §8.1 stop condition says what that is: "P-M4 or P-M6 stays
//     green -> a coverage hole, recorded as a finding, with the direct
//     assertion the spec then owes. Not a pass." This is that assertion, and it
//     is NUMERIC on purpose — a dropped carry moves the hidden state long
//     before it moves a token, which is the whole reason
//     `porting-a-model.md` §3 asks for per-layer activations.
//
//     The construction, in three engines, because a decode step CONSUMES the
//     state it reads and cannot be replayed:
//       A  prefill + one decode through the runner, purely to obtain the
//          runner's own DECODE metadata (T=1, seq_lens=[T+1], the slot mapping
//          for position T).
//       B  the same prefill only, so its pages hold the post-prefill state.
//          The decode forward is then driven over B's pages with A's metadata —
//          valid because both requests are the same prompt on the same blocks
//          and the same state slot, which the case asserts rather than assumes.
//       the host reference over `prompt + tok`, whose LAST row is what B's
//          single decode row must equal.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH paged: a decode step's per-layer output equals the reference's last row") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  const int64_t T = static_cast<int64_t>(prompt.size());

  // ── engine A: prefill + one decode, for the decode metadata ──
  Fixture fa("float32");
  KVCacheConfig kva =
      fa.model->registration().factory->make_kv_cache(fa.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner ra(fa.cfg, *fa.model, kva, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                    /*max_num_batched_tokens=*/64);
  std::vector<NewRequestData> qa;
  qa.push_back(MakeNewReq("R0", prompt, {0, 1}, 0));
  std::map<std::string, int> sa;
  sa["R0"] = static_cast<int>(prompt.size());
  SchedulerOutput pa_step = NewStep(std::move(qa), std::move(sa));
  CHECK_FALSE(ra.execute_model(pa_step).has_value());
  vllm::v1::ModelRunnerOutput oa = ra.sample_tokens(std::nullopt);
  REQUIRE(oa.sampled_token_ids.size() == 1);
  REQUIRE(oa.sampled_token_ids[0].size() == 1);
  const int32_t tok = oa.sampled_token_ids[0][0];

  SchedulerOutput da = DecodeStep({"R0"}, {static_cast<int>(T)}, {1});
  CHECK_FALSE(ra.execute_model(da).has_value());
  (void)ra.sample_tokens(std::nullopt);
  const vllm::v1::CommonAttentionMetadata& dm = ra.last_attn_meta();
  const vllm::v1::GDNAttentionMetadata& dg = ra.last_gdn_meta();
  // The metadata really is a DECODE step, and it really is one token.
  REQUIRE(dm.num_reqs == 1);
  REQUIRE(dm.num_actual_tokens == 1);
  REQUIRE(dm.seq_lens.size() == 1);
  CHECK(dm.seq_lens[0] == static_cast<int32_t>(T + 1));
  REQUIRE(dm.slot_mapping.size() == 1);
  CHECK(dm.slot_mapping[0] == T);  // block 0, offset T (block_size 16 > T)
  CHECK(dg.num_decodes == 1);
  CHECK(dg.num_prefills == 0);

  // ── engine B: the same prefill only ──
  Fixture fb("float32");
  KVCacheConfig kvb =
      fb.model->registration().factory->make_kv_cache(fb.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner rb(fb.cfg, *fb.model, kvb, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                    /*max_num_batched_tokens=*/64);
  std::vector<NewRequestData> qb;
  qb.push_back(MakeNewReq("R0", prompt, {0, 1}, 0));
  std::map<std::string, int> sb;
  sb["R0"] = static_cast<int>(prompt.size());
  SchedulerOutput pb_step = NewStep(std::move(qb), std::move(sb));
  CHECK_FALSE(rb.execute_model(pb_step).has_value());
  (void)rb.sample_tokens(std::nullopt);
  // A's decode metadata is only valid over B's pages if the two prefills agreed
  // on the block table and the state slot. Assert it rather than assume it.
  const bool same_blocks =
      rb.last_attn_meta().block_table_tensor == ra.last_attn_meta().block_table_tensor;
  REQUIRE(same_blocks);
  REQUIRE(dg.non_spec_state_indices_tensor.has_value());
  REQUIRE(rb.last_gdn_meta().non_spec_state_indices_tensor.has_value());
  CHECK((*dg.non_spec_state_indices_tensor)[0] ==
        (*rb.last_gdn_meta().non_spec_state_indices_tensor)[0]);
  // B's pages carry a NON-ZERO recurrent state, or "the carry was read" would be
  // indistinguishable from "the carry was zero anyway".
  const std::vector<float> carried =
      ReadStateRow(rb.gdn_state()[0].ssm_state, (*dg.non_spec_state_indices_tensor)[0]);
  REQUIRE(MaxAbs(carried) > 0.0);

  // ── the decode forward over B's pages, traced ──
  vt::Queue hq = Q();
  const std::vector<int32_t> dec_tokens{tok};
  const std::vector<int32_t> dec_logits{0};
  const std::vector<int32_t> dec_positions{static_cast<int32_t>(T)};
  vllm::ModelForwardInput dec{
      .token_ids = dec_tokens,
      .positions = dec_positions,
      .attn_meta = dm,
      .gdn_meta = dg,
      .attn_kv = const_cast<std::vector<vllm::PagedKvCache>&>(rb.attn_kv()),
      .gdn_state = const_cast<std::vector<vllm::GdnStateCache>&>(rb.gdn_state()),
      .config = fb.cfg,
      .queue = hq,
      .logits_indices = dec_logits,
      .num_reqs = 1};
  NemotronHTrace got;
  got.capture = true;
  (void)vllm::NemotronHPagedForward(fb.host, fb.params, dec, &got);

  // ── the reference over `prompt + tok`, whose LAST row is the answer ──
  std::vector<int32_t> full = prompt;
  full.push_back(tok);
  NemotronHTrace want;
  want.capture = true;
  (void)vllm::NemotronHForward(fb.host, fb.params, full, {static_cast<int32_t>(T)}, hq,
                               &want);

  const int64_t L = fb.params.num_hidden_layers();
  REQUIRE(got.mixer.size() == static_cast<size_t>(L));
  REQUIRE(want.mixer.size() == static_cast<size_t>(L));
  // MEASURED, not invented. A decode step and the reference's last row are two
  // genuinely different reduction orders over the same mathematics — a 1-token
  // scan from carried state and a paged attention read of 13 cached rows,
  // against a 13-token chunk scan and a dense attention over the same 13 rows —
  // so f32 associativity separates them by more than the 2e-4 the whole-prefill
  // arm uses. Driven down to 1e-9 the two arms actually agree to 1.2e-3
  // relative; the band is that measurement rounded up by one significant
  // figure, and `ExpectCloseRel` REQUIREs it still rejects an all-zeros answer.
  constexpr double kRelDecode = 2e-3;
  size_t elements = 0;
  int layers = 0;
  double worst_rel = 0.0;
  for (int64_t l = 0; l < L; ++l) {
    const std::string tag =
        "decode layer " + std::to_string(l) + " (" +
        BlockName(fb.params.layers_block_type[static_cast<size_t>(l)]) + ")";
    // The reference's row at position T; the paged decode's only row.
    const std::vector<float>& wm = want.mixer[static_cast<size_t>(l)];
    REQUIRE(wm.size() == static_cast<size_t>((T + 1) * kHidden));
    const std::vector<float> wrow(
        wm.begin() + static_cast<std::ptrdiff_t>(T * kHidden),
        wm.begin() + static_cast<std::ptrdiff_t>((T + 1) * kHidden));
    const std::vector<float>& grow = got.mixer[static_cast<size_t>(l)];
    REQUIRE(grow.size() == static_cast<size_t>(kHidden));
    elements += ExpectCloseRel(grow, wrow, kRelDecode, tag + " mixer", &worst_rel);
    ++layers;
  }
  // ...and the final normed hidden, which is what lm_head sees.
  {
    REQUIRE(want.final_normed.size() == static_cast<size_t>((T + 1) * kHidden));
    const std::vector<float> wrow(
        want.final_normed.begin() + static_cast<std::ptrdiff_t>(T * kHidden),
        want.final_normed.end());
    REQUIRE(got.final_normed.size() == static_cast<size_t>(kHidden));
    elements += ExpectCloseRel(got.final_normed, wrow, kRelDecode, "decode final_normed",
                               &worst_rel);
  }
  CHECK(layers == static_cast<int>(L));
  CHECK(elements == static_cast<size_t>((L + 1) * kHidden));
  MESSAGE("decode-step carry: compared " << elements << " elements over " << layers
                                         << " layers at T=1 against the reference's row "
                                         << T << "; worst RELATIVE deviation "
                                         << worst_rel << " against a band of "
                                         << kRelDecode);
  unsetenv("VT_KV_CACHE_F32");
}

TEST_CASE("NemotronH paged: a fresh prefill over a DIRTY state slot equals a fresh reference") {
  // ★ THE ZEROING OBLIGATION (gdn_attn.h:126-139), ASSERTED NUMERICALLY.
  //
  // The token-level sibling of this case (the REUSED-slot one above) could not
  // see it: with the zeroing dropped, the second request continued the first
  // one's recurrence and STILL emitted the same tokens — mutations P-M4 (drop
  // the zeroing) and P-M9 (classify every row as a decode, which reaches the
  // same place by never asking for the mask) both survived every token arm.
  // That is the coverage hole the spec's §8.1 says to close with a direct
  // assertion rather than record as a pass.
  //
  // Two engines, because the condition under test is a page that is DIRTY when
  // a fresh request arrives:
  //   A  prefills request RA, leaving a non-zero recurrent state in the slot.
  //   B  prefills request RB through the runner, purely for RB's metadata —
  //      whose `prefill_has_initial_state` is 0, because RB is a new sequence.
  // The forward under test then runs RB's tokens with RB's metadata over A's
  // DIRTY pages. A correct forward zeros the row it gathers and reproduces the
  // fresh reference; one that does not carries RA's recurrence into RB.
  setenv("VT_KV_CACHE_F32", "1", 1);
  const std::vector<int32_t> first{5, 12, 2, 9, 14, 1, 7, 3, 11, 6};
  const std::vector<int32_t> second{1, 7, 3, 9, 2, 14, 5, 11, 0, 6, 8, 4};
  const int64_t T = static_cast<int64_t>(second.size());

  Fixture fa("float32");
  KVCacheConfig kva =
      fa.model->registration().factory->make_kv_cache(fa.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner ra(fa.cfg, *fa.model, kva, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                    /*max_num_batched_tokens=*/64);
  std::vector<NewRequestData> qa;
  qa.push_back(MakeNewReq("RA", first, {0, 1}, 0));
  std::map<std::string, int> sa;
  sa["RA"] = static_cast<int>(first.size());
  SchedulerOutput pa_step = NewStep(std::move(qa), std::move(sa));
  CHECK_FALSE(ra.execute_model(pa_step).has_value());
  (void)ra.sample_tokens(std::nullopt);

  Fixture fb("float32");
  KVCacheConfig kvb =
      fb.model->registration().factory->make_kv_cache(fb.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner rb(fb.cfg, *fb.model, kvb, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                    /*max_num_batched_tokens=*/64);
  std::vector<NewRequestData> qb;
  qb.push_back(MakeNewReq("RB", second, {0, 1}, 0));
  std::map<std::string, int> sb;
  sb["RB"] = static_cast<int>(second.size());
  SchedulerOutput pb_step = NewStep(std::move(qb), std::move(sb));
  CHECK_FALSE(rb.execute_model(pb_step).has_value());
  (void)rb.sample_tokens(std::nullopt);

  const vllm::v1::CommonAttentionMetadata& bm = rb.last_attn_meta();
  const vllm::v1::GDNAttentionMetadata& bg = rb.last_gdn_meta();
  REQUIRE(bg.num_prefills == 1);
  REQUIRE(bg.prefill_has_initial_state.has_value());
  REQUIRE(bg.prefill_has_initial_state->size() == 1);
  // THE PRECONDITION OF THE WHOLE CASE: RB is a fresh sequence, so its mask is
  // 0 and the row it gathers must be zeroed.
  REQUIRE((*bg.prefill_has_initial_state)[0] == 0);
  REQUIRE(bg.non_spec_state_indices_tensor.has_value());
  const int64_t slot = (*bg.non_spec_state_indices_tensor)[0];

  // ...and A's page at that slot really is DIRTY, or "did not read it" would be
  // indistinguishable from "there was nothing to read".
  auto& dirty = const_cast<std::vector<vllm::GdnStateCache>&>(ra.gdn_state());
  size_t dirty_layers = 0;
  for (const vllm::GdnStateCache& c : dirty) {
    if (MaxAbs(ReadStateRow(c.ssm_state, slot)) > 0.0 &&
        MaxAbs(ReadStateRow(c.conv_state, slot)) > 0.0) {
      ++dirty_layers;
    }
  }
  REQUIRE(dirty_layers == dirty.size());

  vt::Queue hq = Q();
  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  std::vector<int32_t> positions(static_cast<size_t>(T), 0);
  vllm::ModelForwardInput input{
      .token_ids = second,
      .positions = positions,
      .attn_meta = bm,
      .gdn_meta = bg,
      .attn_kv = const_cast<std::vector<vllm::PagedKvCache>&>(ra.attn_kv()),
      .gdn_state = dirty,
      .config = fa.cfg,
      .queue = hq,
      .logits_indices = logits_indices,
      .num_reqs = 1};
  NemotronHTrace got;
  got.capture = true;
  (void)vllm::NemotronHPagedForward(fb.host, fb.params, input, &got);

  NemotronHTrace want;
  want.capture = true;
  (void)vllm::NemotronHForward(fb.host, fb.params, second, logits_indices, hq, &want);

  const int64_t L = fb.params.num_hidden_layers();
  REQUIRE(got.mixer.size() == static_cast<size_t>(L));
  constexpr double kRelF32 = 2e-4;
  size_t elements = 0;
  int layers = 0;
  double worst_rel = 0.0;
  for (int64_t l = 0; l < L; ++l) {
    const std::string tag =
        "dirty-slot layer " + std::to_string(l) + " (" +
        BlockName(fb.params.layers_block_type[static_cast<size_t>(l)]) + ")";
    elements += ExpectCloseRel(got.mixer[static_cast<size_t>(l)],
                               want.mixer[static_cast<size_t>(l)], kRelF32,
                               tag + " mixer", &worst_rel);
    ++layers;
  }
  elements += ExpectCloseRel(got.final_normed, want.final_normed, kRelF32,
                             "dirty-slot final_normed", &worst_rel);
  CHECK(layers == static_cast<int>(L));
  CHECK(elements == static_cast<size_t>((L + 1) * T * kHidden));
  MESSAGE("dirty-slot prefill: compared " << elements << " elements over " << layers
                                          << " layers; worst RELATIVE deviation "
                                          << worst_rel << " against a band of "
                                          << kRelF32);
  unsetenv("VT_KV_CACHE_F32");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. G-SAFE — NARROWED, NEVER DELETED.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH paged: G-SAFE still refuses a BATCHED step by name") {
  // The surviving clause. A2-B removes it; until then a two-request step must
  // refuse rather than decode the batch as one concatenated causal sequence —
  // fluent output, wrong tokens, no error, which is precisely what a token gate
  // cannot see.
  Fixture fx;
  const std::vector<int32_t> token_ids{1, 2};
  const std::vector<int32_t> positions{0, 0};
  const std::vector<int32_t> logits_indices{0, 1};
  const vllm::v1::CommonAttentionMetadata attn_meta{};
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv(1);
  std::vector<vllm::GdnStateCache> gdn_state(1);
  vt::Queue q = Q();
  const vllm::ModelForwardInput input{.token_ids = token_ids,
                                      .positions = positions,
                                      .attn_meta = attn_meta,
                                      .gdn_meta = gdn_meta,
                                      .attn_kv = attn_kv,
                                      .gdn_state = gdn_state,
                                      .config = fx.cfg,
                                      .queue = q,
                                      .logits_indices = logits_indices,
                                      .num_reqs = 2};
  CHECK_THROWS_WITH_AS(ModelRegistry::Forward(*fx.model, input),
                       doctest::Contains("NemotronHForCausalLM"), std::runtime_error);
  CHECK_THROWS_WITH_AS(ModelRegistry::Forward(*fx.model, input),
                       doctest::Contains("BATCHED decode is not ported"),
                       std::runtime_error);
  // The message must name where the missing piece is owed, so the next reader
  // is not sent to the weight loader or to the runner's allocation.
  CHECK_THROWS_WITH_AS(ModelRegistry::Forward(*fx.model, input),
                       doctest::Contains("A2-B"), std::runtime_error);
  CHECK_THROWS_WITH_AS(ModelRegistry::Forward(*fx.model, input),
                       doctest::Contains("#810"), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. PORTED UPSTREAM TESTS (AGENTS.md: "port its tests in the same change").
// ═══════════════════════════════════════════════════════════════════════════

// Ported from vllm `tests/v1/worker/test_mamba_utils.py:2136`
// `test_ds_conv_layout_bias_gt_0_byte_equal_to_sd`, a METHOD of
// `class TestPostprocessMambaFusedKernel` (`:410`) @ 5559679229bc.
//
// Upstream's claim: the DS orientation `(dim, state_len)` and the SD
// orientation `(state_len, dim)` are the SAME BYTE COUNT — `_orient_conv_shape`
// (`mamba_utils.py:152-157`) only transposes the pair. Ours is DS
// (nemotron_h_registry.cpp), so this is what makes that a gate rather than a
// comment. The harness adaptation is unavoidable and is stated: upstream
// asserts on a torch tensor's `numel()` after building the two layouts through
// `get_conv_state_layout()`; there is no `VLLM_SSM_CONV_STATE_LAYOUT` here, so
// the twin asserts the product identity on the shapes `MakeNemotronHKVCache`
// actually publishes.
TEST_CASE("NemotronH paged: the DS conv layout is byte-equal to SD (upstream test_mamba_utils.py:2136)") {
  Fixture fx;
  KVCacheConfig kv = fx.model->registration().factory->make_kv_cache(fx.cfg, kBlockSize,
                                                                     kNumBlocks);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  const auto* mamba =
      dynamic_cast<const vllm::v1::MambaSpec*>(kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(mamba != nullptr);
  REQUIRE(mamba->shapes.size() == 2);

  const std::vector<int64_t>& conv = mamba->shapes[0];
  REQUIRE(conv.size() == 2);
  // DS = (dim, state_len).
  CHECK(conv[0] == kConvDim);
  CHECK(conv[1] == kConvKernel - 1);
  // SD would be (state_len, dim). The BYTES are the same product either way —
  // that is upstream's assertion, and it is what makes our orientation a
  // supported upstream mode rather than a divergence.
  const int64_t ds_elems = conv[0] * conv[1];
  const int64_t sd_elems = (kConvKernel - 1) * static_cast<int64_t>(kConvDim);
  CHECK(ds_elems == sd_elems);
  CHECK(ds_elems == static_cast<int64_t>(kConvDim) * (kConvKernel - 1));
  // ...and the assertion is not vacuous: the two orientations are genuinely
  // different SHAPES, so equality of the product is a real claim.
  CHECK(conv[0] != conv[1]);
}

// Ported from vllm `tests/v1/attention/test_mamba_update_block_table.py:75`
// `test_update_block_table_copies_block_idx_to_persistent_buffers` and `:178`
// `test_state_indices_tensor_d_includes_num_speculative_blocks` @ 5559679229bc.
//
// The first says: the per-request mamba state index comes from the BLOCK TABLE,
// not from a slot map. The second's INTENT, at `num_spec == 0`: the decode slot
// vector is ONE COLUMN WIDE and INDEXED, never hardcoded. `num_spec` is 0 for
// this architecture (the MTP head is #517 W5), so the ported assertion is the
// width and the indexing rather than the speculative widening.
TEST_CASE("NemotronH paged: the recurrent state index comes from the block table (upstream test_mamba_update_block_table.py:75)") {
  Fixture fx;
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(fx.cfg, *fx.model, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  const std::vector<int32_t> prompt{1, 7, 3, 9, 2, 14, 5, 11};
  std::vector<NewRequestData> reqs;
  reqs.push_back(MakeNewReq("R0", prompt, {0, 1}, /*state_slot=*/0));
  std::map<std::string, int> sched;
  sched["R0"] = static_cast<int>(prompt.size());
  SchedulerOutput s1 = NewStep(std::move(reqs), std::move(sched));
  CHECK_FALSE(runner.execute_model(s1).has_value());
  (void)runner.sample_tokens(std::nullopt);

  const vllm::v1::GDNAttentionMetadata& gm = runner.last_gdn_meta();
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  // ONE COLUMN WIDE at num_spec == 0, one entry per request. That is upstream's
  // `test_state_indices_tensor_d_includes_num_speculative_blocks` INTENT ported
  // to `num_spec == 0`: the decode slot vector is a vector, indexed, one column
  // wide — not widened for drafts and not hardcoded.
  REQUIRE(gm.non_spec_state_indices_tensor->size() == 1);
  CHECK(gm.num_spec_decodes == 0);
  CHECK(gm.spec_state_indices_num_cols == 0);
  CHECK_FALSE(gm.spec_state_indices_tensor.has_value());

  // The entry comes from the BLOCK TABLE's column 0 (gdn_attn.cpp reads
  // `m.block_table_tensor[r * cols]`), which the runner filled from
  // `remap_gdn_state_slots`. A DEVIATION from upstream that this port makes
  // deliberately and that this case pins: upstream keys the mamba state on the
  // block id itself, while `runner.cpp:993-1005` keys the COMPACT slot on the
  // request identity and writes it into that column — because once a sequence
  // exceeds one mamba block the column collapses to the shared null block id
  // and every long concurrent sequence would map to ONE slot.
  const int64_t assigned = (*gm.non_spec_state_indices_tensor)[0];
  CHECK(assigned >= 0);
  CHECK(assigned < runner.gdn_state_slots());
  const int cols = runner.last_attn_meta().block_table_num_cols;
  REQUIRE(cols >= 1);

  // The prefill leg carries the has_initial_state mask; a fresh request's is 0,
  // which is the input to the zeroing obligation the sibling case gates.
  REQUIRE(gm.prefill_has_initial_state.has_value());
  REQUIRE(gm.prefill_has_initial_state->size() == 1);
  CHECK((*gm.prefill_has_initial_state)[0] == 0);
  REQUIRE(gm.prefill_state_indices.has_value());
  REQUIRE(gm.prefill_state_indices->size() == 1);
  CHECK((*gm.prefill_state_indices)[0] == assigned);

  // One decode step later the same request is a DECODE, so upstream leaves the
  // mask None (a decode always continues a sequence, gdn_attn.py:405) and the
  // state index is unchanged — the sequence keeps its slot for its lifetime.
  SchedulerOutput sd = DecodeStep({"R0"}, {static_cast<int>(prompt.size())}, {1});
  CHECK_FALSE(runner.execute_model(sd).has_value());
  (void)runner.sample_tokens(std::nullopt);
  const vllm::v1::GDNAttentionMetadata& gd = runner.last_gdn_meta();
  CHECK(gd.num_decodes == 1);
  CHECK(gd.num_prefills == 0);
  CHECK(gd.num_decode_tokens == 1);
  CHECK_FALSE(gd.has_initial_state.has_value());
  REQUIRE(gd.non_spec_state_indices_tensor.has_value());
  REQUIRE(gd.non_spec_state_indices_tensor->size() == 1);
  CHECK((*gd.non_spec_state_indices_tensor)[0] == assigned);
}
