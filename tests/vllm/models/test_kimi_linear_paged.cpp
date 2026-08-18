// ROW 7 / kimi-linear.md §20.3 — the shared-paged-runner fold gates (CPU).
//
// Kimi-Linear folded onto the SHARED paged runner (`ModelRegistry::Forward` →
// `KimiLinearModel::ForwardPaged`): the KDA conv+recurrent state lives in the
// runner's MambaSpec `gdn_state` group and the NoPE-MLA latent-KV in the paged
// MLA `attn_kv` group. Cases:
//   (a) B1 KV enablement — a REAL Kimi config.json (no `layer_types`, no
//       explicit `linear_*` keys) loads through LoadHfConfig, the registry
//       resolves + loads the bf16-resident tower, and GPUModelRunner ALLOCATES
//       the two het-KV groups from MakeKimiLinearKVCache without aborting
//       (pre-fold: VT_CHECK at runner.cpp `expected_conv_shape` fails on
//       {0,0},{0,0,0}).
//   (b) FOLD IDENTITY — the paged-runner greedy decode (prefill once + N
//       single-token decode steps through execute_model/sample_tokens) emits
//       the SAME tokens as the CLI-incremental reference
//       (ForwardPrefillIncremental / ForwardDecodeStepIncremental) over the
//       same resident weights. Run under VT_KV_CACHE_F32=1 so every paged
//       cache (MLA latent rows, kpe, conv taps) carries the f32 values the CLI
//       reference carries — any token difference is a state-carry/paging bug,
//       not a rounding one. This is the CPU half of the §20.3 Gate A; the GB10
//       real-checkpoint battery is the binding on-box gate.
//   (c) SLOT ISOLATION — a 2-request batched decode produces, per request, the
//       same tokens as that request's own single-request run (the compact GDN
//       state slots + per-request MLA block tables do not cross-talk).
//
// The synthetic 2-layer checkpoint (layer0 KDA+dense, layer1 NoPE-MLA+MoE)
// mirrors tests/vllm/models/test_kimi_linear_forward.cpp's builder.
#include <doctest/doctest.h>

#include <algorithm>
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
#include "vllm/model_executor/models/kimi_linear.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::ForwardLogits;
using vllm::HfConfig;
using vllm::KimiDecodeCache;
using vllm::KimiLinearModel;
using vllm::KimiLinearWeights;
using vllm::LoadKimiLinearResidentBf16Weights;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::SafetensorsFile;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::MambaSpec;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;
using vt::DType;

namespace {

// ─── synthetic 2-layer checkpoint (mirror test_kimi_linear_forward.cpp) ───────
constexpr int H = 32, NAH = 4, V = 8, DENSE_I = 64, MOE_I = 16, E = 2;
constexpr int KV_LORA = 16, QK_NOPE = 8, QK_ROPE = 4, V_HEAD = 8;
constexpr int KDA_NH = 4, KDA_HD = 8, CONV = 4;
constexpr int KDA_PROJ = KDA_NH * KDA_HD;
constexpr int MLA_QK = QK_NOPE + QK_ROPE;

struct Fx { std::string name, dtype; std::vector<int64_t> shape; std::string bytes; };
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
std::string Bf16Bytes(size_t n, int seed) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
    const float f = (u - 0.5f) * 0.25f;
    uint16_t bf = static_cast<uint16_t>(vt::F32ToBF16(f));
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}
std::string F32Bytes(size_t n, int seed) {
  std::string s(n * 4, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2246822519u + 1u;
  for (size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
    const float f = (u - 0.5f) * 0.25f;
    std::memcpy(&s[i * 4], &f, 4);
  }
  return s;
}
Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed) {
  return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed)};
}
Fx F32(const std::string& n, std::vector<int64_t> sh, int seed) {
  return {n, "F32", sh, F32Bytes(static_cast<size_t>(NumEl(sh)), seed)};
}
std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape},
                   {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}
class TempFile {
 public:
  explicit TempFile(const std::string& bytes, const char* ext = ".safetensors") {
    // Unique PER PROCESS (pid + counter): two concurrent ctest processes with a
    // fixed /tmp/kimi_paged_<counter> name would overwrite each other's fixture
    // mid-read (the cross-process collision hazard).
    static int c = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("kimi_paged_" + std::to_string(::getpid()) + "_" +
              std::to_string(c++) + ext)).string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }
 private:
  std::string path_;
};

std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("model.embed_tokens.weight", {V, H}, s++));
  v.push_back(Bf16("model.norm.weight", {H}, s++));
  v.push_back(Bf16("lm_head.weight", {V, H}, s++));
  const std::string a0 = "model.layers.0.";
  v.push_back(Bf16(a0 + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(a0 + "post_attention_layernorm.weight", {H}, s++));
  const std::string k = a0 + "self_attn.";
  v.push_back(Bf16(k + "q_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "k_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "v_proj.weight", {KDA_PROJ, H}, s++));
  v.push_back(Bf16(k + "f_a_proj.weight", {KDA_HD, H}, s++));
  v.push_back(Bf16(k + "f_b_proj.weight", {KDA_PROJ, KDA_HD}, s++));
  v.push_back(Bf16(k + "b_proj.weight", {KDA_NH, H}, s++));
  v.push_back(Bf16(k + "g_a_proj.weight", {KDA_HD, H}, s++));
  v.push_back(Bf16(k + "g_b_proj.weight", {KDA_PROJ, KDA_HD}, s++));
  v.push_back(Bf16(k + "o_proj.weight", {H, KDA_PROJ}, s++));
  v.push_back(F32(k + "q_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "k_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "v_conv1d.weight", {KDA_PROJ, 1, CONV}, s++));
  v.push_back(F32(k + "dt_bias", {KDA_PROJ}, s++));
  v.push_back(F32(k + "A_log", {KDA_NH}, s++));
  v.push_back(Bf16(k + "o_norm.weight", {KDA_HD}, s++));
  v.push_back(Bf16(a0 + "mlp.gate_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16(a0 + "mlp.up_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16(a0 + "mlp.down_proj.weight", {H, DENSE_I}, s++));
  const std::string a1 = "model.layers.1.";
  v.push_back(Bf16(a1 + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(a1 + "post_attention_layernorm.weight", {H}, s++));
  const std::string m = a1 + "self_attn.";
  v.push_back(Bf16(m + "q_proj.weight", {NAH * MLA_QK, H}, s++));
  v.push_back(Bf16(m + "kv_a_proj_with_mqa.weight", {KV_LORA + QK_ROPE, H}, s++));
  v.push_back(Bf16(m + "kv_a_layernorm.weight", {KV_LORA}, s++));
  v.push_back(Bf16(m + "kv_b_proj.weight", {NAH * (QK_NOPE + V_HEAD), KV_LORA}, s++));
  v.push_back(Bf16(m + "o_proj.weight", {H, NAH * V_HEAD}, s++));
  const std::string mo = a1 + "block_sparse_moe.";
  v.push_back(Bf16(mo + "gate.weight", {E, H}, s++));
  v.push_back(F32(mo + "gate.e_score_correction_bias", {E}, s++));
  v.push_back(Bf16(mo + "shared_experts.gate_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16(mo + "shared_experts.up_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16(mo + "shared_experts.down_proj.weight", {H, MOE_I}, s++));
  for (int e = 0; e < E; ++e) {
    const std::string ep = mo + "experts." + std::to_string(e) + ".";
    v.push_back(Bf16(ep + "w1.weight", {MOE_I, H}, s++));
    v.push_back(Bf16(ep + "w2.weight", {H, MOE_I}, s++));
    v.push_back(Bf16(ep + "w3.weight", {MOE_I, H}, s++));
  }
  return v;
}

// The REAL Kimi config shape: NO layer_types, NO explicit linear_* keys — the
// KDA split lives only in linear_attn_config (B1's synthesis is what makes the
// runner allocation possible).
std::string TinyConfigJson() {
  nlohmann::json j = {
      {"model_type", "kimi_linear"},
      {"architectures", {"KimiLinearForCausalLM"}},
      {"hidden_size", H},
      {"num_hidden_layers", 2},
      {"vocab_size", V},
      {"num_attention_heads", NAH},
      {"num_key_value_heads", NAH},
      {"intermediate_size", DENSE_I},
      {"rms_norm_eps", 1e-5},
      {"tie_word_embeddings", false},
      {"num_experts", E},
      {"num_experts_per_token", 1},
      {"num_shared_experts", 1},
      {"moe_intermediate_size", MOE_I},
      {"first_k_dense_replace", 1},
      {"moe_layer_freq", 1},
      {"moe_router_activation_func", "sigmoid"},
      {"routed_scaling_factor", 2.446},
      {"kv_lora_rank", KV_LORA},
      {"qk_nope_head_dim", QK_NOPE},
      {"qk_rope_head_dim", QK_ROPE},
      {"v_head_dim", V_HEAD},
      {"mla_use_nope", true},
      {"max_position_embeddings", 64},
      {"linear_attn_config",
       {{"kda_layers", nlohmann::json::array({1})},
        {"full_attn_layers", nlohmann::json::array({2})},
        {"num_heads", KDA_NH},
        {"head_dim", KDA_HD},
        {"short_conv_kernel_size", CONV}}},
  };
  return j.dump();
}

// The engine-level attention backends (FLASH_ATTN + ROCM_ATTN) enforce
// block_size % 16 == 0 in get_kv_cache_shape; the runner now validates at
// init, so the fixture uses a real block size (vLLM gate models use 16).
constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 8;
constexpr int kMaxModelLen = 32;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;
  sp.PostInit();
  return sp;
}

NewRequestData MakeNewReq(const std::string& id, std::vector<int32_t> prompt,
                          std::vector<int> fa_blocks, int gdn_block) {
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = prompt;
  nr.sampling_params = Greedy();
  // Kimi group order (MakeKimiLinearKVCache): group 0 = MLA latent, group 1 = KDA.
  nr.block_ids = {std::move(fa_blocks), std::vector<int>{gdn_block}};
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

// Greedy tokens from the paged runner for one request set. Returns tokens per
// request id, in `prompts` order.
std::vector<std::vector<int32_t>> RunnerGreedy(
    const HfConfig& cfg, vllm::LoadedModel& model,
    const std::vector<std::vector<int32_t>>& prompts, int steps) {
  const vllm::ModelRegistration& reg = model.registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(cfg, kBlockSize, kNumBlocks);
  GPUModelRunner runner(cfg, model, kv, Q(), /*max_num_reqs=*/4, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  // Admit all prompts in one prefill step (disjoint FA blocks + GDN slots).
  std::vector<NewRequestData> reqs;
  std::map<std::string, int> sched;
  for (size_t i = 0; i < prompts.size(); ++i) {
    const std::string id = "R" + std::to_string(i);
    const int b0 = static_cast<int>(i) * 2;
    reqs.push_back(MakeNewReq(id, prompts[i], {b0, b0 + 1}, static_cast<int>(i)));
    sched[id] = static_cast<int>(prompts[i].size());
  }
  SchedulerOutput s1 = NewStep(std::move(reqs), std::move(sched));
  CHECK_FALSE(runner.execute_model(s1).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == prompts.size());

  std::vector<std::vector<int32_t>> out(prompts.size());
  std::vector<int> computed(prompts.size()), outputs(prompts.size(), 1);
  std::vector<std::string> ids;
  for (size_t i = 0; i < prompts.size(); ++i) {
    // sample order == admission order in this single-batch harness.
    REQUIRE(m1.sampled_token_ids[i].size() == 1);
    out[i].push_back(m1.sampled_token_ids[i][0]);
    computed[static_cast<int>(i)] = static_cast<int>(prompts[i].size());
    ids.push_back("R" + std::to_string(i));
  }
  for (int s = 1; s < steps; ++s) {
    SchedulerOutput sd = DecodeStep(ids, computed, outputs);
    CHECK_FALSE(runner.execute_model(sd).has_value());
    vllm::v1::ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids.size() == prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      REQUIRE(md.sampled_token_ids[i].size() == 1);
      out[i].push_back(md.sampled_token_ids[i][0]);
      computed[i] += 1;
      outputs[i] += 1;
    }
  }
  return out;
}

// The CLI-incremental reference leg (the §19 18.9 tok/s vehicle): prefill once,
// then N carried decode steps, greedy host argmax.
std::vector<int32_t> CliIncrementalGreedy(const KimiLinearWeights& w,
                                          const std::vector<int32_t>& prompt,
                                          int steps) {
  vt::Queue q = Q();
  vt::Backend& be = vt::GetBackend(q.device.type);
  const int64_t vocab = w.params.vocab_size;
  auto argmax_row = [&](const ForwardLogits& fl) {
    std::vector<float> row(static_cast<size_t>(vocab));
    be.Copy(q, row.data(), fl.device_tensor.data, row.size() * sizeof(float));
    be.Synchronize(q);
    int best = 0;
    float bv = row[0];
    for (int64_t o = 1; o < vocab; ++o)
      if (row[static_cast<size_t>(o)] > bv) {
        bv = row[static_cast<size_t>(o)];
        best = static_cast<int>(o);
      }
    return best;
  };
  KimiDecodeCache cache;
  std::vector<int32_t> positions(prompt.size());
  for (size_t t = 0; t < prompt.size(); ++t) positions[t] = static_cast<int32_t>(t);
  const std::vector<int32_t> li = {static_cast<int32_t>(prompt.size() - 1)};
  ForwardLogits fl =
      KimiLinearModel::ForwardPrefillIncremental(prompt, positions, w, q, cache, li);
  std::vector<int32_t> gen;
  int best = argmax_row(fl);
  gen.push_back(best);
  for (int s = 1; s < steps; ++s) {
    ForwardLogits dfl = KimiLinearModel::ForwardDecodeStepIncremental(
        best, cache.seq_len, w, q, cache);
    best = argmax_row(dfl);
    gen.push_back(best);
  }
  return gen;
}

struct Fixture {
  std::unique_ptr<TempFile> st;
  std::unique_ptr<TempFile> cfg_json;
  std::vector<SafetensorsFile> shards;
  HfConfig cfg;
  Fixture() {
    st = std::make_unique<TempFile>(BuildSt(BuildTensors()));
    cfg_json = std::make_unique<TempFile>(TinyConfigJson(), ".json");
    shards.push_back(SafetensorsFile::Open(st->path()));
    cfg = vllm::LoadHfConfig(cfg_json->path());
  }
};

}  // namespace

// ─── (a) B1 — the runner allocates Kimi's het-KV groups (no abort) ────────────
TEST_CASE("kimi paged: runner allocates MLA latent + KDA state groups from a real config") {
  Fixture fx;
  // The B1 synthesis populated the runner-facing geometry.
  REQUIRE(fx.cfg.layer_types.size() == 2);
  CHECK(fx.cfg.layer_types[0] == "linear_attention");
  CHECK(fx.cfg.layer_types[1] == "full_attention");
  CHECK(fx.cfg.linear_num_key_heads == KDA_NH);
  CHECK(fx.cfg.linear_conv_kernel_dim == CONV);

  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(fx.cfg);
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(fx.shards));
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  REQUIRE(kv.kv_cache_groups.size() == 2);

  // Pre-fold this constructor ABORTED (runner.cpp MambaSpec shape check against
  // config-derived {0,0},{0,0,0}); with B1 it allocates.
  GPUModelRunner runner(fx.cfg, *model, kv, Q(), /*max_num_reqs=*/4, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == 1);
  // 1 MLA layer page (num_kv_heads==1, 576-analog width) + 1 KDA state group.
  REQUIRE(runner.attn_kv().size() == 1);
  CHECK(runner.attn_kv()[0].num_kv_heads == 1);
  CHECK(runner.attn_kv()[0].head_size == KV_LORA + QK_ROPE);
  REQUIRE(runner.gdn_state().size() == 1);
  CHECK(runner.gdn_state()[0].conv_state.shape[1] == 3 * KDA_PROJ);
  CHECK(runner.gdn_state()[0].conv_state.shape[2] == CONV - 1);
  CHECK(runner.gdn_state()[0].ssm_state.shape[1] == KDA_NH);
  CHECK(runner.gdn_state()[0].ssm_state.shape[2] == KDA_HD);
  CHECK(runner.gdn_state()[0].ssm_state.shape[3] == KDA_HD);
}

// ─── (b) FOLD IDENTITY — paged-runner greedy == CLI-incremental greedy ────────
TEST_CASE("kimi paged: runner decode tokens == CLI incremental reference (f32 caches)") {
  // f32 caches: the paged rows carry the same f32 values the CLI reference
  // carries, so identity is exact-by-construction (any diff = a paging bug).
  // The EXACT MLA arm is the fold-identity vehicle (the production default is
  // the FA2 arm, whose GB10 profile gate lives in §21).
  setenv("VT_KV_CACHE_F32", "1", 1);
  setenv("VT_KIMI_PAGED_MLA_FA2", "0", 1);
  Fixture fx;
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(fx.shards));

  const std::vector<int32_t> prompt = {5, 1, 2, 7, 3};
  const int steps = 8;
  const std::vector<std::vector<int32_t>> paged =
      RunnerGreedy(fx.cfg, *model, {prompt}, steps);

  KimiLinearWeights cli_w =
      LoadKimiLinearResidentBf16Weights(fx.shards, fx.cfg, nullptr);
  const std::vector<int32_t> cli = CliIncrementalGreedy(cli_w, prompt, steps);

  REQUIRE(paged.size() == 1);
  REQUIRE(paged[0].size() == static_cast<size_t>(steps));
  REQUIRE(cli.size() == static_cast<size_t>(steps));
  for (int s = 0; s < steps; ++s) {
    INFO("step ", s, " paged=", paged[0][static_cast<size_t>(s)],
         " cli=", cli[static_cast<size_t>(s)]);
    CHECK(paged[0][static_cast<size_t>(s)] == cli[static_cast<size_t>(s)]);
  }
  unsetenv("VT_KV_CACHE_F32");
  unsetenv("VT_KIMI_PAGED_MLA_FA2");
}

// The production regime (bf16 latent/conv caches — vLLM's own cache dtype): the
// tiny random model's argmax margins survive the cache rounding, so the token
// stream still matches the CLI reference.
TEST_CASE("kimi paged: runner decode tokens == CLI reference (production bf16 caches)") {
  Fixture fx;
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(fx.shards));
  const std::vector<int32_t> prompt = {2, 6, 4, 1};
  const int steps = 6;
  const std::vector<std::vector<int32_t>> paged =
      RunnerGreedy(fx.cfg, *model, {prompt}, steps);
  KimiLinearWeights cli_w =
      LoadKimiLinearResidentBf16Weights(fx.shards, fx.cfg, nullptr);
  const std::vector<int32_t> cli = CliIncrementalGreedy(cli_w, prompt, steps);
  REQUIRE(paged.size() == 1);
  for (int s = 0; s < steps; ++s) {
    INFO("step ", s);
    CHECK(paged[0][static_cast<size_t>(s)] == cli[static_cast<size_t>(s)]);
  }
}

// B3's shared MLA arm must execute, not merely compile behind a default-off
// switch.  The first half runs the real absorbed-MQA/FA2 block over bf16 pages
// and requires its greedy stream to agree with the exact-island fold vehicle.
// The second half deliberately gives that arm an f32 page: its arm-specific
// dtype contract must reject it, so mutating the dispatch back to the exact
// island cannot leave this test green.
TEST_CASE("kimi paged: shared MLA arm runs and preserves the tiny greedy stream") {
  Fixture fx;
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(fx.shards));
  const std::vector<int32_t> prompt = {2, 6, 4, 1};
  const int steps = 4;

  unsetenv("VT_KIMI_PAGED_MLA_FA2");
  unsetenv("VT_KV_CACHE_F32");
  const std::vector<std::vector<int32_t>> exact =
      RunnerGreedy(fx.cfg, *model, {prompt}, steps);

  setenv("VT_KIMI_PAGED_MLA_FA2", "1", 1);
  const std::vector<std::vector<int32_t>> shared =
      RunnerGreedy(fx.cfg, *model, {prompt}, steps);
  CHECK(shared == exact);

  // The f32-page rejection probes ForwardPaged DIRECTLY: RunnerGreedy's internal
  // REQUIREs throw doctest's own failure exception (not a std::runtime_error),
  // which would mask the guard's type.
  setenv("VT_KV_CACHE_F32", "1", 1);
  {
    KimiLinearWeights w =
        LoadKimiLinearResidentBf16Weights(fx.shards, fx.cfg, nullptr);
    vt::Queue q = Q();
    const int64_t head = KV_LORA + QK_ROPE;
    const int64_t conv_dim = 3 * KDA_PROJ;
    std::vector<float> mla_page(static_cast<size_t>(2 * 8 * head), 0.0f);
    std::vector<float> conv_buf(static_cast<size_t>(conv_dim) * (CONV - 1), 0.0f);
    std::vector<float> ssm_buf(static_cast<size_t>(KDA_NH) * KDA_HD * KDA_HD, 0.0f);
    vllm::PagedKvCache kv;
    kv.data = mla_page.data();
    kv.dtype = DType::kF32;
    kv.num_blocks = 2;
    kv.block_size = 8;
    kv.num_kv_heads = 1;
    kv.head_size = head;
    std::vector<vllm::PagedKvCache> attn_kv{kv};
    vllm::GdnStateCache gs;
    gs.conv_state = vt::Tensor::Contiguous(conv_buf.data(), DType::kF32, q.device,
                                           {1, conv_dim, CONV - 1});
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf.data(), DType::kF32, q.device,
                                          {1, KDA_NH, KDA_HD, KDA_HD});
    std::vector<vllm::GdnStateCache> gdn_state{gs};
    const std::vector<int32_t> toks = {2, 6};
    vllm::v1::CommonAttentionMetadata am;
    am.query_start_loc = {0, 2};
    am.query_start_loc_cpu = am.query_start_loc;
    am.seq_lens = {2};
    am.seq_lens_cpu = am.seq_lens;
    am.num_computed_tokens_cpu = {0};
    am.num_reqs = 1;
    am.num_actual_tokens = 2;
    am.max_query_len = 2;
    am.max_seq_len = 2;
    am.block_table_tensor = {0, 1};
    am.block_table_num_cols = 2;
    am.slot_mapping = {0, 1};
    vllm::v1::CommonAttentionMetadata gdn_cam = am;
    gdn_cam.block_table_tensor = {0};
    gdn_cam.block_table_num_cols = 1;
    vllm::v1::GDNAttentionMetadataBuilder builder;
    vllm::v1::GDNAttentionMetadata gm = builder.build(0, gdn_cam);
    const std::vector<int32_t> positions = {0, 1};
    const std::vector<int32_t> li = {1};
    vllm::ModelForwardInput in{
        .token_ids = toks,
        .positions = positions,
        .attn_meta = am,
        .gdn_meta = gm,
        .attn_kv = attn_kv,
        .gdn_state = gdn_state,
        .config = fx.cfg,
        .queue = q,
        .logits_indices = li,
        .num_reqs = 1,
    };
    CHECK_THROWS_WITH_AS((void)KimiLinearModel::ForwardPaged(in, w),
                         doctest::Contains("paged latent cache must be bf16"),
                         std::runtime_error);
  }
  unsetenv("VT_KV_CACHE_F32");
  unsetenv("VT_KIMI_PAGED_MLA_FA2");
}

// ─── (b2) LOGITS-EXACT — ForwardPaged logits == CLI incremental logits ────────
// The runner-level token gates above are necessary but weak on a tiny random
// model (an argmax over V=8 can survive a corrupted state). This case drives
// KimiLinearModel::ForwardPaged DIRECTLY over hand-allocated paged caches with
// the REAL GDNAttentionMetadataBuilder segmentation, and requires the [1,V]
// logits row to be byte-equal to the CLI incremental leg at EVERY step — so a
// dropped state scatter, a stale conv tap, or a mis-indexed latent row all go
// RED (verified by mutation).
TEST_CASE("kimi paged: ForwardPaged logits byte-equal the CLI incremental logits") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  setenv("VT_KIMI_PAGED_MLA_FA2", "0", 1);  // the EXACT arm is the identity vehicle
  Fixture fx;
  KimiLinearWeights w = LoadKimiLinearResidentBf16Weights(fx.shards, fx.cfg, nullptr);
  vt::Queue q = Q();
  vt::Backend& be = vt::GetBackend(q.device.type);

  const std::vector<int32_t> prompt = {5, 1, 2, 7, 3};
  const int P = static_cast<int>(prompt.size());
  const int steps = 8;
  const int64_t vocab = w.params.vocab_size;

  // Paged caches: 1 MLA layer page (f32, 2 blocks x 8) + 1 KDA state group.
  const int64_t bs = 8, nblocks = 2;
  const int64_t head = KV_LORA + QK_ROPE;
  const int64_t conv_dim = 3 * KDA_PROJ;
  std::vector<float> mla_page(static_cast<size_t>(nblocks * bs * head), 0.0f);
  std::vector<float> conv_buf(static_cast<size_t>(conv_dim) * (CONV - 1), 0.0f);
  std::vector<float> ssm_buf(static_cast<size_t>(KDA_NH) * KDA_HD * KDA_HD, 0.0f);
  vllm::PagedKvCache kv;
  kv.data = mla_page.data();
  kv.dtype = DType::kF32;
  kv.num_blocks = nblocks;
  kv.block_size = bs;
  kv.num_kv_heads = 1;
  kv.head_size = head;
  std::vector<vllm::PagedKvCache> attn_kv{kv};
  vllm::GdnStateCache gs;
  gs.conv_state = vt::Tensor::Contiguous(conv_buf.data(), DType::kF32, q.device,
                                         {1, conv_dim, CONV - 1});
  gs.ssm_state = vt::Tensor::Contiguous(ssm_buf.data(), DType::kF32, q.device,
                                        {1, KDA_NH, KDA_HD, KDA_HD});
  std::vector<vllm::GdnStateCache> gdn_state{gs};

  // One step through ForwardPaged: `toks` at positions [pos0, pos0+n), context
  // pos0 tokens already absorbed. Returns the gathered [1,V] logits row.
  auto paged_step = [&](const std::vector<int32_t>& toks, int pos0) {
    const int n = static_cast<int>(toks.size());
    vllm::v1::CommonAttentionMetadata am;
    am.query_start_loc = {0, n};
    am.query_start_loc_cpu = am.query_start_loc;
    am.seq_lens = {pos0 + n};
    am.seq_lens_cpu = am.seq_lens;
    am.num_computed_tokens_cpu = {pos0};
    am.num_reqs = 1;
    am.num_actual_tokens = n;
    am.max_query_len = n;
    am.max_seq_len = pos0 + n;
    am.block_table_tensor = {0, 1};
    am.block_table_num_cols = 2;
    for (int t = 0; t < n; ++t) am.slot_mapping.push_back(pos0 + t);
    vllm::v1::CommonAttentionMetadata gdn_cam = am;
    gdn_cam.block_table_tensor = {0};  // GDN state slot 0
    gdn_cam.block_table_num_cols = 1;
    vllm::v1::GDNAttentionMetadataBuilder builder;
    vllm::v1::GDNAttentionMetadata gm = builder.build(0, gdn_cam);
    std::vector<int32_t> positions(toks.size());
    for (int t = 0; t < n; ++t) positions[static_cast<size_t>(t)] = pos0 + t;
    const std::vector<int32_t> li = {n - 1};
    vllm::ModelForwardInput in{
        .token_ids = toks,
        .positions = positions,
        .attn_meta = am,
        .gdn_meta = gm,
        .attn_kv = attn_kv,
        .gdn_state = gdn_state,
        .config = fx.cfg,
        .queue = q,
        .logits_indices = li,
        .num_reqs = 1,
    };
    ForwardLogits fl = KimiLinearModel::ForwardPaged(in, w);
    std::vector<float> row(static_cast<size_t>(vocab));
    be.Copy(q, row.data(), fl.device_tensor.data, row.size() * sizeof(float));
    be.Synchronize(q);
    return row;
  };

  // The CLI leg, capturing per-step logits.
  KimiDecodeCache cache;
  std::vector<int32_t> positions(prompt.size());
  for (size_t t = 0; t < prompt.size(); ++t) positions[t] = static_cast<int32_t>(t);
  const std::vector<int32_t> li = {P - 1};
  auto row_of = [&](const ForwardLogits& fl) {
    std::vector<float> row(static_cast<size_t>(vocab));
    be.Copy(q, row.data(), fl.device_tensor.data, row.size() * sizeof(float));
    be.Synchronize(q);
    return row;
  };
  auto argmax = [&](const std::vector<float>& row) {
    int best = 0;
    for (int64_t o = 1; o < vocab; ++o)
      if (row[static_cast<size_t>(o)] > row[static_cast<size_t>(best)])
        best = static_cast<int>(o);
    return best;
  };

  std::vector<float> cli_row = row_of(KimiLinearModel::ForwardPrefillIncremental(
      prompt, positions, w, q, cache, li));
  std::vector<float> paged_row = paged_step(prompt, 0);
  CHECK(std::any_of(ssm_buf.begin(), ssm_buf.end(),
                    [](float x) { return x != 0.0f; }));
  int mismatches = 0;
  for (int64_t o = 0; o < vocab; ++o)
    if (cli_row[static_cast<size_t>(o)] != paged_row[static_cast<size_t>(o)])
      ++mismatches;
  CHECK(mismatches == 0);

  int tok = argmax(cli_row);
  CHECK(argmax(paged_row) == tok);
  int pos0 = P;
  for (int s = 1; s < steps; ++s) {
    cli_row = row_of(
        KimiLinearModel::ForwardDecodeStepIncremental(tok, cache.seq_len, w, q, cache));
    const std::vector<float> state_before = ssm_buf;
    paged_row = paged_step({tok}, pos0);
    // A decode step must commit the advanced recurrent state back to the
    // runner-owned slot.  Logit equality alone is too weak on this tiny model:
    // deleting the decode GdnStateScatter used to leave its argmax unchanged.
    CHECK(ssm_buf != state_before);
    mismatches = 0;
    for (int64_t o = 0; o < vocab; ++o)
      if (cli_row[static_cast<size_t>(o)] != paged_row[static_cast<size_t>(o)])
        ++mismatches;
    INFO("decode step ", s);
    CHECK(mismatches == 0);
    const int nt = argmax(cli_row);
    CHECK(argmax(paged_row) == nt);
    tok = nt;
    pos0 += 1;
  }
  unsetenv("VT_KV_CACHE_F32");
  unsetenv("VT_KIMI_PAGED_MLA_FA2");
}

// Two fresh prefills in one batch must populate the two state slots named by
// GDNAttentionMetadata.  This pins the slot mapping directly: token-level
// isolation can survive a mutation that sends every fresh prefill to slot 0 on
// a tiny random model.
TEST_CASE("kimi paged: batched prefill writes each request's distinct KDA state slot") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  setenv("VT_KIMI_PAGED_MLA_FA2", "0", 1);  // f32 pages => the exact arm
  Fixture fx;
  KimiLinearWeights w = LoadKimiLinearResidentBf16Weights(fx.shards, fx.cfg, nullptr);
  vt::Queue q = Q();
  const std::vector<int32_t> pa = {5, 1, 2, 7, 3};
  const std::vector<int32_t> pb = {6, 6, 0, 4};
  std::vector<int32_t> toks = pa;
  toks.insert(toks.end(), pb.begin(), pb.end());
  const int na = static_cast<int>(pa.size());
  const int nb = static_cast<int>(pb.size());
  const int nt = na + nb;
  const int64_t head = KV_LORA + QK_ROPE;
  const int64_t conv_dim = 3 * KDA_PROJ;
  const size_t ssm_row = static_cast<size_t>(KDA_NH) * KDA_HD * KDA_HD;

  std::vector<float> mla_page(static_cast<size_t>(4 * 8 * head), 0.0f);
  std::vector<float> conv_buf(static_cast<size_t>(2 * conv_dim) * (CONV - 1), 0.0f);
  std::vector<float> ssm_buf(2 * ssm_row, 0.0f);
  vllm::PagedKvCache kv;
  kv.data = mla_page.data();
  kv.dtype = DType::kF32;
  kv.num_blocks = 4;
  kv.block_size = 8;
  kv.num_kv_heads = 1;
  kv.head_size = head;
  std::vector<vllm::PagedKvCache> attn_kv{kv};
  vllm::GdnStateCache gs;
  gs.conv_state = vt::Tensor::Contiguous(conv_buf.data(), DType::kF32, q.device,
                                         {2, conv_dim, CONV - 1});
  gs.ssm_state = vt::Tensor::Contiguous(ssm_buf.data(), DType::kF32, q.device,
                                        {2, KDA_NH, KDA_HD, KDA_HD});
  std::vector<vllm::GdnStateCache> gdn_state{gs};

  vllm::v1::CommonAttentionMetadata am;
  am.query_start_loc = {0, na, nt};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {na, nb};
  am.seq_lens_cpu = am.seq_lens;
  am.num_computed_tokens_cpu = {0, 0};
  am.num_reqs = 2;
  am.num_actual_tokens = nt;
  am.max_query_len = std::max(na, nb);
  am.max_seq_len = std::max(na, nb);
  am.block_table_tensor = {0, 1, 2, 3};
  am.block_table_num_cols = 2;
  for (int t = 0; t < na; ++t) am.slot_mapping.push_back(t);
  for (int t = 0; t < nb; ++t) am.slot_mapping.push_back(2 * 8 + t);
  vllm::v1::CommonAttentionMetadata gdn_cam = am;
  gdn_cam.block_table_tensor = {0, 1};
  gdn_cam.block_table_num_cols = 1;
  vllm::v1::GDNAttentionMetadataBuilder builder;
  vllm::v1::GDNAttentionMetadata gm = builder.build(0, gdn_cam);
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  CHECK(*gm.non_spec_state_indices_tensor == std::vector<int32_t>{0, 1});

  std::vector<int32_t> positions(static_cast<size_t>(nt));
  for (int t = 0; t < na; ++t) positions[static_cast<size_t>(t)] = t;
  for (int t = 0; t < nb; ++t) positions[static_cast<size_t>(na + t)] = t;
  const std::vector<int32_t> li = {na - 1, nt - 1};
  vllm::ModelForwardInput in{
      .token_ids = toks,
      .positions = positions,
      .attn_meta = am,
      .gdn_meta = gm,
      .attn_kv = attn_kv,
      .gdn_state = gdn_state,
      .config = fx.cfg,
      .queue = q,
      .logits_indices = li,
      .num_reqs = 2,
  };
  (void)KimiLinearModel::ForwardPaged(in, w);

  const auto slot_nonzero = [&](size_t slot) {
    const auto first = ssm_buf.begin() + static_cast<std::ptrdiff_t>(slot * ssm_row);
    return std::any_of(first, first + static_cast<std::ptrdiff_t>(ssm_row),
                       [](float x) { return x != 0.0f; });
  };
  CHECK(slot_nonzero(0));
  CHECK(slot_nonzero(1));
  CHECK_FALSE(std::equal(ssm_buf.begin(), ssm_buf.begin() + ssm_row,
                         ssm_buf.begin() + ssm_row));
  unsetenv("VT_KV_CACHE_F32");
  unsetenv("VT_KIMI_PAGED_MLA_FA2");
}

// ─── (b3) ASYNC DEVICE-MIRROR input ids (ENG-ASYNC-SCHED W4) ──────────────────
// On the GB10 served path the async device mirror patches each decode row's
// sampled token into the RUNNER's device input-id buffer and deliberately
// leaves the host `token_ids` STALE — a model that ignores
// `ModelForwardInput::device_token_ids` embeds garbage (the 9/128 divergence
// this case was cut from). ForwardPaged must embed from the device pointer
// when it is non-null. On CPU a host pointer IS device-addressable, so the
// contract is directly testable: hand the RIGHT ids only through
// device_token_ids (host ids deliberately wrong) and require the logits to
// match a normal run with the right host ids.
TEST_CASE("kimi paged: ForwardPaged embeds device_token_ids over stale host ids") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  setenv("VT_KIMI_PAGED_MLA_FA2", "0", 1);
  Fixture fx;
  KimiLinearWeights w = LoadKimiLinearResidentBf16Weights(fx.shards, fx.cfg, nullptr);
  vt::Queue q = Q();
  vt::Backend& be = vt::GetBackend(q.device.type);
  const int64_t vocab = w.params.vocab_size;
  const int64_t conv_dim = 3 * KDA_PROJ;
  const int64_t head = KV_LORA + QK_ROPE;

  // Two independent single-request paged contexts (fresh caches each).
  auto run_prefill = [&](const std::vector<int32_t>& host_ids,
                         const int32_t* device_ids) {
    std::vector<float> mla_page(static_cast<size_t>(2 * 8 * head), 0.0f);
    std::vector<float> conv_buf(static_cast<size_t>(conv_dim) * (CONV - 1), 0.0f);
    std::vector<float> ssm_buf(static_cast<size_t>(KDA_NH) * KDA_HD * KDA_HD, 0.0f);
    vllm::PagedKvCache kv;
    kv.data = mla_page.data();
    kv.dtype = DType::kF32;
    kv.num_blocks = 2;
    kv.block_size = 8;
    kv.num_kv_heads = 1;
    kv.head_size = head;
    std::vector<vllm::PagedKvCache> attn_kv{kv};
    vllm::GdnStateCache gs;
    gs.conv_state = vt::Tensor::Contiguous(conv_buf.data(), DType::kF32, q.device,
                                           {1, conv_dim, CONV - 1});
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf.data(), DType::kF32, q.device,
                                          {1, KDA_NH, KDA_HD, KDA_HD});
    std::vector<vllm::GdnStateCache> gdn_state{gs};
    const int n = static_cast<int>(host_ids.size());
    vllm::v1::CommonAttentionMetadata am;
    am.query_start_loc = {0, n};
    am.query_start_loc_cpu = am.query_start_loc;
    am.seq_lens = {n};
    am.seq_lens_cpu = am.seq_lens;
    am.num_computed_tokens_cpu = {0};
    am.num_reqs = 1;
    am.num_actual_tokens = n;
    am.max_query_len = n;
    am.max_seq_len = n;
    am.block_table_tensor = {0, 1};
    am.block_table_num_cols = 2;
    for (int t = 0; t < n; ++t) am.slot_mapping.push_back(t);
    vllm::v1::CommonAttentionMetadata gdn_cam = am;
    gdn_cam.block_table_tensor = {0};
    gdn_cam.block_table_num_cols = 1;
    vllm::v1::GDNAttentionMetadataBuilder builder;
    vllm::v1::GDNAttentionMetadata gm = builder.build(0, gdn_cam);
    std::vector<int32_t> positions(host_ids.size());
    for (int t = 0; t < n; ++t) positions[static_cast<size_t>(t)] = t;
    const std::vector<int32_t> li = {n - 1};
    vllm::ModelForwardInput in{
        .token_ids = host_ids,
        .positions = positions,
        .attn_meta = am,
        .gdn_meta = gm,
        .attn_kv = attn_kv,
        .gdn_state = gdn_state,
        .config = fx.cfg,
        .queue = q,
        .logits_indices = li,
        .num_reqs = 1,
    };
    in.device_token_ids = device_ids;
    ForwardLogits fl = KimiLinearModel::ForwardPaged(in, w);
    std::vector<float> row(static_cast<size_t>(vocab));
    be.Copy(q, row.data(), fl.device_tensor.data, row.size() * sizeof(float));
    be.Synchronize(q);
    return row;
  };

  const std::vector<int32_t> right = {5, 1, 2, 7, 3};
  const std::vector<int32_t> stale = {0, 0, 0, 0, 0};  // deliberately wrong host ids
  const std::vector<float> want = run_prefill(right, nullptr);
  const std::vector<float> via_device = run_prefill(stale, right.data());
  int mismatches = 0;
  for (int64_t o = 0; o < vocab; ++o)
    if (want[static_cast<size_t>(o)] != via_device[static_cast<size_t>(o)])
      ++mismatches;
  CHECK(mismatches == 0);
  unsetenv("VT_KV_CACHE_F32");
  unsetenv("VT_KIMI_PAGED_MLA_FA2");
}

// ─── (c) SLOT ISOLATION — batched decode == per-request single runs ───────────
TEST_CASE("kimi paged: 2-request batched decode matches each single-request run") {
  setenv("VT_KV_CACHE_F32", "1", 1);
  setenv("VT_KIMI_PAGED_MLA_FA2", "0", 1);
  Fixture fx;
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(fx.shards));
  const std::vector<int32_t> pa = {5, 1, 2, 7, 3};
  const std::vector<int32_t> pb = {6, 6, 0, 4};
  const int steps = 6;

  const std::vector<std::vector<int32_t>> batched =
      RunnerGreedy(fx.cfg, *model, {pa, pb}, steps);
  const std::vector<std::vector<int32_t>> only_a =
      RunnerGreedy(fx.cfg, *model, {pa}, steps);
  const std::vector<std::vector<int32_t>> only_b =
      RunnerGreedy(fx.cfg, *model, {pb}, steps);

  REQUIRE(batched.size() == 2);
  for (int s = 0; s < steps; ++s) {
    INFO("step ", s);
    CHECK(batched[0][static_cast<size_t>(s)] == only_a[0][static_cast<size_t>(s)]);
    CHECK(batched[1][static_cast<size_t>(s)] == only_b[0][static_cast<size_t>(s)]);
  }
  unsetenv("VT_KV_CACHE_F32");
  unsetenv("VT_KIMI_PAGED_MLA_FA2");
}
