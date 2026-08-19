// THE ASYNCHRONOUS DEVICE-IDENTIFIER GATE for the MoE registrations, entered
// through `ModelRegistry::Forward` (#1305; row `ENG-CUDAGRAPH-BREAK`, spec
// `.agents/specs/eng-cudagraph-break.md`).
//
// WHAT IT MEASURES, and why it enters at the registry rather than at the driver.
// `qwen3_moe_registry.cpp`, `deepseek_v2_registry.cpp` and
// `glm4_moe_lite_registry.cpp` each admit a pure-decode step to a decode-graph
// driver. Before #1305 none of the three looked at
// `ModelForwardInput::device_token_ids` at all, and neither did either model's
// eager arms. On the asynchronous serving path the runner's combine splices each
// decode row's sampled token into the DEVICE identifiers on the main queue and
// leaves the host `token_ids` deliberately stale for decode rows
// (`src/vllm/v1/worker/gpu/runner.cpp`, the mirror arm, which is the default), so
// those models generated from the previous step's identifiers.
//
// The defect is SILENTLY WRONG TOKENS and not a fault, so the gate has to assert
// the identifiers themselves. It does that the way
// `tests/vllm/models/test_kimi_linear_paged.cpp` does for the same contract:
// hand the RIGHT identifiers ONLY through `device_token_ids`, make the host
// vector deliberately wrong, and require the logits to equal a run that had the
// right host identifiers and no mirror. On CPU a host pointer is
// device-addressable, so the contract is directly testable here.
//
// THREE RUNS, because two of them cannot separate the cases:
//
//   A  right host ids, no mirror          -> the reference
//   B  WRONG host ids, no mirror          -> must DIFFER from A
//   C  WRONG host ids, mirror carries A's -> must EQUAL A
//
// B is the control. Without it a model that ignored its identifiers entirely
// would pass C, and so would a gate whose two runs happened to share a buffer.
//
// AND THE SEAM IS ASSERTED, not inferred. `vt::StepInputStats::device_refreshes`
// moves only inside `vt::PersistentStepInput::RefreshFromDevice`. A driver that
// hand-rolled the same copy would produce IDENTICAL logits and leave that counter
// at zero, which is exactly the fifth private copy #1305 exists to stop; and a
// step that re-read the mirror and one that uploaded a stale host vector leave
// the same bytes-shaped destination, so no token gate can separate them either.
//
// WHAT THIS HARNESS CANNOT SEE, named rather than claimed away. A CPU "replay"
// recomputes nothing (`decode_graph_seam_harness.h`), so only the COLD step and
// the CAPTURE step below actually run the forward; the two replay steps return
// the slot's persistent logits unchanged. That is why the comparison is per step
// over the first two steps and why the counters are asserted over all of them.
// The depth-2 four-concurrent battery on a real device is a different gate, it
// needs a GPU and a checkpoint, and the spec's `## Owed` records it as owed.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "decode_graph_seam_harness.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/breakable_graph.h"
#include "vt/persistent_step_input.h"

namespace {

using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::PagedKvCache;
using vllm::SafetensorsFile;
using vllm::v1::CommonAttentionMetadata;
using vllm_test::StaticGraphCpu;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// ─── a synthetic safetensors checkpoint (the kimi paged fixture's shape) ─────
struct Fx {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i)
    s[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}
int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}
std::string Bf16Bytes(size_t n, int seed, float scale) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
    const uint16_t bf = vt::F32ToBF16((u - 0.5f) * 2.0f * scale);
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}
Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.08f) {
  return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed, scale)};
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
    static int c = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("moe_devids_" + std::to_string(::getpid()) + "_" +
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

// The same tiny geometry `test_qwen3_moe_decode_graph_seam.cpp` and
// `test_qwen3_moe_forward.cpp` already gate, so this file runs the SAME
// arithmetic those do.
constexpr int64_t kH = 64, kL = 2, kHq = 4, kHkv = 2, kDh = 16, kV = 100;
constexpr int64_t kE = 4, kTopK = 2, kI = 32;

std::string ConfigJson() {
  nlohmann::json j;
  j["architectures"] = std::vector<std::string>{"Qwen3MoeForCausalLM"};
  j["model_type"] = "qwen3_moe";
  j["hidden_size"] = kH;
  j["num_hidden_layers"] = kL;
  j["num_attention_heads"] = kHq;
  j["num_key_value_heads"] = kHkv;
  j["head_dim"] = kDh;
  j["intermediate_size"] = kI;
  j["moe_intermediate_size"] = kI;
  j["shared_expert_intermediate_size"] = 0;
  j["num_experts"] = kE;
  j["num_experts_per_tok"] = kTopK;
  j["vocab_size"] = kV;
  j["max_position_embeddings"] = 256;
  j["rms_norm_eps"] = 1e-6;
  j["rope_theta"] = 10000000.0;
  j["tie_word_embeddings"] = false;
  j["attention_bias"] = false;
  j["torch_dtype"] = "bfloat16";
  return j.dump();
}

std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("model.embed_tokens.weight", {kV, kH}, s++));
  v.push_back(Bf16("model.norm.weight", {kH}, s++, 0.5f));
  v.push_back(Bf16("lm_head.weight", {kV, kH}, s++));
  for (int64_t l = 0; l < kL; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    const std::string sa = b + "self_attn.";
    const std::string mlp = b + "mlp.";
    v.push_back(Bf16(b + "input_layernorm.weight", {kH}, s++, 0.5f));
    v.push_back(Bf16(b + "post_attention_layernorm.weight", {kH}, s++, 0.5f));
    v.push_back(Bf16(sa + "q_proj.weight", {kHq * kDh, kH}, s++));
    v.push_back(Bf16(sa + "k_proj.weight", {kHkv * kDh, kH}, s++));
    v.push_back(Bf16(sa + "v_proj.weight", {kHkv * kDh, kH}, s++));
    v.push_back(Bf16(sa + "o_proj.weight", {kH, kHq * kDh}, s++));
    v.push_back(Bf16(sa + "q_norm.weight", {kDh}, s++, 0.5f));
    v.push_back(Bf16(sa + "k_norm.weight", {kDh}, s++, 0.5f));
    v.push_back(Bf16(mlp + "gate.weight", {kE, kH}, s++));
    for (int64_t e = 0; e < kE; ++e) {
      const std::string ex = mlp + "experts." + std::to_string(e) + ".";
      v.push_back(Bf16(ex + "gate_proj.weight", {kI, kH}, s++));
      v.push_back(Bf16(ex + "up_proj.weight", {kI, kH}, s++));
      v.push_back(Bf16(ex + "down_proj.weight", {kH, kI}, s++));
    }
  }
  return v;
}

struct Fixture {
  std::unique_ptr<TempFile> st;
  std::unique_ptr<TempFile> cfg_json;
  std::vector<SafetensorsFile> shards;
  HfConfig cfg;
  Fixture(const std::string& config_json, const std::vector<Fx>& tensors) {
    st = std::make_unique<TempFile>(BuildSt(tensors));
    cfg_json = std::make_unique<TempFile>(config_json, ".json");
    shards.push_back(SafetensorsFile::Open(st->path()));
    cfg = vllm::LoadHfConfig(cfg_json->path());
  }
};

// One two-request pure-decode step, both requests at the same position, in their
// own KV block. This is the shape `runner.cpp` builds for a condensed-dense
// decode batch, which is the only shape the mirror ever patches.
struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(int64_t num_blocks, int64_t block_size) {
    for (int64_t l = 0; l < kL; ++l)
      buf.emplace_back(
          static_cast<size_t>(num_blocks * 2 * block_size * kHkv * kDh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = kHkv;
      kv.head_size = kDh;
      attn_kv.push_back(kv);
    }
  }
};

// ─── the DeepSeek-V2 half ────────────────────────────────────────────────────
//
// `DeepseekV2DecodeGraph` is reached by TWO registrations,
// `deepseek_v2_registry.cpp` and `glm4_moe_lite_registry.cpp`, and it got the
// same change as `Qwen3MoeDecodeGraph`. It owes its own gate for the reason
// `decode_graph_seam_harness.h` states once: a driver that kept the old
// behaviour produces identical everything except the identifiers it embedded,
// and nothing but this comparison separates the two.
//
// The geometry is `tests/vllm/models/test_deepseek_v2_decode_graph_seam.cpp`'s
// CPU one: MLA with `q_lora_rank: null` (the V2-Lite direct-q branch), four
// routed experts top-2, NO shared expert and NO dense prefix, so every layer is
// a MoE layer.
constexpr int64_t kDsQkNope = 16, kDsQkRope = 8, kDsVHead = 16, kDsKvLora = 24;
constexpr int64_t kDsHeads = 4, kDsE = 4, kDsMoeI = 16;

std::string DsConfigJson() {
  nlohmann::json j;
  j["architectures"] = std::vector<std::string>{"DeepseekV2ForCausalLM"};
  j["model_type"] = "deepseek_v2";
  j["hidden_size"] = kH;
  j["num_hidden_layers"] = kL;
  j["num_attention_heads"] = kDsHeads;
  j["num_key_value_heads"] = kDsHeads;
  j["vocab_size"] = kV;
  j["intermediate_size"] = 32;
  j["moe_intermediate_size"] = kDsMoeI;
  j["n_routed_experts"] = kDsE;
  j["num_experts_per_tok"] = 2;
  j["n_group"] = 1;
  j["topk_group"] = 1;
  j["norm_topk_prob"] = false;
  j["scoring_func"] = "softmax";
  j["topk_method"] = "greedy";
  j["routed_scaling_factor"] = 1.0;
  j["moe_layer_freq"] = 1;
  j["q_lora_rank"] = nullptr;
  j["rms_norm_eps"] = 1e-6;
  j["rope_theta"] = 10000;
  j["max_position_embeddings"] = 128;
  j["tie_word_embeddings"] = false;
  j["torch_dtype"] = "bfloat16";
  j["rope_scaling"] = {{"type", "yarn"},
                       {"factor", 4},
                       {"beta_fast", 32},
                       {"beta_slow", 1},
                       {"mscale", 0.707},
                       {"mscale_all_dim", 0.707},
                       {"original_max_position_embeddings", 32}};
  j["qk_nope_head_dim"] = kDsQkNope;
  j["qk_rope_head_dim"] = kDsQkRope;
  j["v_head_dim"] = kDsVHead;
  j["kv_lora_rank"] = kDsKvLora;
  j["first_k_dense_replace"] = 0;
  j["n_shared_experts"] = 0;
  return j.dump();
}

std::vector<Fx> DsBuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  const int64_t Dqk = kDsQkNope + kDsQkRope;
  v.push_back(Bf16("model.embed_tokens.weight", {kV, kH}, s++));
  v.push_back(Bf16("model.norm.weight", {kH}, s++, 0.5f));
  v.push_back(Bf16("lm_head.weight", {kV, kH}, s++));
  for (int64_t l = 0; l < kL; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    const std::string sa = b + "self_attn.";
    const std::string mlp = b + "mlp.";
    v.push_back(Bf16(b + "input_layernorm.weight", {kH}, s++, 0.5f));
    v.push_back(Bf16(b + "post_attention_layernorm.weight", {kH}, s++, 0.5f));
    v.push_back(Bf16(sa + "q_proj.weight", {kDsHeads * Dqk, kH}, s++));
    v.push_back(
        Bf16(sa + "kv_a_proj_with_mqa.weight", {kDsKvLora + kDsQkRope, kH}, s++));
    v.push_back(Bf16(sa + "kv_a_layernorm.weight", {kDsKvLora}, s++, 0.5f));
    v.push_back(Bf16(sa + "kv_b_proj.weight",
                     {kDsHeads * (kDsQkNope + kDsVHead), kDsKvLora}, s++));
    v.push_back(Bf16(sa + "o_proj.weight", {kH, kDsHeads * kDsVHead}, s++));
    v.push_back(Bf16(mlp + "gate.weight", {kDsE, kH}, s++));
    for (int64_t e = 0; e < kDsE; ++e) {
      const std::string ex = mlp + "experts." + std::to_string(e) + ".";
      v.push_back(Bf16(ex + "gate_proj.weight", {kDsMoeI, kH}, s++));
      v.push_back(Bf16(ex + "up_proj.weight", {kDsMoeI, kH}, s++));
      v.push_back(Bf16(ex + "down_proj.weight", {kH, kDsMoeI}, s++));
    }
  }
  return v;
}

// One MLA cache per layer: [num_blocks, block_size, kv_lora_rank + qk_rope],
// num_kv_heads == 1, NO separate V (MLAAttentionSpec).
struct DsCachePool {
  std::vector<std::vector<uint16_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  DsCachePool(int64_t num_blocks, int64_t block_size) {
    const int64_t head_size = kDsKvLora + kDsQkRope;
    for (int64_t l = 0; l < kL; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * block_size * head_size), 0);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = 1;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
  }
};

constexpr int64_t kBlock = 8;

CommonAttentionMetadata DecodeMeta(int32_t pos) {
  CommonAttentionMetadata am;
  am.num_reqs = 2;
  am.num_actual_tokens = 2;
  am.query_start_loc = {0, 1, 2};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {pos + 1, pos + 1};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 1;
  am.max_seq_len = pos + 1;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0, 1};  // one block each
  am.slot_mapping = {pos, static_cast<int32_t>(kBlock) + pos};
  am.causal = true;
  return am;
}

// The four decode steps, as the identifiers the model SHOULD see.
const std::vector<std::vector<int32_t>>& TrueIds() {
  static const std::vector<std::vector<int32_t>> v = {
      {11, 42}, {12, 7}, {13, 65}, {14, 3}};
  return v;
}

// Drive `steps` pure-decode steps through ModelRegistry::Forward and return the
// downloaded [2, vocab] logits of each step. `mirror` selects run C: the host
// vector is replaced by zeros and the true identifiers travel only through
// `ModelForwardInput::device_token_ids`.
template <class Pool>
std::vector<std::vector<float>> Run(const Fixture& fx, bool stale_host, bool mirror,
                                    int steps) {
  vt::Queue q = Q();
  vt::Backend& be = vt::GetBackend(q.device.type);
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(fx.shards));
  Pool pool(/*num_blocks=*/2, kBlock);
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::GdnStateCache> gdn_state;
  const std::vector<int32_t> no_gather;

  std::vector<std::vector<float>> out;
  for (int t = 0; t < steps; ++t) {
    const std::vector<int32_t>& truth = TrueIds()[static_cast<size_t>(t)];
    const std::vector<int32_t> stale(truth.size(), 0);
    const std::vector<int32_t>& host = stale_host ? stale : truth;
    const std::vector<int32_t> positions = {t, t};
    const CommonAttentionMetadata am = DecodeMeta(t);
    vllm::ModelForwardInput in{host,      positions, am,      gdn_meta,
                               pool.attn_kv, gdn_state, fx.cfg, q,
                               no_gather};
    in.num_reqs = 2;
    in.pure_decode = true;
    in.gdn_state_slots = 8;
    in.uniform_query_len = 1;
    // On CPU a host pointer IS device-addressable, which is what makes the
    // mirror's contract directly testable without a GPU.
    if (mirror) in.device_token_ids = truth.data();
    const vllm::ForwardLogits fl = ModelRegistry::Forward(*model, in);
    REQUIRE(fl.on_device());
    std::vector<float> rows(static_cast<size_t>(2 * kV));
    be.Copy(q, rows.data(), fl.device_tensor.data, rows.size() * sizeof(float));
    be.Synchronize(q);
    out.push_back(std::move(rows));
  }
  return out;
}

size_t Differing(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++n;
  return n;
}

}  // namespace

TEST_CASE(
    "Qwen3MoeForCausalLM embeds the async mirror's DEVICE ids, not the stale host "
    "vector") {
  Fixture fx(ConfigJson(), BuildTensors());
  REQUIRE(fx.cfg.num_experts == kE);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");
  // The decode-graph arm of the registry admits itself only where the platform
  // reports static-graph mode, which CPU does not. The harness swaps both
  // registries so the driver's OWN predicate is what routes, exactly as the
  // seam gates do.
  StaticGraphCpu harness;

  constexpr int kSteps = 4;

  // RUN A — the reference: the identifiers arrive on the host, no mirror.
  vt::ResetStepInputStats();
  const std::vector<std::vector<float>> ref = Run<CachePool>(fx, /*stale_host=*/false,
                                                  /*mirror=*/false, kSteps);
  {
    const vt::StepInputStats s = vt::GetStepInputStats();
    // The slot binds once and refreshes from the HOST every step; with no mirror
    // the device arm must never be taken.
    CHECK(s.host_refreshes == kSteps);
    CHECK(s.device_refreshes == 0);
    CHECK(s.binds >= 1);
  }

  // RUN B — THE CONTROL. Stale host identifiers and no mirror: the logits must
  // MOVE. Without this arm a model that ignored its identifiers entirely would
  // satisfy run C.
  vt::ResetStepInputStats();
  const std::vector<std::vector<float>> stale = Run<CachePool>(fx, /*stale_host=*/true,
                                                    /*mirror=*/false, kSteps);
  CHECK(vt::GetStepInputStats().device_refreshes == 0);
  // Only the COLD and CAPTURE steps recompute on this harness; a CPU replay
  // returns the slot's persistent logits unchanged, so steps 2 and 3 carry no
  // information either way and are not asserted on.
  CHECK(Differing(ref[0], stale[0]) > 0);
  CHECK(Differing(ref[1], stale[1]) > 0);

  // RUN C — THE GATE. The same stale host vector, with the true identifiers
  // reaching the model ONLY through `device_token_ids`.
  vt::ResetStepInputStats();
  const std::vector<std::vector<float>> via_device = Run<CachePool>(fx, /*stale_host=*/true,
                                                         /*mirror=*/true, kSteps);
  {
    const vt::StepInputStats s = vt::GetStepInputStats();
    // THE SEAM, ASSERTED. `device_refreshes` moves only inside
    // `vt::PersistentStepInput::RefreshFromDevice`; a hand-rolled copy in the
    // driver would produce identical logits and leave this at zero.
    CHECK(s.device_refreshes == kSteps);
    CHECK(s.host_refreshes == kSteps);
  }
  CHECK(Differing(ref[0], via_device[0]) == 0);
  CHECK(Differing(ref[1], via_device[1]) == 0);
  MESSAGE("registry forward, mirror vs host reference, bit for bit: "
          << ref[0].size() << " values per step, " << Differing(ref[0], via_device[0])
          << " and " << Differing(ref[1], via_device[1]) << " differing");
}

TEST_CASE(
    "DeepseekV2ForCausalLM embeds the async mirror's DEVICE ids, not the stale host "
    "vector") {
  Fixture fx(DsConfigJson(), DsBuildTensors());
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");
  StaticGraphCpu harness;

  constexpr int kSteps = 4;

  vt::ResetStepInputStats();
  const std::vector<std::vector<float>> ref =
      Run<DsCachePool>(fx, /*stale_host=*/false, /*mirror=*/false, kSteps);
  {
    const vt::StepInputStats s = vt::GetStepInputStats();
    CHECK(s.host_refreshes == kSteps);
    CHECK(s.device_refreshes == 0);
    CHECK(s.binds >= 1);
  }

  vt::ResetStepInputStats();
  const std::vector<std::vector<float>> stale =
      Run<DsCachePool>(fx, /*stale_host=*/true, /*mirror=*/false, kSteps);
  CHECK(vt::GetStepInputStats().device_refreshes == 0);
  CHECK(Differing(ref[0], stale[0]) > 0);
  CHECK(Differing(ref[1], stale[1]) > 0);

  vt::ResetStepInputStats();
  const std::vector<std::vector<float>> via_device =
      Run<DsCachePool>(fx, /*stale_host=*/true, /*mirror=*/true, kSteps);
  {
    const vt::StepInputStats s = vt::GetStepInputStats();
    CHECK(s.device_refreshes == kSteps);
    CHECK(s.host_refreshes == kSteps);
  }
  CHECK(Differing(ref[0], via_device[0]) == 0);
  CHECK(Differing(ref[1], via_device[1]) == 0);
  MESSAGE("registry forward, mirror vs host reference, bit for bit: "
          << ref[0].size() << " values per step, " << Differing(ref[0], via_device[0])
          << " and " << Differing(ref[1], via_device[1]) << " differing");
}
