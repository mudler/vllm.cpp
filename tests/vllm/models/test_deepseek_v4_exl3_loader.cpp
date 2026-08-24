// MODEL-DSV4-EXL3 W1b — the rank-sliced EXL3 loader arm.
//
// The SparkInfer artifact `0xSero/deepseek-v4-flash-0731-spark` stores its 216
// routed experts as EXL3 trellis tensors SLICED ACROSS FOUR TENSOR-PARALLEL
// RANKS and everything else ("carried") as the un-requantized DeepSeek-V4 FP8
// source tensors. Its `config.json` declares
// `quantization_config.quant_method == "exl3"` with
// `version == "rank-sliced-deepseek-v4-v1"` and
// `hybrid_tr3_tail.tensor_schema ==
//  "layers.{L}.ffn.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"`.
//
// The slicing is upstream's own `LinearEXL3.tp_import_split`
// (exllamav3 @ 2398c056, `modules/quant/exl3.py:296-313`): a split on the OUT
// features takes `svh[first:last]` and `trellis[:, first//16:last//16]`, and a
// split on the IN features takes `suh[first:last]` and
// `trellis[first//16:last//16, :]`. w1 and w3 are out-split, w2 is in-split, so
// coalescing back to TP1 is pure concatenation and is LOSSLESS: 16x16 trellis
// tiles are independent and every rank boundary here is a multiple of 128, the
// Hadamard block size (`exl3_lib/quantize.py:15`).
//
// This suite drives the PRODUCTION loader entry
// (`vllm::LoadDeepseekV4ForCausalLMWeights`, what
// `deepseek_v4_registry.cpp:89` calls) over a hermetic four-rank fixture the
// test writes itself, and it checks the coalesced owners BYTE FOR BYTE against
// the rank inputs it wrote. Real-checkpoint residency (this arm copies the
// tower into host owner buffers, which is ~100 GB on the real artifact) is the
// named MODEL-DSV4-EXL3 W2 residual; see the spec's `## Owed`.
#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

// ── the fixture's own tiny safetensors writer ──────────────────────────────

struct StEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

std::string WriteSafetensors(const std::filesystem::path& path,
                             const std::vector<StEntry>& entries) {
  nlohmann::json header = nlohmann::json::object();
  header["__metadata__"] = {{"format", "pt"}};
  uint64_t offset = 0;
  std::vector<uint8_t> data;
  for (const StEntry& e : entries) {
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {offset, offset + e.bytes.size()}}};
    data.insert(data.end(), e.bytes.begin(), e.bytes.end());
    offset += e.bytes.size();
  }
  const std::string text = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t n = text.size();
  for (int i = 0; i < 8; ++i) {
    const char byte = static_cast<char>((n >> (8 * i)) & 0xff);
    out.write(&byte, 1);
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  if (!out) throw std::runtime_error("failed to write fixture safetensors");
  const auto u8 = path.u8string();
  return std::string(u8.begin(), u8.end());
}

template <typename T>
std::vector<uint8_t> Raw(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}

class TempDir {
 public:
  TempDir() {
    static std::atomic<uint64_t> counter{0};
    static const uint64_t nonce = [] {
      std::random_device rd;
      return (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }();
    path_ = std::filesystem::temp_directory_path() /
            ("dsv4_exl3_" + std::to_string(nonce) + "_" +
             std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// ── the fixture's geometry ─────────────────────────────────────────────────
//
// Every dimension respects what the format actually requires: the trellis tile
// is 16x16 so both features are multiples of 16, and BOTH sides were
// Hadamard-128 transformed at quantization time so both the COALESCED and the
// PER-RANK features are multiples of 128.
constexpr int64_t kHidden = 256;    // hidden_size = w1/w3 in, w2 out
constexpr int64_t kInter = 512;     // moe_intermediate_size = w1/w3 out, w2 in
constexpr int kTp = 4;              // 512 / 4 = 128 per rank: one Hadamard block
constexpr int kBits = 3;            // 3.0 bpw -> K = 3, last trellis dim = 48
constexpr int kExperts = 2;
constexpr int kLayers = 1;

int64_t TrellisElems(int64_t k, int64_t n) { return (k / 16) * (n / 16) * (16 * kBits); }

// Deterministic, position-dependent contents so a misplaced slice cannot alias.
uint16_t TrellisWord(int expert, int proj, int rank, int64_t index) {
  const uint64_t h = 0x9E3779B97F4A7C15ull *
                     (static_cast<uint64_t>(index) * 131u + rank * 7919u +
                      proj * 104729u + expert * 1299709u + 1u);
  return static_cast<uint16_t>((h >> 27) & 0xffffu);
}
uint16_t SignWord(int expert, int proj, int rank, int64_t index, int side) {
  const uint64_t h = 0xD6E8FEB86659FD93ull *
                     (static_cast<uint64_t>(index) * 31u + rank * 1237u +
                      proj * 7717u + expert * 65537u + side * 4099u + 1u);
  return static_cast<uint16_t>((h >> 31) & 0x7bffu);  // stays a finite fp16
}

std::string Base(int layer, int expert, const char* proj) {
  return "layers." + std::to_string(layer) + ".ffn.experts." +
         std::to_string(expert) + "." + proj;
}

// w1 and w3 split on OUT features; w2 splits on IN (exl3.py:296-313).
bool SplitsOut(const char* proj) { return std::strcmp(proj, "w2") != 0; }

struct FixtureOptions {
  std::string version = "rank-sliced-deepseek-v4-v1";
  std::string codebook = "mcg";
  double bits = 3.0;
  int tp = kTp;
  int ranks_written = kTp;         // < tp leaves a rank missing
  std::string drop_tensor;         // one EXL3 tensor to omit entirely
  std::string extra_carried;       // one unroutable carried tensor to add
  bool swap_w2_slice_axis = false; // write w2 sliced on OUT instead of IN
};

nlohmann::json FixtureConfigJson(const FixtureOptions& opt) {
  nlohmann::json raw = nlohmann::json::object();
  raw["architectures"] = nlohmann::json::array({"DeepseekV4ForCausalLM"});
  raw["model_type"] = "deepseek_v4";
  raw["hidden_size"] = kHidden;
  raw["num_hidden_layers"] = kLayers;
  raw["vocab_size"] = 32;
  raw["num_attention_heads"] = 1;
  raw["num_key_value_heads"] = 1;
  raw["head_dim"] = 512;  // ParseDeepseekV4Params scopes the 512-wide MLA only
  raw["qk_rope_head_dim"] = 64;
  raw["q_lora_rank"] = 128;
  raw["o_lora_rank"] = 128;
  raw["o_groups"] = 1;
  raw["rms_norm_eps"] = 1e-6;
  raw["tie_word_embeddings"] = false;
  raw["max_position_embeddings"] = 128;
  raw["num_nextn_predict_layers"] = 1;
  raw["n_routed_experts"] = kExperts;
  raw["num_experts_per_tok"] = 1;
  raw["moe_intermediate_size"] = kInter;
  raw["n_shared_experts"] = 1;
  raw["norm_topk_prob"] = true;
  raw["routed_scaling_factor"] = 1.5;
  raw["swiglu_limit"] = 10.0;
  raw["scoring_func"] = "sqrtsoftplus";
  raw["topk_method"] = "noaux_tc";
  raw["num_hash_layers"] = 0;
  raw["expert_dtype"] = "fp4";
  raw["hc_mult"] = 2;
  raw["hc_sinkhorn_iters"] = 20;
  raw["hc_eps"] = 1e-6;
  raw["index_head_dim"] = 0;
  raw["index_n_heads"] = 0;
  raw["index_topk"] = 0;
  raw["compress_ratios"] = nlohmann::json::array({0});
  raw["quantization_config"] = {
      {"quant_method", "exl3"},
      {"version", opt.version},
      {"bits", opt.bits},
      {"codebook", opt.codebook},
      {"source_format", "packed_e2m1_fp4_with_ue8m0_scales"}};
  raw["hybrid_tr3_tail"] = {
      {"tp", opt.tp},
      {"format", "exl3-trellis"},
      {"tensor_schema",
       "layers.{L}.ffn.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"}};
  return raw;
}

vllm::HfConfig FixtureConfig(const FixtureOptions& opt) {
  vllm::HfConfig c;
  c.model_type = "deepseek_v4";
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = kHidden;
  c.num_hidden_layers = kLayers;
  c.vocab_size = 32;
  c.num_attention_heads = 1;
  c.head_dim = 512;
  c.torch_dtype = "bfloat16";
  c.raw = FixtureConfigJson(opt);
  return c;
}

// The carried half: the un-requantized DeepSeek-V4 source tensors, by the exact
// names the real `carried-*.safetensors` use. This arm ACCOUNTS for them by
// name exactly as the pre-existing safetensors pass does (materializing the
// FP8-block MLA tower is the standing W2b residual of deepseek-v4-flash.md),
// so one element per tensor is enough to prove the routing.
std::vector<StEntry> CarriedEntries(const FixtureOptions& opt) {
  std::vector<StEntry> e;
  const auto one = [&](const std::string& name) {
    e.push_back({name, "F32", {1}, Raw(std::vector<float>{0.25f})});
  };
  one("embed.weight");
  one("norm.weight");
  one("head.weight");
  one("hc_head_base");
  one("hc_head_fn");
  one("hc_head_scale");
  for (int l = 0; l < kLayers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    one(b + "attn_norm.weight");
    one(b + "ffn_norm.weight");
    for (const char* h : {"hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                          "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"})
      one(b + h);
    const std::string a = b + "attn.";
    for (const char* w : {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}) {
      one(a + w + ".weight");
      one(a + w + ".scale");
    }
    one(a + "q_norm.weight");
    one(a + "kv_norm.weight");
    one(a + "attn_sink");
    const std::string f = b + "ffn.";
    one(f + "gate.weight");
    one(f + "gate.bias");
    for (const char* w : {"w1", "w2", "w3"}) {
      one(f + "shared_experts." + w + ".weight");
      one(f + "shared_experts." + w + ".scale");
    }
  }
  // The real artifact carries an MTP tail. vLLM's DeepSeek-V4 loader skips it
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474) and so
  // must this arm, WITHOUT reporting the tensors as unroutable.
  one("mtp.0.attn_norm.weight");
  one("mtp.0.ffn.experts.0.w1.weight");
  if (!opt.extra_carried.empty()) one(opt.extra_carried);
  return e;
}

// One rank shard, sliced exactly as `tp_import_split` would have produced it.
std::vector<StEntry> RankEntries(int rank, const FixtureOptions& opt) {
  std::vector<StEntry> e;
  const std::string suffix = ".rank" + std::to_string(rank);
  for (int l = 0; l < kLayers; ++l) {
    for (int x = 0; x < kExperts; ++x) {
      int proj_index = 0;
      for (const char* proj : {"w1", "w2", "w3"}) {
        const std::string base = Base(l, x, proj) + suffix;
        const bool out_split = opt.swap_w2_slice_axis ? true : SplitsOut(proj);
        const int64_t full_in = SplitsOut(proj) ? kHidden : kInter;
        const int64_t full_out = SplitsOut(proj) ? kInter : kHidden;
        const int64_t k = out_split ? full_in : full_in / opt.tp;
        const int64_t n = out_split ? full_out / opt.tp : full_out;

        std::vector<uint16_t> trellis(static_cast<size_t>(TrellisElems(k, n)));
        for (size_t i = 0; i < trellis.size(); ++i)
          trellis[i] = TrellisWord(x, proj_index, rank, static_cast<int64_t>(i));
        std::vector<uint16_t> suh(static_cast<size_t>(k)), svh(static_cast<size_t>(n));
        // The INVARIANT side is identical across ranks (every rank received the
        // whole vector); the SPLIT side differs per rank.
        for (int64_t i = 0; i < k; ++i)
          suh[static_cast<size_t>(i)] =
              SignWord(x, proj_index, out_split ? 0 : rank, i, 0);
        for (int64_t j = 0; j < n; ++j)
          svh[static_cast<size_t>(j)] =
              SignWord(x, proj_index, out_split ? rank : 0, j, 1);

        const std::vector<std::pair<std::string, StEntry>> four = {
            {base + ".trellis",
             {base + ".trellis", "I16", {k / 16, n / 16, 16 * kBits}, Raw(trellis)}},
            {base + ".suh", {base + ".suh", "F16", {k}, Raw(suh)}},
            {base + ".svh", {base + ".svh", "F16", {n}, Raw(svh)}},
            {base + ".mcg",
             {base + ".mcg", "I32", {}, Raw(std::vector<int32_t>{-877912083})}},
        };
        for (const auto& [name, entry] : four)
          if (name != opt.drop_tensor) e.push_back(entry);
        ++proj_index;
      }
    }
  }
  return e;
}

// doctest::Contains has no `operator&&`, and a refusal is only useful if the
// message names BOTH the offending tensor and the row that owes the arm — so
// capture the message once and assert over it.
std::string ThrowMessage(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}
bool Mentions(const std::string& message, const std::string& needle) {
  return message.find(needle) != std::string::npos;
}

struct Fixture {
  TempDir dir;
  std::vector<vllm::SafetensorsFile> shards;
  vllm::HfConfig config;
};

std::unique_ptr<Fixture> BuildFixture(const FixtureOptions& opt = {}) {
  auto f = std::make_unique<Fixture>();
  f->config = FixtureConfig(opt);
  f->shards.push_back(vllm::SafetensorsFile::Open(
      WriteSafetensors(f->dir.path() / "carried-001.safetensors", CarriedEntries(opt))));
  for (int r = 0; r < opt.ranks_written; ++r) {
    f->shards.push_back(vllm::SafetensorsFile::Open(WriteSafetensors(
        f->dir.path() / ("exl3-layer-000-tp4-rank" + std::to_string(r) + ".safetensors"),
        RankEntries(r, opt))));
  }
  return f;
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("dsv4 exl3: the rank-sliced arm is detected and coalesces TP4 -> TP1") {
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  REQUIRE(w.has_exl3_weights);
  CHECK(w.exl3.tp == kTp);
  CHECK(w.exl3.bits == kBits);
  CHECK(w.exl3.codebook == "mcg");
  CHECK(w.exl3.version == "rank-sliced-deepseek-v4-v1");
  REQUIRE(w.exl3.layers.size() == static_cast<size_t>(kLayers));
  REQUIRE(w.exl3.layers[0].experts.size() == static_cast<size_t>(kExperts));

  // 35 carried + 1 layer * 2 experts * 3 projections * 4 ranks * 4 tensors.
  CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * kTp * 4);
  // The MTP tail is skipped exactly as vLLM's own DeepSeek-V4 loader skips it
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474) — but
  // COUNTED, so the skip is visible rather than silent. On the real artifact
  // these are 3985 NVFP4 draft-head tensors.
  CHECK(w.exl3.skipped_mtp_tensors == 2);

  const vllm::DeepseekV4Exl3Expert& e0 = w.exl3.layers[0].experts[0];
  for (const vllm::DeepseekV4Exl3Linear* lin : {&e0.w1, &e0.w3}) {
    CHECK(lin->in_features == kHidden);
    CHECK(lin->out_features == kInter);
    CHECK(lin->bits == kBits);
    CHECK(lin->suh.size() == static_cast<size_t>(kHidden));
    CHECK(lin->svh.size() == static_cast<size_t>(kInter));
    CHECK(lin->trellis.size() == static_cast<size_t>(TrellisElems(kHidden, kInter)));
  }
  CHECK(e0.w2.in_features == kInter);
  CHECK(e0.w2.out_features == kHidden);
  CHECK(e0.w2.suh.size() == static_cast<size_t>(kInter));
  CHECK(e0.w2.svh.size() == static_cast<size_t>(kHidden));
  CHECK(e0.w2.trellis.size() == static_cast<size_t>(TrellisElems(kInter, kHidden)));
  CHECK(e0.w1.mcg == -877912083);
}

TEST_CASE("dsv4 exl3: the coalesced owners equal the rank inputs BYTE FOR BYTE") {
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  int trellis_mismatch = 0, sign_mismatch = 0;
  for (int x = 0; x < kExperts; ++x) {
    const vllm::DeepseekV4Exl3Expert& ex = w.exl3.layers[0].experts[x];
    int proj_index = 0;
    for (const char* proj : {"w1", "w2", "w3"}) {
      const vllm::DeepseekV4Exl3Linear& lin =
          std::strcmp(proj, "w1") == 0 ? ex.w1
                                       : (std::strcmp(proj, "w2") == 0 ? ex.w2 : ex.w3);
      const bool out_split = SplitsOut(proj);
      const int64_t k = lin.in_features, n = lin.out_features;
      const int64_t tn = n / 16;
      const int64_t words = 16 * kBits;
      for (int r = 0; r < kTp; ++r) {
        // The per-rank slice this rank file was written with.
        const int64_t rk = out_split ? k : k / kTp;
        const int64_t rn = out_split ? n / kTp : n;
        const int64_t rtk = rk / 16, rtn = rn / 16;
        for (int64_t i = 0; i < rtk; ++i) {
          for (int64_t j = 0; j < rtn; ++j) {
            for (int64_t t = 0; t < words; ++t) {
              const uint16_t want =
                  TrellisWord(x, proj_index, r, (i * rtn + j) * words + t);
              // OUT split concatenates along trellis dim 1, IN split along dim 0.
              const int64_t gi = out_split ? i : (r * rtk + i);
              const int64_t gj = out_split ? (r * rtn + j) : j;
              if (lin.trellis[static_cast<size_t>((gi * tn + gj) * words + t)] != want)
                ++trellis_mismatch;
            }
          }
        }
        // The SPLIT sign vector is concatenated; the INVARIANT one came whole.
        if (out_split) {
          for (int64_t j = 0; j < rn; ++j)
            if (lin.svh[static_cast<size_t>(r * rn + j)] !=
                SignWord(x, proj_index, r, j, 1))
              ++sign_mismatch;
        } else {
          for (int64_t i = 0; i < rk; ++i)
            if (lin.suh[static_cast<size_t>(r * rk + i)] !=
                SignWord(x, proj_index, r, i, 0))
              ++sign_mismatch;
        }
      }
      for (int64_t i = 0; i < (out_split ? k : 0); ++i)
        if (lin.suh[static_cast<size_t>(i)] != SignWord(x, proj_index, 0, i, 0))
          ++sign_mismatch;
      for (int64_t j = 0; j < (out_split ? 0 : n); ++j)
        if (lin.svh[static_cast<size_t>(j)] != SignWord(x, proj_index, 0, j, 1))
          ++sign_mismatch;
      ++proj_index;
    }
  }
  CHECK(trellis_mismatch == 0);
  CHECK(sign_mismatch == 0);

  // The split inputs are NOT retained: the tower holds exactly the coalesced
  // bytes, never the four rank copies as well.
  int64_t want_bytes = 0;
  for (int x = 0; x < kExperts; ++x) {
    want_bytes += (TrellisElems(kHidden, kInter) + kHidden + kInter) * 2 * 2;
    want_bytes += (TrellisElems(kInter, kHidden) + kInter + kHidden) * 2;
  }
  CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == want_bytes);
}

TEST_CASE("dsv4 exl3: the arm is REACHED from the registry's production load") {
  auto f = BuildFixture();
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(f->shards);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = vllm::ModelRegistry::Load(f->config, source));
  CHECK(model != nullptr);
}

TEST_CASE("dsv4 exl3: unrepresentable inputs REFUSE BY NAME") {
  SUBCASE("an unknown rank-sliced schema version") {
    FixtureOptions opt;
    opt.version = "rank-sliced-deepseek-v4-v2";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "rank-sliced-deepseek-v4-v2"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a codebook other than mcg") {
    FixtureOptions opt;
    opt.codebook = "mul1";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "mul1"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a missing rank tensor") {
    FixtureOptions opt;
    opt.drop_tensor = "layers.0.ffn.experts.1.w2.rank2.suh";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "layers.0.ffn.experts.1.w2.rank2.suh"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a whole missing rank") {
    FixtureOptions opt;
    opt.ranks_written = 3;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "rank3"));
  }
  SUBCASE("a carried tensor no arm routes") {
    FixtureOptions opt;
    opt.extra_carried = "layers.0.attn.wq_c.weight";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "layers.0.attn.wq_c.weight"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("w2 sliced on the wrong axis") {
    FixtureOptions opt;
    opt.swap_w2_slice_axis = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "w2"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    // Pin the REASON, not just the fact of a throw: a refusal test that passes
    // because some earlier check happened to fire gates nothing. This must be
    // the replicated-side comparison noticing that w2's svh differs per rank.
    CHECK(Mentions(msg, "IN-split"));
    CHECK(Mentions(msg, "svh"));
  }
}
