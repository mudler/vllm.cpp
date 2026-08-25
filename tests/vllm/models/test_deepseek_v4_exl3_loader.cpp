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

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/core/kv_cache_utils.h"  // host_available_memory_bytes
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

// The bytes the COALESCED TP1 tower holds, in one place so the byte-parity case
// and the residency case cannot drift apart. Per expert: w1 and w3 are
// [kHidden, kInter] and w2 is [kInter, kHidden]; each linear is its trellis plus
// its two fp16 sign vectors, every element 16-bit.
int64_t ExpectedTowerBytes() {
  int64_t bytes = 0;
  for (int x = 0; x < kExperts; ++x) {
    bytes += (TrellisElems(kHidden, kInter) + kHidden + kInter) * 2 * 2;
    bytes += (TrellisElems(kInter, kHidden) + kInter + kHidden) * 2;
  }
  return bytes * kLayers;
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
  // ── the NEGATIVE direction of the detection predicate ────────────────────
  // `quantization_config.quant_method`. Anything but "exl3" must take the
  // pre-existing dense arm, and the vehicle that proves it has to be a REAL
  // one: `deepseek_v4_fp8` carries a `quantization_config` of its own.
  std::string quant_method = "exl3";
  bool omit_quant_config = false;      // write no `quantization_config` block
  bool dense_routed_experts = false;   // dense NVFP4 experts, no rank shards
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
  if (opt.omit_quant_config) {
    // No block at all: the unquantized vehicle, and the one shape a predicate
    // that dropped its null guard would misread.
  } else if (opt.quant_method == "exl3") {
    raw["quantization_config"] = {
        {"quant_method", "exl3"},
        {"version", opt.version},
        {"bits", opt.bits},
        {"codebook", opt.codebook},
        {"source_format", "packed_e2m1_fp4_with_ue8m0_scales"}};
  } else {
    // The plain `deepseek_v4_fp8` block, copied in shape from the REAL
    // artifact's own `quantization_config.base_quantization_config`
    // (`0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32`, config.json):
    // `{activation_scheme: dynamic, fmt: e4m3, quant_method: fp8,
    //   scale_fmt: ue8m0, weight_block_size: [128, 128]}`. It carries NO
    // `version` and NO `codebook`, which is exactly why a widened detection
    // predicate would carry it into the EXL3 arm and die there instead of
    // loading.
    raw["quantization_config"] = {
        {"quant_method", opt.quant_method},
        {"activation_scheme", "dynamic"},
        {"fmt", "e4m3"},
        {"scale_fmt", "ue8m0"},
        {"weight_block_size", nlohmann::json::array({128, 128})}};
  }
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
  // The NON-EXL3 vehicle's routed experts: dense NVFP4, the four suffixes the
  // pre-existing arm's name-map requires for `expert_dtype == "fp4"` (the
  // `expert_suffixes` vector in `LoadDeepseekV4ForCausalLMWeights`,
  // `src/vllm/model_executor/models/deepseek_v4_weights.cpp`). Cited by SYMBOL
  // deliberately: the commit that first wrote this comment cited a line range,
  // and the SAME commit inserted 71 lines above it, so the anchor was stale
  // before it was ever read. Present only for the negative-direction cases,
  // where there are no rank shards at all.
  if (opt.dense_routed_experts) {
    for (int l = 0; l < kLayers; ++l) {
      const std::string f = "layers." + std::to_string(l) + ".ffn.experts.";
      for (int x = 0; x < kExperts; ++x)
        for (const char* w : {"w1", "w2", "w3"})
          for (const char* suf : {".weight", ".weight_scale", ".weight_scale_2",
                                  ".input_scale"})
            one(f + std::to_string(x) + "." + w + suf);
    }
  }
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

// REAL fd 2 by dup/dup2, not a `std::cerr` rdbuf swap: the residency line is
// written with `std::fprintf(stderr, ...)`, which an rdbuf swap cannot see. A
// capture that could not see the line it exists to read would return an empty
// string and be indistinguishable from "the loader never emitted it", which is
// the instrument failing toward a verdict about the code
// (`.agents/verification.md`; the same reasoning as
// `tests/vllm/v1/spec_decode/dflash2_runner_fixture.h:468`).
std::string CaptureStderr(const std::function<void()>& body) {
  std::FILE* cap = std::tmpfile();
  REQUIRE(cap != nullptr);
  std::fflush(stderr);
  const int saved = ::dup(STDERR_FILENO);
  REQUIRE(saved >= 0);
  REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);
  body();
  std::fflush(stderr);
  const int restored = ::dup2(saved, STDERR_FILENO);
  ::close(saved);
  std::rewind(cap);
  std::string out;
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) out.append(buf, n);
  std::fclose(cap);
  REQUIRE(restored >= 0);
  return out;
}

// The GiB figure the LOAD printed after `host MemAvailable=`, or -1.0 when the
// line took the unknown branch (or was never emitted). Parsed rather than
// string-matched so the assertion can compare the reported budget against the
// pool it claims to have measured — a substring check for "MemAvailable" alone
// matches both branches of the report, which is exactly the hole MINOR-1 found.
double ReportedMemAvailableGiB(const std::string& log) {
  static const std::string kKey = "host MemAvailable=";
  const size_t at = log.find(kKey);
  if (at == std::string::npos) return -1.0;
  return std::strtod(log.c_str() + at + kKey.size(), nullptr);
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
  const int rank_shards = opt.dense_routed_experts ? 0 : opt.ranks_written;
  for (int r = 0; r < rank_shards; ++r) {
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
  CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == ExpectedTowerBytes());
}

TEST_CASE("dsv4 exl3: the LOAD reports the tower's residency and refuses one that does not fit") {
  // WHY THIS CASE EXISTS. `DeepseekV4Exl3ResidentBytes` had no production caller
  // at all: only the case above called it, which measures a function rather than
  // a capability ("Nothing lands dead", AGENTS.md). It is also exactly the
  // instrument the row's residency RISK needs — the real artifact's trellis alone
  // is ~83.5 GiB (43 layers x 216 experts x 3 projections) on a box whose unified
  // memory OOM-reboots. So the load itself now reports the figure and refuses a
  // projection the host cannot hold, and this case gates both halves.
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w;
  const std::string log = CaptureStderr([&] {
    w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  });
  REQUIRE(w.has_exl3_weights);

  // REACHABILITY: this line is emitted BY THE LOAD, so deleting the production
  // call site takes this case red.
  CAPTURE(log);
  CHECK(log.find("[vt load] dsv4-exl3:") != std::string::npos);
  CHECK(log.find("resident_bytes=" + std::to_string(ExpectedTowerBytes())) !=
        std::string::npos);

  // ...AND IT MUST CARRY THE REAL BUDGET. The two assertions above match BOTH
  // branches of the report, which is how the fresh review (2026-08-24, MINOR-1)
  // wired `host_available` to a literal 0 at the production call site — symbol
  // still referenced, so it compiled (ninja rc=0, 3 steps) — and watched this
  // suite stay 6/6 66/66 SUCCESS while the refusal went silently inert and the
  // load printed `/proc/meminfo unreadable` on a host where it reads fine. The
  // budget the load ACTUALLY used is therefore asserted, branched on whether
  // THIS host can read the pool at all.
  const int64_t budget_now = vllm::v1::host_available_memory_bytes();
  CAPTURE(budget_now);
  if (budget_now > 0) {
    CHECK(log.find("host MemAvailable=") != std::string::npos);
    CHECK(log.find("MemAvailable unknown") == std::string::npos);
    // Same POOL, not merely some non-zero constant. The window is deliberately
    // wide (1/8x .. 8x) because MemAvailable moves under other work on the box
    // between the load and this second read; it is still far tighter than the
    // 1 MiB and 1 TiB brackets the refusal cases below inject, so a call site
    // rewired to either of those constants fails here.
    const double reported = ReportedMemAvailableGiB(log);
    const double now_gib = static_cast<double>(budget_now) / (1024.0 * 1024.0 * 1024.0);
    CAPTURE(reported);
    CAPTURE(now_gib);
    CHECK(reported > 0.0);
    CHECK(reported >= now_gib / 8.0);
    CHECK(reported <= now_gib * 8.0);
  } else {
    // /proc/meminfo is genuinely unreadable here, so the unknown branch is the
    // CORRECT report and the refusal is correctly inert.
    CHECK(log.find("MemAvailable unknown") != std::string::npos);
  }

  // The REFUSAL, driven through the same production function the load calls,
  // with the budget INJECTED. `check_enough_state_memory`
  // (`vllm/v1/core/kv_cache_utils.h`) is parameterised for exactly this reason:
  // a refusal observable only on a box of a chosen size is not gateable.
  const std::string refusal = ThrowMessage([&] {
    (void)vllm::ReportDeepseekV4Exl3Residency(w, /*layers_done=*/1,
                                              /*layers_total=*/43,
                                              /*host_available_bytes=*/1 << 20);
  });
  CAPTURE(refusal);
  CHECK(Mentions(refusal, "MODEL-DSV4-EXL3"));
  CHECK(Mentions(refusal, "MemAvailable"));
  // An UNKNOWN budget never refuses — `host_available_memory_bytes()` returns 0
  // when /proc/meminfo is unreadable, and `VT_DSV4_EXL3_HOST_BUDGET=0` hands the
  // reporter the same 0 on purpose. An unknown budget must not become a false
  // refusal (`host_available_memory_bytes`, `kv_cache_utils.cpp`, keeps the same
  // polarity).
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43, 0);
        }).empty());
  // A budget that holds the projection does not refuse either.
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43, int64_t{1} << 40);
        }).empty());
  // THE INCLUSIVE EDGE (fresh review, NIT-1). The two cases above bracket the
  // threshold at 1 MiB and 1 TiB against a ~304 KiB tower, which catches a
  // direction flip but NOT `projected <= budget` narrowing to `projected <
  // budget`. A projection that EQUALS the budget must load: with layers_done=1
  // the projection is exactly `tower * layers_total`.
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43,
                                                    ExpectedTowerBytes() * 43);
        }).empty());
}

TEST_CASE(
    "dsv4 exl3: VT_DSV4_EXL3_HOST_BUDGET defaults ON; only a '0'-leading value "
    "disables the refusal") {
  // The refusal's budget is a HEURISTIC (`/proc/meminfo` MemAvailable ignores
  // swap, and in a container reports the HOST's pool rather than the cgroup's
  // — see the caveats at the read site in `LoadDeepseekV4Exl3`), so it ships
  // with a same-binary escape hatch. This pins the PARSE, which is factored into
  // the header precisely so it is gateable without mutating the environment
  // (house shape: `AsyncRunnerFlagIsOn`, tests/vllm/v1/worker/
  // test_async_runner_flag.cpp).
  using vllm::Dsv4Exl3HostBudgetFlagIsOn;
  // Default (unset) is ON: the refusal ships enabled, because on the
  // unified-memory box this arm targets an over-commit reboots the machine.
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON, including the explicit opt-in and junk.
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("1"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(""));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("on"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("true"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(" 0"));  // leading space, not a '0' first char
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("10"));
  // Disabled: FIRST character '0'.
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("0"));
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("00"));
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("0abc"));
}

TEST_CASE("dsv4 exl3: a NON-exl3 quantization_config takes the DENSE arm") {
  // The detection predicate was gated only in the POSITIVE direction: every case
  // above hands it an EXL3 checkpoint, so widening `quant_method == "exl3"` to an
  // always-true test left four dsv4 suites green (fresh review, 2026-08-24).
  // That matters because the plain vehicle is not "no quantization_config": the
  // `deepseek_v4_fp8` artifact carries one, and this very checkpoint stores it
  // verbatim as `quantization_config.base_quantization_config`. A regression that
  // widened the predicate would route FP8 into the trellis arm undetected.
  SUBCASE("quant_method fp8 — the deepseek_v4_fp8 vehicle") {
    FixtureOptions opt;
    opt.quant_method = "fp8";
    opt.dense_routed_experts = true;
    auto f = BuildFixture(opt);
    vllm::DeepseekV4Weights w;
    // Through ThrowMessage rather than bare: a widened predicate sends this
    // fixture into the EXL3 arm, which THROWS on the absent `version`, and an
    // uncaught throw is a failed CASE with `assertions: N | N passed` — a red
    // that reads as a pass in the summary line. Captured, it is a named
    // assertion that prints the refusal it got instead.
    const std::string msg = ThrowMessage(
        [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    REQUIRE(msg.empty());
    CHECK_FALSE(w.has_exl3_weights);
    CHECK(w.exl3.layers.empty());
    CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == 0);
    // 35 carried + 1 layer * 2 experts * 3 projections * 4 NVFP4 suffixes. The
    // dense arm walks its own name-map, which the EXL3 arm does not.
    CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * 4);
  }
  SUBCASE("no quantization_config at all") {
    // This one does NOT discriminate the `== "exl3"` widening, because the null
    // guard above it already returns false — it guards the OTHER widening, a
    // predicate that drops that guard.
    //
    // HOW IT DISCRIMINATES, AND WHY THAT IS NOT AN ASSERTION (fresh review,
    // NIT-2). Dropping `qc == nullptr` from `IsExl3Checkpoint` is not a
    // behavioural widening this subcase can observe by value: it is a null
    // dereference inside the predicate itself, so the process dies with SIGSEGV
    // before any CHECK below runs. MEASURED, not assumed: the binary exits 139
    // and doctest prints `5 | 4 passed | 1 failed | 2 skipped` with
    // `assertions: 63 | 63 passed | 0 failed` and `Status: FAILURE!` — a real
    // ctest red under a CLEAN assertion counter, with two later cases never run.
    // DO NOT grep that counter for this case's verdict; read the exit status.
    //
    // It was left this way on purpose. Making it fail by assertion instead needs
    // the deref replaced with a checked accessor, which means deleting the very
    // guard under test; and the sibling `!qc->is_object()` half cannot be
    // discriminated at all, because `nlohmann::json::contains` is already safe
    // on a non-object, so a fixture with a string-valued `quantization_config`
    // would take the dense arm either way and would assert nothing. A
    // memory-safety defect's discriminator is a crash or a sanitizer, not a
    // CHECK.
    FixtureOptions opt;
    opt.omit_quant_config = true;
    opt.dense_routed_experts = true;
    auto f = BuildFixture(opt);
    vllm::DeepseekV4Weights w;
    const std::string msg = ThrowMessage(
        [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    REQUIRE(msg.empty());
    CHECK_FALSE(w.has_exl3_weights);
    CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == 0);
    CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * 4);
  }
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
