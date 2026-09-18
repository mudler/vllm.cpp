// Unit tests for runtime prompt-activated LoRA (ROAD-V1-LORA-RUNTIME).
// Tests DitParseLoraTags: prompt tag parsing, name resolution, and the
// DitRuntimeLoraState lookup helpers. Also tests DitApplyRuntimeLoraDelta:
// the CPU-path additive delta computation.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit_lora.h"
#include "vt/dtype.h"

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

std::string Caught(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// A temp directory that cleans itself up.
struct TempDir {
  std::string path;
  static int counter;
  TempDir() {
    path = "/tmp/vllm_runtime_lora_" + std::to_string(getpid()) + "_" +
           std::to_string(++counter);
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
  std::string child(const std::string& name) const { return path + "/" + name; }
  void touch(const std::string& name) const {
    FILE* f = fopen(child(name).c_str(), "w");
    if (f) fclose(f);
  }
};

int TempDir::counter = 0;

}  // namespace

// ── prompt tag parsing ──────────────────────────────────────────────────────

TEST_CASE("runtime lora: single tag is parsed and prompt cleaned") {
  TempDir dir;
  dir.touch("adapter.safetensors");
  auto r = vllm::DitParseLoraTags("hello <lora:adapter:1.0> world", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].path == dir.child("adapter.safetensors"));
  REQUIRE(r.loras[0].strength == doctest::Approx(1.0));
  REQUIRE(r.clean_prompt == "hello world");
}

TEST_CASE("runtime lora: multiple tags produce multiple specs") {
  TempDir dir;
  dir.touch("a.safetensors");
  dir.touch("b.safetensors");
  auto r = vllm::DitParseLoraTags("<lora:a:0.5> <lora:b:1.5>", dir.path);
  REQUIRE(r.loras.size() == 2);
  REQUIRE(r.loras[0].path == dir.child("a.safetensors"));
  REQUIRE(r.loras[0].strength == doctest::Approx(0.5));
  REQUIRE(r.loras[1].path == dir.child("b.safetensors"));
  REQUIRE(r.loras[1].strength == doctest::Approx(1.5));
  REQUIRE(r.clean_prompt == "");
}

TEST_CASE("runtime lora: duplicate name accumulates strengths") {
  TempDir dir;
  dir.touch("adapter.safetensors");
  auto r = vllm::DitParseLoraTags(
      "<lora:adapter:0.3> mid <lora:adapter:0.7>", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].strength == doctest::Approx(1.0));
  REQUIRE(r.clean_prompt == "mid");
}

TEST_CASE("runtime lora: no tags returns clean prompt as-is") {
  TempDir dir;
  auto r = vllm::DitParseLoraTags("just a prompt", dir.path);
  REQUIRE(r.loras.empty());
  REQUIRE(r.clean_prompt == "just a prompt");
}

TEST_CASE("runtime lora: whitespace around tag is collapsed") {
  TempDir dir;
  dir.touch("adapter.safetensors");
  auto r = vllm::DitParseLoraTags("hello   <lora:adapter:1.0>   world", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.clean_prompt == "hello world");
}

TEST_CASE("runtime lora: tag with path separator in name is refused") {
  TempDir dir;
  std::string err = Caught([&] {
    vllm::DitParseLoraTags("<lora:sub/dir:1.0>", dir.path);
  });
  REQUIRE(Mentions(err, "path separator"));
}

TEST_CASE("runtime lora: bad strength is refused") {
  TempDir dir;
  dir.touch("adapter.safetensors");
  std::string err = Caught([&] {
    vllm::DitParseLoraTags("<lora:adapter:abc>", dir.path);
  });
  REQUIRE(Mentions(err, "not a finite number"));
}

// ── name resolution ─────────────────────────────────────────────────────────

TEST_CASE("runtime lora: exact filename in lora_dir resolves") {
  TempDir dir;
  dir.touch("my_adapter.safetensors");
  auto r = vllm::DitParseLoraTags("<lora:my_adapter.safetensors:1.0>", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].path == dir.child("my_adapter.safetensors"));
}

TEST_CASE("runtime lora: extension probing adds .safetensors") {
  TempDir dir;
  dir.touch("adapter.safetensors");
  auto r = vllm::DitParseLoraTags("<lora:adapter:1.0>", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].path == dir.child("adapter.safetensors"));
}

TEST_CASE("runtime lora: case-insensitive match resolves") {
  TempDir dir;
  dir.touch("Adapter.SAFETENSORS");
  auto r = vllm::DitParseLoraTags("<lora:adapter.safetensors:1.0>", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].path == dir.child("Adapter.SAFETENSORS"));
}

TEST_CASE("runtime lora: case-insensitive match without extension") {
  TempDir dir;
  dir.touch("Adapter.Safetensors");
  auto r = vllm::DitParseLoraTags("<lora:adapter:1.0>", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].path == dir.child("Adapter.Safetensors"));
}

TEST_CASE("runtime lora: absolute path resolves") {
  TempDir dir;
  dir.touch("adapter.safetensors");
  std::string abs = dir.child("adapter.safetensors");
  auto r = vllm::DitParseLoraTags("<lora:" + abs + ":1.0>", dir.path);
  REQUIRE(r.loras.size() == 1);
  REQUIRE(r.loras[0].path == abs);
}

TEST_CASE("runtime lora: nonexistent name is refused") {
  TempDir dir;
  std::string err = Caught([&] {
    vllm::DitParseLoraTags("<lora:nonexistent:1.0>", dir.path);
  });
  REQUIRE(Mentions(err, "was not found"));
}

// ── DitRuntimeLoraState helpers ──────────────────────────────────────────────

TEST_CASE("runtime lora: empty state returns nullptr from Find") {
  vllm::DitRuntimeLoraState state;
  REQUIRE(state.empty());
  REQUIRE(state.Find("blocks.0.attn.qkv_proj.weight") == nullptr);
}

TEST_CASE("runtime lora: Find returns layer for known target") {
  vllm::DitRuntimeLoraState state;
  state.layers["blocks.0.attn.qkv_proj.weight"] = {};
  REQUIRE(!state.empty());
  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  REQUIRE(state.Find("blocks.0.attn.out_proj.weight") == nullptr);
}

// ── DitApplyRuntimeLoraDelta (CPU path) ─────────────────────────────────────

namespace {

// Build a contiguous f32 tensor on CPU backed by `buf`.
vt::Tensor F32Tensor(std::vector<float>& buf, std::initializer_list<int64_t> shape) {
  return vt::Tensor::Contiguous(buf.data(), vt::DType::kF32,
                                vt::Device{vt::DeviceType::kCPU, 0}, shape);
}

// Compute the expected delta: (a @ lora_a^T) @ lora_b^T, then add to base.
// a is [rows, in_f], lora_a is [rank, in_f], lora_b is [out_f, rank].
std::vector<float> ExpectedDeltaAdded(
    int64_t rows, int64_t in_f, int64_t rank, int64_t out_f,
    const std::vector<float>& a,
    const std::vector<float>& lora_a,
    const std::vector<float>& lora_b,
    const std::vector<float>& base_out) {
  std::vector<float> out(rows * out_f);
  for (int64_t m = 0; m < rows; ++m) {
    for (int64_t n = 0; n < out_f; ++n) {
      float delta = 0.0f;
      for (int64_t r = 0; r < rank; ++r) {
        float tmp = 0.0f;
        for (int64_t k = 0; k < in_f; ++k)
          tmp += a[m * in_f + k] * lora_a[r * in_f + k];
        delta += tmp * lora_b[n * rank + r];
      }
      out[m * out_f + n] = base_out[m * out_f + n] + delta;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("runtime lora delta: null lora is a no-op") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<float> a_buf = {1, 2, 3, 4, 5, 6};
  vt::Tensor a = F32Tensor(a_buf, {2, 3});
  std::vector<float> out = {10, 20, 30, 40, 50, 60, 70, 80};
  vllm::DitApplyRuntimeLoraDelta(q, a, out.data(), 2, 4, nullptr);
  CHECK(out[0] == doctest::Approx(10));
  CHECK(out[7] == doctest::Approx(80));
}

TEST_CASE("runtime lora delta: known values match hand-computed delta") {
  const int64_t rows = 2, in_f = 3, rank = 2, out_f = 4;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // a [2,3]
  std::vector<float> a_buf = {1, 2, 3, 4, 5, 6};
  vt::Tensor a = F32Tensor(a_buf, {rows, in_f});

  // lora_a [2,3] — identity-like
  std::vector<float> la_buf = {1, 0, 0, 0, 1, 0};
  vt::Tensor lora_a = F32Tensor(la_buf, {rank, in_f});

  // lora_b [4,2]
  std::vector<float> lb_buf = {1, 0, 0, 1, 1, 1, 0, 0};
  vt::Tensor lora_b = F32Tensor(lb_buf, {out_f, rank});

  vllm::DitRuntimeLoraLayer layer;
  layer.lora_a = lora_a;
  layer.lora_b = lora_b;
  layer.strength = 1.0f;

  std::vector<float> out = {10, 20, 30, 40, 50, 60, 70, 80};
  vllm::DitApplyRuntimeLoraDelta(q, a, out.data(), rows, out_f, &layer);

  const std::vector<float> expected = ExpectedDeltaAdded(
      rows, in_f, rank, out_f, a_buf, la_buf, lb_buf, {10, 20, 30, 40, 50, 60, 70, 80});
  for (int64_t i = 0; i < rows * out_f; ++i)
    CHECK(out[i] == doctest::Approx(expected[i]).epsilon(0.001));
}

TEST_CASE("runtime lora delta: zero lora weights leave output unchanged") {
  const int64_t rows = 2, in_f = 3, rank = 2, out_f = 4;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  std::vector<float> a_buf = {1, 2, 3, 4, 5, 6};
  vt::Tensor a = F32Tensor(a_buf, {rows, in_f});

  std::vector<float> la_buf(rank * in_f, 0.0f);
  std::vector<float> lb_buf(out_f * rank, 0.0f);
  vt::Tensor lora_a = F32Tensor(la_buf, {rank, in_f});
  vt::Tensor lora_b = F32Tensor(lb_buf, {out_f, rank});

  vllm::DitRuntimeLoraLayer layer;
  layer.lora_a = lora_a;
  layer.lora_b = lora_b;
  layer.strength = 1.0f;

  std::vector<float> out = {10, 20, 30, 40, 50, 60, 70, 80};
  vllm::DitApplyRuntimeLoraDelta(q, a, out.data(), rows, out_f, &layer);

  // delta is zero, so out is unchanged
  CHECK(out[0] == doctest::Approx(10));
  CHECK(out[3] == doctest::Approx(40));
  CHECK(out[7] == doctest::Approx(80));
}

TEST_CASE("runtime lora delta: strength folded into lora_b halves delta") {
  const int64_t rows = 2, in_f = 3, rank = 2, out_f = 4;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  std::vector<float> a_buf = {1, 2, 3, 4, 5, 6};
  vt::Tensor a = F32Tensor(a_buf, {rows, in_f});

  std::vector<float> la_buf = {1, 0, 0, 0, 1, 0};
  // Pre-multiply lora_b by 0.5 to simulate strength folding
  std::vector<float> lb_buf = {0.5f, 0, 0, 0.5f, 0.5f, 0.5f, 0, 0};
  vt::Tensor lora_a = F32Tensor(la_buf, {rank, in_f});
  vt::Tensor lora_b = F32Tensor(lb_buf, {out_f, rank});

  vllm::DitRuntimeLoraLayer layer;
  layer.lora_a = lora_a;
  layer.lora_b = lora_b;
  layer.strength = 0.5f;

  std::vector<float> out = {10, 20, 30, 40, 50, 60, 70, 80};
  vllm::DitApplyRuntimeLoraDelta(q, a, out.data(), rows, out_f, &layer);

  // Expected: base + 0.5 * (full delta with unscaled b)
  const std::vector<float> full_lb = {1, 0, 0, 1, 1, 1, 0, 0};
  const std::vector<float> expected = ExpectedDeltaAdded(
      rows, in_f, rank, out_f, a_buf, la_buf, full_lb,
      {10, 20, 30, 40, 50, 60, 70, 80});
  // Since lora_b was pre-multiplied by 0.5, the delta is halved
  for (int64_t i = 0; i < rows * out_f; ++i) {
    const float base_val = (i < 4) ? (10 + i * 10) : (50 + (i - 4) * 10);
    const float half_delta = (expected[i] - base_val) * 0.5f;
    CHECK(out[i] == doctest::Approx(base_val + half_delta).epsilon(0.001));
  }
}

// ── DitLoadRuntimeLoras (safetensors loading) ──────────────────────────────

namespace {

struct LoraEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<float> values;
};

std::string TempStPath(const char* tag) {
  static int counter = 0;
  return std::string("/tmp/vllm_rt_lora_") + tag + "_" +
         std::to_string(getpid()) + "_" + std::to_string(++counter) +
         ".safetensors";
}

void WriteStFile(const std::vector<LoraEntry>& entries,
                 const std::map<std::string, std::string>& metadata,
                 const std::string& path) {
  std::string header = "{";
  bool first = true;
  if (!metadata.empty()) {
    header += "\"__metadata__\":{";
    bool mfirst = true;
    for (const auto& kv : metadata) {
      if (!mfirst) header += ",";
      mfirst = false;
      header += "\"" + kv.first + "\":\"" + kv.second + "\"";
    }
    header += "}";
    first = false;
  }
  std::string payload;
  for (const LoraEntry& e : entries) {
    const size_t at = payload.size();
    size_t bytes = 0;
    if (e.dtype == "F32") {
      bytes = e.values.size() * sizeof(float);
      payload.resize(at + bytes);
      std::memcpy(&payload[at], e.values.data(), bytes);
    } else if (e.dtype == "BF16") {
      bytes = e.values.size() * sizeof(uint16_t);
      for (const float v : e.values) {
        const uint16_t b = vt::F32ToBF16(v);
        payload.append(reinterpret_cast<const char*>(&b), sizeof(b));
      }
    } else if (e.dtype == "F16") {
      bytes = e.values.size() * sizeof(uint16_t);
      for (const float v : e.values) {
        const uint16_t h = vt::F32ToF16(v);
        payload.append(reinterpret_cast<const char*>(&h), sizeof(h));
      }
    }
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      header += (i != 0 ? "," : "") + std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(at) + "," +
              std::to_string(at + bytes) + "]}";
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  const uint64_t n = header.size();
  std::string file(reinterpret_cast<const char*>(&n), sizeof(n));
  file += header;
  file += payload;
  FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  std::fwrite(file.data(), 1, file.size(), f);
  std::fclose(f);
}

// Write a rank-`rank` adapter for `module` (contract target = module + ".weight").
// A [rank, in], B [out, rank]. Returns the file path.
std::string WriteAdapter(int64_t out_features, int64_t rank,
                         int64_t in_features,
                         const std::vector<float>& b,
                         const std::vector<float>& a,
                         const std::map<std::string, std::string>& metadata = {},
                         const std::string& module = "blocks.0.attn.qkv_proj",
                         const std::string& prefix = "diffusion_model.",
                         const std::string& dtype = "F32") {
  const std::string path = TempStPath("adapter");
  WriteStFile(
      {{prefix + module + ".lora_A.weight", dtype, {rank, in_features}, a},
       {prefix + module + ".lora_B.weight", dtype, {out_features, rank}, b}},
      metadata, path);
  return path;
}

vt::Device CpuDev() { return vt::Device{vt::DeviceType::kCPU, 0}; }

const std::vector<std::string> kContract = {"blocks.0.attn.qkv_proj.weight"};
const std::vector<std::string> kPrefixes = {"diffusion_model."};

}  // namespace

TEST_CASE("runtime lora load: empty specs returns empty state") {
  auto state = vllm::DitLoadRuntimeLoras({}, kContract, kPrefixes, CpuDev());
  REQUIRE(state.empty());
}

TEST_CASE("runtime lora load: loads F32 adapter into state") {
  const int64_t out_f = 4, rank = 2, in_f = 3;
  std::vector<float> a_vals = {1, 0, 0, 0, 1, 0};
  std::vector<float> b_vals = {1, 0, 0, 1, 1, 1, 0, 0};
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals);

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  REQUIRE(layer->lora_a.shape[0] == rank);
  REQUIRE(layer->lora_a.shape[1] == in_f);
  REQUIRE(layer->lora_b.shape[0] == out_f);
  REQUIRE(layer->lora_b.shape[1] == rank);
  REQUIRE(layer->strength == doctest::Approx(1.0f));
}

TEST_CASE("runtime lora load: BF16 dtype factors are read") {
  const int64_t out_f = 2, rank = 1, in_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::vector<float> b_vals = {0.5f, 0.25f};
  std::string path =
      WriteAdapter(out_f, rank, in_f, b_vals, a_vals, {}, "blocks.0.attn.qkv_proj",
                   "diffusion_model.", "BF16");

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  // Verify the A factor values were correctly converted from BF16
  const float* a_data = reinterpret_cast<const float*>(layer->lora_a.data);
  CHECK(a_data[0] == doctest::Approx(1.0f).epsilon(0.01));
  CHECK(a_data[1] == doctest::Approx(2.0f).epsilon(0.01));
}

TEST_CASE("runtime lora load: F16 dtype factors are read") {
  const int64_t out_f = 2, rank = 1, in_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::vector<float> b_vals = {0.5f, 0.25f};
  std::string path =
      WriteAdapter(out_f, rank, in_f, b_vals, a_vals, {}, "blocks.0.attn.qkv_proj",
                   "diffusion_model.", "F16");

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  const float* a_data = reinterpret_cast<const float*>(layer->lora_a.data);
  CHECK(a_data[0] == doctest::Approx(1.0f).epsilon(0.01));
  CHECK(a_data[1] == doctest::Approx(2.0f).epsilon(0.01));
}

TEST_CASE("runtime lora load: alpha/rank is folded into lora_b") {
  const int64_t out_f = 2, rank = 4, in_f = 2;
  std::vector<float> a_vals(rank * in_f, 1.0f);
  std::vector<float> b_vals(out_f * rank, 1.0f);
  // alpha = 8, rank = 4, so scale = alpha/rank = 2.0
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals,
                                  {{"lora_alpha", "8"}});

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  // b should be 1.0 * (8/4) * 1.0 = 2.0
  const float* b_data = reinterpret_cast<const float*>(layer->lora_b.data);
  for (int64_t i = 0; i < out_f * rank; ++i)
    CHECK(b_data[i] == doctest::Approx(2.0f).epsilon(0.001));
}

TEST_CASE("runtime lora load: no alpha defaults alpha=rank (scale=1)") {
  const int64_t out_f = 2, rank = 3, in_f = 2;
  std::vector<float> a_vals(rank * in_f, 1.0f);
  std::vector<float> b_vals(out_f * rank, 1.0f);
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals);

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  // No alpha => alpha = rank => scale = 1.0
  const float* b_data = reinterpret_cast<const float*>(layer->lora_b.data);
  for (int64_t i = 0; i < out_f * rank; ++i)
    CHECK(b_data[i] == doctest::Approx(1.0f).epsilon(0.001));
}

TEST_CASE("runtime lora load: strength is folded into lora_b") {
  const int64_t out_f = 2, rank = 2, in_f = 2;
  std::vector<float> a_vals(rank * in_f, 1.0f);
  std::vector<float> b_vals(out_f * rank, 1.0f);
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals);

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 0.5;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  // b should be 1.0 * 1.0 * 0.5 = 0.5
  const float* b_data = reinterpret_cast<const float*>(layer->lora_b.data);
  for (int64_t i = 0; i < out_f * rank; ++i)
    CHECK(b_data[i] == doctest::Approx(0.5f).epsilon(0.001));
}

TEST_CASE("runtime lora load: alpha and strength both fold into lora_b") {
  const int64_t out_f = 2, rank = 4, in_f = 2;
  std::vector<float> a_vals(rank * in_f, 1.0f);
  std::vector<float> b_vals(out_f * rank, 1.0f);
  // alpha = 8, rank = 4 => alpha/rank = 2.0; strength = 0.5
  // scale = 2.0 * 0.5 = 1.0
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals,
                                  {{"lora_alpha", "8"}});

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 0.5;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);
  const float* b_data = reinterpret_cast<const float*>(layer->lora_b.data);
  for (int64_t i = 0; i < out_f * rank; ++i)
    CHECK(b_data[i] == doctest::Approx(1.0f).epsilon(0.001));
}

TEST_CASE("runtime lora load: loaded delta matches hand-computed value") {
  const int64_t rows = 2, out_f = 4, rank = 2, in_f = 3;
  std::vector<float> a_vals = {1, 0, 0, 0, 1, 0};
  std::vector<float> b_vals = {1, 0, 0, 1, 1, 1, 0, 0};
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals);

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());

  const auto* layer = state.Find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(layer != nullptr);

  // Apply the delta and compare with ExpectedDeltaAdded.
  std::vector<float> a_buf = {1, 2, 3, 4, 5, 6};
  vt::Tensor a = F32Tensor(a_buf, {rows, in_f});
  std::vector<float> out = {10, 20, 30, 40, 50, 60, 70, 80};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::DitApplyRuntimeLoraDelta(q, a, out.data(), rows, out_f, layer);

  const std::vector<float> expected = ExpectedDeltaAdded(
      rows, in_f, rank, out_f, a_buf, a_vals, b_vals,
      {10, 20, 30, 40, 50, 60, 70, 80});
  for (int64_t i = 0; i < rows * out_f; ++i)
    CHECK(out[i] == doctest::Approx(expected[i]).epsilon(0.001));
}

TEST_CASE("runtime lora load: unknown contract target is refused") {
  const int64_t out_f = 2, rank = 1, in_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::vector<float> b_vals = {0.5f, 0.25f};
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals, {},
                                  "blocks.0.mlp.fc1");

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  std::string err = Caught([&] {
    vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  });
  REQUIRE(Mentions(err, "contract does not bind"));
}

TEST_CASE("runtime lora load: duplicate adapter targeting same layer is refused") {
  const int64_t out_f = 2, rank = 1, in_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::vector<float> b_vals = {0.5f, 0.25f};
  std::string path1 = WriteAdapter(out_f, rank, in_f, b_vals, a_vals);
  std::string path2 = WriteAdapter(out_f, rank, in_f, b_vals, a_vals);

  std::vector<vllm::DitRuntimeLoraSpec> specs(2);
  specs[0].path = path1;
  specs[0].strength = 1.0;
  specs[1].path = path2;
  specs[1].strength = 1.0;
  std::string err = Caught([&] {
    vllm::DitLoadRuntimeLoras(specs, kContract, kPrefixes, CpuDev());
  });
  REQUIRE(Mentions(err, "already bound"));
}

TEST_CASE("runtime lora load: missing B factor is refused") {
  const int64_t rank = 1, in_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::string path = TempStPath("a_only");
  WriteStFile({{"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "F32",
                {rank, in_f}, a_vals}},
              {}, path);

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  std::string err = Caught([&] {
    vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  });
  REQUIRE(Mentions(err, "no matching B factor"));
}

TEST_CASE("runtime lora load: non-LoRA file is refused") {
  std::string path = TempStPath("not_lora");
  WriteStFile({{"some_random_tensor.weight", "F32", {2, 2}, {1, 2, 3, 4}},
               {"another.weight", "F32", {2, 2}, {5, 6, 7, 8}}},
              {}, path);

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  std::string err = Caught([&] {
    vllm::DitLoadRuntimeLoras({spec}, kContract, kPrefixes, CpuDev());
  });
  REQUIRE(Mentions(err, "not a LoRA adapter"));
}

TEST_CASE("runtime lora load: two adapters targeting different layers load") {
  const std::vector<std::string> contract = {
      "blocks.0.attn.qkv_proj.weight", "blocks.0.attn.out_proj.weight"};
  const int64_t rank = 1, in_f = 2, out_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::vector<float> b_vals = {0.5f, 0.25f};
  std::string path1 = WriteAdapter(out_f, rank, in_f, b_vals, a_vals, {},
                                   "blocks.0.attn.qkv_proj");
  std::string path2 = WriteAdapter(out_f, rank, in_f, b_vals, a_vals, {},
                                   "blocks.0.attn.out_proj");

  std::vector<vllm::DitRuntimeLoraSpec> specs(2);
  specs[0].path = path1;
  specs[0].strength = 1.0;
  specs[1].path = path2;
  specs[1].strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras(specs, contract, kPrefixes, CpuDev());
  REQUIRE(!state.empty());
  REQUIRE(state.Find("blocks.0.attn.qkv_proj.weight") != nullptr);
  REQUIRE(state.Find("blocks.0.attn.out_proj.weight") != nullptr);
}

TEST_CASE("runtime lora load: prefix is stripped from tensor names") {
  const int64_t out_f = 2, rank = 1, in_f = 2;
  std::vector<float> a_vals = {1.0f, 2.0f};
  std::vector<float> b_vals = {0.5f, 0.25f};
  // Write with a different prefix that must be stripped
  std::string path = WriteAdapter(out_f, rank, in_f, b_vals, a_vals, {},
                                  "blocks.0.attn.qkv_proj", "model.diffusion_model.");
  const std::vector<std::string> prefixes = {"model.diffusion_model.", "diffusion_model."};

  vllm::DitRuntimeLoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  auto state = vllm::DitLoadRuntimeLoras({spec}, kContract, prefixes, CpuDev());
  REQUIRE(!state.empty());
  REQUIRE(state.Find("blocks.0.attn.qkv_proj.weight") != nullptr);
}
