// Unit tests for runtime prompt-activated LoRA (ROAD-V1-LORA-RUNTIME).
// Tests DitParseLoraTags: prompt tag parsing, name resolution, and the
// DitRuntimeLoraState lookup helpers. Also tests DitApplyRuntimeLoraDelta:
// the CPU-path additive delta computation.

#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit_lora.h"

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
