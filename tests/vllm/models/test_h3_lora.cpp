// MiniMax-H3 LoRA — H3-specific contract name rewriting and load-time fusion.
//
// Row ROAD-V1-DIT-LORA, .agents/specs/dit-lora-generic.md.
//
// The fusion arithmetic itself is gated in test_ltx2_lora.cpp, because both
// families share the same `DitFuseLoraIntoTensor` / `DitFuseLorasIntoBuffer`
// implementation. This file gates the H3-specific parts:
//
//   * The ComfyUI prefix set {"model.diffusion_model.", "diffusion_model."}:
//     a ComfyUI adapter carries `model.diffusion_model.blocks.0.attn.qkv_proj`,
//     and the H3 contract binds `blocks.0.attn.qkv_proj.weight`. The rewrite
//     must strip BOTH prefixes.
//
//   * The end-to-end path through the generic helpers: `DitOpenLoras` with H3
//     prefixes and contract names, `DitFuseLorasIntoBuffer`, and
//     `DitCheckLorasWereApplied`.
//
//   * The extras seam: `ResolveDitLoraSpecs` correctly parses
//     `lora_path` / `lora_strength` pairs that H3's video engine passes.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/dit_lora.h"
#include "vt/dtype.h"

namespace {

// ── a minimal safetensors writer (same as test_ltx2_lora) ────────────────────

struct LoraEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<float> values;
};

std::string TempPath(const char* tag) {
  static int counter = 0;
  return std::string("/tmp/h3_lora_test_") + tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(++counter) + ".safetensors";
}

void WriteLoraFile(const std::vector<LoraEntry>& entries,
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

// H3 contract names (from EnumerateMiniMaxH3DitTensors).
const char* const kQkvTarget = "blocks.0.attn.qkv_proj.weight";
const char* const kQkvModule = "blocks.0.attn.qkv_proj";
const char* const kOutProjTarget = "blocks.0.attn.out_proj.weight";
const char* const kOutProjModule = "blocks.0.attn.out_proj";

// The H3 ComfyUI prefix set: both must be stripped.
const std::vector<std::string> kH3Prefixes = {"model.diffusion_model.", "diffusion_model."};

// Write a rank-`rank` adapter for `module`, with `b` [out, rank] and `a`
// [rank, in] given row-major. The `prefix` controls the ComfyUI naming.
std::string WriteAdapter(int64_t out_features, int64_t rank, int64_t in_features,
                         const std::vector<float>& b, const std::vector<float>& a,
                         const std::string& module = kQkvModule,
                         const std::string& prefix = "model.diffusion_model.",
                         const std::map<std::string, std::string>& metadata = {}) {
  const std::string path = TempPath("adapter");
  WriteLoraFile(
      {
          {prefix + module + ".lora_A.weight", "BF16", {rank, in_features}, a},
          {prefix + module + ".lora_B.weight", "BF16", {out_features, rank}, b},
      },
      metadata, path);
  return path;
}

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

// A simple identity-ish weight matrix: w[i][j] = i * cols + j.
std::vector<float> IdentityWeight(int64_t rows, int64_t cols) {
  std::vector<float> w(rows * cols);
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      w[i * cols + j] = static_cast<float>(i * cols + j);
  return w;
}

// Compute the expected fused result: W + (B * strength) @ A, in bf16.
std::vector<float> ExpectedFused(int64_t rows, int64_t rank, int64_t cols,
                                 const std::vector<float>& w,
                                 const std::vector<float>& b,
                                 const std::vector<float>& a,
                                 float strength) {
  std::vector<float> out(rows * cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      float delta = 0.0f;
      for (int64_t r = 0; r < rank; ++r) {
        // b is [rows, rank], a is [rank, cols]
        const float bv = vt::BF16ToF32(vt::F32ToBF16(b[i * rank + r] * strength));
        const float av = vt::BF16ToF32(vt::F32ToBF16(a[r * cols + j]));
        delta += bv * av;
      }
      out[i * cols + j] = w[i * cols + j] + delta;
    }
  }
  // The store is bf16
  for (float& v : out) v = vt::BF16ToF32(vt::F32ToBF16(v));
  return out;
}

}  // namespace

// ── contract name rewriting: both prefixes stripped ──────────────────────────

TEST_CASE("h3 lora: model.diffusion_model. prefix is stripped") {
  const int64_t out_f = 6, rank = 2, in_f = 4;
  const std::vector<float> b(out_f * rank, 0.5f);
  const std::vector<float> a(rank * in_f, 0.25f);
  const std::string path = WriteAdapter(out_f, rank, in_f, b, a, kQkvModule,
                                        "model.diffusion_model.");

  vllm::DitLoraSpec spec{path, 1.0};
  std::vector<std::string> contract = {kQkvTarget};
  const std::vector<vllm::DitLoraAdapter> adapters =
      vllm::DitOpenLoras({spec}, contract, kH3Prefixes);
  REQUIRE(adapters.size() == 1);
  REQUIRE(adapters[0].Find(kQkvTarget) != nullptr);
  std::remove(path.c_str());
}

TEST_CASE("h3 lora: diffusion_model. prefix is stripped") {
  const int64_t out_f = 6, rank = 2, in_f = 4;
  const std::vector<float> b(out_f * rank, 0.5f);
  const std::vector<float> a(rank * in_f, 0.25f);
  const std::string path = WriteAdapter(out_f, rank, in_f, b, a, kQkvModule,
                                        "diffusion_model.");

  vllm::DitLoraSpec spec{path, 1.0};
  std::vector<std::string> contract = {kQkvTarget};
  const std::vector<vllm::DitLoraAdapter> adapters =
      vllm::DitOpenLoras({spec}, contract, kH3Prefixes);
  REQUIRE(adapters.size() == 1);
  REQUIRE(adapters[0].Find(kQkvTarget) != nullptr);
  std::remove(path.c_str());
}

TEST_CASE("h3 lora: bare contract name (no prefix) is accepted") {
  const int64_t out_f = 6, rank = 2, in_f = 4;
  const std::vector<float> b(out_f * rank, 0.5f);
  const std::vector<float> a(rank * in_f, 0.25f);
  // No prefix at all — a community checkpoint that already carries bare names.
  const std::string path = WriteAdapter(out_f, rank, in_f, b, a, kQkvModule, "");

  vllm::DitLoraSpec spec{path, 1.0};
  std::vector<std::string> contract = {kQkvTarget};
  const std::vector<vllm::DitLoraAdapter> adapters =
      vllm::DitOpenLoras({spec}, contract, kH3Prefixes);
  REQUIRE(adapters.size() == 1);
  REQUIRE(adapters[0].Find(kQkvTarget) != nullptr);
  std::remove(path.c_str());
}

TEST_CASE("h3 lora: adapter targeting a tensor not in the contract is refused") {
  const int64_t out_f = 6, rank = 2, in_f = 4;
  const std::vector<float> b(out_f * rank, 0.5f);
  const std::vector<float> a(rank * in_f, 0.25f);
  const std::string path = WriteAdapter(out_f, rank, in_f, b, a, "blocks.0.bogus.module");

  vllm::DitLoraSpec spec{path, 1.0};
  std::vector<std::string> contract = {kQkvTarget};
  const std::string err = Caught([&] {
    vllm::DitOpenLoras({spec}, contract, kH3Prefixes);
  });
  REQUIRE(Mentions(err, "does not bind"));
  std::remove(path.c_str());
}

// ── fusion through the generic helpers ───────────────────────────────────────

TEST_CASE("h3 lora: fusion into a bf16 buffer matches W + (B*strength)@A") {
  const int64_t out_f = 6, rank = 2, in_f = 4;
  const float strength = 0.75f;
  std::vector<float> b(out_f * rank);
  std::vector<float> a(rank * in_f);
  for (int64_t i = 0; i < out_f * rank; ++i) b[i] = static_cast<float>(i + 1) * 0.1f;
  for (int64_t i = 0; i < rank * in_f; ++i) a[i] = static_cast<float>(i + 1) * 0.2f;

  const std::string path = WriteAdapter(out_f, rank, in_f, b, a);
  vllm::DitLoraSpec spec{path, strength};
  std::vector<std::string> contract = {kQkvTarget};
  const std::vector<vllm::DitLoraAdapter> adapters =
      vllm::DitOpenLoras({spec}, contract, kH3Prefixes);
  REQUIRE(adapters.size() == 1);

  const std::vector<float> weight = IdentityWeight(out_f, in_f);
  std::vector<uint16_t> buffer(weight.size());
  for (size_t i = 0; i < weight.size(); ++i) buffer[i] = vt::F32ToBF16(weight[i]);

  const bool fused = vllm::DitFuseLorasIntoBuffer(
      adapters, kQkvTarget, {out_f, in_f}, vt::DType::kBF16,
      reinterpret_cast<uint8_t*>(buffer.data()), buffer.size() * sizeof(uint16_t));
  REQUIRE(fused);

  const std::vector<float> expected = ExpectedFused(out_f, rank, in_f, weight, b, a, strength);
  for (int64_t i = 0; i < out_f * in_f; ++i) {
    const float got = vt::BF16ToF32(buffer[i]);
    CHECK(got == doctest::Approx(expected[i]).epsilon(0.01));
  }
  std::remove(path.c_str());
}

TEST_CASE("h3 lora: DitCheckLorasWereApplied refuses a LoRA that fused nothing") {
  const int64_t out_f = 6, rank = 2, in_f = 4;
  const std::vector<float> b(out_f * rank, 0.5f);
  const std::vector<float> a(rank * in_f, 0.25f);
  const std::string path = WriteAdapter(out_f, rank, in_f, b, a, kQkvModule);

  vllm::DitLoraSpec spec{path, 1.0};
  std::vector<std::string> contract = {kQkvTarget, kOutProjTarget};
  const std::vector<vllm::DitLoraAdapter> adapters =
      vllm::DitOpenLoras({spec}, contract, kH3Prefixes);
  REQUIRE(adapters.size() == 1);

  // The adapter targets qkv_proj, but we fuse out_proj instead — so fused == 0.
  const std::vector<float> weight(out_f * in_f, 1.0f);
  std::vector<uint16_t> buffer(weight.size());
  for (size_t i = 0; i < weight.size(); ++i) buffer[i] = vt::F32ToBF16(weight[i]);

  int64_t fused = 0;
  const bool did = vllm::DitFuseLorasIntoBuffer(
      adapters, kOutProjTarget, {out_f, in_f}, vt::DType::kBF16,
      reinterpret_cast<uint8_t*>(buffer.data()), buffer.size() * sizeof(uint16_t));
  // out_proj is a DIFFERENT tensor, so the adapter's qkv_proj pair does not apply.
  CHECK_FALSE(did);
  CHECK(fused == 0);

  const std::string err = Caught([&] {
    vllm::DitCheckLorasWereApplied(adapters, fused);
  });
  REQUIRE(Mentions(err, "fused into ZERO"));
  std::remove(path.c_str());
}

// ── extras resolution ─────────────────────────────────────────────────────────

TEST_CASE("h3 lora: ResolveDitLoraSpecs parses indexed lora_path/strength") {
  const std::string p1 = "/tmp/fake_adapter_1.safetensors";
  const std::string p2 = "/tmp/fake_adapter_2.safetensors";
  std::map<std::string, std::string> extras;
  extras["lora_path"] = p1;
  extras["lora_strength"] = "0.5";
  extras["lora_path_2"] = p2;
  extras["lora_strength_2"] = "1.5";

  const std::vector<vllm::DitLoraSpec> specs = vllm::ResolveDitLoraSpecs(extras);
  REQUIRE(specs.size() == 2);
  CHECK(specs[0].path == p1);
  CHECK(specs[0].strength == doctest::Approx(0.5));
  CHECK(specs[1].path == p2);
  CHECK(specs[1].strength == doctest::Approx(1.5));
}

TEST_CASE("h3 lora: ResolveDitLoraSpecs defaults strength to 1.0") {
  std::map<std::string, std::string> extras;
  extras["lora_path"] = "/tmp/fake.safetensors";
  const std::vector<vllm::DitLoraSpec> specs = vllm::ResolveDitLoraSpecs(extras);
  REQUIRE(specs.size() == 1);
  CHECK(specs[0].path == "/tmp/fake.safetensors");
  CHECK(specs[0].strength == doctest::Approx(1.0));
}

TEST_CASE("h3 lora: ResolveDitLoraSpecs refuses a gap in the index sequence") {
  std::map<std::string, std::string> extras;
  extras["lora_path"] = "/tmp/fake_1.safetensors";
  extras["lora_path_3"] = "/tmp/fake_3.safetensors";  // skips _2
  const std::string err = Caught([&] {
    vllm::ResolveDitLoraSpecs(extras);
  });
  REQUIRE(Mentions(err, "gap"));
}

TEST_CASE("h3 lora: IsDitLoraIndexedExtra recognizes the indexed keys") {
  CHECK(vllm::IsDitLoraIndexedExtra("lora_path_2"));
  CHECK(vllm::IsDitLoraIndexedExtra("lora_strength_3"));
  CHECK(vllm::IsDitLoraIndexedExtra("lora_path_10"));
  CHECK_FALSE(vllm::IsDitLoraIndexedExtra("lora_path"));
  CHECK_FALSE(vllm::IsDitLoraIndexedExtra("lora_strength"));
  CHECK_FALSE(vllm::IsDitLoraIndexedExtra("partition"));
  CHECK_FALSE(vllm::IsDitLoraIndexedExtra("random_key"));
}

TEST_CASE("h3 lora: IsDitLoraExtra recognizes base and indexed keys") {
  // Base keys (index 1, no suffix) — these are the first adapter.
  CHECK(vllm::IsDitLoraExtra("lora_path"));
  CHECK(vllm::IsDitLoraExtra("lora_strength"));
  // Indexed keys (N >= 2).
  CHECK(vllm::IsDitLoraExtra("lora_path_2"));
  CHECK(vllm::IsDitLoraExtra("lora_strength_3"));
  CHECK(vllm::IsDitLoraExtra("lora_path_10"));
  // Non-LoRA keys.
  CHECK_FALSE(vllm::IsDitLoraExtra("partition"));
  CHECK_FALSE(vllm::IsDitLoraExtra("random_key"));
  CHECK_FALSE(vllm::IsDitLoraExtra("lora_path_1"));  // index 1 uses no suffix
  CHECK_FALSE(vllm::IsDitLoraExtra("lora_path_0"));  // no zero index
}
