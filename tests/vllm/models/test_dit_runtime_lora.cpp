// Unit tests for runtime prompt-activated LoRA (ROAD-V1-LORA-RUNTIME).
// Tests DitParseLoraTags: prompt tag parsing, name resolution, and the
// DitRuntimeLoraState lookup helpers.

#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>

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
