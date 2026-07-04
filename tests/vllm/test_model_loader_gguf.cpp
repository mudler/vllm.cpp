// M0.10 Task 2: LoadedEngine::FromModelDir routes a `.gguf` path to the GGUF
// loader (not the safetensors directory path). A full end-to-end engine build
// from a real GGUF is dgx-pending; here we prove the branch selection: a
// `.gguf` file is opened as GGUF (a bad-magic .gguf surfaces the reader's
// "magic" error, which the directory path can never produce), and a non-.gguf
// non-directory path still takes the directory branch.
#include <doctest/doctest.h>

#include <string>

#include "gguf_builder.h"
#include "vllm/entrypoints/model_loader.h"

using gguf_test::TempFile;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

TEST_CASE("FromModelDir routes .gguf files to the GGUF reader") {
  // A .gguf file with a corrupt magic: the GGUF branch is taken, so the
  // reader's "magic" error surfaces (the directory branch cannot produce it).
  TempFile bad(std::string("GGML") + std::string(60, '\0'));
  CHECK_THROWS_WITH_AS(LoadedEngine::FromModelDir(bad.path(), EngineParams{}),
                       doctest::Contains("magic"), std::runtime_error);
}

TEST_CASE("FromModelDir keeps the directory path for non-.gguf inputs") {
  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir("/nonexistent/model/dir", EngineParams{}),
      doctest::Contains("not a directory"), std::runtime_error);
}
