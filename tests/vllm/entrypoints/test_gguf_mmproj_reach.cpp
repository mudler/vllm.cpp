// LOAD-GGUF-MMPROJ, issue #821 — the REACHABILITY gate for the second GGUF.
//
// `test_clip_mmproj_gguf.cpp` gates the reader by calling it. That proves the
// reader works and never proves anything reaches it, which is the exact shape
// `.agents/reachability.md` was written for: before this row the ONLY caller of
// `LoadQwen3VLVisionFromGguf` was a test, and the ONLY caller of
// `MuseGlimmerRefuseMmproj` was a test, "because there is no production path
// that accepts a second GGUF path at all".
//
// So this file drives `LoadedEngine::FromModelDir` — the loader entry point
// every consumer (the C ABI, the OpenAI server, the CLI) arrives through — with
// `EngineParams::mmproj_path` set, and asserts the thrown MESSAGE.
//
// HOW THE PERMITTING CASE IS MEANINGFUL. The synthetic language GGUF carries no
// tokenizer, so a load that gets PAST the projector step dies at
// `tokenizer: GGUF missing kv "tokenizer.ggml.model"` — the step immediately
// after it. Asserting that later message POSITIVELY is the point: a case that
// only checked for the absence of a refusal would also pass if the loader had
// died earlier for an unrelated reason.
//
// THE MUTATION THIS FILE ANSWERS TO. Delete the
// `LoadQwen3VLVisionFromClipMmproj` call in `model_loader.cpp` and the
// half-patch case below stops seeing its refusal and sees the tokenizer error
// instead, which is red here.
#include <doctest/doctest.h>

#include <string>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/gguf_builder.h"
#include "vllm/models/clip_mmproj_fixture.h"

namespace {

using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// A synthetic `qwen35moe` language GGUF: exactly the hparams
// `HfConfigFromGguf` and `ModelRegistry::Resolve` need, and no tokenizer, so
// the load stops one step after the projector. Same shape as
// `test_gguf_device_fit_reach.cpp`'s fixture, which measured that stopping
// point.
std::string BuildLanguageGguf() {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35moe"));
  b.AddKv(U32Kv("qwen35moe.embedding_length", 64));
  b.AddKv(U32Kv("qwen35moe.block_count", 2));
  b.AddKv(U32Kv("qwen35moe.attention.head_count", 4));
  b.AddKv(U32Kv("qwen35moe.attention.head_count_kv", 2));
  b.AddKv(U32Kv("qwen35moe.attention.key_length", 16));
  b.AddKv(U32Kv("qwen35moe.expert_count", 4));
  b.AddKv(U32Kv("qwen35moe.expert_used_count", 2));
  b.AddKv(U32Kv("qwen35moe.expert_feed_forward_length", 32));
  b.AddKv(U32Kv("qwen35moe.expert_shared_feed_forward_length", 32));
  b.AddKv(U32Kv("qwen35moe.ssm.group_count", 2));
  b.AddKv(U32Kv("qwen35moe.ssm.time_step_rank", 4));
  b.AddKv(U32Kv("qwen35moe.ssm.state_size", 8));
  b.AddKv(U32Kv("qwen35moe.ssm.conv_kernel", 4));
  b.AddKv(U32Kv("qwen35moe.full_attention_interval", 4));
  b.AddKv(U32Kv("qwen35moe.context_length", 256));
  b.AddKv(gguf_test::F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(gguf_test::F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddTensor("token_embd.weight", {64, 64}, /*ggml_type=*/0,
              std::string(4096 * 4, '\0'));
  return b.Build();
}

// What `FromModelDir` threw for this (language file, projector file) pair, as
// text. A message is the only thing that can say WHICH step refused.
std::string LoadWithMmproj(const std::string& model_path,
                           const std::string& mmproj_path) {
  vllm::entrypoints::EngineParams params;
  params.mmproj_path = mmproj_path;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(model_path, params);
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

constexpr const char* kTokenizerStop = "tokenizer: GGUF missing kv";

}  // namespace

TEST_CASE("mmproj reach: the loader READS a well-formed projector and carries on") {
  TempFile model(BuildLanguageGguf());
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}));

  const std::string message = LoadWithMmproj(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // It got PAST the projector: the throw is the NEXT step's, not the
  // projector's. Every refusal this row adds names `--mmproj`.
  CHECK(message.find(kTokenizerStop) != std::string::npos);
  CHECK(message.find("--mmproj") == std::string::npos);
  CHECK(message.find("clip mmproj gguf") == std::string::npos);
}

TEST_CASE("mmproj reach: a projector missing the temporal half is refused THROUGH the loader") {
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.omit_patch_embd_1 = true;
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  const std::string message = LoadWithMmproj(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // The reader RAN — it had to walk to the patch embedding to say this — and it
  // ran from the production loader, not from a test that constructed it.
  CHECK(message.find("v.patch_embd.weight.1") != std::string::npos);
  CHECK(message.find("inventing") != std::string::npos);
  // And it fired BEFORE the tokenizer, therefore before any language weight
  // I/O: on the real artifacts that is 931 MB read instead of 17 GB mapped.
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

TEST_CASE("mmproj reach: a language GGUF passed to --mmproj is refused THROUGH the loader") {
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.architecture = "qwen35";
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  const std::string message = LoadWithMmproj(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("--mmproj") != std::string::npos);
  CHECK(message.find("general.architecture") != std::string::npos);
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

TEST_CASE("mmproj reach: MuseGlimmer's recorded refusal now has a PRODUCTION caller") {
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.projector_type = "muse-glimmer";
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  const std::string message = LoadWithMmproj(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // Landing this row CHANGES MuseGlimmer's behaviour: its mmproj refusal stops
  // being a function only `tests/vllm/models/test_muse_glimmer_gguf.cpp` calls
  // and becomes what a user holding `mmproj-kquant.gguf` is told.
  CHECK(message.find("MuseGlimmer GGUF") != std::string::npos);
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

TEST_CASE("mmproj reach: --mmproj on a non-GGUF model path is refused, not ignored") {
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}));

  // A safetensors checkpoint carries its tower in its own shards. Accepting the
  // flag and silently dropping it would load a tower the user did not name.
  const std::string message =
      LoadWithMmproj("/nonexistent/model/dir", mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("--mmproj") != std::string::npos);
  CHECK(message.find(".gguf language file") != std::string::npos);
  // It fires BEFORE the directory probe, so the message is about the flag and
  // not about the missing directory.
  CHECK(message.find("model path is not a directory") == std::string::npos);
}
