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

#include <cstdlib>
#include <memory>
#include <string>

#include "vllm/config/multimodal.h"  // #607 L3 MultiModalConfig
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
//
// `limits` is the engine's multimodal configuration, defaulted, so every case
// that predates #607 L3 passes the pre-L3 defaults unchanged.
std::string LoadWithMmproj(const std::string& model_path,
                           const std::string& mmproj_path,
                           const vllm::MultiModalConfig& limits = {}) {
  vllm::entrypoints::EngineParams params;
  params.mmproj_path = mmproj_path;
  params.multimodal = limits;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(model_path, params);
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

// `--language-model-only`, as the serve flag and the C ABI set it
// (multimodal.py:78-80).
vllm::MultiModalConfig LanguageModelOnly() {
  vllm::MultiModalConfig c;
  c.language_model_only = true;
  return c;
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

// ── THE TOWER SKIP, on the THIRD tower-load call site (#607 L3) ─────────────
//
// `--mmproj` is a third production tower load, and the first cut of the L3 row
// gated the other two and missed this one: `--model x.gguf --mmproj p.gguf
// --language-model-only` zeroed every limit, refused every image request, and
// still read the whole projector — the exact L2 failure L3 exists to close.
//
// HOW THESE CASES SEE IT. The half-patch projector above is refused FROM INSIDE
// `LoadQwen3VLVisionFromClipMmproj` (clip_mmproj_gguf.cpp:237): the reader has
// to walk to the patch embedding to say "inventing". So that message is a
// receipt that the read HAPPENED, and the tokenizer stop one step later is a
// receipt that it did not. That is the whole A/B, and it needs no complete
// engine and no NAS file.
//
// THE MUTATION THESE CASES ANSWER TO. Delete the `SkipTowerForModalities` guard
// around the read in `model_loader.cpp` — or replace the predicate with
// `false` — and the zero-limit case below sees "inventing" again, which is red.
TEST_CASE("mmproj reach: zero limits leave the projector CONSTRUCTED but UNREAD") {
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.omit_patch_embd_1 = true;  // refused from INSIDE the reader, at :237
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  // A, the default limits: the reader runs, walks into the file, and refuses.
  // Restated here rather than borrowed from the case above so the two halves of
  // this comparison are the same call with one argument changed.
  const std::string read = LoadWithMmproj(model.path(), mmproj.path());
  CAPTURE(read);
  CHECK(read.find("inventing") != std::string::npos);

  // B, `--language-model-only`: the SAME broken file, and the loader walks past
  // it to the tokenizer, because nothing ever opened the patch embedding.
  const std::string skipped =
      LoadWithMmproj(model.path(), mmproj.path(), LanguageModelOnly());
  REQUIRE_FALSE(skipped.empty());
  CAPTURE(skipped);
  CHECK(skipped.find("inventing") == std::string::npos);
  CHECK(skipped.find("v.patch_embd.weight.1") == std::string::npos);
  CHECK(skipped.find(kTokenizerStop) != std::string::npos);

  // The other route to zero, which upstream treats identically because
  // `_mark_tower_model` never mentions the flag (interfaces.py:293).
  vllm::MultiModalConfig zeros;
  zeros.limit_per_prompt = {{"image", 0}, {"video", 0}};
  const std::string by_limits = LoadWithMmproj(model.path(), mmproj.path(), zeros);
  CAPTURE(by_limits);
  CHECK(by_limits.find(kTokenizerStop) != std::string::npos);
}

TEST_CASE("mmproj reach: ALL, not ANY — one zero modality still reads the projector") {
  // The projector IS the Qwen3-VL tower, marked `{"image", "video"}` upstream
  // (qwen3_vl.py:1747), so a load that can still serve video must still load it.
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.omit_patch_embd_1 = true;
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  vllm::MultiModalConfig image_only;
  image_only.limit_per_prompt = {{"image", 0}};
  const std::string message =
      LoadWithMmproj(model.path(), mmproj.path(), image_only);
  CAPTURE(message);
  CHECK(message.find("inventing") != std::string::npos);
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

TEST_CASE("mmproj reach: zero limits do NOT silence the flag's own refusals") {
  // Construct-without-initialise stops the STORAGE, not the construction
  // (utils.py:762). A `--mmproj` this build cannot use is still a flag the user
  // typed and cannot use, so it must still cost a message rather than be
  // accepted in silence because the limits happened to be zero.
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.architecture = "qwen35";  // a language GGUF handed to --mmproj
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  const std::string message =
      LoadWithMmproj(model.path(), mmproj.path(), LanguageModelOnly());
  CAPTURE(message);
  CHECK(message.find("--mmproj") != std::string::npos);
  CHECK(message.find("general.architecture") != std::string::npos);
  CHECK(message.find(kTokenizerStop) == std::string::npos);

  // ... and the same for a projector this build has no reader for.
  clip_fixture::Options muse;
  muse.projector_type = "muse-glimmer";
  TempFile muse_mmproj(clip_fixture::Build(clip_fixture::Dims{}, muse));
  const std::string other =
      LoadWithMmproj(model.path(), muse_mmproj.path(), LanguageModelOnly());
  CAPTURE(other);
  CHECK(other.find("MuseGlimmer GGUF") != std::string::npos);
  CHECK(other.find(kTokenizerStop) == std::string::npos);
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

// ── The ENGINE HOLDS IT ─────────────────────────────────────────────────────
//
// Every case above stops at a MESSAGE. The permitting one throws at the
// tokenizer, one step past the projector, so no `LoadedEngine` is ever
// constructed and nothing above observes the tower being HELD. That left the
// positive claim four records make — `LoadedEngine::vision_tower()` holds the
// loaded tower — measured by nothing: deleting `vision_tower_(std::move(
// vision_tower))` from the constructor kept every gate in this tree green.
//
// This case is that missing observation, and it needs a load that COMPLETES,
// which needs a real language GGUF: the synthetic fixture above carries no
// tokenizer and no backbone weights on purpose. So it is env-gated on BOTH
// paths and SKIPS LOUDLY when either is unset — CI reads no NAS file, exactly
// as `test_clip_mmproj_gguf`'s live case does:
//
//   VLLM_CPP_QWEN38_27B_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf
//   VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf
//   ./build/tests/test_gguf_mmproj_reach
//
// Both artifacts are pinned with their bytes and sha256 in `docs/USAGE.md`
// §"The exact files this was gated against" (`unsloth/Qwen3.8-27B-GGUF` @
// `fe1e2a23d973adb629709749dc4f6756df66ef10`).
//
// The geometry below is the PROJECTOR's, restated from that file's own header
// rather than from the reader, and every size is derived from it here rather
// than read back off the same config the engine reports — so a tower whose
// weights do not match the geometry the engine publishes is a disagreement
// between two sources instead of a tautology.
TEST_CASE("mmproj reach: a load that COMPLETES leaves the tower ON THE ENGINE") {
  const char* model_env = std::getenv("VLLM_CPP_QWEN38_27B_GGUF");
  const char* mmproj_env = std::getenv("VLLM_CPP_QWEN38_27B_MMPROJ");
  if (model_env == nullptr || mmproj_env == nullptr) {
    MESSAGE(
        "SKIPPED: set VLLM_CPP_QWEN38_27B_GGUF to Qwen3.8-27B-Q4_K_M.gguf AND "
        "VLLM_CPP_QWEN38_27B_MMPROJ to mmproj-BF16.gguf to observe the loaded "
        "tower being HELD by the engine (docs/USAGE.md pins both files)");
    return;
  }

  vllm::entrypoints::EngineParams params;
  params.mmproj_path = mmproj_env;
  // The smallest KV pool the engine will build. This case is about what the
  // load LEAVES on the engine, not about serving, and a 27B pool is memory
  // this observation does not need.
  params.num_blocks = 8;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
      vllm::entrypoints::LoadedEngine::FromModelDir(model_env, params);
  REQUIRE(engine != nullptr);

  // THE CLAIM. Not "the reader returned a tower" — the four records say the
  // ENGINE holds it, and this is the only line in the tree that reads it back
  // off a `LoadedEngine` a production entry point built.
  const vllm::multimodal::Qwen3VLVisionWeights* tower = engine->vision_tower();
  REQUIRE(tower != nullptr);

  // The geometry the engine publishes beside it, which `vision_tower()`'s own
  // contract says is meaningless without the tower.
  const vllm::multimodal::Qwen3VLVisionConfig& cfg = engine->vision_config();
  CHECK(cfg.hidden_size == 1152);
  CHECK(cfg.num_heads == 16);
  CHECK(cfg.depth == 27);
  CHECK(cfg.intermediate_size == 4304);
  CHECK(cfg.out_hidden_size == 5120);
  CHECK(cfg.patch_size == 16);
  CHECK(cfg.temporal_patch_size == 2);
  CHECK(cfg.spatial_merge_size == 2);
  CHECK(cfg.in_channels == 3);
  CHECK(cfg.num_position_embeddings == 2304);
  CHECK(cfg.deepstack_visual_indexes.empty());

  // ── The WEIGHTS match that geometry ──────────────────────────────────────
  // Sizes computed from the numbers above, so a tower that is present but is
  // some OTHER file's fails here rather than passing on non-nullness.
  const size_t hidden = 1152, depth = 27, ffn = 4304, out_hidden = 5120;
  const size_t patch_in = 3 * 2 * 16 * 16;  // C * temporal * p * p = 1536
  const size_t merged = hidden * 2 * 2;     // hidden * spatial_merge^2 = 4608

  CHECK(tower->patch_proj_w.size() == hidden * patch_in);
  CHECK(tower->patch_proj_b.size() == hidden);
  CHECK(tower->pos_embed_w.size() == 2304 * hidden);
  REQUIRE(tower->blocks.size() == depth);
  for (size_t l = 0; l < depth; ++l) {
    CAPTURE(l);
    const vllm::multimodal::VisionBlockWeights& b = tower->blocks[l];
    CHECK(b.norm1_w.size() == hidden);
    CHECK(b.norm2_w.size() == hidden);
    CHECK(b.qkv_w.size() == 3 * hidden * hidden);
    CHECK(b.qkv_b.size() == 3 * hidden);
    CHECK(b.proj_w.size() == hidden * hidden);
    CHECK(b.fc1_w.size() == ffn * hidden);
    CHECK(b.fc2_w.size() == hidden * ffn);
  }
  // The merger, whose fc1/fc2 are the `mm.0`/`mm.2` pair: `v.post_ln` is the
  // PRE-shuffle norm (1152 wide, not the merged 4608), which is the file
  // saying use_postshuffle_norm is false rather than the reader assuming it.
  CHECK_FALSE(tower->merger.use_postshuffle_norm);
  CHECK(tower->merger.norm_w.size() == hidden);
  CHECK(tower->merger.fc1_w.size() == merged * merged);
  CHECK(tower->merger.fc2_w.size() == out_hidden * merged);
  CHECK(tower->merger.fc2_b.size() == out_hidden);
  CHECK(tower->deepstack_mergers.empty());

  // Not all zeros: a default-constructed tower of the right SHAPE would pass
  // every size check above.
  double sum = 0.0;
  for (size_t i = 0; i < 4096 && i < tower->merger.fc2_w.size(); ++i) {
    sum += static_cast<double>(tower->merger.fc2_w[i]) *
           static_cast<double>(tower->merger.fc2_w[i]);
  }
  CHECK(sum > 0.0);
}

// The same observation on the OTHER arm: what a completed load leaves on the
// engine when every modality the projector serves is at limit 0.
//
// ENV-GATED, for the reason the case above is: the synthetic language GGUF
// carries no tokenizer, so no `LoadedEngine` is ever built from it, and this is
// the only shape in the tree that can read a flag back off a real one. The
// BEHAVIOUR — that the projector's bytes are not read — is gated in CI by the
// message A/B above, which needs no artifact; what is env-gated here is the
// engine's REPORTING of it, and that limitation is recorded under `## Owed` in
// specs/multimodal-track.md rather than left for a reader to discover.
TEST_CASE("mmproj reach: at zero limits a completed load leaves NO tower, and SAYS so") {
  const char* model_env = std::getenv("VLLM_CPP_QWEN38_27B_GGUF");
  const char* mmproj_env = std::getenv("VLLM_CPP_QWEN38_27B_MMPROJ");
  if (model_env == nullptr || mmproj_env == nullptr) {
    MESSAGE(
        "SKIPPED: set VLLM_CPP_QWEN38_27B_GGUF and VLLM_CPP_QWEN38_27B_MMPROJ "
        "to observe --language-model-only leaving the projector UNLOADED on a "
        "real engine (docs/USAGE.md pins both files)");
    return;
  }

  vllm::entrypoints::EngineParams params;
  params.mmproj_path = mmproj_env;
  params.num_blocks = 8;
  params.multimodal.language_model_only = true;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
      vllm::entrypoints::LoadedEngine::FromModelDir(model_env, params);
  REQUIRE(engine != nullptr);

  // Nothing was read...
  CHECK(engine->vision_tower() == nullptr);
  // ... and the engine distinguishes that from "no projector was named", which
  // is the distinction the server's own startup line prints
  // (server_main.cpp, "multimodal towers NOT loaded").
  REQUIRE(engine->skipped_towers().size() == 1);
  CHECK(engine->skipped_towers()[0] == "vision_tower");
  CHECK(engine->mm_config().GetLimitPerPrompt("image") == 0);
}
