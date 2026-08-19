// The llama.cpp `clip` mmproj reader (row `LOAD-GGUF-MMPROJ`, issue #821).
//
// This file gates the READER: the `clip.*` metadata -> `Qwen3VLVisionConfig`
// mapping, the `v.*` / `mm.*` -> `Qwen3VLVisionWeights` name mapping, the
// two-half temporal patch-embedding JOIN, and every refusal. It does NOT prove
// anything reaches the reader — that is `test_gguf_mmproj_reach.cpp`, which
// drives `LoadedEngine::FromModelDir`.
//
// Upstream anchors for every name and role are in
// `include/vllm/model_executor/models/clip_mmproj_gguf.h`, at llama.cpp
// `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/clip_mmproj_gguf.h"
#include "vllm/models/clip_mmproj_fixture.h"

namespace {

using clip_fixture::Dims;
using clip_fixture::Options;
using gguf_test::TempFile;

std::string ThrownBy(const std::string& bytes, bool load_weights) {
  TempFile file(bytes);
  try {
    const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
    vllm::RefuseUnsupportedClipMmproj(gguf, file.path());
    if (load_weights) {
      const vllm::multimodal::Qwen3VLVisionConfig cfg =
          vllm::ClipMmprojVisionConfig(gguf);
      (void)vllm::LoadQwen3VLVisionFromClipMmproj(gguf, cfg);
    }
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

}  // namespace

TEST_CASE("clip mmproj: the config comes from the projector's OWN clip.* kv") {
  const Dims d;
  TempFile file(clip_fixture::Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());

  REQUIRE(vllm::IsClipMmprojGguf(gguf));
  CHECK(vllm::ClipProjectorType(gguf) == "qwen3vl_merger");

  const vllm::multimodal::Qwen3VLVisionConfig cfg =
      vllm::ClipMmprojVisionConfig(gguf);
  CHECK(cfg.hidden_size == d.hidden);
  CHECK(cfg.num_heads == d.heads);
  CHECK(cfg.depth == d.depth);
  CHECK(cfg.intermediate_size == d.inter);
  CHECK(cfg.out_hidden_size == d.out_hidden);
  CHECK(cfg.patch_size == d.patch);
  CHECK(cfg.spatial_merge_size == d.merge);
  // Two, by construction: llama.cpp's temporal merge is exactly two conv2d
  // halves and no `clip.*` key states it.
  CHECK(cfg.temporal_patch_size == 2);
  // Read off the TENSORS, which is the only place the file states them.
  CHECK(cfg.in_channels == d.channels);
  CHECK(cfg.num_position_embeddings == d.num_pos);
  CHECK(cfg.norm_eps == doctest::Approx(d.eps).scale(0.0));
  // Qwen3.8-27B's projector has no DeepStack, matching
  // `deepstack_visual_indexes: []` in its config: not applicable, not owed.
  CHECK(cfg.deepstack_visual_indexes.empty());
}

TEST_CASE("clip mmproj: a DeepStack tap is discovered from the tensor it is named by") {
  const Dims d;
  Options o;
  o.deepstack_layer = 1;
  TempFile file(clip_fixture::Build(d, o));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const vllm::multimodal::Qwen3VLVisionConfig cfg =
      vllm::ClipMmprojVisionConfig(gguf);
  // clip.cpp reads TN_DEEPSTACK_* with the LAYER index, so the present names
  // ARE the indexes; nothing else in the file states them.
  REQUIRE(cfg.deepstack_visual_indexes.size() == 1);
  CHECK(cfg.deepstack_visual_indexes[0] == 1);

  const vllm::multimodal::Qwen3VLVisionWeights w =
      vllm::LoadQwen3VLVisionFromClipMmproj(gguf, cfg);
  REQUIRE(w.deepstack_mergers.size() == 1);
  // The DeepStack merger norms the POST-shuffle width; the main merger norms
  // the pre-shuffle one. Getting this backwards runs and is wrong.
  CHECK(w.deepstack_mergers[0].use_postshuffle_norm);
  CHECK_FALSE(w.merger.use_postshuffle_norm);
}

TEST_CASE("clip mmproj: the two patch-embedding halves INTERLEAVE into one conv3d operand") {
  const Dims d;
  TempFile file(clip_fixture::Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const vllm::multimodal::Qwen3VLVisionConfig cfg =
      vllm::ClipMmprojVisionConfig(gguf);
  const vllm::multimodal::Qwen3VLVisionWeights w =
      vllm::LoadQwen3VLVisionFromClipMmproj(gguf, cfg);

  // [hidden, C * T * p * p] — the flattened conv3d weight the tower reads, and
  // the same width the safetensors `patch_embed.proj.weight` has.
  const int64_t plane = d.plane();
  const int64_t expected =
      d.hidden * d.channels * cfg.temporal_patch_size * plane;
  REQUIRE(static_cast<int64_t>(w.patch_proj_w.size()) == expected);

  // Value-exact, at every position: half 0 supplies t = 0 and half 1 supplies
  // t = 1 INSIDE each channel's p*p block. A concatenation ([all of half 0]
  // then [all of half 1]) has the same size and the same multiset of values, so
  // only a per-position check separates them.
  int64_t checked = 0;
  for (int64_t o = 0; o < d.hidden; ++o) {
    for (int64_t c = 0; c < d.channels; ++c) {
      const int64_t src = (o * d.channels + c) * plane;
      const int64_t dst = (o * d.channels + c) * cfg.temporal_patch_size * plane;
      for (int64_t i = 0; i < plane; ++i) {
        CHECK(w.patch_proj_w[static_cast<size_t>(dst + i)] ==
              doctest::Approx(clip_fixture::PatchHalf0(src + i)).scale(0.0));
        CHECK(w.patch_proj_w[static_cast<size_t>(dst + plane + i)] ==
              doctest::Approx(clip_fixture::PatchHalf1(src + i)).scale(0.0));
        checked += 2;
      }
    }
  }
  // The loop ran: a bound that collapsed to zero would leave every CHECK above
  // unexecuted and the case would still print SUCCESS.
  CHECK(checked == expected);
}

TEST_CASE("clip mmproj: every block and merger tensor lands in its own slot") {
  const Dims d;
  TempFile file(clip_fixture::Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const vllm::multimodal::Qwen3VLVisionConfig cfg =
      vllm::ClipMmprojVisionConfig(gguf);
  const vllm::multimodal::Qwen3VLVisionWeights w =
      vllm::LoadQwen3VLVisionFromClipMmproj(gguf, cfg);

  REQUIRE(static_cast<int64_t>(w.blocks.size()) == d.depth);
  for (int64_t l = 0; l < d.depth; ++l) {
    const auto& b = w.blocks[static_cast<size_t>(l)];
    const float base = static_cast<float>(l) * 1000.0F;
    // The fixture gives each tensor its own base constant, so a swapped pair
    // (ln1 vs ln2, ffn_up vs ffn_down, qkv vs out) fails here rather than
    // merely being the wrong size.
    CHECK(b.norm1_w[0] == doctest::Approx(base + 1.0F).scale(0.0));
    CHECK(b.norm1_b[0] == doctest::Approx(base + 2.0F).scale(0.0));
    CHECK(b.norm2_w[0] == doctest::Approx(base + 3.0F).scale(0.0));
    CHECK(b.norm2_b[0] == doctest::Approx(base + 4.0F).scale(0.0));
    CHECK(b.qkv_w[0] == doctest::Approx(base + 5.0F).scale(0.0));
    CHECK(b.qkv_b[0] == doctest::Approx(base + 6.0F).scale(0.0));
    CHECK(b.proj_w[0] == doctest::Approx(base + 7.0F).scale(0.0));
    CHECK(b.proj_b[0] == doctest::Approx(base + 8.0F).scale(0.0));
    CHECK(b.fc1_w[0] == doctest::Approx(base + 9.0F).scale(0.0));
    CHECK(b.fc1_b[0] == doctest::Approx(base + 10.0F).scale(0.0));
    CHECK(b.fc2_w[0] == doctest::Approx(base + 11.0F).scale(0.0));
    CHECK(b.fc2_b[0] == doctest::Approx(base + 12.0F).scale(0.0));
    // qwen3vl carries a MERGED qkv; the tower reads [3*hidden, hidden].
    CHECK(static_cast<int64_t>(b.qkv_w.size()) == 3 * d.hidden * d.hidden);
    CHECK(static_cast<int64_t>(b.qkv_b.size()) == 3 * d.hidden);
    // `ffn_up` is fc1 and `ffn_down` is fc2: clip.cpp's legacy up/down swap
    // explicitly EXCLUDES PROJECTOR_TYPE_QWEN3VL.
    CHECK(static_cast<int64_t>(b.fc1_w.size()) == d.inter * d.hidden);
    CHECK(static_cast<int64_t>(b.fc2_w.size()) == d.hidden * d.inter);
  }

  // `v.post_ln` is the MERGER's norm, applied before the merge reshape.
  CHECK(w.merger.norm_w[0] == doctest::Approx(21.0F).scale(0.0));
  CHECK(w.merger.norm_b[0] == doctest::Approx(22.0F).scale(0.0));
  CHECK(static_cast<int64_t>(w.merger.norm_w.size()) == d.hidden);
  // `mm.0` -> fc1, `mm.2` -> fc2. There is no `mm.1` in this export.
  CHECK(w.merger.fc1_w[0] == doctest::Approx(31.0F).scale(0.0));
  CHECK(w.merger.fc2_w[0] == doctest::Approx(33.0F).scale(0.0));
  const int64_t merged = d.hidden * d.merge * d.merge;
  CHECK(static_cast<int64_t>(w.merger.fc1_w.size()) == merged * merged);
  CHECK(static_cast<int64_t>(w.merger.fc2_w.size()) == d.out_hidden * merged);
  CHECK(w.deepstack_mergers.empty());

  CHECK(static_cast<int64_t>(w.patch_proj_b.size()) == d.hidden);
  CHECK(static_cast<int64_t>(w.pos_embed_w.size()) == d.num_pos * d.hidden);
}

TEST_CASE("clip mmproj: a projector with ONLY the first patch half is refused BY NAME") {
  const Dims d;
  Options o;
  o.omit_patch_embd_1 = true;
  const std::string message = ThrownBy(clip_fixture::Build(d, o),
                                       /*load_weights=*/true);
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // The MuseGlimmer condition, enforced rather than assumed. The message names
  // the tensor that IS there, the one that is NOT, and both feature counts, so
  // a reader can tell this from a generic missing-tensor error.
  CHECK(message.find("v.patch_embd.weight.1") != std::string::npos);
  CHECK(message.find(std::to_string(d.channels * d.plane())) !=
        std::string::npos);
  CHECK(message.find(std::to_string(2 * d.channels * d.plane())) !=
        std::string::npos);
  CHECK(message.find("inventing") != std::string::npos);
}

TEST_CASE("clip mmproj: a file that is not a clip projector is refused BY NAME") {
  const Dims d;
  Options o;
  o.architecture = "qwen35";  // the LANGUAGE file, passed to --mmproj
  const std::string message = ThrownBy(clip_fixture::Build(d, o),
                                       /*load_weights=*/false);
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("qwen35") != std::string::npos);
  CHECK(message.find("general.architecture") != std::string::npos);
  CHECK(message.find("--mmproj") != std::string::npos);
}

TEST_CASE("clip mmproj: an unsupported projector type is refused BY NAME") {
  const Dims d;
  Options o;
  o.projector_type = "gemma3";
  const std::string message = ThrownBy(clip_fixture::Build(d, o),
                                       /*load_weights=*/false);
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("gemma3") != std::string::npos);
  CHECK(message.find("qwen3vl_merger") != std::string::npos);
}

TEST_CASE("clip mmproj: a muse-glimmer projector gets MuseGlimmer's OWN recorded refusal") {
  const Dims d;
  Options o;
  o.projector_type = "muse-glimmer";
  const std::string message = ThrownBy(clip_fixture::Build(d, o),
                                       /*load_weights=*/false);
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // `MuseGlimmerRefuseMmproj` had exactly one caller before this row, and it
  // was a test. Routing the projector type to it is what gives it a production
  // caller, and restating its reason here instead would let the two drift.
  CHECK(message.find("MuseGlimmer GGUF") != std::string::npos);
  CHECK(message.find("1176") != std::string::npos);
  CHECK(message.find("safetensors checkpoint") != std::string::npos);
}
