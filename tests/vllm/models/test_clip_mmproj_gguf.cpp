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

#include <cstdlib>
#include <cstring>
#include <set>
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

// ── The LIVE artifact ───────────────────────────────────────────────────────
//
// Everything above is synthetic, and stays that way: the synthetic fixture is
// the CI gate, because CI must not depend on a 931 MB file on a NAS share.
// This case is the CONFIRMATION — the same reader, over the bytes a user
// actually holds — and it SKIPS LOUDLY when the file is not named:
//
//   VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf
//     ./build/tests/test_clip_mmproj_gguf
//
// The artifact is `unsloth/Qwen3.8-27B-GGUF` @
// `fe1e2a23d973adb629709749dc4f6756df66ef10`, file `mmproj-BF16.gguf`,
// 931,146,432 bytes, sha256
// 83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53. Its numbers
// below were read OFF THAT FILE's header, not copied from this reader, so they
// are an independent statement of what the container holds.
//
// What this case does NOT do: it is not the committed 334-name manifest. That,
// and the accounting against it in CI, is owed by `QUANT-QWEN38-27B-GGUF-ARM`
// (#821 W2) where the manifest lives.
namespace {

// The tensor names THIS reader consumes, restated from the real file's header
// rather than from the reader's source, so a name the reader gets wrong is a
// disagreement between two sources rather than a tautology.
std::vector<std::string> LiveExpectedTensorNames(int64_t depth) {
  std::vector<std::string> names = {
      "v.patch_embd.weight", "v.patch_embd.weight.1", "v.patch_embd.bias",
      "v.position_embd.weight", "v.post_ln.weight", "v.post_ln.bias",
      "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"};
  for (int64_t l = 0; l < depth; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    for (const char* leaf :
         {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
          "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight",
          "attn_out.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight",
          "ffn_down.bias"}) {
      names.push_back(p + leaf);
    }
  }
  return names;
}

}  // namespace

TEST_CASE("clip mmproj: the REAL Qwen3.8-27B projector maps name for name") {
  const char* env = std::getenv("VLLM_CPP_QWEN38_27B_MMPROJ");
  if (env == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_QWEN38_27B_MMPROJ to the real "
            "mmproj-BF16.gguf to confirm this mapping on the shipped bytes");
    return;
  }
  const vllm::GgufFile gguf = vllm::GgufFile::Open(std::string(env));

  // It is the file this case is about. A wrong file must say so here rather
  // than fail an arithmetic assert forty lines down.
  REQUIRE(vllm::IsClipMmprojGguf(gguf));
  REQUIRE(vllm::ClipProjectorType(gguf) == "qwen3vl_merger");
  REQUIRE(gguf.Tensors().size() == 334);
  REQUIRE_NOTHROW(vllm::RefuseUnsupportedClipMmproj(gguf, std::string(env)));

  // ── The `clip.*` -> Qwen3VLVisionConfig mapping, on the real metadata ────
  const vllm::multimodal::Qwen3VLVisionConfig cfg =
      vllm::ClipMmprojVisionConfig(gguf);
  CHECK(cfg.hidden_size == 1152);            // clip.vision.embedding_length
  CHECK(cfg.num_heads == 16);                // clip.vision.attention.head_count
  CHECK(cfg.depth == 27);                    // clip.vision.block_count
  CHECK(cfg.intermediate_size == 4304);      // clip.vision.feed_forward_length
  CHECK(cfg.out_hidden_size == 5120);        // clip.vision.projection_dim
  CHECK(cfg.patch_size == 16);               // clip.vision.patch_size
  CHECK(cfg.spatial_merge_size == 2);        // clip.vision.spatial_merge_size
  CHECK(cfg.temporal_patch_size == 2);       // by construction; no kv states it
  CHECK(cfg.in_channels == 3);               // off v.patch_embd.weight's shape
  CHECK(cfg.num_position_embeddings == 2304);  // off v.position_embd.weight
  CHECK(cfg.norm_eps == doctest::Approx(1e-6F).epsilon(1e-3));

  // DeepStack, from two independent statements that must agree. We discover a
  // tap from the TENSOR that names it; the file separately declares
  // `clip.vision.is_deepstack_layers`, one bool per block. Qwen3.8-27B taps
  // nothing, so both must be empty — and if a later projector taps something,
  // this is the assert that catches a discovery reading the wrong one.
  CHECK(cfg.deepstack_visual_indexes.empty());
  const vllm::GgufValue* ds = gguf.FindKv("clip.vision.is_deepstack_layers");
  REQUIRE(ds != nullptr);
  REQUIRE(ds->TypeId() == vllm::kGgufArray);
  const vllm::GgufArray& ds_arr = std::get<vllm::GgufArray>(ds->v);
  CHECK(ds_arr.elems.size() == 27);
  int declared_taps = 0;
  for (const vllm::GgufValue& e : ds_arr.elems) {
    if (e.TypeId() == vllm::kGgufBool && std::get<bool>(e.v)) ++declared_taps;
  }
  CHECK(declared_taps == 0);

  // ── The `v.*` / `mm.*` name map: nothing missing, nothing unread ─────────
  //
  // Two directions, checked by two different things. FILE -> READER is the
  // load below: every name the reader asks for must be in the file, and a
  // missing one throws naming itself. READER -> FILE is this set difference,
  // against the restatement above rather than against the reader's own
  // strings, because comparing the reader with itself would pass by
  // construction. The restatement is therefore load-bearing and is read off
  // the artifact's header, not off the reader.
  std::set<std::string> present;
  for (const vllm::GgufTensorInfo& t : gguf.Tensors()) present.insert(t.name);
  std::set<std::string> consumed;
  for (const std::string& n : LiveExpectedTensorNames(cfg.depth)) {
    consumed.insert(n);
  }
  CHECK(consumed.size() == 334);
  std::vector<std::string> missing;   // named, not read
  std::vector<std::string> unread;    // shipped, not consumed
  for (const std::string& n : consumed) {
    if (present.count(n) == 0) missing.push_back(n);
  }
  for (const std::string& n : present) {
    if (consumed.count(n) == 0) unread.push_back(n);
  }
  CAPTURE(missing.empty() ? std::string("-") : missing.front());
  CAPTURE(unread.empty() ? std::string("-") : unread.front());
  CHECK(missing.empty());
  CHECK(unread.empty());

  // ── The two-half temporal patch embedding, on the shipped bytes ──────────
  //
  // The half this reader refuses a file for lacking is PRESENT here, so the
  // real artifact takes the accepting arm; the refusal is for a file that is
  // not this one.
  REQUIRE(present.count("v.patch_embd.weight.1") == 1);
  const vllm::GgufTensorInfo& h0 = gguf.Get("v.patch_embd.weight");
  const vllm::GgufTensorInfo& h1 = gguf.Get("v.patch_embd.weight.1");
  // Torch order [out, C, p, p], the shape the join's own check demands.
  CHECK(h0.shape == std::vector<int64_t>{1152, 3, 16, 16});
  CHECK(h1.shape == h0.shape);
  // Both halves ship F32 on this artifact, so the values below are read
  // STRAIGHT out of the mmap. That deliberately avoids gating the join through
  // the same dequant helper the reader uses, which would prove consistency and
  // not correctness.
  REQUIRE(h0.ggml_type == 0);
  REQUIRE(h1.ggml_type == 0);
  const int64_t half_elems = 1152 * 3 * 16 * 16;
  REQUIRE(h0.nbytes == static_cast<size_t>(half_elems) * sizeof(float));
  std::vector<float> w0(static_cast<size_t>(half_elems));
  std::vector<float> w1(static_cast<size_t>(half_elems));
  std::memcpy(w0.data(), h0.data, h0.nbytes);
  std::memcpy(w1.data(), h1.data, h1.nbytes);
  // The halves must actually DIFFER, or an interleave and a concatenation
  // would be indistinguishable and everything below would prove nothing.
  bool halves_differ = false;
  for (int64_t i = 0; i < half_elems && !halves_differ; ++i) {
    halves_differ = w0[static_cast<size_t>(i)] != w1[static_cast<size_t>(i)];
  }
  REQUIRE(halves_differ);

  const vllm::multimodal::Qwen3VLVisionWeights w =
      vllm::LoadQwen3VLVisionFromClipMmproj(gguf, cfg);
  REQUIRE(static_cast<int64_t>(w.patch_proj_w.size()) == 2 * half_elems);
  // Every position, counted rather than asserted one CHECK at a time: 1.77 M
  // doctest assertions would be the report, not the result.
  const int64_t plane = 16 * 16;
  int64_t compared = 0;
  int64_t wrong = 0;
  for (int64_t o = 0; o < 1152; ++o) {
    for (int64_t c = 0; c < 3; ++c) {
      const int64_t src = (o * 3 + c) * plane;
      const int64_t dst = (o * 3 + c) * 2 * plane;
      for (int64_t i = 0; i < plane; ++i) {
        if (w.patch_proj_w[static_cast<size_t>(dst + i)] !=
            w0[static_cast<size_t>(src + i)]) {
          ++wrong;
        }
        if (w.patch_proj_w[static_cast<size_t>(dst + plane + i)] !=
            w1[static_cast<size_t>(src + i)]) {
          ++wrong;
        }
        compared += 2;
      }
    }
  }
  CHECK(wrong == 0);
  // The loop ran over the whole operand. A bound that collapsed would leave
  // `wrong == 0` true and prove nothing.
  CHECK(compared == 2 * half_elems);

  // ── The rest of the tower, at the widths the real metadata implies ───────
  REQUIRE(w.blocks.size() == 27);
  CHECK(static_cast<int64_t>(w.blocks[0].qkv_w.size()) == 3 * 1152 * 1152);
  CHECK(static_cast<int64_t>(w.blocks[0].fc1_w.size()) == 4304 * 1152);
  CHECK(static_cast<int64_t>(w.blocks[26].fc2_w.size()) == 1152 * 4304);
  CHECK(static_cast<int64_t>(w.pos_embed_w.size()) == 2304 * 1152);
  // `v.post_ln` is 1152 wide — the PRE-shuffle hidden size, not the merged
  // 4608 — which is the file confirming `use_postshuffle_norm = false` rather
  // than this reader assuming it.
  CHECK(static_cast<int64_t>(w.merger.norm_w.size()) == 1152);
  CHECK(w.merger.use_postshuffle_norm == false);
  CHECK(static_cast<int64_t>(w.merger.fc1_w.size()) == 4608 * 4608);
  CHECK(static_cast<int64_t>(w.merger.fc2_w.size()) == 5120 * 4608);
  CHECK(w.deepstack_mergers.empty());
}
