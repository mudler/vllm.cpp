// A SYNTHETIC llama.cpp `clip` mmproj GGUF, built byte-by-byte (row
// `LOAD-GGUF-MMPROJ`, issue #821).
//
// Shared by the reader's own unit gate (`test_clip_mmproj_gguf.cpp`) and by the
// loader REACHABILITY gate (`test_gguf_mmproj_reach.cpp`), so both drive one
// fixture and a change to the layout cannot make one of them quietly stop
// describing the file the other builds.
//
// It is synthetic ON PURPOSE and that is a limit worth stating: it proves the
// NAME MAPPING, the metadata mapping and the temporal-patch JOIN, and it proves
// nothing about a real `mmproj-BF16.gguf`'s numerics. The real-file accounting
// gate (a committed 334-tensor manifest) belongs to
// `QUANT-QWEN38-27B-GGUF-ARM`, not here.
//
// Every tensor name, dim order and metadata key is llama.cpp `b10451` =
// `10bf611e533d81f739128304991c5e133c6aebd8` — `tools/mtmd/clip-impl.h` for the
// KEY_*/TN_* spellings, `tools/mtmd/clip.cpp::clip_model_loader::load_tensors`
// for which of them a `qwen3vl_merger` file carries, and
// `tools/mtmd/models/qwen3vl.cpp::clip_graph_qwen3vl::build` for the roles.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vllm/gguf_builder.h"

namespace clip_fixture {

// Deliberately tiny, and deliberately NOT square anywhere it matters, so a
// transposed or mis-strided read cannot pass by symmetry.
struct Dims {
  int64_t hidden = 8;
  int64_t heads = 2;
  int64_t depth = 2;
  int64_t inter = 16;
  int64_t out_hidden = 12;
  int64_t patch = 2;
  int64_t merge = 2;
  int64_t channels = 3;
  int64_t num_pos = 16;  // 4x4 grid
  float eps = 1e-6F;

  int64_t plane() const { return patch * patch; }
  // The flat width of ONE temporal half of the patch embedding, per output row.
  int64_t half_row() const { return channels * plane(); }
};

// Options that turn the fixture into each REFUSAL case. Every one of them is a
// file a user can actually hold: a language GGUF passed to --mmproj, a
// projector for another family, and the MuseGlimmer half-patch export.
struct Options {
  std::string architecture = "clip";
  std::string general_type = "mmproj";
  std::string projector_type = "qwen3vl_merger";
  // Drop `v.patch_embd.weight.1`, i.e. the MuseGlimmer condition: half the
  // input features the temporal patch embedding needs.
  bool omit_patch_embd_1 = false;
  // Add a DeepStack tap at this layer (-1 = none). Qwen3.8-27B's projector has
  // none; MiniMax-H3's encoder has three.
  int deepstack_layer = -1;
};

inline std::string F32Bytes(int64_t numel,
                            const std::function<float(int64_t)>& value) {
  std::string data(static_cast<size_t>(numel) * 4, '\0');
  for (int64_t i = 0; i < numel; ++i) {
    const float v = value(i);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int b = 0; b < 4; ++b) {
      data[static_cast<size_t>(i * 4 + b)] =
          static_cast<char>((bits >> (8 * b)) & 0xff);
    }
  }
  return data;
}

// `dims` are ggml order (ne0 = fastest). A torch [A, B] tensor is {B, A}.
inline void AddF32(gguf_test::GgufModelBuilder& b, const std::string& name,
                   const std::vector<uint64_t>& ggml_dims,
                   const std::function<float(int64_t)>& value) {
  int64_t numel = 1;
  for (uint64_t d : ggml_dims) numel *= static_cast<int64_t>(d);
  b.AddTensor(name, ggml_dims, /*ggml_type=*/0, F32Bytes(numel, value));
}

// The value stored at flat index `i` of the t=0 patch half. Distinct from the
// t=1 series below by more than a constant, so an interleave that swapped the
// halves is visible.
inline float PatchHalf0(int64_t i) { return 1.0F + static_cast<float>(i); }
inline float PatchHalf1(int64_t i) { return -1.0F - static_cast<float>(i); }

inline std::string Build(const Dims& d, const Options& o = Options{}) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", o.architecture));
  if (!o.general_type.empty())
    b.AddKv(gguf_test::StrKv("general.type", o.general_type));
  if (!o.projector_type.empty())
    b.AddKv(gguf_test::StrKv("clip.projector_type", o.projector_type));
  b.AddKv(gguf_test::U32Kv("clip.vision.embedding_length",
                           static_cast<uint32_t>(d.hidden)));
  b.AddKv(gguf_test::U32Kv("clip.vision.feed_forward_length",
                           static_cast<uint32_t>(d.inter)));
  b.AddKv(
      gguf_test::U32Kv("clip.vision.block_count", static_cast<uint32_t>(d.depth)));
  b.AddKv(gguf_test::U32Kv("clip.vision.projection_dim",
                           static_cast<uint32_t>(d.out_hidden)));
  b.AddKv(gguf_test::U32Kv("clip.vision.attention.head_count",
                           static_cast<uint32_t>(d.heads)));
  b.AddKv(gguf_test::U32Kv("clip.vision.patch_size",
                           static_cast<uint32_t>(d.patch)));
  b.AddKv(gguf_test::U32Kv("clip.vision.spatial_merge_size",
                           static_cast<uint32_t>(d.merge)));
  b.AddKv(gguf_test::F32Kv("clip.vision.attention.layer_norm_epsilon", d.eps));

  const std::vector<uint64_t> patch_dims = {
      static_cast<uint64_t>(d.patch), static_cast<uint64_t>(d.patch),
      static_cast<uint64_t>(d.channels), static_cast<uint64_t>(d.hidden)};
  AddF32(b, "v.patch_embd.weight", patch_dims, PatchHalf0);
  if (!o.omit_patch_embd_1)
    AddF32(b, "v.patch_embd.weight.1", patch_dims, PatchHalf1);
  AddF32(b, "v.patch_embd.bias", {static_cast<uint64_t>(d.hidden)},
         [](int64_t i) { return 7.0F + static_cast<float>(i); });
  AddF32(b, "v.position_embd.weight",
         {static_cast<uint64_t>(d.hidden), static_cast<uint64_t>(d.num_pos)},
         [](int64_t i) { return 0.5F * static_cast<float>(i); });

  for (int64_t l = 0; l < d.depth; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    const auto series = [l](float base) {
      return [l, base](int64_t i) {
        return base + static_cast<float>(l) * 1000.0F + static_cast<float>(i);
      };
    };
    AddF32(b, p + "ln1.weight", {static_cast<uint64_t>(d.hidden)}, series(1.0F));
    AddF32(b, p + "ln1.bias", {static_cast<uint64_t>(d.hidden)}, series(2.0F));
    AddF32(b, p + "ln2.weight", {static_cast<uint64_t>(d.hidden)}, series(3.0F));
    AddF32(b, p + "ln2.bias", {static_cast<uint64_t>(d.hidden)}, series(4.0F));
    AddF32(b, p + "attn_qkv.weight",
           {static_cast<uint64_t>(d.hidden), static_cast<uint64_t>(3 * d.hidden)},
           series(5.0F));
    AddF32(b, p + "attn_qkv.bias", {static_cast<uint64_t>(3 * d.hidden)},
           series(6.0F));
    AddF32(b, p + "attn_out.weight",
           {static_cast<uint64_t>(d.hidden), static_cast<uint64_t>(d.hidden)},
           series(7.0F));
    AddF32(b, p + "attn_out.bias", {static_cast<uint64_t>(d.hidden)},
           series(8.0F));
    AddF32(b, p + "ffn_up.weight",
           {static_cast<uint64_t>(d.hidden), static_cast<uint64_t>(d.inter)},
           series(9.0F));
    AddF32(b, p + "ffn_up.bias", {static_cast<uint64_t>(d.inter)}, series(10.0F));
    AddF32(b, p + "ffn_down.weight",
           {static_cast<uint64_t>(d.inter), static_cast<uint64_t>(d.hidden)},
           series(11.0F));
    AddF32(b, p + "ffn_down.bias", {static_cast<uint64_t>(d.hidden)},
           series(12.0F));
  }

  // The merger: `v.post_ln` is the PRE-shuffle norm and `mm.0` / `mm.2` are
  // fc1 / fc2 (there is no `mm.1` in a qwen3vl_merger export).
  AddF32(b, "v.post_ln.weight", {static_cast<uint64_t>(d.hidden)},
         [](int64_t i) { return 21.0F + static_cast<float>(i); });
  AddF32(b, "v.post_ln.bias", {static_cast<uint64_t>(d.hidden)},
         [](int64_t i) { return 22.0F + static_cast<float>(i); });
  const int64_t merged = d.hidden * d.merge * d.merge;
  AddF32(b, "mm.0.weight",
         {static_cast<uint64_t>(merged), static_cast<uint64_t>(merged)},
         [](int64_t i) { return 31.0F + static_cast<float>(i); });
  AddF32(b, "mm.0.bias", {static_cast<uint64_t>(merged)},
         [](int64_t i) { return 32.0F + static_cast<float>(i); });
  AddF32(b, "mm.2.weight",
         {static_cast<uint64_t>(merged), static_cast<uint64_t>(d.out_hidden)},
         [](int64_t i) { return 33.0F + static_cast<float>(i); });
  AddF32(b, "mm.2.bias", {static_cast<uint64_t>(d.out_hidden)},
         [](int64_t i) { return 34.0F + static_cast<float>(i); });

  if (o.deepstack_layer >= 0) {
    const std::string p =
        "v.deepstack." + std::to_string(o.deepstack_layer) + ".";
    AddF32(b, p + "norm.weight", {static_cast<uint64_t>(merged)},
           [](int64_t i) { return 41.0F + static_cast<float>(i); });
    AddF32(b, p + "norm.bias", {static_cast<uint64_t>(merged)},
           [](int64_t i) { return 42.0F + static_cast<float>(i); });
    AddF32(b, p + "fc1.weight",
           {static_cast<uint64_t>(merged), static_cast<uint64_t>(merged)},
           [](int64_t i) { return 43.0F + static_cast<float>(i); });
    AddF32(b, p + "fc1.bias", {static_cast<uint64_t>(merged)},
           [](int64_t i) { return 44.0F + static_cast<float>(i); });
    AddF32(b, p + "fc2.weight",
           {static_cast<uint64_t>(merged), static_cast<uint64_t>(d.out_hidden)},
           [](int64_t i) { return 45.0F + static_cast<float>(i); });
    AddF32(b, p + "fc2.bias", {static_cast<uint64_t>(d.out_hidden)},
           [](int64_t i) { return 46.0F + static_cast<float>(i); });
  }
  return b.Build();
}

}  // namespace clip_fixture
