// The synthetic `deepseek4v` mmproj GGUF builder, shared by the W3A reader gate
// and the W4 reachability gate (row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411).
//
// EXTRACTED, NOT REWRITTEN. Every byte of the builder below was the local
// fixture of `tests/vllm/models/test_deepseek_v4_mmproj.cpp`. W4 needs the same
// file shape to drive the production loader, and a second hand-written builder
// would be a second description of the artifact, free to drift from the reader
// while both suites stay green. The suite that owned it now includes this
// header and keeps its own refusal helpers, which are doctest-facing rather
// than fixture.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/clip_mmproj_gguf.h"
#include "vt/dtype.h"

namespace dsv4_mmproj_test {

using gguf_test::TempFile;
using vllm::multimodal::DeepSeekV4VisionConfig;

// Deliberately tiny, and deliberately distinct in every axis, so a transposed
// or mis-strided read cannot pass by symmetry. `inter` is neither `hidden` nor
// `2 * hidden`, and `output` is neither.
struct Dims {
  int64_t hidden = 8;
  int64_t heads = 2;
  int64_t depth = 2;
  int64_t inter = 6;
  int64_t output = 12;
  int64_t patch = 2;
  int64_t ratio = 3;
  int64_t channels = 3;
  // NOT the W2 default of 1e-6: a reader that hardcoded the default instead of
  // reading `clip.vision.attention.layer_norm_epsilon` passes with 1e-6 and
  // fails here.
  float eps = 1.5e-5F;

  int64_t patch_dim() const { return channels * patch * patch; }
  int64_t aligner_in() const { return hidden * ratio * ratio; }
};

// A bf16-EXACT, strictly increasing value series.
//
// bf16 keeps 7 explicit mantissa bits, so `(1 + k/128) * 2^e` with integer
// `k` in [0, 127] survives f32 -> bf16 -> f32 unchanged AND gives every flat
// index its own bf16 word. A plain `base + i` series would not: bf16's ULP at
// 20000 is 128, so hundreds of indices would share a word and an off-by-one
// permutation would pass every check below.
inline float Series(int family, int64_t i) {
  const int64_t k = i % 128;
  const int exponent = family + static_cast<int>(i / 128);
  return std::ldexp(1.0F + static_cast<float>(k) / 128.0F, exponent);
}

// Each tensor gets its own exponent family, so a swapped slot (q for k,
// gate for up, ln1 for ln2) lands in a different binade and cannot hide.
constexpr int kFamPatchW = -6;
constexpr int kFamPatchB = -5;
constexpr int kFamLn1 = -4;
constexpr int kFamLn2 = -3;
constexpr int kFamQ = 1;
constexpr int kFamK = 9;
constexpr int kFamV = 17;
constexpr int kFamQBias = 25;
constexpr int kFamKBias = 26;
constexpr int kFamVBias = 27;
constexpr int kFamOutW = 28;
constexpr int kFamOutB = 29;
constexpr int kFamGate = 30;
constexpr int kFamUp = 31;
constexpr int kFamDown = 32;
constexpr int kFamPostLn = -2;
constexpr int kFamMm1W = 33;
constexpr int kFamMm1B = 34;
constexpr int kFamMm2W = 35;
constexpr int kFamMm2B = 36;
constexpr int kFamImgStart = 37;
constexpr int kFamImgEnd = 38;
constexpr int kFamImgPad = 39;
constexpr int kFamNewline = 40;

// A per-layer stride keeps layer 1's tensors out of layer 0's binades.
inline int Fam(int family, int64_t layer) { return family + static_cast<int>(layer) * 64; }

inline std::string F32Bytes(int64_t numel, const std::function<float(int64_t)>& value) {
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

inline std::string Bf16Bytes(int64_t numel, const std::function<float(int64_t)>& value) {
  std::string data(static_cast<size_t>(numel) * 2, '\0');
  for (int64_t i = 0; i < numel; ++i) {
    const uint16_t word = vt::F32ToBF16(value(i));
    data[static_cast<size_t>(i * 2)] = static_cast<char>(word & 0xff);
    data[static_cast<size_t>(i * 2 + 1)] = static_cast<char>((word >> 8) & 0xff);
  }
  return data;
}

inline int64_t Numel(const std::vector<uint64_t>& dims) {
  int64_t n = 1;
  for (uint64_t d : dims) n *= static_cast<int64_t>(d);
  return n;
}

// `dims` are ggml order (ne0 = fastest). A torch [A, B] tensor is {B, A}.
inline void AddF32(gguf_test::GgufModelBuilder& b, const std::string& name,
            const std::vector<uint64_t>& dims, int family) {
  b.AddTensor(name, dims, /*ggml_type=*/0,
              F32Bytes(Numel(dims), [family](int64_t i) { return Series(family, i); }));
}

inline void AddBf16(gguf_test::GgufModelBuilder& b, const std::string& name,
             const std::vector<uint64_t>& dims, int family) {
  b.AddTensor(name, dims, /*ggml_type=*/30,
              Bf16Bytes(Numel(dims), [family](int64_t i) { return Series(family, i); }));
}

// Every refusal case is a file a user can actually hold: a projector for
// another family, a GELU-MLP variant of this one, an export missing a tensor,
// and an export carrying one this reader never reads.
struct Options {
  std::string architecture = "clip";
  std::string general_type = "mmproj";
  std::string projector_type = "deepseek4v";
  bool use_silu = true;
  bool emit_use_silu = true;
  std::string omit_tensor;
  std::string stray_tensor;
  // Emit the FUSED `v.blk.{bid}.attn_qkv.{weight,bias}` instead of the six
  // separate q/k/v tensors. This is what the pinned oracle's own
  // `convert_hf_to_gguf.py` produces, and it is the arm this build does not
  // implement.
  bool fused_qkv = false;
  // Write one named tensor with its ggml dims REVERSED, which is a torch
  // transpose of the same element count. Nothing about the numel changes, so
  // only the reader's shape guard can catch it.
  std::string transpose_tensor;
  // Replace one `clip.vision.*` geometry kv with these raw bytes, keyed by the
  // key itself, so a case can hand the reader a geometry no `U32Kv` can spell.
  // Keyed rather than one field per key: the reader bounds SEVEN of these and
  // every one of them is a `Require` shape, a loop bound or a `resize` argument.
  std::map<std::string, std::string> geometry_kv;
  // Emit ONLY the four tensors the loader reads before it sizes the fused qkv
  // buffer: the patch embedding's weight and bias, and layer 0's two norms.
  // A projector this shape is what turns an absurd `embedding_length` into a
  // bare `std::bad_alloc` rather than a named refusal, and at patch_size 1 it
  // stays about 1.6 MB while doing it.
  bool only_before_qkv = false;
};

// The declared geometry for `key`, or the case's own raw override for it.
inline void AddGeometry(gguf_test::GgufModelBuilder& b, const Options& o,
                 const char* key, int64_t declared) {
  const auto it = o.geometry_kv.find(key);
  if (it != o.geometry_kv.end()) {
    b.AddKv(it->second);
    return;
  }
  b.AddKv(gguf_test::U32Kv(key, static_cast<uint32_t>(declared)));
}

// The builder has no signed-integer kv encoder and it is shared with every
// other GGUF test, so this one stays local: GGUF type 5 is i32.
inline std::string I32Kv(const std::string& key, int32_t val) {
  return gguf_test::GStr(key) + gguf_test::U32Le(5) +
         gguf_test::U32Le(static_cast<uint32_t>(val));
}

inline uint64_t U(int64_t v) { return static_cast<uint64_t>(v); }

inline std::string Build(const Dims& d, const Options& o = Options{}) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", o.architecture));
  if (!o.general_type.empty())
    b.AddKv(gguf_test::StrKv("general.type", o.general_type));
  if (!o.projector_type.empty())
    b.AddKv(gguf_test::StrKv("clip.projector_type", o.projector_type));
  AddGeometry(b, o, "clip.vision.embedding_length", d.hidden);
  AddGeometry(b, o, "clip.vision.feed_forward_length", d.inter);
  AddGeometry(b, o, "clip.vision.block_count", d.depth);
  AddGeometry(b, o, "clip.vision.projection_dim", d.output);
  AddGeometry(b, o, "clip.vision.attention.head_count", d.heads);
  AddGeometry(b, o, "clip.vision.patch_size", d.patch);
  AddGeometry(b, o, "clip.vision.projector.scale_factor", d.ratio);
  b.AddKv(gguf_test::F32Kv("clip.vision.attention.layer_norm_epsilon", d.eps));
  if (o.emit_use_silu) b.AddKv(gguf_test::BoolKv("clip.use_silu", o.use_silu));

  const auto skip = [&o](const char* name) {
    if (o.only_before_qkv) {
      for (const char* kept : {"v.patch_embd.weight", "v.patch_embd.bias",
                               "v.blk.0.ln1.weight", "v.blk.0.ln2.weight"}) {
        if (std::strcmp(kept, name) == 0) return false;
      }
      return true;
    }
    return o.omit_tensor == name;
  };
  // Every tensor goes through these so ONE named tensor can be written with its
  // ggml dims reversed; the shape guard is the only thing that can see it.
  const auto f32 = [&](const std::string& name, std::vector<uint64_t> dims,
                       int family) {
    if (o.transpose_tensor == name) std::reverse(dims.begin(), dims.end());
    AddF32(b, name, dims, family);
  };
  const auto bf16 = [&](const std::string& name, std::vector<uint64_t> dims,
                        int family) {
    if (o.transpose_tensor == name) std::reverse(dims.begin(), dims.end());
    AddBf16(b, name, dims, family);
  };

  // The aligner projection. `mm.1` is the 3x3 unfold's consumer and `mm.2`
  // closes it; there is no `mm.0` in a deepseek4v export.
  if (!skip("mm.1.weight")) bf16("mm.1.weight", {U(d.aligner_in()), U(d.output)}, kFamMm1W);
  if (!skip("mm.1.bias")) f32("mm.1.bias", {U(d.output)}, kFamMm1B);
  if (!skip("mm.2.weight")) bf16("mm.2.weight", {U(d.output), U(d.output)}, kFamMm2W);
  if (!skip("mm.2.bias")) f32("mm.2.bias", {U(d.output)}, kFamMm2B);

  // The four learned sentinel vectors, f32 [projection_dim] in the artifact.
  if (!skip("v.token_embd.img_start"))
    f32("v.token_embd.img_start", {U(d.output)}, kFamImgStart);
  if (!skip("v.token_embd.img_end"))
    f32("v.token_embd.img_end", {U(d.output)}, kFamImgEnd);
  if (!skip("v.token_embd.img_pad"))
    f32("v.token_embd.img_pad", {U(d.output)}, kFamImgPad);
  if (!skip("v.image_newline")) f32("v.image_newline", {U(d.output)}, kFamNewline);

  for (int64_t l = 0; l < d.depth; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    if (!skip((p + "ln1.weight").c_str()))
      f32(p + "ln1.weight", {U(d.hidden)}, Fam(kFamLn1, l));
    if (!skip((p + "ln2.weight").c_str()))
      f32(p + "ln2.weight", {U(d.hidden)}, Fam(kFamLn2, l));
    if (o.fused_qkv) {
      // What `convert_hf_to_gguf.py` writes at the pin: `tensor_mapping.py`
      // maps `vision.blocks.{bid}.attn.wqkv` to V_ENC_ATTN_QKV and
      // `constants.py` spells that `v.blk.{bid}.attn_qkv`, and nothing splits
      // it for this family.
      bf16(p + "attn_qkv.weight", {U(d.hidden), U(3 * d.hidden)}, Fam(kFamQ, l));
      f32(p + "attn_qkv.bias", {U(3 * d.hidden)}, Fam(kFamQBias, l));
    } else {
      if (!skip((p + "attn_q.weight").c_str()))
        bf16(p + "attn_q.weight", {U(d.hidden), U(d.hidden)}, Fam(kFamQ, l));
      if (!skip((p + "attn_k.weight").c_str()))
        bf16(p + "attn_k.weight", {U(d.hidden), U(d.hidden)}, Fam(kFamK, l));
      if (!skip((p + "attn_v.weight").c_str()))
        bf16(p + "attn_v.weight", {U(d.hidden), U(d.hidden)}, Fam(kFamV, l));
      if (!skip((p + "attn_q.bias").c_str()))
        f32(p + "attn_q.bias", {U(d.hidden)}, Fam(kFamQBias, l));
      if (!skip((p + "attn_k.bias").c_str()))
        f32(p + "attn_k.bias", {U(d.hidden)}, Fam(kFamKBias, l));
      if (!skip((p + "attn_v.bias").c_str()))
        f32(p + "attn_v.bias", {U(d.hidden)}, Fam(kFamVBias, l));
    }
    if (!skip((p + "attn_out.weight").c_str()))
      bf16(p + "attn_out.weight", {U(d.hidden), U(d.hidden)}, Fam(kFamOutW, l));
    if (!skip((p + "attn_out.bias").c_str()))
      f32(p + "attn_out.bias", {U(d.hidden)}, Fam(kFamOutB, l));
    if (!skip((p + "ffn_gate.weight").c_str()))
      bf16(p + "ffn_gate.weight", {U(d.hidden), U(d.inter)}, Fam(kFamGate, l));
    if (!skip((p + "ffn_up.weight").c_str()))
      bf16(p + "ffn_up.weight", {U(d.hidden), U(d.inter)}, Fam(kFamUp, l));
    if (!skip((p + "ffn_down.weight").c_str()))
      bf16(p + "ffn_down.weight", {U(d.inter), U(d.hidden)}, Fam(kFamDown, l));
  }

  if (!skip("v.post_ln.weight")) f32("v.post_ln.weight", {U(d.hidden)}, kFamPostLn);
  // ggml order {p, p, C, out} == torch [out, C, p, p], exactly as the artifact
  // stores it and exactly what `data_torch.reshape(shape[0], 3, p, p)` in the
  // pinned converter produced.
  if (!skip("v.patch_embd.weight"))
    f32("v.patch_embd.weight", {U(d.patch), U(d.patch), U(d.channels), U(d.hidden)},
        kFamPatchW);
  if (!skip("v.patch_embd.bias")) f32("v.patch_embd.bias", {U(d.hidden)}, kFamPatchB);

  if (!o.stray_tensor.empty()) f32(o.stray_tensor, {U(d.hidden)}, 50);
  return b.Build();
}

// bf16 word at a flat index of a loaded tensor.
inline uint16_t Word(const vt::Tensor& t, int64_t i) {
  return t.Ptr<uint16_t>()[i];
}

inline float F32At(const vt::Tensor& t, int64_t i) { return t.Ptr<float>()[i]; }
}  // namespace dsv4_mmproj_test
