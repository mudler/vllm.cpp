// The `deepseek4v` arm of the llama.cpp `clip` mmproj reader (row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W3A, issue
// [#2411](https://github.com/mudler/vllm.cpp/issues/2411)).
//
// WHAT THIS GATES. Four layout mismatches between what
// `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`'s `mmproj-BF16.gguf` STORES and
// what the W2 tower CONSUMES. None of them crashes when it is wrong; each one
// produces a tower that runs and is fluent and is wrong, so each one gets its
// own value-exact case here:
//
//   (a) the file stores `attn_q` / `attn_k` / `attn_v` SEPARATELY and W2 wants
//       one fused `qkv_weight [3*hidden, hidden]`. THE SPLIT IS THIS FILE'S,
//       not the family's: the pinned `convert_hf_to_gguf.py` emits the FUSED
//       `v.blk.{bid}.attn_qkv` instead, which this build does not implement and
//       refuses by name;
//   (b) the file stores `ffn_gate` and `ffn_up` SEPARATELY and W2 wants one
//       `mlp_w1_weight [2*intermediate, hidden]`;
//   (c) `v.patch_embd.weight` is a 4-D conv2d weight and W2 wants a 2-D torch
//       Linear weight over an `F.unfold` whose element order is [channel, dy,
//       dx];
//   (d) the file stores every 1-D tensor and the patch embedding as F32 while
//       W2's contract says RMSNorm weights stay f32 and every linear weight and
//       bias is the model dtype.
//
// It also gates the refusals: the unimplemented fused arm, an out-of-range
// geometry, a wrong shape, and a projector that declares no activation.
//
// It does NOT prove that anything reaches this reader. W4 owns the production
// call site for row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, and the
// spec lists the gap under `## Owed`.
//
// THE FIXTURE IS SYNTHETIC, and that limit is the same one
// `clip_mmproj_fixture.h` states: it proves the NAME MAPPING, the METADATA
// mapping and the four joins above, and it proves nothing about the real
// artifact's numerics. Every tensor name, every ggml dim order and every dtype
// below was read off the pinned artifact's own header on 2026-09-05
// (`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` at
// `b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0`, `mmproj-BF16.gguf`, sha256
// `e4914c6c8063d01f4cbb6dafdf2f959c7d06fbe8ad11ae5b11ad032edd42642e`): the
// 2-D linear weights are BF16, every bias, every norm weight, the patch
// embedding and the four sentinel vectors are F32.
//
// Upstream anchors, at the secondary oracle `llama-cpp-dsv4vision` =
// `ggml-org/llama.cpp` release `b10766` =
// `9400c8946e4da5e7694f2c26d6d4e50e14b690fa`, read from the diff that
// introduces `tools/mtmd/models/deepseek4v.cpp` as blob `ffe8f59d9997` — the
// blob that path holds at that pin:
//   conversion/deepseek.py::DeepseekV4FlashVisionModel.modify_tensors — the
//       `w1` chunk that (b) has to undo, and the patch-embedding reshape that
//       (c) has to undo
//   gguf-py/gguf/tensor_mapping.py — which checkpoint tensor becomes which
//       `v.*` / `mm.*` name
//   tools/mtmd/clip.cpp::clip_model_loader, PROJECTOR_TYPE_DEEPSEEK4V — the
//       `clip.*` hyper-parameter reads and the hardcoded `rope_theta`
//   tools/mtmd/models/deepseek4v.cpp::clip_graph_deepseek4v::build — the roles
#include <doctest/doctest.h>

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

namespace {

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
float Series(int family, int64_t i) {
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
int Fam(int family, int64_t layer) { return family + static_cast<int>(layer) * 64; }

std::string F32Bytes(int64_t numel, const std::function<float(int64_t)>& value) {
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

std::string Bf16Bytes(int64_t numel, const std::function<float(int64_t)>& value) {
  std::string data(static_cast<size_t>(numel) * 2, '\0');
  for (int64_t i = 0; i < numel; ++i) {
    const uint16_t word = vt::F32ToBF16(value(i));
    data[static_cast<size_t>(i * 2)] = static_cast<char>(word & 0xff);
    data[static_cast<size_t>(i * 2 + 1)] = static_cast<char>((word >> 8) & 0xff);
  }
  return data;
}

int64_t Numel(const std::vector<uint64_t>& dims) {
  int64_t n = 1;
  for (uint64_t d : dims) n *= static_cast<int64_t>(d);
  return n;
}

// `dims` are ggml order (ne0 = fastest). A torch [A, B] tensor is {B, A}.
void AddF32(gguf_test::GgufModelBuilder& b, const std::string& name,
            const std::vector<uint64_t>& dims, int family) {
  b.AddTensor(name, dims, /*ggml_type=*/0,
              F32Bytes(Numel(dims), [family](int64_t i) { return Series(family, i); }));
}

void AddBf16(gguf_test::GgufModelBuilder& b, const std::string& name,
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
void AddGeometry(gguf_test::GgufModelBuilder& b, const Options& o,
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
std::string I32Kv(const std::string& key, int32_t val) {
  return gguf_test::GStr(key) + gguf_test::U32Le(5) +
         gguf_test::U32Le(static_cast<uint32_t>(val));
}

uint64_t U(int64_t v) { return static_cast<uint64_t>(v); }

std::string Build(const Dims& d, const Options& o = Options{}) {
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
uint16_t Word(const vt::Tensor& t, int64_t i) {
  return t.Ptr<uint16_t>()[i];
}

float F32At(const vt::Tensor& t, int64_t i) { return t.Ptr<float>()[i]; }

std::string ThrownBy(const std::string& bytes, bool load_weights) {
  TempFile file(bytes);
  try {
    const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
    vllm::RefuseUnsupportedDeepSeekV4ClipMmproj(gguf, file.path());
    const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
    vllm::RefuseUnaccountedDeepSeekV4ClipMmproj(gguf, cfg, file.path());
    if (load_weights) (void)vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// `RequireGeometry`'s refusal, spelled out.
//
// `Contains(message, key)` and `Contains(message, value)` are each satisfiable
// by ACCIDENT, and one of them was: with every bound deleted, the fallback
// unaccounted-tensor message prints "enumerated for depth -1", so
// `Contains(neg, "-1")` passed while measuring nothing. Asserting the whole
// phrase ties the case to the refusal it names.
bool RefusedByBound(const std::string& message, const char* key, int64_t value) {
  return Contains(message, std::string(key) + " is " + std::to_string(value) +
                               ", and this reader accepts 1 to ");
}

}  // namespace

TEST_CASE("deepseek4v mmproj: the config comes from the projector's OWN clip.* kv") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());

  REQUIRE(vllm::IsClipMmprojGguf(gguf));
  CHECK(vllm::ClipProjectorType(gguf) == "deepseek4v");
  REQUIRE_NOTHROW(vllm::RefuseUnsupportedDeepSeekV4ClipMmproj(gguf, file.path()));

  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  CHECK(cfg.hidden_size == d.hidden);
  CHECK(cfg.num_heads == d.heads);
  CHECK(cfg.depth == d.depth);
  CHECK(cfg.intermediate_size == d.inter);
  // `projection_dim` is the aligner's output width, and
  // `projector.scale_factor` is the 3x3 downsample ratio
  // (clip.cpp PROJECTOR_TYPE_DEEPSEEK4V reads KEY_PROJ_SCALE_FACTOR into
  // `hparams.n_merge`, and deepseek4v.cpp unfolds with it).
  CHECK(cfg.output_size == d.output);
  CHECK(cfg.downsample_ratio == d.ratio);
  CHECK(cfg.patch_size == d.patch);
  // READ, not hardcoded: the fixture's eps is 1.5e-5 and the W2 default is
  // 1e-6, so a reader that skipped the key would report 1e-6 here.
  CHECK(cfg.norm_epsilon == doctest::Approx(d.eps).scale(0.0));
  // The file states no RoPE theta. llama.cpp hardcodes 10000.0 for this
  // projector in the same case that reads the keys above, and the pinned
  // converter's `get_vision_config` defaults `vision_rope_theta` to the same
  // value without writing a key, so the W2 default stands.
  CHECK(cfg.rope_theta == doctest::Approx(10000.0));
  CHECK(cfg.compute_dtype == vt::DType::kBF16);
}

TEST_CASE("deepseek4v mmproj: (a) attn_q/k/v FUSE into qkv in q,k,v row order") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj loaded =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);

  REQUIRE(loaded.weights.blocks.size() == static_cast<size_t>(d.depth));
  int64_t checked = 0;
  for (int64_t l = 0; l < d.depth; ++l) {
    const vt::Tensor& qkv = loaded.weights.blocks[static_cast<size_t>(l)].qkv_weight;
    const vt::Tensor& bias = loaded.weights.blocks[static_cast<size_t>(l)].qkv_bias;
    REQUIRE(qkv.rank == 2);
    REQUIRE(qkv.shape[0] == 3 * d.hidden);
    REQUIRE(qkv.shape[1] == d.hidden);
    REQUIRE(bias.shape[0] == 3 * d.hidden);
    // The row offsets are NOT a convention chosen here. They are the ones the
    // W2 consumer slices at:
    // src/vllm/model_executor/models/deepseek_v4_vision.cpp reads
    // `RowSlice(layer.qkv_weight, 0, hidden)` as Q, `..., hidden, hidden` as K
    // and `..., 2 * hidden, hidden` as V, and the matching `VectorSlice`s for
    // the bias. Permuting q/k/v here would swap which projection each head
    // attends with and produce a fluent, wrong tower.
    const int families[3] = {Fam(kFamQ, l), Fam(kFamK, l), Fam(kFamV, l)};
    const int bias_families[3] = {Fam(kFamQBias, l), Fam(kFamKBias, l),
                                  Fam(kFamVBias, l)};
    for (int part = 0; part < 3; ++part) {
      for (int64_t i = 0; i < d.hidden * d.hidden; ++i) {
        CHECK(Word(qkv, part * d.hidden * d.hidden + i) ==
              vt::F32ToBF16(Series(families[part], i)));
        ++checked;
      }
      for (int64_t i = 0; i < d.hidden; ++i) {
        CHECK(Word(bias, part * d.hidden + i) ==
              vt::F32ToBF16(Series(bias_families[part], i)));
        ++checked;
      }
    }
  }
  // The loops ran: a bound that collapsed to zero leaves every CHECK above
  // unexecuted and the case still prints SUCCESS.
  CHECK(checked == d.depth * 3 * (d.hidden * d.hidden + d.hidden));
}

TEST_CASE("deepseek4v mmproj: (b) ffn_gate then ffn_up, in that row order") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj loaded =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);

  int64_t checked = 0;
  for (int64_t l = 0; l < d.depth; ++l) {
    const vt::Tensor& w1 = loaded.weights.blocks[static_cast<size_t>(l)].mlp_w1_weight;
    const vt::Tensor& w2 = loaded.weights.blocks[static_cast<size_t>(l)].mlp_w2_weight;
    REQUIRE(w1.shape[0] == 2 * d.inter);
    REQUIRE(w1.shape[1] == d.hidden);
    REQUIRE(w2.shape[0] == d.hidden);
    REQUIRE(w2.shape[1] == d.inter);
    // GATE FIRST, UP SECOND, and that order is read off the consumer rather
    // than guessed. W2 hands `mlp_w1_weight` to
    // `layers::UnquantizedMlpGateUpMethod`, whose `Apply` runs one `MatmulBT`
    // over the merged `[2I, H]` weight and then `vt::SiluAndMul`; the kernel
    // (src/vt/cpu/cpu_ops.cpp::SiluAndMulKernel) reads `gate` at column `j`
    // and `up` at column `d + j`, so the FIRST `intermediate` rows are the
    // gate. The pinned converter split the checkpoint's fused `mlp.w1` with
    // `gate, up = data_torch.chunk(2, dim=0)`, so `ffn_gate` is that first
    // chunk and this reader puts it back where it came from. Swapping the two
    // applies SiLU to the wrong projection and stays fluent.
    for (int64_t i = 0; i < d.inter * d.hidden; ++i) {
      CHECK(Word(w1, i) == vt::F32ToBF16(Series(Fam(kFamGate, l), i)));
      CHECK(Word(w1, d.inter * d.hidden + i) ==
            vt::F32ToBF16(Series(Fam(kFamUp, l), i)));
      CHECK(Word(w2, i) == vt::F32ToBF16(Series(Fam(kFamDown, l), i)));
      checked += 3;
    }
  }
  CHECK(checked == d.depth * 3 * d.inter * d.hidden);
}

TEST_CASE("deepseek4v mmproj: (c) the conv2d patch weight flattens in [channel, dy, dx] order") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj loaded =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);

  const vt::Tensor& w = loaded.weights.patch_weight;
  REQUIRE(w.rank == 2);
  REQUIRE(w.shape[0] == d.hidden);
  REQUIRE(w.shape[1] == d.patch_dim());

  // THE DERIVATION, which is why this is an identity flatten and not a
  // permutation. The pinned checkpoint's `vision.patch_embed.proj` is an
  // `nn.Linear` over patches flattened by `F.unfold`, whose element order is
  // [channel, dy, dx], so its weight is torch [hidden, C * p * p] in exactly
  // that column order. The pinned converter turns it into a conv2d weight with
  // a pure VIEW -- `data_torch.reshape(data_torch.shape[0], 3, p, p)`, which
  // moves no byte -- and llama.cpp then stores it in ggml dim order
  // {p, p, C, out}. `GgufTensorInfo::shape` reverses that back to torch
  // [out, C, p, p], whose row-major flattening is [channel, dy, dx] again. So
  // the correct map is the identity, and any transpose of it (the [dy, dx, c]
  // order a naive conv2d reading would produce) is a fluent, wrong tower.
  int64_t checked = 0;
  for (int64_t o = 0; o < d.hidden; ++o) {
    for (int64_t c = 0; c < d.channels; ++c) {
      for (int64_t dy = 0; dy < d.patch; ++dy) {
        for (int64_t dx = 0; dx < d.patch; ++dx) {
          const int64_t src = ((o * d.channels + c) * d.patch + dy) * d.patch + dx;
          const int64_t dst =
              o * d.patch_dim() + (c * d.patch + dy) * d.patch + dx;
          CHECK(Word(w, dst) == vt::F32ToBF16(Series(kFamPatchW, src)));
          ++checked;
        }
      }
    }
  }
  CHECK(checked == d.hidden * d.patch_dim());

  const vt::Tensor& b = loaded.weights.patch_bias;
  REQUIRE(b.rank == 1);
  REQUIRE(b.shape[0] == d.hidden);
  for (int64_t i = 0; i < d.hidden; ++i) {
    CHECK(Word(b, i) == vt::F32ToBF16(Series(kFamPatchB, i)));
  }
}

TEST_CASE("deepseek4v mmproj: (d) norms stay f32 and every linear takes the model dtype") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj loaded =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);

  // W2's contract (deepseek_v4_vision.h): RMSNorm weights are f32 because the
  // pinned module declares them f32 and widens x before the variance and the
  // affine; every linear weight and bias is `compute_dtype`. The file stores
  // ALL of them f32, so a reader that simply passed the file's dtype through
  // would widen every bias and the whole patch embedding, move twice the bytes
  // on the model path, and leave every token identical.
  CHECK(loaded.weights.patch_weight.dtype == vt::DType::kBF16);
  CHECK(loaded.weights.patch_bias.dtype == vt::DType::kBF16);
  CHECK(loaded.weights.final_norm_weight.dtype == vt::DType::kF32);
  CHECK(loaded.weights.aligner_w1_weight.dtype == vt::DType::kBF16);
  CHECK(loaded.weights.aligner_w1_bias.dtype == vt::DType::kBF16);
  CHECK(loaded.weights.aligner_w2_weight.dtype == vt::DType::kBF16);
  CHECK(loaded.weights.aligner_w2_bias.dtype == vt::DType::kBF16);
  for (const auto& block : loaded.weights.blocks) {
    CHECK(block.norm1_weight.dtype == vt::DType::kF32);
    CHECK(block.norm2_weight.dtype == vt::DType::kF32);
    CHECK(block.qkv_weight.dtype == vt::DType::kBF16);
    CHECK(block.qkv_bias.dtype == vt::DType::kBF16);
    CHECK(block.out_weight.dtype == vt::DType::kBF16);
    CHECK(block.out_bias.dtype == vt::DType::kBF16);
    CHECK(block.mlp_w1_weight.dtype == vt::DType::kBF16);
    CHECK(block.mlp_w2_weight.dtype == vt::DType::kBF16);
  }
  // The norm VALUES stay exactly what the file holds, undegraded by a
  // round trip through bf16.
  for (int64_t l = 0; l < d.depth; ++l) {
    const auto& block = loaded.weights.blocks[static_cast<size_t>(l)];
    for (int64_t i = 0; i < d.hidden; ++i) {
      CHECK(F32At(block.norm1_weight, i) == Series(Fam(kFamLn1, l), i));
      CHECK(F32At(block.norm2_weight, i) == Series(Fam(kFamLn2, l), i));
    }
  }
  for (int64_t i = 0; i < d.hidden; ++i) {
    CHECK(F32At(loaded.weights.final_norm_weight, i) == Series(kFamPostLn, i));
  }
}

TEST_CASE("deepseek4v mmproj: the aligner and the four sentinel vectors land in their own slots") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj loaded =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);

  // `mm.1` consumes the 3x3 unfold of the tower, so its input width is
  // hidden * ratio^2; `mm.2` closes onto the projection dim.
  REQUIRE(loaded.weights.aligner_w1_weight.shape[0] == d.output);
  REQUIRE(loaded.weights.aligner_w1_weight.shape[1] == d.aligner_in());
  REQUIRE(loaded.weights.aligner_w2_weight.shape[0] == d.output);
  REQUIRE(loaded.weights.aligner_w2_weight.shape[1] == d.output);
  // EVERY element, not element 0. `mm.2.weight` is the one SQUARE linear in
  // this projector -- [4096, 4096] on the real artifact -- so a torch/ggml
  // row-versus-column confusion there survives both shape REQUIREs above and
  // leaves element 0 unchanged, because index 0 is the one element a transpose
  // fixes. Only an OFF-DIAGONAL element can see it, and the series gives every
  // flat index its own bf16 word.
  int64_t checked = 0;
  for (int64_t i = 0; i < d.output * d.aligner_in(); ++i) {
    CHECK(Word(loaded.weights.aligner_w1_weight, i) ==
          vt::F32ToBF16(Series(kFamMm1W, i)));
    ++checked;
  }
  for (int64_t i = 0; i < d.output * d.output; ++i) {
    CHECK(Word(loaded.weights.aligner_w2_weight, i) ==
          vt::F32ToBF16(Series(kFamMm2W, i)));
    ++checked;
  }
  for (int64_t i = 0; i < d.output; ++i) {
    CHECK(Word(loaded.weights.aligner_w1_bias, i) ==
          vt::F32ToBF16(Series(kFamMm1B, i)));
    CHECK(Word(loaded.weights.aligner_w2_bias, i) ==
          vt::F32ToBF16(Series(kFamMm2B, i)));
    checked += 2;
  }
  // A bound that collapsed to zero leaves every CHECK above unexecuted and the
  // case still prints SUCCESS.
  CHECK(checked == d.output * d.aligner_in() + d.output * d.output + 2 * d.output);
  // Named explicitly, because the loop above would also pass on a matrix that
  // happened to be symmetric: these two are a transposed PAIR, so one CHECK
  // that they differ states in the test what the loop is protecting.
  CHECK(Word(loaded.weights.aligner_w2_weight, 1) !=
        Word(loaded.weights.aligner_w2_weight, d.output));

  // The sentinels stay f32, which is the dtype the file holds and the dtype
  // llama.cpp concatenates them at. W2 declares no dtype for them because it
  // has no field for them; W4 owns where they are placed, so narrowing here
  // would be a dtype decision made by the wrong wave.
  REQUIRE(loaded.image_start.size() == static_cast<size_t>(d.output));
  REQUIRE(loaded.image_end.size() == static_cast<size_t>(d.output));
  REQUIRE(loaded.image_pad.size() == static_cast<size_t>(d.output));
  REQUIRE(loaded.image_newline.size() == static_cast<size_t>(d.output));
  for (int64_t i = 0; i < d.output; ++i) {
    const size_t u = static_cast<size_t>(i);
    CHECK(loaded.image_start[u] == Series(kFamImgStart, i));
    CHECK(loaded.image_end[u] == Series(kFamImgEnd, i));
    CHECK(loaded.image_pad[u] == Series(kFamImgPad, i));
    CHECK(loaded.image_newline[u] == Series(kFamNewline, i));
  }
}

TEST_CASE("deepseek4v mmproj: the tensor map closes in BOTH directions") {
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);

  // 2 patch + depth * 13 + 1 post_ln + 4 aligner + 4 sentinels. On the real
  // artifact (depth 32) that is 427, which is exactly its tensor count.
  const std::vector<std::string> want =
      vllm::DeepSeekV4ClipMmprojExpectedTensors(cfg);
  CHECK(want.size() == static_cast<size_t>(2 + d.depth * 13 + 1 + 4 + 4));
  CHECK(want.size() == gguf.Tensors().size());
  DeepSeekV4VisionConfig real = cfg;
  real.depth = 32;
  CHECK(vllm::DeepSeekV4ClipMmprojExpectedTensors(real).size() == 427);
  CHECK(ThrownBy(Build(d), /*load_weights=*/true).empty());
}

TEST_CASE("deepseek4v mmproj: a tensor the reader never reads is refused, not dropped") {
  const Dims d;
  Options o;
  // A LEARNED position embedding. A qwen3vl-style export carries one and this
  // tower is RoPE, so a file that had both would otherwise load fine and place
  // every patch at the wrong position: a tower that runs and is wrong.
  o.stray_tensor = "v.position_embd.weight";
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/false);
  CHECK(Contains(message, "v.position_embd.weight"));
  CHECK(Contains(message, "NEVER reads"));
}

TEST_CASE("deepseek4v mmproj: the FUSED attn_qkv arm is refused BY NAME, not blamed on the file") {
  // THE FILE A USER GETS FROM THE ORACLE'S OWN CONVERTER. At the pin,
  // `gguf-py/gguf/tensor_mapping.py` maps `vision.blocks.{bid}.attn.wqkv` to
  // V_ENC_ATTN_QKV and `gguf-py/gguf/constants.py` spells that
  // `v.blk.{bid}.attn_qkv`. Nothing splits it for this family:
  // `conversion/base.py` contains no occurrence of `qkv` at all, the only
  // converter that splits a fused vision qkv is the model-specific
  // `conversion/qwenvl.py`, and
  // `conversion/deepseek.py::DeepseekV4FlashVisionModel.modify_tensors` splits
  // `mlp.w1` only. So `convert_hf_to_gguf.py` emits the FUSED form, 299
  // tensors at depth 32, and this build reads the SPLIT form only.
  //
  // Without the named refusal this file falls through to the unaccounted-tensor
  // refusal, which reports that the FILE carries tensors the reader never
  // reads. That blames the artifact for a gap in this build, and it sends the
  // reader to re-convert a file that is already correct.
  const Dims d;
  Options o;
  o.fused_qkv = true;
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/false);
  CHECK(Contains(message, "attn_qkv"));
  CHECK(Contains(message, "NOT IMPLEMENTED"));
  CHECK(Contains(message, "2411"));
  // The refusal has to arrive BEFORE the unaccounted-tensor one, or the user
  // reads the wrong diagnosis.
  CHECK(!Contains(message, "NEVER reads"));
}

TEST_CASE("deepseek4v mmproj: a projector that declares NO clip.use_silu is refused") {
  // Absent is not "SwiGLU by omission". `tools/mtmd/clip.cpp` at the pin
  // defaults to FFN_GELU_QUICK when neither `use_gelu` nor `use_silu` is set,
  // and W2's MLP is SwiGLU by construction, so a file that states nothing is a
  // file this reader cannot honour. The pinned converter always writes the key,
  // so a projector missing it was not produced by it.
  const Dims d;
  Options o;
  o.emit_use_silu = false;
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/false);
  CHECK(Contains(message, "clip.use_silu"));
}

TEST_CASE("deepseek4v mmproj: a wrong-shaped tensor names BOTH shapes") {
  // The shape guard is what makes the identity patch permutation safe against a
  // mis-read ggml/torch dim convention, and it is a memory-safety boundary
  // besides: a `want` larger than the tensor's numel would publish a
  // `HostView` over a short buffer. A transposed `ffn_gate` keeps the numel
  // identical, so nothing except this guard can see it.
  const Dims d;
  Options o;
  o.transpose_tensor = "v.blk.0.ffn_gate.weight";
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/true);
  CHECK(Contains(message, "v.blk.0.ffn_gate.weight"));
  CHECK(Contains(message, "is [8, 6]"));
  CHECK(Contains(message, "expected [6, 8]"));
}

TEST_CASE("deepseek4v mmproj: an out-of-range block_count is refused BY NAME") {
  // `clip.vision.block_count` becomes a `resize` argument and a loop bound.
  // `KvInt` widens every integer spelling, so a signed one can be negative and
  // an unsigned one can be four billion; both reach `std::vector::resize` as a
  // `size_t`.
  //
  // NOT, TODAY, ON A USER-SUPPLIED `--mmproj`, and `750cc6626`'s body said it
  // was. Nothing in `src/`, `include/`, `examples/` or `tools/` calls
  // `DeepSeekV4ClipMmprojVisionConfig`: this arm is a staged slice that W4 owns
  // the wiring for, so the untrusted input reaches it through this test and
  // nowhere else. The guard is still right -- W4 is what makes the sentence true
  // -- but the arm where it is ALREADY true is the production-reachable Qwen3-VL
  // `ClipMmprojVisionConfig` beside it, and
  // https://github.com/mudler/vllm.cpp/issues/2995 owns that one.
  //
  // THE VALUES HERE ARE DELIBERATELY SMALL, and that is the point rather than
  // a convenience. A red-first case for an unbounded allocation performs the
  // allocation by construction: `block_count = 4000000000` asks for about 80 GB
  // of blocks, and on this box it tripped the GLOBAL Linux OOM killer twice
  // ("Out of memory: Killed process (test_deepseek_v) anon-rss:80197996kB")
  // rather than reporting anything. A test whose only failure mode is
  // `bad_alloc` is a crash, not a gate. So the guard is asserted on the PARSED
  // VALUE: `4096` is absurd for a vision tower the artifact ships at depth 32,
  // it is refused by name, and WITHOUT the guard it allocates a few megabytes
  // and then fails these CHECKs on the message instead of taking the machine
  // down.
  const Dims d;
  Options negative;
  negative.geometry_kv["clip.vision.block_count"] =
      I32Kv("clip.vision.block_count", -1);
  const std::string neg = ThrownBy(Build(d, negative), /*load_weights=*/false);
  CAPTURE(neg);
  CHECK(RefusedByBound(neg, "clip.vision.block_count", -1));

  Options huge;
  huge.geometry_kv["clip.vision.block_count"] =
      gguf_test::U32Kv("clip.vision.block_count", 4096U);
  const std::string big = ThrownBy(Build(d, huge), /*load_weights=*/false);
  CAPTURE(big);
  CHECK(RefusedByBound(big, "clip.vision.block_count", 4096));

  // The other geometry keys are the same class of input and the same class of
  // consequence: a zero `embedding_length` makes every `Require` shape `[0, 0]`
  // and a tower of empty matrices runs and is wrong.
  Options zero_embd;
  zero_embd.geometry_kv["clip.vision.embedding_length"] =
      gguf_test::U32Kv("clip.vision.embedding_length", 0U);
  const std::string zero = ThrownBy(Build(d, zero_embd), /*load_weights=*/false);
  CAPTURE(zero);
  CHECK(RefusedByBound(zero, "clip.vision.embedding_length", 0));
}

// THE OTHER FIVE BOUNDS, which the case above did not hold. A fresh review
// deleted `RequireGeometry` from `head_count`, `feed_forward_length`,
// `projection_dim`, `projector.scale_factor` and `patch_size` -- all five at
// once -- and the suite stayed green at 18 cases and 2198 assertions. The reader
// bounded seven fields and two of them were gated, so the spec and the commit
// that said every field is bounded were true of the code and false of the gate.
//
// Two of the five are worse than "runs and is wrong". `projector.scale_factor`
// at 0 divides by zero in `DeepSeekV4VisionConfig::aligned_rows`, and
// `head_count` at 0 divides by zero in `head_dim()`. The other three build a
// tower of empty matrices that runs and produces fluent nonsense.
//
// Both ends of every bound, because a case for 0 alone leaves the ceiling free
// to be widened to anything. `1 << 21` is above `kMaxGeometry`, which is
// `1 << 20`; if that constant is ever raised past this value the over case stops
// throwing and reds here, which is the argument somebody should have to make.
// BOUNDING THE FACTORS IS NOT BOUNDING THE PRODUCT, and the difference is a
// `std::bad_alloc` on the path the header calls user-supplied.
//
// `kMaxGeometry` admits an `embedding_length` of 65536, which is a sixteenth of
// what it allows. `3 * hidden * hidden` at that width is 12,884,901,888
// elements, and the loader reserves that BEFORE the first file-shaped read, so
// nothing about the file bounds it; at the permitted maximum it asks for about
// 6.6 TB. This file is about 1.6 MB. It declares 65536 at `patch_size` 1 and
// carries only the four tensors the loader reads before that point, and before
// the product bound it passed `RefuseUnsupportedDeepSeekV4ClipMmproj`, passed
// every `RequireGeometry`, and threw a bare `std::bad_alloc` naming neither the
// file nor the key -- exactly what
// `include/vllm/model_executor/models/clip_mmproj_gguf.h` promises cannot
// happen. Third time in this row for this defect class, one multiplication
// removed from the bound each time.
TEST_CASE("deepseek4v mmproj: a geometry whose PRODUCT is absurd is refused BY NAME") {
  Dims d;
  d.hidden = 65536;  // inside kMaxGeometry, and 3 * hidden^2 is 12.9G elements
  d.patch = 1;       // keeps the file that this case writes at about 1.6 MB
  d.depth = 1;
  Options minimal;
  minimal.only_before_qkv = true;
  const std::string bytes = Build(d, minimal);
  // The file is small. It is the DECLARED geometry that is not.
  CHECK(bytes.size() < 4u * 1024u * 1024u);

  const std::string message = ThrownBy(bytes, /*load_weights=*/true);
  CAPTURE(message);
  // Named by the keys that multiplied into it, and refused on the PARSED VALUES
  // before anything is reserved -- so a regression is this assertion rather than
  // a machine death.
  CHECK(Contains(message, "clip.vision.embedding_length"));
  CHECK(Contains(message, "per tensor"));
  CHECK(!Contains(message, "bad_alloc"));
  CHECK(!message.empty());
}

TEST_CASE("deepseek4v mmproj: every clip.* geometry key is bounded BY NAME") {
  const Dims d;
  const char* keys[] = {
      "clip.vision.attention.head_count",
      "clip.vision.feed_forward_length",
      "clip.vision.projection_dim",
      "clip.vision.projector.scale_factor",
      "clip.vision.patch_size",
  };
  constexpr int64_t kAboveMaxGeometry = 1 << 21;
  for (const char* key : keys) {
    CAPTURE(key);

    // Absent in effect. A declared 0 is not a smaller tower, it is no tower.
    Options zero;
    zero.geometry_kv[key] = gguf_test::U32Kv(key, 0U);
    const std::string absent = ThrownBy(Build(d, zero), /*load_weights=*/false);
    CAPTURE(absent);
    CHECK(RefusedByBound(absent, key, 0));

    // Absurd. Refused on the PARSED VALUE, before anything is sized from it,
    // for the reason the block_count case above records at length.
    Options over;
    over.geometry_kv[key] =
        gguf_test::U32Kv(key, static_cast<uint32_t>(kAboveMaxGeometry));
    const std::string big = ThrownBy(Build(d, over), /*load_weights=*/false);
    CAPTURE(big);
    CHECK(RefusedByBound(big, key, kAboveMaxGeometry));
  }
}

TEST_CASE("deepseek4v mmproj: the attention OUTPUT projection lands value-exact in its own slot") {
  // `attn_out` is 32 x 1M parameters on the real artifact and it is the only
  // block tensor with no join to undo, which is exactly why it is easy to leave
  // unmeasured. Sourcing it from `attn_q` instead produces a tower that runs
  // and is fluent, so it gets its own exponent families and its own walk.
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  const DeepSeekV4VisionConfig cfg = vllm::DeepSeekV4ClipMmprojVisionConfig(gguf);
  const vllm::DeepSeekV4ClipMmproj loaded =
      vllm::LoadDeepSeekV4VisionFromClipMmproj(gguf, cfg);

  REQUIRE(loaded.weights.blocks.size() == static_cast<size_t>(d.depth));
  int64_t checked = 0;
  for (int64_t l = 0; l < d.depth; ++l) {
    const auto& block = loaded.weights.blocks[static_cast<size_t>(l)];
    REQUIRE(block.out_weight.rank == 2);
    REQUIRE(block.out_weight.shape[0] == d.hidden);
    REQUIRE(block.out_weight.shape[1] == d.hidden);
    REQUIRE(block.out_bias.rank == 1);
    REQUIRE(block.out_bias.shape[0] == d.hidden);
    for (int64_t i = 0; i < d.hidden * d.hidden; ++i) {
      CHECK(Word(block.out_weight, i) ==
            vt::F32ToBF16(Series(Fam(kFamOutW, l), i)));
      ++checked;
    }
    for (int64_t i = 0; i < d.hidden; ++i) {
      CHECK(Word(block.out_bias, i) ==
            vt::F32ToBF16(Series(Fam(kFamOutB, l), i)));
      ++checked;
    }
  }
  CHECK(checked == d.depth * (d.hidden * d.hidden + d.hidden));
}

TEST_CASE("deepseek4v mmproj: a missing tensor names itself") {
  const Dims d;
  Options o;
  o.omit_tensor = "v.blk.1.ffn_gate.weight";
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/true);
  CHECK(Contains(message, "v.blk.1.ffn_gate.weight"));
}

TEST_CASE("deepseek4v mmproj: a GELU-MLP projector is refused rather than run as SwiGLU") {
  const Dims d;
  Options o;
  o.use_silu = false;
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/false);
  // The tower's MLP is SwiGLU by construction (W2 composes gate/up through
  // `MlpGateUpMethodBase`), and the converter writes `clip.use_silu = true` for
  // exactly that reason. A projector declaring otherwise has an activation this
  // tower cannot express, and running it anyway is fluent and wrong.
  CHECK(Contains(message, "clip.use_silu"));
}

TEST_CASE("deepseek4v mmproj: another family's projector is refused BY NAME") {
  const Dims d;
  Options o;
  o.projector_type = "qwen3vl_merger";
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/false);
  CHECK(Contains(message, "qwen3vl_merger"));
  CHECK(Contains(message, "deepseek4v"));
}

TEST_CASE("deepseek4v mmproj: the Qwen3-VL production refusal is UNCHANGED") {
  // The Qwen3-VL gate in `model_loader.cpp` calls `RefuseUnsupportedClipMmproj`
  // and then goes straight into `LoadQwen3VLVisionFromClipMmproj`. Widening
  // that refusal to admit `deepseek4v` would route this file into the Qwen3-VL
  // reader, so it must keep refusing, and it must keep saying which projector
  // type it does load.
  const Dims d;
  TempFile file(Build(d));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  std::string message;
  try {
    vllm::RefuseUnsupportedClipMmproj(gguf, file.path());
  } catch (const std::exception& e) {
    message = e.what();
  }
  CHECK(Contains(message, "deepseek4v"));
  CHECK(Contains(message, "qwen3vl_merger"));
}

TEST_CASE("deepseek4v mmproj: a language GGUF passed as the projector is refused") {
  const Dims d;
  Options o;
  o.architecture = "deepseek4";
  const std::string message = ThrownBy(Build(d, o), /*load_weights=*/false);
  CHECK(Contains(message, "deepseek4"));
  CHECK(Contains(message, "clip"));
}
