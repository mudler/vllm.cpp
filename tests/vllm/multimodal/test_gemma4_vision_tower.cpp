// G2-impl — Gemma-4 NaFlex SigLIP2 vision-tower per-stage faithfulness gate.
// Runs the C++ Gemma4VisionForward (src/vllm/model_executor/models/gemma4_vision
// .cpp) on the committed G2 image-processor output and asserts each stage matches
// the transformers-eager reference within a stated bf16 tolerance:
//   patch_embed_out -> encoder_last_hidden -> pooled_stripped -> projected
// (the M2a rel-L2 ladder — a wrong stage fails its own ref, localizing the bug).
//
// Refs (VLLM_GEMMA4_VISION_REFS, default the committed vision_refs dir; the two
// intermediate refs are regenerated there by scripts/mm/g2_vision_ref_dump.py):
//   proc_pixel_values.npy [1,P,768] f32, proc_image_position_ids.npy [1,P,2] i64,
//   ref_patch_embedder.npy [1,P,768], ref_encoder_last_hidden.npy [1,P,768],
//   ref_pooled_stripped.npy [T,768], ref_projected.npy [T,2560].
// Weights (VLLM_GEMMA4_VISION_WEIGHTS): the f32 .bin dump from
// scripts/mm/g2_vision_weight_dump.py (~330 MiB, NOT committed). Without either
// env (or CUDA) the gate SKIPs, not fails.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "tests/parity/npy.h"
#include "vllm/model_executor/models/gemma4_vision.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::multimodal::Gemma4VisionBlockWeights;
using vllm::multimodal::Gemma4VisionCapture;
using vllm::multimodal::Gemma4VisionConfig;
using vllm::multimodal::Gemma4VisionWeights;

struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
};
Err Compare(const std::vector<float>& got, const std::vector<float>& ref, int64_t rows,
            int64_t cols, int64_t ref_cols_stride) {
  // Compare the leading `rows`xcols of got against ref (ref may have more rows;
  // padding rows are ignored). ref_cols_stride == cols here (no column padding).
  double num = 0.0, den = 0.0, mx = 0.0;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const double g = got[static_cast<size_t>(r) * cols + c];
      const double v = ref[static_cast<size_t>(r) * ref_cols_stride + c];
      const double d = g - v;
      num += d * d;
      den += v * v;
      mx = std::max(mx, std::abs(d));
    }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

std::vector<float> AsF32(const parity::NpyArray& a) {
  const float* p = reinterpret_cast<const float*>(a.data.data());
  return std::vector<float>(p, p + a.data.size() / sizeof(float));
}
std::vector<int64_t> AsI64(const parity::NpyArray& a) {
  const int64_t* p = reinterpret_cast<const int64_t*>(a.data.data());
  return std::vector<int64_t>(p, p + a.data.size() / sizeof(int64_t));
}

std::vector<float> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open weight: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

// Load a weight, trying candidate rel-names (Gemma4ClippableLinear stores the
// param under `<name>.linear.weight`; a plain nn.Linear under `<name>.weight`).
std::vector<float> WTry(const std::string& dir, const std::vector<std::string>& names) {
  for (const auto& nm : names) {
    std::ifstream f(dir + "/" + nm + ".bin", std::ios::binary);
    if (f.good()) return ReadBin(dir + "/" + nm + ".bin");
  }
  REQUIRE_MESSAGE(false, "no weight found for ", names.front());
  return {};
}

// A single clip scalar (defaults to no-op +/-inf when the .bin is absent).
// The same dumped f32 weight, narrowed to the tower's bf16 store (#1359). Not a
// precision change: `Gemma4VisionForward` put every one of these through
// `vt::F32ToBF16` in `MakeDevBf16` before its first GEMM, so the device bytes are
// the ones this dump always produced. `position_embedding_table` is the one
// weight that stays f32 and therefore keeps using `WTry` — the forward SUMS its
// x and y rows on the host before narrowing.
std::vector<uint16_t> WTryBits(const std::string& dir,
                               const std::vector<std::string>& names) {
  const std::vector<float> f = WTry(dir, names);
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}
float ClipVal(const std::string& dir, const std::string& name, float dflt) {
  std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
  if (!f.good()) return dflt;
  float v = dflt;
  f.read(reinterpret_cast<char*>(&v), sizeof(float));
  return v;
}
vllm::multimodal::Clip LoadClip(const std::string& dir, const std::string& prefix) {
  vllm::multimodal::Clip c;
  c.in_min = ClipVal(dir, prefix + ".input_min", -3.4e38f);
  c.in_max = ClipVal(dir, prefix + ".input_max", 3.4e38f);
  c.out_min = ClipVal(dir, prefix + ".output_min", -3.4e38f);
  c.out_max = ClipVal(dir, prefix + ".output_max", 3.4e38f);
  return c;
}

}  // namespace

TEST_CASE("gemma4_vision_tower_faithful_vs_transformers") {
  const char* wdir_c = std::getenv("VLLM_GEMMA4_VISION_WEIGHTS");
  if (wdir_c == nullptr) {
    MESSAGE("SKIP: set VLLM_GEMMA4_VISION_WEIGHTS (scripts/mm/g2_vision_weight_dump.py)");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const std::string wdir = wdir_c;
  const char* rdir_c = std::getenv("VLLM_GEMMA4_VISION_REFS");
  const std::string rdir = rdir_c != nullptr ? rdir_c : std::string(GEMMA4_VISION_REF_DIR);

  Gemma4VisionConfig cfg;  // E4B vision defaults (header).
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size, hd = cfg.head_dim;

  // --- weights --------------------------------------------------------------
  Gemma4VisionWeights w;
  w.input_proj = WTryBits(wdir, {"patch_embedder.input_proj.linear.weight",
                             "patch_embedder.input_proj.weight"});
  w.position_embedding_table = WTry(wdir, {"patch_embedder.position_embedding_table"});
  w.embed_projection = WTryBits(wdir, {"embed_vision.embedding_projection.linear.weight",
                                   "embed_vision.embedding_projection.weight"});
  w.blocks.resize(static_cast<size_t>(cfg.depth));
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string p = "encoder.layers." + std::to_string(l);
    Gemma4VisionBlockWeights& b = w.blocks[static_cast<size_t>(l)];
    b.input_ln = WTryBits(wdir, {p + ".input_layernorm.weight"});
    b.post_attn_ln = WTryBits(wdir, {p + ".post_attention_layernorm.weight"});
    b.pre_ff_ln = WTryBits(wdir, {p + ".pre_feedforward_layernorm.weight"});
    b.post_ff_ln = WTryBits(wdir, {p + ".post_feedforward_layernorm.weight"});
    auto lin = [&](const std::string& nm) {
      return WTryBits(wdir, {p + ".self_attn." + nm + ".linear.weight", p + ".self_attn." + nm + ".weight"});
    };
    b.q_proj = lin("q_proj");
    b.k_proj = lin("k_proj");
    b.v_proj = lin("v_proj");
    b.o_proj = lin("o_proj");
    b.q_norm = WTryBits(wdir, {p + ".self_attn.q_norm.weight"});
    b.k_norm = WTryBits(wdir, {p + ".self_attn.k_norm.weight"});
    auto mlp = [&](const std::string& nm) {
      return WTryBits(wdir, {p + ".mlp." + nm + ".linear.weight", p + ".mlp." + nm + ".weight"});
    };
    b.gate_proj = mlp("gate_proj");
    b.up_proj = mlp("up_proj");
    b.down_proj = mlp("down_proj");
    b.q_clip = LoadClip(wdir, p + ".self_attn.q_proj");
    b.k_clip = LoadClip(wdir, p + ".self_attn.k_proj");
    b.v_clip = LoadClip(wdir, p + ".self_attn.v_proj");
    b.o_clip = LoadClip(wdir, p + ".self_attn.o_proj");
    b.gate_clip = LoadClip(wdir, p + ".mlp.gate_proj");
    b.up_clip = LoadClip(wdir, p + ".mlp.up_proj");
    b.down_clip = LoadClip(wdir, p + ".mlp.down_proj");
  }
  REQUIRE(static_cast<int64_t>(w.input_proj.size()) == H * cfg.patch_dim());
  REQUIRE(static_cast<int64_t>(w.blocks[0].q_norm.size()) == hd);
  REQUIRE(static_cast<int64_t>(w.blocks[0].gate_proj.size()) == I * H);

  // --- inputs + refs --------------------------------------------------------
  parity::NpyArray pv_a = parity::LoadNpy(rdir + "/proc_pixel_values.npy");
  parity::NpyArray pid_a = parity::LoadNpy(rdir + "/proc_image_position_ids.npy");
  std::vector<float> pixel_values = AsF32(pv_a);         // [1,P,768]
  std::vector<int64_t> position_ids = AsI64(pid_a);      // [1,P,2]
  const int64_t P = pv_a.shape[1];

  Gemma4VisionCapture cap;
  std::vector<float> projected =
      vllm::multimodal::Gemma4VisionForward(pixel_values, position_ids, P, w, cfg, *gpu, &cap);

  const int64_t n_valid = static_cast<int64_t>(cap.patch_embed_out.size()) / H;
  const int64_t n_soft = static_cast<int64_t>(cap.pooled.size()) / H;
  MESSAGE("n_valid=" << n_valid << " n_soft=" << n_soft);

  // --- Tolerance model (M2a bf16-depth envelope). Patch-embed is a single GEMM
  // + host pos-embed -> TIGHT; the encoder is 16 sandwich blocks of two
  // independent bf16 kernel stacks (our vt ops vs transformers eager) -> a
  // smooth accumulation envelope; pooler/projector inherit + one GEMM. RED
  // injection (a wrong op) drives each far past its bound.
  {
    parity::NpyArray r = parity::LoadNpy(rdir + "/ref_patch_embedder.npy");  // [1,P,768]
    Err e = Compare(cap.patch_embed_out, AsF32(r), n_valid, H, H);
    MESSAGE("patch_embed relL2=" << e.rel_l2 << " maxabs=" << e.max_abs);
    CHECK(e.rel_l2 < 5e-3);
  }
  {
    parity::NpyArray r = parity::LoadNpy(rdir + "/ref_encoder_last_hidden.npy");  // [1,P,768]
    Err e = Compare(cap.encoder_out, AsF32(r), n_valid, H, H);
    MESSAGE("encoder relL2=" << e.rel_l2 << " maxabs=" << e.max_abs);
    CHECK(e.rel_l2 < 6e-2);
  }
  {
    parity::NpyArray r = parity::LoadNpy(rdir + "/ref_pooled_stripped.npy");  // [T,768]
    REQUIRE(r.shape[0] == n_soft);
    Err e = Compare(cap.pooled, AsF32(r), n_soft, H, H);
    MESSAGE("pooled relL2=" << e.rel_l2 << " maxabs=" << e.max_abs);
    CHECK(e.rel_l2 < 6e-2);
  }
  {
    parity::NpyArray r = parity::LoadNpy(rdir + "/ref_projected.npy");  // [T,2560]
    REQUIRE(r.shape[0] == n_soft);
    Err e = Compare(projected, AsF32(r), n_soft, cfg.text_hidden_size, cfg.text_hidden_size);
    MESSAGE("projected relL2=" << e.rel_l2 << " maxabs=" << e.max_abs);
    CHECK(e.rel_l2 < 7e-2);
  }
}
