// MiniMax-H3 parity gate. Every assertion here compares our port against the
// UPSTREAM vLLM-Omni implementation (vllm-project/vllm-omni,
// vllm_omni/diffusion/models/minimax_h3/), executed at reduced dimensions by
// scripts/gen-minimax-h3-goldens.py and frozen into minimax_h3_goldens.inc.
//
// WHY A REDUCED-DIMENSION GATE. The shipped H3 checkpoint is ~354 GB and its
// validated serving config is 4x NVIDIA B300 (~133 GB peak per rank); it does not
// fit this project's hardware, so nothing here claims an end-to-end video result.
// What it does claim is exact: the packed layout (including the FP64 position grid
// that feeds RoPE), the flow-matching scheduler, the latent<->token packing, and
// the full DiT forward all reproduce upstream's numbers. See
// .agents/specs/minimax-h3.md sections 0 and 4.
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // W-FP4a: W4A16 exec stats
#include "vllm/model_executor/models/minimax_h3.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "minimax_h3_goldens.inc"
#include "minimax_h3_gguf_manifest.inc"
#include "minimax_h3_pruned_gguf_manifest.inc"
#include "minimax_h3_adaln_curve_goldens.inc"
#include "minimax_h3_audio_vae_goldens.inc"
#include "minimax_h3_audio_vae_encoder_goldens.inc"
#include "minimax_h3_audio_vae_manifest.inc"
#include "minimax_h3_vae_configs.inc"
#include "minimax_h3_nvfp4_manifest.inc"
#include "minimax_h3_video_vae_manifest.inc"
#include "minimax_h3_video_vae_goldens.inc"
#include "minimax_h3_encoder_goldens.inc"

#include "support/max_abs_diff.h"
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "../gguf_builder.h"

#include "vt/backend.h"
#include "vt/quant.h"
#include "vt/device.h"
#include "vt/tensor.h"

using vllm::BuildMiniMaxH3PackedSequence;
using vllm::BuildMiniMaxH3PackedSequenceRef2va;
using vllm::EnumerateMiniMaxH3DitTensors;
using vllm::MiniMaxH3DitDeviceWeights;
using vllm::MiniMaxH3DitForward;
using vllm::MiniMaxH3DitForwardDevice;
using vllm::StageMiniMaxH3DitWeights;
using vllm::MiniMaxH3DitInputs;
using vllm::MiniMaxH3DitOutputs;
using vllm::MiniMaxH3DitParams;
using vllm::MiniMaxH3DitWeights;
using vllm::MiniMaxH3EulerEta0Step;
using vllm::MiniMaxH3PackAudioLatent;
using vllm::MiniMaxH3PackedSequence;
using vllm::MiniMaxH3PatchifyVideoLatent;
using vllm::MiniMaxH3RefBlock;
using vllm::MiniMaxH3ReorderGroupedQkv;
using vllm::MiniMaxH3RfVToX0;
using vllm::MiniMaxH3UnpackAudioTokens;
using vllm::MiniMaxH3UnpatchifyVideoTokens;
using vllm::ParseMiniMaxH3DitParams;

namespace {

// ---------------------------------------------------------------------------
// H3Rand — the exact mirror of the generator's deterministic stream
// (scripts/gen-minimax-h3-goldens.py :: h3_rand). A per-tensor FNV-1a seed plus a
// splitmix64 counter, so both sides build identical tensors without shipping a
// single weight byte.
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<double> H3Rand(const std::string& name, int64_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<double> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    const double unit = static_cast<double>(u >> 11) * 0x1p-53;
    out[static_cast<size_t>(i)] = unit * 2.0 - 1.0;
  }
  return out;
}

// make_param: h3_rand * scale + offset, rounded to f32 (the generator's astype).
std::vector<float> MakeParam(const std::string& name, int64_t count, double scale,
                             double offset = 0.0) {
  const std::vector<double> raw = H3Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(raw[static_cast<size_t>(i)] * scale + offset);
  }
  return out;
}

vt::Device Cpu() { return vt::Device{}; }

vt::Tensor View1D(std::vector<float>& buffer) {
  return vt::Tensor::Contiguous(buffer.data(), vt::DType::kF32, Cpu(),
                                {static_cast<int64_t>(buffer.size())});
}

vt::Tensor View2D(std::vector<float>& buffer, int64_t rows, int64_t cols) {
  REQUIRE(static_cast<int64_t>(buffer.size()) == rows * cols);
  return vt::Tensor::Contiguous(buffer.data(), vt::DType::kF32, Cpu(), {rows, cols});
}

// Max absolute difference against a golden array — the shared, NaN-hardened
// reduction. The local copy this replaces used `std::max(worst, ...)`, which is
// `a < b ? b : a`; `a < NaN` is false, so an all-NaN result against a correct
// golden reduced to 0.0 and passed every bound (issue #449).
using vllm_test::MaxAbsDiff;

template <typename T>
void CheckI64(const std::vector<T>& got, const int64_t* want, size_t count) {
  REQUIRE(got.size() == count);
  for (size_t i = 0; i < count; ++i) CHECK(static_cast<int64_t>(got[i]) == want[i]);
}

// The reduced-dimension arch the generator used (section 5 scalars).
MiniMaxH3DitParams GoldenParams() {
  MiniMaxH3DitParams p;
  p.num_layers = vllm_test::kH3Dit_num_layers;
  p.token_refiner_num_layers = vllm_test::kH3Dit_token_refiner_num_layers;
  p.hidden_size = vllm_test::kH3Dit_hidden_size;
  p.num_attention_heads = vllm_test::kH3Dit_num_attention_heads;
  p.attention_head_dim = vllm_test::kH3Dit_attention_head_dim;
  p.ffn_hidden_size = vllm_test::kH3Dit_ffn_hidden_size;
  p.latents_dim = vllm_test::kH3Dit_latents_dim;
  p.audio_latents_dim = vllm_test::kH3Dit_audio_latents_dim;
  p.patch_size_t = 1;
  p.patch_size_h = 2;
  p.patch_size_w = 2;
  p.text_dim = vllm_test::kH3Dit_text_dim;
  p.timestep_input_dim = vllm_test::kH3Dit_timestep_input_dim;
  p.time_embed_hidden_size = vllm_test::kH3Dit_time_embed_hidden_size;
  p.time_embed_dim = vllm_test::kH3Dit_time_embed_dim;
  p.adaln_out_features = 18 * p.hidden_size;
  p.final_adaln_out_features = 2 * p.hidden_size;
  p.rope_inv_freq_len = vllm_test::kH3Dit_rope_inv_freq_len;
  return p;
}

// Owns every reduced-dimension parameter so the vt::Tensor views stay valid.
struct GoldenWeights {
  MiniMaxH3DitParams params;
  std::vector<std::vector<float>> storage;
  MiniMaxH3DitWeights views;

  std::vector<float>& Add(const std::string& name, int64_t count, double scale,
                          double offset = 0.0) {
    storage.push_back(MakeParam(name, count, scale, offset));
    return storage.back();
  }
};

// Mirrors RefDiT.__init__ in the generator: same names, same scales, same shapes.
// storage is reserved up front because the vt::Tensor views are non-owning.
void BuildBlock(GoldenWeights& w, const std::string& prefix, bool with_adaln) {
  const MiniMaxH3DitParams& p = w.params;
  const int64_t h = p.hidden_size;
  const int64_t inner = p.num_attention_heads * p.attention_head_dim;
  vllm::MiniMaxH3DitBlockWeights block;
  block.norm1 = View1D(w.Add(prefix + ".norm1.weight", h, 0.1, 1.0));
  block.norm2 = View1D(w.Add(prefix + ".norm2.weight", h, 0.1, 1.0));
  block.qkv_proj =
      View2D(w.Add(prefix + ".attn.qkv_proj.weight", 3 * inner * h, 0.05), 3 * inner, h);
  block.q_norm = View1D(w.Add(prefix + ".attn.q_norm.weight", p.attention_head_dim, 0.1, 1.0));
  block.k_norm = View1D(w.Add(prefix + ".attn.k_norm.weight", p.attention_head_dim, 0.1, 1.0));
  block.out_proj = View2D(w.Add(prefix + ".attn.out_proj.weight", h * inner, 0.05), h, inner);
  block.fc1 = View2D(w.Add(prefix + ".mlp.fc1.weight", 2 * p.ffn_hidden_size * h, 0.05),
                     2 * p.ffn_hidden_size, h);
  block.fc2 = View2D(w.Add(prefix + ".mlp.fc2.weight", h * p.ffn_hidden_size, 0.05), h,
                     p.ffn_hidden_size);
  if (with_adaln) {
    block.adaln_w = View2D(
        w.Add(prefix + ".adaln_proj.linear.weight", p.adaln_out_features * p.time_embed_dim, 0.05),
        p.adaln_out_features, p.time_embed_dim);
    block.adaln_b = View1D(w.Add(prefix + ".adaln_proj.linear.bias", p.adaln_out_features, 0.02));
    w.views.blocks.push_back(block);
  } else {
    w.views.refiner.push_back(block);
  }
}

std::unique_ptr<GoldenWeights> BuildGoldenWeights(const MiniMaxH3DitParams& params) {
  auto w = std::make_unique<GoldenWeights>();
  w->params = params;
  const MiniMaxH3DitParams& p = w->params;
  const int64_t h = p.hidden_size;
  const int64_t video_width = p.video_row_width();
  // Reserve so no reallocation invalidates a previously taken view.
  w->storage.reserve(512);

  w->views.video_patch_proj_w =
      View2D(w->Add("video_patch_proj.weight", h * video_width, 0.05), h, video_width);
  w->views.video_patch_proj_b = View1D(w->Add("video_patch_proj.bias", h, 0.02));
  w->views.audio_patch_proj_w =
      View2D(w->Add("audio_patch_proj.weight", h * p.audio_latents_dim, 0.05), h, p.audio_latents_dim);
  w->views.audio_patch_proj_b = View1D(w->Add("audio_patch_proj.bias", h, 0.02));
  w->views.condition_proj_w =
      View2D(w->Add("condition_proj.weight", h * p.text_dim, 0.05), h, p.text_dim);
  w->views.condition_proj_b = View1D(w->Add("condition_proj.bias", h, 0.02));
  w->views.time_proj_in_w =
      View2D(w->Add("time_embedder.proj_in.weight", p.time_embed_hidden_size * p.timestep_input_dim,
                    0.05),
             p.time_embed_hidden_size, p.timestep_input_dim);
  w->views.time_proj_in_b =
      View1D(w->Add("time_embedder.proj_in.bias", p.time_embed_hidden_size, 0.02));
  w->views.time_proj_out_w = View2D(
      w->Add("time_embedder.proj_out.weight", p.time_embed_dim * p.time_embed_hidden_size, 0.05),
      p.time_embed_dim, p.time_embed_hidden_size);
  w->views.time_proj_out_b = View1D(w->Add("time_embedder.proj_out.bias", p.time_embed_dim, 0.02));

  // inv_freq = 10000^(-2i / 2L), computed (not random) on both sides.
  {
    std::vector<float> inv(static_cast<size_t>(p.rope_inv_freq_len));
    for (int64_t i = 0; i < p.rope_inv_freq_len; ++i) {
      inv[static_cast<size_t>(i)] = static_cast<float>(
          std::pow(10000.0, -(2.0 * static_cast<double>(i)) / (2.0 * static_cast<double>(p.rope_inv_freq_len))));
    }
    w->storage.push_back(std::move(inv));
    w->views.rope_inv_freq = View1D(w->storage.back());
  }

  for (int64_t i = 0; i < p.token_refiner_num_layers; ++i) {
    BuildBlock(*w, "token_refiner.blocks." + std::to_string(i), /*with_adaln=*/false);
  }
  w->views.refiner_final_norm = View1D(w->Add("token_refiner.final_norm.weight", h, 0.1, 1.0));
  for (int64_t i = 0; i < p.num_layers; ++i) {
    BuildBlock(*w, "blocks." + std::to_string(i), /*with_adaln=*/true);
  }
  w->views.final_norm = View1D(w->Add("final_layer.norm.weight", h, 0.1, 1.0));
  w->views.final_adaln_w =
      View2D(w->Add("final_layer.adaln_proj.linear.weight", p.final_adaln_out_features * p.time_embed_dim,
                    0.05),
             p.final_adaln_out_features, p.time_embed_dim);
  w->views.final_adaln_b =
      View1D(w->Add("final_layer.adaln_proj.linear.bias", p.final_adaln_out_features, 0.02));
  w->views.video_out_w =
      View2D(w->Add("final_layer.video_out.weight", video_width * h, 0.05), video_width, h);
  w->views.video_out_b = View1D(w->Add("final_layer.video_out.bias", video_width, 0.02));
  w->views.audio_out_w =
      View2D(w->Add("final_layer.audio_out.weight", p.audio_latents_dim * h, 0.05),
             p.audio_latents_dim, h);
  w->views.audio_out_b = View1D(w->Add("final_layer.audio_out.bias", p.audio_latents_dim, 0.02));
  return w;
}

// The section-5 reduced arch, kept as the default so every existing caller is
// unchanged.
std::unique_ptr<GoldenWeights> BuildGoldenWeights() { return BuildGoldenWeights(GoldenParams()); }

// A REAL-RATIO arch: the section-5 reduced dims keep hidden=64 / head_dim=16 /
// rope_inv_freq_len=2, none of which exercises the real checkpoint's head_dim=128
// (rot_dim=96) or its wide hidden state. #70's white latent is a real-scale
// phenomenon, so the ladder's device-vs-host diff must also run at the REAL head
// geometry, where a head_dim- or rope-scale assumption in a shared device op would
// make the device-resident forward diverge from the trusted host loops. Inputs
// (video_width=32, text_dim, audio_latents_dim) are held identical to the reduced
// arch so the same ladder cases feed both; only the internal widths grow.
MiniMaxH3DitParams RealRatioParams() {
  MiniMaxH3DitParams p = GoldenParams();
  p.hidden_size = 256;
  p.num_attention_heads = 2;
  p.attention_head_dim = 128;  // the real checkpoint's head_dim
  p.ffn_hidden_size = 512;
  p.time_embed_hidden_size = 128;
  p.rope_inv_freq_len = 16;  // rot_dim = 96 <= 128, the real ratio
  p.adaln_out_features = 18 * p.hidden_size;
  p.final_adaln_out_features = 2 * p.hidden_size;
  return p;
}

// The fl2va DiT forward case: the packed sequence, the scattered video/audio rows
// and the prompt embeddings the generator used. Owns every buffer the returned
// MiniMaxH3DitInputs points at, so both the CPU forward and the DEVICE forward can
// be driven from exactly the same bytes.
struct DitForwardCase {
  MiniMaxH3PackedSequence packed;
  std::vector<float> x, audio_x, prompt_embeds;
  std::vector<float> unique_timesteps;
  std::vector<int64_t> inverse;
  std::vector<int32_t> refiner_cu;
  int64_t seq_len = 0, num_img = 0, num_audio = 0, num_text = 0, video_width = 0;
  MiniMaxH3DitInputs in;
};

std::unique_ptr<DitForwardCase> BuildDitForwardCase(const MiniMaxH3DitParams& p) {
  auto c = std::make_unique<DitForwardCase>();
  c->packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);

  c->seq_len = c->packed.seq_len;
  c->video_width = p.video_row_width();
  c->num_img = static_cast<int64_t>(c->packed.img_pos.size());
  c->num_audio = static_cast<int64_t>(c->packed.audio_pos.size());
  c->num_text = static_cast<int64_t>(c->packed.text_pos.size());

  // Scatter the same rows the generator scattered.
  c->x.assign(static_cast<size_t>(c->seq_len * c->video_width), 0.0f);
  const std::vector<float> video_rows =
      MakeParam("dit.video_rows", c->num_img * c->video_width, 1.0);
  for (int64_t r = 0; r < c->num_img; ++r) {
    std::memcpy(c->x.data() + c->packed.img_pos[static_cast<size_t>(r)] * c->video_width,
                video_rows.data() + r * c->video_width,
                static_cast<size_t>(c->video_width) * sizeof(float));
  }
  c->audio_x.assign(static_cast<size_t>(c->seq_len * p.audio_latents_dim), 0.0f);
  const std::vector<float> audio_rows =
      MakeParam("dit.audio_rows", c->num_audio * p.audio_latents_dim, 1.0);
  for (int64_t r = 0; r < c->num_audio; ++r) {
    std::memcpy(
        c->audio_x.data() + c->packed.audio_pos[static_cast<size_t>(r)] * p.audio_latents_dim,
        audio_rows.data() + r * p.audio_latents_dim,
        static_cast<size_t>(p.audio_latents_dim) * sizeof(float));
  }
  c->prompt_embeds = MakeParam("dit.prompt_embeds", c->num_text * p.text_dim, 1.0);

  c->unique_timesteps.assign(
      vllm_test::kH3DitUniqueTimesteps,
      vllm_test::kH3DitUniqueTimesteps + vllm_test::kH3Dit_unique_timesteps);
  c->inverse.assign(vllm_test::kH3DitInverseIndices,
                    vllm_test::kH3DitInverseIndices + c->seq_len);
  c->refiner_cu = {0, static_cast<int32_t>(c->num_text), static_cast<int32_t>(c->num_text)};

  MiniMaxH3DitInputs& in = c->in;
  in.seq_len = c->seq_len;
  in.x = c->x.data();
  in.audio_x = c->audio_x.data();
  in.img_position_ids = c->packed.img_position_ids.data();
  in.unique_timesteps = c->unique_timesteps.data();
  in.num_unique_timesteps = static_cast<int64_t>(c->unique_timesteps.size());
  in.inverse_indices = c->inverse.data();
  in.token_tags = c->packed.token_tags.data();
  in.prompt_embeds = c->prompt_embeds.data();
  in.img_pos = c->packed.img_pos.data();
  in.num_img_pos = c->num_img;
  in.audio_pos = c->packed.audio_pos.data();
  in.num_audio_pos = c->num_audio;
  in.text_pos = c->packed.text_pos.data();
  in.num_text_pos = c->num_text;
  in.infer_out_pos = c->packed.img_pos.data();
  in.num_infer_out_pos = c->num_img;
  in.update_mask = c->packed.update_mask.data();
  in.cu_seqlens = c->packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(c->packed.cu_seqlens.size());
  in.refiner_cu_seqlens = c->refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(c->refiner_cu.size());
  return c;
}

// The synthetic checkpoint writers. `WriteMiniMaxH3Nvfp4File`'s callers set
// num_layers == token_refiner_num_layers == 1, which is what makes the
// quantized-GEMM count exactly 11 (refiner 4 + block 5 + condition + final).
//
// One serialized safetensors entry. Shared by the single-file NVFP4 writer and the
// MULTI-SHARD bf16 writer below, so both emit the SAME tensor set from ONE list --
// two copies of the ~30-name DiT layout would drift the moment a tensor is added.
struct H3StEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string PackF32(const std::vector<float>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

// Round-to-nearest-even, the rule vt uses on a bf16 store.
std::string PackBf16(const std::vector<float>& v) {
  std::string out(v.size() * sizeof(uint16_t), '\0');
  for (size_t i = 0; i < v.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &v[i], sizeof(bits));
    const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    const uint16_t half = static_cast<uint16_t>(rounded >> 16);
    std::memcpy(&out[i * sizeof(uint16_t)], &half, sizeof(half));
  }
  return out;
}

// Build the WHOLE DiT tensor set at geometry `want`.
//   quantize   -- projections become the NVFP4 triple (the single-file NVFP4 arm);
//                 otherwise every tensor is written plain.
//   plain_bf16 -- plain tensors are stored BF16, which is what the original bf16
//                 release does, unless their name is in `f32_names`. It is false
//                 with `quantize`, keeping the NVFP4 file byte-for-byte what it was.
std::vector<H3StEntry> BuildMiniMaxH3DitEntries(const MiniMaxH3DitParams& want, bool quantize,
                                                bool plain_bf16,
                                                const std::set<std::string>& f32_names) {
  using Entry = H3StEntry;
  std::vector<Entry> entries;

  auto add_plain = [&](const std::string& name, const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    const std::vector<float> values = MakeParam("nvfp4." + name, numel, 0.1);
    const bool as_f32 = !plain_bf16 || f32_names.count(name) != 0;
    entries.push_back({name, as_f32 ? "F32" : "BF16", shape,
                       as_f32 ? PackF32(values) : PackBf16(values)});
  };
  auto add_quant = [&](const std::string& name, int64_t out_dim, int64_t in_dim) {
    if (!quantize) {
      add_plain(name, {out_dim, in_dim});
      return;
    }
    REQUIRE(in_dim % 16 == 0);
    std::string packed(static_cast<size_t>(out_dim * (in_dim / 2)), '\0');
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = static_cast<char>((i * 37 + 11) & 0xFF);  // deterministic nibbles
    }
    std::string scales(static_cast<size_t>(out_dim * (in_dim / 16)), '\0');
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = static_cast<char>(0x38);  // e4m3 ~ 1.0
    }
    const float global = 0.5f;
    entries.push_back({name, "U8", {out_dim, in_dim / 2}, packed});
    entries.push_back({name + "_scale", "F8_E4M3", {out_dim, in_dim / 16}, scales});
    entries.push_back({name + "_scale_2", "F32", {},
                       std::string(reinterpret_cast<const char*>(&global), sizeof(float))});
  };

  const int64_t inner = want.num_attention_heads * want.attention_head_dim;
  const int64_t video_width = want.video_row_width();
  // Islands stay unquantized, exactly as the real checkpoint has them.
  add_plain("video_patch_proj.weight", {want.hidden_size, video_width});
  add_plain("video_patch_proj.bias", {want.hidden_size});
  add_plain("audio_patch_proj.weight", {want.hidden_size, want.audio_latents_dim});
  add_plain("audio_patch_proj.bias", {want.hidden_size});
  add_plain("condition_proj.bias", {want.hidden_size});
  add_plain("time_embedder.proj_in.weight", {want.time_embed_hidden_size, want.timestep_input_dim});
  add_plain("time_embedder.proj_in.bias", {want.time_embed_hidden_size});
  add_plain("time_embedder.proj_out.weight", {want.time_embed_dim, want.time_embed_hidden_size});
  add_plain("time_embedder.proj_out.bias", {want.time_embed_dim});
  add_plain("rope.inv_freq", {want.rope_inv_freq_len});
  add_quant("condition_proj.weight", want.hidden_size, want.text_dim);
  auto add_block = [&](const std::string& prefix, bool with_adaln) {
    add_plain(prefix + ".norm1.weight", {want.hidden_size});
    add_plain(prefix + ".norm2.weight", {want.hidden_size});
    add_plain(prefix + ".attn.q_norm.weight", {want.attention_head_dim});
    add_plain(prefix + ".attn.k_norm.weight", {want.attention_head_dim});
    add_quant(prefix + ".attn.qkv_proj.weight", 3 * inner, want.hidden_size);
    add_quant(prefix + ".attn.out_proj.weight", want.hidden_size, inner);
    add_quant(prefix + ".mlp.fc1.weight", 2 * want.ffn_hidden_size, want.hidden_size);
    add_quant(prefix + ".mlp.fc2.weight", want.hidden_size, want.ffn_hidden_size);
    if (with_adaln) {
      add_quant(prefix + ".adaln_proj.linear.weight", want.adaln_out_features, want.time_embed_dim);
      add_plain(prefix + ".adaln_proj.linear.bias", {want.adaln_out_features});
    }
  };
  for (int64_t i = 0; i < want.token_refiner_num_layers; ++i)
    add_block("token_refiner.blocks." + std::to_string(i), false);
  add_plain("token_refiner.final_norm.weight", {want.hidden_size});
  for (int64_t i = 0; i < want.num_layers; ++i)
    add_block("blocks." + std::to_string(i), true);
  add_plain("final_layer.norm.weight", {want.hidden_size});
  add_quant("final_layer.adaln_proj.linear.weight", want.final_adaln_out_features, want.time_embed_dim);
  add_plain("final_layer.adaln_proj.linear.bias", {want.final_adaln_out_features});
  add_plain("final_layer.video_out.weight", {video_width, want.hidden_size});
  add_plain("final_layer.video_out.bias", {video_width});
  add_plain("final_layer.audio_out.weight", {want.audio_latents_dim, want.hidden_size});
  add_plain("final_layer.audio_out.bias", {want.audio_latents_dim});
  return entries;
}

// Serialize `entries` as ONE .safetensors file.
void WriteSafetensorsFromEntries(const std::vector<H3StEntry>& entries, const std::string& path) {
  using Entry = H3StEntry;
  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const Entry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  FILE* fh = std::fopen(path.c_str(), "wb");
  REQUIRE(fh != nullptr);
  const uint64_t n = header.size();
  std::fwrite(&n, sizeof(n), 1, fh);
  std::fwrite(header.data(), 1, header.size(), fh);
  for (const Entry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
  std::fclose(fh);
}

// The synthetic single-file NVFP4 DiT (`lilcheaty/MiniMax-H3-NVFP4`'s layout):
// quantized projections as U8 packed [out, in/2] + E4M3 group-16 weight_scale +
// F32 weight_scale_2; islands (patch/time/output/norms) plain F32.
void WriteMiniMaxH3Nvfp4File(const MiniMaxH3DitParams& want, const std::string& path) {
  WriteSafetensorsFromEntries(
      BuildMiniMaxH3DitEntries(want, /*quantize=*/true, /*plain_bf16=*/false, {}), path);
}

// --- the ORIGINAL bf16 release's shape: N shards + model.safetensors.index.json --

std::string ShardFileName(size_t i, size_t n) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "model-%05zu-of-%05zu.safetensors", i + 1, n);
  return std::string(buf);
}

// Write `entries` as a MULTI-SHARD checkpoint under `dir`: contiguous chunks
// across `num_shards` files plus the index. Returns the name -> shard-file map
// the index promises, so the test can assert every tensor resolved to the shard
// it was actually written into.
//
// `omit_payload` names tensors the INDEX still lists but that are deliberately
// left OUT of their shard — a checkpoint that is corrupt in the one way that
// otherwise fails SILENTLY (a skipped weight reads as zeros and renders).
std::map<std::string, std::string> WriteMiniMaxH3ShardedDit(
    const std::vector<H3StEntry>& entries, const std::string& dir, size_t num_shards,
    const std::set<std::string>& omit_payload = {}) {
  REQUIRE(num_shards > 0);
  REQUIRE(entries.size() >= num_shards);
  ::mkdir(dir.c_str(), 0755);

  std::map<std::string, std::string> weight_map;
  std::vector<std::vector<H3StEntry>> per_shard(num_shards);
  for (size_t i = 0; i < entries.size(); ++i) {
    const size_t shard = i * num_shards / entries.size();
    weight_map[entries[i].name] = ShardFileName(shard, num_shards);
    if (omit_payload.count(entries[i].name) == 0) per_shard[shard].push_back(entries[i]);
  }
  for (size_t s = 0; s < num_shards; ++s) {
    WriteSafetensorsFromEntries(per_shard[s], dir + "/" + ShardFileName(s, num_shards));
  }

  nlohmann::json index;
  index["metadata"] = {{"total_size", 0}};
  index["weight_map"] = weight_map;
  FILE* fh = std::fopen((dir + "/model.safetensors.index.json").c_str(), "wb");
  REQUIRE(fh != nullptr);
  const std::string text = index.dump();
  std::fwrite(text.data(), 1, text.size(), fh);
  std::fclose(fh);
  return weight_map;
}

// The REAL release's SHAPE without its 66.3 GB of payload: headers declare every
// tensor at its true size and the payload is a SPARSE hole (ftruncate), so the
// manifest a loader reads is byte-for-byte the real one while the files cost a
// few KB of disk. Only names/shapes/dtypes are ever read from it — never a weight
// byte — which is exactly what --dump-params does on the real directory.
// Returns the total DECLARED payload bytes.
uint64_t WriteMiniMaxH3SparseShardedRelease(const std::vector<vllm::MiniMaxH3TensorSpec>& specs,
                                            const std::string& dir, size_t num_shards) {
  REQUIRE(num_shards > 0);
  ::mkdir(dir.c_str(), 0755);
  std::vector<std::vector<const vllm::MiniMaxH3TensorSpec*>> per_shard(num_shards);
  std::map<std::string, std::string> weight_map;
  for (size_t i = 0; i < specs.size(); ++i) {
    const size_t shard = i * num_shards / specs.size();
    per_shard[shard].push_back(&specs[i]);
    weight_map[specs[i].name] = ShardFileName(shard, num_shards);
  }

  uint64_t declared = 0;
  for (size_t s = 0; s < num_shards; ++s) {
    std::string header = "{";
    uint64_t offset = 0;
    bool first = true;
    for (const vllm::MiniMaxH3TensorSpec* spec : per_shard[s]) {
      // The upstream dtype policy: fp32 ISLANDS stay F32, everything else is BF16
      // — which is what makes the whole DiT ~66.3 GB.
      const uint64_t width = spec->fp32 ? 4u : 2u;
      uint64_t numel = 1;
      for (int64_t d : spec->shape) numel *= static_cast<uint64_t>(d);
      const uint64_t bytes = numel * width;
      if (!first) header += ",";
      first = false;
      header += "\"" + spec->name + "\":{\"dtype\":\"" + (spec->fp32 ? "F32" : "BF16") +
                "\",\"shape\":[";
      for (size_t i = 0; i < spec->shape.size(); ++i) {
        if (i) header += ",";
        header += std::to_string(spec->shape[i]);
      }
      header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                std::to_string(offset + bytes) + "]}";
      offset += bytes;
    }
    header += "}";
    const std::string path = dir + "/" + ShardFileName(s, num_shards);
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    std::fflush(fh);
    // The payload is a HOLE: declared in full, allocated not at all.
    REQUIRE(::ftruncate(fileno(fh), static_cast<off_t>(sizeof(n) + header.size() + offset)) == 0);
    std::fclose(fh);
    declared += offset;
  }

  nlohmann::json index;
  index["metadata"] = {{"total_size", declared}};
  index["weight_map"] = weight_map;
  FILE* fh = std::fopen((dir + "/model.safetensors.index.json").c_str(), "wb");
  REQUIRE(fh != nullptr);
  const std::string text = index.dump();
  std::fwrite(text.data(), 1, text.size(), fh);
  std::fclose(fh);
  return declared;
}

void RemoveShardedDit(const std::string& dir, size_t num_shards) {
  for (size_t s = 0; s < num_shards; ++s) {
    std::remove((dir + "/" + ShardFileName(s, num_shards)).c_str());
  }
  std::remove((dir + "/model.safetensors.index.json").c_str());
  ::rmdir(dir.c_str());
}

}  // namespace

TEST_CASE("minimax_h3: the deterministic weight stream matches the generator") {
  // If this fails, nothing else in this file is meaningful — the two sides would
  // be comparing different tensors, not different implementations.
  // Bit-exact on purpose: the two streams must agree exactly, not approximately.
  const std::vector<double> probe = H3Rand("h3.probe", 8);
  REQUIRE(probe.size() == std::size(vllm_test::kH3RandProbe));
  for (size_t i = 0; i < probe.size(); ++i) {
    CHECK(probe[i] == vllm_test::kH3RandProbe[i]);
  }
}

TEST_CASE("minimax_h3: fl2va packed sequence matches upstream") {
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);

  CHECK(packed.seq_len == vllm_test::kH3Fl2va_seq_len);
  CheckI64(packed.input_ids, vllm_test::kH3Fl2vaInputIds,
           std::size(vllm_test::kH3Fl2vaInputIds));
  CheckI64(packed.token_tags, vllm_test::kH3Fl2vaTokenTags,
           std::size(vllm_test::kH3Fl2vaTokenTags));
  CheckI64(packed.img_pos, vllm_test::kH3Fl2vaImgPos, std::size(vllm_test::kH3Fl2vaImgPos));
  CheckI64(packed.audio_pos, vllm_test::kH3Fl2vaAudioPos, std::size(vllm_test::kH3Fl2vaAudioPos));
  CheckI64(packed.text_pos, vllm_test::kH3Fl2vaTextPos, std::size(vllm_test::kH3Fl2vaTextPos));
  CheckI64(packed.update_mask, vllm_test::kH3Fl2vaUpdateMask,
           std::size(vllm_test::kH3Fl2vaUpdateMask));
  CheckI64(packed.cu_seqlens, vllm_test::kH3Fl2vaCuSeqlens,
           std::size(vllm_test::kH3Fl2vaCuSeqlens));
  CheckI64(packed.document_id, vllm_test::kH3Fl2vaDocumentId,
           std::size(vllm_test::kH3Fl2vaDocumentId));

  // The FP64 position grid feeds RoPE directly and is gated BIT-EXACT: a last-ulp
  // drift here would silently rotate every video token.
  REQUIRE(packed.img_position_ids.size() == std::size(vllm_test::kH3Fl2vaImgPositionIds));
  for (size_t i = 0; i < packed.img_position_ids.size(); ++i) {
    CHECK(packed.img_position_ids[i] == vllm_test::kH3Fl2vaImgPositionIds[i]);
  }
}

TEST_CASE("minimax_h3: ref2va block packed sequence matches upstream") {
  std::vector<MiniMaxH3RefBlock> blocks(2);
  blocks[0].kind = MiniMaxH3RefBlock::Kind::kImage;
  blocks[0].latent_h = 4;
  blocks[0].latent_w = 4;
  blocks[1].kind = MiniMaxH3RefBlock::Kind::kVideoAudio;
  blocks[1].ref_audio_t = 2;
  blocks[1].latent_t = 2;
  blocks[1].latent_h = 4;
  blocks[1].latent_w = 4;

  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequenceRef2va(
      vllm_test::kH3Ref2va_text_len, vllm_test::kH3Ref2va_latent_t, vllm_test::kH3Ref2va_latent_h,
      vllm_test::kH3Ref2va_latent_w, vllm_test::kH3Ref2va_audio_t, blocks,
      vllm_test::kH3Ref2va_audio_channel);

  CHECK(packed.seq_len == vllm_test::kH3Ref2va_seq_len);
  CheckI64(packed.input_ids, vllm_test::kH3Ref2vaInputIds,
           std::size(vllm_test::kH3Ref2vaInputIds));
  CheckI64(packed.token_tags, vllm_test::kH3Ref2vaTokenTags,
           std::size(vllm_test::kH3Ref2vaTokenTags));
  CheckI64(packed.img_pos, vllm_test::kH3Ref2vaImgPos, std::size(vllm_test::kH3Ref2vaImgPos));
  CheckI64(packed.audio_pos, vllm_test::kH3Ref2vaAudioPos, std::size(vllm_test::kH3Ref2vaAudioPos));
  CheckI64(packed.update_mask, vllm_test::kH3Ref2vaUpdateMask,
           std::size(vllm_test::kH3Ref2vaUpdateMask));
  CheckI64(packed.audio_update_mask, vllm_test::kH3Ref2vaAudioUpdateMask,
           std::size(vllm_test::kH3Ref2vaAudioUpdateMask));
  CheckI64(packed.cu_seqlens, vllm_test::kH3Ref2vaCuSeqlens,
           std::size(vllm_test::kH3Ref2vaCuSeqlens));
  REQUIRE(packed.img_position_ids.size() == std::size(vllm_test::kH3Ref2vaImgPositionIds));
  for (size_t i = 0; i < packed.img_position_ids.size(); ++i) {
    CHECK(packed.img_position_ids[i] == vllm_test::kH3Ref2vaImgPositionIds[i]);
  }
}

TEST_CASE("minimax_h3: video patchify and audio pack match upstream, and round-trip") {
  const int64_t c = vllm_test::kH3PatchifyC, t = vllm_test::kH3PatchifyT;
  const int64_t h = vllm_test::kH3PatchifyH, w = vllm_test::kH3PatchifyW;
  const std::vector<float> latent = MakeParam("packing.video_latent", c * t * h * w, 1.0);
  const std::vector<float> rows =
      MiniMaxH3PatchifyVideoLatent(latent, 1, c, t, h, w, 1, 2, 2);
  CHECK(static_cast<int64_t>(rows.size()) ==
        vllm_test::kH3PatchifyRows * vllm_test::kH3PatchifyRowWidth);
  CHECK(MaxAbsDiff(rows, vllm_test::kH3PatchifyRowsGolden, rows.size()) == 0.0);

  const std::vector<float> back =
      MiniMaxH3UnpatchifyVideoTokens(rows, t, h / 2, w / 2, c, 1, 2, 2);
  REQUIRE(back.size() == latent.size());
  for (size_t i = 0; i < back.size(); ++i) CHECK(back[i] == latent[i]);

  const int64_t ac = vllm_test::kH3AudioPackChannels, ad = vllm_test::kH3AudioPackDim;
  const int64_t at = vllm_test::kH3AudioPackT;
  const std::vector<float> audio = MakeParam("packing.audio_latent", ac * ad * at, 1.0);
  const std::vector<float> audio_rows = MiniMaxH3PackAudioLatent(audio, ac, ad, at);
  CHECK(MaxAbsDiff(audio_rows, vllm_test::kH3AudioPackRowsGolden, audio_rows.size()) == 0.0);
  const std::vector<float> audio_back = MiniMaxH3UnpackAudioTokens(audio_rows, ac * at, ac, ad);
  REQUIRE(audio_back.size() == audio.size());
  for (size_t i = 0; i < audio_back.size(); ++i) CHECK(audio_back[i] == audio[i]);
}

TEST_CASE("minimax_h3: flow-matching scheduler matches upstream") {
  const int64_t n = vllm_test::kH3SchedN;
  const std::vector<float> xt = MakeParam("sched.xt", n, 1.0);
  const std::vector<float> v = MakeParam("sched.v", n, 1.0);
  for (int64_t s = 0; s < vllm_test::kH3SchedSteps; ++s) {
    const double t = vllm_test::kH3SchedTimesteps[s];
    const std::vector<float> x0 = MiniMaxH3RfVToX0(xt, v, t);
    CHECK(MaxAbsDiff(x0, vllm_test::kH3SchedX0Golden + s * n, static_cast<size_t>(n)) == 0.0);
    const double sigma_curr = 1.0 - t;
    const std::vector<float> stepped =
        MiniMaxH3EulerEta0Step(xt, x0, sigma_curr, sigma_curr * 0.5);
    CHECK(MaxAbsDiff(stepped, vllm_test::kH3SchedStepGolden + s * n, static_cast<size_t>(n)) <=
          1e-6);
  }
  // The terminal step is a documented identity (scheduling:89-92).
  const std::vector<float> terminal = MiniMaxH3EulerEta0Step(xt, v, 0.0, 0.0);
  for (size_t i = 0; i < terminal.size(); ++i) CHECK(terminal[i] == xt[i]);
}

TEST_CASE("minimax_h3: DiT forward matches upstream at reduced dimensions") {
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;
  const std::unique_ptr<DitForwardCase> c = BuildDitForwardCase(p);
  REQUIRE(c->video_width == vllm_test::kH3Dit_video_row_width);

  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForward(Cpu(), p, weights->views, c->in, vt::DType::kF32);

  // f32 throughout on both sides, so the only slack is summation order inside the
  // GEMM and the softmax; 2e-5 absolute is far below any structural error.
  const double video_err =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsGolden, got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, vllm_test::kH3DitAudioLogitsGolden, got.audio_logits.size());
  INFO("video max|diff| = " << video_err << ", audio max|diff| = " << audio_err);
  CHECK(video_err <= 2e-5);
  CHECK(audio_err <= 2e-5);

  // The pinned keyframe rows must be zeroed by the update mask.
  for (int64_t r = 0; r < c->num_img; ++r) {
    if (c->packed.update_mask[static_cast<size_t>(r)]) continue;
    for (int64_t i = 0; i < c->video_width; ++i) {
      CHECK(got.video_logits[static_cast<size_t>(r * c->video_width + i)] == 0.0f);
    }
  }
}

// ---------------------------------------------------------------------------
// Section 5b: the DiT-forward GEOMETRY LADDER (the #70 below-one-tile close).
//
// The section-5 DiT gate ran ONE geometry: fl2va latent 4x6 (spatial 2x3 = 6
// video tokens). PR #70 root-caused the render grid to the DiT emitting a
// spatially-WHITE latent at REAL token geometry (8x8 = 64 tokens) while the 2x3
// gate matched upstream at 1.6e-7; the divergence lives strictly between 2x3 and
// 8x8. A spatial-mixing bug is in the position/packing/modulation MATH and the
// varlen attention span, NOT in weight values, so it must reproduce with the
// H3Rand random-weight harness at real TOKEN geometry with the SAME reduced
// hidden dims. This ladder (minimax_h3_goldens.inc :: kH3LadderR*) walks the
// video grid 2x3 -> 4x4 -> 6x6 -> 8x8 (+ a rectangle, a multi-frame temporal 3D
// grid, and a larger video+audio packed mix) and, at each rung, gates: (1) the
// packed-sequence layout at that geometry (the 2x3-only packed gate is the same
// blind spot); (2) the HOST forward AND (3) the DEVICE-resident forward (the one
// the real pipeline runs) against the RefDiT oracle; and (4) a SPATIAL-MIXING
// probe -- perturb one video-target token, confirm the per-output response
// matches the oracle byte-for-byte AND that the reduced-dim DiT actually couples
// every target token through the packed bidirectional attention (the property
// #70 found broken). NOTE: the #70 adjacent-cell-COSINE metric does NOT translate
// here -- with random weights the CORRECT reference is already spatially white by
// that metric (spatial coherence is a TRAINED-weights property); so the ladder's
// valid discriminators are oracle-logit equality and information flow, not the
// cosine. See .agents/specs/minimax-h3.md section 8.4.
namespace {

struct LadderGoldens {
  const char* name;
  int64_t text_len, latent_t, latent_h, latent_w, audio_t, audio_channel, frame_count;
  int64_t seq_len, num_img, num_audio, num_text, num_unique, video_width, first_target;
  const int64_t* cu_seqlens;      size_t n_cu;
  const int64_t* img_pos;         size_t n_img;
  const int64_t* audio_pos;       size_t n_audio;
  const int64_t* text_pos;        size_t n_text;
  const int64_t* update_mask;     size_t n_um;
  const int64_t* token_tags;      size_t n_tags;
  const double*  img_position_ids; size_t n_pos;
  const float*   unique_ts;       size_t n_uts;
  const int64_t* inverse;         size_t n_inv;
  const float*   video_logits;    size_t n_vl;
  const float*   audio_logits;    size_t n_al;
  const float*   mix_delta;       size_t n_md;
  double         mix_fraction;
};

#define VT_H3_LADDER_RUNG(i)                                                        \
  LadderGoldens {                                                                   \
    vllm_test::kH3LadderR##i##_name, vllm_test::kH3LadderR##i##_text_len,           \
    vllm_test::kH3LadderR##i##_latent_t, vllm_test::kH3LadderR##i##_latent_h,       \
    vllm_test::kH3LadderR##i##_latent_w, vllm_test::kH3LadderR##i##_audio_t,        \
    vllm_test::kH3LadderR##i##_audio_channel, vllm_test::kH3LadderR##i##_frame_count,\
    vllm_test::kH3LadderR##i##_seq_len, vllm_test::kH3LadderR##i##_num_img,         \
    vllm_test::kH3LadderR##i##_num_audio, vllm_test::kH3LadderR##i##_num_text,      \
    vllm_test::kH3LadderR##i##_num_unique, vllm_test::kH3LadderR##i##_video_width,  \
    vllm_test::kH3LadderR##i##_first_target,                                        \
    vllm_test::kH3LadderR##i##CuSeqlens, std::size(vllm_test::kH3LadderR##i##CuSeqlens),        \
    vllm_test::kH3LadderR##i##ImgPos, std::size(vllm_test::kH3LadderR##i##ImgPos),              \
    vllm_test::kH3LadderR##i##AudioPos, std::size(vllm_test::kH3LadderR##i##AudioPos),          \
    vllm_test::kH3LadderR##i##TextPos, std::size(vllm_test::kH3LadderR##i##TextPos),            \
    vllm_test::kH3LadderR##i##UpdateMask, std::size(vllm_test::kH3LadderR##i##UpdateMask),      \
    vllm_test::kH3LadderR##i##TokenTags, std::size(vllm_test::kH3LadderR##i##TokenTags),        \
    vllm_test::kH3LadderR##i##ImgPositionIds, std::size(vllm_test::kH3LadderR##i##ImgPositionIds),\
    vllm_test::kH3LadderR##i##UniqueTimesteps, std::size(vllm_test::kH3LadderR##i##UniqueTimesteps),\
    vllm_test::kH3LadderR##i##InverseIndices, std::size(vllm_test::kH3LadderR##i##InverseIndices),  \
    vllm_test::kH3LadderR##i##VideoLogits, std::size(vllm_test::kH3LadderR##i##VideoLogits),    \
    vllm_test::kH3LadderR##i##AudioLogits, std::size(vllm_test::kH3LadderR##i##AudioLogits),    \
    vllm_test::kH3LadderR##i##VideoMixDelta, std::size(vllm_test::kH3LadderR##i##VideoMixDelta),\
    vllm_test::kH3LadderR##i##MixFraction[0]                                        \
  }

std::vector<LadderGoldens> AllLadderRungs() {
  std::vector<LadderGoldens> rungs;
#define X(i) rungs.push_back(VT_H3_LADDER_RUNG(i));
  H3_LADDER_FOR_EACH(X)
#undef X
  return rungs;
}

// The ladder analogue of DitForwardCase, parameterised by one rung's geometry.
// Owns every buffer the returned MiniMaxH3DitInputs points at.
struct LadderCase {
  MiniMaxH3PackedSequence packed;
  std::vector<float> x, audio_x, prompt_embeds;
  std::vector<float> unique_timesteps;
  std::vector<int64_t> inverse;
  std::vector<int32_t> refiner_cu;
  int64_t seq_len = 0, num_img = 0, num_audio = 0, num_text = 0, video_width = 0;
  MiniMaxH3DitInputs in;
};

std::unique_ptr<LadderCase> BuildLadderCase(const MiniMaxH3DitParams& p,
                                            const LadderGoldens& g) {
  auto c = std::make_unique<LadderCase>();
  c->packed = BuildMiniMaxH3PackedSequence(g.text_len, g.latent_t, g.latent_h, g.latent_w,
                                           g.audio_t, g.audio_channel,
                                           /*include_keyframe_cond=*/true, {0}, g.frame_count);
  c->seq_len = c->packed.seq_len;
  c->video_width = p.video_row_width();
  c->num_img = static_cast<int64_t>(c->packed.img_pos.size());
  c->num_audio = static_cast<int64_t>(c->packed.audio_pos.size());
  c->num_text = static_cast<int64_t>(c->packed.text_pos.size());

  // The generator seeded each rung's random rows as "ladder.<name>.<field>".
  const std::string base = std::string("ladder.") + g.name;
  c->x.assign(static_cast<size_t>(c->seq_len * c->video_width), 0.0f);
  const std::vector<float> video_rows =
      MakeParam(base + ".video_rows", c->num_img * c->video_width, 1.0);
  for (int64_t r = 0; r < c->num_img; ++r) {
    std::memcpy(c->x.data() + c->packed.img_pos[static_cast<size_t>(r)] * c->video_width,
                video_rows.data() + r * c->video_width,
                static_cast<size_t>(c->video_width) * sizeof(float));
  }
  c->audio_x.assign(static_cast<size_t>(c->seq_len * p.audio_latents_dim), 0.0f);
  if (c->num_audio > 0) {
    const std::vector<float> audio_rows =
        MakeParam(base + ".audio_rows", c->num_audio * p.audio_latents_dim, 1.0);
    for (int64_t r = 0; r < c->num_audio; ++r) {
      std::memcpy(
          c->audio_x.data() + c->packed.audio_pos[static_cast<size_t>(r)] * p.audio_latents_dim,
          audio_rows.data() + r * p.audio_latents_dim,
          static_cast<size_t>(p.audio_latents_dim) * sizeof(float));
    }
  }
  c->prompt_embeds = MakeParam(base + ".prompt_embeds", c->num_text * p.text_dim, 1.0);
  c->unique_timesteps.assign(g.unique_ts, g.unique_ts + g.n_uts);
  c->inverse.assign(g.inverse, g.inverse + g.n_inv);
  c->refiner_cu = {0, static_cast<int32_t>(c->num_text), static_cast<int32_t>(c->num_text)};

  MiniMaxH3DitInputs& in = c->in;
  in.seq_len = c->seq_len;
  in.x = c->x.data();
  in.audio_x = c->audio_x.data();
  in.img_position_ids = c->packed.img_position_ids.data();
  in.unique_timesteps = c->unique_timesteps.data();
  in.num_unique_timesteps = static_cast<int64_t>(c->unique_timesteps.size());
  in.inverse_indices = c->inverse.data();
  in.token_tags = c->packed.token_tags.data();
  in.prompt_embeds = c->prompt_embeds.data();
  in.img_pos = c->packed.img_pos.data();
  in.num_img_pos = c->num_img;
  in.audio_pos = c->packed.audio_pos.data();
  in.num_audio_pos = c->num_audio;
  in.text_pos = c->packed.text_pos.data();
  in.num_text_pos = c->num_text;
  in.infer_out_pos = c->packed.img_pos.data();
  in.num_infer_out_pos = c->num_img;
  in.update_mask = c->packed.update_mask.data();
  in.cu_seqlens = c->packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(c->packed.cu_seqlens.size());
  in.refiner_cu_seqlens = c->refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(c->refiner_cu.size());
  return c;
}

// The mixing threshold; well above the 2e-5 logit tolerance so summation-order
// noise can never flip a "no response" into a spurious "response".
constexpr double kLadderMixEps = 1e-6;

}  // namespace

// The whole ladder in one case: layout + host forward + device forward + spatial
// mixing, at every geometry rung. This is the permanent below-one-tile gate.
TEST_CASE("minimax_h3: DiT-forward geometry ladder matches upstream (host+device, mixing)") {
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;
  vt::Queue q{Cpu(), nullptr};
  const MiniMaxH3DitDeviceWeights staged = StageMiniMaxH3DitWeights(q, p, weights->views);

  REQUIRE(static_cast<int64_t>(AllLadderRungs().size()) == vllm_test::kH3LadderRungs);
  for (const LadderGoldens& g : AllLadderRungs()) {
    const std::string rung = g.name;
    CAPTURE(rung);

    // (1) The packed-sequence LAYOUT at this geometry: the 2x3-only packed gate is
    // the same blind spot, so gate cu_seqlens / positions / masks here too.
    const std::unique_ptr<LadderCase> c = BuildLadderCase(p, g);
    REQUIRE(c->packed.seq_len == g.seq_len);
    REQUIRE(c->num_img == g.num_img);
    REQUIRE(c->num_audio == g.num_audio);
    CheckI64(c->packed.cu_seqlens, g.cu_seqlens, g.n_cu);
    CheckI64(c->packed.img_pos, g.img_pos, g.n_img);
    CheckI64(c->packed.audio_pos, g.audio_pos, g.n_audio);
    CheckI64(c->packed.text_pos, g.text_pos, g.n_text);
    CheckI64(c->packed.update_mask, g.update_mask, g.n_um);
    CheckI64(c->packed.token_tags, g.token_tags, g.n_tags);
    // The FP64 position grid feeds RoPE directly; gate it BIT-EXACT at every grid.
    REQUIRE(c->packed.img_position_ids.size() == g.n_pos);
    for (size_t i = 0; i < g.n_pos; ++i) {
      CHECK(c->packed.img_position_ids[i] == g.img_position_ids[i]);
    }

    // (2) HOST forward vs the RefDiT oracle.
    const MiniMaxH3DitOutputs host =
        MiniMaxH3DitForward(Cpu(), p, weights->views, c->in, vt::DType::kF32);
    const double hv = MaxAbsDiff(host.video_logits, g.video_logits, host.video_logits.size());
    const double ha = MaxAbsDiff(host.audio_logits, g.audio_logits, host.audio_logits.size());
    INFO("host: video max|diff| = " << hv << ", audio max|diff| = " << ha);
    CHECK(hv <= 2e-5);
    CHECK(ha <= 2e-5);

    // (3) DEVICE-resident forward vs the same oracle (the pipeline's own path).
    const MiniMaxH3DitOutputs dev =
        MiniMaxH3DitForwardDevice(q, p, staged.weights, c->in, vt::DType::kF32);
    const double dv = MaxAbsDiff(dev.video_logits, g.video_logits, dev.video_logits.size());
    const double da = MaxAbsDiff(dev.audio_logits, g.audio_logits, dev.audio_logits.size());
    INFO("device: video max|diff| = " << dv << ", audio max|diff| = " << da);
    CHECK(dv <= 2e-5);
    CHECK(da <= 2e-5);

    // The pinned keyframe (cond) rows must be zeroed by the update mask.
    for (int64_t r = 0; r < c->num_img; ++r) {
      if (c->packed.update_mask[static_cast<size_t>(r)]) continue;
      for (int64_t i = 0; i < c->video_width; ++i) {
        CHECK(host.video_logits[static_cast<size_t>(r * c->video_width + i)] == 0.0f);
      }
    }

    // (4) SPATIAL-MIXING probe: perturb one video-target token's input, measure the
    // per-output-row response, and require it to match the oracle AND to show that
    // every target token couples through the packed bidirectional attention.
    const std::unique_ptr<LadderCase> cp = BuildLadderCase(p, g);
    const int64_t pert_pos = cp->packed.img_pos[static_cast<size_t>(g.first_target)];
    for (int64_t i = 0; i < c->video_width; ++i) {
      cp->x[static_cast<size_t>(pert_pos * c->video_width + i)] += 1.0f;
    }
    const MiniMaxH3DitOutputs pert =
        MiniMaxH3DitForward(Cpu(), p, weights->views, cp->in, vt::DType::kF32);
    std::vector<float> delta(static_cast<size_t>(c->num_img), 0.0f);
    for (int64_t r = 0; r < c->num_img; ++r) {
      float worst = 0.0f;
      for (int64_t i = 0; i < c->video_width; ++i) {
        const size_t idx = static_cast<size_t>(r * c->video_width + i);
        worst = std::max(worst, std::abs(pert.video_logits[idx] - host.video_logits[idx]));
      }
      delta[static_cast<size_t>(r)] = worst;
    }
    // The response vector must match the oracle's byte-for-byte (to logit tol).
    CHECK(MaxAbsDiff(delta, g.mix_delta, delta.size()) <= 2e-5);
    // Fraction of OTHER video-target rows that responded to the perturbation.
    int64_t responded = 0, others = 0;
    for (int64_t r = 0; r < c->num_img; ++r) {
      if (!c->packed.update_mask[static_cast<size_t>(r)]) continue;  // cond rows are zeroed
      if (r == g.first_target) continue;
      ++others;
      if (delta[static_cast<size_t>(r)] > kLadderMixEps) ++responded;
    }
    const double frac = others > 0 ? static_cast<double>(responded) / static_cast<double>(others)
                                   : -1.0;
    INFO("mix fraction port=" << frac << " oracle=" << g.mix_fraction);
    CHECK(frac == doctest::Approx(g.mix_fraction).epsilon(1e-9));
    // The reduced-dim DiT MUST spatially couple all target tokens at real geometry
    // (this is exactly the property #70 found broken). If a rung ever drops below
    // full coupling, the spatial mixer regressed.
    if (others > 0) CHECK(g.mix_fraction >= 0.999);
  }
}

// The hidden-dim-scale leg of the ladder: run the SAME geometries at the REAL
// head_dim=128 / rope_inv_freq_len=16 ratio and require the DEVICE-resident
// forward to still match the trusted HOST loops. The reduced arch never exercises
// head_dim=128 or rot_dim=96, so a head_dim- or rope-scale assumption in a shared
// device op (attention tile, RoPE cache) would slip past the section-5b ladder but
// surface here -- exactly the "scale-dependent in hidden dims" escape hatch #70
// leaves open. No oracle is needed: the host forward is the reference the reduced
// ladder already proved correct, and this asks only whether the device path tracks
// it at real head geometry.
TEST_CASE("minimax_h3: DiT device-vs-host forward holds at the REAL head_dim=128 ratio") {
  const MiniMaxH3DitParams p = RealRatioParams();
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights(p);
  vt::Queue q{Cpu(), nullptr};
  const MiniMaxH3DitDeviceWeights staged = StageMiniMaxH3DitWeights(q, p, weights->views);

  for (const LadderGoldens& g : AllLadderRungs()) {
    const std::string rung = g.name;
    CAPTURE(rung);
    // Geometry only from the rung; the arch is the real-ratio one above.
    const std::unique_ptr<LadderCase> c = BuildLadderCase(p, g);
    const MiniMaxH3DitOutputs host =
        MiniMaxH3DitForward(Cpu(), p, weights->views, c->in, vt::DType::kF32);
    const MiniMaxH3DitOutputs dev =
        MiniMaxH3DitForwardDevice(q, p, staged.weights, c->in, vt::DType::kF32);
    const double dv = MaxAbsDiff(dev.video_logits, host.video_logits.data(), host.video_logits.size());
    const double da = MaxAbsDiff(dev.audio_logits, host.audio_logits.data(), host.audio_logits.size());
    INFO("real-ratio device-vs-host: video max|diff| = " << dv << ", audio max|diff| = " << da);
    // f32 summation-order slack only; a structural head_dim/rope regression would be
    // orders of magnitude larger.
    CHECK(dv <= 1e-3);
    CHECK(da <= 1e-3);
  }
}

// The DiT-forward REF2VA rung (goldens section 5c). Every other DiT-forward gate
// runs the FL2VA layout (keyframe-cond prefix sharing the target frame grid, no
// audio reference rows). The ref2va reference-row assembly -- image/video/audio
// reference blocks PREPENDED with their OWN position grid, a separate audio update
// mask, the pinned reference rows carrying the CONDITION timestep -- was never
// forwarded through the DiT, which is exactly how the section 8.9 ref2va grid hid:
// the pure-math layout was gated (goldens section 2) but nothing forwarded a
// ref2va layout end-to-end. This case builds the SAME upstream ref2va layout with
// BuildMiniMaxH3PackedSequenceRef2va, runs the host AND device DiT forward against
// the RefDiT oracle, and asserts the reference rows are masked out and the target
// rows spatially couple. It pins the port's ref2va packing and its DiT forward over
// reference rows to upstream in one gate.
TEST_CASE("minimax_h3: DiT-forward REF2VA rung matches upstream (reference rows, mixing)") {
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;
  vt::Queue q{Cpu(), nullptr};
  const MiniMaxH3DitDeviceWeights staged = StageMiniMaxH3DitWeights(q, p, weights->views);

  // (1) Reconstruct the reference blocks the generator emitted (kind: 0=image,
  // 1=audio, 2=video_audio) and rebuild the identical ref2va layout.
  std::vector<MiniMaxH3RefBlock> blocks(static_cast<size_t>(vllm_test::kH3Ref2vaDit_num_blocks));
  for (size_t i = 0; i < blocks.size(); ++i) {
    blocks[i].kind = static_cast<MiniMaxH3RefBlock::Kind>(vllm_test::kH3Ref2vaDitBlockKinds[i]);
    blocks[i].ref_audio_t = vllm_test::kH3Ref2vaDitBlockRefAudioT[i];
    blocks[i].latent_t = vllm_test::kH3Ref2vaDitBlockLatentT[i];
    blocks[i].latent_h = vllm_test::kH3Ref2vaDitBlockLatentH[i];
    blocks[i].latent_w = vllm_test::kH3Ref2vaDitBlockLatentW[i];
  }
  const MiniMaxH3PackedSequence packed = vllm::BuildMiniMaxH3PackedSequenceRef2va(
      vllm_test::kH3Ref2vaDit_text_len, vllm_test::kH3Ref2vaDit_latent_t,
      vllm_test::kH3Ref2vaDit_latent_h, vllm_test::kH3Ref2vaDit_latent_w,
      vllm_test::kH3Ref2vaDit_audio_t, blocks, vllm_test::kH3Ref2vaDit_audio_channel);

  const int64_t seq_len = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const int64_t video_width = p.video_row_width();
  REQUIRE(seq_len == vllm_test::kH3Ref2vaDit_seq_len);
  REQUIRE(num_img == vllm_test::kH3Ref2vaDit_num_img);
  REQUIRE(num_audio == vllm_test::kH3Ref2vaDit_num_audio);
  REQUIRE(num_text == vllm_test::kH3Ref2vaDit_num_text);

  CheckI64(packed.cu_seqlens, vllm_test::kH3Ref2vaDitCuSeqlens,
           std::size(vllm_test::kH3Ref2vaDitCuSeqlens));
  CheckI64(packed.img_pos, vllm_test::kH3Ref2vaDitImgPos, std::size(vllm_test::kH3Ref2vaDitImgPos));
  CheckI64(packed.audio_pos, vllm_test::kH3Ref2vaDitAudioPos,
           std::size(vllm_test::kH3Ref2vaDitAudioPos));
  CheckI64(packed.text_pos, vllm_test::kH3Ref2vaDitTextPos,
           std::size(vllm_test::kH3Ref2vaDitTextPos));
  CheckI64(packed.update_mask, vllm_test::kH3Ref2vaDitUpdateMask,
           std::size(vllm_test::kH3Ref2vaDitUpdateMask));
  CheckI64(packed.audio_update_mask, vllm_test::kH3Ref2vaDitAudioUpdateMask,
           std::size(vllm_test::kH3Ref2vaDitAudioUpdateMask));
  CheckI64(packed.token_tags, vllm_test::kH3Ref2vaDitTokenTags,
           std::size(vllm_test::kH3Ref2vaDitTokenTags));
  // The FP64 position grid over reference + target rows feeds RoPE; gate BIT-EXACT.
  REQUIRE(packed.img_position_ids.size() == std::size(vllm_test::kH3Ref2vaDitImgPositionIds));
  for (size_t i = 0; i < packed.img_position_ids.size(); ++i) {
    CHECK(packed.img_position_ids[i] == vllm_test::kH3Ref2vaDitImgPositionIds[i]);
  }

  // (2) Build the DiT inputs: scatter the same rung-seeded random rows, and carry
  // the ref2va audio_update_mask + the emitted timestep partition.
  auto build_inputs = [&](std::vector<float>& x, std::vector<float>& audio_x,
                          std::vector<float>& prompt, std::vector<float>& unique,
                          std::vector<int64_t>& inverse, std::vector<int32_t>& refiner_cu,
                          MiniMaxH3DitInputs& in) {
    x.assign(static_cast<size_t>(seq_len * video_width), 0.0f);
    const std::vector<float> vrows = MakeParam("ref2va_dit.r2v.video_rows", num_img * video_width, 1.0);
    for (int64_t r = 0; r < num_img; ++r) {
      std::memcpy(x.data() + packed.img_pos[static_cast<size_t>(r)] * video_width,
                  vrows.data() + r * video_width, static_cast<size_t>(video_width) * sizeof(float));
    }
    audio_x.assign(static_cast<size_t>(seq_len * p.audio_latents_dim), 0.0f);
    const std::vector<float> arows =
        MakeParam("ref2va_dit.r2v.audio_rows", num_audio * p.audio_latents_dim, 1.0);
    for (int64_t r = 0; r < num_audio; ++r) {
      std::memcpy(audio_x.data() + packed.audio_pos[static_cast<size_t>(r)] * p.audio_latents_dim,
                  arows.data() + r * p.audio_latents_dim,
                  static_cast<size_t>(p.audio_latents_dim) * sizeof(float));
    }
    prompt = MakeParam("ref2va_dit.r2v.prompt_embeds", num_text * p.text_dim, 1.0);
    unique.assign(vllm_test::kH3Ref2vaDitUniqueTimesteps,
                  vllm_test::kH3Ref2vaDitUniqueTimesteps + std::size(vllm_test::kH3Ref2vaDitUniqueTimesteps));
    inverse.assign(vllm_test::kH3Ref2vaDitInverseIndices,
                   vllm_test::kH3Ref2vaDitInverseIndices + std::size(vllm_test::kH3Ref2vaDitInverseIndices));
    refiner_cu = {0, static_cast<int32_t>(num_text), static_cast<int32_t>(num_text)};
    in = MiniMaxH3DitInputs{};
    in.seq_len = seq_len;
    in.x = x.data();
    in.audio_x = audio_x.data();
    in.img_position_ids = packed.img_position_ids.data();
    in.unique_timesteps = unique.data();
    in.num_unique_timesteps = static_cast<int64_t>(unique.size());
    in.inverse_indices = inverse.data();
    in.token_tags = packed.token_tags.data();
    in.prompt_embeds = prompt.data();
    in.img_pos = packed.img_pos.data();
    in.num_img_pos = num_img;
    in.audio_pos = packed.audio_pos.data();
    in.num_audio_pos = num_audio;
    in.text_pos = packed.text_pos.data();
    in.num_text_pos = num_text;
    in.infer_out_pos = packed.img_pos.data();
    in.num_infer_out_pos = num_img;
    in.update_mask = packed.update_mask.data();
    in.audio_update_mask = packed.audio_update_mask.data();
    in.cu_seqlens = packed.cu_seqlens.data();
    in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
    in.refiner_cu_seqlens = refiner_cu.data();
    in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());
  };

  std::vector<float> x, audio_x, prompt, unique;
  std::vector<int64_t> inverse;
  std::vector<int32_t> refiner_cu;
  MiniMaxH3DitInputs in;
  build_inputs(x, audio_x, prompt, unique, inverse, refiner_cu, in);
  REQUIRE(static_cast<int64_t>(unique.size()) == vllm_test::kH3Ref2vaDit_num_unique);

  // (3) HOST + DEVICE forward vs the RefDiT oracle.
  const MiniMaxH3DitOutputs host = MiniMaxH3DitForward(Cpu(), p, weights->views, in, vt::DType::kF32);
  const double hv = MaxAbsDiff(host.video_logits, vllm_test::kH3Ref2vaDitVideoLogits, host.video_logits.size());
  const double ha = MaxAbsDiff(host.audio_logits, vllm_test::kH3Ref2vaDitAudioLogits, host.audio_logits.size());
  INFO("host: video max|diff| = " << hv << ", audio max|diff| = " << ha);
  CHECK(hv <= 2e-5);
  CHECK(ha <= 2e-5);

  const MiniMaxH3DitOutputs dev = MiniMaxH3DitForwardDevice(q, p, staged.weights, in, vt::DType::kF32);
  const double dv = MaxAbsDiff(dev.video_logits, vllm_test::kH3Ref2vaDitVideoLogits, dev.video_logits.size());
  const double da = MaxAbsDiff(dev.audio_logits, vllm_test::kH3Ref2vaDitAudioLogits, dev.audio_logits.size());
  INFO("device: video max|diff| = " << dv << ", audio max|diff| = " << da);
  CHECK(dv <= 2e-5);
  CHECK(da <= 2e-5);

  // The pinned reference rows (video AND audio) must be masked to zero in the output.
  for (int64_t r = 0; r < num_img; ++r) {
    if (packed.update_mask[static_cast<size_t>(r)]) continue;
    for (int64_t i = 0; i < video_width; ++i)
      CHECK(host.video_logits[static_cast<size_t>(r * video_width + i)] == 0.0f);
  }
  for (int64_t r = 0; r < num_audio; ++r) {
    if (packed.audio_update_mask[static_cast<size_t>(r)]) continue;
    for (int64_t i = 0; i < p.audio_latents_dim; ++i)
      CHECK(host.audio_logits[static_cast<size_t>(r * p.audio_latents_dim + i)] == 0.0f);
  }

  // (4) SPATIAL-MIXING probe: perturb the first video-TARGET token and require the
  // response to match the oracle and to couple every other target token through the
  // packed bidirectional attention (the property the section 8.9 grid lacked).
  std::vector<float> xp, ap, pp, up;
  std::vector<int64_t> ip;
  std::vector<int32_t> rc;
  MiniMaxH3DitInputs inp;
  build_inputs(xp, ap, pp, up, ip, rc, inp);
  const int64_t pert_pos = packed.img_pos[static_cast<size_t>(vllm_test::kH3Ref2vaDit_first_target)];
  for (int64_t i = 0; i < video_width; ++i) xp[static_cast<size_t>(pert_pos * video_width + i)] += 1.0f;
  const MiniMaxH3DitOutputs pert = MiniMaxH3DitForward(Cpu(), p, weights->views, inp, vt::DType::kF32);
  std::vector<float> delta(static_cast<size_t>(num_img), 0.0f);
  for (int64_t r = 0; r < num_img; ++r) {
    float worst = 0.0f;
    for (int64_t i = 0; i < video_width; ++i) {
      const size_t idx = static_cast<size_t>(r * video_width + i);
      worst = std::max(worst, std::abs(pert.video_logits[idx] - host.video_logits[idx]));
    }
    delta[static_cast<size_t>(r)] = worst;
  }
  CHECK(MaxAbsDiff(delta, vllm_test::kH3Ref2vaDitVideoMixDelta, delta.size()) <= 2e-5);
  int64_t responded = 0, others = 0;
  for (int64_t r = 0; r < num_img; ++r) {
    if (!packed.update_mask[static_cast<size_t>(r)]) continue;
    if (r == vllm_test::kH3Ref2vaDit_first_target) continue;
    ++others;
    if (delta[static_cast<size_t>(r)] > kLadderMixEps) ++responded;
  }
  const double frac = others > 0 ? static_cast<double>(responded) / static_cast<double>(others) : -1.0;
  INFO("ref2va mix fraction port=" << frac << " oracle=" << vllm_test::kH3Ref2vaDitMixFraction[0]);
  CHECK(frac == doctest::Approx(vllm_test::kH3Ref2vaDitMixFraction[0]).epsilon(1e-9));
  CHECK(vllm_test::kH3Ref2vaDitMixFraction[0] >= 0.999);
}

// Brick H3-2b. The DEVICE-RESIDENT forward runs the same graph with every
// activation in device memory. It is NOT bit-identical to the CPU reference and
// does not claim to be -- it reuses the tuned SHARED vt:: ops (vt::RmsNorm reduces
// in f32 where RmsNormRows accumulates in double), so it is held to the SAME
// upstream goldens at the SAME tolerance instead.
//
// Runs on the CPU backend here, which is the point: the device forward's STRUCTURE
// (staging, index plumbing, the strided AdaLN chunk views, on-device row selection)
// is gated without a GPU. The CUDA case below covers the kernels themselves.
static void CheckDeviceForward(vt::Queue& q, const char* label) {
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;
  const std::unique_ptr<DitForwardCase> c = BuildDitForwardCase(p);

  const MiniMaxH3DitDeviceWeights staged =
      StageMiniMaxH3DitWeights(q, p, weights->views);
  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForwardDevice(q, p, staged.weights, c->in, vt::DType::kF32);

  REQUIRE(got.video_logits.size() ==
          static_cast<size_t>(c->num_img * c->video_width));
  REQUIRE(got.audio_logits.size() ==
          static_cast<size_t>(c->num_audio * p.audio_latents_dim));

  const double video_err =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsGolden, got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, vllm_test::kH3DitAudioLogitsGolden, got.audio_logits.size());
  INFO(label << ": video max|diff| = " << video_err << ", audio max|diff| = " << audio_err);
  CHECK(video_err <= 2e-5);
  CHECK(audio_err <= 2e-5);

  // The pinned keyframe rows must still be zeroed.
  for (int64_t r = 0; r < c->num_img; ++r) {
    if (c->packed.update_mask[static_cast<size_t>(r)]) continue;
    for (int64_t i = 0; i < c->video_width; ++i) {
      CHECK(got.video_logits[static_cast<size_t>(r * c->video_width + i)] == 0.0f);
    }
  }
}

// The bf16 sibling of CheckDeviceForward: gates the PRODUCTION dtype policy on the
// device path. Same code, different rounding points -- the device forward rounds in
// place at exactly the reference's dt.Apply sites, so what is being compared is the
// CAST POINTS, not a different algorithm.
static void CheckDeviceForwardBf16(vt::Queue& q, const char* label) {
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;
  const std::unique_ptr<DitForwardCase> c = BuildDitForwardCase(p);

  // Stage with the bf16 policy too: upstream STORES those modules in bf16 (the
  // generator's to_bf16_weights), so the weights must be rounded the same way the
  // golden's were -- staging f32 weights under a bf16 stream would compare a
  // different model, not a different dtype policy.
  const MiniMaxH3DitDeviceWeights staged =
      StageMiniMaxH3DitWeights(q, p, weights->views, vt::DType::kBF16);
  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForwardDevice(q, p, staged.weights, c->in, vt::DType::kBF16);

  const double video_err =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsBf16Golden, got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, vllm_test::kH3DitAudioLogitsBf16Golden, got.audio_logits.size());
  INFO(label << ": bf16 video max|diff| = " << video_err << ", audio max|diff| = " << audio_err);
  CHECK(video_err <= 5e-3);
  CHECK(audio_err <= 5e-3);

  // And the bf16 stream must actually DIFFER from the f32 stream -- otherwise the
  // dtype policy is not being applied and this test proves nothing.
  const double vs_f32 =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsGolden, got.video_logits.size());
  CHECK(vs_f32 > 1e-5);
}

TEST_CASE("minimax_h3: the DEVICE-resident bf16 stream matches upstream (CPU backend)") {
  vt::Queue q{Cpu(), nullptr};
  CheckDeviceForwardBf16(q, "cpu-device-forward-bf16");
}

TEST_CASE("minimax_h3: the DEVICE-resident bf16 stream matches upstream on CUDA") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckDeviceForwardBf16(q, "cuda-device-forward-bf16");
}

TEST_CASE("minimax_h3: the DEVICE-resident DiT forward matches upstream (CPU backend)") {
  vt::Queue q{Cpu(), nullptr};
  CheckDeviceForward(q, "cpu-device-forward");
}

TEST_CASE("minimax_h3: the DEVICE-resident DiT forward matches upstream on CUDA") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckDeviceForward(q, "cuda-device-forward");
}

// H3-RENDER-CLOSE: the one surface #74 left untested. The "REAL head_dim=128
// ratio" device-vs-host case above runs on the CPU BACKEND, and the CUDA cases run
// only at the SMALL fl2va geometry (seq ~<200). The #70 white latent is a REAL
// render: CUDA kernels at REAL SEQ (t2va 512x512/22f -> latent 7x32x32 ->
// seq_len 1920, cu_seqlens=[0,1874,1920] non-causal 2-document) at head_dim=128.
// If a CUDA kernel (varlen non-causal attention, RoPE cache, AdaLN modulate) has a
// scale-dependent bug that its CPU counterpart does not, the render is white while
// every reduced-dim gate is green. This runs the SAME MiniMaxH3DitForwardDevice on
// the CUDA backend vs the trusted CPU host loops at exactly the render geometry and
// step-0 timestep partition; a divergence here IS the bug, a match points at S2.
TEST_CASE("minimax_h3: CUDA device forward tracks the host at the REAL render seq (1920)") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const MiniMaxH3DitParams p = RealRatioParams();  // head_dim=128, rot_dim=96
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights(p);

  // The real t2va render geometry at 512x512/22f (verified on dgx: text_len=8,
  // latent 7x32x32, audio_t=37, seq_len 1920).
  const int64_t text_len = 8, latent_t = 7, latent_h = 32, latent_w = 32;
  const int64_t audio_t = 37, audio_channel = 2;
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      text_len, latent_t, latent_h, latent_w, audio_t, audio_channel,
      /*include_keyframe_cond=*/false, {}, /*frame_count=*/0);
  const int64_t seq_len = packed.seq_len;
  const int64_t video_width = p.video_row_width();
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  REQUIRE(seq_len == 1920);
  REQUIRE(num_img == 1792);

  std::vector<float> x(static_cast<size_t>(seq_len * video_width), 0.0f);
  const std::vector<float> video_rows = MakeParam("h3seq.video_rows", num_img * video_width, 1.0);
  for (int64_t r = 0; r < num_img; ++r) {
    std::memcpy(x.data() + packed.img_pos[static_cast<size_t>(r)] * video_width,
                video_rows.data() + r * video_width,
                static_cast<size_t>(video_width) * sizeof(float));
  }
  std::vector<float> audio_x(static_cast<size_t>(seq_len * p.audio_latents_dim), 0.0f);
  const std::vector<float> audio_rows =
      MakeParam("h3seq.audio_rows", num_audio * p.audio_latents_dim, 1.0);
  for (int64_t r = 0; r < num_audio; ++r) {
    std::memcpy(audio_x.data() + packed.audio_pos[static_cast<size_t>(r)] * p.audio_latents_dim,
                audio_rows.data() + r * p.audio_latents_dim,
                static_cast<size_t>(p.audio_latents_dim) * sizeof(float));
  }
  const std::vector<float> prompt_embeds = MakeParam("h3seq.prompt_embeds", num_text * p.text_dim, 1.0);
  // Step-0 partition (dumped from the real render): all timesteps 0 -> one unique.
  const std::vector<float> unique_timesteps = {0.0f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq_len), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};

  MiniMaxH3DitInputs in;
  in.seq_len = seq_len;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_timesteps.data();
  in.num_unique_timesteps = static_cast<int64_t>(unique_timesteps.size());
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt_embeds.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  const MiniMaxH3DitOutputs host =
      MiniMaxH3DitForward(Cpu(), p, weights->views, in, vt::DType::kF32);
  vt::Queue q = cuda->CreateQueue();
  const MiniMaxH3DitDeviceWeights staged = StageMiniMaxH3DitWeights(q, p, weights->views);
  const MiniMaxH3DitOutputs dev =
      MiniMaxH3DitForwardDevice(q, p, staged.weights, in, vt::DType::kF32);

  const double dv = MaxAbsDiff(dev.video_logits, host.video_logits.data(), host.video_logits.size());
  const double da = MaxAbsDiff(dev.audio_logits, host.audio_logits.data(), host.audio_logits.size());
  INFO("CUDA-vs-host at real seq 1920: video max|diff| = " << dv << ", audio max|diff| = " << da);
  // f32 summation-order slack only (seq 1920 accumulates more than the small
  // cases, so allow 5e-3); a structural CUDA-kernel-at-scale regression would be
  // orders of magnitude larger and is exactly what #70 is hunting.
  CHECK(dv <= 5e-3);
  CHECK(da <= 5e-3);
}

TEST_CASE("minimax_h3: the bf16 production stream matches upstream's dtype policy") {
  // The f32 case above gates the ALGORITHM. This one gates the PRODUCTION dtype
  // policy: upstream's stream is bf16 with fp32 islands (both patch projections,
  // the time embedder, both output heads — minimax_h3_transformer.py:85-101), and
  // the explicit `dtype=_BF16_DTYPE` casts inside `_modulate_scale_shift` /
  // `_modulate_gate`. Same code path, different rounding points.
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;

  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);

  const int64_t seq_len = packed.seq_len;
  const int64_t video_width = p.video_row_width();
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());

  std::vector<float> x(static_cast<size_t>(seq_len * video_width), 0.0f);
  const std::vector<float> video_rows = MakeParam("dit.video_rows", num_img * video_width, 1.0);
  for (int64_t r = 0; r < num_img; ++r) {
    std::memcpy(x.data() + packed.img_pos[static_cast<size_t>(r)] * video_width,
                video_rows.data() + r * video_width,
                static_cast<size_t>(video_width) * sizeof(float));
  }
  std::vector<float> audio_x(static_cast<size_t>(seq_len * p.audio_latents_dim), 0.0f);
  const std::vector<float> audio_rows =
      MakeParam("dit.audio_rows", num_audio * p.audio_latents_dim, 1.0);
  for (int64_t r = 0; r < num_audio; ++r) {
    std::memcpy(audio_x.data() + packed.audio_pos[static_cast<size_t>(r)] * p.audio_latents_dim,
                audio_rows.data() + r * p.audio_latents_dim,
                static_cast<size_t>(p.audio_latents_dim) * sizeof(float));
  }
  const std::vector<float> prompt_embeds =
      MakeParam("dit.prompt_embeds", num_text * p.text_dim, 1.0);
  const std::vector<float> unique_timesteps(
      vllm_test::kH3DitUniqueTimesteps,
      vllm_test::kH3DitUniqueTimesteps + vllm_test::kH3Dit_unique_timesteps);
  const std::vector<int64_t> inverse(vllm_test::kH3DitInverseIndices,
                                     vllm_test::kH3DitInverseIndices + seq_len);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};

  MiniMaxH3DitInputs in;
  in.seq_len = seq_len;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_timesteps.data();
  in.num_unique_timesteps = static_cast<int64_t>(unique_timesteps.size());
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt_embeds.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForward(Cpu(), p, weights->views, in, vt::DType::kBF16);

  // Tolerance is bf16-scale on purpose: both sides round at the same points, but
  // the GEMM accumulation order still differs (ours is the vt CPU kernel, the
  // reference is torch). The gate is that the CAST POINTS agree — a misplaced or
  // missing cast moves the result far more than accumulation order does.
  const double video_err =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsBf16Golden, got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, vllm_test::kH3DitAudioLogitsBf16Golden, got.audio_logits.size());
  INFO("bf16 video max|diff| = " << video_err << ", audio max|diff| = " << audio_err);
  CHECK(video_err <= 5e-3);
  CHECK(audio_err <= 5e-3);

  // And the bf16 stream must differ from the f32 stream — otherwise the dtype
  // policy is not actually being applied and this test proves nothing.
  const double vs_f32 =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsGolden, got.video_logits.size());
  CHECK(vs_f32 > 1e-5);
}

TEST_CASE("minimax_h3: request planning matches upstream") {
  // time_request.py executed verbatim by the generator; the three shape helpers
  // are restated there from pipeline_minimax_h3.py:121-122, 207-222, 393-434.
  for (int64_t i = 0; i < vllm_test::kH3PlanFrameCases; ++i) {
    const int64_t input = vllm_test::kH3PlanFrameInputs[i];
    const int64_t aligned = vllm::MiniMaxH3AlignFrameCount(input);
    CHECK(aligned == vllm_test::kH3PlanFrameAligned[i]);
    const int64_t latent_t = vllm::MiniMaxH3VideoLatentT(aligned);
    CHECK(latent_t == vllm_test::kH3PlanVideoLatentT[i]);
    CHECK(vllm::MiniMaxH3FrameCountFromVideoLatentT(latent_t) ==
          vllm_test::kH3PlanFrameFromLatentT[i]);
    CHECK(vllm::MiniMaxH3AudioLatentT(static_cast<double>(aligned) / 24.0) ==
          vllm_test::kH3PlanAudioLatentT[i]);
  }
  // A latent T that is neither 1 nor 5n+2 is rejected, not silently rounded.
  CHECK_THROWS(vllm::MiniMaxH3FrameCountFromVideoLatentT(4));

  int64_t offset = 0;
  for (int64_t c = 0; c < vllm_test::kH3PlanSigmaCases; ++c) {
    const std::vector<double> sigmas = vllm::MiniMaxH3TimeShiftSigmas(
        vllm_test::kH3PlanSigmaSteps[c], vllm_test::kH3PlanSigmaShifts[c]);
    REQUIRE(static_cast<int64_t>(sigmas.size()) == vllm_test::kH3PlanSigmaLengths[c]);
    for (size_t i = 0; i < sigmas.size(); ++i) {
      CHECK(static_cast<float>(sigmas[i]) ==
            doctest::Approx(vllm_test::kH3PlanSigmasGolden[offset + i]).epsilon(1e-6));
    }
    // Multi-step schedules must terminate at sigma 0 so the last Euler step is
    // the identity (scheduling:89-92).
    if (vllm_test::kH3PlanSigmaSteps[c] > 1) CHECK(sigmas.back() == 0.0);
    offset += static_cast<int64_t>(sigmas.size());
  }
  CHECK_THROWS(vllm::MiniMaxH3TimeShiftSigmas(50, 0.0));
  CHECK_THROWS(vllm::MiniMaxH3TimeShiftSigmas(0, 6.0));

  for (int64_t c = 0; c < vllm_test::kH3PlanRefImageCases; ++c) {
    const std::pair<int64_t, int64_t> shape = vllm::MiniMaxH3ReferenceImageShape(
        vllm_test::kH3PlanRefImageInputs[c * 2], vllm_test::kH3PlanRefImageInputs[c * 2 + 1]);
    CHECK(shape.first == vllm_test::kH3PlanRefImageGolden[c * 2]);
    CHECK(shape.second == vllm_test::kH3PlanRefImageGolden[c * 2 + 1]);
  }
  CHECK_THROWS(vllm::MiniMaxH3ReferenceImageShape(5000, 100));  // aspect out of [1:4, 4:1]

  const char* tasks[] = {"t2va", "ref2va", "t2va", "t2va", "fl2va", "fl2va", "t2va"};
  for (int64_t c = 0; c < vllm_test::kH3PlanShapeCases; ++c) {
    const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
        tasks[c], vllm_test::kH3PlanShapeDurations[c], vllm_test::kH3PlanShapeNumFrames[c],
        vllm_test::kH3PlanShapeHeights[c], vllm_test::kH3PlanShapeWidths[c],
        vllm_test::kH3PlanShapeImageW[c], vllm_test::kH3PlanShapeImageH[c]);
    INFO("shape case " << c << " task=" << tasks[c]);
    CHECK(plan.height == vllm_test::kH3PlanShapeGolden[c * 5 + 0]);
    CHECK(plan.width == vllm_test::kH3PlanShapeGolden[c * 5 + 1]);
    CHECK(plan.num_frames == vllm_test::kH3PlanShapeGolden[c * 5 + 2]);
    CHECK(plan.latent_t == vllm_test::kH3PlanShapeGolden[c * 5 + 3]);
    CHECK(plan.audio_t == vllm_test::kH3PlanShapeGolden[c * 5 + 4]);
  }

  // Task dispatch (pipeline_minimax_h3.py:374-391).
  const std::vector<std::string> fl2va_partition = {"t2va", "fl2va"};
  const std::vector<std::string> ref2va_partition = {"ref2va"};
  CHECK(vllm::MiniMaxH3ResolveTask("", "FL2VA", /*has_image=*/false, fl2va_partition) == "t2va");
  CHECK(vllm::MiniMaxH3ResolveTask("", "FL2VA", /*has_image=*/true, fl2va_partition) == "fl2va");
  CHECK(vllm::MiniMaxH3ResolveTask("", "ref2va", /*has_image=*/false, ref2va_partition) == "ref2va");
  CHECK(vllm::MiniMaxH3ResolveTask("T2VA", "FL2VA", false, fl2va_partition) == "t2va");
  // A partition must refuse a task it does not carry, rather than guessing.
  CHECK_THROWS(vllm::MiniMaxH3ResolveTask("ref2va", "FL2VA", false, fl2va_partition));
}

// The #77 follow-up: the task/partition GUARD. The #70/#74 white render cost three
// campaigns because the driver silently accepted t2va on the Ref2VA NVFP4
// checkpoint — a task/partition mismatch upstream `_resolve_task` RAISES on
// (pipeline_minimax_h3.py:374-391; recipes/MiniMaxAI/MiniMax-H3.md:50-51,289). This
// gates the mirror: the mismatch now FAILS LOUDLY, the correct pairings pass, and
// the community-file (stripped-config) path refuses ambiguity rather than guessing.
TEST_CASE("minimax_h3: the task/partition guard refuses the #77 mismatch") {
  using vllm::MiniMaxH3CheckTaskPartition;
  using vllm::MiniMaxH3PartitionFromFlag;
  using vllm::MiniMaxH3PartitionFromModelIndex;

  // --- Partition detection path 1: the release model_index.json `_minimax_h3`
  // block (pipeline_minimax_h3.py:279-282). Synthetic manifests of BOTH shapes.
  const nlohmann::json fl2va_index = {
      {"_minimax_h3", {{"partition", "fl2va"}, {"tasks", {"t2va", "fl2va"}}}}};
  const nlohmann::json ref2va_index = {
      {"_minimax_h3", {{"partition", "ref2va"}, {"tasks", {"ref2va"}}}}};
  const vllm::MiniMaxH3PartitionInfo fl2va = MiniMaxH3PartitionFromModelIndex(fl2va_index);
  const vllm::MiniMaxH3PartitionInfo ref2va = MiniMaxH3PartitionFromModelIndex(ref2va_index);
  CHECK(fl2va.declared);
  CHECK(fl2va.partition == "fl2va");
  CHECK(fl2va.supported_tasks == std::vector<std::string>{"t2va", "fl2va"});
  CHECK(ref2va.partition == "ref2va");
  CHECK(ref2va.supported_tasks == std::vector<std::string>{"ref2va"});

  // The GUARD BEHAVIOR TABLE (task x partition -> pass/refuse). The whole point of
  // the row: the top-left cell (t2va on Ref2VA) is the #77 failure mode.
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("t2va", ref2va));   // <-- #77: FAIL LOUDLY
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("fl2va", ref2va));  // fl2va needs FL2VA
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("ref2va", ref2va));
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("t2va", fl2va));
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("fl2va", fl2va));
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("ref2va", fl2va));  // ref2va needs Ref2VA
  // Case-insensitive, mirroring `str(requested).lower()` (pipeline:386).
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("T2VA", fl2va));
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("T2VA", ref2va));

  // --- Partition detection path 2: an explicit --partition flag (community
  // GGUF/NVFP4 strip the config). The recipe's one-server-one-partition split.
  const vllm::MiniMaxH3PartitionInfo fl2va_flag = MiniMaxH3PartitionFromFlag("fl2va");
  const vllm::MiniMaxH3PartitionInfo ref2va_flag = MiniMaxH3PartitionFromFlag("ref2va");
  CHECK(fl2va_flag.supported_tasks == std::vector<std::string>{"t2va", "fl2va"});
  CHECK(ref2va_flag.supported_tasks == std::vector<std::string>{"ref2va"});
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("t2va", ref2va_flag));   // same #77 catch
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("fl2va", fl2va_flag));
  CHECK_THROWS(MiniMaxH3PartitionFromFlag("bogus"));  // invalid partition name

  // --- The ambiguous / stripped case: NO config block AND NO --partition. The
  // partition is unknown, and (see the structural check below) unknowable from the
  // weights, so the guard REFUSES every task rather than guessing.
  const vllm::MiniMaxH3PartitionInfo stripped = MiniMaxH3PartitionFromModelIndex(nlohmann::json::object());
  CHECK(stripped.declared);
  CHECK(stripped.partition.empty());
  CHECK(stripped.supported_tasks.empty());
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("t2va", stripped));
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("fl2va", stripped));
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("ref2va", stripped));
  const vllm::MiniMaxH3PartitionInfo empty_flag = MiniMaxH3PartitionFromFlag("");
  CHECK(empty_flag.declared);
  CHECK(empty_flag.supported_tasks.empty());
  CHECK_THROWS(MiniMaxH3CheckTaskPartition("t2va", empty_flag));  // --partition required

  // --- A DEFAULT-CONSTRUCTED (declared=false) request leaves the guard INACTIVE:
  // the pure pipeline-math tests build requests by hand and must be unaffected.
  const vllm::MiniMaxH3PartitionInfo undeclared;
  CHECK_FALSE(undeclared.declared);
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("t2va", undeclared));
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition("ref2va", undeclared));

  // --- The dispatch key: MiniMaxH3TaskOfRequest, the same discriminator
  // MiniMaxH3GenerateT2va feeds the guard. ref_blocks => ref2va, keyframes => fl2va,
  // else t2va.
  vllm::MiniMaxH3T2vaRequest t2va_req;  // no conditioning
  CHECK(vllm::MiniMaxH3TaskOfRequest(t2va_req) == "t2va");
  vllm::MiniMaxH3T2vaRequest fl2va_req;
  fl2va_req.keyframe_frame_indices = {0};
  CHECK(vllm::MiniMaxH3TaskOfRequest(fl2va_req) == "fl2va");
  vllm::MiniMaxH3T2vaRequest ref2va_req;
  ref2va_req.ref_blocks.push_back(vllm::MiniMaxH3RefBlock{});
  CHECK(vllm::MiniMaxH3TaskOfRequest(ref2va_req) == "ref2va");

  // The exact #77 combination as the dispatch sees it: a t2va request pointed at a
  // Ref2VA checkpoint. This is what MiniMaxH3GenerateT2va now refuses.
  t2va_req.partition = ref2va;
  CHECK_THROWS(MiniMaxH3CheckTaskPartition(vllm::MiniMaxH3TaskOfRequest(t2va_req), t2va_req.partition));
  ref2va_req.partition = ref2va;  // the correct pairing passes
  CHECK_NOTHROW(MiniMaxH3CheckTaskPartition(vllm::MiniMaxH3TaskOfRequest(ref2va_req), ref2va_req.partition));

  // --- WHY the stripped case must refuse rather than auto-detect: the two REAL
  // captured manifests (ref2va NVFP4 = 1051 tensors, FL2VA GGUF = 535) carry the
  // IDENTICAL DiT — same base tensor names AND shapes. NVFP4 splits each quantized
  // weight into {weight, weight_scale, weight_scale_2}; normalizing that away, the
  // two name sets are equal, so no tensor-name/shape discriminator exists.
  auto normalize_nvfp4 = [](std::string n) -> std::string {
    for (const char* suffix : {".weight_scale_2", ".weight_scale"}) {
      const std::string s = suffix;
      if (n.size() >= s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0) {
        return n.substr(0, n.size() - s.size()) + ".weight";
      }
    }
    return n;
  };
  std::set<std::string> nvfp4_names, gguf_names;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) {
    nvfp4_names.insert(normalize_nvfp4(t.name));
  }
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) gguf_names.insert(t.name);
  CHECK(nvfp4_names.size() == 535);
  CHECK(gguf_names.size() == 535);
  CHECK(nvfp4_names == gguf_names);  // no structural discriminator => --partition is required
}

TEST_CASE("minimax_h3: the denoise loop advances targets and pins condition rows") {
  // The loop itself has no upstream golden (upstream's own loop test needs the
  // checkpoint), so this gates its INVARIANTS, which are what the CFG-distilled
  // schedule actually guarantees (denoise_loop.py:191-238): pinned keyframe rows
  // are reset to their anchor after EVERY step, target rows move, audio advances
  // on its own sigma schedule, and nothing goes non-finite.
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;

  vllm::MiniMaxH3DenoiseBranch branch;
  branch.packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);
  branch.token_tags = branch.packed.token_tags;

  const int64_t num_img = static_cast<int64_t>(branch.packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(branch.packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(branch.packed.text_pos.size());
  const int64_t video_width = p.video_row_width();
  branch.text_embeddings = MakeParam("loop.prompt_embeds", num_text * p.text_dim, 1.0);

  const std::vector<float> initial_video = MakeParam("loop.video", num_img * video_width, 1.0);
  const std::vector<float> initial_audio =
      MakeParam("loop.audio", num_audio * p.audio_latents_dim, 1.0);
  int64_t cond_rows = 0;
  for (uint8_t flag : branch.packed.update_mask) cond_rows += flag ? 0 : 1;
  REQUIRE(cond_rows > 0);
  const std::vector<float> keyframe = MakeParam("loop.keyframe", cond_rows * video_width, 1.0);

  // A 2-step schedule: enough to exercise the chaining and the per-step reset.
  const std::vector<double> sigmas_video = {0.6, 0.3, 0.0};
  const std::vector<double> sigmas_audio = {0.5, 0.25, 0.0};

  const vllm::MiniMaxH3DenoiseResult result = vllm::MiniMaxH3DenoiseLoop(
      Cpu(), p, weights->views, branch, initial_video, initial_audio, keyframe, {}, sigmas_video,
      sigmas_audio, vt::DType::kF32);

  REQUIRE(result.video_rows.size() == initial_video.size());
  REQUIRE(result.audio_rows.size() == initial_audio.size());
  for (float value : result.video_rows) REQUIRE(std::isfinite(value));
  for (float value : result.audio_rows) REQUIRE(std::isfinite(value));

  // Pinned rows must equal their anchors exactly; target rows must have moved.
  int64_t cond_index = 0, moved = 0;
  for (int64_t r = 0; r < num_img; ++r) {
    if (branch.packed.update_mask[static_cast<size_t>(r)]) {
      for (int64_t i = 0; i < video_width; ++i) {
        if (result.video_rows[static_cast<size_t>(r * video_width + i)] !=
            initial_video[static_cast<size_t>(r * video_width + i)]) {
          ++moved;
          break;
        }
      }
      continue;
    }
    for (int64_t i = 0; i < video_width; ++i) {
      CHECK(result.video_rows[static_cast<size_t>(r * video_width + i)] ==
            keyframe[static_cast<size_t>(cond_index * video_width + i)]);
    }
    ++cond_index;
  }
  CHECK(cond_index == cond_rows);
  CHECK(moved == num_img - cond_rows);  // every target row advanced

  // The terminal sigma is 0, so the last step is the identity on the state and the
  // schedule must land somewhere finite and different from where it started.
  bool audio_moved = false;
  for (size_t i = 0; i < result.audio_rows.size(); ++i) {
    if (result.audio_rows[i] != initial_audio[i]) audio_moved = true;
  }
  CHECK(audio_moved);
}

TEST_CASE("minimax_h3: the audio VAE decoder matches the checkpoint's own remote code") {
  // H3's VAEs are checkpoint REMOTE CODE (loaded via trust_remote_code), so a
  // no-Python engine must REIMPLEMENT them. This gates our reimplementation
  // against the checkpoint's OWN modules, executed at reduced dimensions by
  // scripts/gen-minimax-h3-audio-vae-goldens.py. The remote code is not vendored
  // here; the generator takes a path to a local copy.
  vllm::MiniMaxH3AudioVaeConfig config;
  config.num_mels = vllm_test::kH3AudioVaeNumMels;
  config.upsample_initial_channel = vllm_test::kH3AudioVaeInitialChannel;
  config.upsample_rates = {2, 2};
  config.upsample_kernel_sizes = {4, 4};
  config.resblock_kernel_sizes = {3, 7, 11};
  config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  config.use_tanh_at_final = false;
  config.use_bias_at_final = false;
  config.snake_logscale = true;

  // The kaiser-sinc filter is COMPUTED, not loaded. Prove it first: if the filter
  // is wrong, every anti-aliased activation is wrong and the decoder mismatch
  // would be impossible to localize.
  const std::vector<float> filter = vllm::MiniMaxH3KaiserSincFilter1d(0.5 / 2, 0.6 / 2, 12);
  REQUIRE(filter.size() == std::size(vllm_test::kH3AudioVaeUpFilterGolden));
  double filter_err = 0.0;
  double filter_sum = 0.0;
  for (size_t i = 0; i < filter.size(); ++i) {
    filter_err = std::max(filter_err, std::abs(static_cast<double>(filter[i]) -
                                               vllm_test::kH3AudioVaeUpFilterGolden[i]));
    filter_sum += filter[i];
  }
  INFO("kaiser-sinc filter max|diff| = " << filter_err);
  CHECK(filter_err <= 1e-6);
  CHECK(filter_sum == doctest::Approx(1.0).epsilon(1e-6));  // normalized to sum 1

  // Rebuild every parameter from the shared stream, using the checkpoint's own
  // state_dict names and the generator's per-role scales.
  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam(name, count, scale, offset);
  };
  auto put_conv = [&](const std::string& prefix, int64_t out_channels, int64_t in_channels,
                      int64_t kernel, bool bias) {
    put(prefix + ".parametrizations.weight.original0", out_channels, 0.03, 0.15);
    put(prefix + ".parametrizations.weight.original1", out_channels * in_channels * kernel, 0.08, 0.0);
    if (bias) put(prefix + ".bias", out_channels, 0.05, 0.0);
  };
  auto put_act = [&](const std::string& prefix, int64_t channels) {
    put(prefix + ".act.alpha", channels, 0.2, 0.0);
    put(prefix + ".act.beta", channels, 0.2, 0.0);
  };

  const int64_t initial = config.upsample_initial_channel;
  put_conv("conv_pre", initial, config.num_mels, 7, /*bias=*/true);
  int64_t channels = initial;
  for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
    const int64_t out_channels = initial / (int64_t{1} << (i + 1));
    // ConvTranspose1d weight is [in, out, k]; weight-norm dim 0 is IN.
    const std::string prefix = "ups." + std::to_string(i) + ".0";
    put(prefix + ".parametrizations.weight.original0", channels, 0.03, 0.15);
    put(prefix + ".parametrizations.weight.original1",
        channels * out_channels * config.upsample_kernel_sizes[i], 0.08, 0.0);
    put(prefix + ".bias", out_channels, 0.05, 0.0);
    channels = out_channels;
    for (size_t j = 0; j < config.resblock_kernel_sizes.size(); ++j) {
      const std::string block =
          "resblocks." + std::to_string(i * config.resblock_kernel_sizes.size() + j);
      const int64_t kernel = config.resblock_kernel_sizes[j];
      for (size_t d = 0; d < config.resblock_dilation_sizes[j].size(); ++d) {
        put_conv(block + ".convs1." + std::to_string(d), channels, channels, kernel, true);
        put_conv(block + ".convs2." + std::to_string(d), channels, channels, kernel, true);
        put_act(block + ".activations." + std::to_string(2 * d), channels);
        put_act(block + ".activations." + std::to_string(2 * d + 1), channels);
      }
    }
  }
  put_act("activation_post", channels);
  put_conv("conv_post", 1, channels, 7, /*bias=*/false);

  const std::vector<float> latent =
      MakeParam("audiovae.input", config.num_mels * vllm_test::kH3AudioVaeFrames, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> waveform = vllm::MiniMaxH3AudioVaeDecode(
      config, weights, latent, vllm_test::kH3AudioVaeFrames, &out_samples);

  CHECK(out_samples == vllm_test::kH3AudioVaeOutSamples);
  REQUIRE(waveform.size() == std::size(vllm_test::kH3AudioVaeWaveformGolden));
  const double err =
      MaxAbsDiff(waveform, vllm_test::kH3AudioVaeWaveformGolden, waveform.size());
  INFO("audio VAE waveform max|diff| = " << err);
  CHECK(err <= 1e-5);
  // The golden is deliberately unsaturated: a clamped golden would hide errors.
  for (float value : waveform) {
    CHECK(value > -1.0f);
    CHECK(value < 1.0f);
  }

  // Weight-norm materialization: ||w_c|| must equal g_c exactly.
  const std::vector<float> g = {2.0f, 0.5f};
  const std::vector<float> v = {3.0f, 4.0f, 0.0f, 1.0f};  // rows [3,4] and [0,1]
  const std::vector<float> w = vllm::MiniMaxH3MaterializeWeightNorm(g, v, 2);
  REQUIRE(w.size() == 4);
  CHECK(w[0] == doctest::Approx(2.0f * 3.0f / 5.0f));
  CHECK(w[1] == doctest::Approx(2.0f * 4.0f / 5.0f));
  CHECK(w[2] == doctest::Approx(0.0f));
  CHECK(w[3] == doctest::Approx(0.5f));
}

namespace {

// The reduced audio-VAE ENCODER configuration the generator ran upstream at.
vllm::MiniMaxH3AudioVaeEncoderConfig ReducedAudioEncoderConfig() {
  vllm::MiniMaxH3AudioVaeEncoderConfig cfg;
  cfg.encoder_dim = vllm_test::kH3AudioEncDim;
  cfg.encoder_rates.assign(std::begin(vllm_test::kH3AudioEncRates),
                           std::end(vllm_test::kH3AudioEncRates));
  cfg.latent_dim = vllm_test::kH3AudioEncLatentDim;
  cfg.vae_latent_channels = vllm_test::kH3AudioEncVaeChannels;
  cfg.attn_proj = true;
  cfg.attn_proj_heads = vllm_test::kH3AudioEncHeads;
  return cfg;
}

// Rebuild every encoder parameter from the shared stream, keyed by the CHECKPOINT's
// own state_dict name, applying exactly the per-role scales
// gen-minimax-h3-audio-vae-encoder-goldens.py :: fill_from_stream applies. The rule
// is expressed once, here, for the same reason it is expressed once there: a scale
// that drifts on one side turns the golden into noise.
void PutEncoderParam(vllm::MiniMaxH3AudioVaeWeights& w, const std::string& full, int64_t count,
                     const std::string& stream_prefix, bool one_d_norm = false) {
  auto ends = [&](const std::string& suffix) {
    return full.size() >= suffix.size() &&
           full.compare(full.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  // The loader strips `encoder.`; the STREAM is keyed by the unstripped name.
  std::string key = full;
  if (key.rfind("encoder.", 0) == 0) key = key.substr(std::strlen("encoder."));

  if (ends("zero_k_bias")) {  // a registered buffer that stays zero
    w.tensors[key] = std::vector<float>(static_cast<size_t>(count), 0.0f);
    return;
  }
  double scale = 0.1, offset = 0.0;
  if (ends("original0")) {
    scale = 0.03;
    offset = 0.15;
  } else if (ends("original1")) {
    scale = 0.08;
  } else if (ends(".alpha")) {
    scale = 0.2;
    offset = 1.0;  // Snake1d divides by (alpha + 1e-9); upstream inits it to ones
  } else if (ends("bias")) {
    scale = 0.05;
  } else if (one_d_norm && full.find("norm") != std::string::npos && ends(".weight")) {
    scale = 0.1;
    offset = 1.0;
  }
  w.tensors[key] = MakeParam(stream_prefix + full, count, scale, offset);
}

// Every tensor the reduced encoder + pre_block + mean_proj needs, named exactly as
// the checkpoint names them.
vllm::MiniMaxH3AudioVaeWeights BuildAudioEncoderWeights(
    const vllm::MiniMaxH3AudioVaeEncoderConfig& cfg, const std::string& stream_prefix) {
  vllm::MiniMaxH3AudioVaeWeights w;
  auto conv = [&](const std::string& prefix, int64_t out_ch, int64_t in_ch, int64_t k) {
    PutEncoderParam(w, prefix + ".parametrizations.weight.original0", out_ch, stream_prefix);
    PutEncoderParam(w, prefix + ".parametrizations.weight.original1", out_ch * in_ch * k,
                    stream_prefix);
    PutEncoderParam(w, prefix + ".bias", out_ch, stream_prefix);
  };

  // block.0: WNConv1d(1, encoder_dim, k=7).
  conv("encoder.block.0", cfg.encoder_dim, 1, 7);
  int64_t channels = cfg.encoder_dim;
  for (size_t i = 0; i < cfg.encoder_rates.size(); ++i) {
    channels *= 2;
    const int64_t half = channels / 2;
    const std::string blk = "encoder.block." + std::to_string(i + 1);
    for (int64_t u = 0; u < 3; ++u) {  // three dilated ResidualUnits at dim/2
      const std::string ru = blk + ".block." + std::to_string(u);
      PutEncoderParam(w, ru + ".block.0.alpha", half, stream_prefix);
      conv(ru + ".block.1", half, half, 7);
      PutEncoderParam(w, ru + ".block.2.alpha", half, stream_prefix);
      conv(ru + ".block.3", half, half, 1);
    }
    PutEncoderParam(w, blk + ".block.3.alpha", half, stream_prefix);
    conv(blk + ".block.4", channels, half, 2 * cfg.encoder_rates[i]);
  }
  const size_t rates = cfg.encoder_rates.size();
  PutEncoderParam(w, "encoder.block." + std::to_string(rates + 1) + ".alpha", channels,
                  stream_prefix);
  conv("encoder.block." + std::to_string(rates + 2), cfg.latent_dim, channels, 3);

  // pre_block: AttnProjection(latent_dim -> attn_proj_dim).
  const int64_t in_dim = cfg.latent_dim, out_dim = cfg.attn_proj_dim();
  auto norm = [&](const std::string& prefix, int64_t width) {
    PutEncoderParam(w, prefix + ".weight", width, stream_prefix, /*one_d_norm=*/true);
    PutEncoderParam(w, prefix + ".bias", width, stream_prefix);
  };
  norm("pre_block.norm1", in_dim);
  norm("pre_block.norm3", in_dim);
  norm("pre_block.norm2", out_dim);
  PutEncoderParam(w, "pre_block.attn.q_bias", in_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.attn.v_bias", in_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.attn.zero_k_bias", in_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.attn.qkv.weight", 3 * in_dim * in_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.attn.proj.weight", out_dim * out_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.attn.proj.bias", out_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.proj.weight", out_dim * in_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.proj.bias", out_dim, stream_prefix);
  norm("pre_block.mlp.norm", out_dim);
  const int64_t hidden = 2 * out_dim;  // AttnProjection's mlp_ratio is 2
  PutEncoderParam(w, "pre_block.mlp.w0.weight", hidden * out_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.mlp.w0.bias", hidden, stream_prefix);
  PutEncoderParam(w, "pre_block.mlp.w1.weight", hidden * out_dim, stream_prefix);
  PutEncoderParam(w, "pre_block.mlp.w1.bias", hidden, stream_prefix);
  PutEncoderParam(w, "pre_block.mlp.w2.weight", out_dim * hidden, stream_prefix);
  PutEncoderParam(w, "pre_block.mlp.w2.bias", out_dim, stream_prefix);

  // mean_proj: a PLAIN Conv1d(attn_proj_dim -> vae_latent_channels, k=1).
  PutEncoderParam(w, "mean_proj.weight", cfg.vae_latent_channels * out_dim, stream_prefix);
  PutEncoderParam(w, "mean_proj.bias", cfg.vae_latent_channels, stream_prefix);
  return w;
}

}  // namespace

TEST_CASE("minimax_h3: the audio VAE ENCODER matches the checkpoint's own remote code") {
  // The ANALYSIS half, gated the same way the decoder is: against the CHECKPOINT'S
  // OWN modules (FL2VA/audio_vae/dac_audio_vae.py + dac_attn_proj.py) executed at
  // reduced dimensions by scripts/gen-minimax-h3-audio-vae-encoder-goldens.py, in
  // the exact sequence vLLM-Omni's reference-audio path runs (vae.py:317-325).
  //
  // Three STAGES are checked, not just the end: Encoder, then pre_block, then
  // mean_proj. A single end-to-end number would tell us something is wrong without
  // telling us which of three quite different subsystems it is.
  const vllm::MiniMaxH3AudioVaeEncoderConfig cfg = ReducedAudioEncoderConfig();
  const vllm::MiniMaxH3AudioVaeWeights w = BuildAudioEncoderWeights(cfg, "");

  CHECK(cfg.hop_length() == 4);           // prod(encoder_rates)
  CHECK(cfg.attn_proj_dim() == vllm_test::kH3AudioEncVaeChannels);

  // The input the oracle ran, drawn from the same stream. If this drifts, nothing
  // downstream means anything.
  const std::vector<float> input =
      MakeParam("audiovae.encoder.input", vllm_test::kH3AudioEncSamples, 1.0);
  REQUIRE(input.size() == std::size(vllm_test::kH3AudioEncInputGolden));
  CHECK(MaxAbsDiff(input, vllm_test::kH3AudioEncInputGolden, input.size()) == 0.0);

  // --- stage 1: preprocess + Encoder. The sample count (25) is deliberately NOT a
  // multiple of hop_length (4): a port that skipped `preprocess` would return 6
  // frames instead of 7 and fail here.
  int64_t frames = 0;
  const std::vector<float> enc = vllm::MiniMaxH3AudioVaeEncoderForward(
      cfg, w, input, vllm_test::kH3AudioEncSamples, &frames);
  CHECK(frames == vllm_test::kH3AudioEncFrames);
  CHECK(frames * cfg.hop_length() > vllm_test::kH3AudioEncSamples);  // the pad really happened
  REQUIRE(enc.size() == std::size(vllm_test::kH3AudioEncBlockGolden));
  const double enc_err = MaxAbsDiff(enc, vllm_test::kH3AudioEncBlockGolden, enc.size());
  INFO("audio VAE encoder (conv stack) max|diff| = " << enc_err);
  CHECK(enc_err <= 1e-5);

  // --- stage 2: pre_block (AttnProjection). Our forward takes [frames, latent_dim]
  // rows; the golden is the upstream [1, attn_proj_dim, frames] tensor, so the
  // comparison transposes.
  const int64_t in_dim = cfg.latent_dim, out_dim = cfg.attn_proj_dim();
  std::vector<float> tokens(static_cast<size_t>(frames * in_dim));
  for (int64_t c = 0; c < in_dim; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      tokens[static_cast<size_t>(t * in_dim + c)] = enc[static_cast<size_t>(c * frames + t)];
    }
  }
  const std::vector<float> proj = vllm::MiniMaxH3AudioVaeAttnProjection(cfg, w, tokens, frames);
  REQUIRE(proj.size() == std::size(vllm_test::kH3AudioEncAttnProjGolden));
  double proj_err = 0.0;
  for (int64_t c = 0; c < out_dim; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      proj_err = std::max(proj_err,
                          std::abs(static_cast<double>(proj[static_cast<size_t>(t * out_dim + c)]) -
                                   vllm_test::kH3AudioEncAttnProjGolden[c * frames + t]));
    }
  }
  INFO("audio VAE encoder (AttnProjection) max|diff| = " << proj_err);
  CHECK(proj_err <= 1e-5);

  // --- stage 3: the whole encode, ending at mean_proj (the distribution MEAN).
  int64_t latent_frames = 0;
  const std::vector<float> latent = vllm::MiniMaxH3AudioVaeEncodeToLatent(
      cfg, w, input, vllm_test::kH3AudioEncSamples, &latent_frames);
  CHECK(latent_frames == frames);
  REQUIRE(latent.size() == std::size(vllm_test::kH3AudioEncMeanGolden));
  const double mean_err = MaxAbsDiff(latent, vllm_test::kH3AudioEncMeanGolden, latent.size());
  INFO("audio VAE encode-to-latent max|diff| = " << mean_err);
  CHECK(mean_err <= 1e-5);

  // The MEAN, not a sample: the same reference must encode identically every time,
  // or a reference would condition differently on every run.
  int64_t again_frames = 0;
  const std::vector<float> again = vllm::MiniMaxH3AudioVaeEncodeToLatent(
      cfg, w, input, vllm_test::kH3AudioEncSamples, &again_frames);
  CHECK(MaxAbsDiff(again, latent.data(), again.size()) == 0.0);
  // `logs_proj` must play no part -- the loader never even loads it.
  CHECK_FALSE(w.Has("logs_proj.weight"));
}

TEST_CASE("minimax_h3: audio-VAE encode produces the packed rows a reference block claims") {
  // The row layout is the contract between the encoder and the packed sequence: a
  // block claiming `ref_audio_t * audio_channel` rows must get exactly those rows,
  // CHANNEL-MAJOR, in the DiT's normalized latent space (vae.py:327-341).
  const vllm::MiniMaxH3AudioVaeEncoderConfig cfg = ReducedAudioEncoderConfig();
  const vllm::MiniMaxH3AudioVaeWeights w = BuildAudioEncoderWeights(cfg, "");
  const int64_t width = cfg.vae_latent_channels;
  const int64_t samples = vllm_test::kH3AudioEncSamples;

  // Two channels: the SAME waveform in both, so the two halves must be identical.
  std::vector<float> stereo;
  const std::vector<float> mono = MakeParam("audiovae.encoder.input", samples, 1.0);
  stereo.insert(stereo.end(), mono.begin(), mono.end());
  stereo.insert(stereo.end(), mono.begin(), mono.end());

  int64_t audio_t = 0;
  const std::vector<float> rows =
      vllm::MiniMaxH3AudioVaeEncodeToRows(cfg, w, stereo, 2, samples, {}, {}, &audio_t);
  CHECK(audio_t == vllm_test::kH3AudioEncFrames);
  REQUIRE(rows.size() == static_cast<size_t>(2 * audio_t * width));
  // Channel-major: rows [0, audio_t) are channel 0, [audio_t, 2*audio_t) channel 1.
  double channel_diff = 0.0;
  for (int64_t i = 0; i < audio_t * width; ++i) {
    channel_diff = std::max(channel_diff, std::abs(static_cast<double>(rows[i]) -
                                                   rows[audio_t * width + i]));
  }
  CHECK(channel_diff == 0.0);
  // And row r of channel 0 is the mean_proj latent's frame r, transposed.
  int64_t lf = 0;
  const std::vector<float> latent = vllm::MiniMaxH3AudioVaeEncodeToLatent(cfg, w, mono, samples, &lf);
  for (int64_t t = 0; t < audio_t; ++t) {
    for (int64_t d = 0; d < width; ++d) {
      CHECK(rows[static_cast<size_t>(t * width + d)] ==
            doctest::Approx(latent[static_cast<size_t>(d * audio_t + t)]));
    }
  }

  // With statistics the rows move into the DiT's normalized space: (x - mean) / std.
  std::vector<float> mean(static_cast<size_t>(width)), stdev(static_cast<size_t>(width));
  for (int64_t d = 0; d < width; ++d) {
    mean[static_cast<size_t>(d)] = 0.25f * static_cast<float>(d + 1);
    stdev[static_cast<size_t>(d)] = 1.5f + 0.5f * static_cast<float>(d);
  }
  int64_t nt = 0;
  const std::vector<float> normalized =
      vllm::MiniMaxH3AudioVaeEncodeToRows(cfg, w, stereo, 2, samples, mean, stdev, &nt);
  CHECK(nt == audio_t);
  for (int64_t t = 0; t < audio_t; ++t) {
    for (int64_t d = 0; d < width; ++d) {
      const double raw = rows[static_cast<size_t>(t * width + d)];
      CHECK(normalized[static_cast<size_t>(t * width + d)] ==
            doctest::Approx((raw - mean[static_cast<size_t>(d)]) / stdev[static_cast<size_t>(d)])
                .epsilon(1e-5));
    }
  }
}

TEST_CASE("minimax_h3: reference-audio rows take the SHIPPED 32-wide path, noise aug included") {
  // The golden config narrows 16 -> 4 through the adaptive average pool. The
  // SHIPPED one does not pool at all in the same way, and its row width is the 32
  // that MiniMaxH3AudioCondNoiseAug hard-codes (as upstream does), so both the
  // no-pool branch and the noise-aug delegation are only reachable at 32.
  vllm::MiniMaxH3AudioVaeEncoderConfig cfg;
  cfg.encoder_dim = 2;
  cfg.encoder_rates = {2, 2};
  cfg.latent_dim = 64;
  cfg.vae_latent_channels = vllm::kMiniMaxH3AudioRowWidth;  // 32
  cfg.attn_proj = true;
  cfg.attn_proj_heads = 2;
  // 64 % 32 == 0, so attn_proj_dim is vae_latent_channels itself, and
  // head_dim = 64 / 2 = 32 == out_dim: the adaptive pool is a NO-OP here.
  CHECK(cfg.attn_proj_dim() == vllm::kMiniMaxH3AudioRowWidth);
  const vllm::MiniMaxH3AudioVaeWeights w = BuildAudioEncoderWeights(cfg, "wide.");

  const int64_t channels = vllm::kMiniMaxH3AudioChannels;
  const int64_t samples = 20;  // hop 4 -> 5 latent frames
  const std::vector<float> wave = MakeParam("wide.wave", channels * samples, 0.8);

  vllm::MiniMaxH3RefBlock block{};
  const std::vector<float> clean = vllm::MiniMaxH3EncodeReferenceAudio(
      cfg, w, wave, channels, samples, {}, {}, /*noise_aug=*/1.0, {}, &block);
  CHECK(block.kind == vllm::MiniMaxH3RefBlock::Kind::kAudio);
  CHECK(block.ref_audio_t == samples / cfg.hop_length());
  REQUIRE(clean.size() ==
          static_cast<size_t>(channels * block.ref_audio_t * vllm::kMiniMaxH3AudioRowWidth));
  for (float v : clean) CHECK(std::isfinite(v));

  // noise_aug < 1 must DELEGATE to the already-gated mix rather than reimplement
  // it: out = aug*clean + (1-aug)*noise, over the same rows.
  const std::vector<float> noise = MakeParam("wide.noise", clean.size(), 0.5);
  const double aug = 0.35;
  const std::vector<float> mixed = vllm::MiniMaxH3EncodeReferenceAudio(
      cfg, w, wave, channels, samples, {}, {}, aug, noise, &block);
  REQUIRE(mixed.size() == clean.size());
  double mix_err = 0.0;
  for (size_t i = 0; i < mixed.size(); ++i) {
    const double want = aug * clean[i] + (1.0 - aug) * noise[i];
    mix_err = std::max(mix_err, std::abs(static_cast<double>(mixed[i]) - want));
  }
  INFO("reference-audio noise-aug mix max|diff| = " << mix_err);
  CHECK(mix_err <= 1e-6);
  // And it really MOVED: a delegation that silently returned `clean` would pass
  // every shape check above.
  CHECK(MaxAbsDiff(mixed, clean.data(), clean.size()) > 1e-3);

  // A mono waveform handed in as stereo is refused: the packed layout is
  // channel-major over kMiniMaxH3AudioChannels, so a one-channel encode would
  // silently claim half the rows a block declares.
  CHECK_THROWS(vllm::MiniMaxH3EncodeReferenceAudio(cfg, w, wave, 1, samples, {}, {}, 1.0, {},
                                                   nullptr));
}

TEST_CASE("minimax_h3: a REAL ComfyUI GGUF resolves onto our weight contract") {
  // The whole point of the GGUF arm: the quantized checkpoints are the ones that
  // FIT this hardware. This gates the loader against the actual 535-tensor
  // manifest of `MiniMax-H3-FL2VA-Q3_K_M.gguf` (names/dims/types read from the
  // file's own header by scripts/gen-minimax-h3-gguf-manifest.py) — no weight
  // bytes, no download.
  CHECK(std::string(vllm_test::kH3GgufArchitecture) == "wan");  // ComfyUI's arch id
  CHECK(vllm_test::kH3GgufVersion == 3);
  REQUIRE(vllm_test::kH3GgufTensorCount == static_cast<int64_t>(std::size(vllm_test::kH3GgufTensors)));

  // Resolve every real tensor to its logical (torch) shape.
  std::vector<vllm::MiniMaxH3TensorSpec> manifest;
  manifest.reserve(static_cast<size_t>(vllm_test::kH3GgufTensorCount));
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) {
    const std::vector<int64_t> dims(t.dims, t.dims + t.n_dims);
    const std::vector<int64_t> orig(t.orig_shape, t.orig_shape + t.orig_n_dims);
    vllm::MiniMaxH3TensorSpec spec;
    spec.name = t.name;
    spec.shape = vllm::MiniMaxH3GgufLogicalShape(dims, orig);
    spec.fp32 = t.ggml_type == 0;
    manifest.push_back(std::move(spec));
  }

  // The geometry derived from SHAPES ALONE must be the shipped H3 geometry —
  // a ComfyUI GGUF ships no transformer config, so this is the load path.
  const MiniMaxH3DitParams p = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  CHECK(p.num_layers == 50);
  CHECK(p.token_refiner_num_layers == 2);
  CHECK(p.hidden_size == 5376);
  CHECK(p.num_attention_heads == 56);
  CHECK(p.attention_head_dim == 128);
  CHECK(p.ffn_hidden_size == 14336);
  CHECK(p.latents_dim == 24);
  CHECK(p.audio_latents_dim == 32);
  CHECK(p.text_dim == 5120);
  CHECK(p.timestep_input_dim == 256);
  CHECK(p.time_embed_dim == 2688);
  CHECK(p.rope_inv_freq_len == 16);
  CHECK(p.video_row_width() == 96);

  // And the manifest must cover our contract EXACTLY: the ComfyUI GGUF keeps the
  // checkpoint's own names, so this is an identity map, not a rename table.
  const std::vector<vllm::MiniMaxH3TensorSpec> expected = EnumerateMiniMaxH3DitTensors(p);
  CHECK(manifest.size() == expected.size());
  std::map<std::string, std::vector<int64_t>> got;
  for (const vllm::MiniMaxH3TensorSpec& spec : manifest) got[spec.name] = spec.shape;
  for (const vllm::MiniMaxH3TensorSpec& want : expected) {
    INFO("contract tensor " << want.name);
    const auto it = got.find(want.name);
    REQUIRE(it != got.end());
    CHECK(it->second == want.shape);
  }

  // The AdaLN projections are the reshaped case: ne is block-aligned nonsense
  // ([256, 1016064]) and the true shape comes from comfy.gguf.orig_shape.
  bool saw_reshaped = false;
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) {
    if (std::string(t.name) != "blocks.0.adaln_proj.linear.weight") continue;
    saw_reshaped = true;
    CHECK(t.orig_n_dims == 2);
    CHECK(t.orig_shape[0] == 96768);  // 18 * 5376
    CHECK(t.orig_shape[1] == 2688);   // time_embed_dim
    CHECK(t.dims[0] == 256);          // one Q3_K block along the fastest axis
    // 2688 is NOT a multiple of the 256-element Q3_K block, which is exactly why
    // ComfyUI reshaped it and recorded orig_shape.
    CHECK(2688 % 256 != 0);
  }
  CHECK(saw_reshaped);

  // The reversal rule on a NON-reshaped quantized tensor.
  const std::vector<int64_t> qkv =
      vllm::MiniMaxH3GgufLogicalShape({5376, 21504}, {});
  REQUIRE(qkv.size() == 2);
  CHECK(qkv[0] == 21504);  // 3 * 56 * 128
  CHECK(qkv[1] == 5376);
  // 1-D tensors stay 1-D.
  const std::vector<int64_t> norm = vllm::MiniMaxH3GgufLogicalShape({5376}, {});
  REQUIRE(norm.size() == 1);
  CHECK(norm[0] == 5376);

  // The fp32 islands must still be unquantized in the GGUF: quantizing a patch
  // projection or an output head would silently break the dtype policy.
  std::map<std::string, uint32_t> types;
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) types[t.name] = t.ggml_type;
  for (const char* name : {"blocks.0.norm1.weight", "blocks.0.attn.q_norm.weight",
                           "final_layer.norm.weight", "rope.inv_freq",
                           "time_embedder.proj_in.weight", "final_layer.video_out.weight",
                           "final_layer.audio_out.weight", "audio_patch_proj.bias"}) {
    const auto it = types.find(name);
    INFO("fp32 island " << name);
    REQUIRE(it != types.end());
    CHECK(it->second != 11u);  // not Q3_K
  }
}

TEST_CASE("minimax_h3: the REAL NVFP4 checkpoint lands on our existing NVFP4 layout") {
  // The NVFP4 arm is the SPEED path: sm_121 has native FP4 tensor cores and this
  // project already ships a tuned NVFP4 stack (cutlass FP4 GEMM, the Laguna arm).
  // This gates the real `minimax_h3_ref2va_nvfp4_full.safetensors` manifest, read
  // from the file's own header by range request — no payload downloaded.
  REQUIRE(vllm_test::kH3Nvfp4TensorCount ==
          static_cast<int64_t>(std::size(vllm_test::kH3Nvfp4Tensors)));

  std::map<std::string, const vllm_test::H3Nvfp4Tensor*> by_name;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) by_name[t.name] = &t;

  // The compressed-tensors NVFP4 triple, exactly as our loader already expects:
  //   weight         U8       packed FP4, 2 values per byte
  //   weight_scale   F8_E4M3  one per group of 16 along K
  //   weight_scale_2 F32      one global scale (scalar)
  int64_t packed = 0, block_scales = 0, global_scales = 0;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) {
    const std::string name = t.name;
    if (name.size() > 14 && name.compare(name.size() - 14, 14, "weight_scale_2") == 0) {
      CHECK(std::string(t.dtype) == "F32");
      CHECK(t.rank == 0);  // scalar
      ++global_scales;
    } else if (name.size() > 12 && name.compare(name.size() - 12, 12, "weight_scale") == 0) {
      CHECK(std::string(t.dtype) == "F8_E4M3");
      ++block_scales;
    } else if (std::string(t.dtype) == "U8") {
      ++packed;
    }
  }
  // Every quantized projection carries all three, so the counts must agree.
  CHECK(packed == block_scales);
  CHECK(packed == global_scales);
  CHECK(packed == 258);

  // Spot-check the geometry on the fused qkv of block 0. Logical [21504, 5376]:
  // 21504 = 3 * 56 * 128 output rows, 5376 = hidden.
  const auto qkv = by_name.find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(qkv != by_name.end());
  CHECK(std::string(qkv->second->dtype) == "U8");
  CHECK(qkv->second->rank == 2);
  CHECK(qkv->second->shape[0] == 21504);
  CHECK(qkv->second->shape[1] == 2688);  // 5376 FP4 values packed 2-per-byte

  const auto qkv_scale = by_name.find("blocks.0.attn.qkv_proj.weight_scale");
  REQUIRE(qkv_scale != by_name.end());
  CHECK(qkv_scale->second->shape[0] == 21504);
  CHECK(qkv_scale->second->shape[1] == 336);  // 5376 / 16 -> NVFP4 group size 16
  CHECK(qkv->second->shape[1] * 2 == qkv_scale->second->shape[1] * 16);

  // The fp32/bf16 ISLANDS must stay unquantized here too: quantizing a patch
  // projection, the time embedder, or an output head would break the dtype policy
  // the DiT forward depends on.
  for (const char* name : {"video_patch_proj.weight", "audio_patch_proj.weight",
                           "time_embedder.proj_in.weight", "time_embedder.proj_out.weight",
                           "final_layer.video_out.weight", "final_layer.audio_out.weight",
                           "rope.inv_freq", "blocks.0.norm1.weight",
                           "blocks.0.attn.q_norm.weight", "condition_proj.weight"}) {
    const auto it = by_name.find(name);
    INFO("unquantized island " << name);
    REQUIRE(it != by_name.end());
    CHECK(std::string(it->second->dtype) != "U8");
  }

  // The name map is again the IDENTITY: every non-quantization tensor in the real
  // checkpoint is a name our contract already knows.
  MiniMaxH3DitParams p;  // shipped geometry
  const std::vector<vllm::MiniMaxH3TensorSpec> contract = EnumerateMiniMaxH3DitTensors(p);
  std::map<std::string, std::vector<int64_t>> expected;
  for (const vllm::MiniMaxH3TensorSpec& spec : contract) expected[spec.name] = spec.shape;
  int64_t matched = 0;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) {
    const std::string name = t.name;
    if (name.find("weight_scale") != std::string::npos) continue;  // quant sidecars
    INFO("checkpoint tensor " << name);
    CHECK(expected.count(name) == 1);
    ++matched;
  }
  CHECK(matched == static_cast<int64_t>(contract.size()));
}

TEST_CASE("minimax_h3: the video VAE decoder block matches the checkpoint's own remote code") {
  // The shipped decoder is 36 of these blocks, so this is the repeated unit of the
  // half of the video VAE that generation actually needs. Gated against the
  // checkpoint's OWN base_module.TransformerBlock, executed at reduced dimensions
  // by scripts/gen-minimax-h3-video-vae-goldens.py.
  vllm::MiniMaxH3VideoVaeBlockConfig config;
  config.dim = vllm_test::kH3VideoVaeBlockDim;
  config.heads = vllm_test::kH3VideoVaeBlockHeads;
  config.dim_head = vllm_test::kH3VideoVaeBlockDimHead;
  config.ff_inner = vllm_test::kH3VideoVaeBlockFfInner;
  config.eps = 1e-5;

  const int64_t seq = vllm_test::kH3VideoVaeBlockSeq;
  const int64_t dim = config.dim;
  const int64_t inner = config.heads * config.dim_head;

  vllm::MiniMaxH3AudioVaeWeights weights;  // the same name-keyed parameter bag
  auto put = [&](const std::string& suffix, int64_t count, double scale, double offset) {
    weights.tensors["block." + suffix] = MakeParam("videovae." + suffix, count, scale, offset);
  };
  put("norm1.weight", dim, 0.1, 1.0);
  put("norm2.weight", dim, 0.1, 1.0);
  put("scale1", dim, 0.3, 0.0);
  put("scale2", dim, 0.3, 0.0);
  put("attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
  put("attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
  put("attn.to_out.weight", dim * inner, 0.1, 0.0);
  put("attn.to_out.bias", dim, 0.05, 0.0);
  put("ff.w1.weight", 2 * config.ff_inner * dim, 0.1, 0.0);
  put("ff.w1.bias", 2 * config.ff_inner, 0.05, 0.0);
  put("ff.w2.weight", dim * config.ff_inner, 0.1, 0.0);
  put("ff.w2.bias", dim, 0.05, 0.0);

  const std::vector<float> hidden = MakeParam("videovae.input", seq * dim, 1.0);
  const std::vector<float> got =
      vllm::MiniMaxH3VideoVaeBlockForward(config, weights, "block", hidden, seq);

  REQUIRE(got.size() == std::size(vllm_test::kH3VideoVaeBlockGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3VideoVaeBlockGolden, got.size());
  INFO("video VAE decoder block max|diff| = " << err);
  CHECK(err <= 1e-5);
}

TEST_CASE("minimax_h3: the FULL video-VAE ViT3D decoder matches the checkpoint's remote code") {
  // The whole generation-critical half of the video VAE: pack -> x_embedder ->
  // register/cls tokens -> 3D RoPE -> block stack -> norm_out -> proj_out ->
  // unpatchify. Gated against the checkpoint's OWN ViT3DDecoder.
  vllm::MiniMaxH3VideoVaeDecoderConfig config;
  config.block.dim = vllm_test::kH3VideoVaeBlockDim;
  config.block.heads = vllm_test::kH3VideoVaeBlockHeads;
  config.block.dim_head = vllm_test::kH3VideoVaeBlockDimHead;
  config.block.ff_inner = vllm_test::kH3VideoVaeBlockFfInner;
  config.block.eps = 1e-5;
  config.num_layers = vllm_test::kH3VideoVaeDecLayers;
  config.in_channels = vllm_test::kH3VideoVaeDecInCh;
  config.out_channels = vllm_test::kH3VideoVaeDecOutCh;
  config.patch_size = vllm_test::kH3VideoVaeDecPatch;
  config.patch_size_t = vllm_test::kH3VideoVaeDecPatchT;
  config.num_register_tokens = vllm_test::kH3VideoVaeDecRegisterTokens;
  // rope_apply_dim = int(dim_head * rope_dim_ratio 0.75); must divide by 2*n_dim.
  config.rope_apply_dim = static_cast<int64_t>(config.block.dim_head * 0.75);
  config.rope_theta = 100.0;

  const int64_t dim = config.block.dim;
  const int64_t inner = config.block.heads * config.block.dim_head;
  const int64_t lt = vllm_test::kH3VideoVaeDecT, lh = vllm_test::kH3VideoVaeDecH,
                lw = vllm_test::kH3VideoVaeDecW;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam("videovae.dec." + name, count, scale, offset);
  };
  put("x_embedder.weight", dim * config.in_channels, 0.1, 0.0);
  put("x_embedder.bias", dim, 0.05, 0.0);
  put("register_tokens", config.num_register_tokens * dim, 0.1, 0.0);
  put("norm_out.weight", dim, 0.1, 1.0);
  put("norm_out.bias", dim, 0.05, 0.0);
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;
  put("proj_out.weight", patch_dim * dim, 0.1, 0.0);
  put("proj_out.bias", patch_dim, 0.05, 0.0);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    const std::string b = "transformer_blocks." + std::to_string(l) + ".";
    put(b + "norm1.weight", dim, 0.1, 1.0);
    put(b + "norm2.weight", dim, 0.1, 1.0);
    put(b + "scale1", dim, 0.3, 0.0);
    put(b + "scale2", dim, 0.3, 0.0);
    put(b + "attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
    put(b + "attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
    put(b + "attn.to_out.weight", dim * inner, 0.1, 0.0);
    put(b + "attn.to_out.bias", dim, 0.05, 0.0);
    put(b + "ff.w1.weight", 2 * config.block.ff_inner * dim, 0.1, 0.0);
    put(b + "ff.w1.bias", 2 * config.block.ff_inner, 0.05, 0.0);
    put(b + "ff.w2.weight", dim * config.block.ff_inner, 0.1, 0.0);
    put(b + "ff.w2.bias", dim, 0.05, 0.0);
  }

  const std::vector<float> latent =
      MakeParam("videovae.dec.input", config.in_channels * lt * lh * lw, 1.0);
  vllm::MiniMaxH3VideoFrameShape shape;
  const std::vector<float> frames =
      vllm::MiniMaxH3VideoVaeDecode(config, weights, latent, lt, lh, lw, &shape);

  CHECK(shape.channels == config.out_channels);
  CHECK(shape.t == vllm_test::kH3VideoVaeDecFrameT);
  CHECK(shape.h == vllm_test::kH3VideoVaeDecFrameH);
  CHECK(shape.w == vllm_test::kH3VideoVaeDecFrameW);
  REQUIRE(frames.size() == std::size(vllm_test::kH3VideoVaeDecoderGolden));
  const double err = MaxAbsDiff(frames, vllm_test::kH3VideoVaeDecoderGolden, frames.size());
  INFO("video VAE ViT3D decoder max|diff| = " << err);
  CHECK(err <= 1e-4);
}

// The DEVICE decoder is held to the SAME upstream golden as the portable one --
// not to the portable one's output. Two independent implementations agreeing on a
// wrong answer is a failure mode this project has already been bitten by, and
// gating against the checkpoint's own remote code is what rules it out.
//
// This also covers the two load-time weight folds staging performs (the
// per-head-interleaved to_qkv row permutation, and the per-channel scale fold into
// the preceding projection). Both are silent-failure shaped: a wrong permutation
// still produces plausible finite frames.
static void CheckVideoVaeDecodeDevice(vt::Queue& queue, const char* label) {
  vllm::MiniMaxH3VideoVaeDecoderConfig config;
  config.block.dim = vllm_test::kH3VideoVaeBlockDim;
  config.block.heads = vllm_test::kH3VideoVaeBlockHeads;
  config.block.dim_head = vllm_test::kH3VideoVaeBlockDimHead;
  config.block.ff_inner = vllm_test::kH3VideoVaeBlockFfInner;
  config.block.eps = 1e-5;
  config.num_layers = vllm_test::kH3VideoVaeDecLayers;
  config.in_channels = vllm_test::kH3VideoVaeDecInCh;
  config.out_channels = vllm_test::kH3VideoVaeDecOutCh;
  config.patch_size = vllm_test::kH3VideoVaeDecPatch;
  config.patch_size_t = vllm_test::kH3VideoVaeDecPatchT;
  config.num_register_tokens = vllm_test::kH3VideoVaeDecRegisterTokens;
  config.rope_apply_dim = static_cast<int64_t>(config.block.dim_head * 0.75);
  config.rope_theta = 100.0;

  const int64_t dim = config.block.dim;
  const int64_t inner = config.block.heads * config.block.dim_head;
  const int64_t lt = vllm_test::kH3VideoVaeDecT, lh = vllm_test::kH3VideoVaeDecH,
                lw = vllm_test::kH3VideoVaeDecW;

  // The SAME synthetic weights the portable gate above builds, so the only
  // difference between the two runs is the implementation.
  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam("videovae.dec." + name, count, scale, offset);
  };
  put("x_embedder.weight", dim * config.in_channels, 0.1, 0.0);
  put("x_embedder.bias", dim, 0.05, 0.0);
  put("register_tokens", config.num_register_tokens * dim, 0.1, 0.0);
  put("norm_out.weight", dim, 0.1, 1.0);
  put("norm_out.bias", dim, 0.05, 0.0);
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;
  put("proj_out.weight", patch_dim * dim, 0.1, 0.0);
  put("proj_out.bias", patch_dim, 0.05, 0.0);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    const std::string b = "transformer_blocks." + std::to_string(l) + ".";
    put(b + "norm1.weight", dim, 0.1, 1.0);
    put(b + "norm2.weight", dim, 0.1, 1.0);
    put(b + "scale1", dim, 0.3, 0.0);
    put(b + "scale2", dim, 0.3, 0.0);
    put(b + "attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
    put(b + "attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
    put(b + "attn.to_out.weight", dim * inner, 0.1, 0.0);
    put(b + "attn.to_out.bias", dim, 0.05, 0.0);
    put(b + "ff.w1.weight", 2 * config.block.ff_inner * dim, 0.1, 0.0);
    put(b + "ff.w1.bias", 2 * config.block.ff_inner, 0.05, 0.0);
    put(b + "ff.w2.weight", dim * config.block.ff_inner, 0.1, 0.0);
    put(b + "ff.w2.bias", dim, 0.05, 0.0);
  }

  const std::vector<float> latent =
      MakeParam("videovae.dec.input", config.in_channels * lt * lh * lw, 1.0);

  const vllm::MiniMaxH3VideoVaeDeviceWeights staged =
      vllm::StageMiniMaxH3VideoVaeWeights(queue, config, weights);
  vllm::MiniMaxH3VideoFrameShape shape;
  const std::vector<float> frames = vllm::MiniMaxH3VideoVaeDecodeDevice(
      queue.device, config, staged, latent, lt, lh, lw, &shape);

  CHECK(shape.channels == config.out_channels);
  CHECK(shape.t == vllm_test::kH3VideoVaeDecFrameT);
  CHECK(shape.h == vllm_test::kH3VideoVaeDecFrameH);
  CHECK(shape.w == vllm_test::kH3VideoVaeDecFrameW);
  REQUIRE(frames.size() == std::size(vllm_test::kH3VideoVaeDecoderGolden));
  for (const float v : frames) REQUIRE(std::isfinite(v));
  const double err = MaxAbsDiff(frames, vllm_test::kH3VideoVaeDecoderGolden, frames.size());
  INFO("DEVICE video VAE ViT3D decoder (" << label << ") max|diff| = " << err);
  CHECK(err <= 1e-4);
}

// Tiling is what the untiled decode got WRONG at real resolutions, and the reason
// no earlier gate caught it: every reduced-dimension case here is smaller than one
// tile, where tiled and untiled are the same computation. So this gates BOTH ends.
//
// The decisive evidence is empirical (512x512 -- 3 tiles per axis upstream --
// decoded untiled came out covered in a grid of small squares, while 256x256, the
// one size where tile_size >= input, came out clean). What is gated HERE is the
// mechanism: that a single tile is still bit-identical, and that with several tiles
// each tile's interior equals a direct decode of that tile's own latent slice --
// which is what pins down the slicing, the placement and the seam arithmetic.
// The TEMPORAL chunk arithmetic, against upstream's decode_temporal (klvae.py).
// Pure integer geometry, so it pins the plan down without decoding anything --
// and the plan is the part that was WRONG (we decoded the whole latent in one
// pass where upstream never hands the ViT more than 7 temporal tokens).
TEST_CASE("minimax_h3: the video-VAE temporal chunk plan matches upstream") {
  vllm::MiniMaxH3VideoVaeDecoderConfig c;  // shipped defaults: 17 / 3 / 4
  CHECK(c.clip_length == 17);
  CHECK(c.token_drop == 3);
  CHECK(c.vae_ratio_t == 4);
  // tokens_chunk_size = ceil(17/4); frame_pre_padding = (-17) % 4;
  // token_overlap = (-3) % 5; frame_overlap = max(2*4 - 3, 0).
  CHECK(c.tokens_chunk_size() == 5);
  CHECK(c.frame_pre_padding() == 3);
  CHECK(c.token_overlap() == 2);
  CHECK(c.frame_overlap() == 5);

  // The modulo must be the PYTHON one: (-clip_length) % ratio is 3 in Python and
  // -1 in C, and a negative pre-padding would slice from the wrong end.
  CHECK(c.frame_pre_padding() >= 0);
  CHECK(c.token_overlap() >= 0);

  // For latent_t 12: pseudo = 12 + 3 = 15, a whole number of 5-token chunks, so
  // num_chunks = 15/5 - 1 = 2, each covering tokens [0,7) and [5,12).
  const int64_t latent_t = 12;
  const int64_t pseudo = latent_t + c.token_drop;
  CHECK(pseudo % c.tokens_chunk_size() == 0);
  CHECK(pseudo / c.tokens_chunk_size() - 1 == 2);
}

TEST_CASE("minimax_h3: the TILED video-VAE decode slices and places tiles correctly") {
  vt::Queue queue = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  vllm::MiniMaxH3VideoVaeDecoderConfig config;
  config.block.dim = vllm_test::kH3VideoVaeBlockDim;
  config.block.heads = vllm_test::kH3VideoVaeBlockHeads;
  config.block.dim_head = vllm_test::kH3VideoVaeBlockDimHead;
  config.block.ff_inner = vllm_test::kH3VideoVaeBlockFfInner;
  config.block.eps = 1e-5;
  config.num_layers = vllm_test::kH3VideoVaeDecLayers;
  config.in_channels = vllm_test::kH3VideoVaeDecInCh;
  config.out_channels = vllm_test::kH3VideoVaeDecOutCh;
  config.patch_size = vllm_test::kH3VideoVaeDecPatch;
  config.patch_size_t = vllm_test::kH3VideoVaeDecPatchT;
  config.num_register_tokens = vllm_test::kH3VideoVaeDecRegisterTokens;
  config.rope_apply_dim = static_cast<int64_t>(config.block.dim_head * 0.75);
  config.rope_theta = 100.0;

  const int64_t dim = config.block.dim;
  const int64_t inner = config.block.heads * config.block.dim_head;
  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam("videovae.dec." + name, count, scale, offset);
  };
  put("x_embedder.weight", dim * config.in_channels, 0.1, 0.0);
  put("x_embedder.bias", dim, 0.05, 0.0);
  put("register_tokens", config.num_register_tokens * dim, 0.1, 0.0);
  put("norm_out.weight", dim, 0.1, 1.0);
  put("norm_out.bias", dim, 0.05, 0.0);
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;
  put("proj_out.weight", patch_dim * dim, 0.1, 0.0);
  put("proj_out.bias", patch_dim, 0.05, 0.0);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    const std::string b = "transformer_blocks." + std::to_string(l) + ".";
    put(b + "norm1.weight", dim, 0.1, 1.0);
    put(b + "norm2.weight", dim, 0.1, 1.0);
    put(b + "scale1", dim, 0.3, 0.0);
    put(b + "scale2", dim, 0.3, 0.0);
    put(b + "attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
    put(b + "attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
    put(b + "attn.to_out.weight", dim * inner, 0.1, 0.0);
    put(b + "attn.to_out.bias", dim, 0.05, 0.0);
    put(b + "ff.w1.weight", 2 * config.block.ff_inner * dim, 0.1, 0.0);
    put(b + "ff.w1.bias", 2 * config.block.ff_inner, 0.05, 0.0);
    put(b + "ff.w2.weight", dim * config.block.ff_inner, 0.1, 0.0);
    put(b + "ff.w2.bias", dim, 0.05, 0.0);
  }
  const vllm::MiniMaxH3VideoVaeDeviceWeights staged =
      vllm::StageMiniMaxH3VideoVaeWeights(queue, config, weights);

  const int64_t ratio = vllm::kMiniMaxH3VaeRatio;
  const int64_t ps = config.patch_size;
  const int64_t lt = vllm_test::kH3VideoVaeDecT;

  SUBCASE("a canvas within one tile is the untiled decode, bit for bit") {
    const int64_t lh = 2, lw = 3;  // 32x48 canvas px -- far below tile_size 256
    REQUIRE(vllm::MiniMaxH3SplitTiles(lh * ratio, vllm::kMiniMaxH3VaeTileSize,
                                      vllm::kMiniMaxH3VaeTileOverlapMin, ratio)
                .starts.size() == 1u);
    const std::vector<float> latent =
        MakeParam("videovae.tile.small", config.in_channels * lt * lh * lw, 1.0);
    vllm::MiniMaxH3VideoFrameShape sa{}, sb{};
    const std::vector<float> untiled =
        vllm::MiniMaxH3VideoVaeDecodeDevice(queue.device, config, staged, latent, lt, lh, lw, &sa);
    const std::vector<float> tiled = vllm::MiniMaxH3VideoVaeDecodeTiledDevice(
        queue.device, config, staged, latent, lt, lh, lw, &sb);
    REQUIRE(tiled.size() == untiled.size());
    CHECK(sa.h == sb.h);
    CHECK(sa.w == sb.w);
    CHECK(MaxAbsDiff(tiled, untiled.data(), tiled.size()) == 0.0);
  }

  SUBCASE("a multi-tile canvas places each tile where the plan says") {
    // 20 latent units = 320 canvas px > tile_size 256, so the plan is 2 tiles per
    // axis (starts 0 and 64 px == latent 0 and 4, overlap 192 px == 12 latent).
    const int64_t lh = 20, lw = 20;
    const vllm::MiniMaxH3TilePlan plan = vllm::MiniMaxH3SplitTiles(
        lh * ratio, vllm::kMiniMaxH3VaeTileSize, vllm::kMiniMaxH3VaeTileOverlapMin, ratio);
    REQUIRE(plan.starts.size() == 2u);
    const int64_t tile_lat = plan.lengths[0] / ratio;
    const int64_t ov_lat = plan.overlaps[0] / ratio;

    const std::vector<float> latent =
        MakeParam("videovae.tile.big", config.in_channels * lt * lh * lw, 1.0);
    vllm::MiniMaxH3VideoFrameShape shape{};
    const std::vector<float> tiled = vllm::MiniMaxH3VideoVaeDecodeTiledDevice(
        queue.device, config, staged, latent, lt, lh, lw, &shape);

    CHECK(shape.channels == config.out_channels);
    CHECK(shape.t == lt * config.patch_size_t);
    CHECK(shape.h == lh * ps);
    CHECK(shape.w == lw * ps);
    REQUIRE(tiled.size() == static_cast<size_t>(shape.channels * shape.t * shape.h * shape.w));
    for (const float v : tiled) REQUIRE(std::isfinite(v));

    // Decode tile (0,0) on its own and require the assembled canvas to reproduce it
    // EXACTLY outside the blend region. This is what a wrong slice or a wrong
    // placement breaks -- and both still produce plausible finite frames.
    std::vector<float> sub(static_cast<size_t>(config.in_channels * lt * tile_lat * tile_lat));
    for (int64_t c = 0; c < config.in_channels; ++c) {
      for (int64_t tt = 0; tt < lt; ++tt) {
        for (int64_t y = 0; y < tile_lat; ++y) {
          for (int64_t x = 0; x < tile_lat; ++x) {
            sub[static_cast<size_t>(((c * lt + tt) * tile_lat + y) * tile_lat + x)] =
                latent[static_cast<size_t>(((c * lt + tt) * lh + y) * lw + x)];
          }
        }
      }
    }
    vllm::MiniMaxH3VideoFrameShape ts{};
    const std::vector<float> tile0 = vllm::MiniMaxH3VideoVaeDecodeTiledDevice(
        queue.device, config, staged, sub, lt, tile_lat, tile_lat, &ts);
    REQUIRE(ts.h == tile_lat * ps);

    // The un-blended interior: rows/cols before the first seam starts.
    const int64_t keep = (tile_lat - ov_lat) * ps;
    REQUIRE(keep > 0);
    double worst = 0.0;
    for (int64_t c = 0; c < shape.channels; ++c) {
      for (int64_t tt = 0; tt < shape.t; ++tt) {
        for (int64_t y = 0; y < keep; ++y) {
          for (int64_t x = 0; x < keep; ++x) {
            const double a = tiled[static_cast<size_t>(((c * shape.t + tt) * shape.h + y) *
                                                           shape.w + x)];
            const double b = tile0[static_cast<size_t>(((c * ts.t + tt) * ts.h + y) * ts.w + x)];
            worst = std::max(worst, std::abs(a - b));
          }
        }
      }
    }
    INFO("tile (0,0) interior vs a standalone decode, max|diff| = " << worst);
    CHECK(worst == 0.0);
  }
}

TEST_CASE("minimax_h3: the DEVICE video-VAE decoder matches the checkpoint's remote code") {
  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  CheckVideoVaeDecodeDevice(q, "cpu");
}

TEST_CASE("minimax_h3: the DEVICE video-VAE decoder matches the remote code on CUDA") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckVideoVaeDecodeDevice(q, "cuda");
}

TEST_CASE("minimax_h3: the video VAE decoder is a ViT, and its manifest says so") {
  // W4 scoping, grounded in the real checkpoint rather than in a guess: the video
  // VAE's ENCODER is the 3D CNN (down blocks with Conv3d), but its DECODER — the
  // half we actually need for generation — is a plain transformer stack. That
  // makes W4 materially smaller than "port a 48 KB klvae.py" suggested.
  REQUIRE(vllm_test::kH3VideoVaeTensorCount ==
          static_cast<int64_t>(std::size(vllm_test::kH3VideoVaeTensors)));

  int64_t decoder = 0, encoder = 0, quant_conv = 0, max_block = -1;
  bool saw_conv3d = false;
  for (const vllm_test::H3VideoVaeTensor& t : vllm_test::kH3VideoVaeTensors) {
    const std::string name = t.name;
    CHECK(std::string(t.dtype) == "F32");  // the source checkpoint is fp32 throughout
    if (name.rfind("decoder.", 0) == 0) ++decoder;
    if (name.rfind("encoder.", 0) == 0) {
      ++encoder;
      if (t.rank == 5) saw_conv3d = true;  // [out, in, kt, kh, kw]
    }
    if (name.rfind("quant_conv", 0) == 0 || name.rfind("post_quant_conv", 0) == 0) ++quant_conv;
    const std::string prefix = "decoder.transformer_blocks.";
    if (name.rfind(prefix, 0) == 0) {
      const size_t start = prefix.size();
      size_t end = start;
      while (end < name.size() && name[end] >= '0' && name[end] <= '9') ++end;
      if (end > start) max_block = std::max<int64_t>(max_block, std::stoll(name.substr(start, end - start)));
    }
  }
  CHECK(decoder == 440);
  CHECK(encoder == 116);
  CHECK(quant_conv == 4);
  CHECK(saw_conv3d);          // the ENCODER is the 3D CNN
  CHECK(max_block == 35);     // the DECODER is a 36-block transformer

  // Each decoder block carries exactly the transformer parts we already have
  // primitives for: fused qkv attention, a 2-matrix feed-forward, two norms, and
  // two learned residual scales.
  std::map<std::string, bool> present;
  for (const vllm_test::H3VideoVaeTensor& t : vllm_test::kH3VideoVaeTensors) present[t.name] = true;
  for (const char* suffix : {"attn.to_qkv.weight", "attn.to_qkv.bias", "attn.to_out.weight",
                             "attn.to_out.bias", "ff.w1.weight", "ff.w2.weight",
                             "norm1.weight", "norm2.weight", "scale1", "scale2"}) {
    const std::string name = "decoder.transformer_blocks.0." + std::string(suffix);
    INFO("decoder block part " << name);
    CHECK(present.count(name) == 1);
  }
  // Plus the ViT surround: patch embed, learned mask/register tokens, output head.
  for (const char* name : {"decoder.x_embedder.weight", "decoder.mask_token",
                           "decoder.register_tokens", "decoder.norm_out.weight",
                           "decoder.proj_out.weight", "post_quant_conv.weight"}) {
    INFO("decoder surround " << name);
    CHECK(present.count(name) == 1);
  }
}

TEST_CASE("minimax_h3: the encoder text tower matches upstream, with all three H3 deltas") {
  // The H3-Encoder produces the [seq, 5120] prompt_embeds the DiT consumes. Its
  // ARCHITECTURE is a Qwen3-VL (which this project already ports); what is
  // H3-specific are three deltas, and all three are exercised here:
  //   layer truncation, the UNNORMALIZED output, and DeepStack injection.
  vllm::MiniMaxH3EncoderConfig config;
  config.hidden_size = vllm_test::kH3EncHidden;
  config.num_hidden_layers = vllm_test::kH3EncConfigLayers;
  config.selected_layer = vllm_test::kH3EncSelectedLayer;
  config.num_attention_heads = vllm_test::kH3EncHeads;
  config.num_key_value_heads = vllm_test::kH3EncKvHeads;
  config.head_dim = vllm_test::kH3EncHeadDim;
  config.intermediate_size = vllm_test::kH3EncIntermediate;
  config.rms_norm_eps = 1e-6;
  config.rope_theta = 10000.0;
  config.mrope_section.assign(vllm_test::kH3EncMropeSection,
                              vllm_test::kH3EncMropeSection + 3);

  // DELTA 1: the config claims more layers than are kept.
  CHECK(vllm::MiniMaxH3EncoderNumLayers(config.num_hidden_layers, config.selected_layer) ==
        vllm_test::kH3EncSelectedLayer);
  CHECK(vllm_test::kH3EncConfigLayers > vllm_test::kH3EncSelectedLayer);
  // The shipped rule is 50-of-N.
  CHECK(vllm::kMiniMaxH3EncoderSelectedLayer == 50);
  CHECK(vllm::kMiniMaxH3EncoderHiddenDim == 5120);
  // Truncation must never EXTEND a shallower model.
  CHECK(vllm::MiniMaxH3EncoderNumLayers(8, 50) == 8);

  const int64_t seq = vllm_test::kH3EncSeq;
  const int64_t hidden = config.hidden_size;
  const int64_t q_width = config.num_attention_heads * config.head_dim;
  const int64_t kv_width = config.num_key_value_heads * config.head_dim;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam("encoder." + name, count, scale, offset);
  };
  put("embed_tokens.weight", vllm_test::kH3EncVocab * hidden, 0.1, 0.0);
  for (int64_t l = 0; l < vllm_test::kH3EncSelectedLayer; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    put(p + "self_attn.qkv_proj.weight", (q_width + 2 * kv_width) * hidden, 0.1, 0.0);
    put(p + "self_attn.o_proj.weight", hidden * q_width, 0.1, 0.0);
    put(p + "self_attn.q_norm.weight", config.head_dim, 0.1, 1.0);
    put(p + "self_attn.k_norm.weight", config.head_dim, 0.1, 1.0);
    put(p + "mlp.gate_up_proj.weight", 2 * config.intermediate_size * hidden, 0.1, 0.0);
    put(p + "mlp.down_proj.weight", hidden * config.intermediate_size, 0.1, 0.0);
    put(p + "input_layernorm.weight", hidden, 0.1, 1.0);
    put(p + "post_attention_layernorm.weight", hidden, 0.1, 1.0);
  }

  const std::vector<float> inputs_embeds =
      MakeParam("encoder.inputs_embeds", seq * hidden, 1.0);
  std::vector<int64_t> positions(static_cast<size_t>(3 * seq));
  for (int64_t axis = 0; axis < 3; ++axis) {
    for (int64_t s = 0; s < seq; ++s) positions[static_cast<size_t>(axis * seq + s)] = s;
  }

  // DELTAS 2 + 3: the plain path returns the UNNORMALIZED state; the DeepStack
  // path additionally injects visual features into the first N layers.
  const std::vector<float> plain = vllm::MiniMaxH3EncoderTextForward(
      config, weights, inputs_embeds, positions.data(), seq, nullptr, {});
  REQUIRE(plain.size() == std::size(vllm_test::kH3EncGolden));
  const double plain_err = MaxAbsDiff(plain, vllm_test::kH3EncGolden, plain.size());
  INFO("encoder (plain) max|diff| = " << plain_err);
  CHECK(plain_err <= 1e-4);

  std::vector<uint8_t> visual_mask(static_cast<size_t>(seq));
  for (int64_t s = 0; s < seq; ++s) {
    visual_mask[static_cast<size_t>(s)] =
        static_cast<uint8_t>(vllm_test::kH3EncVisualMask[s]);
  }
  std::vector<std::vector<float>> deepstack;
  for (int64_t i = 0; i < vllm_test::kH3EncDeepstackLayers; ++i) {
    deepstack.push_back(MakeParam("encoder.deepstack." + std::to_string(i),
                                  vllm_test::kH3EncNumVisual * hidden, 0.1, 0.0));
  }
  const std::vector<float> injected = vllm::MiniMaxH3EncoderTextForward(
      config, weights, inputs_embeds, positions.data(), seq, visual_mask.data(), deepstack);
  REQUIRE(injected.size() == std::size(vllm_test::kH3EncDeepstackGolden));
  const double deep_err =
      MaxAbsDiff(injected, vllm_test::kH3EncDeepstackGolden, injected.size());
  INFO("encoder (deepstack) max|diff| = " << deep_err);
  CHECK(deep_err <= 1e-4);

  // DeepStack must actually change the result, or the test proves nothing.
  double delta = 0.0;
  for (size_t i = 0; i < plain.size(); ++i) {
    delta = std::max(delta, std::abs(static_cast<double>(plain[i]) - injected[i]));
  }
  CHECK(delta > 1e-5);
}

// fl2va CONDITIONING, end to end through the pipeline. Everything below was
// already ported and gated in PIECES -- the packed keyframe layout, the VAE 3D-CNN
// encoder, condition-noise augmentation, the denoise loop's pinned condition rows.
// What did not exist was any path from "caller has an image" to "the DiT is
// conditioned on it", so the whole feature was unreachable.
//
// The load-bearing assertion is that conditioning CHANGES THE RESULT. A wiring
// that accepted the rows and ignored them would produce perfectly finite,
// correctly shaped output and pass every structural check.
// The video VAE works in IMAGENET-NORMALIZED pixel space, and the conversions sit
// OUTSIDE ViT3DDecoder -- so the decoder's own 1.19e-07 gate never covered them,
// exactly like post_quant_conv before it. Omitting the output step feeds
// ImageNet-normalized values to a writer that expects [-1, 1]: it casts colour (the
// per-channel means differ) and compresses the dynamic range ~4.4x (std ~0.22),
// which is what "dark and washed out" looks like.
// The decoded AUDIO must last as long as the decoded VIDEO. Nothing checked this,
// and it was wrong: `audio_t` is the PER-CHANNEL latent length (planner: 40 Hz *
// duration) but the pipeline divided it by the channel count, halving the audio.
// Because the muxer passes `-shortest`, that silently truncated the VIDEO too --
// a 124-frame render produced a 61-frame MP4. Every structural check passed
// throughout: shapes were self-consistent, just half as long as intended.
//
// Found by RENDERING, not by the suite, which is the part worth recording: the
// existing gates all assert shape SELF-CONSISTENCY, and a uniformly halved
// pipeline is self-consistent. The invariant below is the one that is not.
TEST_CASE("minimax_h3: decoded audio spans the same duration as the video") {
  // Planner geometry: audio latents run at 40 Hz over num_frames / fps seconds.
  const int64_t num_frames = 124;
  const double seconds = static_cast<double>(num_frames) / vllm::kMiniMaxH3Fps;
  const int64_t audio_t = vllm::MiniMaxH3AudioLatentT(seconds);
  INFO("num_frames=" << num_frames << " seconds=" << seconds << " audio_t=" << audio_t);

  // 40 Hz * 5.1667 s = 207 latent steps PER CHANNEL, not 103.
  CHECK(audio_t == 207);

  // The invariant the bug broke: latent steps / 40 Hz must equal the video
  // duration, to within one latent step.
  const double audio_seconds = static_cast<double>(audio_t) / 40.0;
  INFO("audio " << audio_seconds << " s vs video " << seconds << " s");
  CHECK(std::abs(audio_seconds - seconds) <= 1.0 / 40.0);

  // And the halving specifically: dividing by the 2 channels would give ~half the
  // duration, which is what shipped.
  const double halved = static_cast<double>(audio_t / vllm::kMiniMaxH3AudioChannels) / 40.0;
  CHECK(std::abs(halved - seconds) > 1.0);  // the wrong value is off by ~2.5 s
}

TEST_CASE("minimax_h3: ImageNet pixel de/normalization matches upstream's wrapper") {
  const int64_t n = 5;
  // Round trip: normalize then de-normalize must return the original, for values
  // that stay inside the [0,1] clamp.
  std::vector<float> pixels;
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t i = 0; i < n; ++i) pixels.push_back(-0.6f + 0.3f * static_cast<float>(i));
  }
  std::vector<float> round = pixels;
  vllm::MiniMaxH3VideoNormalizePixels(round, 3, n);
  vllm::MiniMaxH3VideoDenormalizePixels(round, 3, n);
  CHECK(MaxAbsDiff(round, pixels.data(), pixels.size()) <= 1e-5);

  // Exact values against upstream's formula (vae.py:659,693), per channel -- a
  // shared mean/std would round-trip fine yet still cast colour.
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float sd[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> one(3, 0.0f);  // one value per channel, ImageNet-normalized 0
  vllm::MiniMaxH3VideoDenormalizePixels(one, 3, 1);
  for (int64_t c = 0; c < 3; ++c) {
    CHECK(one[static_cast<size_t>(c)] == doctest::Approx(mean[c] * 2.0f - 1.0f).epsilon(1e-5));
  }

  // The clamp happens BEFORE the [-1,1] map (vae.py:693 order): a far
  // out-of-gamut value must saturate at exactly +/-1, not overshoot.
  std::vector<float> hot = {50.0f, -50.0f, 50.0f};
  vllm::MiniMaxH3VideoDenormalizePixels(hot, 3, 1);
  CHECK(hot[0] == doctest::Approx(1.0f));
  CHECK(hot[1] == doctest::Approx(-1.0f));
  CHECK(hot[2] == doctest::Approx(1.0f));

  // And the per-channel std is really applied: an ImageNet-normalized 1.0 must
  // land at a DIFFERENT pixel per channel.
  std::vector<float> unit = {1.0f, 1.0f, 1.0f};
  vllm::MiniMaxH3VideoDenormalizePixels(unit, 3, 1);
  for (int64_t c = 0; c < 3; ++c) {
    CHECK(unit[static_cast<size_t>(c)] ==
          doctest::Approx((mean[c] + sd[c]) * 2.0f - 1.0f).epsilon(1e-5));
  }
  CHECK(unit[0] != doctest::Approx(unit[1]));
}

TEST_CASE("minimax_h3: fl2va keyframe conditioning is wired and load-bearing") {
  const std::unique_ptr<GoldenWeights> dit = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = dit->params;

  // A reduced 3D-CNN encoder: one level, tiny channels. Real geometry is irrelevant
  // here -- what matters is that a real encode feeds real rows into the loop.
  vllm::MiniMaxH3EncoderFcn3dConfig ecfg;
  ecfg.ch = 8;
  ecfg.ch_mult = {1};
  ecfg.space_down = {1};
  ecfg.time_down = {1};
  ecfg.num_res_blocks = 1;
  ecfg.in_channels = 3;
  ecfg.z_channels = 2 * p.latents_dim;  // moments width: mean | logvar
  ecfg.num_groups = 4;

  const int64_t IH = 8, IW = 8;
  vllm::MiniMaxH3AudioVaeWeights ew;
  auto put = [&](const std::string& n, int64_t c, double s, double o) {
    ew.tensors[n] = MakeParam("fl2va.enc." + n, c, s, o);
  };
  put("conv_in.weight", ecfg.ch * ecfg.in_channels * 27, 0.05, 0.0);
  put("conv_in.bias", ecfg.ch, 0.01, 0.0);
  for (int64_t b = 0; b < ecfg.num_res_blocks; ++b) {
    const std::string pre = "down.0.block." + std::to_string(b) + ".";
    put(pre + "norm1.weight", ecfg.ch, 0.1, 1.0);
    put(pre + "norm1.bias", ecfg.ch, 0.05, 0.0);
    put(pre + "conv1.weight", ecfg.ch * ecfg.ch * 27, 0.05, 0.0);
    put(pre + "conv1.bias", ecfg.ch, 0.01, 0.0);
    put(pre + "norm2.weight", ecfg.ch, 0.1, 1.0);
    put(pre + "norm2.bias", ecfg.ch, 0.05, 0.0);
    put(pre + "conv2.weight", ecfg.ch * ecfg.ch * 27, 0.05, 0.0);
    put(pre + "conv2.bias", ecfg.ch, 0.01, 0.0);
  }
  put("norm_out.weight", ecfg.ch, 0.1, 1.0);
  put("norm_out.bias", ecfg.ch, 0.05, 0.0);
  put("conv_out.weight", ecfg.z_channels * ecfg.ch * 27, 0.05, 0.0);
  put("conv_out.bias", ecfg.z_channels, 0.01, 0.0);
  put("quant_conv.weight", ecfg.z_channels * ecfg.z_channels, 0.05, 0.0);
  put("quant_conv.bias", ecfg.z_channels, 0.01, 0.0);

  const std::vector<float> image = MakeParam("fl2va.image", ecfg.in_channels * IH * IW, 1.0);

  SUBCASE("the encoder yields a deterministic latent (the MEAN, not a sample)") {
    vllm::MiniMaxH3VideoFrameShape ls{}, ls2{};
    vllm::MiniMaxH3EncoderFcn3dConfig c = ecfg;
    c.t = 1; c.h = IH; c.w = IW;
    const std::vector<float> a1 = vllm::MiniMaxH3VideoVaeEncodeToLatent(c, ew, image, &ls);
    const std::vector<float> a2 = vllm::MiniMaxH3VideoVaeEncodeToLatent(c, ew, image, &ls2);
    REQUIRE(!a1.empty());
    CHECK(ls.t == ls2.t);
    // z_channels out, not 2*z: the logvar half must be dropped.
    CHECK(static_cast<int64_t>(a1.size()) == (ecfg.z_channels / 2) * ls.t * ls.h * ls.w);
    CHECK(MaxAbsDiff(a1, a2.data(), a1.size()) == 0.0);
    for (const float v : a1) REQUIRE(std::isfinite(v));
  }

  SUBCASE("ref2va IMAGE references are wired, and audio references are REFUSED") {
    vllm::MiniMaxH3T2vaRequest req;
    req.text_len = 4;
    req.latent_t = 2;
    req.latent_h = 4;
    req.latent_w = 4;
    req.audio_t = 4;
    req.audio_channel = 2;
    req.num_steps = 3;

    std::vector<vllm::MiniMaxH3RefBlock> blocks;
    const std::vector<float> ref_rows = vllm::MiniMaxH3EncodeReferenceImages(
        ecfg, ew, p, {image}, IH, IW, &blocks);
    REQUIRE(blocks.size() == 1);
    CHECK(blocks[0].kind == vllm::MiniMaxH3RefBlock::Kind::kImage);
    CHECK(blocks[0].latent_t >= 1);
    REQUIRE(!ref_rows.empty());

    // INVARIANT (RED-first, the spec section 8.9 ref2va-grid root cause): the
    // encoded reference row COUNT must equal exactly what the packed layout
    // allocates for the reference span. MiniMaxH3EncodeReferenceImages returns the
    // RAW VAE-latent grid in its block; BuildMiniMaxH3PackedSequenceRef2va (mirroring
    // upstream) applies the DiT [1,2,2] patch division itself. If the encode
    // PRE-divides (the bug that shipped the grid), the layout under-allocates the
    // reference by patch_h*patch_w and the denoise loop silently truncates the
    // pinned reference to its first quarter -- coherence-destroying but non-throwing,
    // because keyframe_cond_rows is then LONGER than the layout and passes the
    // pin-loop's `>=` check. Nothing else in the suite couples encoded-rows to
    // layout-rows: the section-2 layout gate hand-builds unpatched blocks, and the
    // denoise round-trip below only checks finiteness + motion. This is the coupling.
    const vllm::MiniMaxH3PackedSequence ref_layout =
        vllm::BuildMiniMaxH3PackedSequenceRef2va(req.text_len, req.latent_t, req.latent_h,
                                                 req.latent_w, req.audio_t, blocks,
                                                 req.audio_channel);
    int64_t layout_ref_rows = 0;
    for (const uint8_t upd : ref_layout.update_mask) layout_ref_rows += (upd == 0 ? 1 : 0);
    const int64_t block_grid_rows =
        (blocks[0].latent_h / p.patch_size_h) * (blocks[0].latent_w / p.patch_size_w);
    CHECK(layout_ref_rows == block_grid_rows);
    CHECK(layout_ref_rows * p.video_row_width() == static_cast<int64_t>(ref_rows.size()));

    const int64_t frame_rows = (req.latent_h / p.patch_size_h) * (req.latent_w / p.patch_size_w);
    const std::vector<float> prompt = MakeParam("ref2va.prompt", req.text_len * p.text_dim, 0.2);
    const std::vector<float> nv =
        MakeParam("ref2va.nv", req.latent_t * frame_rows * p.video_row_width(), 0.3);
    const std::vector<float> na =
        MakeParam("ref2va.na", req.audio_t * req.audio_channel * p.audio_latents_dim, 0.3);

    const vllm::MiniMaxH3DenoiseResult plain = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, req, p, dit->views, prompt, nv, na, vt::DType::kF32);

    vllm::MiniMaxH3T2vaRequest r2 = req;
    r2.ref_blocks = blocks;
    r2.keyframe_cond_rows = ref_rows;  // the same pinned-condition mechanism
    const vllm::MiniMaxH3DenoiseResult withref = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, r2, p, dit->views, prompt, nv, na, vt::DType::kF32);
    for (const float v : withref.video_rows) REQUIRE(std::isfinite(v));

    const size_t n = std::min(plain.video_rows.size(), withref.video_rows.size());
    REQUIRE(n > 0);
    double delta = 0.0;
    for (size_t i = 0; i < n; ++i) {
      delta = std::max(delta,
                       std::abs(static_cast<double>(plain.video_rows[i]) - withref.video_rows[i]));
    }
    INFO("ref2va image reference moved the video rows by " << delta);
    CHECK(delta > 1e-5);

    // An audio-bearing block that supplies NO rows must still be refused: the
    // layout would grow around packed rows nothing ever wrote, and the loop would
    // pin whatever happened to be in the buffer.
    vllm::MiniMaxH3T2vaRequest bad = req;
    vllm::MiniMaxH3RefBlock ab;
    ab.kind = vllm::MiniMaxH3RefBlock::Kind::kAudio;
    ab.ref_audio_t = 2;
    bad.ref_blocks = {ab};
    CHECK_THROWS(vllm::MiniMaxH3DenoiseT2va(vt::Device{}, bad, p, dit->views, prompt, nv, na,
                                            vt::DType::kF32));
    // ... and so must one that supplies the WRONG number of rows.
    bad.audio_ref_rows = MakeParam("ref2va.short", (ab.ref_audio_t * req.audio_channel - 1) *
                                                       p.audio_latents_dim, 0.3);
    CHECK_THROWS(vllm::MiniMaxH3DenoiseT2va(vt::Device{}, bad, p, dit->views, prompt, nv, na,
                                            vt::DType::kF32));

    // fl2va and ref2va are mutually exclusive and must say so.
    vllm::MiniMaxH3T2vaRequest both = req;
    both.ref_blocks = blocks;
    both.keyframe_frame_indices = {0};
    CHECK_THROWS(vllm::MiniMaxH3DenoiseT2va(vt::Device{}, both, p, dit->views, prompt, nv, na,
                                            vt::DType::kF32));
  }

  SUBCASE("ref2va VIDEO references carry a temporal extent, and stay silent") {
    const int64_t FT = 3;  // a 3-frame reference clip
    const std::vector<float> clip =
        MakeParam("ref2va.clip", ecfg.in_channels * FT * IH * IW, 1.0);
    vllm::MiniMaxH3RefBlock vb{};
    const std::vector<float> vrows = vllm::MiniMaxH3EncodeReferenceVideo(
        ecfg, ew, p, clip, FT, IH, IW, &vb);
    REQUIRE(!vrows.empty());
    CHECK(vb.kind == vllm::MiniMaxH3RefBlock::Kind::kVideoAudio);
    // ref_audio_t 0 is what keeps it honest: no audio rows are claimed, because
    // the audio-VAE encoder that would produce them is not ported.
    CHECK(vb.ref_audio_t == 0);
    CHECK(vb.latent_t >= 1);

    // A video reference must occupy MORE rows than a single image one: kImage
    // counts exactly one frame however large its latent_t, so a clip that came
    // back as kImage would silently lose its temporal extent.
    std::vector<vllm::MiniMaxH3RefBlock> iblocks;
    const std::vector<float> irows =
        vllm::MiniMaxH3EncodeReferenceImages(ecfg, ew, p, {image}, IH, IW, &iblocks);
    REQUIRE(iblocks.size() == 1);
    CHECK(vrows.size() > irows.size());

    // Same encoded-vs-layout row-count invariant as the image path (section 8.9):
    // the RAW-latent block dims the video encode emits must produce a layout whose
    // reference span holds exactly the encoded rows. A patch-pre-division here would
    // silently truncate the pinned video reference, so pin it in the suite too.
    {
      const vllm::MiniMaxH3PackedSequence vlayout = vllm::BuildMiniMaxH3PackedSequenceRef2va(
          4, 2, 4, 4, 4, {vb}, 2);
      int64_t vref_rows = 0;
      for (const uint8_t upd : vlayout.update_mask) vref_rows += (upd == 0 ? 1 : 0);
      CHECK(vref_rows * p.video_row_width() == static_cast<int64_t>(vrows.size()));
    }

    vllm::MiniMaxH3T2vaRequest req;
    req.text_len = 4;
    req.latent_t = 2;
    req.latent_h = 4;
    req.latent_w = 4;
    req.audio_t = 4;
    req.audio_channel = 2;
    req.num_steps = 3;
    const int64_t frame_rows = (req.latent_h / p.patch_size_h) * (req.latent_w / p.patch_size_w);
    const std::vector<float> prompt = MakeParam("refvid.prompt", req.text_len * p.text_dim, 0.2);
    const std::vector<float> nv =
        MakeParam("refvid.nv", req.latent_t * frame_rows * p.video_row_width(), 0.3);
    const std::vector<float> na =
        MakeParam("refvid.na", req.audio_t * req.audio_channel * p.audio_latents_dim, 0.3);

    const vllm::MiniMaxH3DenoiseResult plain = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, req, p, dit->views, prompt, nv, na, vt::DType::kF32);

    vllm::MiniMaxH3T2vaRequest rv = req;
    rv.ref_blocks = {vb};
    rv.keyframe_cond_rows = vrows;
    const vllm::MiniMaxH3DenoiseResult withvid = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, rv, p, dit->views, prompt, nv, na, vt::DType::kF32);
    for (const float v : withvid.video_rows) REQUIRE(std::isfinite(v));
    const size_t n = std::min(plain.video_rows.size(), withvid.video_rows.size());
    REQUIRE(n > 0);
    double delta = 0.0;
    for (size_t i = 0; i < n; ++i) {
      delta = std::max(delta,
                       std::abs(static_cast<double>(plain.video_rows[i]) - withvid.video_rows[i]));
    }
    INFO("ref2va video reference moved the video rows by " << delta);
    CHECK(delta > 1e-5);

    // A video reference WITH audio but WITHOUT the encoded audio rows must be
    // refused: its audio slots would otherwise be pinned to nothing.
    vllm::MiniMaxH3T2vaRequest bad = req;
    vllm::MiniMaxH3RefBlock loud = vb;
    loud.ref_audio_t = 2;
    bad.ref_blocks = {loud};
    bad.keyframe_cond_rows = vrows;
    CHECK_THROWS(vllm::MiniMaxH3DenoiseT2va(vt::Device{}, bad, p, dit->views, prompt, nv, na,
                                            vt::DType::kF32));
  }

  SUBCASE("keyframe rows reach the denoise loop and CHANGE the result") {
    vllm::MiniMaxH3T2vaRequest req;
    req.text_len = 4;
    req.latent_t = 2;
    req.latent_h = 4;
    req.latent_w = 4;
    req.audio_t = 4;
    req.audio_channel = 2;
    req.num_frames = 5;
    req.num_steps = 3;

    const int64_t frame_rows = (req.latent_h / p.patch_size_h) * (req.latent_w / p.patch_size_w);
    const std::vector<float> prompt =
        MakeParam("fl2va.prompt", req.text_len * p.text_dim, 0.2);
    const std::vector<float> nv =
        MakeParam("fl2va.nv", req.latent_t * frame_rows * p.video_row_width(), 0.3);
    const std::vector<float> na =
        MakeParam("fl2va.na", req.audio_t * req.audio_channel * p.audio_latents_dim, 0.3);

    const vllm::MiniMaxH3DenoiseResult plain = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, req, p, dit->views, prompt, nv, na, vt::DType::kF32);

    vllm::MiniMaxH3T2vaRequest cond = req;
    cond.keyframe_frame_indices = {0};
    cond.keyframe_cond_rows = vllm::MiniMaxH3EncodeKeyframeCondRows(
        ecfg, ew, p, {image}, IH, IW, req.latent_t, /*noise_aug=*/1.0, {});
    REQUIRE(!cond.keyframe_cond_rows.empty());

    // The layout itself must differ: a keyframe adds condition rows.
    const vllm::MiniMaxH3PackedSequence pk_plain = vllm::BuildMiniMaxH3PackedSequence(
        req.text_len, req.latent_t, req.latent_h, req.latent_w, req.audio_t, req.audio_channel,
        false, {}, 0);
    const vllm::MiniMaxH3PackedSequence pk_cond = vllm::BuildMiniMaxH3PackedSequence(
        req.text_len, req.latent_t, req.latent_h, req.latent_w, req.audio_t, req.audio_channel,
        true, {0}, req.num_frames);
    CHECK(pk_cond.img_pos.size() > pk_plain.img_pos.size());

    const vllm::MiniMaxH3DenoiseResult conditioned = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, cond, p, dit->views, prompt, nv, na, vt::DType::kF32);

    for (const float v : conditioned.video_rows) REQUIRE(std::isfinite(v));
    REQUIRE(!conditioned.video_rows.empty());

    // THE assertion: identical prompt, identical noise, identical schedule -- the
    // only difference is the keyframe, so the video MUST move. Ignoring the rows
    // would still produce finite, correctly shaped output.
    const size_t n = std::min(plain.video_rows.size(), conditioned.video_rows.size());
    REQUIRE(n > 0);
    double delta = 0.0;
    for (size_t i = 0; i < n; ++i) {
      delta = std::max(delta, std::abs(static_cast<double>(plain.video_rows[i]) -
                                       conditioned.video_rows[i]));
    }
    INFO("keyframe conditioning moved the video rows by " << delta);
    CHECK(delta > 1e-5);
  }

  // --- the AUDIO half of ref2va. A reduced audio VAE encoder whose latent width
  // is the DiT's own audio row width, so the rows it emits ARE packed rows.
  vllm::MiniMaxH3AudioVaeEncoderConfig acfg;
  acfg.encoder_dim = 4;
  acfg.encoder_rates = {2, 2};
  acfg.latent_dim = 16;
  acfg.vae_latent_channels = p.audio_latents_dim;
  acfg.attn_proj = true;
  acfg.attn_proj_heads = 2;
  const vllm::MiniMaxH3AudioVaeWeights aw = BuildAudioEncoderWeights(acfg, "ref2va.aud.");
  const int64_t kRefSamples = 24;  // 24 / hop 4 -> 6 latent frames

  SUBCASE("ref2va AUDIO references are wired, and CHANGE the result") {
    vllm::MiniMaxH3T2vaRequest req;
    req.text_len = 4;
    req.latent_t = 2;
    req.latent_h = 4;
    req.latent_w = 4;
    req.audio_t = 4;
    req.audio_channel = 2;
    req.num_steps = 3;

    // A stereo reference waveform, encoded by the audio VAE's ENCODER half.
    const std::vector<float> stereo =
        MakeParam("ref2va.wave", req.audio_channel * kRefSamples, 0.9);
    vllm::MiniMaxH3RefBlock ab{};
    const std::vector<float> arows = vllm::MiniMaxH3EncodeReferenceAudio(
        acfg, aw, stereo, req.audio_channel, kRefSamples, {}, {}, /*noise_aug=*/1.0, {}, &ab);
    CHECK(ab.kind == vllm::MiniMaxH3RefBlock::Kind::kAudio);
    CHECK(ab.ref_audio_t == kRefSamples / acfg.hop_length());
    REQUIRE(arows.size() ==
            static_cast<size_t>(ab.ref_audio_t * req.audio_channel * p.audio_latents_dim));
    for (const float v : arows) REQUIRE(std::isfinite(v));

    // The layout must GROW by exactly the rows the block claims -- an audio
    // reference occupies packed audio positions, it does not overwrite targets.
    const vllm::MiniMaxH3PackedSequence pk_plain = vllm::BuildMiniMaxH3PackedSequence(
        req.text_len, req.latent_t, req.latent_h, req.latent_w, req.audio_t, req.audio_channel,
        false, {}, 0);
    const vllm::MiniMaxH3PackedSequence pk_ref = vllm::BuildMiniMaxH3PackedSequenceRef2va(
        req.text_len, req.latent_t, req.latent_h, req.latent_w, req.audio_t, {ab},
        req.audio_channel);
    CHECK(static_cast<int64_t>(pk_ref.audio_pos.size()) ==
          static_cast<int64_t>(pk_plain.audio_pos.size()) + ab.ref_audio_t * req.audio_channel);

    const int64_t frame_rows = (req.latent_h / p.patch_size_h) * (req.latent_w / p.patch_size_w);
    const std::vector<float> prompt = MakeParam("ref2aud.prompt", req.text_len * p.text_dim, 0.2);
    const std::vector<float> nv =
        MakeParam("ref2aud.nv", req.latent_t * frame_rows * p.video_row_width(), 0.3);
    const std::vector<float> na =
        MakeParam("ref2aud.na", req.audio_t * req.audio_channel * p.audio_latents_dim, 0.3);

    const vllm::MiniMaxH3DenoiseResult plain = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, req, p, dit->views, prompt, nv, na, vt::DType::kF32);

    vllm::MiniMaxH3T2vaRequest ra = req;
    ra.ref_blocks = {ab};
    ra.audio_ref_rows = arows;
    const vllm::MiniMaxH3DenoiseResult withaudio = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, ra, p, dit->views, prompt, nv, na, vt::DType::kF32);
    for (const float v : withaudio.audio_rows) REQUIRE(std::isfinite(v));
    for (const float v : withaudio.video_rows) REQUIRE(std::isfinite(v));

    // THE assertion, the audio-side twin of the keyframe one: identical prompt,
    // identical noise, identical schedule -- the only difference is the reference
    // audio, so BOTH streams must move. Wiring that accepts the rows and ignores
    // them would still produce finite, correctly shaped output.
    const size_t na_n = std::min(plain.audio_rows.size(), withaudio.audio_rows.size());
    REQUIRE(na_n > 0);
    double audio_delta = 0.0;
    for (size_t i = 0; i < na_n; ++i) {
      audio_delta = std::max(audio_delta, std::abs(static_cast<double>(plain.audio_rows[i]) -
                                                   withaudio.audio_rows[i]));
    }
    INFO("ref2va audio reference moved the audio rows by " << audio_delta);
    CHECK(audio_delta > 1e-5);

    const size_t nv_n = std::min(plain.video_rows.size(), withaudio.video_rows.size());
    REQUIRE(nv_n > 0);
    double video_delta = 0.0;
    for (size_t i = 0; i < nv_n; ++i) {
      video_delta = std::max(video_delta, std::abs(static_cast<double>(plain.video_rows[i]) -
                                                   withaudio.video_rows[i]));
    }
    INFO("ref2va audio reference moved the VIDEO rows by " << video_delta);
    CHECK(video_delta > 1e-5);

    // And the reference itself is LOAD-BEARING, not merely present: a DIFFERENT
    // waveform must give a different result. Wiring that reserved the rows and
    // then pinned a constant would pass every check above and fail this one.
    const std::vector<float> other =
        MakeParam("ref2va.wave.other", req.audio_channel * kRefSamples, 0.9);
    vllm::MiniMaxH3RefBlock ob{};
    vllm::MiniMaxH3T2vaRequest rb = ra;
    rb.audio_ref_rows = vllm::MiniMaxH3EncodeReferenceAudio(
        acfg, aw, other, req.audio_channel, kRefSamples, {}, {}, /*noise_aug=*/1.0, {}, &ob);
    CHECK(ob.ref_audio_t == ab.ref_audio_t);
    const vllm::MiniMaxH3DenoiseResult otheraudio = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, rb, p, dit->views, prompt, nv, na, vt::DType::kF32);
    double which_delta = 0.0;
    for (size_t i = 0; i < na_n; ++i) {
      which_delta = std::max(which_delta, std::abs(static_cast<double>(withaudio.audio_rows[i]) -
                                                  otheraudio.audio_rows[i]));
    }
    INFO("a DIFFERENT reference waveform moved the audio rows by " << which_delta);
    CHECK(which_delta > 1e-5);
  }

  SUBCASE("ref2va VIDEO+AUDIO references carry both, and both are load-bearing") {
    const int64_t FT = 3;
    const std::vector<float> clip =
        MakeParam("ref2va.avclip", ecfg.in_channels * FT * IH * IW, 1.0);
    vllm::MiniMaxH3RefBlock vb{};
    const std::vector<float> vrows =
        vllm::MiniMaxH3EncodeReferenceVideo(ecfg, ew, p, clip, FT, IH, IW, &vb);
    REQUIRE(!vrows.empty());
    CHECK(vb.ref_audio_t == 0);  // silent until an encoded waveform says otherwise

    vllm::MiniMaxH3T2vaRequest req;
    req.text_len = 4;
    req.latent_t = 2;
    req.latent_h = 4;
    req.latent_w = 4;
    req.audio_t = 4;
    req.audio_channel = 2;
    req.num_steps = 3;

    const std::vector<float> stereo =
        MakeParam("ref2va.avwave", req.audio_channel * kRefSamples, 0.9);
    vllm::MiniMaxH3RefBlock ab{};
    const std::vector<float> arows = vllm::MiniMaxH3EncodeReferenceAudio(
        acfg, aw, stereo, req.audio_channel, kRefSamples, {}, {}, /*noise_aug=*/1.0, {}, &ab);
    // ONE block carrying both: the clip's geometry plus the waveform's extent.
    vllm::MiniMaxH3RefBlock av = vb;
    av.ref_audio_t = ab.ref_audio_t;
    CHECK(av.kind == vllm::MiniMaxH3RefBlock::Kind::kVideoAudio);
    CHECK(av.ref_audio_t > 0);

    const int64_t frame_rows = (req.latent_h / p.patch_size_h) * (req.latent_w / p.patch_size_w);
    const std::vector<float> prompt = MakeParam("ref2av.prompt", req.text_len * p.text_dim, 0.2);
    const std::vector<float> nv =
        MakeParam("ref2av.nv", req.latent_t * frame_rows * p.video_row_width(), 0.3);
    const std::vector<float> na =
        MakeParam("ref2av.na", req.audio_t * req.audio_channel * p.audio_latents_dim, 0.3);

    // The SILENT same-clip reference is the control: the only difference is sound.
    vllm::MiniMaxH3T2vaRequest silent = req;
    silent.ref_blocks = {vb};
    silent.keyframe_cond_rows = vrows;
    const vllm::MiniMaxH3DenoiseResult quiet = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, silent, p, dit->views, prompt, nv, na, vt::DType::kF32);

    vllm::MiniMaxH3T2vaRequest loud = req;
    loud.ref_blocks = {av};
    loud.keyframe_cond_rows = vrows;
    loud.audio_ref_rows = arows;
    const vllm::MiniMaxH3DenoiseResult sounded = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, loud, p, dit->views, prompt, nv, na, vt::DType::kF32);
    for (const float v : sounded.audio_rows) REQUIRE(std::isfinite(v));
    for (const float v : sounded.video_rows) REQUIRE(std::isfinite(v));

    const size_t na_n = std::min(quiet.audio_rows.size(), sounded.audio_rows.size());
    REQUIRE(na_n > 0);
    double audio_delta = 0.0;
    for (size_t i = 0; i < na_n; ++i) {
      audio_delta = std::max(audio_delta, std::abs(static_cast<double>(quiet.audio_rows[i]) -
                                                  sounded.audio_rows[i]));
    }
    INFO("ref2va video+audio moved the audio rows by " << audio_delta);
    CHECK(audio_delta > 1e-5);

    const size_t nv_n = std::min(quiet.video_rows.size(), sounded.video_rows.size());
    REQUIRE(nv_n > 0);
    double video_delta = 0.0;
    for (size_t i = 0; i < nv_n; ++i) {
      video_delta = std::max(video_delta, std::abs(static_cast<double>(quiet.video_rows[i]) -
                                                  sounded.video_rows[i]));
    }
    INFO("ref2va video+audio moved the VIDEO rows by " << video_delta);
    CHECK(video_delta > 1e-5);

    // The VISUAL half must still be load-bearing once audio joins it: encode a
    // different clip and the result must move again.
    const std::vector<float> clip2 =
        MakeParam("ref2va.avclip2", ecfg.in_channels * FT * IH * IW, 1.0);
    vllm::MiniMaxH3RefBlock vb2{};
    vllm::MiniMaxH3T2vaRequest loud2 = loud;
    loud2.keyframe_cond_rows =
        vllm::MiniMaxH3EncodeReferenceVideo(ecfg, ew, p, clip2, FT, IH, IW, &vb2);
    const vllm::MiniMaxH3DenoiseResult sounded2 = vllm::MiniMaxH3DenoiseT2va(
        vt::Device{}, loud2, p, dit->views, prompt, nv, na, vt::DType::kF32);
    double visual_delta = 0.0;
    for (size_t i = 0; i < nv_n; ++i) {
      visual_delta = std::max(visual_delta, std::abs(static_cast<double>(sounded.video_rows[i]) -
                                                    sounded2.video_rows[i]));
    }
    INFO("a DIFFERENT reference clip (with the same audio) moved the video rows by "
         << visual_delta);
    CHECK(visual_delta > 1e-5);
  }
}

TEST_CASE("minimax_h3: the WHOLE t2va path composes end to end") {
  // Not a quality result -- reduced dimensions and random weights -- but a real
  // structural end-to-end exercise of the assembled pipeline: packed layout ->
  // sigma schedules -> a multi-step denoise loop of DiT forwards -> unpatchify /
  // audio unpack -> denormalize -> BOTH VAE decoders -> frames + stereo waveform.
  // Each stage is separately gated against upstream; this proves they COMPOSE and
  // that shapes and finiteness survive the whole path.
  const std::unique_ptr<GoldenWeights> dit = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = dit->params;

  vllm::MiniMaxH3T2vaRequest request;
  request.text_len = 4;
  request.latent_t = 2;
  request.latent_h = 4;
  request.latent_w = 4;
  request.audio_t = 4;      // audio rows = audio_t * audio_channel
  request.audio_channel = 2;
  request.num_steps = 3;    // 3 schedule points -> 2 denoise steps
  request.video_shift = 12.0;
  request.audio_shift = 3.0;

  // --- video VAE decoder sized to the DiT's video latent ---
  vllm::MiniMaxH3VideoVaeDecoderConfig video_config;
  video_config.block.dim = 16;
  video_config.block.heads = 2;
  video_config.block.dim_head = 8;
  video_config.block.ff_inner = 16;
  video_config.block.eps = 1e-5;
  video_config.num_layers = 1;
  video_config.in_channels = p.latents_dim;
  video_config.out_channels = 3;
  video_config.patch_size = 2;
  video_config.patch_size_t = 1;
  video_config.num_register_tokens = 2;
  video_config.rope_apply_dim = 6;
  video_config.rope_theta = 100.0;

  vllm::MiniMaxH3AudioVaeWeights video_weights;
  {
    const int64_t dim = video_config.block.dim;
    const int64_t inner = video_config.block.heads * video_config.block.dim_head;
    auto put = [&](const std::string& n, int64_t c, double sc, double off) {
      video_weights.tensors[n] = MakeParam("e2e.vvae." + n, c, sc, off);
    };
    put("x_embedder.weight", dim * video_config.in_channels, 0.1, 0.0);
    put("x_embedder.bias", dim, 0.05, 0.0);
    put("register_tokens", video_config.num_register_tokens * dim, 0.1, 0.0);
    put("norm_out.weight", dim, 0.1, 1.0);
    put("norm_out.bias", dim, 0.05, 0.0);
    const int64_t patch_dim = video_config.out_channels * video_config.patch_size_t *
                              video_config.patch_size * video_config.patch_size;
    put("proj_out.weight", patch_dim * dim, 0.1, 0.0);
    put("proj_out.bias", patch_dim, 0.05, 0.0);
    for (int64_t l = 0; l < video_config.num_layers; ++l) {
      const std::string b = "transformer_blocks." + std::to_string(l) + ".";
      put(b + "norm1.weight", dim, 0.1, 1.0);
      put(b + "norm2.weight", dim, 0.1, 1.0);
      put(b + "scale1", dim, 0.1, 0.0);
      put(b + "scale2", dim, 0.1, 0.0);
      put(b + "attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
      put(b + "attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
      put(b + "attn.to_out.weight", dim * inner, 0.1, 0.0);
      put(b + "attn.to_out.bias", dim, 0.05, 0.0);
      put(b + "ff.w1.weight", 2 * video_config.block.ff_inner * dim, 0.1, 0.0);
      put(b + "ff.w1.bias", 2 * video_config.block.ff_inner, 0.05, 0.0);
      put(b + "ff.w2.weight", dim * video_config.block.ff_inner, 0.1, 0.0);
      put(b + "ff.w2.bias", dim, 0.05, 0.0);
    }
  }

  // --- audio VAE, with dec_in_proj so it consumes the DiT's audio latent width ---
  vllm::MiniMaxH3AudioVaeConfig audio_config;
  audio_config.num_mels = 8;
  audio_config.upsample_initial_channel = 8;
  audio_config.upsample_rates = {2};
  audio_config.upsample_kernel_sizes = {4};
  audio_config.resblock_kernel_sizes = {3};
  audio_config.resblock_dilation_sizes = {{1}};
  audio_config.use_tanh_at_final = false;
  audio_config.use_bias_at_final = false;
  audio_config.snake_logscale = true;

  vllm::MiniMaxH3AudioVaeWeights audio_weights;
  {
    auto put = [&](const std::string& n, int64_t c, double sc, double off) {
      audio_weights.tensors[n] = MakeParam("e2e.avae." + n, c, sc, off);
    };
    auto put_conv = [&](const std::string& prefix, int64_t oc, int64_t ic, int64_t k, bool bias) {
      put(prefix + ".parametrizations.weight.original0", oc, 0.03, 0.15);
      put(prefix + ".parametrizations.weight.original1", oc * ic * k, 0.08, 0.0);
      if (bias) put(prefix + ".bias", oc, 0.05, 0.0);
    };
    // dec_in_proj: the DiT's audio latent width -> num_mels.
    put("dec_in_proj.weight", audio_config.num_mels * p.audio_latents_dim, 0.1, 0.0);
    put("dec_in_proj.bias", audio_config.num_mels, 0.05, 0.0);
    put_conv("conv_pre", audio_config.upsample_initial_channel, audio_config.num_mels, 7, true);
    const int64_t ch = audio_config.upsample_initial_channel / 2;
    put("ups.0.0.parametrizations.weight.original0", audio_config.upsample_initial_channel, 0.03, 0.15);
    put("ups.0.0.parametrizations.weight.original1",
        audio_config.upsample_initial_channel * ch * 4, 0.08, 0.0);
    put("ups.0.0.bias", ch, 0.05, 0.0);
    put_conv("resblocks.0.convs1.0", ch, ch, 3, true);
    put_conv("resblocks.0.convs2.0", ch, ch, 3, true);
    for (const char* a : {"resblocks.0.activations.0", "resblocks.0.activations.1",
                          "activation_post"}) {
      put(std::string(a) + ".act.alpha", ch, 0.2, 0.0);
      put(std::string(a) + ".act.beta", ch, 0.2, 0.0);
    }
    put_conv("conv_post", 1, ch, 7, false);
  }

  // --- inputs ---
  const std::vector<float> prompt_embeds =
      MakeParam("e2e.prompt_embeds", request.text_len * p.text_dim, 1.0);
  const int64_t frame_rows = (request.latent_h / p.patch_size_h) * (request.latent_w / p.patch_size_w);
  const int64_t video_rows = request.latent_t * frame_rows;
  const int64_t audio_rows = request.audio_t * request.audio_channel;
  const std::vector<float> noise_video =
      MakeParam("e2e.noise_video", video_rows * p.video_row_width(), 1.0);
  const std::vector<float> noise_audio =
      MakeParam("e2e.noise_audio", audio_rows * p.audio_latents_dim, 1.0);

  const vllm::MiniMaxH3T2vaResult out = vllm::MiniMaxH3GenerateT2va(
      Cpu(), request, p, dit->views, video_config, video_weights, audio_config, audio_weights,
      prompt_embeds, noise_video, noise_audio, vt::DType::kF32);

  // Frames: [3, T*pt, H*ps, W*ps].
  CHECK(out.frame_shape.channels == 3);
  CHECK(out.frame_shape.t == request.latent_t * video_config.patch_size_t);
  CHECK(out.frame_shape.h == request.latent_h * video_config.patch_size);
  CHECK(out.frame_shape.w == request.latent_w * video_config.patch_size);
  CHECK(static_cast<int64_t>(out.frames.size()) ==
        out.frame_shape.channels * out.frame_shape.t * out.frame_shape.h * out.frame_shape.w);
  for (float v : out.frames) REQUIRE(std::isfinite(v));

  // Audio: stereo, in [-1, 1], at the H3 sample rate.
  CHECK(out.audio_channels == request.audio_channel);
  CHECK(out.sample_rate == vllm::kMiniMaxH3AudioSampleRate);
  CHECK(out.audio_samples_per_channel > 0);

  // DURATION, not merely self-consistency. `audio_t` is the PER-CHANNEL latent
  // length; the pipeline used to divide it by the channel count, so the decoded
  // audio ran half the video's length and `-shortest` truncated the MP4 to match
  // (a 124-frame render muxed as 61). Every shape check still passed, because a
  // uniformly halved pipeline is self-consistent. The decoder is linear in latent
  // steps, so decoding `audio_t` steps INDEPENDENTLY and requiring the pipeline to
  // have produced exactly that many samples is the check that is not
  // self-consistency: with the bug this is off by the channel count.
  {
    const std::vector<float> probe(
        static_cast<size_t>(p.audio_latents_dim * request.audio_t), 0.0f);
    int64_t expected_samples = 0;
    vllm::MiniMaxH3AudioVaeDecode(audio_config, audio_weights, probe, request.audio_t,
                                  &expected_samples);
    INFO("audio_t=" << request.audio_t << " channels=" << request.audio_channel
                    << " expected=" << expected_samples
                    << " got=" << out.audio_samples_per_channel);
    CHECK(out.audio_samples_per_channel == expected_samples);
    // And the halved value is genuinely different, so the check has teeth.
    int64_t halved_samples = 0;
    const std::vector<float> halved_probe(
        static_cast<size_t>(p.audio_latents_dim * (request.audio_t / request.audio_channel)),
        0.0f);
    vllm::MiniMaxH3AudioVaeDecode(audio_config, audio_weights, halved_probe,
                                  request.audio_t / request.audio_channel, &halved_samples);
    CHECK(halved_samples != expected_samples);
  }
  CHECK(static_cast<int64_t>(out.waveform.size()) ==
        out.audio_channels * out.audio_samples_per_channel);
  for (float v : out.waveform) {
    REQUIRE(std::isfinite(v));
    CHECK(v >= -1.0f);
    CHECK(v <= 1.0f);
  }

  // The denoise loop must have MOVED the latents: if the output equalled the noise
  // the pipeline would be silently bypassing the DiT.
  double moved = 0.0;
  for (size_t i = 0; i < noise_video.size(); ++i) {
    moved = std::max(moved, static_cast<double>(std::abs(noise_video[i])));
  }
  CHECK(moved > 0.0);

  // DEVICE PATH through the WHOLE pipeline. The denoise loop stages the DiT weights
  // once and runs every step device-resident; this asserts the resulting video and
  // audio match the CPU pipeline. Without it the device wiring is reachable only
  // through the DiT-forward unit test, not through the loop that actually drives it.
  {
    vt::Backend* cuda = nullptr;
    try {
      cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
    } catch (...) {
      cuda = nullptr;
    }
    if (cuda == nullptr) {
      MESSAGE("SKIP: no CUDA backend registered (device t2va)");
    } else {
      const vt::Queue dq = cuda->CreateQueue();
      const vllm::MiniMaxH3T2vaResult dev = vllm::MiniMaxH3GenerateT2va(
          dq.device, request, p, dit->views, video_config, video_weights, audio_config,
          audio_weights, prompt_embeds, noise_video, noise_audio, vt::DType::kF32);
      REQUIRE(dev.frames.size() == out.frames.size());
      REQUIRE(dev.waveform.size() == out.waveform.size());
      double fmax = 0.0, amax = 0.0;
      for (size_t i = 0; i < out.frames.size(); ++i) {
        fmax = std::max(fmax, std::abs(static_cast<double>(dev.frames[i] - out.frames[i])));
      }
      for (size_t i = 0; i < out.waveform.size(); ++i) {
        amax = std::max(amax, std::abs(static_cast<double>(dev.waveform[i] - out.waveform[i])));
      }
      INFO("device-vs-cpu t2va: frames " << fmax << ", waveform " << amax);
      // Not bit-identical by construction (the device path reuses the tuned shared
      // ops, whose reductions differ), so this is the same tolerance class the DiT
      // forward is held to, carried through the VAEs.
      CHECK(fmax <= 2e-3);
      CHECK(amax <= 2e-3);
      for (float v : dev.frames) REQUIRE(std::isfinite(v));
    }
  }

  // post_quant_conv: the real checkpoint ships it, and the pipeline must APPLY it
  // to the latent before decoding. The weight set above deliberately omits it (a
  // reduced-dimension set need not carry every wrapper tensor), so re-run WITH one
  // and require the frames to CHANGE. Without this the wiring is a branch no test
  // enters, which is how the step went missing in the first place.
  vllm::MiniMaxH3AudioVaeWeights video_weights_pqc = video_weights;
  const int64_t lc = p.latents_dim;
  std::vector<float> pqc(static_cast<size_t>(lc * lc), 0.0f);
  for (int64_t i = 0; i < lc; ++i) pqc[static_cast<size_t>(i * lc + i)] = 1.0f;  // identity...
  pqc[1] = 0.25f;  // ...plus one off-diagonal term, so it genuinely MIXES channels
  video_weights_pqc.tensors["post_quant_conv.weight"] = pqc;
  video_weights_pqc.tensors["post_quant_conv.bias"] = std::vector<float>(lc, 0.0f);

  const vllm::MiniMaxH3T2vaResult out_pqc = vllm::MiniMaxH3GenerateT2va(
      Cpu(), request, p, dit->views, video_config, video_weights_pqc, audio_config, audio_weights,
      prompt_embeds, noise_video, noise_audio, vt::DType::kF32);
  REQUIRE(out_pqc.frames.size() == out.frames.size());
  double frame_delta = 0.0;
  for (size_t i = 0; i < out.frames.size(); ++i) {
    frame_delta = std::max(frame_delta,
                           std::abs(static_cast<double>(out_pqc.frames[i] - out.frames[i])));
  }
  INFO("post_quant_conv frame delta = " << frame_delta);
  CHECK(frame_delta > 1e-6);
  // The AUDIO path must be untouched by a VIDEO-side wrapper tensor.
  REQUIRE(out_pqc.waveform.size() == out.waveform.size());
  for (size_t i = 0; i < out.waveform.size(); ++i) {
    CHECK(out_pqc.waveform[i] == out.waveform[i]);
  }
}

TEST_CASE("minimax_h3: a ComfyUI-format GGUF loads into a runnable DiT") {
  // Closes the GGUF arm: the manifest test proved names and shapes resolve; this
  // proves a real GGUF file DEQUANTIZES into weights the forward actually runs.
  // A synthetic checkpoint is used so the test needs no download; the shapes and
  // the `comfy.gguf.orig_shape` reshape rule mirror the real file exactly.
  MiniMaxH3DitParams want;  // the geometry the loader must RECOVER from shapes
  want.num_layers = 2;
  want.token_refiner_num_layers = 1;
  want.hidden_size = 64;
  want.num_attention_heads = 4;
  want.attention_head_dim = 16;
  want.ffn_hidden_size = 128;
  want.latents_dim = 8;
  want.audio_latents_dim = 6;
  want.text_dim = 24;
  want.timestep_input_dim = 16;
  want.time_embed_hidden_size = 64;
  want.time_embed_dim = 32;
  want.adaln_out_features = 18 * want.hidden_size;
  want.final_adaln_out_features = 2 * want.hidden_size;
  want.rope_inv_freq_len = 2;

  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "wan"));

  // GGUF stores `ne` REVERSED vs torch, so a logical [out, in] weight is written
  // [in, out]. F32 everywhere keeps the test about the LOADER, not about quant
  // error (the K-quant families go through the same shared dequant entry point).
  auto add = [&](const std::string& name, const std::vector<int64_t>& logical) {
    int64_t numel = 1;
    for (int64_t d : logical) numel *= d;
    const std::vector<float> values = MakeParam("gguf." + name, numel, 0.1);
    std::string bytes(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = logical.rbegin(); it != logical.rend(); ++it) {
      ne.push_back(static_cast<uint64_t>(*it));
    }
    builder.AddTensor(name, ne, /*ggml_type=*/0 /*F32*/, bytes);
  };

  const int64_t inner = want.num_attention_heads * want.attention_head_dim;
  const int64_t video_width = want.video_row_width();
  add("video_patch_proj.weight", {want.hidden_size, video_width});
  add("video_patch_proj.bias", {want.hidden_size});
  add("audio_patch_proj.weight", {want.hidden_size, want.audio_latents_dim});
  add("audio_patch_proj.bias", {want.hidden_size});
  add("condition_proj.weight", {want.hidden_size, want.text_dim});
  add("condition_proj.bias", {want.hidden_size});
  add("time_embedder.proj_in.weight", {want.time_embed_hidden_size, want.timestep_input_dim});
  add("time_embedder.proj_in.bias", {want.time_embed_hidden_size});
  add("time_embedder.proj_out.weight", {want.time_embed_dim, want.time_embed_hidden_size});
  add("time_embedder.proj_out.bias", {want.time_embed_dim});
  add("rope.inv_freq", {want.rope_inv_freq_len});
  auto add_block = [&](const std::string& prefix, bool with_adaln) {
    add(prefix + ".norm1.weight", {want.hidden_size});
    add(prefix + ".norm2.weight", {want.hidden_size});
    add(prefix + ".attn.qkv_proj.weight", {3 * inner, want.hidden_size});
    add(prefix + ".attn.q_norm.weight", {want.attention_head_dim});
    add(prefix + ".attn.k_norm.weight", {want.attention_head_dim});
    add(prefix + ".attn.out_proj.weight", {want.hidden_size, inner});
    add(prefix + ".mlp.fc1.weight", {2 * want.ffn_hidden_size, want.hidden_size});
    add(prefix + ".mlp.fc2.weight", {want.hidden_size, want.ffn_hidden_size});
    if (with_adaln) {
      add(prefix + ".adaln_proj.linear.weight", {want.adaln_out_features, want.time_embed_dim});
      add(prefix + ".adaln_proj.linear.bias", {want.adaln_out_features});
    }
  };
  for (int64_t i = 0; i < want.token_refiner_num_layers; ++i) {
    add_block("token_refiner.blocks." + std::to_string(i), false);
  }
  add("token_refiner.final_norm.weight", {want.hidden_size});
  for (int64_t i = 0; i < want.num_layers; ++i) {
    add_block("blocks." + std::to_string(i), true);
  }
  add("final_layer.norm.weight", {want.hidden_size});
  add("final_layer.adaln_proj.linear.weight", {want.final_adaln_out_features, want.time_embed_dim});
  add("final_layer.adaln_proj.linear.bias", {want.final_adaln_out_features});
  add("final_layer.video_out.weight", {video_width, want.hidden_size});
  add("final_layer.video_out.bias", {video_width});
  add("final_layer.audio_out.weight", {want.audio_latents_dim, want.hidden_size});
  add("final_layer.audio_out.bias", {want.audio_latents_dim});

  const std::string path = "/tmp/minimax_h3_loader_test.gguf";
  {
    const std::string bytes = builder.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), fh) == bytes.size());
    std::fclose(fh);
  }

  const vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::MiniMaxH3GgufDit loaded = vllm::LoadMiniMaxH3DitFromGguf(gguf);

  // The geometry must be RECOVERED from tensor shapes alone.
  CHECK(loaded.params.num_layers == want.num_layers);
  CHECK(loaded.params.token_refiner_num_layers == want.token_refiner_num_layers);
  CHECK(loaded.params.hidden_size == want.hidden_size);
  CHECK(loaded.params.num_attention_heads == want.num_attention_heads);
  CHECK(loaded.params.attention_head_dim == want.attention_head_dim);
  CHECK(loaded.params.ffn_hidden_size == want.ffn_hidden_size);
  CHECK(loaded.params.latents_dim == want.latents_dim);
  CHECK(loaded.params.audio_latents_dim == want.audio_latents_dim);
  CHECK(loaded.params.text_dim == want.text_dim);
  CHECK(loaded.params.rope_inv_freq_len == want.rope_inv_freq_len);
  CHECK(static_cast<int64_t>(loaded.weights.blocks.size()) == want.num_layers);
  CHECK(static_cast<int64_t>(loaded.weights.refiner.size()) == want.token_refiner_num_layers);

  // A loaded weight must carry the LOGICAL (torch) shape, not the reversed ne.
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[0] == 3 * inner);
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[1] == want.hidden_size);

  // ★ THE fp32-ISLAND RULE BINDS THE bf16 HOST LOADER TOO (#244). The rule is
  // single-sourced in MiniMaxH3IsFp32IslandTensor and its contract says every
  // staging path must agree; this HOST loader was a fifth path that re-listed
  // two of the seven names by hand, so five islands silently became bf16 and no
  // gate noticed. Two hazards ride on the same predicate: rope.inv_freq and
  // adaln_t_table are read through an unchecked Ptr<float>() (bf16 bits
  // reinterpreted as garbage floats), while the patch projections, the time
  // embedder and the output heads are correctly TYPED but must not be silently
  // down-converted on one path only.
  {
    const vllm::MiniMaxH3GgufDit bf16_loaded = vllm::LoadMiniMaxH3DitFromGgufBf16(gguf);
    // Assert on the loader's OWN storage split rather than the bound views: this
    // is the decision under test, and it is the map a tensor lands in that
    // decides whether the forward later reads f32 bits or bf16 bits.
    size_t islands = 0, streamed = 0;
    for (const auto& kv : bf16_loaded.shapes) {
      const std::string& iname = kv.first;
      const bool is_island = vllm::MiniMaxH3IsFp32IslandTensor(iname);
      INFO("tensor " << iname << (is_island ? " (fp32 island)" : " (bf16 stream)"));
      CHECK(bf16_loaded.storage.count(iname) == (is_island ? 1u : 0u));
      CHECK(bf16_loaded.bf16_storage.count(iname) == (is_island ? 0u : 1u));
      if (is_island) ++islands; else ++streamed;
    }
    // The fixture carries both patch projections, the time embedder, both output
    // heads and rope.inv_freq, so the island set is non-trivial -- a predicate
    // that matched nothing would pass the loop above vacuously.
    CHECK(islands >= 6);
    CHECK(streamed > islands);
  }

  // And the whole thing must actually RUN: a real forward off GGUF-loaded weights.
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      4, 2, 4, 4, 2, 2, /*include_keyframe_cond=*/false, {}, 0);
  const int64_t seq = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const std::vector<float> x(static_cast<size_t>(seq * video_width), 0.25f);
  const std::vector<float> audio_x(static_cast<size_t>(seq * want.audio_latents_dim), 0.1f);
  const std::vector<float> prompt(static_cast<size_t>(num_text * want.text_dim), 0.2f);
  const std::vector<float> unique_ts = {0.4f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};
  MiniMaxH3DitInputs in;
  in.seq_len = seq;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_ts.data();
  in.num_unique_timesteps = 1;
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForward(Cpu(), loaded.params, loaded.weights, in, vt::DType::kF32);
  CHECK(static_cast<int64_t>(got.video_logits.size()) == num_img * video_width);
  CHECK(static_cast<int64_t>(got.audio_logits.size()) == num_audio * want.audio_latents_dim);
  for (float v : got.video_logits) REQUIRE(std::isfinite(v));
  for (float v : got.audio_logits) REQUIRE(std::isfinite(v));
  std::remove(path.c_str());
}

TEST_CASE("minimax_h3: an NVFP4 checkpoint loads into a runnable DiT") {
  // Closes the NVFP4 arm's loader: the manifest test proved the real checkpoint's
  // layout IS ours; this proves a file in that layout dequantizes into weights the
  // forward runs. Synthetic so no download is needed, but the triple is built
  // exactly as the real file stores it.
  MiniMaxH3DitParams want;
  want.num_layers = 1;
  want.token_refiner_num_layers = 1;
  want.hidden_size = 64;
  want.num_attention_heads = 4;
  want.attention_head_dim = 16;
  want.ffn_hidden_size = 128;
  want.latents_dim = 8;
  want.audio_latents_dim = 6;
  want.text_dim = 32;   // must be a multiple of 16 (the NVFP4 group)
  want.timestep_input_dim = 16;
  want.time_embed_hidden_size = 64;
  want.time_embed_dim = 32;
  want.adaln_out_features = 18 * want.hidden_size;
  want.final_adaln_out_features = 2 * want.hidden_size;
  want.rope_inv_freq_len = 2;

  const int64_t inner = want.num_attention_heads * want.attention_head_dim;
  const int64_t video_width = want.video_row_width();
  const std::string path = "/tmp/minimax_h3_nvfp4_test.safetensors";
  WriteMiniMaxH3Nvfp4File(want, path);

  const vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(path);
  const vllm::MiniMaxH3GgufDit loaded = vllm::LoadMiniMaxH3DitFromNvfp4(st);

  // Geometry recovered from the DEQUANTIZED shapes: a packed [out, in/2] weight
  // must come back as the logical [out, in].
  CHECK(loaded.params.hidden_size == want.hidden_size);
  CHECK(loaded.params.num_attention_heads == want.num_attention_heads);
  CHECK(loaded.params.attention_head_dim == want.attention_head_dim);
  CHECK(loaded.params.ffn_hidden_size == want.ffn_hidden_size);
  CHECK(loaded.params.text_dim == want.text_dim);
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[0] == 3 * inner);
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[1] == want.hidden_size);
  CHECK(loaded.shapes.at("blocks.0.mlp.fc1.weight")[1] == want.hidden_size);
  // The quant sidecars must NOT appear as model tensors.
  CHECK(loaded.storage.count("blocks.0.attn.qkv_proj.weight_scale") == 0);
  CHECK(loaded.storage.count("blocks.0.attn.qkv_proj.weight_scale_2") == 0);

  // The STREAMING stager must recover the same model as the reference loader.
  // It exists because the reference one materializes every weight as host f32 --
  // ~132 GB for the real checkpoint, an OOM kill during load on a 122 GiB unified
  // box -- so the streamed path is what a real run actually uses, and "it loads
  // without OOM" is not the same claim as "it loads the right numbers".
  {
    vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
    vllm::MiniMaxH3DitParams streamed_params;
    const vllm::MiniMaxH3DitDeviceWeights streamed =
        vllm::StreamMiniMaxH3Nvfp4ToDeviceBf16(q, st, &streamed_params);
    CHECK(streamed_params.hidden_size == want.hidden_size);
    CHECK(streamed_params.num_layers == loaded.params.num_layers);
    REQUIRE(streamed.weights.blocks.size() == loaded.weights.blocks.size());

    // A packed weight comes back at the same logical shape, and its VALUES match
    // the reference to bf16 rounding -- the streamed path stores bf16 where the
    // reference keeps f32, so the tolerance is the cast, not the algorithm.
    const vt::Tensor& sq_ = streamed.weights.blocks[0].qkv_proj;
    REQUIRE(sq_.rank == 2);
    CHECK(sq_.shape[0] == 3 * inner);
    CHECK(sq_.shape[1] == want.hidden_size);
    CHECK(sq_.dtype == vt::DType::kBF16);
    const std::vector<float>& ref = loaded.storage.at("blocks.0.attn.qkv_proj.weight");
    const int64_t n = sq_.shape[0] * sq_.shape[1];
    REQUIRE(static_cast<int64_t>(ref.size()) == n);
    const uint16_t* got = static_cast<const uint16_t*>(sq_.data);
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const uint32_t widened = static_cast<uint32_t>(got[i]) << 16;
      float v;
      std::memcpy(&v, &widened, sizeof(v));
      worst = std::max(worst, std::abs(static_cast<double>(v) - ref[static_cast<size_t>(i)]));
    }
    INFO("streamed NVFP4 vs reference loader, max|diff| = " << worst);
    CHECK(worst <= 1e-2);  // bf16 has ~8 mantissa bits; values here are O(1)

    // rope.inv_freq must stay HOST-resident: the forward reads it before any
    // kernel runs, so a device pointer here segfaults rather than misbehaving.
    CHECK(!streamed.rope_inv_freq_host.empty());
    CHECK(streamed.weights.rope_inv_freq.data == streamed.rope_inv_freq_host.data());
  }

  // And it must RUN.
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      4, 2, 4, 4, 2, 2, /*include_keyframe_cond=*/false, {}, 0);
  const int64_t seq = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const std::vector<float> x(static_cast<size_t>(seq * video_width), 0.25f);
  const std::vector<float> audio_x(static_cast<size_t>(seq * want.audio_latents_dim), 0.1f);
  const std::vector<float> prompt(static_cast<size_t>(num_text * want.text_dim), 0.2f);
  const std::vector<float> unique_ts = {0.4f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};
  MiniMaxH3DitInputs in;
  in.seq_len = seq;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_ts.data();
  in.num_unique_timesteps = 1;
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForward(Cpu(), loaded.params, loaded.weights, in, vt::DType::kF32);
  CHECK(static_cast<int64_t>(got.video_logits.size()) == num_img * video_width);
  for (float v : got.video_logits) REQUIRE(std::isfinite(v));
  for (float v : got.audio_logits) REQUIRE(std::isfinite(v));

  // ── W-FP4a: the fp4 SPEED path vs the bf16 arm on the SAME loaded checkpoint ──
  // The bf16 arm dequantizes each NVFP4 projection to bf16 and runs vt::MatmulBT;
  // the fp4 arm keeps the packed FP4 resident and routes those GEMMs through
  // dense_nvfp4::MatmulNvfp4W4A16D (Marlin W4A16 on sm_121a). Both consume the SAME
  // fp4 bytes, so the delta is the GEMM PATH only — not quantization. On the CPU
  // backend the dispatcher has no Marlin op, so it falls back to a redundant-dequant
  // GEMM (numerically the bf16 arm's own math), which makes this a WIRING gate here:
  // it proves the loader sets the fp4 slots and the forward routes them. The real
  // Marlin-vs-bf16 numeric delta is measured on the GB10 (test_ops_nvfp4_matmul /
  // test_linear_method gate the Marlin kernel itself on CUDA at 2e-3/8e-3).
  {
    vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
    vllm::MiniMaxH3DitParams bf16_params, fp4_params;
    const vllm::MiniMaxH3DitDeviceWeights bf16_staged =
        vllm::StreamMiniMaxH3Nvfp4ToDeviceBf16(q, st, &bf16_params);
    const vllm::MiniMaxH3DitDeviceWeights fp4_staged =
        vllm::StreamMiniMaxH3Nvfp4ToDeviceFp4(q, st, &fp4_params);

    // The fp4 loader must have kept the quantized projections PACKED (an Nvfp4Weight
    // per projection) and left their bf16 tensor slot Empty(), while the bf16 loader
    // did the opposite. Getting this backwards is the silent failure this asserts.
    CHECK(!fp4_staged.weights.blocks[0].qkv_fp4.Empty());
    CHECK(fp4_staged.weights.blocks[0].qkv_proj.data == nullptr);
    CHECK(bf16_staged.weights.blocks[0].qkv_fp4.Empty());
    CHECK(bf16_staged.weights.blocks[0].qkv_proj.data != nullptr);
    // The 11 quantized projections of this reduced model (refiner: qkv/out/fc1/fc2;
    // block: +adaln; condition_proj; final_adaln) all keep FP4.
    CHECK(!fp4_staged.weights.condition_fp4.Empty());
    CHECK(!fp4_staged.weights.final_adaln_fp4.Empty());
    CHECK(!fp4_staged.weights.blocks[0].adaln_fp4.Empty());
    CHECK(fp4_staged.weights.refiner[0].adaln_fp4.Empty());  // refiner has no adaln

    const MiniMaxH3DitOutputs bf16_out =
        MiniMaxH3DitForwardDevice(q, bf16_params, bf16_staged.weights, in, vt::DType::kBF16);
    vllm::dense_nvfp4::ResetW4A16Stats();
    const MiniMaxH3DitOutputs fp4_out =
        MiniMaxH3DitForwardDevice(q, fp4_params, fp4_staged.weights, in, vt::DType::kBF16);
    const vllm::dense_nvfp4::Nvfp4W4A16Stats stats = vllm::dense_nvfp4::GetW4A16Stats();

    // POSITIVE signal: the W4A16 dispatcher actually RAN for every quantized
    // projection (11 GEMMs). On CPU these are `fallback_gemms`; on CUDA the same
    // count lands as `marlin_gemms`. A silently-unwired forward would show zero.
    const uint64_t w4a16_calls = stats.marlin_gemms + stats.fallback_gemms;
    INFO("W4A16 GEMMs executed by the fp4 forward = " << w4a16_calls);
    CHECK(w4a16_calls == 11);

    REQUIRE(fp4_out.video_logits.size() == bf16_out.video_logits.size());
    REQUIRE(fp4_out.audio_logits.size() == bf16_out.audio_logits.size());
    for (float v : fp4_out.video_logits) REQUIRE(std::isfinite(v));
    const double video_delta =
        MaxAbsDiff(fp4_out.video_logits, bf16_out.video_logits.data(), fp4_out.video_logits.size());
    const double audio_delta =
        MaxAbsDiff(fp4_out.audio_logits, bf16_out.audio_logits.data(), fp4_out.audio_logits.size());
    INFO("fp4-vs-bf16 (CPU fallback) video max|diff| = " << video_delta
                                                         << ", audio max|diff| = " << audio_delta);
    // CPU fallback == the bf16 arm's own dequant+matmul, so the two agree to matmul
    // reduction-order slack; a real fp4/bf16 divergence only appears on the GB10.
    CHECK(video_delta <= 2e-3);
    CHECK(audio_delta <= 2e-3);
  }
  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// The ORIGINAL bf16 release: 13 safetensors shards, 66.3 GB
// ---------------------------------------------------------------------------
// Every render so far used a QUANTIZED DiT, and Q3_K_M -> Q4_K_M alone turned a
// murky lattice-covered silhouette into a photoreal close-up — so "what does FULL
// PRECISION look like?" is the next question, and until this loader existed we
// could not ask it: every DiT loader took a SINGLE file.

namespace {

// The reduced geometry the sharded gates run at. Mirrors the NVFP4 case's dims so
// the two arms are comparable, with text_dim a multiple of 16.
MiniMaxH3DitParams ShardedGateParams() {
  MiniMaxH3DitParams want;
  want.num_layers = 2;
  want.token_refiner_num_layers = 1;
  want.hidden_size = 64;
  want.num_attention_heads = 4;
  want.attention_head_dim = 16;
  want.ffn_hidden_size = 128;
  want.latents_dim = 8;
  want.audio_latents_dim = 6;
  want.text_dim = 32;
  want.timestep_input_dim = 16;
  want.time_embed_hidden_size = 64;
  want.time_embed_dim = 32;
  want.adaln_out_features = 18 * want.hidden_size;
  want.final_adaln_out_features = 2 * want.hidden_size;
  want.rope_inv_freq_len = 2;
  return want;
}

// The dtype MIX the loader must survive. The real release stores the model bf16;
// these four names are pinned F32 so ONE synthetic checkpoint exercises all four
// (on-disk dtype x device dtype) combinations:
//   BF16 -> bf16 device slot   direct mmap upload, NO host buffer   (the bulk)
//   F32  -> f32 island         direct mmap upload, NO host buffer   (time_embedder)
//   BF16 -> f32 island         widened on the host                  (patch/out heads)
//   F32  -> bf16 device slot   rounded on the host                  (blocks.0.norm1)
std::set<std::string> ShardedGateF32Names() {
  return {"time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
          "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
          "rope.inv_freq", "blocks.0.norm1.weight"};
}

// Every weight view in binder order, so a comparison covers the WHOLE contract
// rather than a spot check.
std::vector<std::pair<std::string, const vt::Tensor*>> AllDitViews(
    const vllm::MiniMaxH3DitWeights& w) {
  std::vector<std::pair<std::string, const vt::Tensor*>> out;
  auto add = [&out](const std::string& name, const vt::Tensor& t) {
    out.emplace_back(name, &t);
  };
  add("video_patch_proj.weight", w.video_patch_proj_w);
  add("video_patch_proj.bias", w.video_patch_proj_b);
  add("audio_patch_proj.weight", w.audio_patch_proj_w);
  add("audio_patch_proj.bias", w.audio_patch_proj_b);
  add("condition_proj.weight", w.condition_proj_w);
  add("condition_proj.bias", w.condition_proj_b);
  add("time_embedder.proj_in.weight", w.time_proj_in_w);
  add("time_embedder.proj_in.bias", w.time_proj_in_b);
  add("time_embedder.proj_out.weight", w.time_proj_out_w);
  add("time_embedder.proj_out.bias", w.time_proj_out_b);
  auto add_block = [&](const std::string& prefix, const vllm::MiniMaxH3DitBlockWeights& b,
                       bool adaln) {
    add(prefix + ".norm1.weight", b.norm1);
    add(prefix + ".norm2.weight", b.norm2);
    add(prefix + ".attn.qkv_proj.weight", b.qkv_proj);
    add(prefix + ".attn.q_norm.weight", b.q_norm);
    add(prefix + ".attn.k_norm.weight", b.k_norm);
    add(prefix + ".attn.out_proj.weight", b.out_proj);
    add(prefix + ".mlp.fc1.weight", b.fc1);
    add(prefix + ".mlp.fc2.weight", b.fc2);
    if (adaln) {
      add(prefix + ".adaln_proj.linear.weight", b.adaln_w);
      add(prefix + ".adaln_proj.linear.bias", b.adaln_b);
    }
  };
  for (size_t i = 0; i < w.refiner.size(); ++i) {
    add_block("token_refiner.blocks." + std::to_string(i), w.refiner[i], false);
  }
  add("token_refiner.final_norm.weight", w.refiner_final_norm);
  for (size_t i = 0; i < w.blocks.size(); ++i) {
    add_block("blocks." + std::to_string(i), w.blocks[i], true);
  }
  add("final_layer.norm.weight", w.final_norm);
  add("final_layer.adaln_proj.linear.weight", w.final_adaln_w);
  add("final_layer.adaln_proj.linear.bias", w.final_adaln_b);
  add("final_layer.video_out.weight", w.video_out_w);
  add("final_layer.video_out.bias", w.video_out_b);
  add("final_layer.audio_out.weight", w.audio_out_w);
  add("final_layer.audio_out.bias", w.audio_out_b);
  return out;
}

}  // namespace

TEST_CASE("minimax_h3: the multi-shard index resolves every tensor to its own shard") {
  const MiniMaxH3DitParams want = ShardedGateParams();
  const std::vector<H3StEntry> entries =
      BuildMiniMaxH3DitEntries(want, /*quantize=*/false, /*plain_bf16=*/true,
                               ShardedGateF32Names());
  const std::string dir = "/tmp/minimax_h3_sharded_index";
  const size_t kShards = 4;
  const std::map<std::string, std::string> promised =
      WriteMiniMaxH3ShardedDit(entries, dir, kShards);

  const vllm::MiniMaxH3ShardedCheckpoint ckpt = vllm::MiniMaxH3ShardedCheckpoint::Open(dir);
  CHECK(ckpt.ShardCount() == kShards);
  CHECK(ckpt.Names().size() == entries.size());
  CHECK(ckpt.IndexPath() == dir + "/model.safetensors.index.json");

  // ★ EVERY tensor resolves to the shard the index named — and to the tensor that
  // was actually written into it, not a same-named one elsewhere. Resolving a
  // tensor to the WRONG shard is the failure that yields a loaded-but-wrong model.
  size_t checked = 0;
  for (const auto& entry : promised) {
    REQUIRE(ckpt.Has(entry.first));
    CHECK(ckpt.ShardOf(entry.first) == entry.second);
    const vllm::StTensor& t = ckpt.Get(entry.first);
    const auto it = std::find_if(entries.begin(), entries.end(),
                                 [&](const H3StEntry& e) { return e.name == entry.first; });
    REQUIRE(it != entries.end());
    CHECK(t.dtype == it->dtype);
    CHECK(t.shape == it->shape);
    REQUIRE(t.nbytes == it->bytes.size());
    CHECK(std::memcmp(t.data, it->bytes.data(), t.nbytes) == 0);
    ++checked;
  }
  CHECK(checked == entries.size());
  CHECK(!ckpt.Has("blocks.99.mlp.fc1.weight"));

  // The shards really are several files, each holding part of the model.
  CHECK(ckpt.ShardFiles().size() == kShards);
  std::set<std::string> distinct;
  for (const auto& entry : promised) distinct.insert(entry.second);
  CHECK(distinct.size() == kShards);

  // GEOMETRY from the shards must equal the SINGLE-FILE path over the same tensors:
  // one file, same entries, same derived params — so sharding is a container
  // question and never a model question.
  const std::vector<vllm::MiniMaxH3TensorSpec> manifest =
      vllm::EnumerateMiniMaxH3ShardedTensors(ckpt);
  const MiniMaxH3DitParams sharded = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  const std::string single = "/tmp/minimax_h3_sharded_single.safetensors";
  WriteSafetensorsFromEntries(entries, single);
  const vllm::SafetensorsFile sf = vllm::SafetensorsFile::Open(single);
  const vllm::MiniMaxH3GgufDit single_loaded = vllm::LoadMiniMaxH3DitFromNvfp4(sf);
  const MiniMaxH3DitParams& one = single_loaded.params;
  CHECK(sharded.num_layers == one.num_layers);
  CHECK(sharded.token_refiner_num_layers == one.token_refiner_num_layers);
  CHECK(sharded.hidden_size == one.hidden_size);
  CHECK(sharded.num_attention_heads == one.num_attention_heads);
  CHECK(sharded.attention_head_dim == one.attention_head_dim);
  CHECK(sharded.ffn_hidden_size == one.ffn_hidden_size);
  CHECK(sharded.latents_dim == one.latents_dim);
  CHECK(sharded.audio_latents_dim == one.audio_latents_dim);
  CHECK(sharded.patch_size_t == one.patch_size_t);
  CHECK(sharded.patch_size_h == one.patch_size_h);
  CHECK(sharded.patch_size_w == one.patch_size_w);
  CHECK(sharded.text_dim == one.text_dim);
  CHECK(sharded.timestep_input_dim == one.timestep_input_dim);
  CHECK(sharded.time_embed_hidden_size == one.time_embed_hidden_size);
  CHECK(sharded.time_embed_dim == one.time_embed_dim);
  CHECK(sharded.adaln_out_features == one.adaln_out_features);
  CHECK(sharded.final_adaln_out_features == one.final_adaln_out_features);
  CHECK(sharded.rope_inv_freq_len == one.rope_inv_freq_len);
  CHECK(sharded.video_row_width() == one.video_row_width());
  CHECK(sharded.rope_rot_dim() == one.rope_rot_dim());
  // ...and it is the geometry that was asked for.
  CHECK(sharded.num_layers == want.num_layers);
  CHECK(sharded.hidden_size == want.hidden_size);
  CHECK(sharded.text_dim == want.text_dim);

  // The manifest also carries the fp32-ISLAND flag, and it must agree with the
  // upstream-derived enumeration name for name (see the island test below).
  for (const vllm::MiniMaxH3TensorSpec& spec : manifest) {
    CHECK(spec.fp32 == vllm::MiniMaxH3IsFp32IslandTensor(spec.name));
  }

  std::remove(single.c_str());
  RemoveShardedDit(dir, kShards);

  // ★ A tensor the index NAMES but its shard does not contain must throw BY NAME.
  // Silently skipping it would leave that weight reading as zeros — a plausible
  // but wrong render rather than an error.
  const std::string broken_dir = "/tmp/minimax_h3_sharded_broken";
  const std::string missing = "blocks.1.mlp.fc2.weight";
  WriteMiniMaxH3ShardedDit(entries, broken_dir, kShards, {missing});
  bool threw = false;
  try {
    const vllm::MiniMaxH3ShardedCheckpoint bad =
        vllm::MiniMaxH3ShardedCheckpoint::Open(broken_dir);
    (void)bad;
  } catch (const std::exception& e) {
    threw = true;
    const std::string what = e.what();
    INFO("missing-tensor error: " << what);
    CHECK(what.find(missing) != std::string::npos);
  }
  CHECK(threw);
  RemoveShardedDit(broken_dir, kShards);

  // A directory with no index at all is refused, not half-loaded.
  CHECK(!vllm::MiniMaxH3ShardedCheckpoint::IsShardedDir("/tmp"));
  CHECK_THROWS(vllm::MiniMaxH3ShardedCheckpoint::Open("/tmp"));
}

TEST_CASE("minimax_h3: the sharded bf16 DiT STREAMS to the device, matching the reference") {
  const MiniMaxH3DitParams want = ShardedGateParams();
  const std::vector<H3StEntry> entries =
      BuildMiniMaxH3DitEntries(want, /*quantize=*/false, /*plain_bf16=*/true,
                               ShardedGateF32Names());
  const std::string dir = "/tmp/minimax_h3_sharded_stream";
  const size_t kShards = 3;
  WriteMiniMaxH3ShardedDit(entries, dir, kShards);
  const vllm::MiniMaxH3ShardedCheckpoint ckpt = vllm::MiniMaxH3ShardedCheckpoint::Open(dir);

  // The NON-STREAMED reference: materialize to host f32, then stage with the
  // existing (load-everything-then-upload) stager. This is the arm the streamer
  // must reproduce, and it is the arm that CANNOT be used on the real 66.3 GB
  // release — it holds the model twice against a 122 GiB UNIFIED pool.
  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  const vllm::MiniMaxH3GgufDit reference = vllm::LoadMiniMaxH3DitFromShards(ckpt);
  const vllm::MiniMaxH3DitDeviceWeights ref_staged =
      vllm::StageMiniMaxH3DitWeights(q, reference.params, reference.weights, vt::DType::kBF16);

  vllm::ResetMiniMaxH3ShardStreamStats();
  MiniMaxH3DitParams streamed_params;
  const vllm::MiniMaxH3DitDeviceWeights streamed =
      vllm::StreamMiniMaxH3ShardedToDeviceBf16(q, ckpt, &streamed_params);
  const vllm::MiniMaxH3ShardStreamStats stats = vllm::GetMiniMaxH3ShardStreamStats();

  CHECK(streamed_params.num_layers == reference.params.num_layers);
  CHECK(streamed_params.hidden_size == reference.params.hidden_size);
  CHECK(streamed_params.text_dim == reference.params.text_dim);

  // ★ THE LOADER RAN, and it produced DEVICE tensors. A green suite over a path
  // that silently fell back to another loader is a failure mode this codebase has
  // already shipped once, so the positive signal is asserted, not assumed.
  INFO("shard-stream stats: shards=" << stats.shards_opened << " tensors="
       << stats.tensors_streamed << " direct=" << stats.direct_uploads << " converted="
       << stats.converted_uploads << " bytes=" << stats.bytes_uploaded << " host_peak="
       << stats.host_peak_bytes);
  CHECK(stats.shards_opened == kShards);
  CHECK(stats.tensors_streamed == entries.size() - 1);  // rope.inv_freq stays host
  CHECK(stats.host_resident == 1);
  CHECK(stats.direct_uploads + stats.converted_uploads == stats.tensors_streamed);
  CHECK(stats.direct_uploads > 0);     // the zero-host-copy path really is taken
  CHECK(stats.converted_uploads > 0);  // and so is the converting one
  CHECK(stats.bytes_uploaded > 0);

  // ★ PEAK HOST MEMORY CANNOT BALLOON. Only ONE tensor's conversion buffer is ever
  // alive, so the peak is bounded by the largest single tensor — NOT by the model.
  // On the real release that bound is ~1 GB against 66.3 GB of weights, and the
  // BF16->bf16 bulk costs nothing at all.
  uint64_t largest = 0;
  for (const H3StEntry& e : entries) {
    largest = std::max<uint64_t>(largest, static_cast<uint64_t>(e.bytes.size()) * 2);
  }
  CHECK(stats.host_peak_bytes <= largest);
  CHECK(stats.host_peak_bytes < stats.bytes_uploaded / 4);

  // Every streamed view points at an allocation this loader OWNS (a device buffer),
  // never into the mmap or into a host reference.
  std::set<const void*> owned;
  for (const std::shared_ptr<void>& s : streamed.storage) owned.insert(s.get());
  CHECK(owned.size() == stats.tensors_streamed);

  // ★ STREAMED == NON-STREAMED, tensor for tensor. Both round f32->bf16 with the
  // same round-to-nearest-even rule and keep the same fp32 islands, so this is
  // BIT-EXACT, not a tolerance.
  const auto ref_views = AllDitViews(ref_staged.weights);
  const auto got_views = AllDitViews(streamed.weights);
  REQUIRE(ref_views.size() == got_views.size());
  size_t compared = 0, island_count = 0;
  for (size_t i = 0; i < ref_views.size(); ++i) {
    const std::string& name = ref_views[i].first;
    const vt::Tensor& a = *ref_views[i].second;
    const vt::Tensor& b = *got_views[i].second;
    INFO("weight " << name);
    REQUIRE(got_views[i].first == name);
    REQUIRE(b.data != nullptr);
    CHECK(owned.count(b.data) == 1);  // it came from THIS loader's device staging
    REQUIRE(a.rank == b.rank);
    for (int r = 0; r < a.rank; ++r) CHECK(a.shape[r] == b.shape[r]);
    // The dtype policy: fp32 ISLANDS stay f32, everything else is bf16.
    const bool island = vllm::MiniMaxH3IsFp32IslandTensor(name);
    CHECK(b.dtype == (island ? vt::DType::kF32 : vt::DType::kBF16));
    CHECK(a.dtype == b.dtype);
    island_count += island ? 1 : 0;
    const size_t bytes = static_cast<size_t>(a.Numel()) * vt::SizeOf(a.dtype);
    CHECK(std::memcmp(a.data, b.data, bytes) == 0);
    ++compared;
  }
  CHECK(compared == ref_views.size());
  CHECK(island_count == 12);  // both patch projections, the time embedder, both heads

  // ★ rope.inv_freq stays on the HOST. The forward builds the cos/sin cache from it
  // BEFORE any kernel runs, so a device pointer here segfaults rather than
  // misbehaving.
  CHECK(!streamed.rope_inv_freq_host.empty());
  CHECK(streamed.weights.rope_inv_freq.data == streamed.rope_inv_freq_host.data());
  CHECK(streamed.weights.rope_inv_freq.dtype == vt::DType::kF32);
  CHECK(owned.count(streamed.weights.rope_inv_freq.data) == 0);
  REQUIRE(static_cast<int64_t>(streamed.rope_inv_freq_host.size()) == want.rope_inv_freq_len);
  for (int64_t i = 0; i < want.rope_inv_freq_len; ++i) {
    CHECK(streamed.rope_inv_freq_host[static_cast<size_t>(i)] ==
          ref_staged.weights.rope_inv_freq.Ptr<float>()[i]);
  }

  // And it must RUN: the streamed weights drive a real device forward, and the
  // non-streamed reference produces the SAME velocity prediction.
  const MiniMaxH3PackedSequence packed =
      BuildMiniMaxH3PackedSequence(4, 2, 4, 4, 2, 2, /*include_keyframe_cond=*/false, {}, 0);
  const int64_t seq = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const int64_t video_width = want.video_row_width();
  const std::vector<float> x(static_cast<size_t>(seq * video_width), 0.25f);
  const std::vector<float> audio_x(static_cast<size_t>(seq * want.audio_latents_dim), 0.1f);
  const std::vector<float> prompt(static_cast<size_t>(num_text * want.text_dim), 0.2f);
  const std::vector<float> unique_ts = {0.4f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};
  MiniMaxH3DitInputs in;
  in.seq_len = seq;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_ts.data();
  in.num_unique_timesteps = 1;
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForwardDevice(q, streamed_params, streamed.weights, in, vt::DType::kBF16);
  const MiniMaxH3DitOutputs ref =
      MiniMaxH3DitForwardDevice(q, reference.params, ref_staged.weights, in, vt::DType::kBF16);
  REQUIRE(got.video_logits.size() == static_cast<size_t>(num_img * video_width));
  REQUIRE(got.video_logits.size() == ref.video_logits.size());
  REQUIRE(got.audio_logits.size() == ref.audio_logits.size());
  for (float v : got.video_logits) REQUIRE(std::isfinite(v));
  for (float v : got.audio_logits) REQUIRE(std::isfinite(v));
  const double video_delta =
      MaxAbsDiff(got.video_logits, ref.video_logits.data(), got.video_logits.size());
  const double audio_delta =
      MaxAbsDiff(got.audio_logits, ref.audio_logits.data(), got.audio_logits.size());
  INFO("streamed-vs-reference forward: video max|diff| = " << video_delta
                                                           << ", audio max|diff| = " << audio_delta);
  // Bit-identical weights through the same graph: the outputs match EXACTLY.
  CHECK(video_delta == 0.0);
  CHECK(audio_delta == 0.0);

  RemoveShardedDit(dir, kShards);
}

TEST_CASE("minimax_h3: a 13-shard 66 GB bf16 release derives the SHIPPED geometry") {
  // The real release's SHAPE at its real size, with the payload as a sparse hole:
  // the headers are byte-for-byte what the 66.3 GB checkpoint declares, so the
  // manifest path — the one `--dump-params <dir>` uses, and the one the streamer
  // derives its geometry from before allocating anything — is gated on the REAL
  // names and shapes without downloading or storing a weight byte.
  MiniMaxH3DitParams shipped;  // the defaults ARE the shipped H3 geometry
  const std::vector<vllm::MiniMaxH3TensorSpec> specs =
      vllm::EnumerateMiniMaxH3DitTensors(shipped);
  const std::string dir = "/tmp/minimax_h3_sharded_release";
  const size_t kShards = 13;  // what the release actually ships
  const uint64_t declared = WriteMiniMaxH3SparseShardedRelease(specs, dir, kShards);
  INFO("declared payload = " << (declared / (1024.0 * 1024.0 * 1024.0)) << " GiB");
  CHECK(declared > 55ull * 1024 * 1024 * 1024);  // ~66.3 GB of bf16 weights
  CHECK(declared < 70ull * 1024 * 1024 * 1024);

  const vllm::MiniMaxH3ShardedCheckpoint ckpt = vllm::MiniMaxH3ShardedCheckpoint::Open(dir);
  CHECK(ckpt.ShardCount() == kShards);
  CHECK(ckpt.Names().size() == specs.size());

  const MiniMaxH3DitParams p =
      vllm::ParseMiniMaxH3DitParamsFromGgufManifest(vllm::EnumerateMiniMaxH3ShardedTensors(ckpt));
  // ★ The same 20 fields --dump-params prints, and the same values the WORKING
  // GGUF arm derives from shapes alone. A mismatch means the name mapping is wrong.
  CHECK(p.num_layers == 50);
  CHECK(p.token_refiner_num_layers == 2);
  CHECK(p.hidden_size == 5376);
  CHECK(p.num_attention_heads == 56);
  CHECK(p.attention_head_dim == 128);
  CHECK(p.ffn_hidden_size == 14336);
  CHECK(p.latents_dim == 24);
  CHECK(p.audio_latents_dim == 32);
  CHECK(p.patch_size_t == 1);
  CHECK(p.patch_size_h == 2);
  CHECK(p.patch_size_w == 2);
  CHECK(p.text_dim == 5120);
  CHECK(p.timestep_input_dim == 256);
  CHECK(p.time_embed_hidden_size == 5376);
  CHECK(p.time_embed_dim == 2688);
  CHECK(p.adaln_out_features == 18 * 5376);
  CHECK(p.final_adaln_out_features == 2 * 5376);
  CHECK(p.rope_inv_freq_len == 16);
  CHECK(p.video_row_width() == 96);
  CHECK(p.rope_rot_dim() == 96);

  // Every tensor of the release resolves, and the fp32-ISLAND split matches the
  // upstream-derived enumeration name for name — the split vt::MatmulBT enforces
  // at the first island GEMM.
  size_t islands = 0;
  for (const vllm::MiniMaxH3TensorSpec& spec : specs) {
    REQUIRE(ckpt.Has(spec.name));
    CHECK(ckpt.Get(spec.name).shape == spec.shape);
    CHECK(spec.fp32 == vllm::MiniMaxH3IsFp32IslandTensor(spec.name));
    islands += spec.fp32 ? 1 : 0;
  }
  CHECK(islands == 13);  // 12 island weights/biases + rope.inv_freq

  RemoveShardedDit(dir, kShards);
}

TEST_CASE("minimax_h3: the NVFP4 fp4 forward runs Marlin W4A16 on CUDA (speed)") {
  // The GB10 leg the CPU wiring gate cannot reach (spec 8.4a): on the CUDA backend
  // the W4A16 dispatcher hits dense_nvfp4::MatmulNvfp4MarlinD, so the fp4 path RAN
  // is provable by `marlin_gemms == 11` (not the CPU `fallback_gemms`). Built at the
  // REAL H3 geometry (one AdaLN block + one refiner) so the per-forward and
  // per-GEMM times, and the fp4-vs-bf16 numeric delta, are the production shapes,
  // not the reduced-dim wiring model. Env-tunable so the same binary sweeps
  // sequence length and reps on dgx: H3_FP4_{LT,LH,LW,AT,AC,TEXT,REPS}.
  namespace dnv = vllm::dense_nvfp4;
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();

  auto env_i = [](const char* k, int64_t dflt) -> int64_t {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoll(v) : dflt;
  };
  const int64_t lt = env_i("H3_FP4_LT", 16), lh = env_i("H3_FP4_LH", 32),
                lw = env_i("H3_FP4_LW", 32), at = env_i("H3_FP4_AT", 8),
                ac = env_i("H3_FP4_AC", 2), text_len = env_i("H3_FP4_TEXT", 64);
  const int reps = static_cast<int>(env_i("H3_FP4_REPS", 8));

  // Real geometry, single-layer: this is what makes the quantized-GEMM count 11.
  MiniMaxH3DitParams want;  // defaults ARE the shipped H3 geometry
  want.num_layers = 1;
  want.token_refiner_num_layers = 1;
  const int64_t video_width = want.video_row_width();
  const std::string path = "/tmp/minimax_h3_nvfp4_cuda_speed.safetensors";
  WriteMiniMaxH3Nvfp4File(want, path);
  const vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(path);

  // Stream BOTH arms to the DEVICE off the SAME file. bf16 dequantizes to bf16 and
  // runs vt::MatmulBT; fp4 keeps the packed FP4 resident and routes each quantized
  // projection through the Marlin W4A16 grouped GEMM.
  vllm::MiniMaxH3DitParams bf16_params, fp4_params;
  const vllm::MiniMaxH3DitDeviceWeights bf16_staged =
      vllm::StreamMiniMaxH3Nvfp4ToDeviceBf16(q, st, &bf16_params);
  const vllm::MiniMaxH3DitDeviceWeights fp4_staged =
      vllm::StreamMiniMaxH3Nvfp4ToDeviceFp4(q, st, &fp4_params);
  CHECK(!fp4_staged.weights.blocks[0].qkv_fp4.Empty());
  CHECK(fp4_staged.weights.blocks[0].qkv_proj.data == nullptr);
  CHECK(bf16_staged.weights.blocks[0].qkv_fp4.Empty());
  CHECK(bf16_staged.weights.blocks[0].qkv_proj.data != nullptr);

  // Production-shaped packed sequence.
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      text_len, lt, lh, lw, at, ac, /*include_keyframe_cond=*/false, {}, 0);
  const int64_t seq = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const std::vector<float> x(static_cast<size_t>(seq * video_width), 0.25f);
  const std::vector<float> audio_x(static_cast<size_t>(seq * want.audio_latents_dim), 0.1f);
  const std::vector<float> prompt(static_cast<size_t>(num_text * want.text_dim), 0.2f);
  const std::vector<float> unique_ts = {0.4f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};
  MiniMaxH3DitInputs in;
  in.seq_len = seq;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_ts.data();
  in.num_unique_timesteps = 1;
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  auto now = [] { return std::chrono::steady_clock::now(); };
  auto us = [](std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
  };
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
  };

  // ── Per-FORWARD timing: the whole DiT step, both arms (the forward returns host
  // vectors, so each call syncs the device — no extra Synchronize needed). ──
  MiniMaxH3DitOutputs bf16_out =
      MiniMaxH3DitForwardDevice(q, bf16_params, bf16_staged.weights, in, vt::DType::kBF16);
  std::vector<double> tb;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = now();
    bf16_out = MiniMaxH3DitForwardDevice(q, bf16_params, bf16_staged.weights, in, vt::DType::kBF16);
    tb.push_back(us(t0, now()));
  }

  // Cold fp4 forward first (builds the Marlin resident/repack), then the counter
  // is asserted on ONE clean forward, then timed.
  MiniMaxH3DitOutputs fp4_out =
      MiniMaxH3DitForwardDevice(q, fp4_params, fp4_staged.weights, in, vt::DType::kBF16);
  dnv::ResetW4A16Stats();
  fp4_out = MiniMaxH3DitForwardDevice(q, fp4_params, fp4_staged.weights, in, vt::DType::kBF16);
  const dnv::Nvfp4W4A16Stats stats = dnv::GetW4A16Stats();
  MESSAGE("H3FP4FWD counters marlin_gemms=" << stats.marlin_gemms
          << " dense_gemms=" << stats.dense_gemms
          << " fallback_gemms=" << stats.fallback_gemms
          << " fused_gate_up=" << stats.fused_gate_up);
  // The POSITIVE proof the Marlin W4A16 path RAN on the GPU for all 11 quantized
  // projections, and NOT the CPU/redundant-dequant fallback. On CPU these would be
  // fallback_gemms. On CUDA the production default (VT_MARLIN_DENSE on, sm_121a)
  // routes each projection through vLLM's OWN dense Marlin GEMM -> `dense_gemms`;
  // VT_MARLIN_DENSE=0 opts into the single-expert MoE-grouped Marlin -> `marlin_gemms`.
  // Either Marlin sub-route is the "this-path-ran" signal; fallback_gemms MUST be 0.
  const uint64_t marlin_family = stats.marlin_gemms + stats.dense_gemms;
  CHECK(marlin_family == 11);
  CHECK(stats.fallback_gemms == 0);
  std::vector<double> tf;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = now();
    fp4_out = MiniMaxH3DitForwardDevice(q, fp4_params, fp4_staged.weights, in, vt::DType::kBF16);
    tf.push_back(us(t0, now()));
  }

  REQUIRE(fp4_out.video_logits.size() == bf16_out.video_logits.size());
  for (float v : fp4_out.video_logits) REQUIRE(std::isfinite(v));
  for (float v : fp4_out.audio_logits) REQUIRE(std::isfinite(v));
  const double video_delta =
      MaxAbsDiff(fp4_out.video_logits, bf16_out.video_logits.data(), fp4_out.video_logits.size());
  const double audio_delta =
      MaxAbsDiff(fp4_out.audio_logits, bf16_out.audio_logits.data(), fp4_out.audio_logits.size());
  double vmax = 0.0;
  for (float v : bf16_out.video_logits) vmax = std::max(vmax, std::abs(static_cast<double>(v)));
  const double mb = median(tb), mf = median(tf);
  MESSAGE("H3FP4FWD seq=" << seq << " reps=" << reps
          << " per_forward_bf16_ms=" << (mb / 1000.0)
          << " per_forward_fp4_ms=" << (mf / 1000.0)
          << " ratio(bf16/fp4)=" << (mf > 0 ? mb / mf : 0.0)
          << " avg_per_marlin_gemm_us_fp4=" << (mf / 11.0));
  MESSAGE("H3FP4FWD fp4-vs-bf16(CUDA Marlin) video max|diff|=" << video_delta
          << " audio max|diff|=" << audio_delta
          << " (bf16 video max|val|=" << vmax << ", rel="
          << (vmax > 0 ? video_delta / vmax : 0.0) << ")");
  // Both arms consume the SAME fp4 bytes; the delta is the GEMM path (Marlin's
  // bf16 tensor-core accumulate vs the bf16 arm's dequant+MatmulBT), so it is
  // matmul-reduction slack amplified through one block, not a quantization error.
  CHECK(std::isfinite(video_delta));
  CHECK(vmax > 0.0);
  CHECK(video_delta <= 0.10 * vmax + 1e-3);  // <=10% relative: sanity, not a tie

  // ── Per-GEMM microbench: the dominant per-token projections at M = seq. Marlin
  // W4A16 (fp4-resident) vs the bf16 arm's own dequant+GEMM, SAME numbers. ──
  dnv::Dev d{*cuda, q};
  auto bench = [&](const char* nm, const vllm::Nvfp4Weight& w) {
    const int64_t N = w.n, K = w.k;
    std::vector<uint16_t> xh(static_cast<size_t>(seq) * K, 0x3DCCu);  // bf16 ~0.1
    dnv::DBuf xb(d, vt::DType::kBF16, {seq, K}, xh.data());
    for (int i = 0; i < 3; ++i) { auto o = dnv::MatmulNvfp4W4A16D(d, xb.t(), w, vt::DType::kBF16); (void)o; }
    cuda->Synchronize(q);
    std::vector<double> f;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = now();
      auto o = dnv::MatmulNvfp4W4A16D(d, xb.t(), w, vt::DType::kBF16);
      cuda->Synchronize(q);
      f.push_back(us(t0, now()));
      (void)o;
    }
    const std::vector<uint16_t> wb = dnv::DequantNvfp4ToBLayout(w);  // [K, N] bf16
    dnv::DBuf wbd(d, vt::DType::kBF16, {K, N}, wb.data());
    for (int i = 0; i < 3; ++i) { dnv::DBuf o(d, vt::DType::kBF16, {seq, N}); vt::Matmul(q, o.t(), xb.t(), wbd.t()); }
    cuda->Synchronize(q);
    std::vector<double> b;
    for (int r = 0; r < reps; ++r) {
      dnv::DBuf o(d, vt::DType::kBF16, {seq, N});
      const auto t0 = now();
      vt::Matmul(q, o.t(), xb.t(), wbd.t());
      cuda->Synchronize(q);
      b.push_back(us(t0, now()));
    }
    const double gf = median(f), gb = median(b);
    MESSAGE("H3FP4GEMM " << nm << " M=" << seq << " N=" << N << " K=" << K
            << " fp4_marlin_us=" << gf << " bf16_us=" << gb
            << " ratio(bf16/fp4)=" << (gf > 0 ? gb / gf : 0.0));
  };
  bench("qkv", fp4_staged.weights.blocks[0].qkv_fp4);
  bench("out", fp4_staged.weights.blocks[0].out_fp4);
  bench("fc1", fp4_staged.weights.blocks[0].fc1_fp4);
  bench("fc2", fp4_staged.weights.blocks[0].fc2_fp4);

  std::remove(path.c_str());
}

TEST_CASE("minimax_h3: the encoder VISION block matches upstream") {
  // The repeated unit of the H3-Encoder's Qwen3-VL vision tower. It differs from
  // the text tower in ways that all matter numerically: LayerNorm (with bias) not
  // RMSNorm, a [q_all|k_all|v_all] qkv layout, fp32 rotary, NON-CAUSAL attention
  // segmented by cu_seqlens, and the TANH-approximate GELU.
  vllm::MiniMaxH3VisionBlockConfig config;
  config.hidden_size = vllm_test::kH3EncVisDim;
  config.num_heads = vllm_test::kH3EncVisHeads;
  config.intermediate_size = vllm_test::kH3EncVisIntermediate;
  config.eps = 1e-6;

  const int64_t dim = config.hidden_size;
  const int64_t seq = vllm_test::kH3EncVisSeq;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& suffix, int64_t count, double scale, double offset) {
    weights.tensors["vb." + suffix] = MakeParam("encoder.vision." + suffix, count, scale, offset);
  };
  put("norm1.weight", dim, 0.1, 1.0);
  put("norm1.bias", dim, 0.05, 0.0);
  put("norm2.weight", dim, 0.1, 1.0);
  put("norm2.bias", dim, 0.05, 0.0);
  put("attn.qkv.weight", 3 * dim * dim, 0.1, 0.0);
  put("attn.qkv.bias", 3 * dim, 0.05, 0.0);
  put("attn.proj.weight", dim * dim, 0.1, 0.0);
  put("attn.proj.bias", dim, 0.05, 0.0);
  put("mlp.linear_fc1.weight", config.intermediate_size * dim, 0.1, 0.0);
  put("mlp.linear_fc1.bias", config.intermediate_size, 0.05, 0.0);
  put("mlp.linear_fc2.weight", dim * config.intermediate_size, 0.1, 0.0);
  put("mlp.linear_fc2.bias", dim, 0.05, 0.0);

  const std::vector<float> hidden = MakeParam("encoder.vision.input", seq * dim, 1.0);
  const std::vector<float> cos(vllm_test::kH3EncVisCos,
                               vllm_test::kH3EncVisCos + std::size(vllm_test::kH3EncVisCos));
  const std::vector<float> sin(vllm_test::kH3EncVisSin,
                               vllm_test::kH3EncVisSin + std::size(vllm_test::kH3EncVisSin));
  std::vector<int32_t> cu;
  for (size_t i = 0; i < std::size(vllm_test::kH3EncVisCuSeqlens); ++i) {
    cu.push_back(static_cast<int32_t>(vllm_test::kH3EncVisCuSeqlens[i]));
  }
  // Two packed segments, so the varlen boundary is genuinely exercised.
  REQUIRE(cu.size() == 3);

  const std::vector<float> got = vllm::MiniMaxH3VisionBlockForward(
      config, weights, "vb", hidden, seq, cos.data(), sin.data(), cu.data(),
      static_cast<int64_t>(cu.size()) - 1);

  REQUIRE(got.size() == std::size(vllm_test::kH3EncVisBlockGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3EncVisBlockGolden, got.size());
  INFO("vision block max|diff| = " << err);
  CHECK(err <= 1e-5);

  // Attention must NOT cross the packed-image boundary: perturbing a token in the
  // SECOND segment must leave the FIRST segment's outputs untouched.
  std::vector<float> perturbed = hidden;
  perturbed[static_cast<size_t>(cu[1] * dim)] += 1.0f;
  const std::vector<float> other = vllm::MiniMaxH3VisionBlockForward(
      config, weights, "vb", perturbed, seq, cos.data(), sin.data(), cu.data(),
      static_cast<int64_t>(cu.size()) - 1);
  for (int64_t i = 0; i < cu[1] * dim; ++i) {
    CHECK(other[static_cast<size_t>(i)] == got[static_cast<size_t>(i)]);
  }
  // ...and must actually change the segment it belongs to.
  bool changed = false;
  for (int64_t i = cu[1] * dim; i < seq * dim; ++i) {
    if (other[static_cast<size_t>(i)] != got[static_cast<size_t>(i)]) changed = true;
  }
  CHECK(changed);
}

TEST_CASE("minimax_h3: the FULL encoder vision tower matches upstream") {
  // Completes the encoder: patch embed -> bilinear-interpolated position embedding
  // -> 2D rotary -> block stack -> DeepStack mergers + final patch merger.
  vllm::MiniMaxH3VisionTowerConfig config;
  config.block.hidden_size = vllm_test::kH3EncVisDim;
  config.block.num_heads = vllm_test::kH3EncVisHeads;
  config.block.intermediate_size = vllm_test::kH3EncVisIntermediate;
  config.block.eps = 1e-6;
  config.depth = vllm_test::kH3EncTowerDepth;
  config.patch_size = vllm_test::kH3EncTowerPatch;
  config.temporal_patch_size = vllm_test::kH3EncTowerTemporalPatch;
  config.in_channels = vllm_test::kH3EncTowerInCh;
  config.spatial_merge_size = vllm_test::kH3EncTowerMerge;
  config.out_hidden_size = vllm_test::kH3EncTowerOutHidden;
  config.num_position_embeddings = vllm_test::kH3EncTowerNumPos;
  config.rope_theta = 10000.0;
  for (size_t i = 0; i < std::size(vllm_test::kH3EncTowerDeepstackIdx); ++i) {
    config.deepstack_visual_indexes.push_back(vllm_test::kH3EncTowerDeepstackIdx[i]);
  }

  const int64_t dim = config.block.hidden_size;
  const int64_t merged_width = dim * config.spatial_merge_size * config.spatial_merge_size;
  std::vector<int64_t> grid;
  for (size_t i = 0; i < std::size(vllm_test::kH3EncTowerGridThw); ++i) {
    grid.push_back(vllm_test::kH3EncTowerGridThw[i]);
  }

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& n, int64_t count, double scale, double offset) {
    weights.tensors[n] = MakeParam("encoder.tower." + n, count, scale, offset);
  };
  const int64_t patch_elems = config.in_channels * config.temporal_patch_size *
                              config.patch_size * config.patch_size;
  put("patch_embed.proj.weight", dim * patch_elems, 0.1, 0.0);
  put("patch_embed.proj.bias", dim, 0.05, 0.0);
  put("pos_embed.weight", config.num_position_embeddings * dim, 0.1, 0.0);
  for (int64_t l = 0; l < config.depth; ++l) {
    const std::string b = "blocks." + std::to_string(l) + ".";
    put(b + "norm1.weight", dim, 0.1, 1.0);
    put(b + "norm1.bias", dim, 0.05, 0.0);
    put(b + "norm2.weight", dim, 0.1, 1.0);
    put(b + "norm2.bias", dim, 0.05, 0.0);
    put(b + "attn.qkv.weight", 3 * dim * dim, 0.1, 0.0);
    put(b + "attn.qkv.bias", 3 * dim, 0.05, 0.0);
    put(b + "attn.proj.weight", dim * dim, 0.1, 0.0);
    put(b + "attn.proj.bias", dim, 0.05, 0.0);
    put(b + "mlp.linear_fc1.weight", config.block.intermediate_size * dim, 0.1, 0.0);
    put(b + "mlp.linear_fc1.bias", config.block.intermediate_size, 0.05, 0.0);
    put(b + "mlp.linear_fc2.weight", dim * config.block.intermediate_size, 0.1, 0.0);
    put(b + "mlp.linear_fc2.bias", dim, 0.05, 0.0);
  }
  // The final merger norms the PRE-shuffle width; the DeepStack mergers norm the
  // POST-shuffle width. Getting these the wrong way round changes the result.
  put("merger.norm.weight", dim, 0.1, 1.0);
  put("merger.norm.bias", dim, 0.05, 0.0);
  put("merger.linear_fc1.weight", merged_width * merged_width, 0.1, 0.0);
  put("merger.linear_fc1.bias", merged_width, 0.05, 0.0);
  put("merger.linear_fc2.weight", config.out_hidden_size * merged_width, 0.1, 0.0);
  put("merger.linear_fc2.bias", config.out_hidden_size, 0.05, 0.0);
  for (size_t d = 0; d < config.deepstack_visual_indexes.size(); ++d) {
    const std::string m = "deepstack_merger_list." + std::to_string(d) + ".";
    put(m + "norm.weight", merged_width, 0.1, 1.0);
    put(m + "norm.bias", merged_width, 0.05, 0.0);
    put(m + "linear_fc1.weight", merged_width * merged_width, 0.1, 0.0);
    put(m + "linear_fc1.bias", merged_width, 0.05, 0.0);
    put(m + "linear_fc2.weight", config.out_hidden_size * merged_width, 0.1, 0.0);
    put(m + "linear_fc2.bias", config.out_hidden_size, 0.05, 0.0);
  }

  const std::vector<float> pixels =
      MakeParam("encoder.tower.pixels", vllm_test::kH3EncTowerPatches * patch_elems, 1.0);
  const vllm::MiniMaxH3VisionTowerResult got =
      vllm::MiniMaxH3VisionTowerForward(config, weights, pixels, grid);

  CHECK(static_cast<int64_t>(got.merged.size()) ==
        vllm_test::kH3EncTowerMergedRows * config.out_hidden_size);
  REQUIRE(got.merged.size() == std::size(vllm_test::kH3EncTowerMergedGolden));
  const double err = MaxAbsDiff(got.merged, vllm_test::kH3EncTowerMergedGolden, got.merged.size());
  INFO("vision tower merged max|diff| = " << err);
  CHECK(err <= 1e-4);

  CHECK(static_cast<int64_t>(got.deepstack.size()) == vllm_test::kH3EncTowerDeepstackCount);
  std::vector<float> flat;
  for (const std::vector<float>& d : got.deepstack) flat.insert(flat.end(), d.begin(), d.end());
  REQUIRE(flat.size() == std::size(vllm_test::kH3EncTowerDeepstackGolden));
  const double deep_err =
      MaxAbsDiff(flat, vllm_test::kH3EncTowerDeepstackGolden, flat.size());
  INFO("vision tower deepstack max|diff| = " << deep_err);
  CHECK(deep_err <= 1e-4);

  // Two images of DIFFERENT sizes were packed, so the position-embedding
  // interpolation and the per-frame cu_seqlens both had to handle a ragged batch.
  CHECK(grid.size() == 6);
  CHECK(grid[1] != grid[4]);
}

TEST_CASE("minimax_h3: the encoder GGUF visual.* loader dequantizes the vision tower") {
  // The record reconciliation (spec §8.8) found the vision tower math was gated ONLY
  // at reduced dims with SYNTHETIC weights and never wired to real weights: the
  // encoder GGUF loader loaded the TEXT tower only and skipped every visual.* tensor.
  // This gates the new loader — that it reads a `visual.*` GGUF, DEQUANTIZES each
  // ggml-block projection + F16 patch/pos to f32, and fills the shared
  // multimodal::Qwen3VLVisionWeights the Qwen3-VL front end consumes — the piece that
  // makes a REAL-weights vision forward possible. The loader is encoding-agnostic (it
  // calls DequantGgufRowToF32 per tensor), so a Q8_0 fixture exercises the exact code
  // path the shipped Q4_K/Q5_K tower takes; the real-file forward is the on-box proof.
  vllm::multimodal::Qwen3VLVisionConfig cfg;  // reduced dims, but the tower's exact shape math
  cfg.hidden_size = 64;
  cfg.num_heads = 4;
  cfg.depth = 3;
  cfg.intermediate_size = 128;
  cfg.out_hidden_size = 96;
  cfg.patch_size = 16;
  cfg.temporal_patch_size = 2;
  cfg.spatial_merge_size = 2;
  cfg.num_position_embeddings = 2304;
  cfg.in_channels = 3;
  cfg.deepstack_visual_indexes = {1};  // one DeepStack merger
  const int64_t dim = cfg.hidden_size;
  const int64_t patch_elems =
      cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
  const int64_t merged = dim * cfg.spatial_merge_size * cfg.spatial_merge_size;

  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "qwen3vl"));
  std::map<std::string, std::vector<float>> orig;  // logical name -> the exact f32 written
  auto add_f32 = [&](const std::string& name, int64_t numel) {
    const std::vector<float> v = MakeParam("vg." + name, numel, 0.05);
    orig[name] = v;
    std::string bytes(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    builder.AddTensor(name, {static_cast<uint64_t>(numel)}, /*ggml_type=*/0 /*F32*/, bytes);
  };
  // A quantized [out, in] projection stored the way the real tower ships it (block
  // encoding). Rows are whole Q8_0 blocks (in % 32 == 0 here).
  auto add_q8 = [&](const std::string& name, int64_t out_dim, int64_t in_dim) {
    const std::vector<float> values = MakeParam("vg." + name, out_dim * in_dim, 0.05);
    orig[name] = values;
    const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, in_dim);
    std::string bytes(static_cast<size_t>(out_dim) * row_bytes, '\0');
    vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
    REQUIRE(q != nullptr);
    for (int64_t r = 0; r < out_dim; ++r)
      q(values.data() + r * in_dim, bytes.data() + static_cast<size_t>(r) * row_bytes, in_dim);
    builder.AddTensor(name, {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
                      /*ggml_type=*/8 /*Q8_0*/, bytes);
  };

  const std::string V = "visual.";
  add_q8(V + "patch_embed.proj.weight", dim, patch_elems);
  add_f32(V + "patch_embed.proj.bias", dim);
  add_f32(V + "pos_embed.weight", cfg.num_position_embeddings * dim);
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string p = V + "blocks." + std::to_string(l);
    add_f32(p + ".norm1.weight", dim);
    add_f32(p + ".norm1.bias", dim);
    add_f32(p + ".norm2.weight", dim);
    add_f32(p + ".norm2.bias", dim);
    add_q8(p + ".attn.qkv.weight", 3 * dim, dim);
    add_f32(p + ".attn.qkv.bias", 3 * dim);
    add_q8(p + ".attn.proj.weight", dim, dim);
    add_f32(p + ".attn.proj.bias", dim);
    add_q8(p + ".mlp.linear_fc1.weight", cfg.intermediate_size, dim);
    add_f32(p + ".mlp.linear_fc1.bias", cfg.intermediate_size);
    add_q8(p + ".mlp.linear_fc2.weight", dim, cfg.intermediate_size);
    add_f32(p + ".mlp.linear_fc2.bias", dim);
  }
  // main merger norms the pre-shuffle width (dim); deepstack norms the post-shuffle (merged)
  add_f32(V + "merger.norm.weight", dim);
  add_f32(V + "merger.norm.bias", dim);
  add_q8(V + "merger.linear_fc1.weight", merged, merged);
  add_f32(V + "merger.linear_fc1.bias", merged);
  add_q8(V + "merger.linear_fc2.weight", cfg.out_hidden_size, merged);
  add_f32(V + "merger.linear_fc2.bias", cfg.out_hidden_size);
  add_f32(V + "deepstack_merger_list.0.norm.weight", merged);
  add_f32(V + "deepstack_merger_list.0.norm.bias", merged);
  add_q8(V + "deepstack_merger_list.0.linear_fc1.weight", merged, merged);
  add_f32(V + "deepstack_merger_list.0.linear_fc1.bias", merged);
  add_q8(V + "deepstack_merger_list.0.linear_fc2.weight", cfg.out_hidden_size, merged);
  add_f32(V + "deepstack_merger_list.0.linear_fc2.bias", cfg.out_hidden_size);

  const std::string path = "/tmp/minimax_h3_visual_q.gguf";
  {
    const std::string bytes = builder.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), fh) == bytes.size());
    std::fclose(fh);
  }

  const vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::multimodal::Qwen3VLVisionWeights vw = vllm::LoadQwen3VLVisionFromGguf(gguf, cfg);

  // STRUCTURE: every field is filled at the flat [out*in] size the tower reads.
  CHECK(static_cast<int64_t>(vw.patch_proj_w.size()) == dim * patch_elems);
  CHECK(static_cast<int64_t>(vw.patch_proj_b.size()) == dim);
  CHECK(static_cast<int64_t>(vw.pos_embed_w.size()) == cfg.num_position_embeddings * dim);
  REQUIRE(static_cast<int64_t>(vw.blocks.size()) == cfg.depth);
  for (const vllm::multimodal::VisionBlockWeights& b : vw.blocks) {
    CHECK(static_cast<int64_t>(b.qkv_w.size()) == 3 * dim * dim);
    CHECK(static_cast<int64_t>(b.qkv_b.size()) == 3 * dim);
    CHECK(static_cast<int64_t>(b.proj_w.size()) == dim * dim);
    CHECK(static_cast<int64_t>(b.fc1_w.size()) == cfg.intermediate_size * dim);
    CHECK(static_cast<int64_t>(b.fc2_w.size()) == dim * cfg.intermediate_size);
    CHECK(static_cast<int64_t>(b.norm1_w.size()) == dim);
  }
  CHECK(vw.merger.use_postshuffle_norm == false);
  CHECK(static_cast<int64_t>(vw.merger.norm_w.size()) == dim);         // pre-shuffle width
  CHECK(static_cast<int64_t>(vw.merger.fc2_w.size()) == cfg.out_hidden_size * merged);
  REQUIRE(vw.deepstack_mergers.size() == cfg.deepstack_visual_indexes.size());
  CHECK(vw.deepstack_mergers[0].use_postshuffle_norm == true);
  CHECK(static_cast<int64_t>(vw.deepstack_mergers[0].norm_w.size()) == merged);  // post-shuffle

  // DEQUANT CORRECTNESS: an f32 tensor round-trips EXACTLY, a Q8_0 tensor within its
  // block tolerance, in the SAME flat order — this is the load that the reduced-dim
  // synthetic gate never exercised on real bytes.
  {
    const std::vector<float>& want = orig[V + "patch_embed.proj.bias"];
    double e = MaxAbsDiff(vw.patch_proj_b, want.data(), want.size());
    INFO("patch bias f32 exact err=" << e);
    CHECK(e == 0.0);
  }
  {
    const std::vector<float>& want = orig[V + "blocks.0.attn.qkv.weight"];
    REQUIRE(vw.blocks[0].qkv_w.size() == want.size());
    double e = MaxAbsDiff(vw.blocks[0].qkv_w, want.data(), want.size());
    INFO("qkv Q8_0 dequant err=" << e);
    CHECK(e <= 5e-3);  // Q8_0 block tolerance
    // and it is NON-degenerate: real spread, not all-zeros.
    double s2 = 0.0;
    for (float f : vw.blocks[0].qkv_w) s2 += double(f) * f;
    CHECK(std::sqrt(s2 / vw.blocks[0].qkv_w.size()) > 1e-3);
  }

  // The production config the driver uses is the measured H3 vision geometry.
  const vllm::multimodal::Qwen3VLVisionConfig prod = vllm::MiniMaxH3EncoderVisionConfig();
  CHECK(prod.hidden_size == 1152);
  CHECK(prod.num_heads == 16);
  CHECK(prod.depth == 27);
  CHECK(prod.out_hidden_size == 5120);
  CHECK(prod.num_position_embeddings == 2304);
  CHECK(prod.deepstack_visual_indexes.size() == 3);
}

TEST_CASE("minimax_h3: condition-noise augmentation matches upstream") {
  // fl2va/ref2va pin their keyframe and reference-audio rows to a NOISED anchor.
  // The mix is trivial; the ROW ACCOUNTING is what this gates -- and the golden
  // feeds our side the SAME noise upstream drew, so the comparison isolates the
  // accounting from torch's RNG (which the pipeline also takes as an input).
  std::vector<int64_t> shapes;
  for (size_t i = 0; i < std::size(vllm_test::kH3CondShapes); ++i) {
    shapes.push_back(vllm_test::kH3CondShapes[i]);
  }
  const std::vector<float> clean =
      MakeParam("cond.clean_video", vllm_test::kH3CondRows * 96, 1.0);
  const std::vector<float> noise(vllm_test::kH3CondNoiseRows,
                                 vllm_test::kH3CondNoiseRows + std::size(vllm_test::kH3CondNoiseRows));
  const std::vector<float> got = vllm::MiniMaxH3ImgvidCondNoiseAug(
      clean, shapes, vllm_test::kH3CondTargetLatentT, vllm_test::kH3CondImgvidFrames,
      vllm_test::kH3CondNoiseAug[0], noise);
  REQUIRE(got.size() == std::size(vllm_test::kH3CondGolden));
  CHECK(MaxAbsDiff(got, vllm_test::kH3CondGolden, got.size()) <= 1e-6);

  std::vector<int64_t> audio_t;
  for (size_t i = 0; i < std::size(vllm_test::kH3CondAudioT); ++i) {
    audio_t.push_back(vllm_test::kH3CondAudioT[i]);
  }
  const std::vector<float> clean_audio =
      MakeParam("cond.clean_audio", vllm_test::kH3CondAudioRows * 32, 1.0);
  const std::vector<float> audio_noise(
      vllm_test::kH3CondAudioNoiseRows,
      vllm_test::kH3CondAudioNoiseRows + std::size(vllm_test::kH3CondAudioNoiseRows));
  const std::vector<float> got_audio = vllm::MiniMaxH3AudioCondNoiseAug(
      clean_audio, audio_t, vllm_test::kH3CondNoiseAug[0], audio_noise);
  REQUIRE(got_audio.size() == std::size(vllm_test::kH3CondAudioGolden));
  CHECK(MaxAbsDiff(got_audio, vllm_test::kH3CondAudioGolden, got_audio.size()) <= 1e-6);

  // noise_aug == 1.0 is the documented identity (the anchor IS the clean latent).
  const std::vector<float> identity =
      vllm::MiniMaxH3ImgvidCondNoiseAug(clean, shapes, vllm_test::kH3CondTargetLatentT,
                                        vllm_test::kH3CondImgvidFrames, 1.0, noise);
  for (size_t i = 0; i < identity.size(); ++i) CHECK(identity[i] == clean[i]);

  // Shape/row-count disagreements must throw, not silently mis-slice.
  CHECK_THROWS(vllm::MiniMaxH3ImgvidCondNoiseAug(clean, {1, 3, 6}, 3, 1, 0.999, noise));
  CHECK_THROWS(vllm::MiniMaxH3ImgvidCondNoiseAug(clean, shapes, 3, 1, 1.5, noise));
  CHECK_THROWS(vllm::MiniMaxH3AudioCondNoiseAug(clean_audio, {}, 0.999, audio_noise));
}

TEST_CASE("minimax_h3: reference-video geometry and frame schedule match upstream") {
  // The PURE-MATH half of reference_video.py. The rest of that module (probe,
  // transcode, frame extraction, audio decode) shells out to ffmpeg and is blocked
  // on the same dependency decision as MP4 muxing -- one decision unlocks both
  // reference-video INPUT decode and generated-video OUTPUT encode.
  for (int64_t c = 0; c < vllm_test::kH3RefVidShapeCases; ++c) {
    const std::pair<int64_t, int64_t> got = vllm::MiniMaxH3ReferenceVideoShape(
        vllm_test::kH3RefVidShapeInputs[c * 2], vllm_test::kH3RefVidShapeInputs[c * 2 + 1]);
    INFO("shape case " << c);
    CHECK(got.first == vllm_test::kH3RefVidShapeGolden[c * 2]);
    CHECK(got.second == vllm_test::kH3RefVidShapeGolden[c * 2 + 1]);
    // Every canvas must land on the 32 grid.
    CHECK(got.first % 32 == 0);
    CHECK(got.second % 32 == 0);
    // The pixel budget is applied BEFORE the snap to 32, so the snapped canvas
    // may exceed it slightly -- upstream does not re-check after rounding. Assert
    // the real invariant (within one grid step on each axis) rather than a
    // stricter one the reference does not hold to.
    const int64_t slack = (got.first + 32) * (got.second + 32);
    CHECK(got.first * got.second <= slack);
    CHECK(got.first * got.second <=
          vllm::kMiniMaxH3RefVideoMaxPixels + 32 * (got.first + got.second) + 32 * 32);
  }
  // Out-of-range aspect ratios are rejected, not clamped.
  CHECK_THROWS(vllm::MiniMaxH3ReferenceVideoShape(5000, 100));

  int64_t idx_offset = 0, blk_offset = 0;
  for (int64_t c = 0; c < vllm_test::kH3RefVidFrameCases; ++c) {
    const vllm::MiniMaxH3ReferenceVideoSchedule got =
        vllm::MiniMaxH3ReferenceVideoFrameSchedule(vllm_test::kH3RefVidFrameCounts[c]);
    INFO("frame case " << c << " count=" << vllm_test::kH3RefVidFrameCounts[c]);
    REQUIRE(static_cast<int64_t>(got.indices.size()) == vllm_test::kH3RefVidIndexLens[c]);
    for (size_t i = 0; i < got.indices.size(); ++i) {
      CHECK(got.indices[i] == vllm_test::kH3RefVidIndicesGolden[idx_offset + i]);
    }
    REQUIRE(static_cast<int64_t>(got.block_timestamps.size()) == vllm_test::kH3RefVidBlockLens[c]);
    for (size_t i = 0; i < got.block_timestamps.size(); ++i) {
      CHECK(got.block_timestamps[i] ==
            doctest::Approx(vllm_test::kH3RefVidBlockTimestamps[blk_offset + i]).epsilon(1e-9));
    }
    // Indices must be strictly increasing and inside the clip.
    for (size_t i = 1; i < got.indices.size(); ++i) CHECK(got.indices[i] > got.indices[i - 1]);
    CHECK(got.indices.back() < vllm_test::kH3RefVidFrameCounts[c]);
    idx_offset += static_cast<int64_t>(got.indices.size());
    blk_offset += static_cast<int64_t>(got.block_timestamps.size());
  }
  CHECK_THROWS(vllm::MiniMaxH3ReferenceVideoFrameSchedule(0));
}

TEST_CASE("minimax_h3: video VAE tiling plan and seam blend match upstream") {
  // The tile plan is not a simple stride: it picks the smallest tile count whose
  // MINIMUM overlaps still cover the input, then distributes the leftover slack in
  // whole vae_ratio units ROUND-ROBIN across the seams. Getting that wrong shifts
  // every tile after the first and shows up as seam artifacts, not as an error.
  int64_t s_off = 0, l_off = 0, o_off = 0;
  for (int64_t c = 0; c < vllm_test::kH3TileCases; ++c) {
    const int64_t input_len = vllm_test::kH3TileInputs[c * 3 + 0];
    const int64_t tile_size = vllm_test::kH3TileInputs[c * 3 + 1];
    const int64_t overlap_min = vllm_test::kH3TileInputs[c * 3 + 2];
    const vllm::MiniMaxH3TilePlan got =
        vllm::MiniMaxH3SplitTiles(input_len, tile_size, overlap_min, vllm_test::kH3TileVaeRatio);
    INFO("tile case " << c << " len=" << input_len << " tile=" << tile_size);
    REQUIRE(static_cast<int64_t>(got.starts.size()) == vllm_test::kH3TileCounts[c]);
    for (size_t i = 0; i < got.starts.size(); ++i) {
      CHECK(got.starts[i] == vllm_test::kH3TileStarts[s_off + i]);
      CHECK(got.lengths[i] == vllm_test::kH3TileLens[l_off + i]);
    }
    for (size_t i = 0; i < got.overlaps.size(); ++i) {
      CHECK(got.overlaps[i] == vllm_test::kH3TileOverlaps[o_off + i]);
    }
    // Structural invariants the plan must satisfy for tiling to be lossless:
    // tiles cover the whole axis, and every seam overlaps by at least the minimum.
    CHECK(got.starts.front() == 0);
    CHECK(got.starts.back() + got.lengths.back() >= input_len);
    for (size_t i = 0; i < got.overlaps.size(); ++i) {
      CHECK(got.overlaps[i] >= overlap_min);
      CHECK(got.overlaps[i] % vllm_test::kH3TileVaeRatio == overlap_min % vllm_test::kH3TileVaeRatio);
    }
    s_off += static_cast<int64_t>(got.starts.size());
    l_off += static_cast<int64_t>(got.lengths.size());
    o_off += static_cast<int64_t>(got.starts.size()) - 1;
  }

  // A tile larger than the input is a single untiled pass with no seams.
  const vllm::MiniMaxH3TilePlan single = vllm::MiniMaxH3SplitTiles(100, 256, 64, 16);
  CHECK(single.starts.size() == 1);
  CHECK(single.lengths[0] == 100);
  CHECK(single.overlaps.empty());

  // The seam cross-fade.
  const std::vector<float> a = MakeParam("tiling.a", vllm_test::kH3TileBlendLen, 1.0);
  const std::vector<float> b = MakeParam("tiling.b", vllm_test::kH3TileBlendLen, 1.0);
  const std::vector<float> blended =
      vllm::MiniMaxH3BlendTiles(a, b, vllm_test::kH3TileBlendExtent);
  REQUIRE(blended.size() == std::size(vllm_test::kH3TileBlendGolden));
  CHECK(MaxAbsDiff(blended, vllm_test::kH3TileBlendGolden, blended.size()) <= 1e-6);
  // The fade starts fully on `a` and ends fully on `b`.
  CHECK(blended.front() == doctest::Approx(a[a.size() - vllm_test::kH3TileBlendExtent]));
  CHECK(blended[static_cast<size_t>(vllm_test::kH3TileBlendExtent)] ==
        doctest::Approx(b[static_cast<size_t>(vllm_test::kH3TileBlendExtent)]));
}

TEST_CASE("minimax_h3: presentation token tags match upstream") {
  // The fl2va "vision-span override" the denoise loop requires callers to have
  // applied. The load-bearing detail: a vision block is
  // <|vision_start|> + pad*count + <|vision_end|>, and the WHOLE block -- markers
  // included -- is tagged VIDEO. Tagging only the pads leaves two markers as TEXT
  // and shifts every AdaLN modulation index after them.
  std::vector<vllm::MiniMaxH3PresentationSpan> spans;
  for (int64_t i = 0; i < vllm_test::kH3PresSpanCount; ++i) {
    vllm::MiniMaxH3PresentationSpan span;
    span.kind = vllm_test::kH3PresSpanKinds[i] == 0
                    ? vllm::MiniMaxH3PresentationSpan::Kind::kVision
                    : vllm::MiniMaxH3PresentationSpan::Kind::kText;
    span.length = vllm_test::kH3PresSpanLens[i];
    spans.push_back(span);
  }

  const std::vector<int64_t> tags = vllm::MiniMaxH3BuildPresentationTokenTags(spans);
  REQUIRE(static_cast<int64_t>(tags.size()) == vllm_test::kH3PresTagCount);
  REQUIRE(tags.size() == std::size(vllm_test::kH3PresTagsGolden));
  for (size_t i = 0; i < tags.size(); ++i) CHECK(tags[i] == vllm_test::kH3PresTagsGolden[i]);

  // The tags must only ever be TEXT or VIDEO -- never the audio or padding tags.
  for (int64_t tag : tags) {
    CHECK((tag == vllm::kMiniMaxH3TagText || tag == vllm::kMiniMaxH3TagVideo));
  }

  // A vision block is pad_count + 2 tokens (the two markers).
  CHECK(vllm::MiniMaxH3VisionBlockTokenLength(3) == 5);
  CHECK(vllm::MiniMaxH3VisionBlockTokenLength(1) == 3);
  CHECK_THROWS(vllm::MiniMaxH3VisionBlockTokenLength(0));
  CHECK_THROWS(vllm::MiniMaxH3BuildPresentationTokenTags({}));

  // Every VIDEO run must be a whole vision block, i.e. its length must equal one
  // of the emitted vision spans -- proving the markers were tagged with the pads.
  std::vector<int64_t> video_runs;
  for (size_t i = 0; i < tags.size();) {
    if (tags[i] != vllm::kMiniMaxH3TagVideo) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < tags.size() && tags[j] == vllm::kMiniMaxH3TagVideo) ++j;
    video_runs.push_back(static_cast<int64_t>(j - i));
    i = j;
  }
  std::vector<int64_t> vision_spans;
  for (const vllm::MiniMaxH3PresentationSpan& span : spans) {
    if (span.kind == vllm::MiniMaxH3PresentationSpan::Kind::kVision) {
      vision_spans.push_back(span.length);
    }
  }
  CHECK(video_runs == vision_spans);
}

TEST_CASE("minimax_h3: the VAE encoder ResnetBlock3D matches upstream") {
  // The repeated unit of the video VAE's 3D-CNN ENCODER (used for image/video
  // CONDITIONING, not for output frames). Two details are load-bearing and both
  // are exercised: the convolution is CAUSAL in time (all padding on the LEFT, so
  // a frame never sees the future), and GroupNorm's statistics span TIME as well
  // as space.
  vllm::MiniMaxH3ResnetBlock3dConfig config;
  config.in_channels = vllm_test::kH3Res3dInCh;
  config.out_channels = vllm_test::kH3Res3dOutCh;
  config.t = vllm_test::kH3Res3dT;
  config.h = vllm_test::kH3Res3dH;
  config.w = vllm_test::kH3Res3dW;
  config.num_groups = vllm_test::kH3Res3dGroups;
  config.eps = 1e-6;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& n, int64_t count, double scale, double offset) {
    weights.tensors["rb." + n] = MakeParam("resnet3d." + n, count, scale, offset);
  };
  put("norm1.weight", config.in_channels, 0.1, 1.0);
  put("norm1.bias", config.in_channels, 0.05, 0.0);
  put("norm2.weight", config.out_channels, 0.1, 1.0);
  put("norm2.bias", config.out_channels, 0.05, 0.0);
  put("conv1.weight", config.out_channels * config.in_channels * 27, 0.1, 0.0);
  put("conv1.bias", config.out_channels, 0.05, 0.0);
  put("conv2.weight", config.out_channels * config.out_channels * 27, 0.1, 0.0);
  put("conv2.bias", config.out_channels, 0.05, 0.0);
  put("nin_shortcut.weight", config.out_channels * config.in_channels, 0.1, 0.0);
  put("nin_shortcut.bias", config.out_channels, 0.05, 0.0);

  const int64_t spatial = config.t * config.h * config.w;
  const std::vector<float> x =
      MakeParam("resnet3d.input", config.in_channels * spatial, 1.0);
  const std::vector<float> got =
      vllm::MiniMaxH3ResnetBlock3dForward(config, weights, "rb", x);

  REQUIRE(got.size() == std::size(vllm_test::kH3Res3dGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3Res3dGolden, got.size());
  INFO("resnet3d max|diff| = " << err);
  CHECK(err <= 1e-4);

  // CAUSALITY, proven rather than assumed: perturbing the LAST frame must leave
  // the FIRST frame's output bit-identical, while the last frame's own output
  // changes. A non-causal (symmetric) temporal pad would break the first check.
  std::vector<float> perturbed = x;
  for (int64_t c = 0; c < config.in_channels; ++c) {
    perturbed[static_cast<size_t>(c * spatial + (config.t - 1) * config.h * config.w)] += 5.0f;
  }
  const std::vector<float> other =
      vllm::MiniMaxH3ResnetBlock3dForward(config, weights, "rb", perturbed);
  // GroupNorm's statistics span time, so a change anywhere perturbs every frame
  // a little -- an exact-equality check on the whole BLOCK would be wrong. Here
  // the weaker (but still meaningful) claim is asserted: the perturbed frame moves
  // strictly more than the first. Strict causality is then proven on the bare
  // CONVOLUTION below, where no norm mixes the frames.
  double first_delta = 0.0, last_delta = 0.0;
  const int64_t frame = config.h * config.w;
  for (int64_t c = 0; c < config.out_channels; ++c) {
    for (int64_t i = 0; i < frame; ++i) {
      first_delta = std::max(first_delta,
                             std::abs(static_cast<double>(other[static_cast<size_t>(c * spatial + i)]) -
                                      got[static_cast<size_t>(c * spatial + i)]));
      const size_t last = static_cast<size_t>(c * spatial + (config.t - 1) * frame + i);
      last_delta = std::max(last_delta,
                            std::abs(static_cast<double>(other[last]) - got[last]));
    }
  }
  CHECK(last_delta > first_delta);

  // The causal convolution alone: with norms bypassed, a future-frame change must
  // NOT reach an earlier frame at all.
  vllm::MiniMaxH3Conv3dSpec spec;
  spec.in_channels = spec.out_channels = 1;
  spec.t = 3;
  spec.h = spec.w = 1;
  spec.kernel_t = 3;
  spec.kernel_h = spec.kernel_w = 1;
  spec.pad_t = 1;
  spec.pad_h = spec.pad_w = 0;
  spec.causal = true;
  const std::vector<float> kernel = {1.0f, 2.0f, 4.0f};
  const std::vector<float> a = {1.0f, 0.0f, 0.0f};
  const std::vector<float> b = {1.0f, 0.0f, 9.0f};  // only the LAST frame differs
  const std::vector<float> ca = vllm::MiniMaxH3CausalConv3d(a, spec, kernel, nullptr);
  const std::vector<float> cb = vllm::MiniMaxH3CausalConv3d(b, spec, kernel, nullptr);
  REQUIRE(ca.size() == cb.size());
  CHECK(ca[0] == cb[0]);   // frame 0 cannot see frame 2
  CHECK(ca[1] == cb[1]);   // frame 1 cannot see frame 2
  CHECK(ca[2] != cb[2]);   // frame 2 does
}

TEST_CASE("minimax_h3: the VAE encoder Downsample3D matches upstream") {
  // The strided conv between encoder levels. Its subtlety is the ASYMMETRIC
  // pre-pad: one pixel on the RIGHT of W and the BOTTOM of H before a stride-2
  // conv with padding (1, 0, 0). Padding symmetrically instead shifts the whole
  // sampling lattice by half a pixel -- no error, just a subtly wrong latent.
  vllm::MiniMaxH3Downsample3dConfig config;
  config.in_channels = vllm_test::kH3Down3dInCh;
  config.out_channels = vllm_test::kH3Down3dOutCh;
  config.t = vllm_test::kH3Down3dT;
  config.h = vllm_test::kH3Down3dH;
  config.w = vllm_test::kH3Down3dW;
  config.time_stride = 2;
  config.space_stride = 2;

  const std::vector<float> weight =
      MakeParam("down3d.conv.weight", config.out_channels * config.in_channels * 27, 0.1);
  const std::vector<float> bias = MakeParam("down3d.conv.bias", config.out_channels, 0.05);
  const std::vector<float> x = MakeParam(
      "down3d.input", config.in_channels * config.t * config.h * config.w, 1.0);

  const std::vector<float> got = vllm::MiniMaxH3Downsample3d(x, config, weight, bias);

  const int64_t expected = config.out_channels * vllm_test::kH3Down3dOutT *
                           vllm_test::kH3Down3dOutH * vllm_test::kH3Down3dOutW;
  CHECK(static_cast<int64_t>(got.size()) == expected);
  REQUIRE(got.size() == std::size(vllm_test::kH3Down3dGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3Down3dGolden, got.size());
  INFO("downsample3d max|diff| = " << err);
  CHECK(err <= 1e-4);

  // Strides genuinely halve each axis (T 3->2 under the causal pad, H/W 4->2).
  CHECK(vllm_test::kH3Down3dOutH == config.h / 2);
  CHECK(vllm_test::kH3Down3dOutW == config.w / 2);
}

TEST_CASE("minimax_h3: the whole VAE 3D-CNN encoder matches upstream") {
  // Completes the video VAE: conv_in -> per level [ResnetBlock3D x N, then a
  // Downsample3D or a 1x1x1 channel match] -> GroupNorm -> SiLU -> conv_out.
  // Conditioning-only -- a t2va generation path never calls this.
  vllm::MiniMaxH3EncoderFcn3dConfig config;
  config.ch = vllm_test::kH3EncFcnCh;
  config.ch_mult.assign(vllm_test::kH3EncFcnChMult,
                        vllm_test::kH3EncFcnChMult + vllm_test::kH3EncFcnLevels);
  config.space_down.assign(vllm_test::kH3EncFcnSpaceDown,
                           vllm_test::kH3EncFcnSpaceDown + vllm_test::kH3EncFcnLevels);
  config.time_down.assign(vllm_test::kH3EncFcnTimeDown,
                          vllm_test::kH3EncFcnTimeDown + vllm_test::kH3EncFcnLevels);
  config.num_res_blocks = vllm_test::kH3EncFcnResBlocks;
  config.in_channels = vllm_test::kH3EncFcnInCh;
  config.z_channels = vllm_test::kH3EncFcnZCh;
  config.t = vllm_test::kH3EncFcnT;
  config.h = vllm_test::kH3EncFcnH;
  config.w = vllm_test::kH3EncFcnW;
  config.num_groups = 32;
  config.eps = 1e-6;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& n, int64_t count, double scale, double offset) {
    weights.tensors[n] = MakeParam("encfcn." + n, count, scale, offset);
  };

  const int64_t levels = vllm_test::kH3EncFcnLevels;
  std::vector<int64_t> block_mid(static_cast<size_t>(levels));
  for (int64_t i = 0; i < levels; ++i) block_mid[i] = config.ch * config.ch_mult[i];
  std::vector<int64_t> block_in(static_cast<size_t>(levels));
  block_in[0] = block_mid[0];
  for (int64_t i = 1; i < levels; ++i) block_in[i] = block_mid[i - 1];

  put("conv_in.weight", block_in[0] * config.in_channels * 27, 0.1, 0.0);
  put("conv_in.bias", block_in[0], 0.05, 0.0);
  for (int64_t l = 0; l < levels; ++l) {
    for (int64_t b = 0; b < config.num_res_blocks; ++b) {
      const std::string prefix = "down." + std::to_string(l) + ".block." + std::to_string(b);
      const int64_t ci = (b == 0) ? block_in[l] : block_mid[l];
      const int64_t co = block_mid[l];
      put(prefix + ".norm1.weight", ci, 0.1, 1.0);
      put(prefix + ".norm1.bias", ci, 0.05, 0.0);
      put(prefix + ".norm2.weight", co, 0.1, 1.0);
      put(prefix + ".norm2.bias", co, 0.05, 0.0);
      put(prefix + ".conv1.weight", co * ci * 27, 0.1, 0.0);
      put(prefix + ".conv1.bias", co, 0.05, 0.0);
      put(prefix + ".conv2.weight", co * co * 27, 0.1, 0.0);
      put(prefix + ".conv2.bias", co, 0.05, 0.0);
      if (ci != co) {
        put(prefix + ".nin_shortcut.weight", co * ci, 0.1, 0.0);
        put(prefix + ".nin_shortcut.bias", co, 0.05, 0.0);
      }
    }
    const std::string ds = "down." + std::to_string(l) + ".downsample";
    if (config.space_down[l] * config.time_down[l] > 1) {
      put(ds + ".conv.weight", block_mid[l] * block_mid[l] * 27, 0.1, 0.0);
      put(ds + ".conv.bias", block_mid[l], 0.05, 0.0);
    }
  }
  const int64_t last = block_mid[levels - 1];
  put("norm_out.weight", last, 0.1, 1.0);
  put("norm_out.bias", last, 0.05, 0.0);
  put("conv_out.weight", config.z_channels * last * 27, 0.1, 0.0);
  put("conv_out.bias", config.z_channels, 0.05, 0.0);

  const std::vector<float> x = MakeParam(
      "encfcn.input", config.in_channels * config.t * config.h * config.w, 1.0);
  vllm::MiniMaxH3VideoFrameShape shape;
  const std::vector<float> got =
      vllm::MiniMaxH3EncoderFcn3dForward(config, weights, x, &shape);

  CHECK(shape.channels == config.z_channels);
  CHECK(shape.t == vllm_test::kH3EncFcnOutT);
  CHECK(shape.h == vllm_test::kH3EncFcnOutH);
  CHECK(shape.w == vllm_test::kH3EncFcnOutW);
  REQUIRE(got.size() == std::size(vllm_test::kH3EncFcnGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3EncFcnGolden, got.size());
  INFO("encoder fcn3d max|diff| = " << err);
  CHECK(err <= 1e-4);
}

TEST_CASE("minimax_h3: the MM processor is Qwen3-VL reuse, with H3's own config") {
  // H3's FL2VA/processor is a stock Qwen3VLProcessor, so the MM front end is REUSE
  // of this project's existing Qwen3-VL processor rather than a new port. What
  // needed proving is that H3's ACTUAL config parses and drives it correctly --
  // including the video bounds, which differ from the image ones.
  const std::string preproc = R"({
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast"
  })";
  const std::string cfg = R"({
    "image_token_id": 151655, "vision_start_token_id": 151652,
    "vision_end_token_id": 151653
  })";
  const std::string preproc_path = "/tmp/minimax_h3_preproc.json";
  const std::string cfg_path = "/tmp/minimax_h3_cfg.json";
  for (const auto& entry : {std::make_pair(preproc_path, preproc), std::make_pair(cfg_path, cfg)}) {
    FILE* fh = std::fopen(entry.first.c_str(), "wb");
    REQUIRE(fh != nullptr);
    std::fwrite(entry.second.data(), 1, entry.second.size(), fh);
    std::fclose(fh);
  }

  const vllm::multimodal::Qwen3VLProcessorConfig config =
      vllm::multimodal::LoadQwen3VLProcessorConfig(preproc_path, cfg_path, "MiniMaxAI/MiniMax-H3");
  CHECK(config.patch_size == 16);
  CHECK(config.temporal_patch_size == 2);
  CHECK(config.merge_size == 2);
  // H3 normalizes to [-1, 1] with mean/std 0.5 -- NOT the usual CLIP statistics.
  CHECK(config.image_mean == doctest::Approx(0.5));
  CHECK(config.image_std == doctest::Approx(0.5));
  CHECK(config.min_pixels == 65536);
  CHECK(config.max_pixels == 16777216);

  // The vision tower consumes patches on a `patch_size * merge_size` grid.
  const int64_t factor = config.patch_size * config.merge_size;
  CHECK(factor == 32);

  // A 768x1344 canvas (H3's default) already sits on the 32 grid and inside the
  // budget, so smart_resize must be the IDENTITY -- resizing it would silently
  // change the conditioning image.
  const std::array<int64_t, 2> keep =
      vllm::multimodal::SmartResize(768, 1344, factor, config.min_pixels, config.max_pixels);
  CHECK(keep[0] == 768);
  CHECK(keep[1] == 1344);

  // Off-grid inputs snap to the grid and stay inside the pixel budget.
  for (const auto& wh : {std::make_pair<int64_t, int64_t>(1000, 700),
                         std::make_pair<int64_t, int64_t>(33, 4000),
                         std::make_pair<int64_t, int64_t>(4000, 33)}) {
    const std::array<int64_t, 2> got =
        vllm::multimodal::SmartResize(wh.first, wh.second, factor, config.min_pixels, config.max_pixels);
    INFO("smart_resize " << wh.first << "x" << wh.second);
    CHECK(got[0] % factor == 0);
    CHECK(got[1] % factor == 0);
    CHECK(got[0] > 0);
    CHECK(got[1] > 0);
  }

  // H3's VIDEO bounds differ from its image bounds (shortest 4096, longest
  // 25165824) and the video budget includes the TEMPORAL extent.
  const int64_t video_min = 4096, video_max = 25165824;
  const std::array<int64_t, 2> video = vllm::multimodal::VideoSmartResize(
      /*num_frames=*/8, 768, 1344, config.temporal_patch_size, factor, video_min, video_max);
  CHECK(video[0] % factor == 0);
  CHECK(video[1] % factor == 0);
  CHECK(video_min < config.min_pixels);   // video floor is LOWER than the image one
  CHECK(video_max > config.max_pixels);   // and its ceiling HIGHER

  std::remove(preproc_path.c_str());
  std::remove(cfg_path.c_str());
}

TEST_CASE("minimax_h3: the decoded waveform serializes to a valid WAV") {
  // Dependency-free audio output. The container question for /v1/videos (MP4, and
  // how to get a muxer) is still open; WAV needs none of it and is required under
  // either outcome.
  const int64_t channels = 2, samples = 5, rate = vllm::kMiniMaxH3AudioSampleRate;
  // CHANNEL-MAJOR, as the VAE emits: all of channel 0, then all of channel 1.
  const std::vector<float> waveform = {0.0f,  0.5f,  -0.5f, 1.0f,  -1.0f,
                                       0.25f, -0.25f, 0.75f, -0.75f, 0.125f};
  const std::string wav = vllm::MiniMaxH3WriteWav(waveform, channels, samples, rate);

  REQUIRE(wav.size() == 44u + static_cast<size_t>(channels * samples * 2));
  CHECK(wav.compare(0, 4, "RIFF") == 0);
  CHECK(wav.compare(8, 4, "WAVE") == 0);
  CHECK(wav.compare(12, 4, "fmt ") == 0);
  CHECK(wav.compare(36, 4, "data") == 0);

  auto u16 = [&](size_t off) {
    return static_cast<uint16_t>(static_cast<uint8_t>(wav[off]) |
                                 (static_cast<uint8_t>(wav[off + 1]) << 8));
  };
  auto u32 = [&](size_t off) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(static_cast<uint8_t>(wav[off + i])) << (8 * i);
    return v;
  };
  CHECK(u32(4) == 36u + static_cast<uint32_t>(channels * samples * 2));
  CHECK(u32(16) == 16u);                       // PCM fmt chunk
  CHECK(u16(20) == 1u);                        // format = PCM
  CHECK(u16(22) == channels);
  CHECK(u32(24) == static_cast<uint32_t>(rate));
  CHECK(u32(28) == static_cast<uint32_t>(rate * channels * 2));  // byte rate
  CHECK(u16(32) == channels * 2);              // block align
  CHECK(u16(34) == 16u);                       // bits per sample
  CHECK(u32(40) == static_cast<uint32_t>(channels * samples * 2));

  // INTERLEAVE: sample s of channel c must sit at frame s, slot c -- getting this
  // backwards yields audio that plays but with the channels time-smeared.
  auto sample_at = [&](int64_t frame, int64_t ch) {
    const size_t off = 44 + static_cast<size_t>((frame * channels + ch) * 2);
    return static_cast<int16_t>(u16(off));
  };
  for (int64_t s2 = 0; s2 < samples; ++s2) {
    for (int64_t c = 0; c < channels; ++c) {
      const float want = waveform[static_cast<size_t>(c * samples + s2)];
      const int16_t got = sample_at(s2, c);
      CHECK(got == static_cast<int16_t>(std::lround(want * 32767.0f)));
    }
  }
  // Full-scale maps to the extremes without wrapping.
  CHECK(sample_at(3, 0) == 32767);
  CHECK(sample_at(4, 0) == -32767);

  // Out-of-range input is clamped, not wrapped.
  const std::vector<float> hot = {2.0f, -2.0f};
  const std::string clamped = vllm::MiniMaxH3WriteWav(hot, 1, 2, rate);
  auto hot_at = [&](size_t i) {
    const size_t off = 44 + i * 2;
    return static_cast<int16_t>(static_cast<uint8_t>(clamped[off]) |
                                (static_cast<uint8_t>(clamped[off + 1]) << 8));
  };
  CHECK(hot_at(0) == 32767);
  CHECK(hot_at(1) == -32767);

  CHECK_THROWS(vllm::MiniMaxH3WriteWav(waveform, 3, samples, rate));  // size mismatch
}

TEST_CASE("minimax_h3: a reference WAV reads back channel-major, and refuses what it cannot fix") {
  // The reader exists for REFERENCE AUDIO: ref2va conditions on a supplied
  // waveform. It is gated against the writer it inverts, because a de-interleave
  // that is backwards produces audio that still plays -- just with the channels
  // time-smeared into each other, which no shape check catches.
  const int64_t rate = vllm::kMiniMaxH3AudioSampleRate;
  const int64_t samples = 9;
  std::vector<float> stereo(static_cast<size_t>(2 * samples));
  for (int64_t t = 0; t < samples; ++t) {
    stereo[static_cast<size_t>(t)] = 0.1f * static_cast<float>(t);              // channel 0
    stereo[static_cast<size_t>(samples + t)] = -0.05f * static_cast<float>(t);  // channel 1
  }
  const std::string wav = vllm::MiniMaxH3WriteWav(stereo, 2, samples, rate);

  int64_t got = 0;
  const std::vector<float> back = vllm::MiniMaxH3ReadWav(wav, 2, rate, &got);
  CHECK(got == samples);
  REQUIRE(back.size() == stereo.size());
  // 16-bit PCM: one quantization step is 1/32768, so the round trip is exact to
  // within half a step.
  const double err = MaxAbsDiff(back, stereo.data(), stereo.size());
  INFO("WAV round-trip max|diff| = " << err);
  CHECK(err <= 1.0 / 32768.0);
  // Channel-major really is channel-major: channel 1 must be the DESCENDING ramp.
  CHECK(back[static_cast<size_t>(samples + 1)] < 0.0f);
  CHECK(back[1] > 0.0f);

  // A MONO file is REPEATED up to the model's channel count (vae.py:305-313), so
  // a mono reference conditions both channels rather than silencing one.
  const std::vector<float> mono(stereo.begin(), stereo.begin() + samples);
  const std::string mono_wav = vllm::MiniMaxH3WriteWav(mono, 1, samples, rate);
  int64_t mono_got = 0;
  const std::vector<float> widened =
      vllm::MiniMaxH3ReadWav(mono_wav, vllm::kMiniMaxH3AudioChannels, rate, &mono_got);
  CHECK(mono_got == samples);
  REQUIRE(widened.size() == static_cast<size_t>(vllm::kMiniMaxH3AudioChannels * samples));
  for (int64_t t = 0; t < samples; ++t) {
    CHECK(widened[static_cast<size_t>(t)] == widened[static_cast<size_t>(samples + t)]);
  }

  // A rate the VAE was not trained at is REFUSED, not resampled: encoding a
  // 44.1 kHz file as 32 kHz would shift every latent frame, and there is no audio
  // dependency here to resample with.
  const std::string wrong_rate = vllm::MiniMaxH3WriteWav(stereo, 2, samples, 44100);
  CHECK_THROWS(vllm::MiniMaxH3ReadWav(wrong_rate, 2, rate, nullptr));
  CHECK_NOTHROW(vllm::MiniMaxH3ReadWav(wrong_rate, 2, /*want_sample_rate=*/0, nullptr));

  // And malformed input is refused rather than read as audio.
  CHECK_THROWS(vllm::MiniMaxH3ReadWav("not a wav at all", 2, rate, nullptr));
  std::string truncated = wav.substr(0, 40);
  CHECK_THROWS(vllm::MiniMaxH3ReadWav(truncated, 2, rate, nullptr));
  // A chunk BEFORE `data` must be walked over, not assumed away: a fixed 44-byte
  // header would read the LIST payload as samples.
  std::string with_list = wav.substr(0, 36);
  const std::string list_body = "INFOxxxx";
  with_list += "LIST";
  for (int i = 0; i < 4; ++i) {
    with_list.push_back(static_cast<char>((list_body.size() >> (8 * i)) & 0xFF));
  }
  with_list += list_body;
  with_list += wav.substr(36);
  int64_t list_got = 0;
  const std::vector<float> after_list = vllm::MiniMaxH3ReadWav(with_list, 2, rate, &list_got);
  CHECK(list_got == samples);
  CHECK(MaxAbsDiff(after_list, back.data(), back.size()) == 0.0);
}

TEST_CASE("minimax_h3: video output -- PPM frames and the MP4 mux command") {
  // The library produces the ARTIFACTS and BUILDS the argv; the example/server
  // layer spawns ffmpeg. Both halves here are pure data transforms, so both are
  // testable without a subprocess.
  vllm::MiniMaxH3VideoFrameShape shape;
  shape.channels = 3;
  shape.t = 2;
  shape.h = 2;
  shape.w = 3;
  const int64_t plane = shape.h * shape.w;
  std::vector<float> frames(static_cast<size_t>(3 * shape.t * plane), 0.0f);
  // Frame 1, pixel (0,0): pure red at full scale; (0,1): black; (1,2): white.
  auto set = [&](int64_t f, int64_t y, int64_t x, float r, float g, float b) {
    frames[static_cast<size_t>((0 * shape.t + f) * plane + y * shape.w + x)] = r;
    frames[static_cast<size_t>((1 * shape.t + f) * plane + y * shape.w + x)] = g;
    frames[static_cast<size_t>((2 * shape.t + f) * plane + y * shape.w + x)] = b;
  };
  set(1, 0, 0, 1.0f, -1.0f, -1.0f);
  set(1, 0, 1, -1.0f, -1.0f, -1.0f);
  set(1, 1, 2, 1.0f, 1.0f, 1.0f);

  const std::string ppm = vllm::MiniMaxH3WritePpmFrame(frames, shape, 1);
  const std::string header = "P6\n3 2\n255\n";
  CHECK(ppm.compare(0, header.size(), header) == 0);
  REQUIRE(ppm.size() == header.size() + static_cast<size_t>(plane * 3));
  auto px = [&](int64_t y, int64_t x, int64_t c) {
    return static_cast<uint8_t>(ppm[header.size() + static_cast<size_t>((y * shape.w + x) * 3 + c)]);
  };
  // [-1, 1] maps to [0, 255]; PPM is row-major INTERLEAVED RGB, unlike the
  // planar [C, T, H, W] the VAE emits.
  CHECK(px(0, 0, 0) == 255);
  CHECK(px(0, 0, 1) == 0);
  CHECK(px(0, 0, 2) == 0);
  CHECK(px(0, 1, 0) == 0);
  CHECK(px(1, 2, 0) == 255);
  CHECK(px(1, 2, 2) == 255);
  // Mid-grey (0.0) lands mid-range, and a different frame is genuinely different.
  CHECK(px(1, 0, 0) == 128);
  CHECK(vllm::MiniMaxH3WritePpmFrame(frames, shape, 0) != ppm);
  CHECK_THROWS(vllm::MiniMaxH3WritePpmFrame(frames, shape, 2));

  // The mux command.
  vllm::MiniMaxH3MuxRequest request;
  request.frame_pattern = "/tmp/h3/frame_%06d.ppm";
  request.audio_path = "/tmp/h3/audio.wav";
  request.output_path = "/tmp/h3/out.mp4";
  request.fps = vllm::kMiniMaxH3Fps;
  request.crf = 18;
  const std::vector<std::string> argv = vllm::MiniMaxH3BuildMp4MuxArgs(request);

  auto has = [&](const std::string& token) {
    return std::find(argv.begin(), argv.end(), token) != argv.end();
  };
  CHECK(argv.front() == "ffmpeg");
  CHECK(argv.back() == request.output_path);
  CHECK(has("-framerate"));
  CHECK(has("24"));                 // H3's fixed output frame rate
  CHECK(has(request.frame_pattern));
  CHECK(has(request.audio_path));
  CHECK(has("libx264"));
  CHECK(has("yuv420p"));            // so every player accepts it
  CHECK(has("aac"));
  CHECK(has("-shortest"));          // no trailing silence or orphan video
  CHECK(has("+faststart"));         // moov atom first => streamable

  // A silent clip must not request an audio codec or -shortest.
  vllm::MiniMaxH3MuxRequest silent = request;
  silent.audio_path.clear();
  const std::vector<std::string> silent_argv = vllm::MiniMaxH3BuildMp4MuxArgs(silent);
  CHECK(std::find(silent_argv.begin(), silent_argv.end(), "aac") == silent_argv.end());
  CHECK(std::find(silent_argv.begin(), silent_argv.end(), "-shortest") == silent_argv.end());
  CHECK(silent_argv.back() == request.output_path);

  vllm::MiniMaxH3MuxRequest bad = request;
  bad.output_path.clear();
  CHECK_THROWS(vllm::MiniMaxH3BuildMp4MuxArgs(bad));
}

TEST_CASE("minimax_h3: config parse enforces the upstream invariants") {
  nlohmann::json config = {
      {"num_layers", 50},        {"token_refiner_num_layers", 2}, {"hidden_size", 5376},
      {"num_attention_heads", 56}, {"attention_head_dim", 128},   {"ffn_hidden_size", 14336},
      {"latents_dim", 24},       {"audio_latents_dim", 32},       {"patch_size", {1, 2, 2}},
      {"text_dim", 5120},        {"timestep_input_dim", 256},     {"time_embed_hidden_size", 5376},
      {"time_embed_dim", 2688},  {"adaln_out_features", 18 * 5376},
      {"final_adaln_out_features", 2 * 5376},                     {"rope_inv_freq_len", 16},
  };
  const MiniMaxH3DitParams p = ParseMiniMaxH3DitParams(config);
  // The SHIPPED MiniMax-H3 geometry.
  CHECK(p.num_layers == 50);
  CHECK(p.hidden_size == 5376);
  CHECK(p.num_attention_heads == 56);
  CHECK(p.video_row_width() == 96);   // 24 * 1 * 2 * 2
  CHECK(p.rope_rot_dim() == 96);      // 6 * 16, rotating 96 of 128 head dims
  CHECK(p.rope_rot_dim() <= p.attention_head_dim);

  // adaln_out_features must stay 6 vectors x 3 modalities x hidden_size.
  nlohmann::json bad = config;
  bad["adaln_out_features"] = 17 * 5376;
  CHECK_THROWS(ParseMiniMaxH3DitParams(bad));
  // patch_size must carry exactly three values.
  nlohmann::json bad_patch = config;
  bad_patch["patch_size"] = {1, 2};
  CHECK_THROWS(ParseMiniMaxH3DitParams(bad_patch));
}

TEST_CASE("minimax_h3: the DiT weight contract covers the shipped checkpoint") {
  MiniMaxH3DitParams p;  // shipped defaults
  const std::vector<vllm::MiniMaxH3TensorSpec> specs = EnumerateMiniMaxH3DitTensors(p);

  // 11 stem entries + refiner (2 x 8 + 1) + blocks (50 x 10) + 7 final entries.
  CHECK(specs.size() == 11 + (2 * 8 + 1) + (50 * 10) + 7);

  auto find = [&](const std::string& name) -> const vllm::MiniMaxH3TensorSpec* {
    for (const vllm::MiniMaxH3TensorSpec& spec : specs) {
      if (spec.name == name) return &spec;
    }
    return nullptr;
  };
  // The 12 fp32 parameters plus the fp32 rope buffer
  // (minimax_h3_transformer.py:85-101) must be marked fp32 and nothing else.
  const char* fp32_names[] = {
      "video_patch_proj.weight",     "video_patch_proj.bias",
      "audio_patch_proj.weight",     "audio_patch_proj.bias",
      "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
      "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
      "final_layer.video_out.weight", "final_layer.video_out.bias",
      "final_layer.audio_out.weight", "final_layer.audio_out.bias",
      "rope.inv_freq",
  };
  size_t fp32_count = 0;
  for (const vllm::MiniMaxH3TensorSpec& spec : specs) fp32_count += spec.fp32 ? 1 : 0;
  CHECK(fp32_count == std::size(fp32_names));
  for (const char* name : fp32_names) {
    const vllm::MiniMaxH3TensorSpec* spec = find(name);
    REQUIRE(spec != nullptr);
    CHECK(spec->fp32);
  }

  // MHA: qkv is 3 * heads * head_dim rows wide, and the text condition projection
  // consumes the encoder's 5120-wide hidden state.
  const vllm::MiniMaxH3TensorSpec* qkv = find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(qkv != nullptr);
  CHECK(qkv->shape[0] == 3 * 56 * 128);
  CHECK(qkv->shape[1] == 5376);
  const vllm::MiniMaxH3TensorSpec* cond = find("condition_proj.weight");
  REQUIRE(cond != nullptr);
  CHECK(cond->shape[1] == 5120);
}

TEST_CASE("minimax_h3: grouped-qkv checkpoint reorder is a pure permutation") {
  // The checkpoint stores [q, k, v] PER HEAD; the fused projection wants
  // [q_all, k_all, v_all] (minimax_h3_transformer.py:139-168). H3 is MHA, so
  // heads_per_group == 1.
  const int64_t groups = 3, head_dim = 2, in_features = 2;
  const int64_t rows = groups * (1 + 2) * head_dim;
  std::vector<float> weight(static_cast<size_t>(rows * in_features));
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = static_cast<float>(i);

  const std::vector<float> out =
      MiniMaxH3ReorderGroupedQkv(weight, groups, 1, head_dim, in_features);
  REQUIRE(out.size() == weight.size());

  const int64_t q_rows = groups * head_dim;
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t r = 0; r < head_dim; ++r) {
      for (int64_t c = 0; c < in_features; ++c) {
        const int64_t src_q = (g * 3 * head_dim + r) * in_features + c;
        const int64_t src_k = (g * 3 * head_dim + head_dim + r) * in_features + c;
        const int64_t src_v = (g * 3 * head_dim + 2 * head_dim + r) * in_features + c;
        CHECK(out[static_cast<size_t>((g * head_dim + r) * in_features + c)] == weight[src_q]);
        CHECK(out[static_cast<size_t>((q_rows + g * head_dim + r) * in_features + c)] ==
              weight[src_k]);
        CHECK(out[static_cast<size_t>((2 * q_rows + g * head_dim + r) * in_features + c)] ==
              weight[src_v]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// KEEP-QUANT GGUF arm. This is the path that makes a quantized H3 run cheap on
// hardware WITHOUT fp4 tensor cores: the ggml block-quant GEMM carries no arch
// gate at all (src/vt/cuda/cuda_quant_dot.cu has no #if), unlike every
// cutlass/marlin/fp4 path. So it runs at native speed on archs where NVFP4 can
// only be emulated.
// ---------------------------------------------------------------------------

static void CheckKeepQuantForward(vt::Queue& q, const char* label) {
  MiniMaxH3DitParams want;
  want.num_layers = 2;
  want.token_refiner_num_layers = 1;
  want.hidden_size = 64;
  want.num_attention_heads = 4;
  want.attention_head_dim = 16;
  want.ffn_hidden_size = 128;
  want.latents_dim = 8;
  want.audio_latents_dim = 6;
  want.text_dim = 32;  // whole Q8_0 blocks, so condition_proj is keep-quant ELIGIBLE
  want.timestep_input_dim = 16;
  want.time_embed_hidden_size = 64;
  want.time_embed_dim = 32;
  want.adaln_out_features = 18 * want.hidden_size;
  want.final_adaln_out_features = 2 * want.hidden_size;
  want.rope_inv_freq_len = 2;

  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "wan"));

  // F32 for anything that is NOT an eligible 2-D projection (norms, biases,
  // rope.inv_freq): the loader must dequantize those either way.
  auto add_f32 = [&](const std::string& name, const std::vector<int64_t>& logical) {
    int64_t numel = 1;
    for (int64_t d : logical) numel *= d;
    const std::vector<float> values = MakeParam("kq." + name, numel, 0.1);
    std::string bytes(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = logical.rbegin(); it != logical.rend(); ++it) {
      ne.push_back(static_cast<uint64_t>(*it));
    }
    builder.AddTensor(name, ne, /*ggml_type=*/0 /*F32*/, bytes);
  };

  // Real Q8_0 blocks, produced by the SAME quantizer a converter would use, so
  // this runs over bytes a real checkpoint could contain.
  auto add_q8 = [&](const std::string& name, int64_t out_dim, int64_t in_dim) {
    const std::vector<float> values = MakeParam("kq." + name, out_dim * in_dim, 0.1);
    const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, in_dim);
    std::string bytes(static_cast<size_t>(out_dim) * row_bytes, '\0');
    vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
    REQUIRE(q != nullptr);
    for (int64_t r = 0; r < out_dim; ++r) {
      q(values.data() + r * in_dim, bytes.data() + static_cast<size_t>(r) * row_bytes, in_dim);
    }
    builder.AddTensor(name, {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
                      /*ggml_type=*/8 /*Q8_0*/, bytes);
  };

  const int64_t inner = want.num_attention_heads * want.attention_head_dim;
  const int64_t video_width = want.video_row_width();

  // fp32 islands stay f32 in the file; the DiT's own projections go Q8_0.
  add_f32("video_patch_proj.weight", {want.hidden_size, video_width});
  add_f32("video_patch_proj.bias", {want.hidden_size});
  add_f32("audio_patch_proj.weight", {want.hidden_size, want.audio_latents_dim});
  add_f32("audio_patch_proj.bias", {want.hidden_size});
  add_q8("condition_proj.weight", want.hidden_size, want.text_dim);
  add_f32("condition_proj.bias", {want.hidden_size});
  add_f32("time_embedder.proj_in.weight", {want.time_embed_hidden_size, want.timestep_input_dim});
  add_f32("time_embedder.proj_in.bias", {want.time_embed_hidden_size});
  add_f32("time_embedder.proj_out.weight", {want.time_embed_dim, want.time_embed_hidden_size});
  add_f32("time_embedder.proj_out.bias", {want.time_embed_dim});
  add_f32("rope.inv_freq", {want.rope_inv_freq_len});

  auto add_block = [&](const std::string& prefix, bool adaln) {
    add_f32(prefix + ".norm1.weight", {want.hidden_size});
    add_f32(prefix + ".norm2.weight", {want.hidden_size});
    add_q8(prefix + ".attn.qkv_proj.weight", 3 * inner, want.hidden_size);
    add_f32(prefix + ".attn.q_norm.weight", {want.attention_head_dim});
    add_f32(prefix + ".attn.k_norm.weight", {want.attention_head_dim});
    add_q8(prefix + ".attn.out_proj.weight", want.hidden_size, inner);
    add_q8(prefix + ".mlp.fc1.weight", 2 * want.ffn_hidden_size, want.hidden_size);
    add_q8(prefix + ".mlp.fc2.weight", want.hidden_size, want.ffn_hidden_size);
    if (adaln) {
      add_q8(prefix + ".adaln_proj.linear.weight", want.adaln_out_features, want.time_embed_dim);
      add_f32(prefix + ".adaln_proj.linear.bias", {want.adaln_out_features});
    }
  };
  for (int64_t i = 0; i < want.token_refiner_num_layers; ++i) {
    add_block("token_refiner.blocks." + std::to_string(i), false);
  }
  add_f32("token_refiner.final_norm.weight", {want.hidden_size});
  for (int64_t i = 0; i < want.num_layers; ++i) {
    add_block("blocks." + std::to_string(i), true);
  }
  add_f32("final_layer.norm.weight", {want.hidden_size});
  add_q8("final_layer.adaln_proj.linear.weight", want.final_adaln_out_features,
         want.time_embed_dim);
  add_f32("final_layer.adaln_proj.linear.bias", {want.final_adaln_out_features});
  add_f32("final_layer.video_out.weight", {video_width, want.hidden_size});
  add_f32("final_layer.video_out.bias", {video_width});
  add_f32("final_layer.audio_out.weight", {want.audio_latents_dim, want.hidden_size});
  add_f32("final_layer.audio_out.bias", {want.audio_latents_dim});

  const std::string path = "/tmp/minimax_h3_keepquant_test.gguf";
  {
    const std::string bytes = builder.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), fh) == bytes.size());
    std::fclose(fh);
  }

  const vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::MiniMaxH3GgufDit dequantized =
      vllm::LoadMiniMaxH3DitFromGguf(gguf, /*keep_quant=*/false);
  const vllm::MiniMaxH3GgufDit kept = vllm::LoadMiniMaxH3DitFromGguf(gguf, /*keep_quant=*/true);

  // The keep-quant load must ACTUALLY have kept things quantized -- otherwise the
  // output comparison below would pass trivially by comparing two identical paths.
  CHECK(dequantized.quant_storage.empty());
  CHECK_FALSE(kept.quant_storage.empty());
  CHECK(kept.quant_dtype.at("blocks.0.attn.qkv_proj.weight") == vt::DType::kQ8_0);
  CHECK(kept.weights.blocks[0].qkv_proj.dtype == vt::DType::kQ8_0);
  // ...and must have left the ineligible tensors alone (1-D norms/biases).
  CHECK(kept.quant_storage.count("blocks.0.norm1.weight") == 0);
  CHECK(kept.weights.blocks[0].norm1.dtype == vt::DType::kF32);
  // Keep-quant is the whole point: the resident bytes must be SMALLER.
  size_t kept_bytes = 0;
  for (const auto& kv : kept.quant_storage) kept_bytes += kv.second.size();
  size_t dequant_bytes = 0;
  for (const auto& kv : dequantized.storage) {
    if (kept.quant_storage.count(kv.first) != 0) dequant_bytes += kv.second.size() * sizeof(float);
  }
  INFO("kept " << kept_bytes << " B vs dequantized " << dequant_bytes << " B");
  CHECK(kept_bytes < dequant_bytes);

  // Both must produce the same answer. vt::MatmulBT routes the block weight to
  // kMatmulBTQuant, whose per-element product is bit-for-bit the dequantized
  // value; only the K-reduction ORDER differs, so this is a matmul tolerance.
  const std::unique_ptr<DitForwardCase> c = BuildDitForwardCase(kept.params);
  const MiniMaxH3DitDeviceWeights staged_deq =
      StageMiniMaxH3DitWeights(q, dequantized.params, dequantized.weights, vt::DType::kF32);
  const MiniMaxH3DitDeviceWeights staged_kq =
      StageMiniMaxH3DitWeights(q, kept.params, kept.weights, vt::DType::kF32);
  const MiniMaxH3DitOutputs out_deq =
      MiniMaxH3DitForwardDevice(q, dequantized.params, staged_deq.weights, c->in, vt::DType::kF32);
  const MiniMaxH3DitOutputs out_kq =
      MiniMaxH3DitForwardDevice(q, kept.params, staged_kq.weights, c->in, vt::DType::kF32);

  REQUIRE(out_kq.video_logits.size() == out_deq.video_logits.size());
  const double video_err =
      MaxAbsDiff(out_kq.video_logits, out_deq.video_logits.data(), out_kq.video_logits.size());
  const double audio_err =
      MaxAbsDiff(out_kq.audio_logits, out_deq.audio_logits.data(), out_kq.audio_logits.size());
  INFO(label << " keep-quant vs dequantized: video " << video_err << ", audio " << audio_err);
  CHECK(video_err <= 2e-3);
  CHECK(audio_err <= 2e-3);
  // And the forward must have produced something, not all zeros -- the failure
  // mode a keep-quant slice hits when its bytes never reach the device.
  double magnitude = 0.0;
  for (float v : out_kq.video_logits) magnitude = std::max(magnitude, std::abs((double)v));
  CHECK(magnitude > 1e-6);
}

TEST_CASE("minimax_h3: the KEEP-QUANT GGUF arm matches the dequantized load (CPU backend)") {
  vt::Queue q{Cpu(), nullptr};
  CheckKeepQuantForward(q, "cpu");
}

// The one that actually matters for the GGUF arm's premise: this exercises
// kMatmulBTQuant's CUDA kernel over H3's own shapes. It is also the only place the
// ALL-ZEROS failure mode is reachable -- a keep-quant slice whose bytes never reach
// the device reads as valid host memory on CPU and as zeros on the GPU.
TEST_CASE("minimax_h3: the KEEP-QUANT GGUF arm matches the dequantized load on CUDA") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckKeepQuantForward(q, "cuda");
}

// ---------------------------------------------------------------------------
// PATH 2: checkpoint loaders. The VAE/encoder FORWARDS were gated against the
// checkpoint's own remote code, which proved the MATH. These gate the BINDING:
// that the SHIPPED file's tensors actually land on the structs those forwards
// read. For the audio VAE they do not do so directly, and this is where that is
// caught.
// ---------------------------------------------------------------------------

TEST_CASE("minimax_h3: the REAL audio-VAE checkpoint maps onto the decoder's names") {
  // Drive the loader's mapping rules over the REAL 1087-tensor manifest (names +
  // shapes, captured from the file's own header by range request; no payload).
  // The two mismatches this pins down are silent-failure shaped:
  //   * the file ships LEGACY weight_g/weight_v, the decoder reads
  //     parametrizations.weight.original0/1;
  //   * BigVGAN is under `decoder.`, but dec_in_proj.* is top level.
  auto mapped = [](const std::string& name) -> std::string {
    if (name.rfind("encoder.", 0) == 0) return "";  // audio ENCODER: not decoded
    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(8);
    if (key.size() >= 7 && key.compare(key.size() - 7, 7, ".filter") == 0) return "";
    const std::string g = ".weight_g", v = ".weight_v";
    if (key.size() > g.size() && key.compare(key.size() - g.size(), g.size(), g) == 0) {
      return key.substr(0, key.size() - g.size()) + ".parametrizations.weight.original0";
    }
    if (key.size() > v.size() && key.compare(key.size() - v.size(), v.size(), v) == 0) {
      return key.substr(0, key.size() - v.size()) + ".parametrizations.weight.original1";
    }
    return key;
  };

  std::set<std::string> decoder_names;
  int64_t skipped_filters = 0, skipped_encoder = 0, weight_norm_pairs = 0;
  for (int64_t i = 0; i < vllm_test::kH3AudioVaeTensorCount; ++i) {
    const std::string name = vllm_test::kH3AudioVaeTensors[i].name;
    if (name.rfind("encoder.", 0) == 0) {
      ++skipped_encoder;
      continue;
    }
    const std::string key = mapped(name);
    if (key.empty()) {
      ++skipped_filters;
      continue;
    }
    CHECK(decoder_names.insert(key).second);  // the mapping must be INJECTIVE
    if (key.find(".parametrizations.weight.original") != std::string::npos) ++weight_norm_pairs;
  }

  // The file really does carry an encoder and really does use the legacy spelling
  // — if either stopped being true these counts would move, and the loader's
  // renaming would silently become dead code.
  CHECK(skipped_encoder > 0);
  CHECK(skipped_filters > 0);
  CHECK(weight_norm_pairs > 0);

  // The decoder's own entry points must be present under the MAPPED names.
  CHECK(decoder_names.count("dec_in_proj.weight") == 1);
  CHECK(decoder_names.count("dec_in_proj.bias") == 1);
  CHECK(decoder_names.count("conv_pre.parametrizations.weight.original0") == 1);
  CHECK(decoder_names.count("conv_pre.parametrizations.weight.original1") == 1);
  CHECK(decoder_names.count("conv_pre.bias") == 1);
  CHECK(decoder_names.count("conv_post.parametrizations.weight.original0") == 1);
  CHECK(decoder_names.count("conv_post.parametrizations.weight.original1") == 1);
  CHECK(decoder_names.count("activation_post.act.alpha") == 1);
  CHECK(decoder_names.count("activation_post.act.beta") == 1);
  CHECK(decoder_names.count("ups.0.0.parametrizations.weight.original0") == 1);
  CHECK(decoder_names.count("resblocks.0.convs1.0.parametrizations.weight.original0") == 1);
  CHECK(decoder_names.count("resblocks.0.convs2.0.parametrizations.weight.original1") == 1);
  CHECK(decoder_names.count("resblocks.0.activations.0.act.alpha") == 1);
  // Nothing may keep the legacy spelling after mapping.
  for (const std::string& n : decoder_names) {
    CHECK(n.find(".weight_g") == std::string::npos);
    CHECK(n.find(".weight_v") == std::string::npos);
    CHECK(n.rfind("decoder.", 0) != 0);
  }

  // The SHIPPED geometry, recovered from the manifest's shapes alone, must equal
  // the config the decoder was gated with.
  auto shape_of = [](const std::string& want) {
    for (int64_t i = 0; i < vllm_test::kH3AudioVaeTensorCount; ++i) {
      if (want == vllm_test::kH3AudioVaeTensors[i].name) return vllm_test::kH3AudioVaeTensors[i];
    }
    FAIL("manifest is missing ", want);
    return vllm_test::kH3AudioVaeTensors[0];
  };
  // dec_in_proj: Conv1d(vae_latent_channels -> num_mels, k=1).
  const auto dip = shape_of("dec_in_proj.weight");
  CHECK(dip.rank == 3);
  CHECK(dip.shape[0] == 2048);  // num_mels
  CHECK(dip.shape[1] == 32);    // vae latent channels
  CHECK(dip.shape[2] == 1);     // k = 1
  // conv_pre: num_mels -> upsample_initial_channel, k=7.
  const auto pre = shape_of("decoder.conv_pre.weight_v");
  CHECK(pre.shape[0] == 1024);  // upsample_initial_channel
  CHECK(pre.shape[1] == 2048);  // num_mels
  CHECK(pre.shape[2] == 7);
  // conv_post: 7 upsamples halve 1024 down to 8, then -> 1 channel, k=7.
  const auto post = shape_of("decoder.conv_post.weight_v");
  CHECK(post.shape[0] == 1);
  CHECK(post.shape[1] == 8);
  CHECK(post.shape[2] == 7);
}

TEST_CASE("minimax_h3: the audio-VAE loader materializes weights the decoder RUNS") {
  // The manifest test above gates the NAME mapping against the real checkpoint.
  // This gates the loader end to end: a real safetensors file, written with the
  // SHIPPED spellings (decoder. prefix, legacy weight_g/weight_v, kaiser-sinc
  // filters present), must produce weights the decoder actually decodes with.
  struct Entry {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::string bytes;
  };
  std::vector<Entry> entries;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    const std::vector<float> values = MakeParam("stvae." + name, numel, 0.05);
    entries.push_back({name, "F32", shape,
                       std::string(reinterpret_cast<const char*>(values.data()),
                                   values.size() * sizeof(float))});
  };
  // A conv, in the checkpoint's LEGACY weight-norm spelling.
  auto add_conv = [&](const std::string& prefix, int64_t out_ch, int64_t in_ch, int64_t k,
                      bool bias) {
    add(prefix + ".weight_g", {out_ch, 1, 1});
    add(prefix + ".weight_v", {out_ch, in_ch, k});
    if (bias) add(prefix + ".bias", {out_ch});
  };

  const int64_t mels = vllm_test::kH3AudioVaeNumMels;
  const int64_t init_ch = vllm_test::kH3AudioVaeInitialChannel;
  add_conv("decoder.conv_pre", init_ch, mels, 7, true);
  // Two upsample stages (rates 2,2 / kernels 4,4), matching the golden config.
  // ConvTranspose1d stores its weight TRANSPOSED — [in, out, k], with the
  // weight-norm magnitude sized by INPUT channels — which the real checkpoint
  // confirms (decoder.ups.0.0.weight_v is [1024, 512, 9], bias [512]).
  auto add_deconv = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch, int64_t k) {
    add(prefix + ".weight_g", {in_ch, 1, 1});
    add(prefix + ".weight_v", {in_ch, out_ch, k});
    add(prefix + ".bias", {out_ch});
  };
  add_deconv("decoder.ups.0.0", init_ch, init_ch / 2, 4);
  add_deconv("decoder.ups.1.0", init_ch / 2, init_ch / 4, 4);
  // 3 resblock kernels x 3 dilations, per upsample stage.
  const std::vector<int64_t> ks = {3, 7, 11};
  for (int64_t u = 0; u < 2; ++u) {
    const int64_t ch = u == 0 ? init_ch / 2 : init_ch / 4;
    for (size_t kidx = 0; kidx < ks.size(); ++kidx) {
      const std::string rb = "decoder.resblocks." + std::to_string(u * 3 + kidx);
      for (int64_t c = 0; c < 3; ++c) {
        add_conv(rb + ".convs1." + std::to_string(c), ch, ch, ks[kidx], true);
        add_conv(rb + ".convs2." + std::to_string(c), ch, ch, ks[kidx], true);
      }
      for (int64_t a = 0; a < 6; ++a) {
        const std::string act = rb + ".activations." + std::to_string(a);
        add(act + ".act.alpha", {ch});
        add(act + ".act.beta", {ch});
        // Present in the file, and the loader must SKIP them (computed at load).
        add(act + ".upsample.filter", {1, 1, 12});
        add(act + ".downsample.lowpass.filter", {1, 1, 12});
      }
    }
  }
  add("decoder.activation_post.act.alpha", {init_ch / 4});
  add("decoder.activation_post.act.beta", {init_ch / 4});
  add("decoder.activation_post.upsample.filter", {1, 1, 12});
  add_conv("decoder.conv_post", 1, init_ch / 4, 7, false);
  // The audio ENCODER shares the file and must be ignored.
  add("encoder.block.0.alpha", {8});

  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const Entry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  const std::string path = "/tmp/minimax_h3_audio_vae_loader.safetensors";
  {
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    for (const Entry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
    std::fclose(fh);
  }

  const vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(path);
  const vllm::MiniMaxH3AudioVaeWeights w = vllm::LoadMiniMaxH3AudioVaeWeights(st);

  // Renamed, de-prefixed, filters and encoder dropped.
  CHECK(w.Has("conv_pre.parametrizations.weight.original0"));
  CHECK(w.Has("conv_pre.parametrizations.weight.original1"));
  CHECK(w.Has("conv_pre.bias"));
  CHECK(w.Has("resblocks.0.activations.0.act.alpha"));
  CHECK_FALSE(w.Has("decoder.conv_pre.weight_g"));   // legacy spelling gone
  CHECK_FALSE(w.Has("conv_pre.weight_g"));
  CHECK_FALSE(w.Has("resblocks.0.activations.0.upsample.filter"));  // computed
  CHECK_FALSE(w.Has("encoder.block.0.alpha"));                      // not decoded

  // And it DECODES: the loaded weights drive the real decoder to a finite,
  // in-range waveform. This is the step a name-only check cannot reach.
  vllm::MiniMaxH3AudioVaeConfig cfg;
  cfg.num_mels = mels;
  cfg.upsample_initial_channel = init_ch;
  cfg.upsample_rates = {2, 2};
  cfg.upsample_kernel_sizes = {4, 4};
  cfg.resblock_kernel_sizes = ks;
  cfg.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  const int64_t frames = vllm_test::kH3AudioVaeFrames;
  const std::vector<float> latent = MakeParam("stvae.input", mels * frames, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::MiniMaxH3AudioVaeDecode(cfg, w, latent, frames, &out_samples);
  CHECK(out_samples == frames * 4);  // two stride-2 upsamples
  REQUIRE(wave.size() == static_cast<size_t>(out_samples));
  for (float v : wave) {
    CHECK(std::isfinite(v));
    CHECK(v >= -1.0f);
    CHECK(v <= 1.0f);
  }
}

TEST_CASE("minimax_h3: the REAL audio-VAE checkpoint maps onto the ENCODER's names") {
  // The mirror of the decoder's manifest test, for the half that had no loader.
  // Same file, same 1087 tensors, opposite selection: `encoder.*` (de-prefixed),
  // plus the TOP-LEVEL `pre_block.*` and `mean_proj.*`. `logs_proj.*` must be left
  // behind -- loading it would imply conditioning samples the distribution.
  auto mapped = [](const std::string& name) -> std::string {
    const bool is_encoder = name.rfind("encoder.", 0) == 0;
    if (!is_encoder && name.rfind("pre_block.", 0) != 0 && name.rfind("mean_proj.", 0) != 0) {
      return "";
    }
    std::string key = is_encoder ? name.substr(std::strlen("encoder.")) : name;
    const std::string g = ".weight_g", v = ".weight_v";
    if (key.size() > g.size() && key.compare(key.size() - g.size(), g.size(), g) == 0) {
      return key.substr(0, key.size() - g.size()) + ".parametrizations.weight.original0";
    }
    if (key.size() > v.size() && key.compare(key.size() - v.size(), v.size(), v) == 0) {
      return key.substr(0, key.size() - v.size()) + ".parametrizations.weight.original1";
    }
    return key;
  };

  std::set<std::string> encoder_names;
  int64_t weight_norm_pairs = 0, skipped_decoder = 0, skipped_logs = 0;
  for (int64_t i = 0; i < vllm_test::kH3AudioVaeTensorCount; ++i) {
    const std::string name = vllm_test::kH3AudioVaeTensors[i].name;
    if (name.rfind("logs_proj.", 0) == 0) {
      ++skipped_logs;
      continue;
    }
    const std::string key = mapped(name);
    if (key.empty()) {
      ++skipped_decoder;
      continue;
    }
    CHECK(encoder_names.insert(key).second);  // the mapping must be INJECTIVE
    if (key.find(".parametrizations.weight.original") != std::string::npos) ++weight_norm_pairs;
  }
  CHECK(skipped_decoder > 0);  // the decoder really does share the file
  CHECK(skipped_logs == 2);    // logs_proj.weight + logs_proj.bias, both left behind
  CHECK(weight_norm_pairs > 0);

  // The encoder's own entry points, under the MAPPED names the forward reads.
  CHECK(encoder_names.count("block.0.parametrizations.weight.original0") == 1);
  CHECK(encoder_names.count("block.0.parametrizations.weight.original1") == 1);
  CHECK(encoder_names.count("block.0.bias") == 1);
  CHECK(encoder_names.count("block.1.block.0.block.0.alpha") == 1);      // ResidualUnit Snake1d
  CHECK(encoder_names.count("block.1.block.0.block.1.parametrizations.weight.original1") == 1);
  CHECK(encoder_names.count("block.1.block.4.parametrizations.weight.original1") == 1);  // stride
  CHECK(encoder_names.count("mean_proj.weight") == 1);
  CHECK(encoder_names.count("pre_block.attn.qkv.weight") == 1);
  CHECK(encoder_names.count("pre_block.attn.zero_k_bias") == 1);
  CHECK(encoder_names.count("pre_block.mlp.w2.weight") == 1);
  for (const std::string& n : encoder_names) {
    CHECK(n.find(".weight_g") == std::string::npos);
    CHECK(n.find(".weight_v") == std::string::npos);
    CHECK(n.rfind("encoder.", 0) != 0);
    CHECK(n.rfind("logs_proj", 0) != 0);
  }

  // The SHIPPED geometry, recovered from the manifest's shapes alone, must equal
  // the config MiniMaxH3AudioVaeEncoderConfig defaults to. This is the check that
  // catches a rates list or a latent width that drifted from the checkpoint.
  auto shape_of = [](const std::string& want) {
    for (int64_t i = 0; i < vllm_test::kH3AudioVaeTensorCount; ++i) {
      if (want == vllm_test::kH3AudioVaeTensors[i].name) return vllm_test::kH3AudioVaeTensors[i];
    }
    FAIL("manifest is missing ", want);
    return vllm_test::kH3AudioVaeTensors[0];
  };
  const vllm::MiniMaxH3AudioVaeEncoderConfig cfg;  // the SHIPPED defaults
  // block.0: WNConv1d(1, encoder_dim, k=7).
  const auto first = shape_of("encoder.block.0.weight_v");
  CHECK(first.shape[0] == cfg.encoder_dim);
  CHECK(first.shape[1] == 1);  // a MONO waveform in
  CHECK(first.shape[2] == 7);
  // Each EncoderBlock's strided conv doubles the channels; its kernel is 2*stride,
  // which is how the manifest pins down `encoder_rates` without a config file.
  int64_t channels = cfg.encoder_dim;
  for (size_t i = 0; i < cfg.encoder_rates.size(); ++i) {
    const auto down =
        shape_of("encoder.block." + std::to_string(i + 1) + ".block.4.weight_v");
    CHECK(down.shape[0] == channels * 2);
    CHECK(down.shape[1] == channels);
    CHECK(down.shape[2] == 2 * cfg.encoder_rates[i]);
    channels *= 2;
  }
  CHECK(channels == cfg.latent_dim);  // 64 * 2^5 == 2048
  // The final conv lands on d_latent with k=3, and the manifest's index proves the
  // block count: 1 conv + 5 EncoderBlocks + 1 Snake1d + 1 conv.
  const auto tail = shape_of("encoder.block." + std::to_string(cfg.encoder_rates.size() + 2) +
                             ".weight_v");
  CHECK(tail.shape[0] == cfg.latent_dim);
  CHECK(tail.shape[1] == channels);
  CHECK(tail.shape[2] == 3);
  // pre_block: AttnProjection(latent_dim -> attn_proj_dim), qkv 3x the INPUT width
  // (the narrowing branch, dac_attn_proj.py:31-37).
  const auto qkv = shape_of("pre_block.attn.qkv.weight");
  CHECK(qkv.shape[0] == 3 * cfg.latent_dim);
  CHECK(qkv.shape[1] == cfg.latent_dim);
  const auto pproj = shape_of("pre_block.proj.weight");
  CHECK(pproj.shape[0] == cfg.attn_proj_dim());
  CHECK(pproj.shape[1] == cfg.latent_dim);
  // mean_proj: Conv1d(attn_proj_dim -> vae_latent_channels, k=1).
  const auto mp = shape_of("mean_proj.weight");
  CHECK(mp.shape[0] == cfg.vae_latent_channels);
  CHECK(mp.shape[1] == cfg.attn_proj_dim());
  CHECK(mp.shape[2] == 1);
}

TEST_CASE("minimax_h3: the audio-VAE ENCODER loader materializes weights that RUN") {
  // End to end for the encoder half: a real safetensors file written with the
  // SHIPPED spellings (encoder. prefix, legacy weight_g/weight_v, a decoder and a
  // logs_proj sharing the file) must load into weights the encoder actually
  // encodes with. Names alone cannot reach that.
  const vllm::MiniMaxH3AudioVaeEncoderConfig cfg = ReducedAudioEncoderConfig();
  struct Entry {
    std::string name;
    std::vector<int64_t> shape;
    std::string bytes;
  };
  std::vector<Entry> entries;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape, double scale,
                 double offset) {
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    const std::vector<float> values = MakeParam("stenc." + name, numel, scale, offset);
    entries.push_back({name, shape,
                       std::string(reinterpret_cast<const char*>(values.data()),
                                   values.size() * sizeof(float))});
  };
  // A weight-normalized conv, in the checkpoint's LEGACY spelling.
  auto add_conv = [&](const std::string& prefix, int64_t out_ch, int64_t in_ch, int64_t k) {
    add(prefix + ".weight_g", {out_ch, 1, 1}, 0.03, 0.15);
    add(prefix + ".weight_v", {out_ch, in_ch, k}, 0.08, 0.0);
    add(prefix + ".bias", {out_ch}, 0.05, 0.0);
  };
  auto add_alpha = [&](const std::string& name, int64_t ch) {
    add(name, {1, ch, 1}, 0.2, 1.0);
  };

  add_conv("encoder.block.0", cfg.encoder_dim, 1, 7);
  int64_t channels = cfg.encoder_dim;
  for (size_t i = 0; i < cfg.encoder_rates.size(); ++i) {
    channels *= 2;
    const int64_t half = channels / 2;
    const std::string blk = "encoder.block." + std::to_string(i + 1);
    for (int64_t u = 0; u < 3; ++u) {
      const std::string ru = blk + ".block." + std::to_string(u);
      add_alpha(ru + ".block.0.alpha", half);
      add_conv(ru + ".block.1", half, half, 7);
      add_alpha(ru + ".block.2.alpha", half);
      add_conv(ru + ".block.3", half, half, 1);
    }
    add_alpha(blk + ".block.3.alpha", half);
    add_conv(blk + ".block.4", channels, half, 2 * cfg.encoder_rates[i]);
  }
  const size_t rates = cfg.encoder_rates.size();
  add_alpha("encoder.block." + std::to_string(rates + 1) + ".alpha", channels);
  add_conv("encoder.block." + std::to_string(rates + 2), cfg.latent_dim, channels, 3);

  const int64_t in_dim = cfg.latent_dim, out_dim = cfg.attn_proj_dim();
  add("pre_block.norm1.weight", {in_dim}, 0.1, 1.0);
  add("pre_block.norm1.bias", {in_dim}, 0.05, 0.0);
  add("pre_block.norm3.weight", {in_dim}, 0.1, 1.0);
  add("pre_block.norm3.bias", {in_dim}, 0.05, 0.0);
  add("pre_block.norm2.weight", {out_dim}, 0.1, 1.0);
  add("pre_block.norm2.bias", {out_dim}, 0.05, 0.0);
  add("pre_block.attn.q_bias", {in_dim}, 0.05, 0.0);
  add("pre_block.attn.v_bias", {in_dim}, 0.05, 0.0);
  add("pre_block.attn.zero_k_bias", {in_dim}, 0.0, 0.0);
  add("pre_block.attn.qkv.weight", {3 * in_dim, in_dim}, 0.1, 0.0);
  add("pre_block.attn.proj.weight", {out_dim, out_dim}, 0.1, 0.0);
  add("pre_block.attn.proj.bias", {out_dim}, 0.05, 0.0);
  add("pre_block.proj.weight", {out_dim, in_dim}, 0.1, 0.0);
  add("pre_block.proj.bias", {out_dim}, 0.05, 0.0);
  add("pre_block.mlp.norm.weight", {out_dim}, 0.1, 1.0);
  add("pre_block.mlp.norm.bias", {out_dim}, 0.05, 0.0);
  const int64_t hidden = 2 * out_dim;
  add("pre_block.mlp.w0.weight", {hidden, out_dim}, 0.1, 0.0);
  add("pre_block.mlp.w0.bias", {hidden}, 0.05, 0.0);
  add("pre_block.mlp.w1.weight", {hidden, out_dim}, 0.1, 0.0);
  add("pre_block.mlp.w1.bias", {hidden}, 0.05, 0.0);
  add("pre_block.mlp.w2.weight", {out_dim, hidden}, 0.1, 0.0);
  add("pre_block.mlp.w2.bias", {out_dim}, 0.05, 0.0);
  add("mean_proj.weight", {cfg.vae_latent_channels, out_dim, 1}, 0.1, 0.0);
  add("mean_proj.bias", {cfg.vae_latent_channels}, 0.05, 0.0);
  // The DECODER and the log-variance head share the file. Both must be left behind.
  add("decoder.conv_pre.weight_g", {8, 1, 1}, 0.1, 0.0);
  add("decoder.conv_pre.weight_v", {8, 4, 7}, 0.1, 0.0);
  add("dec_in_proj.weight", {4, 4, 1}, 0.1, 0.0);
  add("logs_proj.weight", {cfg.vae_latent_channels, out_dim, 1}, 0.1, 0.0);
  add("logs_proj.bias", {cfg.vae_latent_channels}, 0.05, 0.0);

  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const Entry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  const std::string path = "/tmp/minimax_h3_audio_vae_encoder_loader.safetensors";
  {
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    for (const Entry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
    std::fclose(fh);
  }

  const vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(path);
  const vllm::MiniMaxH3AudioVaeWeights w = vllm::LoadMiniMaxH3AudioVaeEncoderWeights(st);

  // Renamed, de-prefixed, and the decoder half dropped.
  CHECK(w.Has("block.0.parametrizations.weight.original0"));
  CHECK(w.Has("block.0.parametrizations.weight.original1"));
  CHECK(w.Has("block.1.block.0.block.0.alpha"));
  CHECK(w.Has("pre_block.attn.qkv.weight"));   // top level: NOT de-prefixed
  CHECK(w.Has("mean_proj.weight"));
  CHECK_FALSE(w.Has("encoder.block.0.weight_g"));  // legacy spelling gone
  CHECK_FALSE(w.Has("block.0.weight_g"));
  CHECK_FALSE(w.Has("conv_pre.parametrizations.weight.original0"));  // decoder half
  CHECK_FALSE(w.Has("dec_in_proj.weight"));
  // The log-variance head is refused on purpose: a reference must be the MEAN.
  CHECK_FALSE(w.Has("logs_proj.weight"));
  CHECK_FALSE(w.Has("logs_proj.bias"));
  // A PLAIN Linear's `.weight` must NOT be mistaken for a materialized weight-norm.
  CHECK_FALSE(w.Has("pre_block.attn.qkv.parametrizations.weight.original0"));
  CHECK_FALSE(w.Has("mean_proj.parametrizations.weight.original0"));

  // And it ENCODES: the loaded weights drive the real encoder to finite rows of
  // the shape a reference block claims.
  int64_t audio_t = 0;
  const int64_t samples = vllm_test::kH3AudioEncSamples;
  std::vector<float> stereo = MakeParam("stenc.wave", samples, 1.0);
  const std::vector<float> copy = stereo;
  stereo.insert(stereo.end(), copy.begin(), copy.end());
  const std::vector<float> rows =
      vllm::MiniMaxH3AudioVaeEncodeToRows(cfg, w, stereo, 2, samples, {}, {}, &audio_t);
  CHECK(audio_t == vllm_test::kH3AudioEncFrames);
  REQUIRE(rows.size() == static_cast<size_t>(2 * audio_t * cfg.vae_latent_channels));
  for (float v : rows) CHECK(std::isfinite(v));

  // A MATERIALIZED weight-norm (the third spelling) must load too: rewriting one
  // conv as a plain `.weight` has to reconstruct an exact (g, v) pair, so the rows
  // come out byte-identical.
  std::vector<Entry> folded;
  for (const Entry& e : entries) {
    if (e.name == "encoder.block.0.weight_g") continue;
    if (e.name == "encoder.block.0.weight_v") {
      // w = g * v / ||v||, folded at "conversion time".
      std::vector<float> v(e.bytes.size() / sizeof(float));
      std::memcpy(v.data(), e.bytes.data(), e.bytes.size());
      const std::vector<float> g = MakeParam("stenc.encoder.block.0.weight_g", cfg.encoder_dim,
                                             0.03, 0.15);
      const std::vector<float> materialized =
          vllm::MiniMaxH3MaterializeWeightNorm(g, v, cfg.encoder_dim);
      folded.push_back({"encoder.block.0.weight", e.shape,
                        std::string(reinterpret_cast<const char*>(materialized.data()),
                                    materialized.size() * sizeof(float))});
      continue;
    }
    folded.push_back(e);
  }
  header = "{";
  offset = 0;
  first = true;
  for (const Entry& e : folded) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  const std::string folded_path = "/tmp/minimax_h3_audio_vae_encoder_folded.safetensors";
  {
    FILE* fh = std::fopen(folded_path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    for (const Entry& e : folded) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
    std::fclose(fh);
  }
  const vllm::SafetensorsFile fst = vllm::SafetensorsFile::Open(folded_path);
  const vllm::MiniMaxH3AudioVaeWeights fw = vllm::LoadMiniMaxH3AudioVaeEncoderWeights(fst);
  CHECK(fw.Has("block.0.parametrizations.weight.original0"));
  int64_t ft = 0;
  const std::vector<float> frows =
      vllm::MiniMaxH3AudioVaeEncodeToRows(cfg, fw, stereo, 2, samples, {}, {}, &ft);
  CHECK(ft == audio_t);
  REQUIRE(frows.size() == rows.size());
  const double fold_err = MaxAbsDiff(frows, rows.data(), rows.size());
  INFO("materialized-weight-norm encode max|diff| = " << fold_err);
  CHECK(fold_err <= 1e-6);
}

TEST_CASE("minimax_h3: the audio-VAE loader accepts a MATERIALIZED weight-norm too") {
  // A third spelling exists in the wild: repackaged community VAE bundles fold the
  // weight-norm at conversion time and ship a plain `<conv>.weight`, with no (g, v)
  // pair at all. The decoder only knows the pair, so the loader reconstructs one --
  // and the reconstruction must be EXACT, not approximate.
  struct Entry {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::string bytes;
  };
  std::vector<Entry> entries;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape,
                 const std::vector<float>& values) {
    entries.push_back({name, "F32", shape,
                       std::string(reinterpret_cast<const char*>(values.data()),
                                   values.size() * sizeof(float))});
  };
  // One materialized conv weight, plus a plain dec_in_proj (NOT weight-normalized,
  // so it must be left exactly alone).
  const std::vector<float> w = MakeParam("matnorm.w", 4 * 3 * 5, 0.3);
  add("decoder.conv_pre.weight", {4, 3, 5}, w);
  add("decoder.conv_pre.bias", {4}, MakeParam("matnorm.b", 4, 0.1));
  const std::vector<float> dip = MakeParam("matnorm.dip", 6 * 2 * 1, 0.2);
  add("dec_in_proj.weight", {6, 2, 1}, dip);

  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const Entry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  const std::string path = "/tmp/minimax_h3_audio_vae_materialized.safetensors";
  {
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    for (const Entry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
    std::fclose(fh);
  }

  const vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(path);
  const vllm::MiniMaxH3AudioVaeWeights got = vllm::LoadMiniMaxH3AudioVaeWeights(st);

  // The pair exists, and the plain name is gone.
  REQUIRE(got.Has("conv_pre.parametrizations.weight.original0"));
  REQUIRE(got.Has("conv_pre.parametrizations.weight.original1"));
  CHECK_FALSE(got.Has("conv_pre.weight"));
  // dec_in_proj is NOT weight-normalized: it must survive verbatim.
  REQUIRE(got.Has("dec_in_proj.weight"));
  CHECK_FALSE(got.Has("dec_in_proj.parametrizations.weight.original0"));
  const std::vector<float>& dip_got = got.Get("dec_in_proj.weight");
  REQUIRE(dip_got.size() == dip.size());
  for (size_t i = 0; i < dip.size(); ++i) CHECK(dip_got[i] == dip[i]);

  // THE POINT: running the decoder's own materialization on the reconstructed pair
  // must return the checkpoint's weight. Anything else is a silently wrong conv.
  const std::vector<float> back = vllm::MiniMaxH3MaterializeWeightNorm(
      got.Get("conv_pre.parametrizations.weight.original0"),
      got.Get("conv_pre.parametrizations.weight.original1"), 4);
  REQUIRE(back.size() == w.size());
  double err = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    err = std::max(err, std::abs(static_cast<double>(back[i] - w[i])));
  }
  INFO("materialized round-trip err = " << err);
  CHECK(err <= 1e-6);
}

TEST_CASE("minimax_h3: the REAL video-VAE checkpoint maps onto the ViT3D decoder's names") {
  // Same shape of gate as the audio VAE, over the real 560-tensor manifest. The
  // mapping is simpler here (no weight-norm spelling change; only the `decoder.`
  // prefix), which is itself worth pinning: if a future checkpoint revision moved
  // to the parametrization spelling, this test fails rather than the decode.
  auto mapped = [](const std::string& name) -> std::string {
    if (name.rfind("encoder.", 0) == 0) return "";
    if (name.rfind("quant_conv.", 0) == 0) return "";  // encoder-side stage
    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(8);
    return key;
  };

  std::set<std::string> names;
  int64_t skipped = 0;
  for (int64_t i = 0; i < vllm_test::kH3VideoVaeTensorCount; ++i) {
    const std::string key = mapped(vllm_test::kH3VideoVaeTensors[i].name);
    if (key.empty()) {
      ++skipped;
      continue;
    }
    CHECK(names.insert(key).second);  // INJECTIVE
  }
  CHECK(skipped > 0);  // the encoder half really is present in the file

  // Every name the ViT3D decoder reads must exist post-mapping.
  CHECK(names.count("x_embedder.weight") == 1);
  CHECK(names.count("x_embedder.bias") == 1);
  CHECK(names.count("register_tokens") == 1);
  CHECK(names.count("norm_out.weight") == 1);
  CHECK(names.count("proj_out.weight") == 1);
  CHECK(names.count("transformer_blocks.0.attn.to_qkv.weight") == 1);
  CHECK(names.count("transformer_blocks.0.attn.to_out.bias") == 1);
  CHECK(names.count("transformer_blocks.0.ff.w1.weight") == 1);
  CHECK(names.count("transformer_blocks.0.ff.w2.bias") == 1);
  CHECK(names.count("transformer_blocks.0.norm1.weight") == 1);
  CHECK(names.count("transformer_blocks.0.scale1") == 1);
  CHECK(names.count("transformer_blocks.35.scale2") == 1);  // 36 blocks, 0..35
  CHECK(names.count("transformer_blocks.36.scale2") == 0);
  // post_quant_conv is kept (it is a REAL step, see below), quant_conv is not.
  CHECK(names.count("post_quant_conv.weight") == 1);
  CHECK(names.count("quant_conv.weight") == 0);
  for (const std::string& n : names) CHECK(n.rfind("decoder.", 0) != 0);
}

TEST_CASE("minimax_h3: post_quant_conv is a real step, and it is a 1x1x1 channel mix") {
  // This tensor sits OUTSIDE ViT3DDecoder, so the decoder gate (8.9e-8 against the
  // checkpoint's own decoder, whose first op is x_embedder) never covered it and
  // nothing in this port applied it. Loading it without applying it would be the
  // worst outcome: a decode that runs, looks plausible, and is wrong.
  //
  // The real shape says exactly what it is.
  auto shape_of = [](const std::string& want) {
    for (int64_t i = 0; i < vllm_test::kH3VideoVaeTensorCount; ++i) {
      if (want == vllm_test::kH3VideoVaeTensors[i].name) return vllm_test::kH3VideoVaeTensors[i];
    }
    FAIL("manifest is missing ", want);
    return vllm_test::kH3VideoVaeTensors[0];
  };
  const auto pqc = shape_of("post_quant_conv.weight");
  CHECK(pqc.rank == 5);
  CHECK(pqc.shape[0] == 24);  // latent channels out
  CHECK(pqc.shape[1] == 24);  // latent channels in
  CHECK(pqc.shape[2] == 1);   // 1x1x1 -> a per-position CHANNEL MIX
  CHECK(pqc.shape[3] == 1);
  CHECK(pqc.shape[4] == 1);
  // ...and it feeds x_embedder, which takes those same 24 channels.
  const auto xe = shape_of("decoder.x_embedder.weight");
  CHECK(xe.shape[1] == pqc.shape[0]);

  // The implementation, checked against a hand-computed contraction.
  const int64_t C = 3, P = 4;
  vllm::MiniMaxH3AudioVaeWeights w;
  w.tensors["post_quant_conv.weight"] = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // [C, C]
  w.tensors["post_quant_conv.bias"] = {0.5f, -0.5f, 1.0f};
  std::vector<float> latent(static_cast<size_t>(C * P));
  for (int64_t i = 0; i < C * P; ++i) latent[static_cast<size_t>(i)] = float(i + 1);
  const std::vector<float> got = vllm::MiniMaxH3VideoVaePostQuantConv(w, latent, C, P);
  REQUIRE(got.size() == latent.size());
  for (int64_t o = 0; o < C; ++o) {
    for (int64_t p = 0; p < P; ++p) {
      double want = w.tensors["post_quant_conv.bias"][static_cast<size_t>(o)];
      for (int64_t i = 0; i < C; ++i) {
        want += w.tensors["post_quant_conv.weight"][static_cast<size_t>(o * C + i)] *
                latent[static_cast<size_t>(i * P + p)];
      }
      CHECK(got[static_cast<size_t>(o * P + p)] == doctest::Approx(want).epsilon(1e-6));
    }
  }
  // A channel MIX, not a passthrough: dropping it would change the decoder's input.
  CHECK(got[0] != doctest::Approx(latent[0]));
}

TEST_CASE("minimax_h3: the encoder loader FUSES q/k/v and gate/up across shards") {
  // The one loader that transforms rather than renames. HF ships q/k/v and gate/up
  // SEPARATE (confirmed from the real FL2VA/text_encoder index: 1058 tensors, 64
  // layers of model.language_model.layers.N.self_attn.{q,k,v}_proj); the port, like
  // vLLM, consumes them FUSED. Row-concatenation ORDER is load-bearing: the forward
  // slices qkv_proj at [0,q) / [q,q+kv) / [q+kv,...), so any other order silently
  // feeds keys into the query path — which would still run, and still be wrong.
  struct Entry {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
  };
  const int64_t H = 8, QD = 16, KVD = 4, FF = 12;
  auto make = [&](const std::string& name, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return Entry{name, shape, MakeParam("enc." + name, n, 0.2)};
  };

  // Two shards, with one layer's tensors SPLIT across them — the case a
  // single-shard test would miss entirely.
  std::vector<std::vector<Entry>> shards(2);
  const std::string lm = "model.language_model.layers.0.";
  shards[0].push_back(make(lm + "input_layernorm.weight", {H}));
  shards[0].push_back(make(lm + "post_attention_layernorm.weight", {H}));
  shards[0].push_back(make(lm + "self_attn.q_proj.weight", {QD, H}));
  shards[0].push_back(make(lm + "self_attn.q_norm.weight", {4}));
  shards[0].push_back(make(lm + "self_attn.k_norm.weight", {4}));
  shards[1].push_back(make(lm + "self_attn.k_proj.weight", {KVD, H}));
  shards[1].push_back(make(lm + "self_attn.v_proj.weight", {KVD, H}));
  shards[1].push_back(make(lm + "self_attn.o_proj.weight", {H, QD}));
  shards[1].push_back(make(lm + "mlp.gate_proj.weight", {FF, H}));
  shards[1].push_back(make(lm + "mlp.up_proj.weight", {FF, H}));
  shards[1].push_back(make(lm + "mlp.down_proj.weight", {H, FF}));
  // A second layer, so truncation has something to cut.
  const std::string lm1 = "model.language_model.layers.1.";
  shards[1].push_back(make(lm1 + "input_layernorm.weight", {H}));
  shards[1].push_back(make(lm1 + "post_attention_layernorm.weight", {H}));
  shards[1].push_back(make(lm1 + "self_attn.q_proj.weight", {QD, H}));
  shards[1].push_back(make(lm1 + "self_attn.k_proj.weight", {KVD, H}));
  shards[1].push_back(make(lm1 + "self_attn.v_proj.weight", {KVD, H}));
  shards[1].push_back(make(lm1 + "self_attn.q_norm.weight", {4}));
  shards[1].push_back(make(lm1 + "self_attn.k_norm.weight", {4}));
  shards[1].push_back(make(lm1 + "self_attn.o_proj.weight", {H, QD}));
  shards[1].push_back(make(lm1 + "mlp.gate_proj.weight", {FF, H}));
  shards[1].push_back(make(lm1 + "mlp.up_proj.weight", {FF, H}));
  shards[1].push_back(make(lm1 + "mlp.down_proj.weight", {H, FF}));
  // Vision tower: already fused upstream, so it must pass through untouched.
  shards[0].push_back(make("model.visual.blocks.0.attn.qkv.weight", {3 * H, H}));
  shards[0].push_back(make("model.visual.patch_embed.proj.bias", {H}));
  shards[0].push_back(make("model.visual.merger.norm.weight", {H}));
  // Must NOT be loaded: H3 reads the UNNORMALIZED output, and lm_head is unused.
  shards[1].push_back(make("model.language_model.norm.weight", {H}));
  shards[1].push_back(make("lm_head.weight", {4, H}));

  std::vector<std::string> paths;
  for (size_t s = 0; s < shards.size(); ++s) {
    std::string header = "{";
    size_t offset = 0;
    bool first = true;
    for (const Entry& e : shards[s]) {
      if (!first) header += ",";
      first = false;
      header += "\"" + e.name + "\":{\"dtype\":\"F32\",\"shape\":[";
      for (size_t i = 0; i < e.shape.size(); ++i) {
        if (i) header += ",";
        header += std::to_string(e.shape[i]);
      }
      const size_t nbytes = e.values.size() * sizeof(float);
      header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                std::to_string(offset + nbytes) + "]}";
      offset += nbytes;
    }
    header += "}";
    const std::string path = "/tmp/minimax_h3_enc_shard" + std::to_string(s) + ".safetensors";
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    for (const Entry& e : shards[s]) {
      std::fwrite(e.values.data(), 1, e.values.size() * sizeof(float), fh);
    }
    std::fclose(fh);
    paths.push_back(path);
  }

  std::vector<vllm::SafetensorsFile> files;
  for (const std::string& p : paths) files.push_back(vllm::SafetensorsFile::Open(p));
  const vllm::MiniMaxH3AudioVaeWeights w = vllm::LoadMiniMaxH3EncoderWeights(files);

  // The fusion is byte-exact and IN ORDER: [q_all | k_all | v_all].
  const std::vector<float>& qkv = w.Get("layers.0.self_attn.qkv_proj.weight");
  const std::vector<float> q = MakeParam("enc." + lm + "self_attn.q_proj.weight", QD * H, 0.2);
  const std::vector<float> k = MakeParam("enc." + lm + "self_attn.k_proj.weight", KVD * H, 0.2);
  const std::vector<float> v = MakeParam("enc." + lm + "self_attn.v_proj.weight", KVD * H, 0.2);
  REQUIRE(qkv.size() == q.size() + k.size() + v.size());
  for (size_t i = 0; i < q.size(); ++i) CHECK(qkv[i] == q[i]);
  for (size_t i = 0; i < k.size(); ++i) CHECK(qkv[q.size() + i] == k[i]);
  for (size_t i = 0; i < v.size(); ++i) CHECK(qkv[q.size() + k.size() + i] == v[i]);

  const std::vector<float>& gu = w.Get("layers.0.mlp.gate_up_proj.weight");
  const std::vector<float> gate = MakeParam("enc." + lm + "mlp.gate_proj.weight", FF * H, 0.2);
  const std::vector<float> up = MakeParam("enc." + lm + "mlp.up_proj.weight", FF * H, 0.2);
  REQUIRE(gu.size() == gate.size() + up.size());
  for (size_t i = 0; i < gate.size(); ++i) CHECK(gu[i] == gate[i]);
  for (size_t i = 0; i < up.size(); ++i) CHECK(gu[gate.size() + i] == up[i]);

  // The separate names must be GONE — leaving them would let a forward silently
  // read an unfused tensor.
  CHECK_FALSE(w.Has("layers.0.self_attn.q_proj.weight"));
  CHECK_FALSE(w.Has("layers.0.mlp.gate_proj.weight"));

  // Vision tower passes through, de-prefixed and unfused.
  CHECK(w.Has("blocks.0.attn.qkv.weight"));
  CHECK(w.Has("patch_embed.proj.bias"));
  CHECK(w.Has("merger.norm.weight"));

  // H3 deltas: NO final norm, no lm_head.
  CHECK_FALSE(w.Has("norm.weight"));
  CHECK_FALSE(w.Has("lm_head.weight"));

  // Truncation: H3 keeps min(num_hidden_layers, 50); the file ships more.
  CHECK(w.Has("layers.1.self_attn.qkv_proj.weight"));
  const vllm::MiniMaxH3AudioVaeWeights trunc =
      vllm::LoadMiniMaxH3EncoderWeights(files, /*max_layers=*/1);
  CHECK(trunc.Has("layers.0.self_attn.qkv_proj.weight"));
  CHECK_FALSE(trunc.Has("layers.1.self_attn.qkv_proj.weight"));
}

TEST_CASE("minimax_h3: the SHIPPED VAE config.json files parse into the decoders' configs") {
  // Assembly needs the real configs, not hand-built ones — including the per-channel
  // latents_mean/std the pipeline denormalizes with. Both files are embedded
  // verbatim, so this gates the parsers against what the checkpoint ACTUALLY says.
  const nlohmann::json audio_json = nlohmann::json::parse(vllm_test::kH3AudioVaeConfigJson);
  vllm::MiniMaxH3LatentStats audio_stats;
  const vllm::MiniMaxH3AudioVaeConfig audio =
      vllm::ParseMiniMaxH3AudioVaeConfig(audio_json, &audio_stats);

  // The subtlety: `latent_dim` (2048) is the MEL count BigVGAN consumes, while
  // `latent_channels` (32) is the VAE latent width that dec_in_proj maps FROM.
  // Reading the wrong one gives a decoder that is wrong by a factor of 64.
  CHECK(audio.num_mels == 2048);
  CHECK(audio.upsample_initial_channel == 1024);
  CHECK(audio.upsample_rates == std::vector<int64_t>{5, 5, 2, 2, 2, 2, 2});
  CHECK(audio.upsample_kernel_sizes == std::vector<int64_t>{9, 9, 4, 4, 4, 4, 4});
  CHECK(audio.resblock_kernel_sizes == std::vector<int64_t>{3, 7, 11});
  CHECK(audio.resblock_dilation_sizes.size() == 3);
  // 7 upsample stages take 1024 channels down to 8, which is what conv_post reads
  // (its real weight_v is [1, 8, 7]) — the config and the manifest agree.
  int64_t ch = audio.upsample_initial_channel;
  for (size_t i = 0; i < audio.upsample_rates.size(); ++i) ch /= 2;
  CHECK(ch == 8);
  CHECK(audio_stats.mean.size() == 32);  // per VAE latent channel
  CHECK(audio_stats.std_dev.size() == 32);

  const nlohmann::json video_json = nlohmann::json::parse(vllm_test::kH3VideoVaeConfigJson);
  vllm::MiniMaxH3LatentStats video_stats;
  const vllm::MiniMaxH3VideoVaeDecoderConfig video =
      vllm::ParseMiniMaxH3VideoVaeDecoderConfig(video_json, &video_stats);
  CHECK(video.num_layers == 36);
  CHECK(video.in_channels == 24);
  CHECK(video.out_channels == 3);
  CHECK(video.num_register_tokens == 4);
  CHECK(video.block.heads == 32);
  CHECK(video.block.dim_head == 64);
  CHECK(video.block.dim == 2048);
  CHECK(video.block.ff_inner == 2048 * 4);
  CHECK(video.rope_theta == doctest::Approx(100.0));
  // rope_apply_dim = int(dim_head * rope_dim_ratio) = int(64 * 0.75).
  CHECK(video.rope_apply_dim == 48);
  CHECK(video_stats.mean.size() == 24);  // per video latent channel
  CHECK(video_stats.std_dev.size() == 24);

  // The parsed geometry must agree with the REAL manifest, not just with itself:
  // x_embedder maps latent channels -> model dim.
  for (int64_t i = 0; i < vllm_test::kH3VideoVaeTensorCount; ++i) {
    if (std::string("decoder.x_embedder.weight") == vllm_test::kH3VideoVaeTensors[i].name) {
      CHECK(vllm_test::kH3VideoVaeTensors[i].shape[0] == video.block.dim);
      CHECK(vllm_test::kH3VideoVaeTensors[i].shape[1] == video.in_channels);
    }
  }
}

TEST_CASE("minimax_h3: the encoder GGUF loads KEEP-QUANT and fuses on QUANTIZED bytes") {
  // The encoder is 32B, so f32 materialization (~128 GB) does not fit the box we
  // test on; keeping the ggml blocks holds it at ~14.6 GB. The interesting claim is
  // that the q/k/v and gate/up fusions can be done on the QUANTIZED bytes: ggml rows
  // are independent block sequences, so concatenating whole rows yields a valid
  // block-quant tensor — no dequantize/requantize round trip, and no precision lost
  // to one. This asserts that byte-for-byte.
  const int64_t H = 256;    // hidden (a whole number of Q8_0 blocks)
  const int64_t QD = 128, KVD = 64, FF = 512;

  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "qwen3vl"));
  std::map<std::string, std::string> raw;  // logical name -> the exact bytes written
  auto add_q8 = [&](const std::string& name, int64_t out_dim, int64_t in_dim) {
    const std::vector<float> values = MakeParam("encq." + name, out_dim * in_dim, 0.1);
    const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, in_dim);
    std::string bytes(static_cast<size_t>(out_dim) * row_bytes, '\0');
    vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
    REQUIRE(q != nullptr);
    for (int64_t r = 0; r < out_dim; ++r) {
      q(values.data() + r * in_dim, bytes.data() + static_cast<size_t>(r) * row_bytes, in_dim);
    }
    raw[name] = bytes;
    builder.AddTensor(name, {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
                      /*ggml_type=*/8 /*Q8_0*/, bytes);
  };
  auto add_f32 = [&](const std::string& name, const std::vector<int64_t>& logical) {
    int64_t n = 1;
    for (int64_t d : logical) n *= d;
    const std::vector<float> v = MakeParam("encq." + name, n, 0.1);
    std::string bytes(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = logical.rbegin(); it != logical.rend(); ++it) ne.push_back(uint64_t(*it));
    builder.AddTensor(name, ne, /*ggml_type=*/0, bytes);
  };

  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    add_f32(p + "input_layernorm.weight", {H});
    add_f32(p + "post_attention_layernorm.weight", {H});
    add_f32(p + "self_attn.q_norm.weight", {64});
    add_f32(p + "self_attn.k_norm.weight", {64});
    add_q8(p + "self_attn.q_proj.weight", QD, H);
    add_q8(p + "self_attn.k_proj.weight", KVD, H);
    add_q8(p + "self_attn.v_proj.weight", KVD, H);
    add_q8(p + "self_attn.o_proj.weight", H, QD);
    add_q8(p + "mlp.gate_proj.weight", FF, H);
    add_q8(p + "mlp.up_proj.weight", FF, H);
    add_q8(p + "mlp.down_proj.weight", H, FF);
  }
  add_q8("model.embed_tokens.weight", 512, H);
  add_f32("model.norm.weight", {H});  // must NOT be bound (H3 reads unnormalized)

  const std::string path = "/tmp/minimax_h3_enc_q.gguf";
  {
    const std::string bytes = builder.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), fh) == bytes.size());
    std::fclose(fh);
  }

  const vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::MiniMaxH3EncoderQuantWeights w = vllm::LoadMiniMaxH3EncoderFromGguf(gguf);

  // Projections stayed QUANTIZED.
  CHECK(w.Get("layers.0.self_attn.qkv_proj.weight").dtype == vt::DType::kQ8_0);
  CHECK(w.Get("layers.0.mlp.gate_up_proj.weight").dtype == vt::DType::kQ8_0);
  CHECK(w.Get("layers.0.self_attn.o_proj.weight").dtype == vt::DType::kQ8_0);
  CHECK(w.Get("embed_tokens.weight").dtype == vt::DType::kQ8_0);
  // Norms are f32.
  CHECK(w.Get("layers.0.input_layernorm.weight").dtype == vt::DType::kF32);

  // THE CLAIM: the fused tensor's bytes are exactly q ++ k ++ v.
  const std::string& q = raw["model.layers.0.self_attn.q_proj.weight"];
  const std::string& k = raw["model.layers.0.self_attn.k_proj.weight"];
  const std::string& v = raw["model.layers.0.self_attn.v_proj.weight"];
  const vt::Tensor& fused = w.Get("layers.0.self_attn.qkv_proj.weight");
  CHECK(fused.shape[0] == QD + 2 * KVD);
  CHECK(fused.shape[1] == H);
  const uint8_t* fb = static_cast<const uint8_t*>(fused.data);
  const std::string want = q + k + v;
  REQUIRE(vt::RowSizeBytes(vt::DType::kQ8_0, H) * (QD + 2 * KVD) == want.size());
  bool same = true;
  for (size_t i = 0; i < want.size(); ++i) {
    if (fb[i] != static_cast<uint8_t>(want[i])) { same = false; break; }
  }
  CHECK(same);  // byte-for-byte, in [q|k|v] order

  const std::string& gate = raw["model.layers.0.mlp.gate_proj.weight"];
  const std::string& up = raw["model.layers.0.mlp.up_proj.weight"];
  const vt::Tensor& gu = w.Get("layers.0.mlp.gate_up_proj.weight");
  CHECK(gu.shape[0] == 2 * FF);
  const uint8_t* gb = static_cast<const uint8_t*>(gu.data);
  const std::string want2 = gate + up;
  bool same2 = true;
  for (size_t i = 0; i < want2.size(); ++i) {
    if (gb[i] != static_cast<uint8_t>(want2[i])) { same2 = false; break; }
  }
  CHECK(same2);

  // The separate names are gone, and H3's deltas hold.
  CHECK_FALSE(w.Has("layers.0.self_attn.q_proj.weight"));
  CHECK_FALSE(w.Has("norm.weight"));       // UNNORMALIZED output
  CHECK_FALSE(w.Has("model.norm.weight"));

  // Geometry recovered from the fused shapes alone.
  CHECK(w.config.hidden_size == H);
  CHECK(w.config.head_dim == 64);
  CHECK(w.config.num_attention_heads == QD / 64);
  CHECK(w.config.num_key_value_heads == KVD / 64);
  CHECK(w.config.intermediate_size == FF);
  CHECK(w.config.num_hidden_layers == 2);

  // MIXED ENCODINGS. The SHIPPED Q4_K_M encoder stores v_proj as Q6_K while
  // q_proj/k_proj are Q4_K (the usual K_M recipe of keeping V at higher precision),
  // so that group CANNOT be byte-concatenated. The loader must keep the members
  // SEPARATE rather than dequantize to force a fusion — which would throw away
  // exactly the precision the recipe exists to keep.
  {
    gguf_test::GgufModelBuilder mixed;
    mixed.AddKv(gguf_test::StrKv("general.architecture", "qwen3vl"));
    // Only Q8_0 has a CPU quantizer, so the second encoding is written as
    // correctly-SIZED arbitrary bytes. That is enough and it is honest: this test
    // exercises the loader's ROUTING decision (mixed encodings => keep separate),
    // and the loader copies bytes without interpreting them. Nothing here
    // dequantizes, so no real Q4_0 payload is needed.
    auto add_typed = [&](const std::string& name, int64_t out_dim, int64_t in_dim,
                         vt::DType dt, uint32_t ggml_type) {
      const size_t row_bytes = vt::RowSizeBytes(dt, in_dim);
      std::string bytes(static_cast<size_t>(out_dim) * row_bytes, '\0');
      vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(dt);
      if (q != nullptr) {
        const std::vector<float> values = MakeParam("encm." + name, out_dim * in_dim, 0.1);
        for (int64_t r = 0; r < out_dim; ++r) {
          q(values.data() + r * in_dim, bytes.data() + static_cast<size_t>(r) * row_bytes, in_dim);
        }
      } else {
        for (size_t i = 0; i < bytes.size(); ++i) {
          bytes[i] = static_cast<char>((i * 31 + 7) & 0xFF);
        }
      }
      mixed.AddTensor(name, {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
                      ggml_type, bytes);
    };
    auto add_f32m = [&](const std::string& name, const std::vector<int64_t>& logical) {
      int64_t n = 1;
      for (int64_t d : logical) n *= d;
      const std::vector<float> v = MakeParam("encm." + name, n, 0.1);
      std::string bytes(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
      std::vector<uint64_t> ne;
      for (auto it = logical.rbegin(); it != logical.rend(); ++it) ne.push_back(uint64_t(*it));
      mixed.AddTensor(name, ne, 0, bytes);
    };
    const std::string pm = "model.layers.0.";
    add_f32m(pm + "input_layernorm.weight", {H});
    add_f32m(pm + "post_attention_layernorm.weight", {H});
    add_f32m(pm + "self_attn.q_norm.weight", {64});
    add_f32m(pm + "self_attn.k_norm.weight", {64});
    add_typed(pm + "self_attn.q_proj.weight", QD, H, vt::DType::kQ8_0, 8);
    add_typed(pm + "self_attn.k_proj.weight", KVD, H, vt::DType::kQ8_0, 8);
    add_typed(pm + "self_attn.v_proj.weight", KVD, H, vt::DType::kQ4_0, 2);  // the odd one
    add_typed(pm + "self_attn.o_proj.weight", H, QD, vt::DType::kQ8_0, 8);
    add_typed(pm + "mlp.gate_proj.weight", FF, H, vt::DType::kQ8_0, 8);
    add_typed(pm + "mlp.up_proj.weight", FF, H, vt::DType::kQ8_0, 8);
    add_typed(pm + "mlp.down_proj.weight", H, FF, vt::DType::kQ8_0, 8);

    const std::string mpath = "/tmp/minimax_h3_enc_mixed.gguf";
    const std::string mbytes = mixed.Build();
    FILE* mf = std::fopen(mpath.c_str(), "wb");
    REQUIRE(mf != nullptr);
    CHECK(std::fwrite(mbytes.data(), 1, mbytes.size(), mf) == mbytes.size());
    std::fclose(mf);

    const vllm::GgufFile mg = vllm::GgufFile::Open(mpath);
    const vllm::MiniMaxH3EncoderQuantWeights mw = vllm::LoadMiniMaxH3EncoderFromGguf(mg);
    // qkv could NOT fuse: the members stay separate, each in its OWN encoding.
    CHECK_FALSE(mw.Has("layers.0.self_attn.qkv_proj.weight"));
    REQUIRE(mw.Has("layers.0.self_attn.q_proj.weight"));
    REQUIRE(mw.Has("layers.0.self_attn.v_proj.weight"));
    CHECK(mw.Get("layers.0.self_attn.q_proj.weight").dtype == vt::DType::kQ8_0);
    CHECK(mw.Get("layers.0.self_attn.v_proj.weight").dtype == vt::DType::kQ4_0);
    // gate/up ARE uniform, so they still fuse.
    CHECK(mw.Has("layers.0.mlp.gate_up_proj.weight"));
    // Geometry must still come out right from the UNFUSED shapes.
    CHECK(mw.config.hidden_size == H);
    CHECK(mw.config.num_attention_heads == QD / 64);
    CHECK(mw.config.num_key_value_heads == KVD / 64);
    CHECK(mw.config.intermediate_size == FF);
  }

  // Truncation, which is how H3 keeps min(num_hidden_layers, 50).
  const vllm::MiniMaxH3EncoderQuantWeights t1 =
      vllm::LoadMiniMaxH3EncoderFromGguf(gguf, /*max_layers=*/1);
  CHECK(t1.Has("layers.0.self_attn.qkv_proj.weight"));
  CHECK_FALSE(t1.Has("layers.1.self_attn.qkv_proj.weight"));
  CHECK(t1.config.num_hidden_layers == 1);
}

TEST_CASE("minimax_h3: the DEVICE keep-quant encoder matches the host f32 reference") {
  // The conditioning path end to end: a GGUF is loaded KEEP-QUANT, staged to a
  // device, and its tower run through vt ops — against the gated host f32 reference
  // over the SAME weights. The tolerance is Q8_0's, not f32's: the two paths differ
  // by the quantization of the projections, which is the whole point of the arm.
  const int64_t H = 128, HEADS = 4, KV = 2, HD = 32, FF = 256, SEQ = 6, LAYERS = 2;
  REQUIRE(HEADS * HD == 128);

  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "qwen3vl"));
  std::map<std::string, std::vector<float>> f32;  // the SAME values the host gets
  auto add_q8 = [&](const std::string& gguf_name, const std::string& host_name, int64_t out_dim,
                    int64_t in_dim) {
    std::vector<float> values = MakeParam("encd." + gguf_name, out_dim * in_dim, 0.08);
    const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, in_dim);
    std::string bytes(static_cast<size_t>(out_dim) * row_bytes, '\0');
    vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
    REQUIRE(q != nullptr);
    for (int64_t r = 0; r < out_dim; ++r) {
      q(values.data() + r * in_dim, bytes.data() + static_cast<size_t>(r) * row_bytes, in_dim);
    }
    // The host reference must see the DEQUANTIZED values, so the only difference
    // between the two paths is where the quantization error enters — not the model.
    f32[host_name] = vllm::DequantGgufRowToF32(8, reinterpret_cast<const uint8_t*>(bytes.data()),
                                               out_dim * in_dim);
    builder.AddTensor(gguf_name, {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
                      8, bytes);
  };
  auto add_f32 = [&](const std::string& gguf_name, const std::string& host_name,
                     const std::vector<int64_t>& logical) {
    int64_t n = 1;
    for (int64_t d : logical) n *= d;
    std::vector<float> v = MakeParam("encd." + gguf_name, n, 0.1);
    f32[host_name] = v;
    std::string bytes(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = logical.rbegin(); it != logical.rend(); ++it) ne.push_back(uint64_t(*it));
    builder.AddTensor(gguf_name, ne, 0, bytes);
  };

  for (int64_t l = 0; l < LAYERS; ++l) {
    const std::string g = "model.layers." + std::to_string(l) + ".";
    const std::string hn = "layers." + std::to_string(l) + ".";
    add_f32(g + "input_layernorm.weight", hn + "input_layernorm.weight", {H});
    add_f32(g + "post_attention_layernorm.weight", hn + "post_attention_layernorm.weight", {H});
    add_f32(g + "self_attn.q_norm.weight", hn + "self_attn.q_norm.weight", {HD});
    add_f32(g + "self_attn.k_norm.weight", hn + "self_attn.k_norm.weight", {HD});
    add_q8(g + "self_attn.q_proj.weight", "", HEADS * HD, H);
    add_q8(g + "self_attn.k_proj.weight", "", KV * HD, H);
    add_q8(g + "self_attn.v_proj.weight", "", KV * HD, H);
    add_q8(g + "self_attn.o_proj.weight", hn + "self_attn.o_proj.weight", H, HEADS * HD);
    add_q8(g + "mlp.gate_proj.weight", "", FF, H);
    add_q8(g + "mlp.up_proj.weight", "", FF, H);
    add_q8(g + "mlp.down_proj.weight", hn + "mlp.down_proj.weight", H, FF);
  }

  const std::string path = "/tmp/minimax_h3_enc_dev.gguf";
  {
    const std::string bytes = builder.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), fh) == bytes.size());
    std::fclose(fh);
  }
  const vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::MiniMaxH3EncoderQuantWeights kq = vllm::LoadMiniMaxH3EncoderFromGguf(gguf);
  REQUIRE(kq.config.hidden_size == H);
  REQUIRE(kq.config.num_attention_heads == HEADS);
  REQUIRE(kq.config.num_key_value_heads == KV);

  // Build the HOST reference's weights from the loader's own (fused) tensors, so
  // both paths see identical numbers — the fused qkv/gate_up are what the host
  // forward reads.
  vllm::MiniMaxH3AudioVaeWeights host;
  for (const auto& kv : kq.views) {
    const vt::Tensor& tt = kv.second;
    if (vt::IsBlockQuant(tt.dtype)) {
      host.tensors[kv.first] = vllm::DequantGgufRowToF32(
          8, static_cast<const uint8_t*>(tt.data), tt.shape[0] * tt.shape[1]);
    } else {
      host.tensors[kv.first] =
          std::vector<float>(tt.Ptr<float>(), tt.Ptr<float>() + tt.Numel());
    }
  }

  vllm::MiniMaxH3EncoderConfig cfg = kq.config;
  cfg.selected_layer = LAYERS;
  cfg.mrope_section = {4, 3, 3};
  cfg.rope_theta = 10000.0;

  const std::vector<float> embeds = MakeParam("encd.embeds", SEQ * H, 1.0);
  std::vector<int64_t> pos(static_cast<size_t>(3 * SEQ));
  for (int64_t a = 0; a < 3; ++a) {
    for (int64_t s = 0; s < SEQ; ++s) pos[static_cast<size_t>(a * SEQ + s)] = s;
  }

  const std::vector<float> want = vllm::MiniMaxH3EncoderTextForward(
      cfg, host, embeds, pos.data(), SEQ, /*visual_pos_mask=*/nullptr, {});

  vt::Queue q{Cpu(), nullptr};
  const vllm::MiniMaxH3EncoderDeviceWeights staged =
      vllm::StageMiniMaxH3EncoderWeights(q, kq);
  const std::vector<float> got = vllm::MiniMaxH3EncoderTextForwardDevice(
      q, cfg, staged, embeds, pos.data(), SEQ);

  REQUIRE(got.size() == want.size());
  double err = 0.0, mag = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    err = std::max(err, std::abs(static_cast<double>(got[i] - want[i])));
    mag = std::max(mag, std::abs(static_cast<double>(want[i])));
  }
  INFO("device keep-quant encoder vs host reference: max|diff| = " << err << " (scale " << mag << ")");
  CHECK(err <= 2e-3);
  CHECK(mag > 1e-3);  // the tower produced something, not zeros
  for (float v : got) REQUIRE(std::isfinite(v));

  // DEEPSTACK: the surface #86 could not cover — the DEVICE forward now takes the
  // visual position mask + per-tap blocks and injects them into the first N layers,
  // exactly like the gated host reference and upstream `_deepstack_process`. Gate the
  // device against the host reference WITH deepstack, and prove it changes the result.
  std::vector<uint8_t> visual_mask(static_cast<size_t>(SEQ), 0);
  visual_mask[1] = 1;
  visual_mask[3] = 1;
  visual_mask[4] = 1;
  int64_t num_visual = 0;
  for (uint8_t m : visual_mask) num_visual += m;
  std::vector<std::vector<float>> deepstack;  // one [num_visual, H] block per injected layer
  for (int64_t l = 0; l < LAYERS; ++l) {
    deepstack.push_back(MakeParam("encd.deepstack." + std::to_string(l), num_visual * H, 0.05));
  }
  const std::vector<float> want_deep = vllm::MiniMaxH3EncoderTextForward(
      cfg, host, embeds, pos.data(), SEQ, visual_mask.data(), deepstack);
  const std::vector<float> got_deep = vllm::MiniMaxH3EncoderTextForwardDevice(
      q, cfg, staged, embeds, pos.data(), SEQ, visual_mask.data(), deepstack);
  REQUIRE(got_deep.size() == want_deep.size());
  double derr = 0.0, dmag = 0.0, delta = 0.0;
  for (size_t i = 0; i < want_deep.size(); ++i) {
    derr = std::max(derr, std::abs(static_cast<double>(got_deep[i] - want_deep[i])));
    dmag = std::max(dmag, std::abs(static_cast<double>(want_deep[i])));
    delta = std::max(delta, std::abs(static_cast<double>(want_deep[i] - want[i])));
  }
  INFO("device keep-quant encoder (deepstack) vs host: max|diff| = " << derr << " (scale " << dmag
                                                                     << ")");
  CHECK(derr <= 2e-3);
  CHECK(delta > 1e-4);  // DeepStack must actually move the conditioning, or the gate is vacuous
  for (float v : got_deep) REQUIRE(std::isfinite(v));
}

TEST_CASE("minimax_h3: the embedding gather decodes ONLY the rows it needs, exactly") {
  // The table is [151936, 5120] — ~1.5 GB even quantized — and a prompt touches a
  // few dozen rows, so the gather decodes rows individually. That is only valid
  // because ggml rows are independent block sequences; this asserts a row-wise
  // gather is BIT-IDENTICAL to slicing a full-table dequant.
  const int64_t VOCAB = 64, HID = 128;
  const std::vector<float> values = MakeParam("emb.table", VOCAB * HID, 0.5);
  const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, HID);
  std::string bytes(static_cast<size_t>(VOCAB) * row_bytes, '\0');
  vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
  REQUIRE(q != nullptr);
  for (int64_t r = 0; r < VOCAB; ++r) {
    q(values.data() + r * HID, bytes.data() + static_cast<size_t>(r) * row_bytes, HID);
  }

  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen3vl"));
  b.AddTensor("model.embed_tokens.weight", {static_cast<uint64_t>(HID), static_cast<uint64_t>(VOCAB)},
              8, bytes);
  // One layer, so the loader has a tower to recover geometry from.
  auto f32t = [&](const std::string& n, const std::vector<int64_t>& shape) {
    int64_t nn = 1;
    for (int64_t d : shape) nn *= d;
    const std::vector<float> v = MakeParam("emb." + n, nn, 0.1);
    std::string bb(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) ne.push_back(uint64_t(*it));
    b.AddTensor(n, ne, 0, bb);
  };
  auto q8t = [&](const std::string& n, int64_t o, int64_t i) {
    const std::vector<float> v = MakeParam("emb." + n, o * i, 0.1);
    const size_t rb = vt::RowSizeBytes(vt::DType::kQ8_0, i);
    std::string bb(static_cast<size_t>(o) * rb, '\0');
    for (int64_t r = 0; r < o; ++r) q(v.data() + r * i, bb.data() + static_cast<size_t>(r) * rb, i);
    b.AddTensor(n, {static_cast<uint64_t>(i), static_cast<uint64_t>(o)}, 8, bb);
  };
  const std::string pl = "model.layers.0.";
  f32t(pl + "input_layernorm.weight", {HID});
  f32t(pl + "post_attention_layernorm.weight", {HID});
  f32t(pl + "self_attn.q_norm.weight", {32});
  f32t(pl + "self_attn.k_norm.weight", {32});
  q8t(pl + "self_attn.q_proj.weight", 128, HID);
  q8t(pl + "self_attn.k_proj.weight", 64, HID);
  q8t(pl + "self_attn.v_proj.weight", 64, HID);
  q8t(pl + "self_attn.o_proj.weight", HID, 128);
  q8t(pl + "mlp.gate_proj.weight", 256, HID);
  q8t(pl + "mlp.up_proj.weight", 256, HID);
  q8t(pl + "mlp.down_proj.weight", HID, 256);

  const std::string path = "/tmp/minimax_h3_emb.gguf";
  {
    const std::string blob = b.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(blob.data(), 1, blob.size(), fh) == blob.size());
    std::fclose(fh);
  }
  const vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::MiniMaxH3EncoderQuantWeights w = vllm::LoadMiniMaxH3EncoderFromGguf(g);
  REQUIRE(w.Has("embed_tokens.weight"));
  CHECK(w.Get("embed_tokens.weight").dtype == vt::DType::kQ8_0);  // stayed quantized

  // The oracle: dequantize the WHOLE table once, then slice.
  const std::vector<float> full = vllm::DequantGgufRowToF32(
      8, reinterpret_cast<const uint8_t*>(bytes.data()), VOCAB * HID);

  const std::vector<int32_t> ids = {0, 7, 63, 7, 1};  // includes a repeat and both ends
  const std::vector<float> got = vllm::MiniMaxH3EncoderEmbedTokens(w, ids);
  REQUIRE(got.size() == ids.size() * static_cast<size_t>(HID));
  for (size_t i = 0; i < ids.size(); ++i) {
    for (int64_t c = 0; c < HID; ++c) {
      // BIT-identical, not approximate: same bytes through the same decoder.
      CHECK(got[i * HID + c] == full[static_cast<size_t>(ids[i]) * HID + c]);
    }
  }
  // An out-of-range id must THROW rather than read past the table.
  CHECK_THROWS(vllm::MiniMaxH3EncoderEmbedTokens(w, {VOCAB}));
  CHECK_THROWS(vllm::MiniMaxH3EncoderEmbedTokens(w, {-1}));
}

// ---------------------------------------------------------------------------
// The ORIGINAL bf16 H3-Encoder: 14 safetensors shards, 63 GB
// ---------------------------------------------------------------------------
// Every H3 render so far conditioned on the Q4_K_M encoder, and nobody had ever
// measured what that quantization does to the conditioning tensor. Asking needs
// the SAME prompt encoded by the unquantized tower — which ships as 14 shards,
// while `--encoder` only ever accepted a GGUF. These gates cover the loader that
// closes that gap: the name map, the two row fusions, the bf16 residency, and —
// the one that has bitten this codebase before — that the new path actually RAN
// rather than silently falling back to the GGUF one.

namespace {

// The reduced geometry the encoder-shard gates run at, at the REAL layout:
// q/k/v and gate/up SEPARATE on disk, GQA (heads > kv_heads), head_dim dividing
// both projection row counts.
struct EncShardGeometry {
  int64_t layers = 3;
  int64_t hidden = 32;
  int64_t heads = 4;
  int64_t kv_heads = 2;
  int64_t head_dim = 8;
  int64_t ffn = 48;
  int64_t vocab = 24;
};

// Round through bf16, so an "F32" copy of a checkpoint holds values the BF16 copy
// can represent EXACTLY. That is what lets the widen-vs-native comparison below
// demand BIT-IDENTICAL outputs rather than a tolerance.
std::vector<float> Bf16RoundTrip(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &v[i], sizeof(bits));
    const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    const uint32_t back = rounded & 0xFFFF0000u;
    std::memcpy(&out[i], &back, sizeof(back));
  }
  return out;
}

// The REAL encoder release's tensor names, at reduced dims. Deliberately includes
// the three things the loader must NOT bind (`model.language_model.norm.weight`,
// `lm_head.weight`, the whole `model.visual.` tower) — a loader that grabs them
// would still "work", and would apply a final RMSNorm H3 does not.
std::vector<H3StEntry> BuildEncoderShardEntries(const EncShardGeometry& g, bool bf16) {
  std::vector<H3StEntry> entries;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    const std::vector<float> values = Bf16RoundTrip(MakeParam("encsh." + name, n, 0.2));
    entries.push_back({name, bf16 ? "BF16" : "F32", shape,
                       bf16 ? PackBf16(values) : PackF32(values)});
  };
  const std::string lm = "model.language_model.";
  add(lm + "embed_tokens.weight", {g.vocab, g.hidden});
  for (int64_t l = 0; l < g.layers; ++l) {
    const std::string p = lm + "layers." + std::to_string(l) + ".";
    add(p + "input_layernorm.weight", {g.hidden});
    add(p + "post_attention_layernorm.weight", {g.hidden});
    add(p + "self_attn.q_norm.weight", {g.head_dim});
    add(p + "self_attn.k_norm.weight", {g.head_dim});
    add(p + "self_attn.q_proj.weight", {g.heads * g.head_dim, g.hidden});
    add(p + "self_attn.k_proj.weight", {g.kv_heads * g.head_dim, g.hidden});
    add(p + "self_attn.v_proj.weight", {g.kv_heads * g.head_dim, g.hidden});
    add(p + "self_attn.o_proj.weight", {g.hidden, g.heads * g.head_dim});
    add(p + "mlp.gate_proj.weight", {g.ffn, g.hidden});
    add(p + "mlp.up_proj.weight", {g.ffn, g.hidden});
    add(p + "mlp.down_proj.weight", {g.hidden, g.ffn});
  }
  // MUST NOT be bound: H3 reads the UNNORMALIZED truncated output, never a head.
  add(lm + "norm.weight", {g.hidden});
  add("lm_head.weight", {g.vocab, g.hidden});
  // The vision tower shares the release and is not part of a text-only encode.
  add("model.visual.blocks.0.attn.qkv.weight", {3 * g.hidden, g.hidden});
  add("model.visual.merger.norm.weight", {g.hidden});
  return entries;
}

const H3StEntry& FindEntry(const std::vector<H3StEntry>& entries, const std::string& name) {
  for (const H3StEntry& e : entries) {
    if (e.name == name) return e;
  }
  REQUIRE_MESSAGE(false, "synthetic encoder checkpoint has no tensor named " << name);
  return entries.front();
}

}  // namespace

TEST_CASE("minimax_h3: the bf16 encoder shards resolve, FUSE and STREAM to the device") {
  const EncShardGeometry g;
  const std::vector<H3StEntry> entries = BuildEncoderShardEntries(g, /*bf16=*/true);
  const std::string dir = "/tmp/minimax_h3_enc_shards";
  const size_t kShards = 4;
  const std::map<std::string, std::string> promised =
      WriteMiniMaxH3ShardedDit(entries, dir, kShards);
  const vllm::MiniMaxH3ShardedCheckpoint ckpt = vllm::MiniMaxH3ShardedCheckpoint::Open(dir);
  CHECK(ckpt.ShardCount() == kShards);
  CHECK(ckpt.Names().size() == entries.size());

  // GEOMETRY FROM SHAPES ALONE — no payload read. The recovery rules are the GGUF
  // loader's (head_dim from q_norm, heads from q_proj rows), so the two arms
  // cannot disagree about what model they are running.
  const vllm::MiniMaxH3EncoderConfig cfg = vllm::MiniMaxH3EncoderConfigFromShards(ckpt);
  CHECK(cfg.num_hidden_layers == g.layers);
  CHECK(cfg.hidden_size == g.hidden);
  CHECK(cfg.num_attention_heads == g.heads);
  CHECK(cfg.num_key_value_heads == g.kv_heads);
  CHECK(cfg.head_dim == g.head_dim);
  CHECK(cfg.intermediate_size == g.ffn);
  // The knobs the shapes cannot carry keep their defaults, and they are the SAME
  // defaults the GGUF arm leaves in place — otherwise an A/B would be comparing
  // two different RoPEs, not two quantizations.
  CHECK(cfg.selected_layer == vllm::kMiniMaxH3EncoderSelectedLayer);
  CHECK(cfg.rope_theta == doctest::Approx(5000000.0));
  CHECK(cfg.mrope_section == std::vector<int64_t>{24, 20, 20});

  vllm::ResetMiniMaxH3EncoderShardStreamStats();
  vt::Queue q{Cpu(), nullptr};
  vllm::MiniMaxH3EncoderConfig streamed_cfg;
  const vllm::MiniMaxH3EncoderDeviceWeights w =
      vllm::StreamMiniMaxH3EncoderShardsToDevice(q, ckpt, /*max_layers=*/0, &streamed_cfg);
  CHECK(streamed_cfg.num_hidden_layers == g.layers);
  CHECK(streamed_cfg.intermediate_size == g.ffn);

  // ★ THE LOADER RAN. A green suite over a path that silently fell back to the
  // GGUF loader is a failure mode this codebase has hit; the counters make the
  // bf16 path OBSERVABLE and this asserts on them rather than on a comment.
  const vllm::MiniMaxH3EncoderShardStreamStats st =
      vllm::GetMiniMaxH3EncoderShardStreamStats();
  CHECK(st.shards_opened == kShards);
  CHECK(st.layers_streamed == static_cast<uint64_t>(g.layers));
  CHECK(st.tensors_streamed == static_cast<uint64_t>(8 * g.layers));
  CHECK(st.fused_groups == static_cast<uint64_t>(2 * g.layers));  // qkv + gate_up per layer
  // Every PROJECTION took the no-host-copy path (mmap -> device); only the four
  // norms per layer needed a conversion, and those are [hidden]/[head_dim].
  CHECK(st.direct_uploads == static_cast<uint64_t>(7 * g.layers));
  CHECK(st.converted_uploads == static_cast<uint64_t>(4 * g.layers));
  CHECK(st.bytes_uploaded > 0);
  // Peak host memory is bounded by the largest CONVERSION, i.e. one norm — it
  // cannot scale with the model. On the real 63 GB release that is the difference
  // between a run and the OOM-kill a previous H3 loader took at anon-rss 125 GB.
  CHECK(st.host_peak_bytes == static_cast<uint64_t>(g.hidden) * sizeof(float));
  CHECK(st.host_peak_bytes * 16 < st.bytes_uploaded);

  // ★ These views are BF16 — a dtype the GGUF loader can never produce (it binds
  // block-quant projections and f32 norms). So this is not the GGUF path wearing
  // a different name.
  const vt::Tensor& qkv = w.Get("layers.0.self_attn.qkv_proj.weight");
  CHECK(qkv.dtype == vt::DType::kBF16);
  CHECK(w.Get("layers.0.mlp.gate_up_proj.weight").dtype == vt::DType::kBF16);
  CHECK(w.Get("layers.0.self_attn.o_proj.weight").dtype == vt::DType::kBF16);
  CHECK(w.Get("layers.0.mlp.down_proj.weight").dtype == vt::DType::kBF16);
  // Norms stay f32: vt::RmsNorm takes an f32 weight.
  CHECK(w.Get("layers.0.input_layernorm.weight").dtype == vt::DType::kF32);
  CHECK(w.Get("layers.0.self_attn.q_norm.weight").dtype == vt::DType::kF32);

  // SHAPES: the fusions are row concatenations, and the forward slices them.
  CHECK(qkv.shape[0] == (g.heads + 2 * g.kv_heads) * g.head_dim);
  CHECK(qkv.shape[1] == g.hidden);
  const vt::Tensor& gu = w.Get("layers.0.mlp.gate_up_proj.weight");
  CHECK(gu.shape[0] == 2 * g.ffn);
  CHECK(gu.shape[1] == g.hidden);
  CHECK(w.Get("layers.0.self_attn.q_norm.weight").shape[0] == g.head_dim);

  // ★ AND THE ORDER IS RIGHT, byte for byte. [q|k|v] and [gate|up] are what the
  // forward assumes; any other order still runs and silently feeds keys into the
  // query path.
  for (int64_t l = 0; l < g.layers; ++l) {
    const std::string src = "model.language_model.layers." + std::to_string(l) + ".";
    const std::string dst = "layers." + std::to_string(l) + ".";
    const std::string want_qkv = FindEntry(entries, src + "self_attn.q_proj.weight").bytes +
                                 FindEntry(entries, src + "self_attn.k_proj.weight").bytes +
                                 FindEntry(entries, src + "self_attn.v_proj.weight").bytes;
    const vt::Tensor& got_qkv = w.Get(dst + "self_attn.qkv_proj.weight");
    REQUIRE(static_cast<size_t>(got_qkv.Numel()) * 2 == want_qkv.size());
    CHECK(std::memcmp(got_qkv.data, want_qkv.data(), want_qkv.size()) == 0);

    const std::string want_gu = FindEntry(entries, src + "mlp.gate_proj.weight").bytes +
                                FindEntry(entries, src + "mlp.up_proj.weight").bytes;
    const vt::Tensor& got_gu = w.Get(dst + "mlp.gate_up_proj.weight");
    REQUIRE(static_cast<size_t>(got_gu.Numel()) * 2 == want_gu.size());
    CHECK(std::memcmp(got_gu.data, want_gu.data(), want_gu.size()) == 0);

    // The unfused projections pass through untouched.
    const std::string want_o = FindEntry(entries, src + "self_attn.o_proj.weight").bytes;
    const vt::Tensor& got_o = w.Get(dst + "self_attn.o_proj.weight");
    CHECK(std::memcmp(got_o.data, want_o.data(), want_o.size()) == 0);
  }

  // The SEPARATE names must be gone — leaving them would let a forward read an
  // unfused tensor and take the mixed-encoding branch by accident.
  CHECK_FALSE(w.Has("layers.0.self_attn.q_proj.weight"));
  CHECK_FALSE(w.Has("layers.0.mlp.gate_proj.weight"));
  // H3 deltas: no final norm, no lm_head, no vision tower on a text-only encode.
  CHECK_FALSE(w.Has("norm.weight"));
  CHECK_FALSE(w.Has("lm_head.weight"));
  CHECK_FALSE(w.Has("blocks.0.attn.qkv.weight"));

  // TRUNCATION — H3 runs min(num_hidden_layers, 50) and the release ships 64, so
  // the loader must be able to stop early. That is not cosmetic: it is 14 layers
  // of a 48.8 GiB residency.
  CHECK(w.Has("layers." + std::to_string(g.layers - 1) + ".self_attn.qkv_proj.weight"));
  vllm::ResetMiniMaxH3EncoderShardStreamStats();
  const vllm::MiniMaxH3EncoderDeviceWeights trunc =
      vllm::StreamMiniMaxH3EncoderShardsToDevice(q, ckpt, /*max_layers=*/1, nullptr);
  CHECK(trunc.Has("layers.0.self_attn.qkv_proj.weight"));
  CHECK_FALSE(trunc.Has("layers.1.self_attn.qkv_proj.weight"));
  CHECK(vllm::GetMiniMaxH3EncoderShardStreamStats().layers_streamed == 1);
  CHECK(vllm::MiniMaxH3EncoderConfigFromShards(ckpt, /*max_layers=*/1).num_hidden_layers == 1);

  // The EMBEDDING gather reads rows out of the mmap without materializing the
  // table, and must be exact — an off-by-one row is a different prompt.
  const std::vector<int32_t> ids = {5, 0, static_cast<int32_t>(g.vocab - 1), 5};
  const std::vector<float> got_rows = vllm::MiniMaxH3EncoderEmbedTokensFromShards(ckpt, ids);
  REQUIRE(got_rows.size() == ids.size() * static_cast<size_t>(g.hidden));
  const std::vector<float> table =
      Bf16RoundTrip(MakeParam("encsh.model.language_model.embed_tokens.weight",
                              g.vocab * g.hidden, 0.2));
  for (size_t i = 0; i < ids.size(); ++i) {
    for (int64_t c = 0; c < g.hidden; ++c) {
      CHECK(got_rows[i * static_cast<size_t>(g.hidden) + c] ==
            table[static_cast<size_t>(ids[i]) * static_cast<size_t>(g.hidden) + c]);
    }
  }
  CHECK_THROWS(vllm::MiniMaxH3EncoderEmbedTokensFromShards(ckpt, {static_cast<int32_t>(g.vocab)}));
  CHECK_THROWS(vllm::MiniMaxH3EncoderEmbedTokensFromShards(ckpt, {-1}));

  RemoveShardedDit(dir, kShards);
}

TEST_CASE("minimax_h3: the bf16 encoder runs the SAME forward as an f32-staged tower") {
  // THE CLAIM UNDER TEST. The unquantized arm keeps its projections BF16 on the
  // device (48.8 GiB for the 50 layers H3 runs; 97.5 GiB as f32, against a 122 GiB
  // UNIFIED pool) and widens each one to f32 immediately before its GEMM. That is
  // a RESIDENCY trick, and this asserts it is nothing more: bf16 -> f32 is exact,
  // so a widened tower and an f32-staged tower holding the same values must agree
  // BIT FOR BIT — not to a tolerance.
  //
  // Without this, "we measured what quantizing the encoder costs" would be
  // confounded by whatever the widening itself did.
  const EncShardGeometry g;
  const int64_t SEQ = 6;

  const std::string bf16_dir = "/tmp/minimax_h3_enc_shards_bf16";
  const std::string f32_dir = "/tmp/minimax_h3_enc_shards_f32";
  const size_t kShards = 3;
  const std::vector<H3StEntry> bf16_entries = BuildEncoderShardEntries(g, /*bf16=*/true);
  const std::vector<H3StEntry> f32_entries = BuildEncoderShardEntries(g, /*bf16=*/false);
  WriteMiniMaxH3ShardedDit(bf16_entries, bf16_dir, kShards);
  WriteMiniMaxH3ShardedDit(f32_entries, f32_dir, kShards);

  const vllm::MiniMaxH3ShardedCheckpoint bf16_ckpt =
      vllm::MiniMaxH3ShardedCheckpoint::Open(bf16_dir);
  const vllm::MiniMaxH3ShardedCheckpoint f32_ckpt =
      vllm::MiniMaxH3ShardedCheckpoint::Open(f32_dir);

  vt::Queue q{Cpu(), nullptr};
  const vllm::MiniMaxH3EncoderDeviceWeights bf16_w =
      vllm::StreamMiniMaxH3EncoderShardsToDevice(q, bf16_ckpt);
  const vllm::MiniMaxH3EncoderDeviceWeights f32_w =
      vllm::StreamMiniMaxH3EncoderShardsToDevice(q, f32_ckpt);
  // The two really are staged differently — otherwise this compares a path to
  // itself, which is the trap the counters exist to catch.
  CHECK(bf16_w.Get("layers.0.self_attn.qkv_proj.weight").dtype == vt::DType::kBF16);
  CHECK(f32_w.Get("layers.0.self_attn.qkv_proj.weight").dtype == vt::DType::kF32);

  vllm::MiniMaxH3EncoderConfig cfg = vllm::MiniMaxH3EncoderConfigFromShards(bf16_ckpt);
  cfg.selected_layer = g.layers;
  cfg.mrope_section = {2, 1, 1};
  cfg.rope_theta = 10000.0;

  const std::vector<float> embeds = MakeParam("encsh.embeds", SEQ * g.hidden, 1.0);
  std::vector<int64_t> pos(static_cast<size_t>(3 * SEQ));
  for (int64_t a = 0; a < 3; ++a) {
    for (int64_t s = 0; s < SEQ; ++s) pos[static_cast<size_t>(a * SEQ + s)] = s;
  }

  const std::vector<float> from_bf16 =
      vllm::MiniMaxH3EncoderTextForwardDevice(q, cfg, bf16_w, embeds, pos.data(), SEQ);
  const std::vector<float> from_f32 =
      vllm::MiniMaxH3EncoderTextForwardDevice(q, cfg, f32_w, embeds, pos.data(), SEQ);

  REQUIRE(from_bf16.size() == static_cast<size_t>(SEQ * g.hidden));
  REQUIRE(from_f32.size() == from_bf16.size());
  CHECK(std::memcmp(from_bf16.data(), from_f32.data(), from_bf16.size() * sizeof(float)) == 0);
  // ...and it produced a tower's output, not zeros.
  double mag = 0.0;
  for (float v : from_bf16) {
    REQUIRE(std::isfinite(v));
    mag = std::max(mag, std::abs(static_cast<double>(v)));
  }
  CHECK(mag > 1e-3);

  RemoveShardedDit(bf16_dir, kShards);
  RemoveShardedDit(f32_dir, kShards);
}

// ---------------------------------------------------------------------------
// PRUNED (AdaLN timestep-CURVE) checkpoints — spec section 8.20, issue #241.
//
// The community `pruned` variants are NOT lossily pruned: ComfyUI's curve form
// (comfy/ldm/minimax/model.py:419-432, 610-615) replaces the sinusoidal +
// 2-layer-MLP time embedder with an `adaln_t_table` that is LERPED at the
// per-row timestep, drops the SiLU before the AdaLN linear, and narrows that
// linear's in_features from time_embed_dim (2688) to the curve width (8).
// ---------------------------------------------------------------------------

TEST_CASE("minimax_h3: the AdaLN curve embedding matches upstream's clamped lerp") {
  const int64_t grid = vllm_test::kH3AdalnCurveGrid;
  const int64_t dim = vllm_test::kH3AdalnCurveDim;
  // The generator's table is h3_rand("adaln_t_table") with scale 1, offset 0.
  const std::vector<float> table = MakeParam("adaln_t_table", grid * dim, 1.0);

  const std::vector<float> t(vllm_test::kH3AdalnCurveT,
                             vllm_test::kH3AdalnCurveT + vllm_test::kH3AdalnCurveTCount);
  const std::vector<float> got =
      vllm::MiniMaxH3AdalnCurveEmbed(table.data(), grid, dim, t.data(),
                                     static_cast<int64_t>(t.size()));
  REQUIRE(static_cast<int64_t>(got.size()) == vllm_test::kH3AdalnCurveEmbCount);
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) -
                                     static_cast<double>(vllm_test::kH3AdalnCurveEmb[i])));
  }
  INFO("max |diff| vs the upstream expression = " << worst);
  CHECK(worst < 1e-6);

  // t = 1.0 must land on the LAST interval, not read past the table: that is the
  // whole reason upstream clamps i0 to grid-2. Row 6 of the golden is t=1.0 and
  // must equal the final table row exactly.
  for (int64_t d = 0; d < dim; ++d) {
    CHECK(got[static_cast<size_t>(6 * dim + d)] ==
          doctest::Approx(table[static_cast<size_t>((grid - 1) * dim + d)]).epsilon(1e-6));
  }
  // Out-of-range timesteps clamp to the curve ends rather than extrapolating.
  for (int64_t d = 0; d < dim; ++d) {
    CHECK(got[static_cast<size_t>(7 * dim + d)] ==
          doctest::Approx(table[static_cast<size_t>(d)]).epsilon(1e-6));
    CHECK(got[static_cast<size_t>(8 * dim + d)] ==
          doctest::Approx(table[static_cast<size_t>((grid - 1) * dim + d)]).epsilon(1e-6));
  }
}

TEST_CASE("minimax_h3: a REAL PRUNED GGUF resolves onto our weight contract") {
  // unsloth/MiniMax-H3-GGUF `minimax_h3_fl2va_pruned-Q8_0.gguf`, header only.
  CHECK(vllm_test::kH3PrunedGgufVersion == 3);
  REQUIRE(vllm_test::kH3PrunedGgufTensorCount ==
          static_cast<int64_t>(std::size(vllm_test::kH3PrunedGgufTensors)));
  // 535 unpruned - 4 time_embedder tensors + 1 adaln_t_table.
  CHECK(vllm_test::kH3PrunedGgufTensorCount == 532);
  CHECK(vllm_test::kH3PrunedGgufTensorCount == vllm_test::kH3GgufTensorCount - 3);

  std::vector<vllm::MiniMaxH3TensorSpec> manifest;
  manifest.reserve(static_cast<size_t>(vllm_test::kH3PrunedGgufTensorCount));
  for (const vllm_test::H3PrunedGgufTensor& t : vllm_test::kH3PrunedGgufTensors) {
    const std::vector<int64_t> dims(t.dims, t.dims + t.n_dims);
    const std::vector<int64_t> orig(t.orig_shape, t.orig_shape + t.orig_n_dims);
    vllm::MiniMaxH3TensorSpec spec;
    spec.name = t.name;
    spec.shape = vllm::MiniMaxH3GgufLogicalShape(dims, orig);
    spec.fp32 = t.ggml_type == 0;
    manifest.push_back(std::move(spec));
  }

  // Not one `time_embedder.*` tensor survives, and `adaln_t_table` is there.
  bool saw_table = false;
  for (const vllm::MiniMaxH3TensorSpec& spec : manifest) {
    CHECK(spec.name.rfind("time_embedder.", 0) != 0);
    if (spec.name == "adaln_t_table") {
      saw_table = true;
      REQUIRE(spec.shape.size() == 2);
      CHECK(spec.shape[0] == 1025);  // the curve grid
      CHECK(spec.shape[1] == 8);     // the curve width
      CHECK(spec.fp32);              // F32 upstream, and read on the host
    }
  }
  CHECK(saw_table);

  // The geometry still comes from the SHAPES ALONE.
  const MiniMaxH3DitParams p = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  CHECK(p.use_adaln_curves());
  CHECK(p.adaln_curve_grid == 1025);
  CHECK(p.time_embed_dim == 8);  // the AdaLN linear's in_features, per the table
  CHECK(p.num_layers == 50);
  CHECK(p.token_refiner_num_layers == 2);
  CHECK(p.hidden_size == 5376);
  CHECK(p.num_attention_heads == 56);
  CHECK(p.attention_head_dim == 128);
  CHECK(p.ffn_hidden_size == 14336);
  CHECK(p.latents_dim == 24);
  CHECK(p.audio_latents_dim == 32);
  CHECK(p.text_dim == 5120);
  CHECK(p.rope_inv_freq_len == 16);
  CHECK(p.video_row_width() == 96);
  CHECK(p.adaln_out_features == 96768);
  CHECK(p.final_adaln_out_features == 10752);

  // Identity name map, exactly as for the unpruned arm.
  const std::vector<vllm::MiniMaxH3TensorSpec> expected = EnumerateMiniMaxH3DitTensors(p);
  CHECK(manifest.size() == expected.size());
  std::map<std::string, std::vector<int64_t>> got;
  for (const vllm::MiniMaxH3TensorSpec& spec : manifest) got[spec.name] = spec.shape;
  for (const vllm::MiniMaxH3TensorSpec& want : expected) {
    INFO("contract tensor " << want.name);
    const auto it = got.find(want.name);
    REQUIRE(it != got.end());
    CHECK(it->second == want.shape);
  }

  // in_features 8 means the AdaLN projections no longer need ComfyUI's
  // quant-block reshape: 8 is not 2688, so there is no orig_shape key at all.
  for (const vllm_test::H3PrunedGgufTensor& t : vllm_test::kH3PrunedGgufTensors) {
    if (std::string(t.name) != "blocks.0.adaln_proj.linear.weight") continue;
    CHECK(t.orig_n_dims == 0);
    CHECK(t.dims[0] == 8);
    CHECK(t.dims[1] == 96768);
  }

  // `adaln_t_table` is an fp32 ISLAND, in the same category as rope.inv_freq:
  // it is F32 upstream and it is consumed on the host.
  CHECK(vllm::MiniMaxH3IsFp32IslandTensor("adaln_t_table"));
  CHECK(vllm::MiniMaxH3IsFp32IslandTensor("rope.inv_freq"));
}

TEST_CASE("minimax_h3: a checkpoint may carry the time embedder or the curve, never both") {
  // The stop condition from spec 8.20: a half-and-half file is a THIRD form and
  // must fail loudly rather than load with one path silently unbound.
  MiniMaxH3DitParams base;
  base.num_layers = 2;
  base.token_refiner_num_layers = 1;
  std::vector<vllm::MiniMaxH3TensorSpec> both = EnumerateMiniMaxH3DitTensors(base);
  both.push_back({"adaln_t_table", {1025, 8}, true});
  CHECK_THROWS(vllm::ParseMiniMaxH3DitParamsFromGgufManifest(both));

  MiniMaxH3DitParams curve = base;
  curve.adaln_curve_grid = 1025;
  curve.time_embed_dim = 8;
  const std::vector<vllm::MiniMaxH3TensorSpec> only_curve =
      EnumerateMiniMaxH3DitTensors(curve);
  const MiniMaxH3DitParams round = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(only_curve);
  CHECK(round.adaln_curve_grid == 1025);
  CHECK(round.time_embed_dim == 8);
  CHECK(round.num_layers == 2);
}

TEST_CASE("minimax_h3: a CONSTRUCTED curve checkpoint reproduces the unpruned forward exactly") {
  // The strongest available gate on the pruned forward, and the only one that
  // pins the SiLU rule.
  //
  // The curve form feeds `adaln_proj.linear` with table[i] directly (no SiLU,
  // comfy/ldm/minimax/model.py:197 + :421); the unpruned form feeds it with
  // silu(time_embedder(t)). So a table whose rows ARE silu(time_embedder(t_k))
  // at the grid points t_k, sampled at exactly those t_k, must drive the two
  // forwards through identical AdaLN inputs — and therefore to identical
  // outputs. Any one of {applying SiLU in the curve path, getting the lerp
  // index wrong, reading in_features from the wrong place} breaks the equality.
  const std::unique_ptr<GoldenWeights> unpruned = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = unpruned->params;
  const std::unique_ptr<DitForwardCase> c = BuildDitForwardCase(p);
  const int64_t m = c->in.num_unique_timesteps;
  REQUIRE(m >= 1);

  // A grid whose points include every unique timestep of the case: put t_k at
  // grid index k and pad the rest, so `pos` lands exactly on an integer.
  const int64_t grid = m + 3;
  std::vector<float> t_on_grid(static_cast<size_t>(m));
  for (int64_t k = 0; k < m; ++k) {
    t_on_grid[static_cast<size_t>(k)] = static_cast<float>(k) / static_cast<float>(grid - 1);
  }

  // Row k of the table = silu(time_embedder(t_on_grid[k])), i.e. exactly what
  // the unpruned AdaLN linear consumes at that timestep. Everything else in the
  // rest of the table is arbitrary and must never be read.
  MiniMaxH3DitInputs probe = c->in;
  probe.unique_timesteps = t_on_grid.data();
  probe.num_unique_timesteps = m;
  const std::vector<float> t_emb =
      vllm::MiniMaxH3SinusoidalTimeEmbed(Cpu(), p, unpruned->views, t_on_grid.data(), m);
  REQUIRE(static_cast<int64_t>(t_emb.size()) == m * p.time_embed_dim);
  std::vector<float> table = MakeParam("curve.filler", grid * p.time_embed_dim, 7.0);
  for (int64_t k = 0; k < m; ++k) {
    for (int64_t d = 0; d < p.time_embed_dim; ++d) {
      const float v = t_emb[static_cast<size_t>(k * p.time_embed_dim + d)];
      table[static_cast<size_t>(k * p.time_embed_dim + d)] =
          v / (1.0f + std::exp(-v));  // silu, the activation the table absorbs
    }
  }

  // The unpruned run, at the grid timesteps.
  const MiniMaxH3DitOutputs want =
      MiniMaxH3DitForward(Cpu(), p, unpruned->views, probe, vt::DType::kF32);

  // The SAME weights, minus the time embedder, plus the table.
  MiniMaxH3DitParams cp = p;
  cp.adaln_curve_grid = grid;
  MiniMaxH3DitWeights cw = unpruned->views;
  cw.time_proj_in_w = vt::Tensor{};
  cw.time_proj_in_b = vt::Tensor{};
  cw.time_proj_out_w = vt::Tensor{};
  cw.time_proj_out_b = vt::Tensor{};
  cw.adaln_t_table = View2D(table, grid, cp.time_embed_dim);
  const MiniMaxH3DitOutputs got = MiniMaxH3DitForward(Cpu(), cp, cw, probe, vt::DType::kF32);

  REQUIRE(got.video_logits.size() == want.video_logits.size());
  REQUIRE(got.audio_logits.size() == want.audio_logits.size());
  const double video_err =
      MaxAbsDiff(got.video_logits, want.video_logits.data(), got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, want.audio_logits.data(), got.audio_logits.size());
  INFO("curve-vs-unpruned video max|diff| = " << video_err << ", audio = " << audio_err);
  // Identical arithmetic on both sides apart from how t_emb was produced, so the
  // slack is only the f32 rounding of silu(t_emb) through the table.
  CHECK(video_err <= 1e-5);
  CHECK(audio_err <= 1e-5);
  // And the equality must be non-trivial: the outputs are not all zero.
  double mag = 0.0;
  for (float v : want.video_logits) mag = std::max(mag, std::abs(static_cast<double>(v)));
  CHECK(mag > 1e-3);

  // A curve checkpoint whose table is not bound must fail LOUDLY rather than
  // read a null pointer.
  MiniMaxH3DitWeights unbound = cw;
  unbound.adaln_t_table = vt::Tensor{};
  CHECK_THROWS(MiniMaxH3DitForward(Cpu(), cp, unbound, probe, vt::DType::kF32));
}
