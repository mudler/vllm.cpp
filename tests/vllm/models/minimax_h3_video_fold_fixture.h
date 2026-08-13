// MiniMax-H3 VIDEO FOLD fixture — the deterministic tiny checkpoint set the
// ARCH-ONE-SURFACE ROW 2 fold gate runs the WHOLE t2va assembly on (CPU-feasible
// by construction: reduced dims, random-but-deterministic weights).
//
// This header is deliberately SELF-CONTAINED file assembly (gguf_builder.h +
// raw safetensors/json writing, no vllm library dependency), because it is used
// on BOTH sides of the fold:
//   1. at the branch BASE it generated the inputs the PRE-fold
//      `minimax-h3-gen` binary rendered, whose frames + WAV are committed as
//      tests/vllm/models/fixtures/minimax_h3_video_fold/ goldens;
//   2. the fold gate (test_minimax_h3_video_fold.cpp) regenerates the SAME
//      bytes at runtime and requires the library seam, the replicated old
//      pipeline, and the rewritten thin-client binary to reproduce those
//      goldens byte-identically.
// Every value is a pure function of the tensor name (FNV-1a -> splitmix64), so
// the generated files are byte-stable across platforms and compilers.
//
// Geometry: the tiny ComfyUI-GGUF DiT of test_minimax_h3.cpp:3786 (2 layers,
// hidden 64, latents 8/6) + a video-VAE whose PARSED config reproduces the
// reduced ViT3D decoder of the "WHOLE t2va path composes end to end" case, and
// the same reduced BigVGAN audio VAE. Render request: --partition fl2va
// --keep-quant --frames 5 --height 32 --width 32 --steps 3 (latent 2x2x2,
// audio_t 8) — keep-quant is the arm the goldens were captured on
// (FixtureModelParams sets dequant_bf16 = 0, test_minimax_h3_video_fold.cpp).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "../gguf_builder.h"

namespace minimax_h3_fold {

namespace fs = std::filesystem;

// ── deterministic parameter stream (FNV-1a name hash -> splitmix64) ──────────
inline std::vector<float> Param(const std::string& name, int64_t count, double scale,
                                double offset = 0.0) {
  uint64_t x = 1469598103934665603ULL;
  for (const char c : name) {
    x ^= static_cast<unsigned char>(c);
    x *= 1099511628211ULL;
  }
  std::vector<float> out(static_cast<size_t>(count));
  for (float& v : out) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * 0x1.0p-53;  // [0, 1)
    v = static_cast<float>((u * 2.0 - 1.0) * scale + offset);
  }
  return out;
}

inline void WriteFileBytes(const std::string& path, const std::string& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("fixture: cannot write " + path);
  if (!bytes.empty() && std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
    std::fclose(f);
    throw std::runtime_error("fixture: short write " + path);
  }
  std::fclose(f);
}

// ── minimal safetensors writer (single file, F32 tensors) ────────────────────
struct StEntry {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> values;
};

inline void WriteSafetensors(const std::vector<StEntry>& entries, const std::string& path) {
  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const StEntry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i != 0) header += ",";
      header += std::to_string(e.shape[i]);
    }
    const size_t bytes = e.values.size() * sizeof(float);
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  std::string file;
  const uint64_t n = header.size();
  file.append(reinterpret_cast<const char*>(&n), sizeof(n));
  file += header;
  for (const StEntry& e : entries) {
    file.append(reinterpret_cast<const char*>(e.values.data()), e.values.size() * sizeof(float));
  }
  WriteFileBytes(path, file);
}

// ── the tiny DiT geometry (test_minimax_h3.cpp:3786) ─────────────────────────
struct FoldDitGeometry {
  int64_t num_layers = 2;
  int64_t token_refiner_num_layers = 1;
  int64_t hidden_size = 64;
  int64_t num_attention_heads = 4;
  int64_t attention_head_dim = 16;
  int64_t ffn_hidden_size = 128;
  int64_t latents_dim = 8;
  int64_t audio_latents_dim = 6;
  int64_t text_dim = 24;
  int64_t timestep_input_dim = 16;
  int64_t time_embed_hidden_size = 64;
  int64_t time_embed_dim = 32;
  int64_t adaln_out_features = 18 * 64;
  int64_t final_adaln_out_features = 2 * 64;
  int64_t rope_inv_freq_len = 2;
  // Derived (patch 1x2x2, the shipped patching): the packed video token width.
  int64_t video_row_width() const { return latents_dim * 1 * 2 * 2; }
};

// The render request the goldens were captured with (the pre-fold binary's
// exact flags). text_len is the prompt_embeds row count.
struct FoldRenderRequest {
  int64_t frames = 5;
  int64_t height = 32;
  int64_t width = 32;
  int64_t steps = 3;
  int64_t text_len = 4;
  const char* partition = "fl2va";  // serves t2va + fl2va (the #77 guard)
};

// ── write the ComfyUI-format F32 GGUF DiT ────────────────────────────────────
inline void WriteFoldDitGguf(const FoldDitGeometry& g, const std::string& path) {
  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "wan"));
  // GGUF stores `ne` REVERSED vs torch: a logical [out, in] weight is written
  // [in, out]. F32 everywhere — the fold gate is about the ASSEMBLY, not quant.
  auto add = [&](const std::string& name, const std::vector<int64_t>& logical) {
    int64_t numel = 1;
    for (const int64_t d : logical) numel *= d;
    const std::vector<float> values = Param("fold.dit." + name, numel, 0.1);
    std::string bytes(reinterpret_cast<const char*>(values.data()),
                      values.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = logical.rbegin(); it != logical.rend(); ++it) {
      ne.push_back(static_cast<uint64_t>(*it));
    }
    builder.AddTensor(name, ne, /*ggml_type=*/0 /*F32*/, bytes);
  };

  const int64_t inner = g.num_attention_heads * g.attention_head_dim;
  const int64_t video_width = g.video_row_width();
  add("video_patch_proj.weight", {g.hidden_size, video_width});
  add("video_patch_proj.bias", {g.hidden_size});
  add("audio_patch_proj.weight", {g.hidden_size, g.audio_latents_dim});
  add("audio_patch_proj.bias", {g.hidden_size});
  add("condition_proj.weight", {g.hidden_size, g.text_dim});
  add("condition_proj.bias", {g.hidden_size});
  add("time_embedder.proj_in.weight", {g.time_embed_hidden_size, g.timestep_input_dim});
  add("time_embedder.proj_in.bias", {g.time_embed_hidden_size});
  add("time_embedder.proj_out.weight", {g.time_embed_dim, g.time_embed_hidden_size});
  add("time_embedder.proj_out.bias", {g.time_embed_dim});
  add("rope.inv_freq", {g.rope_inv_freq_len});
  auto add_block = [&](const std::string& prefix, bool with_adaln) {
    add(prefix + ".norm1.weight", {g.hidden_size});
    add(prefix + ".norm2.weight", {g.hidden_size});
    add(prefix + ".attn.qkv_proj.weight", {3 * inner, g.hidden_size});
    add(prefix + ".attn.q_norm.weight", {g.attention_head_dim});
    add(prefix + ".attn.k_norm.weight", {g.attention_head_dim});
    add(prefix + ".attn.out_proj.weight", {g.hidden_size, inner});
    add(prefix + ".mlp.fc1.weight", {2 * g.ffn_hidden_size, g.hidden_size});
    add(prefix + ".mlp.fc2.weight", {g.hidden_size, g.ffn_hidden_size});
    if (with_adaln) {
      add(prefix + ".adaln_proj.linear.weight", {g.adaln_out_features, g.time_embed_dim});
      add(prefix + ".adaln_proj.linear.bias", {g.adaln_out_features});
    }
  };
  for (int64_t i = 0; i < g.token_refiner_num_layers; ++i) {
    add_block("token_refiner.blocks." + std::to_string(i), false);
  }
  add("token_refiner.final_norm.weight", {g.hidden_size});
  for (int64_t i = 0; i < g.num_layers; ++i) {
    add_block("blocks." + std::to_string(i), true);
  }
  add("final_layer.norm.weight", {g.hidden_size});
  add("final_layer.adaln_proj.linear.weight", {g.final_adaln_out_features, g.time_embed_dim});
  add("final_layer.adaln_proj.linear.bias", {g.final_adaln_out_features});
  add("final_layer.video_out.weight", {video_width, g.hidden_size});
  add("final_layer.video_out.bias", {video_width});
  add("final_layer.audio_out.weight", {g.audio_latents_dim, g.hidden_size});
  add("final_layer.audio_out.bias", {g.audio_latents_dim});

  WriteFileBytes(path, builder.Build());
}

// ── the reduced ViT3D video-VAE decoder + its config.json ────────────────────
// The config keys mirror what ParseMiniMaxH3VideoVaeDecoderConfig reads
// (minimax_h3_vae_loader.cpp:512): heads 2 x dim_head 8 => dim 16, ffn_mult 1,
// register tokens 2, rope_dim_ratio 0.75 => rope_apply_dim 6 — the reduced
// decoder of the e2e composition test. patch_size (16) and patch_size_t (4) are
// struct constants, not config keys, so the decoded canvas is latent*16 spatial
// and latent_t*4-(4-1) temporal (5 frames at latent_t 2).
inline void WriteFoldVideoVae(const FoldDitGeometry& g, const std::string& weights_path,
                              const std::string& config_path) {
  const int64_t dim = 16, heads = 2, dim_head = 8, ff_inner = 16, num_register = 2;
  std::vector<StEntry> entries;
  auto put = [&](const std::string& n, const std::vector<int64_t>& shape, double sc,
                 double off = 0.0) {
    int64_t numel = 1;
    for (const int64_t d : shape) numel *= d;
    entries.push_back({n, shape, Param("fold.vvae." + n, numel, sc, off)});
  };
  const int64_t inner = heads * dim_head;
  put("decoder.x_embedder.weight", {dim, g.latents_dim}, 0.1);
  put("decoder.x_embedder.bias", {dim}, 0.05);
  put("decoder.register_tokens", {num_register, dim}, 0.1);
  put("decoder.norm_out.weight", {dim}, 0.1, 1.0);
  put("decoder.norm_out.bias", {dim}, 0.05);
  const int64_t patch_dim = 3 * 4 * 16 * 16;  // out_channels * pt * ps * ps
  put("decoder.proj_out.weight", {patch_dim, dim}, 0.1);
  put("decoder.proj_out.bias", {patch_dim}, 0.05);
  const std::string b = "decoder.transformer_blocks.0.";
  put(b + "norm1.weight", {dim}, 0.1, 1.0);
  put(b + "norm2.weight", {dim}, 0.1, 1.0);
  put(b + "scale1", {dim}, 0.1);
  put(b + "scale2", {dim}, 0.1);
  put(b + "attn.to_qkv.weight", {3 * inner, dim}, 0.1);
  put(b + "attn.to_qkv.bias", {3 * inner}, 0.05);
  put(b + "attn.to_out.weight", {dim, inner}, 0.1);
  put(b + "attn.to_out.bias", {dim}, 0.05);
  put(b + "ff.w1.weight", {2 * ff_inner, dim}, 0.1);
  put(b + "ff.w1.bias", {2 * ff_inner}, 0.05);
  put(b + "ff.w2.weight", {dim, ff_inner}, 0.1);
  put(b + "ff.w2.bias", {dim}, 0.05);
  WriteSafetensors(entries, weights_path);

  std::string cfg = "{\n";
  cfg += "  \"decoder_num_layers\": 1,\n";
  cfg += "  \"latent_channels\": " + std::to_string(g.latents_dim) + ",\n";
  cfg += "  \"out_channels\": 3,\n";
  cfg += "  \"decoder_num_register_tokens\": 2,\n";
  cfg += "  \"decoder_num_attention_heads\": 2,\n";
  cfg += "  \"decoder_attention_head_dim\": 8,\n";
  cfg += "  \"decoder_ffn_mult\": 1,\n";
  cfg += "  \"decoder_norm_eps\": 1e-5,\n";
  cfg += "  \"decoder_rope_dim_ratio\": 0.75,\n";
  cfg += "  \"decoder_rope_theta\": 100.0,\n";
  cfg += "  \"latents_mean\": [";
  for (int64_t i = 0; i < g.latents_dim; ++i) {
    cfg += (i ? ", " : "") + std::string("0.0");
  }
  cfg += "],\n  \"latents_std\": [";
  for (int64_t i = 0; i < g.latents_dim; ++i) {
    cfg += (i ? ", " : "") + std::string("0.5");
  }
  cfg += "]\n}\n";
  WriteFileBytes(config_path, cfg);
}

// ── the reduced BigVGAN audio VAE + its config.json ──────────────────────────
// Mirrors ParseMiniMaxH3AudioVaeConfig (latent_dim => num_mels, decoder_dim,
// decoder_rates/kernel_sizes, resblock kernels/dilations) and the reduced
// weight set of the e2e composition test, under the checkpoint's `decoder.` /
// weight-norm naming the loader maps (minimax_h3_vae_loader.cpp:108).
inline void WriteFoldAudioVae(const FoldDitGeometry& g, const std::string& weights_path,
                              const std::string& config_path) {
  const int64_t num_mels = 8, decoder_dim = 8, ch = decoder_dim / 2;
  std::vector<StEntry> entries;
  auto put = [&](const std::string& n, const std::vector<int64_t>& shape, double sc,
                 double off = 0.0) {
    int64_t numel = 1;
    for (const int64_t d : shape) numel *= d;
    entries.push_back({n, shape, Param("fold.avae." + n, numel, sc, off)});
  };
  auto put_conv = [&](const std::string& prefix, int64_t oc, int64_t ic, int64_t k, bool bias) {
    put(prefix + ".parametrizations.weight.original0", {oc, 1, 1}, 0.03, 0.15);
    put(prefix + ".parametrizations.weight.original1", {oc, ic, k}, 0.08);
    if (bias) put(prefix + ".bias", {oc}, 0.05);
  };
  // dec_in_proj: audio_latents_dim -> num_mels (a PLAIN Conv1d k=1).
  put("decoder.dec_in_proj.weight", {num_mels, g.audio_latents_dim, 1}, 0.1);
  put("decoder.dec_in_proj.bias", {num_mels}, 0.05);
  put_conv("decoder.conv_pre", decoder_dim, num_mels, 7, true);
  put("decoder.ups.0.0.parametrizations.weight.original0", {decoder_dim, 1, 1}, 0.03, 0.15);
  put("decoder.ups.0.0.parametrizations.weight.original1", {decoder_dim, ch, 4}, 0.08);
  put("decoder.ups.0.0.bias", {ch}, 0.05);
  put_conv("decoder.resblocks.0.convs1.0", ch, ch, 3, true);
  put_conv("decoder.resblocks.0.convs2.0", ch, ch, 3, true);
  for (const char* a : {"decoder.resblocks.0.activations.0",
                        "decoder.resblocks.0.activations.1", "decoder.activation_post"}) {
    put(std::string(a) + ".act.alpha", {ch}, 0.2);
    put(std::string(a) + ".act.beta", {ch}, 0.2);
  }
  put_conv("decoder.conv_post", 1, ch, 7, false);
  WriteSafetensors(entries, weights_path);

  std::string cfg = "{\n";
  cfg += "  \"latent_dim\": " + std::to_string(num_mels) + ",\n";
  cfg += "  \"decoder_dim\": " + std::to_string(decoder_dim) + ",\n";
  cfg += "  \"decoder_rates\": [2],\n";
  cfg += "  \"decoder_kernel_sizes\": [4],\n";
  cfg += "  \"resblock_kernel_sizes\": [3],\n";
  cfg += "  \"resblock_dilation_sizes\": [[1]],\n";
  cfg += "  \"latents_mean\": [";
  for (int64_t i = 0; i < g.audio_latents_dim; ++i) {
    cfg += (i ? ", " : "") + std::string("0.0");
  }
  cfg += "],\n  \"latents_std\": [";
  for (int64_t i = 0; i < g.audio_latents_dim; ++i) {
    cfg += (i ? ", " : "") + std::string("0.5");
  }
  cfg += "]\n}\n";
  WriteFileBytes(config_path, cfg);
}

// ── prompt embeddings (rows of text_dim, little-endian f32) ──────────────────
inline void WriteFoldPromptEmbeds(const FoldDitGeometry& g, const FoldRenderRequest& r,
                                  const std::string& path) {
  const std::vector<float> values =
      Param("fold.prompt_embeds", r.text_len * g.text_dim, 1.0);
  std::string bytes(reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(float));
  WriteFileBytes(path, bytes);
}

// Write the whole fixture set under `dir` (which must exist).
// Files: dit.gguf, video_vae.safetensors, video_vae_config.json,
// audio_vae.safetensors, audio_vae_config.json, prompt_embeds.f32.
inline void WriteFoldFixture(const std::string& dir) {
  fs::create_directories(dir);
  const FoldDitGeometry g;
  const FoldRenderRequest r;
  WriteFoldDitGguf(g, dir + "/dit.gguf");
  WriteFoldVideoVae(g, dir + "/video_vae.safetensors", dir + "/video_vae_config.json");
  WriteFoldAudioVae(g, dir + "/audio_vae.safetensors", dir + "/audio_vae_config.json");
  WriteFoldPromptEmbeds(g, r, dir + "/prompt_embeds.f32");
}

}  // namespace minimax_h3_fold
