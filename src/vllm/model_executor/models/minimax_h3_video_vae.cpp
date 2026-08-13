// MiniMax-H3 VIDEO VAE decoder block — the repeated unit of the ViT3D decoder.
//
// Like the audio VAE, the video VAE is checkpoint REMOTE CODE
// (`FL2VA/video_vae/*.py` under `trust_remote_code`), so it must be REIMPLEMENTED
// rather than adapted. The real 560-tensor manifest
// (tests/vllm/models/minimax_h3_video_vae_manifest.inc) shows the split:
//   * the ENCODER is the 3D CNN (116 tensors, rank-5 Conv3d down blocks);
//   * the DECODER — the only half generation needs — is a **36-block TRANSFORMER**
//     (440 tensors), which is what this file ports.
//
// Per block (base_module.py:200-281), all in fp32:
//   h = h + scale1 * Attention(RMSNorm(h))
//   h = h + scale2 * FeedForward(RMSNorm(h))
// with `scale1`/`scale2` LEARNED PER-CHANNEL VECTORS (not scalars), a gated SiLU
// feed-forward (`w1` produces [gate | up], `w2` projects back), and per-head RMS
// q/k normalization with NO affine weight.
//
// THE TRAP: this ViT's `to_qkv` output is PER-HEAD INTERLEAVED. Upstream does
// `qkv.view(B, S, -1, 3 * dim_head)` then `chunk(3, dim=-1)` (attention.py), so the
// layout is [head0_q | head0_k | head0_v | head1_q | ...] — NOT the
// [q_all | k_all | v_all] that the H3 DiT's fused qkv uses. Reading it the DiT way
// silently produces a plausible-but-wrong image.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <vector>

#include "vllm/support/platform_compat.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// nn.RMSNorm over the last dim, fp32 accumulation. `weight` may be empty, which
// is the qk-norm case (elementwise_affine=False).
void RmsNormLastDim(std::vector<float>& x, int64_t rows, int64_t width,
                    const std::vector<float>* weight, double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    float* row = x.data() + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(row[i]) * row[i];
    const double inv = 1.0 / std::sqrt(sum / static_cast<double>(width) + eps);
    for (int64_t i = 0; i < width; ++i) {
      double value = row[i] * inv;
      if (weight != nullptr) value *= (*weight)[static_cast<size_t>(i)];
      row[i] = static_cast<float>(value);
    }
  }
}

float SiluF(float x) { return x / (1.0f + std::exp(-x)); }

// y = x @ W^T + b, with W [out, in].
std::vector<float> LinearRows(const std::vector<float>& x, int64_t rows, int64_t in_features,
                              const std::vector<float>& weight, const std::vector<float>* bias,
                              int64_t out_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(o)] : 0.0;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_features + i)]) *
               static_cast<double>(weight[static_cast<size_t>(o * in_features + i)]);
      }
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

// One decoder TransformerBlock (base_module.py:200-281). `hidden` is [seq, dim].
std::vector<float> MiniMaxH3VideoVaeBlockForward(const MiniMaxH3VideoVaeBlockConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::string& prefix,
                                                 const std::vector<float>& hidden, int64_t seq,
                                                 const float* rope_cos, const float* rope_sin,
                                                 int64_t rot_dim) {
  const int64_t dim = config.dim;
  const int64_t heads = config.heads;
  const int64_t dim_head = config.dim_head;
  const int64_t inner = heads * dim_head;
  VT_CHECK(dim > 0 && heads > 0 && dim_head > 0, "minimax_h3 video vae: bad block geometry");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == seq * dim,
           "minimax_h3 video vae: hidden size does not match [seq, dim]");

  std::vector<float> h = hidden;

  // --- attention branch ---
  {
    std::vector<float> normed = h;
    RmsNormLastDim(normed, seq, dim, &weights.Get(prefix + ".norm1.weight"), config.eps);

    const std::vector<float> qkv =
        LinearRows(normed, seq, dim, weights.Get(prefix + ".attn.to_qkv.weight"),
                   &weights.Get(prefix + ".attn.to_qkv.bias"), 3 * inner);

    // PER-HEAD INTERLEAVED: [head][q|k|v] within each row.
    std::vector<float> q(static_cast<size_t>(seq * inner));
    std::vector<float> k(static_cast<size_t>(seq * inner));
    std::vector<float> v(static_cast<size_t>(seq * inner));
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t head = 0; head < heads; ++head) {
        const int64_t src = s * 3 * inner + head * 3 * dim_head;
        const int64_t dst = s * inner + head * dim_head;
        for (int64_t d = 0; d < dim_head; ++d) {
          q[static_cast<size_t>(dst + d)] = qkv[static_cast<size_t>(src + d)];
          k[static_cast<size_t>(dst + d)] = qkv[static_cast<size_t>(src + dim_head + d)];
          v[static_cast<size_t>(dst + d)] = qkv[static_cast<size_t>(src + 2 * dim_head + d)];
        }
      }
    }
    // qk RMSNorm has NO affine weight in this checkpoint.
    RmsNormLastDim(q, seq * heads, dim_head, nullptr, config.eps);
    RmsNormLastDim(k, seq * heads, dim_head, nullptr, config.eps);

    // 3D RoPE. cos/sin are per TOKEN (shared across heads) and cover only the
    // first rot_dim head dims; the rest pass through (func.py:82-102).
    if (rope_cos != nullptr && rot_dim > 0) {
      const int64_t half = rot_dim / 2;
      for (int64_t s = 0; s < seq; ++s) {
        const float* cos = rope_cos + s * rot_dim;
        const float* sin = rope_sin + s * rot_dim;
        for (int64_t head = 0; head < heads; ++head) {
          for (std::vector<float>* target : {&q, &k}) {
            float* values = target->data() + (s * heads + head) * dim_head;
            for (int64_t i = 0; i < half; ++i) {
              const float lo = values[i], hi = values[i + half];
              values[i] = lo * cos[i] - hi * sin[i];
              values[i + half] =
                  hi * cos[i + half] + lo * sin[i + half];
            }
          }
        }
      }
    }

    // Full (non-causal) attention per head.
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim_head));
    std::vector<float> attn(static_cast<size_t>(seq * inner), 0.0f);
    std::vector<double> probs(static_cast<size_t>(seq));
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t i = 0; i < seq; ++i) {
        double max_score = -1e30;
        for (int64_t j = 0; j < seq; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < dim_head; ++d) {
            dot += static_cast<double>(q[static_cast<size_t>((i * heads + head) * dim_head + d)]) *
                   static_cast<double>(k[static_cast<size_t>((j * heads + head) * dim_head + d)]);
          }
          probs[static_cast<size_t>(j)] = dot * scale;
          max_score = std::max(max_score, probs[static_cast<size_t>(j)]);
        }
        double denom = 0.0;
        for (int64_t j = 0; j < seq; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(probs[static_cast<size_t>(j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < dim_head; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < seq; ++j) {
            acc += probs[static_cast<size_t>(j)] *
                   static_cast<double>(v[static_cast<size_t>((j * heads + head) * dim_head + d)]);
          }
          attn[static_cast<size_t>((i * heads + head) * dim_head + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }

    const std::vector<float> projected =
        LinearRows(attn, seq, inner, weights.Get(prefix + ".attn.to_out.weight"),
                   &weights.Get(prefix + ".attn.to_out.bias"), dim);
    // scale1 is a learned PER-CHANNEL vector.
    const std::vector<float>& scale1 = weights.Get(prefix + ".scale1");
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        h[static_cast<size_t>(s * dim + d)] +=
            projected[static_cast<size_t>(s * dim + d)] * scale1[static_cast<size_t>(d)];
      }
    }
  }

  // --- feed-forward branch (gated SiLU) ---
  {
    std::vector<float> normed = h;
    RmsNormLastDim(normed, seq, dim, &weights.Get(prefix + ".norm2.weight"), config.eps);

    const int64_t ff_inner = config.ff_inner;
    const std::vector<float> fused =
        LinearRows(normed, seq, dim, weights.Get(prefix + ".ff.w1.weight"),
                   &weights.Get(prefix + ".ff.w1.bias"), 2 * ff_inner);
    std::vector<float> act(static_cast<size_t>(seq * ff_inner));
    for (int64_t s = 0; s < seq; ++s) {
      const float* row = fused.data() + s * 2 * ff_inner;
      for (int64_t i = 0; i < ff_inner; ++i) {
        // chunk(2) gives [gate, up]; the gate is the FIRST half.
        act[static_cast<size_t>(s * ff_inner + i)] = SiluF(row[i]) * row[ff_inner + i];
      }
    }
    const std::vector<float> projected =
        LinearRows(act, seq, ff_inner, weights.Get(prefix + ".ff.w2.weight"),
                   &weights.Get(prefix + ".ff.w2.bias"), dim);
    const std::vector<float>& scale2 = weights.Get(prefix + ".scale2");
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        h[static_cast<size_t>(s * dim + d)] +=
            projected[static_cast<size_t>(s * dim + d)] * scale2[static_cast<size_t>(d)];
      }
    }
  }
  return h;
}

// ---------------------------------------------------------------------------
// The whole ViT3D decoder (vae_vit.py:216-365)
// ---------------------------------------------------------------------------

// RotaryEmbeddingND (base_module.py:157-196) + create_token_ids
// (func.py:12-47, "length_normalized"): coordinates are (i + 0.5)/n scaled to
// [-1, 1), the angle scale is 2*pi (use_angle=True), and the per-axis frequency
// blocks are concatenated then TILED twice to fill rot_dim.
void MiniMaxH3VideoVaeRope(int64_t latent_t, int64_t latent_h, int64_t latent_w,
                           int64_t num_suffix, int64_t rope_apply_dim, double rope_theta,
                           std::vector<float>* cos_out, std::vector<float>* sin_out) {
  constexpr int64_t kNDim = 3;
  VT_CHECK(rope_apply_dim % (2 * kNDim) == 0,
           "minimax_h3 video vae: rope dim must be divisible by 2 * n_dim");
  const int64_t freqs = rope_apply_dim / (2 * kNDim);  // arange(0, 1, 2*n/dim) length
  const int64_t rot_dim = 2 * kNDim * freqs;           // == rope_apply_dim
  const int64_t patches = latent_t * latent_h * latent_w;
  const int64_t seq = patches + num_suffix;

  std::vector<double> inv_freq(static_cast<size_t>(freqs));
  for (int64_t i = 0; i < freqs; ++i) {
    const double step = static_cast<double>(2 * kNDim) / static_cast<double>(rope_apply_dim);
    inv_freq[static_cast<size_t>(i)] = 1.0 / std::pow(rope_theta, static_cast<double>(i) * step);
  }
  auto coord = [](int64_t index, int64_t size) {
    return 2.0 * ((static_cast<double>(index) + 0.5) / static_cast<double>(size)) - 1.0;
  };

  cos_out->assign(static_cast<size_t>(seq * rot_dim), 1.0f);
  sin_out->assign(static_cast<size_t>(seq * rot_dim), 0.0f);
  const int64_t half = kNDim * freqs;
  for (int64_t p = 0; p < patches; ++p) {
    const int64_t w = p % latent_w;
    const int64_t h = (p / latent_w) % latent_h;
    const int64_t t = p / (latent_w * latent_h);
    const std::array<double, 3> ids = {coord(t, latent_t), coord(h, latent_h), coord(w, latent_w)};
    for (int64_t axis = 0; axis < kNDim; ++axis) {
      for (int64_t i = 0; i < freqs; ++i) {
        const double angle = 2.0 * std::numbers::pi_v<double> *
                             ids[static_cast<size_t>(axis)] *
                             inv_freq[static_cast<size_t>(i)];
        const int64_t slot = axis * freqs + i;
        // .tile(2): the [3 * freqs] block is repeated to fill rot_dim.
        (*cos_out)[static_cast<size_t>(p * rot_dim + slot)] = static_cast<float>(std::cos(angle));
        (*sin_out)[static_cast<size_t>(p * rot_dim + slot)] = static_cast<float>(std::sin(angle));
        (*cos_out)[static_cast<size_t>(p * rot_dim + half + slot)] =
            static_cast<float>(std::cos(angle));
        (*sin_out)[static_cast<size_t>(p * rot_dim + half + slot)] =
            static_cast<float>(std::sin(angle));
      }
    }
  }
  // The suffix tokens (register tokens + cls) carry id 0 => cos 1, sin 0, which
  // the assign() above already established.
}

std::vector<float> MiniMaxH3VideoVaeDecode(const MiniMaxH3VideoVaeDecoderConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t latent_t,
                                           int64_t latent_h, int64_t latent_w,
                                           MiniMaxH3VideoFrameShape* out_shape) {
  const int64_t dim = config.block.dim;
  const int64_t patches = latent_t * latent_h * latent_w;
  const int64_t num_suffix = 1 + config.num_register_tokens;  // register tokens + cls
  const int64_t seq = patches + num_suffix;
  VT_CHECK(static_cast<int64_t>(latent.size()) == config.in_channels * patches,
           "minimax_h3 video vae: latent size does not match [C, T, H, W]");

  // _pack_tensors_3d(x, 1, 1): [C,T,H,W] -> [T*H*W, C] (channels last).
  std::vector<float> tokens(static_cast<size_t>(patches * config.in_channels));
  for (int64_t p = 0; p < patches; ++p) {
    for (int64_t c = 0; c < config.in_channels; ++c) {
      tokens[static_cast<size_t>(p * config.in_channels + c)] =
          latent[static_cast<size_t>(c * patches + p)];
    }
  }

  // x_embedder, then [patches | register_tokens | zero cls].
  std::vector<float> hidden(static_cast<size_t>(seq * dim), 0.0f);
  {
    const std::vector<float> embedded =
        LinearRows(tokens, patches, config.in_channels, weights.Get("x_embedder.weight"),
                   &weights.Get("x_embedder.bias"), dim);
    std::copy(embedded.begin(), embedded.end(), hidden.begin());
    const std::vector<float>& registers = weights.Get("register_tokens");
    VT_CHECK(static_cast<int64_t>(registers.size()) == config.num_register_tokens * dim,
             "minimax_h3 video vae: register_tokens size mismatch");
    std::copy(registers.begin(), registers.end(), hidden.begin() + patches * dim);
    // the cls token is an explicit ZERO row (vae_vit.py:311-313)
  }

  std::vector<float> cos, sin;
  MiniMaxH3VideoVaeRope(latent_t, latent_h, latent_w, num_suffix, config.rope_apply_dim,
                        config.rope_theta, &cos, &sin);

  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    hidden = MiniMaxH3VideoVaeBlockForward(
        config.block, weights, "transformer_blocks." + std::to_string(layer), hidden, seq,
        cos.data(), sin.data(), config.rope_apply_dim);
  }

  // norm_out is a LAYER norm here (not RMS): subtract the mean too.
  for (int64_t s = 0; s < seq; ++s) {
    float* row = hidden.data() + s * dim;
    double mean = 0.0;
    for (int64_t i = 0; i < dim; ++i) mean += row[i];
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t i = 0; i < dim; ++i) var += (row[i] - mean) * (row[i] - mean);
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + config.block.eps);
    const std::vector<float>& w = weights.Get("norm_out.weight");
    const bool has_bias = weights.Has("norm_out.bias");
    for (int64_t i = 0; i < dim; ++i) {
      double value = (row[i] - mean) * inv * w[static_cast<size_t>(i)];
      if (has_bias) value += weights.Get("norm_out.bias")[static_cast<size_t>(i)];
      row[i] = static_cast<float>(value);
    }
  }

  // proj_out, then drop the suffix rows.
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;
  const std::vector<float> projected =
      LinearRows(hidden, seq, dim, weights.Get("proj_out.weight"), &weights.Get("proj_out.bias"),
                 patch_dim);

  // _unpack_tensors_3d: [patches, C*pt*ps*ps] -> [C, T*pt, H*ps, W*ps].
  const int64_t pt = config.patch_size_t, ps = config.patch_size;
  const int64_t video_t = latent_t * pt, video_h = latent_h * ps, video_w = latent_w * ps;
  std::vector<float> frames(
      static_cast<size_t>(config.out_channels * video_t * video_h * video_w));
  for (int64_t ti = 0; ti < latent_t; ++ti) {
    for (int64_t hi = 0; hi < latent_h; ++hi) {
      for (int64_t wi = 0; wi < latent_w; ++wi) {
        const int64_t row = (ti * latent_h + hi) * latent_w + wi;
        int64_t k = 0;
        for (int64_t c = 0; c < config.out_channels; ++c) {
          for (int64_t r = 0; r < pt; ++r) {
            for (int64_t p = 0; p < ps; ++p) {
              for (int64_t q = 0; q < ps; ++q) {
                const int64_t dst =
                    ((c * video_t + ti * pt + r) * video_h + hi * ps + p) * video_w + wi * ps + q;
                frames[static_cast<size_t>(dst)] =
                    projected[static_cast<size_t>(row * patch_dim + k)];
                ++k;
              }
            }
          }
        }
      }
    }
  }
  if (out_shape != nullptr) {
    out_shape->channels = config.out_channels;
    out_shape->t = video_t;
    out_shape->h = video_h;
    out_shape->w = video_w;
  }
  return frames;
}

// ---------------------------------------------------------------------------
// Spatial tiling (klvae.py:192-250)
//
// The video VAE decodes large canvases in overlapping tiles and cross-fades the
// seams. The shipped config is tile_size 256 / tile_overlap_min 64 / vae_ratio 16
// (the product of `space_down` [2,2,2,2,1,1] — the "f16" in f16t4).
//
// The tile plan is not a simple stride: it picks the SMALLEST tile count whose
// minimum overlaps still cover the input, then distributes the leftover slack in
// whole `vae_ratio` units ROUND-ROBIN across the seams. Getting that distribution
// wrong shifts every tile after the first and shows up as seam artifacts, not as
// an error.
// ---------------------------------------------------------------------------

MiniMaxH3TilePlan MiniMaxH3SplitTiles(int64_t input_len, int64_t tile_size,
                                      int64_t tile_overlap_min, int64_t vae_ratio) {
  VT_CHECK(input_len > 0, "minimax_h3 tiling: input_len must be positive");
  VT_CHECK(tile_size > 0 && vae_ratio > 0, "minimax_h3 tiling: tile_size/vae_ratio must be positive");
  VT_CHECK(tile_overlap_min >= 0, "minimax_h3 tiling: tile_overlap_min must be >= 0");

  MiniMaxH3TilePlan plan;
  if (tile_size >= input_len) {
    // A single tile covers the whole axis; there are no seams to blend.
    plan.starts = {0};
    plan.lengths = {input_len};
    return plan;
  }

  int64_t n = (input_len + tile_size - 1) / tile_size;  // ceil
  int64_t remaining = 0;
  while (true) {
    remaining = tile_size * n - tile_overlap_min * (n - 1) - input_len;
    if (remaining >= 0) break;
    ++n;
  }
  plan.overlaps.assign(static_cast<size_t>(n - 1), tile_overlap_min);
  const int64_t remaining_units = remaining / vae_ratio;
  for (int64_t i = 0; i < remaining_units; ++i) {
    plan.overlaps[static_cast<size_t>(i % (n - 1))] += vae_ratio;
  }

  plan.starts.push_back(0);
  for (int64_t i = 0; i < n - 1; ++i) {
    plan.starts.push_back(plan.starts.back() + tile_size - plan.overlaps[static_cast<size_t>(i)]);
  }
  plan.lengths.assign(static_cast<size_t>(n), tile_size);
  return plan;
}

// blend (klvae.py:220-250): a linear cross-fade over `blend_extent` elements
// along one axis, followed by the remainder of `b`. `stride` is the distance
// between consecutive elements along that axis.
std::vector<float> MiniMaxH3BlendTiles(const std::vector<float>& a, const std::vector<float>& b,
                                       int64_t blend_extent) {
  const int64_t a_len = static_cast<int64_t>(a.size());
  const int64_t b_len = static_cast<int64_t>(b.size());
  blend_extent = std::min({a_len, b_len, blend_extent});
  VT_CHECK(blend_extent > 0, "minimax_h3 tiling: blend_extent must be positive");

  std::vector<float> out;
  out.reserve(static_cast<size_t>(b_len));
  for (int64_t i = 0; i < blend_extent; ++i) {
    const double wb = static_cast<double>(i) / static_cast<double>(blend_extent);
    const double wa = 1.0 - wb;
    out.push_back(static_cast<float>(a[static_cast<size_t>(a_len - blend_extent + i)] * wa +
                                     b[static_cast<size_t>(i)] * wb));
  }
  for (int64_t i = blend_extent; i < b_len; ++i) out.push_back(b[static_cast<size_t>(i)]);
  return out;
}

}  // namespace vllm
