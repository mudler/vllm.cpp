// MiniMax-Music3 — the DEVICE-RESIDENT acoustic forward. See
// minimax_music3_device.h for what is ported onto which shared op, why fp32
// stays fp32, and the three named reasons this arm is close to but not
// bit-identical to the host reference.
//
// ─── THE ONE STRUCTURAL DECISION IN THIS FILE ────────────────────────────────
//
// The reference works in CHANNEL-MAJOR [channels, length] at the two ends and
// FRAME-MAJOR [length, channels] in the middle, and transposes between them
// (minimax_music3_acoustic.cpp:558-564, :625-631). This file transposes ONCE on
// the host at each boundary and stays FRAME-MAJOR everywhere in between,
// because both 1x1 convolutions are then plain GEMMs:
//
//   conv1d(kernel=1):  out[co][t] = Σ_ci W[co][ci] * in[ci][t]
//   transposed:        out^T[t][co] = Σ_ci in^T[t][ci] * W[co][ci]
//                                   = MatmulBT(in^T, W)
//
// So `preprocess_conv` and `postprocess_conv` need no convolution op at all —
// which matters, because `vt` has no CUDA 1-D convolution provider (spec §11.4)
// and hand-rolling one outside the seam is what AGENTS.md forbids. The two
// host-side transposes it costs are [in_channels, length] tensors: 128 x length
// floats, against the 2.4G MACs per token the block stack runs.
#include "vllm/model_executor/models/minimax_music3_device.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/music3_profile.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"  // GetOp — the stage-time provider refusal
#include "vt/ops.h"

namespace vllm {
namespace models {
namespace music3 {

namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using vt::DType;
using vt::Tensor;

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

void RequireStageSize(const std::vector<float>& values, int64_t expected, const char* what) {
  if (static_cast<int64_t>(values.size()) != expected) {
    Fail(std::string("MiniMax-Music3 DiT stage: ") + what + " is " +
         std::to_string(values.size()) + " values, expected " + std::to_string(expected));
  }
}

// A rank-N f32 view over a fresh device block, owned by `storage`.
//
// The host source is copied and then, when `release` is set, DROPPED — the
// vector is swapped with an empty one rather than merely cleared, because
// `clear()` keeps the capacity and the whole point is to return the 9.7 GB to
// the allocator before the next tensor asks for its device twin.
Tensor UploadF32(vt::Backend& backend, vt::Queue& queue, std::vector<float>& src,
                 const std::vector<int64_t>& shape, bool release,
                 std::vector<std::shared_ptr<void>>* storage) {
  int64_t numel = 1;
  for (int64_t s : shape) numel *= s;
  const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, src.data(), bytes);
  // The copy must have LANDED before the host buffer goes away. On a CPU queue
  // this is a memcpy and the sync is free; on CUDA the source is pageable host
  // memory, so releasing it under an unsynchronized async copy is exactly the
  // use-after-free that reads as a plausible-looking wrong tensor.
  if (release) {
    backend.Synchronize(queue);
    std::vector<float>().swap(src);
  }
  storage->push_back(std::move(owner));
  return MakeTensor(p, DType::kF32, queue.device, shape);
}

// `vt::MatmulBT` + an optional rank-1 row-broadcast bias — the device twin of
// the reference `Linear`. The bias is a SEPARATE add here where the reference
// seeds the accumulator with it; that is difference (2) in the header's
// numerics note, and it is a float32 rounding, not a reordering of the sum.
void LinearDev(vt::Queue& q, Tensor& out, const Tensor& in, const Tensor& weight,
               const Tensor* bias) {
  vt::MatmulBT(q, out, in, weight);
  if (bias != nullptr && bias->data != nullptr) vt::Add(q, out, out, *bias);
}


// ─── THE INTRA-FORWARD SPANS (spec §21.3) ────────────────────────────────────
//
// `denoise.dit_device` (minimax_music3_speech.cpp:321) is ONE leaf over the whole
// forward, and spec §20 measured it at 62.24 % of the developer's run with
// nothing known about its inside. These marks split it.
//
// THEY ARE SPANS, NOT LEAVES. `music3_profile.h` sums leaves and only prints
// spans, so `denoise.dit_device` stays the leaf that partitions the run,
// `sum(leaf)` does not move, `unattributed` stays a real quantity, and §15.7's
// and §20's tables remain comparable value for value.
//
// THEY NEED A SYNCHRONIZE, WHICH IS WHY THEY ARE A SECOND OPT-IN — see
// `profile::DitSpans` (music3_profile.h) for the argument. With that flag unset
// the `span_backend` below is null and every mark is one predicted branch that
// reads no clock and drains no queue, so the default profiled path is byte for
// byte the one spec §20 timed.

// One span boundary: drain the queue, charge everything since `t0` to `name`, and
// return the instant the NEXT span starts from. Contiguous by construction — the
// caller threads the returned timestamp into the next call — so the spans
// partition the forward instead of sampling it, and a bracket somebody forgets to
// place shows up as an inflated neighbour rather than as silence.
std::chrono::steady_clock::time_point DitMark(vt::Backend* backend, vt::Queue& queue,
                                              const char* name,
                                              std::chrono::steady_clock::time_point t0) {
  if (backend == nullptr) return t0;
  backend->Synchronize(queue);
  profile::AddSince(name, t0, /*span=*/true);
  return profile::Now();
}

}  // namespace

Music3DitDeviceWeights StageMusic3DitWeights(vt::Queue& queue,
                                             const MiniMaxMusic3TransformerConfig& config,
                                             DitWeights& weights, bool release_host) {
  const int64_t in_channels = config.in_channels;
  const int64_t concat = config.concat_channels();
  const int64_t inner = config.inner_dim();
  const int64_t attn_inner = config.num_attention_heads * config.attention_head_dim;
  const int64_t ff = config.ff_inner_dim;
  const int64_t fourier = config.fourier_embedding_dim;
  if (static_cast<int64_t>(weights.layers.size()) != config.num_layers) {
    Fail("MiniMax-Music3 DiT stage: the weights carry " +
         std::to_string(weights.layers.size()) + " blocks, the config declares " +
         std::to_string(config.num_layers));
  }

  // REFUSE UP FRONT, before 9.7 GB moves. `vt::GetOp` throws naming the op and
  // the device, so a backend without (say) a cross-attention provider is a
  // one-line refusal at stage time rather than a failure 36 layers into the
  // first of 660 forwards. Every op the forward below calls is listed.
  for (vt::OpId op : {vt::OpId::kMatmulBT, vt::OpId::kAdd, vt::OpId::kLayerNorm,
                      vt::OpId::kSiluAndMul, vt::OpId::kRopeFromCache,
                      vt::OpId::kAttentionCross}) {
    (void)vt::GetOp(op, queue.device.type);
  }

  // The timestep embedder is validated FIRST, before anything is uploaded and
  // therefore before `release_host` destroys anything. A size error found after
  // the 36 blocks had been released would be a correct refusal that had already
  // consumed the caller's weights.
  const int64_t fourier_half = fourier / 2;
  RequireStageSize(weights.time_proj_weight, fourier_half, "time_proj.weight");
  RequireStageSize(weights.time_embed_linear_1_weight, inner * fourier,
                   "time_embed.linear_1.weight");
  RequireStageSize(weights.time_embed_linear_1_bias, inner, "time_embed.linear_1.bias");
  RequireStageSize(weights.time_embed_linear_2_weight, inner * inner,
                   "time_embed.linear_2.weight");
  RequireStageSize(weights.time_embed_linear_2_bias, inner, "time_embed.linear_2.bias");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Music3DitDeviceWeights staged;
  staged.layers.resize(static_cast<size_t>(config.num_layers));
  const bool rel = release_host;
  auto up = [&](std::vector<float>& src, const std::vector<int64_t>& shape, const char* what) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    RequireStageSize(src, numel, what);
    return UploadF32(backend, queue, src, shape, rel, &staged.storage);
  };

  // The 1x1 convolutions ship as [out, in, 1]; the kernel axis is dropped here
  // because the GEMM form above consumes them as [out, in]. The element count
  // is identical, so this is a reinterpretation of the same bytes, not a slice.
  staged.preprocess_conv_weight =
      up(weights.preprocess_conv_weight, {concat, concat}, "preprocess_conv.weight");
  staged.proj_in_weight = up(weights.proj_in_weight, {inner, concat}, "proj_in.weight");

  for (int64_t l = 0; l < config.num_layers; ++l) {
    DitLayerWeights& src = weights.layers[static_cast<size_t>(l)];
    Music3DitDeviceLayer& dst = staged.layers[static_cast<size_t>(l)];
    dst.norm1_weight = up(src.norm1_weight, {inner}, "norm1.weight");
    dst.norm1_bias = up(src.norm1_bias, {inner}, "norm1.bias");
    dst.to_q = up(src.to_q, {attn_inner, inner}, "attn.to_q.weight");
    dst.to_k = up(src.to_k, {attn_inner, inner}, "attn.to_k.weight");
    dst.to_v = up(src.to_v, {attn_inner, inner}, "attn.to_v.weight");
    dst.to_out = up(src.to_out, {inner, attn_inner}, "attn.to_out.0.weight");
    dst.norm2_weight = up(src.norm2_weight, {inner}, "norm2.weight");
    dst.norm2_bias = up(src.norm2_bias, {inner}, "norm2.bias");

    // THE HALF SWAP (minimax_music3_device.h documents why). Upstream computes
    // `value * silu(gate)` with value FIRST; `vt::SiluAndMul` computes
    // `silu(first) * second`. Exchanging the two row blocks of the projection
    // and the two halves of its bias — once, here — makes the shared op compute
    // upstream's expression exactly. Doing it at stage time rather than per step
    // is what keeps it free: 660 forwards x 36 layers would otherwise permute a
    // [seq, 16384] tensor 23 760 times.
    RequireStageSize(src.ff_in_weight, 2 * ff * inner, "ff.net.0.proj.weight");
    RequireStageSize(src.ff_in_bias, 2 * ff, "ff.net.0.proj.bias");
    {
      std::vector<float> swapped(static_cast<size_t>(2 * ff * inner));
      const size_t half = static_cast<size_t>(ff * inner);
      std::memcpy(swapped.data(), src.ff_in_weight.data() + half, half * sizeof(float));
      std::memcpy(swapped.data() + half, src.ff_in_weight.data(), half * sizeof(float));
      if (rel) std::vector<float>().swap(src.ff_in_weight);
      dst.ff_in_weight = up(swapped, {2 * ff, inner}, "ff.net.0.proj.weight (swapped)");
    }
    {
      std::vector<float> swapped(static_cast<size_t>(2 * ff));
      const size_t half = static_cast<size_t>(ff);
      std::memcpy(swapped.data(), src.ff_in_bias.data() + half, half * sizeof(float));
      std::memcpy(swapped.data() + half, src.ff_in_bias.data(), half * sizeof(float));
      if (rel) std::vector<float>().swap(src.ff_in_bias);
      dst.ff_in_bias = up(swapped, {2 * ff}, "ff.net.0.proj.bias (swapped)");
    }
    dst.ff_out_weight = up(src.ff_out_weight, {inner, ff}, "ff.net.2.weight");
    dst.ff_out_bias = up(src.ff_out_bias, {inner}, "ff.net.2.bias");
  }

  staged.proj_out_weight = up(weights.proj_out_weight, {in_channels, inner}, "proj_out.weight");
  staged.postprocess_conv_weight =
      up(weights.postprocess_conv_weight, {in_channels, in_channels}, "postprocess_conv.weight");

  // The timestep embedder stays on the host — see the header. These are COPIES,
  // so `release_host` does not take them: they are 18 MB at the shipped
  // dimensions and they are what keeps `temb` bit-identical to the CPU arm.
  // Their sizes were checked at the top of this function.
  staged.host_time_embed.time_proj_weight = weights.time_proj_weight;
  staged.host_time_embed.time_embed_linear_1_weight = weights.time_embed_linear_1_weight;
  staged.host_time_embed.time_embed_linear_1_bias = weights.time_embed_linear_1_bias;
  staged.host_time_embed.time_embed_linear_2_weight = weights.time_embed_linear_2_weight;
  staged.host_time_embed.time_embed_linear_2_bias = weights.time_embed_linear_2_bias;

  backend.Synchronize(queue);
  return staged;
}

std::vector<float> DitForwardDevice(vt::Queue& queue, const std::vector<float>& latents,
                                    int64_t length, const std::vector<float>& condition,
                                    double timestep,
                                    const MiniMaxMusic3TransformerConfig& config,
                                    const Music3DitDeviceWeights& weights) {
  if (length <= 0) {
    Fail("MiniMax-Music3 DiT: a window of " + std::to_string(length) +
         " latent frames has nothing to denoise");
  }
  const int64_t in_channels = config.in_channels;
  const int64_t condition_dim = config.condition_dim;
  const int64_t concat = config.concat_channels();
  const int64_t inner = config.inner_dim();
  const int64_t heads = config.num_attention_heads;
  const int64_t head_dim = config.attention_head_dim;
  const int64_t attn_inner = heads * head_dim;
  const int64_t ff = config.ff_inner_dim;
  const int64_t seq = length + 1;
  if (static_cast<int64_t>(latents.size()) != in_channels * length) {
    Fail("MiniMax-Music3 DiT: latents [in_channels, length] is " +
         std::to_string(latents.size()) + " values, expected " +
         std::to_string(in_channels * length));
  }
  if (static_cast<int64_t>(condition.size()) != length * condition_dim) {
    Fail("MiniMax-Music3 DiT: condition [length, condition_dim] is " +
         std::to_string(condition.size()) + " values, expected " +
         std::to_string(length * condition_dim));
  }
  if (static_cast<int64_t>(weights.layers.size()) != config.num_layers) {
    Fail("MiniMax-Music3 DiT: the staged weights carry " +
         std::to_string(weights.layers.size()) + " blocks, the config declares " +
         std::to_string(config.num_layers));
  }

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};

  // Null unless VLLM_CPP_MUSIC3_DIT_SPANS=1 (and the profiler itself is armed).
  vt::Backend* const span_backend = profile::DitSpans() ? &backend : nullptr;
  std::chrono::steady_clock::time_point span_t = profile::Now();
  if (span_backend != nullptr) {
    // The window geometry, so the split is read against the shape it was taken
    // at rather than against §21.1's inference from the vocoder's latent count.
    // Both are SUMS over the forwards in the run: divide by `dit.pack`'s call
    // count, which is one per forward.
    profile::Count("dit.seq_sum", seq);
    profile::Count("dit.length_sum", length);
  }

  // `cat((hidden_states, zeros_like(hidden_states), encoder_hidden_states.T))`
  // (:218-219), built directly in the TRANSPOSED [length, concat] orientation.
  // The middle block is a genuine ZERO PAD and not a second copy of the latents.
  std::vector<float> stacked_t(static_cast<size_t>(length * concat), 0.0f);
  for (int64_t t = 0; t < length; ++t) {
    float* row = stacked_t.data() + t * concat;
    for (int64_t c = 0; c < in_channels; ++c) {
      row[c] = latents[static_cast<size_t>(c * length + t)];
    }
    for (int64_t c = 0; c < condition_dim; ++c) {
      row[2 * in_channels + c] = condition[static_cast<size_t>(t * condition_dim + c)];
    }
  }

  span_t = DitMark(span_backend, queue, "dit.pack", span_t);

  DBuf stacked(d, DType::kF32, {length, concat}, stacked_t.data());
  // RESIDUAL 1x1 convolution (:220), as the transposed GEMM this file's header
  // note derives. `pre` then holds conv(x) and the add makes it conv(x) + x.
  DBuf pre(d, DType::kF32, {length, concat});
  vt::MatmulBT(queue, pre.t(), stacked.t(), weights.preprocess_conv_weight);
  vt::Add(queue, pre.t(), pre.t(), stacked.t());

  // The timestep embedding is PREPENDED as one extra token (:227) that the
  // rotary sees and `proj_out` then drops (:236). `hidden` is allocated at the
  // full [seq, inner] and `proj_in` writes STRAIGHT into rows 1..seq — a view,
  // not a copy, so the projection lands where the block stack wants it.
  DBuf hidden(d, DType::kF32, {seq, inner});
  Tensor hidden_tail = MakeTensor(static_cast<float*>(hidden.t().data) + inner, DType::kF32,
                                  queue.device, {length, inner});
  vt::MatmulBT(queue, hidden_tail, pre.t(), weights.proj_in_weight);
  span_t = DitMark(span_backend, queue, "dit.pre", span_t);

  // Row 0: the timestep embedding, computed on the host through the reference's
  // own helpers so it is bit-identical to the CPU arm (header rationale).
  const std::vector<float> temb = DitTimestepEmbedding(
      FourierTimeEmbedding(timestep, weights.host_time_embed.time_proj_weight,
                           config.fourier_embedding_dim),
      config, weights.host_time_embed);
  if (static_cast<int64_t>(temb.size()) != inner) {
    Fail("MiniMax-Music3 DiT: the timestep embedding is " + std::to_string(temb.size()) +
         " values, expected inner_dim = " + std::to_string(inner));
  }
  backend.Copy(queue, hidden.t().data, temb.data(), static_cast<size_t>(inner) * sizeof(float));
  span_t = DitMark(span_backend, queue, "dit.temb", span_t);

  // ── the rotary cache, in the layout vt::RopeFromCache reads ────────────────
  // That op indexes `cache[position * rotary_dim + pair]` for the cosine and
  // `+ half + pair` for the sine, then computes x' = x*c - y*s, y' = x*s + y*c
  // over the LEADING rotary_dim of each head — which is `ApplyPartialRotary`
  // exactly (minimax_music3_acoustic.cpp:500-514), including the untouched tail.
  // `BuildDitRotaryTables` returns cos/sin already duplicated across the two
  // halves of the rotary window; the cache wants ONE half of each, so the first
  // `half` columns of each table are what is packed here.
  const int64_t rotary_dim = config.rotary_dim;
  const int64_t half = rotary_dim / 2;
  const DitRotaryTables tables = BuildDitRotaryTables(seq, rotary_dim, kDitRotaryTheta);
  std::vector<float> cache(static_cast<size_t>(seq * rotary_dim));
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t j = 0; j < half; ++j) {
      cache[static_cast<size_t>(s * rotary_dim + j)] =
          tables.cos[static_cast<size_t>(s * rotary_dim + j)];
      cache[static_cast<size_t>(s * rotary_dim + half + j)] =
          tables.sin[static_cast<size_t>(s * rotary_dim + j)];
    }
  }
  std::vector<int32_t> positions_host(static_cast<size_t>(seq));
  for (int64_t s = 0; s < seq; ++s) positions_host[static_cast<size_t>(s)] = static_cast<int32_t>(s);
  DBuf rope_cache(d, DType::kF32, {seq, rotary_dim}, cache.data());
  DBuf positions(d, DType::kI32, {seq}, positions_host.data());

  vt::RopeArgs rope_args;
  rope_args.rotary_dim = static_cast<int>(rotary_dim);
  rope_args.is_neox_style = true;  // rotate_half over the rotary window
  vt::LayerNormArgs norm_args;
  norm_args.eps = static_cast<float>(kDitLayerNormEps);
  vt::AttentionCrossArgs attn_args;
  attn_args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  span_t = DitMark(span_backend, queue, "dit.rope_build", span_t);

  // The reference's `Attention` helper takes NO mask (:97-103 dispatches with
  // none), so every token attends to every token including the prepended
  // timestep one. `vt::AttentionCross` with a null bias is that op; `vt::
  // Attention` is the CAUSAL one and would silently mask the future here.
  for (int64_t l = 0; l < config.num_layers; ++l) {
    const Music3DitDeviceLayer& layer = weights.layers[static_cast<size_t>(l)];

    DBuf normed(d, DType::kF32, {seq, inner});
    vt::LayerNorm(queue, normed.t(), hidden.t(), &layer.norm1_weight, &layer.norm1_bias,
                  norm_args);
    span_t = DitMark(span_backend, queue, "dit.norm1", span_t);

    DBuf qb(d, DType::kF32, {seq, attn_inner});
    DBuf kb(d, DType::kF32, {seq, attn_inner});
    DBuf vb(d, DType::kF32, {seq, attn_inner});
    LinearDev(queue, qb.t(), normed.t(), layer.to_q, nullptr);
    LinearDev(queue, kb.t(), normed.t(), layer.to_k, nullptr);
    LinearDev(queue, vb.t(), normed.t(), layer.to_v, nullptr);

    span_t = DitMark(span_backend, queue, "dit.qkv", span_t);

    Tensor q3 = Reshape(qb.t(), {seq, heads, head_dim});
    Tensor k3 = Reshape(kb.t(), {seq, heads, head_dim});
    Tensor v3 = Reshape(vb.t(), {seq, heads, head_dim});
    vt::RopeFromCache(queue, q3, &k3, positions.t(), rope_cache.t(), rope_args);

    span_t = DitMark(span_backend, queue, "dit.rope", span_t);

    DBuf attended(d, DType::kF32, {seq, heads, head_dim});
    vt::AttentionCross(queue, attended.t(), q3, k3, v3, nullptr, attn_args);
    span_t = DitMark(span_backend, queue, "dit.attn", span_t);

    Tensor attended2 = Reshape(attended.t(), {seq, attn_inner});
    DBuf projected(d, DType::kF32, {seq, inner});
    LinearDev(queue, projected.t(), attended2, layer.to_out, nullptr);
    vt::Add(queue, hidden.t(), hidden.t(), projected.t());

    span_t = DitMark(span_backend, queue, "dit.attn_out", span_t);

    DBuf normed2(d, DType::kF32, {seq, inner});
    vt::LayerNorm(queue, normed2.t(), hidden.t(), &layer.norm2_weight, &layer.norm2_bias,
                  norm_args);
    span_t = DitMark(span_backend, queue, "dit.norm2", span_t);

    DBuf gated(d, DType::kF32, {seq, 2 * ff});
    LinearDev(queue, gated.t(), normed2.t(), layer.ff_in_weight, &layer.ff_in_bias);
    // `silu(first) * second` on the SWAPPED projection == upstream's
    // `value * silu(gate)`. See StageMusic3DitWeights.
    span_t = DitMark(span_backend, queue, "dit.ff_in", span_t);

    DBuf activated(d, DType::kF32, {seq, ff});
    vt::SiluAndMul(queue, activated.t(), gated.t());
    span_t = DitMark(span_backend, queue, "dit.silu", span_t);

    DBuf ff_out(d, DType::kF32, {seq, inner});
    LinearDev(queue, ff_out.t(), activated.t(), layer.ff_out_weight, &layer.ff_out_bias);
    vt::Add(queue, hidden.t(), hidden.t(), ff_out.t());
    span_t = DitMark(span_backend, queue, "dit.ff_out", span_t);
  }

  // Drop the timestep token (the same [length, inner] view `proj_in` wrote),
  // project, then the RESIDUAL 1x1 convolution (:238) as a GEMM again.
  DBuf out_rows(d, DType::kF32, {length, in_channels});
  vt::MatmulBT(queue, out_rows.t(), hidden_tail, weights.proj_out_weight);
  DBuf post(d, DType::kF32, {length, in_channels});
  vt::MatmulBT(queue, post.t(), out_rows.t(), weights.postprocess_conv_weight);
  vt::Add(queue, out_rows.t(), out_rows.t(), post.t());

  std::vector<float> rows(static_cast<size_t>(length * in_channels));
  span_t = DitMark(span_backend, queue, "dit.post", span_t);
  backend.Copy(queue, rows.data(), out_rows.t().data, rows.size() * sizeof(float));
  backend.Synchronize(queue);
  span_t = DitMark(span_backend, queue, "dit.readback", span_t);

  // Back to the CHANNEL-MAJOR [in_channels, length] the caller and the vocoder
  // both expect.
  std::vector<float> out(static_cast<size_t>(in_channels * length));
  for (int64_t c = 0; c < in_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      out[static_cast<size_t>(c * length + t)] = rows[static_cast<size_t>(t * in_channels + c)];
    }
  }
  span_t = DitMark(span_backend, queue, "dit.untranspose", span_t);
  (void)span_t;
  return out;
}

}  // namespace music3
}  // namespace models
}  // namespace vllm
