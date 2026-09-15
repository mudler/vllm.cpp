// GLM-5.3-Flash W9c-0/W9c-3 — the k-pool op probe and the device-resident
// compose forward. The probe half is always compiled; the compose half grows
// the TU from the 16-line stub into the full `Glm5NextDeviceForward`.
//
// The implementation follows the `kimi_linear_device.cpp` single-queue pattern:
// one `Dev`, one device queue, and host-fallback islands for the ops that have
// no portable `vt::` device provider.
//
// ─── ON DEVICE (vt:: dispatch) ─────────────────────────────────────────────
//   embedding              vt::Embedding
//   RMSNorm (all sites)    vt::RmsNorm (float acc, vs the host's double acc)
//   KDA recurrence          existing device arm in glm5_next_kda.cpp (dev passed through)
//   MoE routed experts      existing device arm in glm5_next_moe.cpp (dev passed through)
//   MoE combine             vt::MoeCombine (via the queue passed to MoeForward)
//   lm_head                 vt::MatmulBT (float acc, vs the host's chunked double)
//
// ─── HOST-FALLBACK ISLANDS (no portable device op) ─────────────────────────
//   (1) mHC sites — MhcPre/MhcPost per token. kDeepseekV4Mhc has no ROCm/CUDA
//       provider in the shared vt:: catalog, so the [T, hc, H] residual stream
//       lives on host because mHC wraps every sublayer.
//   (2) DSA/MLA attention — Attention() is a monolithic host function. The
//       k-pool indexer ops (kGlm5NextKpoolCompress/Select) are CUDA-only, so on
//       a CPU queue the indexer stays on host too.
//   (3) Dense+shared MLP — DenseMlpForward uses deepseek_v4::ClampedSwiGLU,
//       and no vt::ClampedSwiGLU device op exists. The dense MLP stays on host
//       until one is added.
//
// On a CPU queue the vt:: kernels use float32 accumulation where the host
// reference uses double, so the output agrees within a float-vs-double envelope
// rather than byte-exact. On a GPU the device kernels match upstream PyTorch's
// float32 numerics.
#include "vllm/model_executor/models/glm5_next_device.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_forward.h"
#include "vllm/model_executor/models/glm5_next_layer.h"
#include "vllm/model_executor/models/glm5_next_mhc.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_kda.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm::glm5_next {

bool KpoolDeviceOpsAvailable() {
  return vt::OpRegistered(vt::OpId::kGlm5NextKpoolCompress, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kGlm5NextKpoolSelect, vt::DeviceType::kCUDA);
}

namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::DType;
using vt::Tensor;

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("glm5_next device forward: " + what);
}

// Device-resident f32 weight view. On CPU this aliases the host bytes (host-
// pointer aliasing is a CPU property); on CUDA it would need a staging upload
// (the production path uses ResidentWeight for that — not wired here yet).
inline Tensor WF32(const Dev& d, const std::vector<float>& v,
                   const std::vector<int64_t>& shape) {
  return MakeTensor(const_cast<float*>(v.data()), DType::kF32, d.q.device, shape);
}

// Standalone RMSNorm on device: upload [T,H], vt::RmsNorm (float acc), download.
// The host reference uses double accumulation; the device kernel uses float.
// This is the primary source of numeric divergence on a CPU queue.
void DeviceRmsNorm(const Dev& d, float* out, const float* in,
                   const std::vector<float>& weight,
                   int64_t T, int64_t H, float eps) {
  DBuf din(d, DType::kF32, {T, H}, in);
  DBuf dout(d, DType::kF32, {T, H});
  Tensor w = WF32(d, weight, {H});
  vt::RmsNorm(d.q, dout.t(), din.t(), w, vt::RmsNormArgs{eps, /*gemma=*/false});
  dout.Download(d, out);
}

}  // namespace

std::vector<float> Glm5NextDeviceForward(
    const Glm5NextWeights& weights, const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& logits_indices, vt::Queue& queue,
    std::vector<LayerCache>* caches, int64_t lm_head_chunk_bytes) {
  const Glm5NextParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t hc = p.mhc.mult;
  const int64_t L = p.num_hidden_layers;
  const float eps = static_cast<float>(p.rms_norm_eps);
  (void)lm_head_chunk_bytes;  // reserved for production chunking; unused in the compose forward

  if (T <= 0) Fail("the step carries no tokens");
  if (H <= 0 || V <= 0) Fail("hidden_size or vocab_size is 0");
  if (hc <= 0) Fail("hc_mult must be > 0");

  // Build the device from the caller's queue. On a CPU queue this aliases host
  // memory; on CUDA it uploads/downloads. Reached via VT_GLM5_NEXT_DEVICE=1.
  Dev d{vt::GetBackend(queue.device), queue};

  // A CPU host queue for the host-fallback islands (mHC, DSA attention, dense
  // MLP). When the caller's queue is already CPU, reuse it.
  vt::Queue host_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Queue& hq = queue.device.type == vt::DeviceType::kCPU ? queue : host_queue;

  // ── embedding (ON DEVICE) ──────────────────────────────────────────────────
  // Decode the full table to f32, then vt::Embedding (a row gather). On CPU
  // this is byte-identical to the host forward's per-row decode. The full-table
  // decode works for the test geometry; a production path would chunk it
  // (DecodeOwnedTensorToF32 refuses > 1 GiB by name).
  const std::vector<float> embed_table =
      DecodeOwnedTensorToF32(weights.embed_tokens, "token_embd.weight");
  std::vector<float> embeds(static_cast<size_t>(T * H));
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor ttab = WF32(d, embed_table, {V, H});
    DBuf demb(d, DType::kF32, {T, H});
    vt::Embedding(d.q, demb.t(), ttab, dids.t());
    demb.Download(d, embeds.data());
  }

  // ── expand to hidden streams (host) ────────────────────────────────────────
  // The [T, H] embedding is broadcast to [T, hc, H] — the manifold every decoder
  // layer operates on. mHC has no device provider, so the stream stays on host.
  std::vector<float> streams = ExpandToHiddenStreams(embeds, 1, T, hc, H);

  // ── the mask (all ones for a single sequence) ───────────────────────────────
  const std::vector<uint8_t> mask(static_cast<size_t>(T), 1);

  // ── the final norm weight ──────────────────────────────────────────────────
  const std::vector<float> norm_w =
      DecodeOwnedTensorToF32(weights.norm, "output_norm.weight");

  // ── the layer source (streaming, one layer at a time) ───────────────────────
  Glm5NextGgufLayerSource layers(weights);

  // ── the layer loop ──────────────────────────────────────────────────────────
  // Mirrors DecoderLayerForward but swaps the host double-acc RmsNorm for the
  // device float-acc vt::RmsNorm. The mHC sites, DSA attention, and dense MLP
  // stay on host as islands.
  std::vector<int32_t> topk;
  int64_t topk_width = 0;

  for (int64_t i = 0; i < L; ++i) {
    const DecoderLayerWeights& w = layers.Layer(i);

    // ── mHC pre (attn site) — HOST ISLAND ────────────────────────────────
    // One MhcPre per token: collapses [hc, H] → [H] and retains the MhcPreResult
    // the matching MhcPost needs.
    std::vector<float> collapsed(static_cast<size_t>(T * H));
    std::vector<deepseek_v4::MhcPreResult> pre(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> slab(streams.begin() + t * hc * H,
                               streams.begin() + (t + 1) * hc * H);
      pre[static_cast<size_t>(t)] =
          MhcPre(slab, w.attn_hc, p.mhc, H, eps);
      std::copy_n(pre[static_cast<size_t>(t)].layer_input.data(),
                   static_cast<size_t>(H), collapsed.data() + t * H);
    }

    // ── RMSNorm (input_layernorm) — DEVICE ──────────────────────────────
    std::vector<float> normed(static_cast<size_t>(T * H));
    DeviceRmsNorm(d, normed.data(), collapsed.data(), w.input_layernorm,
                  T, H, eps);

    // ── attention — HOST ISLAND (KDA recurrence on device via dev) ──────
    std::vector<float> attn_out;
    if (w.attn_kind == Glm5NextLayerKind::kLinearAttention) {
      // Zero padded rows before the recurrence (no-op for a single sequence).
      for (int64_t t = 0; t < T; ++t) {
        if (mask[static_cast<size_t>(t)] != 0) continue;
        std::fill_n(normed.data() + t * H, static_cast<size_t>(H), 0.0F);
      }
      const glm5_next_kda::Glm5NextKdaDims kd = KdaDimsFrom(p);
      LayerCache* lc =
          caches != nullptr ? &(*caches)[static_cast<size_t>(i)] : nullptr;
      if (lc != nullptr && lc->kda.empty()) lc->kda.resize(1);
      attn_out.assign(static_cast<size_t>(T * H), 0.0F);
      const std::vector<float> out = glm5_next_kda::Glm5NextKdaLayerForward(
          w.kda, normed, kd, T,
          lc != nullptr ? &lc->kda[0] : nullptr, hq, &d);
      if (static_cast<int64_t>(out.size()) != T * H)
        Fail("KDA layer output size mismatch");
      std::copy_n(out.data(), static_cast<size_t>(T * H), attn_out.data());
      // KDA wipes the topk thread.
      topk.clear();
      topk_width = 0;
    } else {
      const MlaDims md = MlaDimsFrom(p);
      const IndexerDims idd = IndexerDimsFrom(p);
      const IndexerRole role = IndexerRoleFor(p, i);
      const IndexerWeights iw = w.dsa.IndexerView();
      const AttentionResult a = Attention(
          md, w.dsa.mla, idd, role.skip_topk ? nullptr : &iw, role, normed, mask,
          topk.empty() ? nullptr : &topk, topk_width, 1, T,
          caches != nullptr ? &(*caches)[static_cast<size_t>(i)].dsa : nullptr);
      attn_out = a.attn_output;
      if (a.propagates_topk) {
        topk = a.topk_indices;
        topk_width = a.topk_width;
      } else {
        topk.clear();
        topk_width = 0;
      }
    }
    if (static_cast<int64_t>(attn_out.size()) != T * H)
      Fail("attention output size mismatch");

    // ── mHC post (attn site) — HOST ISLAND ───────────────────────────────
    // Folds the sublayer's [H] output back onto the [hc, H] stream.
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<float> out(attn_out.begin() + t * H,
                                    attn_out.begin() + (t + 1) * H);
      const std::vector<float> resid(streams.begin() + t * hc * H,
                                      streams.begin() + (t + 1) * hc * H);
      const std::vector<float> mixed =
          MhcPost(out, resid, pre[static_cast<size_t>(t)], hc, H);
      std::copy_n(mixed.data(), static_cast<size_t>(hc * H),
                   streams.data() + t * hc * H);
    }

    // ── mHC pre (ffn site) — HOST ISLAND ─────────────────────────────────
    // The residual is the stream the attention fold just produced.
    const std::vector<float> ffn_residual = streams;
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> slab(ffn_residual.begin() + t * hc * H,
                               ffn_residual.begin() + (t + 1) * hc * H);
      pre[static_cast<size_t>(t)] =
          MhcPre(slab, w.ffn_hc, p.mhc, H, eps);
      std::copy_n(pre[static_cast<size_t>(t)].layer_input.data(),
                   static_cast<size_t>(H), collapsed.data() + t * H);
    }

    // ── RMSNorm (post_attention_layernorm) — DEVICE ─────────────────────
    DeviceRmsNorm(d, normed.data(), collapsed.data(),
                  w.post_attention_layernorm, T, H, eps);

    // ── MLP — HOST ISLAND (dense) / device arm (MoE) ────────────────────
    std::vector<float> mlp_out;
    if (w.mlp_kind == Glm5NextMlpKind::kDense) {
      mlp_out = DenseMlpForward(w.dense_mlp, normed, H, p.intermediate_size,
                                T, static_cast<float>(p.swiglu_limit));
    } else {
      mlp_out = MoeForward(MoeDimsFrom(p), w.moe, normed, T, hq, &d);
    }
    if (static_cast<int64_t>(mlp_out.size()) != T * H)
      Fail("MLP output size mismatch");

    // ── mHC post (ffn site) — HOST ISLAND ───────────────────────────────
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<float> out(mlp_out.begin() + t * H,
                                    mlp_out.begin() + (t + 1) * H);
      const std::vector<float> resid(ffn_residual.begin() + t * hc * H,
                                      ffn_residual.begin() + (t + 1) * hc * H);
      const std::vector<float> mixed =
          MhcPost(out, resid, pre[static_cast<size_t>(t)], hc, H);
      std::copy_n(mixed.data(), static_cast<size_t>(hc * H),
                   streams.data() + t * hc * H);
    }
  }

  // ── HcHeadCollapseMean — HOST ISLAND ───────────────────────────────────────
  // Unweighted mean over the stream axis: [T, hc, H] → [T, H].
  std::vector<float> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> slab(streams.begin() + t * hc * H,
                                   streams.begin() + (t + 1) * hc * H);
    const std::vector<float> col = HcHeadCollapseMean(slab, hc, H);
    std::copy_n(col.data(), static_cast<size_t>(H), hidden.data() + t * H);
  }

  // ── final RMSNorm — DEVICE ──────────────────────────────────────────────────
  DeviceRmsNorm(d, hidden.data(), hidden.data(), norm_w, T, H, eps);

  // ── logits gather (before lm_head) ─────────────────────────────────────────
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    std::iota(want.begin(), want.end(), int64_t{0});
  } else {
    want.reserve(logits_indices.size());
    for (int32_t idx : logits_indices) {
      if (idx < 0 || static_cast<int64_t>(idx) >= T)
        Fail("logits index " + std::to_string(idx) + " is outside [0, " +
             std::to_string(T) + ")");
      want.push_back(idx);
    }
  }

  // ── lm_head — DEVICE ────────────────────────────────────────────────────────
  // vt::MatmulBT with float acc (vs the host's chunked double acc). The full
  // head is decoded to f32; for the test geometry this is tiny. A production
  // path would chunk it to respect the 1 GiB decode ceiling.
  const OwnedTensor& head =
      weights.tied_word_embeddings ? weights.embed_tokens : weights.lm_head;
  const char* head_name =
      weights.tied_word_embeddings ? "token_embd.weight (tied head)" : "output.weight";
  const std::vector<float> head_f32 = DecodeOwnedTensorToF32(head, head_name);

  const int64_t n_out = static_cast<int64_t>(want.size());
  std::vector<float> logits(static_cast<size_t>(n_out) * static_cast<size_t>(V),
                             0.0f);
  {
    // Gather the rows we need.
    std::vector<float> gathered(static_cast<size_t>(n_out * H));
    for (int64_t r = 0; r < n_out; ++r) {
      std::copy_n(hidden.data() + static_cast<size_t>(want[static_cast<size_t>(r)]) * H,
                   static_cast<size_t>(H),
                   gathered.data() + static_cast<size_t>(r) * H);
    }
    DBuf dhid(d, DType::kF32, {n_out, H}, gathered.data());
    Tensor lm = WF32(d, head_f32, {V, H});
    DBuf dlog(d, DType::kF32, {n_out, V});
    vt::MatmulBT(d.q, dlog.t(), dhid.t(), lm);
    dlog.Download(d, logits.data());
  }

  return logits;
}

}  // namespace vllm::glm5_next
