// GLM-5.3-Flash W5b-2b — the engine binding. See `glm5_next_forward.h` for the
// residency ladder, the full-prefix decision and the one divergence from the
// house pattern.
#include "vllm/model_executor/models/glm5_next_forward.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/glm5_next_dsa.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vt/dtype.h"  // vt::DeviceType

namespace vllm::glm5_next {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error("glm5_next forward: " + why);
}

std::string LayerName(int64_t i) { return "blk." + std::to_string(i); }

int64_t BytesOf(const std::vector<float>& v) {
  return static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(float));
}

// What the slot actually cost, MEASURED from the decoded buffers rather than
// predicted from the dims — the same polarity `BridgedDsaLayer::host_f32_bytes`
// uses, and for the same reason: a predictor and the thing it predicts agreeing
// is a fact only when one of them was measured.
int64_t SlotF32Bytes(const DecoderLayerWeights& w) {
  int64_t n = BytesOf(w.input_layernorm) + BytesOf(w.post_attention_layernorm);
  for (const HcSite* s : {&w.attn_hc, &w.ffn_hc}) {
    n += BytesOf(s->fn) + BytesOf(s->base) + BytesOf(s->scale);
  }
  if (w.attn_kind == Glm5NextLayerKind::kLinearAttention) {
    const glm5_next_kda::Glm5NextKdaLayerWeights& k = w.kda;
    n += BytesOf(k.q_proj) + BytesOf(k.k_proj) + BytesOf(k.v_proj) +
         BytesOf(k.q_conv1d) + BytesOf(k.k_conv1d) + BytesOf(k.v_conv1d) +
         BytesOf(k.f_a_proj) + BytesOf(k.f_b_proj) + BytesOf(k.g_a_proj) +
         BytesOf(k.g_b_proj) + BytesOf(k.b_proj) + BytesOf(k.a_log) +
         BytesOf(k.dt_bias) + BytesOf(k.o_norm) + BytesOf(k.o_proj);
  } else {
    n += w.dsa.host_f32_bytes;
  }
  if (w.mlp_kind == Glm5NextMlpKind::kDense) {
    n += BytesOf(w.dense_mlp.gate_proj) + BytesOf(w.dense_mlp.up_proj) +
         BytesOf(w.dense_mlp.down_proj);
  } else {
    n += BytesOf(w.moe.router_weight) + BytesOf(w.moe.e_score_correction_bias) +
         BytesOf(w.moe.shared.gate_proj) + BytesOf(w.moe.shared.up_proj) +
         BytesOf(w.moe.shared.down_proj) +
         // The three banks are NOT bridged; these two are empty by construction
         // and are summed anyway so a change that filled them shows up here.
         BytesOf(w.moe.expert_gate_up) + BytesOf(w.moe.expert_down);
  }
  return n;
}

}  // namespace

// ─── the one-slot layer source ───────────────────────────────────────────────

Glm5NextGgufLayerSource::Glm5NextGgufLayerSource(const Glm5NextWeights& weights,
                                                 int64_t byte_ceiling)
    : weights_(&weights), byte_ceiling_(byte_ceiling) {}

int64_t Glm5NextGgufLayerSource::size() const {
  return static_cast<int64_t>(weights_->layers.size());
}

const DecoderLayerWeights& Glm5NextGgufLayerSource::Layer(int64_t layer_idx) {
  if (layer_idx < 0 || layer_idx >= size()) {
    Fail("layer " + std::to_string(layer_idx) + " is outside [0, " +
         std::to_string(size()) + "); the tower holds " +
         std::to_string(size()) + " decoder layers");
  }
  if (layer_idx == loaded_) return slot_;

  const Glm5NextParams& p = weights_->params;
  const Glm5NextLayerWeights& src = weights_->layers[static_cast<size_t>(layer_idx)];
  const std::string what = LayerName(layer_idx);

  // DROP THE PREVIOUS LAYER FIRST. Assigning over `slot_` would hold both for
  // the length of the bridge, which doubles the peak this whole design exists
  // to bound.
  slot_ = DecoderLayerWeights{};
  experts_.reset();
  loaded_ = -1;
  slot_f32_bytes_ = 0;

  DecoderLayerWeights out;
  // The SCHEDULE decides the arm, and the loader's per-layer flag and the
  // config's `layer_types` are two readings of one file. They are checked
  // against each other here rather than one being trusted:
  // `DecoderLayerForward` refuses a layer whose kind disagrees with
  // `p.layer_types[i]`, and a bridge that built the arm from the loader flag
  // alone would hand it a KDA arm labelled DSA.
  const Glm5NextLayerKind kind = src.is_linear_attention
                                     ? Glm5NextLayerKind::kLinearAttention
                                     : Glm5NextLayerKind::kDeepseekSparseAttention;
  if (kind != p.layer_types[static_cast<size_t>(layer_idx)]) {
    Fail(what + ": the loaded tower says this is a " +
         std::string(Glm5NextLayerKindName(kind)) +
         " layer but `config.layer_types[" + std::to_string(layer_idx) +
         "]` says " +
         std::string(Glm5NextLayerKindName(
             p.layer_types[static_cast<size_t>(layer_idx)])) +
         ". The file's own 46-entry `attention.head_count_kv` is the schedule; "
         "see .agents/specs/glm5-next-flash.md `## Owed` O18.");
  }
  const Glm5NextMlpKind mlp_kind =
      src.is_dense_mlp ? Glm5NextMlpKind::kDense : Glm5NextMlpKind::kSparse;
  if (mlp_kind != p.mlp_layer_types[static_cast<size_t>(layer_idx)]) {
    Fail(what + ": the loaded tower says this is a " +
         std::string(Glm5NextMlpKindName(mlp_kind)) +
         " feed-forward but `config.mlp_layer_types[" +
         std::to_string(layer_idx) + "]` says " +
         std::string(Glm5NextMlpKindName(
             p.mlp_layer_types[static_cast<size_t>(layer_idx)])) +
         ".");
  }
  out.attn_kind = kind;
  out.mlp_kind = mlp_kind;

  out.input_layernorm = DecodeOwnedTensorToF32(
      src.input_layernorm, what + ".attn_norm.weight", byte_ceiling_);
  out.post_attention_layernorm = DecodeOwnedTensorToF32(
      src.post_attention_layernorm, what + ".ffn_norm.weight", byte_ceiling_);
  out.attn_hc = BridgeMhcSite(src.attn_hc, p.mhc, p.hidden_size,
                              what + ".hc_attn", byte_ceiling_);
  out.ffn_hc = BridgeMhcSite(src.mlp_hc, p.mhc, p.hidden_size,
                             what + ".hc_ffn", byte_ceiling_);

  if (kind == Glm5NextLayerKind::kLinearAttention) {
    out.kda = BridgeKdaLayer(src.kda, KdaDimsFrom(p), byte_ceiling_);
  } else {
    out.dsa = BridgeDsaLayer(src.mla, MlaDimsFrom(p), IndexerDimsFrom(p),
                             byte_ceiling_);
  }

  if (mlp_kind == Glm5NextMlpKind::kDense) {
    out.dense_mlp = BridgeMlp(src.dense_mlp, p.hidden_size, p.intermediate_size,
                              what + ".ffn", byte_ceiling_);
  } else {
    const MoeDims md = MoeDimsFrom(p);
    out.moe = BridgeMoeLayer(src.moe, md, what + ".ffn", byte_ceiling_);
    // The per-expert source is OWNED here, so its lifetime is the slot's and
    // the borrowed pointer in `MoeLayerWeights` cannot outlive it. The three
    // banks stay in the file's own encoding and are never decoded whole.
    experts_ = std::make_unique<GgufExpertSource>(src.moe, md, what + ".ffn",
                                                  byte_ceiling_);
    out.moe.expert_source = experts_.get();
  }

  slot_ = std::move(out);
  loaded_ = layer_idx;
  ++bridged_;
  slot_f32_bytes_ = SlotF32Bytes(slot_);
  return slot_;
}

// ─── the forward ─────────────────────────────────────────────────────────────

std::vector<float> Glm5NextHostForward(const Glm5NextWeights& weights,
                                       const std::vector<int32_t>& token_ids,
                                       const std::vector<int32_t>& logits_indices,
                                       vt::Queue& queue,
                                       int64_t lm_head_chunk_bytes) {
  const Glm5NextParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T <= 0) Fail("the step carries no tokens");
  if (H <= 0 || V <= 0) {
    Fail("the resolved config has hidden_size " + std::to_string(H) +
         " and vocab_size " + std::to_string(V) + "; both must be > 0");
  }
  // Every buffer on this path is a host `std::vector<float>`, and
  // `vt::MoeRouterTopK` / `vt::MoeCombine` dispatch on the QUEUE's device. A
  // CUDA queue here hands a device kernel host pointers, which is a crash and
  // not a fallback, so it is refused by name. The device arm of this model is
  // owed; see .agents/specs/glm5-next-flash.md `## Owed`.
  if (queue.device.type != vt::DeviceType::kCPU) {
    Fail("this forward is a HOST f32 reference and was handed a non-CPU queue. "
         "Every glm5_next primitive on this row -- the KDA recurrence, the DSA "
         "indexer, the mHC blocks, the MoE router and the attention -- is host "
         "code, and `vt::MoeRouterTopK` dispatches on the queue's device, so a "
         "device queue here would hand a kernel host pointers. Run this model "
         "on the CPU device; the device arm is owed. See "
         ".agents/specs/glm5-next-flash.md and issue #2241.");
  }

  // ── the embedding gather (`:1447-1448`) ────────────────────────────────────
  //
  // ROW BY ROW out of the block-resident table. `token_embd.weight` is
  // [154880, 4096] on the published checkpoint, 2.36 GiB in f32, and
  // `DecodeOwnedTensorToF32` refuses it by name at the 1 GiB ceiling. What a
  // prompt needs is `T` of its 154,880 rows, at 16 KiB each.
  std::vector<float> embeds(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const int32_t tok = token_ids[static_cast<size_t>(t)];
    if (tok < 0 || static_cast<int64_t>(tok) >= V) {
      Fail("token id " + std::to_string(tok) + " at position " +
           std::to_string(t) + " is outside [0, " + std::to_string(V) + ")");
    }
    const std::vector<float> row = DecodeOwnedTensorRowsToF32(
        weights.embed_tokens, "token_embd.weight", tok, 1);
    if (static_cast<int64_t>(row.size()) != H) {
      Fail("token_embd.weight row " + std::to_string(tok) + " decoded " +
           std::to_string(row.size()) + " floats, expected " + std::to_string(H));
    }
    std::copy(row.begin(), row.end(), embeds.begin() + t * H);
  }

  // ── the stack ──────────────────────────────────────────────────────────────
  //
  // ONE sequence, no padding, so every mask entry is 1. `:1463-1470` guarantees
  // the mask exists for the indexer, which is why it is built rather than left
  // empty.
  const std::vector<uint8_t> mask(static_cast<size_t>(T), 1);
  const std::vector<float> norm =
      DecodeOwnedTensorToF32(weights.norm, "output_norm.weight");
  Glm5NextGgufLayerSource layers(weights);
  // `caches` is null: this path re-runs the whole prefix each step and owns no
  // state. See the header for why, and for what it costs.
  const std::vector<float> hidden =
      TextModelForward(p, norm, layers, embeds, mask, /*batch=*/1, /*seq_len=*/T,
                       /*caches=*/nullptr, queue);
  if (static_cast<int64_t>(hidden.size()) != T * H) {
    Fail("the text model returned " + std::to_string(hidden.size()) +
         " floats, expected " + std::to_string(T * H));
  }

  // ── the logits gather, BEFORE `lm_head` ────────────────────────────────────
  //
  // Empty means EVERY row (`nemotron_h.cpp`, `kimi_linear_forward.cpp`), and
  // the gather happens first so the head never runs on the full `T`.
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    std::iota(want.begin(), want.end(), int64_t{0});
  } else {
    want.reserve(logits_indices.size());
    for (int32_t idx : logits_indices) {
      if (idx < 0 || static_cast<int64_t>(idx) >= T) {
        Fail("logits index " + std::to_string(idx) + " is outside [0, " +
             std::to_string(T) + ")");
      }
      want.push_back(idx);
    }
  }

  // ── `lm_head`, streamed in row chunks ──────────────────────────────────────
  //
  // The head is [154880, 4096] and 2.36 GiB in f32; a chunk is
  // `kLmHeadChunkBytes` and is dropped before the next is decoded. TIE is read
  // off the FILE — `Glm5NextWeights::tied_word_embeddings` is set by whether
  // `output.weight` is present — because llama.cpp's writer decides it that way.
  const OwnedTensor& head =
      weights.tied_word_embeddings ? weights.embed_tokens : weights.lm_head;
  const char* head_name =
      weights.tied_word_embeddings ? "token_embd.weight (tied head)" : "output.weight";
  if (head.rank != 2 || head.shape[0] != V || head.shape[1] != H) {
    Fail(std::string(head_name) + " has rank " + std::to_string(head.rank) +
         " and shape [" + (head.rank > 0 ? std::to_string(head.shape[0]) : "") +
         (head.rank > 1 ? ", " + std::to_string(head.shape[1]) : "") +
         "], expected [" + std::to_string(V) + ", " + std::to_string(H) + "]");
  }
  if (lm_head_chunk_bytes <= 0) {
    Fail("lm_head_chunk_bytes must be > 0, got " +
         std::to_string(lm_head_chunk_bytes));
  }
  const int64_t row_bytes = HostF32RowBytes(head);
  const int64_t chunk_rows =
      std::max<int64_t>(1, row_bytes > 0 ? lm_head_chunk_bytes / row_bytes : V);

  std::vector<float> logits(want.size() * static_cast<size_t>(V), 0.0f);
  for (int64_t first = 0; first < V; first += chunk_rows) {
    const int64_t n = std::min(chunk_rows, V - first);
    const std::vector<float> chunk =
        DecodeOwnedTensorRowsToF32(head, head_name, first, n);
    for (size_t r = 0; r < want.size(); ++r) {
      const float* hr = &hidden[static_cast<size_t>(want[r] * H)];
      float* lr = &logits[r * static_cast<size_t>(V)];
      for (int64_t o = 0; o < n; ++o) {
        const float* wo = &chunk[static_cast<size_t>(o * H)];
        double acc = 0.0;
        for (int64_t i = 0; i < H; ++i) acc += static_cast<double>(wo[i]) * hr[i];
        lr[first + o] = static_cast<float>(acc);
      }
    }
  }
  return logits;
}

}  // namespace vllm::glm5_next
