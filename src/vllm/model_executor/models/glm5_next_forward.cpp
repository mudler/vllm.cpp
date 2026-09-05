// GLM-5.3-Flash W5b-2b — the engine binding. See `glm5_next_forward.h` for the
// residency ladder, the full-prefix decision and the one divergence from the
// house pattern.
#include "vllm/model_executor/models/glm5_next_forward.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // std::getenv
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/glm5_next_diag.h"
#include "vllm/model_executor/models/glm5_next_dsa.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vt/backend.h"      // vt::TryGetBackend
#include "vt/op_provider.h"  // vt::OpRegistered
#include "vt/dtype.h"    // vt::DeviceType

namespace vllm::glm5_next {
namespace {

// W9c-3a's device split is OPT-IN and DEFAULTS OFF, and the reason is a
// measurement rather than caution.
//
// Driven on `dgx:gpu0` against the real 101.24 GiB UD-Q2_K_XL artifact, all three
// `--device cuda` legs died with **SIGSEGV** (rc=139) having emitted no token,
// where `origin/main` produced a clean named refusal in 1066 s. The crash is
// reproducible (3 of 3, against 3 of 3 CPU legs emitting ` Paris.` on the same
// binary, interleaved) and it appears only in the MIXED residency state this
// row's per-layer fit guard creates: each leg logged the device arm engaging
// for one layer and then `DeviceBanksFit` declining a later one, and died after
// that. 94.6758 GiB of expert banks do not fit beside the KV pool and the GGUF
// page cache on a 119 GiB box, which was always the risk; what is broken is the
// FALLBACK that was supposed to make that safe.
//
// Turning a clean refusal into a segfault is strictly worse for a user, so the
// default is restored to exactly what it was and the arm is reachable only when
// asked for by name. The mechanism is not diagnosed -- O46 records the three
// hypotheses already killed, so the next wave does not spend a lease re-killing
// them -- and shipping a guessed fix for a defect that cannot be reproduced off
// the box is the failure this comment exists to refuse.
//
// `1` and nothing else, deliberately: this enables a path MEASURED to crash on
// the one artifact it has been driven against, so it does not get the tree's
// usual "any first character but 0" polarity.
bool DeviceExpertsOptedIn() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GLM5_NEXT_DEVICE_EXPERTS");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

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
  // PER-TENSOR NORMS RIGHT AFTER THE BRIDGE. An unloaded tensor and an
  // all-zero tower are the (b) case of the four this instrument separates, and
  // they are visible here and nowhere later: by the time a zero weight has been
  // multiplied into a hidden state the hidden state is the only thing left to
  // look at, and it is zero for reasons this file cannot distinguish.
  if (diag::Level() > 0) {
    const std::string tag = what + " ";
    diag::Stats((tag + "attn_norm").c_str(), slot_.input_layernorm);
    diag::Stats((tag + "ffn_norm").c_str(), slot_.post_attention_layernorm);
    diag::Stats((tag + "hc_attn.fn").c_str(), slot_.attn_hc.fn);
    diag::Stats((tag + "hc_attn.base").c_str(), slot_.attn_hc.base);
    diag::Stats((tag + "hc_attn.scale").c_str(), slot_.attn_hc.scale);
    diag::Stats((tag + "hc_ffn.fn").c_str(), slot_.ffn_hc.fn);
    if (slot_.attn_kind == Glm5NextLayerKind::kLinearAttention) {
      diag::Stats((tag + "kda.q_proj").c_str(), slot_.kda.q_proj);
      diag::Stats((tag + "kda.k_proj").c_str(), slot_.kda.k_proj);
      diag::Stats((tag + "kda.v_proj").c_str(), slot_.kda.v_proj);
      diag::Stats((tag + "kda.a_log").c_str(), slot_.kda.a_log);
      diag::Stats((tag + "kda.dt_bias").c_str(), slot_.kda.dt_bias);
      diag::Stats((tag + "kda.o_norm").c_str(), slot_.kda.o_norm);
      diag::Stats((tag + "kda.o_proj").c_str(), slot_.kda.o_proj);
    } else {
      diag::Stats((tag + "mla.q_a_proj").c_str(), slot_.dsa.mla.q_a_proj);
      diag::Stats((tag + "mla.q_b_proj").c_str(), slot_.dsa.mla.q_b_proj);
      diag::Stats((tag + "mla.kv_a_proj_with_mqa").c_str(), slot_.dsa.mla.kv_a_proj_with_mqa);
      diag::Stats((tag + "mla.k_b_proj").c_str(), slot_.dsa.mla.k_b_proj);
      diag::Stats((tag + "mla.v_b_proj").c_str(), slot_.dsa.mla.v_b_proj);
      diag::Stats((tag + "mla.o_proj").c_str(), slot_.dsa.mla.o_proj);
    }
    if (slot_.mlp_kind == Glm5NextMlpKind::kDense) {
      diag::Stats((tag + "mlp.gate_proj").c_str(), slot_.dense_mlp.gate_proj);
      diag::Stats((tag + "mlp.up_proj").c_str(), slot_.dense_mlp.up_proj);
      diag::Stats((tag + "mlp.down_proj").c_str(), slot_.dense_mlp.down_proj);
    } else {
      diag::Stats((tag + "moe.router_weight").c_str(), slot_.moe.router_weight);
      diag::Stats((tag + "moe.e_score_bias").c_str(),
                  slot_.moe.e_score_correction_bias);
      diag::Stats((tag + "moe.shared.gate").c_str(), slot_.moe.shared.gate_proj);
      diag::Stats((tag + "moe.shared.down").c_str(), slot_.moe.shared.down_proj);
    }
  }
  return slot_;
}

// ─── the forward ─────────────────────────────────────────────────────────────

std::vector<float> Glm5NextHostForward(const Glm5NextWeights& weights,
                                       const std::vector<int32_t>& token_ids,
                                       const std::vector<int32_t>& logits_indices,
                                       vt::Queue& queue,
                                       std::vector<LayerCache>* caches,
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
  // --- W9c-3a: THE DEVICE SPLIT ---------------------------------------------
  //
  // This forward used to refuse a non-CPU queue outright, and the refusal was
  // RIGHT about its premise: nearly every buffer here is a host
  // `std::vector<float>`, and `vt::MoeRouterTopK`, `vt::MoeCombine` and the two
  // grouped keep-quant GEMMs all require every operand to sit on the queue's
  // device (`src/vt/ops.cpp:214`, `:243`, `:277`). Handing them a CUDA queue
  // over host pointers is a refusal or a crash and never a fallback.
  //
  // What changed is not that premise. It is that ONE arm of this model can now
  // put its operands on the device: the routed-expert keep-quant GEMM, whose
  // banks `dense_attn::ResidentWeight` uploads verbatim and whose activations
  // are `DBuf`s. #2260 landed the last two encodings this artifact needs
  // (`cuda_quant_dot.cu:1832-1843`, IQ2_XS and IQ4_XS, named for this model),
  // which is what makes that arm reachable at all.
  //
  // So the queue is SPLIT rather than the refusal lifted. `host_queue` is a CPU
  // queue this function constructs -- the shape `VaeCpuQueue()`
  // (`ltx2_video_vae.cpp:459`) already uses -- and every host-reference arm runs
  // on it, unchanged and byte-identical to a `--device cpu` run. The caller's
  // device reaches exactly one consumer.
  //
  // READ THAT AS THE DISCLOSURE IT IS. On `--device cuda` this model computes
  // one arm of eleven on the GPU. The KDA recurrence, the k-pool indexer, the
  // eager MLA attention, both mHC sites, the router, the combine, the MLPs, the
  // embedding gather and the `lm_head` all still run on the host. The row's
  // spec records it as O43 and issue #2410 owns the rest.
  vt::Queue host_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Queue& hq = queue.device.type == vt::DeviceType::kCPU ? queue : host_queue;
  // A POINTER and not a `dense_attn::Dev`, because `Dev` holds two references
  // and therefore cannot be assigned after construction. The `Dev` itself is
  // built once, below, at the only place that has one to build.
  vt::Backend* dev_backend = nullptr;
  if (queue.device.type != vt::DeviceType::kCPU) {
    // ASK THE OP TABLE, NOT A DEVICE LIST. This read
    // `queue.device.type != kCUDA` for one commit, and
    // `scripts/check-device-leakage.py` was right to red it: a device name in
    // the device-agnostic layer is a list that goes stale the day another
    // backend registers these ops, and it answers the wrong question anyway.
    // The question is whether THIS device has a provider for the two grouped
    // keep-quant GEMMs, which is exactly what `OpRegistered` answers -- the
    // same move `gguf_keep_quant.cpp` made when its device list became
    // `OpRegistered(kEmbeddingQuant, dev)` (`ops.h:735`).
    //
    // Both ops, because half the pair is not half the capability: the fused
    // gate/up op produces the operand the grouped down-projection consumes.
    const bool gate_up_here =
        vt::OpRegistered(vt::OpId::kMoeGateUpSwiGLUGrouped, queue.device.type);
    const bool down_here =
        vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, queue.device.type);
    // `TryGetBackend`, not `GetBackend`: a registered op with no backend to run
    // it on is a different fault from an unregistered op, and the message says
    // which rather than throwing out of the registry about a device type.
    vt::Backend* backend = vt::TryGetBackend(queue.device);
    if (!gate_up_here || !down_here || backend == nullptr) {
      Fail(std::string(
               "this device cannot run the one primitive of this model that "
               "leaves the host -- the routed-expert keep-quant GEMM. "
               "`vt::MoeGateUpSwiGLUGrouped` provider: ") +
           (gate_up_here ? "yes" : "NO") +
           ", `vt::MatmulBTQuantGrouped` provider: " + (down_here ? "yes" : "NO") +
           ", backend registered: " + (backend != nullptr ? "yes" : "NO") +
           ". Every other primitive here is a host reference, so a device that "
           "cannot run that GEMM has nothing to offer this forward: run it with "
           "--device cpu. See .agents/specs/glm5-next-flash.md section W9c-3a "
           "and issue #2464.");
    }
    // ORDER MATTERS: the op-table refusal above is the more specific diagnosis
    // and must win, so a device with no provider still gets told that rather
    // than being told to set a variable that would not help it.
    if (!DeviceExpertsOptedIn()) {
      Fail("the routed-expert device arm is OPT-IN and is not enabled, so this "
           "forward will not take a non-CPU queue. It defaults off because both "
           "`--device cuda` legs against the published 101.24 GiB UD-Q2_K_XL "
           "artifact on dgx:gpu0 died with SIGSEGV having emitted no token, 3 of 3, "
           "reproducibly, once the expert banks stopped fitting and the arm fell "
           "back to the host mid-model. Run this model with --device cpu, which "
           "emits ` Paris.` on that artifact. Set VT_GLM5_NEXT_DEVICE_EXPERTS=1 "
           "to reach the arm anyway -- that is for debugging the crash, not for "
           "serving. See .agents/specs/glm5-next-flash.md O46 and issue #2464.");
    }
    dev_backend = backend;
  }

  diag::Banner("Glm5NextHostForward");
  if (diag::Level() > 0) {
    std::fprintf(stderr,
                 "[glm5-diag] step: T=%lld H=%lld V=%lld layers=%lld hc=%lld "
                 "caches=%s logits_indices=%zu\n",
                 static_cast<long long>(T), static_cast<long long>(H),
                 static_cast<long long>(V),
                 static_cast<long long>(p.num_hidden_layers),
                 static_cast<long long>(p.mhc.mult),
                 caches == nullptr ? "null" : "present", logits_indices.size());
    diag::Ids("step token_ids", token_ids);
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
  diag::Stats("embeds (token_embd gather)", embeds);

  // ── the stack ──────────────────────────────────────────────────────────────
  //
  // ONE sequence, no padding, so every mask entry is 1. `:1463-1470` guarantees
  // the mask exists for the indexer, which is why it is built rather than left
  // empty.
  const std::vector<uint8_t> mask(static_cast<size_t>(T), 1);
  const std::vector<float> norm =
      DecodeOwnedTensorToF32(weights.norm, "output_norm.weight");
  diag::Stats("output_norm.weight", norm);
  Glm5NextGgufLayerSource layers(weights);
  // `caches` non-null carries the history; `TextModelForward` refuses a vector
  // that is not exactly `num_hidden_layers` long by name
  // (`glm5_next_layer.cpp`), so a binding that produced the wrong count stops
  // here rather than reading a default-constructed state as a zero one.
  // ONE call, two argument sets, and no second copy of the stack: `Dev`'s
  // reference members make it unassignable, so the device one is constructed
  // inside the branch that has a backend and lives exactly as long as the call
  // that reads it.
  const std::vector<float> hidden = [&]() -> std::vector<float> {
    if (dev_backend == nullptr) {
      return TextModelForward(p, norm, layers, embeds, mask, /*batch=*/1,
                              /*seq_len=*/T, caches, hq, /*dev=*/nullptr);
    }
    dense_attn::Dev d{*dev_backend, queue};
    return TextModelForward(p, norm, layers, embeds, mask, /*batch=*/1,
                            /*seq_len=*/T, caches, hq, &d);
  }();
  diag::Stats("hidden (final, entering the head)", hidden);
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
    if (first == 0) diag::Stats("lm_head chunk[0]", chunk);
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
  if (diag::Level() > 0) {
    for (size_t r = 0; r < want.size(); ++r) {
      const std::string tag = "logits[row " + std::to_string(want[r]) + "]";
      diag::TopK(tag.c_str(), &logits[r * static_cast<size_t>(V)],
                 static_cast<size_t>(V), 5);
    }
  }
  return logits;
}

}  // namespace vllm::glm5_next
