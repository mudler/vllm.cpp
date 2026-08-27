// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) — W1 config surface.
//
// Model-private header, deliberately not under include/: nothing outside this
// model needs these types yet, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. W1 ships no capability.
//
// ORACLE. vLLM does NOT implement `qwen4_exp` at ANY revision (read live
// 2026-08-26 at `origin/main` = `6a5e8f5979`: no `qwen4*` path, no registry
// entry). Under AGENTS.md "When vLLM has no implementation" this row runs a
// SPLIT oracle, recorded in `.agents/specs/qwen4-exp-flash-next.md`:
// transformers **5.16.0** (the accepted lane pin) defines the ALGORITHM, and
// vLLM supplies the OPS for every primitive it does implement. This file is
// entirely the first half. Every anchor below is
// `transformers/models/qwen4_exp/{configuration,modular}_qwen4_exp.py` at
// v5.16.0.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Per-layer kind AFTER upstream's rewrite. `__post_init__` replaces every
// `full_attention` entry with `qwen_sparse_attention`, because the published
// checkpoint says `full_attention` for layers that actually run the QSA
// indexer. A reader that takes the checkpoint at face value wires dense
// attention on 12 of 48 layers and is wrong WITHOUT SAYING SO, which is why
// this enum has no `kFullAttention` enumerator at all: the state is
// unrepresentable rather than merely unused.
enum class Qwen4ExpLayerKind {
  kLinearAttention,     // Gated DeltaNet
  kQwenSparseAttention  // QSA
};

// Qwen Sparse Attention. Upstream validates these five as a GROUP: either all
// are present or none is, and a partial set raises naming the missing fields.
struct Qwen4ExpQsaParams {
  int64_t n_heads = 0;         // indexer_n_heads = 4
  int64_t kv_heads = 0;        // indexer_kv_heads — upstream REQUIRES exactly 1
  int64_t head_dim = 0;        // indexer_head_dim = 128
  int64_t budget = 0;          // indexer_budget = 2048 tokens
  int64_t compress_ratio = 0;  // indexer_compress_ratio = 4

  // budget / compress_ratio = 512. Derived, never read from the config.
  //
  // REFUSES rather than divides when the group is absent. QSA is optional as a
  // whole -- upstream treats all-five-absent as "QSA off" -- so a legally
  // parsed config can leave `compress_ratio` at 0, and `budget / 0` is SIGFPE
  // on x86: a crash, not a refusal, and one no downstream gate would attribute
  // to this config. W2 and W4 are the callers, and neither has a reason to
  // check the precondition before asking.
  int64_t block_topk() const {
    if (compress_ratio <= 0) {
      throw std::runtime_error(
          "qwen4_exp: block_topk() needs QSA, but `indexer_compress_ratio` is " +
          std::to_string(compress_ratio) +
          "; the QSA group is absent from this config.");
    }
    return budget / compress_ratio;
  }
};

// Per-Layer Embedding: the hashed n-gram table plus its dilated depthwise conv.
struct Qwen4ExpPleParams {
  // ONE-INDEXED in the checkpoint, and upstream says so in terms
  // ("One-indexed decoder layer ids"). The lookup is
  // `ple_layer_ids.index(layer_idx + 1)`, so `[2]` selects 0-based layer 1.
  // Stored here ALREADY CONVERTED to 0-based, because carrying a one-indexed
  // value through the port is how the off-by-one gets rediscovered. Confirmed
  // three ways: the upstream docstring, the config validator's own
  // `layer_types[layer_id - 1]`, and the published checkpoint, whose PLE
  // tensors all sit under `model.language_model.layers.1.ple.`.
  std::vector<int64_t> layer_ids_zero_based;

  int64_t embed_dim = 0;         // ple_embed_dim = 2560, defaults to hidden_size
  int64_t conv_kernel_size = 0;  // ple_conv_kernel_size = 4
  int64_t ngram_size = 0;        // 3 — ALSO the conv DILATION
  int64_t heads_per_ngram = 0;   // 8
  int64_t ngram_vocab_size_base = 0;          // 20,000,000
  int64_t make_ngram_vocab_size_divisible_by = 0;  // 128
  int64_t split_ngram_parts = 512;  // 128 in the checkpoint; SHARDING only, unused in the forward
  int64_t seed = 1234;            // absent from the published config; the dataclass default

  // (ngram_size - 1) * heads_per_ngram = 16 hash heads per token.
  int64_t ngram_heads() const { return (ngram_size - 1) * heads_per_ngram; }
  // embed_dim / ngram_heads = 160. Refuses rather than divides when the head
  // count is zero, which `ngram_size == 1` produces on a config no PLE layer
  // uses and therefore nothing validates.
  int64_t head_dim_per_ngram() const {
    const int64_t heads = ngram_heads();
    if (heads <= 0) {
      throw std::runtime_error(
          "qwen4_exp: head_dim_per_ngram() needs a positive n-gram head count, "
          "got " + std::to_string(heads) + " from (ngram_size " +
          std::to_string(ngram_size) + " - 1) * heads_per_ngram " +
          std::to_string(heads_per_ngram) + ".");
    }
    return embed_dim / heads;
  }
  // (conv_kernel_size - 1) * ngram_size = 9. NOT `kernel - 1`: the conv is
  // DILATED, so its state is three times deeper than an undilated one and the
  // taps sit at lags {9, 6, 3, 0}.
  int64_t short_conv_state_len() const {
    return (conv_kernel_size - 1) * ngram_size;
  }
};

struct Qwen4ExpParams {
  // --- geometry ---
  int64_t hidden_size = 0;        // 2560
  int64_t num_hidden_layers = 0;  // 48
  int64_t vocab_size = 0;         // 248320
  double rms_norm_eps = 1e-6;

  std::vector<Qwen4ExpLayerKind> layer_types;  // 48 entries after the rewrite

  // --- gated residual (hyper-connections) ---
  // The residual stream is hc_count * hidden_size = 10240 wide through the
  // WHOLE stack. Upstream requires hc_count > 1.
  int64_t hc_count = 0;    // 4
  int64_t hc_lowrank = 0;  // 320

  // --- MoE ---
  int64_t num_experts = 0;                     // 512
  int64_t num_experts_per_tok = 0;             // 10 routed, plus 1 shared
  int64_t moe_intermediate_size = 0;           // 640
  int64_t shared_expert_intermediate_size = 0; // 640

  // --- attention ---
  int64_t num_attention_heads = 0;   // 24
  int64_t num_key_value_heads = 0;   // 2
  int64_t head_dim = 0;              // 256
  // 1.0 is upstream's default: the validator reads
  // `(self.rope_parameters or {}).get("partial_rotary_factor", 1.0)`
  // (configuration_qwen4_exp.py:225) and the class declares no such field.
  double partial_rotary_factor = 1.0;
  int64_t rotary_dim = 0;            // int(head_dim * partial_rotary_factor) = 64

  Qwen4ExpQsaParams qsa;
  Qwen4ExpPleParams ple;

  // --- MTP ---
  int64_t mtp_num_hidden_layers = 0;  // 1

  // 3 when any PLE layer exists (GDN conv, PLE conv, n-gram token history),
  // else 1. Mirrors upstream's `number_of_conv_states`.
  int64_t number_of_conv_states() const {
    return ple.layer_ids_zero_based.empty() ? 1 : 3;
  }
};

// Resolves and VALIDATES. The resolve IS the validation: it throws by name on
// everything unrepresentable, mirroring upstream `validate_architecture`.
Qwen4ExpParams ParseQwen4ExpParams(const HfConfig& config);

// ModelFactory::parse_config hook. Delegates to ParseQwen4ExpParams and
// discards the result, so a malformed config is refused at load rather than at
// first forward.
void ParseQwen4ExpConfig(const HfConfig& config);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_H_
