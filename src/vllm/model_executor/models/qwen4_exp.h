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

  // The RESOLVED per-head vocabulary sizes, when the SOURCE states them.
  //
  // A config.json states none of these: HF derives them from
  // `ngram_vocab_size_base` and `make_ngram_vocab_size_divisible_by` as the
  // successive primes after `base - 1`. A `qwen4exp` GGUF states them outright
  // (`qwen4exp.ple.head_vocab_sizes`) and states NEITHER input, because
  // llama.cpp's converter reads them off the checkpoint's own buffers rather
  // than re-deriving them.
  //
  // Where the source states them they are the AUTHORITY, because they are what
  // the shipped tensor was actually built against; where it does not, the prime
  // chain derives them. Empty means "the source did not say".
  std::vector<int64_t> head_vocab_sizes;

  // The RESOLVED exclusive prefix sum over `head_vocab_sizes`, when the SOURCE
  // states it. `qwen4exp.ple.head_offsets` is written by llama.cpp #27742's
  // converter as an array INDEPENDENT of `head_vocab_sizes` — it reads both off
  // the checkpoint's own buffers — so the two are a genuine cross-check on each
  // other and not two spellings of one number. Empty means "the source did not
  // say", exactly as for `head_vocab_sizes`.
  //
  // Until MODEL-MM-QWEN4-EXP W5g this array was written into the text config by
  // `Qwen4ExpHfConfigFromGguf` and read by NOTHING, so a converter that emitted
  // offsets disagreeing with its own sizes loaded silently.
  std::vector<int64_t> head_offsets;

  // Whether the SOURCE stated `ngram_vocab_size_base`, as opposed to inheriting
  // upstream's 20,000,000 default.
  //
  // THE DISTINCTION IS LOAD-BEARING AND A DEFAULTED VALUE CANNOT CARRY IT. A
  // `config.json` states the base and states no head sizes; a `qwen4exp` GGUF
  // states the head sizes and states NO base, because the converter reads the
  // resolved arrays off the checkpoint instead of re-deriving them. So on the
  // GGUF arm the base is ALWAYS 20,000,000 whatever the file was built from,
  // and comparing a stated head-size set against a chain derived from it
  // compares the file against a default. That comparison happens to hold for
  // the released checkpoint, whose base really is 20,000,000, and refuses every
  // other `qwen4exp` file — which is what stopped W5f's reachability case
  // inside layer 1's PLE block.
  bool ngram_vocab_size_base_stated = false;

  // The n-gram table's row count: `sum(head_vocab_sizes)` rounded UP to
  // `make_ngram_vocab_size_divisible_by`. Zero when the source stated no sizes,
  // which is the caller's signal to derive the chain instead.
  int64_t stated_padded_vocab_size() const {
    if (head_vocab_sizes.empty() || make_ngram_vocab_size_divisible_by <= 0) {
      return 0;
    }
    int64_t total = 0;
    for (int64_t v : head_vocab_sizes) total += v;
    const int64_t d = make_ngram_vocab_size_divisible_by;
    return ((total + d - 1) / d) * d;
  }

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

  // --- linear attention (Gated DeltaNet) ---
  //
  // W1 read these through the shared `HfConfig` and DROPPED them, recording the
  // omission under `## Owed` for "the wave that consumes them". W5 is that wave:
  // the GGUF loader cannot size one Gated DeltaNet tensor without them, and the
  // V-head reorder it inverts is keyed on `num_key_heads != num_value_heads`.
  //
  // 48 value heads is the CHECKPOINT's value, against upstream's declared
  // default of 32. The docstring is not the authority here; the config is.
  int64_t linear_num_key_heads = 0;    // 16
  int64_t linear_num_value_heads = 0;  // 48
  int64_t linear_key_head_dim = 0;     // 128
  int64_t linear_value_head_dim = 0;   // 128
  int64_t linear_conv_kernel_dim = 0;  // 4

  // The primary EOS id, resolved to ONE value. Upstream permits a LIST and
  // `Qwen4ExpTextModel.forward` takes element [0]
  // (`eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id`,
  // modeling_qwen4_exp.py), so element [0] is what this holds.
  //
  // It is not hygiene. EOS seeds the n-gram token history and is emitted at
  // every segment start by `_shift_right_ignore_eos`, so it reaches a
  // `uint64_t` multiply on the FIRST TOKEN OF EVERY SEQUENCE; out of range it
  // overflows int64 and diverges from the oracle in silence. -1 means "the
  // config did not say", which W1 permits only on a config with no PLE layer.
  int64_t eos_token_id = -1;

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

  // --- derived Gated DeltaNet widths -----------------------------------------
  //
  // Refuse rather than return a nonsense width when the linear-attention group
  // is absent: a config with no Gated DeltaNet layer legitimately leaves these
  // at zero, and a caller that multiplies zeros gets a zero-sized tensor and a
  // shape complaint from three layers down instead of the reason.
  int64_t linear_key_dim() const {
    if (linear_num_key_heads <= 0 || linear_key_head_dim <= 0) {
      throw std::runtime_error(
          "qwen4_exp: linear_key_dim() needs the linear-attention group, but "
          "`linear_num_key_heads` is " +
          std::to_string(linear_num_key_heads) + " and `linear_key_head_dim` " +
          std::to_string(linear_key_head_dim) +
          "; this config declares no Gated DeltaNet geometry.");
    }
    return linear_num_key_heads * linear_key_head_dim;
  }
  int64_t linear_value_dim() const {
    if (linear_num_value_heads <= 0 || linear_value_head_dim <= 0) {
      throw std::runtime_error(
          "qwen4_exp: linear_value_dim() needs the linear-attention group, but "
          "`linear_num_value_heads` is " +
          std::to_string(linear_num_value_heads) +
          " and `linear_value_head_dim` " +
          std::to_string(linear_value_head_dim) +
          "; this config declares no Gated DeltaNet geometry.");
    }
    return linear_num_value_heads * linear_value_head_dim;
  }
  // `conv_dim = key_dim * 2 + value_dim` (modeling_qwen4_exp.py:420): the
  // depthwise conv runs over the CONCATENATED q|k|v stream, not over v alone.
  int64_t linear_conv_dim() const {
    return 2 * linear_key_dim() + linear_value_dim();
  }
  // The hyper-connection residual stream width, `hc_count * hidden_size`.
  int64_t stream_width() const { return hc_count * hidden_size; }
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
