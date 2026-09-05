// Qwen4-Exp (`Qwen4ExpForConditionalGeneration`) W2 — the hashed n-gram
// embedding and the PLE dilated depthwise conv, as a portable host reference.
//
// Issue #1987, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md` (`## Port map`, "The n-gram embedding
// is integer-exact or it is silently wrong", and "PLE: a strided-history conv
// with no vLLM op, confirmed").
//
// ─── WHY THIS FILE HAD NO vLLM ANCHOR, AND WHY IT NOW HAS ONE ────────────────
// SUPERSEDED 2026-08-31 (#2489): `nvidia/ple_layer.py` at `e126687a9a` is BOTH
// the dilated conv (`:589-599`) and the n-gram embedding (`:151-436`). Mirror it.
// The measured negative below was true when taken and is kept as history: at
// vLLM `origin/main` = `6a5e8f5979`, `git grep -in dilat` returns zero lines in
// `vllm/model_executor/layers/mamba/`, zero in `csrc/` and zero in `tests/`;
// `layers/conv.py` defines only `Conv2dLayer` and `Conv3dLayer`. Upstream
// reached the same conclusion from the other side and hand-rolled it, saying so
// in a comment: "We cannot use the usual functions/kernels here for the short
// conv as the conv1d has dilation".
//
// ORACLE: huggingface/transformers **v5.16.0**, this row's ACCEPTED lane pin
// (spec `## Oracles`; `v5.16.0` is the FIRST release containing `qwen4_exp` —
// `v5.15.0` returns HTTP 404 for the same path). Every `file:line` below is at
// that tag. `modular_qwen4_exp.py` is the authored delta;
// `modeling_qwen4_exp.py` is its generated expansion and is what the goldens
// were exec'd out of, so both are cited.
//
//   OURS                     <-  UPSTREAM (transformers v5.16.0)
//   SplitMix64               <-  modeling_qwen4_exp.py::_splitmix64 (:979-983)
//                                = modular_qwen4_exp.py (:568-572)
//   BuildLayerMultipliers    <-  ::_build_layer_multipliers (:986-995)
//   IsPrime/FindNthPrimeAfter<-  ::_is_prime (:998-1006), ::_find_nth_prime_after
//                                (:1009-1015)
//   BuildNGramTableLayout    <-  ::Qwen4ExpTextNGramEmbedding.__init__ (:1019-1051)
//   ShiftRightIgnoreEos      <-  ::Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos
//                                (:1053-1067)
//   BuildNGramIds            <-  ::Qwen4ExpTextNGramEmbedding.forward (:1069-1114),
//                                plus cache_utils.py::LinearAttentionLayer
//                                .update_conv_state (:1037-1075) for state 2
//   SignedSqrtGate           <-  ::Qwen4ExpTextPLELayer.forward (:1181)
//   PleShortConv             <-  ::Qwen4ExpTextPLELayer._short_conv (:1150-1167)
//   PleForward               <-  ::Qwen4ExpTextPLELayer.forward (:1169-1189)
//   GroupedRmsNorm (private) <-  ::Qwen4ExpTextRMSNorm (:158-181), group_size arm
//
// ─── W2 SCOPE, STATED SO NOBODY HAS TO INFER IT ──────────────────────────────
// THIS IS A HOST (CPU) REFERENCE AND IT IS NOT YET REACHED FROM ANY PRODUCTION
// ENTRY POINT. `qwen4_exp` has no registry entry, no loader and no
// `ModelRegistry::Forward` arm until W5 assembles the model, so per AGENTS.md
// "Nothing lands dead" this is a STAGED SLICE landing unreached: the wiring is
// owned by row `MODEL-MM-QWEN4-EXP` and tracked by campaign issue #1978, and
// the spec lists it under `## Owed`. Do not read the absence of a call site as
// an oversight; read it as the recorded debt it is.
//
// What is deliberately NOT here: the batched/device arm (see the batching seam
// below), the weight loader and its 128-shard NUMERIC reassembly, grouped
// RMSNorm as a shared layer, the Gated Residual stream (W3), QSA (W4), and any
// speed claim (`## Gates` admits none from this row until G2 passes).
//
// ─── THE BATCHING SEAM ───────────────────────────────────────────────────────
// Every entry point below is PER SEQUENCE and takes its own `PleSequenceState`.
// Nothing loops over the batch, nothing owns a global, and nothing needs a
// host round trip inside a decode step. The n-gram gather is
// `(ngram_size-1) * heads_per_ngram` = 16 uncoalesced random rows per token per
// sequence, so at batch B it is 16*B independent gathers; a device arm replaces
// the body of `BuildNGramIds` + the embedding gather in `PleForward` with one
// kernel over [B, T, 16] and keeps this signature as the single-sequence
// fallback. `PleSequenceState` is deliberately two flat, contiguous buffers so
// a paged KV arm can point at cache pages instead of owning them.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::qwen4_exp {

// The MINIMUM config surface W2 needs. W1 (#1981, PR #1986) owns the real
// `Qwen4ExpParams` / `Qwen4ExpPleParams` in
// `src/vllm/model_executor/models/qwen4_exp.h`; that branch is not on `main` at
// this commit, so this wave declares its own rather than depending on an
// unmerged branch. The three helper names below are W1's names on purpose, so
// the swap in W3 is mechanical. Defaults are the dataclass defaults at
// `modular_qwen4_exp.py:155-163`, and `seed` is 1234 because `config.seed` is
// ABSENT from the published `config.json` (spec, `### The n-gram embedding is
// integer-exact or it is silently wrong`).
struct PleGeometry {
  int64_t hidden_size = 0;
  int64_t hc_count = 0;
  int64_t ple_embed_dim = 0;
  int64_t ple_conv_kernel_size = 4;
  int64_t ngram_size = 3;
  int64_t heads_per_ngram = 8;
  int64_t ngram_vocab_size_base = 20000000;
  int64_t make_ngram_vocab_size_divisible_by = 128;
  // `config.vocab_size`, the UNIGRAM vocabulary. It bounds the multiplier so
  // `token_id * multiplier` cannot overflow int64 — but only while every id
  // that enters the mix really is below it, which is why `BuildNGramIds`
  // refuses one that is not. TWO ids enter the mix, not one; see below.
  int64_t vocab_size = 0;
  // -1 is an UNSET SENTINEL and is refused, not a working default: eos is the
  // second id in the mix and there is no safe value to pick for it. A loader
  // must set it, and `config.eos_token_id` is permitted to be a LIST upstream
  // (modeling_qwen4_exp.py:1032 takes element [0]), which is the realistic way
  // to get it wrong.
  int64_t eos_token_id = -1;
  int64_t seed = 1234;
  double rms_norm_eps = 1e-6;

  // (ngram_size - 1) * heads_per_ngram; 16 for the released config.
  int64_t ngram_heads() const { return (ngram_size - 1) * heads_per_ngram; }
  int64_t head_dim_per_ngram() const { return ple_embed_dim / ngram_heads(); }
  // NINE, not `kernel - 1`: `(4 - 1) * 3`, because the conv is dilated by
  // `ngram_size`. modeling_qwen4_exp.py:1135
  // (`self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation`,
  // with `conv_dilation = config.ngram_size` at :1134).
  int64_t short_conv_state_len() const {
    return (ple_conv_kernel_size - 1) * ngram_size;
  }
  // The n-gram token history kept in conv state 2. modeling_qwen4_exp.py:1023.
  int64_t context_len() const { return ngram_size - 1; }
  // The width of the hyper-connection residual stream: 4 * 2560 = 10240.
  int64_t stream_width() const { return hc_count * hidden_size; }
};

// ─── the hash chain ──────────────────────────────────────────────────────────

// `_splitmix64`, modeling_qwen4_exp.py:979-983. UNSIGNED THROUGHOUT, and that
// is divergence site #1 in the spec: upstream's `>> 30 / 27 / 31` are LOGICAL
// shifts on a non-negative Python int. On `int64_t` they become arithmetic
// shifts, the multiplicand is wrong, and the top bit is set about half the
// time, so it fires immediately and silently.
uint64_t SplitMix64(uint64_t value);

// `_is_prime` / `_find_nth_prime_after`, modeling_qwen4_exp.py:998-1006 and
// :1009-1015.
bool IsPrime(int64_t value);
int64_t FindNthPrimeAfter(int64_t start, int64_t count);

// `_build_layer_multipliers`, modeling_qwen4_exp.py:986-995. Divergence site
// #2 lives here: `_splitmix64(value) % half_bound` is an UNSIGNED modulo. The
// dividend routinely exceeds 2^63 and a signed modulo yields a negative
// residue. Returns `ngram_size` values.
std::vector<int64_t> BuildLayerMultipliers(int64_t unigram_vocab_size,
                                           int64_t ngram_size,
                                           int64_t ple_layer_index,
                                           int64_t seed);

// Everything `Qwen4ExpTextNGramEmbedding.__init__` derives before it sees a
// token. modeling_qwen4_exp.py:1019-1051.
struct NGramTableLayout {
  std::vector<int64_t> head_vocab_sizes;   // [ngram_heads]
  std::vector<int64_t> head_offsets;       // [ngram_heads], exclusive prefix sum
  std::vector<int64_t> layer_multipliers;  // [ngram_size]
  int64_t total_vocab_size = 0;
  int64_t padded_vocab_size = 0;  // rounded UP to make_ngram_vocab_size_divisible_by
};
NGramTableLayout BuildNGramTableLayout(const PleGeometry& geom,
                                       int64_t ple_layer_index);

// ─── the per-sequence state a PLE layer owns ─────────────────────────────────

// Conv state 1 (the PLE conv) and conv state 2 (the n-gram token history).
// Upstream keeps both in the linear-attention cache alongside the GDN conv,
// which is why `number_of_conv_states == 3` on a PLE layer
// (modular_qwen4_exp.py:178-180).
//
// `Reset` seeds `tokens` with EOS and `conv` with zeros, which is EXACTLY what
// upstream's first-call path computes, and the equivalence is worth stating
// because the two states pad DIFFERENTLY:
//   * tokens: upstream's `update_conv_state` pads with 0 — a VALID token id —
//     so the layer works around it with an explicit EOS left-pad
//     (modeling_qwen4_exp.py:1080-1088). Pad with EOS, never with zero.
//   * conv: upstream pads with zeros on both the first-call and the steady
//     path, so a zero-initialised buffer is bit-identical to its first call.
struct PleSequenceState {
  std::vector<float> conv;      // [stream_width, short_conv_state_len]
  std::vector<int64_t> tokens;  // [context_len]
  void Reset(const PleGeometry& geom);
};

// `_shift_right_ignore_eos`, modeling_qwen4_exp.py:1053-1067, one row.
// Position i takes token[i - shift] only when at least `shift` tokens have
// passed since the start of i's EOS-delimited segment; otherwise EOS. An EOS
// token belongs to the segment it TERMINATES, because the "previous EOS" scan
// is strictly-before. `out` holds `seq_len` ids and may not alias `token_ids`.
void ShiftRightIgnoreEos(const int64_t* token_ids, int64_t seq_len,
                         int64_t shift, int64_t eos_token_id, int64_t* out);

// The id half of `Qwen4ExpTextNGramEmbedding.forward`,
// modeling_qwen4_exp.py:1069-1112. Reads `state->tokens` as the left context,
// writes `num_tokens * ngram_heads` ids to `out_ids` (row-major), and advances
// `state->tokens`. Integer-exact: there is no tolerance and no downstream gate
// that localises an error here.
//
// REFUSES BY NAME on any id outside `[0, vocab_size)` — BOTH the `input_ids`
// the caller passes AND `geom.eos_token_id`, which is the one id in the mix
// that does not come from `input_ids`. `Reset` seeds the history with eos and
// `ShiftRightIgnoreEos` emits it at every segment start, so it is on the FIRST
// TOKEN OF EVERY SEQUENCE; `BuildNGramTableLayout` refuses it at construction
// time as well. Upstream has no such check and does not need one, because its
// loader cannot admit one; ours can, and the failure is `token_id * multiplier`
// overflowing int64 and diverging in silence (spec, "That bound holds only
// while every token id is below `vocab_size`"). Measured against transformers
// v5.16.0 at the struct's own `eos = -1` default: upstream row 0
// `[2, 35, 67, 96]`, ours `[8, 30, 67, 96]`, no exception either side.
void BuildNGramIds(const PleGeometry& geom, const NGramTableLayout& layout,
                   const int64_t* input_ids, int64_t num_tokens,
                   PleSequenceState* state, int64_t* out_ids);

// ─── the PLE layer ───────────────────────────────────────────────────────────

// Row-major, bias-free, exactly as `nn.Linear.weight` stores them
// ([out_features, in_features]). `norm_*` are RMSNorm DELTAS: upstream applies
// `(1.0 + weight)` (modeling_qwen4_exp.py:177), so a zeroed buffer is identity.
struct PleWeights {
  const float* ngram_embedding = nullptr;  // [padded_vocab_size, head_dim_per_ngram]
  const float* key_proj = nullptr;         // [stream_width, ple_embed_dim]
  const float* value_proj = nullptr;       // [hidden_size, ple_embed_dim]
  const float* norm_key = nullptr;         // [stream_width]
  const float* norm_query = nullptr;       // [stream_width]
  const float* norm_conv = nullptr;        // [stream_width]
  const float* conv1d = nullptr;           // [stream_width, ple_conv_kernel_size]
};

// `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`,
// modeling_qwen4_exp.py:1181. THE CLAMP IS BEFORE THE SQRT, so the floor on the
// output MAGNITUDE is sqrt(1e-6) = 1e-3 and tiny scores are AMPLIFIED, not
// squashed. Exactly zero maps to zero because `sign(0) == 0`, so the function
// is genuinely discontinuous at the origin — and that origin is reachable on a
// fully masked row. Mirror it; do not tidy it. Clamping after the sqrt is wrong
// by three orders of magnitude in that band.
float SignedSqrtGate(float gate);

// `_short_conv`, modeling_qwen4_exp.py:1150-1167. Depthwise, kernel 4, dilated
// by `ngram_size` = 3, so output t reads lags {9, 6, 3, 0} with weights
// w0..w3 in that order and the lag-0 tap makes it causal. `normed` is
// [num_tokens, stream_width] row-major (the NORMED conv input, never the raw
// hidden state); `out` is the same shape and carries silu already applied.
// Advances `state->conv`.
//
// The state is a genuine 9-deep ring read at stride 3 and cannot be compressed
// to 3 columns even though any single step touches only three of them: 9
// columns x 10240 channels is ~180 KiB per sequence at bf16 for this one layer,
// which is a KV-budget line item rather than a rounding error.
void PleShortConv(const PleGeometry& geom, const float* conv1d_weight,
                  const float* normed, int64_t num_tokens,
                  PleSequenceState* state, float* out);

// `Qwen4ExpTextPLELayer.forward`, modeling_qwen4_exp.py:1169-1189.
//   hidden_states  [num_tokens, stream_width]
//   input_ids      [num_tokens]
//   conv_mask      [num_tokens] of 0/1, or nullptr. `None` in steady-state
//                  decode, so the masking is prefill-only. BOTH the skip term
//                  and the conv input are masked (:1185-1187), and the masked
//                  conv input is what the 9-column state keeps — gated by
//                  `kPleMaskedExpectedOutput`, single-shot and incremental.
//                  It is also a PAIRED obligation with the caller: the
//                  activations are masked here AND `input_ids` must already
//                  carry EOS at padded positions, because the hash reads token
//                  ids rather than activations. Masking only the activations
//                  leaks padding into the hash. That half has no caller yet and
//                  is owed to W5; the spec's `## Owed` names it.
//   out            [num_tokens, stream_width]
// The fork the spec warns about is at the end: the skip term is the UN-NORMED
// `gated_value` and only the NORMED copy enters the conv.
void PleForward(const PleGeometry& geom, const NGramTableLayout& layout,
                const PleWeights& weights, const float* hidden_states,
                const int64_t* input_ids, int64_t num_tokens,
                const unsigned char* conv_mask, PleSequenceState* state,
                float* out);

}  // namespace vllm::qwen4_exp
