// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) W5e-2 — `Qwen4ExpTextPLELayer` as ONE
// production block, and the LAST of the three block seams the layer loop needs.
//
// Issue [#2336](https://github.com/mudler/vllm.cpp/issues/2336), wave issue
// [#2031](https://github.com/mudler/vllm.cpp/issues/2031), campaign issue
// [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── THE GAP THIS FILE CLOSES, IN #2336's OWN MEASUREMENTS ───────────────────
// "`PleForward` (`qwen4_exp_ple.cpp:366`) is a HOST `const float*` reference.
// Callers outside its own translation unit: **zero**." — "`vt::RmsNormGroup` …
// has **zero** callers outside `src/vt/` and its own suite. The op is on `main`
// and nothing routes to it. It is a seam that landed unreached and is still
// unreached." — "The N-GRAM GATHER has no device composition. … nothing under
// `src/` gathers from it."
//
// W5b-5 closed the same gap for QSA (`RunQwen4ExpQsaBlock`) and W5d-4 for MoE
// (`RunQwen4ExpMoeBlock`). This file is PLE's equivalent, shaped after those two
// rather than after a third convention: an owning `…BlockOutput`, a `dense_attn::Dev`
// first operand, `dense_attn::ResidentWeight` over the loader's own `OwnedTensor`s,
// a cache struct of `vt::Tensor`s, and ONE sequence per call.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// vLLM registers `qwen4_exp` at NO revision, so under AGENTS.md "When vLLM has
// no implementation" the ALGORITHM oracle is transformers **5.16.0**, this row's
// accepted lane pin (`.agents/oracles/transformers.md`, sha256
// `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`), at
// `models/qwen4_exp/modeling_qwen4_exp.py`. Every anchor below was re-derived by
// reading that file rather than relayed: #2336 cited the gate as `:1179-1183`
// and it was off by one at BOTH ends, which W5e-1 corrected.
//
//   :1176  embeddings = self.ple_embedding(input_ids, past_key_values)
//   :1177  key_normed  = norm_key(key_proj(embeddings)).unflatten(-1,(hc,H))
//   :1178  value       = value_proj(embeddings)
//   :1179  query_normed= norm_query(hidden_states).unflatten(-1,(hc,H))
//   :1180  gate        = (key_normed*query_normed).sum(-1,keepdim) / sqrt(H)
//   :1181  gate        = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
//   :1182  gated_value = sigmoid(gate) * value.unsqueeze(-2)
//   :1183  gated_value_normed = norm_conv(gated_value.flatten(-2))
//   :1184  gated_value = gated_value.flatten(-2)
//   :1185-87 conv_mask masks BOTH tensors (apply_mask_to_padding_states, :204-213)
//   :1188  output = gated_value + self._short_conv(gated_value_normed, cache)
//
// Note the ORDER at :1183/:1184: `norm_conv` is applied to the FLATTENED
// `gated_value`, and the un-normed flatten that becomes the skip term happens
// after it. Both read the same values; the line order is recorded because a
// port that normed the [T,hc,H] view per (t,j) row would be a different norm.
//
// ─── WHAT IT COMPOSES, AND WHY NOTHING NEW WAS WRITTEN ───────────────────────
// Six shared-seam ops and one host helper, in upstream's order:
//
//   `qwen4_exp::BuildNGramIds`  the splitmix64 hash (:1069-1112). HOST, and it
//        stays host: it is int64 bit-mixing over token IDS, no `vt::` op hashes,
//        and this block is the first caller it has ever had outside its own TU.
//   `vt::Embedding`             the gather (:1114). Admits a BLOCK-QUANTIZED
//        table and decodes one row per gathered id (W6a, #1989), which is the
//        only affordable residency for a 51.2 G-parameter table.
//   `vt::MatmulBT` x2           `key_proj` and `value_proj` (:1177, :1178).
//   `vt::RmsNormGroup` x3       the three `Qwen4ExpTextRMSNorm(group_size=H)`
//        (:1177, :1179, :1183). W5d-1 landed this op FOR these three norms and
//        it has had no production caller until now.
//   `vt::BatchedMatmul`         the :1180 dot, over `[T*hc,1,H] x [T*hc,H,1]`
//        VIEWS of the two `[T,hc*H]` buffers. No copy and no new op: only the
//        innermost dim must be unit-stride. #2336 claimed this and W5e-1's
//        suite ran it; this block is where it ships.
//   `vt::Qwen4ExpPleGate`       :1181-1182 with the :1184 flatten folded into
//        the output layout (W5e-1). The `/ sqrt(hidden_size)` tail of :1180
//        rides as `gate_divisor`.
//   `vt::MulScalar`             the two `apply_mask_to_padding_states` calls at
//        :1186-1187, applied per masked ROW. See the mask paragraph below.
//   `vt::Qwen4ExpPleConv`       `_short_conv` (:1150-1167) with its state read
//        and write-back (W5b-3).
//   `vt::Add`                   the `gated_value + …` join at :1188.
//
// ─── TWO CONTRACTS THAT PRODUCE FLUENT WRONG TEXT, NOT CRASHES ───────────────
// 1. GAMMA POLARITY (#2218). Every `qwen4_exp` gamma is stored RAW as
//    HuggingFace ships it, centred on 0, and EVERY consumer adds the 1;
//    `ssm_norm` is the architecture's single exception. All three norms here
//    therefore pass `RmsNormGroupArgs::gemma = true`. A port that dropped it
//    scales the stream by ~0 and reads as a corrupt checkpoint.
// 2. THE N-GRAM HISTORY IS SEEDED WITH `eos_token_id`, NEVER WITH ZERO, and
//    zero is a VALID token id. Upstream's `update_conv_state` pads with 0 and
//    the model works around it with an explicit EOS left-pad
//    (:1080-1088); `CacheBuffer` zero-fills for the same reason
//    `update_conv_state` does, so the seeding is OURS to perform. The predicate
//    is `past_len == 0` — this sequence has no previous state, which is exactly
//    upstream's `has_previous_state(layer_idx, state_idx=2) == False`.
//
// ─── THE MASK IS A PAIRED OBLIGATION, AND THIS BLOCK ENFORCES BOTH HALVES ────
// `conv_mask` is `None` in steady-state decode, so the masking is prefill-only.
// Upstream masks BOTH `gated_value` (the skip term) and `gated_value_normed`
// (what enters the conv AND what the 9-column state keeps). The OTHER half has
// had no enforcer anywhere in this tree and the spec's `## Owed` has carried it
// since W2: a masked position must ALSO carry EOS in `input_ids`, because the
// hash reads token ids rather than activations, so masking only the activations
// leaks padding into the hash. This block refuses that pair by name instead of
// documenting it.
//
// The mask is applied with `vt::MulScalar` on ROW VIEWS, one launch per masked
// token per tensor, at upstream's own two sites. The shared surface has no
// per-ROW broadcast multiply — `vt::MulColVecF32` scales per output COLUMN and
// `vt::SigmoidGateBf16` refuses by element count — and a masked position is rare
// (it is padding), so the alternative to this is a new general op that nothing
// else would call. Zeroing `value` before the gate would be arithmetically
// identical and is deliberately NOT done: it is an equivalence argument where
// upstream's own two lines are available.
//
// ─── SCOPE, AND WHAT IS NOT HERE ─────────────────────────────────────────────
// A CALLER. `ForwardQwen4ExpForConditionalGeneration` still refuses by name, so
// this composition lands UNREACHED from any production entry point at its merge
// commit. The wiring is `Qwen4ExpTextModel::Forward`, the layer loop, owned by
// row `MODEL-MM-QWEN4-EXP` W5f under #2336 and #2031; the spec's `## Owed`
// records it. What this wave DOES reach, for the first time, is
// `qwen4_exp::BuildNGramIds`, `vt::RmsNormGroup` and `vt::Qwen4ExpPleGate` — all
// three previously had no caller under `src/` at all.
//
// ONE SEQUENCE PER CALL, as `RunQwen4ExpQsaBlockPaged` takes one. A ragged
// multi-request batch needs `query_start_loc` plumbing this block does not
// carry, even though `vt::Qwen4ExpPleConv` itself is batched; the n-gram history
// is per sequence and `BuildNGramIds` takes one stream of ids.
//
// CPU ONLY. Every `vt::` op this block is the first production caller of is
// registered on `kCPU` alone, and the n-gram id build is a host round trip by
// construction. The spec's `## Owed` carries the device arm.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_PLE_BLOCK_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_PLE_BLOCK_H_

#include <cstdint>
#include <memory>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev
#include "vllm/model_executor/models/qwen4_exp.h"          // Qwen4ExpParams
#include "vllm/model_executor/models/qwen4_exp_ple.h"      // PleGeometry, NGramTableLayout
#include "vllm/model_executor/models/qwen4_exp_weights.h"  // Qwen4ExpPleWeights
#include "vt/tensor.h"

namespace vllm {

// The two persistent states one PLE layer owns, as the ENGINE holds them.
//
// Upstream keeps BOTH in the linear-attention cache alongside the GDN conv,
// which is why `number_of_conv_states == 3` on a PLE layer; this tree publishes
// them as slots 2 and 3 of the uniform recurrent group (see
// `MakeQwen4ExpKVCache`'s `[gdn_conv, temporal, ple_conv, ngram]` note).
//
// `state_row` is which row of BOTH tensors this sequence owns. It is a field
// rather than an assumed 0 because the engine's cache is `[num_slots, …]` and a
// sequence's slot is assigned by the runner, not by the layer.
struct Qwen4ExpPleCaches {
  // Conv state 1: the PLE conv's ring. f32 `[N, hc*H, (K-1)*ngram_size]`,
  // READ-WRITE, exactly the shape `vt::Qwen4ExpPleConv` reads and writes.
  //
  // NINE COLUMNS AT THE RELEASED CONFIG, NOT THREE. The conv is dilated by
  // `ngram_size`, so its state is `(K-1)*dilation` deep and a caller that sized
  // it with the Mamba `K-1` formula gets a message naming both numbers.
  vt::Tensor conv_state;
  // Conv state 2: the n-gram token history, i64 `[N, ngram_size - 1]`,
  // READ-WRITE, and HOST-RESIDENT.
  //
  // Its dtype is int64 because it holds TOKEN IDS — upstream takes the dtype
  // from the first tensor written and a port that stored them as floats would
  // ROUND them (spec, "The n-gram embedding is integer-exact or it is silently
  // wrong"). It is host-resident because `qwen4_exp::BuildNGramIds` is a host
  // int64 hash and no `vt::` op computes it; a device-resident one is refused by
  // name rather than read through a host pointer.
  vt::Tensor tokens;
  int64_t state_row = 0;
};

// Owning device-resident output of one PLE block: a `[T, hc*hidden_size]` view
// plus the shared_ptr that returns its pool block to the DevicePool when the
// last reference drops. Mirrors `Qwen4ExpQsaBlockOutput` / `MoeBlockOutput`
// exactly, so the composing forward keeps one release idiom across all three
// block kinds.
//
// THE OUTPUT IS THE hc-WIDE STREAM, not the collapsed hidden state. PLE is
// injected into every hyper-connection stream (upstream's own docstring: "The
// returned tensor has shape `(batch, seq, hc_count * hidden_size)`"), which is
// the opposite polarity to `RunQwen4ExpQsaBlock`, whose input is the COLLAPSED
// 2560-wide state.
struct Qwen4ExpPleBlockOutput {
  vt::Tensor tensor;              // [T, hc*H] at the stream dtype
  std::shared_ptr<void> storage;  // owns the pool block (Pool().Put on release)
};

// The W2 host reference's config surface, projected from `Qwen4ExpParams`.
//
// It exists so the projection happens ONCE, in one place, rather than at each
// call site: `PleGeometry` predates `Qwen4ExpParams` (W2 declared its own
// because W1's branch was not yet on `main`, and said so), and two structs that
// hold the same numbers are two structs that can disagree.
qwen4_exp::PleGeometry Qwen4ExpPleGeometry(const Qwen4ExpParams& p);

// The n-gram table layout for ONE PLE layer — the per-head vocabulary sizes,
// their exclusive-prefix offsets, and the three splitmix64 layer multipliers.
//
// BUILD IT ONCE, AT LOAD. It runs a prime search from `ngram_vocab_size_base`
// (20,000,000 on the released config) and it does not change between steps;
// #2336's third risk on the GDN adapter is the same shape — a per-step rebuild
// of something the load already knew.
//
// `ple_layer_index` is the index INTO `ple.layer_ids_zero_based`, upstream's
// `config.ple_layer_ids.index(layer_idx + 1)`, and it is what the layer
// multipliers are derived from. It is not the decoder layer index.
//
// REFUSES BY NAME when the config STATES per-head vocabulary sizes that
// disagree with the prime chain. A `qwen4exp` GGUF states them outright
// (`qwen4exp.ple.head_vocab_sizes`) because llama.cpp's converter reads them off
// the checkpoint's own buffers, and a `config.json` states none. Where the
// source states them they are the authority for what the shipped table was
// built against, so a disagreement is a real one: the offsets would address the
// wrong rows of a 320-million-row table and every gathered vector would be
// somebody else's, with no shape error and no crash.
qwen4_exp::NGramTableLayout Qwen4ExpPleLayout(const Qwen4ExpParams& p,
                                              int64_t ple_layer_index);

// One `Qwen4ExpTextPLELayer.forward` (:1169-1189), end to end.
//
//   w            this layer's PLE weights, as the loader produced them. Every
//                `norm_*` is the RAW HuggingFace gamma; the `+1` is applied here.
//   ngram_table  `Qwen4ExpWeights::ngram_table`, `[padded_vocab, embed_dim/heads]`.
//                Passed separately because it is MODEL-level, not layer-level:
//                it is one 51.2 G-parameter table and `Qwen4ExpPleWeights` does
//                not own it. May be block-quantized; `vt::Embedding` decodes one
//                row per gathered id rather than expanding the table.
//   layout       from `Qwen4ExpPleLayout`, built once at load.
//   hidden       [T, hc*hidden_size] — the hc-WIDE stream, at the stream dtype
//                (f32 or bf16). NOT the collapsed hidden state; see the output
//                struct above.
//   input_ids    HOST [T], this step's token ids. Host because the hash is.
//                Every id must be in `[0, vocab_size)`: the n-gram mix is bounded
//                below 2^63 only while they are, and an out-of-range id overflows
//                int64 and diverges in SILENCE. `BuildNGramIds` refuses it.
//   conv_mask    HOST [T] of 0/1, or nullptr. `None` in steady-state decode.
//                A masked position must ALSO carry EOS in `input_ids`; that pair
//                is refused by name here.
//   caches       this sequence's two states; see the struct.
//   past_len     how many tokens the sequence already holds. ZERO is the seeding
//                predicate, not merely a position: at `past_len == 0` this block
//                seeds `tokens` with `eos_token_id` and zeroes the conv ring,
//                which is `PleSequenceState::Reset` and is bit-identical to
//                upstream's first-call branch.
Qwen4ExpPleBlockOutput RunQwen4ExpPleBlock(dense_attn::Dev d, const Qwen4ExpPleWeights& w,
                                           const OwnedTensor& ngram_table,
                                           const Qwen4ExpParams& p,
                                           const qwen4_exp::NGramTableLayout& layout,
                                           const vt::Tensor& hidden, const int64_t* input_ids,
                                           const unsigned char* conv_mask,
                                           const Qwen4ExpPleCaches& caches, int64_t past_len);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_PLE_BLOCK_H_
