// Qwen4-Exp W5e-2 — `Qwen4ExpTextPLELayer` as one production block. See
// `qwen4_exp_ple_block.h` for why this file exists, which two silent contracts
// it carries, and what it deliberately does not cover.
//
// ALGORITHM ORACLE: transformers 5.16.0 (this row's accepted lane pin, sha256
// `77fec77d…c459`), `models/qwen4_exp/modeling_qwen4_exp.py`. Every line below
// cites the upstream line it mirrors, re-derived by reading the pinned file.
// OP ORACLE: vLLM, through the `vt::` primitives — this block introduces no
// arithmetic of its own, which is the whole point of it being a composition
// rather than a kernel.
#include "vllm/model_executor/models/qwen4_exp_ple_block.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // ResidentWeight
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using vt::DType;
using vt::Tensor;

// A contiguous ROW-RANGE view of a contiguous tensor, reshaped. Same helper the
// QSA block carries, and for the same reason: the element count must agree,
// which is the check that stops a reshape from quietly renaming a stride.
Tensor RowsView(const Tensor& t, int64_t start, int64_t count,
                const std::vector<int64_t>& shape) {
  VT_CHECK(t.rank >= 1 && t.IsContiguous(),
           "qwen4_exp ple block: RowsView needs a contiguous tensor");
  VT_CHECK(start >= 0 && count >= 0 && start + count <= t.shape[0],
           "qwen4_exp ple block: RowsView range outside the tensor");
  int64_t row_elems = 1;
  for (int i = 1; i < t.rank; ++i) row_elems *= t.shape[i];
  int64_t want = 1;
  for (int64_t s : shape) want *= s;
  VT_CHECK(want == count * row_elems,
           "qwen4_exp ple block: RowsView shape does not cover the rows it names");
  return dense_attn::MakeTensor(
      static_cast<char*>(t.data) +
          static_cast<size_t>(start * row_elems) * vt::SizeOf(t.dtype),
      t.dtype, t.device, shape);
}

// The whole tensor under a different shape, same bytes.
Tensor Reshape(const Tensor& t, const std::vector<int64_t>& shape) {
  return RowsView(t, 0, t.rank == 0 ? 0 : t.shape[0], shape);
}

// A VIEW with caller-chosen strides. `vt::BatchedMatmul` constrains only the
// innermost stride, which is what lets the two [T, hc*H] buffers be read as
// [T*hc, 1, H] and [T*hc, H, 1] with no copy — #2336's claim, and the reason
// the `:1180` dot needs no new op.
Tensor StridedView(const Tensor& t, const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& stride) {
  Tensor v;
  v.data = t.data;
  v.dtype = t.dtype;
  v.device = t.device;
  v.rank = static_cast<int>(shape.size());
  for (int i = 0; i < v.rank; ++i) {
    v.shape[i] = shape[static_cast<size_t>(i)];
    v.stride[i] = stride[static_cast<size_t>(i)];
  }
  return v;
}

}  // namespace

qwen4_exp::PleGeometry Qwen4ExpPleGeometry(const Qwen4ExpParams& p) {
  qwen4_exp::PleGeometry g;
  g.hidden_size = p.hidden_size;
  g.hc_count = p.hc_count;
  g.ple_embed_dim = p.ple.embed_dim;
  g.ple_conv_kernel_size = p.ple.conv_kernel_size;
  g.ngram_size = p.ple.ngram_size;
  g.heads_per_ngram = p.ple.heads_per_ngram;
  g.ngram_vocab_size_base = p.ple.ngram_vocab_size_base;
  g.make_ngram_vocab_size_divisible_by = p.ple.make_ngram_vocab_size_divisible_by;
  g.vocab_size = p.vocab_size;
  // `-1` is `PleGeometry`'s UNSET SENTINEL and W2 refuses it rather than
  // defaulting it, because eos is the second id in the hash mix and there is no
  // safe value to pick. Copied through unchanged so that refusal fires with the
  // name of the config key that is missing, instead of being masked here.
  g.eos_token_id = p.eos_token_id;
  g.seed = p.ple.seed;
  g.rms_norm_eps = p.rms_norm_eps;
  return g;
}

qwen4_exp::NGramTableLayout Qwen4ExpPleLayout(const Qwen4ExpParams& p,
                                              int64_t ple_layer_index) {
  VT_CHECK(ple_layer_index >= 0,
           "qwen4_exp ple layout: ple_layer_index must be >= 0; it is the index INTO "
           "ple_layer_ids (upstream's `config.ple_layer_ids.index(layer_idx + 1)`), "
           "not the decoder layer index");
  const qwen4_exp::PleGeometry geom = Qwen4ExpPleGeometry(p);
  qwen4_exp::NGramTableLayout layout =
      qwen4_exp::BuildNGramTableLayout(geom, ple_layer_index);

  // THE STATED SET IS THE AUTHORITY, AND THE CROSS-CHECK ONLY RUNS WHERE IT
  // CAN BE TRUE. W5g (#2031) corrects both halves of what W5e-2 landed here.
  //
  // W5e-2 always built the layout from the prime chain and then REFUSED when a
  // stated set disagreed with it. That is right for a `config.json`, which
  // states `ngram_vocab_size_base` and no sizes. It is wrong for a `qwen4exp`
  // GGUF, which states the resolved sizes and NO base
  // (`qwen4_exp_gguf_weights.cpp`: "the resolved arrays are the authority here,
  // because they are what the shipped tensor was built against"), so the chain
  // ran from a DEFAULTED 20,000,000 and the comparison was against a default
  // rather than against the file. Two consequences, and only the first was
  // visible:
  //
  //   1. Every `qwen4exp` GGUF whose base is not 20,000,000 was refused by
  //      name, correctly loaded weights and all. That is what stopped W5f's
  //      reachability case inside layer 1.
  //   2. Had it not refused, this was a HEAP OVER-READ, not merely wrong rows.
  //      `head_offsets` and `padded_vocab_size` would have been the chain's —
  //      hundreds of millions — while `NgramTableRows`
  //      (`qwen4_exp_weights.cpp`) sized the actual tensor from the STATED set.
  //      AND THE BOUNDS CHECK DOES NOT CATCH IT: `qwen4_exp_ple.cpp:389` tests
  //      the row against `layout.padded_vocab_size`, the LAYOUT's own value, so
  //      it is computed from the same wrong number that produced the row and it
  //      passes. On the synthetic fixture that is a row index in
  //      `[0, 40000026)` indexing a 128-row allocation. The refusal was the only
  //      thing standing between the GGUF arm and reading off the end of the
  //      buffer. The released checkpoint escapes both because its own base IS
  //      20,000,000, so the chain reproduces the file — which is why nothing
  //      saw this until a forward ran the PLE layer on a small fixture.
  //
  // So: where the source STATES the sizes they build the layout, exactly as
  // `NgramTableRows` already treats them and as `Qwen4ExpPleParams` already
  // documents ("Where the source states them they are the AUTHORITY"). The
  // prime chain is derived only where the source states nothing.
  const std::vector<int64_t>& stated = p.ple.head_vocab_sizes;
  if (!stated.empty()) {
    VT_CHECK(static_cast<int64_t>(stated.size()) == p.ple.ngram_heads(),
             "qwen4_exp ple layout: the config states " + std::to_string(stated.size()) +
                 " per-head vocabulary sizes but this geometry has " +
                 std::to_string(p.ple.ngram_heads()) +
                 " n-gram heads ((ngram_size - 1) * heads_per_ngram)");
    // THE STATED ARRAY COVERS ONE PLE LAYER, and refusing is the honest
    // reading rather than a limitation invented here. Upstream derives head
    // sizes from a GLOBAL head index — `ple_layer_index * heads + head_idx`
    // (`BuildNGramTableLayout`) — so every PLE layer has a DIFFERENT set,
    // while the container states a single flat array of `ngram_heads` entries
    // and says nothing about which layer it belongs to. The released
    // checkpoint has exactly one PLE layer, so index 0 is the only one the
    // array can mean. A file with two would silently give layer 1 layer 0's
    // vocabulary and gather every later head's rows from the wrong offset.
    VT_CHECK(ple_layer_index == 0,
             "qwen4_exp ple layout: the source STATES per-head vocabulary "
             "sizes, but a stated array is one flat set of " +
                 std::to_string(stated.size()) +
                 " entries and upstream derives a DIFFERENT set for every PLE "
                 "layer (from the global head index "
                 "`ple_layer_index * ngram_heads + head`). This is PLE layer "
                 "index " +
                 std::to_string(ple_layer_index) +
                 ", so the stated array cannot be known to describe it. Only "
                 "index 0 is unambiguous, and the released checkpoint has one "
                 "PLE layer");
    // THE CROSS-CHECK SURVIVES, NARROWED TO THE SOURCES THAT CAN SUPPORT IT.
    // A disagreement is not a shape error and would not throw anywhere
    // downstream: `head_offsets` is an exclusive prefix sum, so ONE wrong size
    // shifts every later head's rows inside a table whose row count both sides
    // agree on. Every gathered vector would be somebody else's, with no crash.
    // It runs only when the SOURCE stated the base, because otherwise the
    // chain's inputs are a default and the comparison measures nothing about
    // the file.
    //
    // AND NO SHIPPED SOURCE SATISFIES BOTH HALVES TODAY. Said here rather than
    // left for the next reader to derive: the guard needs a stated head-size
    // set AND a stated base. A `qwen4exp` GGUF states the sizes and never the
    // base — `Qwen4ExpHfConfigFromGguf` writes `ple_head_vocab_sizes` and no
    // `ngram_vocab_size_base`, so `ParseQwen4ExpParams` leaves this flag false.
    // The released `config.json` states the base and never the sizes, so
    // `stated` is empty and the whole branch is skipped. On this head the only
    // thing that drives the loop below is `test_qwen4_exp_ple_block.cpp`'s
    // `GoldenParams()`, which sets the flag by hand.
    //
    // IT IS KEPT RATHER THAN DELETED, and the reason is not inertia. A source
    // that states both is exactly the case it protects — a converter that
    // writes the resolved arrays AND the base it derived them from, which is
    // one commit away in llama.cpp #27742's own converter — and the failure it
    // catches is silent: `head_offsets` is an exclusive prefix sum, so one
    // wrong size re-points every later head at another head's rows with no
    // shape error. Deleting a correct invariant to make the remaining code
    // reachable is the wrong direction. `MODEL-MM-QWEN4-EXP` owns either
    // producing a source that reaches it or removing it; see `## Owed` in
    // `.agents/specs/qwen4-exp-flash-next.md`.
    if (p.ple.ngram_vocab_size_base_stated) {
      for (size_t h = 0; h < stated.size(); ++h) {
        VT_CHECK(stated[h] == layout.head_vocab_sizes[h],
                 "qwen4_exp ple layout: head " + std::to_string(h) +
                     " vocabulary size disagrees — the source STATES " +
                     std::to_string(stated[h]) + " and the prime chain from "
                     "ngram_vocab_size_base " +
                     std::to_string(p.ple.ngram_vocab_size_base) + " derives " +
                     std::to_string(layout.head_vocab_sizes[h]) +
                     ". The stated set is what the shipped table was built against, so "
                     "this is a real disagreement and not a rounding one: the head "
                     "offsets are an exclusive prefix sum, so one wrong size silently "
                     "re-points every later head at another head's rows");
      }
    }
    // REBUILD FROM THE STATED SET. `layer_multipliers` stays the derivation's:
    // it is a splitmix chain over `vocab_size`, `ngram_size`, the layer index
    // and `seed`, none of which the stated array carries.
    layout.head_vocab_sizes.assign(stated.begin(), stated.end());
    layout.head_offsets.clear();
    layout.head_offsets.reserve(stated.size());
    layout.total_vocab_size = 0;
    for (int64_t sz : layout.head_vocab_sizes) {
      layout.head_offsets.push_back(layout.total_vocab_size);
      layout.total_vocab_size += sz;
    }
    const int64_t divisor = p.ple.make_ngram_vocab_size_divisible_by;
    VT_CHECK(divisor > 0,
             "qwen4_exp ple layout: make_ngram_vocab_size_divisible_by must be "
             "positive to pad a stated vocabulary");
    layout.padded_vocab_size =
        ((layout.total_vocab_size + divisor - 1) / divisor) * divisor;
    // THE CONTAINER'S OWN OFFSETS, where it states them. `ParseQwen4ExpParams`
    // has already checked them against the stated sizes; this asserts the
    // array that actually reaches the gather is that one, so the two reads
    // cannot drift apart in a later edit.
    if (!p.ple.head_offsets.empty()) {
      VT_CHECK(p.ple.head_offsets.size() == layout.head_offsets.size(),
               "qwen4_exp ple layout: the source states " +
                   std::to_string(p.ple.head_offsets.size()) +
                   " head offsets against " +
                   std::to_string(layout.head_offsets.size()) + " heads");
      for (size_t h = 0; h < layout.head_offsets.size(); ++h) {
        VT_CHECK(p.ple.head_offsets[h] == layout.head_offsets[h],
                 "qwen4_exp ple layout: head " + std::to_string(h) +
                     " offset disagrees — the source STATES " +
                     std::to_string(p.ple.head_offsets[h]) +
                     " and the exclusive prefix sum over the stated sizes puts "
                     "it at " + std::to_string(layout.head_offsets[h]));
      }
    }
  }
  return layout;
}

Qwen4ExpPleBlockOutput RunQwen4ExpPleBlock(Dev d, const Qwen4ExpPleWeights& w,
                                           const OwnedTensor& ngram_table,
                                           const Qwen4ExpParams& p,
                                           const qwen4_exp::NGramTableLayout& layout,
                                           const Tensor& hidden, const int64_t* input_ids,
                                           const unsigned char* conv_mask,
                                           const Qwen4ExpPleCaches& caches,
                                           int64_t past_len) {
  const int64_t T = hidden.rank == 2 ? hidden.shape[0] : 0;
  const int64_t H = p.hidden_size;
  const int64_t hc = p.hc_count;
  const int64_t W = p.stream_width();
  const int64_t E = p.ple.embed_dim;
  const int64_t heads = p.ple.ngram_heads();
  const int64_t hd = p.ple.head_dim_per_ngram();
  const int64_t K = p.ple.conv_kernel_size;
  const int64_t dilation = p.ple.ngram_size;
  const int64_t L = p.ple.short_conv_state_len();
  const int64_t ctx = p.ple.ngram_size - 1;
  const auto eps = static_cast<float>(p.rms_norm_eps);

  VT_CHECK(T > 0, "qwen4_exp ple block: T must be positive");
  VT_CHECK(hidden.rank == 2 && hidden.shape[1] == W,
           "qwen4_exp ple block: hidden must be [T, hc_count*hidden_size] = [T," +
               std::to_string(W) +
               "] — the hc-WIDE hyper-connection stream, never the collapsed "
               "hidden state (upstream's own docstring: the returned tensor has "
               "shape (batch, seq, hc_count * hidden_size))");
  VT_CHECK(hidden.IsContiguous(), "qwen4_exp ple block: hidden must be contiguous");
  // THE STREAM DTYPE IS INHERITED, NOT CHOSEN (AGENTS.md, "Inherit vLLM
  // defaults"): every buffer below carries `hidden`'s dtype, so a bf16 model
  // moves bf16 bytes through all of it. The ONE f32 buffer is the [T, hc] score,
  // and it is f32 because `vt::Qwen4ExpPleGate` refuses a bf16 score — the
  // argument of a sigmoid must not be rounded, which is the contract that op and
  // `vt::SigmoidGateBf16` both state.
  const DType dt = hidden.dtype;
  VT_CHECK(dt == DType::kF32 || dt == DType::kBF16,
           "qwen4_exp ple block: hidden must be f32 or bf16");
  VT_CHECK(input_ids != nullptr, "qwen4_exp ple block: input_ids must not be null");
  VT_CHECK(past_len >= 0, "qwen4_exp ple block: past_len must not be negative");
  VT_CHECK(heads > 0 && hd > 0 && E == heads * hd,
           "qwen4_exp ple block: ple_embed_dim must be ngram_heads * head_dim_per_ngram");

  // The n-gram table is MODEL-level, and it may be block-quantized: W6a made
  // `GgufTensorRole::kEmbeddingTable` keep-quant eligible and taught
  // `vt::Embedding` to decode one row per gathered id, which is what turns
  // 102.4 GB of expanded bf16 into 28.8 GB of resident blocks. Nothing here
  // expands it.
  VT_CHECK(ngram_table.rank == 2 && ngram_table.shape[1] == hd,
           "qwen4_exp ple block: the n-gram table must be [padded_vocab, "
           "ple_embed_dim/ngram_heads] = [V," + std::to_string(hd) + "]");
  VT_CHECK(ngram_table.shape[0] == layout.padded_vocab_size,
           "qwen4_exp ple block: the n-gram table has " +
               std::to_string(ngram_table.shape[0]) + " rows and the layout addresses " +
               std::to_string(layout.padded_vocab_size) +
               "; the head offsets are an exclusive prefix sum over the LAYOUT, so a "
               "table of another height is a different table");

  const Tensor& conv_state = caches.conv_state;
  const Tensor& tokens = caches.tokens;
  VT_CHECK(conv_state.rank == 3 && conv_state.shape[1] == W && conv_state.shape[2] == L,
           "qwen4_exp ple block: conv_state must be [N," + std::to_string(W) + "," +
               std::to_string(L) +
               "] — (kernel-1)*ngram_size deep, NOT kernel-1: the conv is DILATED, so "
               "the state is nine columns at the released config and not three");
  // ─── THE RING'S DTYPE IS THE STREAM'S, AND THE ORACLE SETTLES IT (W5k) ────
  // This required f32 and the recurrent group publishes bf16, which is the pair
  // that refused `past_len > 0` for four waves. It is not a free choice on either
  // side: transformers 5.16.0 (`.agents/oracles/transformers.md`, the `qwen4_exp`
  // lane pin) allocates each cache slot with the dtype of the tensor that first
  // reaches it — `cache_utils.py:1019-1023`,
  // `torch.zeros(..., dtype=conv_states.dtype, ...)` — and the tensor reaching
  // THIS slot is `hidden_states` at `modeling_qwen4_exp.py:1157-1159`. So the ring
  // carries the MODEL dtype, and running the pinned oracle at bf16 reports
  // `conv_states[1] dtype=torch.bfloat16` where the f32 arm reports `float32`.
  //
  // Requiring EQUALITY with `dt` rather than merely admitting bf16 is the stronger
  // mirror and the one upstream's own construction states: a ring at any dtype
  // other than the stream's is a slot upstream cannot produce. It also makes the
  // publisher's `conv_dtype` and this consumer's stream ONE fact instead of two
  // that can drift apart in silence.
  VT_CHECK(conv_state.dtype == dt,
           std::string("qwen4_exp ple block: conv_state carries ") +
               vt::Name(conv_state.dtype) +
               " and the hyper-connection stream carries " + vt::Name(dt) +
               "; upstream types this cache slot from the model dtype "
               "(cache_utils.py:1019-1023 over the hidden_states at "
               "modeling_qwen4_exp.py:1157-1159), so the ring and the stream are "
               "ONE dtype and never two");
  VT_CHECK(conv_state.IsContiguous(),
           "qwen4_exp ple block: conv_state must be contiguous");
  VT_CHECK(tokens.rank == 2 && tokens.shape[1] == ctx && tokens.dtype == DType::kI64 &&
               tokens.IsContiguous(),
           "qwen4_exp ple block: the n-gram history must be a contiguous i64 [N," +
               std::to_string(ctx) +
               "] — int64 because it holds TOKEN IDS, and a float store would ROUND them");
  // ─── RESIDENCY IS THE ENGINE'S, AND THE HASH STAGES AROUND IT (W5k) ───────
  // This refused anything but a HOST tensor, and the recurrent group publishes a
  // DEVICE one — the second half of the pair that refused `past_len > 0`. The
  // refusal stated a true fact about THIS TREE (the splitmix64 hash at
  // `qwen4_exp::BuildNGramIds` is a host int64 computation and no `vt::` op
  // computes it) and turned it into a requirement on the CACHE, which is the
  // wrong object: where a hash runs does not decide where upstream stores state.
  //
  // Upstream stores it on the COMPUTE DEVICE. `modeling_qwen4_exp.py:1070` takes
  // `input_ids.long()` and `:1089-1091` hands exactly that to
  // `update_conv_state(..., state_idx=2)`, which allocates the slot with
  // `device=conv_states.device` (`cache_utils.py:1019-1023`) — so the history
  // lives wherever `input_ids` live, which on a CUDA run is the GPU. There is no
  // host-residency contract to mirror, and the publisher was right.
  //
  // So the block ACCEPTS either residency and stages the row instead. The state
  // is `ngram_size - 1` int64s — 16 bytes at the released config — so the
  // round trip is two copies of one cache line per PLE layer per step, against a
  // gather over the n-gram table that moves orders of magnitude more. A device
  // splitmix64 would remove even that and is still worth having; it is recorded
  // under `## Owed` as an optimization now, rather than as the blocker it was
  // recorded as before.
  VT_CHECK(tokens.device.type == vt::DeviceType::kCPU ||
               tokens.device == d.q.device,
           "qwen4_exp ple block: the n-gram history must be host-resident or live "
           "on this step's own device; it is on neither");
  VT_CHECK(caches.state_row >= 0 && caches.state_row < conv_state.shape[0] &&
               caches.state_row < tokens.shape[0],
           "qwen4_exp ple block: state_row is outside one of the two caches");

  // ─── THE SEEDING PREDICATE (upstream :1073-1076, :1080-1088) ───────────────
  // `past_len == 0` IS `has_previous_state(layer_idx, state_idx=2) == False`.
  //
  // ZERO IS A VALID TOKEN ID, which is the whole reason this branch exists.
  // `CacheBuffer` zero-fills, exactly as upstream's `update_conv_state` pads
  // with zeros, and upstream works around its own pad with an explicit EOS
  // left-pad at :1086-1087. A port that trusted the zero-filled cache would hash
  // token 0 into the first `ngram_size - 1` positions of every sequence and get
  // a fluent wrong answer.
  //
  // The conv ring is zeroed in the same branch, which is NOT the same statement:
  // a zeroed conv state is bit-identical to upstream's first call
  // (cache_utils.py:1053-1060 left-zero-pads), so this only matters when the
  // runner hands back a slot a previous sequence used. Together the two are
  // exactly `qwen4_exp::PleSequenceState::Reset`.
  // THE ROW IS STAGED ON HOST whichever side it came from. When the cache is
  // already host-resident `stage` is filled from it and written back to it, which
  // is byte-identical to the direct pointer walk this replaces; when it is on the
  // device the same two copies cross the bus. One code path either way, because
  // two would be two places for the write-back to be forgotten.
  const bool tokens_on_host = tokens.device.type == vt::DeviceType::kCPU;
  const size_t hist_bytes = static_cast<size_t>(ctx) * sizeof(int64_t);
  auto* tokens_row = static_cast<char*>(tokens.data) +
                     static_cast<size_t>(caches.state_row) * hist_bytes;
  std::vector<int64_t> stage(static_cast<size_t>(ctx), 0);
  int64_t* hist = stage.data();
  if (past_len == 0) {
    // ZERO IS A VALID TOKEN ID: seed with EOS, never with the zero-filled cache.
    for (int64_t i = 0; i < ctx; ++i) hist[i] = p.eos_token_id;
    // The ring is zeroed by BYTES, which is dtype-agnostic and residency-agnostic
    // in one stroke: all-zero bits is +0.0 in f32, bf16 and f16 alike, and the
    // `Ptr<float>()` walk this replaces was wrong on BOTH counts the moment the
    // ring became a bf16 device buffer.
    d.b.Memset(d.q,
               static_cast<char*>(conv_state.data) +
                   static_cast<size_t>(caches.state_row) * static_cast<size_t>(W * L) *
                       vt::SizeOf(conv_state.dtype),
               0, static_cast<size_t>(W * L) * vt::SizeOf(conv_state.dtype));
  } else if (tokens_on_host) {
    std::memcpy(hist, tokens_row, hist_bytes);
  } else {
    // THE DRAIN IS THE READ'S OTHER HALF, NOT A DEFENSIVE ADDITION (#2496).
    // `Backend::Copy` ENQUEUES: on CUDA it is `cudaMemcpyAsync` on the queue's
    // stream (src/vt/cuda/cuda_backend.cu:116-118), so `stage` holds its own
    // zero-fill until the copy lands. The host reads it a few lines below, at
    // `hash_state.tokens.assign`, and a zeroed history hashes token id 0 into
    // every context slot -- the exact "fluent wrong answer" the seeding branch
    // above warns about, on every step and only when `past_len != 0`.
    //
    // `CUDA_LAUNCH_BLOCKING=1` DOES NOT COVER THIS. It serialises kernel
    // launches; it does not make `cudaMemcpyAsync` synchronous. That is why
    // #2496 reproduces under the serialisation that suppresses #2476.
    //
    // The sibling translation unit already states this rule at the definition of
    // `StageHostWords` -- "nothing is readable until the caller synchronises" --
    // and its `HostWords` wrapper ends in exactly this call. `Synchronize` is a
    // no-op on the base backend, so the host branch above pays nothing.
    d.b.Copy(d.q, hist, tokens_row, hist_bytes);
    d.b.Synchronize(d.q);
  }

  // ─── THE MASK'S PAIRED HALF, ENFORCED (spec `## Owed`, since W2) ───────────
  // The activations are masked at :1186-1187 AND `input_ids` must already carry
  // EOS at every masked position, because the hash at :1101-1106 reads token ids
  // and not activations. Masking only the activations leaks padding into the
  // hash, and nothing downstream can see it: the ids are in range, the gather
  // succeeds, and the answer is wrong. This is the first enforcer that half has
  // ever had.
  if (conv_mask != nullptr) {
    for (int64_t t = 0; t < T; ++t) {
      if (conv_mask[t] != 0) continue;
      VT_CHECK(input_ids[t] == p.eos_token_id,
               "qwen4_exp ple block: conv_mask masks token " + std::to_string(t) +
                   " but input_ids[" + std::to_string(t) + "] is " +
                   std::to_string(input_ids[t]) + " rather than eos_token_id " +
                   std::to_string(p.eos_token_id) +
                   ". Masking is a PAIRED obligation: the n-gram hash reads token ids, "
                   "not activations, so a masked position whose id is not EOS leaks "
                   "padding into the hash and the answer is wrong with no error");
    }
  }

  // ─── :1176  the n-gram ids, then the gather ────────────────────────────────
  // The hash is host int64 bit-mixing (`_splitmix64`, :979-983) over TOKEN IDS,
  // and `qwen4_exp::BuildNGramIds` is W2's port of it. This is its first caller
  // outside its own translation unit. It advances the history in place, which is
  // upstream's `update_conv_state(..., state_idx=2)`.
  const qwen4_exp::PleGeometry geom = Qwen4ExpPleGeometry(p);
  qwen4_exp::PleSequenceState hash_state;
  hash_state.tokens.assign(hist, hist + ctx);
  std::vector<int64_t> ids(static_cast<size_t>(T * heads));
  qwen4_exp::BuildNGramIds(geom, layout, input_ids, T, &hash_state, ids.data());
  for (int64_t i = 0; i < ctx; ++i) hist[i] = hash_state.tokens[static_cast<size_t>(i)];
  // THE WRITE-BACK IS `update_conv_state(..., state_idx=2)` (:1089-1091) and it is
  // what makes the NEXT step's `past_len > 0` read a real history rather than the
  // EOS seed. Unconditional, because the staging buffer is a copy on both arms.
  if (tokens_on_host) {
    std::memcpy(tokens_row, hist, hist_bytes);
  } else {
    // AND THE WRITE-BACK IS DRAINED BEFORE ITS SOURCE DIES (#2496). `stage` is a
    // function-local vector; this function returns a few lines below. An
    // enqueued `cudaMemcpyAsync` still scheduled against it is a DMA reading
    // freed host memory, which is a use-after-free rather than a slow copy.
    d.b.Copy(d.q, tokens_row, hist, hist_bytes);
    d.b.Synchronize(d.q);
  }

  DBuf d_ids(d, DType::kI64, {T * heads}, ids.data());
  // :1114 — `self.ngram_embedding(ngram_ids).flatten(-2)`. The gather emits
  // [T*heads, hd] and the flatten is the same bytes viewed [T, heads*hd].
  DBuf emb(d, dt, {T * heads, hd});
  {
    Tensor table = dense_attn::ResidentWeight(d, ngram_table,
                                              {ngram_table.shape[0], ngram_table.shape[1]});
    Tensor out = emb.t();
    vt::Embedding(d.q, out, table, d_ids.t());
  }
  Tensor embeddings = Reshape(emb.t(), {T, E});

  // ─── :1177  key_normed = norm_key(key_proj(embeddings)) ────────────────────
  DBuf key(d, dt, {T, W});
  vt::MatmulBT(d.q, key.t(), embeddings, dense_attn::ResidentWeight(d, w.key_proj, {W, E}));
  DBuf key_normed(d, dt, {T, W});
  // `gemma = true` and `group_size = hidden_size` are the two halves of
  // `Qwen4ExpTextRMSNorm(hc*H, group_size=H)` (:158-181, :1138). The gamma is
  // stored RAW and the `+1` belongs here (#2218); the group extent is what makes
  // each of the hc streams normalize independently, and `vt::RmsNormGroup`
  // refuses `group_size == 0` rather than degenerating to a whole-row norm.
  const vt::RmsNormGroupArgs norm_args{eps, /*gemma=*/true, H};
  vt::RmsNormGroup(d.q, key_normed.t(), key.t(),
                   dense_attn::ResidentWeight(d, w.norm_key, {W}), norm_args);

  // ─── :1178  value = value_proj(embeddings) ─────────────────────────────────
  DBuf value(d, dt, {T, H});
  vt::MatmulBT(d.q, value.t(), embeddings,
               dense_attn::ResidentWeight(d, w.value_proj, {H, E}));

  // ─── :1179  query_normed = norm_query(hidden_states) ───────────────────────
  DBuf query_normed(d, dt, {T, W});
  vt::RmsNormGroup(d.q, query_normed.t(), hidden,
                   dense_attn::ResidentWeight(d, w.norm_query, {W}), norm_args);

  // ─── :1180  the per-(t, j) dot, WITHOUT a new op ───────────────────────────
  // `(key_normed * query_normed).sum(-1, keepdim=True)` over the [T, hc, H]
  // unflatten is `vt::BatchedMatmul` over VIEWS: [T*hc, 1, H] x [T*hc, H, 1] ->
  // [T*hc, 1, 1]. Only the innermost dim must be unit-stride, so both views are
  // free and neither buffer is copied or tiled. A private scoring loop here
  // would be the parallel path AGENTS.md "Shared seams" forbids.
  //
  // THE SCORE IS f32 AND THE OPERANDS ARE NOT. `BatchedMatmul` accumulates in
  // f32 whatever the operands' width, and the gate needs an unrounded sigmoid
  // argument; the `/ sqrt(hidden_size)` divide is NOT applied here — it rides
  // into the gate as `gate_divisor`, so the op performs upstream's own division
  // rather than a multiply by a pre-computed reciprocal.
  DBuf score(d, DType::kF32, {T, hc});
  {
    const int64_t g = T * hc;
    Tensor a = StridedView(key_normed.t(), {g, 1, H}, {H, H, 1});
    Tensor b = StridedView(query_normed.t(), {g, H, 1}, {H, 1, 1});
    Tensor o = StridedView(score.t(), {g, 1, 1}, {1, 1, 1});
    vt::BatchedMatmul(d.q, o, a, b);
  }

  // ─── :1181-1182, flattened at :1184 ────────────────────────────────────────
  // The signed square root and the broadcast sigmoid scale, as ONE op (W5e-1).
  // The clamp is applied BEFORE the square root, so the floor on |gate| is 1e-3
  // and not 1e-6; exactly zero maps to zero because `sign(0) == 0`, and a fully
  // masked row reaches that origin.
  vt::Qwen4ExpPleGateArgs gate_args;
  gate_args.gate_divisor = static_cast<float>(std::sqrt(static_cast<double>(H)));
  DBuf gated(d, dt, {T, W});
  vt::Qwen4ExpPleGate(d.q, gated.t(), score.t(), value.t(), gate_args);

  // ─── :1183  gated_value_normed = norm_conv(gated_value.flatten(-2)) ────────
  // THE FORK the spec warns about: the conv sees this NORMED copy and the skip
  // term added back at :1188 is the UN-NORMED one, and it is the normed copy the
  // nine-column state keeps.
  DBuf gated_normed(d, dt, {T, W});
  vt::RmsNormGroup(d.q, gated_normed.t(), gated.t(),
                   dense_attn::ResidentWeight(d, w.norm_conv, {W}), norm_args);

  // ─── :1185-1187  apply_mask_to_padding_states, to BOTH ─────────────────────
  // `hidden_states * attention_mask[:, :, None]` (:211) is a per-TOKEN row
  // scale, and the shared surface has no per-row broadcast multiply:
  // `vt::MulColVecF32` scales per output COLUMN and `vt::SigmoidGateBf16`
  // refuses by element count. A masked position is padding and therefore rare,
  // so this is `vt::MulScalar` on the row view — upstream's own two lines at
  // upstream's own two sites, rather than a new general op nothing else calls.
  //
  // BOTH tensors, and masking only one is a real port defect the goldens see:
  // `gated_value` is the skip term and `gated_value_normed` is what enters the
  // conv AND what the ring keeps, so an interior masked position moves output
  // rows t, t+3, t+6 and t+9 as well as its own.
  if (conv_mask != nullptr) {
    for (int64_t t = 0; t < T; ++t) {
      if (conv_mask[t] != 0) continue;
      Tensor g_row = RowsView(gated.t(), t, 1, {1, W});
      Tensor n_row = RowsView(gated_normed.t(), t, 1, {1, W});
      vt::MulScalar(d.q, g_row, g_row, 0.0);
      vt::MulScalar(d.q, n_row, n_row, 0.0);
    }
  }

  // ─── :1188  output = gated_value + self._short_conv(gated_value_normed) ────
  // `vt::Qwen4ExpPleConv` is `_short_conv` (:1150-1167) INCLUDING the state read
  // and the write-back `update_conv_state(..., state_idx=1)` performs around it,
  // so the ring is advanced by the op and not by this file. `state_idx` is
  // resolved by which tensor is passed, and `conv_state_indices` names the cache
  // ROW this sequence owns — a different axis with a deliberately different name
  // (see the op's contract).
  const int32_t qsl_host[2] = {0, static_cast<int32_t>(T)};
  const auto row_host = static_cast<int32_t>(caches.state_row);
  DBuf qsl(d, DType::kI32, {2}, qsl_host);
  DBuf row(d, DType::kI32, {1}, &row_host);
  DBuf conv_out(d, dt, {T, W});
  {
    vt::Qwen4ExpPleConvArgs cargs;
    cargs.dilation = dilation;
    Tensor state = conv_state;
    Tensor idx = row.t();
    vt::Qwen4ExpPleConv(d.q, conv_out.t(), gated_normed.t(),
                        dense_attn::ResidentWeight(d, w.conv1d, {W, K}), state, qsl.t(), &idx,
                        cargs);
  }

  DBuf out(d, dt, {T, W});
  vt::Add(d.q, out.t(), gated.t(), conv_out.t());

  Qwen4ExpPleBlockOutput r;
  r.tensor = out.t();
  r.storage = out.ReleaseShared();
  return r;
}

}  // namespace vllm
