// DeepSeek-V4.1-Flash W3a — **Engram**, as a portable host (CPU) reference.
//
// Row `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`, issue
// `ISSUE-LOCAL-01M2C40RNXB871VW0E560PVBFA`, spec
// `.agents/specs/deepseek-v4-1-flash.md` (`## Work breakdown`, wave W3a).
//
// Engram writes an n-gram lookup into the residual stream, gated by how well
// that lookup matches the stream. It is 39.8% of the released checkpoint's
// bytes and it has four separable pieces, each ported 1:1 with `file:line` on
// both sides:
//
//   (1) EngramLayout       — the prime-sized bucket layout of the hash tables.
//   (2) NgramHashState     — position -> the hash ids of the n-grams ending
//                            there, resolved over three tiers of history.
//   (3) EngramLookup       — gather fp8-e4m3 rows, apply ue8m0 per-32 block
//                            scales, emit bf16; sharded by complete hash heads.
//   (4) EngramPostWkv      — the sigmoid gate that injects the looked-up value
//                            into the `[T, hc_mult, hidden]` manifold.
//
// ─── WHERE IT SITS IN THE FORWARD ────────────────────────────────────────────
// The injection point is BETWEEN `mhc_post_tilelang` and
// `mhc_pre_delayed_tilelang` (`nvidia/model.py:340-348`), on the full
// `[T, hc_mult, hidden]` bf16 stream, "so the mix coefficients see the injected
// stream". That is exactly the manifold
// `include/vllm/model_executor/models/deepseek_v4_mhc.h` already models, so
// this file composes with MhcPost/MhcPre rather than duplicating either.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ vLLM `e77daef89e`) ───
// The oracle revision is the V4.1 completion commit named in the row spec's
// `## Upstream chain`; it is 566 commits PAST our parity pin `e126687a9a`, at
// which `deepseek_v4_1/` does not exist at all. So this is an AHEAD-OF-PIN
// forward port, the same shape `Qwen3_5ForCausalLM` carries today, and no gate
// here claims a denominator against the pinned oracle.
//
//   OURS                      <-  UPSTREAM (vllm/models/deepseek_v4_1/)
//   IsEngramPrime             <-  common/engram.py::_is_prime (:63-85)
//   FindNextEngramPrime       <-  ::find_next_prime (:88-93)
//   EngramLayoutFromConfig    <-  ::EngramLayout.__init__ (:178-204) +
//                                 ::EngramLayout.from_config (:206-210)
//   HashMultiplierBound       <-  ::compute_hash_multipliers (:147-166), the
//                                 `multiplier_bound` half only; see DECISION 2
//   ValidateCompressedTokenMap<-  ::build_compressed_token_map (:96-144) +
//                                 ::NgramHashState.__init__ (:414-421); see
//                                 DECISION 1
//   NgramHashState::WriteCache<-  ::_write_hash_cache_kernel (:213-236)
//   NgramHashState::Forward   <-  ::_hash_ids_kernel (:248-380) driven by
//                                 ::NgramHashState.forward (:465-546)
//   NgramHashState::EnsureCache<- ::NgramHashState.ensure_cache (:439-463)
//   MakeEngramEmbeddingShard  <-  ::ParallelEngramEmbedding.__init__ (:630-696)
//   LoadEngramHeadShard       <-  ::_engram_head_shard_weight_loader (:549-561)
//   EngramLookup              <-  ::_engram_lookup_kernel (:564-619) driven by
//                                 ::ParallelEngramEmbedding.lookup (:716-748)
//   EngramPostWkv             <-  ::_fused_engram_post_wkv_kernel (:765-854)
//                                 driven by ::Engram.forward (:972-1033)
//
// Upstream's own gate is `tests/kernels/test_engram.py` at the same revision,
// 778 lines. Its two scalar oracles `_reference_hashes` (:188-225) and
// `_reference_lookup` (:520-529) are ported verbatim into
// `tests/vllm/models/test_deepseek_v4_1_engram.cpp`, with their parameters,
// fixtures and `rtol=0, atol=0` tolerances preserved.
//
// ─── THE THREE CONSTANTS THAT ARE EASY TO GET SILENTLY WRONG ─────────────────
//
// 1. **The normalizer sentinel is `` (U+E000), a private-use
//    character, NOT the empty string**
//    (engram.py:109, with :106-108 saying why: a token that is
//    exactly one space must survive `Strip()` instead of collapsing to "" and
//    merging with unrelated tokens). Getting this wrong changes every
//    compressed id and therefore every hash, with no exception raised.
//
// 2. **`engram_vocab_size` and `engram_compressed_vocab_size` are unrelated.**
//    16,000,000 seeds the per-bucket PRIME SEARCH (:193, `current =
//    config.engram_vocab_size - 1`); 99,092 bounds the hash MULTIPLIER (:155,
//    `max_long // compressed_vocab_size // 2`). Swapping them does not raise;
//    it rehashes the tables.
//
// 3. **The ue8m0 scale decode here is the BITCAST form, not the arithmetic
//    one.** engram.py:613-614 is
//    `scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)`, whose
//    own comment reads "ue8m0 is a power of two, so its byte *is* the fp32
//    exponent field". Byte 0 therefore decodes to the bit pattern 0x00000000 =
//    **+0.0**. Our existing `vllm::E8M0ToF32`
//    (`src/vllm/model_executor/model_loader/mxfp4_dequant.cpp:15-22`) is the
//    ARITHMETIC form, `ldexp(1, byte - 127)`, and gives 2^-127 at byte 0. Both
//    are correct for their own arm — MXFP8 decodes arithmetically at
//    `quantization/utils/mxfp8_utils.py:66` — so this wave adds a SECOND entry
//    point, `vllm::E8M0BitsToF32`, and leaves `E8M0ToF32` and its TWO
//    production callers untouched: `mxfp4_dequant.cpp:63` and
//    `nvfp4_dequant.cpp:122`. That count is invocations of `vllm::E8M0ToF32`
//    under `src/` and `include/`; tests, comments and the unrelated
//    `E8M0ToF32Half` / `DE8M0ToF32Half` GGUF helpers are not callers of it.
//    An earlier revision of this comment said "six".
//    Upstream's own test never reaches the difference: its scale
//    bytes come from `randint(120, 134)` (test_engram.py:557,571), so byte 0
//    never occurs. Our port adds a deliberate byte-0 case beside the ported
//    parameters and says so at the call site.
//
// ─── W3a SCOPE, STATED SO NOBODY HAS TO INFER IT ─────────────────────────────
// THIS IS A HOST (CPU) REFERENCE AND IT IS NOT REACHED FROM ANY PRODUCTION
// ENTRY POINT AT THIS COMMIT. `deepseek_v41` has no registry entry, no loader
// arm and no `ModelRegistry::Forward` arm; W1 owns the registration and W8 the
// loader. Per AGENTS.md "Nothing lands dead" this is a STAGED SLICE landing
// unreached: the wiring is owned by row
// `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`, tracked by
// `ISSUE-LOCAL-01M2C40RNXB871VW0E560PVBFA`, and the row spec lists it under
// `## Owed`. Do not read the absent call site as an oversight.
//
// Deliberately NOT here: the compressed token map CONSTRUCTION (DECISION 1),
// the NumPy PCG64 multiplier derivation (DECISION 2), tensor-parallel
// collectives (the shard API is per rank and the test concatenates), sequence
// parallelism's `_engram_sp_rows_kernel` row shuffle, CPU-offload/UVA storage
// (a device-residency question with no host meaning), the `wkv` GEMM (which is
// an ordinary `ReplicatedLinear` and belongs to the linear seam, not here), and
// any device port (W5/W6/W7).
//
// ─── THE FIXTURE STANDS IN FOR TABLES IT CANNOT HOLD ─────────────────────────
// The released tables are 384,006,168 and 384,016,682 rows of 128 fp8 values
// plus their scales. A host reference cannot instantiate them and this one does
// not try: `EngramLookup` takes a caller-owned shard of whatever size, and the
// gate runs it at 4096 rows. So the gate proves the INDEXING, the head
// sharding, the bucket disjointness and the dequant arithmetic. It does NOT
// prove anything about the real tables' residency, bandwidth, TLB behaviour or
// load path — every one of those is owed to W5/W8 and none is claimed here.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace vllm::deepseek_v4_1 {

// Cache value for a token that takes no part in an n-gram (an image span).
// engram.py:60 `DEAD_ID = -1`.
inline constexpr int32_t kEngramDeadId = -1;

// The quantization block of the engram tables: one ue8m0 scale per 32
// consecutive columns. engram.py:635 `block_size: int = 32`.
inline constexpr int64_t kEngramQuantBlock = 32;

// ─── (0) the config surface ──────────────────────────────────────────────────

// The eight `engram_*` keys of `text_config`, plus the three model keys the
// gate needs. Named and defaulted from the released
// `deepseek-ai/DeepSeek-V4.1-Flash` `config.json` at revision `dba1be0a`; the
// row spec's `## Upstream chain` carries the full config delta.
struct EngramGeometry {
  // Which backbone layers carry an Engram module. EMPTY MEANS DISABLED, and
  // that is the real feature switch: `EngramLayout.from_config:206-210` returns
  // `None` on an empty list. `vllm/config/engram.py` is only `cpu_offload` and
  // `embedding_across_dp` and switches nothing.
  std::vector<int64_t> layer_ids;
  // One table size per entry of `layer_ids`; the two released values are
  // 384,006,168 and 384,016,682.
  std::vector<int64_t> num_embeddings;
  // 4. The deepest n-gram, so the lookback depth is THREE. The "128 sliding
  // window" in the module docstring (engram.py:24-25) is not a parameter: it is
  // the safety argument for why depth 3 can never outrun window eviction.
  int64_t max_ngram_size = 4;
  int64_t n_heads = 8;
  // 256, and it is the largest number in this struct by consequence. It is the
  // width of a row of the two ~384M-row tables
  // (`engram.embed.weight` is `F8_E4M3 [384006168, 256]`) and, through
  // `n_hash_cols() * head_dim`, the `wkv` INPUT width: 24 * 256 = 6144, which is
  // what `engram.wkv.weight F8_E4M3 [25600, 6144]` carries. An earlier revision
  // of this struct defaulted it to 128 and halved both.
  int64_t head_dim = 256;
  // 16,000,000. Seeds the per-bucket prime search ONLY (engram.py:193).
  int64_t vocab_size = 16000000;
  // 99,092. Bounds the hash multiplier ONLY (engram.py:155).
  int64_t compressed_vocab_size = 99092;
  // 2, and NOT 0. `engram.py:422` is `self.pad_id = token_map[pad_token_id]`,
  // so this is a TOKEN id that still has to be compressed, never a compressed
  // id. An earlier revision defaulted it to 0, which is the value at which
  // upstream's own fixtures send the pad to itself and the distinction
  // disappears. `text_config.engram_pad_token_id` is 2, and the artifact's
  // top-level `pad_token_id` agrees.
  int64_t pad_token_id = 2;

  // From the model config, read by `Engram.__init__` (engram.py:900-903).
  int64_t hidden_size = 5120;
  int64_t hc_mult = 4;
  // 1e-20 for V4.1 (V4 used 1e-6). Small enough that the gate's rsqrt is
  // effectively unregularized, which is why the port keeps it f32 and explicit.
  float rms_norm_eps = 1e-20F;

  // (max_ngram_size - 1) * n_heads; 24 for the released config.
  int64_t n_hash_cols() const { return (max_ngram_size - 1) * n_heads; }
  int64_t lookback_depth() const { return max_ngram_size - 1; }
};

// ─── (1) the prime bucket layout ─────────────────────────────────────────────

// `_is_prime`, engram.py:63-85: deterministic Miller-Rabin with the bases
// (2, 7, 61), which is exact for every n < 2^32. The small-factor pre-screen
// returns `n == p` so the primes 2..37 are not rejected by their own screen.
bool IsEngramPrime(int64_t n);

// `find_next_prime`, engram.py:88-93: the smallest prime STRICTLY above `start`
// that `seen` does not already hold. `seen` is threaded across the whole layout
// and is what keeps every bucket range disjoint.
int64_t FindNextEngramPrime(int64_t start, std::vector<int64_t>* seen);

// `EngramLayout`, engram.py:169-210.
//
// A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max), each
// split over `n_heads` heads. Every (n-gram size, head) pair owns its own
// prime-sized bucket range in the layer's table. `primes` and `offsets` are
// both `[n_layers, n_hash_cols]` row-major, with the n-gram index major and the
// head index minor, exactly as `_hash_ids_kernel` indexes them
// (`col = (shift - 1) * num_heads + head`, engram.py:372).
//
// `offsets` is the EXCLUSIVE prefix sum of `primes` WITHIN a layer
// (engram.py:203), so layer `l`'s buckets tile `[0, sum(primes[l]))`. The
// prime search restarts at `vocab_size - 1` for each n-gram group but `seen` is
// shared globally (engram.py:188-199), so no two columns anywhere share a
// prime.
struct EngramLayout {
  std::vector<int64_t> layer_ids;
  std::vector<int64_t> num_embeddings;
  int64_t max_ngram_size = 0;
  int64_t n_heads = 0;
  int64_t head_dim = 0;
  int64_t compressed_vocab_size = 0;
  int64_t pad_token_id = 0;
  int64_t n_hash_cols = 0;
  std::vector<int64_t> primes;   // [n_layers, n_hash_cols]
  std::vector<int64_t> offsets;  // [n_layers, n_hash_cols]

  int64_t n_layers() const { return static_cast<int64_t>(layer_ids.size()); }
  // The head bucket sizes of one layer, in head order — what
  // `MakeEngramEmbeddingShard` needs. engram.py:913.
  std::vector<int64_t> HeadSizes(int64_t layer_hash_index) const;
};

// `EngramLayout.from_config`, engram.py:206-210. Returns no value when
// `layer_ids` is empty, which is the feature switch. Refuses by name when
// `layer_ids` and `num_embeddings` differ in length (engram.py:186).
std::optional<EngramLayout> EngramLayoutFromConfig(const EngramGeometry& geometry);

// ─── (2) the two constants this reference CONSUMES rather than derives ───────
//
// DECISION 1 (the compressed-vocab map). `build_compressed_token_map`
// (engram.py:96-144) needs the HuggingFace `tokenizers` normalizer chain
// NFKC -> NFD -> StripAccents -> Lowercase -> whitespace collapse -> the
// `` (U+E000) sentinel dance -> Strip, and this tree's tokenizer implements NONE
// of those. The published GGUF artifact carries the finished map as metadata
// (`deepseek41.engram.token_map`, 129,280 entries, beside `multipliers`,
// `primes`, `offsets` and `pad_id`), so a loader READS it. This reference
// therefore takes the map as data and VALIDATES it. Construction from a
// tokenizer is owed, with its own issue, rather than half-implemented: a
// partly-correct Unicode chain produces a plausible map that silently rehashes
// every table, and upstream's `build_compressed_token_map` is itself UNGATED,
// so there is no ported test that would catch it.
//
// DECISION 2 (the multipliers). `compute_hash_multipliers` (engram.py:147-166)
// draws them from NumPy PCG64 seeded `10007 * layer_id`. The artifact carries
// those too (`multipliers`, 8 x u64), so the same rule applies: read and
// validate, do not reimplement a specific PRNG's bounded-integer path. What IS
// ported is the BOUND, because the bound is the load-bearing part: it is the
// only thing that keeps `value * multiplier` inside int64, and an int64
// overflow makes `rolling` negative, at which point C++ truncating `%` and
// Python flooring `%` disagree and the port diverges from its own oracle.

// `multiplier_bound`, engram.py:155: `max(1, (INT64_MAX // cv) // 2)`. Every
// multiplier is `2 * u + 1` for `u` in `[0, bound)`, so every multiplier is odd
// and `< 2 * bound + 1`.
int64_t HashMultiplierBound(int64_t compressed_vocab_size);

// Refuse a multiplier table that cannot be the one upstream would have drawn:
// wrong shape, even, non-positive, or large enough that
// `(compressed_vocab_size - 1) * multiplier` overflows int64.
// `multipliers` is `[n_layers, max_ngram_size]` row-major.
void ValidateHashMultipliers(const std::vector<int64_t>& multipliers,
                             int64_t n_layers, int64_t max_ngram_size,
                             int64_t compressed_vocab_size);

// The contract `build_compressed_token_map` produces and
// `NgramHashState.__init__:414-421` then asserts: ids are assigned in
// first-appearance order, so the set of emitted ids is exactly
// `[0, compressed_vocab_size)` with no gaps. Refuses by name with upstream's
// own message shape when the built size disagrees with the config, "because
// every hash multiplier derives from it, so the engram tables would be silently
// rehashed" (engram.py:416-421). Returns the compressed vocabulary size.
int64_t ValidateCompressedTokenMap(const std::vector<int32_t>& token_map,
                                   int64_t compressed_vocab_size);

// ─── (3) the n-gram hash ─────────────────────────────────────────────────────

// One forward's worth of batch description. Names and shapes mirror
// `NgramHashState.forward` (engram.py:465-475) and the runner call site
// (`nvidia/model.py:587-595`).
//
// History resolves in THREE tiers, newest source first
// (`_hash_ids_kernel:312-366`):
//   1. in batch   — the lookback position is at or after this request's chunk
//                   start, so the id is in `input_ids` itself;
//   2. the runner's lookback window — `lookback_token_ids[req][col]`, where
//                   `col = chunk_start - 1 - lookback`, newest at col 0. A
//                   negative entry means "unknown", not "token -1";
//   3. the V1 slot cache — addressed through the block table, so a position
//                   resolves to the physical slot its KV occupies.
// With no slot cache (the V2 runner) tier 3 is `pad_id` instead.
struct NgramHashInputs {
  const int32_t* input_ids = nullptr;   // [num_tokens]
  const int64_t* positions = nullptr;   // [num_tokens]
  const int32_t* query_start_loc = nullptr;  // [num_reqs + 1]
  // True means DEAD: the token breaks n-grams. `nvidia/model.py:578-580` warns
  // that this polarity is the OPPOSITE of `EngramPostWkv`'s `token_mask`,
  // where true means KEEP.
  const uint8_t* dead_mask = nullptr;   // [num_tokens]
  const int32_t* lookback_token_ids = nullptr;   // [num_reqs, lookback_depth]
  const uint8_t* lookback_dead_mask = nullptr;   // [num_reqs, lookback_depth]
  // Both null on the V2 runner; both required when the slot cache is in use.
  const int64_t* slot_mapping = nullptr;  // [num_tokens], -1 = do not write
  const int32_t* block_table = nullptr;   // [num_block_table_rows, max_blocks]
  int64_t num_tokens = 0;
  int64_t num_reqs = 0;                 // = query_start_loc length - 1
  int64_t num_block_table_rows = 0;
  int64_t max_blocks = 0;
  int64_t lookback_depth = 0;
};

// `NgramHashState`, engram.py:383-546. Holds the compressed token map, the
// multipliers, the bucket layout and — on the V1 runner only — the slot-keyed
// rolling store of compressed ids.
class NgramHashState {
 public:
  // `multipliers` is `[n_layers, max_ngram_size]` row-major and is VALIDATED
  // here. `token_map` is checked for the one property the HASH depends on —
  // every compressed id inside `[0, compressed_vocab_size)`, which is what
  // keeps `value * multiplier` inside int64 — and NOT for the stronger
  // checkpoint property `ValidateCompressedTokenMap` enforces. The two are
  // separate on purpose: upstream asserts the stronger one in
  // `__init__:415-421` only because it BUILDS the map there, and this reference
  // reads a finished map instead (DECISION 1). A caller that loads the map from
  // the artifact calls the validator itself.
  // `pad_id` is `token_map[layout.pad_token_id]` (engram.py:422), derived here
  // so a caller cannot pass the raw pad token by mistake.
  NgramHashState(EngramLayout layout, std::vector<int32_t> token_map,
                 std::vector<int64_t> multipliers, int64_t block_size,
                 bool use_slot_cache);

  // `ensure_cache`, engram.py:439-463. `num_kv_blocks` is the bound KV cache's
  // block count; 0 means unbound (a profile run) and the caller must then skip
  // hashing entirely. Rebinding to a DIFFERENT block count discards the
  // previous history, because graph memory profiling binds a temporary smaller
  // cache first and its hash history must not survive.
  bool EnsureCache(int64_t num_kv_blocks);

  // Bind an already-allocated slot store rather than allocate a fresh one. The
  // slot cache outlives one forward: a new state in the same process must pick
  // up the history its predecessor wrote, not zero it. Upstream expresses this
  // by assigning `state._cache` directly (test_engram.py:281, :389).
  void BindCache(std::vector<int32_t> cache);

  // `forward`, engram.py:465-546. Writes `[num_tokens, n_layers, n_hash_cols]`
  // int32, row-major. Writes the slot cache first when it is in use, because
  // `_hash_ids_kernel` reads back what `_write_hash_cache_kernel` wrote for
  // earlier tokens of the same batch.
  void Forward(const NgramHashInputs& inputs, int32_t* out);

  const EngramLayout& layout() const { return layout_; }
  int64_t pad_id() const { return pad_id_; }
  // The slot cache, for gates that assert history across steps. Empty when the
  // cache is unbound or unused.
  const std::vector<int32_t>& cache() const { return cache_; }

 private:
  void WriteCache(const NgramHashInputs& inputs);

  EngramLayout layout_;
  std::vector<int32_t> token_map_;
  std::vector<int64_t> multipliers_;
  std::vector<int32_t> cache_;
  int64_t block_size_ = 0;
  int64_t bound_kv_blocks_ = 0;
  int64_t pad_id_ = 0;
  bool use_slot_cache_ = false;
};

// ─── (4) the sharded fp8 table lookup ────────────────────────────────────────

// `ParallelEngramEmbedding`'s derived geometry, engram.py:639-656. The table is
// sharded by COMPLETE hash heads, so a rank owns whole bucket ranges and never
// splits one. `part_n_hash_cols` is `cdiv(n_hash_cols, tp_size)`, so when
// `tp_size` does not divide `n_hash_cols` the last ranks own PADDED heads:
// `head_start` can reach or pass `n_hash_cols` and the rank then owns ZERO
// rows. Those padded columns write zeros so the all-gather stays rectangular
// (engram.py:600 bounds `owned` by `head < TOTAL_HEADS` and :606 is the
// `other=0.0` that an unowned row therefore loads; :619 is a bare `)` and an
// earlier revision of this comment cited it by mistake. The
// `count_nonzero == 0` assertion is at test_engram.py:610.)
struct EngramEmbeddingShard {
  int64_t dim = 0;
  int64_t quant_block = kEngramQuantBlock;
  int64_t n_hash_cols = 0;
  int64_t part_n_hash_cols = 0;
  int64_t head_start = 0;
  int64_t tp_size = 1;
  int64_t vocab_start_idx = 0;
  int64_t vocab_end_idx = 0;
  int64_t part_num_embeddings = 0;
};

// engram.py:630-656. `head_sizes` is one bucket size per hash column, in head
// order — `EngramLayout::HeadSizes`. Refuses by name on an empty or
// non-positive head size, and when the head sizes do not fit `num_embeddings`
// (engram.py:641-642).
EngramEmbeddingShard MakeEngramEmbeddingShard(int64_t num_embeddings, int64_t dim,
                                              const std::vector<int64_t>& head_sizes,
                                              int64_t tp_size, int64_t tp_rank);

// `_engram_head_shard_weight_loader`, engram.py:549-561: this rank's complete
// head buckets, narrowed out of the full checkpoint row range at
// `vocab_start_idx`. `row_bytes` is `dim` for the fp8 weight and
// `dim / quant_block` for the ue8m0 scales; both are raw bytes either way, the
// scales because `float8_e8m0fnu` is stored as `uint8`. Refuses by name when
// the shard does not fit the loaded tensor.
std::vector<uint8_t> LoadEngramHeadShard(const EngramEmbeddingShard& shard,
                                         const uint8_t* loaded_weight,
                                         int64_t loaded_rows, int64_t row_bytes);

// `_engram_lookup_kernel`, engram.py:564-619, driven by `lookup` (:716-748).
//
//   weight_fp8   [part_num_embeddings, dim]                fp8-e4m3 bytes
//   scale_ue8m0  [part_num_embeddings, dim / quant_block]  ue8m0 bytes
//   ids          [num_tokens, n_hash_cols]                 int32 hash ids
//   out          [num_tokens, part_n_hash_cols, dim]       bf16 bit patterns
//
// A row this rank does not own — a padded head, or an id outside
// `[vocab_start_idx, vocab_end_idx)` — writes ZEROS, which is what makes the
// concatenation of every rank's output equal the unsharded lookup.
//
// The scale decode is `vllm::E8M0BitsToF32`, the BITCAST form, NOT
// `vllm::E8M0ToF32`. See the header comment: byte 0 is +0.0 here and 2^-127
// there, and nothing upstream pins the two against each other.
void EngramLookup(const EngramEmbeddingShard& shard, const uint8_t* weight_fp8,
                  const uint8_t* scale_ue8m0, const int32_t* ids,
                  int64_t num_tokens, uint16_t* out);

// ─── (5) the gate that injects the lookup into the manifold ─────────────────

// `_fused_engram_post_wkv_kernel`, engram.py:765-854.
//
//   hidden_states [num_tokens, hc_mult, dim]        bf16 bit patterns
//   kv            [num_kv_tokens, (hc_mult+1)*dim]  bf16, the `wkv` output:
//                 one KEY per hc copy, then ONE shared VALUE
//   q_weight      [hc_mult, dim]                    bf16
//   k_weight      [hc_mult, dim]                    bf16
//   token_mask    [num_kv_tokens] or null. TRUE MEANS KEEP — the opposite
//                 polarity to `NgramHashInputs::dead_mask`; false shuts the
//                 gate so the position passes through untouched.
//   out           [num_tokens, hc_mult, dim]        bf16
//
// Per (token, hc copy), all arithmetic in f32 (`.to(tl.float32)` on every load,
// engram.py:810-825), mirroring upstream's dtype polarity exactly — the stream
// is bf16 and only the reduction is widened:
//
//   dot  = sum(hidden * q * k * key) * rsqrt(mean(hidden^2) + eps)
//                                   * rsqrt(mean(key^2) + eps) * rsqrt(dim)
//   gate = sigmoid(copysign(sqrt(max(|dot|, clamp)), dot))
//   out  = hidden + gate * value
//
// THE CLAMP IS INSIDE THE SQRT and it is `max`, not `min`, so the floor on the
// gate input MAGNITUDE is sqrt(1e-6) = 1e-3 and a tiny dot is AMPLIFIED, not
// squashed. `copysign` keeps the sign of the ORIGINAL dot, so the function is
// odd through the origin rather than continuous at it.
//
// `num_kv_tokens` is the SP-local count: a token at or past it reads key and
// value as zero (`source_valid`, engram.py:799), which is how the padded tail
// of a sequence-parallel shard stays finite.
struct EngramGateParams {
  int64_t hc_mult = 0;
  int64_t dim = 0;
  float eps = 1e-20F;
  // engram.py:903 `self.clamp_value = 1e-6`.
  float clamp_value = 1e-6F;
};
void EngramPostWkv(const EngramGateParams& params, const uint16_t* hidden_states,
                   const uint16_t* kv, const uint16_t* q_weight,
                   const uint16_t* k_weight, const uint8_t* token_mask,
                   int64_t num_tokens, int64_t num_kv_tokens, uint16_t* out);

}  // namespace vllm::deepseek_v4_1
