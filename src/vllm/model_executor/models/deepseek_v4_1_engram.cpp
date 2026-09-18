// DeepSeek-V4.1-Flash W3a — Engram host reference. Every function here is a 1:1
// port of the upstream symbol named beside it; the design, the scope and the
// two decisions this wave took are in
// `include/vllm/model_executor/models/deepseek_v4_1_engram.h`.
//
// Ported from vllm/models/deepseek_v4_1/common/engram.py @ vLLM `e77daef89e`
// (the V4.1 completion commit, 566 commits past our parity pin `e126687a9a`, at
// which `deepseek_v4_1/` does not exist).
#include "vllm/model_executor/models/deepseek_v4_1_engram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"  // E8M0BitsToF32
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32
#include "vt/dtype.h"                                        // VT_CHECK, BF16

namespace vllm::deepseek_v4_1 {

namespace {

// `pow(a, d, n)` for the Miller-Rabin witness loop. `n` is below 2^32 on every
// path `_is_prime` is reached with (the primes sit just above
// `engram_vocab_size` = 16,000,000), so the __int128-free product below cannot
// overflow uint64.
uint64_t PowMod(uint64_t base, uint64_t exponent, uint64_t modulus) {
  uint64_t result = 1;
  base %= modulus;
  while (exponent > 0) {
    if ((exponent & 1U) != 0) result = result * base % modulus;
    base = base * base % modulus;
    exponent >>= 1;
  }
  return result;
}

}  // namespace

// ─── (1) the prime bucket layout ─────────────────────────────────────────────

// engram.py:63-85. Deterministic Miller-Rabin for n < 2^32.
//
// MIRRORED DEFECT, and it is mirrored on purpose. A witness `a` that is a
// MULTIPLE of `n` makes `pow(a, d, n)` zero, which is neither 1 nor n-1 and
// never becomes n-1 under squaring, so the candidate is reported composite.
// With the witness set (2, 7, 61) that happens at exactly one value below 2^32:
// **61 itself**, which the small-factor screen does not catch because 61 is not
// in it. Verified against the pinned oracle by executing engram.py:63-85 at
// `e77daef89e`: `_is_prime(61)` is `False`, and 61 is the ONLY n below 2000
// where it disagrees with trial division.
//
// It is unreachable at the only call site: `find_next_prime` is entered with
// `start = engram_vocab_size - 1` = 15,999,999, so 61 is never a candidate. We
// mirror it rather than "fix" it because a port that silently disagrees with
// its oracle on any input is a port whose agreement elsewhere proves nothing —
// and if a future config lowers `engram_vocab_size` below 62, the two
// implementations must still pick the same primes. The gate pins 61 explicitly.
bool IsEngramPrime(int64_t n) {
  if (n < 2) return false;
  // The small-factor screen returns `n == p`, so 2..37 pass through it as
  // primes instead of being rejected by their own divisor (engram.py:67-69).
  for (int64_t p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
    if (n % p == 0) return n == p;
  }
  VT_CHECK(n < (int64_t{1} << 32),
           "engram prime test: the (2, 7, 61) witness set is exact only below "
           "2^32; a larger candidate needs a wider witness set");
  const uint64_t value = static_cast<uint64_t>(n);
  uint64_t d = value - 1;
  int r = 0;
  while (d % 2 == 0) {
    d /= 2;
    ++r;
  }
  for (uint64_t a : {uint64_t{2}, uint64_t{7}, uint64_t{61}}) {
    uint64_t x = PowMod(a, d, value);
    if (x == 1 || x == value - 1) continue;
    bool witnessed = true;
    for (int i = 0; i < r - 1; ++i) {
      x = x * x % value;
      if (x == value - 1) {
        witnessed = false;
        break;
      }
    }
    if (witnessed) return false;
  }
  return true;
}

// engram.py:88-93. `seen` is a growing list rather than a set because it holds
// at most `n_layers * n_hash_cols` entries (48 for the released config) and the
// linear scan is cheaper than a hash at that size.
int64_t FindNextEngramPrime(int64_t start, std::vector<int64_t>* seen) {
  VT_CHECK(seen != nullptr, "engram prime search: `seen` is null");
  int64_t candidate = start + 1;
  for (;;) {
    if (IsEngramPrime(candidate) &&
        std::find(seen->begin(), seen->end(), candidate) == seen->end()) {
      return candidate;
    }
    ++candidate;
  }
}

std::vector<int64_t> EngramLayout::HeadSizes(int64_t layer_hash_index) const {
  VT_CHECK(layer_hash_index >= 0 && layer_hash_index < n_layers(),
           "engram layout: layer hash index out of range");
  const std::vector<int64_t>::const_iterator begin =
      primes.begin() + static_cast<size_t>(layer_hash_index * n_hash_cols);
  return std::vector<int64_t>(begin, begin + static_cast<size_t>(n_hash_cols));
}

// engram.py:178-210.
std::optional<EngramLayout> EngramLayoutFromConfig(const EngramGeometry& geometry) {
  // `from_config:206-210` — an empty `engram_layer_ids` is the feature switch,
  // and it is the ONLY switch. `vllm/config/engram.py` carries `cpu_offload`
  // and `embedding_across_dp` and neither enables nor disables engram.
  if (geometry.layer_ids.empty()) return std::nullopt;

  // engram.py:186.
  VT_CHECK(geometry.layer_ids.size() == geometry.num_embeddings.size(),
           "engram layout: engram_layer_ids and engram_num_embeddings differ in "
           "length");
  VT_CHECK(geometry.max_ngram_size >= 2,
           "engram layout: engram_max_ngram_size must be at least 2, because a "
           "position is hashed as max_ngram_size - 1 n-grams");
  VT_CHECK(geometry.n_heads >= 1, "engram layout: engram_n_heads must be positive");
  VT_CHECK(geometry.vocab_size >= 2,
           "engram layout: engram_vocab_size seeds the bucket prime search and "
           "must be at least 2");

  EngramLayout layout;
  layout.layer_ids = geometry.layer_ids;
  layout.num_embeddings = geometry.num_embeddings;
  layout.max_ngram_size = geometry.max_ngram_size;
  layout.n_heads = geometry.n_heads;
  layout.head_dim = geometry.head_dim;
  layout.compressed_vocab_size = geometry.compressed_vocab_size;
  layout.pad_token_id = geometry.pad_token_id;
  layout.n_hash_cols = geometry.n_hash_cols();

  // engram.py:188-200. `seen` is threaded across EVERY layer and every n-gram
  // group, so no two of the 48 columns share a prime and the bucket ranges stay
  // disjoint. `current` restarts at `vocab_size - 1` per n-gram group (:193),
  // and that restart is INERT: `FindNextEngramPrime` searches UPWARD from
  // `start + 1` and skips every value already in `seen` (:96-106 above), so a
  // restarted search walks back past all of them and still returns the next
  // prime above the whole `seen` set. Hoisting `current` out of the group loop
  // would therefore produce byte-identical primes, and all 48 columns come out
  // globally increasing — not merely increasing within a group. Keep the
  // restart because it is what upstream writes, not because it changes a value.
  std::vector<int64_t> seen;
  layout.primes.reserve(
      static_cast<size_t>(layout.n_layers() * layout.n_hash_cols));
  for (size_t layer = 0; layer < geometry.layer_ids.size(); ++layer) {
    for (int64_t group = 0; group < geometry.max_ngram_size - 1; ++group) {
      int64_t current = geometry.vocab_size - 1;
      for (int64_t head = 0; head < geometry.n_heads; ++head) {
        current = FindNextEngramPrime(current, &seen);
        seen.push_back(current);
        layout.primes.push_back(current);
      }
    }
  }
  // engram.py:203 — the exclusive prefix sum WITHIN a layer.
  layout.offsets.resize(layout.primes.size());
  for (int64_t layer = 0; layer < layout.n_layers(); ++layer) {
    int64_t running = 0;
    for (int64_t col = 0; col < layout.n_hash_cols; ++col) {
      const size_t idx = static_cast<size_t>(layer * layout.n_hash_cols + col);
      layout.offsets[idx] = running;
      running += layout.primes[idx];
    }
  }
  return layout;
}

// ─── (2) the two constants this reference consumes ───────────────────────────

// engram.py:155.
int64_t HashMultiplierBound(int64_t compressed_vocab_size) {
  VT_CHECK(compressed_vocab_size > 0,
           "engram multipliers: engram_compressed_vocab_size must be positive");
  return std::max<int64_t>(
      1, (std::numeric_limits<int64_t>::max() / compressed_vocab_size) / 2);
}

void ValidateHashMultipliers(const std::vector<int64_t>& multipliers,
                             int64_t n_layers, int64_t max_ngram_size,
                             int64_t compressed_vocab_size) {
  VT_CHECK(n_layers > 0 && max_ngram_size > 0,
           "engram multipliers: the table shape must be positive");
  VT_CHECK(multipliers.size() ==
               static_cast<size_t>(n_layers * max_ngram_size),
           "engram multipliers: expected one multiplier per (layer, lookback); "
           "the table does not have n_layers * engram_max_ngram_size entries");
  const int64_t bound = HashMultiplierBound(compressed_vocab_size);
  // The largest compressed id a hash can carry. `value * multiplier` must stay
  // inside int64 for every one of them: an overflow makes `rolling` negative,
  // and a negative dividend is where C++ truncating `%` and the Python oracle's
  // flooring `%` stop agreeing, which turns a port bug into a silent
  // divergence rather than a crash.
  const int64_t max_value = compressed_vocab_size - 1;
  for (int64_t multiplier : multipliers) {
    VT_CHECK(multiplier > 0,
             "engram multipliers: a multiplier must be positive");
    // engram.py:165 `values * 2 + 1` — every multiplier is odd by construction.
    VT_CHECK(multiplier % 2 == 1,
             "engram multipliers: a multiplier must be odd; upstream builds "
             "them as 2 * u + 1 (engram.py:165)");
    VT_CHECK(multiplier < 2 * bound + 1,
             "engram multipliers: a multiplier is above the bound derived from "
             "engram_compressed_vocab_size");
    VT_CHECK(max_value == 0 ||
                 multiplier <= std::numeric_limits<int64_t>::max() / max_value,
             "engram multipliers: (engram_compressed_vocab_size - 1) * "
             "multiplier overflows int64, which would make the rolling hash "
             "negative and silently diverge from the oracle");
  }
}

// engram.py:96-144 (the contract) + :414-421 (the assertion).
int64_t ValidateCompressedTokenMap(const std::vector<int32_t>& token_map,
                                   int64_t compressed_vocab_size) {
  VT_CHECK(!token_map.empty(), "engram token map: the map is empty");
  int64_t built = 0;
  for (int32_t id : token_map) {
    VT_CHECK(id >= 0, "engram token map: a compressed id is negative");
    built = std::max<int64_t>(built, static_cast<int64_t>(id) + 1);
  }
  // `build_compressed_token_map` assigns `len(key_to_new)` in first-appearance
  // order (engram.py:138-142), so the emitted ids are exactly [0, built) with
  // no hole. A hole means the map did not come from that function.
  std::vector<uint8_t> seen(static_cast<size_t>(built), 0);
  for (int32_t id : token_map) seen[static_cast<size_t>(id)] = 1;
  for (size_t id = 0; id < seen.size(); ++id) {
    VT_CHECK(seen[id] != 0,
             "engram token map: the compressed ids are not dense; "
             "build_compressed_token_map assigns them in first-appearance "
             "order, so a gap means this map is not the one the tables were "
             "built for");
  }
  // Upstream's own message, engram.py:416-421.
  VT_CHECK(built == compressed_vocab_size,
           "engram token map: compressed vocab size mismatch against "
           "engram_compressed_vocab_size; every hash multiplier derives from "
           "it, so the engram tables would be silently rehashed");
  return built;
}

// ─── (3) the n-gram hash ─────────────────────────────────────────────────────

NgramHashState::NgramHashState(EngramLayout layout, std::vector<int32_t> token_map,
                               std::vector<int64_t> multipliers, int64_t block_size,
                               bool use_slot_cache)
    : layout_(std::move(layout)),
      token_map_(std::move(token_map)),
      multipliers_(std::move(multipliers)),
      block_size_(block_size),
      use_slot_cache_(use_slot_cache) {
  VT_CHECK(block_size_ > 0, "engram hash: the KV block size must be positive");
  VT_CHECK(!token_map_.empty(), "engram hash: the token map is empty");
  // The hash needs ONE property of the map and this constructor enforces it:
  // every compressed id is inside `[0, engram_compressed_vocab_size)`, because
  // that is the bound `multiplier_bound` was derived from and therefore the
  // only thing keeping `value * multiplier` inside int64.
  //
  // It deliberately does NOT call `ValidateCompressedTokenMap`. That checks a
  // STRONGER property — that the map is dense and its size equals the config —
  // which belongs to the checkpoint, not to the hash, and which upstream
  // asserts in `NgramHashState.__init__:415-421` only because it BUILDS the map
  // there. A caller that reads a finished map (ours, from the artifact
  // metadata) calls the validator itself; a synthetic fixture, like upstream's
  // own `large_hashes` map, is legitimately sparse and still hashes correctly.
  for (int32_t id : token_map_) {
    VT_CHECK(id >= 0 && id < layout_.compressed_vocab_size,
             "engram hash: a compressed token id is outside "
             "[0, engram_compressed_vocab_size); the multiplier bound is "
             "derived from that size, so an id above it overflows the rolling "
             "hash");
  }
  ValidateHashMultipliers(multipliers_, layout_.n_layers(), layout_.max_ngram_size,
                          layout_.compressed_vocab_size);
  VT_CHECK(layout_.primes.size() ==
               static_cast<size_t>(layout_.n_layers() * layout_.n_hash_cols),
           "engram hash: the prime table does not match the layout shape");
  VT_CHECK(layout_.offsets.size() == layout_.primes.size(),
           "engram hash: the offset table does not match the prime table");
  VT_CHECK(layout_.pad_token_id >= 0 &&
               layout_.pad_token_id < static_cast<int64_t>(token_map_.size()),
           "engram hash: engram_pad_token_id is outside the token map");
  // engram.py:422 — the pad the hash uses is the COMPRESSED id of the pad
  // token, not the pad token itself.
  pad_id_ = token_map_[static_cast<size_t>(layout_.pad_token_id)];
}

// engram.py:439-463.
bool NgramHashState::EnsureCache(int64_t num_kv_blocks) {
  if (num_kv_blocks <= 0) {
    // The KV cache is unbound (a profile run): the caller skips hashing.
    cache_.clear();
    bound_kv_blocks_ = 0;
    return false;
  }
  if (!use_slot_cache_) return true;
  if (bound_kv_blocks_ == num_kv_blocks) return true;
  // "Graph memory profiling binds a temporary, smaller KV cache first.
  // Rebinding must discard its hash history" (engram.py:455-456).
  cache_.assign(static_cast<size_t>(num_kv_blocks * block_size_), 0);
  bound_kv_blocks_ = num_kv_blocks;
  return true;
}

// Bind an ALREADY-ALLOCATED slot store instead of allocating a fresh one. This
// is the seam the runner needs across steps: the slot cache outlives one
// forward, and a fresh `NgramHashState` in the same process must pick up the
// history its predecessor wrote rather than zero it. Upstream expresses the
// same thing by assigning `state._cache` directly (test_engram.py:281, :389).
void NgramHashState::BindCache(std::vector<int32_t> cache) {
  VT_CHECK(use_slot_cache_,
           "engram hash: a slot cache cannot be bound when the state runs "
           "without one (the V2 runner supplies every lookback)");
  VT_CHECK(!cache.empty() &&
               static_cast<int64_t>(cache.size()) % block_size_ == 0,
           "engram hash: a bound slot cache must be a whole number of KV "
           "blocks");
  bound_kv_blocks_ = static_cast<int64_t>(cache.size()) / block_size_;
  cache_ = std::move(cache);
}

// engram.py:213-236.
void NgramHashState::WriteCache(const NgramHashInputs& inputs) {
  for (int64_t token = 0; token < inputs.num_tokens; ++token) {
    const int64_t slot = inputs.slot_mapping[static_cast<size_t>(token)];
    if (slot < 0) continue;
    VT_CHECK(slot < static_cast<int64_t>(cache_.size()),
             "engram hash: a slot mapping entry is outside the bound cache");
    const int32_t id = inputs.input_ids[static_cast<size_t>(token)];
    VT_CHECK(id >= 0 && id < static_cast<int64_t>(token_map_.size()),
             "engram hash: an input token id is outside the token map");
    const bool dead = inputs.dead_mask[static_cast<size_t>(token)] != 0;
    cache_[static_cast<size_t>(slot)] =
        dead ? kEngramDeadId : token_map_[static_cast<size_t>(id)];
  }
}

// engram.py:248-380, driven by :465-546.
void NgramHashState::Forward(const NgramHashInputs& inputs, int32_t* out) {
  const int64_t n_layers = layout_.n_layers();
  const int64_t n_hash_cols = layout_.n_hash_cols;
  const int64_t max_ngram = layout_.max_ngram_size;
  const int64_t n_heads = layout_.n_heads;

  // engram.py:488-489 — an empty batch returns the empty output and, crucially,
  // writes nothing to the cache.
  if (inputs.num_tokens == 0) return;
  VT_CHECK(out != nullptr, "engram hash: the output buffer is null");
  VT_CHECK(inputs.input_ids != nullptr && inputs.positions != nullptr &&
               inputs.query_start_loc != nullptr && inputs.dead_mask != nullptr,
           "engram hash: a required input buffer is null");
  VT_CHECK(inputs.lookback_token_ids != nullptr &&
               inputs.lookback_dead_mask != nullptr,
           "engram hash: the runner's lookback window is required; the V1 "
           "runner passes prompt positions only and fills the rest with -1");
  VT_CHECK(inputs.num_reqs > 0, "engram hash: query_start_loc has no request");

  const bool use_cache = use_slot_cache_ && !cache_.empty();
  if (use_slot_cache_) {
    // engram.py:490-506 — "Finish writes before other thread blocks read
    // fallback history." The host reference gets that ordering for free by
    // writing the whole cache first.
    VT_CHECK(!cache_.empty(),
             "engram hash: the slot cache is in use but unbound; call "
             "EnsureCache and skip hashing when it returns false");
    VT_CHECK(inputs.slot_mapping != nullptr && inputs.block_table != nullptr,
             "engram hash: the slot cache needs both slot_mapping and "
             "block_table (engram.py:491-492)");
    WriteCache(inputs);
  }

  for (int64_t token = 0; token < inputs.num_tokens; ++token) {
    // engram.py:290-304 — the upper bound in `query_start_loc[1:]`, including
    // repeated padding boundaries, clamped to the last request.
    int64_t req = 0;
    while (req < inputs.num_reqs &&
           inputs.query_start_loc[static_cast<size_t>(req + 1)] <= token) {
      ++req;
    }
    req = std::min(req, inputs.num_reqs - 1);

    // engram.py:305-307 — the chunk's FIRST position, which is what separates
    // "in this batch" from "history".
    int64_t chunk_idx = inputs.query_start_loc[static_cast<size_t>(req)];
    chunk_idx = std::min(chunk_idx, inputs.num_tokens - 1);
    const int64_t chunk_start = inputs.positions[static_cast<size_t>(chunk_idx)];
    const int64_t position = inputs.positions[static_cast<size_t>(token)];

    bool blocked = false;
    std::vector<int64_t> rolling(static_cast<size_t>(n_layers), 0);
    for (int64_t shift = 0; shift < max_ngram; ++shift) {
      const int64_t lookback = position - shift;
      int64_t source = pad_id_;

      // Tier 1: in this batch (engram.py:313-323).
      if (lookback >= chunk_start) {
        const int64_t batch_idx = std::max<int64_t>(token - shift, 0);
        const int32_t id = inputs.input_ids[static_cast<size_t>(batch_idx)];
        VT_CHECK(id >= 0 && id < static_cast<int64_t>(token_map_.size()),
                 "engram hash: an input token id is outside the token map");
        source = inputs.dead_mask[static_cast<size_t>(batch_idx)] != 0
                     ? kEngramDeadId
                     : token_map_[static_cast<size_t>(id)];
      } else {
        // Tier 2: the runner's lookback window (engram.py:325-342). `col` is
        // measured back from the chunk start, newest at 0. A NEGATIVE entry
        // means "unknown", not "token -1".
        const int64_t col = chunk_start - 1 - lookback;
        const bool in_window = col >= 0 && col < inputs.lookback_depth;
        const int32_t window_token =
            in_window ? inputs.lookback_token_ids[static_cast<size_t>(
                            req * inputs.lookback_depth + col)]
                      : -1;
        if (in_window && window_token >= 0) {
          VT_CHECK(window_token < static_cast<int64_t>(token_map_.size()),
                   "engram hash: a lookback token id is outside the token map");
          source = inputs.lookback_dead_mask[static_cast<size_t>(
                       req * inputs.lookback_depth + col)] != 0
                       ? kEngramDeadId
                       : token_map_[static_cast<size_t>(window_token)];
        } else if (use_cache) {
          // Tier 3: the V1 slot cache, addressed through the block table
          // (engram.py:344-361). Every index is clamped exactly as the kernel
          // clamps it, because a padded replay reaches here with a lookback
          // that belongs to no live request.
          const int64_t clamped = std::min<int64_t>(
              std::max<int64_t>(lookback, 0),
              inputs.max_blocks * block_size_ - 1);
          const int64_t block_row =
              std::min(req, inputs.num_block_table_rows - 1);
          const int64_t block =
              inputs.block_table[static_cast<size_t>(
                  block_row * inputs.max_blocks + clamped / block_size_)];
          const int64_t slot = std::min<int64_t>(
              std::max<int64_t>(block * block_size_ + clamped % block_size_, 0),
              static_cast<int64_t>(cache_.size()) - 1);
          source = cache_[static_cast<size_t>(slot)];
        } else {
          // engram.py:362-363 — with no slot cache the fallback is the pad id.
          source = pad_id_;
        }
      }

      // engram.py:367-368. `blocked` is STICKY: once an n-gram is broken by a
      // sequence start or a dead token, every DEEPER lookback is padded too, so
      // the shorter n-grams that still fit stay meaningful while the longer
      // ones collapse onto the pad. Clearing it per shift silently makes a
      // 4-gram that straddles an image span hash as if the span were absent.
      blocked = blocked || lookback < 0 || source == kEngramDeadId;
      const int64_t value = blocked ? pad_id_ : source;

      for (int64_t layer = 0; layer < n_layers; ++layer) {
        rolling[static_cast<size_t>(layer)] ^=
            value * multipliers_[static_cast<size_t>(layer * max_ngram + shift)];
        if (shift == 0) continue;
        for (int64_t head = 0; head < n_heads; ++head) {
          const int64_t col = (shift - 1) * n_heads + head;
          const size_t param = static_cast<size_t>(layer * n_hash_cols + col);
          const int64_t hashed =
              rolling[static_cast<size_t>(layer)] % layout_.primes[param] +
              layout_.offsets[param];
          out[static_cast<size_t>((token * n_layers + layer) * n_hash_cols +
                                  col)] = static_cast<int32_t>(hashed);
        }
      }
    }
  }
}

// ─── (4) the sharded fp8 / ue8m0 lookup ──────────────────────────────────────

// engram.py:630-656.
EngramEmbeddingShard MakeEngramEmbeddingShard(int64_t num_embeddings, int64_t dim,
                                              const std::vector<int64_t>& head_sizes,
                                              int64_t tp_size, int64_t tp_rank) {
  // engram.py:641-642.
  VT_CHECK(!head_sizes.empty(),
           "engram embedding: the head bucket list is empty");
  for (int64_t size : head_sizes) {
    VT_CHECK(size > 0, "engram embedding: a head bucket size is not positive");
  }
  VT_CHECK(tp_size >= 1 && tp_rank >= 0 && tp_rank < tp_size,
           "engram embedding: the tensor-parallel rank is outside the group");
  VT_CHECK(dim > 0 && dim % kEngramQuantBlock == 0,
           "engram embedding: the head dimension must be a positive multiple of "
           "the ue8m0 quantization block (32)");
  int64_t total = 0;
  for (int64_t size : head_sizes) total += size;
  VT_CHECK(total <= num_embeddings,
           "engram embedding: the head buckets do not fit the table");

  const int64_t n_hash_cols = static_cast<int64_t>(head_sizes.size());
  EngramEmbeddingShard shard;
  shard.dim = dim;
  shard.quant_block = kEngramQuantBlock;
  shard.n_hash_cols = n_hash_cols;
  // `triton.cdiv(n_hash_cols, tp_size)`: when tp_size does not divide the head
  // count the LAST ranks own padded heads, and a rank can own none at all.
  shard.part_n_hash_cols = (n_hash_cols + tp_size - 1) / tp_size;
  shard.head_start = tp_rank * shard.part_n_hash_cols;
  shard.tp_size = tp_size;
  // Python's `head_sizes[:k]` clamps `k`; C++ does not, so clamp explicitly.
  // This is exactly where a rank past the last real head gets an EMPTY range
  // rather than an out-of-bounds read.
  auto prefix_sum = [&](int64_t count) {
    int64_t sum = 0;
    for (int64_t i = 0; i < std::min(count, n_hash_cols); ++i) {
      sum += head_sizes[static_cast<size_t>(i)];
    }
    return sum;
  };
  shard.vocab_start_idx = prefix_sum(shard.head_start);
  shard.vocab_end_idx = prefix_sum(shard.head_start + shard.part_n_hash_cols);
  shard.part_num_embeddings = shard.vocab_end_idx - shard.vocab_start_idx;
  return shard;
}

// engram.py:549-561.
std::vector<uint8_t> LoadEngramHeadShard(const EngramEmbeddingShard& shard,
                                         const uint8_t* loaded_weight,
                                         int64_t loaded_rows, int64_t row_bytes) {
  VT_CHECK(loaded_weight != nullptr || shard.part_num_embeddings == 0,
           "engram embedding: the loaded weight is null");
  VT_CHECK(row_bytes > 0, "engram embedding: a row cannot be zero bytes");
  VT_CHECK(shard.vocab_start_idx + shard.part_num_embeddings <= loaded_rows,
           "engram embedding: the shard does not fit the loaded tensor");
  std::vector<uint8_t> shard_bytes(
      static_cast<size_t>(shard.part_num_embeddings * row_bytes));
  if (!shard_bytes.empty()) {
    std::memcpy(shard_bytes.data(),
                loaded_weight + static_cast<size_t>(shard.vocab_start_idx * row_bytes),
                shard_bytes.size());
  }
  return shard_bytes;
}

// engram.py:564-619, driven by :716-748.
void EngramLookup(const EngramEmbeddingShard& shard, const uint8_t* weight_fp8,
                  const uint8_t* scale_ue8m0, const int32_t* ids,
                  int64_t num_tokens, uint16_t* out) {
  const int64_t rows = num_tokens * shard.part_n_hash_cols;
  if (rows == 0) return;  // engram.py:724-725.
  VT_CHECK(out != nullptr && ids != nullptr,
           "engram lookup: an input or output buffer is null");
  const int64_t dim = shard.dim;
  const int64_t block = shard.quant_block;
  VT_CHECK(block > 0 && dim % block == 0,
           "engram lookup: the head dimension is not a multiple of the ue8m0 "
           "quantization block");
  const int64_t scale_cols = dim / block;

  for (int64_t row = 0; row < rows; ++row) {
    // engram.py:593-594.
    const int64_t head = shard.head_start + row % shard.part_n_hash_cols;
    const int64_t token = row / shard.part_n_hash_cols;
    // A PADDED head (one this rank owns only because `cdiv` rounded up) and an
    // id another rank owns both write ZEROS, which is what makes the
    // concatenation of every rank's output equal the unsharded lookup.
    bool owned = head < shard.n_hash_cols;
    int64_t index = -1;
    if (owned) {
      index = ids[static_cast<size_t>(token * shard.n_hash_cols + head)];
      owned = index >= shard.vocab_start_idx && index < shard.vocab_end_idx;
    }
    if (!owned) {
      std::fill_n(out + static_cast<size_t>(row * dim), static_cast<size_t>(dim),
                  static_cast<uint16_t>(0));
      continue;
    }
    VT_CHECK(weight_fp8 != nullptr && scale_ue8m0 != nullptr,
             "engram lookup: an owned row was requested but the shard storage "
             "is null");
    const int64_t local = index - shard.vocab_start_idx;
    for (int64_t d = 0; d < dim; ++d) {
      const float value =
          vllm::F8E4M3ToF32(weight_fp8[static_cast<size_t>(local * dim + d)]);
      // THE BITCAST FORM (engram.py:613-614), not `E8M0ToF32`. Byte 0 is +0.0
      // here and 2^-127 there; the header says why the two coexist.
      const float scale = vllm::E8M0BitsToF32(
          scale_ue8m0[static_cast<size_t>(local * scale_cols + d / block)]);
      // f32 reduction, bf16 store — upstream's dtype polarity exactly
      // (engram.py:617 `(values.to(tl.float32) * scale).to(tl.bfloat16)`).
      out[static_cast<size_t>(row * dim + d)] = vt::F32ToBF16(value * scale);
    }
  }
}

// ─── (5) the injection gate ──────────────────────────────────────────────────

// engram.py:765-854.
void EngramPostWkv(const EngramGateParams& params, const uint16_t* hidden_states,
                   const uint16_t* kv, const uint16_t* q_weight,
                   const uint16_t* k_weight, const uint8_t* token_mask,
                   int64_t num_tokens, int64_t num_kv_tokens, uint16_t* out) {
  if (num_tokens == 0) return;  // engram.py:998-999.
  const int64_t hc = params.hc_mult;
  const int64_t dim = params.dim;
  VT_CHECK(hc > 0 && dim > 0, "engram gate: hc_mult and dim must be positive");
  VT_CHECK(hidden_states != nullptr && kv != nullptr && q_weight != nullptr &&
               k_weight != nullptr && out != nullptr,
           "engram gate: a required buffer is null");
  const int64_t kv_row = (hc + 1) * dim;

  for (int64_t token = 0; token < num_tokens; ++token) {
    // engram.py:798-799. Past the SP-local token count the key and the value
    // read as zero, which is how a padded shard tail stays finite.
    const bool source_valid = token < num_kv_tokens;
    for (int64_t h = 0; h < hc; ++h) {
      // Every load is widened to f32 (engram.py:810-825): the stream stays
      // bf16 and only the reduction is wide. Do not widen the stream.
      float hidden_sq = 0.0F;
      float key_sq = 0.0F;
      float dot = 0.0F;
      for (int64_t d = 0; d < dim; ++d) {
        const float hidden =
            vt::BF16ToF32(hidden_states[static_cast<size_t>((token * hc + h) * dim + d)]);
        const float key =
            source_valid
                ? vt::BF16ToF32(kv[static_cast<size_t>(token * kv_row + h * dim + d)])
                : 0.0F;
        const float q = vt::BF16ToF32(q_weight[static_cast<size_t>(h * dim + d)]);
        const float k = vt::BF16ToF32(k_weight[static_cast<size_t>(h * dim + d)]);
        hidden_sq += hidden * hidden;
        key_sq += key * key;
        dot += hidden * q * k * key;
      }
      // engram.py:827-830.
      const float hidden_rms =
          1.0F / std::sqrt(hidden_sq / static_cast<float>(dim) + params.eps);
      const float key_rms =
          1.0F / std::sqrt(key_sq / static_cast<float>(dim) + params.eps);
      dot *= hidden_rms * key_rms / std::sqrt(static_cast<float>(dim));
      // engram.py:831-833. THE CLAMP IS INSIDE THE SQRT and it is a MAXIMUM, so
      // the gate input magnitude floors at sqrt(1e-6) = 1e-3 and a tiny dot is
      // AMPLIFIED, not squashed. `copysign` keeps the sign of the ORIGINAL dot.
      float gate_input = std::sqrt(std::max(std::abs(dot), params.clamp_value));
      if (dot < 0.0F) gate_input = -gate_input;
      float gate = 1.0F / (1.0F + std::exp(-gate_input));
      if (token_mask != nullptr) {
        // engram.py:834-840. TRUE MEANS KEEP here, the OPPOSITE polarity to
        // `NgramHashInputs::dead_mask` (nvidia/model.py:578-580).
        const bool active =
            source_valid && token_mask[static_cast<size_t>(token)] != 0;
        if (!active) gate = 0.0F;
      }
      for (int64_t d = 0; d < dim; ++d) {
        const float hidden =
            vt::BF16ToF32(hidden_states[static_cast<size_t>((token * hc + h) * dim + d)]);
        // The VALUE is shared across the hc copies and lives after the hc keys
        // (engram.py:842-846).
        const float value =
            source_valid
                ? vt::BF16ToF32(kv[static_cast<size_t>(token * kv_row + hc * dim + d)])
                : 0.0F;
        out[static_cast<size_t>((token * hc + h) * dim + d)] =
            vt::F32ToBF16(hidden + gate * value);
      }
    }
  }
}

}  // namespace vllm::deepseek_v4_1
