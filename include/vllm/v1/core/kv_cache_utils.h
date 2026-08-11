// Ported from: vllm/v1/core/kv_cache_utils.py @ e24d1b24
//
// Scope (M1.2 Task 1): the physical KV-cache block metadata (`KVCacheBlock`)
// and the intrusive doubly-linked LRU free list (`FreeKVCacheBlockQueue`). This
// is the correctness core of prefix caching's block allocator: the exact
// prev_free_block/next_free_block pointer manipulation and the fake head/tail
// sentinel design are mirrored 1:1 so that eviction order (FIFO/LRU) and O(1)
// middle-removal of an arbitrary cached-but-free block behave identically to
// upstream.
//
// Scope (M1.2 Task 2): the parent-chained, group-aware prefix-cache block
// hashing — NONE_HASH (the first-block sentinel), make_block_hash_with_group_id
// / get_block_hash / get_group_id (packing a BlockHash + 4-byte big-endian group
// id), hash_block_tokens (each block's hash chains the parent hash + this
// block's token ids + optional extra keys), hash_request_tokens (the per-request
// block-hash loop — N hashes for N full blocks, partial trailing block NOT
// hashed) and the pluggable hash function.
//
// HASH FUNCTION FIDELITY: upstream's hash function is `Callable[[Any], bytes]`;
// its two reproducible implementations are `sha256` (pickle serialization) and
// `sha256_cbor` (cbor2 canonical serialization), both SHA-256 of the serialized
// input (vllm/utils/hashing.py). We port `sha256_cbor` BYTE-FOR-BYTE: our
// `CborValue` is the stand-in for Python's "Any", `CborValue::Encode()` matches
// `cbor2.dumps(x, canonical=True)` for the value shapes block hashing produces,
// and `sha256_cbor(value)` == `hashlib.sha256(cbor2.dumps(...)).digest()`. The
// `sha256` (pickle) variant is intentionally NOT ported: Python's pickle opcode
// stream is impractical to reproduce in C++, and its expected test vectors are
// value-identical structurally to the CBOR ones (the upstream tests parametrize
// both and assert the same chaining/group/partial invariants). The hash function
// stays pluggable (`HashFn`) exactly as upstream so a caller may inject another
// hasher, but the shipped concrete hasher is `sha256_cbor`.
//
// Field/method names are kept EXACTLY as upstream (snake_case: block_id,
// ref_cnt, prev_free_block, next_free_block, popleft, remove, append,
// get_all_free_blocks, num_free_blocks, reset_hash, ...) — this overrides the
// repo's usual CamelCase convention because the plan mandates a 1:1 name match.
//
// BlockHash COORDINATION WITH TASK 2:
//   Upstream `BlockHash` and `BlockHashWithGroupId` are `NewType`s over `bytes`.
//   Here they are minimal byte-string aliases so KVCacheBlock can store the
//   optional hash key + its token count now. Task 2 (block hashing) fleshes out
//   the hashing machinery (NONE_HASH, hash_block_tokens, hash_request_tokens,
//   make_block_hash_with_group_id / get_block_hash / get_group_id) ON TOP of
//   these aliases WITHOUT reshaping KVCacheBlock: the stored field stays
//   `std::optional<BlockHashWithGroupId>`, a value type, regardless of how Task
//   2 refines the helpers. If Task 2 needs a strong type it can wrap the alias
//   without touching this struct's layout.
//
// DEVIATIONS, recorded:
//   - `incr_ref()` / `decr_ref()` are added per the M1.2 plan as convenience
//     helpers over `ref_cnt`. Upstream has no such methods — it mutates
//     `block.ref_cnt` directly (e.g. BlockPool.touch). `ref_cnt` is kept public
//     so direct mutation stays possible and matches upstream exactly.
//   - `__repr__` is not ported (not needed).
//   - The queue takes `KVCacheBlock*` (pointers into the pool's externally
//     owned block array) rather than owning the blocks; upstream's Python list
//     likewise holds shared references to the same block objects. The fake
//     head/tail sentinels are value members with stable addresses, so the queue
//     is non-copyable / non-movable.
#ifndef VLLM_V1_CORE_KV_CACHE_UTILS_H_
#define VLLM_V1_CORE_KV_CACHE_UTILS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace vllm::v1 {

struct KVCacheSpec;
struct KVCacheConfig;

// Hybrid-manager-disabled fallback: when full and sliding-window/chunked-local
// specs are mixed, convert local storage to full allocation while preserving
// the compute window/chunk on FullAttentionSpec. Mirrors
// kv_cache_utils.unify_hybrid_kv_cache_specs for the ported spec set.
void unify_hybrid_kv_cache_specs(
    std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>>&
        kv_cache_specs);

// BlockHash represents the hash of a single KV-cache block used for prefix
// caching. Upstream: `NewType("BlockHash", bytes)`. Held as raw bytes.
using BlockHash = std::string;

// BlockHashWithGroupId combines a BlockHash with its KV cache group ID, packed
// into raw bytes: the BlockHash bytes followed by the group id as 4 big-endian
// bytes. Upstream: `NewType("BlockHashWithGroupId", bytes)`.
using BlockHashWithGroupId = std::string;

// ExternalBlockHash — the reproducible, externally-published form of a block
// hash (kv_cache_utils.py:51-54). A union of the raw sha256 bytes and an int
// (the low 64 bits), kept for backward compatibility after block hashing
// defaulted to sha256 bytes. It is consumed ONLY by the KV-cache event payload
// (vllm/distributed/kv_events.py). std::string holds the raw bytes variant;
// uint64_t holds the int-truncated variant.
using ExternalBlockHash = std::variant<std::string, uint64_t>;

// maybe_convert_block_hash (kv_cache_utils.py:79-82): the sole crossing point of
// a block hash into the external event stream.
//   - By DEFAULT (env VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES parses truthy; its
//     upstream default is "1" -> True, envs.py:1816-1817), narrows the digest to
//     its low 64 bits, i.e. `int.from_bytes(hash_bytes, "big") & ((1<<64)-1)` —
//     the last 8 bytes interpreted big-endian.
//   - When the env is set to a falsy value ("0"), passes the raw bytes through.
// Mirrors upstream's `bool(int(os.getenv(name, "1")))` parse.
ExternalBlockHash maybe_convert_block_hash(const BlockHash& hash_bytes);

// ---------------------------------------------------------------------------
// Block-hash packing (make_block_hash_with_group_id / get_block_hash /
// get_group_id). Mirrors upstream: the group id is encoded using 4 bytes in
// big-endian order and appended to the block-hash bytes, avoiding tuple
// allocation while still allowing both components to be recovered.
// ---------------------------------------------------------------------------

// Pack a BlockHash and group id into a BlockHashWithGroupId.
BlockHashWithGroupId make_block_hash_with_group_id(const BlockHash& block_hash,
                                                   uint32_t group_id);

// Extract the BlockHash from a BlockHashWithGroupId (all but the last 4 bytes).
BlockHash get_block_hash(const BlockHashWithGroupId& key);

// Extract the group id from a BlockHashWithGroupId (last 4 bytes, big-endian).
uint32_t get_group_id(const BlockHashWithGroupId& key);

// ---------------------------------------------------------------------------
// The pluggable hash function and its canonical-CBOR concrete implementation.
// ---------------------------------------------------------------------------

// A minimal canonical-CBOR value model: our stand-in for Python's "Any" that
// upstream feeds to the hash function. It supports exactly the value shapes that
// occur in block-hash inputs (unsigned/negative ints, byte strings, UTF-8 text,
// arrays, and null) and serializes them with canonical encoding
// (definite-length, minimal-width integers) — byte-for-byte identical to
// `cbor2.dumps(x, canonical=True)` for those shapes.
class CborValue {
 public:
  static CborValue UInt(uint64_t value);
  static CborValue Int(int64_t value);
  static CborValue Bytes(std::string bytes);
  static CborValue Text(std::string text);
  static CborValue Array(std::vector<CborValue> items);
  static CborValue Null();

  // Append this value's canonical CBOR encoding to out.
  void Encode(std::string& out) const;
  // This value's canonical CBOR encoding.
  std::string Encode() const;

 private:
  enum class Type { kUInt, kNInt, kBytes, kText, kArray, kNull };
  Type type_ = Type::kNull;
  // kUInt: the value; kNInt: -1 - value (the CBOR "argument" of major type 1).
  uint64_t arg_ = 0;
  std::string str_;               // kBytes / kText payload.
  std::vector<CborValue> items_;  // kArray payload.
};

// The hash function type. Mirrors upstream `Callable[[Any], bytes]`; our "Any"
// is CborValue and "bytes" is BlockHash.
using HashFn = std::function<BlockHash(const CborValue&)>;

// SHA-256 over the canonical-CBOR serialization of value. Byte-for-byte
// identical to upstream `vllm.utils.hashing.sha256_cbor`.
BlockHash sha256_cbor(const CborValue& value);

// Raw SHA-256 of a byte string, returned as 32 RAW bytes (not hex). Exposed so
// the KV-offload identity header digests its canonical JSON with the SAME
// implementation the block hashes use — there must never be a second SHA-256 in
// this tree.
std::string sha256_bytes(const std::string& data);

// A single extra key for a block hash: either a text string (a LoRA name or a
// cache salt) or an (identifier, offset) pair (a multi-modal input). Mirrors the
// heterogeneous Python objects upstream places in the extra_keys tuple.
using ExtraKey = std::variant<std::string, std::pair<std::string, int64_t>>;

// The optional tuple of extra keys for one block. std::nullopt mirrors Python
// `None` (no extra keys), which hashes distinctly from an empty tuple.
using ExtraKeys = std::optional<std::vector<ExtraKey>>;

// The hash seed for the first block of any prefix block sequence. Set globally
// by init_none_hash. Upstream: module-global `NONE_HASH`.
extern BlockHash NONE_HASH;

// Where the NONE_HASH chain seed came from. Recorded so the value's provenance
// is auditable and can be written into a persisted-cache identity header (a
// cache written under one seed must never be read under another).
enum class NoneHashSeedSource : int {
  // An explicit seed argument was passed to init_none_hash.
  kExplicit = 0,
  // From VLLM_PREFIX_CACHING_HASH_SEED (our escape hatch).
  kEnvVllmCpp = 1,
  // From PYTHONHASHSEED (upstream's escape hatch, mirrored for parity so an
  // operator's existing vLLM deployment recipe keeps working).
  kEnvPythonHashSeed = 2,
  // No seed anywhere: the fixed built-in default (DETERMINISTIC BY DEFAULT).
  kDefault = 3,
  // VLLM_PREFIX_CACHING_HASH_SEED=random: 32 bytes of std::random_device.
  kRandom = 4,
};

// The provenance of the current NONE_HASH.
struct NoneHashProvenance {
  NoneHashSeedSource source = NoneHashSeedSource::kDefault;
  // The seed text that was hashed (empty for kRandom).
  std::string seed;
  // A short stable name for the source, for logs/records/headers.
  const char* source_name() const;
};

// The fixed built-in chain seed used when nothing else supplies one.
// DEVIATION FROM UPSTREAM, deliberate and recorded (kv-persistence-lmcache.md
// §B5): upstream falls back to os.urandom(32)
// (vllm/v1/core/kv_cache_utils.py:111-112), which makes block hashes differ
// across processes and silently yields a 0% hit rate on any content-addressed
// persisted cache unless the operator sets PYTHONHASHSEED identically
// everywhere (vllm/docs/features/kv_offloading_usage.md:117-120). Our threat
// model (a local library plus server) has no hash-DoS concern, so the
// deterministic value is the DEFAULT and randomness is the opt-in. This string
// is part of our on-disk cache identity: NEVER change it.
inline constexpr const char* kDefaultNoneHashSeed = "vllm.cpp/none_hash/v1";

// Initialize NONE_HASH. Mirrors upstream init_none_hash
// (vllm/v1/core/kv_cache_utils.py:99-114) with the seed resolution order:
//   1. the explicit `seed` argument, if given;
//   2. $VLLM_PREFIX_CACHING_HASH_SEED — the literal value "random" selects
//      upstream's os.urandom(32) behaviour, anything else is used as the seed;
//   3. $PYTHONHASHSEED — upstream's own escape hatch, mirrored;
//   4. kDefaultNoneHashSeed (deterministic; see the deviation note above).
// In every seeded case NONE_HASH = hash_fn(text(seed)), exactly as upstream.
void init_none_hash(const HashFn& hash_fn,
                    std::optional<std::string> seed = std::nullopt);

// The provenance of the NONE_HASH set by the last init_none_hash call.
const NoneHashProvenance& none_hash_provenance();

// KV-cache block metadata. Mirrors upstream's @dataclass(slots=True)
// KVCacheBlock. The prev_free_block / next_free_block links form the intrusive
// doubly-linked free list and should ONLY be manipulated by
// FreeKVCacheBlockQueue.
struct KVCacheBlock {
  explicit KVCacheBlock(int block_id, bool is_null = false)
      : block_id(block_id), is_null(is_null) {}

  // Block ID, ranging from 0 to num_gpu_blocks - 1.
  int block_id;
  // Reference count.
  int ref_cnt = 0;

  // The hash key (block hash + group id) of the block, only available when the
  // block is full and cached. (Upstream: _block_hash.)
  std::optional<BlockHashWithGroupId> block_hash_ = std::nullopt;
  // Number of prefix tokens covered by block_hash_. For full blocks this is the
  // full block boundary; partial aliases can end inside a cache block.
  // (Upstream: _block_hash_num_tokens.)
  std::optional<int> block_hash_num_tokens_ = std::nullopt;

  // Used to construct a doubly linked list for free blocks. These two
  // attributes should only be manipulated by FreeKVCacheBlockQueue.
  KVCacheBlock* prev_free_block = nullptr;
  KVCacheBlock* next_free_block = nullptr;

  // Whether the block is a null block that should never be cached.
  bool is_null = false;

  // Upstream property `block_hash`.
  const std::optional<BlockHashWithGroupId>& block_hash() const {
    return block_hash_;
  }
  // Upstream property `block_hash_num_tokens`.
  const std::optional<int>& block_hash_num_tokens() const {
    return block_hash_num_tokens_;
  }

  // Upstream set_block_hash: asserts the block has no hash yet.
  void set_block_hash(BlockHashWithGroupId block_hash,
                      std::optional<int> num_tokens = std::nullopt);

  // Reset the block hash when the block is evicted.
  void reset_hash();

  // Convenience ref-count helpers (see DEVIATIONS in the file header).
  void incr_ref() { ref_cnt += 1; }
  void decr_ref() { ref_cnt -= 1; }
};

// Organizes a list of KVCacheBlock objects into a doubly linked list of free
// blocks. Implemented (instead of a std::deque / std::list) to support removing
// a block in the middle of the queue in O(1) time by manipulating the
// prev_free_block / next_free_block attributes of the given blocks directly.
//
// The queue is ordered by block ID at the start. When a block is allocated and
// then freed, it is appended back with the eviction order:
//   1. The least recently used block is at the front (LRU).
//   2. If two blocks have the same last accessed time (allocated by the same
//      sequence), the one with more hash tokens (the tail of a block chain) is
//      at the front.
// This order is maintained by reversing the block order when freeing a
// request's blocks — that reversal happens outside this class.
class FreeKVCacheBlockQueue {
 public:
  // Args: blocks — pointers to the KVCacheBlock objects (owned elsewhere, e.g.
  // by the BlockPool's block array). The queue links them via their
  // prev/next_free_block fields.
  explicit FreeKVCacheBlockQueue(const std::vector<KVCacheBlock*>& blocks);

  // Non-copyable / non-movable: the sentinels are value members and blocks hold
  // raw pointers into them, so the queue must have a stable address.
  FreeKVCacheBlockQueue(const FreeKVCacheBlockQueue&) = delete;
  FreeKVCacheBlockQueue& operator=(const FreeKVCacheBlockQueue&) = delete;
  FreeKVCacheBlockQueue(FreeKVCacheBlockQueue&&) = delete;
  FreeKVCacheBlockQueue& operator=(FreeKVCacheBlockQueue&&) = delete;

  // Pop the first free block and reduce num_free_blocks by 1. Throws
  // std::runtime_error("No free blocks available") when empty (upstream raises
  // ValueError with that message).
  KVCacheBlock* popleft();

  // Pop the first n free blocks and reduce num_free_blocks by n.
  std::vector<KVCacheBlock*> popleft_n(int n);

  // Remove a block from the free list and reduce num_free_blocks by 1. O(1).
  void remove(KVCacheBlock* block);

  // Put a block back into the free list (at the tail) and increase
  // num_free_blocks by 1.
  void append(KVCacheBlock* block);

  // Put a list of blocks at the front of the free list.
  void prepend_n(const std::vector<KVCacheBlock*>& blocks);

  // Put a list of blocks back into the free list (at the tail).
  void append_n(const std::vector<KVCacheBlock*>& blocks);

  // Get all free blocks in the free list (front to back). Mainly for testing.
  std::vector<KVCacheBlock*> get_all_free_blocks() const;

  // Number of free blocks, kept in sync with the linked list on every push/pop.
  int num_free_blocks;

  // Fake head and tail sentinels for the doubly linked list. They are NEVER
  // popped, so every real block in the queue is guaranteed to have both a prev
  // and a next block. Public to mirror upstream's accessible attributes (and
  // the ported tests inspect them).
  KVCacheBlock fake_free_list_head{-1};
  KVCacheBlock fake_free_list_tail{-1};
};

// Forward declaration for generate_block_hash_extra_keys / the request block
// hasher (Request is defined in vllm/v1/request.h; only a const-ref is needed
// here).
struct Request;

// The request block hasher. Mirrors upstream's
// `Callable[[Request], list[BlockHash]]` (Request._block_hasher): given a
// request, returns the block hashes for its newly-complete full blocks (the
// ones not yet in request.block_hashes). A null BlockHasher mirrors upstream
// `block_hasher=None` => prefix caching off (update_block_hashes is a no-op).
using BlockHasher = std::function<std::vector<BlockHash>(const Request&)>;

// generate_block_hash_extra_keys — DEFERRED derivation.
//
// Upstream derives a block's extra hash keys from a Request's multi-modal
// features, LoRA name, cache salt, and prompt embeddings. None of those Request
// fields exist in the T0 Request port (see include/vllm/v1/request.h DEFERRED
// list), and the gate models (text-only GDN/MoE) never populate them. For a
// Request with no mm/LoRA/salt/embeds, upstream returns (None, start_mm_idx)
// unchanged — which is exactly what this returns. The signature is kept 1:1 so
// the mm/LoRA/salt/embeds branches can be filled in without a call-site change
// once those Request fields land.
std::pair<ExtraKeys, int> generate_block_hash_extra_keys(const Request& request,
                                                         int start_token_idx,
                                                         int end_token_idx,
                                                         int start_mm_idx);

// Compute the hash of one block's contents chained onto the preceding block(s).
// Mirrors upstream hash_block_tokens: the hash key is
// hash_function((parent_block_hash, tuple(token_ids), extra_keys)). A falsy
// parent (nullopt or an empty hash, matching Python `if not parent_block_hash`)
// is replaced with NONE_HASH. The current block is assumed full.
BlockHash hash_block_tokens(const HashFn& hash_function,
                            const std::optional<BlockHash>& parent_block_hash,
                            const std::vector<int32_t>& curr_block_token_ids,
                            const ExtraKeys& extra_keys = std::nullopt);

// Compute the list of block hashes for a request's token ids at block_size
// granularity. Mirrors the core loop of upstream get_request_block_hasher's
// request_block_hasher (computing from an empty prefix): each full block's hash
// chains the previous block's hash, so hash N depends on hash N-1; a partial
// trailing block is NOT hashed (so 6 tokens at block_size 3 yields 2 hashes).
// per_block_extra_keys supplies the extra keys for each block by index (the
// Request-bound generation is deferred, see generate_block_hash_extra_keys);
// blocks past its end use no extra keys.
std::vector<BlockHash> hash_request_tokens(
    const HashFn& hash_function, int block_size,
    const std::vector<int32_t>& token_ids,
    const std::vector<ExtraKeys>& per_block_extra_keys = {});

// Build the incremental request block hasher. Mirrors upstream
// get_request_block_hasher: the returned closure computes ONLY the not-yet-hashed
// full blocks of a request (starting at
// request.block_hashes.size() * hash_block_size), chaining each block's hash
// onto the previous one (request.block_hashes.back() as the parent). It hashes
// full blocks only; a partial trailing block is left unhashed. Request stores
// this closure as _block_hasher and calls it from update_block_hashes() at
// construction and after each append.
BlockHasher get_request_block_hasher(int hash_block_size,
                                     const HashFn& caching_hash_fn);

// Resolve (scheduler_block_size, hash_block_size) for a KV cache config.
// Ported 1:1 from vllm/v1/core/kv_cache_utils.py:626-688
// (resolve_kv_cache_block_sizes) @ 555967922.
//
// - scheduler_block_size is the token-alignment invariant used by the
//   scheduler. Single group: cache_block_size * dcp. Multiple groups: LCM of
//   every group's effective block size (attention groups scaled by DCP; mamba
//   groups keep their per-rank state, unscaled).
// - hash_block_size is the granularity (the "prefix match unit") at which a
//   request's block hashes are computed. Single group: equals scheduler block
//   size. Multiple groups: `prefix_match_unit` if set, else the GCD of group
//   block sizes; every group block size must be divisible by it. Returns the
//   scheduler block size (disabling finer hashing) when block hashing is
//   inactive (no prefix caching AND no connector) or a mamba group's block size
//   diverges from cache_block_size (mamba_cache_mode != "align").
//
// Deviation from upstream: takes the threaded config values explicitly
// (cache_block_size = cache_config.block_size, prefix_match_unit =
// cache_config.prefix_match_unit, enable_prefix_caching, connector_enabled =
// (kv_transfer_config != None), dcp = decode_context_parallel_size) rather than
// a single VllmConfig, because our config surface is threaded, not one
// dataclass. Throws std::invalid_argument on a non-divisible prefix_match_unit,
// mirroring upstream's ValueError.
std::pair<int, int> resolve_kv_cache_block_sizes(
    const KVCacheConfig& kv_cache_config, int cache_block_size,
    std::optional<int> prefix_match_unit, bool enable_prefix_caching,
    bool connector_enabled, int dcp_world_size = 1);

// ---------------------------------------------------------------------------
// Startup KV sizing: can the pool hold ONE max_model_len sequence?
//
// Ported from vllm/v1/core/kv_cache_utils.py @ 555967922:
//   max_memory_usage_bytes          :791-798  -> kv_memory_needed_bytes
//   estimate_max_model_len          :800-851  -> estimate_max_model_len
//   _check_enough_kv_cache_memory   :751-788  -> check_enough_kv_cache_memory
//   _auto_fit_max_model_len         :1967-2027 -> auto_fit_max_model_len
//
// WHY THIS EXISTS: without it a prompt larger than the KV pool is admitted,
// never allocates, and the engine spins at model_executed=0 with an idle GPU
// forever (external PR #227's report). Upstream never aborts such a waiter —
// scheduler.py:919-940 peeks and `break`s. It prevents the state at startup
// (here) and at admission (input_processor.py:387-432). Issue #83 M4.
//
// DEVIATIONS, all recorded:
//   - Upstream takes `Callable`s (get_needed_memory / estimate_max_model_len) so
//     the expensive device-profiling variants stay lazy. Both are pure
//     arithmetic for us, so they are passed as values.
//   - estimate_max_model_len upstream binary-searches `max_memory_usage_bytes`.
//     Our per-block geometry is exactly linear in the block count
//     (KVBytesPerBlock is block-count independent), so the search has the closed
//     form `num_blocks * block_size` and is written as such.
//   - The remediation sentence keeps upstream's wording and appends the knobs a
//     vllm.cpp user can actually act on: `gpu_memory_utilization` profiling is
//     un-ported (model_loader.cpp ResolveNumBlocks step 3, TODO ROAD-V1-MEM M3),
//     so `--num-blocks` / `--kv-cache-memory` / `--max-model-len` are the levers.

// max_memory_usage_bytes: the KV bytes ONE sequence of `max_model_len` tokens
// occupies, at this block geometry. Blocks are whole, so the token count is
// rounded up to a block boundary.
int64_t kv_memory_needed_bytes(int64_t max_model_len, int block_size,
                               int64_t bytes_per_block);

// estimate_max_model_len: the longest sequence `available_memory` can hold.
// Returns 0 when not even a single block fits.
int64_t estimate_max_model_len(int64_t available_memory,
                               int64_t bytes_per_block, int block_size);

// _check_enough_kv_cache_memory: throws std::invalid_argument (upstream
// ValueError) when the pool cannot hold one `max_model_len` sequence.
// `estimated_max_model_len` <= 0 suppresses the "Based on the available memory"
// clause exactly as upstream does.
void check_enough_kv_cache_memory(int64_t available_memory,
                                  int64_t needed_memory, int64_t max_model_len,
                                  int64_t estimated_max_model_len);

// _auto_fit_max_model_len: the length to serve when the caller did NOT pin one.
// Upstream reduces max_model_len to what the pool holds and logs the reduction;
// it raises when not even one token fits. `derived_max_model_len` is the
// checkpoint's own context length (upstream's `original_max`).
int64_t auto_fit_max_model_len(int64_t derived_max_model_len,
                               int64_t available_memory,
                               int64_t bytes_per_block, int block_size);

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_KV_CACHE_UTILS_H_
