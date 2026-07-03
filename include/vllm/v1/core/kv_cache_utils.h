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

#include <optional>
#include <string>
#include <vector>

namespace vllm::v1 {

// BlockHash represents the hash of a single KV-cache block used for prefix
// caching. Upstream: `NewType("BlockHash", bytes)`. Task 2 owns the hashing.
using BlockHash = std::string;

// BlockHashWithGroupId combines a BlockHash with its KV cache group ID, packed
// into raw bytes. Upstream: `NewType("BlockHashWithGroupId", bytes)`. Task 2
// adds the pack/unpack helpers.
using BlockHashWithGroupId = std::string;

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

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_KV_CACHE_UTILS_H_
