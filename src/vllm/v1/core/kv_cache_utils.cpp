// Ported from: vllm/v1/core/kv_cache_utils.py @ e24d1b24
// See include/vllm/v1/core/kv_cache_utils.h for scope, the BlockHash/Task 2
// coordination note, and recorded deviations.
#include "vllm/v1/core/kv_cache_utils.h"

#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vllm::v1 {

void KVCacheBlock::set_block_hash(BlockHashWithGroupId block_hash,
                                  std::optional<int> num_tokens) {
  // "The block already has a hash. This should not happen."
  assert(!block_hash_.has_value() && !block_hash_num_tokens_.has_value());
  block_hash_ = std::move(block_hash);
  block_hash_num_tokens_ = num_tokens;
}

void KVCacheBlock::reset_hash() {
  block_hash_ = std::nullopt;
  block_hash_num_tokens_ = std::nullopt;
}

FreeKVCacheBlockQueue::FreeKVCacheBlockQueue(
    const std::vector<KVCacheBlock*>& blocks)
    : num_free_blocks(static_cast<int>(blocks.size())) {
  // Initialize doubly links of consecutive blocks.
  for (int i = 0; i < num_free_blocks; ++i) {
    if (i > 0) {
      blocks[i]->prev_free_block = blocks[i - 1];
    }
    if (i < num_free_blocks - 1) {
      blocks[i]->next_free_block = blocks[i + 1];
    }
  }

  // The fake head and tail are NEVER popped, so we can safely assume each real
  // block in the queue has prev and next blocks.
  if (num_free_blocks > 0) {
    // Connect fake_head and fake_tail to the first and last block respectively.
    fake_free_list_head.next_free_block = blocks.front();
    blocks.front()->prev_free_block = &fake_free_list_head;
    fake_free_list_tail.prev_free_block = blocks.back();
    blocks.back()->next_free_block = &fake_free_list_tail;
  } else {
    // For empty list, simply connect the fake head and tail.
    fake_free_list_head.next_free_block = &fake_free_list_tail;
    fake_free_list_tail.prev_free_block = &fake_free_list_head;
  }
}

KVCacheBlock* FreeKVCacheBlockQueue::popleft() {
  if (fake_free_list_head.next_free_block == &fake_free_list_tail ||
      fake_free_list_head.next_free_block == nullptr) {
    assert(num_free_blocks == 0 &&
           "num_free_blocks is out of sync with the free list.");
    throw std::runtime_error("No free blocks available");
  }

  KVCacheBlock* first_block = fake_free_list_head.next_free_block;

  if (first_block->next_free_block == nullptr) {
    // Indicates a bug in the caller's logic.
    throw std::runtime_error(
        "Invalid block found in popleft() "
        "which doesn't have a valid next_free_block");
  }

  // Connect fake_head and the next block of first_block (second block or fake
  // tail).
  fake_free_list_head.next_free_block = first_block->next_free_block;
  first_block->next_free_block->prev_free_block = &fake_free_list_head;

  // Remove the block from the linked list.
  first_block->prev_free_block = nullptr;
  first_block->next_free_block = nullptr;

  num_free_blocks -= 1;
  return first_block;
}

std::vector<KVCacheBlock*> FreeKVCacheBlockQueue::popleft_n(int n) {
  if (n == 0) {
    return {};
  }
  assert(num_free_blocks >= n);
  num_free_blocks -= n;

  KVCacheBlock* curr_block = fake_free_list_head.next_free_block;
  // Pop n blocks from the head of the list.
  std::vector<KVCacheBlock*> ret;
  ret.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    assert(curr_block != nullptr);
    ret.push_back(curr_block);
    KVCacheBlock* last_block = curr_block;
    curr_block = curr_block->next_free_block;
    // Reset prev_free_block and next_free_block of all popped blocks.
    last_block->prev_free_block = nullptr;
    last_block->next_free_block = nullptr;
  }

  if (curr_block != nullptr) {
    // The queue is not empty, connect the fake head to the new first block.
    fake_free_list_head.next_free_block = curr_block;
    curr_block->prev_free_block = &fake_free_list_head;
  }
  return ret;
}

void FreeKVCacheBlockQueue::remove(KVCacheBlock* block) {
  if (block->prev_free_block == nullptr || block->next_free_block == nullptr) {
    // Indicates a bug in the caller's logic.
    throw std::runtime_error("remove() called on an invalid block");
  }

  // Link the previous block to the next block.
  block->prev_free_block->next_free_block = block->next_free_block;
  // Link the next block to the previous block.
  block->next_free_block->prev_free_block = block->prev_free_block;

  // Remove the block from the linked list.
  block->prev_free_block = nullptr;
  block->next_free_block = nullptr;
  num_free_blocks -= 1;
}

void FreeKVCacheBlockQueue::append(KVCacheBlock* block) {
  if (fake_free_list_tail.prev_free_block == nullptr) {
    throw std::runtime_error(
        "prev_free_block of fake_free_list_tail should always exist");
  }
  KVCacheBlock* last_block = fake_free_list_tail.prev_free_block;

  // Connect the new block after the last block.
  last_block->next_free_block = block;
  block->prev_free_block = last_block;

  // Connect the fake tail after the new block.
  block->next_free_block = &fake_free_list_tail;
  fake_free_list_tail.prev_free_block = block;

  num_free_blocks += 1;
}

void FreeKVCacheBlockQueue::prepend_n(const std::vector<KVCacheBlock*>& blocks) {
  if (blocks.empty()) {
    return;
  }

  KVCacheBlock* first_block = fake_free_list_head.next_free_block;
  assert(first_block != nullptr &&
         "next_free_block of fake_free_list_head should always exist");

  KVCacheBlock* prev_block = &fake_free_list_head;
  for (KVCacheBlock* block : blocks) {
    block->prev_free_block = prev_block;
    prev_block->next_free_block = block;
    prev_block = block;
  }

  prev_block->next_free_block = first_block;
  first_block->prev_free_block = prev_block;

  num_free_blocks += static_cast<int>(blocks.size());
}

void FreeKVCacheBlockQueue::append_n(const std::vector<KVCacheBlock*>& blocks) {
  if (blocks.empty()) {
    return;
  }

  KVCacheBlock* last_block = fake_free_list_tail.prev_free_block;
  assert(last_block != nullptr &&
         "prev_free_block of fake_free_list_tail should always exist");
  // Add inter-connections between consecutive blocks.
  for (KVCacheBlock* block : blocks) {
    block->prev_free_block = last_block;
    last_block->next_free_block = block;
    last_block = block;
  }

  // Connect the last block of <blocks> to the fake tail.
  last_block->next_free_block = &fake_free_list_tail;
  fake_free_list_tail.prev_free_block = last_block;

  num_free_blocks += static_cast<int>(blocks.size());
}

std::vector<KVCacheBlock*> FreeKVCacheBlockQueue::get_all_free_blocks() const {
  std::vector<KVCacheBlock*> ret;
  if (fake_free_list_head.next_free_block == nullptr) {
    throw std::runtime_error(
        "next_free_block of fake_free_list_head should always exist");
  }
  // Start from the first block.
  KVCacheBlock* curr_block = fake_free_list_head.next_free_block;
  // As long as next_free_block is available, we haven't reached the fake tail.
  while (curr_block->next_free_block != nullptr) {
    ret.push_back(curr_block);
    curr_block = curr_block->next_free_block;
  }
  return ret;
}

}  // namespace vllm::v1
