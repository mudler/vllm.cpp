// Tests for KVCacheBlock + FreeKVCacheBlockQueue (M1.2 Task 1), ported from
// vllm/tests/v1/core/test_kv_cache_utils.py @ e24d1b24.
//
// Ported cases (the FreeKVCacheBlockQueue / KVCacheBlock behavioral oracle):
//   - test_kv_cache_block
//   - test_free_kv_cache_block_queue_initialization
//   - test_free_kv_cache_block_queue_operations
//   - test_free_kv_cache_block_queue_append_n
//   - test_free_kv_cache_block_queue_prepend_n
//   - test_free_kv_cache_block_queue_popleft_n
//   - test_free_kv_cache_block_queue_get_all_free_blocks
//
// Object-identity comparisons in the upstream tests (`block is blocks[i]`,
// `get_all_free_blocks() == blocks`) are ported as pointer-equality checks,
// since the queue links the same KVCacheBlock objects the pool owns.
//
// NOTE: test_kv_cache_block upstream builds the hash via
// make_block_hash_with_group_id (a Task 2 helper). Task 1 has no hashing yet,
// so the hash key is constructed directly as a byte-string BlockHashWithGroupId.
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"

using vllm::v1::BlockHashWithGroupId;
using vllm::v1::FreeKVCacheBlockQueue;
using vllm::v1::KVCacheBlock;

namespace {

// Build n KVCacheBlocks with block_id = 0..n-1. Reserve up front so no
// reallocation happens; moving the returned vector preserves element addresses
// (the buffer is stolen, elements are not relocated).
std::vector<KVCacheBlock> MakeBlocks(int n) {
  std::vector<KVCacheBlock> blocks;
  blocks.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    blocks.emplace_back(i);
  }
  return blocks;
}

std::vector<KVCacheBlock*> Ptrs(std::vector<KVCacheBlock>& blocks) {
  std::vector<KVCacheBlock*> ptrs;
  ptrs.reserve(blocks.size());
  for (auto& b : blocks) {
    ptrs.push_back(&b);
  }
  return ptrs;
}

}  // namespace

TEST_CASE("test_kv_cache_block") {
  // Test KVCacheBlock initialization.
  KVCacheBlock block(0);
  CHECK(block.block_id == 0);
  CHECK(block.ref_cnt == 0);
  CHECK_FALSE(block.block_hash().has_value());

  // Test reference count manipulation (direct, as upstream; plus the added
  // incr_ref/decr_ref helpers).
  block.ref_cnt += 1;
  CHECK(block.ref_cnt == 1);
  block.ref_cnt -= 1;
  CHECK(block.ref_cnt == 0);
  block.incr_ref();
  CHECK(block.ref_cnt == 1);
  block.decr_ref();
  CHECK(block.ref_cnt == 0);

  // Test block hash setting and resetting. (Upstream uses
  // make_block_hash_with_group_id(BlockHash(b"abc"), 0); Task 1 constructs the
  // packed key directly.)
  BlockHashWithGroupId block_hash =
      std::string("abc") + std::string("\x00\x00\x00\x00", 4);
  block.set_block_hash(block_hash);
  CHECK(block.block_hash().has_value());
  CHECK(block.block_hash().value() == block_hash);

  block.reset_hash();
  CHECK_FALSE(block.block_hash().has_value());
}

TEST_CASE("test_free_kv_cache_block_queue_initialization") {
  // Test with a single block.
  std::vector<KVCacheBlock> blocks = MakeBlocks(1);
  auto ptrs = Ptrs(blocks);
  FreeKVCacheBlockQueue queue(ptrs);
  CHECK(queue.num_free_blocks == 1);
  CHECK(queue.fake_free_list_head.next_free_block == ptrs[0]);
  CHECK(queue.fake_free_list_tail.prev_free_block == ptrs[0]);
}

TEST_CASE("test_free_kv_cache_block_queue_operations") {
  std::vector<KVCacheBlock> blocks = MakeBlocks(5);
  auto b = Ptrs(blocks);
  FreeKVCacheBlockQueue queue(b);

  // Check initial state.
  CHECK(queue.num_free_blocks == 5);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[4]);

  // Pop the first block.
  KVCacheBlock* block1 = queue.popleft();
  CHECK(block1 == b[0]);
  CHECK(queue.num_free_blocks == 4);
  CHECK(queue.fake_free_list_head.next_free_block == b[1]);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[4]);

  // Remove a block from the middle.
  KVCacheBlock* block_to_remove = b[2];
  queue.remove(block_to_remove);
  CHECK(queue.num_free_blocks == 3);
  CHECK(b[1]->next_free_block == b[3]);
  CHECK(b[3]->prev_free_block == b[1]);

  // Append a block back.
  queue.append(block_to_remove);
  CHECK(queue.num_free_blocks == 4);
  CHECK(queue.fake_free_list_tail.prev_free_block == block_to_remove);
  CHECK(block_to_remove->prev_free_block == b[4]);
  CHECK(block_to_remove->next_free_block == &queue.fake_free_list_tail);

  // Pop blocks until empty.
  for (int i = 0; i < 4; ++i) {
    queue.popleft();
  }
  CHECK(queue.num_free_blocks == 0);
  CHECK(queue.fake_free_list_head.next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == &queue.fake_free_list_head);

  // Attempt to pop from an empty queue.
  CHECK_THROWS_WITH_AS(queue.popleft(), "No free blocks available",
                       std::runtime_error);
}

TEST_CASE("test_free_kv_cache_block_queue_append_n") {
  // Create an empty FreeKVCacheBlockQueue.
  std::vector<KVCacheBlock*> empty;
  FreeKVCacheBlockQueue queue(empty);
  std::vector<KVCacheBlock> blocks = MakeBlocks(6);
  auto b = Ptrs(blocks);

  // Append 0 block: fake_head->fake_tail
  queue.append_n({});
  CHECK(queue.num_free_blocks == 0);
  CHECK(queue.fake_free_list_head.next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == &queue.fake_free_list_head);

  // Append 1 block: fake_head->b0->fake_tail
  queue.append_n({b[0]});
  CHECK(queue.num_free_blocks == 1);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[0]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[0]);

  // Append 2 blocks: fake_head->b0->b4->b5->fake_tail
  queue.append_n({b[4], b[5]});
  CHECK(queue.num_free_blocks == 3);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[0]->next_free_block == b[4]);
  CHECK(b[4]->prev_free_block == b[0]);
  CHECK(b[4]->next_free_block == b[5]);
  CHECK(b[5]->prev_free_block == b[4]);
  CHECK(b[5]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[5]);

  // Append 3 blocks: fake_head->b0->b4->b5->b1->b2->b3->fake_tail
  queue.append_n({b[1], b[2], b[3]});
  CHECK(queue.num_free_blocks == 6);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(b[0]->next_free_block == b[4]);
  CHECK(b[4]->next_free_block == b[5]);
  CHECK(b[5]->next_free_block == b[1]);
  CHECK(b[1]->prev_free_block == b[5]);
  CHECK(b[1]->next_free_block == b[2]);
  CHECK(b[2]->prev_free_block == b[1]);
  CHECK(b[2]->next_free_block == b[3]);
  CHECK(b[3]->prev_free_block == b[2]);
  CHECK(b[3]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[3]);
}

TEST_CASE("test_free_kv_cache_block_queue_prepend_n") {
  // Seed the queue with one block so prepend has an existing head to splice in
  // front of (fake_head->b0->fake_tail).
  std::vector<KVCacheBlock> blocks = MakeBlocks(6);
  auto b = Ptrs(blocks);
  std::vector<KVCacheBlock*> seed{b[0]};
  FreeKVCacheBlockQueue queue(seed);

  // Prepend 0 blocks is a no-op.
  queue.prepend_n({});
  CHECK(queue.num_free_blocks == 1);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);

  // Prepend 2 blocks: fake_head->b4->b5->b0->fake_tail
  queue.prepend_n({b[4], b[5]});
  CHECK(queue.num_free_blocks == 3);
  CHECK(queue.fake_free_list_head.next_free_block == b[4]);
  CHECK(b[4]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[4]->next_free_block == b[5]);
  CHECK(b[5]->prev_free_block == b[4]);
  CHECK(b[5]->next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == b[5]);
  CHECK(b[0]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[0]);

  // A second prepend goes ahead of everything previously prepended.
  // fake_head->b1->b2->b4->b5->b0->fake_tail
  queue.prepend_n({b[1], b[2]});
  CHECK(queue.num_free_blocks == 5);
  CHECK(queue.fake_free_list_head.next_free_block == b[1]);
  CHECK(b[1]->next_free_block == b[2]);
  CHECK(b[2]->next_free_block == b[4]);

  // The popleft order reflects the front-to-back queue order.
  std::vector<int> popped;
  for (int i = 0; i < 5; ++i) {
    popped.push_back(queue.popleft()->block_id);
  }
  CHECK(popped == std::vector<int>{1, 2, 4, 5, 0});
  CHECK(queue.num_free_blocks == 0);
}

TEST_CASE("test_free_kv_cache_block_queue_popleft_n") {
  std::vector<KVCacheBlock> blocks = MakeBlocks(6);
  auto b = Ptrs(blocks);
  std::vector<KVCacheBlock*> order{b[1], b[3], b[5], b[4], b[0], b[2]};
  FreeKVCacheBlockQueue queue(order);
  CHECK(queue.num_free_blocks == 6);
  CHECK(queue.fake_free_list_head.next_free_block == b[1]);
  CHECK(b[1]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[1]->next_free_block == b[3]);
  CHECK(b[3]->prev_free_block == b[1]);
  CHECK(b[3]->next_free_block == b[5]);
  CHECK(b[5]->prev_free_block == b[3]);
  CHECK(b[5]->next_free_block == b[4]);
  CHECK(b[4]->prev_free_block == b[5]);
  CHECK(b[4]->next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == b[4]);
  CHECK(b[0]->next_free_block == b[2]);
  CHECK(b[2]->prev_free_block == b[0]);
  CHECK(b[2]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[2]);

  // Pop 0 block: fake_head->b1->b3->b5->b4->b0->b2->fake_tail
  CHECK(queue.popleft_n(0).empty());
  CHECK(queue.num_free_blocks == 6);

  // Pop 1 block: fake_head->b3->b5->b4->b0->b2->fake_tail
  auto r1 = queue.popleft_n(1);
  CHECK(queue.num_free_blocks == 5);
  CHECK(r1.size() == 1);
  CHECK(r1[0] == b[1]);
  for (auto* blk : r1) {
    CHECK(blk->prev_free_block == nullptr);
    CHECK(blk->next_free_block == nullptr);
  }

  // Pop 2 blocks: fake_head->b4->b0->b2->fake_tail
  auto r2 = queue.popleft_n(2);
  CHECK(r2.size() == 2);
  CHECK(queue.num_free_blocks == 3);
  CHECK(r2[0] == b[3]);
  CHECK(r2[1] == b[5]);
  for (auto* blk : r2) {
    CHECK(blk->prev_free_block == nullptr);
    CHECK(blk->next_free_block == nullptr);
  }

  // Pop 3 blocks: fake_head->fake_tail
  auto r3 = queue.popleft_n(3);
  CHECK(r3.size() == 3);
  CHECK(queue.num_free_blocks == 0);
  CHECK(r3[0] == b[4]);
  CHECK(r3[1] == b[0]);
  CHECK(r3[2] == b[2]);
  for (auto* blk : r3) {
    CHECK(blk->prev_free_block == nullptr);
    CHECK(blk->next_free_block == nullptr);
  }
}

TEST_CASE("test_free_kv_cache_block_queue_get_all_free_blocks") {
  std::vector<KVCacheBlock> blocks = MakeBlocks(5);
  auto b = Ptrs(blocks);
  FreeKVCacheBlockQueue queue(b);

  // Check all blocks are correctly retrieved.
  CHECK(queue.get_all_free_blocks() == b);

  // Pop a block and check again.
  queue.popleft();
  CHECK(queue.get_all_free_blocks() ==
        std::vector<KVCacheBlock*>{b[1], b[2], b[3], b[4]});

  // Remove a block and check again.
  KVCacheBlock* block_to_remove = b[2];
  queue.remove(block_to_remove);
  CHECK(queue.get_all_free_blocks() ==
        std::vector<KVCacheBlock*>{b[1], b[3], b[4]});

  // Append a block back and check again.
  queue.append(block_to_remove);
  CHECK(queue.get_all_free_blocks() ==
        std::vector<KVCacheBlock*>{b[1], b[3], b[4], b[2]});
}
