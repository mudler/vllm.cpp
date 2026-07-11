// Tests for KVCacheBlock + FreeKVCacheBlockQueue (M1.2 Task 1) and the
// parent-chained, group-aware prefix-cache block hashing (M1.2 Task 2), ported
// from vllm/tests/v1/core/test_kv_cache_utils.py @ e24d1b24.
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
// Ported hashing cases (Task 2 behavioral oracle):
//   - test_none_hash (seeded reproducibility; different seeds differ)
//   - test_hash_block_tokens (byte-exact vs sha256_cbor)
//   - test_request_block_hasher (mm extra keys, chaining; byte-exact)
//   - test_hash_request_tokens_no_mm_inputs (chaining; byte-exact)
// Plus structural checks the upstream parametrization asserts across hashers:
// parent chaining changes the hash, group id differentiates, the partial last
// block is not hashed, and NONE_HASH is used as the first-block parent.
//
// C5 W1 additionally ports test_unify_hybrid_kv_cache_specs from
// vllm/tests/v1/core/test_kv_cache_utils.py:2420-2487 @ e24d1b24.
//
// HASH-FUNCTION FIDELITY: upstream parametrizes these over `sha256` (pickle) and
// `sha256_cbor` (cbor2 canonical). We port `sha256_cbor` BYTE-FOR-BYTE (see the
// header) and pin absolute goldens against it; `sha256` (pickle) is out of scope
// for byte-parity but its structural invariants are identical and are covered
// here. Goldens were generated with cbor2 6.1.2 + hashlib (canonical=True); the
// raw-CBOR checks below independently prove the serialization matches
// `cbor2.dumps(x, canonical=True)`.
//
// Object-identity comparisons in the upstream tests (`block is blocks[i]`,
// `get_all_free_blocks() == blocks`) are ported as pointer-equality checks,
// since the queue links the same KVCacheBlock objects the pool owns.
//
// NOTE: test_kv_cache_block upstream builds the hash via
// make_block_hash_with_group_id; Task 2 now provides it, and the case below uses
// it (Task 1 constructed the packed key directly).
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::BlockHash;
using vllm::v1::BlockHashWithGroupId;
using vllm::v1::CborValue;
using vllm::v1::ExtraKey;
using vllm::v1::ExtraKeys;
using vllm::v1::FreeKVCacheBlockQueue;
using vllm::v1::hash_block_tokens;
using vllm::v1::hash_request_tokens;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::KVCacheSpec;
using vllm::v1::FullAttentionSpec;
using vllm::v1::make_block_hash_with_group_id;
using vllm::v1::MambaSpec;
using vllm::v1::sha256_cbor;
using vllm::v1::SlidingWindowSpec;
using vllm::v1::unify_hybrid_kv_cache_specs;

namespace {

// Lowercase hex of a byte string, for comparing digests to cbor2+hashlib
// goldens.
std::string ToHex(const std::string& bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 0xf]);
  }
  return out;
}

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

  // Test block hash setting and resetting. Upstream:
  // make_block_hash_with_group_id(BlockHash(b"abc"), 0).
  BlockHashWithGroupId block_hash = make_block_hash_with_group_id("abc", 0);
  CHECK(block_hash == std::string("abc") + std::string("\x00\x00\x00\x00", 4));
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

// ---------------------------------------------------------------------------
// Task 2: block hashing.
// ---------------------------------------------------------------------------

// Independently prove CborValue::Encode() == cbor2.dumps(x, canonical=True).
TEST_CASE("cbor_value_canonical_encoding") {
  CHECK(ToHex(CborValue::Bytes("123").Encode()) == "43313233");
  CHECK(ToHex(CborValue::Array({CborValue::Int(1), CborValue::Int(2),
                                CborValue::Int(3)})
                  .Encode()) == "83010203");
  CHECK(ToHex(CborValue::Text("seed42").Encode()) == "66736565643432");
  CHECK(ToHex(CborValue::Null().Encode()) == "f6");
  // Negative ints (multi-modal offsets can be negative).
  CHECK(ToHex(CborValue::Int(-1).Encode()) == "20");
  CHECK(ToHex(CborValue::Int(-3).Encode()) == "22");
  CHECK(ToHex(CborValue::Int(-257).Encode()) == "390100");
  // Minimal-width integer arguments.
  CHECK(ToHex(CborValue::UInt(23).Encode()) == "17");
  CHECK(ToHex(CborValue::UInt(24).Encode()) == "1818");
  CHECK(ToHex(CborValue::UInt(255).Encode()) == "18ff");
  CHECK(ToHex(CborValue::UInt(256).Encode()) == "190100");
  CHECK(ToHex(CborValue::UInt(65536).Encode()) == "1a00010000");
  // The full (parent, tokens, extra_keys) tuple upstream feeds the hasher.
  CborValue tuple = CborValue::Array(
      {CborValue::Bytes("123"),
       CborValue::Array({CborValue::Int(1), CborValue::Int(2), CborValue::Int(3)}),
       CborValue::Array({CborValue::Text("key1"), CborValue::Text("key2")})});
  CHECK(ToHex(tuple.Encode()) ==
        "83433132338301020382646b657931646b657932");
}

// test_none_hash: seeded init is reproducible (== sha256_cbor(text(seed))),
// yields non-empty 32-byte hashes, and different seeds differ.
TEST_CASE("test_none_hash") {
  init_none_hash(sha256_cbor, "seed42");
  CHECK(vllm::v1::NONE_HASH.size() == 32);
  CHECK(ToHex(vllm::v1::NONE_HASH) ==
        "4dee92a73b42a5488db700713f330b42f7c8b90ae46143db43aef808eb2099d7");
  CHECK(vllm::v1::NONE_HASH == sha256_cbor(CborValue::Text("seed42")));

  const BlockHash seed42 = vllm::v1::NONE_HASH;
  init_none_hash(sha256_cbor, "different seed");
  CHECK(vllm::v1::NONE_HASH != seed42);

  // Unseeded: 32 random bytes, non-empty (upstream os.urandom(32)).
  init_none_hash(sha256_cbor);
  CHECK(vllm::v1::NONE_HASH.size() == 32);
}

// test_hash_block_tokens: block_hash == hash_fn((parent, tokens, extra_keys)),
// pinned byte-for-byte against sha256_cbor.
TEST_CASE("test_hash_block_tokens") {
  const BlockHash parent = "123";
  const std::vector<int32_t> tokens = {1, 2, 3};
  const ExtraKeys extra = std::vector<ExtraKey>{ExtraKey(std::string("key1")),
                                                ExtraKey(std::string("key2"))};

  const BlockHash block_hash = hash_block_tokens(sha256_cbor, parent, tokens, extra);
  const BlockHash expected = sha256_cbor(CborValue::Array(
      {CborValue::Bytes("123"),
       CborValue::Array({CborValue::Int(1), CborValue::Int(2), CborValue::Int(3)}),
       CborValue::Array({CborValue::Text("key1"), CborValue::Text("key2")})}));
  CHECK(block_hash == expected);
  CHECK(ToHex(block_hash) ==
        "75f96f75f082518549ca45da83610d619f8c63528c78f8aef470d43f1647317d");
}

// test_hash_request_tokens_no_mm_inputs: 6 tokens at block_size 3 -> 2 hashes,
// each chaining the previous (block 1 depends on block 0), pinned byte-for-byte.
TEST_CASE("test_hash_request_tokens_no_mm_inputs") {
  init_none_hash(sha256_cbor, "seed42");
  const std::vector<int32_t> tokens = {0, 1, 2, 3, 4, 5};
  const std::vector<BlockHash> hashes = hash_request_tokens(sha256_cbor, 3, tokens);

  REQUIRE(hashes.size() == 2);
  CHECK(hashes[0] == sha256_cbor(CborValue::Array(
                         {CborValue::Bytes(vllm::v1::NONE_HASH),
                          CborValue::Array({CborValue::Int(0), CborValue::Int(1),
                                            CborValue::Int(2)}),
                          CborValue::Null()})));
  CHECK(hashes[1] == sha256_cbor(CborValue::Array(
                         {CborValue::Bytes(hashes[0]),
                          CborValue::Array({CborValue::Int(3), CborValue::Int(4),
                                            CborValue::Int(5)}),
                          CborValue::Null()})));
  CHECK(ToHex(hashes[0]) ==
        "210559fa9941d0cd01ae4176a437cd8865c50b64e06af48138364280356c9aa9");
  CHECK(ToHex(hashes[1]) ==
        "855499304e237ca9bbbfff065f17f73d26744f5f30766249bb6e8f80f1bdf89a");
}

// test_request_block_hasher: per-block mm extra keys (("hashN", 0),), chained,
// pinned byte-for-byte.
TEST_CASE("test_request_block_hasher") {
  init_none_hash(sha256_cbor, "seed42");
  const std::vector<int32_t> tokens = {0, 1, 2, 3, 4, 5};
  const std::vector<ExtraKeys> per_block = {
      std::vector<ExtraKey>{ExtraKey(std::pair<std::string, int64_t>("hash1", 0))},
      std::vector<ExtraKey>{ExtraKey(std::pair<std::string, int64_t>("hash2", 0))},
  };
  const std::vector<BlockHash> hashes =
      hash_request_tokens(sha256_cbor, 3, tokens, per_block);

  REQUIRE(hashes.size() == 2);
  CHECK(ToHex(hashes[0]) ==
        "ae58025bbbd9b4375741f34b9ac177fe0ce7fa4bac8c57483ee86116bb92ee92");
  CHECK(ToHex(hashes[1]) ==
        "3b3218c497c0e23fa619550c8d57781f7f740d70214c4e1e6a54593a5da40911");
  // Different mm keys -> different hashes than the no-mm chain.
  const std::vector<BlockHash> no_mm = hash_request_tokens(sha256_cbor, 3, tokens);
  CHECK(hashes[0] != no_mm[0]);
  CHECK(hashes[1] != no_mm[1]);
}

// Parent chaining: block N's hash depends on block N-1's (and on NONE_HASH for
// block 0).
TEST_CASE("hash_block_tokens_parent_chaining") {
  init_none_hash(sha256_cbor, "seed42");
  const std::vector<int32_t> block0 = {0, 1, 2};
  const std::vector<int32_t> block1 = {3, 4, 5};

  const BlockHash h0 = hash_block_tokens(sha256_cbor, std::nullopt, block0);
  // A falsy parent (nullopt or empty) uses NONE_HASH as the parent.
  CHECK(hash_block_tokens(sha256_cbor, BlockHash(), block0) == h0);
  CHECK(h0 == hash_block_tokens(sha256_cbor, vllm::v1::NONE_HASH, block0));

  const BlockHash h1_after_h0 = hash_block_tokens(sha256_cbor, h0, block1);
  const BlockHash h1_after_none = hash_block_tokens(sha256_cbor, std::nullopt, block1);
  // Same tokens, different parent -> different hash.
  CHECK(h1_after_h0 != h1_after_none);
}

// Group id differentiation via make_block_hash_with_group_id / get_block_hash /
// get_group_id.
TEST_CASE("block_hash_with_group_id") {
  const BlockHash bh = hash_block_tokens(sha256_cbor, std::nullopt, {1, 2, 3});
  const BlockHashWithGroupId g0 = make_block_hash_with_group_id(bh, 0);
  const BlockHashWithGroupId g1 = make_block_hash_with_group_id(bh, 1);

  // Same block hash, different group id -> different packed key.
  CHECK(g0 != g1);
  CHECK(g0.size() == bh.size() + 4);

  // Round-trip both components.
  CHECK(vllm::v1::get_block_hash(g0) == bh);
  CHECK(vllm::v1::get_block_hash(g1) == bh);
  CHECK(vllm::v1::get_group_id(g0) == 0u);
  CHECK(vllm::v1::get_group_id(g1) == 1u);

  // 4-byte big-endian group-id encoding.
  const BlockHashWithGroupId big = make_block_hash_with_group_id("abc", 0x01020304u);
  CHECK(big == std::string("abc") + std::string("\x01\x02\x03\x04", 4));
  CHECK(vllm::v1::get_group_id(big) == 0x01020304u);
}

// Partial last block is not hashed: only full blocks produce a hash.
TEST_CASE("hash_request_tokens_partial_block_not_hashed") {
  init_none_hash(sha256_cbor, "seed42");
  // 7 tokens at block_size 3 -> 2 full blocks, trailing partial ignored.
  const std::vector<BlockHash> h7 =
      hash_request_tokens(sha256_cbor, 3, {0, 1, 2, 3, 4, 5, 6});
  CHECK(h7.size() == 2);
  // Fewer tokens than a block -> no hashes.
  CHECK(hash_request_tokens(sha256_cbor, 3, {0, 1}).empty());
  // Exactly one block.
  CHECK(hash_request_tokens(sha256_cbor, 3, {0, 1, 2}).size() == 1);
  // The full-block prefix of a longer sequence matches the shorter one.
  const std::vector<BlockHash> h6 =
      hash_request_tokens(sha256_cbor, 3, {0, 1, 2, 3, 4, 5});
  CHECK(h7 == h6);
}

// get_request_block_hasher (the incremental Request._block_hasher): the hashes a
// Request accumulates must equal the batch hash_request_tokens oracle
// (byte-exact vs the pin), and appending tokens extends block_hashes only when a
// new full block completes.
TEST_CASE("get_request_block_hasher matches hash_request_tokens (byte-exact)") {
  using vllm::v1::get_request_block_hasher;
  using vllm::v1::Request;
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;

  // 14 tokens -> 3 full blocks; the ctor runs update_block_hashes once.
  std::vector<int32_t> tokens;
  for (int i = 0; i < 14; ++i) tokens.push_back(i);
  Request req("0", tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
              get_request_block_hasher(block_size, sha256_cbor));

  const std::vector<BlockHash> oracle =
      hash_request_tokens(sha256_cbor, block_size, tokens);
  REQUIRE(oracle.size() == 3);
  CHECK(req.block_hashes == oracle);

  // Appending 2 tokens completes a 4th full block (16 tokens total).
  req.AppendOutputToken(std::vector<int32_t>{14, 15});
  std::vector<int32_t> tokens16 = tokens;
  tokens16.push_back(14);
  tokens16.push_back(15);
  const std::vector<BlockHash> oracle16 =
      hash_request_tokens(sha256_cbor, block_size, tokens16);
  REQUIRE(oracle16.size() == 4);
  CHECK(req.block_hashes == oracle16);
  // The first 3 hashes are unchanged (chaining is prefix-stable).
  CHECK(req.block_hashes[0] == oracle[0]);
  CHECK(req.block_hashes[1] == oracle[1]);
  CHECK(req.block_hashes[2] == oracle[2]);

  // Appending 1 token (17 total) does not complete a new full block.
  req.AppendOutputToken(16);
  CHECK(req.block_hashes.size() == 4);

  // A Request with no hasher (upstream block_hasher=None) never hashes.
  Request no_hash("1", tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0);
  CHECK(no_hash.block_hashes.empty());
  no_hash.AppendOutputToken(std::vector<int32_t>{14, 15});
  CHECK(no_hash.block_hashes.empty());
}

TEST_CASE("unify_hybrid_kv_cache_specs converts sliding storage to full") {
  auto full = std::make_shared<FullAttentionSpec>(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
      vt::DType::kF32);
  auto sliding = std::make_shared<SlidingWindowSpec>(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
      vt::DType::kF32, /*sliding_window=*/1024,
      /*head_size_v=*/32, vllm::v1::KVQuantMode::kNone,
      /*page_size_padded=*/32768);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", full}, {"layer_2", sliding}};

  unify_hybrid_kv_cache_specs(specs);
  CHECK(specs["layer_1"] == full);
  auto converted =
      std::dynamic_pointer_cast<FullAttentionSpec>(specs["layer_2"]);
  REQUIRE(converted != nullptr);
  CHECK(converted->block_size == 16);
  CHECK(converted->num_kv_heads == 2);
  CHECK(converted->head_size == 64);
  CHECK(converted->head_size_v == 32);
  CHECK(converted->page_size_padded == std::optional<int64_t>{32768});
  CHECK(converted->sliding_window == std::optional<int>{1024});
}

TEST_CASE("unify_hybrid_kv_cache_specs leaves uniform sliding specs unchanged") {
  auto sliding_a = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/1024);
  auto sliding_b = std::make_shared<SlidingWindowSpec>(
      16, 4, 32, vt::DType::kF32, /*sliding_window=*/1024);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", sliding_a}, {"layer_2", sliding_b}};
  unify_hybrid_kv_cache_specs(specs);
  CHECK(specs["layer_1"] == sliding_a);
  CHECK(specs["layer_2"] == sliding_b);
}

TEST_CASE("unify_hybrid_kv_cache_specs rejects an unconvertible mixed policy") {
  auto sliding_a = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/1024);
  auto sliding_b = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/256);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", sliding_a}, {"layer_2", sliding_b}};
  CHECK_THROWS_WITH_AS(
      unify_hybrid_kv_cache_specs(specs),
      "Hybrid KV cache manager is disabled but failed to convert the KV cache "
      "specs to one unified type.",
      std::invalid_argument);
}
