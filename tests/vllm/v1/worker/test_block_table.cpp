// Tests for BlockTable + MultiGroupBlockTable (M1.5 Task 1) — the per-request
// KV-cache block-id storage + slot mapping the persistent InputBatch consumes.
//
// Ported from vllm/v1/worker/block_table.py @ e24d1b24 (NOT gpu/block_table.py,
// which at e24d1b24 is the unrelated `BlockTables` staged-write design — see the
// header's stale-brief note). Upstream has no direct unit test for BlockTable;
// it is exercised indirectly through InputBatch.condense in
// tests/v1/worker/test_gpu_input_batch.py (which imports BlockTable /
// MultiGroupBlockTable from vllm.v1.worker.block_table and compares them via
// _compare_objs). The cases below are the behavioral oracle taken directly from
// that source's semantics, incl. the M1.5 brief's slot-mapping oracle
// (block_size 16, row [3,7], positions [0,16,17] -> slots [48,112,113]).
//
// compute_slot_mapping reads the "device" buffer (upstream reads
// block_table.gpu), so every slot-mapping case commits first — this also
// demonstrates the host-array cpu-vs-device split.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vllm/v1/worker/gpu/block_table.h"

using vllm::v1::BlockTable;
using vllm::v1::MultiGroupBlockTable;

TEST_CASE("BlockTable.add_row stores block ids for a request row") {
  BlockTable bt(/*block_size=*/16, /*max_num_reqs=*/4,
                /*max_num_blocks_per_req=*/8, /*max_num_batched_tokens=*/32,
                /*kernel_block_size=*/16);

  bt.add_row({3, 7}, /*row_idx=*/1);
  CHECK(bt.num_blocks_per_row[1] == 2);
  CHECK(bt.cpu_block_id(1, 0) == 3);
  CHECK(bt.cpu_block_id(1, 1) == 7);
  // Other rows untouched.
  CHECK(bt.num_blocks_per_row[0] == 0);

  // add_row on a non-empty row resets it first.
  bt.add_row({5}, /*row_idx=*/1);
  CHECK(bt.num_blocks_per_row[1] == 1);
  CHECK(bt.cpu_block_id(1, 0) == 5);
}

TEST_CASE("BlockTable.append_row extends an existing row") {
  BlockTable bt(16, 4, 8, 32, 16);

  bt.add_row({1, 2}, 0);
  bt.append_row({4, 8}, 0);
  CHECK(bt.num_blocks_per_row[0] == 4);
  CHECK(bt.cpu_block_id(0, 0) == 1);
  CHECK(bt.cpu_block_id(0, 1) == 2);
  CHECK(bt.cpu_block_id(0, 2) == 4);
  CHECK(bt.cpu_block_id(0, 3) == 8);

  // append_row with an empty list is a no-op.
  bt.append_row({}, 0);
  CHECK(bt.num_blocks_per_row[0] == 4);
}

TEST_CASE("BlockTable.clear_row / clear reset state") {
  BlockTable bt(16, 4, 8, 32, 16);
  bt.add_row({3, 7}, 1);

  bt.clear_row(1);
  CHECK(bt.num_blocks_per_row[1] == 0);
  CHECK(bt.cpu_block_id(1, 0) == 0);
  CHECK(bt.cpu_block_id(1, 1) == 0);

  bt.add_row({9}, 2);
  bt.commit_block_table(3);
  CHECK(bt.device_block_id(2, 0) == 9);
  bt.clear();
  CHECK(bt.cpu_block_id(2, 0) == 0);
  CHECK(bt.device_block_id(2, 0) == 0);
}

TEST_CASE("BlockTable.move_row copies src row onto tgt (condense)") {
  BlockTable bt(16, 4, 8, 32, 16);
  bt.add_row({3, 7, 11}, /*src=*/2);

  bt.move_row(/*src=*/2, /*tgt=*/0);
  CHECK(bt.num_blocks_per_row[0] == 3);
  CHECK(bt.cpu_block_id(0, 0) == 3);
  CHECK(bt.cpu_block_id(0, 1) == 7);
  CHECK(bt.cpu_block_id(0, 2) == 11);
}

TEST_CASE("BlockTable.swap_row swaps two rows (lengths + contents)") {
  BlockTable bt(16, 4, 8, 32, 16);
  bt.add_row({1, 2}, 0);
  bt.add_row({5, 6, 7}, 1);

  bt.swap_row(0, 1);
  CHECK(bt.num_blocks_per_row[0] == 3);
  CHECK(bt.num_blocks_per_row[1] == 2);
  CHECK(bt.cpu_block_id(0, 0) == 5);
  CHECK(bt.cpu_block_id(0, 1) == 6);
  CHECK(bt.cpu_block_id(0, 2) == 7);
  CHECK(bt.cpu_block_id(1, 0) == 1);
  CHECK(bt.cpu_block_id(1, 1) == 2);
}

TEST_CASE("BlockTable.commit_block_table exposes cpu rows on the device buffer") {
  BlockTable bt(16, 4, 8, 32, 16);
  bt.add_row({3, 7}, 0);

  // Device buffer is stale (all zero) before commit.
  CHECK(bt.device_block_id(0, 0) == 0);
  CHECK(bt.device_block_id(0, 1) == 0);

  bt.commit_block_table(1);
  CHECK(bt.device_block_id(0, 0) == 3);
  CHECK(bt.device_block_id(0, 1) == 7);
}

TEST_CASE("BlockTable.compute_slot_mapping = block_id*block_size + offset") {
  // The M1.5 brief oracle: block_size 16, row [3,7], positions [0,16,17].
  BlockTable bt(16, 4, 8, 32, 16);
  bt.add_row({3, 7}, /*row_idx=*/0);
  bt.commit_block_table(1);  // compute_slot_mapping reads the device buffer.

  const std::vector<int32_t> query_start_loc = {0, 3};  // req 0 has 3 tokens
  const std::vector<int64_t> positions = {0, 16, 17};
  bt.compute_slot_mapping(/*num_reqs=*/1, query_start_loc, positions);

  const auto& slots = bt.slot_mapping();
  CHECK(slots[0] == 48);   // block 3 * 16 + 0
  CHECK(slots[1] == 112);  // block 7 * 16 + 0
  CHECK(slots[2] == 113);  // block 7 * 16 + 1
  // Tail-pad is NO LONGER written by compute_slot_mapping (dead work: the only
  // consumer slices [0, num_tokens) and the decode graph re-pads via
  // BuildPaddedDecode — see the block_table.h padding-deviation note). The fill
  // is bounded to [0, num_tokens); the tail keeps its prior value (0 from the
  // constructor here), NOT PAD_SLOT_ID.
  CHECK(slots[3] == 0);
  CHECK(slots[static_cast<size_t>(bt.max_num_batched_tokens) - 1] == 0);
}

TEST_CASE("BlockTable.compute_slot_mapping over two requests") {
  BlockTable bt(16, 4, 8, 64, 16);
  bt.add_row({3, 7}, 0);  // req 0
  bt.add_row({2}, 1);     // req 1
  bt.commit_block_table(2);

  // req 0: positions [0,16,17]; req 1: positions [0,1] -> block 2.
  const std::vector<int32_t> query_start_loc = {0, 3, 5};
  const std::vector<int64_t> positions = {0, 16, 17, 0, 1};
  bt.compute_slot_mapping(2, query_start_loc, positions);

  const auto& slots = bt.slot_mapping();
  CHECK(slots[0] == 48);
  CHECK(slots[1] == 112);
  CHECK(slots[2] == 113);
  CHECK(slots[3] == 32);  // block 2 * 16 + 0
  CHECK(slots[4] == 33);  // block 2 * 16 + 1
  // Bounded fill: the tail past num_tokens (5) is untouched (0), not PAD_SLOT_ID.
  CHECK(slots[5] == 0);
}

TEST_CASE("BlockTable hybrid blocks expand kv-manager ids to kernel ids") {
  // Allocation block size 32, kernel block size 16 -> blocks_per_kv_block 2.
  BlockTable bt(/*block_size=*/32, /*max_num_reqs=*/2,
                /*max_num_blocks_per_req=*/4, /*max_num_batched_tokens=*/16,
                /*kernel_block_size=*/16);
  CHECK(bt.use_hybrid_blocks == true);
  CHECK(bt.blocks_per_kv_block == 2);
  CHECK(bt.block_size == 16);                 // becomes the kernel block size
  CHECK(bt.max_num_blocks_per_req == 8);      // 4 * 2

  bt.add_row({0, 1, 2}, 0);
  // Each kv block b -> [2b, 2b+1].
  const std::vector<int> expect = {0, 1, 2, 3, 4, 5};
  CHECK(bt.num_blocks_per_row[0] == 6);
  for (int i = 0; i < 6; ++i) {
    CHECK(bt.cpu_block_id(0, i) == expect[static_cast<size_t>(i)]);
  }

  CHECK(BlockTable::map_to_kernel_blocks({0, 1, 2}, 2) ==
        std::vector<int>{0, 1, 2, 3, 4, 5});
  CHECK(BlockTable::map_to_kernel_blocks({5}, 1) == std::vector<int>{5});
}

TEST_CASE("MultiGroupBlockTable fans out to two groups with different sizes") {
  // Two groups: full-attn block_size 16, GDN-state block_size 32.
  MultiGroupBlockTable mgbt(/*max_num_reqs=*/4, /*max_model_len=*/256,
                            /*max_num_batched_tokens=*/32,
                            /*block_sizes=*/{16, 32},
                            /*kernel_block_sizes=*/{16, 32});
  REQUIRE(mgbt.block_tables.size() == 2);
  CHECK(mgbt[0].block_size == 16);
  CHECK(mgbt[1].block_size == 32);

  // add_row supplies per-group block lists.
  mgbt.add_row({{3, 7}, {5}}, /*row_idx=*/0);
  CHECK(mgbt[0].num_blocks_per_row[0] == 2);
  CHECK(mgbt[0].cpu_block_id(0, 0) == 3);
  CHECK(mgbt[0].cpu_block_id(0, 1) == 7);
  CHECK(mgbt[1].num_blocks_per_row[0] == 1);
  CHECK(mgbt[1].cpu_block_id(0, 0) == 5);

  // append_row + move/swap fan out to every group.
  mgbt.append_row({{11}, {6}}, 0);
  CHECK(mgbt[0].num_blocks_per_row[0] == 3);
  CHECK(mgbt[1].num_blocks_per_row[0] == 2);

  mgbt.move_row(0, 1);
  CHECK(mgbt[0].num_blocks_per_row[1] == 3);
  CHECK(mgbt[1].num_blocks_per_row[1] == 2);

  // Per-group slot mapping uses each group's own block_size.
  mgbt.commit_block_table(2);
  const std::vector<int32_t> query_start_loc = {0, 1};
  const std::vector<int64_t> positions = {0};
  mgbt.compute_slot_mapping(1, query_start_loc, positions);
  // Group 0: block 3 * 16 + 0 = 48; group 1: block 5 * 32 + 0 = 160.
  CHECK(mgbt[0].slot_mapping()[0] == 48);
  CHECK(mgbt[1].slot_mapping()[0] == 160);
}

TEST_CASE("MultiGroupBlockTable derives + aligns max_num_blocks per group") {
  // block_size 16 -> align multiple 128/16 = 8; cdiv(256,16)=16 already aligned.
  // block_size 48 -> block_size > ... 48<=128 -> mult 128/48 = 2;
  // cdiv(256,48)=6 -> cdiv(6,2)*2 = 6.
  MultiGroupBlockTable mgbt(2, 256, 16, {16, 48}, {16, 48});
  CHECK(mgbt[0].max_num_blocks_per_req == 16);
  CHECK(mgbt[1].max_num_blocks_per_req == 6);
}
