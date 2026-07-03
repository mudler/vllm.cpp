// Tests for the M1.6 Task 1 attention interface — CommonAttentionMetadata built
// from the M1.5 step-inputs + the AttentionBackend get_kv_cache_shape contract.
//
// Ported from vllm/v1/attention/backend.py @ e24d1b24 (CommonAttentionMetadata)
// + vllm/v1/attention/backends/flash_attn.py::get_kv_cache_shape @ e24d1b24. The
// build oracle mirrors tests/v1/attention/utils.py::create_common_attn_metadata
// (query_start_loc = [0] ++ cumsum(query_lens); max_query_len = max query_lens;
// max_seq_len = max seq_lens; num_actual_tokens = sum query_lens). No CUDA, no
// model: a fake StepInputs on host arrays (2 reqs: prefill N=4 + decode 1).
#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::CommonAttentionMetadata;
using vllm::v1::FlashAttentionBackend;
using vllm::v1::MakeCommonAttentionMetadata;
using vllm::v1::StepInputs;

namespace {

// A fake step-input: 2 reqs. Req 0 is a prefill of 4 tokens (seq_len 4);
// req 1 is a decode of 1 token over a 6-token context (seq_len 6).
StepInputs make_step() {
  StepInputs step;
  step.query_start_loc = {0, 4, 5};    // [0] ++ cumsum([4, 1])
  step.seq_lens = {4, 6};              // num_computed + num_scheduled
  step.num_scheduled_tokens = {4, 1};
  step.input_token_ids = {10, 11, 12, 13, 99};
  step.positions = {0, 1, 2, 3, 5};   // req1 decode at absolute pos 5
  step.logits_indices = {3, 4};       // query_start_loc[1:] - 1
  // One KV cache group; slot ids are arbitrary but sized num_actual_tokens (5).
  step.slot_mapping = {{0, 1, 2, 3, 80}};
  return step;
}

}  // namespace

TEST_CASE("MakeCommonAttentionMetadata: prefill + decode step") {
  const StepInputs step = make_step();
  // block table: req0 needs ceil(4/16)=1 block, req1 ceil(6/16)=1 block.
  // [num_reqs=2, num_cols=1], flattened row-major.
  const std::vector<int32_t> block_table_flat = {7, 42};

  const CommonAttentionMetadata cam = MakeCommonAttentionMetadata(
      step, block_table_flat, /*block_table_num_cols=*/1);

  CHECK(cam.query_start_loc == std::vector<int32_t>{0, 4, 5});
  CHECK(cam.query_start_loc_cpu == std::vector<int32_t>{0, 4, 5});
  CHECK(cam.seq_lens == std::vector<int32_t>{4, 6});
  CHECK(cam.seq_lens_cpu == std::vector<int32_t>{4, 6});
  CHECK(cam.num_reqs == 2);
  CHECK(cam.num_actual_tokens == 5);
  CHECK(cam.max_query_len == 4);
  CHECK(cam.max_seq_len == 6);
  CHECK(cam.causal == true);
  CHECK(cam.batch_size() == 2);

  // naive_query_lens = query_start_loc[1:] - query_start_loc[:-1].
  CHECK(cam.naive_query_lens() == std::vector<int32_t>{4, 1});

  // block table + slot mapping pass through unchanged.
  CHECK(cam.block_table_tensor == block_table_flat);
  CHECK(cam.block_table_num_cols == 1);
  CHECK(cam.slot_mapping == std::vector<int64_t>{0, 1, 2, 3, 80});
}

TEST_CASE("MakeCommonAttentionMetadata: causal flag + group selection") {
  StepInputs step = make_step();
  step.slot_mapping = {{0, 1, 2, 3, 80}, {100, 101, 102, 103, 180}};

  const CommonAttentionMetadata g0 =
      MakeCommonAttentionMetadata(step, {7, 42}, 1, /*causal=*/true,
                                  /*kv_cache_group_id=*/0);
  const CommonAttentionMetadata g1 =
      MakeCommonAttentionMetadata(step, {7, 42}, 1, /*causal=*/false,
                                  /*kv_cache_group_id=*/1);

  CHECK(g0.slot_mapping == std::vector<int64_t>{0, 1, 2, 3, 80});
  CHECK(g0.causal == true);
  CHECK(g1.slot_mapping == std::vector<int64_t>{100, 101, 102, 103, 180});
  CHECK(g1.causal == false);
}

TEST_CASE("MakeCommonAttentionMetadata: empty batch throws") {
  StepInputs step;  // no requests
  CHECK_THROWS_AS(MakeCommonAttentionMetadata(step, {}, 0),
                  std::invalid_argument);
}

TEST_CASE("FlashAttentionBackend: name + kv_cache_shape") {
  const FlashAttentionBackend backend;
  CHECK(backend.get_name() == "FLASH_ATTN");

  // flash_attn.py layout: (num_blocks, 2, block_size, num_kv_heads, head_size).
  const std::vector<int64_t> shape = backend.get_kv_cache_shape(
      /*num_blocks=*/10, /*block_size=*/16, /*num_kv_heads=*/2,
      /*head_size=*/128);
  CHECK(shape == std::vector<int64_t>{10, 2, 16, 2, 128});
}

TEST_CASE("FlashAttentionBackend: block_size must be multiple of 16") {
  const FlashAttentionBackend backend;
  CHECK_THROWS_AS(backend.get_kv_cache_shape(10, 15, 2, 128),
                  std::invalid_argument);
}
