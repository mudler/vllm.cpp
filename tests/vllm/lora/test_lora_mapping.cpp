// LoRA metadata + mapping layer tests.
//
// UPSTREAM tests re-expressed (${VLLM_SOURCE} @ 555967922):
//   tests/lora/test_punica_ops.py:36-290  sgmv segment clustering + reference
//                                         (check_lora_shrink/expand_kernel uses
//                                         generate_data_for_nslices which builds
//                                         the b_seq_start_loc / seq_len_tensor /
//                                         lora_indices_tensor structure that
//                                         compute_meta produces)
//   tests/lora/utils.py:108-200          PunicaTensors, generate_data_for_nslices
//                                         (the sgmv metadata structure)
//   tests/lora/test_layers.py:310-465     LoRAMapping + update_metadata
//                                         integration (builds LoRAMapping from
//                                         index_mapping / prompt_mapping, calls
//                                         punica_wrapper.update_metadata)
//   tests/lora/test_layers_utils.py       _get_lora_device — SKIP (not portable,
//                                         tracked to W4)
//
// We test compute_meta and convert_mapping directly (the kernel apply they feed
// is already gated in W1). The reference is an independent hand-computed
// expected output, not the code under test.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vllm/lora/mapping.h"

using vllm::lora::ConvertedMapping;
using vllm::lora::ConvertMapping;
using vllm::lora::ComputeMeta;
using vllm::lora::LoRAMapping;
using vllm::lora::LoRAMappingType;
using vllm::lora::PunicaWrapperBase;
using vllm::lora::SgmvMeta;

// ---------------------------------------------------------------------------
// compute_meta (punica_wrapper/utils.py:15-50)
// ---------------------------------------------------------------------------

TEST_CASE("compute_meta: single segment") {
  // [1, 1, 1, 1] → one segment, lora_id=1, length=4
  std::vector<int64_t> token_lora = {1, 1, 1, 1};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.lora_indices.size() == 1);
  CHECK(m.lora_indices[0] == 1);
  CHECK(m.seq_lengths.size() == 1);
  CHECK(m.seq_lengths[0] == 4);
  CHECK(m.b_seq_start.size() == 1);
  CHECK(m.b_seq_start[0] == 0);
  CHECK(m.batch_size == 1);
  CHECK(m.max_length == 4);
  CHECK(m.token_nums == 4);
  CHECK_FALSE(m.no_lora);
}

TEST_CASE("compute_meta: multiple segments") {
  // [2, 2, 3, 3, 3, 1] → three segments: [2]x2, [3]x3, [1]x1
  std::vector<int64_t> token_lora = {2, 2, 3, 3, 3, 1};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 3);
  CHECK(m.lora_indices.size() == 3);
  CHECK(m.lora_indices[0] == 2);
  CHECK(m.lora_indices[1] == 3);
  CHECK(m.lora_indices[2] == 1);
  CHECK(m.seq_lengths.size() == 3);
  CHECK(m.seq_lengths[0] == 2);
  CHECK(m.seq_lengths[1] == 3);
  CHECK(m.seq_lengths[2] == 1);
  CHECK(m.b_seq_start.size() == 3);
  CHECK(m.b_seq_start[0] == 0);
  CHECK(m.b_seq_start[1] == 2);
  CHECK(m.b_seq_start[2] == 5);
  CHECK(m.max_length == 3);
  CHECK(m.token_nums == 6);
  CHECK_FALSE(m.no_lora);
}

TEST_CASE("compute_meta: no_lora when whole batch is -1") {
  // punica_wrapper/utils.py:40-41: no_lora when batch_size==1 and
  // lora_indices == -1
  std::vector<int64_t> token_lora = {-1, -1, -1};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 1);
  CHECK(m.lora_indices[0] == -1);
  CHECK(m.no_lora);
}

TEST_CASE("compute_meta: -1 mixed with adapters is not no_lora") {
  // [-1, -1, 2, 2] → two segments; no_lora is false
  std::vector<int64_t> token_lora = {-1, -1, 2, 2};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 2);
  CHECK(m.lora_indices[0] == -1);
  CHECK(m.lora_indices[1] == 2);
  CHECK_FALSE(m.no_lora);
  CHECK(m.token_nums == 4);
}

TEST_CASE("compute_meta: each token is its own segment") {
  // [1, 2, 3] → three segments of length 1
  std::vector<int64_t> token_lora = {1, 2, 3};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 3);
  CHECK(m.seq_lengths[0] == 1);
  CHECK(m.seq_lengths[1] == 1);
  CHECK(m.seq_lengths[2] == 1);
  CHECK(m.b_seq_start[0] == 0);
  CHECK(m.b_seq_start[1] == 1);
  CHECK(m.b_seq_start[2] == 2);
  CHECK(m.max_length == 1);
  CHECK(m.token_nums == 3);
}

TEST_CASE("compute_meta: non-consecutive same id produces separate segments") {
  // [1, 2, 1] → three segments: [1]x1, [2]x1, [1]x1
  // This distinguishes unique_consecutive (3 segments) from plain unique
  // (2 segments) and is the core guarantee of compute_meta.
  std::vector<int64_t> token_lora = {1, 2, 1};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 3);
  CHECK(m.token_nums == 3);
  CHECK(m.max_length == 1);
  CHECK_FALSE(m.no_lora);
  REQUIRE(m.lora_indices.size() == 3);
  CHECK(m.lora_indices[0] == 1);
  CHECK(m.lora_indices[1] == 2);
  CHECK(m.lora_indices[2] == 1);
  REQUIRE(m.seq_lengths.size() == 3);
  CHECK(m.seq_lengths[0] == 1);
  CHECK(m.seq_lengths[1] == 1);
  CHECK(m.seq_lengths[2] == 1);
  REQUIRE(m.b_seq_start.size() == 3);
  CHECK(m.b_seq_start[0] == 0);
  CHECK(m.b_seq_start[1] == 1);
  CHECK(m.b_seq_start[2] == 2);
}

TEST_CASE("compute_meta: empty input") {
  std::vector<int64_t> token_lora;
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 0);
  CHECK(m.token_nums == 0);
  CHECK(m.max_length == 0);
  CHECK_FALSE(m.no_lora);
  CHECK(m.lora_indices.empty());
  CHECK(m.seq_lengths.empty());
  CHECK(m.b_seq_start.empty());
}

TEST_CASE("compute_meta: consecutive same id then different (mirrors generate_data_for_nslices)") {
  // Mirrors the structure generate_data_for_nslices (utils.py:179-189) builds:
  // two batches with seq_length=2 each, first batch lora_id=0, second lora_id=1.
  std::vector<int64_t> token_lora = {0, 0, 1, 1};
  SgmvMeta m = ComputeMeta(token_lora);

  CHECK(m.batch_size == 2);
  CHECK(m.lora_indices[0] == 0);
  CHECK(m.lora_indices[1] == 1);
  CHECK(m.seq_lengths[0] == 2);
  CHECK(m.seq_lengths[1] == 2);
  CHECK(m.b_seq_start[0] == 0);
  CHECK(m.b_seq_start[1] == 2);
  CHECK(m.max_length == 2);
  CHECK(m.token_nums == 4);
  CHECK_FALSE(m.no_lora);
}

// ---------------------------------------------------------------------------
// convert_mapping (punica_wrapper/utils.py:54-160)
// ---------------------------------------------------------------------------

TEST_CASE("convert_mapping: basic two-adapter batch") {
  // Two adapters: lora_id 1 → slot 0, lora_id 2 → slot 1.
  // lora_index_to_id = [1, 2] (slot 0 holds id 1, slot 1 holds id 2).
  LoRAMapping mapping;
  mapping.index_mapping = {1, 1, 2, 2};  // tokens: adapter 1, 1, 2, 2
  mapping.prompt_mapping = {1, 2};        // requests: adapter 1, 2
  mapping.is_prefill = false;

  std::vector<int64_t> lora_index_to_id = {1, 2};
  int64_t max_loras = 4;
  int64_t vocab_size = 512;

  ConvertedMapping conv = ConvertMapping(mapping, lora_index_to_id,
                                         max_loras, vocab_size, 0);

  // base_indices: id 1 → slot 0, id 2 → slot 1
  CHECK(conv.base_indices.size() == 4);
  CHECK(conv.base_indices[0] == 0);
  CHECK(conv.base_indices[1] == 0);
  CHECK(conv.base_indices[2] == 1);
  CHECK(conv.base_indices[3] == 1);

  // sampler_indices: same mapping for prompt_mapping
  CHECK(conv.sampler_indices.size() == 2);
  CHECK(conv.sampler_indices[0] == 0);
  CHECK(conv.sampler_indices[1] == 1);

  // sampler_indices_padded: i + padded[i] * n, n=2
  // [0 + 0*2, 1 + 1*2] = [0, 3]
  CHECK(conv.sampler_indices_padded.size() == 2);
  CHECK(conv.sampler_indices_padded[0] == 0);
  CHECK(conv.sampler_indices_padded[1] == 3);

  // embeddings_indices: [embedding_indices * 0, embedding_indices * 512]
  // embedding_indices = [0, 0, 1, 1] (slot index if adapter, 0 if none)
  // row 0 = [0, 0, 0, 0], row 1 = [0, 0, 512, 512]
  CHECK(conv.embeddings_indices.size() == 8);  // 2 * 4
  CHECK(conv.embeddings_indices[0] == 0);  // row 0, col 0
  CHECK(conv.embeddings_indices[1] == 0);  // row 0, col 1
  CHECK(conv.embeddings_indices[2] == 0);  // row 0, col 2
  CHECK(conv.embeddings_indices[3] == 0);  // row 0, col 3
  CHECK(conv.embeddings_indices[4] == 0);    // row 1, col 0 (slot 0 * 512)
  CHECK(conv.embeddings_indices[5] == 0);    // row 1, col 1 (slot 0 * 512)
  CHECK(conv.embeddings_indices[6] == 512);  // row 1, col 2 (slot 1 * 512)
  CHECK(conv.embeddings_indices[7] == 512);  // row 1, col 3 (slot 1 * 512)

  // indices_len
  CHECK(conv.indices_len[0] == 4);  // base_indices
  CHECK(conv.indices_len[1] == 2);  // sampler_indices
  CHECK(conv.indices_len[2] == 2);  // sampler_indices_padded
  CHECK(conv.indices_len[3] == 4);  // embeddings_indices (column count)
}

TEST_CASE("convert_mapping: -1 sentinel (no adapter)") {
  // Tokens with lora_id <= 0 get slot index -1 in base_indices.
  LoRAMapping mapping;
  mapping.index_mapping = {1, 0, 2, -1};  // 0 and -1 = no adapter
  mapping.prompt_mapping = {1, 0};         // 0 = no adapter
  mapping.is_prefill = false;

  std::vector<int64_t> lora_index_to_id = {1, 2};
  int64_t max_loras = 4;
  int64_t vocab_size = 512;

  ConvertedMapping conv = ConvertMapping(mapping, lora_index_to_id,
                                         max_loras, vocab_size, 0);

  // base_indices: [0, -1, 1, -1]
  CHECK(conv.base_indices[0] == 0);
  CHECK(conv.base_indices[1] == -1);
  CHECK(conv.base_indices[2] == 1);
  CHECK(conv.base_indices[3] == -1);

  // sampler_indices: [0, -1]
  CHECK(conv.sampler_indices[0] == 0);
  CHECK(conv.sampler_indices[1] == -1);

  // sampler_indices_padded: -1 → max_loras-1 = 3, then i + padded * n
  // [0 + 0*2, 1 + 3*2] = [0, 7]
  CHECK(conv.sampler_indices_padded[0] == 0);
  CHECK(conv.sampler_indices_padded[1] == 7);

  // embedding_indices: 0 when no adapter (not -1)
  // row 0 = [0, 0, 0, 0] (all * 0)
  // row 1 = [0, 0, 512, 0] (slot * 512; slot=0 for no-adapter)
  CHECK(conv.embeddings_indices[4] == 0);    // no adapter → 0
  CHECK(conv.embeddings_indices[5] == 0);    // no adapter → 0
  CHECK(conv.embeddings_indices[6] == 512);  // slot 1 * 512
  CHECK(conv.embeddings_indices[7] == 0);    // no adapter → 0
}

TEST_CASE("convert_mapping: unused slots in lora_index_to_id") {
  // lora_index_to_id has -1 (None) for unused slots.
  // Slot 0: id 5, Slot 1: unused (-1), Slot 2: id 3.
  LoRAMapping mapping;
  mapping.index_mapping = {5, 3, 5};
  mapping.prompt_mapping = {5, 3};
  mapping.is_prefill = false;

  std::vector<int64_t> lora_index_to_id = {5, -1, 3};
  int64_t max_loras = 4;
  int64_t vocab_size = 256;

  ConvertedMapping conv = ConvertMapping(mapping, lora_index_to_id,
                                         max_loras, vocab_size, 0);

  // id 5 → slot 0, id 3 → slot 2
  CHECK(conv.base_indices[0] == 0);  // id 5 → slot 0
  CHECK(conv.base_indices[1] == 2);  // id 3 → slot 2
  CHECK(conv.base_indices[2] == 0);  // id 5 → slot 0

  CHECK(conv.sampler_indices[0] == 0);  // id 5 → slot 0
  CHECK(conv.sampler_indices[1] == 2);  // id 3 → slot 2
}

TEST_CASE("convert_mapping: extra_vocab_size is 0 (mirrors punica_base.py:178)") {
  // With extra_vocab_size=0, embeddings_indices row 0 is always 0.
  LoRAMapping mapping;
  mapping.index_mapping = {1, 2};
  mapping.prompt_mapping = {1};
  mapping.is_prefill = false;

  std::vector<int64_t> lora_index_to_id = {1, 2};
  ConvertedMapping conv = ConvertMapping(mapping, lora_index_to_id,
                                         4, 512, 0);

  // Row 0 = embedding_indices * 0 = all zeros
  CHECK(conv.embeddings_indices[0] == 0);
  CHECK(conv.embeddings_indices[1] == 0);
  // Row 1 = embedding_indices * (512 + 0) = slot * 512
  CHECK(conv.embeddings_indices[2] == 0);    // slot 0 * 512
  CHECK(conv.embeddings_indices[3] == 512);  // slot 1 * 512
}

// ---------------------------------------------------------------------------
// PunicaWrapperBase (punica_base.py:124-299)
// ---------------------------------------------------------------------------

TEST_CASE("PunicaWrapperBase: decode (is_prefill=false)") {
  // update_metadata for a decode batch: sets index buffers, does NOT
  // populate sgmv prefill metadata.
  PunicaWrapperBase wrapper(/*max_num_batched_tokens=*/16, /*max_batches=*/8);

  LoRAMapping mapping;
  mapping.index_mapping = {1, 2, 1};  // 3 tokens
  mapping.prompt_mapping = {1, 2};     // 2 requests
  mapping.is_prefill = false;

  std::vector<int64_t> lora_index_to_id = {1, 2};
  wrapper.UpdateMetadata(mapping, lora_index_to_id, /*max_loras=*/4,
                         /*vocab_size=*/512);

  // token_lora_indices (base_indices): id 1 → slot 0, id 2 → slot 1
  CHECK(wrapper.token_lora_indices_len() == 3);
  CHECK(wrapper.token_lora_indices()[0] == 0);
  CHECK(wrapper.token_lora_indices()[1] == 1);
  CHECK(wrapper.token_lora_indices()[2] == 0);

  // sampler_indices
  CHECK(wrapper.sampler_indices_len() == 2);
  CHECK(wrapper.sampler_indices()[0] == 0);
  CHECK(wrapper.sampler_indices()[1] == 1);

  // sampler_indices_padded: [0+0*2, 1+1*2] = [0, 3]
  CHECK(wrapper.sampler_indices_padded_len() == 2);
  CHECK(wrapper.sampler_indices_padded()[0] == 0);
  CHECK(wrapper.sampler_indices_padded()[1] == 3);

  // embeddings_indices_len == 3 (column count)
  CHECK(wrapper.embeddings_indices_len() == 3);

  // Prefill metadata should NOT be set for decode.
  CHECK_FALSE(wrapper.is_prefill());
  CHECK(wrapper.batch_size() == -1);  // uninitialized
}

TEST_CASE("PunicaWrapperBase: prefill (is_prefill=true)") {
  // update_metadata for a prefill batch: sets index buffers AND clusters
  // into sgmv segments.
  PunicaWrapperBase wrapper(/*max_num_batched_tokens=*/16, /*max_batches=*/8);

  LoRAMapping mapping;
  // Two segments: [1,1,1] then [2,2] — 5 tokens total.
  mapping.index_mapping = {1, 1, 1, 2, 2};
  mapping.prompt_mapping = {1, 2};
  mapping.is_prefill = true;

  std::vector<int64_t> lora_index_to_id = {1, 2};
  wrapper.UpdateMetadata(mapping, lora_index_to_id, /*max_loras=*/4,
                         /*vocab_size=*/512);

  CHECK(wrapper.is_prefill());

  // compute_meta should have clustered into 2 segments.
  CHECK(wrapper.batch_size() == 2);
  CHECK(wrapper.token_nums() == 5);
  CHECK(wrapper.max_length() == 3);

  // sgmv metadata
  CHECK(wrapper.seq_start_locs()[0] == 0);
  CHECK(wrapper.seq_start_locs()[1] == 3);
  CHECK(wrapper.seq_lengths()[0] == 3);
  CHECK(wrapper.seq_lengths()[1] == 2);
  // lora_indices_per_batch holds the SLOT indices (from token_lora_indices),
  // not the original LoRA ids. id 1 → slot 0, id 2 → slot 1.
  CHECK(wrapper.lora_indices_per_batch()[0] == 0);
  CHECK(wrapper.lora_indices_per_batch()[1] == 1);

  CHECK_FALSE(wrapper.no_lora());
}

TEST_CASE("PunicaWrapperBase: prefill with no_lora") {
  // Entire batch is -1 (no adapter): no_lora should be true.
  PunicaWrapperBase wrapper(/*max_num_batched_tokens=*/16, /*max_batches=*/8);

  LoRAMapping mapping;
  mapping.index_mapping = {-1, -1, -1, -1};
  mapping.prompt_mapping = {0};
  mapping.is_prefill = true;

  std::vector<int64_t> lora_index_to_id = {1, 2};
  wrapper.UpdateMetadata(mapping, lora_index_to_id, /*max_loras=*/4,
                         /*vocab_size=*/512);

  CHECK(wrapper.is_prefill());
  CHECK(wrapper.no_lora());
  CHECK(wrapper.batch_size() == 1);
}

TEST_CASE("PunicaWrapperBase: second update overwrites first") {
  // Calling update_metadata twice should fully overwrite the previous state.
  PunicaWrapperBase wrapper(/*max_num_batched_tokens=*/16, /*max_batches=*/8);

  // First: prefill with 2 segments.
  LoRAMapping m1;
  m1.index_mapping = {1, 1, 2, 2};
  m1.prompt_mapping = {1, 2};
  m1.is_prefill = true;
  wrapper.UpdateMetadata(m1, {1, 2}, 4, 512);

  CHECK(wrapper.is_prefill());
  CHECK(wrapper.batch_size() == 2);

  // Second: decode with 1 token.
  LoRAMapping m2;
  m2.index_mapping = {1};
  m2.prompt_mapping = {1};
  m2.is_prefill = false;
  wrapper.UpdateMetadata(m2, {1, 2}, 4, 512);

  CHECK_FALSE(wrapper.is_prefill());
  CHECK(wrapper.token_lora_indices_len() == 1);
  // batch_size stays at its previous value for decode (not reset).
  // This mirrors vLLM: _update_prefill_metadata is only called for prefill.
}

TEST_CASE("PunicaWrapperBase: prefill with all-same-adapter") {
  // All tokens use the same adapter: one segment.
  PunicaWrapperBase wrapper(/*max_num_batched_tokens=*/16, /*max_batches=*/8);

  LoRAMapping mapping;
  mapping.index_mapping = {1, 1, 1, 1, 1, 1};
  mapping.prompt_mapping = {1};
  mapping.is_prefill = true;

  wrapper.UpdateMetadata(mapping, {1, 2}, 4, 512);

  CHECK(wrapper.batch_size() == 1);
  CHECK(wrapper.max_length() == 6);
  CHECK(wrapper.token_nums() == 6);
  CHECK_FALSE(wrapper.no_lora());  // not -1
}

TEST_CASE("PunicaWrapperBase: embeddings_indices accessor returns 2-row layout") {
  // Verify the flat [2, max_num_batched_tokens] layout: row 0 at
  // [0, max), row 1 at [max, 2*max).
  PunicaWrapperBase wrapper(/*max_num_batched_tokens=*/8, /*max_batches=*/4);

  LoRAMapping mapping;
  mapping.index_mapping = {1, 2};
  mapping.prompt_mapping = {1, 2};
  mapping.is_prefill = false;

  wrapper.UpdateMetadata(mapping, {1, 2}, 4, 512);

  // With extra_vocab_size=0: row 0 = all zeros, row 1 = slot * vocab_size.
  // batch_size = 2, so indices_len[3] = 2.
  CHECK(wrapper.embeddings_indices_len() == 2);
  // Row 0: [0, 0]
  CHECK(wrapper.embeddings_indices()[0] == 0);
  CHECK(wrapper.embeddings_indices()[1] == 0);
  // Row 1: [0*512, 1*512] = [0, 512]
  CHECK(wrapper.embeddings_indices()[8] == 0);    // row 1, col 0
  CHECK(wrapper.embeddings_indices()[9] == 512);  // row 1, col 1
}
