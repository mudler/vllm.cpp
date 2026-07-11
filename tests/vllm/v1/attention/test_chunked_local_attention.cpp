// Ported from:
//   vllm/tests/v1/attention/test_chunked_local_attention.py:1-204
//   vllm/model_executor/layers/attention/chunked_local_attention.py:30-128
// @ e24d1b24
//
// All six upstream virtual-batch Q/K/block-table vectors are exact. Additional
// deterministic properties check the reusable gather plan, chunk-aligned
// causal-mask equivalence, builder delegation, cudagraph rejection, cached
// backend wrapping and ChunkedLocalAttentionSpec emission. Host vectors stand
// in for upstream's eagerly uploaded device index tensors per backend.h.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/layers/attention/chunked_local_attention.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/utils.h"
#include "vt/dtype.h"

using vllm::ChunkedLocalAttention;
using vllm::ChunkedLocalAttentionBuilder;
using vllm::CreateChunkedLocalAttentionBackend;
using vllm::v1::AttentionCGSupport;
using vllm::v1::AttentionBackend;
using vllm::v1::AttentionImpl;
using vllm::v1::AttentionLayer;
using vllm::v1::AttentionMetadata;
using vllm::v1::AttentionMetadataBuilder;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::FlashAttentionBackend;
using vllm::v1::KVCacheSpecKind;
using vllm::v1::LocalAttentionVirtualBatches;
using vllm::v1::MakeLocalAttentionVirtualBatches;

namespace {

int Cdiv(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

CommonAttentionMetadata MakeMetadata(const std::vector<int32_t>& query_lens,
                                     const std::vector<int32_t>& seq_lens,
                                     int block_size) {
  if (query_lens.size() != seq_lens.size() || query_lens.empty()) {
    throw std::invalid_argument("invalid test metadata shape");
  }
  CommonAttentionMetadata metadata;
  metadata.query_start_loc_cpu = {0};
  int32_t total_query_tokens = 0;
  for (int32_t query_len : query_lens) {
    total_query_tokens += query_len;
    metadata.query_start_loc_cpu.push_back(total_query_tokens);
  }
  metadata.query_start_loc = metadata.query_start_loc_cpu;
  metadata.seq_lens_cpu = seq_lens;
  metadata.seq_lens = seq_lens;
  metadata.num_reqs = static_cast<int>(seq_lens.size());
  metadata.num_actual_tokens = total_query_tokens;
  metadata.max_query_len =
      *std::max_element(query_lens.begin(), query_lens.end());
  metadata.max_seq_len =
      *std::max_element(seq_lens.begin(), seq_lens.end());
  metadata.num_computed_tokens_cpu.reserve(seq_lens.size());
  for (size_t request = 0; request < seq_lens.size(); ++request) {
    metadata.num_computed_tokens_cpu.push_back(
        seq_lens[request] - query_lens[request]);
  }

  metadata.block_table_num_cols = Cdiv(metadata.max_seq_len, block_size);
  for (int request = 0; request < metadata.num_reqs; ++request) {
    for (int column = 0; column < metadata.block_table_num_cols; ++column) {
      metadata.block_table_tensor.push_back(
          request * metadata.block_table_num_cols + column);
    }
  }
  metadata.slot_mapping.resize(static_cast<size_t>(total_query_tokens));
  std::iota(metadata.slot_mapping.begin(), metadata.slot_mapping.end(), 100);
  metadata.causal = false;
  return metadata;
}

std::vector<int32_t> Flatten(
    const std::vector<std::vector<int32_t>>& rows) {
  std::vector<int32_t> flat;
  for (const auto& row : rows) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

struct LocalAttentionTestData {
  std::vector<int32_t> query_lens;
  std::vector<int32_t> seq_lens;
  int attention_chunk_size;
  int block_size;
  std::vector<int32_t> expected_query_lens;
  std::vector<int32_t> expected_key_lens;
  std::vector<std::vector<int32_t>> expected_block_table;
};

std::vector<LocalAttentionTestData> UpstreamCases() {
  return {
      {{4, 10, 5},
       {6, 17, 9},
       4,
       2,
       {2, 2, 1, 4, 4, 1, 4, 1},
       {4, 2, 4, 4, 4, 1, 4, 1},
       {{0, 1}, {2, 3}, {11, 12}, {13, 14},
        {15, 16}, {17, 17}, {20, 21}, {22, 23}}},
      {{8}, {12}, 4, 2, {4, 4}, {4, 4}, {{2, 3}, {4, 5}}},
      {{7}, {10}, 4, 2, {1, 4, 2}, {4, 4, 2}, {{0, 1}, {2, 3}, {4, 4}}},
      {{4}, {6}, 10, 2, {4}, {6}, {{0, 1, 2, 2, 2}}},
      {{6, 6}, {8, 8}, 4, 4, {2, 4, 2, 4}, {4, 4, 4, 4},
       {{0}, {1}, {2}, {3}}},
      {{1}, {5}, 4, 2, {1}, {1}, {{2, 2}}},
  };
}

double QueryValue(int request, int position) {
  return 0.03 * static_cast<double>(1 + request) +
         0.01 * static_cast<double>(1 + position);
}

double KeyValue(int request, int position) {
  return -0.02 * static_cast<double>(1 + request) +
         0.015 * static_cast<double>(1 + position);
}

double ValueValue(int request, int position) {
  return 0.2 * static_cast<double>(request) +
         std::sin(0.1 * static_cast<double>(position));
}

double CausalAttentionReference(int request, int query_position,
                                int key_start) {
  const double query = QueryValue(request, query_position);
  double max_score = -1.0e300;
  for (int key_position = key_start; key_position <= query_position;
       ++key_position) {
    max_score = std::max(
        max_score, query * KeyValue(request, key_position));
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (int key_position = key_start; key_position <= query_position;
       ++key_position) {
    const double weight =
        std::exp(query * KeyValue(request, key_position) - max_score);
    denominator += weight;
    numerator += weight * ValueValue(request, key_position);
  }
  return numerator / denominator;
}

struct RecordingMetadata : AttentionMetadata {
  CommonAttentionMetadata common;
  int common_prefix_len = -1;
  bool fast_build = false;
  std::vector<int32_t> updated_block_table;
  int updated_block_table_num_cols = 0;
  std::vector<int64_t> updated_slot_mapping;
};

class RecordingBuilder final
    : public AttentionMetadataBuilder<RecordingMetadata> {
 public:
  RecordingMetadata build(
      int common_prefix_len,
      const CommonAttentionMetadata& common_attn_metadata,
      bool fast_build = false) override {
    RecordingMetadata metadata;
    metadata.common = common_attn_metadata;
    metadata.common_prefix_len = common_prefix_len;
    metadata.fast_build = fast_build;
    return metadata;
  }

  void update_block_table(RecordingMetadata& metadata,
                          const std::vector<int32_t>& block_table,
                          int block_table_num_cols,
                          const std::vector<int64_t>& slot_mapping) {
    metadata.updated_block_table = block_table;
    metadata.updated_block_table_num_cols = block_table_num_cols;
    metadata.updated_slot_mapping = slot_mapping;
  }
};

class RecordingAttentionImpl final : public AttentionImpl {
 public:
  void forward(const AttentionLayer&, const vt::Tensor&, const vt::Tensor&,
               const vt::Tensor&, const vt::Tensor&,
               const AttentionMetadata&, vt::Tensor& output,
               const vt::Tensor*, const vt::Tensor*) override {
    *static_cast<float*>(output.data) = 42.0f;
  }
};

class RecordingAttentionBackend final : public AttentionBackend {
 public:
  std::string get_name() const override { return "RECORDING_ATTN"; }

  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size, const std::string&) const override {
    return {num_blocks, 2, block_size, num_kv_heads, head_size};
  }

  std::unique_ptr<AttentionImpl> get_impl_cls() const override {
    return std::make_unique<RecordingAttentionImpl>();
  }
};

}  // namespace

TEST_CASE("make_local_attention_virtual_batches: six pinned vectors") {
  const auto cases = UpstreamCases();
  for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
    CAPTURE(case_index);
    const auto& data = cases[case_index];
    const CommonAttentionMetadata common =
        MakeMetadata(data.query_lens, data.seq_lens, data.block_size);
    const LocalAttentionVirtualBatches result =
        MakeLocalAttentionVirtualBatches(
            data.attention_chunk_size, common, data.block_size);
    const CommonAttentionMetadata& local = result.common_attn_metadata;
    const std::vector<int32_t> actual_query_lens = local.naive_query_lens();

    CHECK(actual_query_lens == data.expected_query_lens);
    CHECK(local.seq_lens_cpu == data.expected_key_lens);
    CHECK(local.block_table_tensor == Flatten(data.expected_block_table));
    CHECK(local.block_table_num_cols ==
          data.attention_chunk_size / data.block_size);
    CHECK(local.num_actual_tokens == common.num_actual_tokens);
    CHECK(local.slot_mapping == common.slot_mapping);
    CHECK(local.causal == true);
    CHECK(std::accumulate(actual_query_lens.begin(),
                          actual_query_lens.end(), 0) ==
          std::accumulate(data.query_lens.begin(), data.query_lens.end(), 0));
    for (int32_t query_len : actual_query_lens) {
      CHECK(query_len <= data.attention_chunk_size);
    }
    for (int32_t key_len : local.seq_lens_cpu) {
      CHECK(key_len <= data.attention_chunk_size);
    }

    std::vector<int32_t> replacement = common.block_table_tensor;
    for (int32_t& block : replacement) block += 1000;
    std::vector<int32_t> expected_replacement = local.block_table_tensor;
    for (int32_t& block : expected_replacement) block += 1000;
    CHECK(result.make_virtual_batches_block_table(
              replacement, common.block_table_num_cols) ==
          expected_replacement);
  }
}

TEST_CASE("chunked-local virtual batches: deterministic mask and gather properties") {
  std::mt19937 rng(0xc4a117u);
  for (int trial = 0; trial < 100; ++trial) {
    CAPTURE(trial);
    const int block_size = 1 << (trial % 4);
    const int attention_chunk_size = block_size * (1 + trial % 6);
    const int batch_size = 1 + trial % 5;
    std::vector<int32_t> query_lens;
    std::vector<int32_t> seq_lens;
    for (int request = 0; request < batch_size; ++request) {
      const int query_len =
          1 + static_cast<int>(rng() % (3 * attention_chunk_size));
      const int computed =
          static_cast<int>(rng() % (4 * attention_chunk_size + 1));
      query_lens.push_back(query_len);
      seq_lens.push_back(query_len + computed);
    }
    const CommonAttentionMetadata common =
        MakeMetadata(query_lens, seq_lens, block_size);
    const LocalAttentionVirtualBatches result =
        MakeLocalAttentionVirtualBatches(
            attention_chunk_size, common, block_size);
    const auto& local = result.common_attn_metadata;
    const std::vector<int32_t> local_query_lens = local.naive_query_lens();
    REQUIRE(local_query_lens.size() ==
            result.virtual_batch_request_indices.size());
    REQUIRE(local_query_lens.size() == result.k_seqstarts_absolute.size());

    std::vector<int> next_query_position(static_cast<size_t>(batch_size));
    for (int request = 0; request < batch_size; ++request) {
      next_query_position[static_cast<size_t>(request)] =
          seq_lens[static_cast<size_t>(request)] -
          query_lens[static_cast<size_t>(request)];
    }

    for (size_t virtual_batch = 0; virtual_batch < local_query_lens.size();
         ++virtual_batch) {
      const int request =
          result.virtual_batch_request_indices[virtual_batch];
      const int query_len = local_query_lens[virtual_batch];
      const int key_len = local.seq_lens_cpu[virtual_batch];
      const int computed = key_len - query_len;
      const int key_start = result.k_seqstarts_absolute[virtual_batch];
      CHECK(query_len > 0);
      CHECK(query_len <= attention_chunk_size);
      CHECK(key_len <= attention_chunk_size);
      CHECK(key_start % attention_chunk_size == 0);

      for (int query_index = 0; query_index < query_len; ++query_index) {
        const int query_position = key_start + computed + query_index;
        CHECK(query_position ==
              next_query_position[static_cast<size_t>(request)]++);
        const int dense_local_start =
            query_position / attention_chunk_size * attention_chunk_size;
        CHECK(CausalAttentionReference(request, query_position,
                                       dense_local_start) ==
              doctest::Approx(CausalAttentionReference(
                  request, query_position, key_start)));
      }

      for (int page = 0; page < result.pages_per_local_batch; ++page) {
        const int expected_column = std::min(
            key_start / block_size + page,
            common.block_table_num_cols - 1);
        const int expected_block =
            request * common.block_table_num_cols + expected_column;
        const size_t local_index =
            virtual_batch *
                static_cast<size_t>(result.pages_per_local_batch) +
            static_cast<size_t>(page);
        CHECK(local.block_table_tensor[local_index] == expected_block);
      }
    }
    CHECK(next_query_position ==
          std::vector<int>(seq_lens.begin(), seq_lens.end()));
  }
}

TEST_CASE("ChunkedLocalAttentionBuilder delegates build and block-table update") {
  const CommonAttentionMetadata common = MakeMetadata({4, 10, 5}, {6, 17, 9}, 2);
  RecordingBuilder underlying;
  ChunkedLocalAttentionBuilder<RecordingMetadata, RecordingBuilder> builder(
      /*attention_chunk_size=*/4, /*block_size=*/2, underlying);
  CHECK(builder.get_cudagraph_support() == AttentionCGSupport::kNever);

  auto built = builder.build(/*common_prefix_len=*/7, common,
                             /*fast_build=*/true);
  CHECK(built.metadata.common_prefix_len == 7);
  CHECK(built.metadata.fast_build == true);
  CHECK(built.metadata.common.naive_query_lens() ==
        std::vector<int32_t>{2, 2, 1, 4, 4, 1, 4, 1});

  std::vector<int32_t> replacement = common.block_table_tensor;
  for (int32_t& block : replacement) block += 500;
  const std::vector<int64_t> replacement_slots = {9, 8, 7};
  builder.update_block_table(built, replacement,
                             common.block_table_num_cols,
                             replacement_slots);
  CHECK(built.metadata.updated_block_table ==
        built.virtual_batches.make_virtual_batches_block_table(
            replacement, common.block_table_num_cols));
  CHECK(built.metadata.updated_block_table_num_cols == 2);
  CHECK(built.metadata.updated_slot_mapping == replacement_slots);
}

TEST_CASE("ChunkedLocalAttention backend cache, delegation, and spec emission") {
  auto underlying = std::make_shared<FlashAttentionBackend>();
  auto backend_a = CreateChunkedLocalAttentionBackend(underlying, 32);
  auto backend_b = CreateChunkedLocalAttentionBackend(
      std::make_shared<FlashAttentionBackend>(), 32);
  auto backend_other_chunk =
      CreateChunkedLocalAttentionBackend(underlying, 64);
  CHECK(backend_a == backend_b);
  CHECK(backend_a != backend_other_chunk);
  CHECK(backend_a->get_name() == "ChunkedLocalAttention_32_FLASH_ATTN");
  CHECK(backend_a->get_kv_cache_shape(10, 16, 2, 128) ==
        std::vector<int64_t>{10, 2, 16, 2, 128});
  CHECK(backend_a->get_impl_cls() == nullptr);

  auto recording_backend = CreateChunkedLocalAttentionBackend(
      std::make_shared<RecordingAttentionBackend>(), 32);
  auto recording_impl = recording_backend->get_impl_cls();
  REQUIRE(dynamic_cast<RecordingAttentionImpl*>(recording_impl.get()) !=
          nullptr);
  float output_value = 0.0f;
  vt::Tensor empty;
  vt::Tensor output;
  output.data = &output_value;
  AttentionLayer attention_layer;
  AttentionMetadata attention_metadata;
  recording_impl->forward(attention_layer, empty, empty, empty, empty,
                          attention_metadata, output);
  CHECK(output_value == 42.0f);

  ChunkedLocalAttention layer(
      /*num_heads=*/8, /*head_size=*/128, /*scale=*/0.125f,
      /*attention_chunk_size=*/32, /*num_kv_heads=*/2, underlying);
  CHECK(layer.backend == backend_a);
  auto spec = layer.get_kv_cache_spec(
      /*block_size=*/16, vt::DType::kBF16,
      vllm::v1::KVQuantMode::kNone, /*page_size_padded=*/65536,
      /*indexes_kv_by_block_stride=*/true);
  CHECK(spec->kind() == KVCacheSpecKind::kChunkedLocalAttention);
  CHECK(spec->block_size == 16);
  CHECK(spec->num_kv_heads == 2);
  CHECK(spec->head_size == 128);
  CHECK(spec->attention_chunk_size == 32);
  CHECK(spec->page_size_padded == std::optional<int64_t>{65536});
  CHECK(spec->indexes_kv_by_block_stride == true);
}

TEST_CASE("chunked-local metadata rejects invalid shapes and chunk alignment") {
  const CommonAttentionMetadata common = MakeMetadata({4}, {6}, 2);
  CHECK_THROWS_AS(MakeLocalAttentionVirtualBatches(3, common, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(MakeLocalAttentionVirtualBatches(0, common, 2),
                  std::invalid_argument);

  CommonAttentionMetadata empty_query = common;
  empty_query.query_start_loc = {0, 0};
  empty_query.query_start_loc_cpu = {0, 0};
  empty_query.num_actual_tokens = 0;
  CHECK_THROWS_AS(MakeLocalAttentionVirtualBatches(4, empty_query, 2),
                  std::invalid_argument);

  CHECK_THROWS_AS(ChunkedLocalAttention(
                      8, 128, 0.125f, 0, 2,
                      std::make_shared<FlashAttentionBackend>()),
                  std::invalid_argument);
}

TEST_CASE("chunked-local positive model gate waits on Llama4" *
          doctest::skip(true) *
          doctest::description(
              "MODEL-TEXT-llama4-llama4-for-causal-lm is not ported")) {
  MESSAGE("SKIP MODEL-TEXT-llama4-llama4-for-causal-lm");
}
