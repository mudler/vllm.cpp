// Tests for the M1.6 Task 4 GDN attention metadata — the prefill/decode
// segmentation GDNAttentionMetadataBuilder emits so the M0.7 GDN ops
// (vt::GdnPrefill / vt::GdnDecode) can be driven from a batched step.
//
// Ported from vllm/v1/attention/backends/gdn_attn.py @ e24d1b24 (the non-spec
// build path) + the classification intent of
// tests/v1/attention/test_gdn_metadata_builder.py (the T0-relevant, non-spec
// cases; spec-decode cases are DEFERRED — T0 gate models don't spec-decode).
//
// The builder ASSUMES a decode-first-reordered batch (the runner reorders on
// reorder_batch_threshold=1); every batch below is provided decode-first, as
// upstream split_decodes_and_prefills requires.
#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionBackend;
using vllm::v1::GDNAttentionMetadata;
using vllm::v1::GDNAttentionMetadataBuilder;
using vllm::v1::MakeCommonAttentionMetadata;
using vllm::v1::SplitDecodesAndPrefills;
using vllm::v1::StepInputs;

namespace {

// Build a CommonAttentionMetadata from explicit (seq_len, query_len) pairs, in
// the given order. block_table has one column: block id = 100 + req index, so
// the state index of req r is 100 + r (easy to assert).
CommonAttentionMetadata make_cam(const std::vector<int32_t>& seq_lens,
                                 const std::vector<int32_t>& query_lens) {
  StepInputs step;
  step.seq_lens = seq_lens;
  step.query_start_loc = {0};
  int32_t total = 0;
  for (const int32_t q : query_lens) {
    total += q;
    step.query_start_loc.push_back(total);
  }
  step.num_scheduled_tokens = query_lens;
  step.slot_mapping = {std::vector<int64_t>(static_cast<size_t>(total), 0)};

  std::vector<int32_t> block_table_flat;
  for (size_t r = 0; r < seq_lens.size(); ++r) {
    block_table_flat.push_back(100 + static_cast<int32_t>(r));
  }
  return MakeCommonAttentionMetadata(step, block_table_flat,
                                     /*block_table_num_cols=*/1);
}

}  // namespace

// Core case from the M1.6 Task 4 brief: a mixed batch with 1 decode + 1 prefill,
// provided decode-first (the ordering the builder assumes). The decode req has a
// 6-token context with 5 already computed (query_len 1 ⇒ has_initial_state
// true); the prefill req is a fresh 4-token sequence (num_computed 0 ⇒
// has_initial_state false).
TEST_CASE("GDN build: mixed decode + prefill (decode-first)") {
  // req0: decode  seq_len 6, query_len 1  (computed 5)
  // req1: prefill seq_len 4, query_len 4  (computed 0)
  const CommonAttentionMetadata m = make_cam({6, 4}, {1, 4});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(/*common_prefix_len=*/0, m);

  CHECK(meta.num_decodes == 1);
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_decode_tokens == 1);
  CHECK(meta.num_prefill_tokens == 4);
  CHECK(meta.num_spec_decodes == 0);
  CHECK(meta.num_spec_decode_tokens == 0);
  CHECK(meta.num_actual_tokens == 5);

  // has_initial_state (full batch, decode-first): [decode computed>0=1,
  // prefill computed0=0].
  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{1, 0});

  // Non-spec state indices = block_table col 0 = {100, 101}.
  REQUIRE(meta.non_spec_state_indices_tensor.has_value());
  CHECK(*meta.non_spec_state_indices_tensor == std::vector<int32_t>{100, 101});
  REQUIRE(meta.non_spec_query_start_loc.has_value());
  CHECK(*meta.non_spec_query_start_loc == std::vector<int32_t>{0, 1, 5});

  // Prefill sub-batch = leading decode peeled off + rebased by num_decode_tokens.
  // non_spec_query_start_loc = {0,1,5}; [num_decodes=1:] = {1,5}; -1 = {0,4}.
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 4});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{101});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{0});
}

// Decode-only batch: all query_len==1. num_prefills==0 ⇒ has_initial_state and
// all prefill_* fields are None (gdn_attn.py:405) — the decode kernel reads the
// state via state_indices and needs no has_initial_state mask.
TEST_CASE("GDN build: decodes only") {
  // Mirrors upstream test case "pure_regular_decode".
  const CommonAttentionMetadata m = make_cam({40, 30, 20}, {1, 1, 1});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 3);
  CHECK(meta.num_prefills == 0);
  CHECK(meta.num_decode_tokens == 3);
  CHECK(meta.num_prefill_tokens == 0);
  CHECK(meta.num_spec_decodes == 0);

  CHECK_FALSE(meta.has_initial_state.has_value());
  CHECK_FALSE(meta.prefill_query_start_loc.has_value());
  CHECK_FALSE(meta.prefill_state_indices.has_value());
  CHECK_FALSE(meta.prefill_has_initial_state.has_value());

  // Non-spec state indices are still emitted for the decode kernel.
  REQUIRE(meta.non_spec_state_indices_tensor.has_value());
  CHECK(*meta.non_spec_state_indices_tensor == std::vector<int32_t>{100, 101, 102});
}

// Prefills-only batch of two FRESH sequences (num_computed 0 for both). No decode
// to peel: prefill_query_start_loc == non_spec_query_start_loc, and
// has_initial_state is all-false.
TEST_CASE("GDN build: prefills only (fresh, no initial state)") {
  const CommonAttentionMetadata m = make_cam({4, 7}, {4, 7});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_decode_tokens == 0);
  CHECK(meta.num_prefill_tokens == 11);

  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{0, 0});
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 4, 11});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{100, 101});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{0, 0});
}

// Prefills-only batch of two CONTINUING sequences (chunked prefill: num_computed
// > 0 for both). has_initial_state is all-true — the GDN op must start from the
// carried state.
TEST_CASE("GDN build: prefills only (continuing, has initial state)") {
  // req0: seq_len 10, query_len 4 (computed 6); req1: seq_len 20, query_len 7
  // (computed 13).
  const CommonAttentionMetadata m = make_cam({10, 20}, {4, 7});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_decodes == 0);
  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{1, 1});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{1, 1});
}

// Multiple leading decodes + one prefill: all leading query_len==1 requests are
// decodes; the prefill tail is peeled and rebased.
TEST_CASE("GDN build: multiple decodes + one prefill") {
  // req0,1,2: decodes (query_len 1); req3: fresh prefill seq_len 5, query_len 5
  // (computed 0).
  const CommonAttentionMetadata m = make_cam({30, 40, 50, 5}, {1, 1, 1, 5});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 3);
  CHECK(meta.num_prefills == 1);
  CHECK(meta.num_decode_tokens == 3);
  CHECK(meta.num_prefill_tokens == 5);

  // has_initial_state: decodes computed>0 (29,39,49) true; prefill fresh (0)
  // false.
  REQUIRE(meta.has_initial_state.has_value());
  CHECK(*meta.has_initial_state == std::vector<uint8_t>{1, 1, 1, 0});

  // Prefill sub-batch: non_spec qsl {0,1,2,3,8}; [3:] = {3,8}; - 3 = {0,5}.
  REQUIRE(meta.prefill_query_start_loc.has_value());
  CHECK(*meta.prefill_query_start_loc == std::vector<int32_t>{0, 5});
  REQUIRE(meta.prefill_state_indices.has_value());
  CHECK(*meta.prefill_state_indices == std::vector<int32_t>{103});
  REQUIRE(meta.prefill_has_initial_state.has_value());
  CHECK(*meta.prefill_has_initial_state == std::vector<uint8_t>{0});
}

// Decode-first ORDERING CONTRACT: split_decodes_and_prefills is order-sensitive.
// A batch whose FIRST request is a prefill (query_len > threshold) is classified
// as all-prefills, even if a query_len==1 request follows (utils.py:607-609).
// This documents why the runner must reorder decodes to the front.
TEST_CASE("GDN build: prefill-first is all-prefills (ordering contract)") {
  // req0: prefill (query_len 4), req1: query_len 1 AFTER it.
  const CommonAttentionMetadata m = make_cam({4, 6}, {4, 1});
  GDNAttentionMetadataBuilder builder;
  const GDNAttentionMetadata meta = builder.build(0, m);

  CHECK(meta.num_decodes == 0);
  CHECK(meta.num_prefills == 2);
  CHECK(meta.num_decode_tokens == 0);
  CHECK(meta.num_prefill_tokens == 5);
}

TEST_CASE("SplitDecodesAndPrefills: all-decode fast path") {
  const CommonAttentionMetadata m = make_cam({5, 6}, {1, 1});
  const auto [nd, np, ndt, npt] = SplitDecodesAndPrefills(m, 1);
  CHECK(nd == 2);
  CHECK(np == 0);
  CHECK(ndt == 2);
  CHECK(npt == 0);
}

TEST_CASE("GDNAttentionBackend: name / ssm / no kv-cache shape") {
  const GDNAttentionBackend backend;
  CHECK(backend.get_name() == "GDN_ATTN");
  CHECK(GDNAttentionBackend::is_ssm() == true);
  CHECK_THROWS_AS(backend.get_kv_cache_shape(1, 16, 2, 128), std::logic_error);
}
