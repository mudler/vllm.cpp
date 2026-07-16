// Tests for the decode CUDA-graph capture-size set (rescan §5 item d).
//
// Ported oracle: vllm/config/vllm.py::_set_cudagraph_sizes @ e24d1b24 reduced to
// the FULL-decode-cudagraph regime (batches capped at max_num_seqs by the decode
// dispatcher, vllm/config/compilation.py:1438-1444). For max_num_seqs=32 the
// decode capture set is {1,2,4,8,16,24,32}; the old fixed {1,2,4,8,16,32,64}
// MISSED the 24 bucket and over-padded batches 17-31 to 32.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/decode_graph_sizes.h"

using vllm::DecodeGraphSizes;
using vllm::PadToCaptureSize;

TEST_CASE("DecodeGraphSizes(32) == vLLM's {1,2,4,8,16,24,32} (adds the 24 bucket)") {
  const std::vector<int64_t> expected = {1, 2, 4, 8, 16, 24, 32};
  CHECK(DecodeGraphSizes(32) == expected);
  // 64 > max_num_seqs is never captured (the decode dispatcher caps at 32).
}

TEST_CASE("DecodeGraphSizes derives from max_num_seqs (step-8 grid)") {
  CHECK(DecodeGraphSizes(1) == std::vector<int64_t>{1});
  CHECK(DecodeGraphSizes(2) == std::vector<int64_t>{1, 2});
  CHECK(DecodeGraphSizes(4) == std::vector<int64_t>{1, 2, 4});
  CHECK(DecodeGraphSizes(8) == std::vector<int64_t>{1, 2, 4, 8});
  CHECK(DecodeGraphSizes(16) == std::vector<int64_t>{1, 2, 4, 8, 16});
  // max_num_seqs=64 config: the set extends to 64 (vLLM's rule yields it).
  CHECK(DecodeGraphSizes(64) ==
        std::vector<int64_t>{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64});
}

TEST_CASE("DecodeGraphSizes appends an off-stride max_num_seqs (max batch captured)") {
  // 30 is not a multiple of 8; the largest batch must still have a graph.
  CHECK(DecodeGraphSizes(30) == std::vector<int64_t>{1, 2, 4, 8, 16, 24, 30});
}

TEST_CASE("PadToCaptureSize maps a batch to the smallest captured size >= B") {
  // THE FIX: batches 17..24 now pad to the 24 bucket, not 32.
  CHECK(PadToCaptureSize(24, 32) == 24);
  CHECK(PadToCaptureSize(17, 32) == 24);
  CHECK(PadToCaptureSize(23, 32) == 24);
  // 25..32 still pad to 32.
  CHECK(PadToCaptureSize(25, 32) == 32);
  CHECK(PadToCaptureSize(32, 32) == 32);
  // Small buckets unchanged.
  CHECK(PadToCaptureSize(1, 32) == 1);
  CHECK(PadToCaptureSize(3, 32) == 4);
  CHECK(PadToCaptureSize(5, 32) == 8);
  CHECK(PadToCaptureSize(16, 32) == 16);
  // Beyond max_num_seqs runs eager (-1).
  CHECK(PadToCaptureSize(33, 32) == -1);
  // max_num_seqs=64 config.
  CHECK(PadToCaptureSize(33, 64) == 40);
  CHECK(PadToCaptureSize(64, 64) == 64);
}

TEST_CASE("PadToCaptureSize result is always a member of DecodeGraphSizes") {
  for (int64_t mns : {1, 2, 4, 8, 16, 24, 30, 32, 48, 64}) {
    const std::vector<int64_t> set = DecodeGraphSizes(mns);
    for (int64_t b = 1; b <= mns; ++b) {
      const int64_t s = PadToCaptureSize(b, mns);
      CHECK(s >= b);
      const bool in_set = std::find(set.begin(), set.end(), s) != set.end();
      CHECK(in_set);
    }
  }
}
