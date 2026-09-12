#!/usr/bin/env python3
"""Compile the production ROCm sampling dispatcher with a fake HIP runtime.

Only launch syntax is adapted, preserving its dimensions and stream. These
tests check allocation ownership and launch contracts, not GPU arithmetic.
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]

PRELUDE = r"""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "vt/grow_only_stream_scratch.h"
using hipStream_t = void*;
struct Device { int index; };
struct Queue { Device device; void* handle; uint64_t id; };
struct Tensor {
  int64_t shape[2];
  template <typename T> T* Ptr() const { return nullptr; }
};
constexpr int kSampleSplitBlocks = 128, kVocabBlock = 1024;
bool FastRandomSampleEnabled() { return true; }
bool SampleSplitEnabled() { return true; }
hipStream_t AsStream(const Queue& q) { return q.handle; }
void Check(int result, const char* what) {
  if (result != 0) throw std::runtime_error(what);
}
int hipGetLastError() { return 0; }
enum hipStreamCaptureStatus {
  hipStreamCaptureStatusNone, hipStreamCaptureStatusActive,
  hipStreamCaptureStatusInvalidated
};
hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
int capture_query_error = 0;
size_t capture_queries = 0;
int hipStreamIsCapturing(hipStream_t, hipStreamCaptureStatus* status) {
  ++capture_queries;
  *status = capture_status;
  return capture_query_error;
}
std::vector<void*> allocations;
std::vector<size_t> allocation_sizes;
int hipMallocAsync(void** ptr, size_t bytes, hipStream_t) {
  *ptr = std::malloc(bytes);
  assert(*ptr);
  allocations.push_back(*ptr);
  allocation_sizes.push_back(bytes);
  return 0;
}
struct Launch { float* score; int64_t* index; };
std::vector<Launch> phase_a, phase_b;
struct LaunchConfig {
  std::string kernel;
  size_t grid, block, shared;
  hipStream_t stream;
} config;
void RecordLaunch(const char* kernel, size_t grid, size_t block, size_t shared,
                  hipStream_t stream) {
  config = {kernel, grid, block, shared, stream};
}
size_t expected_rows = 0;
hipStream_t expected_stream = nullptr;
void RandomSampleSplitAK(float* score, int64_t* index, float*, int64_t*, int64_t, int partials) {
  assert(config.kernel == "RandomSampleSplitAK");
  assert(config.grid == expected_rows * partials);
  assert(config.stream == expected_stream);
  phase_a.push_back({score, index});
}
void RandomSampleSplitBK(int64_t*, float* score, int64_t* index, int partials) {
  assert(config.kernel == "RandomSampleSplitBK");
  assert(config.grid == expected_rows);
  assert(config.stream == expected_stream);
  // Every thread writes both shared arrays, and every partial needs a lane.
  // #3022's 1024-thread launch overruns their 128-element capacity.
  assert(config.block <= kSampleSplitBlocks);
  assert(config.block == static_cast<size_t>(partials));
  phase_b.push_back({score, index});
}
void RandomSampleKernelSlow(int64_t*, float*, int64_t*, int64_t) { assert(false); }
void RandomSampleK(int64_t*, float*, int64_t*, int64_t) { assert(false); }
"""

MAIN = r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  Tensor out{}, seeds{};
  auto run = [&](Queue& q, int64_t rows) {
    Tensor probs{{rows, 8192}};
    expected_rows = rows;
    expected_stream = q.handle;
    const size_t before = phase_a.size();
    RandomSampleKernelRocm(q, out, probs, seeds);
    assert(phase_a.size() == before + 1);
    assert(phase_b.size() == phase_a.size());
    assert(phase_a.back().score == phase_b.back().score);
    assert(phase_a.back().index == phase_b.back().index);
    assert(reinterpret_cast<uintptr_t>(phase_a.back().index) % alignof(int64_t) == 0);
    return phase_a.back();
  };
  Queue a{{0}, reinterpret_cast<void*>(1), 101};
  Queue b{{0}, reinterpret_cast<void*>(2), 102};
  if (mode == "devices") {
    b.device.index = 1;
    b.handle = a.handle; // Null/default or reused native handles must not alias.
  }
  if (mode == "reuse") b.handle = a.handle;
  if (mode == "capture" || mode == "invalidated" || mode == "query_error") {
    capture_status = mode == "capture" ? hipStreamCaptureStatusActive
                                       : hipStreamCaptureStatusInvalidated;
    if (mode == "query_error") {
      capture_status = hipStreamCaptureStatusNone;
      capture_query_error = 1;
    }
    bool refused = false;
    try {
      run(a, 1);
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      const char* expected = mode == "capture" ? "pre-warm"
                           : mode == "invalidated" ? "invalidated" : "capture query";
      assert(message.find(expected) != std::string::npos);
      refused = true;
    }
    assert(refused);
    assert(allocations.empty());
    assert(phase_a.empty() && phase_b.empty());
    capture_status = hipStreamCaptureStatusNone;
    capture_query_error = 0;
    const auto first = run(a, 1); // Refusal must not poison the pool entry.
    assert(allocations.size() == 1);
    const auto queries = capture_queries;
    capture_status = hipStreamCaptureStatusActive;
    capture_query_error = 1; // A warmed hit must not call the query at all.
    const auto captured = run(a, 64);
    assert(first.score == captured.score && first.index == captured.index);
    assert(capture_queries == queries);
    capture_status = hipStreamCaptureStatusNone;
    capture_query_error = 0;
    const auto eager = run(a, 64);
    assert(first.score == eager.score && first.index == eager.index);
    assert(allocations.size() == 1);
  } else if (mode == "geometry") {
    for (int64_t rows : {1, 8, 64}) run(a, rows);
  } else if (mode == "growth") {
    const auto first = run(a, 1);
    const auto count = allocations.size();
    for (int64_t rows = 2; rows <= 64; ++rows) {
      const auto next = run(a, rows);
      assert(first.score == next.score);
      assert(first.index == next.index);
    }
    assert(allocations.size() == count);
    size_t bytes = 0;
    for (size_t size : allocation_sizes) bytes += size;
    assert(bytes == 64 * 128 * (sizeof(float) + sizeof(int64_t)));
  } else {
    const auto first = run(a, 64);
    const auto second = run(b, 64);
    assert(first.score != second.score);
    assert(first.index != second.index);
    const auto again = run(a, 1);
    assert(first.score == again.score);
    assert(first.index == again.index);
  }
  for (void* allocation : allocations) std::free(allocation);
}
"""


class RocmSampleScratchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="rocm-sample-dispatch-")
        cls.addClassCleanup(cls.temp.cleanup)
        source = (ROOT / "src/vt/rocm/rocm_sample.hip").read_text()
        begin = source.index("void RandomSampleKernelRocm(")
        end = source.index("void ApplyPenaltiesKernelRocm(", begin)
        dispatch = re.sub(
            r"(\w+)<<<(.*?)>>>",
            lambda match: f'RecordLaunch("{match[1]}", {match[2]}), {match[1]}',
            source[begin:end], flags=re.S,
        )
        path = Path(cls.temp.name) / "dispatch.cpp"
        path.write_text(PRELUDE + dispatch + MAIN)
        cls.binary = Path(cls.temp.name) / "dispatch"
        subprocess.run(
            [os.environ.get("CXX", "c++"), "-std=c++17", "-pthread", "-I",
             str(ROOT / "src"), str(path), "-o", str(cls.binary)], check=True,
        )

    def run_case(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_queues_have_independent_partials(self):
        self.run_case("queues")

    def test_devices_have_independent_partials(self):
        self.run_case("devices")

    def test_reused_native_handles_do_not_reuse_scratch(self):
        self.run_case("reuse")

    def test_batch_growth_preserves_captured_pointers_without_more_allocations(self):
        self.run_case("growth")

    def test_cold_capture_refuses_before_allocation_and_recovers(self):
        self.run_case("capture")

    def test_invalidated_capture_refuses_before_allocation_and_recovers(self):
        self.run_case("invalidated")

    def test_capture_query_error_does_not_publish_scratch(self):
        self.run_case("query_error")

    def test_phase_b_launch_fits_shared_arrays_and_covers_every_partial(self):
        self.run_case("geometry")


if __name__ == "__main__":
    unittest.main()
