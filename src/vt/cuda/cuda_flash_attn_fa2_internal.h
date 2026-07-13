// Internal lifecycle and test hooks for the torch-free FlashAttention-2 adapter.
// Not installed; the public surface remains vt::PagedAttention.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vt::cuda {

// Called by CudaBackend before destroying a stream. FA2 decode scratch is held
// until queue teardown because live CUDA graph executables retain its pointers.
void ReleaseFa2Scratch(int device, void* stream);

namespace testing {

// Exact port of flash_api.cpp::num_splits_heuristic at FA2 2c839c33.
int Fa2DecodeNumSplitsForTesting(int batch_nheads_mblocks, int num_sms,
                                 int num_n_blocks, int max_splits);

void ResetFa2DecodeDebugCounters();
void DisableFa2DecodeDebugCounters();
uint64_t Fa2DecodeLaunchesForTesting();
uint64_t Fa2DecodeSplitLaunchesForTesting();
uint64_t Fa2DecodeNoSplitLaunchesForTesting();
uint64_t Fa2DecodeScratchAllocationsForTesting();
uint64_t Fa2DecodeScratchReusesForTesting();

// Number of capture-stable decode-shape entries owned by one native stream.
// The handle may be retained by a test after DestroyQueue to prove cleanup.
size_t Fa2DecodeScratchShapeCountForTesting(int device, void* stream);

}  // namespace testing
}  // namespace vt::cuda
