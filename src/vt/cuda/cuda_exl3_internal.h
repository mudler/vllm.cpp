// Internal lifecycle and test hooks for the EXL3 CUDA kernels. Not installed; the
// public surface remains vt::Exl3ReconstructGemm.
#pragma once

#include <cstddef>

namespace vt::cuda {

// QUANT-EXL3 W7 (.agents/specs/quant-exl3-recon-scratch.md). Called by CudaBackend
// before destroying a stream. Synchronizes the stream, then frees this
// (device, stream)'s persistent reconstruct scratch, or retires it when a capture
// was handed the block, and erases the entry, so a recycled stream handle never
// inherits it.
void ReleaseExl3ReconScratch(int device, void* stream);

namespace testing {

// Capacity in bytes of one (device, stream)'s persistent reconstruct scratch;
// 0 when the stream has no entry. The handle may be retained by a test after
// DestroyQueue to prove cleanup.
size_t Exl3ReconScratchBytesForTesting(int device, void* stream);
// The entry's current block, or nullptr when the stream has no entry.
void* Exl3ReconScratchPtrForTesting(int device, void* stream);
// Sum of the capacities of every live entry.
size_t Exl3ReconScratchLiveBytesForTesting();
// graph_safe_scratch.h RetiredGraphScratchCount, read inside the library.
size_t RetiredGraphScratchCountForTesting();

}  // namespace testing
}  // namespace vt::cuda
