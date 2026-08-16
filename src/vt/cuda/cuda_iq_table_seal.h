// Device-codebook drift seal. `cuda_quant_iq_tables.cuh` is a HAND TRANSCRIPTION
// of `src/vt/cpu/cpu_quant_iq_tables.h`, and until this header existed nothing
// compared the two: the CPU tests digest the HOST symbols, which the device
// arrays are not, so a slipped literal in the `.cuh` was invisible to every gate
// that did not happen to address that entry. Replaying the CUDA gate's own
// std::mt19937(0x5EED) weight stream, 266 of the 2048 `d_iq1s_grid` entries
// (13.0 %) are never addressed at all, so a drift there is green by luck.
//
// A `__device__` array has no host address, so a plain C++ translation unit
// cannot take its address and cannot call `cudaMemcpyFromSymbol` on it. The copy
// therefore has to happen inside the CUDA TU that defines the tables. This
// header is the CUDA-free declaration of that copy, so the gate in
// `tests/vt/test_cuda_quant_dot.cpp` can memcmp the result against the CPU
// tables without pulling `<cuda_runtime.h>` into a host build.
#ifndef VT_CUDA_IQ_TABLE_SEAL_H_
#define VT_CUDA_IQ_TABLE_SEAL_H_

#include <cstdint>

namespace vt::cuda {

// One host-side copy of every codebook `cuda_quant_iq_tables.cuh` defines. The
// extents are restated here rather than derived, and `cuda_quant_dot.cu`
// static_asserts each one against `sizeof(d_<table>)`, so a device array that
// changes length fails to compile instead of silently truncating the seal.
struct IqTableSnapshot {
  uint8_t kmask_iq2xs[8];
  uint8_t ksigns_iq2xs[128];
  uint64_t iq1s_grid[2048];
  uint64_t iq1xxxs_grid[256];
  uint64_t iq2xxs_grid[256];
  uint32_t iq3xxs_grid[256];
  uint64_t iq2s_grid[1024];
  int8_t kvalues_mxfp4[16];
};

// Copies the device codebooks into `out`. Requires a live CUDA context; throws
// std::runtime_error if any copy fails. Defined in cuda_quant_dot.cu.
void SnapshotIqTablesFromDevice(IqTableSnapshot* out);

}  // namespace vt::cuda

#endif  // VT_CUDA_IQ_TABLE_SEAL_H_
