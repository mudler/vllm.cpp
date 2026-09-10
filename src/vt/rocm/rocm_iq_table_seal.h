// Internal device-table snapshot for the byte-exact I-quant gate (#3067).
// Mirror cuda_iq_table_seal.h without exposing HIP types to the host test.
#ifndef VT_ROCM_IQ_TABLE_SEAL_H_
#define VT_ROCM_IQ_TABLE_SEAL_H_

#include <cstdint>

namespace vt::rocm {

struct IqTableSnapshot {
  uint8_t kmask_iq2xs[8];
  uint8_t ksigns_iq2xs[128];
  uint32_t iq3xxs_grid[256];
  int8_t kvalues_iq4nl[16];
};

// Copy the actual device symbols into out. Requires a live HIP context and
// throws on copy failure. Defined in rocm_quant_dot.hip, the translation unit
// that compiles the `vt::cuda::d_*` tables the ROCm dots actually index, so the
// snapshot reads the device image that executes rather than host literals.
void SnapshotIqTablesFromDevice(IqTableSnapshot* out);

}  // namespace vt::rocm

#endif  // VT_ROCM_IQ_TABLE_SEAL_H_
