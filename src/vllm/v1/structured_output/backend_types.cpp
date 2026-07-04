// Ported from: vllm/v1/structured_output/backend_types.py @ e24d1b24
// See include/vllm/v1/structured_output/backend_types.h for the contract.
//
// The enum, the two ABCs, the bitmask value type and the cache key are all
// header-defined; this translation unit exists to compile-check that the header
// is self-contained. The concrete native backend is M3.4 Task 4.
#include "vllm/v1/structured_output/backend_types.h"
