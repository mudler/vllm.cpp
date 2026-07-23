// CIQ G7 repack-at-load — CPU-internal shared helpers (portable byte shuffles).
//
// The two transforms below are pure BYTE/scale permutations that mirror
// llama.cpp @ 237ad9b96 `make_block_q8_0x4` (repack.cpp:2725) exactly. They
// carry the quant values and fp16 deltas VERBATIM, so a weight repacked by
// RepackQ8_0Rows4 and an activation interleaved by InterleaveQ8_0Rows4 dot to
// the same integer sums as the plain BlockQ8_0 rows they came from — which is
// what lets the repacked GEMM be bit-identical to the tier-0 path. They are
// architecture-independent and always compiled (the round-trip unit test
// exercises them on every CI box); only the GEMM kernels that CONSUME the
// layout are i8mm-gated (cpu_quant_repack_arm.cpp).
#pragma once

#include <cstdint>

#include "cpu_quant_blocks.h"

namespace vt::cpu {

// Interleave FOUR plain q8_0 rows (each `nblocks` contiguous BlockQ8_0) into
// `nblocks` BlockQ8_0x4 written to `dst`. `make_block_q8_0x4(in, 8)` per block:
// out.d[i] = in[i].d; out.qs laid out in 8-byte chunks round-robin across the
// four rows. Used for BOTH the weight repack (rows = 4 weight rows) and the
// activation interleave (rows = 4 activation rows) — the gemm consumes both.
void InterleaveQ8_0Rows4(const BlockQ8_0* row0, const BlockQ8_0* row1,
                         const BlockQ8_0* row2, const BlockQ8_0* row3,
                         BlockQ8_0x4* dst, int64_t nblocks);

}  // namespace vt::cpu
