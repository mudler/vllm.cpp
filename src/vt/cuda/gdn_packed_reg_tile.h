// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Env-flag plumbing for the register-resident packed-decode tiling A/B, split
// into this pure-C++ (CUDA-free) header so the flag predicate is unit-testable
// on the CPU tier (tests/vt/test_gdn_packed_reg_tile.cpp). The kernel it selects
// is CUDA-only and lives in cuda_gdn.cu (GdnPackedDecodeRegTileKernel), a 1:1
// port of vLLM FLA's num_warps=1 / num_stages=3 register-resident
// fused_recurrent_gated_delta_rule_packed_decode_kernel
// (vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-336, launch
// :448-477 @ 702f4814): one warp owns one [BV,BK] value-tile with the state
// block held in REGISTERS across the recurrence, replacing the legacy
// shared-memory-staged, 8-warp NW-shuffle-reduced GdnPackedDecodeKernel.
//
// The flag gates the EXPERIMENTAL register-resident tiling: default OFF after
// the 2026-07-16 DGX proof FAILED both gates (bit-exact oracle boundary FAIL +
// c16 A/B 700.5-701.4 vs legacy 793.6-794.5 tok/s, TPOT 190.5 vs 166.5 ms —
// register-pressure/occupancy collapse from the naive lane-owns-[1,BK=128]
// mapping). VT_GDN_PACKED_REG_TILE=1 opts in for experiments only. This
// preserves the house same-binary
// GDN A/B convention (GdnTritonEnvOn, VT_GDN_CHUNKED, VT_GDN_DECODE_NW): the
// launcher reads it per call (NOT process-cached) so the coarse decode dispatch
// pays a negligible getenv and in-process tests can flip the selection; the
// parse itself is factored here so it is regression-covered on every platform,
// not just DGX.
#ifndef VT_CUDA_GDN_PACKED_REG_TILE_H_
#define VT_CUDA_GDN_PACKED_REG_TILE_H_

namespace vt::cuda {

// Pure predicate for the VT_GDN_PACKED_REG_TILE contract: register-resident
// tiling is OFF by default (legacy shared-memory kernel ships); it is ON only
// when the environment value is present AND its first character is '1'.
// nullptr (unset) and every non-'1'-leading value are OFF. Kept separate from
// the getenv call in the launcher so the parse is unit-testable without
// touching the environment.
inline bool GdnPackedRegTileFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_GDN_PACKED_REG_TILE_H_
