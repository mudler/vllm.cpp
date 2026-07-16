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
// The flag is the SAME-BINARY rollback: default ON (register-resident is the new
// default); VT_GDN_PACKED_REG_TILE=0 (any value whose first character is '0')
// restores the legacy kernel bit-for-bit. This mirrors the house default-ON
// GDN A/B convention (GdnTritonEnvOn, VT_GDN_CHUNKED, VT_GDN_DECODE_NW): the
// launcher reads it per call (NOT process-cached) so the coarse decode dispatch
// pays a negligible getenv and in-process tests can flip the selection; the
// parse itself is factored here so it is regression-covered on every platform,
// not just DGX.
#ifndef VT_CUDA_GDN_PACKED_REG_TILE_H_
#define VT_CUDA_GDN_PACKED_REG_TILE_H_

namespace vt::cuda {

// Pure predicate for the VT_GDN_PACKED_REG_TILE contract: register-resident
// tiling is ON by default; it is OFF (legacy shared-memory kernel) only when the
// environment value is present AND its first character is '0'. nullptr (unset)
// and every non-'0'-leading value are ON. Kept separate from the getenv call in
// the launcher so the parse is unit-testable without touching the environment.
inline bool GdnPackedRegTileFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_GDN_PACKED_REG_TILE_H_
