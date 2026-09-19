ID: ISSUE-LOCAL-01J7QX3M5K1HE3N9V0KQRJ8X2M
Title: Route GLM-5.3-Flash MLA attention through the shared mla::ForwardMlaAttentionBlock seam (W9c-1)
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-17
Updated: 2026-09-17
Closed: -

## Problem

W9c-1 was REFUSED at W9c-3a because the k-pool indexer has no device
implementation and the premise required all DSA arms on device. W9c-3 has
since landed, giving the device forward the compose infrastructure. The refusal
is re-priced: the MLA attention arm can move to device through the shared
`mla::ForwardMlaAttentionBlock` seam while the k-pool indexer stays as a
host-fallback island, the same pattern W9c-3 used for the other arms.

The port requires a loader absorb (`AbsorbMla` producing `kv_b_proj`, `w_uk_t`,
`w_uv`), a `BuildMlaStep` equivalent, `MlaBlockWeights` via `ResidentWeight`, and
a default-constructed `TritonMLAImpl` for pure prefill. The sibling
`GlmMoeDsaForCausalLM` already drives this seam on GPU; the pattern is a direct
port.

## Resolution

Implementation committed on `row/MODEL-MM-GLM53-FLASH-W9C1`:
- `glm5_next_loader.{h,cpp}`: added `AbsorbMla()` that dequantizes
  `attn_k_b`/`attn_v_b`, transposes k, stacks into `kv_b_proj`, calls
  `mla::AbsorbKvBProjBf16` to produce `w_uk_t`/`w_uv`
- `glm5_next_device.cpp`: added `Glm5NextMlaBlockDims()`,
  `Glm5NextResidentMla()`, MLA step construction, replaced host `Attention()`
  with `mla::ForwardMlaAttentionBlock` + download
- `test_glm5_next_forward.cpp`: added ROCm backend detection to W9c-3 device test

CPU test passes: 22/33 with `VT_GLM5_NEXT_DEVICE=1` (11 failures are
pre-existing issue #2348), 33/33 without the device flag. GPU gate pending.
