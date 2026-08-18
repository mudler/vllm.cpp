# ROCm attention backend: registration + engine-level selection (M3, issue #41)

Row: `BACKEND-ROCM`. Spec written at PR time — the implementation already exists
(#1056 registration, #1065 runner selection); this spec records the DECISIONS the
implementation embodies, so they are reviewable and pinned. Open for co-iteration
with the maintainer; the layout decision (§3) and the KV-connector guard (§4) are
the two places a reviewer's correction would land.

## 1. Scope

Two PRs, two concerns:

- **#1056** — register a `ROCM_ATTN` engine-level attention backend for `kROCM`
  (self-registering `RocmAttentionBackend` + a `rocm.cpp`
  `get_attn_backend_priority()` slot). Additive; ZERO selector/model/runner edit,
  matching the `BACKEND-ROCM` row's landing contract.
- **#1065** — reach the previously-dead engine-level selection seam from the
  runner (`initialize_kv_cache`), per attention GROUP, validating each group's KV
  view against the selected backend's declared shape. This is a distinct concern
  from the ROCm row and is tracked separately (see §7 Owed).

Both branch from `0f8580e26`; they touch disjoint files. #1056 must land first.

## 2. Upstream anchors (verified at pin `555967922`)

All citations below were re-derived against the project's upstream pin by
fetching the files at that exact SHA:

| Anchor | Value at pin | Notes |
|---|---|---|
| `vllm/platforms/rocm.py` `_get_backend_priorities` | def at `:407`, returns at `:440-441` | dense + MLA + sparse branches |
| ... AITER gates | `rocm_aiter_ops.is_mha_enabled()` `:434`, `is_aiter_found_and_supported()` `:436` | NOT `on_gfx9()` — that appears only in `get_vit_attn_backend` (`:642`, `on_gfx9()` at `:663`) |
| ... `use_kv_connector` guard | `if not use_kv_connector:` `:432-433` (ROCM_ATTN append) | comment at `:430-431` |
| ... layout quote | `:521-522` "ROCM_ATTN still uses a legacy attention layout (KV is the outer dimension)" | |
| `vllm/v1/attention/backends/rocm_attn.py` `get_kv_cache_shape` | `:247-256` | `%16` check at `:249-251`; returns `(2, num_blocks, block_size, num_kv_heads, head_size)` |
| `rocm.py` `get_attn_backend_cls` | `:545` | the *selector* entry, distinct from `get_vit_attn_backend` |

Dense priority list, mirrored verbatim: `[ROCM_ATTN, ROCM_AITER_FA,
ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN, TURBOQUANT]`. Unregistered names are
skipped by `SelectAttentionBackendName`, so mirroring the full list costs nothing
and needs no gfx9 reasoning.

## 3. KV-layout deviation — ONE exact tracked exception

Upstream `ROCM_ATTN`'s defining property is the K/V-OUTERMOST cache
(`rocm_attn.py:247-256`, `rocm.py:521-522`). Our engine allocates the shared NHD
`(num_blocks, 2, block_size, num_kv_heads, head_size)` layout for every dense
attention layer, and our ROCm paged-attn kernel (`src/vt/rocm/rocm_paged_attn.hip`)
reads exactly that. Registering `ROCM_ATTN` with the NHD shape therefore inverts
upstream's property — it cannot be otherwise, because the K/V-outermost cache
does not exist in this engine.

**Decision: keep the name, record the deviation as one exact tracked exception.**
Rationale: the name identifies the kernel family that actually runs (the ROCm
paged-attn kernel behind `kPagedAttention`/`kROCM`), the selection log reports
`ROCM_ATTN` on real silicon, and registering `FLASH_ATTN` instead (the
Metal/Vulkan/Tenstorrent sidestep) would leave `ROCM_ATTN` permanently dead in
the priority list — a false claim that ROCm attention is unsupported. The
exception is pinned here and in `include/vllm/v1/attention/backend.h`; when (if)
a real upstream-layout ROCm kernel lands, this registration flips shape with it
and stops being an exception.

## 4. KV-connector guard (`use_kv_connector`) — why it does not apply

Upstream appends `ROCM_ATTN` only `if not use_kv_connector` (`rocm.py:432-433`),
protecting connector (LMCache) transfer semantics from its asymmetric native K/V
views. We ship a KV connector (`include/vllm/v1/kv_offload/kv_connector.h`), but
our registered shape is the shared SYMMETRIC NHD layout — the same one
`FLASH_ATTN` allocates, which upstream does use with connectors. The guard's
premise (asymmetric views) does not exist for this registration, so no
`AttnSelectorConfig::use_kv_connector` field is added. Escape hatch: if a future
ROCm kernel adopts the asymmetric layout, this registration flips shape AND the
guard becomes load-bearing.

## 5. Runner selection (PR #1065)

`SelectAttentionBackendName` had ZERO production callers on main — the entire
engine-level registry was dead code. #1065 is the first thing that reaches it:

- Resolution moves INSIDE the `full_attn_group_id_ >= 0` region (a pure-GDN /
  pooling model caches no paged KV and pays no selection).
- Resolution is per GROUP, not per runner: the dense full-attention group resolves
  with `use_mla=false` (eager, throws loudly if the platform has no registered
  dense backend — the empty-list loud-throw design); an MLA group resolves with
  `use_mla=true` and is TOLERANT — on a device with no registered MLA backend
  (e.g. CPU today) the name is empty and execution stays op-driven
  (`TritonMLAImpl` on the fused 3-dim cache), which is not registry-gated. On
  CUDA the MLA group resolves `TRITON_MLA` and validates against the fused
  `(num_blocks, block_size, head_size)` view the engine actually allocates.
- Per-group shape validation: the resolved backend's `get_kv_cache_shape` must
  equal the view the engine allocates — NHD 5-dim for dense groups, fused 3-dim
  for MLA groups — via `vllm::v1::CheckKvCacheShape` (registry.h). A future
  backend with a different layout fails LOUDLY at init.
- `VT_ATTN_SELECT_LOG=1` prints one line per attention layer (backend name,
  device, shape).

## 6. Tests

- `test_rocm_backend.cpp`: priority list == verbatim dense mirror; selection
  resolves `ROCM_ATTN`; NHD shape asserted; AITER/TRITON/TURBOQUANT unregistered.
- `test_attn_backend_registry.cpp`: registration mutation; NEW mis-shaped
  scratch-backend case proving `CheckKvCacheShape` throws on a layout mismatch.
- `test_runner.cpp`: CPU dense resolves `FLASH_ATTN` (behavior-preserving).
- `test_kimi_linear_paged.cpp`: MLA group on CPU — tolerant path (no throw),
  exercised by the existing runner construction (block size 8→16: the `%16`
  contract is now reachable, and is enforced by both FLASH_ATTN and ROCM_ATTN).
- CI: `test_kimi_linear_paged` + `test_bench` were RED after the first #1065
  push; both are fixed by this pass (kimi block 8→16; bench rounds its synthetic
  `seq_budget` block size up to a multiple of 16; `server_main --block-size` now
  validates). See §7.

## 7. Owed / follow-ups

- **Own row + issue + spec for the runner selection** (#1065 is a different
  concern from the ROCm row and should not ride issue #41 long-term). Proposed
  row: `BACKEND-ATTN-SELECTION-RUNNER`; the maintainer offered to help set it up.
- **The `%16` contract becoming reachable** is a deliberate, announced change
  (entry-point fixes in #1065); it deserves its own migration note and, if the
  maintainer prefers, a split PR.
- **backend-matrix.md `BACKEND-ROCM` row** ("empty attn priority — no kernel, so
  no claim" is stale the moment this lands) and **docs/FEATURES.md** /
  **docs/ROCM.md** M3 bullets: updated in #1056.
- **HIP CI**: `test_rocm_backend` assertions are gated on `VLLM_CPP_HIP` and no
  HIP job exists in `.github/workflows/ci.yml`; the gfx1151 evidence is
  contributor-hardware. A HIP CI leg is a separate, valuable follow-up.
- **M4**: vLLM-ROCm oracle token gate on gtr9 (gfx1151) for the ROCM_ATTN path.

## 8. Reachability

- #1056 alone: `RocmAttentionBackend` is exercised by tests only (registry +
  platform selection); nothing in `src/`/`include/` consumes it until #1065
  lands — the staged-slice contract.
- #1065: `initialize_kv_cache` is production
  (`src/vllm/v1/worker/gpu/runner.cpp:413,454`); deleting the selection block reds
  `test_runner.cpp` — the seam is genuinely reached.
