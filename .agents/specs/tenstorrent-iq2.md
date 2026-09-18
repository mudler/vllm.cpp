# Spec: QUANT-GGUF-IQ-TENSTORRENT wave 2 — IQ2_XXS + IQ2_S on the P150

Row: `QUANT-GGUF-IQ-TENSTORRENT`
Issue: `ISSUE-LOCAL-01M2CNC22SYAKJN3YBGNVGSG56`
State: ACTIVE (2026-09-14)
Git integration: one PR for spec and implementation (developer preference).
Base: `origin/main` (wave 1 landed as PR #3193; branch `row/QUANT-GGUF-IQ-TENSTORRENT-W2`
or stack on the wave-1 branch if #3193 is unmerged at start).

## Scope

Serve `DType::kIQ2_XXS` and `DType::kIQ2_S` matmul weights through the TT
int8-dot keep-quant decode, mirroring wave 1's IQ3_XXS shape exactly:
device vec_dot ports, word staging, dispatch, route admission. Q3_K is
explicitly wave 3 (different block structure — min-term K-quant, not the
codebook family) and is out of scope here.

## Upstream anchors

- CPU references (the oracles, bit-exact targets):
  - `VecDotIQ2_XXSQ8_K`, `src/vt/cpu/cpu_quant_dot.cpp:577` (quants.c:855)
  - `VecDotIQ2_SQ8_K`, `src/vt/cpu/cpu_quant_dot.cpp:899` (quants.c:947)
- Blocks: `BlockIQ2_XXS`, `BlockIQ2_S`, `src/vt/cpu/cpu_quant_blocks.h`.
  IQ2_XXS is 66 B = `{u16 d; u16 qs[32]}` (:157) — `qs` holds 8 u32 pairs:
  each 32-sub-block memcpy's two u32s into `aux32`, the low bytes index the
  grid, `aux32[1]` carries the 4-bit scale (>>28) and the four 7-bit sign
  words. IQ2_S is 82 B = `{u16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]}` (:197).
  Both 256 elems; both pair with q8_K per `cpu_quant_traits.cpp`.
- Tables: `kIq2xxsGrid[256]` u64 (:52, 8-byte entries, TERNARY -1/0/+1
  lanes), `kIq2sGrid[1024]` u64 (10-bit index spliced
  `qs[l] | ((qh[ib32] << (8-2l)) & 0x300)`), `kKsignsIq2xs[128]`,
  `kKmaskIq2xs[8]` — `src/vt/cpu/cpu_quant_iq_tables.h`.
- Wave-1 pattern (the template): `kq_vec_dot_iq3_xxs_q8_K` +
  `kernels/iq3xxs_tables.h` + `KeepQuantWordsPerBlock` +
  `enc_sel`/kernel-arm/dispatch + `DeviceKeepQuantSupported` + the pin
  tests — all landed in PR #3193.
- Gate artifact: the same APEX-I-Nano pin (wave-1 spec ## Upstream anchors);
  89 IQ2_S + 44 IQ2_XXS tensors become device-decoded.

## Design

Mirror wave 1 point-for-point:

1. `kernels/iq2_tables.h` (new): `kq_iq2xxs_grid[256]`, `kq_iq2s_grid[1024]`,
   signs+mask tables — byte-identical to the CPU header; the op-level oracle
   test is the drift guard. NOTE the entry size: IQ2 grids are u64 (8 B/entry,
   ternary lanes), unlike IQ3's u32 — size the device reads accordingly.
2. `keepquant_kernel_code.h`: `kq_vec_dot_iq2_xxs_q8_K` and
   `kq_vec_dot_iq2_s_q8_K` — ported statement-for-statement; IQ2_XXS's
   qh-splice and IQ2_S's per-ib32 `ls1/ls2` low/high split exactly as the
   CPU reference; `0.125f * sumf` folds last; no divisions.
3. Staging: `KeepQuantWordsPerBlock(kIQ2_XXS)` and `(kIQ2_S)`: IQ2_XXS
   66 B → 128 B → 32 words; IQ2_S 82 B → 128 B → 32 words (both
   zero-padded, sub-word tail like Q6_K/IQ3_XXS).
4. Kernel dispatch: `enc_sel` 5 = IQ2_XXS, 6 = IQ2_S; kernel arms
   `enc==5`/`enc==6`; activation side stays the existing q8_K quantizer;
   `qb_pad` falls into the `nb*292` arm.
5. Route: `MatmulBTQuantKernel` routes both encodings to the int8-dot arm
   regardless of env; refusal names them. `DeviceKeepQuantSupported`
   kTENSTORRENT adds both; `test_gguf_keep_quant.cpp` pin widens in the
   same change.
6. Docs: extend the `docs/USAGE.md` artifact row's refused arms
   (IQ2_XXS/IQ2_S move out; Q3_K remains named).

## Risks

- u64 grid entries on device: the kernels read packed BYTES via
  `reinterpret_cast<const uint8_t*>` — same as the CPU reference; no
  alignment hazard beyond what wave 1 already carries.
- 10 KB of new tables in the device binary — same order as wave 1's 1 KB
  plus existing kernel sizes; no L1 residency change (tables live in the
  kernel text, read via the same patterns).
- IQ2_S's `qh` splice indexing (`& 0x300`) is the trickiest expression —
  the mutation review must target it (flip the mask, expect red).

## Tests

- Red-first op-level: two new cases in the keep-quant op suite
  (IQ2_XXS×q8_K, IQ2_S×q8_K) vs the CPU oracles; RED = route refusal
  today (named in the registered set). Same sweep-shape pattern as wave 1.
- Default-path legs: extend the wave-1 env-unset pattern to both new
  encodings (the reviewer's dispatch-only mutation must go red).
- Route pin: both admitted on kTENSTORRENT (red before the predicate
  widens).
- Full backend suite stays green (74 + new cases).

## Gates

1. Op-level oracles: both device dots bit-exact vs CPU (116-assertion-class
   sweeps).
2. Route: pin green for both.
3. Backend suite green on the P150.
4. E2E: UNCHANGED from wave 1 — the APEX e2e generation gate stays
   recorded OOM/owed (Q4_K grouped repair-plane residency, keep-quant W4
   territory); wave 2 does not own it. No new device-run bench claim.

## Owed

- Q3_K wave 3 (device port + route).
- W4a grouped E=1 arms for IQ3_XXS/IQ2_XXS/IQ2_S.
- The APEX e2e generation gate (keep-quant W4 residency redesign).

## Stop conditions

- Same as wave 1: soft-float drift that cannot be closed → stop and
  escalate; no residency redesign in this wave.

## Outcome

Landed 2026-09-15 (PR #3200, squash `d4ff05f8c`). Both encodings decode
on-core bit-exact: op sweep 162/162 assertions across 18 shapes x 7
encodings vs the CPU oracles `VecDotIQ2_XXSQ8_K` (quants.c:855) and
`VecDotIQ2_SQ8_K` (quants.c:947); default-path env-unset leg 38
assertions; route pin 61 cases / 12,131 assertions; full backend suite
75/75 cases, 775,034 assertions, 0 failed on the P150. Fresh review
PASS with 7/7 claimed guarantees mutation-verified red, including the
mandatory `& 0x300` qh-splice flip (which lives in IQ2_S, not IQ2_XXS
as the draft spec's Risks section mislocated).

Measured and rejected:

- Plain `constexpr` table placement: the 10,240 B of u64 grids overflow
  the kernel's ~4.8 KB local-data region; the loader refuses at
  tt_elffile.cpp:394. Rejected for `.text` placement with
  `used`+`externally_visible` — a bare `section(".text")` is dropped by
  tt-metal's `-flto=auto` pass and the tables silently reappear in
  `.data`. Placement verified via linked brisc.elf symbols.
- Duplicating the sign/mask tables in `iq2_tables.h`: redefinition
  against wave 1's `iq3xxs_tables.h`; reused instead.

Default values and why:

- Fold constant is `0.125f * sumf` for BOTH encodings — the CPU
  oracles fold with 0.125f; the draft spec's 0.25f was wrong (IQ3_XXS
  is the 0.25f encoding). Corrected in `b05a4e599`.
- Unconditional int8-dot dispatch (no env knob) — same reachability
  rule wave 1 set; the dispatch-only mutation went red on the
  default-path leg, proving the default configuration reaches it.

Gates state: op oracle PASS, route PASS, backend suite PASS; the APEX
e2e generation gate stays recorded OOM/owed (Q4_K grouped repair-plane
residency, keep-quant W4 territory) — unchanged from wave 1, not owned
here. Owed carried forward: Q3_K wave 3, W4a grouped E=1 arms.
