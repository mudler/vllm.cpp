# Spec: kCastBf16/kCastF32 L1 overflow — metadata reshape in NormalizeDevF32Tile

Row: `QUANT-GGUF-IQ-TENSTORRENT` (the APEX e2e owed item owns this blocker)
Issue: `ISSUE-LOCAL-01M2K8WWA0X09VJF55NTS16VTV`
State: ACTIVE (2026-09-15)
Git integration: one PR for spec and implementation (developer preference,
recorded for this row).
Base: `origin/main`.

## Scope

Fix the latent L1 overflow that kills the APEX-I-Nano e2e at the first
forward: `NormalizeDevF32Tile`'s free `ttnn::reshape(x, {1, n})` on a wide
f32 ROW_MAJOR tensor launches `reshape_rm`, whose statically allocated
staging CBs scale with the row width. At the artifact's gated-norm
activation `[T=128, inner=6144]` the f32 row is 3,145,728 B — 2x the
1,572,864 B L1 cap — and program allocation dies with "Statically
allocated dataflow buffers on core range [0-0 - 10-9] grow to 2213200 B"
(`dataflow_buffer.cpp:2608`). Out of scope: the 09-14 DRAM OOM class in
the keep-quant `ttnn::where` repair chain (W4d W2 already landed its fix;
today's runs get past it), any keep-quant wave behavior, and any
throughput claim.

## Attribution (measured 2026-09-15, evidence in the issue)

- NOT a wave regression: the pre-waves binary (`6cff72e41` lineage) fails
  byte-identically today (`apex-e2e-oldbin.log`). The 09-14 run passed
  this program only because DRAM pressure (`--fit` at max_model_len 2048)
  killed it earlier, inside the keep-quant `ttnn::where` chain, before the
  reshape program compiled. `apex-e2e.log` has zero layernorm mentions.
- The diff between the old and new trees touches nothing on this path
  (40 added / 5 removed lines, all keep-quant dispatch; `qwen3_5.cpp`
  zero).
- Artifact dims (parsed from the GGUF header): `embedding_length 5120`,
  `feed_forward_length 17408`, `ssm.inner_size 6144`, 64 trunk blocks.
  The failing program serves the gated-norm activation through
  `kCastBf16` → `ServeDeviceShadowRaw(in, 1, n)` →
  `NormalizeDevF32Tile(dev_in, 1, n)`
  (`src/vt/tenstorrent/tenstorrent_ops.cpp:4177`, `:4215`, fix site
  `:7493-7502`).
- The `(rows, cols)` arguments both call sites pass are `(1, n)` — the
  FLATTENED shape — so the guard `ls != {rows, cols}` is true whenever the
  producer's shadow is natively 2D, and the free reshape runs at full
  width.

## Design

The doctrine comment on `NormalizeDevF32Tile` already states the reshape
on a fresh ROW_MAJOR contiguous buffer is "a pure metadata view" — the
bug is that the free `ttnn::reshape` does not honor that: it launches the
`reshape_rm` device program with L1-scaled staging CBs. The
`CaptureSafeReshape` doctrine (PR #3190, `tenstorrent_ops.cpp:~224`) is
the precedent: a metadata-only reshape (member
`Tensor::reshape(logical, padded)` / `tt::tt_metal::view_device`) changes
the logical shape without a device program and without CB allocation.

1. In `NormalizeDevF32Tile`, when the incoming logical shape differs from
   `(rows, cols)` but the tensor is ROW_MAJOR, contiguous, and numel-
   equal, apply the metadata-only reshape instead of the free
   `ttnn::reshape`. The subsequent `typecast` and `to_layout(TILE)` are
   elementwise and shape-agnostic on the contiguous buffer; the TILE
   rebuild happens at the TRUE `[rows, cols]` geometry, which is what the
   doctrine requires. The free reshape stays for any case the
   metadata view cannot serve (non-contiguous, numel mismatch — the
   guard set), so behavior elsewhere is unchanged.
2. The elementwise ops run at the shadow's native `[T, inner]` shape —
   tile Normalization op count scales with rows exactly as the `[1, n]`
   form's would (same element count, same tiles after padding); no
   arithmetic or layout change. The 1-row form's L1 hazard disappears
   because no staging CB ever spans a 3 MB row.
3. Capture safety: the metadata reshape is capture-safe by construction
   (no program, no cache lookup) — the same property #3190 relied on.

## Tests

- Red-first op-level focused test: through the production dispatch
  (`ModelRegistry`/op dispatch path the e2e uses — `kCastBf16` reached
  with a `[128, 6144]` f32 device shadow and a bf16 out tensor), assert
  the cast completes and the result is bit-exact vs the host RNE cast
  (`StoreElemF32`'s single-round). RED today: the L1 fatal at program
  allocation. Run under the GPU mutex with a cleared tt-metal cache.
- Bit-exactness guard: the existing cast cases in the op suite stay
  green unchanged (small shapes hit the same code path).
- Full backend suite green on the P150 (76/76 baseline).
- The APEX e2e re-run is the W3-class closure: NOT this unit's gate, but
  the first post-fix run gets recorded in the issue either way (pass, or
  the next blocker named).

## Risks

- The metadata reshape on a device shadow must keep the buffer's padded
  shape coherent with the new logical shape (`view_device` semantics;
  the #3190 precedent handles exactly this). A wrong padded-shape
  derivation shows up as the focused test's bit-exact memcmp, not
  silently.
- `to_layout(TILE)` at `[T, inner]` vs `[1, n]`: tile padding rows
  (128 is already tile-aligned; 6144/32 = 192, aligned) — for
  non-aligned rows the tile count could differ from the flattened form's.
  The 27B e2e and the op suite pin the equivalence on the shapes that
  exist in production; if a production shape exists where the two forms
  disagree on tile count, that case must take the old path (assert loudly
  rather than diverge).

## Gates

1. Focused red-first test red → green, bit-exact vs host.
2. Full backend suite green on the P150.
3. 0.8B vehicle battery unchanged if the row's invariant applies (the
   cast path is production-reached there).

## Owed

- The APEX e2e generation gate (this fix unblocks the first forward;
  later blockers get named when reached).
- The 09-14 DRAM OOM record in the row spec gets superseded by this
  attribution in the outcome section.
- F1: a test that consumes the committed served-geometry record (the
  review notes below).
- F2: migrate the hardcoded-geometry NormalizeDevF32Tile consumers
  (kSigmoidGateBf16, L2Norm arms) to served-geometry commits.

## Stop conditions

- If the metadata view cannot serve some production case (guard fires),
  stop and escalate with the shape; do NOT fall back to chunked reshapes
  in this unit.
- If the focused test cannot go red on the current tree (the L1 fatal is
  config-dependent), record the config that reproduces it and stop for a
  decision.

## Review notes (2026-09-15, fresh review PASS)

The fresh review confirmed the regression guard (mutation 3: the free
reshape reintroduced → the exact L1 fatal, red) and the guard
narrowing (static: non-rank-2 path byte-equivalent to pre-fix). Two
findings recorded, non-blocking:

- F1: the served-geometry COMMITMENT in kCastBf16/kCastF32 is
  mutation-invisible to the suite — the oracle is bytes-only, and the
  committed record is not consumed by any observed serve. A follow-up
  test that consumes the committed record (a serve at the native
  geometry asserting no reshape fallback) is the owed test.
- F2: other NormalizeDevF32Tile consumers with hardcoded geometry
  commits (kSigmoidGateBf16, the L2Norm arms) predate this doctrine;
  no regression today (a wide rank-2 shadow was L1-fatal there before
  this fix too), but their migration to served-geometry commits is
  owed with the same shape as this fix.

Both ride the row's Owed list.
