# Spec: Keep-quant chunk policy — tiling-aware cap, eager only

Row: `BACKEND-TENSTORRENT`
Issue: `ISSUE-LOCAL-01M2N8DKVKM2J03FCVBSKYVHXK`
State: ACTIVE (2026-09-16)
Git integration: one PR for spec and implementation (developer preference,
recorded for this row).
Base: `origin/main`, rides on the binary_ng reshape fix (PR #3211 — the e2e
probe only reaches decode with it).

## Scope

The E=1 chunked keep-quant decode (`MatmulBTQuantGroupedKernel`,
`tenstorrent_ops.cpp:3034-3126`) sizes chunks from the 256 MiB f32 plane
budget and the `ceil(N/8)` trace-region term. The budget counts the decoded
f32 plane (`chunk × K × 4` B) but NOT the repair lambda's TILE intermediates:
`repair` tiles the `{B,16,16}` product planes to `{B,32,32}` — a 4× blowup.
At the APEX 27B head `[248320, 5120]` Q6_K the trace term forces 31,040-row
chunks; the largest single TILE alloc inside the repair is then 2,542,796,800
B (317,849,600 B/bank) against a 238,387,200 B largest free block. Decode
dies at the first lm_head call after 64 trunk blocks.

Out of scope: capture-path chunk policy (unchanged), the throughput floor
(#1003), any numerics change (the decode math is untouched).

## Attribution (measured 2026-09-16)

- Red: default-chunk e2e probe crashed with the OOM fatal above (log
  monitor-1789555005-8fa2, line ~3019).
- Green: the identical probe with `VT_TT_KEEPQUANT_CHUNK_BYTES=67108864`
  (64 MiB hard cap → chunk 3,276) completed: exit 0, `Total generated
  tokens: 64` (log monitor-1789563212-406b). This is the FIRST complete
  APEX 27B e2e generation on the P150.
- The cap formula `plane_bytes / (K * 16)` reproduces the proven 3,276-row
  chunk from the default 256 MiB budget: 268435456 / (5120 × 16) = 3,276.

## Design

One guard after the existing env hard cap (`tenstorrent_ops.cpp:3090-3091`):

```cpp
// The repair lambda tiles the {B,16,16} product planes to {B,32,32} — 4x
// the f32 plane the budget above counts, the 27B-head OOM (issue
// ISSUE-LOCAL-01M2N8DKVKM2J03FCVBSKYVHXK).
// Cap eager chunks at the tiled plane; under capture the policy stays:
// many-small-chunks die on the trace region (~3.3 MB stream per chunk).
if (!tt_capture_active())
  chunk = std::min(chunk, std::max<int64_t>(plane_bytes / (K * 16), 1));
```

The policy is sticky PER WEIGHT, not per call: a captured replay must
hash-match the eager warm-up's stream, and a per-call capture flag makes
warm-up and capture diverge (the slice extent differs, the capture-time
`SliceDeviceOperation` misses the program cache, and the poisoned capture
state cascades). The first decode of a weight is always eager — a
capture-time arrival with a cold shadow refuses at `EnsureKeepQuantWords`
(`tenstorrent_ops.cpp:2034`) — so the flag is recorded eagerly and replayed
verbatim under capture. The cap therefore also applies to captured arms
whose first use was eager; when a captured arm trips the trace region
(~3.3 MB stream per chunk, vehicle survey), that is the named lever to
revisit, not a silent policy fork.

The cap is conservative for the other encodings: Q4_K/Q5_K share the repair
path; Q8_0's `{B,32}` planes are already tile-aligned so its largest TILE
alloc is smaller than the cap assumes. Same-numel metadata reshapes carry no
numerics risk; the decode chain and its bit-exact pins are untouched.

## Tests

- Red-first: the default-chunk OOM fatal (crash log cited above) is the red
  evidence; the env-lever green run is the mechanism proof.
- Green-after (the gate): the APEX e2e probe recipe WITHOUT the env knob —
  `--num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 --seed 0
  --temperature 0 --ignore-eos --skip-chat-template` — completes with 64/64
  generated tokens and exit 0.
- Full backend suite green on the P150 (77/77, 524,495 asserts): the 0.8B
  vehicle shapes exercise the chunked arm; where the plane budget does not
  bind, the new cap must not change chunk counts (assert by suite pass).
- Capture safety: no captured-path change; the guard sits behind
  `!tt_capture_active()`.

## Risks

- Eager chunk count rises (~76 chunks for the 27B head vs 8). Measured
  acceptable: the green run ran exactly this chunk count end to end. The
  throughput cost is real and belongs to the #1003 axis, not this fix.
- If a future encoding's repair tiles wider than 4×, the cap is stale. The
  `Q6_K`-derived divisor is named in the comment so the next encoding
  reconciles it.

## Owed

- The captured 27B arm and its chunk-stream budget (named when a MoE/capture
  artifact demands it).
- Throughput: the chunk-decode cost and the TPOT floor (#1003).
- A correctness gate of the 64 generated tokens against an oracle before any
  published bench record (the e2e pass is an unblock datum, not a golden).

## Stop conditions

- If the no-knob probe OOMs at a different site after the cap, stop and
  re-attribute: the cap provably reproduces the green chunk size, so a new
  fatal names a second allocation class.
- If the backend suite regresses on the 0.8B vehicle, stop: the cap changed
  a chunk count where the plane budget was supposed to bind.
