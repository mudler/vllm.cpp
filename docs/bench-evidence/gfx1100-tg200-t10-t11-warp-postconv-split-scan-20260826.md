# GFX1100-TG200 — T10+T11: warp postconv and row-split scan ADOPTED (corrected record)

Date: 2026-08-26 (valid windows 03:20Z and 04:14–04:20Z plus full-config
verification 05:2xZ). Host: local RX 7900 XTX (gfx1100), native `build-hip`,
branch `row/GFX1100-TG200`. Checkpoint sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.

## CORRECTION HISTORY — read before citing

An earlier revision of this file claimed +4.3%/+3.9% from a window whose
outputs were later found DEGENERATE (token loops). Root cause: T10's
GdnPostConvWarpK computed the conv row stride `key_dim+value_dim` instead
of the donor's `2*key_dim+value_dim` ([q|k|v] layout) — decode rows masked
it, prefill rows read wrong memory. The stride is fixed; the claims below
come from post-fix windows whose bodies were coherence-checked. The failed
windows and the process rules they forced (body-content check per arm,
engagement witness per window, all-targets relink) are retained in the
git history of this file.

## T10 — GdnPostConvWarpK (`VT_GDN_POSTCONV_COOP=1`, default OFF)

Warp-per-item remap of the chunked donor (which hands each decode item to
ONE thread walking dk=128 serially twice): lane-strided walks, shfl sumsq
trees. Sumsq association changes → opt-in flag, adjudication owed before
any default flip.
Kernel time (rocpd): **27.9 → 2.76 µs** (10×).
Clean-window A/B x5 interleaved pairs, only the flag varied:
OFF median **82.42 tok/s**, ON median **86.31 tok/s** — ON wins all five
pairs, **+4.7%**. Bodies coherent analytic prose both arms; divergence at
expected tie-flip points.

## T11 — GdnScanCoopSplitK (`VT_GDN_SCAN_SPLIT=1`, requires SCAN_COOP)

Row-split blocks (RS=4: 32→128 blocks at decode) plus register-cached row
segments between the dot and update passes. State rows are independent, so
per-row arithmetic is UNCHANGED: engine outputs are BIT-IDENTICAL — all
five stacked pairs byte-identical across 256 greedy tokens through 24
layers.
Kernel time (rocpd): CoopK **30.4 → 9.57 µs** (3.2×).
A/B x5 interleaved pairs (on the T10-OFF base): OFF median **82.29**,
ON median **84.95** — ON wins all five pairs, **+3.2%**.

## Gate

Focused suite **15/15 cases, 826 assertions** including the T10
COOP-vs-donor NMSE + inertness case. Post-retraction hardening: the arm's
env toggle reads PER CALL (the once-per-process static let the unit test's
ON arm silently reuse the donor — mutation-verified fix, nmse 1.30 RED
with the stride bug reintroduced).

## Full-stack position

All adopted levers on (`MMVQ SKINNY GQA4 SCAN_COOP PREAMBLE_COOP
NORM_QUANT_FUSED RMSNORM_ROW_COOP NORMGATED_COOP POSTCONV_COOP
SCAN_SPLIT`):
- Short prompt (~45 tok): warmup 89.5, steady **99.9 tok/s ×2**.
- Canonical 70-token prompt: warmup 84.3, steady **92.9/92.7 tok/s**,
  coherent.

Prompt-length caveat: tonight's paired A/Bs used the ~45-token prompt;
older windows used longer prompts, so absolute numbers are not
cross-era comparable — the PAIRED DELTAS are the verified quantities. A
formal acceptance-gate rerun (canonical long prompt, idle host, 6-rep
median) on this config remains owed for the campaign's absolute position
record.

## Session ledger context

Adopted across sessions: T5a (+23%), T5b (+13.5%), T6a (+4.6%), T6b
(+4.6%), T8 (+3.2%), T9 (+2.6%), T10 (+4.7%), T11 (+3.2%) — all paired,
all coherence-checked. Closed negative/not-adopted: T5c, T7, T12.
Failed-attempt ledger: 3 of 10.

## Full-config verification (2026-08-26 late, clean GPU)

With the sibling training finished (full VRAM), the complete eleven-flag
config was verified end-to-end:
- Graph replay ENGAGES with all new arms captured: "[DenseDecodeGraph]
  captured ... S=1", "126 total replays" over 128 tokens — capture-safety
  of every arm added this session is empirically confirmed.
- Short prompt (~45 tok): warmup 89.5, steady **99.9/101.1 tok/s**.
- Canonical 70-token prompt: warmup 84.3, steady **92.9/92.7 tok/s**,
  coherent analytic output.

Fresh rocpd budget at this config (8.89 ms/tok kernel busy): the three
streaming families hold 6.33 ms/tok at their audited near-peak rates;
every latency-class kernel added or remapped this session sits at
0.02–0.75 ms/tok. Remaining non-kernel time ~1.9 ms/step decomposes into
the ~290 us sampling round trip plus per-op launch gaps — T13 scope,
requiring the async-serving engine path (the blocking CLI cannot engage
AsyncScheduler), which is the next session's scoped item.

## Async-serving measurement attempt (T13 scope closure, same day)

With real event primitives landed, `VT_ASYNC_RUNNER=1` now resolves
`async_sched_supported=1` (debug-print verified) and the server engages
AsyncScheduler mcb=2 with COHERENT output — the R9700-class garbage is
fixed at the source. But the throughput A/B through the OpenAI endpoint is
a WASH (sync 55.9 vs async 55.7 medians) because the SERVER PATH ITSELF
runs at ~55 tok/s where the CLI reads 92.9 on identical flags: HTTP +
serving-layer overhead dominates and masks any scheduler-overlap gain.
Also noted: two simultaneous engines cannot share the GPU (second load
OOMs / "stopped AsyncLLM"), so dual-server interleaving is unavailable.

Conclusion: the sampling-round-trip lever cannot be measured through the
serving path until the server's own ~40% overhead is attributed, and the
blocking CLI cannot engage AsyncScheduler by construction. The contained
alternative for a future session: one-step-deferred D2H inside
LLLMEngine::step (double-buffer the sampled-id host read) so the sync loop
overlaps detokenization with the next forward — no scheduler change, no
server dependency.

## Host-load sensitivity finding + T15 attempt closed negative (2026-08-26 later)

A post-retraction rmsnorm_row "LDS epilogue" attempt (cache the rounded
bf16 row in shared memory to skip the q8 epilogue's global re-read)
measured a -38% REGRESSION on a clean GPU and was reverted byte-restored:
the gmem re-read it removed was already L1-resident (~5 KB row), while the
u16 LDS access pattern from consecutive lanes incurred heavy bank
conflicts. Attempt recorded; lever closed.

Separately, post-revert verification read 53-58 tok/s with BYTE-IDENTICAL
code to the 92.9 tok/s window — root cause is HOST CPU contention (two
sibling python processes at ~200% each plus a llama-server; load 4.9-5.7
vs 2.5-3.7 in the fast window). Launch-bound decode scales with host
scheduling quality. MEASUREMENT RULE ADDED: engine tok/s numbers are only
comparable at recorded host load; future acceptance runs must log loadavg
per rep (now done) and treat windows above load ~4 as provisional for
absolute claims (paired A/Bs remain valid).

## CORRECTION: wvSplitKSml per-site rates (position-resolved, same capture)

The earlier "~700 GB/s aggregate" read blended three distinct sites. With
each call assigned to its step position across 505 steady steps (72
calls/step = 24 GDN layers x 3 projections), the durations are cleanly
periodic:

| pos%3 | tensor | bytes/call | median us | GB/s |
|---|---|---|---|---|
| 0 | attn_qkv [4096,2560] | 20.97 MB | 46.00 | **456** |
| 1 | attn_gate [4096,2560] | 10.49 MB | 23.72 | **442** |
| 2 | ssm_out [2048x? class] | 10.49 MB | 26.52 | **396** |

(The prior "700 GB/s aggregate" and "911 GB/s on qkv" figures used wrong
byte assignments.) The family therefore HAS headroom: ~0.45-0.6 ms/tok to
a ~550-600 GB/s practical target. The launches are donor-tuned via
`mindiv(N, cu*kYtile, kWvPrGrp)` for other shape classes; a per-shape
launch-config sweep (kYtile/wvPrGrp/split factor) on gfx1100 for exactly
these three (N,K) shapes is the concrete next lever, priced at up to
~+0.5 ms/tok. ArgmaxSplitPhaseA (34 us) and the two-phase argmax total
44.7 us are separate items already recorded.

## T16 launch-config sweep (VT_WVSPLIT_YTILE / VT_WVSPLIT_PRGRP)

Implemented: kYtile templated {1,2,4} with per-call dispatch, plus a
runtime work-groups-per-grouping override. Sweep under host load ~5
(medians of 3): default 50.99; PRGRP=8 51.32; PRGRP=4 50.19; PRGRP=2
46.19 (-9%); YTILE=1 50.57; **YTILE=4 53.55 (+5%)**.

Paired same-window verification x5: baseline median 52.57 vs YTILE=4
53.21 (+1.2%) — distributions overlap; directionally positive but NOT
conclusive under contention. Knob kept default-OFF-equivalent (env unset
= donor config); idle-host re-sweep owed before any adoption. The
position-resolved audit's ~0.45-0.6 ms/tok headroom estimate stands;
the sweep so far captured only a fraction of it.
