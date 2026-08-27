# Spec — `tt_clock_state`: the Tenstorrent clock-state sampler and judge

Issue: [#2005](https://github.com/mudler/vllm.cpp/issues/2005). First
consumer: [#2003](https://github.com/mudler/vllm.cpp/issues/2003) (the P150
default-arm inversion). Owner row: `BACKEND-TENSTORRENT`.

## Scope

A new `tools/bench/tt_clock_state.py` implementing the
[`.agents/benchmarking.md`](../benchmarking.md) "clock is part of the
measurement" contract for Tenstorrent boards, mirroring
`tools/bench/gpu_clock_state.py` rule-for-rule where a TT analog exists:

- CLI verbs: background `sample` (interval loop; one-shot query), stop-time
  summary writing (the summary exists only after the sampler STOPS), and a
  `judge` that reads 2+ summaries and applies the pairing rules.
- Record fields per window: retained n / min / median / max / spread_pct of
  AICLK (ARCCLK, AXICLK recorded alongside), idle count, board identity
  (`BOARD_ID_HIGH/LOW`), firmware bundle version, KMD + tt-smi + UMD
  versions, boot id, host platform.
- Pair rules with the NVIDIA helper's thresholds and constant names:
  same-boot (waivable, waiver stamps a caveat), within-run spread <= 5%,
  cross-arm median offset <= 1%, cross-arm mean offset <= 1%,
  >= MIN_BUSY_SAMPLES (30) retained busy samples, >= MIN_BUSY_FRACTION
  (0.5) busy, refuse an entirely-idle window while still writing its
  evidence. Exit 0 clean, exit 1 when refusal reasons exist.

Explicit v1 non-goals: no in-process pyluwen sampling (subprocess snapshot at
1 Hz is measured feasible: 430 ms); no claimed-max auto-discovery; no change
to any existing harness's invocation path.

## Upstream anchors

No upstream implements TT host-side clock gating; this mirrors OUR OWN
helper as the reference behavior:
`tools/bench/gpu_clock_state.py` — constants `MIN_BUSY_SAMPLES`,
`MIN_BUSY_FRACTION`, `MAX_CROSS_ARM_OFFSET_PCT`, `MAX_CROSS_ARM_MEAN_OFFSET_PCT`,
the stop-only summary shape (#1657), and the
tests/tools/test_gpu_clock_state.py mutation discipline.
Device surface: `tt-smi -s` JSON (tt-smi 6.2.1, pytuwen 0.9.0, UMD 0.9.9).

## Design decisions

- **Busy proxy**: a sample is busy iff a caller-supplied pid
  (`--leg-pid`) is alive AND holds an open fd on `/dev/tenstorrent/*`.
  The leg prints its own generate bounds (#1671); a sampler told which pid
  to watch cannot orphan silently beyond the busy-fraction rule.
- **Claimed max clock**: explicit `--claimed-max-aiclk-mhz` argument plus a
  provenance string, because no tt-smi command exposes it. Never guessed.
- **NOT APPLICABLE fields** (applications cap, persistence mode, live
  throttle bitmap): emitted as null with reason strings, never dropped
  silently. Context telemetry (VCORE/TDP/TDC/temps/fan) recorded per sample
  fold as medians, reported, not gated on in v1.
- **Throttle-unobservability caveat**: every judge output carries it until a
  live throttle signal exists upstream.
- Board identity and firmware/KMD versions are compared UNCONDITIONALLY,
  like the helper compares driver/clock-capability fields across a waived
  boot.

## Risks

- AICLK granularity may quantize below the 5%/1% band edges on boards whose
  governor steps coarsely; thresholds are copied, not re-tuned, so a
  real-band excursion reads as refusal rather than silent pass — acceptable,
  refusal is never wrong about evidence.
- PID-fd race at leg exit can flip the last samples idle; majority-busy rule
  absorbs it by construction.
- Subprocess startup cost caps the practical cadence at ~2 Hz; documented.

## Tests

`tests/tools/test_tt_clock_state.py`, synthetic fixtures, no device needed:
fold math, spread boundary, median offset boundary, mean offset boundary,
busy floor, busy fraction floor, idle-window refusal writes evidence,
boot mismatch refuses + waiver stamps exactly the boot term, cross-board /
cross-firmware / cross-KMD refuse unconditionally under the waiver,
throttle-unobservability caveat present in every summary. Each gate constant
mutated red.

## Gates

Suite green; preflight; then the wired leg (one lock hold): JIT warmup proc,
sampler around each arm (--repeat 5, run 1 discarded, order-alternated
pairs x3), judge per pair, results appended to `.agents/benchmark-record.md`
as the clock-attributed re-measure of #2003, `docs/benchmarks/open-gaps.md`
host-free row updated to attribute or stand down accordingly, comment left
on #2003.

## Stop conditions

If live AICLK proves flat-stuck (governor immovable) the tool still lands
(records the fact) but #2003 stays open with the hybrid-delta step unchanged;
if snapshot latency regresses past ~2 s the wired leg falls back to manual
start/stop windows, noted in the record entry.

## Git integration

One PR: this spec committed first, then tool + tests, then the wired-leg
record changes. Branch `bench/tt-clock-state`.

## Evidence

Wired leg, 2026-08-26 P150 thalia (see
[`../../docs/bench-evidence/tt-p150-clock-attributed-20260826.log`](../../docs/bench-evidence/tt-p150-clock-attributed-20260826.log)):
six per-arm windows around order-alternated #2003 legs. Suite
`tests/tools/test_tt_clock_state.py` 16/16 before and after the leg.
Two findings recorded rather than smoothed over:

1. **The governor is two-state.** Every raw window mixes an ~14-sample idle
   head at 800 with pegged-compute samples: spread 40.74% > 5%, judge
   rc=1 on all six. The refusal is correct under rules written for
   quasi-continuous clocks and is KEPT — it forced the platform question
   this spec now answers explicitly instead of hiding it behind a loosened
   threshold.
2. **Busy-slice refold attributes cleanly.** Because busy was recorded live
   per interval from pid-held device fds (criterion independent of the
   outcome clock values), `tt_refold_busy.py` rebuilds busy-only records:
   distinct AICLK set {1350} in all six windows, spread 0.00%, cross-arm
   offsets 0/0%, judge PASS rc=0.

Throughput re-measured same session: default median 10.880 vs opt-out
median 13.645 tok/s (ratio 1.254) — #2003 stands as a real path difference.

## Owed

- **Verified claimed-max pin**: `1350` is class folklore passed as CLI
  provenance "UNVERIFIED"; pin from vendor docs or UMD range readout (#2005).
- **In-process pyluwen sampling** for sub-second cadence; subprocess startup
  caps practical rate near 2 Hz today.
- **Policy decision owed upstream of any threshold change**: whether a
  two-state governor earns a spread rule scoped to busy slices inside the
  tool itself (a `--spread-scope busy` flag) or stays an offline refold;
  changing it means a red-before mutation here, never a silent loosening.
