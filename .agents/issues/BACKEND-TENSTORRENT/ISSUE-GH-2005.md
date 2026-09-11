ID: ISSUE-GH-2005
Title: No Tenstorrent equivalent of gpu_clock_state.py: every TT figure on record is clock-unattributed, including #2003
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: perf
GitHub: 2005
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT`
>
> `.agents/benchmarking.md` ("The clock is part of the measurement") makes a
> number quotable only with the clock it was taken at, and
> `tools/bench/gpu_clock_state.py` implements that contract for NVIDIA:
> sample/stop-summary/judge, min/median/max over retained busy samples,
> >=30 retained samples, majority-busy fraction >=0.5, within-run spread <=5%,
> cross-arm median offset <=1% AND mean offset <=1%, same-boot
> (`--allow-cross-boot` waives exactly the boot id), throttle-reason triage,
> refuse-idle-window-with-evidence.
>
> There is NO Tenstorrent sibling, so every TT speed figure ever recorded —
> #1604's 10.94-vs-5.34 flip evidence and #2003's inversion (10.822 vs 13.369,
> 2026-08-26) — is labeled clock-unattributed. The #2003 headline could in
> principle be a clock excursion between alternated legs rather than a real
> path difference.
>
> Feasibility measured live on thalia/P150 today (tt-smi 6.2.1, pyluwen
> 0.9.0, tt_umd 0.9.9): `tt-smi -s` emits a JSON snapshot in **430 ms**
> including per-device `AICLK`/`ARCCLK`/`AXICLK`, `VCORE`/`TDP`/`TDC`,
> ASIC/GDDR temperatures, `FAN_RPM`, immutable board identity
> (`BOARD_ID_HIGH/LOW`) and firmware bundle versions — 1 Hz external sampling
> is directly supportable; importable `tt_smi` exists for tighter cadence later.
>
> Gaps vs the NVIDIA contract that a v1 must state, not paper over:
> - claimed-max clock: not exposed by any tt-smi command => explicit,
>   provenance-carrying argument.
> - applications/persistence knobs: none exist on TT => NOT APPLICABLE fields.
> - throttle reasons: no live bitmap (THM_LIMIT_*/VDD_LIMITS are static);
>   context telemetry recorded, throttle-unobservability documented.
> - busy/idle discrimination: no utilization counter; proxy = the measured leg
>   pid holding an fd on `/dev/tenstorrent/*`.
>
> First consumer: #2003 — re-adjudicate the inversion with per-arm clock
> windows before spending effort on the hybrid-path per-op delta.
>
> Plan: `tools/bench/tt_clock_state.py` mirroring the helper's CLI verbs,
> record schema and thresholds, rule-for-rule; mutation-tested sibling suite;
> one wired Qwen3-0.6B leg landing the re-measure. Owner row
> `BACKEND-TENSTORRENT` (spans all TT devices; spec-first).
>

## Resolution

GitHub records closing pull request #2368 (https://github.com/mudler/vllm.cpp/pull/2368) merged on 2026-08-31 as commit `511115a1bec95c21756e02fd320628ba4cea9ea3`. GitHub closed issue #2005 on 2026-08-31.
