ID: ISSUE-GH-1604
Title: Flip VT_TT_HOST_FREE_DECODE on by default: 5x decode A/B on P150 (Qwen3-0.6B 27.1 vs 5.34 tok/s, Mistral-7B 12.2 vs 2.35 tok/s); both device golden pairs re-adjudicated in the same change
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: UNKNOWN
GitHub: 1604
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-22
Closed: 2026-08-22

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> Row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (spec `tenstorrent-host-free-forward.md`) ends env-gated. The row's own recorded bar for a default flip was: golden re-adjudication (done, #1488 / #1514) plus a same-binary A/B tok/s win. Both halves now exist.
>
> ## The A/B (2026-08-21, Blackhole P150, head `8399d6121` = #1514 tree)
>
> Same binary, same box, batch 1, greedy, `vllm-cli --repeat`, warm legs (leg 1 discarded —
> first-generation JIT, named cause), sequential under one GPU lock:
>
> | Model | Default (env unset) | `VT_TT_HOST_FREE_DECODE=1` | Ratio |
> |---|---|---|---|
> | Qwen3-0.6B | 5.34 tok/s (5.345/5.408/5.341) | **27.1** (27.376/27.128/26.755) | **5.1×** |
> | Mistral-7B-v0.3 | 2.35 tok/s (2.354/2.344) | **12.2** (12.230; 13.788 on a 57-tok stop leg) | **~5.2×** |
>
> Also measured: graph capture alone buys ~nothing on the default path (captured 5.34 vs
> eager `VLLM_CPP_CUDAGRAPH=0` 5.34) — the win is removing the host sync per decode step,
> not replay itself. Noise band ±0.5% (qwen) / ±2% (mistral). Far above the row's ≥12.5
> tok/s replay bar. Logs: `/tmp/perf-qwen-{captured,eager,hostfree}.log`,
> `/tmp/perf-mistral-{captured,hostfree}.log`.
>
> Usability on the same head: Qwen3-0.6B paged gate 16/16 (max gap 0.375 nats); Mistral-7B
> paged gate 16/16 (strict 12/16, max gap 0.062 nats, untied `kMatmul` lm_head on device).
> Both exit-139 after the green summary = pre-existing #1486 teardown.
>
> ## Scope of the flip (must all ride one change)
>
> 1. Default `VT_TT_HOST_FREE_DECODE` ON (env still able to turn it off for A/B).
> 2. **Both device golden pairs re-captured and re-adjudicated under the new default**: the
>    flip changes the default decode path, so the just-refreshed `qwen3_greedy_0_6b` TT pair
>    (#1514) AND the `mistral_greedy_7b` TT pair (captured 2026-08-12) both go stale.
>    `VT_DUMP_IDS` + `qwen3-neartie-gap-transformers.py` for each, same method as #1488.
> 3. Concurrency coverage beyond batch 1: today's A/B is batch 1. The spec's owed items
>    (`DecodePosCache` keyed on bare `num_reqs`; batch-size change after first capture is
>    refused) must be exercised by the async-serving / multi-seq gates under the new default
>    before it lands.
> 4. Records: `docs/BENCHMARKS.md` rows + `.agents/benchmark-record.md` entry with these
>    A/B legs; STATUS lifecycle if the row closes.
>
> ## Prerequisites
>
> - #1498 (the #1476 fix) — merged first.
> - #1514 (golden refresh + print fix) — merged second; the flip stacks on it.
>
> Owner: row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`.

## Resolution

GitHub records closing pull request #1630 (https://github.com/mudler/vllm.cpp/pull/1630) merged on 2026-08-22 as commit `333509dc5a788d167cbe34b5482fcfaa1e3c9e61`. GitHub closed issue #1604 on 2026-08-22.
