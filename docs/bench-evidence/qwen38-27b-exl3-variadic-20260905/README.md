# `qwen38-27b-exl3-variadic-20260905` — evidence

The variadic serving-load run behind
[`docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md`](../../benchmarks/qwen38-27b-exl3-variadic-gb10.md).
Its method is
[`docs/benchmarks/variadic-load-methodology.md`](../../benchmarks/variadic-load-methodology.md)
and its design is
[`.agents/specs/bench-qwen38-exl3-variadic.md`](../../../.agents/specs/bench-qwen38-exl3-variadic.md).
Issue [#2970](https://github.com/mudler/vllm.cpp/issues/2970).

## What is here

| File | What it is |
|---|---|
| `job-as-run.sh` | the job as submitted to `rc`, byte-identical to `benchmarks/variadic/job.sh` at the measured commit |
| `results.txt` | the job's own `RESULT` lines: device, boot id, lease, checkpoint sha256 values, `G-BYTES`, `G-RESOLVED`, the build recipe, the paged-route decision, and each leg's client return code |
| `report.md` | the report as `benchmarks/variadic/report.py` printed it over the nine completed legs |
| `corpus-manifest.json` | the prompt corpus: every source sha256, the seed, the band weights, the realised character lengths, and the corpus's own sha256 |
| `corpus-token-histogram.md` | the corpus measured with the target checkpoint's own tokenizer, before the lease |
| `*.clientlog` | one per leg, the client's own stdout including its `CLIENT_RESULT` line |
| `mutation-controls.txt` | both rounds of mutations run against `tests/scripts/test_variadic_harness.py`, and what each one broke |
| `numpy-percentile-crosscheck.txt` | 27,000 pairs of `report.percentile` against `numpy.percentile` |
| `headtohead-recomputed-report.md` | the report over the PREDECESSOR's four legs, via `benchmarks/variadic/adapt_headtohead.py` |

The harness itself is committed at
[`benchmarks/variadic/`](../../../benchmarks/variadic) and is not duplicated
here. `job-as-run.sh` is kept because the job is the one file whose exact
submitted bytes a reader cannot recover from the branch alone.

The per-request JSON records stay on the share at
`/mnt/nas_share/rc/exl3-variadic/out/`, with the server logs. They are the input
to every number on the published page: each leg's file carries one record per
request with its band, dispatch time, time to first token, latency, chunk
arrival times, first-chunk character count, and the server's own `usage`. They
are not committed because they are several megabytes of per-token timing.

## Reproducing from these files

`benchmarks/variadic/report.py --dir <leg-json-directory>` recomputes every
table from the records. It needs no GPU, no lease and no numpy, so a correction
to a statistic costs nothing on the fleet.
