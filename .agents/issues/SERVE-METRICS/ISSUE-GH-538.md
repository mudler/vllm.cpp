ID: ISSUE-GH-538
Title: Wire production AsyncLLM requests into Prometheus /metrics
Row: SERVE-METRICS
State: OPEN
Kind: UNKNOWN
GitHub: 538
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-12
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> The shipped `vllm-server --enable-metrics` exposes the Prometheus catalog, but production requests run through `AsyncLLM` while the logger is attached only to the unused synchronous `LLMEngine`. Live scrapes therefore leave `num_requests_running`, token counters, request-success counters, and timing histograms at zero even while requests complete.
>
> ## Reproduction
>
> Run four concurrent `/v1/chat/completions` requests against a server started with `--enable-metrics` and scrape `/metrics` at 100 ms cadence. Requests complete correctly, but all nine aligned samples report `num_requests_running=0`, `num_requests_waiting=0`, with token and success counters unchanged.
>
> ## Scope
>
> Complete W6 in `.agents/specs/prometheus-metrics.md`:
>
> - attach the existing `PrometheusStatLogger` to production `AsyncLLM`;
> - build and record `IterationStats` in the output-handler path;
> - make concurrent `Record()` / `Expose()` safe;
> - preserve the null-logger no-stats path;
> - add RED-first async metric and concurrent-scrape tests;
> - gate affected async/sync/Prometheus/server targets.
>
> No model kernels, sampling, scheduler policy, or Hermes client changes.

## Resolution

-
