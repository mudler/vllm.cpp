# Benchmarking protocol — match or beat vLLM on EVERY axis (never below)

**Acceptance rule (non-negotiable):** for any change *and* for the MVP throughput
gate, vllm.cpp must be **equal to or better than vLLM on EVERY measured axis**.
Below vLLM on any axis is **NOT accepted** — it is an open gap to close, and the
MVP is not met until every axis is at parity or above. "Near parity" (0.9×,
0.98×) does **not** count. Equal-or-above, never below.

## The axes — measure ALL of them, BOTH gate models, at the operating point

vLLM is the ground truth on every one. Re-measure vLLM on the IDENTICAL workload
each time (its baselines drift — 35B 2768→3145, 27B 397→452 across re-measures);
never compare against a stale denominator.

**Higher-is-better — ours must be ≥ vLLM:**
- Total token throughput (tok/s) — the primary gate axis.
- Output/generation throughput (tok/s).
- Requests/s (req/s).

**Lower-is-better — ours must be ≤ vLLM:**
- TTFT (time-to-first-token = prefill latency): mean, median, P99.
- TPOT / ITL (inter-token latency = decode latency): mean, median, P99.
- Peak memory footprint (must fit and not exceed vLLM's at the same workload).

**Precondition — NOT an axis you may trade off:**
- Correctness: token-for-token vs the pinned vLLM oracle (greedy 16/16) +
  chunked-prefill `max|diff|=0`. You may **never** "win" a throughput/latency
  number with divergent output — a metric measured on wrong tokens is void.

## How to run it (every change that could affect perf, and every gate check)

1. **Re-measure the vLLM denominator** on the IDENTICAL workload (`vllm bench
   throughput` / `bench serve`, `~/venvs/vllm-oracle`) — same model, in/out len,
   concurrency, sampling. Its number drifts; a stale baseline is disqualifying.
2. **Same-binary A/B** for our change (toggle ON vs OFF) to isolate its effect,
   then compare the ON arm vs the fresh vLLM number on every axis.
3. **Both gate models** (Qwen3.6-35B-A3B-NVFP4 and 27B-NVFP4), each at its
   **large-concurrency operating point** (the gate is *large concurrency*; a
   single-wave `np == concurrency` run hides sustained-load amortization — use
   sustained load, and size `--num-blocks` so the KV cache is not starved).
4. **Record every axis + the ratio** in [parity-ledger.md](parity-ledger.md).
   A ledger row that leaves ANY axis below vLLM is an OPEN GAP, not a done change.

## Reproduction is a GATE

A benchmark number is **not accepted until it reproduces**. A single run is a
rumor; a reproducible run is evidence.

- **Record the exact repro recipe** with every result (in the ledger): the
  commit SHA, the full `vllm-bench`/`vllm bench` command (model snapshot path,
  in/out len, num-prompts, concurrency, num-blocks, seed, sampling), the build
  flags/arch, and the vLLM oracle command + version used for the denominator.
  Someone must be able to paste it and get the same number.
- **Re-run to confirm.** A claimed win or a gate number must be reproduced —
  re-run it (≥2–3×) and confirm it lands within run-noise (this campaign saw
  ±2–4%; report the spread). A number that doesn't reproduce is void, not a win.
- **Same-binary A/B** (toggle ON vs OFF) so the delta is attributable to the
  change, not to build/config drift.
- **Guard against contamination** — the box must be idle for the measurement:
  no concurrent build/bench/other-process on the GPU (verify `ps` + `free -g`);
  a contended run inflates latency and is void (we lost a dense-fp8 A/B this way
  and had to re-run). Kill strays with `pkill -9 -x vllm-bench` before each run.
- **Prefer a scripted, re-runnable harness** over ad-hoc one-offs so any result
  can be regenerated on demand; keep the gate config stable (never silently
  re-base it without re-running vLLM on the new config).

If a result cannot be reproduced on demand under these rules, it does not count
toward the gate.

## Acceptance / gate decision

- A change is a WIN only if it moves an axis toward-or-past vLLM **without
  regressing any other axis below vLLM** (or below our own prior best).
- The MVP throughput gate is MET only when, for BOTH models at large concurrency:
  total throughput ≥ vLLM (≥ 1.0×) **AND** TTFT ≤ vLLM **AND** TPOT ≤ vLLM
  **AND** peak memory ≤ vLLM **AND** correctness holds — i.e. **match-or-beat on
  every axis, both models**. Anything below vLLM on any axis keeps the gate open.

See also: [gates.md](gates.md) (the MVP gates), [parity-lever-protocol.md](parity-lever-protocol.md)
(how to find the levers to close a below-vLLM axis).
