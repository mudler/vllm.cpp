# Benchmarking protocol — match or beat every applicable floor on every axis

**Acceptance rule (non-negotiable):** vLLM remains the compatibility and CUDA
performance floor wherever it supports the same workload. Each backend also has
a native performance floor and a leader comparison below. vllm.cpp must be
**equal to or better than every applicable floor on EVERY measured axis**.
Below a floor on any axis is an open gap. "Near parity" (0.9×, 0.98×) does not
count. A leader that is not yet a binding floor is still measured and tracked;
it becomes binding once the spike proves the workload/features are equivalent.

## Backend-native floors and leader comparisons

| Backend / operating point | Compatibility oracle | Binding performance floor | Leader comparison |
|---|---|---|---|
| NVIDIA CUDA, large concurrency | pinned vLLM | production graphed vLLM | SGLang when it supports the same model/quant |
| NVIDIA CUDA, low concurrency 1/2/4/8/16 | pinned vLLM | better of production vLLM and equivalent SGLang | both are always reported |
| CPU + GGUF | pinned vLLM behavior where shared; same-file llama.cpp tokens/logits | same-file llama.cpp | strongest current CPU engine found by the spike |
| Vulkan | vLLM behavior where shared | same-model/quant llama.cpp Vulkan | other equivalent native Vulkan engine if identified |
| Apple MLX/Metal | pinned vLLM behavior + MLX primitive/MLX-LM numerics | equivalent oMLX or MLX-LM result, whichever is better per axis | llama.cpp Metal also reported when it accepts the same file |
| Intel XPU / ROCm | pinned vLLM on that platform | production vLLM platform backend | strongest equivalent native engine identified by the backend spike |

"Equivalent" means the same model weights/quantization, prompt tokens, input and
output lengths, sampling, cache/prefix state, concurrency and serving features.
If conversion is necessary, record it and do not turn the converted result into
a binding floor until correctness/quality equivalence is established. Full
matrix: [specs/competitive-benchmarks.md](specs/competitive-benchmarks.md).

## The denominator is vLLM's PRODUCTION config (CUDA graphs ON) — never `--enforce-eager`

**The bar is vLLM's REAL/production throughput, which uses CUDA graphs**
(piecewise/full cudagraph + torch.compile). `vllm bench throughput` with
`--enforce-eager` measures a HANDICAPPED vLLM (no graphs) and is **NOT** the bar.
MEASURED 2026-07-07 (27B, in1024/out128, conc-16, np96, GM0.6, same box):
**vLLM graphed 758.91 vs vLLM eager 634.28 = +19.6%.** The `--enforce-eager`
baseline was an obsolete near-term crutch from when our own engine was eager;
now that our engine has CUDA graphs, the honest, apples-to-apples denominator is
**graphed vLLM** (drop `--enforce-eager`). Match-or-beat is measured against THAT.
(Subtlety worth noting, not an excuse: much of vLLM's eager→graphed jump is
Python per-op dispatch elimination that our C++ never paid — indeed our C++ eager
≈ vLLM eager — so the *real* target inside the +19.6% is vLLM's graph-hidden +
tuned PREFILL/compute kernels, which IS closable, not the Python tax.)

## The axes — measure ALL of them, BOTH gate models, at the operating point

vLLM is the ground truth on every one. Re-measure vLLM on the IDENTICAL workload
each time (its baselines drift — 35B 2768→3145, 27B 397→452 across re-measures);
never compare against a stale denominator. **And measure vLLM in its production
CUDA-graph config (see above), not `--enforce-eager`.**

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

**Feature/milestone checkpoint — do not stack unbenchmarked speed-sensitive
changes.** Each feature or milestone that can plausibly change runtime speed,
latency, launch structure, memory traffic, scheduling, batching, loading, or
peak memory is its own benchmark checkpoint. Finish and record its isolated
same-binary A/B plus the fresh applicable floor before layering the next such
change on top. If two changes are technically inseparable, the committed spike
must declare the combined checkpoint and explain why they cannot be toggled or
measured separately. Missing hardware leaves the row in `GATING` with an exact
handoff recipe; it never converts an unmeasured milestone into `DONE`.

Every feature/iteration checkpoint also updates the public
[`README.md`](../README.md) stage and
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) disposition **in the same
change**. This includes an attempted benchmark that fails closed: record it as
`FAILED` or `VOID`, name why no ratio is binding, and retain the exact next
reproduction command. A CPU-only/non-performance feature records
`NOT APPLICABLE` plus its correctness gates; unavailable hardware records
`PENDING`. The append-only ledger remains the detailed evidence source, while
`docs/BENCHMARKS.md` is the concise user-facing scoreboard and current
checkpoint.

1. **Re-measure the vLLM denominator** on the IDENTICAL workload (`vllm bench
   throughput` / `bench serve`, `~/venvs/vllm-oracle`) — same model, in/out len,
   concurrency, sampling. Its number drifts; a stale baseline is disqualifying.
2. **Same-binary A/B** for our change (toggle ON vs OFF) to isolate its effect,
   then compare the ON arm vs the fresh vLLM number on every axis.
3. **Both gate models** (Qwen3.6-35B-A3B-NVFP4 and 27B-NVFP4), each at its
   **large-concurrency operating point** (the gate is *large concurrency*; a
   single-wave `np == concurrency` run hides sustained-load amortization — use
   sustained load, and size `--num-blocks` so the KV cache is not starved).
4. **Run the applicable native/leader arms** from the table above. For CUDA
   serving, include the low-concurrency sweep when latency/scheduling could
   change. For new backends, the area spike fixes representative models that fit
   the hardware without pretending a small-model gate proves 27B/35B scale.
5. **Record every axis + every ratio** in [parity-ledger.md](parity-ledger.md),
   then refresh the concise accepted/pending/failed/void summary in
   [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) and the current stage in
   [`README.md`](../README.md). A ledger row that leaves ANY axis below an
   applicable floor is an open gap.

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
- **Guard against contamination with the shared mutex** — wrap the complete
  server/reference/A/B/profile series in one `flock /tmp/gpu -c '...'`, per
  `/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md`. A number is void
  unless the lock was held for its entire run. Inspect `fuser -v /tmp/gpu` on a
  timeout; never kill a PID you did not start. CPU-only comparisons use an
  equivalent host lock or an otherwise idle, pinned-core machine.
- **Prefer a scripted, re-runnable harness** over ad-hoc one-offs so any result
  can be regenerated on demand; keep the gate config stable (never silently
  re-base it without re-running vLLM on the new config).
- **Thermal/power + memory-return disclosure for multi-arm series** (adopted
  from spark-bench `matrix_driver.sh`; GB10 is a compact unified-memory box):
  snapshot `nvidia-smi -q -d TEMPERATURE,POWER` before/after each measured leg
  and record it with the results; between engine legs, verify available memory
  returns to the recorded baseline (leak check) and drop page caches before the
  next leg. A leg run on a throttling or memory-leaking box voids the
  cross-leg comparison, same as contention. On a non-root benchmark host, the
  complete enumerated set of benchmark-owned checkpoint, corpus, client and
  server files may instead be evicted with `POSIX_FADV_DONTNEED` only when a
  retained report hashes that inventory and `mincore(2)` proves **zero resident
  pages** afterward; an unverified/best-effort advisory call does not qualify.
  See [specs/competitive-benchmarks.md](specs/competitive-benchmarks.md)
  § "Folded: spark-bench".

If a result cannot be reproduced on demand under these rules, it does not count
toward the gate.

## Acceptance / gate decision

- A change is a WIN only if it moves an axis toward-or-past vLLM **without
  regressing any other axis below vLLM** (or below our own prior best).
- A speed-sensitive feature or milestone cannot enter `DONE` on aggregate
  release evidence alone: its own checkpoint must identify its attributable
  delta, fresh floor, run spread, commands, and all applicable axes.
- The MVP throughput gate is MET only when, for BOTH models at large concurrency:
  total throughput ≥ vLLM (≥ 1.0×) **AND** TTFT ≤ vLLM **AND** TPOT ≤ vLLM
  **AND** peak memory ≤ vLLM **AND** correctness holds — i.e. **match-or-beat on
  every axis, both models**. Post-MVP backend rows additionally remain open while
  below their applicable native floor.

See also: [gates.md](gates.md) (the MVP gates), [parity-lever-protocol.md](parity-lever-protocol.md)
(how to find the levers to close a below-vLLM axis).
