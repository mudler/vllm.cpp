# Spike: CUDA online serving gate

**Row:** `SERVE-GATE-ONLINE` · **state:** accepted; execution is `ACTIVE`.
The first campaign under `~/work/vllm.cpp-latency/latres` is diagnostic only:
both vLLM model starts failed and several 35B local arms aborted, so it supplies
no denominator or release number. The second pre-W2 campaign supplies a complete
27B diagnostic, but its 35B local arm completed only 1/6 c1 requests before a
CUDA illegal-memory-access abort. Its vLLM arm completed the greedy ladder and
default c1, then hung for more than 60 minutes in default c16/np96 rep1 with 16
open sockets and no log progress. The owned campaign was snapshotted/hashed at
DGX `latres2/diagnostic-vllm35-hang-20260711T0734CEST`, terminated, and released
the GPU lock. Neither pre-W2 campaign is gate evidence.

The post-W2 `31d053f` 27B grid completed 2,016 exact-count requests and all
memory-return checks, but its 0.926–0.961× ratios remain void. The repeated ours
trace became a prefix-cache hit while vLLM caching was off; the profiler also
preloaded all prompts, used max-seqs 16, and forced max model length 1152 rather
than the production contract. `ed6247d` subsequently captured a complete exact
vLLM c16/48 trace (kernel summary SHA-256 `8213…df64`), but it has no comparable
ours arm. The historical offline denominators are also reopened: pinned
`vllm bench throughput` used temperature 1 while ours used temperature 0, and
the 27B budgets were 8192 versus 2048.

The 2026-07-12 vLLM release audit invalidates every old-oracle denominator.
Clean `b5c6e4f` remains useful historical W1/W2/W3 diagnosis, but it executed
vLLM 0.24.0 with FlashInfer 0.6.12 while the source pin and v0.25.0 target use
0.6.13. Its c1→c32 totals
0.9933/0.9520/0.9657/0.9760/1.0213/1.0218× and all latency/memory ratios are
non-binding. The replacement clean `3cc490c` campaign was deliberately stopped
and is **VOID** at 28/36 groups, 1,602/2,016 requests, four memory returns and
no paired trace; no partial rate is reusable. Its owned process group exited
and the GPU/ports/lock returned idle.

The executable benchmark contract now pins pip vLLM 0.25.0 at tag commit
`702f481`, with FlashInfer 0.6.13. The isolated staging environment has passed
package/version/import/CLI checks except for a documented NVIDIA
cuSPARSELt-wheel tag defect: PyPI supplied the correct aarch64 binary and direct
load succeeds, while the wheel's internal `manylinux2014_sbsa` tag makes
`pip check` report unsupported. A lock-held real 27B production-graph offline
smoke loads the 24.57-GiB checkpoint, compiles, autotunes FlashInfer, captures
graphs and completes exact 16-input/1-output counts; its cold rate is not a
benchmark. Clean server smoke and atomic oracle-path activation also pass.

The replacement campaign is immutable clean `9cc7191`, with source/build at
`~/work/vllm.cpp-online-gate/checkpoints/9cc71918dbdc10f014c02feb9bab1d00963a16fe`
and evidence at the matching `evidence/` path. Plan/oracle/build-log/source-
corpus/vLLM-corpus SHA-256 are `5a04cdcf…b2`, `6d39cb90…10c`,
`10786029…6a`, `41bd634a…7a` and `b048d789…5dc`; server/model-gate binaries
are `ffddab5f…bd` / `a24fc776…37`. The first no-GPU metadata recorder command
failed before output because direct script invocation omitted the repository
module path. Its corrected `python -m tools.bench.online_gate` invocation,
plan validation, exact-source check, corpus conversion and fresh sm_121a build
pass. No model gate or timed request has run. The full 27B c1-c32 grid plus
paired traces is `ACTIVE` and will own one lock for the entire series. Every
throughput, latency and memory axis must pass before any 35B performance run.

## Scope

Compare the production HTTP server with pinned vLLM on both gate checkpoints at
concurrency 1, 2, 4, 8, 16 and 32 (the large-concurrency gate point). Both
servers run with `max_num_seqs=32` and identical per-model token budgets
(27B=2048, 35B=8192), production model length 262144, and explicit prefix
caching off. Record request
rate, input/output/total throughput, TTFT, TPOT, ITL, end-to-end latency, errors,
and peak host/GPU memory. The same prompts, tokenizer output, arrival policy,
sampling parameters, warmup, and request count are mandatory.
Each timed invocation first issues one full-concurrency warmup wave so lazy
capture/JIT work cannot contaminate the reported request set.
The paired profiler keeps only 16 requests resident and admits one replacement
per completion, matching the online closed-loop client instead of preloading all
48 prompts.

## Upstream chain

- CLI and benchmark orchestration:
  `/home/mudler/_git/vllm/vllm/entrypoints/cli/benchmark/serve.py:1` and
  `vllm/benchmarks/serve.py:1`.
- Pinned vLLM computes `max_concurrent_requests` with inclusive integer-second
  buckets at `vllm/benchmarks/serve.py:656-706`; adjacent sequential requests
  can therefore both occupy one bucket. Binding saturation uses the retained
  detailed half-open intervals instead.
- CLI test contract: `tests/benchmarks/test_serve_cli.py:1`.
- Production serving path: `vllm/entrypoints/openai/api_server.py` and the V1
  async engine/streaming path reached by that server.
- Actual CUDA dispatch is verified with an `nsys` trace of ours plus vLLM's
  mandated torch-profiler fallback (nsys breaks V1 EngineCore startup on GB10),
  per [parity-lever-protocol.md](../parity-lever-protocol.md).

## Our baseline

The server entry point is `examples/server/main.cpp:63-171`; the benchmark
metrics implementation is `examples/bench/bench_core.h:96-468` with unit
coverage at `tests/examples/test_bench.cpp:15-48`. Offline throughput gates
exist, but they do not prove online queueing, streaming, connection handling,
or TTFT/ITL parity. `tools/bench/online_gate.py` now constructs the unmodified
vLLM v0.25.0 oracle command audited against target `702f481`, freezes
exact-token corpus views and rejects partial results while accepting pinned-
vLLM's producer-ahead DELTA aggregation;
`online_gate_summary.py` makes any missing/mismatched
model, stream, memory-return, trace or every-axis artifact non-binding. The
failed pre-W2 campaigns remain diagnostics; partial rows are never reused as
successful repetitions. W0 of `KV-PREFIX-CACHE` now supplies the pinned hybrid
default-off policy and an explicit server override. The trace contract rejects
cache-policy, admission, max-sequence, model-length, or repeated-duration drift.

## Port map

| vLLM surface | Local surface |
|---|---|
| server startup/readiness | `examples/server/main.cpp` plus `/health` probe |
| OpenAI request/SSE stream | `src/vllm/entrypoints/openai/` |
| async scheduling/stream updates | current engine/server bridge; gaps map to `SERVE-ASYNC-LLM` |
| scheduler/cache operating point | explicit server `max_num_seqs`, `max_num_batched_tokens`, and `--[no-]enable-prefix-caching` flags; harness fixes identical values on both arms and records the resolved policy |
| benchmark request builder and metrics | staged vLLM 0.25.0 + FlashInfer 0.6.13 `vllm bench serve` command audited against target `702f481` + schema validation in `tools/bench/online_gate.py`; aggregation only in `tools/bench/online_gate_summary.py`; the canonical oracle path changes only after fingerprint/model/server validation |
| exact corpus | `tools/bench/make_serve_low_corpus.py` source corpus via the dry-run-recorded `<pinned-vLLM-bin>/python -m tools.bench.make_serve_low_corpus` command + hash-preserving vLLM CustomDataset view in `online_gate.py` |
| build/oracle provenance | clean exact-HEAD CMake refresh + hashed command/log/binary; hashed pip launcher, Python, venv `ninja`, benchmark modules, dist metadata/RECORD, plus exact pandas 2.2.3 runtime/package/METADATA/RECORD preflight in `online_gate.py` |
| lifecycle/resources | `scripts/dgx-online-serving.sh`; process-tree/GPU sampling in `tools/bench/sample_process_memory.py`; rootless enumerated `POSIX_FADV_DONTNEED` + `mincore` proof in `tools/bench/drop_file_cache.py` |
| GPU/runtime trace | ours under `nsys`; vLLM LLM-API torch profile via the closed-loop `tools/bench/profile_vllm_online_gate.py`; metadata fixes c16 admission, max-seqs 32, production model length, cache-off, corpus hash, and three repetitions; kernel-event aggregation in `summarize_torch_kernels.py` |

## Tests to port

- `tests/benchmarks/test_serve_cli.py`: argument validation and command wiring,
  ported to `tests/tools/test_online_gate_client.py`.
- The local server's explicit scheduler-capacity CLI is build/run-gated by
  `test_server_help` in `examples/CMakeLists.txt`.
- `tests/benchmarks/test_custom_dataset_seed.py`: deterministic custom-corpus
  selection, ported to the same client/corpus suite; that suite also executes
  the planned pinned-interpreter/module prefix and executes its module suffix so
  an ambient or unimportable clean-shell preparation command fails before GPU
  work.
- Pinned profiler example/API behavior is covered by
  `tests/tools/test_online_gate_trace.py`; output-repeatability metadata and
  project every-axis/void propagation are covered by
  `tests/tools/test_online_gate_{client,summary}.py`.
- The profiler test drives a fake engine through closed-loop replacement
  admission and proves the resident request count never exceeds c16. Client and
  summary cases reject missing cache-off flags or drifted cache/admission/
  max-sequence/model-length metadata; loaded-engine/coordinator tests cover the
  underlying model-default no-prefix path.
- Rootless eviction inventory, inode deduplication, zero-residency proof,
  failure reporting and overwrite refusal are covered by
  `tests/tools/test_drop_file_cache.py`; client/summary suites reject resident
  or missing cache reports.
- The oracle-manifest contract rejects missing or drifted pandas before any
  build or GPU lock. The version comes from pinned vLLM's CUDA test requirement
  (`requirements/test/cuda.txt:742`), while `setup.py:1247` declares pandas as a
  `bench` extra.
- The same manifest hashes and requires executable `${oracle_venv}/bin/ninja`
  before any GPU lock. The vLLM profiler prepends that venv to `PATH` because
  FlashInfer may JIT an uncached FP4 module in a spawned EngineCore process; an
  importable Python package alone does not make its executable visible.
- Client contracts cover vLLM's bucket-boundary false overlap: the upstream
  bucketed peak is retained, while exact start + TTFT + ITL intervals must reach
  the configured concurrency and use end-before-start ordering at ties.
- Client contracts also cover producer-ahead DELTA aggregation. Pinned vLLM's
  `RequestOutputCollector` merges queued DELTA outputs, and its benchmark client
  explicitly counts output tokens from native usage because one streamed choice
  may contain multiple tokens. Exact native output counts remain mandatory;
  retained ITLs are inter-chunk timings and may therefore number fewer than
  `output_len - 1`. More timing events than that bound remain invalid.
- Applicable `tests/entrypoints/openai/` streaming, disconnect, usage, and error
  cases before relying on HTTP measurements.
- Local conformance and benchmark tests remain prerequisites. Any upstream case
  blocked by `SERVE-ASYNC-LLM` lands skipped with this row ID and the dependency.

## Gates

One `flock /tmp/gpu` per model holds its model gate, server start, readiness,
warmup, all interleaved repetitions, shutdown, memory sampling, and paired
traces. The c32 client point is backed by an explicit 32-sequence scheduler on
both arms rather than the local server's historical default of eight. Both
servers explicitly disable prefix caching, and the paired trace uses closed-loop
c16 admission, max-seqs 32, production max model length, and three cache-off
repetitions whose durations differ by no more than 20%. Each
model/point has
at least three valid repetitions and a fresh pinned-vLLM denominator. The
commit-bound model gate is the correctness precondition before performance is
compared; every timed request separately requires exact native prompt/output
counts. Generated texts and profiler output digests are retained as diagnostics,
not an equality gate: production FP4 accumulation variants can choose different
greedy near-ties, including between vLLM warmup and measured repetitions. A
text mismatch may not hide a failed model gate, partial request, error, or count
drift. vllm.cpp must be no worse on every throughput,
latency, error, and memory axis; a failed start or request voids that arm. The
configured concurrency must be reached by an exact sweep of each request's
observed half-open `[start_time, start_time + ttft + sum(itls))` interval.
Pinned vLLM's
inclusive one-second `max_concurrent_requests` remains hashed diagnostic data
but is not a saturation oracle because it can overcount sequential boundaries.
The committed summary exits nonzero unless every detailed result, generated-text
diagnostic pair/count, stream probe, process/GPU-memory sample, thermal snapshot, cache/
memory-return record, clean-HEAD build provenance, exact pip-oracle runtime
inventory, model gate and paired trace is present and hash-valid.

DGX does not delegate global `/proc/sys/vm/drop_caches` to the benchmark user.
The gate therefore enumerates every regular checkpoint, corpus, client and
server file, deduplicates by inode, calls `POSIX_FADV_DONTNEED`, and uses
`mincore(2)` to require zero resident pages. Before/after reports for every leg
and before/between/after reports for paired traces retain the inventory hash,
logical byte count and file-level residency; the summary re-hashes and validates
each raw report. A successful advisory call without zero-residency proof is
still fatal. A warmed real 27B probe covered 49 files / 26.55 GB, observed
1,199,611,904 resident bytes before, and reached zero afterward (report SHA-256
`21bbcc7594a661d8ce22979f6f7009f2fb8e02b0ad2ee02d297373ee14320069`).

## Dependencies

`SERVE-OAI-BASIC`, `SERVE-CLI-BENCH`, sampler correctness, both model/quant
gates, and the benchmark protocol are prerequisites. `SERVE-ASYNC-LLM` is an
explicit dependency wherever the synchronous bridge prevents equivalent
low-concurrency overlap or streaming behavior.

The executable oracle environment must include pandas 2.2.3 and an executable
`ninja` in the same venv `bin` directory. On 2026-07-12 the canonical
`~/venvs/vllm-oracle` was promoted to validated vLLM 0.25.0/FlashInfer 0.6.13
with pandas 2.2.3, python-dateutil 2.9.0.post0, pytz 2024.2 and tzdata 2024.2;
v0.24.0 is preserved under its retired rollback path. Only pandas participates
directly in the CustomDataset path. Both pandas and the profiler's Ninja
executable are independently file-hashed by the gate, and the profiler receives
the venv-prefixed `PATH`. The sole `pip check` warning is a disclosed NVIDIA
cuSPARSELt internal `sbsa` wheel-tag defect; the installed wheel URL is aarch64,
the ELF is AArch64, direct library load succeeds, and this exception is never
reported as a clean dependency check.

## Work breakdown

1. Reproduce each failed startup outside measurement and record the root cause.
2. Freeze the tokenized request corpus and machine-readable result schema
   (**implemented and CPU-gated**).
3. Add readiness/failure guards that abort the whole locked series on bad arms
   (**implemented and CPU-gated**).
4. Run interleaved vllm.cpp/vLLM repetitions for both models and all points with
   explicit cache-off, identical sampling, scheduler settings, and model length.
   The immutable `9cc7191` W3-B 27B source/build/corpus preflight is complete;
   execute its model gate, 36 timed groups and six returns next. Hold 35B.
5. Capture one representative paired execution trace per model (`nsys` ours,
   torch-profiler vLLM on the identical 48-prompt/c16 token shape). The prior
   old-oracle W3-B trace is lifecycle-clean and diagnostic; the binding
   `9cc7191` v0.25 trace remains part of the active series. 35B remains gated.
6. Drive the top traced lever through its owning row. W3-B closes the original
   wide FP4 tactic-family mismatch. Re-rank only after the exact W3-B grid shows
   which decode/latency axes remain below floor; do not infer a lever from the
   profiled total-throughput win while normalized TPOT is still 0.9673x.
7. Append commands, raw artifact hashes, results, and ratios to the ledger.

Claims may split diagnosis/client hardening from execution, but only one claim
owns the DGX campaign and its result directory.

## Risks and decisions

- Offline and online throughput are different gates, but both require identical
  sampling and scheduler/cache configuration within their own comparison. The
  historical direct-library values fail that rule and are non-binding.
- A server that never becomes model-ready is not a zero-throughput result; the
  comparison is invalid and must be rerun.
- Provisioning, compilation, downloads, and cache warmup stay outside the
  measurement lock window.
- The exact same checkpoint is required; converted weights are non-binding.
- A profiler dependency that is importable but not executable through the
  spawned EngineCore `PATH` is a failed preflight, not a reason to omit the
  paired trace or reuse a lock-released series.
- Cross-engine generated-text equality is not a substitute for the model gate.
  The frozen synthetic continuations are performance load, and direct evidence
  showed FP4 variants select different valid branches; retain exact-match counts
  and all profiler digests so repeatability remains visible without voiding an
  otherwise exact-count, correctness-preconditioned performance arm.
- Logical input-token counts do not expose prefix hits. Explicit cache policy
  and stable repeated duration are mandatory evidence; a cache-hit ours trace
  cannot be compared with cache-off vLLM.
- The pre-W2 server's `cudaFree` error is an asynchronous surfacing point, not
  a proven faulting kernel; reproduce current main under sanitizer before any
  fix or root-cause claim.
