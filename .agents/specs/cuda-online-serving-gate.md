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

## Scope

Compare the production HTTP server with pinned vLLM on both gate checkpoints at
concurrency 1, 2, 4, 8, 16 and 32 (the large-concurrency gate point). Both
servers run with `max_num_seqs=32` and identical per-model token budgets
(27B=2048, 35B=8192). Record request
rate, input/output/total throughput, TTFT, TPOT, ITL, end-to-end latency, errors,
and peak host/GPU memory. The same prompts, tokenizer output, arrival policy,
sampling parameters, warmup, and request count are mandatory.
Each timed invocation first issues one full-concurrency warmup wave so lazy
capture/JIT work cannot contaminate the reported request set.

## Upstream chain

- CLI and benchmark orchestration:
  `/home/mudler/_git/vllm/vllm/entrypoints/cli/benchmark/serve.py:1` and
  `vllm/benchmarks/serve.py:1`.
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
pip-vLLM 0.24.0 oracle command audited against the source pin, freezes
exact-token corpus views and rejects partial or bundled-stream results;
`online_gate_summary.py` makes any missing/mismatched
model, stream, memory-return, trace or every-axis artifact non-binding. The
failed pre-W2 campaigns remain diagnostics; partial rows are never reused as
successful repetitions.

## Port map

| vLLM surface | Local surface |
|---|---|
| server startup/readiness | `examples/server/main.cpp` plus `/health` probe |
| OpenAI request/SSE stream | `src/vllm/entrypoints/openai/` |
| async scheduling/stream updates | current engine/server bridge; gaps map to `SERVE-ASYNC-LLM` |
| scheduler operating point | explicit server `max_num_seqs` and `max_num_batched_tokens` flags; harness fixes identical values on both arms |
| benchmark request builder and metrics | pip-vLLM 0.24.0 `vllm bench serve` command audited against `e24d1b24` + schema validation in `tools/bench/online_gate.py`; aggregation only in `tools/bench/online_gate_summary.py` |
| exact corpus | `tools/bench/make_serve_low_corpus.py` source corpus via the dry-run-recorded `<pinned-vLLM-bin>/python -m tools.bench.make_serve_low_corpus` command + hash-preserving vLLM CustomDataset view in `online_gate.py` |
| build/oracle provenance | clean exact-HEAD CMake refresh + hashed command/log/binary; hashed pip launcher, Python, benchmark modules, dist metadata/RECORD, plus exact pandas 2.2.3 runtime/package/METADATA/RECORD preflight in `online_gate.py` |
| lifecycle/resources | `scripts/dgx-online-serving.sh`; process-tree/GPU sampling in `tools/bench/sample_process_memory.py`; rootless enumerated `POSIX_FADV_DONTNEED` + `mincore` proof in `tools/bench/drop_file_cache.py` |
| GPU/runtime trace | ours under `nsys`; vLLM LLM-API torch profile via `tools/bench/profile_vllm_online_gate.py`; kernel-event aggregation in `summarize_torch_kernels.py` |

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
  `tests/tools/test_online_gate_trace.py`; project every-axis/void propagation
  is covered by `tests/tools/test_online_gate_summary.py`.
- Rootless eviction inventory, inode deduplication, zero-residency proof,
  failure reporting and overwrite refusal are covered by
  `tests/tools/test_drop_file_cache.py`; client/summary suites reject resident
  or missing cache reports.
- The oracle-manifest contract rejects missing or drifted pandas before any
  build or GPU lock. The version comes from pinned vLLM's CUDA test requirement
  (`requirements/test/cuda.txt:742`), while `setup.py:1247` declares pandas as a
  `bench` extra.
- Applicable `tests/entrypoints/openai/` streaming, disconnect, usage, and error
  cases before relying on HTTP measurements.
- Local conformance and benchmark tests remain prerequisites. Any upstream case
  blocked by `SERVE-ASYNC-LLM` lands skipped with this row ID and the dependency.

## Gates

One `flock /tmp/gpu` per model holds its model gate, server start, readiness,
warmup, all interleaved repetitions, shutdown, memory sampling, and paired
traces. The c32 client point is backed by an explicit 32-sequence scheduler on
both arms rather than the local server's historical default of eight. Each
model/point has
at least three valid repetitions and a fresh pinned-vLLM denominator. Generated
tokens must pass the configured deterministic/logprob correctness gate before
performance is compared. vllm.cpp must be no worse on every throughput,
latency, error, and memory axis; a failed start or request voids that arm. The
committed summary exits nonzero unless every detailed result, exact generated
text pair, stream probe, process/GPU-memory sample, thermal snapshot, cache/
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

The executable oracle environment must include pandas 2.2.3. On 2026-07-11 the
isolated `~/venvs/vllm-oracle` was completed with pandas 2.2.3,
python-dateutil 2.9.0.post0, pytz 2024.2 and tzdata 2024.2; these match the
pinned CUDA test lock. Only pandas participates directly in the CustomDataset
path and is independently version/file-hashed by the gate.

## Work breakdown

1. Reproduce each failed startup outside measurement and record the root cause.
2. Freeze the tokenized request corpus and machine-readable result schema
   (**implemented and CPU-gated**).
3. Add readiness/failure guards that abort the whole locked series on bad arms
   (**implemented and CPU-gated**).
4. Run interleaved vllm.cpp/vLLM repetitions for both models and all points.
5. Capture one representative paired execution trace per model (`nsys` ours,
   torch-profiler vLLM on the identical 48-prompt/c16 token shape).
6. Append commands, raw artifact hashes, results, and ratios to the ledger.

Claims may split diagnosis/client hardening from execution, but only one claim
owns the DGX campaign and its result directory.

## Risks and decisions

- Offline and online throughput are different gates.
- A server that never becomes model-ready is not a zero-throughput result; the
  comparison is invalid and must be rerun.
- Provisioning, compilation, downloads, and cache warmup stay outside the
  measurement lock window.
- The exact same checkpoint is required; converted weights are non-binding.
- The pre-W2 server's `cudaFree` error is an asynchronous surfacing point, not
  a proven faulting kernel; reproduce current main under sanitizer before any
  fix or root-cause claim.
