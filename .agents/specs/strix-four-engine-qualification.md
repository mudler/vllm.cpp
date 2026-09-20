# Qwen3-4B four-engine qualification and measurement

Row: `BACKEND-GATE-ROCM-SGLANG`. Issue: [#3053](https://github.com/mudler/vllm.cpp/issues/3053).

This supplement implements Tasks 2 and 3 of [the approved benchmark](strix-four-engine-qwen3-4b.md).
Use a fresh implementer, an independent mutation reviewer, and operator verification.

## Scope and prerequisites

Keep all engine revisions and the checkpoint revision in the parent spec unchanged.
Use Qwen3-4B BF16, six raw prompts, 128 generated tokens, temperature zero, and top-p one.
Run concurrency one and four with four available scheduler slots in both cases.
The real conversion audit must pass before llama.cpp qualification.
The reviewed SGLang preparation, AITER import, and model startup must pass before accepting SGLang results.
Label that engine `patched SGLang` in every result.
No engine can be replaced or omitted from the report because it fails.

## Files and architecture

Create `tools/bench/strix_four_engine/qualify.py`, `python_adapter.py`, and `native_adapter.cpp`.
Create scoped tests under `tests/tools` for standard-library Python contracts and native scheduler fixtures.
An optional local CMake file may build the two native adapter targets; do not edit product build files.
Do not modify product APIs, engine implementations, converters, pins, or sampling semantics.
Adapters are thin clients of public APIs. C++ may include public `vllm.h` or `llama.h` and the existing nlohmann JSON header.
Never include internal engine headers. No serialized Python callbacks or insecure worker serialization.

The CLI is `python3 -m tools.bench.strix_four_engine.qualify --manifest INPUT.json --output UNUSED_DIRECTORY`.
The manifest binds source revisions, binary hashes, environment inventories, model hashes, and the passing conversion audit.
Verify bindings before and after execution. Refuse existing output directories.
The CLI does not download, install, rebuild, change pins, or select fallback backends.
Use explicitly provided adapter commands; reject shell command strings and run argv arrays without a shell.

## Inputs and engine configuration

Tokenize the six raw prompts with the pinned checkpoint tokenizer without a chat template.
Record the tokenizer files and each exact prompt-token array.
Pass those same arrays to every engine's token-input API and record the arrays actually consumed.
This proves consumed-input equality, not independent tokenization by vllm.cpp, which has no identified public tokenizer API.
Check llama.cpp tokenization separately through its public tokenizer and refuse mismatches.

| Engine | Required settings |
|---|---|
| vllm.cpp | Context 2048; four slots; prefix caching value 2, meaning disabled; BF16 KV; native execution defaults |
| vLLM | Model BF16; KV auto resolving to BF16; context 2048; four slots; prefix caching disabled; enforce_eager false |
| patched SGLang | Model and KV BF16; context 2048; four running requests; radix cache disabled; Triton attention; CUDA-graph-disable false |
| llama.cpp | All layers offloaded; n_ctx 8192; n_seq_max 4; kv_unified false; BF16 K and V; AUTO flash attention |

For llama.cpp, preserve public-library defaults n_batch 2048, n_ubatch 512, four generation and batching threads, and offload flags.
Assert llama_n_ctx_seq equals 2048. Label these as public-library defaults, not llama-server defaults.
Preserve the HIP-graph build default and record actual backend resolution.
For SGLang set max_total_tokens to 8192. For vLLM set kv_cache_memory_bytes to 1207959552.
Record physical cache allocations separately; equal logical context capacity does not prove equal allocation.
Do not claim default ROCm graph execution for vllm.cpp without runtime evidence.

## Scheduling protocol

Each persistent adapter accepts JSON-lines configure, run, and shutdown commands.
Configuration binds the engine identity, verified model, raw prompts, token arrays, and fixed settings.
Run commands name phase, concurrency, and repetition. Results carry a schema version and matching command identifier.
Reject malformed, duplicate, missing, or out-of-order results. Bound subprocess runtime and log size; retain errors.

Load one engine at a time under the operator's Strix lease.
Ours uses at most c concurrent blocking vllm_complete_tokens calls sharing one engine.
Python adapters use at most c active asynchronous public requests, not one six-request batch pretending to be concurrency four.
llama.cpp uses one context-owning batch loop with explicit positions and sequence IDs.
Sample every needed logits row before the next decode overwrites it.
Clear each completed sequence's KV before reusing the slot. Refill slots in corpus order.
Never call llama_decode concurrently on the same context.

Use greedy sampling without an EOS mask, continue after EOS, and return exactly 128 generated token IDs.
Do not double-accept a token after llama_sampler_sample.
Use actual generated IDs, not IDs recovered by retokenizing output text.

## Correctness and timing

First run qualification at both concurrency levels for all engines, using pinned vLLM as the reference.
Require exact prompt and generated IDs, correct dtypes, and repeatability before accepting performance results.
If any engine fails, retain its named failure and diagnostics. Do not publish an accepted four-engine speed ratio.

After qualification passes, run one warmup corpus and three measured corpora at each concurrency for each engine.
Retain every repetition. Recheck generated IDs during measurements.
Measure with a monotonic clock immediately before first dispatch until all six requests complete.
Corpus throughput is 768 divided by elapsed seconds, never an average of per-request rates.
Record per-request dispatch/completion times, consumed and generated IDs, finish reason, status, and errors.
Report all repetition values, median, range, and ratios with their denominators.
Do not infer time-to-first-token or inter-token latency from a blocking completion; mark them unavailable.

Record source, binary, model, and audit hashes, resolved settings, environment, and contention state.
Record sampled process-tree RSS with its sampling interval and method; distinguish it from allocator peaks.
Record GPU memory only when a verified measurement interface is available. Never equate shared host memory with dedicated VRAM.
The operator captures matching workloads with the same profiler before accepting path or performance claims.
Profiler flags require measured tool help and successful trace evidence; this CLI must not invent a profiling recipe.
Reproduce accepted results on an idle host with unchanged binaries. Missing required evidence remains pending, not passing.

## Source anchors to read before implementation

- Ours: include/vllm.h public token-completion API; src/capi/vllm_c.cpp:993-1017; qwen3_dense.cpp:98-102 at the parent pin.
- llama.cpp: src/llama-context.cpp:290-301 and 3512-3547; ggml/include/ggml.h:232; common/common.cpp:291-296.
- llama.cpp: src/llama-sampler.cpp:895-960; ggml/CMakeLists.txt:216; ggml/src/ggml-hip/CMakeLists.txt:109-110.
- vLLM: config/model.py:241 and 1350-1358; public AsyncLLM.generate and AsyncEngineArgs executing paths.
- SGLang: python/sglang/srt/server_args.py:1452 and the public Engine.async_generate executing path.

Resolve every anchor against the exact parent pin. A missing API or legitimate mismatch needs a design correction before implementation.

## Tests, review, and stop conditions

Write intended failing CLI tests before implementation. Keep standard-library discovery free of undeclared dependencies.
Test wrong IDs/settings, prefix-cache leakage, incorrect capacity, concurrency overruns, slot reuse without KV clearing,
logits-row misassociation, EOS masking/stopping, truncation, repeated streaming chunks, bad timing boundaries,
discarded repetitions, model/audit hash changes, subprocess errors, and output overwrite refusal.
Native scheduler fixtures must exercise the production adapter through public API test doubles with no GPU dependency.
The independent reviewer mutates each guarantee and removes production call sites; the focused tests must fail.
Run focused tests and the full applicable gate, then have the operator repeat them on the immutable reviewed head.
The operator alone schedules real GPU qualification and measurement under a lease.

Stop accepted qualification on conversion failure, wrong dtype/backend, token mismatch, nonrepeatability, or inadequate capacity.
Do not relax correctness, change a pin, mask a failure, or infer a speed advantage to complete the report.

## Owed

#3053 owns implementation, independent review, operator gates, real qualification, traces, throughput, and publication.

## Adapter manifest and runtime evidence

The version 1 manifest contains `schema`, `run_id`, `model`, `gguf`, `audit`, `engines`, `timeout_seconds`, and `log_bytes`.
`model` contains the pinned `revision`, local `directory`, and `files`, mapping relative names to SHA256 values.
`gguf` and `audit` are file bindings with absolute `path` and `sha256` fields.
Audit provenance keeps its original paths. Qualification compares model revision and file hashes, not the audit machine's directory names.

`engines` has exactly the four labels in the parent scope.
Each engine contains `source_revision`, `source`, `binary`, `command`, and `environment`.
`source` binds a git archive whose PAX comment names the pinned revision.
Patched SGLang additionally binds `patch`, whose SHA256 must match the reviewed gfx1151 compatibility patch.
`binary` binds the command's executable. `command` is an argument array, never a shell string.
`environment` contains the complete explicit `variables` dictionary and `files`, a list of bound runtime files.
The operator supplies the adapter scripts, shared libraries, package inventories, and other runtime dependencies in that list.
Every absolute file argument must have a binding. Record and verify transitive dependencies in the operator's environment inventory.
No implicit environment inheritance changes the selected runtime. Lease device visibility must match the caller.
The CLI requires `RC_DEVICE=strix:gpu0` and a nonempty `RC_JOB_ID`.

Each engine optionally contains a bound `runtime_evidence` JSON file.
Its schema contains `schema: 1`, `verified: true`, `engine`, `source_revision`, `binary_sha256`, `model_revision`, and `model_files`.
It also contains `evidence_run_id`, `environment_sha256`, `workload_sha256`, `resolved`, and nonempty `artifacts`.
Each artifact has `path`, `sha256`, and `kind`, which is `trace` or `log`.
llama.cpp evidence additionally binds `gguf_sha256`.
`environment_sha256` hashes the complete engine environment object as sorted compact JSON.
`workload_sha256` hashes sorted compact JSON containing `prompts`, `prompt_ids`, `tokens: 128`, `concurrency: [1,4]`, and `settings`.
`settings` and `resolved` contain `backend: rocm`, `model_dtype: bfloat16`, `kv_dtype: bfloat16`,
`context_per_sequence: 2048`, `slots: 4`, and `prefix_caching: false`.

The evidence is an explicit operator attestation backed by hashed artifacts, not automatic interpretation of profiler output.
`evidence_run_id` identifies an earlier diagnostic run. It must not pretend to identify the new invocation's `run_id`.
The earlier run must use identical artifacts, environment, settings, and workload.
Missing evidence retains startup and correctness diagnostics with `PENDING` qualification. It permits no throughput acceptance.
Wrong or unverified evidence is a named failure. Requested configuration never proves a resolved ROCm backend.
This distinction is required because the pinned public `vllm.h` exposes no text-engine backend or cache introspection.
Do not use device value 2 to request ROCm. That public API value requires CUDA.

The controller first loads pinned vLLM to get canonical tokenizer IDs.
It executes two qualification corpora per concurrency on every engine, then stops each process group before loading another engine.
After all four qualify, it reloads each engine for one warmup and three measured corpora per concurrency.
Weights stay resident across that engine's corpora. Repeated cumulative vLLM chunks replace prior chunks, never concatenate them.
Token mismatches retain raw outputs and continue the remaining qualification corpora.
Per-engine files preserve failures even when final binding verification fails.
The report records unavailable per-token latency and GPU memory explicitly.
Process-group RSS is a sampled measurement, not an allocator peak. Idle same-binary reproduction remains an operator obligation.

Build the native adapters separately with the local CMake project.
Supply `VLLM_INCLUDE`, `VLLM_LIBRARY`, `LLAMA_INCLUDE`, `LLAMA_LIBRARY`, `GGML_INCLUDE`, and `JSON_INCLUDE`.
The project does not fetch dependencies or rebuild the engines. Use the pinned public headers and already verified libraries.
Run Python engines with the bound interpreter and absolute `python_adapter.py` path.
Run native engines with the corresponding `strix-vllm-adapter` or `strix-llama-adapter` executable.

The focused command is:

```sh
python3 -m unittest tests.tools.test_strix_four_engine_qualify tests.tools.test_strix_native_adapters tests.tools.test_strix_python_adapters
```

The native fixtures compile the actual adapter against public API doubles without a GPU.
The Python fixtures execute the actual adapter with public API doubles and a persistent event loop.
The initial CLI test failed because the production qualification module did not exist.
Evidence: `/home/mudler/.cache/strix3053-adapters-red.log`.
The native and Python implementations preceded their dedicated doubles, so this change does not claim independent test-first evidence for those components.
Their deletion mutations and focused gates remain explicit verification obligations.

Twenty scratch mutations fail the focused behavior tests.
They change throughput arithmetic, token equality, missing evidence, evidence bindings, timing, concurrency, and lease enforcement.
They also remove native completion and decode calls, corrupt KV clearing and positions, misassociate logits, disable EOS continuation,
double-accept llama.cpp samples, serialize c4 work, disable vLLM graphs, and corrupt cumulative streaming checks.
Every scratch source is restored byte-for-byte after its mutation.
Logs use `/home/mudler/.cache/strix3053-adapters-mut-<name>.log`.
Initial scratch edits with incorrect indentation were discarded setup attempts, not mutation evidence.
The final focused log is `/home/mudler/.cache/strix3053-adapters-final-focused-04.log`.
The immutable-head preflight log is `/home/mudler/.cache/strix3053-adapters-full.log`.
The implementer handoff records observed exits and skipped gates. No GPU correctness or performance result is claimed by these fixtures.

## Matched-adapter review repairs

The independent review of `7be0abf7306683bebfad8b7abbc4b7264acb2316` found two production defects and three coverage gaps.
The fresh repair retains the original test-first chronology above. New regressions do not retroactively satisfy that obligation.

SGLang owns the event loop used by server-info, receiver futures, and generation.
At `f63458b5`, `entrypoints/engine.py:277-281` creates a loop when no loop is running.
`engine.py:999-1002` starts server-info work on that loop.
`managers/tokenizer_manager.py:1822-1831` retains the receiver's first loop.
The adapter reuses `Engine.loop` for both qualification repetitions and closes that loop at shutdown.
The source-faithful double creates a real loop-bound future. Using another loop fails through the production JSON-lines entry point.
The initial regression exited 1 with `Future attached to a different loop` before the production repair.
Its log is `/home/mudler/.cache/strix3053-adapter-repair-red-sg-loop.log`.

Runtime evidence requires a nonblank string identity distinct from the current manifest's `run_id`.
The CLI regression rebinds every evidence file after changing only that identity.
The corrected regression exited 1 before the guard changed because current-run and malformed identities produced `PASS`.
Its log is `/home/mudler/.cache/strix3053-adapter-repair-red-evidence-02.log`.
The first test attempt expected the wrong aggregate failure label. That setup error is not red evidence.

Eight negative controller cases independently rebind a wrong archive revision, audit result, tensor count, model hashes,
GGUF hash, converter revision, converter inventory, or insecure environment.
The environment case rebinds its runtime evidence too. Unrelated hash guards cannot mask the targeted refusal.
A positive case copies the audit unchanged and relocates the model directory. Content identity remains sufficient.

The Python fixtures keep requested arguments valid while changing resolved runtime values.
They cover missing HIP, CUDA identity, unavailable devices, wrong architecture, seven vLLM settings, seven SGLang settings, and insufficient SGLang capacity.
The llama.cpp fixture measures the peak number of independent logits sequences in actual public `llama_decode` batches for each corpus.
It requires peaks of 1 and 4 at the respective concurrency levels, including the second repetition.
Native production code remains unchanged. This fixture proves scheduler overlap, not real GPU token correctness.

The focused command remains unchanged. Repair logs use `/home/mudler/.cache/strix3053-adapter-repair-`.
The repaired focused suite passed 12 tests in `focused-final.log`.
All 34 isolated mutations exited 1 with failing assertions in `mutations-03.log`.
Each source file was restored byte-for-byte after each mutation. Individual logs use `mut-<name>.log`.
The initial mutation runner had an invalid patch header. That setup failure is not mutation evidence.
A removed identity-type guard initially survived because an incidental exception still refused the input.
The repaired fixture requires the named runtime-evidence refusal. The repeated mutation now fails that assertion.
The immutable-head full gate, independent review, operator rerun, real qualification, matching traces, and performance acceptance remain separate obligations.
