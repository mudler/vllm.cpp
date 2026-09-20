# Isolate llama.cpp c4 input corruption on Strix

Row: `BACKEND-GATE-ROCM-SGLANG`.
Issue: [#3075](https://github.com/mudler/vllm.cpp/issues/3075).
Parent: [matched four-engine benchmark](strix-four-engine-qwen3-4b.md), #3053.

## Now

The developer approved the measurement-first design on 8 September 2026.
The first deliverable isolates the failure through the pinned public API.
No root cause, comparator repair, or accepted performance result exists yet.
This spec does not change the parent row's lifecycle or correctness gate.
One pull request carries this issue's committed spec and implementation.
The base is `d2b68ef03dd010c42696048a08c4f63c98701008`.

## Scope and exclusions

Add a standalone first-prefill probe under `tools/bench/strix_four_engine/`.
The probe compares independent prompts, one combined batch, and synchronized partitions.
It does not sample, refill slots, clear live KV, or run a throughput measurement.
Preserve the existing benchmark adapter and qualification controller.
Do not change engine code, pins, dependencies, graph settings, or tolerances.
A comparator patch needs reproduced causality and a committed repair supplement.
A patched comparator must retain the stock baseline and identify its patch.

## Pins and evidence

Use llama.cpp `10bf611e533d81f739128304991c5e133c6aebd8`.
Use Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
Use the audited BF16 GGUF, SHA256
`7f0934a8d35ccf7d0f3e7f99492d57b5c5db3cb435efab55df7a367ac2ecbcf6`.
The native library, dependent libraries, source archive, and model need current hash bindings.

Qualification job `802b32c9-c817-4c18-825c-231ba684274e` failed.
Its `qualification-59a9c319-12/result.json` has SHA256
`1d518a475a79947d9921be92e9893ed74a4831359063284524b3c32a0db4f400`.
The parent issue retains the operator's evidence at
[the qualification report](https://github.com/mudler/vllm.cpp/issues/3053#issuecomment-5586767582).
The c4 corruption survived c4-first ordering and diagnostic graph disabling.
The graph-disabled run is not a performance denominator.
These observations reject two explanations, not every graph or scheduler defect.

## Executing source and ownership

All llama.cpp anchors below refer to the pinned revision.

- `src/llama-kv-cache.cpp:709` selects `split_equal(..., true, 0)`.
- `src/llama-batch.cpp:510` partitions equal-length prefixes across consecutive sequence IDs.
- `src/llama-batch.cpp:749` preserves original output indices in each microbatch.
- `src/llama-context.cpp:1816,1863` submits graphs and copies terminal logits asynchronously.
- `src/llama-context.cpp:1972` orders public output rows.
- `src/llama-context.cpp:850,3744` synchronizes and resolves public logits indices.
- `examples/parallel/parallel.cpp:350,430` requests terminal rows and samples original batch indices.
- `tests/test-llama-archs.cpp:309` primarily covers single-sequence output selection.
- `ggml/src/ggml-cuda/ggml-cuda.cu:5306` admits pinned-host buffers on integrated GPUs.
- `ggml/src/ggml-backend.cpp:1401,1613` protects copied inputs, but supported host buffers can avoid that copy.
- `src/llama-context.cpp:1339,1380` synchronizes reused inputs only for pipeline parallelism before rewriting inputs.

The host-input race is a hypothesis, not a measured diagnosis.
Existing upstream owners are
[issue 28056](https://github.com/ggml-org/llama.cpp/issues/28056),
[issue 25992](https://github.com/ggml-org/llama.cpp/issues/25992), and
[PR 27311](https://github.com/ggml-org/llama.cpp/pull/27311).
Do not duplicate their upstream issue or adopt the broad patch without attribution.
Local #2557 concerns the different CPU-offload/Q4_K case.
vLLM remains the model oracle; this probe diagnoses a comparator's public API.

## Probe interface

Add `llama_prefill_probe.cpp`, compiled against only pinned public llama and JSON headers.
Invocation: `llama-prefill-probe --config CONFIG`.
CONFIG is one JSON object with these required keys:

- `schema`: integer `1`.
- `gguf`: a nonempty string containing the host-local GGUF path.
- `prompts`: the parent's six raw prompt strings, in their declared order.
- `prompt_ids`: six nonempty arrays of canonical integer token IDs, in the same order.

Reject missing or unknown keys and incorrect JSON types, including booleans used as integers.
Refuse configuration files larger than 1 MiB and out-of-range token IDs.
Require the loaded vocabulary size to equal 151936.
Verify all six tokenizations without BOS or a chat template.
Use only the first four prompts for this first-prefill experiment.
Require their lengths to equal `[6,5,6,7]`; refuse a different experiment.

Match the native adapter's context settings: full GPU offload, 8192 total tokens,
four sequence slots, 2048 tokens per sequence, separate KV streams, and BF16 K/V.
Retain public defaults for batch size, microbatch size, threads, and AUTO flash attention.
Assert the defaults already checked by `native_adapter.cpp`.
Keep graph-disable environment variables absent in operator runs.

Load the model once. Run three repetitions of each mode.
Create a fresh context for each combined or partitioned repetition.
Create a fresh context for each independent prompt in each repetition.
Preserve each prompt's original sequence ID, including independent runs.

Modes:

1. `independent`: decode each prompt alone and immediately copy its terminal logits.
2. `combined`: submit all 24 tokens in original sequence-major order.
   Copy all four terminal rows before any sampling or further decode.
3. `partitioned`: submit the pinned scheduler's inferred partitions as separate public decode calls.
   Synchronize after every call, including calls without terminal outputs.
   Copy each available terminal row before the next call.

The partition's original token indices are:

- `[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]`.
- `[5]`.
- `[16,22]`.
- `[23]`.

Preserve original positions, sequence IDs, and terminal flags in these partitions.
Combined terminal indices are `[5,10,16,23]`.
Partitioned terminal capture order is `[10,5,16,23]`.
The public getter receives the terminal index within its actual submitted batch.
Never pass an original combined index into a smaller partition's public getter.
Emit the mapping explicitly so a reader can audit the comparison.

## Output and errors

Write JSONL records to stdout; send library diagnostics to stderr.
Emit a header, 36 terminal-score records, and one final completion record.
Every record has integer `schema: 1` and a string `type`.
The header has `type: header`, `vocab_size: 151936`, and `repetitions: 3`.
It also records the requested and public-getter-resolved context settings separately.
Do not label requested settings as runtime verification.

Score records have `type: scores` and these fields:

- `mode`: `independent`, `combined`, or `partitioned`.
- `repetition`: integer from zero through two.
- `sequence_id`: integer from zero through three.
- `original_terminal_index` and `batch_terminal_index`: integer getter mappings.
- `prompt_ids`: the complete canonical input for this sequence.
- `submitted_batches`: an ordered array of batches consumed by this context so far.
- `logits`: all 151936 finite FP32 scores in vocabulary order.
- `argmax`: the integer token ID with the highest score.
- `top10`: ten objects with integer `token_id` and finite numeric `logit`.

Each submitted batch is an array of token objects with integer `original_index`,
`token_id`, `position`, and `sequence_id`, plus boolean `logits_requested`.
This metadata describes all input rows, not only the sequence whose logits are copied.
Break exact score ties by ascending token ID.
Use round-trip float serialization and state that scores are raw logits.
Deep-copy each row before another getter or decode can overwrite borrowed storage.

The final record has `type: diagnostic_complete` and integer `score_records: 36`.
It never says `PASS` or accepted correctness.
Exit zero means the probe completed its declared capture, not that engines agree.
Exit nonzero on invalid configuration, tokenization mismatch, allocation failure,
decode failure, missing logits, nonfinite logits, or output failure.
Never emit a completion record after an error.
Release model, context, and batch allocations on every exit path.
The operator bounds runtime and output size and rejects partial records.

## Task 1: implement and test the public probe {id: 1, deps: []}

A fresh implementer owns only the new probe, its focused tests, and this issue's evidence.
Add `tests/tools/test_strix_llama_prefill.py`.
Compile the real CLI against a stateful public-API double for CPU tests.
Write the smallest failing CLI test before implementing the probe.
Model distinct logits per sequence and invalidate borrowed output storage on later calls.
Verify exact submitted tokens, positions, sequence IDs, flags, and getter indices.
Verify fresh-context boundaries, synchronization order, all 36 records, and complete vocabulary capture.
Exercise bad tokenization, changed defaults, invalid lengths, failed decode, null logits,
nonfinite scores, write failure, and resource cleanup.
Deleting the CLI's decode or capture call must fail the focused suite.
Mutate each claimed guard and restore the scratch tree byte-for-byte.
Do not replace these behavioral checks with source-text assertions.
Compile against the real pinned public headers before operator execution.
The fake runtime checks API use; it cannot establish real model correctness.

Verify: `python3 -m unittest tests.tools.test_strix_llama_prefill`.
Full gate: `scripts/agent-preflight.sh --quiet`.
Record all skips separately; an exit-zero preflight does not establish omitted hardware gates.
Fresh independent review follows the implementer's focused and full gates.
The operator reruns the focused and full gates on the immutable reviewed head.

### Task 1 implementation evidence

The standalone probe uses public llama headers at the pinned revision.
It does not modify the existing adapters or the comparator library.
The CPU double models borrowed logits, distinct sequences, and allocation lifetimes.
Its event records expose submitted rows, getter indices, synchronization, and context boundaries.
These tests establish probe behavior, not real model correctness.

The compiled CLI scaffold first returned no records and accepted invalid configurations.
The focused suite failed on those behaviors before the capture implementation existed.
The red log is `/home/mudler/.cache/strix3075-red.log`, exit 1, SHA256
`84af4c6e31aaa280e8300732636d149db986ec5849930214d64975d2cb2ddd78`.
The final focused command passed five tests in 5.442 seconds:
`python3 -m unittest tests.tools.test_strix_llama_prefill`.
Its log is `/home/mudler/.cache/strix3075-final-focused-01.log`.
The real-header syntax check exited zero with this command:

```sh
c++ -std=c++20 -fsyntax-only \
  -I /home/mudler/.cache/strix3053-audit-llama/include \
  -I /home/mudler/.cache/strix3053-audit-llama/ggml/include \
  -I third_party tools/bench/strix_four_engine/llama_prefill_probe.cpp
```

Thirty-seven behavioral mutations each made the focused suite fail.
They removed decode, capture, synchronization, configuration guards, runtime guards,
or resource cleanup, and changed indices, ordering, dtype, and offload settings.
Each scratch mutation was restored with a byte-identical `cmp` check.
The logs use `/home/mudler/.cache/strix3075-mut-<name>.log`.
Names are `decode`, `capture`, `sync`, `position`, `sequence`, `flags`, `getter`,
`finite`, `tie`, `fullrow`, `output`, `length`, `range`, `idtype`, `schema`,
`size`, `defaults`, `resolved`, `tokenizer`, `vocab`, `layers`, `kv`, `dtype`,
`model_free`, `null`, `model`, `context`, `batch`, `gguf`, `prompts`, `arrays`,
`empty`, `duplicate`, `batch_free`, `backend_free`, `context_free`, and `v_dtype`.
No mutation result depends only on a compiler error.

The startup preflight overlapped edits and is not an immutable baseline.
It exited 1 because the tools suite failed. Five argument-dependent gates were skipped.
The final full-gate log is `/home/mudler/.cache/strix3075-final-preflight-01.log`.
The operator handoff records its exact exit and the unchanged source/index identities.
The full gate uses a private `/tmp` mount and runs as `mudler`.
Hardware gates, real logits, and the input-race hypothesis remain unverified by Task 1.

### Task 1 review repair evidence

The independent review of `bea3ee533c25398aa69708326f3a5dfe0c2c4a6f` found three gaps.
The probe did not refuse unavailable GPU offload.
Six resolved getters lacked independent invalid-result fixtures.
The tests did not check three required header values or the top-ten scores.
The review is `/home/mudler/.cache/strix3075-review-report.md`.
This repair preserves the original test-first history and all existing capture behavior.

The probe now queries `llama_supports_gpu_offload()` after backend initialization and before model loading.
The backend guard releases resources when the query returns false.
This mirrors `native_adapter.cpp` and refuses a clearly unsupported experiment.
It does not attest actual offload or the Strix device.
The pinned function also admits RPC backends, as `src/llama.cpp:101` records.
Task 2 still owns device identity and actual offload verification.

The new CLI test first failed because unavailable offload still returned exit zero.
Its command was
`python3 -m unittest tests.tools.test_strix_llama_prefill.PrefillProbeTests.test_refuses_missing_gpu_before_model_load`.
The log `/home/mudler/.cache/strix3075-repair-red-gpu.log` records exit 1.
After the one-line guard, the full focused suite passed six tests in 5.064 seconds.
That log is `/home/mudler/.cache/strix3075-repair-green-gpu.log`, exit 0.
The final focused suite passed six tests in 4.977 seconds, exit 0:
`/home/mudler/.cache/strix3075-repair-focused.log`.
The pinned public-header syntax command documented earlier also passed, exit 0:
`/home/mudler/.cache/strix3075-repair-header.log`.

Six independent getter faults now leave requested parameters and defaults valid.
Each wrong getter result must cause the resolved-context refusal and resource cleanup.
The header checks require integer schema 1, vocabulary 151936, and three repetitions.
Each top-ten score must equal its corresponding entry in the independently checked full row.
These coverage repairs do not change production output or getter behavior.

Sixteen isolated mutations each failed the full focused suite with an assertion failure.
The mutations removed the GPU guard or backend cleanup, or moved the guard after model loading.
Six replaced individual resolved getters with expected constants.
Four omitted or corrupted the required header fields and top-ten scores.
Three changed header integer types without changing their numeric values.
All ten mutations that survived the original review now fail.
Each mutation was immediately restored with a byte-identical `cmp`, exit 0.
The scratch directory is `/home/mudler/.cache/strix3075-repair-mut-Kmslyo`.
Logs use `/home/mudler/.cache/strix3075-repair-mut-NAME.log`.
Names are `gpu_guard`, `gpu_after_load`, `backend_cleanup`, `resolved_ctx`,
`resolved_seq_max`, `resolved_batch`, `resolved_ubatch`, `resolved_threads`,
`resolved_threads_batch`, `header_schema`, `header_vocab`, `header_repetitions`,
`top10_scores`, `header_schema_type`, `header_vocab_type`, and `header_repetitions_type`.
The first missing-schema assertion raised `KeyError`; an explicit membership assertion replaced it.
The repeated mutation failed that assertion, not test setup or compilation.

The unchanged startup preflight passed all executed checks with exit 0.
Its log is `/home/mudler/.cache/strix3075-repair-startup.log`.
It ran as `mudler` with a private ext4 `/tmp` before any source or test edit.
Five argument-dependent checks were skipped: ARM ISA, CPU ISA, CUDA fat-gencode, PR size, and Triton AOT multiarch.
The immutable repair handoff records the final preflight and exact-range checks separately.
No CPU fixture result establishes hardware correctness or an input-race diagnosis.

## Task 2: reproduce under a lease {id: 2, deps: [1]}

The operator builds the reviewed probe against the bound pinned public library.
Use ccache, at most four build jobs, and host-local model storage.
Claim `strix:gpu0` through `rc`; never run GPU work outside its lease.
Hash the actual loaded libraries and artifacts before and after the experiment.
Record exact commands, source revisions, device identity, environment, and contention.
Bound each probe invocation to 15 minutes and each output stream to 256 MiB.

Validate all records and compare complete rows across modes and repetitions.
Report argmax identities, margins, maximum absolute differences, and nonfinite counts.
Name each comparison explicitly; do not introduce an acceptance tolerance.
Report independent-run argmax repeatability first, separately from numerical row differences.
Compare combined argmax IDs with the previously recorded incorrect c4 opening IDs.
Report whether all three partitioned repetitions restore each independent opening argmax.
If independent openings vary, or combined openings do not reproduce the observed corruption,
report the localization as inconclusive rather than declaring recovery.
Small FP32 row differences alone do not establish corruption or a race.
Reproduced wrong openings before sampling localize that failure below the sampling lifecycle.
Restoration under synchronized partitioning implicates asynchronous microbatch execution only as a candidate.
Neither result identifies which input or output buffer races.

## Risks and stop conditions

Stop on changed bindings, wrong model/device, or a different inferred partition.
Stop on incomplete records or unverified public defaults.
If first-prefill corruption does not reproduce, retain that negative result.
Then scope the next production lifecycle probe; do not invent a successful diagnosis.
Do not weaken the parent's correctness gate to admit corrupt opening tokens.
Do not publish timing from this synchronizing diagnostic as throughput.

## Owed

Issue #3075 retains ownership of buffer-causality proof and a reviewed comparator repair.
Issue #3076 owns the separate vllm.cpp c4 performance gap.
Issue #3077 owns the separate distributional calibration proposal.
The current probe does not complete any of those acceptance obligations.
