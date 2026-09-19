# Capture original Qwen3-4B scores on Strix

Row: `BACKEND-GATE-ROCM-SGLANG`. Issue: `ISSUE-GH-3077`.
Design base: `1f46d44e15cb49dde32f21c484d974eb0af2c7a3`.

## Now

The developer approved original-score capture and worker-start profiling design.
Replay can replace token IDs before extraction when the extracted logits remain unmodified.
This approval permits diagnostic implementation after this committed design and independent review.
It ratifies no numerical tolerance, distributional gate, forced-output pass, or performance acceptance.
The owning matrix remains `INVENTORIED`. This supplement changes no lifecycle state.
Use one integration change for this issue, with the design committed before code.

## Scope and dependencies

Build a bounded diagnostic through public incremental generation on production model paths.
Keep the strict token gate. Stable oracle behavior remains token-exact.
Separate source feasibility, instrumentation equivalence, oracle calibration, held-out validation, and candidate comparison.
Each stage has its own evidence and disposition. A source inspection is not a hardware result.

The complete historical parent contracts are available at
`b923ac2c4c24e9c608d4a6e02e868538be7536fa`:
`.agents/specs/strix-four-engine-qwen3-4b.md`,
`.agents/specs/strix-four-engine-qualification.md`, and
`.agents/specs/strix-c1-c4-c32-workload.md`.
Read those complete objects before implementation. They are provenance, not permission to merge old harness code wholesale.
Current main lacks these harness files. Port only the diagnostic's required public-entry transport and tests.
Preserve provenance for every adapted function. Do not import the four-engine benchmark or lifecycle campaign as incidental scope.

The separate performance issue numbered 3076 owns the reusable worker-start
profiling mechanism at commit
`0d5f89a54858b217e42860b625473a39eeb19d57`. It also owns
ordinary-production baseline evidence when every binding in this design
matches exactly. `ISSUE-GH-3077` owns the mode-aware integration and trace
evidence for capture-only, self-replay, and common-replay runs.
The issue-3076 CLI does not accept score-capture or replay mode fields.
This design does not change the profiler or engine kernels. Missing traces
prevent equivalence acceptance but do not prevent CPU diagnostic implementation.

## Immutable inputs

- vLLM: `e126687a9a828d513c01a07cd69f025f27d63280`.
- vllm.cpp: `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb`.
- Model: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Model and KV cache: BF16. Record actual resolution and annotated F32 sampler outputs separately.
- Corpus: six canonical raw prompts from historical `tools/bench/strix_vllm_oracle/runtime.py`.
- Sampling: greedy, temperature 0, top-p 1, 128 generated tokens, ignore EOS, no speculation or reusable prefix cache.
- Tokenization: exact model tokenizer, no chat template, no added BOS, identical canonical prompt IDs.

Bind all model shards, tokenizer files, configs, binaries, libraries, source patches, environments, and prompt arrays by SHA-256.
The diagnostic workload ID is `strix-qwen3-4b-original-scores-v1`.
It contains exactly six requests at c1 and c4, with request indices 0 through 5.
Each complete capture has 768 score rows per concurrency.
It is a bounded diagnostic subset, not the 96-request c1/c4/c32 benchmark or evidence of c32 coverage.
Retain the parent production capacity settings across diagnostic arms. Record actual admission and batch shapes independently of client concurrency.

## Source chain at the pins

The vLLM archive was inspected on 19 September 2026. Its SHA-256 is
`5509708748c28fef153c96ad98b341b4ecdb500a60dd7dcd6fa330ff96bc709b`.
Archive provenance: `/mnt/nas_share/rc/strix-vllm-3043.GkMABu/vllm-e126687.tar`.
These anchors refer to that archive at the vLLM pin, not current upstream.

| Source | Contract |
|---|---|
| `vllm/engine/arg_utils.py:547,550,1783,1786` | Public `max_logprobs`, `logprobs_mode`, and `enable_trace_replay` reach model configuration |
| `vllm/config/vllm.py:1071,2599,2722` | Replay requires V2; custom logits processors are unsupported in V2 |
| `vllm/sampling_params.py:376,802,908` | Public trace IDs, full-vocabulary limits, replay validation |
| `vllm/v1/engine/input_processor.py:125,161,370` | Enablement refusal and request-local replay normalization |
| `vllm/v1/engine/async_llm.py:551-615` | Public generation yields request outputs from one collector per request |
| `vllm/v1/engine/output_processor.py:48-96,395-419` | The collector can merge delta output; delta slicing can contain more than one completed row |
| `vllm/logprobs.py:30-93,167-210` | `FlatLogprobs` retains primitive arrays and avoids per-entry `Logprob` objects |
| `vllm/v1/worker/gpu/model_runner.py:442,445,1132,1328` | V2 sampler configuration and request-ID to state-index mapping |
| `vllm/v1/worker/gpu/sample/states.py:59` | Request `logprobs=-1` resolves to vocabulary size |
| `vllm/v1/worker/gpu/sample/sampler.py:140,152,157,177,216,303` | Sample, overwrite IDs, extract original logits, mask unfinished prefill |
| `vllm/v1/worker/gpu/sample/trace_replay.py:34,45,62` | Replay changes sampled IDs only, indexed by request and generated position |
| `vllm/v1/worker/gpu/sample/logprob.py:110,127,131` | Full vocabulary uses top-k token IDs and gathers raw scores as F32 |
| `vllm/v1/worker/gpu/input_batch.py:495,552` | Partial-prefill output masking and committed-length update |
| `vllm/v1/worker/gpu/async_utils.py:142,175,192` | Copy score tensors to CPU and trim sampled IDs by actual count |
| `vllm/v1/engine/logprobs.py:69,112,349` | Convert token-keyed scores into public request output |
| Ours `include/vllm.h:716,780,910` | Public per-request logits callback and token completion API |
| Ours `src/capi/vllm_c.cpp:267,955` | Copy callback into sampling parameters and submit through shared generation |
| Ours `src/vllm/v1/worker/gpu/input_batch.cpp:315,538,786,908` | Preserve callback ownership through compaction and swaps |
| Ours `src/vllm/v1/sample/sampler.cpp:441` | Reach callback at sampler processor stage |
| Ours `src/vllm/v1/sample/logits_processor/builtin.cpp:75` | Synchronize, expose host row, invoke callback, copy back when required |

Source SHA-256 bindings:

- V2 `sample/sampler.py`: `408b08c4ac66de271f476460490f360ef2a9b73528a346d132a22326e1674369`.
- V2 `sample/trace_replay.py`: `d2a7d3b3b55379efabe4740a4b7f14030126dc11be36f6ea974164ce42316a50`.
- V2 `sample/logprob.py`: `f47479a818192acb6b297c79fa67a3b2f9820e3492a394d5adda777e9b7c832e`.
- `config/vllm.py`: `b3e83df1b0ff6bb0db6af6b9032ad49983d3edb25361cf98e14e73841e5620d0`.
- Ours public header: `bb2c9202af3b7a196c35e63fd49c1845eb4457f938a55abfb3db4d114e722e12`.

The callback's introduction is commit `ca1c1b51d`, found with `git log -S'logits_processor' -- include/vllm.h`.
No local trace-replay harness history was found with `git log -S'trace_decode_token_ids' -- tools/bench`.
An open-pull-request search for 3077 returned no entries on 19 September 2026.

## Capture design

Use public `AsyncLLM.generate` with token prompts and unique request IDs for vLLM.
Set engine `logprobs_mode="raw_logits"`, `max_logprobs=-1`, and request `logprobs=-1` for capture.
Keep `prompt_logprobs=None`, `n=1`, and all masking, bias, penalty, and constraint features disabled.
For replay, enable `enable_trace_replay=True` and supply `trace_decode_token_ids` per request.
Require exactly 128 valid trace IDs and enough remaining context before submission.
Normalization truncates traces to available context and bounds `max_tokens` by trace length.
It clears stop conditions and forces ignore-EOS semantics. Refuse any normalization that changes the declared 128-token workload.
Do not install a custom vLLM logits processor. Its V1 fallback or V2 rejection invalidates this diagnostic route.

The sampler first generates a token from the real distribution, then overwrites only its ID.
Raw extraction subsequently gathers the original logits. Temporal extraction after ID replacement is permitted.
Capture after numerical modification of scores is prohibited.
The full-vocabulary result includes a sampled-token entry plus vocabulary entries in score order.
Reconstruct dense rows by token ID, never by return position or dictionary iteration order.
Deduplicate only the documented sampled-token duplicate when both scores are exactly equal.
Require all vocabulary IDs exactly once after that normalization. Reject unexplained duplicates, missing IDs, or inconsistent duplicate scores.
Do not use cumulative logprob fields as logits or substitute top-k-only distributions.

For ours, invoke `vllm_complete_tokens` concurrently through the public header.
Copy each callback's original F32 row before modifying the mutable sampler row for replay.
The generated token prefix passed to the callback must exactly match the declared continuation prefix.
Capture-only callbacks write no values. Replay callbacks retain the original copy, then force the declared next ID for continuation only.
The persisted original copy must remain finite and byte-identical before and after forcing.
Use per-request callback state. Record errors without throwing across the ABI, then fail the request after generation.
No callback, missing callback, or incorrect callback count is a failure.
The host callback can alter synchronization and scheduling. Its existence alone establishes no production-path equivalence.

Persist rows as bounded binary F32 arrays with explicit endianness, shape, and SHA-256.
Use a separate metadata stream keyed by run, engine, request ID, request index, prompt index, and generated position.
Each row records prompt and consumed-prefix hashes, original argmax and runner-up, replay ID, phase, vocabulary size, and source dtype.
Position zero is first-prefill output. Positions 1 through 127 require incremental KV decode witnesses.
Do not turn whole-prefix forward calls into decode evidence. Partial-prefill steps with no emitted token produce no accepted score row.
Check request mapping after compaction, refill, and tails. Completion ordering must not define request identity.

The pinned public vLLM API emits `RequestOutput` objects from
`AsyncLLM.generate` at `vllm/v1/engine/async_llm.py:551-615`.
`RequestOutputCollector` can merge pending delta outputs when the producer
outpaces the consumer at `vllm/v1/engine/output_processor.py:48-96`.
The adapter therefore sets `flat_logprobs=True`,
`output_kind=RequestOutputKind.DELTA`, `stream_interval=1`, and
`detokenize=False`. It reads the primitive `FlatLogprobs` arrays directly.
It never iterates the container into per-row dictionaries. The adapter does
not claim one-token streaming because the engine can clamp the interval or
merge pending output. These settings change transport representation and
delivery only. They do not change scores, token selection, replay IDs, or the
engine schedule.

Treat each yielded object as a batch of 1 through 128 completed score rows.
Start all c4 request generators concurrently and keep one consumer coroutine
per request. Do not serialize engine requests or wait for one request to finish
before consuming another. Each consumer copies every delivered batch into its
bounded binary artifact and metadata stream immediately. It then hashes and
flushes those bytes. Next, release each public `RequestOutput`, its completion
output, and its `FlatLogprobs` arrays. Clear every adapter reference.

The score manifest fixes these independent limits:

- `max_completed_requests_in_flight = 4`;
- `max_completed_request_object_bytes = 3221225472`;
- `max_resident_score_row_bytes = 311164928`;
- `max_resident_score_buffer_bytes = 13958643712`; and
- `max_persisted_score_bytes = 8589934592`.

The public result can carry one sampled-token duplicate per row. One complete
request therefore has 19,447,936 entries: 128 times 151,937. The pinned 64-bit
CPython 3.12 ABI uses 28 bytes for an integer, 24 bytes for a float, and 8 bytes
for one list slot. A conservative 12.5% list-capacity allowance makes each slot
9 bytes. The accumulated and pending `FlatLogprobs` containers share primitive
objects but own two sets of four list arrays. The upper calculation is therefore
152 bytes per entry: 28 + 24 + 28 + 8 times 9. This is 2,956,086,272 bytes per
request. The 3 GiB object cap leaves 265,139,200 bytes for container headers,
request objects, and a transient row.

The 311,164,928-byte row limit is the aggregate normalized F32 payload for four
128-row requests: 4 times 128 times 151,936 times 4 bytes. The 13 GiB absolute
resident ceiling holds four 3 GiB request envelopes and that raw payload. It
leaves 762,576,896 bytes of stop headroom. The object budget includes the live
public output, its accumulated `FlatLogprobs`, and a pending merged output. The
row budget covers adapter staging. At startup, require the exact Python object
sizes and list-growth bound used by this calculation. Fail before subprocess
start if the interpreter or output representation exceeds any input.

The manifest also records `cgroup_memory_limit_bytes`,
`post_model_baseline_rss_bytes`, `baseline_peak_rss_bytes`, and
`engine_supervisor_reserve_bytes`. Derive the reserve from the largest matched
ordinary-production increase above post-model steady state plus the largest
positive adjacent-sample increase. Bind the sampling period and every input
sample. Do not supply a default reserve.

Set the effective resident ceiling to the lower of 13 GiB and measured
headroom. Measured headroom is the cgroup memory limit minus post-model resident
bytes and the derived engine/supervisor reserve. Before score collection, prove
that this effective ceiling holds all four 3 GiB request envelopes plus the
311,164,928-byte raw payload. A missing finite cgroup limit or missing matched
baseline measurement is `PENDING` before subprocess start. Insufficient
measured headroom is `FAILING` before score collection. It does not authorize a
smaller c4 workload or serialized requests.

During collection, record each object's deep resident byte count, all live
score-object bytes, resident binary-buffer bytes, cgroup current and peak
memory, and process peak RSS. Refuse another delivered batch when its complete
reserved size can cross the effective ceiling. Refuse a write when the next
complete row can cross 8 GiB. Abort before either bound, preserve the bounded
failure record, and publish no incomplete artifact.

## Equivalence before calibration

Run the following controls on each engine, c1 and c4, with the same immutable binaries and model inputs.

1. Baseline: production free-running generation without score capture or replay.
2. Capture-only: free-running generation with capture, without replay.
3. Self-replay: capture plus replay of the capture-only run's complete continuation IDs.
4. Common replay: both engines consume the same fixed oracle continuation prefixes.

Retain all runs, including failures. Record unforced argmax separately from replay IDs.
A replay-ID match proves prefix control only. It cannot satisfy token correctness.
Compare baseline and capture-only tokens, and capture-only and self-replay original rows at identical prefixes and shapes.
Require exact equality for instrumentation checks. An observed difference remains unresolved, even when oracle nondeterminism seems plausible.
Do not choose a tolerance to make instrumentation pass.
Because baseline has no score output, prove score immutability separately with the sampler tests and self-replay comparison.
Never claim baseline score equality from token equality alone.

Trace baseline, capture-only, self-replay, and common-replay for each engine and each c1 and c4 stratum.
The complete equivalence set therefore contains 16 engine, mode, and concurrency trace strata.
The separate performance issue numbered 3076 supplies only the reviewed worker-start mechanism and ordinary-production evidence.
`ISSUE-GH-3077` owns the mode-aware integration and the capture-only, self-replay, and common-replay traces.

Use this public entry for every new mode-aware trace:

```text
python3 -m tools.bench.strix_score_capture \
  --manifest <score-manifest.json> \
  --worker-profile-manifest <profile-manifest.json> \
  --output <new-directory>
```

The score manifest selects the engine, mode, and c1 or c4 stratum. The profile
manifest uses schema `vllm.cpp/strix-qwen3-worker-profile/v1` from exact commit
`0d5f89a54858b217e42860b625473a39eeb19d57`. It supplies the immutable
profiler identity, configuration, resource bounds, and expected process roles.
It contains no score-capture or replay mode field.

Import `tools.bench.strix_worker_profile.launch` and require contract
`vllm.cpp/profiled-process-tree-launch/v1`. The issue-3077 adapter supplies one
validated launch request. The request contains the exact mode-specific
production argument array, engine identity, supervisor and GPU-owner role
bindings, environment allowlist, profiler binding, lifecycle limits, and new
output directory. The callable returns the structured worker-profile result.
That result contains the schema and launch identifiers, lifecycle receipts,
process tree, uniquely identified GPU worker, finalized trace artifacts,
bindings, counts, and fail-closed checklist.

The launch callable owns manifest validation, package and binary binding,
pre-import bootstrap, subprocess lifecycle, receipts, bounds, finalization,
and artifact checks. `ISSUE-GH-3077` owns only the mode-specific argument and
role binding through that callable. It must not copy the bootstrap, implement
profiler startup, bypass the lifecycle owner, or invoke the issue-3076 CLI as
if the CLI understood score modes.

Commit `0d5f89a54858b217e42860b625473a39eeb19d57` is a dependency pin, not
code that this design claims is present. Before implementation or hardware,
require that exact commit to be base-reachable from the implementation head.
If it is not base-reachable, return
`PENDING: worker-profile dependency 0d5f89a54858b217e42860b625473a39eeb19d57 is not base-reachable`
before import or subprocess start. After it becomes base-reachable, an import,
contract-version, manifest-schema, or result-schema mismatch is `FAILING`.
There is no CLI, copied-bootstrap, older-schema, or unprofiled fallback.

Reuse issue-3076 baseline evidence only when the engine and c1 or c4 window are
separately identifiable. All these bindings must match this design exactly:

- the source pins, model revision, tokenizer hash, and model-shard hashes;
- the binaries, libraries, build IDs, profiler, and configuration bytes;
- the prompt IDs, request order, token count, and c1 or c4 schedule; and
- the dtype, graph policy, scheduler shapes, sampling fields, and resource bounds.

The artifacts must also satisfy this design's comparison and completeness
rules. Otherwise, issue 3077 captures a new baseline through the public entry.

Bind the worker-profile manifest and result schema exactly as defined at
`0d5f89a54858b217e42860b625473a39eeb19d57`. That immutable design separates
profiler-harness revision `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660`
from nested SDK source revision
`97f5574fe2fdc7bef44fb01545347912ee9f1779`. Its recovery identity is the
exact six-package set. The set includes
`libsqlite3-0_3.45.1-1ubuntu2.8_amd64.deb`, size 701,602, SHA-256
`b1190bb72359f5fcc47406aa46065eaf4f1ca208085c51224a52b04bedc0b4bb`.
Its extracted SQLite library has SHA-256
`85265a9d4afca6f4b325ceb078b669c754fb881abed4cafe91ccebe9d625d975`
and build ID `5701975a7ab1644d59e6b20df0257e183eafa78e`.

Commit `74ca0f06` records historical worker-ownership semantics only. It does
not define the implementation schema or recovery identity. Reuse the
`Lifecycle and observation window`, `Artifact schema`, and `Completeness rules`
from exact commit `0d5f89a54858b217e42860b625473a39eeb19d57` without weakening them.
Each engine and mode must first pass the mode-aware readiness route. Each
readiness output has a 10-minute wall timeout, a 256 MiB aggregate stop
threshold, and a 192 MiB per-file limit.

Write each engine, mode, and concurrency trace to a new output directory. Each
directory has a 30-minute wall timeout and a 60-second cleanup timeout. It has
a 1 GiB aggregate trace-output stop threshold. It uses a 768 MiB per-file trace limit.
Run the two engines for one mode and concurrency sequentially as one pair with
a 75-minute lease budget. Eight pairs cover the four modes at c1 and c4. A
pair does not combine its two directory limits. Preserve failed and partial
artifacts under the issue-3076 failure schema. Publish a passing result only
after finalization and post-run binding checks succeed.

The 8 GiB score-data bound and these trace-output bounds are independent limits.
Neither quota includes the other. A larger trace or score bound requires a new
reviewed design and a capacity check.

Bind trace tool/version/configuration, process lineage, loaded binaries, graph configuration, graph replay events, and complete capture boundaries.
Record each scheduled request, query length, KV length, active sequences, padded dimensions, dtype, attention backend, and numerical kernel identity.
Compare the executed model and LM-head graph path. Separate expected capture, copy, top-k, and ID-replay operations from numerical model operations.
An extra capture operation is not automatically a numerical-path change, but missing graph coverage cannot establish equivalence.
Reject eager substitution, V1 fallback, changed precision, unknown kernels, dropped intervals, incomplete graph traces, or unmatched scheduler shapes.
If capture changes scheduling, report the mismatch and retain evidence. Do not infer equivalence from requested concurrency.
Do not publish instrumented throughput as production throughput.
Profiler timing remains diagnostic-only.

## Calibration and later ratification

Before opening candidate scores, freeze an oracle-only manifest with the six prompts, c1/c4 strata, phase strata, and run partition.
Use three complete oracle capture/replay repetitions for exploratory calibration and two new repetitions held out from calibration.
These are bounded diagnostic collection counts, not a ratified statistical sample size or acceptance rule.
Keep held-out artifacts sealed by hash until a proposed rule is committed.
Summarize token repeatability, full-row discrepancies, argmax margins, and divergence positions for each concurrency and phase independently.
State exactly which original rows and consumed prefixes each comparison uses.
Candidate scores must not select thresholds, exclusions, statistics, or calibration samples.

The ratification proposal must name the statistic, sample unit, sample counts, uncertainty method, multiple-comparison policy, and acceptance threshold.
It must include oracle-only evidence, held-out validation, corruption controls, and a justification for each stratum.
Any threshold adoption requires explicit developer ratification. Until then, every candidate remains under the strict existing token gate.
Do not import CUDA, 27B, or quantized-model tolerances. Do not extrapolate this subset to the 96-request benchmark.

## Implementation and verification contract

Create `tools/bench/strix_score_capture/` with a public CLI, manifest validator, Python adapter, public-header native adapter, and artifact validator.
Create `tests/tools/test_strix_score_capture.py` for public-entry CPU fixtures.
Use `python3 -m tools.bench.strix_score_capture --manifest PATH --output UNUSED_DIRECTORY` for unprofiled CPU fixtures.
Use the worker-profile-manifest form in `Equivalence before calibration` for hardware trace collection.
The manifest explicitly selects baseline, capture-only, self-replay, or common-replay mode and binds all identities and limits.
The result states `DIAGNOSTIC_COMPLETE`, `FAILING`, or a precise `PENDING` dependency. It never emits a correctness pass.
Do not modify `src/`, `include/`, benchmark acceptance, model pins, or upstream runtime sources in this issue.

The fresh implementer first runs a public-entry test that fails because capture is absent.
Port applicable cases from pinned `tests/v1/worker/test_gpu_trace_replay.py`,
`tests/v1/sample/test_trace_replay_params.py`, `tests/v1/engine/test_input_processor_trace_replay.py`,
and `tests/v1/sample/test_logprobs.py`. Preserve upstream parameters and refusals.
CPU fixtures prove transport and validation, not GPU numerical equivalence. Retain that distinction in every report.

Focused gate: `python3 -m unittest tests.tools.test_strix_score_capture`.
Compile the native adapter against the exact pinned public header before hardware execution.
Full gate: `scripts/agent-preflight.sh`. Record exact commands, exits, revisions, logs, and artifact-dependent omissions.
A fresh reviewer mutates each guarantee in a scratch copy and restores all bytes after each mutation.

Required mutation cases include altered original scores, post-force copies, skipped callbacks, duplicate or swapped request IDs,
wrong vocabulary order, missing rows, duplicate inconsistent scores, nonfinite values, wrong consumed prefixes, and partial-prefill rows.
Also mutate input hashes, model/KV dtype, phase labels, trace truncation, replay normalization, changed runner,
eager execution, missing graph intervals, changed scheduler shapes, and the production adapter call site.
Revert the launch dependency to an unversioned issue-3076 seam and require the
focused suite to fail before subprocess start. Remove the completed-request
release and resident-memory guard independently. Each mutation must fail before
the run can exceed its declared memory bound.
Each mutation must fail its intended regression rather than fixture setup.
The operator reruns the focused and full gates on the reviewed immutable head, then performs leased equivalence checks.

## Risks and stop conditions

Full-vocabulary output increases host memory, transfer traffic, and synchronization. Stream artifacts and measure actual resource headroom before collection.
The historical temporary environments are absent. Restore exact archived environments only through the operator's authorized lease workflow.
Restoration is not permission to repin engines, model weights, or dependencies.
The network share was restored on 19 September 2026 but has limited free space. Do not allocate large images for this diagnostic.
Stop hardware acceptance for absent production capture, changed numerical paths, mismatched prefixes, unsupported V2 behavior, or incomplete traces.
Keep the specific failed interface and evidence. Do not substitute eager execution or whole-prefix scoring.
No speed claim, tolerance, or issue closure follows from this design commit.

## Owed

This row owns diagnostic implementation, the mode-aware profile adapter,
capture and replay trace evidence, independent mutation review, leased
equivalence, oracle calibration, and the later ratification proposal. The
separately owned performance work supplies the reusable worker-start mechanism
and exactly matching ordinary-production baseline evidence.
