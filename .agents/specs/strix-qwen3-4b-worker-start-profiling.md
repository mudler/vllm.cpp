# Profile Strix Qwen3-4B workers from process start

Row: `BACKEND-GATE-ROCM-SGLANG`

Issue: [#3076](https://github.com/mudler/vllm.cpp/issues/3076).

## Now

The user approved worker-start profiling of both engines on 2026-09-19. This
document defines the diagnostic harness and evidence contract before its
implementation. It performs no GPU work, attributes no performance difference,
and changes no inference default, correctness threshold, or tolerance. The row
remains `INVENTORIED` until an implementation and its required review land.

Repository policy supplies the one-pull-request default for the later
implementation. No row-specific preference selects a split pull request. A
fresh implementer must work from this committed design, a fresh reviewer must
mutate the immutable implementation, and the operator must run every hardware
gate itself.

## Record placement

The campaign's earlier
`.agents/specs/strix-qwen3-4b-c4-performance.md` exists at commit
`b923ac2c4c24e9c608d4a6e02e868538be7536fa`, but not on current `main`.
That 555-line record includes an attach-first sequence and completed historical
campaign details. Restoring it would make stale campaign state current and
would obscure the narrower decision now being reviewed.

This file is the per-issue design for `ISSUE-GH-3076`. The generic owning row
continues to use `.agents/specs/competitive-benchmarks.md`; this design does not
change that row's lifecycle. This shape avoids an unrelated edit to a shared
keyed matrix and preserves the old record at its immutable commit. Inspect the
old record with:

```sh
git show b923ac2c4c24e9c608d4a6e02e868538be7536fa:.agents/specs/strix-qwen3-4b-c4-performance.md
```

## Problem

Two historical qualification corpora reported diagnostic concurrency-four
medians of 61.5947184118 output tokens/s for vllm.cpp and 79.5022690204 for
production vLLM. The token gate failed. These values are not accepted benchmark
results and cannot justify a product change.

The missing evidence is a complete matched trace from the actual GPU worker in
each production process tree. The earlier attach route passed only a safety and
identity preflight. It did not prove that late attachment observes HIP graphs
captured before attachment. Retained rocprofiler SDK source shows that a
previous `rocprofiler_configure` provider can activate profiling before the
attachment proxy queues are created. Retained process evidence identifies
Torch/Kineto as that provider in the production-vLLM target.

The earlier vLLM process-start experiment reached production
`FULL_AND_PIECEWISE` graph capture for sizes 1, 2, 4, and 8, but artifact
finalization failed. Its worker result grew to 490,690,366 bytes, the parent
HIP temporary file remained empty, and the profiler reported a ring-buffer
mapping error. Its partial archive cannot establish a complete request corpus,
complete graph replay, or clean shutdown.

Issue #3076 therefore needs a bounded, fail-closed worker-start profiler. It
must establish which process executed each GPU operation, prove that profiling
was active before runtime initialization, and finalize every required artifact
before the normal engine shutdown completes.

## Scope

In scope:

- one public diagnostic entry point and versioned input manifest;
- one launch owner for each arm, from profiler initialization through artifact
  finalization and production worker shutdown;
- production vLLM V2 and vllm.cpp process trees with their existing supervisor,
  worker, interprocess communication, graph, scheduler, and sampling behavior;
- the same pinned profiler build, configuration bytes, semantic trace
  categories, workload, and resource limits on both arms;
- a short readiness run before the full bounded corpus;
- fail-closed provenance, lifecycle, trace-completeness, scheduler-shape,
  graph-replay, and artifact-finalization checks;
- CPU red, green, and scratch-mutation tests before leased hardware work; and
- recovery from a host reboot using immutable source and evidence archives on
  the network-attached storage (NAS) volume.

Out of scope:

- late attachment to an initialized process;
- eager mode, vLLM V1, or a nonproduction execution denominator;
- `LD_PRELOAD`, an API shim, or another interposer that can change the
  numerical execution route;
- a global package or profiler install;
- a kernel optimization, default change, tolerance change, or performance
  attribution;
- treating instrumented time as accepted throughput or latency; and
- publishing a benchmark or closing the performance gap.

## Fixed inputs

The manifest must reject any different value instead of silently substituting
it:

| Input | Required value |
|---|---|
| vllm.cpp | `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb` |
| production vLLM | `e126687a9a828d513c01a07cd69f025f27d63280` |
| profiler harness repository | `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660` |
| nested rocprofiler SDK source | `97f5574fe2fdc7bef44fb01545347912ee9f1779` |
| profiler SDK | `1.1` |
| model | `Qwen/Qwen3-4B` |
| model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| dtype | BF16 |
| prompts | the six raw prompts in the retained qualification manifest |
| sampling | greedy, 128 requested output tokens |
| concurrency | one and four, with concurrency four required for attribution |

The implementation must bind every prompt byte and the whole ordered schedule
by SHA256. It must also bind the tokenizer and every model shard before and
after each arm. The retained qualification manifest already records these
values and is the recovery source, not a license to accept a changed file.

The profiler configuration is one canonical byte sequence. Both arms must
record the same configuration SHA256 and the same tool and SDK-library hashes
and build IDs. The required semantic categories are HIP runtime APIs, kernel
dispatch, HSA APIs and AQL queue or dispatch activity, graph capture and replay,
and external correlation markers. The implementation must derive the exact
supported rocprofiler spellings from the pinned build and store them in the
manifest. A missing category, ignored option, or differing resolved
configuration fails the pair.

The retained SDK 1.1 tool binary has SHA256
`478df9af09b74707652d9d5574ef37151ff1234d09c68843972c1409a505cdd0`.
The earlier environment recorded its registration library binary with SHA256
`40a5ecd8ca25dc3facb132b730066354812fe82d6889625d43371dbb0655b1ca`
and build ID `82dd8833b65c17523a3054f6b54da0e7a8831c82`. These values identify
built artifacts, not either source commit. The implementation must rebuild or
recover the pinned SDK binaries, verify these identities where the same
artifact is used, and record any newly built identity without treating it as
interchangeable with the retained binary.

## Source anchors

### Production vLLM

At vLLM commit `e126687a9a828d513c01a07cd69f025f27d63280`:

- `vllm/v1/worker/gpu/model_runner.py:1517` enters `execute_model`;
- `model_runner.py:1541-1554` derives the actual request and token shapes;
- `model_runner.py:1570-1580` dispatches compiled graphs and selects eager only
  for profiling or an explicit compiled-mode bypass;
- `model_runner.py:1591-1594` prepares persistent inputs and attention state;
- `model_runner.py:1664-1667` handles full-graph capture metadata;
- `model_runner.py:1731-1734` records actual batch size and the full-graph
  decision;
- `model_runner.py:1737-1743` enters full-graph replay;
- `model_runner.py:1766-1773` enters piecewise replay; and
- `vllm/v1/worker/gpu/cudagraph_utils.py:439-452` executes full-graph replay.

These are the production V2 anchors. The implementation must not replace them
with V1, `--enforce-eager`, or a synthetic model call.

### vllm.cpp

At vllm.cpp commit `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb`:

- `src/vllm/v1/worker/gpu/runner.cpp:3155` routes through
  `ModelRegistry::Forward`;
- `src/vllm/model_executor/models/qwen3_dense.cpp:85-103` enters the registered
  Qwen3 implementation and dispatches its graph at line 102;
- `src/vllm/model_executor/models/qwen3.cpp:1026-1040` opens the full-graph
  capture scope;
- `qwen3.cpp:1116` replays after capture; and
- `qwen3.cpp:1177-1186` resolves whether the static graph route is admissible.

The trace, not source inference, must establish the resolved vllm.cpp graph
mode and whether replay executed. A source candidate is not evidence that a
kernel or fallback ran.

### Profiler registration and attachment

The retained nested rocprofiler SDK source has these decisive anchors:

- `rocprofiler_register.cpp:335` resolves `rocprofiler_configure` with
  `dlsym(RTLD_DEFAULT, ...)`;
- `rocprofiler_register.cpp:342-363` accepts an existing provider before it
  considers loading another tool;
- `rocprofiler_register.cpp:790-840` creates the attachment proxy path only
  under its attachment and activation conditions, then propagates API tables
  to the active provider; and
- `sdk-attach.cpp` requests registration attach or detach but cannot recreate
  graph activity that occurred before its observation window.

The production-vLLM provider observation found both relevant GOT relocations
for `rocprofiler_configure` bound to the same symbol in `libtorch_cpu.so`. Its
status is `OBSERVED_NOT_COMPATIBILITY_PROOF`, and it is not a trace. Together
with the SDK source, it is enough to reject late attach as the authority for
already captured graphs.

## Evidence retained for recovery

All paths below are under
`/mnt/nas_share/rc/strix-four-engine-3053.X94a3J`. The implementation must
verify a retained file before using it and must copy no large archive to the
shared checkout.

| Evidence | SHA256 | Meaning |
|---|---|---|
| `qualification-manifest-12.json` | `4f0b0cbf0e06b7917405edc22bf852a31b037fce19361bde596841afda3c3f10` | pinned workload, model files, runtime, and built identities |
| `profiler-8952c3c9-source.tar` | `e4b6f98e097356e79848ec8640d980a8699f4ce0c2da36d6d740bfb88cc138a7` | profiler-harness repository source at `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660` |
| `vllmcpp-6e3cbfb-source.tar` | `df4ab94e5670ff7f7de2fe6427107153616643dbda9996b311398c91cfbb9fda` | vllm.cpp source pin |
| `oracle-c8d019447-source.tar` | `fa3ca302acb5eda709cfa567dd957411bee6a4a3e00f23632641e6130eb17f35` | retained vLLM oracle harness source |
| `got-provider-observation.jmEF9j/got-provider.json` | `48a8f328c3f1066464cddb4a64b0bd44a56b947592196e629f3cfc32594e5785` | Torch provider observation, not a trace |
| `vllm-matched-trace-06.log` | `9140e80cd1934b700b7d982a9d33b931f1a04cd2fd8aa008f795a2e71616938d` | process-start attempt and finalization failure |
| `vllm-trace06-preserve-07.log` | `dc5c732b9c69fd6d3bf31ace3735f07df673e3665f82aaac2a922c1b853f1d94` | preserved sizes and failure evidence |
| `vllm-trace06-partial-07.tar` | `fd25b19fbd02fc805e31c25a2ce8f1e21f6e1ad9b6135702d484b82cd4c44afc` | partial diagnostic output, never trace authority |

The harness at `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660` checked out the ROCm
monorepo and built `projects/rocprofiler-sdk` from nested source pin
`97f5574fe2fdc7bef44fb01545347912ee9f1779`. Historical build logs record
that relationship. The archive SHA256 binds the harness archive bytes, not the
nested SDK source or a built binary.

The separate retained SDK source anchor files include
`sdk-attachment-source.TktBaK/rocprofiler_register.cpp` with SHA256
`a54e43b6546c006b635b263012b7d90abbaf18f64d022dff1a8a9adafc255f0e`
and `sdk-attachment-source.TktBaK/sdk-attach.cpp` with SHA256
`df9e1241170289b2753b1bab4f3ab0c81f8038df4314ccd8b5212ed62fd7b0b6`.
The historical patched controller was only `BUILT_NOT_HARDWARE_VALIDATED` and
does not become part of this design.

If the host reboots, reconstruct only from these verified archives and the
pinned model cache on NAS. Use a fresh worker-local directory under `/tmp`, at
most four build jobs, and a project virtual environment under `/workspace`.
Copy finalized, bounded evidence back to a new NAS directory. Do not use a
global install, reuse an unverified build, or allocate the 61 GiB remaining on
NAS for duplicate model or source trees.

## Alternatives and decision

### Rejected: attach after the engine starts

The attach safety preflight proved process identity and avoided mutation. It
did not prove trace completeness. Torch/Kineto can register before attachment,
and the attachment proxy path is conditional. A trace that misses previously
captured HIP graphs fails the required graph-replay evidence. More retries
cannot change this mechanism.

### Rejected: profile only a generic parent process

Wrapping a supervisor is insufficient unless the profiler follows the actual
GPU worker from its first runtime initialization and produces a worker receipt.
A parent-only trace can be parseable while omitting the executing process. The
harness must reject that result rather than infer inheritance from a command
line.

### Rejected: preload or intercept runtime APIs

`LD_PRELOAD`, a HIP shim, or an API replacement can change library resolution,
capture, synchronization, or numerical execution. Such a result would not
measure the production route.

### Selected: one process-tree owner and a worker-owned start wrapper

The diagnostic entry point owns the normal production process tree but places
the pinned profiler wrapper at the GPU worker's existing launch seam. The
wrapper starts before that worker imports Torch or initializes ROCm, passes one
immutable profiler configuration and one run identifier, then transfers
control to the unmodified production worker target. An in-process startup hook
is admissible only when it is the first worker action before any runtime import
and has the same ordering and identity receipts as the executable wrapper.
The worker emits a startup receipt before runtime initialization and a shutdown
receipt after trace finalization.

For vllm.cpp, the production public adapter is the GPU owner. For vLLM, the
production V2 supervisor retains its existing EngineCore worker and
interprocess communication. Its worker wrapper must preserve the EngineCore
target, arguments, inherited file descriptors, process group, result channel,
error propagation, and shutdown order. It must not replace EngineCore with an
in-process model call. The harness identifies the owner from profiler records,
the worker receipt, and observed GPU activity; all three must agree. If no
production launch seam can start the pinned profiler before the normal worker,
implementation stops. It does not fall back to a generic parent-only wrapper or
attach.

This process-tree form is used identically on both arms. Engine-specific code
may decode existing logs or emit diagnostic markers, but it cannot change
model inputs, scheduler decisions, tensor types, graph policy, sampling, or
worker lifetime.

## Public diagnostic contract

The implementation adds one repository entry point:

```text
python3 tools/bench/strix_worker_profile/worker.py \
  --phase readiness|trace \
  --manifest <manifest.json> \
  --engine vllmcpp|vllm \
  --output <new-directory>
```

The command accepts no implicit engine, model, profiler, or workload defaults.
Arguments stored in the manifest are arrays and are never interpolated through
a shell. The output directory must not exist. Hardware phases require
`RC_DEVICE=strix:gpu0` and a nonempty `RC_JOB_ID`. The command rejects symlinks,
paths outside its declared worker-local and NAS roots, inherited tuning
variables, `LD_PRELOAD`, eager or V1 switches, and an unbounded profiler
command.

The manifest schema identifier is
`vllm.cpp/strix-qwen3-worker-profile/v1`. It contains:

- the engine name, complete source revision, source-archive path and SHA256,
  build recipe, executable path, library paths, and expected identities;
- the profiler harness revision and source archive, nested SDK source revision
  and version, executable, libraries, build IDs, configuration bytes and
  SHA256, resolved categories, and output limits;
- the model repository, revision, dtype, file paths, byte counts, and SHA256
  values;
- the six prompt byte strings and their hashes, tokenizer identity, sampling
  fields, concurrency schedule, warmup schedule, and expected 128-token stop;
- the permitted environment, ROCm library roots, device, lease, timeout,
  per-file limit, aggregate-output limit, and cleanup timeout; and
- the exact production command arrays and the expected supervisor and worker
  lifecycle roles.

Unknown fields, duplicate JSON keys, missing full revisions, relative paths,
and schema-version drift fail before a subprocess starts. The implementation
records the raw manifest and its SHA256 in every phase result.

## Lifecycle and observation window

One owner controls this sequence:

1. Verify the lease, NAS free space, manifest, archives, model files,
   executable, libraries, profiler, build IDs, environment, and new output
   directory.
2. Record a pre-run binding manifest, the boot ID, device identity, lease job,
   process limits, and monotonic and wall-clock start times.
3. Start the worker-owned profiler wrapper before the executing worker
   initializes Torch, HSA, or HIP. Require startup receipts from the owner and
   worker that prove this ordering while preserving the normal supervisor path.
4. Run the declared warmup. Open the recorded observation window only after
   warmup and close it after the last matched request completes.
5. Run the complete ordered corpus. Emit correlated phase, request-dispatch,
   request-complete, and scheduler-step markers. Record actual request, token,
   padded-batch, graph-batch, and active-sequence shapes from the executing
   scheduler or runner.
6. Request profiler finalization while the owner still controls the process
   tree. Wait within the cleanup bound for each required worker artifact to
   close and become parseable.
7. Record the worker's finalization receipt. Then allow normal production
   shutdown and record the worker and supervisor exit statuses.
8. Recheck every bound file and build ID. Hash each artifact, write a failure
   result for any discrepancy, and publish the complete result atomically only
   after every required check succeeds.

Profiling may observe process startup and graph capture, because those records
are required to prove readiness. Kernel-time summaries and request attribution
use only the bounded corpus window. Warmup activity is labeled separately and
never folded into the corpus. An early end-of-sequence, missing request,
duplicated marker, reordered request, or requested-versus-actual shape mismatch
fails the arm.

## Readiness and trace phases

`readiness` uses a fresh production process and the same profiler tool,
configuration, model, and route as `trace`. It runs one declared warmup and one
prompt at concurrency one. It has a 10-minute wall timeout, a 256 MiB aggregate
output stop threshold, and a 192 MiB per-file limit. It proves only:

- profiler initialization precedes runtime initialization;
- the profiler follows the actual GPU worker;
- every required semantic category produces a parseable record;
- graph capture or the resolved no-graph decision is explicit;
- a request marker joins to actual scheduler shapes and GPU activity; and
- finalization precedes normal shutdown.

Both arms must pass readiness before either full trace starts. A failed
readiness run is preserved and stops the campaign. It is not retried blindly.

`trace` uses a new production process for each arm. It runs the same declared
warmup and complete six-prompt, 128-token corpus at concurrency one and four.
Each arm has a 30-minute wall timeout, a 1 GiB aggregate-output stop threshold,
a 768 MiB per-file limit, and a 60-second cleanup timeout. The whole paired
stage has a 75-minute lease budget. These are stop thresholds, not exact disk
quotas; the monitor can observe a bounded overshoot between samples. A larger
bound requires a new reviewed design and a NAS capacity check.

Run the two trace arms sequentially inside one `rc` lease, on the same device,
with no unrelated GPU job. Alternate the first arm on a later reproduction if
the pair must be repeated. Never use SSH in place of the lease.

## Artifact schema

Each phase writes bounded logs and a final `result.json`. A failure writes
`result.failed.json` with the same provenance fields and the known partial
artifacts. It never renames that file to a passing result.

`result.json` contains:

- schema identifier, phase, engine, run identifier, manifest path and SHA256,
  start and end times, boot ID, lease job, device, and status;
- pre-run and post-run bindings for sources, model files, executable, every
  loaded engine and profiler library, configuration, and build IDs;
- the process tree with process IDs, start times, executable identities,
  lifecycle roles, exit statuses, and the uniquely identified GPU owner;
- profiler-start, runtime-start, worker-ready, observation-window,
  finalization, worker-exit, and supervisor-exit receipts in causal order;
- the resolved production mode, V2 status for vLLM, graph policy, capture
  sizes, replay mode, dtype, sampling values, and inherited environment;
- every warmup and corpus request, token counts, marker IDs, requested
  concurrency, actual scheduler shapes, graph selection, and matching trace
  correlation IDs;
- counts for HIP API, kernel, HSA/AQL, graph-capture, graph-replay, and marker
  records, including dropped-record and parser-diagnostic counts;
- a list of every artifact with relative path, semantic type, byte count,
  SHA256, producing process, finalization state, and parse result; and
- a fail-closed checklist whose fields name the evidence that satisfied each
  requirement.

Artifact paths are relative to the new output directory. The harness refuses
special files, symlinks, path escapes, duplicate paths, zero-byte required
files, artifacts still open at finalization, unrecognized process owners, and
any file whose recorded length or hash changes during the post-run check.

## Completeness rules

A trace arm passes only when all of these claims are supported by records from
the identified GPU worker:

- profiler initialization happened before Torch, HSA, or HIP initialization;
- the production engine mode, graph defaults, resolved graph mode, dtype,
  sampling, model, binaries, libraries, and build IDs match the manifest before
  and after the run;
- all six prompts ran once in their declared order at both concurrency levels,
  with 128 generated tokens unless the manifest's declared stop reason applies
  identically to both engines;
- every request and scheduler step has a unique external correlation marker
  and recorded actual shape;
- HIP API, kernel dispatch, HSA/AQL, and marker streams are parseable and have
  no dropped or truncated tail records;
- graph capture events recorded before the corpus remain visible, and every
  corpus replay joins to its dispatched kernels and request marker;
- a resolved no-replay route is explicit evidence, not the absence of graph
  records;
- worker trace files finalized before normal worker and supervisor shutdown;
  and
- both arms use byte-identical profiler configuration and workload schedules.

A trace that starts after graph capture, contains only parent-process events,
misses a category, lacks actual scheduler shapes, omits pre-existing graph
identity, leaves an unfinalized temporary file, exceeds a bound, or requires a
parser warning to ignore corruption fails. Partial records remain diagnostic
and may guide the next design, but cannot support attribution.

## Diagnostic timing and accepted timing

Profiler timestamps may describe ordering and trace shares within one arm.
They do not become the throughput denominator and must not be compared as an
accepted performance ratio. Instrumentation overhead can differ across process
trees and kernels.

Only after both complete traces identify one measured, reachable difference may
a separate implementation spec select a lever. Any performance acceptance then
requires uninstrumented, same-binary A/B runs on an idle leased host, the
declared correctness gate, identical model and workload, warmed repetitions,
and reported concurrency-one and concurrency-four throughput, latency, and
memory. This design changes no threshold or tolerance and declares no ceiling.

## CPU test-first implementation

The fresh implementer first adds
`tests/tools/test_strix_worker_profile.py`. The initial focused run must fail
because the public entry point or required guard does not exist. Preserve that
red result. Implement only enough to make the focused suite pass, then run the
full gate.

CPU tests use temporary files and real bounded subprocesses where lifecycle
ordering matters. They simulate the profiler and production process tree; they
do not claim GPU coverage. At minimum they prove:

- strict manifest parsing rejects duplicate keys, unknown fields, incomplete
  pins, wrong hashes, path escapes, symlinks, existing outputs, and engine or
  profiler mismatches;
- the pair validator rejects different profiler configuration bytes,
  categories, tool libraries, workload bytes, or resource bounds;
- the environment guard rejects eager, V1, `LD_PRELOAD`, tuning variables, and
  unexpected runtime library roots;
- a fake supervisor and GPU worker preserve their arguments, file descriptors,
  exit status, and shutdown order under the launch owner;
- a startup receipt after a simulated runtime-init marker fails;
- a parent trace without the GPU-worker receipt and GPU activity fails;
- missing HIP API, kernel, HSA/AQL, graph-capture, graph-replay, marker, actual
  shape, or pre-existing graph identity fails independently;
- a legitimate explicit no-replay decision remains distinguishable from a
  missing graph stream;
- duplicated, reordered, early-stopped, or incomplete corpus requests fail;
- dropped records, an unclosed file, a zero-byte required file, a parse error,
  a changing post-run hash, or finalization after worker exit fails;
- timeout, aggregate-size, per-file, fatal GPU diagnostic, and cleanup failures
  preserve bounded evidence and fail the command; and
- readiness cannot be reused as a full trace or as a timing result.

The fresh reviewer mutates each guard and its production call site in a scratch
copy. At minimum, remove the pre-runtime ordering check, worker ownership check,
one category check, graph-replay join, scheduler-shape join, finalization-order
check, post-run binding check, output bound, and eager or preload refusal one at
a time. The focused suite must detect each mutation. Restore the tree
byte-for-byte after every mutation.

## Hardware stages and gates

No hardware stage belongs to this design-only change. The later implementation
uses these gates in order:

1. Focused CPU red:
   `python3 -m unittest tests.tools.test_strix_worker_profile -v`.
2. Focused CPU green with the same command.
3. Full repository preflight with `scripts/agent-preflight.sh`.
4. Fresh static and mutation review of the immutable implementation commit.
5. Operator rerun of the focused suite and full preflight.
6. One leased readiness arm for vllm.cpp, then one for production vLLM.
7. One leased full trace per arm, sequentially, using the same accepted
   manifest and profiler configuration.
8. An offline pair check that rehashes both outputs, verifies every completeness
   rule, and emits a diagnostic comparison without a performance verdict.

The hardware report records every command exit, omitted gate, resource stop,
and failed completeness rule. A passing CPU suite cannot replace a hardware
receipt. An implementer or reviewer report cannot replace the operator's gate.

## Risks

- Profiler output can exceed the bounded budget before a sampled monitor reacts.
  Per-file and aggregate limits bound the overrun and preserve failure evidence.
- Production vLLM can spawn more than one Python child. Process-tree identity,
  startup receipts, and GPU records must agree on one executing worker.
- Profiler finalization can deadlock or fail after useful records exist. The
  cleanup timeout preserves partial artifacts but does not convert them to a
  passing trace.
- Diagnostic markers can perturb timing. They are accepted only for correlation,
  and instrumented timing is never the acceptance denominator.
- A reboot can invalidate worker-local builds. The recovery path rebuilds from
  verified NAS archives and rebinds every binary and library.
- Current NAS and local free space are narrow. The implementation must check
  capacity before each phase and must not duplicate model artifacts.

## Stop conditions

Stop without attribution or optimization when any of these occurs:

- the Strix lease is absent, lost, or shared with an unrelated GPU job;
- a pin, archive, model file, binary, library, build ID, configuration, or
  before-and-after binding differs;
- the pinned profiler cannot start before the production GPU worker initializes
  its runtime or cannot follow that worker without attach or interposition;
- the production V2 or normal vllm.cpp route cannot run with the declared
  defaults;
- a graph mode, actual scheduler shape, request marker, trace category, or
  worker owner is ambiguous;
- any record is dropped, corrupt, truncated, unfinalized, or exceeds a bound;
- the full corpus or token contract is incomplete or differs across arms;
- a GPU fault, host reboot, cleanup failure, or normal-shutdown failure occurs;
  or
- an observed difference requires a product decision outside this spec.

Preserve the bounded failure record and name the exact missing authority or
mechanism. Do not switch to attach, eager mode, V1, another profiler, a larger
unreviewed resource bound, or a different engine pin. Complete matched traces
are the prerequisite for the next decision, not evidence that any particular
kernel should change.
