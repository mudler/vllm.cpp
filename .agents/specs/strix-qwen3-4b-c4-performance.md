# Close the Strix c4 gap through shared graph execution

Row: `BACKEND-GATE-ROCM-SGLANG`.
Issue: [#3076](https://github.com/mudler/vllm.cpp/issues/3076).
Parent: [four-engine comparison](strix-four-engine-qwen3-4b.md), #3053.

## Now

The developer approves all five optimization paths on 8 September 2026.
The developer prioritizes graph replay and requests traces of pinned vLLM.
The design covers shared execution across accelerators, not a ROCm-only graph interface.
No accepted speed advantage exists.
The diagnostic c4 medians are 61.5947184118 and 79.5022690204 output tokens per second.
Correctness failed, so neither number is an accepted benchmark.
The target requires approximately 29 percent more throughput from vllm.cpp.
This campaign never treats that target as a predicted result.

The implementation base is `e576842bb52667579867b32468a81bd1b82dde81`.
The measured engine baseline remains `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb`.
The inspected graph and embedding paths are unchanged between those revisions.
The branch retains the earlier adapter and probe dependencies without claiming they landed.
Use one integration change per issue, with each implementation spec committed first.

## Scope and decisions

The selected approach reuses existing graph and persistent-input interfaces.
A kernel-first campaign can miss launch and synchronization costs.
A new graph runtime duplicates implemented CUDA, ROCm, and Tenstorrent mechanisms.
Neither alternative supplies a reason to replace the existing shared seam.

Investigate all five paths in the order below.
Implement a performance change only after a trace identifies its contribution.
A negative experiment remains recorded with its exact workload and rejected hypothesis.
Do not promise that every proposed optimization improves this model.

1. Reach safe production decode replay through the shared graph seam.
2. Remove avoidable allocation and synchronization in embedding validation.
3. Improve asynchronous device-token feedback without stale inputs or lost errors.
4. Optimize measured small-batch matrix operations and fusion dispatch.
5. Evaluate the head-size-128, GQA4 attention path on its actual executing shapes.

The first execution stage is the bounded operator trace preflight below.
This campaign spec does not authorize an unreviewed profiling harness or product patch.
Each product change needs its own bounded committed design under the existing owning issue.
Do not bundle unrelated kernel changes into graph admission.

## Fixed workload and reference

Keep Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
Keep BF16 model weights and the audited model artifacts from the parent spec.
Use the parent's six raw prompts, canonical token IDs, and 128 output tokens.
Use greedy sampling and concurrency 1 and 4.
Keep model capacity, prefix-cache policy, and all other matched settings unchanged.

Pin vLLM to `e126687a9a828d513c01a07cd69f025f27d63280`.
Use its production graph configuration, never an eager performance denominator.
Run native and vLLM traces with the same profiler and identical workload.
Do not compare a current-main vLLM path against the pinned engine.
Preserve generated-kernel evidence before ruling out a dispatch choice.

## Source map and existing ownership

Local anchors refer to the implementation base.

- `include/vt/backend.h:237-251` owns capture, replay, and graph handles.
- `src/vt/breakable_graph.cpp:143-156` replays segments and eager breaks in order.
- `src/vllm/v1/worker/gpu/runner.cpp:3155` calls the model registry.
- `src/vllm/model_executor/models/qwen3_dense.cpp:102` reaches dense graph dispatch.
- `src/vllm/model_executor/models/qwen3.cpp:1026-1050` captures the dense decode path.
- `include/vllm/model_executor/models/step_token_ids.h:58-103` refreshes persistent token inputs.
- `include/vt/persistent_step_input.h:209` exposes device-to-device input refresh.
- `src/vt/rocm/rocm_embedding.hip:125-142` allocates, synchronizes, reads, and frees the validation flag.
- `src/vt/cuda/cuda_ops.cu:994-1052` uses persistent error slots and deferred checks.
- `src/vt/rocm/rocm_paged_attn.hip:2047` excludes GQA4 from the fused branch.

CUDA's error-slot implementation is a local candidate, not the primary oracle.
Its deferred-error behavior and capture safety need independent verification before any port.
The Qwen3 async graph decline remains mandatory until its failure is explained.
The owning spec falsifies the old claim that token refresh alone removes the failure.
Existing Tenstorrent capture already embeds inside replay.
Do not erase that behavior during shared work.

| Backend | Existing state | Campaign obligation |
|---|---|---|
| CUDA | Capture and model admission exist | Preserve replay and diagnose async interaction |
| ROCm | HIP capture exists; model admission is false | Coordinate bounded device/model activation |
| Tenstorrent | Mesh traces and family-scoped defaults exist | Preserve captured inputs and request-boundary behavior |
| Metal | Capture capability is false | Separate backend implementation and hardware gates |
| Vulkan | Capture capability is false | Separate backend implementation and hardware gates |

Issue #3041 owns current ROCm activation.
Issues #3049 and #3056 own its capture-cleanup and checker repairs.
PR #2777 is earlier activation work; PR #2779 is earlier device-mirror work.
Their model, precision, and device differ from the Strix BF16 comparison.
Do not import their performance claims or edit their branches without coordination.
The shared graph row owns the unexplained asynchronous decline.
Existing Metal, Vulkan, and Tenstorrent rows retain backend ownership.

Only `strix:gpu0` is the currently resolved campaign device.
The developer's next-accelerator choice remains pending.
Shared source and CPU fixture review can proceed without that choice.
Hardware acceptance for another accelerator requires its own resolved device and oracle.

## Pinned vLLM execution chain

These anchors refer to the pinned vLLM revision.

- `vllm/v1/engine/async_llm.py:437,551` submits and generates requests.
- `vllm/v1/engine/core.py:665-700` schedules overlapped work when that mode is resolved.
- `vllm/v1/worker/gpu_model_runner.py:4344-4374` selects graph mode and padded shapes.
- `vllm/compilation/cuda_graph.py:240-260,357-361` dispatches and replays a graph.
- `vllm/model_executor/models/qwen3.py:158-169` reaches projections, normalization, rotary embedding, and attention.
- `vllm/v1/attention/backends/rocm_attn.py:452-480` dispatches paged attention.
- `vllm/platforms/rocm.py:410-421` checks the gfx1x custom-attention conditions.
- `csrc/rocm/attention.cu:3438-3442,3568-3573,3672-3681` defines the conditional Navi execution path.
- `vllm/v1/worker/gpu_model_runner.py:1883-1906,3834-3850` feeds device tokens into later inputs.
- `vllm/v1/worker/gpu_model_runner.py:322-361` copies output asynchronously and waits at consumption.

These anchors identify candidates, not measured kernel selection.
Record resolved attention layout, block size, asynchronous mode, and batch descriptors.
A requested concurrency of four does not mean every decode batch has four requests.
The six-prompt workload includes refill and tail behavior.

## Stage 1: establish a safe trace attachment {id: 1, deps: []}

The operator uses the existing persistent adapters and bound environments.
Every GPU operation runs inside an exclusive resource-controller lease.
Use `rc run -d strix:gpu0` with the stated runtime bounds.
Never reach a fleet device through an unleased shell.
Do not change the model, engine libraries, dependencies, or graph flags.

Pin ROCm profiler source to `97f5574fe2fdc7bef44fb01545347912ee9f1779`.
The worker profiler reports SDK 1.1.0.
The pinned attachment documentation supports `rocprofv3 --attach PID`.
The controller computes an attach-symbol offset from its own loaded registration library.
It adds that offset to the target's library mapping.
Therefore, equal library names do not establish compatible attachment.

Job `7f5d0bb6-0d19-4656-b8ab-decc8e548c68` measured different Torch and system registration libraries.
That job inspected Torch import, not the final GPU-owning EngineCore process.
Never reuse its import PID as an inference target.

Perform these read-only checks before injection:

1. Start the bound adapter with `ROCP_TOOL_ATTACH=1` before runtime initialization.
2. Configure and warm the production model through the existing protocol.
3. Resolve the GPU-owning process through ancestry, start time, command, and device ownership.
4. Record unique canonical register and SDK mappings from that process.
5. Record library hashes, ELF build IDs, and attach/detach symbol offsets.
6. Establish the controller's actual registration-library resolution.
7. Require identical registration bytes and offsets before any attach request.

Controller-side loader selection must not replace target libraries.
A path supplied in an environment variable is not proof of the library actually loaded.
Stop on ambiguous mappings, identity mismatch, or an unverified loader recipe.
Retain the failed preflight evidence and the owning issue.
Do not work around this refusal by injecting a guessed library.

Verify: the retained process identity and library evidence satisfy every check above.
This is an operator diagnostic stage, not a new repository implementation.
Any required collector or loader change needs a committed implementation supplement and test-first review.

### Approved supplement: verify the attachment controller before injection

The developer approved this supplement after the Stage 1 evidence in
[#3076](https://github.com/mudler/vllm.cpp/issues/3076#issuecomment-5591474605).
It implements the controller-local preflight described in that evidence.
The issue, row, integration shape, and engine pins remain unchanged.

Job `445e6bd4-9484-49a6-9f3c-b2c259018e8c` completed the c4 warmup on 8 September 2026.
Each of its six requests returned 128 tokens.
The warmed EngineCore mapped Torch's registration library and SDK, not the system copies.
Shutdown failed because adapter descendants survived.
The failure prevented post-run binding validation, so that validation remains pending.
No attachment or accepted timing occurred.

The pinned profiler touches the target before resolving its local registration symbol.
Upstream anchors below use `projects/rocprofiler-sdk/source/` at the profiler pin above:

- `lib/rocprofv3-attach/rocprofv3_attach.cpp:154` attaches through ptrace.
- The same file writes target data at lines 166 and 183.
- Line 191 requests the registration call.
- `lib/rocprofv3-attach/ptrace_session.cpp:197-229` selects the local library.
- Lines 767 through 805 resolve the symbol and calculate the target address.
- `bin/rocprofv3-attach.py:48-53` loads the controller library and calls its entry point.

The selected change patches that pinned controller, not either inference engine.
Track the patch under `tools/bench/strix_four_engine/patches/`.
Track its bounded build and preflight entry under `tools/bench/strix_four_engine/`.
Record the upstream revision, patch hash, build recipe, and actual controller dependencies.
Build into a new private directory. Do not overwrite the installed profiler or oracle environments.
Label all later captures as using the patched profiler on both engines.

The entry point performs these checks in the actual attachment controller process:

1. Resolve the registration library through the existing loader implementation.
2. Resolve both registration symbols through the implementation used for attachment.
3. Identify the actual loaded file, load base, build ID, and symbol offsets.
4. Read the target's identity and mappings without ptrace or target memory writes.
5. Require one unambiguous registration object in the target's mapping namespace.
6. Compare backing-file bytes and exported symbol offsets with the controller object.
7. Report the target SDK identity separately without substituting either target library.
8. Recheck the process start time and file identities before reporting success.

Compare bytes from opened backing files, not a basename or an environment variable.
Reject deleted, replaced, unreadable, malformed, or ambiguous mapped objects.
Resolve target paths through the target's filesystem namespace.
Reject missing symbols and symbols resolved from a different object.
Treat multiple segments of one mapped object as one object, not as duplicate libraries.
Group segments by backing-file identity and ELF load instance, including load bias.
Separate loads of identical bytes remain ambiguous.
Handle mapping offsets when deriving the load base.
Do not accept a stale process identifier after process exit or reuse.
If a required identity cannot be established, return a nonzero result with its reason.

Preflight mode never creates a ptrace attachment or calls registration attach or detach.
It never writes target memory, sends target signals, or changes target libraries.
A successful report describes the observed snapshot, not permission for a later unchecked attachment.
The attachment path runs the same check before its first target mutation.
It retains the resolved controller handle and repeats target identity checks before attachment.
Use the checked symbol addresses and load biases for actual attachment and detachment.
Do not recalculate those addresses through the existing first-basename mapping lookup.
Those checks are observations, not an atomic lock on the process or its mappings.
The operator parks the controlled worker between adapter requests and keeps it alive through detach.
The worker must not unload or replace registration libraries during that interval.
Retain process and backing-file handles through the guarded operation.
After seizing the target, revalidate its identity before target writes or injected calls.
On mismatch, release the attachment without injecting code or writing target data.
Report that seizure occurred. Do not describe this rejection as an untouched target.
If the runtime cannot maintain this boundary, refuse attachment and report the reason.
This contract does not claim safety against a hostile process that changes mappings concurrently.
Preserve ordinary attachment and detach semantics after the guard succeeds.
Do not add a bypass switch to the campaign entry point.

A standalone `ctypes` imitation cannot prove the controller's loader namespace.
An `LD_LIBRARY_PATH` recipe or `--echo` output cannot prove the selected library.
Target-library replacement changes the measured runtime and is excluded.
These alternatives remain rejected for the reasons established in Stage 1.

The smallest regression enters the patched controller's public preflight entry point.
Use CPU fixture libraries and a live child process. A GPU is not needed for these tests.
Exercise the real loader rather than replacing its selection with a test-only implementation.
An identical fixture must pass with the expected identity and symbol offsets.
A same-name library with different bytes must fail before target mutation.
Test missing symbols, ambiguous objects, nonzero mapping offsets, and deleted or replaced files.
Test unreadable mappings, malformed inputs, process exit, and identity changes during validation.
Instrument forbidden operations and require zero calls in every preflight case.
Exercise the guarded attachment entry with rejection before any ptrace operation.
Test success-report write failures and require a nonzero process exit.
The consumer requires both a complete report and exit zero.

The fresh reviewer deletes the production guard and mutates each identity comparison.
Each mutation must fail its focused test. Restore the scratch tree after each mutation.
The operator reruns the focused tests and the full repository gate at the reviewed head.
Verify with `python3 -m unittest discover -s tests/scripts -p 'test_rocprof_attach_preflight.py' -v`.
Run `scripts/agent-preflight.sh` as the full gate and name every omitted hardware gate.
Preserve upstream attachment tests and document harness adaptations at the pinned revision.

After local review, run preflight against the warmed production worker inside a Strix lease.
Do not attach if the registration identities differ or the SDK integration remains unverified.
Retain binding checks even when shutdown fails, without converting teardown failure into success.
Diagnosing or changing adapter shutdown behavior remains outside this supplement.
Stage 2 still requires finalized graph and kernel evidence before accepting a trace.
This supplement makes no correctness, replay-coverage, or throughput claim.

#### Controller preparation and CPU evidence

`tools/bench/strix_four_engine/rocprof_attach_preflight.py` supplies two commands.
`prepare --source SOURCE --output NEW_DIRECTORY` requires a clean checkout at the profiler pin.
It creates a private clone and initializes the pinned SDK submodules.
It applies the tracked patch from the clone root with an explicit SDK directory.
It builds the real `rocprofv3-attach` target with ccache and four compiler jobs.
The default preparation limit is 1,200 seconds, including source acquisition.
`--prefix-path` supplies resolved CMake dependency prefixes.
`--openssl-root` supplies private OpenSSL development files for SHA256.
Preparation never installs into an oracle environment or the system profiler.
`build.json` records the revision, patch hash, commands, controller hash, and resolved dependencies.
The build disposition remains `BUILT_NOT_HARDWARE_VALIDATED`.

`preflight --controller LIBRARY --pid PID --report NEW_FILE` calls the controller's public entry.
The Python wrapper does not reproduce the controller's loader or ELF resolver.
It requires exit zero and a complete controller report before publication.
It records the controller's actual mapped dependencies after the loader runs.
Its report is a snapshot, not authority for a later attachment.
The guarded attachment entry repeats identity checks and uses the retained symbol addresses.
The operator must still verify SDK integration before any hardware attachment.

The CPU harness compiles the pinned attachment translation units.
Its fixture copies retain the upstream MIT license and exact source bytes.
It substitutes declarations for logging, SDK types, and generated build configuration.
It does not substitute the loader or symbol resolver.
The address-propagation test intercepts ptrace at the system-call boundary.
It observes the actual register values for attachment and detachment without writing target memory.
This test does not prove that hardware attachment or trace finalization works.

The first focused run failed because the public preflight entry was absent.
A later nonzero-load-address fixture rejected `dladdr.dli_fbase` as the ELF load bias.
The implementation now uses the loader's actual link map.
The preparation regression exposed a successful `git apply` that changed zero files from the SDK subdirectory.
The harness now applies the patch through the production preparation function before compiling its fixtures.

The patch preserves upstream `tests/rocprofv3/attachment/attach-once` and `attach-twice` unchanged.
Their kernel, copy, HSA, and agent checks require a real GPU trace.
Those upstream hardware tests remain pending operator execution.
The CPU fixtures add preflight coverage and do not replace the upstream hardware tests.

#### Independent review repairs

The repair base is `b3cc253d6712fcf150c18bddbe45d5a3f02575c2`.
Independent review found incomplete report acceptance, ineffective identity regressions,
surviving preparation descendants, and unmapped registration code accepted as valid.
The earlier implementation chronology remains unchanged.

The wrapper now requires every identity field with its documented type and format.
Controller and target registration hashes and build IDs must agree.
Symbol offsets and calculated addresses must fit unsigned 64-bit values.
These checks validate the report contract, not a second ELF resolver.
Incomplete or inconsistent reports cannot reach publication.

Preparation gives each command its own session and process group.
Timeouts and command errors kill that group and wait up to five seconds for cleanup.
The fork regressions close the child's standard streams to prevent pipe ownership from hiding a survivor.
The fixture also bounds its own cleanup when the production cleanup is mutated.
This contract covers owned build descendants, not hostile children that escape the process group.

The controller now checks every file-backed byte of each required ELF load segment.
The actual registration addresses must lie in executable mappings at the correct file offsets.
Revalidation compares mapping permissions as well as addresses and backing identities.
The live fixture removes a symbol page after loading a multi-page text segment.
That fixture returned PASS before the repair.
Removing an interior page and changing executable permissions are separate regression cases.
The original non-executable live fixture already failed through the extra-mapping guard.
Its failure is not recorded as a previously accepted case.

The public CLI red runs and focused results are retained under
`/home/mudler/.cache/strix3076-repair-`:

- `red-f1.log`: exit 1, with 38 false-PASS assertions before report validation.
- `green-f1.log`: exit 0, two wrapper tests passed.
- `red-f3.log`: exit 1, both timeout and command-error descendants remained alive.
- `green-f3.log`: exit 0, both preparation tests passed.
- `red-f2.log`: exit 1, permission revalidation accepted changed mappings and reached ptrace.
- `red-f4.log`: exit 1, the unmapped symbol page produced a PASS report.
- `focused.log`: exit 0, all 20 tests passed in 8.070 seconds.

Independent syscall faults exercise mapped inode and device checks, incomplete extra loads,
exported offsets, and changes before target and SDK revalidation.
Deleting each corresponding guard makes the new tests fail.
The permission, symbol mapping, and complete-segment checks have separate mutation witnesses.
Deleting report validation, process-group cleanup, or session ownership also fails its focused test.
`mutations.json` records each command result and byte-for-byte scratch restoration.
Each mutation has its own `mut-NAME.log` with the intended assertion failure.

The startup preflight began before edits but continued while the repair changed.
Its log is diagnostic evidence, not an immutable baseline or final gate.
Run the final full preflight on the frozen repair commit before acceptance.
CPU fixtures do not establish real controller build compatibility, GPU attachment,
graph coverage, adapter teardown correctness, or benchmark acceptance.

### Loader reservation compatibility repair (#3090)

Issue [#3090](https://github.com/mudler/vllm.cpp/issues/3090) owns this repair under the same row.
It completes the approved preflight's handling of a valid shared-object layout.
Keep the existing controller entry points, profiler pin, and target-library restrictions.

Job `dba7164c-bf1f-4a6e-8194-d0e03404fc9d` completed the six-request c4 warmup.
The controller then refused `unaccounted mapped load instance`; it did not attach.
Before and after model/engine bindings matched; shutdown still reported a surviving descendant.
The SDK's recorded no-access page occupies relative addresses `[0x78c000, 0x78d000)`.
Its neighboring ELF segments end and begin at those page-rounded boundaries.
The current guard requires each mapping to fit inside one file-backed load segment.
That requirement necessarily rejects this reservation page.
The generic message does not independently identify which object emitted the rejection.

A CPU diagnostic at `c9ae31fb5e7511b48c54f0f4745dd76fcea11b5b` confirms the general defect.
A single ordinary `dlopen` of a fixture linked with `-z max-page-size=0x200000` creates reservation gaps.
The unchanged public preflight rejects it without forbidden operations.
Evidence: `controller-preflight-c9.wLC4Iu/cpu-gap-diagnostic-c9.json` in the campaign evidence directory.
Its SHA-256 is `986e9bb30543f487dd38dd5ab9b1dc370e83b7ab6272f5e1468fd615a8cd540b`.
This diagnostic does not replace the implementer's red-first regression.

Reference: glibc tag `glibc-2.39`, `elf/dl-map-segments.h:75-111`.
The loader reserves the shared object's span using its first load command.
It removes access from excess space before mapping subsequent segments.
Lines 135-173 describe zero-filled allocation tails, which are not reservation holes.
This reference establishes the mechanism, not the worker's exact libc revision.

Classify reservation mappings separately from loaded segment bytes.
Accept a reservation only when its permissions are exactly `---p` and its backing identity matches.
Require the already validated unique load bias and checked page/address arithmetic.
Require the entire mapping to lie in a proven hole between load allocations.
Use page-rounded `p_memsz` allocation bounds; do not mistake a zero-filled tail for a hole.
Reject overlap with any load allocation, mappings outside the object, and ambiguous load instances.
Require the offset relation established by the first load's reservation mapping.
Do not infer a reservation from permissions or membership in the object's overall span alone.
Keep full file-backed segment coverage, executable-symbol checks, and exact revalidation unchanged.
Identify the object and rejected mapping in a reservation refusal's diagnostic.

The fresh implementer first adds a public-entry regression using an actual loader-created gap.
Record its ELF headers and mappings so that the fixture proves the intended layout.
Add independent negatives for permissions, offsets, bounds, backing identity, and allocation-tail overlap.
Retain the extra-instance, missing-code-page, report-integrity, and revalidation tests.
The reviewer removes reservation acceptance and mutates each claimed classification guard.
Every mutation must fail the intended regression, with scratch restoration checked byte-for-byte.
Run the focused suite and full gate on the immutable result; the operator repeats the gate.
Rebuild the real controller and retry the same read-only Strix preflight with before/after bindings.

Reject the alternative of ignoring all no-access mappings: it hides unrelated mappings.
Leaving the guard unchanged remains a safe refusal, but does not complete this compatibility repair.
No SDK substitution, attachment-shim injection, teardown change, or throughput claim belongs to this repair.
Stop on an unproven loader layout or a material change to the existing safety contract.

Implementation evidence for #3090:

- The fresh public-entry test first failed at both virtual bases, `0` and `0x200000`.
  Both failures reported `unaccounted mapped load instance`; the forbidden-operation spy recorded no operations.
  `profiler3090-red.log` retains the actual ELF headers, mappings, commands, and failures.
- The focused command is `python3 -m unittest discover -s tests/scripts -p 'test_rocprof_attach_preflight.py' -v`.
  It passes 26 tests, including actual registration and SDK reservations.
  A prefixed, alignment-preserving ELF fixture also exercises a nonzero first-load file offset through real `dlopen`.
  The suite retains duplicate-instance, missing-code-page, report, and revalidation regressions.
- Twenty mutations fail their intended assertions, with byte-for-byte restoration after each mutation.
  They cover acceptance, permissions, start/end alignment, offsets, both bounds, allocation overlap, inode/device identity,
  both first-load origins, `p_memsz`, allocation rounding, page/size metadata, arithmetic overflow, diagnostics, and the production accounting call.
  `profiler3090-mutations-complete.log` has SHA-256 `f83e26eb52a08e3e2fb4f7f71d7ada674cb47d556bf3a9af975cd8ca8318f66d`.
  The initial sweep exposed redundant lower-bound checks and a rounding mutation masked by an earlier refusal.
  The final classification uses one lower bound; the allocation-tail test checks the exact rejected interval.
  The nonzero-offset fixture detects a hardcoded-zero origin that survived the initial fixtures.
- Logs reside under `verification-clean-3076.nWvRSX/reservations-3090` in the campaign evidence directory.
  The immutable handoff includes the full preflight result and explicit range checks.
  Hardware compilation artifacts remain outside these CPU tests.
  Independent review, the operator's gate, the real controller rebuild, and read-only Strix preflight remain required.
  This evidence makes no attachment, teardown, trace-completeness, or throughput claim.

## Stage 2: capture matched execution {id: 2, deps: [1]}

Attach the identical pinned profiler to one verified GPU owner per engine.
Use a unique host-local profiler temporary directory and separate output directory.
Prepare attachment before graph capture, then attach after model warmup.
Keep the profiler's standard input open.
Wait for the flushed `Press Enter to detach...` readiness message.
Submit one complete c4 corpus through the existing adapter protocol.
After its complete reply, detach with a newline while the engine remains alive.
Validate finalized output before normal engine shutdown.
Repeat the validated procedure separately at c1.

Bound each engine job to 20 minutes and each profiler output set to 1 GiB.
Record artifact bindings before and after the run.
Do not accept an exit code as proof that the trace is complete.
Do not suppress the existing adapter teardown failure.
Record process states before cleanup when descendants survive.

Require parseable finalized traces, complete corpus coverage, and the intended GPU process.
Require graph replay calls and corresponding replayed-kernel records.
An attach mechanism that misses previously captured HIP graphs fails this stage.
Reject dropped records, missing tails, zero-byte spools, and finalization errors.
The attachment interval describes the corpus, not inferred per-token boundaries.
Use explicit scheduling and correlation evidence for finer attribution.
If attachment fails these checks, specify a separate single-owner launch method before using it.
Never reuse the invalid multiprocess preload trace or switch to an eager denominator.

Verify: all trace conditions pass for both engines on the same workload.
Retain diagnostic token outputs separately from trace validity.
Profiler instrumentation cannot supply accepted throughput.

## Stage 3: prove shared graph correctness {id: 3, deps: [2]}

Coordinate existing owners before implementation.
Use the existing backend, model registry, graph, and persistent-input seams.
Separate backend capability from device/model admission.
Warm every required allocation and matrix workspace before capture.
Test changing tokens, positions, block tables, active slots, padding, and batch shapes.
Test request replacement, cancellation, slot reuse, recapture, and outstanding-work teardown.
Preserve backend-specific request boundaries and model restrictions.

Reproduce the async graph failure in a diagnostic variant before changing its mitigation.
Compare depth 1 and depth 2 with independently initialized eager and graph state.
Do not remove the production decline because a refactor looks correct.
A reviewer deletes the production replay call and freezes each mutable input.
Every claimed guarantee needs a failing regression and an effective mutation.

Verify: the owning scoped model and backend gates pass, including actual replay witnesses.
A GPU allocation, configured mode, or graph-capture log alone does not satisfy this stage.
Additional accelerator results remain pending their resolved devices and model gates.

## Stages 4 through 7: measure the remaining levers

After a complete trace, commit the smallest design for each measured candidate.
Keep separate controls for graph replay, validation synchronization, and token feedback.
Measure interactions after the individual changes, not by adding their percentage gains.
Preserve invalid-token reporting, dtype, sampling, and production reachability.

For matrix operations, compare actual entry points, algorithms, compute types, and output types.
For fusion, preserve intermediate rounding behavior and the shared fusion interfaces.
For GQA4, measure the exact BF16 head-size-128 workload rather than enabling a historical losing branch.
Record a rejected lever when it fails correctness or does not improve end-to-end time.
Each change needs fresh implementation, mutation review, and operator verification.

Verify: each scoped gate passes and the uninstrumented same-binary A/B measures its effect.
Trace results determine priority within these stages without removing any investigation.

## Acceptance and stop conditions

The parent correctness gate remains token-exact unless the developer ratifies a separate distributional gate.
Issue #3077 owns that calibration proposal; this spec supplies no numeric tolerance.
Issue #3075 owns llama.cpp causality and comparator repair.
Neither issue prevents diagnostic attribution, but both retain their applicable benchmark obligations.

Run the full four-engine comparison only after its correctness and teardown gates pass.
Use the parent warmup and three measured corpora at each concurrency.
Report throughput, latency, and memory with source, model, environment, and contention identities.
Require c4 throughput at least equal to production vLLM before claiming that target is met.
Keep every below-floor axis open.
Do not claim a ceiling, sum isolated gains, or change precision to manufacture a BF16 win.

## Profiler review boundary coverage repair (#3076)

The coverage-only successor to `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660`
changes no production code. Public preflight CLI fixtures now reject otherwise-valid
reports with float or mismatched PIDs, zero or overflowing start times, NUL paths,
63- or 65-character SHA hex, odd-length build-ID hex, negative or overflowing SDK
load biases, zero symbol offsets, and overflowing resolved symbol addresses.
Both registration symbols and each load-instance field have boundary witnesses.
Valid controls include minimum and maximum unsigned values where permitted,
the largest non-overflowing resolved address, and a one-byte build ID.

The successful public preparation fixture uses offline Git metadata and CMake
configuration shims. Real `git apply` patches the pinned source fixtures, and real
`g++` compiles those sources with the existing CPU harness headers. The test loads
the resulting controller and resolves its `preflight` export without attachment.
Removing the production patch call still builds an unpatched controller but fails
the export assertion. This fixture is not production SDK build evidence.

Focused verification: `python3 -m unittest tests.scripts.test_rocprof_attach_preflight`
passed 21 tests in 10.417 seconds. Independent scratch mutations removed PID type,
PID equality, each start-time bound, NUL rejection, SHA length, build-ID parity,
each unsigned bound, positive offsets, address-overflow rejection, and the public
preparation patch call. Each mutation produced exit 1; restoring the controller
CLI byte-for-byte returned the two affected tests to green. The first mutation
run supplies the intended red evidence for this coverage-only repair; no
production implementation changed between red and green.

Evidence is retained under the campaign NAS directory in
`profiler3076-boundary-focused.log`, `profiler3076-boundary-mutations.log`, and
`profiler3076-boundary-mutations-final.log`. The exact-head full gate and fresh
review remain pending at commit creation. Ordinary `/tmp` startup preflight is
not accepted as a replacement for the full gate in the private ext4 namespace.
No GPU job, hardware attachment, engine change, or performance claim is included.

## Owed

Issue #3076 owns complete matched traces and the open Strix c4 performance target.
Existing backend issues own their implementation slices and hardware deployment gates.
Additional accelerator scheduling awaits the developer's device priority.
This plan changes no runtime default and completes no performance acceptance obligation.
