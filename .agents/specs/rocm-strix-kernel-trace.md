# Strix Halo decode attribution

Row: `BACKEND-GATE-ROCM-LLAMACPP`

Issue: [#3015](https://github.com/mudler/vllm.cpp/issues/3015).

## Now

Investigation on base `f98b638673b4d2edc0250eec56d229357ea38ab1`.
The user directed autonomous resumption on 2026-09-07. This spec precedes
the diagnostic harness. The operator selected the recorded, authorized local
merge path after independent review and operator verification.
The historical warm-clock correction is complete. New traces contain invalid
timestamps and remain unusable for timing attribution under #3040. The
switch-only measurement completed with independently reproduced diagnostic
pairs. Correctness remains failing and warm output equality remains pending.
Issue #3015 remains open.

## Scope

Measure the executing kernels on Strix Halo before selecting a product change.
Trace vllm.cpp and stock llama.cpp with the same profiler, artifact, prompt,
generation count, and single-request workload in one resource-controller lease.
Separate model loading, prefill, cold generation, and warm decode where the
trace permits it. Report unresolved boundaries explicitly.

Measure existing vllm.cpp switches independently against the default binary:
`VT_ROCM_Q8K_BLOCK=1` and `VT_ROCM_Q6K_SMALL_PRIVATE=1`. These are diagnostic
experiments for [#3018](https://github.com/mudler/vllm.cpp/issues/3018) and
[#3017](https://github.com/mudler/vllm.cpp/issues/3017). Do not change defaults.
The Q4/Q5 hypothesis remains owned by
[#3016](https://github.com/mudler/vllm.cpp/issues/3016).

## References and upstream anchors

- [Previous arm spec](bench-rocm-strix-vllmcpp-arm-head.md) records the
  workload and provenance failure this run must avoid.
- [Survey](../../docs/benchmarks/qwen38-27b-q4km-gfx1151.md) records earlier
  values and the failing correctness gate.
- [llama.cpp pin](../oracles/llama-cpp.md) requires stock `b10451`, commit
  `10bf611e` expanded and checked against the configured local oracle.
- [Primary pin](../upstream-sync.md) now requires vLLM `e126687a9`.
  Historical `5559679229` results cannot stand for the new pin.
- `src/vt/rocm/rocm_grouped_gemm.hip` contains `KQuantGemmK`,
  `KQuantDecodeCoopWarps`, `DotQ6K`, and the Q8_K selector.
- Read the stock llama.cpp executing quantized matrix-vector path before
  attributing a trace difference to its implementation.

## Design

### Correct the historical clock window

Issue #3015 also owns the clock-window ambiguity in the published lease-2
survey. Reuse its committed `n*.err.txt` timestamps and `clock-n*.jsonl.gz`
samples. Preserve the historical throughput values, raw artifacts, existing
`RESULT` literals, and `rederive.py` contract. Label the existing clock means
as whole-leg samples, including model loading and teardown.

Add a separate narrow reproduction command beside `rederive.py`. It reads
committed evidence by default and joins sample Unix timestamps to each
generation's inclusive start and end. Require exactly four ordered,
nonoverlapping generation windows per leg and valid samples in each window.
Report each generation, warm generations 2 through 4, and whole-leg values.
Pool samples without averaging per-leg averages. Test the command entry with
synthetic cold, warm, and outside-window samples before implementation.
Mutate cold exclusion and timestamp bounds to prove the tests distinguish
the reported windows. Run the command on committed lease-2 evidence and
publish its warm clock and activity values with the exact recipe.

The warm windows include prefill and generation. Busy percentage measures
sampled activity, not occupancy or a quantitative bound on host idle time.
Keep `TOKEN_GATE=FAIL` explicit. This correction proves no new throughput or
correctness result. A fresh reviewer checks the immutable change before landing.

### Collect new paired traces

#### Stop a live process when it reports a GPU fault

Issue [#3039](https://github.com/mudler/vllm.cpp/issues/3039) tracks delayed
fault handling in lease `43c3e415-e8e5-4918-ab42-b4d9c3dcab29`. The process
reported `HW Exception ... GPU Hang`, but the profiler did not exit. The
existing post-exit check therefore waited for the 1,200-second timeout.

During managed execution, scan newly appended bytes from this command's
stdout and stderr captures for `GPU Hang`, `Memory access fault`, and
`HW Exception`. Start after any existing capture bytes. Never scan argv,
`command.json`, or unrelated logs. Retain enough boundary bytes to detect a
diagnostic split across reads. Bound each read and scan a snapshot of the
current file length, so continuous output cannot indefinitely postpone the
existing timeout and size checks.

On detection, raise a named fatal-diagnostic error before accepting process
completion. Use the existing bounded cleanup for only this command's named
container. Preserve captures through the measurement `finally` path. Do not
reset a GPU, change inference defaults, or accept a faulted trace.

The red test launches a real CPU subprocess that writes a fatal diagnostic
and waits. Exercise the actual CLI entry, require prompt failure before its
timeout, and inspect cleanup and preserved evidence. Cover fragmented output,
stdout and stderr, and nonmatching pre-existing logs and command metadata.
Mutate the live scan, its call site, and evidence preservation. Run the
combined focused suites and full preflight on the final immutable commit.

#### Build and measure

Use `strix:gpu0` through a bounded `rc run`. The operator owns the lease.
Build from clean, asserted sources in unique worker-local directories. Use
ccache and at most four build jobs. Record revisions, archive hashes,
compiler versions, build commands, binary hashes, and inherited tuning flags.
Verify the artifact after copying to a unique worker-local directory:
`Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, SHA256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

Use prompt `The capital of France is`, greedy generation, 64 output tokens,
and one sequence. Run both engines through their production CLI paths.
Confirm both token counts and prompt processing from logs. A refused or
early-EOS run is not a comparable timing leg.

For each vllm.cpp switch, use three default/candidate pairs with alternating
order. Each process loads once and generates four times; discard generation
one. Compare emitted output and token counts before interpreting a timing
change. Preserve full logs. An output difference prevents acceptance of a
default change. Trace timings are diagnostic; use unprofiled runs for timing.

Sample clocks to worker-local files and fold samples inside recorded warm
generation timestamps. Retain whole-process measurements with explicit labels.
GPU activity percentages do not measure occupancy or prove a numeric bound
on host stalls. Kernel traces, not byte shares, determine kernel time shares.

### Run switch pairs without a new trace

An explicit `--phase switches` runs the existing 12 unprofiled legs from a
verified build state. This phase uses the same manifest, image, binary,
library, model, lease, and tuning guards as `measure`. It reuses the existing
alternating pair loop and result fold. It omits only the two baseline profiler
legs. The default `measure` phase continues to capture those traces first.

The operator reports that the unprofiled 64-token control completed. A
kernel-only control also completed, but its profiler reported swapped
timestamps. Issue [#3040](https://github.com/mudler/vllm.cpp/issues/3040)
owns the trace-validity gap. Switch results make no trace timing claim.
The new phase writes `trace-status.json` with `trace_run=false` and a
`PENDING` status naming #3040. It never labels an omitted trace successful.
The carried token gate remains `FAIL`, and warm output equality remains
`PENDING` under the existing CLI limitation.

The switches phase requires `--state` and never rebuilds or alters the
manifest. Existing trace directories do not block this phase. Existing
switch-leg directories still refuse reuse through the existing directory
creation guard. Runtime fault handling belongs to #3039 and does not change
in this slice. Actual CLI tests prove phase selection, identity rejection,
all 12 alternating legs, isolated tuning, and the pending trace label.

## Tests and gates

The implementer first proves harness validation rejects missing files, wrong
hashes, incomplete legs, and mismatched workload where the harness checks them.
The focused tests must fail before implementation and detect scratch mutations.
A fresh reviewer reviews an immutable commit and mutates each claimed guarantee.
Run the repository preflight and report every failure or omitted gate.
The operator reruns focused tests and the hardware recipe independently.

`TOKEN_GATE=FAIL` is carried from the previous survey, not remeasured by this
diagnostic run. No throughput ratio is a parity result. No default changes or
performance acceptance occur without the applicable correctness gate.

## Risks and stop conditions

Stop a hardware run on a GPU fault, lost lease, identity mismatch, or missing
profiler support. Install missing tools within the lease when practical.
Preserve completed evidence and identify the exact blocked measurement.
If the current vLLM pin cannot execute this ROCm workload, mark that arm
PENDING; do not substitute the historical pin and claim current parity.
Do not clear a quarantined device or use SSH without a lease.

## Evidence

Initial resource-controller inspection reports Strix ready and no running jobs.
Probe job `cf4724ba-1084-40a5-8699-8990ad9adbd7` confirms the staged artifact,
ROCm 7.2.4 directories, and the previous build image still exist.

### Diagnostic harness validation

The runner is `tools/bench/strix_kernel_trace/worker.py`. Its tests are
`tests/tools/test_strix_kernel_trace.py`, discovered by the existing tools
suite in preflight and continuous integration. The runner changes no defaults.

Run the focused gate with
`python3 -m unittest tests.tools.test_strix_kernel_trace`.
The first run exited 1 because the runner did not exist. After implementation,
four tests passed. The archive revision and llama count tests then failed with
two missing-function errors. After those guards were implemented, seven tests
passed. `git diff --check` exited 0.

Fourteen in-memory scratch mutations were detected by the focused suite:
file hash, file size, archive revision, llama pin, llama counts, generation
count, prompt count, duration agreement, cold exclusion, clock window,
pair completeness, output equality, process status, and runtime output bound.
The original runner remained byte-identical. This set is not a completeness
claim. The llama-pin mutation triggered the downstream archive parser as an
error, so independent review must verify that guard with a valid wrong-pin
archive too.

At the original harness checkpoint, the helper invoked preflight before
editing and reported its result as PENDING. The completed combined-runtime
validation is recorded in the hardware evidence below. Hardware compilation
and measurement are owned by the operator and are not established by these
Python tests.

The manifest supplies the model path, image, clock device, profiler command
array, and each source archive's full revision and SHA256. Create raw tar
archives with `git archive --format=tar <revision>`. The build verifies the
embedded commit ID and archive hash before extraction. The build phase writes
`build-state.json`. The measure phase requires that state and the identical
manifest, then rechecks the image, binary, library, and model hashes.

Run `python3 tools/bench/strix_kernel_trace/worker.py --phase build --manifest
<manifest.json> --output <build-output>` inside the operator's lease. Then run
the same command with `--phase measure --state <build-output>/build-state.json`
and a separate output directory. Both phases require `RC_DEVICE=strix:gpu0`
and `RC_JOB_ID`. Builds use ccache and four jobs in a new worker-local directory.

Matched baseline traces each run one generation. The llama production target
is `llama-completion`. At the pin, `tools/completion/completion.cpp:44`
documents `-no-cnv`, and `common/sampling.cpp:559` and `:574` print actual
sample and prompt counts. The runner rejects different counts before recording
matched traces. The profiler command comes from the operator's installed help.
Trace interpretation remains a manual gate.

The switch experiment checks emitted cold text and all completion counts.
Warm text equality remains PENDING because `examples/cli/main.cpp:321` prints
only the first completion. Clock summaries use generations 2 through 4 and
their emitted Unix timestamps. These windows contain prefill and generation,
not decode alone. The carried token gate remains FAIL.

### Review repair: container lifecycle and command-path guards

The first review rejected the harness on container cleanup, inherited image
tuning, output monitoring, and tests that did not enter the command path.
The repair runs each container with a unique name. A bounded `finally` block
stops and removes that name after success, process failure, timeout, or excess
output. Cleanup failure stops the harness with an error.

Both phases inspect `Config.Env` and reject `VT_`, `GGML_`, `HSA_`, `HIP_`,
`ROCR_`, and `PYTORCH_` variables. Containers run the inspected image ID.
Measurement records the image environment in `image-environment.json`.
The existing manifest and build-state format remain accepted. Measurement
rechecks the environment even when an older build state lacks `image_env`.

The host monitors the aggregate bytes in each leg directory, including logs
and profiler files, against 512 MiB every 100 ms. This is a sampled stop
threshold, not a strict disk quota. A writer can overshoot between samples.
The container's per-file limit remains an additional guard. Build commands
use the same lifecycle and monitor the build output directory.

The repair's red run exited 1 with missing container names and a missing
managed execution function. The focused suite then passed 20 tests. Tests
execute the actual `__main__` block with temporary archives, state, and logs.
External build and GPU commands are simulated. Real CPU subprocesses prove
timeout cleanup, failure cleanup, host-output detection, and profiler-file
detection. A real subprocess also exceeds host output through the CLI path.
Different cold and warm timings pin each pair's warm median and ratio.
Every vllm.cpp leg sets `VT_OP_PROVIDER_STATS=1` and retains the complete log.
An explicit `[vt reference-tier]` warning rejects a leg, as does a missing
kernel message. Absence of a warning alone does not prove zero fallback.

Repair evidence resides in `/tmp/strix-repair-red.log`,
`/tmp/strix-repair-green.log`, and `/tmp/strix-repair-mutations.log`.
All 35 scratch mutations were detected, including the 25 original reviewer
mutations. The runner remained byte-identical after the mutation run.
The final immutable head receives one full preflight run. Its result and
omitted environment or hardware gates belong to the handoff evidence.

### Second review: test command identity and artifact rejection

The second reviewer found eight surviving mutations in 43 checks at
`881a0199f`. A fresh test implementer reproduced those eight survivors before
editing. The repair changes tests and this evidence only. The runtime remains
byte-identical to `881a0199f`, SHA256
`bc3c23523f657b824897a433e8b1f1e394d7f08018a260b78d13580ef0ccae4a`.

The command tests now reuse the build state during measurement. Temporary
binaries and libraries carry real hashes. Separate cases corrupt the copied
model and tamper with each measured binary and library. The tests reject
missing and incorrect lease identities through the actual command entry.
Command assertions check the immutable image, profiler prefix, greedy flags,
prompt, token count, sequence count, alternating pair order, and four build jobs.
The wrong-pin archive contains an extractable member. Removing the pin guard
now fails with an unmet rejection assertion, rather than a tar parser error.
The CLI output-limit test requires at least 8192 captured stdout bytes, so
command metadata alone cannot satisfy its failure condition.

`python3 -m unittest tests.tools.test_strix_kernel_trace` passed 23 tests.
The baseline mutation run recorded eight survivors in
`/tmp/strix-test-repair-red.log`. After the repair, all 43 mutations were
detected in `/tmp/strix-test-repair-mutations.log`. The eight new detections
include intended assertion failures for the missing checks and changed
commands. The unchanged runner was checked after each full mutation pass.
`git diff --check` passed. The test-repair checkpoint reported full preflight
as PENDING. The completed combined-runtime validation is recorded in the
hardware evidence below.

### Historical warm clock correction evidence

Spec commit `2c09026ef` precedes the correction. The initial focused command,
`python3 -m unittest tests.tools.test_strix_clock_windows`, failed because
`clock_windows.py` did not exist. The completed command passed five tests.
Eleven scratch mutations were detected, including cold inclusion, either
missing timestamp bound, strict instead of inclusive bounds, omitted windows,
overlap, missing samples, invalid samples, averaging leg means, and deletion
of the command entry. The script remained byte-identical after every mutation.

Run the reproduction from the repository root:

```sh
python3 docs/bench-evidence/qwen38-27b-q4km-gfx1151-ourarm-head-20260905/clock_windows.py
```

The committed samples produce 3,216 whole-leg readings, mean 2228.2011815920396
MHz and 71.30783582089552 percent busy. Generations 2 through 4 retain 1,703
samples, mean 2872.811509101585 MHz and 100 percent busy. Per-generation and
per-leg details appear in the command output. The original `rederive.py`
still reports 16 checked claims and zero mismatches. Its committed fallback
was exercised by making only the shared `RESULT.json` existence probe return
false during execution. No file or original calculation changed.

Local evidence: `/tmp/strix-clock-window-red.log`,
`/tmp/strix-clock-window-green.log`, `/tmp/strix-clock-window-mutations.log`,
`/tmp/strix-clock-window-result.json`, and
`/tmp/strix-clock-window-historical-committed.log`. These are reproduction
outputs, not replacements for the committed raw records. That checkpoint
reported full preflight as PENDING. The completed combined-runtime validation
is recorded in the hardware evidence below.

### Live fault-stop repair evidence

Spec commit `bbc43c44f` precedes the #3039 runtime repair. The initial CPU
reproduction exited 1 with four timeout errors across the CLI and fragmented
stdout and stderr cases. Each writer emitted a fatal diagnostic and stayed
alive. Cleanup ran only after the three-second test timeout.

The repair opens independent readers at each capture's current end before
launching the child. It scans new bytes during the existing polling loop and
after the exit poll, with 64 KiB reads and a 64-byte boundary suffix. It reads
only a snapshot of the file size per poll. Metadata and earlier capture bytes
remain outside the scan. Detection uses the existing named-container cleanup.
The measurement `finally` path preserves stderr, command metadata, clock
samples, and partial profiler evidence without recording a successful trace.

`python3 -m unittest tests.tools.test_strix_kernel_trace tests.tools.test_strix_clock_windows`
passed 32 tests. Real CPU fault writers are rejected before the tests'
2.5-second bound, with a three-second timeout configured. Cleanup commands
are simulated in these CPU tests, so they establish neither GPU recovery nor
real Podman cleanup latency. No GPU run was used for this repair.

Local evidence resides in `/tmp/strix-fault-stop-red.log`,
`/tmp/strix-fault-stop-green.log`, and `/tmp/strix-fault-stop-mutations.log`.
All 52 mutations were detected: the earlier 43 guarantees and nine live-scan,
capture-boundary, and evidence-preservation changes. The runtime remained
byte-identical after the pass. The final immutable
head's preflight result and any skips belong to the implementing handoff.

### Switches-only phase validation

Spec commit `ff043df14` precedes the phase implementation. The new command
tests first exited 1 because `switches` was not an accepted phase. The focused
suite then passed 28 tests. All 50 scratch mutations were detected, including
the previous 43 and seven phase, trace-label, schedule, and state mutations.
The worker remained byte-identical after mutation. `managed_run` remains
unchanged from `95f690ff7`, so #3039 can supply its separate runtime repair.

Run `python3 tools/bench/strix_kernel_trace/worker.py --phase switches --manifest
<manifest.json> --state <build-output>/build-state.json --output <new-output>`
inside the operator's lease. The output records tracing as not run and pending
under #3040. This command establishes no hardware result until the operator
runs and validates the pairs. Evidence: `/tmp/strix-switches-red.log`,
`/tmp/strix-switches-green.log`, and `/tmp/strix-switches-mutations.log`.
The final immutable head receives one full preflight run, with its result and
omissions reported in the handoff.

### Hardware evidence

The [2026-09-07 capture bundle](../../docs/bench-evidence/strix-kernel-trace-3015-20260907/README.md)
preserves source and artifact pins, build logs, both failed profiling attempts,
the completed kernel-only inventory, and all 12 switch legs. Its checksum
manifests verify the packaged files and the original uncompressed captures.
The README reproduces the raw-data fold with the existing pinned worker.

The full trace reported a GPU hang and 16 timestamp swap warnings; the
completed kernel-only control reported 62. The latter contains 85,737 dispatch
rows, but neither capture supports kernel timing attribution. Scratch and VGPR
counts describe resource metadata, not spill traffic. A separate version probe
pins the installed profiler to the inspected timestamp-adjustment source.

Independent folding reproduces all 12 raw legs and six paired summaries.
Median default/candidate elapsed-time ratios are 1.067395 for
`VT_ROCM_Q8K_BLOCK` and 0.892823 for `VT_ROCM_Q6K_SMALL_PRIVATE`.
These unprofiled warm whole-completion measurements include prefill.
`TOKEN_GATE=FAIL` is carried rather than remeasured; warm output equality is
PENDING because the CLI exposes only cold text. No performance result or
default change is accepted.

Combined runtime `dcb5351cd` passed 37 focused tests. Independent review
detected all 59 mutations. Reviewer and operator full preflights each exited
0 with no failed checks and five argument-dependent skips: ARM ISA, CPU ISA,
CUDA fat-gencode, PR size, and Triton multiarch. Raw validation logs are in
the bundle. These completed results supersede the earlier pending checkpoints
without replacing their red-test and mutation provenance.

## Owed

The matched timing trace remains owed under
[#3040](https://github.com/mudler/vllm.cpp/issues/3040). The completed
kernel-only capture proves dispatch inventory only. Its timestamp warnings
prevent timing attribution. The delayed live-fault response in
[#3039](https://github.com/mudler/vllm.cpp/issues/3039) is repaired by the
runtime in this change, pending landing. A GPU reset is not part of the repair.

Issue [#3040](https://github.com/mudler/vllm.cpp/issues/3040) owns valid
profiler timestamps and the resulting kernel-time attribution.

The product fixes in #3016, #3017, and #3018 stay on their owning rows.
This issue does not close until the clock window and paired trace obligations
are both satisfied. The broader survey and correctness work remain #2921
and #2497.
