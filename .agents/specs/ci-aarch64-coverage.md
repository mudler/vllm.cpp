# The suite runs on x86-64 and ships on aarch64 — decide which aarch64 lane to buy

Issue: [#1385](https://github.com/mudler/vllm.cpp/issues/1385)
Row: `GATE-CI-AARCH64-COVERAGE`
Branch: `row/CI-AARCH64-COVERAGE-1385`
Base SHA: `e2a9e035dbf8662f4bd87fc21a768d184f547c73`
State: `ACTIVE` (W0 decision committed at `e2a9e035d`; W1a and W1b land the
workflow change at `bacb71109`; W2 and W3 wait on the measurement)

**No matrix owns continuous integration (CI) infrastructure.** `MODEL`, `QUANT`,
`KERNEL` and `BACKEND` are the four matrix prefixes
`scripts/check-agent-record.py` recognises, and `ENG`, `KV`, `PAR`, `SAMPLE`,
`TOOLS`, `SPEC`, `SERVE`, `LORA`, `ATTN` and `LOAD` are the engine-matrix
prefixes. None of them owns a workflow file. This row therefore uses the
unplaced `GATE-` form that `GATE-CI-CONCURRENCY` and
[`main-verifiability.md`](main-verifiability.md) already use, and the missing
surface is recorded under `## Owed` rather than repaired by inventing a prefix.
Adding a prefix is a checker semantic change and needs its own row.

## Now

W0 answered one question and changed no workflow file: **which aarch64 lane
does this project buy, and what does it cost?** The recommendation is in
`## Recommendation`, and it deliberately landed in its own change so that a
broken `ci.yml` could not take every other row's gate down with it.

**W1a and W1b are now implemented, on branch
`row/GATE-CI-AARCH64-COVERAGE-W1` from base `bacb71109`.** Two things land:

- `build-test-cpu-arm64-full`, a new `ubuntu-24.04-arm` job that configures
  with the same four `-D` flags as the fast arm job, builds every target at
  `-j 2`, and runs `ctest` **serially**. It is `schedule` and
  `workflow_dispatch` only, and `continue-on-error: true`, exactly as
  `## Recommendation` states. It joins `baseline-summary`'s `needs:` list and
  `EXPECTED_JOBS`; see the note below.
- **Nine tokenizer targets on the existing per-pull-request arm job**, plus one
  step that executes them. This is the half that makes
  `test_tokenizer_parity` run on aarch64 for the first time and so closes the
  first `## Owed` item of
  [`prompt-token-divergence.md`](prompt-token-divergence.md).

**The counts, re-derived at `bacb71109`** rather than carried from
`e2a9e035d`. `tests/CMakeLists.txt` now defines **567** `vllm_cpp_add_test`
targets, all 567 names unique and all of them in that one file. **16** are
guarded off under the arm lane's flags — 11 `VLLM_CPP_SERVER`, 2
`VLLM_CPP_HIP`, and one each of `VLLM_CPP_METAL`, `VLLM_CPP_VULKAN` and
`VLLM_CPP_TENSTORRENT` — so **551 configure**. Configuring the tree locally
with those four flags and running `ctest -N` enumerates **577** CTest entries.
The fast arm job builds **4**, so the ratio is 4 of 567 targets, or **0.71 %**,
against the 0.72 % measured at the W0 base. The gap did not close on its own;
it widened by 15 targets.

**And it moved again during this row's own implementation.** At the merge head
that takes `origin/main` `849a7dd73` (#1805, #1806), the same commands read
**568** targets, **552** configuring, and **578** CTest entries enumerated with
those four flags. One target in about an hour. That is the argument for the
shape `## Recommendation` chose: the full-suite job keeps no list, so a new
test joins the aarch64 lane by existing, and nothing in `ci.yml` carries a
count a later pull request has to remember to bump.

**The duration model reads OPTIMISTIC against the one rate this row could
measure directly, and the recommendation survives it anyway.** W0 divided the
x86-64 runner's 36.3-minute build into about 11 minutes of library, server and
examples and about 25 minutes for the test executables, giving **2.7 s per
target at `-j 2`**. Measured here instead of divided: 20 previously unbuilt
targets, `cmake --build build-arm-probe --target <20 names> -j 2`, against an
already-built library, took **83 s wall, or 4.15 s per target**, on a Ryzen 9
9950X3D under a load average of 13.7. That is 1.5x the modelled rate on a CPU
faster than either hosted runner, so the model's per-target term is the
optimistic half of it, not the pessimistic one.

Carried through: 552 configuring targets at 4.15 s is about **38 minutes** of
test-executable build. Add the arm lane's own measured 8.0-minute library
build and the CTest wall, which W0 puts at about 10 minutes on x86-64 and about
14 if `test_ltx2_video` -- 43.0 % of that wall by itself -- runs 2x slower on
aarch64. So **about 55 to 75 minutes**, not 45 to 55.

**It still fits, and the decision does not move.** The budget under
`cuda-fat-build`'s measured 123.0-minute median finish, less the arm pool's
11.8-minute median queue, is about **111 minutes**. 75 is inside it with about
36 minutes of margin, and `timeout-minutes: 120` is still above the estimate
and below a whole cron interval. This is an INFERENCE from one local rate on
different silicon, not a measurement of `ubuntu-24.04-arm`. G1 remains the
thing that settles it, and G2's demotion rule remains the thing that acts on
the answer.

**The new job joins the baseline reader, which W0 did not spell out.**
`main-baseline.py::verdict()` grades **every** job the API payload carries, not
only the expected ones, so this lane moves the published verdict whether or not
it is named. Naming it is what makes `baseline-summary` *wait*: unlisted, the
summary can publish while the job is still running, and an unfinished job reads
as `pending`, which is not green either. That would have been a spurious
non-verdict introduced by this change, so the job is added to the `needs:` list
and to `EXPECTED_JOBS`, and the literal pin in
`tests/scripts/test_main_baseline.py` moves 12 → 13 with its reason.

**`main`'s baseline is RED for unrelated reasons on the day this was written**,
and `## Stop conditions` says a new job must not land into a red lane. Measured
2026-08-23 over the newest nine scheduled runs: every one is RED on
`windows-msvc-cpu`, `windows-msvc-vulkan` and `agent-record`, with
`macos-metal-mlx` `missing`. The new job's own conclusion is still readable —
`main-baseline.py` prints per-job names, so `build-test-cpu-arm64-full` will
appear by name in `failed:` or in the covered set. What is NOT readable while
those three are red is the aggregate verdict, so W2's promotion rule cannot be
evaluated from a green baseline until they are triaged. This is recorded, not
worked around.

## Scope

**In.** Re-derive the coverage gap on the current tree. Price each candidate
lane against measured Actions data. Recommend one lane, with its runner cost,
what it detects, what it does not detect, and what it obliges a future pull
request to do.

**Out.** The workflow edit itself. The prompt-token divergence of
[`prompt-token-divergence.md`](prompt-token-divergence.md), which this row can
give an execution lane but cannot diagnose. Any graphics processing unit (GPU)
lane: the fleet devices are reached through an `rc` lease and never through a
standing runner process, and `## Options rejected` records why.

## Our baseline — the counts, re-derived at the base SHA

Issue #1385 measured `528` targets at `c9724b5ee`. The tree has moved.

| Measurement | Value at `e2a9e035d` | Command |
|---|---|---|
| `vllm_cpp_add_test` targets | **552** | `grep -c 'vllm_cpp_add_test(' tests/CMakeLists.txt` |
| Targets defined anywhere else | 0 | `grep -rn 'vllm_cpp_add_test(' --include=CMakeLists.txt .` |
| Duplicate target names | 0 | 552 names, 552 unique |
| Direct `add_test(NAME ...)` registrations | 32 | 21 in `tests/CMakeLists.txt`, 11 in `examples/CMakeLists.txt` |
| Total CTest entries on the x86-64 lane | **584** | CI run 32465485947, `100% tests passed, 0 tests failed out of 584` |
| Targets built on the aarch64 lane | **4** | `.github/workflows/ci.yml:1121-1122` |

So the ratio is **4 of 552 targets**, or **4 of 584 CTest entries**, which is
0.72 % and 0.68 %. The four are `test_cpu_isa_arm`, `test_ops_matmul_elem`,
`test_ops_quant_dot` and `test_ops_quant_repack`, and all four are
instruction-set-architecture (ISA) dispatch and kernel-tier gates.

The aarch64 job runs no CTest invocation at all. It runs the four binaries
directly, eight times natively and six times under `qemu-aarch64`.

**The lane's own configure flags exclude more than the target list does.**
`.github/workflows/ci.yml:1110` sets `VLLM_CPP_BUILD_EXAMPLES=OFF`,
`VLLM_CPP_SERVER=OFF` and `VLLM_CPP_CUDA=OFF`. Counted by CMake condition:

| Condition | Targets | Reachable on the aarch64 lane today |
|---|---|---|
| unconditional | 535 | yes |
| `VLLM_CPP_SERVER` | 10 | no, the lane sets it `OFF` |
| `VLLM_CPP_HIP` | 2 | no, off on both lanes |
| `VLLM_CPP_METAL` | 1 | no, off on both lanes |
| `VLLM_CPP_VULKAN` | 1 | no, off on both lanes |
| `VLLM_CPP_TENSTORRENT` | 1 | no, off on both lanes |
| `CMAKE_SYSTEM_NAME STREQUAL "Linux"` | 1 | yes |
| `NOT WIN32` | 1 | yes |

**537 targets configure** under the aarch64 lane's current flags, and 4 build.
The 10 server targets and the 11 example CTest entries need a flag change as
well as a target-list change, which matters for the probe in
`## What it detects and what it does not`.

## Upstream chain

vLLM defines no CI lane for this project, so there is no upstream to mirror
here. The anchors are local, and they are the surfaces this decision reads:

- `.github/workflows/ci.yml:1096` — `build-test-cpu-arm64`, the only job on an
  aarch64 runner.
- `.github/workflows/ci.yml:1110` — the lane's configure step and its four
  `-D` flags.
- `.github/workflows/ci.yml:1121-1122` — the four-target `--target` list.
- `.github/workflows/ci.yml:1005` — `build-test-cpu`, the x86-64 lane that
  carries the correctness signal for the other 548 targets.
- `.github/workflows/ci.yml:764` — `cuda-fat-build`, the run's longest job and
  therefore the length of the critical path a new job must beat.
- `.github/workflows/release.yml:101` and `:234` — `cpu_arm64` and
  `cuda_arm64`, the shipped aarch64 bundles.
- `tests/CMakeLists.txt:18` — `vllm_cpp_add_test`, which registers one CTest
  entry per target and sets `SKIP_RETURN_CODE 77`.

## The cost model, measured

Every number below comes from the GitHub Actions REST application programming
interface (API) over the eight most recent `schedule` runs of `ci.yml`, read on
2026-08-21. The `schedule` lane is the right population: its long jobs are not
cancelled by the next push, so its durations are complete.

| Job | Runs | Queue delay min/median/max (min) | Duration min/median/max (min) |
|---|---:|---|---|
| `build-test-cpu-arm64` | 8 | 0.1 / **11.8** / 21.9 | 10.9 / **11.2** / 11.3 |
| `build-test-cpu` | 8 | 0.1 / **42.9** / 79.9 | 40.9 / **46.2** / 46.9 |
| `build-newest-gcc` | 8 | 0.1 / 16.4 / 81.7 | 18.4 / 34.2 / 41.7 |
| `sanitize-cpu (address,undefined)` | 8 | 0.1 / 20.3 / 67.1 | 16.7 / 74.0 / 77.6 |
| `windows-msvc-cpu` | 8 | 0.1 / 6.6 / 49.6 | 1.0 / 23.2 / 28.0 |
| `cuda-fat-build` | 8 | 0.1 / **23.0** / 74.0 | 97.8 / **100.0** / 105.4 |

Four facts follow, and each one changes the answer.

1. **The money cost is zero, and it is measured, not assumed.**
   `GET /repos/mudler/vllm.cpp/actions/runs/32465485947/timing` returns
   `total_ms: 0` for all 16 Ubuntu jobs and both Windows jobs. The repository
   is `PUBLIC`, so GitHub-hosted runners, including `ubuntu-24.04-arm`, bill
   nothing. The currency here is queue time and concurrency slots.
2. **`cuda-fat-build` owns the critical path, not `build-test-cpu`.** At the
   medians it finishes 123.0 minutes after the run starts, against 89.1 minutes
   for `build-test-cpu`. Any new job shorter than 123 minutes from run start
   adds **zero** wall-clock time to a pull request.
3. **The aarch64 pool is the less contended pool.** Its median queue delay is
   11.8 minutes against 42.9 for the pool that serves `build-test-cpu`. Adding
   work to the aarch64 pool is measurably cheaper than adding it to the x86-64
   pool.
4. **A whole scheduled run already costs about 357 job-minutes.** The eight runs
   sum to 228, 344, 345, 347, 390, 396, 399 and 408 job-minutes.

**The build dominates, not the test run.** On the x86-64 lane the `Build` step
took 36.3 minutes and the `Test` step took 10.1 minutes. On the aarch64 lane
the `Build` step took 8.0 minutes for the library plus four test executables,
and the two execution steps took 3.0 minutes together.

**Estimated cost of building all 537 configurable targets on aarch64.** Take
the x86-64 lane's 36.3-minute build as the library, the server, the examples
and 552 test executables at `-j 2`. Take the aarch64 lane's 8.0 minutes as the
library plus four executables. Attributing about 11 minutes of the x86-64 build
to the library, the server and the examples leaves about 25 minutes for 552 test
executables, which is about 2.7 seconds of wall time each. At the same rate the
537 aarch64 targets cost about 24 minutes, so the aarch64 build is about
**32 minutes** and the whole job is about **45 to 55 minutes**.

**This is a model, not a measurement, and `## Work breakdown` W1 replaces it
with one.** The model assumes that a GitHub-hosted `ubuntu-24.04-arm` runner
compiles at about the rate of a GitHub-hosted `ubuntu-latest` runner. Nothing in
this tree measures that. The estimate is used only to choose which option to
measure first, and the decision rule in `## Gates` demotes the job if the
measurement lands above the critical path.

**The CTest wall is dominated by one test.** Summing the per-test times of run
32465485947 gives 605.93 seconds against the reported 606.55, and
`test_ltx2_video` alone is 260.62 seconds, or 43.0 % of it. The next four are
`test_ops_quant_fp8_group_cpu` 74.56, `test_mla_attention_block` 32.36,
`test_serve_low_tools` 31.02 and `test_chat_mm` 26.66. A 2x slowdown of that one
test moves the aarch64 CTest wall from about 10 minutes to about 14.

## Options considered

### A. A curated aarch64 subset, per pull request

Extend the `--target` list of the existing job with a chosen set.

The principle has to be stated before the set, or the set is arbitrary. The
correct principle is **select the targets whose behaviour the x86-64 lane cannot
falsify**, which is the set of divergences the C++ standard and the two
application binary interfaces (ABIs) permit:

1. **The memory model.** aarch64 is weakly ordered and x86-64 is
   total-store-order. A missing acquire or release annotation is invisible on
   x86-64 forever and can reorder on aarch64. This class is larger than the
   tokenizer and nothing in the tree tests it.
2. **`char` signedness.** `char` is unsigned on aarch64 and signed on x86-64, so
   byte-oriented parsing can diverge: the tokenizer, UTF-8 handling, GGUF,
   safetensors and JSON.
3. **Floating-point contraction.** The two targets contract multiply-add
   differently by default, which moves accumulation results.
4. **Hash and pointer ordering.** A container iteration order that depends on
   an address can rank differently, which changes a merge order rather than
   losing data.

**Measured, this principle does not produce a small set.** Applying regular
expressions for classes 1, 2 and 4 over each target's own translation units
selects 38, 266 and 22 targets, and the union is **291 of 552, or 52.7 %**. The
grep over-selects, because naming `nlohmann` is not the same as depending on
`char` signedness. It still bounds the answer: a principled sensitivity filter
selects about half the suite, so a "curated subset" saves about half the build
and gives up the other half of the coverage. **Rejected**, because half a suite
costs most of a full suite and answers a question the full suite answers
completely.

### B. The full suite on a schedule, never per pull request

A second aarch64 job that builds everything and runs CTest, restricted to
`schedule` and `workflow_dispatch`.

Cost: about 45 to 55 minutes per scheduled run, six runs a day, so about 5 to 6
aarch64-hours a day and **zero** pull-request cost. It also lands inside the
population that `scripts/main-baseline.py` already reads, so its verdict is
published by an existing tool with no new reader.

Its weakness is detection latency. The cron is `17 */4 * * *`, so a defect can
be up to four hours and, at the recorded rate of about 55 pushes a day, about
nine commits old before anything reports it.

### C. A native-runner tier, or the fleet through `rc`

**Rejected on policy, not on cost.** `AGENTS.md` requires an `rc` lease for
`dgx:gpu0`, `thor:gpu0` and `orin:gpu0`. A self-hosted GitHub runner is a
standing process that takes the box outside the lease, which makes the fleet
report the device free while a job is on it. That is exactly the collision
`.agents/environment.md` records, and it already voided a speed axis in
`.agents/specs/minimax-music3.md` §13.10 on 2026-08-17. A second mutex that
cannot see the first is worse than one.

A non-fleet aarch64 machine would avoid the policy collision, but this row can
name no such host. Under `AGENTS.md`, a host name read from a document is
another developer's resolved value and never a default, so this option stays
`PENDING` on one question to the developer and is not recommended.

## Recommendation

**Buy option B first, then promote it to a per-pull-request job once its
duration is measured.** Add one new job, `build-test-cpu-arm64-full`, that
configures the tree on `ubuntu-24.04-arm` with the same flags the existing job
uses, builds every target, and runs `ctest`. Land it `schedule` and
`workflow_dispatch` only, with `continue-on-error: true`. Promote it to every
event, and remove `continue-on-error`, after `## Gates` G2 and G3 pass.

Alongside it, add the nine tokenizer targets to the existing per-pull-request
aarch64 job. Their measured x86-64 execution time is 21.06 seconds in total:
`test_bpe` 10.28, `test_tokenizer_parity` 6.37, `test_tokenizer_parity_gpt4o`
1.76, `test_tokenizer_parity_deepseek` 1.74, `test_tokenizer_parity_mistral`
0.72, `test_pretokenizer` 0.15, `test_detokenizer` 0.02, `test_tiktoken_bpe`
0.02, plus `test_bpe_equivalence` 7.64. Their build cost at the model's rate is
about 25 seconds. The job has about 30 minutes of headroom under the critical
path, so this is free.

**Why this shape and not the full suite per pull request immediately.** The
model says the job fits under the critical path with about 60 minutes to spare,
and the model is unmeasured. Landing it schedule-only costs nothing if the model
is wrong and produces the measurement that decides the promotion. The repository
has the precedent: `.github/workflows/ci.yml` lands `sanitize-cpu` with
`continue-on-error: true` for its first appearance so that a pre-existing finding
cannot block unrelated work, and removes the flag once the findings are triaged.

**Why not a curated subset at all.** The measurement in option A settles it: a
defensible selection principle picks about half the suite, and half a suite still
pays most of the build. A subset also needs a list, and a list that every new
test must join is a shared file that every pull request writes, which
`AGENTS.md` `## Records` forbids. The full suite needs no list.

### The runner cost, stated plainly

| Item | Cost |
|---|---|
| Money | **$0.** Measured: `timing` reports `total_ms: 0` for every job; the repository is public |
| Added job-minutes per scheduled run | about **45 to 55**, against a measured median of 357, so **+13 to +15 %** |
| Added job-minutes per pull-request run, W1 | **0** for the full-suite job; about **0.5** for the tokenizer targets |
| Added wall-clock to time-to-green, W1 | **0** |
| Added wall-clock to time-to-green, W2 | **0** while the job finishes inside 123.0 minutes from run start; the arm pool's median queue is 11.8 minutes, so the budget for the job itself is about 111 minutes against an estimate of 45 to 55 |
| Aarch64 runner hours per day, W1 | about **5 to 6** at six scheduled runs |
| Aarch64 runner hours per day, W2 | about 5 to 6, plus about 50 minutes for each pull-request run |

## What it detects and what it does not

**It detects, for the first time, any of these on aarch64:** an engine, model,
loader, sampler, quantization-arm, tokenizer, attention or key-value-cache
defect that reproduces on the committed fixtures. 537 targets configure under
the lane's flags and 581 of 584 CTest entries executed on the last green x86-64
run, so the fixtures are almost entirely present rather than skipped.

**The #1355 probe: would it run?** Partly, and the honest answer has three
parts.

1. **`test_tokenizer_parity` executes on aarch64 under this proposal, and it is
   a real gate there.** It is hermetic: `tests/parity/goldens/tokenizer_qwen36/`
   commits `tokenizer.json`, `corpus.txt` and `encodings.json`, the test loads
   them from `PARITY_GOLDENS_DIR`, and it compares **identifiers**, not counts,
   plus a byte-exact decode round trip. It has no skip path. That closes the
   first item under `## Owed` in
   [`prompt-token-divergence.md`](prompt-token-divergence.md), which states that
   the tokenizer parity goldens never execute on Arm.
2. **It is not the discriminating test, and it may not fire.** The divergence
   correlates with combining marks: the prompts it was measured on carry 74 to
   150 each. Measured here, the committed corpus carries **30 combining marks in
   total** over 99 lines, with a maximum of **13 in any one line**. So the
   golden corpus is one or two orders of magnitude weaker than the input that
   produced the anomaly. Running it on aarch64 is a necessary condition, not a
   sufficient one.
3. **The test the divergence spec actually names does not build on this lane.**
   That spec's discriminating step is `examples/tokenize` compiled natively on
   aarch64 over the same prompt bytes, and the aarch64 job sets
   `VLLM_CPP_BUILD_EXAMPLES=OFF` at `.github/workflows/ci.yml:1110`. The same
   flag block sets `VLLM_CPP_SERVER=OFF`, which puts the third named mechanism,
   the `/v1/completions` request-parse segment, out of reach as well. Neither is
   fixed by a target list. Both are recorded under `## Owed`.

**It does not detect:**

- Anything that needs a graphics processing unit or a staged checkpoint. Three
  CTest entries already report `Skipped` on x86-64 for that reason, and a
  GitHub-hosted aarch64 runner has no NVIDIA device.
- CUDA code generation for `sm_121a` or `sm_110`. The runner has no CUDA
  toolkit and no device. `cuda-fat-build` compiles those architectures on
  x86-64, and only the fleet executes them.
- A defect that needs a real checkpoint to express. The suite's fixtures are
  synthetic or small by design.
- A data race that the weak memory model permits but does not exhibit on this
  particular runner. Execution on aarch64 makes a reordering possible; it does
  not make it certain. A thread-sanitizer lane on aarch64 would, and it is
  recorded under `## Owed`.

## What it obliges every future pull request to do

**Nothing.** This is a deliberate property of the recommendation and the reason
option A was rejected on records grounds as well as cost grounds.

- The full-suite job builds the default target set, so a new test joins it by
  existing. There is no list to append to and no counter to bump.
- The per-pull-request tokenizer addition is nine names in `ci.yml`. A new
  tokenizer test does not have to join them, because the scheduled full-suite
  job already runs it.
- No ratchet moves. `RUNNABLE_BASELINE` in `scripts/check-gate-commands.py` is
  keyed on matrix rows, and this row is not one. `UNOWNED_HIGH_WATER` in
  `scripts/check-agent-record.py` does not move, because the index row this
  change appends names an owning row.

The one new obligation falls on the operator, not on a contributor: a red
aarch64 job is a real red and must be triaged, not muted. W2 removes
`continue-on-error` precisely so that it cannot be ignored.

## Design

W2 adds one job. It reuses the existing job's flags so that the two lanes
configure the same tree, and it is a sibling rather than an extension so that
the fast four-target gate keeps its current 11.2-minute duration.

- `runs-on: ubuntu-24.04-arm`.
- The same four `-D` flags as `.github/workflows/ci.yml:1110`, so the two
  aarch64 lanes cannot drift.
- `cmake --build build-arm-full -j 2`. The `-j 2` bound is not a guess: the
  x86-64 job's own comment records that a bare `-j` kills the runner with
  `ld` signal 9 during the parallel link of the test executables.
- `ctest --test-dir build-arm-full --output-on-failure`, **serial**. This tree
  has recorded tests that starve under `ctest -j`, and `.agents/verification.md`
  requires a serial re-run before calling such a failure a regression. A lane
  whose first duty is to find genuine aarch64 defects must not add a second
  source of red.
- A job-level `concurrency` group in the same shape as every other job in the
  file, keyed on `github.run_id` for `schedule` and `workflow_dispatch` and on
  `github.ref` otherwise, with `cancel-in-progress` off for the two
  non-cancellable events.
- `if: github.event.action != 'closed'`, matching every other job, so a closed
  pull request enters the concurrency group without gating.
- `timeout-minutes: 120`. `cuda-fat-build` uses 180; 120 is above the estimate
  and below the point where a hung job holds a slot for the whole cron interval.

W1b appends nine names to the existing job's `--target` list at
`.github/workflows/ci.yml:1121-1122` and adds one execution step that runs the
nine binaries.

## Port and harness map

There is nothing to port. The work is a workflow edit plus its checkers.

| Change | File |
|---|---|
| The new full-suite aarch64 job | `.github/workflows/ci.yml` |
| The tokenizer targets on the fast aarch64 job | `.github/workflows/ci.yml` |
| Workflow-shape assertions | `tests/scripts/` |

## Tests to port

None from upstream. The tests this row's implementation owes are in `## Gates`.

## Dependencies

- No graphics processing unit, no lease and no `rc` job.
- No new secret and no new runner label. `ubuntu-24.04-arm` is already in use at
  `.github/workflows/ci.yml:1096`.
- The measurement in G1 needs authority to run `gh workflow run ci.yml --ref
  <branch>` on a task branch. That authority is not assumed here.

## Risks

1. **Targets that do not build on aarch64.** `test_cpu_isa_x86` is the obvious
   candidate. This is the expected first finding, not an argument against the
   lane, and it is why W1 lands `continue-on-error: true`.
2. **Targets that build and fail on aarch64.** Also expected, also the point.
   Each one is a defect on the shipping architecture that x86-64 could not see.
   Each gets its own issue and its own row rather than a mute.
3. **The duration model is wrong.** Mitigated by G1: the job is measured on a
   branch before it lands, and by G2's demotion rule.
4. **Concurrency-slot contention.** The measured aarch64 queue is 11.8 minutes
   median against 42.9 for the x86-64 pool, so the aarch64 pool has headroom
   today. If a hosted-runner cap is shared across pools, W2 raises everyone's
   queue. G3 measures the pull-request queue delay for a week after promotion
   and demotes on a regression.
5. **A duplicate YAML key yields zero jobs.** PyYAML parses a duplicate mapping
   key without complaint and GitHub rejects the file, so a local parse is not
   evidence. `## Gates` G4 states the two checks that are.
6. **A flake becomes the lane's reputation.** A lane that reds for a reason
   nobody trusts gets ignored, which is worse than no lane. Serial `ctest`, the
   `-j 2` build bound and `continue-on-error` on first landing are all aimed at
   this.
7. **The tokenizer targets pass on aarch64 and the divergence is still open.**
   This is the likely outcome, given the corpus measurement above. It must not
   be read as evidence that the tokenizer is correct on aarch64. The
   `## Owed` items carry the rest.

## Gates

The implementation row runs these. This row runs G0 only, because it changes no
workflow file.

**G0 — the record gates, on this spec.** Green means the spec, the issue index
row and the anchors are consistent.

```sh
scripts/agent-preflight.sh --quiet
```

**G1 — the duration measurement, before the job is required.** Push the branch
carrying the new job and dispatch the workflow against it, then read the job's
own duration and queue delay from the API rather than from a summary.

```sh
gh workflow run ci.yml --ref row/CI-AARCH64-COVERAGE-1385-impl
gh api "repos/mudler/vllm.cpp/actions/runs/<RUN_ID>/jobs?per_page=100" \
  -q '.jobs[] | select(.name=="build-test-cpu-arm64-full") | [.name,.conclusion,.started_at,.completed_at] | @tsv'
```

**G2 — the promotion rule, stated before the measurement.** Promote the job to
every event only if its median duration plus the aarch64 pool's median queue
delay stays below 123.0 minutes from run start, which is `cuda-fat-build`'s
measured median finish. Above it, the job stays `schedule` and
`workflow_dispatch` only, and this spec's `## Outcome` records the measured
value and the demotion.

**G3 — the contention rule, after promotion.** Re-measure the pull-request queue
delay of `build-test-cpu` and `build-test-cpu-arm64` one week after promotion,
over at least eight runs. A median above the recorded 42.9 and 11.8 minutes is a
regression and demotes the job back to `schedule` only.

**G4 — the workflow is valid on the forge, not only in PyYAML.** A duplicate
mapping key parses locally and yields zero jobs on GitHub, so a local parse is
not evidence. Three checks, in order:

```sh
python3 -c "import yaml,sys; yaml.load(open('.github/workflows/ci.yml'), yaml.SafeLoader)"
actionlint .github/workflows/ci.yml
gh api "repos/mudler/vllm.cpp/actions/runs/<RUN_ID>/jobs?per_page=100" -q '.jobs[].name'
```

The first catches a syntax error. The second catches a duplicate key, an
unknown key and an invalid `runs-on`, which PyYAML does not. **The third is the
only one that is evidence**, because it reads the job list GitHub itself built
from the file. G4 is met when that list contains every job name the file
declares, counted, not eyeballed.

**G5 — the workflow-shape checkers stay green.** `check-release-workflow.py`
pins the release workflow byte-for-byte and `check-runner-routing-consistency.py`
reads runner labels, so both run before the push.

```sh
python3 scripts/check-release-workflow.py
python3 scripts/check-runner-routing-consistency.py
```

## Work breakdown

| Wave | Work | Gate |
|---|---|---|
| W0 | This spec, the issue index row, the decision | G0 |
| W1a | The `build-test-cpu-arm64-full` job, `schedule` and `workflow_dispatch` only, `continue-on-error: true` | G1, G4, G5 |
| W1b | The nine tokenizer targets on the existing per-pull-request aarch64 job | G4, G5 |
| W2 | Triage every W1a finding into its own issue and row, then remove `continue-on-error` and promote to every event | G2 |
| W3 | Re-measure contention and confirm or demote | G3 |

W1a and W1b are separate commits in one pull request. W2 cannot start before
W1a has produced at least three green or triaged scheduled runs.

## Evidence

Observed, on 2026-08-21, at base SHA `e2a9e035d`:

- The target counts and the CMake condition table in `## Our baseline`, derived
  from `tests/CMakeLists.txt` in this worktree.
- The job and step timings in `## The cost model, measured`, from
  `GET /repos/mudler/vllm.cpp/actions/runs/{id}/jobs` over the eight most recent
  `schedule` runs: 32485355223, 32465485947, 32448496119, 32438054699,
  32415741727, 32394151274, 32372689367, 32351192860.
- The billing evidence, from
  `GET /repos/mudler/vllm.cpp/actions/runs/32465485947/timing`.
- The CTest totals and every per-test duration, from the job log of run
  32465485947, job 96721070505.
- The combining-mark count of `tests/parity/goldens/tokenizer_qwen36/corpus.txt`,
  from a Python pass over `unicodedata.category`.
- The class-1, class-2 and class-4 target counts in option A, from regular
  expressions over each target's own translation units.

Inferred, and labelled as such where it is used:

- The 45-to-55-minute duration estimate. It assumes equal compile rates on the
  two hosted runner pools, which nothing here measures. G1 replaces it.
- The attribution of about 11 minutes of the x86-64 build to the library, the
  server and the examples. It is a division of one measured total, not a second
  measurement.

Refuted while writing this spec:

- **"A curated subset is the cheap option."** A stated selection principle
  picks 291 of 552 targets, so the subset saves about half a build and gives up
  about half the coverage.
- **"The per-pull-request cost is the test run."** The x86-64 lane spends 36.3
  minutes building and 10.1 minutes testing.
- **"`build-test-cpu` is the critical path."** `cuda-fat-build` is 100.0
  minutes median against 46.2.

## Owed

These are real gaps that this row does not close. Each needs an issue, and this
row does not open one, because opening an issue is a remote write and this
session has no recorded authority for one.

- **`examples/tokenize` does not build on the aarch64 lane.** The lane sets
  `VLLM_CPP_BUILD_EXAMPLES=OFF`, and the prompt-token divergence spec names that
  binary, compiled natively on aarch64 over the anomalous prompt bytes, as its
  discriminating test. A target list does not reach it; a flag change does.
- **The server path does not build on the aarch64 lane.** The same flag block
  sets `VLLM_CPP_SERVER=OFF`, so the ten server targets and the eleven example
  CTest entries cannot configure there, and the `/v1/completions` request-parse
  segment stays unexecuted on the shipping architecture.
- **The anomalous corpus is not in the tree.** The committed golden corpus
  carries 30 combining marks over 99 lines. A corpus that reproduces the
  measured input class is owed, and it is owed on x86-64 as well.
- **No sanitizer lane runs on aarch64.** `sanitize-cpu` is `ubuntu-latest` only,
  so the weak-memory-model class that motivates option A's first selection rule
  is executed but not instrumented.
- **No matrix owns CI infrastructure.** `GATE-CI-CONCURRENCY`,
  `GATE-DOC-CHECKPOINT-STATES`, `GATE-SQUASH-TRAILERS` and this row are all
  unplaced. Either a CI matrix exists or the `GATE-` form is recognised, and
  either one is a checker semantic change with its own row, its own red-before
  test and its own green-after evidence.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the developer wants the full suite on
  every pull request immediately. That is a defensible choice, and it is a
  capacity decision this row cannot take alone: it is the difference between
  zero and about 50 added job-minutes on each of about 55 runs a day.
- Stop and report `NEEDS_DECISION` if G1 measures the job above 123.0 minutes
  from run start. The recommendation's central cost claim is then false, and
  the choice between a permanent scheduled lane and a curated subset has to be
  re-taken against the real number.
- Stop and report `NEEDS_CONTEXT` if a non-fleet aarch64 host is intended for
  option C. This row must not read a host name out of a document.
- Stop if `ci.yml` is red on `main` for an unrelated reason. A new job must not
  land into a red lane, because its own first verdict then cannot be read.
