# MUSIC3-DIT-ARM-REACH — the DiT device arm's production switch gets a gate

Row `MUSIC3-DIT-ARM-REACH`. Issue
[#1131](https://github.com/mudler/vllm.cpp/issues/1131). The MiniMax-Music3 lane
is [#672](https://github.com/mudler/vllm.cpp/issues/672) and its parent spec is
[`minimax-music3.md`](minimax-music3.md); §14 landed the device DiT and §19
landed the depth decoder's twin of this repair. §15–§21 of that file are taken
by other rows, so this row keeps its own spec rather than appending a §22 to a
file two concurrent rows are already writing (`AGENTS.md` `## Records`).

## Scope

**In.** The production selection of `Music3DenoiseDeviceArm`, its half-set
refusal, and the branch inside `Music3DenoiseChunks` that turns an engaged arm
into `DitForwardDevice` calls. A gate that enters through the production denoise
function and asserts **which arm ran**.

**Out.** `DitForwardDevice` itself, `StageMusic3DitWeights`, the intra-DiT spans,
every tolerance, and every number. Those are §14 and §21 and this row does not
touch them. No performance claim is made or changed here. Nothing in
`vocoder1d.cpp`, `vt::Conv1d`/`ConvTranspose1d`, `ForRows`/`Threadpool` or the op
provider seam is touched, because a concurrent row owns those files.

## The gap, as #1131 measured it

#1131 applied two mutations to the shipped code, built them, and ran the full
suite:

| mutation | result |
|---|---|
| `on_device = false` in `Music3DenoiseChunks` — the production call site disabled | `test_minimax_music3_acoustic` 32/283, `_speech` 9/223, `test_speech_engine` 11/38 — all GREEN, unchanged |
| `half_set() -> false` — the arm refusal disabled | the same three suites — all GREEN, unchanged |

So a change that silently stopped the DiT reaching the device, or that removed
the refusal protecting a half-staged arm, was invisible to every gate in the
tree. The run stays correct and becomes hours long, which is the failure mode a
token or golden gate is structurally unable to see: the two arms agree by design.

**Why the obvious gate cannot exist.** The engine selects on
`queue_.device.type != vt::DeviceType::kCPU`. On a CPU-only runner that
condition can never be true, and not by accident:
`src/vllm/multimodal/speech_engine.cpp::SpeechEngineDeviceType` refuses
`--speech-device 1` outright when no accelerator backend is registered, so
`queue_` is `kCPU` or the engine never constructs. A branch written at that line
is unreachable from any gate CI owns. That is why #1131 stayed open, and it is
the same structural wall `minimax-music3.md` §19.5 hit on the depth arm.

## Upstream anchors

None new. vLLM has no MiniMax-Music3 implementation
(`.agents/model-matrix.md`, BEYOND-PIN AND OUT-OF-REPO), and this row adds no
numerical behaviour: the arm it gates is `minimax-music3.md` §14's, whose upstream
anchors are `modular_pipelines/minimax_music3/denoise.py` and
`transformer_minimax_music3.py` and are unchanged. This row moves a branch out of
an engine method into a named function and writes tests.

## Design

Three moves, in the order a reader needs them.

1. **The rule leaves the engine.** `Music3SelectDitArm(queue, config, weights,
   release_host, staged)` holds the condition. It runs on **both** sides of that
   condition, so a CPU gate can drive it — with a CPU queue, and with a
   fabricated non-CPU one. This mirrors `Music3SelectDepthArm`
   (`minimax-music3.md` §19.5) exactly, which is deliberate: two arms selected by
   two differently-shaped pieces of code on the same switch is how one of them
   drifts.

   **Three outcomes, and the third is the defect.** A CPU queue stages nothing
   and returns a disengaged arm, so `Music3DenoiseChunks` keeps the host
   `DitForward` every Music3 gate was taken on. Any other device stages and
   returns an engaged arm, or `StageMusic3DitWeights` refuses **by name** because
   this build has no provider for it. A non-CPU queue quietly taking the host arm
   must never happen.

   `kCUDA` is deliberately excluded from the fabricated device list. On a CUDA
   build with no device the staging fails inside the CUDA runtime, and a call
   designed to fail latches a sticky error that the next unrelated kernel reports
   as its own. Every listed entry takes the identical branch, so the rule is
   covered and the latch is not armed.

2. **`Music3DitDeviceWeights` gains `staged()`.** Without it, "the CPU queue
   selected the host arm and nothing was staged" and "something was staged and
   the arm was dropped" are indistinguishable from the returned arm alone.
   Mirrors `Music3DepthDeviceWeights::staged()`.

3. **The gate enters through `Music3DenoiseChunks`**, the production function the
   engine calls — not through `DitForwardDevice`, which is a class gate and is
   what #1131 says already exists.

**Which arm ran, asserted from two instruments that answer different questions.**
Output equality cannot answer it. `denoise.dit_device` / `denoise.dit_host` is
the production profile bucket the engine's own `profile::Report` prints, and it
says which branch the loop **selected**. `dit.pack` lives inside
`DitForwardDevice`, so it says the device forward's **body executed** — a
mislabelled bucket cannot fake it. Both are asserted for an exact count, never
for movement: one bracket spans both classifier-free-guidance branches, so
`denoise.dit_*` counts `steps x windows` and the forward count is twice that.
The host arm's case asserts the device bucket and `dit.pack` are **absent**,
which is the control that makes the device counts mean something.

## Tests and gates

`tests/vllm/models/test_minimax_music3_acoustic.cpp`, three new cases:

| case | what it proves |
|---|---|
| the DiT arm SELECTION stages on a device queue and never on a CPU one | the rule, on both sides of the condition it is written in; `release_host=true` releases nothing on the CPU path; a null staging slot is refused |
| HALF a denoise device arm is refused by name, not ignored | the predicate on both halves, **and** that `Music3DenoiseChunks` acts on it |
| the PRODUCTION denoise loop takes the device arm, and says which ran | the reachability gate: the production entry point, an engaged arm, and an exact-count assertion of which arm executed |

The focused gate is `./build/tests/test_minimax_music3_acoustic`. Adjacent
suites run for regression: `test_minimax_music3_speech`,
`test_minimax_music3_ar`, `test_music3_profile`.

Wave 3 adds two tree gates, both already on the preflight and CI lanes:
`scripts/check-test-registration.py`, which now pins the CTest label selection,
and `tests/scripts/test_check_test_registration.py`, which carries the four
mutations that hold it.

## Risks and decisions

**D1. The fixture is a fabricated denoise, not the shipped one.** Reduced DiT
geometry (2 layers, 5 channels), a 4-frame window, 2 steps. It cannot say
anything about numbers, and it is not asked to: correctness against upstream's
own goldens is `CheckDeviceDit`'s job in the same file, at the same tolerance,
and nothing here relaxes it. What a reduced geometry **can** carry, and carries
exactly, is a call count.

**D2. The condition encoder's input and output rates are made equal in the
fixture**, so `ConditionLatentLength` is the identity and the latent length is
the frame count. This is a fixture choice for readability of the counts, not a
claim about the shipped rates, which are 24000/960 in and 44100/512 out.

**D3. The numeric host/device comparison in the reachability case is a control,
not a gate.** Two arms that agree by design cannot tell a reader which ran. It is
there to catch an arm that threw halfway and left a plausible tensor behind, and
it says so in the test.

**D4. `Music3SelectDitArm` declines a CPU queue by design**, so the reachability
case builds its engaged arm by hand on a CPU queue rather than through the
selector. Those are two different subjects and they are two different cases.

## Evidence this row owes

Every mutation reports compile status **before** any test result, the binary
`sha256`, and `git diff --stat`, because a mutation that fails to build and one
that never applied both read as a passing test. The tree is restored
byte-for-byte and the restored binary is hashed back to the baseline.

**MET on both legs.** Wave 1 hashed all four CPU binaries back
(`RESTORE VERIFIED: True`, six times). Wave 2's CUDA binary hashes back too --
`ed268392...` baseline and restored, `ce8bd1f8...` mutated -- see `### Restore`.

Filled in under `## Outcome` when taken.

## Owed

Owned by row `MUSIC3-DIT-ARM-REACH`, per `.agents/reachability.md` and
`AGENTS.md` `## Nothing lands dead`.

* ~~**The engine's one-line call to `Music3SelectDitArm` is reachable but not
  gated**~~ **CLOSED** by wave 2 below, on `thor:gpu0` under `rc` lease
  `f63f60e8-957a-4062-92f8-54e5bbb49d92`, 2026-08-23. The gate is
  `tests/parity/test_minimax_music3_device_arm_real.cpp`, and deleting the call
  reds it: `REQUIRE(staging != nullptr)`, 1 case / 0 passed / 1 failed,
  `Status: FAILURE!`. What made the residual real was never in doubt and is now
  measured: with the call gone the run took the host arm, `denoise.dit_host`
  196.786 s against `denoise.dit_device` 0.527 s, for 0.24 s of audio.
* **The selector's `release_host` pass-through on the DEVICE path is not
  gated** ([#1131](https://github.com/mudler/vllm.cpp/issues/1131), row
  `MUSIC3-DIT-ARM-REACH`). MEASURED, not assumed: mutation M4b below replaces
  `release_host` with a literal `false` inside `Music3SelectDitArm` and every
  suite stays green. The reason is the same structural one — on a CPU-only build
  the device path is either not taken or refuses at staging, so the flag never
  reaches `StageMusic3DitWeights` from the selector. What IS gated is
  `release_host`'s own semantics (`release_host EMPTIES the source, and the
  staged copy still runs`, same file) and the CPU path's non-release (the new
  selection case). The 9.7 GB peak the flag exists for is a property of the real
  checkpoint on Jetson Thor and belongs to `minimax-music3.md` §14.

## The GPU leg — closing the engine's own call site (wave 2)

Everything above ran on a CPU-only build and left one thing unreached: the
engine's two-line call to `Music3SelectDitArm` at
`src/vllm/model_executor/models/minimax_music3_speech.cpp:701`. Mutation M5
deleted it and every suite stayed green. This wave closes that, and it closes it
the only way it can be closed — by running the shipped engine on a real
accelerator against the real checkpoint, inside an `rc` lease.

### What is being gated, precisely

Not the arm, not the kernels, not the numbers. **The line that turns the arm
on.** A change that deleted it would return a 2.4B fp32 DiT to the host loops,
produce a correct song many hours late, and red nothing in the tree.

### The gate

`tests/parity/test_minimax_music3_device_arm_real.cpp`, one executable, one
`ctest` entry named `test_minimax_music3_device_arm_real`.

**It enters through `include/vllm.h`.** `vllm_speech_engine_load` with
`vllm_speech_model_params.device = 1`, then `vllm_synthesize`. That is a
production entry point by `AGENTS.md` `## Nothing lands dead`'s own list, and its
reach does not depend on any other surface agreeing with it.
`examples/minimax_music3_gen` is a thin client of exactly these two calls. The
server's `/v1/audio/speech` route is **not**, and saying so avoids a claim
nothing measured: `ApiServer::handle_audio_speech`
(`src/vllm/entrypoints/openai/api_server.cpp:612`) goes through an internal
`synthesizer_` seam rather than through the C ABI, so it is a second production
path onto the same engine and not a client of this one. Nothing in the
gate constructs `Music3SpeechEngine`, reaches into `Music3DenoiseChunks`, or
builds an arm by hand. Those are wave 1's subjects and they are already gated.

**Which arm ran is read off three instruments, and never off the audio.** The two
arms agree by design, so output equality answers nothing:

| instrument | what it proves | expected |
|---|---|---|
| `acoustic.dit_staging` | `Music3SelectDitArm` **was called and took the device branch** — the span is inside the function, after the CPU early-return | present, `calls == 1` |
| `denoise.dit_device` | the production denoise loop **selected** the device branch | present, `calls == steps x windows` |
| `dit.pack` | the device forward's **body executed** — it lives inside `DitForwardDevice`, so a mislabelled bucket cannot fake it | present, `calls == 2 x steps x windows` |
| `denoise.dit_host` | the control that makes the three above mean something | **absent** |

`windows` is not a constant the test asserts against itself: it is read from the
engine's own `denoise.windows` counter, which is the length of the chunk vector
the loop returned. `steps` is the request's. So the count assertion is arithmetic
over two independently produced quantities rather than agreement with whatever
was found.

**The three are not independent of each other on the call-site mutation, and
claiming they were would be the easy overstatement.** `denoise.dit_device` is
emitted under `on_device = device_arm.engaged()`
(`minimax_music3_speech.cpp:250,346`), and an arm is engaged only through fields
`Music3SelectDitArm` sets, so deleting the engine's call reds all three at once.
`acoustic.dit_staging` is the **most direct** of them -- its span sits inside the
selector past the CPU early return, so it answers the call site and nothing else
-- and the other two corroborate it rather than answer it independently.

They do separate on **other** defects, which is why all three are asserted.
`acoustic.dit_staging` present with `denoise.dit_device` absent is an arm staged
and then dropped by the loop, which is wave 1's M1. `dit.pack` below
`2 x steps x windows` with the label unchanged is a per-classifier-free-guidance-branch
fallback to the host forward: at `steps = 2`, `windows = 1` that reads `dit.pack`
2 against 4 while `denoise.dit_host` stays absent and the other two instruments
stay exactly correct, which is wave 1's M6 in its partial form.

The waveform is checked for being finite and non-degenerate. That is a control
against an arm that threw halfway and left a plausible buffer behind, and the
case says so; it is not a numerical gate and no tolerance here is a claim.

### What runs it, and what happens where it cannot

**It is GPU-only and checkpoint-only by nature, and it is labelled so.** The
`ctest` entry carries `LABELS "gpu;checkpoint;music3"`, so
`ctest -L gpu` selects it. `-LE gpu` is NOT the complement a reader would assume
and `docs/USAGE.md` says so rather than implying it: this is the only labelled
test in the tree, and the six other checkpoint-gated suites
(`test_minimax_music3_{ar,llm,acoustic,quant,e2e}_real`,
`test_muse_glimmer_real_weights`) carry no label at all and, unlike this one, do
not exit 77 — they print a `SKIP` line and return, so CTest reports them
**Passed**.

**And the label itself fails open, which for this row is in scope rather than
beside it.** Measured by the fresh review on CMake 3.28.3: with the label
renamed, `ctest -L gpu` prints `No tests were found!!!` and returns **0**. The
row's own GPU leg recorded `ctest_L_gpu_rc=0` as its "THE LABEL IS REAL" prong,
and that value is also what a zero-test selection returns — only the printed
`Total Tests: 1` separated them, and nothing asserted it. A row whose whole
subject is a gate that measures nothing while printing green cannot ship its own
discovery mechanism in that state.

So the selection is pinned in `scripts/check-test-registration.py`, the one
production layer here that already asks CMake and CTest what exists instead of
parsing CMake text. `REQUIRED_LABEL_SELECTIONS` maps `gpu` to the **exact** set
`{test_minimax_music3_device_arm_real}`, read from
`ctest --show-only=json-v1` over a configured tree, and the expectation is a
literal in the checker — never read back out of `tests/CMakeLists.txt`, because
a checker that reads its expectation from the file it checks is a tautology.
The diagnostic names both sides of the comparison and says that an empty
selection is the dangerous case, per `.agents/verification.md` `## Make the
instrument say what it is measuring`.

Four mutations hold it, in `LabelSelectionMutationTests` of
`tests/scripts/test_check_test_registration.py` (M44-M47 of that suite's fixed
manifest): the label renamed, the `set_tests_properties` call deleted, a second
test taking the label, and the labelled registration removed outright. The first
of those was additionally run against the **shipped** `tests/CMakeLists.txt`,
and the evidence is in `## Outcome, wave 3` below.

**Why not a checker of its own.** A second checker would need its own preflight
and CI wiring and its own mutation suite for a pin that is one dictionary entry;
`check-test-registration.py` already configures the tree once and already owns
"the registration promise became vacuous", which is exactly what a label
selecting nothing is. The CPU-only configure it already performs sees this test,
because the entry is registered unconditionally — no accelerator is needed to
read what `-L gpu` would select.

It **skips loudly and exits 77** — never 0 — when either precondition is absent:

* no accelerator, decided by calling
  `multimodal::SpeechEngineDeviceType(1, "minimax-music3")` and catching its
  refusal, which is the same resolution the engine itself performs;
* no checkpoint, from `VLLM_CPP_MUSIC3_CHECKPOINT` or
  `${CHECKPOINT_ROOT}/minimax-music3`.

`vllm_cpp_add_test` registers `SKIP_RETURN_CODE 77`, so CTest reports **Skipped**
rather than Passed. A doctest case that returned early would print
`assertions: 0` and `Status: SUCCESS!`, which is the trap this repository has hit
twice.

**A gate nobody runs is the defect being fixed, so where it runs is recorded
rather than implied.** No CI runner can execute it: CI has no accelerator and no
28.5 GB checkpoint. It runs inside an `rc` lease on a fleet device, and the
recipe is in `docs/USAGE.md` beside the other checkpoint-gated Music3 gates.

### Risks and decisions, wave 2

**D5. The intra-DiT spans are armed, and that perturbs timing.** `dit.pack` needs
`VLLM_CPP_MUSIC3_DIT_SPANS=1`, which inserts a `Backend::Synchronize` at every
bracket inside the device forward. This gate makes no timing claim, so the
perturbation costs it nothing — and it buys the one instrument that cannot be
faked by a mislabelled bucket. `minimax-music3.md` §21.3's measurement of that
perturbation is unaffected and is not re-derived here.

**D6. The request is the smallest one that still produces a window.**
`audio_duration_s = 0.24`, `num_inference_steps = 2`, `seed = 7`. A routing
assertion needs a window, not a song. The reduced request is why the mutation leg
is affordable: with the call deleted the run takes the HOST arm, and a host arm
at full duration is the thirty-hour failure this row exists to prevent.

**D7. No fabricated device, and no call designed to fail.** Wave 1 excluded
`kCUDA` from its fabricated-queue list because a refusal latches a sticky CUDA
error the next unrelated kernel reports as its own. This wave makes no such call:
every path it takes is a path a user takes.

### Evidence this wave owes

The lease id, the device, the worker's boot id, `findmnt` for the path the
checkpoint was actually read from, and `SRC_BYTES == DST_BYTES` for the staging.
Compile status before any verdict, `git diff --stat` and the applied hunk count
for the mutation, and `test cases:` / `assertions:` / `Status:` plus the `[SKIP]`
count for every suite quoted.

The acceptance criterion is one line: **with the engine's call to
`Music3SelectDitArm` deleted, this gate reds.**

## Stop conditions

Stop and report rather than widening scope if: the reachability mutation stays
green after the change (the gate is not entering through the production path and
the design is wrong, not the test); a fabricated non-CPU queue reaches a CUDA
code path on a CUDA build; the fixture's denoise cannot be made to run without
touching a file the concurrent vocoder row owns; or closing the engine's call
site would need a GPU lease.

## Outcome

Taken on a Release CPU-only build (gcc 13.3.0, `-Werror`,
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
-DVLLM_CPP_TENSTORRENT=OFF`). No GPU, no `rc` lease, no checkpoint.

**Which head the numbers are from, because two are involved and pretending
otherwise would be the easy lie.** The six-mutation sweep ran at `1edbd74a0`.
That commit was then amended to `d307c3587` to add the `docs/USAGE.md` claim its
`include/vllm/` edit obliges — `scripts/check-doc-checkpoint.py` classifies every
path under `include/vllm/` as `user_usage`, and it judges each commit
individually. The amend touched no compiled input, and that is measured rather
than argued: all four test binaries hash **identically** across it
(`test_minimax_music3_acoustic` `65415857…`, `_speech` `f4238f8a…`, `_ar`
`7f35c50e…`, `test_music3_profile` `e86c7f47…`). M1 and M5 — the acceptance
criterion and the residual — were additionally re-run at `d307c3587` and gave
the identical verdicts recorded below.

### The green gate

| suite | test cases | assertions | Status |
|---|---|---|---|
| `test_minimax_music3_acoustic` | 39 / 39 passed | 386 / 386 passed | SUCCESS! |
| `test_minimax_music3_speech` | 9 / 9 passed | 223 / 223 passed | SUCCESS! |
| `test_minimax_music3_ar` | 37 / 37 passed | 649 / 649 passed | SUCCESS! |
| `test_music3_profile` | 7 / 7 passed | 50 / 50 passed | SUCCESS! |
| `test_speech_engine` | 11 / 11 passed | 38 / 38 passed | SUCCESS! |

The three new cases are 3 cases and 41 assertions of the acoustic total (14 + 6
+ 21); with them excluded the suite is 36 cases / 345 assertions. The counts are
reported because a count is the only thing that separates a gate from a case that
returned early, and this suite already carries one of those: `the DEVICE-resident
DiT matches upstream on CUDA` runs **0 assertions** on a CPU-only build and
prints a loud SKIP. Pre-existing, unchanged, and named here so no reader takes
the 386 as CUDA coverage.

`non-CPU DiT selection: 0 staged, 5 refused by name` on this build — every
fabricated device took the refusal arm, which is the honest outcome when no
second backend is registered, and the case asserts the third outcome cannot
happen either way.

### The mutations

Every row reports `compile_rc` **before** any verdict, the applied hunk count,
`git diff --stat`, and the mutated binary `sha256`; the tree was restored
byte-for-byte after each and all four binaries hashed back to baseline
(`RESTORE VERIFIED: True`, six times).

| id | mutation | compile_rc | `test_minimax_music3_acoustic` |
|---|---|---|---|
| M1 | `on_device = false` in `Music3DenoiseChunks` — **#1131's own mutation 1, the production call site** | 0 | **RED** — 38/39 cases, 375/376 assertions, `FAILURE!`; `REQUIRE(device_bucket != nullptr)` |
| M2 | `Music3DenoiseDeviceArm::half_set() -> false` — **#1131's own mutation 2** | 0 | **RED** — 38/39 cases, 382/386 assertions, `FAILURE!`; both predicates and both loop refusals |
| M3 | `Music3SelectDitArm` returns a disengaged arm for every queue | 0 | **RED** — 38/39 cases, 380/386 assertions, `FAILURE!`; 5 devices silently took the host arm plus the outcome count |
| M4 | selector drops `release_host` (first spelling) | **1** | **NO VERDICT** — `-Werror=unused-parameter` orphaned the parameter. Recorded rather than dropped: a mutation that fails to build reads as a passing test |
| M4b | the same, in a form that compiles | 0 | GREEN — the residual `## Owed` names |
| M6 | the `denoise.dit_device` LABEL kept and `DitForwardDevice`'s body replaced by `DitForward` | 0 | **RED** — 38/39 cases, 378/379 assertions, `FAILURE!`; `REQUIRE(pack != nullptr)` |
| M5 | the ENGINE's one-line call to `Music3SelectDitArm` deleted | 0 | GREEN — the residual `## Owed` names |

M1 is the acceptance criterion of `.agents/reachability.md` `## The reachability
mutation`: deleting the production call site reds the focused gate. M6 is why the
gate reads two instruments and not one — with only the profile bucket, a forward
that had been swapped for the host one under an unchanged label would have passed.

M1's assertion total is 376 rather than 386 because `REQUIRE` aborts the case; the
count is reported as measured rather than normalised.

### What was rejected, and why

**A gate that constructs `Music3SpeechEngine` was rejected.** It would be the
direct answer to M5 and it cannot exist here: the constructor resolves a 28.5 GB
checkpoint, and `SpeechEngineDeviceType` refuses device 1 on a CPU-only build
before a queue is made. Building a synthetic checkpoint would gate the loader,
not the switch.

**Moving the staging inside `Music3DenoiseChunks` was rejected.** It would remove
the deletable engine line entirely, and it changes a public seam with an `= {}`
default that `test_minimax_music3_e2e_real.cpp` already uses, for a gain that is
a mutation artefact rather than a capability.

**An `Music3DitDeviceForwardCount()` counter was rejected in favour of the
existing spans.** #1309 already landed `Music3DepthDeviceForwardCount()` and no
production run reads it; the `dit.*` spans and the `denoise.dit_*` buckets are
instruments the engine's own `profile::Report` prints, so they answer the
reachability question without adding a symbol whose only reader is a test.

## Outcome, wave 2 — the GPU leg

Taken inside an `rc` lease on `thor:gpu0`, job
`f63f60e8-957a-4062-92f8-54e5bbb49d92`, 2026-08-23 21:54Z-22:23Z. Worker pod
`rc-worker-kk96r`, boot id `e2112cac-660b-434e-911d-33cbd29b9176`, unchanged
across the whole run. NVIDIA Thor, `compute_cap 11.0`, driver 595.78, 14
aarch64 cores, 125 GB. Toolkit installed by the job: nvcc 13.0.88
(`cuda-toolkit-13-0`). Build: Release, `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`, named targets at
`-j 8`, `configure_rc=0`, `compile_rc=0` in 247 s.

Tree under test: `bc61ce5182446475485f54504a742d9e38d1326a` on
`row/MUSIC3-DIT-ARM-REACH`, cloned inside the worker and asserted against the
requested SHA before anything was built (`got_sha` equal, `clean_tree=yes`).

**Staging, because a gate that reads 28.5 GB over CIFS measures the share.**
`/workspace` is `//192.168.68.102/Data[/rc]`, `cifs`. The checkpoint was copied
to `/tmp/m3reach-ckpt` on the worker's own overlay in 1158 s, and the copy was
verified by bytes rather than by existence:
`SRC_BYTES == DST_BYTES == 28517617303`, a hard failure otherwise. The load the
gate then paid was `load.ar_weights` 8.145 s and `load.acoustic_weights`
2.466 s, against the 780 s + 249 s this lane measured over CIFS.

### The gate, green

| suite | test cases | assertions | Status | `[SKIP]` lines |
|---|---|---|---|---|
| `test_minimax_music3_device_arm_real` | 1 / 1 passed | 22 / 22 passed | SUCCESS! | 0 |
| `test_minimax_music3_acoustic` | 40 / 40 passed | 403 / 403 passed | SUCCESS! | 0 |
| `test_minimax_music3_speech` | 9 / 9 passed | 223 / 223 passed | SUCCESS! | 0 |
| `test_minimax_music3_ar` | 37 / 37 passed | 649 / 649 passed | SUCCESS! | 0 |
| `test_music3_profile` | 7 / 7 passed | 50 / 50 passed | SUCCESS! | 0 |

CTest reported `1/1 Test #70: test_minimax_music3_device_arm_real ... Passed
17.66 sec`, and `ctest -N -L gpu` selected exactly that one test, which is the
label doing its job rather than being declared.

`test_minimax_music3_acoustic` is 403 assertions here against 345 + 41 + 9 = 395
on a CPU-only build: the `the DEVICE-resident DiT matches upstream on CUDA` case
runs 8 assertions on this box that it skips on a runner. That case is §14's and
this row does not touch it; it is named because a reader comparing the two totals
would otherwise have to guess.

### WHICH ARM RAN, from the run's own instruments

`device 1 resolves to 'cuda'` and `vllm_speech_engine_device()` returned 1, so
what follows is the granted device arm and not a request echoed back.

| bucket | calls | seconds | what it proves |
|---|---|---|---|
| `acoustic.dit_staging` | 1 | 1.219 | the ENGINE CALLED `Music3SelectDitArm` and the selector took the device branch |
| `denoise.dit_device` | 2 | 0.527 | the production denoise loop SELECTED the device branch |
| `dit.pack` | 4 | 0.000 | `DitForwardDevice`'s BODY executed |
| `denoise.dit_host` | ABSENT | -- | the control |

The counts are arithmetic over quantities the run produced: `request.steps` 2
and `denoise.windows` 1, so `denoise.dit_device` is `steps x windows = 2` and
`dit.pack` is `2 x steps x windows = 4`, one bracket spanning both
classifier-free-guidance branches. The other fifteen intra-DiT spans agree with
the same arithmetic (`dit.norm1` and the eight other per-layer spans at 144 =
36 layers x 4 forwards). The request was `audio_duration_s = 0.24`,
`num_inference_steps = 2`, `seed = 7`, resolving to `request.max_frames` 6 and
`ar.frames` 6. The waveform control: 10240 samples x 2 ch at 44100 Hz, peak
0.00397296, every value finite.

### THE ACCEPTANCE CRITERION — mutation M7

`.agents/reachability.md` `## The reachability mutation`: delete the production
call site and rerun the focused gate. The call is two lines; deleting them
outright would not compile, so the deletion keeps the declaration and
default-constructs the arm, which is the same thing the tree would contain if
the call had never been written.

```diff
     Music3DitDeviceWeights staged_dit;
-    const Music3DenoiseDeviceArm arm = Music3SelectDitArm(
-        queue_, config_.transformer, acoustic.dit, /*release_host=*/true, &staged_dit);
+    const Music3DenoiseDeviceArm arm;  // MUTATION #1131: the engine's CALL deleted
+    (void)staged_dit;
```

`mutation_applied=1` (the replace asserts its own match count), `hunk_count=1`,
`git diff --stat` 1 file changed / 2 insertions / 2 deletions, and
**`compile_rc=0` reported BEFORE any verdict** -- a mutation that fails to build
reads as a passing test, which is why the rc is printed first and why M4 of
wave 1 is recorded as NO VERDICT rather than dropped.

| id | mutation | compile_rc | `test_minimax_music3_device_arm_real` |
|---|---|---|---|
| M7 | the ENGINE'S OWN CALL to `Music3SelectDitArm`, deleted | 0 | **RED** -- 1 case / 0 passed / 1 failed, 9 assertions / 8 passed / 1 failed, `Status: FAILURE!`; CTest `***Failed 211.54 sec`. `REQUIRE(staging != nullptr)` at `:267` |

**And what the red is made of, which is the point.** With the call gone the run
did not fail, break, or produce different audio. It produced
`denoise.dit_host` **196.786 s, 93.43% of the run**, where the shipped tree
produced `denoise.dit_device` **0.527 s** in the same bucket position -- on
**0.24 seconds** of audio, from a change that alters no number anywhere. Two
different ratios live in that paragraph and they are named separately so a
reader is not left deriving one from the other: the DENOISE BUCKET ratio is
196.786 / 0.527 = **373x**, and the CTest WALL ratio for the whole gate is
211.54 s mutated against 17.66 s shipped = **12x**, because the run also loads a
28.5 GB checkpoint and synthesises either way. Neither is a performance claim
(see `### What this leg did NOT do`). That is #1131's whole argument, executed
rather than asserted: the arms agree by design, so nothing that reads the output
can see this, and until this gate existed nothing in the tree could.

M5 of wave 1 -- the same deletion, on a CPU-only build -- stayed green. It is
unchanged and remains recorded above; what changed is that a gate now exists
that the deletion reds.

### Restore

`git checkout --` on the one file, then `touch`. `git status --porcelain` empty,
`HEAD` still `bc61ce518`, rebuild `compile_rc=0`, and the gate re-run green:
1 / 1 cases, 22 / 22 assertions, `SUCCESS!`, CTest `Passed 13.84 sec`.

**The restored binary hashes back to the baseline.** Three `sha256` lines of
`tests/test_minimax_music3_device_arm_real` are in job
`f63f60e8-957a-4062-92f8-54e5bbb49d92`'s log, and they are two distinct values,
not three:

| stage | `sha256` |
|---|---|
| baseline, after the first build | `ed268392882246f8cf7ae78549086f3044be62bfb1f2269d7ae4b287bcec397c` |
| mutated, after the M7 build | `ce8bd1f894dec7c5a06bce5ff14d8bd88984b765c0a75df5233ebda1bbfa8130` |
| restored, after the restore build | `ed268392882246f8cf7ae78549086f3044be62bfb1f2269d7ae4b287bcec397c` |

So `restored == baseline`, and the CUDA link on this toolchain reproduced
byte-for-byte across a rebuild. The fresh review reproduced the same answer
independently at head `718546680` on `thor:gpu0` in a second job with a second
build directory: baseline `ba9b62cb...` == restored `ba9b62cb...`, printed as
`HASH VERDICT: RESTORED == BASELINE`.

**An earlier draft of this section said the opposite, and it is worth recording
why rather than only deleting it.** It compared the restored hash against the
MUTATED one instead of against the baseline, read the inequality as
irreproducibility, and generalised it into "this build links CUDA objects and
its link is not bit-reproducible". That sentence is deleted rather than
softened. It is the sentence a later row would have cited to waive a hash
restore on any CUDA build, and nothing measured it. The restore here is proven
by all three of the pair it claimed plus the hash: an empty
`git status --porcelain` against the exact SHA, a green re-run, and
`restored == baseline`.

### What this leg did NOT do

It did not measure speed. `denoise.dit_host` 196.786 s against
`denoise.dit_device` 0.527 s is the shape of the defect, on one geometry, with
the intra-DiT spans armed and therefore with a `Backend::Synchronize` at every
bracket inside the device forward. It is not a ratio anybody may quote:
`.agents/benchmarking.md` governs those and this run took no clock window, no
idle-host control and no A/B.

It did not touch the second `## Owed` entry, the selector's `release_host`
pass-through on the device path. That flag IS now exercised end to end -- the
engine passes `true` and `acoustic.dit_staging` fired -- but nothing in this gate
would red if it were replaced by `false`, because the gate asserts routing and
not residency. The entry stands as written.

It observed, without asserting, that `ar.depth_staging` fired once in the same
run. That is `minimax-music3.md` §19.7's twin residual for the depth arm, and it
is reachable by exactly this method. This row does not claim it.

## Outcome, wave 3 — the fresh review's findings, repaired

The fresh review at head `718546680` returned **FAIL**. It reproduced the
acceptance criterion on `thor:gpu0` in its own job and its own build directory
and confirmed the red is `REQUIRE(staging != nullptr)` rather than a timeout, so
the gate itself is sound. Every finding was in the record, and one of them was a
measured statement that is false.

### F1 — the row stated the opposite of what it measured

`### Restore` claimed the restored CUDA binary does not hash back to the
baseline, and generalised that into "this build links CUDA objects and its link
is not bit-reproducible". Both halves are wrong. Job
`f63f60e8-957a-4062-92f8-54e5bbb49d92`'s log carries three `sha256` lines for
`tests/test_minimax_music3_device_arm_real` and they hold **two** values:
`ed268392...` after the baseline build, `ce8bd1f8...` after the M7 build, and
`ed268392...` again after the restore build. The row compared **restored against
mutated** and read the inequality as irreproducibility. The review reproduced
the correct answer independently at this head — `ba9b62cb...` baseline against
`ba9b62cb...` restored, printed as `HASH VERDICT: RESTORED == BASELINE`.

The generalisation is **deleted rather than softened**, and the reason is that it
travels: it is the sentence a later row would cite to waive a hash restore on any
CUDA build, and it already reached one reviewer brief as a binding instruction.
If some translation unit here genuinely does not link reproducibly, that is a
separate measured claim for whoever measures it. The `## Evidence this row owes`
line — "the restored binary is hashed back to the baseline" — is therefore MET on
both legs rather than waived on one.

### F2 — the `gpu` label failed open, and the label is this row's own recipe

Measured by the review on CMake 3.28.3: rename the label and `ctest -L gpu`
prints `No tests were found!!!` and returns **0**. Pinned in
`scripts/check-test-registration.py` — see `### What runs it, and what happens
where it cannot` for the design and for why it lives in that checker rather than
in a new one.

**The mutation, on the shipped `tests/CMakeLists.txt` rather than on a fixture.**
`LABELS "gpu;checkpoint;music3"` renamed to `LABELS "device;checkpoint;music3"`,
`mutation_applied=1`, `git diff --stat` 1 file changed / 1 insertion / 1
deletion for that file, 1 hunk:

| stage | `scripts/check-test-registration.py` |
|---|---|
| shipped | **rc 0** — `OK: ... the configured tree matches the pinned label selection (-L gpu -> 1 [test_minimax_music3_device_arm_real]) ...` |
| label renamed | **rc 1** — `ERROR: ctest -L gpu selects 0 test(s) [<none>]; REQUIRED_LABEL_SELECTIONS ... pins 1 [test_minimax_music3_device_arm_real]` |
| restored | **rc 0**, and `git hash-object tests/CMakeLists.txt` is `9a74e857a5a8b51e23a80d23e3b73f3c3af9f0f1` before and after, with `git status --porcelain` empty for that path |

`tests/scripts/test_check_test_registration.py` gains
`LabelSelectionMutationTests`: a passing baseline plus M44 (label renamed), M45
(`set_tests_properties` deleted), M46 (a second test takes the label) and M47
(the labelled registration removed outright). M46 is why the pin is an exact set
and not a floor. The fixed mutation manifest and its production-pinned digest are
updated in the same change, because the checker refuses a suite whose inventory
has drifted from the manifest.

### F3 — `docs/USAGE.md` described a convention that exists at n = 1

The page offered `-L gpu` as "the device gates" and `-LE gpu` as "everything
else". Neither is true: this is the only labelled test in the tree, and the six
other checkpoint-gated suites carry no label and do not exit 77 — they print a
`SKIP` line, return, and CTest reports them **Passed**. The section now names the
one labelled gate, names the six unlabelled suites and their Passed-on-skip
behaviour, and tells the reader to read `Total Tests:` rather than the exit
status.

### F4 — the depth twin got its own issue before this row's `Closes` fires

#1131 named both device-arm twins. This row closes the DiT one, so
`minimax-music3.md` §19.7's un-struck depth entry would have been left pointing
at a closed issue. Filed as
[#1839](https://github.com/mudler/vllm.cpp/issues/1839), retracked in §19.7 and
appended to `.agents/issue-index.md`. `Closes #1131` stands: the DiT arm
genuinely is closed.

The depth twin is genuinely ungated, and the reason is worth carrying: §19.6's
"device path TAKEN" leg rides `test_minimax_music3_ar`, whose observable
`Music3DepthDeviceForwardCount()` is a counter §19.5 itself records as
unreachable from production. `ar.depth_staging` — the instrument that would
answer the call site — is emitted at `minimax_music3_llm.cpp:582` and read by
nothing. This row's own `thor:gpu0` run fired it once, so it is known live.

### F5 — an arithmetic non-sequitur beside two measured numbers

"a 12x wall-clock difference" sat immediately after "196.786 s ... 0.527 s",
which is 373x. The 12x is the unstated CTest wall ratio, 211.54 / 17.66 = 11.98.
No number changed; both ratios are now named with the quantity each belongs to,
and neither is a performance claim.

### F6 — "thin clients" was loose for the server

`/v1/audio/speech` goes through `ApiServer::handle_audio_speech`
(`api_server.cpp:612`) and an internal `synthesizer_` seam, not the C ABI. It is
a second production path onto the same engine, not a client of this one. The
example genuinely does drive `vllm.h`. Reachability is unaffected, because
`include/vllm.h` is independently a listed production entry point.

### The instrument-independence claim, narrowed

The review noted that `denoise.dit_device` is emitted under
`on_device = device_arm.engaged()` (`minimax_music3_speech.cpp:250,346`), so it
would red on the call-site deletion by itself. The three instruments are
therefore **not** independent on M7: `acoustic.dit_staging` is the most direct
and the other two corroborate. They do separate on other defects, and the two
that matter are now named in `### What is being gated, precisely`.

### What wave 3 did NOT do

It did not re-run the device leg. The review reproduced the acceptance criterion
on hardware at this head and no finding disputes a measured value from
`f63f60e8-957a-4062-92f8-54e5bbb49d92`; a second lease would spend a fleet
device to re-derive a number two jobs already agree on. No number in
`## Outcome, wave 2` is changed by wave 3 — the only value corrected is the
hash comparison, which was a reading of the log rather than a measurement.
