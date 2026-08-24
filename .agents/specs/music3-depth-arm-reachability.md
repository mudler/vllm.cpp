# MUSIC3-DEPTH-ARM-REACH — the depth decoder's production switch gets a gate

Row `MUSIC3-DEPTH-ARM-REACH`. Issue
[#1839](https://github.com/mudler/vllm.cpp/issues/1839). The MiniMax-Music3 lane
is [#672](https://github.com/mudler/vllm.cpp/issues/672) and its parent spec is
[`minimax-music3.md`](minimax-music3.md); §19 landed the device depth decoder and
§19.7 carries the residual this row discharges. §15–§21 of that file are taken by
other rows, so this row keeps its own spec rather than appending a §22 to a file
two concurrent rows are already writing (`AGENTS.md` `## Records`).

It is the exact twin of [`music3-dit-arm-reachability.md`](music3-dit-arm-reachability.md)
(row `MUSIC3-DIT-ARM-REACH`, #1131, PR #1821, merged `404f0cdc5`), and it is
deliberately not a re-derivation of it. That row's method is mirrored move for
move: a parity gate entering through `include/vllm.h` with `device = 1`, labelled
`gpu;checkpoint;music3`, exiting 77 without its preconditions, asserting which arm
ran from the engine's own profile buckets, closed by deleting the production call
site and showing the gate red.

## Scope

**In.** The engine's own call to `Music3SelectDepthArm`
(`src/vllm/model_executor/models/minimax_music3_speech.cpp:638`), and a gate that
enters through a production entry point and asserts that the RVQ depth decoder
ran on the accelerator. The two profile counters that make "which branch the
production append lambda selected" readable from the engine's own
`profile::Report`.

**Out.** `DepthDecoderAppendDevice`, `StageMusic3DepthWeights`, the bf16 band,
the drawn-code comparison, the residency mask, and every number. Those are §19,
they are gated there, and this row touches none of them. The DiT arm's call site
is `MUSIC3-DIT-ARM-REACH`'s and is closed. **No performance claim is made or
changed here.**

## The gap, as #1839 and §19.5 measured it

§19.5 deleted the engine's two-line call and ran the tree:

| mutation | result |
|---|---|
| the engine's call to `Music3SelectDepthArm` deleted | `test_minimax_music3_ar` 37/37 · 640/640 and `test_minimax_music3_speech` 9/9 · 223/223 — GREEN, unchanged |

So a change that silently returned the 0.646 B RVQ depth decoder to its host
reference loops was invisible to every gate in the tree. The run stays correct and
becomes much longer — §19.1 measured that stage at **48.4 %** of a run on
`thor:gpu0`, a 0.646 B model costing **6.3x** the 8.6 B language model beside it.
Two arms that agree by design cannot be separated by a token gate, a golden or a
tolerance.

**Why the obvious gate cannot exist.** The engine selects on
`queue_.device.type != vt::DeviceType::kCPU`. On a CPU-only runner that condition
can never be true, and not by accident:
`src/vllm/multimodal/speech_engine.cpp::SpeechEngineDeviceType` refuses
`--speech-device 1` outright when no accelerator backend is registered, so
`queue_` is `kCPU` or the engine never constructs.

**Why §19.6's "device path TAKEN" leg does not close it either, and this is the
half that matters.** That leg rides `test_minimax_music3_ar`, whose observable is
`Music3DepthDeviceForwardCount()` — a counter §19.5 itself records as unreachable
from production, its only readers being the tests written for it
(`tests/vllm/models/test_minimax_music3_ar.cpp:1325,1351,1583,1589,1753,1758`).
A counter no production run reads measures a class, not a capability. The one
instrument that would answer the call site is `ar.depth_staging`
(`minimax_music3_llm.cpp:582`) and **no test in the tree read it** before this
row.

## Upstream anchors

None new. vLLM has no MiniMax-Music3 implementation (`.agents/model-matrix.md`,
BEYOND-PIN AND OUT-OF-REPO), and this row adds no numerical behaviour: the arm it
gates is §19's, whose upstream anchors are
`models/transformers/minimax_music3_rvq_depth_decoder.py` and `encoders.py:117-142`
and are unchanged. This row writes a test and adds two counters.

## Design

Three moves.

1. **The gate enters through the C ABI.** `AGENTS.md` `## Nothing lands dead`
   lists what counts as a production entry point and `include/vllm.h` is the
   first item on it. `tests/parity/test_minimax_music3_depth_arm_real.cpp` calls
   `vllm_speech_engine_load` with `device = 1` and then `vllm_synthesize`. It
   constructs no engine, calls no `Music3DepthStage`, stages nothing and builds
   no arm by hand — every one of those is `test_minimax_music3_ar`'s subject and
   is gated there. The only thing that may select the device arm is the shipped
   engine executing the line under test.

   `/v1/audio/speech` is a **second** production path onto the same engine
   through `ApiServer::handle_audio_speech` and an internal `synthesizer_` seam,
   not through the C ABI; `examples/minimax_music3_gen` genuinely is a thin
   client of these two calls. Reachability does not depend on either: the C ABI
   is independently a listed entry point.

2. **`Music3DepthStage`'s append lambda says which branch it took.**
   `profile::Count("ar.depth_device", 1)` and `profile::Count("ar.depth_host", 1)`
   are the depth twin of `denoise.dit_device` / `denoise.dit_host`. They exist
   because the alternative observable — `Music3DepthDeviceForwardCount()` — is
   read by no production run, which is exactly #1839's complaint, and because
   the record that filed #1839 already describes the gate as asserting
   "`ar.depth_staging` `calls == 1` with the host bucket absent" and no host
   bucket existed. `profile::Count` writes a `seconds = -1` pure counter, so it
   joins no leaf sum and cannot make the split's parts exceed its whole; with the
   instrument off it returns on one predicted branch and reads no clock.

3. **Both counters are driven on both sides of the branch from a CPU runner.**
   `test_minimax_music3_ar.cpp`'s composed-stage case already runs
   `Music3DepthStage` twice in one process — once with a default-constructed arm
   and once with an arm staged onto a CPU `vt::Queue` — which is the only place
   in the tree where both sides of that branch execute. The assertions go there
   rather than into a fourth fixture.

**Which arm ran, and an honest statement of what is direct and what corroborates.**

| instrument | what it answers |
|---|---|
| `ar.depth_staging` | the engine CALLED `Music3SelectDepthArm` and the selector took the device branch. Emitted past the CPU early return, so present iff both. **The direct answer to #1839.** |
| `ar.depth_device` | the production append lambda SELECTED the device branch, once per appended position. Held to `ar.depth_forward`, which brackets every append on both arms. |
| `ar.depth_host` | **ABSENT.** The control that makes the two above mean something rather than merely be present. |

**They are not independent of one another on the call-site mutation, and this row
says so rather than implying otherwise.** `ar.depth_device` is emitted under
`device_arm.engaged()`, and an arm is engaged only through the fields
`Music3SelectDepthArm` sets, so deleting the engine's call reds both at once.
`ar.depth_staging` is the most direct; `ar.depth_device` corroborates. They
separate on **other** defects, which is why both are asserted: an arm that stages
and is then dropped by the loop moves the first and not the second, and a partial
fallback that serves some positions from the host moves the third.

**The body-executed question is deliberately not asked here.** The DiT twin
asserted `dit.pack` because its `denoise.dit_device` bracket and its
`DitForwardDevice` body are far apart, so a body swapped under an unchanged label
was reachable. Here `ar.depth_device` sits in the same branch as the
`DepthDecoderAppendDevice` call it labels with nothing between them, and
`test_minimax_music3_ar.cpp`'s composed-stage case already holds
`Music3DepthDeviceForwardCount()` to an exact `num_codebooks` per frame. That is
a class assertion in the place a class assertion belongs, and this row does not
move it into a capability gate.

**The counts are arithmetic over quantities the run produced**, never against
constants written in the test: `ar.frames` from the generation loop,
`ar.depth_stage` from the engine's own span, `ar.depth_forward` from the bracket
around every append. `num_codebooks` is a checkpoint property, so only
`ar.depth_forward % ar.depth_stage == 0` is asserted and the quotient is
reported.

## Tests and gates

| gate | where it runs | what it proves |
|---|---|---|
| `tests/parity/test_minimax_music3_depth_arm_real.cpp` | `thor:gpu0` inside an `rc` lease | the engine's own call site, on a real accelerator against the real checkpoint |
| `test_minimax_music3_ar`, composed-stage case | any runner | `ar.depth_device` / `ar.depth_host` are live on **both** sides of the branch |

The parity entry carries `LABELS "gpu;checkpoint;music3"` and exits **77** when
the device or the checkpoint is absent, so CTest reports **Skipped** and never
Passed. `scripts/check-test-registration.py` pins `ctest -L gpu` to the exact
two-name set, because `ctest -L` prints `No tests were found!!!` and returns **0**
over an empty selection (measured, CMake 3.28.3) — a renamed label would make the
documented recipe fail open, which is the same defect class the labelled gate
exists to close.

`tests/scripts/test_check_test_registration.py` gains `M48`: renaming the label on
**one** of two labelled entries. That case only exists at n > 1 and is the one a
floor cannot see — one gate silently leaving the lane while the other keeps it
non-empty.

Adjacent suites run for regression: `test_minimax_music3_speech`,
`test_minimax_music3_acoustic`, `test_music3_profile`.

## Risks and decisions

**D1. A short request, not a song.** `--duration 0.24 --steps 2`. This is a
routing assertion and it makes the reachability mutation affordable: with the
engine's call deleted the depth decoder returns to the host loops, and the DiT
twin measured its own host fallback at 196.8 s against 0.527 s on device for the
same 0.24 s of audio. Nothing about the duration is a claim about anything.

**D2. No performance claim.** Any wall-clock difference between the shipped run
and the mutated one is reported as what it is — the reason the gate exists — and
never as a ratio this row asserts. `.agents/benchmarking.md`'s clock rules are
not satisfied by this job and this row does not pretend they are.

**D3. The two new counters are on the request thread.** `music3_profile.h` is
single-threaded by contract; `Music3DepthStage`'s append lambda runs on the
request thread, like every other bracket in the file.

**D4. `Music3DepthDeviceForwardCount()` is not used by the new gate.** It is
`MUSIC3-DEPTH-DEVICE`'s class instrument and it stays where it is. Reading it
from a capability gate would put the row's answer on a symbol no production run
reads, which is the defect #1839 names.

## Evidence this row owes

Every mutation reports `compile_rc` **before** any test result, the applied hunk
count, `git diff --stat`, and the binary `sha256`, because a mutation that fails
to build and one that never applied both read as a passing test. The tree is
restored byte-for-byte and the restored binary is hashed **against the baseline**,
never against the mutated one — the DiT twin's first draft made exactly that
comparison and concluded from it that this toolchain's CUDA link is not
bit-reproducible. It is; that sentence is deleted there and is not repeated here.

Filled in under `## Outcome` when taken.

## Stop conditions

Stop and report rather than widening scope if: the reachability mutation stays
green after the change (the gate is not entering through the production path and
the design is wrong, not the test); the mutated run's red is a timeout rather than
a named assertion; the fleet device cannot be leased and the only alternative is
`ssh` plus a file mutex the fleet cannot see; or closing the call site would need
a change to `Music3DepthStage`'s public seam.

## Owed

Owned by row `MUSIC3-DEPTH-ARM-REACH`, per `.agents/reachability.md` and
`AGENTS.md` `## Nothing lands dead`.

Nothing. The residual this row exists to discharge is closed on hardware below.
The `release_host` pass-through on the depth selector's device path is
`MUSIC3-DEPTH-DEVICE`'s and is unchanged here; the DiT arm's equivalent residual
is `MUSIC3-DIT-ARM-REACH`'s and is likewise unchanged.

## Outcome

Two legs: a CPU-only Release build on `mudler-ubuntu-box` for the branch's own
gate, and an `rc` lease on `thor:gpu0` for the acceptance criterion.

### The CPU leg

Release, gcc, `-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
-DVLLM_CPP_TENSTORRENT=OFF`, `configure_rc=0`, `compile_rc=0`.

| suite | test cases | assertions | Status | `[SKIP]` |
|---|---|---|---|---|
| `test_minimax_music3_ar` | 37 / 37 passed | 655 / 655 passed | SUCCESS! | 0 |
| `test_minimax_music3_speech` | 9 / 9 passed | 223 / 223 passed | SUCCESS! | 0 |
| `test_music3_profile` | 7 / 7 passed | 50 / 50 passed | SUCCESS! | 0 |
| `test_minimax_music3_acoustic` | 40 / 40 passed | 395 / 395 passed | SUCCESS! | 0 |

`test_minimax_music3_ar` is 655 against `origin/main`'s 649: six new assertions,
three on each side of the branch. `test_minimax_music3_acoustic` is 395 here and
403 on `thor:gpu0` because §14's `the DEVICE-resident DiT matches upstream on
CUDA` case runs eight assertions there that it skips on a CPU-only build;
pre-existing, `MUSIC3-DIT-ARM-REACH`'s, and named so no reader has to guess.

`test_minimax_music3_depth_arm_real` exits **77** on this build, quoting
`SpeechEngineDeviceType`'s own refusal, and CTest reports it
`71 - test_minimax_music3_depth_arm_real (Skipped)`.

**Mutation C1 — the new CPU-side instrument has teeth.** Deleting
`profile::Count("ar.depth_device", 1)` from the production lambda:
`mutation_applied=1`, `hunk_count=1`, `git diff --stat` 1 file / 1 deletion,
**`compile_rc=0` reported before the verdict**, mutated binary
`0fe1524bad50e28d…` against baseline `58342939c4c72f58…`. `test_minimax_music3_ar`
went **RED** — 37 cases / 36 passed / 1 failed, 635 assertions / 634 passed /
1 failed, `Status: FAILURE!`, at
`test_minimax_music3_ar.cpp:1813: FATAL ERROR: REQUIRE( device_bucket != nullptr )`.
Restored: `git status --porcelain` empty, binary back to `58342939c4c72f58…`
(**RESTORE VERIFIED against the BASELINE: True**), suite back to 37/37 · 655/655.

**The label pin bites.** Renaming `gpu` off the depth entry alone in the shipped
`tests/CMakeLists.txt` takes `scripts/check-test-registration.py` from `rc 0` to
`rc 1`:
`ERROR: ctest -L gpu selects 1 test(s) [test_minimax_music3_device_arm_real];
REQUIRED_LABEL_SELECTIONS ... pins 2 [test_minimax_music3_depth_arm_real,
test_minimax_music3_device_arm_real]`. The restored file hashes back to
`ab66e87cff5d75355096c0ad4a7eb501c5ef5c35` (sha1).
`tests/scripts/test_check_test_registration.py` is 58 tests, `OK`, 0 skipped.

### The GPU leg — `thor:gpu0` inside an `rc` lease

2026-08-24 08:38:37Z-08:44:56Z. Worker pod `rc-worker-kk96r`, boot id
`e2112cac-660b-434e-911d-33cbd29b9176`, unchanged start to end — the same pod and
the same boot as `MUSIC3-DIT-ARM-REACH`'s job, so the box has not rebooted
between the twins. NVIDIA Thor, `compute_cap 11.0`, driver 595.78, 14 aarch64
cores, 125748 MB. Toolkit installed by the job: nvcc 13.0.88
(`cuda-toolkit-13-0`). Build: Release, `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`, six named targets at
`-j 8`, `configure_rc=0`, `compile_rc=0` in 248 s. Job script and every artifact:
`/mnt/nas_share/rc/m3depth/` (`/workspace/m3depth/` from inside the lease).

**The `rc` job UUID is UNVERIFIED.** The submitting client's header scrolled out
of a `tail` and `rc ps` lists only running and queued jobs, so the run is
identified by device, submitter string
(`claude/MUSIC3-DEPTH-ARM-REACH (#1839: ...)`), pod, boot id, the UTC window
above and `out/job.log` on the NAS. Recorded as missing rather than guessed.

Tree under test: `9649a4c12903547cbe362369c9db7689c01fb3eb` on
`row/MUSIC3-DEPTH-ARM-REACH`, cloned inside the worker and asserted against the
requested SHA before anything was built (`clone_rc=0`, `got_sha` equal,
`clean_tree=yes`).

**The first submission was REFUSED by its own disk guard and that is recorded
rather than dropped**: `/tmp` had 78 GiB free against `NEED_GB=110`, so the job
exited 95 without touching the device. ENOSPC on this box presents as unrelated
compile errors, which is what the guard exists for. The budget now depends on
whether the checkpoint has to be copied — 45 GiB when it is already staged,
110 GiB when it is not — and the refused attempt's log is kept beside the run.

**Staging.** `/workspace` is `//192.168.68.102/Data[/rc]`, `cifs`. The gate read
`/tmp/m3reach-ckpt`, on the worker's own `overlay`, which the DiT twin's job
staged on this same pod and left behind. **Reused is not trusted**: the same hard
byte comparison a fresh copy gets was applied, `SRC_BYTES == DST_BYTES ==
28517617303`, with `findmnt` printed for both the NAS source and the path the
gate actually read.

### The gate, green

| suite | test cases | assertions | Status | `[SKIP]` |
|---|---|---|---|---|
| `test_minimax_music3_depth_arm_real` | 1 / 1 passed | 23 / 23 passed | SUCCESS! | 0 |
| `test_minimax_music3_device_arm_real` (the DiT twin, control) | 1 / 1 passed | 22 / 22 passed | SUCCESS! | 0 |
| `test_minimax_music3_acoustic` | 40 / 40 passed | 403 / 403 passed | SUCCESS! | 0 |
| `test_minimax_music3_speech` | 9 / 9 passed | 223 / 223 passed | SUCCESS! | 0 |
| `test_minimax_music3_ar` | 37 / 37 passed | 655 / 655 passed | SUCCESS! | 0 |
| `test_music3_profile` | 7 / 7 passed | 50 / 50 passed | SUCCESS! | 0 |

CTest: `1/1 Test #71: test_minimax_music3_depth_arm_real ... Passed 15.49 sec`.
`ctest -N -L gpu` selected exactly two tests, `#70` and `#71`, which is the label
doing its job rather than being declared. Baseline binary
`1f8e424f611ab9973fc529798565f5b3649c9ce1ed4d50d06ebbfbc23aeac8e4`.

### WHICH ARM RAN, from the run's own instruments

`device 1 resolves to 'cuda'` and `vllm_speech_engine_device()` returned 1, so
what follows is the granted device arm and not a request echoed back.

| bucket | calls | seconds | what it proves |
|---|---|---|---|
| `ar.depth_staging` | 1 | 1.149 | the ENGINE CALLED `Music3SelectDepthArm` and the selector took the device branch |
| `ar.depth_device` | 56 | (counter) | every appended position took the device branch |
| `ar.depth_forward` | 56 | 0.327 | the denominator: every append on either arm |
| `ar.depth_host` | ABSENT | -- | the control |

The counts are arithmetic over quantities the run produced: `ar.frames` 6,
`request.max_frames` 6, `ar.depth_stage` 7 spans, and 56 / 7 = **8** appends per
stage, which is the checkpoint's `num_codebooks` — derived, never written into
the test. The request was `audio_duration_s = 0.24`, `num_inference_steps = 2`,
`seed = 7`. The waveform control: 10240 samples x 2 ch, every value finite.

### THE ACCEPTANCE CRITERION — mutation M1

`.agents/reachability.md` `## The reachability mutation`: delete the production
call site and rerun the focused gate. The call is two lines; deleting them
outright would not compile, so the deletion keeps the declaration and
default-constructs the arm, which is what the tree would contain if the call had
never been written.

```diff
     Music3DepthDeviceWeights staged_depth;
-    const Music3DepthDeviceArm depth_arm = Music3SelectDepthArm(
-        queue_, ar.depth_config, ar.depth, /*release_host=*/true, &staged_depth);
+    const Music3DepthDeviceArm depth_arm;  // MUTATION #1839: the engine's CALL deleted
+    (void)staged_depth;
```

`mutation_applied=1` (the replace asserts its own match count), `hunk_count=1`,
`git diff --stat` 1 file changed / 2 insertions / 2 deletions, and
**`compile_rc=0` reported BEFORE any verdict** — a mutation that fails to build
reads as a passing test. Mutated binary
`9296b26859e90bc5a2454e5794f2a6e86ca50cdc545a5425fe29a6b335e50bfe`, which
differs from the baseline.

**RED.** `gate_mutated_rc=8`; 1 case / 0 passed / **1 failed**; 12 assertions /
11 passed / **1 failed**; `Status: FAILURE!`; CTest
`***Failed 19.07 sec`; at

```
test_minimax_music3_depth_arm_real.cpp:295: FATAL ERROR:
  REQUIRE( staging != nullptr ) is NOT correct!
  values: REQUIRE( nullptr != nullptr )
```

**The red is a named assertion and not a clock.** Neither labelled entry carries
a `TIMEOUT` property — dumped from `ctest --show-only=json-v1` in the job:
`test_minimax_music3_depth_arm_real properties={'LABELS': [...], 'RUN_SERIAL':
True, 'SKIP_RETURN_CODE': 77, 'WORKING_DIRECTORY': ...}`, `HAS_TIMEOUT=False`,
and the same for the DiT entry. `vllm_cpp_add_test` still sets only
`SKIP_RETURN_CODE 77`. The one case-insensitive `timeout` match anywhere in the
mutated log is CTest's own `Test timeout computed to be: 10000000`, its
no-timeout sentinel; the grep was read rather than counted, because a count of 1
would otherwise have been indistinguishable from a real timeout.

**And what the red is made of is the argument itself.** The mutated run did not
fail, break or produce different audio. Its buckets:

| bucket | shipped | mutated |
|---|---|---|
| `ar.depth_staging` | 1 call, 1.149 s | **ABSENT** |
| `ar.depth_device` | 56 | **ABSENT** |
| `ar.depth_host` | ABSENT | **56** |
| `ar.depth_forward` | 56 calls, 0.327 s | 56 calls, 5.446 s |
| `ar.depth_stage` | 7 spans, 0.519 s | 7 spans, 5.591 s |
| `acoustic.dit_staging` / `denoise.dit_device` | 1 / 2 | 1 / 2, unchanged |

The DiT buckets are unchanged, which is what says the mutation is scoped to the
depth call site and did not disable the accelerator generally.

**No performance claim is made, and the ratios are named with the quantity each
belongs to** rather than left adjacent to each other. `ar.depth_forward`
5.446 / 0.327 = **16.7x** in that bucket; `ar.depth_stage` 5.591 / 0.519 =
**10.8x**; the CTest wall for the whole gate 19.07 / 15.49 = **1.23x**, because
the run loads and stages 28.5 GB either way. None of these satisfies
`.agents/benchmarking.md` and none is quoted as a speed. They are on **0.24
seconds** of audio and 6 frames, from a change that alters no number anywhere.

**And the tree stayed green.** Under the identical mutation,
`test_minimax_music3_ar` 37/37 · 655/655 `SUCCESS!` and
`test_minimax_music3_speech` 9/9 · 223/223 `SUCCESS!` — #1839's recorded
blindness, reproduced at this head, on the same binary set that reds the new
gate.

### Restore, verified against the BASELINE

`git checkout` of the one file, `git status --porcelain` empty, head back at
`9649a4c1...`, `compile_rc=0`, and
`RESTORED_BINARY_SHA256 = 1f8e424f611ab9973fc529798565f5b3649c9ce1ed4d50d06ebbfbc23aeac8e4`
against `BASELINE_BINARY_SHA256 = 1f8e424f...` — **RESTORE VERIFIED (restored ==
BASELINE): True**. The comparison is against the baseline and never against the
mutated binary; the DiT twin's first draft made the second comparison and
concluded from it that this toolchain's CUDA link is not bit-reproducible. It is,
measured here on the same box and the same toolkit. The restored gate re-ran
1/1 · 23/23 `SUCCESS!`, `Passed 14.86 sec`.

### What was rejected, and why

**Building the gate on `Music3DepthDeviceForwardCount()` was rejected.** It is
the observable §19.6's "device path TAKEN" leg already rides, and #1839's
complaint is precisely that no production run reads it. A capability gate whose
answer comes from a symbol written for a test measures the class again.

**Adding a second bucket inside `DepthDecoderAppendDevice` was rejected.**
`ar.depth_device` sits in the same branch as the call it labels with nothing
between them, so the `dit.pack` failure mode — a body swapped under an unchanged
label — has no room to occur, and the class-level body assertion already exists
in `test_minimax_music3_ar.cpp` at an exact count.

**A fourth test fixture for the two new buckets was rejected.** The composed-stage
case is the only place in the tree where `Music3DepthStage` runs on both sides of
the branch in one process, so the assertions belong there.
