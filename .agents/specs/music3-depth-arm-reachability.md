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
`AGENTS.md` `## Nothing lands dead`. Filled in under `## Outcome` when taken.

## Outcome

Filled in when the leased run completes.
