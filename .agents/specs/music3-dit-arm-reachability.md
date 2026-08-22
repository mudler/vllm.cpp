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

Filled in under `## Outcome` when taken.

## Owed

Owned by row `MUSIC3-DIT-ARM-REACH`, per `.agents/reachability.md` and
`AGENTS.md` `## Nothing lands dead`.

* **The engine's one-line call to `Music3SelectDitArm` is reachable but not
  gated** ([#1131](https://github.com/mudler/vllm.cpp/issues/1131), row
  `MUSIC3-DIT-ARM-REACH`). `--speech-device 1` reaches it and no CI gate can:
  the engine needs the 28.5 GB checkpoint and a real accelerator, and
  `SpeechEngineDeviceType` refuses device 1 on a CPU-only build before a queue
  exists. The **rule** it calls is now gated on both sides of its condition and
  both of #1131's own mutations now red; what is owed is the call itself. This is
  the identical residual `minimax-music3.md` §19.7 carries for the depth arm, and
  it closes with a `thor:gpu0` leg under an `rc` lease, not on a CPU runner.
  **#1131 stays open for it.**
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
