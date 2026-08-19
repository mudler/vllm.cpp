# NemotronH device MoE: the router pair is device-resident, not re-uploaded per token

**Issue:** [#1312](https://github.com/mudler/vllm.cpp/issues/1312).
**Owning row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)), whose A2-Q2a slice
built the device MoE arm and whose A2-P slice made it reachable from
`ForwardNemotronHForCausalLM`.
**Base:** `origin/main` @ `edbc47ce0db9ce0893ccc7220bd290caed9d8a4b`.
**Pinned oracle:** vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` (0.26.0.dev0),
per [`upstream-sync.md`](../upstream-sync.md). vLLM holds the NemotronH router
as an `nn.Parameter` (`vllm/model_executor/layers/fused_moe/layer.py`, the
`gate` linear built in `nemotron_h.py`), so there is no per-step transfer
upstream to mirror. This row removes a local deviation; it does not add a
behaviour.

---

## 1. Scope

`NemotronHMoeBlockDevice` in
`src/vllm/model_executor/models/nemotron_h_device.cpp`, and the two helpers it
calls: `UploadOwned` and `DenseMarlinE1`.

Explicitly out of scope, because two other agents hold the same file:
`NemotronHMamba2MixerDevice` ([#1311](https://github.com/mudler/vllm.cpp/issues/1311)),
and anything on the unmerged
[#1289](https://github.com/mudler/vllm.cpp/pull/1289) branch. This row bases on
`origin/main`.

## 2. The defect

`NemotronHMoeBlockDevice` called `UploadOwned` for `mixer.gate.weight` and for
`mixer.gate.e_score_correction_bias` on every forward. `UploadOwned` ends in
`d.b.Synchronize(d.q)`.

`Synchronize` is not a formality on this backend. `src/vt/cuda/cuda_backend.cu`
implements it as `cudaStreamSynchronize(AsStream(q))`, which blocks the calling
host thread until every operation previously submitted to that stream has
completed. It is a real pipeline drain. The refutation the issue asked for
therefore fails: the 46 drains per decode token are serialisation, not only
bandwidth. Only the Vulkan backend makes `Synchronize` a batch flush
(`src/vt/vulkan/vulkan_backend.cpp:110`), and the CPU base implementation is
the no-op; neither runs this arm.

The released checkpoint has 23 MoE layers, so a decode token paid 46 host->device
copies and 46 full stream drains for weights that do not change after load.

## 3. The fix, and why it is the existing seam

The tree already states this idiom three times: `dense_attn::ResidentWeight`
(`include/vllm/model_executor/models/dense_attn_block.h:177`),
`::ResidentWeightF32` (`:208`), and `ResidentIn<NemotronHMoeMarlinResident>` in
this very function. `AGENTS.md` "Shared seams" binds a capability to the
existing seam unless the seam cannot represent it. It can.

`NemotronHMoeMarlinResident` gains two slots, `gate` and `bias`, allocated and
filled on first use exactly as the Marlin arena already is, and the call site
reads them through `ResidentOwned`.

**The refusal keeps its polarity.** `UploadOwned`'s four `VT_CHECK`s move into
`RequireOwned`, which `ResidentOwned` runs on **every** call, before the
residency test. A mis-shaped or absent router weight therefore still refuses by
name on every forward, not only the first, which is what keeps the existing
`CHECK_THROWS` case honest and keeps the device arm's refusal identical to the
host arm's `RequireWeight` (`nemotron_h.cpp:180`).

**One drain survives, once per layer per process.** The first-use copy keeps its
`Synchronize`. It is not load-bearing for lifetime -- `w.bytes` is the model's
own storage and outlives every forward -- but removing it is a second change
with its own argument, and one drain per layer at load is not a thing worth
reasoning about. What this row removes is the *per-token* drain.

## 4. The secondary defect, and what happens to it

`DenseMarlinE1` built `std::vector<float> ones(M)` and handed it to a `DBuf`
constructor, which issues an **async** `cudaMemcpyAsync` and does not wait
(`dense_device_glue.h:126`). The vector then died at the end of the function
while the copy could still be in flight: the exact use-after-free shape
`UploadAs` documents at `:183-189`, unsynchronised.

It is fixed here, and by removal rather than by adding a drain. `topk_weights`
is dereferenced by the kernel **only** under `if (mul_topk_weights)`
(`src/vt/cuda/marlin/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h:518`
loads it, `:1879` reads it, `:1797` and `:1880-1884` are the other two branches
on the same flag), and this call site passes `mul_topk_weights = false`. The op
requires a correctly shaped f32 tensor and reads nothing from it. So the buffer
is allocated and left unwritten, and both the hazard and 2 host->device copies
per MoE layer per token go away with it.

## 5. Owed, not fixed here

`DenseMarlinE1` still routes a **dense** GEMV through the grouped-MoE alignment
machinery: a `Memset` of the expert-id vector plus a `MarlinMoeAlignBlockSize`
launch, twice per MoE layer per token, to describe a single expert that every
row routes to. `dense_nvfp4_gemm.h:38-43` records the E=1 grouped GEMM as the
documented dense route, so this is a deliberate structure and not an accident;
whether the dense Marlin entry point is cheaper at M=1 is a measurement nobody
has taken. Filed as [#1341](https://github.com/mudler/vllm.cpp/issues/1341) and
owned by this spec.

## 6. Tests

**Red first, and the red is a count.** A new case in
`tests/vllm/models/test_nemotron_h_moe_device.cpp` installs a delegating
`vt::Backend` over the real CUDA backend that counts `Copy` calls **by source
host pointer** and `Synchronize` calls, then runs
`NemotronHMoeBlockDeviceHostIO` three times over one `NemotronHMoeWeights`.
Counting by source pointer rather than by byte count removes any chance that an
unrelated transfer of the same size is mistaken for the router's.

The case asserts, and prints, per call:

- forward 1 uploads the gate once and the bias once;
- forwards 2 and 3 upload **neither**;
- the steady-state `Synchronize` count is 2, and each of the two is named:
  `UploadAs` of the activation and `DownloadF32` of the result, both artefacts
  of the host-in/host-out test seam. Before this change it was 4.

The case count is asserted non-zero and the per-forward counts are asserted
individually, so a run that measured nothing cannot report a pass.

**What this instrument does not reach.** The synthetic NVFP4 fixture is the only
operand that reaches the device MoE arm without a 21 GiB checkpoint, and its
entry point is `NemotronHMoeBlockDeviceHostIO`, a seam. Production reachability
of the changed lines is carried by the A3 end-to-end gate, which enters through
`include/vllm.h` alone and routes `ForwardNemotronHForCausalLM` ->
`NemotronHPagedForward` -> `NemotronHMoeBlockDevice`
(`nemotron_h_device.cpp:1627`). Both are required. Neither is sufficient.

## 7. Gates

1. `test_nemotron_h_moe_device` on GB10: the counting case red before, green
   after, and the existing equivalence and refusal cases still green.
2. The A3 end-to-end token gate on the released
   `nemotron-3.5-lightning-30b-nvfp4` at revision `29f2d174`, through
   `examples/nemotron_h_gen`: `TOKEN MATCH: 96/96`, `mode=decode`,
   `STRICT PASS`. Correctness first: a speed result is not read until this one
   is green.
3. The local host gate, `scripts/agent-preflight.sh` and the CPU test suite.

## 8. Risks

- **R1: the refusal weakens to first-call-only.** Closed by construction:
  `RequireOwned` runs before the residency test, on every call.
- **R2: removing a drain exposes an ordering assumption.** Every operation in
  this block runs on the one queue `d.q`, and this file opens no aux stream, so
  stream order carries what the drain used to. The remaining host reads
  (`DownloadF32`, `:875`, `:1259`, `:1570`) keep their own syncs.
- **R3: the resident allocation leaks per weights object.** It is process
  lifetime, exactly as the 16.5 GiB Marlin arena in the same struct already is,
  and 1.34 MiB per layer against that is not a new class of cost.
- **R4: an unwritten `topk_weights` buffer becomes live if someone flips
  `mul_topk_weights`.** The call site passes the literal `false` and the comment
  cites the kernel lines, so the flip has to walk past the reason.

## 9. Records this change owes

No lifecycle state changes, so no `STATUS`/`BENCHMARKS`/`NOW` write is owed. The
issue index gains its two rows. `docs/` is untouched: no command, key, feature
or headline changes.

## 10. Now

**State at this commit: LANDED and GATED.** Implementation and gates in one pull
request, per the AGENTS.md default (no
recorded preference for this row and no split case applies).

---

## 11. Outcome

**Measured 2026-08-19 on `dgx:gpu0` (GB10, sm_121a), one `rc run` lease, job
`75f0121b-ed8a-42cc-a639-e00bfefafb9c`, one build tree for both arms.** Full
evidence in [PR #1348](https://github.com/mudler/vllm.cpp/pull/1348).

**The question this row had to answer first.** `Synchronize` is a REAL drain:
`src/vt/cuda/cuda_backend.cu:110-112` is `cudaStreamSynchronize(AsStream(q))`.
The issue's refutation clause fails, so #1312 was serialisation as well as
bandwidth and the resident fix is the right size.

**What was measured.** Per MoE block per forward, steady state: gate uploads
1 -> 0, bias uploads 1 -> 0, host drains charged to the router 2 -> 0. The two
that remain (2 -> 2) are the test seam's own `UploadAs` and `DownloadF32`, which
`NemotronHPagedForward` does not have. At 23 MoE layers that is 46 copies and 46
drains per decode token, to zero. The answer did not move: worst relative
deviation 0 over 128 elements, twice, with the element count asserted against
the geometry.

RED `9d55ef92f`: 3 cases / 2 passed / 1 failed, 47 assertions / 42 passed / 5
failed, exit 1. GREEN `e551bfa17`: 3 / 3 passed, 47 / 47, exit 0. The mutation
that defeats the residency test builds clean (`BUILD_MUT_RC=0`, zero `error`
lines) and turns the case red again at the same five assertions, so the case
detects the defect rather than passing beside it.

**A3, at GREEN, on the released checkpoint at revision `29f2d174`:**
`TOKEN MATCH: 96/96 over 3 prompt(s) (full rows=3, short rows=0, mode=decode)`,
`STRICT PASS`, exit 0, through `include/vllm.h` alone. Engine load 254.5 s;
per-prompt wall 335.87 / 323.42 / 322.71 s. **This is the first green sm_121a
A3 leg** -- [`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md)
section 10 records the host leg green and the sm_121a leg as the one that
remained. Reconciling that record is A2-P's, not this row's, so it is reported
rather than edited here.

**REJECTED: a speed claim.** No A/B was run and none is offered. The numbers
settle the opposite point, and it is the useful one: 32 tokens in 335.87 s is
10.50 s per token, and 46 drains at the tree's own ~66 us figure
(`deepseek_v4.cpp:203`, a DeepSeek-V4 comment rather than a measurement on this
arm) is of order 3 ms, about 0.03% of the step. #1312 is worth fixing because it
is free and because an unsynchronised pageable upload is a correctness hazard,
NOT because it was the decode gap. It never could have been.

**This is not a ceiling.** The next traceable step is the one the A3 walls point
at: the 23 Mamba2 blocks still compute their FP8 projections on the HOST every
decode token (#940, #1289, #1311). A 10.50 s token has to be attributed there
before anything else on this model is worth measuring.

**Why each default has its value.**

- *The first-use copy keeps its `Synchronize`.* One drain per layer per process
  is not what this row is about, and `w.bytes` outliving the copy makes removing
  it a separate argument with its own evidence.
- *`RequireOwned` runs before the residency test, not inside it.* Folding the
  refusal into the copy would refuse a wrong weight once and then accept it
  silently for the rest of the process.
- *The slots are raw `void*` on the resident struct, not `shared_ptr` with a
  deleter.* That matches the eleven fields already on
  `NemotronHMoeMarlinResident` and the identical `MoeMarlinResident` in
  `qwen3_5.cpp:861`. Neither frees, so both leak their arena at teardown; that is
  the tree-wide shape for a process-lifetime Marlin arena and changing it here
  would be a one-field deviation inside a struct that does the opposite. Noted,
  not filed, because no measurement establishes it as a defect rather than a
  deliberate lifetime.
- *`topk_weights` is allocated and left unwritten rather than zeroed.* A device
  `Memset` would be 46 more launches per token to initialise a buffer the kernel
  provably does not read, which is the cost this row exists to remove.
