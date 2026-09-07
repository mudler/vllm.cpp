# GLM-5.3-Flash expert placement

- Row: `MODEL-MM-GLM53-FLASH`.
- Issue: [#3019](https://github.com/mudler/vllm.cpp/issues/3019).
- Campaign: [GLM-5.3-Flash](glm5-next-flash.md).
- Base: `e2fb2f06d9944c4bbe66531034479d370df67815`.
- Scope: a placement repair for the existing expert arm. This document precedes implementation.

## Now

The scoped placement repair is implemented after the committed design.
The production regression, focused CPU checks, and implementation mutations pass.
The campaign remains `ACTIVE`; fresh review and hardware verification remain pending.
This slice does not complete the model or establish performance parity.

## Problem and evidence

The issue records a Strix run with `VT_GLM5_NEXT_DEVICE_EXPERTS=1 VT_CPU_MOE=1`.
Job `0680eea4-195b-4df4-b4c1-19b7d04b3762` installed CPU placement for 45 layers.
The forward then announced device expert execution and reached an unsupported ROCm dtype.
The matching CPU control emitted ` Paris. Paris is` and exited 0.
The issue retains the original command, output, duration, and provenance.
These are recovered observations, not measurements from this design session.

Source inspection at the base establishes the mismatch:

| Production stage | Source and observed behavior |
|---|---|
| Registry load | `src/vllm/model_executor/models/glm5_next_registry.cpp::LoadGlm5NextForConditionalGeneration` constructs policy from the model source device |
| Bank residency | `src/vllm/model_executor/models/glm5_next_loader.cpp::LoadStackedExperts`, line 215, calls the shared policy |
| Placement resolution | `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp::GgufLoadPolicy::ComputeDeviceFor`, line 449, asks the active resolved plan |
| Bank bridge | `src/vllm/model_executor/models/glm5_next_bridge.cpp::AdmitMoeQuantBanks`, line 581, retains host views and source pointers |
| Forward dispatch | `src/vllm/model_executor/models/glm5_next_forward.cpp::Glm5NextHostForward`, line 288, passes the caller's device through the stack |
| Incorrect upload | `src/vllm/model_executor/models/glm5_next_moe.cpp::MoeExpertsKeepQuant`, line 222, selects device execution without reading placement |

The loader correctly asks CPU capability for CPU-placed banks.
The forward subsequently changes the execution device without repeating admission.
The defect therefore includes placement, not only a missing encoding check.

History is anchored by `git log -S'ComputeDeviceFor'` and `git log -S'MoeExpertsKeepQuant'`.
Placement-aware residency arrived in `2e127b4d7`.
The Flash device arm arrived in `c4df6dcee`.
The existing cache repair remains present. Do not repeat #2480.

## Reference and upstream anchors

The global vLLM pin remains `e126687a9a`.
The campaign's separate reconciliation owns merged Flash source `98ed0856f3`.
That reconciliation must settle model reference status before new numerical claims.
This repair introduces no new model arithmetic or quantization kernel.

At `98ed0856f3`, `vllm/models/glm5next/nvidia/model.py::Glm5NextMoE`
constructs `FusedMoEFactory` at lines 225 to 247.
Its forward computes router logits and invokes those experts at lines 264 to 267.
Keep the existing routed and shared expert arithmetic unchanged.

CPU compute placement is the existing local extension to this architecture.
Its shared resolver mirrors llama.cpp `b10451`, commit
`10bf611e533d81f739128304991c5e133c6aebd8`.
In `src/llama-model-loader.cpp::llama_model_loader::create_tensor`,
lines 1178 to 1200 select the first matching override with `regex_search`.
A CPU override selects a CPU buffer candidate.
Reuse `DevicePlacement`, `MoePlacementPlan`, and `GgufLoadPolicy::ComputeDeviceFor`.
Do not copy their pattern parsing, precedence, or partial-bank refusal logic.

Do not confuse CPU compute placement with vLLM parameter offload.
At the global pin, `vllm/model_executor/offloader/uva.py::_maybe_offload_to_cpu`,
lines 124 to 147 copy non-UVA parameters back to the device for execution.
That mechanism is not the contract of `VT_CPU_MOE`.

The current model numerical fixtures remain regression evidence.
No upstream test directly exercises this local placement handoff.
Adapt the existing production GGUF fixture and preserve its reference values.
The hardware obligations below remain necessary. Source inspection is not an oracle run.

## Design

### Preserve the resolved device with the loaded model

Resolve each sparse layer's routed-expert target during load through
`GgufLoadPolicy::ComputeDeviceFor` with `kStackedExpertWeight`.
Use the exact bank names already supplied to `LoadStackedExperts`.
Check agreement among gate, up, and down targets through the shared placement contract.

Store the resolved target with `Glm5NextMoeWeights`.
Carry that assignment through `BridgeMoeLayer` into the runtime expert weights.
The forward must use this captured assignment.
A later load or replacement of the process-global placement plan must not change an existing model.

Distinguish a captured target from absent metadata on manually constructed unit fixtures.
An optional device field can represent this distinction.
Production loads always fill the field, including unplaced loads.
Absent fixture metadata retains the existing caller-device behavior.
Do not use an unconditional CPU default that silently disables the existing device tests.

The source pointers retain their current ownership and lifetime.
Do not use host `OwnedTensor::View().device` as compute placement.
Those views describe current bytes, and a device-bound bank can still be host-resident before its first upload.

### Execute on that target

A captured CPU assignment selects the existing host expert arm.
It must not call `DeviceBanksFit`, allocate device expert buffers, upload expert banks, or invoke device expert GEMMs.
This is intentional placement and must not report a device-memory fallback.

A captured target matching the caller's device retains the existing device arm.
Preserve the opt-in guard, provider checks, staging floor, and fallback behavior.
A missing source or insufficient memory still selects the existing host path.
Do not broaden default device support.

Refuse a captured non-CPU target that conflicts with a supplied device context before allocating or uploading expert operands.
The error names the layer and both devices.
Keep direct host-reference calls with no device context usable.
They remain explicit host calculations, not a request to execute on another accelerator.

Consume the shared placement resolution without duplicating transfer code.
`include/vllm/model_executor/moe_placement_seam.h::RunMoePlaced`
expects device tensors and returns device buffers.
Flash already has host activations and host combine output.
Do not upload those activations merely to feed that helper and download them again.
If a reusable selection helper is necessary, extend the shared seam with this host-input shape.
Do not implement a second placement parser or a parallel copy-down/copy-up sequence.

### Admit the actual device operands

Keep `RouteGgufTensor` total: an unsupported device encoding can still select `kExpandBf16`.
Preserve `VT_CPU_REF`, disabled keep-quant, and legitimate expanded host execution.
An unconditional new load refusal would change those existing paths.

After selecting actual device keep-quant execution, validate all three source dtypes with `DeviceKeepQuantSupported`.
Perform this check before `ResidentWeight` and activation allocations.
Check the down bank independently from the gate/up pair.
Use the actual source dtype because the source supplies the uploaded bytes.
Retain the bridge's shape, repack, and gate/up dtype checks.
A refusal names the bank, dtype, execution device, and missing device arm.

CPU-placed banks must not be checked against accelerator encoding support.
An unsupported accelerator encoding whose bank intentionally executes on CPU remains valid.
A bank that takes the existing memory fallback also remains a host calculation.

The earliest correct backstop is before device staging, when actual preserved banks and execution target are known.
The issue's original universal load-refusal criterion is therefore superseded.
Load-time routing continues to prevent unsupported ordinary device keep-quant residency through expansion.
No global policy change or backend assertion change is required.

## Alternatives and risks

A universal accelerator check in the loader rejects legitimate CPU placement.
It also changes the documented expansion policy. Reject that approach.

Reading the active plan on every forward loses model ownership when another load replaces the plan.
Capture the resolved target instead.

Passing device tensors through `RunMoePlaced` would add transfers around already-host activations.
Reuse its resolver contract without manufacturing redundant device buffers.

The fake backend must report sufficient memory.
Its inherited `DeviceMemoryInfo=false` would hide the defective branch behind fallback.
Numeric agreement alone also hides dispatch defects because host and fake-device kernels can compute the same values.
Count expert uploads and grouped calls separately.

The opt-in value is cached once per process.
Run enabled and disabled cases in separate processes.
Do not remove the cache solely to make tests easier.

## Tests and gates

The fresh implementer writes and runs the smallest failing production-entry regression before product edits.
Reuse `tests/support/glm5_next_gguf_fixture.h` and the forward fixture topology.
Enter through `ModelRegistry::Load` and `ModelRegistry::Forward`.
Construct `ModelSource::FromGguf` with an explicit target device.
The existing `LoadThroughRegistry` helper uses the current platform and cannot establish this distinction.

A dedicated test executable can register a fake backend and both grouped expert operations.
Use host storage with explicit copy accounting, or extend the existing shadow-backend pattern.
Register fake operations only in that executable, so other tests retain their provider-refusal coverage.
Preserve actual activation values and identify expert copies by source bank addresses.
A fake backend that only throws cannot prove correct mixed execution.

Required cases:

1. CPU placement on a non-CPU model source executes a nonzero multi-token forward with zero accelerator expert uploads and grouped calls.
2. A mixed placement uses at least two sparse layers. Only unplaced layers upload banks and execute the fake-device grouped pair.
3. No placement preserves the existing device path when explicitly enabled. Device calls and uploads must be observable.
4. Replacing the global plan after load leaves the first model's assignment unchanged.
5. Default opt-in OFF still refuses the non-CPU forward. Exact `1` remains the enabled spelling.
6. CPU-placed banks unsupported by ROCm remain admitted and run on CPU. Test actual block geometry and data, not a retagged buffer.
7. Actual device banks with an unsupported dtype refuse before any expert upload or grouped call. Exercise down independently from gate/up.
8. Expanded banks, CPU reference, no-source fixtures, and insufficient-memory fallback retain their existing behavior.
9. A conflicting captured accelerator target refuses before upload and names both devices.
10. Partial bank placement keeps the existing shared resolver refusal and first-match precedence.

Use the current ROCm capability predicate for a real unsupported dtype case.
Q4_0 has 32-element blocks and keeps the existing tiny hidden size usable.
A fake backend must not invent a different product capability table.
A loader-admission test can query ROCm policy without allocating a ROCm buffer.
A narrow injected invalid runtime state can test the pre-upload backstop, but label it as a backstop test.
It does not replace the ordinary production loader/forward placement regression.

Use more than one prompt token and distinct hidden rows.
Check finite, nonzero logits and equality with the matching host control when fake kernels use the same arithmetic.
Run prefill and continuation to detect a target lost when the bridge rebuilds a layer.
Preserve existing numerical tolerances and fixture parameters.

Focused gate, after a clean CPU build with tests enabled:

```sh
ctest --test-dir build-glm53-placement --output-on-failure -R 'test_glm5_next_|test_gguf_keep_quant|test_device_placement|test_moe_placement'
```

Include the dedicated placement executable in that registered set.
Run its opt-in OFF and ON processes explicitly.
Compile with `-j 2` on the coordinating host.
Clean-rebuild affected targets after header changes.
Run the full CPU CTest suite and `scripts/agent-preflight.sh --fail-on-skip`.
For preflight, distinguish failures from argument-requiring checker skips.
An unchanged, proven baseline is not a green result.
Record exact commands, source SHA, binary hashes, exit status, and evidence paths.

### Fresh review mutations

Review the immutable implementation head in a scratch worktree.
Each mutation must make its focused test fail:

- Delete the production placement handoff from loader to bridge.
- Ignore the captured CPU assignment and restore unconditional caller-device staging.
- Read the active global plan instead of captured placement after another load.
- Remove capability admission for gate/up, then separately for down.
- Remove the mismatch refusal.
- Delete the production call site that reaches the new selection.
- Disable the legitimate unplaced device arm to prove the positive control detects it.

Restore the tree byte-for-byte after each mutation.
A surviving mutation is a finding, not a disclosed test limitation.
A fresh implementer repairs findings. The operator reruns the gates.

### Hardware obligations

After focused and full local gates, run on leased Strix and DGX when available.
Use the already-authorized, pinned GLM-5.3-Flash UD-Q2_K_XL artifact and the same binary for each A/B.
Record artifact hashes, source revision, device state, opt-in, placement, and full commands.

On Strix, repeat #3019's CPU control and opted-in `VT_CPU_MOE=1` leg.
Require successful generation and evidence that CPU-placed experts never reach ROCm GEMMs.
On DGX, preserve the successful opt-in device path and test mixed placement with prefill plus continuation.
Run the actual available oracle on the identical workload under the reconciled campaign contract.
If the oracle or device is unavailable, name that external blocker and leave its gate `PENDING`.
No timing becomes an accepted speed result before the campaign correctness gate.

## Owed

- #3019 owns this repair and its focused, full, mutation, and hardware evidence until landing.
- [#2410](https://github.com/mudler/vllm.cpp/issues/2410) owns full device forward under `MODEL-MM-GLM53-FLASH`.
- [#2942](https://github.com/mudler/vllm.cpp/issues/2942) retains the broader ROCm Flash campaign.
- Backend encoding work in [#2782](https://github.com/mudler/vllm.cpp/pull/2782) does not discharge placement or admission.

## Implementation evidence (2026-09-07)

The implementation starts from committed design `472f1ea8118ddaeabc4810d2356d1c6b4edfe26b`.
Its worktree is `/home/mudler/_git/vllm.cpp-glm53-placement-impl`.
The branch is `row/MODEL-MM-GLM53-FLASH-PLACEMENT-IMPL-3019`.
The original shared checkout remains untouched.
The campaign remains `ACTIVE`; hardware and fresh review remain pending.

### Harness and production boundary

`tests/vllm/models/test_glm5_next_placement.cpp` enters through the registry loader and forward.
The source explicitly targets ROCm, while the fake backend uses real CPU allocation and arithmetic.
Bank-address counters distinguish expert uploads from KV and activation transfers.
Prefill uses three tokens; continuation uses two tokens with the same model and cache topology.
The global plan changes before both calls, after model loading.

The shared `Topology` and `Step` definitions move mechanically into
`tests/support/glm5_next_forward_fixture.h`.
Their extracted bytes equal the original forward test's lines 109 to 337 at the committed design.
The equality check exits 0, and the original forward test passes unchanged.
The GGUF fixture retains Q8_0 by default.
Its optional Q4_0 arm stores valid 32-element blocks with nonzero scales and distinct packed values.
No dtype retagging substitutes for quantized bytes.

The unsupported-bank tests separately label deliberately injected invalid runtime state.
Gate/up and down enter through the registry; a stale up-view case directly exercises the runtime backstop.
That case validates the source dtype independently from a previously bridged Q8_0 host view.
It does not replace the ordinary production placement regression.

### Test-first sequence

All local logs use prefix `/tmp/glm53-placement-3019-`.
The build directory is `/dev/shm/glm53-placement-build.THK1Vs`.
Configuration uses Ninja, Release, CPU only, and ccache for both compilers:

```sh
cmake -S . -B /dev/shm/glm53-placement-build.THK1Vs -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_VULKAN=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache
```

The initial build exposed a fixture namespace error, corrected before running the regression.
That compile failure is not behavioral red evidence.
With no product edits, the initial enabled executable exits 1:

```sh
VT_GLM5_NEXT_DEVICE_EXPERTS=1 \
  /dev/shm/glm53-placement-build.THK1Vs/tests/test_glm5_next_placement
```

`red.log` records four cases, 163 assertions, and 14 failures.
CPU-placed Q8_0 and Q4_0 banks incorrectly upload and invoke device kernels.
The mixed layer also uploads three banks and executes again on continuation.
The corresponding exact test-only patch is `red.patch`.
Its SHA-256 is `930fa13ca0f891dc483c2f7d8157e85cac2aa176be6e3c1722c848c602b98be2`.
The red executable hash is `b3fd1927e3a08e7cfe893ec576ff43b22196fdf545896f4969ee7284c98130e5`.
The test source hash is `f77103f67759cbb0e118abf48f8b041b6028b0e88d7ae882b5dd78cbf1de4132`.

The first product change only captures and carries placement, selecting host execution for CPU assignments.
A clean rebuild precedes `stage1-green.log`: four cases and 139 assertions pass.
New mismatch and source-admission tests then run before either refusal guard exists:

```sh
VT_GLM5_NEXT_DEVICE_EXPERTS=1 \
  /dev/shm/glm53-placement-build.THK1Vs/tests/test_glm5_next_placement \
  '--test-case=*conflicting*,*backstop*'
```

`guards-red.log` records exit 1, two cases, 58 assertions, and 26 failures.
The guard-red executable hash is `1549bd6c4d1959aa2144e5cd2cfba0fcd460cda592a546f2188cb391dcd7dcaa`.
Its test source hash is `f9088e243c9ebe8fd2128198082eb5d368ff1b688573dda89cad499b5583f298`.
`stage1-product.patch` preserves the exact intermediate product delta.
Its SHA-256 is `7f5cdd6edf9dd8429c7d009e43b3604fe10481f8268d4c74125bb01883982710`.

The final guards reuse the existing capability function without changing its body or device sets.
The shared header now declares that already externally linked function.
`guards-green.log` records all four registered placement/forward processes passing.
`focused.log` records the specified focused CTest expression passing all 16 tests.
These are CPU mock and host regression results, not accelerator or oracle measurements.

The final registration also tests the unset default through
`cmake -E env --unset=VT_GLM5_NEXT_DEVICE_EXPERTS`.
It selects the same production CPU-placement case.
`default-green.log` records all four placement processes passing: enabled, unset, zero, and nonexact `11`.

### Implementation mutations and restored build

The implementer temporarily mutates only the isolated task worktree.
`apply_patch` performs each mutation and its inverse.
`final-product.sha256` verifies exact restoration of loader, bridge, MoE, and forward source after every mutation.
`mutations.json` preserves every replacement, command, build status, test status, and restoration result.
Each `mutation-<name>.log` contains the corresponding behavioral failure.
These checks supplement, but do not replace, the fresh review's independent scratch mutations.

| Mutation name | Observed failure |
|---|---|
| `loader-handoff` | CPU uploads, mixed-layer dispatch, and mismatch refusal fail; three cases fail |
| `bridge-handoff` | The same three production cases fail when the bridge loses metadata |
| `cpu-target` | Two valid CPU-placement cases throw a CPU/ROCm mismatch |
| `mutable-plan` | Initial checks pass; continuation records one device call instead of two, with ten failed assertions |
| `gate-up-admission` | Gate/up backstops upload unsupported sources; twelve assertions fail |
| `down-admission` | The independent down backstop uploads and executes; seven assertions fail |
| `actual-source-dtype` | Checking cached views misses the changed up source; five assertions fail |
| `device-mismatch` | The mismatched accelerator uploads and executes; seven assertions fail |
| `production-call` | Removing the call loses positive uploads and dispatch; two cases fail |
| `positive-device-arm` | Disabling device execution loses the positive control; two cases fail |
| `default-optin` | Enabling only the unset default removes the expected refusal; two assertions fail |

All eleven variants compile successfully before their tests run.
Ten doctest commands exit 1; the default-only CTest command exits 8.
The first attempt to delete the sole production call failed compilation with `-Werror=unused-function`.
That attempt is not behavioral evidence.
The successful mutation retains an address-only reference, without invoking the function.

`full-build.log` records the clean CPU build passing all 2012 steps.
`restored-build.log` records the full canonical rebuild after mutations, exit 0.
The restored placement executable SHA-256 is `07f04b15ecd6670e585b7ee7a7db58f03593454cdc6b1245ca5c0dcd0e8495c5`.
The restored original forward executable SHA-256 is `ed7af8472d3e0e3f3868ca3e97f02af2450337c80c9c106c355187b1a6468d4f`.

The same-config regeneration enables `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for the CPU ISA audit:

```sh
python3 scripts/check-cpu-isa-build.py \
  --compile-commands /dev/shm/glm53-placement-build.THK1Vs/compile_commands.json
```

`cpu-isa.log` records exit 0: portable baseline and exact x86 tiers pass.
The compilation database SHA-256 is `8d3d51a1fa78be6d96315df05768b12e902d1dd4def8e3c96f41b60868ff54eb`.

The restored focused expression passes all 17 registered tests in `focused-final.log`.
The full command runs every registered CPU-build test:

```sh
ctest --test-dir /dev/shm/glm53-placement-build.THK1Vs --output-on-failure
```

`full-ctest.log` records exit 0 in 183.87 seconds: 744 registered tests, 737 passes, seven skips, zero failures.
The skipped tests are `test_modelopt_mixed_precision_checkpoint`,
`test_minimax_music3_device_arm_real`, `test_minimax_music3_depth_arm_real`,
`test_cuda_deepseek_v4`, `test_voxtral_e2e`, `test_cuda_embedding_quant`, and `test_qwen35_paged_engine`.
Their checkpoint or GPU execution is outside this scoped CPU placement repair.
None is an executed pass or replacement for the Flash hardware obligations.

### Remaining verification

The startup preflight runs `--fail-on-skip` on the clean committed design.
It exits 1 with zero failed checks and five argument-requiring skips.
The tools suites pass, and all 25 syntax translation units pass.
Its log is `startup-preflight.log`.
The skipped checks require ARM/CPU build artifacts, CUDA gencode, Triton artifacts, or a PR range.
They are not executed passes.

The final staged command is `scripts/agent-preflight.sh --staged --fail-on-skip`.
Read the implementation commit body for its exact result and `staged-preflight.log` for its output.
The exported Python wrapper limits `check-tree-compiles.py` to `--jobs 2`.
It appends full tools-suite output to `staged-tools.log` and preserves the Python exit status.
This logging wrapper changes no checker or skip rule.
The CPU ISA artifact check above separately closes that applicable bare-sweep skip.
ARM, CUDA, and Triton artifact audits do not apply to this CPU-only change.
The explicit PR-size base/head result belongs to the post-commit handoff because it needs the committed head.

Fresh review and operator verification remain pending.
Strix, DGX, and same-workload oracle gates remain pending external leased execution by the operator.

## Scope and stop conditions

Product scope is the existing Flash GGUF loader, bridge, expert dispatch, and directly required shared selection seam.
Test scope is its fixture and registered production regression.
Do not change kernels, the global oracle pin, model arithmetic, CLI defaults, or unrelated models.
Do not claim full GPU forward, model token parity, or speed parity from this repair.

The operator owns campaign and oracle reconciliation separately.
Keep this issue's implementation and evidence together.
Close #3019 only when its repair lands with the required review and gate disposition.

Return `NEEDS_CONTEXT` if a required source, permission, or artifact is unavailable.
Return `NEEDS_DECISION` for a scope change or a model behavior choice outside this contract.
Do not replace a missing hardware gate with a mock result.
