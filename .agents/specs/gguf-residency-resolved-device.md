# ENG-GGUF-RESIDENCY-RESOLVED-DEVICE — the GGUF residency policy reads the engine's resolved device, not the platform probe

Issue: [#2392](https://github.com/mudler/vllm.cpp/issues/2392).
Base: `9fb40279d` (`origin/main` at the claim).

## Scope

Two seams answer the question "which device will this load run on", and they
answer it differently by construction.

The ENGINE resolves it. `ResolveModelDeviceType`
(`src/vllm/entrypoints/model_loader.cpp`) routes an explicit selection through
`LoadedEngine::ResolveExplicitDeviceType`, whose `kCPU` arm carries its own
comment: "Explicit CPU never consults the accelerator probe: even on a
CUDA-capable build/process this selects the CPU queue."

The GGUF RESIDENCY POLICY probes it. `GgufLoadPolicy::FromEnv()` and
`RouteGgufTensor` both call `vllm::platforms::CurrentPlatform().device_type()`,
and `CurrentPlatform()` answers `kCUDA` whenever the CUDA platform registered —
which `src/vllm/platforms/cuda.cpp`'s `Registrar` does on any process where
`cudaGetDeviceCount` reports a usable GPU, regardless of what the caller asked
for. `qwen4_exp_registry.cpp` probes it the same way when it hands the
architecture's load-time refusal its device argument.

So on a CUDA-capable process, `--device cpu` selects the CPU queue and gets the
CUDA residency policy. The two disagree on every GGUF model, not on one:

* `RouteGgufTensor`'s device gate (`DeviceKeepQuantSupported`,
  `DeviceQuantGatherSupported`) decides per tensor whether a block encoding
  stays resident. On `kCUDA` the gather arm is false, so a keep-quant embedding
  table expands to bf16 that a CPU load would have kept.
* `GgufLoadPolicy::FromEnv` resolves `keep_quant` from
  `GgufQuantComputeAvailable()`, `keep_f16` from `DeviceKeepF16Supported`,
  `nvfp4_fp4` from `GgufNvfp4ComputeAvailable()` and `elem_kn_repack` from a
  `kCPU` equality test — four flags, all on the probe.
* `qwen4_exp` turns the disagreement into a refusal the user cannot satisfy.
  `qwen4_exp_weights.cpp`'s PLE guard throws on any device without a block
  gather, and its remedy text reads "Load this model with `--device cpu`" — the
  thing the user just did.

This is the same defect class the device-fit refusal already fixed one level up.
`#1136` found `CheckDeviceWeightFit` "refusing a checkpoint by naming a device
nothing was going to run on", and the repair was to resolve the device once and
pass it. `model_loader.cpp` now binds `const platforms::Platform& target =
platforms::GetPlatform(ResolveModelDeviceType(...))` and hands `target` to the
fit check — and then, fifty lines further down, builds the residency policy for
the same load with `GgufLoadPolicy::FromEnv()`, which probes. The fix landed for
the bound and not for the policy the bound describes.

It is latent rather than live only because every run that exercised these paths
was a CPU-only build, where the probe also answers `kCPU`. Two CUDA waves are
enabling CUDA in the same build right now.

### Out of scope

* `DeviceQuantGatherSupported`'s own answer (wave KGATHER owns it).
* The `qwen4_exp` op kernels and their registration (wave CUOPS owns them).
* The safetensors arm. `ModelSource::device` is set by `FromGguf` and read by
  the GGUF loaders only; no safetensors path consults a residency policy.
* Non-GGUF `CurrentPlatform()` readers. This row does not sweep the tree for
  the probe; it fixes the loads whose residency it decides.

## Upstream chain

No vLLM mirror. vLLM resolves `DeviceConfig` once at config-creation time
(`vllm/config/device.py:61-66` @ `555967922`, `vllm/engine/arg_utils.py:1878`)
and every later consumer reads that resolved value; nothing in the weight load
re-probes the accelerator. This row makes our GGUF residency policy read the
resolved value the same way, and it is the polarity `ResolveExplicitDeviceType`
already documents on our side.

## Design

The resolved device travels as a value along the path the load already takes.
No new global.

1. `GgufLoadPolicy` gains a `device` field, and `FromEnv` gains a required
   `vt::DeviceType` parameter. The no-argument `FromEnv()` is REMOVED rather
   than kept as a defaulted overload: a default is what let every caller get
   the probe by saying nothing, so removing it makes the compiler the checker.
   Every one of the policy's four device-dependent flags resolves from that
   parameter.
2. `RouteGgufTensor` gains a required `vt::DeviceType` last parameter, replacing
   its internal `CurrentPlatform()` call. Same reasoning: no default.
   `GgufLoadPolicy::Route` passes its own `device`, so a policy and the tensors
   it routes can never disagree.
3. `ModelSource` gains a `device` field, required by `ModelSource::FromGguf`.
   `ModelSource` is already the per-load CONTEXT and not only the checkpoint —
   `load_queue` beside it is an engine-selected execution resource and
   `multimodal` is a borrowed engine config — so the engine's resolved device
   belongs there. This is what reaches the eight GGUF registries, which are the
   only production callers with no policy argument of their own.
4. Every GGUF registry builds `GgufLoadPolicy::FromEnv(source.device)` and
   passes it to its loader. `qwen4_exp_registry` additionally passes
   `source.device` in place of the probe for the PLE refusal.
5. `entrypoints/model_loader.cpp` passes the device it ALREADY resolved for the
   fit check to `ModelSource::FromGguf`, to the policy it builds for the fit
   check, and to the MTP head's policy.

The alternative — a process-global "resolved device" that `CurrentPlatform()`-
style readers consult — was rejected. It has no upstream analogue, it makes the
value ambient exactly where the bug was ambience, and it cannot express two
engines in one process on different devices.

## Risks and decisions

* **Removing `FromEnv()` touches 35 test call sites.** That is the cost of
  making the compiler enforce the parameter. Each becomes an explicit
  `FromEnv(vt::DeviceType::kCPU)`, which states what the test always meant on a
  CPU-only build and now says so.
* **`ModelSource::device` defaults to `kCPU` on the struct.** The safetensors
  factories do not set it and no safetensors path reads it. `FromGguf` requires
  it, so no GGUF load can reach a loader without one.
* **Behaviour change on a CUDA process with `--device cpu`.** That is the fix.
  On every other combination the resolved device equals the probe, so the load
  is byte-identical: `--device cuda` resolves `kCUDA`, `--device auto` resolves
  through `ResolveAutoDevice` which already agrees with the probe except where
  `CreateQueue()` fails (in which case the resolved answer is `kCPU` and is the
  more correct one, for #1136's reason).
* **No GPU here.** The device-aware case is expressible on a CPU-only build
  because the device is now a parameter: a test can hand the production
  registry hook a CUDA-resolved `ModelSource` and read the refusal. What it
  cannot do is prove the CUDA platform registers and `CurrentPlatform()` then
  answers `kCUDA`; that half is recorded under `## Owed`.

## Tests

`tests/vllm/models/test_qwen4_exp_gguf_weights.cpp` — the production hook:

1. RED: `ModelRegistry::Load` with a `ModelSource` whose resolved device is
   `kCUDA` must refuse with the PLE gather message. Before the registry fix
   this passes the probe's `kCPU` and the load succeeds.
2. `ModelRegistry::Load` with a `kCPU`-resolved `ModelSource` must load — the
   `--device cpu`-on-a-CUDA-box case, which is the one that is wrong today.
3. Every non-CPU resolved device refuses through the same hook.

`tests/vllm/test_gguf_keep_quant.cpp` — the route and the policy:

4. `RouteGgufTensor(..., kCUDA)` and `RouteGgufTensor(..., kCPU)` on the same
   quantized embedding table give different residencies, so the parameter is
   load-bearing.
5. `GgufLoadPolicy::FromEnv(kCUDA)` and `FromEnv(kCPU)` differ in `keep_quant`
   on a build where the quantized GEMM is CPU-only, and `Route` carries the
   policy's device.

## Gates

```sh
ctest --test-dir build -R 'test_qwen4_exp_gguf_weights|test_gguf_keep_quant' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Evidence

Measured on the dev box, CPU-only build (`cmake -S . -B build
-DCMAKE_BUILD_TYPE=Release -G Ninja`, no `VLLM_CPP_CUDA`), every rc read from
the command's own status and not from a wrapper's.

**RED — behavioural, not a compile error.** With `ModelSource::device` present
and every other call site already threaded, but `qwen4_exp_registry.cpp` still
filling the loader's `device` argument from
`platforms::CurrentPlatform().device_type()`:

```text
BUILD_RC=0
./build/tests/test_qwen4_exp_gguf_weights -tc="*resolved device*"
  test cases:  1 | 0 passed | 1 failed | 12 skipped
  assertions: 13 | 5 passed | 8 failed
  Status: FAILURE!    rc=1
  CHECK( msg.find("cuda") != npos ) is NOT correct!   logged: msg :=
  CHECK_THROWS_AS( LoadThroughRegistry(g, d) ) did NOT throw at all!  (x4)
```

The refusal message is EMPTY because the load simply succeeded: a
CUDA-resolved `ModelSource` reached the loader as `kCPU`, which is the defect.

**GREEN — the same 13 assertions, after the one-line registry change:**

```text
BUILD_RC=0
  test cases:  1 | 1 passed | 0 failed | 12 skipped
  assertions: 13 | 13 passed | 0 failed
  Status: SUCCESS!    rc=0
```

**Full suites, re-run ON THE MERGE COMMIT** (`origin/main` moved 23 commits
mid-row; these are not inherited from the pre-merge measurement), `BUILD_RC=0`:

| target | cases | assertions | rc |
|---|---|---|---|
| `test_qwen4_exp_gguf_weights` | 13 | 3087 | 0 |
| `test_gguf_keep_quant` | 45 | 6504 | 0 |
| `test_gguf_device_fit` | 21 | 155 | 0 |
| `test_qwen4_exp_gguf_load_plan` | 10 | 7462 | 0 |
| `test_glm5_next_gguf_load` | 16 | 8731 | 0 |
| `test_qwen3_5_gguf_mtp` | 4 | 18 | 0 |
| `test_weight_residency_reach` | 7 | 76 | 0 |
| `test_weight_residency_config` | 39 | 516 | 0 |

**MUTATION, taken at the merge commit** so a later commit cannot silently
disarm it. `source.device` reverted to
`platforms::CurrentPlatform().device_type()`, rebuilt (`rc 0`, so the mutation
really compiled and really ran):

```text
  test cases:   13 |   12 passed | 1 failed
  assertions: 3087 | 3079 passed | 8 failed
  Status: FAILURE!    rc=1
```

Exactly one case moves and 3079 assertions do not. That is the measurement of
why this survived: the whole pre-existing suite is blind to it. Restored
byte-for-byte, verified by `sha256sum -c` and a clean `git status`, rebuilt and
re-run to 13/3087/rc 0.

`scripts/agent-preflight.sh --staged`: `PREFLIGHT_RC=0`.

## Owed

* **The hardware half of the test.** Nothing here proves that on a real
  CUDA-capable process `CurrentPlatform()` answers `kCUDA` while
  `ResolveModelDeviceType(..., Device::kCPU)` answers `kCPU`. The test pins that
  the RESOLVED device decides; it takes the resolution itself from
  `ResolveExplicitDeviceType`, whose own suite
  (`tests/vllm/entrypoints/test_loaded_engine_dense.cpp`) pins the `kCPU` arm.
  An end-to-end `--device cpu` GGUF load on a fleet GPU box closes it.
* ~~**`quant_repack` has no device term**~~
  ([#2406](https://github.com/mudler/vllm.cpp/issues/2406)). CLOSED by wave
  DEBTFIX, `.agents/specs/debtfix-glue-rank-bound-and-repack-device.md`. The
  flag now resolves through `QuantRepackForDevice(..., dev)`, gated
  `dev == kCPU` exactly as its sibling `elem_kn_repack` always was, so every
  device-dependent decision in `GgufLoadPolicy` reads the resolved device and
  this row's headline claim is finally true without a caveat. Two things that
  row asked for are answered there rather than here: the "untestable on x86"
  objection is met by making the ISA answer a PARAMETER of the decision, which
  is this row's own move for `dev`; and the "performance trade" objection is
  settled the way `## Gates` settles it, with the cost named — an aarch64
  `--device cuda` load that reaches the CUDA-to-CPU drain loses the i8mm
  interleave. The aarch64 CUDA MEASUREMENT of that trade is still owed and is
  recorded under that spec's `## Owed`.
* **Six of eight registry hooks are ungated** (review F4). Pointing
  `laguna_registry`, `qwen3_5_moe`, `qwen3_5_dense`, `deepseek_v4_registry`,
  `glm_moe_dsa_registry` or `muse_glimmer_registry` at a wrong device is
  undetectable; only `glm5_next` and `qwen4_exp` red. The code is correct — this
  is gate strength, and it is the honest generalisation of this row's own
  "one case moves, 3079 assertions do not". `qwen4_exp` is gated because its PLE
  guard makes a wrong device observable without a GPU; the others need a
  per-architecture residency assertion that does not exist yet.
* **The rest of the tree's `CurrentPlatform()` readers.** This row fixes the
  GGUF residency path. A sweep for other load-time or config-time readers that
  should take the engine's resolved device is not done here and has no owner.
* ~~**The CUDA gather arm itself.**~~ CLOSED by
  [#2396](https://github.com/mudler/vllm.cpp/pull/2396), which landed
  `OpId::kEmbeddingQuant` and made `DeviceQuantGatherSupported` read the op
  registry. `--device cuda` on `qwen4_exp` now loads on a build that links the
  CUDA registrar, and [#2391](https://github.com/mudler/vllm.cpp/pull/2391)
  landed the four op arms behind it. This row's contribution is unchanged and
  still needed: it is what makes `--device cpu` reach the CPU policy on that
  same box, which the gather flip does not address.

## Review

A fresh review returned FINDINGS, not PASS, and reproduced this row's headline
numbers byte-for-byte before doing so. What it changed:

* **F1 — the production call site was ungated.** The row's own gate entered
  through `ModelRegistry::Load` but built its `ModelSource` by hand, so
  `FromModelDir`'s `FromGguf(gguf, gguf_device)` was exercised by nothing:
  mutating that argument to a wrong constant left 26,509 assertions across
  eight suites green. Closed by a `FromModelDir` case (section 7 of
  `test_qwen4_exp_gguf_weights.cpp`), which is possible without a GPU only
  because this architecture's PLE guard makes a wrong device observable.
* **F2 — the device was re-resolved, not carried.** Two independent
  `ResolveModelDeviceType` calls, and that function is not pure on
  `--device auto`: `ResolveAutoDevice` decides by attempting `CreateQueue()`.
  The tokenizer and the mmproj vision tower load between the two calls, so a
  post-growth `cudaStreamCreate` failure could have made one load bound as CUDA
  and routed as CPU. Now one resolution, carried.
* **F3 — a collision with #2396, which has since become a collision with
  `main`.** The new gather cases re-encoded "gather is CPU-only" as fact, in
  sites that merge cleanly and stay green on a GPU-less CI, so the red would
  have landed unseen. #2396 has now merged, and the repair is in two parts.
  Cases that only need a NON-GATHERING device use `kMETAL`, which registers no
  gather in any build. Cases that are ABOUT CUDA ask the op table, mirroring
  #2396's own pattern. Four assertions of this row's were falsified outright by
  the flip — two in section (6) and two in section (7) of
  `test_qwen4_exp_gguf_weights.cpp` — and every one of them was a hardcoded
  verdict about a capability this row does not own.

  **The merged expression was in neither branch.** #2396 asked the registry
  about `platforms::CurrentPlatform()`, which was right when one device was
  routed and it was the host's; this row routes a device parameter over three
  devices. `gather_device_capable` and `gather_kept` now both ask
  `OpRegistered(kEmbeddingQuant, kRouteDev)` — #2396's question, this row's
  device.
* **F5 — a sentence of this row's own was false**, and it named a real latent
  bug. See `## Owed` and #2406, which wave DEBTFIX has since closed.
* **F6 — `(expect_keep ? kept : expanded)++` counted the EXPECTATION**, so the
  totality table's two totals restated their own arithmetic and could never
  fire. Now counts the observed residency, and the table runs over three
  devices instead of one, which turns three compile-time constants back into
  device-derived terms.

**And a build break this row's own gates could not see.** `test_cuda_quant_dot`
took its `platforms/interface.h` include INSIDE `#ifdef VLLM_CPP_CUDA` while
its use site is outside, so `build-newest-gcc` and `build-test-cpu` went red on
CI while every focused suite here stayed green — because no target in the list
those suites name builds that TU. A focused green over a target list that omits
the broken translation unit is indistinguishable from a passing build. The gate
for this row is a FULL build, and a reported green names the targets it built.

## Stop conditions

* Stop and report if the mismatch turns out not to be reachable — if some
  caller upstream of the policy already forces the probe and the resolved
  device to agree. It does not; `ResolveExplicitDeviceType`'s `kCPU` arm is
  unconditional.
* Stop if a CUDA build is required to demonstrate the red. It is not: the
  device becomes a parameter, so both arms run on a CPU-only build.

## Now

REVIEW. Spec, implementation and tests are on `row/ENG-GGUF-RESIDENCY-RESOLVED-DEVICE`,
in that commit order. Awaiting a fresh reviewer and the operator's own gate
re-run.
