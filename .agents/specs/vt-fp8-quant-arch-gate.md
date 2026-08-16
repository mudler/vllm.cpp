# VT-FP8-QUANT-ARCH-GATE — `QuantFp8Static` is trapped in the cutlass-fp8 build gate

| | |
|---|---|
| Issue | [#960](https://github.com/mudler/vllm.cpp/issues/960) (with [#844](https://github.com/mudler/vllm.cpp/issues/844), the same defect from the fallback's end) |
| Owning row | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` ([#517](https://github.com/mudler/vllm.cpp/issues/517)), whose A2-Q1 unit ([#810](https://github.com/mudler/vllm.cpp/issues/810)) is the caller this blocks |
| Base | [`vt-fp8-shared-seam.md`](vt-fp8-shared-seam.md) ([#940](https://github.com/mudler/vllm.cpp/issues/940)) — the seam this makes reachable |
| Kind | bug — build gate / op registration |
| Branch | `row/VT-FP8-QUANT-ARCH-GATE-960` |

## What is wrong

`vt::QuantFp8Static`'s **only** CUDA registration is
`src/vt/cuda/cuda_matmul_fp8_cutlass.cu:376` @ `0e1bee42f`. `CMakeLists.txt:1668`
adds that translation unit to the `vllm` target **only when
`VT_CUTLASS_FP8_ARCHS` is non-empty**:

```cmake
set(_FP8_CUTLASS_SOURCES)
if(VT_CUTLASS_FP8_ARCHS)
  set(_FP8_CUTLASS_SOURCES src/vt/cuda/cuda_matmul_fp8_cutlass.cu)
endif()
```

The kernel body has **no cutlass dependency whatsoever** — zero `cutlass` /
`CUTLASS` tokens in `QuantFp8StaticKernelCuda` (`:353-370`). It is
`out[i] = e4m3_rne_sat(x[i] * (1/input_scale))`, a grid-stride elementwise loop
over a hardware convert intrinsic. It shared a TU with the cutlass sm120 fp8
GEMM for authorship reasons and inherited that GEMM's arch set.

On sm_110 (Thor) `VT_CUTLASS_FP8_ARCHS` is empty — `cutlass-fp8: DISABLED (no
requested arch in [110] provides it)` is that arch's **documented normal
profile**, not a misconfiguration — so `OpId::kQuantFp8Static` is not registered
for `DeviceType::kCUDA` at all. That is true of **every** CUDA arch outside the
cutlass-fp8 cell, not only Thor.

**Nothing refuses first.** The op's GEMM partner `kMatmulFp8CublasLt` **is**
registered unconditionally (`src/vt/cuda/cuda_matmul.cu:920`), and the model-layer
predicate `MatmulFp8CutlassD` keys on *that*, so the guard passes. The missing
quant then resolves through `src/vt/op_provider.cpp:501` to the portable CPU
reference tier — eligible because `CudaBackend::UnifiedMemory()` is true — which
dereferences **device** pointers on the host and takes the process down:

```
[vt reference-tier] op=QuantFp8Static device=cuda has NO native kernel; running the PORTABLE CPU fallback (correct but slow)
SIGSEGV
```

Nothing silently dequantizes, and nothing refuses either: it crashes one call
later, under a banner that says "correct but slow".

## Scope

**In scope.** Relocate the `kQuantFp8Static` CUDA registration into a translation
unit that is unconditionally compiled for CUDA, so every CUDA arch gets the
native kernel; pin that placement with a runtime test and a structural checker.

**Explicitly not in scope.**

- The kernel's arithmetic, its dtype dispatch, and every dispatch condition — all
  move byte-for-byte.
- #844's class: the reference tier still runs a host kernel over device pointers
  for any *other* op that lacks a native CUDA kernel. This change removes one
  live **instance** and does not address the class. #844 stays open, and the
  refusal-instead-of-crash repair is a separate, larger change to the tier.
- Wiring NemotronH to anything (A2-Q1, #810 / #517).
- `MatmulFp8Cutlass` itself. Its arch gate is **correct**: it is a cutlass sm120
  kernel and on an arch that cannot build it a missing registration is an honest
  refusal. The point of this row is precisely that the two ops are different in
  this respect and must stop sharing a gate.

## Upstream anchors

Pinned oracle `5559679229bc961848b121ccdeaa8fa5d79bec98`
([`upstream-sync.md`](../upstream-sync.md)).

| Ours | Upstream |
|---|---|
| `QuantFp8StaticKernel` | `csrc/quantization/w8a8/fp8/common.cuh:58-77` `scaled_fp8_conversion` (`:62` `x = val * scale`, `:68` clamp to ±448, `:71` hardware RNE convert) |
| the reciprocal formed once by the caller | `csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31` `1.0f / scale[...]` |
| per-tensor (one group over the tensor) | `csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:204-210` (`scale.numel() == 1`) |
| the method is static, `input_scale` a scalar | `vllm/model_executor/layers/quantization/modelopt.py:510-513`, `:528` |

Upstream has no analogue of the defect to mirror: vLLM builds
`static_scaled_fp8_quant` from `csrc/quantization/w8a8/fp8/common.cu`, which is
in the unconditional `VLLM_EXT_SRC` list, while its cutlass `scaled_mm` sources
are added under `CUDA_ARCHS` intersections. **The relocation restores upstream's
own partition**, it does not invent one.

## Design

A new, unconditionally compiled TU: **`src/vt/cuda/cuda_quant_fp8.cu`**, added to
the `target_sources(vllm PRIVATE ...)` list directly inside `if(VLLM_CPP_CUDA)`.
The kernel, its two helpers and its registration move verbatim; only the local
`Check()` prefix changes from `matmul_fp8_cutlass` to `quant_fp8`, which was
wrong the moment the code moved and is not on any asserted path.

**Why a new TU and not an existing one.** Three candidates were considered.

- `cuda_matmul.cu` already hosts the unconditional fp8 GEMM registrations
  (`kMatmulFp8CublasLt`, `:920`), which is the strongest argument for it — the
  quant's partner already lives there. Against: it is the cuBLAS/cuBLASLt GEMM
  wrapper TU, and an elementwise activation quant is not a GEMM.
- `cuda_ops.cu` is unconditional, already includes `<cuda_fp8.h>`, and hosts
  `RmsNormQuantFp8` — literally the *fused* arm of this same math, whose
  `RmsNormF32ToFp8Dev` is deliberately the identical convert and whose
  bit-identity claim to `RmsNorm(bf16) + QuantFp8Static` depends on it. Against:
  it is a 3.6k-line general-kernel TU.
- **A file named for the op.** Chosen. The defect *is* "this kernel's compilation
  is governed by a feature it does not use", and both alternatives re-create a
  weaker form of it — the kernel's build would again be coupled to an unrelated
  file's requirements and includes. A dedicated TU makes the invariant readable
  in the CMake diff (the file is in the unconditional list, full stop), is what
  the structural checker can assert without inference, is cheap to compile, and
  is the natural home for the fp8 activation-quant family as it grows. It also
  mirrors AGENTS.md §"Shared seams": new capability arrives as **additive files**.

The cross-references between the three fp8 sites are written into all three
files, so the relation survives the split.

## Risks

| Risk | Handling |
|---|---|
| A behaviour change on GB10, where the op was already registered | Before/after on `dgx.casa` at `121a` with `cutlass-fp8: ENABLED` asserted in the configure log; four fp8 suites, identical case/assertion counts required |
| A duplicate registration masking the move | `RegisterOpProvider` takes first-registration-wins, so a stray second copy would be invisible at run time. Clause (d) of the checker forbids any second `kCUDA` registration of the op |
| Someone "fixes" it back by wrapping the registration in `#ifdef VT_CUTLASS_FP8` | Clause (c): the registration must sit at preprocessor-conditional depth 0. That mutation is a test case |
| The per-source gencode assignment | The new TU is not in `_VT_CUDA_FEATURE_SOURCES`, so `CMakeLists.txt:2231-2236` gives it the full `${VLLM_CPP_CUDA_ARCHITECTURES}` list — which is the point |
| A green checker that parsed nothing | The checker fails if the unconditional source list comes back empty, and `test_empty_source_list_is_not_a_pass` pins that |

## Tests and gates

**G4 (runtime pin), `tests/vt/test_ops_fp8_cpu.cpp`.** `OpRegistered(kQuantFp8Static,
kCUDA)` on any CUDA **build** — it needs no CUDA device, because the registration
is a table fill that runs before `main`, and "which build" is exactly the axis the
defect lived on. Its second assertion requires `kMatmulFp8Cutlass` to track
`VT_CUTLASS_FP8` instead, so the case proves the two are now **independent**
rather than merely that one of them is present: on Thor it reads
`CHECK(true) / CHECK_FALSE(false)`, on GB10 `CHECK(true) / CHECK(true)`.

**G2 (byte gate), same file, pre-existing.** CPU vs CUDA `QuantFp8Static`, byte
for byte, zero tolerance, five scales. It could not be *run* on a non-cutlass-fp8
CUDA arch before this change — it crashed. Landing this closes the arm that
[`vt-fp8-w8a8-cpu-arm.md`](vt-fp8-w8a8-cpu-arm.md) recorded as owed on that arch.

**`scripts/check-cuda-op-arch-gate.py` (structural pin)** + its suite
`tests/scripts/test_check_cuda_op_arch_gate.py`.

*Why both, argued rather than assumed.* G4 is the stronger statement: it observes
the property that matters — the op resolves for CUDA — instead of a proxy for it.
But it can only speak on a host that BUILT the CUDA backend **without**
cutlass-fp8, and **no CI job produces that build**: the GB10 gate host resolves
`cutlass-fp8: ENABLED`, where the defect is unreachable by construction, and every
other job is CPU-only. G4 would not have caught #960 before it landed; it caught
it here only because a human carried the binary to Thor. The checker reads the
build description, runs in the ordinary checker lane on every host including
CPU-only CI, and fails at PR time on the machine of whoever moves the
registration back. Neither instrument subsumes the other: the runtime test is the
claim, the checker is the tripwire.

The checker asserts four clauses per entry, with no inference about what a kernel
"needs" — HOME (the TU is in the unconditional CUDA source list), REGISTERED
(exactly one live registration in it), UNGUARDED (at preprocessor depth 0),
EXCLUSIVE (no other CUDA source registers the same op for `kCUDA`). It runs the
C++ side through `checker_text.normalize_source`, so a commented-out or `#if 0`-ed
registration reads as absent, which is what it is.

## Stop conditions

- Stop and report `NEEDS_DECISION` if relocating is not behaviour-preserving on
  some arch — i.e. if any GB10 suite differs before/after.
- Stop if the registration cannot be made unconditional for a demonstrable
  reason. It can: the kernel compiles for sm_110 with zero warnings.
- Do not extend the checker's `REQUIRED` table beyond ops whose kernels are
  genuinely arch-independent. For a cutlass/Marlin/FA2 kernel the feature gate is
  the correct behaviour.

## Evidence

Base SHA `0e1bee42f16b5f3fb3ae5a23869f6fd97bfc037d`.

### Thor (sm_110), CUDA 13.0.88, `-DVLLM_CPP_CUDA_ARCHITECTURES=110`, no cutlass

Configure on all three builds: `CUDA target architectures: 110`,
`CUDA feature cutlass-fp8: DISABLED (no requested arch in [110] provides it)`,
`CUTLASS not found`. `BUILD_EXIT=0`, `warnings: 0`, `enospc: 0` each time.
Disk 319 G free before and after.

| Tree | binary sha256 | `test_ops_fp8_cpu` |
|---|---|---|
| base `0e1bee42f` | `6b4d4df071a6…` | `test cases: 2 \| 1 passed \| 1 failed \| 2 skipped` · `assertions: 43 \| 43 passed \| 0 failed` · `Status: FAILURE!` · **exit 139 (SIGSEGV)** |
| base + G4 only (RED-first) | `63b7940e8609…` | G4 isolated: `test cases: 1 \| 0 passed \| 1 failed \| 4 skipped` · `assertions: 2 \| 1 passed \| 1 failed` · `Status: FAILURE!` · exit 1 |
| base + G4 + fix | `690bf71448ea…` | `test cases: 5 \| 5 passed \| 0 failed \| 0 skipped` · `assertions: 62 \| 62 passed \| 0 failed` · `Status: SUCCESS!` · exit 0 |

The base run reproduces #960 verbatim, including the trap it names:
`assertions: 43 | 43 passed | 0 failed` printed beside `Status: FAILURE!`, so
anything grepping the assertions line alone reads a crash as green.

The RED-first run is the important one. G4's **first** assertion failed
(`CHECK( vt::OpRegistered(vt::OpId::kQuantFp8Static, DeviceType::kCUDA) )` →
`values: CHECK( false )`) while its **second** passed
(`CHECK_FALSE(...kMatmulFp8Cutlass...)` → `CHECK_FALSE( false )`), so the case was
not vacuous and the arch genuinely lacks the cutlass GEMM. Non-zero case count in
both directions, and the case name carries no comma (`-tc` splits on commas: a
comma would have selected nothing and reported `SUCCESS!` with exit 0).

The green run is where **G2** — CPU vs CUDA, byte for byte — executes on sm_110
for the first time and passes: `bad == 0` at every one of five scales over 4096
elements each, and no `[vt reference-tier]` banner is printed at all. 62 = the 60
assertions this suite reports on GB10, plus G4's 2.

### GB10 (sm_121a), `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`, CUTLASS `$HOME/cutlass`

<!-- BEFORE/AFTER TABLE: filled from the two runs below. -->

### Local (CPU-only)

`check-cuda-op-arch-gate --report`, its 14-case suite, `check-device-leakage`
(DSR 32 == baseline 32), and the full preflight.

## Outcome

<!-- Filled at DONE. -->

## Now

Landed on `main` via the row branch. `QuantFp8Static` is registered for CUDA on
every arch; the FP8 W8A8 arm is unblocked on non-cutlass-fp8 CUDA archs, which is
the base #810/#517 A2-Q1 needs. #844's class remains open.
