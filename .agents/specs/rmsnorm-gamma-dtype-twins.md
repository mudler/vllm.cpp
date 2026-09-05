# The two RmsNorm twins that still weld the gamma dtype, and the kF16 divergence

**Row:** `MODEL-MM-QWEN4-EXP` (wave RMSTWINS)
**Issues:** [#2492](https://github.com/mudler/vllm.cpp/issues/2492),
[#2503](https://github.com/mudler/vllm.cpp/issues/2503)
**Parent spec:** [`qwen4-exp-cuda-rmsnorm-weight-dtype.md`](qwen4-exp-cuda-rmsnorm-weight-dtype.md),
which lists both under `## Owed`
**Base:** `origin/main` at `f6105b2e5`

## Scope

Close the last two items of correctness debt behind the `qwen4_exp` CUDA row.

1. **#2492.** `RmsNormRowKernel` in `src/vt/rocm/rocm_rmsnorm.hip` and
   `RmsNormQuantFp8RowKernel` in `src/vt/cuda/cuda_ops.cu` both still take the
   gamma as `const Tin*` -- the ACTIVATION's element type -- so both dispatchers
   have to refuse `w.dtype != x.dtype`. Give each its own `Tw`, and NARROW the
   refusal instead of deleting it, exactly as
   [#2493](https://github.com/mudler/vllm.cpp/pull/2493) did for the CUDA
   `RmsNorm`.
2. **#2503 half 1.** `vt::RmsNorm` admits `IsFloat(weight.dtype)`
   (`src/vt/ops.cpp:1030`), and `IsFloat` is `{kF32, kF16, kBF16}` (`:24`). The
   CPU kernel widens a kF16 gamma like any other (`cpu_ops.cpp:554-557`); the
   CUDA dispatcher refused it. **Support it** rather than record it, and say why.
3. **#2503 half 2.** Settle the claimed blast radius of the bf16-activation /
   f32-gamma pairing #2477 newly admitted, by measurement rather than assumption.

Out of scope, each recorded under `## Owed`: the kF16 ACTIVATION divergence, the
ROCm `RmsNormPlusAdd`/`DualRmsNormPlusRes` welds, and #2488/#2476/#2496.

## The oracle

vLLM is this row's primary oracle and it never couples the two dtypes.
`Qwen4ExpRMSNorm.forward` upcasts the gamma on its own:

```python
return (normalized * (1.0 + self.weight.float())).to(input_dtype)
```

`vllm/models/qwen4_exp/nvidia/ple_layer.py:80`. `.float()` accepts ANY float
gamma dtype, `float16` included, which is the whole basis for supporting kF16
below rather than recording its refusal.

**This citation is OFF-PIN and is a forward reference.** It is read at vLLM
`origin/main` `cdefd9d499`. This project's pin is `5559679229`, which has no
`vllm/models/qwen4_exp/` at all -- vLLM landed the architecture after the pin.
The path is `vllm/models/`, NOT `vllm/model_executor/models/`.
[#2502](https://github.com/mudler/vllm.cpp/issues/2502) is reconciling the pin.
Nothing here is gateable against `5559679229`.

## Design

### #2492, both twins

Same shape both times, and it is the shape `cuda_ops.cu` already carries:

* the kernel takes `template <typename Tin, typename Tw, ...>` and `const Tw* w`;
* the launcher forwards `Tw`;
* a new `DispatchRmsNormWeight` / `DispatchRmsNormQuantFp8Weight` switches on
  `w.dtype` and instantiates the right `Tw`;
* the old `VT_CHECK(w.dtype == x.dtype, ...)` becomes that switch's `default:`
  arm, which names the dtype it got.

The refusal is NARROWED, not deleted. Widening the check on its own would have
read a bf16 gamma through a `const float*` and run off the end of it. `Load()`
is overloaded for every gamma element type, so each kernel body and its f32
arithmetic are unchanged, and every pairing that already worked stays
bit-identical.

### #2503 half 1: support kF16, do not record it

Four reasons, in order of weight:

1. **The oracle upcasts any float gamma.** `self.weight.float()` has no dtype
   arm. Mirroring vLLM means accepting what vLLM accepts.
2. **The CPU sibling already accepts it.** `cuda_qwen4_exp.cu:60-62` states the
   rule: "a device arm that refused a dtype its CPU sibling accepts would be a
   divergence to record". Removing the divergence is strictly better than
   recording it, and it removes it for ROCm at the same time.
3. **This tree already promises kF16 on CUDA.** `CudaPlatform::supported_dtypes`
   returns `{kBF16, kF16, kF32}` (`src/vllm/platforms/cuda.cpp:112`), mirroring
   upstream `interface.py:181-187`. `CpuPlatform` and `RocmPlatform` say the
   same.
4. **The change cannot regress a working pairing.** It adds one `case` to a
   `switch` and one `__device__ Load` overload. No existing arm's code changes.

The alternative the issue offers -- narrowing `vt::RmsNorm`'s admission so the
seam stops promising kF16 -- was rejected because it would make the seam refuse
what both the CPU kernel and the oracle accept, to make a device arm's gap look
intentional.

**What this does NOT do.** kF16 is admitted as a GAMMA only. `x` and `out` stay
f32/bf16 on the CUDA and ROCm arms, while the CPU kernel widens a kF16
activation too and `IsFloat(x.dtype)` admits it at the seam. That is the SAME
divergence class, one operand over, and it is recorded under `## Owed` rather
than fixed: closing it means a f16 `Tin` instantiation through both vectorised
decode-fast paths, the residual round-trip and `LoadVec`, which is a different
change with a different risk profile.

### #2503 half 2: the blast radius

See `## Evidence`. The short version is that the claim in the issue does not
survive contact with the loaders.

## Risks

* ~~**The ROCm edit cannot be compiled anywhere.**~~ **RETIRED, and it was the
  risk this wave paid to remove.** It was written when the only AMD device
  reachable was busy, and it said this change would land uncompiled. It did not:
  `hipcc` 7.2.4 built the TU clean for `gfx1151` on `strix:gpu0` with zero
  warnings, negative control detected, at the exact bytes that are landing (see
  `## Evidence`). What SURVIVES the retirement is the structural half: no lane in
  `.github/workflows/` invokes `hipcc`, `VLLM_CPP_HIP` defaults to `AUTO` = OFF
  (`CMakeLists.txt:124`), so the compile is a DATED REPORT on one revision and not
  a standing gate. The next edit to this file starts from where this one did.
* **The CUDA edits are compiled by `cuda-fat-build` and executed by nothing.**
  There is no CI lane that runs GPU tests, so every CUDA case in this suite is
  `[SKIP]` in practice.
* **`__half` is a new element type in both files.** `cuda_fp16.h` was not
  previously included by `cuda_ops.cu` and `hip_fp16.h` was not previously
  included by `rocm_rmsnorm.hip`. A missing include is a compile error, not a
  silent wrong answer.
* Instantiation count for `RmsNormRowKernel` goes from 16 to 24 and for
  `RmsNormQuantFp8RowKernel` from 4 to 12. Compile time and archive size grow;
  no runtime path changes.

## Tests

`tests/vt/test_ops_rmsnorm_weight_dtype.cpp` (extended) and
`tests/vt/test_ops_rmsnorm_quant_fp8_weight_dtype.cpp` (new).

Every case names the production site it models, and every CPU case compares
against an independently recomputed reference rather than against the op itself.

| Case | Op / pairing | The site it models |
|---|---|---|
| 1 | `RmsNorm`, f32 act / bf16 gamma | `qwen4_exp_qsa_block.cpp:705` -- the site that refused in #2477. A REAL production pairing |
| 2 | `RmsNorm`, bf16 act / f32 gamma | **NO production site.** It models the seam's admission at `ops.cpp:1030`, and its comment now says so instead of naming five architectures that cannot produce it |
| 3 | `RmsNorm`, bf16 act / {f32, f16, bf16} gamma | **NO production site for the f16 arm.** It models `ops.cpp:1030` + `IsFloat` at `:24` and the CPU arm at `cpu_ops.cpp:554-557` |
| 4 | `RmsNormQuantFp8`, bf16 act / bf16 gamma | `qwen3_5.cpp:6869` and `:6866` -- the op's ONLY production caller, and the anchor the mixed arms are compared against |
| 5 | `RmsNormQuantFp8`, bf16 act / {f32, f16} gamma | **NO production site.** It models the seam's admission at `ops.cpp:748` |

**The bar is BYTE EQUALITY, not a tolerance.** Every gamma element in cases 3 and
5 is `k/64` with `|k| <= 8`, which is exact in f32, binary16 and bfloat16 alike,
so the three dtype arms must produce byte-identical output. A tolerance would
have passed a build that read a 16-bit gamma through a `const float*`: that reads
`2H` bytes out of an `H*2`-byte buffer and much of the garbage is denormal-small,
which a 1%-relative bar absorbs. Each case asserts the exactness premise with
`REQUIRE` before it relies on it.

Case 3 CHANGES POLARITY in this wave: #2493 pinned it as a CUDA refusal with
`CHECK_THROWS_WITH(..., doctest::Contains("unsupported weight dtype"))`; it now
pins CPU/CUDA agreement. That is the point of the change and the test says so.

## Gates

```sh
ctest --test-dir build -R 'test_ops_rmsnorm' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Evidence

### #2503 half 2: the blast radius does not exist, and the issue's premise is false

The issue asserts that the bf16-activation / f32-gamma pairing #2477 newly
admitted is "reachable wherever a GGUF keeps its gamma at F32", and names
`glm4.cpp:173`, `gemma3.cpp:252`, `gemma2.cpp:259` and `:281`, `gemma.cpp:162`,
`deepseek_v2.cpp:509`. **Both halves of that are false, and the second is false
by construction.**

1. **None of those five architectures has a GGUF arm at all.** Each refuses a
   non-safetensors source by name at its registry door, before any weight is
   read:

   | Architecture | Refusal |
   |---|---|
   | `gemma_registry.cpp:48-51` | `"Model architecture GemmaForCausalLM does not support GGUF weights"` |
   | `gemma2_registry.cpp:49-52` | `"...Gemma2ForCausalLM does not support GGUF weights"` |
   | `gemma3_registry.cpp:50-53` | `"...Gemma3ForCausalLM does not support GGUF weights"` |
   | `glm4_registry.cpp:50-53` | `"...Glm4ForCausalLM does not support GGUF weights"` |
   | `deepseek_v2_registry.cpp:68-71` | `"...DeepseekV2ForCausalLM does not support GGUF weights"` |

2. **Their safetensors gammas are bf16, unconditionally.** All six sites take
   their gamma through `dense_attn::ResidentWeight`, which preserves
   `OwnedTensor::dtype` (`dense_attn_block.h:181`), over an owned tensor built by
   `dense_loaders::LoadBf16Direct`, which is `vt::DType::kBF16` on both its
   borrow and its copy arm (`dense_weight_loaders.h:355-374`):
   `glm4_weights.cpp:66`, `gemma3_weights.cpp:58`, `gemma2_weights.cpp:54`,
   `gemma_weights.cpp:52`, `deepseek_v2_weights.cpp:427`.

**Where f32 gammas DO live, and why they are not newly admitted either.** This
tree builds f32 norm gammas on five chains: kimi-linear (`kimi_linear.h:179`,
`:254`, f32 `std::vector` fields, `WF32` at `kimi_linear_device.cpp:292-295`),
Laguna (`laguna.cpp:2568`), the MiniMax-H3 text encoder
(`minimax_h3_encoder_sharded.cpp:228`, which comments *"Norms go to f32
(vt::RmsNorm's contract)"*) and its video VAE
(`minimax_h3_video_vae_device.cpp:79`), and Qwen3.5's q-norm
(`qwen3_5.cpp:5329`, `ResidentWeightF32`). **Every one of them pairs the f32
gamma with an f32 ACTIVATION**, so every one satisfied the old
`w.dtype == x.dtype` equality and none of them changed behaviour.

**Conclusion.** The set of production sites whose behaviour went from *throws* to
*runs* because of #2477 is exactly the `qwen4_exp` QSA norms
(`qwen4_exp_qsa_block.cpp:705`, f32 activation against a bf16 gamma) that the fix
was written for. There is no unmeasured blast radius, and the end-to-end run the
issue asks for would prove nothing about a pairing no loader can produce. The
fast-path observation in the issue -- that `TryLaunchRmsNormDecodeFast` requires
`w.dtype == DType::kBF16` (`cuda_ops.cu`) so an f32 gamma silently misses the
2.41x kernel -- is a true statement about the guard with no production victim,
for the same reason.

**One live consequence was found and fixed in this flow.**
`qwen3_5.cpp:5334-5337` selects between `ResidentWeight` and `ResidentWeightF32`
for the k-norm gamma, and its comment said *"RmsNorm requires w.dtype ==
x.dtype"*. #2493 falsified that sentence. The comment is corrected here; the
selection itself is left alone, because the f32 upcast of a bf16 on-disk gamma is
exact, so both arms feed the kernel identical values and removing the branch
would change a shipped model's code for no measured gain.

### #2503 half 1: kF16 is served, and it is UNREACHED

No loader in this tree builds a rank-1 kF16 norm gamma. Every rank-1 norm weight
resolves to kF32 or kBF16 (the chains above), the rank-1 kF16 tensors that do
exist are the EXL3 Hadamard scales `suh`/`svh`
(`deepseek_v4_exl3_device.cpp:64-65`), which are consumed by `vt::Exl3Gemm` and
never by a norm, and `nemotron_h_weights.cpp:1044-1050` -- the one channel that
turns a config `dtype` string into a gamma dtype -- refuses `"float16"` and
`"half"` by name.

So `case DType::kF16` in `DispatchRmsNormWeight` and
`DispatchRmsNormQuantFp8Weight` **lands unreached by any model.** Under "Nothing
lands dead" that is admissible only when it is named, so it is named here, in the
commit body, and in the pull request body. Row `MODEL-MM-QWEN4-EXP` owns it and
[#2503](https://github.com/mudler/vllm.cpp/issues/2503) tracks it. It is served
rather than refused because the SEAM promises it (`ops.cpp:1030`, `:748`,
`IsFloat` at `:24`), the CPU arm keeps that promise
(`cpu_ops.cpp:554-557`, `:1042`), `CudaPlatform::supported_dtypes` advertises it
(`platforms/cuda.cpp:111-113`), and the oracle upcasts any float gamma.

### The gate, read literally

Green baseline first, because a mutation kill against a red baseline is void:

```
NINJA_RC=0
test_ops_rmsnorm_weight_dtype            RC=0   6 cases / 263 assertions
test_ops_rmsnorm_quant_fp8_weight_dtype  RC=0   2 cases / 195 assertions
test_ops_rmsnorm                         RC=0  10 cases /  36 assertions
ctest --test-dir build -R 'test_ops_rmsnorm' --output-on-failure   CTEST_RC=0, 4/4
```

The assertion counts are stated because `assertions: 0 ... SUCCESS!` at rc 0 is a
skip wearing a pass. Both CUDA arms print `[SKIP]` on this host and no CI lane
runs GPU tests, so those cases contribute nothing to those counts.

### Mutations, with counted applied-ness

Each mutation asserts the ORIGINAL anchor count goes 1 -> 0 and the mutant count
0 -> 1 before anything is built, requires the rebuild's own rc to be 0 (a build
failure reads as a passing test), and is restored with `cmp -s` returning 0 and
an empty `git status` for the file.

| # | Mutation | Applied-ness | Build | Result |
|---|---|---|---|---|
| MUT-1 | `cpu_ops.cpp` `RmsNormKernel`: `WidenRowToF32(w.dtype, ...)` -> `WidenRowToF32(x.dtype, ...)` -- literally re-welding the gamma to the activation, the defect this wave removes | 1->0 / 0->1 | rc 0 | **KILLED.** rc 1, 3 of 6 cases, 6 of 263 assertions, including `from_f16 == from_f32` and `from_bf16 == from_f32` |
| MUT-2 | `cpu_ops.cpp` `RmsNormQuantFp8Kernel`: read the gamma through a copy of `w` whose `dtype` is forced to `x.dtype` -- the same re-weld on the fp8 twin | 1->0 / 0->1 | rc 0 | **KILLED.** rc 1, 1 of 2 cases, 3 of 195 assertions: both mixed-gamma equalities AND the load-bearing bump |
| MUT-3 | `cpu_matmul_elem.cpp` `WidenRowToF32`: the `kF16` arm widens with `BF16ToF32` -- isolates the f16 path alone | 1->0 / 0->1 | rc 0 | **KILLED, discriminatingly.** rc 1 and exactly ONE assertion, `from_f16 == from_f32`. The fp8 suite stayed rc 0, correctly: its kernel reads the gamma through `LoadF32`, not `WidenRowToF32` |

MUT-3's negative half is the informative one. It shows the two suites reach the
gamma through different code, and that the f16 equality is not riding on the
other assertions.

**No mutation could be run against either device arm ON THIS HOST.** There is no
`nvcc` and no `hipcc` here and no CUDA or HIP header anywhere on it (`find /
-maxdepth 4 -name cuda_bf16.h` returns nothing). The ROCm arm was subsequently
COMPILED on a leased `strix:gpu0`, and that job carried its own negative control
-- breaking the `case DType::kF16` arm and requiring `hipcc` to reject it, which
it did (`NEGCTL_NINJA_RC=1`). So the ROCm edit is no longer resting on the
brace/paren/bracket balance check that was the only instrument available when
this paragraph was first written; that check remains what it was, which is not a
compiler. The CUDA edit is compiled by `cuda-fat-build` in CI and by nothing
here, and neither device arm has been RUN.

**The reachability mutation the protocol asks for was NOT run, and cannot be for
this change.** Deleting `qwen3_5.cpp:6869` in a scratch copy would not move either
suite, because both enter through the `vt::RmsNorm` / `vt::RmsNormQuantFp8` op
seam rather than through `ModelRegistry::Forward`. What is true and checkable
instead: the f32-gamma and bf16-gamma arms of both dispatchers ARE reached by
production (`qwen4_exp_qsa_block.cpp:705` and `qwen3_5.cpp:6869` respectively),
and the kF16 arm is reached by nothing, which is declared above.

### What was compiled, and where

| Change | Compiled by | Executed by |
|---|---|---|
| `tests/vt/test_ops_rmsnorm_weight_dtype.cpp` | this host, `cmake --build build -j 2` | this host, `ctest` |
| `tests/vt/test_ops_rmsnorm_quant_fp8_weight_dtype.cpp` | this host | this host |
| `src/vt/cuda/cuda_ops.cu` | `cuda-fat-build` in CI ONLY -- there is no `nvcc` on this host | nothing; no CI lane runs GPU tests |
| `src/vt/rocm/rocm_rmsnorm.hip` | **`hipcc` 7.2.4 for gfx1151 on `strix:gpu0`, clean, 0 warnings, TWICE -- once as a single TU and once inside a full 22-object HIP build of the MERGED tree** -- see below. NOT by this host and NOT by CI: there is no `hipcc` and no HIP header here, and `grep -rlE 'hipcc\|ROCM\|rocm' .github/workflows/` returns nothing | nothing |

### The ROCm compile, and why it is a lease rather than a lane

`strix:gpu0` (`gpu_model=Radeon-8060S`, gfx1151, an RDNA3.5 APU) is the fleet's
only AMD device, and a **compile-only** job there is the one check available for
this file. It is not a CI substitute: `VLLM_CPP_HIP` defaults to `AUTO` = OFF
(`CMakeLists.txt:124`) and no workflow sets it, so nothing in this repository
will ever compile a `.hip` TU on its own.

The job is `/workspace/rmstwins-rocm/run.sh`, staged with a `git archive` of
this branch's exact head so the compile is of the tree under review rather than
of a re-clone that could drift. It holds the lease for one `cmake` configure and
ONE object file -- no model, no test, no GPU work -- and it refuses to report
anything unless it can first prove three things:

1. **It has the right tree.** Five counted guards on the extracted
   `rocm_rmsnorm.hip` before `hipcc` is allowed to claim anything: the
   4-parameter kernel template = 1, `const Tw* w` = 1, `DispatchRmsNormWeight`
   = 4, `__half` = 2, and the REMOVED weld `weight dtype must match x` = **0**.
   A compile of the wrong tree reads exactly like a compile of the right one.
2. **The object did not already exist.** A pre-existing `.o` would read as a
   pass, so its presence before the build is a hard abort.
3. **`hipcc` actually compiled THIS file.** A green `ninja` can mean "nothing to
   do". After the real compile the job breaks one line of the code this wave
   added -- the `case DType::kF16: LaunchRmsNorm<Tin, __half>` arm -- asserts the
   edit applied (count 1), and REQUIRES the rebuild to fail. If that still
   succeeds the verdict is `INCONCLUSIVE`, not `PASS`. The file is then restored
   and rebuilt clean.

`CCACHE_DIR` stays on `/tmp` with the shared store reached through
`CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote`, because this file's own
§build-cache measurement recorded the alternative -- a ccache dir on the CIFS
`/workspace` -- producing 278 lock failures with `ccache -s -v` never returning.

**Result: `COMPILE=PASS`.** `rc` job `d99a30fa-3336-4b90-bc9e-877717c9b54e`
on `strix:gpu0`, 2026-09-02T02:02:55Z, 1m56s wall:

```
gfx_enumerated: gfx1151          rocm_version: 7.2.4
template <typename Tin, typename Tw, typename Tout, typename Tres>  1 (want 1)
const Tw* w (the decoupled gamma pointer)                           1 (want 1)
DispatchRmsNormWeight (defn + 2 call sites + comment)               4 (want 4)
__half (the new gamma element type)                                 2 (want 2)
the REMOVED weld 'weight dtype must match x'                        0 (want 0)
CMAKE_RC=0     -- ROCm backend: ENABLED for arch(es) [gfx1151]
OBJ=CMakeFiles/vllm.dir/src/vt/rocm/rocm_rmsnorm.hip.o
pre-build: object absent, as required
NINJA_RC=0
typo_applied_count=1 (want 1)
NEGCTL_NINJA_RC=1 (want NON-ZERO)      restored byte-for-byte: yes
RESULT NEGCTL=DETECTED
RESULT COMPILE=PASS  file=src/vt/rocm/rocm_rmsnorm.hip arch=gfx1151
RESULT HEAD_SHA=2d1fdc19bd60cffec64fd2e66eb8abb5b5c1e1c5
RESULT WARNINGS=0
```

**The negative control is what makes the `NINJA_RC=0` mean anything.** A green
`ninja` can mean "nothing to do". Breaking the `case DType::kF16` arm made the
rebuild FAIL, so `hipcc` demonstrably compiled this file's new code rather than
skipping it or serving a cache hit.

**The verdict is attached to a revision, and that revision is still current.**
The job compiled tree `2d1fdc19b`; this branch has since taken two merges from
`origin/main`. `git diff 2d1fdc19b HEAD -- src/vt/rocm/rocm_rmsnorm.hip` is
EMPTY and the file's sha256 is `164392b422f93f65...` at both, so the report
covers the bytes that are landing. If this file changes again the report does
not follow it -- a leased compile is a dated report, not a standing gate.

**A SECOND job compiled the WHOLE ROCm backend on the MERGED tree.** Job 1
proved one TU at `2d1fdc19b`; this branch then took two merges, and
BACKEND-ROCM-EXL3 had just written up that exact gap for itself (`2e200e2c6`:
"the merged tree had never been compiled under HIP ... I should have queued that
before pushing, not after"). Its own full-HIP build was at `9dcc34cc3`, which
PREDATES this change and therefore compiled the OLD `rocm_rmsnorm.hip`. Neither
existing result covered the merged tree WITH this change, so job
`f1323c94-cbe0-469d-b2d4-8a3e553b336f` did:

```
HEAD_SHA=fcd446b033877fc436a75aabbeb1231465909a59   (this branch's head)
const Tw* w  1 (want 1)   DispatchRmsNormWeight  4 (want 4)
removed weld 0 (want 0)   cuda fp8 dispatcher    3 (want 3)
CMAKE_RC=0        hip_object_targets=22
NINJA_RC=0        hip_objects_built=22 (configured 22)
rocm_rmsnorm.hip.o  187176 bytes
RESULT HIPBUILD=PASS  target=vllm hip_objects=22 arch=gfx1151
RESULT WARNINGS=6     RESULT ELAPSED=291s
```

It asserts `hip_objects_built >= 20` after the build rather than trusting
`NINJA_RC`, because a configure that quietly dropped the HIP sources would give a
green ninja with nothing to show for it.

**The 6 warnings are NOT in this wave's file.** They are three distinct
`-Wunused-value` diagnostics in `src/vt/graph_dedup_runtime.h` (`:146`, `:279`,
`:283`), each emitted from two TUs. All three are intentional discards of a
`hipError_t`: `:146` is `ClearLatchedError`, whose own comment says its whole job
is to CONSUME the runtime's sticky per-thread error, and the other two are
`DestroyExec`/`DestroyGraph` cleanup helpers. They appear only under HIP because
ROCm marks `hipError_t` `nodiscard` where CUDA does not, which is precisely the
class of thing no lane can currently see. `rocm_rmsnorm.hip` itself compiles with
ZERO warnings in both jobs. They are reported here rather than filed, because an
intentional discard in a destroy path is not a defect.

**What it does NOT say.** No kernel in that file has been RUN on an AMD device;
this is a compile, and the row's `## Owed` still carries the absence of any lane
that compiles `src/vt/rocm/` at all. It also says nothing about the other twenty
`.hip` TUs, which this job did not build.

A `FAIL` is a defect found before landing and is the whole point of paying for
the lease. It would be owned by this row and repaired on top, not hidden.


## Stop conditions

This condition was DISCHARGED rather than left standing: the compile was taken
on `strix:gpu0` before landing, so a ROCm compile failure is no longer something
for a later reader with an AMD box to find. A RUNTIME failure still is, because
nothing in this wave executed a ROCm kernel, and that remains attributable to
this wave by the same reasoning.

## Owed

- [#2542](https://github.com/mudler/vllm.cpp/issues/2542) --- the kF16
  ACTIVATION divergence. `IsFloat(x.dtype)` admits kF16 at `ops.cpp:1030`, the
  CPU kernel widens it (`cpu_ops.cpp:561-562`, `:576`), and both
  `RmsNormKernelCuda` and `RmsNormKernelRocm` refuse it by name. Same class as
  #2503 half 1, one operand over, and a much larger change: `Tin` also drives
  `ResRound`, `LoadVec`, both decode-fast guards and the `out.dtype` switch.
- [#2543](https://github.com/mudler/vllm.cpp/issues/2543) ---
  `RmsNormPlusAddRocm` (`rocm_rmsnorm.hip`) and `DualRmsNormPlusResRocm` weld
  every operand's dtype to one `T`, while `vt::RmsNormPlusAdd`'s non-ROCm
  composed reference (`src/vt/fused_ops.cpp:19-31`) is `RmsNorm` + `Add` and
  therefore accepts a mixed gamma. A third instance of the weld, in the same
  uncompilable file. Not fixed here: the two twins #2492 names are the scope, and
  every additional blind ROCm line is unverifiable surface.
- #2488, #2476, #2496 stay with the parent spec.
- A CI lane that executes GPU tests, and a ROCm lane that merely COMPILES
  `src/vt/rocm/`. The second is the one this wave felt: it had to buy a device
  lease to learn whether one translation unit compiles, and that answer does not
  survive the next edit to this file. BACKEND-ROCM-EXL3 paid for the same thing
  independently on the same day (`2e200e2c6`, job `d218dc2d`, a full HIP
  `cmake`/`build`/`ctest` at `9dcc34cc3`) and wrote up the identical lesson --
  "I should have queued that before pushing, not after". Two waves buying the
  same missing lane in one day is the argument for building it. A compile-only
  ROCm lane costs a container and retires the whole "verified by reading" class.

## Now

`ACTIVE`.
