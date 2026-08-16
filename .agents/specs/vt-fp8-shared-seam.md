# VT-FP8-SHARED-SEAM — the FP8 W8A8 linear path becomes a shared seam

Issue: [#940](https://github.com/mudler/vllm.cpp/issues/940).
Owning row: `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` (the row that forces it; [#517](https://github.com/mudler/vllm.cpp/issues/517),
spec [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md)).
Related: [`vt-fp8-w8a8-cpu-arm.md`](vt-fp8-w8a8-cpu-arm.md) — #468/#842's op-tier
registrations, whose §Residual gap names exactly the model-tier half this row
closes; [`perf-fp8-alpha-fold.md`](perf-fp8-alpha-fold.md) — the folded-alpha
lever whose arithmetic this seam carries unchanged.

AGENTS.md §"Shared seams": *"If a shared seam cannot represent the upstream
behavior, extend it or record one exact tracked exception. Never hand-roll a
parallel path."*

NVFP4 already honours that:
`include/vllm/model_executor/models/dense_nvfp4_gemm.h` is a real seam with a
policy layer at `compressed_tensors/schemes/nvfp4.h`. FP8 W8A8 did not. The
weight struct `Fp8Weight` was in a header
(`qwen3_5_weights.h:273 @ c7cb59fbb`), but everything that made it *usable* sat
inside one translation unit:

| Entry point | Was (`qwen3_5.cpp @ c7cb59fbb`) |
|---|---|
| `ResidentFp8` | `:1458` |
| `DenseCublasLtFp8Enabled` | `:1478` |
| `MatmulFp8CutlassD` | `:1495` (CUDA guard `:1497-1498`) |
| `MatmulFp8CutlassPreQuantD` | `:1517` (CUDA guard `:1519-1520`) |

A second model therefore had three options and the policy forbids two: include a
`.cpp`, copy the entry points, or extend the seam. This row extends it.

## Scope

**IN.**

1. `include/vllm/model_executor/models/dense_fp8_gemm.h` — the four definitions
   above, moved VERBATIM, mirroring `dense_nvfp4_gemm.h`'s shape (preamble with
   the upstream chain, `namespace vllm::dense_fp8`, header-only, the CUDA guard
   travelling with the code it guards).
2. `include/vllm/model_executor/layers/quantization/fp8.h` — `Fp8W8A8LinearMethod`
   + `MakeLinearMethod(const OwnedTensor&, const Fp8Weight&)`, mirroring
   `compressed_tensors/schemes/nvfp4.h`'s `Nvfp4W4A16LinearMethod` +
   `MakeLinearMethod`.
3. `src/vllm/model_executor/models/qwen3_5.cpp` — calls the extracted seam. The
   ~14 call sites are UNCHANGED text; what changes is that the three names they
   call are now two one-line type adapters plus a `using`.
4. `tests/vllm/model_executor/layers/test_linear_method.cpp` — the two CPU cases
   that make the policy layer's selection and its reach into the seam
   falsifiable.

**OUT.**

- **Wiring NemotronH to this seam.** That is A2-Q under #517, and it is what the
  seam exists for, but a model port inside an extraction would make the
  byte-identity gate below unreadable.
- **Any numerics, tolerance, guard or dispatch condition.** Specifically the
  CUDA-only refusal keyed on `kMatmulFp8CublasLt` is carried across unchanged —
  see §Residual gap.
- `ResidentFp8Qkv` / `ResidentFp8Qkvz` and the merged-QKV(z) fp8 path. They are
  35B-specific merged-operand builders, not the general linear seam, and #940
  does not name them.
- The missing device-upload accounting recorded under §Found, not fixed.

### Why the seam is templated on `Dev`/`DBuf`

`qwen3_5.cpp` carries its own anonymous-namespace `Dev`/`DBuf` — the KNOWN
DUPLICATION `dense_nvfp4_gemm.h:45-50` records, deliberately, because unifying
the device-glue families is a separate gate-model-touching refactor. Those types
are layout-identical to `dense_attn::Dev`/`DBuf` but are *distinct types*, so a
non-template header could only have been COPIED into qwen3_5.cpp, not called by
it.

That copy is the failure mode #940 exists to prevent: a seam sitting dead beside
the production path, where "byte-identical" is vacuously true because nothing
routed through it. Templating on the two glue types instead gives ONE definition
with two instantiations — qwen3_5.cpp instantiates it with its own types
(generating the code it had), the `vllm::layers` policy layer instantiates it
with the shared ones. The mutation table below is what turns that from a claim
into a measurement.

## Upstream chain

Our loader (`LoadFp8Raw`, `qwen3_5_weights.cpp:423`) accepts both the
compressed-tensors and the ModelOpt spelling of the same per-tensor FP8 W8A8
checkpoint and reduces them to one `Fp8Weight`. Upstream, all three entry points
delegate to the same shared `fp8_linear.apply_weights`, which is why one method
here covers every spelling:

| Concern | Upstream (pin `555967922`) |
|---|---|
| compressed-tensors scheme | `compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:60,201-207` |
| ModelOpt scheme | `modelopt.py:444,531-537` |
| generic fp8 linear | `fp8.py:267,446` |
| static per-tensor act quant | `utils/quant_utils.py:124` `kFp8StaticTensorSym`, handed to `init_fp8_linear_kernel` at `modelopt.py:511-512` → our `vt::QuantFp8Static` |
| per-tensor scaled epilogue | the folded `alpha = input_scale * weight_scale` |
| scheme selection | `base_config.py:180` `get_quant_method` → `MakeLinearMethod` |

**Header placement.** `quantization/fp8.h`, not `quantization/schemes/fp8.h`.
#940's text names `schemes/nvfp4.h` for a file that actually lives at
`compressed_tensors/schemes/nvfp4.h`, so it is using a shorthand, and inventing a
`quantization/schemes/` directory would mirror nothing upstream. `fp8.py` sits
directly under `quantization/` in vLLM, and `layers/quantization/
modelopt_mixed_precision.h` is the local precedent for a header-only policy
header in that same place.

## Design

`dense_fp8_gemm.h` is header-only and carries, in order:

- `DenseCublasLtFp8Enabled()` — `VT_DENSE_CUBLASLT_FP8`, default ON. Unchanged
  lever, unchanged spelling, unchanged default.
- `ResidentFp8(DevT, const Fp8Weight&)` — the lazy one-shot upload of the raw
  fp8 `[N,K]` bytes, owned by the (const) weight's `mutable shared_ptr`.
- `MatmulFp8CutlassD<DBufT>(DevT, x, w, out_dtype)` — `QuantFp8Static` then the
  folded-alpha fp8 GEMM (cuBLASLt by default, cutlass under the lever).
- `MatmulFp8CutlassPreQuantD<DBufT>(DevT, a_fp8, w, out_dtype)` — the same GEMM
  over an activation a preceding fused epilogue already quantized. Upstream this
  is the `x: torch.Tensor | QuantizedActivation` overload of the same apply.

`quantization/fp8.h` adds nothing computational: `Apply` is
`MatmulFp8CutlassD<DBuf>`, `ApplyPreQuantized` is
`MatmulFp8CutlassPreQuantD<DBuf>`, and `MakeLinearMethod` selects on
`Fp8Weight::Empty()` exactly as the NVFP4 factory selects on
`Nvfp4Weight::Empty()`.

### Residual gap — carried, not closed

`MatmulFp8CutlassD` refuses on a host queue because its guard asks for
`kMatmulFp8CublasLt`, registered for kCUDA only, while the ops it would actually
run (`kQuantFp8Static`, `kMatmulFp8Cutlass`) DO have CPU reference arms since
#468/#842. That mismatch is recorded in
[`vt-fp8-w8a8-cpu-arm.md`](vt-fp8-w8a8-cpu-arm.md) §Residual gap and pinned at
`tests/vt/test_ops_fp8_cpu.cpp:445-453`. Widening it is a dispatch change; an
extraction that quietly widened it would be invisible to a byte-identity gate,
which is the whole reason it is out of scope here. The new CPU case re-pins it
at the model tier so it stays visible rather than assumed closed.

## Risks

- **The seam lands dead beside the production path.** Mitigated by construction
  (one definition, instantiated) and MEASURED by mutations M3–M4 below, which
  perturb the header and require a Qwen3.5 CUDA arm to go red.
- **A behaviour change hides inside a move.** Mitigated by moving the bodies
  verbatim and by the assertion-count identity below; the only textual change is
  `MakeTensor` → `dense_attn::MakeTensor` (identical body,
  `dense_device_glue.h:47` vs `qwen3_5.cpp:583`).
- **Template instantiation differs from the inline code.** Both instantiations
  are `inline` in the same TU as before; the CPU full gate and the CUDA arm are
  what test this rather than the argument.

## Tests and gates

Qwen3.5 byte-identity is the gate. Assertion counts before and after must be
identical for every pre-existing suite; the two NEW cases are additive and
declared as such.

| Suite | Before (`c7cb59fbb`) | After |
|---|---|---|
| `test_qwen3_5_gdn_spec_routing` | 6 cases / 52 assertions | 6 / 52 |
| `test_ops_fp8_cpu` | 4 / 56 | 4 / 56 |
| `test_qwen27_paged_forward` | 29 / 765 | 29 / 765 |
| `test_qwen27_dense_forward` | 9 / 583 | 9 / 583 |
| `test_linear_method` (CPU box; dgx adds 2 `VT_MARLIN_NVFP4` cases / 9 assertions) | 6 / 76 | 8 / 88 (+2 new cases) |

Full CPU gate: clean configure + `cmake --build build -j 12` (0 warnings under
`-Werror`) + `ctest -j 4`.

Mutation table (the "the seam is LIVE" proof) is recorded in §Evidence.

## Found, not fixed

`ResidentFp8` — and its merged siblings `ResidentFp8Qkv` / `ResidentFp8Qkvz` —
`Alloc` + `Copy` the fp8 weight bytes to the device without calling
`vllm::load_stats::AddDeviceUpload` and without the post-upload
`AdoptDeviceBytesAsHost` step. Both are performed by every other resident-weight
helper in the same file: `ResidentWeight` (`qwen3_5.cpp:1008,1015 @ c7cb59fbb`)
and `ResidentNvfp4` (`:1105,1110,1115,1120`), and `dense_nvfp4_gemm.h:294-328`
carries the comment explaining why the pair is mandatory ("this is the one
host->device move of those bytes and it must be accounted and followed by the
same post-upload residency step every other qualifying weight gets", ENG-LOAD
-DIRECT-UPLOAD / #150).

Consequences, both plausible and neither measured here: the 35B fp8 tower's
upload is missing from load accounting, and its device pages are never re-tagged,
which is the shape of the GB10 weight-residency ATS penalty. The first is
BOUNDED, and #974 says so: `load_stats` has exactly one non-test consumer,
`PrintLoadBytes` (`src/vllm/entrypoints/model_loader.cpp:133`), a stderr
diagnostic line behind `LoadStatsEnabled()`. Nothing allocates, sizes or
schedules on that counter, so the under-report costs the accuracy of a
diagnostic, not a decision. The second consequence carries no such bound.

Carried across UNCHANGED — repairing it inside an extraction would be exactly the behaviour
change hidden in a move that this row's gate cannot see. Filed as
[#974](https://github.com/mudler/vllm.cpp/issues/974).

## Owed

- [#974](https://github.com/mudler/vllm.cpp/issues/974) — the FP8 resident
  helpers upload without `load_stats::AddDeviceUpload` and without
  `AdoptDeviceBytesAsHost`, which every other resident-weight helper in the same
  file performs. Found here, carried across unchanged, and detailed under
  §Found, not fixed.
- **A gate sensitive to the pre-quantized arm's arithmetic.** M7 proves
  `MatmulFp8CutlassPreQuantD` is REACHED by two CUDA cases; M5 proves no
  available assertion can SEE a change to its folded alpha. The mechanism is
  **scale invariance of each predicate, NOT cancellation between the two arms.**
  Those cases compare nothing: each arm carries its own predicate, and both are
  invariant under a rescale of the arm they watch —
  `CHECK(silu_nonzero == 0)` (`tests/vllm/models/test_qwen3_5_gdn_spec_routing.cpp:895`)
  holds under ANY non-zero scalar, and `CHECK(max_sigmoid > 0.0)` (`:904`) holds
  under any POSITIVE one. `w.alpha * 2.0F` is positive and non-zero, so M5 could
  not have moved either.

  M9 is the disproof of the cancellation reading, because `w.alpha` → `0.0F` is
  a factor COMMON to both arms — exactly what cancellation predicts would stay
  green — and it turns those two cases RED, the only new failures being
  `CHECK(max_sigmoid > 0.0)` at `:904` while `CHECK(silu_nonzero == 0)` at
  `:895` stays green. Recorded as M9 in §Evidence.

  So what is owed is a gate sensitive to **any positive rescale** of that arm,
  not merely to a factor the two arms do not share. A per-projection
  `weight_scale` error is exactly that case: positive, non-zero, and invisible to
  both predicates above. The equality the header's own comment claims is the
  shape of it — `MatmulFp8CutlassPreQuantD(QuantFp8Static(x, w.input_scale), w)`
  against `MatmulFp8CutlassD(x, w)` on the same activation, which is sensitive to
  every positive factor. Coverage this extraction did not create and did not
  close.

## Protocol deviation: this spec was not committed before the implementation

AGENTS.md §"Spec before code": *"Commit the spec before implementation. Never
write the spec after the implementation."* This file does not satisfy that. It
was added in `a0693813a`, the SAME commit as the extraction, and the superseded
attempt (PR #972) had the same shape. Commit order is what proves the spec came
first when one pull request carries both, and here it proves the opposite.

It is recorded rather than repaired because it cannot be repaired: `main` is
never force-pushed and neither is a branch under review, so the history that
shows one commit cannot be rewritten into two. Renaming it would be worse — a
reviewer reading `git log` would see a spec commit that never existed.

What the deviation cost, stated plainly so a reviewer can price it: the scope,
the byte-identity gate and the two deliberate divergences below were argued
AFTER the code existed, so none of them constrained the implementation the way a
committed spec would have. What it did not cost: the mutation table in §Evidence
was designed and run against the landed code by a session that did not write it,
and the fresh review reproduced it independently, so the claims here are
measured rather than asserted. That is evidence about the code, not an excuse
for the ordering.

This is visible debt, not success. A reviewer who does not accept the argument
does not merge it.

## Stop conditions

- The extraction cannot be made behaviour-preserving — return `NEEDS_DECISION`
  with the demonstration rather than adapting the numerics.
- A mutation of the extracted code leaves every Qwen3.5 arm green: the seam is
  not on the production path and the byte-identity claim is vacuous. Do not
  land; re-wire until a mutation bites.

## Evidence

### CPU gate (this box)

Run THREE times. A merge that brings in new source is a new binary, so the extraction's own green does not cover either post-merge tree.

| At | build (`-j 12`, Ninja, clean configure) | `ctest -j 4` | time |
|---|---|---|---|
| `a0693813a` (extraction) | exit 0, 0 warnings under `-Werror` | 485/485 passed, 0 failed | 828.96 s |
| `018c9d1cb` (post-merge, `origin/main` @ `e5351776c`) | exit 0, 0 warnings under `-Werror` | 485/485 passed, 0 failed | 682.90 s |
| `66c1e805c` (post-merge, `origin/main` @ `c90e3fc02`) | exit 0, 0 warnings under `-Werror` | 487/488 passed, **1 failed** | 837.59 s |

All three skip the same 2 absent-fixture tests (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`). The third run's 488 is 485 plus the 3 tests `origin/main` @ `c90e3fc02` brought in.

**The one failure is the tracked flake [#294](https://github.com/mudler/vllm.cpp/issues/294)**, not a regression, and it was re-run before being called that, per [`verification.md`](../verification.md) ("tests that starve under `ctest -j` are re-run serially before being called a regression"):

- #294 records `test_async_llm` reusing an aborted request id as racing the core abort, at a **26% failure rate under contention, on main**.
- It PASSED in both earlier full runs of this same tree (0.14 s and 0.37 s) and FAILED in 0.31 s only in the third, which ran with the box at load average 29-35 from concurrent sessions.
- Re-run serially **3/3 green**, 15/15 assertions each.

Disk 91% before, 99% at peak, 93% after; the build tree was removed afterwards.

Pre-existing suites at `c7cb59fbb` -> `a0693813a`, identical: `test_qwen3_5_gdn_spec_routing` 6/52, `test_ops_fp8_cpu` 4/56, `test_qwen27_paged_forward` 29/765, `test_qwen27_dense_forward` 9/583, every one `Status: SUCCESS!`. `test_linear_method` 6/76 -> 8/88 (two new cases, additive) — the CPU-only box's shape; on dgx the same suite reads 10 / 97 because `VT_MARLIN_NVFP4` adds 2 cases and 9 assertions.

### Mutation table

**Every arm below ran on `dgx.casa`**, GB10 sm_121a, CUDA 13.0.88, image `vllmcpp-build:gb10`, `-j 4`, `$HOME/gpu.lock` held with `flock -n` for every test run. Both configure logs printed `CUDA feature cutlass-fp8: ENABLED for [121a]`, `cutlass-nvfp4: ENABLED for [121a]`, `fa2: ENABLED for [121a]`, `CUTLASS found at /cutlass` and `Triton AOT: ... sm_121a` — so no arm is voided. The seam header was restored byte-for-byte after every arm and verified against a pristine `sha256`, and each series' BASE re-run reproduced its own first binary sha exactly.

CPU arms are `test_linear_method`; GPU arms are `test_qwen3_5_gdn_spec_routing`. **Each kind carries its own BASE control measured in the same tree**, because a mutation row on a box with pre-existing failures cannot be read without one.

Two trees, stated per row. **T1 = `a0693813a`** (the extraction) carries CONTROL, BASE, M3, M4, M5, M7, M8. **T2 = `32980afda`** (this branch's reviewed head, transferred with `git archive`, clean configure + full build, `build rc=0`, 0 `error:`, 0 `warning:`) carries GDN_BASE, LM_BASE, M1, M2 and M9. Binary shas are comparable WITHIN a tree, never across the two.

| Arm | Tree | Perturbation | `compile_exit` | `error:` | binary sha16 | `[doctest] test cases:` | `[doctest] assertions:` | `Status:` / exit |
|---|---|---|---|---|---|---|---|---|
| **CONTROL** (GPU) | T1 | none, and `qwen3_5.cpp` replaced by its `c7cb59fbb` (pre-extraction) content in the SAME tree, SAME flags | 0 | 0 | `8b740f86eeb7da5d` | `12 \| 11 passed \| 1 failed` | `123 \| 119 passed \| 4 failed` | `FAILURE!` / 1 |
| **BASE** (GPU) | T1 | none | 0 | 0 | `090bc6e47a478cb2` | `12 \| 11 passed \| 1 failed` | `123 \| 119 passed \| 4 failed` | `FAILURE!` / 1 |
| M3 (GPU) | T1 | drop the `input_scale`: `QuantFp8Static(..., w.input_scale)` → `..., 1.0F` | 0 | 0 | `91cfeeb337fec5a3` | `12 \| 11 passed \| 1 failed` | `123 \| 107 passed \| 16 failed` | `FAILURE!` / 1 |
| M4 (GPU) | T1 | change the alpha fold: `w.alpha` → `w.alpha * 2.0F` in `MatmulFp8CutlassD` | 0 | 0 | `6a7fb2f31de34576` | `12 \| 11 passed \| 1 failed` | `123 \| 107 passed \| 16 failed` | `FAILURE!` / 1 |
| M5 (GPU) | T1 | same alpha fold in `MatmulFp8CutlassPreQuantD` | 0 | 0 | `ecebc93903e7d801` | `12 \| 11 passed \| 1 failed` | `123 \| 119 passed \| 4 failed` | **UNCHANGED vs BASE — negative result** |
| M7 (GPU) | T1 | `VT_CHECK(false, ...)` as the first statement of `MatmulFp8CutlassPreQuantD` | 0 | 0 | `84fd9f9d2d7386f7` | `12 \| 9 passed \| 3 failed` | `103 \| 99 passed \| 4 failed` | `FAILURE!` / 1 |
| M8 (GPU) | T1 | `VT_CHECK(false, ...)` as the first statement of `MatmulFp8CutlassD` | 0 | 0 | `dc8bcfc1e97fafac` | `12 \| 11 passed \| 1 failed` | `91 \| 91 passed \| 0 failed` | `FAILURE!` / 1 |
| **GDN_BASE** (GPU) | T2 | none | 0 | 0 | `47a9960ac92d4b66` | `12 \| 11 passed \| 1 failed` | `123 \| 119 passed \| 4 failed` | `FAILURE!` / 1 |
| **LM_BASE** (CPU) | T2 | none | 0 | 0 | `f8f5b2d3a0116980` | `10 \| 9 passed \| 1 failed` | `97 \| 95 passed \| 2 failed` | `FAILURE!` / 1 |
| M1 (CPU) | T2 | delete the CUDA guard in `MatmulFp8CutlassD` | 0 | 0 | `2a3a1a8e5df413fd` | `10 \| 8 passed \| 2 failed` | `97 \| 94 passed \| 3 failed` | `FAILURE!` / 1 |
| M2 (CPU) | T2 | delete the CUDA guard in `MatmulFp8CutlassPreQuantD` | 0 | 0 | `15d702dd127b1c35` | `10 \| 8 passed \| 2 failed` | `97 \| 94 passed \| 3 failed` | `FAILURE!` / 1 |
| M9 (GPU) | T2 | zero the alpha fold: `w.alpha` → `0.0F` in `MatmulFp8CutlassPreQuantD` (both GEMM arms; anchor count asserted `== 2`) | 0 | 0 | `6c9335e8b6a53228` | `12 \| 9 passed \| 3 failed` | `123 \| 115 passed \| 8 failed` | `FAILURE!` / 1 |
| **GDN_BASE re-run** (GPU) | T2 | none, after every T2 mutation was reverted | 0 | 0 | `47a9960ac92d4b66` (identical to GDN_BASE) | `12 \| 11 passed \| 1 failed` | `123 \| 119 passed \| 4 failed` | `FAILURE!` / 1 |

**Reading it.**

*The BASE red is inherited, not introduced.* `test_qwen3_5_gdn_spec_routing` reads 119/123 on GB10 at `main`, which is exactly what #907 already records for this box. The CONTROL row proves it in the same tree rather than by citation: pre-extraction `qwen3_5.cpp`, same build directory, same flags, distinct binary, and the result is identical down to the individual mismatch counts (`30504`, `48756`, `30504`, `48756`) and the same four `H=5120 / T=3` combinations. **That equality is the byte-identity evidence** — the extraction reproduces the production numerics including a pre-existing defect.

*The seam is on the live Qwen3.5 path.* M3 and M4 each take the failures from 4 to **16** (every combination in the case), from a source change inside `dense_fp8_gemm.h`, with a clean compile and a distinct binary. The split arm of `ProjectGdnFp8QkvzForTest` (`qwen3_5.cpp:6693-6694`) calls the extracted `MatmulFp8CutlassD` while the merged arm does not, so perturbing the seam breaks their bitwise equality. A seam sitting dead beside the model cannot do that.

*M5 is a negative result and is reported as one.* Doubling alpha in the pre-quantized arm changed nothing. M7 shows the arm is nevertheless REACHED: forcing that function to throw turns **two additional cases** red — `GDN gate POLARITY on the FP8 tail: GdnBlockPaged (CUDA)` and `... the MIXED spec batch (CUDA)`.

*M9 is why nothing SAW the doubling, and it corrects the first reading of M5.* The mechanism is **scale invariance of each predicate, not cancellation between the two arms** — those two cases compare their arms against nothing at all. Each arm has its own predicate: `CHECK(silu_nonzero == 0)` (`tests/vllm/models/test_qwen3_5_gdn_spec_routing.cpp:895`), invariant under ANY non-zero rescale, and `CHECK(max_sigmoid > 0.0)` (`:904`), invariant under any POSITIVE one. `w.alpha * 2.0F` is positive and non-zero, so M5 could not have moved either.

M9 sets that same alpha to `0.0F` — a factor COMMON to both arms, which the cancellation reading predicts would stay green — and it turns those two cases RED. The failure detail is what settles it: the four NEW assertion failures are all `:904 CHECK( max_sigmoid > 0.0 )` with `values: CHECK( 0 > 0 )` and `max_sigmoid := 0` (two dims each in both fp8 polarity cases), while `CHECK(silu_nonzero == 0)` at `:895` stays green with `silu_nonzero := 0`. The pre-existing four at `:525` (`30504`, `48756`, `30504`, `48756`) are untouched, which is how `123 | 119 passed | 4 failed` becomes `123 | 115 passed | 8 failed`. What the suite therefore owes is a gate sensitive to any POSITIVE rescale of that arm — see §Owed.

*M8 is also the reason to read cases and not just assertions.* It prints `assertions: 91 | 91 passed | 0 failed` — and `Status: FAILURE!` with exit 1, because the throw aborted the case before its `CHECK`s ran. An assertion-only reading of that line would have called a fully-blocked GEMM a pass.

*M1 and M2, corrected.* As first recorded these two rows were attributed to `dgx.casa` and read `8 | 7 passed | 1 failed` / `88 | 87 passed | 1 failed`. That is the CPU-ONLY box's shape, not this host's: `test_linear_method` puts 2 cases and 9 assertions behind `VT_MARLIN_NVFP4` (`tests/vllm/model_executor/layers/test_linear_method.cpp:105-272`), so a CUDA build here reads 10 cases / 97 assertions, and the 2 pre-existing failures sit inside one of those Marlin cases — `linear_method: MXFP4 fused gate_up ~= split`, both at `:247 CHECK( after == before + 1 )` with `values: CHECK( 0 == 1 )`. Those two are #907's `test_linear_method 83 of 85` plus this row's 12 additive assertions, all passing (85 + 12 = 97, 83 + 12 = 95). The rows above are RE-RUNS on `dgx.casa` against the `LM_BASE` control measured in the same tree, and the conclusion is unchanged: each guard deletion turns exactly ONE further case red, `linear_method: the fp8 w8a8 method reaches the shared seam in both arms`, at the `CHECK_THROWS_WITH_AS` that pins that guard's message (`:540` for M1, `:550` for M2) — the deleted guard lets the call reach `vt::MatmulFp8CublasLt`, which throws a different message.

*M6 is unused, and no arm is missing.* The labels run M1-M5 and M7-M9. Nothing was measured under an M6 label and nothing was withheld: the number was skipped when the GPU arms were planned, and the gap went unexplained until this round. It stays unused rather than renumbered, because `73d67f9d2` and this row's fresh review both cite M7 and M8 by those labels, and renumbering would silently break every citation. M9 continues the sequence.

### The inherited red

`test_qwen3_5_gdn_spec_routing` is 119/123 on GB10 at `main` and is already recorded as such by [#907](https://github.com/mudler/vllm.cpp/issues/907). It was NOT inherited on trust: the CONTROL arm above reproduces it from the pre-extraction `qwen3_5.cpp` in the same build tree, with the identical four failing combinations and the identical mismatch counts. Nothing here narrows or widens it.

## Outcome

Landed as a pure extraction. **§Evidence above** carries the before/after
assertion counts, the mutation table — thirteen rows: the pre-extraction
CONTROL, three BASE baselines, one restoration re-run, and eight mutation arms
M1-M5 and M7-M9 — and the full-gate result; `73d67f9d2` moved
that evidence out of the PR body and into this file, so this section points here
and not there.

**Measured.** The gate was met: Qwen3.5's CUDA result is identical to the
pre-extraction CONTROL down to the individual mismatch counts, and two
independent perturbations inside the extracted code (M3, M4) move it, so the
seam is on the production path rather than beside it. M7 and M8 show both entry
points are reached; M1, M2 and M9 show three further guarantees are gated.

**Rejected.** A non-template header, because `qwen3_5.cpp`'s
anonymous-namespace `Dev`/`DBuf` are distinct types from the shared ones, so a
non-template seam could only have been COPIED into the production TU — the dead
seam beside the path that #940 exists to prevent. Also rejected:
`quantization/schemes/fp8.h`, a directory that mirrors nothing upstream, and
repairing the #974 accounting gap in-flow, because a byte-identity gate is blind
to exactly that class of change.

**Defaults.** None were introduced. `VT_DENSE_CUBLASLT_FP8` keeps its spelling
and its ON default, and the CUDA-only refusal keyed on `kMatmulFp8CublasLt`
keeps its condition; both travelled with the code they belong to. The one
textual change in the moved bodies is `MakeTensor` → `dense_attn::MakeTensor`.

**Owed and open**: #974, the gate sensitive to any positive rescale of the
pre-quantized arm (§Owed), and the protocol deviation recorded above.

## Now

`DONE` — the seam exists and Qwen3.5 routes through it. Wiring
`MODEL-NEMOTRON-H`'s 46 FP8 W8A8 projections to it is A2-Q under #517 and is
NOT part of this row.
