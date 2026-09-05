# CUDA `RmsNorm` refuses the f32 activation the QSA block hands it

**Row:** `MODEL-MM-QWEN4-EXP` (wave SANITIZE)
**Issue:** [#2477](https://github.com/mudler/vllm.cpp/issues/2477)
**State:** `ACTIVE`
**Base:** `origin/main` at `855905f59`

> **This spec was wrong in its first four commits and is corrected here.** It
> claimed the gamma was f32 and the activation bf16, and named
> `qwen4_exp_qsa_block.cpp:632` as the refusing site. The opposite is true. The
> error and its consequences are kept in `## The error this spec made`, because
> the reasoning that produced it is the reusable part.

## Scope

Make the CUDA `RmsNorm` kernel read its gamma in the gamma's own dtype, so a
`--device cuda` forward of `qwen4_exp` on the released GGUF artifact stops
refusing at `src/vt/cuda/cuda_ops.cu:463`.

**This treats a symptom, not the cause.** The cause is
[#2488](https://github.com/mudler/vllm.cpp/issues/2488): the QSA block widens
`q` to f32 where vLLM keeps it at the bf16 model dtype. See
`## What actually refuses, and why this is still worth landing`.

Out of scope, each recorded under `## Owed`: #2488, #2503, #2492, #2476, #2496.

## What actually refuses

`RmsNormKernelCuda` held `VT_CHECK(w.dtype == x.dtype)`. That was an honest
mirror of the kernel behind it: `RmsNormRowKernel` was declared
`template <typename Tin, typename Tout, typename Tres>` and took the gamma as
`const Tin* w`, the **activation's** type. Widening the check alone would have
read a bf16 gamma through a `const float*` and run off its end.

**Every gamma in this model is bf16 at runtime.** The released
`unsloth/Qwen3.8-Flash-Next-GGUF` stores norm tensors as `F32` on disk, but the
loader converts: `qwen4_exp_weights.cpp`'s `LoadNormBf16` runs `DequantAll` into
an f32 vector and returns `Bf16From(...)`, which is `MakeTensor(kBF16, ...)`
(`glm_moe_dsa_loader.cpp:145-151`). All four QSA gammas take that path
(`q_norm`, `k_norm`, `idx_q_norm`, `idx_k_norm`), `LoadNormBf16` takes no policy
or device argument that could make it emit f32, `ResidentWeight` preserves
dtype, and those are the only assignments to those fields in `src/vllm/`.

**The activation is what diverges.** `qwen4_exp_qsa_block.cpp:694` allocates
`q_f32` at `DType::kF32` and `vt::AttnGateSplit` fills it. vLLM chunks `q`
straight off the bf16 QKV GEMM and hands it to `q_norm` unwidened
(`vllm/model_executor/models/qwen3_next.py:384-440`), so **upstream never runs
this op mixed on this architecture**.

| site | activation | gamma | verdict |
|---|---|---|---|
| `:632` `idx_q_norm` | `q_index_raw` (`:592`, `hidden.dtype`) bf16 | bf16 | passes |
| `:705` `q_norm` | **`q_f32` (`:694`) f32** | bf16 | **refuses — the wall** |
| `:736` `k_norm` | `k_raw` (`:732`, `hidden.dtype`) bf16 | bf16 | passes |

### Which of the three shapes this is

`qwen4-exp-matmul-bt-mixed-dtype.md` set this row's taxonomy. This is its
**shape 1**: an activation widened to f32 that should have stayed bf16. The fix
belongs at the producer and the CUDA combo can stay unsupported.

It is **not** shape 2. Nothing upstream runs this op mixed here.

### What actually refuses, and why this is still worth landing

The shape-1 fix is #2488: narrow `q_f32` to `hidden.dtype`. That is not done
here because `vt::AttnGateSplit`'s f32 output is welded into the op's own CUDA
signature (`cuda_glue.cu:181`,
`AttnGateSplitKernel(float* q_out, float* gate_out, ...)`) and shared with
`qwen3_5.cpp:5328` and `:5501`, so narrowing it moves a second model's path and
needs that model's gate.

Decoupling `Tw` from `Tin` is nevertheless correct on its own terms, and lands
as a symptom fix that is explicitly labelled one:

- vLLM's `GemmaRMSNorm` upcasts the gamma independently
  (`vllm/models/qwen4_exp/nvidia/ple_layer.py:80`, `1.0 + self.weight.float()`).
- This tree's CPU `RmsNormKernel` widens `w` and `x` separately
  (`cpu_ops.cpp:554-557`, `:576`).
- `RmsNormGroupKernelCuda` already carries an independent weight tag
  (`cuda_rms_norm_group.cu:194`).
- `cuda_qwen4_exp.cu:60-62`, from an earlier wave of this row, states the rule:
  "a device arm that refused a dtype its CPU sibling accepts would be a
  divergence to record".

Values are unchanged for every pairing that already worked, because `Load()` is
overloaded for both element types and the f32 arithmetic is untouched.

## The error this spec made

The first four commits asserted the gamma was f32 because the **GGUF stores it
as F32**, and built the whole diagnosis on that: shape 2 rather than shape 1,
`:632` rather than `:705` as the wall, and #2488 argued down to a bandwidth note
on the false premise that `:705` "pairs an f32 activation with the f32 gamma and
passes today".

The measurement was of the artifact. The claim was about the tensor. The loader
sits between them, and `LoadNormBf16` is 8 lines long.

Shape 1 was then "ruled out" by reasoning about `:632` — the one site that does
not refuse. Checking the site that actually throws would have inverted the
answer immediately.

## Design

`RmsNormRowKernel<Tin, Tw, Tout, Tres>` takes `const Tw* w`. `Load()` is already
overloaded for `float` and `__nv_bfloat16` (`cuda_ops.cu:51-52`), so the body is
unchanged. `LaunchRmsNormRes` and `LaunchRmsNorm` forward `Tw`;
`RmsNormKernelCuda` dispatches the gamma dtype separately and refuses an
unsupported one by name.

Neither vectorized fast path changes: `TryLaunchRmsNormDecodeFast` requires
`w.dtype == kBF16` and `TryLaunchRmsNormDecodeFastF32` requires `w.dtype == kF32`.

## Risks

- **A too-wide dtype is invisible to a token gate.** This change widens no
  buffer, but it also does not narrow the one that is too wide (#2488).
- **kF16.** `vt::RmsNorm` admits `IsFloat(weight.dtype)` (`ops.cpp:1030`, `:24`)
  and CPU takes a kF16 gamma; CUDA refuses it. That refusal is **not new** — the
  old equality check refused it too, since `x` is only ever f32 or bf16 — but it
  is a recorded divergence now, with a test pinning it (#2503).
- **Blast radius.** The bf16-activation/f32-gamma pairing previously threw and
  now runs, reachable from `glm4.cpp:173`, `gemma3.cpp:252`,
  `gemma2.cpp:259`/`:281`, `gemma.cpp:162`, `deepseek_v2.cpp:509` for any GGUF
  keeping F32 gammas, with the 2.41x fast kernel silently not engaging (#2503).

## Tests

`tests/vt/test_ops_rmsnorm_weight_dtype.cpp` covers three things:

1. **The production pairing** — f32 activation, bf16 gamma, `Dh=256`,
   instantiating `RmsNormRowKernel<float, __nv_bfloat16, __nv_bfloat16, float>`,
   which is what `:705` issues. CPU against an independently recomputed
   reference, plus a load-bearing-gamma check so the tolerance cannot be
   vacuous, plus CPU-vs-CUDA agreement.
2. **The newly admitted pairing** — bf16 activation, f32 gamma (#2503's blast
   radius).
3. **The kF16 divergence** — CPU accepts, CUDA throws. This exists so the
   `default:` arm cannot be replaced by a fall-through without a test going red.
   It asserts the dispatcher's own message rather than merely that something
   threw: a bare `CHECK_THROWS` would pass on a mutated build whose 256-byte
   over-read happened to fault, because `Check(cudaGetLastError(), ...)` would
   throw and the case could not tell the two apart.

**"Pinned by a test" is not "pinned in CI".** Every CUDA arm above sits behind a
`[SKIP]` when no device is present, and no CI lane here runs GPU tests. In CI
today the `default:` arm and the production instantiation are held by nothing
that executes; the device evidence comes from a leased run and is recorded as
such. That is a property of this repository's lanes, not of this change, and it
is listed under `## Owed` rather than presented as coverage.

**Red-before-green, stated as it was taken:** the pre-fix dispatcher refused any
`w.dtype != x.dtype`, so case 1 could not have run. That refusal was observed as
a server 500 from #2477; it was **not** captured by running this file against a
pre-fix binary. The transition therefore shows the pairing is now accepted, and
is not evidence about which pairing was at fault.

## Gates

- `tests/vt/test_ops_rmsnorm*` focused, then the full gate.
- End to end: `--device cuda` on the released UD-IQ1_S artifact.

## The precondition that refused a good tree

Run 1 staged the artifact in 1766 s and refused it:
`FIX IN SOURCE: Tw=1 dispatch=3 old_check=2 test=1`. `old_check` grepped the bare
phrase `weight dtype must match x`, which also appears at `cuda_ops.cu:578`
(`rmsnorm_quant_fp8`) and `:3689` (`fused_chain`). The RmsNorm count was 0, as
intended.

A grep precondition also cannot detect the defect that mattered: the same tree
failed `cuda-fat-build` on
`cuda_ops.cu:485: error: macro "VT_CHECK" passed 3 arguments, but takes just 2`,
and every `FIX IN SOURCE` grep passed on it, because the strings they look for
were present. **Only a compiler can say a change compiles.**

Both failures are kept for their direction. Too broad refuses a good tree and
shows up at once; too narrow passes a bad tree and every arm below it then
measures something other than what it names.

## Evidence

`thor:gpu0` (NVIDIA Thor, `compute_cap 11.0`, driver 595.78, CUDA 13.0.88),
released UD-IQ1_S artifact (72,546,461,344 bytes, shard-1 sha256
`88a1420825a9…`, verified before any arm ran). Binary sha256 `70e522df28d7…`
built for `sm_110` from `e934fb002` (1662 s, `cuda_libs=2`, `cu_objects=41`).

**The three cases above have run on a device**, at this head, on `thor:gpu0`
with the CUDA architecture PROBED from the device rather than guessed (`110`):

```
TEST SOURCE: case1=1 case2=1 case3=1 prodH256=1
UNIT rc=0 assertions=9 skip_lines=0
UNIT CUDA ARMS EXERCISED: YES
[doctest] test cases: 5 | 5 passed | 0 failed | 0 skipped
[doctest] assertions: 9 | 9 passed | 0 failed |
```

Nine is the exact expected count (2 + 2 + 1 + 2 + 2), which is what makes
`0 skipped` mean something. The kF16 case threw the message it claims to pin:
`vt: cuda rmsnorm: unsupported weight dtype (f32/bf16 only), got f16`.

An earlier device run reported `assertions=4`; that was the PREVIOUS revision of
the test, which pinned only the bf16-activation/f32-gamma pairing.

| arm | `CUDA_LAUNCH_BLOCKING` | outcome | wall | IMA | refusal |
|---|---|---|---|---|---|
| ARM FIX | 0 | HTTP 500, #2476 | 237 s | 3 | **0** |
| ARM FIXLB | 1 | HTTP 200, 8 tokens **(tokens WRONG, #2496)** | 200 s | 0 | **0** |
| ARM SAN | 0 + memcheck | #2476, localised | 253 s | yes | 0 |

`rmsnorm-dtype refusal present: 0` in both end-to-end arms is this change's own
result. Host `sys_used` peaked at 40.0 / 42.5 GiB, so the n-gram table stayed
quantized.

## What this did NOT fix

```
GPU : 11751 271 271 271 271 271 0 0        " Paris\n\n\n\n\n\n\n\n\n\n!!"
CPU : 11751 13 15767 411 2029 11 1092 369  " Paris. Given this fact, what is"
```

Token 0 agrees; tokens 1-7 do not. That is
[#2496](https://github.com/mudler/vllm.cpp/issues/2496). The CPU sequence is the
control recorded by the previous wave and was **not** re-measured here.

**#2496 and #2476 are TWO defects, and the tempting shared-cause reading is
DISPROVED.** I proposed that one out-of-bounds read explained both, faulting when
the overrun hit an unmapped page and returning garbage when it hit a mapped one.
The TOKENDIV wave measured #2496's serialized token sequence as **bit-stable
across builds** — byte-identical to a run of a different binary on a different
tree. A single defect cannot be both bit-reproducible across builds and
suppressible by serialisation. That bit-identity also rules out the whole
async-copy and ordering class as a cause of #2496.

#2496 is therefore **deterministic and structural, not timing-dependent**, and
its cause is unknown. Any candidate must explain a correct token 0 **and** a
bit-stable wrong decode, which points at an extent depending on **token count**
(prefill `T=5` versus decode `T=1`) rather than on timing. The GDN QKVZ over-read
stays the leading candidate on those grounds, and the GDNGEMM wave is testing it
by printing operand extents at both token counts.

**#2488 is excluded as a cause of #2496**: `q_f32` has no device branch and no
prefill/decode fork, so it was present on the CORRECT CPU run too. #2488 remains
the cause of #2477. Both statements are true and they are about different
defects.

[#2476](https://github.com/mudler/vllm.cpp/issues/2476) did not dissolve.
`compute-sanitizer memcheck` localised it to a cuBLASLt GEMM reading out of
bounds (`nvjet_sm110_tst_512x8_64x3_2x1_v_bz_TNT`) issued by `vt::MatmulBT` from
`MatmulBf16D` inside `ProjectGdnQkvz`, under `Qwen4ExpTextModelForward`.

### What the ordering evidence does and does not support

Supported: on the **fixed** binary, #2476 occurs with launches asynchronous and
does not occur with `CUDA_LAUNCH_BLOCKING=1`; #2477's refusal is absent from
both. On the **pre-fix** binary — a different tree, measured by the wave that
filed #2476 — the async arm faulted without reaching the refusal, so the
faulting work was enqueued before `:705`.

Not supported, and previously overclaimed here: that the fault is "a race rather
than a fixed bad index" — a use-after-free, a premature host free and a missing
stream sync each have a fixed index and each is suppressed by serialisation; and
that it is in "layers 0-2" — the sanitizer names the op and the call chain, not a
layer index, and the sound bound is only "work enqueued before `:705`".

## Upstream citations are OFF-PIN

The vLLM paths cited here are read at `origin/main` `25efcfa788`. The project pin
is `5559679229`, which has **no** `vllm/models/qwen4_exp/` at all — vLLM landed
this architecture after the pin. `docs/FEATURES.md:128` still says vLLM
implements it at no revision.
[#2502](https://github.com/mudler/vllm.cpp/issues/2502) is reconciling that.
Every vLLM `file:line` in this spec is a forward reference and is not gateable
against the pin.

## Stop conditions

A further wall after #2477 clears is a reportable result, not a failure.

## Owed

- [#2488](https://github.com/mudler/vllm.cpp/issues/2488) — **the actual cause.**
  `qsa_block.cpp:694-696` widens `q` and the output gate to f32 where vLLM keeps
  the bf16 model dtype. Blocked on `AttnGateSplit`'s shared f32 signature.
- ~~[#2503](https://github.com/mudler/vllm.cpp/issues/2503)~~ and
  ~~[#2492](https://github.com/mudler/vllm.cpp/issues/2492)~~ — CLOSED by wave
  RMSTWINS, [`rmsnorm-gamma-dtype-twins.md`](rmsnorm-gamma-dtype-twins.md). Both
  twins now take their own `Tw`; kF16 is served rather than recorded; and the
  blast radius #2503 asserted was MEASURED AND REFUTED — none of the five
  architectures it named has a GGUF arm at all. That wave's own `## Owed` carries
  what it did not close: [#2542](https://github.com/mudler/vllm.cpp/issues/2542)
  (the kF16 ACTIVATION, the same class one operand over) and
  [#2543](https://github.com/mudler/vllm.cpp/issues/2543) (`RmsNormPlusAddRocm`,
  a third instance of the weld in the same uncompilable file).
- [#2476](https://github.com/mudler/vllm.cpp/issues/2476), localised, still
  blocking the gate.
- [#2496](https://github.com/mudler/vllm.cpp/issues/2496), the token divergence —
  a SECOND defect, deterministic and structural, cause unknown. Not this change's
  and not #2476's.
- A CI lane that executes GPU tests. Until one exists, every `[SKIP]`ped CUDA arm
  in this suite is documentation of intent rather than an executing gate.

## Now

`ACTIVE`. The row's CUDA forward is unblocked at this wall and stops at #2476.
