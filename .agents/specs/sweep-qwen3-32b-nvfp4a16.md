# SPIKE: Breadth-Sweep Quant Bring-Up — Qwen3-32B-NVFP4A16 (`Qwen3ForCausalLM` + compressed-tensors **NVFP4A16 / W4A16**)

Rank **3** of the ranked sweep queue ([breadth-sweep-plan.md](breadth-sweep-plan.md) §B.3).
Owner: `CLAIM-QUANT-NVFP4-CT-W4A16`. Rows: `QUANT-NVFP4-CT-W4A16` (new, the
owning row) and `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (extended — same
architecture, new scheme).

Pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; dgx oracle `~/venvs/vllm-oracle`
= vLLM **0.25.0**. Base `aa65ce7`.

---

## 0. Purpose — why this row, and what it isolates

This row serves two user priorities at once: **#2 "models vLLM supports"** and
**#4 "more quants"**. Its value is that it is a *controlled experiment* in
**quantization-scheme additivity**:

- the dense `Qwen3ForCausalLM` forward is **already DONE and token-exact**
  (0.6B + 4B), so the transformer is a fixed, known-good baseline;
- the ONLY variable this checkpoint introduces is the **storage/compute scheme
  of the Linear weights**;
- it simultaneously scales the dense path to **64 layers** (hidden 5120, 64 Q
  heads / 8 KV heads, intermediate 25600) — a size-scale check for free.

So a pass is evidence a new quantization scheme is additive; a failure localizes
to the loader or the W4A16 GEMM and **never** to the transformer. That is a
sharper instrument than any of the model rows around it.

## 1. CHECKPOINT REALITY — premise verified BEFORE implementation

The OPT row taught that `breadth-sweep-plan.md` §B.1's "present" column is not
trustworthy (it claimed OPT present when only a 36 KB `config.json` existed).
Verified on dgx **2026-07-21, BEFORE any code**:

| Claim (§B.1) | Reality | Verdict |
|---|---|---|
| `RedHatAI--Qwen3-32B-NVFP4A16` present | snapshot `c572e992…` complete: 5 real shards totalling **20.6 GB** + index + full tokenizer; **1603 tensors** | ✅ **plan was CORRECT** |
| `architectures: Qwen3ForCausalLM`, 64L dense | confirmed | ✅ |
| compressed-tensors NVFP4A16 (W4A16) | confirmed: `format: nvfp4-pack-quantized`, `group_size 16`, `num_bits 4`, `type float`, `strategy tensor_group`, `symmetric`, **`input_activations: null`**, `ignore: ["lm_head"]` | ✅ |

The oracle also **loads and generates coherently** from it (see §6), so the row
was fully unblocked. `df -h /` = 193 GiB free; no download needed. §B.1 of the
sweep plan is annotated with this verification rather than corrected.

## 2. ON-DISK LAYOUT — measured, not assumed

Our existing NVFP4 assets are **modelopt**-flavoured (35B) or **compressed-tensors
W4A4** (27B). compressed-tensors has its own discovery and naming, so the layout
was read off the checkpoint rather than assumed:

```
<proj>.weight_packed        U8       [N, K/2]    two E2M1 nibbles per byte
<proj>.weight_scale         F8_E4M3  [N, K/16]   one fp8 scale per 16 elements
<proj>.weight_global_scale  F32      [1]         per-tensor DIVISOR
```
and, decisively, **no `<proj>.input_global_scale` anywhere** — that absence is
what makes this W4A16 rather than the 27B's W4A4.

Not quantized (no config group targets them / explicit `ignore`):
`model.embed_tokens.weight` BF16, all norms BF16, **`lm_head.weight` BF16** and
**untied** (`tie_word_embeddings: false` — unlike Qwen3-0.6B).

**Fused-shard global scales are bit-identical within every group** (measured at
layers 0/31/63: q=k=v, gate=up), so vLLM's lossy `max()` collapse is exactly
lossless on this checkpoint. The `max()` is implemented anyway, for faithfulness.

## 3. What vLLM ACTUALLY does — source map + OBSERVED dispatch

### 3a. Scheme selection (source)
`_is_nvfp4_format` (`compressed_tensors.py:401-417`) requires all of
`strategy==tensor_group`, `symmetric`, `group_size==16`, `type==FLOAT`,
`num_bits==4` — this config satisfies every one. `_get_scheme_from_parts:684`
checks NVFP4 **first** (`:696`), and with `input_quant is None` returns
**`CompressedTensorsW4A4Fp4(use_a16=True)`** (`:698`).

> **KEY FINDING: there is NO `CompressedTensorsW4A16Fp4` class upstream.** W4A16
> is the *same* W4A4 scheme class with an `a16` flag. Anyone porting this by
> searching for a "W4A16 scheme" will not find one. Also note `format:
> "nvfp4-pack-quantized"` is effectively **inert** on this path — selection is
> driven purely by the `weights` QuantizationArgs, because `:696` precedes the
> `pack_quantized` format check at `:735-736`.

### 3b. Kernel selection (source) — a16 FORCES Marlin
`init_nvfp4_linear_kernel` (`kernels/linear/__init__.py:842`), decisive lines
**`:879-881`**:
```python
elif linear_backend == "auto" and use_a16:
    force_kernel = MarlinNvFp4LinearKernel  # "Force a16 (Marlin) when running weight-only quantization."
```
This **bypasses the capability registry entirely** (`_POSSIBLE_NVFP4_KERNELS`,
`:407-424`, is never consulted). The cutlass/flashinfer alternatives are
unreachable for W4A16 — and could not serve it anyway, since they are W4A4
kernels needing a quantized activation.

### 3c. What vLLM OBSERVABLY dispatched on sm_121 (the ground truth)
Per the trace-the-execution directive, this was **read off the oracle run**, not
inferred:
```
INFO  [__init__.py:929] Using MarlinNvFp4LinearKernel for NVFP4 GEMM
WARNING [marlin.py:34] Your GPU does not have native support for FP4 computation
        but FP4 quantization is being used. Weight-only FP4 compression will be
        used leveraging the Marlin kernel.
INFO  [core.py:114] ... quantization=compressed-tensors ...
```
So: **Marlin, unconditionally**. No nsys was needed — vLLM names the selected
linear kernel in its own startup log, which is a stronger and cheaper signal
than a kernel-name diff.

### 3d. Scale/global-scale arithmetic (source)
- scheme: `weight_global_scale = 1.0 / layer.weight_global_scale.max()`
  (`compressed_tensors_w4a4_nvfp4.py:111-114`; warning at `:101-108` when the
  fused shards disagree). CT stores a **divisor**.
- marlin prep: `prepare_fp4_layer_for_marlin` (`marlin_utils_fp4.py:221-306`) —
  int32 view + transpose `:248`, stock `ops.gptq_marlin_repack` `:252-259` (there
  is **no NVFP4-specific repack op**), `marlin_permute_scales` `:273-279`,
  `nvfp4_marlin_process_scales` `:281-284` (→`:61-122`: the `[0,2,1,3]`
  interleave, the `scale_factor` power-of-2, the `<<1` E4M3→S0E5M3 trick),
  `nvfp4_marlin_process_global_scale` `:142-154` (folds `2^(126-7)` for bf16).
- GEMM: `ops.marlin_gemm` → `csrc/libtorch_stable/quantization/marlin/marlin.cu:545`
  (scale-type resolution `:600-611`; **group_size is NOT passed** — it is derived
  as `size_k / b_scales.size(0)` `:726-727`). Global scale is a scalar epilogue
  multiply, `marlin_template.h:1655-1657`.

Our vendored, already-validated lift of exactly these primitives is
`include/vt/cuda/marlin_repack.h` + `vt::MoeGroupedGemmNvfp4Marlin`.

## 4. ADDITIVITY LEDGER — which seams HELD, which LEAKED

### (a) HELD — reused with no change
- **The whole transformer.** Forward body, decoder layer, norms, RoPE, q/k-norm,
  paged FA2, KV spec, runner, sampler, registry, fusion catalog: untouched.
- **The compute kernel.** `vt::MoeGroupedGemmNvfp4Marlin` (num_experts=1) is
  already the validated 35B W4A16 dense GEMM. **ZERO new kernel code** — the
  single biggest additivity result of this row. The scheme that vLLM forces
  (Marlin) is the scheme we already had.
- **`Nvfp4Weight`.** Already carried `alpha`/`IsTrueW4A4()` precisely to separate
  W4A16 from W4A4; no struct change was needed for the semantics.
- **The loader helper family.** `dense_weight_loaders.h` took the new helpers
  append-only, exactly as the OPT row appended `LoadMergedBf16Vector`.

### (b) LEAKED — shared code that had to be TOUCHED
1. **`dense_attn_block.h` needed a layering split.** The NVFP4 helpers must sit
   *beneath* `AttnBlock` (which calls them) but *above* `Dev`/`DBuf` (which they
   use), and all three lived in one header → include cycle. Fixed by relocating
   `Dev`/`MakeTensor`/`Reshape`/`DevicePoolPolicy`/`DBuf` **verbatim** into the
   new `dense_device_glue.h`, same namespace. Pure relocation, byte-identical,
   zero consumer changes. *A genuine seam gap: the device glue was not factored
   finely enough for a second consumer family.*
2. **`Qwen3DenseAttnWeights` / `Qwen3DenseMlpWeights` gained fp4 fields.** Two
   parallel weight representations in one struct, discriminated by `Empty()`.
   Mirrors `qwen3_5_weights.h:211` (`q_proj_fp4` beside `q_proj`), so this is
   the tree's established idiom — but it is still additive-by-widening, not
   additive-by-composition.
3. **Two forward dispatch branches** (`AttnBlock` qkv+o_proj, `MlpBlock`).

### (c) LEAKED — the standing architectural gap this row re-confirms
**There is still NO `QuantizationConfig` / `LinearMethod` abstraction.** vLLM
resolves a scheme from `config.json` through a registry
(`quant_method` → config class → `get_scheme()` per layer → a kernel registry).
We instead **probe tensor names** (`has(proj + ".weight_packed")`) inside each
model's loader, exactly as the 27B and 35B do. That worked here — the probe is
unambiguous and the W4A16-vs-W4A4 discriminator (`input_global_scale` presence)
is genuinely on disk — but it means:
- every new model must re-implement the probe in its own loader;
- `quantization_config` in `config.json` is never parsed or validated;
- a checkpoint whose config and tensors disagree is not detected.

This is the **next quant-additivity seam**, and it is now the load-bearing one:
the kernel layer proved additive, the config/dispatch layer did not. Recorded as
a follow-up, deliberately NOT done in this row (it is a cross-cutting refactor
touching the 27B/35B loaders, and this row's whole value is a controlled
experiment).

### (d) Honest verdict vs the PR-#4 standard
New files: 3 (`dense_nvfp4_gemm.h`, 2 tests) + 1 pure-relocation header + 1
oracle script + goldens. Shared touches: 5 files, of which one is a verbatim
move, two are the model's own weight struct/loader, and two are dispatch
branches. **No runner, sampler, kernel, registry or engine edit.** Comfortably
passes the PR-#4 test for a new *quantization scheme* — the scatter PR #4
exhibited does not recur.

## 5. Design decisions

- **D1 — W4A16 helpers are a SEPARATE header, not an extraction from
  `qwen3_5.cpp`.** `qwen3_5.cpp` retains its own copies. Rationale: the full
  dispatch tree there is entangled with the true-W4A4 cutlass path (swizzled
  blockscales, device alpha, `ScaledFp4Quant`), and extracting it would put the
  27B/35B hot path at risk for zero benefit to this row. This *knowingly*
  duplicates ~200 lines, and follows the precedent set when `dense_attn_block.h`
  duplicated the same device glue rather than removing it from `qwen3_5.cpp`.
  Recorded in §Risks/decisions as debt, with the unification scoped as its own
  gate-model-touching change.
- **D2 — merged qkv is ALWAYS merged on the fp4 path.** vLLM owns one merged
  `qkv_proj` parameter and repacks it whole into one Marlin operand; a
  Marlin-interleaved operand is not row-addressable, so the 3-shard A/B that
  exists on the BF16 path has no fp4 analog. `VT_QWEN3_ATTN_F32=1` is rejected on
  the quantized path (Marlin requires a bf16 activation).
- **D3 — gate/up stay SEPARATE operands at load, fused at repack.** Lets the
  forward choose vLLM's merged `size_n=2I` layout (default) or the split
  two-GEMM A/B from the same loaded bytes, mirroring the 35B shared expert.
- **D4 — `max()`-then-reciprocate on merged globals.** vLLM's arithmetic, done
  over the on-disk divisors so no intermediate ULP is lost.

## 6. Gate selection — MEASURED

Per [[near-tie-distributional-gate]] the bar is chosen by first measuring
**vLLM's own** greedy determinism on THIS checkpoint.
`scripts/qwen3-32b-nvfp4a16-oracle-capture.py --runs 5` (per-prompt, batch=1 —
the gate's regime):

```
=== vLLM SELF-DETERMINISM report: RedHatAI/Qwen3-32B-NVFP4A16  N=6 T=16 K=5 ===
  prompt[0..5] deterministic (1 sequence over 5 runs)
=== ALL DETERMINISTIC -> STRICT token-exact gate  (K=5; 0 multi-member cells) ===
```

⇒ **STRICT token-exact** is the bar. No near-tie band is used at all. (Consistent
with OPT-125m and Qwen3-Coder, both deterministic; the non-determinism seen on
tiny dense models is a small-model near-tie phenomenon, not a general one.)

vLLM's generations are coherent (`"The capital of France is"` → `" Paris. This
is a fact that is widely known and accepted. However, when"`), confirming the
oracle arm is sound. **Prompt tokenizations are committed as goldens** — Qwen3
prepends no BOS, and on the OPT row a BOS bug scored 0/6 while emitting fluent
English, with the tokenization golden the thing that caught it in one run.

## 6b. GATE — strict scored 4/6; the TEACHER-FORCING isolation then SETTLED it

Run on dgx (clean `121a` build, `flock /tmp/gpu`, goldens md5-verified before AND
after the `git archive` transfer):

```
Qwen3-32B-NVFP4A16 W4A16 execution counters:
    marlin_gemms=18432  fused_gate_up=6144  fallback_gemms=0
Qwen3-32B-NVFP4A16 STRICT correctness gate: 4/6 prompts token-exact (67/96 tokens)
DIVERGENCE prompt[2] first bad tok=1  ours=1874  vLLM=883    (" group" vs " man")
DIVERGENCE prompt[5] first bad tok=1  ours=17788 vLLM=17330  (" moon"  vs " Moon")
```

The gate was NOT loosened on that evidence. The named next measurement — the
ratified teacher-forcing isolation — was then RUN (§6c), and it proved the
residual is bf16 near-tie drift that **vLLM itself exhibits**, not a defect of
ours. The row now closes under the ratified near-tie-robust gate WITH the
per-position nats evidence committed. See §6c for the numbers and §6d for the
final gate table.

### What the failure signature rules OUT
- **Not the loader / scales / layout.** A wrong global scale, a divisor-vs-
  multiplier flip, a wrong group size, or a mis-stacked merged operand produces
  garbage or systematic drift. Instead **4/6 prompts are exact for all 16
  tokens**, and — decisively — **the prefill argmax (token 0) is correct on ALL
  SIX prompts**. The dequantized weights are right.
- **Not tokenization.** All six prompt-tokenization goldens match (the OPT BOS
  class of bug is excluded by construction).
- **Not a silent fallback.** `fallback_gemms=0` with 18432 Marlin GEMMs and 6144
  fused gate_up launches: the new W4A16 Marlin path is what ran, everywhere.
- **Not vLLM non-determinism.** vLLM is deterministic here (0 multi-valued
  cells), so this is a real forward difference on our side.

### What it points AT
Both divergences are at **token 1 — the FIRST DECODE step (M=1)**, while every
prefill (large M) is exact. That is a decode-shape-specific numerical
difference, and there is a concrete structural reason for one:

> **We route a DENSE W4A16 linear through the MoE grouped Marlin entry point
> (`vt::MoeGroupedGemmNvfp4Marlin` with `num_experts=1`), whereas vLLM routes it
> through the DENSE entry point (`apply_fp4_marlin_linear` → `ops.marlin_gemm`,
> `csrc/.../marlin/marlin.cu:545`).** These are two different kernels in vLLM's
> own csrc with different tile/split-K schedules, hence different f32
> accumulation order. At large M the difference is below the bf16 rounding
> boundary; at M=1 it re-resolves semantic near-ties (" Moon"→" moon").

This is exactly the question the row was asked to answer — *"determine whether
the DENSE W4A16 path needs a different Marlin entry point"* — and the measured
answer is **YES**. The 35B passes 315/315 with the same helper because its
W4A16 Marlin sinks (shared expert, lm_head) sit beside a vLLM arm that is itself
MoE-Marlin; this is the first row where vLLM takes the *dense* Marlin entry.

### Isolation A/B — RUN, and it REFRAMED the diagnosis
`VT_NVFP4_MARLIN=0` re-ran the identical binary through the naive
redundant-dequant `vt::MatmulNvfp4` instead of Marlin:

```
marlin_gemms=0  fused_gate_up=0  fallback_gemms=30720
STRICT correctness gate: 3/6 prompts token-exact (56/96 tokens)
DIVERGENCE prompt[2] tok=1  ours=1874  vLLM=883     <- SAME as Marlin
DIVERGENCE prompt[5] tok=1  ours=17788 vLLM=17330   <- SAME as Marlin
DIVERGENCE prompt[3] tok=5  ours=17    vLLM=16      <- NEW (Marlin had it exact)
```

Two conclusions, both useful:

1. **Marlin is the BETTER-matching kernel (4/6 vs 3/6), as it should be** — it is
   what vLLM runs, including vLLM's own lossy S0E5M3 scale re-encode and
   `2^(126-7)` exponent-bias fold. The "mathematically purer" naive dequant is
   *further* from the oracle, which is a good sign that our Marlin scale
   processing is faithful. This weakens the "dense-vs-grouped Marlin entry
   point" hypothesis as the primary cause.
2. **The two token-1 divergences are INVARIANT across both of our GEMM kernels.**
   A difference that survives swapping the entire quantized GEMM is very
   unlikely to originate in the quantized GEMM.

### Hypothesis at the time: the PRE-EXISTING dense-forward bf16 drift
The dense `Qwen3ForCausalLM` forward is **not** bit-identical to vLLM — it closes
under the ratified **near-tie-robust** gate, not a strict one
(`MODEL-TEXT-qwen3-qwen3-for-causal-lm`: 0.6B strict 10–12/16 + near-tie band
4–6/16; 4B strict 10–11/16 + band 5–6/16). This row inherits that forward
unchanged and then runs it **64 layers deep**, where the same per-op bf16
rounding drift accumulates further. 2/6 prompts landing on semantic near-ties
(`" Moon"`→`" moon"`, `" man"`→`" group"`) is squarely in family with that
documented baseline behavior — and both divergences are at the *first decode
step*, exactly the prefill-vs-decode near-tie boundary the dense row
characterized at length.

**That was a hypothesis, not a conclusion** — and §6c is the measurement that
settled it.

## 6c. THE DECIDING MEASUREMENT — teacher-forcing, RUN 2026-07-21

`scripts/qwen3-32b-nvfp4a16-neartie-gap.py` (new; mirrors
`scripts/qwen3coder-neartie-gap.py`) feeds vLLM OUR exact generated sequence as
`prompt_token_ids` and reads `prompt_logprobs`, giving vLLM's OWN per-position
distribution on OUR prefix. Our ids come from the gate run under `VT_DUMP_IDS=1`.

A property of THIS row makes the measurement unusually sharp: **every prefill
argmax is exact on all six prompts**, so at the two ROOT divergences (token 1)
the prefix is **BIT-IDENTICAL** between us and vLLM. The gap there is therefore
not a prefix-drift artefact — it is a direct comparison of our decode-step logit
ordering against vLLM's at the same state.

```
=== teacher-forced near-tie gap: RedHatAI/Qwen3-32B-NVFP4A16 (OUR prefix) ===
  29 token-divergent positions; max gap 0.0625 nats (worst = prompt[2] tok1)
  28 of 29 positions: gap EXACTLY 0.000000 nats (our token IS vLLM's argmax)
```

The two ROOT divergences:

| position | ours | vLLM greedy | vLLM **teacher-forced argmax** | gap |
|---|---|---|---|---|
| prompt[5] tok1 | 17788 `" moon"` | 17330 `" Moon"` | **17788 = OURS** | **0.000000 nats** |
| prompt[2] tok1 | 1874 `" group"` | 883 `" man"` | 883 | **0.0625 nats** |

- **prompt[5] is an EXACT bf16 TIE.** vLLM's own logprobs are bit-identical for
  both tokens (`-0.727154` each), and vLLM's teacher-forced argmax is **OUR**
  token while its incremental greedy decode chose the other. **vLLM contradicts
  ITSELF at this position** — exactly the prefill-argmax-vs-incremental-decode
  disagreement the dense row characterized. There is no "correct" token to
  match here.
- **prompt[2] gaps 0.0625 nats**, and a batch-composition sweep shows **vLLM's
  OWN separation at that same position moves by 0.125 nats** (0.1875 alone vs
  0.0625 batched with 3 fillers — same model, same prefix, different GEMM
  tiling/scheduling). Our gap is therefore SMALLER than vLLM's own numerical
  jitter at that position. It is also smaller than the **0.25-nat** gap already
  ratified on the **UNQUANTIZED** dense Qwen3-4B row.
- **All 27 remaining divergences are downstream CASCADE** from those two flips
  (once token 1 differs the sequences are different texts), and every one has
  gap **0.000000** — i.e. our forward is emitting vLLM's own argmax at each.

### Verdict — NOT a W4A16 defect; PRE-EXISTING dense bf16 near-tie drift
The quant path is exonerated on four independent counts: the CPU forward proof is
bit-identical (max |Δlogit| = 0), `fallback_gemms=0` against 18432 Marlin GEMMs,
the divergences are INVARIANT across both of our quantized GEMMs
(`VT_NVFP4_MARLIN=0` reproduces them), and now every teacher-forced gap is
≤ 0.0625 nats with 28/29 exactly zero. The signature is identical in character to
the dense `Qwen3ForCausalLM` row's own measured baseline (0.6B: 60 divergent
positions, **all** gap 0.0; 4B: max gap 0.25), now run 64 layers deep.

**The finding belongs against the DENSE row (`MODEL-TEXT-qwen3-qwen3-for-causal-lm`),
not the quant row.** This row's new code is correct.

### Gate disposition — ratified near-tie-robust, evidence-backed
`tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp` now uses the SAME
ratified gate as the dense Qwen3 and Qwen3-Coder rows: PASS iff our token is
within `kNearTieMnats = 500` (0.5 nats) of vLLM's teacher-forced argmax given OUR
prefix, with the STRICT count still reported. Measured worst case **62 mnats —
8× inside the bar**. The band admits ONLY what vLLM's own logits cannot separate;
a larger gap, or our token outside vLLM's top-K, still FAILS. A hard `REQUIRE`
anchor asserts the engine still emits the exact `our_ids.npy` sequence the gaps
describe, so the band can never silently cover tokens we no longer produce.

```
Qwen3-32B-NVFP4A16 correctness gate: 6/6 prompts PASS
  (STRICT token-exact: 4/6 = 67/96 tokens; near-tie-band only: 2/6;
   max teacher-forced gap 0.062 nats @ prompt[2] tok=1 (bar 0.5);
   0 forward-divergent; vLLM self-determinism: 0 multi-valued cells)
Qwen3-32B-NVFP4A16 W4A16 execution counters:
  marlin_gemms=18432 fused_gate_up=6144 fallback_gemms=0
[doctest] assertions: 142 | 142 passed | 0 failed
```

## 6d. Full gate evidence (dgx, clean `121a` build, `flock /tmp/gpu`)

| Gate | Result |
|---|---|
| CUDA `-Werror` clean build | **0 warnings / 0 errors** |
| Loader gate `test_qwen3_32b_nvfp4a16_load` | **5144/5144 PASS** — 5 shards / 1603 tensors all mapped, BF16 arms provably empty, `alpha==0` everywhere, zero `input_global_scale`, L0 qkv divisors 6208/6208/6208 → max 6208 → `scale2` 1.61082e-4, NO leftover tensor |
| Forward doctest (CPU synthetic) | **PASS**, NVFP4 vs BF16-on-dequantized **max abs Δlogit = 0** (bit-identical), counters 10/10 |
| SACRED engine gate (ratified near-tie-robust) | **6/6 prompts PASS — STRICT token-exact 4/6 (67/96 tokens) + near-tie band 2/6; max teacher-forced gap 0.062 nats vs the 0.5-nat bar; 0 forward-divergent**; 142/142 assertions (incl. the 96 hard anchor REQUIREs) |
| TEACHER-FORCING isolation (the deciding measurement) | **29 divergent positions, max gap 0.0625 nats, 28/29 EXACTLY 0.0** — prompt[5] tok1 is an EXACT tie where vLLM's teacher-forced argmax is OURS (vLLM contradicts its own greedy); prompt[2] tok1 gaps 0.0625 while vLLM's OWN separation there moves 0.125 nats with batch composition |
| Execution counters (positive signal) | `marlin_gemms=18432`, `fused_gate_up=6144`, `fallback_gemms=0` — the W4A16 path PROVABLY ran |
| `VT_NVFP4_MARLIN=0` A/B | 3/6 strict (worse), SAME two divergences + one more |
| `test_ops_nvfp4_fp4` | 27002/27002 PASS |
| `test_ops_moe_grouped` | 146/146 PASS |
| `test_ops_moe_grouped_bf16` | 19/19 PASS |
| REGRESSION 27B `test_qwen27_paged_engine` | **235/235 UNCHANGED** |
| REGRESSION 35B `test_qwen36_paged_engine` | **315/315 UNCHANGED** |
| REGRESSION Qwen3-Coder | **138/138 UNCHANGED** |
| REGRESSION Qwen3-dense | **664/664 UNCHANGED** |
| REGRESSION OPT `test_opt_paged_engine` | **36/36 UNCHANGED** |
| `compute-sanitizer memcheck` (engine path) | **ERROR SUMMARY: 0 errors** |

### Disposition
**Correctness CLOSED** under the ratified near-tie-robust gate, with the
per-position nats evidence committed (`our_ids.npy` +
`neartie_gap_mnats.npy`). Row stays **`ACTIVE`** — never `DONE` — because **W5
SPEED was never started**, per the DONE = correctness + speed policy.

## 7. Key findings

1. **The premise held here** (unlike OPT): the checkpoint is genuinely complete.
   Verifying first cost minutes and is now a standing step.
2. **vLLM has no W4A16 NVFP4 scheme class** — it is `W4A4Fp4(use_a16=True)`.
3. **`use_a16` FORCES Marlin**, bypassing capability dispatch; on sm_121 there is
   no alternative, confirmed observationally.
4. **The kernel layer was fully additive** — the scheme vLLM forces is the one we
   already vendored for the 35B. Zero kernel work.
5. **The config/scheme-selection layer is the remaining gap** — still a
   tensor-name probe, still per-model.
6. **CPU proof is bit-exact**: the synthetic NVFP4 forward equals the BF16
   forward on the dequantized weights with **max |Δlogit| = 0**.
7. **The strict gate scored 4/6, and the teacher-forcing isolation PROVED why
   (§6c):** all 29 divergent positions gap ≤ **0.0625 nats**, **28/29 exactly
   0.0**. The root flip on prompt[5] is an EXACT bf16 tie where **vLLM's own
   teacher-forced argmax is OUR token** and vLLM disagrees with its own greedy
   decode. **This is the pre-existing dense-forward bf16 near-tie drift, not a
   W4A16 defect — and it is recorded against the DENSE row.** The gate closes
   under the ratified near-tie-robust bar with that evidence committed, NOT by
   assertion.
8. **A near-tie gap is only meaningful against the ORACLE'S OWN jitter.** vLLM's
   separation at prompt[2] tok1 moves **0.125 nats** purely from batch
   composition (0.1875 alone → 0.0625 batched). Measuring that turned an
   ambiguous 0.0625-nat gap into a decided one. Make this calibration a standing
   step whenever a near-tie gap is non-zero.
9. **Marlin matches vLLM better than a "purer" dequant does (4/6 vs 3/6 strict)** —
   evidence that mirroring vLLM's lossy S0E5M3 scale re-encode is correct, and a
   reminder that fidelity to the oracle beats numerical purity.
10. **The prefill-exact / decode-flip signature made the isolation SHARP.** With
   every prefill argmax exact, the prefix at the root divergences is
   bit-identical to vLLM's, so the teacher-forced gap is a direct single-position
   logit comparison rather than a prefix-drift artefact. Check for this property
   before interpreting a gap.

---

## 8. Structured spike contract (stable row `QUANT-NVFP4-CT-W4A16`)

The prose above (§0–§7) is the full spike; this restates it in the record-checker's structured fields.

### Scope
Add compressed-tensors **NVFP4A16 (W4A16 — NVFP4 weights, BF16 activations)** as a loadable + natively-computed quantization scheme on the ALREADY-DONE dense `Qwen3ForCausalLM` forward, proven end-to-end on `RedHatAI/Qwen3-32B-NVFP4A16` (64L dense, hidden 5120, GQA 64/8, head_dim 128, intermediate 25600, untied BF16 lm_head). In scope: the compressed-tensors `nvfp4-pack-quantized` loader (per-Linear scheme probe, divisor→reciprocal scale convention, merged-shard `max()` global-scale collapse), the W4A16 GEMM dispatcher (forced-Marlin mirroring vLLM, with a naive-dequant CUDA fallback and a host dequant reference), the fused merged `gate_up` Marlin layout, execution counters proving the path ran, the loader gate, the synthetic forward doctest, and the SACRED strict token-exact engine gate. Out of scope: a general `QuantizationConfig`/`LinearMethod` abstraction (recorded as the next seam, §4c); the true-W4A4 fp4-activation path (stays with the 27B); unifying this header with `qwen3_5.cpp`'s duplicate copies (decision D1); speed parity (W5, explicitly pending).

### Upstream chain
`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` @ `e24d1b24` — `from_config:227`, `_quantization_scheme_map_from_config:297,324-330`, `_is_nvfp4_format:401-417`, `get_scheme:805,823,841,850`, `_get_scheme_from_parts:684,696-698` (→ `CompressedTensorsW4A4Fp4(use_a16=True)`; there is NO separate W4A16 class), `should_ignore_layer`/`find_matched_target` (`utils.py:50,113`). Scheme `schemes/compressed_tensors_w4a4_nvfp4.py:29-32` (`init_nvfp4_linear_kernel(use_a16=...)`), `:38-93` create_weights (weight_packed `:53-63`, weight_global_scale `:66-70`, weight_scale `:73-84`, input_global_scale ONLY when `not use_a16` `:86-91`), `:95-141` process_weights_after_loading (`1.0/…max()` `:111-114`), `:143-149` apply. Kernel choice `vllm/model_executor/kernels/linear/__init__.py:842,879-881,884-892`; wrapper `kernels/linear/nvfp4/marlin.py:21-57`. Repack/scales `layers/quantization/utils/marlin_utils_fp4.py:34-35,61-122,142-154,157-218,221-306` (stock `ops.gptq_marlin_repack` `:252-259`, `marlin_permute_scales` `:273-279`). CUDA `csrc/libtorch_stable/quantization/marlin/marlin.cu:545,600-611,726-727,783-794,885`, `gptq_marlin_repack.cu:16,285`, `dequant.h:434-449,534`, `marlin_template.h:355-358,1655-1657`. Model `vllm/model_executor/models/qwen3.py:271-274` (`packed_modules_mapping`). Checkpoint `RedHatAI/Qwen3-32B-NVFP4A16/config.json`.

### Our baseline
The dense `Qwen3ForCausalLM` forward is DONE and token-exact (`MODEL-TEXT-qwen3-qwen3-for-causal-lm`, 0.6B + 4B gates PASS 16/16) — this row changes ONLY the Linear storage/compute scheme on it. Reused unchanged: `vt::MoeGroupedGemmNvfp4Marlin` + `include/vt/cuda/marlin_repack.h` (the vendored, bit-exact Marlin NVFP4 W4A16 lift already validated as `QUANT-NVFP4-MO-W4A16` on the 35B), `Nvfp4Weight` (`qwen3_5_weights.h:93-129`, whose `alpha`/`IsTrueW4A4()` already discriminate W4A16 from W4A4), `vllm::DequantNvfp4ToBf16` (`nvfp4_dequant.h:59`), `dense_weight_loaders.h`, `dense_attn_block.h::AttnBlock`, `device_pool.h`, the model-shape-agnostic runner, the fusion catalog. Nearest prior art: `QUANT-NVFP4-CT-W4A4` (27B — same compressed-tensors NAMES and divisor convention, but W4A4) and `QUANT-NVFP4-MO-W4A16` (35B — same W4A16 COMPUTE, but modelopt names).

### Port map
New: `include/vllm/model_executor/models/dense_nvfp4_gemm.h` (the W4A16 dispatcher `MatmulNvfp4W4A16D` + `ResidentNvfp4` + Marlin resident repack/align-cache/workspace + fused `GateUpFusedMarlinD` + `DequantNvfp4ToBLayout` CPU fallback + `Nvfp4W4A16Stats` counters); `include/vllm/model_executor/models/dense_device_glue.h` (PURE VERBATIM RELOCATION of `Dev`/`MakeTensor`/`Reshape`/`DevicePoolPolicy`/`DBuf` out of `dense_attn_block.h`, same namespace, to break the new include cycle); `scripts/qwen3-32b-nvfp4a16-oracle-capture.py`; `tests/vllm/models/test_qwen3_32b_nvfp4a16_{load,paged_engine}.cpp`; goldens `tests/parity/goldens/qwen3_32b_nvfp4a16_greedy/`. Touched: `dense_weight_loaders.h` (APPEND-ONLY `IsCtNvfp4Projection` / `LoadCtNvfp4W4A16` / `LoadMergedCtNvfp4W4A16` / `ReadCtF32Scalar`); `qwen3.h` (fp4 fields + `IsNvfp4()` on the attn/MLP weight structs); `qwen3_weights.cpp` (per-Linear `.weight_packed` probe selecting the fp4 arm); `qwen3.cpp::MlpBlock` + `dense_attn_block.h::AttnBlock` (two dispatch branches); `tests/CMakeLists.txt`. NOT touched: `qwen3_5.cpp`/`qwen3_5_dense_weights.cpp` (the 27B/35B hot path is byte-untouched by construction — see decision D1), any kernel, the runner, the sampler, the registry.

### Tests to port
(a) Loader gate `tests/vllm/models/test_qwen3_32b_nvfp4a16_load.cpp` — mirrors `tests/quantization/test_compressed_tensors.py` (scheme-selection + weight-shape assertions): all 1603 tensors mapped with correct dtypes/shapes and NO leftover, the BF16 arms provably EMPTY, `alpha == 0` / `IsTrueW4A4()` false on every operand, zero `input_global_scale` in the checkpoint, and the merged-qkv `scale2 == 1/max(divisor_q,k,v)` collapse. (b) Forward doctest `tests/vllm/models/test_qwen3_forward.cpp` (new case) — synthetic NVFP4 vs BF16-on-dequantized, checkpoint-free, plus counter assertions. (c) SACRED engine gate `tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp` — mirrors `tests/models/language/generation/test_common.py` greedy-vs-reference against the vLLM 0.25.0 oracle plus vLLM's committed prompt tokenizations, under the ratified near-tie-robust bar (`kNearTieMnats = 500`, identical to the dense Qwen3 and Qwen3-Coder gates), reporting the STRICT count alongside and hard-`REQUIRE`-anchoring our exact sequence to `our_ids.npy`. (d) TEACHER-FORCING isolation `scripts/qwen3-32b-nvfp4a16-neartie-gap.py` (mirrors `scripts/qwen3coder-neartie-gap.py`) + the `VT_DUMP_IDS=1` bootstrap in the gate — produces the `our_ids.npy` / `neartie_gap_mnats.npy` evidence that LICENSES the band. Existing suites re-run as regression: `test_ops_nvfp4_fp4`, `test_ops_moe_grouped`, `test_ops_moe_grouped_bf16`, `test_qwen27_paged_engine`, `test_qwen36_paged_engine`, `test_qwen3coder_paged_engine`, `test_qwen3_paged_engine`, `test_opt_paged_engine`.

### Gates
GATE SELECTION MEASURED, not assumed, in TWO stages. Stage 1: `scripts/qwen3-32b-nvfp4a16-oracle-capture.py --runs 5` reports ALL SIX prompts deterministic with **0 multi-valued (prompt,pos) cells**, so the gate was first run **STRICT token-exact** — and scored **4/6 / 67/96 tokens**. Stage 2 (the deciding one, §6c): the ratified TEACHER-FORCING isolation `scripts/qwen3-32b-nvfp4a16-neartie-gap.py` measured **all 29 divergent positions at ≤ 0.0625 nats, 28/29 EXACTLY 0.0**, with the prompt[5] root flip an EXACT bf16 tie at which **vLLM's own teacher-forced argmax is OUR token** (vLLM contradicts its own greedy decode) and the prompt[2] root flip at 0.0625 nats against **0.125 nats of vLLM's OWN batch-composition jitter** at that same position. ⇒ the honest bar is the **ratified near-tie-robust gate** (`kNearTieMnats = 500`), identical to the dense Qwen3 / Qwen3-Coder rows, with the per-position nats evidence COMMITTED (`our_ids.npy`, `neartie_gap_mnats.npy`) — the band admits only what vLLM's own logits cannot separate, and a hard anchor `REQUIRE` keeps it pinned to the exact sequence we emit. Required: dgx clean build `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_TRITON=ON -DCMAKE_CUDA_ARCHITECTURES=121a` with CUDA `-Werror` 0 warnings / 0 errors; the **6/6 near-tie-robust gate (0 forward-divergent, max gap 0.062 nats)**; the loader gate; the forward doctest; **POSITIVE EXECUTION SIGNAL** — `Nvfp4W4A16Stats.marlin_gemms > 0` AND `fused_gate_up > 0` (a passing gate alone does NOT prove the new path ran); **REGRESSION, non-negotiable and UNCHANGED**: 27B `test_qwen27_paged_engine` 235/235, 35B `test_qwen36_paged_engine` 315/315, Qwen3-Coder 6/6, Qwen3-dense 16/16, OPT `test_opt_paged_engine` 6/6; `test_ops_nvfp4_fp4` + `test_ops_moe_grouped` + `test_ops_moe_grouped_bf16` green; `compute-sanitizer memcheck` 0 on the engine path. W5 speed (every-axis vs graphed vLLM 0.25.0) is a SEPARATE gate and is explicitly PENDING — the row therefore lands `ACTIVE`, never `DONE`, per the DONE=correctness+speed policy.

### Dependencies
Checkpoint `RedHatAI/Qwen3-32B-NVFP4A16` present and complete on dgx (verified §1; 5 shards / 20.6 GB / 1603 tensors) — no download. Oracle vLLM 0.25.0 present and confirmed loading this checkpoint. Builds on delivered rows: `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (the token-exact dense forward this row perturbs), `QUANT-NVFP4-MO-W4A16` (the vendored Marlin W4A16 GEMM, reused verbatim — the reason there is no kernel work), `QUANT-NVFP4-CT-W4A4` (the compressed-tensors naming + divisor convention), `MODEL-FACTORY-registry`, `BACKEND-PLATFORM`. GB10-only (Marlin path is CUDA sm_121a); the loader gate and forward doctest run host-only. No multi-GPU, no new third-party dependency.

### Work breakdown
**W0 spike + records (LANDED 2026-07-21):** checkpoint premise verified BEFORE implementation (§1 — plan's §B.1 claim CORRECT here, unlike OPT); on-disk layout measured (§2); vLLM's scheme + kernel selection mapped to `file:line` AND confirmed OBSERVATIONALLY (§3 — `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`). **W1 layering (LANDED 2026-07-21):** `dense_device_glue.h` PURE VERBATIM RELOCATION of `Dev`/`DBuf`/`MakeTensor`/`Reshape`/`DevicePoolPolicy` out of `dense_attn_block.h` (same namespace ⇒ every `using namespace dense_attn;` consumer resolves unchanged and every dense forward stays byte-identical), breaking the cycle that the new NVFP4 header would otherwise create. **W2 loader (LANDED 2026-07-21):** append-only `LoadCtNvfp4W4A16` / `LoadMergedCtNvfp4W4A16` / `IsCtNvfp4Projection` in `dense_weight_loaders.h` + fp4 fields on `Qwen3DenseAttnWeights`/`Qwen3DenseMlpWeights` + the per-Linear `.weight_packed` probe in `qwen3_weights.cpp`; divisor→reciprocal convention, `max()`-then-reciprocate on merged qkv/gate_up, explicit REJECTION of a W4A4 checkpoint (`input_global_scale` present) so the weight-only path can never silently swallow one. **W3 forward (LANDED 2026-07-21):** `dense_nvfp4_gemm.h` W4A16 dispatcher mirroring vLLM's forced-Marlin selection + fused merged `gate_up` (vLLM's `size_n=2I` operand) with a split two-GEMM A/B via `MoeSiluMul`, wired at the two GEMM sites. `Nvfp4W4A16Stats` counters added so the path is OBSERVABLE. **CPU proof green:** the synthetic NVFP4-vs-BF16-on-dequantized doctest agrees with **max |Δlogit| = 0 (bit-identical)** and counters confirm 10/10 quantized GEMMs took the W4A16 dispatcher; host build `-Werror` clean. **W4 correctness gate (SACRED) — RUN 2026-07-21; strict scored 4/6 (§6b).** Gate selection measured DETERMINISTIC ⇒ strict was run FIRST; goldens + vLLM tokenizations captured and committed (md5-verified before AND after the `git archive` transfer — never rsync, which silently overwrote `tests/` goldens twice). RESULT: 4/6 prompts / 67/96 tokens token-exact, with the W4A16 path PROVEN exercised (`marlin_gemms=18432`, `fused_gate_up=6144`, `fallback_gemms=0`). Both divergences are at token 1 (the first DECODE step); every prefill argmax is exact; all six tokenizations match. `VT_NVFP4_MARLIN=0` A/B scores 3/6 and reproduces the SAME two divergences, so the residual is INVARIANT across both of our quantized GEMMs. **W4b DIAGNOSIS — the named teacher-forcing isolation RUN 2026-07-21 (§6c), and it SETTLED the question.** New `scripts/qwen3-32b-nvfp4a16-neartie-gap.py` + a `VT_DUMP_IDS=1` bootstrap in the gate: **all 29 divergent positions gap ≤ 0.0625 nats, 28/29 EXACTLY 0.0** (our token IS vLLM's own argmax given our prefix). prompt[5] tok1 is an EXACT bf16 tie (`" moon"`/`" Moon"` bit-identical logprobs) where vLLM's teacher-forced argmax is OURS and vLLM contradicts its own incremental greedy; prompt[2] tok1 gaps 0.0625 nats while vLLM's OWN separation there moves **0.125 nats** with batch composition alone (measured 0.1875 → 0.0625), and is smaller than the 0.25-nat gap already ratified on the UNQUANTIZED dense Qwen3-4B row. The remaining 27 are downstream cascade. **VERDICT: NOT a W4A16 defect — this is the pre-existing dense-forward bf16 near-tie drift, run 64 layers deep, and it is recorded against the DENSE row `MODEL-TEXT-qwen3-qwen3-for-causal-lm`.** The quant path is exonerated on four independent counts (bit-exact CPU proof, `fallback_gemms=0`, GEMM-invariance, ≤0.0625-nat gaps). Gate converted to the ratified near-tie-robust bar WITH the evidence committed (never by assertion): **6/6 PASS — strict 4/6 + band 2/6, max gap 0.062 nats, 0 forward-divergent, 142/142 assertions**. **W5 speed — NOT STARTED, explicitly PENDING;** the row stays `ACTIVE` until every-axis parity vs graphed vLLM 0.25.0 is measured. Expected shape: the 35B already showed the Marlin W4A16 dense GEMM beating the naive redundant-dequant kernel by a wide margin, but a 64-layer dense at small M is a different operating point and must be measured, not assumed.

### Risks/decisions
**D1 (accepted debt) — the W4A16 helpers are DUPLICATED, not extracted.** `qwen3_5.cpp` keeps its own copies of `ResidentNvfp4`/`MarlinDenseResident`/`DenseAlignCache`/`MatmulNvfp4MarlinD`/the fused pair. Extracting them would require untangling the true-W4A4 cutlass path (swizzled blockscale, device alpha, `ScaledFp4Quant`) that shares those helpers, putting the 27B 235/235 + 35B 315/315 gates at risk for no benefit to this row. This follows the tree's own precedent (`dense_attn_block.h` likewise duplicated the device glue rather than removing it from `qwen3_5.cpp`). MITIGATION: the duplication is documented in the new header's preamble and scoped as its own gate-model-touching change. **D2 — no 3-shard qkv A/B on the fp4 path** (a Marlin-interleaved operand is not row-addressable); `VT_QWEN3_ATTN_F32=1` is explicitly rejected on the quantized path since Marlin needs a bf16 activation. **D3 — `max()` global-scale collapse is LOSSY in general.** vLLM discards the non-max shard scales (`compressed_tensors_w4a4_nvfp4.py:111-114`, warning `:101-108`). We reproduce it exactly rather than "improving" it, because matching vLLM's numerics is the gate. On THIS checkpoint the shards' divisors are bit-identical within every fused group (measured at layers 0/31/63), so the collapse is exactly lossless here — but a future checkpoint could trigger real accuracy loss in BOTH engines identically. **R1 (residual, the real one) — no `QuantizationConfig`/`LinearMethod` abstraction.** Scheme selection remains a per-model tensor-name probe; `quantization_config` in `config.json` is never parsed, so a config/tensor disagreement goes undetected and every new model re-implements the probe. Named as the next quant-additivity seam (§4c); deliberately not attempted here to keep this row a controlled experiment. **R2 — divergence-triage plan, EXECUTED and CLOSED (§6b/§6c).** The plan named the quant-path suspects (merged-shard global-scale collapse, divisor-vs-multiplier convention, group-16 scale layout, Marlin scale swizzle/exponent-bias, fused-vs-split `gate_up`) and required that the gate NOT be loosened without proof. All were ruled out: the `VT_NVFP4_MARLIN=0` A/B showed GEMM-invariance, the CPU proof is bit-exact, `fallback_gemms=0`, and the teacher-forcing isolation measured every divergent position at ≤ 0.0625 nats with 28/29 exactly 0.0 — attributing the residual to the PRE-EXISTING dense-forward bf16 near-tie drift, against the dense row. The band was adopted only AFTER that proof, with the per-position nats evidence committed as goldens and an anchor `REQUIRE` binding it to our exact sequence. LESSON: a near-tie gap must be read against the ORACLE'S OWN jitter — measuring vLLM's 0.125-nat batch-composition movement at the same position is what made an ambiguous 0.0625 decisive. **R3 — the counters are the guard against a false pass:** without `marlin_gemms > 0` / `fused_gate_up > 0` a mis-wired probe that silently fell back could pass unnoticed.
