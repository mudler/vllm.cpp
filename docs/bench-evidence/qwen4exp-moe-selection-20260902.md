# The `qwen4_exp` MoE routers DO pick different experts, and the reason is that a third of the boundaries are EXACT TIES, 2 September 2026

Wave MOEDIV of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2552](https://github.com/mudler/vllm.cpp/issues/2552).

**The one-line result.** The two arms' expert selections **DIFFER** — 75 of 240
prefill token-slots on the arm #2552 measured, 78 of 240 on the production CUDA
arm. **They are not a defect.** At `num_experts = 512` and
`num_experts_per_tok = 10` the router's top-k boundary is an EXACT bf16 tie at
**79 of 240 (32.9%)** of prefill token-slots on the CPU control, and within ONE
representable bf16 step at 55.8%. At a tie the selection carries no information
at all: it is decided by the lowest-index tie-break, and any perturbation
anywhere upstream reshuffles it. **The flip rate is the tie rate** — 31.3% of
slots flip against 32.9% tied.

**And #2552's own headline number is NOT a flip.** At decoder layer 0 with the
Gated DeltaNet source removed, which is the arm the issue's table was taken on,
the two arms select the **same experts** for all five tokens. The `7.269e-05`
residue there is arithmetic, and this run says which GEMM produces it.

## 1. What was run, and on what

**Tree.** `8e2ba6931dd5aae33955ffdd8e7258ff5d90eef0` — `origin/main` `a99b9c69a`
plus this wave's `VT_MOE_SEL_FP` tap and nothing else. Source tarball sha256
`be5f31ec1d0b3c7ce4ed724647cd9295caa48db62dbff1689e501654d775fd44`.

**Host.** `thor:gpu0` under an `rc` lease, job
`a5bf074b-0f1a-490d-ab51-5d561857ef9e`, pod `rc-worker-n8smh`, aarch64, NVIDIA
Thor, compute capability 11.0, driver 595.78, nvcc 13.0.88, built `sm_110`.
`CMAKE rc=0`; `BUILD rc=0 wall=921s objects=585`; server
`b3d5d97c86ffa75f6de9e70e7a036f495962c39c8d76d8ed182b9a44c83beacf`, ONE binary
for all three arms.

**Artifact.** The released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, shard 1
sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`,
72,546,461,650 bytes, verified inside the lease before any arm ran.

**Request.** `examples/vllm-server`, greedy, `max_tokens=8`, prompt
`The capital of France is`, `--block-size 16 --num-blocks 128 --max-model-len
256`, `VT_CPU_QUANT_REPACK=0` on every arm — the same request PREFILLDIV used.
Geometry read off the tap itself: `T=5 E=512 k=10`, 48 MoE calls per prefill
step, so **the MoE call index IS the decoder layer index at step 0**.

**The tap gate ran on this box and this toolchain**, not only on the development
machine: `TAP GATE OFF rc=0 … assertions: 121 | 121 passed | 0 failed` and
`TAP GATE ON rc=0 … assertions: 172 | 172 passed | 0 failed`.

**The instrument asserts that it ran.** Every arm printed 960 `moesel` lines,
384 digests and 48 prefill calls, and every arm's last digest reads
`lines=576`, which is the derivable total (48 prefill calls x 5 tokens + 336
decode calls x 1). The comparator re-derives it and prints
`COUNTED-PROPERTY … agree=True` for both sides of both comparisons.

**Two independent instruments agree on the MoE block's input, bit for bit.**
This tap's `x` axis reads the same tensor `VT_Q4EXP_LAYER_FP` calls `L00
mhc.mix`, from a different tree, and the numbers are identical:

| arm | this tap, `x` at call 0 | [PREFILLDIV](qwen4exp-cuda-prefill-divergence-20260902.md), `L00 mhc.mix` |
|---|---|---|
| CPU-CTRL | 3613.82031 | 3613.82031 |
| CUDA-PROD | 3615.62777 | 3615.62777 |
| CUDA-GDNSEQ | 3613.74301 | 3613.74301 |

Both `rel` values reproduce too: `4.999e-04` and `2.139e-05`.

## 2. The selections

`sel` is an FNV-1a hash over each token's SORTED selected ids, so one string
comparison per layer is the set-equality assertion. Step 0 only.

| comparison | prefill calls | `sel` mismatched | token-slots | FLIPPED | first flip |
|---|---|---|---|---|---|
| CPU-CTRL vs CUDA-PROD | 48 | 40 | 240 | **78** | call 0 (layer 0), tok 2 |
| CPU-CTRL vs CUDA-GDNSEQ | 48 | 38 | 240 | **75** | **call 1 (layer 1)**, tok 2 |

**Layer 0 separates the two sources here exactly as it did for the values.** On
the `VT_GDN_CHUNKED=0` arm — the one whose MoE residue #2552 named — layer 0's
selections AGREE on all five tokens, and the first flip is at layer 1. On the
production arm layer 0 already flips, because the chunked Gated DeltaNet has
put 23x more difference into that layer's MoE input (`4.999e-04` against
`2.139e-05`). So the flip threshold at layer 0 sits between those two numbers.

## 3. Why they flip: the boundary is a TIE

`ulps` is the number of representable **bf16** values between the largest
REJECTED logit and the smallest SELECTED one, under the sign-magnitude total
order. `ulps = 0` means they are the SAME bf16 value.

| arm | slots | `ulps`=0 | 1 | 2 | 3 | 4 | 5 | >=6 | tied | <=1 ulp |
|---|---|---|---|---|---|---|---|---|---|---|
| CPU-CTRL | 240 | **79** | 55 | 31 | 32 | 16 | 13 | 14 | **32.9%** | 55.8% |
| CUDA-PROD | 240 | 63 | 62 | 48 | 33 | 9 | 13 | 12 | 26.2% | 52.1% |
| CUDA-GDNSEQ | 240 | 72 | 68 | 41 | 29 | 10 | 9 | 11 | 30.0% | 58.3% |

The median MoE call's smallest margin is `minulps = 0` on every arm. The first
flip on the GDNSEQ arm reads

```
base  ids=16,19,56,58,61,230,250,320,426,493 ulps=0 lo=-4.125(0xc084) hi=-4.125(0xc084)
other ids=16,48,56,58,61,230,250,320,426,493 ulps=0 lo=-4.125(0xc084) hi=-4.125(0xc084)
```

— identical bit patterns on both sides of the boundary, on both arms. Expert 19
and expert 48 hold the same bf16 logit as the tenth-place expert, and which one
is selected is decided by nothing but which index the greedy scan reaches
first. **A discrete selection has bimodal error, and here a third of the
boundaries have no margin at all.**

That is the whole mechanism, and the arithmetic is the trigger rather than the
cause. `logit` (the router GEMM's own output) differs by only `2.378e-05` at
layer 0 on the GDNSEQ arm — no amplification over its `2.139e-05` input — yet
by layer 5 the block INPUT differs by `1.630e-02`, three orders of magnitude
higher. That jump is flipped experts entering the residual stream, not rounding
compounding.

## 4. Where the layer-0 residue actually comes from

With the selections equal, the digest's four `sum|x|` axes decompose the block.
Layer 0, `VT_GDN_CHUNKED=0`, `rel` against the CPU control:

| axis | what it is | rel | vs the `2.139e-05` input |
|---|---|---|---|
| `x` | the block input | 2.139e-05 | 1.0x |
| `logit` | the bf16 router logits | 2.378e-05 | 1.1x |
| `exp` | the assembled per-slot expert outputs | **1.421e-04** | **6.6x** |
| `shr` | the shared expert | 4.310e-05 | 2.0x |

and PREFILLDIV's `L00 moe` (the combined block output) reads `7.269e-05`,
between `exp` and `shr` as the weighted combine of the two requires. **The
router GEMM is not the amplifier. The keep-quant grouped expert GEMM is.**

## 5. Is it a defect? No, and here is the reading

**The router mirrors vLLM.** Read at vLLM
`cdefd9d4997f00da72dc6245cc60678b50761b7e`, a **FORWARD REFERENCE** 1566 commits
beyond this row's parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, which
has no `vllm/models/qwen4_exp/` at all:

| anchor | what it says |
|---|---|
| `vllm/models/qwen4_exp/nvidia/model.py:160` | `class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)` — it adds a sequence-parallel refusal and `n_shared_experts`, and inherits the router untouched |
| `vllm/model_executor/models/qwen3_next.py:170-176` | `self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False, quant_config=None)` — **no `params_dtype`**, so the gate weight is the resolved model dtype |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:897-902` | `router_logits, _ = self.gate(hidden_states)` — plain `F.linear`, bf16 in, **bf16 out** |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:26-41` -> `ops.topk_softmax` | the widening to f32 happens INSIDE the top-k kernel, after the GEMM |
| `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py:69` | the CPU arm of the same: `gating_output = gating_output.float()` |

Ours: `MoeBlock`'s reference path computes `logits = MatmulBf16(...)` into a
**bf16** `[T,E]` buffer and hands it to `vt::MoeRouterTopK`, whose CPU
(`src/vt/cpu/cpu_ops.cpp:2951`) and CUDA (`src/vt/cuda/cuda_moe.cu:64`) kernels
both widen into an f32 softmax and take a greedy top-k with a strict `>`
ascending scan, so the lowest expert index wins an exact tie. Same width, same
widening point, same tie-break. **vLLM routes this model on bf16 logits at 512
experts too, so upstream sits on the same knife edge.**

vLLM does own a fp32-capable gate — `GateLinear`
(`vllm/model_executor/layers/fused_moe/router/gate_linear.py:18-33`), a
`PluggableLayer` whose five dispatch tiers include three that emit fp32 logits.
**Qwen3Next does not opt into it**, so none of those tiers is reachable for this
architecture and Tier 5 (`F.linear`) is what runs. That is recorded as a thing
to mirror IF upstream ever flips this model onto `GateLinear`, and it is not a
licence to widen ours first: doing so would move us AWAY from the oracle.

**The expert GEMM's arm-to-arm difference is a documented re-association, and
the secondary oracle has the same one.** `src/vt/cuda/cuda_quant_dot.cu:2158-2170`
states it outright — "The INTEGER core (sum x.qs*y.qs per 32-block) is
bit-identical; only the per-block float scale sum is reassociated (warp tree vs
CPU sequential)" — and the activation quantizer is a bit-exact port either way
(`cpu_quant_act.cpp:51-81` against `cuda_quant_dot.cu:1174-1195`: ternary amax,
`d = amax/127`, `F32ToF16(d)`, `roundf(x*id)`). The released UD-IQ1_S expert
towers are IQ4_NL / Q5_0, the 32-element Q8_0-activation lane
(`cuda_quant_dot.cu:2528-2534`). llama.cpp @ `237ad9b961f009ae19ac29dbce4cd0c1251f94b3`
splits the same way: `ggml/src/ggml-cuda/mmvq.cu:566-641` accumulates per-lane
partials and finishes with `warp_reduce_sum`, while the CPU `vec_dot`
(`ggml/src/ggml-cpu/arch/x86/quants.c:841-852`) sums sequentially.

**So neither arm is wrong.** vLLM's own CPU and CUDA backends would disagree on
this model for the same two reasons, and at a 32.9% exact-tie rate the
disagreement is not a small one. #2552 is answered rather than fixed.

## 6. What this run RETIRES, and what it says about a gate

**#2552's own hypothesis is half right and the halves are separable.** The issue
asked whether the layer-0 MoE residue is a selection flip. It is **not** — layer
0 on the GDNSEQ arm agrees on every token, and the residue decomposes onto the
expert GEMM. But from layer 1 the flips are real, they are the dominant source
of everything downstream, and they were invisible to every value tap taken on
this row.

**It also confirms PREFILLDIV's gate conclusion with a mechanism.** That wave
observed that token agreement is not monotone in numerical distance and proposed
no CPU-vs-CUDA token gate. This says why: a third of the routing decisions in
this model carry zero margin, so the token sequence is a function of tie-break
order, not of accuracy. **No CPU-vs-CUDA token-exactness gate can be well posed
for `qwen4_exp` at bf16 routing.** What IS well posed and is now measurable is
the selection-set agreement rate and the tie-rate histogram of §3, which are
properties of one arm and comparable across trees. A gate against **vLLM** stays
the right target and stays OWED: it needs an artifact vLLM can load, and vLLM's
GGUF support is an out-of-tree plugin (`docs/features/quantization/gguf.md`)
while every safetensors arm of this model exceeds the largest fleet box.

## 7. The raw outputs

Committed beside this file in
[`qwen4exp-moe-selection-20260902/`](qwen4exp-moe-selection-20260902/), so every
number above is checkable rather than transcribed: `results.txt` (the job's own
`RESULT` lines, including both `TAP GATE` rcs and every arm's counted property),
`cmp-CUDA-PROD.txt` and `cmp-CUDA-GDNSEQ.txt` (the full comparisons, with the
first twenty flips of each phase spelled out), and `digests-<arm>.txt` (all 384
digest lines per arm — the `sel` hash, `minulps` and the four `sum|x|` axes for
every MoE call of every step). The per-token value lines are NOT committed: at
576 per arm they add nothing the digests and the flip lists do not already
carry.

## 8. What this is NOT

n = 1. One prompt, one artifact, one box, one repetition, greedy only, UD-IQ1_S
only, `num_reqs = 1`. **NOT a token gate:** no oracle decoded this prompt, and
the CPU arm is a CONTROL, not an oracle — nothing here says which arm is closer
to correct. No speed number: the tap synchronises once per MoE call for the
shared-expert readback, so the wall times are instrumented and compare to
nothing. The "vLLM sits on the same knife edge" conclusion in §5 is a **SOURCE
READ, not a measurement**; no vLLM process was run on this checkpoint.

**THE DECODE COLUMNS ARE VOID AND ARE PUBLISHED ONLY TO SAY SO.** The comparator
reports 336 of 336 decode token-slots flipped, and that number means nothing:
on this tree the CUDA arm answers
`11751 271 271 271 271 271 0 0` (production) and
`11751 271 271 11751 271 271 271 0` (`VT_GDN_CHUNKED=0`) against the CPU
control's `11751 13 15767 411 2029 11 1092 369`, so from step 1 the two arms are
running different token sequences and a 100% flip rate is a different-INPUT
artifact rather than a numerics result.

**That degeneracy is itself a finding, and it has a shelf life.** PREFILLDIV
recorded a fluent CUDA arm (`11751 13 15767 411 1928 11 628 567`) — but it
measured a tree that had
[#2550](https://github.com/mudler/vllm.cpp/pull/2550)'s decode fix merged in.
**On `origin/main` at `a99b9c69a`, which is what this run measured, `--device
cuda` does NOT reproduce that**; the fluency depended on a pull request that had
not landed. **#2550 landed at `bb78d1ee8` while this file was being written**,
so the branch carrying this evidence merges it and the sentence above is a
statement about the measured tree, not about `main` today. Nothing here is
re-derived on the post-#2550 tree: the prefill result does not depend on it (step
0 consumes the prompt, not a sampled token), and a decode-phase selection
comparison is now well posed and is left as the cheap next measurement rather
than claimed.
