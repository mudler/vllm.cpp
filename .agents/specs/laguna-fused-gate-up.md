# `PERF-LAGUNA-FUSED-GATEUP` — Laguna's grouped MoE quantizes its activation twice

Issue [#2061](https://github.com/mudler/vllm.cpp/issues/2061). Owning row
`MODEL-TEXT-laguna-laguna-for-causal-lm`. Lever #1 of the re-ranked list in
[`laguna-s21-w7-speed-2026-07-31.md`](laguna-s21-w7-speed-2026-07-31.md) §W11.

## Why this row exists, and why the one next to it does not

W11 took a GO/NO-GO on device-residency for this model and DEMOTED it: after W8
and W9 landed, decode is GPU-compute-bound, GPU-busy (2.56 s) ≈ host sync time
(2.59 s) ≈ the decode wall, and the host is serially WAITING on real kernels
rather than idling between them. A later reading of `LagunaFfnBlock` proposed
exactly the rework W11 had already priced at ~0.02 s/tok and called not worth a
multi-brick campaign; that proposal is retracted in
[#2050](https://github.com/mudler/vllm.cpp/issues/2050). **This spec starts from
W11's measurement rather than from a fresh source reading**, which is the
difference that decides which of the two rows was worth opening.

## Scope

| Field | Content |
|---|---|
| In | The grouped-MoE gate/up pair in `LagunaFfnBlock` (`laguna.cpp:1166-1170`): route it through `vt::MoeGateUpSwiGLUGrouped` so the activation is quantized ONCE, with a runtime refusal and fallback when the two expert towers do not share a block-quant dtype |
| Out | The `down` grouped GEMM (unchanged, one call already); the fp4/NVFP4 arm, which is a different branch and a different bottleneck ([[laguna-gap-is-gpu-compute-not-host]]); lever #2, the keep-quant GEMV bandwidth pass; device-residency, DEMOTED by W11; the placement seam, which Laguna joins only when its FFN grows a `[T,H] -> DBuf` entry |
| Gate model | `unsloth/Laguna-S-2.1-GGUF UD-Q4_K_XL`, the checkpoint W11 measured |

## The defect, exactly

```cpp
const std::vector<float> eg = LqGemmGrouped(q, lw.moe.experts_gate, arep, eids, Pk, moe_I, H);
const std::vector<float> eu = LqGemmGrouped(q, lw.moe.experts_up,   arep, eids, Pk, moe_I, H);
const std::vector<float> eact = GateUpSilu(eg, eu, Pk, moe_I);
```

Each `LqGemmGrouped` quantizes `arep` to Q8_K internally. W11 measured
`QuantizeQ8KKernel` at **12.4% of decode GPU time**; the second quantization of
an activation already quantized is pure duplicate.

`vt::MoeGateUpSwiGLUGrouped` quantizes once, runs both GEMMs and applies the
SwiGLU in its epilogue, returning `eact` directly. Its contract
(`include/vt/ops.h`): `out[P,N]` f32, `act[Pa,K]` with `Pa == 1` meaning
broadcast, `gate_w`/`up_w` `[E*N,K]` in the SAME block-quant dtype,
`expert_ids[P]` i32, and a float `limit`.

## Two hazards W11 does not name

**`arep` may be unnecessary.** It is `Pk` memcpy'd copies of one row, built by a
host loop. The fused op's `Pa == 1` broadcast may accept the single row directly,
which removes the `Pk x H` staging buffer and its copy loop as well as the
duplicate quantize. Check the op's contract; do not assume.

**A UD quant need not give gate and up the same dtype.** `UD-Q4_K_XL` is a
DYNAMIC quant that varies type per tensor by design, and the fused op requires
one dtype across both towers. A checkpoint whose `experts_gate` is Q4_K and whose
`experts_up` is Q5_K cannot take this path at all. **The gate model is exactly
such a checkpoint**, so this is the first thing to measure and it may bound the
whole row: if the two towers differ there, the lever applies to fewer
checkpoints than W11's 12.4% implies. Read the dtypes off the real file before
writing code.

## Gates

- **Bit-exactness was demanded here and is NOT achievable. The demand was written
  before the two expressions were compared, and W2 corrected it.**

  ```
  Laguna GateUpSilu     : (g / d) * u          , d = 1 + exp(-g)
  shared fused epilogue : g * (1.0f / d) * u
  ```

  A divide against a reciprocal-then-multiply: one rounding against two. Measured
  over a deterministic sweep, **20.3% of values differ, worst case 2 ULP, worst
  relative 2.4e-7**, and the sign never flips. `limit=+inf` DOES disable the
  clamp exactly, so the clamp is not the source — the arithmetic form is.

  Byte-identity would mean changing the SHARED op's epilogue, and DeepSeek-V4 is
  gated against that same op, so the change would move ITS golden to fix ours.
  The divergence is therefore BOUNDED rather than removed:
  `test_laguna_fused_gate_up` pins ≤2 ULP, <1e-6 relative and sign preservation,
  so a later change that widens the gap fails in a unit test instead of as a
  moved token nobody attributes.

  **The correctness bar moves to the project's actual one: TOKEN-exactness.** The
  fused arm ships default-OFF until a token gate on the real checkpoint shows the
  2-ULP epilogue does not move an output token. That gate needs a GPU and is W3's,
  and until it passes the two-call arm remains both the default and the
  reference.
- **Mixed-dtype refusal asserted**, with the fallback taking the existing
  two-call path, so a checkpoint that cannot use the lever is slower and never
  wrong.
- **Speed, after correctness**, same-binary A/B under one `rc` lease, decode
  only, reported ours-versus-ours.
- **Inertness**: with the fused arm off, the forward is byte-identical to today.

## Denominator caution

Do not quote `27.8 tok/s`, `15x` or `18x`. They came from an unrecorded Poolside
fork branch with no commit SHA and are superseded; the re-take against the stock
pin `b10451` is owed under
[#1003](https://github.com/mudler/vllm.cpp/issues/1003). This row's number is an
ours-versus-ours A/B, which needs no external denominator.

## Work breakdown

| ID | Work | Gate |
|---|---|---|
| W1 | ~~Read the REAL `UD-Q4_K_XL` tensor table~~ **DONE 2026-08-27, see `## W1` below: 47/47 layers pair, zero mismatch, lever available** | dtypes recorded from the file |
| W2 | ~~Route the pair through the fused op~~ **DONE**: `VT_LAGUNA_FUSED_GATEUP=1`, default-OFF, with the same-dtype precondition falling back to the two-call arm | divergence bounded at 2 ULP and sign-preserving (`test_laguna_fused_gate_up`, 111,776 assertions); existing Laguna suites unchanged |
| W3 | ~~Same-binary A/B~~ **DONE, see `## W3`**: warm order-balanced A/B, both arms repeated. Tokens DIFFER deterministically, so the default is NOT flipped; +4.28% warm is recorded as a direction at n=2 | `DETERMINISM=PASS`, `W3B_RESULT=TOKENS_DIFFER_DETERMINISTICALLY` |

W1 is first and is deliberately not code. The row's whole premise is that both
towers share a dtype on a checkpoint whose quantization is dynamic by design, and
that is a fact about a file rather than a thing to discover from a failing test.

## W1 — MEASURED: gate and up share a dtype on every layer, so the lever is available

Read 2026-08-27 from the REAL checkpoint, `unsloth/Laguna-S-2.1-GGUF`
@ `750f92f90cf54159c4d7a610cb7b3e74498e75c6`, `UD-Q4_K_XL`, by HTTP RANGE request
against the three shards — no 69 GiB download and no GPU. The tensor table is
parsed out of the GGUF headers, so these are the file's own bytes rather than a
loader's report of them.

**Shard 1 carries no tensors.** Its `content-length` is 3,683,648 bytes exactly,
which is the whole file rather than a truncated range, and its header declares
`tensor_count = 0` with 72 metadata keys. It is a metadata and vocabulary shard.
Shards 2 and 3 hold all 814 tensors, so the counts below cover the whole model.

| Tensor | Q4_K | Q5_K | Q6_K | layers |
|---|---:|---:|---:|---:|
| `ffn_gate_exps.weight` | 46 | 1 | 0 | 47 |
| `ffn_up_exps.weight` | 46 | 1 | 0 | 47 |
| `ffn_down_exps.weight` | 0 | 45 | 2 | 47 |

**The hazard does not bite: 47 layers carry both towers and ZERO mismatch.** The
UD quant does deviate from Q4_K, and where it does it moves gate and up TOGETHER
— layer 47 is Q5_K on both. So `gate_w` and `up_w` satisfy
`vt::MoeGateUpSwiGLUGrouped`'s same-dtype requirement on every layer of this
checkpoint, and the fused path is available for all of them.

`down` is a different distribution (Q5_K with two Q6_K layers) and does not
matter: it is a separate single grouped call that this row does not touch.

**`block_count` is 48 and only 47 layers have routed experts.** Layer 0 has no
`*_exps` tensors at all — it is a dense layer carrying `attn_*` plus a dense FFN,
the usual `first_k_dense_replace` shape. So a per-layer plan over this model must
expect 47 of 48, and a fused-arm count of 48 would be the bug.

Model config, from the same headers: `expert_count = 256`,
`expert_used_count = 10`, `expert_feed_forward_length = 1024`,
`embedding_length = 3072`, `feed_forward_length = 12288`.

**What W1 does NOT establish.** That the dtypes match is necessary and not
sufficient: the fused epilogue must still reproduce `GateUpSilu` byte for byte,
which is W2's gate, and nothing here measures speed. It also holds for THIS
checkpoint only — a different UD quant may pair differently, which is why the
runtime refusal and fallback stay in W2's scope rather than being dropped now
that this one is clean.

## W3 — MEASURED: the lever is worth ~4%, and it changes tokens, so it stays OFF

Run on `dgx:gpu0` (GB10) under `rc` on 2026-08-27/28, against the real
`unsloth/Laguna-S-2.1-GGUF` `UD-Q4_K_XL` @ `750f92f9` staged to the shared NAS.
Same binary, same weights, same prompt, 32 tokens; the arms differ only in
`VT_LAGUNA_FUSED_GATEUP`.

### W3a ran once per arm and produced one real result and one artefact

`TOKEN_GATE=FAIL` — the streams share two tokens and diverge at position 2
(`350` against `290`), then cascade, which is what one changed token does
autoregressively.

It also printed off=3.7328 tok/s against on=8.0328, a 2.15x gap. **That number is
an artefact and must not be quoted.** W11 priced this whole lever at 12.4% of
decode GPU, of which this removes about half, so 2.15x is two orders of magnitude
past the ceiling. The tell was the OFF arm sitting at half its own known speed:
it ran FIRST, against a 68 GiB checkpoint freshly written to CIFS, and paid the
page-cache faults the second arm never saw. A single run per arm cannot see that,
and W3a's design could not have caught it.

### W3b: warm, order-balanced, and each arm repeated

`warmup (discarded) -> off1 -> on1 -> on2 -> off2`, so neither arm owns "first"
and the page-cache cost is paid before anything is timed.

| Arm | runs | mean tok/s | within-arm spread |
|---|---|---:|---:|
| OFF (two-call, default) | 7.7734, 7.9334 | **7.853** | 2.04% |
| ON (fused) | 8.1763, 8.2032 | **8.190** | 0.33% |

**`DETERMINISM=PASS`.** Both arms reproduce themselves across repeats, which is
checked BEFORE any arm-versus-arm claim: had an arm differed from itself, the
token divergence could not have been attributed to the epilogue at all, and that
would have been the finding.

**`W3B_RESULT=TOKENS_DIFFER_DETERMINISTICALLY`.** W3a's FAIL is real and
reproducible.

### Two hypotheses, both resolved

**The cold/warm reading holds.** Warm OFF is 7.85 tok/s, matching W11's ~7.7.
W3a's 2.15x was its cold first run, demonstrated rather than argued.

**"The fused arm does less work" is REFUTED**, and it was the more serious
possibility: a wrong scale fold or a mishandled dtype would produce the same
token divergence while looking like a speedup. The measured **+4.28%** sits UNDER
W11's <=6% ceiling for this lever. Skipped work would have shown a gain far above
it. The implausible number was worth distrusting, and the real one being MODEST
is what clears the arm of computing something different.

### Verdict: the arm stays default-OFF

The lever is real and worth about 4%. It also moves an output token, which is the
measured 2-ULP epilogue landing on a near-tie argmax. `## Gates` committed to
refusing that trade before any of these numbers existed:

> either the fused arm is byte-identical, or the row records the divergence and
> stops rather than trading correctness for 6%.

That is a rule rather than a rationalisation, and it is applied here.

### What this does NOT establish

**n=2 per arm, ONE prompt, 32 tokens.** The +4.28% is a DIRECTION, not a ratified
number: the gap is only 2.1x the OFF within-arm spread, which is thin. Nothing
here is a speed claim, and no llama.cpp denominator is quoted — `27.8 tok/s` and
every ratio from it remain superseded under #1003.

**One prompt cannot show the divergence is always a near-tie.** It shows this
prompt's token 2 was one. A prompt whose margins are wider might never diverge,
and a longer generation might diverge more; neither was measured.

## W4 — the wider sweep, PARTIAL: 1 of 1 prompts measured diverged, five unmeasured

W3 recorded that one prompt at 32 tokens shows a divergence EXISTS and cannot show
how often. W4 sweeps six prompts at 256 tokens to put a rate on it. **It is
recorded here incomplete**, because the measured part is decision-relevant on its
own and the unmeasured part is blocked on infrastructure rather than on analysis.

### What was measured

**Prompt 0, "The capital of France is": DIVERGES first at token position 2 of
256**, on `dgx:gpu0`. This is an independent reproduction of W3 — a different run,
a different container, and EIGHT TIMES the generation length — landing on the same
prompt at the same position. It also rules out one hopeful reading: the divergence
is not a rare late-generation event, at least on this prompt.

Five prompts (a primes list, a Python function, a word problem, a long-form
paragraph, and a French factual) are **UNMEASURED**. Nothing about them is
implied by prompt 0.

### Why the sweep is not finished, and it is not analysis

Seven leases were spent on harness and environment faults, every one of them the
author's rather than the tree's, and they share a single root: **each fix encoded
an assumption taken from the box last seen.** Recorded because the pattern is the
finding:

| Fault | What it would have produced |
|---|---|
| `xxd` absent in the worker | False refusal of a valid checkpoint |
| `--token-ids` read as an OUTPUT flag | A gate comparing files never written |
| `decode_hp` timings inside the token diff | `FAIL` on every run regardless of tokens |
| `--idle-timeout 40m` against a 37.9 min cadence | `rc` killing a healthy job |
| `nvcc` install unverified | A 16-minute silently CPU-only build |
| `lib64` glob missing `targets/sbsa-linux/lib` | "library absent" on a box that had it |
| `find \| head -1` selecting a **stub** | Linking a no-op library, with cmake returning 0 |
| The cublasLt guard made FATAL | Rejecting dgx, the box that had always built |

The last is the general lesson: **a guard must not be stricter than the thing it
guards.** cublasLt is now a hint, and cmake — which is the authority on whether a
toolkit is usable — decides.

### A real negative result about the fleet

**Thor cannot run this sweep, and the reason is measured.** It loads this
checkpoint in **2887 s (48.1 min)** against dgx's ~16 min, so thirteen loads is
**10.4 hours** there against 3.5 on dgx. Thor is also sm_110 and needs its own
arch and library paths (`targets/sbsa-linux`, not `lib64`), which the sweep script
now detects rather than assumes. Even with the toolchain fixed, this sweep should
not run on Thor: it would hold a shared device for ten hours to answer what dgx
answers in three.

### What this does and does not support

It does NOT establish a rate. One prompt is one prompt, and the sweep exists
precisely because W3's single result could not generalise. Quoting "100% of
prompts diverge" from n=1 would repeat the error this row keeps catching.

What it does support is that the divergence reproduces across runs, containers and
generation lengths, so it is a property of the arm rather than of one execution.
Combined with W3's `DETERMINISM=PASS`, the fused arm is deterministic and
deterministically different. **The default stays OFF**, which is where W3 put it
and where this evidence keeps it.

## Now

`ACTIVE`, and the row's question is answered. W1 measured the dtype pairing
(47/47 layers), W2 built the arm and bounded its epilogue divergence at 2 ULP,
W3 measured that the divergence moves a token and that the lever is worth ~4%
warm. **The arm ships default-OFF and the two-call path remains the reference.**

What is owed, and neither is a blocker on the above:

- The wider token sweep, **still owed for five of six prompts** (see `## W4`).
  Prompt 0 reproduced at 256 tokens; the rest are blocked on dgx availability, and
  Thor is ruled out on measured load time. A rate would only change the decision
  if it came back near ZERO, which prompt 0 argues against.
- A ratified speed number, if the arm is ever defaulted on: n=2 on one prompt is
  a direction. That needs repeats on an idle box.
