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

- **Bit-exactness, and it comes first.** The two-call arm stays as the reference;
  the fused arm must reproduce it byte-for-byte on the same weights. W11 expects
  `limit=+inf` to reduce to plain `silu(gate) * up`, but an expectation is not an
  assertion and Laguna's own `GateUpSilu` decides it. **A near-tie is not
  acceptable**: either the fused arm is byte-identical, or the row records the
  divergence and stops rather than trading correctness for 6%.
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
| W1 | Read the REAL `UD-Q4_K_XL` tensor table and record whether `experts_gate` and `experts_up` share a block-quant dtype, per layer. This can bound or kill the row and costs no code | dtypes recorded from the file |
| W2 | Route the pair through `vt::MoeGateUpSwiGLUGrouped` behind a default-OFF flag, with the mixed-dtype refusal and fallback | bit-exact vs the two-call arm; refusal asserted |
| W3 | Same-binary A/B under one lease, decode only, and flip the default if it is both bit-exact and faster | measured ratio recorded |

W1 is first and is deliberately not code. The row's whole premise is that both
towers share a dtype on a checkpoint whose quantization is dynamic by design, and
that is a fact about a file rather than a thing to discover from a failing test.

## Now

`READY`. Spec committed, no implementation. Next action is W1, which needs the
checkpoint but no GPU.
