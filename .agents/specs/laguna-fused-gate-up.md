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
| W1 | ~~Read the REAL `UD-Q4_K_XL` tensor table~~ **DONE 2026-08-27, see `## W1` below: 47/47 layers pair, zero mismatch, lever available** | dtypes recorded from the file |
| W2 | Route the pair through `vt::MoeGateUpSwiGLUGrouped` behind a default-OFF flag, with the mixed-dtype refusal and fallback | bit-exact vs the two-call arm; refusal asserted |
| W3 | Same-binary A/B under one lease, decode only, and flip the default if it is both bit-exact and faster | measured ratio recorded |

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

## Now

`ACTIVE`. W1 measured and recorded above: the lever is available on the gate
model, on all 47 expert layers. Next action is W2 — route the pair through
`vt::MoeGateUpSwiGLUGrouped` behind a default-OFF flag, keep the mixed-dtype
refusal and fallback, and gate bit-exactness against the two-call arm. W2 needs
no GPU; W3's A/B does.
