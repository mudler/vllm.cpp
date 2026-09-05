# The Q4_K_M oracle disagrees with itself, and by more than the gate's own margins

Row `QUANT-QWEN38-27B-GGUF-ARM`, for
[#2534](https://github.com/mudler/vllm.cpp/issues/2534). Spec:
[`qwen38-27b-q4km-token-exactness.md`](../../.agents/specs/qwen38-27b-q4km-token-exactness.md).
This is a DIAGNOSIS of the failure recorded in
[`qwen38-27b-q4km-token-gate-20260823.md`](qwen38-27b-q4km-token-gate-20260823.md)
and it produces no gate result. `TOKEN_GATE` is untouched and still `FAIL`.

## What ran

One `rc` job on `thor:gpu0`, 2026-09-02, worker `rc-worker-n8smh`:

| `rc` job | Purpose | Window (UTC) |
|---|---|---|
| `deb6322d-bd06-4dd1-a5ac-2dec9987fbe1` | four runs of the STOCK oracle against itself | 00:08:24 to 00:27:30 |

Evidence on the share at `/mnt/nas_share/rc/q4ktok-thor/`, which the worker sees
as `/workspace/q4ktok-thor/`: `job/` holds every script,
`out/rc-worker-n8smh-20260902T000824Z/` holds every log. Nothing reached the box
by `ssh`. No vllm.cpp was built; this job measures the ORACLE only.

## Identity, re-asserted rather than trusted

```text
llama_pin              = 10bf611e533d81f739128304991c5e133c6aebd8  (tag b10451)
llama_src_porcelain    = EMPTY, before AND after the harness was built
gguf_sha256            = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
gguf_size              = 17106775008
host                   = rc-worker-n8smh (thor:gpu0), 14 cores, aarch64
system_info            = NEON=1 ARM_FMA=1 FP16_VA=1 MATMUL_INT8=1 SVE=1 SVE_CNT=16 DOTPROD=1 REPACK=1
```

`job/oracle_logits.cpp` is the 2026-08-23 `oracle_tokens.cpp` harness plus two
knobs: `use_extra_bufts` from the environment, and teacher forcing with a
full-logits dump. It links the stock libllama built at the pin from a byte-clean
tree and calls the public `llama.h` API only. `use_extra_bufts` is the SAME field
the stock `-nr/--no-repack` flag sets (`common/arg.cpp:2413-2416`), so both
settings are stock oracle behaviour and neither is a patch.

## Four controls, all clean

1. **`R1_IDENTITY=EXACT`.** The repack-on run reproduced the 2026-08-23 recorded
   token ids on 6 of 6 prompts, from a FRESH build on a DIFFERENT worker
   (`n8smh`, not `kk96r`). The oracle's determinism is confirmed a third time and
   this harness is validated against the gate it is compared to.
2. **The `blk.64` accounting re-observed.** Both configurations emitted exactly
   the 15 `unused tensor blk.64.*` warnings.
3. **The lever is observed, not assumed.** The harness prints `USE_EXTRA_BUFTS 1`
   and `0`; llama.cpp's own `load time` falls from 651.47 ms to 302.30 ms when
   the repack work is skipped.
4. **R4 reproduces the recorded margins to six decimal places.** Teacher-forcing
   the oracle along our 2026-08-23 ids returns the same six rank-2 steps with the
   same logits: `p0/7 gap 0.058050`, `p1/34 0.085434`, `p1/35 0.124247`,
   `p2/20 0.178236`, `p4/14 0.115482`, `p5/32 0.027185`. Nothing has drifted.

## Result 1: the oracle fails its own 6-of-6 gate, 1 of 6

`ORACLE_SELF_DIVERGENCES=1/6`. Identical artifact, prompts, token counts and
sampling; the only difference is `use_extra_bufts`.

```text
ORACLE_SELF prompt=1 DIVERGE first_diff_index=34 repack_on=3095 repack_off=198
```

That is one of the six steps the gate lost, and **198 is the token vllm.cpp
produced there**. Given a different one of its own Q4_K kernels, the oracle emits
our token at the exact step it convicted us on.

## Result 2: every contested gap is inside the oracle's own noise

`R3` (repack off) was teacher-forced along `R1`'s own ids, so both walked the
same sequence and the full logit vectors are comparable at every step.
71,516,160 logits compared elementwise.

The repack is a byte PERMUTATION of the same quantized values
(`make_block_q4_Kx8`, `repack.cpp:2836-2870` copies the eight deltas and mins and
interleaves the quants), so the dequantized weights are identical on both sides
and only the order and granularity of the fp32 arithmetic differs.

```text
GLOBAL max_abs=1.365718e+00 rms=7.886647e-02 n=71516160 argmax_flips=1
```

| per step, over 288 steps | min | median | max |
|---|---:|---:|---:|
| max abs logit delta | **0.2020** | 0.3790 | 1.3657 |
| rms logit delta | 0.0412 | 0.0729 | 0.1856 |

The six contested gaps are 0.027185 to 0.178236. **Every one is below the
MINIMUM**, so there is no step at which the oracle's self-perturbation is small
enough to resolve any of them, and five of six sit below its median rms.

## What this means for the arm

**It does not clear the arm.** A perturbation of exactly this size flips 1 of 6
prompts; ours flips 5 of 6. Were our error the same size we would flip at about
the oracle's rate. We do not, so we carry an additional and materially larger
term. Removing it is this row's work.

**It does bound what the gate can ever mean.** A decision at a 0.03-to-0.18-logit
margin is not a property of llama.cpp but of which of its four Q4_K rounding
schedules executed. `TOKEN_GATE=PASS` at 6 of 6 therefore requires pinning the
oracle's EXECUTED kernel path -- host architecture, `-mcpu` feature set, repack
state and batch size -- and not only its revision. That is a tightening of the
gate's definition. The tolerance stays exact and no band is reached for.

**The honest target is the noise floor, not zero.** An engine that does not
bit-reproduce `ggml_gemv_q4_K_8x8_q8_K` sits at about 1 of 6. Reaching 0 of 6
needs that aarch64 kernel ported, which is architecture-specific by construction
and would not transfer to the gfx1151 arm the standing goal names.

## What was refuted

- **The `QuantizeQ8KK` hypothesis, on the CPU path.** `QuantizeRowQ8_K` is
  byte-identical to `quantize_row_q8_K_ref` over 18,000 rows and 4.6 M quants.
- **A Q6_K port defect.** `VecDotQ6_KQ8_K` is bit-identical to the generic body.
- **A Q4_K/Q5_K port defect.** They differ from `_generic` by 1.1e-06 to
  4.3e-06 absolute, the same size as ggml's disagreement with itself.
- **A wiring defect**, already refuted on 2026-08-23 by 282 of 288 steps at
  rank 1 and re-confirmed here by R4.

Those four were measured on an x86-64 dev box against `libggml` built at the pin;
recipe and figures in the spec's Term B section.

## Still owed

The same-binary A/B that measures the bf16 activation term (`VT_ACT_F32`) is
staged as `job/job2.sh` and was queued behind other work at the time of writing.
Until it runs, the size of that term in this model is argued from its per-store
magnitude and from the RATE gap above, not measured end to end.
