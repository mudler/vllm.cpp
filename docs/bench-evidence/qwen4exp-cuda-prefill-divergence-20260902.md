# WHERE the `qwen4_exp` CUDA prefill leaves the CPU prefill, layer by layer, 2 September 2026

Wave PREFILLDIV of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2547](https://github.com/mudler/vllm.cpp/issues/2547).

**The one-line result.** The first tensor on the path whose value differs between
the two arms is **decoder layer 0's Gated DeltaNet block output**, at
`rel(sum|x|) = 3.525e-04`, produced from an input that is **bit-identical** on
both arms. A same-binary A/B names the mechanism: `VT_GDN_CHUNKED=0` drops that
tap to `1.062e-06`, a **332x** reduction. And the mechanism is **not a defect** —
it is vLLM's own algorithm. **The GPU does NOT emit the CPU sequence, in either
configuration.**

## 1. What was run, and on what

**Tree.** `0283fae36b3d925e295aa2b9679338e05923e963` — this wave's branch merged
with `row/QWEN4EXP-DECODEDIV-2496`
([#2550](https://github.com/mudler/vllm.cpp/pull/2550)), so ONE binary carries
both #2550's decode fix and this wave's layer fingerprint. Source tarball sha256
`964f1170c05eac983bfa4344a76d6b45fc74b100b2efd97f7fdf5e2db64f7a27`. That merge
is not a landing shape and is not on any row branch, so it is pushed as
`measure/q4exp-prefilldiv-20260902` for the sole purpose of keeping the sha
above resolvable: an evidence file whose tree cannot be checked out is a
citation, not evidence.

**#2550 HAS SINCE LANDED**, at `bb78d1ee8`, so the dependency this measurement
was composed to satisfy is now on `main` on its own. The measured tree stops
being a tree that exists nowhere and becomes the composition `main` now carries:
the decode fix plus this wave's fingerprint. Nothing measured here is restated on
that basis -- the numbers below were taken on `0283fae36` and are reported
against it -- but a reader no longer has to reconstruct the pairing to reproduce
them.

**Host.** `thor:gpu0` under an `rc` lease, job
`814d530c-e742-4706-8ba3-85f39ccb7c31`, pod `rc-worker-n8smh`, aarch64, NVIDIA
Thor, compute capability 11.0, driver 595.78, nvcc 13.0.88, built `sm_110`.
`CMAKE rc=0`; `BUILD rc=0 wall=1524s objects=613`; server
`c3b355deb75efb0071fe6ec21d5067f5e2520722c672487fd67939c363e1ee58`, one binary
for all three arms.

**Artifact.** The released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, shard 1
sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`,
72,546,461,650 bytes, verified inside the lease before any arm ran.

**Request.** `examples/vllm-server`, greedy, `max_tokens=8`, prompt
`The capital of France is`, `--block-size 16 --num-blocks 128 --max-model-len
256`, no `CUDA_LAUNCH_BLOCKING`, `VT_CPU_QUANT_REPACK=0` on every arm. Config:
48 layers, hidden 2560, `hc_count` 4, T = 5, PLE on layer 1, sparse attention
every 4th layer from index 3.

**Instrument.** `VT_Q4EXP_LAYER_FP=3`. Each arm printed **1314** lines and
**437 taps per step** on all three steps — the derivable total (2 outside the
loop, 9 per decoder layer, 2 more on the one PLE layer, 1 for the model output).
No arm logged an error line.

## 2. The tokens

| arm | token ids | agrees with the control |
|---|---|---|
| `--device cpu` (the control) | `11751 13 15767 411 2029 11 1092 369` | — |
| `--device cuda` | `11751 13 15767 411 1928 11 628 567` | **5 of 8** (indices 0,1,2,3,5) |
| `--device cuda`, `VT_GDN_CHUNKED=0` | `11751 13 15767 264 1103 314 5656 321` | **3 of 8** (indices 0,1,2) |

The first two rows reproduce #2547's published sequences exactly, on a different
tree, which is what makes this run a re-measurement rather than a new anecdote.

**The third row is the result that matters and it is negative.** The sequential
arm is 2.8x CLOSER to the control on the prefill hidden state (§3) and agrees on
FEWER token ids. Agreement between our two arms is not a monotone function of
the distance between them, because the decode is an argmax over near-ties. A
"CPU-vs-CUDA token-exact" gate is therefore not well posed for this architecture
at this precision, and a change that improves the numbers can lose ids.

## 3. The measurement

Step 0, `rel = |a-b| / max(|a|,|b|)` on `sum|x|`. Every row is the same tap on
the same binary and the same prompt.

| tap | CPU-CTRL sum\|x\| | CUDA-PROD sum\|x\| | rel | CUDA-GDNSEQ sum\|x\| | rel |
|---|---|---|---|---|---|
| `emb` | 37.242316 | 37.242316 | 0.000e+00 | 37.242316 | 0.000e+00 |
| `wide` | 148.969264 | 148.969264 | 0.000e+00 | 148.969264 | 0.000e+00 |
| `L00 in` | 148.969264 | 148.969264 | 0.000e+00 | 148.969264 | 0.000e+00 |
| `L00 ahc.mix` | 6517.5925 | 6517.5925 | 0.000e+00 | 6517.5925 | 0.000e+00 |
| `L00 ahc.inj` | 0.952026367 | 0.952026367 | 0.000e+00 | 0.952026367 | 0.000e+00 |
| `L00 blk` | 1073.65489 | 1074.03345 | 3.525e-04 | 1073.65375 | 1.062e-06 |
| `L00 s.attn` | 232.225395 | 232.270443 | 1.939e-04 | 232.224302 | 4.707e-06 |
| `L00 mhc.mix` | 3613.82031 | 3615.62777 | 4.999e-04 | 3613.74301 | 2.139e-05 |
| `L00 mhc.inj` | 2.92212772 | 2.92224121 | 3.884e-05 | 2.92218876 | 2.089e-05 |
| `L00 moe` | 389.976283 | 390.025768 | 1.269e-04 | 389.947936 | 7.269e-05 |
| `L00 s.mlp` | 392.881629 | 392.853498 | 7.160e-05 | 392.854855 | 6.815e-05 |
| `out` (model output) | 27964.6752 | 28054.1436 | 3.189e-03 | 27996.313 | 1.130e-03 |

**Layer 0 is the only layer that isolates a block.** Its input is bit-identical
on both CUDA arms, so `L00 blk` measures the block's own contribution alone;
every later layer's `blk` measures propagation from an already-different input
and cannot attribute anything. That is why the 332x collapse appears at layer 0
and nowhere else.

**Two independent sources, and layer 0 separates them.** The Gated DeltaNet
contribution is removable: `blk` 332x, `s.attn` 41x, `mhc.mix` 23x. The MoE
contribution is not: `moe` moves only 1.7x and `s.mlp` only 1.05x, so with the
GDN source removed a residue of `7.269e-05` per layer survives at the MoE from
an input differing by `2.1e-05`. It is the second source, it is not measured
here beyond that, and it is what remains open.

## 4. Why the first source is NOT a defect

The CUDA arm runs `GdnPrefillChunkedCuda` (`src/vt/cuda/cuda_gdn.cu:6218`), the
chunked WY decomposition. The CPU arm runs an exact sequential recurrence.
`VT_GDN_CHUNKED=0` routes CUDA to `GdnScanCuda`, the same sequential recurrence
— which is why that lever removes the difference.

**vLLM implements the chunked one.** Read at vLLM
`5559679229bc961848b121ccdeaa8fa5d79bec98` — **the parity pin itself**
(`.agents/upstream-sync.md`'s ` ```parity-pin ` block and
`.agents/oracles/vllm.md` both carry that revision) — in the vendored
Flash-Linear-Attention tree `vllm/third_party/flash_linear_attention/ops/`.
This sentence read "a FORWARD REFERENCE beyond this row's pin" when it was
written; that was wrong, and it understated the citation. Nothing else in this
file changes: every anchor below was already read at that revision, so calling
it the pin strengthens the measurement rather than restating it:

| tensor | FLA dtype | anchor | ours |
|---|---|---|---|
| `h`, the per-chunk boundary states | bf16 store, fp32 registers | `chunk_delta_h.py:352`, `:88-94`, `:142` | `TSc = Tin` = bf16 (`cuda_gdn.cu:5678`, `:5718`) — MATCHES |
| `u`, `w` | bf16, fp32 accumulate | `wy_fast.py:137-138`, `:92-93`, `:116-117` | `TSc` = bf16 — MATCHES |
| `v_new` | bf16 | `chunk_delta_h.py:357` | `TSc` = bf16 — MATCHES |
| `A` = K·Kᵀ, the matrix the triangular solve runs on | **fp32** | `chunk_scaled_dot_kkt.py:161` with `output_dtype=torch.float32` passed at `chunk.py:146` | `Am[BT,BT]` and `Tf[BT,BT]` are `sizeof(float)` (`cuda_gdn.cu:5827-5831`) — MATCHES |
| `final_state` | **fp32** | `chunk_delta_h.py:353-355` | `float* state` with fp32 WMMA accumulators (`cuda_gdn.cu:3774`) — MATCHES |

There is no `state_dtype` parameter and no switch that widens `h`; the caller
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1438-1450`)
passes none.

**This table is INCOMPLETE, and wave GDNDECOMP found the two it omits.** Both
are rounding sites FLA performs and neither appears above:

| tensor | FLA dtype | anchor | note |
|---|---|---|---|
| `A^-1`, the triangular inverse | **bf16** | `chunk.py:50` passes `output_dtype=k.dtype` to `solve_tril`; stored `fp_downcast_rounding="rtne"` at `solve_tril.py:96,203,425-460` | `A` is f32, its INVERSE is not |
| `h` as the operand of `w @ h^T` | **bf16** | `chunk_delta_h.py:178` `tl.trans(b_h1).to(b_w.dtype)` | the running state is f32 in registers and is cast DOWN to feed the product that reads it |

Their sizes, and the split of this row's `3.525e-04` into a reassociation term
and a dtype term, are measured in
[`gdn-chunked-decomposition-20260902.md`](gdn-chunked-decomposition-20260902.md)
([#2612](https://github.com/mudler/vllm.cpp/issues/2612)). The one-line result
of that wave: the chunked reassociation is exact (`2.4e-17`) and the whole gap
is the bf16 placement above.

**So the CUDA arm mirrors the oracle and the CPU arm is the outlier.** Our CPU
sequential recurrence is MORE accurate than vLLM's chunked kernel, not more
correct. Making the two arms agree by changing CUDA would move the CUDA arm AWAY
from vLLM, which AGENTS.md "vLLM is the reference" forbids. #2547's premise —
that a defect stands between `qwen4_exp` and correct CUDA output — is falsified
at the first divergence by this measurement.

## 5. What this run RETIRES

**The named candidate is cleared, by measurement rather than by argument.**
#2547 named `vt::Qwen4ExpGatedResidual`'s CUDA arm because it routes three
projections through the shared `vt::MatmulBT` and the source says "a device GEMM
re-associates the K reduction, so this arm is NOT bit-identical to its CPU
sibling". On the real weights at the real width it IS bit-identical: `ahc.mix`,
`ahc.inj` and the embedding gather and the widen upstream of them all read
`0.000e+00`. The bf16 store absorbs the re-association at this operand
distribution.

**#2547's evidence block transposes its two arms.** It records `cpu … sumabs=28054.1
… v=0.353516,…` and `cuda … sumabs=27964.7 … v=0.326172,…`. This run reproduces
both numbers and finds them the other way round: the arm whose `out` is
`28054.1436` with `v0=0.353515625` is the CUDA arm, and the arm whose `out` is
`27964.6752` with `v0=0.326171875` is the CPU arm — the one that emitted the
control token ids. The conclusion #2547 drew is unaffected; the labels are not.

## 6. What this is NOT

n = 1. One prompt, one artifact, one box, one repetition, greedy only, UD-IQ1_S
only, `num_reqs = 1`. **NOT a token gate**: no oracle decoded this prompt, and
the CPU arm is a CONTROL, not an oracle — §4 is the measurement that says so. No
speed number: the instrument drains the queue at all 437 taps per step, so the
wall times above are instrumented wall times and are not comparable to anything.
The `moe` residue of §3 is named, not diagnosed.
