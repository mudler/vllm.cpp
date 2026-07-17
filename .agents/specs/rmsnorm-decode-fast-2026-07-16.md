# Decode RMSNorm kernel-efficiency port (KERNEL-EW-NORM-ACT) — 2026-07-16

**Row:** `KERNEL-EW-NORM-ACT`. **Claim:** `CLAIM-EW-NORM-ACT-1`. **Kind:** the
grounded, same-profiler-adjudicated efficiency lever the
[decode-norm-quant reconcile](decode-norm-quant-fusion-reconcile-2026-07-16.md)
reassigned here (the "fusion" framing is refuted; what remains is per-launch
kernel efficiency). **Verdict:** CONFIRMED (honest same-instrument decode gap
3.18–3.56×, far over the 1.3× bar) and PORTED behind a default-OFF flag.

## Scope

- In: the decode RMSNorm launches — the 129 standalone `RmsNorm` launches/step
  (`input_layernorm` ×64 + `post_attention_layernorm` ×64 + final norm ×1, ALL
  at hidden `H=5120` on the 27B; q/k head norms are fused into the attention
  preamble, the gated norm is a separate kernel). Add a vectorized decode kernel
  behind `VT_RMSNORM_DECODE_FAST` (default OFF), 1:1 with vLLM's own CUDA
  vectorized rms-norm. No model/scheduler change.
- Out: `RmsNormGated` (separate `KERNEL-GDN-*`), FP4/FP8 quant kernels, the
  attention-preamble fused q/k norm, prefill RMSNorm behavior semantics.
- Prohibited files (other claims): `src/vt/cuda/cuda_gdn.cu`,
  `gdn_packed_reg_tile.*`, `src/vllm/v1/core/sched/*`, runner async paths
  (`async_output.*`), `sampler.cpp`.

## Upstream chain (what vLLM ACTUALLY launches at decode)

- **Production (traced) kernel:** Inductor-generated
  `triton_red_fused__to_copy_add_copy__fused_add_rms_norm*` /
  `…_scaled_fp4_quant_zeros_*` (the `scaled_fp4_quant` substring is the graph-
  region label; the body stores bf16, quant is separate — reconcile F1). Trace
  families `dgx:~/work/vllm.cpp-gdn-stateio-trace/20260716/vllm/c16-decode-families.txt`
  (5 variants, 129 launches/step total, all M=16×H=5120).
- **Portable C++ mirror target** (per the ground-every-impl + portable-fusion
  policy — Triton AOT is not sanctioned for this family): vLLM's OWN CUDA
  vectorized rms-norm, the CUDA embodiment of the same reduction/vectorization:
  `csrc/libtorch_stable/layernorm_kernels.cu:106-173`
  (`fused_add_rms_norm_kernel<scalar_t, width=8>`), launch `:310-363`
  (`block = min(hidden, max_block_size=1024)` for `num_tokens < 256`, 16-byte
  `_f16Vec` loads, `cub::BlockReduce<float,1024>`), with `_f16Vec` packed bf16
  `operator+=` / f32 `sum_squares` at `csrc/type_convert.cuh:115-194`, @ vLLM pin
  `e24d1b24`.

## Our baseline

`RmsNormRowKernel` (`src/vt/cuda/cuda_ops.cu:57`): one block/row, `kBlock=256`
threads, SCALAR bf16 loads, shared-mem f32 tree reduction, TWO passes over H.
Grid = num_tokens only ⇒ at decode M=2..32 the kernel runs at ~28% of GB10 peak
BW (8.44–8.53 µs) while vLLM's vectorized 1024-thread kernel is at ~roofline
(≈650 KB / 273 GB/s ≈ 2.4 µs).

## Phase-1 same-profiler adjudication (nsys `cuda_gpu_kern_sum`, isolated, H=5120)

Both sides measured with the SAME instrument (nsys pure-kernel GPU duration,
launch-overhead-free) on isolated harnesses, at the REAL 27B decode shape
`M × 5120` bf16 (evidence `dgx:~/work/vllm.cpp-ewnorm-phase1`). vLLM side = a
faithful `torch.compile` of vLLM's native RMSNorm decomposition (Inductor emits
`triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0`, the production family).

| M (concurrency) | ours `RmsNormRowKernel` µs | vLLM `triton_red` µs | honest ratio |
|---|---|---|---|
| 2  | 8.44 | 2.37 | **3.56×** |
| 4  | 8.49 | 2.39 | 3.55× |
| 8  | 8.44 | 2.58 | 3.27× |
| 16 | 8.46 | 2.53 | **3.35×** |
| 32 | 8.53 | 2.68 | 3.18× |

**Honest per-launch gap 3.18–3.56× ⇒ ≥1.3× at every decode shape ⇒ lever
CONFIRMED** (the cross-profiler confound the reconcile flagged is removed: this is
nsys-vs-nsys, isolated-vs-isolated). Honest ms/step (129 launches × Δ): **c16
≈ 0.765 ms/step, c2 ≈ 0.783 ms/step**. The c2 number is ~33% of the whole c2
decode gap (~2.4 ms/step) — the batch-independent-launch-count argument
(rescan-lost-lanes §3): 129 launches are fixed, so the fixed cost is a LARGER
fraction of the smaller c2 mean. (This upgrades the reconcile's ≤1.5×/≤0.5 ms
estimate — it capped at the old V1 candidate, not at vLLM's actual roofline.)

Context: ours in-situ nsys was 15.55 µs/launch (2006/129, stateio-trace) — the
isolated 8.46 µs shows the in-trace number is contention-inflated ~1.84×; the
isolated 3.35× is the conservative honest kernel-efficiency gap.

## Port (`RmsNormRowFastKernel`, spike-validated before implement)

1:1 port of vLLM `fused_add_rms_norm_kernel<bf16, width=8>`: one block/row,
`block = min(1024, 32·⌊H/32⌋)` (1024 at decode), 16-byte `RmsNormBf16x8` (=4
packed `bf162`) vectorized loads, packed `__hadd2` residual add (==
`_f16Vec::operator+=`), f32 `sum_squares`, two-stage warp-shuffle block reduction
(sum semantics of `cub::BlockReduce<float,1024>`; CUB not vendored), pass-2
residual reload, `x·inv·w` in f32 → bf16 with the gemma `(1+w)` fold to match our
shipped kernel. Selection (`TryLaunchRmsNormDecodeFast`, `cuda_ops.cu`) mirrors
vLLM's alignment/width guard: flag ON AND bf16 in/out AND bf16 residual AND
`H%8==0` AND all four pointers 16-byte aligned; else the shipped kernel.

**Spike (isolated, evidence `dgx:~/work/vllm.cpp-ewnorm-phase1/fast_nsys.cu`):**
event graph-replay V0 10.25 µs vs V2 4.10 µs (c2–c16), 4.58 µs (c32) ⇒
**2.24–2.50×**; nsys pure-kernel V2 c32 = 2.83 µs (≈ vLLM 2.68 µs, 1.06× — parity;
3.02× over V0's 8.53 µs). bf16 parity vs the shipped kernel: **EXACT (0 mism) at
c2/c4/c8/c16**, 2/163840 elements 1-ULP (maxabs 0.0078) at c32; residual stream
bit-identical all M.

## Port map

| Upstream (`e24d1b24`) | Local | Notes / deviation |
|---|---|---|
| `csrc/libtorch_stable/layernorm_kernels.cu:106-173` (`fused_add_rms_norm_kernel<scalar_t,width=8>`) | `src/vt/cuda/cuda_ops.cu` `RmsNormRowFastKernel` | 1:1; bf16 only; adds our gemma `(1+w)` fold to match the shipped `RmsNormRowKernel` semantics |
| `csrc/libtorch_stable/layernorm_kernels.cu:310-363` (launch: `block=min(hidden,1024)` for `num_tokens<256`, width/alignment guard) | `src/vt/cuda/cuda_ops.cu` `TryLaunchRmsNormDecodeFast` + `LaunchRmsNorm` hook | same guard; adds the `VT_RMSNORM_DECODE_FAST` default-OFF gate |
| `csrc/type_convert.cuh:115-194` (`_f16Vec` `operator+=` / `sum_squares`) | `src/vt/cuda/cuda_ops.cu` `RmsNormBf16x8` + `__hadd2` / f32 accumulate | packed bf16 add + f32 sum_squares, bit-identical intent |
| `cub::BlockReduce<float,1024>` | two-stage warp-shuffle block reduce (in `RmsNormRowFastKernel`) | CUB not vendored in this build; sum semantics equivalent |
| env-flag plumbing (house pattern) | `src/vt/cuda/rmsnorm_decode_fast.h` (`RmsNormDecodeFastFlagIsOn`) | pure predicate, mirrors `gdn_packed_reg_tile.h` |
| `tests/kernels/core/test_layernorm.py` (`fused_add_rms_norm`) | `tests/vt/test_cuda_ops.cpp` (`RunRmsNormDecodeFastCase`) + `tests/vt/test_rmsnorm_decode_fast.cpp` | CUDA parity re-expression + CPU flag predicate |

## Work breakdown

- W1 (DONE): Phase-1 same-profiler adjudication — both-side isolated nsys table,
  confirm ≥1.3× (result: 3.18-3.56×). Evidence `dgx:~/work/vllm.cpp-ewnorm-phase1`.
- W2 (DONE): spike the vLLM-csrc-mirror kernel isolated; validate roofline + bf16
  parity before touching production (`fast_nsys.cu`).
- W3 (DONE): implement `RmsNormRowFastKernel` + `TryLaunchRmsNormDecodeFast` +
  `rmsnorm_decode_fast.h`, default OFF; CPU flag test + CUDA parity test; clean
  `-Werror` CPU rebuild + full CPU battery + record surfaces.
- W4 (NEXT, DGX/orchestrator, one flock/series): CUDA parity case ON≡OFF; both
  model gates with `VT_RMSNORM_DECODE_FAST=1` (token exactness); interleaved c16
  A/B (3 pairs + w0) + c2 A/B flag on/off.
- W5 (follow-up): flip default ON only if W4 token gates hold AND the A/B wins;
  else leave OFF and record.

### Numerics

`__hadd2` residual add is bit-identical to the shipped `ResRound<bf16>(f32-add)`
for bf16 inputs (both round the exact sum RNE) ⇒ the residual STREAM is bit-exact.
The variance REDUCTION is reordered (1024-thread block vs 256-thread tree) ⇒ NOT
bit-identical (documented one-bf16-ulp / token-exactness hazard; 35B fp8 is
ULP-sensitive). This is why the port ships default-OFF.

## Tests to port / add

- CPU-tier flag predicate `tests/vt/test_rmsnorm_decode_fast.cpp` (default-OFF /
  '1'-leading parse), house pattern of `test_gdn_packed_reg_tile.cpp`.
- CUDA-tier parity `tests/vt/test_cuda_ops.cpp` (`RunRmsNormDecodeFastCase`,
  M∈{2,4,8,16,32}×5120 bf16 ×{gemma,plain}): fast (flag ON) vs shipped (flag OFF),
  output within one bf16 ulp, residual bit-identical. DGX-gated (skips on CPU).
- Upstream test the port is spec'd against: vLLM `tests/kernels/core/test_layernorm.py`
  (`fused_add_rms_norm`); the CUDA parity case is our re-expression.

## Gates

- Correctness: CPU battery + tools green (flag predicate); DGX CUDA parity case
  ON≡OFF within tol; both model gates (27B 235/235, 35B) with the flag ON — if
  tokens shift, follow house numerics policy (document, keep OFF, do NOT
  re-golden).
- Performance: interleaved c16 A/B flag ON/OFF (3 pairs + w0) AND a c2 A/B; accept
  = measurable TPOT reduction with no throughput regression. Given ≤0.78 ms/step
  isolated on ~168 ms TPOT (~0.46%), an in-situ null at c16 is plausible (isolated-
  fast ≠ in-situ-fast, per the reg-tile lever) — the c2 lane is the target.
- Default flip ON only in a follow-up commit if BOTH token gates hold AND the A/B
  shows a win; else land OFF, record honestly.

## Dependencies

None blocking. Overlaps the refuted-fusion reconcile (records) and the GDN
recurrence lever (separate). Hardware: dgx GB10 sm_121a, one flock/series.

## Risks / decisions

Numerics-changing mirror ⇒ default OFF, gated. Product/parity: mirroring vLLM
means the vectorized 1024-thread reduction; its 1-ULP drift is the same hazard
already accepted per-arch for the attention preamble. No re-goldening.

## DGX proof verdict + DEFAULT FLIP (2026-07-16/17, `CLAIM-EW-NORM-ACT-1`; finalized by the orchestrator after the owning agent was repeatedly killed by API-529s)

Evidence root `dgx:~/work/vllm.cpp-ewnorm-act-src` (gate3.out, gate4.out,
proof JSONs; immutable).

**gate3 — token gates: ALL PASS, both flags, both models** (source `5a53fb5`,
build `-DVLLM_CPP_TRITON=ON`): 27B paged-forward flag-off 17/17 cases + 84/84
asserts, flag-on 17/17 + 84/84; 35B flag-off 4/4 + 8/8, flag-on 4/4 + 8/8.
The reordered-reduction 1-ULP hazard did not surface in either greedy gate.
gate3's c16/c2 A/B legs were **VOID** — the build lacked the CUTLASS FP4/FA2
fast paths (c16 ~50 tok/s / TPOT ~2.6 s / TTFT ~32 s; the same defect that
voided the W3 round-1 A/B; see the dgx build rule below).

**gate4 — corrected-build A/B (CUTLASS FP4 + FA2 hard-verified in the
configure log): the fast kernel WINS at c16.** Interleaved w0 + 3 pairs,
binding c16-r1 corpus: fast r1/r2/r3 tput **801.71/802.37/799.48** tok/s,
meanTPOT **164.88/164.78/165.14** ms vs legacy r1/r2 **792.97/793.15**,
**166.56/166.68** — paired **+8.7/+9.2 tok/s (+1.1%)**, meanTPOT
**−1.68/−1.90 ms**, matching the c2/c8 attribution's ~1.65 ms RMSNorm
prediction (`62d4762`). Leg disposition: **c16 legacy-r3 VOID** (641.14 tok/s
/ TPOT 193.9 / TTFT 4033 — a ~20% interference anomaly ~20:00Z, cause not
identified; verdict rests on the 2 clean pairs + w0 sanity 800.86, majority
rule). c2 legs show the documented small-sample arrival lottery: pooled
medians fast ≈107.9 vs legacy ≈108.3 ms (parity-to-slightly-better; NOT
over-claimed; the c2 in-situ credit waits for the next grid).

**FLIP: `VT_RMSNORM_DECODE_FAST` default ON; `=0` rolls back** to the
one-block-per-row `RmsNormRowKernel` (kept for bit-exact reproduction of
pre-flip token streams). Acceptance met: both token gates hold AND the A/B
wins (TPOT down, tput up). Flag tests inverted RED→GREEN
(`tests/vt/test_rmsnorm_decode_fast.cpp`); launcher/header comments updated.
The next binding grid runs the fast kernel by default.

## 2026-07-17 — DEFAULT ROLLED BACK to OFF (27B paged_engine oracle divergence; `CLAIM-ASYNC-SCHED-W3`)

The 2026-07-16 default-ON flip is REVERTED. Its gate exercised only
`test_qwen27_paged_FORWARD` (17/17 + 84/84) and the 35B forward — NOT the full
production greedy stream. The `CLAIM-ASYNC-SCHED-W3` async-default-flip DGX
re-confirmation ran `test_qwen27_paged_ENGINE` (the 16-token pip-vLLM oracle
production stream, `tests/parity/goldens/qwen36_logits_27b/greedy_ids.npy`) and
found **fast-ON diverges from the oracle: 234/235 fast-ON vs 235/235 fast-OFF**,
IDENTICALLY across all three async arms (async-independent). The divergence:
tokens 1–6 match, then output token 7 flips (`271` vs oracle `198`) and cascades.

Root cause: the vectorized kernel's reordered 1024-thread block reduction (vs the
shipped 256-thread tree) produces a ~1-ULP f32 difference that flips a documented
27B whitespace/near-tie greedy argmax; and vLLM's ACTUAL oracle rmsnorm is the
Inductor-generated `triton_red_fused…rms_norm`, not the csrc
`fused_add_rms_norm_kernel<bf16,8>` this kernel mirrors — so bit-parity vs the
ORACLE token stream was never guaranteed (only vs our own shipped kernel, which is
itself oracle-exact on this stream). Token-exactness vs the pip-vLLM oracle is the
sacrosanct precondition, so the shipped `RmsNormRowKernel` is the default again.

`VT_RMSNORM_DECODE_FAST` predicate reverted to default-OFF / '1'-opt-in
(`src/vt/cuda/rmsnorm_decode_fast.h`); flag test inverted back RED→GREEN
(`tests/vt/test_rmsnorm_decode_fast.cpp`, 10 asserts); launcher/parity-test
comments updated. `RmsNormRowFastKernel` remains available via
`VT_RMSNORM_DECODE_FAST=1`. `KERNEL-EW-NORM-ACT` stays DONE (the norm/act family is
implemented + oracle-exact via the shipped kernel); the **fast-kernel default-ON
perf lever REOPENS** — a re-attempt must reproduce the oracle's Inductor-Triton
numerics (match the actual reduction), not just the csrc kernel, and must gate on
`test_qwen27_paged_ENGINE` (the production stream), not only `paged_FORWARD`.
Evidence `dgx:~/work/vllm.cpp-async-flip`; ledger #L503.
