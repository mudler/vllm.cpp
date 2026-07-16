# Decode norm/quant "fusion" lever — reconciliation (2026-07-16)

**Row:** `KERNEL-EW-NORM-QUANT` (stays `PARTIAL`). **Claim:**
`CLAIM-EW-NORM-QUANT-RECONCILE`. **Kind:** grounded, evidence-first
reconciliation of the "decode norm/quant fusion (~2.0 ms/step at c16)" lever
against the project's own body-dump + audit + a fresh correct-state trace +
a direct decode-shape microbench. **Verdict:** the FUSION lever is REFUTED (no
rmsnorm+fp4quant fusion exists in vLLM's production denominator to mirror; the
one real fusion — silu+fp4quant — is already mirrored). The residual gap is a
MODEST, cross-profiler-confounded, non-bit-exact kernel-EFFICIENCY headroom that
belongs to `KERNEL-EW-NORM-ACT`, not a fusion. No production code changes.

## Scope

- In: reconcile the c16 decode norm/quant/activation glue attribution; produce
  the vLLM→ours per-kernel table + corrected number; close or redirect the
  lever; correct the records that drifted back to the misleading "fusion"
  framing (the 2026-07-16 `SUMMARY.json` note and the README/BENCHMARKS/state
  text it seeded).
- Out: no kernel/model implementation (the reconciliation shows there is no
  confirmed new fusion to implement). Any efficiency work is a separate
  `KERNEL-EW-NORM-ACT` spike gated on an in-situ A/B (see §Redirect).
- Prohibited files (other claims): `src/vt/cuda/cuda_gdn.cu`,
  `gdn_packed_reg_tile.*`, `src/vllm/v1/engine/core.cpp`,
  `src/vllm/v1/core/sched/*`.

## Evidence

1. **Correct-state c16 kernel trace, 2026-07-16** —
   `dgx:~/work/vllm.cpp-gdn-stateio-trace/20260716` (`SUMMARY.json`,
   `ours/c16-r1-decode-span-families.txt`, `vllm/c16-decode-families.txt`).
   Ours = nsys `--cuda-graph-trace=node` of `build-fix2`/`6dd24df` packed decode
   (gate 235/235), steady window [93 s,153 s]. vLLM = torch profiler c16 48p/3rep
   at `--mamba-ssm-cache-dtype float32`, 1476 pure-decode windows.
2. **`3f256ab` Inductor body-dump, 2026-07-13** —
   `.agents/specs/nvfp4-small-m-dispatch.md:954-967`; generated graph/subgraph
   SHA `d58f81b8…9401` / `466e359a…9dd8`; vLLM-kernel SHA `e4e916d1…565`.
3. **2026-07-14 parity rescan** —
   `.agents/specs/parity-rescan-2026-07-14.md`; disposition recorded at
   `docs/BENCHMARKS.md:211` ("RMSNorm/generated partitions — CLOSED / DISPROVEN
   as a parity gap").
4. **Benchmark-equivalence audit, 2026-07-15** —
   `.agents/specs/benchmark-equivalence-audit-2026-07-15.md` (`fuse_norm_quant`
   is `False`; `fuse_act_quant` is `True`; production-graphed denominator).
5. **Decode-shape RMSNorm microbench, 2026-07-16 (this claim)** —
   `dgx:~/work/vllm.cpp-ewnorm-spike/rmsnorm_decode_spike.cu` (+`spike`), run
   under `flock /tmp/gpu`, sm_121a. Isolated V0 (shipped kernel) vs V1
   (single-pass, shared-staged, bf16x2-vectorized, warp/block reduce).

## The reconciliation table (per c16 decode step)

vLLM kernel (torch-profiler) → our kernel(s) (nsys node) → per-step Δ. Counts
are per decode window/step; both arms run the SAME per-step kernel counts.

| vLLM kernel / family | cnt/step | vLLM µs | our kernel(s) | cnt/step | our µs | Δ µs | fused in denom? |
|---|---|---|---|---|---|---|---|
| `triton_red_*fused_add_rms_norm*` (names contain `scaled_fp4_quant`; **body stores BF16**, quant NOT folded) | 129 | 391 | `RmsNormRowKernel<bf16,bf16,bf16>` (add + gemma-RMSNorm → bf16) | 129 | 2006 | +1.62 | **NO** — both stop after add+RMSNorm→bf16 |
| `cvt_fp16_to_fp4` (standalone `scaled_fp4_quant.out`) | 144 | 342 | `ScaledFp4QuantKernel` | 144 | 641 | +0.30 | **NO** — count parity 144=144, both separate |
| `silu_mul_cvt_fp16_to_fp4` (ActivationQuantFusionPass, `fuse_act_quant=True`) | 64 | in 369 | `SiluAndMulFp4QuantKernel` (`VT_FUSE_SILU_QUANT` ON) | 64 | in 630 | +0.26 | **YES both — already mirrored** |
| `triton_poi_*mean_mul_pow_rsqrt*silu*` (GDN gated-norm; quant separate) | 48 | ~0 folded | `RmsNormGatedKernel` | 48 | 403 | +0.40 | gated-norm fused both; quant separate both |
| — norm/quant/act glue subtotal — | | **~1102** | | | **~3680** | **~2.58** | |

Other families at parity: GEMM+MoE+attention 106.79 vs 106.71 (+0.08); GDN
packed recurrence 21.31 vs 19.24 (+2.06, a SEPARATE recurrence-tiling lever owned
by `KERNEL-GDN-PACKED-DECODE`); GDN conv-update 0.584 vs 0.432 (+0.15).

## Reconciliation findings

**F1 — rmsnorm+fp4quant fusion is REFUTED (three independent proofs).**
(a) The `3f256ab` dumped Inductor body of the `…fused_add_rms_norm_scaled_fp4_quant…`
kernels "stops after residual-add + RMSNorm and stores BF16; the wrapper then
invokes `torch.ops._C.scaled_fp4_quant.out` separately, yielding the traced
`cvt_fp16_to_fp4`" (`nvfp4-small-m-dispatch.md:956-960`). (b) `fuse_norm_quant`
is `False` in the oracle config (audit §Real difference; rescan). (c) **Count
parity in the FRESH 2026-07-16 trace**: vLLM runs `cvt_fp16_to_fp4` at **144/win**,
exactly equal to ours' 144 `ScaledFp4Quant`/step, and both run 129 rmsnorm/step.
The `…scaled_fp4_quant…` substring in the triton names is the Inductor
graph-region label, not a fused quant — the "misleading trace name" the records
warn against twice (`kernel-matrix.md:54`, `nvfp4-small-m-dispatch.md:961`).
**Ours already has vLLM's exact structure: separate add+RMSNorm→bf16, then a
separate FP4 quant.** There is nothing to fuse that vLLM fuses here.

**F2 — the one real fusion (silu+fp4quant) is already mirrored.** vLLM's
`ActivationQuantFusionPass` (`fuse_act_quant=True`) fuses silu_and_mul with the
FP4 quant (`silu_mul_cvt_fp16_to_fp4`, 64/win). We already mirror it:
`SiluAndMulFp4QuantKernel` behind `VT_FUSE_SILU_QUANT` (default ON), bit-exact vs
the composition. FlashInfer's fused `add_rmsnorm_fp4quant` / `rmsnorm_fp4quant`
CuTe-DSL kernels exist but are NOT selected on this workload/config (the trace
shows Inductor triton + `cvt_fp16_to_fp4`, not a flashinfer fused call).

**F3 — the residual delta is EFFICIENCY, and cross-profiler-confounded.** With
fusion excluded, the ~2.58 µs·k gap is per-kernel efficiency (rmsnorm 391 vs 2006;
quant 342 vs 641). But this compares nsys graph-node timing (ours) against torch
CUPTI (vLLM) — the exact cross-profiler comparison the 2026-07-14 rescan already
flagged, when it called the +1.81 ms rmsnorm residual "a cross-profiler artifact"
and removed it from the lever queue (`BENCHMARKS.md:211`). The microbench (F5)
confirms the in-trace number overstates the kernel: ours' RmsNorm is 6-9 µs
**isolated**, not 15.5 µs. The in-trace 2006 µs carries graph-node + GEMM-bandwidth
contention overhead that no norm/quant change removes.

**F4 — the 2026-07-16 `SUMMARY.json` note regressed the disposition.** Its
`"note":"vLLM fuses add+rmsnorm+fp4quant"` and `named lever … Inductor
add+RMSNorm+FP4-quant … decode fusion` are name-based inferences that contradict
the 2026-07-13 body-dump and the 2026-07-14 rescan. The body-dump/rescan
disposition STANDS; this reconciliation restores it and corrects the
README/BENCHMARKS/state text seeded from that note.

**F5 — microbench (isolated, graph-replay timed, 2000 reps, sm_121a).**

| M | H | V0 (shipped) µs | V1 (single-pass vec) µs | speedup | bf16 parity |
|---|---|---|---|---|---|
| 16 | 2048 | 6.35 | 4.99 | 1.27× | exact (0 mism) |
| 16 | 3072 | 8.50 | 6.42 | 1.32× | exact |
| 16 | 4096 | 9.18 | 6.15 | 1.49× | exact |
| 32 | 3072 | 8.19 | 6.15 | 1.33× | **1 ULP** (maxabs 0.0156) |

V1 is 1.27-1.49× on the ISOLATED kernel, but its bf16x2-vectorized /
warp-shuffle reduction reorders the f32 accumulation ⇒ occasional 1-ULP drift —
the same token-exactness hazard that keeps the fused attention preamble
per-arch (records: 1-ULP greedy diverged within 16 tokens on ULP-sensitive
arms). A strictly bit-exact variant may only skip the pass-2 residual reload
(small). So the real recoverable headroom is a NON-bit-exact ≤1.5× on a 6-9 µs
kernel — an optimistic c16 ceiling of ~0.3-0.5 ms/step, gated on the 27B (fp4)
token gate tolerating the drift AND an in-situ A/B.

## Corrected attribution number

- **Fusion-attributable headroom: ~0 ms/step.** vLLM's production denominator
  does NOT fuse rmsnorm+fp4quant (F1); the silu+fp4quant fusion is already
  mirrored (F2). The "~2.0 ms norm/quant FUSION lever" SHRINKS to zero once
  grounded in evidence — mirroring vLLM here means keeping norm and quant
  separate, which we already do.
- **Efficiency-attributable headroom (reassigned to `KERNEL-EW-NORM-ACT`):**
  bounded, cross-profiler-confounded, NON-bit-exact. Optimistic ceiling
  ~0.3-0.5 ms/step at c16 from a ≤1.5× kernel rewrite; much of the trace's
  +2.58 ms is graph/contention/cross-profiler, not a kernel a norm/quant change
  can remove (F3, F5).

## Redirect (the real lever, if pursued)

The grounded lever is decode RMSNorm/quant kernel EFFICIENCY under
`KERNEL-EW-NORM-ACT` (mirror vLLM's Inductor `triton_red` single-pass vectorized
reduction; upstream vectorized/CUB reference `csrc/libtorch_stable/
layernorm_kernels.cu:54-173,251-360`). Requirements before any implementation:

1. A strictly bit-exact-vs-`RmsNormRowKernel` variant, OR a per-arch 1-ULP
   token gate proving 27B (fp4) greedy stays 16/16 (35B fp8 is ULP-sensitive →
   OFF), behind a default-OFF `VT_RMSNORM_DECODE_FAST`-style flag with a
   CPU-tier RED flag test + bit-exact/documented-tolerance parity test.
2. An **in-situ** interleaved c16 A/B (isolated microbench ≠ in-situ; the
   `KERNEL-GDN-PACKED-DECODE` reg-tile lever proved isolated-fast can be
   in-situ-slow via occupancy/contention). Acceptance: clears the ~0.5% non-tail
   band. Given the ≤0.5 ms ceiling on a ~168 ms TPOT (~0.3%), a null result is
   the likely outcome — spike first, do not speculatively implement.

## Gates (this reconciliation)

Records-only: clean CPU battery untouched (no src change); doc-checkpoint +
agent-record checkers green; README/BENCHMARKS/matrix/ledger/state/coordination
corrected in the same commit; microbench evidence preserved under the DGX root.

## Dependencies

None blocking. Overlaps `KERNEL-EW-NORM-ACT` (efficiency) and the parallel GDN
recurrence-tiling lever (`KERNEL-GDN-PACKED-DECODE`, separate ~2.06 ms) — both
excluded here.

## Risks / decisions

Product/parity call: mirroring vLLM (the standing directive) means NOT adding a
rmsnorm+fp4quant fusion, because vLLM does not do it in this config. Chasing the
misleading trace name would violate ground-in-upstream and the twice-recorded
"no FP4 spike from a misleading trace name" rule. The efficiency redirect is
real but modest and non-bit-exact; it is deliberately left as a spiked, in-situ-
A/B-gated `KERNEL-EW-NORM-ACT` step rather than a speculative implementation.
