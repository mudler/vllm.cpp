# ROCm GDN kernel family (gfx1100) — BACKEND-ROCM M3 slice

**Issue:** [#41](https://github.com/mudler/vllm.cpp/issues/41) (ROCm backend umbrella; roadmap
issue-table row for `BACKEND-ROCM`).
**Claim:** `CLAIM-ROCM-GDN-KERNELS` (coordination.md).
**Base:** pinned `origin/main` `5812b8b6`.
**Hardware:** 4× RX 7900 XTX (`gfx1100`, discrete), ROCm 7.14, Linux Mint 22.3 (reporter's box).

## The gap, verified

Qwen3.5-0.8B (`Qwen3_5ForConditionalGeneration`; hybrid GDN — `layer_types` 3:1
linear:full, `full_attention_interval=4`, `linear_conv_kernel_dim=4`) on discrete gfx1100
throws `vt: no kernel for op 77 on device type 5` (`src/vt/op_provider.cpp`). The CPU
reference tier cannot install on a dGPU by design (`UnifiedMemory()=false`), so M2 for
this model needs native ROCm kernels for the ten ops the model calls that have none
(evidence: [#41 M2 attempt](https://github.com/mudler/vllm.cpp/issues/41#issuecomment-5230043080);
staged-state note: [#41 follow-up](https://github.com/mudler/vllm.cpp/issues/41#issuecomment-5237179696)).
Classic-dense M2 already runs all-native on this lane (Qwen3-0.6B, zero fallbacks —
that model needs none of these ops).

## Op inventory (the ten)

| OpId | Name | Semantics (gdn-semantics.md) | CPU oracle | CUDA donor |
|---|---|---|---|---|
| 77 | `kGdnStateGather` | indexed state rows → f32 working | cpu_ops.cpp:1666 | cuda_gdn.cu:248-335 |
| 78 | `kGdnStateScatter` | f32 working → indexed rows | cpu_ops.cpp:1708 | cuda_gdn.cu:278-335 |
| 5 | `kCausalConv1dFwd` | causal depthwise conv, prefill (§2) | cpu_ops.cpp:1021 | cuda_gdn.cu:479-520 (scalar) |
| 6 | `kCausalConv1dUpdate` | single-step decode conv (§3) | cpu_ops.cpp:1081 | cuda_gdn.cu:884-913 |
| 9 | `kRmsNormGated` | gated RMSNorm, norm_before_gate (§5) | cpu_ops.cpp:1210 | cuda_gdn.cu RmsNormGatedRowKernel |
| 58 | `kSigmoidGateBf16` | sigmoid·mul; **out bf16, attn f32/bf16, gate f32** (ops.cpp contract) | cpu_ops.cpp:2273 | no CUDA reg — CPU composite is the donor |
| 10 | `kGdnPrefill` | gated-delta recurrence (§7/§8) | cpu_ops.cpp:1331 | cuda_gdn.cu:1856 GdnScanKernel |
| 11 | `kGdnDecode` | single-step recurrence | cpu_ops.cpp:1368 | same scan, decode mode |
| 65 | `kGdnPostConv` | fused conv-split+l2norm+g/beta | cpu_ops.cpp:2337 | cuda_gdn.cu:1155 |
| 67 | `kAttnQkNormRopeGate` | fused full-attn preamble (gemma/plain) | cpu_ops.cpp:956 | cuda_ops.cu:1429 area |

## Execution-path finding (decides the design)

`qwen3_5.cpp` `IndexedGdnOpsNative()` gates the indexed state-I/O arm on
`kCausalConv1dUpdate` + `kGdnDecode` + `kGdnStateGather` + `kGdnStateScatter` all being
**natively** registered (`OpRegistered` excludes the reference tier). With the full set
registered, the model takes the CUDA lane's device-resident path with **zero model or
runner edits**. The row-copy arm is not a viable discrete fallback (host round trips per
GDN layer per token; the Vulkan record measured the cost). `needs_weight_staging()` for
discrete ROCm stays false in this row — the platform comment defers it to a measured
residency row after M2 runs.

## Routing decision

The portable `GdnScanKernel` serves both `kGdnPrefill` (qsl != null) and `kGdnDecode`
(qsl == null, optional state_idx + NULL-block zero-out). The CUDA lane's perf machinery —
WMMA chunked prefill (`#if __CUDA_ARCH__ >= 800`), fused/packed decode, register/tiled
conv variants, Triton-AOT cubins, spec-decode and fp8-quant variants — is NVIDIA-only or
not on this model's path and stays unported (docs/ROCM.md §6). Perf variants are M5
levers, and upstream's ROCm answer (Triton/CK) is a separate later decision.

## PR slicing (one family per PR, throw-order; each ends with an M2 rerun)

1. **GDN-STATE-IO**: 77/78 (the current throw) — this PR's first slice.
2. **GDN-CONV**: 5/6. 3. **GDN-NORMGATE**: 9/58. 4. **GDN-CORE**: 10/11.
5. **GDN-FUSED**: 65/67. After 4 the model completes prefill+decode; 5 restores the fused
fast path (VT_GLUE_FUSE structure parity with the CUDA lane).

## Pre-claim validation state (scratch, on gfx1100; landed through the lifecycle here)

All ten kernels were hand-translated from the donors and validated BEFORE any tree
change, in standalone harnesses against independent host references (68/68 checks:
state paths and dtype conversions **bit-exact**; compute outputs at expf/FMA ulp level,
max rel ≤ 2e-5). A complete drop-in TU compiles clean with the build's exact production
flags (`-O3 -ffp-contract=off -std=c++20 --offload-arch=gfx1100`). Five red-first
cross-device cases were assembled into a scratch copy of `test_backend_cross_device.cpp`
and run against the real library: 16/16 green with GDN cases correctly skipping
unregistered ops. Writing them surfaced four `ops.cpp` contract requirements the kernels
satisfy (compact-arm per-token state rows; RmsNormGated rank-3 shape matching; rank-2
preamble inputs; SigmoidGateBf16's f32 gate). Scratch root: `~/gdn-spike/` on the
reporter's box (harnesses + byte-exact logs).

## Test plan (red-first per family)

1. New `test_backend_cross_device` cases per family, CPU-oracle compared — byte paths
   (gather/scatter, conv-state write-back/roll) **bit-exact**; arithmetic at NMSE ≤ 5e-4.
   Written RED before each registration lands (skip-today, run-once-registered).
2. Model level: the §5.2 M2 rerun per family — Qwen3.5-0.8B vs `--device cpu`, greedy,
   tokens + `VT_OP_PROVIDER_STATS` posted on #41; the final family carries the e2e claim
   attempt.
3. Full `ctest` suite per family PR; gates reported exactly as observed.

## M2 disposition (ratified context)

The near-tie regime is on record: three boards (gfx1100, gfx1103, gfx1200) show
GPU-vs-CPU greedy near-tie flips on Qwen3-0.6B near-tied prompts, deterministic per
backend; joral's #269 analysis (uniform 0.01–0.6% per-layer drift; two real vLLM-ROCm
oracles disagreeing with *each other* at K=5) and the maintainer's #273 merge comment
ratified the **distributional gate** for this regime ("a token-exact bar cannot close
it"), and zero-fallbacks-under-`VT_OP_PROVIDER_STATS` as the discrete-M2 mechanism
evidence. This row's M2 claim for Qwen3.5-0.8B follows that disposition: e2e completion,
all-native op resolution, determinism, and the distributional comparison vs the CPU
backend — not strict token-exactness.

## Risks

- **Scan perf at M5** (portable scan vs chunked/fused): accepted — correctness first;
  the perf lever is named, not hidden.
- **bf16 state arms at model level**: covered at op level by the bit-exact scratch
  checks; the cross-device cases compare f32-state arms (matching the file's tier
  structure) — model-level bf16 state behavior is observed in the M2 rerun.
- **Wave64 (gfx9/CDNA)**: none of these kernels use warp-width-sensitive primitives;
  the scan's `__syncthreads` structure is width-agnostic. gfx9 boards remain untested
  (unverbraucht's MI50 offer on #41) — a later row.

## Stop conditions

- A family fails its cross-device gate and the defect is not resolvable from the CPU
  oracle + donor re-read → stop, post the failing evidence on #41.
- `IndexedGdnOpsNative(kROCM)` does not flip after family 4 (model path assumption
  broken) → stop, NEEDS_DECISION on #41.
- Any maintainer redirection of scope/slicing → this spec is amended in the same change.

## Outcome

(pending — filled at DONE)
