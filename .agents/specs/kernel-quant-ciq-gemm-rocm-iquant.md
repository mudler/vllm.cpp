# Spec: KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT

- Issue: [#1940](https://github.com/mudler/vllm.cpp/issues/1940)
- Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT` — the remaining IQ4_XS coverage
  for the ROCm keep-quant GEMM. It is a sibling of
  `KERNEL-QUANT-CIQ-GEMM-ROCM`, which owns the RDNA4 WMMA tensor-core tile.
  The GFX1100 reconstruction carries IQ3_XXS and six other formats through
  the normal loader. This row retains IQ4_XS only. Owning feature row:
  `BACKEND-ROCM` (#41).
- Base: `bcade48d6f7e6666f88ffaaf3d2b78af24c35d7d` (`upstream/main`,
  2026-09-05).
- Pull request shape: separate spec and implementation pull requests
  (developer decision 2026-09-05, recorded in
  `.agents/developer-preferences.md`). This pull request lands the spec only.

## Scope

The ROCm wrapper now serves eleven keep-quant formats. Its internal GDN
provider owns Q8_0/Q4_K/Q5_K/Q6_K. `rocm_quant_dot.hip` owns Q2_K, Q3_K,
IQ2_XXS, IQ3_XXS, IQ2_S, IQ1_S, and IQ1_XXXS. IQ4_XS still expands to bf16
before ROCm compute. This can exhaust host RAM on a consumer box. The measured
35B-A3B case expands 19.4 GB of IQ4_XS tensors to approximately 70 GB.

This row ports the **IQ4_XS scalar dot kernel and device admission**. IQ4_XS
is ggml id 23 and uses the 136-byte `BlockIQ4_XS` type
(`cpu_quant_blocks.h:228-234`). It is the format that blocks the motivating
checkpoint's expert-down projections. IQ3_XXS was part of the original plan,
but the GFX1100 reconstruction now carries it.

IQ4_XS uses the `kQK_K` 256-element super-block and the `BlockQ8_K`
activation encoding already resident on this ROCm path. It needs no new
activation kernel.

Out of scope, and not attempted in this row:

- **The other missing ROCm formats:** Q4_0, Q5_0, IQ2_XS, IQ4_NL, IQ3_S, and
  MXFP4. Issue #1940 tracks these gaps.
- **IQ1_M.** It has no reader traits and is not part of this compute row.
- **Any WMMA/tensor-core tile for IQ4_XS.** The scalar tier lands first. A
  codebook format does not map to the linear-scale/min k-quant tile directly.
- **ROCm device-fit / residency sizing** (`--fit`, `VT_DEVICE_WEIGHT_BUDGET_BYTES`,
  `GgufExpertTowersReachSlotLane`). That is `BACKEND-ROCM-IQ-EXPERT-RESIDENCY`,
  already merged for the formats it covers. This row is the compute
  prerequisite that lets keep-quant residency apply to IQ4_XS tensors on ROCm;
  it does not touch the fit or budget logic.
- **The nwarps=8 decode-scaling table** (`ROCM-KQUANT-NWARPS-DECODE`,
  merged). The IQ4_XS port reuses it initially because `nsb = K/256` matches
  the existing path. The gate re-measures this choice.

### Motivating case

`joral`'s ROCm box (RX 9060 XT, gfx1200, 16 GiB VRAM, 62 GiB host RAM)
SIGSEGVs loading a 35B-A3B MoE checkpoint at IQ4_XS. The loader now keeps its
IQ3_XXS, IQ1_S, Q2_K, and Q3_K towers quantized on ROCm. IQ4_XS still expands.
The 19.4 GB of IQ4_XS tensors expand past what a streamed-expert lane can hold
in 62 GiB of host RAM. This row's IQ4_XS port addresses that remaining crash.

## What already exists to port from

IQ4_XS is **already ported on CPU and CUDA**. Issue #1940's imported body
predates that support. `cuda_quant_dot.cu:650` (`DotIQ4XS`) and
`cpu_quant_dot.cpp` (`VecDotIQ4_XSQ8_K`) both exist and are wired into
`IsCudaKeepQuantSupported` and `HasQuantDotKernel`. This row is a third backend
adaptation of an existing algorithm.

| Format | CUDA dot (adapt from) | CPU dot (cross-check oracle) | llama.cpp source |
|---|---|---|---|
| `IQ4_XS` | `DotIQ4XS`, `cuda_quant_dot.cu:650` | `VecDotIQ4_XSQ8_K`, `cpu_quant_dot.cpp:844` | `quants.c:1283` `ggml_vec_dot_iq4_xs_q8_K_generic` |

The CUDA bodies are the ones to adapt line-for-line into
`rocm_grouped_gemm.hip`'s existing `__device__ inline float Dot*(const
Block*, const BlockQ8_K*)` shape (matching `DotQ4K`/`DotQ5K`/`DotQ6K`'s
signature), because CUDA already carries the oracle-verified accumulation
order (see Risks) that a fresh transcription from the CPU generic body could
silently reassociate.

`IQ4_XS` needs only `d_kvalues_iq4nl` (16 entries, already small enough to
inline as a `__constant__`/`__device__` array directly, matching CUDA's
approach).

## Design

Add `DotIQ4XS` beside the current format arms with the same `__device__ inline
float Dot*(const Block*, const BlockQ8_K*)` signature. Dispatch it through the
existing `nsb = K / 256` loop. This is a dispatch-table extension, not a new
kernel family.

Wire IQ4_XS into:

- `DeviceKeepQuantSupported`'s `kROCM` arm (`gguf_keep_quant.cpp:136-145`) —
  add `dt == vt::DType::kIQ4_XS`.
- The two refusal-message switches in `rocm_grouped_gemm.hip` (`:889`,
  `:959`) — remove IQ4_XS from the unsupported list and add its dispatch case.
- Whatever grouped/plain GEMM dtype switch selects `DotQ4K` etc. today
  (mirror the CUDA `WType` enum shape if ROCm has an equivalent, otherwise
  the existing block-dtype `switch`).

## Upstream anchor

llama.cpp, pin `b10451` per `.agents/upstream-sync.md`.

- `ggml/src/ggml-cpu/quants.c:1283` `ggml_vec_dot_iq4_xs_q8_K_generic`.
- `ggml/src/ggml-cpu/quants.c:999` `ggml_vec_dot_iq3_xxs_q8_K_generic`.
- `ggml/src/ggml-common.h:454-460` `block_iq4_xs`, `:385-400` `block_iq3_xxs`.
- `ggml/src/ggml-cuda/mmvq.cu:387-388` — the comment this issue's title
  refers to: llama.cpp's own `nwarps`-scaling table excludes `Q3_K` and the
  `IQ2`/`IQ3` families by name because their vec_dot does a grid lookup,
  citing register pressure and lookup-table contention at higher thread
  counts. Whether that split holds for our own dot bodies is unmeasured
  (see Risks); IQ4_XS's dot has no grid lookup (a 16-entry codebook fits a
  register array) and is not implicated by that comment.

## Risks

- **FMA contraction on IQ4_XS's float-accumulation body.** IQ4_XS's dot is
  the one format in this row (and in the whole quant-dot family) whose core
  is not a single integer accumulator: it forms `d1`/`d2` as f32 and folds
  in per-sub-block `sumf +=` steps, eight per super-block
  (`cuda_quant_dot.cu:606-680`, extensively commented on exactly this
  point). On CUDA that required `__fmul_rn`/`__fadd_rn` in place of ordinary
  `*`/`+`, because nvcc's default `-fmad=true` silently contracts the
  textual two-rounding sequence into a single-rounding FMA and two of eight
  real super-blocks then disagreed with the oracle by 1-4 ULP. **This may
  not reproduce on ROCm**: `CMakeLists.txt:414` already applies
  `-ffp-contract=off` to `$<COMPILE_LANGUAGE:HIP>` project-wide, unlike CUDA
  where the project's `-ffp-contract=off` is CXX-only and never reaches
  `.cu`/`.cuh` translation units. Verify this empirically before assuming it
  (a W0-style probe: compile the naive `sumf += d1 * x` form, diff against
  the CPU oracle on the same real super-blocks CUDA's golden vectors use,
  and inspect the generated ISA for `v_fma_f32` if any block disagrees) —
  do not carry the CUDA workaround over unexamined, and do not assume the
  flag alone is sufficient without a measured check, matching how the CUDA
  side only added the intrinsics after measuring a real disagreement rather
  than as a precaution.
- **The nwarps=8 decode table (`ROCM-KQUANT-NWARPS-DECODE`) may not transfer.**
  IQ4_XS shares the existing `nsb = K/256` decomposition, so it compiles
  against the current launch shape. Re-measure the `nwarps=8` choice rather
  than inheriting it. A regression here is a decode-throughput question, not
  a correctness question.
- **Interaction with `BACKEND-ROCM-IQ-EXPERT-RESIDENCY`'s already-merged
  fit logic.** That row's `GgufExpertTowersReachSlotLane` and the ROCm
  device-fit budget was built and measured while IQ4_XS towers expanded to
  bf16 on ROCm. Landing keep-quant compute for IQ4_XS changes their resident
  size (compressed, not bf16-expanded) and
  therefore changes what `--fit` decides — re-run that row's device-fit
  gate after this lands rather than assuming its prior sizing still holds;
  flag a regression there as `NEEDS_DECISION` rather than silently
  reconciling it inside this row.

## Tests

- Extend `test_ops_quant_dot.cpp`'s existing IQ4_XS `vec_dot`
  golden-vector gates (`iq2xs_iq4xs_dot_golden.h`, already committed and
  sourced from real `unsloth/GLM-5.3-Flash-GGUF` checkpoint bytes) to a new
  `test_rocm_quant_dot.cpp`, same shape as the CUDA gate
  (`test_cuda_quant_dot.cpp`): bit-exact for IQ4_XS against the same
  real-checkpoint golden values CUDA's gate uses, since
  bit-exactness is the property the FMA-contraction risk above is actually
  about.
- `test_backend_cross_device.cpp`: add IQ4_XS to the CPU-vs-ROCM cross-check.
- Rerun `ROCM-KQUANT-NWARPS-DECODE`'s own measurement recipe
  (`rocprofv3 --kernel-trace` on a real quant-matched trace workload) for
  IQ4_XS specifically, to answer the nwarps question this issue was
  filed to test — record the result (transfers / does not transfer) rather
  than assuming either.
- `ctest -R 'rocm|cross_device'`, zero regression on the four existing
  formats' numerics.
- End-to-end: reload the motivating checkpoint (or a same-format synthetic
  fixture if the real 35B-A3B artifact is not staged on the gate host) on
  `isravale` (RX 9060 XT, gfx1200) or an `rc`-leased ROCm fleet device, and
  confirm keep-quant residency replaces the prior bf16 SIGSEGV — this is
  the row's actual acceptance criterion, not merely the unit-level dot
  gates.

## Owed

- The other missing ROCm formats (Q4_0, Q5_0, IQ2_XS, IQ4_NL, IQ3_S,
  MXFP4): tracked by #1940 and left for a follow-on row.
- A WMMA/tensor-core tile for IQ4_XS, if the scalar tier's measured
  throughput warrants one (mirroring how `KERNEL-QUANT-CIQ-GEMM-ROCM`
  followed the existing scalar formats): not attempted here.
- The nwarps re-measurement itself, if it is not completed within this
  row's implementation wave for lack of GPU time: record as `PENDING` on a
  named lease/box, never silently dropped.

## Stop conditions

- `NEEDS_DECISION`: any request to extend this row's scope to another missing
  ROCm format or to a tensor-core tile for IQ4_XS.
- `NEEDS_DECISION`: if the FMA-contraction probe (Risks) shows
  `-ffp-contract=off` does *not* fully protect IQ4_XS's accumulation on
  this toolchain, before deciding whether to port CUDA's `__fmul_rn`-style
  workaround (HIP's equivalent, if one exists) or to gate the format at
  NMSE instead of bit-exactness the way Q4_K/Q5_K/Q6_K already are.
- 20 failed attempts within the implementation wave: stop, report the
  measured position and the next traceable hypothesis.

## Now

`SPIKE`. This pull request lands the spec only; no product code changes in
this change. The GFX1100 reconstruction supplies the former IQ3_XXS scope.
Next: W0 probes the FMA-contraction question on gfx1200, then W1 ports
`DotIQ4XS` with its focused gate before the combined `ctest` sweep.
