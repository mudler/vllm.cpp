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
  `.agents/developer-preferences.md`). The spec landed in its own pull
  request first; this implementation is the second.

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

- **FMA contraction on IQ4_XS's float-accumulation body — MEASURED, RESOLVED
  IN FAVOR OF THE SIMPLER PATH.** IQ4_XS's dot is the one format in this row
  (and in the whole quant-dot family) whose core is not a single integer
  accumulator: it forms `d1`/`d2` as f32 and folds in per-sub-block
  `sumf +=` steps, eight per super-block (`cuda_quant_dot.cu:606-680`,
  extensively commented on exactly this point). On CUDA that required
  `__fmul_rn`/`__fadd_rn` in place of ordinary `*`/`+`, because nvcc's
  default `-fmad=true` silently contracts the textual two-rounding sequence
  into a single-rounding FMA and two of eight real super-blocks then
  disagreed with the oracle by 1-4 ULP. **W0/W1 measured this directly on
  the target hardware (RX 9060 XT, gfx1200, ROCm 7.2, `isravale`):** plain
  `*`/`+` (no non-fused intrinsics) in `DotIQ4XS` is BIT-EXACT against the
  oracle's own per-super-block numbers, over the SAME four real
  `unsloth/GLM-5.3-Flash-GGUF` super-blocks and the SAME expected bits
  (`iq2xs_iq4xs_dot_golden.h`) CUDA's gate uses, both isolated (k=256, one
  contributing lane, zero reassociation possible) and combined (k=1024,
  four lanes, the same `__shfl_down_sync` tree CUDA's comment derives) —
  `tests/vt/test_backend_cross_device.cpp`, "ROCm IQ4_XS dots the ORACLE's
  own numbers on REAL checkpoint bytes", 13/13 assertions green. The
  hypothesis held: `CMakeLists.txt:414`'s project-wide
  `-ffp-contract=off` on `$<COMPILE_LANGUAGE:HIP>` is sufficient on its own,
  so `DotIQ4XS` on ROCm uses plain `*`/`+` and does **not** carry CUDA's
  `__fmul_rn`/`__fadd_rn` workaround. This is a measured result, not an
  assumption carried over — the whole point of naming this as a risk was to
  force the check rather than inherit the CUDA fix by habit.
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

Landed, on `isravale` (RX 9060 XT, gfx1200, ROCm 7.2.3), GPU work under
`flock ${GPU_LOCK:-$HOME/gpu.lock}` throughout:

- **`test_backend_cross_device.cpp`**, three cases touched/added, run
  standalone and as part of the full file (41/42 cases, 83998/83999
  assertions — the one failure is `MoeSiluMul matches the CPU oracle within
  NMSE <= 5e-4`, confirmed PRE-EXISTING and unrelated: byte-identical
  mismatch reproduced on an independent binary built from the sibling
  `KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4-w1` worktree, which touches neither this
  kernel nor this dtype):
  - "non-grouped keep-quant GEMM (...IQ4_XS/IQ3_XXS) matches the CPU
    oracle" — both new formats added to the existing table-driven CPU-vs-
    ROCm case, NMSE ≤ 5e-4, random valid blocks (unconstrained lookup
    indices need no in-range fixture change).
  - "grouped quant expert GEMM (...IQ4_XS/IQ3_XXS) matches the CPU oracle"
    — same extension on the grouped/MoE path, the one the motivating
    checkpoint's routed experts actually use.
  - "ROCm IQ4_XS dots the ORACLE's own numbers on REAL checkpoint bytes"
    (NEW) — the bit-exact gate the FMA-contraction risk needed, ported from
    `test_cuda_quant_dot.cpp`'s `CheckCudaOracleDot` shape onto the same
    golden vectors: bit-exact per-superblock (k=256, one contributing lane)
    and warp-reduction-order-exact combined (k=1024, four lanes, primary
    bit-equality + secondary reassociation-bound check). 13/13 assertions.
- **`test_gguf_keep_quant.cpp`**: the exhaustive per-device totality table's
  hand-mirrored ROCm predicate and its `gemm_kept` constant (8 → 10) updated
  to admit IQ4_XS; IQ3_XXS is not in this test's `all_types` enumeration
  (a pre-existing gap shared with Q2_K, not closed by this row) and is left
  to the cross-device gate above. 52/52 cases, 10325/10325 assertions.
- **`test_gguf_device_fit.cpp`**: `#2516`'s two ROCm residency pins split
  per-tensor (IQ4_XS's `down_exps` now expects `kKeepQuant` on ROCm;
  IQ2_XS's `gate_exps` is unaffected and still expects `kExpandBf16`,
  since #1940's other five formats stay owed); the all-or-nothing
  "NO PLAN" case is unchanged in outcome (`CHECK_FALSE` still holds, because
  the still-unsupported IQ2_XS tower alone fails the lane) with its comment
  corrected to say why. 24/24 cases, 182/182 assertions.
- `ctest -R 'rocm|cross_device'` (plus the individually-run ROCm suites
  `test_rocm_arch`/`test_rocm_backend`/`test_exl3_rocm`/
  `test_gemma4_rocm_fp8_seams`/`test_rocm_fp8_kv_cache`): zero regression,
  all green.

**End-to-end reload — the row's actual acceptance criterion — LANDED.**
`Nail-Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf` (19.39 GB on disk, `isravale`
`/home/justin/Nail/`) is the real motivating checkpoint, not a stand-in: its
own header histogram is `{BF16: 2, F32: 308, IQ4_XS: 391, Q5_K: 51,
Q6_K: 1}` — every quantized tensor in the file is one of the three dtypes
this row's target hardware now has a keep-quant kernel for (read with
`docs/bench-evidence/limb3-vehicle-search-20260904/gguf_header.py` before
running anything, not assumed from the filename).

```
VT_DEVICE_WEIGHT_BUDGET_BYTES=13000000000 \
./build-hip/examples/vllm-cli --model /home/justin/Nail/Nail-Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf \
  --device auto --max-num-seqs 1 --kv-cache-dtype fp8 --kv-cache-memory 2000000000 \
  --prompt "The capital of France is" --max-tokens 16
```

```
engine: device placement INSTALLED: 15 layers run their routed experts on cpu, the rest on rocm (resolved against 40 layers, origin fit)
engine: device placement: --fit placed 15 layer(s) (6417285120 B) to bring a 19333564672 B footprint under a 13000000000 B budget
vllm-cli: run=1/1 finish_reason=length prompt_tokens=5 completion_tokens=16 secs=3.919 tok_s=4.083
 Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
```

The decisive number is the **19,333,564,672 B (~18.01 GiB) footprint** --
it matches the file's on-disk size, not the ~70 GiB a bf16 expansion of
these tensors would produce. That is the keep-quant residency actually
taking effect on ROCm, not merely compiling: before this row,
`DeviceKeepQuantSupported` routed every IQ4_XS tower to `kExpandBf16` here
and the streamed-expert lane's blow-up SIGSEGV'd this same box on this
family of checkpoint (`vllm-cpp-rocm-crash-iq4xs` session memory). Clean
exit, coherent completion, zero crash. Not a synthetic fixture, not a
narrower stand-in geometry -- the actual artifact the row exists for.

Not done in this wave (see Owed):

- The `ROCM-KQUANT-NWARPS-DECODE` re-measurement (`rocprofv3 --kernel-trace`
  on a real quant-matched trace workload) — this issue's own stated reason
  for existing beyond plain coverage. `isravale` has no `rocprofv3` profiling
  set up in this session; the correctness gates above stand on their own,
  but the nwarps question is still open. The 4.083 tok/s figure above is NOT
  a substitute measurement for it: it is a mixed CPU+ROCm run at a
  CPU-offload-heavy split, not an isolated ROCm-kernel throughput number.

## Owed

- The other missing ROCm formats (Q4_0, Q5_0, IQ2_XS, IQ4_NL, IQ3_S,
  MXFP4): tracked by #1940 and left for a follow-on row.
- A WMMA/tensor-core tile for IQ4_XS, if the scalar tier's measured
  throughput warrants one (mirroring how `KERNEL-QUANT-CIQ-GEMM-ROCM`
  followed the existing four formats' scalar tier): not attempted here.
- The nwarps re-measurement itself: `PENDING`, not completed within this
  implementation wave for lack of a `rocprofv3` profiling setup on
  `isravale`, not silently dropped. The correctness gates (Tests) are
  unaffected by this being open.

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

`ACTIVE`. W0 (FMA-contraction probe), W1 (`DotIQ4XS`) and W2 (`DotIQ3XXS`)
are LANDED in this pull request, on both the plain (`MatmulBTQuantKernelRocm`
/ `KQuantGemmK`) and grouped/MoE (`MatmulBTQuantGroupedKernelRocm` /
`GroupedKQ8K`) arms, plus `DeviceKeepQuantSupported`'s ROCm admission list.
Gated per the Tests section above, on target hardware (`isravale`,
RX 9060 XT / gfx1200), zero regression. The FMA-contraction risk resolved in
favor of the simpler path: HIP's project-wide `-ffp-contract=off` is
sufficient, no CUDA-style non-fused-multiply workaround needed.

**The real-checkpoint end-to-end reload also LANDED**, after this pull
request was first drafted: `Nail-Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf` loads and
generates coherent tokens on `isravale`, with the resident footprint
(~18.01 GiB) matching the on-disk size rather than a bf16 blow-up — see
Tests. That was the row's actual acceptance criterion, and it is now
satisfied on the artifact that motivated the row, not a synthetic
stand-in.

Remaining before `DONE`: only the `ROCM-KQUANT-NWARPS-DECODE`
re-measurement (`PENDING`, see Owed) — it does not block this pull request,
since the row's own scope is coverage and correctness, and it is named
rather than silently dropped.
