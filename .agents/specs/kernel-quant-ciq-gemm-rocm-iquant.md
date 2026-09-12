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
  request first. Pull request #3029 carries the implementation and the
  reconciliation below.

## Reconciliation with main, 2026-09-09

Main restructured the ROCm quant-GEMM dispatch while #3029 was open, and this
section records what the branch had to give up and what it kept.

`src/vt/rocm/rocm_quant_dot.hip` is new on main. It owns the `kROCM`
registration of `OpId::kMatmulBTQuant` and `OpId::kMatmulBTQuantGrouped`
(`rocm_ops.hip`). The former entry points in `rocm_grouped_gemm.hip` were
renamed `MatmulBTQuantKernelRocmGdn` and `MatmulBTQuantGroupedKernelRocmGdn`,
and the new wrapper delegates to them for Q8_0, Q4_K, Q5_K and Q6_K only.
Every other dtype resolves through the wrapper's own `enum class WType`,
`DotSuperblock<WType>` specializations, `IsRocmKeepQuantSupported`,
`LaunchGemm` and `LaunchGroupedGemm`.

**IQ3_XXS leaves this row.** Main implements and registers it in
`rocm_quant_dot.hip` as `WType::kIQ3_XXS`. Its `DotIQ3XXS` reads the same
`d_iq3xxs_grid`, `d_ksigns_iq2xs` and `d_kmask_iq2xs` tables in the same
order as the body this branch wrote, and defers the same 0.25 final factor
to the warp reduction. Both are ports of `cuda_quant_dot.cu::DotIQ3XXS`, and
they differ only in how the tables are named. The branch's copy sat inside the
renamed GDN function, which the wrapper never reaches for IQ3_XXS, so keeping
it would have landed dead code beside a working implementation.

**IQ4_XS stays with this row and changes address.** No part of
`rocm_quant_dot.hip` mentions IQ4_XS on main, and `DeviceKeepQuantSupported`
does not admit it. Main's own `docs/USAGE.md` says the format "remains
separate work in #3029". The branch's `DotIQ4XS` body is preserved unchanged
in substance and moves into the wrapper's `WType` system, because the GDN
function it used to live in is unreachable for this dtype.

**The device-codebook seal follows the tables it seals.** The branch added
`SnapshotIqTablesFromDevice` beside a ROCm copy of the codebooks in
`rocm_quant_iq_tables.h`. Main's wrapper reads `vt/cuda/cuda_quant_iq_tables.cuh`
instead, which is plain `__device__` syntax and compiles under HIP. The seal
moves into `rocm_quant_dot.hip` and snapshots those symbols, and the ROCm
copy of the tables is deleted. A seal over a table no kernel reads measures
nothing.

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

The CUDA body is the one to adapt line-for-line into
`rocm_quant_dot.hip`'s existing `__device__ inline float Dot*(const Block*,
const BlockQ8_K*)` shape (matching `DotQ4K`/`DotIQ3XXS`/`DotIQ2S`'s
signature), because CUDA already carries the oracle-verified accumulation
order (see Risks) that a fresh transcription from the CPU generic body could
silently reassociate.

`IQ4_XS` needs only `d_kvalues_iq4nl` (16 entries). The wrapper already
includes `vt/cuda/cuda_quant_iq_tables.cuh`, which defines that table, so the
port adds no table of its own.

## Design

Add `DotIQ4XS` beside the wrapper's other format arms with the same
`__device__ inline float Dot*(const Block*, const BlockQ8_K*)` signature.
Dispatch it through the existing `nsb = K / 256` loop. This is a dispatch-table
extension, not a new kernel family.

Wire IQ4_XS into `src/vt/rocm/rocm_quant_dot.hip`:

- `enum class WType` — one new value, `kIQ4_XS`.
- `DotSuperblock<WType::kIQ4_XS>` — the one-line specialization every sibling
  format has.
- `FinalFactor<W>` — IQ4_XS keeps the default 1.0. Its per-sub-block delta is
  already folded into `d1` and `d2` inside the dot body.
- `IsRocmKeepQuantSupported` — one `case DType::kIQ4_XS`.
- The `LaunchGemm` and `LaunchGroupedGemm` switches — one case each. A missing
  case there throws by design (#967), so both arms move together.

Then wire the loader and the refusals:

- `DeviceKeepQuantSupported`'s `kROCM` arm (`gguf_keep_quant.cpp:136-145`) —
  add `dt == vt::DType::kIQ4_XS`.
- The two refusal messages in `rocm_grouped_gemm.hip` — move IQ4_XS from the
  `unimplemented` list to the `wrapper-owned` list. The GDN provider itself
  gains no IQ4_XS arm, because the wrapper never delegates this dtype to it.

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

Every ROCm number this row measured before 2026-09-09 was taken against the
superseded code location, so it is retained as history and does not gate the
reconciled tree. The contributor run at 41/42 cases and the operator's
reproduction at head `fa39a45a3` (48/48 cases, 84,104 assertions) both
exercised `DotIQ4XS` and `DotIQ3XXS` inside `rocm_grouped_gemm.hip`, which the
merge removed. The operator's later `strix:gpu0` run covered the device-table
seal, which moves in the same change. The gate below replaces all three.

### Measured on the reconciled tree

`isravale` (RX 9060 XT, `gfx1200`, ROCm 7.2.3), every GPU command under
`flock ${GPU_LOCK:-$HOME/gpu.lock}`, `llama-server.service` confirmed
`inactive` before and after.

RED, at merge commit `cf1f39396` with the port not yet applied:
`test_backend_cross_device` 45/48, three cases failing. Each threw
`vt rocm: matmul_bt_quant: no keep-quant kernel for dtype iq4_xs` or its
grouped twin -- "non-grouped keep-quant GEMM ... matches the CPU oracle",
"grouped quant expert GEMM ... matches the CPU oracle", and "ROCm IQ4_XS dots
the ORACLE's own numbers on REAL checkpoint bytes". The loader suites were
green there, which is the point: a GEMM the loader never routes to cannot be
caught by the loader's own tests.

GREEN, after the port:

- `test_backend_cross_device` **48/48**, 84,104/84,104 assertions, five
  consecutive full runs. One earlier run of the same binary reported one
  failed case and did not reproduce in five; the recorded flake on this box is
  `MoeSiluMul` (#1954).
- `test_gguf_keep_quant` **55/55**, 11,980/11,980 assertions.
- `test_gguf_device_fit` **24/24**, 182/182. `test_gguf_device_fit_reach`
  **21/21**, 100/100.
- `ctest -R 'rocm|cross_device'`: 10/10 targets pass, `test_rocm_quant_dot`
  included.
- The bit-exact IQ4_XS oracle case passes in the new location, so the
  `-ffp-contract=off` finding holds where the code now lives rather than only
  where it was measured. The flag is on `rocm_quant_dot.hip`'s own compile
  line, read from the generated build command and not from `CMakeLists.txt`.
- The device-codebook seal passes over the `vt::cuda` symbols the ROCm dots
  index, which is the copy that now executes.

Three `#2516` cases in `test_gguf_keep_quant.cpp` had to change encoding. They
need one format the CPU keeps and ROCm expands, and they had been re-pointed at
IQ4_XS when ROCm gained IQ1_S. Admitting IQ4_XS made them red for the right
reason, and they now use IQ2_XS, which no ROCm provider implements.

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

### End-to-end reload on the reconciled tree, 2026-09-12

The reviewer's ask on pull request #3029 was specific: the pre-reconciliation
reload evidence named no sha256, reported a calculated footprint rather than a
measured peak RSS, and compared coherent text rather than tokens. This
re-runs the same checkpoint on the reconciled tree and closes all three.

`Nail-Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf` (19,389,012,960 B on disk,
`/home/justin/Nail/`), sha256
`aeff61097613c4d8e9f418572d58afd14bf5771de236e1dd50869dd809343a23`. The
file's own GGUF metadata (`general.base_model.0.repo_url`,
`general.quantized_by`) names the base model as
`https://huggingface.co/Qwen/Qwen3.6-35B-A3B`, quantized by Unsloth. The
developer's best recollection of the source repo is
`peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP` (2026-09-12, unverified
against the file and no revision recorded, since fetch-time provenance
was not tracked); the sha256 above, not the repo name, is this row's pinned
identity for the artifact.

Measured on `isravale` (RX 9060 XT, `gfx1200`, ROCm 7.2.3), every GPU command
under `flock ${GPU_LOCK:-$HOME/gpu.lock}`, `llama-server.service` confirmed
`inactive` before and after both runs:

```
VT_DEVICE_WEIGHT_BUDGET_BYTES=13000000000 \
./build-hip/examples/vllm-cli --model /home/justin/Nail/Nail-Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf \
  --device auto --max-num-seqs 1 --kv-cache-dtype fp8 --kv-cache-memory 2000000000 \
  --temperature 0 --prompt "The capital of France is" --max-tokens 16
```

ROCm run: `--fit` placed 15 of 40 layers' routed experts on CPU to bring the
19,333,564,672 B (~18.01 GiB) weight footprint under the 13 GB budget; the
rest ran keep-quant on ROCm. `prompt_tokens=5 completion_tokens=16
secs=3.562 tok_s=4.492`. **Peak RSS was MEASURED, not calculated**: sampled
every 200 ms over the process tree via `tools/bench/sample_process_memory.py`
(`/proc/<pid>/smaps_rollup`), peak **20,135,752 KiB (~19.20 GiB)** over 45
samples, `peak_mem_available_drop_kib=2,078,764` (~1.98 GiB). Output: " Paris.
The capital of Germany is Berlin. The capital of Italy is Rome."

CPU run, same prompt, same checkpoint, same seed-free greedy decode
(`--device cpu`, no ROCm code path involved at all): `prompt_tokens=5
completion_tokens=16 secs=5.044 tok_s=3.172`. Output: " Paris. The capital of
Germany is Berlin. The capital of Italy is Rome." -- **byte-identical to the
ROCm run**, 16/16 completion tokens agreeing under greedy decoding. This is
the token-level correctness comparison the pre-reconciliation evidence
lacked: two independently-computed paths (all-CPU vs the ROCm keep-quant
GEMM this row ports) producing the same decode on the real checkpoint, not
merely each individually producing readable text.

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
- `Nail-Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf`'s exact source revision:
  `UNRECORDED`. The developer's best recollection names the repo
  (`peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP`, see Tests), but it was
  not verified against the staged file and no revision was recorded at
  fetch time. The sha256 pinned under "End-to-end reload on the reconciled
  tree" is the artifact's identity until this is confirmed or the file is
  re-fetched by an explicit revision.

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

`ACTIVE`. Pull request #3029 merged main's quant-dot dispatch and dropped
IQ3_XXS from this row, for the reasons under "Reconciliation with main". At
that merge commit the three IQ4_XS cases in `test_backend_cross_device.cpp`
are red, each throwing `no keep-quant kernel for dtype iq4_xs`. That is the
red half of this row's red-and-green pair.

That port has now landed in the same pull request. `DotIQ4XS` sits in
`rocm_quant_dot.hip`'s `WType` system, `DeviceKeepQuantSupported` admits
IQ4_XS as ROCm's twelfth keep-quant format, the codebook seal moved to the
translation unit that owns the tables, and the ROCm copy of those tables is
deleted. The gate is under "Measured on the reconciled tree".

The FMA-contraction risk resolved the same way it did before the move, and it
was re-measured rather than carried: HIP's project-wide `-ffp-contract=off`
reaches this translation unit, and the bit-exact oracle case passes with plain
`*` and `+`.

The fresh end-to-end reload also landed, on the reconciled tree this time: a
measured (not calculated) peak RSS, a pinned sha256, and a CPU-vs-ROCm
token-level comparison producing a byte-identical completion -- see
"End-to-end reload on the reconciled tree, 2026-09-12" under Tests.

Remaining before `DONE`: the `ROCM-KQUANT-NWARPS-DECODE` re-measurement,
`PENDING` for want of a `rocprofv3` setup on `isravale`, and confirming the
checkpoint's exact source revision (Owed; the repo itself is recorded from
the developer's recollection, unverified).
