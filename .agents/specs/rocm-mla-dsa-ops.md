# ROCm MLA/DSA op campaign — the missing native kernels on `gfx1151`

Row: `BACKEND-ROCM`.
Issue: [#2715](https://github.com/mudler/vllm.cpp/issues/2715).
Base SHA: `ca07f6e948e44958b355c7d0f229546e21d9ba06`.

## Scope

Price every op #2715 names, decide whether the work is one wave or a campaign,
and land W1. It does **not** land `kMlaDecodeAttention` or the DSA indexer pair;
§Split says why those are their own waves, and §Owed records them.

## What #2715 asks, and the two corrections this spec makes to it

#2715 lists eight ops with no ROCm registration. The list is **correct** — I
re-derived it at the base SHA above, both ways the issue derived it: a direct
search of `src/vt/rocm/` finds none of the eight, and enumerating every `OpId::`
token under `src/vt/rocm/` yields 53 distinct ops (the issue said 51; the tree
moved), none of them these.

Two premises around that list do not survive reading the tree.

### Correction 1 — all eight have a CUDA sibling to mirror

`.agents/specs/rocm-glm53-dsa.md` §W1.3 marks `kConcatAndCacheMla`,
`kMlaDecodeAttention` and `kFusedNormRope` as having no CUDA registration, and
§W1.5 rests part of its "this is a campaign" verdict on *"two of which
(`kMlaDecodeAttention`, `kConcatAndCacheMla`) have no CUDA registration to mirror
either"*. That is false, and it was already false at that spec's own base SHA
`11fed3ba`:

| Op | CUDA registration |
|---|---|
| `kConcatAndCacheMla` | `src/vt/cuda/cuda_cache.cu:308` |
| `kMlaDecodeAttention` | `src/vt/cuda/cuda_mla_attn.cu:806` |
| `kFusedNormRope` | registered on CUDA; `src/vt/rocm/rocm_mla_fused_norm_rope.hip:27` names `cuda_ops.cu:1163-1245` as its donor |

Verified with `git show 11fed3ba:src/vt/cuda/cuda_mla_attn.cu | grep -n
'OpId::kMlaDecodeAttention, DeviceType::kCUDA'` (line 806) and the same for
`cuda_cache.cu` (line 308). The verdict survives — this is still a campaign —
but for the reason in §Split, not for a missing donor. Every one of the eight has
a structural donor in this tree.

### Correction 2 — the list of eight is not complete

Two more ops are ROCm-unregistered, have both a CUDA and a CPU sibling, and are
on an MLA path: `kMlaPrefillAttention` and `kMergeAttnStates`. The first is not
a guess. The `kFusedNormRope` landing commit `9f3e6e223` names the ops its
GLM-5.3 run on `strix:gpu0` observed on the reference tier, and its list is
**five**: `ConcatAndCacheMla`, `ConcatMlaNopeRope`, **`MlaPrefillAttention`**,
`BatchedMatmul`, `MlaDecodeAttention`. So `kMlaPrefillAttention` was measured on
this model, on this board, and #2715 does not list it.

It is not scoped here and it is not a W2 or W3 item either. Its CUDA arm is
inside `#ifdef VLLM_CPP_FLASH_ATTN` (`cuda_mla_prefill.cu`) and is the vendored
FA-2 launcher, so there is nothing to mirror: a ROCm arm is a new kernel against
the CPU reference, not a port. `kMergeAttnStates` is a scalar elementwise kernel
(`cuda_mla_prefill.cu:365-450`) and is cheap, but no ROCm run has yet observed
it. Both are recorded under `## Owed`.

### Correction 3 — `kFusedChain` cannot disqualify a default-configuration run

`kFusedChain` is the **Tier-1 interpreter**, and `src/vt/ops.cpp:1287-1298`
reaches it only when `RecipeIsTier1Able(recipe) && FusedTier() == 1`.
`FusedTier()` reads `VT_FUSED_TIER` and returns 0 unless the value is exactly
`1` (`include/vt/fused_recipe.h:176-179`). Unset, every `vt::FusedChain` call
takes the Tier-0 composite, which is device-agnostic and dispatches to the
standalone ops — all of which ROCm already registers.

So on the default configuration `OpId::kFusedChain` is never resolved on ROCm,
never installs the reference tier, and never increments
`GetReferenceTierHits()`. `VT_FUSED_TIER=1` is additionally a **rejected**
performance arm on every backend that measured it — `.agents/benchmark-record.md`
records "REJECTED - VT_FUSED_TIER=1" for CUDA and Metal and "`VT_FUSED_TIER`
stays 0" for Vulkan.

**Seven ops gate the ROCm speed axis for GLM-5.3, not eight.** `kFusedChain` is
real work and worth doing, but landing it moves nothing and not landing it
blocks nothing. It is recorded under §Owed as non-gating rather than carried in
the campaign's critical path.

## Pricing

Measured by reading each CUDA donor at the base SHA. "Donor" is the span that
has to be mirrored; "HIP est." adds the file header, the `Check`/`AsStream`
boilerplate every `src/vt/rocm/*.hip` file carries, and the registrar.

| Op | Donor | Donor lines | Kind | HIP est. | Test est. | Risk |
|---|---|---|---|---|---|---|
| `kConcatMlaNopeRope` | `cuda_mla_attn.cu:752-802` | 50 | pure copy, 1 kernel | 60 | 80 | none — bit-exact by construction |
| `kConcatAndCacheMla` | `cuda_cache.cu:243-297` | 55 | pure copy, 1 kernel | 65 | 90 | none |
| `kGatherMlaCache` | `cuda_mla_prefill.cu:302-361` | 60 | pure gather, 1 kernel | 70 | 110 | none |
| `kBatchedMatmul` | `cuda_matmul.cu:605-690` | 85 | cuBLASLt strided-batch wrapper | 120 | 100 | low — row-major NN operand swap; the donor pattern exists at `rocm_matmul_hipblaslt.hip:589-618` |
| `kFusedChain` | `cuda_ops.cu:3670-3800` | 130 | recipe interpreter, tree reduce | 160 | 0 (arm exists at `test_backend_cross_device.cpp:1097`) | low — the tree reduce is wavefront-agnostic |
| `kDsaIndexerLogits` | `cuda_dsa_indexer.cu:70-150,262-290` | 110 | warp-reduce dot | 130 | 130 (pair) | medium — 32-lane shuffle tree; `warpSize` is 32 on `gfx1151` so it ports, but the reduction ORDER is the gate |
| `kDsaTopkSelect` | `cuda_dsa_indexer.cu:150-260,292-305` | 125 | block selection, O(topk·n) | 140 | (in pair) | medium — the tie rule (larger logit, then SMALLER index) and ASCENDING emission are load-bearing for dense-identity |
| `kMlaDecodeAttention` | `cuda_mla_attn.cu:140-750` less the concat | 560 | 2-stage split-KV flash decode | 600 | 200 | **high** — see below |
| **Total** | | **1,175** | | **1,345** | **710** | |

`kMlaDecodeAttention`'s risk is not its line count. It carries four things none
of the others do:

1. **Three optional arms**, each its own gate: the sliding window
   (`win_left`), the DSA selected-slot list (`sel`/`sel_cnt`, whose
   full-selection case must reproduce the dense reduction *bit for bit*), and
   the per-head attention sink (which stage 2 must count once per row, not once
   per split — `cuda_mla_attn.cu:396-406` says a gate run only at
   `num_kv_splits == 1` would pass a kernel that is wrong everywhere else).
2. **A shared-memory ceiling this backend has not measured.** The donor calls
   `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)` above 48 KiB
   (`cuda_mla_attn.cu:566-570`). At GLM-5.3 geometry (`head_size` 576) the
   request is `(16 + 8) * 576 * 4 = 55,296 B`. HIP has no equivalent opt-in, so
   whether `gfx1151` accepts it is a **measurement**, not a derivation, and the
   `kNTileSmall` fall-back exists precisely for when it does not.
3. **Device capabilities ROCm does not currently probe.** `LaunchMlaDecode`
   needs `multiprocessor_count` for the split heuristic. `DeviceCaps` in
   `src/vt/rocm/rocm_backend.hip:68-77` carries no such field and is file-local.
4. **Capture-safe scratch.** The `[B, Hq, splits, Dv+1]` f32 workspace is
   grow-only and retired-not-freed because a captured graph may have baked the
   pointer. ROCm has the precedent (`rocm_exl3.hip:361-380`) but the sizing
   argument (`alloc_splits` at the constant bound, so the buffer never grows
   mid-run) has to be re-made against `hipGraph` capture, which ROCm does
   support (`rocm_backend.hip:314-355`).

## Split — this is a campaign, and here is the shape

**A partial port moves nothing on the speed axis.** One reference-tier hit
disqualifies a run (`docs/ROCM.md:60-61`), so the axis stays void until the last
gating op lands. That is an argument about *measurement*, not about *review
size*: 1,345 lines of new HIP across eight kernels in one pull request is not
reviewable, and the fresh-reviewer mutation loop — rebuild per mutation, on a
leased box, over eight kernels' worth of guarantees — is where it would stall.

Three waves. Each is complete and gated on its own; only the third moves the
axis.

| Wave | Ops | HIP | Why together |
|---|---|---|---|
| **W1** (this spec) | `kConcatMlaNopeRope`, `kConcatAndCacheMla`, `kGatherMlaCache`, `kBatchedMatmul` | ~315 | Every one is on the MLA path on the default configuration, and every one is bit-exact against the CPU oracle — a copy, a gather, and a GEMM whose donor pattern this backend already has. No numerical judgement calls, so the wave can be gated to equality rather than to a tolerance. |
| **W2** | `kDsaIndexerLogits`, `kDsaTopkSelect` | ~270 | The pair is one unit: the top-k consumes the logits the first op writes, and the tie/emission rule is only checkable end to end. |
| **W3** | `kMlaDecodeAttention` | ~600 | The four risks above. Landing it closes the disqualification, so the speed axis is measurable for the first time at W3 and not before. |

`kFusedChain` is not in any wave — see Correction 3.

## W1 — what this pull request lands

Four ops, in one new file `src/vt/rocm/rocm_mla_ops.hip` plus the batched GEMM
beside its siblings in `src/vt/rocm/rocm_matmul_hipblaslt.hip`.

### Upstream anchors

vLLM at the pinned oracle `5559679229` (`.agents/upstream-sync.md`).
`GlmMoeDsaForCausalLM` maps to `vllm/model_executor/models/deepseek_v2.py:1930`.

| Op | Upstream | Structural donor in this tree |
|---|---|---|
| `kConcatAndCacheMla` | `vllm/csrc/libtorch_stable/cache_kernels.cu:401-442` `concat_and_cache_mla_kernel`, launched at `:899-900` | `src/vt/cuda/cuda_cache.cu:243-297` |
| `kConcatMlaNopeRope` | `vllm/csrc/.../cache_kernels.cu:1572-1584` + `concat_mla_q.cuh:13,21-24,50-53` | `src/vt/cuda/cuda_mla_attn.cu:752-802` |
| `kGatherMlaCache` | `vllm/csrc/libtorch_stable/cache_kernels.cu:992-1064`, launched at `:1142-1145` | `src/vt/cuda/cuda_mla_prefill.cu:302-361` |
| `kBatchedMatmul` | `torch.bmm` as vLLM calls it in the MLA projections (`vllm/model_executor/layers/mla.py`); no vLLM kernel — the reference is the BLAS contract | `src/vt/cuda/cuda_matmul.cu:605-690`, and the strided-batch call shape at `src/vt/rocm/rocm_matmul_hipblaslt.hip:589-618` |

### Design

The three MLA data ops are **pure copies**. They are mirrored on the raw storage
word (`uint16_t`/`uint32_t`) rather than on a float type, which is the house
precedent for a copy — `ConcatAndCacheMlaKernel` in the CUDA donor already does
exactly this (`cuda_cache.cu:295-305`) and says why: "the auto path is a
bit-exact element copy, so Word is the raw storage type and no dtype conversion
appears". Word-typing makes f32/bf16/f16 two cases instead of three and makes
bit-exactness structural rather than argued.

Every kernel keeps the donor's index arithmetic element for element, including
the `slot < 0` padded-token skip (`cuda_cache.cu:253`) and the
`token_id >= batch_end` early return (`cuda_mla_prefill.cu:316`), because those
are upstream's own edge cases and a gate that never feeds them proves nothing.

`kBatchedMatmul` is `hipblasGemmStridedBatchedEx` with the row-major operand
swap this file already uses for the BT batched lane: for row-major
`C[g] = A[g] @ B[g]`, the col-major view is `C^T = B^T · A^T`, so the call is
`(OP_N, OP_N, m=N, n=M, k=K)` with `B` first at `ldb = b.stride[1]`, `A` second
at `lda = a.stride[1]`, and `ldc = out.stride[1]`, batch strides taken from each
operand's `stride[0]`. The `k == 0` case memsets, matching the donor, and a
strided (non-dense) `out` is zeroed row by row rather than wholesale.

### Tests and gates

New arms in `tests/vt/test_backend_cross_device.cpp`, in the file's existing
style: build the inputs on the host, run the CPU oracle, run the same call on
every registered non-CPU device, compare.

Each arm asserts **three** things, and the middle one is the one that matters:

1. The device result equals the CPU oracle. For the three copies this is
   **bit-exact** equality on the raw words, not an NMSE bound. For
   `kBatchedMatmul` it is NMSE <= 5e-4, the bar the existing GEMM arm uses.
2. `vt::OpRegistered(op, device)` is **true**. This is the load-bearing
   assertion. `OpRegistered` is a native-only probe by design
   (`src/vt/op_provider.cpp:788-806`), so it is false while the op serves from
   the reference tier — and the numeric assertion alone **cannot** see the
   difference, because the reference tier computes the same answer on the host.
   A test without this assertion is green today, before any kernel exists.
3. `vt::GetReferenceTierHits()` does not increase across the call. This is the
   same quantity `docs/ROCM.md:60-61` disqualifies a performance result on, so
   the gate and the disqualification read the same number.

The call goes through the `vt::` seam (`vt::ConcatAndCacheMla` etc.), which is
the same `ops.cpp` chokepoint `mla_attention.cpp:952` calls, not through a
hand-constructed provider entry.

### Reachability

The production entry points are
`src/vllm/model_executor/layers/attention/mla_attention.cpp:364,952,1009,1078`,
reached from `ModelRegistry::Forward`. Nothing new is added to the model layer:
this wave changes which implementation `GetOp` returns for a call the model
already makes on every MLA step. The mutation that proves it is deleting the
`RegisterOp` line — assertion 2 must go red, and it must go red *while assertion
1 stays green*, which is the whole point.

### Stop conditions

- A copy op that is not bit-exact stops the wave. It is a pure copy; anything
  but equality means the index arithmetic is wrong, and widening to a tolerance
  would hide it.
- `MoeSiluMul` bf16 is a **known standing red** on `gfx1151` (±1 ULP vs the CPU
  oracle, #1954/#1513, reproduced on this board by
  `.agents/specs/rocm-glm53-dsa.md` §W3). It is pre-existing and this wave
  touches no arithmetic path. It is reported, never tolerated into green.
- No speed number is admissible from this wave. Three of the seven gating ops
  are still unregistered after it lands, so `GetReferenceTierHits()` on a
  GLM-5.3 run stays non-zero and the axis stays **VOID**.

## Evidence (`strix:gpu0`, rc job `be04b292`, 2026-09-03)

gfx1151, ROCm/HIP 7.2.53211-97f5574fe2, `-DVLLM_CPP_HIP=ON
-DVLLM_CPP_HIP_ARCHITECTURES=gfx1151`, `ninja -j 4`, `cmake rc=0` with
`ROCm backend: ENABLED for arch(es) [gfx1151]` read off the configure log rather
than assumed. **No `ccache`**, deliberately: `.agents/specs/rocm-glm53-dsa.md`
§W5 records three runs on this board where a ccache-enabled HIP build failed to
link `vllm-cli` and a ccache-free build of the identical source linked rc=0
(#2506). The host usage sheet asks for ccache; that measurement outranks it here.

RED is not "the tests with the kernels commented out". It is the tree at the
spec commit `a1be8c1dc` — no `rocm_mla_ops.hip` on disk, zero `kConcatAndCacheMla`
matches under `src/vt/rocm/`, both printed by the job from the tree — carrying
only the new test file.

| Arm | `ninja` | test cases | assertions |
|---|---|---|---|
| **RED** (before the kernels) | rc=0, 271 s | 5 / 4 passed / **1 failed** / 32 skipped | 11 / 7 passed / **4 failed** |
| **GREEN** | rc=0, 260 s | 5 / **5 passed** / 0 failed / 32 skipped | 46 / **46 passed** / 0 failed |
| GREEN, whole suite | — | 37 / 36 passed / 1 failed | 83,849 / 83,848 passed / 1 failed |
| restored after every mutation | rc=0, 3 s | 5 / 5 passed / 0 failed | 46 / 46 passed / 0 failed |

**Read the two counts together.** RED's four failures are the four
`OpRegistered` CHECKs, by name in the log, against a NON-ZERO assertion count —
so the cases asserted and failed rather than throwing. And RED's other four
cases **passed**, on 7 assertions that are all `REQUIRE(ref != seed)`: the
device loop bodies never ran, because `OpAvailable` was false. That is the
spec's §Tests claim measured rather than argued — the oracle-equality
assertions are green with no kernel at all. GREEN's assertion count rises 11 ->
46, which is those loop bodies executing.

The one full-suite failure is `test_backend_cross_device.cpp:2278`,
`CHECK(got == ref_b)` inside "MoeSiluMul matches the CPU oracle" — the bf16 ±1
ULP red #1954/#1513 track, already reproduced on this board by
`.agents/specs/rocm-glm53-dsa.md` §W3. This wave touches no arithmetic on that
path. It is reported, not widened into green.

### Mutations

Each patches one guarantee, **rebuilds** (`ninja rc=0` printed for every arm, so
no mutation is a build failure wearing a test result), runs, restores, and
proves the restore by sha256 against a manifest taken from the pristine tree
before any mutation. `ALL FILES RESTORED BYTE-FOR-BYTE` at the end.

| # | Mutation | Result | assertions |
|---|---|---|---|
| M1 | delete `RegisterOp(kConcatAndCacheMla, kROCM)` | **KILLED** | 41 / 40 passed / 1 failed |
| M2 | `if (slot < 0) return;` -> `slot < -1000000` (drop the padded-token skip) | **SURVIVED** | 46 / 46 passed / 0 failed |
| M3 | `broadcast = rope.shape[1] == 1 && heads > 1` -> `false` | **KILLED** | 46 / 45 passed / 1 failed |
| M4 | drop the `seq_starts` offset in the gather | **KILLED** | 46 / 45 passed / 1 failed |
| M5 | `lda = a.stride[1]` -> `a.shape[2]` in the batched GEMM | **KILLED** | 46 / 45 passed / 1 failed |

M1 is also the reachability proof this wave owes. Deleting the registration
reds assertion 2 while the numeric case goes back to passing vacuously (41
assertions, not 46) — the two measure different things, and only one of them
can see a missing kernel.

**M2 SURVIVED, and the defect was the test.** `slot == -1` gives `block = -1/4 =
0` and `offset = -1 % 4 = -1` under C++ truncation, so the entry address is
NEGATIVE: a kernel with the skip removed writes BEFORE the cache, never inside
it, and every word the case compared was still correct. The case now brackets
the cache with guard bands and asserts both untouched, so an out-of-range write
is seen.

**The re-measurement is PENDING, not a pass.** It is rc job `e88fd335` on
`strix:gpu0`, submitted 2026-09-03 and still queued behind another row's job
when this was written. Until it reports, this wave has four killed mutations and
one repaired-but-unre-measured case, and nothing here says otherwise. Its output
lands at `/workspace/rocm-mla-2715/out2/` (`job2.log`, `probe.log`, `m2b.log`,
`m2b_restored.log`) and is readable with `rc logs e88fd335`; the job builds the
BRANCH HEAD bytes, not the tarball job1 used, and prints each file's sha256
against `head.sha256` so the tree it measured is nameable.

### What the W1 gate does NOT cover, stated rather than implied

`GatherMlaCacheKernelHip` carries upstream's `token_id >= batch_end` early
return (`cache_kernels.cu:1019`), and **no case in this wave makes it fire**.
The grid is `num_tokens` and the last request's `cu_seq_lens` entry equals
`num_tokens`, so `token_id < batch_end` always holds and the branch is dead in
the gate. It is ported because it is upstream's, not because it is checked;
exercising it needs a `token_to_seq` that names a request the token is past,
which is the ragged-batch shape this harness does not build. Recorded here so
nobody reads "four killed mutations" as coverage of every line.

### Also pending on `e88fd335`: what an unregistered ROCm op actually does here

#2715 says the missing ops are a slow path on gfx1151 and a refusal elsewhere.
That may be stale by one same-day commit. `6b97a6800` (#2511) narrowed the
managed-allocation branch to `PageableMemoryAccess == 1` and made the unified
claim FOLLOW the allocator (`include/vt/rocm/rocm_arch.h:180-195`); gfx1151
reports that attribute 0; and `docs/ROCM.md:83-85` states outright that gfx1151
and gfx1103 therefore get "plain `hipMalloc` and no reference tier". The
`9f3e6e223` GLM-5.3 run that observed five ops ON the tier is NOT an ancestor of
that commit — `git merge-base --is-ancestor 6b97a6800 9f3e6e223` answers no — so
it measured the older allocator.

If the board confirms it, the missing MLA ops are a **REFUSAL** on this part and
W2/W3 block GENERATION, not only measurement. **This spec does not assert that.**
The branch adds a case that asks `vt::ReferenceTierEligible(kROCM)` — the public,
side-effect-free safety gate — and asserts the consequence in both directions, so
whichever answer the board gives is measured rather than derived.

## Now

W1 in flight. W2 and W3 unclaimed.

## Owed

- `kMlaDecodeAttention` on ROCm — W3. Owner `BACKEND-ROCM`, issue #2715.
- `kDsaIndexerLogits` + `kDsaTopkSelect` on ROCm — W2. Owner `BACKEND-ROCM`,
  issue #2715.
- `kMlaPrefillAttention` on ROCm — NOT a port. Owner `BACKEND-ROCM`, issue
  #2715. Its CUDA arm is the vendored FA-2 launcher behind `VLLM_CPP_FLASH_ATTN`,
  so a ROCm arm is a new kernel written against the CPU reference. Measured on
  the reference tier by `9f3e6e223`'s own GLM-5.3 run and absent from #2715's
  list.
- `kMergeAttnStates` on ROCm — a scalar elementwise kernel, cheap, but no ROCm
  run has observed it. Owner `BACKEND-ROCM`, issue #2715.
- `kFusedChain` on ROCm — **non-gating**. Owner `BACKEND-ROCM`, issue #2715.
  Reached only under `VT_FUSED_TIER=1`, which is a rejected performance arm on
  every backend that measured it, so it cannot disqualify a default run.
- The ROCm speed axis for GLM-5.3 stays **VOID** until W2 and W3 both land.
