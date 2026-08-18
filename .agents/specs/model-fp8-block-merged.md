# MODEL-FP8-BLOCK-MERGED — the merged `gate_up` and QKV projections of the block-wise FP8 arm

Issue: [#1189](https://github.com/mudler/vllm.cpp/issues/1189), milestone **M6**.
Row: `MODEL-FP8-BLOCK-MERGED`.
Pinned oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), asserted as the HEAD of the local checkout before
any `file:line` below was read.

## Scope

M4 (`281b4bc76`) made a block-wise FP8 checkpoint run, with one GEMM per
projection. Upstream runs one `MergedColumnParallelLinear` for `gate_up_proj`
and one `QKVParallelLinear` for `q`/`k`/`v`. This row collapses those two groups
into one GEMM each.

1. **`vt::kFp8BlockGateUpSwiGLU` and `vt::kFp8BlockQkv`**, two additive
   `constexpr MergedGemmGroup` descriptors, and `vt::MergedGemmFp8Block`, the
   dispatch that realizes them.
2. **`dense_fp8_block::ResidentFp8BlockMerged`**, the lazily-built, once-only
   N-concatenated device operand — packed bytes and scale grid — plus the guard
   that decides whether a group may be concatenated at all.
3. **`layers::Fp8BlockMlpGateUpMethod`** and the
   `MakeMlpGateUpMethod(..., const Fp8BlockWeight&, const Fp8BlockWeight&, ...)`
   overload — the `MlpGateUpMethodBase` policy layer over that body, the seam
   `AGENTS.md` §"Shared seams" names for a mergeable MLP projection.
4. **The Qwen3.5 dense forward**: `DenseMlpBlock`'s block arm runs ONE gate_up
   GEMM, and `ProjectFullAttnQkv`'s block arm runs ONE QKV GEMM.

**Out of scope**: the mainloop-scaled CUTLASS kernel for `sm_121a` (#1189 M5),
which is the only milestone that needs a GPU; the GDN `in_proj_qkvz` merge (see
`## Owed`); the MoE and GGUF loaders. No CUDA kernel lands here, no GPU is
leased, and no checkpoint is downloaded.

## Why this is SMALLER than the per-tensor merge, established rather than assumed

The per-tensor FP8 merged QKV in this tree (`qwen3_5.cpp::MergedFp8QkvD`) is a
workaround for a limitation block scales do not have. Per-tensor alphas cannot
concatenate: each shard folds its own `input_scale * weight_scale` scalar, so
that arm runs the merged GEMM with `alpha = 1`, applies a per-output-column
alpha vector afterwards (`vt::MulColVecF32` or the cuBLASLt alpha-vector
epilogue), guards on all three shards sharing ONE `input_scale`, and is OFF by
default behind `VT_FP8_MERGED_QKV`.

A block scale is indexed by `n / block_n`, so N-concatenation is exact whenever
each shard's rows begin on a block boundary. Merged row `r` of shard `j` at
local index `l` is `offset_j + l`; its block is `(offset_j + l) / block_n`,
which equals `offset_j / block_n + l / block_n` exactly when `offset_j` is a
multiple of `block_n` — i.e. when every PRECEDING shard's N is. The merged grid
is then the row-concatenation of the shard grids, and
`sum_i cdiv(N_i, block_n) == cdiv(sum_i N_i, block_n)`. No alpha vector, no
shared-scale guard, no opt-in flag.

That is upstream's own rule. `validate_fp8_block_shape`
(`vllm/model_executor/layers/quantization/utils/fp8_utils.py:1197-1244`) raises
for a merged block linear unless every partition except the LAST is a multiple
of `block_n`:

```python
:1234  if not is_tp_split and is_merged_gemm:
:1235      # In case of merged matrices, we allow the last
:1236      # matrix to not be a multiple of block size
:1237      sizes_to_check = output_partition_sizes[:-1]
```

## The guard, and the divergence it records

`adjust_block_scale_shard` (`vllm/model_executor/layers/linear.py:86-95`)
ceil-divides the offset and the size INDEPENDENTLY, which is only consistent
under the rule above. Upstream also carries an escape hatch that DISABLES the
validation whenever any partition is ragged
(`linear.py:532-557`, consulted at `fp8_utils.py:1207`), so at this pin a ragged
NON-final shard is not refused by the validator. It is still unrunnable, and
that is measured rather than argued — running upstream's own
`adjust_block_scale_shard` at the pin over a `QKVParallelLinear` with
`q=256, k=64, v=64` and `block_n=128`, against the scale parameter
`create_fp8_scale_parameter` allocates for it (`fp8_utils.py:1281-1298`,
`cdiv(384,128) == 3` rows):

```text
scale param rows = 3
q: narrow(0, offset=0, size=2)  -> last row 1  in-bounds=True
k: narrow(0, offset=2, size=1)  -> last row 2  in-bounds=True
v: narrow(0, offset=3, size=1)  -> last row 3  in-bounds=False
```

The `v` shard indexes off the end of the parameter. So the hatch moves the
failure rather than enabling the geometry, and there is no correct merged GEMM
for a ragged non-final shard.

**We therefore refuse it by name** rather than splitting silently or slicing
silently. The refusal names the group, the offending shard, its N, and
`block_n`. This is the one deliberate divergence in the row: upstream reaches
the same conclusion one layer later and less legibly, and a silent split would
mean accepting a checkpoint vLLM cannot load, which no token gate can see.

A ragged FINAL shard IS supported, because upstream allows exactly that case,
and the same run confirms the concatenation is what upstream's slicing produces:
for `q=256, k=128, v=100` the shard grids land at rows `[0,2)`, `[2,3)`, `[3,4)`
of a 4-row parameter, and `sum_i cdiv(N_i,128) == 4 == cdiv(484,128)`.

## Upstream, read end to end

| What | Where |
|---|---|
| the model declares BOTH merges, `qkv_proj` and `gate_up_proj` | `vllm/model_executor/models/qwen3_5.py:288-298` (`packed_modules_mapping`) |
| `gate_up_proj` is ONE `MergedColumnParallelLinear`, `output_sizes = [I, I]` | `vllm/model_executor/layers/linear.py:660`, block-scale slice at `:797-801`, `:835-839` |
| `q`/`k`/`v` are ONE `QKVParallelLinear`, shard offsets `0`, `Hq*Dh`, `(Hq+Hkv)*Dh` | `linear.py:1021`, `:1247-1260`, `:1314-1332` |
| the block scale shard slice, ceil-divided on offset AND size | `linear.py:86-95` (`adjust_block_scale_shard`) |
| the merged-partition divisibility rule, last shard exempt | `fp8_utils.py:1229-1244` (`validate_fp8_block_shape`) |
| the escape hatch that skips that validation | `linear.py:532-557`, read at `fp8_utils.py:1207` |
| the scale parameter's `cdiv` rows over the SUMMED partition | `fp8_utils.py:1281-1298` (`create_fp8_scale_parameter`) |
| the apply the merged GEMM still is — quant then block-scaled mm | `kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:97-135` |
| `out_dtype` is `torch.get_default_dtype()`, the model dtype | `fp8.py:284`, `:391-392` |
| the SwiGLU tail after the merged gate_up | `layers/activation.py` `SiluAndMul` |

## Design

### The seam

`AGENTS.md` requires a mergeable MLP projection to route through
`layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`. Both fit, and no
exception is recorded.

`vt::MergedGemmGroup` gains two `constexpr` descriptors and nothing else
changes about the existing one:

```c++
kFp8BlockGateUpSwiGLU = {2, MergedEpilogue::kSiluMulClamp, kNoMergedFastOp, ...}
kFp8BlockQkv          = {3, MergedEpilogue::kNone,         kNoMergedFastOp, ...}
```

`fast_op` is `kNoMergedFastOp` on both, which is the honest value: there is no
fused single-launch kernel for this family, and #1189 M5 owns the CUDA arm. The
existing `vt::MergedGemm` entry point is not widened — its operand list is the
grouped-MoE one (`expert_ids`, `[E*N,K]` towers, a broadcast activation) and a
dense block-FP8 group has none of those. `vt::MergedGemmFp8Block` is the second
realization function under the SAME descriptor type, so the tiering rule
(`fast_op` if registered, else the composite) is expressed once per family
rather than re-derived at a model site.

`MergedEpilogue::kSiluMulClamp` with `limit == +inf` is the plain SwiGLU tail,
as `merged_gemm.h` already records; the realization is `vt::SiluAndMul` over the
contiguous `[M,2I]` merged output, and a finite limit is refused because no
clamped kernel is bound on this arm.

### One body, three instantiations, no copy

The merge helpers are templates on `Dev`/`DBuf` for the reason
`dense_fp8_block_gemm.h` already records: `qwen3_5.cpp` carries its own
anonymous-namespace glue types, so a non-template header would have to be
re-typed into it. `layers::Fp8BlockMlpGateUpMethod::Apply` and the two
`qwen3_5.cpp` call sites are three instantiations of one definition.

### The resident operand

`Fp8BlockMergedResident` is two `mutable std::shared_ptr<void>` handles on the
weight struct, exactly like `d_qkv_fp8_packed` / `d_gate_up_packed` beside them.
`ResidentFp8BlockMerged` builds both ONCE: the shard packed bytes copied
back-to-back into one `[sum N_i, K]` buffer, and the shard scale grids copied
back-to-back into one `[sum cdiv(N_i,bn), cdiv(K,bk)]` buffer. The per-shard
residents are then never built, so the merged arm costs no duplicate device
bytes.

### Output dtype

bf16 at both sites, unchanged from M4, because that is upstream's `out_dtype`
there. The merged QKV emits ONE bf16 `[M, Nq+Nk+Nv]` buffer and the q/k/v views
are row-strided slices of it — the same shape the fp4 merged arm produces, and
the reason the site keeps its `packed_consumers` precondition.

### What the merged QKV still inherits from its site

`ProjectFullAttnQkv` only exposes row-strided Q/K/V views when the fused attn
preamble consumes them (`packed_consumers`); the split `AttnGateSplit` path
assumes contiguous rows. That precondition predates this row and gates the fp4
and per-tensor fp8 merged arms identically, so the block arm carries it too. It
is a property of the CONSUMER, not of the scales, and it is the only condition
under which this row takes the split path.

## Risks

| Risk | Control |
|---|---|
| the merged GEMM is not the split GEMMs — a mis-indexed scale row, an off-by-one concatenation | G1 and G2 are BYTE-IDENTITY, not tolerance: merged output `memcmp`-equal to the split path's, at both sites. A tolerance would hide exactly the misalignment the guard exists for |
| a ragged non-final shard silently mis-slices | G3 requires the named refusal, and asserts the message carries the group, the shard, its N and `block_n` |
| a ragged FINAL shard is refused although upstream allows it | G4 merges one and asserts byte-identity with the split path |
| the merge is wired but nothing selects it | G5 drives `ModelRegistry::Forward` and asserts the per-forward block-GEMM count DROPPED to the merged figure; the reachability mutation deletes the production call site and reds it |
| a silent dequant, or a duplicated device copy of the shards | G6 asserts the merged resident is exactly `sum(N_i)*K` fp8 bytes and `sum(cdiv(N_i,bn))*cdiv(K,bk)*4` scale bytes, and that the per-shard residents were never built |
| the scale grid collapses to one row | G7 perturbs ONE row of ONE shard's grid AFTER the merge point and requires the merged output to move only in that shard's rows |
| the seam is a class nothing constructs | G8 asserts `MakeMlpGateUpMethod` selects it by weight presence and pins the concrete type; the CAPABILITY's reach is G5's, and the class's own binding status is under `## Owed` |
| the bf16 / per-tensor fp8 / fp4 arms regress | G9 is the negative control, plus the whole existing dense suite |

## Tests

`tests/vllm/model_executor/models/test_fp8_block_merged.cpp`, registered in
`tests/CMakeLists.txt`, plus the M4 suite updated for the new GEMM count.

- **G1** merged `gate_up` is BYTE-IDENTICAL to the split gate/up GEMMs plus the
  same SwiGLU tail, over shards whose scale grids differ from EACH OTHER. That
  last condition is asserted, not assumed: the shared fixture's grid is a
  function of position alone, so two same-shaped shards share it and a
  wrong-order concatenation would be invisible.
- **G2** merged QKV is BYTE-IDENTICAL to the three split GEMMs, per shard.
- **G3** a ragged NON-final shard is refused by name, at both group shapes.
- **G4** a ragged FINAL shard merges, and is byte-identical to the split path.
- **G5** reachability: the block-wise fixture through `ModelRegistry::Load` ->
  `Prepare` -> `Forward` on a CPU queue dispatches exactly the MERGED count of
  block GEMMs, and the logits are finite and non-vacuous.
- **G6** the merged resident's byte counts, and that the shard residents stayed
  null.
- **G7** one perturbed scale row moves only the rows of the shard that owns it.
- **G8** `MakeMlpGateUpMethod` selection and concrete type.
- **G9** the negative controls: the bf16 fixture dispatches no block GEMM.

## Gates

| Gate | Command |
|---|---|
| focused | `ctest -R test_fp8_block_merged --output-on-failure` |
| the M4 suite, whose GEMM count this row changes | `ctest -R test_fp8_block_linear --output-on-failure` |
| the M3 load half, unchanged | `ctest -R test_fp8_block_weight_load --output-on-failure` |
| the seam | `ctest -R test_linear_method --output-on-failure` |
| the M1/M2 ops, unchanged | `ctest -R "test_ops_quant_fp8_group_cpu\|test_ops_matmul_fp8_block_cpu"` |
| the merged-GEMM descriptor's existing instance | `ctest -R test_merged_gemm --output-on-failure` |
| the dense forward, unchanged | `ctest -R "test_qwen27_paged_forward\|test_qwen35_paged_forward\|test_qwen27_dense_forward\|test_model_registry"` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |

No GPU lease is taken and none is needed.

## What is gated, plainly

The CPU reference GEMM from #1189 M2 is what makes this checkable without
hardware, and byte-identity against the split path is a strong statement about
the CONCATENATION. It is not a statement about speed and not a statement about
vLLM's tokens.

**There is no CUDA kernel and there is no token gate.** `Qwen/Qwen3.8-27B-FP8`
still refuses at `Prepare` on a CUDA device, because #1189 M5 owns the
mainloop-scaled CUTLASS kernel. No throughput, latency or memory number appears
anywhere in this row, and none may be inferred from it: merging two GEMMs into
one is a launch-count argument, and this arm's only kernel is a scalar CPU
reference that makes no speed claim at all.

## Owed

- **No model binds `layers::Fp8BlockMlpGateUpMethod` yet.** The CAPABILITY is
  reached — `ModelRegistry::Forward` -> the Qwen3.5 dense forward ->
  `dense_fp8_block::Fp8BlockGateUpSwiGLUD` — and the method is the
  `MlpGateUpMethodBase` policy layer over that one body. Its own binding status
  is identical to `Fp8BlockLinearMethod`'s (#1189 M4) and to
  `Nvfp4W4A16MlpGateUpMethod`'s: the models that route through the seam
  (`qwen3.cpp`, `qwen3_vl.cpp`, `voxtral.cpp`) carry no `Fp8BlockWeight` field,
  and giving them one is a loader change those rows own. This is the
  staged-slice exception of `.agents/reachability.md`, named here, in the commit
  body and in the pull request body, and owned by `MODEL-FP8-BLOCK-MERGED` under
  [#1189](https://github.com/mudler/vllm.cpp/issues/1189).
- **The GDN `in_proj_qkvz` merge.** Upstream runs `in_proj_qkvz` as one
  `MergedColumnParallelLinear` and this row leaves the block arm's `in_proj_qkv`
  and `in_proj_z` as two GEMMs. The machinery here covers it unchanged — it is
  the same N-concatenation with the same guard — but the site has its own
  packed-decode dtype predictor (`GdnProjectedMixedQkvDType`) that the merged
  operand would have to agree with, which is a second question this row does not
  answer. Owned by #1189.
- **`process_fp8_weight_block_strategy`** (`BlockScaledMMLinearKernel.py:89-95`)
  may pad or re-lay-out a merged weight for a particular kernel. Nothing here
  needs it, because the CPU reference reads `[N,K]` directly; M5 owes whichever
  transform its kernel wants, and a merged operand is the shape it will see.
- **An f32 KV cache refuses this arm, and every other bf16 arm**
  ([#1249](https://github.com/mudler/vllm.cpp/issues/1249)), unchanged from M4.
- **The column-major and TMA-aligned activation-scale layouts** (#1189 M1's
  `## Owed`), unchanged from M4.

## Stop conditions

Stop and report `NEEDS_DECISION` if any of the following holds.

- The pinned oracle's checkout is not at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`.
- `vt::MergedGemmGroup` cannot describe the group. It can, and §Design says how;
  if that turns out to be wrong the answer is to EXTEND the descriptor or record
  one exact tracked exception, never to write a second path.
- Byte-identity between the merged and split arms cannot be achieved. A
  tolerance is NOT the fallback: a merged GEMM that is only approximately the
  split one is a mis-concatenation, and the row should stop rather than widen.

Stop and report `NEEDS_CONTEXT` if the work requires a GPU lease or a checkpoint
download. The row is scoped so that it needs neither.

## Evidence

CPU only, no GPU lease, no checkpoint. Default configure
(`cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON`), which is CI's, so
`NDEBUG` is not defined and asserts are live.

**RED before.** `test_fp8_block_merged`, the dispatch-count case alone, against
the unmerged tree: `CHECK( 13 == 9 )`, `1 failed`, `Status: FAILURE!`.

**GREEN after.** `test_fp8_block_merged` 9 cases / 852 assertions / 0 failed,
`Status: SUCCESS!`. `test_fp8_block_linear` 8 / 66 / 0, unchanged in shape from
before the row (its expected count moved 13 -> 9, which is the merge).

**Mutations.** Each printed `compile_rc` and the mutation delta before the
verdict, ran the focused gate, restored from a tar snapshot and verified the
restore by `sha256sum`. `git checkout -- .` was not used, because it reads the
index rather than the working tree.

| mutation | compile_rc | result |
|---|---|---|
| the merged `gate_up` production call site deleted (M4's split pair restored) | 0 | RED, the reachability case, 9 -> 10 GEMMs |
| the merged QKV production call site deleted | 0 | RED, the reachability case, 9 -> 11 GEMMs |
| the ragged-non-final guard neutered (`if (false)`) | 0 | RED, the refusal case, 5 assertions |
| the scale-grid concatenation reversed | 0 | RED, 4 cases: both byte-identity cases, the ragged-final case, the row-mapping case |
| the packed-byte concatenation reversed | 0 | RED, 3 cases |

**A measured weakness in the fixture, found by a mutation that did NOT red.**
The scale-grid reversal first left the `gate_up` byte-identity case GREEN.
`MakeFp8Block`'s grid is `0.0625 + 0.125*((r*3 + c*5) % 5)`, a function of the
POSITION only, so two shards of the same shape get byte-identical grids whatever
their seeds — and a wrong-order concatenation of two identical grids is
invisible. The suite now gives each shard its own grid and asserts that
precondition (`RequireScalesDiffer`) before comparing anything. This is recorded
because the first run of that mutation is exactly what a green gate over nothing
looks like.

**What is NOT established.** No token gate, no CUDA kernel, no throughput,
latency or memory number. The model-level case asserts the TOPOLOGY (one GEMM
per merged group) and the forward's health and bit-reproducibility; the merged
GEMM's VALUES are gated at the GEMM boundary, byte-for-byte against the split
arm.

## Outcome

**What the row measured.** The merge is exact: `memcmp`-equal to the split arm
at both group shapes and at a ragged final shard, over shards whose scale grids
differ. Nine block GEMMs per forward where M4 ran thirteen.

**What was rejected, and why.**

- *Falling back to the split path when a group is not concatenable.* Rejected.
  Upstream's `validate_fp8_block_shape` declares that geometry invalid and its
  scale-parameter slicing indexes off the end of the parameter, so a split
  fallback would accept a checkpoint vLLM cannot load. No token gate can see
  that, which is what makes it a divergence rather than a convenience.
- *An opt-in flag, mirroring `VT_FP8_MERGED_QKV`.* Rejected. That flag exists
  because the per-tensor merge is only byte-identical under a shared
  `input_scale` and needs an alpha vector; neither is true here, so a flag would
  name a choice with no second side.
- *A tolerance-based comparison.* Rejected as a stop condition in the spec and
  never used. A scale row read one index off produces SMALL errors.
- *Widening `vt::MergedGemm` to carry this group.* Rejected. Its operand list is
  the grouped-MoE one and a dense block-FP8 group has no `expert_ids`, no
  `[E*N,K]` towers and no broadcast activation. The descriptor is shared; the
  realization function is a sibling.
- *Merging the GDN `in_proj_qkvz` in the same change.* Deferred, under
  `## Owed`. The machinery covers it, but the site has its own packed-decode
  dtype predictor that the merged operand has to agree with, which is a second
  question.

**Why each default has its value.** The merge is unconditional because block
scales concatenate losslessly and there is nothing to trade. `out_dtype` is bf16
at both sites because that is upstream's `out_dtype` there. `fast_op` is
`kNoMergedFastOp` on both descriptors because no fused kernel exists, and naming
one that does not would make the tiering rule a lie.

## Now

`DONE` — M6 of #1189. M1 (`ad5f175e7`), M2 (`770e49486`), M3 (`09597106e`) and
M4 (`281b4bc76`) are `DONE`; **M5 is open and is the only milestone that needs a
GPU**, and until it lands a CUDA device refuses this checkpoint at `Prepare`.
