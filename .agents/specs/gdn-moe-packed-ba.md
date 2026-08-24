# GDN-MOE-PACKED-BA — the MoE loader builds the one merged `in_proj_ba` owner, so packed GDN decode reaches the 35B

Row: `GDN-MOE-PACKED-BA`
Issues: [#1169](https://github.com/mudler/vllm.cpp/issues/1169) (primary — the
merged `in_proj_ba` owner is dense-only, so `has_packed_ba` is false on every
MoE checkpoint),
[#1793](https://github.com/mudler/vllm.cpp/issues/1793) (owed — the GGUF MoE
loader keeps the shards split)
Base SHA: `5d638b67e`
Upstream pin: see [`.agents/upstream-sync.md`](../upstream-sync.md)
(`5559679229bc961848b121ccdeaa8fa5d79bec98`).
Parent row: `GDN-MOE-BF16-OUT` ([spec](gdn-moe-bf16-out.md)), which found this
gap while correcting its own premise and listed it under `## Owed`.
Matrix: [`.agents/kernel-matrix.md`](../kernel-matrix.md).

## Now

**2026-08-24, graphed gates PASSED at the DEFAULT env.** T3 on merged main
`732e9ddf8` (#1741 + #1859 landed): gates (a) and (b) pass GRAPHED with no
environment variables -- default arm token-exact `packed_launches=0`, levers
arm token-exact `packed_launches=60` (capture dispatches), rollback
token-exact -- and the weights gate holds on the real shard. Evidence table
in [fix-fp8-plan-capture.md](fix-fp8-plan-capture.md) `## Evidence`. Still
owed before `DONE`: the same-binary TPOT A/B, gate (c) on the bf16 35B, and
the GGUF arm (#1793).

**2026-08-24, GPU half measured in a `dgx:gpu0` lease (`## Evidence`).** The
weights gate passes on the real 35B NVFP4 shard; eagerly, the default arm is
token-exact with `packed_launches=0`, the two-lever arm is token-exact with
**`packed_launches=450 triton_launches=450`** (15 decode steps x 30 GDN
layers: the vendored FLA cubin ran on every GDN decode step of a MoE checkpoint
for the first time), and the rollback arm is token-exact with 0. The GRAPHED
production arm throws at the first decode-graph capture on every arm, and so
does `main` without this row in the same lease: it is
[#1732](https://github.com/mudler/vllm.cpp/issues/1732) (cuBLASLt 13.3
heuristic inside capture, first failing call the MoE router GEMM), not this
row, and PR #1741 cherry-picked does not clear it alone. A follow-up spike
(2026-08-24, `logs/spike-1741-fp8.log`) measured the remaining gap as #1741's
own `## Owed` fp8 item: with #1741 the bf16/f32 lanes stop querying in capture
and the failure moves to the fp8 cuBLASLt lane (`VT_FP8_PLAN_CACHE` default
OFF, queried per call); **#1741 + `VT_FP8_PLAN_CACHE=1` passes the graphed
gate token-exact on all three arms** (default; levers with
`packed_launches=60`; rollback), and either alone fails. The row stays
`ACTIVE`: the graphed production-default gate, the TPOT A/B and gate (c) are
`PENDING` on #1741 landing together with its owed fp8-lane extension (or the
default flip it names), which is that row's work, now measured for it.

`READY` → `ACTIVE`: the CPU half is implemented on `row/GDN-MOE-PACKED-BA`
(product tree `6ae3028d7` + the implementation commit on top). `LoadGdn` now
builds the one merged `in_proj_ba` owner, the split fields stay empty, and the
consumer comments that called the owner dense-only are rewritten. Test 1 went
red at the spec commit and green at the head; Test 2 measured bit-identity on
CPU (max abs difference 0 on prefill and decode); the full CPU ctest is green
apart from one pre-existing link failure named under `## Evidence`.

What stays `PENDING`, and who runs it: the GPU gates in `## Gates` are
operator gates on `dgx:gpu0` inside an `rc` lease. They are split by checkpoint
and arm, because the loader term is now satisfied on every MoE safetensors
checkpoint but it was not the only term on the NVFP4 35B. On
`nvidia/Qwen3.6-35B-A3B-NVFP4` the GDN tower is native FP8 on the default arm
(`LoadGdn` takes the `DenseNativeEnabled()` branch and fills `in_proj_qkv_fp8`;
`tests/vllm/test_qwen36_weights.cpp:186` pins it), so `gdn_fp8_tower` is true
at the eligibility (`qwen3_5.cpp:4762-4763`) and its `dtype_compatible` term
is false at the default for two reasons: `(!gdn_fp8_tower ||
PackedGdnDecodeFp8TowerEnabled())` with `VT_GDN_PACKED_DECODE_FP8_TOWER`
default OFF (#365, `PERF-27B-GDN-PACKED-REACHABLE`), and the predicted
`mixed_qkv` dtype is F32 unless `VT_GDN_FP8_IN_BF16=1`
(`GdnFp8MergedInProjDType`), against the BF16 pin in
`GdnPackedDecodeDTypesCompatible`. The NVFP4 35B default arm therefore reads
`packed_launches 0/0` with or without this change, and reaches the packed leg
only under `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1`, the same
two default-OFF levers the 27B NVFP4 bridge `PERF-GDN-PACKED-BRIDGE` used. On
the bf16 35B (`Qwen/Qwen3.6-35B-A3B`: bf16 GDN tower, split bf16 qkvz arm,
`mixed_qkv` BF16 by `GdnInDType()` at the default) the default arm reaches the
packed leg after this change. The loader change is still not inert on the
NVFP4 35B default arm: `ProjectGdnBA` now runs one merged `MatmulBTRawD` GEMM
(F32 out when the packed leg is not selected) instead of two split
`MatmulF32D` GEMMs, so the 315/315 token-exact golden there gates the loader
change itself, not the packed leg. The two real-shard subcases in
`tests/vllm/test_qwen36_weights.cpp` now assert the merged owner on the 35B
NVFP4 shard; they skip on the CPU host (shard absent) and must run on the GPU
host. The row does not reach `DONE` until those land. While preparing the GPU
gate the operator found that `tests/parity/test_qwen36_paged_engine.cpp` still
pinned the pre-#1169 dense-only contract (`packed_launches != 0` threw), which
made gate (b) unpassable; this branch repairs it to derive the expectation from
the subset of terms that vary across this spec's three gate arms — the
runtime lever (`VT_GDN_PACKED_DECODE`), the two #365 levers, the merged-qkvz
env selection, and the predicted mixed dtype — with the rest pinned by the
checkpoint and named in the test's cannot-see paragraph; it asserts the
counter's sign rather than an exact count because `generate()` runs under
CUDA-graph capture and replay. A scoped review caught the missing
runtime-lever term in the first derivation before any GPU run: without it,
gate (b)'s rollback sub-arm would have failed on a correct binary, and a
CPU-tier truth-table case in the test now pins the derivation.

The gap was verified against the base SHA before implementation: `in_proj_ba`
was written at one site in the tree
(`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:514`), the MoE
safetensors loader loaded the two shards split
(`src/vllm/model_executor/models/qwen3_5_weights.cpp:746-747` at `5d638b67e`),
and the eligibility predicate still reads `!w.in_proj_ba.Empty()`
(`src/vllm/model_executor/models/qwen3_5.cpp:4779`). No open pull request
touched either site. The packed leg is default-ON
(`PackedGdnDecodeRuntimeEnabled`, `MergedGdnBaEnabled`, both `qwen3_5.cpp`).
On the bf16 35B the loader was the only term keeping the vendored FLA cubin
unreached at the default. On the NVFP4 35B it was one of two terms; the other
is the #365 fp8-tower term described above, which this row does not touch.

Pull request shape: one pull request, spec commit first. No preference is
recorded for this row and no split case applies (the implementer works on the
row branch where the spec commit precedes the code). This is the repository
default, not an inferred preference.

## Scope

One loader change and the records it invalidates.

- **E1.** `LoadGdn` in `src/vllm/model_executor/models/qwen3_5_weights.cpp`
  builds `g.in_proj_ba` with `LoadMergedBf16RawNK(get, {la + "in_proj_b.weight",
  la + "in_proj_a.weight"})` and leaves `g.in_proj_b` / `g.in_proj_a` empty,
  exactly as `LoadGdnDense` does at `qwen3_5_dense_weights.cpp:514`. The change
  is independent of the tower dtype branch: `in_proj_b` and `in_proj_a` are bf16
  on every Qwen3.5/3.6 checkpoint (the `modelopt_mixed` 35B keeps them on the
  ignore list, notes §3.6 cited at `qwen3_5_dense_weights.cpp:486`), which is
  why the dense loader merges them unconditionally too.
- **E2.** The memory plan `PlanQwen3_5MoeLoad` (`qwen3_5_weights.cpp:1478-1479`)
  keeps planning the same bytes. If the planner is keyed on tensor names that a
  test cross-checks against the loaded owners, plan the merged owner once under
  its own name; otherwise leave the two entries as they are. The implementer
  reads the planner before deciding and says which in the commit body.
- **E3.** Repair the stale sentence in the `KERNEL-GDN-PACKED-DECODE` row of
  [`.agents/kernel-matrix.md`](../kernel-matrix.md): *"excluded at the model
  level by the dense-only `ShouldUsePackedGdnDecode` (`qwen3_5.cpp:49`) and the
  launcher guard rejects its `Hv=32` shape (`cmake` H=32=35B) anyway (clean
  fallback; a 35B cubin would be dead code)"*. Neither half is true at the base:
  the predicate carries no `dense_model` term since `GDN-MOE-BF16-OUT` landed
  (`5ae2c100f`), and `TryTritonPackedDecode` accepts `Hv=32` and dispatches
  `gdn_decode_h32_default` (`src/vt/cuda/cuda_gdn.cu:5207`, `:5239`). Replace it
  with the true statement and the row ID that closed it.
- **E4.** Records: this spec, a `GDN-MOE-PACKED-BA` row in
  `.agents/kernel-matrix.md` beside `GDN-MOE-BF16-OUT`, and one appended
  `.agents/issue-index.md` row for #1793 (owed here). The index's existing
  #1169 row (owner `—`, owed by `gdn-moe-bf16-out.md`) stands as it is: the
  index is keyed on the issue number, `scripts/check-agent-record.py` refuses
  `issue #1169 listed twice`, and the index is append-only, so the row cannot
  be rewritten either. The hand-off of #1169 to this row is recorded by this
  spec's `Issues:` line and by the `Owner` column of the `GDN-MOE-PACKED-BA`
  row in `.agents/kernel-matrix.md`.

Out of scope, and why:

- **The GGUF MoE loader** (`qwen3_5_gguf_weights.cpp:1082-1099`). Three
  residency routes, a V-row reorder and the `gdn_expand_nk` orientation switch
  make its merged owner a second byte-exactness argument. Filed as #1793, owed
  below.
- **`in_proj_qkvz` on the MoE arm.** Same class of gap, different consumer
  (`ProjectGdnQkvz` already carries the separate-fp8 arm the 35B runs, and on
  the bf16 35B the split bf16 arm). It is not a term in
  `ShouldUsePackedGdnDecode`, so it does not gate reachability of the packed
  leg. It stays with #1169's note and is not owed by this row.
- **Any change to `ProjectGdnBA`, the eligibility predicate, the dtype rule, or
  the kernels.** The consumer is already general: a populated `in_proj_ba`
  selects the merged GEMM on a staging platform and row-slices the same owner
  into the two legacy F32 GEMMs everywhere else (`qwen3_5.cpp:3634-3672`).
- **Default flips.** Both toggles are already default-ON. This row adds no
  environment variable.

## Upstream anchors

At `555967922`:

- `vllm/model_executor/models/qwen3_5.py:281-297` — `Qwen3_5ForCausalLMBase`
  carries `packed_modules_mapping["in_proj_ba"] = ["in_proj_b", "in_proj_a"]`.
  `Qwen3_5ForCausalLM` (`:377`) and `Qwen3_5MoeForCausalLM` (`:381`) both
  derive from it, so upstream stacks the two shards into one parameter on the
  MoE arm too. `:405` repeats the mapping on the conditional-generation class.
- `vllm/model_executor/models/qwen3_5.py:210-220` — `Qwen3_5Model` maps the
  checkpoint's `.in_proj_b` → `(.in_proj_ba, 0)` and `.in_proj_a` →
  `(.in_proj_ba, 1)`: shard 0 is `b`, shard 1 is `a`, so the merged row order
  is `[b; a]`.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:414-419` and
  `:516-530` — one `MergedColumnParallelLinear` with `output_sizes =
  [num_v_heads] * 2` for Qwen3.5 (`gqa_interleaved_layout=False`). The comment
  at `:411` records that `ba_proj` does not take block-wise fp8, which is why
  the shards are bf16 on every quantized checkpoint.
- `qwen_gdn_linear_attn.py:802`, `:844`, `:905`, `:945` — every forward arm
  invokes `self.in_proj_ba(hidden_states)` once and splits the result. There
  is no dense-vs-MoE branch on this projection anywhere upstream.

The local consumer that already mirrors this: `ProjectGdnBA`
(`src/vllm/model_executor/models/qwen3_5.cpp:3634-3672`), whose header comment
(`:3552-3563`) says the topology was enabled only for the 27B loader "which is
the only path that populates `in_proj_ba`". That sentence and the one at
`:96-103` become stale on this row and are rewritten in the same change.

## Design

The whole product change is one call:

```cpp
// LoadGdn, qwen3_5_weights.cpp — replaces lines 746-747.
// vLLM owns ONE physical in_proj_ba on every Qwen3.5/3.6 GDN layer, dense and
// MoE alike (packed_modules_mapping on Qwen3_5ForCausalLMBase, qwen3_5.py:297;
// shard order b=0, a=1 at :217-218 @ 555967922). Mirror it here as the dense
// loader does at qwen3_5_dense_weights.cpp:514: one [2*Hv, H] nk owner with
// rows [b; a], and the split fields deliberately empty — ProjectGdnBA slices
// the owner where it needs the legacy pair, and never retains duplicate bytes.
// This is what makes `has_packed_ba` true on the 35B (#1169).
g.in_proj_ba = LoadMergedBf16RawNK(
    get, {la + "in_proj_b.weight", la + "in_proj_a.weight"});
```

What that changes at run time, by platform:

| platform | before (split, Matmul-B `[H,Hv]` x2) | after (one nk owner `[2Hv,H]`) |
|---|---|---|
| CUDA (`needs_weight_staging`) | two `MatmulF32D` GEMMs, F32 out | one `MatmulBTRawD` GEMM, out dtype `MergedGdnBaOutputDType(packed_decode)`: BF16 under packed decode, F32 otherwise; packed decode now eligible |
| CPU | two `MatmulF32D` GEMMs, F32 out | two `MatmulBTRawD` GEMMs over row slices of the owner, F32 out |

The CPU arm therefore changes GEMM entry point (`vt::Matmul` on a `[K,N]`
weight → `vt::MatmulBT` on an `[N,K]` slice) without changing the math or the
output dtype. That is the same transition the dense 27B made under
`KERNEL-GDN-PACKED-DECODE` W1, and the dense CPU forward tests held. Test 2
below measures it on the MoE synthetic model rather than assuming it.

Resident-weight lifetime: `ResidentWeight(d, weights.in_proj_ba)` caches the
device copy keyed on the owner, the split fields stay empty so nothing uploads
twice, and the release walk at `qwen3_5.cpp:8515-8530` already visits
`in_proj_ba`. No new lifetime is introduced; the dense path has run this shape
since W1.

## Risks

1. **A CPU token moves on a MoE checkpoint** because `MatmulBT` over a slice and
   `Matmul` over a transposed owner accumulate in a different order. The
   gated CPU MoE suites (`test_qwen35_paged_forward`, the Qwen3.5-4B plain
   gate if present) and the GPU `test_qwen36_paged_engine` 315/315 decide this.
   If a CPU golden moves, the row stops and records it: the dense arm took this
   transition token-exact, so a moved MoE token is a finding, not a tolerance.
2. **The packed leg is reached for the first time on `Hv=32`, `Hk=16`**: on
   the bf16 35B (`Qwen/Qwen3.6-35B-A3B`) at the default, and on the NVFP4 35B
   only under `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1`. The
   launcher accepts the shape and the `gdn_decode_h32` cubin exists per arch,
   but nothing has executed it end to end on a real checkpoint. The GPU gate
   asserts the counter moved (`packed_launches`, `triton_launches`) and that
   every token still matches the golden. A correctness failure here is a
   kernel-side finding that belongs to `KERNEL-GDN-PACKED-DECODE`, and this row
   then stays `ACTIVE` with the loader change landed behind the same-binary
   `VT_GDN_PACKED_DECODE=0` rollback, which restores the split F32 pair from
   the same owner.
3. **Merged-owner BF16 output under packed decode** (`MergedGdnBaOutputDType`).
   This is the dtype vLLM emits, and the 27B runs it token-exact; on the 35B it
   is new. Covered by the same gate.
4. **`PlanQwen3_5MoeLoad` drift.** If a planner test cross-checks names, E2
   catches it; bytes are identical either way.
5. **Memory.** Zero delta: one owner of `2*Hv*H` bf16 replaces two of `Hv*H`.

## Tests

Red-first through a production entry point. The hermetic test comes first and
is what the fresh implementer proves red, then green.

- **Test 1 (hermetic, the red-first test).** Extend
  `tests/vllm/test_qwen36_weights.cpp` (or a new `tests/vllm/models/
  test_qwen35_moe_gdn_ba_owner.cpp` if the fixture is cleaner there) with a
  synthetic `vllm::TensorResolver` that serves the full GDN layer tensor set of
  one MoE linear-attention layer as in-memory `StTensor`s (bf16 `in_proj_b`
  `[Hv,H]` and `in_proj_a` `[Hv,H]` with distinct known bytes; the rest of the
  layer minimal but present). Call `vllm::LoadQwen3_5MoeLayer(...)` — the
  production loader entry — and assert: `gdn.in_proj_ba` is non-empty, BF16,
  rank 2, shape `{2*Hv, H}`, `nk == true`; its first `Hv` rows are
  byte-identical to the `in_proj_b` source bytes and the next `Hv` rows to
  `in_proj_a`; `gdn.in_proj_b` and `gdn.in_proj_a` are empty. **Red at the
  base:** `in_proj_ba` is empty. Use `Hv=32`, `H=2048`-scaled-down shapes that
  keep the test under a second.
- **Test 2 (CPU forward, byte-exactness of the consumer).** In
  `tests/vllm/models/test_qwen35_paged_forward.cpp`, add a case that builds the
  synthetic MoE model twice: once with the split fields (the existing fixture at
  `:165-166`) and once with a merged `in_proj_ba` owner holding the same bytes
  in `[b; a]` row order (nk). Run the existing paged decode sequence on both and
  compare outputs element-wise. Assert bit-identity and print the max abs
  difference. If the CPU GEMM entry points do not reduce in the same order,
  the assertion fails — that is Risk 1 and the implementer returns
  `NEEDS_DECISION` with the measured difference instead of widening the test.
- **Test 3 (GPU, operator gate, not runnable by the implementer).** Three
  parts, by checkpoint and arm:
  - **(a) NVFP4 35B, default arm.** `test_qwen36_paged_engine` 315/315
    token-exact on `nvidia/Qwen3.6-35B-A3B-NVFP4`, with the packed-decode
    counters read from the run and expected `0/0`: the #365 fp8-tower term
    keeps the default arm off the packed leg. This gates the loader change
    (`ProjectGdnBA` now runs the merged GEMM there), not the packed leg.
  - **(b) NVFP4 35B under `VT_GDN_PACKED_DECODE_FP8_TOWER=1
    VT_GDN_FP8_IN_BF16=1`.** `test_qwen36_paged_engine` 315/315 token-exact
    with `packed_launches > 0` and, on sm_121a, `triton_launches ==
    packed_launches`; `0/0` when `VT_GDN_PACKED_DECODE=0` is added to the same
    two levers.
  - **(c) bf16 35B, `Qwen/Qwen3.6-35B-A3B`.** Default arm `packed_launches >
    0` (and `triton_launches == packed_launches` on sm_121a), and greedy token
    identity between the default arm and `VT_GDN_PACKED_DECODE=0` on the same
    prompts. No oracle golden exists for this checkpoint, so the identity is
    arm-vs-arm on the same binary, not a token-exact gate against vLLM.
- **Existing suites that must stay green:** `test_qwen36_weights`,
  `test_qwen35_plain_weights`, `test_qwen27_paged_forward`, `test_ops_gdn`,
  `test_qwen3_5_gdn_spec_routing`, `test_model_registry`, `test_runner`.

## Gates

| Gate | Command | Owner |
|---|---|---|
| focused red→green | `ctest --test-dir build -R 'qwen36_weights|qwen35_paged_forward' --output-on-failure` | implementer |
| preflight | `scripts/agent-preflight.sh` | implementer, reviewer |
| full CPU suite | `ctest --test-dir build --output-on-failure` | implementer |
| (a) NVFP4 35B default arm: token-exact + counters | `ctest --test-dir build -R test_qwen36_paged_engine --output-on-failure` on `dgx:gpu0` inside an `rc` lease; 315/315, counters expected `0/0` (gates the loader change, not the packed leg) | operator |
| (b) NVFP4 35B, packed leg: token-exact + counters | the same command under `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1`; 315/315 with `packed_launches > 0` and, on sm_121a, `triton_launches == packed_launches`; `0/0` with `VT_GDN_PACKED_DECODE=0` added | operator |
| (b) NVFP4 35B same-binary speed A/B | `VT_GDN_PACKED_DECODE=0` vs default, both under the same two levers, c1 decode TPOT, `nvidia/Qwen3.6-35B-A3B-NVFP4`, graphed, identity asserted, idle box, 3 reps, `dgx:gpu0` inside an `rc` lease | operator |
| (c) bf16 35B default arm: counters + identity | `Qwen/Qwen3.6-35B-A3B` on `dgx:gpu0` inside an `rc` lease; default arm `packed_launches > 0`, greedy token identity vs `VT_GDN_PACKED_DECODE=0` on the same prompts (no oracle golden exists for this checkpoint) | operator |
| (c) bf16 35B same-binary speed A/B | `VT_GDN_PACKED_DECODE=0` vs default, c1 decode TPOT, `Qwen/Qwen3.6-35B-A3B`, graphed, identity asserted, idle box, 3 reps | operator |
| reviewer mutation | delete the `LoadMergedBf16RawNK` call (restore the split pair) in a scratch copy; Test 1 must go red. Delete the Test 2 merged-owner arm's `nk` flag; the consumer `VT_CHECK` must fire | reviewer |

The operator reruns the GPU gate itself. An implementer or reviewer report is
an input.

## Evidence

CPU half, measured by the fresh implementer on the CPU-only host (20 cores, no
GPU; `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF`). Each item names the tree it ran on.

### Test 1 (hermetic, red then green)

`tests/vllm/models/test_qwen35_moe_gdn_ba_owner.cpp`, registered as
`test_qwen35_moe_gdn_ba_owner`. Enters through `vllm::LoadQwen3_5MoeLayer`
(the production per-layer entry; `LoadQwen3_5Moe` calls the same
`LoadLayerImpl`) over an in-memory resolver serving the 18-tensor request set of
one MoE `linear_attention` layer on the all-bf16 tower.

Red, at the spec commit `6ae3028d7` (product tree = base `5d638b67e`), with the
new test compiled against the unchanged loader:

```text
tests/vllm/models/test_qwen35_moe_gdn_ba_owner.cpp:171: FATAL ERROR:
REQUIRE_FALSE( gdn.in_proj_ba.Empty() ) is NOT correct!
  values: REQUIRE_FALSE( true )
[doctest] test cases: 1 | 0 passed | 1 failed | 0 skipped
[doctest] assertions: 3 | 2 passed | 1 failed |
[doctest] Status: FAILURE!
```

Green, at the implementation head (this commit):

```text
[doctest] test cases: 1 | 1 passed | 0 failed | 0 skipped
[doctest] assertions: 23 | 23 passed | 0 failed |
[doctest] Status: SUCCESS!
```

The green run asserts: owner non-empty, `kBF16`, rank 2, shape `{2*Hv, H}` =
`{8, 64}`, `nk == true`; rows `[0, Hv)` byte-identical to the `in_proj_b`
source pattern and rows `[Hv, 2Hv)` to `in_proj_a` (0 mismatching elements in
each half, the two patterns provably distinct); `in_proj_b` and `in_proj_a`
empty; every other GDN field loaded as before; `in_proj_qkvz` still empty.

### Test 2 (CPU forward bit-identity)

`tests/vllm/models/test_qwen35_paged_forward.cpp`, case "qwen35 paged MoE: a
merged in_proj_ba owner is bit-identical to the split pair on CPU". The
synthetic MoE model (3 GDN layers, `Hv=4`, `H=32`) is built twice from the same
seeds, once with the split `[H,Hv]` pair and once with the merged `[2*Hv, H]`
nk owner (rows `[b; a]`, the transpose of the stacked pair), and the existing
prefill-then-decode sequence runs on both through `ModelRegistry::Forward`.
Measured at the implementation head:

```text
merged-vs-split in_proj_ba, prefill: max|diff| = 0, elements with differing bits = 0 / 200
merged-vs-split in_proj_ba, decode:  max|diff| = 0, elements with differing bits = 0 / 40
```

Verdict: bit-identical. The `vt::MatmulBT`-over-slice arm and the `vt::Matmul`
arm produce the same float bits on CPU, so Risk 1 did not fire and no tolerance
was widened. Both registrations of the binary (`test_qwen35_paged_forward` and
`test_qwen35_paged_forward_gdn_out_f32`) pass.

### Mutations (IMP-MUTATE), at the implementation head

1. Restore the split pair in `LoadGdn` (replace the `LoadMergedBf16RawNK` call
   with the two `LoadBf16Transposed` calls), rebuild `libvllm.a` and the test:
   Test 1 fails at the same assertion as the red run (`REQUIRE_FALSE(
   gdn.in_proj_ba.Empty() )`, 1 failed / 3). Restored byte-for-byte: the file's
   sha256 (`55510766e6c5…`) and the sha256 of `git diff -- <file>` are equal
   before and after; rebuilt; Test 1 green again (23/23).
2. Flip the merged owner's `nk` to `false` in Test 2's fixture, rebuild: the
   consumer `VT_CHECK` fires as the case exception `vt: qwen3_5 merged GDN BA:
   invalid packed owner at src/vllm/model_executor/models/qwen3_5.cpp:3643`
   (plus the fixture's own `nk` checks, 3 failed / 39). That proves
   `ProjectGdnBA` reads the owner on the merged arm. Restored byte-for-byte
   (diff sha256 equal before and after), rebuilt, case green (48/48).

### Existing suites

Two existing assertions pinned the old split contract on the MoE loader and
were moved to the new one-owner invariant in the same change (the intent of
each case, that the b/a tail is independent of the tower arm, is unchanged):

- `tests/vllm/models/test_qwen3_8_text_only.cpp` "the bf16 GDN tower loads
  through the MoE arm": now asserts `in_proj_ba` non-empty, `nk`, shape
  `{2*Hv, H}`, split fields empty. Suite 19/19, 67855 assertions.
- `tests/vllm/test_qwen36_weights.cpp` "full-layer load" and the
  `VT_DENSE_NATIVE=0` subcase: now assert the owner `{64, 2048}` on the real
  35B NVFP4 shard. These skip on this host (shard absent) and are owed to the
  operator's GPU run.

`PlanQwen3_5MoeLoad` (E2) is unchanged: it plans the loader's checkpoint
request set by name, and the merged load still requests `in_proj_b.weight` and
`in_proj_a.weight`. Case 5a of `test_qwen3_8_text_only` (the plan is exactly
the loader's request set, both directions, through `LoadQwen3_5Moe` over a
real file) stays green, which is the executable form of that statement.

Full CPU suite at the implementation head, `ctest --test-dir build -j 6
--output-on-failure`:

```text
99% tests passed, 1 tests failed out of 578
Total Test time (real) = 306.11 sec
The following tests FAILED:
         69 - test_minimax_music3_e2e_real (Not Run)
```

The one failure is pre-existing and unrelated: `test_minimax_music3_e2e_real`
does not link under `-DVLLM_CPP_SERVER=OFF` (undefined
`vllm::entrypoints::openai::ApiServer::*`; `api_server.cpp` is compiled only
when `VLLM_CPP_SERVER` is ON, and `tests/CMakeLists.txt:265` registers the suite
unconditionally). Neither file is touched by this branch
(`git diff --stat origin/main..HEAD -- tests/CMakeLists.txt CMakeLists.txt
tests/parity/` shows only the new registration line).

Preflight: `scripts/agent-preflight.sh` exit 0;
`scripts/agent-preflight.sh --staged` exit 0.

### GPU half: what ran on 2026-08-24 (operator, `dgx:gpu0` inside an `rc` lease)

Tree `f1cc4fb61`, built in the lease for sm_121a with the lease-staged CUDA
13.3.73 toolkit (cuBLASLt 13.6.0.2), CUTLASS sm120a NVFP4, FA2, vendored
Triton AOT sm_121a; `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `491c2f1e` staged
NAS-to-local through `scripts/rc-stage-checkpoint.sh` (#1807; 35 files
copied+verified once, `ALREADY STAGED ... nothing read` on every rerun). Job
script and logs: `/mnt/nas_share/rc/gdn-moe-packed-ba/{gate-ab.sh,logs/}`.

| arm | env | result |
|---|---|---|
| `test_qwen36_weights`, real shard | default | 10/10 cases, 174/174 assertions: the merged `in_proj_ba` owner asserted on the real 35B NVFP4 shard, split fields empty |
| (a) default, EAGER | `VLLM_CPP_CUDAGRAPH=0` | **token-exact**, 3/3 cases, 320/320 assertions; `packed_launches=0 triton_launches=0 packed_expected=0` |
| (b) levers on, EAGER | `VLLM_CPP_CUDAGRAPH=0 VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1` | **token-exact**, 320/320; **`packed_launches=450 triton_launches=450 packed_expected=1`**, which is 15 pure-decode steps x 30 GDN layers: every decode step of every GDN layer ran the vendored Triton FLA `gdn_decode_h32` cubin, and the greedy continuation still matches the vLLM golden |
| (b) rollback, EAGER | the two levers `+ VT_GDN_PACKED_DECODE=0` | token-exact, 320/320; `packed_launches=0` |
| (a), (b), (b)-rollback, GRAPHED (production default) | as above without `VLLM_CPP_CUDAGRAPH=0` | **throw at the first decode-graph capture** on every arm: `vt cuda: matmul_fp8_cutlass: cudaMallocAsync alpha: operation failed due to a previous error during capture` (144-148 assertions pass before the throw; the next case then trips the latched error in `marlin_repack`) |

The graphed throw is the lease environment, not this row, and three
measurements say so: it fires on the DEFAULT arm, where the packed leg is not
selected; **`main` without the row (`1791ec90e`) built and run in the same
lease throws identically graphed and passes eagerly 315/315**
(`logs/base-graphed.log`, `logs/base-eager.log`); and
`CUBLASLT_LOG_LEVEL=4` on that base run names the first failing call
(`logs/cublaslt-base-graphed.log:965`): after 482 successful eager
`cublasLtMatmulAlgoGetHeuristic` queries, the first query inside capture fails
with `Could not obtain green context information` on a bf16 M=1 GEMM
`[1x2048] x [2048x256]`, the MoE router gate. That is
[#1732](https://github.com/mudler/vllm.cpp/issues/1732) (cuBLASLt 13.3
heuristic inside capture) verbatim, on GB10. A second build of this head with
PR [#1741](https://github.com/mudler/vllm.cpp/pull/1741) cherry-picked
(`src-f1cc4fb61+1741.tar.gz`, merge tree `afd1dd2c`) fails at the same point,
so that fix does not cover this call site; the data point is recorded on both
threads.

What this settles and what it does not. The loader change is correct on the
real 35B (weights gate plus token-exact default arm), the packed leg is reached
and token-exact under the two #365 levers with the exact host-dispatch count
the geometry predicts, and the rollback restores the split F32 pair. The
production-configuration (graphed) run of the same gate is `PENDING` on #1732
in every lease, for this row and for every other 35B gate, until a fix lands
or the lease toolkit changes; the same-binary TPOT A/B and gate (c) need the
graphed arm and wait with it. Two lease-recipe defects cost two runs and are
fixed in the job script for the next reader: the staged toolkit's `lib/`
carries no SONAME links (`libcudart.so.13`), and the loader path must be
exported.

### GPU half as specified (for the record)

- (a) `test_qwen36_paged_engine` 315/315 on the NVFP4 35B default arm, with
  the counters (expected `0/0`).
- (b) `test_qwen36_paged_engine` 315/315 on the NVFP4 35B under
  `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1`, with the counters
  (`packed_launches > 0`), and the A/B table for that arm vs
  `VT_GDN_PACKED_DECODE=0` under the same two levers: TPOT c1 per arm, 3 reps,
  the ratio, and the box's idle state.
- (c) bf16 35B: default-arm counters, the arm-vs-`VT_GDN_PACKED_DECODE=0`
  greedy identity, and the TPOT A/B table.
- `test_qwen36_weights` on the 35B NVFP4 shard (the two updated subcases).

Staging precondition, found by the operator: the NVFP4 35B snapshot
(`491c2f1ea524c639598bf8fa787a93fed5a6fbce`) exists only in the dgx host's
`~/.cache/huggingface/hub`, which a leased worker cannot see (`/workspace` is
the NAS `rc/` subfolder), and no copy exists on the NAS, so gates (a) and (b)
are `PENDING` on authority to stage it (about 22 GB). The bf16 35B is on the
NAS at `checkpoints/qwen3.6-35b-a3b-bf16`, outside `rc/`, so it needs a
NAS-internal copy into `rc/` before gate (c) can run.

## Owed

- [#1793](https://github.com/mudler/vllm.cpp/issues/1793) — the GGUF MoE loader
  keeps `in_proj_b`/`in_proj_a` split across three residency routes, so packed
  GDN decode stays unreached on GGUF MoE. Owed here until a row claims it.
- The `in_proj_qkvz` merged owner on the MoE arm (noted in #1169, not a
  reachability term for the packed leg). Not owed by this row; recorded so the
  next reader does not expect it here.

## Stop conditions

- Test 2 is not bit-identical on CPU → `NEEDS_DECISION` with the measured
  difference. Do not widen the tolerance and do not land a shape term.
- `test_qwen36_paged_engine` moves a token with the packed leg selected →
  land the loader change only behind `VT_GDN_PACKED_DECODE=0` evidence, file
  the kernel finding against `KERNEL-GDN-PACKED-DECODE`, row stays `ACTIVE`.
- `dgx:gpu0` unavailable → CPU gates land, GPU gates `PENDING` with the lease
  named; the row does not reach `DONE`.
