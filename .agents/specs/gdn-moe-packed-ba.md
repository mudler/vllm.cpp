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

`SPIKE` → `READY` on this spec's commit. The gap is verified against the base
SHA: `in_proj_ba` is written at one site in the tree
(`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:514`), the MoE
safetensors loader loads the two shards split
(`src/vllm/model_executor/models/qwen3_5_weights.cpp:746-747`), and the
eligibility predicate still reads `!w.in_proj_ba.Empty()`
(`src/vllm/model_executor/models/qwen3_5.cpp:4779`). No open pull request
touches either site. The packed leg is now default-ON
(`PackedGdnDecodeRuntimeEnabled`, `MergedGdnBaEnabled`, both `qwen3_5.cpp`),
so on the 35B the loader is the only term keeping the vendored FLA cubin
unreached.

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
  `.agents/kernel-matrix.md` beside `GDN-MOE-BF16-OUT`, and two appended
  `.agents/issue-index.md` rows (#1169 now owned by this row; #1793 owed here).

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
2. **The packed leg is reached for the first time on `Hv=32`, `Hk=16`.** The
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
- **Test 3 (GPU, operator gate, not runnable by the implementer).**
  `test_qwen36_paged_engine` 315/315 token-exact on
  `nvidia/Qwen3.6-35B-A3B-NVFP4`, with the packed-decode counters read from the
  run: `packed_launches > 0` and, on sm_121a, `triton_launches == packed_launches`
  on the default arm; `0/0` with `VT_GDN_PACKED_DECODE=0`.
- **Existing suites that must stay green:** `test_qwen36_weights`,
  `test_qwen35_plain_weights`, `test_qwen27_paged_forward`, `test_ops_gdn`,
  `test_qwen3_5_gdn_spec_routing`, `test_model_registry`, `test_runner`.

## Gates

| Gate | Command | Owner |
|---|---|---|
| focused red→green | `ctest --test-dir build -R 'qwen36_weights|qwen35_paged_forward' --output-on-failure` | implementer |
| preflight | `scripts/agent-preflight.sh` | implementer, reviewer |
| full CPU suite | `ctest --test-dir build --output-on-failure` | implementer |
| 35B token-exact + counters | `ctest --test-dir build -R test_qwen36_paged_engine --output-on-failure` on `dgx:gpu0` inside an `rc` lease | operator |
| same-binary speed A/B | `VT_GDN_PACKED_DECODE=0` vs default, c1 decode TPOT, `nvidia/Qwen3.6-35B-A3B-NVFP4`, graphed, identity asserted, idle box, 3 reps | operator |
| 35B bf16 arm | `Qwen/Qwen3.6-35B-A3B` token-exact if staged on the host | operator |
| reviewer mutation | delete the `LoadMergedBf16RawNK` call (restore the split pair) in a scratch copy; Test 1 must go red. Delete the Test 2 merged-owner arm's `nk` flag; the consumer `VT_CHECK` must fire | reviewer |

The operator reruns the GPU gate itself. An implementer or reviewer report is
an input.

## Evidence

To be filled by the implementer (CPU) and the operator (GPU), each naming the
tree SHA the measurement ran on.

- Test 1 red output at the base, green output at the head.
- Test 2 max abs difference and the bit-identity verdict.
- `test_qwen36_paged_engine` 315/315 and the counter values, both arms.
- A/B table: TPOT c1 per arm, 3 reps, the ratio, and the box's idle state.

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
