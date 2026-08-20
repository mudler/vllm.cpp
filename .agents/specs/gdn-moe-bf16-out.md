# GDN-MOE-BF16-OUT — narrow the GDN recurrence output and the z gate to bf16 on the MoE arms

Row: `GDN-MOE-BF16-OUT`
Issues: [#1168](https://github.com/mudler/vllm.cpp/issues/1168) (primary — f32
core/z on every MoE checkpoint),
[#1169](https://github.com/mudler/vllm.cpp/issues/1169) (owed — the merged
`in_proj_ba` owner is dense-only),
[#1170](https://github.com/mudler/vllm.cpp/issues/1170) (owed — the GDN Triton AOT
arms are pinned to 48 or 32 linear V-heads)
Base SHA: `dd8a3b0e184c14312c9be7017d9f2f5a4ec85071`
Upstream pin: see [`.agents/upstream-sync.md`](../upstream-sync.md)
(`5559679229bc961848b121ccdeaa8fa5d79bec98`).
Secondary oracle: `sglang` @ `f63458b5be`, see [`.agents/oracles/sglang.md`](../oracles/sglang.md).

## Now

Both edits in `## Design` are implemented, on `row/GDN-MOE-BF16-OUT-IMPL` off
`fd64c76ee`. `GdnOutDType()` takes no model-shape argument and resolves bf16 from
`VT_GDN_OUT_BF16` alone, the three call sites drop `cfg.num_experts == 0`, and
`detail::ShouldUsePackedGdnDecode` no longer carries `e.dense_model`. The env
decision is extracted into the pure `detail::GdnOutBf16FlagIsOn(const char*)`,
the second shape `## Tests` offers, because the resolver caches its `getenv` and
a process can only observe one value of it.

**Nothing here is measured, and no GPU gate has run.** The row cannot reach
`DONE` on this evidence. What ran and what is owed:

| Gate | Result |
|---|---|
| `test_qwen35_paged_forward` (CPU, the MoE reachability vehicle) | 5/5 cases, 13/13 assertions |
| `test_qwen27_paged_forward` (CPU, the two predicates) | 31/31 cases, 770/770 assertions |
| `test_ops_gdn`, `test_qwen27_dense_forward`, `test_qwen3_5_gdn_spec_routing`, `test_model_registry`, `test_runner`, the three expert-stream suites | all green on the CPU tier |
| `scripts/agent-preflight.sh` | `PASS`, exit 0 |
| full `ctest` | **not run.** The implementer's host had 6 GB free, and a full test build does not fit. `PENDING`, named resource |
| `test_qwen27_paged_engine` 235/235 | **not run.** Skipped with `assertions: 0` — no 27B checkpoint and no GPU on the implementer's host. `PENDING` |
| `test_qwen36_paged_engine` 315/315 | **not run.** `PENDING`, same reason. This is the row's ONE measurable checkpoint |
| correctness on `nvidia/Qwen3.6-35B-A3B-NVFP4` | `PENDING`, GPU host |
| `VT_GDN_OUT_BF16=0|1` same-binary speed A/B, per leaf | `PENDING`, GPU host |
| `nsys --cuda-graph-trace=node` memory-format confirmation | `PENDING`, GPU host |
| bf16 into `vt::RmsNormGatedQuantFp8` (the 35B's fp8 `out_proj` gated norm) | **not run.** Unreachable on the CPU tier: no fp8 `out_proj` on the synthetic model and no fp8 platform. `PENDING`, GPU host |
| GGUF MoE | `PENDING`. See `## Stop conditions`: if it cannot be gated, do NOT reintroduce a shape term |
| 2.4T | `PENDING`, named resource, by construction |

The GPU gates belong to the operator, and the correctness gate comes first. A
fresh reviewer sees the immutable head next; the mutation `## Design` names is to
restore the `dense_model` form of `GdnOutDType` in a scratch copy, which must
turn `test_qwen35_paged_forward`'s bf16 case red.

**Fresh review 1 -> repair.** The reviewer confirmed both edits correct and bound
by tests, and found the LEVER unpinned in one direction and a false-red generator
in the other. `GdnOutDType()` mutated to `return DType::kBF16;` — severed from its
parser and from `VT_GDN_OUT_BF16` entirely — left `test_qwen35_paged_forward` 5/5
13/13 and `test_qwen27_paged_forward` 31/31 770/770 GREEN, while
`VT_GDN_OUT_BF16=0` on the unmutated binary FAILED 4/5 cases 11/13 assertions
because the new case asserted bf16 unconditionally and the resolver's cached
`getenv` cannot be neutralised in-process. The rollback that the A/B above is
built on was therefore both ungated and a red an operator would read as a
regression. Repaired by giving the resolver a `detail::` declaration, asserting
against the environment as the test file reads it directly, and registering the
same binary a second time under `VT_GDN_OUT_BF16=0` — the shape
`tests/CMakeLists.txt` already uses for a read-once lever. Both arms are now
6/6 cases 15/15 assertions, and the severing mutation is RED in the `=0` arm
(3 assertions across 2 cases).

**The row now has a lifecycle surface.** `GDN-MOE-BF16-OUT` existed only in
`.agents/issue-index.md` and in this file, so there was nothing to move when its
state changed. It is recorded in [`.agents/kernel-matrix.md`](../kernel-matrix.md)
beside `KERNEL-GDN-AOT-BF16`, at `GATING`: implemented, CPU tier green, every GPU
gate `PENDING`. That gap came from the spec pull request rather than from the
implementation. `KERNEL-GDN-AOT-BF16`'s own "Every 35B path stays f32" is
corrected in the same edit, because Edit 1 is what makes it false — the same
obligation as `## A record consequence outside this row`. `docs/STATUS.md` and
`docs/BENCHMARKS.md` are written with it, because a new lifecycle state is a
claim about the project and `scripts/check-doc-checkpoint.py` says so at the
gate. Both entries say the same thing this section does: **no number, none
claimed, every measurable axis owed to a GPU host.** An entry that recorded a
capability without recording that nothing is measured would be the more
misleading half.

**The fp8-fused gated norm is OWED, not tested.** See `## Risks`: Test 3 cannot
reach `vt::RmsNormGatedQuantFp8`, so bf16 has never entered that op on a real
checkpoint. It belongs to the 35B GPU gate together with the rows above.

**Collision check, run against `origin/main` at `fd64c76ee` before the edits:**
neither collides yet. `row/REFACTOR-DTYPE-CONSISTENCY` has not landed —
`include/vllm/model_executor/models/activation_dtype.h` does not exist on `main`,
and the branch is still local. `row/PERF-GDN-BF16-CHAIN` has not landed either;
the only `main` commit whose message names it is `918d4546e`
(`PERF-FP8-ALPHA-FOLD`), which merely says it composes with that branch. Both
still have to be reconciled onto this row's result when they land, which is what
`## Owed` records.

## Scope

IN: the resolved dtype of the two tensors `GdnOutDType` controls — `dcore`, the
GDN recurrence output `[T, Hv, Dv]`, and `z`, the output gate — on **MoE**
Qwen3.5/3.6/3.8 checkpoints, plus the now-redundant `dense_model` term in
`detail::ShouldUsePackedGdnDecode`.

OUT, deliberately:

| Excluded | Why |
|---|---|
| the dense 27B arm | already bf16 by default at `dd8a3b0e1`. This row must leave it byte-identical, and `test_qwen27_paged_engine` 235/235 is the control that proves it. |
| the GDN `in_proj` chain dtype (`mixed_qkv`, conv, post-conv) | that is `GdnInDType` and `VT_GDN_FP8_IN_BF16`, owned by `row/PERF-GDN-BF16-CHAIN` (#417). Different tensors, different lever. |
| the attention gate buffer `gatef` | `row/PERF-27B-BF16-FP8-OUT` Phase C (#417). |
| the per-column fp8 alpha pass | `row/PERF-FP8-ALPHA-FOLD` (#402 Lever B). |
| building the merged `in_proj_ba` owner in the MoE loader | #1169. It is a loader change with its own resident-weight lifetime and byte-exactness argument. Listed under `## Owed`. |
| generating `h128` GDN AOT specializations | #1170. Listed under `## Owed`. |

### Which checkpoints this actually moves

| checkpoint | arch | linear V-heads | gateable here |
|---|---|---:|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | MoE | 32 | **yes** — `test_qwen36_paged_engine`, 315/315 |
| `Qwen/Qwen3.8-2.4T-A95B` | MoE | 128 | **no** — ~4.8 TB bf16 against 128 GB unified memory |
| `Qwen3.8-27B` / `Qwen3.6-27B` | dense | 48 | not affected; already bf16 |

State this plainly in the pull request body as well. The row's measurable surface
is **one checkpoint**. Any claim that it moves the 27B is wrong, and any claim
about the 2.4T is unmeasured by construction.

## Upstream anchors

vLLM is the primary reference and it implements this path, so it is the only
reference for the behavior.

| Ask | Upstream answer @ `5559679` |
|---|---|
| What dtype is the GDN recurrence output? | **bf16.** `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:870-873` — `core_attn_out = torch.zeros((num_tokens, num_v_heads // tp_size, head_v_dim), dtype=hidden_states.dtype, ...)`. The ROCm arm at `:805-808` does the same with `torch.empty`. |
| What dtype is the `z` gate? | **bf16.** `:843` projects `mixed_qkvz` through `self.in_proj_qkvz`, and `:859-860` splits `z` out of it as a view. |
| What dtype is the gated-RMSNorm weight? | **bf16.** `:459-465` constructs `RMSNormGated(head_v_dim, ...)` with no dtype override, so the parameter takes the default model dtype. |
| Does any of it branch on dense vs MoE? | **No.** `Qwen3_5ForCausalLMBase` (`vllm/model_executor/models/qwen3_5.py:280-297`) is the shared base of the dense and MoE causal-LM arms and carries one `packed_modules_mapping`. vLLM resolves one model dtype and every layer inherits it. |
| Does upstream gate packed recurrent decode on model shape? | **No.** `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` defaults `True` with no shape term (`vllm/envs.py:124`). |

Corroborated by the secondary oracle, which matters only as agreement and never
outranks the above: SGLang builds `RMSNormGated(..., dtype=config.torch_dtype)`
(`python/sglang/srt/models/qwen3_5.py:522-536`), slices `z` from the bf16
`projected_states_qkvz` (`:515-527`), allocates the packed decode output as
`mixed_qkv.new_empty(...)` (`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:79-83`),
and keys `supports_packed_decode` on the platform alone (`:43`).

So our f32 is a deviation from the **primary** oracle. This is not a case where
the two references disagree.

## The defect

`src/vllm/model_executor/models/qwen3_5.cpp:3190` @ `dd8a3b0e1`:

```cpp
DType GdnOutDType(bool dense_model) {
  static const int override = [] {
    const char* e = std::getenv("VT_GDN_OUT_BF16");
    if (e == nullptr) return -1;
    return e[0] == '0' ? 0 : 1;
  }();
  const bool bf16 = override >= 0 ? override != 0 : dense_model;
  return bf16 ? DType::kBF16 : DType::kF32;
}
```

Three call sites pass `cfg.num_experts == 0` — `:3749`, `:4155`, `:4386`. The
default is bf16 on a dense checkpoint and f32 on every MoE checkpoint.

`outdt` reaches:

- `dcore` `[T, Hv, Dv]` at `:3807`, `:4253`, `:4265`, `:4285`, `:4587`;
- `z` at `:3584` and `:3688-3690`;
- the gated-norm weight `dnw`, which must match (`ResidentWeightF32` at f32,
  native `ResidentWeight` at bf16) — `:3817-3818`, `:4301-4302`, `:4733-4734`.

Each of `dcore` and `z` is stored once and loaded once. f32 doubles both, so the
cost is four halved passes over the largest per-layer GDN activation.

### The tree already documented this and deferred it

`qwen3_5.cpp:3172-3189` is a 18-line comment that derives the whole finding and
then declines to act on it:

> vLLM keeps `core_attn_out` and the z gate bf16 (the gated-RMSNorm consumes
> them) ... Our `dcore` (recurrence out) + `z` were f32 (a more-precise deviation
> that doubled the `[T,Hv,Dv]` core traffic in/out of `GdnDecode`/`GdnPrefill`
> AND the gated-norm read) ... **Keep every unmeasured 35B arm, including GGUF,
> on its prior f32 default; the explicit env override remains available for its
> later independently gated campaign.**

**This row is that campaign.** The derivation is not repeated here; the comment
is the record and `.agents/specs/nvfp4-small-m-dispatch.md:706-711` records the
27B flip that produced it (`KERNEL-GDN-AOT-BF16`).

### Why no gate caught it

CLAUDE.md, "Inherit vLLM defaults": *"A token gate cannot detect a dtype that is
too wide. The tokens still match, and the goldens still pass, although the path
moves twice the bytes."* `test_qwen36_paged_engine` is 315/315 today **because**
f32 is the more precise deviation. The gate is doing its job and is structurally
blind to this axis.

## Design

Two edits, in this order. They must land together.

### Edit 1 — the dtype, which is the only lever

Make `GdnOutDType` take no model-shape argument, mirroring `GdnInDType()`
(`:3117-3123`) exactly:

```cpp
DType GdnOutDType() {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_GDN_OUT_BF16");
    return e == nullptr || e[0] != '0';
  }();
  return bf16 ? DType::kBF16 : DType::kF32;
}
```

The three call sites drop `cfg.num_experts == 0`. `VT_GDN_OUT_BF16=0` stays the
same-binary f32 rollback, and it is now the rollback for both arms rather than
for the dense one alone.

Removing the parameter is deliberate and is not cosmetic. While the signature
accepts a model shape, a reader has to check every call site to learn what the
default is, and a future call site can reintroduce the split silently. vLLM has
no such parameter because it has no such decision.

Nothing downstream needs new code. Verified at `dd8a3b0e1`:

- `vt::RmsNormGated` and `vt::RmsNormGatedQuantFp8` already require only
  `gate.dtype == x.dtype && weight.dtype == x.dtype` (`src/vt/ops.cpp:685-688`),
  and the CUDA implementation dispatches both `float` and `__nv_bfloat16`
  (`src/vt/cuda/cuda_gdn.cu:2164-2181`).
- `dnw` already follows `outdt` at all three sites.
- `GdnDecode`/`GdnPrefill` already dispatch a `Tout=bf16` path, per the comment
  at `:3178-3181`.
- The bf16 chunk_o AOT specialization for the 35B geometry is vendored:
  `src/vt/cuda/triton_aot_vendored/*/gdn_chunko_bf16_h32.*`.

### Edit 2 — the term that is redundant once Edit 1 lands

Remove `e.dense_model` from `detail::ShouldUsePackedGdnDecode` (`:76-84`), the
field from `detail::GdnPackedDecodeEligibility`
(`src/vllm/model_executor/models/qwen3_5_internal.h:62-76`), and the
`cfg.num_experts == 0` argument at `:4406`.

**Order is the design point, and it is why the two edits are one change.**

- The term entered at `f344decf4`, "feat(gdn): dispatch exact packed decode",
  whose body calls it one of the "real-model safety gates". It is a day-one
  staging gate, never revisited, and no measurement supports it. Neither
  reference has an equivalent.
- Removing it **before** Edit 1 is inert and therefore untestable through any
  production entry point: `GdnPackedDecodeDTypesCompatible` (`:115-122`) pins
  `core_out` to bf16, `core_out` is `outdt`, and `outdt` is f32 on a MoE
  checkpoint. The removal would change no observable behavior, which is exactly
  the shape `.agents/reachability.md` warns about.
- Removing it **after** Edit 1 removes a term that has become redundant behind
  the dtype rule. That is the correct sequence and the only one in which the
  removal is reachable.
- Leaving it in place after Edit 1 leaves a shape term that contradicts both
  references and that no measurement supports.

### What Edit 2 does NOT achieve, corrected here

The premise this row started from was that Edit 2 unlocks packed GDN decode on
the MoE arms. **It does not**, and the correction is #1169.
`ShouldUsePackedGdnDecode` also requires `e.has_packed_ba`, populated at `:4410`
as `!w.in_proj_ba.Empty()`, and `in_proj_ba` is written at exactly one site in the
tree — `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:437`, the dense
loader. The MoE loader loads the shards split (`qwen3_5_weights.cpp:563-564`) and
so does the GGUF loader (`qwen3_5_gguf_weights.cpp:1067-1070`). `qwen3_5.cpp:3220`
states it in the tree: *"the only path that populates `in_proj_ba`"*.

So packed decode on a MoE checkpoint needs Edit 1, Edit 2, **and** the MoE merged
`in_proj_ba` owner. This row ships the first two and owes the third. Say this in
the pull request body; do not let "unlocks packed decode" stand unqualified.

### Blast radius, per arm

| arm | in_proj owner | what Edit 1 changes |
|---|---|---|
| 35B-A3B NVFP4, merged fp8 leaf | `in_proj_qkv_fp8` + `in_proj_z_fp8`, `:3619-3657` | `VT_GDN_FP8_IN_BF16` is default OFF (`:3163-3169`, on only for a leading `'1'`), so the merged GEMM still emits f32. `z_raw.dtype != outdt` then takes the `CastBf16` branch at `:3651-3655`, which **adds a pass that does not exist today**. Measure it; it may make the merged arm a net loss until #417's toggle also flips. |
| 35B-A3B NVFP4, split fp8 leaf | `in_proj_qkv_fp8` + `in_proj_z_fp8`, `:3688-3691` | the `z` GEMM emits `outdt` directly through `MatmulFp8CutlassD(..., outdt)`. Clean narrowing, no new pass. |
| bf16 / GGUF MoE | split `in_proj_qkv` + `in_proj_z` | `z` moves from `MatmulF32D` to `MatmulBf16D` (`:3690`). `mixed_qkv` is unchanged at `indt`. |
| dense 27B | unchanged | already bf16; must stay byte-identical. |

One arm that could have moved and does not: `ShouldUseMergedGdnQkvz` (`:3572-3574`)
takes `indt == outdt` as its `uniform_dtype` term, which Edit 1 flips from false to
true on a MoE model. It cannot select anything, because the merged bf16 `in_proj_qkvz`
owner is also built only by the dense loader — no MoE or GGUF loader writes that
field either. Verified by grep at `dd8a3b0e1`. Record it so a reviewer does not have
to re-derive it.

### A record consequence outside this row

`GdnFp8MergedInProjDType` (`:3280-3282`) requires `outdt == BF16` as its third
term, and `qwen3_5.cpp:3598-3600` says that term *"is what confines the change to
the 27B: the 35B is MoE, so `GdnOutDType(dense_model=false)` is f32 by default and
this stays inert there."* After Edit 1 that sentence is false as written. The arm
stays inert on the 35B only because `VT_GDN_FP8_IN_BF16` is default OFF, which is a
**toggle** term rather than a **model-shape** term.

[#521](https://github.com/mudler/vllm.cpp/issues/521) asks for exactly this
correction in `.agents/specs/perf-fp8-alpha-fold.md`, on the pre-Edit-1 reasoning.
The implementer of this row must repair the three in-tree comments that state the
model-shape reason — `qwen3_5.cpp:3598-3600`, `src/vt/cuda/cuda_gdn.cu:1879` and `:1920`,
`src/vllm/model_executor/models/qwen3_5_internal.h:142-144` — in the same change,
because Edit 1 is what makes them wrong. The spec correction #521 owns stays with
#521's row.

### Reachability

Both edits are reached from a production entry point at their own merge commit:
`ModelRegistry::Forward` -> the Qwen3.5/3.6 MoE decode path -> `GdnBlockPaged`
(`:4386`) on the default configuration, with no toggle set. Nothing here is
staged or unreached, so no `## Owed` entry is needed for the edits themselves.

The fresh reviewer mutates for it: revert `GdnOutDType` to the `dense_model` form
in a scratch copy and rerun the focused gate. A gate that stays green measured the
predicate, not the capability.

## Tests

RED first, in this order. Capture each red result before the change.

1. **CPU tier, the dtype predicate.** `GdnOutDType` is file-local today, so the
   implementer either declares it in `qwen3_5_internal.h` beside the existing
   `detail::` predicates or extracts a pure `detail::` helper for the env
   decision, matching the shape `PackedGdnDecodeFp8TowerFlagIsOn` already uses.
   If `row/REFACTOR-DTYPE-CONSISTENCY` has landed first, the function is already
   header-visible at `include/vllm/model_executor/models/activation_dtype.h:187`
   and no extraction is needed.
   Assert: the default resolves bf16 with no model-shape input; `VT_GDN_OUT_BF16=0`
   resolves f32; `=1` resolves bf16. RED today for the MoE case.
   Mutation proof: restoring the `dense_model` branch must turn it red.
   **Fresh review 1 added the missing half.** Pinning the extracted parser does
   not pin that the RESOLVER reads it, and the two are separable in silence: see
   `## Now`. The resolver therefore gets a `detail::` declaration and its own
   case, and because it caches its `getenv` the two env values are two ctest
   registrations of the same binary rather than two assertions in one process —
   the `_glue_fuse_off` / `test_dense_gateup_fused_marlin_off_*` shape. Neither
   case computes its expectation through the parser under test.
2. **CPU tier, the eligibility predicate.** In
   `tests/vllm/models/test_qwen27_paged_forward.cpp`, a `GdnPackedDecodeEligibility`
   with every remaining term true selects packed decode without any model-shape
   input. RED today.
   **These tests currently pin the dense-only semantics** — `:386-389`, `:666-692`,
   `:897` set `e.dense_model = true` and one `rejects` case asserts that `false`
   deselects. Those assertions are the record of the staging gate, so the change
   must repair them with an argument, never delete them to reach green.
   `tests/vt/test_ops_gdn.cpp:1758` carries a comment describing `in_dt=f32` as
   "the 35B MoE core/z/weight"; the f32 op case stays as the rollback's cover and
   only its comment is stale.
3. **Reachability, through the entry point.** The smallest failing test enters
   through `ModelRegistry::Forward` on a MoE config and observes the dtype of
   `dcore` and `z` at the GDN block. A test that calls `GdnOutDType` by hand proves
   the predicate works and proves nothing about what the model runs.
4. **Model gates at their exact counts.** `test_qwen27_paged_engine` **235/235**
   and `test_qwen36_paged_engine` **315/315**. A changed assertion count is RED
   even when it prints "passed". The 27B count must additionally be reached over a
   byte-identical dense path.

Port no upstream test here: the upstream behavior is a dtype default, not a
kernel, and it has no dedicated test to port. Record that as the reason rather
than as an omission.

## Gates

- **Focused:** the four tests above, at their exact counts.
- **Full:** CUDA `ctest -j 1` on `sm_121a` (`-j 4` OOM-reboots the box, see
  [`.agents/environment.md`](../environment.md)). Configure with
  `-DVLLM_CPP_TRITON=ON`; without `VLLM_CPP_TRITON_CHUNKO_BF16` the bf16 chunk_o
  AOT arm is not compiled in (`src/vt/cuda/cuda_gdn.cu:5327`) and the 35B prefill
  silently falls to the hand path, which would make the prefill measurement a
  measurement of the wrong binary.
- **Correctness, and it comes first.** Greedy continuation on
  `nvidia/Qwen3.6-35B-A3B-NVFP4` against the pinned oracle, under the ratified
  distributional gate the 35B already uses. Establish this before reading any
  speed number.
- **Inertness on the dense arm.** `test_qwen27_paged_engine` 235/235 with no
  toggle set, and again under `VT_GDN_OUT_BF16=1`. Both must be the same result,
  because the dense arm was already bf16.
- **Speed, same binary.** `VT_GDN_OUT_BF16=0|1` A/B on the 35B, order-alternated,
  3 reps, medians, idle host, band measured first, under the `rc` lease
  (`.agents/environment.md`). Report decode and prefill separately: the two arms
  of `ProjectGdnQkvz` differ (the merged fp8 leaf gains a `CastBf16` pass, the
  split leaf does not), so one aggregate number cannot attribute the result.
- **Memory format, not only tokens.** `nsys --cuda-graph-trace=node` on both arms.
  Confirm `dcore` and `z` moved to bf16 by the byte counts, and confirm whether a
  new `CastBf16Kernel` appears on the merged fp8 leaf and at what cost.
- **2.4T:** `PENDING`, named resource. This hardware cannot hold it. Do not
  convert the absence into a claim in either direction.

## Risks

- **The near-tie branch, and this is the real one.** On the 27B, f32 core/z took
  the alternate near-tie branch once the full NVFP4 tactic stack was repaired,
  while bf16 reproduced native vLLM 16/16 (`qwen3_5.cpp:3183-3186`;
  `.agents/specs/nvfp4-small-m-dispatch.md:706-711`). The 35B equivalent: its gate
  is a 315-assertion token comparison and its decode has documented near-tie
  positions, so narrowing the gated-RMSNorm input by one bf16 ulp can flip a top-2
  margin at some position. The direction is toward the oracle, so a flip should
  land **on** vLLM's token rather than away from it — but that is a prediction, not
  a result. If a count moves, re-derive against the pinned oracle's own greedy
  output and report `NEEDS_DECISION`. **Never adjust a golden.**
- **A new pass on the merged fp8 leaf.** See the blast-radius table. This can make
  the merged arm slower while the split arm gets faster. It is a measurement, not
  a blocker, and it is why the A/B reports the two leaves separately.
- **First bf16 exposure of the fp8-fused gated norm.** The 35B's `out_proj` is
  W8A8 fp8, so it takes `vt::RmsNormGatedQuantFp8` (`:4873`); the 27B's is fp4 and
  never did (`:4896-4897`). The op accepts bf16 and the CUDA kernel dispatches it,
  but no gate has run that combination on a real checkpoint. Test 3 was assigned
  to reach it and **cannot**: the call is guarded by FOUR terms,
  `!w.out_proj_fp8.Empty() && GdnOutFp8FuseEnabled() && GlueFuseEnabled() &&
  supports_fp8()` (`:4854-4855`), the synthetic CPU model carries no
  fp8 `out_proj`, and the CPU platform does not report fp8. So Test 3 exercises
  the plain `vt::RmsNormGated` arm and this risk is UNMET on the CPU tier by
  construction, not by omission. It is owed to the 35B GPU gate and recorded
  under `## Now` and `## Owed`; do not read the sentence above as satisfied.
- **GGUF MoE is unmeasured.** The comment being replaced named GGUF explicitly as
  an arm to leave alone. If the GGUF MoE arm cannot be gated in this row, see
  `## Stop conditions`.
- **The 2.4T inherits the flip unmeasured.** It also misses every GDN Triton AOT
  arm (#1170), so its bf16 path would run the spilling hand kernels. Nothing here
  can measure that, and the row must not claim otherwise.

## Evidence

Everything below was read at `dd8a3b0e1` or at the recorded oracle revision.
Nothing in this spec is measured; the row has produced no numbers.

| Claim | Where |
|---|---|
| the default is bf16 dense / f32 MoE | `src/vllm/model_executor/models/qwen3_5.cpp:3190-3197`, call sites `:3749`, `:4155`, `:4386` |
| `outdt` is `dcore` and `z` | `:3807`, `:4253`, `:4587`; `:3584`, `:3688-3690` |
| the gated-norm weight follows it | `:3817-3818`, `:4301-4302`, `:4733-4734` |
| the tree already derived the finding | `:3172-3189` |
| packed decode carries the shape term | `:76-84`, populated `:4406`; entered at `f344decf4` |
| the dtype rule already deselects MoE | `:115-122` |
| `has_packed_ba` is dense-only | `qwen3_5_dense_weights.cpp:437` is the one writer; `qwen3_5_weights.cpp:563-564`; `qwen3_5_gguf_weights.cpp:1067-1070` |
| vLLM keeps core/z at the model dtype | `qwen_gdn_linear_attn.py:870-873`, `:843`, `:859-860`, `:459-465` @ `5559679` |
| vLLM does not gate packed decode on shape | `vllm/envs.py:124` @ `5559679` |
| SGLang agrees | `sglang/srt/models/qwen3_5.py:522-536`, `:515-527`; `gdn_triton.py:43`, `:79-83` @ `f63458b5be` |
| the bf16 chunk_o AOT arm exists for `Hv=32` | `src/vt/cuda/triton_aot_vendored/*/gdn_chunko_bf16_h32.*`, guarded by `VLLM_CPP_TRITON_CHUNKO_BF16` at `cuda_gdn.cu:5327` |
| the AOT arms are pinned to 48 or 32 | `cuda_gdn.cu:5207`, `:5264`, `:5298`, `:5361` |
| the 2.4T has 128 linear V-heads | [`qwen38-text-only.md`](qwen38-text-only.md) |
| the hand kernel spills | [`.agents/kernel-matrix.md`](../kernel-matrix.md), `KERNEL-GDN-PACKED-DECODE`: Triton REG:205 / 0 spill against hand REG:255 + STACK:48 |

### The gap was verified against open work before this spec was written

Four unmerged local branches touch activation dtypes on this model. **None of them
narrows `GdnOutDType` on the MoE arm**, so this row is not a duplicate of any of
them. Each is measured against `origin/main` at `dd8a3b0e1`; all four sit **440
commits behind** it, carry no pull request, and have no spec on `main`.

| Branch | What it does | Does it cover this row? |
|---|---|---|
| `row/PERF-GDN-BF16-CHAIN` (#417, #402) | narrows the GDN **input** chain on fp8 towers: `mixed_qkv`, the conv in/out, the post-conv read, the attention gate, and the dense input-layernorm fp8 quant | **No.** Its levers are `VT_GDN_IN_BF16` and `VT_GDN_FP8_IN_BF16`. Its only mention of `GdnOutDType` is an added comment. See the interaction below — this one is not merely disjoint. |
| `row/PERF-27B-BF16-FP8-OUT` (#339, #417) | narrows the fp8 input projections' GEMM output dtype and the attention gate buffer | **No.** Its spec at `:206-209` treats this row's subject as a *scoping fact*: *"The merged GDN arm is inert on the 35B, by design. `GdnOutDType(cfg.num_experts == 0)` returns f32 for an MoE."* Its diff does not touch the function. |
| `row/REFACTOR-DTYPE-CONSISTENCY` | relocates the model dtype decisions into `include/vllm/model_executor/models/activation_dtype.h` behind a `model_dtype::` namespace | **No — but it MOVES the target.** `GdnOutDType` lands at `activation_dtype.h:187` **verbatim**, comment and `dense_model` default byte-identical, with `qwen3_5.cpp` reduced to `using model_dtype::GdnOutDType;`. Behavior is unchanged, so the gap survives the refactor intact. |
| `row/GUARD-DTYPE-DEFAULTS` | dtype-default guards | **No.** Its diff does not mention `GdnOutDType` at all. |

[#521](https://github.com/mudler/vllm.cpp/issues/521) reports the same
`GdnOutDType` fact a fourth time and asks explicitly for a **spec correction, not
a code change**: *"The spec is a record, so the repair is a correction in the spec
... not a code change."*

So the finding is known and recorded four times, and no open row, branch, or issue
owns narrowing it. That is the gap this row fills.

### Interaction: `row/PERF-GDN-BF16-CHAIN` uses the MoE f32 default as a bound

This one is not disjoint and must not be treated as such. That branch's lever
resolves through `GdnFp8MergedInProjDType` (`qwen3_5.cpp:3280-3282`), whose **third
term is `outdt == BF16`**, and the branch relies on that term being false on a MoE
checkpoint to confine its narrowing to the dense 27B. Edit 1 makes the term true
there.

The consequence is a **rule** consequence, not a default flip:
`VT_GDN_FP8_IN_BF16` stays default OFF (`:3163-3169`, on only for a leading `'1'`),
so nothing moves on the 35B by default in either order. What changes is *why* it
stays inert — a toggle term instead of a model-shape term. Whoever lands
`row/PERF-GDN-BF16-CHAIN` after this row must revisit its eligibility rule and its
Risks section, because "confined to the 27B by the model shape" stops being true
and "off by default" becomes the only remaining bound. Recorded under `## Owed`.

If that branch lands **first**, this row's implementer re-reads its eligibility
rule before Edit 1 and reports `NEEDS_DECISION` rather than silently widening it.

### What this row deliberately does not take on

The same audit found those four branches 440 commits behind `main` with no pull
request and no committed spec on `main`, and found that issues #417 and #426 carry
no row in [`.agents/issue-index.md`](../issue-index.md). Both are real record gaps
and neither is this row's. This spec records them so a later reader does not
mistake the silence for absence, and contradicts none of it.

## Stop conditions

- Any 35B token-gate count moves -> `NEEDS_DECISION`. Do not adjust a golden and
  do not widen a tolerance.
- The GGUF MoE arm cannot be gated in this row -> do **not** reintroduce a
  model-shape term to exclude it. Either gate it, or leave the default at f32 for
  every arm and land the code with the flip behind `VT_GDN_OUT_BF16=1` only,
  naming the missing gate. A shape term would recreate the defect this row exists
  to remove.
- The merged fp8 leaf's new `CastBf16` pass costs more than the narrowing saves
  and the split leaf cannot be isolated -> report both numbers and stop. Do not
  chase it into `VT_GDN_FP8_IN_BF16`, which is #417's lever.
- The disk or the GPU lease is unavailable -> the gates stay `PENDING` with the
  resource named. Never convert an unrun gate into a pass.
- Do not touch the loader (#1169) or the AOT specializations (#1170) in this row.

## Owed

- [#1169](https://github.com/mudler/vllm.cpp/issues/1169) — the merged
  `in_proj_ba` owner is built only by the dense loader, so `has_packed_ba` is
  false on every MoE checkpoint. Until it is closed, packed GDN decode stays
  unreachable on the MoE arms even after Edit 1 and Edit 2, and the stale
  `KERNEL-GDN-PACKED-DECODE` sentence in
  [`.agents/kernel-matrix.md`](../kernel-matrix.md) — *"the launcher guard rejects
  its `Hv=32` shape anyway"* — stays wrong, because `TryTritonPackedDecode` accepts
  `Hv=32` and dispatches `gdn_decode_h32_default` (`cuda_gdn.cu:5207`, `:5239`).
- [#1170](https://github.com/mudler/vllm.cpp/issues/1170) — all four GDN Triton
  AOT arms are pinned to 48 or 32 linear V-heads, so `Qwen3.8-2.4T-A95B` (128)
  runs the hand CUDA kernels the vendoring exists to replace. Not measurable on
  this hardware.
- **The first bf16 pass through `vt::RmsNormGatedQuantFp8`** (guard
  `qwen3_5.cpp:4854-4855`, call `:4873`; reached only when
  `!w.out_proj_fp8.Empty() && GdnOutFp8FuseEnabled() && GlueFuseEnabled() &&
  supports_fp8()`). `## Risks` names
  it and `## Tests` assigned it to Test 3, which cannot reach it on the CPU tier
  for the two structural reasons recorded there. The op accepts bf16 and the CUDA
  kernel dispatches it, so this is an unexercised combination rather than a known
  defect — but nothing here has run it, and the 35B correctness gate is where it
  first will. Owed to that gate; do not treat the Risks entry as discharged.
- `row/PERF-GDN-BF16-CHAIN` (local, unmerged) — its lever's third term is
  `outdt == BF16` and it uses the MoE f32 default as an eligibility bound. Edit 1
  removes that bound. The branch's eligibility rule and Risks section need
  revisiting before it lands; the default-OFF `VT_GDN_FP8_IN_BF16` toggle is then
  the only thing keeping it inert on the 35B. No behavior moves in either merge
  order, so this is a record and review obligation rather than a code one. See
  the interaction note above.
