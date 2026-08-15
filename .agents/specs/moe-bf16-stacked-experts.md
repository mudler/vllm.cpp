# MoE bf16 / 3-D-stacked routed experts: the arm Qwen3.8 needs

**Rows:** `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm`,
`MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#740](https://github.com/mudler/vllm.cpp/issues/740)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Teach `LoadQwen3_5Moe` to read **3-D stacked, unquantized (bf16)** routed experts,
selected by tensor presence, and gate it token-exact on a real published Qwen
bf16 MoE checkpoint.

In scope:

- a stacked bf16 reader for the routed experts, dispatched by what the shard
  index actually contains;
- narrowing `CheckMoeExpertLayoutSupported` so it stops refusing what now loads
  and keeps refusing what remains unsupported;
- a token-exact greedy gate on `Qwen/Qwen3.6-35B-A3B` (bf16) against the pinned
  oracle;
- byte-identical inertness for the NVFP4 path serving the gated 27B / 35B /
  Coder rows.

Out of scope: the shared expert, the attention tower, GDN, the compressed-tensors
`weight_packed` spelling, GGUF, MTP for 3.8, any speed claim, and any claim about
executing `Qwen/Qwen3.8-2.4T-A95B` itself.

## Why this is gateable, and why that was previously misjudged

[#490](https://github.com/mudler/vllm.cpp/issues/490) recorded this arm as owed
but framed it as hardware-blocked, because Qwen3.8 is ~4.8 TB bf16 against
GB10's 128 GB. That framing was too narrow: **what needs proving is the layout,
not the scale.**

`Qwen/Qwen3.6-35B-A3B` (bf16) carries the identical shape at a size that fits:

| | Qwen3.8-2.4T-A95B | Qwen3.6-35B-A3B bf16 |
|---|---|---|
| routed experts | 3-D stacked | 3-D stacked |
| scale tensors | none | none |
| backbone prefix | `model.` | `model.language_model.` |
| size | ~4.8 TB | **71.9 GB, 26 shards** |

71.9 GB fits GB10's unified memory and vLLM runs it, so this arm gets a real
token-exact gate. The differing prefix is a bonus rather than a complication: the
same run exercises `ResolveQwen3_5BackbonePrefix` (landed in #490) on the
namespace it was written for.

**This will be the first time the MoE loader reads a published Qwen bf16 repo.**
Every existing gate reads the requantized `nvidia/Qwen3.6-35B-A3B-NVFP4`, whose
experts are per-expert and quantized — which is precisely why the gap survived
undetected until Qwen3.8 exposed it.

## Upstream chain

| Upstream anchor | Contract to mirror |
|---|---|
| pinned vLLM `vllm/model_executor/models/qwen3_5_moe.py` expert loading | Stacked `experts.gate_up_proj` / `experts.down_proj` are the native published layout; the loader slices per expert rather than requiring per-expert tensors. |
| pinned vLLM `vllm/model_executor/layers/fused_moe/layer.py` weight loader | Expert-parallel slicing of a stacked tensor, and the gate/up split within `gate_up_proj`. |
| local `src/vllm/model_executor/models/gemma4_weights.cpp:326` | In-tree precedent for dispatching between stacked and per-expert layouts by tensor presence. |
| local `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` | The dense arm already routes BF16 / FP8 / NVFP4 by presence; the MoE arm should mirror that shape, not invent a new one. |

Read the pinned oracle's own slicing order before writing any of this. The gate
is token-exact, so a transposed or mis-split `gate_up_proj` will show up as
divergence, not as a load error.

## Design

1. **Dispatch by presence, once.** Probe the shard index for the stacked spelling
   versus the per-expert one, decide a single time, and thread the decision —
   mirroring `ResolveQwen3_5BackbonePrefix`, which resolves the namespace once
   rather than per lookup. A per-lookup fallback would let a checkpoint load half
   from each layout and still appear to succeed.
2. **The stacked reader slices, it does not materialize.** `gate_up_proj` is
   `[num_experts, 2 * moe_intermediate, hidden]`; `down_proj` is
   `[num_experts, hidden, moe_intermediate]`. Confirm both against the real index
   rather than trusting this spec. The gate/up split within the stacked tensor
   must match upstream's ordering exactly.
3. **Alignment.** Stacked slices are offsets into an mmap'd safetensors payload
   and carry no alignment guarantee. Route every typed read through
   `vt::LoadUnaligned` (`include/vt/unaligned.h`), which is the seam
   [#627](https://github.com/mudler/vllm.cpp/issues/627) exists for and which has
   already been bypassed twice by new loaders.
4. **Narrow the refusal, do not delete it.** `CheckMoeExpertLayoutSupported`
   currently refuses three shapes. The stacked-expert branch becomes supported;
   the unquantized-`lm_head` and per-expert-but-unquantized branches must be
   re-examined on their own merits rather than removed as collateral. Every
   refusal test that changes meaning gets updated deliberately, with the reason
   recorded.

## Risks

- **Regression on three gated rows.** This is the shared loader behind 27B / 35B
  / Coder. Mitigated by byte-identical goldens plus re-run SACRED gates — not by
  a green suite.
- **Silent mis-slicing.** A wrong gate/up split or expert stride loads cleanly and
  produces wrong logits. Only the token-exact gate catches it, which is why the
  synthetic unit test is necessary but explicitly not sufficient.
- **Refusal over-narrowing.** Making the refusal permissive enough to accept
  stacked bf16 could also accept a genuinely unsupported shape. Each remaining
  refusal branch needs its own surviving test.
- **Checkpoint availability.** 71.9 GB over 26 shards must be staged before the
  gate can run; that is a prerequisite, not a finding.

## Tests

1. Synthetic stacked-layout load through the production `LoadQwen3_5Moe`, both
   residency paths, RED-first.
2. Byte-equality: the same logical weights expressed stacked and per-expert must
   load to identical bytes.
3. Refusal tests: each shape that remains unsupported still refuses, naming the
   missing piece; the stacked shape no longer does.
4. Inertness: 27B / 35B / Coder suites unchanged, golden md5 unchanged.

## Loadability of the 2.4T itself, without 4.8 TB

The developer's bar is explicit: **the 2.4T must be provably loadable on hardware
that can hold it**, even though nothing here can. "It refuses honestly" is not
that bar, and neither is "the reader works at 35B" on its own.

Splitting the claim into the two halves that can each be proven:

1. **Byte-level correctness of the reader** — proven token-exact at 35B on the
   identical layout (see Gates below). This is what establishes that stacked bf16
   experts are read, sliced and placed correctly at all.
2. **Name, shape and dtype resolution for the 2.4T specifically** — proven by a
   **load-plan dry run against the real published index**. `model.safetensors.index.json`
   for `Qwen/Qwen3.8-2.4T-A95B` is ~1 MB and enumerates all 1609 tensors; it costs
   nothing to fetch and pin. The loader must be able to resolve, for every tensor
   it would request: the exact name, the shape implied by the config (92 layers,
   512 experts, hidden 8192, `moe_intermediate_size` 2048, `num_experts_per_tok`
   10), and the dtype — **without allocating or reading a single weight byte**.

Requirement: a `--dry-run`-style plan path (or a test-only entry point) that walks
the full load for a given config + index and reports every tensor it would fetch,
with the shape and dtype it expects, then asserts that set is exactly satisfied by
the index. Commit the pinned 2.4T index as a test fixture, as this row already
does for the 2.4T `config.json`.

This is the honest form of "it would load on adequate hardware": every name
resolves, every shape agrees with the config, every dtype is one the reader
handles, and the arithmetic that derives per-expert offsets from the stacked
tensors is exercised at 2.4T's actual dimensions rather than 35B's.

**What it deliberately does NOT claim:** a generated token, throughput, memory
headroom, or that any allocation path survives at that scale. Those need the
hardware. State that limit wherever the result is recorded.

Do the same dry run for `Qwen/Qwen3.6-35B-A3B` bf16 and require it to agree with
the real load that follows — otherwise the dry run is an unvalidated model of the
loader rather than a projection of it.

## Gates

- Focused suites plus the full serial gate.
- **The token-exact gate on `Qwen/Qwen3.6-35B-A3B` bf16 is OWED, not met, and
  cannot be met by this row.** This spec was internally inconsistent when
  written and the contradiction is recorded rather than quietly dropped: §Scope
  excludes the attention tower, the shared expert and `lm_head`, while this
  section made a token gate on that checkpoint binding. Both cannot hold. The
  published repo has **zero** scale tensors anywhere — verified against its
  pinned index — so it is bf16 *throughout*, not merely in its experts, while
  `LoadQwen3_5Moe` still requires per-tensor FP8 for the towers and NVFP4 for
  the shared expert and `lm_head`. Such a checkpoint therefore still refuses at
  load, and downloading 71.9 GB would buy a refusal the dry run already proves
  for free at the real index and the real dimensions.
  The remaining bf16 arms are owed to their own row; until then **no token has
  been generated through the stacked reader on any checkpoint.** Both arms identical prompts, token counts, sampling and batching.
- SACRED 27B / 35B / Coder re-run on the GPU box; goldens byte-identical.
- The synthetic tests alone do not close this row. A stacked reader that loads
  without error and produces wrong logits passes every CPU test in this list.

## Evidence required

- RED capture for the stacked-layout test before the reader exists.
- The token-exact gate's real counts, or an explicit statement that it did not
  run and why.
- Golden md5 before/after, and SACRED counts from an actual run.

## Stop conditions

- If the stacked slicing order cannot be established from the pinned oracle's
  source, stop and return `NEEDS_DECISION` rather than inferring it from shapes —
  a plausible-looking guess produces wrong logits, not an error.
- If narrowing the refusal would require deleting a test rather than updating it,
  stop and say so.
- Do not claim Qwen3.8 runs. This row makes the architecture loadable and proves
  it at 35B; the 2.4T checkpoint remains unrunnable on size alone.

## Slicing order, as established from the pinned oracle's source

Phase 1's stop condition did not fire. The order is upstream's, read at the
parity pin `555967922` in
`vllm/model_executor/layers/fused_moe/routed_experts.py`:

| Question | Answer | Anchor |
|---|---|---|
| Which half is gate? | chunk **0** is `w1` (gate), chunk 1 is `w3` (up) | `:1081-1082` |
| `gate_up_proj` layout | normalized to `[E, 2I, H]`, hidden LAST | `:923-926` |
| The split | `chunk(2, dim=1)` — halves of the `2I` axis | `:928` |
| `down_proj` layout | normalized to `[E, H, I]`, hidden at dim -2 | `:929-932` |
| Expert stride | dim 0 (`unbind()` defaults to dim 0) | `:942` |
| Stacked detection | `loaded_weight.dim() == 3` | `:907` |

The third element of a `fused_mapping` tuple is normally an expert id; for the
two fused entries upstream **repurposes it as the chunk index**, and says so in
its own comment at `:927`. That is what makes `chunk(2, dim=1)[expert_id]`
coherent rather than out of range.

Because a swapped gate/up loads cleanly and only shows up in logits, that one
bit was confirmed three further ways: `_load_w13` narrowing w1 to destination
offset 0 and w3 to `shard_size` (`:494-500`, with `SHARD_ID_TO_SHARDED_DIM` at
`:656`); `unquantized_fused_moe_method.py:97-106` allocating `w13_weight` as
`[num_experts, 2 * intermediate, hidden]`; and `activation.py:118-143`, where
`SiluAndMul` is `silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2`, so
SiLU lands on the FIRST half of the fused operand — a confirmation that does not
depend on loader bookkeeping at all. (An earlier revision of this section cited
`activation.py:77,:161` for that last one; those lines are `FatreluAndMul` and
`SiluAndMulWithClamp`. The claim was right and the anchor was wrong, and a wrong
anchor is how a right claim stops being checkable.)

**And HuggingFace declares the axis order outright**, which is what makes
upstream's runtime `shape[-1] != hidden` probe a compatibility branch rather than
the authority. transformers 5.3.0
`models/qwen3_5_moe/modeling_qwen3_5_moe.py:820-821` declares `gate_up_proj` as
`[num_experts, 2 * intermediate_dim, hidden_dim]` and `down_proj` as
`[num_experts, hidden_dim, intermediate_dim]`, and `:842` consumes it as
`linear(x, gate_up_proj[e]).chunk(2, dim=-1)` — `F.linear` being `x @ W.T`,
output column j is row j of W, so gate is rows `[0, I)`.
`modular_qwen3_5_moe.py:164` names the same split in its TP plan:
`"experts.gate_up_proj": "packed_colwise"`. The identical declaration appears in
`qwen3_next` (`:828-829`) and `qwen3_moe` (`:226-227`).

Both published repos ship the **canonical** orientation, read from each shard's
own safetensors header on 2026-08-14: `Qwen/Qwen3.8-2.4T-A95B` `gate_up_proj`
BF16 `[512, 4096, 8192]` / `down_proj` `[512, 8192, 2048]`, and
`Qwen/Qwen3.6-35B-A3B` `[256, 1024, 2048]` / `[256, 2048, 512]`. The transposed
branch mirrors upstream's own `shape[-1] != hidden` probe and is exercised only
synthetically.

## The 2.4T load-plan dry run: result

`PlanQwen3_5MoeLoad` walks the whole load for a config and reports every tensor
it would fetch, without allocating or reading a weight byte. Against the pinned,
verbatim `model.safetensors.index.json` (1609 tensors, 213 shards, 4.89 TB) plus
a shape manifest captured from the shards' own headers, the partition is exact
and clean:

- **1014 planned BF16 tensors: every one present, with exactly the planned
  shape.** That is the whole backbone this arm reads — embeds, norms, every
  layernorm, the router gate, the shared-expert gate, the GDN's bf16 tail, both
  attention norms, and the 184 stacked routed-expert tensors.
- **All 184 loader-*enforced* expert shapes agree** at 512 experts / hidden 8192
  / intermediate 2048 — `[512, 4096, 8192]` and `[512, 8192, 2048]`.
- **1429 planned non-BF16 tensors: none satisfied** — 853 absent (`weight_scale`
  / `weight_scale_2`, which a bf16 repo simply does not have) and 576 present
  but BF16 where the loader demands U8 or F8_E4M3.
- **The unsatisfied set is exactly the three out-of-scope arms**: the FP8
  attention tower, the FP8 GDN tower, the NVFP4 shared expert, and the NVFP4
  `lm_head`. Nothing outside them.
- The only published tensors the plan does not want are `mtp.*` (19, the
  separately-loaded draft head) and, on the 35B, `model.visual.*` (333, the
  vision tower).

**The spec's Phase-3 wording — "asserts that set is exactly satisfied by the
index" — cannot hold as written while §Scope excludes the tower, shared expert
and head.** A published repo is bf16 *throughout*. The dry run therefore asserts
the honest form: every BF16 request resolves exactly, and every unsatisfied
request is a named, already-owed arm. That is a stronger statement than a bare
pass, because it enumerates precisely what a 2.4T load would still be missing
rather than hiding it.

The same dry run runs against the real `Qwen/Qwen3.6-35B-A3B` index, which says —
**before anyone stages 71.9 GB** — that the binding gate's checkpoint will hit
the same three owed arms.

### Both fixtures are pinned to a commit, and the manifest is reproducible

The generator originally fetched `/resolve/main/`. That is the defect
[#471](https://github.com/mudler/vllm.cpp/issues/471) exists for at the gate
level and the one `unsloth/...-27B` demonstrated at the checkpoint level, where a
repo was silently re-quantized from NVFP4 to FP8 under its own name: a manifest
captured from a moving ref records shapes nobody can re-derive. Every fetch is
now pinned, and the revisions are emitted into the generated header and asserted
in the suite:

| Repo | Revision | Published bytes |
|---|---|---|
| `Qwen/Qwen3.8-2.4T-A95B` | `207bd685a7e3696cfaff12ded7c6a7ea0f88c996` | 4,892,365,451,008 |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 71,903,645,408 |

Verified rather than asserted, on 2026-08-14:

- the committed `model.safetensors.index.json` is **byte-identical** (md5
  `c5a85e3e3e255a2560ff6906b0f44577`) to
  `huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/207bd685.../model.safetensors.index.json`;
- re-running the generator at both pinned revisions **reproduces
  `qwen3_5_stacked_shapes.inc` byte for byte** (1609 tensors / 213 shards and
  1045 / 26), with each repo's summed `numel * sizeof(dtype)` equal to the
  `metadata.total_size` its own index declares.

That is what makes the manifest usable as ground truth: it is a capture anyone
can re-derive, not a transcription anyone must trust.

### Why the plan is a projection of the loader and not a second model of it

The suite builds a checkpoint from the plan *alone* and requires the production
`LoadQwen3_5Moe` to read it (the plan is sufficient), then removes each planned
tensor in turn and requires the load to fail naming that tensor (the plan is
necessary). A plan entry the loader never wanted survives its own deletion; a
tensor the loader wants that the plan omits breaks the first load. Both
directions are mutation-proven (M10, M11).

The spec asked for this agreement to be shown against a real 35B load. That load
needs the 72 GB checkpoint and the GPU box, so it is **owed to Phase 5**; the
delete-one-at-a-time round trip is the CPU-side substitute and is strictly
finer-grained.

## Outcome

**Implemented, CPU-gated, GPU gate NOT run.**

Measured: 11/11 test cases, 30892 assertions, `Status: SUCCESS!`. RED captured
first — the stacked cases threw the old refusal (2 cases / 6 assertions failed).
15 mutations, every one RED, source restored byte-exact.

**A fixture defect was found and fixed, and it matters.** The synthetic payload
generator was `0x3d00 + ((i*37 + e*7) & 0x1ff)` — affine in the element index
with period 512. Swapping gate/up shifts the source offset by `I*H` and dropping
the expert stride by `2*I*H`; when those are multiples of the period, both
defects land on identical values. Measured, not reasoned: with that generator,
mutating the reader to swap gate/up *and* to read expert 0 for every expert left
the suite fully green. The replacement is an xorshift-multiply finalizer, and
the case now REQUIREs that the specific offsets the mutations exchange hold
different values. M1 (gate/up swap) and M2 (stride dropped) both go RED.

Rejected: narrowing `CheckMoeExpertLayoutSupported` by deleting its stacked
branch's tests. The two refusal subcases were **retained with their subject
narrowed** — a fully published index still refuses, one layer further in, at the
unquantized `lm_head`. Deleting them would have dropped the only CPU-visible pin
on the claim most likely to be over-read from this row.

### What the CPU evidence does and does not establish

Establishes: the stacked bf16 reader loads through the production entry point on
both residency paths and both namespaces; the gate/up split, expert stride and
both orientations are byte-exact against a payload derived independently of the
loader; the layout is resolved once per checkpoint and threaded; a mixed index
and a non-BF16 stacked tensor are refused by name; the NVFP4 arm is untouched;
and every name and BF16 shape the reader would request from both published repos
resolves, at their real dimensions.

Does **not** establish: a single generated token. The binding token-exact gate on
`Qwen/Qwen3.6-35B-A3B` bf16 has not run. A wrong gate/up split or expert stride
is caught here only because the synthetic payload was built to catch it — at toy
dimensions, against an expectation this row's author wrote. The oracle has not
been consulted at runtime. Nothing here shows the 2.4T allocates, and nothing
here claims Qwen3.8 runs.

## Now

Row lifecycle is still recorded as `READY` above, deliberately. Phases 1-4 landed
on `row/MODEL-MOE-BF16-STACKED-EXPERTS` — slicing order established from upstream
source, stacked bf16 reader implemented and mutation-gated, the load-plan dry run
running against both published indices, and the CPU suite green on a clean
`-Werror` build — but the implementer's authority covered
`qwen3_5_weights.{h,cpp}`, `tests/` and this spec only. **A lifecycle move owes
`docs/STATUS.md`, `docs/BENCHMARKS.md` and the `.agents/roadmap_v1.md` row in the
same change**, and those are the operator's to make; recording `ACTIVE` here
without them would leave the projections disagreeing with the spec.

**Owed, and needing the GB10:** stage `Qwen/Qwen3.6-35B-A3B` (71.9 GB, 26
shards) and run the binding token-exact greedy gate against the pinned oracle;
re-run the SACRED 27B / 35B / Coder gates and confirm goldens byte-identical.
That gate will hit the three owed arms this row's dry run enumerates — the FP8
attention and GDN towers, the NVFP4 shared expert and the NVFP4 `lm_head` — so
loading a published bf16 repo end to end needs those arms too. They are separate
rows, and the dry run now names them precisely rather than leaving them to be
discovered.
