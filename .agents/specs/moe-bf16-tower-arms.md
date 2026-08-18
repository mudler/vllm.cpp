# MoE bf16 towers, shared expert and lm_head: the rest of a published bf16 repo

**Rows:** `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm`,
`MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#864](https://github.com/mudler/vllm.cpp/issues/864)
**Lifecycle:** `DONE`
**Owner:** unassigned

## Scope

Give `LoadQwen3_5Moe` a **bf16 arm, selected by tensor presence**, for the four
components that still demand a quantized dtype, and gate the whole thing
token-exact on a published Qwen bf16 MoE checkpoint.

In scope:

- GDN tower (`in_proj_qkv`, `in_proj_z`, `out_proj`) — `qwen3_5_weights.cpp:507-515`;
- attention tower (`q/k/v/o_proj`) — `:536-546`;
- shared expert (`gate/up/down_proj`) — `:760-762`;
- `lm_head` — `:834`;
- narrowing `CheckMoeExpertLayoutSupported` as each arm becomes supported;
- the token-exact gate on `Qwen/Qwen3.6-35B-A3B` bf16 that [#740](https://github.com/mudler/vllm.cpp/issues/740)
  could not meet.

Out of scope: GGUF, MTP, the compressed-tensors `weight_packed` spelling, any
speed claim, and any claim about executing `Qwen/Qwen3.8-2.4T-A95B` itself.

## Why the existing conditional does not already do this

`DenseNativeEnabled()` switches between fp8-**resident** (`LoadFp8Raw`) and
fp8-**dequant** (`LoadFp8Transposed`). **Both branches assume fp8 input.** It is
a build/env lever for an A/B, not a layout probe, so a bf16 tensor fails either
way. This row adds a third path chosen by what the checkpoint actually contains.

Verified: `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.8-2.4T-A95B` both carry **zero**
`weight_scale`, `input_scale` or `scale_inv` tensors anywhere. They are bf16
throughout, not merely in their experts.

## Upstream chain

| Anchor | Contract to mirror |
|---|---|
| pinned vLLM `qwen3_5_moe.py` / `fused_moe/layer.py` | The unquantized path is the default; quantization is a method layered on it, not a precondition for loading. |
| local `qwen3_5_dense_weights.cpp` (`IsNvfp4Projection`, `LoadDenseLmHead` / `LoadLmHeadAnyDtype`) | The dense arm already routes BF16 / FP8 / NVFP4 **by tensor presence**. Adopt that shape; do not invent a parallel one. |
| local `ResolveQwen3_5MoeExpertLayout` (#740) | Resolve a layout **once per checkpoint** and thread it, rather than probing per lookup. |
| local `include/vt/unaligned.h` | Every typed read of a safetensors payload goes through `vt::LoadUnaligned`. |

Read the dense arm's routing before writing anything. If the MoE and dense
probes disagree about what "this projection is bf16" means, that divergence is
itself a defect.

## Design

1. **Probe once, thread the decision.** One resolution per checkpoint covering
   all four components, mirroring #740's expert-layout resolver. A per-lookup
   fallback would let a checkpoint load half quantized and half bf16 and still
   appear to succeed — the exact failure #740's mixed-index refusal exists to
   prevent.
2. **Do not widen `DenseNativeEnabled()`.** It is an A/B lever with recorded
   evidence attached; overloading it with a third meaning destroys that. The
   bf16 arm is selected by presence, independently of it.
3. **Alignment.** `vt::LoadUnaligned` for every typed read. This seam has now
   been bypassed three times by new code paths; assume the next reader will
   forget unless the pattern is obvious at the call site.
4. **Narrow the refusal deliberately.** As each arm lands,
   `CheckMoeExpertLayoutSupported` stops refusing that shape and keeps refusing
   the rest. Update each refusal test with its reason; never delete one as
   collateral.

## Risks

- **Regression on three gated rows.** This is the shared loader behind 27B / 35B
  / Coder. Mitigated by byte-identical goldens **plus re-run SACRED gates** — not
  by a green suite. `test_qwen36_weights` reports SUCCESS while skipping its
  real-checkpoint arms, so it proves nothing on its own.
- **Silent wrong-dtype load.** A bf16 tensor read through a path expecting fp8
  scales, or a missing dequant, produces wrong logits rather than an error. Only
  the token-exact gate catches it.
- **Probe divergence.** If the MoE probe disagrees with the dense one, a
  checkpoint could route differently through two loaders in the same build.
- **Memory.** bf16 is ~2x the fp8 footprint; 71.9 GB of weights on a 128 GB box
  leaves little headroom, and a global OOM there has already killed unrelated
  processes. Size the run before starting it.

## Tests

1. Synthetic bf16 load through production `LoadQwen3_5Moe` for each of the four
   components, RED-first.
2. Mixed-dtype refusal: a checkpoint quantized in some components and bf16 in
   others is refused, naming what disagrees.
3. Probe agreement: the MoE and dense probes classify the same projection
   identically.
4. Inertness: 27B / 35B / Coder suites unchanged, golden md5 unchanged.

## Gates

- Focused suites plus the full serial gate.
- **Binding: token-exact greedy on `Qwen/Qwen3.6-35B-A3B` bf16 (71.9 GB, 26
  shards) vs the pinned oracle**, identical prompts, token counts, sampling and
  batching. This is the first real token evidence for the stacked expert reader
  landed in #740, which has none.
- SACRED 27B / 35B / Coder re-run on GPU; goldens byte-identical.
- The CPU suite is necessary and **not sufficient** — every failure mode above
  except a missing tensor passes it.

## Evidence required

- RED capture per component before its arm exists.
- The token-exact gate's real counts, or an explicit statement that it did not
  run and why.
- Golden md5 before/after and SACRED counts from an actual run, not a
  construction argument.

## Stop conditions

- If the dense arm's probe cannot be reused or mirrored faithfully, stop and
  return `NEEDS_DECISION` rather than writing a second, subtly different
  classifier.
- If closing an arm would require widening `DenseNativeEnabled()`, stop and say
  so.
- Do not claim `Qwen/Qwen3.8-2.4T-A95B` runs. This row makes the architecture
  loadable and proves it at 35B; the 2.4T remains unrunnable on size alone.

## Outcome

**All four arms landed, selected by tensor presence, and the CPU gate is
green.** `LoadQwen3_5Moe` now resolves the GDN tower, the attention tower, the
shared expert and `lm_head` ONCE per checkpoint from the shard index, through
`ResolveQwen3_5MoeTowerDtypes` / `ClassifyQwen3_5Projection`, and threads the
decision the way #740 threads the routed-expert layout. `DenseNativeEnabled()`
was not touched.

### What was measured

- RED captured per component on the base commit `b4220cf45`, each failing for
  its own reason: `expected F8_E4M3 for ...self_attn.q_proj.weight`,
  `expected F8_E4M3 for ...linear_attn.in_proj_qkv.weight`,
  `expected U8 for ...mlp.shared_expert.gate_proj.weight`, and
  `an unquantized lm_head is not implemented ...`.
- `test_qwen3_8_text_only` went 11 cases / 37,118 assertions to 18 cases /
  67,816 assertions, Status SUCCESS.
- The PUBLISHED `Qwen/Qwen3.6-35B-A3B` (1045 tensors) and
  `Qwen/Qwen3.8-2.4T-A95B` (1609 tensors) manifests now satisfy the load plan
  COMPLETELY — `missing` and `mismatched` both empty, where before this row both
  were required to be non-empty. All four components resolve BF16 from the real
  manifests' own dtypes, and neither repo carries a `weight_scale`,
  `input_scale` or `scale_inv` tensor anywhere.

### What was rejected, and why

**One decision for the whole checkpoint.** §Tests asks for "a checkpoint
quantized in some components and bf16 in others" to be refused. It is NOT, and
deliberately: `nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed` — an FP8 attention
tower beside an NVFP4 MLP and a BF16 GDN in-projection — and the dense arm reads
it by asking per projection. Refusing a cross-component mix would diverge from
the dense ladder this row's own §Stop-conditions require it to mirror, and would
refuse a checkpoint that already loads through the sibling loader. So the four
components are four INDEPENDENT decisions, and what is refused is a component
that disagrees with ITSELF — layer 0's `q_proj` BF16 beside layer 4's F8_E4M3 —
naming both projections and both dtypes. That is the failure the once-per-
checkpoint discipline exists to prevent; a cross-component mix is not.

**Reuse of the dense classifier by call rather than by mirror.**
`IsNvfp4Projection` lives in `qwen3_5_dense_weights.cpp`'s anonymous namespace
and that file was outside this task's authority, so `ClassifyQwen3_5Projection`
mirrors the ladder instead and a test BINDS the two behaviorally: the same
synthetic projection is loaded through the production dense loader and the slot
it filled is compared against what the classifier says, for BF16, FP8 and
ModelOpt NVFP4. Comparing the classifier with itself would prove nothing.
Routing the dense arm's own ladder through the shared function is owed and needs
its own row.

**Why the head is not `LoadDenseLmHead`.** That entry point applies
`DenseLmHeadFp4Enabled()`, sets `keep_dequant_b`, and zeroes `alpha`; adopting it
would change the NVFP4 head behavior of three gated rows. The NVFP4 arm stays
`LoadNvfp4Raw` byte-unchanged and the bf16 arm is `LoadBf16Transposed`.

### One defect found and fixed in flow

The plan-deletion sweep showed that removing an NVFP4 projection's
`weight_scale_2` made the classifier fall through to BF16 and report
`expected BF16 for lm_head.weight` — the exact shape of complaint #490 exists to
stop, because it reads as a corrupt checkpoint rather than an absent tensor. The
DENSE arm has the same hole. `ClassifyQwen3_5Projection` now refuses a `.weight`
that is neither BF16 nor absent, naming the missing `weight_scale_2` /
`weight_packed` / `weight_scale` companion. No well-formed projection changes
answer.

### What the CPU evidence does NOT establish

No token, no throughput, no memory headroom, and no weight byte of a real
checkpoint. Every failure mode of these four arms except a missing tensor — a
wrong dtype path, a missing dequant, a transpose that loads cleanly — produces
WRONG LOGITS rather than an error. Only the binding token-exact gate closes
those, and a green suite must not be read as correctness.

### The binding gate RAN, and PASSED — the first token this loader ever produced from a published bf16 Qwen MoE repo

Measured 2026-08-15 on the GB10, from the staged checkpoint described below.
Greedy, 7 prompts x 3 repeats x 16 tokens, identical prompts, token counts,
sampling and batching on both arms, against the pinned oracle asserted per run
as vLLM `0.23.1rc1.dev1511+g555967922`, flashinfer `0.6.15.post1`, torch
`2.13.0+cu130`. Build fast path asserted (`cutlass-nvfp4`, `cutlass-fp8`,
`fp4-mma`, `marlin-nvfp4`, `fa2` all ENABLED for `[121a]`, Triton AOT
`sm_121a`). The oracle's own greedy decode was deterministic 7/7 across repeats,
so a token-exact gate — not a distributional one — is the right instrument here.

**6/7 prompts STRICT 16/16; 108/112 positions.** The one divergence is prompt
`"import numpy as np"` at position 7: oracle `464 "import"`, ours `1445 "from"`,
**both at logprob `-0.8293954133987427`**, `top2_gap_mnats = 0.0`, our token at
rank 2 in the oracle's top-20. An exact tie, broken differently —
`torch.argmax` takes the lowest maximal index and our on-device argmax takes the
higher. **PASS under the ratified near-tie doctrine.** It is not filed away as
noise: [#910](https://github.com/mudler/vllm.cpp/issues/910) owes the mirrored
tie-break, which would make this 7/7 STRICT.

**`108/112` is NOT a quality score.** Only the first divergence in a prompt is
validly adjudicable; after it the two arms carry different prefixes and every
later position compares two different conditionings. Recording it as a ratio
would invite exactly that misreading, so the binding statement is "6 of 7
prompts strict; the seventh diverges once, on a bit-identical logprob".

SACRED inertness — the risk §Risks names first, answered with real counts rather
than a green suite — 3 of 3 against the shared loader all four arms changed,
`GOLDENS_BYTE_IDENTICAL=1` on every one: `test_qwen36_paged_engine` 2/2 cases
**315/315** (`M0-EXIT: produced 16/16 tokens`), `test_qwen27_paged_engine` @
`890bdef7a42feba6d83b6e17a03315c694112f2a` **235/235**,
`test_qwen3coder_paged_engine` @ `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`
**138/138**. 688 assertions in total.

**No throughput, latency or memory number exists for this checkpoint**, and none
is implied. §Scope excluded speed claims and the run measured tokens.

## Owed

- [#910](https://github.com/mudler/vllm.cpp/issues/910): our on-device argmax
  resolves an exact logit tie toward the HIGHER token id where `torch.argmax`
  takes the lower. Found while adjudicating this row's gate. Benign here and
  deterministic everywhere, so it is a permanent behavioral divergence from the
  reference at any tied position, and it costs a STRICT gate. Needs a RED-first
  test that constructs a real tie — one asserting only "argmax returns a
  maximum" passes both conventions — plus inertness across the existing greedy
  goldens.

## Now

**This row is `DONE`.** Every gate it declared is met: the CPU suite (476/476
serial, `-Werror`, RED captured per component), the binding token-exact greedy
gate on `Qwen/Qwen3.6-35B-A3B` bf16 against the pinned oracle, and the SACRED
27B / 35B / Coder re-run with byte-identical goldens.

The checkpoint is staged and verified at
`$CHECKPOINT_ROOT/qwen3.6-35b-a3b-bf16`, revision
`995ad96eacd98c81ed38be0c5b274b04031597b0` — the same revision the committed
shape manifest pins — with all 26 shard sha256 digests recomputed against the
Hugging Face download metadata and the summed tensor bytes equal to the index's
declared `71,903,645,408`. The previous session's blocker (`dgx.casa` carrying
an `ltx2-gen` render, 78 GiB of 119 GiB resident) had cleared.

Neither owning matrix row moves off `PARTIAL` on this evidence.
`MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` still owes its
image/video token gates ([#891](https://github.com/mudler/vllm.cpp/issues/891)),
and `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` still owes a run through
`Qwen3_5MoeForCausalLM` itself, whose only published checkpoint does not fit
this hardware ([#490](https://github.com/mudler/vllm.cpp/issues/490)). What this
row closes is the loader, and it closes it with a token.
