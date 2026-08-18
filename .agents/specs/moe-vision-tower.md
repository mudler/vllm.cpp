# MoE vision tower (M2/M3): image and video on the GDN-hybrid MoE backbone

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#891](https://github.com/mudler/vllm.cpp/issues/891)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Run the Qwen3.5 vision tower on the **MoE** backbone, and gate image and video
token-exact — the arm the dense row already has and this one does not.

In scope:

- stop dropping the checkpoint's `model.visual.*` tensors in the MoE loader;
- the tower forward on the MoE backbone, gated on mm input so the text path is
  byte-identical;
- image→text and video→text token-exact gates on `Qwen/Qwen3.6-35B-A3B`.

Out of scope: audio (this family ships none — no `audio_config`, no
`audio_token_id`), the deepstack path (`deepstack_visual_indexes: []`, compiled
out upstream at `qwen3_vl.py:1709-1716`), any speed claim, and any claim about
executing `Qwen/Qwen3.8-2.4T-A95B`.

## What is missing, precisely

**The tower forward, not the plumbing.** M0/M1 landed the mm input pipeline and
the processor-parity gate passes, so images and video already reach the model
correctly. They have nowhere to go on this backbone.

The dense arm is the template and the proof it is tractable: a forked GDN-hybrid
VL forward, gated on mm input, achieving **image 32/32** and **video 32/32**
token-exact while leaving the text path byte-identical.

## Upstream chain

| Anchor | Contract to mirror |
|---|---|
| pinned vLLM `qwen3_5.py` `Qwen3_5MoeForConditionalGeneration` | The MoE conditional-generation class composes the same vision tower as the dense one over a different text backbone. The tower is not MoE-specific. |
| pinned vLLM `qwen3_vl.py:1709-1716` | `deepstack_visual_indexes: []` compiles the deepstack path out for this family. |
| local: the dense arm's VL forward (`MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`, M3-b/M3-d) | Fork gated on mm input so text stays byte-identical; reuse the windowed tower and video MRoPE rather than writing a second one. |
| local `.agents/specs/mm-tools-scoping-2026-07-10.md` | Records that the 35B's `model.visual.*` tensors are present and **silently dropped** by the loader today. |

Read the dense arm's forward before writing anything. If the two towers diverge
in anything but which backbone consumes their output, that divergence is itself
a finding.

## Design

1. **Load the tower.** The 35B bf16 checkpoint carries 333 `model.visual.*`
   tensors. They are dropped today. A silent drop is the defect class this
   project keeps rediscovering — make their absence a refusal, not a shrug.
2. **Fork the forward, gated on mm input.** Exactly as the dense arm does. When
   no mm input is present the text path must execute the identical instruction
   sequence it does today.
3. **Reuse, do not reimplement.** The windowed tower, the video MRoPE and the
   processor are shared with the dense arm. A second implementation would drift.

## Risks

- **Text regression on a 315/315 gate.** The MoE text row is gated and heavily
  used. Mitigated by byte-identical goldens plus a re-run text gate — not by a
  green suite.
- **Silent modality drop.** If the tower loads but is never invoked, image input
  degenerates to text-only and still produces plausible tokens. The gate must
  compare against the oracle's mm output, not merely check it did not crash.
- **Divergence from the dense tower.** Two implementations of one tower will
  drift; the dense one is already gated at 32/32 both modalities.
- **Memory.** The 35B bf16 is 71.9 GB plus a tower on a 119 GiB box, and a
  global OOM there has already killed unrelated processes.

## Tests

1. Image→text on `Qwen/Qwen3.6-35B-A3B`, token-exact vs the pinned oracle, using
   the **same harness** as the dense arm's 32/32 run.
2. Video→text likewise.
3. Text inertness: the MoE text gate stays 315/315, goldens byte-identical.
4. Loader: a checkpoint whose `visual.*` tensors are absent is refused naming
   them, rather than silently loading a text-only model.

## Gates

- Focused suites plus the full serial gate.
- **Binding: image and video token-exact vs the pinned oracle at 35B.**
- Text inertness proven by re-run, not argued.
- The CPU suite is necessary and **not sufficient**: a tower that loads and is
  never invoked passes every offline test and still answers image prompts from
  text alone.

## Evidence required

- RED capture before the tower runs.
- Real counts per modality, or an explicit statement that one did not run.
- Golden md5 before/after and the text gate's count from an actual run.

## Stop conditions

- If the dense arm's tower cannot be reused, stop and return `NEEDS_DECISION`
  rather than writing a second one.
- If text inertness cannot be held byte-identical, stop — that gate outranks
  this feature.
- Do not claim the 2.4T runs. This row proves the MoE vision path at 35B.

## Now

Row stays `PARTIAL`. **Implementation landed (#891); the binding gate is OWED.**

Landed:

- `LoadQwen3_5MoeVision` reads the checkpoint's `model.visual.*` through the
  SHARED `LoadQwen3VLVisionWeights` the dense 27B arm is gated on at image 32/32
  + video 32/32. The dense tower WAS reusable, so no second tower was written.
  Geometry comes from the checkpoint's `vision_config`; `out_hidden_size` is the
  text hidden size (2048 on the 35B, 5120 on the 27B), the one field the two arms
  disagree on. A checkpoint with NO `model.visual.*` tensor is refused naming
  them.
- `Qwen3_5MoeVLGenerateGreedy[Video]`, forked and gated on mm input. The dense
  arm's greedy core is now TEMPLATED on the weights arm rather than copied, and
  the image/video prefill plan (mask, row-count check, MRoPE index, decode delta)
  is one shared implementation. `ForwardLayers` gained the `mrope_cos_sin`
  injection point its dense sibling already had; `nullptr` on every text and
  graph-captured caller. The MoE decode graph is taken only where
  `ForwardQwen3_5Moe` itself takes it (fp4 CUDA).

Evidence obtained:

- Full CPU suite **479/479 passed, 0 failed**, SERIAL `ctest`, exit 0, on a clean
  out-of-tree `-Werror` build (0 warnings, 0 errors).
- `test_qwen3_5_moe_vision` **7 cases / 38 assertions**. The forward gate is an
  EXACT reduction, not a "the output moved" heuristic: one visual token on a
  1x1x1 LLM grid makes MRoPE degenerate to 1-D positions, so scattering
  `embed_tokens[k]` must reproduce a plain TEXT greedy run over the substituted
  prompt token for token. The complementary 8x8-grid case requires the VL run to
  DIFFER from the 1-D run, which is what sees the MRoPE cache.
- Four mutations driven RED and restored byte-exact: killing the scatter (2 cases
  red), ignoring the injected MRoPE cache (1 case red -- the degenerate case
  cannot see it, which is why the second case exists), dropping the absent-tower
  refusal, dropping the row-count check.
- Thor (`kairos-4db2`, sm_110): clean CUDA build; `test_qwen3_5_moe_vision_hw`
  loads the real 333 `model.visual.*` tensors off `Qwen/Qwen3.6-35B-A3B` and runs
  the tower on the committed fixture image. **sm_110 has fa2 and
  cutlass-fp8/nvfp4 legitimately DISABLED, so that ran the FALLBACK attention
  path** -- valid for correctness, and NOT coverage of the shipped GB10 path. No
  speed number was taken.

**OWED, and nothing here substitutes for it: the image and video token-exact
gates vs the pinned oracle at 35B.** Both blockers were external, not a missing
implementation:

- the pinned oracle cannot run on Thor -- vLLM does not import there
  (`libcuda.so.1` absent on the host; `torch.cuda.is_available()` is False), so
  no 35B mm golden can be captured on that box and none is committed;
- the vision-inclusive bf16 35B is ~67 GiB, against Thor's documented ceiling
  (`.agents/environment.md`: this box REBOOTS instead of OOM-killing, and a 52 GB
  load took it down three times). The full e2e case in
  `test_qwen3_5_moe_vision_hw` is therefore behind `VLLM_MOE_VISION_E2E=1` and
  was NOT run;
- `dgx.casa`, where the dense arm's 32/32 goldens were captured, was mid-run on a
  sibling row's owed gates and off-limits.

**A tower that loads but is never invoked passes every offline test and still
answers image prompts from text alone.** The CPU reduction above closes that on
the synthetic model; only the oracle comparison closes it on the real one.

### Update 2026-08-15: the TEXT arm of this checkpoint is now oracle-gated; the VISION arm is not

The sibling rows [#740](https://github.com/mudler/vllm.cpp/issues/740) and
[#864](https://github.com/mudler/vllm.cpp/issues/864) ran their binding
token-exact greedy gate on `Qwen/Qwen3.6-35B-A3B` bf16
(@ `995ad96eacd98c81ed38be0c5b274b04031597b0`) against the pinned oracle on the
GB10 and PASSED — 6/7 prompts strict 16/16, the seventh an exact-tie divergence
at a bit-identical logprob (#910). That closes the loader and the text backbone
underneath this row's tower. **It says nothing about image or video**, and it is
recorded here only so the next reader does not mistake a green sibling for
coverage of this one.

`#908`'s dense-arm regression check is likewise PARTIAL: the dense TEXT gate at
`2f2bce926` is **235/235**, identical to pre-merge and a true before/after
(binary md5 `db889909d4…` vs `49ded1ece8…`, 500 TUs recompiled). The dense
**image/video** arm was NOT re-verified — network-blocked — so the claim "#891
did not regress the dense arm" holds for text and is UNVERIFIED for the
modalities this row is actually about.

This row's own claim is unchanged and deliberately narrow: **the tower loads and
computes**, not that it produces correct tokens. On Thor (sm_110) the real 333
`model.visual.*` tensors load and the tower runs — grid `[1,28,28] →
[196,2048]`, finite, absmax 2.08 — with text inertness 315/315 unchanged. That
box has `fa2` and CUTLASS fp8/nvfp4 legitimately DISABLED, so it exercised the
FALLBACK attention path: valid for correctness, and NOT coverage of the shipped
GB10 path.

**Still OWED, and it is the whole point of this row: image and video token-exact
vs the pinned oracle at 35B, on GB10, through the shipped fast path.** Row stays
`PARTIAL`; this spec stays `READY` until that gate produces counts.
