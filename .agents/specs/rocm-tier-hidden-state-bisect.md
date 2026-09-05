# Localising the ROCm/CPU tier divergence to a layer and an op

Row `BACKEND-ROCM`. Issue
[#2590](https://github.com/mudler/vllm.cpp/issues/2590).

Sibling records:
[#2546](https://github.com/mudler/vllm.cpp/issues/2546) (the gate run that
measured the term),
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) and
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (both fixed and landed;
neither explains this),
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the decode number this
gate blocks).

Predecessor specs:
[`rocm-gfx1151-q4k-token-gate-v2.md`](rocm-gfx1151-q4k-token-gate-v2.md), which
measured the term and lists it under `## Owed`, and
[`qwen38-27b-q4km-logit-dump.md`](qwen38-27b-q4km-logit-dump.md), whose
`## Outcome` names the per-layer hidden-state bisect as the next dispatch and
records why one statistic was not enough.

## Now

`ACTIVE` — measured. The instrument is built and gated, the comparison ran on
`strix:gpu0`, and §7's outcomes **P2 and P3 are both selected, on different
steps**. Nothing is fixed, because nothing here is a defect. See `## Outcome`.

## 1. Why this row is different from every predecessor on this arm

Every previous divergence measurement on the Q4_K_M arm compared **us against an
oracle**, and the oracle at this pin is not deterministic across its own kernel
paths: three of the six contested steps in the v2 gate are llama.cpp disagreeing
with itself, and at those three our two tiers emit the same token. So an
oracle-scored result mixes two effects and the denominator itself is contested.

Three steps are not like that. At `p1/45`, `p3/45` and `p4/14` **our own ROCm
tier and our own CPU tier compute a different argmax over an identical prefix on
an identical artifact, with no oracle in the comparison** (v2 evidence, the
`A vs D` column). Two implementations of one computation disagree. One is wrong,
or both are, and the comparison needs nothing external to adjudicate it.

That makes this row oracle-free. It needs no llama.cpp build, no token gate, no
scoring policy, and no ratification of any band. It needs one instrument that
neither tier has ever had: the per-layer hidden state, from both tiers, on the
same prefix.

The term is not one-signed, and any explanation has to survive that. ROCm loses
`p1/45` and `p3/45` and **wins** `p4/14`, which is one of the three steps the
CPU tier is convicted at. "ROCm is less accurate" is refuted before this row
starts.

## 2. Scope

In scope:

1. Repairing the two env-gated activation dumps this file already carries in the
   dense paged path so that a cross-tier comparison is possible and auditable
   (§3). They are dumps that already exist; none of them can currently support
   this comparison, and §3.1 says exactly why for each.
2. One host-side comparator that joins two dump directories and prints the
   per-layer delta profile (§4).
3. One `rc` job on `strix:gpu0` that builds two binaries from **one** source
   tree — the HIP arm and a CPU-only arm — runs both over the same prompt with
   the dumps on, and produces the profile (§5).
4. Reporting where the two tiers separate, and whether that is a defect or
   arithmetic (§6), against outcomes pre-registered in §7.
5. Fixing it, if §6 names a defect that is ours.

Out of scope, deliberately:

- **The token gate's scoring policy.** Adjudicated and landed under #2534: the
  ratified band fails this arm on `n_divergent`. No scoring change is in scope
  and none is argued for here.
- **Every speed, latency and memory number.** `AGENTS.md` §Gates admits none from
  this arm until its declared token gate passes. The harness prints none and
  none is transcribed.
- **#2511 and #2534.** Both fixed, both landed, neither re-litigated.
- **The oracle.** No llama.cpp build, no oracle leg, no margin recomputation.
  Every number this row produces is ours against ours.

## 3. The instrument

### 3.1 What exists, and why none of it can answer this question

Three dumps already exist in `src/vllm/model_executor/models/qwen3_5.cpp`, and
each of them fails this comparison in a different way. They are listed with their
defects because "the instrument already exists" is exactly the belief that would
make somebody run it and read a wrong answer.

| dump | where | why it cannot answer this |
|---|---|---|
| `VT_DUMP_ACT`, per layer | the dense paged loop | writes `hidden` ONLY. In this model the state after layer *l* is `hidden + res`: the two halves are carried separately and summed at the final norm. Half a stream cannot be compared. |
| `VT_DUMP_ACT`, `layer_-1_{hidden,res}` | the MoE paged loop | a pre-layer snapshot only, and on the wrong model class — a 27B Q4_K_M GGUF is the DENSE arm. |
| `VT_DUMP_ACT_SUB`, four stages | `RunDenseLayerPaged` | keyed by a `static thread_local` counter that is never reset, so a file name encodes an invocation ordinal rather than (step, layer), and any extra forward call on either side silently shifts the whole join. |

All three share two defects that matter more than any of the above:

- **A failed open writes nothing and says nothing.** Each writer is
  `if (f != nullptr) { fwrite; fclose; }`. A dump into a directory that does not
  exist, or a short write onto a full disk, is indistinguishable from a dump that
  was never switched on. `AGENTS.md` §Gates and
  [`verification.md`](../verification.md) name this class, the logit-dump row
  burned a lease on exactly it (`ALIGNMENT=BROKEN checked=0 bad=6`), and it is
  the first thing repaired here.
- **Nothing records what was dumped.** The blobs are raw bytes with no dtype, no
  shape and no provenance. A reader must infer the element width from the file
  size, and two dumps of different dtypes compare as garbage without either side
  saying so.

### 3.2 What this row builds

One writer, used by every activation dump in the dense path.

1. **It refuses by name.** A failed `fopen`, a short `fwrite` and a failed
   `fclose` are each a `VT_CHECK` naming the env variable, the path and the
   errno. A dump that writes nothing fails the run.
2. **It writes a manifest.** `<dir>/manifest.tsv`, one row per blob:
   `step, layer, stage, dtype, rows, cols, bytes, path`. The comparator joins on
   the first three and refuses on any disagreement in the rest.
3. **It narrates once.** One stderr line per process, before the first blob:
   the directory, the device type the model is running on, the model class, the
   layer count, and which of the two knobs is set. A reader of the log can see
   what the instrument was pointed at without reading the source.
4. **It counts.** One stderr line at the end of each forward: how many blobs this
   step wrote. Zero is a refusal, not a silence.

Then the two knobs are keyed correctly and made complete:

- `VT_DUMP_ACT=<dir>` writes **both** halves of the residual stream, per
  (step, layer): `s<step>_l<layer>_hidden.bin` and `s<step>_l<layer>_res.bin`.
  The comparator reconstructs `hidden + res`, which is the quantity the final
  norm consumes and the only one that is the model's hidden state.
- `VT_DUMP_ACT_SUB=<dir>` writes the four existing stages per (step, layer):
  `post_input_norm`, `block_out`, `post_attn_norm`, `mlp_out`. `block_out` is
  the GDN block's output on a linear-attention layer and the full-attention
  block's output on the others, which is what separates the two families.

`step` is the forward-call ordinal of the dense paged forward, taken explicitly
rather than inferred from a running counter. `layer` is the loop index, passed
into `RunDenseLayerPaged` as an argument the way the MoE `RunLayerPaged` already
takes one, so the sub-stage files stop depending on a thread-local ordinal.

Neither knob is new: both are already on
[`../../scripts/env-doc-allowlist.txt`](../../scripts/env-doc-allowlist.txt).
This row adds no environment variable.

### 3.3 The dump must not change what it measures

The dumps force a device synchronize per layer, so they are inert by default and
must never be set on a graph-capturing run.

**The graph control this section first demanded is unnecessary, and reading the
tree rather than assuming is what removed it.** The draft required the dumped
ROCm leg to run `VLLM_CPP_CUDAGRAPH=0` and to be compared against a
graph-captured leg, on the belief that ROCm decode is captured by default. It is
not. `vt`'s ROCm backend can capture (`RocmBackend::SupportsGraphCapture()` is
true, `src/vt/rocm/rocm_backend.hip:337`), but the driver's predicate is a
conjunction, and its third term is
`platforms::GetPlatform(...).support_static_graph_mode()`
(`src/vllm/model_executor/models/qwen3_5_dense.cpp:179-181`). `RocmPlatform` does
not override it and takes the base `false`
(`include/vllm/platforms/interface.h:344`), which `src/vllm/platforms/rocm.cpp:91-97`
states in its own words: engaging a real model's decode-graph path is a later
work item. `CpuPlatform` leaves it false too. **Both tiers are eager already**,
both enter the same `DenseForwardLayers`, and a knob would have changed nothing
while implying it had.

That leaves one control, which fails the job rather than being reported:

- **Identity.** The generated ids with the dumps ON are byte-identical to the
  same run with them OFF, per tier. Without this the instrument could be
  reporting its own perturbation, and no reading of the profile would be worth
  anything.

### 3.4 Two candidates the artifact itself decides, before any GPU work

Both are residency divergences between the tiers rather than arithmetic, and
both are cheap to settle. They are recorded here because a candidate refuted for
free is worth more than a candidate ranked by argument, and this campaign has
already spent four dispatches on the latter.

- **F16 weights expand on ROCm and are kept on the CPU tier.**
  `DeviceKeepF16Supported` is `dev != kROCM`
  (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:154-156`), so an F16
  file weight loses three mantissa bits on one tier and not the other.
  **REFUTED for this artifact.** Its 866 tensors are 456 F32, 294 Q4_K, 67 Q6_K,
  48 Q5_K and one Q8_0 (`blk.64.nextn.eh_proj`, the MTP head, which this arm does
  not run). There is no ggml type 1 in the file, read from its own tensor table.
  Every quantized type present is in ROCm's keep-quant set
  (`DeviceKeepQuantSupported`, `:136-149`), so the two tiers keep identical
  residency for every weight the forward touches.
- **The vocabulary table does NOT keep identical residency.**
  `DeviceQuantGatherSupported` asks whether `kEmbeddingQuant` is registered
  (`gguf_keep_quant.cpp:194-196`); ROCm registers only the float-table
  `kEmbedding`, so `token_embd.weight` — Q4_K in this file, 248320 x 5120 —
  expands to bf16 on ROCm and stays Q4_K blocks on the CPU tier. **This is live**,
  and it is the reason §3.2 adds the post-embedding snapshot the dense path never
  had: without a `layer -1` row, "the two tiers embed the token differently" and
  "layer 0 computes differently" are the same measurement.

## 4. The comparator

`scripts/tier-hidden-delta.py`. Host-side, no device, no build.

It reads two dump directories and their manifests, joins on
`(step, layer, stage)`, and for every joined pair reports `max_abs`, `rms`,
`rel_l2` and the index of the worst element. It prints, in its own output and in
words, **which directory it took as A and which as B**, how many rows each
manifest declared, how many joined, and how many were dropped and why. A
comparator that silently intersects two half-populated directories reports a
clean profile over almost nothing; the drop count is what makes that visible.

It refuses, by name, on: a missing manifest, a manifest with zero rows, a shape
or dtype disagreement on a joined key, a join that covers fewer keys than either
side declared, and a blob whose byte length disagrees with its manifest row.

## 5. The measurement

One `rc` lease on `strix:gpu0`, detached, never `ssh`.

**One source tree, ONE binary, two tiers.** The draft of this section said the
CPU tier is a `-DVLLM_CPP_HIP=OFF` build and that the job therefore builds twice.
It does not have to. `vllm-bench` never sets `EngineParams::device`, so it
resolves `Device::kAuto` through the platform registry, and `RocmPlatform`
registers at static init only when `vt::rocm::DeviceAvailable()` — that is,
`hipGetDeviceCount(&n) == hipSuccess && n > 0`
(`src/vllm/platforms/rocm.cpp:178-190`, `src/vt/rocm/rocm_backend.hip:542-545`).
Removing the device from the container's view selects `kCPU` from **the same
`libvllm.so`**. One build, one library hash printed once and used by both arms,
and "the two tiers" cannot become "two trees" by construction rather than by
assertion.

**The tier is measured, not requested.** An environment variable that quietly
did not take would leave the CPU leg running on the board and the comparison
would read as two tiers agreeing perfectly — the exact shape of an instrument
pointed at the wrong thing. So each leg asserts which kernels actually ran, from
`VT_OP_PROVIDER_STATS=1`: the ROCm leg needs a non-zero `device=5` count and zero
reference-tier hits, and the CPU leg needs a `device=5` count of **zero** and a
non-zero `device=0` count. Either way round, the job fails rather than reports.

**One host.** Both arms run on the same box, in the same lease, on the same
boot, over the same artifact staged to worker-local `/tmp` and re-hashed there.
The recorded CPU-tier result was measured on `thor` (aarch64) and the ROCm one
on `strix` (x86-64), so the recorded pair also crosses an architecture. Running
both here on x86-64 removes that variable, and §7's P3 is the outcome where it
turns out to have been carrying part of the effect.

**One prompt at a time.** `p1` ("The three primary colors are"), 48 tokens,
greedy, concurrency 1, MTP off, the same `--num-blocks 256` and seed the
predecessor harnesses used. `p3` and `p4` follow on the same binaries. All three
are run: §Stop conditions forbids reporting a result from one step.

Assertions inside the job, each failing it rather than being reported:

1. `git rev-parse HEAD` in the worker clone equals the branch head this spec
   names.
2. Both builds are clean (`rm -rf` first) and both `libvllm.so` are hashed. The
   HIP and CPU libraries must differ; two equal hashes mean one build did not
   happen. `vllm-cli` is a ~26 KB thin client and is never the identity.
3. The ROCm leg reports `reference_tier_hits=0` with a non-zero `device=5` op
   count, with the dump on. A leg that fell back to the portable tier did not
   measure the ROCm tier.
4. Both identity controls of §3.3 pass.
5. Each manifest declares `layers x steps` rows and the comparator's drop count
   is zero.

## 6. Reading the profile: separating a defect from fp non-associativity

Two tiers reduce in different orders and will differ in the last bits at every
layer. **That difference is not the finding**, and a report of "the first
non-zero layer" would be a report of layer 0 every time. Three readings decide
it, and all three are printed:

- **The run-to-run floor.** Each tier is run twice with the dumps on and
  compared against itself. Both are expected to be exactly zero (each tier was
  self-reproducible across six legs in the v2 gate). A zero self-delta means the
  cross-tier profile has no run-to-run component and every part of it is the
  tier. A non-zero one changes the question and is reported as the headline.
- **Shape, not magnitude.** Under pure rounding, `rel_l2` accumulates smoothly
  across the 64 layers. A **discontinuity** — one layer or one stage where the
  delta jumps by a factor the neighbouring layers do not — is a defect signature.
  A smooth ramp is not, however large it gets.
- **Amplitude against the thing that flipped.** A per-layer delta only matters if
  it can move the final logit gap by the size of the gap that actually flipped:
  0.131054 at `p1/45`, 0.006284 at `p3/45` and 0.092752 at `p4/14`, as this
  run's HIP oracle measured them. The profile is therefore read together with the
  final logit delta at the same step, which `VT_DUMP_LOGITS` (#2534, already
  landed and reachable from both sampling paths) supplies on both arms.

**A key-set mismatch is a finding, not a harness error.** The comparator joins on
`(step, layer, stage)` and refuses when the two sides do not cover the same keys.
That refusal has a meaning here beyond hygiene: the GDN block dumps different
stage names on different branches, so two tiers that took structurally different
routes through the mixer would not produce the same key set. Reported as what it
is, and never worked around with `--allow-drops` before it is explained.

**The GDN split is reported separately.** This model is a GDN hybrid: linear-
attention layers and full-attention layers are different code, and the
predecessor note has flagged that split since 2026-08-23. Every table separates
them, and a term that lives in only one family is a much stronger localisation
than one that lives in both.

## 7. Pre-registered outcomes

Written before any of our hidden states have been seen. The logit-dump row's
`## Outcome` records why this matters and also why it is not sufficient: every
branch there was framed on one statistic, and the cause lived in a property that
statistic could not express. So §7.5 admits a finding outside the statistic, and
it is not a courtesy clause.

**P1 — a discontinuity.** One layer, or one stage inside one layer, where the
cross-tier `rel_l2` jumps against its neighbours. Then the term is localised to
that op, the GDN/full-attention split says which family carries it, and the row
proceeds to a red-before/green-after fix. This is the shape the gfx1100 0.8B
divergence had, where `fa0_q` read `rms-rel 1.196` with its siblings clean and
the cause was a kernel dispatched on the source dtype instead of the output
dtype.

**P2 — a smooth ramp.** No discontinuity; `rel_l2` grows monotonically from
layer 0 at a rate consistent with per-layer rounding, and reaches a final logit
delta of the size of the contested gaps. Then the divergence is irreducible
ordering between two kernel families, the row says so plainly with the profile
as evidence, and it states what that means for token-exactness BETWEEN our
tiers — which is a different and weaker claim than token-exactness against an
oracle, and one this project has never written down.

**P3 — it does not reproduce.** The x86-64 CPU arm agrees with the ROCm arm at
all three steps. Then part of the recorded `A vs D` divergence was the aarch64
CPU kernels, not the ROCm ones, the recorded comparison is restated as a
three-way one, and the row reports that rather than forcing P1 or P2 onto it.
This outcome is admissible and is not a failure of the experiment.

**P4 — the trunk is clean and the head is not.** The per-layer profile stays at
the rounding floor through layer 63 and the separation appears at the final norm
or the `lm_head`. Then this is #2534's resolution term on the ROCm side rather
than a trunk defect, and the fix belongs at the head.

**P5 — admitted outside the statistic.** The profile is not stationary across
the 48 steps: clean at most steps and spiking only at the contested ones. That
would say the term is input-dependent — a shape none of P1 to P4 assumes — and
would redirect the bisect from "which layer" to "which input". It is reported if
it occurs.

## 8. Risks

- **The instrument perturbs the arm.** Controlled in §3.3. Without the identity
  control, the whole profile could be an artefact of the synchronize.
- **The divergence needs the incremental KV cache.** These are decode steps
  reached after 45 tokens of generation. Re-prefilling the prefix in one shot
  would run different kernels and might not reproduce the flip, so both arms
  generate naturally from the prompt and the dump keys on the step.
- **The container has no `github.com` egress.** Sources are staged as a bundle
  from the share and the staged tree is verified against its own revision, the
  way the predecessor jobs do it.
- **The box.** `gfx1151` completed 6 of 6 legs of a heavier workload after
  #2511, with no knobs. If it faults here that contradicts a landed result and
  is the headline, not a nuisance.
- **A CPU-only 27B Q4_K_M decode is slow.** Only the three contested prompts are
  run, and the job is resumable per arm.

## 9. Gates

This row produces a **localisation**, and a fix only if §6 names one. It changes
no gate. `TOKEN_GATE` stays `FAIL` as the v2 evidence left it, and no speed,
latency or memory axis becomes admissible.

The instrument's own gate is a hermetic test of the writer and the comparator:
the refusal fires on an unwritable directory, the manifest round-trips, and the
comparator refuses a shape mismatch and a short join. Each assertion is
mutation-proven: deleting the guard reds exactly its own case.

## 10. Evidence required

- The rc job id, the device, the boot id, the raw log paths, both `libvllm.so`
  hashes and the asserted source revision.
- The two identity controls and the eager/graph control, verbatim.
- The self-delta (run-to-run floor) for each tier.
- The per-layer delta profile for at least one contested step, both tiers, with
  the GDN and full-attention layers separated.
- The first layer and stage where the tiers separate beyond the floor, with the
  calibration of §6 that distinguishes a defect from a ramp.
- The outcome named against §7.
- If fixed: red-before and green-after, and what happens to all three steps.

## 11. Stop conditions

- **Do not report a fix from one step.** All three are checked and each one's
  result is stated, including a step the fix does not move.
- **Do not weaken the token gate**, and do not touch its scoring. Not in scope.
- **Do not quote a speed number**, even as a by-product.
- **"Localised but not fixed" is a complete answer.** Report it rather than
  attempting a speculative fix.
- **If the cause is irreducible fp ordering, say so plainly.** That is a real
  finding and it changes what token-exactness between our tiers can mean.
- If the eager/graph control of §3.3 fails, stop and report that; no per-layer
  profile is readable until it is resolved.
- `NEEDS_CONTEXT` rather than guessing.

## Outcome

Measured 2026-09-02, `rc` job `5af7552a-5fba-4e51-92d8-f4cdfd3b21ca` (the profile)
and `b2d8024c-273c-4207-8f38-4b23756c133b` (the logits) on `strix:gpu0`, worker
`rc-worker-lcjhd`, one boot, one `libvllm.so`
`3b752d296144dda7b00779f25cfe95f9dfe131fdbf523e5e4e923948783ea7aa`, built from
`007012e49c9b1389725accd07ca30bdc1404e677`. Evidence:
[`qwen38-27b-q4km-tier-divergence-bisect-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-tier-divergence-bisect-20260902.md).

**Two of §7's outcomes are selected, and the pre-registration is what makes that
readable rather than a hedge.**

**P3 for `p4/14`.** It does not reproduce. With both arms on one host our x86-64
CPU tier and our ROCm tier emit all 48 ids identically, `22486` at step 14
included. #2590's `A vs D` column crossed an ARCHITECTURE as well as a tier — the
recorded CPU-tier run was `thor`, aarch64 — so p4/14 is the CPU tier's own
architecture dependence and #2590 narrows from three steps to two.
[#2608](https://github.com/mudler/vllm.cpp/issues/2608) owns it, against
`QUANT-QWEN38-27B-GGUF-ARM`.

**P2 for `p1/45` and `p3/45`.** Both reproduce exactly, ids for ids. The profile
is a smooth ramp with no discontinuity: the final `rel_l2` is 2.63e-02, 2.84e-02
and 2.68e-02 on the three prompts against a bf16 random-walk prediction of
2.55e-02, so **1.03, 1.11 and 1.05 times what two independent bf16 residual
streams MUST differ by**. The run-to-run floor is exactly zero over 9360 compared
per-layer rows, so every part of that is the tier.

**And the amplitude closes it.** The cross-tier pre-sampler logit delta is
`max_abs` 0.33 to 0.39 on every comparable step, while the margins at the two
flips are 0.044262 / 0.123070 and 0.011835 / 0.058807. Over the three prompts,
**17 steps are at risk (their smaller margin is below that step's delta), 2 of
them flipped, and 0 flips occur outside that set.** Being at risk is necessary
and not sufficient — 15 of 17 did not flip — so the account is not a tautology,
and no flip needs a cause beyond the size of the delta.

**Localised anyway, and this is the part the pre-registration did not anticipate.**
P2 was written as "no localisation, it is arithmetic". It is arithmetic AND it
localises: the 16 full-attention layers carry 84% / 74% / 79% of the variance
against the 48 GDN layers, and inside the layer the full-attention mixer
amplifies the incoming relative error by **2.160** where the GDN mixer returns
1.500 — with the SHARED MLP amplifying 1.697 and 1.671, agreeing to 1.6%, which
is what makes the mixer difference a localisation rather than a scaling artefact.
The right reading is that the attention block is the one with the most bf16
stores and two reduction orders over a paged KV cache, not that it is wrong.

**What the row measured that it did not set out to.**

- **The token embedding is bit-identical on every comparable step of all three
  prompts.** §3.4 flagged the `token_embd` residency asymmetry as live — Q4_K
  blocks on the CPU tier, expanded bf16 on ROCm. It contributes nothing. The two
  dequantizers agree to the bit.
- **The two tiers take different GDN conv1d branches on the prefill step**: ROCm
  the indexed `GdnStateGather` arm, the CPU tier the `GatherStateF32` fallback.
  96 keys appear only on ROCm and 48 only on the CPU tier, all on step 0. It is
  not the cause of anything measured, and it is a real difference in which code
  runs. Visible only because the comparator refuses a key-set mismatch.
- **Prompt 4 is the control that makes the size of the effect concrete**: the
  same 2.7% divergence, and identical tokens at all 48 steps. The divergence
  produces a different token only where the model's own margin is smaller than
  it.

**What this means for the row, stated because it is the useful part.**
Token-exactness BETWEEN our tiers is not available at bf16 and never was. It is a
different and weaker claim than token-exactness against an oracle, and this is
the first measurement of it in this project. `TOKEN_GATE` is unchanged, `FAIL`,
and no speed, latency or memory axis becomes admissible.

**Lesson, kept because it cost a mutation.** The first version of this row's
reachability gate keyed on the TOTAL blob count and passed with both production
call sites deleted, because the GDN stage probes fire off the same environment
variable. A count that cannot reach zero when the thing it counts is removed is
not a measurement of it. The counter is now split, and the mutation reds the gate
while leaving the plain binary and the hermetic writer suite green — which is the
difference between measuring a class and measuring a capability, visible in one
run.

## Owed

- [#2608](https://github.com/mudler/vllm.cpp/issues/2608): the aarch64-vs-x86-64
  CPU-tier divergence at p4/14, against `QUANT-QWEN38-27B-GGUF-ARM`. The
  instrument this row built runs on two CPU arms with no GPU on either side.
- [#2610](https://github.com/mudler/vllm.cpp/issues/2610): the GDN prefill conv
  branch asymmetry between the tiers — ROCm takes the indexed
  `GdnStateGather` arm, the CPU tier the `GatherStateF32` fallback, and only
  ROCm emits the `gdn_core` probe. Measured, not explained, and not the cause of
  anything here.
- #2590 stays open on `p1/45` and `p3/45`, now with a cause and no fix: they are
  near-ties decided by an irreducible bf16 accumulation, and closing them would
  mean changing what the residual stream is stored in.
- **What the row does NOT establish, said because the next reader will want it.**
  This measures our two tiers against EACH OTHER. It says nothing about which of
  them is closer to the oracle at a contested step, and it must not be quoted for
  that: the oracle is not in any comparison here, and its own greedy decode is
  not deterministic across its kernel paths. The declared token gate is against
  llama.cpp and is unchanged.
