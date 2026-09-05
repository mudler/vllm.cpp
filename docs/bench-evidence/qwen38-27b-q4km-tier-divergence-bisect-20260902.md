# Where the ROCm and CPU tiers separate on Qwen3.8-27B Q4_K_M, and why

The per-layer hidden-state comparison the 2026-08-23 evidence has owed since it
was written, run for the first time. Issue
[#2590](https://github.com/mudler/vllm.cpp/issues/2590), row `BACKEND-ROCM`,
spec
[`rocm-tier-hidden-state-bisect.md`](../../.agents/specs/rocm-tier-hidden-state-bisect.md).

**Oracle-free.** No llama.cpp was built, no oracle leg was run, and no margin was
recomputed. Every number below is our ROCm tier against our CPU tier, one host,
one lease, one boot, one `libvllm.so`.

**No speed, latency or memory figure appears below.** `AGENTS.md` §Gates admits
none from this arm until its declared token gate passes. It does not pass, and
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) stays blocked.

## Disposition

**Two of the three steps #2590 owns reproduce on one host. The third does not,
and it is not a ROCm-local term at all.** Where the two tiers do differ, the
difference is not a defect in any layer or op: it is the accumulated
disagreement of two independent bf16 residual streams, measured at **1.03 to
1.11 times** what that must be, with **84% / 74% / 79% of it carried by the 16
full-attention layers** rather than the 48 GDN layers, and with the token
embedding **bit-identical on every comparable step of every prompt**.

| step | recorded in #2590 | measured here, ONE host | what it is |
|---|---|---|---|
| p1/45 | ROCm 303, CPU 1521 | **ROCm 303, CPU 1521** | reproduces; a real cross-tier flip |
| p3/45 | ROCm 25, CPU 393 | **ROCm 25, CPU 393** | reproduces; a real cross-tier flip |
| p4/14 | ROCm 22486, CPU 4593 | **both 22486, 48 of 48 ids EQUAL** | does NOT reproduce |

The third row is the correction. #2590's `A vs D` column compared a `gfx1151`
ROCm run on x86-64 against a CPU-tier run on **`thor`, aarch64**, so it crossed
an architecture as well as a tier. Run on one host, our x86-64 CPU tier and our
ROCm tier agree on all 48 tokens of prompt 4. **p4/14 is the CPU tier's own
architecture dependence, not a ROCm term**, and #2590 owns two steps rather than
three.

## What ran

`rc` job `5af7552a-5fba-4e51-92d8-f4cdfd3b21ca` on `strix:gpu0`, worker
`rc-worker-lcjhd`, boot id `a5bc8128-f6ad-4767-8614-6923f88032e1`, 2026-09-02.
Nothing reached the box by `ssh`.

```text
source_revision          = 007012e49c9b1389725accd07ca30bdc1404e677
gguf_sha256              = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
libvllm.so               = 3b752d296144dda7b00779f25cfe95f9dfe131fdbf523e5e4e923948783ea7aa
vllm-bench               = 2a386db06b0df29a378fc67e928615a2856892e998461cbe547c48d7b2296c36
ONE_LIBRARY_BOTH_TIERS   = 3b752d296144dda7b00779f25cfe95f9dfe131fdbf523e5e4e923948783ea7aa
```

**The branch head is not the measured object, and the difference is proven inert
rather than argued.** The two jobs built `007012e49`; the branch carries three
later commits, two of which touch product code — `act_dump.h` and `qwen3_5.cpp`,
for the split stream counter, the two `EndStep` floors and bringing the MoE arm
onto the same writer. None of that can change what the DENSE dump writes, and the
check is executable: `test_qwen27_paged_forward` run with `VT_DUMP_ACT` at both
objects produces **659 files that are byte-for-byte identical**, manifest
included. Every figure below is therefore what the current tree would produce.

**One library serves both tiers, and that is the point.** `vllm-bench` never sets
`EngineParams::device`, so it resolves `Device::kAuto` through the platform
registry, and `RocmPlatform` registers only when `hipGetDeviceCount() > 0`.
Removing the device from the container's view selects `kCPU` from the same
object. "The two tiers" therefore cannot become "two trees" by construction, not
by assertion.

**The tier is measured, not requested.** Each leg asserts which kernels ran, from
`VT_OP_PROVIDER_STATS=1`, and the job fails rather than reports:

```text
rocm_*  device5=20  device0=0   reference_tier_hits=0
cpu_*   device5=0   device0=20  reference_tier_hits=0
```

An environment variable that quietly did not take would have left the CPU leg on
the board and produced two tiers agreeing perfectly, which is the shape of an
instrument pointed at the wrong thing.

**One honest wrinkle in the harness, recorded rather than smoothed.** Neither
job's final `JOB_VERDICT` line reached the share: the log-sync loop is killed by
the `EXIT` trap before the last copy, so both logs end a few lines short. Nothing
here rests on a job's own verdict line. Every assertion this document cites is a
line the log carries explicitly — the op-selection counts, the manifest row
counts, the identity controls, the alignment controls — and each of them is a
`fail()` in the job, so a run that reached the artifacts below cannot have
violated one. All nine profile tarballs and all six logit tarballs are present.

## The controls

| control | result |
|---|---|
| the dumps do not change the ids, ROCm | `IDENTITY_CONTROL_ROCM=PASS`, 48 of 48 identical to the same run with the dump off |
| the dumps do not change the ids, CPU | `IDENTITY_CONTROL_CPU=PASS` |
| the ROCm arm reproduces the recorded gate | `first_diff=NONE` against the v2 gate's recorded ids, on all three prompts |
| the manifests describe the same computation | 15456 keys joined per prompt; the comparator refuses a dtype or shape disagreement |
| **the run-to-run floor** | **EXACTLY ZERO.** The ROCm arm run twice, same lease, same boot: 15552 of 15552 keys joined with no drops and **9360 of 9360 compared rows at `max_abs` 0.0**, the 3120 residual-stream positions included |

The floor is what makes the rest of this document readable. A tier that is
bit-reproducible against itself at every layer of every step has no run-to-run
component, so **every part of the cross-tier profile below is the tier** and none
of it is noise. The v2 gate proved this only at the token level, across six legs;
this proves it per layer.

## The profile

`scripts/tier-hidden-delta.py` joins the two dumps on `(step, layer, stage)` and
reports the delta of `hidden + res`, which is this model's hidden state — the two
halves are carried separately and summed only at the final norm.
`scripts/tier-delta-attribution.py` reads it.

**Steps after the first divergence are excluded.** Once the tiers emit different
tokens they consume different inputs. The exclusion is load-bearing and visible:
at the post-embedding snapshot the excluded steps of prompt 1 reach `max_abs`
0.4492 against **exactly 0** on every comparable step.

| prompt | comparable steps | full-attn share | GDN share | final `rel_l2` (median) | measured / bf16 prediction |
|---|---:|---:|---:|---:|---:|
| p1 | 46 | **84.0%** | 16.0% | 2.6320e-02 | **1.032** |
| p3 | 46 | **74.4%** | 25.6% | 2.8423e-02 | **1.114** |
| p4 | 48 | **78.8%** | 21.2% | 2.6818e-02 | **1.051** |

The prediction is not a fit. Both tiers carry the residual stream in bf16, whose
unit roundoff is `2^-8 = 3.9062e-03`; round-to-nearest gives an RMS relative
error of `eps/sqrt(3) = 2.2553e-03` per store; two independent roundings differ
by `sqrt(2)` times that, `3.1894e-03`; a random walk over 64 layers reaches
**2.5516e-02**. The three measurements are 1.03, 1.11 and 1.05 times it.

Per-layer, the increment of `rel_l2` squared — the quantity that is additive
under independent rounding, and the reason the ratio to the previous layer is not
used to rank anything:

| family | layers | median increment (p1) | per-layer relative step |
|---|---:|---:|---:|
| full attention | 16 | 3.6599e-05 | 6.0497e-03 |
| GDN | 48 | 2.3297e-06 | 1.5263e-03 |

A full-attention layer therefore adds about **1.9 bf16-store-equivalents** of
difference and a GDN layer about **0.5**, so the GDN block CONTRACTS the
difference it is handed while the full-attention block roughly doubles it. Both
are what those blocks are: a full-attention layer rounds Q, K, V, the KV cache
and O to bf16 on top of the two residual stores; the GDN block ends in a gated
RMS norm.

## Inside the layer: which op

`VT_DUMP_ACT_SUB`, prompt 1, both tiers, the same 46 comparable steps. 18432 keys
on each side, joined with **zero drops**. Every figure is a median over the
2208 GDN and 736 full-attention layer-steps.

**The download-artefact control first**, because it decides whether anything
below is a measurement at all. The same post-input-norm buffer is read three
times per layer through three different patterns — pooled `DBuf`, a direct
backend copy, and again after the mixer has run. All three agree to the bit on
all 2944 layer-steps: `max |post_input_norm - DIRECT| = 0`,
`max |post_input_norm - RECHECK| = 0`.

| family | stage | median `rel_l2` | relative-error gain over the previous stage |
|---|---|---:|---:|
| GDN | `post_input_norm` | 2.2596e-02 | — |
| GDN | `block_out` | 3.0406e-02 | **1.500** |
| GDN | `post_attn_norm` | 2.6943e-02 | 0.898 |
| GDN | `mlp_out` | 4.5062e-02 | 1.697 |
| full | `post_input_norm` | 2.1029e-02 | — |
| full | **`block_out`** | 3.8881e-02 | **2.160** |
| full | `post_attn_norm` | 2.7653e-02 | 0.727 |
| full | `mlp_out` | 4.4096e-02 | 1.671 |

**The op is the full-attention mixer.** Both families are handed the same
incoming relative error — 2.10e-02 against 2.26e-02, a 7% difference — and the
full-attention block returns it multiplied by **2.160** where the GDN block
returns it multiplied by 1.500. The two norms contract, as a norm over a growing
residual must. The MLP is the strongest internal check available: it is literally
the same `DenseMlpBlock` code in both families, and it amplifies by **1.697 and
1.671**, agreeing to 1.6%. A stage that differs between the families while the
shared stage does not is a real localisation and not a scaling artefact.

**It is a bigger block, not a broken one.** A full-attention layer rounds Q, K, V,
the bf16 KV cache and O on top of the two residual stores, and the two tiers read
that cache back in different orders; a GDN layer ends in a gated RMS norm. Taken
back to the whole-model total, the per-layer steps are **1.897** store-equivalents
for a full-attention layer and **0.479** for a GDN one, which over 16 and 48
layers is 68.56 store-variances, or **1.0712 stores per layer averaged over the
model**. Its square root, **1.0350**, is the factor by which a one-store-per-layer
prediction should be missed — and the measured `MEASURED / PREDICTED` on this
prompt is **1.032**. The decomposition and the end-to-end number agree to 0.3%,
computed independently from per-layer increments and from the last layer.

## The amplitude, against the thing it has to cross

`rc` job `b2d8024c-273c-4207-8f38-4b23756c133b`, same lease queue, same box, same
build, `VT_DUMP_LOGITS` (#2534's instrument) on both tiers for the same three
prompts. `ALIGNMENT=OK` on every leg: the dumped argmax equals the recorded
output id at 48 of 48 steps, so the two files describe the same contexts.

A per-layer delta only decides a token if it is of the size of the decision it
would have to overturn, so the two are reported together. The **cross-tier
pre-sampler logit delta is stable at `max_abs` 0.33 to 0.39 and `rms` 0.065 to
0.088 on every comparable step of every prompt.** Against it:

| prompt | comparable steps | steps at risk | of those, flipped | flips outside the at-risk set |
|---|---:|---:|---:|---:|
| p1 | 46 | 7 | **1** (step 45) | **0** |
| p3 | 46 | 1 | **1** (step 45) | **0** |
| p4 | 48 | 9 | **0** | **0** |

A step is *at risk* when the smaller of the two tiers' own `top1 - top2` margins
is below that step's cross-tier delta: the tiers disagree by more than the
decision was won by.

**Every cross-tier token disagreement, on all three prompts, is at a step whose
own margin is smaller than the delta. Zero are outside.** At `p1/45` the ROCm
tier's own margin is **0.044262** and the CPU tier's is **0.123070**, against a
delta of 0.348661. At `p3/45` they are **0.011835** and **0.058807** against
0.393238.

**Being at risk is necessary and not sufficient**, which is what stops this from
being a tautology: 15 of the 17 at-risk steps across the three prompts did not
flip, because the delta lands on 248320 logits and only sometimes on the two that
are being compared. Prompt 4 carries nine at-risk steps and flips none.

So nothing about these flips needs a cause beyond the size of the delta. There is
no residue for a defect to explain.

## Three candidates the measurement refutes

- **The token-embedding residency asymmetry.** `token_embd.weight` is Q4_K in
  this file and ROCm registers no `kEmbeddingQuant`, so the vocabulary table
  expands to bf16 on ROCm and stays Q4_K blocks on the CPU tier. It contributes
  **nothing**: the post-embedding snapshot is bit-identical on 46 of 46, 46 of 46
  and 48 of 48 comparable steps. The two dequantizers agree to the bit.
- **The F16 expansion asymmetry.** `DeviceKeepF16Supported` is `dev != kROCM`, so
  an F16 file weight loses mantissa on one tier only. The artifact has **no F16
  tensor**: its 866 tensors are 456 F32, 294 Q4_K, 67 Q6_K, 48 Q5_K and one Q8_0
  (`blk.64.nextn.eh_proj`, the MTP head this arm does not run). Read from the
  file's own tensor table, no GPU needed.
- **A single defective op.** The largest single-layer growth in any profile is at
  **layer 1 or 2**, where `rel_l2` is 1e-3 and a 14x ratio is 1e-2; only 10 of
  3026 layer transitions exceed 3x and **all ten are at layers 0 to 3**. No layer
  anywhere carries an increment out of line with its family, and inside the layer
  the shared MLP amplifies identically on both families.
- **A download artefact in the instrument.** The same buffer read three times
  through three patterns agrees to the bit on all 2944 layer-steps.

## One structural difference, recorded because it is real

The two tiers take different branches of the GDN conv1d **on the prefill step**:
ROCm dumps `gdn_conv` (the indexed `GdnStateGather`/`GdnStateScatter` arm) and
`gdn_core`, the CPU tier dumps `gdn_conv2` (the `GatherStateF32` fallback arm)
and no `gdn_core`. 96 keys appear only on ROCm and 48 only on the CPU tier, all
of them on step 0 and all of them GDN-internal. The residual-stream keys match
exactly, 3120 on each side.

This is visible only because the comparator refuses a key-set mismatch by
default. It is not the cause of anything measured here — the GDN family carries
16 to 26% of a variance that is already at the bf16 floor, and the GDN layers
CONTRACT the difference they are handed — but it is a genuine difference in which
code the two tiers run, and
[#2610](https://github.com/mudler/vllm.cpp/issues/2610) owns it.

## What this means, stated plainly

**The remaining cross-tier divergence is irreducible under a bf16 residual
stream.** Two tiers that reduce in different orders and store the stream in bf16
must disagree by about 2.6% relative at layer 63, and they disagree by 2.6 to
2.8%. The full-attention mixer is where most of it enters, and that is a
localisation rather than a defect: it is the block with the most bf16 stores and
the one whose two implementations reduce a paged KV cache in different orders.
There is no layer to fix and no op to fix.

**The lever that would shrink it is not a lever this gate wants.** An f32
residual (`VT_BF16_RESIDUAL=0`) removes the per-layer store, but it leaves the
bf16 KV cache, the bf16 projections and the two reduction orders, so it cannot
take the tiers to token-exactness with each other — and it was already measured
against the ORACLE on the CPU tier, where it did not move the divergence rate and
made two prompts worse. Nothing here reopens it.

**That size of disagreement puts every step whose margin is under about 0.35 at
risk**, which is the measured `max_abs` logit delta and not a round number
chosen for the sentence. It does not decide them all: 15 of the 17 at-risk steps
across the three prompts kept their token, because the delta lands on 248320
logits and only sometimes on the two being compared. Prompt 4 is the control that
makes this concrete — the same 2.7% hidden-state divergence, nine at-risk steps,
and **identical tokens at all 48**. The contested population is exactly this one:
every step either tier is convicted at has an oracle gap below 0.20.

**So token-exactness BETWEEN our tiers is not available at bf16, and it was never
a property this arm had.** It is a different and weaker claim than token-exactness
against an oracle, and this is the first time the project has measured it.

## What is not admissible from this

- No speed, latency or memory axis. The gate has not passed.
- No claim that either tier is more accurate. They differ by the arithmetic; the
  oracle decides accuracy and it is not in this comparison.
- No claim about the four contested steps that are the oracle disagreeing with
  itself. This document does not touch them.
- p4/14 is **not** evidence about ROCm. It is evidence about the CPU tier across
  architectures, and it needs its own issue.
