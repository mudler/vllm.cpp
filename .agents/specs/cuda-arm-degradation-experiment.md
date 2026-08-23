# W0h — is the CUDA arm WORSE than the CPU arm, or only different?

Issue [#1736](https://github.com/mudler/vllm.cpp/issues/1736). Wave **W0h** of
row `ENG-EXPERT-STREAM-DEVICE` ([engine-matrix.md](../engine-matrix.md), KV
cache and memory), spec
[expert-stream-device-slots.md](expert-stream-device-slots.md), issue
[#1124](https://github.com/mudler/vllm.cpp/issues/1124). This wave settles that
row's **G0-CORRECT** gate. It owns no product code and no capability.

## Now

`SPEC ONLY`. Nothing has run. This document is the pre-registration: the
decision rule below is written before any measurement, and a later commit that
changes a threshold after a run has been taken is a defect in that commit, not a
refinement of this one. Read the rule at the commit that first landed it:

```sh
git log --follow --oneline -- .agents/specs/cuda-arm-degradation-experiment.md
```

**That log has more than one commit before the first run, and here is why, so a
later auditor does not have to guess.** A fresh review of the first draft
returned FAIL, and its repair moved cells in R1, R2 and R3 and added two
preconditions. **Nothing had run, and nothing has run now**, so no threshold was
set against a number: the repairs close holes that all pointed the same way, at
a favourable verdict reached by weakness rather than by evidence. R1 gained an
`UNDERPOWERED` cell so an interval containing 0 cannot become NOT-DISTINGUISHED
on low power alone; R2 gained the `n = 0` cell it left undefined; R3's
SYSTEMATIC cell gained the direction clause it needed; P6 blocks the run on
[#1746](https://github.com/mudler/vllm.cpp/issues/1746), a defect in this
design's own instrument; and P7 probes the corpus's origin, which P3 was
credited with and cannot do. **The audit test is unchanged and is the run date:
any commit to this path dated after the first measurement is the defect the
paragraph above names.**

## Why this is a wave of `ENG-EXPERT-STREAM-DEVICE` and not a new row

The call, and the argument for it, because either shape is arguable.

**The gap already has an owner.** Two entries in
[expert-stream-device-slots.md](expert-stream-device-slots.md) `## Owed` name
exactly this question: "A ratified gate for a two-arm comparison whose greedy
path is a coin flip", and "WHAT the divergence IS. It is expert ROUTING and not
sampling, and its CAUSE is still unnamed". `AGENTS.md` says to reconcile the
record when work already has an owner, rather than to implement beside it. A
second row would give one obligation two owners, which is the shape #777 (two
GPU mutexes) and #1731 (one issue indexed twice) each cost this project a
measurement or a red gate.

**It decides an existing declared gate.** G0-CORRECT belongs to
`ENG-EXPERT-STREAM-DEVICE`. A verdict on that gate that lived in another row's
record would be a measurement of one row stored inside another, which
`AGENTS.md` §Records names as the coupling to avoid.

**It reuses that row's whole apparatus** and adds no seam of its own: the same
checkpoint, the same `dgx:gpu0` lease, the same harness lineage
(`benchmarks/expert_stream_device_w0e.cpp`), the same two arms. It produces no
capability, so it owes no `docs/FEATURES.md` row, no matrix row and no ABI
surface.

**Against the call, stated rather than hidden.** The experiment has its own
oracle, its own instrument and its own pre-registered rule, and the method
generalizes to at least one other open two-arm divergence (`A2-Q1` on Nemotron,
[#1388](https://github.com/mudler/vllm.cpp/issues/1388)). Generalizing it is
scope creep here; see `## Out of scope`.

**Why the design lives in its own file rather than inside the row spec.** Two
reasons, both about pre-registration. A rule that decides a verdict must be
auditable by `git log --follow` on one path, and
[expert-stream-device-slots.md](expert-stream-device-slots.md) is 1,100 lines
that several lanes append to, where an edit to a threshold would be invisible in
a diff of a much larger change. And this file has one job, so a reader can
confirm in one screen that no number moved after a run. The tree has precedent
for a spec with no matrix row of its own:
[clock-cross-arm-mean.md](clock-cross-arm-mean.md),
[gate-27b-fp8-tower-golden.md](gate-27b-fp8-tower-golden.md) and
[nemotron-oracle-golden-provenance.md](nemotron-oracle-golden-provenance.md) are
three. `ENG-EXPERT-STREAM-DEVICE` still owns the wave; this file is its design,
and the row spec links it.

## Scope

**In scope.** Measure whether the `--device cuda` arm assigns systematically
worse likelihood than the `--device cpu` arm to identical held-out text on
`Qwen3.8-2.4T-A95B UD-Q1_0`, and report one verdict against a rule written
first.

**Out of scope**, each with its reason, in `## Out of scope`.

## What is established, and what is not

Every number below is in [`../benchmark-record.md`](../benchmark-record.md)
under `ENG-EXPERT-STREAM-DEVICE W0g`. This section does not restate them; it
states what they do and do not license.

**Established.** The arms load the same weights at both ends of the stack: the
router gate fingerprint agrees on all 184 dump records, and the embedding output
is bit-identical over all 40,960 bf16 values. Both top-k implementations rank
their own logits correctly on 5 of the 552 token-rows each dump holds, which is
0.91 % and is a sample, not a property of the implementations. The first visible
flip is an exact bf16 tie: experts 205 and 212 both read -4.937500 on CPU, and
212 reads -4.906250 on CUDA. The hidden-state divergence ramps from 0.56 % at
MoE block 0 to 13.4 % at block 91. The arms already select different experts in
the first MoE block of the first forward.

**Not established: whether the CUDA arm is worse.** Three grounds.

### 1. The CUDA continuation degenerates

The arms agree for 8 tokens, then the CUDA text falls into a mechanical
recursion in which each sentence re-uses the previous object (` Paris. Paris is
a city located in France. France is a country located in Europe. Europe is a
continent located on Earth...`), against a CPU continuation that keeps adding
new content (`...in the northern part of France, on the Seine River...`). A coin
flip between two equally good tokens does not produce that.

**One continuation is one sample.** That is exactly why the design below uses a
fixed multi-prompt corpus: a single degenerate continuation can be luck, and a
consistent pattern cannot.

### 2. Every comparison so far is arm-against-arm, with no oracle

Neither arm is ground truth, so a measured difference between them structurally
cannot say which one is wrong. `AGENTS.md` makes vLLM the reference and not our
CPU arm. The engine-matrix row currently reads "the correctness reference is our
own CPU arm and the gate is token-exactness against it", which is a statement
about what is available and not about what is true. vLLM cannot run this
checkpoint, so the row's reference is the best reachable one and still not an
oracle.

### 3. The growth-rate argument is weaker than it was stated to be

The argument offered for "partly systematic" was: 0.56 % to 13.4 % is 23.9x over
91 blocks; uncorrelated rounding accumulates as `sqrt(91) = 9.54`; a systematic
error accumulates about linearly at 91x; so 23.9x sits above the noise
prediction.

**A two-point reading is not the test, and the full table does not support the
conclusion.** A least-squares fit of `log(divergence)` on `log(block + 1)` over
all eight recorded points gives an exponent of **0.651**, standard error
**0.066**, and a 95 % interval of **[0.489, 0.813]** (`t(0.975, 6) = 2.447`).
That interval **includes** the 0.5 a random walk predicts and **excludes** the
1.0 a systematic error predicts.

```sh
python3 -c "
import math
pts=[(0,0.56),(1,1.63),(2,1.86),(5,4.20),(10,4.38),(20,5.72),(45,11.0),(91,13.4)]
xs=[math.log(b+1) for b,_ in pts]; ys=[math.log(v) for _,v in pts]
n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
p=sxy/sxx; a=my-p*mx
s2=sum((y-(a+p*x))**2 for x,y in zip(xs,ys))/(n-2)
print(p, math.sqrt(s2/sxx))"
```

Two further limits travel with that table. Its eight points come from **one
prompt and one prefill**. And its statistic is a mean of per-element ratios,
which [`../benchmark-record.md`](../benchmark-record.md) already warns is not
comparable with the ratio of means it reports twelve lines above.

So the ramp is carried below as **R3, a pre-registered test to run with more
data**, and not as evidence for a defect. Grounds 1 and 2 carry this experiment
on their own.

## Upstream anchors

**There is no upstream implementation of this path, and that is the anchor.**
Inference-time disk expert paging is absent in pinned vLLM
(`vllm/model_executor/offloader/uva.py:21`, CPU-blanket UVA over whole
parameters; `vllm/model_executor/offloader/prefetch.py:557-560`, cpu-only), as
the engine-matrix row already records. vLLM cannot load
`Qwen3.8-2.4T-A95B UD-Q1_0` on a 119.631 GiB box, so the primary oracle cannot
run this workload and cannot be the reference here.

**What IS mirrored, and where.** The instrument is vLLM's logits-processor
stage, which this tree already ports: `include/vllm/logits_processor_callback.h`
carries the C++ analogue of `vllm/sampling_params.py`'s
`SamplingParams.logits_processors`, applied at vLLM's non-argmax-invariant
stage, after `allowed_token_ids` / `bad_words` / `min_tokens` / `logit_bias` and
before penalties. The upstream anchor for that ordering is
`vllm/v1/sample/sampler.py:399`, **as recorded by the header itself rather than
re-read at the pin for this spec**; the implementer re-verifies it, because a
line anchor goes stale and an anchor read in the wrong tree still lands on
plausible unrelated code. Our side is `apply_logits_processors`, called once
from `src/vllm/v1/sample/sampler.cpp`, which `git grep -n
'apply_logits_processors(' -- src` shows is the single call site. Teacher forcing
is a mask applied at that stage. The experiment therefore adds no seam; it uses
one.

**The secondary oracle is llama.cpp**, for the reason `AGENTS.md` §`When vLLM has
no implementation` gives: it is the only tree that defines the encoding this
checkpoint stores its experts in. See `## The oracle arm`.

## Design

### The instrument: the ABI logits processor, not a new seam

`vllm_logits_processor` (`include/vllm.h`, ABI v8) is a per-request host
callback invoked once per decode step, before sampling, with the request's
generated token ids so far and a **mutable** f32 view of that request's logits
row. `src/vllm/v1/sample/logits_processor/builtin.cpp`
`apply_logits_processors` synchronizes the backend, obtains what it INTENDS to be
a host-addressable view of the `[num_reqs, vocab]` logits (the pointer itself
where it judges the backend unified, a staged copy otherwise), calls each
registered processor, and copies back where it staged.

**On the CUDA arm that judgement is currently WRONG, and it is a prerequisite of
this experiment rather than a detail of it.** See the paragraph on the W0e fault
below, and **P6**.

That gives the experiment three properties it would otherwise have to build:

1. **It is a production entry point.** The harness is an ABI client and includes
   no internal header, which is what `AGENTS.md` §`Shared seams` requires of
   examples and servers.
2. **The host/device staging is the seam's job and not the harness's**, so one
   callback serves both arms and the experiment adds no arm-specific code.
   **This is the property #1746 currently breaks**: on GB10 the seam takes the
   unified branch and hands the callback a raw `cudaMalloc` pointer, so today
   the two arms do NOT receive the same kind of pointer, and the paragraph below
   reads out why.
3. **It needs no scratch seam of its own.** The scratch instrument that read
   `logits` in the **completion** callback SIGSEGVs on the CUDA arm
   (`SCRIPT_EXIT=139`, carried under the row's `## Owed`).

**What this instrument is NOT: host-addressable by construction.** An earlier
draft of this file said it was, and that claim is FALSE on the target hardware.
The chain, read rather than assumed:

* `apply_logits_processors` gates on `b.UnifiedMemory()`
  (`src/vllm/v1/sample/logits_processor/builtin.cpp:93`) and then takes the
  device pointer straight as a host pointer, `host =
  static_cast<float*>(logits.data)` (`:98`).
* On GB10 that predicate is TRUE: `CudaBackend` is constructed with
  `caps.pageable_memory_access && caps.integrated`
  (`src/vt/cuda/cuda_backend.cu:363`), which holds on an integrated part.
* `CudaBackend::Alloc` is nevertheless a plain `cudaMalloc`
  (`src/vt/cuda/cuda_backend.cu:80-82`), and CUDA never overrides
  `DeviceMemoryIsHostAddressable()`, so that narrower predicate is the base
  `false` at `include/vt/backend.h:77`. ROCm, Metal and Vulkan each override it;
  CUDA alone does not.

So the seam hands a `cudaMalloc` pointer to a host dereference on the arm this
experiment is about. `src/vt/op_provider.cpp:866-873` documents exactly this
class beside the narrow predicate, recording that asking `UnifiedMemory()` where
the question is host-dereferenceability "COST TWO CRASHES (#844, #1435)", plus
#960. **It is filed as
[#1746](https://github.com/mudler/vllm.cpp/issues/1746) and is not fixed here:**
this wave lands no product code, and a separate implementer owns that repair.

**#1746 is therefore a PREREQUISITE for running W0h, not a follow-up.** Until it
lands, the instrument this design is built on faults on the CUDA arm, and a run
attempted before it either crashes or — worse — reads whatever the host maps at
that address. **P6** gates the run on it, and **P1** still prints
`cudaPointerGetAttributes` on the row pointer the callback receives, which is the
reading the row's `## Owed` says the W0e fault never got.

### Teacher forcing

For each corpus item, the prompt ids are the request's prompt and the
continuation ids are forced one at a time. At decode step `t` the processor:

1. reads the **unmodified** logits row and records
   `NLL_t = -log softmax(logits)[target_t]`, accumulated in `double`;
2. records the full logit row, or the top-`K` of it, for the cross-arm delta;
3. sets every logit except `target_t` to a large finite negative value, so
   greedy sampling emits `target_t`.

**The mask uses `-1e30` and not `-inf`**, so that no later stage can produce a
NaN from `inf` minus `inf`. Sampling runs greedy with every penalty at its
neutral default, because the processor stage runs BEFORE penalties and a
non-neutral penalty would edit the masked row after the mask.

Both arms therefore walk the **identical** token sequence, and their KV caches
are built from the identical sequence. Any per-position difference is pure
numerics, with no trajectory drift mixed in. That is the confound every earlier
comparison carried.

**The forced continuation is natural text from the corpus, never either arm's
own generation.** Forcing an arm onto its own greedy output hands that arm the
maximum-probability token at every position by construction, and the NLL
comparison would then measure which arm produced the text. This is the single
easiest way to get a confident wrong answer out of this design, and it is named
here so a reviewer can check for it.

### NLL is the quality metric, and it needs no oracle

Two implementations of the same weights that are equally faithful assign nearly
the same likelihood to the same held-out text. A materially higher CUDA NLL on
text neither arm wrote means the CUDA arm models that text worse. That is a
quality statement, not a difference statement, and it is available with no third
implementation. What it cannot do is say whether **both** arms are wrong
together; only the oracle arm can, which is why the oracle arm exists.

### Bias versus noise

Rounding is zero-mean. A defect is not. Per position, the paired logit delta
`d = logits_cuda - logits_cpu` restricted to the CPU arm's top-`K` tokens is
tested for a nonzero mean by a sign test (**R2**), and the same delta is
reported as a function of block depth from a block-wise hidden-state dump
(**R3**). Reporting the magnitude alone would not separate the two.

### The corpus

**16 prompts plus 1 labelled anchor. 20 forced positions each.**

| Item | Content | Counts in R1 |
|---|---|---|
| `anchor` | ids `760,6511,314,9338,369` ("The capital of France is"), the W0g prompt | **No** |
| `p01`..`p16` | held-out natural text, 16 to 40 prompt ids, 20 continuation ids | Yes |

**Why the anchor is excluded from the primary statistic.** It was chosen because
it degenerated. Including a prompt selected on its outcome biases the corpus
toward the outcome. It is run and reported separately, because reproducing the
known divergence is the check that the harness reaches the same behavior W0g
saw.

**How the 16 are chosen, and how the choice is pinned.** The selection rule is
fixed before the run and does not look at either arm: take a public English text
at a pinned revision, walk it in order, and accept a passage when its first
sentence tokenizes to 16 to 40 ids, and the next 20 ids are all in vocabulary
and hold no end-of-sequence token; stop at 16 accepted passages.
The **resulting id arrays are then written verbatim** into
`benchmarks/w0h_corpus.json`, together with the source, its revision, the
tokenizer's checkpoint revision, and a sha256 of the file. The run reads the
file and drives the engine through `vllm_complete_tokens` (`include/vllm.h`),
which takes prompt ids and skips tokenization. Nothing re-tokenizes at run time,
so a tokenizer change cannot silently move the corpus, and the same id arrays
drive the oracle arm.

**Why 16 and 20.** The bootstrap unit is the prompt, so `P` sets the statistical
power and `L` only reduces per-prompt variance. The budget is a lease: W0g
measured a steady decode median of 9.09 s/token on CPU and 4.72 s/token on CUDA
and a load of about 265 s per arm, so `2 x 17 x 20 = 680` forced positions cost
roughly 78 minutes of decode plus 34 prefills and two loads. That fits one lease
with headroom for the probes. **If the lease permits less, cut `L` and never
`P`, and never below `P = 12`**, because a bootstrap over 12 units is already
thin and a bootstrap over fewer is not a measurement.

### The oracle arm

`llama-cpp-unsloth` ([`../oracles/llama-cpp-unsloth.md`](../oracles/llama-cpp-unsloth.md)),
pin `36fe8e1cc7f2b3b8c92fdda0ab07600141921786`, branch `iq1-narrow`, pinned
2026-08-15. It is the only tree that defines `GGML_TYPE_IQ1_XXXS = 66`, the type
this checkpoint stores about 97 % of its parameters in.

**Its record reads `gateable = no`**, owed by
[#933](https://github.com/mudler/vllm.cpp/issues/933). `AGENTS.md` admits an
oracle as gateable only after it demonstrably builds and runs the model, and
this fork has done neither here: the encoding was ported from its source, which
is weaker than a running comparison. Its own file records that running it needs
the full 370 GiB checkpoint and, per Unsloth's documentation, at least 450 GB of
RAM, against a 119.631 GiB box.

**`llama-cpp` became gateable on 2026-08-23 and that does NOT make this arm
gateable, which is the reading a later reader is most likely to get wrong.**
[#1740](https://github.com/mudler/vllm.cpp/pull/1740) flipped the STOCK oracle
`llama-cpp` at `10bf611e5` (`b10451`) to `gateable = yes`, demonstrated on
`Qwen3.8-27B-Q4_K_M.gguf`, a 17 GB Q4_K_M artifact. Two things keep it out of
this experiment. That pin's highest ggml type is `GGML_TYPE_Q2_0 = 42`, so it
**cannot read type 66** and cannot open this checkpoint at all. And gateability
is recorded per oracle: the fork's own pin block still reads `gateable = no`
with `evidence = #933`, unchanged by #1740. Verify both before relying on
either:

```sh
sed -n '/```oracle-pin/,/```/p' .agents/oracles/llama-cpp-unsloth.md
sed -n '/```oracle-pin/,/```/p' .agents/oracles/llama-cpp.md
```

**Why it is still worth a bounded probe.** llama.cpp mmaps the GGUF by default,
so its pages are file-backed and evictable, and the checkpoint is on local NVMe
at `/home/mudler/ckpt/qwen3.8-q1_0`. A 450 GB figure is a comfort requirement,
not a demonstrated floor, and the box has 119.631 GiB of RAM plus 30,625 MiB of
swap. **That is a hypothesis and is labelled one.** It is tested by **P2**
before any lease time is committed to the arm.

**What it would give.** `llama-perplexity` computes exactly the statistic this
experiment defines, over the same GGUF and the same encoding. Driven with the
pinned id arrays rather than with text, it gives a third NLL per position, which
is the only thing in this design that can (a) supply the materiality anchor R4
needs and (b) detect a defect that **both** our arms share.

**What its absence costs, stated so nobody reads the experiment as complete
without it.** R1 still returns a directional verdict, because a paired
comparison of our two arms needs no third party. R4 has no anchor, so the
experiment can say "CUDA is systematically worse" and cannot say "by an amount
that matters". And a shared defect stays invisible: if both arms are wrong in
the same direction, R1 reads NOT-DISTINGUISHED and the experiment reports a tie
between two wrong arms. That last one is the reason the oracle arm is not
optional in principle, only in practice.

**Two obligations if the arm runs.** Assert the tree and not only the commit, as
[`../oracles/llama-cpp.md`](../oracles/llama-cpp.md) requires: build from a
fresh clone or `git archive` of the pinned SHA, or assert `git status
--porcelain` empty, and record the binary's sha256 either way. And re-verify the
anchors, because a fork **branch** can be rebased under a name and the oracle
file says so. A successful run also discharges part of #933, which is recorded
there and not claimed here.

### The in-tree oracle switch is excluded, with the arithmetic

`VT_CPU_REF=1` forces **every** tensor to `kExpandBf16`:

```cpp
// src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:156-157
// The oracle switch wins over everything (spec gate 2).
if (cpu_ref) return GgufResidency::kExpandBf16;
```

Two independent derivations, both re-run for this spec rather than copied:

| Derivation | Result |
|---|---|
| whole file at the expert bpw: `369.97 GiB x (16 / 1.1875) = 369.97 x 13.4737` | **4,984.9 GiB = 4.87 TiB**, 41.7x the box |
| by parameter: `2.4T x 0.97 x 2 B` experts + `2.4T x 0.03 x 2 B` rest | 4,336.2 + 134.1 = **4,470.3 GiB = 4.37 TiB**, 37.4x the box |

The first slightly overstates, because the roughly 48.1 GiB the checkpoint holds
outside the expert towers is not stored at 1.1875 bpw. Both land in the same
place: **multiple tebibytes against 119.631 GiB**. The switch is not loadable
for this checkpoint and is not a candidate. Recorded here so nobody proposes it
again.

## Constraints

**Every sampler stage upstream of the instrument is pinned to neutral, not
only the penalties.** `## Design` says NLL is read from the **unmodified** logits
row, and that is only true if nothing edits the row before the callback runs.
`src/vllm/v1/sample/sampler.cpp:425-440` runs four stages BEFORE
`apply_logits_processors` at `:441`, in upstream's order
(`allowed_token_ids` -> `bad_words` -> `min_tokens` -> `logit_bias`), and the
penalties run after. So the pin has two halves:

| Stage | Where | Pinned to | Why it is not free |
|---|---|---|---|
| `allowed_token_ids` | `sampler.cpp:426-428` | absent (`allowed_token_ids_mask` has no value) | it writes `-inf` into every id outside the allow set, which would make an NLL of a masked target read `inf` |
| `bad_words` | `sampler.cpp:429-431` | empty | same, per matched suffix |
| `min_tokens` | `sampler.cpp:434` | empty | it writes `-inf` on the stop ids; the corpus excludes an end-of-sequence target, so a non-empty map would only bias, not fault |
| `logit_bias` | `sampler.cpp:435` | empty | it ADDS a per-id constant, which shifts the softmax denominator and therefore every NLL at that position |
| penalties (repetition, frequency, presence) | `sampler.cpp:443-448` | `no_penalties` true | already stated under `## Design`: they run AFTER the mask and would edit the masked row |

Two of the four are genuine no-ops at their defaults and this is stated so the
pin is read as a check rather than as a change: `apply_min_tokens`
(`builtin.cpp:18`) and `apply_logit_bias` (`builtin.cpp:46`) each early-return
on an empty input, and the other two are guarded at the call site by
`has_value()` and `empty()`. **A default is not a probe**, so **P1** asserts all
five readings rather than trusting the harness to have left them alone.

**Clock state is captured, and cannot be pinned by this row.**
`tools/bench/gpu_clock_state.py` exists (`BENCH-ASSERT-CLOCK-STATE`,
[#543](https://github.com/mudler/vllm.cpp/issues/543)) and went unused on the
W0g run, which is recorded as owed. Every run under this spec opens a clock
window per arm through that helper's own `sample` CLI, and records the SM clock
across the measured window, `clocks.max.sm`, `clocks.applications.graphics`, the
active throttle reasons, persistence mode and the **boot id**. W0g ran under an
application clock pin of 2418 MHz against a 3003 MHz maximum, discovered after
the fact.

**Pinning is not available to this row.** Inside an `rc` lease `nvidia-smi -lgc`
returns `LGC_RC=4` even as root, because the worker's `CapBnd` holds no
`CAP_SYS_ADMIN` ([`lease-gpu-capability.md`](lease-gpu-capability.md), #1354).
The host path does work, and [`../environment.md`](../environment.md) records
that `.agents/developer-preferences.md` scopes `rc hold` plus `ssh` to the
`BENCH-QWEN38-27B-SOTA` campaign. **That preferences file is untracked and is
absent from this checkout, so this spec does not read an authority out of a
document.** The authority for this row is unresolved and its gate stays
`PENDING` until the developer answers. Nothing in this design needs it.

**A clock excursion cannot bias this result, and the record is still required.**
NLL is a function of the weights and the inputs, not of the clock rate. The
window is recorded because the constraint requires it, because provenance makes
the run reproducible, and because a window that turns out unusable must be
visible rather than absent. A run whose clock window fails still yields its NLL
verdict, and says so.

**Lease discipline: `rc run`, not `rc hold`.** `rc run` claims the device, runs
the command on the host, streams the output, **releases the device**, and exits
with the command's status. The release is therefore in the tool and not in a
plan, which is the property asked for: a hold on this row once sat idle for 76
minutes because its release lived in a plan. `rc hold` gives a lease and no
execution, so it is reached for only if a host shell is genuinely needed, and
then it carries `--reason` (which `rc hold` requires) and a short `--ttl`.

**No host page-cache drop is needed.** W0g Run A dropped it and measured the
effect at about 7 % of decode wall time. This experiment publishes no time, so
the cache state does not enter any result. Dropping it would require the host
shell this row cannot take.

**Weights from local NVMe.** `/home/mudler/ckpt/qwen3.8-q1_0`, not the CIFS
share. Serving this lane over SMB measures the NAS.

**The per-token slice figures this spec uses, corrected and re-derived here.**
The engine-matrix row says "2790 slices per token at 2,490,368 B is 6.95 GB per
token". **The slice count is 2790 only if the model has 93 MoE blocks, and the
W0g dump says it has 92.** That dump is self-verifying: `8 + 184*44 +
sum(T)*17488 = 9,661,480` gives `sum(T) = 552`, `552 = 92*5 + 92*1` fixes 92 MoE
blocks over one 5-token prefill and one decode step, and `17488` fixes the
top-10 of 512 at `H = 8192`. Three projections per expert therefore give
`92 x 10 x 3 = 2760` slices per token.

| Quantity | Value | Derivation |
|---|---|---|
| slices demanded per token | **2760** | `92 x 10 x 3` |
| bytes demanded per token | **6.873 GB** | `2760 x 2,490,368` |
| decode hit rate | **43.4 %** | measured |
| bytes actually READ per token | **3.89 GB** | `2760 x (1 - 0.434) x 2,490,368` |

The read figure is the one that sizes the storage path, and it is 56 % of the
demand rather than all of it. Reproduce both:

```sh
python3 -c "s=2490368; n=92*10*3; print(n, n*s/1e9, n*(1-0.434)*s/1e9)"
```

**Two premises this spec deliberately does NOT rest on.** It does not assume the
lane is I/O bound: on the repository's own QD1 curve at 2.76 GB/s single-
threaded the read term is `3.890 / 2.76 = 1.409 s` of a 9.055 s CPU step, which
is `1.409 / 9.055 = 15.6 %`, so **read-issue order can move at most that whole
1.41 s**, because the best a perfect ordering can do is remove the term
entirely. An earlier draft put the bound at "about 0.64 s"; that number followed
from nothing this file states, and a bound whose derivation is absent is not a
bound, so it is withdrawn rather than re-justified. No tighter figure is claimed
here, because none is derivable without a measurement this wave does not take. And it says nothing about
routing skew or static prefill-chosen expert pinning, which is a separate and
probably closed question: FreeToken (arXiv 2608.16157, Fig. 4b) replays real
routing traces and reports prefill-chosen static pinning missing 59 % against
demand-driven LRU's 39 % at equal capacity, and a 32-expert pin over 92 blocks
and 3 projections is 20.48 GiB, larger than the 18.55 GiB configuration already
measured 3.6x to 4.1x slower. **Both are noted so that nobody reads their
absence as an oversight, and neither enters any result here**, because this wave
measures likelihood and publishes no time.

**Build inside `vllmcpp-build:gb10`.** The host has no `nvcc`. `/usr/bin/time`
does not exist in that image, so no script may call it. Build with `-j 4`;
unconstrained parallelism has OOM-rebooted this box.

**Only `dgx:gpu0` can hold this model.** It is a fleet device, so `rc` is the
only path to it and `ssh` is not.

**This experiment produces no speed claim. `G0-SPEED` stays VOID** under every
outcome of the rule below.

## The pre-registered decision rule

**This section is the binding copy. It is written before any measurement.** A
commit that changes a threshold here after a run has been taken is a defect in
that commit. Notation: `NLL_A(i)` is the negative log likelihood arm `A` assigns
to the corpus target at position `i`; positions are the `P x L` teacher-forced
positions of the 16 corpus prompts; the anchor prompt is excluded.

### R1 — primary, binding, directional

`delta = mean over all positions of (NLL_cuda(i) - NLL_cpu(i))`.

Interval: a paired bootstrap, **10,000 resamples**, resampling **PROMPTS** and
not positions, because positions inside a prompt share a context and are not
independent. Report the two-sided 95 % percentile interval.

Consistency: `C` = the number of the 16 prompts whose per-prompt mean NLL is
higher on CUDA than on CPU.

`M` is the equivalence margin, defined below.

| Verdict | Condition |
|---|---|
| **DEGRADED** | the interval excludes 0, and `delta > 0`, and `C >= 12` |
| **CPU-DEGRADED** | the interval excludes 0, and `delta < 0`, and `C <= 4` |
| **INCONSISTENT** | the interval excludes 0 and the matching `C` condition fails |
| **NOT-DISTINGUISHED** | the interval includes 0 **and** lies entirely inside `[-M, +M]` |
| **UNDERPOWERED** | the interval includes 0 and is not contained in `[-M, +M]`, which includes every case where `M` is unavailable |

**Why `C >= 12`, with the arithmetic rather than a round fraction.** A mean can
be carried by one prompt, so a directional claim about the arms must hold prompt
by prompt as well as in aggregate. Under the null that each prompt is equally
likely to fall either way, `C` is `Binomial(16, 0.5)`, and
`P(C >= 12) = 2517 / 65536 = 0.0384`. So the consistency clause is a one-sided
sign test at about the 3.8 % level, which is the nearest cut to a conventional
5 % that the discrete distribution offers: `C >= 11` gives 10.5 % and is too
loose, and `C >= 13` gives 1.1 % and would refuse a real effect on 16 prompts.

```sh
python3 -c "from math import comb; print(sum(comb(16,k) for k in range(12,17))/2**16)"
```

`INCONSISTENT` is a real outcome and is reported as itself. It is never rounded
to either side.

**Why an interval that merely CONTAINS 0 is not a result, and why the margin is
`M = S` rather than a number.** An interval containing 0 is produced by a real
tie and by a run with too little resolution to tell, and those are not the same
finding. Only one of them may open the distributional-gate door that R5's
NOT-DISTINGUISHED row opens, so the rule is stated as an equivalence test: the
whole interval must lie inside a stated margin, not merely straddle 0. A
two-sided 95 % interval contained in `[-M, +M]` is TOST at 2.5 % per side, which
is conservative against the conventional 5 % TOST, and it is used because R1
already computes that interval.

The margin has to answer "smaller than WHAT", and the only measured answer this
design contains is R4's `S = |mean NLL_cpu - mean NLL_llamacpp|`: the distance
between two implementations of the same weights that are each faithful. A
CUDA-against-CPU difference inside `S` is not separable from the ordinary spread
among faithful implementations of this checkpoint. So `M = S`.

**No principled margin is derivable without the oracle arm, and this file
invents none.** R4 is `PENDING` on
[#933](https://github.com/mudler/vllm.cpp/issues/933), so as this spec lands `M`
is unavailable and **NOT-DISTINGUISHED is unreachable**: an R1 interval that
includes 0 reads `UNDERPOWERED`, which R5 composes to `UNDETERMINED`. That is
the intended polarity, stated plainly rather than discovered later. The
alternative — a margin picked so that a reachable verdict exists — is exactly
the after-the-fact threshold this section exists to prevent.

**What the design can resolve, so "underpowered" is a number and not a word.**
Let `sd` be the between-prompt standard deviation of the per-prompt mean NLL
difference. The run measures it; this file cannot know it in advance, which is
why the MDE is pre-registered as a multiple of `sd` and not as a nat count. On
the Gaussian approximation to the paired bootstrap the 95 % half-width is about
`t(0.975, P-1) x sd / sqrt(P)`:

| `P` | `t(0.975, P-1)` | half-width | reading |
|---|---|---|---|
| **16**, as designed | 2.13145 | `2.13145 / 4 = 0.533 x sd` | R1 excludes 0 for a true `abs(delta)` above `0.533 x sd` |
| **12**, the floor | 2.20099 | `2.20099 / sqrt(12) = 0.635 x sd` | the floor costs `0.635 / 0.533 - 1 = 19 %` of the resolution |

`0.533 x sd` is therefore the pre-registered minimum detectable effect, and
`M > 0.533 x sd` is the condition under which NOT-DISTINGUISHED is reachable at
all rather than arithmetically impossible. **A bootstrap percentile interval is
not a `t` interval**, so the table is the advance approximation used to state
the MDE; the run reports the bootstrap's own realized half-width beside it, and
the verdict table above is evaluated against the realized interval and never
against this approximation.


### R2 — bias versus noise

For each position, take `d_j = logits_cuda[j] - logits_cpu[j]` over `j` in the
CPU arm's top-`K` tokens, `K = 64`. Count `n_+ = #{j : d_j > 0}` and apply a
two-sided binomial sign test against `p = 0.5` at `alpha = 0.01`. Let `R` be the
fraction of positions that reject.

| Verdict | Condition |
|---|---|
| **SYSTEMATIC** | `R > 0.05` |
| **NOT-SYSTEMATIC** | `R <= 0.05` |
| **UNDETERMINED** | `N = 0`: every scored position was fully tied, so `R` has no denominator |

**The 0.05 threshold, with the correlation quantified instead of asserted.** The
null predicts `R = 0.01`. Where 0.05 sits relative to that depends entirely on
how independent the positions are, and the two ends of that range are both
computable:

| Assumption | `N_eff` | `SE = sqrt(0.01 x 0.99 / N_eff)` | `(0.05 - 0.01) / SE` |
|---|---|---|---|
| positions independent | 320 | 0.00556 | **7.19** |
| positions inside a prompt perfectly correlated, so each prompt contributes one | 16 | 0.02487 | **1.61** |

```sh
python3 -c "
import math
for ne in (320, 16):
    se = math.sqrt(0.01*0.99/ne); print(ne, round(se,5), round(0.04/se,2))"
```

With the design effect `deff = 1 + (L - 1) x rho` at `L = 20`, the second row is
`rho = 1`, `deff = 20`, `N_eff = 320 / 20 = 16`. **So the bar is between 1.6 and
7.2 standard errors above the null, and which it is depends on `rho`, which is
unmeasured.** That is an assumption and is labelled one; it is not a
justification for the number, which stays where it was pre-registered.

**Why this matters in one direction rather than both.** R2 gates only the
favourable outcome — R5 reaches NOT-DISTINGUISHED only when R2 reads
NOT-SYSTEMATIC — so conservatism here makes the favourable verdict EASIER. The
threshold does not move, because moving it after the fact is the defect this
section prevents. What the run owes instead is the measurement that would
falsify the assumption: **report the intraclass correlation `rho` of the
per-position reject indicator across prompts, the implied `deff = 1 + 19 x rho`,
`N_eff = 320 / deff`, the implied `SE`, and `(0.05 - 0.01) / SE` beside `R`.**
If that last figure exceeds the conventional 2 — which happens at
`N_eff > 0.0099 / 0.02^2 = 24.75`, so `deff < 320 / 24.75 = 12.93`, so
`rho < 11.93 / 19 = 0.628` — then the threshold was conservative toward
NOT-SYSTEMATIC by the amount reported, and a NOT-SYSTEMATIC reading carries that
number with it. A published conservatism is auditable; an asserted one is not.

**Ties, and the `n = 0` cell the earlier draft left undefined.** Ties
(`d_j == 0`) are excluded from `n_+` and from `K`, and the excluded count is
reported, because a large tie count is itself a finding about the two kernels.
A position at which ALL `K` deltas tie leaves `n = 0`, and a sign test on zero
observations has no verdict. **This is a live cell rather than a formality: W0g
measured an exact bf16 tie in this very system** (experts 205 and 212 both at
-4.937500 on CPU). Such a position is excluded from BOTH the numerator and the
denominator of `R` and counted as `N_tie`, which is reported. Exclusion is the
neutral choice: scoring it as non-rejecting would deflate `R` toward
NOT-SYSTEMATIC, which is the favourable verdict, and scoring it as rejecting
would inflate `R` toward SYSTEMATIC. If the exclusions leave `N = 0`, R2 reads
**UNDETERMINED** and R5 composes that to `UNDETERMINED`, never to
NOT-DISTINGUISHED.

### R3 — depth exponent

From a block-wise hidden-state dump at a fixed set of teacher-forced positions,
fit `log(divergence)` on `log(block + 1)` by least squares and take the 95 %
interval on the slope `p`.

| Verdict | Condition |
|---|---|
| **SYSTEMATIC** | the interval excludes 0.5 and lies entirely ABOVE it |
| **SUB-RANDOM-WALK** | the interval excludes 0.5 and lies entirely BELOW it |
| **ACCUMULATION** | the interval includes 0.5 and excludes 1.0 |
| **UNDETERMINED** | the interval includes both 0.5 and 1.0 |

**The direction clause is the repair of a defect in the earlier draft**, which
read SYSTEMATIC on any interval excluding 0.5. An interval lying entirely BELOW
0.5 also excludes it, and it means the per-block perturbations partly CANCEL
with depth — sub-random-walk growth, which is neither a random walk nor an error
of consistent sign. Labelling that SYSTEMATIC would invert the sentence that
follows this table. It gets its own cell and is reported as itself. The four
cells are exhaustive: an interval either includes 0.5 or not; if not it is above
or below; if it does, it either includes 1.0 or not.

A random walk over independent per-block perturbations predicts `p = 0.5`; an
error that adds with a consistent sign predicts `p` near 1.0. The eight points
already recorded give `p = 0.651 +/- 0.066`, interval `[0.489, 0.813]`. That
interval INCLUDES 0.5 and EXCLUDES 1.0, so applying the cells above reads
**ACCUMULATION** — the same reading `## What is established, and what is not`
gives it, and **not the UNDETERMINED an earlier draft of this line asserted**,
which was a misapplication of this rule to its own data rather than a different
rule. It is a reading over ONE prompt and ONE prefill and it decides nothing;
R3 exists to re-run it over the corpus. **The divergence statistic must be stated with the run**,
because a mean of per-element ratios and a ratio of means are different
quantities that this project has already confused once.

**R3 is REPORTED-ONLY and cannot move R5.** Its cells name what the depth curve
looks like; none of them enters the composition below in either direction. This
is stated so that a named verdict is not mistaken for a lever.

### R4 — materiality, conditional on the oracle arm

The scale of "two faithful implementations of the same weights disagree by this
much" is measured, not assumed: `S = |mean NLL_cpu - mean NLL_llamacpp|`.

| Verdict | Condition |
|---|---|
| **MATERIAL** | `delta >= S` |
| **SUB-MATERIAL** | `0 < delta < S` |
| **PENDING** | the oracle arm did not run |

**If the oracle arm does not run, no materiality anchor exists and none is
invented.** R4 reads `PENDING` and names #933. R1's directional verdict stands
alone and is reported as directional. A magnitude threshold picked from nothing
would be exactly the after-the-fact threshold this section exists to prevent.

Second oracle reading, reported whenever the arm runs: if **both** arms sit
materially above the oracle in the same direction, the defect is shared and
neither of our arms is ground truth. That outcome is reported on its own and is
not folded into R1.

**R4's VERDICT is reported-only and cannot move R5**, exactly as R3's is.
`MATERIAL`, `SUB-MATERIAL` and `PENDING` name what the magnitude means; none of
them enters the composition below.

**R4's MEASUREMENT is a different thing from R4's verdict, and it does reach
R5.** `S` is R1's equivalence margin `M`, so whether the oracle arm RAN decides
whether R1's NOT-DISTINGUISHED cell is reachable at all. That is deliberate and
is written here so the coupling is visible rather than surprising: without a
measured scale there is no margin, and without a margin an interval containing 0
is `UNDERPOWERED` and not a tie. `S` is measured against a pinned oracle rather
than chosen, and it touches only the branch where R1's interval includes 0 —
DEGRADED and CPU-DEGRADED are decided by the interval and `C` alone, so a large
`S` cannot convert an adverse verdict into a favourable one.

### R5 — composition into one verdict

| Row verdict | Condition |
|---|---|
| **DEGRADED** | R1 is DEGRADED, whatever R2, R3 and R4 read |
| **NOT-DISTINGUISHED** | R1 is NOT-DISTINGUISHED and R2 is NOT-SYSTEMATIC, whatever R3 and R4 read |
| **UNDETERMINED** | every other combination, including R1 UNDERPOWERED, R1 INCONSISTENT, R1 CPU-DEGRADED, R2 UNDETERMINED, and any failed precondition P0 to P7 |

**Only R1 and R2 compose. R3 and R4 are reported-only**, and both rows above say
so rather than only the first, because a rule that is silent about an input in
one cell invites the reading that the input applies there. R1 alone can produce
DEGRADED; R1 and R2 together are required for NOT-DISTINGUISHED; everything else
is UNDETERMINED.

What each verdict means for the row, so the consequence is fixed in advance too:

* **DEGRADED.** G0-CORRECT stays FAILING and the divergence is a defect. No
  distributional gate may be ratified on this evidence. The next step is the
  localization the row's `## Owed` already carries.
* **NOT-DISTINGUISHED.** The token-exact cross-arm instrument is measuring a
  coin flip on this workload. Ratifying a distributional gate becomes a live
  option **for the operator**, which `AGENTS.md` reserves as an explicit act.
  This wave does not ratify one and does not recommend one. **This is an
  equivalence result and not an absence of evidence**: R1's cell requires the
  interval inside `[-M, +M]`, so an interval that merely contains 0 reads
  UNDERPOWERED and lands in the row below instead. That distinction is the whole
  reason the margin exists, because the door this bullet opens must not be
  opened by a thin run.
* **UNDETERMINED.** Nothing changes. G0-CORRECT stays FAILING.

`G0-SPEED` stays VOID in all three.

## Preconditions, probed before the measurement

Each is cheap, each can void the run, and each is reported with its result. A
failed precondition gives `UNDETERMINED` under R5; it never gives a silent pass.

* **P0 — the arms are two arms.** The CPU and CUDA legs run the same binary and
  the same checkpoint, and a control that the arms computed differs between
  them. `AGENTS.md` and [`../benchmarking.md`](../benchmarking.md) both require
  it: a pair that measured one artifact twice has already produced a "no
  difference" result in this tree.
* **P1 — the processor is reached, on both arms, over an unedited row.** The
  callback increments a counter, and the counter must read exactly `L` per
  corpus item on each arm. A greedy request must route through
  `apply_logits_processors`; if it does not, the instrument is absent and the
  run is void, not passing. The same probe reports the five upstream sampler
  readings the `## Constraints` table pins — `allowed_token_ids_mask` absent,
  `bad_words_token_ids` empty, `min_tokens` empty, `logit_bias` empty,
  `no_penalties` true — and **any one of them non-neutral voids the run**,
  because the four that run before `sampler.cpp:441` edit the row the NLL is
  read from and the fifth edits the row the mask wrote. It also prints
  `cudaPointerGetAttributes` on the row pointer the callback receives, on the
  CUDA arm, which is the reading the row's `## Owed` says the W0e segfault never
  got.
* **P2 — the oracle arm, bounded.** One time-boxed attempt to build the pinned
  fork and run `llama-perplexity` over a short pinned id sequence from the same
  GGUF on local NVMe. A pass makes R4 live and is reported to #933. A failure is
  recorded with its exact failure mode, R4 reads `PENDING`, and no further lease
  time goes to the arm.
* **P3 — teacher forcing actually forces.** The emitted ids equal the corpus ids
  at every position on both arms. If an arm emits anything else, the mask did
  not apply and every NLL from that run is void.
* **P4 — the clock window is real.** At least 30 retained busy samples and a
  majority of the window busy, per
  [`bench-assert-clock-state.md`](bench-assert-clock-state.md). A failed window
  is recorded and does not void the NLL verdict, for the reason under
  `## Constraints`.
* **P5 — the streaming lane stayed live, which is this row's G0-LIVE.** The
  harness snapshots `detail::ExpertStreamSnapshot()` immediately after the
  prefill step and again at the end of each corpus item, and the **decode-phase
  `exhausted` delta must be 0** with `forced == 0`. Prefill exhaustion is
  structural on this checkpoint and is not the criterion; the difference is.
  These prompts are 16 to 40 ids against the 5 W0g used, so the protected slice
  set is larger and the decode-phase delta is not free. A non-zero delta or a
  non-zero `forced` voids the run: a slice served by the forced fallback is not
  the arm the experiment is comparing.
* **P6 — the CUDA logits seam is safe to dereference.** This precondition
  exists because the instrument itself is currently defective on the arm under
  test: `apply_logits_processors` gates on `UnifiedMemory()` where it needs
  `DeviceMemoryIsHostAddressable()`, so it hands a `cudaMalloc` pointer to a
  host dereference on GB10 ([#1746](https://github.com/mudler/vllm.cpp/issues/1746),
  the chain is read out in `## Design`). Three readings, all of them cheap, and
  **all three must hold before any corpus item runs**:
  1. #1746 is CLOSED and its fix is an ancestor of the source SHA the run
     records — `git log --oneline --grep '#1746'` and `git merge-base --is-ancestor`,
     not a report that it landed;
  2. `builtin.cpp`'s gate reads `DeviceMemoryIsHostAddressable()` in the tree
     that produced the binary, asserted by `git grep -n` in the recorded SHA;
  3. P1's `cudaPointerGetAttributes` on the CUDA arm reports a pointer the host
     may dereference, or the seam reports that it staged a copy instead.
  **A failed P6 stops the run before it starts.** It gives `UNDETERMINED` under
  R5, never a verdict, and this is the correct polarity: an instrument that
  faults produces no number, and an instrument that silently reads unmapped
  host memory produces a number that is worse than none. Nothing in this wave
  may work around #1746 in the harness; a private staging copy in an ABI client
  would measure a different seam from the one the design names.
* **P7 — the corpus is what it says it is, probed and not only recorded.**
  `## The corpus` records the source, its revision, the tokenizer revision and a
  sha256. **A sha256 pins the file against itself and says nothing about where
  the ids came from**, and the failure this guards is the one `## Design` calls
  the single easiest way to get a confident wrong answer: a continuation built
  from an arm's own generation hands that arm the maximum-probability token at
  every position, and **P3 cannot see it** — P3 checks that the emitted ids equal
  the corpus ids, which is exactly what forcing guarantees, so P3 passes by
  construction on a contaminated corpus. The probe is therefore about ORIGIN:
  1. fetch the pinned source at its pinned revision and record its sha256; a
     revision that no longer resolves, or a body whose hash differs from the one
     recorded, fails;
  2. for every one of the 17 items, detokenize `prompt_ids ++ continuation_ids`
     with the pinned tokenizer revision and assert the result is a **contiguous
     substring** of that source body, at the byte offset the corpus file records;
  3. assert the 16 scored items' offsets are strictly increasing and
     non-overlapping, which is what "walk it in order" in the selection rule
     means and which a hand-edited item would break.
  Reading 2 is the one that matters: text either arm generated is not a span of
  a pinned public document, so a contaminated item cannot pass it, while every
  item the selection rule actually produced does. **A failed P7 voids the run
  and gives `UNDETERMINED`.** It never downgrades to a warning, because the
  failure it detects produces a confident DEGRADED rather than a missing one.

## Risks and decisions

| Risk | Decision |
|---|---|
| Forcing the arms onto either arm's own greedy output would hand that arm every maximum-probability token, and the run would report a confident DEGRADED that measured which arm wrote the text. | The corpus continuation is held-out natural text, and **P7 probes its ORIGIN**. **P3 cannot probe this and an earlier draft of this row credited it with doing so**: P3 asserts the emitted ids equal the corpus ids, which forcing guarantees, so P3 passes by construction on a contaminated corpus. The recorded provenance and sha256 in `## The corpus` are a record and not a probe either — a sha256 pins the file against itself. P7 detokenizes every item against the pinned source at its pinned revision and requires a contiguous, non-overlapping, in-order span. A failure voids the run. |
| The logits-processor stage might not run on a greedy request. | P1. A counter that reads 0 voids the run rather than passing it. |
| **`apply_logits_processors` dereferences a `cudaMalloc` pointer on the CUDA arm.** It gates on `UnifiedMemory()` (`builtin.cpp:93`), which GB10 answers TRUE, and then casts `logits.data` to a host pointer (`:98`), while `CudaBackend::Alloc` is a plain `cudaMalloc` and CUDA never overrides the narrow `DeviceMemoryIsHostAddressable()`. This is the instrument the whole design rests on, and the fault class already cost #844, #1435 and #960. | **NOT mitigated here; blocked on [#1746](https://github.com/mudler/vllm.cpp/issues/1746), which is a PREREQUISITE for the run and not a follow-up.** This wave lands no product code and a separate implementer owns the repair. **P6** refuses the run until the fix is in the built binary, and a failed P6 gives `UNDETERMINED` under R5. P1 prints `cudaPointerGetAttributes` on the row pointer regardless. |
| 16 prompts is thin for a bootstrap, so an underpowered run produces an interval containing 0 and would map straight onto the verdict that opens the distributional-gate door. | **Gated, not merely stated.** R1's NOT-DISTINGUISHED cell requires the interval to lie INSIDE the equivalence margin `M = S` and not merely to contain 0; an interval that contains 0 and is wider than the margin reads `UNDERPOWERED`, which R5 composes to UNDETERMINED. The pre-registered minimum detectable effect is `0.533 x sd` at `P = 16` and `0.635 x sd` at the `P = 12` floor, and the run reports the realized half-width beside the verdict. The bootstrap unit stays the prompt, `P` is never traded for `L`, and `P` never goes below 12. **While R4 is PENDING on #933 there is no margin, so NOT-DISTINGUISHED is unreachable** and this risk cannot be realised at all. |
| The oracle arm may not run at all. | P2 time-boxes it. R4 reads PENDING and R1 stands alone. What that costs is written in `## The oracle arm`. |
| The fork branch `iq1-narrow` can be rebased under its name. | Re-verify the pin and the anchors at run time, per the oracle file's own instruction. |
| A logits processor forces a host round-trip per step and changes decode timing. | Irrelevant here and stated so no later reader mines these logs for a rate: this run publishes no speed number, and its decode times are not comparable with W0g's. |
| A 16-to-40 id prompt is longer than W0g's 5 and can force the in-place tower fallback during decode. | P5 gates it, and the fallback is pre-registered rather than improvised: shorten every prompt to the shortest first sentence the selection rule accepts, re-derive the corpus file, and **re-run the whole corpus**. Never mix a shortened item into a run taken at the other length. |
| The row spec and this file could drift. | The row spec links this file and does not restate the rule. This file is the binding copy of the rule, and says so. |
| Another lane could edit a threshold here after a run. | `## Now` names the audit command. A reviewer checks `git log --follow` on this path against the run date. |

## Tests

This wave lands no product code, so it ports no upstream test. What it owes when
the harness lands, as a separate dispatch:

| Test | What it pins | Mutations that must red it |
|---|---|---|
| corpus loader | `benchmarks/w0h_corpus.json` parses to 17 items, the ids match the recorded sha256, and no item's continuation holds an end-of-sequence id | corrupt one id; drop the sha256 check; accept a 15-item file |
| the forcing processor | after the mask, `argmax(logits) == target`, and the recorded NLL comes from the **pre-mask** row | record NLL after the mask (it would read 0 everywhere); mask all but the wrong id; leave one competitor unmasked |
| the statistics | R1, R2, R3 and R5 return the pre-registered verdict on synthetic inputs constructed for **each cell of each table**, including R1 `INCONSISTENT`, R1 `UNDERPOWERED` (both the wider-than-`M` case and the `M`-unavailable case), R2 `UNDETERMINED` at `N = 0`, R3 `SUB-RANDOM-WALK`, and R5's UNDETERMINED catch-all | move a threshold by one step in each direction; resample positions instead of prompts; drop the tie exclusion in R2; drop the `[-M, +M]` containment from R1 so an underpowered interval reads NOT-DISTINGUISHED; drop the ABOVE clause from R3 so an interval below 0.5 reads SYSTEMATIC; score a fully-tied position as non-rejecting; let R3 or R4 change an R5 cell |
| the preconditions | P6 refuses a binary whose `builtin.cpp` still gates on `UnifiedMemory()`, and P7 refuses a corpus item whose ids do not detokenize to a contiguous in-order span of the pinned source | pass a pre-#1746 tree to P6; replace one item's continuation with an arm's own generation and confirm P7 REDS while P3 stays green, which is the pair that proves P3 could not have caught it |
| reachability | the harness enters the engine through `vllm_complete_tokens` and the ABI header only | delete the production call site per [`../reachability.md`](../reachability.md) |

**The statistics tests are the ones that matter for pre-registration**, because
they make the rule executable. A rule that only exists in prose can be
reinterpreted after a run; a rule with a test cannot be moved without a visible
diff to the test.

## Gates

This wave declares **no new gate of its own**. It produces one verdict for an
existing gate.

* **G0-CORRECT (`ENG-EXPERT-STREAM-DEVICE`).** Unchanged and still FAILING. This
  wave says what its failure MEANS, under R5. It does not change the gate's
  definition, and it does not make it pass.
* **G0-SPEED (`ENG-EXPERT-STREAM-DEVICE`).** VOID, unchanged, under every
  outcome.
* **W0h-VERDICT.** Exactly one of DEGRADED, NOT-DISTINGUISHED or UNDETERMINED,
  with R1 to R4 each reported as a named result, the preconditions each reported
  as passed or failed, and the clock window recorded. A permanent report-only
  state is not a result.

## Evidence, and what a run must record

Nothing has run. When it does, the run records: the `rc` job id and device; the
source SHA and the built binary's sha256; the container image; the checkpoint
path, size and revision; `benchmarks/w0h_corpus.json` and its sha256; the full
environment block (`VT_GGUF_PREFAULT`, `VT_MOE_EXPERT_STREAM`,
`VT_MOE_EXPERT_STREAM_SLOTS`, `--max-num-seqs`, sampling parameters); the clock
window per arm; the per-position NLL and logit-delta arrays for both arms; the
P0 to P7 results; the R2 tie count `N_tie` with the measured `rho`, `deff`,
`N_eff` and implied standard error; the R1 realized bootstrap half-width and the
`sd` it implies; and the R1 to R5 verdicts with the exact command that
reproduces each from the arrays.

Per [`../benchmarking.md`](../benchmarking.md) §Recording it, that record goes to
[`../benchmark-record.md`](../benchmark-record.md). **No public document is owed
by this wave.** `AGENTS.md` §`Public documents` triggers `docs/BENCHMARKS.md` on
a public benchmark ID being added, removed, or changing disposition, and this
wave adds no benchmark ID and publishes no number. A lifecycle change owes the
moved row spec's `## Now`, and this wave moves no row: `ENG-EXPERT-STREAM-DEVICE`
stays `ACTIVE`. What the run does owe is a W0h bullet in that row spec's
`## Now`, written in the same commit that lands the result.

## Stop conditions

* **P6 fails.** Stop before the run starts. #1746 has not landed, so
  `apply_logits_processors` dereferences a `cudaMalloc` pointer on the very arm
  under test. Report `UNDETERMINED`, and do NOT work around it in the harness:
  a private staging copy in an ABI client measures a different seam from the one
  this design names.
* **P7 fails.** Stop. The corpus is not a span of the pinned source, so its
  provenance is unproven and a contaminated item would produce a confident
  DEGRADED that P3 cannot see. Report `UNDETERMINED`, re-derive the corpus from
  the selection rule, and re-run whole.
* **P1 or P3 fails.** Stop. The instrument is not measuring what the design
  says. Report `UNDETERMINED` and fix the instrument in a separate dispatch.
* **P0 fails.** Stop. Two arms that are one arm produce a tie by construction.
* **P5 fails.** Stop. A non-zero decode-phase `exhausted` delta or a non-zero
  `forced` means a slice came from the fallback, so the run measured a mixture
  of two paths rather than the arm. Shorten the prompts per the pre-registered
  fallback and re-run the whole corpus.
* **The lease ends before both arms complete the same corpus.** Stop and report
  what ran. A partial corpus is not a smaller corpus: the bootstrap unit count
  changes, and R1's thresholds are written for `P = 16`. Re-run whole.
* **`P < 12` would be needed to fit the lease.** Stop and re-plan. Do not lower
  `P`.
* **The result is DEGRADED.** Stop this wave. Do not begin the localization in
  the same flow; it is a different measurement with a different instrument, and
  the row's `## Owed` already owns it.
* **Any impulse to move a threshold after seeing a number.** Stop and return
  `NEEDS_DECISION` to the operator. That is what pre-registration is for.

## Out of scope

* **Fixing the divergence.** This wave measures. A fix needs the first differing
  operation named, which is a separate measurement.
* **Localizing the first differing operation inside block 0.**
  [expert-stream-device-slots.md](expert-stream-device-slots.md) `## Owed`
  already owns it and it needs a different instrument.
* **Ratifying a distributional gate.** `AGENTS.md` reserves that for the
  operator as an explicit act. A NOT-DISTINGUISHED verdict makes the decision
  available and does not take it.
* **Generalizing the method to other two-arm divergences**, such as `A2-Q1` on
  Nemotron ([#1388](https://github.com/mudler/vllm.cpp/issues/1388)), which has
  the same shape. Lifting a method that has never run once is speculation. If
  W0h returns a usable verdict, the lift is a candidate row then.
* **Re-deriving the top-k selection over all 552 token-rows.** Needs no lease,
  only the two W0g dumps, and the row's `## Owed` already carries it.
* **A speed number.** `G0-SPEED` stays VOID. The decode times this run produces
  are not comparable with W0g's, because the logits processor adds a host
  round-trip per step.

## Owed

| Owed | Why it is open |
|---|---|
| **The harness itself.** `benchmarks/w0h_corpus.json`, the forcing processor, the P0 to P7 probes, the statistics, and their tests. | Deliberate. This wave is the pre-registration, and `AGENTS.md` §`Spec before code` requires the spec to be committed first. The implementation is a separate dispatch against this file. |
| **`gateable` for `llama-cpp-unsloth` on this checkpoint** ([#933](https://github.com/mudler/vllm.cpp/issues/933)). | Not owned here. P2 is a bounded probe whose result is reported to #933 either way. A pass discharges part of that debt; this file does not claim the discharge. |
| **`apply_logits_processors` dereferences a `cudaMalloc` pointer on GB10** ([#1746](https://github.com/mudler/vllm.cpp/issues/1746)). | **Not owned here, and a PREREQUISITE rather than a follow-up.** This wave lands no product code and a separate implementer owns the repair. It matters to this file more than an ordinary dependency does, because the defect is in the instrument the whole design rests on: the seam asks `UnifiedMemory()` where it needs `DeviceMemoryIsHostAddressable()`, which is the class `src/vt/op_provider.cpp:866-873` records as having cost #844, #1435 and #960. P6 refuses the run until the fix is an ancestor of the recorded source SHA. |
| **Clock pinning on `dgx:gpu0` for rows outside `BENCH-QWEN38-27B-SOTA`** ([#1354](https://github.com/mudler/vllm.cpp/issues/1354)). | Not owned here, and this wave does not need it: NLL does not depend on the clock rate. Named so that the recorded clock window is read as provenance and not as a controlled variable. |
| **The W0e completion-callback SIGSEGV on the CUDA arm.** | Still owned by the row, and no longer unexplained-with-no-candidate. [#1746](https://github.com/mudler/vllm.cpp/issues/1746) is a **candidate cause of the same class**: a CUDA `logits` pointer dereferenced on the host because a seam asked `UnifiedMemory()` where it needed `DeviceMemoryIsHostAddressable()`, which is what #844, #1435 and #960 each were. **It is a candidate and not a confirmed cause**: nobody has re-run W0e against the fix, the completion callback is a different seam from `apply_logits_processors`, and until that re-run happens the attribution is a hypothesis. Confirming or refuting it stays with the row. P1 prints `cudaPointerGetAttributes` on **this** instrument's pointer, which is a reading about this seam and not a diagnosis of that one. |
