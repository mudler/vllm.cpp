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
`apply_logits_processors` synchronizes the backend, obtains a host-addressable
view of the `[num_reqs, vocab]` logits (the pointer itself on a unified backend,
a staged copy on a discrete one), calls each registered processor, and copies
back where it staged.

That gives the experiment three properties it would otherwise have to build:

1. **It is a production entry point.** The harness is an ABI client and includes
   no internal header, which is what `AGENTS.md` §`Shared seams` requires of
   examples and servers.
2. **It works identically on both arms.** The host/device staging is inside the
   seam, so the CPU and CUDA arms run the same callback over the same kind of
   pointer.
3. **It does not repeat the W0e fault.** The scratch instrument that read
   `logits` in the **completion** callback SIGSEGVs on the CUDA arm
   (`SCRIPT_EXIT=139`, cause unmeasured, carried under the row's `## Owed`).
   That is a different seam. This one is host-addressable by construction. The
   precondition is still probed rather than assumed: see **P1**.

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
threaded the read term is about 1.41 s of a 9.055 s CPU step, near 15.6 %, so
read-issue order can move at most about 0.64 s. And it says nothing about
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

| Verdict | Condition |
|---|---|
| **DEGRADED** | the interval excludes 0, and `delta > 0`, and `C >= 12` |
| **CPU-DEGRADED** | the interval excludes 0, and `delta < 0`, and `C <= 4` |
| **NOT-DISTINGUISHED** | the interval includes 0 |
| **INCONSISTENT** | the interval excludes 0 and the matching `C` condition fails |

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

### R2 — bias versus noise

For each position, take `d_j = logits_cuda[j] - logits_cpu[j]` over `j` in the
CPU arm's top-`K` tokens, `K = 64`. Count `n_+ = #{j : d_j > 0}` and apply a
two-sided binomial sign test against `p = 0.5` at `alpha = 0.01`. Let `R` be the
fraction of positions that reject.

| Verdict | Condition |
|---|---|
| **SYSTEMATIC** | `R > 0.05` |
| **NOT-SYSTEMATIC** | `R <= 0.05` |

The null predicts `R = 0.01`. With `N = 320` positions the binomial standard
error on that rate is `sqrt(0.01 x 0.99 / 320) = 0.0056`, so the 0.05 threshold
sits about 7 standard errors above the null. It is set well above the nominal
rate on purpose, because positions inside a prompt are correlated and a
threshold at the nominal `alpha` would over-reject. Ties (`d_j == 0`) are
excluded from `n_+` and from `K`, and the excluded count is reported, because a
large tie count is itself a finding about the two kernels.

### R3 — depth exponent

From a block-wise hidden-state dump at a fixed set of teacher-forced positions,
fit `log(divergence)` on `log(block + 1)` by least squares and take the 95 %
interval on the slope `p`.

| Verdict | Condition |
|---|---|
| **SYSTEMATIC** | the interval excludes 0.5 |
| **ACCUMULATION** | the interval excludes 1.0 and includes 0.5 |
| **UNDETERMINED** | the interval includes both |

A random walk over independent per-block perturbations predicts `p = 0.5`; an
error that adds with a consistent sign predicts `p` near 1.0. The eight points
already recorded give `p = 0.651 +/- 0.066`, interval `[0.489, 0.813]`, which is
**UNDETERMINED** by this rule. R3 exists to re-run it over the corpus rather
than over one prompt. **The divergence statistic must be stated with the run**,
because a mean of per-element ratios and a ratio of means are different
quantities that this project has already confused once.

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

### R5 — composition into one verdict

| Row verdict | Condition |
|---|---|
| **DEGRADED** | R1 is DEGRADED, whatever R2, R3 and R4 read |
| **NOT-DISTINGUISHED** | R1 is NOT-DISTINGUISHED and R2 is NOT-SYSTEMATIC |
| **UNDETERMINED** | every other combination, including R1 INCONSISTENT, R1 CPU-DEGRADED, and any failed precondition |

What each verdict means for the row, so the consequence is fixed in advance too:

* **DEGRADED.** G0-CORRECT stays FAILING and the divergence is a defect. No
  distributional gate may be ratified on this evidence. The next step is the
  localization the row's `## Owed` already carries.
* **NOT-DISTINGUISHED.** The token-exact cross-arm instrument is measuring a
  coin flip on this workload. Ratifying a distributional gate becomes a live
  option **for the operator**, which `AGENTS.md` reserves as an explicit act.
  This wave does not ratify one and does not recommend one.
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
* **P1 — the processor is reached, on both arms.** The callback increments a
  counter, and the counter must read exactly `L` per corpus item on each arm.
  A greedy request must route through `apply_logits_processors`; if it does not,
  the instrument is absent and the run is void, not passing. The same probe
  prints `cudaPointerGetAttributes` on the row pointer the callback receives, on
  the CUDA arm, which is the reading the row's `## Owed` says the W0e segfault
  never got.
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

## Risks and decisions

| Risk | Decision |
|---|---|
| Forcing the arms onto the CPU arm's own greedy output would hand the CPU arm every maximum-probability token. | The corpus continuation is held-out natural text. Named in `## Design` and probed by P3. |
| The logits-processor stage might not run on a greedy request. | P1. A counter that reads 0 voids the run rather than passing it. |
| The CUDA arm segfaulted the last time an instrument read `logits`. | Different seam. `apply_logits_processors` synchronizes and takes a host-addressable view before calling. P1 prints `cudaPointerGetAttributes` anyway, because the earlier fault was never diagnosed. |
| 16 prompts is thin for a bootstrap. | Stated rather than hidden. The bootstrap unit is the prompt, `P` is never traded for `L`, and `P` never goes below 12. A NOT-DISTINGUISHED verdict from `P = 16` is a statement about power as much as about the arms, and is reported that way. |
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
| the statistics | R1, R2 and R3 return the pre-registered verdict on synthetic inputs constructed for each cell of each table, including INCONSISTENT and UNDETERMINED | move a threshold by one step in each direction; resample positions instead of prompts; drop the tie exclusion in R2 |
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
P0 to P5 results; and the R1 to R5 verdicts with the exact command that
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
| **The harness itself.** `benchmarks/w0h_corpus.json`, the forcing processor, the statistics, and their tests. | Deliberate. This wave is the pre-registration, and `AGENTS.md` §`Spec before code` requires the spec to be committed first. The implementation is a separate dispatch against this file. |
| **`gateable` for `llama-cpp-unsloth` on this checkpoint** ([#933](https://github.com/mudler/vllm.cpp/issues/933)). | Not owned here. P2 is a bounded probe whose result is reported to #933 either way. A pass discharges part of that debt; this file does not claim the discharge. |
| **Clock pinning on `dgx:gpu0` for rows outside `BENCH-QWEN38-27B-SOTA`** ([#1354](https://github.com/mudler/vllm.cpp/issues/1354)). | Not owned here, and this wave does not need it: NLL does not depend on the clock rate. Named so that the recorded clock window is read as provenance and not as a controlled variable. |
| **The W0e completion-callback SIGSEGV on the CUDA arm.** | Still unexplained, still owned by the row. P1 prints `cudaPointerGetAttributes` on a **different** seam's pointer, which is a reading about this instrument and not a diagnosis of that one. |
