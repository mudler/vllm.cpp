# The expert selection at the disagreeing forwards, read on the matched pair, 6 September 2026

Wave MOESEL-RESULT of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2998](https://github.com/mudler/vllm.cpp/issues/2998), closing the open clause of
[#2547](https://github.com/mudler/vllm.cpp/issues/2547) that
[#2969](https://github.com/mudler/vllm.cpp/issues/2969) narrowed, and placing the
matched pair on the no-flip side of the layer-0 bracket
[#2552](https://github.com/mudler/vllm.cpp/issues/2552) left open.
What it leaves owed is [#2999](https://github.com/mudler/vllm.cpp/issues/2999).

**The one-line result.** An instrument has now read `qwen4_exp`'s expert selection
on **every forward the request runs**, including forwards 4, 6 and 7, where the two
arms' sampled ids disagree. The selection flips **extensively** between the arms:
**296 of 576 slots over 8 forwards**. At the algorithm-matched pair's layer 0 it
does **not** flip at all, which places `4.324e-05` on the no-flip side of #2552's
bracket.

**Read the limit before the result.** **The flips do not explain the token
disagreement, and this document does not claim they do.** Forward 5 flips 48 of 48
expert slots and its sampled id **agrees**. Forward 7 flips 48 of 48 and its id
**disagrees**. Forward 4 **disagrees** while flipping fewer slots (33 of 48) than
agreeing forward 5 flips. A per-forward flip count therefore separates nothing, in
either direction. What carries the three disagreeing ids is still unexplained; it
is narrowed here, not closed, and [#2999](https://github.com/mudler/vllm.cpp/issues/2999)
owns what remains.

## 1. Provenance

**The log is committed here because it will not be re-readable.** `rc logs` on this
fleet ages out within a day and non-monotonically. That is an observation from
this campaign's own operator on other jobs, not a property this log shows: a job
one day old returned zero lines while an older one still returned 316. It is why
the capture is committed rather than cited. The full 369-line
capture is [`qwen4exp-moe-selection-fwd-20260906/rc-logs-9e0864da.txt`](qwen4exp-moe-selection-fwd-20260906/rc-logs-9e0864da.txt),
sha256 `80647d6afa73c2630669a631b5810a8fbf518271b3877127170e2c0c14ed9ec2`. It was
captured twice, from two sessions, and the two captures are byte-for-byte equal.

| Field | Value |
|---|---|
| `rc` job | `9e0864da-9b37-4309-b863-04810de0e068` |
| Device | `thor:gpu0` |
| Worker | `rc-worker-n8smh`, `aarch64`, 14 cpus |
| Silicon | `NVIDIA Thor, 11.0, 595.78` |
| `state` | `succeeded` |
| `exit_code` | `0` |
| `queued_at` | `2026-09-05T21:10:41Z` |
| `started_at` | `2026-09-06T02:15:40Z` |
| `finished_at` | `2026-09-06T02:21:41Z` |
| Job's own DONE marker | `MOESEL-DONE-OK 2026-09-06T02:19:57Z` |

**`exit_code = 0` is not the result and nothing here rests on it.** It says the
script ended. A job on another row of this campaign exited 0 while a read inside it
logged `RC=1`. Every claim below is quoted from the job's own per-step output. All
15 `SUM ..._RC=` markers in the log read `0`; the log contains no `RC=` value
greater than zero, checked with a positive control on the same pattern shape (30
`RC=` lines matched, 0 of them non-zero).

**The binary is wave ARMTOKENS', by digest, and it carries the tap.**

```text
### 02:15:47 === B. THE BINARY IS WAVE ARMTOKENS', BY DIGEST ===
SUM BINCOPY_RC=0
RESULT BINARY sha256=1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f (expect 1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f)
SUM BINSHA_RC=0
RESULT BINARY EXECS: rc=0 help_lines=37
RESULT BINARY CARRIES: VT_MOE_SEL_FP=1(>=1) moesel_fmt=2(>=1) poscontrol=3(>=1) negcontrol=0(==0)
SUM BININSTR_RC=0
```

`negcontrol=0` is asserted beside the digest. A build in which the tap's negative
control had been compiled in would report a non-zero count there, and the run would
be reading an instrument rather than the model.

**The artifact is the released checkpoint, verified inside the lease.**

```text
### 02:15:53 === C. THE ARTIFACT IS THE SAME ONE, STAGED WORKER-LOCAL ===
RESULT ARTIFACT STAGE REUSED (no copy)
RESULT ARTIFACT shard1 sha256=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd (expect 88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd)
RESULT ARTIFACT shards=3 staged_bytes=72546461650
SUM ARTIFACT_RC=0
```

That is `unsloth/Qwen3.8-Flash-Next-GGUF` @ `8bdc666649440e9bdc97e16f3f75782c98478ff5`,
path `UD-IQ1_S`, as [`docs/USAGE.md`](../USAGE.md) records it.

## 2. The three arms, and the assertion that the tap did not change the answer

One binary, `VT_MOE_SEL_FP=512` and `VT_CPU_QUANT_REPACK=0` on each.

| Arm | Device | Sampled ids | Expected | Drift |
|---|---|---|---|---|
| A-CPU | `cpu` | `11751 13 15767 411 2029 11 1092 369` | same | `NONE` |
| B-CUDA | `cuda` | `11751 13 15767 411 1928 11 628 567` | same | `NONE` |
| C-CPU2 | `cpu` | `11751 13 15767 411 2029 11 1092 369` | same | `NONE` |

Each arm asserts its ids against the sequence the **same binary** produced with the
tap off. **An instrument whose failure looks like a result is this row's defining
trap**, and this is the guard against it: a readback that perturbed sampling would
be caught here rather than published as a finding.

```text
RESULT ARM A-CPU TOKEN IDS: 11751 13 15767 411 2029 11 1092 369
RESULT ARM A-CPU EXPECTED  : 11751 13 15767 411 2029 11 1092 369
RESULT ARM A-CPU DRIFT: NONE (the tap did not perturb the sampled ids)
SUM ARM_A-CPU_DRIFT=0
RESULT ARM B-CUDA TOKEN IDS: 11751 13 15767 411 1928 11 628 567
RESULT ARM B-CUDA EXPECTED  : 11751 13 15767 411 1928 11 628 567
RESULT ARM B-CUDA DRIFT: NONE (the tap did not perturb the sampled ids)
SUM ARM_B-CUDA_DRIFT=0
RESULT ARM C-CPU2 TOKEN IDS: 11751 13 15767 411 2029 11 1092 369
RESULT ARM C-CPU2 EXPECTED  : 11751 13 15767 411 2029 11 1092 369
RESULT ARM C-CPU2 DRIFT: NONE (the tap did not perturb the sampled ids)
SUM ARM_C-CPU2_DRIFT=0
SUM TOKENDRIFT_RC=0
```

Each arm recorded 960 `moesel` lines, of which 384 are digests and 576 are
per-token value lines, and each arm's own last digest reports `lines=576`, which
agrees with the number of value lines seen.

**The `Killed` lines in the log are teardown, not a crash.** Each arm's block
carries `RESULT ARM <TAG> FIRST ERROR LINE:` with **nothing after it**. That field
is a grep over the server log for `engine-fatal|vt cuda:|illegal|terminate
called|what()`, and it matched none of them on any arm. The
`job.sh: line 122: ... Killed` line that follows is bash's own job-control message
for the `kill -9 $SRV` the harness runs to tear the server down after the arm's
request has already returned `http=200`. Read the empty error field, not the
`Killed`.

**The budget was not the limiter; the request length was.** `VT_MOE_SEL_FP=512`
allows 512 MoE calls, and 384 were taken. The capture carries the forward count
directly, on two independent lines: `RESULT IDS` lists **eight** sampled token ids
per arm, and the comparator asserts the structure it derived from the data as
`STRUCTURE forwards=8 moe_blocks_per_forward=48` beside
`contiguous_and_uniform=True`. 8 forwards x 48 blocks = **384** MoE calls against a
512 budget, so **the tap covered every forward this request ran** and stopped
because the request finished. It did not stop on its budget.

**The arms' `steps completed: 11` is a scheduler count, not a forward count**, and
the split between the two is not quoted here. `job.sh:168` derives that field as
`grep -c 'core-step end'` over each arm's `server.log`, which lives on the fleet
share that §10 declares scratch; the strings `core-step` and `model_executed`
appear **zero** times in this capture (positive controls in the same command
shape over the same file: `steps completed` 6, `RESULT` 243). The 384-call
conclusion above does not need that split, because `RESULT IDS` and `STRUCTURE`
are both in the capture and already over-determine it.

## 3. The negative control: A against C flips nothing

A and C are the same arm run twice. Zero flips is also what a broken comparison
prints, so a zero here is what makes the zero-or-not from A against B readable.

```text
RESULT SELFWD CONTROL-A-vs-C GROUP DIVERGING forwards=[4, 6, 7] slots=144 FLIPPED=0
RESULT SELFWD CONTROL-A-vs-C GROUP AGREEING forwards=[1, 2, 3, 5] slots=192 FLIPPED=0
RESULT SELFWD CONTROL-A-vs-C HIGHEST FORWARD REACHED = 7 (the open clause needs >= 7)
RESULT SELFWD CONTROL-A-vs-C VERDICT total_flipped_slots=0 over_forwards=8
RESULT CONTROL A-vs-C total_flipped_slots=0 (MUST be 0)
SUM CONTROL_FLIPS=0
```

Zero on all 8 forwards, individually, in the log's per-forward rows. The
comparator discriminates.

## 4. The measurement: A against B, grouped by forward

The comparator derives blocks-per-forward from the data rather than assuming 48,
and asserts the structure before it quotes a verdict:

```text
RESULT SELFWD MEASURE-CPU-vs-CUDA STRUCTURE calls_common=384 prefill_calls(L)=48 decode_calls=336 T_prefill=5
RESULT SELFWD MEASURE-CPU-vs-CUDA STRUCTURE contiguous_and_uniform=True
RESULT SELFWD MEASURE-CPU-vs-CUDA STRUCTURE forwards=8 moe_blocks_per_forward=48
```

| fwd | phase | sampled id | ids agree | calls | slots | **flipped** | hash mismatch | first flip layer |
|---|---|---|---|---|---|---|---|---|
| 0 | prefill | `11751` | yes | 48 | 240 | **53** | 35 | 4 |
| 1 | decode | `13` | yes | 48 | 48 | **22** | 22 | 12 |
| 2 | decode | `15767` | yes | 48 | 48 | **29** | 29 | 5 |
| 3 | decode | `411` | yes | 48 | 48 | **17** | 17 | 9 |
| **4** | decode | `2029` / `1928` | **no** | 48 | 48 | **33** | 33 | 8 |
| 5 | decode | `11` | yes | 48 | 48 | **48** | 48 | 0 |
| **6** | decode | `1092` / `628` | **no** | 48 | 48 | **46** | 46 | 1 |
| **7** | decode | `369` / `567` | **no** | 48 | 48 | **48** | 48 | 0 |

```text
RESULT SELFWD MEASURE-CPU-vs-CUDA GROUP DIVERGING forwards=[4, 6, 7] slots=144 FLIPPED=127
RESULT SELFWD MEASURE-CPU-vs-CUDA GROUP AGREEING forwards=[1, 2, 3, 5] slots=192 FLIPPED=116
RESULT SELFWD MEASURE-CPU-vs-CUDA HIGHEST FORWARD REACHED = 7 (the open clause needs >= 7)
RESULT SELFWD MEASURE-CPU-vs-CUDA VERDICT total_flipped_slots=296 over_forwards=8
```

### 4.1 What this establishes

**The row's "no instrument has observed a disagreeing step" clause is discharged
by measurement.** `HIGHEST FORWARD REACHED = 7`, and forwards 4, 6 and 7 each carry
48 calls and a per-forward verdict. #2969 argued from committed data that the
`VT_MOE_SEL_FP` window reaches those forwards, and was careful to say it had not
observed anything there. It has now been observed.

**The top-k selection genuinely flips between the arms, extensively.** 296 of 576
slots, over every forward, on a comparator whose negative control flips nothing.

### 4.2 What this does NOT establish, and the shape that makes it clear

**It does not establish that the flips cause the token disagreement.** The two
groups both flip heavily: 127 of 144 slots on the disagreeing forwards, 116 of 192
on the agreeing ones. Stated as counts that reads as "pervasive in both", which is
already enough to refuse the causal claim. The per-forward rates say something
sharper, and it is an **inversion** rather than an absence of signal:

| Comparison | Agreeing forward | Disagreeing forward |
|---|---|---|
| Highest flip rate in the run | **forward 5, 48/48 = 100%** | forward 7, 48/48 = 100% |
| Against the least-flipping disagreeing forward | forward 5, 100% | **forward 4, 33/48 = 69%** |

**The maximum flip rate in this run is reached by one forward from each group at
once.** Forward 5 flips every expert slot in all 48 layers and its sampled id
agrees on both arms; forward 7 flips every expert slot and its id disagrees. Two
forwards with an identical selection verdict land on opposite sides of the token
verdict. And the *lowest*-flipping disagreeing forward, forward 4 at 69%, flips
less than agreeing forward 5 at 100%.

**The flip rate rises roughly with decode depth on both populations** (45.8%,
60.4%, 35.4%, 68.8%, 100%, 95.8%, 100% for forwards 1 to 7). That is consistent
with divergence accumulating in the recurrent and KV state, but it is an
**observation and not a mechanism**: the run is one request, the ordering is not
monotone (forward 3 is the lowest at 35.4%), and nothing here isolates depth from
anything else that changes with it.

**The agreeing group is the log's own `GROUP AGREEING forwards=[1, 2, 3, 5]`, and
it excludes prefill forward 0.** Forward 0 is not comparable with a decode
forward: it carries 240 slots for five tokens against every decode forward's 48
for one, and its per-token flip count is the one figure §9 cannot reproduce from
committed data. State the direction plainly, because a chosen grouping invites the
question. Including forward 0 would make the agreeing group 169 of 432, or 39.1%,
against the disagreeing group's 88.2% -- it **widens** the gap between the two
populations and makes the causal reading look *stronger* than the log's own
grouping does. The exclusion therefore costs this document's conclusion rather
than buying it.

## 5. Layer 0 on the bracketed pair: no flip, and the bracket narrows

`x` at call 0 is the layer-0 MoE input, and the job asserts it against the pair
#2552 bracketed, so that a run on the wrong pair reports as such rather than as a
finding:

```text
### 02:19:55 === E2. THE PAIR IS THE BRACKETED ONE, ASSERTED ON A SECOND INSTRUMENT ===
RESULT CALL0 x A-CPU =3615.47142 (recorded L00 mhc.mix CPU-CHUNKED = 3615.47142)
RESULT CALL0 x B-CUDA=3615.62777 (recorded L00 mhc.mix CUDA-CTRL  = 3615.62777)
RESULT PAIR IDENTITY: CONFIRMED -- this is the 4.324e-05 matched pair
SUM PAIRIDENT_RC=0
```

```text
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 x_base=3615.47142 x_other=3615.62777 (compare with the recorded L00 mhc.mix pair)
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 sel_base=5bbd84f0b21b362b sel_other=5bbd84f0b21b362b hash_equal=True
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 LAYER-0 VERDICT: tokens=5 FLIPPED=0 -> NO FLIP (layer 0 agrees on every token)
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 tok=0 equal=True base_ulps=4 other_ulps=4 base_ids=32,46,71,130,160,215,226,286,309,342 other_ids=32,46,71,130,160,215,226,286,309,342
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 tok=1 equal=True base_ulps=2 other_ulps=2 base_ids=68,119,289,293,338,369,386,441,471,492 other_ids=68,119,289,293,338,369,386,441,471,492
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 tok=2 equal=True base_ulps=1 other_ulps=1 base_ids=21,76,101,265,293,301,314,338,361,369 other_ids=21,76,101,265,293,301,314,338,361,369
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 tok=3 equal=True base_ulps=1 other_ulps=1 base_ids=211,226,366,367,386,414,434,441,446,503 other_ids=211,226,366,367,386,414,434,441,446,503
RESULT SELFWD MEASURE-CPU-vs-CUDA CALL0 tok=4 equal=True base_ulps=0 other_ulps=0 base_ids=21,88,96,148,265,337,378,406,462,463 other_ids=21,88,96,148,265,337,378,406,462,463
```

All five prefill tokens select the same ten experts on both arms, in the same
order, with an equal `sel` digest, at boundary margins of 4, 2, 1, 1 and 0 ulps.
**Note the margins**: token 4's boundary is an exact bf16 tie on both arms and the
selection still agrees, so this is not a case where the tie question was avoided.

### 5.1 #2552's bracket resolves in the NO-FLIP direction, on a second instrument

| Pair | `L00 mhc.mix` | Layer-0 selection | Source |
|---|---|---|---|
| CPU-CTRL against CUDA `VT_GDN_CHUNKED=0` | `2.139e-05` | no flip | #2552 |
| **CPU-chunked against CUDA-chunked**, the algorithm-matched pair | **`4.324e-05`** | **NO FLIP** | **this reading** |
| CPU-CTRL against CUDA-PROD | `4.999e-04` | flip at token 2 | #2552 |

**The no-flip bound lifts from `2.139e-05` to `4.324e-05`.** The layer-0
expert-flip threshold now sits between `4.324e-05` (no flip) and `4.999e-04`
(flip). The reading is on a different instrument from the one that set the
bracket, and it reproduces the bracketing pair's own tensor values at call 0
before it reports anything.

**The bracket's two endpoints use CPU-CTRL as their base and this row uses
CPU-chunked**, so placing `4.324e-05` between them presumes the layer-0 flip
threshold is monotone in this norm across heterogeneous pairs. That presumption is
inherited from #2552's bracket and is not established by anything here. What this
reading establishes without it is the unconditional half: **at this pair, at layer
0, the selection does not flip.**

### 5.2 The mechanistic separation #2877 recorded as unknown

#2552 decomposed the layer-0 MoE residue onto two terms: the keep-quant grouped
expert GEMM's scale-sum reassociation, and a bimodal top-k selection term. #2877
recorded that **which of the two produces the residue at the matched pair was
unknown**, because the norm cannot say which side of the bracket `4.324e-05` is on
and a selection is a discrete property with bimodal error.

**This reading makes that separation at layer 0.** The matched pair's layer-0
residue contains **no selection flip at all**, so the bimodal top-k term is not
what produces it there. What remains is the keep-quant grouped expert GEMM's
scale-sum reassociation.

**"What remains" is an elimination inside a two-term decomposition, and it is only
as exhaustive as that decomposition.** The routing weights are themselves not
bit-identical at this call -- §7's `AXIS logit first_call_not_bit_identical=0` --
and a routing-weight difference that changes no selection belongs cleanly to
neither of #2552's two terms. So read the conclusion as "the top-k term is ruled
out here", which the flip verdict supports directly, rather than as "the GEMM
reassociation is the whole residue", which the two-term shape assumes. That limit
is inherited from #2552's decomposition and is not introduced by this reading.

**Read the scope exactly.** This is a statement about **layer 0**, on **this
pair**, in the **prefill**. It says nothing about the other 47 layers, where §4's
table shows the selection flipping heavily from forward 0 onward (forward 0's first
flip is at layer 4). It is not a claim that the top-k term is inert anywhere else,
and it is not an explanation of the three disagreeing ids.

## 6. Tie rates, and why they are not #2552's 32.9%

```text
RESULT SELFWD MEASURE-CPU-vs-CUDA TIE-RATE base exact_bf16_ties=143 of 576 boundaries = 24.8%
RESULT SELFWD MEASURE-CPU-vs-CUDA TIE-RATE other exact_bf16_ties=133 of 576 boundaries = 23.1%
RESULT SELFWD CONTROL-A-vs-C TIE-RATE base exact_bf16_ties=143 of 576 boundaries = 24.8%
RESULT SELFWD CONTROL-A-vs-C TIE-RATE other exact_bf16_ties=143 of 576 boundaries = 24.8%
```

| Reading | Exact bf16 ties at the top-k boundary | Population |
|---|---|---|
| #2552 | **32.9%** | prefill only, on a different pair |
| This run, A-CPU | **24.8%** (143 of 576) | all 8 forwards, 48 layers, the matched pair |
| This run, B-CUDA | **23.1%** (133 of 576) | all 8 forwards, 48 layers, the matched pair |

**These are not the same measurement, and the difference is not a discrepancy.**
#2552's 32.9% was taken over the prefill alone and on a different arm pair. This
run's boundaries are 576 slots spanning one prefill and seven decodes on the
matched pair. Comparing the two numbers as if they measured the same population
would be the error; they are placed side by side only to show that the tie density
this row has always described is present here too, at the same order of magnitude,
which is the whole of what the comparison supports.

C-CPU2 reproduces A-CPU's 143 exactly, as a repeat of the same arm must.

## 7. No ratio is quoted, in either direction

`rel(sumabs)` is a difference of norms, not a norm of differences, so it cannot
rank the magnitudes on this row. The comparator prints, for each of `x`, `logit`,
`exp` and `shr`, only the first call at which the axis stops being bit-identical,
and marks each line `[ORDERING ONLY]`:

```text
RESULT SELFWD MEASURE-CPU-vs-CUDA AXIS x     first_call_not_bit_identical=0 (forward=0) [ORDERING ONLY]
RESULT SELFWD MEASURE-CPU-vs-CUDA AXIS logit first_call_not_bit_identical=0 (forward=0) [ORDERING ONLY]
RESULT SELFWD MEASURE-CPU-vs-CUDA AXIS exp   first_call_not_bit_identical=0 (forward=0) [ORDERING ONLY]
RESULT SELFWD MEASURE-CPU-vs-CUDA AXIS shr   first_call_not_bit_identical=0 (forward=0) [ORDERING ONLY]
```

All four axes stop being bit-identical at call 0, which is the layer-0 MoE block of
the prefill. That ordering is consistent with the pair identity in §5, where the two
arms' `x` differs at call 0 while the selection does not. **No magnitude is claimed
from it.**

## 8. What this document does NOT establish

- **A cause for the three disagreeing token ids.** §4.2 is the reason: selection
  flipping is pervasive on both populations and its rate inverts against the token
  verdict at the clearest comparison in the run. Owed under
  [#2999](https://github.com/mudler/vllm.cpp/issues/2999).
- **That any flip reaches the sampled logit.** The tap reads which experts were
  chosen. It does not read whether the choice moved the value `argmax` reads.
  Separating those needs a tap on the routed output or the final logit row at the
  disagreeing forwards, on the same matched pair, and it needs the artifact and a
  GPU. That is the next traceable step, and it is named in #2999.
- **Anything about the other 47 layers' selection at the matched pair's prefill.**
  §5's verdict is layer 0 only, which is the layer #2552 bracketed.
- **Any magnitude, in either direction.** §7.
- **A token gate.** No oracle decoded this prompt. vLLM cannot run this GGUF
  artifact, and the llama.cpp arm aborts in `build_delta_net_chunking` before
  loading a byte. Both arms are ours.
- **A speed number.** The wall times in the log are liveness on n=1 legs. They are
  also not comparable across arms: A-CPU answered in 5 s, C-CPU2 in 6 s and
  B-CUDA in 112 s, and the job's own heartbeat records a load average of 6.15
  during the B-CUDA request (`### hb 02:17:40 load=6.15 5.57 5.55`). Nothing here
  is a device comparison.

## 9. Reproduction from committed data

The per-arm 384-line digest subsets are committed. They reproduce the log's
hash-mismatch column and the negative control with no share, no device and no
lease. The `moesel` value lines (576 per arm) are **not** committed, so the
prefill's token-level flip count of 53 is not reproducible from this repository.
The comparator's own output for both comparisons is committed as
[`cmp-measure.txt`](qwen4exp-moe-selection-fwd-20260906/cmp-measure.txt) and
[`cmp-control.txt`](qwen4exp-moe-selection-fwd-20260906/cmp-control.txt), which
carry the same per-forward verdicts the job also wrote as `measure.json` and
`ctrl.json`. **Those two JSON files are deliberately not committed, and the reason
is duplication rather than any checker.** Every one of `cmp-measure.txt`'s 74 lines
and `cmp-control.txt`'s 42 lines is present verbatim in the committed log, so the
two JSONs would restate verdicts this directory already carries twice. No checker
would have refused them: `scripts/check-pr-size.py`'s `DOC` pattern is
`docs/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:md|png|svg|json)\Z`, `json` is in
that allowlist, and `classify_path` returns `public_document` for
`docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906/measure.json` when it is
executed. **Do not read `.py`'s behaviour onto `.json` here.** `.py` genuinely is
absent from `DOC` and does raise `unclassified repository path`, which is #2629's
gap and this row's near neighbour; `.json` is in the allowlist and does not. The
two extensions are not the same case, and the fail-closed precedent belongs only
to the first.

```sh
python3 - <<'EOF'
d = 'docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906/digests-%s.txt'
def load(tag):
    out = {}
    for ln in open(d % tag):
        f = dict(t.split('=', 1) for t in ln.split() if '=' in t)
        out[int(f['call'])] = (f['sel'], int(f['T']))
    return out
A, B, C = load('A-CPU'), load('B-CUDA'), load('C-CPU2')
assert set(A) == set(B) == set(C) == set(range(384))
assert sorted(c for c in A if A[c][1] > 1) == list(range(48))       # prefill
assert sorted(c for c in A if A[c][1] == 1) == list(range(48, 384))  # decode
mism = lambda O: [sum(A[c][0] != O[c][0] for c in range(48*f, 48*f+48)) for f in range(8)]
print('A-vs-B', mism(B))   # [35, 22, 29, 17, 33, 48, 46, 48]
print('A-vs-C', mism(C))   # [0, 0, 0, 0, 0, 0, 0, 0]
EOF
```

Both lines reproduce the log's own `hash_mismatch` column exactly.

**Git confirms the negative control by content address.** `digests-A-CPU.txt` and
`digests-C-CPU2.txt` are stored under one blob, `4678dd52cdeba66ef6c3d206e1aefea40b594265`,
because the two files are byte-identical. Arm C is arm A run again, and its 384
digests came back the same bytes. That is the same zero the comparator reports,
arrived at without running the comparator. For the seven
decode forwards, where each call carries one token, `hash_mismatch` **is** the
flipped-slot count, so forwards 1 to 7 of §4's table are reproducible from this
repository alone.

## 10. Provenance of the instrument

The job script and comparator are unchanged from wave MOESEL and are committed at
[`qwen4exp-moe-selection-forward4-20260905/`](qwen4exp-moe-selection-forward4-20260905/).
Both were verified byte-identical to the copies the lease executed:

| File | sha256 |
|---|---|
| `job.sh` | `65c2f40ef3096a0f275bfa36b2e806b74b446ecd51ae0cccb26a89cb427771b9` |
| `selfwd.py` | `33cb0a7b1d09aff9ed163161c9e2776a8fcbc456fec9f0439b97d9966eb84e1a` |

`selfwd.py` was qualified against committed MOEDIV output before it was pointed at
this run, where it reproduces both the `L00 mhc.mix` pair 3613.82031 against
3615.62777 and #2552's recorded flip at token 2. That qualification is recorded in
[the MOESEL evidence file](qwen4exp-moe-selection-forward4-20260905.md) §3, and it
is what licenses the reading above.

**The fleet share is scratch, not a record surface.** The job's raw output at
`/workspace/moesel-2552-fwd4/out/` on `thor` (`/mnt/nas_share/rc/moesel-2552-fwd4/out/`
from a host that mounts it) holds the per-arm `sel.txt` (960 lines each) and each
arm's `server.log`. Nothing in this document depends on those files surviving.
