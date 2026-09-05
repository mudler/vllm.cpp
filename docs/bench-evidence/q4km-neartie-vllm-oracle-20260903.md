# The ratified near-tie band, scored against the PRIMARY oracle

Issue [#2809](https://github.com/mudler/vllm.cpp/issues/2809), row
`QUANT-Q4K-NEARTIE-VLLM-ORACLE`, spec
[`qwen38-27b-q4km-neartie-vllm-oracle.md`](../../.agents/specs/qwen38-27b-q4km-neartie-vllm-oracle.md).

**No throughput, latency or memory figure is taken for vllm.cpp anywhere below,
and no cross-engine ratio is computed.** `AGENTS.md` §Gates admits a performance
result from an arm only after that arm's declared token gate passes, and the
Qwen3.8-27B Q4_K_M ROCm arm's reads `FAIL` at 3 of 6
([token-gate v2](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)).
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) has already had one
measurement retracted for exactly this.

**Nothing here is a ratification, and nothing here recommends one.** §8 of
[`qwen38-27b-q4km-neartie-band-adjudication.md`](../../.agents/specs/qwen38-27b-q4km-neartie-band-adjudication.md)
records that whether the ratified band may be applied to this arm at all is the
maintainer's decision under
[#2534](https://github.com/mudler/vllm.cpp/issues/2534). This document is an
input to that decision and takes it no further. No verdict of #2497, #2534,
[#2546](https://github.com/mudler/vllm.cpp/issues/2546) or
[#2740](https://github.com/mudler/vllm.cpp/issues/2740) is rescored, and no
gate's declared oracle changes: this arm's token gate remains gated against
llama.cpp `b10451` and remains `FAIL`.

## Why this could not be measured before

The adjudication spec scored the four conjuncts of the ratified band on this arm
against **llama.cpp `b10451`**, a secondary oracle, and §7 says why it had no
choice:

> **No harness exists either.** The ratified instrument reads vLLM
> `prompt_logprobs`.

[#2788](https://github.com/mudler/vllm.cpp/pull/2788) removed the constraint. The
pinned vLLM `5559679229` builds and runs on `strix:gpu0`, loads the gated
`Qwen3.8-27B-Q4_K_M.gguf` through `vllm-gguf-plugin`, and generates 48 greedy
tokens for all six gate prompts, twice per configuration
([evidence](oracle-vllm-gfx1151-20260903.md)). So the instrument can now be
pointed at the oracle it was written for.

#2788 also measured that **the oracle's own greedy decode differs between its two
supported configurations** on this workload: eager and compiled disagree on 2 of
6 prompts, with each configuration byte-identical to its own repeat.
`AGENTS.md` §Gates admits an explicitly ratified distributional gate exactly when
the oracle's greedy decode is non-deterministic, and that precondition is now
measured for the primary oracle rather than only for llama.cpp.

## What ran

`rc` job `e1afb349-98c4-4e8b-a684-26fdeaa4ba24` on `strix:gpu0`, worker
`rc-worker-lcjhd`, boot id `a5bc8128-f6ad-4767-8614-6923f88032e1`, x86-64, 32
cores, 2026-09-03 18:17:37Z to 19:15:27Z. One lease, one build, one boot.
Nothing reached the box by `ssh`. The verbatim job log is
[`job.txt`](q4km-neartie-vllm-oracle-20260903/job.txt), the build output is
[`build.log.gz`](q4km-neartie-vllm-oracle-20260903/build.log.gz), and the
harness's own stdout for each configuration is beside them.

```text
HSA_OVERRIDE_GFX_VERSION = UNSET                   asserted by the job AND again in-process
INHERITED_DEVICE_ENV     = ['PYTORCH_ROCM_ARCH']   see the wrinkle below
DEVICE                   = gfx1151                 read from torch, asserted, not assumed
VLLM_VERSION             = 0.26.0.dev0+g5559679229 the pinned primary oracle
TORCH                    = 2.13.0+rocm7.2, hip 7.2.53211
ON_GFX1151 = True, ON_GFX1X = True                 vllm.platforms.rocm agrees with the board
vllm-pin.tar.gz          sha256 7d8bd182…3a98e5,  6110 files
src_manifest (LC_ALL=C)  b037645415bb07eccbf9b69f6a8d69b51b14131bac3f1e93a143961af0f46483
plugin wheel             sha256 6b7e0f4f…58821,  _C_gguf.abi3.so 4417184250995733a61d352146e36062d60e4a8c44581eb2631fbcc48600f0de
plugin device code       1 bundle, amdgcn-amd-amdhsa--gfx1151, and no other arch
GGUF_BYTES               17106775008 = EXPECTED    verified ON the worker
minMemAvailable          54371 MB over 949 samples
```

**One inherited variable, recorded rather than smoothed.** `PYTORCH_ROCM_ARCH`
is set to `gfx1151` by the build phase and is still in the environment when the
harness runs. It is a build-time architecture list, it names the architecture
the board actually is, and `torch` independently reported `gfx1151` from the
device. It is not `HSA_OVERRIDE_GFX_VERSION`, which was `UNSET` and is asserted
absent twice. The variable is reported because the harness prints every
`HSA_*`, `ROCR_*`, `PYTORCH_*` and `HIP_*` it inherits, and a report that named
none of them would be the thing worth doubting.

**The GGUF's sha256 was not recomputed in this job.** Its size was verified on
the worker and matches the byte count the denominator run pinned
(17,106,775,008). It is the same `/workspace/ckpt` artifact #2788 and the v2
token gate used, whose sha256 is `7e78da5d…fe169`. Size is a weaker check than
a hash and this record says so rather than implying a hash was taken.

## The four conjuncts, both configurations

`result == PASS && n_divergent == 0 && over_band_failures == 0 && worst_gap <= 0.5`,
with the definitions of §3.1 of the spec. `result` is the band-only field and is
**never** the verdict on its own; the `FOUR CONJUNCTS` column is.

### Oracle: vLLM `5559679229`, compiled (`enforce_eager=False`)

| stream | `result` | `n_divergent` | over-band | `worst_gap` | token mismatches | exact ties | **FOUR CONJUNCTS** |
|---|---|---:|---:|---:|---:|---:|---|
| **our ROCm arm** | PASS | **0 of 288** | 0 | **0.000000** | **0** | 0 | **PASS** |
| the oracle's own decode (`self`) | PASS | 3 of 288 | 0 | 0.125000 | 3 | 0 | FAIL |
| llama.cpp `b10451` | PASS | 1 of 288 | 0 | 0.125000 | 1 | 0 | FAIL |
| corrupted stream (control) | **FAIL** | 1 of 48 | 1 | 21.242187 | 1 | 0 | FAIL |

### Oracle: vLLM `5559679229`, eager (`enforce_eager=True`)

| stream | `result` | `n_divergent` | over-band | `worst_gap` | token mismatches | exact ties | **FOUR CONJUNCTS** |
|---|---|---:|---:|---:|---:|---:|---|
| **our ROCm arm** | PASS | 4 of 288 | 0 | 0.250000 | 4 | 0 | **FAIL** |
| the oracle's own decode (`self`) | PASS | 6 of 288 | 0 | 0.125000 | 6 | 0 | FAIL |
| llama.cpp `b10451` | PASS | 3 of 288 | 0 | 0.125000 | 3 | 0 | FAIL |
| corrupted stream (control) | **FAIL** | 1 of 48 | 1 | 21.070312 | 1 | 0 | FAIL |

Every divergent step, in full:

```text
oracle compiled | ours       (none)
oracle compiled | self       p4/13 ours   6165 lp -3.238075  top  19820 lp -3.175575  gap 0.062500  rank 2
                             p4/38 ours     17 lp -0.921848  top     18 lp -0.796848  gap 0.125000  rank 2
                             p5/32 ours     15 lp -1.562733  top     16 lp -1.437733  gap 0.125000  rank 2
oracle compiled | llamacpp   p5/32 ours     15 lp -1.562733  top     16 lp -1.437733  gap 0.125000  rank 2

oracle eager    | ours       p1/36 ours     11 lp -0.909987  top    369 lp -0.784987  gap 0.125000  rank 2
                             p1/45 ours    303 lp -1.349277  top   1521 lp -1.099277  gap 0.250000  rank 3
                             p2/29 ours     20 lp -1.438583  top     16 lp -1.313583  gap 0.125000  rank 2
                             p3/45 ours     25 lp -1.374806  top    393 lp -1.249806  gap 0.125000  rank 2
oracle eager    | self       p1/35, p2/4, p3/45, p4/14, p4/28, p5/32   worst 0.125000, all rank 2
oracle eager    | llamacpp   p1/36, p2/29, p5/32                       worst 0.125000, all rank 2
```

**No divergence anywhere is an exact tie.** All 17 divergent steps across both
configurations and all three streams have a strictly positive gap. That is the
whole difference between this arm and the Voxtral precedent, which passed this
band at `worst_gap 0.0000` because its divergences were exact ties. It was true
of the llama.cpp adjudication and it is true of the primary oracle too.

## Read against the outcomes fixed in advance

Spec §3.6 enumerated six. **Four of them fire, and they are reported together
because any one of them alone misdescribes the result.**

- **P1 fires for compiled.** All four conjuncts hold for our arm against vLLM
  compiled. Under that configuration the band would `PASS`. It still needs the
  maintainer's ratification under #2534 before it may be quoted as anything.
- **P2 fires for eager.** `n_divergent = 4`, every gap inside the band. `FAIL`
  on the binding limb. The band-only `result` reads `PASS` there and is not the
  verdict.
- **P4 fires.** The two configurations disagree on the verdict, PASS against
  FAIL. §3.4 fixed in advance that neither is averaged into the other and that
  the choice between them is the maintainer's. This row does not make it.
- **P5 fires, in both configurations.** The self-consistency control is not
  zero. It is **3 of 288 under compiled and 6 of 288 under eager**, so no
  verdict above may be read without that floor beside it. See below.

P3 does not fire: no gap anywhere exceeds 0.5 nats, and the largest outside the
control is 0.250. P6 does not fire: the control discriminates.

## The instrument's own floor, and what the 0 of 288 therefore is

The self-consistency control teacher-forces each configuration on **its own**
recorded greedy tokens. The teacher-forced distribution comes from a prefill
over the whole sequence and the greedy tokens came from step-by-step decode, so
the two need not agree. They do not:

```text
oracle compiled, scoring vLLM compiled's OWN greedy decode   3 of 288 divergent
oracle eager,    scoring vLLM eager's OWN greedy decode      6 of 288 divergent
```

**Our arm's 0 of 288 under compiled is therefore below the instrument's own
floor.** Our recorded stream agrees with the compiled prefill argmax at every
one of the 288 positions, at three of which the oracle's own decode does not.
This is a fact about where two near-tie coin flips landed, not a claim that our
engine is more correct than vLLM. The mechanism is visible in the numbers: at
`p5/32` the compiled prefill puts token 16 ahead of token 15 by 0.125 nats; our
arm emitted 16, and vLLM's own compiled decode emitted 15, as did llama.cpp.

A gate whose passing threshold sits below the floor of the instrument that
scores it is a gate that can be passed by luck. **That is the single most
important limit on the PASS above** and it is the reason §3.5 required this
control at all.

## The number this is NOT, stated before anyone infers it

**This is not a token-exact result, and the arm's declared token gate does not
pass.** The two measure different things and they point in opposite directions.

The declared token gate compares **free-running** greedy streams: each engine
generates 48 tokens from the prompt and the streams are compared. The band above
is **teacher-forced**: the oracle is fed our own prefix at every step, so the two
sequences can never drift apart. Substituting the primary oracle into the
free-running comparison makes this arm's reading **worse**, not better, and
#2788 already published the figure:

```text
free-running, 6 prompts x 48 greedy tokens, first divergence per prompt

vllm.cpp ROCm arm  vs  llama.cpp b10451     3 of 6 prompts diverge   <- the DECLARED gate, FAIL
vllm.cpp ROCm arm  vs  vLLM compiled        5 of 6 prompts diverge
vllm.cpp ROCm arm  vs  vLLM eager           4 of 6 prompts diverge
vLLM compiled      vs  llama.cpp b10451     3 of 6 prompts diverge
vLLM compiled      vs  vLLM eager           2 of 6 prompts diverge
```

Source: [`score-compiled.md`](oracle-vllm-gfx1151-20260903/score-compiled.md)
(`VLLMCPP_vs_VLLM_DIVERGENCES=5/6`), re-derived independently here from the same
two committed streams by
[`verify.py`](q4km-neartie-vllm-oracle-20260903/verify.py).

So no reading of this document supports "the token gate now passes against the
primary oracle". **`TOKEN_GATE` for this arm stays declared against llama.cpp
`b10451` and stays `FAIL`**, and a reading against vLLM would be `FAIL` at 5 of
6. The `TOKEN_GATE=FAIL` in
[`qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)
is **not rescored** by this row. It stands exactly as measured, against the
secondary oracle it declares, and this document changes nothing about it.

**No throughput, latency, memory or ratio figure for vllm.cpp appears anywhere
in this document, and none may be taken from this arm.** `AGENTS.md` §Gates
requires the declared token-exact gate first, spec §2 excludes every such figure
by name, and spec §9's stop conditions repeat it. #2497 already carries one
retraction for exactly this.

## The oracle is not self-consistent, and the denominator choice does real work

vLLM's two supported configurations, both byte-identical to their own repeats,
disagree with each other:

```text
free-running:   compiled vs eager differ on 2 of 6 prompts (p3 from step 45, p4 from step 13)
teacher-forced: compiled scoring its own decode      3 of 288
                eager    scoring its own decode      6 of 288
```

Our arm's result flips with that choice: `PASS` against compiled, `FAIL` at 4 of
288 against eager. **The choice of configuration is therefore not a formality;
it decides the verdict.** `AGENTS.md` §Gates says a distributional gate is
admissible when the oracle's greedy decode is non-deterministic, and that
precondition is now measured for the primary oracle. It also says such a gate
must be **explicitly ratified**, which this one, for this arm, is not.

`AGENTS.md` §Gates separately forbids `--enforce-eager` as the denominator for a
**performance** comparison, and that rule is about throughput, where eager
disables the compiled path a production deployment would use. It does not
nominate compiled as the correctness denominator, and this document does not
read it as doing so. Both configurations are reported, neither is averaged into
the other, and the choice belongs to #2534.

## The instrument discriminates

One recorded step of prompt 0 was replaced by a token the oracle cannot prefer.
The harness refuses to emit its `DONE_MARKER` unless that registers a divergence
above 1 nat.

```text
oracle compiled  p0/10  ours 9999 lp -21.252218  top 13 lp -0.010032  gap 21.242187  rank 79792
oracle eager     p0/10  ours 9999 lp -21.081854  top 13 lp -0.011542  gap 21.070312  rank 78230

INSTRUMENT_CONTROL corrupt-step discriminates = PASS   (both configurations)
DONE_MARKER_Q4KM_NEARTIE_VLLM                          (both configurations)
```

An instrument that cannot fail cannot pass, and this one fails by four orders of
magnitude on the band it is asked to police.

## Chain of custody

The four input streams were sha256'd on the worker before the run, and every one
is byte-identical to the copy committed in this repository:

```text
ours_gen_ids_1.json        8b542c718fd38721d5dd3286a77c91ed30ab495b0c604783b9a2681fcc1ad107
tokens-gguf-compiled.json  034a1e303fc39ac72044e2d4bfa79b084cc10b2867b9ec701288c1ab2978de25
tokens-gguf-eager.json     350b5fae5de4a25a62fdc2a839eefbfc69e2538f1eadfaa463b60a4c5d39434a
oracle_hip.txt             6edd7e5c97648940d15fb57a214fe36a09010579ae9799f533af6b49ca262d11
prompts_sha256             c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e
```

Nothing was re-generated for this run. Our arm's stream is the recorded one, so
nothing about our engine could move under the measurement.

The two result files carry the sha256 the job printed in its own postcondition,
and the committed `.gz` decompresses to exactly those bytes:

```text
neartie-compiled.json  f2edef503eaef227751bb8ed519bc106ee61fd35578e9800866bb2372ad0c19c
neartie-eager.json     9b6fed75d24d8ee1cbef9557ae29ce25bc5d18518cb96f7c1601f7ac8cbb3a37
```

## Verification

Every figure in this document is recomputed from the per-step record by
[`verify.py`](q4km-neartie-vllm-oracle-20260903/verify.py), which shares no code
with the harness, reads the committed artifacts, and refuses to agree with the
harness's own `summary` until they match:

```console
$ python3 docs/bench-evidence/q4km-neartie-vllm-oracle-20260903/verify.py
CLAIMS_CHECKED = 126
MISMATCHES     = 0
```

It checks three things a recount alone would not: that each scored stream, rebuilt
from its per-step `our_tok` fields, byte-equals the committed artifact it claims
to be; that the negative control fails; and that the published v2 llama.cpp
table re-derives from the same two committed streams.

**It was mutated to prove it can fail.** Four mutations in a scratch copy, the
tree restored byte-for-byte after each:

| mutation | detected by |
|---|---|
| shift one `top_lp` in `compiled/ours` so a step becomes divergent | the `gap == top_lp - our_lp` assertion |
| change one token of the scored `ours` stream | the stream-identity check against `ours_gen_ids_1.json` |
| neutralise the corrupted step so the control passes | six control checks, including `four_conjuncts` PASS where FAIL is required |
| set the harness's own `summary.n_divergent` to 7 | the recount disagreeing with the summary |

All four exited 1. The restored tree exits 0 at 126 of 126.

## What this row does not do

- It does not ratify anything and does not recommend a ratification. #2534 is
  the maintainer's.
- It does not rescore or alter #2497, #2534, #2546 or #2740.
- It changes no gate's declared oracle. This arm's token gate stays llama.cpp
  `b10451` and stays `FAIL`.
- It takes no performance figure, and none may be taken from this arm.
- It touches no `src/` or `include/`.
