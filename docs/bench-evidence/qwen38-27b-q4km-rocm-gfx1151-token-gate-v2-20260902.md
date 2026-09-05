# Qwen3.8-27B Q4_K_M on ROCm `gfx1151` vs llama.cpp `b10451`, second attempt

The first run of this arm's declared token gate that the board could complete.
Issue [#2546](https://github.com/mudler/vllm.cpp/issues/2546), row
`BACKEND-ROCM`, spec
[`rocm-gfx1151-q4k-token-gate-v2.md`](../../.agents/specs/rocm-gfx1151-q4k-token-gate-v2.md).
Predecessor:
[`qwen38-27b-q4km-rocm-gfx1151-token-gate-20260902.md`](qwen38-27b-q4km-rocm-gfx1151-token-gate-20260902.md),
which returned `TOKEN_GATE=NOT_MEASURABLE` over 17 GPU resets in 18 legs.

It mirrors the CPU tier's run of the same gate deliberately: the same six
prompts byte for byte (`prompts_sha256`
`c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`), the same 48
tokens, greedy, batch 1, MTP off on both sides, the same artifact, the same
oracle pin, the same comparison scripts. The comparison between the tiers was
the point of the exercise and it is what this document mostly contains.

**No speed, latency or memory figure appears below.** `AGENTS.md` §Gates admits a
performance result from an arm only after that arm's declared token gate passes,
and [#2497](https://github.com/mudler/vllm.cpp/issues/2497) has already had one
measurement retracted for exactly that. The gate does not pass. Throughput
blocks were incidentally printed by the harness and are deliberately not
transcribed.

## Disposition

**`TOKEN_GATE=FAIL` on `gfx1151`. Tokenizer 6 of 6 exact, generation 3 of 6
divergent. 6 of 6 legs completed, 0 board faults, and all six legs are
byte-identical.**

Three results, in the order they matter.

1. **The board completes this workload now.** 6 clean legs out of 6, zero GPU
   resets, on the shipped default with no environment knob set. The predecessor
   measured 17 faults in 18 legs of the same workload. `27da7787e` (#2511) is
   what changed.
2. **The arm is scoreable and it fails, at 3 of 6.** Every one of the three
   losses is a rank-2 near-tie against the oracle: over 288 decode steps our
   token was the oracle's rank-1 on 285 and its rank-2 on 3, and nothing worse
   than rank 2 occurs anywhere.
3. **The ROCm tier does NOT diverge where the CPU tier diverges.** Same rate,
   3 of 6 on each side, and a **disjoint** set of first divergences: ROCm loses
   prompts 1, 3 and 5 while the post-#2534 CPU tier loses prompts 1, 2 and 4,
   and even the shared prompt is a different step (45 against 34).
   `TIER_DIVERGENCE_SAME_INDEX=0`. On three prompts our two tiers emit
   **different tokens on the same prefix**, so the ROCm arm carries a term of its
   own that nobody has scoped.

## What ran

`rc` job `85e4091e-1042-4225-9a92-6797449fddf9` on `strix:gpu0`, worker
`rc-worker-lcjhd`, boot id `a5bc8128-f6ad-4767-8614-6923f88032e1`, x86-64,
32 cores, 2026-09-02. Nothing reached the box by `ssh`. One lease, one build,
one boot. Raw logs on the share under
`/mnt/nas_share/rc/rocm-tokgate-strix-v2/out/rc-worker-lcjhd-20260902T155343Z/`.

The harness is the predecessor's, reused rather than rewritten, with the build
phase replaced and `HSA_ENABLE_SDMA=0` removed. Its six earlier defects and the
assertions each of them left behind are documented in the predecessor evidence;
none of them fired here.

## Measured identity

Asserted inside the job, which fails on a mismatch rather than reporting one.

```text
gguf_sha256              = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
gguf_size                = 17106775008                     (verified ON the worker)
prompts_sha256           = c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e
llama_pin                = 10bf611e533d81f739128304991c5e133c6aebd8  (tag b10451)
llama_src_files          = 3425
llama_src_manifest       = 56c26d15c2acf11b8621ac26663b4316dc29719d765ba1d95231ffacaddf3cda
llama_src_manifest_after_harness = 56c26d15... (unchanged; the harness built out of tree)
vllmcpp_source_revision  = 6a859c170a44b18ebb78dc0f27af46df7b20d54c
product_tree_vs_27da7787e = 0 differing files under src include examples CMakeLists.txt cmake tests
libvllm.so               = 2c3ac67eae458763a2c282e1fc601b60d46d408296a7d56955a36d876c9b0b65
vllm-cli                 = 1d442559b73aeb6763a7722ebee30a5303c64c74cf6928761ecca62c502453f8
vllm-bench               = d7292e3895fb4887149ec3c5902cc08fa080f88e0f8851d61848753466b6501d
tokenize                 = 8673794406f1ef3a6239941c0a48a52e3d68a23b78303ac4e7b07335b589ce77
llama-completion         = 34435957ed321e3fa9e660c2737ed16fe1830a2d5ef4c0fb3c01343c4f7a6bc6
oracle-tokens            = 985ffd646c3a658f558d96108497d625b08512724f038fb47d305a79a8a0cee3
oracle-margin            = ef369639041570383b2333acbf3301ee6269eadafdf7f0dc83ed7a66f090967a
```

The source manifest is computed under `LC_ALL=C` and the run prints the
collation beside the value, because `sort` collates by locale and the
predecessor's first submission refused a correct tree over exactly that.

### The two landed fixes are proven present, not assumed

`vllm-cli` is a ~26 KB thin client and the kernels are in `libvllm.so`, so the
library is the identity that matters. Four assertions, each of which fails the
job:

| # | assertion | result |
|---|---|---|
| 1 | the staged clone is the named object | `6a859c170` = expected |
| 2 | the branch changes no product code relative to `27da7787e` | 0 differing files; the only addition is this row's spec |
| 3 | `#2534`'s routing line is in the source, `#2511`'s narrowing is in the source | `SOURCE_FIX_2534=PRESENT`, `SOURCE_FIX_2511=PRESENT` |
| 4 | the build is CLEAN (`rm -rf` first) and the artifact moved | `build_rc=0`; `libvllm.so` `2c3ac67e…`, not the predecessor's `41bb4052…` |

The `#2511` fix additionally carries a string literal that `11fed3ba5` cannot
contain, so it is grepped for in the binary, **and the marker is shown to
discriminate rather than merely to be found**: the pre-fix library still on the
worker reports `ABSENT` for the same grep.

```text
BINARY_FIX_2511=PRESENT in the new libvllm.so
control_marker_in_the_OLD_library=ABSENT   -- the marker discriminates the two builds
old_libvllm 7604aca84258fd836206ab23585d78876817b50138dcc2b2b8258f6af5a4b7ff
```

One honest wrinkle, recorded rather than smoothed: the old library **on the
worker** is `7604aca8…`, not the predecessor's `41bb4052…`. That is #2511's own
knob rebuild, which overwrote the tree the predecessor measured. The `!=
41bb4052…` assertion was therefore checked against the recorded value rather
than against a file, and the `ABSENT` contrast was taken on `7604aca8…`. Both
are pre-narrowing libraries and both lack the marker.

`#2534` has no string literal, so no grep can witness it. Assertions 1, 2 and 3
are what carries it: a clean build of an asserted tree that contains the routing
line necessarily compiles the routing line.

## The board: 6 clean legs of 6, zero faults

Six independent legs, each its own process and its own model load, on the
shipped default configuration with `VT_OP_PROVIDER_STATS=1` and **nothing else**
set. N is 6 by design. The tally below is derived from deduplicated `MLEG`
lines, not from a summary line.

```text
MLEG 1 rc=0 status=OK rocm_device_lines=20 reference_tier_hits=0 fault_lines=0
MLEG 2 rc=0 status=OK rocm_device_lines=20 reference_tier_hits=0 fault_lines=0
MLEG 3 rc=0 status=OK rocm_device_lines=20 reference_tier_hits=0 fault_lines=0
MLEG 4 rc=0 status=OK rocm_device_lines=20 reference_tier_hits=0 fault_lines=0
MLEG 5 rc=0 status=OK rocm_device_lines=20 reference_tier_hits=0 fault_lines=0
MLEG 6 rc=0 status=OK rocm_device_lines=20 reference_tier_hits=0 fault_lines=0

OUR_LEGS_OK=6 OUR_LEGS_BOARD_FAULT=0 OUR_LEGS_HARNESS_ERROR=0 OUR_LEGS_TOTAL=6
SELF_REPRODUCIBLE=YES across legs [1, 2, 3, 4, 5, 6]
```

6 unique `MLEG` lines for a 6-leg design, 6 `status=OK`, and **zero occurrences
of `GPU Hang`, `Memory access fault` or `HW Exception` anywhere in the job log**.

| run | workload | legs | board faults |
|---|---|---:|---:|
| predecessor, `11fed3ba5`, `HSA_ENABLE_SDMA=0` on every leg | this gate | 18 | **17** |
| this run, `27da7787e`, no knobs at all | this gate | 6 | **0** |

`HSA_ENABLE_SDMA` is **not set anywhere in this job** and the job asserts its own
inherited environment is empty of `HSA_*` and `VT_*` before it starts. It is
retired: it halved the fault rate on a single prefill, did nothing at gate size,
and the allocator was the cause. It is not a workaround and must not be
described as one.

**The native ROCm path is fully engaged.** Every leg reported
`reference_tier_hits=0` with 20 `device=5` (`kROCM`) op selections, and the job
fails on a non-zero reference-tier count. This matters more than it did before:
`27da7787e` withdraws the host-addressability claim on this part, so the CPU
reference tier's eligibility moved, and the assertion is measuring a changed
thing rather than restating an old one.

**All six legs are byte-identical**, verified independently off the six
`ours_gen_ids_*.json` artifacts rather than from the job's own verdict line. The
two-agreeing-legs precondition is satisfied six times over.

## The oracle's executed kernel path, pinned

Recorded per leg rather than inferred, because `b10451`'s greedy decode is not
deterministic across its own kernel paths.

```text
ORACLE_N_GPU_LAYERS     99   (the HIP arm on gfx1151)
ORACLE_USE_EXTRA_BUFTS  1
ORACLE_SYSTEM_INFO      ROCm : NO_VMM = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 |
                        AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 |
                        AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 |
                        AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1
ORACLE_DEVICE 0         ROCm0  Radeon 8060S Graphics
ORACLE_DEVICE 1         CPU    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
ORACLE_N_VOCAB          248320
ORACLE_ADD_BOS          0
ORACLE_EOS              248046
host arch               x86-64
```

**`ORACLE_REPRO=YES`.** Today's oracle leg reproduces the predecessor's recorded
`GEN_IDS` on **6 of 6 prompts, identically, 48 ids each**. The predecessor could
not make this check because it had nothing to compare against. The denominator
did not move between the two runs, so the change in the verdict is entirely on
our side.

`CHAIN_OF_CUSTODY=EXACT`: the harness's detokenized prompt-plus-generation for
prompt 0 is byte-identical (225 bytes each) to the stock `llama-completion`
binary's stdout on the same recipe, from the same build, in the same lease.
`llama-completion` is the binary the 2026-08-23 run used; `llama-cli` at this pin
applies a chat template and cannot reproduce a raw-completion recipe.

The degenerate `n_gpu_layers = 0` oracle leg was **not run**. It is a failed
instrument owed by [#2557](https://github.com/mudler/vllm.cpp/issues/2557), and
running it again would have spent lease time to re-observe a known defect.
`compare_tiers.py` was therefore fed the HIP oracle in both of its oracle slots,
so its `A vs C` column restates `A vs B` and **no conclusion is drawn from it**.

## The `blk.64` asymmetry, re-observed

`b10451` ignores all 15 `blk.64.*` tensors, four of them the `nextn.*` MTP head.
The stock control run emitted exactly **15** `unused tensor blk.64.*` warnings —
re-observed on the HIP path here rather than cited. Our arm runs with MTP off,
so both engines decode the same 851 tensors and the comparison is matched *work*
and not only matched weights.

## Tokenizer: EXACT, 6 of 6

`examples/tokenize`, reading the GGUF's own vocab, reproduced the oracle's
`PROMPT_IDS` line for line at lengths 6, 5, 6, 7, 11, 7.
`TOKENIZER_DIVERGENCES=0/6`.

## Generation: 3 of 6 divergent, every loss a rank-2 near-tie

Scored leg 1, one of six agreeing clean legs.

| prompt | verdict | first diff | ours | oracle | oracle gap at that step |
|---|---|---:|---:|---:|---:|
| 0 `The capital city of France is` | **TOKEN-EXACT 48/48** | — | — | — | — |
| 1 `The three primary colors are` | DIVERGE | 45 | 303 | 1521 | 0.131054 |
| 2 `Water boils at a temperature of` | **TOKEN-EXACT 48/48** | — | — | — | — |
| 3 `The Pythagorean theorem states that` | DIVERGE | 45 | 25 | 393 | 0.006284 |
| 4 `In 1969, humans first walked on` | **TOKEN-EXACT 48/48** | — | — | — | — |
| 5 `A prime number is a natural number` | DIVERGE | 32 | 16 | 15 | 0.053452 |

```text
TOKENIZER_DIVERGENCES=0/6
GENERATION_DIVERGENCES=3/6
TOKEN_GATE=FAIL
```

Teacher-forced along our own tokens, the oracle's rank of our token over all
288 decode steps:

```text
285 steps  rank 1
  3 steps  rank 2
```

Nothing worse than rank 2 occurs. The three losses are the three rank-2 rows,
and each is a first divergence.

### The near-tie step SET, which carries more than the first-diff column

Nineteen of the 288 steps have an oracle `top1 - top2` gap below 0.20. Our arm
takes the oracle's top-1 on **16 of those 19** and loses 3.

Every gap in this table is **this run's HIP oracle**, teacher-forced along our own
ROCm tokens. They are therefore comparable with each other and are NOT the same
quantity as the recorded aarch64 margins (0.085434, 0.115482, 0.178236) the CPU
tier's evidence quotes at its own three steps: those come from the other kernel
path, which is the very thing this oracle is not deterministic across.

| step | gap | our rank |
|---|---:|---|
| p0/7 | 0.052456 | 1 |
| p1/27 | 0.127657 | 1 |
| **p1/34** | 0.059769 | 1 |
| p1/35 | 0.076401 | 1 |
| p1/36 | 0.067351 | 1 |
| **p1/45** | 0.131054 | **2** |
| p2/4 | 0.058796 | 1 |
| p2/9 | 0.050880 | 1 |
| **p2/20** | 0.149452 | 1 |
| p2/29 | 0.023709 | 1 |
| p2/39 | 0.192465 | 1 |
| **p3/45** | 0.006284 | **2** |
| p4/9 | 0.163641 | 1 |
| p4/13 | 0.134695 | 1 |
| **p4/14** | 0.092752 | 1 |
| p4/16 | 0.190178 | 1 |
| p4/17 | 0.129532 | 1 |
| **p5/32** | 0.053452 | **2** |
| p5/38 | 0.194795 | 1 |

The bolded rows are the six steps either tier is convicted at. **The ROCm arm
wins all three of the CPU tier's contested steps and loses three others in the
same population.** Its worst loss, 0.131054, is larger than two of the three
gaps the CPU tier loses, so this is not a smaller error landing elsewhere; it is
a comparably sized error landing elsewhere.

## THE COMPARISON: ROCm's divergences are not the CPU tier's

The CPU tier's comparison point is its **post-#2534** result, reproduced from
`gen_ids_F32.json` of rc job `c0b3fc6d-…` on `thor:gpu0`, scored against the
recorded aarch64 oracle ids: 3 of 6, prompts 1 at 34, 2 at 20 and 4 at 14, with
0, 3 and 5 token-exact. The retired 5-of-6 figure predates #2534 and comparing
against it would confound the two fixes.

```text
prompt A vs B (ROCm/HIP oracle)   D vs E (CPU tier/aarch64 oracle)   A vs D (ours: ROCm vs CPU)
0      EXACT                      EXACT                              EXACT
1      k=45  303/1521             k=34  198/3095                     k=45  303/1521
2      EXACT                      k=20  13/539                       EXACT
3      k=45  25/393               EXACT                              k=45  25/393
4      EXACT                      k=14  4593/22486                   k=14  22486/4593
5      k=32  16/15                EXACT                              EXACT

TIER_DIVERGENCE_SAME_INDEX=0
TIER_DIVERGENCE_SAME_INDEX_AND_IDS=0
TIER_BOTH_EXACT=1
TIER_ONE_SIDED=4
ROCM_IDS_EQUAL_CPU_IDS=3/6
```

**Same rate, disjoint divergences.** The spec fixed the reporting rule for this
outcome before the data, and the honest label is the plain one: 3 of 6 on each
side, one prompt in common, and that one at a different step. A moved index is
what this class of perturbation does, so the index alone would prove nothing.
What decides it is the direct `A vs D` column, which involves **no oracle at
all**: our own two tiers emit different tokens on the same prefix at three
steps.

Resolving each of the six contested steps to what all four sides emit separates
two effects that the first-diff column mixes:

| step | ROCm | CPU tier | HIP oracle | aarch64 oracle | what it is |
|---|---:|---:|---:|---:|---|
| p1/34 | 198 | 198 | 198 | 3095 | our two tiers AGREE; the two oracle paths disagree |
| p2/20 | 13 | 13 | 13 | 539 | our two tiers AGREE; the two oracle paths disagree |
| p5/32 | 16 | 16 | 15 | 16 | our two tiers AGREE; the two oracle paths disagree |
| p1/45 | **303** | **1521** | 1521 | 3095 | our two tiers DIFFER; ROCm is wrong |
| p3/45 | **25** | **393** | 393 | 393 | our two tiers DIFFER; ROCm is wrong |
| p4/14 | **22486** | **4593** | 22486 | 22486 | our two tiers DIFFER; ROCm is right |

Three of the six are the **oracle** disagreeing with itself across its own kernel
paths, at exactly the three prompts the predecessor already measured that on
(1, 2, 5). At those three steps our ROCm and CPU arms emit the same token, and
only the choice of denominator decides who is convicted. **p5/32 is the cleanest
case: both our tiers emit 16, the aarch64 oracle emits 16, the HIP oracle emits
15, so the ROCm tier is convicted there for the oracle's own kernel-path
sensitivity and not for anything our arm did differently.**

The other three are ours. At p1/45, p3/45 and p4/14 the ROCm and CPU arms of
vllm.cpp compute a different argmax over an identical prefix on an identical
artifact. That is a **ROCm-local numerics term**, and this is the first
measurement of it. It is not one-signed: ROCm loses two of those steps and wins
the third, and the one it wins (p4/14) is a step the CPU tier is convicted at.

**Consequence for ownership.** #2534's residual term does not explain the ROCm
result. Two of the three prompts ROCm fails (3 and 5) are prompts the CPU tier
passes, and the third fails at a different step. A separate issue against
`BACKEND-ROCM` owes the three-step ROCm-local term.

## What is not admissible

- **No speed, latency or memory axis.** The gate has not passed. Nothing in this
  document may be quoted for one, and #2497 stays blocked.
- **The `ngl=0` oracle leg** was not run; the `A vs C` column restates `A vs B`.
- **Any claim that the ROCm arm is more accurate than the CPU tier.** It wins
  three near-ties and loses three others of comparable size; 3 of 6 against 3 of
  6 is a tie in rate, measured against denominators that are not the same
  kernel path.
- **Any claim that `HSA_ENABLE_SDMA=0` does anything.** It was not set. The
  allocator was the cause.
- **The three oracle-self-disagreement steps as evidence about our arm.** They
  are evidence about the oracle.

## The next traceable hypothesis

No ceiling is declared.

1. **The ROCm-local term at p1/45, p3/45 and p4/14.** Three steps where our two
   tiers disagree on an identical prefix, all inside the near-tie population.
   The discriminating experiment is to run the CPU reference tier and the ROCm
   tier on the same host over the same prefix and dump the pre-sampler logits at
   those three steps, which localises the term to a layer rather than to a
   device. Note that `27da7787e` withdraws host addressability on this part, so
   the reference tier is no longer reachable in-process here and the comparison
   needs a deliberate arm.
2. **The gate's own definition still needs the executed oracle path.** Half of
   the six contested steps across the two tiers are the oracle disagreeing with
   itself. A 6-of-6 token-exactness demand at 0.006-to-0.19 gaps is partly a
   demand to bit-reproduce one kernel schedule. This run supplies a third
   measurement of that effect, now with `ORACLE_REPRO=YES` proving the effect is
   reproducible rather than noise.
3. **#2534's residual magnitude term still owns the CPU tier's three.** It does
   not own ROCm's.
