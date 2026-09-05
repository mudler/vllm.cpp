# Qwen3.8-27B Q4_K_M on ROCm `gfx1151` vs llama.cpp `b10451`

The first attempt to run this arm's declared token gate on the board. Issue
[#2546](https://github.com/mudler/vllm.cpp/issues/2546), row `BACKEND-ROCM`,
spec [`rocm-gfx1151-q4k-token-gate.md`](../../.agents/specs/rocm-gfx1151-q4k-token-gate.md).

It mirrors the CPU tier's run of the same gate,
[`qwen38-27b-q4km-token-gate-20260823.md`](qwen38-27b-q4km-token-gate-20260823.md),
deliberately: the same six prompts byte for byte (`prompts_sha256`
`c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`), the same 48
tokens, greedy, batch 1, MTP off on both sides, the same artifact, the same
oracle pin, the same comparison script. The comparison between the two tiers was
the point of the exercise, and it is what this document mostly contains.

**No speed, latency or memory figure appears below.** `AGENTS.md` §Gates admits a
performance result from an arm only after that arm's declared token gate passes,
and [#2497](https://github.com/mudler/vllm.cpp/issues/2497) has already had one
measurement retracted for exactly that. Throughput blocks were incidentally
printed by the harness and are deliberately not transcribed here.

## Disposition

**`TOKEN_GATE=NOT_MEASURABLE` on `gfx1151`. 17 of 18 legs of this workload ended
in a GPU reset, including 12 of 12 in a controlled two-arm run.** The reason is
the board, not the numerics: no leg produced token ids, so there is nothing to
score and no divergence to attribute.

That is the headline, and it is a result rather than a failure to deliver one.
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) established that this
board faults on a **single prefill**. This establishes that it **cannot complete
a gate-sized workload at all** — the failure does not average out over a
realistic run, it compounds. Nobody had measured that.

A second result came out of the oracle side on the way, and it changes how the
CPU tier's recorded `FAIL` should be read: **the oracle disagrees with itself on
3 of these 6 prompts**, and each disagreement lands on a step the CPU gate
convicted us at.

## Finding 1: the oracle disagrees with ITSELF on 3 of these 6 prompts

`.agents/oracles/llama-cpp.md` already records that `b10451`'s greedy decode is
not deterministic across its own supported kernel paths, measured at **1 of 6**
prompts under the stock `-nr/--no-repack` flag. Changing the kernel path further
— from aarch64 CPU kernels to this board's own gfx1151 HIP kernels, one
artifact, one recipe, one pin, greedy on both — moves **3 of 6**:

```text
ORACLE_SELF_DIVERGENCES=3/6
prompt 1 index 34: aarch64 3095  ->  HIP 198
prompt 2 index 20: aarch64  539  ->  HIP  13
prompt 5 index 32: aarch64   16  ->  HIP  15
prompts 0, 3, 4: IDENTICAL 48/48
```

**Every one of those three reversals lands on a step the 2026-08-23 gate
convicted us at, and in every case the oracle's new token is OUR token.**

| prompt | index | ours (CPU tier) | oracle, aarch64 | oracle, gfx1151 HIP | |
|---|---:|---:|---:|---:|---|
| 0 | 7 | 9338 | 9564 | 9564 | oracle unchanged |
| 1 | 34 | **198** | 3095 | **198** | oracle now emits OUR token |
| 2 | 20 | **13** | 539 | **13** | oracle now emits OUR token |
| 4 | 14 | 4593 | 22486 | 22486 | oracle unchanged |
| 5 | 32 | **15** | 16 | **15** | oracle now emits OUR token |

Scoring the CPU tier's **own recorded output** against the same oracle running on
its GPU kernels instead of its aarch64 CPU kernels turns its `5 of 6` into
`4 of 6`, and makes prompt 5 token-exact.

This does not clear the arm, and it is not a licence to pick an oracle path. Two
of the five divergences — prompts 0 and 4 — survive both of the oracle's kernel
paths, so they are ours. What it does establish is that a 6-of-6 token-exactness
demand against this oracle at these margins is partly a demand to bit-reproduce
one particular kernel schedule, and which schedule is a choice the gate has been
making silently. `AGENTS.md` §Gates requires the pinned oracle on each side; this
says the pin must name the executed path, which is why this run records
`use_extra_bufts`, `n_gpu_layers`, `system_info` and the device list for every
oracle leg.

## Finding 2: the board cannot currently complete this workload

Eighteen independent legs of the gate workload across three jobs, each leg its
own process and its own model load, **every one with `HSA_ENABLE_SDMA=0` set**.
Seventeen ended in a GPU reset.

The decisive run is the controlled one: six rounds, two arms, order alternated
between rounds, every leg counted.

```text
ARM default  ok=0  board_fault=6  harness_error=0  total=6
ARM adopt0   ok=0  board_fault=6  harness_error=0  total=6
TOTAL_OK_LEGS=0
SELF_REPRODUCIBLE=UNTESTED (0 clean leg)
TOKEN_GATE=NOT_MEASURABLE
REASON=no leg on either arm produced token ids; every failure was a board fault
```

| job | legs | board fault | harness error | completed |
|---|---:|---:|---:|---:|
| `ba83d1e1-…` | 1 | 0 | 1 | **1** (generation finished; ids lost to a read-only path) |
| `2644106f-…` | 5 | 5 | 0 | 0 |
| `3a518d75-…` | 12 | 12 | 0 | 0 |
| **pooled** | **18** | **17** | **1** | **1** |

The second arm, `adopt0`, ran with `VT_ADOPT_DEVICE_BYTES=0` — a runtime flag
needing no rebuild, which #2511's own A/B on the *minimal* repro read at 2
failures in 9 against base's 7 in 9. **At gate size it shows no benefit: 6 of 6,
identical to the default arm.** Both arms are reported because reporting only one
would repeat the survivorship error described below.

Signatures across the twelve controlled legs: **11 `GPU Hang`, 1 `Memory access
fault ... Page not present`**, which #2511 records as one defect. Every leg died
at **zero stdout bytes** — inside model load or the first prefill, never
mid-generation — consistent with #2511's finding that one prefill forward
reproduces it.

**These are board faults, not harness failures, and the distinction is
load-bearing.** An earlier run of this same job lost all five legs to a
`cannot write output token IDs` permissions error with `fault_lines=0`; had the
classifier not separated the two, this document would report a fault rate that
was really a read-only directory. The leg classifier now has three outcomes
(`OK` / `BOARD_FAULT` / `HARNESS_ERROR`) and a run whose legs die of harness
errors refuses to quote a fault rate at all.

**The workload is completable.** In that earlier run, one leg ran to completion —
6 successful requests, 42 input tokens, 288 generated tokens — and died only at
the file write. So the board is not categorically incapable of this workload; it
resets partway through most attempts at it.

**The native ROCm path is fully engaged when it faults.** All twelve legs of the
controlled run reported `reference_tier_hits=0`, with 17 to 21 `device=5`
(`kROCM`) op selections each and `stdout_bytes=0`. #2511
established that on the minimal repro; this is the independent confirmation on a
**gate-sized** workload, and it rules out the reading that the faults involve a
CPU fallback or an unregistered op. What fails is our native Q4_K kernels on the
board, doing the work the gate asks for.

**On the rate, and what it may not be quoted as.** `HSA_ENABLE_SDMA=0` halves the
fault rate rather than removing it: pooled over #2511's jobs, 16 of 21 legs fault
without it and 6 of 18 with it, and an earlier `0 in 10` figure double-counted
its own log lines.

**At this workload size the flag shows no measurable benefit at all.** It was set
on all 18 legs and 17 of them faulted; the two-arm run went 6/6 on both arms with
it set throughout. On the minimal repro it halves the rate. That is the complete
correction: the flag is worth something on one prefill and nothing established on
a gate-sized run, and it was never a fix.

That pooled 6-in-18 rate was measured on the **minimal** repro — `--max-tokens 1
--repeat 1`, a single prefill. A gate leg here is roughly **six prefills plus 288
decode steps**. The exposure window is larger by orders of magnitude, so a
near-certain fault rate on this workload is **consistent with** 6-in-18 on the
minimal one rather than contradicting it, and the two denominators must not be
compared as if they were the same experiment. Every count in this document is a
count of legs of *this* workload, on *this* boot.

Read the right way round, that is the contribution: **the board cannot sustain a
gate-sized Q4_K workload at all**, which nobody had measured before. #2511 knew
the board faults on one prefill. This says the failure does not average out over
a realistic run — it compounds.

## What ran

| `rc` job | purpose | outcome |
|---|---|---|
| `ba83d1e1-8320-4ff1-9a45-d52184a2d3fc` | the full gate | killed by me at leg 1 on a harness defect; its leg had completed generation |
| `2644106f-3123-4c18-8d42-d4db11fe62e8` | the full gate | oracle legs captured, `CHAIN_OF_CUSTODY=EXACT`; 5/5 board faults on our arm |
| `3a518d75-1214-4b54-b4ed-b6d5d5ca76f9` | 12-leg two-arm follow-up | 12/12 board faults, 0 harness errors, 0 clean legs |

Worker `rc-worker-lcjhd`, `strix:gpu0`, boot id
`a5bc8128-f6ad-4767-8614-6923f88032e1`, 32 cores. Nothing reached the box by
`ssh`. Raw logs on the share under `/mnt/nas_share/rc/rocm-tokgate-strix/out/`.

**Five earlier submissions produced no measurement**, each stopped by a distinct
defect, and they are listed because two of them would have produced a confident
wrong number rather than an error.

Each is listed with the assertion it now carries, because the assertion is the
reusable part. A future run of this harness names its own failure.

| # | `rc` job | defect | what it now asserts about itself |
|---|---|---|---|
| 1 | `2ae76f37-…` | locale-dependent source manifest: `sort` collates by locale, so one tree hashed `2e700801…` on the devbox (`en_US.UTF-8`) and `56c26d15…` in the container (`C`); the job refused a **correct** tree | manifest computed under `LC_ALL=C` and the run **prints the collation** beside the value; on a mismatch it diffs a per-file reference manifest and names `EXTRA_ON_WORKER` / `MISSING_ON_WORKER` / `CONTENT_DIFFERS` by path |
| 2 | `9ea7144e-…` | CMake cache path: the build tree was configured in-container at `/local/…`, so driving it by its other name was refused | both mounts exist and the build is driven through `/local`, with the reason recorded at the call site |
| 3 | `8161be67-…` | no `ccache` in the container, which that tree's `CMakeCache.txt` references permanently | sources the fleet's own `ccache-setup.sh`, fails on a non-zero build rc, and greps the build log for `ccache: not found` so the next occurrence names itself |
| 4 | `9038d1d4-…` | a helper shell function named `crun` — also the OCI runtime podman uses. `timeout` is an external binary and cannot see a shell function, so every timeout-wrapped call handed its entrypoint to the **container runtime** and got `unknown command` | renamed `podrun`, timeout taken as an **argument** not a prefix, `assert_podrun_is_ours` prints `podrun_resolves_to=function` at step 0, every invocation logs verbatim argv, and all three oracle binaries are smoke-started before the leg that matters |
| 5 | `ba83d1e1-…` | container-side writes into a read-only mount, losing the oracle's detokenized text | the harness gets a writable outdir; the oracle leg's own rc is gated (its rc 9 is "could not write"), and `hip_full_0.txt` must be non-empty |
| 6 | `ba83d1e1-…` | same root cause, different blast radius: `vllm-bench --output-token-ids` could not write, and **all five legs would have been recorded as board faults** with `fault_lines=0` in plain sight | ids written to a writable path; the leg classifier has **three** outcomes (`OK` / `BOARD_FAULT` / `HARNESS_ERROR`) and a run whose legs die of harness errors refuses to quote a fault rate at all |

Defects 4 and 6 are the instructive ones, and they fail in opposite directions.
**4 produced a plausible, correctly-formatted wrong answer rather than an error** —
an oracle that appeared to run and fail — and the unwrapped call sites worked, so
nothing looked wrong until step 6. **6 would have corrupted the one number this
document exists to report.** Defect 1 failed in the safe direction, but the
identical defect on a host whose locale matched the devbox would have *passed* a
contaminated tree.

Defects 4 and 5 are the instructive ones. **4 produced a plausible,
correctly-formatted wrong answer rather than an error**: `timeout` is an external
binary and cannot see a shell function, so every timeout-wrapped call resolved
`crun` from `PATH`, handed the real container runtime its entrypoint path, and
returned `unknown command` with rc=1 — which reads exactly like an oracle that
ran and failed. The unwrapped call sites worked, so nothing looked wrong until
step 6. **5 would have corrupted the number this document exists to report**: all
five legs failed with `cannot write output token IDs` and `fault_lines=0`, and a
two-outcome classifier would have recorded them as five board faults. Defect 1
failed in the safe direction, but the identical defect on a host whose locale
matched the devbox would have *passed* a contaminated tree.

## Measured identity

Asserted inside the job, which fails on a mismatch.

```text
gguf_sha256              = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
gguf_size                = 17106775008                     (verified ON the worker)
llama_pin                = 10bf611e533d81f739128304991c5e133c6aebd8  (tag b10451)
llama_src_files          = 3425
llama_src_manifest       = 56c26d15c2acf11b8621ac26663b4316dc29719d765ba1d95231ffacaddf3cda
llama_src_manifest_after_harness = 56c26d15... (unchanged; the harness built out of tree)
vllmcpp_source_revision  = 11fed3ba56b8f823c07032416982a44a8c0967b5
libvllm.so               = 41bb40529c957a7ccc048eba6c44974c94e0dba4d8325c49c53835aa6fdf4301
vllm-cli                 = a703b83dd8954ba6dd3cbe82efcd38083c1d55492bbbaecf5c406f7c6efd646f
vllm-bench               = 4e2d83651fbb0a11cafd9465c63d61eb58415d050f1eba080405dcb7c7b5a499
tokenize                 = b2c005c79aecea2e9d0d10548ecc113a823fa408ba9b1a6abbbb474d0b0c2286
host                     = rc-worker-lcjhd (strix:gpu0), x86-64, 32 cores
```

The source manifest is computed under `LC_ALL=C`, and the run prints the
collation beside the value. `git archive` of the pin, the staged tarball and the
worker's tree all reduce to `56c26d15...`; under `en_US.UTF-8` the same trees
reduce to `2e700801...`, which is what made the first submission refuse a correct
tree. A manifest that is a property of whoever ran it is not an identity.

`vllm-cli` is a ~26 KB thin client, so `libvllm.so` is the identity that matters
and both are recorded. The library's hash is re-recorded after the client build
to prove the thing under test did not move.

### The tree this measures

The binary is the one #2511 characterised, at `11fed3ba5`. **Six** commits touch
`src/vt/rocm/` or `include/vt/rocm/` between it and this document's branch base
`c9366de65` — the count is stated against a named object because `origin/main`
moved 88 commits while this ran, and a count against a moving ref is false by the
time anyone reads it. None of the six changes the Q4_K decode path: `f424b53fe` (`FlushPending`, whose only call site
is guarded by `slot.ref_selected` at `src/vt/op_provider.cpp:707-711` and never
fires on a run with zero reference-tier hits), `456f9cd6f` (the `Fmt == 3` Q6_K
body, `VT_ROCM_Q6K_SMALL_PRIVATE`, default OFF and measured worse), `3fede4162`
and `e38403663` (`IQ3_S`), `1dc686479` and `9dcc34cc3` (EXL3). `1dc686479` moves
four f16/bf16 device codec helpers into a new header and `DF16ToF32` decodes a
Q4_K superblock scale, so it was checked rather than assumed: the four function
bodies hash `3f8da2d5ce1760bd1d72acdf147285576ec03aa4e409fdeb7c38b69b41a363b8` on
both sides of the gap, and only comments differ.

Three further ROCm commits landed on `origin/main` after this branch's base, the
substantive one being `8bca27b26` (decouples the RmsNorm gamma dtype from the
activation dtype and adds a `kF16` gamma path; its own body states the arithmetic
is bit-identical for pairings that already worked, and ours is bf16 gamma with
bf16 activations). **None can affect this measurement in either direction,
because no leg produced token ids** — a numerics change cannot move a verdict that
was never scored. A future scoring run on a rebuilt binary must account for it.
The spec carries the full assessment.

## The oracle's executed kernel path, pinned

Recorded per leg rather than inferred, because this oracle's greedy decode is not
deterministic across its own kernel paths.

```text
ORACLE_N_GPU_LAYERS     99   (the HIP arm)   /   0 (the CPU arm, see below)
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

### Chain of custody

`CHAIN_OF_CUSTODY=EXACT`. The harness's detokenized prompt-plus-generation for
prompt 0 is byte-identical to the stock `llama-completion` binary's stdout on the
same recipe, run in the same lease from the same build. So the ids this gate
would score belong to a sequence the stock binary demonstrably produced.
`llama-completion` is the binary the 2026-08-23 run used; `llama-cli` at this pin
is the conversation tool and applies a chat template, so it cannot reproduce a
raw-completion recipe.

The harness `oracle_tokens.cpp` is the 2026-08-23 source, unchanged except for
reading `n_gpu_layers` and `use_extra_bufts` from the environment and printing
`system_info` and the device list. It links stock `libllama` built at the pin and
calls the public `llama.h` API only. `AGENTS.md` admits this as an unavoidable
adaptation of the harness, because `llama-completion` at `b10451` prints token
pieces and never token ids.

### The x86-64 CPU oracle leg is a FAILED instrument, and was not used

A second oracle leg at `n_gpu_layers = 0` from the same build was intended as a
bridge: the CPU tier's recorded oracle ids came from aarch64, so an x86-64 CPU
leg would have separated the host term from the kernel-path term. It emits **one
repeated token for the whole 48-token completion on 5 of 6 prompts** (distinct-id
counts 1, 3, 1, 1, 1, 1) while the `ngl=99` leg from the same binary in the same
job is coherent. It is recorded as an observation, not a diagnosis, and is
tracked by [#2557](https://github.com/mudler/vllm.cpp/issues/2557) with the three
discriminating experiments named. **No conclusion in this document rests on it.**
The tier comparison uses the HIP leg against the recorded aarch64 leg instead.

## The `blk.64` asymmetry, re-observed

`b10451` loads 851 of the artifact's 866 tensors and ignores all 15 of `blk.64`
(289,527,808 bytes), four of them the `nextn.*` MTP head. The stock control run
emitted exactly **15** `unused tensor blk.64.*` warnings — re-observed on the HIP
path here rather than cited. Our arm runs with MTP off (no `--speculative-config`),
so both engines decode the same 851 tensors and the comparison is matched *work*
and not only matched weights.

## Tokenizer: EXACT, 6 of 6

`examples/tokenize`, reading the GGUF's own vocab, reproduced the oracle's
`PROMPT_IDS` line for line at lengths 6, 5, 6, 7, 11, 7. The #1355 prompt-token
undercount does not appear on this path.

## Generation: NOT MEASURED

No leg produced token ids, so there is no generation table, no divergence index,
no logit margin and no comparison of our ROCm output against either tier. The
gate's tolerance is unchanged and no band was reached for; the instrument simply
could not read the arm.

Two things that were ready and are not spent: `compare_tokens.py` (the
2026-08-23 verdict script, unmodified) and `compare_tiers.py`, which answers
whether ROCm diverges at the same indices with the same ids as the CPU tier.
The latter was mutation-proved before the run — fed the recorded CPU-tier ids it
reproduces that run's 5 divergences at the recorded indices and ids, and an arm
mutated to diverge at index 3 instead of 7 drops it to 4 with
`DIFFERENT index (ROCm 3, CPU 7)`. Both are committed and the oracle ids are
captured, so the moment one reproducible pair of legs exists this is a scoring
job and not a measurement job.

## Why ONE clean leg could not have produced a verdict either

Worth stating explicitly, because a lone completed run among many faulted ones is
the most tempting artefact this board can produce, and it is the least
trustworthy.

The surviving hypothesis for the fault is managed-memory migration on a device
that reports `PageableMemoryAccess = 0` and so cannot take a recoverable page
fault, with the fault site at `KQuantGemmK<u16, 0>` — the Q4_K arm, the path this
gate scores. **A defect that can corrupt a page can hand back wrong bytes without
faulting.** On this board, survival is therefore not independent of correctness:
selecting the leg that completed is selecting on the very variable the gate
measures, and a divergence found in it could not be distinguished from
corruption.

So the rule this run applies is: **two or more clean legs that agree with each
other, or no verdict.** One clean leg yields `NOT_MEASURABLE`, and even two
agreeing legs would be reported with the plain statement that they were drawn
from a population that mostly faulted. This is stricter than a naive reading of
the spec's reproducibility precondition, which only refused legs that *disagree*;
refusing a lone survivor closes the case where there is nothing to disagree with.

## What is not admissible

- **No speed, latency or memory axis.** The gate has not passed. Nothing in this
  document may be quoted for one.
- **The `ngl=0` oracle leg**, which is a failed instrument (#2557).
- **Any claim that `HSA_ENABLE_SDMA=0` works.** It halves the rate. Five legs of
  this workload faulted with it set, and a run that had come out clean would not
  have been evidence either.
- **Any inference that the ROCm numerics are better than the CPU tier's** from
  Finding 1. The three reversed steps are the oracle disagreeing with itself on
  the CPU tier's output; they say nothing yet about what our ROCm arm emits.

## The next traceable hypothesis

No ceiling is declared.

1. **The board fault blocks everything and is owned by #2511.** Its surviving
   hypothesis is managed-memory migration on a device reporting
   `PageableMemoryAccess = 0`, which cannot take a recoverable page fault; the
   fault site is `KQuantGemmK<u16, 0>`, the Q4_K arm, which is the path this gate
   scores. Until a leg completes reproducibly there is no ROCm token verdict.
2. **The two surviving CPU-tier divergences are the real target.** Prompts 0 and
   4 diverge against both of the oracle's kernel paths, so they are not oracle
   self-noise. [#2534](https://github.com/mudler/vllm.cpp/issues/2534) owns them,
   and its leading term — our CPU decode running bf16 activations where the
   oracle runs f32 — is shared with the ROCm arm.
3. **The gate's own definition needs the executed path.** A 6-of-6 demand at
   0.03-to-0.18-logit margins is partly a demand to bit-reproduce one kernel
   schedule. This run supplies a second measurement of that effect (3 of 6, next
   to the recorded 1 of 6) and the machinery to record it.
