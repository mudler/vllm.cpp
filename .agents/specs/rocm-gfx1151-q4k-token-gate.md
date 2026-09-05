# Spec — the Q4_K_M token gate on gfx1151

Row `BACKEND-ROCM`. Issue
[#2546](https://github.com/mudler/vllm.cpp/issues/2546).

Sibling records: [#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the
quant-matched decode number this gate gates),
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) (the SDMA fault that
made this measurement impossible until today),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (the CPU-tier numerics
defect this measurement is compared against), and
[#821](https://github.com/mudler/vllm.cpp/issues/821) (the Q4_K_M arm).

## Now

`ACTIVE` — the measurement ran and returned `TOKEN_GATE=NOT_MEASURABLE`.

**17 of 18 legs of this workload ended in a GPU reset, including 12 of 12 in a
controlled two-arm run with zero harness errors.** No leg produced token ids, so
there is no ROCm token verdict and no divergence to attribute. The stop condition
this spec declared in advance is the one that fired, and it fired cleanly.

Two results stand on their own:

- **The board cannot complete a gate-sized Q4_K workload.** #2511 established a
  fault on a single prefill; this establishes that the failure compounds over a
  realistic run rather than averaging out. `HSA_ENABLE_SDMA=0` was set on every
  leg and shows no measurable benefit at this workload size.
- **The oracle disagrees with itself on 3 of these 6 prompts** across its own
  kernel paths, and each disagreement lands on a step the 2026-08-23 CPU gate
  convicted us at, emitting our token. That is a second, larger measurement of
  the effect `.agents/oracles/llama-cpp.md` records at 1 of 6.

Everything upstream of the legs is settled and reusable: artifact sha verified on
the worker, a 3425-file source manifest under `LC_ALL=C`,
`CHAIN_OF_CUSTODY=EXACT`, `blk.64` re-observed at 15 on the HIP path, both oracle
kernel paths pinned and recorded, and the tokenizer exact on 6 of 6. The moment a
reproducible pair of legs exists, this becomes a scoring job rather than a
measurement job.

Evidence:
[`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-20260902.md).

This spec was committed before the job that produced the number, and it declared
the verdict shape, the identity assertions and the stop conditions in advance, so
that no threshold could move after the fact. Two of them were tightened while the
result was still unknown: the reporting rule for a same-rate/different-index
outcome, and the refusal to score fewer than two agreeing clean legs.

## Scope

In scope: running the declared Qwen3.8-27B Q4_K_M token gate on `strix:gpu0`
(`gfx1151`, Radeon 8060S, ROCm 7.2.4) against pinned llama.cpp `b10451`, and
reporting `TOKEN_GATE=PASS` or `TOKEN_GATE=FAIL` with per-prompt divergence
indices, token ids and logit margins.

In scope: the comparison against the CPU tier's recorded 5-of-6 result on the
same artifact and the same prompts, which decides whether the two tiers carry
one shared term or two.

Out of scope, deliberately:

- **Every speed, latency and memory number.** `AGENTS.md` §Gates admits a
  performance result from an arm only after that arm's declared token gate
  passes. #2497 already had one measurement retracted for quoting a number from
  this arm. No figure of that kind is produced, recorded or quoted here, even
  as a by-product.
- **Fixing the numerics.** #2534 owns the CPU-tier defect and its fix is likely
  shared. This row measures and reports.
- **The SDMA cause.** #2511 owns it. `HSA_ENABLE_SDMA=0` is used here as a
  workaround with that provenance stated, never as a finding.

## The gate

Mirrors `docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`, which is
the CPU tier's run of the same declared gate
(`.agents/specs/qwen38-27b-quant-arms.md` §Gates). Reusing its method is the
point: a different method would not be comparable, and the comparison is the
most valuable output.

- Six raw completion prompts, no chat template, byte-identical to the file the
  2026-08-23 run used.
- 48 tokens each, greedy, concurrency 1, `ignore_eos`, MTP off on both sides.
- One artifact: `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B, sha256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, staged to
  worker-local `/tmp` and re-hashed there, because `/workspace` is CIFS and a
  CIFS mmap is not a run surface.
- Verdict: `TOKEN_GATE=PASS` only when the tokenizer matches on 6 of 6 and the
  generation matches on 6 of 6. Anything else is `FAIL`. The tolerance is exact
  and no band is reached for.
- There is a third outcome, and it is not a softened `FAIL`:
  `TOKEN_GATE=NOT_MEASURABLE`, when every leg faulted or when the clean legs
  disagree with each other. It says the instrument could not read the arm, which
  is a different claim from the arm being wrong, and collapsing the two would
  misreport a board problem as a numerics problem.

## Pinning the oracle's EXECUTED PATH, not only its revision

`.agents/oracles/llama-cpp.md` records, from a measurement on `thor:gpu0`, that
`b10451`'s greedy decode is **not deterministic across its own supported kernel
paths**: the stock `-nr/--no-repack` flag changes its own greedy tokens on 1 of
these 6 prompts, and its per-step self-perturbation of the final logits is min
0.2020, median 0.3790, max 1.3657. A gate that pins only the revision is
therefore under-specified.

This run records, for every oracle leg:

- `llama_print_system_info()` verbatim, which names the compiled feature set;
- `use_extra_bufts`, set explicitly rather than defaulted, and printed;
- `n_gpu_layers`, so the HIP and CPU legs are distinguishable in the record;
- the host architecture and the enumerated ggml devices;
- the source identity of the pinned tree, as a content manifest hash rather
  than a tarball hash, because gzip framing is not content. The manifest is
  computed under `LC_ALL=C`, and the run prints the collation beside the value.

**That last clause is not tidiness, and it cost the first submission.** Run 1 of
this job refused the staged llama.cpp tree at step 2: it computed
`56c26d15...` against an expected `2e700801...` and stopped. The tree was
correct and the INSTRUMENT was wrong. `sort` collates by locale; the devbox that
derived the expected value ran `en_US.UTF-8` and the leased container ran `C`, so
the two hosts ordered the same 3425 file names differently and hashed the same
bytes to two different values. Re-derived under `LC_ALL=C`, `git archive` of the
pin, the staged tarball and the worker's own tree all give `56c26d15...`.

This is `verification.md` §"Make the instrument say what it is measuring" in its
cheapest form. The failure was loud and fired before the lease was spent, which
is the good case; the same defect pointed the other way would have PASSED a
contaminated tree on a host whose locale happened to match. The manifest is now a
property of the tree rather than of whoever ran it, and the run states the
collation in its own output so a reader can see the wiring.

Two oracle legs run, in one lease, from one build:

| leg | `n_gpu_layers` | what it pins |
|---|---|---|
| `hip` | 99 | the board's own kernel path, the denominator for an on-board gate |
| `cpu` | 0 | the x86-64 CPU path on the same host, the bridge to the CPU tier |

The second leg exists because the CPU tier's recorded oracle ids came from
**aarch64** `thor`. Comparing our gfx1151 ids straight to those would confound
three differences at once. An x86-64 CPU oracle leg on the same host, from the
same build, isolates the kernel-path term from the host term.

## The harness adaptation, and why it is not a second oracle

`llama-completion` at `b10451` prints token pieces and never token ids, so the
2026-08-23 run wrote `oracle_tokens.cpp`, which links the stock libllama built
at the pin, calls the public `llama.h` API only, and mirrors `completion.cpp`'s
own tokenize and detokenize choices. `AGENTS.md` admits this as an unavoidable
adaptation of the harness. This run reuses that source with three additions,
all of them recording rather than deciding: `n_gpu_layers`, `use_extra_bufts`
and the `system_info` line.

Chain of custody is re-established on this host rather than inherited: the
harness's detokenized prompt-plus-generation for prompt 0 is compared byte for
byte against a stock `llama-cli` run on the same recipe. A `DIVERGE` there
voids the oracle side and the job says so instead of reporting a gate result.

## Confirming the legs, not assuming them

**`HSA_ENABLE_SDMA=0` halves the fault rate. It does not remove it.** An earlier
tally reported 0 failures in 10 legs with the flag against 8 in 10 without it,
and that tally double-counted its own log lines: 50 `MLEG` lines, 25 unique.
Corrected and pooled over every #2511 job the rate is **16 of 21 legs without the
flag and 6 of 18 with it**, and the later properly-alternated 20-leg A/B reads
9 of 10 against 4 of 10. So a leg here has roughly a **two-in-five chance of
dying in a GPU reset**, and the flag is the best arm available rather than a fix.

Three consequences, each wired into the job rather than noted beside it.

**Legs are expected to fault, so they are counted.** Five independent legs run,
each its own process and its own model load. The record carries `failures / N`.
A faulted leg produced no ids at all and is not scored; a verdict from a leg
that died mid-generation would be meaningless.

**A completed leg is not evidence the flag works.** At `p = 0.4` five clean legs
in a row happens about one run in a hundred. Reporting only the run that came
out clean is exactly the survivorship error that produced the 0-in-10 figure, so
the observed rate is reported with the verdict and the run is not repeated until
the board cooperates.

**Reproducibility is a precondition of the verdict, not a diagnostic.** The
surviving hypothesis is managed-memory migration on a device that reports
`PageableMemoryAccess = 0` and so cannot take a recoverable page fault, and the
fault site is `KQuantGemmK<u16, 0>` — the **Q4_K** arm, which is the path this
gate scores. A defect that can corrupt a page can in principle hand back wrong
bytes *without* faulting, and that would present as a token divergence. So the
clean legs are compared to each other first, and a disagreement between them
yields `TOKEN_GATE=NOT_MEASURABLE` rather than a divergence count.

**The kernel itself is exonerated.** A standalone launch at the exact production
geometry ran 9,000 launches over 30 legs with zero failures while the board
failed 5 of 6 model legs in the same lease, so the fault is not an
out-of-bounds index inside that kernel. That is #2511's finding and is recorded
here only because it shapes what this measurement's limitations section may
claim.

## The tree this measures, stated as a bounded claim

The binary under test is the one #2511 characterised: `libvllm.so` built from
`11fed3ba56b8f823c07032416982a44a8c0967b5`, Release, `VLLM_CPP_HIP=ON`,
`VLLM_CPP_HIP_ARCHITECTURES=gfx1151`. Scoring the gate on that artifact keeps
the fault-rate evidence and the token verdict on one binary, which is worth
more here than currency.

That base is behind `origin/main`, and "behind" is not a useful thing to say.
The gap is enumerated instead — but **a count against a moving `origin/main` goes
stale, so it is stated against a NAMED object**. Measured at this branch's base
`c9366de65`, **exactly six commits** touch `src/vt/rocm/` or `include/vt/rocm/`,
and none changes the Q4_K decode path:

```sh
git log --oneline 11fed3ba5..c9366de65 -- src/vt/rocm/ include/vt/rocm/   # six
```

**`origin/main` moved 88 commits while this row ran, and three more ROCm commits
landed in that window**, which is exactly how a count committed as prose becomes
false on merge. They are assessed rather than left to be discovered:

| commit | assessment |
|---|---|
| `8bca27b26` | decouples the RmsNorm gamma dtype from the activation dtype on ROCm and adds a `kF16` gamma path. Its own body states that the arithmetic "is bit-identical for every pairing that already worked"; our path is a bf16 gamma with bf16 activations, a pairing that already worked, so it adds a capability rather than moving this arm's numerics |
| `cd5e9d078` | a merge |
| `fcd446b03` | a record change: the ROCm arm compiles |

**None of the three can affect this measurement in either direction, for a reason
that does not depend on reading them: no leg produced token ids.** A numerics
change cannot move a verdict that was never scored. They are listed because a
future scoring run on a rebuilt binary must account for `8bca27b26`, and because
the six-commit claim above must be re-derived at whatever object this row
eventually merges onto rather than trusted.

| commit | what it is | why it cannot move this measurement |
|---|---|---|
| `f424b53fe` | `RocmBackend::FlushPending` | its only call site is guarded by `slot.ref_selected` (`src/vt/op_provider.cpp:707-711`), so it fires only when the portable reference tier is selected. Every #2511 leg reported zero reference-tier hits and this run asserts the same, so it never executes here |
| `456f9cd6f` | the `Fmt == 3` Q6_K register-resident body | ours, `VT_ROCM_Q6K_SMALL_PRIVATE`, default OFF, and measured WORSE (6 of 6 failures against 4 of 6). It must stay off and this run does not set it |
| `3fede4162`, `e38403663` | `IQ3_S` | a different ggml type; one line of `rocm_grouped_gemm.hip` each |
| `1dc686479`, `9dcc34cc3` | EXL3 trellis on gfx1151 | a different quantization format |

`1dc686479` is the only one that needed a check beyond its subject, because it
moves four f16/bf16 device codec helpers out of `rocm_grouped_gemm.hip` into
`include/vt/rocm/rocm_f16_codec.h`, and `DF16ToF32` decodes a Q4_K superblock
scale. The move is comment-only for the code: the four function bodies hash
`3f8da2d5ce1760bd1d72acdf147285576ec03aa4e409fdeb7c38b69b41a363b8` on **both**
sides of the gap. Only added comments differ.

So the Q4_K fault path and the Q4_K arithmetic are unchanged across the gap.
The evidence states this list rather than a commit count, because a list is
checkable and a count reads as unquantified staleness.

## How a same-rate, different-index result is to be reported

Fixed here, before the data, because it is exactly the finding a reader would
want massaged afterwards.

The CPU-tier work has measured that under perturbations which did **not** move
the divergence rate, the first-diff INDEX moved readily: prompt 1 from 34 to
21, prompt 2 from 20 to 4. The index is therefore not a stable signal. What is
stable is the RATE and the SET of near-tie steps.

Three outcomes and their honest labels:

- **Same rate, same indices, same ids** — the strongest available evidence for
  one shared term in the quant path.
- **Same rate, different indices** — reported as exactly that, and NOT as
  evidence of a ROCm-local cause. A moved index is what this class of
  perturbation does. The margins decide, and the comparison of the near-tie
  step SETS carries more than the first-diff column.
- **Different rate** — a real difference in magnitude between the tiers, and
  the first thing worth chasing.

No result is to be forced onto the recorded `7 / 34 / 20 / - / 14 / 32`
pattern. A same-rate, different-index answer reported plainly is a better
result than a match argued into existence.

## Risks

- **The board faults anyway.** Contradicts a 10-leg result and is important
  news. Stop condition below.
- **A silent CPU fallback.** `gfx1151` is integrated, so the portable reference
  tier is reachable and a fallback would be invisible in the ids. The run sets
  `VT_OP_PROVIDER_STATS=1` and records the reference-tier hit count; a non-zero
  count means the leg measured the CPU tier wearing a ROCm label.
- **A stale binary.** `vllm-cli` is a ~26 KB thin client and the kernels are in
  `libvllm.so`. Both are hashed, and the source revision is printed by the build
  step rather than assumed from a directory name.
- **A previous run's log read as this run's verdict.** Logs are deleted before
  submission and every wait keys on the `rc` job UUID.

## Evidence

`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-20260902.md`,
following the conventions of the 2026-08-23 CPU-tier document: what ran, the
measured identity chain, the `blk.64` asymmetry and how it was handled, the
oracle's chain of custody, the gate, and what is not admissible.

Raw logs on the share under `/mnt/nas_share/rc/rocm-tokgate-strix/`.

## Stop conditions

- **Every leg faults**: report `TOKEN_GATE=NOT_MEASURABLE` with the count out of
  N. This is an ordinary outcome at the corrected rate, not a contradiction of
  anything, and it is a legitimate result rather than a reason to retry until
  the board yields a clean run.
- **The clean legs disagree with each other**: report `TOKEN_GATE=NOT_MEASURABLE`.
  A non-deterministic arm cannot be scored for token-exactness.
- **Fewer than two agreeing clean legs**: report `TOKEN_GATE=NOT_MEASURABLE`.
  This is stricter than the clause above and it closes a case that clause cannot
  see. Refusing legs that *disagree* does nothing when only one leg survives,
  and a lone survivor is the least trustworthy evidence this board produces, not
  the most: if the fault is memory corruption on the Q4_K path, a leg can
  complete while having been handed wrong bytes, so survival is not independent
  of correctness and selecting the completed leg selects on the variable the
  gate measures. Two agreeing legs are the minimum, and even then the evidence
  states that they were drawn from a population that mostly faulted.
- The oracle's chain of custody is not `EXACT` or `PREFIX`: stop, the oracle
  side is not bound to the stock binary.
- A reference-tier hit count above zero on our arm: stop, the leg did not
  measure the ROCm tier.
- Anything that would need a speed number to interpret: stop, and say that the
  question is not answerable before the gate passes.

## Owed

- **The ROCm token verdict itself.** It is unmeasured, not failing. It stays owed
  by this row and by [#2546](https://github.com/mudler/vllm.cpp/issues/2546),
  and it is blocked on #2511 rather than on anything in the numerics.
- The fix for whatever divergence it eventually measures. If the ROCm
  divergences match the CPU tier's indices and ids, #2534 owns it. If they do
  not, ROCm carries its own term and that term gets its own issue against this
  row.
- #2497's quant-matched decode number stays blocked until this gate passes.
- The degenerate `n_gpu_layers = 0` oracle leg is owed by
  [#2557](https://github.com/mudler/vllm.cpp/issues/2557), which names the three
  discriminating experiments. No conclusion here rests on it.
