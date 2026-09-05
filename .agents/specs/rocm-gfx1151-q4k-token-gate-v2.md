# Spec — the Q4_K_M token gate on gfx1151, second attempt, on a board that can finish

Row `BACKEND-ROCM`. Issue
[#2546](https://github.com/mudler/vllm.cpp/issues/2546).

Sibling records: [#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the
quant-matched decode number this gate gates),
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) (the allocator defect
that made the first attempt unmeasurable, now fixed and landed),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (the logits-resolution
defect whose fix moved the CPU tier from 5 of 6 to 3 of 6), and
[#821](https://github.com/mudler/vllm.cpp/issues/821) (the Q4_K_M arm).

Predecessor spec:
[`rocm-gfx1151-q4k-token-gate.md`](rocm-gfx1151-q4k-token-gate.md), which
declared this gate and returned `TOKEN_GATE=NOT_MEASURABLE` because 12 of 12
legs of a controlled run ended in a GPU reset. This spec does not redefine the
gate. It re-runs the same declared gate on a binary that carries two fixes the
first attempt did not have, and it states in advance what may and may not be
concluded from the difference.

## Now

`ACTIVE` — the measurement ran and returned `TOKEN_GATE=FAIL` at 3 of 6.

**The board completed 6 legs of 6 with zero GPU resets**, on the shipped default
with no environment knob set, against the predecessor's 17 faults in 18 legs of
the same workload. All six legs are byte-identical. `27da7787e` is what changed,
and this is the first token verdict this arm has ever had.

The gate fails, and its three losses are **disjoint** from the post-#2534 CPU
tier's three. Same rate, different prompts: ROCm loses 1, 3 and 5, the CPU tier
loses 1, 2 and 4, and the one prompt in common is a different step (45 against
34). Resolving all six contested steps to what all four sides emit splits them
in half: three are the oracle disagreeing with itself across its own kernel
paths, at exactly the three prompts the predecessor measured that on, and at
those our two tiers emit the SAME token. The other three — p1/45, p3/45, p4/14 —
are steps where our ROCm and CPU arms compute a different argmax over an
identical prefix with no oracle in the comparison. That is a ROCm-local term,
measured here for the first time, and
[#2590](https://github.com/mudler/vllm.cpp/issues/2590) owns it.

Every ROCm loss is a rank-2 near-tie: over 288 steps the oracle ranks our token
1 on 285 and 2 on 3, nothing worse occurs, and our arm takes the top-1 on 16 of
the 19 steps whose oracle gap is below 0.20 — including all three the CPU tier
is convicted at.

`ORACLE_REPRO=YES` on 6 of 6 prompts and `CHAIN_OF_CUSTODY=EXACT`, so the
denominator did not move between the two attempts and the change in verdict is
entirely on our side.

Evidence:
[`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md).

This spec was committed before the job that produced the number, in commit
`6a859c170`, and it declared the verdict shape, the identity assertions, the
same-rate/different-index reporting rule and the stop conditions in advance, so
that no threshold could move after the fact.

## Why this is newly possible

Two commits landed on `origin/main` between the first attempt and this one, and
each of them changes something the first attempt could not measure around.

- **`27da7787e` (#2511).** `UseManagedAlloc` is narrowed by
  `pageable_memory_access`, so `gfx1151` — which reports
  `PageableMemoryAccess = 0` and therefore cannot take a recoverable page fault
  — no longer receives migratable `hipMallocManaged` memory under a live queue.
  Measured on the board across one build, one boot, arms interleaved: the fixed
  arm 0 failures in 10 legs against the pre-fix arm 9 in 10, with the clean legs
  producing 1136 characters of correct text at gate size. That is the reason a
  leg can now be expected to finish.
- **`292fba645` (#2534).** A GGUF keep-quant `lm_head` now routes to the
  f32-output GEMM instead of the bf16-output helper, so the logits the sampler
  compares carry f32 resolution. On the CPU tier this moved the same six-prompt
  gate from 5 diverging prompts to **3**: prompts 1 (index 34), 2 (index 20) and
  4 (index 14) still diverge; prompts 0, 3 and 5 are token-exact.

The first consequence is that the ROCm arm may now be scoreable. The second is
that the CPU tier's comparison point has moved, and this run compares against
the **post-fix** 3-of-6 result rather than the retired 5-of-6 one. Comparing
against 5-of-6 would confound the two fixes.

## Scope

In scope: running the declared Qwen3.8-27B Q4_K_M token gate on `strix:gpu0`
(`gfx1151`, Radeon 8060S, ROCm 7.2.4) against pinned llama.cpp `b10451`, from a
binary built at `27da7787ecd702b33a74304f4d8288c6bbeb4d00`, and reporting
`TOKEN_GATE=PASS` or `TOKEN_GATE=FAIL` with per-prompt divergence indices, token
ids and the oracle's recorded margin at each first divergence.

In scope: the observed fault rate out of N, reported whatever it is. A board
that still faults contradicts a 10-leg result and is the news, not a nuisance.

In scope: the comparison against the CPU tier's **post-#2534** 3-of-6 result on
the same artifact and the same prompts, which decides whether the two tiers
carry one shared term or two.

Out of scope, deliberately:

- **Every speed, latency and memory number.** `AGENTS.md` §Gates admits a
  performance result from an arm only after that arm's declared token gate
  passes, and #2497 has already had one measurement retracted on this row for
  quoting a number from this arm. None is produced, recorded or quoted here,
  even as a by-product, and the harness prints none.
- **Fixing whatever divergence this measures.** This row measures and reports.
  #2534 owns the CPU-tier term; a ROCm-local term, if one appears, gets its own
  issue against this row.
- **`HSA_ENABLE_SDMA=0`.** RETIRED. It was a partial mitigation of a symptom
  and the allocator was the cause: it halved the fault rate on a single prefill,
  showed no benefit at gate size, and was set on all 18 legs of the first
  attempt while 17 of them faulted. It is not set on any leg here and it is not
  described as a workaround anywhere in the evidence.
- **`VT_ADOPT_DEVICE_BYTES=0`.** The first attempt's second arm. It was a
  candidate mitigation for a cause that is now known and fixed, and it showed no
  benefit at gate size. One arm runs here: the shipped default.

## The gate

Unchanged from the predecessor spec, which mirrors
`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`, the CPU tier's run
of the same declared gate (`.agents/specs/qwen38-27b-quant-arms.md` §Gates).
Reusing the method is the point: a different method would not be comparable, and
the comparison is the most valuable output.

- Six raw completion prompts, no chat template, byte-identical to the file the
  2026-08-23 run used (`prompts_sha256`
  `c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`).
- 48 tokens each, greedy, concurrency 1, MTP off on both sides.
- One artifact: `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B, sha256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, staged to
  worker-local `/tmp` and re-hashed there, because `/workspace` is CIFS and a
  CIFS mmap is not a run surface.
- Verdict: `TOKEN_GATE=PASS` only when the tokenizer matches on 6 of 6 and the
  generation matches on 6 of 6. Anything else is `FAIL`. The tolerance is exact
  and no band is reached for.
- The third outcome survives and is not a softened `FAIL`:
  `TOKEN_GATE=NOT_MEASURABLE`, when no leg completes, when the clean legs
  disagree with each other, or when fewer than two clean legs agree. It says the
  instrument could not read the arm, which is a different claim from the arm
  being wrong.

## The binary under test, and how the two fixes are proven present in it

The first attempt scored a binary built at `11fed3ba5`, deliberately, to keep
the fault-rate evidence and the token verdict on one artifact. That choice does
not survive its own premise: both landed fixes are upstream of that object, so a
run on it would measure neither. This run therefore builds fresh at
`27da7787ecd702b33a74304f4d8288c6bbeb4d00`, which is `origin/main` at the time
this spec was committed, and it proves the rebuild rather than assuming it.

Four assertions, in the job, each failing the job rather than being reported:

1. **The staged tree is the named object.** `git rev-parse HEAD` in the
   worker's clone must equal `27da7787ecd702b33a74304f4d8288c6bbeb4d00`.
2. **The source carries both fixes.** The clone must contain the #2534 routing
   line `if (vt::IsBlockQuant(lm_head.dtype)) return MatmulF32D(d, x, lm_head);`
   in `src/vllm/model_executor/models/qwen3_5.cpp`, and the #2511 narrowing
   `p.managed_alloc = a.pageable_memory_access;` in
   `include/vt/rocm/rocm_arch.h`.
3. **The build is clean and from that tree.** The build directory is removed
   first, so every object comes from the asserted source; an incremental build
   over a cached tree could not carry that claim.
4. **The artifact moved, and carries the #2511 marker.** `libvllm.so` is
   hashed and must differ from the first attempt's
   `41bb40529c957a7ccc048eba6c44974c94e0dba4d8325c49c53835aa6fdf4301`, and
   `grep -a` must find the `hipDeviceAttributePageableMemoryAccess = 0` refusal
   string, which is a literal `27da7787e` introduces and `11fed3ba5` cannot
   contain. Where the first attempt's library is still on the worker the job
   greps it too and prints ABSENT beside PRESENT, so the marker is shown to
   discriminate rather than merely to be found.

`vllm-cli` is a ~26 KB thin client and the kernels are in `libvllm.so`, so the
library is the identity that matters. Hashing the client is the defect #2511's
own harness hit, where a correct rebuild read as byte-identical.

The #2534 fix has no string literal, so it cannot be grepped for. Assertions 1,
2 and 3 are what carries it: a clean build of an asserted tree that contains the
routing line necessarily contains the routing line's code.

## Pinning the oracle's EXECUTED PATH, not only its revision

Unchanged from the predecessor spec and restated because it binds this run.
`.agents/oracles/llama-cpp.md` records that `b10451`'s greedy decode is **not
deterministic across its own supported kernel paths**, and the first attempt
measured a second and larger instance of that effect: changing from aarch64 CPU
kernels to gfx1151 HIP kernels moves 3 of these 6 prompts. A gate that pins only
the revision is under-specified.

This run re-runs the HIP oracle leg rather than only carrying the recorded one,
and records for it: `llama_print_system_info()` verbatim, `use_extra_bufts` set
explicitly and printed, `n_gpu_layers`, the host architecture and enumerated
ggml devices, and the source identity of the pinned tree as a content manifest
hash computed under `LC_ALL=C` with the collation printed beside the value.

Re-running it buys one thing the carried file cannot: the fresh `GEN_IDS` are
compared to the recorded ones from job `2644106f-…` and the result is printed as
`ORACLE_REPRO=YES|NO`. A `NO` is a finding about the oracle, not about us, and
it would void the denominator; the predecessor could not make that check because
it had nothing to compare against.

The degenerate `n_gpu_layers = 0` oracle leg is **not run**. It is a failed
instrument owed by [#2557](https://github.com/mudler/vllm.cpp/issues/2557), it
emitted one repeated token on 5 of 6 prompts, no conclusion rested on it, and
running it again would spend 40 minutes of a shared lease to re-observe a known
defect. `compare_tiers.py` takes the HIP oracle on both of its oracle inputs and
its `A vs C` column therefore restates `A vs B`; the evidence says so.

## The chain of custody

Re-established on this host rather than inherited. The harness's detokenized
prompt-plus-generation for prompt 0 is compared byte for byte against a stock
`llama-completion` run on the same recipe, from the same build, in the same
lease. `llama-completion` is the binary the 2026-08-23 run used; `llama-cli` at
this pin is the conversation tool and applies a chat template, so it cannot
reproduce a raw-completion recipe. A `DIVERGE` there voids the oracle side and
the job says so instead of reporting a gate result.

`llama-cli` is never invoked without `-no-cnv` and never with an open stdin. One
such run wrote 24.9 GB to the share.

## Confirming the legs, not assuming them

The design is **six independent legs**, each its own process and its own model
load, on the shipped default configuration, with no environment knob set beyond
`VT_OP_PROVIDER_STATS=1`. N is 6 by design and the record carries
`failures / 6`. The tally is derived from the design and from deduplicated leg
lines, never from a printed summary line: a summary on this row has already
inverted its own results by reading the wrong field, and a leg tally on the
sibling row was wrong by a factor of two because the log prints each leg twice.

The three-outcome classifier survives: `OK`, `BOARD_FAULT`, `HARNESS_ERROR`. A
run whose legs die of harness errors refuses to quote a fault rate at all. The
first attempt would have reported five board faults over a read-only directory
without it.

**Reproducibility remains a precondition of the verdict, not a diagnostic**, and
the reason is narrower now but not gone. The fixed allocator removes the
mechanism that could hand back wrong bytes without faulting, which is exactly
why the predecessor refused a lone survivor. That refusal is kept anyway,
because the argument for it was never "the allocator is broken" but "survival is
not independent of correctness while anything can corrupt memory", and this run
cannot prove the second clause about a board it is measuring for the first time
in this configuration.

So: **two or more clean legs that agree with each other, or no verdict.**

## Risks

- **The board faults anyway.** It would contradict the 10-leg post-fix result
  and it is important news. Reported as `failures / 6` with the signatures,
  never smoothed.
- **A silent CPU fallback.** `gfx1151` is integrated, so the portable reference
  tier is reachable and a fallback would be invisible in the ids. Every leg sets
  `VT_OP_PROVIDER_STATS=1` and the job fails on a non-zero reference-tier hit
  count or a zero `device=5` count. This matters more than it did: `27da7787e`
  withdraws the host-addressability claim on this part, so the tier's
  eligibility has moved and the assertion is now measuring a changed thing.
- **A stale binary.** Addressed by the four assertions above.
- **A previous run's log read as this run's verdict.** The output directory is
  new, logs are deleted before submission, and every wait keys on the `rc` job
  UUID.
- **The pod lost `/tmp`.** The worker's staged trees, its podman image and its
  llama.cpp build all live under `/tmp`. The job re-stages each of them from the
  share when absent rather than failing, and prints which path it took.

## Evidence

`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`,
following the conventions of the predecessor document and of the 2026-08-23
CPU-tier document: what ran, the measured identity chain, the fault rate out of
N, the oracle's pinned executed path and its chain of custody, the gate, the
tier comparison, and what is not admissible.

Raw logs on the share under `/mnt/nas_share/rc/rocm-tokgate-strix-v2/`.

## How a same-rate, different-index result is to be reported

Carried forward unchanged from the predecessor spec, and fixed here before the
data, because it is exactly the finding a reader would want massaged afterwards.

The CPU-tier work has measured that under perturbations which did **not** move
the divergence rate, the first-diff INDEX moved readily. The index is therefore
not a stable signal on its own. What is stable is the RATE and the SET of
near-tie steps.

- **Same rate, same indices, same ids** — the strongest available evidence for
  one shared term in the quant path.
- **Same rate, different indices** — reported as exactly that, and NOT as
  evidence of a ROCm-local cause.
- **Different rate** — a real difference in magnitude between the tiers, and the
  first thing worth chasing.

No result is to be forced onto the recorded `- / 34 / 20 / - / 14 / -` pattern.
A same-rate, different-index answer reported plainly is a better result than a
match argued into existence.

## Stop conditions

- **Every leg faults**: `TOKEN_GATE=NOT_MEASURABLE` with the count out of 6, and
  the contradiction with the post-fix 10-leg result stated as the headline.
- **The clean legs disagree with each other**: `TOKEN_GATE=NOT_MEASURABLE`. A
  non-deterministic arm cannot be scored for token-exactness.
- **Fewer than two agreeing clean legs**: `TOKEN_GATE=NOT_MEASURABLE`. A lone
  survivor is not a verdict.
- **The oracle's chain of custody is not `EXACT` or `PREFIX`**: stop; the oracle
  side is not bound to the stock binary.
- **`ORACLE_REPRO=NO`**: report it and do not score against either file until it
  is explained. The denominator moved.
- **A reference-tier hit count above zero on our arm**: stop; the leg did not
  measure the ROCm tier.
- **Anything that would need a speed number to interpret**: stop, and say that
  the question is not answerable before the gate passes.

## Owed

- **The ROCm-local divergence term at p1/45, p3/45 and p4/14.** The divergences
  did NOT match the CPU tier's, so ROCm carries a term of its own and it has its
  own issue against this row:
  [#2590](https://github.com/mudler/vllm.cpp/issues/2590).

  **ANSWERED 2026-09-02, and one third of it was not ours.**
  [`rocm-tier-hidden-state-bisect.md`](rocm-tier-hidden-state-bisect.md) ran the
  per-layer comparison on ONE host, from one `libvllm.so`, with no oracle in it.
  `p4/14` does not reproduce: our x86-64 CPU tier and the ROCm tier emit all 48
  ids identically, so that step was this document's CPU comparison point crossing
  an ARCHITECTURE (`thor`, aarch64) and not a ROCm term at all —
  [#2608](https://github.com/mudler/vllm.cpp/issues/2608) owns it.
  `p1/45` and `p3/45` reproduce exactly and are the arithmetic: the final
  cross-tier `rel_l2` is 1.03 to 1.11 times what two independent bf16 residual
  streams must produce, the run-to-run floor is exactly zero, and every flip sits
  at a step whose own margin is below the cross-tier logit delta. **There is no
  layer or op to fix.** This row's gate is unchanged and still `FAIL`.
- #2534's residual magnitude term still owns the CPU tier's three steps. It does
  not explain ROCm's.
- #2497's quant-matched decode number stays blocked until this gate passes.
  `AGENTS.md` §Gates admits no speed, latency or memory axis from this arm, and
  none was produced or recorded here.
- The degenerate `n_gpu_layers = 0` oracle leg stays owed by
  [#2557](https://github.com/mudler/vllm.cpp/issues/2557). It was not run and no
  conclusion here rests on it.
