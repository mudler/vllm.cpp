# Spec — the llama.cpp denominator for Qwen3.8-27B Q4_K_M on gfx1151

Row `BACKEND-GATE-ROCM-LLAMACPP`. Issue
[#2497](https://github.com/mudler/vllm.cpp/issues/2497), which owns the
quant-matched decode number on `strix:gpu0` and already carries one retraction
for taking a cross-engine number ahead of the correctness gate.

Sibling records: [#2381](https://github.com/mudler/vllm.cpp/issues/2381) (AMD
clock recording, which no in-tree harness does), and
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) (the allocator defect
whose fix made this board finish a leg at all).

Predecessor evidence:
[`rocm-strix-qwen38-q4km-20260901.md`](../../docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md),
whose speed axis is marked INADMISSIBLE and quotable as nothing.

## Now

The oracle side is measured and recorded. Our arm's side is staged and
deliberately unrun.

**The denominator is `12.233 tok/s`**, the median of 6 legs, spread `0.303%` of
the median, 6 of 6 legs `rc=0`, taken on 2026-09-02 in `rc` job
`ff18a029-cd10-42d1-a5f7-9129c1c8af09`. It reproduces the 2026-09-01 lease's
three legs on a different lease. The executed path is pinned by the source
content manifest and by the sha256 of every binary and shared object the run
linked, because this oracle's greedy decode is not deterministic across its own
kernel paths and the revision alone under-specifies it.

**The custody chain has three links, two closed and one open, and the records
say which is which.** Link 1, upstream commit to source content, is CLOSED:
the job only compared the staged tree against a constant in its own script,
which fixes the tree and not its provenance, so the link was closed separately
by reproducing the manifest from `git archive` of `10bf611e…aebd8` out of
`ggml-org/llama.cpp`, where the tag `b10451` resolves to that commit and the
file count matches at 3,425 under `LC_ALL=C` on both sides. Link 3, compiled
tree to executed bytes, is CLOSED by the sha256 of every binary and shared
object the run linked. **Link 2, source content to those bytes, is OPEN.** No
compiler ran in this lease — `job.log:44` reads
`llamacpp_build=ALREADY PRESENT`, there is no `llamacpp_build_rc` and no build
log, and the step 4 and step 5 headers are one second apart — so the binaries
were carried from the 2026-09-01 lease, which records the identical `llama-bench`
and `llama-cli` sha256s, and whose source reached the box as a `git archive`
tarball whose only recorded identity is its own sha256. A tarball sha256 is not
a pure function of a commit, so no committed artifact ties the verified source to
the executed bytes. `build_commit` reading `unknown` is a consequence of staging
from a tarball with no `.git`, not an unpinned build.

The staged arm refuses on `STRIX_ARM_SPEED_RATIFIED_BY`, and the refusal is
proven rather than asserted: **14 mutations of the guard were applied and 14
were detected** by `tests/scripts/test_rocm_strix_ourarm_staged.py`. That is a
count over a CHOSEN set and it is not a completeness claim, which the tree then
demonstrated: a fifteenth mutation, weakening the issue-reference term
`#[0-9]+` to `#?[0-9]+`, was **not** detected — the suite stayed green at 13
passed while `STRIX_ARM_SPEED_RATIFIED_BY='ratified 2026-09-02 by the operator'`
went from `rc=3` to `rc=0`, degrading "naming means an issue reference" to
"contains any digit". The cause was the negative population: every long refused
value carried no digit at all, because a date is the digit a real ratification
always has. Five values that are long, carry digits and carry no `#` now kill
that mutant. **Read the number as a lower bound on detection, never an upper
one.**

Three of the original fourteen read
NOT DETECTED on the first pass and were repaired rather than recorded as passes
— the length floor and the issue-reference term were being tested as one
condition, and the reference-tier assertion was a substring test that
`VT_OP_PROVIDER_STATS_DISABLED=1` walked straight through. A fourth clause, a
bare `-z` test, was **deleted**: no mutation could break it, because the length
floor already decided the empty and the unset case, and an assertion nothing can
falsify is not a guarantee.

The staged arm's reference-tier assertion was itself defective when it was
written, and `tests/scripts/test_ltx25_ab_memwatch.py` caught it before it
landed. It read the count as `grep -c … || echo 0`, and `grep -c` prints its own
`0` and exits 1 on no match, so the fallback ran on top of the count and produced
the two-line value `0\n0`; an absent stderr capture also read as `0`, which is
exactly the clean reference-tier result the assertion exists to refuse. It now
tests the file is readable first and reports `UNREAD` when it is not, so a
missing capture can no longer pass for a clean one.

The `BACKEND-GATE-ROCM-LLAMACPP` matrix row stays `INVENTORIED`. Nothing here
makes the ROCm floor gateable: a floor is a comparison, and this row has only
one side of one. What the row's record gains is a measured denominator and the
withdrawal of a ratio it was still quoting after that ratio was retracted.

## What this spec is for, and what it refuses to do

`AGENTS.md` §Gates admits a performance result from an arm only after that arm's
declared token-exact gate passes. Our ROCm `gfx1151` arm's gate is
`TOKEN_GATE=FAIL`, 3 of 6 prompts
([`qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)),
so **no throughput, latency or memory figure for vllm.cpp may be produced,
recorded or quoted by this row**, and no ratio may be computed. #2497's first
comment did exactly that and had to be withdrawn.

What is admissible without our arm's gate is the **oracle alone**. llama.cpp's
own decode speed on this board is a single-engine fact. `AGENTS.md` §Gates and
§"When vLLM has no implementation" both require an oracle to demonstrably build
and run the model before it is trusted as a denominator, and
`.agents/benchmarking.md` says the same in one line: "Prove the oracle actually
*runs* the model before trusting it as a denominator." That proof is this row's
deliverable, and it is worth taking now because it is the half of #2497 that our
correctness state does not block.

The second deliverable is the other half, **staged and refusing to run**. When
the token gate is ratified, the paired measurement should take one lease and not
a design session. Staging it now, in the tree, under review, is cheaper than
staging it under time pressure later — and a staged script that can be launched
by accident is how #2497 got its retraction, so this one refuses to start.

## Scope

Two artifacts, one measured and one inert.

1. **Measured — the oracle.** llama.cpp at the recorded pin `b10451` =
   `10bf611e533d81f739128304991c5e133c6aebd8`, `gfx1151` binaries **carried into
   the lease from the 2026-09-01 one, not built in it** (`job.log:44` reads
   `llamacpp_build=ALREADY PRESENT`), decoding the plain
   `Qwen3.8-27B-Q4_K_M.gguf` on `strix:gpu0`. Repeated warm
   legs, order recorded, executed kernel path pinned, AMD clock state sampled per
   leg by an ad-hoc sampler that says it is ad-hoc.
2. **Staged, not run — our arm.** `.agents/scripts/rocm-strix-ourarm-staged.sh`,
   the order-alternated paired job that becomes admissible when the token gate
   passes. It refuses to start unless `STRIX_ARM_SPEED_RATIFIED_BY` names the
   decision that ratified it.

### Artifact

`unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10`,
`Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, sha256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. The sha256
is verified **on the worker**, against the staged worker-local copy, before any
timing runs, and the job aborts on a mismatch rather than reporting one.

The `UD` family is not used. Our loader could not read it when #2497 was opened
([#2510](https://github.com/mudler/vllm.cpp/issues/2510)) and the published
`UD-Q4_K_XL` bytes have moved in place under an unchanged name, so it is not a
pin. The 12.616 tok/s harness-sanity figure the predecessor took on
`UD-Q4_K_M` is a **different artifact** and is context, not this row's number.

### Excluded

- Running our engine for timing, on any leg, for any duration.
- Computing, writing or implying a ratio between the two engines.
- `HSA_ENABLE_SDMA`. It is retired (#2511) and is not set anywhere.
- Any change to engine code.

## Pinning the oracle's EXECUTED PATH, not only its revision

`.agents/oracles/llama-cpp.md` records that `b10451`'s decode is not
deterministic across its own supported kernel paths, and
`rocm-gfx1151-q4k-token-gate-v2.md` §"Pinning the oracle's EXECUTED PATH"
measured a second instance of that on this very board. A denominator that pins
only the revision is under-specified, so the run records:

| Term | How it is obtained |
|---|---|
| Source identity | content manifest of the staged tree, `LC_ALL=C` collation printed beside the value, compared to the recorded `56c26d15…` |
| Binary identity | sha256 of `llama-bench` and `llama-cli`, plus every `libllama`/`libggml` shared object |
| `n_gpu_layers` | the `-ngl` argument, echoed back out of `llama-bench`'s own JSON |
| `use_extra_bufts` | source-anchored: `src/llama-model.cpp:2489` defaults it `true` and `tools/llama-bench/llama-bench.cpp:1218-1267` sets no override, so this run's value is `true`; confirmed at runtime by the `load_tensors:` buffer-type lines |
| `system_info` | `llama-cli -v`, which reaches `common.cpp:417`'s trace-level emission of `common_params_get_system_info` |
| host arch, thread count | `uname -m`, `nproc`, and `llama-bench`'s own `n_threads`, `cpu_info`, `gpu_info` and `backends` JSON fields |
| enumerated devices | `llama-bench --list-devices` |

`llama-bench` exposes no switch for `use_extra_bufts`, which is why that row is
answered from the source rather than from a flag. With `-ngl 99` every loaded
tensor sits in the ROCm buffer, so the CPU repack buffer types the flag admits
hold nothing and the flag cannot move this measurement. The run prints the
buffer lines rather than asserting that.

`llama-cli` is never invoked without `-no-cnv` and never with an open stdin. One
such run wrote 24.9 GB to the share. `llama-bench` and `llama-cli` link
`*-impl.so` beside them, so `build-llamacpp/bin` is on `LD_LIBRARY_PATH` or they
exit 127.

## Design of the measured run

- One `rc` lease on `strix:gpu0`, submitted detached, one job at a time, and the
  fleet's contention state recorded from `rc devices` at submission.
- The GGUF is staged from CIFS `/workspace` to worker-local `/tmp` and its
  sha256 verified there. `/workspace` is never a run surface.
- **6 legs**, each one `llama-bench -m … -p 0 -n 64 -ngl 99 -r 3 -o json`. A leg
  is one process: it loads the model, does `llama-bench`'s own warmup, and times
  3 generations of 64 tokens. `N = 6` comes from this design and not from
  counting log lines. `grep -c 'avg_ts' job.log` reads 13 against six legs,
  because the per-leg echo, the fold's `population` string and the whole
  `RESULT.json` all land in the same log. Re-emission is the mechanism, not
  `tee`, which writes each line once. `fold.py` refuses in BOTH directions
  against the legs actually on disk, so the design cannot be lowered to drop
  one.
- Legs run in sequence 1..6 and the sequence index is recorded with each. There
  is one engine, so there is nothing to alternate; the order is recorded so a
  monotonic drift is visible as one.
- The per-leg figure is `avg_ts` from that leg's own JSON. The reported figure is
  the **median over the 6 legs**, with the min-max spread stated.
- A leg is discarded only for a named cause, and the cause is printed with it. No
  leg is discarded for being the first one unless it differs, in which case the
  named cause is the cold page cache and it is stated as such.
- AMD clocks are sampled at 4 Hz per leg from
  `/sys/class/drm/card*/device/{gpu_busy_percent,pp_dpm_sclk}` to worker-local
  disk, because a 4 Hz flush at CIFS stalls the sampler and distorts the sample
  spacing the window is judged on.

### The clock, and why two windows are reported

No in-tree harness samples AMD clocks. `tools/bench/gpu_clock_state.py` reads
NVIDIA fields and does not run on this board; #2381 owns the gap. So the sampler
here is **ad-hoc**, carried beside the job as the predecessor's was, and every
figure it produces is labelled ad-hoc rather than presented as a gate reading.

Two windows are reported per leg and they answer different questions. The whole
window includes the 17 GiB model load, which on this APU dominates the raw
spread and says nothing about decode. The **>=90%-busy compute window** is the
part where the GPU is actually generating. Both are printed, with sample counts,
so a reader can see which one a figure came from.

`.agents/benchmarking.md`'s 5% within-run SM-clock spread ceiling is **not
applied**. It was calibrated on a datacenter part with persistence mode and its
own power budget; this APU shares one package power budget with 32 CPU cores and
**no compute window either run has ever sampled on it satisfies that ceiling**:
8.3–11.0% across these six legs, and 8.2–12.0% across the predecessor's five
(8.2 and 12.0 being our arm and a harness control, 9.5–10.7 the three llama.cpp
legs). Those are two different ranges, and the evidence document said they were
one until 2026-09-03. The argument is unchanged at the true numbers, and
stronger: eleven of eleven windows, two engines, two leases, an idle box, all
between 1.6x and 2.4x the ceiling. An AMD rule needs its own calibration and
cannot inherit the NVIDIA number. That observation is recorded for #2381 and is
not repaired here.

The cross-arm clause of that section — that two arms' clock medians and means
must agree, because the offset is the term that lands in the ratio — has no
subject in this run, because there is one arm and there is no ratio.

## The staged run, and why it refuses

`.agents/scripts/rocm-strix-ourarm-staged.sh` carries the paired design in full:
order-alternated rounds, the leg count, the clock window per leg,
`VT_OP_PROVIDER_STATS=1` asserted to zero CPU reference-tier hits, the artifact
sha verified on the worker, and the same ad-hoc clock sampler.

It refuses to start unless `STRIX_ARM_SPEED_RATIFIED_BY` is set to a value
naming the decision that ratified the arm's speed axis — a value carrying an
issue reference, so `=1` is refused. The refusal is the first thing the script
does, before it reads a path, stages a byte or touches a device, and it names the
gate that is failing and the evidence that records it.

The script computes **no ratio**, and says why in its own output: the ratio is
arithmetic a reader can do, but the decision that it is admissible is not, and
that decision is exactly what the env var is asserting. Printing the two medians
without dividing them keeps the assertion where a human made it.

## Risks

- **The board faults.** #2511 landed and the token gate then completed 6 legs of
  6. A fault rate here would contradict that and is reported as a finding rather
  than retried into silence.
- **The pod restarted and dropped the build.** The job verifies the binaries and
  rebuilds from the pinned tarball when they are absent; it never assumes them.
- **A leg measures a shared box.** The fleet state is recorded at submission and
  the job runs inside the lease, one job at a time.
- **The sampler retains nothing.** An empty or entirely idle window is reported
  as such and the clock figure is withheld; it does not silently become a
  narrower one.

## Tests

The staged script's refusal is the one behavior in this row that a test can
falsify, and `tests/scripts/test_rocm_strix_ourarm_staged.py` does that:
unset, empty, and a value with no issue reference each refuse with a non-zero
status and a message naming the token gate; a well-formed value gets past the
guard. The test is red before the script exists and each assertion is mutated to
prove it detects the defect.

**It runs on a lane and not only on a developer box.** The suite was registered
in `scripts/agent-preflight.sh` alone, and CI enumerates its
`tests/scripts/test_*.py` steps explicitly with no glob, so the row's one
falsifiable behaviour was enforced by a locally-run preflight. It now has its
own CI step. Nothing in it can skip: `bash` and the standard library over one
committed script, no GPU, no lease, no toolchain, no network.

The measured run's own assertions are inside the job — the artifact sha, the
source manifest, the binary presence, the leg count — and each aborts rather than
reporting.

**`fold.py`'s leg-count refusal runs in BOTH directions**, and against what is
on disk rather than against its own loop. It refused `--legs 7` and a forced
`rc=1` leg, and reported `MEASURED` with `rc=0` on `--legs 5`, silently folding
five and ignoring `leg6.*` entirely: a leg could be dropped by lowering the
declared design, which is the one failure a leg-count guard exists to stop. It
also carried an assertion (`if len(legs) != args.legs`) that no mutation could
ever break, because `legs` was built by a loop of exactly `args.legs`
iterations. The directory is now enumerated and compared against the design in
both directions, and the unfalsifiable assertion is gone rather than kept for
its comment value. Proven by execution, on a scratch copy so the committed
`RESULT.json` was never the write target:

| Input | before | after |
|---|---|---|
| `--legs 5`, six legs on disk | `MEASURED`, `rc=0` | `INCOMPLETE`, `rc=3`, names leg 6 |
| `--legs 6` | `MEASURED`, `rc=0` | `MEASURED`, `rc=0` |
| `--legs 7` | `INCOMPLETE`, `rc=3` | `INCOMPLETE`, `rc=3`, names leg 7 |
| `--legs 6`, `leg3.*` deleted | — | `INCOMPLETE`, `rc=3`, names leg 3 |
| `--legs 6`, `leg2.rc` forced to 1 | `INCOMPLETE`, `rc=3` | `INCOMPLETE`, `rc=3` |

Re-folding the committed evidence at `--legs 6` with the repaired script
reproduces the committed `RESULT.json` **byte for byte**, so no statistic moved.

**`docs/bench-evidence/<run-id>/` artifact kinds now classify.** `fold.py`,
`amd_clock_sample.py`, six `clock-leg*.jsonl` and six `leg*.rc` — all fourteen
— had no class in `scripts/check-pr-size.py`, which fails closed. Its suite
sweeps every TRACKED path, so this branch was red against the whole tree and
landing it would have reddened `main` itself and refused every later change
touching that checker through its own evidence contract. It went unseen because
the checker is CI-only and `agent-preflight.sh` skips it. Repaired the way the
checker's own comments prescribe: NAME the artifact kinds, keep the extension
list a closed enumeration, do not relax the directory shape.

**Two of the fourteen were repaired by somebody else while this branch was in
review.** #2629 landed a `py` arm on 2026-09-03 for
`docs/bench-evidence/gdn-chunked-decomposition-20260902/`, hours before this
merge, which classifies `fold.py` and `amd_clock_sample.py` too. Same failure,
same day, two directories, found independently — which is the argument for the
rule the checker's comments state, not against it. **After merging `origin/main`
this row's own remaining contribution is twelve paths, not fourteen: the six
`.jsonl` and the six `.rc`**, and every number below is restated at that base
rather than left standing from before the merge. The `.py` assertion is kept as
a regression guard and is labelled as green-at-base, because these two files are
this row's own and a later narrowing of that arm would take them silently.

The two arms are also different in kind, and the comment says so rather than
folding them together: `.py` is the RECIPE, and `.jsonl` and `.rc` are the job's
OUTPUT. The merge also brought a correction this row must not undo — #2629
found the `.sh`/`.cu` premise "nothing outside `docs/` references this
directory" to be false as a universal claim, so the narrow form is re-checked
for these two arms and the universal form is not repeated. Verified: no tracked
file outside `docs/` names a `docs/bench-evidence/…*.jsonl` or `…*.rc` path at
all.

## Gates

```sh
python3 -m pytest tests/scripts/test_rocm_strix_ourarm_staged.py -q
python3 -m pytest tests/scripts/test_check_pr_size.py -q
python3 scripts/check-pr-size.py --base origin/main --head HEAD
scripts/agent-preflight.sh
```

The two `check-pr-size` lines are listed because **preflight skips that
checker**, which is how an unclassified path reached a fresh review. Run them by
name or they do not run at all.

No gate here measures the GPU. The measured figure is evidence, not a gate; a
number this row produces about the oracle cannot be re-derived by CI.

## Evidence

[`docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md`](../../docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md).

**The checker change's red-before and green-after.** `check-pr-size.py` is a
semantic checker change, so it carries both.

Taken FIRST against the pre-merge base, and RE-TAKEN against `origin/main`
after the merge, because a merge can falsify a sentence inside the same branch
and this one did.

```
RED, pre-merge base (before #2629 landed its `py` arm)
  $ python3 scripts/check-pr-size.py --base origin/main --head HEAD
  ERROR: PR size check could not classify the change: unclassified repository
  path 'docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902/amd_clock_sample.py'
  rc=1
  $ python3 -m pytest tests/scripts/test_check_pr_size.py -q
  AssertionError: Lists differ: [...] != []   (14 additional elements)
  1 failed, 6 passed, 39 subtests
  base 5,631 tracked / 0 unclassified; head 5,660 / 14

RED, post-merge base (origin/main's checker over the merged tree)
  5,682 tracked / 12 unclassified — six clock-leg*.jsonl, six leg*.rc.
  `fold.py` and `amd_clock_sample.py` now classify at base; #2629 did that.

GREEN (head checker, merged tree)
  $ python3 scripts/check-pr-size.py --base origin/main --head HEAD
  OK: every explicit path class is within its review budget.   rc=0
  $ python3 -m pytest tests/scripts/test_check_pr_size.py -q
  56 passed, 169 subtests
```

`test_a_measurement_runs_own_artifact_kinds_classify_as_evidence` is the paired
evidence. Its `.jsonl` and `.rc` assertions raise `ValueError` against the base
checker, verified by loading `origin/main`'s `check-pr-size.py` directly rather
than by reading it; its `.py` assertion is labelled a regression guard and is
green at base as well as at head. It also pins the directions that must NOT
move — `leg1.pickle`, a flat `docs/bench-evidence/*.jsonl` and a `.rc` outside
the directory all still fail closed, and `leg1.json` stays `public_document`,
because the evidence arm is tested before `DOC` and admitting `.json` there
would silently reclassify `docs/bench-evidence/mxfp4-qwen`'s golden.

**The guard mutation the original set missed.** Weakening `#[0-9]+` to
`#?[0-9]+` in the staged script left the suite green at `13 passed` while
`STRIX_ARM_SPEED_RATIFIED_BY='ratified 2026-09-02 by the operator'` went
`rc=3` → `rc=0`. With the five added negative values the same mutant is caught
by `5 failed`. Both runs were taken with the mutation verified present on disk,
and the tree was restored byte-for-byte after each (`git diff --stat` empty).

**The count of machine-checked claims is not quoted, and that is the
reconciliation.** The landing commit body of 2026-09-02 said "292 machine-checked
claims, 0 mismatches"; the fresh review's own verification reported 331 over the
same artifacts. Neither is wrong about the tree — they bucketed a table cell,
a per-leg triple and a sub-field differently — and neither is reproducible from
anything committed, so the difference is in the counting and not in the
evidence. What IS reproducible is stated instead: every transcribed number is
re-derived from `leg{1..6}.json` and the raw clock samples without running
`fold.py`, with **0 mismatches**; two independent re-derivations agree exactly;
and a clean re-fold of the committed directory reproduces `RESULT.json` byte for
byte. A count nobody can redo is the shape `.agents/benchmarking.md` warns
about, where a number quoted often gets treated as measured.

**The forbidden figure is deleted and not restated.** The staged script carried
a residency figure attributed to our loader on the `blk.64` line. This spec
forbids this row producing, recording or quoting ANY throughput, latency or
memory figure for vllm.cpp, with no exception written, and "how much our loader
holds" is one however casually it is phrased. The warning to a future reader
stays; the number is gone from the script, from this spec and from the pull
request body, because repeating it to explain its removal is still quoting it.
`git log -S` recovers it when the token gate passes and the figure becomes
admissible.

**Evidence curation is now recorded.** `system-info-extract.txt` is a hand-made
selection that no step of `job-as-run.sh` produces, and the `use_extra_bufts`
and tensor-count arguments rest on it. The 348,370-byte source it derives from is
committed as `system-info.err.gz`, and the evidence document carries the command
that checks the derivative against it byte-for-byte. `CPU_REPACK` appears **0
times** in all 4,952 lines of that source, so the argument no longer depends on
the curation at all. `rc-devices-before.txt`, the fleet state at submission this
spec requires, sat UNTRACKED in the worktree root while the document quoted it;
it is committed beside the other artifacts — the same defect class as the
`job.log` gitignore bug this row already repaired.

## Stop conditions

- Any instruction, temptation or convenience that produces a vllm.cpp throughput
  number on this board stops the work. The gate is failing and the number is
  inadmissible.
- A ratio. If a ratio is about to be written, the row has left its scope.
- A single-leg figure. A denominator reported from one leg is an anecdote.
- The board faulting, which is reported at its measured rate.

## Owed

- The paired measurement itself, once the token gate passes.
  [#2497](https://github.com/mudler/vllm.cpp/issues/2497) owns it and the staged
  script is what it runs.
- An AMD clock helper with its own calibrated spread rule.
  [#2381](https://github.com/mudler/vllm.cpp/issues/2381) owns it; every clock
  figure this row produces is ad-hoc until then.
- **Link 2 of the custody chain**, source content to executed bytes. Closing it
  needs a lease that compiles and records its own compiler invocation, which
  this row did not take and did not need for a single-engine figure.
  [#2497](https://github.com/mudler/vllm.cpp/issues/2497) owns it and gets it
  for free on the retake, which cannot reuse today's binaries.
- **`scripts/check-agent-record.py:463`'s stale comment**, which still calls this
  row "no spec of its own". Not repaired here, and the reason is a real trap
  rather than a preference: `check-pr-size.py` requires every change to a
  governance checker to carry semantic mutation evidence, and a comment has no
  semantics to mutate, so any paired test written for it passes against the base
  checker too and the contract reports it as not being evidence. Fabricating a
  mutation for a comment is worse than the stale comment.
  [#2631](https://github.com/mudler/vllm.cpp/issues/2631) owns it and proposes
  the shape: exempt only when the two revisions are equal with comments and
  docstrings stripped.
