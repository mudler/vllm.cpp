# cuBLASLt cross-process algo stability (#2750)

Status: **SPEC ONLY. The harness is committed and has NEVER RUN on a device.**
No number in this file is a measurement; every one of them is a threshold
written before the measurement, which is the order this protocol requires.

Owning row: `KERNEL-GEMM-BF16`

Issue: [#2750](https://github.com/mudler/vllm.cpp/issues/2750)

Harness: [`scripts/dgx-gemm-tactic-draw-survey.sh`](../../scripts/dgx-gemm-tactic-draw-survey.sh)
and [`tools/bench/gemm_tactic_draw_survey.py`](../../tools/bench/gemm_tactic_draw_survey.py)

## Scope

Measure one thing: **does one cuBLASLt selection key answer with one algo
config across FRESH PROCESSES on one device, one driver and one binary?**

`src/vt/cuda/gemm_plan_cache.h:22` and `src/vt/cuda/fp8_plan_cache.h:17` both
justify caching the `cublasLtMatmulAlgoGetHeuristic` result with a
**per-process** determinism premise. That premise is exactly strong enough for
what those caches claim, because within one process a cache changes *when* the
heuristic runs and never *what* it returns. Nothing in the tree states or
measures the cross-process question, and no cache persists the selection, so
every process re-resolves it.

**In scope**

- N fresh processes, one binary, one device, one driver, one workload, with
  `VT_GEMM_ALGO_LOG=1`.
- A per-key comparison of `algoId` / `tile` / `stages` / `splitK`.
- Two populations, because they answer different questions and either alone can
  mislead. The **draw** population is N processes whose NVFP4 plan maps differ
  (they are the same processes #2751 needs); the **frozen** population is the
  repeated scoring legs of ONE draw, in which the FP4 map is byte-identical and
  process identity is the only variable. A disagreement between the two is
  itself a finding and is reported rather than folded away.

**Explicitly out of scope**

- Persisting the cuBLASLt heuristic. #2750 states this by name: persistence is
  warranted only in the unstable case, and building it before the measurement
  would be the feature this issue exists to refuse.
- Any statement about the cuBLASLt heuristic beyond the selection keys the
  workload actually produced. The report carries a `scope` field saying so, and
  a shape absent from `keys_common` is unmeasured, never stable.

## Upstream chain

vLLM defines the behaviour wherever it implements it, and here the executing
chain runs into a closed CUDA library. What is anchorable, and what is not:

| Concern | Anchor | What it settles |
|---|---|---|
| The selection call | `cublasLtMatmulAlgoGetHeuristic`, reached from `src/vt/cuda/cuda_matmul.cu` at the three bf16/f32 sites and the two fp8 sites | the function whose determinism is in question |
| The diagnostic we read | `src/vt/cuda/gemm_algo_log.h` and `MaybeLogGemmAlgo` at `src/vt/cuda/cuda_matmul.cu:248` | our own mirror of what upstream logs under `CUBLASLT_LOG_LEVEL` and torch `_scaled_mm` verbose; we have no torch, so this flag is the equivalent |
| The premise being tested | `src/vt/cuda/gemm_plan_cache.h:22`, `src/vt/cuda/fp8_plan_cache.h:17` | the claim, in our own words, that the measurement checks |
| The pinned oracle | [`.agents/upstream-sync.md`](../upstream-sync.md) | the revision any later vLLM-side comparison must be taken at |

**The vLLM-side half is NOT claimed here and is owed.** Whether the pinned
oracle re-resolves the same heuristic per process, and with what result, has not
been measured, and this spec does not assert it. Reading torch's caching as an
answer would be reasoning about the executing chain from its documentation
rather than from a trace, which is what
[`.agents/porting.md`](../porting.md) refuses. `## Owed` names it.

### `VT_GEMM_ALGO_LOG` has SIX emit sites and only ONE of them is a selection

All six are behind the same `GemmAlgoLogEnabled()`, so a run that turns the
instrument on gets all six in one stderr:

| Site | Line | Shape |
|---|---|---|
| `MaybeLogGemmAlgo` | `cuda_matmul.cu:270` | **the selection**, one per unique key |
| `MaybeLogFp8PlanRefusal` | `cuda_matmul.cu:805` | `REFUSED`: a plan that was never built |
| `MaybeLogDenseAlgoCandidates` | `cuda_matmul.cu:842` | `DENSE-CANDIDATES`, on a failed heuristic query |
| `MaybeLogDenseAlgoCandidates` | `cuda_matmul.cu:858` | `DENSE-CANDIDATE`, ONE PER RANKED CANDIDATE |
| `MaybeLogFp8AlgoCandidates` | `cuda_matmul.cu:883` | `CANDIDATES`, on a failed heuristic query |
| `MaybeLogFp8AlgoCandidates` | `cuda_matmul.cu:900` | `CANDIDATE rank=`, ONE PER RANKED CANDIDATE |

Only the selection line carries `a=`, `b=`, `c=` and `epilogue=`, and that is
what the judge requires before it reads a line as a selection. This is not a
parsing detail. The dense dump is reached from the TN-bt lane
(`cuda_matmul.cu:582`) -- the bf16 lane this row is about -- and it repeats one
shape once per candidate, so a parser that accepts any `[VT_GEMM_ALGO]` line
keeps RANK 0 of a heuristic LIST and compares that across processes. Two
processes whose candidate lists came back in a different order then report
`UNSTABLE`: the verdict that, per the table below, turns this row into a
benchmark-validity repair and licenses the persistent cuBLASLt cache #2750 says
must be built ONLY in its case 4.

## Our baseline

| Quantity | Value | Where it comes from |
|---|---|---|
| Cross-process selection agreement | **NEVER MEASURED** | this is the gap #2750 files |
| Per-process determinism | asserted, not measured | `gemm_plan_cache.h:22`, `fp8_plan_cache.h:17` |
| The premise's cited evidence | **UNREACHABLE** | `fp8_plan_cache.h:19` cites `.agents/state.md`, which does not exist; the protocol retired the state log |
| The analogous confounder one lane over | REAL, and removed by freezing | paired NVFP4 runs shared 18--33 of 64 tactic IDs and 5 of 6 output hashes differed, against 64/64 once frozen ([`nvfp4-persistent-plan-cache.md`](nvfp4-persistent-plan-cache.md) §"W3-C3 corrected frozen-plan component result") |

That last row is why this is worth a lease rather than a shrug. The same class
of uncontrolled per-process selection was real on the NVFP4 lane, it did not
show up as an error, and it was only visible as spread.

## Port and harness map

Nothing is ported. Both instruments already ship, and this row adds only the
driver and the judge.

| Piece | File | Role |
|---|---|---|
| lease-side driver | `scripts/dgx-gemm-tactic-draw-survey.sh` | toolkit, source, CUTLASS, one `-j 4` build in `/tmp`, checkpoint staging, preflight, draws, scoring legs, reduction |
| judge | `tools/bench/gemm_tactic_draw_survey.py` | parsers, preconditions, `algo_stability`, the verdict; standard library only, `--dry-run` walks it with no device |
| leg ledger | `tools/bench/c8_leg_runner.py` and `tools/bench/resumable_legs.py` | interleaved plan, append-on-completion, resume, cross-boot refusal, terminal control |
| clock attribution | `tools/bench/gpu_clock_state.py` | one window per leg, opened by the leg itself |
| checkpoint staging | `scripts/rc-stage-checkpoint.sh` | one verified copy off CIFS, never a gate model read from the share |

## Tests to port

**None, and the absence is the finding.** No upstream test covers this: it is a
question about a closed CUDA library's heuristic, not about a vLLM contract, so
there is no upstream parameterization, fixture or tolerance to preserve.

The executable spec for the judging half is therefore local:

- `python3 tools/bench/gemm_tactic_draw_survey.py draw --dry-run` walks the
  record / resume / precondition path with a fixture whose bytes are the
  instruments' own format strings, on a host with no GPU.
- Every predicate is mutated against that fixture, and each mutation must turn
  the verdict red: one key dropped from one run makes `algo_stability`
  `INCOMPARABLE`; a single run is `INCOMPARABLE` rather than trivially stable; a
  candidate-dump line must not move the verdict at all.
- **The fixture moves ONE of its four shapes with the draw index and holds the
  other three still**, so the walk reads `UNSTABLE` with exactly one unstable
  key out of four. A fixture in which every draw agreed could not detect a
  tainted dedupe key: folding `algoId` into `algo_key` changed no verdict and
  survived the whole battery, and with the id in the key a genuinely unstable
  run would report `INCOMPARABLE` -- the two processes look like they saw
  different keys -- for ever. The three still shapes are the control.

## Gates

The commands are exact. They are run **by the operator, under a lease**; this
row's author has never held the device.

```sh
# 1. one job, all phases, resumable. --tactic-set defaults to `full`, which is
#    what the product ships (Fp4FullTacticsEnabled is on unless the value starts
#    with '0'); this row's question does not depend on the arm, but the arm is
#    recorded in every report so a reader never has to guess which one ran.
bash scripts/dgx-gemm-tactic-draw-survey.sh \
     --evidence /workspace/gemm-draw-survey/<stamp> \
     --src /workspace/gemm-draw-survey/src.tar.gz \
     --model /workspace/ckpt/<nvfp4-checkpoint> \
     --tactic-set full --draws 8 --score-reps 3 --concurrency 2

# 2. the judgement, offline, on the evidence the job left behind
python3 tools/bench/gemm_tactic_draw_survey.py reduce \
     --evidence /workspace/gemm-draw-survey/<stamp>
```

The verdict is the `issue_2750_draw_processes` block of `REPORT.json`:

| Verdict | Condition | What it closes |
|---|---|---|
| `STABLE` | at least 2 processes, identical key sets, and every observed key answered with ONE `(algoId, tile, stages, splitK)` tuple | #2750 closes with that evidence, the premise's dead `.agents/state.md` citation is repaired to point at this file, and **nothing is persisted** |
| `UNSTABLE` | any key answered with 2+ tuples | #2750 becomes a benchmark-validity repair on this row: every same-binary A/B on the bf16/f32 lane carried an uncontrolled variable, and `src/vt/cuda/nvfp4_persistent_cache.cpp` is the shape to reuse |
| `INCOMPARABLE` | fewer than 2 runs, a key missing from some run, or a SELECTION key repeated inside one process | no verdict; the processes did not run the same thing, or `LogOncePerKey` did not dedupe and only the first line per key was kept, so the comparison would run over a subset of what was emitted; the run is repeated |

**A refusal is a result and is reported as one.** The judge exits with a named
code before it prints any verdict when an instrument did not run: `70` for zero
`[VT_GEMM_ALGO]` lines, `71` when lines exist but none has a bf16 input (so the
run says nothing about this row), `76` when the processes saw different shapes,
`79` when they were not one binary, and `81` when one evidence root pooled both
NVFP4 tactic-set arms. A run that logged nothing must never read as a run that
found nothing wrong.

**Coverage is stated, never generalised.** `keys_bf16_input` and `bf16_keys`
name exactly which shapes were observed. Four shapes are four shapes.

## Dependencies

- A lease on `dgx:gpu0`. `rc run` or `rc hold`, never `ssh`
  ([`.agents/environment.md`](../environment.md)).
- A CUDA toolkit and a CUTLASS tree inside the worker. Neither is preinstalled;
  the driver resolves or repairs both and refuses by name (`38`, `36`).
- A checkpoint. **This row defaults none.** `--model` is required.
- `tools/bench/gpu_clock_state.py`, whose window cannot be pinned inside a lease
  (`LGC_RC=4`, [#1354](https://github.com/mudler/vllm.cpp/issues/1354)); the SM
  clock can only be sampled, so a pairing may be refused with no lever to fix it.

## Work breakdown

| Slice | State | Content |
|---|---|---|
| W1 harness + gates | **THIS CHANGE** | the driver, the judge, the mutation battery, and these thresholds, written before the run |
| W2 the run | owed to the operator | one lease, the command above, evidence under `/workspace` |
| W3 the verdict | blocked on W2 | close #2750 on `STABLE`, or re-scope this row on `UNSTABLE` |
| W4 record repair | rides with W3 | the two stale comments #2750 names, repaired in the flow that lands the answer |

## Risks and decisions

- **The workload decides the coverage.** An NVFP4 checkpoint exercises the FP4
  lane richly and the bf16 cuBLASLt lane only through whatever stays
  unquantized. The judge refuses a run with no bf16 key (`71`) rather than
  reporting on none, and a second arm on a bf16 checkpoint widens the key set if
  the first run's `keys_bf16_input` is thin.
- **The NVFP4 tactic-set arm is a variable this row does not control and must
  still record.** `VT_FP4_FULL_TACTICS` is default-ON and changes which CUTLASS
  candidates run, not which cuBLASLt shapes are queried, so it should not move
  this row's answer. "Should not" is a hypothesis: the arm is stamped on every
  report, and if the `full` and `w1` roots disagree on cuBLASLt stability that
  disagreement is a finding rather than a nuisance.
- **A model load per process is the cost.** Every draw and every scoring leg is
  a fresh process and pays one load. That is not overhead to optimise away: a
  process that reused a load would not be a fresh selection, which is the whole
  question.
- **The clock cannot be pinned in a lease.** The identity half of this row does
  not need a clock; the speed half on the NVFP4 row does, and may be refused on
  within-run spread. The two halves are reported separately so one refusal does
  not void the other.
- **Two mutexes that do not exclude each other have already voided a
  measurement here.** The lease is the mutex on a fleet device; nothing in this
  harness takes `$HOME/gpu.lock` instead of one.

## Now

`KERNEL-GEMM-BF16` does not change lifecycle state in this change. The harness
and the thresholds land; the measurement is owed.

## Owed

- [#2750](https://github.com/mudler/vllm.cpp/issues/2750) -- the measurement
  itself, and the two stale record defects it names:
  `src/vt/cuda/fp8_plan_cache.h:19` cites a retired `.agents/state.md`, and
  `src/vt/cuda/gemm_plan_cache.h:29` says the fp8 cache "ships off" while
  `fp8_plan_cache.h:23` records it default-ON since #1843. Both ride with the
  change that lands the answer, per #2750's own "in the same flow" clause.
- The vLLM-side half of the upstream chain: whether the pinned oracle
  re-resolves the same heuristic per process, traced rather than inferred. Not
  claimed anywhere in this file.
