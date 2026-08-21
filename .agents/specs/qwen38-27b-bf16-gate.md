# Qwen3.8-27B (bf16): the token gate and the speed axes

**Row:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`
(`.agents/model-matrix.md`) — `Qwen/Qwen3.8-27B` declares
`Qwen3_5ForConditionalGeneration`, the architecture that row owns.
**Issue:** [#915](https://github.com/mudler/vllm.cpp/issues/915)
**Related:** [#821](https://github.com/mudler/vllm.cpp/issues/821) owns the
NVFP4 / Q4_K_M arms of the same checkpoint; [#910](https://github.com/mudler/vllm.cpp/issues/910)
owns the tie-break divergence this gate ran into three times.
**Lifecycle:** `PARTIAL`
**Owner:** unassigned

## Scope

`Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, bf16, text
path. Two axes, in order: a greedy token gate against the pinned oracle, then —
only if that gate is clean — throughput, latency and memory against vLLM's
production configuration.

Out of scope: the NVFP4 and Q4_K_M arms (#821), the vision path, any fix for
#910, and advancing the parity pin.

## Why this row needs no port

[`porting-inventory.md`](../porting-inventory.md) deviation 17 records the
finding this spec measures against: `Qwen/Qwen3.8-27B` "is the already-gated
Qwen3.6-27B shape retrained — `config.json` differs in exactly one key
(`transformers_version`) and the safetensors tensor-name set is identical (1199
names, zero difference either direction)". So no loader, forward or registry
change is owed, and none was made. What was owed is evidence, and the same
entry says so: "its own token-exact gate is OWED and unrun".

## Design of the adjudication

Greedy, 7 prompts x 16 tokens, both arms on the same prompts and token counts.
The oracle capture is deterministic across 3 repeats (`deterministic: true`,
`multi_member_cells: 0`), so a strict per-prompt comparison is well-posed.

**Only the first divergence per prompt is adjudicable.** After it the two arms
are conditioned on different prefixes, so any later position compares two
different conditionings rather than one disagreement. Three prompts diverge, so
there are three numbers — not thirty-four. A position count over the whole grid
is not a quality score and is not recorded as one.

At each first divergence the two arms agree on every preceding token, so both
are conditioned on a BYTE-IDENTICAL prefix. Feeding that prefix to the pinned
oracle and reading its **fp32** next-token distribution measures exactly one
thing: how far apart the oracle itself holds its own choice and ours.

## Risks

- **A bf16 instrument cannot resolve a bf16 tie.** A `transformers` CPU probe
  reported all three as exact ties, and every runner-up gap it printed was an
  exact multiple of 0.125 — one bf16 ULP in that exponent range. It could not
  have reported anything else, so it is a secondary cross-check with that
  limitation stated, never the answer. vLLM computes logprobs from fp32 and can
  separate pairs bf16 collapses.
- **A re-decode probe assumes the prefix it needs.** Reading the distribution
  at step `d` of a fresh greedy decode is only valid if that decode reproduced
  the captured prefix. Mitigated by a second, teacher-forced probe that feeds
  the prefix as token IDs and asserts both the echoed prefix and that the
  oracle's top-1 equals the captured token.
- **A stale binary measures a tree that does not exist.** Mitigated by
  rebuilding at `origin/main` and re-running both axes on that binary.
- **A degraded build voids every number.** Mitigated by asserting the
  configure-log fast-path lines (CUTLASS, FA2, Triton AOT `sm_121a`) and
  aborting the build otherwise.

## Tests

This row changes no `src/`, `include/` or `tests/` file, so it ports no test.
`git diff origin/main..HEAD` over those three paths is empty and that emptiness
is the claim. What it adds is evidence, listed under Evidence required.

## Gates

- Token gate: every first-divergence position within `kNearTieMnats = 500` of
  the oracle's teacher-forced argmax, on the pinned oracle, or the row fails.
- Speed: recorded only if the token gate is clean. vLLM's **production**
  configuration is the denominator — never `--enforce-eager`. Same client
  (`vllm bench serve`) drives both arms.
- Both under `$HOME/gpu.lock`, clocks pinned, contention recorded per leg.

## Evidence required

- Oracle identity asserted (`vllm.__version__`, `flashinfer`) with ABORT on
  mismatch, plus a `Python.h` precondition so a missing header aborts loudly
  rather than dying inside Triton's JIT.
- Per divergence: the oracle top-2 gap in millinats, our token's rank in the
  oracle top-20, and a verdict against the band.
- The build recipe, revisions, checkpoint size, binary md5, boot id, SM clock
  and the contention actually observed.

## Stop conditions

- If any divergence is out of band, STOP: report it as a real divergence, run
  no benchmark on that checkpoint, and let no record imply it is
  baseline-ready. Do not attempt a fix.
- If the box is not quiet at a leg boundary, wait or drop the leg. A number
  taken under load is worse than no number.

## Now

`PARTIAL`, and it stays `PARTIAL` after the 2026-08-19 re-measure. The token
axis is closed and passing. The speed axis is closed only at **c4** (0.963x
throughput, 1.008x ITL).

**Our half of the c1/c8 debt is DISCHARGED.**
[#931](https://github.com/mudler/vllm.cpp/issues/931) is closed and the fix
holds on hardware: with `VT_SERVER_SSE_PING_S=0`, three reps at each
concurrency on an idle leased box completed **162 of 162** requests with
`failed=0` and zero non-empty `errors` entries on every leg. c1 output
throughput **4.4040 tok/s** (CV 0.039%), median TPOT 218.11 ms, median ITL
216.56 ms, median TTFT 883.78 ms. c8 output throughput **22.6402 tok/s**
(CV 0.205%), total token throughput **196.10 tok/s, CORRUPTED — do not quote it**
([#1355](https://github.com/mudler/vllm.cpp/issues/1355), `## Owed`), median TPOT
250.57 ms, median ITL 232.83 ms, median TTFT 3623.5 ms. Both teardowns
`TEARDOWN_VERDICT=CLEAN`.

**And read the two output-throughput figures with the same caveat, bounded rather
than dismissed.** `output_lens` is `[128]xN` on both arms, so TPOT and ITL stand.
`output_throughput` divides output tokens by a WALL, and a genuinely truncated
prompt shortens that wall, so it is biased UP: at the c1 marginal prefill cost the
missing 202 tokens are ~0.22 s of a 174.39 s wall (~0.13%), and the missing 2,080
at c8 are ~1.1-1.6 s of 271.0 s (~0.4-0.6%). Both exceed the 0.039% and 0.205% CVs
published beside them, so "unaffected" is wrong and the bias sits outside our own
stated precision. Shorter context also cheapens decode, so these are lower bounds.
The median TTFTs (883.78 ms ours, 876.4 ms vLLM) ARE comparable: our two short
prompts are the two LOWEST TTFTs in every rep, so on both arms the median falls on
a 1024-token request.

**Neither cell became a ratio, and the two halves are blocked differently.** At
c1 the pinned oracle `0.1.dev1+g555967922` also completed every request on its
production graphed configuration (`enforce_eager=False`,
`cudagraph_mode: FULL_AND_PIECEWISE`, capture sizes `[1..64]`, `FLASH_ATTN`,
read back from its own startup line): **4.2835 tok/s** (CV 0.033%), median TPOT
228.36 ms, median ITL 226.86 ms, median TTFT 876.4 ms, cold start 426 s. Both
absolutes stand as facts with their own clock blocks, and
`gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three
pairings, so **no ratio is derived — the c1 ratio is OWED, not withheld for
being unflattering**.

The refusal is about spread, not disagreement. Cross-arm: same boot
`3fd9745a-d25a-426c-ba3c-97c958a85515`, both arms at a 2489 MHz median, offset
**0.0%**. Within-run: ours 13.58/26.36/14.34%, vLLM 10.16/17.48/18.52%, against
the 5% ceiling, with `SwThermalSlowdown` in every window and
`HwSlowdown+HwThermal` in one of ours. **All six of our legs and all three of
vLLM's breached that ceiling**, c8's included (12.92-14.99%).

**At c8 the vLLM denominator is NOT MEASURABLE on this box at the recorded
configuration.** That is the c8 answer, not a gap in it, and it is a statement
about headroom and guard granularity here rather than a claim that vLLM is
defective. A HOST REBOOT is now OBSERVED on this box on the benchmark day —
`boot_id` changed, so it was not merely a lost k3s pod — but nothing ties that
reboot to the worker loss behind this verdict, and `## Outcome` does not claim
it does. What the reboot establishes on its own is narrower: against the reboot
class of failure a sampling watchdog is not a guard at any cadence, because a
userspace sampler dies with the kernel. Neither half creates a denominator.
Detail, trajectory and arithmetic in
[`../benchmark-record.md`](../benchmark-record.md).

**What advances this row next is NOT a quieter box.** The 2026-08-20 re-measure
was planned and DID NOT RUN, and the reason it did not run is recorded here
rather than dropped: `dgx:gpu0` was held by other sessions for the whole window.
`rc devices` read `busy` throughout, first `claude/dflash2-cuda-gate-w4r` and
then `claude/mudler-ubuntu-box/fullmodel-guided-r12` at `BUDGET_S=14400`, with a
job queued behind it. No lease was taken, no leg was measured, and no number
below moves. What the window produced instead is the CAUSE of the c1 refusal,
read out of the artifacts that already existed at
`/mnt/nas_share/rc/q38bf16/out/`.

**The SM-clock excursion that fails the clock gate is generated ONCE PER REQUEST
by the measured workload.** It is not thermal weather, not heat soak and not
contention, and this is established from TIMING alone, which is why it needs
neither the die reading [#1386](https://github.com/mudler/vllm.cpp/issues/1386)
asks for nor a fresh lease.

- **The excursions land on the request train.** Across the three vLLM c1 legs,
  all 12 flagged samples sit within **0.07 to 0.35 s** of a request start under
  one constant per-leg offset, and the three offsets agree to 0.1 s
  (15.60 / 15.59 / 15.69 s). Our three legs align the same way at a constant
  **1.27 to 1.73 s** lag (offsets 21.94 / 19.81 / 19.82 s), so the phase differs
  between the arms and the LOCK does not.
- **The period is each arm's own per-request wall time, not a fixed interval.**
  Seven of our ten gaps cluster in **28.39-28.67 s**, against
  `128 / 4.4040 tok/s = 29.06 s` predicted from our own published c1 throughput;
  all five of vLLM's span **29.61-29.67 s** against
  `128 / 4.2835 tok/s = 29.88 s`. The two arms differ in the same direction and
  by about the same proportion as their throughputs, which a free-running
  hardware cycle would not do. Our remaining three gaps are the irregular ones
  in the next bullet.
- **The train reproduces an IRREGULARITY, not merely a period.** Our request
  train carries one odd gap, request 3 to request 4, at 31.63 / 31.71 / 31.66 s
  against 28.4-28.6 s everywhere else, because request index 2 carries the ~4 s
  TTFT outlier of
  [#1365](https://github.com/mudler/vllm.cpp/issues/1365). The excursion train
  carries the same odd gap in the same three legs, at **31.77 / 31.79 / 31.82 s**.
  A coincidence can match a period. It does not match a defect.
- **c8 says the same thing at wave granularity.** Its 48 requests start in eight
  waves (0, 33.9, 68.3, 102.8, 136.6, 170.3, 204.0, 237.7 s) in rep 1, where each
  of the seven bursts that has a successor ends **2.2 to 5.3 s** before the next
  wave starts. The burst also WIDENS with the batch rather than with elapsed
  time: all **26** c1 bursts across the six c1 legs are exactly one sample wide,
  while rep 1 of c8 runs 1, 4, 4, 3, 3, 4, 5, 2.
- **No true gap is longer than one period.** Over the nine windows the 40
  inter-burst gaps are 35 in 28.4-35.5 s plus five at 59.3 / 60.3 / 60.4 / 85.5 /
  90.1 s, which are 2x and 3x the base period: excursions the 1.10 s sampler
  stepped over, not quiet stretches.

**Three consequences, and the first one closes a route this section used to
name.** *Waiting for a thermally quiet window is not a route on any box*, because
the cause is the work being measured and it will recur on an idle, cold machine.
*The gate's observation floor and its spread rule are mutually exclusive on this
workload*: `MIN_BUSY_SAMPLES = 30` at the measured 1.10 s cadence is ~33.0 s of
observed busy GPU, and the excursion period is 28.4-35.5 s with a 33.2 s median,
so the shortest admissible window is already about one period long and the real
194 s leg spans six. *Pinning is the route with a precedent, and the
reading of that precedent is INFERRED rather than measured here.* The one
c1-class cell that ever passed, c4, was taken at a pinned FLAT 2184 MHz, below
the 2489 MHz median this decode runs at. A cap under the governor's trip point
would leave no headroom for the excursion, which would explain "flat" as a
consequence of the pin rather than as luck — but that campaign's clock samples
were not re-read here, so the explanation is a hypothesis and the only OBSERVED
part is that the passing cell was pinned and the refused ones were not.

**Stated at its own scope.** The excursion's proximate MECHANISM is not
instrumented here. The prefill is the obvious candidate, being the one dense
compute burst at the head of each request, but the sampler records no per-phase
marker and this evidence establishes request-locking rather than prefill
specifically. The excursions are also common-mode: both arms show them, both
arms' medians are identical at 2489 MHz, and `median_offset_pct` is **0.0**. What
refuses the pairing is the WITHIN-RUN spread rule, whose own comment in
`tools/bench/gpu_clock_state.py` says its job is "to detect that a window was not
ONE state at all". On this workload the window is honestly not one state, and the
two states are prefill and decode.

**The clock-gate repair is SPEC ONLY and has not landed in code.**
`96ed8346f` changed exactly two files, `.agents/issue-index.md` and
`.agents/specs/lease-clock-pinning.md`. `tools/bench/gpu_clock_state.py` still
carries its single commit `51ec6bed5`, still sets `MAX_WITHIN_RUN_SPREAD_PCT =
5.0` and has no drift term. Verified at two `main` tips half an hour apart,
`a50c57d69` and `141402e6c`, which carry the same blob `c88ca348` — a moving ref
is not a pin, so both SHAs are named. Re-running `compare` on the three archived c1 pairings at that tool
returns **real exit code 1** on each, unpiped, naming both the spread rule and
the throttle rule on both arms. So nothing about the recorded `DISCARD` has
changed, and the spec that landed says so itself in its own opening.

**Still true, and unchanged by any of the above.** The c8 vLLM denominator does
not exist at the recorded configuration, and it must not be attempted on this
box: that leg rebooted the host on 2026-08-19, `boot_id` changed, and a userspace
sampler dies with the kernel so no watchdog at any floor guards it.

## Outcome

**Measured, token axis.** `Qwen/Qwen3.8-27B` @`1d4bf0f2`, bf16, 55,586,114,863
bytes over 18 shards, against the pinned oracle `0.23.1rc1.dev1511+g555967922`
with FlashInfer `0.6.15.post1`, greedy, 7 prompts x 16 tokens. The oracle
capture is deterministic over 3 repeats (`multi_member_cells: 0`). **4/7 prompts
STRICT 16/16**, and the three first-divergence positions adjudicate as **exact
fp32 ties**:

| Prompt | Pos | Oracle | Ours | Top-2 gap | Oracle - ours | Our rank |
|---|---:|---|---|---:|---:|---:|
| `Once upon a time,` | 2 | 1814 `" world"` | 22960 `" magical"` | 0.000 mnats | 0.000 mnats | 3 |
| `The largest planet in our solar system is` | 1 | 11 `","` | 13 `"."` | 0.000 mnats | 0.000 mnats | 2 |
| `import numpy as np` | 8 | 16309 `" matplotlib"` | 27180 `" scipy"` | 0.000 mnats | 0.000 mnats | 2 |

The teacher-forced logprobs are `-1.5257947444915771`, `-0.74363690614700317` and
`-1.4836434125900269`; the greedy re-decode reads `-1.524564266204834`,
`-0.7444034814834595` and `-1.4876692295074463`. The two conditioning paths
differ in the last few thousandths of a nat, which is the point of reading it
twice — and the GAP is exactly `0.0` under both, on both members of each tied
pair. `INTEGRITY_OK=True`: for all three, the engine echoed the supplied prefix
back unchanged and its teacher-forced top-1 equals the captured greedy token, so
prefill and incremental decode do not disagree here.

`ALL_TIES_OR_IN_BAND` against `kNearTieMnats = 500`. Every one is
[#910](https://github.com/mudler/vllm.cpp/issues/910) and nothing else: the
oracle's pick carries the LOWER token id and ours the HIGHER at a bit-identical
logprob. `Once upon a time,` is a THREE-way tie — `" world"`, `" land"` and
`" magical"` all sit at `-1.524564266204834`.

**The shared prefix is proven, not assumed.** The adjudication is only valid if
both arms are conditioned on the same bytes. Two things establish it. Our
tokenizer reproduces the oracle's prompt ids **7/7 exactly**, checked directly
through `examples/tokenize` against `capture.json`'s `prompt_ids` — which the
token gate itself could not show, because our server's `echo` does not return
prompt tokens. And the generated ids agree up to the divergence by construction
of "first divergence".

**Rejected as the answer, kept as a cross-check.** A `transformers` CPU probe
called all three exact ties, and its own output refutes it as evidence: every
runner-up gap it printed was a multiple of **0.125**, one bf16 ULP in that
exponent range. An instrument that cannot resolve below one ULP cannot report
anything but a tie, so agreement with it is not confirmation. It is recorded
under the existing [transformers](../oracles/transformers.md) pin with that
limitation attached.

**Read twice, on the oracle's fp32 logprobs.** A greedy re-decode reads the
distribution at step `d` of the oracle's own deterministic decode. An
independent teacher-forced probe feeds the prefix as token IDs, then asserts
both that the engine echoed that prefix back and that the oracle's top-1 equals
the captured token — so a decode that had silently wandered before `d` cannot
pass as a valid conditioning. Both abort on an oracle-identity mismatch
(`vllm.__version__`, `flashinfer`) and on a missing `Python.h`, which otherwise
surfaces as an opaque failure inside Triton's JIT.

**The result does not depend on a stale tree.** The first run used a binary
built at `4a183b731`, 13 commits behind `main`, and `qwen3_5.cpp`,
`qwen3_5_dense.cpp` and `qwen3_5_weights.cpp` had all changed in between — so it
bound nothing. Rebuilt at `11a42dc4c` with the fast path asserted (CUTLASS, FA2,
Triton AOT `sm_121a`, `sm_121a` baked into the library) and re-run: **the same
4/7, the same three positions, the same tokens**. The intervening changes are
empirically inert on this path rather than assumed to be.

**Measured, speed axis, and one of three cells is all it supports.** Against
vLLM's production graphed configuration (no `--enforce-eager`,
`--language-model-only` on both arms, same `vllm bench serve` client, one
`flock`, clocks pinned to a flat 2184 MHz, one boot id), random 1024-in /
128-out, 6 prompts per concurrency unit, paired reps interleaved ours/vLLM:

| Axis | c1 | c4 | c8 |
|---|---:|---:|---:|
| ours / vLLM completed | 5,5,5 / 6,6,6 of 6 | 24x3 / 24x3 of 24 | 36,37,36 / 48x3 of 48 |
| Output throughput | WITHHELD | **0.963x** | WITHHELD |
| Median ITL | 1.013x | 1.008x | 1.021x |
| Median TPOT | 1.014x | 0.980x | 0.925x |
| Median TTFT | 0.733x | 0.881x | 1.268x |

Plus cold start **53 s vs 780 s = 14.7x** and host memory after warmup
**42.5 vs 110.1 GiB = 2.59x**, the latter caveated because vLLM's figure is set
by `--gpu-memory-utilization 0.85` pre-reserving KV rather than by the model.

**Two cells are WITHHELD, and that is the finding.** Our server failed 1 of 6
requests at c1 in all three reps and 12/11/12 of 48 at c8, where vLLM failed none
in nine legs on the identical workload from the identical client
([#931](https://github.com/mudler/vllm.cpp/issues/931)). `output_throughput`
divides tokens by a wall duration that still contains the dead request, so the
c1 cell reads 0.677x while median TPOT in the SAME file reads 1.014x in our
favour. Recording 0.677x would have published a 32% deficit contradicted by the
evidence beside it. No harness here asserted `failed == 0` before summarising a
ratio, which is the gap that let this become quotable in the first place.

The failures are silent server-side: a read-only sidecar sampled our server's
log live during a failing leg and it is 27 lines, all startup, no error, no
request line, no rejection. So the cause is not named yet, and a controlled
reproduction recording HTTP status and exception class is the next step.

**SETTLED, and the two halves of it are not equally strong.** This spec carried
one question as owed: whether the HOST rebooted or only the k3s pod was lost
when the c8 vLLM worker died. An `rc run` job on `dgx:gpu0`, job id
`97cf3e63-e4a4-4506-bde7-f19f19be3bbf`, read the answer back. This is the probe
log in full and verbatim, 312 bytes, sha256
`25b88023d85dbd7c751389be6427547ceb75f58992be18ba6d7f421a0418fd94`:

```text
rc: queued at position 1 for dgx:gpu0
rc: job 97cf3e63-e4a4-4506-bde7-f19f19be3bbf on dgx:gpu0
BOOT_ID_NOW=64c495a3-8c9c-4b20-8496-a97efda0e332
BOOT_ID_AT_BENCH=3fd9745a-d25a-426c-ba3c-97c958a85515
VERDICT=REBOOTED -- the host boot_id changed since the 2026-08-19 benchmark
UPTIME_S=38868
MemAvailable_MB=117436
```

**THE HOST REBOOTED. That half is OBSERVED and it is certain.**
`/proc/sys/kernel/random/boot_id` is generated once per kernel boot and is
kernel-wide, so a changed value is a reboot and cannot be anything else. A pod
restart, a container teardown and a `k3s` restart all leave it alone. The
disjunction is retired: it was not merely a lost pod.

**WHEN it rebooted is DERIVED, and it rests on an assumption this measurement
does not test.** `UPTIME_S=38868` is `/proc/uptime` as read from INSIDE the `rc`
worker, so it is the HOST's uptime only if that worker does not virtualize
`/proc`. `lxcfs` and its equivalents do virtualize `/proc/uptime`; none of them
can touch `boot_id`, which is why the observed half does not inherit this
caveat. The worker is a k3s pod, where an unvirtualized `/proc` is the default,
but nothing in these artifacts asserts that `lxcfs` is absent. The VALUE argues
against virtualization by itself — a container-scoped `/proc/uptime` reports the
POD's age, and 38,868 s is 10.8 h, which is not a plausible age for a job
submitted minutes before the read — but an argument is not the assertion, so the
caveat stands. Establishing it costs one line in a later job and no lease was
taken for it here.

**The bound is ONE-SIDED, and the interval an earlier draft of this section
recorded was not.** That draft also said no tighter figure was honest, which was
wrong in both directions: a tighter UPPER bound exists and the LOWER bound is
not established at all. 38,868 s is 10:47:48. The probe
log's last write is 21:29:35.606816Z, and that mtime is when the FINAL line
(`MemAvailable_MB`) landed, so `UPTIME_S` was read strictly BEFORE it. The mtime
is an UPPER bound on the read instant, not a midpoint, and 21:30:02Z — the run's
reported completion — is later still. Both of the endpoints recorded earlier sit
at or after the true read instant, so the interval they formed excluded the
whole region the value occupies. The arithmetic:

```text
21:29:35.606816Z  -  10:47:48  =  10:41:47.606816Z
```

So the derived boot is **at or before 2026-08-19T10:41:47.6Z**, half a second
later still if `UPTIME_S` rounds `/proc/uptime` rather than truncating it; the
probe script was not retained, so which one is unknown. No LOWER bound is
stated. One would have to come from when the probe STARTED rather than from
anything in the log's content, and it would cross the same unpinned clock
boundary as the upper one.

**The two clocks are not pinned to each other and this measurement does not
quantify the skew.** 21:29:35.606816Z is the mtime of a file on the LOCAL host,
written there by the `rc` client. `UPTIME_S` was read on `dgx`, and the table
below is `dgx`-side content plus mtimes on the shared CIFS mount. Neither is the
local host's clock, so the derived instant is being compared across an offset
nothing here measures.

| Time (UTC) | Strength | Event |
|---|---|---|
| 10:18:51 | observed | c1 legs complete, `TEARDOWN_VERDICT=CLEAN`, `MemAvailable` 116,869 MB |
| 10:18:54 | observed | c8 vLLM server launched, `SERVER_PID=123868`; also `job.log`'s last CONTENT line |
| 10:25:07 | observed | c8 server answers `GET /health 200 OK`; `vllm-server-c8.log`'s last CONTENT line |
| 10:25:26 | observed | last `MemAvailable` sample, 6,261 MB; nothing after it. `mem.samples` mtime is 10:25:28.570640 |
| **at or before 10:41:47.6** | **derived** | **upper bound on the boot of the kernel now running** |
| 11:26:32.079 | mtime only | `job.log` and `vllm-server-c8.log` both re-stamped. NOT a content write and NOT established as liveness |

**The 11:26 row is a file mtime, and an earlier draft recorded it as `11:26:00`,
"the job's last write".** It is neither. The actual mtimes are 11:26:32.079517600
(`job.log`) and 11:26:32.079047100 (`vllm-server-c8.log`), **0.47 ms apart**,
which is one bulk event touching both files rather than a job writing to either.
No CONTENT anywhere in `out/vllm-20260819T095758Z/` is later than 10:25:07
(`vllm-server-c8.log`) or 10:25:26 (`mem.samples`), and `job.log`'s last content
line is the 10:18:54 launch banner.

**Two readings of that row are possible, they are incompatible, and NEITHER is
established.** If a process flushed at 11:26:32 then the box was alive at
11:26:32, which contradicts a boot at or before 10:41:47.6Z that ended it. If it
was a reaper or a CIFS flush, the last evidence of anything alive is
`mem.samples` at 10:25:28.570640 and the derived boot is about 16 minutes AFTER
it. The row is recorded at mtime strength and the tension is left open, not
resolved by preferring whichever reading fits.

**A second open discrepancy, from this box's own prior reboot.**
[`../environment.md`](../environment.md) records that one with an exact
`journalctl --list-boots` pair: boot `-1` ending 09:10:15Z against boot `0`
beginning 09:13:55Z, about **3m40s** of downtime. Apply that shape here and a
boot at 10:41:47 puts the box GOING DOWN around 10:38, thirteen minutes after
every writer in the evidence directory had already stopped, which does not fit a
simple crash at 10:25:28. Recorded as an open discrepancy and left unresolved:
the 3m40s is one sample from one reboot, the bound above is one-sided so the
true boot may be much earlier, and the two clocks are unpinned.

**The WARMUP clause belongs to the worker LOSS, not to the boot.** The loss was
during the untimed warmup before any timed leg ran; that is observed. At the
derived boot instant nothing had written for about 16 minutes, so that instant is
not "during the warmup". Measured against the observed marks, the bound sits at
most 16m41s after the `/health` 200 and at most 16m22s after the last memory
sample.

**NOT established: that the reboot killed the worker.** Three things are
consistent with it — the bound above, the descent `NOTES.txt` FINDING 4 records
(9,738 -> 6,261 MB over ~40 s, then the worker lost between two 2-second
samples), and this box's documented habit of rebooting instead of OOM-killing
(`.agents/environment.md`). Consistency is not a trace. Nothing here ties the
reboot to the worker's death, so it stays a hypothesis with a one-sided derived
bound, exactly as this repository requires, and no record may read it as a cause.

**No number moves.** This settles a provenance question and creates no
denominator. The c8 vLLM denominator is still NOT MEASURABLE at the recorded
configuration, the c1 pairing is still `PAIRING_VERDICT=DISCARD`, and neither
cell becomes a ratio because of this finding. What the finding does is explain
the shape of the absence.

**What the reboot DOES establish about watchdogs, stated at its own scope.** A
watchdog is a userspace process on the box, and a kernel reboot ends it with
everything else, so against the REBOOT CLASS of failure there is no floor and no
cadence at which a sampler lives long enough to report the event. That is a
general property of samplers and kernels and it needs no causal link. It does
NOT strengthen the campaign's separate conclusion about the c8 denominator,
which would require the reboot to be what killed this worker — the thing the
paragraph above declines to claim. `boot_id`, read after the fact, is the only
instrument that saw the reboot at all.

**What was NOT established.** Nothing about the vision path on this checkpoint,
nothing about its NVFP4 or Q4_K_M arms (#821), no claim that #910 is fixed (this
is the second checkpoint to be costed by it), and no throughput number at c1 or
c8 until #931 closes. Concurrencies above 8 were not run.

## Owed

- [#915](https://github.com/mudler/vllm.cpp/issues/915) stays OPEN. Our arm's
  c1/c8 debt is discharged; the vLLM half is not. The c1 ratio is refused by the
  clock gate and the c8 denominator does not exist, so the row's own question —
  what our speed is against vLLM's production configuration at c1 and c8 — is
  still unanswered.
- **The c1 ratio now needs an INSTRUMENT decision, not another lease.** `## Now`
  establishes that the excursion refusing the pairing is generated once per
  request by the measured workload, so re-running the campaign reproduces it on
  any box and at any temperature. Two routes exist and this row picks neither,
  because both are somebody else's row. **Pin the clock**, which needs the host
  path and is what [#1354](https://github.com/mudler/vllm.cpp/issues/1354) owns;
  on the hypothesis in `## Now`, a cap below the 2489 MHz decode
  median would remove the headroom the excursion needs, which would explain the
  flat c4 window; re-reading that campaign's clock samples would confirm it. Or **scope the within-run
  spread rule to the state it claims to police**, since its own comment says its
  job is to detect that a window was not one state, and prefill against decode is
  two states the benchmark exists to measure. That second route changes gate
  semantics, so it owes its own row, spec, red-before mutation and green-after
  evidence, and it must never be reached by widening the ceiling until the
  recorded windows pass.
- **A finding this row produced for a NEIGHBOURING issue, owed to its owner
  rather than acted on here.**
  [#1386](https://github.com/mudler/vllm.cpp/issues/1386) says the nine windows
  "cannot distinguish a load transition from a thermal excursion" and proposes
  adding `temperature.gpu` and `power.draw` to `QUERY_FIELDS` to settle it. The
  alignment in `## Now` settles it from timing alone, and it settles it as a
  **load transition**: the excursions are locked to the request train on both
  arms, their period is each arm's own per-request wall time, and they reproduce
  the irregular gap that #1365's TTFT outlier injects. The two proposed fields
  stay worth having, and they are no longer what decides this question. This row
  files nothing and edits nothing under #1386; relaying it is owed.
- **The campaign `NOTES.txt` on the share names a superseded binary.** Its `PINS`
  block records `ab0b9a1e...` as the measured binary, which was the
  2026-08-18 build. The tree that produced every 2026-08-19 leg is
  `7d0c3caf...`, which `out/RESULT.txt` records beside `BUILD_RC=0` and
  `WANT_SHA=1dac4f9a70195b282d16c536f319e8b171c925f8`, and which re-hashes to
  that value on the share today. The share is not a tracked surface, so the
  correction is recorded here rather than applied there.
- **The boot TIME, which is derived and one-sided.** The 2026-08-19 HOST reboot
  is OBSERVED from `boot_id` and owes nothing to this item. Only the derived
  bound — at or before 10:41:47.6Z — depends on the `dgx:gpu0` `rc` worker not
  virtualizing `/proc/uptime`, and on a local-to-`dgx` clock offset that nothing
  measures. `journalctl --list-boots` retires both at once, because it prints the
  boot's own down and up timestamps in the host's clock and needs neither
  `/proc/uptime` nor an independently known boot. **Whether a LEASED job can reach
  it is itself unestablished, and this item must not assume that it can.** The
  `boot_id` bullet in [`../environment.md`](../environment.md) states that the
  boot list is a HOST instrument a pod does not have, which is exactly why
  `boot_id` was the instrument here; a leased worker runs as root and may or may
  not have the host journal mounted, and no artifact settles that. So the job is:
  run `journalctl --list-boots` from inside a lease and record whether it answers
  at all, and print `/proc/uptime` beside `date -u` in the same job so the
  virtualization question falls out of the comparison either way. If the lease
  cannot read the boot list, this item needs host access and stays owed. The
  `thor:gpu0` build reboot that `../environment.md` records under
  [#1380](https://github.com/mudler/vllm.cpp/issues/1380) reaches the same
  instrument from the other side and leaves the same access question open, so one
  job that answers it discharges part of both. No lease was taken here.
- **Two open discrepancies around that bound, recorded rather than resolved.**
  The `11:26:32` mtimes on `job.log` and `vllm-server-c8.log` are 0.47 ms apart
  and admit two incompatible readings, one of which puts the box alive after the
  derived boot. And the only `journalctl` downtime this repository has measured
  on this box is ~3m40s, which applied here would put it going down around 10:38,
  thirteen minutes after every writer had stopped. `## Outcome` states both. The
  job above settles the second and bounds the first, if it can read the boot
  list at all.
- [#1355](https://github.com/mudler/vllm.cpp/issues/1355), the prompt-token
  divergence found in this campaign's raw result files: our server reported
  5,942 prompt tokens where vLLM reported 6,144 for the identical
  client-generated prompts, 19 of 48 short at c8. Whether we under-report
  `usage.prompt_tokens` or actually truncate the prompt is not decidable from
  these artifacts. **It corrupts `total_token_throughput` on both legs** — c1
  38.4776 tok/s and c8 196.10 tok/s are computed over 5,942 and 47,072 input
  tokens where the workload intends 6,144 and 49,152 — and it biases
  `output_throughput` up by more than that figure's own CV; `## Now` carries the
  bound. Quoting either total-token figure, or setting our 38.4776 beside vLLM's
  38.5516, compares two different workloads.
- [#1365](https://github.com/mudler/vllm.cpp/issues/1365), a reproducible ~4 s
  TTFT outlier on request 3 of every c1 leg of ours, which the oracle does not
  have: index 2 reads 3.981 / 3.924 / 4.006 / 3.955 s across the warmup leg and
  all three reps against 0.73-0.93 s for every other request, four legs of four,
  on a 1024-token prompt like requests 4, 5 and 6. Nothing published is wrong,
  because the median of six averages ranks three and four and the outlier never
  occupies either; it moves the MEAN (ours 1347.6-1372.6 ms against the oracle's
  873.3-900.2 ms) and costs ~3.1 s of the 174.39 s c1 wall. The cause is not
  chased here.
- The checkpoint size disagrees between records: this spec's `## Outcome` says
  55,586,114,863 bytes and the campaign's `NOTES.txt` says 55,586,040,114, a
  difference of 74,749 bytes. The 2026-08-19 run DID re-derive it, and it agrees
  with `NOTES.txt`: `out/bench-20260819T035148Z/job.log:47,49` print
  `CKPT_SRC_BYTES=55586040114` and `CKPT_DST_BYTES=55586040114` over the staged
  tree that then served every leg. What is unresolved is WHY the `## Outcome`
  figure differs, which the artifacts cannot settle.
