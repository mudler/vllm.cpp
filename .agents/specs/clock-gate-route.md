# BENCH-CLOCK-GATE-ROUTE — the route to a Qwen3.8-27B c1 ratio, and the one route that is not available

**Row:** `BENCH-CLOCK-GATE-ROUTE`
**Lifecycle:** `BLOCKED` on [#1354](https://github.com/mudler/vllm.cpp/issues/1354)
**Issues:** [#1354](https://github.com/mudler/vllm.cpp/issues/1354) (Route A, the
capability grant), [#1546](https://github.com/mudler/vllm.cpp/issues/1546) (Route
C, the cross-arm excursion-symmetry term, filed by this row)
**Consequence lands on:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`
(`.agents/model-matrix.md:94`), which stays `PARTIAL`. This row moves no
lifecycle state and accepts no measurement, so it owes no `docs/STATUS.md` and no
`docs/BENCHMARKS.md` edit.
**Owed ratio:** [#915](https://github.com/mudler/vllm.cpp/issues/915),
[#979](https://github.com/mudler/vllm.cpp/issues/979)
**Amends nothing.** It decides between the two routes
[`lease-clock-pinning.md`](lease-clock-pinning.md) names, and adds a third.

**Amended by its own row on 2026-08-23.** §The 2026-08-23 settle run records a
measurement that falsifies one prediction this file made, and §The decision and
§Now carry the correction. Read that section before quoting any conclusion here
about what a settle procedure can reach.

**Read this first.** No route in this file recovers the nine discarded
2026-08-19 windows. Every number taken in them stays discarded, on both of its
independent grounds, and none of them may reappear as a result of any instrument
change. A ratio comes from a fresh measurement or it does not come at all.

## The row has no matrix row, and that is a named gap

`BENCH-*` is not a matrix prefix. `scripts/check-agent-record.py` carries ten
engine prefixes and five matrix prefixes, and none of them owns a benchmarking
instrument. The precedent is exact: `BENCH-ASSERT-CLOCK-STATE`, the row that owns
this same helper for [#543](https://github.com/mudler/vllm.cpp/issues/543), also
has no matrix row and appears only in
[`../issue-index.md`](../issue-index.md), and `ENV-LEASE-CLOCK-PINNING` records
its own row as owed at claim. Adding a `BENCH` prefix is a checker semantic
change and needs its own spec plus a red-before test, so this row does not make
one in passing. The gap is listed under `## Owed`.

## Scope

**In scope.** One decision, with its evidence: how the Qwen3.8-27B bf16 c1
cross-engine ratio owed by #915 and #979 can be measured on `dgx:gpu0` at all,
and whether it can.

**Out of scope.** Implementing any instrument change. `AGENTS.md` routes a
semantic checker change through spec, red-before test or mutation, and
green-after evidence, and this row writes the spec half only. Re-deriving the
discarded ratio, which is refused above and forbidden below. The fleet capability
grant itself, which is the fleet owner's decision under #1354.

## The two findings, re-derived here

Both findings landed on `main` in `7e07bbc91` (PR #1519). Both were re-derived
for this spec from the raw `*.samples.json` files at
`/mnt/nas_share/rc/q38bf16/out/`, and not read out of the commit. The
re-derivation reproduced the committed per-leg table exactly, on all nine
windows and every column.

| leg | n | min | p5 | median | p95 | max | `spread_pct` | p5-p95 band | drift | mean cost | labelled | cadence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours c1 r1 | 155 | 2177 | 2489.0 | 2489 | 2515.0 | 2515 | 13.58% | 1.04% | 0.00% | 0.136% | 4 | 1.099 s |
| ours c1 r2 | 155 | 1859 | 2489.0 | 2489 | 2489.0 | 2515 | 26.36% | 0.00% | 0.00% | 0.402% | 5 | 1.100 s |
| ours c1 r3 | 156 | 2158 | 2489.0 | 2489 | 2515.0 | 2515 | 14.34% | 1.04% | 0.00% | 0.204% | 5 | 1.100 s |
| ours c8 r1 | 245 | 2138 | 2328.2 | 2515 | 2515.0 | 2515 | 14.99% | 7.43% | 0.00% | 1.039% | 26 | 1.108 s |
| ours c8 r2 | 246 | 2177 | 2333.0 | 2515 | 2515.0 | 2515 | 13.44% | 7.24% | 0.00% | 0.976% | 24 | 1.110 s |
| ours c8 r3 | 244 | 2190 | 2353.0 | 2515 | 2515.0 | 2515 | 12.92% | 6.44% | 0.00% | 0.918% | 21 | 1.109 s |
| vLLM c1 r1 | 163 | 2262 | 2489.0 | 2489 | 2515.0 | 2515 | 10.16% | 1.04% | 1.04% | 0.116% | 4 | 1.097 s |
| vLLM c1 r2 | 163 | 2080 | 2489.0 | 2489 | 2489.0 | 2515 | 17.48% | 0.00% | 0.00% | 0.249% | 4 | 1.097 s |
| vLLM c1 r3 | 163 | 2054 | 2489.0 | 2489 | 2489.0 | 2515 | 18.52% | 0.00% | 0.00% | 0.307% | 4 | 1.098 s |

`labelled` is the count of retained busy samples carrying a non-benign throttle
bit under `BENIGN_THROTTLE_MASK`. Percentiles are linear-interpolation
percentiles, the convention [`lease-clock-pinning.md`](lease-clock-pinning.md)
pins. Mean cost is `|median - mean| / median * 100` over the retained busy
series.

### Finding 1 reproduces exactly

The commit's "flagged sample" is the **throttle-labelled** sample, not a sample
below a clock threshold. Reading it that way reproduces every published figure:

- The three vLLM c1 legs carry **12** labelled samples. Under one constant
  per-leg offset each labelled burst starts within **0.30 / 0.35 / 0.28 s** of a
  request start, at offsets **15.60 / 15.59 / 15.69 s**. The commit reports
  0.07-0.35 s at those three offsets.
- Our three c1 legs align at offsets **21.94 / 19.81 / 19.82 s** with residuals
  **1.73 / 1.68 / 1.64 s**. The commit reports the same three offsets and a
  1.27-1.73 s lag.
- The irregularity reproduces. Our request train has one long gap at request
  3 to 4, and the excursion train carries it at **31.77 / 31.79 / 31.82 s**,
  which is the commit's figure to the centisecond.
- At c8 the burst width scales with the batched prefill. Rep 1 runs
  **1, 4, 4, 3, 3, 4, 5, 2** samples wide, which the commit also reports, while
  all 26 c1 bursts are one sample wide.
- Over the nine windows there are **40** inter-burst gaps. **35** fall in
  **28.39-35.48 s** and the other five are **59.28 / 60.34 / 60.41 / 85.46 /
  90.08 s**, which are two and three times the base period.

**So the excursion is generated once per request by the measured workload.** It
is not weather, not heat soak and not another tenant. The period is each arm's
own per-request wall time, the phase differs between the arms and the lock does
not, and the train carries the request train's own defect.

Two limits on that finding, both stated in `7e07bbc91` and both re-checked here.
The proximate mechanism is not instrumented, so "request-locked" is established
and "prefill" is not. And the driver's `SwThermalSlowdown` label is not refuted
by the timing: the timing establishes **when** the excursions arrive, never
**why**.

### Finding 2 is right, and the exhaustive sweep states it more precisely

`MIN_BUSY_SAMPLES = 30` at the measured 1.097 s cadence demands about **32.9 s**
of retained busy samples, and the excursion period is 28.39-35.48 s with a
median of 32.3 s. The commit concludes that any admissible window necessarily
spans an excursion.

That conclusion is right where it matters and is stronger than "necessarily" can
carry everywhere, so this spec states the measured version. Sliding a 30-sample
window over every leg and scoring `spread_pct` on each position gives:

| leg | 30-sample windows | windows at or under 5.0% | best `spread_pct` | shortest span |
|---|---:|---:|---:|---:|
| ours c1 r1 | 126 | **0** | 5.22% | 31.84 s |
| ours c1 r2 | 126 | **0** | 5.22% | 31.86 s |
| ours c1 r3 | 127 | 7 | 4.18% | 31.87 s |
| ours c8 r1 | 216 | **0** | 5.92% | 31.99 s |
| ours c8 r2 | 217 | **0** | 5.69% | 32.08 s |
| ours c8 r3 | 215 | **0** | 5.69% | 32.06 s |
| vLLM c1 r1 | 134 | 52 | 0.76% | 31.79 s |
| vLLM c1 r2 | 134 | 25 | 0.24% | 31.79 s |
| vLLM c1 r3 | 134 | 25 | 0.24% | 31.81 s |

Five of the nine legs contain **no** admissible sub-window at all. Four contain
some. **That difference changes nothing**, for a reason that is not about the
statistic: a leg's throughput number covers the whole leg, so a clock record cut
to the quiet part of that leg does not describe the measured work, which is the
claim `MIN_BUSY_FRACTION` exists to protect. Selecting the sub-window by the
statistic under gate is post-hoc selection on the gate. The whole-leg window is
the record, and on the whole-leg window all nine fail.

**No amount of quiet on the box changes that**, because the cause is the work.
The commit's conclusion stands.

**One sentence above is now measured wrong, and §The 2026-08-23 settle run
carries the correction.** "The cause is the work" holds. The reading that
therefore no unpinned lease window can pass both rules does not. Two windows of
2026-08-23 passed both, because a window holding one long request holds one
request head instead of six. The lever nobody in this file considered is the
window's **request count**, not the box's quiet.

## The 2026-08-23 settle run, and the half of Finding 2 it falsifies

`#1354` option 2 asks for a settle-and-hold procedure that reaches the ceiling
without a pin. [`lease-gpu-capability.md`](lease-gpu-capability.md) §The
settle-and-hold alternative declined to run it, and §Finding 2 above predicted
it could not work: *"No amount of quiet on the box changes that, because the
cause is the work."* An `rc` job ran the procedure on `dgx:gpu0` on 2026-08-23,
and **the reachability half of that prediction is false**. Two windows passed
the spread rule and the throttle rule at the same time. They are the first
windows in this campaign to do so.

The mechanism half of the prediction survives, and §The discriminator below says
exactly which half survived and why the two are not the same claim.

**Every figure in this section was re-derived** from the raw
`*.samples.json`, `*.requests.json` and `thermal.csv` files at
`/mnt/nas_share/rc/clk1354/out/settle-20260823T004328Z/`, with
`build_clock_record` and `clock_reasons` at `8eecc05a9`. The job log's own
tables were read only to check that the re-derivation agrees, and it does, on
every column of all six windows.

### Option 1 is still closed, re-measured rather than assumed

`nvidia-smi -lgc 2190` and `-rgc` both returned `4` and *"The current user does
not have permission to change clocks for GPU 0000000F:01:00.0"* (`lgc.log`,
`rgc.log`, `job.log:13-19`). The refusal is on a **new pod**
(`rc-worker-4b8lj`) and a **new boot** (`f50a1be6-7626-4a2f-90f0-08c77b16c9f1`),
four days after the three jobs #1354 recorded. The pod reports
`CapPrm = CapEff = CapBnd = 0x00000000a80425fb`, which is the default OCI set:
bit 21 is clear, so `CAP_SYS_ADMIN` is absent from the **bounding** set and no
call inside the container can restore it. That confirms
[`lease-gpu-capability.md`](lease-gpu-capability.md) on a second pod.

### The six windows

The box was idle at the start (`COTENANTS_AT_START=0`), the server binary
matched its recorded sha256, and teardown left no compute app behind. One
server served the whole series.

| window | shape | n | idle | min | median | max | spread | spread <= 5 | throttle union | throttle clean |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| cold-steady | 1 request | 191 | 3 | 2476 | 2489 | 2535 | 2.37% | yes | `0x0` | **yes** |
| cold-train | 6 requests | 158 | 3 | 2047 | 2489 | 2489 | 17.76% | no | `0x0, 0x20` | no |
| hot1-steady | 1 request | 191 | 3 | 2405 | 2489 | 2489 | 3.37% | yes | `0x0` | **yes** |
| hot1-train | 6 requests | 158 | 3 | 2242 | 2489 | 2489 | 9.92% | no | `0x0, 0x20` | no |
| hot2-steady | 1 request | 192 | 3 | 2411 | 2489 | 2489 | 3.13% | yes | `0x0, 0x20` | no |
| hot2-train | 6 requests | 158 | 3 | 2067 | 2489 | 2515 | 18.00% | no | `0x0, 0x20, 0x68` | no |

A `steady` window holds one request of 1024 input and 950 output tokens for
210.8 s. A `train` window holds six requests of 1024 input and 128 output tokens
back to back, 173.5 s in total. Both shapes ran once cold and twice hot, so the
two axes separate.

**Every window fails a third rule.** `clock_reasons` refuses a record whose
`persistence_mode` is not `Enabled`, and all six read `Disabled`. §Persistence
mode below establishes what that is and what it is not.

### The discriminator is the request count, not the heat

Finding 1 is **confirmed** on a new boot four days later, and it is now
per-request rather than per-period. Each of the 21 requests carries one SM clock
minimum, and that minimum lands a median **3.64 s** after its own request start
(range 1.00 s to 4.17 s, n=21), in both window shapes.

Ten of the 21 requests carry a non-benign throttle bit, so the per-request label
probability is `p = 10/21 = 0.476`. A window is throttle-clean when none of its
requests is labelled:

| shape | requests | predicted clean | predicted of 3 | observed |
|---|---:|---:|---:|---:|
| steady | 1 | `(1-p)^1` = 52.4% | 1.57 | **2 of 3** |
| train | 6 | `(1-p)^6` = 2.1% | 0.06 | **0 of 3** |

The settle removed no excursion. It removed the **opportunities**: one request
head inside the window instead of six. So "the cause is the work" is upheld, and
"therefore no window can be clean" does not follow from it, which is the
inference this spec made and is corrected here.

**Heat is not the discriminator, and the thermal record says so directly.** The
labelled-sample rate over `thermal.csv` reads 0.19, 0.91, 1.00 and 1.00 per
minute over the four quarters of the job. It rises when the load starts and then
stays flat across quarters 2, 3 and 4, whose busy rows span 69-85 C, 73-84 C and
71-85 C. `cold-train` and `hot2-train` carry 3 and 2 labelled samples;
`hot1-train`, which ran between them, carries 4.

### The one excursion a steady window keeps is also the shallowest kind

The spread rule passes for a second and different reason. The first request of a
window dips least, because it follows an idle gap:

| request index | n | minimum depth below the window median | mean |
|---:|---:|---|---:|
| 0 | 6 | 0.52, 2.09, 2.85, 2.85, 3.13, 3.37 | **2.47%** |
| 1 | 3 | 0.00, 8.08, 10.45 | 6.17% |
| 2 | 3 | 7.03, 7.55, 17.76 | 10.78% |
| 3 | 3 | 3.13, 4.42, 4.94 | 4.16% |
| 4 | 3 | 1.81, 4.42, 5.75 | 3.99% |
| 5 | 3 | 3.66, 9.92, 16.95 | 10.18% |

Every index-0 minimum measured sits under the 5.0% ceiling, and no index-0
minimum reaches the depth the deeper train excursions reach. A steady window is
therefore clean on spread because of the excursion's **depth**, and clean on
throttle with probability `1-p` because of its **count**. The two rules pass for
two different reasons. Do not read one as evidence for the other.

### The thermal record upholds the driver's label

`thermal.csv` samples temperature and power beside the clock, at 1.089 s over
2625.5 s, 2411 rows and none unparseable. Those are the two fields
[#1386](https://github.com/mudler/vllm.cpp/issues/1386) asks for and
`QUERY_FIELDS` does not sample, so this is the first die reading this repository
can set beside a `SwThermalSlowdown` label. Over the 2070 busy rows:

| population | n | temperature median | temperature max | power median | SM clock median |
|---|---:|---:|---:|---:|---:|
| carries a non-benign bit | 34 | 82 C | 85 C | 62.7 W | 2275 MHz |
| carries no non-benign bit | 2036 | 75 C | 85 C | 43.2 W | 2489 MHz |

Of the 36 busy rows at or above 80 C, **30 are labelled**. Of the 12 at or above
84 C, **11 are labelled**. The lowest temperature on any labelled row is 76 C.

**This does not scope the throttle rule, and it moves §Scoping the throttle rule
is REFUSED in the direction of keeping it.** The label is not noise and it is
not misplaced: it lands where the die is hottest and the package draws most
power. The reading is a correlation over 34 rows on one boot, and the sampler
still cannot see a transient shorter than a second, so #1386 is not discharged
by a side channel that no committed record schema carries.

### The job log's own thermal summary is decimated, and it reads clean because of that

`job.log` §10 prints every 63rd row. That view shows 0 labelled rows, a maximum
temperature of 77 C and a maximum power of 43.9 W. The full file holds **34**
labelled rows, a maximum of **85 C** and a maximum of **81.1 W**. One labelled
row in 63 survives the decimation, so 0.54 rows were expected in that view and 0
appeared. The longest run at a flat 2489 MHz is **208 s**, not the whole late
span; 183 of the 1891 busy rows in the last 2400 s sit off 2489 MHz.

**A reader who quotes §10 will report a settle the file does not contain.** Read
`thermal.csv`, not the summary printed from it.

### What the two clean windows do not buy

- **A pairing needs both arms clean.** `(1-p)^2 = 27.4%` of steady pairings have
  both arms throttle-clean. All four pairings the job ran returned
  `PAIRING_VERDICT=DISCARD`, re-derived here with `compare_clock_records` at
  `8eecc05a9`, which reproduces the job's verdict and adds no reason of its own.
- **A steady window is not reliably clean.** `hot2-steady` is spread-clean at
  3.13% and still carries `0x20`. The result is 2 of 3, not 3 of 3.
- **Persistence mode refuses all six windows**, so this run could not have
  produced an established pairing whatever the clocks did.
- **Both arms of every pairing were our arm.** The job exercised the gate. It
  ran no engine comparison, so no pairing here could have been a ratio.
- **A steady window measures a different workload.** #915 and #979 owe a c1
  ratio at 1024 input and 128 output. The clean windows ran 1024 input and 950
  output. Substituting the shape that satisfies the gate is selection on the
  gate, which §Finding 2 refuses for sub-windows and which does not become
  admissible because the selection happens before the run instead of after it.
  Whether the owed ratio may be re-specified onto a long-generation shape is a
  decision for #915 and #979, and this spec does not make it.

**No ratio is derivable from this run, and none is recorded.** Every pairing was
discarded. A number a clock gate discarded never reappears as a result.

### Persistence mode is a separate lever, and it is available

`clock_reasons` has refused a record whose `persistence_mode` is not `Enabled`
since `51ec6bed5`. All six windows read `Disabled`, because the box reports
`Disabled` at `job.log:12`, before any window ran, and **the job never ran
`nvidia-smi -pm 1`.** There is no `-pm` call anywhere in `job.log`. So the
refusal is a procedure gap in this run, not a new capability wall.

**It is not a fourth blocked option and it is not part of option 1.** On
2026-08-22 the `SPEC-DFLASH2` speed gate ran `nvidia-smi -pm 1` **inside a
lease**, with no `ssh` and no host access, and it succeeded: *"Enabled
persistence mode via daemon for GPU 0000000F:01:00.0."* `-lgc` in the same job
still returned 4. `.agents/environment.md` records this and instructs every
leased measurement to read persistence mode at the start and set it if it is
off. This run did the first half and not the second.

**One consequence reaches Route A.** The 2026-08-15 pinned series that §Route A
cites as a demonstrated pass ran with persistence mode `Disabled`
([`../benchmark-record.md`](../benchmark-record.md), CLOCKS paragraph), and the
persistence rule already existed when it ran. That series therefore demonstrates
a pass on the spread rule and the throttle rule, and not on the persistence
rule. Whether the c4 cell's `established` verdict was ever produced by
`clock_reasons` over that record is not established here, and it is listed under
`## Owed`.

## What actually refuses the pairing, and it is two rules

Running the committed `tools/bench/gpu_clock_state.py` at `51ec6bed5` against the
nine committed per-leg records returns **exactly two reasons for each**, and the
second does not depend on the first:

```text
### ours c1 r2  n=2
   - clock: ours c1 r2 SM-clock spread over the measured window is 26.36%, above the 5.0% ceiling; the number is NOT ESTABLISHED
   - clock: ours c1 r2 was throttled during the measured window (HwSlowdown, HwThermalSlowdown, SwThermalSlowdown); the number is NOT ESTABLISHED
```

All three c1 pairings return four reasons and `established` unset, while
`median_offset_pct` is **exactly 0.0** on every one.

**The throttle rule has the same shape as the spread rule.**
`build_clock_record` folds the per-sample throttle strings into a `set`, so one
labelled sample in a window makes the whole window labelled, and the union can
only grow with `n`. Measured over the nine windows: **97 of 1690** retained busy
samples carry a non-benign bit, which is **5.74%**, and the union makes 100% of
every window inherit it. There is no duration term, no fraction term and no depth
term.

This is the fact that decides the row. **The rule that actually blocks the ratio
is the throttle rule, not the spread rule.** Replacing the spread statistic
changes no verdict on any of the nine windows, and
[`lease-clock-pinning.md`](lease-clock-pinning.md) says so in its own opening.

## Route A — pin the clock

**A pinned run of this exact workload already passed both rules.** On 2026-08-15,
on the same checkpoint `Qwen/Qwen3.8-27B` @ `1d4bf0f2`, over the host and
`flock` path, `nvidia-smi -lgc 2190` was accepted (`gpuClkMin 2190, gpuClkMax
2190`) and the legs ran at a **flat 2184 MHz** with
`clocks_event_reasons.active = 0x0` throughout, one boot id
`03717c9d-63c8-4652-a8fe-a63d012c5718`
([`../benchmark-record.md`](../benchmark-record.md), CLOCKS paragraph). One leg
reports n=861 retained busy samples, min 2158, median 2184, max 2184, a
`spread_pct` of **1.19%**. The c4 cell of that series is **established** and
reads 0.963x output throughput and 1.008x median inter-token latency
(`../model-matrix.md:94`).

So Route A is not a hypothesis. It removes the spread refusal and the throttle
refusal together, on this workload, on this box, on this checkpoint. It was the
only route in this file with that property until 2026-08-23, when option 2
cleared both rules twice out of three on a different workload shape without a
pin (§The 2026-08-23 settle run). Route A remains the only route that clears
them **on this workload** and the only one that clears them **every time**.
Route A also does not clear the persistence rule by itself: the 2026-08-15
series ran with persistence mode `Disabled`, so the pin recipe has to run
`nvidia-smi -pm 1` first.

**Its precondition is not ours to grant.** #1354 measured `LGC_RC=4` as root
inside an `rc` lease in three separate jobs, and `AGENTS.md` forbids reaching a
fleet device by `ssh`. Route A therefore waits on a capability grant to the
`rc-worker` container, which `lease-clock-pinning.md` §The fleet change specifies
and which the fleet owner decides.

### The 2184 MHz caveat, established rather than repeated

The records flag the caveat as inferred: the one cell that passed was pinned at
2184 MHz, **12.3% below** the 2489 MHz median this decode runs at unpinned, so a
pinned ratio may not be the ratio users get. This row was asked to establish
whether it holds. **It is not refuted, and the only two readings that exist
disagree by more than either one's own noise.**

| reading | clock | c1 median TPOT ratio (theirs / ours) |
|---|---|---:|
| 2026-08-15, pinned | 2184 MHz | **1.014x** (`../model-matrix.md:94`) |
| 2026-08-19, unpinned | 2489 MHz median | **1.047x** (228.3413 / 218.1295 ms) |

Both legs' own repeat spread is tiny: median TPOT CV is 0.01% over three ours
legs and 0.01% over three vLLM legs at 2489 MHz. A 3.3-point difference is two
orders of magnitude above that.

**The pair is not a controlled A/B and must not be read as the transfer
coefficient.** Four things moved between the two readings: the boot, the date,
our tree, and the SSE keepalive, which was on for the 2184 reading and off for
the 2489 one, and which made our arm fail 1 of 6 c1 requests in the pinned series
([#931](https://github.com/mudler/vllm.cpp/issues/931)). The honest reading is
narrow and sufficient: **the caveat is live at an estimated few percentage
points, on the axis this row exists to establish, and no evidence in the tree
retires it.**

The caveat is also **axis-dependent**, and that part is measurable from the
archived artifacts without any GPU time:

- **A decode-latency order statistic barely moves.** In `c1-r2.json`, per request
  and over 127 inter-token latencies, **0 to 2** exceed 1.05x that request's own
  median on our arm and 0 to 2 on vLLM's. The deepest excursion in the whole
  campaign, 1859 MHz against a 2489 median, produced a maximum inter-token
  latency of 234.73 ms against a 216.6 ms median, which is 8.4% on at most two
  tokens of 127. The median sits at rank 64, so the excursion population cannot
  reach it.
- **Time-to-first-token is the exposed axis**, because the excursion lands on the
  request head. Its repeat CV is **2.20%** for our arm against **0.01%** for
  median inter-token latency in the same three files. A pin that cuts the clock
  by 12.3% transfers into prefill nearly one for one, and the two arms' prefill
  implementations differ most.

So a pinned ratio is safest exactly where #915 and #979 need it, and least safe
on time-to-first-token. That does not discharge the caveat. It bounds it.

### What Route A must carry, and it is not in any record today

**A two-pin control.** Run the c1 pairing at two pinned clocks, both arms at
both pins, and report the ratio at each. If the ratio is invariant across the two
pins within the legs' own repeat spread, the caveat is discharged empirically and
the recorded ratio is the production ratio. If it is not invariant, the clock
sensitivity difference between the arms is measured rather than assumed, and the
ratio is recorded with the pin it was taken at and the slope beside it.

Nobody has ever run this control. Without it a pinned ratio carries an unbounded
caveat that no argument in this file removes, and the row would have traded one
unattributed number for another.

## Route B — scope the within-run spread rule

`lease-clock-pinning.md` designs this route in full: replace `(max - min) /
median` with a p5-p95 band at the same 5.0 ceiling, plus a drift term at 2.0 and
a mean-cost term at 1.0.

### Is it a forbidden widening? For the spread rule, no, and here is the proof

`AGENTS.md`: *"Never make a red gate green by deleting an assertion or widening
its scope."* A rule that refuses this workload, rewritten so it stops refusing
this workload, is exactly that shape. The defence has to be that the rule
measured the wrong thing, and it has to be executable rather than asserted.

**The executable part.** A widening admits a superset. This replacement does not,
and one window separates them. Take 200 samples that step from 2400 MHz to
2300 MHz at the halfway point and never return. Recomputed for this spec:

| window | `spread_pct` today | band | drift | mean cost | today | after |
|---|---:|---:|---:|---:|---|---|
| 200 samples, 2400 -> 2300 step | 4.255% | 4.255% | **4.255%** | 0.000% | **passes** | **refused, on drift** |
| 200 samples, 2400 -> 2280 step | 5.128% | 5.128% | 5.128% | 0.000% | refused | refused |
| 155 samples, 2489 flat, one at 1200 | 51.788% | 0.000% | 0.000% | 0.334% | refused | **passes** |
| 155 samples, 2489 flat, five at 1200 | 51.788% | 0.000% | 0.000% | **1.671%** | refused | refused, on mean cost |

Row one is a **sustained two-state window that today's rule passes and the
replacement refuses**. A pure widening cannot produce that row. Row three is the
loosening, stated rather than discovered: one sample in 155 can move a
clock-proportional leg by at most 0.334%, which is under every threshold in the
file, and row four shows the mean-cost term is what separates it from five such
samples.

**The argument part.** A spread rule exists to catch a GPU whose clock state
drifted between the arms or across a run. That is the
[#543](https://github.com/mudler/vllm.cpp/issues/543) failure: two probes eight
minutes apart inside one boot at 2398 and 1781, a **sustained level shift** of
about 26% however it is normalized, which repriced a byte-identical
`marlin::Marlin` by +9.65%. A drift term is a level-shift detector by
construction, and it catches that case directly. A range statistic is not a
level-shift detector; it is a maximum, and its numerator is non-decreasing in
`n` by construction. Finding 1 shows what it actually caught here: a per-request
load transition, common-mode across the arms, whose entire population is worth
at most 0.402% of clock in either arm and whose cross-arm median offset is
exactly 0.0%.

**A reviewer may reject that and should be able to.** Three grounds, stated so
the rejection is available rather than argued away:

1. The replacement admits a real 25%-deep sample that the old rule refused. The
   defence is a bound on transfer, not a claim that nothing happened.
2. The route was chosen after the gate refused this workload and only after.
   That is the shape of motivated reasoning whatever the arithmetic says.
3. The mean-cost term averages over the whole window while the excursion is
   phase-locked to the request head. It therefore bounds an integral metric such
   as output throughput, and it does **not** bound a phase-scoped metric such as
   time-to-first-token, whose own repeat CV is 220 times the inter-token
   latency's in the same files. This spec adds that limit, which
   `lease-clock-pinning.md` does not state.

### Route B is not a route to the ratio

It changes no verdict on any of the nine windows, because the throttle rule
refuses them independently, and it will change no verdict on a future unpinned
run of the same workload for the same reason. It is a correct repair to a
statistic that is wrong on its own terms, already specced under #1354, and this
row neither duplicates nor blocks it.

### Scoping the throttle rule is REFUSED

Reaching a ratio through Route B alone would need the throttle rule scoped too,
from "any non-benign bit anywhere in the window" to some fraction or duration.
**This row refuses that, and the refusal is the important half of its answer.**

- **No case exists that the replacement catches and the current rule misses.**
  For the spread rule the 2400 -> 2300 step is that case. For the throttle rule
  there is no equivalent, because a bit is present or absent and a fraction rule
  can only ever admit more. A change that only ever loosens is the forbidden
  shape by definition.
- **Finding 1 does not refute the label.** It establishes that the excursions
  arrive on the request train. The driver's label is the only physical
  attribution anybody recorded, and #1386 records that no thermal or electrical
  field exists to test it against.
- **The label correlates with a real clock reduction.** Of the 97 labelled
  samples, **95 sit below their window's median** and only **2** sit at or above
  it. The label is not noise.

So the honest statement is: the throttle rule may well be mis-scoped in the same
way the spread rule is, and **this evidence cannot show that**. Establishing it
needs the die reading #1386 asks for. Until then the rule stands as written.

## Route C — the third route, and what it does and does not buy

Two additive changes the data supports and neither existing spec names. Both
make the gate refuse **more**, never less.

**C1, the cross-arm excursion-symmetry term.**
`compare_clock_records` bounds the cross-arm median offset and nothing bounds the
difference in excursion burden between the arms. The two are independent on this
evidence: `median_offset_pct` is exactly 0.0 on all three pairings, while the
arms' mean cost differs by **0.020 / 0.153 / 0.103** points rep for rep. The
median cannot see the excursion population, which is precisely the part that does
not cancel between the arms and therefore the part that transfers into the ratio.
The term encodes Finding 1 directly: a workload-generated excursion appears in
both arms, a GPU-state defect appears in one, and today's verdict cannot tell
them apart. Filed as
[#1546](https://github.com/mudler/vllm.cpp/issues/1546).

**C2, scope the verdict to the metric class.** The gate returns one verdict for a
whole record whatever axis is being established. That is wrong in both
directions on this evidence. A periodic request-locked excursion covering 2.6% of
the window can move an integral metric by at most its mean cost, cannot reach a
median order statistic over 127 samples per request at all, and lands hardest on
time-to-first-token. A verdict that says "established for median inter-token
latency, not established for time-to-first-token" is a **narrowing** of what the
current single verdict claims. It needs the request markers the sampler does not
emit, so it changes the record schema and owes its own row, like #1386.

**Neither part is a route to this ratio.** C1 adds a condition. C2 needs a schema
change. Both are worth landing; neither produces a number.

## The decision

**Recommend Route A, conditioned on #1354, carrying a two-pin control.** It is
the only route with a demonstrated pass on this workload, and the two-pin control
converts its one live caveat from an argument into a measurement.

**Revised 2026-08-23: option 2 is now a second live route, at a measured
reliability, and it is not a substitute for Route A.** §The 2026-08-23 settle
run measured a settle-and-hold procedure that clears the spread rule and the
throttle rule together, 2 times out of 3, on a **one-request** window. Four
things bound what that buys, and all four are measured rather than argued:

1. A pairing needs both arms clean. The per-request label probability is
   `p = 0.476`, so `(1-p)^2 = 27.4%` of steady pairings clear the throttle rule
   on both arms. Roughly one attempt in four.
2. Every window of the run failed the persistence rule, which is fixed by one
   command that already works inside a lease.
3. The clean window measures 1024 input and 950 output tokens. #915 and #979 owe
   a ratio at 1024 and 128. Option 2 buys a clean window for a workload the
   owed ratio does not name.
4. The run compared our arm against our arm. It exercised the gate and produced
   no engine pairing.

**So the recommendation stands and its fallback changes.** Route A is still
first, because it clears the rules on the owed workload and clears them every
time. Option 2 is no longer a route this spec may call unreachable, so the
declined-grant paragraph below no longer converts a refusal straight into a
recorded negative. It now carries the workload the negative applies to.

**Land Route B on its own merits under #1354**, as `lease-clock-pinning.md`
specifies, with this spec's third risk added: the mean-cost term does not bound a
phase-scoped metric.

**Land Route C1 as an additive term** under #1546. **Refuse the throttle-rule
scoping** on the evidence available today.

**If the fleet owner declines the capability grant**, then the answer to #915 and
#979 is that the Qwen3.8-27B bf16 c1 cross-engine ratio at 1024 input and 128
output tokens is **not measurable on `dgx:gpu0` over the lease access path**, and
both issues record that rather than carrying an open promise. That is a
legitimate outcome and this spec says so. **The negative is now bounded by the
workload**, because option 2 reaches both rules on a one-request shape. Whether
to re-specify the owed ratio onto that shape is a decision for #915 and #979,
and this spec does not make it. Selecting a workload because it satisfies the
gate is selection on the gate, and it needs the same argument a threshold change
needs.

Two absolutes stand unpaired: ours 4.4040 tok/s at CV 0.039%, vLLM 4.2835 tok/s
at CV 0.033%, both `failed=0` on every leg. **Their quotient is not a ratio and
must never be written as one.**

## What no route recovers

The nine 2026-08-19 windows. They carry two independent refusals, and no
instrument change in this file removes either one from them. Re-scoring archived
data under a rule written after that data was refused is the move `AGENTS.md`
forbids, whatever the rule's argument. The numbers that were discarded stay
discarded, including the c1 median-TPOT and median-inter-token-latency figures
this spec uses to bound the excursion's reach. **Those figures are used here as
evidence about the instrument. They are not results and they are not a ratio.**

## Risks

- **The whole file can be read as arguing a red green.** Route B is that shape
  and §Is it a forbidden widening states the case for and the three grounds
  against. The throttle refusal is the control: a row that only wanted the gate
  quiet would have scoped that rule too, because that is the rule that blocks
  the number.
- **The 2184-against-2489 pair is confounded four ways** and is offered as a
  reason to run the two-pin control, never as the coefficient. Reading 3.3 points
  as measured is the misuse this paragraph exists to prevent.
- **Route A may not be granted**, and then the row's answer is a recorded
  negative rather than a number. Stated in §The decision so it is not a surprise.
- **The mean-cost bound is an integral over a 1.097 s sampler.** A sub-second dip
  is invisible, so "single sample, isolated" describes the sampling and not
  necessarily the physics. Carried from `lease-clock-pinning.md` §What was not
  measured.
- **`n = 3` everywhere.** Every repeat figure in this file is three legs of one
  workload on one boot.
- **A pinned box is a shared-host mutation.** `benchmarking.md` forbids leaving a
  box pinned, and a two-pin control doubles the number of pin and release cycles
  a run performs. The reset trap is not optional.

## Tests

This row changes no `src/`, `include/`, `tests/` or `tools/` file. `git diff
origin/main..HEAD` over those paths is empty and that emptiness is the claim.

The tests each recommended change owes when it is implemented:

- **Route B** owes the seven cases and the threshold mutation set in
  [`lease-clock-pinning.md`](lease-clock-pinning.md) §Tests, unchanged, plus one
  case this spec adds: the mean-cost term must be asserted to bound an integral
  metric only, so a case that pins the term against a phase-scoped metric must
  fail rather than pass.
- **Route C1** owes a red-before pair on `compare_clock_records`. One pairing
  where both arms carry the same excursion burden and the term passes, and one
  where the arms' mean cost differs by more than the ceiling while
  `median_offset_pct` is still 0.0 and the term refuses. The second case is red
  before the change by construction, because no expression reads the quantity
  today. The mutation is to move the new ceiling in the passing direction and
  show the second case turns green, printing `git diff --stat` so a mutation
  that never applied cannot read as a pass.
- **Route A** owes no unit test. It owes the `rc run` acceptance test in
  `lease-clock-pinning.md` §Gates, then the two-pin control, then a c1 pairing
  whose `compare` returns `established` with no waiver.

## Gates

- **This row's own gate is the checker set**, since it ships prose only:
  `check-agent-record.py`, `check-issue-index-append-only.py`,
  `check-commit-style.py`, `check-commit-trailers.py`, `check-doc-checkpoint.py`,
  each run unpiped with its real exit code read.
- **No GPU gate, no lease, and no `rc` job.** Every number in this file came from
  artifacts that already existed on the NAS and from arithmetic on them. The
  fleet is shared and this decision did not need it.
- **The gate that would accept a ratio is unchanged**: `gpu_clock_state compare`
  returning `established` with no waiver, on a fresh paired series, both arms,
  same boot, vLLM in its production configuration and never `--enforce-eager`.

## Stop conditions

- **Stop and report `NEEDS_DECISION` if the fleet owner declines the
  `CAP_SYS_ADMIN` grant.** The recommendation collapses to the recorded negative
  in §The decision, and #915 and #979 must be updated in that same flow.
- **Stop if the two-pin control shows the ratio moves with the pin** by more than
  the legs' own repeat spread. The ratio is then pin-dependent, and reporting a
  single number for it is wrong however clean its clock record is.
- **Stop if any implementer proposes re-scoring the 2026-08-19 windows.** That is
  the one outcome this row exists to prevent.
- **Stop if Route B's 2400 -> 2300 case cannot be made red before the change.**
  The replacement would then catch nothing the old rule missed, and it is a
  widening after all.

## Evidence

Raw sample windows, nine `*.samples.json` files with their per-leg records, at
`/mnt/nas_share/rc/q38bf16/out/bench-20260819T035148Z/` (six ours legs) and
`/mnt/nas_share/rc/q38bf16/out/vllm-20260819T095758Z/` (three vLLM legs). The
tenth directory `vllm-20260819T073125Z/` holds an aborted run whose first clock
record retains a **single** busy sample, so it is not one of the nine and no
figure here uses it.

Every statistic in this spec was recomputed from those raw files. The
`clock_reasons()` transcript is `tools/bench/gpu_clock_state.py` at `51ec6bed5`
run against the nine committed per-leg records. The helper still carries
`MAX_WITHIN_RUN_SPREAD_PCT = 5.0` and `MIN_BUSY_SAMPLES = 30` and has no drift
term: `96ed8346f` landed the spec proposing the repair and touched
`.agents/issue-index.md` and `.agents/specs/lease-clock-pinning.md` only.

Bench artifacts for the latency figures: `c1-r2.json` in the first directory and
`vllm-c1-r2.json` in the second, `itls` arrays, 127 entries per request.

The 2184 MHz pinned series is [`../benchmark-record.md`](../benchmark-record.md),
the CLOCKS paragraph of the 2026-08-15 Qwen3.8-27B entry, and the c4 cell is
`../model-matrix.md:94`.

## Owed

- **The matrix row.** `BENCH-*` has no matrix and adding the prefix is a checker
  semantic change. Named in §The row has no matrix row. The same gap already
  carries `BENCH-ASSERT-CLOCK-STATE`, so this row is the second instance and not
  the first.
- [#1354](https://github.com/mudler/vllm.cpp/issues/1354) stays open until the
  capability grant lands or is declined. Its option 2 is no longer unmeasured:
  §The 2026-08-23 settle run records what a settle-and-hold procedure delivers
  and at what reliability.
- **A settle run that fixes the two known procedure gaps.** Run `nvidia-smi
  -pm 1` at the start, and take both arms as one-request windows so a pairing
  can clear the throttle rule on both sides. Nobody has run that. Until somebody
  does, option 2 has produced clean **windows** and no clean **pairing**.
- **Whether any `established` verdict resting on the 2026-08-15 pinned records
  was produced by `clock_reasons` over those records.** Those records carry
  persistence mode `Disabled`, which the rule refuses, so the c4 cell of
  `../model-matrix.md:94` and the persistence rule cannot both be right as
  read. This spec measures the input and does not resolve the verdict.
- [#1546](https://github.com/mudler/vllm.cpp/issues/1546), the cross-arm
  excursion-symmetry term, filed by this row. **Implemented** by
  `BENCH-CLOCK-CROSS-ARM` in
  [`clock-cross-arm-mean.md`](clock-cross-arm-mean.md), which bounds the two
  arms' MEAN SM clocks rather than the difference of their excursion burdens;
  §The statistic there measures the pairing where the difference of burdens
  reads exactly 0.0000 while the mean clocks are 2.28% apart. It refuses more
  and never less, and it changes no verdict on any of the nine windows.
- [#1386](https://github.com/mudler/vllm.cpp/issues/1386), the thermal and
  electrical fields. This row depends on it: without a die reading the throttle
  rule cannot be shown mis-scoped, and §Scoping the throttle rule is REFUSED is
  the consequence. **A side channel now carries the two fields once.** The
  2026-08-23 job wrote `thermal.csv` beside the sampler, and §The thermal record
  upholds the driver's label reads it: the labelled rows sit at the top of the
  temperature and power distributions. That is one boot, 34 rows, and a file no
  committed record schema knows about, so #1386 stays owed.
- The c1 ratio owed by [#915](https://github.com/mudler/vllm.cpp/issues/915) and
  [#979](https://github.com/mudler/vllm.cpp/issues/979). **Blocked on #1354, and
  recorded as not measurable over the lease access path if that grant is
  declined.**
- Route C2, the metric-class verdict. It needs request markers in the sampler,
  changes the record schema, and owes its own row and spec.

## Now

`BLOCKED`. Nothing is implemented and no ratio is measured. The decision is made
and Route A's execution waits on a fleet-owner decision under #1354. Written
2026-08-21 from artifacts that already existed; no lease was taken and no job was
queued.

**Updated 2026-08-23 from the settle run at
`/mnt/nas_share/rc/clk1354/out/settle-20260823T004328Z/`.** Option 1 is closed,
re-measured on a new pod and a new boot. Option 2 is partly available: two
windows cleared the spread rule and the throttle rule together, 2 of 3 attempts,
on a one-request shape. Every pairing was `DISCARD` and no ratio is derivable
from that run. This session took no lease, ran no job and used no GPU; it read
the artifacts the job left and re-derived every figure from them.
