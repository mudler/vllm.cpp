# ENV-LEASE-CLOCK-PINNING — the driver wants `CAP_SYS_ADMIN`, and the spread rule scores the deepest single sample

Issue: [#1354](https://github.com/mudler/vllm.cpp/issues/1354) — `nvidia-smi -lgc`
is refused inside an `rc` lease, and every clock-pinned figure in this repository
was taken over the retired host + `ssh` + `flock` path.
Row: `ENV-LEASE-CLOCK-PINNING` (proposed; the roadmap row is owed at claim, and
the work below is two changes, one to the fleet and one to
`tools/bench/gpu_clock_state.py`). The branch carrying this spec is
`row/SPEC-LEASE-CLOCK-PINNING`, which does not match that ID. Recorded rather
than renamed mid-review: the row opened at claim carries the ID above, and a
rename here would only move the mismatch to the merged history.
Amends: [`bench-assert-clock-state.md`](bench-assert-clock-state.md), whose
`MAX_WITHIN_RUN_SPREAD_PCT` this spec proposes to replace rather than widen.
Prior art: [#1265](https://github.com/mudler/vllm.cpp/issues/1265) — the same
class. A capability the records assume, which the current access path does not
provide.

**Read this before anything else in the file.** Neither half of this row recovers
the discarded Qwen3.8-27B c1 ratio. All nine of the 2026-08-19 windows are
refused a **second** time, by the untouched throttle rule, and that refusal does
not depend on the statistic this row replaces. The instrument change is a repair
to a statistic that is wrong on its own terms. It is not a route to a number.

## The defect, in two independent halves

**Half one is a missing Linux capability, and it is fleet-wide.** The refusal is
not about `uid`. The NVIDIA driver's administrator test is

    // kernel-open/nvidia/os-interface.c
    NvBool NV_API_CALL os_is_administrator(void) { return NV_IS_SUSER(); }

    // kernel-open/common/inc/nv-linux.h
    #define NV_IS_SUSER()                   capable(CAP_SYS_ADMIN)

so a privileged RM control — which is what `nvmlDeviceSetGpuLockedClocks`, and
therefore `nvidia-smi -lgc`, issues — is gated on `CAP_SYS_ADMIN` and never on
`uid == 0`. **That quotation is an external reference nobody here has read at a
pinned revision** (§Evidence), so what the behavioural evidence below
establishes on its own is that container root lacks *something* host root has,
and not *which* capability it is. A container process can be root and still
fail it, which is exactly what the job reported:

```text
$ nvidia-smi -lgc 2190
The current user does not have permission to change clocks for GPU 0000000F:01:00.0.
LGC_RC=4
```

The worker's own manifest says why. In `infra-flux-kube`,
`manifests/dgx/rc-worker.yaml`, the `rc-worker` DaemonSet declares
`runtimeClassName: nvidia`, `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, and the `worker` container — the
long-lived container that leased jobs run *inside* — **declares no
`securityContext` at all**. The only `securityContext` in the file sits on the
`workspace-perms` initContainer. With none declared the container gets the
default OCI capability set, which does not contain `CAP_SYS_ADMIN`.
`manifests/thor/rc-worker.yaml` and `manifests/orin/rc-worker.yaml` have the same
shape, so this is a property of the fleet and not of one box.

Three things this is **not**, each ruled out rather than assumed:

- **Not `NVIDIA_DRIVER_CAPABILITIES`.** That variable selects which userspace
  driver components the container runtime injects. `utility` is what supplies
  `nvidia-smi` and NVML, and NVML plainly works — every `--query-gpu` in the run
  succeeded and produced 155 to 246 samples per window. No value of that
  variable adds a Linux capability.
- **Not the device cgroup.** `/dev/nvidiactl` is reachable; the queries that
  travel the same node succeed.
- **Not the driver refusing a containerised caller as such.** The same driver on
  the same box accepted `-lgc 2190` from the host on 2026-08-15
  (`gpuClkMin 2190, gpuClkMax 2190`). Host root holds `CAP_SYS_ADMIN`; container
  root does not.

**One reading in the issue does not survive contact with the log.** The harness
ran `nvidia-smi -pm 1 2>&1 | tail -1` and got `All done.`, which looks like a
successful NVML write beside a refused one. It is not evidence: `job.log:9` of
the same run reports `NVIDIA GB10, 12.1, 580.173.02, Enabled, 3003 MHz` — read
**before** the `-pm` call — so persistence was already `Enabled` and the call was
a no-op. `nvidia-smi` also prints `All done.` after a per-GPU failure line, and
`| tail -1` discards that line. **No NVML write is known to succeed inside a
lease.**

**A second, independent failure decides the c1 pairing, and it is not the
spread rule.** Every one of the nine 2026-08-19 windows also carries a
**non-benign throttle reason**. All nine record `0x20` (`SwThermalSlowdown`);
ours c1 rep 2 also records `0x68` (`HwSlowdown | HwThermalSlowdown |
SwThermalSlowdown`). `BENIGN_THROTTLE_MASK` is `0x1 | 0x2 | 0x100`, so
`_throttle_offenders` returns non-empty on every one of them and the committed
`clock_reasons()` appends the throttle reason **independently of the spread
reason**. Run against the nine committed records at
`/mnt/nas_share/rc/q38bf16/out/`, today's `tools/bench/gpu_clock_state.py`
returns two reasons for each, for example:

```text
### ours c1 r2  throttle=['0x0000000000000000', '0x0000000000000020', '0x0000000000000068']
   - clock: ours c1 r2 SM-clock spread over the measured window is 26.36%, above the 5.0% ceiling; the number is NOT ESTABLISHED
   - clock: ours c1 r2 was throttled during the measured window (HwSlowdown, HwThermalSlowdown, SwThermalSlowdown); the number is NOT ESTABLISHED
### vLLM c1 r1  throttle=['0x0000000000000000', '0x0000000000000020']
   - clock: vLLM c1 r1 SM-clock spread over the measured window is 10.16%, above the 5.0% ceiling; the number is NOT ESTABLISHED
   - clock: vLLM c1 r1 was throttled during the measured window (SwThermalSlowdown); the number is NOT ESTABLISHED
```

**So the instrument change does NOT recover the discarded Qwen3.8-27B c1
ratio.** With the spread rule replaced exactly as designed below, all nine
windows still fail on the throttle rule, all three c1 pairings still return
`PAIRING_VERDICT=DISCARD`, and the c1 ratio stays owed to
[#915](https://github.com/mudler/vllm.cpp/issues/915) and
[#979](https://github.com/mudler/vllm.cpp/issues/979). #1354 already records the
throttle facts; a reader who takes this row as unblocking that ratio has read it
wrongly. What this row fixes is a statistic that is wrong on its own terms, which
is worth fixing whether or not any particular pairing survives it. The throttle
rule is untouched, and §Design credits it as the co-guard that catches the
physically alarming windows the band alone would pass.

**Half two is the instrument, and it is not caused by half one.**
`summarize_sm_clocks` computes

    spread_pct = (max - min) / median * 100

which is a range statistic over the raw samples. It scores the single deepest
excursion in the window and nothing else. Two consequences, both measured on the
2026-08-19 evidence at `/mnt/nas_share/rc/q38bf16/out/`:

**Its numerator only ever grows with window length.** `max - min` is
non-decreasing in `n` by construction, and only a moving median can offset it.
Recomputing
`spread_pct` over growing prefixes of the same leg (our c1 rep 2, n=155):

| prefix n | `spread_pct` (min/max) | p5–p95 band |
|---:|---:|---:|
| 30 | 6.79% | 2.33% |
| 60 | 14.10% | 0.93% |
| 90 | 16.43% | 0.44% |
| 120 | 26.36% | 0.00% |
| 155 | 26.36% | 0.00% |

The longer the sampler watches, the worse the window scores, while the
distribution it is describing gets *tighter*. That is the same perverse
incentive `MIN_BUSY_SAMPLES` was added to remove on the other side — the window
nobody watched outscoring the one that was — reintroduced by the statistic
itself. `MIN_BUSY_SAMPLES` bounds the failure from below; nothing bounds it from
above.

**It does not bound what it is read as bounding.** The nine 2026-08-19 windows.
Every percentile in this spec, in its tables and in its test cases, is the
**linear-interpolation** percentile of `statistics`-style convention
(`p = v[lo] + (h - lo) * (v[hi] - v[lo])`, `h = (n - 1) * q`), because the cases
below pin exact values and the two conventions disagree by enough to matter:
ours c8 r1 reads **7.43%** on the band by linear interpolation and **7.48%** by
nearest rank. `drift` is `|median(last third) - median(first third)| / median`
over the retained busy series. Naming the **series** settles the only split
convention that changes a number on this evidence, which is the busy series
against the sampler's whole elapsed span; the drift paragraph below measures
that disagreement on ours c1 r1. It does **not** settle how the thirds are cut
*within* the busy series, and count-floor (`n // 3`), count-ceiling
(`ceil(n / 3)`) and time-based thirds are three distinct rules. Recomputed from
the raw samples, all three agree on every one of the nine windows — 0.0000% on
eight of them and 1.0446% on vLLM c1 r1 — and on §Tests case 1, at 4.2553%.
**That choice is therefore left unpinned**, and deliberately: no recorded window
discriminates it, and a case invented to discriminate it would pin a convention
rather than measure one. An implementer picks one of the three and records which
beside the constant.

| leg | n | min | p5 | median | p95 | max | `spread_pct` | p5–p95 band | drift | mean cost | samples <95% of median | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours c1 r1 | 155 | 2177 | 2489 | 2489 | 2515 | 2515 | 13.58% | 1.04% | 0.00% | 0.136% | 5 | 1 |
| ours c1 r2 | 155 | 1859 | 2489 | 2489 | 2489 | 2515 | 26.36% | 0.00% | 0.00% | 0.402% | 5 | 1 |
| ours c1 r3 | 156 | 2158 | 2489 | 2489 | 2515 | 2515 | 14.34% | 1.04% | 0.00% | 0.204% | 4 | 1 |
| ours c8 r1 | 245 | 2138 | 2328 | 2515 | 2515 | 2515 | 14.99% | 7.43% | 0.00% | 1.039% | 28 | 5 |
| ours c8 r2 | 246 | 2177 | 2333 | 2515 | 2515 | 2515 | 13.44% | 7.24% | 0.00% | 0.976% | 29 | 5 |
| ours c8 r3 | 244 | 2190 | 2353 | 2515 | 2515 | 2515 | 12.92% | 6.44% | 0.00% | 0.918% | 29 | 5 |
| vLLM c1 r1 | 163 | 2262 | 2489 | 2489 | 2515 | 2515 | 10.16% | 1.04% | **1.04%** | 0.116% | 2 | 1 |
| vLLM c1 r2 | 163 | 2080 | 2489 | 2489 | 2489 | 2515 | 17.48% | 0.00% | 0.00% | 0.249% | 4 | 1 |
| vLLM c1 r3 | 163 | 2054 | 2489 | 2489 | 2489 | 2515 | 18.52% | 0.00% | 0.00% | 0.307% | 4 | 1 |

At c1 the excursions are **single samples, isolated, and periodic**. Our c1 rep 2
dips at 46.6 s, 78.4 s, 107.0 s, 135.6 s and 164.0 s — spacings of 31.8, 28.6,
28.6 and 28.4 s against a median request E2EL of **28.591 s** in the same file.
So the dips fall on the request period. That is one of the two readings below
and not yet a finding, for the reason the next paragraph gives.

**Two readings of those dips exist and this spec cannot settle which is right.**
The periodicity says boundary effect. The driver says thermal: every one of the
five carries a throttle reason in the same record, at `utilization.gpu = 96`.

```text
 46.60 s  2346 MHz  util 96  0x…0020        78.39 s  2164 MHz  util 96  0x…0020
107.00 s  2106 MHz  util 96  0x…0068       135.62 s  1859 MHz  util 96  0x…0020
164.01 s  2359 MHz  util 96  0x…0020
```

The two readings are not reconciled by the data available, and an earlier draft
of this spec asserted the first over the second without engaging it. Three facts
bound what can be said. The periodicity is real and matches the request period to
within 3 s over five events. The driver's own label is also real, and it is the
only physical attribution anybody recorded. And the labelling is **not uniform
across comparable excursions**: our c1 rep 1 dips on the same period — 48.83,
80.60, 109.28, 137.98 and 166.07 s, at 2177, 2320, 2210, 2359 and 2268 MHz — and
**two of its five carry `0x0000000000000000`**, no throttle bit at all, at the
same `utilization.gpu = 96` as the three that carry `0x20`. The five are **not
all of one depth**, and the weaker claim is the one the samples support: the
unlabelled 2210 MHz is deeper than two of the three labelled dips (2320 and
2268), while the unlabelled 2359 MHz is shallower than every labelled one. So a
dip of a depth the driver elsewhere labels arrives unlabelled, at the same
utilization and on the same period, which is the inconsistency. A single
mechanism that the driver labels only sometimes, or two mechanisms that coincide
on the request boundary, both fit.

**Distinguishing them needs temperature or power over the window, and neither
was collected** (§What was not measured). So this spec states the disjunction
rather than a cause. Nothing downstream depends on resolving it: the three-term
rule is justified by the mean-cost transfer bound below, which holds whatever
produced the dips, and the throttle rule fires on these windows either way.

**Drift is not 0.00% in all nine windows, and the one exception is the
phenomenon the drift term exists for.** vLLM c1 rep 1 reads **1.04%**, and the
shape is convention-free: **all 44 of its samples at 2515 MHz sit at indices 1
to 46 of 163**, counted from **zero** over the retained busy series, as the
sample list is indexed — 2 to 47 if counted from one, and the convention is
named here for the same reason the percentile convention is named above. So the
entire high state falls inside the first third however the
thirds are cut, and the last third is 52 of 54 (53 of 55 by time) at 2489. That
is a genuine one-way state change that never returns. It reads 1.04% under
every median-based split convention tried — count-floor thirds, count-ceiling
thirds, and time-based thirds of the retained busy series. Ours c1 rep 1 also
reads 1.04% when the thirds are taken over the **sampler's whole elapsed span**
rather than over the retained busy series, and 0.00% when they are taken over the
busy series; its first third is an even 25/25 split between the two states, so the
median lands on the boundary and the convention decides. The other seven windows
read 0.00% under all four conventions. Nothing here changes a verdict — 1.04% is
under the 2.0 ceiling — but the observed maximum is **~1.9x below the ceiling,
not infinitely below**, and §Design owes that margin an argument rather than the
claim that no drift exists.

The transfer bound follows from the mean, not the extremum. Time-weighted mean
SM clock against the median: −0.14%, −0.40%, −0.20% at c1 (ours) and +0.12%,
−0.25%, −0.31% (vLLM). The *entire* dip population is worth at most 0.40% of clock, so at
the tool's own physics ceiling of 1.0 percentage point of time per percentage
point of clock it can move a leg by at most 0.40%. The cross-arm offset on the
**mean** is **−0.101%**, against **0.000%** on the median. Measured throughput
agrees: 4.4068 / 4.4027 / 4.4040 tok/s across the three c1 legs, a range of
0.093%, with the ordering matching the mean-clock ordering (r1 > r3 > r2 on both
axes) and the magnitude coming in under the mean-clock prediction, as a partly
memory-bound decode should. `n = 3`, so that agreement corroborates and does not
prove.

**The proposed statistic is not vacuous, and this is the check that matters.**
The c8 windows carry a genuinely dispersed population — 28-29 of ~245 samples
below 95% of the median, in runs of up to 5 s, 9 to 10 separate runs — and their
p5–p95 band is 6.4% to 7.4%. A 5% band ceiling **still fails all three of them**.
The rule discriminates.

## Scope

**In scope.**

1. **The fleet change**, recommended and not performed here: grant
   `CAP_SYS_ADMIN` to the `worker` container of the `rc-worker` DaemonSet on
   `dgx`, `thor` and `orin`. This is a human decision about privilege on a shared
   host, argued under `## The fleet change` below.
2. **The instrument change**: replace `MAX_WITHIN_RUN_SPREAD_PCT` over
   `(max - min) / median` with a percentile band plus an explicit drift term, in
   `tools/bench/gpu_clock_state.py`, with the red-first tests below.
3. **The record reconciliation** the instrument change invalidates:
   `.agents/benchmarking.md` §"The clock is part of the measurement" and the
   argument block in [`bench-assert-clock-state.md`](bench-assert-clock-state.md).

**Out of scope.**

- Re-deriving the discarded Qwen3.8-27B c1 ratio. It is
  [#915](https://github.com/mudler/vllm.cpp/issues/915)'s, and **landing either
  half of this row does not enable it**: the nine captured windows carry
  non-benign throttle reasons and stay `DISCARD` on that rule alone. It needs
  fresh GPU time on a box where the clocks can be pinned, which is what the fleet
  half is for.
- `MAX_CROSS_ARM_OFFSET_PCT`, `MIN_BUSY_SAMPLES` and `MIN_BUSY_FRACTION`. They
  are untouched. The cross-arm rule passed perfectly on the very pairing this
  spec exists for, which is the reason to leave it alone.
- Any standing pin on a fleet box. `benchmarking.md` already forbids leaving a
  box pinned, and nothing here changes that.

## The fleet change

The minimal patch, on each of `manifests/{dgx,thor,orin}/rc-worker.yaml`, on the
`worker` container and not on the pod:

```yaml
        - name: worker
          image: ghcr.io/mudler/rc-worker:edge
          securityContext:
            capabilities:
              add: ["SYS_ADMIN"]
```

`privileged: true` also works and is strictly worse: it grants every capability,
disables seccomp and AppArmor, and relaxes the device cgroup, when one capability
is what the driver asks for.

**This is the patch to try, and it is not an established diagnosis.** That the
one capability is `CAP_SYS_ADMIN` rests entirely on the driver-source quotation,
which §Evidence records as an external reference nobody here has read at a
pinned revision; the behavioural evidence identifies only that container root
lacks something host root has. The `rc run` acceptance test in §Gates is what
falsifies it. An `LGC_RC` still 4 after the grant says the capability is a
different one, and the answer then is to identify which — not to reach for
`privileged: true`, whose cost is above and whose success would name nothing.

**Say the cost plainly.** Leased jobs run inside this same long-lived container,
so this grants `CAP_SYS_ADMIN` to every job anybody submits through `rc`, not
only to the worker. `CAP_SYS_ADMIN` is close to root-on-the-host in practice.
Against that: the pod already runs as root, already mounts a `hostPath`, already
holds cluster RBAC that can scale Deployments across every namespace, and the
submitters are the same people who hold `ssh` on the box. The marginal exposure
is small, and it is still a decision for the fleet's owner rather than for this
row.

**It is worth taking, and the reason is measured.** Pinning demonstrably works on
this hardware once the capability is there. On 2026-08-15, from the host, a
requested 2190 delivered a **flat 2184 MHz** over n=861 retained busy samples —
min 2158, median 2184, max 2184, a `spread_pct` of **1.19%** — with
`clocks_event_reasons.active = 0x0` for the whole series and persistence mode
`Disabled`. So pinning both passes the existing ceiling and removes thermal
throttling entirely, and it does so *below* the boost point rather than by
fighting it.

Two consequences a reader must carry:

- **A pinned number is ~12% slower in absolute terms.** 2184 against a sampled
  median of 2489 is a 12.3% clock deficit. That is the price of comparability and
  it is only payable when both arms pay it. Never set a pinned absolute beside an
  unpinned one.
- **Persistence mode is not the lever.** The pinned, flat, unthrottled series ran
  with persistence `Disabled`; every throttling window of 2026-08-19 ran with it
  `Enabled`. Persistence governs whether the driver stays resident between
  clients, not DVFS under load. It is also not known to be settable from a lease,
  for the `-pm` reason above.

## Design

Three named thresholds replace one, and each answers a different question about
the window.

    MAX_WITHIN_RUN_BAND_PCT = 5.0     # (p95 - p5) / median * 100
    MAX_WITHIN_RUN_DRIFT_PCT = 2.0    # |median(last third) - median(first third)| / median * 100
    MAX_EXCURSION_MEAN_COST_PCT = 1.0 # |median - mean| / median * 100

- **The band** replaces the range. It answers "was this window one state", which
  is the job `bench-assert-clock-state.md` states for the spread rule, and it
  answers it without letting one sample decide. 5.0 is carried over unchanged and
  deliberately: the argument for that number in the prior spec is an argument
  about how far apart two *states* may be, and it transfers to the band intact.
  It rejects the two-state failure that rule exists for (~26% either way) and the
  c8 windows above at 6.4-7.4%. It also accepts the only clean #543 capture, at
  3.32% on the band against 3.68% on the range — but **that acceptance carries no
  calibration weight**, because `p95 - p5 <= max - min` holds for every
  distribution, so any window the range accepted the band accepts too. It is
  stated as a consistency check with the prior spec, not as evidence for 5.0. The
  evidence for 5.0 is the prior spec's own two-sided argument, unchanged, plus
  the c8 rejection above, which is the only place the band and the range disagree
  about a verdict on real data.
- **The drift term** is what actually catches #543's within-boot failure — two
  probes eight minutes apart at 2398 and 1781 — in the case where the window
  straddles the transition and a band alone might not. Two independent arguments
  put 2.0 there, and the second is the one the observed maximum needs.
  **First**, 2.0 is twice the cross-arm offset ceiling, on the argument that a
  systematic move *inside* one arm may not exceed twice what is tolerated
  *between* arms. **Second, it must accept one steady-state change, and 1.04% is
  what one costs on this device.** Across the 1,690 retained busy samples of the
  nine windows this GPU sat in two states and only two: 2515 MHz (713 samples)
  and 2489 MHz (826), together **91.1%** of every sample taken. The remaining 151
  samples are scattered over 50 distinct values, none of them appearing more than
  10 times, and are transitions rather than states. The two states are 26 MHz
  apart, which is **1.04% of 2489** — exactly the drift vLLM c1 rep 1 reads, and
  not a coincidence: that window is a move from one steady state to the other and
  nothing else. A ceiling that rejected it would
  void an ordinary window for ending in the other steady boost state, which is
  DVFS working, not the #543 failure. So the ceiling has to sit above 1.04%, and
  2.0 leaves 0.96 points over the only such move this box makes. Nothing sits in
  the admitted band between one state change and a regime change: the next
  plateau anybody has recorded here is the pinned 2184 MHz, **12.3%** away, and
  #543's own within-boot disagreement is **~26%**. The margin is therefore ~1.9x
  over the observed maximum and ~13x under the failure, and it is not tightened
  below 1.04% for the reason just given.
- **The mean-cost term** is the honest bound on what the excursions can transfer
  into a ratio, and it is the term that makes dropping the range statistic safe.
  A window whose mean sits within 1.0% of its median cannot move a
  clock-proportional kernel by more than 1.0%, by the same physics argument
  `MAX_CROSS_ARM_OFFSET_PCT` already rests on. It is an absolute value because
  an upward excursion transfers exactly as much as a downward one. The nine
  windows read 0.116% to 1.039%, and **the c8 legs are not "at" the ceiling**:
  c8 r1 reads 1.039%, which is **above** 1.0 and fails this term on its own, and
  c8 r2 and r3 read 0.976% and 0.918%, which pass it. All three fail the band
  anyway, so no verdict turns on the distinction — which is why the term needs
  the dedicated case in §Tests rather than a listed window.

**Say the loosening plainly.** Dropping the range statistic does admit a class of
window the old rule refused, and it is a real class rather than a hypothetical
one. Take 155 samples at a flat 2489 MHz with a single sample at 1200: today's
`spread_pct` is **51.79%** and the window is refused; under the new rule the band
is **0.00%**, the drift **0.00%** and the mean cost **0.334%**, and it passes.
That is the intended outcome — one sample out of 155 can move a
clock-proportional leg by at most 0.334%, which is under every threshold in this
file — but it must be stated as a loosening and not discovered later.

**And the loosening has a partial co-guard that is not in this rule at all.**
The physically alarming version of that window — one where the GPU really was
held back — is often caught by the **throttle** rule, which this row leaves
untouched: `_throttle_offenders` refuses on any non-benign bit regardless of what
the clock statistics say, and that is exactly what fires on all nine windows of
2026-08-19, as §The defect records. On this evidence the deepest excursion of all,
c1 rep 2's **1859 MHz** sample at 25% below its median, arrives carrying `0x20`
and is refused on that bit alone.

**Say the limit of that co-guard too.** It is not a guarantee, and the same
evidence shows why: two of the five excursions in ours c1 rep 1 — 2210 MHz and
2359 MHz — arrive with `0x0000000000000000`, no bit set at all (§The defect). So
a deep sample the driver declines to label is possible, and for that sample the
mean-cost term is the only bound left. The claim this spec makes is therefore the
weak one: the throttle rule catches the labelled cases, and the mean-cost term
bounds the transfer of the unlabelled ones to under 1.0% whether or not anything
labels them.

`spread_pct` is **kept in the record and stops being a gate term.** Removing it
would silently rewrite the meaning of every clock record already committed, and a
reader comparing an old record with a new one needs the old field to still be
there. It is reported, not asserted — the same demotion the prior spec applied to
the transfer coefficient.

### Why not simply restore pinning and leave the gate alone

Because the gate is wrong on its own terms, and pinning does not make it right.
The band and the drift term are what a pinned window would pass *for the right
reason* — one state, no drift — rather than by the accident of never dipping.
And two of the three fleet devices may not get the capability: `thor` and `orin`
are the same manifest shape, and a future box need not be. An instrument that can
only produce a verdict on a pinned box is an instrument that stops working the
next time the access path moves, which is the failure this row is named after.

## Risks

- **It looks like widening an assertion to turn a red green, and that is
  forbidden.** It has to be argued, not asserted. The defence is that the
  statistic changes rather than its threshold: the ceiling stays 5.0, two new
  terms are *added*, and the tests below include a case the current rule misses
  and the new rule catches. A change that only ever loosens has no such case. If
  the reviewer cannot mutate the new rule into failing on a two-state window,
  this section is wrong and the change must not land.
- **Percentiles over a small window are coarse.** Under the linear-interpolation
  convention named above, at `MIN_BUSY_SAMPLES = 30` the p5 falls between the 2nd
  and 3rd sorted samples (`h = 1.45`) and the p95 between the 28th and 29th
  (`h = 27.55`). So a 30-sample window excludes the single most extreme sample at
  each end outright and weights the second-most-extreme at 0.55 and 0.45. It can
  hide one excursion at each end and half-hide a second. That is what the
  mean-cost term is for, and it is stated rather than hidden. Under nearest rank
  the p5 would be the 2nd sorted sample and the p95 the 29th; the convention is
  named because the cases below pin exact values and the two disagree by 0.05
  points on ours c8 r1.
- **The three-term rule can pass a window with a sustained mid-window step of
  under 2.0%.** The old rule passed one under 5%, so this residual is smaller
  than the one it replaces, but it is not zero. `MAX_CROSS_ARM_OFFSET_PCT` is
  what qualifies the ratio; this rule only establishes that each arm was one
  state.
- **The change does admit windows the old rule refused, and one class of them is
  named and measured.** 155 samples at a flat 2489 with a single sample at 1200
  is refused today at `spread_pct` 51.79% and accepted after the change at band
  0.00 / drift 0.00 / mean cost 0.334%. §Design argues why that is correct and
  §Tests case 5 pins it. The residual risk is that the throttle rule, which
  catches the labelled version of such a window, does **not** catch every version:
  two excursions in the 2026-08-19 evidence carry no throttle bit at all.
- **`n = 3` throughput corroboration.** The agreement between mean-clock ordering
  and throughput ordering rests on three legs of one workload on one boot. It is
  offered as corroboration of a physics bound, never as the argument for it.
- **Granting `CAP_SYS_ADMIN` widens the blast radius of any `rc` submitter.**
  Named above; the decision is the fleet owner's.

## Tests

`tests/tools/test_gpu_clock_state.py`, red before the change in every case.

1. **The case the current rule misses.** A synthetic window of 200 samples that
   steps from 2400 to 2280 at the halfway point and never returns: a 5.0%
   sustained two-state window with **no** excursion beyond it. The current
   `spread_pct` reads 5.13% against a 5.0% ceiling and only barely fails; shift
   the step to 2300 and it reads **4.255%** and **passes today**. On that second
   variant the new rule's band also reads 4.255% and passes, and the mean cost
   reads **0.000%** and passes, so **only the drift term catches it**, at 4.255%
   against a 2.0 ceiling. Red first on the second variant, and assert the term by
   name so the case cannot be satisfied by the band.
2. **The case the current rule fails wrongly — on the spread reason, and only
   on that one.** Replay our c1 rep 2 window verbatim from committed fixture
   data: `spread_pct` 26.36%, band 0.00%, drift 0.00%, mean cost 0.402%. The
   assertion is about **which** reason survives, never about `reasons` becoming
   empty, because this window also carries `SwThermalSlowdown` and `0x68` and the
   throttle rule fires on it independently (§The defect). Before the change,
   `clock_reasons` returns exactly two reasons, one naming the spread and one
   naming the throttle. After the change it returns exactly one, and that one
   names the throttle. Assert both lists element-wise: the spread/band reason is
   **gone** and the throttle reason is **byte-identical** across the change. An
   earlier draft of this case asserted an empty list, which is an outcome the
   real samples cannot produce and which no implementer could have made pass
   without substituting a hand-built distribution — forbidden by §Stop
   conditions. Pinned to real samples for the same reason.
3. **The case the new rule must still fail.** Replay our c8 rep 1 window
   verbatim: band 7.43% under linear interpolation (7.48% under nearest rank; the
   case pins the linear value). `reasons` non-empty before and after, and the
   band *message* must name the band rather than the range. As in case 2 the
   throttle reason is present on both sides and is asserted to be unchanged, so
   the case measures the band term and not the count.
4. **Window length must not decide the verdict — under EXTENSION, not
   repetition.** Duplicating a window changes neither its min, its max nor its
   median, so `spread_pct` of a window and of that window repeated twice are
   equal: measured on c1 rep 2, **26.356% both ways**, band 0.00% both ways. An
   earlier draft asserted they differ, which is arithmetically impossible. The
   defect is about **extending** a window with new samples, and the real leg
   states it executably. Over growing prefixes of c1 rep 2:

   | prefix n | `spread_pct` | band |
   |---:|---:|---:|
   | 30 | 6.79% | 2.33% |
   | 60 | 14.10% | 0.93% |
   | 90 | 16.43% | 0.44% |
   | 120 | 26.36% | 0.00% |
   | 155 | 26.36% | 0.00% |

   Assert that `spread_pct` is **non-decreasing** across that sequence and rises
   by more than 19 points from n=30 to n=155, while the band is
   **non-increasing** and falls to 0.00%. Assert also the degenerate identity the
   old draft got wrong — that duplication moves neither statistic — so a later
   reader cannot reintroduce it.
5. **The mean-cost term must be load-bearing on a case of its own.** No window in
   the table turns on it: loosening `MAX_EXCURSION_MEAN_COST_PCT` from 1.0 flips
   no listed verdict, because c8 r1 at 1.039% still fails the band at 7.43%. So
   the term needs its own pair, matched in everything except the quantity it
   measures. Both windows are n=155, flat at 2489 MHz with the low samples spread
   evenly through the window, so both read `spread_pct` 51.79%, band 0.00% and
   drift 0.00% — identical on every other term:

   | window | mean cost | new verdict |
   |---|---:|---|
   | 154 at 2489, **1** at 1200 | 0.334% | established |
   | 150 at 2489, **5** at 1200 | 1.671% | refused, naming the mean cost |

   Only the mean-cost term can separate them. This is also the case that proves
   §Design's bounded-loosening statement: the first row is refused today at
   51.79% of spread and accepted after the change. It is a synthetic
   distribution, permitted here because it is the *threshold provenance* case
   rather than a replay of a real verdict, and §Stop conditions forbids a
   hand-built distribution only for cases 2 and 3.
6. **`spread_pct` survives in the record.** Assert the field is still present and
   still equals `(max - min) / median * 100`, and that no gate expression reads
   it. The second half is the mutation: change the constant it used to be
   compared against and no test may move.
7. **Threshold mutation set**, matching the prior spec's discipline: each of the
   three constants moved in the passing direction must turn at least one case
   red, printing `git diff --stat` and any `compile_err` so a mutation that never
   applied cannot read as a pass. Case 5 is what discharges this rule for
   `MAX_EXCURSION_MEAN_COST_PCT`; without it that constant ships unguarded.

### The cases this change invalidates

`tests/tools/test_gpu_clock_state.py` carries **63** cases at `a4efbfb15`, and
several of them assert the statistic this row replaces. A bare "+7 cases" is
therefore the wrong expectation, in both directions: six existing cases are
**rewritten in place under the same names**, so they change meaning without
moving the count at all, and any of them that has to split into one case per term
moves the count in a way the added cases do not account for. A count check alone
cannot tell a rewritten assertion from an untouched one. The affected cases, by
name:

| case | file anchor `@ a4efbfb15` | what happens to it |
|---|---|---|
| `test_over_spread_window_is_not_established` | `:269-274` | rewritten: the message must name the band, and the fixture (`[1781, 2100, 2398]`) is rescored on the band |
| `test_the_threshold_is_inclusive_on_both_sides` | `:276-279` | rewritten against `MAX_WITHIN_RUN_BAND_PCT`; the n=3 fixture needs restating because p5/p95 interpolate |
| `test_a_within_run_defect_on_either_arm_propagates` | `:513-518` | rewritten: `"spread"` in the reason becomes the band term |
| `test_legs_at_different_clocks_widen_the_merged_spread` | `:546-551` | rewritten: the fold must widen the **band**, which is the property that actually has to hold across legs |
| `test_the_spread_threshold_accepts_the_clean_window_and_rejects_the_disagreement` | `:661-667` | rewritten onto the band; note that the accept half is a tautology (§Design) and the reject half is the load-bearing one |
| `test_the_spread_ceiling_is_not_held_to_the_offsets_criterion` | `:688-700` | rewritten; the residual argument transfers to the band, and the three-term rule changes what the residual is |
| `test_both_arms_spread_pct_are_surfaced_next_to_the_ratio` | `:520-528` | **kept unchanged** — `spread_pct` stays in the record, and this case is what proves it |
| `test_the_flat_degraded_window_is_also_established` | `:263-267` | **kept unchanged** — still established under all three new terms |

The expectation is therefore: 6 cases rewritten in place, 1 case (`:276-279`)
possibly split in two if the inclusive boundary needs a separate fixture per
term, and 7 cases added, against 2 named cases that must not move. The gate below
states it as a delta over a named baseline rather than as a bare count.

## Gates

- `tests/tools/test_gpu_clock_state.py` green. The baseline is **63** cases at
  `a4efbfb15`; the expected head count is **70**, being 63 minus 0 removed plus 7
  added, with 6 of the 63 rewritten in place under the same names — or **71** if
  `test_the_threshold_is_inclusive_on_both_sides` has to split into one case per
  term, which §The cases this change invalidates leaves open. Report the
  before and after counts explicitly. A count that does not move is a failure,
  not a pass; so is a count that moves without the named cases in §The cases this
  change invalidates having been touched, because that would mean the old
  assertions still pass and the statistic did not change.
- Full CPU gate, since the helper is standard-library-only and runs with no GPU.
- No GPU gate is owed by the instrument half. The fleet half owes one: a single
  `rc run` on `dgx:gpu0` reporting `LGC_RC=0`, followed by one
  `gpu_clock_state.py` window showing the pin held. That is the acceptance test
  for the capability grant and it cannot be run before the grant.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the fleet owner declines
  `CAP_SYS_ADMIN`. The instrument half still stands alone; the spec's framing
  does not.
- Stop if test case 1 cannot be made red before the change. That would mean the
  new rule catches nothing the old one missed, and the change is then a widening
  after all.
- Stop if the committed fixture for cases 2 and 3 cannot be taken from the
  2026-08-19 evidence. A hand-built distribution proves the arithmetic and not
  the claim.

## What was not measured

Three absences bound what this spec may conclude, and each of them touches an
argument above.

- **No temperature and no power exist for these nine windows.** `QUERY_FIELDS` in
  `tools/bench/gpu_clock_state.py` samples index, name, driver version, SM clock,
  maximum SM clock, applications clock, throttle reasons, persistence mode and
  utilization — and nothing thermal or electrical. There is therefore no way to
  test the driver's `SwThermalSlowdown` label against a die temperature, on these
  windows or on any window this helper has ever recorded. That is why §The defect
  states the dip disjunction instead of resolving it, and why the throttle rule
  is left exactly as it is: the only physical attribution in the record is the
  driver's, and second-guessing it would need an instrument that does not exist
  yet. Adding the two fields is a candidate follow-up and is **not** in this
  row's scope; it changes the record schema and owes its own spec. It is filed
  as [#1386](https://github.com/mudler/vllm.cpp/issues/1386) and listed under
  §Owed.
- **The "a 2190 pin will hold" expectation comes from a different boot.** The
  flat 2184 MHz series of 2026-08-15 ran on boot
  `03717c9d-63c8-4652-a8fe-a63d012c5718` over the host + `flock` path. All nine
  windows of 2026-08-19 ran on boot `3fd9745a-d25a-426c-ba3c-97c958a85515` inside
  a lease. This repository's own rule is that two arms on different boots are not
  comparable, and that rule does not stop applying because the second reading is
  an expectation rather than a ratio. So the pinning evidence establishes that
  the capability works on this hardware; it does not establish what a pin would
  deliver on the boot the nine windows were taken on. The `rc run` acceptance
  test in §Gates exists precisely because that has to be re-measured after the
  grant rather than inherited.
- **No sub-second structure.** The sampler runs at `--interval 1`, so a dip
  narrower than a second is invisible and a dip that reads as one sample may have
  been shorter. This does not weaken the mean-cost bound, which is an integral
  over what was observed, but it does mean the "single sample, isolated" language
  above describes the sampling and not necessarily the physics.

## Evidence

Raw sample windows, twelve `*.samples.json` files, at
`/mnt/nas_share/rc/q38bf16/out/{bench-20260819T035148Z,vllm-20260819T073125Z,vllm-20260819T095758Z}/`.
Per-leg records beside them; `CLOCKS.txt` in the first directory folds them. The
refusal is at `job.log:65-66` of `bench-20260819T035148Z` and at `job.log:19-20`
of both `vllm-*` directories. The pre-`-pm` persistence reading is
**`job.log:9`** — `job.log:8` is the `BOOT_ID=` line, and an earlier draft of
this spec named it by mistake.

Every statistic in this spec was recomputed from those raw files, and the
`clock_reasons()` transcript in §The defect is that function at `a4efbfb15` run
against the committed per-leg records.

Driver source quoted above is an **external reference, not verified locally**:
`NVIDIA/open-gpu-kernel-modules`, `kernel-open/nvidia/os-interface.c`
(`os_is_administrator`) and `kernel-open/common/inc/nv-linux.h` (`NV_IS_SUSER`),
at <https://github.com/NVIDIA/open-gpu-kernel-modules>. There is no checkout of
that repository on this host and no revision is pinned for it, so the quotation
rests on the reference and on the behavioural evidence in §The defect — host root
accepted `-lgc` on the same driver and the same box that refused container root —
rather than on a `file:line` anybody here has read. Confirming it owes either a
checkout at a named revision or a link to the exact blob.

Worker manifests: `infra-flux-kube`, `manifests/{dgx,thor,orin}/rc-worker.yaml`,
at `7ce8c77`.

## Owed

- The roadmap row. This spec is committed first, per "Spec before code"; the row
  is opened at claim.
- [#1354](https://github.com/mudler/vllm.cpp/issues/1354) stays open until both
  halves land, because either alone leaves the other's defect in place.
- The re-run of the Qwen3.8-27B bf16 c1 pairing, owned by
  [#915](https://github.com/mudler/vllm.cpp/issues/915) and
  [#979](https://github.com/mudler/vllm.cpp/issues/979). **Neither half of this
  row unblocks it on the existing captures**, because the throttle rule refuses
  all nine windows independently of the statistic this row replaces; it needs
  fresh GPU time on a box where the clocks can be pinned.
- Temperature and power in `QUERY_FIELDS`, so a future `SwThermalSlowdown` label
  can be checked against a die reading. Named in §What was not measured; it
  changes the record schema and owes its own row and spec. Tracked by
  [#1386](https://github.com/mudler/vllm.cpp/issues/1386), which carries the
  measured consequence — the five dips of ours c1 rep 1, two of them unlabelled
  — and is owed under this bullet until a row claims it.

## Now

`PROPOSED`. Nothing is implemented. The diagnosis is complete and the two changes
are independent; neither has been made. Revised 2026-08-19 after a hostile
review: the direction and the three-term design are unchanged and no threshold
moved, but the spec's account of its own evidence was wrong in three places — it
omitted the throttle rule that refuses all nine windows on its own, it asserted a
boundary-effect reading of the c1 dips over the driver's contrary label in the
same file, and two of its test cases (2 and 4) asserted outcomes the data cannot
produce. Those are repaired above, together with the false "drift is 0.00% in all
nine windows" claim, the missing mean-cost case, the unnamed percentile
convention, the uninventoried test rewrites, and the unstated absences.
