# ENV-LEASE-GPU-CAPABILITY — the ask is `CAP_SYS_ADMIN`, and it is missing from the worker's BOUNDING set on every fleet device measured

Issue: [#1354](https://github.com/mudler/vllm.cpp/issues/1354) — `nvidia-smi -lgc`
is refused inside an `rc` lease, in a job running as root.
Row: `ENV-LEASE-GPU-CAPABILITY` (proposed; the roadmap row is opened at claim).
Amends: [`lease-clock-pinning.md`](lease-clock-pinning.md) §Evidence, which
records its driver-source quotation as "an external reference nobody here has
read at a pinned revision" and names what confirming it owes — "either a
checkout at a named revision or a link to the exact blob". This row pays that,
and adds the measurement of the container side that no record held.

**Read this first, because two things changed under this issue and neither is
what the issue says.**

1. **This row produces no number and recovers no ratio.** It identifies a
   capability. The nine 2026-08-19 windows carry a second, independent throttle
   refusal that no capability grant removes from archived data, exactly as
   [`clock-gate-route.md`](clock-gate-route.md) §What no route recovers states.
2. **Clock pinning is no longer unavailable everywhere.** On 2026-08-21 the
   `BENCH-QWEN38-27B-SOTA` campaign pinned `dgx:gpu0` from the leased HOST and
   held 0.29% SM-clock spread over a ten-minute decode load (#1354 comment,
   2026-08-21T14:45Z). That is Route A of `clock-gate-route.md`, demonstrated.
   It does **not** repair the pod path, which is the default path for every row
   without that campaign's scoped `ssh` authorization, and it is the pod path
   this row is about.

## Scope

**In scope.** Name the missing capability, at a pinned revision on the driver
side and by measurement on the container side, so the fleet owner receives a
patch and an acceptance test rather than a symptom.

**Out of scope.** The `rc` worker change itself, which lives in
`github.com/mudler/resource-controller` and `infra-flux-kube` and over which this
row has no authority. The spread-rule statistic, owned by
[`lease-clock-pinning.md`](lease-clock-pinning.md) half two. Any benchmark
number.

## Half one: the driver side, now pinned

`nvidia-smi -lgc` reaches the driver through NVML's
`nvmlDeviceSetGpuLockedClocks`. NVML and `nvidia-smi` are closed source, so the
control they issue cannot be named from a pinned blob. What **can** be named,
and now is, is the whole privilege surface the driver exposes to Linux.

At `NVIDIA/open-gpu-kernel-modules` tag **`580.173.02`** — the driver version
`dgx:gpu0` reported in the refusing job (`job.log:9`, `580.173.02`) — the OS
abstraction layer contains **exactly three** capability tests and nothing else:

| Test | Blob and line | Reached by |
|---|---|---|
| `capable(CAP_SYS_ADMIN)` | `kernel-open/common/inc/nv-linux.h:537` | `NV_IS_SUSER()`, and through it `os_is_administrator()` |
| `capable(CAP_PERFMON)` | `kernel-open/nvidia/os-interface.c:390` | `os_check_access(RS_ACCESS_PERFMON)` |
| `capable(CAP_SYS_NICE)` | `kernel-open/nvidia/os-interface.c:397` | `os_check_access(RS_ACCESS_NICE)` |

```c
// kernel-open/nvidia/os-interface.c:377-381
// return TRUE if the caller is the super-user
NvBool NV_API_CALL os_is_administrator(void)
{
    return NV_IS_SUSER();
}

// kernel-open/common/inc/nv-linux.h:537
#define NV_IS_SUSER()                   capable(CAP_SYS_ADMIN)
```

`os_check_access` closes with `default: { return NV_FALSE; }`
(`os-interface.c:399-401`), so an access right that is neither `PERFMON` nor
`NICE` is refused outright rather than falling through to the administrator
test. A privileged control that is not one of those two named rights therefore
reaches `os_is_administrator`, and that is `CAP_SYS_ADMIN`.

**Three consequences, and the third is the one to act on.**

- The refusal is **not about `uid`**. `capable()` tests the calling process's
  effective capability set. A container process can be `uid=0` and fail it, which
  is precisely what the job reported.
- The candidate set is **bounded at three**. The driver has no fourth Linux
  privilege axis to discover, so the ask cannot be open-ended.
- `CAP_PERFMON` and `CAP_SYS_NICE` are the two named narrow rights, and they
  cover profiling counters and thread priority. Neither names clock control.
  `CAP_SYS_ADMIN` is the remaining test and the one to request.

**What this does NOT establish, stated rather than buried.** That
`nvmlDeviceSetGpuLockedClocks` calls `os_is_administrator` rather than
`os_check_access` is an inference from the enumeration above, not a read of the
call site, because the call site is in closed source. The enumeration is what
makes the inference tight: were it `os_check_access`, the right would have to be
`PERFMON` or `NICE`, and §The ask's acceptance test discriminates all three.

## Half two: the container side, now measured

The container half was never measured; it was read off manifests.
`lease-clock-pinning.md` §The defect derives it from `infra-flux-kube`
`manifests/{dgx,thor,orin}/rc-worker.yaml` at `7ce8c77`, where the `worker`
container declares no `securityContext` and therefore takes the default OCI
capability set. This row measured it from inside two leases.

`rc run` on `thor:gpu0` (job `34e3fb43`) and `orin:gpu0` (job `dd72fa7b`),
2026-08-21T17:01Z, both as `uid=0(root)`:

```text
CapInh: 0000000000000000
CapPrm: 00000000a80425fb
CapEff: 00000000a80425fb
CapBnd: 00000000a80425fb
CapAmb: 0000000000000000
```

**Byte-identical on both devices**, and `0x00000000a80425fb` is the default
14-capability OCI set: `CAP_CHOWN`, `CAP_DAC_OVERRIDE`, `CAP_FOWNER`,
`CAP_FSETID`, `CAP_KILL`, `CAP_SETGID`, `CAP_SETUID`, `CAP_SETPCAP`,
`CAP_NET_BIND_SERVICE`, `CAP_NET_RAW`, `CAP_SYS_CHROOT`, `CAP_MKNOD`,
`CAP_AUDIT_WRITE`, `CAP_SETFCAP`. Against the three tests above:

```text
  CAP_SYS_ADMIN    Eff=no Prm=no Bnd=no
  CAP_PERFMON      Eff=no Prm=no Bnd=no
  CAP_SYS_NICE     Eff=no Prm=no Bnd=no
```

**`CapBnd` is the load-bearing column, and it is why no in-job workaround
exists.** The bounding set is an upper limit on what the process or any of its
descendants can ever hold. A capability absent from `CapBnd` cannot be regained
by `setcap`, by a setuid binary, by `NoNewPrivs` being 0, or by re-execing. So
this is a container **configuration** fact and not a job-authorship one, and no
amount of care inside a leased job can route around it. `NoNewPrivs: 0` and
`Seccomp: 0` in the same read rule out a seccomp filter as an alternative
explanation.

`thor:gpu0` reproduced the refusal itself on its own driver
(`595.78`, `NVIDIA Thor`), which the issue had only from `dgx`:

```text
$ nvidia-smi -lgc 1000
The current user does not have permission to change clocks for GPU 00000000:01:00.0.
LGC_RC=4
```

So the refusal is now measured on **two** boxes with **two** driver versions
(`580.173.02` on dgx, `595.78` on thor), and the capability mask is measured on
**two** boxes. The manifest-derived fleet-wide claim is a measurement rather than
an inference.

### What the probe ruled out, each by a reading rather than an assumption

- **Not `NVIDIA_DRIVER_CAPABILITIES`.** Both workers carry
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, and every NVML **read** in the
  same job succeeded (`READ_RC=0` on thor, a full `--query-gpu` row returned).
  That variable selects which userspace driver components the runtime injects. No
  value of it adds a Linux capability.
- **Not the device nodes.** `/dev/nvidiactl` is `crw-rw-rw- root root` and
  present on both, alongside `/dev/nvidia0`. The refusal is not a file-permission
  or device-cgroup denial.
- **Not an `NVreg` knob.** `/proc/driver/nvidia/params` is readable in the lease
  and reports `RmProfilingAdminOnly: 1` on both boxes. That parameter governs
  **profiling counters**, which is the `RS_ACCESS_PERFMON` axis above, and it is
  a different gate from clock control. No parameter in that file names clock
  control, so nobody should reach for it expecting `-lgc` to start working.
- **Not the driver refusing a containerised caller as such.** The same driver on
  the same box accepted `-lgc` from the host on 2026-08-15 and again on
  2026-08-21. Host root holds `CAP_SYS_ADMIN`; container root does not.

### The `-pm` reading is still not falsified, and this probe did not falsify it

`lease-clock-pinning.md` §The defect records that the 2026-08-19 harness ran
`nvidia-smi -pm 1 2>&1 | tail -1`, that `| tail -1` discards a per-GPU failure
line, and that **no NVML write is known to succeed inside a lease**. This probe
captured the full output of `-pm`, `-ac`, `-rac` and `-am` rather than a tail,
and it still cannot retire that sentence:

```text
--- 7b nvidia-smi -pm Disabled (set to the value it already has) ---
Persistence mode is already Disabled for GPU 00000000:01:00.0.
All done.
PM_RC=0
```

`already Disabled` is `nvidia-smi` short-circuiting before it issues the write,
so `PM_RC=0` is a no-op and not a successful NVML write. `-ac` and `-rac`
returned `The requested functionality has been deprecated.` with `rc=0` on this
Tegra part, which is likewise not a write. **The sentence stands, and this row
neither confirms nor removes it.** Retiring it needs a device whose pre-state
differs from the value being set; the probe deliberately sets each control to
the value it already holds so that it cannot mutate a shared box, and that
design is what costs it the answer. A cheap follow-up is named under `## Owed`.

## The ask

**One line, on the `worker` container and not on the pod**, in
`infra-flux-kube` `manifests/{dgx,thor,orin}/rc-worker.yaml`:

```yaml
        - name: worker
          securityContext:
            capabilities:
              add: ["SYS_ADMIN"]
```

**The acceptance test, which is what makes this falsifiable.** Submit through
`rc run` and read `LGC_RC`:

```sh
rc run -d dgx:gpu0 -- bash -c 'nvidia-smi -lgc 2100; echo LGC_RC=$?; nvidia-smi -rgc'
```

- `LGC_RC=0` — the diagnosis was right and the row closes.
- `LGC_RC=4` still — the control does **not** reach `os_is_administrator`, and
  the answer is then `PERFMON` or `NICE` by the enumeration above. Try
  `add: ["PERFMON"]` next. **Do not reach for `privileged: true`**, whose success
  would name nothing and whose cost is below.

**If the fleet owner wants least privilege first**, `add: ["PERFMON"]` is the
cheaper probe and the same test discriminates it. It is the narrower grant and
the enumeration says it is one of only two alternatives.

**`privileged: true` also works and is strictly worse.** It grants every
capability, disables seccomp and AppArmor, and relaxes the device cgroup, where
one capability is what the driver asks for.

**Say the cost plainly.** Leased jobs run inside this same long-lived container,
so the grant reaches every job anybody submits through `rc`, not only the worker.
`CAP_SYS_ADMIN` is close to root-on-the-host in practice. Against that: the pod
already runs as root, already mounts a `hostPath`, already holds cluster RBAC
that can scale Deployments across every namespace, and the submitters are the
same people who hold `ssh` on the box. The marginal exposure is small. It is
still the fleet owner's decision and not this row's.

## What today's host-path result changes about the priority

It lowers the urgency and it does not remove the defect. Three readings, all
narrow:

- **A ratio is derivable on `dgx:gpu0` again**, by the host path, under a lease,
  for a campaign that holds the scoped authorization. That is a real repair to
  the measurement programme.
- **The authorization is campaign-scoped.** `.agents/developer-preferences.md`
  grants `rc hold` plus `ssh` to `BENCH-QWEN38-27B-SOTA` and states "the standing
  rule above is unchanged for every other row". So for every other row the pod
  path is still the only path, and the pod path is still refused.
- **The grant would retire the exception rather than manage it.** With the
  capability in the manifest, `-lgc` works from `rc run` and no row needs a
  scoped `ssh` authorization to pin a clock. That is the argument for taking it
  even though a workaround now exists for one campaign.

## The settle-and-hold alternative was NOT attempted, and why

`#1354` option 2 asks for a thermal settle-and-hold procedure that reaches under
5% within-run spread without pinning, **demonstrated rather than assumed**. This
row did not attempt it, and the reason is that it could not have been
demonstrated honestly today:

- `dgx:gpu0` — the only box the 5% failure was ever measured on, and the only
  GB10 — was held by `BENCH-QWEN38-27B-SOTA` for the whole session with a live
  server container on it. A thermal procedure measured on a contended box is not
  a procedure.
- `rc hold` is forbidden to this row by the dispatching operator, and the host
  `ssh` authorization is scoped to another campaign.
- `thor:gpu0` and `orin:gpu0` were idle and are the wrong instrument: both report
  `[N/A]` for `clocks.max.sm` and `clocks.sm` under `--query-gpu`, so the
  quantity the 5% rule bounds is not readable there at all.

**And the demonstrated host-path pin makes it the wrong thing to buy.** A settle
procedure would be a second, weaker method for a job that a one-line manifest
change does properly. It stays a fallback for the case where the grant is
declined, and it is recorded under `## Owed` rather than guessed at here. Naming
a procedure this row did not run would be exactly the "assumed rather than
demonstrated" the issue refuses.

## The rule question, and why this row does not touch a threshold

`#1354` option 3 asks whether lease-measured pairs should use a different clock
rule. **This row proposes no threshold change**, and that is deliberate:

- The rule question is already argued, twice, on evidence, in merged specs.
  `lease-clock-pinning.md` half two replaces `spread_pct` with three named terms
  because a range statistic over raw samples scores the deepest single excursion
  and its numerator only grows with window length. `clock-gate-route.md` §Is it a
  forbidden widening states the case for and the three grounds against, and
  §Scoping the throttle rule is REFUSED declines the half that would actually
  turn the red green. Neither is implemented; neither needs this row.
- **Nothing here would be argued from physics.** A capability diagnosis is not
  evidence about a statistic, and proposing a ceiling change on the strength of
  it would be the widening the issue forbids.

`MAX_WITHIN_RUN_SPREAD_PCT` stays `5.0`. `BENIGN_THROTTLE_MASK` stays
`0x1 | 0x2 | 0x100`. No constant in `tools/bench/gpu_clock_state.py` moves.

## Risks

- **The capability may be the wrong one.** The NVML call site is closed source
  and the identification rests on an enumeration plus an inference. §The ask's
  acceptance test is what falsifies it, and the fallback is named rather than
  left to a reach for `privileged: true`.
- **`dgx:gpu0`'s capability mask is not measured.** thor and orin agree
  byte-for-byte and the manifests are the same shape, but the box the figures
  come from was busy. A probe is queued; until it returns, dgx's mask is
  inferred. Recorded under `## Owed`.
- **The grant is a fleet-wide privilege increase** and reaches every submitter.
  Stated in §The ask so that it is a decision and not a side effect.
- **A pinned box is a shared-host mutation.** If the grant lands, every pinning
  run owes a reset trap, and `benchmarking.md` forbids leaving a box pinned. The
  probe in this row resets unconditionally for that reason.
- **Two driver versions, two architectures, `n = 1` job each.** The capability
  mask is a container property and is not expected to vary by run, but no repeat
  was taken.

## Tests

This row changes no `src/`, `include/`, `tests/` or `tools/` file. `git diff
origin/main..HEAD` over those paths is empty and that emptiness is the claim: a
capability diagnosis that moved a threshold would be the thing #1354 forbids.

The instrument work each half owes when it is implemented is unchanged and lives
in the specs that own it — `lease-clock-pinning.md` §Tests for the three-term
statistic, `clock-gate-route.md` §Tests for the additive terms. This row adds
none and weakens none.

## Gates

- The driver-source chain is quoted at a **named tag** with `file:line`, and the
  tag matches the driver version of the box that refused. `## Evidence` carries
  the fetch.
- The container capability mask is read from `/proc/self/status` **inside a
  lease**, on more than one fleet device, with the raw hex recorded so a reader
  can decode it independently.
- The refusal is reproduced on a device other than the one in the issue.
- No constant, ceiling or assertion in `tools/bench/gpu_clock_state.py` changes.
  `git diff origin/main..HEAD -- tools/` is empty.
- `scripts/agent-preflight.sh --staged` passes before commit.

## Stop conditions

- **Stop if the acceptance test returns `LGC_RC=4` after a `SYS_ADMIN` grant.**
  Do not escalate to `privileged: true`. Report the result and try `PERFMON`.
- **Stop rather than infer `dgx:gpu0`'s mask** if the queued probe does not run.
  A pending measurement is pending, not absent and not confirmed.
- **Stop rather than propose a settle procedure that was not run on a quiet
  GB10.** The issue asks for a demonstration and there is no partial credit.
- **Stop rather than move a threshold.** No red in this row is repaired by a
  wider assertion.

## What was not measured

- **`dgx:gpu0`'s capability mask.** Queued behind a live campaign at submission
  time. Its *refusal* is measured, three times on 2026-08-19.
- **The NVML call site.** Closed source. Bounded, not read.
- **Whether any NVML write succeeds in a lease.** §The `-pm` reading explains why
  this probe's own no-op design forfeited the answer.
- **Any thermal or settle behaviour.** §The settle-and-hold alternative.
- **`CAP_PERFMON`'s effect**, and `CAP_SYS_ADMIN`'s. Neither can be measured
  without the grant; that is the whole shape of this issue.
- **Whether the `-lgc` refusal costs anything on thor or orin.** Both report
  `[N/A]` for SM clocks, so the ceiling the refusal would matter to is not
  readable there.

## Evidence

Driver source, fetched 2026-08-21 from
`https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/580.173.02/`:
`kernel-open/common/inc/nv-linux.h` (1852 lines, `NV_IS_SUSER` at `:537`) and
`kernel-open/nvidia/os-interface.c` (2738 lines, `os_is_administrator` at
`:378-381`, `os_check_access` at `:383-404`). `grep -n 'capable(\|CAP_'` over
both returns the four hits tabulated in §Half one and no others, which is what
bounds the candidate set at three.

Lease probes, 2026-08-21T17:01Z, `rc` jobs `34e3fb43-ef4c-49c4-98ae-46437ef10e2e`
(`thor:gpu0`, worker `rc-worker-m4d7t`, kernel `6.8.12-1021-tegra`, NVRM
`r595_00`) and `dd72fa7b-dab6-4406-9c8f-83d0de75d917` (`orin:gpu0`, worker
`rc-worker-lnvw6`, kernel `5.15.148-tegra`, NVRM `540.4.0`). Both `uid=0(root)`.
`orin:gpu0` carries **no `nvidia-smi`** (`READ_RC=127`), so its contribution is
the capability mask only and not the refusal.

The 2026-08-21 host-path pin is the `#1354` comment of 2026-08-21T14:45Z, taken
by `BENCH-QWEN38-27B-SOTA` under its own authorization. It is cited here and was
not re-derived by this row.

Worker manifests: `infra-flux-kube`, `manifests/{dgx,thor,orin}/rc-worker.yaml`
at `7ce8c77`, quoted from `lease-clock-pinning.md` §The defect and not re-read by
this row.

## Owed

- The roadmap row. This spec is committed first, per "Spec before code".
- [#1354](https://github.com/mudler/vllm.cpp/issues/1354) stays open until the
  grant lands or is declined. This row supplies the ask; it does not close the
  issue.
- **`dgx:gpu0`'s capability mask.** The same probe, queued at submission time
  behind `BENCH-QWEN38-27B-SOTA`. One `rc run` of about sixty seconds.
- **Whether any NVML write succeeds in a lease**, which needs one control whose
  pre-state differs from the value set — `nvidia-smi -pm 1` on a box reading
  `persistence_mode Disabled`, with the full output kept and the prior value
  restored. It would either retire or confirm `lease-clock-pinning.md`'s "no NVML
  write is known to succeed inside a lease".
- **The settle-and-hold procedure**, #1354 option 2, if and only if the grant is
  declined. It needs a quiet GB10 and it owes its own row.
- The instrument halves, unchanged and owned elsewhere:
  [`lease-clock-pinning.md`](lease-clock-pinning.md) half two and
  [`clock-gate-route.md`](clock-gate-route.md) Routes B and C2.
- [#1386](https://github.com/mudler/vllm.cpp/issues/1386), the thermal and
  electrical fields, unchanged by this row.

## Now

`PROPOSED`. The diagnosis is complete on two of three fleet devices and the ask
is a one-line manifest change this row has no authority to make. Nothing is
implemented, no constant moved, and no number was produced. Written 2026-08-21.
