# GPU lease methodology: a fleet device is reached by a lease

Row: `ENV-GPU-LEASE-METHODOLOGY`.
Issues: [#1129](https://github.com/mudler/vllm.cpp/issues/1129),
[#1130](https://github.com/mudler/vllm.cpp/issues/1130).

## Scope

State one rule in the root policy file:
[resource-controller](https://github.com/mudler/resource-controller), whose
client is `rc`, manages the shared GPUs, and claiming a FLEET DEVICE through it
is the required path for GPU work, replacing the `flock` file mutex as the
default. On a GPU that is not a fleet device, the file mutex remains the
instruction.

In scope:

- `AGENTS.md` gains the rule and the link to the tool.
- `.agents/environment.md` gains the conditional in its how-to.
- The records this row's own measurements made stale, which is the second half
  of `## Records` in `AGENTS.md`: a correction rides in the pull request whose
  change made the record stale.

Out of scope:

- Copying the `leasing-a-gpu` skill into this repository. A copy of another
  document goes stale without saying so.
- Any checker change. This row changes no checker semantics, so it owes no
  red-before mutation under `## Changing the rules or a checker`.
- The oracle migration owed by #1129. The row that takes #1129 owns it.

## The defect, corrected against the tree

The brief for this edit said the root file's Commands section and its
GPU-adjacent prose point at the old mechanism. That is not what the tree holds.
Measured at `36381f346`, `AGENTS.md` matches `flock` zero times and names no
mutex, no lock path, and no `ssh` procedure at all.

The real defect is an absence, and it is the stronger argument. `AGENTS.md`
says of itself that it "contains the complete policy" and that files under
`.agents/` "cannot add or weaken a rule in this file". The lease requirement
lives only in `.agents/environment.md`, which is a task guide. Under the root
file's own terms it is therefore guidance and not a rule. A reader who follows
the root file alone is told nothing about how to reach a GPU.

## Why the rule is conditional, and what the condition is

The developer stated the requirement as "when the host has it, it is the
default". The rule stays conditional, because the hosts are not identical and a
reader on a box that the fleet does not manage still needs an instruction.

**The first revision of this rule keyed that condition on the wrong property,
and the wrong property reopens the failure the rule exists to close.** It said
"when the host has `rc`", and `.agents/environment.md` told the reader to test
it with `rc devices`. That tests whether the machine you are typing on carries
the client. It does not test whether the device is fleet-managed. A host with no
client but with `ssh` access to `dgx` or `thor` was therefore routed to the file
mutex alone, which is exactly the 2026-08-17 collision recorded below: one
session on `flock` over `ssh` while another held the same box through `rc`, two
mutexes that cannot see each other, and a VOIDed speed axis.

The failure mode is worse than a gap, because the fallback is the dangerous
side. The old text named `command not found` as the signal for the second case.
A controller that is unreachable or refuses authentication answers with neither
that string nor a device list, and the likelier misread of that silence is "no
`rc` here, take the `flock`". The controller does lose contact: `thor:gpu0` read
`unknown (no contact 1m0s)` during this row's review.

So the rule now names the property that matters, which is whether the DEVICE is
fleet-managed, and it names `dgx:gpu0`, `thor:gpu0` and `orin:gpu0` so that
membership is checkable with no client at all. On a fleet device, a missing
client and an unreachable controller both mean get the client or report the
controller down. Neither is a route to `ssh` plus `flock`. The genuine
non-fleet case keeps its instruction, and it is now scoped to a GPU that is not
a fleet device rather than to a shell that lacks a binary.

## Why the bypass is not a style preference

Two mutexes that do not exclude each other are worse than one, and the cost is
measured. On 2026-08-17 one session took the file mutex over `ssh` while another
session held the same box through `rc`. Neither mutex excluded the other.
`.agents/specs/minimax-music3.md` §13.10 retains a whole speed axis as VOID
because of it, and `.agents/benchmark-record.md` records the window in which the
fleet reported `thor:gpu0` free while it was in use. That is the #777 failure
again, in which this repository carried two GPU mutexes and neither serialised
the other.

## What a lease can carry

The limit is now precise, measured on 2026-08-17 through two
`rc run -d dgx:gpu0` jobs and recorded in `.agents/environment.md`. A lease
carries bytes and not executables. The worker reads and writes the shared
`/workspace`, refuses direct execution from it because the mount pins
`file_mode=0664`, and runs staged content through `sh FILE`, through the dynamic
loader, or after a copy to `/tmp`.

**The last clause of this section said the worker cannot produce or fetch a
runtime, because it has no compiler, no downloader and no Python. That is a
`dgx:gpu0` reading and it does not generalise.** Later the same day, five
`rc run` jobs on `thor:gpu0` measured a worker running as `uid=0(root)` with
`/usr/bin/gcc`, `/usr/bin/python3` and a working `apt-get`, and a `torch` and
`triton` tree staged on `/workspace` imported, initialized CUDA and compiled and
ran a Triton kernel. See
[`lease-runtime-staging.md`](lease-runtime-staging.md)
([#1146](https://github.com/mudler/vllm.cpp/issues/1146)), which also states what
that result does not establish: it is `thor:gpu0` at capability (11,0) only, the
GB10 is `sm_121a` and UNMEASURED, and the pinned vLLM oracle is not in that tree.
The oracle itself was then built inside a lease on `dgx:gpu0`, on 2026-08-18
([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
[`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). A model run is still
untested.

## The correction has to reach the spec that owns the blocker

This branch wrote the strong claim in `a751f8887` and falsified part of it in
`36381f346`, and that correcting commit touched `.agents/NOW.md` and
`.agents/environment.md` only. `.agents/specs/mtp-k-gt-1.md` kept the
unqualified "no vLLM leg of any row can run on `dgx.casa` by ANY
lease-compliant path" in its `## Owed` table and again in its `## Now`, beside
the now-measured-false reason "could not start it if it could".

That spec is where the blocked row's owner looks, because #1129's index row
names `SPEC-MTP-K-GT-1` as the owning row. A reader who follows "Resume this row
only after #1129 has a path" and then reads "could not start it if it could"
concludes that staging is futile, and never attempts the route that measured
green. Both sites therefore take the calibrated form that
`.agents/environment.md` already carries: the probe NARROWS #1129 and does not
close it, the relocated virtual environment is UNMEASURED, and the load-bearing
reason is that nothing has staged a runtime on the NAS.

**The UNMEASURED clause in that form was answered on the same day, and both sites
were corrected again.** A relocated runtime does start inside a worker, on
`thor:gpu0`. See [`lease-runtime-staging.md`](lease-runtime-staging.md) and
#1146.

**The ORACLE clause was answered the next day, and both sites were corrected a
third time.** On 2026-08-18 a lease on `dgx:gpu0` BUILT the pinned oracle from
source against a staged CUDA toolkit, and the installed wheel imports and
reports `cuda True NVIDIA GB10`
([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
[`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). The correction that
reaches the blocked rows is the exact one: they are unblocked for the BUILD step
and still blocked for a MODEL RUN. A reader who rounds that to "unblocked"
schedules a measurement that cannot be taken.

The same substitution repairs the derivation in the how-to. The old sentence
read "no host toolchain, the worker has no compiler, SO no lease-compliant
path". Those premises stopped entailing that conclusion once the later probe
showed that staged bytes execute. The claim is still true and its reason is
different, so the reason is what the text now states.

## Risks

The one checker this edit can break is `test_gpu_lock_one_truth` (#777), which
requires exactly one `**GPU mutex:**` bullet in `.agents/environment.md` and
requires that bullet to name both `${GPU_LOCK}` and `$HOME/gpu.lock`. The edit
adds no second mutex statement and does not touch that bullet, which already
reads "this runs INSIDE an `rc` lease, never instead of one". `AGENTS.md` is
outside that checker's scanned set, and the rule there spells the canonical
`${GPU_LOCK:-$HOME/gpu.lock}` so it can never read as a second truth. Naming the
device rather than the client does not change that count, because the fallback
stays one sentence in each file and neither is a `**GPU mutex:**` bullet.

The device list is a second record of a fact that `rc devices` also reports, so
it can go stale when the fleet changes. That is accepted rather than avoided.
The alternative is a condition a reader can only evaluate with the client, and
this row exists because that condition sent a client-less reader to the wrong
mutex. The list sits beside the fleet table in `.agents/environment.md`, which
already carries the same three names and the date it was read. Both files also
say the list is a LOWER bound, so a fleet that grows past it widens the rule
rather than exempting the new device.

If the conditional rule cannot be written without a second mutex statement, stop
and return `NEEDS_DECISION`. Do not weaken the checker.

## Gates

```sh
scripts/agent-preflight.sh
python3 -m unittest discover -s tests/scripts -p 'test_gpu_lock_one_truth.py'
```

`test_gpu_lock_one_truth` is the focused gate and stays green. The full preflight
is the row gate.

## Evidence

- Probe jobs `1cb56f84-62bf-4c90-b138-9bd4c3b0617a` and
  `c692d5a0-ec3d-4498-86e4-e86a2864e91a` on `dgx:gpu0`, 2026-08-17.
- `.agents/environment.md`, "The lease carries bytes, and the exec bit is a
  mount option".
- `.agents/specs/minimax-music3.md` §13.10 for the VOIDed speed axis.

## Stop conditions

- Stop if the edit needs a second mutex statement. Return `NEEDS_DECISION`.
- Stop if `test_gpu_lock_one_truth` goes red. Never widen it to pass.

## Owed

- #1129 is now closed, and its recorded cause was falsified on 2026-08-17. The
  "UNMEASURED" clause this line used to carry is answered: a relocated CUDA
  runtime does start inside a worker, on `thor:gpu0`. The follow-on claim that
  nothing has staged the ORACLE is answered too, on 2026-08-18 and on
  `dgx:gpu0`, where a lease built the pin and imported it
  ([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
  [`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). What is still owed is
  a MODEL RUN inside a lease. `ENV-LEASE-RUNTIME-STAGING` with
  [#1146](https://github.com/mudler/vllm.cpp/issues/1146), and
  `ENV-ORACLE-WHEEL-IN-LEASE` with #1185, own the rest.

## Now

The rule is stated in `AGENTS.md` and the conditional is in
`.agents/environment.md`, keyed on the device and naming the three fleet
devices. The narrowing of #1129 now reads the same way in
`.agents/environment.md` and in `.agents/specs/mtp-k-gt-1.md`, so the blocked
row's owner is told that staging is untried rather than futile. Staging was then
tried, and it worked on `thor:gpu0`
([`lease-runtime-staging.md`](lease-runtime-staging.md), #1146). The pinned
oracle itself was then built inside a lease on `dgx:gpu0`
([`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md), #1185), so the blocked
rows are unblocked for the build step and still blocked for a model run. The
next step is a model run.
