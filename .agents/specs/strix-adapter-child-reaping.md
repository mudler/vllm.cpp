# Reap owned Strix adapter descendants

Row: `BACKEND-GATE-ROCM-SGLANG`.
Issue: [#3108](https://github.com/mudler/vllm.cpp/issues/3108).
Parent: [matched comparison](strix-four-engine-qwen3-4b.md), #3053.
Base: `4666c0d3f50202c61fde3fa75c97c33c35af930d`.

## Now

The user requests autonomous validation repair and matched Strix optimization.
The controller preflight repair passes actual Strix validation, but teardown fails.
The controller lifecycle implementation and CPU process fixtures are prepared.
Fresh review, immutable-head full verification, and actual Strix acceptance remain required.
Use one scoped integration change, with this spec committed before implementation.

## Evidence and source

Job `5a78caa8-ee66-4ccf-b9e1-6572464af54e` retained the original shutdown policy.
Before and after the real `killpg(group, 0)`, only tracker PID83808/start8526239 remained.
It was already a zombie with parent PID1 and no device descriptors.
The EngineCore had disappeared by the shutdown reply.
The complete, non-atomic observations classify this instrumented run only.
They do not retroactively classify the older post-cleanup snapshots.
Evidence: campaign `teardown-observation-3053.qBJJHR/shutdown-timeline.json`.
SHA256: `5e6103e4fbc2e61cead0b04ba762615bd0dce4b60696745664b8b4a56610efc1`.

At the base, `tools/bench/strix_four_engine/qualify.py:142-144` starts a new session.
Lines 196-214 wait for the parent, check group absence, and clean up the group.
Lines 285-343 run adapters sequentially, closing one before the next.
`python_adapter.py:129-131,154-158` stops the engine and loop, acknowledges, and exits.
Do not change those inference-engine operations for this repair.

The exact recovered interpreter is Python 3.12.3 with base prefix `/usr`.
Its provenance and standard-library sources reside in `venv-lifecycle-sources.RdbAqT`.
`resource_tracker.py:73-88` closes the tracker pipe and waits in private `_stop()`.
`util.py:323-365` runs multiprocessing finalizers and joins children at exit.
Later exit callbacks can still use resources; an early private stop has unproven ordering.
Reject that alternative instead of inserting a blocking private call blindly.

Linux man-pages 6.18 supplies the operating-system contract:
[PR_SET_CHILD_SUBREAPER](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)
assigns orphaned descendants to their nearest ancestor subreaper.
[waitpid](https://man7.org/linux/man-pages/man2/waitpid.2.html)
supports nonblocking waits restricted to an owned process group.
This is controller lifecycle behavior, not an alternative inference oracle.

## Design

Make the controller a scoped subreaper before it creates an Adapter subprocess.
Use a new small lifecycle module reached by the existing Adapter constructor and close path.
The owning controller must be Linux, on its main thread, with exclusive Adapter lifecycle ownership.
Refuse nested or concurrent owners and unsupported or competing reaping arrangements.
Check SIGCHLD handling and auto-reap flags; do not silently replace a signal handler.
Read and retain the original subreaper state, check every system-call result, and verify the new state.
No child may launch after a failed acquisition.
The process-wide setting requires explicit ownership; a Python lock alone does not establish absence of foreign reapers.
Support the existing sequential campaign CLI. Refuse an imported-controller arrangement whose exclusivity cannot be established.

Keep Popen responsible for waiting for its exact direct child and retaining that exit status.
Only after that parent has been waited upon, drain adopted children using `waitpid(-owned_pgid, WNOHANG)`.
Never use unrestricted `waitpid(-1, ...)` and never consume the direct parent's status in this drain.
Record each returned PID and status. Nonzero exit or signal death fails normal shutdown.
A zero return means a child is alive and cannot become successful teardown through a grace period.
ECHILD does not prove absence: retain the existing `killpg(owned_pgid, 0)` absence check.
Successful shutdown requires normal parent exit, protocol completion, normal adopted-child statuses,
and actual disappearance of the process group.
Do not ignore zombies; success follows actual reaping.

Bound normal draining by a monotonic one-second deadline and 4096 returned children.
No blocking wait and no sleep belongs to normal acceptance.
On any error, retain the original error, run the existing owned-group cleanup,
wait for the direct parent, and perform bounded adopted-child cleanup before releasing ownership.
Error cleanup has a five-second total bound and cannot turn the original failure into success.
Surface cleanup exhaustion, abnormal statuses, and restoration errors as additional evidence.
Restoring the old subreaper flag does not undo adoption and cannot prove cleanup.
Restore and verify the prior flag on successful and failed construction, close, and cleanup paths.
Release the local lifecycle lock only after restoration has been attempted and its result recorded.
Repeated close must be safe without reacquisition or touching another lifecycle.
Preserve unrelated process groups and engine/model/environment files.

Retain lifecycle evidence through the controller's result/error path, with bounded diagnostic output.
No private Python tracker API, new external supervisor executable, or engine topology change is required.
Do not modify the c32 workload, profiler, precision, sampling, graphs, or token acceptance here.

## Tests and review

First reproduce the failure through the real Adapter and Python adapter CLI with public engine doubles
and a real multiprocessing resource tracker. Use deterministic process handshakes, not host PID1 timing.
A fixture supervisor may retain orphaned children to make the original failure deterministic.
Record red before production changes. Restore and reap every fixture-owned process.

Required independent cases include:

- Owned tracker and zero-exit adopted zombie: pass only after actual reaping and group disappearance.
- No tracker, repeated close, constructor failure, engine shutdown error, and protocol timeout.
- Live adopted child, nonzero child exit, signal death, and ECHILD with a still-existing group: fail.
- Foreign-group child: remain unreaped by this lifecycle.
- Nested/concurrent ownership, unsupported platform, prctl failure, SIGCHLD handler or auto-reaping: refuse.
- Normal-drain bound, cleanup deadline, state restoration failure, and original-error preservation.
- Direct-parent exit status remains Popen-owned; child drain cannot steal it.

The fresh reviewer deletes acquisition, production reaping, and the original group-absence call separately.
Mutate group selection, nonblocking mode, exit-status checks, limits, exclusivity, and restoration guarantees.
Every claimed guarantee needs an intended failing regression and exact scratch restoration.
Focused command: `python3 -m unittest discover -s tests/tools -p 'test_strix*adopt*.py' -v`.
Also run the existing qualification and Python-adapter suites.
The implementer may use a clearer focused filename but must record the exact command before review.
Full gate: `scripts/agent-preflight.sh`; retain each omitted artifact gate explicitly.
The operator repeats verification on the immutable reviewed head.

## Hardware acceptance and stop conditions

Run the reviewed controller against the unchanged recovered vLLM worker in a Strix lease.
Require Qwen3-4B c4 warmup, normal shutdown, recorded child statuses, and absent owned process group.
Require byte-identical model and engine bindings before and after the run.
Then verify the same lifecycle with patched SGLang; native adapters must not regress.
This does not establish token parity, profiler attachment, or accepted throughput.
Stop for unsupported signal/reaper ownership, escaped unaccounted descendants,
unbounded cleanup, changes to engine numerics, or a necessary policy relaxation.
Keep failures visible rather than treating restored controller state as successful teardown.

## Owed

#3053 retains complete four-engine correctness and benchmark qualification.
#3107 owns the c1/c4/c32 workload extension; #3076 owns measured optimization and traces.

## Implementation evidence (#3108)

The production qualification CLI enters `controller_lifecycle()` before `execute()`.
Controlled imported drivers use the same public context:

```python
from tools.bench.strix_four_engine.child_lifecycle import controller_lifecycle
from tools.bench.strix_four_engine.qualify import Adapter

with controller_lifecycle():
    adapter = Adapter(record, output, limit, timeout)
    try:
        adapter.exchange(command)
    finally:
        adapter.close()
```

The caller declares exclusive process lifecycle ownership, not merely exclusive Adapter use.
The checks reject existing children, other OS threads, inherited ownership after fork,
native/Python SIGCHLD handlers, auto-reaping, and an unowned subreaper state.
The caller must not introduce competing reapers or threads while the context is active.
Remaining controller children after owned-group disappearance cause refusal and poison ownership.
Bounded PID, PGID, and start-time snapshots describe unresolved ownership, not historical ancestry.
Missing, unreadable, oversized, malformed, or drifting inventories also refuse acceptance.
No foreign group is signaled or reaped, including an adopted descendant that calls `setsid()`.

The controller ABI is deliberately limited to measured Linux x86-64 glibc 2.39.
Strix job `63e6b768` confirmed that version; unsupported versions fail before launch.
The source binding follows glibc tag `glibc-2.39`,
`sysdeps/unix/sysv/linux/bits/sigaction.h`, `bits/types/__sigset_t.h`,
and `sysdeps/unix/sysv/linux/libc_sigaction.c`.
A compiled public-header fixture verifies size 152 and field offsets 0, 8, 136, and 144.
Only the kernel's 64 initialized signal-mask bits are compared; glibc's unused storage is not state.
No engine dependency, private multiprocessing API, or GPU operation changes.

The first real Adapter/Python CLI fixture failed with `adapter descendants survived shutdown`.
It creates a real resource tracker through the public multiprocessing semaphore API.
An outer fixture supervisor retains ownership on the unpatched path and reaps its own leftovers.
Additional red cases covered premature zero-exit protocol bypass, cleanup error preservation,
abandoned transport ownership, and escaped/foreign-child acceptance.
The focused command is `python3 -m unittest tests.tools.test_strix_adopted_children -v`.
Existing coverage runs with
`python3 -m unittest tests.tools.test_strix_four_engine_qualify tests.tools.test_strix_python_adapters`.
The standard full gate discovers the new suite through its existing `tests/tools/test_*.py` entry.
Logs retain intended red, restored green, independent scratch mutation results, and exact restoration hashes.
Full-gate and fresh-review outcomes belong to the immutable implementation handoff, not an inferred result here.

This change does not establish throughput, token parity, or profiler attachment.
Actual recovered vLLM and patched SGLang teardown still require the operator's Strix lease.

## Boundary coverage repair (#3108)

Independent review found five missing regression guarantees at `c820b05491dd7649c49bc08a37386b8dc3694ec8`.
The existing 22 lifecycle tests accepted each corresponding mutant independently.
Separate boundary probes rejected each mutant before the test repair.
The production lifecycle implementation required no change.

Four added fixtures exercise the controller and Adapter entry points.
They reject a failed native SIGCHLD query before launch.
They independently reject a new foreign child and changed SIGCHLD state after controller acquisition, before Adapter launch.
They reject a changed child list when the first inventory is empty, independently of PID start-time checks.
The escaped-child fixture now inspects syscall attempts after cleanup returns.
Assertions inside syscall doubles are insufficient because cleanup catches those exceptions.
The fixture still prevents foreign group signals and foreign waits before issuing any real syscall.

Verification used Python 3.12.3 on Linux x86-64 with glibc 2.39, without a GPU.
`python3 -m unittest tests.tools.test_strix_adopted_children tests.tools.test_strix_four_engine_qualify tests.tools.test_strix_python_adapters -v`
passed all 37 tests with exit 0.
The repaired lifecycle suite passed 26 tests before and after the mutation sweep.
Each of the five independent mutations failed its intended repaired test with exit 1.
The foreign-group mutation adds only a signal-0 probe, never a destructive foreign signal.
Scratch restoration matched the original lifecycle SHA256 after every mutation:
`e91c9f49a8c209dda37b8d543b6761bf2a0d6952065b19450d10d767c13eddb6`.

Evidence resides under campaign `verification-clean-3076.nWvRSX` on the resolved NAS.
`coverage3108-before-mutations.json` and `coverage3108-after-mutations.json` retain exact mutations, commands, exits, and log paths.
`coverage3108-combined.log` retains the combined suite output.
The runner is `/home/mudler/.cache/strix3108-coverage-mutations.py`, invoked with `before` and `after`.
The immutable-head full gate, fresh review, and operator verification remain required at this evidence checkpoint.
Artifact-dependent omissions must remain explicit in the full-gate handoff.
