# Publish lifecycle diagnostic results once

Row: `BACKEND-GATE-ROCM-SGLANG`.
Issue: [#3110](https://github.com/mudler/vllm.cpp/issues/3110).
Parent: [owned child reaping](strix-adapter-child-reaping.md), #3108;
[matched comparison](strix-four-engine-qwen3-4b.md), #3053.
Base: `4954b465fa20917016966bf73d001ceead7216c1`.

## Now

This is a spec-only checkpoint. Implementation, fresh review, and operator
validation remain pending. Commit this spec before implementation. Use one pull
request for this issue under the campaign's recorded integration shape.
This checkpoint changes no lifecycle state or acceptance requirement.

## Evidence and source

Strix job `e6dba009-a81b-4e0f-a74e-b6f39cb3b43d` exited 1.
Its vLLM engine record reports PASS: parent exit 0, adopted tracker PID85906
exit 0, group absent, state restored, and unchanged bindings. Its aggregate
reports FAIL. Those are separate results; the engine result does not repair
the failed aggregate or establish parity or throughput acceptance.

Evidence directory: campaign `strix-four-engine-3053.X94a3J`,
`child3108-validation.d6iTss/run` on the resolved NAS.
The retained `result.json` SHA256 is
`c4288940ee39899aff8b22f61bfe7c2ed5dc5768a05736d6ea4cb0cf4e43acd7`.
The scratch driver is `verification-clean-3076.nWvRSX/child3108/validate-hardware-194a.py`.
Its SHA256 is `ca12e50a70039242f71a4ac82dd8803d2ea9eb360e75af788503643c0383520a`.
These are evidence locators, not defaults for another developer's environment.

Driver line 115 publishes the finalized engine record. Line 116 publishes an
unfinished aggregate inside the engine's `finally` block. Line 121 publishes
the aggregate again at terminal completion. The second aggregate publication
raises `FileExistsError`; a multi-engine run can fail even earlier on its next
engine's aggregate publication. Do not overwrite or reinterpret the old result.

At the base, `tools/bench/strix_four_engine/audit.py:208-242` implements
`publish_result`: exclusive pending file, flush and fsync, then Linux
`renameat2(RENAME_NOREPLACE)`. A collision fails without replacing the target.
The publisher is correct and remains unchanged.
The scratch driver uses the public `qualify.Adapter` and
`child_lifecycle.controller_lifecycle` paths. Preserve their ownership contract
from the parent spec; do not introduce another reaper or engine shutdown path.
This is orchestration and evidence publication, not inference behavior; no
upstream numerical behavior or oracle pin changes.

## Scope and design

Track a reusable diagnostic command under `tools/bench/strix_four_engine/`,
with a focused public workflow entry in `docs/USAGE.md` in the implementation
change. Keep artifact verification separate from CPU-testable orchestration,
but require the real command entry point to reach both. Do not land a test-only
helper or leave the repaired command only in campaign scratch.

Preserve the scratch driver's fail-closed source and workload bindings:

- Require the source archive, expected archive SHA256, expected source revision,
  manifest, expected manifest SHA256, and fresh output directory explicitly.
  Trusted invocation values replace historical constants only as explicit
  inputs. Never derive an expected value from the artifact being checked.
- Verify archive and manifest hashes before launching an adapter. Check archive
  revision metadata and imported module origins against the verified source.
  Retain safe archive extraction and record driver and source provenance.
- Retain explicit Strix lease identity checks before hardware execution.
  Missing authority or binding is refusal, not a guessed value.
- Validate model and engine bindings before each engine and after teardown;
  require unchanged manifest bytes and binding observations. Preserve the
  canonical six prompts, resolved configuration, c4 warmup, and structural
  `validate_run` check. Do not extend the workload in this issue.

Run selected engines sequentially. Use the existing public lifecycle context
and Adapter constructor, exchanges, and close. Finalize an engine record only
after teardown, direct-parent status, adopted-child evidence, actual owned-group
absence, restoration, and post-run bindings have been checked. Retain original
execution errors together with teardown, binding, and publication errors.
Constructor failures and partially initialized runs must also produce failure
evidence when the output destination remains usable.

Publish each finalized engine record once at `<output>/<engine>/result.json`.
Use the existing immutable publisher without weakening its collision behavior.
Retain each finalized record in memory for the terminal aggregate; its entry
must equal the corresponding published engine record. Never mutate a finalized
entry afterward.

Publish `<output>/result.json` exactly once, after the engine loop terminates.
There is no interim publication to this final path. Aggregate PASS requires all
selected engines to have completed and passed, with no publication error.
Otherwise publish FAIL and return nonzero. A failed engine or publication must
stop the loop before another engine launches. Successful single- and
multi-engine runs return 0 only after final publication succeeds.

If an engine publication fails, retain already published records and include
the failure in the terminal aggregate when that publication is possible. If
the final publication itself fails, return nonzero and retain the error on
stderr; never replace an existing aggregate or manufacture a successful one.
Retries require a fresh output directory. Do not delete previous evidence or
add overwrite, hard-link, or mutable-file fallbacks.

## Tests and red trigger

Add `tests/tools/test_strix_lifecycle_diagnostic.py`. The proposed focused command
is `python3 -m unittest tests.tools.test_strix_lifecycle_diagnostic -v`.
Use the command's real `main` or public orchestration entry, with expensive
engine operations mocked and the real `publish_result` on a temporary Linux
filesystem. No GPU or installed inference engine is needed for this regression.

First reproduce the scratch control flow with one successful fake engine.
Expect exit 0, one finalized engine record, and one PASS aggregate. Record the
intended red: duplicate aggregate publication raises EEXIST after leaving the
initial FAIL aggregate. Then make the minimum complete orchestration repair.

Required independent cases include:

- Single- and multi-engine success: one publication attempt per engine and one
  final aggregate attempt, ordered after all finalized engine records.
- Configuration or warmup failure, constructor failure, and teardown failure:
  preserve errors, finalize FAIL, and never launch the next engine.
- Nonzero parent or adopted-child status, unresolved children, cleanup errors,
  missing group absence, or failed restoration: none may become PASS.
- Archive, revision, import-origin, lease, manifest, and pre/post binding
  refusals: no launch after a failed prerequisite; no PASS after changed bindings.
- Real target collision and injected engine/final publication failures: nonzero
  result, no overwrite, no later engine, and retained available evidence.
- Aggregate entries equal the finalized engine files; command configuration
  reaches canonical prompts, resolved settings, and structural validation.

Spies may wrap the real publisher. Assert observed attempts and ordering after
the operation returns or raises. Assertions raised only inside mocks can be
caught by cleanup and do not prove the guarantee. Assert both failure status
and the intended reason; unrelated exceptions are not a regression proof.

Fresh review mutates the duplicate aggregate write, final aggregate call,
stop-after-failure branch, binding checks, teardown/status guards, and public
command call site separately. Each claimed guarantee must reject its intended
mutation. Restore scratch bytes after each mutation; never mutate the reviewed
worktree or use destructive foreign-process probes.

## Gates and handoff

Run the focused suite and existing `test_strix_adopted_children`,
`test_strix_four_engine_qualify`, and `test_strix_python_adapters` suites.
Run `scripts/agent-preflight.sh` before edits and the complete
`scripts/agent-preflight.sh --staged` before each commit. Record terminal exits,
exact revisions, commands, logs, and every artifact-dependent omission. Run
explicit range classification, commit-style, and trailer checks for the handoff.
A fresh reviewer checks the immutable implementation head and mutation evidence;
the operator repeats the applicable gate itself.

The operator then runs the reviewed driver in a Strix lease, using fresh output
and explicitly bound unchanged artifacts. Require engine records, one terminal
aggregate, correct process exit, and all lifecycle/binding checks. Preserve the
original failed job alongside the new evidence. The spec author and CPU fixture
implementer have no GPU authority from this issue alone.

## Risks and stop conditions

Separating artifact bootstrap from orchestration can bypass origin checks;
tests must enter the same public command and prove those checks remain reachable.
Early serialization can freeze an incomplete engine snapshot; compare published
records against the aggregate only after teardown. Error handling must not let
a publication exception hide the original execution failure.

Stop for missing binding authority, unavailable source evidence, a proposed
immutable-publisher relaxation, or a necessary production lifecycle or adapter
change. Return NEEDS_CONTEXT for missing context and NEEDS_DECISION for a scope
conflict. Do not solve either by guessing or widening this repair.

## Owed

#3110 owns implementation, independent review, and operator diagnostic rerun.
#3108 retains the remaining actual multi-engine lifecycle validation.
#3053 owns full matched correctness and benchmark qualification; #3077 owns
diagnostic numerical scoping. Strict token-exact acceptance remains binding.
No parity, throughput, profiling, tolerance, engine, or pin change belongs here.

## Implementation evidence

The public command is
`tools/bench/strix_four_engine/lifecycle_diagnostic.py`. Its `main` enters the
verified archive bootstrap, then `execute` and `run_engine`. The existing
publisher, lifecycle context, adapters, and validators remain unchanged.

The operator clarified the existing selection boundary before handoff: the
first engine must be vLLM or patched SGLang. Those adapters establish canonical
IDs. `native_adapter.cpp:49` requires an array before either native model loads.
The command refuses native-first selection before launch; it does not invent
IDs or launch an unselected engine. #3111 owns a separate bound-reference input.
Tests cover both Python-first orders and enforce the native configure contract
in the expensive-engine double.

The initial regression executed the original scratch driver with fake engine
work and the real immutable publisher. It failed with `FileExistsError(17)` at
the second aggregate publication. A separate public-entry test failed because
the new command did not yet exist. Logs are retained under
`/home/mudler/.cache/strix3110-red-legacy-focused.log` and
`/home/mudler/.cache/strix3110-red-public-entry.log`. The native-first refusal
and missing adopted-child metadata each have separate red observations.

The bound CPU fixture imports the actual qualification validators, lifecycle
context, and publisher from an explicitly hashed archive. Only expensive engine
operations are replaced. It records configure, run, close, validation, and
publication calls. Real result collisions preserve existing bytes. It does not
prove real GPU execution, process teardown, token parity, or throughput.

The combined diagnostic, adopted-child, qualification, and Python-adapter suites
passed 51 tests in 52.166 seconds before the final selection and path-identity
cases. The final gate evidence accompanies the immutable implementation
handoff. The pre-edit full gate exited 0; its five argument-dependent omissions
remain explicit in `/home/mudler/.cache/strix3110-startup-full.log`.

Scratch mutations cover publication order and call sites, stop conditions,
binding checks, canonical inputs, lifecycle observations, and provenance.
Each completed mutation restored bytes with `cmp`. The full witness inventory
is `/home/mudler/.cache/strix3110-mutations.json`; individual logs use the
`strix3110-mutant-` prefix. Two aggregate predicates initially survived because
multi-engine failures also triggered the incomplete-count check. Single-engine
cases now isolate both predicates and fail their mutations. Prompt-count and
ID-mismatch mutations still encounter structural refusal downstream; the tests
also pin the earlier diagnostic. These are not independent runtime proofs.

Independent review and the operator's leased diagnostic remain pending. The
original failed job and its partial records retain their original disposition.
