# ORACLE-PIN-SOURCE-BUILD: test independent version comparisons

Issue: [#3078](https://github.com/mudler/vllm.cpp/issues/3078)
Row: `ORACLE-PIN-SOURCE-BUILD`
Base: `aa3b531a846c78083b99f5238197fec9d8863a08`
Prior spec: `0afa3f744`, revised at `56f44f7c2`.
Historical design and evidence: [original proposal](../completed/oracle-pin-source-build-proposal.md).

## Scope

Main already fixes the record test in `d45887e12`. Issue #2931 stays closed.
Preserve `test_metadata_and_runtime_are_compared_as_two_independent_fields`
byte-for-byte. Retain the two behavioral tests from PR #2941.
Correct the five associated comments and the stale commit example in #2949.
Exclude inherited benchmark and placement changes, oracle pins, and runtime behavior.

## Design and upstream anchors

Each observed version must match its own recorded constant.
Patch the distribution constant to a suffixed value to separate the comparisons.
Call `record_oracle_manifest` through the existing test helper and require
the exact `oracle version drift` refusal.
This is a local harness rule. No upstream implementation or GPU run is required.
The historical proposal records source-build evidence and the pre-#520 defect.

## Risks

A test could pass because a later fixture fails. Match the refusal text.
Current equal constants could hide a wrong comparison. Patch them apart.
Preserve main's nonempty assertions and its prefix relation.
Comments describe the build modes without storing another current commit hash.

## Tests and gates

Run both added tests against three scratch mutations: metadata compared with
the runtime constant, runtime comparison deleted, and runtime compared with
the distribution constant. Each relevant test must fail for the intended reason.
Restore the production file byte-for-byte after each mutation.

Run `python3 -m unittest tests.tools.test_oracle_pin -v`,
`python3 -m unittest discover -s tests/tools -t . -p 'test_*.py'`,
`python3 scripts/check-oracle-pins.py`, its `--self-test`,
the issue record gate, commit style and size range checks, and full preflight.
Preserve declared skips. Record any environmental baseline failure separately.

## Evidence

Implementation records commands, results, mutation failures, and restoration hashes.
Fresh review and operator verification remain required before landing.

On 8 September 2026, the oracle module passed 27 tests and tools passed 801 tests.
The pin checker passed 14 oracles. Its self-test passed 49 fixtures.
Logs: `/tmp/pr2941-oracle-green.log` and `/tmp/pr2941-tools.log`.
Each of the three mutation runs exited 1 in
`/tmp/pr2941-mutation-{1,2,3}.log`. The matching new test failed because
the expected version refusal was absent. A later FlashInfer refusal did not
mask the missing guard. After each mutation, `git diff --exit-code --
tools/bench/online_gate.py` exited 0 in the scratch worktree.

## Outcome

The surviving change adds executable comparison coverage without changing a
version value or comparison. Main's record test remains byte-for-byte equal.
Replacing that test was rejected because main already preserves its assertions.
The tests use a derived suffix so they remain independent of future pin values.
The #2949 comment now refers to the recorded commit segment without duplicating
the current hash. Fresh review and final operator gates remain pending.

## Stop conditions

Return `NEEDS_DECISION` if a pin or runtime change becomes necessary.
Stop if the behavioral tests survive a claimed mutation.
Do not run GPU work or reopen #2931.

## Owed

- [#2949](https://github.com/mudler/vllm.cpp/issues/2949) is fixed in this
  change by removing the stale hash example. The landing PR closes it.
- A runnable precompiled aarch64 oracle remains unproved. The historical
  proposal preserves that evidence debt. This change makes no build claim.

## Now

`ORACLE-PIN-SOURCE-BUILD` is `ACTIVE` for #3078.
