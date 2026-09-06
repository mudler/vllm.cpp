# Reconcile the merged ROCm Q8_K records

Issue: [#2599](https://github.com/mudler/vllm.cpp/issues/2599)

Row: `BACKEND-ROCM`

## Now

`ACTIVE`: pull request
[#2472](https://github.com/mudler/vllm.cpp/pull/2472) merged as
`9f96b74465441ebbee3651f4b316cdb0bf183715`. The accepted `gfx1100` Q8_K
implementation is upstream, but two record projections still describe its
pre-merge state.

Issue [#1876](https://github.com/mudler/vllm.cpp/issues/1876) is `OPEN` after a
maintainer reopened it. Its body starts with the canonical
`Row: BACKEND-ROCM` ownership line. It owns `gfx1200` and `gfx1201` runtime and
default validation. Both architectures remain `PENDING`, and an unset
`VT_ROCM_Q8K_BLOCK` continues to select the legacy arm on them.

Issue [#2598](https://github.com/mudler/vllm.cpp/issues/2598) is `CLOSED` as a
duplicate of #1876. This specification precedes the record-only implementation
owned by #2599.

## Scope

The later implementation makes these changes:

- Reconcile the `## Now`, `## Issue ownership`, current-state outcome text, and
  `## Owed` section in
  [`rocm-q8k-cooperative-quantizer.md`](rocm-q8k-cooperative-quantizer.md).
- Retain the accepted `gfx1100` evidence and architecture-scoped default.
- Record merged #2472 and its merge SHA.
- Record the reopened, canonical row ownership of #1876.
- Record closed duplicate #2598 without moving its work away from #1876.
- Remove the hard-coded ROCm operation total from both ROCm cells in
  [`docs/FEATURES.md`](../../docs/FEATURES.md).
- Describe the native ROCm surface without a total and direct readers to the
  derived recount method in the [ROCm guide](../../docs/ROCM.md).
- Update this specification's `## Outcome` after the implementation passes its
  gates.

The implementation excludes these changes:

- Product code, tests, operation registrations, and backend behavior.
- The accepted Q8_K arithmetic, performance evidence, and `gfx1100` default.
- The legacy unset default on `gfx1200`, `gfx1201`, unknown architectures, and
  architecture-resolution failure.
- Matrix lifecycle state, issue records, `.agents/NOW.md`, and unrelated
  repository records.
- `docs/ROCM.md`, whose recount method is already correct.
- A replacement operation total in any tracked file.
- GitHub issue, pull request, or merge state.

## Observed stale state

The pinned base is `5649e07d2120df4c5d33fd1d245336490c790e2b`, with tree
`ca90dfb99f7665115580272090d36d1df3d64257`.

At that base, the Q8_K specification contains these stale current-state
passages:

- Lines 16 to 18 say the implementation is not upstream.
- Line 22 says #1876 lacks a canonical `Row:` line.
- Lines 27 to 29 describe a proposed pull request instead of merged #2472.
- Lines 686 to 688 again say the implementation is not upstream.

The accepted `gfx1100` outcome at lines 670 to 680 remains correct. The pending
`gfx1200` and `gfx1201` ownership at lines 704 to 712 also remains correct, but
it needs the reopened issue state and canonical ownership.

Two ROCm cells in `docs/FEATURES.md` store a hard-coded total:

- The backend table at line 302 uses `52 registered ops`.
- The unsupported-work table at line 404 uses `52 registered ops`.

The [ROCm guide](../../docs/ROCM.md#current-backend-surface) already says that
the operation table is derived. Lines 112 to 126 document a scan that reads
wrapped `RegisterOp` calls and restricts the match to `DeviceType::kROCM`.
Storing the scan result in `docs/FEATURES.md` would create a cross-file live
measurement and let the total drift again.

## Design

### Reconcile Q8_K state

Rewrite only present-state claims in the existing Q8_K specification. Keep the
accepted byte, reachability, performance, memory, and `gfx1100` default
evidence unchanged.

The corrected projection must state all of these facts together:

- #2472 is merged at
  `9f96b74465441ebbee3651f4b316cdb0bf183715`.
- #1876 is `OPEN` after its 2 September 2026 reopen event.
- The #1876 body has canonical `BACKEND-ROCM` row ownership.
- #1876 owns pending `gfx1200` and `gfx1201` runtime and default validation.
- #2598 is a closed duplicate and owns no separate validation arm.
- The accepted `gfx1100` result and cooperative unset default remain in force.
- Unset remains legacy on `gfx1200` and `gfx1201` until each architecture has
  its own accepted evidence.

Do not turn the accidental close and reopen into a technical state change. Do
not rewrite historical pre-landing evidence that remains factually correct.

### Remove stored operation totals

Delete the numeric operation total from each ROCm cell in `docs/FEATURES.md`.
Keep the qualitative list of native operation families and the existing
hardware evidence.

Each cell must tell readers to use `docs/ROCM.md` for the device-specific,
wrapped-call-safe recount. Do not copy the command output into this
specification, `docs/FEATURES.md`, or another tracked file.

## Risks

- A broad rewrite can alter accepted `gfx1100` evidence instead of correcting
  only its merge state.
- Closing the duplicate can accidentally move external validation away from
  the reopened owning issue.
- Removing a total can also remove useful qualitative ROCm capability text.
- A vague guide link can fail to tell readers where the derived recount lives.
- A replacement total would recreate the cross-file measurement defect.
- GitHub state can change between specification and implementation.

## Tests

This specification commit changes no product behavior. It has no product
failure to make red, so fabricating red-first evidence is prohibited. The
red-first requirement begins with the later documentation implementation.

For this specification commit:

1. Inspect the live states of #2599, #2598, #1876, and #2472.
2. Confirm both local links resolve and the issue and pull request links are
   readable.
3. Review the prose against `.agents/style/prose.md`.
4. Run `git diff --check` before staging and for the staged change.
5. Run `python3 scripts/check-agent-record.py`.
6. Run `scripts/agent-preflight.sh --staged` and report every skip by name and
   reason. Exit zero with a skip is not a green preflight.
7. Validate the committed message with the commit-style and trailer gates.
8. Prove the commit changes exactly this specification file.

For the later implementation:

1. Assert that both ROCm cells remain present in `docs/FEATURES.md`.
2. Assert that neither cell contains a numeric registered-operation total.
3. Assert that both cells refer to the derived recount in `docs/ROCM.md`.
4. Assert that the stale Q8_K phrases are absent.
5. Assert that the corrected Q8_K specification contains the merge SHA, all
   three issue dispositions, both architecture boundaries, and canonical row
   ownership.
6. Execute the recount command from `docs/ROCM.md` with `pipefail` and require
   exit zero. Suppress its output instead of recording a total.
7. Run `git diff --check`, `python3 scripts/check-agent-record.py`, and the full
   staged preflight.

## Negative mutation

Mutation does not apply to this specification-only commit because it contains
no implementation or test behavior.

The later reviewer must use a scratch copy and run these mutations separately:

1. Reintroduce a numeric registered-operation total into the first ROCm cell.
2. Reintroduce a numeric registered-operation total into the second ROCm cell.
3. Remove the derived-recount reference from either ROCm cell.
4. Restore one stale Q8_K present-state phrase.
5. Change the `gfx1200` or `gfx1201` unset policy from legacy to cooperative.

Each mutation must fail the matching focused check for the intended reason.
Restore the scratch tree byte-for-byte after each mutation.

## Gates

The specification gate is satisfied only when every applicable command exits
zero, all local links resolve, the staged preflight's skips are classified, and
the commit contains exactly one tracked file.

The implementation gate is satisfied only when both public cells contain no
stored total, both refer to the derived recount, and the Q8_K record contains
the complete corrected state. The implementation must not run a GPU gate
because it changes no product behavior.

An inherited or argument-dependent result is acceptable only when the exact
same result appears on the pinned base. Record the command, exit status, base
result, and changed-head result. An unproved difference is `FAILING`.

## Evidence

The specification uses these observations from 2 September 2026:

- `gh issue view 2599` returned `OPEN` with `Row: BACKEND-ROCM`.
- `gh pr view 2472` returned `MERGED` and merge commit
  `9f96b74465441ebbee3651f4b316cdb0bf183715`.
- `gh issue view 1876` returned `OPEN` with `stateReason=REOPENED` and the
  canonical row line.
- The #1876 timeline records `localai-org-maint-bot` reopening it at
  `2026-09-02T19:27:25Z`.
- The #1876 maintainer comments retain the `gfx1200` and `gfx1201` validation
  arm under `BACKEND-ROCM`.
- `gh issue view 2598` returned `CLOSED`. Its closure comment identifies it
  as a duplicate of reopened #1876.
- `git log -S` attributes both stale Q8_K phrases to `39398fece`.
- `git show --stat 9f96b7446` confirms that #2472 carried the Q8_K
  implementation, tests, specification, and user documentation.
- Direct inspection found the stale Q8_K passages, both public totals, and the
  derived recount method at the anchors in `## Observed stale state`.

The implementation evidence must record the immutable base, head, and tree.
It must also record the exact diff, command results, skips, and mutation
restorations. It must not record a new operation total.

## Git integration

Use one pull request for the specification and implementation. This is the
recorded repository default for `BACKEND-ROCM` issue #2599.

Commit this specification before the implementation. A fresh implementer then
makes the scoped record edits, and a fresh reviewer inspects and mutates the
immutable implementation head. The operator reruns the declared gates before
landing.

The specification-authoring task changes and commits only this file. It does
not push, open a pull request, merge, or edit GitHub state.

## Owed

- Issue #2599 owns the later record-only implementation and this
  specification's completed outcome.
- [Issue #1876](https://github.com/mudler/vllm.cpp/issues/1876) continues to own
  `gfx1200` and `gfx1201` runtime and default validation under
  `BACKEND-ROCM`.
- The unset policy stays legacy on both external architectures until their
  individual gates pass.
- Closed duplicate #2598 owns no work and requires no record update.

## Outcome

`COMPLETE`: issue #2599's record-only repair changes exactly these files:

- `.agents/specs/rocm-q8k-cooperative-quantizer.md`
- `.agents/specs/rocm-q8k-record-reconciliation.md`
- `docs/FEATURES.md`

The focused assertion recorded the required stale-state failure. It then
passed against both public cells and all corrected Q8_K current-state sections.
The wrapped-call-safe, device-specific recount exited 0 with its output
suppressed.

Six separate scratch mutations each failed for the intended reason. They
covered both numeric totals, both guide references, one stale state, and one
external-architecture default. Every restoration was byte-identical and
returned the focused assertion to exit 0.

The whitespace, local-link, scoped-prose, record, and documentation gates
exited 0. The controlled full preflight returned skip-aware exit 0. It is not
green because five NumPy-dependent suites and five argument-dependent tools
were skipped. The tree compile scope was a derived empty set, not a skip.

No derived operation total is stored. The accepted `gfx1100` result and default
remain unchanged, and #1876 stays open for `gfx1200` and `gfx1201` validation.

## Stop conditions

- Return `NEEDS_CONTEXT` if #2599, #2598, #1876, #2472, the merge SHA, or the
  recount policy cannot be read.
- Return `NEEDS_DECISION` if live state contradicts the corrected contract or
  if the work needs a new ownership decision.
- Return `BLOCKED` if a required edit falls outside the declared authority.
- Return `BLOCKED` if a declared verification cannot pass or match a proven
  unchanged baseline.
- Stop if the implementation changes product code, tests, matrix state,
  `.agents/NOW.md`, `docs/ROCM.md`, or an unrelated record.
- Stop if either public cell retains or gains a numeric operation total.
- Stop if the Q8_K reconciliation weakens accepted `gfx1100` evidence or widens
  the unset default to `gfx1200` or `gfx1201`.
