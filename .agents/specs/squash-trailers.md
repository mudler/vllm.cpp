# The pull request body became the commit message, so it is gated like one

Issues: [#848](https://github.com/mudler/vllm.cpp/issues/848),
[#829](https://github.com/mudler/vllm.cpp/issues/829)
Row: `GATE-SQUASH-TRAILERS`

A repository setting moved where a landed commit message comes from. The
obligation moved with it and the enforcement did not. This row moves the
enforcement.

## Why

### What #829 actually was

The reported symptom was that a squash repeats the trailer block once per
commit. The repetition is real and it is harmless, because
`git interpret-trailers --parse` reads only the trailing block.

The parse breaks for a different reason. GitHub writes a `---------` separator
between concatenated commit messages, and on a multi-commit squash the last one
lands between the protocol trailer block and the `Co-authored-by:` line GitHub
appends. `join_trailing_trailer_paragraphs` fuses trailer-shaped paragraphs at
the end and deliberately stops at anything else, which is correct: a prose
paragraph must still terminate the block. The separator is not trailer-shaped,
so the block is orphaned and the gate reports trailers the commit plainly
carries as missing.

That is why it only ever happened to multi-commit pull requests. A single-commit
squash has no separator.

### Why the fix was a setting and not a checker

Measured against the unmodified checker:

| Simulated squash body | Result |
|---|---|
| `PR_BODY` shape, body plus `Co-authored-by:` | green |
| `COMMIT_MESSAGES` shape, two bodies plus `---------` | red |

Teaching the checker to swallow the second shape needs two widenings at once.
It has to step over a forge-generated separator, and it has to admit repeated
blocks. The result would accept a genuinely malformed message. Removing the
cause costs one setting, so the setting changed to `PR_BODY` on 2026-08-14 and
the checker stays strict.

### What the setting broke

The pull request body is now the commit message.
`.github/pull_request_template.md` carries no trailer block, so a pull request
opened from the template and merged lands a commit with no trailers at all.

The failure moved from "wrong shape, caught before merge" to "no trailers,
caught after merge on `main`". That is strictly worse, and #822 keeps it
invisible: recent `main` push runs are `cancelled` by the concurrency group, so
the red never surfaces.

`check-commit-trailers.py` accepts `--range` only. It reads committed objects,
and it cannot read the body that is about to become one.

## Design

**The template carries the block.** `.github/pull_request_template.md` ends with
the bare `FOLLOWING_AGENTS_PROTOCOL` paragraph and the three trailers, so the
default body already satisfies the contract.

**The checker reads a message, not only a range.** Add `--message-file PATH`,
where `-` is standard input. It calls `validate_commit_message(..., strict=True)`
unchanged. The pull request body is then held to exactly the rule its landed
commit will be held to, by the same code, which is the only way the two cannot
drift.

**`AGENTS.md` records that the setting is part of the contract.** No gate can
hold it: reading a repository setting needs a network call, and no checker here
may make one. The note sits in the landing section and names the failure to look
for, so a silent revert is diagnosed by reading rather than rediscovered.

**CI validates the body on the pull request lane.** The existing trailer step
gains a branch that writes `github.event.pull_request.body` to a file and runs
the new mode. It is skipped on the push lane, where there is no body.

**Both squash shapes are pinned.** A regression test asserts the `PR_BODY` shape
is green and the `---------` shape is red, so the mechanism above is executable
rather than remembered. A second case asserts the shipped template passes the
checker, which is what stops the template and the rule from drifting apart.

## Scope

In scope: `AGENTS.md` (the landing note), `.github/pull_request_template.md`, `scripts/check-commit-trailers.py`
(one new argument, no rule change), `.github/workflows/ci.yml` (one step),
`tests/scripts/test_check_commit_trailers.py`, and the issue index rows.

Out of scope and stated as owed:

- #822, the cancelled `main` push runs. It is why both this and #829 stayed
  invisible, and it is not this row's fix.
- Any change to the trailer rules themselves. The rule is unchanged. Only the
  surfaces it is applied to change.

## Gates

| Gate | Red before | Green after |
|---|---|---|
| Template lacks the block | Strip it, the template case fires | Case names the template |
| Body lacks the block | Feed a bodyless message, the mode exits non-zero | Mode reports the missing trailers |
| `PR_BODY` shape | Assert green | Green |
| `---------` shape | Assert red | Red, naming the marker rule |
| `--message-file -` | Feed on standard input | Same verdict as a file |

Full gate: `scripts/agent-preflight.sh --staged`, then
`python3 scripts/agent-integration.py --base origin/main`.

## Risks

**The body is editable after the pull request opens.** CI validates the body at
the moment it runs, and a later edit is not re-validated unless the event fires
again. The landed commit is still walked by the push-lane range check, so the
error surfaces, only later. Closing that gap properly needs #822 first.

**A body that satisfies the checker can still be a bad description.** This gates
the trailer contract, never the prose.

## Stop conditions

Stop and return `NEEDS_DECISION` if validating the body requires relaxing any
trailer rule. The rule is not in scope, and a relaxation here would be the
second widening this row exists to avoid.

## Owed

- [#849](https://github.com/mudler/vllm.cpp/issues/849). An issue that is never
  appended to the index is invisible to the ownership gate this row's
  predecessor built. #829, #848 and #822 were all open with no row, which is how
  this was found. Closing it needs a GitHub query, and no gate here may use the
  network, so the candidate shapes are an operator-cadence report or a
  branch-scoped check of the issue numbers a commit cites.

## Now

Implementing. The repository setting is already `PR_BODY`.
