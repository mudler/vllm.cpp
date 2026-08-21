# Task guide — coordinating parallel work

How a coordinator runs several rows at once without agents colliding. The rules
are in [`AGENTS.md`](../AGENTS.md); this is the method.

## The shape of a campaign

The operator is a coordinator: it holds the plan and the GPU, merges reviewed
PRs, and delegates everything else to fresh agents with bounded briefs in their
own worktrees. It does not implement work that should be independently reviewed
— writing it and reviewing it in one context defeats the review.

**Several coordinators may run at once.** `scripts/agent-role.py claim operator`
records this worktree and never refuses; `scripts/agent-role.py show` lists the
other live coordinators — worktree, session, host, and time since their last
heartbeat — and prunes anything past the 2-hour TTL. What keeps concurrent
coordinators from colliding is not that file: `main` is never force-pushed, so a
plain `git push` refuses any non-fast-forward. When yours is rejected, fetch,
re-merge, re-run the gate, and push again.

For each row: confirm the issue, commit the spec, dispatch an implementer,
dispatch a *different* reviewer, return findings to a new implementer, rerun the
gate yourself, then merge or close. A dispatched helper needs a spec that is
reachable from its base before dispatch. This mechanical requirement does not
select the pull request shape for work that does not use helper dispatch.

Write the reachability mutation into the reviewer's `Required evidence` when you
dispatch. The versioned contract in [`prompts/reviewer.md`](prompts/reviewer.md)
is a closed grammar that `check-prompt-contract.py` pins, so it carries no rule
row for this, and the envelope is where the obligation reaches the reviewer. An
envelope that omits it gets a review that cannot report an unreached change
([`reachability.md`](reachability.md)).

## Isolation

Parallel agents get their own worktree. Never let two agents write the same
tree.

Stage explicit paths. `git add -A` mid-write picks up another agent's
half-finished work, and the resulting commit is very hard to unpick afterwards.

Pin the base SHA when you create the worktree and push with a lease. A landing
built on `origin/main` resolved at push time can silently absorb — and revert —
another session's merge.

Brief every dispatched agent to wait in the foreground and return once.
Backgrounded builds surface as a completed agent with no result, which reads
exactly like success.

**A merged pull request is necessary before you remove a worktree. It is not
sufficient.** Check both of these first, and check them in this order:

1. `git log --oneline @{u}..HEAD` is EMPTY, so nothing local is unpushed.
2. The content reached `main`, checked BY CONTENT. Use `git diff <commit>
   origin/main -- <the paths it touched>`, or `git log -S'<a phrase it added>'
   origin/main`.

**Ancestry answers neither question.** `main` is squash-only, so no commit from
a branch is ever an ancestor of `main` and no patch-id survives the squash.
`git merge-base --is-ancestor` returns false for work that landed perfectly, and
`git cherry origin/main` marks landed commits `+`. Measured on the PR #1035
worktree, which was reaped carrying two commits on neither `main` nor its remote
branch: those two instruments both said the work was missing, step 1 said it was
unpushed, and step 2 said it had landed. Step 2 was right, and only running both
steps tells you which case you are in
([#1130](https://github.com/mudler/vllm.cpp/issues/1130)).

Rescue an uncertain branch to a `rescue/<name>` ref rather than deleting it.
Three exist and none is adjudicated: `rescue/es-cuda-grouped-unpushed`,
`rescue/cuda-breadth-sm75-audit` and `rescue/fp8-native`.

## Claims

`.agents/coordination.md` holds who is doing what *now*. It is overwritten, not
appended. The canonical hierarchy it defines is one-way: the roadmap orders big
areas, area matrices hold the exhaustive row inventory, specs hold the analysis,
this file holds the live claim, and the matrices plus the parity ledger hold the
evidence. Detailed status prose never flows upward.

If the remote cannot be queried, the claim state is `REMOTE_UNVERIFIED`. Unknown
is not absence, and it authorizes no cleanup.

## Handing off unfinished work

Refresh `.agents/NOW.md` — live claims, current gate, next actions, stamp — and
update the coordination entry. Put the durable findings in the row's spec
`## Outcome`.

Record the immutable head, the exact gates run, the evidence roots, any
prohibitions, the blocker, and the first resume command. The next session should
need `NOW.md`, the row's spec, and `git log` — nothing else.

Routine discussion and git housekeeping are not checkpoints and do not need a
record.

## Landing

Land squashes or a local `--no-ff` merge commit carrying the trailers. GitHub
authors its own message for `gh pr merge --merge`, which loses them.

A squash lands the pull request BODY, so read it before you merge with
`python3 scripts/agent-pr-body.py --pr <N>`. The CI job that reads it sits in a
queue and has been outrun by a merge (#1263).

Verified PRs merge in-session. Obsolete PRs close with the reason recorded.
Never end a session with a verified, unmerged PR.
