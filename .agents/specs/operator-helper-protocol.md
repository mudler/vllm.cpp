# Operator / helper-agent protocol

User-directed 2026-08-04. Status: **accepted design, not yet enforced.** This
document is the contract; the CI guards named in § Enforcement are the work it
implies. `AGENTS.md` is deliberately untouched until this is reviewed.

## Scope

How multiple agent sessions work on this repository at the same time without
corrupting the record or each other. Defines two roles, how a session learns
which one it is, what each may do, how work is reserved, and what the operator
merges on.

Out of scope: what work to do (that is the roadmap), and how to do it (that is
`AGENTS.md` and `.agents/directives.md`, which both roles follow in full).

## Our baseline — why this exists

Concurrent sessions are already happening and already cost us. On 2026-08-04 two
sessions pushed to `main` within minutes:

- both wrote `docs/STATUS.md`, `docs/BENCHMARKS.md` and `.agents/NOW.md`;
- neither claimed anything in `coordination.md`, the surface that exists to
  prevent exactly this;
- the second push was rejected twice and rebased twice onto a moving `main`;
- **a three-way merge silently produced a VARIANT of the other session's binding
  performance numbers.** Not a conflict, no marker, just wrong text. It was
  caught only by diffing against `origin/main` and hashing the individual lines.

Separately, the same day's audit found `coordination.md` carrying 106 rows
claiming "implementation in flight" with nobody flying them, because claiming
was free and un-expiring while releasing cost anchor work.

Both failures are structural, not carelessness. The protocol below is built so
that neither can recur by construction rather than by discipline.

## Roles

**Operator.** One at a time. Owns integration: reviews and merges PRs, keeps the
record coherent, runs the GPU. The only role that lands anything on `main`.

**Helper.** Any number, bounded by disk (see § Dependencies). Picks one ready
row, works in an isolated worktree, opens a PR. Never touches `main`.

### Determining the role — DECLARE, then MATERIALIZE, then derive

Derivation alone does NOT work, and assuming it did was an error in the first
draft of this spec (user-caught, 2026-08-04). The common case is several
sessions started from the SAME environment: same machine, same primary checkout,
same branch. Nothing distinguishes them. A helper only becomes environmentally
recognisable AFTER it has taken a worktree and a branch, so derivation is
circular if used to decide the role in the first place.

Acquisition and persistence are different problems and need different mechanisms:

**1. Acquire — ASK.** Unless the user has said which role this session is, the
session ASKS before doing anything else. This is unavoidable: the information
does not exist in the environment yet. It is also cheap, once per session.

**2. Materialize — make the answer a FACT.** Immediately on answering:

- *operator*: atomically acquire `.agents/operator.lock` (create-exclusive, so a
  second self-declared operator FAILS rather than racing), carrying session id,
  host, PID and a heartbeat timestamp;
- *helper*: create its worktree and `row/<ROW-ID>` branch and open the draft PR
  straight away, before any other work.

After this step the session IS distinguishable from the outside, which is what
the first draft wrongly assumed at the start.

**3. Persist and re-derive — never ask twice.** The role is recorded in a
session-scoped marker (keyed by session id, outside the repo). Re-derivation, not
memory, is what survives context compaction: a session that has forgotten its
role reads the marker and the lock rather than guessing or asking again.

**4. Resolution rides on the mandatory preflight.** `scripts/agent-preflight.sh`
resolves and PRINTS the role every run, and refuses to pass if a session has not
declared one. That is what stops the role from being a prose convention: the
check is embedded in the tool both roles must already run at session start and
before every commit.

The lock is also the mutual-exclusion mechanism, not just a label. If it is
already held, this session CANNOT be the operator whatever it was told, and the
correct response is to say so and offer the helper role instead. The lock needs a
TTL and heartbeat so a crashed operator does not block everyone (§ Risks).

Multiple helpers from one environment are fine and collision-free: each takes a
distinct worktree named for its row, and the row is already exclusive because the
PR is the claim.

## Operator rules

1. **Merge pass first.** The first action of an operator session is to review
   open PRs and merge what is ready. Integration debt is what kills parallel
   setups; doing this last means never doing it.
2. **Features only via sub-agents.** The operator may drive feature work, but
   only by dispatching sub-agents or a dynamic workflow — never by writing the
   feature itself. Enforced mechanically: see § Enforcement.
3. **Directly permitted, because review needs it:** reading any diff, running
   any gate, resolving merge conflicts, fixing doc obligations, retuning a
   ratchet, and running benchmarks. An operator that cannot touch anything
   cannot review competently and degrades into a rubber stamp. The distinction
   is *implementing a feature* versus *integrating one*.
4. **Sole owner of `main`,** and of the GPU (§ Dependencies).
5. Never force-push `main`. A bad push is repaired with a follow-up commit.

## Helper rules

1. **Isolated worktree, always.** Never the primary checkout.
2. **Branch is `row/<ROW-ID>`** — e.g. `row/ENG-MOE-SHARED-AUX`. The row ID in
   the branch name is what makes the claim machine-derivable.
3. **Open the PR as a DRAFT at the START, not at the end.** The draft PR *is*
   the reservation. A PR that appears only at the end leaves a multi-hour window
   in which the claim is invisible, which is the 2026-08-04 race.
4. **Gates green before ready-for-review:** `scripts/agent-preflight.sh` passes,
   and the PR body carries the evidence (what ran, what it proves).
5. **One row per PR**, size-capped, so review stays cheap and the operator does
   not become the bottleneck.
6. **No speed claims.** The GPU is operator-serialized; a contended benchmark is
   void by our own protocol.
7. Rebase onto `main` before marking ready. Force-pushing your own `row/*`
   branch is fine; `main` is never touched.

## Port map — claims are PR-derived

`coordination.md`'s hand-maintained claim table is replaced by a **generated**
view: open draft/ready PR on `row/<ROW-ID>` = that row is reserved; merged or
closed = released. No upkeep, so no rot.

- a row with no PR and no operator reservation is free;
- the operator reserves by opening a placeholder PR or holding the lock;
- anything not backed by a PR expires on a TTL and is auto-released;
- the generated table is a report, never hand-edited.

This is the structural fix for the 106 stale rows: the claim can no longer
outlive the work, because the claim *is* the work.

## READY-FOR-HELPER — what a helper may pick

A roadmap/matrix row is pickable only if ALL hold:

1. a committed `.agents/specs/<slug>.md` covering the spike contract;
2. its gates are defined and runnable;
3. it is CPU-gateable, or its hardware need is named and the row says so;
4. it does not depend on an unmerged PR;
5. no open PR already claims it.

Anything else is invisible to helpers. Without this a helper picks a row whose
state is a lie and produces a PR the operator must reject — the 2026-08-04
audit found 98 rows in exactly that condition.

## Gates — what the operator merges on

Merge criteria are mechanical wherever possible, so review is fast and does not
depend on who reviews:

1. `scripts/agent-preflight.sh` green on the PR head;
2. every state the PR moves is backed by real anchors, and the target state is
   chosen against what the row PUBLICLY claims — a blanket demotion is NOT
   state-neutral (moving 15 gated models to `ANCHOR-BACKFILL` would have
   silently withdrawn their `✅` support claim);
3. same-change doc obligations satisfied (`check-doc-checkpoint.py`);
4. evidence in the PR body matches what the record now says;
5. no speed number that did not come from the operator's serialized runs.

### The record-merge rule (hard, and the failure is SILENT)

`docs/STATUS.md`, `docs/BENCHMARKS.md`, `docs/FEATURES.md`, `.agents/NOW.md`,
`coordination.md` and the area matrices are **keyed records and MUST NOT be
resolved by a three-way text merge.** On 2026-08-04 git produced a plausible
variant of another session's binding numbers with no conflict marker.

Resolution procedure, always: **take `main`'s version wholesale, re-apply your
own edit on top, then verify the other side's lines are byte-identical to
`main`** (hash them). Union-append only for the genuinely append-only logs
(`state.md`, `parity-ledger.md`, `benchmark-record.md`), and repair state-log
order with `scripts/sort-state-tail.py --apply`.

## Dependencies

- **GPU.** One GB10. Benchmarks are operator-only under `${GPU_LOCK}`; a
  contended series is void.
- **Disk.** Worktrees are cheap (shared object store); **build directories are
  not** — roughly 21 GB each, and ENOSPC has previously produced bogus test
  failures on this project. Concurrent helper count is bounded by disk, each
  helper owns its own build dir, and it is pruned when the PR closes.
- **CI.** The diff-scoped gates must never regain a `cancel-in-progress` group,
  or a PR's commit range goes unvalidated.

## Tests to port

None from upstream; this is project protocol, not a vLLM port. Each guard below
ships with a mutation test in `tests/scripts/`, matching every existing record
checker.

## Work breakdown

| W | Item | Gate |
|---|---|---|
| W0 | **LANDED** `scripts/agent-role.py` — role machinery: `.agents/operator.lock` (create-exclusive + TTL + heartbeat), session-scoped role marker, and role resolution printed by `agent-preflight.sh`, which FAILS when a session has not declared one | mutation test |
| W1 | **LANDED (report-only)** `check-role-discipline.py`: a commit on `main` touching feature paths must arrive via a merged `row/*` PR, not a direct push | mutation test |
| W2 | **LANDED** `scripts/claim-view.py` — generated claim view from PR state (`--apply` online, `--check` offline, 14-day TTL) | mutation test + a run against live PRs |
| W3 | **LANDED** `scripts/ready-for-helper.py` — the pickable queue asserting the 5 conditions | mutation test |
| W4 | **LANDED** `.github/pull_request_template.md` + `scripts/check-pr-size.py` (enforced on `row/*` PRs, reported otherwise) | CI |
| W5 | **LANDED** folded into `AGENTS.md` T0 and `.agents/workflow.md` step 0 | `check-protocol-consistency.py` |

W1 is the enforcement mechanism for "operator does features only via
sub-agents": it does not attempt to detect who typed the code, it makes the
*path* the rule — feature code reaches `main` only through a reviewed `row/*`
PR, whoever produced it.

**Activation.** W1 is ENFORCING since `44e8225c` (user-directed 2026-08-05).
`ROLE_DISCIPLINE_SINCE` names that cutover. Turning it on retroactively would redden
history created under the current, explicitly sanctioned direct-push policy. Set
that constant to the cutover commit when the protocol is adopted, and every
commit after it is enforced. Likewise `agent-preflight.sh` PRINTS the role every
run but only fails on `--require-role`, so an undeclared session is visible
before it is fatal.

## Risks/decisions

- **Decided (user, 2026-08-04):** operator may drive features, but only via
  sub-agents; claims PR-derived.
- **Corrected (user, 2026-08-04):** the first draft made environment-derivation
  the PRIMARY way to determine the role. That is circular — the common case is
  several sessions started from ONE environment, indistinguishable until a role
  has already been taken. Role acquisition is therefore an explicit DECLARATION
  (asked when the user has not said), immediately MATERIALIZED into a lock or a
  worktree+PR so it becomes a fact, and only then re-derivable. Deriving is how
  the role SURVIVES, not how it is decided.
- **Risk: review becomes the bottleneck.** Mitigated by one-row size-capped PRs,
  mechanical merge criteria, and merge-first sessions. Watch the open-PR count;
  if it grows monotonically the cap is too loose.
- **Risk: the lock is stale.** `.agents/operator.lock` needs a TTL and a
  heartbeat, or a crashed operator blocks everyone. Breaking a stale lock must be
  logged, never silent.
- **Risk: the user declares two operators.** Handled by making lock acquisition
  create-exclusive: the second declaration FAILS and is offered the helper role,
  rather than both proceeding and racing on `main`.
- **Risk: helpers starve.** If READY-FOR-HELPER is too strict the queue is
  empty. 98 rows currently fail condition 1 or 2, so the first practical step is
  the anchor backfill, not the protocol.
- **Open: sub-agent attribution.** W1 enforces the path, not authorship. If we
  later want "this feature was produced by a sub-agent" to be provable, it needs
  a trailer convention, and trailers are self-asserted.
- **Not addressed:** two humans working simultaneously. This protocol assumes
  one operator; concurrent human pushes remain outside it.
