# ENG-RECORD-CLAIM-AGREEMENT — make one class of duplicated-fact drift mechanically detectable

Row: `ENG-RECORD-CLAIM-AGREEMENT`
Matrix: [engine-matrix.md](../engine-matrix.md)
Base SHA: `e1097c5e4`
Issues: `ISSUE-LOCAL-01M2A93TXHQ9J11AQ82SX7DX60` (the checker) and
`ISSUE-LOCAL-01M2A9490HQ72W884178ZGRZX6` (the four stale records this sweep
found, which land beside it)

## Now

`READY`. This spec is committed before the checker, and it carries the
measurement that decides the checker's ONE rule and rejects three others.

## Scope

One checker, `scripts/check-record-claim-agreement.py`, with ONE rule:

> A commit sha written in the strict form `` `<sha>` (<YYYY-MM-DD> `` under
> `.agents/` must carry the SAME date at every site that writes it that way.

In scope: the checker, its mutation-backed tests, its registration in
`scripts/agent-preflight.sh` and CI, and the one live disagreement the rule
finds today. Out of scope: every other claim shape, for the reasons measured
below. This checker does not attempt to decide whether a date is CORRECT — only
whether the tree states it consistently.

### Why this row exists

The DeepSeek-V4.1-Flash scoping change (`97cb6964b`, 2026-09-12) failed **seven
consecutive fresh reviews**, every one on the same shape: a fact corrected in
one place and left standing in another. Ten instances. **Six were introduced by
the repair rounds themselves**, and the tenth was in the commit message of the
session that had spent six rounds hunting the other nine.

`check-agent-record.py` and `check-model-checklist.py` stayed GREEN through all
ten. They verify structure, cardinality and anchor liveness. Nothing in this
tree compares two copies of one fact, and a fact here lives in up to six places:
a checklist cell, a detail cell, counting prose, an oracle file, a spec section
and an issue file. `AGENTS.md` §Records already prescribes the cure for the
storage half of this ("a value that is derived at read time and is not stored"),
and stops at counts.

### The measurement that chose the rule

Run over `.agents/**/*.md` excluding the frozen `completed/` archive:

| Trigger | Shas | Sites | Flagged | Real |
|---|---|---|---|---|
| sha and a date anywhere on one line | 609 | — | 29 | few |
| date bound to the sha (`(DATE`, `, DATE`, ` on DATE`) | 144 | — | 4 | 1 |
| **strict: `` `sha` (DATE `` ** | **56** | **88** | **1** | **1** |

**The strict row is reproducible, and here is what reproduces it.** An earlier
draft stated the pattern in prose as `` `<7-40 hex>[…]` \s* \( <YYYY-MM-DD> ``
and called the row reproducible without saying how. That prose is ambiguous in
two ways that both move the numbers, so it is replaced by a command. Save this
as `/tmp/strictform.py` and run it from the repository root:

```python
import re, subprocess, collections, sys
REV = sys.argv[1] if len(sys.argv) > 1 else ''      # '' = working tree
PAT = re.compile(r'`([0-9a-f]{7,40})(?:…|\.\.\.)?`[ \t]*\((\d{4}-\d{2}-\d{2})')
ls = (['git', 'ls-tree', '-r', '--name-only', REV, '.agents/'] if REV
      else ['git', 'ls-files', '.agents/'])
files = sorted(f for f in subprocess.run(ls, capture_output=True, text=True).stdout.split()
               if f.endswith('.md') and not f.startswith('.agents/completed/'))
read = (lambda f: subprocess.run(['git', 'show', REV + ':' + f],
                                 capture_output=True, text=True).stdout) if REV \
       else (lambda f: open(f, encoding='utf-8').read())
sites = [(m[0], m[1], f) for f in files for m in PAT.findall(read(f))]
shas = sorted({s for s, _, _ in sites}, key=len, reverse=True)
canon = {s: next((t for t in shas if len(t) > len(s) and t.startswith(s)), s) for s in shas}
by = collections.defaultdict(set)
for s, d, _ in sites:
    by[canon[s]].add(d)
print(REV or 'WORKTREE', 'sites', len(sites), 'literal shas', len(shas),
      'after prefix-merge', len(by), 'flags', sum(1 for v in by.values() if len(v) > 1))
```

Run on 2026-09-12 it prints:

```text
$ python3 /tmp/strictform.py e1097c5e4
e1097c5e4 sites 88 literal shas 56 after prefix-merge 52 flags 1
$ python3 /tmp/strictform.py
WORKTREE sites 90 literal shas 56 after prefix-merge 52 flags 0
```

**Two readings are fixed by that command, and the prose left both open.**
(1) The whitespace between the closing backtick and the `(` is `[ \t]*`, a
SAME-LINE reading. Run literally with `\s*`, which crosses newlines, the same
corpus gives `sites 100 literal shas 62 after prefix-merge 56 flags 1` at the
base and `sites 102 ... flags 0` in the tree — a different pair of numbers for
the same stated rule, which is why the reading is now written down rather than
implied. The same-line reading is the one W1 must implement, because a sha at
the end of one line and a date opening the next are not one citation.
(2) The prefix-merge rule this spec decides on below IS applied, and it is the
`after prefix-merge` column: 52, not 56. **The table row above reports the
LITERAL sha count, 56, because that is what the earlier draft measured**, and
both numbers are printed here so neither reading can be mistaken for the other.
The literal count is the wider one and W1's tests pin the merge behaviour
directly, so the rule does not depend on which column is quoted.

An earlier draft of this row said 49 shas and 81 sites; those came from a
narrower walk and are not reproducible under any reading of the rule as this
spec states it. **The two numbers that decide the rule are unchanged and are
reproduced by the command above: one flag at the base, zero in the working
tree**, which is this change's RED-BEFORE and GREEN-AFTER. W1 restates none of
these counts as a constant.

**The two looser rows above are NOT re-derived, and they are not reproducible
from this spec**, because it records the triggers in prose rather than as the
patterns that were run. Re-running plausible reconstructions of them at the base
SHA gives 1313 shas / 156 flags for the first and 125 shas / 3 flags for the
second, against the 609/29 and 144/4 recorded. Saying that is more useful than
silently replacing one unreproducible pair with another. What survives
re-measurement is the POLARITY that rejected them, and it survives under every
reconstruction tried: each looser trigger flags one to two orders of magnitude
more than the strict form, and the additional flags are dominated by a sha
legitimately carrying a second date. W1 owes the exact patterns beside any
number it keeps.

The strict form is the shipped rule: one flag, and it is a true positive. The
looser triggers fail because **a sha legitimately carries several different
dates** — its author date, its committer date, the date we pinned it, the date
we measured against it. "On 2026-09-03 the parity pin advanced to `e126687a9a`"
and "`e126687a9a` (2026-08-31)" are both correct and describe different events.
A rule that cannot tell those apart would fire on ordinary work, and a gate that
fires on ordinary work is the defect, not the discipline.

### The one live disagreement, and the class it belongs to

**At the base SHA `e1097c5e4`, and no longer in the working tree**,
`a3d330289752192754277638fe5c09eb2fb49763` (tenstorrent/tt-metal) carried the
date 2026-05-20 in the strict form in
[tenstorrent-gdn.md](tenstorrent-gdn.md) and 2026-08-09 in the strict form in
[tt-metal-trace-replay-write-desync.md](tt-metal-trace-replay-write-desync.md).
Re-read the pair with
`git show e1097c5e4:.agents/specs/tenstorrent-gdn.md | grep -n a3d33028`.
**Both dates are deliberately NOT re-written in the strict form anywhere in this
spec**, because a checker that collects the strict form would collect this
narrative too and then flag the document that explains the rule — the same
self-reference trap as a stored count inside its own file. This change
reconciles `tenstorrent-gdn.md` onto the committer date, so the strict form now
agrees at both sites.

Resolved against the forge on 2026-09-12
(`gh api repos/tenstorrent/tt-metal/commits/a3d3302897…`): **author**
`2026-05-19T22:41:16Z`, **committer** `2026-08-09T08:12:00Z`. Neither record is
inventing anything — one took the author date (and rendered it 2026-05-20,
which is what `2026-05-19T22:41:16Z` reads as in any zone at or east of
`+01:19`), the other took the committer date.

**Instance 6 of the ten belongs to the same CLASS and NOT to the same
mechanism**, and an earlier draft of this paragraph called it "the identical
trap", which the forge falsifies. vLLM's `98ed0856f3` was recorded as 2026-09-04
in three places from its `+0800` AUTHOR rendering. Read both ways on 2026-09-12:

```text
git log -1 --format='%aI %cI' 98ed0856f3
  -> 2026-09-04T00:40:35+08:00   2026-09-03T09:40:35-07:00
gh api repos/vllm-project/vllm/commits/98ed0856f3… --jq '.commit|{a:.author.date,c:.committer.date}'
  -> {"a":"2026-09-03T16:40:35Z","c":"2026-09-03T16:40:35Z"}
gh api repos/tenstorrent/tt-metal/commits/a3d3302897… --jq '.commit|{a:.author.date,c:.committer.date}'
  -> {"a":"2026-05-19T22:41:16Z","c":"2026-08-09T08:12:00Z"}
```

`98ed0856f3`'s author and committer stamps are the SAME INSTANT,
`2026-09-03T16:40:35Z`; they differ only in ZONE, `+0800` against `-0700`, and
the `+0800` rendering crosses midnight into 2026-09-04. `a3d3302897`'s two
stamps are genuinely DIFFERENT MOMENTS, 82 days apart — authored 19 May, merged
9 August.

**So these are two distinct mechanisms by which one sha acquires two defensible
dates:** (i) one instant rendered in two zones across a date boundary, and (ii)
an author date and a merge date that are different moments. What the rule gates
is the shared CONSEQUENCE and not a shared cause — two records writing different
dates for one sha in the strict form. The honest recurrence claim is therefore
about the class, not the confusion: **a sha's stated date depends on which stamp
and which zone the writer read**, and that class produced a defect in two
upstream repositories by two different routes. That is the argument for gating
the consequence, which is the only part a checker can see.

## Our baseline

`scripts/check-agent-record.py` (2400+ lines) already parses every matrix row,
validates issue references and anchors, and prints per-matrix cardinalities.
`scripts/check-model-checklist.py` enforces the model rollup against parsed row
states — the only existing agreement check in the tree, and it is confined to
one file's counts. Neither reads two documents and compares a shared value.

## Upstream chain

None. vLLM has no equivalent; this is a repository-protocol gate, not a ported
behaviour. The rule's subject matter is this tree's own records.

## Port map

`scripts/check-record-claim-agreement.py`, new. Registered in the `CHECKERS`
array of [`scripts/agent-preflight.sh`](../../scripts/agent-preflight.sh) and in
the checker job of `.github/workflows/ci.yml`, beside `check-model-checklist`.

## Tests to port

None to port. New: `tests/scripts/test_check_record_claim_agreement.py`,
mirroring the structure of `tests/scripts/test_check_oracle_pins.py`.

## Gates

- **RED-BEFORE, on the live tree**: the checker must FAIL on `.agents/` as it
  stands at the base SHA, naming `a3d3302897` and both sites. This is real
  red-before evidence and not a synthetic fixture.
- **GREEN-AFTER**: the checker passes once the two tt-metal records are
  reconciled in this same change.
- **Mutation**: each assertion the tests claim must be proven by mutating the
  checker and observing the specific test fail — a same-date pair must not
  flag; a differing-date pair must flag; the `completed/` archive must be
  excluded; a sha written in prose form with a different date must NOT flag.
- Both existing record checkers stay green, read by TEXT and not exit code.

## Dependencies

- No network. The checker never resolves a sha; it compares what the tree says
  against what the tree says elsewhere. Resolving `a3d3302897` against the
  forge was done once, by hand, to decide which record to correct.
- Python 3 only, no new package.

## Work breakdown

- **W0 (this spec).** The measurement, the rule, and the three rejected rules.
- **W1.** The checker, its tests, its registration, and the tt-metal
  reconciliation that takes it from red to green.

## Risks/decisions

- **Decision: ONE rule, not a framework.** Three candidate rules were measured
  and REJECTED, and they are recorded here so nobody re-derives them:
  1. **Issue-state agreement** (a document calling `#N` owed while the canonical
     local issue says closed). INFEASIBLE offline: there is no local record for
     `#2794` at all, the very issue that motivated it, and **831 of 1050** local
     issues carry `State: UNKNOWN`. The rule would be inert and would still miss
     its own motivating case.
  2. **Validating a stated date against git.** IMPOSSIBLE in CI: the shas are
     upstream (vLLM, llama.cpp, tt-metal), this checker may make no network
     call, and `.env`'s `VLLM_SOURCE` is empty in a fresh checkout.
  3. **General natural-language claim agreement** ("five subsystems" versus
     six, "three boxes" versus four, a lane called admissible in one cell and
     expired in another). NOT mechanically extractable. Four of the ten
     instances were of this kind, and they remain a review obligation. Saying
     so is more honest than a regex that pretends otherwise.
- **Risk: the rule is narrow.** It would have caught 1 of the 10 instances that
  motivated it, plus the pre-existing tt-metal one. It is shipped because its
  precision on the live corpus is 1/1 and because its subject CLASS — a sha
  whose stated date depends on which stamp and which zone the writer read — has
  now produced a defect in two upstream repositories, by two different
  mechanisms (§"The one live disagreement, and the class it belongs to" states
  both, with the commands). It is not shipped
  because it solves the general problem, and the two mechanisms are NOT one
  confusion recurring.
- **Risk: a legitimate second date in the strict form.** If a record ever needs
  to write two different dates for one sha in the strict parenthetical form, the
  correct repair is to say WHICH date each is (`(committer 2026-08-09)`), not to
  widen the checker. The tests pin that the strict form is the trigger.
- **Decision: an ABBREVIATED sha with a trailing ellipsis inside the backticks
  IS the strict form, and W1's tests must pin that in both directions.** The
  corpus writes shas both ways. [tenstorrent-gdn.md](tenstorrent-gdn.md) spells
  the tt-metal sha in full before its date, while several `gh api` invocations
  and narrative citations abbreviate it and close the backticks with `…`. If
  W1's pattern is anchored on a bare hex run (`` `[0-9a-f]{7,40}` ``) it will
  not match an elided citation, and the checker is then silently blind to a
  whole citation style while reading as green. The DECISION is that the pattern
  accepts a 7-to-40 hex run optionally followed by `…` or `...` before the
  closing backtick, and that a prefix match against a longer sha in the same
  corpus is treated as the SAME sha. W1 owes three cases that fail if that
  changes: the ellipsis form IS collected; a pair written once abbreviated and
  once in full is compared as ONE sha rather than two; and a bare abbreviation
  with no date after it is NOT collected. **This spec deliberately writes no
  sha-and-date pair in the strict form itself**, so the checker cannot flag the
  document that defines it; W1 must not "fix" a red on this file by widening the
  rule, per the stop condition below. **One exception, stated because the
  checker will see it:** §"The measurement that chose the rule" quotes
  `` `e126687a9a` (2026-08-31) `` as an ILLUSTRATION of a legitimate second
  date, and **this bullet quotes it a second time**, so the strict form occurs
  TWICE in this file and not once. An earlier draft of this sentence said it was
  the only such pair here and reported "89 sites", counting its own second
  occurrence out. Re-derived on 2026-09-12 with the command in §"The measurement
  that chose the rule" (same-line reading), the working tree carries **90
  sites**, 56 literal shas, 52 after the prefix-merge this bullet's own decision
  requires, and **zero flags**: both occurrences write the same date, so
  `e126687a9a` agrees with itself and with every other site. If a future record
  writes `e126687a9a` with a different date in the strict form, both
  illustrations become real flags at once, and the repair is to quote them
  outside the strict form here rather than to widen the rule.
- **Decision: no waiver list.** `AGENTS.md` has no waiver registry. A change
  that needs an exception argues for it in its own commit message.

## Owed

- The four natural-language instances of this shape have no gate and remain a
  fresh-reviewer obligation. Recorded, not solved.
- ~~`.agents/kernel-matrix.md`'s `## Count invariants` block stores measurements
  of its own table that have drifted (38 practical rows against 60 `KERNEL-*`
  rows today, and a lifecycle baseline summing to the stale 38).~~ **DONE in
  this same change**, not owed: the two stored bullets are retired in favour of
  a dated derivation that states it is not an invariant, and the live
  cardinality is read from `scripts/check-agent-record.py` (`KERNEL=60` on
  2026-09-12). An earlier draft of this bullet listed the work as owed while the
  change performed it — the very defect this row exists to make detectable.

## Stop conditions

- Stop if the rule starts flagging ordinary work. A gate that fires on correct
  records is the defect; narrow the trigger or withdraw it.
- Stop if making the gate green would mean deleting a claim rather than
  reconciling it.
