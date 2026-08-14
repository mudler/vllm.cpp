# One PR carries the spec and its code, and the prose around both has a stated style

Issues: [#811](https://github.com/mudler/vllm.cpp/issues/811),
[#812](https://github.com/mudler/vllm.cpp/issues/812)
Row: `POLICY-SINGLE-PR-AND-STYLE`

Two changes land together because they are the same change seen twice: how the
work is packaged for review, and how the words around it are written. Neither
touches product code.

## Why

### The PR shape

`AGENTS.md` §*Spec before code* says the spec "is committed *before*
implementation, never written up afterwards". That sentence is about **commit
order**. It is being read as **PR order** — land a spec-only PR, wait for the
merge, open a second PR with the code.

The reading is understandable and it is wrong. Commit order is what the rule
protects, because commit order is what proves the spec was not written up
afterwards, and `git log` on the branch shows it either way. Two PRs buy nothing
extra and cost a review round trip, leave a spec on `main` describing code no
reader can see yet, and let the two halves drift.

### The prose

`AGENTS.md` says what a commit must **carry** and what each public document is
**for**. It says nothing about how any of it is written. Commit subjects, commit
bodies, PR descriptions, spec prose, record prose and session output are written
to whatever house style the individual agent arrived with.

Two skills already carry the rules and already exist outside the repo, so no
agent loads them and nothing points at them.

## What is actually there, measured before touching anything

### One enforcer, and it is correct

Exactly one thing in the tree forces the split, and only for one flow:

    scripts/ready-for-helper.py:383
        missing.append("no base-reachable committed spec")

The base is `origin/main`, and `.agents/coordination.md:1938` states the same
rule in prose. A helper agent expands a base commit into a disposable checkout
and runs the row's readiness contract from it, so for a **dispatched helper** the
spec genuinely has to be on `main` before the helper starts. That refusal is not
the defect. It is load-bearing, and this change does not touch it.

Nothing else in the tree — no checker, no template, no guide — requires two pull
requests. The rest is prose being over-read.

### What a commit-style gate can honestly check

Measured over the last 200 non-merge commits, before proposing any checker:

| Candidate check | Violations | Verdict |
|---|---:|---|
| Subject does not end in `.` | 0 / 200 | safe; already the de-facto convention |
| Commit has a body | 23 / 200 | include — the only check with real teeth |
| Subject <= 72 characters | 164 / 200 | **reject** |
| Body wrapped at 72 columns | 3658 lines | **reject** |

The two rejections are the whole risk assessment for part C. A subject-length
gate would fire on 82% of ordinary work and a wrap gate on nearly every commit;
the per-class PR line budgets were retired on 2026-08-10 for precisely this
failure mode, and re-introducing it under a different name would be the same
mistake. They are recorded here so they are not re-proposed next quarter.

The body check is the one deliberate cost. It would have failed 23 of the last
200 commits, and one sentence satisfies it. Developer decision, 2026-08-14:
include it — the reader of a commit already has the diff, and what they do not
have is the reason.

## Design

### A. The PR shape

`AGENTS.md` §*Spec before code* gains a paragraph. One PR carries both, spec
first. Splitting stays available and is never refused; three cases make it the
right answer rather than a fallback:

- **A helper dispatch** — the mechanical case above.
- **A large campaign** — many waves are easier to review when scope is agreed
  before code exists.
- **A scoping change** — a PR that deliberately adds rows, issues and specs and
  no code is already spec-only, and nothing is being split.

The agent asks which shape the developer wants **when it claims the row**, before
the spec is written, and recommends the single PR. Asking later is worse than not
asking: by then the spec is committed and the split has already been decided by
whichever branch the agent happened to be on. The answer is an operational
preference, so it is recorded under `## Git integration` in
`.agents/developer-preferences.md` and not asked again. With nothing recorded and
none of the three cases in play, the default is one PR.

**The single PR is a recommendation, never an enforcement.** A developer who asks
for the split gets the split. No checker is added for this, deliberately: the
shape of a pull request is a developer's call, and a gate that overrode it would
be enforcing the opposite of what this change is for.

`.agents/workflow.md:21` describes the helper sequence and is correct as written.
It gains one sentence saying so, because its position at the top of the
coordination guide is what invites the general reading.

### B. The prose

One copy of each rule set, harness-neutral:

- `.agents/style/commits.md` — from the `writing-commits-and-prs` skill. Commit
  subjects and bodies, PR titles and descriptions, branch names, changelog lines.
- `.agents/style/prose.md` — from the `writing-technical-english` skill.
  ASD-STE100 plus the Google developer documentation style guide. Specs, records,
  docs, READMEs, error messages, session prose.

`.claude/skills/<name>/SKILL.md` become frontmatter stubs pointing at those two
files, so Claude Code surfaces them automatically while every other harness reads
the same single copy through `AGENTS.md`.

Each ported file opens with a **house rules that outrank this file** header. The
generic skills do not know about this repo's mandatory `FOLLOWING_AGENTS_PROTOCOL`
line and three trailers, its `Signed-off-by` prohibition, or its
`type(ROW-ID): subject` convention. Where they differ, the repo wins.

`AGENTS.md` gains a short `## How we write` section pointing at both files.

**The style binds new prose only.** No existing file is rewritten to satisfy it.
`AGENTS.md` is dense, rhetorical and does not comply; an agent rewriting it to
pass a style rule would be a worse outcome than the drift. The rule says so in
its own text, because otherwise it will happen.

### C. The checker

A new `scripts/check-commit-style.py`, carrying the two measured-safe checks and
nothing else. It is a new checker rather than an extension of
`check-commit-trailers.py`, whose name would then no longer describe what it
enforces — and a checker's own message is the authority on what it enforces.

- Walks the same commit range preflight already walks for trailers.
- Skips merge commits, which have no authored body.
- Applies from a named cutover commit forward, the same git-native mechanism
  `check-commit-trailers.py` already uses for pre-contract history. History is
  not rewritten and not retro-failed.

## Scope

New: `.agents/style/commits.md`, `.agents/style/prose.md`,
`.claude/skills/writing-commits-and-prs/SKILL.md`,
`.claude/skills/writing-technical-english/SKILL.md`,
`scripts/check-commit-style.py`, `tests/scripts/test_check_commit_style.py`.

Edited: `AGENTS.md` (§*Spec before code*, new §*How we write*),
`.agents/workflow.md`, `.agents/porting-a-model.md:52`,
`.agents/developer-preferences.example.md` (the new `## Git integration` key),
`.github/pull_request_template.md`, `scripts/agent-preflight.sh` (wire the
checker), `.agents/roadmap_v1.md` (the issue table, two rows).

Explicitly **not** touched: `scripts/ready-for-helper.py`,
`.agents/coordination.md:1938`. The base-reachable-spec proof stays exactly as it
is. Changing it would be a checker semantics change owing its own red-before
evidence, and it is the correct behaviour for the flow it guards.

Also not touched: any existing prose, anywhere, for style reasons.

## Gates

Part A and part B are prose. Their gate is `scripts/agent-preflight.sh` green and
a fresh reviewer reading the result.

Part C adds a checker, so it owes the evidence `AGENTS.md` §*Changing the rules or
a checker* requires:

- **Red before.** A crafted commit whose subject ends in a period and whose body
  is empty passes the tree as it stands today. Captured before the checker exists.
- **Green after.** The same commit is reported, with both failures named
  separately. A commit that fixes only one of the two still fails on the other.
- **No false red on ordinary work.** The checker runs over the last 200 non-merge
  commits and reports exactly the 23 empty-body cases and zero trailing-period
  cases, matching the measurement above. A different count means the checker is
  measuring something other than what was surveyed, and the design is wrong.
- **Cutover holds.** With the cutover set, the same 200-commit run is silent.
- Full `scripts/agent-preflight.sh` green. No checker starts passing something it
  previously failed.

## Risks

**The body check adds friction to trivial commits.** It would have failed 11.5% of
recent history. Accepted with the developer's explicit decision, and one sentence
clears it.

**An agent rewrites existing files to satisfy the new style.** The most likely
failure of part B, and the reason "new prose only" is written into the rule
itself rather than left implicit.

**The three split cases become a loophole.** "This is a large campaign" could
excuse any split. It does not need guarding: the split was never forbidden and the
developer decides. The cases are there to name what is legitimate, not to gate it.

**Disk.** The shared checkout sits at 92% with 188 worktrees, and this change is
docs-only. Noted because ENOSPC in this repo surfaces as false policy refusals
from checkers, which would be misread as a defect in this diff.

## Stop conditions

Stop and report rather than proceeding if:

- The checker cannot reproduce the 0 / 23 measurement over the surveyed range.
  That means the survey and the implementation disagree about what they count,
  and shipping either one would be shipping a number nobody verified.
- Removing the two-PR reading turns out to break a flow other than helper
  dispatch. That would mean a second enforcer exists that this spec did not find,
  and the design needs it before it is written, not after.
- Either ported skill contradicts a standing `AGENTS.md` rule in a way the house
  rules header cannot resolve by simple precedence. That is a real conflict and a
  developer decision, not an editing problem.

## Now

Spec committed as the first commit on `row/POLICY-SINGLE-PR-AND-STYLE`, which is
this change dogfooding its own rule. Implementation follows in the same PR, after
developer review of this file.
