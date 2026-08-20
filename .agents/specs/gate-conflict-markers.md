# Spec — no record gate can see a conflict marker

Issue: [#1417](https://github.com/mudler/vllm.cpp/issues/1417)
Row: `GATE-CONFLICT-MARKERS` (unplaced record/gate defect; the tracked tree is a
record surface, not a matrix row)
State: `ACTIVE`

## Scope

Splice a merge conflict into `docs/STATUS.md` and every record gate passes. The
measurement was re-derived for this spec on a detached scratch worktree at
`b537a5344`, not quoted from the issue. Five lines were inserted into the
capability table: a start marker, a duplicated keyed row, a separator, the same
row again, and an end marker. `git diff --stat` read
`docs/STATUS.md | 5 +++++`.

| Checker | Scope | rc | Message |
|---|---|---|---|
| `scripts/check-public-doc-tables.py` | tree | 0 | `OK: docs/BENCHMARKS.md and docs/FEATURES.md are human-readable keyed tables...` |
| `scripts/check-agent-record.py` | tree | 0 | `agent record OK: ENGINE=168 MODEL=377 ...` |
| `scripts/check-doc-checkpoint.py` | range | 0 | `OK: public documents match the claims this change makes.` |
| `scripts/check-issue-index-append-only.py` | range | 0 | `OK: issue index append-only` |

The two range-scoped checkers were run over a detached scratch **commit**, not
over the working tree. A working-tree mutation of a commit-reading checker
returns 0 because the checker never reads the mutated bytes, and that 0 looks
exactly like a gate that cannot detect the defect.

This is not hypothetical. An earlier revision of the branch for
[#1414](https://github.com/mudler/vllm.cpp/issues/1414) carried a `docs/STATUS.md`
mangled by stale working-tree state from a pre-squash branch, together with an
unrelated spec file deleted outright. The full record gate set ran and reported
clean. A person caught it by reading `git diff --stat`.

In scope:

1. One new tree-scoped checker, `scripts/check-conflict-markers.py`.
2. Its registration in `scripts/agent-preflight.sh` and in
   `.github/workflows/ci.yml`.
3. Its mutation suite, `tests/scripts/test_check_conflict_markers.py`.
4. The creation-mutation entry a new checker owes `scripts/check-pr-size.py`
   and its suite, because the checker has no version at the merge base.
5. One in-flow repair the entry above uncovered
   ([#1448](https://github.com/mudler/vllm.cpp/issues/1448)):
   `classify_path` has no entry for a per-run
   `docs/bench-evidence/<run-id>/<file>` directory, so ten tracked paths are
   unclassified and `test_check_pr_size.py`'s sweep case is red on
   `origin/main`. That red blocks the checker-evidence contract of every change
   that edits `scripts/check-pr-size.py`, which registering a new checker
   requires, so this row could not prove its own contract while it stood. The
   repair names the surface and does not widen a rule, and classification of
   all 4811 tracked paths was captured before and after to prove that exactly
   those ten move.

Out of scope, and each for a stated reason:

- **Any change to the four checkers above.** They measure what they were built
  to measure. Widening one of them to also parse markdown well-formedness would
  make the same gap reappear in the next checker that reads a file for its own
  reason. The issue asks for one check in one place, and that is what this is.
- **Markdown well-formedness in general.** A half-merged table row is one shape
  of a broken document. This row refuses the shape that a merge tool writes and
  that nothing else writes, and it does not become a markdown validator.
- **The other half of the #1414 incident, the deleted spec file.** A deletion
  leaves no bytes to grep. It needs a diff-scoped check, it has a different
  design, and merging it into this row would give one checker two jobs.

## Upstream anchors

None. This is a repository checker. vLLM has no equivalent surface, so no
oracle applies and none is claimed.

## Design

`scripts/check-conflict-markers.py` reads every tracked path with
`git ls-files -z`, reads its working-tree bytes, and refuses three line shapes.
Working-tree bytes rather than a commit: the incident this row closes happened
in a working tree, and a checkout in CI carries the committed bytes anyway, so
one read covers both.

### The rule

A line is a **start marker** when it begins with seven `<` characters and a
space. A line is an **end marker** when it begins with seven `>` characters and
a space. Either one, anywhere in a tracked text file, fails the gate.

A line that is exactly seven `=` characters and nothing else is a **separator**,
and it fails **only when it lies inside an open hunk**: a start marker appeared
on an earlier line of the same file and no end marker has appeared since.

The separator rule is conditional because a bare row of `=` characters is legal
markdown. It is the setext heading underline, and it is a horizontal rule. A
gate that fires on ordinary work is the defect, not the discipline. Two
independent narrowings keep it off ordinary text:

- **Exactly seven.** The tree carries lines that start with seven or more `=`
  characters today, in `docs/bench-evidence/gdn-replayssm-w0-20260818/*.log`,
  `tests/parity/goldens/tokenizer_deepseek_v2/corpus.txt` and
  `tests/parity/goldens/tokenizer_qwen36/corpus.txt`. Every one of them is
  longer than seven, so a `^=+` rule would have fired on five files on arrival.
- **Adjacency.** Even a line of exactly seven is silent unless a real marker
  opened a hunk above it.

The separator therefore adds no detection power over the two markers on its own,
and that is deliberate. Its job is to name the middle of the hunk in the report,
so the reader sees the whole conflict rather than its first line.

The diff3 `|||||||` marker is deliberately absent. A diff3 conflict still
carries the start and end markers, so nothing escapes, and a line of seven `|`
characters is a plausible empty row in a repository whose records are wide
markdown tables.

### No allowlist, and no self-exclusion

The checker builds its own patterns from character repetition (`"<" * 7`)
instead of writing a marker literal. Its suite builds fixtures in a temporary
git repository. Neither file therefore contains a marker at the start of a line,
so neither needs an exclusion, and the repository needs no allowlist file for
any change to append itself to. `AGENTS.md` § *Records* forbids a surface that
every pull request must edit, and an allowlist is exactly that surface.

### The cheap reject, and what it is not

A marker line necessarily contains the marker as a substring, so a buffer
holding neither substring anywhere cannot hold one at the start of a line. The
per-line pass is skipped for such a buffer, which is nearly every file in the
tree. This is a **pure optimization**: it decides nothing the full scan would
decide differently. It is measured as such in `## Evidence`, where deleting it
leaves the whole suite green, and that green is the expected result rather than
a gap. The semantic guard is the adjacency test, and the case above proves it.

The scan reads bytes and never decodes a whole file. Decoding 142 MB of tracked
text to find a marker in none of it cost more than reading it: 0.9 s of CPU
against 0.29 s.

### The unmerged index, counted once

`git ls-files` emits stages 1, 2 and 3 for a path with an unresolved merge, so
a live conflict names the same file three times. `tracked_paths` deduplicates
with `dict.fromkeys`, which keeps git's order. Without it the file is read three
times, its findings print three times, and `examined` over-counts by two per
conflicted path: `9 findings in 1 file; examined 3 tracked text files` for one
file. The verdict was 1 either way, so this never caused a miss, and the count
was wrong in exactly the state the gate exists for.

`test_an_unmerged_index_is_counted_once` builds the state by making `git merge`
fail, not by writing markers into a file, and it asserts the three stage entries
before it asserts anything else. Only a real merge failure puts three stages in
the index, and a fixture that faked it would pass with the dedupe removed.

### What is skipped, and counted

Three classes are skipped, and each is counted and reported rather than dropped
in silence:

- **Binary files.** A NUL byte in the first 8192 bytes decides it, which is
  git's own heuristic. `tests/**/fixtures/*.bin` and the `.npy` goldens carry
  arbitrary bytes, and a marker byte sequence inside a tensor dump is not a
  conflict.
- **Symlinks.** Reading one follows it out of the repository or fails on a
  broken link. The link is not text, and its target is examined on its own if
  it is tracked.
- **Paths absent from the working tree.** A tracked file deleted in the working
  tree has no bytes to read.

The binary test reads the first 8192 bytes and stops there for a binary file, so
the 175 MB of tracked bytes are not all read.

A file the run could not read is none of those three. It is not a skip, because
the run does not know what it holds, and it is not a finding, because an
`OSError` is not a merge conflict. It takes its own list, its own exit status of
2 and its own remedy. Filing it under the findings made an I/O error exit 1 and
print "Resolve the merge before committing", which named the wrong problem and
the wrong repair.

### The report

A clean run prints the counts on one line and exits 0:

    conflict markers: 0 findings in 4744 tracked text files (58 binary, 0
    symlink, 0 absent skipped, 4806 tracked paths)

A run that examined zero text files exits 2 and says so. A gate that examined
nothing has not reported, and a `git ls-files` that returns nothing is a broken
invocation rather than a clean tree.

A failing run names `path:line`, the shape, and the line, and it prints the
remedy: resolve the merge, or, for a document that quotes a marker on purpose,
indent the quoted marker so it does not start at column 0.

### Where it runs

Two registrations, which is this repository's convention for a checker that
must not depend on a human remembering it:

- `CHECKERS` in `scripts/agent-preflight.sh`, so it runs before every commit and
  every push, beside the four gates it complements.
- A step in the `agent-record` job of `.github/workflows/ci.yml`, so a pull
  request is gated even when preflight was never run.

`SUITES` in preflight and the same CI step carry the mutation suite.

## Risks

**A document that quotes a conflict marker at column 0.** A future guide about
resolving merges would fire this gate. No tracked file does today, measured with
a grep whose pattern was first proven against a positive control. The remedy is
in the failure message and costs two spaces of indentation, and it is per
instance and in the text, not a registry entry. Recorded here as the deliberate
cost of having no allowlist.

**A conflict written with a marker length other than seven.** Git writes seven.
A hand-typed six or eight passes. The gate refuses what the tool produces, which
is the incident it exists to catch.

**A `git` that fails.** `git ls-files` returning non-zero, or an invocation from
outside a repository, exits 2 with the git error attached. Unknown is not
absence.

**The checker resolves its root from its own path.** Run from a linked worktree,
`scripts/check-conflict-markers.py` reads that worktree, because the script file
is physically inside it. Run by absolute path from the shared checkout, it reads
the shared checkout. `--root` makes the target explicit, and the report names
the resolved root so no run has to be trusted about which tree it read.

**Cost.** The tree scan must stay cheap enough to sit in preflight. Measured in
`## Evidence`.

**The unmerged index was NOT SEEN when this spec was first written, and is
recorded here as a miss rather than as a considered trade-off.** The first
version of this section reasoned about which files the scan reads and never
asked what `git ls-files` returns while a merge is in flight. A fresh review
found the triple count. It is repaired above, and the case that proves the
repair is falsifiable is named in `## Tests`. The general lesson is the one this
row already carries: a gate is measured by what it says it examined, and the
count is easiest to get wrong in the very state the gate exists for.

## Tests

`tests/scripts/test_check_conflict_markers.py`, registered in preflight's
`SUITES` and in CI. Every fixture case builds a temporary git repository and
runs the shipped checker against it with `--root`, so no case can pass on a
fixture the tree does not have, and no case writes into the repository.

The red-before case is the first one: markers written into a scratch copy must
produce a NON-ZERO exit.

- `test_a_full_conflict_hunk_is_refused` — start, separator and end spliced into
  a tracked markdown file. Requires exit 1 and requires the report to name the
  line number of each of the three.
- `test_a_lone_start_marker_is_refused` and
  `test_a_lone_end_marker_is_refused` — half a conflict is still a conflict.
- `test_a_bare_separator_alone_is_not_refused` — a setext heading underline of
  exactly seven `=` characters, in a file with no marker. Exit 0. This is the
  case that keeps the gate off ordinary documents.
- `test_a_long_rule_of_equals_is_not_refused` — the shape the shipped evidence
  logs and tokenizer corpora carry.
- `test_a_separator_outside_an_open_hunk_is_not_reported` — a marker-free file
  whose separator follows other text is silent.
- `test_a_separator_after_the_hunk_closes_is_not_named` — a file that carries a
  closed hunk AND a legal setext underline below it. The exit code is 1 either
  way, so this case asserts the REPORT: lines 2, 4 and 6 are named and line 8 is
  not. **Written because the first mutation run found the adjacency guard was
  not load-bearing.** Dropping `open_at is not None` left every other case
  green, because a file with no marker leaves the scan early and the guard never
  decides anything there. The absence assertion carries its own positive control
  in the same case: three named lines prove the match works before the fourth
  claims a line is absent.
- `test_a_binary_file_with_marker_bytes_is_skipped` — exit 0, and the report
  counts one binary skip.
- `test_a_symlink_is_skipped` and `test_a_tracked_path_absent_from_the_worktree_is_skipped`.
- `test_an_untracked_file_is_not_examined` — the gate reads what the repository
  tracks.
- `test_crlf_line_endings_are_refused` — a conflict written by a tool on Windows.
- `test_a_tree_with_no_text_files_exits_two` — the vacuity floor.
- `test_an_unmerged_index_is_counted_once` — a real `git merge` conflict, with
  the three stage entries asserted first. Reds without the dedupe.
- `test_an_unreadable_tracked_file_exits_two_and_names_no_merge` — a `chmod 000`
  tracked file exits 2, is named, and does not draw the merge remedy.
- `test_two_paths_sharing_a_colon_prefix_count_as_two_files` — two tracked paths
  that share everything before a colon. The old file count split each report
  line on its first colon and collapsed them into one.
- `test_the_shipped_tree_is_clean` — runs against the real repository root,
  requires exit 0, and requires the reported examined count to EQUAL git's own
  tracked text set, derived at read time from
  `git grep -I --name-only -e ''` plus the tracked files that have no lines at
  all. The first version asserted `> 1000` against a real 3733, which is 27% of
  the count and would have stayed green over a scan that collapsed to markdown
  alone. Equality is strictly stronger and stores no number, so it is not the
  drift lock `AGENTS.md` forbids: both sides are re-derived on every run. A file
  that `.gitattributes` marks binary while carrying no NUL byte would separate
  the two sets, and the failure message names that cause beside the other one.
- `test_the_checker_is_registered_in_preflight_and_ci` — the gate runs somewhere.

## Gates

Each command run bare, with its exit code echoed unpiped. `cmd | tail` reports
`tail`'s status, which is how this tree has produced false green verdicts.

| Gate | Command | Expected |
|---|---|---|
| G1 red-before | the four record gates on a mangled scratch commit | all rc 0, which is the defect |
| G2 red-before | `python3 tests/scripts/test_check_conflict_markers.py` before the checker exists | non-zero |
| G3 green | `python3 scripts/check-conflict-markers.py` | rc 0, count reported |
| G4 green | `python3 scripts/check-conflict-markers.py --root <mangled worktree>` | rc 1, naming the three lines |
| G5 green | `python3 tests/scripts/test_check_conflict_markers.py` | rc 0, `Ran N tests` with N greater than zero, `OK` |
| G6 mutation | the detection core deleted in a scratch copy | the suite goes RED |
| G7 unchanged | the four existing checkers on the clean tree | rc 0, byte-identical output before and after this change |
| G8 full | `scripts/agent-preflight.sh` | green |

## Evidence

Captured verbatim with exit codes in the pull request body: the red-before
transcript, the green-after transcript, the self-mutation result, the examined
count on a clean tree, and the wall time of the tree scan.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the rule fires on any file of
  `origin/main` as it stands. That would mean the rule is wrong, not the tree,
  and widening it to pass is the one move the protocol forbids.
- Stop and report `NEEDS_DECISION` if closing the gap needs an allowlist file
  that every change must append to. That is a lock, and a lock is not a gate.
- Stop if the suite can only go green by widening the marker set beyond the
  three shapes #1417 names.
- Stop if the tree scan costs more than one second of CPU. The bound is CPU and
  not wall time on purpose: this box runs several agent sessions at once, and
  the same scan measured 1.17 s to 4.60 s of wall against 0.29 s of CPU. A wall
  bound would report the other sessions, not this gate.

## Now

`ACTIVE`. The spec commits first, then the checker, its suite, and its two
registrations in one pull request.
