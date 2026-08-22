# The forge's separator belongs to the co-author block, not to concatenation

Issue: [#861](https://github.com/mudler/vllm.cpp/issues/861)
Row: `GATE-SQUASH-SEPARATOR`

This row corrects a conclusion two earlier rows reached from a simulation, and
the correction was produced by the first commit that actually landed under the
setting those rows changed.

## Why

#829 and #850 concluded that `squash_merge_commit_message = PR_BODY` removed the
`---------` separator, on the theory that GitHub wrote it between concatenated
commit messages. The evidence was a simulation that appended `Co-authored-by:`
without a separator. That is not what the forge does, and a simulation is not the
forge.

`617d6f452`, the merge of #850 and the first squash landed under `PR_BODY`,
fails the gate:

```
617d6f45286b: [trailers] Following-Agents-Protocol must appear exactly once
617d6f45286b: [attribution] AI-Assisted must appear exactly once
```

Its message contains exactly one trailer block, one `FOLLOWING_AGENTS_PROTOCOL`
marker, and still one `---------`. So GitHub writes the rule above the
`Co-authored-by:` block it appends, independently of where the body came from.
`join_trailing_trailer_paragraphs` stops at the first paragraph that is not
trailer-shaped, so the separator still orphans the block.

The setting change remains correct and stays. It makes the body the single
source of the message, which is why the block now appears once. It simply was
not sufficient.

## Design

#850 rejected a checker change because swallowing the old shape needed two
widenings at once, stepping over the separator and admitting repeated blocks.
That reasoning was sound and no longer applies: under `PR_BODY` the body appears
once, so there are no repeats to admit. One widening remains.

`join_trailing_trailer_paragraphs` steps over a paragraph that is entirely a run
of hyphens, and only when trailer-shaped paragraphs sit on BOTH sides of it. A
prose paragraph still terminates the block, so trailers buried mid-message stay
invalid. That is the property the helper exists to protect and it is untouched.

Repeated blocks stay red. Nothing in this change admits them, and the case that
pins that is kept.

## Scope

In scope: `scripts/check-commit-trailers.py` (the fuse and one pattern),
`tests/scripts/test_check_commit_trailers.py`, and the issue index row.

Out of scope and stated as owed:

- #822, the cancelled `main` push runs. It is why a landed commit failing this
  gate does not surface, and it is why every instance of this defect has been
  found by hand. Not this row's fix.

## Gates

Six properties, each asserted in both directions.

| Message shape | Verdict |
|---|---|
| Block, separator, `Co-authored-by:` (the real forge shape) | green |
| Block, `Co-authored-by:`, no separator | green |
| Block then a prose paragraph | red |
| Block, separator, then prose | red |
| Trailers buried mid-message | red |
| Separator with prose above it | red |
| Two blocks with a separator between them | red |

The real landed commit `617d6f452` is the primary case: the checker runs over
`origin/main~1..origin/main` and must pass.

## Risks

**A commit body that legitimately ends with a horizontal rule.** It is stepped
over only when trailer-shaped paragraphs sit on both sides, so a rule that ends
a message, or one with prose on either side, is untouched.

**The separator's exact form is the forge's choice.** The pattern matches a run
of three or more hyphens as a whole paragraph rather than the literal nine, so a
change in width does not reopen the defect. A change in character would, and the
landed-commit case is what would report it.

## Part 2 — the AUTHOR-side divider: a bare `---` in a body (#1563)

Issue: [#1563](https://github.com/mudler/vllm.cpp/issues/1563)

Part 1 is this class arriving from the FORGE side, where GitHub writes
`---------` above the `Co-authored-by:` block it appends. This part is the same
class arriving from the AUTHOR side, where a person writes an ordinary markdown
horizontal rule into a pull request body. The two are not the same string and
they do not have the same effect: nine hyphens orphan a trailer block, three
hyphens truncate the message. It is a separate part rather than a widening of
Part 1 for that reason, and it is on THIS row because `.agents/issue-index.md`
already names `GATE-SQUASH-SEPARATOR` as #1563's owner and that file is
append-only, so a new row ID would leave the index pointing at a row that did
not do the work.

### Why

`squash_merge_commit_message = PR_BODY` makes a pull request body the landed
commit message. Git treats a line of `---` followed by whitespace or end of line
as the start of the patch section, so everything below the first one is not part
of the message. A body that visibly ends with the trailer block can therefore
land a commit whose trailers no `git interpret-trailers` call can see, and
`scripts/check-commit-trailers.py` reports the cause as the opposite of what it
is.

Three shapes were measured against the checker at `db648fb88`, each with the
`---` as the only difference from a body that passes.

**Shape 1, the rule ABOVE the trailers. The message is actively false.**

```text
$ python3 scripts/check-commit-trailers.py --message-file shape1 --filled
commit trailer check FAILED:
  - [trailers] Following-Agents-Protocol must appear exactly once
  - [attribution] AI-Assisted must appear exactly once
rc=1
```

`Following-Agents-Protocol` appears exactly once in that body. The checker says
it must appear exactly once, as though it appeared zero times or twice, so a
reader counts occurrences, finds one, counts again, and dumps bytes before
thinking to test the parser's own framing. The marker rule, which is computed
from the raw paragraphs and is therefore correct, stays silent — so both errors
that do fire point away from the cause.

**Shape 2, the rule BELOW the trailers. The checker passes it.**

```text
$ python3 scripts/check-commit-trailers.py --message-file shape2 --filled
OK: commit trailer contract
rc=0
```

The body is `subject / prose / marker / trailers / --- / prose`. Git truncates at
the `---`, so the trailer block is the last paragraph of what it reads and parses
clean. `test_prose_after_the_trailers_still_fails` pins that trailing prose is
red; writing `---` above that prose turns the same body green. A rule the gate
already holds is bypassed by three characters.

**Shape 3, the rule as a three-hyphen separator.**
`join_trailing_trailer_paragraphs` steps over a paragraph that is a run of three
or more hyphens, so a body of `block / --- / Co-authored-by:` passes, and the
landed commit's forge attribution is below the divider and invisible to
`git interpret-trailers`.

**A `---` in a landed message is real corruption, not a checker artefact.**
Measured on a scratch repository: a commit whose message is shape 1, put through
`git format-patch -1 --stdout` and `git am`, comes back as

```text
subject

prose
```

with `%(trailers)` empty. Everything below the divider — the prose, the marker
and all three trailers — is gone, and `git am` exits 0. That is what the divider
convention is for, and it is why this row detects the line rather than teaching
the checker to ignore it.

### Design

**Mirror git's own rule, do not approximate it.** `find_patch_start` in git's
`trailer.c` scans lines for `skip_prefix(s, "---", &v) && isspace(*v)`.
`patch_section_line(message)` returns the 1-based line number of the first line
that satisfies exactly that, and `None` otherwise. Probed against
`git interpret-trailers --parse` at git 2.43.0, which is the authority the
checker already shells out to:

| line | divider | why |
|---|---|---|
| `---` | yes | the next character is the newline, which is whitespace |
| `--- ` | yes | trailing space |
| `--- a/f.c` | yes | the `git format-patch` diff header this convention exists for |
| `----` | no | the next character is a hyphen |
| `---x` | no | the next character is not whitespace |
| `---------` | no | the forge separator `GATE-SQUASH-SEPARATOR` handles, unchanged |

The forge separator is not a divider and keeps its existing treatment. That is
measured, not assumed: it is why `617d6f452` orphaned a block instead of
truncating a message.

**One new rule, `[framing]`, reported instead of the parse-derived errors.** When
a divider is present, the trailer map is not evidence about the trailers the
author wrote — the parser could not see past line N — so reporting
`Following-Agents-Protocol must appear exactly once` from it is reporting a
measurement of a message nobody sent. `_strict_errors` therefore emits the
framing error and skips every check that reads the trailer map. The marker check
stays, because it reads the raw paragraphs and is unaffected. A body that carries
a divider AND a genuine trailer defect reports the divider first and the trailer
defect on the next run, which is one accurate error at a time rather than two
inaccurate ones at once.

**Both doors, one implementation.** The rule lands in `validate_commit_message`,
which is the single function the range walk and the `--message-file` door both
call. This is the property the checker already relies on and states in its own
comment at `scripts/check-commit-trailers.py:439-443`, and it is not weakened
here.

**Applied to landed commits as well, because nothing landed carries one.**
Measured over all 3118 commits reachable from `main` at `db648fb88`: zero contain
a line that satisfies git's divider rule. So the range walk cannot be turned
permanently red by this, and there is no need for a `--message-file`-only carve
out that would give the two doors two rules.

**Rejected: `git interpret-trailers --no-divider`.** It is one flag and it makes
a markdown rule inert, and it is the wrong repair. It would make the checker
report a body clean whose trailers `git interpret-trailers` cannot read and whose
`format-patch`/`am` round trip deletes them, so the gate would be measuring a
message that git does not agree exists. The checker's job is to predict what
lands, not to be more forgiving than the tool that reads it.

### Which body is being read

The two callers read different bytes, and this row does not change that or blur
it.

| Caller | Bytes | Sees an edit after the last push |
|---|---|---|
| `.github/workflows/ci.yml` `PR_BODY` step | `github.event.pull_request.body`, the FROZEN event payload | no |
| `scripts/agent-pr-body.py --pr N` | `gh pr view --json body`, the LIVE body | yes |
| the squash that lands | the LIVE body at merge time | yes |

So the CI guard can be green on a body that is not the one that lands, which is
the #1263 neighbourhood and is not repaired here. The framing rule is added to
the shared checker, so both doors gain it, and the test suite pins the source of
each rather than asserting that the two agree.

### Scope

In scope: `scripts/check-commit-trailers.py` (the detector and one rule),
`tests/scripts/test_check_commit_trailers.py`, one line in
`.github/pull_request_template.md` warning the author, and the issue index row.

Out of scope and stated as owed:

- #1263, the frozen payload. A body edited after the final push is still not
  re-read by CI. Named in the table above so the two doors are not confused;
  not repaired here.
- Any change to what the contract ACCEPTS. Nothing here turns a red body green.

### Gates

| Message shape | Before | After |
|---|---|---|
| `---` above the trailers | red, naming the trailers | red, naming the `---` and its line |
| `---` below the trailers | **green** | red, naming the `---` and its line |
| `---` as a three-hyphen separator before `Co-authored-by:` | **green** | red, naming the `---` and its line |
| `--- a/file.c` anywhere | green or misreported | red, naming the line |
| `----`, `---x`, `---------` | unchanged | unchanged |
| the shipped pull request template | green | green |
| every commit reachable from `main` | green | green |

`python3 -m unittest tests.scripts.test_check_commit_trailers` is the focused
gate. `scripts/agent-preflight.sh` is the full one.

### Risks

**A body that legitimately wants a horizontal rule.** Markdown has two other
thematic breaks, `***` and `___`, and neither is a git divider. The error names
them, so the author is not left guessing.

**A commit message that quotes a diff.** It is now rejected. It was already
broken — its trailers were below the divider and invisible — so this makes an
existing defect visible rather than creating one.

**The detector drifting from git.** It is pinned by a case per row of the divider
table above, each asserted against `git interpret-trailers --parse` in the same
test, so a git version that changed the convention would report itself.

## Now

Part 1 landed with #861. Part 2 is implementing.
