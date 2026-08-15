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

## Now

Implementing.
