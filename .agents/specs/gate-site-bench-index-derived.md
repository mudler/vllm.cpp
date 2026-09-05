# The rendered benchmark index is checked against the tree, not against a literal

Identity: `GATE-SITE-BENCH-INDEX-DERIVED`

Issue: [#2643](https://github.com/mudler/vllm.cpp/issues/2643)

Predecessor: `GATE-CI-SITE-HUGO-LANE`
([spec](gate-ci-site-hugo-lane.md), [#1754](https://github.com/mudler/vllm.cpp/issues/1754),
PR [#1830](https://github.com/mudler/vllm.cpp/pull/1830)). That row installed
Hugo on the CI lane and named this literal as the debt it was arming. Its
`## Owed` entry is discharged here.

## Now

`agent-record` is red on `main`, so it is red on every pull request built from
it, and it is charged to changes that cannot have caused it.
`tests/scripts/test_check_site.py`'s
`test_rendered_benchmark_index_links_resolve_to_emitted_pages` derived a set of
detail slugs from the tree and then compared its size with the literal `10`:

```python
self.assertEqual(len(detail_hrefs), 10)
```

`454094f93` published `docs/benchmarks/qwen38-27b-exl3-gb10.md` and its index
row in `docs/BENCHMARKS.md`, which is exactly what AGENTS.md requires of a
published benchmark. The tree went to eleven pages and eleven links, the
literal stayed at ten, and the suite reported `AssertionError: 11 != 10`.

Ordinary, correct work reds the gate. AGENTS.md names that as the defect rather
than the discipline, and it separately forbids the shape: "Never store a
measurement of one file inside another file."

## Scope

Replace the stored count with the invariant the case is named for, derived from
the tree and the render at read time.

Out of scope: `scripts/check-site.py`, `scripts/check-benchmark-index.py`, the
Hugo configuration, and the other six cases in the suite. None of them is
changed.

## Design

The population is **every `docs/benchmarks/*.md`**, section pages included, and
not only the files whose stem is a benchmark ID. `docs/BENCHMARKS.md` carries
one table row per file in that directory, and `scripts/check-benchmark-index.py`
already refuses an index row with no detail file and a detail file with no index
row. The source side of the relation is therefore a bijection by contract, and
the question left for a rendered page is whether Hugo carried it across intact.

Three changes make that checkable without storing anything:

1. `benchmark_slug(href)` reads the slug off the **URL shape** — the last
   segment of an href whose parent path is the `docs/benchmarks` section. The
   old code filtered the hrefs by the very slugs it was about to compare them
   with, so a link to a page that does not exist vanished from the comparison
   instead of failing it.
2. The case asserts **set equality** between the linked slugs and
   `docs/benchmarks/*.md`, and reports both differences by name. Both sides move
   together when a benchmark is published, so no literal has to be edited.
3. `BASE_PATH` is derived from `BASE_URL` rather than written twice, so the
   href the render emits and the path the test looks up on disk cannot drift.

**Duplicates are deliberately tolerated.** #2643 prescribes "linked exactly
once". Uniqueness is derivable, but it is not a property correct content cannot
violate: one benchmark's prose may legitimately link another's detail page from
inside the same table, and that would red a correct index. The defect uniqueness
would catch — two index rows for one benchmark — is already refused by
`scripts/check-benchmark-index.py`, which gates the source side. Asserting it
here would buy nothing and would re-create the class of failure this row exists
to remove. This is the one deviation from the issue's prescribed repair.

## The vacuity guard

Set equality over two empty sets passes. A render that emitted no table, or a
`docs/benchmarks/` that lost every file, would satisfy the comparison and report
a pass — the mute switch this repository has been bitten by before. Both sides
are asserted non-empty first, explicitly, with their own messages, and mutation
M6 below proves that guard fires.

## Risks

- **A future index table that links outside the benchmarks section.** Handled:
  `benchmark_slug` answers `None` for it and the comparison ignores it.
- **A benchmarks-section link carrying an anchor.** Handled: `urlparse` puts the
  fragment in its own field, so the path still resolves to the slug.
- **A sub-directory under `docs/benchmarks/`.** There is none, and the glob does
  not recurse. A page one level deeper would answer `None` and go unchecked; the
  source-side checker still owns it.

## Tests

`tests/scripts/test_check_site.py`, the case named above. No new case: the
change is to what this one asserts, and a second case over the same render would
be a third gate sharing one target list.

## Gates

```sh
hugo version
python3 scripts/check-site.py
python3 scripts/check-benchmark-index.py
python3 tests/scripts/test_check_site.py
```

That is the CI `agent-record` step at `.github/workflows/ci.yml`, plus the
source-side index checker the design leans on.

## Evidence

Measured on this branch, `hugo v0.146.3+extended`, base
`a700e8da69203586aff9ef753607c0dec0191c59`.

Red before, unmodified tree, eleven pages present:

```text
FAIL: test_rendered_benchmark_index_links_resolve_to_emitted_pages
AssertionError: 11 != 10
```

Green after: `Ran 7 tests ... OK`.

Six mutations, one at a time, each asserted to have changed the tree's sha256
before the case ran and each restored byte-identically afterwards. A mutation
that never applied reads as a pass, so the harness refuses to report one.

| # | Mutation | Result |
|---|---|---|
| M1 | index row links `benchmarks/ghost-benchmark.md`, no such page | DETECTED, set equality, `linked but absent from the tree ['ghost-benchmark']` |
| M2 | delete `docs/benchmarks/mlx-lm-apple-m4.md`, keep its index row | DETECTED, set equality |
| M3 | add `docs/benchmarks/orphan-probe.md` with no index row | DETECTED, `present in the tree but unlinked ['orphan-probe']` |
| M4 | publish a twelfth benchmark: page **and** index row | GREEN, which is the point of the row |
| M5 | `excludeFiles` one linked page from the Hugo mount, so the link renders and the page is not emitted | DETECTED by the emitted-page loop, which proves it is still armed independently of M1 |
| M6 | remove every detail page **and** every index row, so both sides are empty | DETECTED by the non-emptiness guard, `docs/benchmarks/ holds no detail pages to link` |

M5 matters because M1 and M2 both fail at set equality before the emitted-page
loop runs. Without M5 that loop would be unproven.

## Stop conditions

Stop and escalate if the repair needs `scripts/check-site.py` or the Hugo
configuration to change: that is a different row. Stop if the source-side
bijection `scripts/check-benchmark-index.py` enforces turns out not to hold,
because this design leans on it.

## Owed

Nothing. One drift was found here and is owned elsewhere:
[#2678](https://github.com/mudler/vllm.cpp/issues/2678), filed against
`ENG-DOCS-SITE`, whose engine-matrix row cites six line anchors into
`tests/scripts/test_check_site.py` that land on no test and undercounts the
cases. It was already wrong at `a700e8da6`, before this change, which only
moved the cases further down. It is not repaired here because citing the cases
by name would leave the cell carrying no anchor `is_test_anchor` recognises,
which is a decision about the anchor contract rather than about this row, and
because `.agents/engine-matrix.md` is a keyed record two open pull requests
already write.

## Outcome

Pending merge.
