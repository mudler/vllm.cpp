# doc-checkpoint: the quickstart page is a landing source

Issue: [#1520](https://github.com/mudler/vllm.cpp/issues/1520)
Row: `ROAD-V1-QUICKSTART`. This change exists to unblock that row, and it lands
on its own branch, with its own gate evidence, ahead of it. It is a checker
semantics repair and not a roadmap item of its own.

## Now

`IMPLEMENTING`. The one-line widening, its three tests, and the mutation
evidence are on `row/FIX-DOC-CHECKPOINT-LANDING-SOURCE`. The gate evidence below
is captured on that branch.

## Scope

`scripts/check-doc-checkpoint.py` refuses a `README.md` claim change that does
not also touch a member of `LANDING_SOURCE_FILES`. The set names
`.agents/mission.md`, `CMakeLists.txt`, three `benchmarks/demo/*.json` files,
and the two example mains.

In scope: add `docs/QUICKSTART.md` to that set.

Out of scope, and deliberately not changed here:

- `claims_changed`. The alternative repair distinguishes README growth from
  README shrinkage. This change does not attempt it. See `## Why` for the
  reason.
- Every other document. No prefix and no class is admitted. One exact path
  joins the set.
- The `landing_page` class semantics. It continues to permit a `README.md`
  change and never to demand one.
- `USER_USAGE_FILES` and `FEATURE_SURFACE_FILES`, which are untouched.

## Why

The gate's message states the rule it protects:

> The README is the landing page. Routine checkpoints belong in the
> purpose-specific docs. Co-edited public projections never justify README
> churn.

Both named failures are a README that grows to say something a purpose-specific
document already says. `row/DOCS-QUICKSTART-1281` is the opposite. It adds
`docs/QUICKSTART.md` and cuts the README `## Quickstart` block from three
command fences to a four-line pointer at that page. The README loses material,
and the material moves to the purpose-specific document. The gate reds on the
discipline its own message argues for.

Measured at head `1b6e458c8` of that branch: `scripts/agent-preflight.sh`
reports 88 gates ok, zero skips, and one failure, `doc-checkpoint range`. The
branch touches no member of `LANDING_SOURCE_FILES`, and it honestly should not,
because nothing in `.agents/mission.md` became untrue.

The widening is narrow and it has a direction. Every existing member of the set
is something the README quotes: the mission statement it paraphrases, the build
entry point its build line invokes, the demo measurements its numbers come from,
and the two example mains its commands run. `docs/QUICKSTART.md` is the same
relation with the direction made explicit. The README's claim about where a
reader starts changed because that page now exists and now owns the starting
procedure. The page is a source and not a projection. Nothing else in the tree
records what it says, and the README defers to it rather than duplicating it.

This is why the set is the right place for the repair. The rule is "a README
claim needs an underlying source", and the quickstart page is one. The two
alternatives are worse:

- Teaching `claims_changed` to allow a shrinking README treats size as a proxy
  for honesty. A README can also shrink by deleting a claim that is true and
  load-bearing, so the proxy admits a change the rule exists to catch, and it
  makes a general function answer a question about one document.
- Accepting that the landing page cannot point at the landing document drops
  half of `ROAD-V1-QUICKSTART` to satisfy a gate, which inverts the relation
  between the record and the work.

The previous implementer on `row/DOCS-QUICKSTART-1281` refused to widen the set
inside its own change, because `AGENTS.md` names making a red gate green that
way as the forbidden move. That refusal was correct, and this spec exists so the
same edit is made deliberately, with its argument and its evidence, instead of
as a convenience inside the change it unblocks.

## Design

One entry added to `LANDING_SOURCE_FILES`, with a comment beside it that records
the date, the issue, the relation that admits it, and the class it does not
admit. No function changes. `classify()` already turns a member of the set into
the `landing_page` class, and `errors_for()` already treats that class as
permission rather than obligation, so the entry needs no other code.

Polarity, stated because it is the risk: this WIDENS a gate. A change that
edits `docs/QUICKSTART.md` may now also make any `README.md` claim change,
including one that grows the README. That is the same permission every existing
member of the set already carries, and it is the permission the set is for.

## Risks

- **The permission is not scoped to the quickstart claim.** A change touching
  the quickstart page can rewrite any part of the README. Accepted, because it
  is the shape of the whole set and narrowing it needs a claim-level model of
  the README that no part of this checker has. Recorded under `## Owed`.
- **The entry is an exact path with no existence check.** If the page is
  renamed or removed, the entry goes stale silently and the README loses its
  permission with no message that says why. Accepted for now, because the same
  is already true of all seven existing members, and a path-existence assertion
  over the set is a separate change with its own red-first evidence. Recorded
  under `## Owed`.
- **The page does not exist on `main` yet.** It arrives with
  `row/DOCS-QUICKSTART-1281`. Until then the entry matches no changed path and
  the gate behaves exactly as it does today. This is deliberate. The gate repair
  lands first so the row it unblocks does not have to carry it.

## Tests

`tests/scripts/test_doc_checkpoint.py`, in `SupportSurfaces`:

1. `test_the_quickstart_page_is_a_landing_source`. `README.md` plus
   `docs/QUICKSTART.md` is accepted. **RED before** the change, with the exact
   `landing source` refusal.
2. `test_the_quickstart_page_permits_but_does_not_demand_readme`.
   `docs/QUICKSTART.md` on its own owes nothing. Green before and after. It pins
   that the entry adds permission and not an obligation.
3. `test_an_unrelated_document_never_licenses_readme_churn`. `README.md` plus
   `docs/BUILD.md`, `docs/ROCM.md` or `docs/RELEASES.md` is still refused. Green
   before and after. This is the property the change must not break, and it
   states it on ordinary documents, which is the class the quickstart page
   belongs to by path. The existing
   `test_a_coedited_projection_never_licenses_readme_churn` states the same
   property on a public projection and is also untouched.

## Gates

- `python3 tests/scripts/test_doc_checkpoint.py`
- `python3 scripts/check-doc-checkpoint.py --commit <head>` on this branch
- `scripts/agent-preflight.sh` and `scripts/agent-preflight.sh --staged`
- `documentation-checkpoint` and `agent-record` in CI

## Owed

- The permission is document-wide and not claim-scoped. Any landing source
  licenses any README claim change. Owned by this row, and the set has carried
  the property since it was written.
- No gate asserts that every member of `LANDING_SOURCE_FILES` names a path that
  exists. A renamed or deleted member fails silently toward refusing a README
  change. Owned by this row.

## Stop conditions

Stop and report if adding the entry turns any existing test in
`tests/scripts/test_doc_checkpoint.py` green by absence rather than by intent,
or if a case proving that an unrelated document cannot license README churn
stops failing under mutation. Either result means the set is the wrong place for
the repair and the trigger needs a claim-level model instead.
