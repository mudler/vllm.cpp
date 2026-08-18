# doc-checkpoint: key `feature_surface` off registrations, not paths

Issue: [#595](https://github.com/mudler/vllm.cpp/issues/595)
Row: none. This is a checker-semantics repair, not a roadmap item; #595 is
carried under `## Owed` below, which is the index's other admissible shape.

## Now

`IMPLEMENTING`. The trigger change and its tests are written; the gate evidence
below is captured on this branch.

## Scope

`scripts/check-doc-checkpoint.py:79` sets

```python
FEATURE_SURFACE_PREFIXES = ("src/vllm/model_executor/models/",)
```

so **any** edit to any file under that directory classifies the change as
`feature_surface` and demands a `docs/FEATURES.md` edit.

In scope: replace that path trigger with a content trigger — a change to the set
of `REGISTER_VLLM_MODEL(...)` registrations in the touched file.

Out of scope, and explicitly NOT fixed here:

- `FEATURE_SURFACE_FILES` (the four `.agents/*-matrix.md` records). Those are
  claim surfaces; editing one **is** a claim change and keeps its path trigger.
- `USER_USAGE_FILES` / `CMakeLists.txt`, which is the identical shape reported
  separately as [#515](https://github.com/mudler/vllm.cpp/issues/515).
- The lock that #595 names in full. A genuine new architecture still writes the
  shared `docs/FEATURES.md` table. This change removes the contention for every
  fix, refactor and port phase that changes no registration; it does not
  relocate the obligation to a per-row surface, which is #595's larger ask.

## Why

The checker's own docstring already argues this:

> A one-line compile fix owed three public-doc edits, so this gate produced 16
> of the last 20 red CI runs, and it had accreted SIX hardcoded exact-path-set
> escape hatches -- one per legitimate change it had blocked.
> [...] Editing src/ alone owes nothing.

The 2026-08-11 rewrite removed path classification for `src/`, `include/` and
`tests/` generally, but kept it for `models/`. So the failure the rewrite exists
to prevent still reproduces, restricted to model files.

Measured cost on 2026-08-16. `e34d71379` (#1054) is a one-line lambda-capture
change to `models/qwen3_5_weights.cpp` that alters no capability. The gate
demanded `docs/FEATURES.md`; the change answered with prose in `BENCHMARKS.md`,
`FEATURES.md` and `STATUS.md`; that prose crossed the `check-public-doc-tables`
paragraph budgets, which reds `main` **and** runs in the pre-push hook, so every
branch in the repository was blocked from pushing. That is
[#1055](https://github.com/mudler/vllm.cpp/issues/1055) (fixed by #1057),
re-filed by a second agent as
[#1062](https://github.com/mudler/vllm.cpp/issues/1062) with #1064 as a
duplicate fix PR. [#1058](https://github.com/mudler/vllm.cpp/issues/1058) is a
third #1054 fallout and remains open. The repair for the MSVC break the same
commit introduced (#1068, PR #1069) hit the identical demand and had to argue an
exception in its commit body.

Two shared-file gates in series, each individually defensible, took `main` down
and blocked every push.

## Design

`REGISTER_VLLM_MODEL(` is the registry's own entry point, and
`scripts/check-supported-models.py` already gates `docs/FEATURES.md` against
exactly that set, so the signal is authoritative and already load-bearing.

Add, mirroring the existing `measurement_changes` shape so the trigger is
content-based and reads through `blob()`:

```python
def registrations(text: str) -> set[str]
def registration_changes(paths, before, after) -> list[str]
```

`classify()` adds `feature_surface` when `registration_changes` is non-empty,
and no longer adds it from `FEATURE_SURFACE_PREFIXES`.

Polarity, stated because it is the risk: this NARROWS a gate. Adding an
architecture, removing one, and renaming one all still owe `docs/FEATURES.md`,
because all three change the registration set. A file added with a registration
reads as a change from the empty set, and a deleted file reads as a change to
it.

## Risks

- **Narrowing lets a real claim change through.** A capability change inside an
  already-registered model — a new quantized arm, a refusal that becomes a
  render — changes no registration and would no longer be demanded. Accepted:
  that class was never reliably caught either, since the gate could be satisfied
  by any `FEATURES.md` edit including an unrelated one, and `docs/FEATURES.md`
  row content is separately gated by `check-supported-models.py`. Recorded under
  `## Owed`.
- **Regex vs the real parser.** `REGISTER_VLLM_MODEL` inside a comment or a
  string would count. Accepted: the same false positive fails toward DEMANDING a
  doc edit, which is the safe direction.

## Tests

`tests/scripts/test_doc_checkpoint.py`, red before the implementation:

1. `test_editing_a_registered_model_owes_nothing` — a model file whose
   registration set is unchanged demands nothing. **RED before**, because the
   path trigger fires today.
2. `test_a_new_model_registration_owes_the_feature_surface` — adding a
   `REGISTER_VLLM_MODEL` without touching `docs/FEATURES.md` is refused.
3. `test_removing_a_registration_owes_the_feature_surface` — deleting one is
   refused.
4. `test_a_matrix_record_still_owes_the_feature_surface` — the
   `.agents/*-matrix.md` path trigger is untouched.

2, 3 and 4 are green before and after: they pin what must NOT be widened away.

## Gates

- `python3 tests/scripts/test_doc_checkpoint.py`
- `python3 scripts/check-doc-checkpoint.py --commit <head>` on this branch
- `documentation-checkpoint` and `agent-record` in CI

## Owed

- The capability-change-without-registration-change class above, which no gate
  now covers. Owned by this row; tracked on #595.
- #595 itself stays open. This change does not relocate the obligation to a
  per-row surface, which is what closes it.
- #515 is the identical shape for `CMakeLists.txt` -> `docs/USAGE.md` and is not
  touched here.

## Stop conditions

Stop and report if removing the path trigger makes any existing test in
`tests/scripts/test_doc_checkpoint.py` green-by-absence rather than by intent —
that is the "never make a red gate green by deleting an assertion" failure, and
it means the trigger needs narrowing rather than replacing.
