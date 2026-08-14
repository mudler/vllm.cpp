# GATE-PR-SIZE-BINARY — retire the fail-closed binary guard

**Row:** `GATE-PR-SIZE-BINARY`
**Issue:** [#615](https://github.com/mudler/vllm.cpp/issues/615)
**Base:** `origin/main` `7572b0f4e`
**Status:** ACTIVE, 2026-08-13

## 1. Scope

One behavioural change to `scripts/check-pr-size.py`: remove the error raised
for a changed path that git reports as binary.

**In scope.** The `change.lines is None` branch in `change_errors`, the two
places in that file that advertise it, the two cases in
`tests/scripts/test_check_pr_size.py` that pin it, and the three descriptions of
the guard that live outside it: the `pr-size` job comment in
`.github/workflows/ci.yml` and the font rationale in `website/README.md`.

The `SITE_ASSET` comment is deliberately **not** edited: it never described the
guard, and it reads correctly once the guard is gone.

**Out of scope.** Explicit path classification, the checker-evidence contract,
the role checks, and the retired line budget. None of them changes. This is not
a size rule and it does not reopen one — the per-class budgets were retired on
2026-08-10 by developer decision and stay retired.

## 2. Anchors

Local, not upstream — this is a project governance checker with no vLLM
counterpart.

| What | Where |
|---|---|
| The guard | `scripts/check-pr-size.py:480-481` |
| Its advertisement | `scripts/check-pr-size.py:56-57` |
| The classifier that already protects us | `scripts/check-pr-size.py` `classify_path`, raises `ValueError` on any unclassified path |
| The `asset` class the guard contradicts | `scripts/check-pr-size.py:160-165` (`SITE_ASSET`) |
| The guard's landing commit | `450a1b696`, 2026-08-10 |
| The golden precedent it post-dates | `971d55063`, 2026-08-09 |
| Blocked work | [#431](https://github.com/mudler/vllm.cpp/pull/431) |

## 3. Design

`change_errors` currently short-circuits on binaries before any class-specific
rule runs:

```python
if change.lines is None:
    errors.append(f"binary change {change.path!r} is not reviewable as text")
    continue
```

Delete the branch. Everything downstream already tolerates `lines is None` —
the checker-evidence contract tests `evidence_change.lines is None` explicitly
rather than assuming an int, so a binary simply cannot serve as mutation
evidence, which remains correct.

Classification runs *before* this branch and is unchanged, so the ordering after
the edit is: classify (raise on unknown) → class-specific rules. An unclassified
binary is still refused, by the classifier, with the message that names the real
defect — an unclassified path — instead of one that names an unfixable property
of the file.

**Why the guard is not load-bearing.** Its stated job is that a binary "is not
reviewable as text". True, and irrelevant: nothing else in this checker reviews
text either. It classifies paths and enforces an evidence contract. The property
that keeps an unreviewable blob out of the tree is that it must first earn a
class, and that check is the one being kept.

**Why not an exemption list instead.** An allowlist of blessed binary paths is a
shared must-write surface — every golden-bearing PR would edit it, which is
precisely the lock AGENTS.md forbids. Classification already partitions these
paths by *where they live*, which is the derived-at-read-time shape.

## 4. Risks and decisions

| Risk | Assessment |
|---|---|
| **The retirement is wider than the problem it solves** | Accepted, and the operator should merge knowing it. Goldens live under `tests/`, so they classify as `product` — permitting them necessarily permits binaries across `src/`, `scripts/`, `tools/` and `benchmarks/`, and the same delete admits them to `procedure`, `project_record`, `ci` and `vendored_dependency` too. Measured before/after, not inferred: an opaque blob at `src/vllm/blob.dat`, and binary bytes replacing `.agents/workflow.md` or `ci.yml`, were refused at BASE and pass at HEAD. What still refuses them: `.gitignore` eats the realistic accidental blob, every change arrives on a PR, and a binary in a diff is maximally visible to a reviewer. Judged an acceptable trade because the guard could not tell a golden from a rootkit, so keeping it meant blocking correctness evidence — which a correctness-first project should not do. |
| A narrower fix was available | True, and it is the honest limit of this change. A derived regex class for `tests/parity/goldens/…` would be the same derived-at-read-time shape this spec praises in `SITE_ASSET`, and would not admit binaries to `src/`. It was not taken because it leaves the `asset` and `website/static/` cases still refused and would need a second class the next time a lane needs a binary — but it is a real alternative, and §3's earlier appeal to the `SITE_ASSET` comment over-generalized: that comment speaks for `website/static/` only, not for every class. |
| This reads as weakening a gate to go green | It is a deliberate retirement, argued in the commit message per the no-waiver-registry rule, not a repair of a red run. No PR of mine is unblocked by it; the beneficiaries are #431 and future golden work. |
| Goldens become unreviewable in practice | Unchanged by this edit — they are unreviewable as text either way. Golden provenance is enforced by the parity gates and the oracle-identity requirements, which is where it belongs. |
| The retirement is silently reversed later | The RED-first test in §5 asserts the new behaviour directly, so a reintroduction turns it red. |

## 5. Tests

RED-first, in `tests/scripts/test_check_pr_size.py`:

1. `test_a_classified_binary_is_accepted` — a binary at a classified path
   (`tests/parity/goldens/.../our_ids.npy`, and a `website/static/` asset)
   produces **no** error. **RED before the change** for the intended reason:
   the guard fires.
2. `test_an_unclassified_binary_is_still_refused` — a binary at an unclassified
   path still errors, and the error names classification, not binaryness. This
   is the guard rail that keeps the retirement scoped. Green both before and
   after (the classifier raises first), so it is a regression pin, not evidence.
3. `test_retiring_the_budget_did_not_retire_the_other_contracts` **drops its
   binary clause entirely**, because cases 1 and 2 now carry that coverage
   directly and duplicating it inside a multi-contract test would hide which
   contract failed. Its classification and checker-evidence clauses are
   untouched and still bite. (Drafted as "rewrite the clause"; deleting it was
   the better shape once 1 and 2 existed, and this line records what shipped.)
4. Delete `test_binary_changes_fail_closed_instead_of_becoming_free`, which
   states the retired rule and cannot survive it.

## 6. Gates

- `python3 -m pytest tests/scripts/test_check_pr_size.py` green, with case 1
  shown RED on the unmodified checker first.
- `python3 scripts/check-pr-size.py --base <base> --head <head>` classifies this
  PR's own change without error.
- `scripts/agent-preflight.sh --staged` clean.
- The checker-evidence contract must be satisfied *by this very PR*: it changes
  a `governance_checker`, so it must ship executable mutation evidence in
  `tests/scripts/test_check_pr_size.py`. It does.

## 7. Evidence

**RED before**, on the unmodified checker, for the intended reason — all four
subtests of `test_a_classified_binary_is_accepted` die on the guard:

```
AssertionError: Lists differ:
  ["binary change 'website/static/fonts/sora-700.woff2' is not reviewable as text"] != []
4 failed, 43 passed, 119 subtests passed
```

`test_an_unclassified_binary_is_still_refused` was already green here, as §5
predicted — classification runs first, so it is a rail and not the evidence.

**GREEN after:** `43 passed, 123 subtests passed`. Wider governance suite
(`test_check_pr_size` + `test_agent_record`): `92 passed, 125 subtests`.

**The retirement does what it is for**, checked directly against a
golden-bearing change:

```
golden-bearing PR errors -> NONE (was: 2 refusals)
unclassified binary      -> ["unclassified repository path 'junk/blob.bin'"]
```

**The checker accepts its own diff**, satisfying the evidence contract it
enforces on `governance_checker` paths:
`check-pr-size.py --base 7572b0f4e --head <head>` → `OK`, exit 0.
`check-commit-trailers.py` → `OK: commit trailer contract`, exit 0.

**Stop condition §8 checked, not assumed.** `python3 -m pytest tests/scripts/
--ignore=tests/scripts/test_cpu_kernel_bench.py`, run sequentially on both
trees:

| tree | result |
|---|---|
| HEAD | `9 failed, 1276 passed, 3 skipped, 1510 subtests` |
| BASE `7572b0f4e` | `9 failed, 1275 passed, 3 skipped, 1506 subtests` |

The **failure sets are identical** — the same six `test_gen_vulkan_spirv` shader
subfailures, the same `test_check_windows_portability` subfailure, and the same
`test_mlx_system_headers` and `test_now_render` failures. All pre-existing; this
change adds no failure. The `+1 passed / +4 subtests` delta is exactly this
change: one test deleted, two added, four new subtests.

`test_cpu_kernel_bench.py` fails collection on unmodified main too (it wants a
built benchmark binary). `test_cpu_x86_llamacpp_floor` is order-dependent under
a loaded parallel run and passes sequentially on both trees.

An earlier draft of this section recorded `8 failed, 20 passed, 2 skipped` for
this check. That was a **three-file subset** mislabelled as the whole suite; no
stated command produced it. The substantive claim it was offered for — identical
failure sets before and after — is unchanged and is what the table above shows.

## 8. Stop conditions

- If removing the branch turns any other case in the suite red for a reason not
  named in §5, stop — that is a load-bearing use of the guard this spec did not
  find, and the design in §3 is wrong.
- If the checker cannot classify its own diff after the edit, stop.
- If a reviewer judges that classification alone does not carry the protection,
  stop and escalate rather than widening the change.
