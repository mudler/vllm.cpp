# The lane that renders the docs site installs the renderer

Identity: `GATE-CI-SITE-HUGO-LANE`

Issue: [#1754](https://github.com/mudler/vllm.cpp/issues/1754)

Related, separately owned:
[#1722](https://github.com/mudler/vllm.cpp/issues/1722) with
[PR #1726](https://github.com/mudler/vllm.cpp/pull/1726) (the developer-box
guard), and [#1764](https://github.com/mudler/vllm.cpp/issues/1764) (a
three-job umbrella that names this red as one of its three).

Stored-count drift lock this change arms:
[#1828](https://github.com/mudler/vllm.cpp/issues/1828). See `## Owed`.

## Now

`agent-record` is red on `main` and on every pull request cut from it.
`tests/scripts/test_check_site.py`'s
`test_rendered_benchmark_index_links_resolve_to_emitted_pages`, added by
[#1714](https://github.com/mudler/vllm.cpp/pull/1714) at `1db7e59cf`, shells out
to `hugo`. No job in `.github/workflows/ci.yml` installs it, so `subprocess.run`
raises `FileNotFoundError`, `unittest` records an ERROR, and the step exits 1.
The regression window is measured: the scheduled baseline at `deb53c6a3` was
green on this job and `08c81a892` and every scheduled run after it is red.

## Scope

Provision Hugo on the CI lane that runs the docs-site suite, and gate that
provisioning so it cannot be removed silently.

Out of scope: the contents of `tests/scripts/test_check_site.py`. This change
edits none of its lines, deliberately — see `## Design` and `## Owed`.

## The design decision, and why the obvious fix is rejected

The one-line repair is `shutil.which("hugo") or self.skipTest(...)`. It makes
the job green and it is what #1722 asks for, so it deserves a reason rather
than a preference.

An absent binary has two failure modes and neither of them is a verdict on this
repository:

| binary | guard | what CI reports | what was rendered |
|---|---|---|---|
| absent | none | ERROR, job red | nothing |
| absent | skip | `OK (skipped=1)`, job green | nothing |
| present | either | `ok`, job green | the whole site |

Only the third row is a measurement. The first two differ in how loudly they
say the same thing, and the second says it in the register this repository has
been bitten by before: a skip wearing a pass. A guard alone would retire the red
by arranging for the assertion #1714 was written for to run **nowhere** —
PR #1726's `agent-record` is already SUCCESS for exactly that reason, having
rendered no site.

So the guard is not the fix; it is the choice of which non-verdict a missing
binary produces. Installing the binary is the fix, and it is what
[#1754](https://github.com/mudler/vllm.cpp/issues/1754) names as option 2 and
calls, together with the guard, "the only combination where the test both cannot
error and actually runs". This change lands that half. #1726 lands the other,
and the two touch disjoint files so they compose without a conflict.

The escalation clause in the dispatch — a skip is acceptable only when paired
with a CI-side assertion that the lane did not skip — is therefore not reached.
Nothing here prevented the install.

## Design

Three edits to `.github/workflows/ci.yml`, all inside the `agent-record` job.

1. A job-level `HUGO_VERSION: "0.146.3"`, the same value
   `.github/workflows/gh-pages.yml` pins at workflow level, and a
   `peaceiris/actions-hugo@v3` step with `extended: true` — the same action, the
   same major, the same build as the job that publishes the site. Reusing the
   published lane's mechanism rather than inventing one is the point: a lane
   rendering on a different Hugo returns a verdict about a site nobody visits.
   The pin is not arbitrary. `website/hugo.toml` uses `excludeFiles`, which Hugo
   deprecates from 0.153 in favour of `files`; bumping past that without
   migrating the key publishes `docs/bench-evidence` and `docs/superpowers`.
2. `hugo version` as the first line of the step that runs the suite. It is the
   line that fails legibly, and early, if the setup step ever stops producing a
   binary.
3. Registration of the new guard suite on the same job.

One new file, `tests/scripts/test_ci_site_lane.py`, holds four invariants over
the workflow and two over its own non-vacuity:

- The lane installs Hugo, exactly once, **before** the step that renders.
- It asks for the extended build.
- Its pin equals `gh-pages.yml`'s, resolved through `${{ env.NAME }}` in both
  files rather than compared as template strings, and it is an exact
  `MAJOR.MINOR.PATCH` rather than a floating tag.
- The rendering step probes the binary before invoking the suite.

The invariants are derived from the workflow, not written as a list of job
names: whichever job runs `tests/scripts/test_check_site.py` is the job that
must provision Hugo, so a step that moves to another job stays covered.

Two of the six cases exist only to defeat vacuity, because that is the shape
this file would otherwise take. `test_the_resolver_finds_the_lane_it_is_about`
fails if the resolver stops seeing the suite, which every other case needs to be
true to mean anything, and the two loops over setup steps count what they saw
and fail on zero rather than passing an empty iteration.

## Risks

- **Job minutes.** `ci.yml:99` records `agent-record` as a 3.8-minute Ubuntu
  job. `peaceiris/actions-hugo` downloads one release archive; the site itself
  renders in 0.5 s locally, inside the suite that already runs. Accepted: the
  alternative is a 3.8-minute job that renders nothing.
- **A second copy of the version.** Two files carrying one pin is the drift
  shape this repository calls a lock. It is answered by holding them equal in a
  test rather than by a shared file that every pull request would have to write.
- **A network dependency on the lane.** The action fetches from GitHub releases.
  A fetch failure fails the step, which is a legible infrastructure red and not
  a silent pass.

## Tests

- `tests/scripts/test_ci_site_lane.py` — six cases, run by `agent-record`.
- `tests/scripts/test_check_site.py` — unchanged, and now actually executed on
  the lane rather than erroring on it.

## Gates

```sh
python3 tests/scripts/test_ci_site_lane.py
python3 scripts/check-site.py
python3 tests/scripts/test_check_site.py
scripts/agent-preflight.sh
```

## Evidence

Host: `mudler-ubuntu-box`, x86_64, Python 3.12.3, `hugo
v0.146.3+extended linux/amd64`. Base `d60692c89`.

**Red before, the reported failure.** `tests/scripts/test_check_site.py` at
sha256 `4ee0f66d…`, run with a `PATH` from which `/home/mudler/.local/bin` is
removed so `command -v hugo` reports nothing:

```
test_rendered_benchmark_index_links_resolve_to_emitted_pages ... ERROR
FileNotFoundError: [Errno 2] No such file or directory: 'hugo'
Ran 7 tests in 0.272s
FAILED (errors=1)
```

exit 1 — the CI text, reproduced locally.

**Red before, the new guard.** `tests/scripts/test_ci_site_lane.py` against a
scratch tree holding `origin/main`'s two workflow files (`ci.yml` at sha256
`f9f351e1…`, verified equal to `git show origin/main:.github/workflows/ci.yml`):
`Ran 6 tests`, `FAILED (failures=4)`, exit 1, naming
`agent-record: hugo setup steps at indices []` and `agent-record: the step runs
the site suite without probing hugo`.

**Green after.** On this tree:

- `tests/scripts/test_ci_site_lane.py -v`: `Ran 6 tests`, `OK`, exit 0, six
  `ok`, zero skips.
- `tests/scripts/test_check_site.py -v` with Hugo on `PATH`: `Ran 7 tests`,
  `OK`, exit 0, the case reporting `ok` and **not** `skipped`. Zero skips is the
  load-bearing half of that line.

**Mutations.** Each applied to the tree, proved applied by a diff, parsed or
compiled, then restored and re-verified by sha256 (`ci.yml`
`8f311a70…`, `test_check_site.py` `4ee0f66d…`).

| mutation | result |
|---|---|
| delete only the `peaceiris/actions-hugo@v3` step | `FAILED (failures=3)`: absent setup, and both vacuity floors fire |
| `HUGO_VERSION` `0.146.3` → `0.147.0` | `FAILED (failures=1)`: `'0.147.0' != '0.146.3'`, naming the `excludeFiles` consequence |
| `assertEqual(len(detail_hrefs), 10)` → `11`, Hugo present | `FAILED (failures=1)`: `AssertionError: 10 != 11` |

The third is the one that answers the question this change exists for. It fails
from inside the rendered site, which is only reachable when the binary is there,
so it proves the protected assertion **executed** rather than skipped.

## Owed

- [#1828](https://github.com/mudler/vllm.cpp/issues/1828) — `test_check_site.py`
  asserts a literal `10` rendered detail links, a stored count of
  `docs/benchmarks/*.md`. Until this change it was inert in CI, because the case
  never reached the line; installing Hugo arms it. Measured on `d60692c89`: 10
  slugs, 10 table hrefs, no duplicate target, no unlinked slug — a bijection,
  and therefore derivable at read time. Left out of scope because deriving it
  changes what the case asserts, which `AGENTS.md` routes through the normal
  row, spec and fresh-review path rather than the in-flow rule, and because
  `tests/scripts/test_check_site.py` has an open pull request against the same
  case.

## Stop conditions

- Stop and escalate if `peaceiris/actions-hugo@v3` cannot install 0.146.3 on
  `ubuntu-latest`. The fallback is not a skip; it is a decision about which lane
  renders the site.
- Stop if the pin has to move. Moving it past 0.153 is a `website/hugo.toml`
  migration and belongs to `ENG-DOCS-SITE`, not here.
