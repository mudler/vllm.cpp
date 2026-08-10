# GitHub Pages docs site — Implementation Record

**Status:** built. Five tasks, all landed. The row stays `READY` — not because
the work is unstarted, but because the lifecycle move itself is blocked; see
delta 11 and [Remaining](#remaining).

**Goal:** Publish the 11 `docs/*.md` as a browsable site at
`https://mudler.github.io/vllm.cpp/` without creating a second copy of the prose.

Spec: [`.agents/specs/gh-pages-docs-site.md`](../../../.agents/specs/gh-pages-docs-site.md).
Row `ENG-DOCS-SITE`. Issue [#224](https://github.com/mudler/vllm.cpp/issues/224).

> **Why this document is short.** It was originally a 1289-line build plan whose
> bulk was the full text of every template, stylesheet and checker. Those files
> now exist in the tree. Keeping a second copy of them here would be precisely
> the drift the spec exists to prevent — *anything that must be kept in sync is
> the defect* — so the code blocks are gone and this is now a record of what was
> built, what changed on contact with reality, and what is left.

## Architecture as built

A Hugo site at `website/` mounts `../docs` read-only through `module.mounts`.
The mounted files have no front matter, so everything front matter would carry
is derived:

| Normally front matter | Derived from | File |
|---|---|---|
| `title:` | the file's first `# H1` | `layouts/partials/title.html` |
| `weight:` | position in a data file | `data/nav.yaml` |
| link fixups | a Goldmark render hook | `layouts/_default/_markup/render-link.html` |

**No file under `docs/` is modified, moved, renamed, or given front matter.**
The only file this branch adds under `docs/` is this record.

Design direction is a *technical broadsheet*: prose gets a 68ch measure while
tables, code and figures break out and run the full column. That inversion is
the single most important layout decision, because these documents are majority
table and the tables are the content.

## Tasks

- [x] **W1 — Teach the size gate the path.** `scripts/check-pr-size.py` +
  `tests/scripts/test_check_pr_size.py`. `website/**` → `public_document`.
  Landed alone and first, because `classify_path` fails closed and nothing else
  could be committed until it knew the path. RED-before verified.
- [x] **W2 — The mount and the skeleton.** `website/hugo.toml`,
  `content/_index.md`, `data/nav.yaml`, `layouts/_default/baseof.html`,
  `index.html`, `list.html`, `partials/head.html`.
- [x] **W3 — Titles and links.** `partials/title.html`,
  `_markup/render-link.html`, `_markup/render-heading.html`.
- [x] **W4 — The guard.** `scripts/check-site.py` + six mutation tests, wired
  into `ci.yml`.
- [x] **W5 — Style, search, deploy.** `assets/css/site.css`,
  `assets/css/chroma.css`, `layouts/_default/index.json`, `static/`,
  `website/README.md`, `.github/workflows/gh-pages.yml`.

## Deltas from the plan as written

Every one of these was found by building and running the thing, not by reading
it. Each is a trap for the next person.

| # | What happened | Resolution |
|---|---|---|
| 1 | **A one-shot `hugo` build is not proof the site works.** `hugo --minify` was green while `hugo server` crashed: the server renders page kinds with no source file (`/tags`, `/categories`, `/docs`), where `.File.ContentBaseName` nil-dereferenced | title partial guards `.File`; unused kinds disabled in `hugo.toml` |
| 2 | **`/docs/` rendered blank.** Hugo generates the section page whether or not anything links to it. This also surfaced delta 1 — before it existed those kinds had no layout and were never rendered | added `layouts/_default/list.html` |
| 3 | **Code blocks had no syntax highlighting at all.** `noClasses = false` emits class names and no colours; nothing shipped the classes, and the build said nothing | `assets/css/chroma.css`, generated for both schemes |
| 4 | **The "broken tables" were not a parsing bug.** Goldmark parses every table correctly — separator counts match rendered `<table>` counts in every document (BENCHMARKS 16, FEATURES 15, USAGE 11, ENVIRONMENT 5, STATUS 1). A 46rem prose measure plus `display:block` did the damage | the broadsheet inversion, tabular figures, sticky header, per-table scroll box |
| 5 | **Webfonts could not ship.** `check-pr-size.py` refuses binary changes outright, and a waiver is for one-time migrations with an expiry, not a permanent asset | stacks only: a serif for display, the mono for labels and table headers; zero third-party requests |
| 6 | **Logos were duplicated into `website/static/`** — the very copy this design forbids, applied to artwork | mounted from `assets/` with `includeFiles` |
| 7 | **Both logos rendered, stacked.** `.sidebar .brand img` (0,2,1) outranked a bare `.logo-dark { display: none }` (0,1,0) | matched specificity; `display` kept out of the shared sizing rule |
| 8 | TOC threshold 4 -> 5 sections: at 4, `RELEASES.md` (68 lines) got furniture | `single.html` |
| 9 | Home page opened by explaining its own build mechanism | rewritten around MANIFESTO.md's argument, condensed and linked rather than copied |

**10. The README pointer could not land.** `README.md` was already 29,990 of the
30,000-character budget `check-readme-structure.py` enforces; a 95-character link
block takes it to 30,085. Freeing space means cutting landing-page prose, which
is the maintainer's editorial call. **Still owed.**

**11. The row could not move to `GATING`.** `check-doc-checkpoint.py` requires a
lifecycle move to update `docs/STATUS.md`, `docs/BENCHMARKS.md` and
`.agents/NOW.md` in the same change — correctly, a state change is a claim. But
all three sit at their shrink-only ratchets: at this branch's base `STATUS.md`
measures **exactly** its 243,588-character ratchet (zero headroom) and `NOW.md`
is 5,986 of 6,000, so a one-line note in either fails the gate. The move
therefore requires collapsing unrelated superseded narrative first — which the
checker prescribes, but which means editing other people's prose from inside a
docs-site change. Left as a separate deliberate edit. **This is not specific to
this row:** any lifecycle move is blocked the same way until STATUS has room.

## Verification

Run from the repository root:

```sh
hugo --minify -s website          # 14 pages, no warnings
python3 scripts/check-site.py     # site OK: 11 published docs, nav in bijection
python3 tests/scripts/test_check_site.py
python3 tests/scripts/test_check_pr_size.py
scripts/agent-preflight.sh --staged
```

Observed on this branch:

| Check | Result |
|---|---|
| Pages emitted | 12 `index.html` (home + `/docs/` + 11 docs) + `index.json` |
| `docs/bench-evidence`, `docs/superpowers` in `public/` | absent |
| `href` ending in `.md` anywhere in `public/` | none |
| Protocol links rewritten in `docs/status/` | 48 to the GitHub blob base |
| Title derivation | `docs/kv-offload/` titles "KV offload and external KV caches" |
| Search index | 11 pages |
| `git status --porcelain docs/` | only this record |

The deploy workflow re-asserts the middle three on every run, because each is a
silent-failure mode: a render hook that stops matching emits the raw
destination rather than erroring, and a Hugo bump past 0.153 would ignore
`excludeFiles` and publish both excluded trees.

## Remaining

- [ ] **README pointer** (delta 7). Needs ~95 characters freed from `README.md`,
  or an explicit budget change. Maintainer's call.
- [ ] **Enable GitHub Pages** on `mudler/vllm.cpp` with the source set to
  GitHub Actions. A repository setting, not a file; the workflow is inert
  without it. This is the spec's stop condition and the reason the row is
  `GATING` rather than `DONE`.
- [ ] **Free room on the public surfaces, then move the row** (delta 11).
  `docs/STATUS.md` needs space before `READY -> GATING -> DONE` can be recorded
  at all; the engine-matrix summary counts move in the same edit, and the spec
  gains its `## Outcome` section when the page resolves.

## Notes for whoever touches this next

- **Never edit a file under `docs/` to make the site render better.** The whole
  design rests on the mount being read-only. If a document genuinely cannot
  render, that is a `NEEDS_DECISION`, not a quick fix.
- **`relURL` everywhere.** The site is published under `/vllm.cpp/`, not a
  domain root; a hardcoded `/docs/status/` works locally and 404s in production.
- **Check `hugo server`, not just `hugo`.** See delta 1.
- **Adding a document to `docs/` means adding it to `data/nav.yaml`.**
  `check-site.py` fails the build otherwise, which is the point.
