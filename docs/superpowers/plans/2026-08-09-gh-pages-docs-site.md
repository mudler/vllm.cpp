# GitHub Pages docs site — Implementation Record

**Status:** built. Five tasks, all landed. The row stays `READY` — not because
the work is unstarted, but because the lifecycle move itself is blocked; see
delta 9 and [Remaining](#remaining).

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

Everything below was discovered by building and running the thing, not by
reading. Recorded because each one is a trap for the next person.

**1. A one-shot `hugo` build is not proof the site works.** `hugo --minify` was
green while `hugo server` crashed. The server renders page kinds the one-shot
build had no layout for — `/tags`, `/categories`, `/docs` — and those are
generated pages with **no source file**, so `.File.ContentBaseName` in the title
partial nil-dereferenced and took the whole build down. The partial now guards
`.File`; `hugo.toml` disables `taxonomy`, `term`, `rss` and `sitemap`, none of
which this site has any content for.

**2. `/docs/` rendered blank.** Hugo generates the section page whether or not
anything links to it, so trimming the address bar landed on an empty document.
Added `layouts/_default/list.html`. This is also what surfaced delta 1 — before
it existed, those kinds had no layout and were never rendered.

**3. Code blocks had no syntax highlighting at all.** `hugo.toml` sets
`noClasses = false`, which emits class names and no inline colours. Nothing
shipped those classes, so every code block rendered as flat text and the build
said nothing. `assets/css/chroma.css` is generated from
`hugo gen chromastyles` for both schemes; `website/README.md` records the
command.

**4. The stylesheet had to become a Hugo template.** Font `@font-face` URLs were
hardcoded to `/vllm.cpp/fonts/...`, which the plan's own global constraint
forbids. CSS inlined into `<style>` resolves relative URLs against the *page*,
not the stylesheet, so relative paths cannot work either. `site.css` is now run
through `resources.ExecuteAsTemplate` and uses `relURL`.

**5. The first stylesheet made the tables look broken, and the tables were
fine.** Worth stating precisely, because the instinct is to blame the parser:
Goldmark parses every table correctly — separator counts match rendered
`<table>` counts exactly in every document (BENCHMARKS 16, FEATURES 15, USAGE
11, ENVIRONMENT 5, STATUS 1). The damage was a 46rem prose measure plus
`display:block` applied to documents whose cells hold whole paragraphs. Hence
the broadsheet inversion above, plus tabular figures, a sticky header row, and a
scroll box that belongs to the table rather than the page.

**6. The TOC threshold moved from 4 sections to 5.** At 4, `RELEASES.md` — 68
lines — got a table of contents, which is furniture. STATUS has 11 sections and
needs one.

**7. The README pointer could not land.** `README.md` was already 29,990 of the
30,000-character budget `check-readme-structure.py` enforces; a 95-character
link block takes it to 30,085 and the gate refuses it. Freeing space means
cutting existing landing-page prose, which is an editorial decision for the
maintainer rather than something to do quietly. **Still owed** — see below.

**8. Fonts and palette came from the project, not from taste.** The logo
(`assets/logo.svg`) supplies the real brand colours — `#0f7f96` teal, `#3abbd2`
cyan, `#131a20` ink — which replaced the README badge colours guessed at in the
spec. Sora 700 and Geist Mono are self-hosted from the LocalAI site assets (both
SIL OFL, attributed in `website/README.md`) so the site makes no third-party
requests and works under a strict CSP.

**9. The row could not be moved to `GATING`, and the reason is worth recording.**
`check-doc-checkpoint.py` requires any lifecycle move to update `docs/STATUS.md`,
`docs/BENCHMARKS.md` and `.agents/NOW.md` in the same change — correctly, since a
state change is a claim about the project. But all three are at their
shrink-only ratchets right now: at this branch's base, `docs/STATUS.md` measures
**exactly** its 243,588-character ratchet (zero headroom, "this page may only
shrink") and `.agents/NOW.md` is 5,986 of its 6,000. A one-line note in either
one fails the gate. `check-public-doc-tables.py` also caps a BENCHMARKS cell at
220 characters.

So the move requires collapsing unrelated superseded narrative to buy space —
which the checker explicitly prescribes ("collapse the superseded narrative …
then lower the ratchet in the same change"), but which means editing other
people's prose from inside a docs-site change. That is a separate, deliberate
edit, not something to fold in here, so the row stays `READY` with its code and
evidence anchors filled in, and the move is listed below. **This is not specific
to this row:** any lifecycle move in the repository is blocked the same way
until STATUS has room.

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
- [ ] **Free room on the public surfaces, then move the row** (delta 9).
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
