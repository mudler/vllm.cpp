# GitHub Pages docs site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the 11 `docs/*.md` as a browsable site at `https://mudler.github.io/vllm.cpp/` without creating a second copy of the prose.

**Architecture:** A Hugo site at `website/` mounts `../docs` read-only through `module.mounts`. Because the mounted files have no front matter, three small layout mechanisms replace it: the page title is extracted from each file's first `# H1`, the sidebar order comes from `website/data/nav.yaml`, and a Goldmark render hook rewrites `.md` links — internal ones to site URLs, `../` escapes to GitHub blob URLs. A `scripts/check-site.py` guard holds the two content invariants the site silently depends on.

**Tech Stack:** Hugo 0.146.3 extended (no theme, no submodule, no Go module, no npm), GitHub Actions, Python 3 for the checker.

Spec: [`.agents/specs/gh-pages-docs-site.md`](../../../.agents/specs/gh-pages-docs-site.md). Row `ENG-DOCS-SITE`. Issue [#224](https://github.com/mudler/vllm.cpp/issues/224).

## Global Constraints

- **Never modify, move, rename, or add front matter to any file under `docs/`.** The only permitted `docs/` change in this whole plan is adding this plan file itself. `git diff --stat origin/main -- docs/` must show nothing else.
- **Hugo is pinned to `0.146.3` extended**, matching the locally installed binary and LocalAI's CI. Do not use features added after it. At this version `excludeFiles` is the correct mount key (`files` replaces it only from 0.153).
- **No theme, no git submodule, no Go module, no npm.** Everything is layouts and CSS in-repo.
- **Every commit** carries a bare `FOLLOWING_AGENTS_PROTOCOL` line plus the trailers `Following-Agents-Protocol: true`, `AI-Assisted: true`, `Assisted-by: Claude-Code:claude-opus-5 [ClaudeCode]`. Never `Co-Authored-By` or `Signed-off-by`.
- **Run `scripts/agent-preflight.sh --staged` before every commit.** Never weaken a checker to make it pass.
- The published base URL is `https://mudler.github.io/vllm.cpp/`, so every site-internal href must carry the `/vllm.cpp/` prefix. Always produce it with `relURL`, never hardcode it.
- The GitHub blob base for protocol links is `https://github.com/mudler/vllm.cpp/blob/main/`.

## File Structure

| File | Responsibility |
|---|---|
| `website/hugo.toml` | Site config: baseURL, mounts of `../docs`, goldmark, output formats |
| `website/data/nav.yaml` | The one place sidebar order and labels are declared |
| `website/content/_index.md` | Home page prose |
| `website/layouts/_default/baseof.html` | HTML skeleton shared by every page |
| `website/layouts/_default/single.html` | A doc page: sidebar + article |
| `website/layouts/index.html` | Home: sidebar + intro + card list |
| `website/layouts/partials/title.html` | Derives a page title from its first `# H1` |
| `website/layouts/partials/sidebar.html` | Renders `nav.yaml` into links |
| `website/layouts/partials/head.html` | `<head>`: meta, title, inlined CSS |
| `website/layouts/_default/_markup/render-link.html` | Link rewriting (both rules) |
| `website/layouts/_default/index.json` | Search index (title + headings per page) |
| `website/assets/css/site.css` | The entire visual design |
| `website/static/favicon.png` | Copied from `assets/favicon.png` |
| `website/README.md` | Local preview, the Hugo pin, the dormant CNAME path |
| `scripts/check-site.py` | The two content invariants |
| `tests/scripts/test_check_site.py` | Mutation tests for the guard |
| `.github/workflows/gh-pages.yml` | Build on PR, build + deploy on `main` |

Modified: `scripts/check-pr-size.py` (classify `website/**`), `tests/scripts/test_check_pr_size.py` (its test), `.github/workflows/ci.yml` (run the guard), `README.md` (one pointer line).

---

### Task 1: Teach the size gate about `website/`

`classify_path` fails closed. Until it knows `website/`, no later task can even be committed through the PR gate, so this lands first and alone.

**Files:**
- Modify: `scripts/check-pr-size.py` (the `PUBLIC_DOCUMENT_FILES`/`DOC` block near line 144, and `classify_path` near line 354)
- Test: `tests/scripts/test_check_pr_size.py:74` (`test_each_mutable_surface_has_an_explicit_class`)

**Interfaces:**
- Consumes: nothing.
- Produces: `classify_path(path: str) -> str` returns `"public_document"` for any path under `website/`.

- [ ] **Step 1: Write the failing test**

In `tests/scripts/test_check_pr_size.py`, inside `test_each_mutable_surface_has_an_explicit_class`, add these three entries to the `expected` dict (after the `"docs/STATUS.md": "public_document",` line):

```python
            "website/hugo.toml": "public_document",
            "website/layouts/_default/baseof.html": "public_document",
            "website/assets/css/site.css": "public_document",
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/scripts/test_check_pr_size.py`
Expected: FAIL. Three subtests error with `ValueError: unclassified repository path 'website/hugo.toml'` (and the other two paths) — the checker raises rather than returning a class.

- [ ] **Step 3: Write the minimal implementation**

In `scripts/check-pr-size.py`, add the regex next to the other compiled patterns (immediately after the `DOC = re.compile(...)` line):

```python
# The published documentation site. Its layouts, CSS and config are prose and
# presentation for a PUBLIC surface, reviewed the way the documents themselves
# are -- not product code, and not CI. `website/**` is a single class on
# purpose: splitting layouts from config would let a large redesign hide half
# its diff in the cheaper bucket.
SITE = re.compile(r"website/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
```

Then in `classify_path`, extend the existing `public_document` branch (currently `if path in PUBLIC_DOCUMENT_FILES or DOC.fullmatch(path):`) to:

```python
    if path in PUBLIC_DOCUMENT_FILES or DOC.fullmatch(path) or SITE.fullmatch(path):
        return "public_document"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/scripts/test_check_pr_size.py`
Expected: PASS, all tests.

Then confirm no neighbouring class was widened:

Run: `python3 scripts/check-pr-size.py --help`
Expected: exits 0 and prints usage (the module imports cleanly).

- [ ] **Step 5: Commit**

```bash
scripts/agent-preflight.sh --staged
git add scripts/check-pr-size.py tests/scripts/test_check_pr_size.py
git commit -F - <<'EOF'
chore(ENG-DOCS-SITE): classify website/** as public_document (#224)

`classify_path` fails closed, so every path in a PR must map to exactly one
review budget. `website/**` maps to none, and the checker raises
`ValueError: unclassified repository path 'website/hugo.toml'` — which means the
docs-site PR cannot pass the project's own size gate until the classifier learns
the path. This lands first and alone for that reason.

`public_document` (budget 2500) is the honest class: the site's layouts, CSS and
config are prose and presentation for a PUBLIC surface, reviewed the way the
documents themselves are. Not product code, and not CI — the workflow that
publishes the site keeps its own `ci` class.

One class for the whole directory, deliberately: splitting layouts from config
would let a large redesign hide half its diff in the cheaper bucket, which is
the "no blanket directory exemption" failure AGENTS.md names.

Verified: RED first — the three new expectations in
`test_each_mutable_surface_has_an_explicit_class` raise `ValueError` before the
change; green after. No existing assertion touched, no other class widened.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Claude-Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 2: A site skeleton that builds and excludes what it must

**Files:**
- Create: `website/hugo.toml`, `website/content/_index.md`, `website/data/nav.yaml`
- Create: `website/layouts/_default/baseof.html`, `website/layouts/_default/single.html`, `website/layouts/index.html`, `website/layouts/partials/head.html`
- Create: `website/.gitignore`

**Interfaces:**
- Consumes: Task 1's classifier (so this can be committed).
- Produces: a buildable Hugo site whose mounted pages live at `/docs/<lowercased-basename>/`. `website/data/nav.yaml` is a list under key `docs`, each entry `{file: "USAGE.md", label: "Using vllm.cpp"}` — Tasks 3, 4 and 5 all read this shape.

- [ ] **Step 1: Write the config**

Create `website/hugo.toml`:

```toml
# The documentation site. It owns NO content: `docs/` is mounted read-only
# below, so the markdown on this site and the markdown in the repository are
# the same bytes. Adding a copy under content/ would create a second surface
# free to drift from the protocol-governed originals -- see
# .agents/specs/gh-pages-docs-site.md.
#
# baseURL is overridden by CI with the value actions/configure-pages reports,
# so this literal only affects local builds.
baseURL = 'https://mudler.github.io/vllm.cpp/'
languageCode = 'en-GB'
defaultContentLanguage = 'en'
title = 'vllm.cpp'
enableEmoji = true

# Hugo 0.146.3. `excludeFiles` is the correct key at this version; it is
# deprecated from 0.153 in favour of `files`. A Hugo bump past that MUST
# migrate this or the excluded trees below start publishing.
[[module.mounts]]
source = "content"
target = "content"

[[module.mounts]]
source = "../docs"
target = "content/docs"
excludeFiles = ["bench-evidence/**", "superpowers/**"]

[[module.mounts]]
source = "assets"
target = "assets"

[[module.mounts]]
source = "static"
target = "static"

[params]
  description = 'vllm.cpp — a from-scratch C++20 inference engine, gated token-for-token against vLLM.'
  repo = 'https://github.com/mudler/vllm.cpp'
  # Where `../`-escaping links in docs/ are sent. See layouts/_default/_markup/render-link.html.
  blobBase = 'https://github.com/mudler/vllm.cpp/blob/main/'

[markup]
  [markup.tableOfContents]
    startLevel = 2
    endLevel = 3
  [markup.goldmark.renderer]
    unsafe = true
  [markup.highlight]
    noClasses = false
    lineNos = false

[outputs]
  home = ['html', 'json']
  page = ['html']
```

- [ ] **Step 2: Write the nav data and home content**

Create `website/data/nav.yaml`. The order is reading order, not alphabetical — alphabetical would open the site on `BENCHMARKS.md`:

```yaml
# The sidebar. This is the ONLY place doc order and display labels are
# declared: the mounted files carry no front matter, so there is no `weight:`
# or `title:` to read. scripts/check-site.py asserts this list and the mounted
# set are in bijection, so a new doc cannot go missing from the sidebar and a
# deleted one cannot linger here.
docs:
  - file: USAGE.md
    label: Usage
  - file: BUILD.md
    label: Building
  - file: FEATURES.md
    label: Features
  - file: BENCHMARKS.md
    label: Benchmarks
  - file: STATUS.md
    label: Status
  - file: ENVIRONMENT.md
    label: Environment variables
  - file: RELEASES.md
    label: Binary releases
  - file: SPECULATIVE-DECODING.md
    label: Speculative decoding
  - file: KV-OFFLOAD.md
    label: KV offload
  - file: SGLANG-COMPAT.md
    label: SGLang compatibility
  - file: ROCM.md
    label: ROCm backend
```

Create `website/content/_index.md`:

```markdown
---
title: Documentation
---

**vllm.cpp** is a from-scratch C++20 inference engine with no Python and no
PyTorch at inference time. Every architecture is gated token-for-token against a
pinned vLLM oracle, so "grounded in vLLM" is a test result rather than a design
claim.

These pages are the repository's own `docs/` directory, rendered. They are the
same bytes you get from a checkout — nothing here is a copy.
```

Create `website/.gitignore`:

```gitignore
public/
resources/
.hugo_build.lock
```

- [ ] **Step 3: Write the minimal layouts**

Create `website/layouts/partials/head.html`:

```html
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ partial "title.html" . }} · {{ site.Title }}</title>
<meta name="description" content="{{ site.Params.description }}">
<link rel="icon" href="{{ "favicon.png" | relURL }}">
{{ with resources.Get "css/site.css" }}
<style>{{ .Content | safeCSS }}</style>
{{ end }}
```

Create `website/layouts/_default/baseof.html`:

```html
<!doctype html>
<html lang="en">
<head>{{ partial "head.html" . }}</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <div class="shell">
    {{ partial "sidebar.html" . }}
    <main id="main">{{ block "main" . }}{{ end }}</main>
  </div>
</body>
</html>
```

Create `website/layouts/_default/single.html`:

```html
{{ define "main" }}
<article class="doc">
  {{ .Content }}
</article>
{{ end }}
```

Create `website/layouts/index.html`:

```html
{{ define "main" }}
<article class="doc">
  <h1>vllm.cpp documentation</h1>
  {{ .Content }}
  <ul class="cards">
    {{ range site.Data.nav.docs }}
      {{ $slug := lower (strings.TrimSuffix ".md" .file) }}
      <li><a href="{{ printf "docs/%s/" $slug | relURL }}">{{ .label }}</a></li>
    {{ end }}
  </ul>
</article>
{{ end }}
```

Create a placeholder `website/layouts/partials/title.html` so the build works before Task 3 replaces it:

```html
{{- if .IsHome -}}Documentation{{- else -}}{{ .File.ContentBaseName }}{{- end -}}
```

Create a placeholder `website/layouts/partials/sidebar.html`:

```html
<nav class="sidebar">
  <a class="brand" href="{{ "" | relURL }}">vllm.cpp</a>
  <ul>
    {{ range site.Data.nav.docs }}
      {{ $slug := lower (strings.TrimSuffix ".md" .file) }}
      <li><a href="{{ printf "docs/%s/" $slug | relURL }}">{{ .label }}</a></li>
    {{ end }}
  </ul>
</nav>
```

Create an empty `website/assets/css/site.css` (Task 5 fills it) and copy the favicon:

```bash
mkdir -p website/assets/css website/static
: > website/assets/css/site.css
cp assets/favicon.png website/static/favicon.png
```

- [ ] **Step 4: Build and verify the page set exactly**

Run:

```bash
hugo --minify -s website
find website/public -name index.html | sort
```

Expected: exactly 12 `index.html` files — `website/public/index.html` plus one per doc under `website/public/docs/<slug>/`, with slugs `benchmarks build environment features kv-offload releases rocm sglang-compat speculative-decoding status usage`.

Now verify the exclusions actually held — this is the check that keeps 18 benchmark logs and 36 specs off the public internet:

```bash
test ! -e website/public/docs/bench-evidence && echo "bench-evidence EXCLUDED ok"
test ! -e website/public/docs/superpowers && echo "superpowers EXCLUDED ok"
```

Expected: both lines print. If either path exists, stop — do not continue, and do not "fix" it by deleting the directory after the build.

- [ ] **Step 5: Commit**

```bash
scripts/agent-preflight.sh --staged
git add website
git commit -F - <<'EOF'
feat(ENG-DOCS-SITE): Hugo site that mounts docs/ read-only (#224)

The site owns no content. `[[module.mounts]] source = "../docs"` puts the
repository's own markdown into Hugo's content tree without copying it, so the
prose on the site and the prose in the checkout are the same bytes and cannot
drift. No file under docs/ is touched.

`excludeFiles` keeps `bench-evidence/` (18 raw benchmark logs) and
`superpowers/` (36 specs and plans) off the published site; the build is
verified against both paths rather than trusted. That key is deprecated from
Hugo 0.153 in favour of `files`, which is why the version is pinned and the
migration is written down in the config next to it.

Sidebar order lives in data/nav.yaml because the mounted files have no front
matter to carry a `weight:`. Reading order, not alphabetical — alphabetical
would open the site on BENCHMARKS.md.

Titles and link rewriting are placeholders here and land next; this commit is
the skeleton that builds.

Verified: `hugo --minify -s website` emits exactly 12 pages (home + 11 docs);
`website/public/docs/bench-evidence` and `.../superpowers` do not exist.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Claude-Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 3: Titles from the first H1, and link rewriting

Both mechanisms replace front matter, and a site with one but not the other is not worth landing: titles alone leave 139 dead links, links alone leave every browser tab reading `STATUS`.

**Files:**
- Modify: `website/layouts/partials/title.html`
- Create: `website/layouts/_default/_markup/render-link.html`

**Interfaces:**
- Consumes: Task 2's site; `site.Params.blobBase`.
- Produces: nothing later tasks call directly. Task 4's guard enforces the `# H1` precondition this relies on.

- [ ] **Step 1: Write the title partial**

Replace `website/layouts/partials/title.html` entirely:

```html
{{- /*
  The mounted docs carry no front matter, so `.Title` is EMPTY for every one of
  them. Hugo has no "use the first heading" mode, so take it from the content.
  All 11 docs open with an `# H1` and scripts/check-site.py keeps it that way;
  the ContentBaseName fallback is a belt-and-braces path that should never fire.
*/ -}}
{{- if .IsHome -}}
  Documentation
{{- else -}}
  {{- $h1 := index (findRE `(?m)^#\s+(.+)$` .RawContent 1) 0 -}}
  {{- with $h1 -}}
    {{- strings.TrimSpace (strings.TrimPrefix "#" .) -}}
  {{- else -}}
    {{- $.File.ContentBaseName -}}
  {{- end -}}
{{- end -}}
```

- [ ] **Step 2: Write the link render hook**

Create `website/layouts/_default/_markup/render-link.html`:

```html
{{- /*
  Two rules, because docs/*.md links point at two different worlds.

  1. A sibling doc (`BENCHMARKS.md`, `./USAGE.md`) is on this site -> rewrite to
     the pretty URL Hugo generated for it, /vllm.cpp/docs/benchmarks/.
  2. Anything escaping docs/ (`../.agents/**`, `../AGENTS.md`, `../include/**`)
     is NOT published -- the protocol tree is deliberately absent -- so send it
     to the same file on GitHub instead of 404ing. There were 139 such links at
     the time of writing and the count moves with every commit, which is why
     this is a mechanical rule rather than a list.

  Anchors are preserved in both cases. Absolute URLs and bare `#anchor` links
  pass through untouched.
*/ -}}
{{- $dst := .Destination -}}
{{- $out := $dst -}}
{{- $parsed := urls.Parse $dst -}}
{{- if and (not $parsed.Scheme) (not (strings.HasPrefix $dst "#")) -}}
  {{- $frag := "" -}}
  {{- $path := $dst -}}
  {{- if strings.Contains $dst "#" -}}
    {{- $parts := split $dst "#" -}}
    {{- $path = index $parts 0 -}}
    {{- $frag = printf "#%s" (index $parts 1) -}}
  {{- end -}}
  {{- if strings.HasPrefix $path "../" -}}
    {{- $out = printf "%s%s%s" site.Params.blobBase (strings.TrimPrefix "../" $path) $frag -}}
  {{- else if strings.HasSuffix $path ".md" -}}
    {{- $base := path.Base $path -}}
    {{- $slug := lower (strings.TrimSuffix ".md" $base) -}}
    {{- $out = printf "%s%s" (printf "docs/%s/" $slug | relURL) $frag -}}
  {{- end -}}
{{- end -}}
<a href="{{ $out }}"{{ with .Title }} title="{{ . }}"{{ end }}{{ if strings.HasPrefix $out "http" }} rel="noopener"{{ end }}>{{ .Text | safeHTML }}</a>
```

- [ ] **Step 3: Build and verify both rules on real pages**

Run:

```bash
rm -rf website/public && hugo --minify -s website
```

Verify no site page still links to a `.md` file — the whole point of rule 1 and 2:

```bash
grep -roh 'href="[^"]*\.md[^"]*"' website/public --include=index.html | sort -u
```

Expected: no output. Any hit is an unrewritten link.

Verify rule 2 sent the largest cluster to GitHub (`docs/STATUS.md` alone holds 70 of these):

```bash
grep -c 'github.com/mudler/vllm.cpp/blob/main/.agents/' website/public/docs/status/index.html
```

Expected: a number well above 50.

Verify rule 1 produced a site-internal link with the `/vllm.cpp/` prefix:

```bash
grep -o 'href="/vllm.cpp/docs/[a-z-]*/"' website/public/docs/benchmarks/index.html | head -3
```

Expected: at least one match.

Verify the titles are real prose rather than filenames:

```bash
grep -o '<title>[^<]*</title>' website/public/docs/kv-offload/index.html
```

Expected: `<title>KV offload and external KV caches · vllm.cpp</title>` — the file's own H1, not `KV-OFFLOAD`.

- [ ] **Step 4: Confirm docs/ is still untouched**

Run: `git status --porcelain docs/`
Expected: no output at all. If anything appears, revert it — the mount is read-only by design.

- [ ] **Step 5: Commit**

```bash
scripts/agent-preflight.sh --staged
git add website/layouts
git commit -F - <<'EOF'
feat(ENG-DOCS-SITE): derive titles from H1 and rewrite both link classes (#224)

Front-matter-less pages render with an EMPTY `.Title` — measured, not assumed:
the scratch mount build emitted `/docs/status/ :: ` with nothing after the
separator. Hugo has no "use the first heading" mode, so the title partial takes
the first `# H1` out of `.RawContent`. All 11 docs comply today and the next
commit makes that a gate rather than a convention.

The link hook handles the two worlds docs/*.md point at. Sibling docs become
pretty site URLs. Everything escaping docs/ — `../.agents/**`, `../AGENTS.md`,
`../include/**` — goes to the same file on GitHub, because the protocol tree is
deliberately not published and those links would otherwise 404. There were 139
of them at this commit and 135 ten commits earlier, which is exactly why the
rule is mechanical instead of a list of fixes.

Anchors survive both rewrites; absolute URLs and bare `#anchor` links are
untouched.

Verified: no `href` anywhere in public/ ends in `.md`; docs/status/ carries 70+
links to the `.agents/` blob base; docs/benchmarks/ carries site-internal
`/vllm.cpp/docs/...` hrefs; docs/kv-offload/ titles as "KV offload and external
KV caches" rather than "KV-OFFLOAD". `git status docs/` clean.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Claude-Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 4: The guard

The site rests on two facts about content that no existing checker holds. Both are true today by convention, and a convention that a build silently depends on is a latent breakage.

**Files:**
- Create: `scripts/check-site.py`, `tests/scripts/test_check_site.py`
- Modify: `.github/workflows/ci.yml` (after the `Every production env var is documented or classified` step, around line 84)

**Interfaces:**
- Consumes: `website/data/nav.yaml` in the shape Task 2 defined.
- Produces: `scripts/check-site.py` exits 0 when clean, 1 with one `ERROR:` line per violation on stderr.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_check_site.py`:

```python
#!/usr/bin/env python3
"""Mutation tests for scripts/check-site.py.

Each test copies the real tree into a scratch directory, breaks exactly one
invariant, and asserts the checker reports it. Reading the checker is not
evidence that it catches anything; breaking the thing it claims to catch is.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-site.py"


def run_in(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tree / "scripts" / "check-site.py")],
        capture_output=True,
        text=True,
        cwd=tree,
    )


class SiteGuardTests(unittest.TestCase):
    def scratch(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        tree = tmp / "repo"
        (tree / "scripts").mkdir(parents=True)
        shutil.copy(CHECKER, tree / "scripts" / "check-site.py")
        shutil.copytree(ROOT / "docs", tree / "docs")
        shutil.copytree(ROOT / "website", tree / "website")
        return tree

    def test_the_shipped_tree_is_clean(self) -> None:
        result = run_in(self.scratch())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_doc_without_an_h1_is_caught(self) -> None:
        tree = self.scratch()
        target = tree / "docs" / "USAGE.md"
        body = target.read_text().split("\n", 1)[1]
        target.write_text("Using vllm.cpp\n" + body)
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("USAGE.md", result.stderr)
        self.assertIn("H1", result.stderr)

    def test_a_doc_missing_from_nav_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        kept = [
            line
            for line in nav.read_text().splitlines(keepends=True)
            if "ROCM.md" not in line and "ROCm backend" not in line
        ]
        nav.write_text("".join(kept))
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ROCM.md", result.stderr)

    def test_a_nav_entry_with_no_file_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        nav.write_text(nav.read_text() + "  - file: GHOST.md\n    label: Ghost\n")
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("GHOST.md", result.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/scripts/test_check_site.py`
Expected: all four tests ERROR — `scripts/check-site.py` does not exist yet, so `shutil.copy` raises `FileNotFoundError`.

- [ ] **Step 3: Write the checker**

Create `scripts/check-site.py`:

```python
#!/usr/bin/env python3
"""The documentation site depends on two facts about docs/ that nothing else holds.

The site at website/ mounts docs/ read-only and adds no front matter, so it
derives what it needs from the files themselves:

  1. Every published doc opens with an `# H1`, because that heading IS the page
     title (website/layouts/partials/title.html). A doc without one publishes
     with its filename in the browser tab.
  2. website/data/nav.yaml lists exactly the published set, because that file IS
     the sidebar. A new doc absent from it is invisible on the site; a deleted
     doc still listed in it is a dead link.

Both are conventions today. A convention a build silently depends on is a latent
breakage, so they are gated here.

This checker never modifies docs/ and never reads Hugo's output. It compares the
source tree against nav.yaml, so it is fast, needs no Hugo, and fails for a
reason a reader can act on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NAV = ROOT / "website" / "data" / "nav.yaml"

H1 = re.compile(r"^#\s+\S")
NAV_FILE = re.compile(r"^\s*-\s+file:\s*(\S+)\s*$")


def published_docs() -> set[str]:
    """The docs the site actually publishes.

    A NON-recursive glob, which is what makes this agree with website/hugo.toml:
    the mount excludes `bench-evidence/**` and `superpowers/**`, and those are
    the only subdirectories of docs/. Matching top-level *.md therefore yields
    exactly the published set, with no exclusion list here to fall out of sync
    with the one in the config.

    If docs/ ever grows a subdirectory that IS published, this glob and the
    mount's excludeFiles both have to learn about it, and this checker will say
    so: the new pages would be absent from nav.yaml.
    """
    return {path.name for path in DOCS.glob("*.md")}


def nav_entries() -> list[str]:
    """Filenames listed in nav.yaml, in order.

    Parsed with a regex rather than a YAML library on purpose: this checker runs
    in CI jobs that install nothing, and the file's shape is fixed and simple.
    A malformed line simply does not match, and the bijection check below then
    reports the file as missing from the nav -- it fails closed, not open.
    """
    if not NAV.exists():
        print(f"ERROR: {NAV.relative_to(ROOT)} does not exist", file=sys.stderr)
        raise SystemExit(1)
    return [
        match.group(1)
        for line in NAV.read_text(encoding="utf-8").splitlines()
        if (match := NAV_FILE.match(line))
    ]


def main() -> int:
    errors: list[str] = []
    published = published_docs()

    for name in sorted(published):
        first = ""
        for line in (DOCS / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                first = line
                break
        if not H1.match(first):
            errors.append(
                f"docs/{name}: must open with an `# H1` -- it is the page title on "
                f"the site (website/layouts/partials/title.html); found {first!r}"
            )

    listed = nav_entries()
    for name in sorted(published - set(listed)):
        errors.append(
            f"docs/{name}: published but absent from website/data/nav.yaml, so it "
            f"has no sidebar entry and is unreachable on the site"
        )
    for name in sorted(set(listed) - published):
        errors.append(
            f"website/data/nav.yaml lists {name}, which is not a published doc -- "
            f"a dead sidebar link"
        )
    duplicates = sorted({name for name in listed if listed.count(name) > 1})
    for name in duplicates:
        errors.append(f"website/data/nav.yaml lists {name} more than once")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"site OK: {len(published)} published docs, nav in bijection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Make it executable: `chmod +x scripts/check-site.py`

- [ ] **Step 4: Run the tests and the checker**

Run: `python3 tests/scripts/test_check_site.py`
Expected: PASS, 4 tests. If `test_the_shipped_tree_is_clean` fails, the real tree violates an invariant — fix the tree, not the test.

Run: `python3 scripts/check-site.py`
Expected: `site OK: 11 published docs, nav in bijection`, exit 0.

- [ ] **Step 5: Wire it into CI**

In `.github/workflows/ci.yml`, immediately after the `Every production env var is documented or classified` step (the one running `check-env-doc.py`, around line 84-87), insert:

```yaml
      - name: The docs site's content invariants hold
        run: |
          python3 scripts/check-site.py
          python3 tests/scripts/test_check_site.py
```

- [ ] **Step 6: Commit**

```bash
scripts/agent-preflight.sh --staged
git add scripts/check-site.py tests/scripts/test_check_site.py .github/workflows/ci.yml
git commit -F - <<'EOF'
feat(ENG-DOCS-SITE): gate the two content facts the site depends on (#224)

The site derives what front matter would normally carry, so it depends on two
properties of docs/ that no checker held: every published doc opens with an
`# H1` (that heading IS the page title), and data/nav.yaml lists exactly the
published set (that file IS the sidebar). Both were true by convention. A
convention a build silently depends on is a latent breakage — a doc landing
without an H1 would publish with its filename in the browser tab, and a new doc
absent from the nav would be unreachable, neither failing anything.

The checker compares the source tree against nav.yaml rather than reading
Hugo's output, so it needs no Hugo, runs in a CI job that installs nothing, and
fails with a line a reader can act on. nav.yaml is parsed by regex rather than a
YAML library for the same reason; a malformed line fails closed, reported as a
doc missing from the nav.

Mutation-verified, not just read: strip the H1 from USAGE.md -> caught; drop
ROCM.md from nav.yaml -> caught; add a nav entry with no file -> caught; the
shipped tree -> clean.

Verified: `check-site.py` reports 11 published docs, nav in bijection;
`test_check_site.py` 4/4.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Claude-Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 5: Style, search, deploy, and the pointers

**Files:**
- Modify: `website/assets/css/site.css`, `website/layouts/partials/sidebar.html`
- Create: `website/layouts/_default/index.json`, `website/README.md`, `.github/workflows/gh-pages.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the deployed site.

- [ ] **Step 1: Write the stylesheet**

Replace `website/assets/css/site.css`. Colours are the cyan and green the README badges already use, defined as custom properties so the dark/light pair is one override block:

```css
/* vllm.cpp docs. Palette taken from the README badges (#3ec8e0 cyan,
   #7ee787 green) so the site and the repository front page agree. */
:root {
  --bg: #ffffff;
  --fg: #1c2128;
  --muted: #6e7681;
  --rule: #d0d7de;
  --accent: #0b7285;
  --accent-2: #1a7f37;
  --code-bg: #f6f8fa;
  --sidebar-bg: #f6f8fa;
  --measure: 46rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #8b949e;
    --rule: #30363d;
    --accent: #3ec8e0;
    --accent-2: #7ee787;
    --code-bg: #161b22;
    --sidebar-bg: #010409;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.skip {
  position: absolute; left: -9999px;
}
.skip:focus {
  left: 1rem; top: 1rem; z-index: 10; padding: .5rem .75rem;
  background: var(--accent); color: var(--bg); border-radius: 4px;
}
.shell { display: flex; align-items: flex-start; }
.sidebar {
  position: sticky; top: 0; flex: 0 0 16rem; height: 100vh; overflow-y: auto;
  padding: 1.5rem 1rem; background: var(--sidebar-bg);
  border-right: 1px solid var(--rule);
}
.sidebar .brand {
  display: block; margin-bottom: 1rem; font-weight: 700; font-size: 1.1rem;
  color: var(--fg); text-decoration: none;
}
.sidebar ul { list-style: none; margin: 0; padding: 0; }
.sidebar li { margin: .15rem 0; }
.sidebar a {
  display: block; padding: .3rem .5rem; border-radius: 4px;
  color: var(--fg); text-decoration: none; font-size: .95rem;
}
.sidebar a:hover { background: var(--code-bg); }
.sidebar a.active { background: var(--code-bg); color: var(--accent); font-weight: 600; }
#site-search {
  width: 100%; margin-bottom: 1rem; padding: .4rem .5rem;
  border: 1px solid var(--rule); border-radius: 4px;
  background: var(--bg); color: var(--fg); font: inherit; font-size: .9rem;
}
main { flex: 1 1 auto; min-width: 0; padding: 2.5rem 2rem 6rem; }
.doc { max-width: var(--measure); }
.doc h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 1rem; }
.doc h2 {
  margin-top: 2.5rem; padding-bottom: .3rem; font-size: 1.4rem;
  border-bottom: 1px solid var(--rule);
}
.doc h3 { margin-top: 1.75rem; font-size: 1.15rem; }
.doc a { color: var(--accent); }
.doc code {
  padding: .15em .35em; border-radius: 3px;
  background: var(--code-bg); font-size: .9em;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.doc pre {
  padding: .9rem 1rem; overflow-x: auto; border-radius: 6px;
  background: var(--code-bg); border: 1px solid var(--rule);
}
.doc pre code { padding: 0; background: none; }
.doc blockquote {
  margin: 1.2rem 0; padding: .1rem 1rem; color: var(--muted);
  border-left: 3px solid var(--accent-2);
}
/* Wide tables are common in these docs (STATUS.md especially). Let the table
   scroll inside its own box rather than the whole page. */
.doc table { display: block; overflow-x: auto; border-collapse: collapse; max-width: 100%; }
.doc th, .doc td { padding: .4rem .6rem; border: 1px solid var(--rule); text-align: left; }
.doc th { background: var(--code-bg); }
.doc img { max-width: 100%; }
.cards { list-style: none; padding: 0; display: grid; gap: .6rem;
  grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); }
.cards a {
  display: block; padding: .8rem 1rem; border: 1px solid var(--rule);
  border-radius: 6px; color: var(--fg); text-decoration: none;
}
.cards a:hover { border-color: var(--accent); color: var(--accent); }
@media (max-width: 55rem) {
  .shell { display: block; }
  .sidebar { position: static; height: auto; width: auto; border-right: 0;
    border-bottom: 1px solid var(--rule); }
  main { padding: 1.5rem 1.1rem 4rem; }
}
```

- [ ] **Step 2: Add search and the active-page marker to the sidebar**

Replace `website/layouts/partials/sidebar.html`:

```html
<nav class="sidebar">
  <a class="brand" href="{{ "" | relURL }}">vllm.cpp</a>
  <input id="site-search" type="search" placeholder="Filter pages" aria-label="Filter pages">
  <ul id="site-nav">
    {{ range site.Data.nav.docs }}
      {{ $slug := lower (strings.TrimSuffix ".md" .file) }}
      {{ $url := printf "docs/%s/" $slug | relURL }}
      <li><a href="{{ $url }}"{{ if eq $.RelPermalink $url }} class="active"{{ end }}>{{ .label }}</a></li>
    {{ end }}
  </ul>
  <script>
  // 11 pages. A filter over the titles in the sidebar plus the headings in
  // index.json is the whole feature; anything heavier would be more machinery
  // than corpus.
  (function () {
    var box = document.getElementById('site-search');
    var items = Array.prototype.slice.call(
      document.querySelectorAll('#site-nav li'));
    var index = null;
    fetch("{{ "index.json" | relURL }}")
      .then(function (r) { return r.json(); })
      .then(function (d) { index = d; })
      .catch(function () { index = []; });
    box.addEventListener('input', function () {
      var q = box.value.trim().toLowerCase();
      items.forEach(function (li) {
        var a = li.querySelector('a');
        var hit = !q || a.textContent.toLowerCase().indexOf(q) !== -1;
        if (!hit && index) {
          hit = index.some(function (p) {
            return p.url === a.getAttribute('href') &&
              p.headings.toLowerCase().indexOf(q) !== -1;
          });
        }
        li.hidden = !hit;
      });
    });
  })();
  </script>
</nav>
```

- [ ] **Step 3: Emit the search index**

Create `website/layouts/_default/index.json`:

```json-template
{{- $pages := slice -}}
{{- range where site.RegularPages "Section" "docs" -}}
  {{- $headings := slice -}}
  {{- range findRE `(?m)^#{1,3}\s+(.+)$` .RawContent -}}
    {{- $headings = $headings | append (strings.TrimSpace (strings.TrimLeft . "# ")) -}}
  {{- end -}}
  {{- $pages = $pages | append (dict
      "title" (partial "title.html" .)
      "url" .RelPermalink
      "headings" (delimit $headings " ")) -}}
{{- end -}}
{{- $pages | jsonify -}}
```

Note: the file must be named exactly `index.json` under `layouts/_default/` — Hugo matches it to the `json` output format declared for `home` in `hugo.toml`.

- [ ] **Step 4: Build and verify the styling and index**

Run:

```bash
rm -rf website/public && hugo --minify -s website
test -f website/public/index.json && echo "index.json emitted ok"
python3 -c "import json;d=json.load(open('website/public/index.json'));print(len(d),'pages indexed');print(d[0]['title'],d[0]['url'])"
grep -c 'sidebar' website/public/docs/status/index.html
```

Expected: `index.json emitted ok`; `11 pages indexed` followed by a real title and URL; the grep returns a non-zero count.

Preview it and look at it in both colour schemes:

```bash
hugo server -s website
```

Expected: the site loads at the printed address, the sidebar lists 11 pages in reading order with the current one marked, typing in the filter narrows the list, and wide tables in Status scroll inside their own box rather than stretching the page.

- [ ] **Step 5: Write the deploy workflow**

Create `.github/workflows/gh-pages.yml`:

```yaml
name: docs site

# Build on every PR that touches the site or the documents it publishes, so a
# site that does not compile fails review instead of main. Deploy only from
# main.
on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'website/**'
      - 'assets/**'
      - '.github/workflows/gh-pages.yml'
  pull_request:
    paths:
      - 'docs/**'
      - 'website/**'
      - 'assets/**'
      - '.github/workflows/gh-pages.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

# Never cancel a deploy in progress: a half-published Pages artifact is worse
# than a slightly stale one.
concurrency:
  group: pages
  cancel-in-progress: false

env:
  # Pinned deliberately. website/hugo.toml uses `excludeFiles`, which Hugo
  # deprecates from 0.153 in favour of `files`; bumping past that without
  # migrating the key would publish docs/bench-evidence and docs/superpowers.
  HUGO_VERSION: "0.146.3"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: ${{ env.HUGO_VERSION }}
          extended: true

      - name: The site's content invariants hold
        run: python3 scripts/check-site.py

      - name: Configure Pages
        id: pages
        if: github.event_name != 'pull_request'
        uses: actions/configure-pages@v5

      # On a PR there is no Pages base URL to ask for, and none is needed: the
      # job only has to prove the site compiles.
      - name: Build
        working-directory: website
        run: |
          hugo --minify \
            --baseURL "${{ steps.pages.outputs.base_url || 'https://example.invalid/' }}/"

      - name: The excluded trees stayed excluded
        run: |
          test ! -e website/public/docs/bench-evidence
          test ! -e website/public/docs/superpowers

      - name: Upload artifact
        if: github.event_name != 'pull_request'
        uses: actions/upload-pages-artifact@v3
        with:
          path: website/public

  deploy:
    if: github.event_name != 'pull_request'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 6: Write the site README and the repo pointer**

Create `website/README.md`:

```markdown
# The documentation site

Hugo site published to <https://mudler.github.io/vllm.cpp/> by
[`.github/workflows/gh-pages.yml`](../.github/workflows/gh-pages.yml).

## It owns no content

`hugo.toml` mounts `../docs` read-only. The markdown on the site and the
markdown in the repository are the same bytes, so they cannot drift. **Never add
a copy of a document under `content/`, and never add front matter to a file in
`docs/`.** Titles come from each file's first `# H1`, the sidebar comes from
`data/nav.yaml`, and `scripts/check-site.py` gates both.

## Local preview

```sh
hugo server -s website     # requires Hugo 0.146.3 extended
hugo --minify -s website   # one-shot build into website/public/
python3 scripts/check-site.py
```

The Hugo version is pinned in the workflow. `hugo.toml` uses `excludeFiles`,
which Hugo deprecates from 0.153 in favour of `files`; bumping past that version
without migrating the key would publish `docs/bench-evidence/` and
`docs/superpowers/`.

## A custom domain, later

Put the hostname in `static/CNAME` and point the DNS at GitHub Pages. Nothing
else changes: the base URL comes from `actions/configure-pages` at build time,
not from a literal in the config. The name is currently parked behind the vLLM
trademark question.
```

In the repository `README.md`, add a documentation pointer. Insert it immediately after the badge block (after the `</p>` that closes the badges, before the "Brought to you by" line):

```markdown
<p align="center">
  <b><a href="https://mudler.github.io/vllm.cpp/">Documentation</a></b>
</p>
```

- [ ] **Step 7: Verify the workflow parses and the whole gate is green**

Run:

```bash
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/gh-pages.yml')); print('workflow parses ok')"
python3 scripts/check-site.py
python3 tests/scripts/test_check_site.py
python3 scripts/check-readme-structure.py
python3 tests/scripts/test_check_pr_size.py
rm -rf website/public && hugo --minify -s website && echo "site builds ok"
git status --porcelain docs/
```

Expected: `workflow parses ok`; the site checker reports 11 docs; 4/4 site tests; the README checker exits 0; the pr-size tests pass; `site builds ok`; and `git status --porcelain docs/` prints **nothing**.

- [ ] **Step 8: Commit**

```bash
scripts/agent-preflight.sh --staged
git add website .github/workflows/gh-pages.yml README.md
git commit -F - <<'EOF'
feat(ENG-DOCS-SITE): style, filter, and publish the docs site (#224)

The visual layer, the search such as it is, and the deploy.

CSS is ~130 lines on custom properties, in the cyan and green the README badges
already use so the site and the repository front page agree. One override block
covers dark mode. Wide tables — STATUS.md especially — scroll inside their own
box rather than stretching the page.

Search is a filter over 11 page titles plus their headings from a generated
index.json. At this corpus size that IS the feature; a search library would be
more machinery than content.

The workflow builds on every PR touching docs/, website/ or assets/ and deploys
only from main, so a site that does not compile fails review instead of main. It
re-runs check-site.py and re-asserts that bench-evidence/ and superpowers/ stayed
out of public/ — the exclusion is load-bearing enough to verify on both sides.
Hugo is pinned to 0.146.3 with the reason in the file: hugo.toml uses
`excludeFiles`, deprecated from 0.153, and a silent bump would publish both
excluded trees. `concurrency: pages` never cancels in progress, because a
half-published artifact is worse than a stale one.

The base URL comes from actions/configure-pages rather than a literal, so a
custom domain later is one static/CNAME file; website/README.md records that
path and the trademark question parking it.

Verified: workflow parses; site builds; index.json indexes 11 pages;
check-site.py + its 4 mutation tests green; check-readme-structure.py green;
`git status --porcelain docs/` empty.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Claude-Code:claude-opus-5 [ClaudeCode]
EOF
```

---

## Closing the row

After Task 5, before the PR is marked ready:

- [ ] Run the full staged gate once more: `scripts/agent-preflight.sh --staged`, `python3 scripts/check-agent-record.py`, `python3 scripts/check-pr-size.py`.
- [ ] Confirm the whole-branch `docs/` diff contains only this plan file: `git diff --stat origin/main -- docs/`.
- [ ] Move `ENG-DOCS-SITE` from `READY` to `GATING` in `.agents/engine-matrix.md`, updating the summary counts in the same edit (`READY` 1 → 0, `GATING` 2 → 3 in the *Serving, API, CLI, library* row and the Total row). It becomes `DONE` only once the site is reachable — see the stop condition below.
- [ ] Add the `## Outcome` section to `.agents/specs/gh-pages-docs-site.md`: what was measured, what was rejected (themes, and why), and why the Hugo version is pinned where it is.

**Stop condition, carried from the spec:** GitHub Pages must be enabled on `mudler/vllm.cpp` with the source set to GitHub Actions. That is a repository setting, not a file, and it is outside an implementer's authority. If it is not enabled, the workflow is inert: report it and leave the row at `GATING` rather than claiming a published site.

## Notes for the implementer

- **The one rule that matters:** if you find yourself editing a file under `docs/`, stop. Nothing in this plan requires it, and the whole design exists to avoid it. If a doc genuinely cannot render without a change, return `NEEDS_DECISION`.
- **Hugo template errors are terse.** When a build fails, `hugo -s website --logLevel debug` names the template and line.
- **`relURL` everywhere.** The site is published under `/vllm.cpp/`, not at a domain root, so a hardcoded `/docs/status/` link works locally and 404s in production.
- **Do not add front matter to make something easier.** Every mechanism here exists specifically because the mounted files have none, and adding it to one file breaks the rule for all of them.
