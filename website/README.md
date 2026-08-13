# The documentation site

Hugo site published to <https://mudler.github.io/vllm.cpp/> by
[`.github/workflows/gh-pages.yml`](../.github/workflows/gh-pages.yml).
Design: [`.agents/specs/gh-pages-docs-site.md`](../.agents/specs/gh-pages-docs-site.md).

## It owns no content

`hugo.toml` mounts `../docs` read-only. The markdown on the site and the
markdown in the repository are the same bytes, so they cannot drift.

**Never add a copy of a document under `content/`, and never add front matter to
a file in `docs/`.** Titles come from each file's first `# H1`, sidebar order
from `data/nav.yaml`, and internal links from the render hook in
`layouts/_default/_markup/`.

`scripts/check-site.py` gates the first two: every published doc must open with
an `# H1`, and `data/nav.yaml` must be in bijection with the mounted set.

## Local preview

```sh
hugo server -s website                 # requires Hugo 0.146.3 extended
hugo server -s website --bind 0.0.0.0  # reachable from the network
hugo --minify -s website               # one-shot build into website/public/
python3 scripts/check-site.py
```

Check `hugo server`, not just `hugo`: the server renders page kinds the one-shot
build may skip, and a template that crashes on them passes a green build.

The Hugo version is pinned in the workflow. `hugo.toml` uses `excludeFiles`,
which Hugo deprecates from 0.153 in favour of `files`; **bumping past that
version without migrating the key would publish `docs/bench-evidence/` and
`docs/superpowers/`.** The workflow re-asserts their absence after every build.

## Layout

`hugo.toml` holds the mounts and pins. `data/nav.yaml` is the sidebar.
`content/_index.md` is the home page. `layouts/partials/title.html` derives
titles, `layouts/_default/_markup/` holds the link and heading render hooks, and
`assets/css/` holds the design plus the generated highlighting.

### Regenerating the syntax highlighting

`hugo.toml` sets `noClasses = false`, so Chroma emits class names and no inline
colours. Without `assets/css/chroma.css` every code block renders as flat text.

```sh
hugo gen chromastyles --style=github       # light
hugo gen chromastyles --style=github-dark  # dark, wrapped in a prefers-color-scheme block
```

## No webfonts, no third-party requests

The site loads nothing from another host. No CDN, no font service, no
analytics. Typography is font stacks only: a serif does the display work and the
monospace carries labels, kickers and table headers.

Shipping woff2 files was considered and rejected, though no longer for the
original reason: `check-pr-size.py` used to refuse binary changes outright, and
that guard was retired 2026-08-13 (#615), so a woff2 under `website/static/`
would now pass the gate as an `asset`. It stays rejected on its own merits —
a self-hosted face is bytes every visitor pays for and a licence someone has to
keep tracking, and the stacks below already carry the design.

The logos and favicon are **mounted** from the repository's `assets/`, not
copied here, the same rule as `docs/`: a logo refreshed upstream must not
leave a stale twin on the site. `hugo.toml`'s `includeFiles` keeps the rest of that
directory (about 1 MB of unrelated artwork) off the published site.

## A custom domain, later

Put the hostname in `static/CNAME` and point DNS at GitHub Pages. Nothing else
changes: the base URL comes from `actions/configure-pages` at build time, not
from a literal in the config. The name is currently parked behind the vLLM
trademark question.
