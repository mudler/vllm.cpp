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
    for name in sorted({name for name in listed if listed.count(name) > 1}):
        errors.append(f"website/data/nav.yaml lists {name} more than once")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"site OK: {len(published)} published docs, nav in bijection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
