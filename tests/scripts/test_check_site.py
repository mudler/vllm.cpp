#!/usr/bin/env python3
"""Mutation tests for scripts/check-site.py.

Each test copies the real tree into a scratch directory, breaks exactly one
invariant, and asserts the checker reports it. Reading the checker is not
evidence that it catches anything; breaking the thing it claims to catch is.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-site.py"

# The site is rendered under a path prefix, exactly as GitHub Pages serves it.
# `BASE_PATH` is derived from `BASE_URL` rather than written twice, so the
# rendered href and the on-disk lookup cannot drift apart.
BASE_URL = "https://example.invalid/vllm.cpp/"
BASE_PATH = urlparse(BASE_URL).path


def benchmark_slug(href: str) -> str | None:
    """The detail-page slug an index href names, or None if it names none.

    A benchmark detail page is emitted at `<base>/docs/benchmarks/<slug>/`, so
    the slug is the last path segment of an href whose parent is that section.
    The index page itself, a link back out to another doc and an off-site URL
    all answer None.
    """
    path = PurePosixPath(urlparse(href).path.rstrip("/"))
    parent = path.parent
    if parent.name != "benchmarks" or parent.parent.name != "docs":
        return None
    return path.name


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.table_hrefs: list[str] = []
        self._table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                if self._table_depth:
                    self.table_hrefs.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table_depth -= 1


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
        # Only what the checker reads: the top-level docs and the nav file.
        # Copying docs/bench-evidence and docs/superpowers would make every test
        # noticeably slower for no coverage.
        (tree / "docs").mkdir()
        for doc in (ROOT / "docs").glob("*.md"):
            shutil.copy(doc, tree / "docs" / doc.name)
        (tree / "website" / "data").mkdir(parents=True)
        shutil.copy(
            ROOT / "website" / "data" / "nav.yaml",
            tree / "website" / "data" / "nav.yaml",
        )
        return tree

    def test_the_shipped_tree_is_clean(self) -> None:
        result = run_in(self.scratch())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nav in bijection", result.stdout)

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
        self.assertIn("absent from", result.stderr)

    def test_a_nav_entry_with_no_file_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        nav.write_text(nav.read_text() + "  - file: GHOST.md\n    label: Ghost\n")
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("GHOST.md", result.stderr)
        self.assertIn("dead sidebar link", result.stderr)

    def test_a_duplicated_nav_entry_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        nav.write_text(nav.read_text() + "  - file: ROCM.md\n    label: ROCm again\n")
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("more than once", result.stderr)

    def test_a_missing_nav_file_fails_closed(self) -> None:
        tree = self.scratch()
        (tree / "website" / "data" / "nav.yaml").unlink()
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not exist", result.stderr)

    def test_rendered_benchmark_index_links_resolve_to_emitted_pages(self) -> None:
        public = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, public, ignore_errors=True)
        result = subprocess.run(
            [
                "hugo",
                "--minify",
                "-s",
                str(ROOT / "website"),
                "--destination",
                str(public),
                "--baseURL",
                BASE_URL,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        index = public / "docs" / "benchmarks" / "index.html"
        parser = LinkCollector()
        parser.feed(index.read_text(encoding="utf-8"))

        # The population is every `docs/benchmarks/*.md`, section pages
        # (`at-a-glance`, `how-we-measure`, `memory`, `open-gaps`, `reproduce`)
        # included, and NOT only the files whose stem is a benchmark ID.
        # `docs/BENCHMARKS.md` carries one table row per file in that directory,
        # and `scripts/check-benchmark-index.py` already refuses an index row
        # with no detail file and a detail file with no index row. So the source
        # side of this relation is a bijection by contract, and the question
        # left for a RENDERED page is whether Hugo carried it across intact.
        detail_slugs = {
            path.stem for path in (ROOT / "docs" / "benchmarks").glob("*.md")
        }
        # DERIVED from the URL shape, never from `detail_slugs`: filtering the
        # hrefs by the slugs they are about to be compared against would make a
        # link to a page that does not exist vanish from the comparison instead
        # of failing it.
        linked = [
            slug
            for slug in (benchmark_slug(href) for href in parser.table_hrefs)
            if slug is not None
        ]

        # Non-emptiness first, and explicitly. Set equality over two empty sets
        # is a gate with its mute switch on: a render that emitted no table at
        # all, or a `docs/benchmarks/` that lost every file, would satisfy the
        # comparison below and report a pass.
        self.assertTrue(
            detail_slugs, "docs/benchmarks/ holds no detail pages to link"
        )
        self.assertTrue(
            linked, "the rendered benchmark index links no detail page at all"
        )
        # Both sides are read off the tree and the render, so publishing a
        # benchmark moves both together and no literal here has to be edited.
        # Duplicates are deliberately tolerated: one detail page may legitimately
        # be linked from another row's prose, and a duplicate INDEX ROW is
        # already refused by scripts/check-benchmark-index.py.
        self.assertEqual(
            set(linked),
            detail_slugs,
            "the rendered benchmark index and docs/benchmarks/ disagree: "
            f"linked but absent from the tree {sorted(set(linked) - detail_slugs)}, "
            f"present in the tree but unlinked {sorted(detail_slugs - set(linked))}",
        )

        for href in parser.table_hrefs:
            if benchmark_slug(href) is None:
                continue
            emitted = (
                public
                / urlparse(href).path.removeprefix(BASE_PATH)
                / "index.html"
            )
            self.assertTrue(
                emitted.is_file(), f"rendered benchmark link has no emitted page: {href}"
            )


if __name__ == "__main__":
    unittest.main()
