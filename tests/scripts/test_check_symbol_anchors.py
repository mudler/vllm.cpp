#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-symbol-anchors.py (#1143, #1139).

The case this suite exists to pin is `test_verdict_depends_on_the_citing_text`.
#911 shipped an anchor checker that read its expectation out of the file it was
checking and reported 27/27 FRESH while five anchors pointed at unrelated code.
A checker of that shape cannot distinguish two citations that name DIFFERENT
symbols in the SAME file, because nothing it reads varies. This one must, and
that case asserts it does.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-symbol-anchors.py"
CHECKED_RE = re.compile(r"in-repo checked (\d+) \(fresh (\d+), stale (\d+)\)")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


class Tree:
    """A throwaway tree with one cited file and one citing file."""

    def __init__(self, cited_path: str, cited_body: str, citing_body: str,
                 citing_path: str = "note.md"):
        self.box = tempfile.TemporaryDirectory()
        root = Path(self.box.name)
        for rel, body in ((cited_path, cited_body), (citing_path, citing_body)):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        self.root = root

    def __enter__(self) -> Path:
        return self.root

    def __exit__(self, *exc) -> None:
        self.box.cleanup()


class SymbolAnchorTests(unittest.TestCase):
    def test_self_test_sweeps_a_non_empty_corpus(self):
        result = run("--self-test")
        self.assertEqual(result.returncode, 0, result.stdout)
        match = re.search(r"self-test fixtures: (\d+), failures: (\d+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertGreater(int(match.group(1)), 0, "a corpus of zero fixtures proves nothing")
        self.assertEqual(int(match.group(2)), 0, result.stdout)

    def test_this_tree_is_green_over_a_non_zero_population(self):
        result = run()
        self.assertEqual(result.returncode, 0, result.stdout)
        match = CHECKED_RE.search(result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertGreater(
            int(match.group(1)), 0,
            "a green run that checked nothing is a skip wearing a pass",
        )
        self.assertEqual(int(match.group(3)), 0, result.stdout)

    def test_fresh_symbol_passes(self):
        with Tree("src/vllm/a.cpp", "struct Widget {};\n",
                  "see `src/vllm/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("in-repo checked 1 (fresh 1, stale 0)", result.stdout)

    def test_renamed_symbol_is_reported(self):
        with Tree("src/vllm/a.cpp", "struct Gadget {};\n",
                  "see `src/vllm/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not contain `Widget`", result.stdout)
        self.assertIn("note.md:1", result.stdout)

    def test_verdict_depends_on_the_citing_text(self):
        """The anti-tautology case (#911).

        One cited file, unchanged, cited twice. The two verdicts must differ,
        which is only possible if the expectation comes from the CITING side.
        """

        body = "struct Widget {};\n"
        with Tree("src/vllm/a.cpp", body, "`src/vllm/a.cpp::Widget`\n") as root:
            good = run("--root", str(root))
        with Tree("src/vllm/a.cpp", body, "`src/vllm/a.cpp::Gadget`\n") as root:
            bad = run("--root", str(root))
        self.assertEqual(good.returncode, 0, good.stdout)
        self.assertEqual(bad.returncode, 1, bad.stdout)
        self.assertIn("fresh 1, stale 0", good.stdout)
        self.assertIn("fresh 0, stale 1", bad.stdout)

    def test_a_call_site_satisfies_a_citation(self):
        with Tree("src/vllm/a.cpp", "  ModelRegistry::Load(config);\n",
                  "`src/vllm/a.cpp::ModelRegistry::Load`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_longer_identifier_does_not_satisfy_a_shorter_citation(self):
        with Tree("src/vllm/a.cpp", "  LoadShards(dir);\n",
                  "`src/vllm/a.cpp::Load`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_a_local_path_that_does_not_exist_is_reported(self):
        with Tree("src/vllm/a.cpp", "struct Widget {};\n",
                  "`src/vllm/a.cpp::Widget` and `src/vllm/gone.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("names a path under this repository that does not exist", result.stdout)

    def test_an_upstream_path_is_bucketed_not_failed(self):
        with Tree("src/vllm/a.cpp", "struct Widget {};\n",
                  "`src/vllm/a.cpp::Widget` and `vllm/config/model.py::ModelConfig`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("upstream/unknown 1", result.stdout)

    def test_zero_checked_citations_is_a_failure(self):
        with Tree("src/vllm/a.cpp", "struct Widget {};\n",
                  "`vllm/config/model.py::ModelConfig` only\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("0 in-repo citations were checked", result.stdout)

    def test_the_frozen_archive_is_skipped_and_counted(self):
        with Tree("src/vllm/a.cpp", "struct Widget {};\n",
                  "`src/vllm/a.cpp::Gadget` is stale here\n",
                  citing_path=".agents/completed/old/STATE-1.md") as root:
            (root / "live.md").write_text("`src/vllm/a.cpp::Widget`\n", encoding="utf-8")
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1 frozen files skipped", result.stdout)

    def test_prose_without_a_path_is_not_a_citation(self):
        with Tree("src/vllm/a.cpp", "struct Widget {};\n",
                  "`Qwen3_5MTPKind::kMoe` and `src/vllm/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1 citations", result.stdout)
        self.assertIn("in-repo checked 1", result.stdout)

    def test_parity_pin_is_read_from_the_recorded_block(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_csa", CHECKER)
            module = importlib.util.module_from_spec(spec)
            # `from __future__ import annotations` + @dataclass needs the module
            # visible in sys.modules while its body executes.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            pin = module.parity_pin(ROOT / ".agents/upstream-sync.md")
        finally:
            sys.path.pop(0)
        self.assertRegex(pin, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
