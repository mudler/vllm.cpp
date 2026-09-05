#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-env-doc.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-env-doc.py"
SPEC = importlib.util.spec_from_file_location("check_env_doc", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_env_doc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_env_doc
SPEC.loader.exec_module(check_env_doc)

undocumented = check_env_doc.undocumented_env_vars


def _doc_text() -> str:
    return (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")


class UndocumentedEnvVarTests(unittest.TestCase):
    def test_documented_var_passes(self) -> None:
        self.assertEqual(
            undocumented({"VT_FOO"}, {"VT_FOO"}, set()), []
        )

    def test_allowlisted_var_passes(self) -> None:
        self.assertEqual(
            undocumented({"VT_FOO"}, set(), {"VT_FOO"}), []
        )

    def test_undocumented_var_fails(self) -> None:
        # A scanned var covered by neither surface is reported.
        self.assertEqual(
            undocumented({"VT_NEW_KNOB"}, set(), set()), ["VT_NEW_KNOB"]
        )

    def test_mixed_reports_only_the_uncovered(self) -> None:
        result = undocumented(
            {"VT_A", "VT_B", "VLLM_C"},
            documented={"VT_A"},
            allowlisted={"VT_B"},
        )
        self.assertEqual(result, ["VLLM_C"])

    def test_helpers_harvest_names(self) -> None:
        doc = "The `VT_CPU_REF` knob and `VLLM_CPP_CPU_THREADS`."
        self.assertEqual(
            check_env_doc.documented_names(doc),
            {"VT_CPU_REF", "VLLM_CPP_CPU_THREADS"},
        )
        allow = "# comment\nVT_GDN_TMA\nVT_MOE_DECODE   # trailing\n\n"
        self.assertEqual(
            check_env_doc.allowlisted_names(allow), {"VT_GDN_TMA", "VT_MOE_DECODE"}
        )

    def test_shipped_tree_is_fully_covered(self) -> None:
        # The real repo must pass: every scanned name is documented or allowlisted.
        scanned = check_env_doc.scan_env_names(ROOT)
        documented = check_env_doc.documented_names(
            (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")
        )
        allowlisted = check_env_doc.allowlisted_names(
            (ROOT / "scripts/env-doc-allowlist.txt").read_text(encoding="utf-8")
        )
        self.assertEqual(undocumented(scanned, documented, allowlisted), [])
        self.assertGreater(len(scanned), 100)  # the sweep actually found names

    def test_inherited_variables_have_exact_public_internal_split(self) -> None:
        public = {
            "VT_GEMMA4_EXPERT_VRAM_MB",
            "VT_SERVER_MAX_NEW_TOKENS",
            "VT_SERVER_MAX_PROMPT_CHARS",
        }
        internal = {
            "VT_GEMMA4_BATCH_EXPERTS",
            "VT_GEMMA4_CUSTOM_EXPERT",
            "VT_GEMMA4_FP8_NATIVE",
            "VT_GEMMA4_FUSED_EXPERTS",
            "VT_GEMMA4_HOST_AXPY",
            "VT_GEMMA4_PROFILE",
            "VT_ROCM_GEMM_COMPUTE",
            "VT_ROCM_GEMV",
            "VT_ROCM_HIPBLASLT",
        }
        inherited = public | internal
        scanned = check_env_doc.scan_env_names(ROOT)
        documented = check_env_doc.documented_names(
            (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")
        )
        allowlisted = check_env_doc.allowlisted_names(
            (ROOT / "scripts/env-doc-allowlist.txt").read_text(encoding="utf-8")
        )

        self.assertLessEqual(inherited, scanned)
        self.assertEqual(inherited & documented, public)
        self.assertEqual(inherited & allowlisted, internal)
        self.assertTrue((inherited & documented).isdisjoint(inherited & allowlisted))

    def test_a_fabricated_new_var_would_fail(self) -> None:
        # Mutation: pretend the code grew a new undocumented var; it must trip.
        scanned = check_env_doc.scan_env_names(ROOT)
        documented = check_env_doc.documented_names(
            (ROOT / "docs/ENVIRONMENT.md").read_text(encoding="utf-8")
        )
        allowlisted = check_env_doc.allowlisted_names(
            (ROOT / "scripts/env-doc-allowlist.txt").read_text(encoding="utf-8")
        )
        mutated = set(scanned) | {"VT_A_BRAND_NEW_UNDOCUMENTED_KNOB"}
        result = undocumented(mutated, documented, allowlisted)
        self.assertIn("VT_A_BRAND_NEW_UNDOCUMENTED_KNOB", result)


class UnreadDocumentedVarTests(unittest.TestCase):
    """The REVERSE direction (#2389): documented in a table, read by nothing.

    Every case here calls a symbol the BASE checker does not define, so the
    whole class errors against BASE and passes against HEAD. That is the
    evidence contract check-pr-size.py asks a checker change to carry.
    """

    def test_a_documented_and_read_var_passes(self) -> None:
        self.assertEqual(
            check_env_doc.unread_documented_vars({"VT_FOO"}, {"VT_FOO"}, {}), []
        )

    def test_a_documented_and_unread_var_fails(self) -> None:
        self.assertEqual(
            check_env_doc.unread_documented_vars({"VT_DEAD"}, set(), {}), ["VT_DEAD"]
        )

    def test_a_declared_exception_is_tolerated(self) -> None:
        self.assertEqual(
            check_env_doc.unread_documented_vars(
                {"VT_DEAD"}, set(), {"VT_DEAD": "owned by #2385"}
            ),
            [],
        )

    def test_an_exception_whose_var_left_the_table_is_stale(self) -> None:
        # Self-clearing: once the doc row goes, the entry has to go too.
        self.assertEqual(
            check_env_doc.stale_unread_exceptions(
                set(), set(), {"VT_GONE": "a reason"}
            ),
            ["VT_GONE"],
        )

    def test_an_exception_whose_var_gained_a_reader_is_stale(self) -> None:
        self.assertEqual(
            check_env_doc.stale_unread_exceptions(
                {"VT_REVIVED"}, {"VT_REVIVED"}, {"VT_REVIVED": "a reason"}
            ),
            ["VT_REVIVED"],
        )

    def test_a_live_exception_is_not_stale(self) -> None:
        self.assertEqual(
            check_env_doc.stale_unread_exceptions(
                {"VT_DEAD"}, set(), {"VT_DEAD": "a reason"}
            ),
            [],
        )

    def test_an_exception_without_a_reason_is_refused(self) -> None:
        self.assertEqual(
            check_env_doc.unreasoned_unread_exceptions({"VT_A": "why", "VT_B": "  "}),
            ["VT_B"],
        )

    # --- complication 1: a comment is not a read ---------------------------

    def test_strip_comments_removes_a_line_comment(self) -> None:
        source = 'const char* k = "VT_REAL";  // see "VT_ONLY_IN_A_COMMENT"\n'
        stripped = check_env_doc.strip_comments(source)
        self.assertIn("VT_REAL", stripped)
        self.assertNotIn("VT_ONLY_IN_A_COMMENT", stripped)

    def test_strip_comments_removes_a_block_comment(self) -> None:
        source = '/* getenv("VT_BLOCK_ONLY") is gone */ getenv("VT_REAL");'
        stripped = check_env_doc.strip_comments(source)
        self.assertIn("VT_REAL", stripped)
        self.assertNotIn("VT_BLOCK_ONLY", stripped)

    def test_strip_comments_keeps_a_slash_inside_a_string(self) -> None:
        # A `//` inside a literal must not open a comment and swallow real code.
        source = 'url("http://x"); getenv("VT_AFTER_A_URL");'
        stripped = check_env_doc.strip_comments(source)
        self.assertIn("VT_AFTER_A_URL", stripped)

    def test_strip_comments_survives_a_digit_separator(self) -> None:
        # `'` in 1'000'000 is not a char literal; mistaking it eats the reads.
        source = "int n = 1'000'000;\ngetenv(\"VT_AFTER_A_SEPARATOR\");"
        stripped = check_env_doc.strip_comments(source)
        self.assertIn("VT_AFTER_A_SEPARATOR", stripped)

    def test_scan_read_sites_ignores_a_comment_quoted_name(self) -> None:
        """Pins the strip_comments CALL SITE inside scan_read_sites.

        The shipped tree cannot prove this on its own: its one comment-only
        knob is written in markdown backticks, so the quoted-literal regex
        already excludes it and deleting the strip would not move any tree
        assertion. A comment that QUOTES the name -- the shape a reader leaves
        behind when they delete the lookup and describe it instead -- is the
        case that needs its own tree.
        """

        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "src").mkdir()
            (root / "src/a.cpp").write_text(
                '// the old path called getenv("VT_COMMENTED_OUT_KNOB")\n'
                '/* and getenv("VT_BLOCK_COMMENTED_KNOB") too */\n'
                'const char* v = getenv("VT_ACTUALLY_READ");\n',
                encoding="utf-8",
            )
            found = check_env_doc.scan_read_sites(root)
        self.assertEqual(found, {"VT_ACTUALLY_READ"})

    def test_the_comment_only_knob_that_motivated_this_check_is_now_GONE(self) -> None:
        """`VT_QWEN35_STAGE_MIN_FREE_FRAC` was the case a name grep passes.

        Its only occurrence in compiled code was a `//` comment, so it was
        documented and unread, and it shipped as the single UNREAD_EXCEPTIONS
        entry so this gate could land without editing a row another row owned.

        ENG-WEIGHT-RESIDENCY / #2385 then removed the row, the staleness guard
        reported the entry BY NAME, and the entry was deleted rather than
        rewritten. So the assertion here is the END STATE: gone from the tables,
        gone from the header comment, and not allowlisted -- an escape hatch that
        opened and closed once, end to end, which is the only evidence that it is
        self-clearing rather than permanent.

        The GUARANTEE this case used to carry -- that a comment-quoted name is
        not counted as read -- has no real-tree instance left, so it is pinned
        synthetically by `test_scan_read_sites_ignores_a_comment_quoted_name`.
        Keeping a real-tree assertion against a subject that no longer exists
        would be a test that passes because its premise vanished.
        """

        name = "VT_QWEN35_STAGE_MIN_FREE_FRAC"
        self.assertNotIn(name, check_env_doc.table_documented_names(_doc_text()))
        self.assertNotIn(name, check_env_doc.scan_read_sites(ROOT))
        self.assertNotIn(name, check_env_doc.UNREAD_EXCEPTIONS)
        # It survives only as PROSE in the page, which the table parser must not
        # mistake for a documented row -- otherwise removing a knob could never
        # be explained on the page that documented it.
        self.assertIn(name, _doc_text())

    # --- complication 2: a shipped binary outside src/ still counts ---------

    def test_a_knob_read_only_in_examples_counts_as_read(self) -> None:
        """VT_BENCH_PRETOKENIZE lives in examples/bench/, the vllm-bench binary."""

        name = "VT_BENCH_PRETOKENIZE"
        self.assertIn(name, check_env_doc.table_documented_names(_doc_text()))
        self.assertNotIn(name, check_env_doc.scan_env_names(ROOT))  # src/+include/
        self.assertIn(name, check_env_doc.scan_read_sites(ROOT))  # + examples/
        self.assertIn("examples", check_env_doc.READ_SITE_ROOTS)

    def test_read_site_roots_exclude_the_uncompiled_trees(self) -> None:
        # benchmarks/ and tools/ are not add_subdirectory'd; a read there is not
        # a read anybody can reach, and counting it would hide a dead knob.
        self.assertNotIn("benchmarks", check_env_doc.READ_SITE_ROOTS)
        self.assertNotIn("tools", check_env_doc.READ_SITE_ROOTS)

    # --- complication 3: the read need not be a literal getenv -------------

    def test_a_knob_read_through_a_helper_counts_as_read(self) -> None:
        """VT_GGUF_KEEP_QUANT is read via EnvOnOr(...), never a bare getenv."""

        name = "VT_GGUF_KEEP_QUANT"
        self.assertIn(name, check_env_doc.table_documented_names(_doc_text()))
        self.assertIn(name, check_env_doc.scan_read_sites(ROOT))
        source = (
            ROOT / "src/vllm/model_executor/model_loader/gguf_keep_quant.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(f'EnvOnOr("{name}"', source)
        self.assertNotIn(f'getenv("{name}")', source)

    # --- the table parser --------------------------------------------------

    def test_table_names_come_from_the_first_column_only(self) -> None:
        doc = (
            "| Variable | Default | What it does |\n"
            "|---|---|---|\n"
            "| `VT_IN_THE_TABLE` | off | mentions `VT_IN_A_CELL` |\n"
            "\nProse naming `VT_IN_PROSE` is not a table row.\n"
        )
        self.assertEqual(
            check_env_doc.table_documented_names(doc), {"VT_IN_THE_TABLE"}
        )

    def test_the_table_set_is_a_real_subset_of_the_documented_set(self) -> None:
        doc = _doc_text()
        table = check_env_doc.table_documented_names(doc)
        self.assertLessEqual(table, check_env_doc.documented_names(doc))
        self.assertGreater(len(table), 100)  # the parser actually found rows

    # --- the shipped tree --------------------------------------------------

    def test_shipped_tree_has_no_undeclared_dead_knob(self) -> None:
        doc = _doc_text()
        table = check_env_doc.table_documented_names(doc)
        read = check_env_doc.scan_read_sites(ROOT)
        self.assertEqual(
            check_env_doc.unread_documented_vars(
                table, read, check_env_doc.UNREAD_EXCEPTIONS
            ),
            [],
        )
        self.assertEqual(
            check_env_doc.stale_unread_exceptions(
                table, read, check_env_doc.UNREAD_EXCEPTIONS
            ),
            [],
        )
        self.assertEqual(
            check_env_doc.unreasoned_unread_exceptions(check_env_doc.UNREAD_EXCEPTIONS),
            [],
        )

    def test_the_gemma4_knob_that_was_never_wired_is_gone_from_the_tables(self) -> None:
        """#2389's own repair: the row went, rather than being caveated."""

        name = "VT_GEMMA4_MLP_MOE_PARALLEL"
        self.assertNotIn(name, check_env_doc.table_documented_names(_doc_text()))
        self.assertNotIn(name, check_env_doc.scan_read_sites(ROOT))
        self.assertNotIn(name, check_env_doc.UNREAD_EXCEPTIONS)

    def test_a_fabricated_dead_doc_row_would_fail(self) -> None:
        # Mutation: pretend the page grew a row for a knob nothing reads.
        doc = _doc_text()
        table = check_env_doc.table_documented_names(doc) | {"VT_A_BRAND_NEW_DEAD_ROW"}
        read = check_env_doc.scan_read_sites(ROOT)
        self.assertIn(
            "VT_A_BRAND_NEW_DEAD_ROW",
            check_env_doc.unread_documented_vars(
                table, read, check_env_doc.UNREAD_EXCEPTIONS
            ),
        )

    def test_the_checker_exits_nonzero_on_a_dead_doc_row(self) -> None:
        """End to end: main() must actually REPORT the reverse direction.

        Mutating only the pure function proves the predicate; this proves main()
        calls it. Without the call site, main() returns 0 on a dead row.
        """

        import io
        import contextlib
        from unittest import mock

        with mock.patch.object(
            check_env_doc,
            "UNREAD_EXCEPTIONS",
            dict(check_env_doc.UNREAD_EXCEPTIONS),
        ), mock.patch.object(
            check_env_doc,
            "table_documented_names",
            lambda text: {"VT_A_BRAND_NEW_DEAD_ROW"},
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(
                io.StringIO()
            ):
                rc = check_env_doc.main()
        self.assertEqual(rc, 1)
        self.assertIn("VT_A_BRAND_NEW_DEAD_ROW", err.getvalue())


if __name__ == "__main__":
    unittest.main()
