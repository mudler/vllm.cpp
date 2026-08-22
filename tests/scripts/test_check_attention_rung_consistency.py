#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-attention-rung-consistency.py.

The mutation cases below are the point of the file. A checker that reports zero
drift on a green tree proves nothing on its own: it reports zero drift when its
regex matches nothing at all, which is exactly how #1544's defect went unseen for
nine call sites. `MutationTests` below holds the cases that guard against it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-attention-rung-consistency.py"
SPEC = importlib.util.spec_from_file_location("check_attention_rung", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

MODELS = "src/vllm/model_executor/models"
ALLOWLIST = ROOT / "scripts/attention-rung-allowlist.txt"

MARKED = """\
void Forward() {
  // VT-ATTN-NAIVE: reference arm of the dense/paged equivalence gate.
  vt::Attention(q, out, qq, kk, vv, args);
}
"""

UNMARKED = """\
void Forward() {
  vt::Attention(q, out, qq, kk, vv, args);
}
"""


class CallDetectionTests(unittest.TestCase):
    def test_unmarked_site_is_found(self) -> None:
        self.assertEqual(mod.scan_file(UNMARKED), [(2, False)])

    def test_marked_site_is_found_and_credited(self) -> None:
        self.assertEqual(mod.scan_file(MARKED), [(3, True)])

    def test_fast_rungs_are_never_sites(self) -> None:
        # The trailing `\(` is what excludes these, not the `\b`. Measured: all
        # five still fail to match with the `\b` removed.
        for fast in (
            "vt::AttentionDenseFlash(q, o, a, b, c, args);",
            "vt::AttentionDenseFast(q, o, a, b, c, args);",
            "vt::AttentionDenseFa2(q, o, a, b, c, args);",
            "vt::AttentionCross(q, o, a, b, c, args);",
            "vt::PagedAttention(q, o, a, b, c, args);",
        ):
            self.assertEqual(mod.scan_file(fast), [], fast)

    def test_a_commented_out_call_is_not_a_site(self) -> None:
        self.assertEqual(mod.scan_file("// vt::Attention(q, o, a, b, c, args);\n"), [])

    def test_a_block_commented_call_is_not_a_site(self) -> None:
        text = "/* legacy:\n vt::Attention(q, o, a, b, c, args);\n*/\n"
        self.assertEqual(mod.scan_file(text), [])

    def test_an_if_zero_call_is_not_a_site(self) -> None:
        text = "#if 0\nvt::Attention(q, o, a, b, c, args);\n#endif\n"
        self.assertEqual(mod.scan_file(text), [])

    def test_line_numbers_survive_normalization(self) -> None:
        # normalize_source is position-preserving; a checker that reported a line
        # from the normalized text would drift by every stripped block comment.
        text = "/* a\n b\n c */\n\nvt::Attention(q, o, a, b, c, args);\n"
        self.assertEqual(mod.scan_file(text), [(5, False)])
        self.assertEqual(text.splitlines()[4].strip()[:14], "vt::Attention(")

    def test_two_sites_are_reported_independently(self) -> None:
        text = MARKED + "\n" * 40 + UNMARKED
        sites = mod.scan_file(text)
        self.assertEqual([marked for _, marked in sites], [True, False])


class MarkerTests(unittest.TestCase):
    def test_reason_must_be_substantive(self) -> None:
        self.assertIsNone(mod.marker_reason("  // nothing to see here"))
        self.assertEqual(mod.marker_reason("// VT-ATTN-NAIVE: x"), "x")
        # ...but a one-character reason does not satisfy the site.
        self.assertFalse(mod.has_marker(["// VT-ATTN-NAIVE: x", "vt::Attention(a);"], 2))

    def test_marker_must_be_a_comment(self) -> None:
        # A string literal naming the marker is not a record.
        self.assertIsNone(mod.marker_reason('const char* s = "VT-ATTN-NAIVE: nope at all";'))

    def test_marker_window_is_bounded(self) -> None:
        lines = ["// VT-ATTN-NAIVE: a genuine recorded reason"] + [""] * 40
        lines.append("vt::Attention(a);")
        self.assertFalse(mod.has_marker(lines, len(lines)))
        near = ["// VT-ATTN-NAIVE: a genuine recorded reason"] + [""] * 5
        near.append("vt::Attention(a);")
        self.assertTrue(mod.has_marker(near, len(near)))

    def test_marker_on_the_call_line_counts(self) -> None:
        line = "vt::Attention(a);  // VT-ATTN-NAIVE: the eager rung of the A/B"
        self.assertTrue(mod.has_marker([line], 1))


class DriftTests(unittest.TestCase):
    def test_marked_site_never_drifts(self) -> None:
        self.assertEqual(
            mod.drift_sites({f"{MODELS}/nemotron_h.cpp": [(675, True)]}, set()), []
        )

    def test_unmarked_site_drifts(self) -> None:
        self.assertEqual(
            mod.drift_sites({f"{MODELS}/muse_glimmer_vision.cpp": [(639, False)]}, set()),
            [(f"{MODELS}/muse_glimmer_vision.cpp", 639)],
        )

    def test_allowlisted_stem_passes(self) -> None:
        self.assertEqual(
            mod.drift_sites({f"{MODELS}/ltx2.cpp": [(959, False)]}, {"ltx2"}),
            [],
        )

    def test_mixed_reports_only_the_unmarked(self) -> None:
        self.assertEqual(
            mod.drift_sites(
                {
                    f"{MODELS}/whisper_audio.cpp": [(324, True)],
                    f"{MODELS}/qwen3_5.cpp": [(5279, True)],
                    f"{MODELS}/muse_glimmer_vision.cpp": [(639, False)],
                    f"{MODELS}/ltx2.cpp": [(959, False)],
                },
                allowlisted={"ltx2"},
            ),
            [(f"{MODELS}/muse_glimmer_vision.cpp", 639)],
        )

    def test_stale_entry_is_reported_and_not_fatal(self) -> None:
        scanned = {f"{MODELS}/ltx2.cpp": [(959, True)]}
        self.assertEqual(mod.stale_allowlist_entries(scanned, {"ltx2"}), ["ltx2"])
        self.assertEqual(mod.drift_sites(scanned, {"ltx2"}), [])

    def test_a_header_and_a_cpp_sharing_a_stem_do_not_collide(self) -> None:
        # Keyed on the PATH: keyed on the stem, ltx2.h would overwrite ltx2.cpp and
        # the checker would silently scan one file instead of two.
        scanned = {
            f"{MODELS}/ltx2.cpp": [(959, False)],
            "include/vllm/model_executor/models/ltx2.h": [(31, False)],
        }
        self.assertEqual(len(scanned), 2)
        self.assertEqual(len(mod.drift_sites(scanned, set())), 2)
        self.assertEqual(mod.drift_sites(scanned, {"ltx2"}), [])

    def test_allowlist_parsing(self) -> None:
        text = "# comment\nltx2  # trailing reason\nltx2_device\n\n"
        self.assertEqual(mod.allowlisted_names(text), {"ltx2", "ltx2_device"})


class ShippedTreeTests(unittest.TestCase):
    def scan(self):
        return mod.scan_models(), mod.allowlisted_names(
            ALLOWLIST.read_text(encoding="utf-8")
        )

    def test_shipped_tree_is_green(self) -> None:
        scanned, allowed = self.scan()
        self.assertEqual(mod.drift_sites(scanned, allowed), [])

    def test_the_population_is_not_empty(self) -> None:
        # A scanner that matches nothing is green for the wrong reason: an empty
        # scan and a clean tree file the same report. This is the guard against a
        # regex that stops matching after a rename, and it asserts only that the
        # scan still finds SOMETHING.
        #
        # It deliberately does NOT pin the count. A raw total is a measurement of
        # the model tree stored in this file, so it reds on every row that
        # legitimately REMOVES a vt::Attention call -- which is precisely the rows
        # the allowlist exists to unblock, and precisely the drift lock AGENTS.md
        # `## Records` forbids: never store a measurement of one file inside
        # another file. The floor of 9 this replaces had zero headroom against a
        # shipped tree of exactly 9 sites, so #1545's routing change alone would
        # have turned `main` red. Issue #1629.
        #
        # What a count would have bought is covered without the coupling:
        # test_the_six_deliberate_sites_carry_a_marker names its files, and
        # test_every_allowlisted_stem_names_a_real_model_source below catches the
        # bogus entry a total never could.
        scanned, _ = self.scan()
        self.assertGreaterEqual(sum(len(v) for v in scanned.values()), 1)

    def test_the_six_deliberate_sites_carry_a_marker(self) -> None:
        scanned, _ = self.scan()
        for stem in (
            "whisper_audio",
            "qwen3_vl_vision",
            "kimi_linear_device",
            "qwen3_5",
            "nemotron_h",
            "nemotron_h_device",
        ):
            path = f"{MODELS}/{stem}.cpp"
            self.assertIn(path, scanned, path)
            self.assertTrue(all(m for _, m in scanned[path]), path)

    def test_allowlist_holds_only_the_in_flight_stems(self) -> None:
        # It is not a parking lot. Growth is a review decision, and this pins the
        # set so growth is visible in a diff of this file.
        #
        # The set is EMPTY, which is the enforcement closed rather than the guard
        # switched off: all three original stems were deleted once their removing
        # rows landed (47a918d8f for muse_glimmer_vision, 90e8c3c85 for ltx2 and
        # ltx2_device), and #1663 removed them here. An empty expected set still
        # reds on the next silent append, which is the whole point of this case.
        #
        # It does NOT make the suite vacuous. drift_sites is what excuses a call,
        # and with an empty allowlist it excuses nothing, so
        # test_shipped_tree_is_green above now measures the tree on its markers
        # alone. The stems it used to cover are re-asserted positively below.
        _, allowed = self.scan()
        self.assertEqual(allowed, set())

    def test_the_formerly_allowlisted_stems_pass_on_their_own_merit(self) -> None:
        # The obligation the allowlist deferred, now stated where a regression
        # would be read: each of the three stems is green because its file earned
        # it, never because a line in a text file excused it.
        #
        # ltx2 and ltx2_device still NAME vt::Attention -- a host CPU-only arm and
        # the OFF arm of a same-binary A/B -- so they are asserted as MARKED.
        # muse_glimmer_vision was routed to vt::AttentionDenseFlash outright, so
        # it is asserted ABSENT from the scan. Asserting the same thing about all
        # three would be false of one of them in either direction.
        scanned, _ = self.scan()
        for stem in ("ltx2", "ltx2_device"):
            path = f"{MODELS}/{stem}.cpp"
            self.assertIn(path, scanned, path)
            self.assertTrue(all(marked for _, marked in scanned[path]), path)
        self.assertNotIn(f"{MODELS}/muse_glimmer_vision.cpp", scanned)

    def test_every_allowlisted_stem_names_a_real_model_source(self) -> None:
        # Pins that every allowlisted stem names a model source that exists, so a
        # typo is reported at the typo.
        #
        # DORMANT rather than dead now that the allowlist is empty: this iterates
        # over nothing and asserts nothing on the shipped tree. That is the
        # correct state for it -- it is a guard on a file that is currently
        # empty, and it fires on the first TYPO added to it. Not on the first
        # addition: both arms were measured, and they differ. Appending
        # `zzz_bogus_model` reds this case and the pinning case together, while
        # appending `whisper_audio` -- a real stem, so a real source file -- reds
        # the pinning case and `test_deleting_a_marker_goes_red` and leaves THIS
        # case green, because the stem it names exists. Deleting this case
        # because it is quiet today would remove the typo report from exactly
        # the edit that needs it.
        #
        # Keyed on the FILE existing, never on scan membership. A stem stops having
        # a call site the moment its removing row lands -- that is the state the
        # allowlist is built to survive, which the checker's own
        # stale_allowlist_entries docstring states -- so asserting the stem is
        # still in `scanned` would rebuild exactly the lock #1629 removes.
        _, allowed = self.scan()
        for stem in sorted(allowed):
            sources = [
                models_dir / f"{stem}{suffix}"
                for models_dir in mod.MODEL_DIRS
                for suffix in (".cpp", ".h")
            ]
            self.assertTrue(
                any(path.is_file() for path in sources),
                f"allowlisted stem {stem!r} names no model source: none of "
                + ", ".join(str(path) for path in sources)
                + " exists. It excuses nothing, and the checker reports the "
                "mismatch only indirectly: an unexcused call site in the file "
                "the stem was meant to cover, or a STALE line naming the "
                "misspelling.",
            )


class MutationTests(unittest.TestCase):
    """Guards for the regressions the checker exists to catch."""

    def setUp(self) -> None:
        self.scanned, self.allowed = mod.scan_models(), mod.allowlisted_names(
            ALLOWLIST.read_text(encoding="utf-8")
        )

    def test_a_new_unmarked_model_goes_red(self) -> None:
        mutated = dict(self.scanned)
        new = f"{MODELS}/some_new_vision_tower.cpp"
        mutated[new] = [(412, False)]
        self.assertEqual(mod.drift_sites(mutated, self.allowed), [(new, 412)])

    def test_a_new_unmarked_call_in_a_HEADER_goes_red(self) -> None:
        # The bypass the .h glob closes: a call moved into an inline function.
        mutated = dict(self.scanned)
        hdr = "include/vllm/model_executor/models/some_new_tower.h"
        mutated[hdr] = [(88, False)]
        self.assertEqual(mod.drift_sites(mutated, self.allowed), [(hdr, 88)])

    def test_deleting_a_marker_goes_red(self) -> None:
        mutated = dict(self.scanned)
        path = f"{MODELS}/whisper_audio.cpp"
        mutated[path] = [(line, False) for line, _ in self.scanned[path]]
        self.assertTrue(mod.drift_sites(mutated, self.allowed))

    def test_a_second_unmarked_call_in_a_marked_file_goes_red(self) -> None:
        # Per-SITE, not per-file: a file that already records one reason must not
        # launder a new naive call added elsewhere in it.
        mutated = dict(self.scanned)
        path = f"{MODELS}/qwen3_5.cpp"
        mutated[path] = list(self.scanned[path]) + [(9999, False)]
        self.assertIn((path, 9999), mod.drift_sites(mutated, self.allowed))

    def test_a_stub_reason_goes_red(self) -> None:
        lines = ["// VT-ATTN-NAIVE: todo", "vt::Attention(a);"]
        self.assertFalse(mod.has_marker(lines, 2))

    def test_widening_the_regex_to_the_fast_rungs_is_visible(self) -> None:
        # These pin the trailing `\(` and the `\s*`, never the `\b`. Measured: with
        # the `\b` removed, this suite and the shipped tree both stay green.
        #
        # Three checker comments name a cause that measurement refutes, and none is
        # repairable here: check-pr-size.py refuses a comment-only edit to a
        # governance checker (#1631). They are the widening paragraph (:58-61), the
        # `_NAIVE_CALL` comment (:93-96), and the `excused` cause (:252-255).
        self.assertIsNone(mod._NAIVE_CALL.search("vt::AttentionDenseFlash(a);"))
        self.assertIsNotNone(mod._NAIVE_CALL.search("vt::Attention (a);"))


class GreenReportTests(unittest.TestCase):
    """What the OK line tells a reader who never opens the allowlist.

    A green that prints only "9 sites, 6 marked" reads as three unaccounted sites
    or as nothing at all, depending on whether the reader does the subtraction. The
    number that matters is how many sites carry NO reason and pass anyway. Nothing
    asserted this line before, so the count could be dropped or silently go wrong
    without a red.
    """

    def report(self) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = mod.main()
        self.assertEqual(code, 0, buffer.getvalue())
        return buffer.getvalue()

    def report_over(self, scanned, allowlist_text: str) -> str:
        """`main()`'s own report path, driven over a CONSTRUCTED tree.

        The report reads two module-level names, so both are redirected for the
        call: `scan_models`, to a dict built by hand exactly as `MutationTests`
        builds one, and `ALLOWLIST`, to a temporary file. Nothing the model tree
        or the real allowlist does can then change what these cases assert, which
        is the whole reason they do not read either one.
        """
        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = Path(tmp) / "attention-rung-allowlist.txt"
            allowlist.write_text(allowlist_text, encoding="utf-8")
            with mock.patch.object(mod, "scan_models", lambda: scanned):
                with mock.patch.object(mod, "ALLOWLIST", allowlist):
                    with contextlib.redirect_stdout(buffer):
                        code = mod.main()
        self.assertEqual(code, 0, buffer.getvalue())
        return buffer.getvalue()

    def test_the_ok_line_counts_the_sites_an_allowlist_excuses(self) -> None:
        # The guard proper: the "unmarked and excused" branch is exercised with a
        # NON-ZERO count on a tree this file owns. The constructed scan holds one
        # allowlisted file carrying an unmarked site -- the debt the green hides --
        # beside a marked one, plus a marked site nothing excuses, so a checker
        # that dropped the clause, or hard-coded the number, cannot produce this
        # line. Both mutations were run against this case and both turn it red.
        #
        # A checker that printed `sites - marked` in place of `excused` is NOT on
        # that list, and no constructed scan can add it. `main()` reaches the OK
        # line only when `drift_sites` is empty, and `drift_sites` is empty exactly
        # when no unmarked site sits outside an allowlisted file -- so on every
        # green the checker can print, every unmarked site is excused and
        # `excused == sites - marked` identically. That substitution was mutated
        # into the checker and the whole suite stayed green, so the claim is
        # recorded here as unpinnable rather than left standing as a guarantee this
        # case does not carry.
        report = self.report_over(
            {
                f"{MODELS}/ltx2.cpp": [(10, False), (20, True)],
                f"{MODELS}/whisper_audio.cpp": [(30, True)],
            },
            "# in flight\nltx2  # a row already open removes this call\n",
        )
        self.assertEqual(
            report.strip(),
            "OK (attention rung): 3 vt::Attention call site(s) in 2 model source "
            "file(s); 2 carry a recorded reason, 1 unmarked and excused by 1 "
            "allowlisted in-flight stem(s).",
        )

    def test_the_ok_line_reports_zero_when_no_stem_is_allowlisted(self) -> None:
        # The state the allowlist exists to REACH: every in-flight row has landed
        # and the file parks no stem. The checker prints the count even at zero,
        # and this pins that, so no case has to require the shipped tree to still
        # carry excused debt in order to keep the branch covered.
        report = self.report_over(
            {f"{MODELS}/whisper_audio.cpp": [(30, True)]},
            "# nothing in flight\n",
        )
        self.assertEqual(
            report.strip(),
            "OK (attention rung): 1 vt::Attention call site(s) in 1 model source "
            "file(s); 1 carry a recorded reason, 0 unmarked and excused by 0 "
            "allowlisted in-flight stem(s).",
        )

    def test_the_ok_line_reports_the_excused_sites(self) -> None:
        # The shipped-tree half: whatever the tree's excused count IS, the OK line
        # must state it. `excused` is RE-DERIVED here and never pinned, so this
        # holds at 3 today and at 0 once the last in-flight stem is cleaned up.
        #
        # It deliberately does NOT assert `excused > 0`. That floor was a
        # measurement of the model tree stored in this file: it passes only while
        # some stem is still parked on the allowlist, so the row that removes the
        # LAST one turns this case red while the checker itself is green at rc=0 --
        # the drift lock AGENTS.md `## Records` forbids, and the same shape as the
        # `>= 9` population floor #1629 removed. The branch it was standing in for
        # is covered above, on a tree this file constructs.
        scanned = mod.scan_models()
        allowed = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        excused = sum(
            1
            for path, sites in scanned.items()
            if Path(path).stem in allowed
            for _, marked in sites
            if not marked
        )
        self.assertIn(f"{excused} unmarked and excused by", self.report())

    def test_the_excused_count_is_not_sites_minus_marked(self) -> None:
        # The subtraction a reader would do by hand.
        #
        # Read this as a record of the two DEFINITIONS, not as a gate on the OK
        # line: it never calls `main()`, and it derives both numbers itself.
        # Holding the OK line to `excused` rather than to the subtraction is not
        # something this file can do from any scan -- the identity is spelled out
        # in test_the_ok_line_counts_the_sites_an_allowlist_excuses.
        scanned = {
            "src/vllm/model_executor/models/ltx2.cpp": [(10, False), (20, True)],
            "src/vllm/model_executor/models/whisper_audio.cpp": [(30, True)],
        }
        allowed = {"ltx2"}
        excused = sum(
            1
            for path, sites in scanned.items()
            if Path(path).stem in allowed
            for _, marked in sites
            if not marked
        )
        sites = sum(len(v) for v in scanned.values())
        marked = sum(1 for v in scanned.values() for _, m in v if m)
        self.assertEqual(excused, 1)
        self.assertEqual(sites - marked, 1)
        # They agree HERE, and on any scan the checker reports OK on, because no
        # unmarked site sits outside the allowlist -- which is what the assert
        # below states. Dropping the allowlisted file's marked site does not
        # separate them either: it lowers `sites` and `marked` together. They
        # diverge only once an unmarked site lands in a file no stem excuses, and
        # that scan is a red.
        self.assertEqual(mod.drift_sites(scanned, allowed), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
