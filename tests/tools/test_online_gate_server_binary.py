"""The gate harness must consume the server binary the build actually emits.

`examples/CMakeLists.txt` declares the CMake TARGET ``server`` and sets its
``OUTPUT_NAME`` to ``vllm-server``, so the file on disk is ``examples/vllm-server``.
The harness checked the on-disk PATH ``examples/server``, which stopped existing
when the release packaging renamed the output (`1a02ab4f`, W6): ``--execute``
aborted with "provenance-recorded build did not produce examples/server" before
any server started, so the 27B/35B ratio could not be measured at all.

A target name and an output name are allowed to differ; that is exactly why the
consumer must not hardcode a guess at either. These cases pin the harness to the
declared ``OUTPUT_NAME``, so renaming the artifact again fails HERE — cheaply, on
any box — instead of on the gate host after a multi-hour build.

The stale spelling is not confined to one driver, so the guard below SCANS for it
rather than naming the files it already knows about: naming files is how #222's
first repair (`2b262622`) missed the whole Python half, and how its second
(`8fce04d3`) still left three live sites — a quick start in ``docs/USAGE.md`` and
two in ``README.md``. The scan includes the corrected README commands too.

The repair on ``main`` does not hardcode the new name either: it resolves
``examples/<OUTPUT_NAME>`` and keeps the pre-rename path as a REPLAY FALLBACK, so
a pre-W6 evidence tree still replays against the binary it was recorded with.
That is a legitimate consumer of ``examples/server``, and ``FALLBACK_PATTERNS``
is what tells it apart from the defect — by the shape of the alternative, and
only in a file that resolves the declared name too.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES_CMAKE = REPO_ROOT / "examples" / "CMakeLists.txt"
HARNESS_PY = REPO_ROOT / "tools" / "bench" / "online_gate.py"
HARNESS_SH = REPO_ROOT / "scripts" / "dgx-online-serving.sh"

# Sibling modules under tests/tools import `tools.bench.*` at module scope and
# are run as `python3 -m unittest tests.tools.<name>` from the repository root.
# This module also carries a `__main__` entry point, so make the direct
# `python3 tests/tools/<name>.py` invocation resolve the same packages instead
# of dying on ImportError halfway through the suite.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def declared_server_output_name() -> str:
    """The OUTPUT_NAME the build gives the ``server`` target.

    Parsed from the build definition rather than duplicated here, so this test
    cannot drift into agreeing with a stale harness.
    """
    text = EXAMPLES_CMAKE.read_text(encoding="utf-8")
    match = re.search(
        r"set_target_properties\(\s*server\s+PROPERTIES\s+OUTPUT_NAME\s+([A-Za-z0-9_.-]+)",
        text,
    )
    if match:
        return match.group(1)
    # No OUTPUT_NAME override => the artifact is named after the target.
    if re.search(r"add_executable\(\s*server\b", text):
        return "server"
    raise AssertionError(
        f"could not determine the server executable name from {EXAMPLES_CMAKE}"
    )


# --- repository-wide scan for the stale artifact path -----------------------
#
# `examples/server/` is also the SOURCE directory of the server example, and
# `server` is still the CMake target name, so neither spelling is forbidden on
# its own. What is forbidden is resolving the BUILT FILE at `<dir>/examples/
# server`. The three shapes below match exactly that and nothing else:
#
#   * a path with a directory component in front of it -- `${build_dir}/
#     examples/server`, `build/examples/server` -- and not continuing into
#     `examples/server/main.cpp`;
#   * pathlib's `"examples" / "server"` join;
#   * a check on the artifact's BASENAME -- `server.name != "server"` -- the same
#     defect with the directory factored out, and how
#     `tools/bench/gdn_packed_component.py` rejected the manifest that
#     `online_gate.py record-execution` had just written for it.
#
# Prose about the target (`examples/server is a thin client`) has no leading
# path component and is therefore not matched.
STALE_ARTIFACT_PATTERNS = (
    re.compile(r"[^\s`]/examples/server(?![/\w.\-])"),
    re.compile(r'"examples"\s*/\s*"server"'),
    re.compile(r'\.(?:name|stem)\s*[!=]=\s*"server"'),
)

# The ONE shape that may still resolve the stale path: the alternative of a
# fallback whose primary is the declared artifact. `2b262622`/`8fce04d3` chose
# that over hardcoding the new name, so a pre-W6 evidence tree replays against
# the binary it was recorded with instead of silently selecting a different one.
#
# The exemption is SHAPE-based, not "this file mentions the new name": a
# file-level exemption would let a fresh hardcode ride in beside the fallback.
# It is ALSO conditioned on the file resolving the declared name (`_stale_lines`),
# so deleting the primary turns the fallback back into a plain hardcode.
#
# `gdn_packed_component.py`'s basename check needs no entry: `server.name not in
# ("vllm-server", "server")` admits both names, so it never matches
# STALE_ARTIFACT_PATTERNS at all, while a regression to `!= "server"` still does.
FALLBACK_PATTERNS = (
    # shell: `[[ -x ${server_bin} ]] || server_bin="${build_dir}/examples/server"`
    re.compile(r'\|\|\s*\w+="\$\{build_dir\}/examples/server"\s*$'),
    # python: `return current if current.exists() else build_dir / "examples" / "server"`
    re.compile(r'\belse\s+\w+\s*/\s*"examples"\s*/\s*"server"\s*$'),
)

# Files whose content is EXECUTED, BUILT or READ BY A USER.  `.agents/` and
# `benchmarks/` are deliberately absent: they are append-only records that quote
# the command a past campaign actually ran, and rewriting them would falsify
# evidence rather than repair a consumer.
SCANNED_DIRECTORIES = (
    ".github",
    "cmake",
    "docker",
    "docs",
    "examples",
    "include",
    "release",
    "scripts",
    "src",
    "tests",
    "tools",
)
SCANNED_ROOT_FILES = ("CMakeLists.txt", "CONTRIBUTING.md", "README.md")
SCANNED_SUFFIXES = frozenset(
    {"", ".cfg", ".cmake", ".cpp", ".cu", ".cuh", ".h", ".in", ".json",
     ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)
# This module has to spell the forbidden shapes out to search for them.
SCAN_EXEMPT = frozenset({pathlib.Path(__file__).resolve()})


def _scanned_files() -> list[pathlib.Path]:
    files = [REPO_ROOT / name for name in SCANNED_ROOT_FILES]
    for directory in SCANNED_DIRECTORIES:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in SCANNED_SUFFIXES
        )
    return [
        path
        for path in files
        if path.is_file() and path.resolve() not in SCAN_EXEMPT
    ]


def _stale_lines(text: str, expected: str) -> list[tuple[int, str]]:
    """`(lineno, text)` for lines resolving the built server at `examples/server`.

    A recognized replay fallback is exempt, but ONLY in a file that also
    resolves the declared name -- with its primary gone, the same line is an
    ordinary hardcode and is reported again.
    """
    exempt = f"examples/{expected}" in text or f'"examples" / "{expected}"' in text
    stale: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not any(pattern.search(line) for pattern in STALE_ARTIFACT_PATTERNS):
            continue
        if exempt and any(pattern.search(line) for pattern in FALLBACK_PATTERNS):
            continue
        stale.append((number, line.strip()))
    return stale


def stale_server_artifact_references() -> list[str]:
    """`<file>:<line>: <text>` for every live consumer of the stale path."""
    expected = declared_server_output_name()
    hits: list[str] = []
    for path in sorted(_scanned_files()):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(REPO_ROOT)
        hits.extend(
            f"{rel}:{number}: {line}" for number, line in _stale_lines(text, expected)
        )
    return hits


# --- the shell driver's own dispatch, parsed rather than string-matched -----


_MODEL_KEY_RE = re.compile(r"\$\{model\}\s*==\s*([A-Za-z0-9_.-]+)")
_BRANCH_RE = re.compile(r"(?:el)?if \[\[(?P<condition>.+)\]\]; then")
_TEST_NAME_RE = re.compile(r"test_name=(?P<name>\S+)")


def _unique_line(text: str, predicate) -> str:
    matches = [line for line in text.splitlines() if predicate(line)]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one matching line in {HARNESS_SH.name}, found "
            f"{len(matches)}: {matches!r}"
        )
    return matches[0]


def accepted_model_keys() -> frozenset[str]:
    """The `--model` values the driver's argument guard admits.

    Parsed from the guard LINE, so an assertion about one key can never be
    satisfied by a different line that happens to contain the same substring.
    """
    text = HARNESS_SH.read_text(encoding="utf-8")
    guard = _unique_line(
        text,
        lambda line: line.startswith("[[ ${model} ==") and line.rstrip().endswith("]] || {"),
    )
    return frozenset(_MODEL_KEY_RE.findall(guard))


def correctness_gate_by_model_key(keys) -> dict[str, str]:
    """Replay the driver's `test_name` if/elif/else dispatch for each key.

    Returns the `test_name` each `--model` value would actually select. A key
    that reaches the fall-through arm resolves to that arm's value, which is the
    defect this exists to catch: `27n` silently benched with no golden.
    """
    text = HARNESS_SH.read_text(encoding="utf-8")
    branches: list[tuple[frozenset[str] | None, str]] = []
    condition: frozenset[str] | None = None
    in_chain = False
    for raw in text.splitlines():
        line = raw.strip()
        branch = _BRANCH_RE.fullmatch(line)
        if branch is not None:
            condition = frozenset(_MODEL_KEY_RE.findall(branch.group("condition")))
            in_chain = True
            continue
        if line == "else" and in_chain:
            condition = None  # the fall-through arm matches every remaining key
            continue
        if line == "fi":
            in_chain = False
            condition = None
            continue
        assignment = _TEST_NAME_RE.fullmatch(line)
        if assignment is not None:
            if not in_chain:
                raise AssertionError(
                    f"test_name={assignment.group('name')} is assigned outside an "
                    "if/elif/else chain; this parser no longer models the driver"
                )
            branches.append((condition, assignment.group("name")))
    if not branches:
        raise AssertionError(f"found no test_name dispatch in {HARNESS_SH.name}")
    resolved: dict[str, str] = {}
    for key in keys:
        for models, name in branches:
            if models is None or key in models:
                resolved[key] = name
                break
    return resolved


class ServerBinaryNameContract(unittest.TestCase):
    def test_cmake_declares_a_server_executable(self) -> None:
        self.assertTrue(
            EXAMPLES_CMAKE.is_file(), f"missing build definition: {EXAMPLES_CMAKE}"
        )
        self.assertTrue(declared_server_output_name())

    # `examples/` also holds vllm-cli, vllm-bench and the per-model generators,
    # which the harness legitimately references, so these assert on the stale
    # SPELLING rather than rejecting every leaf that is not the server.

    def _assert_no_stale_spelling(self, path: pathlib.Path) -> None:
        """Reject `examples/server` ONLY while the build emits something else.

        With `OUTPUT_NAME` removed the pair of assertions would otherwise be
        self-contradictory: the artifact IS `examples/server`, and the harness
        would be required both to name it and never to mention it.
        """
        expected = declared_server_output_name()
        if expected == "server":
            return
        stale = _stale_lines(path.read_text(encoding="utf-8"), expected)
        self.assertEqual(
            [],
            stale,
            f"{path.name} still resolves the server as examples/server "
            f"(the CMake TARGET name) outside a replay fallback; the build emits "
            f"examples/{expected}, so the gate aborts before starting a server",
        )

    def test_python_harness_uses_the_declared_output_name(self) -> None:
        expected = declared_server_output_name()
        text = HARNESS_PY.read_text(encoding="utf-8")
        self.assertTrue(
            f'"examples" / "{expected}"' in text,
            f"{HARNESS_PY.name} never resolves the server as examples/{expected}, "
            "the artifact the build emits",
        )
        self._assert_no_stale_spelling(HARNESS_PY)

    def test_shell_harness_uses_the_declared_output_name(self) -> None:
        expected = declared_server_output_name()
        text = HARNESS_SH.read_text(encoding="utf-8")
        self.assertTrue(
            f"examples/{expected}" in text,
            f"{HARNESS_SH.name} never resolves the server as examples/{expected}, "
            "the artifact the build emits",
        )
        self._assert_no_stale_spelling(HARNESS_SH)

    def test_diagnostic_text_names_the_real_artifact(self) -> None:
        """The abort message must name the file a builder can look for."""
        expected = declared_server_output_name()
        text = HARNESS_SH.read_text(encoding="utf-8")
        self.assertTrue(
            f"did not produce examples/{expected}" in text,
            "the failure message must name the artifact the build actually emits, "
            "or it sends the reader looking for a file that never existed",
        )

    def test_the_scan_can_see_the_stale_path(self) -> None:
        """The scanner is only evidence if it fails on a planted occurrence."""
        cases = {
            '  server = build / "examples" / "server"': True,
            '    "${build_dir}/examples/server"': True,
            "build/examples/server --model /path/to/model": True,
            '"<VLLM_CPP_BUILD>/examples/server",': True,
            '    if server.name != "server" or server.parent.name != "examples":': True,
            "// examples/server/main.cpp:743-1096 @ fc636c76": False,
            "# lives in the library so examples/server can be a thin client": False,
            '        "examples/server/",': False,
            '    "${build_dir}/examples/vllm-server"': False,
            "build_targets=(server)": False,
            '    if server.name != "vllm-server":': False,
            # main's basename check admits BOTH names, so it is not the defect.
            '    if server.name not in ("vllm-server", "server"):': False,
            "  add_executable(server server/main.cpp)": False,
        }
        for line, is_stale in cases.items():
            with self.subTest(line=line):
                self.assertEqual(
                    is_stale,
                    any(p.search(line) for p in STALE_ARTIFACT_PATTERNS),
                    "the scan must match a stale artifact reference, and must not "
                    "reject the source directory or the CMake target name",
                )

    def test_the_scan_accepts_a_replay_fallback_and_nothing_else(self) -> None:
        """`main`'s repair keeps `examples/server` as a FALLBACK, not a hardcode.

        Three properties a naive exemption gets wrong: the documented shapes
        pass; the same line WITHOUT its primary is reported again; and a plain
        hardcode beside a primary is still reported, so "this file mentions the
        new name" can never launder a fresh occurrence.
        """
        expected = "vllm-server"
        fallbacks = {
            "shell": (
                'server_bin="${build_dir}/examples/vllm-server"\n'
                '[[ -x ${server_bin} ]] || server_bin="${build_dir}/examples/server"\n'
            ),
            "python": (
                '    current = build_dir / "examples" / "vllm-server"\n'
                '    return current if current.exists() else build_dir / "examples" / "server"\n'
            ),
        }
        for shape, text in fallbacks.items():
            with self.subTest(shape=shape):
                self.assertEqual([], _stale_lines(text, expected))
                orphan = text.splitlines()[1] + "\n"
                self.assertEqual(
                    1,
                    len(_stale_lines(orphan, expected)),
                    "a fallback with no primary is just a hardcode",
                )
        hardcoded = (
            'server_bin="${build_dir}/examples/vllm-server"\n'
            '    --root "${build_dir}/examples/server" \\\n'
        )
        self.assertEqual(
            1,
            len(_stale_lines(hardcoded, expected)),
            "resolving the declared name elsewhere in the file must not exempt a "
            "plain hardcode; the exemption is the fallback SHAPE",
        )

    def test_both_drivers_keep_the_pre_rename_replay_fallback(self) -> None:
        """The tolerated shape is present, not merely tolerated in the abstract.

        An evidence tree recorded before the W6 rename names `examples/server`;
        dropping the fallback makes those trees unreplayable while the scan
        stays silent about it.
        """
        gdn = REPO_ROOT / "scripts" / "dgx-gdn-packed-component.sh"
        for path in (HARNESS_PY, HARNESS_SH, gdn):
            with self.subTest(path=path.name):
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertTrue(
                    any(p.search(line) for line in lines for p in FALLBACK_PATTERNS),
                    f"{path.name} no longer falls back to the pre-rename "
                    "examples/server, so a pre-W6 evidence tree cannot replay",
                )

    def test_repository_scan_detects_a_stale_readme_command(self) -> None:
        """Exercise README discovery and reporting without editing user files."""
        if declared_server_output_name() == "server":
            self.skipTest("the target name IS the output name; nothing to reject")
        readme = REPO_ROOT / "README.md"
        read_text = pathlib.Path.read_text

        def with_stale_readme(path, *args, **kwargs):
            if path == readme:
                return "# Quick start\nbuild/examples/server --help\n"
            return read_text(path, *args, **kwargs)

        with mock.patch.object(pathlib.Path, "read_text", with_stale_readme):
            self.assertIn(
                "README.md:2: build/examples/server --help",
                stale_server_artifact_references(),
                "the repository scan must inspect README commands",
            )
            with self.assertRaises(AssertionError):
                self.test_no_live_consumer_resolves_the_stale_artifact_path()

    def test_no_live_consumer_resolves_the_stale_artifact_path(self) -> None:
        expected = declared_server_output_name()
        if expected == "server":
            self.skipTest("the target name IS the output name; nothing to reject")
        hits = stale_server_artifact_references()
        self.assertEqual(
            [],
            hits,
            "these live files resolve the server executable at examples/server, a "
            f"path the build has not produced since it was renamed to "
            f"examples/{expected}; each one aborts or misleads at run time:\n"
            + "\n".join(hits),
        )


class NvidiaQwen27bModelKey(unittest.TestCase):
    """The `nvidia` 27B NVFP4 must have a canonical gate recipe.

    `tools/bench/online_gate.py` carried keys for `unsloth/Qwen3.6-27B-NVFP4`
    ("27") and `nvidia/Qwen3.6-35B-A3B-NVFP4` ("35"), but none for
    `nvidia/Qwen3.6-27B-NVFP4` -- the checkpoint the parity campaign targets and
    every recorded 0.85x number was taken on. Without a key it can only be
    benched by an ad-hoc harness enforcing none of the pinned revision, oracle
    inventory, cache-drop or one-lock checks, so its numbers can never become
    accepted evidence. Key "27n" closes that.

    The shell-side cases PARSE the driver's argument guard and `test_name`
    dispatch rather than searching for a substring. Substring assertions had ZERO
    coverage here: `${model} == 27n` occurs in the guard line and
    `${model} == 27 || ${model} == 27n` in the dispatch line, so each assertion
    was satisfied by the OTHER line and both survived deleting what they checked.
    """

    KEY = "27n"
    REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
    REPOSITORY = "nvidia/Qwen3.6-27B-NVFP4"
    DENSE_27B_GATE = "test_qwen27_paged_engine"

    def test_python_harness_knows_the_key(self) -> None:
        from tools.bench.online_gate import (
            MAX_MODEL_LEN,
            MAX_NUM_BATCHED_TOKENS,
            MODEL_REPOSITORIES,
            MODEL_REVISIONS,
        )

        self.assertEqual(MODEL_REVISIONS.get(self.KEY), self.REVISION)
        self.assertEqual(MODEL_REPOSITORIES.get(self.KEY), self.REPOSITORY)
        # A dense 27B shares the dense batched-token gate value with key "27";
        # only the 35B MoE prefills the wider 8192 chunk.
        self.assertEqual(MAX_NUM_BATCHED_TOKENS.get(self.KEY), 2048)
        self.assertEqual(MAX_MODEL_LEN.get(self.KEY), MAX_MODEL_LEN["27"])

    def test_key_is_distinct_from_the_unsloth_27b(self) -> None:
        """@0893e160 and @890bdef7 are different models, not two spellings."""
        from tools.bench.online_gate import MODEL_REPOSITORIES, MODEL_REVISIONS

        self.assertNotEqual(MODEL_REVISIONS["27"], MODEL_REVISIONS[self.KEY])
        self.assertNotEqual(MODEL_REPOSITORIES["27"], MODEL_REPOSITORIES[self.KEY])

    def test_shell_driver_accepts_every_python_model_key(self) -> None:
        """The guard LINE must admit 27n -- and exactly the keys the tables define."""
        from tools.bench.online_gate import MODEL_REVISIONS

        accepted = accepted_model_keys()
        self.assertIn(
            self.KEY,
            accepted,
            f"dgx-online-serving.sh rejects --model {self.KEY}; its argument guard "
            f"admits only {sorted(accepted)}",
        )
        self.assertEqual(
            frozenset(MODEL_REVISIONS),
            accepted,
            "the driver's --model guard and MODEL_REVISIONS must name the same "
            "keys, or a key exists in exactly one of the two harnesses",
        )

    def test_shell_driver_gates_it_on_the_dense_27b_correctness_test(self) -> None:
        """It must run a paged-engine golden, not fall through to the no-golden arm."""
        from tools.bench.online_gate import MODEL_REVISIONS

        resolved = correctness_gate_by_model_key(sorted(MODEL_REVISIONS))
        self.assertEqual(
            self.DENSE_27B_GATE,
            resolved.get(self.KEY),
            "the nvidia 27B must share the dense 27B paged-engine correctness gate; "
            "falling through to the else-arm would bench it with no golden at all",
        )
        # The whole dispatch, so a repair to one arm cannot silently move another.
        self.assertEqual(
            {
                "27": self.DENSE_27B_GATE,
                "27n": self.DENSE_27B_GATE,
                "35": "test_qwen36_paged_engine",
                "q3mxfp4": "mxfp4_smoke_battery",
            },
            resolved,
        )

    def test_trace_only_refusal_names_the_checkpoint_not_the_architecture(self) -> None:
        """27n IS a Qwen3.6-27B dense graph, so that cannot be the reason it is refused.

        The trace contracts in `TRACE_PRIMARY_GRAPH_CONTRACTS` are node counts
        captured on the unsloth checkpoint, so the refusal is about WHICH 27B,
        not about the architecture.
        """
        text = HARNESS_SH.read_text(encoding="utf-8")
        refusal = _unique_line(
            text, lambda line: "trace-only control is defined only" in line
        )
        self.assertIn(
            "unsloth",
            refusal,
            "the trace-only refusal must name the checkpoint its node counts were "
            f"captured on; {self.KEY} is a Qwen3.6-27B dense graph too, so naming "
            "the architecture tells the reader nothing about why it is refused",
        )


class ModelGateScopeIsRecorded(unittest.TestCase):
    """A recorded model gate must prove it compared tokens, and say for WHICH checkpoint.

    `tests/parity/test_qwen27_paged_engine.cpp` emits a loud MESSAGE and returns
    0 when its snapshot is absent, so ctest exits 0 whether the gate compared
    sixteen tokens or none. `record_model_gate` then wrote ``"passed": true``
    unconditionally and the summary checked only that flag, the key, the SHA and
    the log hash. On a box holding @0893e160 but not @890bdef7 -- the EXPECTED
    shape for key "27n" -- that marked `preflight/model-gate/27n.json` passed
    with zero tokens compared.

    Two facts have to reach the evidence: that the gate ran (a proof line it
    prints only after comparing), and which checkpoint its goldens belong to
    (for 27n, deliberately NOT the benched one). Driver prose reaches neither.
    """

    def test_every_dispatched_gate_has_a_contract(self) -> None:
        from tools.bench.online_gate import MODEL_GATE_CONTRACTS, MODEL_REVISIONS

        dispatched = set(correctness_gate_by_model_key(sorted(MODEL_REVISIONS)).values())
        self.assertEqual(
            dispatched,
            set(MODEL_GATE_CONTRACTS),
            "every test_name the driver can select must carry a proof marker and a "
            "golden-revision record, or its evidence is unfalsifiable",
        )

    def test_proof_markers_are_still_emitted_by_their_gates(self) -> None:
        """Parsed from the gate's own source, so a reworded MESSAGE fails HERE."""
        from tools.bench.online_gate import MODEL_GATE_CONTRACTS

        parity = REPO_ROOT / "tests" / "parity"
        sources = {
            "test_qwen27_paged_engine": parity / "test_qwen27_paged_engine.cpp",
            "test_qwen36_paged_engine": parity / "test_qwen36_paged_engine.cpp",
            "mxfp4_smoke_battery": REPO_ROOT / "tools/bench/mxfp4_smoke_gate.py",
        }
        self.assertEqual(set(sources), set(MODEL_GATE_CONTRACTS))
        for test_name, source in sources.items():
            with self.subTest(test_name=test_name):
                marker = MODEL_GATE_CONTRACTS[test_name]["proof"]
                self.assertEqual(
                    1,
                    source.read_text(encoding="utf-8").count(marker),
                    f"{source.name} no longer emits exactly one {marker!r}; the gate "
                    "evidence would accept a run that never compared a token",
                )

    def test_the_27b_golden_revision_tracks_the_gate_that_pins_it(self) -> None:
        """Parsed from the header the C++ gate actually resolves its snapshot with."""
        from tools.bench.online_gate import MODEL_GATE_CONTRACTS

        header = (REPO_ROOT / "tests/parity/hf_snapshot.h").read_text(encoding="utf-8")
        pinned = re.findall(
            r"kQwen27NvfP4Revision\s*=\s*\n?\s*\"([0-9a-f]{40})\"", header
        )
        self.assertEqual(1, len(pinned), f"unexpected revision pins: {pinned!r}")
        self.assertEqual(
            pinned[0],
            MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"]["golden_revision"],
            "the recorded golden revision must be the one the gate resolves, or the "
            "evidence names a checkpoint the test never loaded",
        )

    def test_driver_captures_the_gate_output_so_the_proof_reaches_the_log(self) -> None:
        """`--output-on-failure` alone prints nothing for a PASSING test."""
        text = HARNESS_SH.read_text(encoding="utf-8")
        invocation = _unique_line(text, lambda line: line.strip().startswith("ctest "))
        self.assertIn(
            " -V ",
            f" {invocation.strip()} ",
            "the recorded model-gate log must carry the test's own output, or the "
            "proof line it prints can never be checked",
        )

    def _record(self, tmp: pathlib.Path, *, model_key: str, log_text: str):
        from tools.bench.online_gate import record_model_gate

        log = tmp / f"{model_key}.log"
        log.write_text(log_text, encoding="utf-8")
        return record_model_gate(
            tmp / f"{model_key}.json",
            log=log,
            model_key=model_key,
            test_name="test_qwen27_paged_engine",
            vllm_cpp_sha="d" * 40,
        )

    def test_a_skipped_gate_is_not_a_passed_gate(self) -> None:
        import tempfile

        from tools.bench.online_gate import HarnessError

        skipped = (
            "1/1 Test #7: test_qwen27_paged_engine ...   Passed    0.31 sec\n"
            "MESSAGE: 27B checkpoint absent; skipping (dgx-only) — "
            "unsloth/Qwen3.6-27B-NVFP4 snapshot not present\n"
            "[doctest] Status: SUCCESS!\n"
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            with self.assertRaisesRegex(HarnessError, "compared no token"):
                self._record(tmp, model_key="27n", log_text=skipped)

    def test_the_benched_and_golden_checkpoints_are_both_recorded(self) -> None:
        import tempfile

        from tools.bench.online_gate import MODEL_GATE_CONTRACTS, MODEL_REVISIONS

        proof = MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"]["proof"]
        golden = MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"]["golden_revision"]
        passed = f"1/1 Test #7 ... Passed\n{proof}\n[doctest] Status: SUCCESS!\n"
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            unsloth = self._record(tmp, model_key="27", log_text=passed)
            nvidia = self._record(tmp, model_key="27n", log_text=passed)
        self.assertEqual(golden, unsloth["golden_revision"])
        self.assertEqual(MODEL_REVISIONS["27"], unsloth["model_revision"])
        self.assertIs(True, unsloth["golden_covers_benched_checkpoint"])
        # The whole point of key 27n: the goldens belong to a DIFFERENT
        # checkpoint, so the precondition is build sanity and the evidence says so
        # in a field a summary can read, not only in a shell comment.
        self.assertEqual(golden, nvidia["golden_revision"])
        self.assertEqual(MODEL_REVISIONS["27n"], nvidia["model_revision"])
        self.assertIs(False, nvidia["golden_covers_benched_checkpoint"])


if __name__ == "__main__":
    unittest.main()
