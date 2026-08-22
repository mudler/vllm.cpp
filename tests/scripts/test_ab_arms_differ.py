#!/usr/bin/env python3
"""The two-arm control, and the reproduction that shows why the hash is not one.

`.agents/specs/ab-arms-control.md`, #1516. The `sha256` guard
`.agents/specs/minimax-music3.md` §16.6a drew is unfalsifiable for an end-to-end
pair: the timed program is a client of the shared library, CMake writes the
build-tree RPATH into it, and two build directories therefore hash differently
whatever the source says.

`IdenticalSourceReproduction` builds that shape for real -- one SHARED library
carrying the change, one thin client linking it, two byte-identical source trees,
two build directories -- and asserts the contrast in both directions: the OLD
guard passes where the NEW control fails, and a real change passes both. It is
skipped BY NAME when the box has no `cmake` or C++ compiler, so a missing
toolchain cannot read as a pass.

`VerdictContract` pins the tool's decisions on fabricated inputs and needs no
toolchain, so the tool is never untested.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/ab-arms-differ.py"

EXIT_PASS = 0
EXIT_FATAL = 5

# One SHARED library carrying the change, one thin client that links it. This is
# `minimax-music3-gen` against `vllm::shared` (`examples/CMakeLists.txt:425-426`,
# `CMakeLists.txt:2633`) reduced to the two properties that matter: the change is
# not in the timed file, and nothing sets CMAKE_SKIP_BUILD_RPATH.
CMAKELISTS = """cmake_minimum_required(VERSION 3.16)
project(abdemo CXX)
add_library(abdemo_shared SHARED lib.cpp)
add_executable(abdemo-client main.cpp)
target_link_libraries(abdemo-client PRIVATE abdemo_shared)
"""

# DEMO_CHECK mirrors VT_CHECK (`include/vt/dtype.h:11-17`), which embeds
# `__FILE__`. CMake compiles with absolute source paths and this tree sets no
# `-ffile-prefix-map`, so the library carries its own source directory as bytes.
# That is why hashing the LIBRARY instead of the client is not the repair either:
# it is stable across two build dirs and differs across two source dirs, and
# separate clones are the shape §16.6a's own repair mandates.
LIB_SOURCE = """#include <stdexcept>
#include <string>
#define DEMO_CHECK(cond, msg)                                             \\
  do {                                                                    \\
    if (!(cond)) {                                                        \\
      throw std::runtime_error(std::string("demo: ") + (msg) + " at " +   \\
                               __FILE__ + ":" + std::to_string(__LINE__));\\
    }                                                                     \\
  } while (0)
extern "C" int depth_forward_calls(int frames) {
  DEMO_CHECK(frames > 0, "frames must be positive");
  return frames * %d;
}
"""

MAIN_SOURCE = """#include <cstdio>
extern "C" int depth_forward_calls(int frames);
int main() {
  std::printf("ar.depth_forward calls=%d\\n", depth_forward_calls(101));
  return 0;
}
"""


def run_tool(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        text=True, capture_output=True,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolchain_reason() -> str | None:
    """Why the compiled reproduction cannot run here, or ``None``."""

    if shutil.which("cmake") is None:
        return "no cmake on PATH"
    if not any(shutil.which(name) for name in ("c++", "g++", "clang++")):
        return "no C++ compiler on PATH"
    return None


class VerdictContract(unittest.TestCase):
    """Every decision, on inputs no compiler is needed to make."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)

    def artifact(self, name: str, content: bytes) -> Path:
        path = self.directory / name
        path.write_bytes(content)
        return path

    def test_two_arms_that_are_one_artifact_are_fatal(self) -> None:
        """§16.6a's original defect, which this must not soften."""
        one = self.artifact("one", b"same bytes")
        two = self.artifact("two", b"same bytes")
        result = run_tool(
            "--artifact-a", str(one), "--artifact-b", str(two),
            "--control", "calls", "1414", "808",
        )
        self.assertEqual(result.returncode, EXIT_FATAL, result.stdout)
        self.assertIn("ARMS_IDENTICAL", result.stdout)
        self.assertIn("HASHES_DIFFER=no", result.stdout)

    def test_a_hash_only_verdict_is_refused_by_name(self) -> None:
        """The whole of #1516: two different hashes, offered alone, are not
        evidence that the arms differ in the change."""
        result = run_tool(
            "--artifact-a", str(self.artifact("a", b"aaaa")),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
        )
        self.assertEqual(result.returncode, EXIT_FATAL, result.stdout)
        # The COLON matters: `NO_CONTROL_MOVED` contains `NO_CONTROL`, so a
        # substring assertion on the shorter token passes when the refusal came
        # from the wrong rule. Caught by mutating the `if not controls` branch
        # away, which left the suite green.
        self.assertIn("VERDICT=FATAL NO_CONTROL:", result.stdout)
        self.assertIn("HASHES_DIFFER=yes", result.stdout)

    def test_a_control_that_did_not_move_is_fatal(self) -> None:
        result = run_tool(
            "--artifact-a", str(self.artifact("a", b"aaaa")),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
            "--control", "ar.depth_forward", "808", "808",
        )
        self.assertEqual(result.returncode, EXIT_FATAL, result.stdout)
        self.assertIn("VERDICT=FATAL NO_CONTROL_MOVED:", result.stdout)

    def test_one_moved_control_among_several_passes(self) -> None:
        result = run_tool(
            "--artifact-a", str(self.artifact("a", b"aaaa")),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
            "--control", "vocoder.decode_window", "1", "1",
            "--control", "ar.depth_forward", "1414", "808",
        )
        self.assertEqual(result.returncode, EXIT_PASS, result.stdout)
        self.assertIn("VERDICT=PASS", result.stdout)
        self.assertIn("1 of 2 control(s) moved", result.stdout)

    def test_an_embedded_root_is_reported_with_its_offset(self) -> None:
        """The RPATH leg, stated rather than left for the reader to suspect."""
        root = str(self.directory / "bld-old")
        result = run_tool(
            "--artifact-a", str(self.artifact("a", b"xx" + root.encode())),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
            "--root-a", root,
            "--control", "calls", "1414", "808",
        )
        self.assertEqual(result.returncode, EXIT_PASS, result.stdout)
        self.assertIn("HASH_LOCATION_DEPENDENT=yes", result.stdout)
        self.assertIn("at offset 2", result.stdout)

    def test_location_dependence_never_decides_the_verdict(self) -> None:
        """It explains what the hash leg is worth. It is not itself a leg: a
        contaminated hash with a moved control is still a real pair."""
        root = str(self.directory / "bld-old")
        contaminated = run_tool(
            "--artifact-a", str(self.artifact("a", root.encode())),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
            "--root-a", root, "--control", "calls", "1414", "808",
        )
        clean = run_tool(
            "--artifact-a", str(self.artifact("c", b"cccc")),
            "--artifact-b", str(self.artifact("d", b"dddd")),
            "--root-a", root, "--control", "calls", "1414", "808",
        )
        self.assertEqual(contaminated.returncode, clean.returncode)
        self.assertIn("HASH_LOCATION_DEPENDENT=yes", contaminated.stdout)
        self.assertIn("HASH_LOCATION_DEPENDENT=no", clean.stdout)

    def test_an_unmeasured_probe_does_not_read_as_a_clean_one(self) -> None:
        """A third state, printed. An absent hook that looks armed is how a
        guard stops being one."""
        result = run_tool(
            "--artifact-a", str(self.artifact("a", b"aaaa")),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
            "--control", "calls", "1414", "808",
        )
        self.assertIn("HASH_LOCATION_DEPENDENT=UNMEASURED", result.stdout)
        self.assertNotIn("HASH_LOCATION_DEPENDENT=no", result.stdout)

    def test_a_missing_artifact_is_a_usage_error_not_a_verdict(self) -> None:
        result = run_tool(
            "--artifact-a", str(self.directory / "absent"),
            "--artifact-b", str(self.artifact("b", b"bbbb")),
            "--control", "calls", "1414", "808",
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        # `python3 <missing script>` also exits 2, so this case would survive
        # the tool being deleted. The message is what makes it a test of the
        # tool rather than of the interpreter.
        self.assertIn("not a readable file", result.stderr)


@unittest.skipIf(toolchain_reason() is not None, toolchain_reason() or "")
class IdenticalSourceReproduction(unittest.TestCase):
    """The case the guard exists to catch, BUILT rather than described.

    Two byte-identical source trees, two build directories. The old guard sees
    two hashes and passes. The new control sees a call count that did not move
    and refuses. Then the change is made for real and both pass, so the tool is
    not one that refuses everything.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.workdir = Path(tempfile.mkdtemp(prefix="ab-arms-"))
        # Equal-length arm names, so an RPATH-only difference cannot change the
        # file SIZE. That is what the reported 72 744-byte pair looked like.
        cls.arms = {}
        for arm, multiplier in (("old", 8), ("new", 8)):
            source = cls.workdir / f"src-{arm}"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(CMAKELISTS)
            (source / "lib.cpp").write_text(LIB_SOURCE % multiplier)
            (source / "main.cpp").write_text(MAIN_SOURCE)
            cls.arms[arm] = source

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def build(self, arm: str) -> Path:
        source = self.arms[arm]
        build = self.workdir / f"bld-{arm}"
        for command in (
            ["cmake", "-S", str(source), "-B", str(build),
             "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "-j", "4"],
        ):
            done = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return build

    def counts(self, build: Path) -> str:
        done = subprocess.run(
            [str(build / "abdemo-client")], text=True, capture_output=True,
            env={"LD_LIBRARY_PATH": str(build), "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.strip().split("calls=")[1]

    def test_the_old_guard_passes_where_the_new_control_fails(self) -> None:
        old_source = (self.arms["old"] / "lib.cpp").read_bytes()
        new_source = (self.arms["new"] / "lib.cpp").read_bytes()
        self.assertEqual(old_source, new_source, "the arms must be one source")

        old_build = self.build("old")
        new_build = self.build("new")
        client_old = old_build / "abdemo-client"
        client_new = new_build / "abdemo-client"

        # THE OLD GUARD, as `scripts/music3-vocoder-conv-ab.sh` wrote it before
        # this row: two hashes, and a FATAL only when they are equal.
        self.assertNotEqual(
            sha256(client_old), sha256(client_new),
            "the reproduction requires the vacuous pass; see the spec",
        )
        self.assertEqual(
            client_old.stat().st_size, client_new.stat().st_size,
            "equal sizes are what an RPATH-only difference looks like",
        )

        # Hashing the LIBRARY instead is not the repair either: two source dirs
        # give two hashes there too, through the embedded `__FILE__` path.
        self.assertNotEqual(
            sha256(old_build / "libabdemo_shared.so"),
            sha256(new_build / "libabdemo_shared.so"),
            "identical source at two paths must still hash apart",
        )

        # THE NEW CONTROL, on the same two artifacts.
        control_old = self.counts(old_build)
        control_new = self.counts(new_build)
        self.assertEqual(control_old, control_new, "identical source, identical work")
        result = run_tool(
            "--artifact-a", str(client_old), "--artifact-b", str(client_new),
            "--root-a", str(old_build), "--root-b", str(new_build),
            "--control", "ar.depth_forward", control_old, control_new,
        )
        self.assertEqual(result.returncode, EXIT_FATAL, result.stdout)
        self.assertIn("VERDICT=FATAL NO_CONTROL_MOVED:", result.stdout)
        self.assertIn("HASH_LOCATION_DEPENDENT=yes", result.stdout)

    def test_a_real_change_passes_both(self) -> None:
        """The positive control. A guard that refuses everything is not one."""
        old_build = self.build("old")
        # Registered BEFORE the edit, not written after the assertions. A
        # restore on the last line runs only when the test PASSES, which makes
        # the sibling case depend on this one's success rather than on the tree.
        self.addCleanup(
            (self.arms["new"] / "lib.cpp").write_text, LIB_SOURCE % 8
        )
        (self.arms["new"] / "lib.cpp").write_text(LIB_SOURCE % 4)
        new_build = self.build("new")
        control_old = self.counts(old_build)
        control_new = self.counts(new_build)
        self.assertNotEqual(control_old, control_new)
        result = run_tool(
            "--artifact-a", str(old_build / "abdemo-client"),
            "--artifact-b", str(new_build / "abdemo-client"),
            "--root-a", str(old_build), "--root-b", str(new_build),
            "--control", "ar.depth_forward", control_old, control_new,
        )
        self.assertEqual(result.returncode, EXIT_PASS, result.stdout)


class Reachability(unittest.TestCase):
    """A control nothing calls is a control nobody applies."""

    def test_the_committed_recipe_drives_the_tool(self) -> None:
        script = (ROOT / "scripts/music3-vocoder-conv-ab.sh").read_text()
        self.assertIn("ab-arms-differ.py", script)
        self.assertIn("--control", script)

    def test_the_benchmarking_guide_names_it_and_stops_recommending_the_hash(
        self,
    ) -> None:
        guide = (ROOT / ".agents/benchmarking.md").read_text()
        self.assertIn("scripts/ab-arms-differ.py", guide)
        self.assertIn("VT_OP_PROVIDER_DISABLE", guide)

    def test_the_spec_and_the_record_agree_on_which_leg_is_load_bearing(
        self,
    ) -> None:
        music3 = (ROOT / ".agents/specs/minimax-music3.md").read_text()
        record = (ROOT / ".agents/benchmark-record.md").read_text()
        self.assertIn("ab-arms-control.md", music3)
        self.assertIn("ab-arms-control.md", record)


if __name__ == "__main__":
    unittest.main()
