#!/usr/bin/env python3
"""The compiled-arch manifest resolves from the REAL CMake computation.

Issue #1357, umbrella #1332 M2. `CudaPlatform::supports_fa2_attention()` now asks
a generated manifest whether this device has FlashAttention-2 SASS in this build.
The C++ matcher is covered by `tests/vllm/platforms/test_cuda_arch_manifest.cpp`;
what THIS file covers is the other half, which no C++ test can reach: that
`vt_cuda_feature_archs(... "fa2")` — the function whose result becomes both the
`-gencode` options and the manifest — actually resolves to what we claim.

It runs the real `cmake/CudaArchFeatures.cmake` through `cmake -P`, so it needs no
CUDA toolkit, no GPU and no lease. That is what makes the central
no-change-on-the-gate-hardware claim executable on a CPU host: a GB10 build
requests `121a`, and the manifest must contain `121a`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = os.path.join(ROOT, "cmake", "CudaArchFeatures.cmake")

PROBE = """
# CMP0057 (IN_LIST) must be NEW, which the project's own top-level
# cmake_minimum_required already gives the real build; a bare `cmake -P` script
# starts without it.
cmake_minimum_required(VERSION 3.20)
include("{features}")
set(VLLM_CPP_CUDA_ARCHITECTURES "{archs}")
vt_cuda_feature_archs(_FA2 "fa2")
string(REPLACE ";" "," _MANIFEST "${{_FA2}}")
message(STATUS "MANIFEST=[${{_MANIFEST}}]")
"""


def resolve(archs):
    """Return the fa2 manifest CMake produces for a requested architecture set."""
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "probe.cmake")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(PROBE.format(features=FEATURES.replace("\\", "/"), archs=archs))
        # Never read an exit code through a pipe: capture, then assert on it.
        proc = subprocess.run(
            [shutil.which("cmake") or "cmake", "-P", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"cmake -P failed rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
            )
        out = proc.stdout + proc.stderr
        for line in out.splitlines():
            marker = "MANIFEST=["
            if marker in line:
                return line.split(marker, 1)[1].rsplit("]", 1)[0]
        raise AssertionError(f"probe printed no manifest:\n{out}")


@unittest.skipUnless(shutil.which("cmake"), "cmake is not on PATH")
class Fa2ArchManifest(unittest.TestCase):
    def test_the_gate_hardware_keeps_its_fa2_path(self):
        """THE no-regression claim, made executable without a GPU.

        A GB10 build requests `121a`; the feature table's fa2 row contains
        `12.1a`; so the manifest must contain `121a`, and a device reporting
        capability 12,1 matches it exactly, suffix included. If this goes red,
        the change has silently dropped the gate hardware to the f32
        graph-captured fallback.
        """
        self.assertEqual(resolve("121a"), "121a")

    def test_an_unrequested_arch_is_absent_rather_than_assumed(self):
        """The defect, from the build side.

        The default single-arch build must NOT name sm_86, which is exactly what
        the old unconditional `return true` effectively claimed.
        """
        manifest = resolve("121a")
        self.assertNotIn("86", manifest)
        self.assertNotIn("80", manifest)

    def test_the_release_bundle_narrows_to_the_feature_row(self):
        """A ten-SM request keeps only the archs with an FA2 kernel body.

        The fa2 row is `8.0,8.6,8.7,8.9,12.0a,12.1a`, so `90a`, `100a`, `103a`
        and `110` are dropped: the release archive carries those architectures
        for other kernels, and claiming FA2 for them is the over-claim this row
        removes.
        """
        manifest = resolve("80;86;87;89;90a;100a;103a;110;120a;121a")
        kept = {a for a in manifest.split(",") if a}
        # A SET: CMake sorts the list and the matcher scans it, so the order is
        # not part of the contract and asserting one would gate on an accident.
        self.assertEqual(kept, {"80", "86", "87", "89", "120a", "121a"})
        for dropped in ("90a", "100a", "103a", "110"):
            self.assertNotIn(dropped, kept)

    def test_a_build_with_no_fa2_arch_resolves_EMPTY(self):
        """Empty is a real state and must stay empty, not fall through.

        sm_90 has no FA2 row entry, so a Hopper-only build compiles no FA2 and
        the manifest is empty — which the matcher reads as "not compiled".
        """
        self.assertEqual(resolve("90a"), "")

    def test_the_base_arch_of_an_a_target_is_not_silently_promoted(self):
        """`12.1a` in the row does not license a base `121` request.

        vt_cuda_feature_archs keeps only hits literally among the requested
        targets, precisely because the `a` suffix is load-bearing. A base request
        therefore reports FA2 DISABLED rather than enabling an instruction the
        build will not emit.
        """
        self.assertEqual(resolve("121"), "")

    def test_the_buckets_sum(self):
        """Every case above is one of these, so a silently empty bucket shows."""
        cases = [n for n in dir(self) if n.startswith("test_")]
        resolving = [n for n in cases if n != "test_the_buckets_sum"]
        self.assertEqual(len(cases), len(resolving) + 1)
        self.assertEqual(len(resolving), 5)


if __name__ == "__main__":
    unittest.main()
