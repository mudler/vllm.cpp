#!/usr/bin/env python3
"""Mutation suite for scripts/check-device-leakage.py (the DSR ratchet, work row
`S1` of .agents/specs/accelerator-seam-audit.md).

An unpoliced checker is worse than none: a metric that silently stops catching
leaks converts a real guarantee into a green badge. So every claim the ratchet
makes is asserted here against a MUTANT — a synthetic shared-layer tree with one
deliberate defect planted in it — in the style of cmake/CudaArchFeaturesTest.cmake
(hard expectations, standalone, no CUDA toolkit and no GPU) and
tests/scripts/test_agent_record.py (mutate, then require the specific error).

The four things that must be true, and are each proven by a mutant below:

  1. A device reference ADDED to the shared layer FAILS the check (M1-M6, M18).
  2. A platform-leg site does NOT fail it — adding a backend must never trip the
     ratchet (M7) — but the per-file allowance is an EXACT budget, so it cannot
     be used as a blanket exemption for that file (M8, M9).
  3. A REDUCTION also fails until the baseline is lowered in the same commit,
     which is what makes it a ratchet rather than a threshold (M10-M12).
  4. The escape hatch is real, bounded and LOUD (M13-M15).

    python3 tests/scripts/test_device_leakage.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-device-leakage.py"
SPEC = importlib.util.spec_from_file_location("device_leakage", CHECKER)
assert SPEC is not None and SPEC.loader is not None
dl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dl
SPEC.loader.exec_module(dl)


# --- a minimal synthetic shared layer ----------------------------------------
# Small enough to reason about exactly, shaped like the real tree: one model file
# (pure shared layer), one platform leg and one registrar (both allowlisted), and
# one device-leg file that must not be scanned at all.
BASE_TREE: dict[str, str] = {
    "src/vllm/model_executor/models/toy.cpp": (
        "#include \"vt/ops.h\"\n"
        "namespace vllm {\n"
        "bool FastPath(Dev d) {\n"
        "  return d.q.device.type == vt::DeviceType::kCUDA;\n"
        "}\n"
        "}  // namespace vllm\n"
    ),
    "src/vllm/platforms/cuda.cpp": (
        "#include <cuda_runtime.h>\n"
        "DeviceType device_type() const override { return DeviceType::kCUDA; }\n"
        "Backend& backend() const { return vt::GetBackend(DeviceType::kCUDA); }\n"
        "void reg() { RegisterPlatform(DeviceType::kCUDA, &platform); }\n"
    ),
    "src/vllm/platforms/platform.cpp": (
        "static const DeviceType kOrder[] = {DeviceType::kCUDA, DeviceType::kCPU};\n"
    ),
    "include/vllm/platforms/interface.h": (
        "bool is_cuda() const { return device_type() == DeviceType::kCUDA; }\n"
    ),
    "src/vllm/v1/attention/backend.cpp": (
        "const AttentionBackendRegistrar kA{vt::DeviceType::kCUDA, \"TRITON_MLA\", f};\n"
        "const AttentionBackendRegistrar kB{vt::DeviceType::kCUDA, \"FLASH_ATTN\", g};\n"
    ),
    "src/vllm/v1/attention/backends/gdn_attn.cpp": (
        "const AttentionBackendRegistrar kG{vt::DeviceType::kCUDA, \"GDN_ATTN\", h};\n"
    ),
    # A device leg. NOT part of the scanned shared layer: this is exactly where
    # device-specific code is SUPPOSED to live, and counting it would invert the
    # metric's meaning.
    "src/vt/cuda/marlin.cu": (
        "void Marlin() { Use(vt::DeviceType::kCUDA); }\n"
        "#ifdef VT_MARLIN_NVFP4\n#endif\n"
    ),
}

# The synthetic tree's own DSR: one `kCUDA` in toy.cpp; everything else is either
# allowlisted platform leg or outside the scanned roots.
BASE_BUCKETS = {"kcuda": 1, "is_cuda": 0, "cuda_inc": 0, "vt_ifdef": 0}


class Tree:
    """A synthetic tree plus a baseline file, both disposable."""

    def __init__(self, files: dict[str, str], buckets: dict[str, int] | None = None):
        self.dir = Path(tempfile.mkdtemp(prefix="dsr-mutant-"))
        for rel, text in files.items():
            path = self.dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.baseline = self.dir / "baseline.json"
        self._prev_baseline = dl.BASELINE_PATH
        dl.BASELINE_PATH = self.baseline
        if buckets is None:
            buckets = BASE_BUCKETS
        self.set_baseline(buckets)

    def set_baseline(self, buckets: dict[str, int]) -> None:
        import json

        self.baseline.write_text(
            json.dumps({"total": sum(buckets.values()), "buckets": buckets}),
            encoding="utf-8",
        )

    def run(self, *extra: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = dl.main(["--root", str(self.dir), *extra])
        return rc, out.getvalue(), err.getvalue()

    def scan(self):
        return dl.scan(self.dir)

    def write(self, rel: str, text: str) -> None:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append(self, rel: str, text: str) -> None:
        path = self.dir / rel
        path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")

    def close(self) -> None:
        dl.BASELINE_PATH = self._prev_baseline
        shutil.rmtree(self.dir, ignore_errors=True)


class DsrRatchetMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Tree(dict(BASE_TREE))
        self.addCleanup(self.tree.close)

    def require(self, text: str, needle: str) -> None:
        if needle not in text:
            raise AssertionError(f"missing expected message {needle!r} in:\n{text}")

    # --- sanity: the unmutated tree passes -----------------------------------

    def test_M0_unmutated_tree_passes(self) -> None:
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "ratchet holds")

    # --- 1. an ADDED device reference must FAIL ------------------------------

    def test_M1_planted_kcuda_in_shared_layer_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool Extra(Dev d) { return d.q.device.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'kcuda': 2 > baseline 1")

    def test_M2_planted_is_cuda_call_site_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool E(Dev d) { return GetPlatform(d.q.device.type).is_cuda(); }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'is_cuda': 1 > baseline 0")

    def test_M3_unconditional_cuda_include_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp", '#include "vt/cuda/marlin.h"\n'
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'cuda_inc': 1 > baseline 0")

    def test_M4_guarded_cuda_include_does_not_count(self) -> None:
        # The distinction the metric exists to make: an include under a CUDA/VT_*
        # guard still compiles on a non-CUDA platform, so it is not leakage.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            '#ifdef VLLM_CPP_CUDA\n#include "vt/cuda/marlin.h"\n#endif\n',
        )
        self.assertEqual(self.tree.scan().counts["cuda_inc"], 0)

    def test_M5_cuda_include_in_the_portable_else_arm_counts(self) -> None:
        # The negative arm of a CUDA guard is the PORTABLE build; a CUDA include
        # there breaks it exactly as an unguarded one would.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            '#ifdef VLLM_CPP_CUDA\n#else\n#include "vt/cuda/marlin.h"\n#endif\n',
        )
        self.assertEqual(self.tree.scan().counts["cuda_inc"], 1)

    def test_M6_planted_vt_build_gate_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "#ifdef VT_CUTLASS_NVFP4\nint x;\n#endif\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'vt_ifdef': 1 > baseline 0")

    def test_M18_leak_hidden_behind_a_helper_still_counts(self) -> None:
        # Risk 2 of the audit: the metric must not be gameable by moving the test
        # into a helper — the helper lives in the shared layer and is counted.
        self.tree.write(
            "include/vllm/model_executor/models/toy_helper.h",
            "inline bool OnCuda(Dev d) { return d.q.device.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'kcuda': 2 > baseline 1")

    # --- 2. the allowlist exempts the platform leg, EXACTLY ------------------

    def test_M7_adding_a_backend_leg_does_not_trip_the_ratchet(self) -> None:
        # `platforms/cuda.cpp` is allowlisted "*": it IS the CUDA platform. Adding
        # a method to it — or a whole new platform file — must stay green, or the
        # metric would punish the very additivity it exists to protect.
        self.tree.append(
            "src/vllm/platforms/cuda.cpp",
            "bool more() { return t == DeviceType::kCUDA; }\n",
        )
        self.tree.write(
            "src/vllm/platforms/metal.cpp",
            "DeviceType device_type() const { return DeviceType::kMETAL; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 0, err)

    def test_M8_extra_reference_in_a_budgeted_allowlist_file_fails(self) -> None:
        # backend.cpp is allowed EXACTLY its 2 registrar keys. A third `kCUDA` is
        # not a registrar key, and a per-file allowlist that swallowed it would be
        # a blanket exemption — which is how ratchets rot.
        self.tree.append(
            "src/vllm/v1/attention/backend.cpp",
            "bool Hack(Dev d) { return d.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE: src/vllm/v1/attention/backend.cpp")
        self.require(err, "has 3 reference(s) but the allowlist expects exactly 2")

    def test_M9_removing_an_allowlisted_registrar_fails_until_declared(self) -> None:
        # The allowlist is a ratchet in both directions: it may not stay wider
        # than the platform leg it describes.
        self.tree.write(
            "src/vllm/v1/attention/backend.cpp",
            "const AttentionBackendRegistrar kA{vt::DeviceType::kCUDA, \"TRITON_MLA\", f};\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE")
        self.require(err, "has 1 reference(s) but the allowlist expects exactly 2")

    # --- 3. monotonic decrease ------------------------------------------------

    def test_M10_a_reduction_fails_until_the_baseline_is_lowered(self) -> None:
        self.tree.write(
            "src/vllm/model_executor/models/toy.cpp",
            "bool FastPath(Dev d) { return vt::OpRegistered(kOp, d.q.device); }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR baseline STALE in bucket 'kcuda': 0 < baseline 1")
        self.require(err, "must lower the baseline in the SAME")

    def test_M11_lowering_the_baseline_in_the_same_change_passes(self) -> None:
        self.tree.write(
            "src/vllm/model_executor/models/toy.cpp",
            "bool FastPath(Dev d) { return vt::OpRegistered(kOp, d.q.device); }\n",
        )
        rc, _out, err = self.tree.run("--write-baseline")
        self.assertEqual(rc, 0, err)
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR 0 == baseline 0")

    def test_M12_write_baseline_refuses_to_ratchet_upward(self) -> None:
        # The one move the ratchet must never make, even when asked directly.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool Extra(Dev d) { return d.q.device.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run("--write-baseline")
        self.assertEqual(rc, 1)
        self.require(err, "REFUSING to write a HIGHER baseline")

    # --- 4. the escape hatch: real, bounded, and LOUD ------------------------

    def test_M13_dsr_allow_on_the_same_line_exempts_and_is_printed(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool E(Dev d) { return d.t == vt::DeviceType::kCUDA; }"
            "  // DSR-ALLOW(S6): pending the OpRegistered migration\n",
        )
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR-ALLOW exemptions in force: 1")
        self.require(out, "pending the OpRegistered migration")

    def test_M14_dsr_allow_on_the_preceding_line_exempts(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "// DSR-ALLOW(S3): capability field not yet on Platform\n"
            "bool E(Dev d) { return d.t == vt::DeviceType::kCUDA; }\n",
        )
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR-ALLOW exemptions in force: 1")

    def test_M15_malformed_dsr_allow_does_not_exempt(self) -> None:
        # No row id and no reason means no traceable owner, so it must not buy an
        # exemption — otherwise the hatch degrades into a magic comment.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool E(Dev d) { return d.t == vt::DeviceType::kCUDA; }  // DSR-ALLOW\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'kcuda'")

    # --- composition: prose is not leakage, code is --------------------------

    def test_M16_device_name_in_a_line_comment_is_not_counted(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "// historical note: this used to branch on kCUDA before row M3a\n",
        )
        self.assertEqual(self.tree.scan().counts["kcuda"], 1)

    def test_M17_device_name_in_a_block_comment_or_string_is_not_counted(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "/* kCUDA and is_cuda() described in prose */\n"
            'void C() { VT_CHECK(ok, "this kernel needs kCUDA"); }\n',
        )
        counts = self.tree.scan().counts
        self.assertEqual(counts["kcuda"], 1)
        self.assertEqual(counts["is_cuda"], 0)

    def test_M19_device_legs_under_src_vt_are_not_scanned(self) -> None:
        # src/vt/cuda|metal|vulkan are device legs by definition. If they were
        # scanned, the metric would count the kernels themselves and be useless.
        self.tree.append(
            "src/vt/cuda/marlin.cu",
            "void More() { Use(vt::DeviceType::kCUDA); }\n#ifdef VT_CUTLASS_FP8\n#endif\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 0, err)


class RealTreeTests(unittest.TestCase):
    """Hard expectations about THIS repository, not a synthetic mutant."""

    def test_baseline_matches_the_tree_exactly(self) -> None:
        # This is the CI gate itself, asserted here too so a local run catches it
        # before the push.
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = dl.main([])
        self.assertEqual(rc, 0, out.getvalue() + err.getvalue())

    def test_every_allowlisted_path_exists(self) -> None:
        # A stale allowlist entry is a silent exemption for a file that may later
        # be recreated with different contents.
        for rel in dl.ALLOWLIST:
            self.assertTrue((ROOT / rel).is_file(), f"allowlisted path missing: {rel}")

    def test_allowlist_entries_all_state_a_reason(self) -> None:
        for rel, buckets in dl.ALLOWLIST.items():
            for bucket, (_count, reason) in buckets.items():
                self.assertIn(bucket, dl.BUCKETS, f"{rel}: unknown bucket {bucket}")
                self.assertGreater(
                    len(reason), 40, f"{rel}[{bucket}] allowlist reason is not a reason"
                )

    def test_leakage_is_concentrated_in_one_model_file(self) -> None:
        # The audit's headline finding, asserted as an executable fact rather than
        # a claim in a document: the shared layer's device branching is not spread
        # thin, it is piled into the model TU that has no `layers/` library to put
        # it in. If this ever stops being true, the S4/S7 plan is mis-aimed.
        res = dl.scan(ROOT)
        hot = "src/vllm/model_executor/models/qwen3_5.cpp"
        self.assertIn(hot, res.per_file)
        self.assertGreater(sum(res.per_file[hot].values()) * 2, res.total)


if __name__ == "__main__":
    unittest.main()
