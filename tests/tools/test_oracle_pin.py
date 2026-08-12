"""The harness's oracle identity must equal the pin record, and must REFUSE the rollback.

Row `BENCH-ORACLE-PIN-RECONCILE`, issue #520, spec
`.agents/specs/bench-oracle-pin-reconcile.md`.

These tests exist because the previous constants were a second copy of
`.agents/upstream-sync.md` and the copy drifted for 17 days, during which the
gate did not merely prefer the old oracle -- it RAISED on the pinned one. Test 1
makes that drift impossible to reintroduce silently. Tests 2 and 3 are the #375
mutation: they fail if the commit term is weakened or deleted, which is the only
term that separates the pin from a rollback that runs, is deterministic, and
reports a perfectly clean version number.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import tempfile
import types
import unittest
from unittest import mock

import tools.bench.gdn_packed_component as gdn_component
from tools.bench.gdn_packed_component import summarize_evidence
from tools.bench.online_gate import (
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_REVISIONS,
    record_execution_manifest,
    record_oracle_manifest,
)
from tools.bench.serve_low_common import (
    FLASHINFER_VERSION,
    HarnessError,
    VLLM_COMMIT,
    VLLM_DISTRIBUTION_VERSION,
    VLLM_ORACLE_VERSION,
    assert_oracle_commit,
    read_parity_pin,
    sha256_file,
)

from tests.tools.test_gdn_packed_component import SOURCE_SHA, _write_complete_evidence

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIN_RECORD = ROOT / ".agents" / "upstream-sync.md"

# The rollback's EXACT measured strings (dgx, ~/venvs/vllm-oracle-v0.25.0-stage,
# 2026-08-12). Hardcoded on purpose: this is the artifact the harness must
# refuse, so it may not be derived from whatever the constants happen to say.
ROLLBACK_RUNTIME_VERSION = "0.25.0"
ROLLBACK_FLASHINFER_VERSION = "0.6.13"
ROLLBACK_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"


class ParityPinRecordTests(unittest.TestCase):
    def test_constants_are_the_pin_record_not_a_copy(self) -> None:
        """Re-parsed independently of the loader, so a loader bug cannot agree with itself."""

        text = PIN_RECORD.read_text(encoding="utf-8")
        blocks = re.findall(r"^```parity-pin$\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)
        self.assertEqual(len(blocks), 1, "exactly one parity-pin block must exist")
        fields = dict(
            (part.strip() for part in line.split("=", 1))
            for line in blocks[0].splitlines()
            if line.strip()
        )
        self.assertEqual(VLLM_COMMIT, fields["vllm_commit"])
        self.assertEqual(VLLM_ORACLE_VERSION, fields["vllm_runtime_version"])
        self.assertEqual(VLLM_DISTRIBUTION_VERSION, fields["vllm_distribution_version"])
        self.assertEqual(FLASHINFER_VERSION, fields["flashinfer_version"])

    def test_the_rollback_is_not_the_pin(self) -> None:
        """Guards the reconcile itself: these are the values that were enforced."""

        self.assertNotEqual(VLLM_ORACLE_VERSION, ROLLBACK_RUNTIME_VERSION)
        self.assertNotEqual(FLASHINFER_VERSION, ROLLBACK_FLASHINFER_VERSION)
        self.assertNotEqual(VLLM_COMMIT, ROLLBACK_COMMIT)

    def test_metadata_and_runtime_strings_differ_on_the_pin(self) -> None:
        """The reason `metadata == runtime == CONST` had to be replaced, pinned as a fact."""

        self.assertNotEqual(VLLM_DISTRIBUTION_VERSION, VLLM_ORACLE_VERSION)
        self.assertTrue(VLLM_DISTRIBUTION_VERSION.startswith(VLLM_ORACLE_VERSION))


class ParityPinLoaderTests(unittest.TestCase):
    def _write(self, body: str) -> pathlib.Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = pathlib.Path(directory.name) / "upstream-sync.md"
        path.write_text(body, encoding="utf-8")
        return path

    def test_missing_record_is_fatal(self) -> None:
        with self.assertRaisesRegex(HarnessError, "unreadable"):
            read_parity_pin(pathlib.Path("/nonexistent/upstream-sync.md"))

    def test_absent_block_is_fatal_not_defaulted(self) -> None:
        path = self._write("# no pin here\n")
        with self.assertRaisesRegex(HarnessError, "exactly one"):
            read_parity_pin(path)

    def test_two_blocks_are_fatal(self) -> None:
        block = "```parity-pin\nvllm_commit = " + "a" * 40 + "\n```\n"
        with self.assertRaisesRegex(HarnessError, "exactly one"):
            read_parity_pin(self._write(block + block))

    def test_unknown_field_is_fatal(self) -> None:
        path = self._write("```parity-pin\nvllm_branch = main\n```\n")
        with self.assertRaisesRegex(HarnessError, "unknown parity-pin field"):
            read_parity_pin(path)

    def test_malformed_line_is_fatal(self) -> None:
        path = self._write("```parity-pin\nvllm_commit\n```\n")
        with self.assertRaisesRegex(HarnessError, "malformed"):
            read_parity_pin(path)

    def test_missing_field_is_fatal(self) -> None:
        path = self._write("```parity-pin\nvllm_commit = " + "a" * 40 + "\n```\n")
        with self.assertRaisesRegex(HarnessError, "omits"):
            read_parity_pin(path)

    def test_abbreviated_commit_is_fatal(self) -> None:
        """A short SHA would silently weaken the prefix check below."""

        path = self._write(
            "```parity-pin\n"
            "vllm_commit = 555967922\n"
            "vllm_runtime_version = 0.1\n"
            "vllm_distribution_version = 0.1\n"
            "flashinfer_version = 0.1\n"
            "```\n"
        )
        with self.assertRaisesRegex(HarnessError, "full 40-hex SHA"):
            read_parity_pin(path)

    def test_committed_record_parses(self) -> None:
        fields = read_parity_pin(PIN_RECORD)
        self.assertEqual(fields["vllm_commit"], VLLM_COMMIT)


class OracleCommitAssertionTests(unittest.TestCase):
    """THE #375 MUTATION. Delete the commit term and every test here fails."""

    def test_the_pin_is_accepted(self) -> None:
        assert_oracle_commit(VLLM_ORACLE_VERSION)
        # The distribution string carries the same segment plus a suffix.
        assert_oracle_commit(VLLM_DISTRIBUTION_VERSION)

    def test_the_rollback_is_refused(self) -> None:
        """`0.25.0` runs, is deterministic, and looks healthy. It is not the pin."""

        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit(ROLLBACK_RUNTIME_VERSION)

    def test_a_version_with_no_commit_segment_is_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit("0.26.0.dev0")

    def test_a_different_commit_is_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit("0.23.1rc1.dev1511+ge24d1b24")

    def test_none_is_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit(None)

    def test_a_prefix_shorter_than_seven_is_refused(self) -> None:
        """Otherwise `+g5` would match, and a one-character oracle check is no check."""

        with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
            assert_oracle_commit(f"0.1+g{VLLM_COMMIT[:6]}")

    def test_a_longer_prefix_of_the_pin_is_accepted(self) -> None:
        assert_oracle_commit(f"0.1+g{VLLM_COMMIT[:12]}")
        assert_oracle_commit(f"0.1+g{VLLM_COMMIT}")


class OracleIdentityIsWiredIntoEveryEntryPointTests(unittest.TestCase):
    """The guard above is only a guarantee where it is CALLED.

    `OracleCommitAssertionTests` exercises `assert_oracle_commit` as a pure
    function, which proves what it computes and nothing about whether the three
    manifest entry points consult it. Every case here feeds a rollback-shaped
    manifest or module to an entry point and requires the entry point's OWN
    refusal, so deleting a call site or an equality is red rather than silent.

    Each case matches the exact refusal, not merely `HarnessError`. These
    functions raise it for a dozen unrelated reasons, so an unanchored
    `assertRaises` would stay green on a gutted identity check that simply
    failed later for want of a fixture file.

    The three `commit-drift` cases patch the pin's version CONSTANTS to the
    rollback's version string. That is deliberate: at today's pin the exact
    equality against `VLLM_ORACLE_VERSION` strictly dominates the commit
    assertion (the pinned string already contains `+g555967922`), so no input
    can reach the assertion while the equality holds. Patching the constants
    models the pin shape where it is the operative term -- a release-numbered
    oracle -- and is the only way to prove the call site exists at all.
    """

    def _record_oracle_manifest(
        self, *, metadata_version: str, runtime_version: str
    ) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = pathlib.Path(directory.name)
        module = types.SimpleNamespace(
            __version__=runtime_version, __file__=str(root / "vllm" / "__init__.py")
        )
        with (
            mock.patch.dict("sys.modules", {"vllm": module}),
            mock.patch(
                "tools.bench.online_gate.importlib.metadata.distribution",
                return_value=types.SimpleNamespace(version=metadata_version),
            ),
        ):
            record_oracle_manifest(root / "oracle.json", client=root / "vllm-client")

    def test_record_oracle_refuses_a_rollback_runtime_version(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle version drift"):
            self._record_oracle_manifest(
                metadata_version=VLLM_DISTRIBUTION_VERSION,
                runtime_version=ROLLBACK_RUNTIME_VERSION,
            )

    def test_record_oracle_refuses_rollback_distribution_metadata(self) -> None:
        with self.assertRaisesRegex(HarnessError, "oracle version drift"):
            self._record_oracle_manifest(
                metadata_version=ROLLBACK_RUNTIME_VERSION,
                runtime_version=VLLM_ORACLE_VERSION,
            )

    def test_record_oracle_asserts_the_commit_behind_the_version(self) -> None:
        with (
            mock.patch(
                "tools.bench.online_gate.VLLM_ORACLE_VERSION",
                ROLLBACK_RUNTIME_VERSION,
            ),
            mock.patch(
                "tools.bench.online_gate.VLLM_DISTRIBUTION_VERSION",
                ROLLBACK_RUNTIME_VERSION,
            ),
            self.assertRaisesRegex(HarnessError, "oracle commit drift"),
        ):
            self._record_oracle_manifest(
                metadata_version=ROLLBACK_RUNTIME_VERSION,
                runtime_version=ROLLBACK_RUNTIME_VERSION,
            )

    def _record_execution_manifest(self, *, runtime_version: str) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = pathlib.Path(directory.name)
        snapshot = root / MODEL_REVISIONS["27"]
        snapshot.mkdir(parents=True)
        (snapshot / "model-00001-of-00001.safetensors").write_text(
            "weights\n", encoding="utf-8"
        )
        oracle_manifest = root / "27-oracle.json"
        oracle_manifest.write_text(
            json.dumps(
                {
                    "oracle_version": VLLM_DISTRIBUTION_VERSION,
                    "runtime_version": runtime_version,
                    "client_contract_source_commit": VLLM_COMMIT,
                }
            ),
            encoding="utf-8",
        )
        record_execution_manifest(
            root / "27-component.json",
            model_key="27",
            vllm_cpp_sha="d" * 40,
            build_dir=root / "build",
            client=root / "vllm",
            snapshot=snapshot,
            configure_log=root / "configure.log",
            build_command=root / "build-command.txt",
            build_log=root / "build.log",
            oracle_manifest=oracle_manifest,
            port=8001,
            num_blocks=4736,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS["27"],
            profile_control=True,
        )

    def test_record_execution_refuses_a_rollback_shaped_oracle_manifest(self) -> None:
        """The manifest is a FILE: another venv, another day, or hand-edited."""

        with self.assertRaisesRegex(HarnessError, "oracle runtime version differs"):
            self._record_execution_manifest(runtime_version=ROLLBACK_RUNTIME_VERSION)

    def test_record_execution_asserts_the_commit_behind_the_version(self) -> None:
        with (
            mock.patch(
                "tools.bench.online_gate.VLLM_ORACLE_VERSION",
                ROLLBACK_RUNTIME_VERSION,
            ),
            self.assertRaisesRegex(HarnessError, "oracle commit drift"),
        ):
            self._record_execution_manifest(runtime_version=ROLLBACK_RUNTIME_VERSION)

    def _component_evidence(self, *, runtime_version: str) -> pathlib.Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = pathlib.Path(directory.name)
        # Same substitution the component suite makes: the evidence fixture
        # hashes the DGX's nvcc, which no CPU CI host has.
        true_binary = shutil.which("true")
        if true_binary is None:
            self.skipTest("the platform has no 'true' executable")
        compiler = mock.patch.object(
            gdn_component, "DGX_CUDA_COMPILER", pathlib.Path(true_binary)
        )
        compiler.start()
        self.addCleanup(compiler.stop)
        _write_complete_evidence(root)
        oracle_path = root / "execution" / "27-oracle.json"
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        oracle["runtime_version"] = runtime_version
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        execution_path = root / "execution" / "27-component.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        # The execution manifest records the oracle manifest's own digest, so an
        # edited oracle manifest must be re-hashed or the artifact check fires
        # first and this case would prove nothing about the identity terms.
        execution["artifacts"]["oracle_manifest"]["sha256"] = sha256_file(oracle_path)
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        return root

    def test_component_evidence_refuses_a_rollback_shaped_oracle_manifest(self) -> None:
        root = self._component_evidence(runtime_version=ROLLBACK_RUNTIME_VERSION)
        with self.assertRaisesRegex(HarnessError, "oracle manifest differs"):
            summarize_evidence(root, SOURCE_SHA)

    def test_component_evidence_asserts_the_commit_behind_the_version(self) -> None:
        with mock.patch.object(
            gdn_component, "VLLM_ORACLE_VERSION", ROLLBACK_RUNTIME_VERSION
        ):
            root = self._component_evidence(
                runtime_version=ROLLBACK_RUNTIME_VERSION
            )
            execution_path = root / "execution" / "27-component.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["vllm_oracle_version"] = ROLLBACK_RUNTIME_VERSION
            execution_path.write_text(json.dumps(execution), encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "oracle commit drift"):
                summarize_evidence(root, SOURCE_SHA)


if __name__ == "__main__":
    unittest.main()
