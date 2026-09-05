#!/usr/bin/env python3
"""Real NumPy output checks with a controlled fake vLLM, never a GPU oracle."""

import json
import os
from pathlib import Path
import shutil
import unittest
import zipfile

import numpy as np

from test_qwen3_capture_tools import CaptureFixture, sha


class CaptureOutputTests(CaptureFixture):
    def capture(self, *extra, env=None):
        result = self.run_script([*self.args(), "--per-prompt", *extra], env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((self.out / "oracle-provenance.json").read_text())

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_ten_runs_publish_numpy_and_actual_provenance(self):
        external = self.root / "evidence/capture.json"
        for dtype in ("auto", "bfloat16", "fp8_e4m3"):
            provenance = self.capture("--kv-cache-dtype", dtype, "--seed", "17",
                                      "--provenance-out", str(external))
            ids = np.load(self.out / "greedy_ids.npy", allow_pickle=False)
            dist = np.load(self.out / "greedy_dist.npy", allow_pickle=False)
            self.assertEqual(ids.dtype, np.dtype("<i4"))
            self.assertEqual(ids.shape, (16, 2))
            self.assertEqual(dist.shape, (16, 2, 10))
            np.testing.assert_array_equal(ids[:, 0], np.arange(700, 716))
            for repeat in range(10):
                np.testing.assert_array_equal(dist[:, :, repeat], ids)
            self.assertEqual(provenance["regime"], "qwen3_5_strict")
            self.assertTrue(provenance["deterministic"])
            self.assertEqual(provenance["repetitions"], 10)
            self.assertEqual(provenance["runtime"]["revision"], self.revision)
            self.assertEqual(provenance["runtime"]["wheel"]["sha256"], sha(self.wheel))
            self.assertEqual(provenance["runtime"]["image"]["source"], "launcher_attestation")
            self.assertIn("not independently", provenance["runtime"]["image"]["verification"])
            self.assertEqual(provenance["model"]["files"]["config.json"]["sha256"],
                             sha(self.model / "config.json"))
            self.assertEqual(provenance["cache"]["requested"], dtype)
            self.assertEqual(provenance["cache"]["resolved"], "bfloat16" if dtype == "auto" else dtype)
            self.assertIsNone(provenance["cache"]["physical_dtype"])
            self.assertEqual(provenance["outputs"]["greedy_ids.npy"]["sha256"],
                             sha(self.out / "greedy_ids.npy"))
            self.assertEqual(provenance["batching"], {"batch_size": 1, "concurrency": 1})
            self.assertEqual(provenance["arguments"]["seed"], 17)
            self.assertEqual(len(provenance["prompts_sha256"]), 64)
            self.assertEqual(external.read_bytes(), (self.out / "oracle-provenance.json").read_bytes())
            calls = self.calls()
            self.assertFalse(calls[0]["llm"]["enforce_eager"])
            self.assertEqual(calls[0]["llm"]["kv_cache_dtype"], dtype)
            self.assertEqual(calls[0]["llm"]["revision"], "a" * 40)
            self.assertEqual(calls[0]["llm"]["tokenizer_revision"], "a" * 40)
            self.assertEqual(calls[0]["llm"]["seed"], 17)
            self.assertEqual(len(calls), 161)
            self.assertEqual(calls[1]["sampling"]["max_tokens"], 2)
            self.assertEqual(calls[1]["sampling"]["seed"], 17)
            shutil.rmtree(self.out)
            external.unlink()
            self.log.unlink()

    def test_nondeterminism_and_prompt_drift_write_nothing(self):
        for variable in ("FAKE_TOKEN_DRIFT", "FAKE_PROMPT_DRIFT"):
            result = self.run_script([*self.args(), "--per-prompt"], env={variable: "1"})
            self.assert_refused(result, "NONDETERMINISTIC")

    def test_short_outputs_use_the_same_padding_in_both_arrays(self):
        self.capture(env={"FAKE_SHORT": "1"})
        ids = np.load(self.out / "greedy_ids.npy", allow_pickle=False)
        dist = np.load(self.out / "greedy_dist.npy", allow_pickle=False)
        np.testing.assert_array_equal(ids[:, -1], -1)
        np.testing.assert_array_equal(dist[:, -1, :], -1)

    def test_normalized_sampling_records_the_upstream_seed_sentinel(self):
        provenance = self.capture("--seed", "-1")
        self.assertIsNone(provenance["sampling"]["seed"])
        self.assertIsNone(provenance["sampling_normalized"]["seed"])
        self.assertIsNone(provenance["sampling_resolved"])
        self.assertEqual(provenance["arguments"]["seed"], -1)

    def test_legacy_replacement_and_manifestless_teacher_forcing_remain_usable(self):
        (self.model / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        self.write_metadata()
        args = ["--model", str(self.model), "--out-dir", str(self.out), "--max-tokens", "2", "--runs", "2", "--per-prompt"]
        for repeat in range(2):
            result = self.run_script(args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (self.out / "oracle-provenance.json").unlink()
        our = np.load(self.out / "greedy_ids.npy", allow_pickle=False)
        our.astype("<i4").tofile(self.out / "our_ids.i32")
        self.log.unlink()
        result = self.run_script(["--model", str(self.model), "--golden-dir", str(self.out), "--max-tokens", "2"], near=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 17)
        manifest = json.loads((self.out / "neartie-provenance.json").read_text())
        self.assertEqual(manifest["regime"], "legacy_distributional")
        self.assertEqual(manifest["provenance_status"], "incomplete")

    def test_observed_cache_and_model_identity_must_agree(self):
        self.assert_refused(self.run_script([*self.args(), "--kv-cache-dtype", "bfloat16"],
                                           env={"FAKE_CACHE_DTYPE": "fp8_e4m3"}), "ARTIFACT_MISMATCH")
        self.assert_refused(self.run_script(env={"FAKE_IDENTITY_DRIFT": "1"}), "ARTIFACT_MISMATCH")

    def test_existing_outputs_are_not_overwritten(self):
        self.out.mkdir()
        sentinel = self.out / "greedy_ids.npy"
        sentinel.write_bytes(b"existing evidence")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"existing evidence")

    def test_explicit_default_manifest_path_is_allowed(self):
        provenance = self.capture("--provenance-out", str(self.out / "oracle-provenance.json"))
        self.assertEqual(provenance["regime"], "qwen3_5_strict")

    def test_publication_failure_rolls_back_new_outputs(self):
        blocker = self.root / "blocker"
        blocker.write_bytes(b"existing file")
        result = self.run_script([*self.args(), "--per-prompt", "--provenance-out", str(blocker / "manifest.json")])
        self.assert_refused(result, "ARTIFACT_MISMATCH")
        self.assertEqual(blocker.read_bytes(), b"existing file")

    def test_capture_rechecks_artifacts_package_and_scripts_after_generation(self):
        script = self.project / "scripts/qwen3-oracle-capture.py"
        for variable, path in (("FAKE_MUTATE_MODEL", self.model / "model.safetensors"),
                               ("FAKE_MUTATE_PACKAGE", self.source / "vllm/__init__.py"),
                               ("FAKE_MUTATE_SCRIPT", script)):
            original = path.read_bytes()
            result = self.run_script([*self.args(), "--per-prompt"],
                                     env={variable: str(script) if variable == "FAKE_MUTATE_SCRIPT" else "1"})
            self.assert_refused(result, "ARTIFACT_MISMATCH")
            path.write_bytes(original)

    def test_installed_wheel_vcs_prefix_is_verified_and_qualified(self):
        # Normal wheel builds can record +g<short SHA> with __commit_id__=None.
        # This fixture proves metadata verification, never wheel gateability.
        (self.source / ".git").rename(self.root / "saved-source-git")
        package = self.source / "vllm/__init__.py"
        original = package.read_text()
        metadata = self.source / "vllm-0.28.1.dist-info/METADATA"
        metadata.parent.mkdir()
        for prefix, accepted in ((self.revision[:7], True), (self.revision[:9], True),
                                 (self.revision, True), ("0" * 9, False)):
            version = "0.28.1rc1.dev132+g" + prefix
            package.write_text(original.replace("controlled-test-fixture", version))
            metadata.write_text(f"Name: vllm\nVersion: {version}\n")
            with zipfile.ZipFile(self.wheel, "w") as archive:
                archive.write(package, "vllm/__init__.py")
                archive.write(metadata, "vllm-0.28.1.dist-info/METADATA")
            runtime = json.loads(self.runtime_manifest.read_text())
            runtime["wheel_sha256"] = sha(self.wheel)
            self.runtime_manifest.write_text(json.dumps(runtime))
            if accepted:
                provenance = self.capture()
                self.assertEqual(provenance["runtime"]["revision"], prefix)
                self.assertEqual(provenance["runtime"]["revision_verification"],
                                 "installed_version_vcs_prefix")
                np.load(self.out / "greedy_ids.npy", allow_pickle=False).tofile(self.out / "our_ids.i32")
                result = self.run_script(near=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                near = json.loads((self.out / "neartie-provenance.json").read_text())
                self.assertEqual(near["runtime"]["revision"], prefix)
                for name in ("our_ids.npy", "neartie_gap_mnats.npy", "neartie-provenance.json"):
                    (self.out / name).unlink()
                # The same verified bytes can be captured from clean source,
                # then consumed from an installed wheel with a shorter prefix.
                provenance["runtime"].update(revision=self.revision,
                                             revision_verification="clean_git_source")
                (self.out / "oracle-provenance.json").write_text(json.dumps(provenance))
                result = self.run_script(near=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for name in ("our_ids.npy", "neartie_gap_mnats.npy", "neartie-provenance.json"):
                    (self.out / name).unlink()
                provenance["runtime"].update(revision=self.revision[:8],
                                             revision_verification="installed_version_vcs_prefix")
                (self.out / "oracle-provenance.json").write_text(json.dumps(provenance))
                self.assert_publication_refused(self.args(True), near=True)
                shutil.rmtree(self.out)
            else:
                self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def prepare_neartie(self):
        self.capture()
        our = np.load(self.out / "greedy_ids.npy", allow_pickle=False)
        our[0, 0] = 9999
        our.astype("<i4").tofile(self.out / "our_ids.i32")
        self.log.unlink()
        return our

    def test_teacher_forcing_uses_our_prefix_and_preserves_gap_units(self):
        our = self.prepare_neartie()
        result = self.run_script(near=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        gaps = np.load(self.out / "neartie_gap_mnats.npy", allow_pickle=False)
        self.assertEqual(gaps.dtype, np.dtype("<i4"))
        self.assertEqual(gaps[0, 0], 99_999_000)
        np.testing.assert_array_equal(gaps[1:, :], 250)
        np.testing.assert_array_equal(np.load(self.out / "our_ids.npy", allow_pickle=False), our)
        calls = self.calls()
        self.assertEqual(len(calls), 161)
        self.assertEqual(calls[1]["prompts"]["prompt_token_ids"], [500, 600, 9999, 700])
        self.assertEqual(calls[1]["sampling"]["max_tokens"], 1)
        self.assertEqual(calls[1]["sampling"]["prompt_logprobs"], 20)

    def test_neartie_refuses_cache_mode_seed_and_token_mismatches(self):
        self.prepare_neartie()
        for extra in (("--kv-cache-dtype", "fp8_e4m3"), ("--execution-mode", "eager"),
                      ("--seed", "19"), ("--max-tokens", "3")):
            result = self.run_script([*self.args(True), *extra], near=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ARTIFACT_MISMATCH", result.stderr)
            self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())

    def test_neartie_refuses_input_hash_drift_and_missing_provenance(self):
        self.prepare_neartie()
        path = self.out / "p0_prompt.i32"
        original = path.read_bytes()
        path.write_bytes(np.array([500, 601], dtype="<i4").tobytes())
        result = self.run_script(near=True)
        self.assertIn("ARTIFACT_MISMATCH", result.stderr)
        path.write_bytes(original)
        (self.out / "oracle-provenance.json").unlink()
        result = self.run_script(near=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ARTIFACT_MISMATCH", result.stderr)

    def test_neartie_refuses_changed_capture_identity_metadata(self):
        self.prepare_neartie()
        path = self.out / "oracle-provenance.json"
        original = path.read_text()
        changes = (("regime", "legacy_distributional"), ("deterministic", False),
                   ("cache.resolved", "fp8_e4m3"), ("runtime.requested_revision", "b" * 40),
                   ("runtime.wheel.sha256", "0" * 64), ("runtime.image.digest", "sha256:" + "0" * 64),
                   ("model.requested_revision", "b" * 40))
        for key, value in changes:
            data = json.loads(original)
            target = data
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            path.write_text(json.dumps(data))
            result = self.run_script(near=True)
            self.assertNotEqual(result.returncode, 0, key)
            self.assertIn("ARTIFACT_MISMATCH", result.stderr, key)
            self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())
        path.write_text(original)

    def test_neartie_does_not_replace_existing_strict_outputs(self):
        self.prepare_neartie()
        sentinel = self.out / "neartie_gap_mnats.npy"
        sentinel.write_bytes(b"existing near-tie evidence")
        result = self.run_script(near=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"existing near-tie evidence")

    def test_neartie_refuses_nonfinite_and_sub_millinat_drift(self):
        self.prepare_neartie()
        for variable, reason in (("FAKE_NONFINITE", "NONFINITE"),
                                 ("FAKE_GAP_DRIFT", "NONDETERMINISTIC")):
            result = self.run_script(near=True, env={variable: "1"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(reason, result.stderr)
            self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())

    def test_neartie_rechecks_its_runtime_and_input_bytes(self):
        self.prepare_neartie()
        script = self.project / "scripts/qwen3-neartie-gap.py"
        for variable, path in (("FAKE_MUTATE_MODEL", self.model / "model.safetensors"),
                               ("FAKE_MUTATE_PACKAGE", self.source / "vllm/__init__.py"),
                               ("FAKE_MUTATE_SCRIPT", script),
                               ("FAKE_MUTATE_INPUT", self.out / "our_ids.i32")):
            original = path.read_bytes()
            result = self.run_script(near=True, env={variable: str(path) if variable in
                                                   ("FAKE_MUTATE_SCRIPT", "FAKE_MUTATE_INPUT") else "1"})
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("ARTIFACT_MISMATCH", result.stderr)
            self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())
            path.write_bytes(original)

    def test_legacy_distribution_is_preserved_but_cannot_feed_strict_capture(self):
        config = self.model / "config.json"
        config.write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        self.write_metadata()
        result = self.run_script(["--model", str(self.model), "--out-dir", str(self.out),
                                  "--max-tokens", "2", "--runs", "10", "--per-prompt"],
                                 env={"FAKE_TOKEN_DRIFT": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        provenance = json.loads((self.out / "oracle-provenance.json").read_text())
        self.assertEqual(provenance["regime"], "legacy_distributional")
        self.assertFalse(provenance["deterministic"])
        self.assertIsNone(provenance["runtime"]["wheel"]["sha256"])
        # A caller cannot relabel this capture by selecting a strict model later.
        config.write_text(json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}))
        self.write_metadata()
        np.zeros((16, 2), dtype="<i4").tofile(self.out / "our_ids.i32")
        result = self.run_script(near=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ARTIFACT_MISMATCH", result.stderr)

    def assert_publication_refused(self, args, *, near=False, protected=(), external=None):
        before = {p.name: p.read_bytes() for p in self.out.iterdir()} if self.out.exists() else {}
        originals = {p: p.read_bytes() for p in protected}
        result = self.run_script(args, near=near)
        try:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("ARTIFACT_MISMATCH", result.stderr)
            self.assertEqual({p: p.read_bytes() for p in protected}, originals)
            self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}
                             if self.out.exists() else {}, before)
            if external:
                self.assertFalse(external.exists())
        finally:
            # Keep red runs and independent mutations isolated between cases.
            for path, data in originals.items():
                path.write_bytes(data)
            if self.out.exists():
                for path in self.out.iterdir():
                    if path.name not in before:
                        path.unlink()
                for name, data in before.items():
                    (self.out / name).write_bytes(data)
            if external:
                external.unlink(missing_ok=True)

    def test_sampling_records_constructor_state_without_claiming_request_resolution(self):
        capture = self.capture("--seed", "-1")
        self.assertIsNone(capture["sampling_normalized"]["seed"])
        self.assertIsNone(capture["sampling_resolved"])
        self.assertEqual(capture["sampling_resolution"]["status"], "unobserved")
        self.assertIn("clone", capture["sampling_resolution"]["limit"])
        np.load(self.out / "greedy_ids.npy", allow_pickle=False).tofile(self.out / "our_ids.i32")
        result = self.run_script([*self.args(True), "--seed", "-1"], near=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        near = json.loads((self.out / "neartie-provenance.json").read_text())
        self.assertEqual(near["sampling_normalized"], capture["sampling_normalized"])
        for key in ("sampling", "teacher_forcing_sampling"):
            self.assertIsNone(near[key + "_resolved"])
            self.assertEqual(near[key + "_resolution"]["status"], "unobserved")
            self.assertIn("clone", near[key + "_resolution"]["limit"])
        self.assertEqual(near["teacher_forcing_sampling_normalized"]["prompt_logprobs"], 20)

    def test_neartie_refuses_changed_constructor_sampling_state(self):
        self.prepare_neartie()
        path = self.out / "oracle-provenance.json"
        capture = json.loads(path.read_text())
        # A differing field outside the three-item summary must also refuse.
        capture["sampling_normalized"] = {**capture.get("sampling_normalized", {}), "ignore_eos": True}
        path.write_text(json.dumps(capture))
        result = self.run_script(near=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ARTIFACT_MISMATCH", result.stderr)
        self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())

    def test_neartie_requires_complete_observed_capture_identity(self):
        self.prepare_neartie()
        path = self.out / "oracle-provenance.json"
        original = path.read_text()
        changes = (("provenance_status", "incomplete"), ("provenance_status", None),
                   ("model.missing", ["artifact revision: config.json"]), ("model.missing", None),
                   ("runtime.missing", ["observed source or installed VCS revision"]),
                   ("runtime.missing", None), ("runtime.revision", "b" * 40),
                   ("runtime.revision", None), ("runtime.revision", self.revision[:6]),
                   ("runtime.revision_verification", None),
                   ("runtime.revision_verification", "unverified"),
                   ("runtime.revision_verification", "installed_version_vcs_prefix"),
                   ("runtime.version", None), ("runtime.version", "different-runtime-version"),
                   ("resolved_model_identity", None),
                   ("resolved_model_identity.model_type", "qwen3"),
                   ("resolved_model_identity.architectures", ["Qwen3ForCausalLM"]),
                   ("resolved_model_identity.text_model_type", "qwen3"))
        external = self.root / "near.json"
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = json.loads(original)
                target = data
                parts = key.split(".")
                for part in parts[:-1]:
                    target = target[part]
                if value is None:
                    del target[parts[-1]]
                else:
                    target[parts[-1]] = value
                path.write_text(json.dumps(data))
                self.assert_publication_refused([*self.args(True), "--provenance-out", str(external)],
                                                near=True, external=external)
        path.write_text(original)

    def prepare_legacy_neartie(self):
        (self.model / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        self.write_metadata()
        result = self.run_script(["--model", str(self.model), "--out-dir", str(self.out),
                                  "--max-tokens", "2", "--runs", "1", "--per-prompt"])
        self.assertEqual(result.returncode, 0, result.stderr)
        np.load(self.out / "greedy_ids.npy", allow_pickle=False).tofile(self.out / "our_ids.i32")
        # Legacy teacher forcing needs neither the manifest nor distribution.
        for name in ("oracle-provenance.json", "greedy_dist.npy"):
            (self.out / name).unlink()

    def test_legacy_neartie_provenance_preserves_every_input_and_alias(self):
        self.prepare_legacy_neartie()
        for source in sorted(self.out.iterdir()):
            for alias in (False, True):
                with self.subTest(source=source.name, alias=alias):
                    destination = self.root / "input-link" if alias else source
                    if alias:
                        destination.symlink_to(source)
                    try:
                        self.assert_publication_refused(["--model", str(self.model), "--golden-dir", str(self.out),
                                                         "--max-tokens", "2", "--provenance-out", str(destination)],
                                                        near=True)
                    finally:
                        if alias:
                            destination.unlink()

    def test_legacy_publication_preserves_capture_inputs(self):
        (self.model / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        self.write_metadata()
        inputs = [self.model / "config.json", self.model / "model.safetensors",
                  self.model / ".cache/huggingface/download/config.json.metadata",
                  self.wheel, self.runtime_manifest, self.source / "vllm/__init__.py",
                  *(self.project / "scripts" / name for name in
                    ("qwen3-oracle-capture.py", "qwen3-neartie-gap.py", "qwen3_oracle_common.py")),
                  self.project / "tests/parity/test_qwen35_paged_engine.cpp"]
        for near in (False, True):
            if near:
                self.prepare_legacy_neartie()
            for source in inputs:
                for alias in (False, True):
                    with self.subTest(near=near, source=source, alias=alias):
                        destination = self.root / "input-link" if alias else source
                        if alias:
                            destination.symlink_to(source)
                        try:
                            self.assert_publication_refused([*self.args(near), "--runs", "1", "--provenance-out",
                                                             str(destination)], near=near, protected=(source,))
                        finally:
                            if alias:
                                destination.unlink()

    def test_legacy_output_symlink_cannot_overwrite_an_input(self):
        self.prepare_legacy_neartie()
        original = (self.out / "our_ids.i32").read_bytes()
        output = self.out / "our_ids.npy"
        output.symlink_to(self.out / "our_ids.i32")
        result = self.run_script(["--model", str(self.model), "--golden-dir", str(self.out),
                                  "--max-tokens", "2"], near=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ARTIFACT_MISMATCH", result.stderr)
        self.assertEqual((self.out / "our_ids.i32").read_bytes(), original)
        self.assertTrue(output.is_symlink())
        self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())

    def test_legacy_neartie_protects_symlinked_inputs(self):
        self.prepare_legacy_neartie()
        local_ids = self.out / "our_ids.i32"
        actual_ids = self.root / "actual_ids.i32"
        local_ids.rename(actual_ids)
        local_ids.symlink_to(actual_ids)
        self.assert_publication_refused(["--model", str(self.model), "--golden-dir", str(self.out),
                                         "--max-tokens", "2", "--provenance-out", str(actual_ids)],
                                        near=True, protected=(actual_ids,))

    def test_legacy_provenance_preserves_hardlinked_inputs(self):
        self.prepare_legacy_neartie()
        sources = [self.model / "config.json", self.model / "model.safetensors",
                   self.model / ".cache/huggingface/download/config.json.metadata",
                   self.wheel, self.runtime_manifest, self.source / "vllm/__init__.py",
                   *(self.project / "scripts" / name for name in
                     ("qwen3-oracle-capture.py", "qwen3-neartie-gap.py", "qwen3_oracle_common.py")),
                   self.project / "tests/parity/test_qwen35_paged_engine.cpp"]
        for near in (False, True):
            inputs = sources + (list(self.out.iterdir()) if near else [])
            for source in inputs:
                with self.subTest(near=near, source=source):
                    destination = self.root / "hardlink-manifest.json"
                    destination.hardlink_to(source)
                    try:
                        self.assert_publication_refused([*self.args(near), "--runs", "1", "--provenance-out",
                                                         str(destination)], near=near, protected=(source,))
                    finally:
                        destination.unlink()

    def test_legacy_output_hardlink_cannot_overwrite_an_input(self):
        self.prepare_legacy_neartie()
        original = (self.out / "our_ids.i32").read_bytes()
        output = self.out / "our_ids.npy"
        output.hardlink_to(self.out / "our_ids.i32")
        result = self.run_script(["--model", str(self.model), "--golden-dir", str(self.out),
                                  "--max-tokens", "2"], near=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ARTIFACT_MISMATCH", result.stderr)
        self.assertEqual((self.out / "our_ids.i32").read_bytes(), original)
        self.assertTrue(output.samefile(self.out / "our_ids.i32"))
        self.assertFalse((self.out / "neartie_gap_mnats.npy").exists())

    def test_legacy_publication_protects_selected_distribution_metadata(self):
        self.prepare_legacy_neartie()
        # Discover metadata on another sys.path entry, with no RECORD file.
        # CPython Distribution.metadata tries METADATA, PKG-INFO, then egg-info.
        site = self.root / "metadata-site"
        site.mkdir()
        self.env["PYTHONPATH"] = str(site) + os.pathsep + self.env["PYTHONPATH"]
        for layout in ("dist-info", "egg-info-directory", "egg-info-file", "empty-metadata"):
            metadata_root = site / ("vllm-unrelated-build.dist-info" if layout == "dist-info"
                                    else "vllm.egg-info")
            if layout == "egg-info-file":
                source = metadata_root
            else:
                metadata_root.mkdir()
                source = metadata_root / ("METADATA" if layout == "dist-info" else "PKG-INFO")
            source.write_text("Name: vllm\nVersion: controlled-test-fixture\n")
            sources = [source]
            if layout == "empty-metadata":
                empty = metadata_root / "METADATA"
                empty.write_bytes(b"")
                sources.append(empty)
            for near in (False, True):
                for consumed in sources:
                    for alias in ("direct", "symlink", "hardlink"):
                        with self.subTest(layout=layout, near=near, input=consumed.name, alias=alias):
                            destination = consumed if alias == "direct" else self.root / "metadata-alias.json"
                            if alias == "symlink":
                                destination.symlink_to(consumed)
                            elif alias == "hardlink":
                                destination.hardlink_to(consumed)
                            try:
                                self.assert_publication_refused(
                                    [*self.args(near), "--runs", "1", "--provenance-out", str(destination)],
                                    near=near, protected=tuple(sources))
                            finally:
                                if alias != "direct":
                                    destination.unlink()
            if metadata_root.is_dir():
                shutil.rmtree(metadata_root)
            else:
                metadata_root.unlink()

    def test_legacy_publication_refuses_aliases_between_output_destinations(self):
        self.prepare_legacy_neartie()
        for near in (False, True):
            names = (("our_ids.npy", "neartie_gap_mnats.npy", "neartie-provenance.json") if near else
                     ("greedy_ids.npy", "greedy_dist.npy", "oracle-provenance.json"))
            # Every pair of payload, default manifest, and additional manifest.
            outputs = [*(self.out / name for name in names), self.root / "additional.json"]
            for left in range(len(outputs)):
                for right in range(left + 1, len(outputs)):
                    source, destination = outputs[left], outputs[right]
                    for alias in ("direct", "symlink", "hardlink", "dangling-symlink"):
                        if alias == "direct" and right != 3:
                            continue  # CLI payload names are fixed and distinct.
                        if alias == "direct" and left == 2:
                            continue  # Explicit default manifest is a valid single target.
                        with self.subTest(near=near, source=source.name, target=destination.name, alias=alias):
                            originals = {p: p.read_bytes() for p in self.out.iterdir()}
                            try:
                                source.write_bytes(b"previous output")
                                destination.unlink(missing_ok=True)
                                external = source if alias == "direct" else outputs[-1]
                                if alias in ("symlink", "dangling-symlink"):
                                    destination.symlink_to(source)
                                    if alias == "dangling-symlink":
                                        source.unlink()
                                elif alias == "hardlink":
                                    destination.hardlink_to(source)
                                # A dangling link has no bytes to snapshot; record its target.
                                before = {p: (p.readlink() if p.is_symlink() else p.read_bytes())
                                          for p in [*self.out.iterdir(), *([destination] if right == 3 and alias != "direct" else [])]}
                                result = self.run_script([*self.args(near), "--runs", "1", "--provenance-out", str(external)], near=near)
                                self.assertNotEqual(result.returncode, 0, result.stdout)
                                self.assertIn("ARTIFACT_MISMATCH", result.stderr)
                                self.assertEqual(set(self.out.iterdir()), {p for p in before if p.parent == self.out})
                                for path, value in before.items():
                                    self.assertEqual(path.readlink() if path.is_symlink() else path.read_bytes(), value)
                                if external not in before:
                                    self.assertFalse(external.exists())
                            finally:
                                # Unlink aliases before restoring any bytes, including red runs.
                                for path in [*self.out.iterdir(), outputs[-1]]:
                                    path.unlink(missing_ok=True)
                                for path, data in originals.items():
                                    path.write_bytes(data)

    def test_legacy_explicit_default_manifest_and_replacement_keep_valid_hashes(self):
        self.prepare_legacy_neartie()
        for near in (False, True):
            manifest = self.out / ("neartie-provenance.json" if near else "oracle-provenance.json")
            for repeat in range(2):
                result = self.run_script([*self.args(near), "--runs", "1", "--provenance-out", str(manifest)], near=near)
                self.assertEqual(result.returncode, 0, result.stderr)
                provenance = json.loads(manifest.read_text())
                for name, record in provenance["outputs"].items():
                    self.assertEqual(sha(self.out / name), record["sha256"])
                    self.assertEqual((self.out / name).stat().st_size, record["size"])


    def test_legacy_capture_records_and_rechecks_consumed_distribution_metadata(self):
        self.prepare_legacy_neartie()
        metadata = self.source / "vllm-0.28.1.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text("Name: vllm\nVersion: controlled-test-fixture\n")
        for near in (False, True):
            manifest = self.out / ("neartie-provenance.json" if near else "oracle-provenance.json")
            result = self.run_script([*self.args(near), "--runs", "1"], near=near)
            self.assertEqual(result.returncode, 0, result.stderr)
            runtime = json.loads(manifest.read_text())["runtime"]
            self.assertEqual(runtime.get("distribution_metadata"), {
                str(metadata): {"sha256": sha(metadata), "size": metadata.stat().st_size}})
            before = {p: p.read_bytes() for p in self.out.iterdir()}
            original = metadata.read_bytes()
            try:
                result = self.run_script([*self.args(near), "--runs", "1"], near=near,
                                         env={"FAKE_MUTATE_INPUT": str(metadata)})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("runtime identity changed during capture", result.stderr)
                self.assertEqual({p: p.read_bytes() for p in self.out.iterdir()}, before)
            finally:
                metadata.write_bytes(original)



if __name__ == "__main__":
    unittest.main(verbosity=2)
