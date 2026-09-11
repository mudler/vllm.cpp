#!/usr/bin/env python3
"""No-GPU CLI and provenance tests. Fixtures are not an oracle runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
MODEL_REV = "a" * 40
SCRIPTS = ("qwen3-oracle-capture.py", "qwen3-neartie-gap.py")
FAKE_VLLM = '''
import json, os
from pathlib import Path
from types import SimpleNamespace as NS
__version__ = "controlled-test-fixture"
def log(value):
    with open(os.environ["FAKE_LOG"], "a") as stream:
        stream.write(json.dumps(value) + "\\n")
class SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Pinned vLLM sampling_params.py:494 normalizes this sentinel.
        if self.seed == -1:
            self.seed = None
class LLM:
    def __init__(self, **kwargs):
        log({"llm": kwargs})
        self.model_path = Path(kwargs["model"])
        cfg = json.loads((Path(kwargs["model"]) / "config.json").read_text())
        if os.environ.get("FAKE_IDENTITY_DRIFT"):
            cfg["model_type"] = "qwen3"
        if os.environ.get("FAKE_RUNTIME_QWEN35"):
            cfg["model_type"] = "qwen3_5"
        model = NS(dtype="torch.bfloat16", revision=kwargs.get("revision"),
                   tokenizer_revision=kwargs.get("tokenizer_revision"),
                   seed=kwargs.get("seed", 0), enforce_eager=kwargs.get("enforce_eager", False),
                   hf_config=NS(**cfg))
        cache = NS(cache_dtype=os.environ.get("FAKE_CACHE_DTYPE", kwargs.get("kv_cache_dtype", "auto")))
        self.llm_engine = NS(vllm_config=NS(model_config=model, cache_config=cache))
        self.calls = 0
    def generate(self, prompts, sp):
        self.calls += 1
        if self.calls == 1:
            if os.environ.get("FAKE_MUTATE_MODEL"):
                (self.model_path / "model.safetensors").write_bytes(b"changed during capture")
            if os.environ.get("FAKE_MUTATE_PACKAGE"):
                with Path(__file__).open("a") as stream:
                    stream.write("\\n# changed during capture\\n")
            for variable in ("FAKE_MUTATE_SCRIPT", "FAKE_MUTATE_INPUT"):
                if os.environ.get(variable):
                    with Path(os.environ[variable]).open("a") as stream:
                        stream.write("\\n# changed during capture\\n")
        log({"prompts": prompts, "sampling": vars(sp)})
        if isinstance(prompts, dict):
            full = prompts["prompt_token_ids"]
            probs = [None, {600: NS(logprob=-1.0)}]
            for token in full[2:]:
                d = {10000: NS(logprob=-1.0)}
                if token != 9999:
                    lp = float("nan") if os.environ.get("FAKE_NONFINITE") else -1.25
                    if os.environ.get("FAKE_GAP_DRIFT") and self.calls > 16:
                        lp -= 0.0001
                    d[token] = NS(logprob=lp)
                probs.append(d)
            return [NS(prompt_logprobs=probs)]
        results = []
        for prompt in prompts:
            index = int(os.environ["FAKE_PROMPTS"].split("||").index(prompt))
            ids = [700 + index] * sp.max_tokens
            prompt_ids = [500 + index, 600]
            if os.environ.get("FAKE_TOKEN_DRIFT") and self.calls > 144:
                ids[-1] += 1
            if os.environ.get("FAKE_PROMPT_DRIFT") and self.calls > 144:
                prompt_ids[0] += 1
            if os.environ.get("FAKE_SHORT"):
                ids = ids[:-1]
            results.append(NS(prompt=prompt, prompt_token_ids=prompt_ids,
                              outputs=[NS(token_ids=ids, text="fixture")]))
        return results
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_script(path):
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": ModuleType("numpy")}), mock.patch.object(
        sys, "path", [str(path.parent), *sys.path]
    ):
        spec.loader.exec_module(module)
    return module


class CaptureFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qwen3-capture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "tests/parity").mkdir(parents=True)
        for filename in (*SCRIPTS, "qwen3_oracle_common.py"):
            source = ROOT / "scripts" / filename
            if source.exists():
                shutil.copy2(source, self.project / "scripts" / filename)
        shutil.copy2(ROOT / "tests/parity/test_qwen35_paged_engine.cpp",
                     self.project / "tests/parity/test_qwen35_paged_engine.cpp")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"},
        }))
        (self.model / "model.safetensors").write_bytes(b"not real model weights")
        self.write_metadata()
        self.source = self.root / "source"
        (self.source / "vllm").mkdir(parents=True)
        (self.source / "vllm/__init__.py").write_text(FAKE_VLLM)
        self.git("init", "-q")
        self.git("add", "vllm")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
                 "commit", "-qm", "controlled fixture")
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.write(self.source / "vllm/__init__.py", "vllm/__init__.py")
        self.runtime_manifest = self.root / "runtime.json"
        self.runtime_manifest.write_text(json.dumps({
            "vllm_revision": self.revision, "wheel_sha256": sha(self.wheel),
            "image_digest": "sha256:" + "c" * 64,
        }))
        self.out = self.root / "golden"
        self.log = self.root / "calls.jsonl"
        prompts = load_script(self.project / "scripts" / SCRIPTS[0]).PROMPTS
        self.env = dict(os.environ, PYTHONPATH=os.pathsep.join(
                            [str(self.source), os.environ.get("PYTHONPATH", "")]), FAKE_LOG=str(self.log),
                        FAKE_PROMPTS="||".join(prompts), PYTHONDONTWRITEBYTECODE="1")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def write_metadata(self):
        for path in self.model.iterdir():
            if not path.is_file():
                continue
            data = path.read_bytes()
            etag = (hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                    if path.suffix == ".json" else hashlib.sha256(data).hexdigest())
            metadata = self.model / ".cache/huggingface/download" / (path.name + ".metadata")
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(f"{MODEL_REV}\n{etag}\n0\n")

    def args(self, near=False):
        return ["--model", str(self.model), "--model-revision", MODEL_REV,
                "--vllm-revision", self.revision, "--vllm-wheel", str(self.wheel),
                "--runtime-manifest", str(self.runtime_manifest), "--max-tokens", "2",
                "--golden-dir" if near else "--out-dir", str(self.out)]

    def run_script(self, args=None, *, near=False, env=None):
        return subprocess.run([sys.executable, str(self.project / "scripts" / SCRIPTS[near]),
                               *(self.args(near) if args is None else args)],
                              env=dict(self.env, **(env or {})), text=True, capture_output=True)

    def assert_refused(self, result, reason):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(reason, result.stderr, result.stdout + result.stderr)
        self.assertFalse(self.out.exists() and any(self.out.iterdir()))


class CaptureContractTests(CaptureFixture):
    def test_options_and_eager_alias(self):
        for script in SCRIPTS:
            module = load_script(self.project / "scripts" / script)
            required = ["--model", "model", "--golden-dir", "unused"] if "neartie" in script else []
            for mode in ("production", "eager"):
                for dtype in ("auto", "bfloat16", "fp8_e4m3"):
                    args = module._parse_args([*required, "--execution-mode", mode,
                                              "--kv-cache-dtype", dtype, "--seed", "17"])
                    kwargs = module._llm_kwargs(args)
                    self.assertEqual(kwargs["kv_cache_dtype"], dtype)
                    self.assertEqual(kwargs["seed"], 17)
                    self.assertIs(kwargs["enforce_eager"], mode == "eager")
            args = module._parse_args([*required, "--enforce-eager"])
            self.assertEqual(args.execution_mode, "eager")
            self.assertIn("eager diagnostic", module._mode_narration(args))
            with self.assertRaises(SystemExit):
                module._parse_args([*required, "--enforce-eager", "--execution-mode", "production"])

    def test_all_three_prompt_sources_are_checked(self):
        for filename in ("scripts/" + SCRIPTS[0], "scripts/" + SCRIPTS[1],
                         "tests/parity/test_qwen35_paged_engine.cpp"):
            path = self.project / filename
            original = path.read_text()
            path.write_text(original.replace("The capital of France is", "Changed prompt", 1))
            for near in (False, True):
                self.assert_refused(self.run_script(near=near), "PROMPTS_MISMATCH")
                self.assertFalse(self.log.exists(), "prompt mismatch must refuse before LLM construction")
            path.write_text(original)

    def test_strict_identity_inputs_are_required(self):
        for option in ("--model-revision", "--vllm-revision", "--vllm-wheel", "--runtime-manifest"):
            argv = self.args()
            index = argv.index(option)
            del argv[index:index + 2]
            self.assert_refused(self.run_script(argv), "ARTIFACT_MISMATCH")

    def test_model_revision_and_content_are_verified(self):
        path = self.model / "config.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")
        path.write_bytes(original)
        meta = self.model / ".cache/huggingface/download/config.json.metadata"
        meta.write_text(meta.read_text().replace(MODEL_REV, "b" * 40))
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def test_weight_bytes_are_checked_against_download_identity(self):
        (self.model / "model.safetensors").write_bytes(b"different weights")
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def test_dirty_source_is_refused_even_when_the_wheel_matches_it(self):
        source = self.source / "vllm/__init__.py"
        source.write_text(source.read_text() + "\n# dirty source rebuilt into wheel\n")
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.write(source, "vllm/__init__.py")
        runtime = json.loads(self.runtime_manifest.read_text())
        runtime["wheel_sha256"] = sha(self.wheel)
        self.runtime_manifest.write_text(json.dumps(runtime))
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def test_vllm_revision_and_dirty_source_are_refused(self):
        argv = self.args()
        argv[argv.index("--vllm-revision") + 1] = "b" * 40
        self.assert_refused(self.run_script(argv), "ARTIFACT_MISMATCH")
        with (self.source / "vllm/__init__.py").open("a") as stream:
            stream.write("\n# dirty\n")
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def test_runtime_manifest_and_wheel_are_verified(self):
        original = self.runtime_manifest.read_text()
        for field, value in (("wheel_sha256", "0" * 64), ("vllm_revision", "b" * 40),
                             ("image_digest", "mutable:latest")):
            data = json.loads(original)
            data[field] = value
            self.runtime_manifest.write_text(json.dumps(data))
            self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")
        self.runtime_manifest.write_text(original)
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("vllm/__init__.py", FAKE_VLLM + "\n# changed wheel\n")
        data = json.loads(original)
        data["wheel_sha256"] = sha(self.wheel)
        self.runtime_manifest.write_text(json.dumps(data))
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def test_strict_runtime_needs_an_observed_source_or_installed_vcs_revision(self):
        (self.source / ".git").rename(self.root / "saved-source-git")
        self.assert_refused(self.run_script(), "observed source or installed VCS revision")

    def test_imported_and_installed_runtime_versions_must_agree(self):
        metadata = self.source / "vllm-0.28.1.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text("Name: vllm\nVersion: different-installed-version\n")
        self.assert_refused(self.run_script(), "imported and installed vLLM versions differ")

    def test_untracked_package_bytes_and_missing_artifact_metadata_are_refused(self):
        extra = self.source / "vllm/untracked.py"
        extra.write_text("untracked = True\n")
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")
        extra.unlink()
        (self.model / ".cache/huggingface/download/model.safetensors.metadata").unlink()
        self.assert_refused(self.run_script(), "ARTIFACT_MISMATCH")

    def test_runtime_identity_cannot_promote_unverified_legacy_inputs(self):
        (self.model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        self.write_metadata()
        self.assert_refused(self.run_script(["--model", str(self.model), "--out-dir",
                                            str(self.out), "--runs", "1"],
                                           env={"FAKE_RUNTIME_QWEN35": "1"}),
                            "ARTIFACT_MISMATCH")

    def test_strict_short_repetition_count_is_refused(self):
        self.assert_refused(self.run_script([*self.args(), "--runs", "9"]), "ten")
        self.assertFalse(self.log.exists(), "short strict capture must refuse before LLM construction")

    def test_each_qwen35_identity_indicator_prevents_legacy_downgrade(self):
        for config in ({"model_type": "qwen3_5"},
                       {"model_type": "qwen3", "architectures": ["Qwen3_5ForCausalLM"]},
                       {"model_type": "qwen3", "text_config": {"model_type": "qwen3_5_text"}}):
            (self.model / "config.json").write_text(json.dumps(config))
            self.write_metadata()
            self.assert_refused(self.run_script(["--model", str(self.model), "--out-dir",
                                                str(self.out), "--runs", "1"]),
                                "ARTIFACT_MISMATCH")

    def test_options_reject_invalid_counts_and_cache_modes(self):
        for script in SCRIPTS:
            module = load_script(self.project / "scripts" / script)
            required = ["--model", "model", "--golden-dir", "unused"] if "neartie" in script else []
            for option, value in (("--runs", "0"), ("--repetitions", "-1"),
                                  ("--max-tokens", "0"), ("--kv-cache-dtype", "float32")):
                with self.assertRaises(SystemExit):
                    module._parse_args([*required, option, value])


class CaptureRegistrationTests(unittest.TestCase):
    def test_preflight_runs_both_suites_or_reports_numpy_pending(self):
        source = (ROOT / "scripts/agent-preflight.sh").read_text()
        match = re.search(r"# QWEN3-CAPTURE-TOOLS: begin\n(.*?)# QWEN3-CAPTURE-TOOLS: end", source, re.S)
        self.assertIsNotNone(match, "capture suites have no preflight registration")
        prelude = '''run() { printf 'RUN %s\\n' "$1"; shift; "$@"; }
skip() { printf 'SKIP %s %s\\n' "$1" "$*"; }
python3() {
  if test "$1" = -c; then test "$CAPTURE_HAVE_NUMPY" = 1;
  else printf 'PYTHON %s\\n' "$*"; fi
}
'''
        for available in ("0", "1"):
            result = subprocess.run(["bash", "-c", prelude + match.group(1)], text=True,
                                    capture_output=True, env=dict(os.environ, CAPTURE_HAVE_NUMPY=available))
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("test_qwen3_capture_tools", "test_qwen3_capture_outputs"):
                self.assertIn(("RUN " if available == "1" else "SKIP ") + name, result.stdout)
                if available == "1":
                    self.assertIn("PYTHON tests/scripts/" + name + ".py", result.stdout)
            if available == "0":
                self.assertIn("PENDING", result.stdout)
                self.assertNotIn("RUN ", result.stdout)

    def test_ci_runs_both_suites_after_its_existing_numpy_install(self):
        source = (ROOT / ".github/workflows/ci.yml").read_text()
        for name in ("test_qwen3_capture_tools", "test_qwen3_capture_outputs"):
            command = "          python3 tests/scripts/" + name + ".py"
            self.assertIn(command, source)
            prefix = source[:source.index(command)]
            block = prefix[prefix.rfind("        run: |") :]
            self.assertIn("apt-get install -y --no-install-recommends python3-numpy", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
