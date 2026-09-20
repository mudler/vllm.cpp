"""Dedicated real-format, production-CLI tests for exact Qwen3 conversion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import safetensors

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tools/bench/strix_four_engine/audit.py"
MODEL = "1cfa9a7208912126459214e8b04321603b3df60c"
PIN = "10bf611e533d81f739128304991c5e133c6aebd8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}


def inventory_sha(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ConversionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = Path(os.environ["STRIX_AUDIT_CONVERTER"]).resolve()
        cls.source_inventory = inventory(cls.converter)
        if inventory_sha(cls.source_inventory) != "ad7a105b10602373f7936b15fe9ff76c4ba349b982c985115e2520b9172e0ad7":
            raise AssertionError("STRIX_AUDIT_CONVERTER is not the pinned source tree")
        sys.path.insert(0, str(cls.converter / "gguf-py"))
        import gguf
        cls.gguf = gguf

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native = self.root / "native"
        self.native.mkdir()
        self.gguf_path = self.root / "model.gguf"
        self.output = self.root / "result.json"
        self.config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                       "num_hidden_layers": 36, "tie_word_embeddings": True}
        self.mapping = {"model.embed_tokens.weight": "token_embd.weight",
                        "model.norm.weight": "output_norm.weight"}
        # Independent literal pairs, not imported from the implementation.
        pairs = [("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
                 ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                 ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output"),
                 ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
                 ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
                 ("mlp.down_proj", "ffn_down")]
        for i in range(36):
            for source, dest in pairs:
                self.mapping[f"model.layers.{i}.{source}.weight"] = f"blk.{i}.{dest}.weight"
        self.source = {}
        self.dest = {}
        for i, (name, target) in enumerate(self.mapping.items()):
            shape = (4,) if "norm" in name else (4, 32)
            values = (np.arange(np.prod(shape), dtype=np.float32) + (i + 1) * 32).reshape(shape)
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            values = (bits.astype(np.uint32) << 16).view(np.float32)
            self.source[name] = {"dtype": "BF16", "shape": list(shape), "data": bits.tobytes()}
            self.dest[target] = values

    def fixture(self, mode=None):
        dtype_override = {}
        matrix = "blk.0.attn_q.weight"
        norm = "output_norm.weight"
        if mode == "bf16_value": self.dest[matrix][0, 0] += 32
        if mode == "f32_value": self.dest[norm][0] += 0.125
        if mode == "shape": self.dest[matrix] = self.dest[matrix].reshape(8, 16)
        if mode == "permutation": self.dest[matrix] = self.dest[matrix][::-1].copy()
        if mode == "missing": del self.dest[matrix]
        if mode == "extra": self.dest["output.weight"] = self.dest["token_embd.weight"].copy()
        if mode == "source_extra": self.source["lm_head.weight"] = self.source["model.embed_tokens.weight"]
        if mode == "source_missing": del self.source["model.layers.0.self_attn.q_proj.weight"]
        if mode == "source_dtype":
            self.source["model.norm.weight"] = {"dtype": "F32", "shape": [4], "data": self.dest[norm].tobytes()}
        if mode == "source_u16": self.source["model.norm.weight"]["dtype"] = "U16"
        if mode == "quantized": dtype_override[matrix] = self.gguf.GGMLQuantizationType.Q8_0
        if mode == "destination_f16_zero":
            self.dest[matrix].fill(0)
            self.source["model.layers.0.self_attn.q_proj.weight"]["data"] = bytes(256)
            dtype_override[matrix] = self.gguf.GGMLQuantizationType.F16
        if mode == "late_value":
            values = np.ones((1025, 1024), dtype=np.float32)
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            self.source["model.layers.0.self_attn.q_proj.weight"] = {
                "dtype": "BF16", "shape": [1025, 1024], "data": bits.tobytes()}
            values[-1, -1] = 2
            self.dest[matrix] = values
        if mode == "nonfinite":
            self.dest[norm][0] = np.inf
            self.source["model.norm.weight"]["data"] = (self.dest[norm].view(np.uint32) >> 16).astype(np.uint16).tobytes()
        if mode == "architecture": self.config["architectures"] = ["Qwen2ForCausalLM"]
        if mode == "untied": self.config["tie_word_embeddings"] = False
        if mode == "both_wrong_rank":
            self.dest[matrix] = self.dest[matrix].reshape(-1)
            self.source["model.layers.0.self_attn.q_proj.weight"]["shape"] = [128]
        (self.native / "config.json").write_text(json.dumps(self.config))
        safetensors.serialize_file({name: dict(tensor, dtype={"BF16": "bfloat16", "F32": "float32", "U16": "uint16"}[tensor["dtype"]])
                                   for name, tensor in self.source.items()}, str(self.native / "model.safetensors"))
        (self.native / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            name: "model.safetensors" for name in self.source}}))
        writer = self.gguf.GGUFWriter(str(self.gguf_path), "qwen2" if mode == "gguf_architecture" else "qwen3")
        for name, values in self.dest.items():
            kind = dtype_override.get(name, self.gguf.GGMLQuantizationType.F32 if values.ndim == 1
                                     else self.gguf.GGMLQuantizationType.BF16)
            data = self.gguf.quants.quantize(values, kind)
            writer.add_tensor(name, data, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        if mode == "duplicate":
            # The writer rejects duplicates. Corrupt a same-length name after valid writing.
            blob = self.gguf_path.read_bytes()
            self.gguf_path.write_bytes(blob.replace(b"blk.0.attn_k.weight", b"blk.0.attn_q.weight"))
        manifest = {"model": {"revision": MODEL, "directory": str(self.native), "files": inventory(self.native)},
                    "converter": {"revision": PIN, "directory": str(self.converter),
                                  "inventory_sha256": inventory_sha(self.source_inventory)}}
        if mode == "model_hash": manifest["model"]["files"]["config.json"] = "0" * 64
        if mode == "converter_hash": manifest["converter"]["inventory_sha256"] = "0" * 64
        if mode == "model_revision": manifest["model"]["revision"] = "0" * 40
        if mode == "converter_revision": manifest["converter"]["revision"] = "0" * 40
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        if mode == "existing": self.output.write_text("preserve me")
        return manifest

    def run_cli(self):
        return subprocess.run([sys.executable, str(CLI), "--manifest", str(self.manifest),
                               "--gguf", str(self.gguf_path), "--output", str(self.output)],
                              text=True, capture_output=True, timeout=60)

    def test_complete_inventory_and_exact_hashes(self):
        manifest = self.fixture()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(self.output.read_text())
        self.assertEqual(record["result"], "PASS")
        self.assertEqual(record["tensor_count"], 398)
        self.assertEqual(record["gguf_sha256"], sha(self.gguf_path))
        self.assertEqual(record["model"], manifest["model"])
        self.assertEqual(record["converter"], manifest["converter"])
        self.assertEqual(list(self.root.glob(".audit-pending-*")), [])
        self.assertEqual({t["source"]: t["destination"] for t in record["tensors"]}, self.mapping)
        for tensor in record["tensors"]:
            source = self.source[tensor["source"]]
            self.assertEqual(tensor["shape"], source["shape"])
            self.assertEqual(tensor["elements"], np.prod(source["shape"]))
            self.assertEqual(tensor["source_dtype"], "BF16")
            self.assertEqual(tensor["destination_dtype"], "F32" if len(source["shape"]) == 1 else "BF16")
            self.assertEqual(tensor["promotion"], "BF16 to F32 exact" if len(source["shape"]) == 1 else None)
            self.assertTrue(tensor["exact"])

    def test_output_write_and_close_failures_do_not_publish(self):
        # Intercept real file I/O, not the audit, so all 398 comparisons execute.
        script = '''import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("audit", sys.argv[1])
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
mode = sys.argv[2]; original = pathlib.Path.open
class Fault:
    def __init__(self, handle): self.handle = handle
    def __enter__(self): return self
    def __getattr__(self, name): return getattr(self.handle, name)
    def write(self, text):
        result = self.handle.write(text)
        if mode == "write" and text == "\\n":
            raise OSError("injected output write failure")
        return result
    def __exit__(self, *args):
        self.handle.close()
        if mode == "close": raise OSError("injected output close failure")
def opened(path, mode="r", *args, **kwargs):
    handle = original(path, mode, *args, **kwargs)
    return Fault(handle) if mode == "x" else handle
pathlib.Path.open = opened
sys.argv = sys.argv[1:2] + sys.argv[3:]
sys.exit(audit.main())
'''
        for mode in ("write", "close"):
            with self.subTest(mode=mode):
                self.setUp()
                self.fixture()
                result = subprocess.run([sys.executable, "-c", script, str(CLI), mode,
                                         "--manifest", str(self.manifest), "--gguf", str(self.gguf_path),
                                         "--output", str(self.output)], text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("injected output " + mode + " failure", result.stderr)
                self.assertFalse(self.output.exists(), "failed audit published a result")
                self.assertEqual(list(self.root.glob(".audit-pending-*")), [])

    def test_refusals_leave_no_passing_record(self):
        for mode in ("bf16_value", "f32_value", "shape", "permutation", "missing", "duplicate", "extra",
                     "source_extra", "source_dtype", "quantized", "nonfinite", "architecture", "untied",
                     "model_hash", "converter_hash", "model_revision", "converter_revision", "existing",
                     "both_wrong_rank", "source_u16", "destination_f16_zero", "source_missing", "gguf_architecture",
                     "late_value"):
            with self.subTest(mode=mode):
                # Fresh real fixtures isolate each refusal.
                self.setUp()
                self.fixture(mode)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0, mode)
                self.assertIn("AUDIT_FAIL", result.stderr, result.stderr)
                if mode == "existing": self.assertEqual(self.output.read_text(), "preserve me")
                else: self.assertFalse(self.output.exists(), mode)

    def test_independent_binding_and_architecture_refusals(self):
        cases = {
            "unbound": "unbound source shard",
            "unexpected": "unexpected source shard",
            "duplicate_json": "duplicate JSON key",
            "escape": "source path escapes model directory",
            "symlink": "converter symlink",
            "model_type": "architecture mismatch",
            "layers": "architecture mismatch",
            "misplaced": "duplicate or misplaced source tensor",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode):
                self.setUp()
                manifest = self.fixture()
                if mode == "unexpected":
                    shutil.copyfile(self.native / "model.safetensors", self.native / "extra.safetensors")
                if mode in ("model_type", "layers"):
                    self.config["model_type" if mode == "model_type" else "num_hidden_layers"] = (
                        "qwen2" if mode == "model_type" else 35)
                    (self.native / "config.json").write_text(json.dumps(self.config))
                if mode == "duplicate_json":
                    path = self.native / "config.json"
                    path.write_text('{"model_type":"qwen3",' + path.read_text()[1:])
                if mode == "misplaced":
                    groups = [dict(list(self.source.items())[::2]), dict(list(self.source.items())[1::2])]
                    for shard, group in zip(("model.safetensors", "other.safetensors"), groups):
                        safetensors.serialize_file({name: dict(value, dtype="bfloat16") for name, value in group.items()},
                                                   str(self.native / shard))
                    # Correct tensor set and bytes, but the index assigns each tensor to the wrong shard.
                    index = {name: "other.safetensors" if name in groups[0] else "model.safetensors"
                             for name in self.source}
                    (self.native / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
                manifest["model"]["files"] = inventory(self.native)
                if mode == "unbound":
                    del manifest["model"]["files"]["model.safetensors"]
                if mode == "escape":
                    outside = self.root / "outside.json"
                    outside.write_text("{}")
                    manifest["model"]["files"]["../outside.json"] = sha(outside)
                if mode == "symlink":
                    converter = self.root / "converter"
                    shutil.copytree(self.converter, converter, ignore=shutil.ignore_patterns(".git", "__pycache__"))
                    license_path = converter / "LICENSE"
                    outside = self.root / "license"
                    license_path.rename(outside)
                    license_path.symlink_to(outside)
                    # Identical inventory bytes isolate symlink refusal from the hash guard.
                    self.assertEqual(inventory_sha(inventory(converter)), inventory_sha(self.source_inventory))
                    manifest["converter"]["directory"] = str(converter)
                self.manifest.write_text(json.dumps(manifest))
                result = self.run_cli()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.output.exists())

    def test_library_refuses_foreign_submodule_with_pinned_root(self):
        self.fixture()
        script = '''import importlib.util, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
sys.path.insert(0, str(pathlib.Path(manifest["converter"]["directory"]) / "gguf-py"))
import gguf, gguf.gguf_reader
# Keep the real reader behavior and pinned root. Only the submodule origin is foreign.
gguf.gguf_reader.__file__ = sys.argv[4]
spec = importlib.util.spec_from_file_location("audit", sys.argv[1])
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
try:
    audit.audit_model(manifest, pathlib.Path(sys.argv[3]))
except ValueError as error:
    print(error); sys.exit(2)
sys.exit(0)
'''
        outside = self.root / "gguf_reader.py"
        shutil.copyfile(self.converter / "gguf-py/gguf/gguf_reader.py", outside)
        result = subprocess.run([sys.executable, "-c", script, str(CLI), str(self.manifest),
                                 str(self.gguf_path), str(outside)], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("GGUF import outside pinned converter", result.stdout)

    def test_publication_refuses_racing_outputs_and_unsupported_rename(self):
        script = '''import ctypes, errno, importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("audit", sys.argv[1])
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
mode = sys.argv[2]; output = pathlib.Path(sys.argv[-1]); original = audit.audit_model
def raced(*args):
    result = original(*args)
    if mode == "race": output.write_text("concurrent result")
    if mode == "platform": audit.sys.platform = "win32"
    return result
audit.audit_model = raced
if mode == "missing": audit.ctypes.CDLL = lambda *args, **kwargs: object()
if mode == "unsupported":
    class Rename:
        def __call__(self, *args):
            ctypes.set_errno(errno.EOPNOTSUPP); return -1
    class Lib: renameat2 = Rename()
    audit.ctypes.CDLL = lambda *args, **kwargs: Lib()
sys.argv = sys.argv[1:2] + sys.argv[3:]
sys.exit(audit.main())
'''
        for mode in ("race", "unsupported", "platform", "missing"):
            with self.subTest(mode=mode):
                self.setUp()
                self.fixture()
                result = subprocess.run([sys.executable, "-c", script, str(CLI), mode,
                                         "--manifest", str(self.manifest), "--gguf", str(self.gguf_path),
                                         "--output", str(self.output)], text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("atomic no-replace publication " + (
                    "requires Linux renameat2" if mode in ("platform", "missing") else "failed"), result.stderr)
                if mode == "race": self.assertEqual(self.output.read_text(), "concurrent result")
                else: self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".audit-pending-*")), [])

    def test_publication_refuses_sync_failure_and_reports_failed_cleanup(self):
        # Removing fsync must expose an incorrectly published PASS. Removing the
        # cleanup diagnostic must hide a retained staging file from the caller.
        script = '''import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("audit", sys.argv[1])
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
mode = sys.argv[2]
def sync_failure(fd): raise OSError("injected sync failure")
audit.os.fsync = sync_failure
if mode == "cleanup":
    original = pathlib.Path.unlink
    def cleanup_failure(path, *args, **kwargs):
        if path.name.startswith(".audit-pending-"): raise OSError("injected unlink failure")
        return original(path, *args, **kwargs)
    pathlib.Path.unlink = cleanup_failure
sys.argv = sys.argv[1:2] + sys.argv[3:]
sys.exit(audit.main())
'''
        for mode in ("sync", "cleanup"):
            with self.subTest(mode=mode):
                self.setUp()
                self.fixture()
                result = subprocess.run([sys.executable, "-c", script, str(CLI), mode,
                                         "--manifest", str(self.manifest), "--gguf", str(self.gguf_path),
                                         "--output", str(self.output)], text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("AUDIT_FAIL: injected sync failure", result.stderr)
                self.assertFalse(self.output.exists())
                pending = list(self.root.glob(".audit-pending-*"))
                if mode == "cleanup":
                    self.assertIn("AUDIT_PENDING_CLEANUP_FAIL: injected unlink failure", result.stderr)
                    self.assertEqual(len(pending), 1)
                    record = json.loads(pending[0].read_text())
                    self.assertEqual(record["tensor_count"], 398)
                    self.assertEqual(record["result"], "PASS")
                else:
                    self.assertEqual(pending, [])

    def test_ci_invocation_reaches_the_dedicated_suite(self):
        import yaml
        workflow = yaml.safe_load((ROOT / ".github/workflows/strix-conversion-audit.yml").read_text())
        steps = workflow["jobs"]["exact-conversion"]["steps"]
        step = next(s for s in steps if s.get("name") == "Run the real-format audit tests")
        # Execute the actual workflow command with an unavailable converter.
        # A removed/no-op invocation cannot reproduce the required setup failure.
        (self.root / ".audit-venv").symlink_to(Path(sys.prefix), target_is_directory=True)
        (self.root / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
        env = dict(os.environ, STRIX_AUDIT_CONVERTER=str(self.root / "missing-converter"))
        result = subprocess.run(["sh", "-c", step["run"]], cwd=self.root, env=env,
                                text=True, capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STRIX_AUDIT_CONVERTER is not the pinned source tree", result.stderr)

    def test_library_rechecks_source_converter_and_gguf_after_comparison(self):
        self.fixture()
        # Run each library invocation in a fresh interpreter, as the CLI does.
        script = '''import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("audit", sys.argv[1])
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
gguf = pathlib.Path(sys.argv[3]); target = pathlib.Path(sys.argv[4])
original = audit.digest; calls = 0
def changed(path):
    global calls
    result = original(path)
    if pathlib.Path(path) == gguf:
        calls += 1
        if calls == 1: target.write_bytes(target.read_bytes() + b" ")
    return result
audit.digest = changed
try:
    audit.audit_model(manifest, gguf)
except ValueError as error:
    print(error); sys.exit(2)
sys.exit(0)
'''
        for target in (self.native / "config.json", self.gguf_path):
            with self.subTest(target=target.name):
                before = target.read_bytes()
                try:
                    result = subprocess.run([sys.executable, "-c", script, str(CLI), str(self.manifest),
                                             str(self.gguf_path), str(target)], text=True, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertRegex(result.stdout, "source hash mismatch|GGUF changed during audit")
                finally:
                    target.write_bytes(before)
        converter = self.root / "converter"
        shutil.copytree(self.converter, converter, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        manifest = json.loads(self.manifest.read_text())
        manifest["converter"]["directory"] = str(converter)
        self.manifest.write_text(json.dumps(manifest))
        result = subprocess.run([sys.executable, "-c", script, str(CLI), str(self.manifest),
                                 str(self.gguf_path), str(converter / "LICENSE")],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("converter inventory mismatch", result.stdout)
        # Rebinding a modified tree to itself cannot authenticate the revision.
        manifest["converter"]["inventory_sha256"] = inventory_sha(inventory(converter))
        self.manifest.write_text(json.dumps(manifest))
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("converter inventory mismatch", result.stderr)
        self.assertFalse(self.output.exists())

    def test_library_refuses_an_already_imported_foreign_reader(self):
        self.fixture()
        outside = self.root / "foreign"
        shutil.copytree(self.converter / "gguf-py" / "gguf", outside / "gguf",
                        ignore=shutil.ignore_patterns("__pycache__"))
        script = '''import importlib.util, json, pathlib, sys
sys.path.insert(0, sys.argv[4]); import gguf
spec = importlib.util.spec_from_file_location("audit", sys.argv[1])
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
try:
    audit.audit_model(json.loads(pathlib.Path(sys.argv[2]).read_text()), pathlib.Path(sys.argv[3]))
except ValueError as error:
    print(error); sys.exit(2)
sys.exit(0)
'''
        result = subprocess.run([sys.executable, "-c", script, str(CLI), str(self.manifest),
                                 str(self.gguf_path), str(outside)], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("GGUF import outside pinned converter", result.stdout)


if __name__ == "__main__":
    unittest.main()
