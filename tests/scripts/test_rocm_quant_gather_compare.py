"""Exercise the row comparator with explicitly synthetic validator inputs.

The two fixture reports preserve actual successful Q4_0 captures. The temporary
matrix uses their metadata with synthetic model bytes and repeated test tokens.
It is a validator test, not an oracle measurement or a model parity result.

The last class covers `compare.py` instead. It runs the whole tool over one
synthesized single-case matrix so the report's `scope` line is measured through
the command line and through the module API, including the default that issue
#3126 is about.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "rocm_quant_gather"
FIXTURES = Path(__file__).parent / "fixtures" / "rocm_quant_gather"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("gather_compare", TOOLS / "compare_models.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    tool_spec = importlib.util.spec_from_file_location("gather_compare_tool", TOOLS / "compare.py")
    tool = importlib.util.module_from_spec(tool_spec)
    tool_spec.loader.exec_module(tool)
finally:
    sys.path.pop(0)


class CompareModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="gather-validator-")
        base = Path(cls.directory.name)
        cls.args = SimpleNamespace(**{name: base / name for name in
            ("native", "primary", "secondary", "fixtures", "config")})
        cls.args.native_log = base / "native.log"
        cls.args.primary_memory = base / "memory.json"
        for name in ("native", "primary", "secondary", "fixtures", "config"):
            getattr(cls.args, name).mkdir()
        (cls.args.secondary / "controls").mkdir()
        (cls.args.config / "config.json").write_text('{"synthetic_validator": true}\n')
        cls.token_control = json.loads((FIXTURES / "primary-token.json").read_text())
        cls.memory_control = json.loads((FIXTURES / "primary-memory.json").read_text())
        cls.primary_path = cls.args.primary / "Q4_0-p0-r0.json"
        cls.args.native_log.write_text(
            "1: Test command: /synthetic/test_capi_rocm_embedding_quant\n" +
            "1: [kv-alloc] block_size=16 num_kv_heads=1 head_size=64 dtype=2 "
            "page_size_bytes=4096 num_blocks=16\n" * 228)

        def capture(prefix):
            Path(str(prefix) + "-tokens.txt").write_text("47 19 4 20\n")
            Path(str(prefix) + "-logits-f32.bin").write_bytes(struct.pack("<512f", *([0.0] * 512)))

        for name in comparator.FORMATS:
            model = cls.args.fixtures / (name + ".gguf")
            model.write_text("synthetic validator model " + name)
            for repeat in range(3):
                for p, prompt in enumerate(comparator.PROMPTS):
                    stem = f"{name}-r{repeat}-p{p}"
                    for arm in ("quant", "dense"):
                        capture(cls.args.native / (stem + "-" + arm))
                    if name in comparator.SECONDARY:
                        prefix = cls.args.secondary / (stem + "-oracle")
                        capture(prefix)
                        Path(str(prefix) + ".log").write_text("physical_context=256\n")
                    elif name != "Q8_K":
                        report = deepcopy(cls.token_control)
                        report.update(model=comparator.seal(model),
                                      config=comparator.seal(cls.args.config / "config.json"),
                                      prompt=prompt, repeat=repeat)
                        (cls.args.primary / f"{name}-p{p}-r{repeat}.json").write_text(json.dumps(report))
        for oracle in ("stock", "fork"):
            for repeat in range(3):
                for p in range(2):
                    for arm in ("pristine", "overlay"):
                        capture(cls.args.secondary / "controls" / f"{oracle}-{arm}-r{repeat}-p{p}")
        cls.memory = deepcopy(cls.memory_control)
        cls.memory.update(model=comparator.seal(cls.args.fixtures / "Q4_0.gguf"),
                          config=comparator.seal(cls.args.config / "config.json"))
        cls.args.primary_memory.write_text(json.dumps(cls.memory))
        cls.primary = json.loads(cls.primary_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def reject(self, path, report):
        original = path.read_bytes()
        try:
            path.write_text(json.dumps(report))
            with self.assertRaises((ValueError, KeyError)):
                comparator.compare(self.args)
        finally:
            path.write_bytes(original)

    def test_valid_synthetic_matrix(self):
        report = comparator.compare(self.args)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["cases"]), 114)

    def reject_many(self, path, base, mutations):
        for label, change in mutations:
            with self.subTest(label=label):
                report = deepcopy(base)
                change(report)
                self.reject(path, report)

    def test_primary_exception_after_generation_is_rejected(self):
        report = deepcopy(self.primary)
        report.update(status="PENDING: primary model refused before qualification",
                      exception_type="RuntimeError", exception="observer failed after tokens")
        self.reject(self.primary_path, report)

    def test_primary_status_only_refusal_is_rejected(self):
        # A PENDING status with no exception fields must still not qualify.
        report = deepcopy(self.primary)
        report.update(status="PENDING: primary model refused before qualification")
        self.reject(self.primary_path, report)

    def test_primary_traceback_only_is_rejected(self):
        report = deepcopy(self.primary)
        report.update(traceback="Traceback (most recent call last):\n  observer")
        self.reject(self.primary_path, report)

    def test_primary_executed_status_with_exception_is_rejected(self):
        # EXECUTED status with an exception field preserves tokens and must be
        # rejected on the exception alone (issue #3113's core scenario).
        report = deepcopy(self.primary)
        report.update(exception="observer raised after token write")
        self.reject(self.primary_path, report)

    def test_primary_pinned_identities_are_rejected_when_wrong(self):
        def wrong_primary_pin(report):
            report["primary_pin"] = "0" * 40

        def wrong_plugin_pin(report):
            report["plugin_pin"] = "0" * 40

        self.reject_many(self.primary_path, self.primary, [
            ("primary_pin", wrong_primary_pin), ("plugin_pin", wrong_plugin_pin)])

    def test_primary_runtime_identities_are_rejected_when_wrong(self):
        def wrong_vllm(report):
            report["vllm"] = "0.28.0"

        def wrong_plugin(report):
            report["plugin"] = "0.0.4"

        def wrong_torch(report):
            report["torch"] = "2.11.0"

        def wrong_torch_git(report):
            report["torch_git"] = "0" * 40

        def wrong_hip(report):
            report["hip"] = "6.0.0"

        self.reject_many(self.primary_path, self.primary, [
            ("vllm", wrong_vllm), ("plugin", wrong_plugin), ("torch", wrong_torch),
            ("torch_git", wrong_torch_git), ("hip", wrong_hip)])

    def test_primary_workload_identity_is_rejected_when_wrong(self):
        def wrong_model(report):
            report["model"] = {"sha256": "0" * 64, "bytes": 1}

        def wrong_config(report):
            report["config"] = {"sha256": "0" * 64, "bytes": 1}

        def wrong_prompt(report):
            report["prompt"] = [1, 0, 63, 127]

        def wrong_repeat(report):
            report["repeat"] = 1

        self.reject_many(self.primary_path, self.primary, [
            ("model", wrong_model), ("config", wrong_config),
            ("prompt", wrong_prompt), ("repeat", wrong_repeat)])

    def test_primary_dtype_and_request_are_rejected_when_wrong(self):
        def fp32_dtype(report):
            report["resolved_model_dtype"] = "torch.float32"

        def fp8_kv(report):
            report["requested"]["kv_cache_dtype"] = "fp8"

        def non_greedy(report):
            report["requested"]["greedy"] = False

        def different_capacity(report):
            report["cache_capacity_contract"]["bf16_kv_payload_bytes"] = 16384

        self.reject_many(self.primary_path, self.primary, [
            ("fp32_dtype", fp32_dtype), ("fp8_kv", fp8_kv),
            ("non_greedy", non_greedy), ("different_capacity", different_capacity)])

    def test_memory_report_path_rejects_unqualified_records(self):
        def exception_after_generation(report):
            report.update(exception_type="RuntimeError", exception="observer failed after tokens")

        def status_refusal(report):
            report["status"] = "PENDING: primary model refused before qualification"

        def wrong_primary_pin(report):
            report["primary_pin"] = "0" * 40

        def wrong_plugin_pin(report):
            report["plugin_pin"] = "0" * 40

        def non_greedy(report):
            report["requested"]["greedy"] = False

        def fp32_dtype(report):
            report["resolved_model_dtype"] = "torch.float32"

        self.reject_many(self.args.primary_memory, self.memory, [
            ("exception_after_generation", exception_after_generation),
            ("status_refusal", status_refusal), ("wrong_primary_pin", wrong_primary_pin),
            ("wrong_plugin_pin", wrong_plugin_pin), ("non_greedy", non_greedy),
            ("fp32_dtype", fp32_dtype)])

    def test_memory_storage_geometry_is_rejected_when_invalid(self):
        def offset_five(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                report[phase][0]["cache_tensors"][0]["storage_offset_elements"] = 5

        def offset_negative(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                report[phase][0]["cache_tensors"][0]["storage_offset_elements"] = -1

        def offset_float(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                report[phase][0]["cache_tensors"][0]["storage_offset_elements"] = 0.0

        def offset_bool(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                report[phase][0]["cache_tensors"][0]["storage_offset_elements"] = True

        def shrunk_storage(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                tensor = report[phase][0]["cache_tensors"][0]
                tensor["storage_bytes"] = 32768
                tensor["allocator"]["block_bytes"] = 32768

        def grown_shape(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                tensor = report[phase][0]["cache_tensors"][0]
                tensor["shape"] = [16, 2, 16, 128]
                tensor["numel"] = 65536
                tensor["payload_bytes"] = 131072
                tensor["storage_bytes"] = 131072

        def inactive_allocator(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                report[phase][0]["cache_tensors"][0]["allocator"]["state"] = "inactive"

        def wrong_layout(report):
            for phase in ("memory_before_generation", "memory_after_generation"):
                report[phase][0]["resolved_layout"] = "BNLH"

        def changed_during_generation(report):
            tensor = report["memory_after_generation"][0]["cache_tensors"][0]
            tensor["storage_address"] = tensor["storage_address"] - 4096

        self.reject_many(self.args.primary_memory, self.memory, [
            ("offset_five", offset_five), ("offset_negative", offset_negative),
            ("offset_float", offset_float), ("offset_bool", offset_bool),
            ("shrunk_storage", shrunk_storage), ("grown_shape", grown_shape),
            ("inactive_allocator", inactive_allocator), ("wrong_layout", wrong_layout),
            ("changed_during_generation", changed_during_generation)])


class CompareReportScopeTest(unittest.TestCase):
    """Measure the comparison report's scope line over one synthesized case.

    `compare.py` reads a manifest, a native report, and a primary report, so the
    smallest end-to-end input is one primary codec case carrying both output
    dtypes. Zeros on both sides keep the comparison itself trivial: this class
    asks what the report says its scope is, not what the run compared.
    """

    # Issue #3126 is about this label, so the literal is pinned here rather than
    # imported from the tool under test.
    DEFAULT_SCOPE = ("synthetic operation parity; original fixture and "
                     "model gates remain separate")
    NAME = "Q4_0-256"
    WIDTH = 256
    ROWS = 128
    TOKENS = 4

    @staticmethod
    def seal(path):
        data = Path(path).read_bytes()
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gather-scope-")
        self.base = Path(self.directory.name)
        row_bytes = self.WIDTH // 32 * 18  # Q4_0 stores 18 bytes per 32 values.
        self.packed_bytes = self.ROWS * row_bytes
        packed = self.base / (self.NAME + ".packed")
        packed.write_bytes(bytes(self.packed_bytes))
        ids = self.base / (self.NAME + ".ids")
        ids.write_bytes(struct.pack("<4i", 0, 127, 64, 0))
        self.manifest = self.base / "manifest.json"
        self.manifest.write_text(json.dumps(
            {"version": 1, "source": "synthetic-scope-case", "cases": [
                {"name": self.NAME, "type": 2, "rows": self.ROWS, "width": self.WIDTH,
                 "packed": packed.name, "ids": ids.name}]}) + "\n")
        self.native = self.base / "native"
        self.primary = self.base / "primary"
        self.secondary = self.base / "secondary"
        for directory in (self.native, self.primary, self.secondary):
            directory.mkdir()
        elements = self.TOKENS * self.WIDTH
        native_cases, primary_cases = [], []
        for dtype, size in (("f32", elements * 4), ("bf16", elements * 2)):
            native_result = self.native / f"{self.NAME}-{dtype}.bin"
            native_result.write_bytes(bytes(size))
            native_cases.append({"name": self.NAME, "type": 2, "dtype": dtype,
                                 "table_bytes": self.packed_bytes, "id_dtype": "i32",
                                 "id_bytes": self.TOKENS * 4, "output_bytes": size,
                                 "scratch_bytes": 16, "selected_packed_bytes": 0,
                                 "selections": 1, "provider": "synthetic-scope-case",
                                 "result": native_result.name})
            primary_result = self.primary / f"{self.NAME}-{dtype}.bin"
            primary_result.write_bytes(bytes(size))
            primary_cases.append({"name": self.NAME, "type": 2, "dtype": dtype,
                                  "packed": self.seal(packed), "ids": self.seal(ids),
                                  "table_bytes": self.packed_bytes,
                                  "selected_packed_bytes": row_bytes,
                                  "result": primary_result.name,
                                  "seal": self.seal(primary_result)})
        (self.native / "report.json").write_text(
            json.dumps({"version": 1, "cases": native_cases}) + "\n")
        (self.primary / "report.json").write_text(
            json.dumps({"version": 1, "cases": primary_cases}) + "\n")

    def tearDown(self):
        self.directory.cleanup()

    def run_cli(self, output, *extra):
        argv = ["compare.py", str(self.manifest), str(self.native), str(self.primary),
                str(self.secondary), str(output), *extra]
        original = sys.argv
        sys.argv = argv
        try:
            with redirect_stdout(io.StringIO()) as printed:
                tool.main()
        finally:
            sys.argv = original
        return json.loads(Path(output).read_text()), printed.getvalue()

    def test_cli_default_scope_is_the_previous_literal(self):
        report, printed = self.run_cli(self.base / "default-report.json")
        self.assertTrue(report["all_pass"])
        self.assertEqual(report["scope"], self.DEFAULT_SCOPE)
        self.assertEqual(printed.strip(), "PASS: 2 outputs, 2 byte-exact")

    def test_cli_scope_value_is_carried_into_the_report(self):
        label = "original 32-fixture coverage; model and synthetic gates remain separate"
        report, _ = self.run_cli(self.base / "explicit-report.json", "--scope", label)
        self.assertTrue(report["all_pass"])
        self.assertEqual(report["scope"], label)

    def test_api_default_scope_is_the_previous_literal(self):
        output = self.base / "api-report.json"
        with redirect_stdout(io.StringIO()):
            tool.compare(self.manifest, self.native, self.primary, self.secondary, output)
        self.assertEqual(json.loads(output.read_text())["scope"], self.DEFAULT_SCOPE)


if __name__ == "__main__":
    unittest.main()
