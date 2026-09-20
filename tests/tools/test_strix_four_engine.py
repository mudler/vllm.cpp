import hashlib
import contextlib
import math
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "tools/bench/strix_four_engine/prepare.py"
UPSTREAM_SETUP = "# Copyright 2025 SGLang Team. All Rights Reserved.\n#\n# Licensed under the Apache License, Version 2.0 (the \"License\");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an \"AS IS\" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n# ==============================================================================\n\nimport os\nimport platform\nimport sys\nfrom pathlib import Path\n\nimport torch\nfrom setuptools import find_packages, setup\nfrom torch.utils.cpp_extension import BuildExtension, CUDAExtension\n\nroot = Path(__file__).parent.resolve()\narch = platform.machine().lower()\n\n\ndef _get_version():\n    with open(root / \"pyproject.toml\") as f:\n        for line in f:\n            if line.startswith(\"version\"):\n                return line.split(\"=\")[1].strip().strip('\"')\n\n\noperator_namespace = \"sgl_kernel\"\ninclude_dirs = [\n    root / \"include\",\n    root / \"include\" / \"impl\",\n    root / \"csrc\",\n]\n\nsources = [\n    \"csrc/allreduce/custom_all_reduce.hip\",\n    \"csrc/allreduce/deterministic_all_reduce.hip\",\n    \"csrc/allreduce/quick_all_reduce.cu\",\n    \"csrc/common_extension_rocm.cc\",\n    \"csrc/elementwise/activation.cu\",\n    \"csrc/elementwise/deepseek_v4_topk.cu\",\n    \"csrc/elementwise/dsv4_norm_rope.cu\",\n    \"csrc/elementwise/topk.cu\",\n    \"csrc/grammar/apply_token_bitmask_inplace_cuda.cu\",\n    \"csrc/moe/moe_align_kernel.cu\",\n    \"csrc/moe/moe_topk_softmax_kernels.cu\",\n    \"csrc/moe/moe_topk_sigmoid_kernels.cu\",\n    \"csrc/speculative/eagle_utils.cu\",\n    \"csrc/kvcacheio/transfer.cu\",\n    \"csrc/memory/weak_ref_tensor.cpp\",\n    \"csrc/elementwise/pos_enc.cu\",\n]\n\ncxx_flags = [\"-O3\"]\nlibraries = [\"hiprtc\", \"amdhip64\", \"c10\", \"torch\", \"torch_python\"]\nextra_link_args = [\"-Wl,-rpath,$ORIGIN/../../torch/lib\", f\"-L/usr/lib/{arch}-linux-gnu\"]\n\ndefault_target = \"gfx942\"\namdgpu_target = os.environ.get(\"AMDGPU_TARGET\", default_target)\n\nif torch.cuda.is_available():\n    try:\n        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(\":\")[0]\n    except Exception as e:\n        print(f\"Warning: Failed to detect GPU properties: {e}\")\nelse:\n    print(f\"Warning: torch.cuda not available. Using default target: {amdgpu_target}\")\n\nif amdgpu_target not in [\"gfx942\", \"gfx950\"]:\n    print(\n        f\"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. Expected 'gfx942' or 'gfx950'.\"\n    )\n    sys.exit(1)\n\nfp8_macro = (\n    \"-DHIP_FP8_TYPE_FNUZ\" if amdgpu_target == \"gfx942\" else \"-DHIP_FP8_TYPE_E4M3\"\n)\n\n# Dynamic shared-memory budget for the TopK kernels.\n# - gfx942 (MI300/MI325): LDS is typically 64KB per workgroup -> keep dynamic smem <= ~48KB\n#   (leaves room for static shared allocations in the kernel).\n# - gfx95x (MI350): LDS is larger (e.g. 160KB per CU) -> allow the original 128KB dynamic smem.\ntopk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target == \"gfx942\" else 32 * 1024 * 4\n\nhipcc_flags = [\n    \"-DNDEBUG\",\n    f\"-DOPERATOR_NAMESPACE={operator_namespace}\",\n    \"-O3\",\n    \"-Xcompiler\",\n    \"-fPIC\",\n    \"-std=c++17\",\n    f\"--amdgpu-target={amdgpu_target}\",\n    \"-DENABLE_BF16\",\n    \"-DENABLE_FP8\",\n    fp8_macro,\n    f\"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}\",\n]\n\next_modules = [\n    CUDAExtension(\n        name=\"sgl_kernel.common_ops\",\n        sources=sources,\n        include_dirs=include_dirs,\n        extra_compile_args={\n            \"nvcc\": hipcc_flags,\n            \"cxx\": cxx_flags,\n        },\n        libraries=libraries,\n        extra_link_args=extra_link_args,\n        py_limited_api=False,\n    ),\n]\n\nsetup(\n    name=\"sglang-kernel\",\n    version=_get_version(),\n    packages=find_packages(where=\"python\"),\n    package_dir={\"\": \"python\"},\n    ext_modules=ext_modules,\n    cmdclass={\"build_ext\": BuildExtension.with_options(use_ninja=True)},\n    options={\"bdist_wheel\": {\"py_limited_api\": \"cp39\"}},\n)\n"
PIN = "f63458b5beaceabbd9d749b9fc956370e1b649e6"


class PreparationTests(unittest.TestCase):
    def probe_identity(self, local, properties, backend="hip", error=None):
        spec = importlib.util.spec_from_file_location("prepare3072", PREPARE)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        torch = types.SimpleNamespace(__file__=str(local / "venv/torch.py"), __version__="torch-pin",
            version=types.SimpleNamespace(hip="7.2", cuda=None),
            cuda=types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx1151")))
        query = mock.Mock(return_value=properties, side_effect=error)
        active = types.SimpleNamespace(utils=types.SimpleNamespace(get_device_properties=query),
            get_current_target=lambda: types.SimpleNamespace(backend=backend, arch="gfx1151"))
        triton = types.SimpleNamespace(__file__=str(local / "venv/triton.py"), __version__="triton-pin",
            runtime=types.SimpleNamespace(driver=types.SimpleNamespace(active=active)))

        def command(label, argv):
            self.assertEqual(argv[:3], ["python", "-I", "-c"])
            self.assertEqual(argv[-2], tool.IDENTITY)
            output = io.StringIO()
            with mock.patch.dict(sys.modules, torch=torch, triton=triton), \
                    mock.patch("importlib.metadata.distributions", return_value=[
                        types.SimpleNamespace(metadata={"Name": "triton"})]), \
                    mock.patch.object(sys, "argv", ["-c", argv[-1]]), contextlib.redirect_stdout(output):
                exec(argv[-2], {})
            return output.getvalue()

        session = types.SimpleNamespace(local=local, output=local, command=command,
            manifest={"expected": {"torch": "torch-pin", "triton": "triton-pin"}})
        result = tool.identity(session, "python", "dependencies")
        query.assert_called_once_with(0)
        self.assertEqual(json.loads((local / "identity-dependencies.json").read_text()), result)
        return result

    def test_identity_reads_hip_limit_without_torch_member(self):
        for limit in (65536, 98304):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                result = self.probe_identity(Path(directory), {"max_shared_mem": limit, "arch": "gfx1151"})
                self.assertEqual(result["shared_memory_per_block"], limit)
                self.assertEqual(result["shared_memory_source"],
                                 "triton.runtime.driver.active.utils.get_device_properties(0).max_shared_mem")

    def test_identity_rejects_invalid_hip_limits(self):
        for limit in (None, "65536", 65536.0, True, False, -1, 0, 65535):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "shared.memory"):
                    self.probe_identity(Path(directory), {"max_shared_mem": limit, "arch": "gfx1151"})
                self.assertFalse((Path(directory) / "identity-dependencies.json").exists())

    def test_identity_rejects_missing_hip_limit(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(KeyError):
            self.probe_identity(Path(directory), {"arch": "gfx1151"})

    def test_identity_rejects_non_hip_driver(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "HIP backend"):
            self.probe_identity(Path(directory), {"max_shared_mem": 65536, "arch": "gfx1151"}, backend="cuda")

    def test_identity_rejects_hip_arch_disagreement(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "architecture"):
            self.probe_identity(Path(directory), {"max_shared_mem": 65536, "arch": "gfx942"})

    def test_identity_propagates_hip_query_error(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "HIP query failed"):
            self.probe_identity(Path(directory), {}, error=RuntimeError("HIP query failed"))

    def test_emitted_identity_detects_all_triton_namespace_owners(self):
        spec = importlib.util.spec_from_file_location("prepare3053", PREPARE)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        for names in (("triton-rocm",), ("pytorch_triton_rocm",),
                      ("triton-rocm", "pytorch-triton-rocm"), ("triton", "pytorch-triton-rocm")):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                local = Path(directory)
                torch = types.SimpleNamespace(__file__=str(local / "venv/torch.py"), __version__="torch-pin",
                    version=types.SimpleNamespace(hip="7.2", cuda=None),
                    cuda=types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(
                        gcnArchName="gfx1151", shared_memory_per_block=65536)))
                active = types.SimpleNamespace(utils=types.SimpleNamespace(get_device_properties=lambda _: {
                    "max_shared_mem": 65536, "arch": "gfx1151"}),
                    get_current_target=lambda: types.SimpleNamespace(backend="hip", arch="gfx1151"))
                triton = types.SimpleNamespace(__file__=str(local / "venv/triton.py"), __version__="triton-pin",
                    runtime=types.SimpleNamespace(driver=types.SimpleNamespace(active=active)))
                def command(label, argv):
                    output = io.StringIO()
                    distributions = [types.SimpleNamespace(metadata={"Name": n}) for n in names]
                    with mock.patch.dict(sys.modules, torch=torch, triton=triton), \
                            mock.patch("importlib.metadata.distributions", return_value=distributions), \
                            mock.patch.object(sys, "argv", ["-c", argv[-1]]), contextlib.redirect_stdout(output):
                        exec(argv[-2], {})
                    return output.getvalue()
                session = types.SimpleNamespace(local=local, output=local, command=command,
                    manifest={"expected": {"torch": "torch-pin", "triton": "triton-pin"}})
                if len(names) > 1:
                    with self.assertRaisesRegex(ValueError, "competing Triton"):
                        tool.identity(session, "python", "dependencies")
                else:
                    self.assertEqual(tool.identity(session, "python", "dependencies")["triton_distributions"], list(names))

    def test_cli_binds_runtime_and_ignores_inherited_search_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest, env = self.fixture(folder)
            env.update(PATH="/unverified", LD_LIBRARY_PATH="/unverified",
                       ROCM_PATH="/unverified", ROCM_HOME="/unverified", HIP_PATH="/unverified")
            result = self.cli(manifest, folder / "output", env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((folder / "output/rocm.json").read_text())
            root = str(folder / "rocm")
            self.assertEqual(state["root"], root)
            self.assertEqual(state["PATH"], root + "/bin:" + root + "/llvm/bin:/usr/bin:/bin")
            self.assertEqual(state["LD_LIBRARY_PATH"], root + "/lib:" + root + "/lib64")
            self.assertEqual(state["compiler"]["realpath"], root + "/bin/hipcc")
            self.assertEqual(state["runtime"][0]["realpath"], root + "/lib/libamdhip64.so.7")
            success = json.loads((folder / "output/success.json").read_text())
            self.assertEqual(success["rocm"], state)

    def test_cli_accepts_contained_toolkit_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest, env = self.fixture(folder)
            data = json.loads(manifest.read_text())
            alias = folder / "rocm-alias"
            alias.symlink_to(folder / "rocm")
            clang = folder / "rocm/llvm/bin/clang++"
            clang.write_bytes(b"recorded compiler identity")
            clang.chmod(0o755)
            data["rocm"]["root"] = str(alias)
            library = folder / "rocm/lib/libamdhip64.so"
            library.symlink_to("libamdhip64.so.7")
            data["rocm"]["runtime"][0]["path"] = str(library)
            manifest.write_text(json.dumps(data))
            result = self.cli(manifest, folder / "output", env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((folder / "output/rocm.json").read_text())
            self.assertEqual(state["root"], str(folder / "rocm"))
            self.assertEqual(state["runtime"][0]["realpath"], str(folder / "rocm/lib/libamdhip64.so.7"))
            self.assertEqual(state["clang"], {"path": str(clang), "realpath": str(clang),
                                            "sha256": hashlib.sha256(clang.read_bytes()).hexdigest()})

    def test_cli_refuses_invalid_rocm_bindings(self):
        for fault in ("missing", "relative-root", "relative-file", "hash", "outside",
                      "duplicate", "compiler-mode", "compiler-selection", "runtime-empty",
                      "duplicate-alias", "compiler-name", "compiler-hash", "missing-root",
                      "outside-symlink", "clang-escape", "post-clang",
                      "missing-file", "missing-libs", "directory-escape", "post-hash", "post-mode"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest, env = self.fixture(folder)
                data = json.loads(manifest.read_text())
                rocm = data["rocm"]
                if fault == "missing": del data["rocm"]
                if fault == "relative-root": rocm["root"] = os.path.relpath(rocm["root"])
                if fault == "relative-file": rocm["runtime"][0]["path"] = os.path.relpath(rocm["runtime"][0]["path"])
                if fault == "hash": rocm["runtime"][0]["sha256"] = "0" * 64
                if fault == "outside": rocm["runtime"][0] = data["dependencies"][0]
                if fault == "outside-symlink":
                    library = Path(rocm["runtime"][0]["path"])
                    outside = folder / "outside-runtime.so"
                    outside.write_bytes(library.read_bytes())
                    library.unlink()
                    library.symlink_to(outside)
                if fault == "clang-escape":
                    (folder / "rocm/llvm/bin/clang++").symlink_to(folder / "bin/hipcc")
                if fault == "post-clang":
                    (folder / "rocm/llvm/bin/clang++").write_bytes(b"compiler")
                    env["STRIX_PREPARE_FAULT"] = "clang-mutation"
                if fault == "duplicate": rocm["runtime"].append(dict(rocm["runtime"][0]))
                if fault == "duplicate-alias":
                    alias = folder / "rocm/lib/alias.so"
                    alias.symlink_to("libamdhip64.so.7")
                    rocm["runtime"].append(dict(rocm["runtime"][0], path=str(alias)))
                if fault == "compiler-mode": Path(rocm["compiler"]["path"]).chmod(0o644)
                if fault == "compiler-name":
                    alias = folder / "rocm/bin/compiler-alias"
                    alias.symlink_to("hipcc")
                    rocm["compiler"]["path"] = str(alias)
                if fault == "compiler-hash": rocm["compiler"]["sha256"] = "0" * 64
                if fault == "compiler-selection":
                    target = folder / "rocm/other/hipcc"
                    target.parent.mkdir()
                    target.write_bytes(Path(rocm["compiler"]["path"]).read_bytes())
                    target.chmod(0o755)
                    rocm["compiler"]["path"] = str(target)
                if fault == "missing-root": rocm["root"] = str(folder / "absent")
                if fault == "runtime-empty": rocm["runtime"] = []
                if fault == "missing-file": Path(rocm["runtime"][0]["path"]).unlink()
                if fault == "missing-libs":
                    Path(rocm["runtime"][0]["path"]).rename(folder / "rocm/runtime.so")
                    rocm["runtime"][0]["path"] = str(folder / "rocm/runtime.so")
                    (folder / "rocm/lib").rmdir()
                    (folder / "rocm/lib64").rmdir()
                if fault == "directory-escape":
                    (folder / "rocm/lib64").rmdir()
                    (folder / "rocm/lib64").symlink_to(folder)
                if fault == "post-hash": env["STRIX_PREPARE_FAULT"] = "runtime-mutation"
                if fault == "post-mode": env["STRIX_PREPARE_FAULT"] = "compiler-mode-mutation"
                manifest.write_text(json.dumps(data))
                result = self.cli(manifest, folder / "output", env)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("ROCm", result.stderr)
                self.assertFalse((folder / "output/success.json").exists())

    def test_bf16_gate_enters_compiled_operation_at_both_shapes(self):
        spec = importlib.util.spec_from_file_location("prepare3053", PREPARE)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        for tokens in (1, 4):
            torch = mock.MagicMock()
            torch.bfloat16 = "bf16"
            class Input:
                def __getitem__(self, index):
                    if index == (Ellipsis, slice(9728, None)):
                        return 3.0
                    if index == (Ellipsis, slice(None, 9728)):
                        return 2.0
                    raise AssertionError(index)
            torch.randn.return_value = Input()
            torch.nn.functional.silu.side_effect = lambda value: value / (1 + math.exp(-value))
            torch.testing = types.SimpleNamespace(assert_close=mock.Mock())
            actual = types.SimpleNamespace(dtype="bf16", shape=(tokens, 9728))
            kernel = types.SimpleNamespace(silu_and_mul=mock.Mock(return_value=actual))
            pytest = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=lambda *args: lambda fn: fn))
            with mock.patch.dict(sys.modules, torch=torch, sgl_kernel=kernel, pytest=pytest):
                namespace = {}
                exec(tool.BF16_TEST, namespace)
                namespace["test_qwen3_bf16_silu"](tokens)
            torch.randn.assert_called_once_with(tokens, 19456, device="cuda", dtype="bf16")
            kernel.silu_and_mul.assert_called_once_with(torch.randn.return_value)
            torch.testing.assert_close.assert_called_once()
            self.assertIs(torch.testing.assert_close.call_args.args[0], actual)
            self.assertAlmostEqual(torch.testing.assert_close.call_args.args[1], 5.284782467867294)
            self.assertEqual(torch.testing.assert_close.call_args.kwargs, {"rtol": 1.6e-2, "atol": 1e-5})

    def test_cli_prepares_isolated_patched_rocm(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest, env = self.fixture(folder)
            result = self.cli(manifest, folder / "output", env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((folder / "output/success.json").read_text())
            self.assertEqual(state["comparator"], "patched SGLang")
            self.assertEqual(state["source_revision"], PIN)
            self.assertEqual(state["patch"]["changed"], ["sgl-kernel/setup_rocm.py"])
            patch_path = PREPARE.with_name("patches") / "sglang-f63458b5-gfx1151.patch"
            self.assertEqual(state["patch"]["sha256"], hashlib.sha256(patch_path.read_bytes()).hexdigest())
            with tarfile.open(folder / "source.tar") as archive:
                before = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                          for m in archive.getmembers() if m.isfile()}
            self.assertEqual(state["patch"]["before"], before)
            after = dict(before)
            after["sgl-kernel/setup_rocm.py"] = "5fe84d7afb969073bff8b1bfcb0e224c36964aca0e7b025ea25caec55407fe9a"
            self.assertEqual(state["patch"]["after"], after)
            ninja = folder / "rocm/bin/ninja"
            self.assertEqual(state.get("ninja"), {"path": str(ninja), "realpath": str(ninja),
                "sha256": hashlib.sha256(ninja.read_bytes()).hexdigest(), "version": "1.13.2"})
            self.assertEqual(state["identity"]["arch"], "gfx1151")
            inspection = state["extension_inspection"]
            copied = Path(inspection["inspection"])
            installed = Path(state["identity"]["extension"])
            self.assertEqual(inspection["source"], str(installed))
            self.assertEqual(copied.read_bytes(), installed.read_bytes())
            self.assertFalse(copied.is_symlink())
            self.assertFalse(copied.is_relative_to(Path(state["local"]) / "venv"))
            self.assertFalse(copied.is_relative_to(Path(state["local"]) / "source"))
            self.assertEqual(inspection["installed_sha256"], hashlib.sha256(installed.read_bytes()).hexdigest())
            self.assertEqual(inspection["inspected_sha256"], inspection["installed_sha256"])
            self.assertEqual(inspection["package_before"], inspection["package_after"])
            self.assertTrue(Path(str(copied) + '.0.hipv4-gfx1151').is_file())
            self.assertTrue((copied.parent / 'offload-cwd.bundle').is_file())
            command = json.loads((folder / 'output/logs/extension-targets.command.json').read_text())
            self.assertEqual(command['cwd'], str(copied.parent))
            self.assertEqual(command['argv'][-1], str(copied))
            self.assertTrue(state["hip_build_rules"])
            commands = [json.loads(p.read_text())["argv"] for p in (folder / "output/logs").glob("*.command.json")]
            self.assertTrue(any("setup_rocm.py" in c and "bdist_wheel" in c for c in commands))
            self.assertTrue(any("sglang[srt_hip]" in c and "--no-index" in c for c in commands))
            self.assertTrue(any("--offloading" in c for c in commands))
            self.assertTrue(any("-m" in c and "pytest" in c for c in commands))
            self.assertTrue(any("test_qwen3_bf16_silu.py" in str(c) for c in commands))
            for filename in ("test_activation.py", "test_topk.py"):
                self.assertTrue(any(filename in str(c) and "pytest" in c for c in commands))
            self.assertTrue(any(c[-2:] == ["pip", "check"] for c in commands))
            self.assertNotEqual(state["local"], str(folder))
            self.assertFalse(any("launch_server" in str(c) for c in commands))
            self.assertEqual((folder / "dependency.whl").read_bytes(), b"bound wheel")
            original = (folder / "output/success.json").read_bytes()
            result = self.cli(manifest, folder / "output", env)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((folder / "output/success.json").read_bytes(), original)

    def test_cli_refuses_inspection_failures(self):
        for fault, diagnostic in (("extension", "extension architecture"),
                                  ("inspector-error", "command failed: extension-targets"),
                                  ("copy-mismatch", "copy hash mismatch"),
                                  ("installed-mutation", "package changed"),
                                  ("package-mutation", "package changed"),
                                  ("package-symlink", "package changed"),
                                  ("package-file-bytes", "package changed"),
                                  ("package-symlink-target", "package changed"),
                                  ("package-directory", "package changed")):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest, env = self.fixture(folder)
                env['STRIX_PREPARE_FAULT'] = fault
                result = self.cli(manifest, folder / 'output', env)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse((folder / 'output/success.json').exists())
                self.assertTrue((folder / 'output/failure.json').exists())
                self.assertFalse((folder / 'output/logs/upstream-tests.command.json').exists())

    def test_cli_refuses_unverified_inputs_and_failures(self):
        for fault, diagnostic in (("lease", "lease"), ("hash", "sha256"), ("source-hash", "sha256"),
                                  ("marker", "revision marker"),
                                  ("pin", "source pin"), ("cuda", "ROCm identity"),
                                  ("arch", "ROCm identity"), ("extension", "extension architecture"),
                                  ("torch-version", "ROCm identity"), ("triton-version", "ROCm identity"),
                                  ("triton-overlap", "ROCm identity"), ("outside", "outside isolated"),
                                  ("pip", "command failed"), ("timeout", "command timeout"),
                                  ("skip", "upstream tests"), ("mutation", "source mutation"),
                                  ("count", "upstream tests"),
                                  ("limits", "invalid resource limits"), ("disk", "headroom"),
                                  ("log", "log output limit"), ("ninja-mutation", "Ninja changed")):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest, env = self.fixture(folder)
                data = json.loads(manifest.read_text())
                env["STRIX_PREPARE_FAULT"] = fault
                if fault == "lease": env.pop("RC_JOB_ID")
                if fault == "hash": data["dependencies"][0]["sha256"] = "0" * 64
                if fault == "source-hash": data["source"]["sha256"] = "0" * 64
                if fault == "marker":
                    source = Path(data["source"]["path"])
                    source.write_bytes(source.read_bytes().replace(PIN.encode(), b"0" * 40, 1))
                    data["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                if fault == "pin": data["source"]["revision"] = "0" * 40
                if fault == "limits": data["limits"]["build_timeout"] = -1
                if fault == "timeout": data["limits"]["build_timeout"] = 0.2
                if fault == "disk": data["limits"]["min_disk_bytes"] = 10**18
                if fault == "log": data["limits"]["max_log_bytes"] = 1024
                manifest.write_text(json.dumps(data))
                result = self.cli(manifest, folder / "output", env)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse((folder / "output/success.json").exists())
                self.assertTrue((folder / "output/failure.json").exists())

    def test_cli_refuses_same_byte_ninja_reselection(self):
        for fault in ("ninja-selected-path", "ninja-resolved-path"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest, env = self.fixture(folder)
                ninja = folder / "rocm/bin/ninja"
                original = ninja.read_bytes()
                target = folder / "rocm/ninja-original"
                target.write_bytes(original)
                target.chmod(0o755)
                ninja.unlink()
                selected = folder / "rocm/llvm/bin/ninja" if fault == "ninja-selected-path" else ninja
                selected.symlink_to(target)
                env["STRIX_PREPARE_FAULT"] = fault
                result = self.cli(manifest, folder / "output", env)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Ninja changed", result.stderr)
                self.assertFalse((folder / "output/success.json").exists())
                self.assertTrue((folder / "output/failure.json").exists())
                self.assertEqual(ninja.read_bytes(), original)
                self.assertTrue(os.access(ninja, os.X_OK))
                if fault == "ninja-selected-path":
                    self.assertNotEqual(ninja, selected)
                    self.assertEqual(ninja.resolve(), selected.resolve())
                else:
                    self.assertEqual(ninja, selected)
                    self.assertNotEqual(ninja.resolve(), target)
                    self.assertEqual(target.read_bytes(), original)

    def test_cli_refuses_inherited_compiler_override(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest, env = self.fixture(folder)
            env["CXXFLAGS"] = "-DUNVERIFIED_OVERRIDE"
            result = self.cli(manifest, folder / "output", env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inherited compiler override", result.stderr)

    def cli(self, manifest, output, env):
        launch = [sys.executable, str(PREPARE)]
        if env.get('STRIX_PREPARE_FAULT') == 'copy-mismatch':
            launch = [sys.executable, '-c', COPY_MISMATCH, str(PREPARE)]
        return subprocess.run([*launch, "--manifest", str(manifest), "--output", str(output)],
                              env=dict(env, STRIX_PREPARE_OUTPUT=str(output)), text=True, capture_output=True, timeout=30)

    def fixture(self, folder):
        binary = folder / "bin"
        binary.mkdir()
        for name in ("python-fixture", "hipcc", "llvm-objdump", "ccache"):
            path = binary / name
            path.write_text(FAKE_COMMAND)
            path.chmod(0o755)
        rocm = folder / "rocm"
        for name in ("bin", "llvm/bin", "lib", "lib64"):
            (rocm / name).mkdir(parents=True)
        for name in ("hipcc", "llvm-objdump", "ccache", "ninja"):
            path = rocm / "bin" / name
            path.write_text(FAKE_COMMAND)
            path.chmod(0o755)
        runtime = rocm / "lib/libamdhip64.so.7"
        runtime.write_bytes(b"bound ROCm runtime")
        source = folder / "source.tar"
        files = {"sgl-kernel/setup_rocm.py": UPSTREAM_SETUP,
                 "sgl-kernel/pyproject_rocm.toml": 'version = "0.4.4"\n',
                 "sgl-kernel/pyproject.toml": "old kernel config",
                 "python/pyproject.toml": "old config", "python/pyproject_other.toml": "hip config",
                 "sgl-kernel/tests/test_activation.py": "fixture activation", "sgl-kernel/tests/test_topk.py": "fixture topk"}
        with tarfile.open(source, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": PIN}) as archive:
            for name, value in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(value.encode())
                archive.addfile(info, io.BytesIO(value.encode()))
        wheel = folder / "dependency.whl"
        wheel.write_bytes(b"bound wheel")
        def record(path):
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = folder / "manifest.json"
        manifest.write_text(json.dumps({"source": dict(record(source), revision=PIN),
            "python": str(binary / "python-fixture"), "dependencies": [record(wheel)],
            "rocm": {"root": str(rocm), "compiler": record(rocm / "bin/hipcc"),
                     "runtime": [record(runtime)]},
            "expected": {"torch": "2.13.0+rocm7.2", "triton": "3.7.1"},
            "limits": {"build_timeout": 10, "test_timeout": 10, "min_mem_bytes": 1, "min_disk_bytes": 1}}))
        env = dict(os.environ, RC_DEVICE="strix:gpu0", RC_JOB_ID="11111111-1111-1111-1111-111111111111",
                   PATH=str(binary) + os.pathsep + os.environ["PATH"])
        return manifest, env

    def test_upstream_setup_policy_and_exact_patch(self):
        spec = importlib.util.spec_from_file_location("prepare3053", PREPARE)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            setup = root / "sgl-kernel/setup_rocm.py"
            setup.parent.mkdir()
            setup.write_text(UPSTREAM_SETUP)
            other = root / "unrelated"
            other.write_text("unchanged")
            result = self.policy(setup, "gfx1151")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unsupported GPU architecture", result.stdout)
            patch = tool.apply_compatibility_patch(root)
            self.assertEqual(patch["changed"], ["sgl-kernel/setup_rocm.py"])
            self.assertEqual(other.read_text(), "unchanged")
            after = setup.read_bytes()
            for arch, fp8, lds in (("gfx1151", "E4M3", 49152),
                                    ("gfx942", "FNUZ", 49152),
                                    ("gfx950", "E4M3", 131072)):
                with self.subTest(arch=arch):
                    result = self.policy(setup, arch)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    flags = json.loads(result.stdout)["flags"]
                    self.assertIn("--amdgpu-target=" + arch, flags)
                    self.assertIn("-DHIP_FP8_TYPE_" + fp8, flags)
                    self.assertIn("-DSGL_TOPK_DYNAMIC_SMEM_BYTES=" + str(lds), flags)
            result = self.policy(setup, "gfx1200")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unsupported GPU architecture", result.stdout)
            setup.write_text(UPSTREAM_SETUP)
            tool.apply_compatibility_patch(root)
            self.assertEqual(setup.read_bytes(), after)
            setup.write_text(UPSTREAM_SETUP.replace('default_target = "gfx942"', 'default_target = "gfx950"'))
            with self.assertRaisesRegex(ValueError, "upstream setup hash"):
                tool.apply_compatibility_patch(root)

    def policy(self, setup, arch):
        return subprocess.run([sys.executable, "-c", POLICY, str(setup), arch],
                              text=True, capture_output=True, timeout=10)

    def test_patch_refuses_wrong_postimage_and_unexpected_files(self):
        spec = importlib.util.spec_from_file_location("prepare3053", PREPARE)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        run = subprocess.run
        for fault in ("postimage", "new", "changed", "deleted"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                setup = root / "sgl-kernel/setup_rocm.py"
                setup.parent.mkdir()
                setup.write_text(UPSTREAM_SETUP)
                other = root / "unrelated"
                other.write_text("unchanged")
                def tamper(argv, **kwargs):
                    result = run(argv, **kwargs)
                    if "--check" not in argv:
                        if fault == "postimage": setup.write_text(setup.read_text() + "\n# unexpected\n")
                        if fault == "new": (root / "new-file").write_text("unexpected")
                        if fault == "changed": other.write_text("changed")
                        if fault == "deleted": other.unlink()
                    return result
                with mock.patch.object(tool.subprocess, "run", side_effect=tamper):
                    with self.assertRaisesRegex(ValueError, "unexpected compatibility patch source mutation"):
                        tool.apply_compatibility_patch(root)


COPY_MISMATCH = '''import runpy, shutil, sys
original = shutil.copyfile
def corrupt(source, target, **kwargs):
    result = original(source, target, **kwargs)
    if str(target).endswith('common_ops.so'):
        from pathlib import Path
        Path(target).write_bytes(b'corrupted copy')
    return result
shutil.copyfile = corrupt
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''

FAKE_COMMAND = r'''#!/usr/bin/python3
import json, os, pathlib, shutil, sys, time
args=sys.argv[1:]; exe=pathlib.Path(sys.argv[0]); fault=os.environ.get('STRIX_PREPARE_FAULT')
selected=os.environ.get('ROCM_PATH')
if selected:
    assert os.environ['ROCM_HOME']==selected and os.environ['HIP_PATH']==selected
    assert os.environ['PATH']==selected+'/bin:'+selected+'/llvm/bin:/usr/bin:/bin'
    assert os.environ['LD_LIBRARY_PATH']==selected+'/lib:'+selected+'/lib64'
if exe.name=='hipcc':
    assert (pathlib.Path(os.environ['STRIX_PREPARE_OUTPUT'])/'rocm.json').is_file()
    print('HIP 7.2'); sys.exit()
if exe.name=='ccache': print('ccache 4'); sys.exit()
if exe.name=='ninja': print('1.13.2'); sys.exit()
if exe.name=='llvm-objdump':
    operand=pathlib.Path(args[-1])
    pathlib.Path(str(operand)+'.0.hipv4-gfx1151').write_text('extracted bundle')
    (pathlib.Path.cwd()/'offload-cwd.bundle').write_text('cwd bundle')
    captured=json.loads((pathlib.Path(os.environ['STRIX_PREPARE_OUTPUT'])/'identity-complete.json').read_text())
    installed=pathlib.Path(captured['extension'])
    if fault=='installed-mutation': installed.write_text('changed extension')
    if fault=='package-mutation': (installed.parent/'unexpected.py').write_text('changed package')
    if fault=='package-symlink': (installed.parent/'unexpected.so').symlink_to(installed)
    if fault=='package-file-bytes': (installed.parent/'sgl_kernel.py').write_text('modified')
    if fault=='package-symlink-target':
        link=installed.parent/'package-link'
        link.unlink(); link.symlink_to('torch.py')
    if fault=='package-directory': (installed.parent/'empty-directory').mkdir()
    if fault=='inspector-error': sys.exit(7)
    print('gfx942' if fault=='extension' else 'gfx1151'); sys.exit()
if args[:2]==['-m','venv']:
    root=pathlib.Path(args[-1]); (root/'bin').mkdir(parents=True)
    shutil.copyfile(exe,root/'bin/python'); (root/'bin/python').chmod(0o755)
    sys.exit()
root=exe.parent.parent
assert os.environ['AMDGPU_TARGET']=='gfx1151'
assert os.environ['PYTORCH_ROCM_ARCH']=='gfx1151'
assert os.environ['MAX_JOBS']=='4'
assert os.environ['PIP_NO_INDEX']=='1'
if fault=='timeout': time.sleep(20)
if fault=='log': print('x'*2048)
if args[:2]==['-m','pip']:
    assert '--no-index' in args or args[2] in ('check','freeze')
    if fault=='pip': sys.exit(3)
    if args[2]=='wheel':
        path=pathlib.Path(args[args.index('--wheel-dir')+1]); path.mkdir(exist_ok=True)
        (path/'sglang.whl').write_text('built')
    if fault=='mutation':
        source=root.parent/'source/sgl-kernel/setup_rocm.py'
        if source.exists(): source.write_text(source.read_text()+'\n# mutation\n')
    if args[2]=='freeze': print('torch==2.13.0+rocm7.2')
elif 'setup_rocm.py' in args:
    assert 'bdist_wheel' in args
    assert os.environ['PYTORCH_NVCC']=='ccache hipcc'
    (pathlib.Path.cwd()/'build.ninja').write_text('nvcc = ccache hipcc\n')
    path=pathlib.Path(args[args.index('--dist-dir')+1]); path.mkdir(exist_ok=True)
    (path/'sglang_kernel.whl').write_text('built')
    if fault=='runtime-mutation':
        (pathlib.Path(os.environ['ROCM_PATH'])/'lib/libamdhip64.so.7').write_bytes(b'changed')
    if fault=='compiler-mode-mutation':
        (pathlib.Path(os.environ['ROCM_PATH'])/'bin/hipcc').chmod(0o644)
    if fault=='clang-mutation':
        (pathlib.Path(os.environ['ROCM_PATH'])/'llvm/bin/clang++').write_bytes(b'changed compiler')
    if fault=='ninja-mutation':
        (pathlib.Path(os.environ['ROCM_PATH'])/'bin/ninja').write_text('changed')
    if fault=='ninja-selected-path':
        toolkit=pathlib.Path(os.environ['ROCM_PATH'])
        (toolkit/'bin/ninja').symlink_to(toolkit/'ninja-original')
    if fault=='ninja-resolved-path':
        toolkit=pathlib.Path(os.environ['ROCM_PATH'])
        replacement=toolkit/'ninja-replacement'
        replacement.write_bytes((toolkit/'ninja-original').read_bytes())
        replacement.chmod(0o755)
        (toolkit/'bin/ninja').unlink()
        (toolkit/'bin/ninja').symlink_to(replacement)
elif '-c' in args:
    lib=root/'lib'; lib.mkdir(exist_ok=True)
    paths={}
    for name in ('torch','triton','sglang','sgl_kernel'):
        path=lib/(name+'.py'); path.write_text('identity'); paths[name]=str(path)
    extension=lib/'common_ops.so'; extension.write_text('extension')
    if fault=='package-symlink-target' and not (lib/'package-link').is_symlink():
        (lib/'package-link').symlink_to('sgl_kernel.py')
    if fault=='outside': paths['triton']='/outside/triton/__init__.py'
    print(json.dumps({'hip':None if fault=='cuda' else '7.2','cuda':None,
        'arch':'gfx942' if fault=='arch' else 'gfx1151',
        'torch':'wrong' if fault=='torch-version' else '2.13.0+rocm7.2',
        'triton':'wrong' if fault=='triton-version' else '3.7.1',
        'paths':paths,'extension':str(extension),'shared_memory_per_block':65536,
        'triton_distributions':['triton-rocm','triton'] if fault=='triton-overlap' else ['triton-rocm']}))
elif args[:2]==['-m','pytest']:
    assert not list((root/'lib').glob('common_ops.so.*')), 'collection rejected polluted common_ops package'
    bf16=next(pathlib.Path(a) for a in args if a.endswith('test_qwen3_bf16_silu.py'))
    assert bf16.is_file()
    path=pathlib.Path(next(a.split('=',1)[1] for a in args if a.startswith('--junitxml=')))
    path.write_text('<testsuites><testsuite tests="'+('2' if fault=='count' else '1059')+'" failures="0" errors="0" skipped="'+('1' if fault=='skip' else '0')+'"/></testsuites>')
else: raise RuntimeError(args)
'''

POLICY = r"""
import json, os, runpy, sys, types
setup, arch = sys.argv[1:]
# CPU adaptation executes the exact upstream setup entrypoint. No GPU is used.
for name in ("torch", "torch.utils", "torch.utils.cpp_extension", "setuptools"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["torch"].cuda = types.SimpleNamespace(
    is_available=lambda: True,
    get_device_properties=lambda i: types.SimpleNamespace(gcnArchName=arch + ":xnack-"))
sys.modules["setuptools"].find_packages = lambda **kw: []
def finish(**kw):
    print(json.dumps({"flags": kw["ext_modules"][0]["extra_compile_args"]["nvcc"]}))
sys.modules["setuptools"].setup = finish
sys.modules["torch.utils.cpp_extension"].BuildExtension = types.SimpleNamespace(with_options=lambda **kw: None)
sys.modules["torch.utils.cpp_extension"].CUDAExtension = lambda **kw: kw
os.environ["AMDGPU_TARGET"] = "gfx942"
# Only the version file is a harness adaptation, independent of target policy.
from pathlib import Path
Path(setup).with_name("pyproject.toml").write_text('version = "0.4.4"\n')
runpy.run_path(setup, run_name="__main__")
"""
