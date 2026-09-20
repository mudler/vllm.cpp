#!/usr/bin/env python3
"""Prepare the explicitly patched SGLang comparator inside a Strix lease (#3053).

The JSON manifest supplies source {revision,path,sha256}, an absolute Python
interpreter, rocm {root,compiler,runtime} with path/sha256 file bindings,
dependency wheels [{path,sha256}], expected {torch,triton}, and
resource limits. Only the bound wheels enter the isolated environment.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.bench.strix_vllm_oracle.worker import (  # shared bounded CPU/process primitives
    Session, digest, extract, inventory, save, verify_file,
)

PIN = "f63458b5beaceabbd9d749b9fc956370e1b649e6"
PATCH = Path(__file__).with_name("patches") / "sglang-f63458b5-gfx1151.patch"
SETUP = "sgl-kernel/setup_rocm.py"
BEFORE = "8a37e537157e76cd2064e971842b9f7f7c8d075942b980f478c945afd876d6ac"
AFTER = "5fe84d7afb969073bff8b1bfcb0e224c36964aca0e7b025ea25caec55407fe9a"
GPU_TESTS = ("test_activation.py", "test_topk.py")
# Pinned upstream cases: 3 * 7 * 5 * 9 activation; 32 + 48 + 32 TopK; 2 BF16.
EXPECTED_TESTS = 1059

# Additional BF16 coverage, separate from the unchanged upstream FP16 suite.
# Reference: f63458b5 sgl-kernel/tests/test_activation.py:10-18. Shapes are the
# Qwen3-4B 9728-wide MLP at decode concurrency 1 and 4. PyTorch's BF16 default
# tolerance applies to this added dtype, not to the upstream FP16 test.
BF16_TEST = '''import pytest
import torch
import sgl_kernel

@pytest.mark.parametrize("tokens", [1, 4])
def test_qwen3_bf16_silu(tokens):
    torch.manual_seed(3053)
    x = torch.randn(tokens, 2 * 9728, device="cuda", dtype=torch.bfloat16)
    expected = x[..., 9728:] * torch.nn.functional.silu(x[..., :9728])
    actual = sgl_kernel.silu_and_mul(x)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (tokens, 9728)
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-5)
'''


def apply_compatibility_patch(source, env=None):
    before = inventory(source)
    if before.get(SETUP) != BEFORE:
        raise ValueError("upstream setup hash mismatch")
    for flags in (("--check",), ()):
        subprocess.run(["git", "apply", *flags, "--whitespace=error", str(PATCH)],
                       cwd=source, env=env, check=True, capture_output=True, timeout=30)
    after = inventory(source)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed != [SETUP] or after[SETUP] != AFTER:
        raise ValueError("unexpected compatibility patch source mutation")
    return dict(sha256=digest(PATCH), changed=changed, before=before, after=after)


IDENTITY = r'''
import importlib, importlib.metadata as metadata, json, sys
import torch, triton
paths = {"torch": torch.__file__, "triton": triton.__file__}
driver = triton.runtime.driver.active
if driver.get_current_target().backend != "hip":
    raise ValueError("ROCm identity requires the HIP backend")
# Triton 3.5.1 AMD driver.c:199-209 returns hipDeviceProp_t.sharedMemPerBlock.
properties = driver.utils.get_device_properties(0)
arch = torch.cuda.get_device_properties(0).gcnArchName
if properties["arch"].split(":")[0] != arch.split(":")[0]:
    raise ValueError("HIP and Torch device architecture mismatch")
data = dict(hip=torch.version.hip, cuda=torch.version.cuda,
            arch=arch, shared_memory_per_block=properties["max_shared_mem"],
            shared_memory_source="triton.runtime.driver.active.utils.get_device_properties(0).max_shared_mem",
            torch=torch.__version__, triton=triton.__version__, paths=paths,
            triton_distributions=[d.metadata["Name"] for d in metadata.distributions()
                                  if d.metadata["Name"].lower().replace("_", "-")
                                  in ("triton", "triton-rocm", "pytorch-triton-rocm")])
if sys.argv[1] == "complete":
    for name in ("sglang", "sgl_kernel"):
        paths[name] = importlib.import_module(name).__file__
    import sgl_kernel
    data["extension"] = sgl_kernel.common_ops.__file__
print(json.dumps(data))
'''


def identity(session, python, phase):
    data = json.loads(session.command("identity-" + phase, [python, "-I", "-c", IDENTITY, phase]))
    expected = session.manifest["expected"]
    if type(data["shared_memory_per_block"]) is not int or data["shared_memory_per_block"] < 65536:
        raise ValueError("ROCm shared-memory limit must be an integer of at least 65536 bytes")
    if (not data["hip"] or data["cuda"] is not None or data["arch"].split(":")[0] != "gfx1151"
            or data["torch"] != expected["torch"] or data["triton"] != expected["triton"]
            or len(data["triton_distributions"]) != 1):
        raise ValueError("ROCm identity mismatch or competing Triton distributions")
    paths = list(data["paths"].values())
    if phase == "complete":
        paths.append(data["extension"])
    if not all(Path(p).resolve().is_relative_to((session.local / "venv").resolve()) for p in paths):
        raise ValueError("ROCm identity imported outside isolated venv")
    save(session.output / ("identity-" + phase + ".json"), data)
    return data


def rocm_runtime(manifest):
    """Verify the selected toolkit without consulting inherited search paths."""
    try:
        record = manifest["rocm"]
        root_path = Path(record["root"])
        if not root_path.is_absolute() or not root_path.is_dir():
            raise ValueError("absolute existing root required")
        root = root_path.resolve(strict=True)
        seen = set()

        def bound_file(binding):
            path = Path(binding["path"])
            if not path.is_absolute() or not path.is_file():
                raise ValueError("absolute existing file required")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("file outside selected root")
            if resolved in seen:
                raise ValueError("duplicate file binding")
            seen.add(resolved)
            verify_file(binding)
            return dict(path=str(path), realpath=str(resolved), sha256=binding["sha256"])

        compiler = bound_file(record["compiler"])
        if Path(compiler["path"]).name != "hipcc" or not os.access(compiler["realpath"], os.X_OK):
            raise ValueError("bound compiler must be executable hipcc")
        if not isinstance(record["runtime"], list) or not record["runtime"]:
            raise ValueError("runtime file bindings required")
        runtime = [bound_file(binding) for binding in record["runtime"]]

        def directories(names):
            paths = []
            for name in names:
                path = root / name
                if path.exists():
                    if not path.is_dir() or not path.resolve(strict=True).is_relative_to(root):
                        raise ValueError("directory outside selected root or not a directory")
                    paths.append(str(path))
            if not paths:
                raise ValueError("toolkit search directories missing")
            return paths

        executable_path = os.pathsep.join(directories(("bin", "llvm/bin")) + ["/usr/bin", "/bin"])
        library_path = os.pathsep.join(directories(("lib", "lib64")))
        selected = shutil.which("hipcc", path=executable_path)
        if selected is None or str(Path(selected).resolve()) != compiler["realpath"]:
            raise ValueError("search path does not select bound compiler")
        result = dict(root=str(root), compiler=compiler, runtime=runtime,
                      PATH=executable_path, LD_LIBRARY_PATH=library_path)
        clang = root / "llvm/bin/clang++"
        if clang.exists():
            if not clang.is_file() or not clang.resolve(strict=True).is_relative_to(root):
                raise ValueError("clang compiler outside selected root")
            result["clang"] = dict(path=str(clang), realpath=str(clang.resolve()), sha256=digest(clang))
        return result
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError("ROCm runtime: " + str(error)) from error


def inspect_extension(session, captured):
    """Keep llvm-objdump's extracted bundles outside the installed package."""
    extension = Path(captured["extension"])
    package = Path(captured["paths"]["sgl_kernel"]).parent

    def package_inventory():
        return {str(p.relative_to(package)): (
            ["symlink", os.readlink(p)] if p.is_symlink() else
            ["file", digest(p)] if p.is_file() else ["directory"])
            for p in sorted(package.rglob("*"))}

    before = package_inventory()
    installed_hash = digest(extension)
    scratch = Path(tempfile.mkdtemp(prefix="extension-inspection-", dir=session.local))
    copied = scratch / extension.name
    shutil.copyfile(extension, copied)
    copied_hash = digest(copied)
    evidence = dict(source=str(extension), inspection=str(copied),
                    installed_sha256=installed_hash, inspected_sha256=copied_hash,
                    package_before=before)
    save(session.output / "extension-inspection.json", evidence)
    if copied.is_symlink() or copied_hash != installed_hash:
        raise ValueError("extension inspection copy hash mismatch")
    try:
        targets = session.command("extension-targets", ["llvm-objdump", "--offloading", copied], cwd=scratch)
    finally:
        after = package_inventory()
        evidence["package_after"] = after
        save(session.output / "extension-inspection.json", evidence)
        if digest(extension) != installed_hash or after != before:
            raise ValueError("installed extension or package changed during inspection")
    if set(re.findall(r"gfx[0-9a-f]+", targets)) != {"gfx1151"}:
        raise ValueError("extension architecture mismatch")
    return evidence


def prepare(manifest, output):
    if os.environ.get("RC_DEVICE") != "strix:gpu0" or not os.environ.get("RC_JOB_ID"):
        raise ValueError("preparation requires a strix:gpu0 lease")
    uuid.UUID(os.environ["RC_JOB_ID"])
    for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "PYTORCH_NVCC", "HIPCC_COMPILE_FLAGS_APPEND"):
        if os.environ.get(key):
            raise ValueError("inherited compiler override: " + key)
    source_record = manifest["source"]
    if source_record["revision"] != PIN:
        raise ValueError("source pin mismatch")
    verify_file(source_record)
    with tarfile.open(source_record["path"]) as archive:
        if archive.pax_headers.get("comment", "").strip() != PIN:
            raise ValueError("source archive revision marker mismatch")
    dependencies = manifest["dependencies"]
    if not dependencies:
        raise ValueError("explicit dependency wheels required")
    names = set()
    for record in dependencies:
        path = Path(record["path"])
        if path.suffix != ".whl" or path.name in names or path.name.lower().startswith("nvidia_"):
            raise ValueError("unbound, duplicate, or CUDA dependency wheel")
        names.add(path.name)
        verify_file(record)
    interpreter = Path(manifest["python"])
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise ValueError("absolute Python interpreter required")
    rocm = rocm_runtime(manifest)
    local = Path(tempfile.mkdtemp(prefix="strix-sglang3053-", dir="/tmp"))
    session = Session(manifest, output, local)
    try:
        session.env = {k: v for k, v in session.env.items() if not k.startswith("VLLM_")}
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "HIP_CLANG_PATH"):
            session.env.pop(key, None)
        session.env.update(PATH=rocm["PATH"], LD_LIBRARY_PATH=rocm["LD_LIBRARY_PATH"],
                           ROCM_PATH=rocm["root"], ROCM_HOME=rocm["root"], HIP_PATH=rocm["root"])
        session.env.update(AMDGPU_TARGET="gfx1151", PYTORCH_ROCM_ARCH="gfx1151",
                           # AMD Torch 2.9.1+rocm7.2.1 cpp_extension.py:2842 selects this before HIP.
                           PYTORCH_NVCC="ccache hipcc",
                           PIP_NO_INDEX="1", SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0+f63458b5")
        save(output / "environment.json", {k: v for k, v in session.env.items()
             if k not in os.environ or k in ("RC_DEVICE", "RC_JOB_ID")})
        save(output / "manifest.json", manifest)
        save(output / "rocm.json", rocm)
        session.command("hipcc", ["hipcc", "--version"])
        session.command("ccache", ["ccache", "--version"])
        source = local / "source"
        extract(source_record, source)
        patch = apply_compatibility_patch(source, session.env)
        save(output / "patch.json", patch)
        # Exact metadata substitutions from docker/rocm.Dockerfile:295-300.
        for directory, config in (("sgl-kernel", "pyproject_rocm.toml"), ("python", "pyproject_other.toml")):
            shutil.copyfile(source / directory / config, source / directory / "pyproject.toml")
        prepared = inventory(source)
        wheelhouse = local / "dependencies"
        wheelhouse.mkdir()
        for record in dependencies:
            target = wheelhouse / Path(record["path"]).name
            shutil.copyfile(record["path"], target)
            verify_file(record, target)
        session.command("venv", [interpreter, "-m", "venv", local / "venv"])
        python = local / "venv/bin/python"
        session.command("dependencies", [python, "-m", "pip", "install", "--no-index", "--no-deps",
                                         *sorted(wheelhouse.glob("*.whl"))])
        session.command("dependency-check", [python, "-m", "pip", "check"])
        identity(session, python, "dependencies")
        # cpp_extension.py:2295,2564 invokes bare ninja from this fixed PATH,
        # not necessarily the Ninja distribution installed in the venv.
        ninja_path = shutil.which("ninja", path=session.env["PATH"])
        if ninja_path is None:
            raise ValueError("Ninja executable missing from build PATH")
        ninja = dict(path=ninja_path, realpath=str(Path(ninja_path).resolve()),
                     sha256=digest(ninja_path),
                     version=session.command("ninja", [ninja_path, "--version"]).strip())
        save(output / "ninja.json", ninja)
        built = local / "built-wheels"
        built.mkdir()
        session.command("kernel-build", [python, "setup_rocm.py", "bdist_wheel", "--dist-dir", built],
                        cwd=source / "sgl-kernel")
        hip_rules = {str(p.relative_to(source)): digest(p) for p in (source / "sgl-kernel").rglob("build.ninja")
                     if "nvcc = ccache hipcc" in p.read_text()}
        if not hip_rules:
            raise ValueError("HIP build did not use the ccache compiler hook")
        kernel_wheels = list(built.glob("*.whl"))
        if len(kernel_wheels) != 1:
            raise ValueError("one built kernel wheel required")
        session.command("kernel-install", [python, "-m", "pip", "install", "--no-index", "--no-deps", kernel_wheels[0]])
        session.command("sglang-build", [python, "-m", "pip", "wheel", "--no-index", "--no-deps",
                                         "--no-build-isolation", "--wheel-dir", built, "."], cwd=source / "python")
        sglang_wheels = [p for p in built.glob("*.whl") if p not in kernel_wheels]
        if len(sglang_wheels) != 1:
            raise ValueError("one built SGLang wheel required")
        session.command("sglang-install", [python, "-m", "pip", "install", "--no-index", "--no-deps", sglang_wheels[0]])
        session.command("hip-extras", [python, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
                                       "--find-links", built, "sglang[srt_hip]"])
        session.command("pip-check", [python, "-m", "pip", "check"])
        captured = identity(session, python, "complete")
        inspection = inspect_extension(session, captured)
        report = local / "upstream-tests.xml"
        bf16_test = local / "test_qwen3_bf16_silu.py"
        bf16_test.write_text(BF16_TEST)
        session.command("upstream-tests", [python, "-m", "pytest", "-q",
                        *[str(source / "sgl-kernel/tests" / p) for p in GPU_TESTS], bf16_test, "--junitxml=" + str(report)],
                        timeout=session.limits["test_timeout"])
        suites = list(ET.parse(report).getroot().iter("testsuite"))
        if (not suites or sum(int(s.get("tests", "0")) for s in suites) != EXPECTED_TESTS
                or any(int(s.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped"))):
            raise ValueError("upstream tests did not all execute successfully")
        shutil.copyfile(report, output / "upstream-tests.xml")
        if any(not (source / p).is_file() or digest(source / p) != h for p, h in prepared.items()):
            raise ValueError("unexplained source mutation during preparation")
        for record in dependencies:
            verify_file(record)
        verify_file(source_record)
        freeze = session.command("freeze", [python, "-m", "pip", "freeze", "--all"])
        (output / "installed.txt").write_text(freeze)
        if rocm_runtime(manifest) != rocm:
            raise ValueError("ROCm runtime changed during preparation")
        if (shutil.which("ninja", path=session.env["PATH"]) != ninja["path"]
                or str(Path(ninja["path"]).resolve()) != ninja["realpath"]
                or digest(ninja["path"]) != ninja["sha256"]):
            raise ValueError("Ninja changed during preparation")
        save(output / "success.json", dict(comparator="patched SGLang", source_revision=PIN,
             rocm=rocm, ninja=ninja,
             source_archive_sha256=source_record["sha256"], patch=patch, local=str(local),
             hip_build_rules=hip_rules,
             build_metadata={p: prepared[p] for p in ("sgl-kernel/pyproject.toml", "python/pyproject.toml")},
             bf16_test_sha256=digest(bf16_test),
             identity=captured, extension_sha256=digest(captured["extension"]), extension_inspection=inspection,
             built_wheels=inventory(built), prepare_sha256=digest(__file__),
             shared_runner_sha256=digest(Path(sys.modules[Session.__module__].__file__))))
    finally:
        session.preserve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Do not write a failure record into a previous result.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        prepare(json.loads(args.manifest.read_text()), args.output)
    except Exception as error:
        save(args.output / "failure.json", {"error": str(error), "comparator": "patched SGLang"})
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
