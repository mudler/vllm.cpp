#!/usr/bin/env python3
"""Isolated current-pin Strix oracle build and run worker (#3043)."""
import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
import zipfile

PYTHON = sys.executable
TOOLCHAIN = None
RUNTIME = Path(__file__).with_name("runtime.py")
VERSION = "0.28.1rc1.dev132+ge126687a9"
DEFAULT_LIMITS = dict(build_timeout=7200, run_timeout=3600, test_timeout=1800,
                      min_mem_bytes=6442450944, min_disk_bytes=21474836480,
                      max_log_bytes=536870912)

VLLM_REV = "e126687a9a828d513c01a07cd69f025f27d63280"
PLUGIN_REV = "d4c1f0d082fc7cd4350da56689109a01c1f29d6c"
PLUGIN_SHA = "9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b"
MODEL_SIZE = 17106775008
MODEL_SHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
SELECTED_TRITON_SHA = "e91ffa46d095b252248297292dd22bcbacd53a125a0c2eefbbbf74925a320bc3"
TUNING = ("VT_", "GGML_", "HSA_", "HIP_", "ROCR_", "PYTORCH_", "VLLM_", "TORCH_", "TRITON_", "CUDA_", "CMAKE_")
INJECTION = {"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def manifest_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_file(record, path=None):
    path = Path(path or record["path"])
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ValueError(f"size mismatch: {path}")
    if digest(path) != record["sha256"]:
        raise ValueError(f"sha256 mismatch: {path}")


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive or asset path: {name}")
    return path


def archive_members(archive):
    members = archive.getmembers()
    for member in members:
        path = safe_relative(member.name)
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise ValueError("unsafe archive link")
            parts = list(path.parent.parts if member.issym() else ())
            for part in target.parts:
                if part == "..":
                    if not parts:
                        raise ValueError("unsafe archive link")
                    parts.pop()
                elif part != ".":
                    parts.append(part)
        elif not (member.isfile() or member.isdir()):
            raise ValueError("unsafe archive special file")
    return members


def verify_sources(manifest):
    for name, revision in (("vllm", VLLM_REV), ("plugin", PLUGIN_REV)):
        record = manifest["sources"][name]
        if record["revision"] != revision:
            raise ValueError("source pin mismatch")
        if name == "plugin" and record["sha256"] != PLUGIN_SHA:
            raise ValueError("plugin archive pin mismatch")
        verify_file(record)
        with tarfile.open(record["path"]) as archive:
            if name == "vllm" and archive.pax_headers.get("comment", "").strip() != revision:
                raise ValueError("archive commit marker mismatch")
            archive_members(archive)


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and not p.is_symlink()}


def extract(record, target):
    verify_file(record)
    target.mkdir()
    with tarfile.open(record["path"]) as archive:
        archive.extractall(target, members=archive_members(archive), filter="data")


def installed_inventory(identity):
    result = {}
    for name in ("vllm_path", "plugin_path"):
        root = Path(identity[name]).parent
        result[str(root)] = inventory(root)
    for path in identity["extensions"]:
        result[path] = digest(path)
    return result


def selected_triton_wheel(manifest):
    record = manifest["dependencies"]["selected_triton"]
    if (record.get("distribution") != "triton" or record.get("version") != "3.8.0"
            or record.get("sha256") != SELECTED_TRITON_SHA):
        raise ValueError("selected Triton wheel pin mismatch")
    verify_file(record)
    metadata_root = "triton-3.8.0.dist-info"
    with zipfile.ZipFile(record["path"]) as wheel:
        files = {}
        for member in wheel.infolist():
            path = safe_relative(member.filename)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (not path.parts or path.parts[0] not in ("triton", metadata_root)
                    or "\\" in member.filename or str(path) != member.filename.rstrip("/")
                    or mode not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("unsafe selected Triton wheel member")
            if member.is_dir():
                continue
            if member.filename in files:
                raise ValueError("duplicate selected Triton wheel member")
            files[member.filename] = member
        record_name = metadata_root + "/RECORD"
        if record_name not in files or metadata_root + "/METADATA" not in files:
            raise ValueError("selected Triton wheel metadata missing")
        metadata = BytesParser().parsebytes(wheel.read(metadata_root + "/METADATA"))
        if metadata.get("Name", "").replace("_", "-").lower() != "triton" or metadata.get("Version") != "3.8.0":
            raise ValueError("selected Triton wheel metadata mismatch")
        rows = {}
        for row in csv.reader(wheel.read(record_name).decode().splitlines()):
            if len(row) != 3 or row[0] in rows:
                raise ValueError("invalid selected Triton RECORD")
            rows[row[0]] = row[1:]
        if set(rows) != set(files) or rows[record_name] != ["", ""]:
            raise ValueError("selected Triton RECORD coverage mismatch")
        namespace = {}
        modes = {}
        for name, member in files.items():
            if name == record_name:
                continue
            with wheel.open(member) as stream:
                sha = hashlib.file_digest(stream, "sha256")
            encoded = base64.urlsafe_b64encode(sha.digest()).decode().rstrip("=")
            if rows[name] != ["sha256=" + encoded, str(member.file_size)]:
                raise ValueError("selected Triton RECORD hash or size mismatch")
            if name.startswith("triton/"):
                namespace[name.removeprefix("triton/")] = sha.hexdigest()
                modes[name.removeprefix("triton/")] = (member.external_attr >> 16) & 0o777
        if ("__init__.py" not in namespace or not any(p.startswith("backends/amd/") for p in namespace)
                or not any(".so" in Path(p).name for p in namespace)):
            raise ValueError("selected Triton compiler/backend files missing")
    return record, namespace, modes


def compiler_inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("unsafe selected Triton namespace")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("unsafe selected Triton namespace member")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def compiler_modes(root):
    return {str(p.relative_to(root)): p.stat().st_mode & 0o7777
            for p in Path(root).rglob("*") if p.is_file()}


def validate_identity(identity, venv, expected, selection=None):
    if (identity.get("platform") != "rocm"
            or identity.get("device_arch", "").split(":")[0] != "gfx1151"
            or identity.get("torch") != expected["torch"]
            or identity.get("torchvision") != expected["torchvision"]
            or identity.get("triton") != expected["triton_runtime"]
            or identity.get("triton_rocm_distribution") != expected["triton-rocm"]
            or identity.get("triton_distribution") != (expected.get("triton") if selection else None)):
        raise ValueError("runtime identity mismatch")
    for key in ("runtime_version", "distribution_version"):
        if not re.fullmatch(re.escape(VERSION) + r"(?:\.rocm[0-9]+)?", identity.get(key, "")):
            raise ValueError("runtime identity version mismatch")
    paths = [identity["vllm_path"], identity["plugin_path"], *identity["extensions"]]
    if len(identity["extensions"]) < 2 or not all(Path(p).resolve().is_relative_to(venv.resolve()) for p in paths):
        raise ValueError("runtime identity imported outside isolated venv")
    if not all(identity.get("plugin_predicates", {}).get(k) is True for k in ("Q4_K", "Q5_K", "Q6_K", "Q8_0")):
        raise ValueError("runtime identity lacks native plugin predicates")
    if selection and (identity.get("triton") != selection["version"]
                      or identity.get("triton_path") != str(Path(selection["namespace"]) / "__init__.py")):
        raise ValueError("selected Triton imported path/version mismatch")


class Session:
    def __init__(self, manifest, output, local=None):
        self.manifest = manifest
        self.output = output
        self.local = Path(local or tempfile.mkdtemp(prefix="strix-vllm3043-", dir="/tmp"))
        self.triton_selection = None
        self.logs = self.local / ("logs-" + output.name)
        self.logs.mkdir()
        self.limits = {**DEFAULT_LIMITS, **manifest.get("limits", {})}
        if any(type(v) not in (int, float) or v <= 0
               or (type(v) is float and not math.isfinite(v)) for v in self.limits.values()):
            raise ValueError("invalid resource limits")
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("PIP_", "PYTHON")) and k != "VIRTUAL_ENV"}
        self.env.update(PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:" + os.environ.get("PATH", ""),
                        MAX_JOBS="4", CMAKE_BUILD_PARALLEL_LEVEL="4", PYTORCH_ROCM_ARCH="gfx1151",
                        VLLM_TARGET_DEVICE="rocm", VLLM_USE_PRECOMPILED="0", VLLM_USE_PRECOMPILED_RUST="0",
                        VLLM_DISABLE_SCCACHE="1", CMAKE_C_COMPILER_LAUNCHER="ccache",
                        CMAKE_CXX_COMPILER_LAUNCHER="ccache", CMAKE_HIP_COMPILER_LAUNCHER="ccache",
                        SETUPTOOLS_SCM_PRETEND_VERSION=VERSION, PIP_DISABLE_PIP_VERSION_CHECK="1",
                        PIP_CONFIG_FILE=os.devnull,
                        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for variable, folder in (("PIP_CACHE_DIR", "pip-cache"), ("CCACHE_DIR", "ccache"),
                                 ("XDG_CACHE_HOME", "cache"), ("HF_HOME", "hf-cache"), ("TMPDIR", "tmp")):
            path = self.local / folder
            path.mkdir(exist_ok=True)
            self.env[variable] = str(path)
        save(output / "environment.json", {k: self.env[k] for k in self.env if k not in os.environ or k in ("RC_DEVICE", "RC_JOB_ID")})

    def headroom(self):
        match = re.search(r"^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)
        if not match or int(match[1]) * 1024 < self.limits["min_mem_bytes"]:
            raise ValueError("memory headroom exhausted")
        if min(shutil.disk_usage(self.local).free, shutil.disk_usage(self.output).free) < self.limits["min_disk_bytes"]:
            raise ValueError("disk headroom exhausted")

    def preserve(self):
        shutil.copytree(self.logs, self.output / "logs", dirs_exist_ok=True, symlinks=False)

    def command(self, name, argv, *, cwd=None, timeout=None):
        self.headroom()
        path = self.logs / (name + ".log")
        save(self.logs / (name + ".command.json"), {"argv": list(map(str, argv)), "cwd": str(cwd or self.local),
                                                   "timeout": timeout or self.limits["build_timeout"]})
        started = time.monotonic()
        process = None
        status = None
        tail = b""
        preserved = started
        try:
            with path.open("wb") as writer, path.open("rb") as reader:
                process = subprocess.Popen(list(map(str, argv)), cwd=cwd or self.local, env=self.env,
                                           stdout=writer, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
                while True:
                    self.headroom()
                    now = time.monotonic()
                    if now - started > (timeout or self.limits["build_timeout"]):
                        raise TimeoutError(f"command timeout: {name}")
                    if now - preserved >= 1:
                        self.preserve()
                        preserved = now
                    if sum(p.stat().st_size for p in self.logs.glob("*.log")) > self.limits["max_log_bytes"]:
                        raise ValueError("log output limit exceeded")
                    chunk = reader.read(65536)
                    if chunk:
                        text = tail + chunk
                        if re.search(rb"GPU Hang|Memory access fault|HW Exception", text):
                            raise ValueError("fatal GPU diagnostic")
                        tail = text[-64:]
                        continue
                    self.preserve()
                    status = process.poll()
                    if status is not None:
                        # Drain after observing exit: a final diagnostic is still fatal.
                        chunk = reader.read()
                        if re.search(rb"GPU Hang|Memory access fault|HW Exception", tail + chunk):
                            raise ValueError("fatal GPU diagnostic")
                        break
                    time.sleep(0.1)
            if status:
                raise ValueError(f"command failed: {name}: {status}")
        finally:
            if process is not None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            save(self.logs / (name + ".exit.json"), {"returncode": status, "seconds": time.monotonic() - started})
            self.preserve()
        return path.read_text(errors="replace")

    def compiler_namespace(self, python, name):
        # -S prevents site/.pth startup from importing resolver-created code
        # before its compiler namespace has been verified. Python 3.12's -S
        # skips venv prefix setup, so bind the install scheme explicitly.
        value = self.command(name, [python, "-I", "-S", "-c",
            "import sys, sysconfig; print(sysconfig.get_path('purelib', vars={'base': sys.argv[1], 'platbase': sys.argv[1]}))",
            self.local / "venv"])
        site = Path(value.strip())
        if not site.is_absolute() or not site.resolve().is_relative_to((self.local / "venv").resolve()):
            raise ValueError("selected Triton namespace outside isolated venv")
        return site / "triton"

    def select_compiler(self, python):
        record, files, modes = selected_triton_wheel(self.manifest)
        namespace = self.compiler_namespace(python, "compiler-site-build")
        previous = compiler_inventory(namespace)
        quarantine = Path(tempfile.mkdtemp(prefix="compiler-quarantine-", dir=self.local)) / "triton"
        namespace.rename(quarantine)
        namespace.mkdir()
        with zipfile.ZipFile(record["path"]) as archive:
            for relative in files:
                target = namespace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("triton/" + relative) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                target.chmod(modes[relative])
        self.triton_selection = dict(namespace=str(namespace), wheel_sha256=record["sha256"],
            distribution="triton", version="3.8.0", files=files, modes=modes,
            native_files={p: sha for p, sha in files.items() if p.endswith(".so")},
            amd_files={p: sha for p, sha in files.items() if p.startswith("backends/amd/")},
            quarantine=str(quarantine), quarantine_files=previous,
            nonselected_metadata="triton-rocm distribution metadata is nonauthoritative for imported compiler bytes")
        self.verify_compiler(python, "compiler-site-selected")

    def verify_compiler(self, python, name):
        if not self.manifest["dependencies"].get("selected_triton"):
            if self.triton_selection is not None:
                raise ValueError("unexpected selected Triton state")
            return
        record, files, modes = selected_triton_wheel(self.manifest)
        proof = self.triton_selection
        namespace = self.compiler_namespace(python, name)
        if (not proof or proof.get("namespace") != str(namespace)
                or proof.get("wheel_sha256") != record["sha256"]
                or proof.get("distribution") != "triton" or proof.get("version") != "3.8.0"
                or proof.get("files") != files
                or proof.get("modes") != modes or compiler_modes(namespace) != modes
                or proof.get("native_files") != {p: sha for p, sha in files.items() if p.endswith(".so")}
                or proof.get("amd_files") != {p: sha for p, sha in files.items() if p.startswith("backends/amd/")}
                or compiler_inventory(namespace) != files):
            raise ValueError("selected Triton compiler tree/state mismatch")
        quarantine = Path(proof["quarantine"])
        if (not quarantine.resolve().is_relative_to(self.local.resolve())
                or compiler_inventory(quarantine) != proof["quarantine_files"]):
            raise ValueError("selected Triton quarantine mismatch")
        save(self.output / "triton-selection.json", proof)

    def probe(self, python, name):
        self.verify_compiler(python, name + "-compiler-site")
        report = self.local / (name + ".json")
        self.command(name, [python, RUNTIME, "--mode", "identity", "--output", report], timeout=120)
        identity = json.loads(report.read_text())
        validate_identity(identity, Path(python).parent.parent, self.manifest["dependencies"]["expected_versions"], self.triton_selection)
        shutil.copyfile(report, self.output / report.name)
        return identity


def build(manifest, output):
    session = Session(manifest, output)
    local = session.local
    save(output / "local.json", {"local": str(local)})
    session.headroom()
    sources = {}
    before = {}
    for name, record in manifest["sources"].items():
        sources[name] = local / name
        extract(record, sources[name])
        before[name] = inventory(sources[name])
    save(output / "source-before.json", before)
    venv = local / "venv"
    session.command("venv", [PYTHON, "-m", "venv", venv])
    python = venv / "bin/python"
    for name, command in (("compiler", "/opt/rocm/bin/hipcc"), ("ccache", "ccache")):
        session.command(name, [TOOLCHAIN or command, "--version"])
    version_file = Path("/opt/rocm/.info/version")
    if version_file.exists():
        shutil.copyfile(version_file, output / "rocm-version.txt")
    dependencies = manifest["dependencies"]
    verify_file(dependencies["requirements"])
    requirements = local / "requirements.txt"
    shutil.copyfile(dependencies["requirements"]["path"], requirements)
    constraints = local / "constraints.txt"
    expected = dependencies["expected_versions"]
    if expected.get("torch") != "2.13.0+rocm7.2" or any(not re.fullmatch(r"[0-9A-Za-z.+_-]+", v) for v in expected.values()):
        raise ValueError("invalid dependency version binding")
    constraints.write_text("\n".join(f"{name}=={expected[name]}" for name in ("torch", "torchvision", "triton-rocm", "triton") if name in expected) + "\n")
    indexes = dependencies["indexes"]
    if not indexes or any(not url.startswith("https://") for url in indexes):
        raise ValueError("explicit HTTPS dependency indexes required")
    resolver = ["--index-url", indexes[0]]
    for url in indexes[1:]:
        resolver += ["--extra-index-url", url]
    cache = local / "dependency-wheels"
    cache.mkdir()
    for wheel in dependencies.get("wheels", []):
        verify_file(wheel)
        target = cache / Path(wheel["path"]).name
        if not target.name.endswith(".whl") or target.exists():
            raise ValueError("invalid or duplicated dependency wheel")
        shutil.copyfile(wheel["path"], target)
        verify_file(wheel, target)
    resolver += ["--find-links", str(cache), "--constraint", str(constraints)]
    pip = [python, "-m", "pip"]
    session.command("bootstrap", pip + ["install", *resolver, "--report", output / "bootstrap-report.json", "-r", requirements])
    for name, source in sources.items():
        config = tomllib.loads((source / "pyproject.toml").read_text())
        required = config["build-system"]["requires"]
        if required:
            session.command(name + "-build-requires", pip + ["install", *resolver, "--report", output / (name + "-build-report.json"), *required])
    wheels = local / "wheels"
    wheels.mkdir()
    for record in dependencies.get("source_archives", []):
        name = record["name"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in sources:
            raise ValueError("invalid dependency source name")
        source = local / ("dep-" + name)
        extract(record, source)
        session.command(name + "-wheel", pip + ["wheel", "--no-build-isolation", "--no-deps", "--wheel-dir", wheels, source])
    for name, source in sources.items():
        session.command(name + "-wheel", pip + ["wheel", "--no-build-isolation", "--no-deps", "--wheel-dir", wheels, source])
    built_wheels = sorted(wheels.glob("*.whl"))
    if not any(p.name.startswith("vllm-") for p in built_wheels) or not any(p.name.startswith("vllm_gguf_plugin-") for p in built_wheels):
        raise ValueError("current source wheels missing")
    session.command("runtime-install", pip + ["install", *resolver, "--report", output / "runtime-install-report.json", *built_wheels])
    session.command("pip-check", pip + ["check"])
    if dependencies.get("selected_triton"):
        session.select_compiler(python)
        session.command("pip-check-selected", pip + ["check"])
    session.command("pip-freeze", pip + ["freeze", "--all"])
    after = {name: inventory(source) for name, source in sources.items()}
    save(output / "source-after.json", after)
    for name in sources:
        for path, sha in before[name].items():
            # setup.py get_vllm_version explicitly generates this file.
            if name == "vllm" and path == "vllm/_version.py":
                continue
            if after[name].get(path) != sha:
                raise ValueError(f"source changed during build: {name}/{path}")
    identity = session.probe(python, "build-identity")
    for i, path in enumerate(identity["extensions"]):
        text = session.command(f"offload-{i}", [TOOLCHAIN or "/opt/rocm/llvm/bin/llvm-objdump", "--offloading", path])
        targets = set(re.findall(r"amdhsa--(gfx[0-9a-z]+)", text))
        if targets != {"gfx1151"}:
            raise ValueError(f"extension device targets mismatch: {targets}")
    shutil.copytree(wheels, output / "wheels", symlinks=False)
    save(output / "wheel-hashes.json", {p.name: digest(p) for p in built_wheels})
    save(output / "dependency-wheel-hashes.json", inventory(cache))
    save(output / "build-state.json", {"status": "BUILT", "local": str(local),
         "manifest_sha256": manifest_digest(manifest), "identity": identity,
         "installed": installed_inventory(identity), "source_after": after,
         "worker_sha256": digest(__file__), "runtime_sha256": digest(RUNTIME),
         "triton_selection": session.triton_selection})


def run(manifest, state, output):
    local = Path(state["local"])
    if not local.resolve().is_relative_to("/tmp") or not local.name.startswith("strix-vllm3043-"):
        raise ValueError("invalid local build state")
    if state["worker_sha256"] != digest(__file__) or state["runtime_sha256"] != digest(RUNTIME):
        raise ValueError("worker state identity mismatch")
    if installed_inventory(state["identity"]) != state["installed"]:
        raise ValueError("installed identity mismatch")
    for name, recorded in state["source_after"].items():
        current = inventory(local / name)
        if any(current.get(path) != sha for path, sha in recorded.items()):
            raise ValueError("source identity mismatch")
    session = Session(manifest, output, local)
    session.triton_selection = state.get("triton_selection")
    python = local / "venv/bin/python"
    session.probe(python, "run-identity")
    model = manifest["model"]
    if model["sha256"] != MODEL_SHA or model["bytes"] != MODEL_SIZE:
        raise ValueError("model artifact pin mismatch")
    assets = Path(tempfile.mkdtemp(prefix="assets-", dir=local))
    records = [dict(model, relative_path="model/Qwen3.8-27B-Q4_K_M.gguf"), *manifest["assets"]]
    seen = set()
    for record in records:
        relative = safe_relative(record["relative_path"])
        if str(relative) in seen:
            raise ValueError("duplicate asset path")
        seen.add(str(relative))
        session.headroom()
        verify_file(record)
        destination = assets / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], destination)
        verify_file(record, destination)
    required = {"model/config.json", "tokenizer/config.json", "tokenizer/tokenizer_config.json",
                "tokenizer/tokenizer.json", "mmproj/mmproj-BF16.gguf"}
    if not required.issubset(seen):
        raise ValueError("required model assets missing")
    tokens = local / ("tokens-" + output.name + ".json")
    try:
        session.command("generation", [python, RUNTIME, "--mode", "generate", "--assets", assets,
                                        "--output", tokens], timeout=session.limits["run_timeout"])
    finally:
        if tokens.exists():
            shutil.copyfile(tokens, output / "tokens.json")
        session.verify_compiler(python, "post-generation-compiler-site")
    generated = json.loads(tokens.read_text())
    validate_identity(generated["identity"], local / "venv", manifest["dependencies"]["expected_versions"], session.triton_selection)
    from runtime import ENGINE_KWARGS, PROMPT_IDS
    if generated["engine_kwargs"] != ENGINE_KWARGS:
        raise ValueError("production compilation configuration mismatch")
    records = generated.get("records", [])
    if (len(records) != 6 or [r.get("prompt_ids") for r in records] != PROMPT_IDS
            or any(len(r.get("gen_ids", [])) != 48 or any(type(t) is not int or t < 0 for t in r["gen_ids"]) for r in records)):
        raise ValueError("six complete 48-token outputs required")
    if not generated.get("resolved_config", {}).get("dtype"):
        raise ValueError("resolved dtype missing")
    testdir = local / ("packed-tests-" + output.name)
    testdir.mkdir()
    testname = "test_fused_recurrent_packed_decode.py"
    source = local / "vllm/tests/kernels" / testname
    shutil.copyfile(source, testdir / testname)
    if digest(source) != digest(testdir / testname):
        raise ValueError("upstream test copy mismatch")
    report = output / "packed-tests.xml"
    try:
        session.command("packed-tests", [python, "-m", "pytest", str(testdir / testname), "-ra",
                                          "--junitxml=" + str(report)], cwd=testdir,
                        timeout=session.limits["test_timeout"])
    finally:
        session.verify_compiler(python, "post-tests-compiler-site")
    suites = ET.parse(report).getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    save(output / "upstream-tests.json", counts)
    if counts != dict(tests=8, failures=0, errors=0, skipped=0):
        raise ValueError("upstream tests incomplete, failed, or skipped")
    save(output / "result.json", {"gateability": "PASS", "token_gate": "FAIL (carried, not remeasured)",
                                   "upstream_tests": {"status": "PASS", **counts}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "run"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if os.environ.get("RC_DEVICE") != "strix:gpu0" or not os.environ.get("RC_JOB_ID"):
        raise ValueError("requires a strix:gpu0 lease")
    def interrupted(signum, _frame):
        raise RuntimeError(f"termination signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    tuning = sorted(k for k in os.environ if k.startswith(TUNING) or k in INJECTION)
    if tuning:
        raise ValueError(f"inherited tuning: {tuning}")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output already exists")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("manifest schema mismatch")
    if args.phase == "run":
        if not args.state:
            raise ValueError("run requires successful build state")
        state = json.loads(args.state.read_text())
        if state.get("status") != "BUILT":
            raise ValueError("run requires successful build state")
        if state.get("manifest_sha256") != manifest_digest(manifest):
            raise ValueError("build manifest mismatch")
    args.output.mkdir(parents=True)
    save(args.output / "manifest.json", manifest)
    save(args.output / "manifest-identity.json", {"sha256": manifest_digest(manifest)})
    try:
        verify_sources(manifest)
        if args.phase == "build":
            build(manifest, args.output)
        else:
            run(manifest, state, args.output)
    except Exception as exc:
        save(args.output / "failure.json", {"status": "FAILED", "phase": args.phase, "error": str(exc)})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
