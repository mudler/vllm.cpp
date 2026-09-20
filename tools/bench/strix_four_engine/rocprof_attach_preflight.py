#!/usr/bin/env python3
"""Prepare the pinned patched controller and call its read-only public preflight.

This is a controller entry wrapper, not a replacement ELF or loader resolver.
Acceptance requires a complete report AND process exit zero. It authorizes no
later unchecked attachment. The patched attachment entry repeats all checks.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PIN = "97f5574fe2fdc7bef44fb01545347912ee9f1779"
PATCH = Path(__file__).parent / "patches/rocprof-controller-preflight.patch"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    """Publish without overwriting an existing report; failures never return zero."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_controller_patch(clone, run):
    directory = "--directory=projects/rocprofiler-sdk"
    run(["git", "apply", directory, "--check", str(PATCH.resolve())], cwd=clone)
    run(["git", "apply", directory, str(PATCH.resolve())], cwd=clone)


def prepare(args):
    deadline = time.monotonic() + args.timeout
    commands = []

    def run(command, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("preparation time limit expired")
        commands.append(command)
        if kwargs.pop("capture_output", False):
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = subprocess.Popen(command, text=True, start_new_session=True, **kwargs)
        try:
            stdout, stderr = process.communicate(timeout=remaining)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except BaseException:
            # The command owns this process group. Build workers must not outlive
            # a failed preparation, even when they closed inherited output pipes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)
            raise

    source = args.source.resolve(strict=True)
    revision = run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    if revision != PIN:
        raise ValueError("source revision differs from the pinned profiler")
    dirty = run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True).stdout
    if dirty:
        raise ValueError("source checkout is not clean")
    if args.output.exists():
        raise ValueError("preparation output already exists")
    args.output.mkdir(parents=False)
    clone = args.output / "source"
    # The source and all its submodules remain unchanged. Submodule revisions come
    # from the pinned tree, not the remote branches that happen to exist today.
    run(["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(clone)])
    run(["git", "-C", str(clone), "checkout", "--detach", PIN])
    modules_path = "projects/rocprofiler-sdk/external"
    run(["git", "-C", str(clone), "submodule", "update", "--init", "--recursive", "--jobs", "4", "--", modules_path])
    modules = run(["git", "-C", str(clone), "submodule", "status", "--recursive", "--", modules_path], capture_output=True).stdout
    if any(line[0] != " " for line in modules.splitlines()):
        raise ValueError("submodule revision is not pinned")
    project = clone / "projects/rocprofiler-sdk"
    apply_controller_patch(clone, run)
    build = args.output / "build"
    configure = ["cmake", "-S", str(project), "-B", str(build), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                 "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", "-DROCPROFILER_BUILD_TESTS=OFF",
                 "-DROCPROFILER_BUILD_SAMPLES=OFF", "-DROCPROFILER_BUILD_BENCHMARK=OFF"]
    if args.prefix_path:
        configure.append("-DCMAKE_PREFIX_PATH=" + args.prefix_path)
    if args.openssl_root:
        configure.append("-DOPENSSL_ROOT_DIR=" + str(args.openssl_root.resolve(strict=True)))
    env = dict(os.environ, CCACHE_DIR=str(args.output / "ccache"))
    with (args.output / "build.log").open("x") as log:
        run(configure, env=env, stdout=log, stderr=subprocess.STDOUT)
        run(["cmake", "--build", str(build), "--target", "rocprofv3-attach", "-j", "4"],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    libraries = {p.resolve() for p in build.rglob("librocprofv3-attach.so.1")}
    if len(libraries) != 1:
        raise ValueError("build did not produce one attachment controller")
    library = libraries.pop()
    linked = run(["ldd", str(library)], capture_output=True).stdout
    if "not found" in linked:
        raise ValueError("controller dependency is unavailable")
    dependencies = []
    for line in linked.splitlines():
        candidates = [word for word in line.split() if word.startswith("/")]
        for filename in candidates:
            path = Path(filename).resolve(strict=True)
            dependencies.append({"path": str(path), "sha256": sha256(path)})
    report = {"schema": 1, "status": "BUILT_NOT_HARDWARE_VALIDATED", "upstream_revision": PIN,
              "patch_sha256": sha256(PATCH), "controller": str(library),
              "controller_sha256": sha256(library), "submodules": modules,
              "commands": commands, "dependencies": dependencies}
    publish(args.output / "build.json", report)
    print(json.dumps(report, sort_keys=True))


def validate_report(report, pid):
    """Validate the controller's evidence, without duplicating its ELF resolver."""
    def require(ok):
        if not ok:
            raise ValueError("incomplete or inconsistent controller preflight report")

    def uint(value):
        return type(value) is int and 0 <= value < 2**64

    require(type(report) is dict)
    require(report.get("status") == "PASS" and type(report.get("pid")) is int
            and report["pid"] == pid and report.get("target_mutated") is False)
    start = report.get("start_time")
    require(type(start) is str and re.fullmatch(r"[0-9]+", start) is not None
            and 0 < int(start) < 2**64)
    for key in ("controller", "target", "sdk"):
        obj = report.get(key)
        require(type(obj) is dict)
        path = obj.get("path")
        require(type(path) is str and path.startswith("/") and "\0" not in path)
        digest, build = obj.get("sha256"), obj.get("build_id")
        require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)
        require(type(build) is str and re.fullmatch(r"(?:[0-9a-f]{2})+", build) is not None)
        require(uint(obj.get("load_bias")))
    for key in ("sha256", "build_id"):
        require(report["controller"][key] == report["target"][key])
    for key in ("attach_offset", "detach_offset"):
        offset = report.get(key)
        require(uint(offset) and offset > 0)
        for obj in ("controller", "target"):
            require(report[obj]["load_bias"] + offset < 2**64)


def preflight(args):
    if not 0 < args.pid <= 2147483647:
        raise ValueError("invalid target process identifier")
    controller = args.controller.resolve(strict=True)
    if args.report.exists():
        raise ValueError("preflight report already exists")
    before = sha256(controller)
    lib = ctypes.CDLL(str(controller), mode=ctypes.RTLD_GLOBAL)
    entry = lib.preflight
    entry.argtypes = [ctypes.c_uint32, ctypes.c_int]
    entry.restype = ctypes.c_int
    with tempfile.TemporaryFile() as stream:
        result = entry(args.pid, stream.fileno())
        if result != 0:
            raise ValueError("actual controller rejected preflight")
        stream.seek(0)
        report = json.load(stream)
    validate_report(report, args.pid)
    if sha256(controller) != before:
        raise ValueError("controller changed during preflight")
    dependencies = []
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            paths.add(fields[5])
    for filename in sorted(paths):
        path = Path(filename).resolve(strict=True)
        dependencies.append({"path": str(path), "sha256": sha256(path)})
    report.update(schema=1, upstream_revision=PIN, patch_sha256=sha256(PATCH),
                  controller_binary=str(controller), controller_sha256=before,
                  controller_process_dependencies=dependencies,
                  disposition="SNAPSHOT_ONLY_NOT_ATTACHMENT_AUTHORITY")
    publish(args.report, report)
    print(json.dumps({"status": "PASS", "report": str(args.report),
                      "sha256": sha256(args.report)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--prefix-path", default="")
    build.add_argument("--openssl-root", type=Path)
    build.add_argument("--timeout", type=int, default=1200, choices=range(1, 1201), metavar="SECONDS")
    check = commands.add_parser("preflight")
    check.add_argument("--controller", required=True, type=Path)
    check.add_argument("--pid", required=True, type=int)
    check.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else preflight)(args)
        return 0
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
        print("profiler preflight: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
