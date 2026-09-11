#!/usr/bin/env python3
"""Build and measure the #3015 diagnostic inside the operator's Strix lease.

Inputs are explicit pins. Commands are arrays, never interpolated shell code.
Whole-generation timing includes prefill. No result establishes token parity.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

PROMPT = "The capital of France is"
MODEL_SIZE = 17106775008
MODEL_SHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
LLAMA_REV = "10bf611e533d81f739128304991c5e133c6aebd8"
SWITCHES = ("VT_ROCM_Q8K_BLOCK", "VT_ROCM_Q6K_SMALL_PRIVATE")
PODMAN = ["podman", "--storage-driver=vfs", "--root", "/tmp/podman-pr66-root-vfs",
          "--runroot", "/tmp/podman-pr66-run-vfs"]
TUNING_PREFIXES = ("VT_", "GGML_", "HSA_", "HIP_", "ROCR_", "PYTORCH_")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, sha256, size=None):
    path = Path(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"size mismatch: {path}")
    actual = digest(path)
    if actual != sha256:
        raise ValueError(f"sha256 mismatch: {path}: {actual} != {sha256}")
    return actual


def verify_archive(pin, engine):
    verify_file(pin["archive"], pin["sha256"])
    if not re.fullmatch(r"[0-9a-f]{40}", pin["revision"]):
        raise ValueError("full source revision required")
    if engine == "llamacpp" and pin["revision"] != LLAMA_REV:
        raise ValueError("llama.cpp pin differs from b10451")
    with Path(pin["archive"]).open("rb") as stream:
        actual = subprocess.check_output(["git", "get-tar-commit-id"], stdin=stream, text=True).strip()
    if actual != pin["revision"]:
        raise ValueError(f"archive revision mismatch for {engine}")


def parse_llama(text, prompt_tokens):
    # b10451 common/sampling.cpp:559 and :575 print the actual sampled and
    # prompt-evaluated counts. completion.cpp:684 decrements n_remain per sample.
    samples = re.findall(r"common_perf_print:\s+samplers time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens", text)
    prompts = re.findall(r"common_perf_print:\s+prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens", text)
    if samples != ["64"] or prompts != [str(prompt_tokens)]:
        raise ValueError("llama completion or prompt workload mismatch")
    return {"completion_tokens": 64, "prompt_tokens": prompt_tokens}


def parse_ours(text, repeat=4):
    pattern = (rf"vllm-cli: run=(\d+)/{repeat} finish_reason=(\S+) prompt_tokens=(\d+) "
               r"completion_tokens=(\d+) secs=([\d.]+) tok_s=[\d.]+")
    timing = re.findall(pattern, text)
    windows = re.findall(rf"vllm-cli: run=(\d+)/{repeat} generate_start_unix=([\d.]+) "
                         r"generate_end_unix=([\d.]+)", text)
    if [int(x[0]) for x in timing] != list(range(1, repeat + 1)) or [int(x[0]) for x in windows] != list(range(1, repeat + 1)):
        raise ValueError("incomplete or duplicated generation records")
    result = []
    for row, window in zip(timing, windows):
        run, reason, prompt, count, seconds = row
        start, end, seconds = float(window[1]), float(window[2]), float(seconds)
        if reason != "length" or int(count) != 64 or int(prompt) <= 0:
            raise ValueError("workload mismatch or early EOS")
        if not all(math.isfinite(v) for v in (start, end, seconds)) or seconds <= 0 or end <= start:
            raise ValueError("invalid generation time")
        if abs((end - start) - seconds) > 0.01:
            raise ValueError("wall and monotonic duration disagree")
        if result and start < result[-1]["end"]:
            raise ValueError("generation windows overlap")
        result.append(dict(run=int(run), prompt_tokens=int(prompt), completion_tokens=int(count),
                           seconds=seconds, start=start, end=end))
    if len({r["prompt_tokens"] for r in result}) != 1:
        raise ValueError("prompt workload mismatch")
    return result


def fold_clocks(runs, samples):
    retained = []
    for run in runs[1:]:
        window = [s["sclk_mhz"] for s in samples
                  if run["start"] <= s["timestamp"] <= run["end"] and s["sclk_mhz"] is not None]
        if not window:
            raise ValueError("warm generation has no valid clock samples")
        retained.extend(window)
    return {"window": "warm whole generations 2 through 4, Unix epoch timestamps",
            "sample_count": len(retained), "sclk_mhz_median": statistics.median(retained),
            "sclk_mhz_min": min(retained), "sclk_mhz_max": max(retained)}


def fold_pairs(legs):
    expected = {(s, p, a) for s in SWITCHES for p in range(3) for a in ("default", "candidate")}
    keys = [(x["switch"], x["pair"], x["arm"]) for x in legs]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("incomplete or duplicate paired legs")
    prompts = {r["prompt_tokens"] for leg in legs for r in leg["runs"]}
    if len(prompts) != 1:
        raise ValueError("paired prompt workload mismatch")
    for leg in legs:
        if leg["returncode"] != 0 or len(leg["runs"]) != 4:
            raise ValueError("failed or incomplete leg")
        if any(r["completion_tokens"] != 64 for r in leg["runs"]):
            raise ValueError("paired completion workload mismatch")
    by_key = dict(zip(keys, legs))
    comparisons = []
    for switch in SWITCHES:
        for pair in range(3):
            control = by_key[switch, pair, "default"]
            candidate = by_key[switch, pair, "candidate"]
            if control["output_sha256"] != candidate["output_sha256"]:
                raise ValueError("emitted cold output mismatch")
            a = statistics.median(r["seconds"] for r in control["runs"][1:])
            b = statistics.median(r["seconds"] for r in candidate["runs"][1:])
            comparisons.append(dict(switch=switch, pair=pair, default_seconds=a,
                                    candidate_seconds=b, default_over_candidate=a / b))
    return {"token_gate": "FAIL (carried, not remeasured)",
            "measurement": "unprofiled warm whole completion, includes prefill",
            "output_check": "cold text bytes only; CLI does not emit warm text or token IDs",
            "warm_output_equality": "PENDING", "pairs": comparisons}


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def inspect_image(manifest):
    image = subprocess.check_output(PODMAN + ["image", "inspect", "--format", "{{.Id}}", manifest["image"]], text=True).strip()
    env = json.loads(subprocess.check_output(PODMAN + ["image", "inspect", "--format", "{{json .Config.Env}}", image], text=True)) or []
    tuning = [value for value in env if value.split("=", 1)[0].startswith(TUNING_PREFIXES)]
    if tuning:
        raise ValueError(f"image tuning variables: {tuning}")
    return image, env


def check_live_faults(captures):
    """Scan new child output, retaining diagnostic fragments across reads."""
    for capture in captures:
        reader = capture["reader"]
        end = os.fstat(reader.fileno()).st_size
        while reader.tell() < end:
            chunk = reader.read(min(65536, end - reader.tell()))
            if not chunk:
                break
            text = capture["tail"] + chunk
            match = re.search(rb"GPU Hang|Memory access fault|HW Exception", text)
            if match:
                raise ValueError(f"fatal GPU diagnostic in {reader.name}: {match.group().decode()}")
            capture["tail"] = text[-64:]


def managed_run(argv, *, stdout, stderr, timeout, output_dir=None, max_bytes=536870912):
    """Stop the named container on every exit, including timeout and output growth.

    The aggregate output threshold is sampled every 100 ms, so an active writer
    can exceed it between samples. The container's per-file limit also applies.
    """
    name = argv[argv.index("--name") + 1]
    process = None
    captures = []
    started = time.monotonic()
    try:
        for stream in (stdout, stderr):
            if isinstance(getattr(stream, "name", None), (str, os.PathLike)):
                stream.flush()
                reader = Path(stream.name).open("rb")
                reader.seek(0, os.SEEK_END)
                captures.append({"reader": reader, "tail": b""})
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL)
        while True:
            if output_dir is not None:
                size = sum(p.stat().st_size for p in Path(output_dir).rglob("*") if p.is_file() and not p.is_symlink())
                if size > max_bytes:
                    raise ValueError(f"aggregate output exceeds {max_bytes} bytes")
            check_live_faults(captures)
            status = process.poll()
            if status is not None:
                check_live_faults(captures)
                return subprocess.CompletedProcess(argv, status)
            if time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(0.1)
    finally:
        for capture in captures:
            capture["reader"].close()
        failures = []
        for cleanup in (["stop", "--ignore", "--time", "2", name], ["rm", "--force", "--ignore", name]):
            try:
                subprocess.run(PODMAN + cleanup, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15, check=True)
            except (subprocess.SubprocessError, OSError) as exc:
                failures.append(str(exc))
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        if failures:
            raise RuntimeError("container cleanup failed: " + "; ".join(failures))


def command(argv, log, timeout=3600):
    print(json.dumps({"command": list(map(str, argv)), "log": str(log)}), flush=True)
    with Path(log).open("w") as stream:
        stream.write(json.dumps(list(map(str, argv))) + "\n")
        stream.flush()
        result = managed_run(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT,
                             timeout=timeout, output_dir=Path(log).parent)
        result.check_returncode()


def container(manifest, local, entrypoint, args, env=None, limit_output=False):
    result = PODMAN + ["run", "--rm", "--name", "strix-3015-" + uuid.uuid4().hex,
                      "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
                      "-v", f"{local}:{local}:rw", "-v", "/workspace/ccache:/workspace/ccache:rw"]
    if limit_output:
        result += ["--ulimit", "fsize=536870912:536870912"]
    for key, value in (env or {}).items():
        result += ["-e", f"{key}={value}"]
    return result + ["--entrypoint", str(entrypoint), manifest["image"], *map(str, args)]


def build(manifest, output):
    local = Path(tempfile.mkdtemp(prefix="strix-3015-", dir="/tmp"))
    print(f"Fresh worker directory: {local}", flush=True)
    state = {"local": str(local), "manifest": manifest, "binaries": {}, "sources": {},
             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             "rc_job_id": os.environ.get("RC_JOB_ID"), "created_unix": time.time()}
    save(output / "build-state.partial.json", state)
    image, image_env = inspect_image(manifest)
    state["image_id"] = image
    state["image_env"] = image_env
    runtime = dict(manifest, image=image)
    command(container(runtime, local, "ccache", ["--version"]), output / "ccache-version.log")
    command(container(runtime, local, "/opt/rocm/lib/llvm/bin/clang++", ["--version"]), output / "compiler-version.log")
    for engine in ("vllmcpp", "llamacpp"):
        pin = manifest["sources"][engine]
        verify_archive(pin, engine)
        source, target = local / engine, local / f"build-{engine}"
        source.mkdir()
        with tarfile.open(pin["archive"], "r:") as archive:
            archive.extractall(source, filter="data")
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                 "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache"]
        if engine == "vllmcpp":
            flags += ["-DVLLM_CPP_CUDA=OFF", "-DVLLM_CPP_HIP=ON", "-DVLLM_CPP_HIP_ARCHITECTURES=gfx1151",
                      "-DROCM_PATH=/opt/rocm", "-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++",
                      "-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4", "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/rocm/lib",
                      "-DVLLM_CPP_TRITON=OFF", "-DVLLM_CPP_SERVER=OFF", "-DVLLM_CPP_BUILD_EXAMPLES=ON",
                      "-DVLLM_CPP_BUILD_TESTS=OFF"]
            binary, build_target = target / "examples/vllm-cli", "vllm-cli"
        else:
            flags += ["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1151", "-DCMAKE_HIP_ARCHITECTURES=gfx1151",
                      "-DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang", "-DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++",
                      "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_CURL=OFF"]
            binary, build_target = target / "bin/llama-completion", "llama-completion"
        env = {"CCACHE_DIR": "/workspace/ccache", "CCACHE_BASEDIR": str(source)}
        command(container(runtime, local, "cmake", ["-S", source, "-B", target, *flags], env), output / f"{engine}-configure.log")
        command(container(runtime, local, "cmake", ["--build", target, "-j", "4", "--target", build_target], env), output / f"{engine}-build.log", 7200)
        state["sources"][engine] = pin
        paths = [binary] + sorted(p for p in target.rglob("*.so*") if p.is_file() and not p.is_symlink())
        state["binaries"][engine] = {str(p): digest(p) for p in paths}
        state[engine] = str(binary)
        shutil.copy2(target / "CMakeCache.txt", output / f"{engine}-CMakeCache.txt")
    model = local / "Qwen3.8-27B-Q4_K_M.gguf"
    shutil.copyfile(manifest["model"], model)
    verify_file(model, MODEL_SHA, MODEL_SIZE)
    state["model"] = str(model)
    save(output / "build-state.json", state)
    return state


def sample_clocks(stop, path, device):
    with path.open("w") as output:
        while not stop.is_set():
            before = time.time()
            try:
                raw = (device / "pp_dpm_sclk").read_text()
                match = re.search(r"(\d+)\s*[Mm][Hh][Zz].*\*", raw)
                clock = int(match[1]) if match else None
                busy = (device / "gpu_busy_percent").read_text().strip()
                error = None
            except OSError as exc:
                raw, clock, busy, error = None, None, None, str(exc)
            output.write(json.dumps(dict(timestamp=before, read_end_unix=time.time(),
                                         sclk_mhz=clock, sclk_raw=raw, busy_percent=busy, error=error)) + "\n")
            output.flush()
            stop.wait(0.25)


def measure(manifest, state, output, *, baseline_trace=True):
    local = Path(state["local"])
    if manifest != state["manifest"]:
        raise ValueError("measure manifest differs from build manifest")
    image, image_env = inspect_image(manifest)
    if image != state["image_id"]:
        raise ValueError("build image changed")
    save(output / "image-environment.json", {"image_id": image, "env": image_env})
    runtime = dict(manifest, image=image)
    for files in state["binaries"].values():
        for path, sha in files.items():
            verify_file(path, sha)
    verify_file(state["model"], MODEL_SHA, MODEL_SIZE)
    device = Path(manifest["clock_device"])
    if not (device / "pp_dpm_sclk").is_file():
        raise ValueError("clock device lacks pp_dpm_sclk")
    prefix = manifest["profiler_prefix"]
    if not isinstance(prefix, list) or not prefix or not any("{trace_dir}" in arg for arg in prefix):
        raise ValueError("profiler_prefix must be a command array with {trace_dir}")
    legs = []

    def run(engine, tag, repeat, tuning=None, profile=False):
        folder = local / tag
        folder.mkdir()
        env = {"LD_LIBRARY_PATH": f"{local}/build-{engine}:{local}/build-{engine}/bin:/opt/rocm/lib"}
        env.update(tuning or {})
        if engine == "vllmcpp":
            env["VT_OP_PROVIDER_STATS"] = "1"
            args = ["--model", state["model"], "--prompt", PROMPT, "--max-tokens", "64",
                    "--temperature", "0", "--repeat", str(repeat), "--max-num-seqs", "1"]
        else:
            args = ["-m", state["model"], "-p", PROMPT, "-n", "64", "-ngl", "99", "--temp", "0", "-no-cnv", "--seed", "1"]
        app = [state[engine], *args]
        if profile:
            app = [s.replace("{trace_dir}", str(folder / "trace")) for s in prefix] + app
        argv = container(runtime, local, app[0], app[1:], env, limit_output=True)
        save(folder / "command.json", {"argv": argv, "tuning": tuning, "profiled": profile,
                                      "prompt": PROMPT, "max_tokens": 64, "repeat": repeat})
        stop = threading.Event()
        sampler = threading.Thread(target=sample_clocks, args=(stop, folder / "clocks.jsonl", device))
        sampler.start()
        try:
            with (folder / "stdout").open("wb") as stdout, (folder / "stderr").open("wb") as stderr:
                completed = managed_run(argv, stdout=stdout, stderr=stderr, timeout=1200, output_dir=folder)
            save(folder / "exit.json", {"returncode": completed.returncode})
        finally:
            stop.set()
            sampler.join()
            shutil.copytree(folder, output / tag)
        text = (folder / "stderr").read_text(errors="replace")
        if completed.returncode != 0 or re.search(r"GPU Hang|Memory access fault|HW Exception|no kernel for op|\[vt reference-tier\]", text):
            raise ValueError(f"leg {tag} failed; see captured logs")
        if engine == "vllmcpp":
            runs = parse_ours(text, repeat)
            clocks = (fold_clocks(runs, [json.loads(line) for line in (folder / "clocks.jsonl").read_text().splitlines()])
                      if repeat == 4 else {"warm_clock_window": "not applicable to cold trace"})
            return dict(runs=runs, clocks=clocks, output_sha256=digest(folder / "stdout"), returncode=0)
        return {"stderr": text}

    if baseline_trace:
        # Capture the baseline before experimental switches can invalidate a leg.
        ours = run("vllmcpp", "trace-vllmcpp", 1, profile=True)
        llama = run("llamacpp", "trace-llamacpp", 1, profile=True)
        matched = parse_llama(llama["stderr"], ours["runs"][0]["prompt_tokens"])
        save(output / "trace-status.json", {"status": "PENDING manual kernel attribution",
                                           "matched_counts": matched,
                                           "trace_window": "whole process; includes load and prefill",
                                           "token_gate": "FAIL (carried, not remeasured)"})
    else:
        save(output / "trace-status.json", {"status": "PENDING #3040: trace not run in switches phase",
                                           "trace_run": False,
                                           "token_gate": "FAIL (carried, not remeasured)"})
    for switch in SWITCHES:
        for pair in range(3):
            order = ("default", "candidate") if pair % 2 == 0 else ("candidate", "default")
            for arm in order:
                tag = f"{switch}-{pair}-{arm}"
                leg = run("vllmcpp", tag, 4, {switch: "1"} if arm == "candidate" else {})
                legs.append(dict(switch=switch, pair=pair, arm=arm, **leg))
                save(output / "legs.json", legs)
    save(output / "paired-result.json", fold_pairs(legs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("build", "measure", "switches"), required=True)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if os.environ.get("RC_DEVICE") != "strix:gpu0" or not os.environ.get("RC_JOB_ID"):
        raise ValueError("requires operator-owned rc lease on strix:gpu0")
    inherited = {k: v for k, v in os.environ.items() if k.startswith(TUNING_PREFIXES)}
    if inherited:
        raise ValueError(f"inherited tuning variables: {inherited}")
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / f"{args.phase}-identity.json", {"rc_device": os.environ["RC_DEVICE"], "rc_job_id": os.environ["RC_JOB_ID"],
                                                    "uname": list(os.uname()), "inherited_tuning": inherited})
    if args.phase == "build":
        build(manifest, args.output)
    else:
        if not args.state:
            parser.error(f"{args.phase} requires --state from this build")
        measure(manifest, json.loads(args.state.read_text()), args.output,
                baseline_trace=args.phase == "measure")


if __name__ == "__main__":
    main()
