#!/usr/bin/env python3
"""Validate a built vllm.cpp container image -- ENG-RELEASE-CONTAINERS.

Three audits, in the order a failure is cheapest to diagnose:

1. CONFIG -- what the image declares: entrypoint, uid, port, volumes,
   healthcheck, labels. Read from the image metadata, not from the Dockerfile,
   so a build that silently dropped an instruction is caught.
2. LAYOUT -- what the image contains: /opt/vllm is exactly the tree
   scripts/package-server.py stages, plus the lane's runtime libraries and
   nothing else. No compiler, no build tree, no source, no weights.
3. BOOT -- what the image does: with a model mounted it serves /health and
   /version and shuts down cleanly on SIGTERM.

The boot audit is OPTIONAL and its absence is reported, never silently skipped:
an image validated without --model has NO runtime evidence, which is exactly
what the release contract means by build-verified but not runtime-verified.

Usage:
  scripts/validate-container-image.py --image REF --lane cpu --version 0.0.1 \
      [--model DIR] [--expect-revision SHA]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The staged tree scripts/package-server.py produces, plus what the image adds
# on top of it. Anything else in /opt/vllm is undeclared and fails.
STAGED_FILES = frozenset(
    {
        "bin/vllm-server",
        "VERSION",
        "release-manifest.json",
        "sbom.spdx.json",
        "THIRD_PARTY_NOTICES",
    }
)
IMAGE_ADDED_FILES = frozenset({"bin/vllm-healthcheck"})
ALLOWED_PREFIXES = ("share/licenses/", "lib/")

REQUIRED_LABELS = (
    "org.opencontainers.image.source",
    "org.opencontainers.image.revision",
    "org.opencontainers.image.version",
    "io.vllm-cpp.lane",
    "io.vllm-cpp.channel",
)

# A runtime image that can compile is a build tree that escaped its stage.
FORBIDDEN_TOOLS = ("cc", "gcc", "g++", "cmake", "ninja", "nvcc", "ld")


def run(argv: list[str], **kwargs) -> tuple[int, str]:
    completed = subprocess.run(
        argv, capture_output=True, text=True, check=False, **kwargs
    )
    return completed.returncode, completed.stdout + completed.stderr


def in_image(image: str, script: str) -> tuple[int, str]:
    """Run a shell snippet inside the image with the entrypoint overridden."""
    return run(
        ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", script]
    )


def check_config(image: str, lane: str, version: str, revision: str | None) -> list[str]:
    errors: list[str] = []
    code, output = run(["docker", "image", "inspect", image])
    if code != 0:
        return [f"docker image inspect {image} failed: {output.strip()}"]

    config = json.loads(output)[0]["Config"]

    entrypoint = config.get("Entrypoint") or []
    if entrypoint != ["/opt/vllm/bin/vllm-server"]:
        errors.append(
            f"ENTRYPOINT must be ['/opt/vllm/bin/vllm-server'] so flags pass straight "
            f"through, found {entrypoint!r}"
        )
    if config.get("Cmd"):
        errors.append(
            f"image must not set CMD (found {config['Cmd']!r}): the server already "
            "defaults to 0.0.0.0:8000 and a CMD would be replaced by the user's flags"
        )

    if config.get("User") != "1000:1000":
        errors.append(
            f"image must run as 1000:1000, found {config.get('User')!r}; a server "
            "that runs as root in a container is an escalation waiting for a bug"
        )

    if "8000/tcp" not in (config.get("ExposedPorts") or {}):
        errors.append("image must EXPOSE 8000/tcp, the server's default bind port")

    volumes = config.get("Volumes") or {}
    for mount in ("/models", "/cache"):
        if mount not in volumes:
            errors.append(f"image must declare the {mount} volume")

    healthcheck = config.get("Healthcheck") or {}
    if not healthcheck.get("Test"):
        errors.append("image must declare a HEALTHCHECK against /health")

    labels = config.get("Labels") or {}
    for label in REQUIRED_LABELS:
        if not labels.get(label):
            errors.append(f"image is missing the required label {label}")
    if labels.get("io.vllm-cpp.lane") != lane:
        errors.append(
            f"io.vllm-cpp.lane must be {lane!r}, found {labels.get('io.vllm-cpp.lane')!r}"
        )
    if labels.get("org.opencontainers.image.version") != version:
        errors.append(
            f"image version label must be {version!r}, found "
            f"{labels.get('org.opencontainers.image.version')!r}"
        )
    if revision and labels.get("org.opencontainers.image.revision") != revision:
        errors.append(
            f"image revision label must be {revision!r}, found "
            f"{labels.get('org.opencontainers.image.revision')!r}"
        )

    return errors


def check_layout(image: str, version: str) -> list[str]:
    errors: list[str] = []

    code, listing = in_image(image, "cd /opt/vllm && find . -type f | sed 's|^\\./||' | sort")
    if code != 0:
        return [f"could not list /opt/vllm inside {image}: {listing.strip()}"]

    found = {line.strip() for line in listing.splitlines() if line.strip()}
    missing = sorted((STAGED_FILES | IMAGE_ADDED_FILES) - found)
    if missing:
        errors.append(f"/opt/vllm is missing staged files: {missing}")

    undeclared = sorted(
        name
        for name in found
        if name not in STAGED_FILES
        and name not in IMAGE_ADDED_FILES
        and not name.startswith(ALLOWED_PREFIXES)
    )
    if undeclared:
        errors.append(
            f"/opt/vllm carries undeclared files: {undeclared}; the image is the staged "
            "bundle and nothing else"
        )

    if not any(name.startswith("share/licenses/") for name in found):
        errors.append("/opt/vllm carries no share/licenses entries")

    code, staged_version = in_image(image, "cat /opt/vllm/VERSION")
    if code != 0:
        errors.append("could not read /opt/vllm/VERSION")
    elif version not in staged_version:
        errors.append(
            f"staged VERSION does not carry {version!r}; the image and its tag disagree "
            f"about what was built. VERSION says: {staged_version.strip()[:200]!r}"
        )

    code, manifest = in_image(image, "cat /opt/vllm/release-manifest.json")
    if code != 0:
        errors.append("could not read /opt/vllm/release-manifest.json")
    else:
        try:
            json.loads(manifest)
        except json.JSONDecodeError as error:
            errors.append(f"release-manifest.json is not valid JSON: {error}")

    code, _ = in_image(image, "test -x /opt/vllm/bin/vllm-server")
    if code != 0:
        errors.append("/opt/vllm/bin/vllm-server is missing or not executable")

    code, ffmpeg = in_image(image, "command -v ffmpeg")
    if code != 0:
        errors.append(
            "ffmpeg does not resolve on PATH: /v1/videos cannot succeed in this image, "
            "and a container has no host PATH to inherit"
        )

    code, uid = in_image(image, "id -u")
    if code != 0 or uid.strip() != "1000":
        errors.append(f"container does not run as uid 1000, found {uid.strip()!r}")

    for tool in FORBIDDEN_TOOLS:
        code, _ = in_image(image, f"command -v {tool}")
        if code == 0:
            errors.append(
                f"runtime image can run {tool!r}: a compiler or linker in the runtime "
                "stage means the build tree escaped its stage"
            )

    for path in ("/src", "/opt/vllm/build", "/root/.cache"):
        code, _ = in_image(image, f"test -e {path}")
        if code == 0:
            errors.append(f"runtime image still carries {path}")

    code, objects = in_image(
        image, "find /opt/vllm \\( -name '*.o' -o -name '*.a' \\) 2>/dev/null | head -5"
    )
    if objects.strip():
        errors.append(f"/opt/vllm carries build objects: {objects.split()}")

    # 2>/dev/null matters: the container runs as uid 1000, so an unfiltered find
    # floods stderr with "Permission denied" for /root and friends, and treating
    # that noise as a match reports baked weights in an image that has none.
    code, weights = in_image(
        image,
        "find / -xdev \\( -name '*.safetensors' -o -name '*.gguf' \\) 2>/dev/null | head -3",
    )
    if weights.strip():
        errors.append(f"image bakes model weights: {weights.split()}; weights are mounted")

    return errors


def wait_for_health(url: str, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last = "no attempt made"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True, response.read().decode("utf-8", "replace")
                last = f"HTTP {response.status}"
        except (urllib.error.URLError, OSError) as error:
            last = str(error)
        time.sleep(2)
    return False, last


def check_boot(
    image: str, model: Path, port: int, timeout: float, gpus: str | None = None
) -> list[str]:
    errors: list[str] = []
    name = f"vllm-cpp-smoke-{port}"

    # --gpus is what turns a build result into RUNTIME evidence for an
    # accelerator lane: without it the cuda image runs its CPU paths and proves
    # nothing about the GPU it was built for. The driver still comes from the
    # host through the container runtime; the image never carries one.
    gpu_args = ["--gpus", gpus] if gpus else []

    run(["docker", "rm", "--force", name])
    code, output = run(
        [
            "docker", "run", "--detach", "--name", name,
            *gpu_args,
            "--publish", f"127.0.0.1:{port}:8000",
            "--volume", f"{model}:/models/smoke:ro",
            image,
            "--model", "/models/smoke",
        ]
    )
    if code != 0:
        return [f"container did not start: {output.strip()}"]

    try:
        healthy, detail = wait_for_health(f"http://127.0.0.1:{port}/health", timeout)
        if not healthy:
            _, logs = run(["docker", "logs", "--tail", "40", name])
            errors.append(
                f"/health never returned 200 within {timeout:.0f}s ({detail}). "
                f"Container logs:\n{logs}"
            )
            return errors

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/version", timeout=10) as r:
                if r.status != 200:
                    errors.append(f"/version returned HTTP {r.status}")
        except (urllib.error.URLError, OSError) as error:
            errors.append(f"/version failed: {error}")

        # The HEALTHCHECK the image declares must itself pass inside the container,
        # not merely exist: an unrunnable healthcheck marks the container unhealthy
        # forever and takes an orchestrator's rollout with it.
        code, output = run(["docker", "exec", name, "/opt/vllm/bin/vllm-healthcheck"])
        if code != 0:
            errors.append(f"declared HEALTHCHECK command failed inside the container: {output.strip()}")

        # SIGTERM, not SIGKILL: an image that has to be killed loses in-flight work
        # on every ordinary orchestrator restart.
        code, output = run(["docker", "stop", "--timeout", "30", name])
        if code != 0:
            errors.append(f"docker stop failed: {output.strip()}")
        else:
            _, state = run(["docker", "inspect", "--format", "{{.State.ExitCode}}", name])
            exit_code = state.strip()
            if exit_code not in {"0", "143"}:
                errors.append(
                    f"container exited {exit_code} on SIGTERM; a clean shutdown is 0 "
                    "(or 143 when the signal itself terminates the process)"
                )
    finally:
        run(["docker", "rm", "--force", name])

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expect-revision")
    parser.add_argument("--model", type=Path, help="model directory for the boot smoke")
    parser.add_argument(
        "--gpus",
        help="pass through to `docker run --gpus` (e.g. all) so the boot smoke is "
        "real runtime evidence for an accelerator lane",
    )
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--boot-timeout", type=float, default=180.0)
    args = parser.parse_args()

    errors: list[str] = []
    errors += check_config(args.image, args.lane, args.version, args.expect_revision)
    errors += check_layout(args.image, args.version)

    runtime_verified = False
    if args.model is not None:
        if not args.model.is_dir():
            errors.append(f"--model {args.model} is not a directory")
        else:
            boot_errors = check_boot(
                args.image, args.model.resolve(), args.port, args.boot_timeout, args.gpus
            )
            errors += boot_errors
            runtime_verified = not boot_errors

    if errors:
        print(f"ERROR: {args.image} failed container validation:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"container image OK: {args.image} lane={args.lane} version={args.version}")
    print(f"  config, layout: verified")
    if runtime_verified:
        where = f"on --gpus {args.gpus}" if args.gpus else "on CPU paths only (no --gpus)"
        print(
            f"  boot: /health 200, /version 200, declared healthcheck passed, clean "
            f"SIGTERM, {where}"
        )
    else:
        print("  boot: NOT RUN (no --model): this image has NO runtime evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
