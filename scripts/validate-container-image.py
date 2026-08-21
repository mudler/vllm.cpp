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
4. HUB REACH -- whether the image can actually TALK to huggingface.co. It boots
   the image with `--model does-not-exist/nope` and requires the failure to be
   an ANSWER FROM THE HUB: a completed TLS session that ended in an HTTP
   status. MEASURED on 20 August 2026, the live hub answers 401 rather than 404
   for an unknown repository asked anonymously (`GET
   /api/models/does-not-exist/nope/refs` -> 401), so both statuses count as
   verified and the audit says which it saw. ENG-HF-MODEL-DOWNLOAD (#1280) makes
   `--model org/repo` fetch a checkpoint, and that needs transport layer
   security plus a trust store. Both are BUILD-TIME and IMAGE-TIME decisions
   that can silently resolve OFF: `VLLM_CPP_HF_DOWNLOAD` downgrades itself with
   a warning when no OpenSSL is found, and `ca-certificates`/`libssl3` can be
   dropped from a layer. A symbol check on the binary passes in every one of
   those states. Only a completed round trip to the hub does not.

The boot audit is OPTIONAL and its absence is reported, never silently skipped:
an image validated without --model has NO runtime evidence, which is exactly
what the release contract means by build-verified but not runtime-verified. The
hub-reach audit runs by default, needs network egress from the container, and
reports UNVERIFIED rather than OK when the container could not reach the hub at
all. An unknown is not a pass.

Usage:
  scripts/validate-container-image.py --image REF --lane cpu --version 0.0.1 \
      [--model DIR] [--expect-revision SHA] [--skip-hub-reach]
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
    image: str,
    model: Path,
    port: int,
    timeout: float,
    gpus: str | None = None,
    runtime: str | None = None,
) -> list[str]:
    errors: list[str] = []
    name = f"vllm-cpp-smoke-{port}"

    # --gpus is what turns a build result into RUNTIME evidence for an
    # accelerator lane: without it the cuda image runs its CPU paths and proves
    # nothing about the GPU it was built for. The driver still comes from the
    # host through the container runtime; the image never carries one.
    gpu_args = ["--gpus", gpus] if gpus else []
    # Tegra/L4T needs --runtime nvidia and REJECTS --gpus outright:
    #   "invoking the NVIDIA Container Runtime Hook directly (e.g. specifying
    #    the docker --gpus flag) is not supported"
    # so a GPU lane cannot be validated on Jetson through --gpus alone.
    if runtime:
        gpu_args = ["--runtime", runtime, *gpu_args]

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
        # `-t`, not `--timeout`: the long form only exists on newer Docker (the
        # Jetson node runs 27.5.1 and rejects it), and `-t` is accepted by both.
        code, output = run(["docker", "stop", "-t", "30", name])
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


# The repository identifier the hub-reach audit asks for. It is a well-formed
# `org/repo` -- `IsValidHfRepoId` accepts it -- that does not exist, so the hub
# ANSWERS rather than serving, and nothing is downloaded. MEASURED on
# 20 August 2026 the answer is HTTP 401, not 404: an anonymous `GET
# /api/models/does-not-exist/nope/refs` cannot be told apart from a request for
# a private repository, so the hub declines to say which. `classify_hub_reach`
# accepts 401, 403 and 404 for that reason, and the module docstring above and
# `HUB_ANSWER_AUTH` below record the same measurement.
HUB_REACH_MODEL = "does-not-exist/nope"

# The message `HfRefuseHttpsWithoutTls` prints when the build carries no
# transport layer security. Seeing THIS instead of a hub answer is the exact
# failure this audit exists for: the image shipped with the feature disabled.
NO_TLS_MARKERS = ("cannot speak HTTPS", "VLLM_CPP_OPENSSL", "VLLM_CPP_HF_DOWNLOAD")

# A completed HTTP conversation with the hub. 401 is what the live hub actually
# returned when this was measured, and 403 and 404 are accepted as proof of the
# same thing -- the handshake finished and the hub judged the request -- because
# a mirror may answer an unknown, gated or renamed name any of those ways, and
# each one rules out both failure modes the audit is looking for.
HUB_ANSWER_404 = "HuggingFace answered HTTP 404"
HUB_ANSWER_AUTH = "HuggingFace refused repository"

# `HfParseUrl`/`ApiGet` report a transport failure this way. It means the
# container never got an answer: no egress, no DNS, a proxy, or a missing trust
# store. That is not evidence either way about the build, so it is UNVERIFIED.
HUB_UNREACHABLE = "cannot reach https://"


def classify_hub_reach(code: int, output: str) -> tuple[str, str]:
    """Classify one `--model does-not-exist/nope` run.

    Returns (status, detail) where status is "ok", "fail" or "unverified".
    Pure string work, so tests/scripts/test_validate_container_image.py can
    drive every branch without a docker daemon.
    """
    if code == 0:
        return (
            "fail",
            "the container exited 0 for a repository that does not exist; the "
            "resolver did not run or did not refuse",
        )
    for marker in NO_TLS_MARKERS:
        if marker in output:
            return (
                "fail",
                "this image cannot speak HTTPS: the binary printed the build-option "
                f"message containing {marker!r}, so VLLM_CPP_HF_DOWNLOAD resolved "
                "OFF or no TLS library was found at build time. `--model org/repo` "
                "cannot reach huggingface.co from this image",
            )
    if HUB_ANSWER_404 in output:
        return ("ok", f"the hub answered HTTP 404 for {HUB_REACH_MODEL}")
    if HUB_ANSWER_AUTH in output:
        return (
            "ok",
            f"the hub answered with an authorization status for {HUB_REACH_MODEL}; "
            "the TLS session completed",
        )
    if HUB_UNREACHABLE in output:
        return (
            "unverified",
            "the container could not reach the hub at all (no egress, no DNS, or "
            "no trust store). This proves nothing about the build; rerun on a host "
            f"with network access. Output: {output.strip()[-400:]}",
        )
    return (
        "unverified",
        "the container failed for a reason this audit does not recognise, so it "
        f"says nothing about TLS. Output: {output.strip()[-400:]}",
    )


def check_hub_reach(image: str, timeout: float) -> tuple[str, str]:
    """Boot the image against an unknown repository and read what it refused with."""
    try:
        code, output = run(
            ["docker", "run", "--rm", image, "--model", HUB_REACH_MODEL],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # A hang is not a verdict. The server may be waiting on a proxy that
        # never answers, which says nothing about how it was built.
        return (
            "unverified",
            f"the container did not exit within {timeout:.0f}s, so the audit has "
            "no reading",
        )
    return classify_hub_reach(code, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expect-revision")
    parser.add_argument("--model", type=Path, help="model directory for the boot smoke")
    parser.add_argument(
        "--docker-runtime",
        help="pass through to `docker run --runtime` (e.g. nvidia). Required on "
        "Tegra/L4T, which rejects --gpus",
    )
    parser.add_argument(
        "--gpus",
        help="pass through to `docker run --gpus` (e.g. all) so the boot smoke is "
        "real runtime evidence for an accelerator lane",
    )
    parser.add_argument(
        "--skip-hub-reach",
        action="store_true",
        help="do not boot the image against huggingface.co. The absence is "
        "REPORTED, so an image validated this way carries no evidence that "
        "--model org/repo works",
    )
    parser.add_argument("--hub-reach-timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--boot-timeout", type=float, default=180.0)
    args = parser.parse_args()

    errors: list[str] = []
    errors += check_config(args.image, args.lane, args.version, args.expect_revision)
    errors += check_layout(args.image, args.version)

    hub_reach_status = "skipped"
    hub_reach_detail = "not run (--skip-hub-reach): this image has NO evidence that --model org/repo works"
    if not args.skip_hub_reach:
        hub_reach_status, hub_reach_detail = check_hub_reach(
            args.image, args.hub_reach_timeout
        )
        if hub_reach_status == "fail":
            errors.append(f"hub reach: {hub_reach_detail}")

    runtime_verified = False
    if args.model is not None:
        if not args.model.is_dir():
            errors.append(f"--model {args.model} is not a directory")
        else:
            boot_errors = check_boot(
                args.image,
                args.model.resolve(),
                args.port,
                args.boot_timeout,
                args.gpus,
                args.docker_runtime,
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
        selectors = []
        if args.docker_runtime:
            selectors.append(f"--runtime {args.docker_runtime}")
        if args.gpus:
            selectors.append(f"--gpus {args.gpus}")
        where = f"on {' '.join(selectors)}" if selectors else "on CPU paths only (no GPU)"
        print(
            f"  boot: /health 200, /version 200, declared healthcheck passed, clean "
            f"SIGTERM, {where}"
        )
    else:
        print("  boot: NOT RUN (no --model): this image has NO runtime evidence")
    if hub_reach_status == "ok":
        print(f"  hub reach: verified, {hub_reach_detail}")
    else:
        print(f"  hub reach: {hub_reach_status.upper()}, {hub_reach_detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
