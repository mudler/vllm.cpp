#!/usr/bin/env python3
"""Static least-privilege and immutable-handoff gate for W8 release.yml."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by the fail-closed caller path.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/release.yml"
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"


class WorkflowYamlError(ValueError):
    """The workflow is not an unambiguous YAML mapping."""


if yaml is not None:

    class UniqueWorkflowLoader(yaml.SafeLoader):
        """Safe YAML loader that rejects duplicate keys and references."""

        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(yaml.AliasEvent):
                raise WorkflowYamlError("YAML aliases are not allowed")
            event = self.peek_event()
            if getattr(event, "anchor", None) is not None:
                raise WorkflowYamlError("YAML anchors are not allowed")
            return super().compose_node(parent, index)

        def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
            if not isinstance(node, yaml.MappingNode):
                raise WorkflowYamlError("expected a YAML mapping")
            mapping: dict[Any, Any] = {}
            for key_node, value_node in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    raise WorkflowYamlError("YAML merge keys are not allowed")
                key = self.construct_object(key_node, deep=deep)
                try:
                    duplicate = key in mapping
                except TypeError as exc:
                    raise WorkflowYamlError("YAML mapping key is not scalar") from exc
                if duplicate:
                    raise WorkflowYamlError(f"duplicate YAML key {key!r}")
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping


def load_workflow_yaml(text: str) -> dict[Any, Any]:
    """Load one workflow without YAML features that obscure effective authority."""

    if yaml is None:
        raise WorkflowYamlError("PyYAML is required for workflow validation")
    try:
        document = yaml.load(text, Loader=UniqueWorkflowLoader)
    except yaml.YAMLError as exc:
        raise WorkflowYamlError(f"invalid workflow YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise WorkflowYamlError("workflow root must be a mapping")
    return document


def job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    return match.group(1) if match else ""


def action_steps(text: str, action: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^      - ", text)]
    starts.append(len(text))
    return [
        text[start:end]
        for start, end in zip(starts, starts[1:])
        if re.search(rf"(?m)^        uses: {re.escape(action)}$", text[start:end])
    ]


def step_mapping_blocks(step: str, name: str) -> list[str]:
    lines = step.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if line != f"        {name}:":
            continue
        end = index + 1
        while end < len(lines) and not re.match(r"^        \S", lines[end]):
            end += 1
        blocks.append("\n".join(lines[index + 1 : end]))
    return blocks


def mapping_value(step: str, mapping: str, key: str) -> str | None:
    blocks = step_mapping_blocks(step, mapping)
    if len(blocks) != 1:
        return None
    values = re.findall(
        rf"(?m)^          {re.escape(key)}:\s*([^\n]+?)\s*$", blocks[0]
    )
    return values[0] if len(values) == 1 else None


def mapping_paths(step: str) -> list[str]:
    blocks = step_mapping_blocks(step, "with")
    if len(blocks) != 1:
        return []
    lines = blocks[0].splitlines()
    for index, line in enumerate(lines):
        match = re.fullmatch(r"          path:\s*(.*)", line)
        if match is None:
            continue
        if match.group(1) != "|":
            return [match.group(1)]
        paths: list[str] = []
        for candidate in lines[index + 1 :]:
            if not candidate.startswith("            "):
                break
            paths.append(candidate.strip())
        return paths
    return []


def workflow_steps(block: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^      - ", block)]
    starts.append(len(block))
    return [block[start:end] for start, end in zip(starts, starts[1:])]


def validate_pr_ci(text: str) -> list[str]:
    """Require native Windows proof, without release authority, on every lane
    that can certify a tree.

    The `if:` below is part of the pinned schema, so it is the reason this
    function had to change for #503 at all. Until 2026-08-17 it read
    `github.event_name == 'pull_request'`, which is what kept `windows-msvc-cpu`
    and `windows-msvc-vulkan` off the schedule/dispatch lane that
    `scripts/main-baseline.py` grades -- and a baseline that never runs a
    compiling gate publishes GREEN for the wrong reason.

    WHAT THIS STILL REFUSES, and it is the whole point of pinning a literal
    rather than a substring: `always()`, `github.event_name != 'push'`, a bare
    `true`, or any other broadening still fails the equality, because the
    expected value is one exact string and not a predicate over strings. What
    changed is which three events that string names, not the strength of the
    binding. `push` is absent deliberately (55 pushes/day, two windows-2022
    runners each, on a lane whose jobs cancel one another by design).
    """

    try:
        workflow = load_workflow_yaml(text)
    except WorkflowYamlError as exc:
        return [f"PR CI workflow YAML is ambiguous or invalid: {exc}"]
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return ["PR CI workflow must contain one jobs mapping"]

    errors: list[str] = []
    contracts = (
        (
            "windows-msvc-cpu",
            "CPU",
            "cpu",
            "build-pr-windows-cpu",
        ),
        (
            "windows-msvc-vulkan",
            "Vulkan",
            "vulkan",
            "build-pr-windows-vulkan",
        ),
    )
    for job, backend_label, backend, build_dir in contracts:
        actual = jobs.get(job)
        if actual is None:
            errors.append(f"PR CI is missing required {job} job")
            continue
        expected_script = (
            "$env:VERSION = (Get-Content release/release-version.json -Raw | ConvertFrom-Json).version\n"
            "$env:SOURCE_DATE_EPOCH = (git show -s --format=%ct HEAD).Trim()\n"
            "./scripts/build-windows-release.ps1 `\n"
            f"  -Backend {backend} `\n"
            f"  -ArtifactId windows-x86_64-msvc-{backend} `\n"
            f"  -BuildDir $env:GITHUB_WORKSPACE/{build_dir}\n"
        )
        expected = {
            # PR proof plus the two baseline-lane events (#503). `push` excluded.
            "if": (
                "github.event_name == 'pull_request' "
                "|| github.event_name == 'schedule' "
                "|| github.event_name == 'workflow_dispatch'"
            ),
            "permissions": {"contents": "read"},
            "runs-on": "windows-2022",
            "timeout-minutes": 180,
            "steps": [
                {"uses": "actions/checkout@v4"},
                {
                    "name": "Prove PowerShell, static CRT, and unsupported-tier contracts",
                    "run": "./scripts/build-windows-release.ps1 -ContractTest",
                },
                {
                    "name": (
                        "Build and execute the native Windows "
                        f"{backend_label} focused gate"
                    ),
                    "env": {
                        "EVIDENCE_URL": (
                            "${{ github.server_url }}/${{ github.repository }}"
                            "/actions/runs/${{ github.run_id }}"
                        ),
                        "SOURCE_SHA": "${{ github.sha }}",
                    },
                    "run": expected_script,
                },
            ],
        }
        if actual != expected:
            errors.append(
                f"{job} must exactly match the read-only native Windows PR proof schema"
            )
    return errors


def named_step(block: str, name: str) -> str | None:
    matches = [
        step
        for step in workflow_steps(block)
        if re.search(rf"(?m)^      - name: {re.escape(name)}$", step)
    ]
    return matches[0] if len(matches) == 1 else None


def download_is_bound(block: str, artifact_ids: str, path: str) -> bool:
    matches = [
        step
        for step in action_steps(block, "actions/download-artifact@v4")
        if mapping_value(step, "with", "artifact-ids") == artifact_ids
    ]
    return len(matches) == 1 and mapping_value(matches[0], "with", "path") == path


def upload_is_bound(block: str, artifact_name: str, paths: list[str]) -> bool:
    matches = [
        step
        for step in action_steps(block, "actions/upload-artifact@v4")
        if mapping_value(step, "with", "name") == artifact_name
    ]
    return len(matches) == 1 and mapping_paths(matches[0]) == paths


def step_run_script(step: str) -> str | None:
    lines = step.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "        run: |"]
    if len(starts) != 1:
        return None
    script: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if re.match(r"^        \S", line):
            break
        if line and not line.startswith("          "):
            return None
        script.append(line[10:] if line else "")
    return "\n".join(script)


def logical_shell_commands(step: str) -> list[str] | None:
    script = step_run_script(step)
    if script is None:
        return None
    logical: list[str] = []
    current: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if current:
                logical.append(" ".join(current))
                current = []
            continue
        continued = stripped.endswith("\\")
        current.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            logical.append(" ".join(current))
            current = []
    if current:
        return None
    return logical


def shell_commands(step: str) -> list[list[str]] | None:
    logical = logical_shell_commands(step)
    if logical is None:
        return None

    commands: list[list[str]] = []
    try:
        for command in logical:
            lexer = shlex.shlex(
                command, posix=True, punctuation_chars=";&|<>()"
            )
            lexer.whitespace_split = True
            lexer.commenters = "#"
            commands.append(list(lexer))
    except ValueError:
        return None
    return commands


def executable_shell_segments(command: str) -> tuple[list[list[str]], bool] | None:
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote == "'":
            current.append(char)
            if char == "'":
                quote = None
            index += 1
            continue
        if char == '"':
            current.append(char)
            quote = None if quote == '"' else '"'
            index += 1
            continue
        if char == "\\":
            current.append(char)
            index += 1
            if index < len(command):
                current.append(command[index])
                index += 1
            continue
        if command.startswith(("$(", "<(", ">("), index) or char == "`":
            return [], True
        if quote == '"':
            current.append(char)
            index += 1
            continue
        if char == "'":
            current.append(char)
            quote = "'"
            index += 1
            continue
        if char == "#" and (
            not current or current[-1].isspace() or current[-1] in ";|&(){}"
        ):
            break
        if char in ";|&(){}":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            while index < len(command) and command[index] in ";|&":
                index += 1
            continue
        current.append(char)
        index += 1
    if quote is not None:
        return None
    segment = "".join(current).strip()
    if segment:
        segments.append(segment)

    words: list[list[str]] = []
    try:
        for segment in segments:
            words.append(shlex.split(segment, comments=True, posix=True))
    except ValueError:
        return None
    return words, False


def shell_assignment(word: str) -> bool:
    name, separator, _ = word.partition("=")
    return (
        bool(separator)
        and bool(name)
        and (name[0].isalpha() or name[0] == "_")
        and all(char.isalnum() or char == "_" for char in name[1:])
    )


def executable_prefix(words: list[str]) -> list[str]:
    prefixes = {
        "!",
        "builtin",
        "command",
        "do",
        "elif",
        "else",
        "env",
        "exec",
        "if",
        "nohup",
        "sudo",
        "then",
        "time",
        "until",
        "while",
    }
    index = 0
    while index < len(words):
        word = words[index]
        if shell_assignment(word) or word in prefixes:
            index += 1
            continue
        if word.startswith("-") and index > 0 and words[index - 1] in prefixes:
            index += 1
            continue
        break
    return words[index:]


def step_command_is_bound(
    block: str,
    name: str,
    prefix: tuple[str, ...],
    expected_options: dict[str, str],
) -> bool:
    step = named_step(block, name)
    commands = shell_commands(step) if step is not None else None
    if commands is None:
        return False
    matches = [command for command in commands if command[: len(prefix)] == list(prefix)]
    if len(matches) != 1:
        return False
    arguments = matches[0][len(prefix) :]
    if len(arguments) % 2 != 0:
        return False
    options: dict[str, str] = {}
    for index in range(0, len(arguments), 2):
        option, value = arguments[index : index + 2]
        if not option.startswith("--") or option in options:
            return False
        options[option] = value
    return options == expected_options


def step_has_exact_command(
    block: str, name: str, expected: tuple[str, ...]
) -> bool:
    step = named_step(block, name)
    commands = shell_commands(step) if step is not None else None
    return commands is not None and commands.count(list(expected)) == 1


def action_input_is_bound(
    block: str, name: str, action: str, key: str, value: str
) -> bool:
    step = named_step(block, name)
    return (
        step is not None
        and re.search(rf"(?m)^        uses: {re.escape(action)}$", step)
        is not None
        and mapping_value(step, "with", key) == value
    )


def job_has_shell_command(block: str, prefix: tuple[str, ...]) -> bool:
    for step in workflow_steps(block):
        if "        run: |" not in step:
            continue
        logical = logical_shell_commands(step)
        if logical is None:
            return True
        for command in logical:
            scanned = executable_shell_segments(command)
            if scanned is None or scanned[1]:
                return True
            if any(
                executable_prefix(segment)[: len(prefix)] == list(prefix)
                for segment in scanned[0]
            ):
                return True
    return False


def validate(text: str) -> list[str]:
    errors: list[str] = []
    required_global = (
        "  workflow_dispatch: {}",
        "  push:\n    tags: ['v*']",
        "permissions:\n  contents: read",
    )
    for fragment in required_global:
        if fragment not in text:
            errors.append(f"workflow is missing required global contract: {fragment!r}")
    if re.search(r"(?m)^\s*pull_request:", text):
        errors.append("release workflow must not run for pull requests")
    if "continue-on-error" in text:
        errors.append("release workflow may not continue after an error")

    primary_build_jobs = (
        "cpu_x86",
        "cpu_arm64",
        "cpu_musl",
        "cuda_x86",
        "cuda_arm64",
        "vulkan_x86",
        "metal_arm64",
        "mlx_arm64",
        "cpu_windows",
        "vulkan_windows",
    )
    release_outputs = (
        ("build-release-cpu-x86", "linux-x86_64-glibc-cpu", "tar.gz"),
        ("build-release-cpu-arm64", "linux-aarch64-glibc-cpu", "tar.gz"),
        ("build-release-cpu-musl", "linux-x86_64-musl-cpu-static", "tar.gz"),
        ("build-release-cuda-x86", "linux-x86_64-glibc-cuda", "tar.gz"),
        ("build-release-cuda-arm64", "linux-aarch64-glibc-cuda", "tar.gz"),
        ("build-release-vulkan-x86", "linux-x86_64-glibc-vulkan", "tar.gz"),
        ("build-release-metal-arm64", "macos-arm64-metal", "tar.gz"),
        ("build-release-mlx-arm64", "macos-arm64-metal-mlx", "tar.gz"),
        ("build-release-windows-cpu", "windows-x86_64-msvc-cpu", "zip"),
        ("build-release-windows-vulkan", "windows-x86_64-msvc-vulkan", "zip"),
    )
    read_only_jobs = (
        "plan",
        *primary_build_jobs,
        "build",
        "verify",
    )
    blocks = {
        name: job_block(text, name)
        for name in (*read_only_jobs, "attest", "publish")
    }
    for name, block in blocks.items():
        if not block:
            errors.append(f"release workflow is missing {name} job")
    for name in read_only_jobs:
        block = blocks[name]
        if "    permissions:\n      contents: read" not in block:
            errors.append(f"{name} job must have contents: read only")
        permission_block = re.search(r"(?m)^    permissions:\n((?:      [^\n]*\n)+)", block)
        if permission_block and "write" in permission_block.group(1):
            errors.append(f"{name} job unexpectedly has write permission")

    build = blocks["build"]
    expected_needs = "    needs: [plan, " + ", ".join(primary_build_jobs) + "]"
    if expected_needs not in build:
        errors.append("handoff build job must depend on every primary release tuple")
    for name in primary_build_jobs:
        reference = f"${{{{ needs.{name}.outputs.artifact_id }}}}"
        if reference not in build:
            errors.append(f"handoff build job does not consume immutable {name} output")
    for build_dir, artifact_id, archive_format in release_outputs:
        archive = (
            f"{build_dir}/release/vllm.cpp-${{{{ needs.plan.outputs.version }}}}-"
            f"{artifact_id}.{archive_format}"
        )
        if any(
            f"            {archive}{suffix}\n" not in text
            for suffix in ("", ".sha256", ".provenance.json")
        ):
            errors.append(
                "every release upload path must use its canonical versioned archive name"
            )
    if (
        text.count(".sha256\n") != len(release_outputs)
        or text.count(".provenance.json\n") != len(release_outputs)
    ):
        errors.append("release workflow must upload exactly ten archive triplets")

    windows_contracts = (
        ("cpu_windows", "cpu", "build-release-windows-cpu"),
        ("vulkan_windows", "vulkan", "build-release-windows-vulkan"),
    )
    for job, backend, build_dir in windows_contracts:
        block = blocks[job]
        required = (
            "    needs: plan\n",
            "    permissions:\n      contents: read\n",
            "    runs-on: windows-2022\n",
            "      - uses: actions/checkout@v4\n",
            "        run: ./scripts/build-windows-release.ps1 -ContractTest\n",
            "          SOURCE_SHA: ${{ github.sha }}\n",
            "          VERSION: ${{ needs.plan.outputs.version }}\n",
            "          $env:SOURCE_DATE_EPOCH = (git show -s --format=%ct HEAD).Trim()\n",
            "          $release = Get-Content release/release-version.json -Raw | ConvertFrom-Json\n",
            "          if ($release.version -ne $env:VERSION) { throw \"release version drift\" }\n",
            "          ./scripts/build-windows-release.ps1 `\n"
            f"            -Backend {backend} `\n"
            f"            -ArtifactId windows-x86_64-msvc-{backend} `\n"
            f"            -BuildDir $env:GITHUB_WORKSPACE/{build_dir}\n",
        )
        if any(fragment not in block for fragment in required):
            errors.append(f"{job} must bind the exact native Windows build contract")
        if block.count("./scripts/build-windows-release.ps1") != 2:
            errors.append(f"{job} must run exactly one contract test and one native build")
        if block.count("          SOURCE_SHA: ${{ github.sha }}\n") != 2:
            errors.append(f"{job} must bind both Windows steps to the workflow source SHA")

    attest = blocks["attest"]
    for permission in (
        "      contents: read",
        "      id-token: write",
        "      attestations: write",
        "      artifact-metadata: write",
    ):
        if permission not in attest:
            errors.append(f"attest job is missing {permission.strip()}")
    if "uses: actions/attest@v4" not in attest:
        errors.append("attest job must use actions/attest@v4")
    tag_gate = "startsWith(github.ref, 'refs/tags/v')"
    if tag_gate not in attest or "needs.plan.outputs.publish == 'true'" not in attest:
        errors.append("attest job must require an exact tag and approved publish plan")

    publish = blocks["publish"]
    if "    permissions:\n      contents: write" not in publish:
        errors.append("publish job alone must receive contents: write")
    if any(permission in publish for permission in ("id-token: write", "attestations: write")):
        errors.append("publish job must not receive attestation authority")
    if "    environment: release" not in publish:
        errors.append("publish job must use the protected release environment")
    if "    needs: [plan, verify, attest]" not in publish:
        errors.append("publish job must consume only plan plus verified and attested handoffs")
    if "uses: actions/checkout@v4" not in publish:
        errors.append("publish job must check out the exact tagged publisher implementation")
    if tag_gate not in publish or "needs.plan.outputs.publish == 'true'" not in publish:
        errors.append("publish job must require an exact tag and approved publish plan")
    if job_has_shell_command(
        publish, ("gh", "release", "create")
    ) or job_has_shell_command(publish, ("gh", "release", "upload")):
        errors.append("release workflow must not bypass the byte-bound publisher")

    required_handoff = (
        "name: release-plan-${{ github.sha }}",
        "name: release-unverified-${{ github.sha }}",
        "name: release-verified-${{ github.sha }}",
        "artifact-ids: ${{ needs.plan.outputs.artifact_id }}",
        "artifact-ids: ${{ needs.build.outputs.artifact_id }}",
        "artifact-ids: ${{ needs.verify.outputs.artifact_id }}",
        "overwrite: false",
        "if-no-files-found: error",
    )
    for fragment in required_handoff:
        if fragment not in text:
            errors.append(f"immutable artifact handoff is missing {fragment!r}")
    plan_artifact = "${{ needs.plan.outputs.artifact_id }}"
    primary_artifacts = ",".join(
        f"${{{{ needs.{name}.outputs.artifact_id }}}}"
        for name in primary_build_jobs
    )
    verified_artifact = "${{ needs.verify.outputs.artifact_id }}"
    verify = blocks["verify"]
    root_contracts = (
        step_command_is_bound(
            blocks["plan"],
            "Compute an explicit dry-run or exact-tag plan",
            ("python3", "scripts/release_pipeline.py", "plan"),
            {
                "--event": "${{ github.event_name }}",
                "--ref": "${{ github.ref }}",
                "--sha": "${{ github.sha }}",
                "--release-version": "release/release-version.json",
                "--matrix": "release/release-matrix.json",
                "--output": "release-plan.json",
                "--github-output": "$GITHUB_OUTPUT",
            },
        ),
        download_is_bound(build, plan_artifact, "plan"),
        download_is_bound(build, primary_artifacts, "release-assets"),
        step_command_is_bound(
            build,
            "Produce the byte-bound release handoff",
            ("python3", "scripts/release_pipeline.py", "handoff"),
            {
                "--plan": "plan/release-plan.json",
                "--assets-dir": "release-assets",
                "--output": "release-handoff.json",
            },
        ),
        upload_is_bound(
            build,
            "release-unverified-${{ github.sha }}",
            ["release-handoff.json", "release-assets"],
        ),
        download_is_bound(verify, plan_artifact, "plan"),
        download_is_bound(
            verify, "${{ needs.build.outputs.artifact_id }}", "unverified"
        ),
        step_has_exact_command(
            verify,
            "Verify handoff against plan and workflow SHA",
            ("cp", "-a", "unverified/release-assets", "verified/release-assets"),
        ),
        step_command_is_bound(
            verify,
            "Verify handoff against plan and workflow SHA",
            ("python3", "scripts/release_pipeline.py", "verify"),
            {
                "--plan": "plan/release-plan.json",
                "--handoff": "unverified/release-handoff.json",
                "--assets-dir": "verified/release-assets",
                "--output": "verified/verified-handoff.json",
                "--sha": "${{ github.sha }}",
            },
        ),
        step_command_is_bound(
            verify,
            "Generate byte-derived release indexes",
            ("python3", "scripts/release_index.py"),
            {
                "--assets-dir": "verified/release-assets",
                "--handoff": "verified/verified-handoff.json",
                "--json-output": "verified/release-index.json",
                "--markdown-output": "verified/RELEASE_INDEX.md",
                "--retention-days": "7",
            },
        ),
        upload_is_bound(
            verify, "release-verified-${{ github.sha }}", ["verified"]
        ),
        download_is_bound(attest, verified_artifact, "verified"),
        action_input_is_bound(
            attest,
            "Attest verified bytes",
            "actions/attest@v4",
            "subject-path",
            "verified/release-assets/**",
        ),
        download_is_bound(publish, verified_artifact, "verified"),
        step_command_is_bound(
            publish,
            "Publish the exact verified release assets",
            ("python3", "scripts/release_pipeline.py", "publish"),
            {
                "--handoff": "verified/verified-handoff.json",
                "--assets-dir": "verified/release-assets",
                "--index-json": "verified/release-index.json",
                "--index-markdown": "verified/RELEASE_INDEX.md",
                "--tag": "$tag",
            },
        ),
    )
    if not all(root_contracts):
        errors.append(
            "release workflow must bind each handoff stage to its declared root"
        )
    uploads = text.count("uses: actions/upload-artifact@v4")
    download_steps = action_steps(text, "actions/download-artifact@v4")
    downloads = len(download_steps)
    if uploads < 3:
        errors.append("release workflow requires immutable plan, asset, and verified uploads")
    if text.count("overwrite: false") != uploads:
        errors.append("every artifact upload must refuse overwrite")
    if text.count("if-no-files-found: error") != uploads:
        errors.append("every artifact upload must fail when its explicit file is missing")
    if downloads < 5 or text.count("artifact-ids:") != downloads:
        errors.append("every cross-job handoff must use an exact immutable artifact ID")
    flatten_values = []
    for step in download_steps:
        with_blocks = step_mapping_blocks(step, "with")
        flatten_values.append(
            re.findall(
                r"(?m)^          merge-multiple:\s*(true|'true'|\"true\")\s*$",
                with_blocks[0],
            )
            if len(with_blocks) == 1
            else []
        )
    if any(values not in (["true"], ["'true'"], ['"true"']) for values in flatten_values):
        errors.append("every artifact download must flatten into its declared path")
    if re.search(r"(?m)^\s+path:\s*[^\n]*[*?]", text):
        errors.append("release workflow artifact paths must not use wildcards")
    return errors


def main() -> int:
    errors = validate(WORKFLOW.read_text(encoding="utf-8"))
    errors.extend(validate_pr_ci(CI_WORKFLOW.read_text(encoding="utf-8")))
    if errors:
        print("release workflow policy FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("release workflow policy: least privilege and immutable handoff OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
