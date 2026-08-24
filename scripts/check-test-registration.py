#!/usr/bin/env python3
"""Require critical regression tests to remain buildable and CTest-registered.

The CPU CI lane builds every target and runs ``ctest``, but both promises become
vacuous when a regression test is accidentally removed from ``tests/CMakeLists.txt``.
This tree gate pins the small set of tests whose review explicitly requires a
non-vacuous registration guard.  It also verifies that the shared helper still
creates an executable *and* registers that executable with CTest.

It pins CTest LABELS for the same reason, and the reason is measured rather than
assumed.  A gate whose preconditions no runner has is invoked by label -- the
documented recipe is ``ctest -L gpu`` -- and ``ctest -L`` prints
``No tests were found!!!`` and returns **0** when the label selects nothing
(CMake 3.28.3).  So a renamed or deleted label turns a documented gate into a
command that measures nothing while reporting success, which is the exact defect
class the labelled gate itself exists to close.  ``REQUIRED_LABEL_SELECTIONS``
holds the expected selection as a literal in this file, never read back out of
``tests/CMakeLists.txt``, because a checker that reads its expectation from the
file it checks is a tautology.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_CMAKE = ROOT / "tests/CMakeLists.txt"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"
MUTATION_SUITE = ROOT / "tests/scripts/test_check_test_registration.py"
MUTATION_MANIFEST = ROOT / "tests/scripts/check_test_registration_mutations.txt"
MUTATION_MANIFEST_SHA256 = (
    "9c35a1373af09bab9bb9fb65e1ee7bb15e7e831c379b00487eda5235a2bbcf9a"
)

REQUIRED_TESTS = {
    "test_device_selection": "vllm/entrypoints/test_device_selection.cpp",
}

# label -> the EXACT set of CTest test names it may select.  Exact, not a floor:
# a floor cannot see a second test that quietly joins a lane whose whole purpose
# is that a human runs it deliberately inside a lease.  Adding a labelled gate is
# a one-line addition here and is meant to be a deliberate record.
REQUIRED_LABEL_SELECTIONS = {
    "gpu": (
        "test_minimax_music3_depth_arm_real",
        "test_minimax_music3_device_arm_real",
    ),
}

def _without_line_comments(text: str) -> str:
    """Remove ``#`` comments while preserving quoted ``#`` characters."""

    cleaned: list[str] = []
    for line in text.splitlines():
        quoted = False
        escaped = False
        kept: list[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quoted:
                kept.append(char)
                escaped = True
                continue
            if char == '"':
                quoted = not quoted
                kept.append(char)
                continue
            if char == "#" and not quoted:
                break
            kept.append(char)
        cleaned.append("".join(kept))
    return "\n".join(cleaned)


def _configure(
    source_dir: Path, build_dir: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Configure CMake after requesting its semantic target codemodel."""

    query = build_dir / ".cmake/api/v1/query/codemodel-v2"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()
    command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _codemodel_targets(
    build_dir: Path,
) -> tuple[str | None, dict[str, dict[str, object]]]:
    """Return one configuration and its targets from the CMake File API.

    Multi-config generators describe several artifact paths.  Release is the
    CI contract and must be selected consistently for both the codemodel and
    the subsequent CTest query.
    """

    reply = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    if not indexes:
        return None, {}
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    codemodel_file = index["reply"]["codemodel-v2"]["jsonFile"]
    codemodel = json.loads((reply / codemodel_file).read_text(encoding="utf-8"))
    configurations = codemodel.get("configurations", [])
    if not configurations:
        return None, {}
    configuration = next(
        (entry for entry in configurations if entry.get("name") == "Release"),
        configurations[0],
    )
    targets: dict[str, dict[str, object]] = {}
    for summary in configuration.get("targets", []):
        detail = json.loads((reply / summary["jsonFile"]).read_text(encoding="utf-8"))
        targets[summary["name"]] = detail
    name = configuration.get("name")
    return (name if isinstance(name, str) and name else None), targets


def _ctest_tests(
    build_dir: Path, configuration: str | None
) -> dict[str, dict[str, object]]:
    """Return configured CTest command/properties keyed by test name.

    CTest omits ``command`` from JSON when a target executable does not exist
    yet.  The caller materializes disposable placeholders at the File-API
    artifact paths before asking for this document, so the command is the
    resolved executable path rather than an uninterpreted CMake token.
    """

    command = ["ctest", "--test-dir", str(build_dir)]
    if configuration is not None:
        command.extend(["-C", configuration])
    command.append("--show-only=json-v1")
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return {
        test["name"]: {
            "command": test.get("command", []),
            "properties": test.get("properties", []),
        }
        for test in document.get("tests", [])
        if isinstance(test.get("name"), str)
    }


def _cmake_truthy(value: object) -> bool:
    """Interpret CMake's documented false constants; unknown values fail closed."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().upper()
    return normalized not in {
        "",
        "0",
        "FALSE",
        "OFF",
        "NO",
        "N",
        "IGNORE",
        "NOTFOUND",
    } and not normalized.endswith("-NOTFOUND")


def _target_artifact(build_dir: Path, detail: dict[str, object]) -> Path | None:
    """Resolve the single configured executable artifact for a File-API target."""

    artifacts = detail.get("artifacts", [])
    if detail.get("type") != "EXECUTABLE" or not isinstance(artifacts, list):
        return None
    paths = [entry.get("path") for entry in artifacts if isinstance(entry, dict)]
    if len(paths) != 1 or not isinstance(paths[0], str):
        return None
    return (build_dir / paths[0]).resolve()


def _materialize_ctest_targets(
    build_dir: Path, targets: dict[str, dict[str, object]], required: dict[str, str]
) -> dict[str, Path]:
    """Make unbuilt configured target paths resolvable to CTest JSON.

    Only the disposable configure directory is touched.  Existing artifacts are
    never replaced; absent artifacts become empty executable placeholders long
    enough for ``ctest --show-only=json-v1`` to resolve target-name commands.
    """

    artifacts: dict[str, Path] = {}
    for target in required:
        detail = targets.get(target)
        if detail is None:
            continue
        artifact = _target_artifact(build_dir, detail)
        if artifact is None:
            continue
        artifacts[target] = artifact
        if artifact.exists():
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
        artifact.chmod(0o700)
    return artifacts


def _configured_contract_errors(
    source_dir: Path,
    build_dir: Path,
    required: dict[str, str],
    extra_args: list[str] | None = None,
) -> list[str]:
    """Ask CMake/CTest what exists instead of interpreting CMake source text."""

    configured = _configure(source_dir, build_dir, extra_args)
    if configured.returncode != 0:
        transcript = configured.stdout + configured.stderr
        errors = ["CMake configure failed while proving required test registration"]
        for target in sorted(required):
            errors.append(f"missing required test target {target} in configured codemodel")
        if "already exists" in transcript:
            for target in sorted(required):
                errors.append(f"required test target {target} is registered 2 times")
        errors.append(
            "vllm_cpp_add_test does not create an executable with its configured sources"
        )
        return errors

    configuration, targets = _codemodel_targets(build_dir)
    artifacts = _materialize_ctest_targets(build_dir, targets, required)
    tests = _ctest_tests(build_dir, configuration)
    errors: list[str] = []
    for target, expected_source in sorted(required.items()):
        detail = targets.get(target)
        if detail is None:
            errors.append(f"missing required test target {target} in configured codemodel")
            errors.append(
                "vllm_cpp_add_test does not create an executable with its configured sources"
            )
            continue
        actual_sources = {
            Path(source["path"]).as_posix() for source in detail.get("sources", [])
        }
        if expected_source not in actual_sources:
            actual = ", ".join(sorted(actual_sources)) or "<no sources>"
            errors.append(
                f"required test target {target} must compile {expected_source}; got {actual}"
            )
        artifact = artifacts.get(target)
        if artifact is None:
            errors.append(
                f"required test target {target} has no single configured executable artifact"
            )
        if target not in tests:
            errors.append(
                f"required test target {target} is not registered with CTest; "
                "vllm_cpp_add_test does not register that executable with CTest"
            )
        elif artifact is not None:
            test = tests[target]
            command = test.get("command", [])
            if not isinstance(command, list):
                command = []
            actual_command: Path | None = None
            if len(command) == 1:
                candidate = Path(command[0])
                actual_command = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (build_dir / candidate).resolve()
                )
            if actual_command != artifact:
                rendered = shlex.join(command) if command else "<unresolved command>"
                errors.append(
                    f"CTest test {target} must execute configured target {target} exactly; "
                    f"got {rendered}"
                )
            properties = test.get("properties", [])
            if not isinstance(properties, list):
                properties = []
            disabled = [
                prop.get("value")
                for prop in properties
                if isinstance(prop, dict) and prop.get("name") == "DISABLED"
            ]
            if any(_cmake_truthy(value) for value in disabled):
                errors.append(f"CTest test {target} must not be DISABLED")
    return errors


def _test_labels(test: dict[str, object]) -> set[str]:
    """Return the CTest LABELS of one ``--show-only=json-v1`` test entry."""

    properties = test.get("properties", [])
    if not isinstance(properties, list):
        return set()
    labels: set[str] = set()
    for prop in properties:
        if not isinstance(prop, dict) or prop.get("name") != "LABELS":
            continue
        value = prop.get("value")
        if isinstance(value, str):
            labels.add(value)
        elif isinstance(value, list):
            labels.update(entry for entry in value if isinstance(entry, str))
    return labels


def _label_selection_errors(
    tests: dict[str, dict[str, object]], selections: dict[str, tuple[str, ...]]
) -> list[str]:
    """Compare the configured LABELS against this file's literal pin.

    The diagnostic names BOTH sides of the comparison in words.  ``ctest -L``
    reports success over an empty selection, so a reader who is handed only a
    return code cannot tell a passing lane from an absent one.
    """

    errors: list[str] = []
    for label, expected in sorted(selections.items()):
        selected = sorted(
            name for name, test in tests.items() if label in _test_labels(test)
        )
        wanted = sorted(expected)
        if selected == wanted:
            continue
        errors.append(
            f"ctest -L {label} selects {len(selected)} test(s) "
            f"[{', '.join(selected) if selected else '<none>'}]; "
            f"REQUIRED_LABEL_SELECTIONS in scripts/check-test-registration.py pins "
            f"{len(wanted)} [{', '.join(wanted) if wanted else '<none>'}]. "
            "Compared the LABELS property of the CONFIGURED tree against the literal "
            "pin in this checker, not against tests/CMakeLists.txt text. "
            "An empty selection is the dangerous case: ctest -L returns 0 over it"
        )
    return errors


def _configured_label_errors(
    source_dir: Path,
    build_dir: Path,
    selections: dict[str, tuple[str, ...]],
    extra_args: list[str] | None = None,
) -> list[str]:
    """Ask CMake/CTest which tests a label selects, then compare with the pin."""

    configured = _configure(source_dir, build_dir, extra_args)
    if configured.returncode != 0:
        return [
            "CMake configure failed while proving CTest label selection; "
            "no label could be read, which fails closed"
        ]
    configuration, _ = _codemodel_targets(build_dir)
    return _label_selection_errors(_ctest_tests(build_dir, configuration), selections)


def label_errors(
    cmake_text: str, selections: dict[str, tuple[str, ...]] | None = None
) -> list[str]:
    """Return violations of the CTest label-selection contract."""

    if selections is None:
        selections = REQUIRED_LABEL_SELECTIONS
    with tempfile.TemporaryDirectory(prefix="vllm-label-unit-") as temporary:
        root = Path(temporary)
        (root / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        for source in {
            *REQUIRED_TESTS.values(),
            "vllm/entrypoints/other.cpp",
            "parity/test_minimax_music3_device_arm_real.cpp",
            "parity/test_minimax_music3_depth_arm_real.cpp",
        }:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int label_guard_dummy;\n", encoding="utf-8")
        return _configured_label_errors(root, root / "build", selections)


def registration_errors(
    cmake_text: str, required: dict[str, str] | None = None
) -> list[str]:
    """Return violations of the executable + CTest registration contract."""

    if required is None:
        required = REQUIRED_TESTS
    with tempfile.TemporaryDirectory(prefix="vllm-registration-unit-") as temporary:
        root = Path(temporary)
        (root / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        for source in {
            *required.values(),
            "vllm/entrypoints/other.cpp",
        }:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int registration_guard_dummy;\n", encoding="utf-8")
        return _configured_contract_errors(root, root / "build", required)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _literal_block(lines: list[str], header_index: int) -> list[str]:
    parent_indent = _indent(lines[header_index])
    raw: list[str] = []
    for candidate in lines[header_index + 1 :]:
        if not candidate.strip():
            raw.append("")
            continue
        if _indent(candidate) <= parent_indent:
            break
        raw.append(candidate)
    nonblank = [line for line in raw if line.strip()]
    if not nonblank:
        return []
    content_indent = min(_indent(line) for line in nonblank)
    return [line[content_indent:] if line.strip() else "" for line in raw]


def _yaml_mapping(line: str, indent: int, sequence: bool = False) -> tuple[str, str] | None:
    """Parse one direct YAML mapping key in the workflow's structural subset."""

    if _indent(line) != indent:
        return None
    content = line[indent:]
    if sequence:
        if not content.startswith("-"):
            return None
        content = content[1:].lstrip()
        if not content:
            return None
    match = re.match(r"^(?P<key>'[^']*'|\"[^\"]*\"|[^:#]+?)\s*:\s*(?P<value>.*)$", content)
    if match is None:
        return None
    key = match.group("key").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1]
    return key.strip(), match.group("value").strip()


def _unconditional_ci_run_blocks(text: str) -> list[list[str]]:
    """Return literal run blocks owned by unconditional Actions jobs/steps.

    This is deliberately a narrow GitHub-Actions structural parser, not a YAML
    implementation: it recognizes the canonical ``jobs -> job -> steps -> -``
    hierarchy and direct ``if``/``run`` fields.  A run block in prose, a sibling
    mapping, or a conditional job/step never enters the result.
    """

    lines = text.splitlines()
    blocks: list[list[str]] = []
    jobs_index = next(
        (i for i, line in enumerate(lines) if line == "jobs:"), None
    )
    if jobs_index is None:
        return blocks

    job_starts = [
        i
        for i in range(jobs_index + 1, len(lines))
        if (mapping := _yaml_mapping(lines[i], 2)) is not None
        and mapping[1] == ""
    ]
    for job_pos, job_start in enumerate(job_starts):
        job_end = job_starts[job_pos + 1] if job_pos + 1 < len(job_starts) else len(lines)
        job_lines = lines[job_start + 1 : job_end]
        job_fields = {
            mapping[0]
            for line in job_lines
            if (mapping := _yaml_mapping(line, 4)) is not None
        }
        if "if" in job_fields:
            continue
        steps_offset = next(
            (
                i
                for i, line in enumerate(job_lines)
                if _yaml_mapping(line, 4) == ("steps", "")
            ),
            None,
        )
        if steps_offset is None:
            continue
        steps_start = job_start + 1 + steps_offset + 1
        step_starts = [
            i
            for i in range(steps_start, job_end)
            if _indent(lines[i]) == 6 and lines[i][6:].startswith("-")
        ]
        for step_pos, step_start in enumerate(step_starts):
            step_end = (
                step_starts[step_pos + 1]
                if step_pos + 1 < len(step_starts)
                else job_end
            )
            step_fields: dict[str, tuple[str, int]] = {}
            first = _yaml_mapping(lines[step_start], 6, sequence=True)
            if first is not None:
                step_fields[first[0]] = (first[1], step_start)
            for index in range(step_start + 1, step_end):
                mapping = _yaml_mapping(lines[index], 8)
                if mapping is not None:
                    step_fields[mapping[0]] = (mapping[1], index)
            if {"if", "continue-on-error", "shell"} & step_fields.keys():
                continue
            run = step_fields.get("run")
            if run is not None and re.fullmatch(r"\|[-+]?", run[0]):
                run_index = run[1]
                blocks.append(_literal_block(lines, run_index))
    return blocks


def _direct_commands(block: list[str]) -> list[list[str]] | None:
    """Parse a literal block that contains only direct, unconditional commands."""

    commands: list[list[str]] = []
    for line in block:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line != line.lstrip():
            return None
        try:
            argv = shlex.split(stripped, comments=True, posix=True)
        except ValueError:
            return None
        if not argv or any(token in {";", "&&", "||", "|", "&"} for token in argv):
            return None
        commands.append(argv)
    return commands


def _ci_has_active_guard_step(ci_text: str) -> bool:
    expected = [
        ["python3", "scripts/check-test-registration.py"],
        ["python3", "tests/scripts/test_check_test_registration.py"],
    ]
    return any(
        _direct_commands(block) == expected
        for block in _unconditional_ci_run_blocks(ci_text)
    )


def _active_ci_commands(ci_text: str) -> set[tuple[str, ...]]:
    commands: set[tuple[str, ...]] = set()
    for block in _unconditional_ci_run_blocks(ci_text):
        parsed = _direct_commands(block)
        if parsed is not None:
            commands.update(tuple(command) for command in parsed)
    return commands


def _bash_array_values(text: str, name: str) -> list[str] | None:
    """Read one top-level Bash array assignment by its actual variable name."""

    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == f"{name}=("]
    if len(starts) != 1:
        return None
    values: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.strip() == ")":
            return values
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            return None
        values.extend(tokens)
    return None


def _trace_preflight_commands(text: str) -> tuple[int, list[tuple[str, ...]]]:
    """Execute preflight with Python/Git shims and return Python argv traces."""

    with tempfile.TemporaryDirectory(prefix="vllm-preflight-trace-") as temporary:
        root = Path(temporary)
        script = root / "scripts/agent-preflight.sh"
        script.parent.mkdir(parents=True)
        script.write_text(text, encoding="utf-8")
        script.chmod(0o700)
        (root / ".agents").mkdir()
        (root / ".agents/NOW.md").write_text("trace-only\n", encoding="utf-8")
        shim_dir = root / "shim"
        shim_dir.mkdir()
        trace = root / "python.trace"
        python = shim_dir / "python3"
        python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\0' \"$@\" >> \"$VLLM_REGISTRATION_TRACE\"\n"
            "printf '\\0' >> \"$VLLM_REGISTRATION_TRACE\"\n",
            encoding="utf-8",
        )
        python.chmod(0o700)
        git = shim_dir / "git"
        git.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        git.chmod(0o700)
        environment = os.environ.copy()
        environment["PATH"] = f"{shim_dir}{os.pathsep}{environment.get('PATH', '')}"
        environment["VLLM_REGISTRATION_TRACE"] = str(trace)
        result = subprocess.run(
            ["bash", str(script), "--quiet", "--no-require-role"],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        raw = trace.read_bytes() if trace.exists() else b""
    invocations = []
    for record in raw.split(b"\0\0"):
        if not record:
            continue
        invocations.append(
            tuple(token.decode("utf-8") for token in record.split(b"\0") if token)
        )
    return result.returncode, invocations


def _preflight_execution_errors(text: str) -> list[str]:
    returncode, invocations = _trace_preflight_commands(text)
    errors: list[str] = []
    checker = ("scripts/check-test-registration.py",)
    suite = ("tests/scripts/test_check_test_registration.py",)
    if invocations.count(checker) != 1:
        errors.append("preflight does not execute CHECKERS through its checker loop")
    if invocations.count(suite) != 1:
        errors.append("preflight does not execute SUITES through its suite loop")
    if returncode != 0:
        errors.append(f"instrumented preflight execution failed with rc={returncode}")
    return errors


def wiring_errors(preflight_text: str, ci_text: str) -> list[str]:
    """Return missing preflight/CI wiring for this checker and its mutations."""

    preflight_source = preflight_text
    preflight_text = _without_line_comments(preflight_source)
    errors: list[str] = []
    checkers = _bash_array_values(preflight_text, "CHECKERS")
    suites = _bash_array_values(preflight_text, "SUITES")
    if checkers is None or "check-test-registration" not in checkers:
        errors.append("check-test-registration is missing from preflight CHECKERS")
    if suites is None or "test_check_test_registration" not in suites:
        errors.append("test_check_test_registration is missing from preflight SUITES")
    errors.extend(_preflight_execution_errors(preflight_source))
    active_ci_commands = _active_ci_commands(ci_text)
    if ("python3", "scripts/check-test-registration.py") not in active_ci_commands:
        errors.append("check-test-registration is missing from the explicit CI checker step")
    if (
        "python3",
        "tests/scripts/test_check_test_registration.py",
    ) not in active_ci_commands:
        errors.append("test_check_test_registration is missing from the CI mutation suite")
    if not _ci_has_active_guard_step(ci_text):
        errors.append(
            "CI guard step must contain the checker and mutation suite as direct active commands"
        )
    return errors


def _method_calls(method: ast.AST) -> list[ast.Call]:
    return [call for call in ast.walk(method) if isinstance(call, ast.Call)]


def mutation_suite_integrity_errors(
    source: str,
    manifest_text: str | None = None,
    *,
    manifest_path: Path = MUTATION_MANIFEST,
) -> list[str]:
    """Pin the suite/manifest pair from this independent production layer."""

    errors: list[str] = []
    if manifest_path.resolve() != MUTATION_MANIFEST.resolve():
        errors.append("mutation suite must use the canonical manifest path")
    if manifest_text is None:
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            return errors + [f"mutation manifest is unreadable: {exc}"]
    digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if digest != MUTATION_MANIFEST_SHA256:
        errors.append("mutation manifest differs from the production-pinned digest")
    manifest = {
        line
        for line in manifest_text.splitlines()
        if line and not line.startswith("#")
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return errors + [f"mutation suite is not valid Python: {exc}"]
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    actual = {name for name in methods if name.startswith("test_M")}
    if actual != manifest:
        errors.append("mutation inventory differs from fixed manifest")

    for name in sorted(actual & manifest):
        calls = {
            call.func.attr
            for call in _method_calls(methods[name])
            if isinstance(call.func, ast.Attribute)
        }
        if not {"assert_error", "assert_wiring_error", "assertTrue"} & calls:
            errors.append(f"{name} has no semantic outcome assertion")

    m42 = methods.get("test_M42_byte_identical_alternate_manifest_path_fails")
    m42_path_assertion = False
    if m42 is not None:
        for call in _method_calls(m42):
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and call.func.attr == "assertEqual"
                and len(call.args) == 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "errors"
                and isinstance(call.args[1], ast.List)
                and len(call.args[1].elts) == 1
                and isinstance(call.args[1].elts[0], ast.Constant)
                and call.args[1].elts[0].value
                == "mutation suite must use the canonical manifest path"
            ):
                continue
            m42_path_assertion = True
            break
    if not m42_path_assertion:
        errors.append(
            "test_M42 must assert the exact canonical-manifest path diagnostic"
        )

    for name, production_call in {
        "assert_error": "registration_errors",
        "assert_wiring_error": "wiring_errors",
    }.items():
        method = methods.get(name)
        if method is None:
            errors.append(f"{name} helper is missing")
            continue
        calls = _method_calls(method)
        if not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "assertTrue"
            for call in calls
        ):
            errors.append(f"{name} has no direct semantic assertion")
        if not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "mod"
            and call.func.attr == production_call
            for call in calls
        ):
            errors.append(f"{name} does not call {production_call}")

    wrapper = methods.get("_suite_integrity_errors")
    wrapper_ok = False
    if wrapper is not None:
        for call in _method_calls(wrapper):
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "mod"
                and call.func.attr == "mutation_suite_integrity_errors"
            ):
                continue
            path_keywords = [kw for kw in call.keywords if kw.arg == "manifest_path"]
            wrapper_ok = (
                len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "source"
                and len(path_keywords) == 1
                and isinstance(path_keywords[0].value, ast.Attribute)
                and isinstance(path_keywords[0].value.value, ast.Name)
                and path_keywords[0].value.value.id == "mod"
                and path_keywords[0].value.attr == "MUTATION_MANIFEST"
            )
    if not wrapper_ok:
        errors.append("suite integrity wrapper does not use the canonical manifest")

    integrity = methods.get("test_suite_integrity_contract_is_pinned")
    if integrity is None:
        errors.append("required mutation-suite integrity method is missing")
    else:
        calls = _method_calls(integrity)
        called_wrapper = any(
            isinstance(call.func, ast.Name) and call.func.id == "_suite_integrity_errors"
            for call in calls
        )
        asserted = any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "assertEqual"
            for call in calls
        )
        if not (called_wrapper and asserted):
            errors.append("mutation-suite integrity method has no pinned assertion")
    return errors


def check_tree(root: Path = ROOT) -> list[str]:
    paths = {
        "tests/CMakeLists.txt": root / "tests/CMakeLists.txt",
        "scripts/agent-preflight.sh": root / "scripts/agent-preflight.sh",
        ".github/workflows/ci.yml": root / ".github/workflows/ci.yml",
        "tests/scripts/test_check_test_registration.py": root
        / "tests/scripts/test_check_test_registration.py",
        "tests/scripts/check_test_registration_mutations.txt": root
        / "tests/scripts/check_test_registration_mutations.txt",
    }
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        return [f"required registration-guard input is missing: {path}" for path in missing]

    configure_args = [
        "-DVLLM_CPP_CUDA=OFF",
        "-DVLLM_CPP_HIP=OFF",
        "-DVLLM_CPP_VULKAN=OFF",
        "-DVLLM_CPP_METAL=OFF",
        "-DVLLM_CPP_MLX=OFF",
        "-DVLLM_CPP_TRITON=OFF",
        "-DVLLM_CPP_BUILD_TESTS=ON",
        "-DVLLM_CPP_BUILD_EXAMPLES=OFF",
        "-DVLLM_CPP_SERVER=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    with tempfile.TemporaryDirectory(prefix="vllm-registration-tree-") as temporary:
        build_dir = Path(temporary) / "build"
        registration = _configured_contract_errors(
            root,
            build_dir,
            {
                target: f"tests/{source}"
                for target, source in REQUIRED_TESTS.items()
            },
            configure_args,
        )
        # The labelled gate is registered unconditionally, so the CPU-only
        # configure above already knows about it: no accelerator is needed to
        # read what `ctest -L gpu` would select.
        registration += _configured_label_errors(
            root, build_dir, REQUIRED_LABEL_SELECTIONS, configure_args
        )
    integrity = mutation_suite_integrity_errors(
        paths["tests/scripts/test_check_test_registration.py"].read_text(encoding="utf-8"),
        paths["tests/scripts/check_test_registration_mutations.txt"].read_text(encoding="utf-8"),
        manifest_path=paths["tests/scripts/check_test_registration_mutations.txt"],
    )
    return registration + wiring_errors(
        paths["scripts/agent-preflight.sh"].read_text(encoding="utf-8"),
        paths[".github/workflows/ci.yml"].read_text(encoding="utf-8"),
    ) + integrity


def main() -> int:
    errors = check_tree()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    selection = ", ".join(
        f"-L {label} -> {len(names)} [{', '.join(sorted(names))}]"
        for label, names in sorted(REQUIRED_LABEL_SELECTIONS.items())
    )
    print(
        "OK: required regression tests have executable + CTest registration, "
        f"the configured tree matches the pinned label selection ({selection}), "
        "and the guard is wired into preflight/CI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
