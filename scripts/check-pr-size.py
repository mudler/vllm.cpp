#!/usr/bin/env python3
"""Enforce finite review budgets for explicit repository path classes."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    from scripts.policy_contract import PolicyRule, Waiver, load_policy, load_waivers
except ModuleNotFoundError:
    from policy_contract import PolicyRule, Waiver, load_policy, load_waivers


ROOT = Path(__file__).resolve().parents[1]

PATH_CLASSES = frozenset(
    {
        "product",
        "governance_checker",
        "governance_test",
        "governance_support",
        "policy",
        "procedure",
        "append_only_record",
        "project_record",
        "public_document",
        "design",
        "ci",
        "configuration",
        "asset",
        "evidence",
        "vendored_dependency",
        "generated",
    }
)
PATH_CLASS_BUDGETS = {
    "product": 900,
    "governance_checker": 6000,
    "governance_test": 3000,
    "governance_support": 1800,
    "policy": 1200,
    "procedure": 3000,
    "append_only_record": 5000,
    "project_record": 4000,
    "public_document": 2500,
    "design": 1500,
    "ci": 800,
    "configuration": 800,
    "asset": 3000,
    "evidence": 8000,
    "vendored_dependency": 8000,
    # A REVIEW budget is a budget on what a human reads. Nobody reads a hex blob,
    # and re-deriving one by eye is not review. These files are emitted by a
    # tracked generator from reviewed sources, and a dedicated gate reproduces
    # them BYTE-FOR-BYTE from those sources on every push, so their correctness is
    # established mechanically rather than by reading the diff. The reviewable
    # surface is the GENERATOR and its INPUTS, both of which stay `product` or
    # `governance_checker` and keep their own tighter budgets.
    "generated": 8000,
}

# Machine-generated artifacts, each of which MUST be (a) emitted by a tracked
# generator in this repository, (b) reproduced byte-for-byte by a gate that runs
# in CI, and (c) marked "GENERATED FILE - DO NOT EDIT BY HAND" at its head.
# Adding a path here without all three is how this class would become a hole.
GENERATED_FILES = frozenset(
    {
        # scripts/gen-vulkan-spirv.py, from src/vt/vulkan/shaders/*.comp.
        # Reproduced by `gen-vulkan-spirv.py --check` in the vulkan-spirv-freshness
        # CI job; the GLSL it compiles stays `product` and is what review reads.
        "src/vt/vulkan/vulkan_spirv.cpp",
    }
)

POLICY_FILES = frozenset(
    {
        ".agents/policy.csv",
        ".agents/waivers.csv",
        ".agents/governance-tasks.csv",
        ".agents/policy-cutover",
    }
)
APPEND_ONLY_FILES = frozenset(
    {
        ".agents/state.md",
        ".agents/benchmark-record.md",
        ".agents/parity-ledger.md",
    }
)
PROJECT_RECORD_FILES = frozenset(
    {
        ".agents/NOW.md",
        ".agents/coordination.md",
        ".agents/roadmap_v1.md",
        ".agents/porting-inventory.md",
        ".agents/engine-matrix.md",
        ".agents/feature-matrix.md",
        ".agents/model-matrix.md",
        ".agents/quantization-matrix.md",
        ".agents/kernel-matrix.md",
        ".agents/backend-matrix.md",
        ".agents/sglang-matrix.md",
    }
)
PROCEDURE_FILES = frozenset(
    {
        "AGENTS.md",
        # A tracked SYMLINK to AGENTS.md, for tools that look for CLAUDE.md. It is
        # the same procedure text and shares its budget; without this the checker
        # failed closed on every change that touched it.
        "CLAUDE.md",
        ".agents/workflow.md",
        ".agents/verification.md",
        ".agents/porting.md",
        ".agents/directives.md",
        ".agents/ai-coding-assistants.md",
        ".agents/benchmark-protocol.md",
        ".agents/discipline.md",
        ".agents/gates.md",
        ".agents/prompts/implementer.md",
        ".agents/prompts/operator.md",
        ".agents/prompts/reviewer.md",
        ".agents/backends.md",
        ".agents/developer-preferences.example.md",
        ".agents/environment.md",
        ".agents/mission.md",
        ".agents/parity-lever-protocol.md",
        ".agents/test-porting.md",
        ".agents/upstream-sync.md",
        ".agents/vllm-v1-v2.md",
    }
)
GOVERNANCE_SUPPORT_FILES = frozenset(
    {
        "scripts/policy_contract.py",
        "scripts/agent-role.py",
        "scripts/claim-view.py",
        "scripts/ready-for-helper.py",
        "scripts/agent-preflight.sh",
    }
)
PRODUCT_CHECKER_FILES = frozenset({"scripts/check-release-binary-contract.py"})
PUBLIC_DOCUMENT_FILES = frozenset(
    {
        "README.md",
        "CONTRIBUTING.md",
        # Landed by a9a8581d and never classified, so the checker failed closed on
        # it the same way it did on CLAUDE.md.
        "MANIFESTO.md",
        "docs/STATUS.md",
        "docs/BENCHMARKS.md",
        "docs/FEATURES.md",
        "docs/USAGE.md",
    }
)
CHECKER = re.compile(r"scripts/check-[a-z0-9]+(?:-[a-z0-9]+)*\.(?:py|sh)\Z")
CHECKER_TEST = re.compile(r"tests/scripts/test_[a-z0-9]+(?:_[a-z0-9]+)*\.py\Z")
CI = re.compile(r"\.github/(?:workflows/[A-Za-z0-9_.-]+\.ya?ml|dependabot\.yml|pull_request_template\.md)\Z")
DESIGN = re.compile(r"docs/superpowers/specs/[0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9-]+\.md\Z")
DOC = re.compile(r"docs/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:md|png|svg|json)\Z")
SPEC = re.compile(r"\.agents/specs/[A-Za-z0-9_.-]+\.md\Z")
SPEC_EVIDENCE = re.compile(r"\.agents/specs/[A-Za-z0-9_.-]+\.(?:patch|json|log)\Z")
COMPLETED = re.compile(r"\.agents/completed/[A-Za-z0-9_.-]+\.md\Z")
SYNC_RECORD = re.compile(r"\.agents/sync/[A-Za-z0-9_.-]+\.md\Z")
HOOK = re.compile(r"\.githooks/(?:README\.md|[A-Za-z0-9_.-]+)\Z")
BENCH_EVIDENCE = re.compile(r"(?:benchmarks/(?:demo|media)|docs/bench-evidence)/[A-Za-z0-9_.-]+\.(?:json|png|gif|mp4|log)\Z")
ASSET = re.compile(r"assets/[A-Za-z0-9_.-]+\.(?:png|svg)\Z")
RELEASE_MANIFEST_FIXTURE = re.compile(
    r"tests/scripts/fixtures/release_manifest/v[0-9]+/[a-z0-9-]+\.json\Z"
)

CHECKER_EVIDENCE_OVERRIDES = {
    "scripts/check-agent-record.py": "tests/scripts/test_agent_record.py",
    "scripts/check-policy.py": "tests/scripts/test_policy_contract.py",
    "scripts/check-role-discipline.py": "tests/scripts/test_check_pr_size.py",
    "scripts/check-doc-checkpoint.py": "tests/scripts/test_doc_checkpoint.py",
    "scripts/check-protocol-consistency.py": "tests/scripts/test_check_protocol_consistency.py",
    # Its suite predates the test_check_<name> convention and CI runs it under
    # the older name, so the derived path pointed at a file that does not
    # exist and NO change to this checker could ever satisfy its own evidence
    # rule. Mapped to the file CI actually runs.
    "scripts/check-device-leakage.py": "tests/scripts/test_device_leakage.py",
}

# New entrypoints cannot appear in their own historical policy enforcement
# cells before they exist. This closed creation map binds their affected rules
# until the policy cutover can name the entrypoint directly.
CREATED_CHECKER_RULES = {
    "scripts/check-commit-trailers.py": (
        "POL-COMMIT-TRAILERS",
        "POL-AI-ATTRIBUTION",
        "POL-WAIVER-EXACT",
    ),
    "scripts/check-pr-size.py": (
        "POL-PATH-CLASSIFICATION",
        "POL-PR-SIZE",
    ),
}
DISABLED_CREATION_CHECKER = (
    b"#!/usr/bin/env python3\n"
    b'\"\"\"Deliberately disabled creation-contract mutation.\"\"\"\n'
)
CREATION_MUTATIONS = {
    "scripts/check-commit-trailers.py": (
        b"#!/usr/bin/env python3\n"
        b'"""Deliberately disabled creation-contract mutation."""\n'
        b"def parsed_trailers(message): return ''\n"
        b"def validate_commit_message(message, *, strict): return []\n"
        b"def validate_range(*args, **kwargs): return []\n"
        b"def exact_waiver(*args, **kwargs): return None\n"
        b"def validate_waiver_targets(*args, **kwargs): return None\n"
    ),
    "scripts/check-policy.py": DISABLED_CREATION_CHECKER,
    "scripts/check-pr-size.py": DISABLED_CREATION_CHECKER,
    "scripts/check-prompt-contract.py": DISABLED_CREATION_CHECKER,
}
EVIDENCE_TIMEOUT_SECONDS = 120
TEST_COUNT = re.compile(r"Ran ([0-9]+) tests? in ")


@dataclass(frozen=True)
class ChangedPath:
    path: str
    added: int | None
    removed: int | None

    @property
    def lines(self) -> int | None:
        if self.added is None or self.removed is None:
            return None
        return self.added + self.removed


@dataclass(frozen=True)
class EvidenceResult:
    checker: str
    test_module: str
    rule_ids: tuple[str, ...]
    head_tests: int
    head_passed: bool
    base_tests: int
    base_failed: bool
    detail: str = ""


def _canonical_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        bool(path)
        and not candidate.is_absolute()
        and "\\" not in path
        and "//" not in path
        and candidate.as_posix() == path
        and all(part not in {"", ".", ".."} for part in candidate.parts)
    )


def classify_path(path: str) -> str:
    """Return one closed path class or reject an unknown/noncanonical path."""

    if not _canonical_path(path):
        raise ValueError(f"noncanonical repository path {path!r}")
    # Ahead of every other rule: a generated artifact under src/ would otherwise
    # fall through to `product` and spend a human-review budget on hex.
    if path in GENERATED_FILES:
        return "generated"
    if path in POLICY_FILES:
        return "policy"
    if path in APPEND_ONLY_FILES:
        return "append_only_record"
    if path in PROJECT_RECORD_FILES:
        return "project_record"
    if path == ".agents/upstream-inventory.json":
        return "project_record"
    if path in PROCEDURE_FILES or SPEC.fullmatch(path) or COMPLETED.fullmatch(path):
        return "procedure"
    if SPEC_EVIDENCE.fullmatch(path) or SYNC_RECORD.fullmatch(path) or BENCH_EVIDENCE.fullmatch(path):
        return "evidence"
    if path in GOVERNANCE_SUPPORT_FILES:
        return "governance_support"
    if DESIGN.fullmatch(path):
        return "design"
    if path in PUBLIC_DOCUMENT_FILES or DOC.fullmatch(path):
        return "public_document"
    if path in PRODUCT_CHECKER_FILES:
        return "product"
    if CHECKER.fullmatch(path):
        return "governance_checker"
    if CHECKER_TEST.fullmatch(path):
        return "governance_test"
    if CI.fullmatch(path):
        return "ci"
    if HOOK.fullmatch(path):
        return "ci"
    if ASSET.fullmatch(path) or RELEASE_MANIFEST_FIXTURE.fullmatch(path):
        return "asset"
    if path.startswith("third_party/"):
        return "vendored_dependency"
    if path in {
        "release/manifest-v1.schema.json",
        "scripts/env-doc-allowlist.txt",
    }:
        return "configuration"
    if path in {
        "CMakeLists.txt", ".env.example", ".gitignore", ".dockerignore",
        ".clang-format", ".gitattributes", "flake.lock", "flake.nix",
        "LICENSE", "NOTICE",
    } or re.fullmatch(r"docker/Dockerfile\.[A-Za-z0-9_.-]+", path):
        return "configuration"
    if path.startswith(("src/", "include/", "examples/", "tools/", "cmake/", "tests/", "scripts/", "benchmarks/", "triton_kernels/")):
        return "product"
    raise ValueError(f"unclassified repository path {path!r}")


def recognized_evidence(checker_path: str) -> str:
    override = CHECKER_EVIDENCE_OVERRIDES.get(checker_path)
    if override is not None:
        return override
    name = Path(checker_path).stem.removeprefix("check-").replace("-", "_")
    return f"tests/scripts/test_check_{name}.py"


def requires_reviewed_pr(path: str) -> bool:
    """Return whether a canonical governed path requires reviewed PR arrival."""

    return classify_path(path) in PATH_CLASSES


def parse_numstat(output: str) -> list[ChangedPath]:
    changes: list[ChangedPath] = []
    seen: set[str] = set()
    for number, line in enumerate(output.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"numstat line {number} must have exactly three fields")
        added_raw, removed_raw, path = fields
        if path in seen:
            raise ValueError(f"duplicate numstat path {path!r}")
        seen.add(path)
        classify_path(path)
        if added_raw == removed_raw == "-":
            changes.append(ChangedPath(path, None, None))
            continue
        if not added_raw.isascii() or not added_raw.isdecimal() or not removed_raw.isascii() or not removed_raw.isdecimal():
            raise ValueError(f"numstat line {number} has invalid line counts")
        changes.append(ChangedPath(path, int(added_raw), int(removed_raw)))
    return changes


def change_errors(
    changes: list[ChangedPath],
    *,
    evidence_results: dict[str, EvidenceResult] | None = None,
    waivers: tuple[Waiver, ...] = (),
    waiver_scope: str = "",
) -> list[str]:
    errors: list[str] = []
    totals = {path_class: 0 for path_class in PATH_CLASSES}
    changed_paths = {change.path: change for change in changes}
    for change in changes:
        try:
            path_class = classify_path(change.path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if change.lines is None:
            errors.append(f"binary change {change.path!r} has no reviewable line budget")
            continue
        totals[path_class] += change.lines
        if path_class == "governance_checker":
            evidence = recognized_evidence(change.path)
            evidence_change = changed_paths.get(evidence)
            if evidence_change is None or evidence_change.lines is None or evidence_change.lines <= 0:
                errors.append(
                    f"checker change {change.path!r} requires semantic mutation evidence in {evidence}"
                )
            elif evidence_results is not None:
                proof = evidence_results.get(change.path)
                if proof is None:
                    errors.append(
                        f"checker change {change.path!r} has no executable mutation result"
                    )
                elif proof.checker != change.path:
                    errors.append(f"checker evidence identity mismatch for {change.path!r}")
                elif proof.test_module != evidence.removesuffix(".py").replace("/", "."):
                    errors.append(f"checker evidence test mismatch for {change.path!r}")
                elif not proof.rule_ids:
                    errors.append(f"checker change {change.path!r} has no affected POL rule IDs")
                elif proof.head_tests <= 0:
                    errors.append(f"checker change {change.path!r} executed no HEAD tests")
                elif not proof.head_passed:
                    errors.append(f"HEAD checker/test pair failed for {change.path!r}: {proof.detail}")
                elif proof.base_tests <= 0:
                    errors.append(f"checker change {change.path!r} executed no BASE mutation tests")
                elif not proof.base_failed:
                    errors.append(
                        f"BASE checker stayed green for {change.path!r}; changed test is not semantic evidence"
                    )
    for path_class in sorted(PATH_CLASSES):
        budget = PATH_CLASS_BUDGETS[path_class]
        if totals[path_class] > budget:
            applicable = [
                waiver
                for waiver in waivers
                if waiver.rule_id == "POL-PR-SIZE" and waiver.scope == waiver_scope
            ]
            if len(applicable) > 1:
                raise ValueError(
                    f"duplicate applicable waivers for POL-PR-SIZE {waiver_scope}"
                )
            if applicable:
                continue
            errors.append(
                f"{path_class} changes total {totals[path_class]} lines, over the {budget}-line budget"
            )
    return errors


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    )


def resolve_commit(repo: Path, revision: str) -> str:
    if not revision or "\x00" in revision or "\n" in revision:
        raise ValueError(f"invalid revision {revision!r}")
    try:
        oid = git(
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
            repo=repo,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"could not resolve revision {revision!r}") from exc
    if re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        raise ValueError(f"revision {revision!r} did not resolve to one commit")
    return oid


def require_ancestor(repo: Path, base_oid: str, head_oid: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", base_oid, head_oid],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=EVIDENCE_TIMEOUT_SECONDS,
        shell=False,
    )
    if result.returncode == 1:
        raise ValueError("base must be an ancestor of head")
    if result.returncode != 0:
        raise ValueError("could not establish base/head ancestry")


def changed_paths(base: str, head: str, *, repo: Path = ROOT) -> list[ChangedPath]:
    base_oid = resolve_commit(repo, base)
    head_oid = resolve_commit(repo, head)
    require_ancestor(repo, base_oid, head_oid)
    return parse_numstat(
        git("diff", "--no-renames", "--numstat", base_oid, head_oid, repo=repo)
    )


def load_role_discipline():
    spec = importlib.util.spec_from_file_location(
        "check_role_discipline", ROOT / "scripts/check-role-discipline.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("could not load role-discipline checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def affected_policy_rules(
    checker_path: str, rules: dict[str, PolicyRule], root: Path
) -> tuple[str, ...]:
    direct = tuple(
        sorted(
            rule_id
            for rule_id, rule in rules.items()
            if checker_path in {part.strip() for part in rule.enforcement.split(";")}
        )
    )
    declared = CREATED_CHECKER_RULES.get(checker_path, ())
    rule_ids = tuple(sorted(set(direct) | set(declared)))
    if not rule_ids:
        raise ValueError(f"{checker_path} has no affected POL rule mapping")
    for rule_id in rule_ids:
        rule = rules.get(rule_id)
        if rule is None:
            raise ValueError(f"{checker_path} maps unknown policy rule {rule_id}")
        if not (root / rule.procedure).is_file():
            raise ValueError(
                f"{checker_path} rule {rule_id} has no existing procedure {rule.procedure}"
            )
    return rule_ids


def _sanitized_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _run_test_module(worktree: Path, module: str) -> tuple[int, bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", module],
            cwd=worktree,
            env=_sanitized_env(worktree),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=EVIDENCE_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"semantic evidence timed out for {module}") from exc
    counts = [int(value) for value in TEST_COUNT.findall(result.stdout)]
    if len(counts) != 1 or counts[0] <= 0:
        raise ValueError(f"semantic evidence did not execute tests for {module}")
    return counts[0], result.returncode == 0, result.stdout[-2000:]


def _base_checker(repo: Path, base_oid: str, checker_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{base_oid}:{checker_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=EVIDENCE_TIMEOUT_SECONDS,
        shell=False,
    )
    if result.returncode == 0:
        return result.stdout
    if checker_path not in CREATION_MUTATIONS:
        raise ValueError(
            f"{checker_path} is absent at BASE and has no closed creation mutation"
        )
    return CREATION_MUTATIONS[checker_path]


def executable_evidence(
    repo: Path,
    base: str,
    head: str,
    changes: list[ChangedPath],
    rules: dict[str, PolicyRule],
) -> dict[str, EvidenceResult]:
    """Prove each checker change red-before/green-after in an isolated worktree."""

    base_oid = resolve_commit(repo, base)
    head_oid = resolve_commit(repo, head)
    require_ancestor(repo, base_oid, head_oid)
    changed = {item.path for item in changes}
    checkers = sorted(
        item.path
        for item in changes
        if classify_path(item.path) == "governance_checker"
    )
    results: dict[str, EvidenceResult] = {}
    for checker_path in checkers:
        evidence_path = recognized_evidence(checker_path)
        if evidence_path not in changed:
            continue
        module = evidence_path.removesuffix(".py").replace("/", ".")
        rule_ids = affected_policy_rules(checker_path, rules, repo)
        container = Path(
            tempfile.mkdtemp(prefix="vllm-policy-evidence-", dir="/dev/shm")
        )
        worktree = container / "worktree"
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "worktree",
                    "add",
                    "--quiet",
                    "--detach",
                    str(worktree),
                    head_oid,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=EVIDENCE_TIMEOUT_SECONDS,
                shell=False,
                env=_sanitized_env(container),
            )
            head_count, head_passed, head_detail = _run_test_module(worktree, module)
            target = worktree / checker_path
            target.write_bytes(_base_checker(repo, base_oid, checker_path))
            target.chmod(0o755)
            base_count, base_passed, base_detail = _run_test_module(worktree, module)
            results[checker_path] = EvidenceResult(
                checker=checker_path,
                test_module=module,
                rule_ids=rule_ids,
                head_tests=head_count,
                head_passed=head_passed,
                base_tests=base_count,
                base_failed=not base_passed,
                detail=head_detail if not head_passed else base_detail,
            )
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=EVIDENCE_TIMEOUT_SECONDS,
                shell=False,
                env=_sanitized_env(container),
            )
            shutil.rmtree(container, ignore_errors=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", default="", help="accepted for CI compatibility")
    parser.add_argument("--pr-number", default="")
    args = parser.parse_args()
    del args.branch
    try:
        if args.pr_number and (
            not args.pr_number.isascii()
            or not args.pr_number.isdecimal()
            or int(args.pr_number) <= 0
        ):
            raise ValueError("--pr-number must be a positive decimal integer")
        base_oid = resolve_commit(ROOT, args.base)
        head_oid = resolve_commit(ROOT, args.head)
        changes = changed_paths(base_oid, head_oid)
        rules = load_policy(ROOT)
        waivers = tuple(load_waivers(ROOT, rules))
        evidence = executable_evidence(ROOT, base_oid, head_oid, changes, rules)
        errors = change_errors(
            changes,
            evidence_results=evidence,
            waivers=waivers,
            waiver_scope=f"pr:{args.pr_number}" if args.pr_number else "",
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: PR size check could not classify the change: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: every explicit path class is within its review budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
