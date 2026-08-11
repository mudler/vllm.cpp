#!/usr/bin/env python3
"""Enforce explicit path classification and the checker-evidence contract.

The per-class LINE BUDGETS this file used to enforce were retired 2026-08-10;
see the note where they stood. What remains: every changed path must classify
explicitly, binaries fail closed, a governance-checker change must carry
executable mutation evidence, and product paths must arrive on a PR."""

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
# NO LINE BUDGET. The per-class budgets that used to live here were retired on
# 2026-08-10 by developer decision, on measured grounds: over the last 22 merged
# PRs the `product` budget of 900 lines was exceeded by 9 of them (41%), and
# tests were 33-57% of the diff in every large one. A gate that fires on four
# changes in ten is not a budget, it is noise that teaches people to waive it,
# and charging RED-first mutation tests against the same allowance as kernel
# code penalised exactly the discipline the rest of this document demands.
# Reviewability is now a review judgement, not an arithmetic one.
#
# Everything else this checker enforces is unchanged and is NOT a size rule:
# explicit path classification (no blanket directory exemptions), the
# fail-closed binary guard, the checker-change mutation-evidence contract, and
# the role checks that keep product paths on a PR.

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
    }
)
APPEND_ONLY_FILES = frozenset(
    {
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
        # The per-model coverage checklist that porting.md points at (#318). Same
        # procedure class as its sibling guides; listed explicitly rather than
        # letting .agents/ become a blanket exemption.
        ".agents/porting-a-model.md",
        ".agents/benchmarking.md",
        ".agents/bugfixing.md",
        ".agents/prompts/implementer.md",
        ".agents/prompts/operator.md",
        ".agents/prompts/reviewer.md",
        ".agents/backends.md",
        ".agents/developer-preferences.example.md",
        ".agents/environment.md",
        ".agents/mission.md",
        ".agents/parity-lever-protocol.md",
        ".agents/upstream-sync.md",
        ".agents/vllm-v1-v2.md",
    }
)
GOVERNANCE_SUPPORT_FILES = frozenset(
    {
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
# The published documentation site. Its layouts, CSS, config and assets are
# prose and presentation for a PUBLIC surface, reviewed the way the documents
# themselves are -- not product code, and not CI (the workflow that publishes it
# keeps its own `ci` class). One class for the whole directory on purpose:
# splitting layouts from config would let a large redesign hide half its diff in
# the cheaper bucket, which is the blanket-exemption failure AGENTS.md names.
SITE = re.compile(r"website/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
# `website/static/` is the site's artwork and fonts -- logos, a favicon, woff2.
# Not prose, and the binaries among them have no reviewable line budget at all,
# so they take the `asset` class the same way any other shipped artwork does.
# Kept as a separate pattern rather than an extension list: what makes these
# assets is WHERE they live, and static/ is Hugo's name for exactly that.
SITE_ASSET = re.compile(r"website/static/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
SPEC = re.compile(r"\.agents/specs/[A-Za-z0-9_.-]+\.md\Z")
SPEC_EVIDENCE = re.compile(r"\.agents/specs/[A-Za-z0-9_.-]+\.(?:patch|json|log)\Z")
COMPLETED = re.compile(r"\.agents/completed/[A-Za-z0-9_.-]+\.md\Z")
# One file per active claim (ENG-RECORD-CONFLICT-SURFACES, #364). The claims
# TABLE in coordination.md was insert-at-one-anchor, so every concurrent claim
# collided there -- 8 of the 16 conflicting open PRs at origin/main d928e2c3,
# including six from ONE author's sequential ROCm stack whose only conflict was
# this. A claim in its own file has one writer and cannot collide. Classified
# with the other per-row records it now resembles.
CLAIM = re.compile(r"\.agents/claims/[A-Za-z0-9_.-]+\.md\Z")
# Retired state evidence, moved wholesale under completed/ when history became
# git. It is archived evidence, classified like every other completed record.
COMPLETED_STATE_EVENT = re.compile(
    r"\.agents/completed/state-events/\d{4}-\d{2}/STATE-[A-Za-z0-9-]+\.md\Z"
)
SYNC_RECORD = re.compile(r"\.agents/sync/[A-Za-z0-9_.-]+\.md\Z")
HOOK = re.compile(r"\.githooks/(?:README\.md|[A-Za-z0-9_.-]+)\Z")
BENCH_EVIDENCE = re.compile(r"(?:benchmarks/(?:demo|media)|docs/bench-evidence)/[A-Za-z0-9_.-]+\.(?:json|png|gif|mp4|log)\Z")
STATE_MIGRATION_MANIFEST = ".agents/completed/state-migration-manifest.csv"
STATE_MIGRATION_MANIFEST_ARCHIVE = re.compile(
    r"\.agents/completed/state-migration-manifest-"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?\.csv\Z"
)
ASSET = re.compile(r"assets/[A-Za-z0-9_.-]+\.(?:png|svg)\Z")
RELEASE_MANIFEST_FIXTURE = re.compile(
    r"tests/scripts/fixtures/release_manifest/v[0-9]+/[a-z0-9-]+\.json\Z"
)

# RETIRED SURFACES -- deleted from the tree, still classified ON PURPOSE.
#
# `classify_path` fails closed on an unknown path, and a deleted file still
# appears in the diff of the commit that removes it, so a surface with no class
# reds the very change that retires it and every later range spanning that
# commit. That is why these stay.
#
# They live HERE, in one table with the reason written once, because the same
# retention used to be spread through POLICY_FILES, PROJECT_RECORD_FILES,
# PROCEDURE_FILES and two module-level regexes with the reason written in only
# ONE of them -- which reads as abandoned scaffolding rather than a deliberate
# fail-closed guard, and was mistaken for exactly that. A path may appear here or
# in a live group, never both.
#
# Each entry keeps the class its path resolved to while it was live, so the
# review budget a historical diff spends does not move.
RETIRED_PATHS = {
    # `policy: retire the waiver registry` (#281): a registry of exceptions is a
    # state log, so an exception now argues for itself in the commit that needs
    # it and `git log --grep` is the record. Same reasoning as policy.csv below.
    ".agents/waivers.csv": "policy",
    "scripts/waivers.py": "governance_support",
    "tests/scripts/test_waivers.py": "checker_test",
    # `policy: the code is the state, git is the history` (0f3e44ee): AGENTS.md
    # became the single normative surface and git became the history.
    ".agents/policy.csv": "policy",
    ".agents/policy-cutover": "policy",
    ".agents/state.md": "project_record",
    ".agents/state.csv": "project_record",
    # `policy: optimize the agent protocol for strict, executable compliance`
    # (1a021b1b, #128): folded into workflow/verification/porting.md.
    ".agents/ai-coding-assistants.md": "procedure",
    ".agents/benchmark-protocol.md": "procedure",
    ".agents/directives.md": "procedure",
    ".agents/discipline.md": "procedure",
    ".agents/gates.md": "procedure",
    ".agents/test-porting.md": "procedure",
}
RETIRED_PATTERNS = (
    # The CSV index and the per-event files of the retired state record, both
    # removed by 0f3e44ee. Their ARCHIVE under `.agents/completed/state-events/`
    # is LIVE evidence -- 160 files, moved verbatim rather than deleted -- and
    # classifies through COMPLETED_STATE_EVENT, not here.
    (re.compile(r"\.agents/state-index/\d{4}-\d{2}-\d{3}\.csv\Z"), "append_only_record"),
    (
        re.compile(r"\.agents/state-events/\d{4}-\d{2}/STATE-[A-Za-z0-9-]+\.md\Z"),
        "append_only_record",
    ),
)

CHECKER_EVIDENCE_OVERRIDES = {
    "scripts/check-agent-record.py": "tests/scripts/test_agent_record.py",
    "scripts/check-role-discipline.py": "tests/scripts/test_check_role_discipline.py",
    "scripts/check-doc-checkpoint.py": "tests/scripts/test_doc_checkpoint.py",
    # Its suite predates the test_check_<name> convention and CI runs it under
    # the older name, so the derived path pointed at a file that does not
    # exist and NO change to this checker could ever satisfy its own evidence
    # rule. Mapped to the file CI actually runs.
    "scripts/check-device-leakage.py": "tests/scripts/test_device_leakage.py",
    "scripts/check-release-workflow.py": "tests/scripts/test_release_pipeline.py",
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
    ),
    "scripts/check-arm-isa-build.py": DISABLED_CREATION_CHECKER,
    "scripts/check-cpu-isa-build.py": DISABLED_CREATION_CHECKER,
    "scripts/check-cuda-fat-gencode.py": DISABLED_CREATION_CHECKER,
    "scripts/check-release-workflow.py": DISABLED_CREATION_CHECKER,
    "scripts/validate-release-archive.py": DISABLED_CREATION_CHECKER,
    "scripts/check-pr-size.py": DISABLED_CREATION_CHECKER,
    "scripts/check-prompt-contract.py": DISABLED_CREATION_CHECKER,
    "scripts/check-triton-aot-multiarch.py": DISABLED_CREATION_CHECKER,
    # A new checker has no BASE version to mutate, so it registers the disabled
    # form its own tests must reject. The empty stub exits 0 and prints nothing,
    # which fails every case in tests/scripts/test_check_site.py -- including
    # the clean-tree case, which asserts the "nav in bijection" line.
    "scripts/check-site.py": DISABLED_CREATION_CHECKER,
    # ENG-RELEASE-CONTAINERS. Both suites load the checker as a module and call
    # into it (check_shape/check_dockerfile, validate), so the disabled stub --
    # which defines none of them -- fails every case rather than passing a
    # reduced one.
    "scripts/check-container-matrix.py": DISABLED_CREATION_CHECKER,
    "scripts/check-container-workflow.py": DISABLED_CREATION_CHECKER,
}
SELF_CHECKER = "scripts/check-pr-size.py"
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


def retired_class(path: str) -> str | None:
    """The class a RETIRED surface keeps, or None when the path is not retired."""

    known = RETIRED_PATHS.get(path)
    if known is not None:
        return known
    for pattern, path_class in RETIRED_PATTERNS:
        if pattern.fullmatch(path):
            return path_class
    return None


def classify_path(path: str) -> str:
    """Return one closed path class or reject an unknown/noncanonical path."""

    if not _canonical_path(path):
        raise ValueError(f"noncanonical repository path {path!r}")
    # Ahead of every other rule: a generated artifact under src/ would otherwise
    # fall through to `product` and spend a human-review budget on hex.
    if path in GENERATED_FILES:
        return "generated"
    retired = retired_class(path)
    if retired is not None:
        return retired
    if path in POLICY_FILES:
        return "policy"
    if path in APPEND_ONLY_FILES:
        return "append_only_record"
    if path in PROJECT_RECORD_FILES:
        return "project_record"
    if path == ".agents/upstream-inventory.json":
        return "project_record"
    if (
        path in PROCEDURE_FILES
        or SPEC.fullmatch(path)
        or CLAIM.fullmatch(path)
        or COMPLETED.fullmatch(path)
        or COMPLETED_STATE_EVENT.fullmatch(path)
    ):
        return "procedure"
    if (
        path == STATE_MIGRATION_MANIFEST
        or STATE_MIGRATION_MANIFEST_ARCHIVE.fullmatch(path)
        or SPEC_EVIDENCE.fullmatch(path)
        or SYNC_RECORD.fullmatch(path)
        or BENCH_EVIDENCE.fullmatch(path)
    ):
        return "evidence"
    if path in GOVERNANCE_SUPPORT_FILES:
        return "governance_support"
    if DESIGN.fullmatch(path):
        return "design"
    if SITE_ASSET.fullmatch(path):
        return "asset"
    if path in PUBLIC_DOCUMENT_FILES or DOC.fullmatch(path) or SITE.fullmatch(path):
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
        "release/release-matrix.json",
        "release/container-matrix.json",
        "scripts/env-doc-allowlist.txt",
    }:
        return "configuration"
    if path in {
        "CMakeLists.txt", ".env.example", ".gitignore", ".dockerignore",
        ".clang-format", ".gitattributes", "flake.lock", "flake.nix",
        "LICENSE", "NOTICE",
    } or re.fullmatch(r"docker/Dockerfile(?:\.[A-Za-z0-9_.-]+)?", path) or re.fullmatch(
        r"docker/[A-Za-z0-9_.-]+\.sh", path
    ):
        # The suffixed form covered docker/Dockerfile.arm64. ENG-RELEASE-CONTAINERS
        # adds the unsuffixed multi-lane docker/Dockerfile and its healthcheck
        # script, and this function FAILS CLOSED, so leaving them out rejected the
        # whole change rather than misclassifying it.
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
) -> list[str]:
    errors: list[str] = []
    changed_paths = {change.path: change for change in changes}
    for change in changes:
        try:
            path_class = classify_path(change.path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if change.lines is None:
            errors.append(f"binary change {change.path!r} is not reviewable as text")
            continue
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
        evidence = executable_evidence(ROOT, base_oid, head_oid, changes)
        errors = change_errors(
            changes,
            evidence_results=evidence,
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
