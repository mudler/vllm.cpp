#!/usr/bin/env python3
"""Mutation tests for the accepted binary-release spike contract."""

from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-release-binary-contract.py"
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_release_binary_contract", CHECKER
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(checker)

CONTRACT_PATHS = (
    "scripts/check-release-binary-contract.py",
    ".agents/specs/release-binary-matrix.md",
    ".agents/engine-matrix.md",
    ".agents/roadmap_v1.md",
    ".agents/NOW.md",
    ".agents/coordination.md",
    ".agents/completed/state-events/2026-08/STATE-20260809T160000-001.md",
    "docs/STATUS.md",
    "docs/BENCHMARKS.md",
    ".github/workflows/ci.yml",
    "scripts/agent-preflight.sh",
    "tests/scripts/test_check_release_binary_contract.py",
)

REQUIRED_TEST_METHODS = (
    "test_repository_contract_passes",
    "test_spec_identity_is_fail_closed",
    "test_each_primary_cuda_sm_is_required",
    "test_primary_cuda_must_stay_one_fat_binary_per_host_abi",
    "test_per_sm_cuda_must_not_become_primary",
    "test_primary_cpu_must_stay_one_adaptive_binary_per_host_abi",
    "test_x86_64_baseline_must_not_require_avx2",
    "test_each_exact_machine_field_is_fail_closed",
    "test_work_table_has_explicit_deps_column",
    "test_each_work_dependency_edge_is_pinned",
    "test_each_human_work_row_id_occurs_exactly_once",
    "test_optional_w12_does_not_block_w13",
    "test_each_required_record_anchor_is_fail_closed",
    "test_release_lifecycle_and_honesty_are_fail_closed",
    "test_public_release_rows_remain_pending",
    "test_human_w12_is_optional_and_cannot_replace_w10",
    "test_human_primary_artifact_contract_matches_machine_block",
    "test_unknown_machine_fields_are_fail_closed",
    "test_each_human_work_dependency_is_pinned",
    "test_primary_cuda_mutation_inventory_literal_is_pinned",
    "test_work_dependency_mutation_inventory_literal_is_pinned",
    "test_each_semantic_inventory_consumer_is_pinned",
    "test_each_semantic_inventory_consumer_body_is_pinned",
    "test_checker_guard_map_keysets_are_exact",
    "test_required_mutation_test_inventory_is_pinned",
    "test_backend_policy_machine_fields_are_required",
    "test_backend_policy_prose_is_fail_closed",
    "test_preflight_and_ci_wiring_is_an_executable_contract",
    "test_preflight_wiring_mutations_fail",
    "test_ci_wiring_mutations_fail",
)

PRIMARY_CUDA_SMS = (
    "80",
    "86",
    "87",
    "89",
    "90a",
    "100a",
    "103a",
    "110",
    "120a",
    "121a",
)

GUARD_MAP_KEYS = {
    "TEST_LITERAL_INVENTORIES": (
        "PRIMARY_CUDA_SMS",
        "EXACT_MACHINE_FIELDS",
        "EXPECTED_DEPS",
        "HUMAN_WORK_IDS",
        "RECORD_ANCHORS",
        "LIFECYCLE_RECORD_MUTATIONS",
        "PUBLIC_PENDING_MUTATIONS",
        "W10_W12_HUMAN_MUTATIONS",
        "PRIMARY_ARTIFACT_PROSE_MUTATIONS",
        "INVENTORY_CONSUMER_METHODS",
        "CONSUMER_FLOW_MUTATIONS",
        "UNKNOWN_MACHINE_FIELD_MUTATIONS",
        "HUMAN_WORK_DEPS",
        "GUARD_MAP_KEYS",
        "BACKEND_POLICY_PROSE_MUTATIONS",
        "PREFLIGHT_WIRING_MUTATIONS",
        "CI_WIRING_MUTATIONS",
    ),
    "TEST_INVENTORY_CONSUMERS": (
        "PRIMARY_CUDA_SMS",
        "EXACT_MACHINE_FIELDS",
        "EXPECTED_DEPS",
        "HUMAN_WORK_IDS",
        "RECORD_ANCHORS",
        "LIFECYCLE_RECORD_MUTATIONS",
        "PUBLIC_PENDING_MUTATIONS",
        "W10_W12_HUMAN_MUTATIONS",
        "PRIMARY_ARTIFACT_PROSE_MUTATIONS",
        "INVENTORY_CONSUMER_METHODS",
        "CONSUMER_FLOW_MUTATIONS",
        "UNKNOWN_MACHINE_FIELD_MUTATIONS",
        "HUMAN_WORK_DEPS",
        "GUARD_MAP_KEYS",
        "BACKEND_POLICY_PROSE_MUTATIONS",
        "PREFLIGHT_WIRING_MUTATIONS",
        "CI_WIRING_MUTATIONS",
    ),
}

RECORD_ANCHORS = {
    ".agents/engine-matrix.md": "| `ENG-RELEASE-BINARIES` |",
    ".agents/roadmap_v1.md": "| REL | `ROAD-V1-RELEASE` |",
    # RELOCATED by ENG-NOW-DERIVED (#374): the release row's live position moved
    # out of the shared digest and into the row's own spec, so this restatement
    # follows it. Independently written, as the contract requires.
    ".agents/specs/release-binary-matrix.md": "**ACTIVE; required W1-W11/W13 implemented in #196.**",
    ".agents/coordination.md": (
        "**Server binary release W1-W13 (`ENG-RELEASE-BINARIES`, 2026-08-09,"
    ),
    ".agents/completed/state-events/2026-08/STATE-20260809T160000-001.md": (
        "# W6 installed server package green"
    ),
    "docs/STATUS.md": "#196 binary pipeline implemented; no published binaries",
    "docs/BENCHMARKS.md": (
        "| **Binary release matrix (ACTIVE; required W1-W11/W13 implemented in #196)** |"
    ),
}

LIFECYCLE_RECORD_MUTATIONS = (
    (
        ".agents/engine-matrix.md",
        "`ACTIVE` | `CLAIM-ENG-RELEASE-BINARIES-W1-W13` |",
        "`DONE` | `CLAIM-ENG-RELEASE-BINARIES-W1-W13` |",
        "engine-matrix release lifecycle",
    ),
    (
        ".agents/engine-matrix.md",
        "hosted ten-SM completion, full eight-tuple dry run, matching-hardware "
        "gates, and tagged publication remain pending",
        "hosted ten-SM completion, full eight-tuple dry run, matching-hardware "
        "gates, and tagged publication are complete",
        "engine-matrix release lifecycle",
    ),
    (
        ".agents/roadmap_v1.md",
        "`ACTIVE` | Required W1-W11/W13 implementation is complete",
        "`DONE` | Required W1-W11/W13 implementation is complete",
        "roadmap release lifecycle",
    ),
    (
        ".agents/roadmap_v1.md",
        "no published binary exists",
        "published binaries exist",
        "roadmap release lifecycle",
    ),
    (
        ".agents/coordination.md",
        "| `ACTIVE` | 2026-08-09 — required W1-W11/W13 implementation complete;",
        "| `DONE` | 2026-08-09 — required W1-W11/W13 implementation complete;",
        "coordination release lifecycle",
    ),
    (
        ".agents/coordination.md",
        "hosted ten-SM completion, full eight-tuple dry run, matching-hardware "
        "gates, rebase/merge, and tagged publication pending",
        "hosted ten-SM completion, full eight-tuple dry run, matching-hardware "
        "gates, rebase/merge, and tagged publication complete",
        "coordination release lifecycle",
    ),
    (
        ".agents/coordination.md",
        "W12 optional/non-primary |",
        "W12 required/primary |",
        "coordination release lifecycle",
    ),
    (
        ".agents/completed/state-events/2026-08/STATE-20260809T160000-001.md",
        "The row remains `ACTIVE`. W1-W4 and W7-W13 remain pending",
        "The row is `DONE`. Every release gate is complete",
        "state release lifecycle",
    ),
)

EXPECTED_DEPS = {
    "W1": "",
    "W2": "W1",
    "W3": "",
    "W4": "",
    "W5": "",
    "W6": "",
    "W7": "W1,W2,W3,W4,W5,W6",
    "W8": "W5,W7",
    "W9": "W3,W4,W5,W6,W7",
    "W10": "W1,W2,W5,W6,W7",
    "W11": "W5,W6,W7",
    "W12": "W1,W2,W5,W6,W7",
    "W13": "W5,W7,W8,W9,W10,W11",
}

HUMAN_WORK_IDS = (
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W8",
    "W9",
    "W10",
    "W11",
    "W12",
    "W13",
)

PUBLIC_PENDING_MUTATIONS = (
    (
        "docs/BENCHMARKS.md",
        "**PENDING:** hosted full matrix, matching hardware, tagged publish",
        "**SHIPPED:** archive, runtime, correctness, and performance evidence "
        "complete",
        "docs/BENCHMARKS.md release row",
    ),
    (
        "docs/STATUS.md",
        "Subset; #196 binary pipeline implemented; no published binaries",
        "Supported; #196: RELEASE DONE/ARTIFACTS✓",
        "docs/STATUS.md release row",
    ),
)

W10_W12_HUMAN_MUTATIONS = (
    (
        "optional single-SM CUDA diagnostic/performance variants",
        "required primary single-SM CUDA release variants replacing W10",
        "W12 deliverable",
    ),
    (
        "generated from the same explicit matrix and evidence; never advertised "
        "as the primary KISS download or used to bypass W10",
        "the primary KISS download; W10 may be bypassed",
        "W12 exit gate",
    ),
)

PRIMARY_ARTIFACT_PROSE_MUTATIONS = (
    (
        "The primary CPU download is\none conservative-baseline, runtime-adaptive "
        "binary per OS+host ABI; the primary\nCUDA download is one fat binary per "
        "OS+host ABI containing every supported SM.",
        "The primary CPU download is one binary per ISA; the primary CUDA "
        "download is one binary per SM.",
        "human primary CPU/CUDA contract",
    ),
    (
        "For x86_64, the baseline must run without AVX2: portable/SSE2 code "
        "remains\ncallable, and higher instructions live only in per-function or "
        "per-TU tiers.",
        "For x86_64, AVX2 is required by the baseline.",
        "human x86_64 no-AVX2 contract",
    ),
)

EXACT_MACHINE_FIELDS = {
    "lifecycle": "ACTIVE",
    "manifest_schema": "vllm.cpp.release-manifest.v1",
    "delivery_pull_request": "196",
    "delivery_mode": "single-pr-W1-W13",
    "work_W5_status": "implemented",
    "work_W6_status": "implemented",
    "work_W12_policy": "optional-non-blocking",
    "archive_claims": "pending",
    "runtime_claims": "pending",
    "metal_channel": "stable-after-runtime-gate",
    "mlx_channel": "preview",
    "vulkan_channel": "preview",
    "musl_channel": "experimental-preview",
    "musl_scope": "cpu-only-no-gpu",
    "rocm_channel": "blocked",
    "gpu_driver_boundary": "external-host-never-bundled",
    "required_anchor_paths": (
        ".agents/engine-matrix.md,.agents/roadmap_v1.md,.agents/NOW.md,"
        ".agents/coordination.md,.agents/completed/state-events/2026-08/"
        "STATE-20260809T160000-001.md,docs/STATUS.md,"
        "docs/BENCHMARKS.md,docs/FEATURES.md,release/manifest-v1.schema.json,"
        "scripts/release_manifest.py,tests/scripts/test_release_manifest.py,"
        "examples/CMakeLists.txt,scripts/package-server.py,"
        "tests/scripts/test_server_package.py"
    ),
}

INVENTORY_CONSUMER_METHODS = {
    "PRIMARY_CUDA_SMS": "test_each_primary_cuda_sm_is_required",
    "EXACT_MACHINE_FIELDS": "test_each_exact_machine_field_is_fail_closed",
    "EXPECTED_DEPS": "test_each_work_dependency_edge_is_pinned",
    "HUMAN_WORK_IDS": "test_each_human_work_row_id_occurs_exactly_once",
    "RECORD_ANCHORS": "test_each_required_record_anchor_is_fail_closed",
    "LIFECYCLE_RECORD_MUTATIONS": "test_release_lifecycle_and_honesty_are_fail_closed",
    "PUBLIC_PENDING_MUTATIONS": "test_public_release_rows_remain_pending",
    "W10_W12_HUMAN_MUTATIONS": "test_human_w12_is_optional_and_cannot_replace_w10",
    "PRIMARY_ARTIFACT_PROSE_MUTATIONS": (
        "test_human_primary_artifact_contract_matches_machine_block"
    ),
    "GUARD_MAP_KEYS": "test_checker_guard_map_keysets_are_exact",
    "BACKEND_POLICY_PROSE_MUTATIONS": "test_backend_policy_prose_is_fail_closed",
    "PREFLIGHT_WIRING_MUTATIONS": "test_preflight_wiring_mutations_fail",
    "CI_WIRING_MUTATIONS": "test_ci_wiring_mutations_fail",
}

CONSUMER_FLOW_MUTATIONS = ("continue", "break", "wrap_false")

UNKNOWN_MACHINE_FIELD_MUTATIONS = (("unexpected_field", "x"),)

HUMAN_WORK_DEPS = {
    "W1": "",
    "W2": "W1",
    "W3": "",
    "W4": "",
    "W5": "",
    "W6": "",
    "W7": "W1,W2,W3,W4,W5,W6",
    "W8": "W5,W7",
    "W9": "W3,W4,W5,W6,W7",
    "W10": "W1,W2,W5,W6,W7",
    "W11": "W5,W6,W7",
    "W12": "W1,W2,W5,W6,W7",
    "W13": "W5,W7,W8,W9,W10,W11",
}

BACKEND_POLICY_FIELDS = {
    "metal_channel": "stable-after-runtime-gate",
    "mlx_channel": "preview",
    "vulkan_channel": "preview",
    "musl_channel": "experimental-preview",
    "musl_scope": "cpu-only-no-gpu",
    "rocm_channel": "blocked",
    "gpu_driver_boundary": "external-host-never-bundled",
}

BACKEND_POLICY_PROSE_MUTATIONS = (
    (
        "| `macos-arm64-metal` | stable after M-series runtime gate |",
        "| `macos-arm64-metal` | preview |",
        "Metal release channel",
    ),
    (
        "| `macos-arm64-metal-mlx` | preview until its exact bundled MLX tuple "
        "is runtime/correctness-gated |",
        "| `macos-arm64-metal-mlx` | stable |",
        "MLX release channel",
    ),
    (
        "| `linux-x86_64-glibc-vulkan` | preview |",
        "| `linux-x86_64-glibc-vulkan` | stable |",
        "Vulkan release channel",
    ),
    (
        "| `linux-x86_64-musl-cpu-static` | experimental preview | literal-static "
        "feasibility lane; CPU only; see the static boundary below |",
        "| `linux-x86_64-musl-cpu-static` | stable | literal-static feasibility "
        "lane with CUDA |",
        "musl experimental CPU-only policy",
    ),
    (
        "| ROCm/HIP | blocked |",
        "| ROCm/HIP | preview |",
        "ROCm release channel",
    ),
    (
        "it never claims to bundle a GPU driver.",
        "it bundles the GPU driver.",
        "external host GPU-driver boundary",
    ),
    (
        "The one literal-static experiment is\n"
        "`linux-x86_64-musl-cpu-static`. It is CPU-only",
        "The one literal-static experiment is\n"
        "`linux-x86_64-musl-cpu-static`. It includes GPU runtimes",
        "musl CPU-only/no-GPU boundary",
    ),
)

PREFLIGHT_WIRING_MUTATIONS = (
    (
        "  check-release-binary-contract\n",
        "",
        "preflight CHECKERS",
    ),
    (
        "  test_check_release_binary_contract\n",
        "",
        "preflight SUITES",
    ),
    (
        'for checker in "${CHECKERS[@]}"; do',
        'for checker in "${CHECKERS[@]}"; do\n  continue',
        "execute release CHECKERS",
    ),
    (
        'for suite in "${SUITES[@]}"; do',
        'for suite in "${SUITES[@]}"; do\n  continue',
        "execute release SUITES",
    ),
    (
        "CHECKERS=(\n",
        "INERT_CHECKERS=(\n",
        "preflight CHECKERS",
    ),
    (
        "SUITES=(\n",
        "INERT_SUITES=(\n",
        "preflight SUITES",
    ),
)

CI_WIRING_MUTATIONS = (
    (
        "          python3 scripts/check-release-binary-contract.py\n",
        "",
        "CI checker",
    ),
    (
        "          python3 tests/scripts/test_check_release_binary_contract.py\n",
        "",
        "CI step",
    ),
    (
        "          python3 scripts/check-release-binary-contract.py\n"
        "          python3 tests/scripts/test_check_release_binary_contract.py\n",
        "          if false; then\n"
        "            python3 scripts/check-release-binary-contract.py\n"
        "            python3 tests/scripts/test_check_release_binary_contract.py\n"
        "          fi\n",
        "direct active commands",
    ),
    (
        "          python3 scripts/check-release-binary-contract.py\n",
        '          echo "python3 scripts/check-release-binary-contract.py"\n',
        "direct active commands",
    ),
    (
        "      - name: Accepted binary-release design and record anchors stay in sync\n"
        "        run: |\n",
        "      - name: Accepted binary-release design and record anchors stay in sync\n"
        "        if: ${{ false }}\n"
        "        run: |\n",
        "direct active commands",
    ),
    (
        "  agent-record:\n",
        "  agent-record:\n    if: ${{ false }}\n",
        "direct active commands",
    ),
)


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check-release-binary-contract.py"),
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class RepoCopy:
    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        for relative in CONTRACT_PATHS:
            source = ROOT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return root

    def __exit__(self, *unused: object) -> None:
        self._tmp.cleanup()


def mutate(root: Path, relative: str, before: str, after: str = "") -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    if before not in text:
        raise AssertionError(f"mutation target missing in {relative}: {before!r}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")


def delete_test_method(root: Path, method: str) -> None:
    relative = "tests/scripts/test_check_release_binary_contract.py"
    path = root / relative
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    matches = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one method {method!r}, found {len(matches)}")
    node = matches[0]
    start = min(
        [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
    ) - 1
    del lines[start : node.end_lineno]
    path.write_text("".join(lines), encoding="utf-8")


def mutate_checker_guard_map(
    root: Path, guard_map: str, mutation: str, key: str
) -> None:
    relative = "scripts/check-release-binary-contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == guard_map
            for target in node.targets
        )
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise AssertionError(f"expected one literal guard map {guard_map!r}")
    mapping = assignments[0].value
    keys = [ast.literal_eval(node) for node in mapping.keys]

    if mutation == "add":
        if key in keys:
            raise AssertionError(f"guard key already exists in {guard_map}: {key}")
        mapping.keys.append(ast.Constant(value=key))
        mapping.values.append(ast.Constant(value=None))
    else:
        if keys.count(key) != 1:
            raise AssertionError(
                f"expected one guard key {key!r} in {guard_map}, found {keys.count(key)}"
            )
        index = keys.index(key)
        if mutation == "delete":
            del mapping.keys[index]
            del mapping.values[index]
        elif mutation == "rename":
            mapping.keys[index] = ast.Constant(value=f"RENAMED_{key}")
        else:
            raise AssertionError(f"unknown guard-map mutation {mutation!r}")

    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


def bypass_checker_keyset_enforcement(root: Path) -> None:
    relative = "scripts/check-release-binary-contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "tuple"
            and len(left.args) == 1
            and isinstance(left.args[0], ast.Name)
            and left.args[0].id == "guard_map"
        ):
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one guard-map keyset enforcement, found {len(matches)}"
        )
    matches[0].test = ast.Constant(value=False)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


def mutate_inventory_consumer_flow(
    root: Path, inventory: str, mutation: str
) -> None:
    relative = "tests/scripts/test_check_release_binary_contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method_name = INVENTORY_CONSUMER_METHODS[inventory]
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    loops = []
    for node in ast.walk(method):
        if not isinstance(node, ast.For):
            continue
        source = node.iter
        if isinstance(source, ast.Name) and source.id == inventory:
            loops.append(node)
        elif (
            isinstance(source, ast.Call)
            and isinstance(source.func, ast.Attribute)
            and isinstance(source.func.value, ast.Name)
            and source.func.value.id == inventory
            and source.func.attr == "items"
        ):
            loops.append(node)
    if len(loops) != 1:
        raise AssertionError(
            f"expected one {inventory} consumer loop in {method_name}, found {len(loops)}"
        )
    loop = loops[0]
    if mutation == "continue":
        loop.body.insert(0, ast.Continue())
    elif mutation == "break":
        loop.body.insert(0, ast.Break())
    elif mutation == "wrap_false":
        loop.body = [ast.If(test=ast.Constant(False), body=loop.body, orelse=[])]
    else:
        raise AssertionError(f"unknown consumer-flow mutation {mutation!r}")
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


def bypass_unknown_field_enforcement(root: Path) -> None:
    relative = "scripts/check-release-binary-contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "extra"
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one unknown-field equality, found {len(matches)}"
        )
    matches[0].test = ast.Constant(False)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


def mutate_human_work_dependency(root: Path, work: str, replacement: str) -> None:
    relative = ".agents/specs/release-binary-matrix.md"
    path = root / relative
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith(f"| {work} |")]
    if len(matches) != 1:
        raise AssertionError(f"expected one human work row {work}, found {len(matches)}")
    index = matches[0]
    cells = lines[index].split("|")
    cells[2] = f" {replacement} "
    lines[index] = "|".join(cells)
    path.write_text("".join(lines), encoding="utf-8")


def bypass_human_work_dependency_enforcement(root: Path) -> None:
    relative = "scripts/check-release-binary-contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Attribute)
            and isinstance(left.func.value, ast.Name)
            and left.func.value.id == "rows"
            and left.func.attr == "get"
        ):
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one human dependency equality, found {len(matches)}"
        )
    matches[0].test = ast.Constant(False)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


class LiveContract(unittest.TestCase):
    def test_repository_contract_passes(self) -> None:
        result = run_checker(ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class MissingReviewGuardTests(unittest.TestCase):
    def test_backend_policy_machine_fields_are_required(self) -> None:
        for field, value in BACKEND_POLICY_FIELDS.items():
            with self.subTest(field=field):
                self.assertEqual(checker.EXPECTED_FIELDS[field], value)

    def test_preflight_and_ci_wiring_is_an_executable_contract(self) -> None:
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertEqual(checker.wiring_errors(preflight, ci), [])

    def test_backend_policy_prose_is_fail_closed(self) -> None:
        for before, after, reason in BACKEND_POLICY_PROSE_MUTATIONS:
            with self.subTest(reason=reason), RepoCopy() as root:
                mutate(root, ".agents/specs/release-binary-matrix.md", before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_preflight_wiring_mutations_fail(self) -> None:
        for before, after, reason in PREFLIGHT_WIRING_MUTATIONS:
            with self.subTest(reason=reason, mutation=before), RepoCopy() as root:
                mutate(root, "scripts/agent-preflight.sh", before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_ci_wiring_mutations_fail(self) -> None:
        for before, after, reason in CI_WIRING_MUTATIONS:
            with self.subTest(reason=reason, mutation=before), RepoCopy() as root:
                mutate(root, ".github/workflows/ci.yml", before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)


class AcceptedDesignMutations(unittest.TestCase):
    SPEC = ".agents/specs/release-binary-matrix.md"

    def assert_mutation_fails(self, before: str, after: str, reason: str) -> None:
        with RepoCopy() as root:
            mutate(root, self.SPEC, before, after)
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(reason, result.stdout + result.stderr)

    def test_spec_identity_is_fail_closed(self) -> None:
        self.assert_mutation_fails(
            "identity=ENG-RELEASE-BINARIES",
            "identity=ENG-RELEASE-ARCHIVES",
            "identity",
        )

    def test_each_primary_cuda_sm_is_required(self) -> None:
        for sm in PRIMARY_CUDA_SMS:
            with self.subTest(sm=sm):
                self.assert_mutation_fails(
                    "primary_cuda_sms=80,86,87,89,90a,100a,103a,110,120a,121a",
                    "primary_cuda_sms=" + ",".join(
                        value for value in PRIMARY_CUDA_SMS if value != sm
                    ),
                    "primary CUDA SM set",
                )

    def test_primary_cuda_must_stay_one_fat_binary_per_host_abi(self) -> None:
        self.assert_mutation_fails(
            "primary_cuda_artifact=one-fat-binary-per-os-host-abi",
            "primary_cuda_artifact=one-binary-per-sm",
            "primary CUDA artifact",
        )

    def test_per_sm_cuda_must_not_become_primary(self) -> None:
        self.assert_mutation_fails(
            "per_sm_cuda=optional-non-primary",
            "per_sm_cuda=primary",
            "per-SM CUDA",
        )

    def test_primary_cpu_must_stay_one_adaptive_binary_per_host_abi(self) -> None:
        self.assert_mutation_fails(
            "primary_cpu_artifact=one-adaptive-binary-per-os-host-abi",
            "primary_cpu_artifact=one-binary-per-isa",
            "primary CPU artifact",
        )

    def test_x86_64_baseline_must_not_require_avx2(self) -> None:
        self.assert_mutation_fails(
            "x86_64_baseline=portable-sse2-without-avx2",
            "x86_64_baseline=avx2-required",
            "x86_64 baseline",
        )

    def test_each_exact_machine_field_is_fail_closed(self) -> None:
        for field, expected in EXACT_MACHINE_FIELDS.items():
            with self.subTest(field=field), RepoCopy() as root:
                mutate(
                    root,
                    self.SPEC,
                    f"{field}={expected}",
                    f"{field}=BROKEN",
                )
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn("expected", result.stdout + result.stderr)

    def test_unknown_machine_fields_are_fail_closed(self) -> None:
        for field, value in UNKNOWN_MACHINE_FIELD_MUTATIONS:
            with self.subTest(field=field), RepoCopy() as root:
                mutate(
                    root,
                    self.SPEC,
                    "lifecycle=ACTIVE",
                    f"lifecycle=ACTIVE\n{field}={value}",
                )
                bypass_unknown_field_enforcement(root)
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn("unknown fields", result.stdout + result.stderr)


class WorkGraphMutations(unittest.TestCase):
    SPEC = ".agents/specs/release-binary-matrix.md"

    def test_work_table_has_explicit_deps_column(self) -> None:
        with RepoCopy() as root:
            mutate(root, self.SPEC, "| Work | Deps | Deliverable | Exit gate |", "| Work | Deliverable | Exit gate |")
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Deps column", result.stdout + result.stderr)

    def test_each_work_dependency_edge_is_pinned(self) -> None:
        for work, deps in EXPECTED_DEPS.items():
            with self.subTest(work=work), RepoCopy() as root:
                mutate(root, self.SPEC, f"work_{work}={deps}", f"work_{work}=BROKEN")
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{work} dependencies", result.stdout + result.stderr)

    def test_each_human_work_dependency_is_pinned(self) -> None:
        for work, expected in HUMAN_WORK_DEPS.items():
            with self.subTest(work=work, expected=expected), RepoCopy() as root:
                mutate_human_work_dependency(root, work, "W99")
                bypass_human_work_dependency_enforcement(root)
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(
                    "human work-table dependency mirror",
                    result.stdout + result.stderr,
                )

    def test_each_human_work_row_id_occurs_exactly_once(self) -> None:
        for work in HUMAN_WORK_IDS:
            with self.subTest(work=work, mutation="duplicate"), RepoCopy() as root:
                path = root / self.SPEC
                row = next(
                    line
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.startswith(f"| {work} |")
                )
                mutate(root, self.SPEC, row, f"{row}\n{row}")
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("exactly once", result.stdout + result.stderr)

        with self.subTest(mutation="missing"), RepoCopy() as root:
            path = root / self.SPEC
            row = next(
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("| W1 |")
            )
            mutate(root, self.SPEC, row)
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("missing", result.stdout + result.stderr)

        with self.subTest(mutation="unexpected"), RepoCopy() as root:
            mutate(root, self.SPEC, "| W13 |", "| W14 |")
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("unexpected", result.stdout + result.stderr)

    def test_optional_w12_does_not_block_w13(self) -> None:
        with RepoCopy() as root:
            mutate(root, self.SPEC, "work_W13=W5,W7,W8,W9,W10,W11", "work_W13=W5,W7,W8,W9,W10,W11,W12")
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("W13 dependencies", result.stdout + result.stderr)


class RecordAnchorMutations(unittest.TestCase):
    def test_each_required_record_anchor_is_fail_closed(self) -> None:
        for relative, anchor in RECORD_ANCHORS.items():
            with self.subTest(path=relative), RepoCopy() as root:
                mutate(root, relative, anchor)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"{relative} is missing required release anchor", result.stdout + result.stderr)

    def test_release_lifecycle_and_honesty_are_fail_closed(self) -> None:
        for relative, before, after, reason in LIFECYCLE_RECORD_MUTATIONS:
            with self.subTest(path=relative, mutation=before), RepoCopy() as root:
                mutate(root, relative, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

        with RepoCopy() as root:
            mutate(
                root,
                "tests/scripts/test_check_release_binary_contract.py",
                '        "engine-matrix release lifecycle",',
                '        "renamed engine lifecycle",',
            )
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("LIFECYCLE_RECORD_MUTATIONS", result.stdout + result.stderr)


class HumanContractMutations(unittest.TestCase):
    SPEC = ".agents/specs/release-binary-matrix.md"

    def test_public_release_rows_remain_pending(self) -> None:
        for relative, before, after, reason in PUBLIC_PENDING_MUTATIONS:
            with self.subTest(path=relative), RepoCopy() as root:
                mutate(root, relative, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_human_w12_is_optional_and_cannot_replace_w10(self) -> None:
        for before, after, reason in W10_W12_HUMAN_MUTATIONS:
            with self.subTest(reason=reason), RepoCopy() as root:
                mutate(root, self.SPEC, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_human_primary_artifact_contract_matches_machine_block(self) -> None:
        for before, after, reason in PRIMARY_ARTIFACT_PROSE_MUTATIONS:
            with self.subTest(reason=reason), RepoCopy() as root:
                mutate(root, self.SPEC, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_primary_cuda_mutation_inventory_literal_is_pinned(self) -> None:
        with RepoCopy() as root:
            mutate(
                root,
                "tests/scripts/test_check_release_binary_contract.py",
                '    "120a",\n    "121a",\n)\n\nGUARD_MAP_KEYS',
                '    "120a",\n)\n\nGUARD_MAP_KEYS',
            )
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PRIMARY_CUDA_SMS", result.stdout + result.stderr)

    def test_work_dependency_mutation_inventory_literal_is_pinned(self) -> None:
        with RepoCopy() as root:
            mutate(
                root,
                "tests/scripts/test_check_release_binary_contract.py",
                '    "W2": "W1",',
                '    "W2": "",',
            )
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("EXPECTED_DEPS", result.stdout + result.stderr)

    def test_each_semantic_inventory_consumer_is_pinned(self) -> None:
        mutations = (
            ("for sm in PRIMARY_CUDA_SMS:", "for sm in ():", "PRIMARY_CUDA_SMS"),
            (
                "for work, deps in EXPECTED_DEPS.items():",
                "for work, deps in {}.items():",
                "EXPECTED_DEPS",
            ),
            (
                "for relative, anchor in RECORD_ANCHORS.items():",
                "for relative, anchor in {}.items():",
                "RECORD_ANCHORS",
            ),
            (
                "for relative, before, after, reason in LIFECYCLE_RECORD_MUTATIONS:",
                "for relative, before, after, reason in ():",
                "LIFECYCLE_RECORD_MUTATIONS",
            ),
            (
                "for relative, before, after, reason in PUBLIC_PENDING_MUTATIONS:",
                "for relative, before, after, reason in ():",
                "PUBLIC_PENDING_MUTATIONS",
            ),
            (
                "for before, after, reason in W10_W12_HUMAN_MUTATIONS:",
                "for before, after, reason in ():",
                "W10_W12_HUMAN_MUTATIONS",
            ),
            (
                "for before, after, reason in PRIMARY_ARTIFACT_PROSE_MUTATIONS:",
                "for before, after, reason in ():",
                "PRIMARY_ARTIFACT_PROSE_MUTATIONS",
            ),
        )
        for before, after, inventory in mutations:
            with self.subTest(inventory=inventory), RepoCopy() as root:
                mutate(
                    root,
                    "tests/scripts/test_check_release_binary_contract.py",
                    before,
                    after,
                )
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(inventory, result.stdout + result.stderr)

    def test_each_semantic_inventory_consumer_body_is_pinned(self) -> None:
        loop = ast.parse(
            "for item in ITEMS:\n"
            "    def identity(value):\n"
            "        return value\n"
        ).body[0]
        self.assertIsInstance(loop, ast.For)
        baseline = checker._consumer_body_digest(loop)
        nested = next(
            node for node in ast.walk(loop) if isinstance(node, ast.FunctionDef)
        )
        nested._fields = (*nested._fields, "type_params")
        nested.type_params = []
        self.assertEqual(checker._consumer_body_digest(loop), baseline)

        for inventory, method in INVENTORY_CONSUMER_METHODS.items():
            for mutation in CONSUMER_FLOW_MUTATIONS:
                with (
                    self.subTest(
                        inventory=inventory,
                        method=method,
                        mutation=mutation,
                    ),
                    RepoCopy() as root,
                ):
                    mutate_inventory_consumer_flow(root, inventory, mutation)
                    result = run_checker(root)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    output = result.stdout + result.stderr
                    self.assertIn(inventory, output)
                    self.assertIn("consumer body", output)

    def test_checker_guard_map_keysets_are_exact(self) -> None:
        for guard_map, keys in GUARD_MAP_KEYS.items():
            for key in keys:
                for mutation in ("delete", "rename"):
                    with (
                        self.subTest(
                            guard_map=guard_map,
                            key=key,
                            mutation=mutation,
                        ),
                        RepoCopy() as root,
                    ):
                        mutate_checker_guard_map(root, guard_map, mutation, key)
                        result = run_checker(root)
                        self.assertNotEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        self.assertIn(
                            "semantic mutation guard map keyset",
                            result.stdout + result.stderr,
                        )

            with self.subTest(guard_map=guard_map, mutation="add"), RepoCopy() as root:
                mutate_checker_guard_map(root, guard_map, "add", "EXTRA_GUARD")
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(
                    "semantic mutation guard map keyset",
                    result.stdout + result.stderr,
                )

            with (
                self.subTest(guard_map=guard_map, mutation="bypass"),
                RepoCopy() as root,
            ):
                mutate_checker_guard_map(root, guard_map, "delete", keys[0])
                bypass_checker_keyset_enforcement(root)
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(keys[0], result.stdout + result.stderr)

    def test_required_mutation_test_inventory_is_pinned(self) -> None:
        for method in REQUIRED_TEST_METHODS:
            for mutation in ("delete", "rename"):
                with self.subTest(method=method, mutation=mutation), RepoCopy() as root:
                    if mutation == "delete":
                        delete_test_method(root, method)
                    else:
                        mutate(
                            root,
                            "tests/scripts/test_check_release_binary_contract.py",
                            f"    def {method}(",
                            f"    def renamed_{method}(",
                        )
                    result = run_checker(root)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn(
                        "required mutation-test inventory",
                        result.stdout + result.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
