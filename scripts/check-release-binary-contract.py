#!/usr/bin/env python3
"""Fail closed when the accepted binary-release spike contract drifts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


BEGIN = "<!-- release-binary-contract:begin -->"
END = "<!-- release-binary-contract:end -->"
SPEC_PATH = ".agents/specs/release-binary-matrix.md"
TEST_PATH = "tests/scripts/test_check_release_binary_contract.py"
PREFLIGHT_PATH = "scripts/agent-preflight.sh"
CI_PATH = ".github/workflows/ci.yml"

IDENTITY = "ENG-RELEASE-BINARIES"
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
WORK_DEPS = {
    "W1": (),
    "W2": ("W1",),
    "W3": (),
    "W4": (),
    "W5": (),
    "W6": (),
    "W7": ("W1", "W2", "W3", "W4", "W5", "W6"),
    "W8": ("W5", "W7"),
    "W9": ("W3", "W4", "W5", "W6", "W7"),
    "W10": ("W1", "W2", "W5", "W6", "W7"),
    "W11": ("W5", "W6", "W7"),
    "W12": ("W1", "W2", "W5", "W6", "W7"),
    "W13": ("W5", "W7", "W8", "W9", "W10", "W11"),
}

ANCHORS = {
    ".agents/engine-matrix.md": "| `ENG-RELEASE-BINARIES` |",
    ".agents/roadmap_v1.md": "| REL | `ROAD-V1-RELEASE` |",
    # RELOCATED 2026-08-11 (ENG-NOW-DERIVED, #374). This anchor used to pin a
    # per-ROW line inside .agents/NOW.md, which is one of the requirements that
    # made that file a surface every PR had to keep current. The row's live
    # position now lives in the row's OWN spec under `## Now` -- one writer, and
    # the row-owned source of its live position.
    ".agents/specs/release-binary-matrix.md": "**ACTIVE; required W1-W11/W13 implemented and v0.0.2 published.**",
    ".agents/coordination.md": "**Server binary release W1-W13 (`ENG-RELEASE-BINARIES`, 2026-08-09,",
    ".agents/completed/state-events/2026-08/STATE-20260809T160000-001.md": "# W6 installed server package green",
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
        "v0.0.2 published eight archive/checksum/provenance triplets plus two indexes",
        "v0.0.2 publication is pending",
        "engine-matrix release lifecycle",
    ),
    (
        ".agents/roadmap_v1.md",
        "`ACTIVE` | v0.0.2 published eight primary archive/checksum/provenance triplets",
        "`DONE` | v0.0.2 published eight primary archive/checksum/provenance triplets",
        "roadmap release lifecycle",
    ),
    (
        ".agents/roadmap_v1.md",
        "Windows W14-W16 are implemented for one PR",
        "Windows v0.0.3-pre.1 is published",
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
        "hosted ten-SM completion, full eight-tuple dry run, matching-hardware gates, rebase/merge, and tagged publication pending",
        "hosted ten-SM completion, full eight-tuple dry run, matching-hardware gates, rebase/merge, and tagged publication complete",
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

BACKEND_POLICY_PROSE = {
    "Metal release channel": (
        "| `macos-arm64-metal` | stable after M-series runtime gate |"
    ),
    "MLX release channel": (
        "| `macos-arm64-metal-mlx` | preview until its exact bundled MLX tuple "
        "is runtime/correctness-gated |"
    ),
    "Vulkan release channel": "| `linux-x86_64-glibc-vulkan` | preview |",
    "musl experimental CPU-only policy": (
        "| `linux-x86_64-musl-cpu-static` | experimental preview | "
        "literal-static feasibility lane; CPU only; see the static boundary below |"
    ),
    "ROCm release channel": "| ROCm/HIP | blocked |",
    "external host GPU-driver boundary": "it never claims to bundle a GPU driver.",
    "musl CPU-only/no-GPU boundary": (
        "The one literal-static experiment is "
        "`linux-x86_64-musl-cpu-static`. It is CPU-only"
    ),
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
    ("  check-release-binary-contract\n", "", "preflight CHECKERS"),
    ("  test_check_release_binary_contract\n", "", "preflight SUITES"),
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
    ("CHECKERS=(\n", "INERT_CHECKERS=(\n", "preflight CHECKERS"),
    ("SUITES=(\n", "INERT_SUITES=(\n", "preflight SUITES"),
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

HUMAN_CONTRACT = {
    "human primary CPU/CUDA contract": (
        "The primary CPU download is one conservative-baseline, runtime-adaptive "
        "binary per OS+host ABI; the primary CUDA download is one fat binary per "
        "OS+host ABI containing every supported SM."
    ),
    "human x86_64 no-AVX2 contract": (
        "For x86_64, the baseline must run without AVX2: portable/SSE2 code "
        "remains callable, and higher instructions live only in per-function or "
        "per-TU tiers."
    ),
}

WORK_CONTENT = {
    "W10": (
        "primary Linux CUDA fat bundles for x86_64 and aarch64 host ABIs",
        "each extracted archive contains all ten SMs and six exact AOT trees; "
        "per-SM evidence remains independent; no host ABI is inferred from the "
        "other",
    ),
    "W12": (
        "optional single-SM CUDA diagnostic/performance variants",
        "generated from the same explicit matrix and evidence; never advertised "
        "as the primary KISS download or used to bypass W10",
    ),
}

PUBLIC_PENDING_MUTATIONS = (
    (
        ".agents/specs/release-binary-matrix.md",
        "**ACTIVE; required W1-W11/W13 implemented and v0.0.2 published.**",
        "**DONE; all release work published.**",
        "missing required release anchor",
    ),
    (
        ".agents/roadmap_v1.md",
        "Windows W14-W16 are implemented for one PR; native hosted gates, merged-SHA ten-tuple dry run, matching-hardware evidence, `v0.0.3-pre.1` publication and 32-asset audit remain pending",
        "Windows publication is complete",
        "roadmap release lifecycle",
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

EXPECTED_TEST_LITERAL_INVENTORY_KEYS = (
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
)

EXPECTED_TEST_INVENTORY_CONSUMER_KEYS = (
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
)

EXPECTED_GUARD_MAP_KEYS = {
    "TEST_LITERAL_INVENTORIES": EXPECTED_TEST_LITERAL_INVENTORY_KEYS,
    "TEST_INVENTORY_CONSUMERS": EXPECTED_TEST_INVENTORY_CONSUMER_KEYS,
}

TEST_LITERAL_INVENTORIES = {
    "PRIMARY_CUDA_SMS": PRIMARY_CUDA_SMS,
    "EXACT_MACHINE_FIELDS": {
        "lifecycle": "ACTIVE",
        "manifest_schema": "vllm.cpp.release-manifest.v1",
        "delivery_pull_request": "196",
        "delivery_mode": "single-pr-W1-W13",
        "work_W5_status": "implemented",
        "work_W6_status": "implemented",
        "work_W12_policy": "optional-non-blocking",
        "archive_claims": "published-v0.0.2",
        "published_tag": "v0.0.2",
        "published_sha": "7020de93652ca920424a10ac5255b34810dd2f24",
        "published_run": "31466516224",
        "published_asset_count": "26",
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
            "STATE-20260809T160000-001.md,release/manifest-v1.schema.json,"
            "scripts/release_manifest.py,tests/scripts/test_release_manifest.py,"
            "examples/CMakeLists.txt,scripts/package-server.py,"
            "tests/scripts/test_server_package.py"
        ),
    },
    "EXPECTED_DEPS": {work: ",".join(deps) for work, deps in WORK_DEPS.items()},
    "HUMAN_WORK_IDS": tuple(WORK_DEPS),
    "RECORD_ANCHORS": ANCHORS,
    "LIFECYCLE_RECORD_MUTATIONS": LIFECYCLE_RECORD_MUTATIONS,
    "PUBLIC_PENDING_MUTATIONS": PUBLIC_PENDING_MUTATIONS,
    "W10_W12_HUMAN_MUTATIONS": W10_W12_HUMAN_MUTATIONS,
    "PRIMARY_ARTIFACT_PROSE_MUTATIONS": PRIMARY_ARTIFACT_PROSE_MUTATIONS,
    "INVENTORY_CONSUMER_METHODS": {
        "PRIMARY_CUDA_SMS": "test_each_primary_cuda_sm_is_required",
        "EXACT_MACHINE_FIELDS": "test_each_exact_machine_field_is_fail_closed",
        "EXPECTED_DEPS": "test_each_work_dependency_edge_is_pinned",
        "HUMAN_WORK_IDS": "test_each_human_work_row_id_occurs_exactly_once",
        "RECORD_ANCHORS": "test_each_required_record_anchor_is_fail_closed",
        "LIFECYCLE_RECORD_MUTATIONS": (
            "test_release_lifecycle_and_honesty_are_fail_closed"
        ),
        "PUBLIC_PENDING_MUTATIONS": "test_public_release_rows_remain_pending",
        "W10_W12_HUMAN_MUTATIONS": (
            "test_human_w12_is_optional_and_cannot_replace_w10"
        ),
        "PRIMARY_ARTIFACT_PROSE_MUTATIONS": (
            "test_human_primary_artifact_contract_matches_machine_block"
        ),
        "GUARD_MAP_KEYS": "test_checker_guard_map_keysets_are_exact",
        "BACKEND_POLICY_PROSE_MUTATIONS": "test_backend_policy_prose_is_fail_closed",
        "PREFLIGHT_WIRING_MUTATIONS": "test_preflight_wiring_mutations_fail",
        "CI_WIRING_MUTATIONS": "test_ci_wiring_mutations_fail",
    },
    "CONSUMER_FLOW_MUTATIONS": ("continue", "break", "wrap_false"),
    "UNKNOWN_MACHINE_FIELD_MUTATIONS": (("unexpected_field", "x"),),
    "HUMAN_WORK_DEPS": {
        work: ",".join(deps) for work, deps in WORK_DEPS.items()
    },
    "GUARD_MAP_KEYS": EXPECTED_GUARD_MAP_KEYS,
    "BACKEND_POLICY_PROSE_MUTATIONS": BACKEND_POLICY_PROSE_MUTATIONS,
    "PREFLIGHT_WIRING_MUTATIONS": PREFLIGHT_WIRING_MUTATIONS,
    "CI_WIRING_MUTATIONS": CI_WIRING_MUTATIONS,
}

TEST_INVENTORY_CONSUMERS = {
    "PRIMARY_CUDA_SMS": (
        "test_each_primary_cuda_sm_is_required",
        ("sm",),
        False,
    ),
    "EXACT_MACHINE_FIELDS": (
        "test_each_exact_machine_field_is_fail_closed",
        ("field", "expected"),
        True,
    ),
    "EXPECTED_DEPS": (
        "test_each_work_dependency_edge_is_pinned",
        ("work", "deps"),
        True,
    ),
    "HUMAN_WORK_IDS": (
        "test_each_human_work_row_id_occurs_exactly_once",
        ("work",),
        False,
    ),
    "RECORD_ANCHORS": (
        "test_each_required_record_anchor_is_fail_closed",
        ("relative", "anchor"),
        True,
    ),
    "LIFECYCLE_RECORD_MUTATIONS": (
        "test_release_lifecycle_and_honesty_are_fail_closed",
        ("relative", "before", "after", "reason"),
        False,
    ),
    "PUBLIC_PENDING_MUTATIONS": (
        "test_public_release_rows_remain_pending",
        ("relative", "before", "after", "reason"),
        False,
    ),
    "W10_W12_HUMAN_MUTATIONS": (
        "test_human_w12_is_optional_and_cannot_replace_w10",
        ("before", "after", "reason"),
        False,
    ),
    "PRIMARY_ARTIFACT_PROSE_MUTATIONS": (
        "test_human_primary_artifact_contract_matches_machine_block",
        ("before", "after", "reason"),
        False,
    ),
    "INVENTORY_CONSUMER_METHODS": (
        "test_each_semantic_inventory_consumer_body_is_pinned",
        ("inventory", "method"),
        True,
    ),
    "CONSUMER_FLOW_MUTATIONS": (
        "test_each_semantic_inventory_consumer_body_is_pinned",
        ("mutation",),
        False,
    ),
    "UNKNOWN_MACHINE_FIELD_MUTATIONS": (
        "test_unknown_machine_fields_are_fail_closed",
        ("field", "value"),
        False,
    ),
    "HUMAN_WORK_DEPS": (
        "test_each_human_work_dependency_is_pinned",
        ("work", "expected"),
        True,
    ),
    "GUARD_MAP_KEYS": (
        "test_checker_guard_map_keysets_are_exact",
        ("guard_map", "keys"),
        True,
    ),
    "BACKEND_POLICY_PROSE_MUTATIONS": (
        "test_backend_policy_prose_is_fail_closed",
        ("before", "after", "reason"),
        False,
    ),
    "PREFLIGHT_WIRING_MUTATIONS": (
        "test_preflight_wiring_mutations_fail",
        ("before", "after", "reason"),
        False,
    ),
    "CI_WIRING_MUTATIONS": (
        "test_ci_wiring_mutations_fail",
        ("before", "after", "reason"),
        False,
    ),
}

TEST_INVENTORY_BODY_DIGESTS = {
    "PRIMARY_CUDA_SMS": "43e348a6fefad920d5ac461ef34868d20c05af64d3c9f032c62af469a358dee9",
    "EXACT_MACHINE_FIELDS": "f7389b004be2b5665456e893abfa8ebb1b404c84711e86aba9673fcf8775c971",
    "EXPECTED_DEPS": "b7d4608bab17632a8a02e7da6f7b8f656415c9ed08dc4ee18f7268709ec91512",
    "HUMAN_WORK_IDS": "49195d0f7cd3f40d48c9f1282e4b9ead9571ca3a546c449c427801cac8fc8bdd",
    "RECORD_ANCHORS": "5d354a9ed8590deccdc62890e403af66a21c416cb45e9788aacc4dce14364500",
    "LIFECYCLE_RECORD_MUTATIONS": "ab35f4e72fe180cd3ef4675939d5ff8c709aff4c0474412d47ee78988c61199d",
    "PUBLIC_PENDING_MUTATIONS": "69a3fc11686ccea3856b61f473796499466dc234e4aca952fce79bef2157714d",
    "W10_W12_HUMAN_MUTATIONS": "17cb0586bf5ea235ba668bd0a4ae90345a33e125f211909e7f97267ec9e59dc8",
    "PRIMARY_ARTIFACT_PROSE_MUTATIONS": "17cb0586bf5ea235ba668bd0a4ae90345a33e125f211909e7f97267ec9e59dc8",
    "GUARD_MAP_KEYS": "701e4821bee926c2e074dbf2b97ff4a93bebb610cc6bed76e06063cab8974758",
    "INVENTORY_CONSUMER_METHODS": "916894a32d88026a883cc1f316d949eb116ee1fced36d635b585d7bf3372b01d",
    "CONSUMER_FLOW_MUTATIONS": "6f69f9e361d38c325fbc455c31ec7211578131368624312e024448afdfc01e83",
    "UNKNOWN_MACHINE_FIELD_MUTATIONS": "b69a6bd26c8417e04994815042ba1520968b906d6bfee4b3413ffc0dafafc5f2",
    "HUMAN_WORK_DEPS": "54a501b903eb3c97023084393666f9f63d289ab9a78e22f389c32bfc1711573b",
    "BACKEND_POLICY_PROSE_MUTATIONS": "c5fea18a668932c4768cb9feb4746fd444b3df7e7ec15df1a588141898d28f2d",
    "PREFLIGHT_WIRING_MUTATIONS": "d442c6d188efd624bffc9e94a7750d6a527c7b693affde5cbc33304f9e95272e",
    "CI_WIRING_MUTATIONS": "7e20ed4d041fee98f96bb751e4435ad266d75ec8fbc9c5e1197a5d21940a6424",
}

EXACT_MACHINE_FIELDS = {
    "lifecycle": "ACTIVE",
    "manifest_schema": "vllm.cpp.release-manifest.v1",
    "delivery_pull_request": "196",
    "delivery_mode": "single-pr-W1-W13",
    "work_W5_status": "implemented",
    "work_W6_status": "implemented",
    "work_W12_policy": "optional-non-blocking",
    "archive_claims": "published-v0.0.2",
    "published_tag": "v0.0.2",
    "published_sha": "7020de93652ca920424a10ac5255b34810dd2f24",
    "published_run": "31466516224",
    "published_asset_count": "26",
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
        "STATE-20260809T160000-001.md,release/manifest-v1.schema.json,"
        "scripts/release_manifest.py,tests/scripts/test_release_manifest.py,"
        "examples/CMakeLists.txt,scripts/package-server.py,"
        "tests/scripts/test_server_package.py"
    ),
}

EXPECTED_FIELDS = {
    "identity": IDENTITY,
    "primary_cuda_artifact": "one-fat-binary-per-os-host-abi",
    "primary_cuda_sms": ",".join(PRIMARY_CUDA_SMS),
    "per_sm_cuda": "optional-non-primary",
    "primary_cpu_artifact": "one-adaptive-binary-per-os-host-abi",
    "x86_64_baseline": "portable-sse2-without-avx2",
    **EXACT_MACHINE_FIELDS,
    **{f"work_{work}": ",".join(deps) for work, deps in WORK_DEPS.items()},
}

WORK_ROW = re.compile(
    r"^\|\s*(W[0-9]+)\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|",
    re.M,
)

STATE_RELEASE_HEADING = (
    "## Outcome"
)
STATE_RELEASE_LIFECYCLE = (
    "The row remains `ACTIVE`. W1-W4 and W7-W13 remain pending"
)


def parse_contract(text: str) -> tuple[dict[str, str], list[str]]:
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        return {}, [
            f"{SPEC_PATH} must contain exactly one machine-readable release "
            f"contract block ({BEGIN} ... {END})"
        ]
    start = text.find(BEGIN) + len(BEGIN)
    end = text.find(END, start)
    if end < start:
        return {}, [f"{SPEC_PATH} has a malformed release contract block"]

    fields: dict[str, str] = {}
    errors: list[str] = []
    for line in text[start:end].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            errors.append(f"release contract line is not key=value: {stripped!r}")
            continue
        key, value = stripped.split("=", 1)
        if key in fields:
            errors.append(f"release contract repeats field {key!r}")
        fields[key] = value
    return fields, errors


def _field_error(key: str, actual: str | None, expected: str) -> str:
    names = {
        "identity": "release spec identity",
        "primary_cuda_artifact": "primary CUDA artifact",
        "primary_cuda_sms": "primary CUDA SM set",
        "per_sm_cuda": "per-SM CUDA policy",
        "primary_cpu_artifact": "primary CPU artifact",
        "x86_64_baseline": "x86_64 baseline",
        "work_W12_policy": "W12 policy",
    }
    for work in WORK_DEPS:
        names[f"work_{work}"] = f"{work} dependencies"
    label = names.get(key, f"release contract field {key}")
    return f"{label} is {actual!r}; expected {expected!r}"


def _normalize_deps(cell: str) -> tuple[str, ...]:
    value = cell.replace("`", "").strip()
    if value in {"", "—", "-", "[]"}:
        return ()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _normalize_prose(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def backend_policy_errors(spec_text: str) -> list[str]:
    """Return drift from the accepted backend release-channel boundaries."""

    normalized = _normalize_prose(spec_text)
    return [
        f"{reason} must remain {statement!r}"
        for reason, statement in BACKEND_POLICY_PROSE.items()
        if _normalize_prose(statement) not in normalized
    ]


def _without_line_comments(text: str) -> str:
    """Remove shell comments while preserving quoted hash characters."""

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


def _yaml_mapping(
    line: str, indent: int, sequence: bool = False
) -> tuple[str, str] | None:
    if _indent(line) != indent:
        return None
    content = line[indent:]
    if sequence:
        if not content.startswith("-"):
            return None
        content = content[1:].lstrip()
        if not content:
            return None
    match = re.match(
        r"^(?P<key>'[^']*'|\"[^\"]*\"|[^:#]+?)\s*:\s*(?P<value>.*)$",
        content,
    )
    if match is None:
        return None
    key = match.group("key").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1]
    return key.strip(), match.group("value").strip()


def _unconditional_ci_run_blocks(text: str) -> list[list[str]]:
    """Return direct run blocks owned by unconditional Actions jobs and steps."""

    lines = text.splitlines()
    blocks: list[list[str]] = []
    jobs_index = next((i for i, line in enumerate(lines) if line == "jobs:"), None)
    if jobs_index is None:
        return blocks
    job_starts = [
        i
        for i in range(jobs_index + 1, len(lines))
        if (mapping := _yaml_mapping(lines[i], 2)) is not None
        and mapping[1] == ""
    ]
    for job_pos, job_start in enumerate(job_starts):
        job_end = (
            job_starts[job_pos + 1]
            if job_pos + 1 < len(job_starts)
            else len(lines)
        )
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
                blocks.append(_literal_block(lines, run[1]))
    return blocks


def _direct_commands(block: list[str]) -> list[list[str]] | None:
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


def _active_ci_commands(ci_text: str) -> set[tuple[str, ...]]:
    commands: set[tuple[str, ...]] = set()
    for block in _unconditional_ci_run_blocks(ci_text):
        parsed = _direct_commands(block)
        if parsed is not None:
            commands.update(tuple(command) for command in parsed)
    return commands


def _ci_has_active_release_step(ci_text: str) -> bool:
    expected = [
        ["python3", "scripts/check-release-binary-contract.py"],
        ["python3", "tests/scripts/test_check_release_binary_contract.py"],
    ]
    return any(
        _direct_commands(block) == expected
        for block in _unconditional_ci_run_blocks(ci_text)
    )


def _bash_array_values(text: str, name: str) -> list[str] | None:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == f"{name}=("]
    if len(starts) != 1:
        return None
    values: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.strip() == ")":
            return values
        try:
            values.extend(shlex.split(line, comments=True, posix=True))
        except ValueError:
            return None
    return None


def _trace_preflight_commands(text: str) -> tuple[int, list[tuple[str, ...]]]:
    """Execute preflight with shims and return every Python argv it owns."""

    with tempfile.TemporaryDirectory(prefix="vllm-release-preflight-trace-") as temporary:
        root = Path(temporary)
        script = root / PREFLIGHT_PATH
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
            "printf '%s\\0' \"$@\" >> \"$VLLM_RELEASE_TRACE\"\n"
            "printf '\\0' >> \"$VLLM_RELEASE_TRACE\"\n",
            encoding="utf-8",
        )
        python.chmod(0o700)
        git = shim_dir / "git"
        git.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        git.chmod(0o700)
        environment = os.environ.copy()
        environment["PATH"] = f"{shim_dir}{os.pathsep}{environment.get('PATH', '')}"
        environment["VLLM_RELEASE_TRACE"] = str(trace)
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
        if record:
            invocations.append(
                tuple(token.decode("utf-8") for token in record.split(b"\0") if token)
            )
    return result.returncode, invocations


def wiring_errors(preflight_text: str, ci_text: str) -> list[str]:
    """Require this checker and suite to execute through preflight and CI."""

    errors: list[str] = []
    uncommented = _without_line_comments(preflight_text)
    checkers = _bash_array_values(uncommented, "CHECKERS")
    suites = _bash_array_values(uncommented, "SUITES")
    if checkers is None or "check-release-binary-contract" not in checkers:
        errors.append("release checker is missing from preflight CHECKERS")
    if suites is None or "test_check_release_binary_contract" not in suites:
        errors.append("release mutation suite is missing from preflight SUITES")
    if suites is None or "test_release_manifest" not in suites:
        errors.append("W5 manifest suite is missing from preflight SUITES")
    if suites is None or "test_release_windows_metadata" not in suites:
        errors.append("W15 Windows metadata suite is missing from preflight SUITES")
    returncode, invocations = _trace_preflight_commands(preflight_text)
    checker_argv = ("scripts/check-release-binary-contract.py",)
    suite_argv = ("tests/scripts/test_check_release_binary_contract.py",)
    manifest_suite_argv = ("tests/scripts/test_release_manifest.py",)
    windows_suite_argv = ("tests/scripts/test_release_windows_metadata.py",)
    if invocations.count(checker_argv) != 1:
        errors.append("preflight does not execute release CHECKERS through its checker loop")
    if invocations.count(suite_argv) != 1:
        errors.append("preflight does not execute release SUITES through its suite loop")
    if invocations.count(manifest_suite_argv) != 1:
        errors.append("preflight does not execute the W5 manifest suite exactly once")
    if invocations.count(windows_suite_argv) != 1:
        errors.append("preflight does not execute the W15 Windows metadata suite exactly once")
    if returncode != 0:
        errors.append(f"instrumented preflight execution failed with rc={returncode}")
    active = _active_ci_commands(ci_text)
    if ("python3", "scripts/check-release-binary-contract.py") not in active:
        errors.append("release checker is missing from the explicit CI checker step")
    if (
        "python3",
        "tests/scripts/test_check_release_binary_contract.py",
    ) not in active:
        errors.append("release mutation suite is missing from the explicit CI step")
    if ("python3", "tests/scripts/test_release_manifest.py") not in active:
        errors.append("W5 manifest suite is missing from an unconditional CI step")
    if ("python3", "tests/scripts/test_release_windows_metadata.py") not in active:
        errors.append("W15 Windows metadata suite is missing from an unconditional CI step")
    if not _ci_has_active_release_step(ci_text):
        errors.append(
            "CI release step must contain checker and suite as direct active commands"
        )
    return errors


def _table_record(
    root: Path,
    relative: str,
    prefix: str,
    cell_count: int,
    label: str,
    errors: list[str],
) -> tuple[str, ...] | None:
    path = root / relative
    if not path.is_file():
        errors.append(f"{label} record {relative} is missing")
        return None
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(rows) != 1:
        errors.append(
            f"{label} record must have exactly one row starting {prefix!r}; "
            f"found {len(rows)}"
        )
        return None
    cells = tuple(cell.strip() for cell in rows[0].split("|")[1:-1])
    if len(cells) != cell_count:
        errors.append(
            f"{label} record must have {cell_count} cells; found {len(cells)}"
        )
        return None
    return cells


def _release_lifecycle_errors(root: Path) -> list[str]:
    errors: list[str] = []

    engine = _table_record(
        root,
        ".agents/engine-matrix.md",
        "| `ENG-RELEASE-BINARIES` |",
        9,
        "engine-matrix release lifecycle",
        errors,
    )
    if engine is not None and (
        engine[7] != "`ACTIVE`"
        or "Required W1-W11/W13 implementation is complete" not in engine[4]
        or "v0.0.2 published eight archive/checksum/provenance triplets" not in engine[5]
        or "Windows v0.0.3-pre.1 extension remain pending" not in engine[5]
    ):
        errors.append(
            "engine-matrix release lifecycle must keep v0.0.2 published and the Windows prerelease pending"
        )

    roadmap = _table_record(
        root,
        ".agents/roadmap_v1.md",
        "| REL | `ROAD-V1-RELEASE` |",
        7,
        "roadmap release lifecycle",
        errors,
    )
    if roadmap is not None and (
        roadmap[5] != "`ACTIVE`"
        or "v0.0.2 published eight primary archive/checksum/provenance triplets" not in roadmap[6]
        or "Windows W14-W16 are implemented for one PR" not in roadmap[6]
        or "publication and 32-asset audit remain pending" not in roadmap[6]
    ):
        errors.append(
            "roadmap release lifecycle must keep v0.0.2 published and Windows hosted publication pending"
        )

    coordination = _table_record(
        root,
        ".agents/coordination.md",
        "| `CLAIM-ENG-RELEASE-BINARIES-W1-W13` |",
        8,
        "coordination release lifecycle",
        errors,
    )
    if coordination is not None and (
        coordination[6] != "`ACTIVE`"
        or "Complete W1-W13 contract in one PR" not in coordination[5]
        or "required W1-W11/W13 implementation complete" not in coordination[7]
        or "tagged publication pending" not in coordination[7]
        or "W12 optional/non-primary" not in coordination[7]
    ):
        errors.append(
            "coordination release lifecycle must keep required implementation ACTIVE while hosted publication remains pending"
        )

    state_path = root / ".agents/completed/state-events/2026-08/STATE-20260809T160000-001.md"
    if not state_path.is_file():
        errors.append(
            "state release lifecycle record "
            ".agents/completed/state-events/2026-08/STATE-20260809T160000-001.md is missing"
        )
    else:
        state_text = state_path.read_text(encoding="utf-8")
        if state_text.count(STATE_RELEASE_HEADING) != 1:
            errors.append(
                "state release lifecycle must have exactly one revised-release "
                "checkpoint"
            )
        else:
            start = state_text.index(STATE_RELEASE_HEADING) + len(
                STATE_RELEASE_HEADING
            )
            end = state_text.find("\n## ", start)
            section = state_text[start:] if end < 0 else state_text[start:end]
            if STATE_RELEASE_LIFECYCLE not in _normalize_prose(section):
                errors.append("state release lifecycle must keep W6 ACTIVE while W1-W4/W7-W13 remain pending")
    return errors


def _top_level_literal(tree: ast.Module, name: str) -> tuple[object, int]:
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(declarations) != 1:
        return None, len(declarations)
    try:
        return ast.literal_eval(declarations[0].value), 1
    except (TypeError, ValueError, SyntaxError):
        return None, 1


def _target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)) and all(
        isinstance(item, ast.Name) for item in target.elts
    ):
        return tuple(item.id for item in target.elts)
    return ()


def _inventory_loop(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    inventory: str,
    target_names: tuple[str, ...],
    mapping_items: bool,
) -> ast.For | None:
    matches: list[ast.For] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.For) or _target_names(node.target) != target_names:
            continue
        if not mapping_items and isinstance(node.iter, ast.Name):
            if node.iter.id == inventory:
                matches.append(node)
        if (
            mapping_items
            and isinstance(node.iter, ast.Call)
            and not node.iter.args
            and not node.iter.keywords
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr == "items"
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == inventory
        ):
            matches.append(node)
    return matches[0] if len(matches) == 1 else None


_VERSION_ONLY_AST_FIELDS = frozenset({"type_params"})


def _canonical_ast(value: object) -> object:
    """Return a Python-version-stable representation of an AST value."""
    if isinstance(value, ast.AST):
        return (
            type(value).__name__,
            tuple(
                (name, _canonical_ast(field_value))
                for name, field_value in ast.iter_fields(value)
                if name not in _VERSION_ONLY_AST_FIELDS
            ),
        )
    if isinstance(value, list):
        return tuple(_canonical_ast(item) for item in value)
    return value


def _consumer_body_digest(loop: ast.For) -> str:
    body = ast.Module(body=loop.body, type_ignores=[])
    serialized = repr(_canonical_ast(body)).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _test_inventory_errors(root: Path) -> list[str]:
    path = root / TEST_PATH
    if not path.is_file():
        return [f"required mutation-test inventory file {TEST_PATH} is missing"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return [f"required mutation-test inventory is not valid Python: {error}"]

    method_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    methods = [node.name for node in method_nodes]
    expected = set(REQUIRED_TEST_METHODS)
    actual = set(methods)
    errors: list[str] = []
    if len(methods) != len(REQUIRED_TEST_METHODS) or actual != expected:
        errors.append(
            "required mutation-test inventory drifted: "
            f"expected {len(REQUIRED_TEST_METHODS)} named methods "
            f"{sorted(expected)}, found {len(methods)} {sorted(methods)}"
        )

    declared, declaration_count = _top_level_literal(tree, "REQUIRED_TEST_METHODS")
    if declaration_count != 1 or declared != REQUIRED_TEST_METHODS:
        errors.append(
            "required mutation-test inventory declaration does not match the "
            "checker's exact count and names"
        )

    guard_maps = (
        (
            "TEST_LITERAL_INVENTORIES",
            EXPECTED_TEST_LITERAL_INVENTORY_KEYS,
            TEST_LITERAL_INVENTORIES,
        ),
        (
            "TEST_INVENTORY_CONSUMERS",
            EXPECTED_TEST_INVENTORY_CONSUMER_KEYS,
            TEST_INVENTORY_CONSUMERS,
        ),
    )
    for name, expected_keys, guard_map in guard_maps:
        if tuple(guard_map) != expected_keys:
            errors.append(
                "semantic mutation guard map keyset drifted for "
                f"{name}: expected {expected_keys!r}, found {tuple(guard_map)!r}"
            )

    for inventory in EXPECTED_TEST_LITERAL_INVENTORY_KEYS:
        if inventory not in TEST_LITERAL_INVENTORIES:
            errors.append(
                "semantic mutation guard map TEST_LITERAL_INVENTORIES is missing "
                f"expected entry {inventory}"
            )
            continue
        expected_value = TEST_LITERAL_INVENTORIES[inventory]
        declared_value, count = _top_level_literal(tree, inventory)
        if count != 1 or declared_value != expected_value:
            errors.append(
                f"semantic mutation inventory {inventory} must be one explicit "
                "top-level literal matching the checker's independent production "
                f"contract; found {declared_value!r}"
            )

    methods_by_name = {node.name: node for node in method_nodes}
    for inventory in EXPECTED_TEST_INVENTORY_CONSUMER_KEYS:
        if inventory not in TEST_INVENTORY_CONSUMERS:
            errors.append(
                "semantic mutation guard map TEST_INVENTORY_CONSUMERS is missing "
                f"expected entry {inventory}"
            )
            continue
        method_name, target_names, mapping_items = TEST_INVENTORY_CONSUMERS[inventory]
        method = methods_by_name.get(method_name)
        loop = (
            None
            if method is None
            else _inventory_loop(method, inventory, target_names, mapping_items)
        )
        if loop is None:
            errors.append(
                f"semantic mutation inventory {inventory} is not consumed by "
                f"{method_name} through its required literal iteration"
            )
            continue
        actual_digest = _consumer_body_digest(loop)
        expected_digest = TEST_INVENTORY_BODY_DIGESTS[inventory]
        if actual_digest != expected_digest:
            errors.append(
                f"semantic mutation inventory {inventory} consumer body drifted; "
                f"expected {expected_digest}, found {actual_digest}"
            )
    return errors


def contract_errors(root: Path) -> list[str]:
    spec = root / SPEC_PATH
    if not spec.is_file():
        return [f"{SPEC_PATH} is missing"]
    text = spec.read_text(encoding="utf-8")
    fields, errors = parse_contract(text)

    missing = set(EXPECTED_FIELDS) - set(fields)
    extra = set(fields) - set(EXPECTED_FIELDS)
    if missing:
        errors.append(f"release contract is missing fields: {sorted(missing)}")
    if extra:
        errors.append(f"release contract has unknown fields: {sorted(extra)}")
    if set(fields) != set(EXPECTED_FIELDS):
        errors.append(
            "release contract schema has missing or unknown fields: "
            f"expected {sorted(EXPECTED_FIELDS)}, found {sorted(fields)}"
        )
    for key, expected in EXPECTED_FIELDS.items():
        if fields.get(key) != expected:
            errors.append(_field_error(key, fields.get(key), expected))

    if "Status: accepted contract with required W1-W11/W13 implementation complete for\n`ENG-RELEASE-BINARIES`" not in text:
        errors.append(
            "release spec identity/status line must name the accepted required implementation state"
        )

    header = "| Work | Deps | Deliverable | Exit gate |"
    if header not in text:
        errors.append("release work table is missing its explicit Deps column")
    parsed_rows = WORK_ROW.findall(text)
    work_counts = Counter(work for work, _, _, _ in parsed_rows)
    expected_work = set(WORK_DEPS)
    missing_work = sorted(expected_work - set(work_counts))
    unexpected_work = sorted(set(work_counts) - expected_work)
    duplicate_work = sorted(work for work, count in work_counts.items() if count > 1)
    if missing_work or unexpected_work or duplicate_work:
        errors.append(
            "release work table IDs must occur exactly once; "
            f"missing={missing_work}, unexpected={unexpected_work}, "
            f"duplicates={duplicate_work}"
        )
    rows = {work: _normalize_deps(deps) for work, deps, _, _ in parsed_rows}
    for work, expected in WORK_DEPS.items():
        if rows.get(work) != expected:
            errors.append(
                f"{work} dependencies in work table are {rows.get(work)!r}; "
                f"expected {expected!r}"
            )
    if rows != WORK_DEPS:
        errors.append(
            "human work-table dependency mirror drifted: "
            f"expected {WORK_DEPS!r}, found {rows!r}"
        )

    work_content = {
        work: (_normalize_prose(deliverable), _normalize_prose(exit_gate))
        for work, _, deliverable, exit_gate in parsed_rows
    }
    for work, (deliverable, exit_gate) in WORK_CONTENT.items():
        actual = work_content.get(work)
        if actual is None or actual[0] != deliverable:
            errors.append(
                f"{work} deliverable is {None if actual is None else actual[0]!r}; "
                f"expected {deliverable!r}"
            )
        if actual is None or actual[1] != exit_gate:
            errors.append(
                f"{work} exit gate is {None if actual is None else actual[1]!r}; "
                f"expected {exit_gate!r}"
            )

    normalized_spec = _normalize_prose(text)
    for label, statement in HUMAN_CONTRACT.items():
        if statement not in normalized_spec:
            errors.append(
                f"{label} must match the accepted machine-readable release contract"
            )
    errors.extend(backend_policy_errors(text))

    for relative, anchor in ANCHORS.items():
        path = root / relative
        if not path.is_file() or anchor not in path.read_text(encoding="utf-8"):
            errors.append(f"{relative} is missing required release anchor {anchor!r}")
    errors.extend(_release_lifecycle_errors(root))
    preflight = root / PREFLIGHT_PATH
    ci = root / CI_PATH
    if not preflight.is_file() or not ci.is_file():
        errors.append("release checker wiring inputs are missing")
    else:
        errors.extend(
            wiring_errors(
                preflight.read_text(encoding="utf-8"),
                ci.read_text(encoding="utf-8"),
            )
        )
    errors.extend(_test_inventory_errors(root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (used by the mutation suite)",
    )
    args = parser.parse_args(argv)
    errors = contract_errors(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Release binary contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
