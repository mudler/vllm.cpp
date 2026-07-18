#!/usr/bin/env python3
"""Validate the canonical .agents tables and lifecycle contracts."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"

MATRICES = {
    "MODEL": (AGENTS / "model-matrix.md", 326),
    "QUANT": (AGENTS / "quantization-matrix.md", 81),
    "KERNEL": (AGENTS / "kernel-matrix.md", 31),
    "BACKEND": (AGENTS / "backend-matrix.md", 53),
}

ENGINE_MATRIX = AGENTS / "engine-matrix.md"
ENGINE_PREFIXES = (
    "ENG",
    "KV",
    "PAR",
    "SAMPLE",
    "TOOLS",
    "SPEC",
    "SERVE",
    "LORA",
    "ATTN",
    "LOAD",
)
ENGINE_ROWS = 106

MATRIX_PATHS = [ENGINE_MATRIX, *(path for path, _ in MATRICES.values())]
REQUIRED = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "docs/BENCHMARKS.md",
    AGENTS / "roadmap_v1.md",
    AGENTS / "coordination.md",
    AGENTS / "feature-matrix.md",
    AGENTS / "specs/model-family-inventory.md",
    AGENTS / "specs/feature-anchor-backfill.md",
    *MATRIX_PATHS,
]

STATES = {
    "INVENTORIED",
    "SPIKE",
    "READY",
    "ACTIVE",
    "GATING",
    "PARTIAL",
    "DONE",
    "BLOCKED",
    "ANCHOR-BACKFILL",
    "BUILD-ONLY",
    "UNTRACED",
}
READY_STATES = {"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"}
EVIDENCED_STATES = {
    "PARTIAL",
    "ANCHOR-BACKFILL",
    "GATING",
    "DONE",
    "BUILD-ONLY",
    "UNTRACED",
}

ALL_PREFIXES = (*ENGINE_PREFIXES, *MATRICES)
ID_RE = re.compile(
    rf"(?:{'|'.join(re.escape(prefix) for prefix in ALL_PREFIXES)})-"
    r"[A-Za-z0-9_.-]+"
)
STATE_RE = re.compile(r"`(" + "|".join(re.escape(state) for state in STATES) + r")`")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CLAIM_RE = re.compile(r"CLAIM-[A-Za-z0-9_.-]+")
LINE_FRAGMENT_RE = re.compile(r"L(\d+)(?:-L?(\d+))?")
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
RAW_LOCAL_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:src|include|tests|examples|cmake|scripts|tools|\.github/workflows)/"
    r"[A-Za-z0-9_./-]+|CMakeLists\.txt):(\d+)(?:-(\d+))?"
)

CODE_ANCHOR_PREFIXES = (
    "src/",
    "include/",
    "examples/",
    "cmake/",
    "scripts/",
    "tools/",
    ".github/workflows/",
)
TEST_ANCHOR_PREFIXES = (
    "tests/",
    "scripts/",
    ".github/workflows/",
)
CODE_ANCHOR_FILES = {"CMakeLists.txt"}
EVIDENCE_ANCHOR_FILES = {
    ".agents/parity-ledger.md",
    ".agents/state.md",
}

SPEC_REQUIREMENTS = {
    "Scope": ("scope",),
    "Upstream chain": ("upstream chain",),
    "Our baseline": ("our baseline",),
    "Port map": ("port map", "port and harness map"),
    "Tests to port": ("tests to port",),
    "Gates": ("gates",),
    "Dependencies": ("dependencies",),
    "Work breakdown": (
        "work breakdown",
        "non overlapping work breakdown",
        "work decomposition",
    ),
    "Risks/decisions": ("risks", "risk /", "risk and", "risks/decisions"),
}


@dataclass(frozen=True)
class ClaimRow:
    path: Path
    line_no: int
    item_id: str
    state: str
    header: tuple[str, ...]
    cells: tuple[str, ...]
    raw: str

    def field(self, name: str) -> str:
        index = field_index(self.header, name)
        return self.cells[index] if index is not None and index < len(self.cells) else ""


def markdown_files() -> list[Path]:
    return [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "docs/BENCHMARKS.md",
        *sorted(AGENTS.rglob("*.md")),
    ]


def split_cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def normalize_header(value: str) -> str:
    value = value.replace("`", "").replace("*", "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def field_index(header: tuple[str, ...], field: str) -> int | None:
    for index, value in enumerate(header):
        if field == "id" and value == "id":
            return index
        if field == "item" and (
            value.startswith("item")
            or value == "encoding"
            or "method scheme" in value
            or "cache dtype mode" in value
            or value == "scope"
            or value == "surface"
        ):
            return index
        if field == "upstream" and "upstream" in value:
            return index
        if field == "code" and ("our code" in value or "local evidence" in value):
            return index
        if field == "tests" and (
            "tests evidence" in value
            or "test evidence" in value
            or "local evidence" in value
            or value == "evidence"
        ):
            return index
        if field == "spec" and "spike" in value:
            return index
        if field == "state" and (value == "state" or value.endswith(" state")):
            return index
        if field == "owner" and "owner" in value:
            return index
    return None


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_claim_rows(path: Path, errors: list[str]) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    header: tuple[str, ...] | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            header = None
            continue
        cells = split_cells(line)
        normalized = tuple(normalize_header(cell) for cell in cells)
        if "id" in normalized and any(value == "state" or value.endswith(" state") for value in normalized):
            header = normalized
            continue
        if is_separator(cells):
            continue
        if header is None or not cells:
            continue

        item_id = cells[0].strip().strip("`")
        if not ID_RE.fullmatch(item_id):
            continue
        if len(cells) != len(header):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} has {len(cells)} cells; "
                f"header has {len(header)}"
            )
            continue
        state_index = field_index(header, "state")
        state_cell = cells[state_index] if state_index is not None else ""
        state_matches = STATE_RE.findall(state_cell)
        if len(state_matches) != 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} must have exactly one canonical state"
            )
            continue
        rows.append(
            ClaimRow(path, line_no, item_id, state_matches[0], header, tuple(cells), line)
        )
    return rows


def check_links(errors: list[str]) -> None:
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path, _, fragment = target.partition("#")
            resolved = (source.parent / target_path).resolve()
            if not resolved.exists():
                errors.append(f"{source.relative_to(ROOT)}: dangling link {raw_target}")
                continue
            line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
            if line_match is None or not resolved.is_file():
                continue
            line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
            start = int(line_match.group(1))
            end = int(line_match.group(2) or start)
            if start < 1 or end < start or end > line_count:
                errors.append(
                    f"{source.relative_to(ROOT)}: out-of-range line anchor {raw_target} "
                    f"(file has {line_count} lines)"
                )


def table_ids(prefix: str, path: Path, errors: list[str]) -> list[str]:
    return [
        row.item_id
        for row in parse_claim_rows(path, errors)
        if row.item_id.startswith(prefix + "-")
    ]


def check_matrices(errors: list[str]) -> tuple[list[ClaimRow], dict[str, ClaimRow]]:
    rows: list[ClaimRow] = []
    for path in MATRIX_PATHS:
        rows.extend(parse_claim_rows(path, errors))

    by_id: dict[str, ClaimRow] = {}
    for row in rows:
        prior = by_id.get(row.item_id)
        if prior is not None:
            errors.append(
                f"{row.path.relative_to(ROOT)}:{row.line_no}: duplicate ID {row.item_id}; "
                f"first at {prior.path.relative_to(ROOT)}:{prior.line_no}"
            )
        else:
            by_id[row.item_id] = row

    for prefix, (path, expected) in MATRICES.items():
        count = sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path)
        if count != expected:
            errors.append(f"{path.relative_to(ROOT)}: {count} {prefix} rows; expected {expected}")

    engine_count = sum(
        any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
        for row in rows
        if row.path == ENGINE_MATRIX
    )
    if engine_count != ENGINE_ROWS:
        errors.append(
            f"{ENGINE_MATRIX.relative_to(ROOT)}: {engine_count} engine rows; expected {ENGINE_ROWS}"
        )
    return rows, by_id


def check_engine_summary(rows: list[ClaimRow], errors: list[str]) -> None:
    lines = ENGINE_MATRIX.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    total: list[str] | None = None
    for line in lines:
        if line.startswith("| Area | Rows |"):
            header = [normalize_header(cell) for cell in split_cells(line)]
        elif header is not None and line.startswith("| **Total** |"):
            total = [cell.replace("*", "").strip() for cell in split_cells(line)]
            break
    if header is None or total is None or len(header) != len(total):
        errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing or malformed lifecycle summary")
        return

    actual_rows = [row for row in rows if row.path == ENGINE_MATRIX]
    expected = {"rows": len(actual_rows)}
    expected.update(
        {normalize_header(state): sum(row.state == state for row in actual_rows) for state in STATES}
    )
    for index, name in enumerate(header[1:], 1):
        if name not in expected:
            continue
        try:
            recorded = int(total[index])
        except ValueError:
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: non-numeric total for {name}")
            continue
        if recorded != expected[name]:
            errors.append(
                f"{ENGINE_MATRIX.relative_to(ROOT)}: summary {name}={recorded}; actual {expected[name]}"
            )


def is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("`").lower()
    return not normalized or normalized in {"-", "none", "unassigned", "open", "leaf open"}


def local_line_anchors(value: str, source: Path) -> list[str]:
    anchors: list[str] = []
    for raw_target in LINK_RE.findall(value):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, _, fragment = target.partition("#")
        line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
        if line_match is None:
            continue
        resolved = (source.parent / target_path).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(line_match.group(1))
        end = int(line_match.group(2) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    for match in RAW_LOCAL_ANCHOR_RE.finditer(value):
        resolved = (ROOT / match.group(1)).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    return anchors


def is_code_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in CODE_ANCHOR_FILES or path.startswith(CODE_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def is_test_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in EVIDENCE_ANCHOR_FILES or path.startswith(TEST_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def ledger_line_anchors(value: str, source: Path) -> list[str]:
    return [
        path
        for path in local_line_anchors(value, source)
        if path == ".agents/parity-ledger.md"
    ]


def commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def local_spec_paths(row: ClaimRow) -> list[Path]:
    paths: list[Path] = []
    spec_root = (AGENTS / "specs").resolve()
    for raw_target in LINK_RE.findall(row.field("spec")):
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (row.path.parent / target).resolve()
        try:
            resolved.relative_to(spec_root)
        except ValueError:
            continue
        if resolved.suffix == ".md" and resolved.is_file():
            paths.append(resolved)
    return paths


def has_substantive_spec_content(lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}\s+", stripped) or stripped.startswith("```"):
            continue
        if stripped.startswith("|"):
            cells = split_cells(stripped)
            if is_separator(cells):
                continue
            next_nonempty = next(
                (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
                "",
            )
            if next_nonempty.startswith("|") and is_separator(split_cells(next_nonempty)):
                continue
            if len(cells) >= 2 and any(not is_placeholder(cell) for cell in cells[1:]):
                return True
            continue
        if not is_placeholder(stripped):
            return True
    return False


def structured_spec_fields(text: str) -> set[str]:
    lines = text.splitlines()
    fields: set[str] = set()
    for index, line in enumerate(lines):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading is not None:
            level = len(line) - len(line.lstrip("#"))
            end = len(lines)
            for candidate in range(index + 1, len(lines)):
                next_heading = re.match(r"^(#{1,6})\s+", lines[candidate])
                if next_heading is not None and len(next_heading.group(1)) <= level:
                    end = candidate
                    break
            if has_substantive_spec_content(lines[index + 1 : end]):
                fields.add(normalize_header(heading.group(1)))
            continue
        if line.startswith("|"):
            cells = split_cells(line)
            next_nonempty = next(
                (
                    candidate.strip()
                    for candidate in lines[index + 1 :]
                    if candidate.strip()
                ),
                "",
            )
            is_header = next_nonempty.startswith("|") and is_separator(
                split_cells(next_nonempty)
            )
            if (
                len(cells) >= 2
                and not is_header
                and not is_separator(cells)
                and any(not is_placeholder(cell) for cell in cells[1:])
            ):
                fields.add(normalize_header(cells[0]))
    return fields


def missing_spec_requirements(text: str) -> list[str]:
    fields = structured_spec_fields(text)
    missing: list[str] = []
    for label, alternatives in SPEC_REQUIREMENTS.items():
        normalized = [normalize_header(alternative) for alternative in alternatives]
        if not any(
            field == alternative or field.startswith(alternative + " ")
            for field in fields
            for alternative in normalized
        ):
            missing.append(label)
    return missing


def check_spec(row: ClaimRow, errors: list[str]) -> None:
    specs = local_spec_paths(row)
    location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
    if not specs:
        errors.append(f"{location}: {row.item_id} {row.state} has no real .agents/specs link")
        return
    token = f"`{row.item_id}`"
    matching = [path for path in specs if token in path.read_text(encoding="utf-8")]
    if not matching:
        errors.append(f"{location}: no linked spec names exact stable token `{row.item_id}`")
        return
    text = matching[0].read_text(encoding="utf-8")
    for label in missing_spec_requirements(text):
        errors.append(
            f"{matching[0].relative_to(ROOT)}: spec for {row.item_id} lacks structured {label}"
        )


def parse_active_claims(errors: list[str]) -> dict[str, set[str]]:
    path = AGENTS / "coordination.md"
    claims: dict[str, set[str]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("| `CLAIM-"):
            continue
        cells = split_cells(line)
        claim_match = CLAIM_RE.search(cells[0])
        if claim_match is None:
            continue
        claim = claim_match.group(0)
        if claim in claims:
            errors.append(f"{path.relative_to(ROOT)}:{line_no}: duplicate active claim {claim}")
            continue
        claims[claim] = set(ID_RE.findall(cells[1])) if len(cells) > 1 else set()
    return claims


def check_row_contracts(
    rows: list[ClaimRow], by_id: dict[str, ClaimRow], errors: list[str]
) -> None:
    active_claims = parse_active_claims(errors)
    required_fields = ("id", "item", "upstream", "code", "tests", "spec", "state", "owner")

    for row in rows:
        location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
        for field in required_fields:
            if field_index(row.header, field) is None:
                errors.append(f"{location}: {row.item_id} table lacks semantic {field} column")

        if is_placeholder(row.field("item")):
            errors.append(f"{location}: {row.item_id} has no item description")
        if is_placeholder(row.field("upstream")):
            errors.append(f"{location}: {row.item_id} has no upstream anchor/target")

        if row.state in EVIDENCED_STATES:
            if not is_code_anchor(row.field("code"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact local code anchor")
            if not is_test_anchor(row.field("tests"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact test/evidence anchor")

        if row.state in READY_STATES:
            check_spec(row, errors)

        if row.state in {"SPIKE", "ACTIVE"}:
            claim_match = CLAIM_RE.search(row.field("owner"))
            if claim_match is None:
                errors.append(f"{location}: {row.state} row {row.item_id} has no CLAIM-* owner")
            else:
                claim = claim_match.group(0)
                if row.item_id not in active_claims.get(claim, set()):
                    errors.append(
                        f"{location}: owner {claim} does not claim active row {row.item_id} in coordination.md"
                    )

        if row.state == "DONE":
            if not ledger_line_anchors(row.field("tests"), row.path):
                errors.append(f"{location}: DONE row {row.item_id} lacks exact parity-ledger link")
            owner = row.field("owner").strip().strip("`")
            if COMMIT_RE.fullmatch(owner) is None:
                errors.append(
                    f"{location}: DONE row {row.item_id} owner is not the hexadecimal closing commit"
                )
            elif not commit_exists(owner):
                errors.append(
                    f"{location}: DONE row {row.item_id} closing commit {owner} does not exist"
                )

    for claim, item_ids in active_claims.items():
        if not item_ids:
            errors.append(f".agents/coordination.md: active claim {claim} has no stable row IDs")
        for item_id in item_ids:
            row = by_id.get(item_id)
            if row is None:
                errors.append(f".agents/coordination.md: {claim} references unknown row {item_id}")
            elif row.state not in {"SPIKE", "ACTIVE"}:
                errors.append(
                    f".agents/coordination.md: {claim} references {item_id} in state {row.state}, not SPIKE/ACTIVE"
                )


def check_model_invariants(errors: list[str]) -> None:
    path = AGENTS / "model-matrix.md"
    rows: list[tuple[list[str], str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `MODEL-"):
            continue
        cells = split_cells(line)
        if len(cells) < 3:
            continue
        aliases = re.findall(r"`([^`]+)`", cells[1])
        targets = [value for value in re.findall(r"`([^`]+)`", cells[2]) if "::" in value]
        if aliases and targets:
            rows.append((aliases, targets[-1]))

    actual = {
        "rows": len(rows),
        "memberships": sum(len(aliases) for aliases, _ in rows),
        "architectures": len({alias for aliases, _ in rows for alias in aliases}),
        "targets": len({target for _, target in rows}),
        "modules": len({target.split("::", 1)[0] for _, target in rows}),
    }
    expected = {
        "rows": 324,
        "memberships": 373,
        "architectures": 356,
        "targets": 310,
        "modules": 261,
    }
    if actual != expected:
        errors.append(f"{path.relative_to(ROOT)}: model inventory {actual}, expected {expected}")


def check_table_shapes(paths: list[Path], errors: list[str]) -> None:
    for path in sorted(set(paths)):
        expected_pipes: int | None = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|"):
                expected_pipes = None
                continue
            pipes = len(re.findall(r"(?<!\\)\|", line))
            if expected_pipes is None:
                expected_pipes = pipes
            elif pipes != expected_pipes:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: table has {pipes} pipes; expected {expected_pipes}"
                )


def check_spec_location(errors: list[str]) -> None:
    misplaced = re.compile(r"(?:spec|scoping|semantics|feasibility|notes)", re.I)
    allowed = {"benchmark-protocol.md"}
    for path in AGENTS.glob("*.md"):
        if path.name not in allowed and misplaced.search(path.stem):
            errors.append(f"{path.relative_to(ROOT)}: feature spec/scoping file belongs in .agents/specs/")


def check_roadmap(by_id: dict[str, ClaimRow], errors: list[str]) -> None:
    path = AGENTS / "roadmap_v1.md"
    expected_blocks = [
        "ROAD-V1-A",
        "ROAD-V1-C1",
        "ROAD-V1-C2",
        "ROAD-V1-C3",
        "ROAD-V1-C4",
        "ROAD-V1-C5",
        "ROAD-V1-C6",
        "ROAD-V1-C7",
        "ROAD-V1-C8",
        "ROAD-V1-C9",
        "ROAD-V1-D1",
        "ROAD-V1-D2",
        "ROAD-V1-D3",
        "ROAD-V1-D4",
        "ROAD-V1-D5",
    ]
    seen: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = split_cells(line)
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        block_id = cells[1].strip().strip("`")
        if not re.fullmatch(r"ROAD-V1-[A-Z0-9]+", block_id):
            continue
        seen.append((int(cells[0]), block_id))
        if STATE_RE.fullmatch(cells[5].strip()) is None:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: portfolio State cell needs exactly one state"
            )
    if seen != list(enumerate(expected_blocks)):
        errors.append(f"{path.relative_to(ROOT)}: portfolio order {seen}; expected {list(enumerate(expected_blocks))}")

    for source in (path, AGENTS / "coordination.md"):
        text = source.read_text(encoding="utf-8")
        for item_id in set(re.findall(r"`(" + ID_RE.pattern + r")`", text)):
            if item_id not in by_id:
                errors.append(f"{source.relative_to(ROOT)}: references unknown stable row {item_id}")


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing canonical record: {path.relative_to(ROOT)}")

    rows: list[ClaimRow] = []
    by_id: dict[str, ClaimRow] = {}
    if not errors:
        check_links(errors)
        rows, by_id = check_matrices(errors)
        check_engine_summary(rows, errors)
        check_row_contracts(rows, by_id, errors)
        check_model_invariants(errors)
        spec_paths = [path for row in rows if row.state in READY_STATES for path in local_spec_paths(row)]
        check_table_shapes(
            [AGENTS / "roadmap_v1.md", AGENTS / "coordination.md", *MATRIX_PATHS, *spec_paths],
            errors,
        )
        check_spec_location(errors)
        check_roadmap(by_id, errors)

    if errors:
        for error in dict.fromkeys(errors):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    counts = [
        "ENGINE="
        + str(
            sum(
                any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
                for row in rows
                if row.path == ENGINE_MATRIX
            )
        )
    ]
    for prefix, (path, _) in MATRICES.items():
        counts.append(
            f"{prefix}="
            + str(sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path))
        )
    print("agent record OK: " + " ".join(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
