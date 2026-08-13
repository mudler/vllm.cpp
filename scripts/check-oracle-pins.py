#!/usr/bin/env python3
"""GATE-ORACLE-PINS (#647) — every oracle is named, pinned, and honest about it.

`AGENTS.md` §"When vLLM has no implementation" admits a secondary oracle only
where vLLM implements nothing, only from a fixed set, and only at a recorded
pin. This checker is what stops that rule decaying into prose:

  * every `.agents/oracles/<id>.md` carries exactly one ```oracle-pin block with
    every required key present and non-empty, and no key nobody defined;
  * exactly one record is `role = primary`, and it is `vllm`;
  * the AGENTS.md admissible-oracle table and the directory name the SAME ids,
    in both directions — so an oracle cannot be admitted in prose without a pin
    file, nor pinned in a file nobody is allowed to use;
  * `gateable = yes` requires a revision that is not `UNPINNED` AND evidence
    that is a path EXISTING IN THIS TREE. An issue number is a promise of a
    measurement, so it is refused here and required in the other direction:
    `gateable = no` must name the issue that owes it, which is what keeps an
    ungateable lane visible debt instead of a mid-campaign discovery.

WHAT IT DOES NOT DO, so nobody cites it for more than it delivers:

  * It does not verify that a pin EXISTS upstream. It is deliberately
    network-free — a gate that fails when GitHub is unreachable fails on the
    wrong thing. A fabricated 40-hex string passes shape and fails review.
  * It does not read the vLLM parity pin out of `upstream-sync.md` and compare
    it to `vllm.md`. Two transcriptions of one value is the drift this registry
    exists to stop; `vllm.md` says so in prose and points there. Teaching the
    checker to sync them would BLESS the duplication, not remove it.
  * The `**Secondary oracle:**` declaration check is OPT-IN by syntax. A spec
    that merely mentions llama.cpp in prose is not scanned, because a checker
    that fired on the word "SGLang" would go red on hundreds of files that are
    doing nothing wrong and would be silenced within a week.

Run with `--self-test` to sweep the FIXTURES corpus below in both directions:
every `bad=True` fixture must be reported and every `bad=False` one must not.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / ".agents/oracles"
AGENTS_MD = ROOT / "AGENTS.md"
DECLARATION_ROOTS = (ROOT / ".agents",)

REQUIRED_KEYS = (
    "id",
    "role",
    "upstream",
    "scope",
    "pin",
    "pin_label",
    "pinned_on",
    "gateable",
    "evidence",
)

PRIMARY_ID = "vllm"

BLOCK = re.compile(r"^```oracle-pin\n(.*?)^```", re.M | re.S)
FIELD = re.compile(r"^([a-z_]+)\s*=\s*(.*)$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISSUE_REF = re.compile(r"^#\d+$")
# The AGENTS.md table between the registry markers; the `id` column is fenced.
REGISTRY_REGION = re.compile(
    r"<!--\s*oracle-registry:begin\s*-->(.*?)<!--\s*oracle-registry:end\s*-->", re.S
)
TABLE_ID = re.compile(r"^\|[^|]*\|\s*`([a-z0-9-]+)`\s*\|")
DECLARATION = re.compile(r"\*\*Secondary oracle:\*\*\s*`([a-z0-9-]+)`")


@dataclass
class Record:
    path: Path
    fields: dict[str, str]


def parse_record(path: Path, text: str, errors: list[str]) -> Record | None:
    """Extract the one oracle-pin block, reporting a missing or duplicated one."""
    blocks = BLOCK.findall(text)
    if len(blocks) != 1:
        errors.append(
            f"{path.name}: expected exactly one ```oracle-pin block, found {len(blocks)}"
        )
        return None
    fields: dict[str, str] = {}
    for line_no, line in enumerate(blocks[0].splitlines(), 1):
        if not line.strip():
            continue
        match = FIELD.match(line)
        if match is None:
            errors.append(f"{path.name}: oracle-pin line {line_no} is not `key = value`: {line!r}")
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in fields:
            errors.append(f"{path.name}: duplicate key {key!r} in oracle-pin block")
        fields[key] = value
    return Record(path=path, fields=fields)


def check_record(record: Record, errors: list[str]) -> None:
    """Field-level rules for one record."""
    name = record.path.name
    fields = record.fields

    for key in REQUIRED_KEYS:
        if key not in fields:
            errors.append(f"{name}: oracle-pin is missing required key {key!r}")
        elif not fields[key]:
            errors.append(f"{name}: oracle-pin key {key!r} is empty")
    for key in fields:
        if key not in REQUIRED_KEYS:
            errors.append(f"{name}: oracle-pin carries unknown key {key!r}")

    stem = record.path.stem
    if fields.get("id") and fields["id"] != stem:
        errors.append(f"{name}: id {fields['id']!r} does not match filename stem {stem!r}")

    role = fields.get("role")
    if role and role not in ("primary", "secondary"):
        errors.append(f"{name}: role {role!r} is neither 'primary' nor 'secondary'")

    pinned_on = fields.get("pinned_on")
    if pinned_on and not ISO_DATE.match(pinned_on):
        errors.append(f"{name}: pinned_on {pinned_on!r} is not an ISO date")

    gateable = fields.get("gateable")
    if gateable and gateable not in ("yes", "no"):
        errors.append(f"{name}: gateable {gateable!r} is neither 'yes' nor 'no'")

    evidence = fields.get("evidence", "")
    if gateable == "yes":
        if fields.get("pin") == "UNPINNED":
            errors.append(f"{name}: gateable = yes but pin is UNPINNED — name the revision")
        if ISSUE_REF.match(evidence):
            errors.append(
                f"{name}: gateable = yes cites issue {evidence} as evidence; "
                "an issue is a promise of a measurement, not one"
            )
        elif evidence and not (ROOT / evidence).exists():
            errors.append(f"{name}: evidence path {evidence!r} does not exist in this tree")
    elif gateable == "no" and evidence and not ISSUE_REF.match(evidence):
        errors.append(
            f"{name}: gateable = no must name the owing issue as `#N`, got {evidence!r}"
        )


def agents_registry_ids(text: str, errors: list[str]) -> list[str]:
    """The `id` column of the admissible-oracle table in AGENTS.md."""
    region = REGISTRY_REGION.search(text)
    if region is None:
        errors.append("AGENTS.md: oracle-registry markers are missing")
        return []
    ids = [m.group(1) for line in region.group(1).splitlines() if (m := TABLE_ID.match(line))]
    if not ids:
        errors.append("AGENTS.md: oracle-registry region names no oracle ids")
    return ids


def check_registry(files: dict[Path, str], agents_ids: list[str]) -> list[str]:
    """Whole-registry rules over `{path: text}` plus the ids AGENTS.md admits."""
    errors: list[str] = []
    records: list[Record] = []
    for path in sorted(files):
        record = parse_record(path, files[path], errors)
        if record is None:
            continue
        check_record(record, errors)
        records.append(record)

    primaries = [r for r in records if r.fields.get("role") == "primary"]
    if len(primaries) != 1:
        errors.append(
            f"registry: expected exactly one role = primary, found {len(primaries)}"
        )
    for primary in primaries:
        if primary.fields.get("id") != PRIMARY_ID:
            errors.append(
                f"{primary.path.name}: the primary oracle is {PRIMARY_ID!r}, "
                f"not {primary.fields.get('id')!r} — a secondary never outranks vLLM"
            )

    registry_ids = {r.path.stem for r in records}
    for oracle_id in sorted(registry_ids - set(agents_ids)):
        errors.append(
            f"{oracle_id}: pinned in .agents/oracles/ but absent from the AGENTS.md table"
        )
    for oracle_id in sorted(set(agents_ids) - registry_ids):
        errors.append(
            f"{oracle_id}: admitted by the AGENTS.md table but has no "
            ".agents/oracles/ record, so it has no pin"
        )
    return errors


def check_declarations(files: dict[Path, str], registry_ids: set[str]) -> list[str]:
    """`**Secondary oracle:** \\`id\\`` must name a registered oracle."""
    errors: list[str] = []
    for path in sorted(files):
        for declared in DECLARATION.findall(files[path]):
            if declared not in registry_ids:
                errors.append(
                    f"{path}: declares secondary oracle `{declared}`, which has no "
                    ".agents/oracles/ record"
                )
    return errors


# Corpus swept by --self-test, in BOTH directions. The bound is honest: it
# covers the rules below and nothing else, so a rule added without a fixture is
# covered only by the unit tests in tests/scripts/test_check_oracle_pins.py.
_GOOD = """```oracle-pin
id = fixture
role = secondary
upstream = https://github.com/example/fixture
scope = a path vLLM does not implement
pin = UNPINNED
pin_label = none
pinned_on = 2026-08-13
gateable = no
evidence = #647
```
"""

FIXTURES: tuple[tuple[str, str, bool], ...] = (
    ("fixture.md", _GOOD, False),
    ("fixture.md", _GOOD.replace("pinned_on = 2026-08-13\n", ""), True),
    ("fixture.md", _GOOD.replace("scope = a path vLLM does not implement", "scope ="), True),
    ("fixture.md", _GOOD.replace("gateable = no", "gateable = someday"), True),
    ("fixture.md", _GOOD.replace("evidence = #647", "evidence = soon"), True),
    ("fixture.md", _GOOD.replace("pinned_on = 2026-08-13", "pinned_on = august"), True),
    ("fixture.md", _GOOD.replace("role = secondary", "role = advisory"), True),
    ("fixture.md", _GOOD + _GOOD, True),
    ("renamed.md", _GOOD, True),
    (
        "fixture.md",
        _GOOD.replace("gateable = no\nevidence = #647", "gateable = yes\nevidence = #647"),
        True,
    ),
    (
        "fixture.md",
        _GOOD.replace(
            "gateable = no\nevidence = #647", "gateable = yes\nevidence = AGENTS.md"
        ),
        True,  # gateable = yes with pin = UNPINNED
    ),
)


def self_test() -> int:
    failures: list[str] = []
    for name, text, bad in FIXTURES:
        errors: list[str] = []
        record = parse_record(Path(name), text, errors)
        if record is not None:
            check_record(record, errors)
        reported = bool(errors)
        if reported != bad:
            verb = "was not reported" if bad else "was reported"
            failures.append(f"fixture {name} {verb}: {errors}")
    for failure in failures:
        print(f"self-test: {failure}", file=sys.stderr)
    if failures:
        print(f"self-test FAILED ({len(failures)} of {len(FIXTURES)})", file=sys.stderr)
        return 1
    print(f"self-test ok ({len(FIXTURES)} fixtures)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="sweep the fixture corpus")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    if not ORACLES.is_dir():
        print(f"{ORACLES.relative_to(ROOT)}: oracle registry directory is missing", file=sys.stderr)
        return 1

    files = {
        path: path.read_text(encoding="utf-8")
        for path in sorted(ORACLES.glob("*.md"))
        if path.name != "README.md"
    }
    if not files:
        print(f"{ORACLES.relative_to(ROOT)}: registry holds no oracle records", file=sys.stderr)
        return 1

    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    errors: list[str] = []
    agents_ids = agents_registry_ids(agents_text, errors)
    errors.extend(check_registry(files, agents_ids))

    registry_ids = {path.stem for path in files}
    declarations = {
        path.relative_to(ROOT): path.read_text(encoding="utf-8")
        for root in DECLARATION_ROOTS
        for path in sorted(root.rglob("*.md"))
    }
    errors.extend(check_declarations(declarations, registry_ids))

    for error in errors:
        print(f"oracle-pins: {error}", file=sys.stderr)
    if errors:
        print(
            f"oracle-pins FAILED ({len(errors)} error(s)) — see AGENTS.md "
            '"When vLLM has no implementation"',
            file=sys.stderr,
        )
        return 1
    print(f"oracle-pins ok ({len(files)} oracles pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
