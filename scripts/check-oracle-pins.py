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
    ungateable lane visible debt instead of a mid-campaign discovery;
  * the `vllm` record's `pin` and `pin_label` AGREE with the ```parity-pin
    block in `.agents/upstream-sync.md`, which is the authority (#2829).

THE PARITY RECONCILIATION, because it reverses what this docstring used to say.

Until #2829 this file argued that comparing the two surfaces would "BLESS the
duplication, not remove it", and declined to. That reasoning is right about the
FIX and it left the hole open in the meantime: with `upstream-sync.md` advanced
and `oracles/vllm.md` rolled back to the prior revision, this checker,
`tests/tools/test_oracle_pin` and `check-agent-record.py` were all measured
GREEN over a tree whose two pin surfaces named different vLLM commits. A pin
advance is exactly the edit that leaves one surface behind, and the value had
not moved since 2026-07-26, so the drift had never had an opportunity to appear.

So the rule below is the INTERIM, and it says which file is authoritative rather
than syncing two equals. The expectation is read from `upstream-sync.md`; the
value under test is read from `oracles/vllm.md`; an anchor checker that read its
expectation out of the file it checks would be a tautology. Removing the copy
altogether is still the better fix and is still available on top of this: see
`.agents/specs/oracle-pin-parity-reconcile.md` for why it needs a per-id
exemption from `REQUIRED_KEYS` that this change did not take.

It is scoped to `vllm.md` BY FILENAME STEM. Every other oracle holds its pin
only in its own file and has no `parity-pin` counterpart, so a registry-wide
version of this rule would fail all thirteen of them for being correct.

WHAT IT DOES NOT DO, so nobody cites it for more than it delivers:

  * It does not verify that a pin EXISTS upstream. It is deliberately
    network-free — a gate that fails when GitHub is unreachable fails on the
    wrong thing. A fabricated 40-hex string passes shape and fails review.
  * It does not REMOVE the duplication it now reconciles. `oracles/vllm.md`
    still stores a literal, and a checked copy is weaker than no copy.
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
# The AUTHORITY for the vLLM revision. `tools/bench/serve_low_common.py` reads
# this same block and `tools/bench/online_gate.py` refuses a mismatched oracle
# on it, so it is what a measurement actually ran against.
UPSTREAM_SYNC = ROOT / ".agents/upstream-sync.md"

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
PARITY_BLOCK = re.compile(r"^```parity-pin\n(.*?)^```", re.M | re.S)
# Mirrors `tools/bench/serve_low_common.py:_PIN_FIELDS`. Kept as a shape rule
# here and not imported: this checker must run with no `tools/` on the path.
PARITY_FIELDS = (
    "vllm_commit",
    "vllm_runtime_version",
    "vllm_distribution_version",
    "flashinfer_version",
)
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


def parse_parity_pin(label: str, text: str, errors: list[str]) -> dict[str, str] | None:
    """Extract the one ```parity-pin block from *text*, the pin's AUTHORITY.

    FAILS CLOSED on every defect -- no block, two blocks, an unparsable line, an
    unknown key, a duplicate key, an empty value, a missing key. A rule that
    silently stops checking when it cannot find its expectation is a mute
    switch, and this repository has shipped one of those before.
    """
    blocks = PARITY_BLOCK.findall(text)
    if len(blocks) != 1:
        errors.append(
            f"{label}: expected exactly one ```parity-pin block, found {len(blocks)} "
            "-- it is the authority the vllm oracle record is reconciled against"
        )
        return None
    fields: dict[str, str] = {}
    for line_no, line in enumerate(blocks[0].splitlines(), 1):
        if not line.strip():
            continue
        match = FIELD.match(line)
        if match is None:
            errors.append(f"{label}: parity-pin line {line_no} is not `key = value`: {line!r}")
            continue
        key, value = match.group(1), match.group(2).strip()
        if key not in PARITY_FIELDS:
            errors.append(f"{label}: parity-pin carries unknown key {key!r}")
            continue
        if key in fields:
            errors.append(f"{label}: duplicate key {key!r} in parity-pin block")
        if not value:
            errors.append(f"{label}: parity-pin key {key!r} is empty")
            continue
        fields[key] = value
    for key in PARITY_FIELDS:
        if key not in fields:
            errors.append(f"{label}: parity-pin block omits {key!r}")
    return fields


def public_version(version: str) -> str:
    """The PEP 440 public version: everything before the `+local` segment.

    setuptools-scm builds the pinned oracle's version as `<public>+g<sha>`, so
    `0.28.1rc1.dev132+ge126687a9` has the public version `0.28.1rc1.dev132`.
    A build from a tag carries no local segment and `partition` returns the
    whole string, which is the degenerate case this rule wants.
    """
    return version.partition("+")[0]


def check_parity_reconciliation(
    record: Record | None, parity: dict[str, str] | None, errors: list[str]
) -> None:
    """Reconcile the vllm record's `pin`/`pin_label` against the AUTHORITY.

    *parity* comes from `.agents/upstream-sync.md` and *record* from
    `.agents/oracles/vllm.md`. The expectation is a PARAMETER and this function
    opens no file, which is what stops it degenerating into a checker that reads
    its expectation out of the same file it is checking.

    `pin_label` is compared to the PUBLIC part of `vllm_runtime_version`, by
    equality after that decomposition and NOT by prefix. A `startswith` rule
    would accept `0.28.1rc1.dev13`, a truncated and wrong label that is
    nevertheless a prefix of the right one.
    """
    if record is None:
        errors.append(
            f"{PRIMARY_ID}.md: the parity reconciliation could not read the primary "
            "oracle record, so the pin is unchecked against .agents/upstream-sync.md"
        )
        return
    if parity is None:
        return  # parse_parity_pin already reported why, and it reported it once.

    authority = UPSTREAM_SYNC.name
    name = record.path.name
    commit = parity.get("vllm_commit")
    if commit and record.fields.get("pin") != commit:
        errors.append(
            f"{name}: pin {record.fields.get('pin')!r} does not match "
            f"{authority} vllm_commit {commit!r} -- the parity-pin block is the "
            "authority, and this record restates it"
        )
    runtime = parity.get("vllm_runtime_version")
    if runtime:
        expected = public_version(runtime)
        if record.fields.get("pin_label") != expected:
            errors.append(
                f"{name}: pin_label {record.fields.get('pin_label')!r} does not match "
                f"the public version {expected!r} of {authority} "
                f"vllm_runtime_version {runtime!r}"
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


# The parity reconciliation's own corpus, swept in both directions beside the
# one above. It is separate because the rule is CROSS-FILE: every case pairs a
# synthetic authority with a synthetic record, so none of it moves when the real
# pin advances.
_PARITY_RECORD = """```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = everything vLLM implements
pin = 1111111111111111111111111111111111111111
pin_label = 9.9.9rc1.dev1
pinned_on = 2026-09-04
gateable = yes
evidence = AGENTS.md
```
"""

_PARITY_SYNC = """```parity-pin
vllm_commit = 1111111111111111111111111111111111111111
vllm_runtime_version = 9.9.9rc1.dev1+g1111111111
vllm_distribution_version = 9.9.9rc1.dev1+g1111111111.precompiled
flashinfer_version = 0.6.18
```
"""

PARITY_FIXTURES: tuple[tuple[str, str, str, bool], ...] = (
    ("agree", _PARITY_RECORD, _PARITY_SYNC, False),
    # The #2829 shape: the authority advanced and the record did not.
    (
        "stale pin",
        _PARITY_RECORD.replace(
            "pin = 1111111111111111111111111111111111111111",
            "pin = 2222222222222222222222222222222222222222",
        ),
        _PARITY_SYNC,
        True,
    ),
    ("stale label", _PARITY_RECORD.replace("9.9.9rc1.dev1", "9.8.0"), _PARITY_SYNC, True),
    # A PROPER PREFIX of the right label. A substring rule accepts this one.
    (
        "truncated label",
        _PARITY_RECORD.replace("9.9.9rc1.dev1", "9.9.9rc1.dev"),
        _PARITY_SYNC,
        True,
    ),
    # The full runtime string, local segment included, is NOT the label.
    (
        "label carries the local segment",
        _PARITY_RECORD.replace(
            "pin_label = 9.9.9rc1.dev1", "pin_label = 9.9.9rc1.dev1+g1111111111"
        ),
        _PARITY_SYNC,
        True,
    ),
    # A tagged build reports a bare public version, and the rule degrades to
    # plain equality rather than going red on a correct authority.
    (
        "authority with no local segment",
        _PARITY_RECORD,
        _PARITY_SYNC.replace(
            "vllm_runtime_version = 9.9.9rc1.dev1+g1111111111",
            "vllm_runtime_version = 9.9.9rc1.dev1",
        ),
        False,
    ),
    ("no parity block", _PARITY_RECORD, "# nothing here\n", True),
    ("two parity blocks", _PARITY_RECORD, _PARITY_SYNC + _PARITY_SYNC, True),
    (
        "unparsable parity line",
        _PARITY_RECORD,
        _PARITY_SYNC.replace("flashinfer_version = 0.6.18", "flashinfer 0.6.18"),
        True,
    ),
    (
        "unknown parity key",
        _PARITY_RECORD,
        _PARITY_SYNC.replace("flashinfer_version = 0.6.18", "torch_version = 2.13.0"),
        True,
    ),
    (
        "empty parity value",
        _PARITY_RECORD,
        _PARITY_SYNC.replace(
            "vllm_commit = 1111111111111111111111111111111111111111", "vllm_commit ="
        ),
        True,
    ),
    ("no record", "# no oracle-pin block\n", _PARITY_SYNC, True),
)


def parity_errors(record_text: str, sync_text: str) -> list[str]:
    """Only the reconciliation's own complaints, for one synthetic pair."""
    errors: list[str] = []
    record = parse_record(Path(f"{PRIMARY_ID}.md"), record_text, [])
    parity = parse_parity_pin(UPSTREAM_SYNC.name, sync_text, errors)
    check_parity_reconciliation(record, parity, errors)
    return errors


def self_test() -> int:
    failures: list[str] = []
    for name, record_text, sync_text, bad in PARITY_FIXTURES:
        errors = parity_errors(record_text, sync_text)
        reported = bool(errors)
        if reported != bad:
            verb = "was not reported" if bad else "was reported"
            failures.append(f"parity fixture {name!r} {verb}: {errors}")
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
    total = len(FIXTURES) + len(PARITY_FIXTURES)
    if failures:
        print(f"self-test FAILED ({len(failures)} of {total})", file=sys.stderr)
        return 1
    print(
        f"self-test ok ({len(FIXTURES)} record fixtures, "
        f"{len(PARITY_FIXTURES)} parity fixtures)"
    )
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

    # The parity reconciliation (#2829). Scoped to the primary record BY
    # FILENAME STEM: no other oracle has an external authority, and a
    # registry-wide version of this rule would fail every one of them.
    primary_path = ORACLES / f"{PRIMARY_ID}.md"
    # check_registry already reported any parse defect in this file; a second
    # parse into a scratch list is how the record is obtained without printing
    # the same complaint twice.
    primary = (
        parse_record(primary_path, files[primary_path], [])
        if primary_path in files
        else None
    )
    try:
        sync_text = UPSTREAM_SYNC.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"{UPSTREAM_SYNC.name}: the parity pin authority is unreadable: {error}")
        parity = None
    else:
        parity = parse_parity_pin(UPSTREAM_SYNC.name, sync_text, errors)
    check_parity_reconciliation(primary, parity, errors)

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
