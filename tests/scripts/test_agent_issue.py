#!/usr/bin/env python3
"""Focused command tests for canonical local issue operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import issue_records as records

SPEC = importlib.util.spec_from_file_location("agent_issue", ROOT / "scripts" / "agent-issue.py")
assert SPEC and SPEC.loader
agent_issue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_issue
SPEC.loader.exec_module(agent_issue)

ULID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
LOCAL_ID = f"ISSUE-LOCAL-{ULID}"


class TrackingFS(agent_issue.LocalFileSystem):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def atomic_write(self, path: Path, text: str, *, old_path: Path | None = None) -> None:
        self.events.append(f"local:{path.name}")
        super().atomic_write(path, text, old_path=old_path)


class FailingWriteFS(TrackingFS):
    def __init__(self, events: list[str], *, fail_on: int) -> None:
        super().__init__(events)
        self.fail_on = fail_on
        self.writes = 0

    def atomic_write(self, path: Path, text: str, *, old_path: Path | None = None) -> None:
        self.writes += 1
        if self.writes == self.fail_on:
            raise OSError("fixture write failure")
        super().atomic_write(path, text, old_path=old_path)


class FailingPublishFS(TrackingFS):
    def publish_directory(self, source: Path, target: Path) -> None:
        raise OSError("fixture rename failure")


class FakeGitHub:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.issues: dict[int, agent_issue.GitHubIssue] = {}
        self.next_number = 900
        self.fail_create: Exception | None = None
        self.fail_update: Exception | None = None
        self.fail_read: Exception | None = None
        self.mismatch_after_write = False
        self.comments_read = 0
        self.comments_written = 0
        self.pages: dict[str | None, agent_issue.GitHubIssuePage] = {
            None: agent_issue.GitHubIssuePage((), None)
        }
        self.page_cursors: list[str | None] = []


    def read(self, number: int) -> agent_issue.GitHubIssue:
        self.events.append(f"remote:read:{number}")
        if self.fail_read:
            raise self.fail_read
        try:
            return self.issues[number]
        except KeyError as error:
            raise agent_issue.GitHubNotFound(f"GitHub issue #{number} is unavailable") from error

    def list_page(self, cursor: str | None) -> agent_issue.GitHubIssuePage:
        self.page_cursors.append(cursor)
        self.events.append(f"remote:list:{cursor}")
        try:
            return self.pages[cursor]
        except KeyError as error:
            raise agent_issue.GitHubError(f"missing fixture page {cursor}") from error

    def create(self, projection: records.GitHubProjection) -> int:
        self.events.append("remote:create")
        if self.fail_create:
            raise self.fail_create
        number = self.next_number
        self.next_number += 1
        body = projection.body + "remote drift\n" if self.mismatch_after_write else projection.body
        self.issues[number] = agent_issue.GitHubIssue(
            number=number,
            title=projection.title,
            body=body,
            state="OPEN",
            created_at="2026-08-31T10:00:00Z",
            updated_at="2026-08-31T10:00:00Z",
            closed_at=None,
        )
        return number

    def update(self, number: int, projection: records.GitHubProjection) -> None:
        self.events.append(f"remote:update:{number}")
        if self.fail_update:
            raise self.fail_update
        old = self.issues[number]
        body = projection.body + "remote drift\n" if self.mismatch_after_write else projection.body
        self.issues[number] = replace(
            old,
            title=projection.title,
            body=body,
            state=projection.state,
            updated_at="2026-08-31T11:00:00Z",
            closed_at="2026-08-31T11:00:00Z" if projection.state == "CLOSED" else None,
        )


def intake_archive_source() -> bytes:
    return (
        "# Issue index\n\nFrozen archive\n"
        "| Issue | Row | Title | Kind |\n"
        "|---:|---|---|---|\n"
        "| [#55](https://github.com/mudler/vllm.cpp/issues/55) "
        "| — | Archived 55 | bug |\n"
    ).encode()


def operations(
    tmp_path: Path,
    *,
    events: list[str] | None = None,
    github: FakeGitHub | None = None,
    clock: Callable[[], date] | None = None,
    owed: records.OwedLookup | None = None,
    filesystem: agent_issue.FileSystem | None = None,
):
    event_log = events if events is not None else []
    client = github if github is not None else FakeGitHub(event_log)
    root = tmp_path / ".agents" / "issues"
    return agent_issue.IssueOperations(
        issues_root=root,
        filesystem=filesystem or TrackingFS(event_log),
        clock=clock or (lambda: date(2026, 8, 31)),
        ulid=lambda: ULID,
        github=client,
        rows={"ROW-A", "ROW-B"},
        owed=owed if owed is not None else (lambda _issue_id: 0),
        frozen_archive=intake_archive_source(),
    ), client, event_log


def seed(op: agent_issue.IssueOperations, record: records.IssueRecord) -> Path:
    path = records.issue_path(record, op.issues_root)
    op.filesystem.atomic_write(path, records.render_issue_record(record))
    return path


def open_record(**changes: object) -> records.IssueRecord:
    values: dict[str, object] = {
        "id": LOCAL_ID,
        "title": "Canonical issue",
        "row": "ROW-A",
        "state": "OPEN",
        "kind": "BUG",
        "github": None,
        "mirror": "PENDING",
        "availability": "FULL",
        "created": "2026-08-01",
        "updated": "2026-08-01",
        "closed": "-",
        "problem": "Reproduction and evidence.",
        "resolution": "-",
    }
    values.update(changes)
    return records.IssueRecord(**values)


def test_create_allocates_stable_local_id_and_validates_before_atomic_write(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path, owed=lambda _issue_id: 1)

    result = op.create(title="New issue", kind="BUG", problem="Observed failure.")

    assert result.path == op.issues_root / "_owed" / f"{LOCAL_ID}.md"
    assert result.record.id == LOCAL_ID
    assert result.record.created == "2026-08-31"
    assert result.record.updated == "2026-08-31"
    assert result.record.mirror == "PENDING"
    assert records.parse_issue_file(result.path) == result.record
    assert events == [f"local:{LOCAL_ID}.md"]
    assert github.issues == {}


def test_create_rejects_invalid_row_before_any_write(tmp_path: Path) -> None:
    op, _github, events = operations(tmp_path)

    with pytest.raises(agent_issue.IssueOperationError, match="canonical and claimable"):
        op.create(title="New issue", kind="BUG", problem="Evidence.", row="NO-SUCH-ROW")

    assert events == []


def test_update_moves_rows_without_changing_identity_or_created(tmp_path: Path) -> None:
    op, _github, events = operations(tmp_path)
    original = open_record(github=42, mirror="SYNCED")
    source = seed(op, original)
    events.clear()

    result = op.update(source, row="ROW-B", title="Updated canonical issue")

    assert result.path == op.issues_root / "ROW-B" / f"{LOCAL_ID}.md"
    assert not source.exists()
    assert result.record.id == LOCAL_ID
    assert result.record.created == "2026-08-01"
    assert result.record.updated == "2026-08-31"
    assert result.record.mirror == "DIVERGED"
    assert result.record.title == "Updated canonical issue"
    assert events == [f"local:{LOCAL_ID}.md"]


def test_update_rejects_direct_identity_and_state_changes(tmp_path: Path) -> None:
    op, _github, events = operations(tmp_path)
    source = seed(op, open_record())
    events.clear()

    with pytest.raises(agent_issue.IssueOperationError, match="update contract.*ID.*State"):
        op.update(source, id=f"ISSUE-GH-{''}9", state="CLOSED")

    assert records.parse_issue_file(source) == open_record()
    assert events == []


def test_update_to_owed_requires_an_explicit_ownership_reference(tmp_path: Path) -> None:
    op, _github, events = operations(tmp_path)
    source = seed(op, open_record())
    events.clear()
    op.owed = lambda _issue_id: 0

    with pytest.raises(agent_issue.IssueOperationError, match="exactly one owning spec"):
        op.update(source, row=None)

    result = op.update(source, row=None, owed_reference=True)
    assert result.path.parent.name == "_owed"


def test_mirror_creates_remote_only_after_local_validation_and_reads_back_before_synced(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path)
    source = seed(op, open_record())
    events.clear()

    result = op.mirror(source)

    remote = github.issues[900]
    assert remote.title == "Canonical issue"
    assert remote.state == "OPEN"
    assert remote.body == (
        f"Row: ROW-A\n\nLocal-Issue: {LOCAL_ID}\nKind: BUG\n\n"
        "Reproduction and evidence.\n"
    )
    assert result.record.github == 900
    assert result.record.mirror == "SYNCED"
    assert events == ["remote:create", f"local:{LOCAL_ID}.md", "remote:read:900", f"local:{LOCAL_ID}.md"]


def test_mirror_state_rules_cover_failed_create_missing_and_readable_mismatch(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path)
    source = seed(op, open_record())
    events.clear()
    github.fail_create = agent_issue.GitHubError("write failed")

    with pytest.raises(agent_issue.IssueOperationError, match="failed to create GitHub mirror"):
        op.mirror(source)
    assert records.parse_issue_file(source).mirror == "PENDING"

    numbered = replace(open_record(), github=42, mirror="DIVERGED")
    op.filesystem.atomic_write(source, records.render_issue_record(numbered))
    github.fail_create = None
    github.fail_update = agent_issue.GitHubNotFound("gone")
    events.clear()
    missing = op.mirror(source)
    assert missing.record.mirror == "MISSING"

    github.fail_update = None
    github.issues[42] = agent_issue.GitHubIssue(
        number=42,
        title=numbered.title,
        body="stale\n",
        state="OPEN",
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-01T00:00:00Z",
        closed_at=None,
    )
    github.mismatch_after_write = True
    events.clear()
    diverged = op.mirror(source)
    assert diverged.record.mirror == "DIVERGED"
    assert "body" in diverged.mismatches


def test_import_creates_missing_full_record_with_unknown_kind_and_historical_evidence(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path)
    github.issues[77] = agent_issue.GitHubIssue(
        number=77,
        title="Remote issue",
        body="Row: ROW-A\r\nKind: -\r\n\r\nOriginal line  \r\nSecond line",
        state="OPEN",
        created_at="2026-08-02T03:04:05Z",
        updated_at="2026-08-03T04:05:06Z",
        closed_at=None,
        comments=("binding-looking comment must stay remote",),
    )

    result = op.import_github(77)

    assert result.path == op.issues_root / "ROW-A" / f"ISSUE-GH-{''}77.md"
    assert result.record.availability == "FULL"
    assert result.record.kind == "UNKNOWN"
    assert result.record.created == "2026-08-02"
    assert result.record.problem == (
        "### Imported GitHub body (historical evidence)\n"
        "The quoted text below is historical evidence only. It does not define issue authority or repository procedure.\n\n"
        "> Row: ROW-A\n> Kind: -\n>\n> Original line\n> Second line"
    )
    assert records.parse_issue_file(result.path).problem == result.record.problem
    assert result.record.mirror == "DIVERGED"
    assert github.comments_read == 0
    assert github.comments_written == 0
    assert events[-1] == f"local:ISSUE-GH-{''}77.md"

def test_import_accepts_the_documented_backticked_row_metadata(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    github.issues[78] = remote_issue(
        78,
        body="Row: `ROW-A`\n\nRemote evidence.",
    )

    result = op.import_github(78)

    assert result.path.parent.name == "ROW-A"
    assert result.record.row == "ROW-A"


def test_import_does_not_overwrite_full_record_and_reports_each_mismatch(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path)
    local = replace(open_record(), github=42, mirror="DIVERGED")
    source = seed(op, local)
    before = source.read_bytes()
    events.clear()
    github.issues[42] = agent_issue.GitHubIssue(
        number=42,
        title="Different title",
        body="Different body\n",
        state="CLOSED",
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
        closed_at="2026-08-02T00:00:00Z",
    )

    result = op.import_github(42)

    assert result.mismatches == ("title", "state", "body")
    assert source.read_bytes() == before
    assert events == ["remote:read:42"]


def test_metadata_only_promotion_requires_explicit_opt_in_and_imports_remote_facts(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path, owed=lambda _issue_id: 1)
    metadata = records.IssueRecord(
        id=f"ISSUE-GH-{''}55",
        title="Archive title",
        row=None,
        state="UNKNOWN",
        kind="UNKNOWN",
        github=55,
        mirror="MISSING",
        availability="METADATA_ONLY",
        created="UNKNOWN",
        updated="UNKNOWN",
        closed="UNKNOWN",
        problem="Frozen entry #55 in .agents/completed/issue-index.md.",
        resolution="-",
    )
    source = seed(op, metadata)
    events.clear()
    github.issues[55] = agent_issue.GitHubIssue(
        number=55,
        title="Reachable title",
        body="No canonical metadata here.",
        state="OPEN",
        created_at="2026-07-01T00:00:00Z",
        updated_at="2026-07-02T00:00:00Z",
        closed_at=None,
    )

    with pytest.raises(agent_issue.IssueOperationError, match="--promote-metadata-only"):
        op.import_github(55)
    assert source.read_bytes() == records.render_issue_record(metadata).encode()

    promoted = op.import_github(55, promote_metadata_only=True)
    assert promoted.promoted is True
    assert promoted.record.availability == "FULL"
    assert promoted.record.state == "OPEN"
    assert promoted.record.kind == "UNKNOWN"
    assert promoted.record.created == "2026-07-01"
    assert promoted.record.updated == "2026-07-02"
    assert "Imported GitHub body (historical evidence)" in promoted.record.problem


def test_close_requires_resolution_and_writes_local_close_before_remote(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path)
    local = replace(open_record(), github=42, mirror="SYNCED")
    source = seed(op, local)
    github.issues[42] = agent_issue.GitHubIssue(
        number=42,
        title=local.title,
        body=records.github_projection(local).body,
        state="OPEN",
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-01T00:00:00Z",
        closed_at=None,
    )
    events.clear()

    with pytest.raises(agent_issue.IssueOperationError, match="resolution evidence"):
        op.close(source, resolution="-")
    assert events == []

    result = op.close(source, resolution="Fixed by focused evidence.")

    assert result.record.state == "CLOSED"
    assert result.record.closed == "2026-08-31"
    assert result.record.updated == "2026-08-31"
    assert result.record.resolution == "Fixed by focused evidence."
    assert result.record.mirror == "SYNCED"
    assert events[0] == f"local:{LOCAL_ID}.md"
    assert events[1:] == ["remote:update:42", "remote:read:42", f"local:{LOCAL_ID}.md"]
    assert "## Resolution\n\nFixed by focused evidence.\n" in github.issues[42].body


@pytest.mark.parametrize("command", ["update", "mirror", "close"])
def test_relative_issue_paths_exclude_the_same_discovered_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    op, _github, events = operations(tmp_path, owed=lambda _issue_id: 0)
    source = seed(op, open_record())
    events.clear()
    monkeypatch.chdir(tmp_path)
    relative = source.relative_to(tmp_path)

    if command == "update":
        result = op.update(relative, title="Changed through a relative path")
    elif command == "mirror":
        result = op.mirror(relative)
    else:
        result = op.close(relative, resolution="Closed through a relative path.")

    assert result.path.exists()
    assert len(tuple(op.filesystem.issue_files(op.issues_root))) == 1


def test_row_assignment_rejects_a_stale_owed_reference(tmp_path: Path) -> None:
    op, _github, events = operations(tmp_path, owed=lambda _issue_id: 1)
    source = seed(op, open_record(row=None))
    events.clear()

    with pytest.raises(agent_issue.IssueOperationError, match="must not retain an owed reference"):
        op.update(source, row="ROW-A")

    assert source.exists()
    assert not (op.issues_root / "ROW-A" / source.name).exists()
    assert events == []


def test_row_assignment_accepts_the_real_owed_reference_removal(tmp_path: Path) -> None:
    owed_ids = {LOCAL_ID}
    op, _github, events = operations(
        tmp_path,
        owed=lambda issue_id: int(issue_id in owed_ids),
    )
    source = seed(op, open_record(row=None))
    events.clear()
    owed_ids.clear()

    result = op.update(source, row="ROW-A")

    assert result.path.parent.name == "ROW-A"
    assert not source.exists()
    assert events == [f"local:{LOCAL_ID}.md"]


def test_row_assignment_accepts_a_validated_prospective_ownership_set(tmp_path: Path) -> None:
    op, _github, events = operations(tmp_path, owed=lambda _issue_id: 1)
    source = seed(op, open_record(row=None))
    events.clear()

    result = op.update(source, row="ROW-A", prospective_owed=())

    assert result.path.parent.name == "ROW-A"
    assert not source.exists()
    assert events == [f"local:{LOCAL_ID}.md"]


def test_row_move_removes_destination_when_source_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filesystem = agent_issue.LocalFileSystem()
    source = tmp_path / "ROW-A" / f"{LOCAL_ID}.md"
    destination = tmp_path / "ROW-B" / source.name
    source.parent.mkdir()
    source.write_text("original\n", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_source_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == source:
            raise PermissionError("source unlink denied")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_unlink)

    with pytest.raises(PermissionError, match="source unlink denied"):
        filesystem.atomic_write(destination, "replacement\n", old_path=source)

    assert source.read_text(encoding="utf-8") == "original\n"
    assert not destination.exists()


def test_github_create_returns_the_number_before_any_closed_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = agent_issue.GhCliClient()
    calls: list[tuple[str, ...]] = []
    closed = replace(
        open_record(),
        state="CLOSED",
        closed="2026-08-31",
        updated="2026-08-31",
        resolution="Resolution evidence.",
    )

    def fake_run(arguments: tuple[str, ...] | list[str], *, stdin: str | None = None) -> str:
        calls.append(tuple(arguments))
        if len(calls) == 1:
            return "https://github.com/example/project/issues/912\n"
        raise agent_issue.GitHubError("remote close failed")

    monkeypatch.setattr(client, "_run", fake_run)

    number = client.create(records.github_projection(closed))

    assert number == 912
    assert len(calls) == 1


def test_closed_mirror_persists_new_number_before_remote_close_and_reuses_it(
    tmp_path: Path,
) -> None:
    op, github, events = operations(tmp_path, owed=lambda _issue_id: 0)
    closed = replace(
        open_record(),
        state="CLOSED",
        closed="2026-08-31",
        updated="2026-08-31",
        resolution="Resolution evidence.",
    )
    source = seed(op, closed)
    events.clear()
    github.fail_update = agent_issue.GitHubError("remote close failed")

    failed_close = op.mirror(source)

    persisted = records.parse_issue_file(source)
    assert failed_close.record.github == 900
    assert failed_close.record.mirror == "DIVERGED"
    assert persisted.github == 900
    assert persisted.mirror == "DIVERGED"
    assert github.issues[900].state == "OPEN"
    assert events == ["remote:create", f"local:{LOCAL_ID}.md", "remote:update:900"]

    github.fail_update = None
    events.clear()
    retried = op.mirror(source)

    assert retried.record.github == 900
    assert retried.record.mirror == "SYNCED"
    assert set(github.issues) == {900}
    assert github.next_number == 901
    assert events == ["remote:update:900", "remote:read:900", f"local:{LOCAL_ID}.md"]


def test_close_reads_one_operation_date_for_updated_and_closed(tmp_path: Path) -> None:
    clock_values = iter((date(2026, 8, 31), date(2026, 9, 1)))
    calls = 0

    def clock() -> date:
        nonlocal calls
        calls += 1
        return next(clock_values)

    op, _github, events = operations(
        tmp_path,
        clock=clock,
        owed=lambda _issue_id: 0,
    )
    source = seed(op, open_record())
    events.clear()

    result = op.close(source, resolution="Resolution evidence.")

    assert result.record.updated == "2026-08-31"
    assert result.record.closed == "2026-08-31"
    assert calls == 1


def test_cli_contract_errors_are_nonzero_and_name_the_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    op, _github, _events = operations(tmp_path)

    status = agent_issue.run_cli(
        ["update", str(tmp_path / "missing.md"), "--state", "CLOSED"],
        operations=op,
    )

    assert status != 0
    assert "update contract" in capsys.readouterr().err


def test_migration_cli_reports_each_spec_reference_rewrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    owner = tmp_path / ".agents" / "specs" / "owner.md"
    rewrite = agent_issue.SpecReferenceRewrite(
        path=owner,
        line=5,
        old="#20",
        new=f"ISSUE-GH-{''}20",
    )

    class StubOperations:
        def migrate_archive(self, *_args, **_kwargs):
            return agent_issue.MigrationResult((), (), (rewrite,))

    status = agent_issue.run_cli(
        ["migrate", "--staging-root", str(tmp_path / "staging")],
        operations=StubOperations(),
    )

    assert status == 0
    output = capsys.readouterr().out
    assert f"{owner}:5" in output
    assert f"#20 -> ISSUE-GH-{''}20" in output



def metadata_only_owed_record() -> records.IssueRecord:
    return records.IssueRecord(
        id=f"ISSUE-GH-{''}55",
        title="Archive title",
        row=None,
        state="UNKNOWN",
        kind="UNKNOWN",
        github=55,
        mirror="MISSING",
        availability="METADATA_ONLY",
        created="UNKNOWN",
        updated="UNKNOWN",
        closed="UNKNOWN",
        problem="Frozen entry #55 in .agents/completed/issue-index.md.",
        resolution="-",
    )


def intake_record(number: int = 55) -> records.IssueRecord:
    archived = (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{number}) "
        f"| — | Archived {number} | bug |"
    )
    return replace(
        metadata_only_owed_record(),
        id=f"ISSUE-GH-{number}",
        github=number,
        problem=(
            "Archive: `.agents/completed/issue-index.md:6`\n\n"
            "### Frozen archive evidence\n\n"
            f"> {archived}"
        ),
    )


def seed_intake(
    op: agent_issue.IssueOperations, record: records.IssueRecord
) -> Path:
    path = records.issue_path(record, op.issues_root, intake=True)
    op.filesystem.atomic_write(path, records.render_issue_record(record))
    return path


def test_intake_update_promotes_atomically_without_changing_identity(
    tmp_path: Path,
) -> None:
    op, _github, events = operations(tmp_path)
    source = seed_intake(op, intake_record())
    events.clear()

    result = op.update(source, row="ROW-A")

    assert result.path == op.issues_root / "ROW-A" / f"ISSUE-GH-{''}55.md"
    assert result.record.id == f"ISSUE-GH-{''}55"
    assert result.path.exists()
    assert not source.exists()
    assert events == [f"local:ISSUE-GH-{''}55.md"]


def test_intake_update_to_owed_requires_exactly_one_prospective_owner(
    tmp_path: Path,
) -> None:
    op, _github, events = operations(tmp_path)
    source = seed_intake(op, intake_record())
    events.clear()

    with pytest.raises(agent_issue.IssueOperationError, match="exactly one"):
        op.update(source, title="Triaged")
    assert source.exists()
    assert events == []

    result = op.update(source, title="Triaged", owed_reference=True)
    assert result.path.parent.name == "_owed"
    assert result.record.id == f"ISSUE-GH-{''}55"
    assert not source.exists()


def test_close_rejects_intake_without_writing_or_contacting_github(
    tmp_path: Path,
) -> None:
    op, _github, events = operations(tmp_path)
    source = seed_intake(op, intake_record())
    events.clear()

    with pytest.raises(agent_issue.IssueOperationError, match="_intake"):
        op.close(source, resolution="Not allowed.")

    assert source.exists()
    assert events == []


def row_owned_remote_issue() -> agent_issue.GitHubIssue:
    return agent_issue.GitHubIssue(
        number=55,
        title="Reachable title",
        body="Row: ROW-A\nKind: BUG\n\nRemote evidence.",
        state="OPEN",
        created_at="2026-07-01T00:00:00Z",
        updated_at="2026-07-02T00:00:00Z",
        closed_at=None,
    )


def test_metadata_only_promotion_uses_source_and_prospective_ownership(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path, owed=lambda issue_id: int(issue_id == f"ISSUE-GH-{''}55"))
    source = seed(op, metadata_only_owed_record())
    events.clear()
    github.issues[55] = row_owned_remote_issue()

    result = op.import_github(55, promote_metadata_only=True, prospective_owed=())

    destination = op.issues_root / "ROW-A" / f"ISSUE-GH-{''}55.md"
    assert result.path == destination
    assert result.promoted is True
    assert result.record.row == "ROW-A"
    assert result.record.availability == "FULL"
    assert destination.exists()
    assert not source.exists()
    assert events == ["remote:read:55", f"local:ISSUE-GH-{''}55.md"]




def test_import_promotes_intake_atomically_and_preserves_id_and_github(
    tmp_path: Path,
) -> None:
    op, github, events = operations(tmp_path)
    source = seed_intake(op, intake_record())
    events.clear()
    github.issues[55] = row_owned_remote_issue()

    result = op.import_github(55, promote_metadata_only=True)

    assert result.promoted is True
    assert (result.record.id, result.record.github) == (f"ISSUE-GH-{''}55", 55)
    assert result.path == op.issues_root / "ROW-A" / f"ISSUE-GH-{''}55.md"
    assert result.path.exists()
    assert not source.exists()
    assert events == ["remote:read:55", f"local:ISSUE-GH-{''}55.md"]
def test_metadata_only_promotion_rejects_stale_owed_reference_after_row_move(tmp_path: Path) -> None:
    op, github, events = operations(tmp_path, owed=lambda issue_id: int(issue_id == f"ISSUE-GH-{''}55"))
    source = seed(op, metadata_only_owed_record())
    events.clear()
    github.issues[55] = row_owned_remote_issue()

    with pytest.raises(agent_issue.IssueOperationError, match="must not retain an owed reference"):
        op.import_github(55, promote_metadata_only=True)

    assert source.exists()
    assert not (op.issues_root / "ROW-A" / source.name).exists()
    assert events == ["remote:read:55"]


def remote_issue(
    number: int,
    *,
    body: str = "Row: ROW-A\n\nRemote evidence.",
    state: str = "OPEN",
    closed_at: str | None = None,
) -> agent_issue.GitHubIssue:
    return agent_issue.GitHubIssue(
        number=number,
        title=f"Issue {number}",
        body=body,
        state=state,
        created_at="2026-07-01T00:00:00Z",
        updated_at="2026-07-02T00:00:00Z",
        closed_at=closed_at,
    )


def frozen_archive(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "archive" / ".agents" / "completed" / "issue-index.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# Issue index\n\n"
        "| Issue | Row | Title | Kind |\n"
        "|---:|---|---|---|\n"
        + "".join(rows),
        encoding="utf-8",
    )
    return path


def archive_row(
    number: int,
    row: str = "`ROW-A`",
    kind: str = "bug",
    *,
    url_number: int | None = None,
) -> str:
    target_number = number if url_number is None else url_number
    return (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{target_number}) "
        f"| {row} | Archived {number} | {kind} |\n"
    )


def owed_spec(tmp_path: Path, name: str, reference: str) -> Path:
    path = tmp_path / ".agents" / "specs" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Spec\n\n## Owed\n\n- {reference}: owner\n\n## Now\n", encoding="utf-8")
    return path


def test_migration_preserves_every_canonical_archive_row_before_other_owners(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(
        tmp_path,
        archive_row(1168, "`ROW-A`"),
        archive_row(1574, "`ROW-B`"),
    )
    owed_spec(tmp_path, "wrong-owner", "#1168")
    github.pages = {
        None: agent_issue.GitHubIssuePage(
            (
                remote_issue(1168, body="Row: -\n\nRemote evidence."),
                remote_issue(1574, body="Remote evidence."),
            ),
            None,
        )
    }

    result = op.migrate_archive(archive, staging_root=tmp_path / "staging", required_full=())

    by_number = {record.github: record for record in result.records}
    assert by_number[1168].row == "ROW-A"
    assert by_number[1574].row == "ROW-B"
    assert result.spec_rewrites == ()


def test_migration_recovers_archive_owner_for_existing_rowless_metadata_record(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(
        tmp_path,
        owed=lambda issue_id: int(issue_id == f"ISSUE-GH-{''}1168"),
    )
    archive = frozen_archive(tmp_path, archive_row(1168, "`ROW-A`"))
    current = replace(
        metadata_only_owed_record(),
        id=f"ISSUE-GH-{''}1168",
        title="Preserved archive title",
        github=1168,
        problem="Frozen #1168 in .agents/completed/issue-index.md.",
    )
    source = seed(op, current)
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(
        archive,
        staging_root=tmp_path / "staging",
        required_full=(),
    )

    assert source.read_bytes() == records.render_issue_record(current).encode()
    assert result.paths == (
        tmp_path
        / "staging"
        / ".agents"
        / "issues"
        / "ROW-A"
        / f"ISSUE-GH-{''}1168.md",
    )
    assert result.records == (replace(current, row="ROW-A"),)
    assert result.spec_rewrites == ()


def test_migration_recovers_unique_spec_owner_for_existing_intake_record(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(55, "—"))
    current = replace(
        intake_record(),
        problem=intake_record().problem.replace(":6`", ":5`"),
    )
    source = seed_intake(op, current)
    owner = owed_spec(
        tmp_path,
        "owner",
        "[#55](https://github.com/mudler/vllm.cpp/issues/55)",
    )
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(
        archive,
        staging_root=tmp_path / "staging",
        required_full=(),
    )

    assert source.read_bytes() == records.render_issue_record(current).encode()
    assert result.paths[0].parent.name == "_owed"
    assert result.records == (current,)
    assert result.spec_rewrites == (
        agent_issue.SpecReferenceRewrite(
            path=owner,
            line=5,
            old="[#55](https://github.com/mudler/vllm.cpp/issues/55)",
            new=f"ISSUE-GH-{''}55",
        ),
    )


def test_migration_rechecks_existing_intake_for_ambiguous_and_absent_owners(
    tmp_path: Path,
) -> None:
    ambiguous = tmp_path / "ambiguous"
    op, github, _events = operations(ambiguous)
    archive = frozen_archive(ambiguous, archive_row(55, "—"))
    current = replace(
        intake_record(),
        problem=intake_record().problem.replace(":6`", ":5`"),
    )
    seed_intake(op, current)
    owed_spec(ambiguous, "one", "#55")
    owed_spec(ambiguous, "two", "issues/55")
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    with pytest.raises(agent_issue.IssueOperationError, match="multiple.*spec owners"):
        op.migrate_archive(
            archive,
            staging_root=ambiguous / "staging",
            required_full=(),
        )

    residue = tmp_path / "residue"
    op, github, _events = operations(residue)
    archive = frozen_archive(residue, archive_row(55, "—"))
    source = seed_intake(op, current)
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(
        archive,
        staging_root=residue / "staging",
        required_full=(),
    )

    assert source.read_bytes() == records.render_issue_record(current).encode()
    assert result.paths[0].parent.name == "_intake"
    assert result.records == (current,)


def test_migration_recovers_one_existing_owed_spec_and_outputs_stable_rewrite(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "—"))
    owner = owed_spec(
        tmp_path,
        "owner",
        "[#20](https://github.com/mudler/vllm.cpp/issues/20)",
    )
    owner_before = owner.read_bytes()
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(archive, staging_root=tmp_path / "staging", required_full=())
    assert owner.read_bytes() == owner_before

    assert result.paths[0].parent.name == "_owed"
    assert result.spec_rewrites == (
        agent_issue.SpecReferenceRewrite(
            path=owner,
            line=5,
            old="[#20](https://github.com/mudler/vllm.cpp/issues/20)",
            new=f"ISSUE-GH-{''}20",
        ),
    )


def test_migration_stops_on_multiple_existing_spec_owners(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "—"))
    owed_spec(tmp_path, "one", "#20")
    owed_spec(tmp_path, "two", "https://github.com/mudler/vllm.cpp/issues/20")
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    with pytest.raises(agent_issue.IssueOperationError, match="multiple.*spec owners"):
        op.migrate_archive(archive, staging_root=tmp_path / "staging", required_full=())

    assert not (tmp_path / "staging").exists()

def test_migration_does_not_treat_a_longer_issue_url_as_the_same_owner(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "—"))
    owner = owed_spec(
        tmp_path,
        "owner",
        "https://github.com/mudler/vllm.cpp/issues/20",
    )
    owed_spec(
        tmp_path,
        "different-issue",
        "https://github.com/mudler/vllm.cpp/issues/200",
    )
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(
        archive,
        staging_root=tmp_path / "staging",
        required_full=(),
    )

    assert result.paths[0].parent.name == "_owed"
    assert [rewrite.path for rewrite in result.spec_rewrites] == [owner]


def test_migration_uses_intake_only_for_true_ownerless_metadata_residue(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "—"))
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(archive, staging_root=tmp_path / "staging", required_full=())

    assert result.paths[0].parent.name == "_intake"
    assert result.records[0].problem == (
        "Archive: `.agents/completed/issue-index.md:5`\n\n"
        "### Frozen archive evidence\n\n"
        "> | [#20](https://github.com/mudler/vllm.cpp/issues/20) "
        "| — | Archived 20 | bug |"
    )

def test_migration_uses_one_stable_owner_over_legacy_number_mentions(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "—"))
    owner = owed_spec(tmp_path, "owner", f"ISSUE-GH-{''}20.")
    owed_spec(tmp_path, "legacy-evidence", "#20")
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(
        archive,
        staging_root=tmp_path / "staging",
        required_full=(),
    )

    assert result.paths[0].parent.name == "_owed"
    assert result.spec_rewrites == ()
    assert owner in agent_issue._spec_owners_and_rewrites(
        tmp_path / ".agents" / "specs",
        20,
    )[0]

def test_migration_assigns_an_evidence_resolved_row_to_a_rowless_full_record(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "—"))
    github.pages = {
        None: agent_issue.GitHubIssuePage(
            (remote_issue(20, body="Remote evidence without row metadata."),),
            None,
        )
    }

    result = op.migrate_archive(
        archive,
        staging_root=tmp_path / "staging",
        owner_resolutions={20: "ROW-A"},
        required_full=(20,),
    )

    assert result.paths[0].parent.name == "ROW-A"
    assert result.records[0].row == "ROW-A"


def test_migration_reconciles_a_stale_archived_row_from_evidence(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "`REMOVED-ROW`"))
    owed_spec(tmp_path, "one", "#20")
    owed_spec(tmp_path, "two", "issues/20")
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    result = op.migrate_archive(
        archive,
        staging_root=tmp_path / "staging",
        owner_resolutions={20: "ROW-A"},
        required_full=(),
    )

    assert result.paths[0].parent.name == "ROW-A"
    assert result.records[0].row == "ROW-A"

def test_migration_rejects_noncanonical_archived_owner_instead_of_using_intake(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(20, "`REMOVED-ROW`"))
    github.pages = {None: agent_issue.GitHubIssuePage((), None)}

    with pytest.raises(agent_issue.IssueOperationError, match="noncanonical archived row"):
        op.migrate_archive(archive, staging_root=tmp_path / "staging", required_full=())

def test_default_rows_includes_stable_rows_declared_by_specs(
    tmp_path: Path,
) -> None:
    spec = tmp_path / ".agents" / "specs" / "bench.md"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "# Benchmark row\n\nRow: `BENCH-ONLY`\n",
        encoding="utf-8",
    )

    assert "BENCH-ONLY" in agent_issue._default_rows(tmp_path)


def test_migration_rejects_archive_display_and_url_number_mismatch_before_remote_evidence(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(7, url_number=9))
    staging = tmp_path / "staging"

    with pytest.raises(agent_issue.IssueOperationError, match="displayed #7.*URL.*#9"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert github.page_cursors == []
    assert not staging.exists()


def test_migration_unions_paginated_remote_and_archive_numbers_with_exact_missing_fallback(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(
        tmp_path,
        owed=lambda issue_id: issue_id in {f"ISSUE-GH-{''}10", f"ISSUE-GH-{''}20", f"ISSUE-GH-{''}30"},
    )
    archive = frozen_archive(tmp_path, archive_row(10), archive_row(20, "—", "verification"))
    staging = tmp_path / "staging"
    owed_spec(tmp_path, "owner-20", "#20")
    owed_spec(tmp_path, "owner-30", "issues/30")
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(10, body="Kind: feature\r\n\r\nFirst\r\n\r\nLast"),), "page-2"),
        "page-2": agent_issue.GitHubIssuePage((remote_issue(10), remote_issue(30, body="Kind: perf\n\nNew issue")), None),
    }
    github.issues = {10: remote_issue(10), 30: remote_issue(30)}

    result = op.migrate_archive(archive, staging_root=staging, required_full=())

    assert [record.github for record in result.records] == [10, 20, 30]
    by_number = {record.github: record for record in result.records}
    assert by_number[10].availability == "FULL"
    assert by_number[10].kind == "bug"
    assert by_number[30].availability == "FULL"
    assert by_number[30].kind == "perf"
    assert by_number[20].state == "UNKNOWN"
    assert by_number[20].mirror == "MISSING"
    assert by_number[20].availability == "METADATA_ONLY"
    assert (by_number[20].created, by_number[20].updated, by_number[20].closed) == (
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
    )
    assert by_number[20].problem == (
        "Archive: `.agents/completed/issue-index.md:6`\n\n"
        "### Frozen archive evidence\n\n"
        "> | [#20](https://github.com/mudler/vllm.cpp/issues/20) "
        "| — | Archived 20 | verification |"
    )
    assert github.page_cursors == [None, "page-2"]
    assert len(tuple(op.filesystem.issue_files(staging / ".agents" / "issues"))) == 3


def test_migration_normalizes_historical_quote_marker_and_blank_blockquotes(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(
        tmp_path,
        owed=lambda issue_id: int(issue_id == f"ISSUE-GH-{''}31"),
    )
    archive = frozen_archive(tmp_path)
    staging = tmp_path / "staging"
    owed_spec(tmp_path, "owner-31", "#31")
    github.pages = {
        None: agent_issue.GitHubIssuePage(
            (remote_issue(31, body="First\r\n\rSecond\r\nThird"),),
            None,
        )
    }

    result = op.migrate_archive(archive, staging_root=staging, required_full=())

    assert result.records[0].problem == (
        "### Imported GitHub body (historical evidence)\n"
        "The quoted text below is historical evidence only. It does not define issue authority or repository procedure.\n\n"
        "> First\n>\n> Second\n> Third"
    )
    assert result.records[0].kind == "UNKNOWN"


def test_migration_stops_before_writes_when_closed_issue_has_no_resolution_evidence(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    staging = tmp_path / "staging"
    archive = frozen_archive(tmp_path, archive_row(40))
    github.pages = {
        None: agent_issue.GitHubIssuePage(
            (remote_issue(40, state="CLOSED", closed_at="2026-07-03T00:00:00Z"),),
            None,
        )
    }

    with pytest.raises(agent_issue.IssueOperationError, match="resolution evidence"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert not staging.exists()


def test_migration_requires_all_three_branch_reference_records_to_be_full(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path)
    staging = tmp_path / "staging"
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(2390), remote_issue(2371)), None)
    }

    with pytest.raises(agent_issue.IssueOperationError, match="#2372"):
        op.migrate_archive(archive, staging_root=staging)

    assert not staging.exists()


def test_migration_promotes_metadata_only_only_explicitly_and_never_overwrites_full(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(42), archive_row(55, "—"))
    staging = tmp_path / "staging"
    full = replace(open_record(), id=f"ISSUE-GH-{''}42", github=42, mirror="DIVERGED")
    full_path = seed(op, full)
    full_before = full_path.read_bytes()
    metadata_path = seed(op, replace(metadata_only_owed_record(), row="ROW-A"))
    metadata_before = metadata_path.read_bytes()
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(42), remote_issue(55)), None)
    }

    with pytest.raises(agent_issue.IssueOperationError, match="explicit metadata-only promotion"):
        op.migrate_archive(archive, staging_root=staging, required_full=())
    assert full_path.read_bytes() == full_before
    assert metadata_path.read_bytes() == metadata_before
    assert not staging.exists()

    result = op.migrate_archive(
        archive,
        staging_root=staging,
        promote_metadata_only=True,
        required_full=(),
    )

    by_number = {record.github: record for record in result.records}
    assert full_path.read_bytes() == full_before
    assert by_number[55].availability == "FULL"

    staged_issues = staging / ".agents" / "issues"
    assert (staged_issues / "ROW-A" / full_path.name).read_bytes() == full_before
    assert (
        records.parse_issue_file(staged_issues / "ROW-A" / metadata_path.name).availability
        == "FULL"
    )


@pytest.mark.parametrize("owed_count", [0, 2])
def test_migration_rejects_rowless_issue_without_exactly_one_real_owed_owner(
    tmp_path: Path,
    owed_count: int,
) -> None:
    staging = tmp_path / "staging"
    op, github, _events = operations(
        tmp_path,
        owed=lambda issue_id: owed_count if issue_id == f"ISSUE-GH-{''}60" else 0,
    )
    archive = frozen_archive(tmp_path)
    github.pages = {
        None: agent_issue.GitHubIssuePage(
            (remote_issue(60, body="Remote evidence without an owner."),),
            None,
        )
    }

    with pytest.raises(agent_issue.IssueOperationError, match="exactly one owning spec"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert not staging.exists()
    assert not tuple(staging.parent.glob(f".{staging.name}.*"))


def test_migration_validates_every_existing_record_before_replacement_planning(
    tmp_path: Path,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path, archive_row(10))
    staging = tmp_path / "staging"
    invalid_full = replace(
        open_record(),
        id=f"ISSUE-GH-{''}10",
        github=11,
        mirror="DIVERGED",
    )
    source = seed(op, invalid_full)
    source_before = source.read_bytes()
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(10),), None)
    }

    with pytest.raises(
        agent_issue.IssueOperationError,
        match="ISSUE-GH identity and GitHub number must match exactly",
    ):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert source.read_bytes() == source_before
    assert not staging.exists()
    assert not tuple(staging.parent.glob(f".{staging.name}.*"))


def test_migration_removes_temporary_tree_after_a_record_write_fails(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    events: list[str] = []
    filesystem = FailingWriteFS(events, fail_on=2)
    op, github, _events = operations(tmp_path, events=events, filesystem=filesystem)
    archive = frozen_archive(tmp_path)
    github.pages = {
        None: agent_issue.GitHubIssuePage(
            (remote_issue(70), remote_issue(71)),
            None,
        )
    }

    with pytest.raises(agent_issue.IssueOperationError, match="staging write"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert not staging.exists()
    assert not tuple(staging.parent.glob(f".{staging.name}.*"))


def test_migration_removes_temporary_tree_after_publish_rename_fails(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    events: list[str] = []
    filesystem = FailingPublishFS(events)
    op, github, _events = operations(tmp_path, events=events, filesystem=filesystem)
    archive = frozen_archive(tmp_path)
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(72),), None)
    }

    with pytest.raises(agent_issue.IssueOperationError, match="publish"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert not staging.exists()
    assert not tuple(staging.parent.glob(f".{staging.name}.*"))


def test_migration_atomic_publish_rejects_a_racing_empty_directory_without_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path)
    staging = tmp_path / "staging"
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(74),), None)
    }
    original_renameat2 = agent_issue._RENAMEAT2
    assert original_renameat2 is not None
    racing_inode: int | None = None

    def create_target_then_rename(
        old_directory: int,
        old_path: bytes,
        new_directory: int,
        new_path: bytes,
        flags: int,
    ) -> int:
        nonlocal racing_inode
        staging.mkdir()
        racing_inode = staging.stat().st_ino
        return original_renameat2(
            old_directory,
            old_path,
            new_directory,
            new_path,
            flags,
        )

    monkeypatch.setattr(agent_issue, "_RENAMEAT2", create_target_then_rename)

    with pytest.raises(agent_issue.IssueOperationError, match="publish"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert racing_inode is not None
    assert staging.is_dir()
    assert staging.stat().st_ino == racing_inode
    assert not tuple(staging.iterdir())
    assert not tuple(staging.parent.glob(f".{staging.name}.*"))


def test_atomic_publish_rejects_a_dangling_symlink_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("new", encoding="utf-8")
    target = tmp_path / "target"
    dangling_destination = tmp_path / "caller-owned-missing-directory"
    target.symlink_to(dangling_destination, target_is_directory=True)

    with pytest.raises(FileExistsError):
        agent_issue.LocalFileSystem().publish_directory(source, target)

    assert source.is_dir()
    assert target.is_symlink()
    assert target.readlink() == dangling_destination


def test_atomic_publish_fails_safely_when_no_noreplace_primitive_is_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    monkeypatch.setattr(agent_issue.sys, "platform", "unsupported")

    with pytest.raises(OSError) as raised:
        agent_issue.LocalFileSystem().publish_directory(source, target)

    assert raised.value.errno == agent_issue.errno.ENOTSUP
    assert source.is_dir()
    assert not target.exists()


def test_migration_rejects_an_existing_target_without_reading_or_merging_it(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    sentinel = staging / "keep"
    sentinel.write_text("unchanged", encoding="utf-8")
    op, github, _events = operations(tmp_path)
    archive = frozen_archive(tmp_path)
    github.pages = {
        None: agent_issue.GitHubIssuePage((remote_issue(73),), None)
    }

    with pytest.raises(agent_issue.IssueOperationError, match="already exists"):
        op.migrate_archive(archive, staging_root=staging, required_full=())

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert tuple(staging.iterdir()) == (sentinel,)
    assert github.page_cursors == []