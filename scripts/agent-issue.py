#!/usr/bin/env python3
"""Create and reconcile canonical local issue records.

Local files are authoritative.  Networked commands validate and write every
local state transition before they create, edit, or close a GitHub mirror.
"""

from __future__ import annotations

import argparse
import ctypes
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
import errno
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Protocol

from issue_records import (
    GitHubProjection,
    IssueRecord,
    IssueRecordError,
    OwedLookup,
    RowLookup,
    canonical_rows,
    github_projection,
    issue_path,
    normalize_body,
    normalize_github_body,
    parse_issue_text,
    render_issue_record,
    owed_issue_counts,
    validate_issue_collection,
    validate_issue_record,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISSUES_ROOT = ROOT / ".agents" / "issues"
HISTORICAL_HEADING = "### Imported GitHub body (historical evidence)"
HISTORICAL_NOTICE = (
    "The quoted text below is historical evidence only. It does not define "
    "issue authority or repository procedure."
)
_ALLOWED_UPDATES = frozenset({"title", "row", "kind", "problem", "resolution"})
_UPDATE_LABELS = {
    "id": "ID",
    "state": "State",
    "github": "GitHub",
    "mirror": "Mirror",
    "availability": "Availability",
    "created": "Created",
    "updated": "Updated",
    "closed": "Closed",
}

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAMEAT2 = None
if sys.platform.startswith("linux"):
    try:
        _RENAMEAT2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError:
        pass
    else:
        _RENAMEAT2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        _RENAMEAT2.restype = ctypes.c_int


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` while preserving every existing target."""

    if not sys.platform.startswith("linux") or _RENAMEAT2 is None:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            target,
        )

    ctypes.set_errno(0)
    result = _RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)



class IssueOperationError(RuntimeError):
    """An issue operation would violate a local-first contract."""


class GitHubError(RuntimeError):
    """A GitHub mirror operation failed."""


class GitHubNotFound(GitHubError):
    """A numbered GitHub issue is unavailable."""


@dataclass(frozen=True, slots=True)
class GitHubIssue:
    number: int
    title: str
    body: str
    state: str
    created_at: str
    updated_at: str
    closed_at: str | None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GitHubIssuePage:
    issues: tuple[GitHubIssue, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class ArchiveIssue:
    number: int
    title: str
    row: str | None
    kind: str
    citation: str


@dataclass(frozen=True, slots=True)
class SpecReferenceRewrite:
    path: Path
    line: int
    old: str
    new: str


@dataclass(frozen=True, slots=True)
class MigrationResult:
    records: tuple[IssueRecord, ...]
    paths: tuple[Path, ...]
    spec_rewrites: tuple[SpecReferenceRewrite, ...] = ()



@dataclass(frozen=True, slots=True)
class OperationResult:
    path: Path
    record: IssueRecord
    mismatches: tuple[str, ...] = ()
    promoted: bool = False


class FileSystem(Protocol):
    def read_text(self, path: Path) -> str: ...
    def read_bytes(self, path: Path) -> bytes: ...


    def exists(self, path: Path) -> bool: ...

    def issue_files(self, root: Path) -> Iterable[Path]: ...

    def atomic_write(
        self, path: Path, text: str, *, old_path: Path | None = None
    ) -> None: ...

    def temporary_sibling(self, target: Path) -> Path: ...

    def publish_directory(self, source: Path, target: Path) -> None: ...

    def remove_tree(self, path: Path) -> None: ...


def _normalized_path(path: str | Path) -> Path:
    """Return one absolute lexical path for ownership and identity comparisons."""

    return Path(os.path.abspath(os.fspath(path)))


class LocalFileSystem:
    """The real filesystem boundary, injectable in focused command tests."""

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def exists(self, path: Path) -> bool:
        return path.exists()

    def issue_files(self, root: Path) -> Iterable[Path]:
        return sorted(root.glob("**/*.md")) if root.exists() else ()

    def atomic_write(
        self, path: Path, text: str, *, old_path: Path | None = None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        moving = (
            old_path is not None
            and _normalized_path(old_path) != _normalized_path(path)
        )
        destination_written = False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            destination_written = True
            if moving and old_path is not None and old_path.exists():
                old_path.unlink()
        except BaseException:
            if destination_written and moving:
                path.unlink(missing_ok=True)
            else:
                temporary_path.unlink(missing_ok=True)
            raise

    def temporary_sibling(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
        )

    def publish_directory(self, source: Path, target: Path) -> None:
        _rename_noreplace(source, target)

    def remove_tree(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)


class GitHubClient(Protocol):
    def read(self, number: int) -> GitHubIssue: ...

    def list_page(self, cursor: str | None) -> GitHubIssuePage: ...

    def create(self, projection: GitHubProjection) -> int: ...

    def update(self, number: int, projection: GitHubProjection) -> None: ...


class GhCliClient:
    """Explicit GitHub boundary used only by networked CLI commands."""

    def _run(self, arguments: Sequence[str], *, stdin: str | None = None) -> str:
        try:
            process = subprocess.run(
                ["gh", *arguments],
                input=stdin,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise GitHubError(f"could not execute gh: {error}") from error
        if process.returncode:
            message = process.stderr.strip() or process.stdout.strip() or "gh command failed"
            if re.search(r"not found|could not resolve|HTTP 404", message, re.IGNORECASE):
                raise GitHubNotFound(message)
            raise GitHubError(message)
        return process.stdout

    def read(self, number: int) -> GitHubIssue:
        output = self._run(
            [
                "issue",
                "view",
                str(number),
                "--json",
                "number,title,body,state,createdAt,updatedAt,closedAt",
            ]
        )
        try:
            value = json.loads(output)
            return GitHubIssue(
                number=int(value["number"]),
                title=str(value["title"]),
                body=str(value.get("body") or ""),
                state=str(value["state"]).upper(),
                created_at=str(value["createdAt"]),
                updated_at=str(value["updatedAt"]),
                closed_at=str(value["closedAt"]) if value.get("closedAt") else None,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise GitHubError(f"GitHub issue #{number} returned an invalid read-back") from error

    def list_page(self, cursor: str | None) -> GitHubIssuePage:
        query = (
            "query($owner:String!,$name:String!,$cursor:String){"
            "repository(owner:$owner,name:$name){"
            "issues(first:100,after:$cursor,orderBy:{field:CREATED_AT,direction:ASC}){"
            "nodes{number title body state createdAt updatedAt closedAt}"
            "pageInfo{hasNextPage endCursor}}}}"
        )
        arguments = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            "owner=mudler",
            "-F",
            "name=vllm.cpp",
        ]
        if cursor is not None:
            arguments.extend(("-F", f"cursor={cursor}"))
        output = self._run(arguments)
        try:
            connection = json.loads(output)["data"]["repository"]["issues"]
            values = connection["nodes"]
            page_info = connection["pageInfo"]
            issues = tuple(
                GitHubIssue(
                    number=int(value["number"]),
                    title=str(value["title"]),
                    body=str(value.get("body") or ""),
                    state=str(value["state"]).upper(),
                    created_at=str(value["createdAt"]),
                    updated_at=str(value["updatedAt"]),
                    closed_at=(
                        str(value["closedAt"]) if value.get("closedAt") else None
                    ),
                )
                for value in values
            )
            has_next = bool(page_info["hasNextPage"])
            next_cursor = page_info.get("endCursor") if has_next else None
            if has_next and not isinstance(next_cursor, str):
                raise ValueError("missing pagination cursor")
            return GitHubIssuePage(issues, next_cursor)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise GitHubError("GitHub issue enumeration returned an invalid page") from error

    def create(self, projection: GitHubProjection) -> int:
        output = self._run(
            ["issue", "create", "--title", projection.title, "--body-file", "-"],
            stdin=projection.body,
        )
        match = re.search(r"/issues/([1-9][0-9]*)\s*$", output)
        if not match:
            raise GitHubError("GitHub create did not return an issue number")
        return int(match.group(1))

    def update(self, number: int, projection: GitHubProjection) -> None:
        self._run(
            [
                "issue",
                "edit",
                str(number),
                "--title",
                projection.title,
                "--body-file",
                "-",
            ],
            stdin=projection.body,
        )
        command = "close" if projection.state == "CLOSED" else "reopen"
        self._run(["issue", command, str(number)])


def _default_ulid() -> str:
    """Generate one standards-shaped ULID without an external dependency."""

    milliseconds = int(__import__("time").time_ns() // 1_000_000)
    value = (milliseconds << 80) | int.from_bytes(secrets.token_bytes(10), "big")
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    encoded = ["0"] * 26
    for index in range(25, -1, -1):
        encoded[index] = alphabet[value & 31]
        value >>= 5
    return "".join(encoded)


def _remote_date(value: str | None, field: str) -> str:
    if not value or len(value) < 10:
        raise IssueOperationError(f"import contract requires GitHub {field} date")
    candidate = value[:10]
    try:
        date.fromisoformat(candidate)
    except ValueError as error:
        raise IssueOperationError(f"import contract requires a valid GitHub {field} date") from error
    return candidate


def _remote_state(value: str) -> str:
    state = value.upper()
    if state not in {"OPEN", "CLOSED"}:
        raise IssueOperationError("import contract requires GitHub state OPEN or CLOSED")
    return state


def _metadata_line(body: str, name: str) -> str | None:
    prefix = f"{name}: "
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
                value = value[1:-1]
            return value or None
    return None


def _historical_problem(body: str) -> str:
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    quoted = [f"> {line}" if line else ">" for line in normalized.split("\n")]
    return normalize_body(
        f"{HISTORICAL_HEADING}\n{HISTORICAL_NOTICE}\n\n" + "\n".join(quoted)
    )


def _resolution_from_remote(body: str) -> str | None:
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    marker = "\n## Resolution\n"
    if marker not in f"\n{normalized}":
        return None
    resolution = normalize_body((f"\n{normalized}").split(marker, 1)[1])
    return resolution if resolution not in {"", "-"} else None


def _compare(projection: GitHubProjection, remote: GitHubIssue) -> tuple[str, ...]:
    mismatches: list[str] = []
    if projection.title != remote.title:
        mismatches.append("title")
    if projection.state != remote.state.upper():
        mismatches.append("state")
    if projection.body != normalize_github_body(remote.body):
        mismatches.append("body")
    return tuple(mismatches)


def _archive_issues(text: str) -> dict[int, ArchiveIssue]:
    """Parse and deduplicate the frozen Markdown table by GitHub number."""

    found: dict[int, ArchiveIssue] = {}
    for line_number, line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").splitlines(),
        start=1,
    ):
        number_match = re.match(
            r"^\|\s*\[#([1-9][0-9]*)\]\(([^)]+)\)\s*\|",
            line,
        )
        if number_match is None:
            continue
        displayed_number = int(number_match.group(1))
        url = number_match.group(2)
        url_number_match = re.search(
            r"/issues/([1-9][0-9]*)(?:[/?#][^)\s]*)?\Z",
            url,
        )
        if url_number_match is None:
            raise IssueOperationError(
                f"migration archive row {line_number} issue URL is malformed"
            )
        url_number = int(url_number_match.group(1))
        if displayed_number != url_number:
            raise IssueOperationError(
                f"migration archive row {line_number} displayed "
                f"#{displayed_number} but its URL identifies #{url_number}"
            )
        cells = line[1:-1].split("|") if line.endswith("|") else line[1:].split("|")
        if len(cells) < 4:
            raise IssueOperationError(
                f"migration archive row {line_number} is malformed"
            )
        number = displayed_number
        row_value = cells[1].strip().strip("`")
        row = None if row_value in {"", "-", "—"} else row_value
        title = "|".join(cells[2:-1]).strip().replace("\\|", "|")
        kind_value = cells[-1].strip()
        kind = kind_value if kind_value not in {"", "-", "—"} else "UNKNOWN"
        found.setdefault(
            number,
            ArchiveIssue(
                number=number,
                title=title,
                row=row,
                kind=kind,
                citation=(
                    f"Archive: `.agents/completed/issue-index.md:{line_number}`\n\n"
                    "### Frozen archive evidence\n\n"
                    f"> {line}"
                ),
            ),
        )
    return found


def _owed_section_lines(text: str) -> list[tuple[int, str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line == "## Owed":
            start = index + 1
            break
    if start is None:
        return []
    found: list[tuple[int, str]] = []
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            break
        found.append((index + 1, lines[index]))
    return found


def _spec_owners_and_rewrites(
    specs_root: Path,
    number: int,
) -> tuple[tuple[Path, ...], tuple[SpecReferenceRewrite, ...]]:
    issue_id = f"ISSUE-GH-{number}"
    pattern = re.compile(
        rf"(?P<link>\[#{number}\]\(https?://[^)\s]+/issues/{number}(?![0-9])(?:[/?#][^)\s]*)?\))"
        rf"|(?P<stable>(?<![A-Z0-9_-]){re.escape(issue_id)}(?![A-Z0-9_-]))"
        rf"|(?P<url>https?://[^\s)]+/issues/{number}(?![0-9])(?:[/?#][^\s)]*)?)"
        rf"|(?P<number>(?<![A-Za-z0-9])#{number}(?![0-9]))"
        rf"|(?P<path>(?<![A-Za-z0-9])issues/{number}(?![0-9]))"
    )
    stable_owners: list[Path] = []
    legacy_owners: list[Path] = []
    rewrites: list[SpecReferenceRewrite] = []
    for path in sorted(specs_root.glob("*.md")):
        matches: list[tuple[int, re.Match[str]]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise IssueOperationError(
                f"migration contract cannot read owning spec {path}: {error}"
            ) from error
        for line_number, line in _owed_section_lines(text):
            matches.extend((line_number, match) for match in pattern.finditer(line))
        if not matches:
            continue
        if any(match.lastgroup == "stable" for _, match in matches):
            stable_owners.append(path)
            continue
        legacy_owners.append(path)
        for line_number, match in matches:
            rewrites.append(
                SpecReferenceRewrite(
                    path=path,
                    line=line_number,
                    old=match.group(0),
                    new=issue_id,
                )
            )
    if stable_owners:
        return tuple(stable_owners), ()
    return tuple(legacy_owners), tuple(rewrites)


class IssueOperations:
    """Local-first issue operations with all external boundaries injected."""

    def __init__(
        self,
        *,
        issues_root: str | Path,
        filesystem: FileSystem,
        clock: Callable[[], date],
        ulid: Callable[[], str],
        github: GitHubClient,
        rows: RowLookup,
        owed: OwedLookup,
        frozen_archive: bytes | None = None,
    ) -> None:
        self.issues_root = Path(issues_root)
        self.filesystem = filesystem
        self.clock = clock
        self.ulid = ulid
        self.github = github
        self.rows = rows
        self.owed = owed
        self.frozen_archive = frozen_archive

    def _today(self) -> str:
        value = self.clock()
        if not isinstance(value, date):
            raise IssueOperationError("clock contract must return a date")
        return value.isoformat()

    def _row_valid(self, row: str) -> bool:
        return bool(self.rows(row)) if callable(self.rows) else row in self.rows

    def _load(self, path: str | Path) -> IssueRecord:
        candidate = Path(path)
        try:
            return parse_issue_text(self.filesystem.read_text(candidate))
        except (OSError, IssueRecordError) as error:
            raise IssueOperationError(f"local issue contract failed for {candidate}: {error}") from error

    def _records(self, *, excluding: Path | None = None) -> list[IssueRecord]:
        materialized: list[IssueRecord] = []
        excluded = _normalized_path(excluding) if excluding is not None else None
        for path in self.filesystem.issue_files(self.issues_root):
            if excluded is not None and _normalized_path(path) == excluded:
                continue
            try:
                materialized.append(parse_issue_text(self.filesystem.read_text(path)))
            except (OSError, IssueRecordError) as error:
                raise IssueOperationError(f"local issue collection contract failed for {path}: {error}") from error
        return materialized

    def _validate(
        self,
        record: IssueRecord,
        path: Path,
        *,
        replacing_path: Path | None = None,
        owed_reference: bool = False,
        prospective_owed: OwedLookup | None = None,
    ) -> None:
        if prospective_owed is not None:
            owed = prospective_owed
        elif record.row is None and owed_reference:
            owed = lambda issue_id: int(issue_id == record.id)
        else:
            owed = self.owed
        try:
            validate_issue_record(
                record,
                path,
                self.rows,
                owed,
                frozen_archive=self.frozen_archive,
            )
            validate_issue_collection([*self._records(excluding=replacing_path), record])
        except IssueRecordError as error:
            raise IssueOperationError(f"local issue contract violated: {error}") from error

    def _write(
        self,
        record: IssueRecord,
        path: Path,
        *,
        old_path: Path | None = None,
        owed_reference: bool = False,
        prospective_owed: OwedLookup | None = None,
    ) -> OperationResult:
        self._validate(
            record,
            path,
            replacing_path=old_path,
            owed_reference=owed_reference,
            prospective_owed=prospective_owed,
        )
        try:
            self.filesystem.atomic_write(path, render_issue_record(record), old_path=old_path)
        except OSError as error:
            raise IssueOperationError(f"atomic local write failed for {path}: {error}") from error
        return OperationResult(path=path, record=record)

    def _validate_move_source(
        self,
        current: IssueRecord,
        source: Path,
        target_row: str | None,
        prospective_owed: OwedLookup | None,
    ) -> OwedLookup | None:
        """Validate pre-move ownership and select post-move owed ownership."""

        moving_from_owed = source.parent.name == "_owed" and target_row is not None
        self._validate(
            current,
            source,
            replacing_path=source,
            owed_reference=moving_from_owed,
        )
        return prospective_owed if moving_from_owed else None

    def create(
        self,
        *,
        title: str,
        kind: str,
        problem: str,
        row: str | None = None,
        resolution: str = "-",
        owed_reference: bool = False,
    ) -> OperationResult:
        suffix = self.ulid()
        issue_id = suffix if suffix.startswith("ISSUE-LOCAL-") else f"ISSUE-LOCAL-{suffix}"
        today = self._today()
        record = IssueRecord(
            id=issue_id,
            title=title,
            row=row,
            state="OPEN",
            kind=kind,
            github=None,
            mirror="PENDING",
            availability="FULL",
            created=today,
            updated=today,
            closed="-",
            problem=problem,
            resolution=resolution,
        )
        path = issue_path(record, self.issues_root)
        if self.filesystem.exists(path):
            raise IssueOperationError(f"identity contract violated: {issue_id} already exists")
        return self._write(record, path, owed_reference=owed_reference)

    def update(
        self,
        path: str | Path,
        *,
        owed_reference: bool = False,
        prospective_owed: OwedLookup | None = None,
        **changes: object,
    ) -> OperationResult:
        forbidden = sorted(set(changes) - _ALLOWED_UPDATES)
        if forbidden:
            labels = [_UPDATE_LABELS.get(name, name) for name in forbidden]
            raise IssueOperationError(
                "update contract permits only Title, Row, Kind, Problem, and Resolution; "
                "direct changes to " + ", ".join(labels) + " are forbidden"
            )
        if not changes:
            raise IssueOperationError("update contract requires at least one canonical change")
        source = Path(path)
        current = self._load(source)
        values = {name: changes.get(name, getattr(current, name)) for name in _ALLOWED_UPDATES}
        row = values["row"]
        if row == "-":
            row = None
        candidate_row = row if row is None else str(row)
        candidate_owed = self._validate_move_source(
            current,
            source,
            candidate_row,
            prospective_owed,
        )
        github = current.github
        metadata_only = current.availability == "METADATA_ONLY"
        candidate = replace(
            current,
            title=str(values["title"]),
            row=candidate_row,
            kind=str(values["kind"]),
            problem=str(values["problem"]),
            resolution=str(values["resolution"]),
            mirror=(
                current.mirror
                if metadata_only
                else ("DIVERGED" if github is not None else "PENDING")
            ),
            updated=current.updated if metadata_only else self._today(),
        )
        target = issue_path(candidate, self.issues_root)
        return self._write(
            candidate,
            target,
            old_path=source,
            owed_reference=owed_reference,
            prospective_owed=candidate_owed,
        )

    def _record_mirror_state(
        self,
        path: Path,
        record: IssueRecord,
        mirror: str,
        mismatches: tuple[str, ...] = (),
    ) -> OperationResult:
        candidate = replace(record, mirror=mirror)
        if candidate != record:
            self._write(candidate, path, old_path=path)
        return OperationResult(path=path, record=candidate, mismatches=mismatches)

    def _read_back(
        self, path: Path, record: IssueRecord, projection: GitHubProjection
    ) -> OperationResult:
        assert record.github is not None
        try:
            remote = self.github.read(record.github)
        except GitHubNotFound:
            return self._record_mirror_state(path, record, "MISSING")
        except GitHubError as error:
            return self._record_mirror_state(path, record, "DIVERGED", (str(error),))
        mismatches = _compare(projection, remote)
        return self._record_mirror_state(
            path,
            record,
            "DIVERGED" if mismatches else "SYNCED",
            mismatches,
        )

    def mirror(self, path: str | Path) -> OperationResult:
        source = Path(path)
        record = self._load(source)
        self._validate(record, source, replacing_path=source)
        try:
            projection = github_projection(record)
        except IssueRecordError as error:
            raise IssueOperationError(f"mirror contract violated: {error}") from error

        if record.github is None:
            try:
                number = self.github.create(projection)
            except GitHubError as error:
                raise IssueOperationError(f"failed to create GitHub mirror; local record remains PENDING: {error}") from error
            numbered = replace(record, github=number, mirror="DIVERGED")
            self._write(numbered, source, old_path=source)
            if numbered.state == "CLOSED":
                try:
                    self.github.update(number, projection)
                except GitHubNotFound:
                    return self._record_mirror_state(source, numbered, "MISSING")
                except GitHubError as error:
                    return self._record_mirror_state(
                        source, numbered, "DIVERGED", (str(error),)
                    )
            return self._read_back(source, numbered, projection)

        try:
            self.github.update(record.github, projection)
        except GitHubNotFound:
            return self._record_mirror_state(source, record, "MISSING")
        except GitHubError as error:
            return self._record_mirror_state(source, record, "DIVERGED", (str(error),))
        return self._read_back(source, record, projection)

    def _import_record(
        self,
        remote: GitHubIssue,
        *,
        issue_id: str,
        resolution: str | None,
        kind: str | None = None,
    ) -> IssueRecord:
        state = _remote_state(remote.state)
        remote_row = _metadata_line(remote.body, "Row")
        row = remote_row if remote_row and remote_row != "-" and self._row_valid(remote_row) else None
        remote_kind = _metadata_line(remote.body, "Kind")
        imported_kind = kind or (
            remote_kind if remote_kind and remote_kind != "-" else "UNKNOWN"
        )
        imported_resolution = resolution or _resolution_from_remote(remote.body) or "-"
        if state == "CLOSED" and normalize_body(imported_resolution) in {"", "-"}:
            raise IssueOperationError(
                "import contract requires non-placeholder resolution evidence for a CLOSED issue"
            )
        return IssueRecord(
            id=issue_id,
            title=remote.title,
            row=row,
            state=state,
            kind=imported_kind,
            github=remote.number,
            mirror="DIVERGED",
            availability="FULL",
            created=_remote_date(remote.created_at, "Created"),
            updated=_remote_date(remote.updated_at, "Updated"),
            closed=_remote_date(remote.closed_at, "Closed") if state == "CLOSED" else "-",
            problem=_historical_problem(remote.body),
            resolution=imported_resolution,
        )

    def import_github(
        self,
        number: int,
        *,
        promote_metadata_only: bool = False,
        resolution: str | None = None,
        owed_reference: bool = False,
        prospective_owed: OwedLookup | None = None,
    ) -> OperationResult:
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise IssueOperationError("import contract requires a positive GitHub issue number")
        try:
            remote = self.github.read(number)
        except GitHubError as error:
            raise IssueOperationError(f"import contract could not read GitHub issue #{number}: {error}") from error
        if remote.number != number:
            raise IssueOperationError("import identity contract requires the requested and returned GitHub numbers to match")

        matches: list[tuple[Path, IssueRecord]] = []
        for path in self.filesystem.issue_files(self.issues_root):
            record = self._load(path)
            if record.github == number:
                matches.append((path, record))
        if len(matches) > 1:
            raise IssueOperationError(f"import identity contract found duplicate GitHub number #{number}")

        if matches:
            source, current = matches[0]
            if current.availability == "FULL":
                self._validate(current, source, replacing_path=source)
                try:
                    projection = github_projection(current)
                except IssueRecordError as error:
                    raise IssueOperationError(f"import comparison contract violated: {error}") from error
                mismatches = _compare(projection, remote)
                desired = "DIVERGED" if mismatches else "SYNCED"
                if desired != current.mirror:
                    current = replace(current, mirror=desired)
                    self._write(current, source, old_path=source)
                return OperationResult(source, current, mismatches)
            if not promote_metadata_only:
                self._validate(current, source, replacing_path=source)
                raise IssueOperationError(
                    "metadata-only promotion contract requires --promote-metadata-only"
                )
            promoted = self._import_record(
                remote,
                issue_id=current.id,
                resolution=resolution,
            )
            candidate_owed = self._validate_move_source(
                current,
                source,
                promoted.row,
                prospective_owed,
            )
            target = issue_path(promoted, self.issues_root)
            projection = github_projection(promoted)
            mismatches = _compare(projection, remote)
            promoted = replace(promoted, mirror="DIVERGED" if mismatches else "SYNCED")
            self._write(
                promoted,
                target,
                old_path=source,
                owed_reference=owed_reference,
                prospective_owed=candidate_owed,
            )
            return OperationResult(target, promoted, mismatches, promoted=True)

        imported = self._import_record(
            remote,
            issue_id=f"ISSUE-GH-{number}",
            resolution=resolution,
        )
        projection = github_projection(imported)
        mismatches = _compare(projection, remote)
        imported = replace(imported, mirror="DIVERGED" if mismatches else "SYNCED")
        target = issue_path(imported, self.issues_root)
        self._write(imported, target, owed_reference=owed_reference)
        return OperationResult(target, imported, mismatches)

    def migrate_archive(
        self,
        archive_path: str | Path,
        *,
        staging_root: str | Path,
        promote_metadata_only: bool = False,
        resolutions: Mapping[int, str] | None = None,
        owner_resolutions: Mapping[int, str] | None = None,
        required_full: Collection[int] = (2390, 2371, 2372),
        specs_root: str | Path | None = None,
    ) -> MigrationResult:
        """Publish the complete archive/remote union without changing source records."""

        staging_root = Path(staging_root)
        if self.filesystem.exists(staging_root):
            raise IssueOperationError(
                f"migration staging target already exists: {staging_root}"
            )

        archive_path = Path(archive_path)
        try:
            frozen_archive = self.filesystem.read_bytes(archive_path)
            archived = _archive_issues(frozen_archive.decode("utf-8"))
        except (OSError, UnicodeError) as error:
            raise IssueOperationError(
                f"migration contract cannot read frozen archive {archive_path}: {error}"
            ) from error
        resolved_owners = dict(owner_resolutions or {})
        for number, row in resolved_owners.items():
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number <= 0
                or not isinstance(row, str)
                or not self._row_valid(row)
            ):
                raise IssueOperationError(
                    "migration owner resolutions require positive issue numbers "
                    "and canonical row IDs"
                )
        for archive in archived.values():
            if (
                archive.row is not None
                and not self._row_valid(archive.row)
                and archive.number not in resolved_owners
            ):
                raise IssueOperationError(
                    f"migration found noncanonical archived row {archive.row!r} "
                    f"for #{archive.number}"
                )

        remotes: dict[int, GitHubIssue] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            try:
                page = self.github.list_page(cursor)
            except GitHubError as error:
                raise IssueOperationError(
                    f"migration contract could not enumerate every GitHub issue: {error}"
                ) from error
            for remote in page.issues:
                if remote.number <= 0:
                    raise IssueOperationError(
                        "migration identity contract requires positive GitHub numbers"
                    )
                remotes.setdefault(remote.number, remote)
            if page.next_cursor is None:
                break
            if page.next_cursor in seen_cursors:
                raise IssueOperationError(
                    "migration pagination contract encountered a repeated cursor"
                )
            seen_cursors.add(page.next_cursor)
            cursor = page.next_cursor

        existing: list[tuple[Path, IssueRecord]] = []
        for path in self.filesystem.issue_files(self.issues_root):
            record = self._load(path)
            if record.availability != "METADATA_ONLY":
                try:
                    validate_issue_record(record, path, self.rows, self.owed)
                except IssueRecordError as error:
                    raise IssueOperationError(
                        f"migration existing local contract violated for {path}: {error}"
                    ) from error
            existing.append((path, record))
        try:
            validate_issue_collection(record for _, record in existing)
        except IssueRecordError as error:
            raise IssueOperationError(
                f"migration existing local collection contract violated: {error}"
            ) from error

        existing_by_number = {
            record.github: (path, record)
            for path, record in existing
            if record.github is not None
        }
        for number in archived:
            if number in remotes:
                continue
            current = existing_by_number.get(number)
            if current is not None and current[1].availability == "FULL":
                continue
            try:
                remote = self.github.read(number)
            except GitHubNotFound:
                continue
            except GitHubError as error:
                raise IssueOperationError(
                    f"migration contract could not determine reachability for #{number}: {error}"
                ) from error
            if remote.number != number:
                raise IssueOperationError(
                    "migration identity contract requires requested and returned "
                    "GitHub numbers to match"
                )
            remotes[number] = remote

        union_numbers = sorted(set(archived) | set(remotes))
        resolution_evidence = resolutions or {}
        owner_specs_root = (
            Path(specs_root)
            if specs_root is not None
            else self.issues_root.parent / "specs"
        )
        planned: dict[int, IssueRecord] = {}
        selected: dict[int, IssueRecord] = {}
        target_owners = {
            record.id: path.parent.name
            for path, record in existing
        }
        spec_rewrites: list[SpecReferenceRewrite] = []

        for number in union_numbers:
            current_pair = existing_by_number.get(number)
            current = current_pair[1] if current_pair is not None else None
            remote = remotes.get(number)
            archive = archived.get(number)
            if current is not None and current.availability == "FULL":
                selected[number] = current
                continue

            archived_row = (
                archive.row
                if archive is not None
                and archive.row is not None
                and self._row_valid(archive.row)
                else None
            )
            resolved_row = (
                resolved_owners.get(number) if archived_row is None else None
            )
            spec_owners: tuple[Path, ...] = ()
            rewrites: tuple[SpecReferenceRewrite, ...] = ()
            if archived_row is None and resolved_row is None:
                spec_owners, rewrites = _spec_owners_and_rewrites(
                    owner_specs_root,
                    number,
                )
                if len(spec_owners) > 1:
                    names = ", ".join(str(path) for path in spec_owners)
                    raise IssueOperationError(
                        f"migration found multiple existing spec owners for "
                        f"#{number}: {names}"
                    )

            if remote is not None:
                if current is not None and not promote_metadata_only:
                    raise IssueOperationError(
                        f"migration requires explicit metadata-only promotion for #{number}"
                    )
                archive_kind = (
                    archive.kind
                    if archive is not None and archive.kind != "UNKNOWN"
                    else None
                )
                imported = self._import_record(
                    remote,
                    issue_id=current.id if current is not None else f"ISSUE-GH-{number}",
                    resolution=resolution_evidence.get(number),
                    kind=archive_kind,
                )
                if archived_row is not None:
                    imported = replace(imported, row=archived_row)
                elif spec_owners:
                    imported = replace(imported, row=None)
                elif resolved_row is not None:
                    imported = replace(imported, row=resolved_row)
                if imported.row is None and not spec_owners:
                    raise IssueOperationError(
                        f"migration FULL record #{number} requires a canonical "
                        "archived row, remote row, or exactly one owning spec"
                    )
                mismatches = _compare(github_projection(imported), remote)
                candidate = replace(
                    imported,
                    mirror="DIVERGED" if mismatches else "SYNCED",
                )
                owner = candidate.row if candidate.row is not None else "_owed"
            else:
                if current is not None:
                    candidate = current
                    if archived_row is not None:
                        candidate = replace(candidate, row=archived_row)
                        owner = archived_row
                    elif spec_owners:
                        candidate = replace(candidate, row=None)
                        owner = "_owed"
                    elif resolved_row is not None:
                        candidate = replace(candidate, row=resolved_row)
                        owner = resolved_row
                    else:
                        candidate = replace(candidate, row=None)
                        owner = "_intake"
                else:
                    if archive is None:
                        raise IssueOperationError(
                            f"migration contract lost union identity #{number}"
                        )
                    candidate = IssueRecord(
                        id=f"ISSUE-GH-{number}",
                        title=archive.title,
                        row=archived_row,
                        state="UNKNOWN",
                        kind=archive.kind,
                        github=number,
                        mirror="MISSING",
                        availability="METADATA_ONLY",
                        created="UNKNOWN",
                        updated="UNKNOWN",
                        closed="UNKNOWN",
                        problem=archive.citation,
                        resolution="-",
                    )
                    if archived_row is not None:
                        owner = archived_row
                    elif spec_owners:
                        owner = "_owed"
                    elif resolved_row is not None:
                        candidate = replace(candidate, row=resolved_row)
                        owner = resolved_row
                    else:
                        owner = "_intake"

            if spec_owners:
                spec_rewrites.extend(rewrites)
            planned[number] = candidate
            selected[number] = candidate
            target_owners[candidate.id] = owner

        replacements = {record.id: record for record in planned.values()}
        final_records = [
            replacements.get(record.id, record)
            for _, record in existing
        ]
        existing_ids = {record.id for _, record in existing}
        final_records.extend(
            record
            for record in replacements.values()
            if record.id not in existing_ids
        )
        try:
            validate_issue_collection(final_records)
        except IssueRecordError as error:
            raise IssueOperationError(
                f"migration local collection contract violated: {error}"
            ) from error

        missing_required = sorted(
            number
            for number in required_full
            if number not in selected
            or selected[number].availability != "FULL"
        )
        if missing_required:
            references = ", ".join(f"#{number}" for number in missing_required)
            raise IssueOperationError(
                "migration requires FULL canonical records for branch references "
                + references
            )

        def migrated_path(record: IssueRecord, root: Path) -> Path:
            return root / target_owners[record.id] / f"{record.id}.md"

        def migrated_owed(issue_id: str) -> int:
            return int(target_owners.get(issue_id) == "_owed")

        try:
            temporary_root = self.filesystem.temporary_sibling(staging_root)
        except OSError as error:
            raise IssueOperationError(
                f"migration could not create temporary staging tree: {error}"
            ) from error

        temporary_issues_root = temporary_root / ".agents" / "issues"
        published_issues_root = staging_root / ".agents" / "issues"
        try:
            for record in sorted(final_records, key=lambda item: item.id):
                path = migrated_path(record, temporary_issues_root)
                try:
                    self.filesystem.atomic_write(path, render_issue_record(record))
                except OSError as error:
                    raise IssueOperationError(
                        f"migration staging write failed for {path}: {error}"
                    ) from error

            staged: list[tuple[Path, IssueRecord]] = []
            try:
                for path in self.filesystem.issue_files(temporary_issues_root):
                    record = parse_issue_text(self.filesystem.read_text(path))
                    validate_issue_record(
                        record,
                        path,
                        self.rows,
                        migrated_owed,
                        frozen_archive=frozen_archive,
                    )
                    staged.append((path, record))
                staged_records = [record for _, record in staged]
                validate_issue_collection(staged_records)
            except (OSError, IssueRecordError) as error:
                raise IssueOperationError(
                    f"migration staged collection contract violated: {error}"
                ) from error

            expected = {record.id: record for record in final_records}
            materialized = {record.id: record for _, record in staged}
            if len(staged) != len(final_records) or materialized != expected:
                raise IssueOperationError(
                    "migration staged collection does not match the complete plan"
                )

            try:
                self.filesystem.publish_directory(temporary_root, staging_root)
            except OSError as error:
                raise IssueOperationError(
                    f"migration staging publish failed for {staging_root}: {error}"
                ) from error
        except BaseException as error:
            try:
                self.filesystem.remove_tree(temporary_root)
            except OSError as cleanup_error:
                raise IssueOperationError(
                    "migration failed and could not remove temporary staging tree "
                    f"{temporary_root}: {cleanup_error}"
                ) from error
            raise

        records = tuple(selected[number] for number in union_numbers)
        paths = tuple(
            migrated_path(record, published_issues_root)
            for record in records
        )
        return MigrationResult(
            records,
            paths,
            tuple(
                sorted(
                    spec_rewrites,
                    key=lambda item: (str(item.path), item.line, item.old),
                )
            ),
        )

    def close(self, path: str | Path, *, resolution: str) -> OperationResult:
        source = Path(path)
        if source.parent.name == "_intake":
            raise IssueOperationError(
                "close contract rejects _intake records; promote ownership first"
            )
        current = self._load(source)
        self._validate(current, source, replacing_path=source)
        if current.state != "OPEN":
            raise IssueOperationError("close contract requires an OPEN local issue")
        if normalize_body(resolution) in {"", "-"}:
            raise IssueOperationError("close contract requires non-placeholder resolution evidence")
        operation_date = self._today()
        closed = replace(
            current,
            state="CLOSED",
            closed=operation_date,
            updated=operation_date,
            resolution=resolution,
            mirror="DIVERGED" if current.github is not None else "PENDING",
        )
        self._write(closed, source, old_path=source)
        if closed.github is None:
            return OperationResult(source, closed)
        projection = github_projection(closed)
        try:
            self.github.update(closed.github, projection)
        except GitHubNotFound:
            return self._record_mirror_state(source, closed, "MISSING")
        except GitHubError as error:
            return self._record_mirror_state(source, closed, "DIVERGED", (str(error),))
        return self._read_back(source, closed, projection)


def _default_rows(root: Path) -> set[str]:
    return canonical_rows(root)


def _default_owed(root: Path) -> OwedLookup:
    return owed_issue_counts(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a canonical local issue")
    create.add_argument("--title", required=True)
    create.add_argument("--kind", required=True)
    create.add_argument("--problem", required=True)
    create.add_argument("--row")
    create.add_argument("--resolution", default="-")
    create.add_argument("--owed-spec", help="spec that will own a new _owed ID in the same commit")

    update = subparsers.add_parser("update", help="update canonical local fields")
    update.add_argument("file")
    for name in ("title", "row", "kind", "problem", "resolution"):
        update.add_argument(f"--{name}")
    for name in ("id", "state", "github", "mirror", "availability", "created", "updated", "closed"):
        update.add_argument(f"--{name}")
    update.add_argument("--owed-spec", help="spec that will own an issue moved to _owed")

    mirror = subparsers.add_parser("mirror", help="write and read back the optional GitHub mirror")
    mirror.add_argument("file")

    imported = subparsers.add_parser("import-github", help="import one reachable GitHub issue")
    imported.add_argument("number", type=int)
    imported.add_argument("--promote-metadata-only", action="store_true")
    imported.add_argument("--resolution")
    imported.add_argument("--owed-spec", help="spec that will own a new _owed ID in the same commit")

    migrate = subparsers.add_parser(
        "migrate",
        help="stage the frozen-archive and paginated-GitHub union",
    )
    migrate.add_argument(
        "--staging-root",
        required=True,
        help="caller-owned repository-shaped staging directory",
    )
    migrate.add_argument(
        "--archive",
        default=str(ROOT / ".agents" / "completed" / "issue-index.md"),
    )
    migrate.add_argument("--promote-metadata-only", action="store_true")
    migrate.add_argument(
        "--resolutions-json",
        help="JSON object mapping closed issue numbers to resolution evidence",
    )


    close = subparsers.add_parser("close", help="close locally before closing the GitHub mirror")
    close.add_argument("file")
    close.add_argument("--resolution", required=True)
    return parser


def _validate_owed_spec(path: str | None) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError as error:
        raise IssueOperationError(f"_owed ownership contract cannot read --owed-spec {candidate}: {error}") from error
    if not re.search(r"(?m)^## Owed\s*$", text):
        raise IssueOperationError("_owed ownership contract requires --owed-spec to contain ## Owed")
    return True


def _load_resolutions(path: str | None) -> dict[int, str]:
    if path is None:
        return {}
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IssueOperationError(
            f"migration resolution evidence cannot be read from {candidate}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise IssueOperationError(
            "migration resolution evidence must be a JSON object"
        )
    resolutions: dict[int, str] = {}
    for raw_number, evidence in value.items():
        try:
            number = int(raw_number)
        except (TypeError, ValueError) as error:
            raise IssueOperationError(
                "migration resolution evidence keys must be issue numbers"
            ) from error
        if number <= 0 or not isinstance(evidence, str):
            raise IssueOperationError(
                "migration resolution evidence requires positive issue numbers "
                "and string values"
            )
        resolutions[number] = evidence
    return resolutions


def _frozen_archive_source(root: Path) -> bytes | None:
    try:
        return (root / ".agents" / "completed" / "issue-index.md").read_bytes()
    except OSError:
        return None


def run_cli(argv: Sequence[str] | None = None, *, operations: IssueOperations | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        if operations is not None:
            op = operations
        else:
            op = IssueOperations(
                issues_root=DEFAULT_ISSUES_ROOT,
                filesystem=LocalFileSystem(),
                clock=date.today,
                ulid=_default_ulid,
                github=GhCliClient(),
                rows=_default_rows(ROOT),
                owed=_default_owed(ROOT),
                frozen_archive=_frozen_archive_source(ROOT),
            )
        if arguments.command == "create":
            result = op.create(
                title=arguments.title,
                kind=arguments.kind,
                problem=arguments.problem,
                row=arguments.row,
                resolution=arguments.resolution,
                owed_reference=_validate_owed_spec(arguments.owed_spec),
            )
        elif arguments.command == "update":
            changes = {
                name: getattr(arguments, name)
                for name in (*_ALLOWED_UPDATES, *_UPDATE_LABELS)
                if getattr(arguments, name, None) is not None
            }
            owed_reference = _validate_owed_spec(arguments.owed_spec)
            result = op.update(
                arguments.file,
                owed_reference=owed_reference,
                **changes,
            )
        elif arguments.command == "mirror":
            result = op.mirror(arguments.file)
        elif arguments.command == "import-github":
            result = op.import_github(
                arguments.number,
                promote_metadata_only=arguments.promote_metadata_only,
                resolution=arguments.resolution,
                owed_reference=_validate_owed_spec(arguments.owed_spec),
            )
        elif arguments.command == "migrate":
            result = op.migrate_archive(
                arguments.archive,
                staging_root=arguments.staging_root,
                promote_metadata_only=arguments.promote_metadata_only,
                resolutions=_load_resolutions(arguments.resolutions_json),
            )
        else:
            result = op.close(arguments.file, resolution=arguments.resolution)
    except (IssueOperationError, IssueRecordError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if isinstance(result, MigrationResult):
        print(
            f"staged {len(result.records)} canonical issue records under "
            f"{arguments.staging_root}"
        )
        for rewrite in result.spec_rewrites:
            print(
                f"rewrite {rewrite.path}:{rewrite.line}: "
                f"{rewrite.old} -> {rewrite.new}"
            )
    else:
        print(result.path)
        if result.promoted:
            print("promoted METADATA_ONLY record to FULL", file=sys.stderr)
        if result.mismatches:
            print("GitHub mirror differs: " + ", ".join(result.mismatches), file=sys.stderr)
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
