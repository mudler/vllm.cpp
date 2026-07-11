#!/usr/bin/env python3
"""Drop benchmark-owned file pages without root and prove zero residency.

Linux ``POSIX_FADV_DONTNEED`` is the non-privileged equivalent needed by the
online serving gate on hosts where global ``/proc/sys/vm/drop_caches`` is not
delegated.  Every regular file below the requested roots is enumerated,
deduplicated by inode, advised in full, and checked with ``mincore(2)``.  A
non-zero resident-page count is fatal rather than a best-effort success.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import mmap
import os
import pathlib
import stat
import sys
import time
from collections.abc import Sequence
from typing import Any


class CacheDropError(RuntimeError):
    """A cache-eviction precondition or proof failed."""


_LIBC = ctypes.CDLL(None, use_errno=True)
_MINCORE = _LIBC.mincore
_MINCORE.argtypes = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_ubyte),
)
_MINCORE.restype = ctypes.c_int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: pathlib.Path, value: Any) -> None:
    if path.exists():
        raise CacheDropError(f"refusing to overwrite cache-drop report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _regular_files(roots: Sequence[pathlib.Path]) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    duplicate_count = 0
    for root in roots:
        if not root.exists():
            raise CacheDropError(f"cache-drop root is absent: {root}")
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            try:
                metadata = candidate.stat()
            except FileNotFoundError as error:
                raise CacheDropError(
                    f"cache-drop file disappeared during inventory: {candidate}"
                ) from error
            if not stat.S_ISREG(metadata.st_mode):
                continue
            key = (metadata.st_dev, metadata.st_ino)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            entries.append(
                {
                    "path": str(candidate.absolute()),
                    "resolved_path": str(candidate.resolve(strict=True)),
                    "size_bytes": metadata.st_size,
                }
            )
    if not entries:
        raise CacheDropError("cache-drop inventory contains no regular files")
    entries.sort(key=lambda item: item["resolved_path"])
    return entries, duplicate_count


def resident_bytes(file_descriptor: int, size_bytes: int) -> int:
    """Return page-cache residency for one regular file via ``mincore(2)``."""

    if size_bytes == 0:
        return 0
    page_size = mmap.PAGESIZE
    page_count = (size_bytes + page_size - 1) // page_size
    vector = (ctypes.c_ubyte * page_count)()
    mapping = mmap.mmap(file_descriptor, size_bytes, access=mmap.ACCESS_COPY)
    try:
        anchor = ctypes.c_char.from_buffer(mapping)
        try:
            result = _MINCORE(
                ctypes.c_void_p(ctypes.addressof(anchor)),
                ctypes.c_size_t(size_bytes),
                vector,
            )
        finally:
            del anchor
    finally:
        mapping.close()
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return sum(1 for value in vector if value & 1) * page_size


def drop_file_cache(
    roots: Sequence[pathlib.Path],
    *,
    retries: int = 3,
) -> dict[str, Any]:
    """Evict and verify all regular files below ``roots``."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise CacheDropError("POSIX_FADV_DONTNEED is unavailable on this platform")
    if retries < 1:
        raise CacheDropError("cache-drop retries must be positive")
    entries, duplicate_count = _regular_files(roots)
    inventory_bytes = _canonical_json(entries).encode("utf-8")
    os.sync()
    resident_before = 0
    resident_after = 0
    for entry in entries:
        descriptor = os.open(entry["resolved_path"], os.O_RDONLY | os.O_CLOEXEC)
        try:
            before = resident_bytes(descriptor, entry["size_bytes"])
            after = before
            attempts = 0
            while after and attempts < retries:
                attempts += 1
                os.posix_fadvise(
                    descriptor,
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )
                after = resident_bytes(descriptor, entry["size_bytes"])
                if after:
                    time.sleep(0.05)
            entry["resident_before_bytes"] = before
            entry["resident_after_bytes"] = after
            entry["attempts"] = attempts
            resident_before += before
            resident_after += after
        finally:
            os.close(descriptor)
    result = {
        "duplicate_inode_count": duplicate_count,
        "file_count": len(entries),
        "file_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "files": entries,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "logical_bytes": sum(entry["size_bytes"] for entry in entries),
        "method": "posix_fadvise-dontneed+mincore",
        "page_size_bytes": mmap.PAGESIZE,
        "resident_after_bytes": resident_after,
        "resident_before_bytes": resident_before,
        "roots": [str(root.absolute()) for root in roots],
        "succeeded": resident_after == 0,
    }
    if resident_after:
        raise CacheDropError(
            f"cache-drop proof retained {resident_after} resident bytes"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise CacheDropError(f"refusing to overwrite cache-drop report: {args.output}")
    try:
        report = drop_file_cache(args.root)
    except (CacheDropError, OSError) as error:
        report = {
            "error": f"{type(error).__name__}: {error}",
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "posix_fadvise-dontneed+mincore",
            "roots": [str(root.absolute()) for root in args.root],
            "succeeded": False,
        }
        _write_json_atomic(args.output, report)
        print(_canonical_json(report), file=sys.stderr)
        return 1
    _write_json_atomic(args.output, report)
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
