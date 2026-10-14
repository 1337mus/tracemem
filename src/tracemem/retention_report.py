"""Read-only analysis of persisted TraceMem snapshot pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

from .ledger import CandidateEntry, CandidateLedger, _entry_from_json
from .retention import _parse_citation, _validate_selector, plan_source_erasure
from .schema import EvidenceRecord
from .storage import EvidenceStore

_MAX_DATABASE_BYTES = 64 * 1024 * 1024
_ERROR_MESSAGE = "retention snapshots could not be inspected"
_EVIDENCE_COLUMNS = (
    "record_id",
    "revision",
    "source",
    "session_id",
    "message_id",
    "part_index",
    "role",
    "text",
    "recorded_at",
    "source_locator",
    "line_number",
    "content_kind",
)
_LEDGER_COLUMNS = {
    "candidate_entries": ("entry_id", "payload_json"),
    "candidate_lineage": ("entry_id", "superseded_id"),
}
_LIMITATIONS = (
    "No deletion or store mutation is performed.",
    "Inputs must be quiescent rollback-mode databases; live WAL snapshots are unsupported.",
    "The two copied stores are not a guaranteed atomic cross-store transaction.",
    "Adversarial or ABA concurrent file changes are outside this diagnostic.",
    "FTS and vector-cache contents/counts are outside the planner; logical vector-cache contents "
    "and counts are not certified.",
    "External exports, backups, unlinked copies, free SQLite pages, and in-memory caches "
    "cannot be discovered.",
    "Semantic dependence and complete information-flow detection are not guaranteed.",
)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    canonical_path: Path
    data: bytes
    sha256: str
    stat_key: tuple[int, int, int, int, int]


def _path(value: str | os.PathLike[str]) -> Path:
    try:
        return Path(os.fsdecode(os.fspath(value))).expanduser()
    except (TypeError, ValueError, OSError):
        raise ValueError(_ERROR_MESSAGE) from None


def _stat_key(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _check_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        try:
            os.lstat(sidecar)
        except FileNotFoundError:
            continue
        except OSError:
            raise ValueError(_ERROR_MESSAGE) from None
        raise ValueError(_ERROR_MESSAGE)


def _read_snapshot(path: Path) -> _Snapshot:
    _check_sidecars(path)
    descriptor = -1
    try:
        link_stat = os.lstat(path)
        if stat.S_ISLNK(link_stat.st_mode) or not stat.S_ISREG(link_stat.st_mode):
            raise ValueError(_ERROR_MESSAGE)
        flags = os.O_RDONLY | os.O_NONBLOCK
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (link_stat.st_dev, link_stat.st_ino)
            or before.st_size > _MAX_DATABASE_BYTES
        ):
            raise ValueError(_ERROR_MESSAGE)
        chunks: list[bytes] = []
        length = 0
        while length <= _MAX_DATABASE_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_DATABASE_BYTES + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
        if length > _MAX_DATABASE_BYTES:
            raise ValueError(_ERROR_MESSAGE)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if _stat_key(before) != _stat_key(after) or len(data) != after.st_size:
            raise ValueError(_ERROR_MESSAGE)
        if len(data) < 100 or data[:16] != b"SQLite format 3\x00":
            raise ValueError(_ERROR_MESSAGE)
        if data[18] != 1 or data[19] != 1:
            raise ValueError(_ERROR_MESSAGE)
        canonical = path.resolve(strict=True)
        canonical_stat = os.stat(canonical)
        if (canonical_stat.st_dev, canonical_stat.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(_ERROR_MESSAGE)
        _check_sidecars(path)
        return _Snapshot(
            path,
            canonical,
            data,
            hashlib.sha256(data).hexdigest(),
            _stat_key(after),
        )
    except (OSError, sqlite3.Error):
        raise ValueError(_ERROR_MESSAGE) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_unchanged(initial: _Snapshot) -> None:
    current = _read_snapshot(initial.path)
    if (
        current.canonical_path != initial.canonical_path
        or current.stat_key != initial.stat_key
        or current.sha256 != initial.sha256
        or len(current.data) != len(initial.data)
    ):
        raise ValueError(_ERROR_MESSAGE)


def _memory_connection(snapshot: _Snapshot) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.deserialize(snapshot.data)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        integrity = tuple(row[0] for row in connection.execute("PRAGMA integrity_check"))
        foreign_key_errors = tuple(connection.execute("PRAGMA foreign_key_check"))
        if integrity != ("ok",) or foreign_key_errors:
            raise ValueError(_ERROR_MESSAGE)
        return connection
    except Exception:
        connection.close()
        raise


def _validate_table_columns(
    connection: sqlite3.Connection, table: str, expected: tuple[str, ...]
) -> None:
    rows = tuple(connection.execute(f"PRAGMA table_xinfo({table})"))
    if tuple(row["name"] for row in rows) != expected or any(row["hidden"] != 0 for row in rows):
        raise ValueError(_ERROR_MESSAGE)


def _load_evidence(connection: sqlite3.Connection) -> tuple[EvidenceRecord, ...]:
    _validate_table_columns(connection, "evidence", _EVIDENCE_COLUMNS)
    EvidenceStore._validate_existing(connection)
    columns = ", ".join(_EVIDENCE_COLUMNS)
    rows = connection.execute(f"SELECT {columns} FROM evidence ORDER BY record_id, revision")
    records: list[EvidenceRecord] = []
    for row in rows:
        record = EvidenceRecord(
            source=row["source"],
            session_id=row["session_id"],
            message_id=row["message_id"],
            part_index=row["part_index"],
            role=row["role"],
            text=row["text"],
            recorded_at=row["recorded_at"],
            source_locator=row["source_locator"],
            line_number=row["line_number"],
            content_kind=row["content_kind"],
        )
        if record.record_id != row["record_id"] or record.revision != row["revision"]:
            raise ValueError(_ERROR_MESSAGE)
        records.append(record)
    return tuple(records)


def _load_ledger(
    connection: sqlite3.Connection,
) -> tuple[tuple[CandidateEntry, ...], tuple[tuple[str, str], ...]]:
    CandidateLedger._validate(connection)
    for table, columns in _LEDGER_COLUMNS.items():
        _validate_table_columns(connection, table, columns)
    rows = connection.execute(
        "SELECT entry_id, payload_json FROM candidate_entries ORDER BY entry_id"
    )
    entries = tuple(_entry_from_json(row["entry_id"], row["payload_json"]) for row in rows)
    sql_edges = tuple(
        sorted(
            (row["entry_id"], row["superseded_id"])
            for row in connection.execute("SELECT entry_id, superseded_id FROM candidate_lineage")
        )
    )
    payload_edges = tuple(
        sorted((entry.entry_id, parent_id) for entry in entries for parent_id in entry.supersedes)
    )
    if sql_edges != payload_edges:
        raise ValueError(_ERROR_MESSAGE)
    return entries, sql_edges


def inspect_source_retention(
    evidence_path: str | os.PathLike[str],
    ledger_path: str | os.PathLike[str],
    *,
    source: str,
    session_id: str,
) -> dict[str, Any]:
    """Inspect two persisted stores and return a private retention report."""
    try:
        selected_source = _validate_selector(source)
        selected_session = _validate_selector(session_id)
        source_snapshot = _read_snapshot(_path(evidence_path))
        ledger_snapshot = _read_snapshot(_path(ledger_path))
        if source_snapshot.stat_key[:2] == ledger_snapshot.stat_key[:2]:
            raise ValueError(_ERROR_MESSAGE)

        source_connection = _memory_connection(source_snapshot)
        try:
            ledger_connection = _memory_connection(ledger_snapshot)
            try:
                records = _load_evidence(source_connection)
                entries, _ = _load_ledger(ledger_connection)
                records_by_key = {(record.record_id, record.revision): record for record in records}
                for entry in entries:
                    for citation in entry.verification.citations:
                        key = _parse_citation(citation, records_by_key)
                        if key not in records_by_key:
                            raise ValueError(_ERROR_MESSAGE)
                impact = plan_source_erasure(
                    records,
                    entries,
                    source=selected_source,
                    session_id=selected_session,
                )
            finally:
                ledger_connection.close()
        finally:
            source_connection.close()

        _verify_unchanged(source_snapshot)
        _verify_unchanged(ledger_snapshot)
        return {
            "schema_version": 1,
            "source_snapshots": {
                "evidence": {
                    "sha256": source_snapshot.sha256,
                    "bytes": len(source_snapshot.data),
                },
                "ledger": {"sha256": ledger_snapshot.sha256, "bytes": len(ledger_snapshot.data)},
            },
            "evidence_revision_count": len(records),
            "candidate_count": len(entries),
            "impact": asdict(impact),
            "snapshot": "quiescent rollback-mode database copies",
            "limitations": _LIMITATIONS,
        }
    except Exception:
        raise ValueError(_ERROR_MESSAGE) from None


def main(argv: list[str] | None = None) -> int:
    """Run the retention report command-line interface."""
    try:
        parser = _ArgumentParser(
            prog="python -m tracemem.retention_report",
            description="Create a bounded read-only retention report from local store snapshots.",
        )
        parser.add_argument("--evidence-db", required=True)
        parser.add_argument("--ledger-db", required=True)
        parser.add_argument("--source", required=True)
        parser.add_argument("--session-id", required=True)
        parser.add_argument("--output", required=True)
        arguments = parser.parse_args(argv)

        report = inspect_source_retention(
            arguments.evidence_db,
            arguments.ledger_db,
            source=arguments.source,
            session_id=arguments.session_id,
        )
        payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        output = _output_path(arguments.output)
        input_paths = (
            _canonical_input_path(arguments.evidence_db),
            _canonical_input_path(arguments.ledger_db),
        )
        reserved = {
            _path_key(reserved_path)
            for input_path in input_paths
            for reserved_path in (
                input_path,
                *(Path(f"{input_path}{suffix}") for suffix in ("-wal", "-shm", "-journal")),
            )
        }
        if _path_key(output) in reserved:
            raise ValueError(_ERROR_MESSAGE)
        _write_report(output, payload)
        return 0
    except SystemExit as error:
        return int(error.code or 0)
    except Exception:
        sys.stderr.write("retention report could not be created\n")
        return 2


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid command line")


def _output_path(value: str) -> Path:
    path = _path(value)
    if not path.name:
        raise ValueError(_ERROR_MESSAGE)
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        raise ValueError(_ERROR_MESSAGE) from None
    if not parent.is_dir():
        raise ValueError(_ERROR_MESSAGE)
    return parent / path.name


def _canonical_input_path(value: str) -> Path:
    try:
        return _path(value).resolve(strict=True)
    except OSError:
        raise ValueError(_ERROR_MESSAGE) from None


def _path_key(path: Path) -> str:
    """Compare paths conservatively across case- or normalization-insensitive filesystems."""
    normalized = os.path.normpath(os.fspath(path))
    return unicodedata.normalize("NFC", normalized).casefold()


def _write_report(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short report write")
            offset += written
    except OSError:
        raise ValueError(_ERROR_MESSAGE) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
