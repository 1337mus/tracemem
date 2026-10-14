"""Local SQLite store for append-only evidence and derived retrieval caches."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path

from .schema import EVIDENCE_APPLICATION_ID, EVIDENCE_SCHEMA_VERSION, EvidenceRecord

_TOKEN_RE = re.compile(r"[^\W_] +", re.UNICODE | re.VERBOSE)
_REQUIRED_TABLES = {"evidence", "evidence_heads", "evidence_vectors", "evidence_fts"}
_REQUIRED_TRIGGERS = {
    "evidence_immutable_insert",
    "evidence_immutable_update",
    "evidence_immutable_delete",
    "evidence_head_fts",
    "evidence_head_fts_update",
    "evidence_head_fts_delete",
}


class EvidenceStore:
    """Append-only evidence storage with revision-aware reads."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()
        self._ensure_parent(self.path.parent)
        created = self._create_database_file(self.path)
        if not created:
            self._preflight_existing_database(self.path)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if created:
                self._initialize(connection)
            else:
                self._validate_existing(connection)
            self._connection = connection
        except Exception:
            if connection is not None:
                connection.close()
            raise

    @staticmethod
    def _preflight_existing_database(path: Path) -> None:
        try:
            uri = path.resolve(strict=True).as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, timeout=10, isolation_level=None, uri=True)
        except (OSError, sqlite3.DatabaseError, ValueError):
            raise ValueError(
                "existing file is not a supported TraceMem evidence database"
            ) from None
        try:
            connection.row_factory = sqlite3.Row
            EvidenceStore._validate_existing(connection)
        finally:
            connection.close()

    @staticmethod
    def _ensure_parent(parent: Path) -> None:
        missing: list[Path] = []
        current = parent
        while not current.exists():
            missing.append(current)
            if current.parent == current:
                break
            current = current.parent
        for directory in reversed(missing):
            try:
                os.mkdir(directory, 0o700)
            except FileExistsError:
                continue
            os.chmod(directory, 0o700)

    @staticmethod
    def _create_database_file(path: Path) -> bool:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            try:
                metadata = os.lstat(path)
            except OSError:
                raise ValueError("cannot inspect evidence database path") from None
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("evidence database path must not be a symlink") from None
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("evidence database path must be a regular file") from None
            return False
        except OSError:
            raise ValueError("cannot create evidence database") from None
        else:
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            return True

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        try:
            connection.executescript(
                f"""
                BEGIN IMMEDIATE;
                CREATE TABLE evidence (
                    record_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    part_index INTEGER NOT NULL CHECK(part_index >= 0),
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    recorded_at TEXT,
                    source_locator TEXT,
                    line_number INTEGER CHECK(line_number IS NULL OR line_number >= 1),
                    content_kind TEXT NOT NULL,
                    PRIMARY KEY(record_id, revision)
                ) WITHOUT ROWID;
                CREATE TABLE evidence_heads (
                    record_id TEXT PRIMARY KEY,
                    revision TEXT NOT NULL,
                    FOREIGN KEY(record_id, revision)
                        REFERENCES evidence(record_id, revision)
                );
                CREATE TABLE evidence_vectors (
                    record_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY(record_id, revision, fingerprint),
                    FOREIGN KEY(record_id, revision)
                        REFERENCES evidence(record_id, revision) ON DELETE CASCADE
                );
                CREATE VIRTUAL TABLE evidence_fts USING fts5(
                    record_id UNINDEXED,
                    revision UNINDEXED,
                    text,
                    tokenize='unicode61'
                );
                CREATE TRIGGER evidence_immutable_update
                BEFORE UPDATE ON evidence BEGIN
                    SELECT RAISE(ABORT, 'evidence rows are immutable');
                END;
                CREATE TRIGGER evidence_immutable_delete
                BEFORE DELETE ON evidence BEGIN
                    SELECT RAISE(ABORT, 'evidence rows are immutable');
                END;
                CREATE TRIGGER evidence_immutable_insert
                BEFORE INSERT ON evidence
                WHEN EXISTS (
                    SELECT 1 FROM evidence
                    WHERE record_id = new.record_id AND revision = new.revision
                ) BEGIN
                    SELECT RAISE(ABORT, 'evidence revisions are immutable');
                END;
                CREATE TRIGGER evidence_head_fts
                AFTER INSERT ON evidence_heads BEGIN
                    DELETE FROM evidence_fts WHERE rowid = new.rowid;
                    INSERT INTO evidence_fts(rowid, record_id, revision, text)
                    SELECT new.rowid, evidence.record_id, evidence.revision, evidence.text
                    FROM evidence
                    WHERE evidence.record_id = new.record_id
                        AND evidence.revision = new.revision;
                END;
                CREATE TRIGGER evidence_head_fts_update
                AFTER UPDATE OF revision ON evidence_heads BEGIN
                    DELETE FROM evidence_fts WHERE rowid = new.rowid;
                    INSERT INTO evidence_fts(rowid, record_id, revision, text)
                    SELECT new.rowid, evidence.record_id, evidence.revision, evidence.text
                    FROM evidence
                    WHERE evidence.record_id = new.record_id
                        AND evidence.revision = new.revision;
                END;
                CREATE TRIGGER evidence_head_fts_delete
                AFTER DELETE ON evidence_heads BEGIN
                    DELETE FROM evidence_fts WHERE rowid = old.rowid;
                END;
                PRAGMA application_id = {EVIDENCE_APPLICATION_ID};
                PRAGMA user_version = {EVIDENCE_SCHEMA_VERSION};
                COMMIT;
                """
            )
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise

    @staticmethod
    def _validate_existing(connection: sqlite3.Connection) -> None:
        try:
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            objects = {
                (row["type"], row["name"])
                for row in connection.execute(
                    "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'trigger')"
                )
            }
        except sqlite3.DatabaseError:
            raise ValueError("existing file is not a TraceMem evidence database") from None
        tables = {name for kind, name in objects if kind == "table"}
        triggers = {name for kind, name in objects if kind == "trigger"}
        if (
            application_id != EVIDENCE_APPLICATION_ID
            or schema_version != EVIDENCE_SCHEMA_VERSION
            or not _REQUIRED_TABLES.issubset(tables)
            or not _REQUIRED_TRIGGERS.issubset(triggers)
        ):
            raise ValueError("existing file is not a supported TraceMem evidence database")

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def ingest(self, records: Iterable[EvidenceRecord]) -> int:
        inserted = 0
        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                if not isinstance(record, EvidenceRecord):
                    raise TypeError("ingest accepts EvidenceRecord values only")
                existing = connection.execute(
                    """
                    SELECT source, session_id, message_id, part_index, role, text,
                        recorded_at, source_locator, line_number, content_kind
                    FROM evidence WHERE record_id = ? AND revision = ?
                    """,
                    (record.record_id, record.revision),
                ).fetchone()
                values = (
                    record.source,
                    record.session_id,
                    record.message_id,
                    record.part_index,
                    record.role,
                    record.text,
                    record.recorded_at,
                    record.source_locator,
                    record.line_number,
                    record.content_kind,
                )
                if existing is None:
                    connection.execute(
                        """
                    INSERT INTO evidence (
                        record_id, revision, source, session_id, message_id, part_index,
                        role, text, recorded_at, source_locator, line_number, content_kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            record.record_id,
                            record.revision,
                            *values,
                        ),
                    )
                    inserted += 1
                elif tuple(existing) != values:
                    raise ValueError("evidence revision hash collision")
                connection.execute(
                    """
                    INSERT INTO evidence_heads(record_id, revision) VALUES (?, ?)
                    ON CONFLICT(record_id) DO UPDATE SET revision = excluded.revision
                    WHERE evidence_heads.revision != excluded.revision
                    """,
                    (record.record_id, record.revision),
                )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        return inserted

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> EvidenceRecord:
        return EvidenceRecord(
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

    def records(self, current_only: bool = True) -> list[EvidenceRecord]:
        if not isinstance(current_only, bool):
            raise TypeError("current_only must be a bool")
        if current_only:
            rows = self._connection.execute(
                """
                SELECT evidence.* FROM evidence
                JOIN evidence_heads AS heads
                    ON heads.record_id = evidence.record_id
                    AND heads.revision = evidence.revision
                ORDER BY evidence.record_id
                """
            )
        else:
            rows = self._connection.execute("SELECT * FROM evidence ORDER BY record_id, revision")
        return [self._record_from_row(row) for row in rows]

    def get(self, record_id: str, revision: str) -> EvidenceRecord:
        row = self._connection.execute(
            "SELECT * FROM evidence WHERE record_id = ? AND revision = ?",
            (record_id, revision),
        ).fetchone()
        if row is None:
            raise KeyError((record_id, revision))
        return self._record_from_row(row)

    def lexical_search(self, query: str, limit: int = 10) -> list[tuple[EvidenceRecord, float]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if len(query) > 4096:
            raise ValueError("query is too long")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 1000:
            raise ValueError("limit must be an integer between 0 and 1000")
        if limit == 0:
            return []
        tokens = _TOKEN_RE.findall(query)
        if not tokens:
            return []
        match_expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        rows = self._connection.execute(
            """
            SELECT evidence.*, -bm25(evidence_fts) AS score
            FROM evidence_fts
            JOIN evidence ON evidence.record_id = evidence_fts.record_id
                AND evidence.revision = evidence_fts.revision
            JOIN evidence_heads AS heads
                ON heads.record_id = evidence.record_id
                AND heads.revision = evidence.revision
            WHERE evidence_fts MATCH ?
            ORDER BY score DESC, evidence.record_id ASC
            LIMIT ?
            """,
            (match_expression, limit),
        )
        return [(self._record_from_row(row), float(row["score"])) for row in rows]

    def vector_cache(self, record_id: str, revision: str, fingerprint: str) -> list[float] | None:
        self._validate_cache_key(record_id, revision, fingerprint)
        row = self._connection.execute(
            """
            SELECT vector_json FROM evidence_vectors
            WHERE record_id = ? AND revision = ? AND fingerprint = ?
            """,
            (record_id, revision, fingerprint),
        ).fetchone()
        if row is None:
            return None
        try:
            values = json.loads(row["vector_json"])
        except (TypeError, json.JSONDecodeError):
            raise ValueError("stored vector cache is malformed") from None
        if not isinstance(values, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("stored vector cache is malformed")
        return [float(value) for value in values]

    def cache_vector(
        self,
        record_id: str,
        revision: str,
        fingerprint: str,
        vector: Sequence[float],
    ) -> None:
        self._validate_cache_key(record_id, revision, fingerprint)
        if isinstance(vector, (str, bytes, bytearray)):
            raise TypeError("vector must be a sequence of finite numbers")
        normalized: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("vector must contain finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("vector must contain finite numbers")
            normalized.append(number)
        serialized = json.dumps(normalized, separators=(",", ":"), allow_nan=False)
        self._connection.execute(
            """
            INSERT INTO evidence_vectors(record_id, revision, fingerprint, vector_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(record_id, revision, fingerprint)
            DO UPDATE SET vector_json = excluded.vector_json
            """,
            (record_id, revision, fingerprint, serialized),
        )

    @staticmethod
    def _validate_cache_key(record_id: str, revision: str, fingerprint: str) -> None:
        if any(
            not isinstance(value, str) or not value for value in (record_id, revision, fingerprint)
        ):
            raise ValueError("vector cache keys must be non-empty strings")


__all__ = ["EvidenceStore"]
