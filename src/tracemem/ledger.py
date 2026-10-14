"""Private append-only storage for verified candidate proposals."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

from .claims import Claim, EvidenceSpan, VerificationResult, verify_candidate
from .schema import EvidenceRecord
from .storage import EvidenceStore

_APPLICATION_ID = 0x54434C44  # ASCII "TCLD"
_SCHEMA_VERSION = 1
_REQUIRED_TABLES = {"candidate_entries", "candidate_lineage"}
_REQUIRED_TRIGGERS = {
    "candidate_entries_no_duplicate",
    "candidate_entries_no_update",
    "candidate_entries_no_delete",
    "candidate_lineage_no_duplicate",
    "candidate_lineage_no_update",
    "candidate_lineage_no_delete",
}
_ENTRY_FIELDS = {"claim", "verification", "extractor_id", "recorded_at", "supersedes"}


@dataclass(frozen=True, slots=True)
class CandidateEntry:
    """Immutable proposal snapshot and deterministic evidence-integrity result."""

    entry_id: str
    claim: Claim
    verification: VerificationResult
    extractor_id: str
    recorded_at: str
    supersedes: tuple[str, ...]


class CandidateLedger:
    """Append-only private SQLite ledger; verification never asserts semantic truth."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path).expanduser()
        EvidenceStore._ensure_parent(self.path.parent)
        created = EvidenceStore._create_database_file(self.path)
        if not created:
            self._preflight(self.path)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if created:
                self._initialize(connection)
            else:
                self._validate(connection)
            os.chmod(self.path, 0o600)
            self._connection = connection
        except Exception:
            if connection is not None:
                connection.close()
            raise

    @classmethod
    def _preflight(cls, path: Path) -> None:
        try:
            uri = path.resolve(strict=True).as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, timeout=10, isolation_level=None, uri=True)
        except (OSError, sqlite3.DatabaseError, ValueError):
            raise ValueError("existing file is not a supported candidate ledger") from None
        try:
            connection.row_factory = sqlite3.Row
            cls._validate(connection)
        except (sqlite3.DatabaseError, TypeError, ValueError):
            raise ValueError("existing file is not a supported candidate ledger") from None
        finally:
            connection.close()

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            f"""
            BEGIN IMMEDIATE;
            CREATE TABLE candidate_entries (
                entry_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE candidate_lineage (
                entry_id TEXT NOT NULL REFERENCES candidate_entries(entry_id),
                superseded_id TEXT NOT NULL REFERENCES candidate_entries(entry_id),
                PRIMARY KEY(entry_id, superseded_id)
            ) WITHOUT ROWID;
            CREATE TRIGGER candidate_entries_no_duplicate
            BEFORE INSERT ON candidate_entries
            WHEN EXISTS (SELECT 1 FROM candidate_entries WHERE entry_id = new.entry_id)
            BEGIN SELECT RAISE(ABORT, 'candidate entries are immutable'); END;
            CREATE TRIGGER candidate_entries_no_update
            BEFORE UPDATE ON candidate_entries
            BEGIN SELECT RAISE(ABORT, 'candidate entries are immutable'); END;
            CREATE TRIGGER candidate_entries_no_delete
            BEFORE DELETE ON candidate_entries
            BEGIN SELECT RAISE(ABORT, 'candidate entries are immutable'); END;
            CREATE TRIGGER candidate_lineage_no_duplicate
            BEFORE INSERT ON candidate_lineage
            WHEN EXISTS (
                SELECT 1 FROM candidate_lineage
                WHERE entry_id = new.entry_id AND superseded_id = new.superseded_id
            ) BEGIN SELECT RAISE(ABORT, 'candidate lineage is immutable'); END;
            CREATE TRIGGER candidate_lineage_no_update
            BEFORE UPDATE ON candidate_lineage
            BEGIN SELECT RAISE(ABORT, 'candidate lineage is immutable'); END;
            CREATE TRIGGER candidate_lineage_no_delete
            BEFORE DELETE ON candidate_lineage
            BEGIN SELECT RAISE(ABORT, 'candidate lineage is immutable'); END;
            PRAGMA application_id = {_APPLICATION_ID};
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
            """
        )

    @staticmethod
    def _validate(connection: sqlite3.Connection) -> None:
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        objects = {
            (row["type"], row["name"])
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            )
        }
        tables = {name for kind, name in objects if kind == "table"}
        triggers = {name for kind, name in objects if kind == "trigger"}
        if (
            application_id != _APPLICATION_ID
            or version != _SCHEMA_VERSION
            or not _REQUIRED_TABLES.issubset(tables)
            or not _REQUIRED_TRIGGERS.issubset(triggers)
        ):
            raise ValueError("database is not a supported candidate ledger")
        columns = {
            "candidate_entries": ("entry_id", "payload_json"),
            "candidate_lineage": ("entry_id", "superseded_id"),
        }
        for table, expected in columns.items():
            actual = tuple(row["name"] for row in connection.execute(f"PRAGMA table_info({table})"))
            ddl = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()[0]
            if actual != expected or "WITHOUT ROWID" not in ddl.upper():
                raise ValueError("candidate ledger table definition is unsupported")
        foreign_keys = {
            (row["table"], row["from"], row["to"])
            for row in connection.execute("PRAGMA foreign_key_list(candidate_lineage)")
        }
        if foreign_keys != {
            ("candidate_entries", "entry_id", "entry_id"),
            ("candidate_entries", "superseded_id", "entry_id"),
        }:
            raise ValueError("candidate ledger lineage constraints are unsupported")
        trigger_sql = {
            row["name"]: row["sql"].upper()
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        expected_triggers = {
            "candidate_entries_no_duplicate": ("INSERT", "candidate_entries"),
            "candidate_entries_no_update": ("UPDATE", "candidate_entries"),
            "candidate_entries_no_delete": ("DELETE", "candidate_entries"),
            "candidate_lineage_no_duplicate": ("INSERT", "candidate_lineage"),
            "candidate_lineage_no_update": ("UPDATE", "candidate_lineage"),
            "candidate_lineage_no_delete": ("DELETE", "candidate_lineage"),
        }
        if any(
            f"BEFORE {event} ON {table.upper()}" not in trigger_sql[name]
            or "RAISE(ABORT" not in trigger_sql[name]
            for name, (event, table) in expected_triggers.items()
        ):
            raise ValueError("candidate ledger immutability triggers are unsupported")

    def __enter__(self) -> CandidateLedger:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def append(
        self,
        claim: Claim,
        records: Sequence[EvidenceRecord],
        extractor_id: str,
        recorded_at: str,
        supersedes: tuple[str, ...] = (),
    ) -> CandidateEntry:
        if type(claim) is not Claim:
            raise TypeError("claim must be a Claim")
        extractor_id = _required_text(extractor_id, "extractor_id", 256)
        recorded_at = _canonical_utc(recorded_at)
        supersedes = _validate_supersedes(supersedes)
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of EvidenceRecord values")
        result = verify_candidate(claim, records)
        payload = _entry_payload(claim, result, extractor_id, recorded_at, supersedes)
        encoded = _canonical_json(payload)
        entry_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json FROM candidate_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != encoded:
                    raise ValueError("candidate entry hash collision")
                connection.commit()
                return _entry_from_json(entry_id, encoded)
            if entry_id in supersedes:
                raise ValueError("an entry cannot supersede itself")
            for target in supersedes:
                found = connection.execute(
                    "SELECT 1 FROM candidate_entries WHERE entry_id = ?", (target,)
                ).fetchone()
                if found is None:
                    raise ValueError(f"unknown superseded entry: {target}")
            connection.execute(
                "INSERT INTO candidate_entries(entry_id, payload_json) VALUES (?, ?)",
                (entry_id, encoded),
            )
            connection.executemany(
                "INSERT INTO candidate_lineage(entry_id, superseded_id) VALUES (?, ?)",
                ((entry_id, target) for target in supersedes),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        return _entry_from_json(entry_id, encoded)

    def get(self, entry_id: str) -> CandidateEntry:
        entry_id = _validate_entry_id(entry_id)
        row = self._connection.execute(
            "SELECT payload_json FROM candidate_entries WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            raise KeyError(entry_id)
        return _entry_from_json(entry_id, row["payload_json"])

    def entries(self) -> tuple[CandidateEntry, ...]:
        rows = self._connection.execute("SELECT entry_id FROM candidate_entries ORDER BY entry_id")
        return tuple(self.get(row["entry_id"]) for row in rows)


def _required_text(value: object, name: str, limit: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be non-empty text of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} contains invalid Unicode") from None
    return value


def _canonical_utc(value: object) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("recorded_at must be a canonical UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ValueError("recorded_at must be a canonical UTC timestamp ending in Z") from None
    fraction = f".{parsed.microsecond:06d}".rstrip("0") if parsed.microsecond else ""
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%S") + fraction + "Z"
    if canonical != value or parsed.tzinfo != UTC:
        raise ValueError("recorded_at must be a canonical UTC timestamp ending in Z")
    return canonical


def _validate_supersedes(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("supersedes must be a tuple of entry IDs")
    ids = tuple(_validate_entry_id(item) for item in value)
    if len(set(ids)) != len(ids):
        raise ValueError("supersedes cannot contain duplicate entry IDs")
    return ids


def _validate_entry_id(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("entry_id must be a lowercase SHA-256 hex digest")
    return value


def _entry_payload(
    claim: Claim,
    verification: VerificationResult,
    extractor_id: str,
    recorded_at: str,
    supersedes: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "claim": asdict(claim),
        "verification": {
            "status": verification.status,
            "errors": list(verification.errors),
            "citations": list(verification.citations),
        },
        "extractor_id": extractor_id,
        "recorded_at": recorded_at,
        "supersedes": list(supersedes),
    }


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _entry_from_json(entry_id: str, encoded: str) -> CandidateEntry:
    payload = json.loads(encoded)
    if type(payload) is not dict or set(payload) != _ENTRY_FIELDS:
        raise ValueError("stored candidate entry is malformed")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if _canonical_json(payload) != encoded or digest != entry_id:
        raise ValueError("stored candidate entry integrity check failed")
    raw_claim = payload["claim"]
    spans = tuple(EvidenceSpan(**span) for span in raw_claim["evidence"])
    claim = Claim(
        subject=raw_claim["subject"],
        predicate=raw_claim["predicate"],
        object=raw_claim["object"],
        speaker_role=raw_claim["speaker_role"],
        attribution=raw_claim["attribution"],
        modality=raw_claim["modality"],
        polarity=raw_claim["polarity"],
        valid_from=raw_claim["valid_from"],
        valid_to=raw_claim["valid_to"],
        evidence=spans,
    )
    raw_verification = payload["verification"]
    status = raw_verification["status"]
    if status not in {"REVIEW", "REJECT"}:
        raise ValueError("stored candidate verification status is invalid")
    verification = VerificationResult(
        status, tuple(raw_verification["errors"]), tuple(raw_verification["citations"])
    )
    return CandidateEntry(
        entry_id,
        claim,
        verification,
        payload["extractor_id"],
        payload["recorded_at"],
        tuple(payload["supersedes"]),
    )


__all__ = ["CandidateEntry", "CandidateLedger"]
