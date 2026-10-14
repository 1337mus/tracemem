"""Tests for the private, append-only candidate proposal ledger."""

from __future__ import annotations

import sqlite3
import stat
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from tracemem.claims import Claim, EvidenceSpan
from tracemem.ledger import _APPLICATION_ID, _SCHEMA_VERSION, CandidateLedger
from tracemem.schema import EvidenceRecord


def _review_pair() -> tuple[Claim, EvidenceRecord]:
    text = "I prefer tea."
    claim = Claim(
        "the user",
        "PREFERS",
        "tea",
        "user",
        "direct_user_statement",
        "preference",
        "positive",
        None,
        None,
        (EvidenceSpan("session", "message", 0, len(text), text),),
    )
    record = EvidenceRecord("synthetic", "session", "message", 0, "user", text)
    return claim, record


class CandidateLedgerPersistenceTests(unittest.TestCase):
    def test_append_computes_verification_and_round_trips_after_reopen(self) -> None:
        claim, record = _review_pair()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "ledger.sqlite3"
            with CandidateLedger(path) as ledger:
                entry = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                self.assertEqual(entry.verification.status, "REVIEW")
                self.assertEqual(entry.verification.errors, ())
                self.assertEqual(
                    entry.verification.citations,
                    (f"{record.record_id}@{record.revision}:0:{len(record.text)}",),
                )
                self.assertEqual(len(entry.entry_id), 64)
                with self.assertRaises(FrozenInstanceError):
                    entry.extractor_id = "changed"  # type: ignore[misc]

            with CandidateLedger(path) as reopened:
                self.assertEqual(reopened.get(entry.entry_id), entry)
                self.assertEqual(reopened.entries(), (entry,))

    def test_identical_append_is_idempotent_and_metadata_changes_identity(self) -> None:
        claim, record = _review_pair()

        with tempfile.TemporaryDirectory() as directory:
            with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                first = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                repeated = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                changed = ledger.append(claim, [record], "extractor-v2", "2026-10-14T10:00:00Z")

                self.assertEqual(repeated, first)
                self.assertNotEqual(changed.entry_id, first.entry_id)
                self.assertEqual(len(ledger.entries()), 2)
            with CandidateLedger(Path(directory) / "second.sqlite3") as other:
                independent = other.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                self.assertEqual(independent.entry_id, first.entry_id)

    def test_pinned_citation_survives_a_later_source_revision(self) -> None:
        claim, original = _review_pair()
        revised = EvidenceRecord(
            original.source,
            original.session_id,
            original.message_id,
            original.part_index,
            original.role,
            original.text + " I also like toast.",
        )
        self.assertEqual(original.record_id, revised.record_id)
        self.assertNotEqual(original.revision, revised.revision)

        with tempfile.TemporaryDirectory() as directory:
            with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                first = ledger.append(claim, [original], "extractor-v1", "2026-10-14T10:00:00Z")
                ledger.append(claim, [revised], "extractor-v1", "2026-10-14T10:01:00Z")

                self.assertEqual(
                    ledger.get(first.entry_id).verification.citations,
                    (f"{original.record_id}@{original.revision}:0:{len(original.text)}",),
                )

    def test_reject_is_preserved_when_later_evidence_supports_a_new_proposal(self) -> None:
        claim, record = _review_pair()

        with tempfile.TemporaryDirectory() as directory:
            with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                rejected = ledger.append(claim, [], "extractor-v1", "2026-10-14T10:00:00Z")
                reviewed = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")

                self.assertEqual(rejected.verification.status, "REJECT")
                self.assertEqual(reviewed.verification.status, "REVIEW")
                self.assertNotEqual(rejected.entry_id, reviewed.entry_id)
                self.assertEqual(ledger.get(rejected.entry_id), rejected)
                self.assertEqual(
                    {entry.entry_id for entry in ledger.entries()},
                    {rejected.entry_id, reviewed.entry_id},
                )

    def test_correction_lineage_preserves_both_immutable_entries(self) -> None:
        claim, record = _review_pair()
        correction = replace(claim, object="coffee")

        with tempfile.TemporaryDirectory() as directory:
            with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                original = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                corrected = ledger.append(
                    correction,
                    [record],
                    "extractor-v2",
                    "2026-10-14T10:01:00Z",
                    supersedes=(original.entry_id,),
                )
                branch = ledger.append(
                    replace(claim, object="matcha"),
                    [record],
                    "extractor-v3",
                    "2026-10-14T10:02:00Z",
                    supersedes=(original.entry_id,),
                )

                self.assertEqual(corrected.supersedes, (original.entry_id,))
                self.assertEqual(branch.supersedes, (original.entry_id,))
                self.assertEqual(ledger.get(original.entry_id), original)
                self.assertEqual(
                    {entry.entry_id for entry in ledger.entries()},
                    {original.entry_id, corrected.entry_id, branch.entry_id},
                )

    def test_unknown_or_duplicate_supersedes_targets_leave_no_partial_entry(self) -> None:
        claim, record = _review_pair()
        correction = replace(claim, object="coffee")

        with tempfile.TemporaryDirectory() as directory:
            with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                original = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                before = ledger.entries()
                with self.assertRaisesRegex(ValueError, "unknown superseded entry"):
                    ledger.append(
                        correction,
                        [record],
                        "extractor-v2",
                        "2026-10-14T10:01:00Z",
                        supersedes=(original.entry_id, "0" * 64),
                    )
                with self.assertRaisesRegex(ValueError, "duplicate entry IDs"):
                    ledger.append(
                        correction,
                        [record],
                        "extractor-v2",
                        "2026-10-14T10:01:00Z",
                        supersedes=(original.entry_id, original.entry_id),
                    )

                self.assertEqual(ledger.entries(), before)

    def test_sql_update_delete_and_replace_cannot_mutate_entries_or_lineage(self) -> None:
        claim, record = _review_pair()
        correction = replace(claim, object="coffee")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite3"
            with CandidateLedger(path) as ledger:
                original = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                corrected = ledger.append(
                    correction,
                    [record],
                    "extractor-v2",
                    "2026-10-14T10:01:00Z",
                    supersedes=(original.entry_id,),
                )
                raw = sqlite3.connect(path)
                try:
                    row = raw.execute(
                        "SELECT payload_json FROM candidate_entries WHERE entry_id = ?",
                        (original.entry_id,),
                    ).fetchone()
                    self.assertIsNotNone(row)
                    payload = row[0]
                    mutations = (
                        (
                            "UPDATE candidate_entries SET payload_json = ? WHERE entry_id = ?",
                            ("{}", original.entry_id),
                        ),
                        (
                            "DELETE FROM candidate_entries WHERE entry_id = ?",
                            (original.entry_id,),
                        ),
                        (
                            "INSERT OR REPLACE INTO candidate_entries(entry_id, payload_json) "
                            "VALUES (?, ?)",
                            (original.entry_id, payload),
                        ),
                        (
                            "UPDATE candidate_lineage SET superseded_id = ? WHERE entry_id = ?",
                            (corrected.entry_id, corrected.entry_id),
                        ),
                        (
                            "DELETE FROM candidate_lineage WHERE entry_id = ?",
                            (corrected.entry_id,),
                        ),
                        (
                            "INSERT OR REPLACE INTO candidate_lineage(entry_id, superseded_id) "
                            "VALUES (?, ?)",
                            (corrected.entry_id, original.entry_id),
                        ),
                    )
                    for sql, parameters in mutations:
                        with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                            raw.execute(sql, parameters)
                finally:
                    raw.close()

                self.assertEqual(ledger.get(original.entry_id), original)
                self.assertEqual(ledger.get(corrected.entry_id), corrected)

    def test_alien_database_is_rejected_without_changing_its_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alien.sqlite3"
            connection = sqlite3.connect(path)
            try:
                connection.execute("CREATE TABLE sentinel(value TEXT)")
                connection.execute("INSERT INTO sentinel VALUES ('leave unchanged')")
                connection.commit()
            finally:
                connection.close()
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "supported candidate ledger"):
                CandidateLedger(path)

            self.assertEqual(path.read_bytes(), before)

    def test_database_with_ledger_id_but_wrong_schema_is_rejected_read_only(self) -> None:
        trigger_specs = (
            ("candidate_entries_no_duplicate", "INSERT", "candidate_entries"),
            ("candidate_entries_no_update", "UPDATE", "candidate_entries"),
            ("candidate_entries_no_delete", "DELETE", "candidate_entries"),
            ("candidate_lineage_no_duplicate", "INSERT", "candidate_lineage"),
            ("candidate_lineage_no_update", "UPDATE", "candidate_lineage"),
            ("candidate_lineage_no_delete", "DELETE", "candidate_lineage"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spoof.sqlite3"
            connection = sqlite3.connect(path)
            try:
                connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.execute("CREATE TABLE candidate_entries(wrong TEXT)")
                connection.execute("CREATE TABLE candidate_lineage(wrong TEXT)")
                for name, event, table in trigger_specs:
                    connection.execute(
                        f"CREATE TRIGGER {name} BEFORE {event} ON {table} BEGIN SELECT 1; END"
                    )
                connection.commit()
            finally:
                connection.close()
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "supported candidate ledger"):
                CandidateLedger(path)

            self.assertEqual(path.read_bytes(), before)

    def test_new_paths_are_private_and_database_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "private" / "ledger"
            path = nested / "candidate.sqlite3"
            with CandidateLedger(path):
                self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            target = Path(directory) / "target.sqlite3"
            target.write_bytes(b"do not replace")
            link = Path(directory) / "linked.sqlite3"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                CandidateLedger(link)
            self.assertEqual(target.read_bytes(), b"do not replace")
            directory_path = Path(directory) / "existing-directory"
            directory_path.mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                CandidateLedger(directory_path)

    def test_invalid_extractor_timestamp_and_supersedes_metadata_are_rejected(self) -> None:
        claim, record = _review_pair()

        with tempfile.TemporaryDirectory() as directory:
            with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                invalid_calls = (
                    {"extractor_id": " "},
                    {"recorded_at": "2026-10-14T10:00:00+00:00"},
                    {"recorded_at": "2026-10-14T10:00:00.000000Z"},
                    {"supersedes": []},
                )
                for changes in invalid_calls:
                    arguments: dict[str, object] = {
                        "extractor_id": "extractor-v1",
                        "recorded_at": "2026-10-14T10:00:00Z",
                        "supersedes": (),
                    }
                    arguments.update(changes)
                    with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                        ledger.append(
                            claim,
                            [record],
                            arguments["extractor_id"],  # type: ignore[arg-type]
                            arguments["recorded_at"],  # type: ignore[arg-type]
                            arguments["supersedes"],  # type: ignore[arg-type]
                        )
                self.assertEqual(ledger.entries(), ())


if __name__ == "__main__":
    unittest.main()
