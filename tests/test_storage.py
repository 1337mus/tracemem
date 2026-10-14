import hashlib
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tracemem


class EvidenceRecordTests(unittest.TestCase):
    def test_record_id_stays_stable_when_evidence_text_is_edited(self) -> None:
        record_type = getattr(tracemem, "EvidenceRecord", None)
        self.assertIsNotNone(record_type, "EvidenceRecord is part of the public API")

        original = record_type(
            source="claude",
            session_id="session-1",
            message_id="message-1",
            part_index=0,
            role="user",
            text="old wording",
        )
        edited = record_type(
            source="claude",
            session_id="session-1",
            message_id="message-1",
            part_index=0,
            role="user",
            text="new wording",
        )

        self.assertEqual(original.record_id, edited.record_id)

    def test_revision_changes_when_text_changes(self) -> None:
        record_type = getattr(tracemem, "EvidenceRecord", None)
        self.assertIsNotNone(record_type, "EvidenceRecord is part of the public API")
        original = record_type("codex", "session-2", "message-2", 0, "assistant", "exact text 🧪")
        edited = record_type("codex", "session-2", "message-2", 0, "assistant", "changed text 🧪")

        self.assertNotEqual(original.revision, edited.revision)


class EvidenceStoreTests(unittest.TestCase):
    def test_fts_rowid_delete_uses_virtual_table_rowid_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                rowid_plan = store._connection.execute(
                    "EXPLAIN QUERY PLAN DELETE FROM evidence_fts WHERE rowid = ?", (7,)
                ).fetchall()
                record_id_plan = store._connection.execute(
                    "EXPLAIN QUERY PLAN DELETE FROM evidence_fts WHERE record_id = ?", ("id",)
                ).fetchall()

        rowid_details = [row["detail"] for row in rowid_plan]
        record_id_details = [row["detail"] for row in record_id_plan]
        self.assertTrue(any("VIRTUAL TABLE INDEX" in detail for detail in rowid_details))
        self.assertTrue(any(detail.endswith(":=") for detail in rowid_details))
        self.assertTrue(any("VIRTUAL TABLE INDEX" in detail for detail in record_id_details))
        self.assertTrue(all(not detail.endswith(":=") for detail in record_id_details))

    def test_lexical_search_uses_current_unicode_revision(self) -> None:
        old = tracemem.EvidenceRecord(
            "synthetic", "session-unicode", "message-1", 0, "user", "Crème brûlée legacy"
        )
        current = tracemem.EvidenceRecord(
            "synthetic",
            "session-unicode",
            "message-1",
            0,
            "user",
            "Crème brûlée 東京 current",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                store.ingest([old])
                store.ingest([current])

                unicode_hits = store.lexical_search("brûlée 東京")
                obsolete_hits = store.lexical_search("legacy")
                exact_old_revision = store.get(old.record_id, old.revision)

        self.assertEqual([record for record, _ in unicode_hits], [current])
        self.assertEqual(obsolete_hits, [])
        self.assertEqual(exact_old_revision, old)

    def test_schema_v2_database_is_rejected_without_writes_or_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "evidence.sqlite3"
            with tracemem.EvidenceStore(database_path) as store:
                store.ingest(
                    [
                        tracemem.EvidenceRecord(
                            "synthetic", "session-schema-v2", "message-1", 0, "user", "keep me"
                        )
                    ]
                )
            with sqlite3.connect(database_path) as connection:
                connection.execute("PRAGMA user_version = 2")
            before = hashlib.sha256(database_path.read_bytes()).digest()
            sidecars_before = sorted(
                path.name for path in database_path.parent.glob("evidence.sqlite3-*")
            )

            with self.assertRaisesRegex(ValueError, "supported TraceMem evidence database"):
                tracemem.EvidenceStore(database_path)

            self.assertEqual(hashlib.sha256(database_path.read_bytes()).digest(), before)
            sidecars_after = sorted(
                path.name for path in database_path.parent.glob("evidence.sqlite3-*")
            )
            self.assertEqual(sidecars_after, sidecars_before)

    def test_schema_without_head_delete_trigger_is_rejected_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "evidence.sqlite3"
            with tracemem.EvidenceStore(database_path):
                pass
            with sqlite3.connect(database_path) as connection:
                connection.execute("DROP TRIGGER evidence_head_fts_delete")
            before = hashlib.sha256(database_path.read_bytes()).digest()

            with self.assertRaisesRegex(ValueError, "supported TraceMem evidence database"):
                tracemem.EvidenceStore(database_path)

            self.assertEqual(hashlib.sha256(database_path.read_bytes()).digest(), before)
            self.assertEqual(list(database_path.parent.glob("evidence.sqlite3-*")), [])

    def test_fts_removes_deleted_head_and_reinsert_uses_new_rowid(self) -> None:
        removed = tracemem.EvidenceRecord(
            "synthetic", "session-fts-delete", "message-a", 0, "user", "obsolete marker"
        )
        survivor = tracemem.EvidenceRecord(
            "synthetic", "session-fts-delete", "message-b", 0, "user", "survivor marker"
        )
        reinserted = tracemem.EvidenceRecord(
            "synthetic", "session-fts-delete", "message-a", 0, "user", "current marker"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                store.ingest([removed, survivor])
                old_rowid = store._connection.execute(
                    "SELECT rowid FROM evidence_heads WHERE record_id = ?",
                    (removed.record_id,),
                ).fetchone()[0]
                survivor_rowid = store._connection.execute(
                    "SELECT rowid FROM evidence_heads WHERE record_id = ?",
                    (survivor.record_id,),
                ).fetchone()[0]
                store._connection.execute(
                    "DELETE FROM evidence_heads WHERE record_id = ?", (removed.record_id,)
                )
                store.ingest([reinserted])

                heads = {
                    row["record_id"]: row["rowid"]
                    for row in store._connection.execute(
                        "SELECT rowid, record_id FROM evidence_heads"
                    )
                }
                fts = {
                    row["record_id"]: (row["rowid"], row["revision"], row["text"])
                    for row in store._connection.execute(
                        "SELECT rowid, record_id, revision, text FROM evidence_fts"
                    )
                }
                old_text_hits = store._connection.execute(
                    "SELECT rowid FROM evidence_fts WHERE evidence_fts MATCH 'obsolete'"
                ).fetchall()

        self.assertNotEqual(old_rowid, heads[removed.record_id])
        self.assertEqual(len(fts), 2)
        self.assertEqual(
            fts[removed.record_id], (heads[removed.record_id], reinserted.revision, reinserted.text)
        )
        self.assertEqual(
            fts[survivor.record_id], (survivor_rowid, survivor.revision, survivor.text)
        )
        self.assertEqual(old_text_hits, [])

    def test_fts_rowid_tracks_head_rowid_when_an_earlier_head_is_updated(self) -> None:
        first = tracemem.EvidenceRecord(
            "synthetic", "session-fts-rowid", "message-a", 0, "user", "first version"
        )
        later = tracemem.EvidenceRecord(
            "synthetic", "session-fts-rowid", "message-b", 0, "user", "later head"
        )
        updated = tracemem.EvidenceRecord(
            "synthetic", "session-fts-rowid", "message-a", 0, "user", "updated version"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                store.ingest([first, later])
                store.ingest([updated])

                heads = {
                    row["record_id"]: row["rowid"]
                    for row in store._connection.execute(
                        "SELECT rowid, record_id FROM evidence_heads"
                    )
                }
                self.assertLess(heads[first.record_id], heads[later.record_id])
                fts = {
                    row["record_id"]: (row["rowid"], row["revision"], row["text"])
                    for row in store._connection.execute(
                        "SELECT rowid, record_id, revision, text FROM evidence_fts"
                    )
                }
                obsolete_hits = store._connection.execute(
                    "SELECT rowid FROM evidence_fts WHERE evidence_fts MATCH 'first'"
                ).fetchall()
                exact_old_revision = store.get(first.record_id, first.revision)
                current_hits = store.lexical_search("updated")

        self.assertEqual(len(fts), 2)
        self.assertEqual(
            fts[first.record_id], (heads[first.record_id], updated.revision, updated.text)
        )
        self.assertEqual(fts[later.record_id], (heads[later.record_id], later.revision, later.text))
        self.assertEqual(obsolete_hits, [])
        self.assertEqual(exact_old_revision, first)
        self.assertEqual([record for record, _ in current_hits], [updated])

    def test_repeated_ingest_is_idempotent(self) -> None:
        store_type = getattr(tracemem, "EvidenceStore", None)
        self.assertIsNotNone(store_type, "EvidenceStore is part of the public API")
        record = tracemem.EvidenceRecord(
            "synthetic", "session-3", "message-3", 0, "user", "Keep the exact words."
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "evidence.sqlite3"
            with store_type(database_path) as store:
                self.assertEqual(store.ingest([record]), 1)
                self.assertEqual(store.ingest([record]), 0)
                self.assertEqual(store.records(), [record])

    def test_vector_cache_is_bound_to_the_source_revision_and_fingerprint(self) -> None:
        store_type = getattr(tracemem, "EvidenceStore", None)
        self.assertIsNotNone(store_type, "EvidenceStore is part of the public API")
        record = tracemem.EvidenceRecord(
            "synthetic", "session-4", "message-4", 0, "assistant", "A vector cache."
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with store_type(Path(temporary_directory) / "evidence.sqlite3") as store:
                store.ingest([record])
                store.cache_vector(record.record_id, record.revision, "model-v1", [1.0, -0.25])

                self.assertEqual(
                    store.vector_cache(record.record_id, record.revision, "model-v1"),
                    [1.0, -0.25],
                )
                self.assertIsNone(store.vector_cache(record.record_id, record.revision, "model-v2"))

    def test_lexical_search_returns_higher_score_for_the_stronger_match(self) -> None:
        store_type = getattr(tracemem, "EvidenceStore", None)
        self.assertIsNotNone(store_type, "EvidenceStore is part of the public API")
        exact = tracemem.EvidenceRecord(
            "synthetic", "session-5", "message-5a", 0, "user", "The zero copy cache is enabled."
        )
        partial = tracemem.EvidenceRecord(
            "synthetic", "session-5", "message-5b", 0, "assistant", "The cache is local."
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with store_type(Path(temporary_directory) / "evidence.sqlite3") as store:
                store.ingest([exact, partial])

                hits = store.lexical_search("copy cache", limit=10)

        self.assertEqual([record for record, _ in hits], [exact, partial])
        self.assertGreater(hits[0][1], hits[1][1])

    def test_reobserving_an_earlier_revision_makes_it_current_again(self) -> None:
        record_type = tracemem.EvidenceRecord
        original = record_type("synthetic", "session-6", "message-6", 0, "user", "original wording")
        edited = record_type("synthetic", "session-6", "message-6", 0, "user", "edited wording")
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                self.assertEqual(store.ingest([original]), 1)
                self.assertEqual(store.ingest([edited]), 1)
                self.assertEqual(store.ingest([original]), 0)

                self.assertEqual(store.records(), [original])
                self.assertCountEqual(store.records(current_only=False), [original, edited])

    def test_unknown_database_is_preflighted_read_only_and_left_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "unrelated.sqlite3"
            unrelated = sqlite3.connect(database_path)
            try:
                unrelated.execute("CREATE TABLE unrelated_table(value TEXT)")
                unrelated.execute("INSERT INTO unrelated_table VALUES ('keep me')")
            finally:
                unrelated.close()
            before = hashlib.sha256(database_path.read_bytes()).digest()
            real_connect = sqlite3.connect
            calls: list[tuple[str, bool]] = []

            def connect_spy(
                database: object, *args: object, **kwargs: object
            ) -> sqlite3.Connection:
                calls.append((str(database), bool(kwargs.get("uri", False))))
                return real_connect(database, *args, **kwargs)

            with patch("tracemem.storage.sqlite3.connect", side_effect=connect_spy):
                with self.assertRaises(ValueError):
                    tracemem.EvidenceStore(database_path)

            after = hashlib.sha256(database_path.read_bytes()).digest()

        self.assertEqual(after, before)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1])
        self.assertIn("mode=ro", calls[0][0])

    def test_new_database_and_parent_have_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory) / "private" / "nested"
            database_path = parent / "evidence.sqlite3"

            with tracemem.EvidenceStore(database_path):
                self.assertEqual(stat.S_IMODE(database_path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(parent.parent.stat().st_mode), 0o700)

    def test_symlink_database_path_is_rejected_without_changing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "unrelated.sqlite3"
            unrelated = sqlite3.connect(target)
            try:
                unrelated.execute("CREATE TABLE unrelated_table(value TEXT)")
                unrelated.execute("INSERT INTO unrelated_table VALUES ('preserve')")
            finally:
                unrelated.close()
            link = Path(temporary_directory) / "evidence.sqlite3"
            link.symlink_to(target)
            before = hashlib.sha256(target.read_bytes()).digest()

            with self.assertRaisesRegex(ValueError, "symlink"):
                tracemem.EvidenceStore(link)

            self.assertEqual(hashlib.sha256(target.read_bytes()).digest(), before)

    def test_ingest_rolls_back_every_record_if_iteration_fails(self) -> None:
        first = tracemem.EvidenceRecord("synthetic", "session-7", "message-7a", 0, "user", "first")
        second = tracemem.EvidenceRecord(
            "synthetic", "session-7", "message-7b", 0, "assistant", "second"
        )

        def failing_records():
            yield first
            yield second
            raise RuntimeError("synthetic iterator failure")

        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                with self.assertRaisesRegex(RuntimeError, "synthetic iterator failure"):
                    store.ingest(failing_records())

                self.assertEqual(store.records(current_only=False), [])

    def test_ingest_rolls_back_valid_records_before_an_invalid_value(self) -> None:
        valid = tracemem.EvidenceRecord(
            "synthetic", "session-rollback", "message-1", 0, "user", "valid first record"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                with self.assertRaisesRegex(TypeError, "EvidenceRecord values only"):
                    store.ingest([valid, object()])

                self.assertEqual(store.records(current_only=False), [])

    def test_evidence_rows_reject_update_delete_and_replace(self) -> None:
        record = tracemem.EvidenceRecord(
            "synthetic", "session-8", "message-8", 0, "user", "immutable text"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with tracemem.EvidenceStore(Path(temporary_directory) / "evidence.sqlite3") as store:
                store.ingest([record])
                with self.assertRaises(sqlite3.IntegrityError):
                    store._connection.execute(
                        "UPDATE evidence SET text = ? WHERE record_id = ?",
                        ("changed text", record.record_id),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    store._connection.execute(
                        "DELETE FROM evidence WHERE record_id = ?", (record.record_id,)
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    store._connection.execute(
                        """
                        INSERT OR REPLACE INTO evidence (
                            record_id, revision, source, session_id, message_id, part_index,
                            role, text, recorded_at, source_locator, line_number, content_kind
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.record_id,
                            record.revision,
                            record.source,
                            record.session_id,
                            record.message_id,
                            record.part_index,
                            record.role,
                            "replaced text",
                            record.recorded_at,
                            record.source_locator,
                            record.line_number,
                            record.content_kind,
                        ),
                    )

                self.assertEqual(store.get(record.record_id, record.revision), record)

    def test_evidence_rejects_rowid_replace_from_connection_with_recursive_triggers_off(
        self,
    ) -> None:
        version_one = tracemem.EvidenceRecord(
            "synthetic", "session-rowid", "message-rowid", 0, "user", "version one"
        )
        version_two = tracemem.EvidenceRecord(
            "synthetic", "session-rowid", "message-rowid", 0, "user", "version two"
        )
        replacement = tracemem.EvidenceRecord(
            "synthetic", "session-rowid", "different-message", 0, "user", "replacement"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "evidence.sqlite3"
            with tracemem.EvidenceStore(database_path) as store:
                store.ingest([version_one])
                store.ingest([version_two])
                head = store._connection.execute(
                    "SELECT revision FROM evidence_heads WHERE record_id = ?",
                    (version_one.record_id,),
                ).fetchone()
                self.assertEqual(head["revision"], version_two.revision)
                self.assertIsNone(
                    store._connection.execute(
                        "SELECT 1 FROM evidence_vectors WHERE record_id = ? AND revision = ?",
                        (version_one.record_id, version_one.revision),
                    ).fetchone()
                )

                raw_connection = sqlite3.connect(database_path)
                try:
                    raw_connection.execute("PRAGMA recursive_triggers = OFF")
                    self.assertEqual(
                        raw_connection.execute("PRAGMA recursive_triggers").fetchone()[0], 0
                    )
                    with self.assertRaises(sqlite3.DatabaseError):
                        raw_connection.execute(
                            """
                            INSERT OR REPLACE INTO evidence (
                                rowid, record_id, revision, source, session_id, message_id,
                                part_index, role, text, recorded_at, source_locator,
                                line_number, content_kind
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                1,
                                replacement.record_id,
                                replacement.revision,
                                replacement.source,
                                replacement.session_id,
                                replacement.message_id,
                                replacement.part_index,
                                replacement.role,
                                replacement.text,
                                replacement.recorded_at,
                                replacement.source_locator,
                                replacement.line_number,
                                replacement.content_kind,
                            ),
                        )
                    raw_connection.commit()
                finally:
                    raw_connection.close()

                self.assertEqual(
                    store.get(version_one.record_id, version_one.revision), version_one
                )
                self.assertEqual(
                    store.get(version_two.record_id, version_two.revision), version_two
                )
                with self.assertRaises(KeyError):
                    store.get(replacement.record_id, replacement.revision)


if __name__ == "__main__":
    unittest.main()
