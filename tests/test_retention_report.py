"""Persisted-store retention report tests with fictional evidence only."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from tracemem.claims import Claim, EvidenceSpan
from tracemem.ledger import CandidateLedger
from tracemem.retention import plan_source_erasure
from tracemem.retention_report import inspect_source_retention
from tracemem.schema import EvidenceRecord
from tracemem.storage import EvidenceStore


def _record(text: str = "I prefer fictional café tea.") -> EvidenceRecord:
    return EvidenceRecord(
        source="fictional-provider",
        session_id="fictional-session",
        message_id="fictional-message",
        part_index=0,
        role="user",
        text=text,
        content_kind="conversation",
    )


def _claim(record: EvidenceRecord) -> Claim:
    return Claim(
        "fictional user",
        "PREFERS",
        "fictional café tea",
        "user",
        "direct_user_statement",
        "preference",
        "positive",
        None,
        None,
        (
            EvidenceSpan(
                record.session_id,
                record.message_id,
                0,
                len(record.text),
                record.text,
            ),
        ),
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[1]
    source_root = repository / "src"
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) + (
        os.pathsep + prior_pythonpath if prior_pythonpath else ""
    )
    return subprocess.run(
        [sys.executable, "-m", "tracemem.retention_report", *arguments],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class PersistedRetentionReportTests(unittest.TestCase):
    def test_reports_hashes_counts_and_the_existing_planner_without_changing_inputs(self) -> None:
        record = _record()

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.sqlite3"
            ledger_path = Path(directory) / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path) as ledger:
                entry = ledger.append(
                    _claim(record),
                    [record],
                    "fictional-extractor",
                    "2026-10-14T10:00:00Z",
                )

            os.chmod(evidence_path, 0o640)
            os.chmod(ledger_path, 0o600)
            stable_mtime_ns = 1_700_000_000_000_000_000
            os.utime(evidence_path, ns=(stable_mtime_ns, stable_mtime_ns))
            os.utime(ledger_path, ns=(stable_mtime_ns, stable_mtime_ns))
            evidence_before = evidence_path.read_bytes()
            ledger_before = ledger_path.read_bytes()
            evidence_stat_before = evidence_path.stat()
            ledger_stat_before = ledger_path.stat()
            evidence_hash = hashlib.sha256(evidence_before).hexdigest()
            ledger_hash = hashlib.sha256(ledger_before).hexdigest()
            with EvidenceStore(evidence_path) as store, CandidateLedger(ledger_path) as ledger:
                records = store.records(current_only=False)
                entries = ledger.entries()
            expected_impact = asdict(
                plan_source_erasure(
                    records,
                    entries,
                    source="fictional-provider",
                    session_id="fictional-session",
                )
            )

            report = inspect_source_retention(
                str(evidence_path),
                str(ledger_path),
                source="fictional-provider",
                session_id="fictional-session",
            )

            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["evidence_revision_count"], 1)
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(
                report["source_snapshots"],
                {
                    "evidence": {"sha256": evidence_hash, "bytes": len(evidence_before)},
                    "ledger": {"sha256": ledger_hash, "bytes": len(ledger_before)},
                },
            )
            self.assertEqual(report["impact"], expected_impact)
            self.assertIn(entry.entry_id, report["impact"]["direct_candidate_ids"])
            limitations = " ".join(report["limitations"]).casefold()
            self.assertIn("fts", limitations)
            self.assertIn("logical vector-cache contents and counts are not certified", limitations)
            serialized = json.dumps(report, ensure_ascii=False)
            for private_text in (
                "fictional-provider",
                "fictional-session",
                "fictional café tea",
                record.text,
                str(evidence_path),
                str(ledger_path),
            ):
                self.assertNotIn(private_text, serialized)
            self.assertEqual(evidence_path.read_bytes(), evidence_before)
            self.assertEqual(ledger_path.read_bytes(), ledger_before)
            self.assertEqual(
                evidence_path.stat().st_mode & 0o777, evidence_stat_before.st_mode & 0o777
            )
            self.assertEqual(ledger_path.stat().st_mode & 0o777, ledger_stat_before.st_mode & 0o777)
            self.assertEqual(evidence_path.stat().st_mtime_ns, evidence_stat_before.st_mtime_ns)
            self.assertEqual(ledger_path.stat().st_mtime_ns, ledger_stat_before.st_mtime_ns)

    def test_detects_source_or_ledger_change_during_analysis(self) -> None:
        record = _record()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path) as ledger:
                ledger.append(
                    _claim(record),
                    [record],
                    "fictional-extractor",
                    "2026-10-14T10:00:00Z",
                )

            for changed_path in (evidence_path, ledger_path):
                with self.subTest(changed_path=changed_path.name):
                    planner = plan_source_erasure
                    original_bytes = changed_path.read_bytes()

                    def mutate_then_plan(
                        *args: object,
                        _changed_path: Path = changed_path,
                        _planner=planner,
                        **kwargs: object,
                    ) -> object:
                        _changed_path.write_bytes(_changed_path.read_bytes() + b"x")
                        return _planner(*args, **kwargs)  # type: ignore[operator]

                    try:
                        with (
                            patch(
                                "tracemem.retention_report.plan_source_erasure",
                                side_effect=mutate_then_plan,
                            ),
                            self.assertRaises(ValueError) as caught,
                        ):
                            inspect_source_retention(
                                str(evidence_path),
                                str(ledger_path),
                                source="fictional-provider",
                                session_id="fictional-session",
                            )
                    finally:
                        changed_path.write_bytes(original_bytes)
                    self.assertNotIn(str(changed_path), str(caught.exception))

    def test_database_opens_are_memory_only_and_store_writers_are_not_constructed(self) -> None:
        record = _record()

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.sqlite3"
            ledger_path = Path(directory) / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path) as ledger:
                ledger.append(
                    _claim(record),
                    [record],
                    "fictional-extractor",
                    "2026-10-14T10:00:00Z",
                )

            original_connect = sqlite3.connect
            opened: list[object] = []

            def connect_memory_only(
                database: object, *args: object, **kwargs: object
            ) -> sqlite3.Connection:
                opened.append(database)
                if database != ":memory:":
                    raise AssertionError("SQLite must only open in-memory copies")
                return original_connect(database, *args, **kwargs)  # type: ignore[arg-type]

            with (
                patch("tracemem.retention_report.sqlite3.connect", side_effect=connect_memory_only),
                patch.object(EvidenceStore, "__init__", side_effect=AssertionError("writer used")),
                patch.object(
                    CandidateLedger, "__init__", side_effect=AssertionError("writer used")
                ),
            ):
                inspect_source_retention(
                    str(evidence_path),
                    str(ledger_path),
                    source="fictional-provider",
                    session_id="fictional-session",
                )
            self.assertEqual(opened, [":memory:", ":memory:"])

    def test_validates_selectors_before_opening_either_snapshot(self) -> None:
        with patch("tracemem.retention_report._read_snapshot") as read_snapshot:
            with self.assertRaises(ValueError) as caught:
                inspect_source_retention(
                    "/private/provider/path.sqlite3",
                    "/private/ledger/path.sqlite3",
                    source="private-selector",
                    session_id="",
                )
            read_snapshot.assert_not_called()
        self.assertNotIn("private", str(caught.exception))

    def test_rejects_schema_swaps_alien_corrupt_and_generated_column_inputs(self) -> None:
        record = _record()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path):
                pass

            alien_path = root / "alien.sqlite3"
            with sqlite3.connect(alien_path) as connection:
                connection.execute("CREATE TABLE unrelated(value TEXT)")
            corrupt_path = root / "corrupt.sqlite3"
            corrupt_path.write_bytes(b"fictional corrupt database")
            generated_path = root / "generated-column.sqlite3"
            generated_path.write_bytes(evidence_path.read_bytes())
            with sqlite3.connect(generated_path) as connection:
                connection.execute(
                    "ALTER TABLE evidence ADD COLUMN synthetic_secret TEXT "
                    "GENERATED ALWAYS AS ('fictional generated text') VIRTUAL"
                )

            invalid_pairs = (
                (ledger_path, evidence_path),
                (alien_path, ledger_path),
                (corrupt_path, ledger_path),
                (generated_path, ledger_path),
            )
            for evidence_input, ledger_input in invalid_pairs:
                with (
                    self.subTest(input=evidence_input.name),
                    self.assertRaises(ValueError) as caught,
                ):
                    inspect_source_retention(
                        str(evidence_input),
                        str(ledger_input),
                        source="fictional-provider",
                        session_id="fictional-session",
                    )
                self.assertEqual(
                    str(caught.exception), "retention snapshots could not be inspected"
                )

    def test_rejects_record_identity_payload_hash_and_sql_lineage_mismatches(self) -> None:
        record = _record()
        claim = _claim(record)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path) as ledger:
                parent = ledger.append(
                    claim, [record], "fictional-extractor", "2026-10-14T10:00:00Z"
                )
                child = ledger.append(
                    replace(claim, object="fictional revised tea"),
                    [record],
                    "fictional-extractor",
                    "2026-10-14T10:01:00Z",
                    supersedes=(parent.entry_id,),
                )

            evidence_tampered = root / "identity-mismatch.sqlite3"
            evidence_tampered.write_bytes(evidence_path.read_bytes())
            with sqlite3.connect(evidence_tampered) as connection:
                connection.execute("DROP TRIGGER evidence_immutable_update")
                connection.execute("UPDATE evidence SET text = 'tampered fictional text'")
                connection.execute(
                    "CREATE TRIGGER evidence_immutable_update BEFORE UPDATE ON evidence "
                    "BEGIN SELECT RAISE(ABORT, 'evidence rows are immutable'); END"
                )

            payload_tampered = root / "payload-mismatch.sqlite3"
            payload_tampered.write_bytes(ledger_path.read_bytes())
            with sqlite3.connect(payload_tampered) as connection:
                connection.execute("DROP TRIGGER candidate_entries_no_update")
                connection.execute(
                    "UPDATE candidate_entries SET payload_json = '{}' WHERE entry_id = ?",
                    (parent.entry_id,),
                )
                connection.execute(
                    "CREATE TRIGGER candidate_entries_no_update BEFORE UPDATE ON "
                    "candidate_entries BEGIN SELECT RAISE(ABORT, "
                    "'candidate entries are immutable'); END"
                )

            lineage_tampered = root / "lineage-mismatch.sqlite3"
            lineage_tampered.write_bytes(ledger_path.read_bytes())
            with sqlite3.connect(lineage_tampered) as connection:
                connection.execute("DROP TRIGGER candidate_lineage_no_delete")
                connection.execute("DELETE FROM candidate_lineage")
                connection.execute(
                    "CREATE TRIGGER candidate_lineage_no_delete BEFORE DELETE ON "
                    "candidate_lineage BEGIN SELECT RAISE(ABORT, "
                    "'candidate lineage is immutable'); END"
                )

            for evidence_input, ledger_input in (
                (evidence_tampered, ledger_path),
                (evidence_path, payload_tampered),
                (evidence_path, lineage_tampered),
            ):
                with self.subTest(input=ledger_input.name), self.assertRaises(ValueError) as caught:
                    inspect_source_retention(
                        str(evidence_input),
                        str(ledger_input),
                        source="fictional-provider",
                        session_id="fictional-session",
                    )
                self.assertEqual(
                    str(caught.exception), "retention snapshots could not be inspected"
                )
            self.assertEqual(child.supersedes, (parent.entry_id,))

    def test_checks_foreign_keys_for_both_in_memory_databases(self) -> None:
        record = _record()
        claim = _claim(record)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path) as ledger:
                parent = ledger.append(
                    claim, [record], "fictional-extractor", "2026-10-14T10:00:00Z"
                )
                ledger.append(
                    replace(claim, object="fictional child"),
                    [record],
                    "fictional-extractor",
                    "2026-10-14T10:01:00Z",
                    supersedes=(parent.entry_id,),
                )

            evidence_orphan = root / "evidence-orphan.sqlite3"
            evidence_orphan.write_bytes(evidence_path.read_bytes())
            with sqlite3.connect(evidence_orphan) as connection:
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute("DROP TRIGGER evidence_immutable_delete")
                connection.execute("DELETE FROM evidence")
                connection.execute(
                    "CREATE TRIGGER evidence_immutable_delete BEFORE DELETE ON evidence "
                    "BEGIN SELECT RAISE(ABORT, 'evidence rows are immutable'); END"
                )

            ledger_orphan = root / "ledger-orphan.sqlite3"
            ledger_orphan.write_bytes(ledger_path.read_bytes())
            with sqlite3.connect(ledger_orphan) as connection:
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute("DROP TRIGGER candidate_entries_no_delete")
                connection.execute(
                    "DELETE FROM candidate_entries WHERE entry_id = ?", (parent.entry_id,)
                )
                connection.execute(
                    "CREATE TRIGGER candidate_entries_no_delete BEFORE DELETE ON "
                    "candidate_entries BEGIN SELECT RAISE(ABORT, "
                    "'candidate entries are immutable'); END"
                )

            for evidence_input, ledger_input in (
                (evidence_orphan, ledger_path),
                (evidence_path, ledger_orphan),
            ):
                with self.subTest(input=evidence_input.name), self.assertRaises(ValueError):
                    inspect_source_retention(
                        str(evidence_input),
                        str(ledger_input),
                        source="fictional-provider",
                        session_id="fictional-session",
                    )

    def test_cli_creates_exclusive_private_report_without_touching_inputs(self) -> None:
        record = _record()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            output_path = root / "report.json"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path) as ledger:
                ledger.append(
                    _claim(record),
                    [record],
                    "fictional-extractor",
                    "2026-10-14T10:00:00Z",
                )
            evidence_before = evidence_path.read_bytes()
            ledger_before = ledger_path.read_bytes()
            evidence_metadata = evidence_path.stat()
            ledger_metadata = ledger_path.stat()

            result = _run_cli(
                "--evidence-db",
                str(evidence_path),
                "--ledger-db",
                str(ledger_path),
                "--source",
                "fictional-provider",
                "--session-id",
                "fictional-session",
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            report_bytes = output_path.read_bytes()
            report = json.loads(report_bytes)
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(output_path.stat().st_mode & 0o777, 0o600)
            for private_text in (
                "fictional-provider",
                "fictional-session",
                record.text,
                str(evidence_path),
                str(ledger_path),
            ):
                self.assertNotIn(private_text, report_bytes.decode("utf-8"))
            self.assertEqual(evidence_path.read_bytes(), evidence_before)
            self.assertEqual(ledger_path.read_bytes(), ledger_before)
            self.assertEqual(
                evidence_path.stat().st_mode & 0o777, evidence_metadata.st_mode & 0o777
            )
            self.assertEqual(ledger_path.stat().st_mode & 0o777, ledger_metadata.st_mode & 0o777)
            self.assertEqual(evidence_path.stat().st_mtime_ns, evidence_metadata.st_mtime_ns)
            self.assertEqual(ledger_path.stat().st_mtime_ns, ledger_metadata.st_mtime_ns)

            second_result = _run_cli(
                "--evidence-db",
                str(evidence_path),
                "--ledger-db",
                str(ledger_path),
                "--source",
                "fictional-provider",
                "--session-id",
                "fictional-session",
                "--output",
                str(output_path),
            )
            self.assertNotEqual(second_result.returncode, 0)
            self.assertEqual(output_path.read_bytes(), report_bytes)
            for private_text in (str(output_path), "fictional-provider", "fictional-session"):
                self.assertNotIn(private_text, second_result.stderr)

    def test_cli_rejects_invalid_and_reserved_output_destinations_before_creation(self) -> None:
        record = _record()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([record])
            with CandidateLedger(ledger_path):
                pass

            invalid_output = root / "invalid-input-report.json"
            invalid_result = _run_cli(
                "--evidence-db",
                str(evidence_path),
                "--ledger-db",
                str(evidence_path),
                "--source",
                "private-provider",
                "--session-id",
                "private-session",
                "--output",
                str(invalid_output),
            )
            self.assertNotEqual(invalid_result.returncode, 0)
            self.assertFalse(invalid_output.exists())
            for private_text in (str(evidence_path), "private-provider", "private-session"):
                self.assertNotIn(private_text, invalid_result.stderr)

            alias_parent = root / "parent-alias"
            alias_parent.symlink_to(root, target_is_directory=True)
            reserved_output = alias_parent / "evidence.sqlite3-wal"
            reserved_result = _run_cli(
                "--evidence-db",
                str(evidence_path),
                "--ledger-db",
                str(ledger_path),
                "--source",
                "fictional-provider",
                "--session-id",
                "fictional-session",
                "--output",
                str(reserved_output),
            )
            self.assertNotEqual(reserved_result.returncode, 0)
            self.assertFalse(reserved_output.exists())
            self.assertFalse(Path(f"{evidence_path}-wal").exists())

            case_aliased_sidecar = root / "evidence.sqlite3-WAL"
            case_aliased_result = _run_cli(
                "--evidence-db",
                str(evidence_path),
                "--ledger-db",
                str(ledger_path),
                "--source",
                "fictional-provider",
                "--session-id",
                "fictional-session",
                "--output",
                str(case_aliased_sidecar),
            )
            self.assertNotEqual(case_aliased_result.returncode, 0)
            self.assertFalse(case_aliased_sidecar.exists())
            self.assertNotIn(str(case_aliased_sidecar), case_aliased_result.stderr)

            output_symlink = root / "report-link.json"
            output_symlink.symlink_to(root / "not-created.json")
            link_result = _run_cli(
                "--evidence-db",
                str(evidence_path),
                "--ledger-db",
                str(ledger_path),
                "--source",
                "fictional-provider",
                "--session-id",
                "fictional-session",
                "--output",
                str(output_symlink),
            )
            self.assertNotEqual(link_result.returncode, 0)
            self.assertFalse((root / "not-created.json").exists())

    def test_cli_parse_errors_do_not_echo_private_arguments(self) -> None:
        private_sentinel = "fictional-private-path-/var/private/customer-42.sqlite3"

        result = _run_cli(f"--unrecognized-option={private_sentinel}")

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(private_sentinel, result.stderr)
        self.assertNotIn("unrecognized-option", result.stderr)
        self.assertEqual(result.stderr, "retention report could not be created\n")

    def test_refuses_missing_symlink_fifo_sparse_same_inode_sidecars_and_wal_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.sqlite3"
            ledger_path = root / "ledger.sqlite3"
            with EvidenceStore(evidence_path):
                pass
            with CandidateLedger(ledger_path):
                pass

            invalid_paths: list[Path] = [root / "missing.sqlite3"]
            symlink_path = root / "input-link.sqlite3"
            symlink_path.symlink_to(evidence_path)
            invalid_paths.append(symlink_path)
            fifo_path = root / "input.fifo"
            os.mkfifo(fifo_path)
            invalid_paths.append(fifo_path)
            sparse_path = root / "sparse.sqlite3"
            with sparse_path.open("wb") as sparse_file:
                sparse_file.truncate(64 * 1024 * 1024 + 1)
            invalid_paths.append(sparse_path)

            for invalid_path in invalid_paths:
                with (
                    self.subTest(path_kind=invalid_path.name),
                    self.assertRaises(ValueError) as caught,
                ):
                    inspect_source_retention(
                        str(invalid_path),
                        str(ledger_path),
                        source="fictional-provider",
                        session_id="fictional-session",
                    )
                self.assertNotIn(str(invalid_path), str(caught.exception))

            with self.assertRaises(ValueError):
                inspect_source_retention(
                    str(evidence_path),
                    str(evidence_path),
                    source="fictional-provider",
                    session_id="fictional-session",
                )

            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = Path(f"{evidence_path}{suffix}")
                for dangling in (False, True):
                    with self.subTest(sidecar=suffix, dangling=dangling):
                        if dangling:
                            sidecar.symlink_to(root / "absent-sidecar-target")
                        else:
                            sidecar.touch()
                        try:
                            with self.assertRaises(ValueError):
                                inspect_source_retention(
                                    str(evidence_path),
                                    str(ledger_path),
                                    source="fictional-provider",
                                    session_id="fictional-session",
                                )
                        finally:
                            sidecar.unlink()

            wal_format_path = root / "wal-format.sqlite3"
            wal_bytes = bytearray(evidence_path.read_bytes())
            wal_bytes[18] = 2
            wal_bytes[19] = 2
            wal_format_path.write_bytes(wal_bytes)
            with self.assertRaises(ValueError):
                inspect_source_retention(
                    str(wal_format_path),
                    str(ledger_path),
                    source="fictional-provider",
                    session_id="fictional-session",
                )


if __name__ == "__main__":
    unittest.main()
