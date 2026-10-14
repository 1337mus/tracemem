import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class SyntheticBenchmarkCliTests(unittest.TestCase):
    def test_offline_benchmark_reports_diagnostics_without_corpus_text(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.find_spec("tracemem.__main__")
        self.assertIsNotNone(spec, "python -m tracemem entry point is not implemented")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")

        process = subprocess.run(
            [sys.executable, "-m", "tracemem", "benchmark"],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(set(result["modes"]), {"bm25", "vector", "hybrid"})
        self.assertEqual(result["answerable_queries"], 14)
        self.assertEqual(result["unknown_queries"], 3)
        self.assertEqual(len(result["input_sha256"]), 64)
        self.assertIn("python", result["environment"])
        self.assertIn("evidence_schema", result["index_versions"])
        self.assertEqual(result["bootstrap"], {"samples": 1000, "seed": 1729})
        self.assertIn("weak lexical", result["embedder"]["label"])
        self.assertLess(result["modes"]["bm25"]["recall_at_k"], 1.0)
        self.assertIn("paraphrase_failure", result["modes"]["bm25"]["case_breakdown"])
        self.assertIn("q02", result["modes"]["hybrid"]["insufficient_evidence_query_ids"])
        self.assertNotIn("analytics database vacuum", process.stdout)
        self.assertNotIn("nightly database maintenance", process.stdout)
        self.assertNotIn(str(root), process.stdout)

    def test_raw_claude_ingest_to_sqlite_search_preserves_exact_citation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            input_directory = workspace / "private-source"
            database_directory = workspace / "private-index"
            input_directory.mkdir()
            database_directory.mkdir()
            source_path = input_directory / "conversation.jsonl"
            database_path = database_directory / "evidence.sqlite"
            direct_text = "The polar backup cluster is flint-91, and the schedule is 04:35 UTC."
            events = [
                {
                    "type": "user",
                    "sessionId": "claude-session-native",
                    "timestamp": "2026-09-25T10:00:00Z",
                    "message": {
                        "id": "native-user-1",
                        "role": "user",
                        "content": [
                            {"type": "text", "text": direct_text},
                            {"type": "text", "text": "Nightly run waits until 04:35 UTC."},
                        ],
                    },
                },
                {
                    "type": "assistant",
                    "sessionId": "claude-session-native",
                    "timestamp": "2026-09-25T10:00:05Z",
                    "message": {
                        "id": "native-assistant-1",
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Maybe move it to flint-92 next year."}
                        ],
                    },
                },
            ]
            source_bytes = b"".join(
                json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n" for event in events
            )
            source_path.write_bytes(source_bytes)
            before_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

            ingest = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tracemem",
                    "ingest",
                    "--provider",
                    "claude",
                    "--input",
                    str(source_path),
                    "--db",
                    str(database_path),
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(ingest.returncode, 0, ingest.stderr)
            ingest_result = json.loads(ingest.stdout)
            self.assertGreater(ingest_result["inserted_revisions"], 0)
            self.assertNotIn(direct_text, ingest.stdout)

            search = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tracemem",
                    "search",
                    "--db",
                    str(database_path),
                    "--query",
                    "which backup cluster is used",
                    "--mode",
                    "bm25",
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(search.returncode, 0, search.stderr)
            search_result = json.loads(search.stdout)
            hit = search_result["hits"][0]

            from tracemem import EvidenceStore

            with EvidenceStore(database_path) as store:
                record = store.get(hit["record_id"], hit["revision"])
            expected_citation = f"{record.record_id}@{record.revision}:0:{len(record.text)}"
            self.assertEqual(record.source, "claude")
            self.assertEqual(record.session_id, "claude-session-native")
            self.assertEqual(record.message_id, "native-user-1")
            self.assertEqual(record.part_index, 0)
            self.assertEqual(record.text, direct_text)
            self.assertEqual(hit["citation"], expected_citation)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), before_hash)

    def test_path_guards_reject_source_sidecars_and_missing_search_database(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            source_directory = workspace / "private-source"
            outside_directory = workspace / "outside"
            source_directory.mkdir()
            outside_directory.mkdir()
            source_path = source_directory / "transcript.jsonl"
            source_path.write_text("do not print this text\n", encoding="utf-8")
            symlink_directory = workspace / "source-alias"
            symlink_directory.symlink_to(source_directory, target_is_directory=True)
            symlink_target = outside_directory / "target.sqlite"
            database_symlink = outside_directory / "index-link.sqlite"
            from tracemem import EvidenceStore

            with EvidenceStore(symlink_target):
                pass
            database_symlink.symlink_to(symlink_target)
            missing_database = workspace / "missing.sqlite"
            cases = [
                (
                    "database path is source file",
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source_path),
                        "--db",
                        str(source_path),
                    ],
                    source_path,
                ),
                (
                    "database path is source directory",
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source_path),
                        "--db",
                        str(source_directory),
                    ],
                    source_directory,
                ),
                (
                    "database sidecar in source directory",
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source_path),
                        "--db",
                        str(source_directory / "sidecar.sqlite"),
                    ],
                    source_directory / "sidecar.sqlite",
                ),
                (
                    "database inside source directory descendant",
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source_path),
                        "--db",
                        str(source_directory / "nested" / "evidence.sqlite"),
                    ],
                    source_directory / "nested" / "evidence.sqlite",
                ),
                (
                    "database path is a symlink",
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source_path),
                        "--db",
                        str(database_symlink),
                    ],
                    database_symlink,
                ),
                (
                    "symlink parent resolves to source file",
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source_path),
                        "--db",
                        str(symlink_directory / source_path.name),
                    ],
                    source_path,
                ),
                (
                    "search database does not exist",
                    [
                        "search",
                        "--db",
                        str(missing_database),
                        "--query",
                        "a deliberate explicit query",
                    ],
                    missing_database,
                ),
            ]

            for label, arguments, protected_path in cases:
                with self.subTest(label=label):
                    process = subprocess.run(
                        [sys.executable, "-m", "tracemem", *arguments],
                        cwd=root,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(process.returncode, 2, process.stderr)
                    self.assertNotIn("do not print this text", process.stdout + process.stderr)
                    if label == "search database does not exist":
                        self.assertFalse(protected_path.exists())
                    if label == "database path is a symlink":
                        with EvidenceStore(symlink_target) as store:
                            self.assertEqual(store.records(current_only=False), [])

    def test_ndcg_ideal_uses_k_when_fewer_than_k_hits_exist(self) -> None:
        from tracemem import EvidenceRecord, EvidenceStore
        from tracemem.benchmark import run_benchmark

        record = EvidenceRecord(
            source="test",
            session_id="session",
            message_id="present",
            part_index=0,
            role="user",
            text="Cerulean backup belongs in storage.",
        )
        query = {
            "query_id": "short-results",
            "query": "cerulean backup",
            "relevant_message_ids": ["present", "absent"],
            "sufficient_sets": [["present"]],
        }

        with TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest([record])
                result = run_benchmark(
                    store,
                    [query],
                    input_sha256="0" * 64,
                    k=5,
                    bootstrap_samples=8,
                )

        metrics = result["modes"]["bm25"]["per_query"][0]
        self.assertAlmostEqual(metrics["recall_at_k"], 0.5)
        self.assertLess(metrics["ndcg"], 1.0)


if __name__ == "__main__":
    unittest.main()
