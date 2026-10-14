from __future__ import annotations

import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracemem.adapters import AdapterStats, read_transcript
from tracemem.cli import main
from tracemem.schema import EvidenceRecord
from tracemem.storage import EvidenceStore


class TranscriptBoundsTests(unittest.TestCase):
    def test_invalid_line_limit_is_rejected_before_source_open(self) -> None:
        cases = [
            (True, TypeError, "max_line_bytes must be an integer"),
            (1.5, TypeError, "max_line_bytes must be an integer"),
            (0, ValueError, "max_line_bytes must be between 1 and 67108864"),
            (67108865, ValueError, "max_line_bytes must be between 1 and 67108864"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "missing.jsonl"
            with patch.object(Path, "open", side_effect=AssertionError("source opened")):
                for limit, exception, message in cases:
                    with self.subTest(limit=limit), self.assertRaisesRegex(exception, message):
                        list(read_transcript(source, "claude", max_line_bytes=limit))

    def test_exact_crlf_limit_and_eof_records_preserve_identity_and_stats(self) -> None:
        claude_event = {
            "type": "user",
            "sessionId": "claude-session",
            "timestamp": "2026-10-14T08:00:00Z",
            "message": {
                "id": "claude-message",
                "role": "user",
                "content": [{"type": "text", "text": "café 🧪"}],
            },
        }
        codex_meta = {"type": "session_meta", "payload": {"id": "codex-session"}}
        codex_event = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "codex-message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "EOF ✓"}],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            claude_path = workspace / "claude.jsonl"
            claude_line = json.dumps(claude_event, ensure_ascii=False).encode("utf-8") + b"\r\n"
            claude_path.write_bytes(claude_line + b'{"type":"unknown"}\r\n{bad}\n')
            claude_stats = AdapterStats()
            claude_records = list(
                read_transcript(
                    claude_path, "claude", stats=claude_stats, max_line_bytes=len(claude_line)
                )
            )
            claude_locator = hashlib.sha256(str(claude_path.resolve()).encode()).hexdigest()

            codex_path = workspace / "codex.jsonl"
            codex_path.write_bytes(
                (
                    json.dumps(codex_meta) + "\r\n" + json.dumps(codex_event, ensure_ascii=False)
                ).encode()
            )
            codex_stats = AdapterStats()
            codex_records = list(
                read_transcript(codex_path, "codex", stats=codex_stats, max_line_bytes=4096)
            )
            codex_locator = hashlib.sha256(str(codex_path.resolve()).encode()).hexdigest()

        self.assertEqual(
            claude_records,
            [
                EvidenceRecord(
                    "claude",
                    "claude-session",
                    "claude-message",
                    0,
                    "user",
                    "café 🧪",
                    "2026-10-14T08:00:00Z",
                    claude_locator,
                    1,
                    "mixed_untrusted",
                )
            ],
        )
        self.assertEqual(claude_stats.role_counts, {"user": 1})
        self.assertEqual(claude_stats.malformed_records, 1)
        self.assertEqual(claude_stats.unsupported_records, 1)
        self.assertEqual(
            codex_records,
            [
                EvidenceRecord(
                    "codex",
                    "codex-session",
                    "codex-message",
                    0,
                    "assistant",
                    "EOF ✓",
                    None,
                    codex_locator,
                    2,
                    "conversation",
                )
            ],
        )
        self.assertEqual(codex_stats.role_counts, {"assistant": 1})
        self.assertEqual(codex_stats.malformed_records, 0)
        self.assertEqual(codex_stats.unsupported_records, 0)

    def test_oversized_physical_line_is_rejected_after_only_a_bounded_read(self) -> None:
        class ReadGuard:
            def __init__(self, stream: object, requests: list[int]) -> None:
                self.stream = stream
                self.requests = requests

            def __enter__(self) -> ReadGuard:
                return self

            def __exit__(self, *args: object) -> None:
                self.stream.close()  # type: ignore[attr-defined]

            def __iter__(self) -> object:
                raise AssertionError("transcript was consumed by unbounded iteration")

            def readline(self, size: int = -1) -> bytes:
                self.requests.append(size)
                return self.stream.readline(size)  # type: ignore[attr-defined,no-any-return]

        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "large.jsonl"
            source.write_bytes(b"{" + b"x" * 2_000_000)
            requests: list[int] = []
            original_open = Path.open

            def guarded_open(
                path: Path, mode: str = "r", *args: object, **kwargs: object
            ) -> ReadGuard:
                return ReadGuard(original_open(path, mode, *args, **kwargs), requests)

            with patch.object(Path, "open", guarded_open):
                with self.assertRaisesRegex(ValueError, "^transcript line exceeds the byte limit$"):
                    next(read_transcript(source, "claude", max_line_bytes=1024))

        self.assertEqual(requests, [1025])

    def test_invalid_utf8_remains_a_sanitized_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "invalid.jsonl"
            source.write_bytes(b'{"text":"\xff"}\n')
            with self.assertRaisesRegex(ValueError, "^transcript source is not valid UTF-8$"):
                list(read_transcript(source, "claude", max_line_bytes=100))

    def test_late_oversized_line_rolls_back_new_rows_and_existing_heads(self) -> None:
        event = {
            "type": "assistant",
            "sessionId": "stable-session",
            "message": {
                "id": "stable-message",
                "role": "assistant",
                "content": [{"type": "text", "text": "before"}],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            source = workspace / "transcript.jsonl"
            database = workspace / "index" / "evidence.sqlite"
            database.parent.mkdir()
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with EvidenceStore(database) as store:
                store.ingest(read_transcript(source, "claude"))
                before_all = store.records(current_only=False)
                before_heads = store.records(current_only=True)

                event["message"]["content"] = [
                    {"type": "text", "text": "changed head"},
                    {"type": "text", "text": "new row must roll back"},
                ]
                first_line = json.dumps(event).encode("utf-8") + b"\n"
                source.write_bytes(first_line + b"{" + b"x" * 2048)
                with self.assertRaisesRegex(ValueError, "^transcript line exceeds the byte limit$"):
                    store.ingest(read_transcript(source, "claude", max_line_bytes=len(first_line)))

                self.assertEqual(store.records(current_only=False), before_all)
                self.assertEqual(store.records(current_only=True), before_heads)

    def test_cli_passes_limit_and_rejects_small_cap_without_changing_source(self) -> None:
        event = {
            "type": "user",
            "sessionId": "cli-session",
            "message": {
                "id": "cli-message",
                "role": "user",
                "content": [{"type": "text", "text": "CLI Unicode ✓"}],
            },
        }
        raw_line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            source_directory = workspace / "source"
            index_directory = workspace / "index"
            source_directory.mkdir()
            index_directory.mkdir()
            source = source_directory / "transcript.jsonl"
            source.write_bytes(raw_line)
            source.chmod(0o640)
            before_bytes = source.read_bytes()
            before_stat = source.stat()
            accepted_stdout = io.StringIO()
            with contextlib.redirect_stdout(accepted_stdout):
                accepted = main(
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source),
                        "--db",
                        str(index_directory / "accepted.sqlite"),
                        "--max-line-bytes",
                        str(len(raw_line)),
                    ]
                )
            rejected_stderr = io.StringIO()
            with contextlib.redirect_stderr(rejected_stderr):
                rejected = main(
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(source),
                        "--db",
                        str(index_directory / "rejected.sqlite"),
                        "--max-line-bytes",
                        str(len(raw_line) - 1),
                    ]
                )
            after_bytes = source.read_bytes()
            after_stat = source.stat()

        self.assertEqual(accepted, 0)
        self.assertEqual(json.loads(accepted_stdout.getvalue())["inserted_revisions"], 1)
        self.assertEqual(rejected, 2)
        self.assertEqual(
            rejected_stderr.getvalue(),
            "tracemem: ingest failed; no transcript contents were printed\n",
        )
        self.assertNotIn("CLI Unicode", rejected_stderr.getvalue())
        self.assertEqual(after_bytes, before_bytes)
        self.assertEqual(stat.S_IMODE(after_stat.st_mode), stat.S_IMODE(before_stat.st_mode))
        self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
