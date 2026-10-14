from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tracemem.adapters import AdapterStats, _transcript_source, read_transcript
from tracemem.cli import main
from tracemem.storage import EvidenceStore


def _event_line(message_id: str, text: str, provider: str = "claude") -> bytes:
    if provider == "claude":
        event = {
            "type": "assistant",
            "sessionId": "fictional-session",
            "message": {
                "id": message_id,
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        }
    else:
        event = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _mutate_source(kind: str, source: Path, first_line: bytes) -> None:
    if kind == "append":
        with source.open("ab") as target:
            target.write(_event_line("appended", "fictional-appended"))
    elif kind == "truncate":
        with source.open("r+b") as target:
            target.truncate(len(first_line))
    elif kind == "inplace":
        with source.open("r+b") as target:
            target.seek(len(first_line))
            target.write(_event_line("secondary", "after!"))
    elif kind == "replace":
        replacement = source.with_suffix(".replacement")
        replacement.write_bytes(first_line + _event_line("secondary", "replacement"))
        os.replace(replacement, source)
    else:
        raise AssertionError(f"unknown fixture mutation: {kind}")


class TranscriptConsistencyTests(unittest.TestCase):
    def test_stable_strict_reads_match_default_records_and_stats(self) -> None:
        for provider in ("claude", "codex"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                source = Path(temporary_directory) / "fictional.jsonl"
                source.write_bytes(_event_line("fictional", "evidence", provider))
                before_bytes = source.read_bytes()
                before_stat = source.stat()
                default_stats = AdapterStats()
                strict_stats = AdapterStats()
                default_records = list(read_transcript(source, provider, stats=default_stats))
                strict_records = list(
                    read_transcript(source, provider, stats=strict_stats, require_quiescent=True)
                )

                self.assertEqual(strict_records, default_records)
                self.assertEqual(strict_stats, default_stats)
                self.assertEqual(source.read_bytes(), before_bytes)
                after_stat = source.stat()
                self.assertEqual(
                    (
                        after_stat.st_dev,
                        after_stat.st_ino,
                        after_stat.st_size,
                        after_stat.st_mtime_ns,
                        after_stat.st_ctime_ns,
                        stat.S_IMODE(after_stat.st_mode),
                    ),
                    (
                        before_stat.st_dev,
                        before_stat.st_ino,
                        before_stat.st_size,
                        before_stat.st_mtime_ns,
                        before_stat.st_ctime_ns,
                        stat.S_IMODE(before_stat.st_mode),
                    ),
                )

    def test_require_quiescent_must_be_boolean_before_source_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "fictional.jsonl"
            with (
                patch.object(Path, "open", side_effect=AssertionError("Path.open called")),
                patch("os.open", side_effect=AssertionError("os.open called")),
            ):
                with self.assertRaisesRegex(TypeError, "^require_quiescent must be a boolean$"):
                    list(read_transcript(source, "claude", require_quiescent=1))

    def test_strict_mode_rejects_leaf_symlink_while_default_follows_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "fictional.jsonl"
            link = root / "alias.jsonl"
            event = {
                "type": "assistant",
                "message": {
                    "id": "fictional-message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "fictional evidence"}],
                },
            }
            target.write_text(json.dumps(event) + "\n", encoding="utf-8")
            link.symlink_to(target)

            default_records = list(read_transcript(link, "claude"))
            with self.assertRaisesRegex(
                ValueError, "^transcript source is not a quiescent regular file$"
            ):
                list(read_transcript(link, "claude", require_quiescent=True))

        self.assertEqual([record.text for record in default_records], ["fictional evidence"])

    def test_strict_mode_rejects_self_referential_leaf_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "self-loop.jsonl"
            source.symlink_to(source)
            with self.assertRaisesRegex(
                ValueError, "^transcript source is not a quiescent regular file$"
            ):
                list(read_transcript(source, "claude", require_quiescent=True))

    def test_cli_optin_preserves_leaf_path_for_symlink_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            source_directory.mkdir()
            source = source_directory / "fictional.jsonl"
            alias = source_directory / "fictional-alias.jsonl"
            indexes = root / "indexes"
            indexes.mkdir()
            event = {
                "type": "assistant",
                "message": {
                    "id": "fictional-message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "fictional evidence"}],
                },
            }
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            alias.symlink_to(source)
            before = source.read_bytes()
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                default_status = main(
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(alias),
                        "--db",
                        str(indexes / "default.sqlite"),
                    ]
                )
                strict_status = main(
                    [
                        "ingest",
                        "--provider",
                        "claude",
                        "--input",
                        str(alias),
                        "--db",
                        str(indexes / "strict.sqlite"),
                        "--require-quiescent",
                    ]
                )
            after = source.read_bytes()

        self.assertEqual(default_status, 0)
        self.assertEqual(strict_status, 2)
        self.assertEqual(
            stderr.getvalue(),
            "tracemem: ingest failed; no transcript contents were printed\n",
        )
        self.assertEqual(after, before)

    def test_quiescent_mutations_abort_store_transaction_and_default_accepts(self) -> None:
        cases = ("append", "truncate", "inplace", "replace")
        for kind in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                source = root / "fictional.jsonl"
                database = root / "strict.sqlite"
                first_line = _event_line("stable", "after")
                source.write_bytes(_event_line("stable", "before"))
                with EvidenceStore(database) as store:
                    store.ingest(read_transcript(source, "claude"))
                    before_all = store.records(current_only=False)
                    before_heads = store.records(current_only=True)
                    source.write_bytes(first_line + _event_line("secondary", "before!"))
                    strict_iterator = read_transcript(source, "claude", require_quiescent=True)

                    def mutate_after_first_record(
                        strict_iterator=strict_iterator,
                        kind=kind,
                        source=source,
                        first_line=first_line,
                    ):
                        yield next(strict_iterator)
                        _mutate_source(kind, source, first_line)
                        yield from strict_iterator

                    with self.assertRaisesRegex(
                        ValueError, "^transcript source is not a quiescent regular file$"
                    ):
                        store.ingest(mutate_after_first_record())
                    self.assertEqual(store.records(current_only=False), before_all)
                    self.assertEqual(store.records(current_only=True), before_heads)

                default_source = root / "default.jsonl"
                default_source.write_bytes(_event_line("stable", "before"))
                with EvidenceStore(root / "default.sqlite") as store:
                    store.ingest(read_transcript(default_source, "claude"))
                    default_source.write_bytes(first_line + _event_line("secondary", "before!"))

                    def mutate_default_after_first_record(
                        kind=kind, default_source=default_source, first_line=first_line
                    ):
                        iterator = read_transcript(default_source, "claude")
                        yield next(iterator)
                        _mutate_source(kind, default_source, first_line)
                        yield from iterator

                    self.assertGreaterEqual(store.ingest(mutate_default_after_first_record()), 1)
                    self.assertEqual(store.records()[0].text, "after")

    def test_special_paths_and_regular_to_fifo_open_race_are_refused(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "directory"
            directory.mkdir()
            fifo = root / "fifo"
            os.mkfifo(fifo)
            missing = root / "missing.jsonl"
            for source in (directory, fifo, missing):
                with (
                    self.subTest(source=source.name),
                    self.assertRaisesRegex(
                        ValueError, "^transcript source is not a quiescent regular file$"
                    ),
                ):
                    list(read_transcript(source, "claude", require_quiescent=True))

            raced = root / "raced.jsonl"
            raced.write_bytes(_event_line("fictional", "evidence"))
            original_open = os.open

            def replace_with_fifo(path, flags, *args, **kwargs):
                raced.unlink()
                os.mkfifo(raced)
                return original_open(path, flags, *args, **kwargs)

            with patch.object(_transcript_source.os, "open", side_effect=replace_with_fifo):
                with self.assertRaisesRegex(
                    ValueError, "^transcript source is not a quiescent regular file$"
                ):
                    list(read_transcript(raced, "claude", require_quiescent=True))

    def test_quiescent_open_uses_nonblocking_nofollow_and_cloexec_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "fictional.jsonl"
            source.write_bytes(_event_line("fictional", "evidence"))
            original_open = os.open
            observed_flags: list[int] = []

            def inspect_open(path, flags, *args, **kwargs):
                observed_flags.append(flags)
                return original_open(path, flags, *args, **kwargs)

            with patch.object(_transcript_source.os, "open", side_effect=inspect_open):
                list(read_transcript(source, "claude", require_quiescent=True))

        self.assertEqual(len(observed_flags), 1)
        self.assertTrue(observed_flags[0] & os.O_NONBLOCK)
        for optional_flag in ("O_NOFOLLOW", "O_CLOEXEC"):
            if hasattr(os, optional_flag):
                self.assertTrue(observed_flags[0] & getattr(os, optional_flag))

    def test_early_iterator_close_releases_descriptor_without_eof_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "fictional.jsonl"
            source.write_bytes(_event_line("first", "first") + _event_line("second", "second"))
            descriptors: list[int] = []
            original_enter = _transcript_source.QuiescentTranscriptSource.__enter__

            def capture_enter(guard):
                stream = original_enter(guard)
                descriptors.append(stream.fileno())
                return stream

            with (
                patch.object(
                    _transcript_source.QuiescentTranscriptSource, "__enter__", capture_enter
                ),
                patch.object(
                    _transcript_source.QuiescentTranscriptSource,
                    "verify_unchanged",
                    side_effect=AssertionError("partial iteration must not verify EOF"),
                ),
            ):
                iterator = read_transcript(source, "claude", require_quiescent=True)
                next(iterator)
                iterator.close()

        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])

    def test_strict_mode_preserves_utf8_and_line_limit_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_utf8 = root / "invalid.jsonl"
            invalid_utf8.write_bytes(b'{"text":"\xff"}\n')
            too_long = root / "long.jsonl"
            too_long.write_bytes(b"{" + b"x" * 20)
            with self.assertRaisesRegex(ValueError, "^transcript source is not valid UTF-8$"):
                list(read_transcript(invalid_utf8, "claude", require_quiescent=True))
            with self.assertRaisesRegex(ValueError, "^transcript line exceeds the byte limit$"):
                list(read_transcript(too_long, "claude", require_quiescent=True, max_line_bytes=10))
