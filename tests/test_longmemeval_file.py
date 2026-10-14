"""Contract tests for bounded fictional LongMemEval file scanning."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path


class LongMemEvalFileTests(unittest.TestCase):
    @staticmethod
    def _fixture_bytes() -> bytes:
        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        return fixture_path.read_bytes()

    def test_file_scan_matches_bytes_adapter_and_hashes_exact_input(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        raw = self._fixture_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(b" \n" + raw + b"\t\n")
            exact = path.read_bytes()

            expected = parse_longmemeval_bytes(exact, namespace="study", variant="s")
            result = scan_longmemeval_file(
                path,
                namespace="study",
                variant="s",
                question_ids=tuple(question.question_id for question in expected),
            )

        self.assertEqual(result.selected_questions, expected)
        self.assertEqual(result.input_sha256, hashlib.sha256(exact).hexdigest())
        self.assertEqual(result.input_bytes, len(exact))
        self.assertEqual(result.question_count, 2)

    def test_unselected_integer_answer_is_validated_and_input_is_unchanged(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        items = json.loads(self._fixture_bytes())
        items[1]["answer"] = -12
        raw = json.dumps(items, ensure_ascii=False).encode("utf-8")
        expected = parse_longmemeval_bytes(raw, namespace="study", variant="s")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(raw)
            before = path.stat()
            before_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            result = scan_longmemeval_file(
                path,
                namespace="study",
                variant="s",
                question_ids=(expected[0].question_id,),
            )

            after = path.stat()
            after_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(result.selected_questions, (expected[0],))
        self.assertEqual(result.question_count, 2)
        self.assertEqual(result.input_sha256, before_hash)
        self.assertEqual(after_hash, before_hash)
        self.assertEqual(stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))
        self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)

    def test_empty_selection_validates_everything_and_retains_no_questions(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(self._fixture_bytes())
            result = scan_longmemeval_file(path, namespace="study", variant="s")

        self.assertEqual(result.question_count, 2)
        self.assertEqual(result.selected_questions, ())

    def test_requested_id_order_does_not_change_source_order(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        raw = self._fixture_bytes()
        source_ids = tuple(item["question_id"] for item in json.loads(raw))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(raw)
            result = scan_longmemeval_file(
                path,
                namespace="study",
                variant="s",
                question_ids=tuple(reversed(source_ids)),
            )

        self.assertEqual(
            tuple(question.question_id for question in result.selected_questions), source_ids
        )

    def test_scanner_handles_escaped_content_across_varied_read_boundaries(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        items = json.loads(self._fixture_bytes())
        items[0]["question"] = 'A "quoted" thought, a \\ slash, and 東京.'
        raw = json.dumps(items, ensure_ascii=False).encode("utf-8")
        expected = parse_longmemeval_bytes(raw, namespace="study", variant="s")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(raw)
            with patch.object(longmemeval_file, "_CHUNK_BYTES", 7):
                result = longmemeval_file.scan_longmemeval_file(
                    path,
                    namespace="study",
                    variant="s",
                    question_ids=(items[0]["question_id"],),
                )

        self.assertEqual(result.selected_questions, (expected[0],))

    def test_input_kinds_are_rejected_without_opening_streams(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "regular.json"
            regular.write_bytes(self._fixture_bytes())
            symlink = root / "link.json"
            symlink.symlink_to(regular)
            fifo = root / "pipe.json"
            if hasattr(os, "mkfifo"):
                os.mkfifo(fifo)
            rejected = (
                (root, "regular non-symlink file"),
                (symlink, "regular non-symlink file"),
                (root / "missing.json", "file could not be read"),
            )
            if fifo.exists():
                rejected += ((fifo, "regular non-symlink file"),)
            for path, error in rejected:
                with self.subTest(path_type=path.name):
                    with self.assertRaisesRegex(ValueError, error) as failure:
                        scan_longmemeval_file(path, namespace="study", variant="s")
                    self.assertNotIn(str(path), str(failure.exception))

    def test_invalid_path_errors_are_static_and_do_not_expose_path_text(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        private_path_text = "private-selector-marker\x00file.json"
        with self.assertRaisesRegex(ValueError, "LongMemEval file could not be read") as failure:
            scan_longmemeval_file(private_path_text, namespace="study", variant="s")

        self.assertNotIn("private-selector-marker", str(failure.exception))

    def test_sparse_file_over_four_gib_is_rejected_from_metadata(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.json"
            with path.open("wb") as sparse:
                sparse.truncate(4 * 1024 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "exceeds 4 GiB limit"):
                scan_longmemeval_file(path, namespace="study", variant="s")

    def test_item_total_and_nesting_limits_are_enforced(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file

        raw = self._fixture_bytes()
        limits = (
            ("_MAX_ITEM_BYTES", 10, "question exceeds 64 MiB limit"),
            ("_MAX_FILE_BYTES", len(raw) - 1, "exceeds 4 GiB limit"),
            ("_MAX_NESTING_DEPTH", 2, "exceeds nesting limit"),
        )
        for name, value, error in limits:
            with self.subTest(limit=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fictional.json"
                path.write_bytes(raw)
                with patch.object(longmemeval_file, name, value):
                    with self.assertRaisesRegex(ValueError, error):
                        longmemeval_file.scan_longmemeval_file(path, namespace="study", variant="s")

    def test_total_limit_is_enforced_again_after_open_during_reads(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file

        raw = self._fixture_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(raw)
            original_scan = longmemeval_file._scan_stream

            def grow_after_stat(reader: object, **kwargs: object) -> object:
                with path.open("ab") as changed:
                    changed.write(b" ")
                return original_scan(reader, **kwargs)  # type: ignore[arg-type]

            with (
                patch.object(longmemeval_file, "_MAX_FILE_BYTES", len(raw)),
                patch.object(longmemeval_file, "_scan_stream", side_effect=grow_after_stat),
            ):
                with self.assertRaisesRegex(ValueError, "exceeds 4 GiB limit"):
                    longmemeval_file.scan_longmemeval_file(path, namespace="study", variant="s")

    def test_mutation_during_scan_is_rejected(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(self._fixture_bytes())
            original_scan = longmemeval_file._scan_stream

            def mutate_after_scan(reader: object, **kwargs: object) -> object:
                result = original_scan(reader, **kwargs)  # type: ignore[arg-type]
                with path.open("ab") as changed:
                    changed.write(b" ")
                return result

            with patch.object(longmemeval_file, "_scan_stream", side_effect=mutate_after_scan):
                with self.assertRaisesRegex(ValueError, "changed while it was being scanned"):
                    longmemeval_file.scan_longmemeval_file(path, namespace="study", variant="s")

    def test_whitespace_suffix_over_64_mib_is_streamed_and_hashed(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        raw = self._fixture_bytes()
        suffix_chunk = b" " * (1024 * 1024)
        suffix_chunks = 65
        expected_digest = hashlib.sha256()
        expected_digest.update(raw)
        expected_bytes = len(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional-large.json"
            with path.open("wb") as output:
                output.write(raw)
                for _ in range(suffix_chunks):
                    output.write(suffix_chunk)
                    expected_digest.update(suffix_chunk)
                    expected_bytes += len(suffix_chunk)

            result = scan_longmemeval_file(path, namespace="study", variant="s")

        self.assertGreater(result.input_bytes, 64 * 1024 * 1024)
        self.assertEqual(result.input_bytes, expected_bytes)
        self.assertEqual(result.input_sha256, expected_digest.hexdigest())
        self.assertEqual(result.question_count, 2)

    def test_scan_does_not_change_input_mode_or_mtime(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(self._fixture_bytes())
            before = path.stat()
            scan_longmemeval_file(path, namespace="study", variant="s")
            after = path.stat()

        self.assertEqual(stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))
        self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)

    def test_duplicate_question_ids_across_items_are_rejected(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        items = json.loads(self._fixture_bytes())
        items[1]["question_id"] = items[0]["question_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_text(json.dumps(items), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "LongMemEval input does not match the documented schema"
            ):
                scan_longmemeval_file(path, namespace="study", variant="s")

    def test_malformed_late_question_prevents_a_selected_prefix_result(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        items = json.loads(self._fixture_bytes())
        items[1]["question_type"] = "unknown-fictional-type"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_text(json.dumps(items), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "LongMemEval input does not match the documented schema"
            ):
                scan_longmemeval_file(
                    path,
                    namespace="study",
                    variant="s",
                    question_ids=(items[0]["question_id"],),
                )

    def test_strict_json_and_utf8_rejections_are_sanitized(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        raw = self._fixture_bytes()
        malformed_inputs = (
            (b"null", "LongMemEval input is not valid JSON"),
            (raw.rstrip()[:-1] + b",]", "LongMemEval input is not valid JSON"),
            (
                raw.replace(b'"role": "user"', b'"role": "user", "role": "user"', 1),
                "LongMemEval input is not valid JSON",
            ),
            (raw.replace("café".encode(), b"bad\xff", 1), "LongMemEval input is not valid UTF-8"),
        )
        for malformed, error in malformed_inputs:
            with self.subTest(error=error):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "private-fixture.json"
                    path.write_bytes(malformed)
                    with self.assertRaisesRegex(ValueError, error) as failure:
                        scan_longmemeval_file(path, namespace="study", variant="s")
                self.assertNotIn(str(path), str(failure.exception))
                self.assertNotIn("fictional-question", str(failure.exception))

    def test_identity_and_selection_validation_precede_path_access(self) -> None:
        from tracemem.adapters.longmemeval_file import scan_longmemeval_file

        missing = Path("/private/should-not-be-inspected.json")
        invalid_calls = (
            ({"namespace": "", "variant": "s"}, "namespace must be a nonblank string"),
            ({"namespace": "study", "variant": "unknown"}, "variant must be one of s, m, oracle"),
            (
                {"namespace": "study", "variant": "s", "question_ids": ["id"]},
                "question_ids must be a tuple",
            ),
            (
                {"namespace": "study", "variant": "s", "question_ids": (" ",)},
                "question_ids must be a tuple",
            ),
            (
                {"namespace": "study", "variant": "s", "question_ids": ("\ud800",)},
                "question_ids must be a tuple",
            ),
            (
                {"namespace": "study", "variant": "s", "question_ids": ("same", "same")},
                "question_ids must be a tuple",
            ),
        )
        for keywords, error in invalid_calls:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error) as failure:
                    scan_longmemeval_file(missing, **keywords)
                self.assertNotIn("should-not-be-inspected", str(failure.exception))

    def test_question_id_in_file_has_a_patched_byte_limit(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(self._fixture_bytes())
            with patch.object(longmemeval_file, "_MAX_ID_BYTES", 5):
                with self.assertRaisesRegex(ValueError, "question ID byte limit"):
                    longmemeval_file.scan_longmemeval_file(path, namespace="study", variant="s")

    def test_requested_question_id_has_a_patched_byte_limit(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(self._fixture_bytes())
            with patch.object(longmemeval_file, "_MAX_ID_BYTES", 5, create=True):
                with self.assertRaisesRegex(ValueError, "question_ids must be a tuple"):
                    longmemeval_file.scan_longmemeval_file(
                        path,
                        namespace="study",
                        variant="s",
                        question_ids=("sixsix",),
                    )

    def test_cumulative_seen_question_id_bytes_are_bounded(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(self._fixture_bytes())
            with patch.object(longmemeval_file, "_MAX_SEEN_ID_BYTES", 5, create=True):
                with self.assertRaisesRegex(ValueError, "question ID budget"):
                    longmemeval_file.scan_longmemeval_file(path, namespace="study", variant="s")

    def test_question_limit_is_checked_before_reading_an_excess_question(self) -> None:
        from unittest.mock import patch

        from tracemem.adapters import longmemeval_file
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        items = json.loads(self._fixture_bytes())
        raw = json.dumps([items[0], {}], ensure_ascii=False).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_bytes(raw)
            with (
                patch.object(longmemeval_file, "_MAX_QUESTIONS", 1),
                patch.object(
                    longmemeval_file,
                    "parse_longmemeval_bytes",
                    wraps=parse_longmemeval_bytes,
                ) as parse_item,
            ):
                with self.assertRaisesRegex(ValueError, "exceeds 10000 question limit"):
                    longmemeval_file.scan_longmemeval_file(path, namespace="study", variant="s")

        self.assertEqual(parse_item.call_count, 2)


if __name__ == "__main__":
    unittest.main()
