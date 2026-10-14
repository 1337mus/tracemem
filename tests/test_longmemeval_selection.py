"""Contract tests for the selected-file LongMemEval retrieval diagnostic."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments import evaluate_longmemeval_retrieval as evaluator

_NAMESPACE = "fictional-selection-study"


def _question(
    question_id: str,
    query: str,
    session_id: str,
    content: str,
    *,
    has_answer: bool = True,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": query,
        "answer": "private fictional answer",
        "question_date": "2038-01-02",
        "haystack_session_ids": [session_id],
        "haystack_dates": ["2038-01-01"],
        "haystack_sessions": [[{"role": "user", "content": content, "has_answer": has_answer}]],
        "answer_session_ids": [session_id] if has_answer else [],
    }


def _cli_args(input_path: Path, output_path: Path, *question_ids: str) -> list[str]:
    arguments = [
        "--input",
        str(input_path),
        "--namespace",
        _NAMESPACE,
        "--variant",
        "s",
        "--output",
        str(output_path),
    ]
    for question_id in question_ids:
        arguments.extend(("--question-id", question_id))
    return arguments


class LongMemEvalSelectionTests(unittest.TestCase):
    def test_selected_file_report_matches_bytes_metrics_and_hashes_full_source(self) -> None:
        selected = _question(
            "fictional-selected",
            "PRIVATE_SELECTED_QUERY: where is the amber compass?",
            "fictional-selected-session",
            "PRIVATE_SELECTED_TEXT: the amber compass is in a blue cabinet.",
        )
        unselected = _question(
            "fictional-unselected",
            "PRIVATE_UNSELECTED_QUERY: where is the copper lantern?",
            "fictional-unselected-session",
            "PRIVATE_UNSELECTED_TEXT: a copper lantern hangs near a window.",
        )
        raw = (
            b" \n"
            + json.dumps([selected, unselected], ensure_ascii=False).encode("utf-8")
            + b"\t\n"
        )
        selected_raw = json.dumps([selected], ensure_ascii=False).encode("utf-8")
        expected = evaluator.evaluate_retrieval_bytes(
            selected_raw, namespace=_NAMESPACE, variant="s"
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional-source.json"
            path.write_bytes(raw)
            report = evaluator.evaluate_retrieval_file(
                path,
                namespace=_NAMESPACE,
                variant="s",
                question_ids=(selected["question_id"],),
            )

        self.assertEqual(report.get("schema_version"), 2)
        self.assertEqual(report.get("input_sha256"), hashlib.sha256(raw).hexdigest())
        self.assertEqual(report.get("input_bytes"), len(raw))
        self.assertEqual(
            report.get("selection"),
            {
                "kind": "explicit_question_ids",
                "source_question_count": 2,
                "evaluated_question_count": 1,
            },
        )
        for field in (
            "variant",
            "cutoffs",
            "modes",
            "environment",
            "evidence_schema_version",
            "embedding",
            "rrf_k",
            "counts",
            "questions",
            "aggregates",
        ):
            self.assertEqual(report.get(field), expected[field], field)
        self.assertEqual(report["questions"][0]["question_id"], selected["question_id"])

        serialized = json.dumps(report, ensure_ascii=False)
        for private_text in (
            "PRIVATE_SELECTED_QUERY",
            "PRIVATE_SELECTED_TEXT",
            "PRIVATE_UNSELECTED_QUERY",
            "PRIVATE_UNSELECTED_TEXT",
            "private fictional answer",
            str(path),
            _NAMESPACE,
        ):
            self.assertNotIn(private_text, serialized)

    def test_selection_counts_and_exclusions_use_only_selected_questions(self) -> None:
        scored = _question(
            "fictional-scored", "Where is a gold leaf?", "scored-session", "A gold leaf is here."
        )
        unlabeled = _question(
            "fictional-unlabeled",
            "Where is an unlabeled shell?",
            "unlabeled-session",
            "An unlabeled shell is there.",
            has_answer=False,
        )
        abstention = _question(
            "fictional-abstention_abs",
            "Does an imaginary tunnel exist?",
            "abstention-session",
            "An imaginary tunnel exists.",
        )
        excluded = _question(
            "fictional-not-selected", "Where is a hidden kite?", "excluded", "A hidden kite."
        )
        items = [scored, unlabeled, abstention, excluded]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_text(json.dumps(items), encoding="utf-8")
            report = evaluator.evaluate_retrieval_file(
                path,
                namespace=_NAMESPACE,
                variant="s",
                question_ids=(
                    scored["question_id"],
                    unlabeled["question_id"],
                    abstention["question_id"],
                ),
                cutoffs=(1, 3),
            )

        self.assertEqual(
            report["counts"],
            {"question_count": 3, "abstention_count": 1, "unlabeled_count": 1, "scored_count": 1},
        )
        for mode in evaluator._MODES:
            cells = {item["question_id"]: item["retrieval"][mode] for item in report["questions"]}
            self.assertEqual(
                cells[unlabeled["question_id"]][0]["exclusion_reason"], "missing_turn_labels"
            )
            self.assertEqual(cells[abstention["question_id"]][0]["exclusion_reason"], "abstention")
            for aggregate in report["aggregates"][mode]:
                self.assertEqual(aggregate["question_count"], 3)
                self.assertEqual(aggregate["abstention_count"], 1)
                self.assertEqual(aggregate["unlabeled_count"], 1)
                self.assertEqual(aggregate["scored_question_count"], 1)

    def test_bytes_report_keeps_schema_one_and_its_original_field_set(self) -> None:
        item = _question(
            "fictional-bytes-only", "Where is a tin star?", "bytes-session", "A tin star."
        )
        report = evaluator.evaluate_retrieval_bytes(
            json.dumps([item]).encode("utf-8"), namespace=_NAMESPACE, variant="s"
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(
            set(report),
            {
                "schema_version",
                "input_sha256",
                "variant",
                "cutoffs",
                "modes",
                "environment",
                "evidence_schema_version",
                "embedding",
                "rrf_k",
                "counts",
                "questions",
                "aggregates",
            },
        )

    def test_late_invalid_unselected_item_refuses_before_store_and_cli_output(self) -> None:
        selected = _question(
            "fictional-before-error", "Where is a silver button?", "valid", "A silver button."
        )
        invalid = _question(
            "fictional-late-invalid", "Where is a paper sail?", "invalid", "A paper sail."
        )
        invalid["question_type"] = "unknown-fictional-kind"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "late-invalid.json"
            output_path = root / "must-not-exist.json"
            input_path.write_text(json.dumps([selected, invalid]), encoding="utf-8")

            with patch.object(
                evaluator.tempfile,
                "TemporaryDirectory",
                side_effect=AssertionError("retrieval store created before full validation"),
            ):
                with self.assertRaisesRegex(ValueError, "documented schema"):
                    evaluator.evaluate_retrieval_file(
                        input_path,
                        namespace=_NAMESPACE,
                        variant="s",
                        question_ids=(selected["question_id"],),
                    )

            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                evaluator.main(_cli_args(input_path, output_path, selected["question_id"]))
            self.assertIn("documented schema", stderr.getvalue())
            self.assertFalse(output_path.exists())
            self.assertNotIn(str(input_path), stderr.getvalue())
            self.assertNotIn("fictional-before-error", stderr.getvalue())

    def test_request_order_does_not_change_report_question_order(self) -> None:
        first = _question(
            "fictional-order-first", "Which moss is on a stone?", "order-a", "Green moss."
        )
        second = _question(
            "fictional-order-second", "Which shell is by the shore?", "order-b", "A blue shell."
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_text(json.dumps([first, second]), encoding="utf-8")
            report = evaluator.evaluate_retrieval_file(
                path,
                namespace=_NAMESPACE,
                variant="s",
                question_ids=(second["question_id"], first["question_id"]),
            )

        self.assertEqual(
            [item["question_id"] for item in report["questions"]],
            [first["question_id"], second["question_id"]],
        )

    def test_selector_cutoff_and_query_preflight_do_not_create_stores(self) -> None:
        from unittest.mock import patch

        short = _question(
            "fictional-short", "Where is the fictional glass marble?", "short", "A glass marble."
        )
        long_selected = _question(
            "fictional-long-selected", "q" * 4097, "long", "A fictional long query."
        )
        items = [short, long_selected]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_text(json.dumps(items), encoding="utf-8")

            invalid_selections: tuple[tuple[object, str], ...] = (
                ((), "question_ids must be a nonempty tuple"),
                ([short["question_id"]], "question_ids must be a nonempty tuple"),
                (
                    (short["question_id"], short["question_id"]),
                    "tuple of at most five",
                ),
                (("missing-question",), "requested question IDs were not found"),
                (("a", "b", "c", "d", "e", "f"), "tuple of at most five"),
            )
            for selection, error in invalid_selections:
                with self.subTest(selection_type=type(selection).__name__, size=len(selection)):
                    with patch.object(
                        evaluator.tempfile,
                        "TemporaryDirectory",
                        side_effect=AssertionError(
                            "retrieval store created before selection failed"
                        ),
                    ):
                        with self.assertRaisesRegex(ValueError, error):
                            evaluator.evaluate_retrieval_file(
                                path,
                                namespace=_NAMESPACE,
                                variant="s",
                                question_ids=selection,  # type: ignore[arg-type]
                            )

            with (
                patch.object(
                    evaluator.tempfile,
                    "TemporaryDirectory",
                    side_effect=AssertionError("retrieval store created before cutoff failed"),
                ),
                patch.object(
                    evaluator,
                    "scan_longmemeval_file",
                    side_effect=AssertionError("file opened before cutoffs validated"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "cutoffs"):
                    evaluator.evaluate_retrieval_file(
                        path,
                        namespace=_NAMESPACE,
                        variant="s",
                        question_ids=(short["question_id"],),
                        cutoffs=(1, 1),
                    )

            with patch.object(
                evaluator.tempfile,
                "TemporaryDirectory",
                side_effect=AssertionError("retrieval store created before query failed"),
            ):
                with self.assertRaisesRegex(ValueError, "4096-character limit"):
                    evaluator.evaluate_retrieval_file(
                        path,
                        namespace=_NAMESPACE,
                        variant="s",
                        question_ids=(long_selected["question_id"],),
                    )

    def test_valid_unselected_overlong_query_is_not_evaluated(self) -> None:
        selected = _question(
            "fictional-short-selected", "Where is a bronze leaf?", "short-session", "A bronze leaf."
        )
        unselected = _question(
            "fictional-long-omitted", "q" * 4097, "long-session", "An omitted query."
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            path.write_text(json.dumps([selected, unselected]), encoding="utf-8")
            report = evaluator.evaluate_retrieval_file(
                path,
                namespace=_NAMESPACE,
                variant="s",
                question_ids=(selected["question_id"],),
            )

        self.assertEqual(report["counts"]["question_count"], 1)
        self.assertEqual(
            [item["question_id"] for item in report["questions"]], [selected["question_id"]]
        )

    def test_cli_selective_mode_handles_large_source_privately_and_keeps_default_cap(self) -> None:
        selected = _question(
            "fictional-large-selected",
            "Where is the amber pebble?",
            "large-selected-session",
            "An amber pebble rests in a quiet bowl.",
        )
        unselected = _question(
            "fictional-large-unselected",
            "Where is the paper comet?",
            "large-unselected-session",
            "A paper comet crosses an empty sky.",
        )
        raw = json.dumps([selected, unselected], ensure_ascii=False).encode("utf-8")
        suffix_chunk = b" " * (1024 * 1024)
        suffix_count = 65
        expected_digest = hashlib.sha256()
        expected_digest.update(raw)
        expected_bytes = len(raw) + len(suffix_chunk) * suffix_count
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "large-fictional.json"
            report_path = root / "private-report.json"
            with input_path.open("wb") as output:
                output.write(raw)
                for _ in range(suffix_count):
                    output.write(suffix_chunk)
                    expected_digest.update(suffix_chunk)
            before = input_path.stat()

            result = evaluator.main(_cli_args(input_path, report_path, selected["question_id"]))
            saved_bytes = report_path.read_bytes()
            report = json.loads(saved_bytes)
            after = input_path.stat()

            self.assertEqual(result, 0)
            self.assertGreater(before.st_size, 64 * 1024 * 1024)
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["input_bytes"], expected_bytes)
            self.assertEqual(report["input_sha256"], expected_digest.hexdigest())
            self.assertEqual(report["counts"]["question_count"], 1)
            self.assertEqual(report["selection"]["source_question_count"], 2)
            self.assertEqual(report["selection"]["evaluated_question_count"], 1)
            self.assertEqual(stat.S_IMODE(report_path.stat().st_mode), 0o600)
            self.assertEqual(after.st_size, before.st_size)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))

            unchanged_digest = hashlib.sha256()
            with input_path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    unchanged_digest.update(chunk)
            self.assertEqual(unchanged_digest.hexdigest(), expected_digest.hexdigest())

            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                evaluator.main(_cli_args(input_path, report_path, selected["question_id"]))
            self.assertIn("output path already exists", stderr.getvalue())
            self.assertEqual(report_path.read_bytes(), saved_bytes)

            default_output = root / "default-report.json"
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                evaluator.main(_cli_args(input_path, default_output))
            self.assertFalse(default_output.exists())
            self.assertIn("64 MiB", stderr.getvalue())

    def test_cli_repeatable_question_id_enables_file_selection(self) -> None:
        first = _question(
            "fictional-cli-first",
            "Where is a sapphire feather?",
            "fictional-cli-first-session",
            "A sapphire feather rests in a blue drawer.",
        )
        second = _question(
            "fictional-cli-second",
            "Where is a paper moon?",
            "fictional-cli-second-session",
            "A paper moon hangs above a desk.",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "fictional.json"
            output_path = root / "selected-report.json"
            input_path.write_text(json.dumps([first, second]), encoding="utf-8")

            result = evaluator.main(
                _cli_args(input_path, output_path, first["question_id"], second["question_id"])
            )

            self.assertEqual(result, 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(
            [question["question_id"] for question in report["questions"]],
            [first["question_id"], second["question_id"]],
        )


if __name__ == "__main__":
    unittest.main()
