from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_qa_contrast_reviews as qa_reviews
from tests.test_qa_contrast_answer_analysis import _fixture_receipt


def _completed_response(
    template: dict[str, object], reviewer_id: str = "reviewer-a"
) -> dict[str, object]:
    response = copy.deepcopy(template)
    for row in response["responses"]:  # type: ignore[union-attr]
        row.update(
            source_answer="correct",
            prompt_support="supported",
            rationale="The label follows from the frozen review task.",
            reviewer_id=reviewer_id,
            reviewer_kind="human",
            timestamp="2026-10-15T00:00:00Z",
        )
    return response


class QAContrastReviewAnalysisTests(unittest.TestCase):
    def test_unique_task_counts_include_zeroes_for_each_review_label(self) -> None:
        from experiments.analyze_qa_contrast_reviews import analyze_reviews

        receipt, _audit, _references = _fixture_receipt()
        _bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=41)
        response = _completed_response(template)

        report = analyze_reviews(receipt, [response], seed=41)

        reviewer = report["reviewers"][0]
        self.assertEqual(reviewer["unique_task_summary"]["denominator"], 46)
        self.assertEqual(
            reviewer["unique_task_summary"]["source_answer_counts"],
            {"correct": 46, "incorrect": 0, "uncertain": 0},
        )
        self.assertEqual(
            reviewer["unique_task_summary"]["prompt_support_counts"],
            {"supported": 46, "unsupported": 0, "uncertain": 0},
        )

    def test_expanded_cells_join_asymmetric_labels_and_keep_correlated_strata(self) -> None:
        from collections import Counter

        from experiments.analyze_qa_contrast_reviews import analyze_reviews

        receipt, _audit, _references = _fixture_receipt()
        _bundle, key, template = qa_reviews.prepare_reviews(receipt, seed=43)
        response = _completed_response(template)
        source_labels = ("correct", "incorrect", "uncertain")
        support_labels = ("supported", "unsupported", "uncertain")
        response_by_task = {row["task_id"]: row for row in response["responses"]}
        task_key = {row["task_id"]: row for row in key["tasks"]}
        for index, task_id in enumerate(sorted(task_key)):
            response_by_task[task_id]["source_answer"] = source_labels[index % 3]
            response_by_task[task_id]["prompt_support"] = support_labels[(index * 2 + 1) % 3]
        response["responses"].reverse()

        report = analyze_reviews(receipt, [response], seed=43)

        expanded = report["reviewers"][0]["expanded_cells"]
        self.assertEqual(expanded["denominator"], 72)
        self.assertEqual(expanded["unique_prompt_count"], 46)
        self.assertEqual(len(expanded["cells"]), 72)
        expected_cells = {}
        for task in key["tasks"]:
            source = response_by_task[task["task_id"]]
            for mapping in task["cells"]:
                expected_cells[mapping["cell_id"]] = {
                    "task_id": task["task_id"],
                    "prompt_id": task["prompt_id"],
                    "question_id": task["question_id"],
                    "cell_id": mapping["cell_id"],
                    "mode": mapping["mode"],
                    "ceiling": mapping["ceiling"],
                    "is_answerable": task["is_answerable"],
                    "source_answer": source["source_answer"],
                    "prompt_support": source["prompt_support"],
                }
        reported_cells = {cell["cell_id"]: cell for cell in expanded["cells"]}
        self.assertEqual(len(reported_cells), 72)
        self.assertEqual(reported_cells, expected_cells)
        self.assertEqual(
            [(row["mode"], row["ceiling"]) for row in expanded["groups"]],
            sorted(
                {(cell["mode"], cell["ceiling"]) for row in key["tasks"] for cell in row["cells"]}
            ),
        )
        for cell in expanded["cells"]:
            source = response_by_task[cell["task_id"]]
            provenance = task_key[cell["task_id"]]
            self.assertEqual(cell["prompt_id"], provenance["prompt_id"])
            self.assertEqual(cell["question_id"], provenance["question_id"])
            self.assertEqual(cell["is_answerable"], provenance["is_answerable"])
            self.assertEqual(cell["source_answer"], source["source_answer"])
            self.assertEqual(cell["prompt_support"], source["prompt_support"])
            self.assertEqual(
                set(cell),
                {
                    "task_id",
                    "prompt_id",
                    "question_id",
                    "cell_id",
                    "mode",
                    "ceiling",
                    "is_answerable",
                    "source_answer",
                    "prompt_support",
                },
            )

        for group in expanded["groups"]:
            rows = [
                (task, mapping)
                for task in key["tasks"]
                for mapping in task["cells"]
                if mapping["mode"] == group["mode"] and mapping["ceiling"] == group["ceiling"]
            ]
            group_responses = [response_by_task[task["task_id"]] for task, _mapping in rows]
            self.assertEqual(group["denominators"], {"total": 6, "answerable": 4, "unknown": 2})
            self.assertEqual(
                group["unique_prompt_count"], len({task["prompt_id"] for task, _ in rows})
            )
            self.assertEqual(
                group["source_answer_counts"],
                {
                    label: Counter(row["source_answer"] for row in group_responses)[label]
                    for label in source_labels
                },
            )
            self.assertEqual(
                group["prompt_support_counts"],
                {
                    label: Counter(row["prompt_support"] for row in group_responses)[label]
                    for label in support_labels
                },
            )

    def test_reviewers_remain_separate_and_provenance_is_input_order_independent(self) -> None:
        from experiments import analyze_qa_contrast_reviews as analyzer

        receipt, _audit, _references = _fixture_receipt()
        _bundle, key, template = qa_reviews.prepare_reviews(receipt, seed=47)
        response_a = _completed_response(template, "reviewer-a")
        response_b = _completed_response(template, "reviewer-b")
        for row in response_b["responses"]:
            row.update(
                source_answer="incorrect", prompt_support="unsupported", reviewer_kind="agent"
            )

        report = analyzer.analyze_reviews(receipt, [response_b, response_a], seed=47)

        self.assertEqual(
            analyzer.analyze_reviews(receipt, [response_a, response_b], seed=47), report
        )
        self.assertEqual(
            [row["reviewer_id"] for row in report["reviewers"]], ["reviewer-a", "reviewer-b"]
        )
        self.assertEqual(
            report["reviewers"][0]["unique_task_summary"]["source_answer_counts"],
            {"correct": 46, "incorrect": 0, "uncertain": 0},
        )
        self.assertEqual(
            report["reviewers"][1]["unique_task_summary"]["source_answer_counts"],
            {"correct": 0, "incorrect": 46, "uncertain": 0},
        )
        self.assertEqual(report["reviewers"][0]["reviewer_kind_self_attested"], "human")
        self.assertEqual(report["independent_human_reviewers_verified"], 0)
        self.assertEqual(report["authored_reference_human_label_count"], key["human_label_count"])
        self.assertNotIn("human_label_count", report)
        self.assertEqual(report["dev_only"], True)
        self.assertEqual(report["provisional"], True)
        self.assertEqual(report["test"], [])
        self.assertEqual(report["independent_trials"], False)
        self.assertNotIn("The label follows from the frozen review task.", json.dumps(report))
        self.assertNotIn('"rationale"', json.dumps(report))

        def canonical_hash(document: dict[str, object]) -> str:
            payload = json.dumps(
                document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        provenance = report["provenance"]
        self.assertEqual(provenance["seed"], 47)
        self.assertEqual(provenance["bundle_sha256"], key["bundle_sha256"])
        self.assertEqual(provenance["receipt_sha256"], key["receipt_sha256"])
        self.assertEqual(
            provenance["literal_analysis_report_sha256"], key["literal_analysis_report_sha256"]
        )
        self.assertEqual(
            provenance["review_preparer_source_sha256"], key["review_preparer_source_sha256"]
        )
        self.assertEqual(
            provenance["accounting_tool_source_sha256"],
            hashlib.sha256(Path(analyzer.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["response_documents"],
            [
                {"reviewer_id": "reviewer-a", "sha256": canonical_hash(response_a)},
                {"reviewer_id": "reviewer-b", "sha256": canonical_hash(response_b)},
            ],
        )

    def test_summary_and_group_counts_are_reported_within_each_answerability_stratum(self) -> None:
        from collections import Counter

        from experiments.analyze_qa_contrast_reviews import analyze_reviews

        receipt, _audit, _references = _fixture_receipt()
        _bundle, key, template = qa_reviews.prepare_reviews(receipt, seed=53)
        response = _completed_response(template)
        response_by_task = {row["task_id"]: row for row in response["responses"]}
        for task in key["tasks"]:
            row = response_by_task[task["task_id"]]
            row["source_answer"] = "correct" if task["is_answerable"] else "uncertain"
            row["prompt_support"] = "supported" if task["is_answerable"] else "unsupported"

        report = analyze_reviews(receipt, [response], seed=53)
        reviewer = report["reviewers"][0]

        def assert_strata(accounting: dict[str, object], rows: list[dict[str, object]]) -> None:
            labels_a = ("correct", "incorrect", "uncertain")
            labels_b = ("supported", "unsupported", "uncertain")
            expected = {
                "all": rows,
                "answerable": [row for row in rows if row["is_answerable"]],
                "unknown": [row for row in rows if not row["is_answerable"]],
            }
            for name, subset in expected.items():
                actual = accounting["strata"][name]
                self.assertEqual(actual["denominator"], len(subset))
                self.assertEqual(
                    actual["source_answer_counts"],
                    {
                        label: Counter(row["source_answer"] for row in subset)[label]
                        for label in labels_a
                    },
                )
                self.assertEqual(
                    actual["prompt_support_counts"],
                    {
                        label: Counter(row["prompt_support"] for row in subset)[label]
                        for label in labels_b
                    },
                )
                self.assertEqual(
                    actual["joint_counts"],
                    {
                        source: {
                            support: sum(
                                row["source_answer"] == source and row["prompt_support"] == support
                                for row in subset
                            )
                            for support in labels_b
                        }
                        for source in labels_a
                    },
                )

        task_rows = [
            {
                **response_by_task[task["task_id"]],
                "is_answerable": task["is_answerable"],
            }
            for task in key["tasks"]
        ]
        expanded_rows = [
            {
                **response_by_task[task["task_id"]],
                "is_answerable": task["is_answerable"],
            }
            for task in key["tasks"]
            for _cell in task["cells"]
        ]
        assert_strata(reviewer["unique_task_summary"], task_rows)
        assert_strata(reviewer["expanded_cells"], expanded_rows)
        for group in reviewer["expanded_cells"]["groups"]:
            group_rows = [
                {
                    **response_by_task[task["task_id"]],
                    "is_answerable": task["is_answerable"],
                }
                for task in key["tasks"]
                for cell in task["cells"]
                if cell["mode"] == group["mode"] and cell["ceiling"] == group["ceiling"]
            ]
            assert_strata(group, group_rows)
            self.assertEqual(
                {
                    name: group["strata"][name]["denominator"]
                    for name in ("all", "answerable", "unknown")
                },
                {"all": 6, "answerable": 4, "unknown": 2},
            )

    def test_duplicate_reviewer_id_is_rejected_even_when_kinds_differ(self) -> None:
        from experiments.analyze_qa_contrast_reviews import analyze_reviews

        receipt, _audit, _references = _fixture_receipt()
        _bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=59)
        response_a = _completed_response(template, "reviewer-same")
        response_b = _completed_response(template, "reviewer-same")
        for row in response_b["responses"]:
            row["reviewer_kind"] = "agent"

        with self.assertRaisesRegex(ValueError, "duplicate reviewer ID"):
            analyze_reviews(receipt, [response_a, response_b], seed=59)

    def test_incomplete_response_document_is_rejected_and_empty_input_is_invalid(self) -> None:
        from experiments.analyze_qa_contrast_reviews import analyze_reviews

        receipt, _audit, _references = _fixture_receipt()
        _bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=61)
        response = _completed_response(template)
        response["responses"].pop()

        with self.assertRaisesRegex(ValueError, "missing response task ID"):
            analyze_reviews(receipt, [response], seed=61)
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            analyze_reviews(receipt, [], seed=61)

    def test_cli_writes_private_report_exclusively_without_heavy_model_imports(self) -> None:
        from experiments import analyze_qa_contrast_reviews as analyzer

        receipt, _audit, _references = _fixture_receipt()
        _bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=67)
        response = _completed_response(template)
        response_b = _completed_response(template, "reviewer-b")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            response_path, response_path_b = root / "responses.json", root / "responses-b.json"
            output_path = root / "report.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            response_path.write_text(json.dumps(response), encoding="utf-8")
            response_path_b.write_text(json.dumps(response_b), encoding="utf-8")
            original_import = builtins.__import__

            def reject_heavy_import(name: str, *args: object, **kwargs: object) -> object:
                if name.split(".", 1)[0] in {"modal", "torch", "transformers", "tokenizers"}:
                    raise AssertionError(f"offline review accounting imported {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=reject_heavy_import):
                status = analyzer.main(
                    [
                        str(receipt_path),
                        "--seed",
                        "67",
                        "--responses",
                        str(response_path),
                        "--responses",
                        str(response_path_b),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["reviewer_count"], 2
            )

            output_path.write_text("keep", encoding="utf-8")
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = analyzer.main(
                    [
                        str(receipt_path),
                        "--seed",
                        "67",
                        "--responses",
                        str(response_path),
                        "--responses",
                        str(response_path_b),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep")
            self.assertNotIn("Traceback", errors.getvalue())

    def test_cli_rejects_duplicate_json_and_symlink_output_without_tracebacks(self) -> None:
        from experiments import analyze_qa_contrast_reviews as analyzer

        receipt, _audit, _references = _fixture_receipt()
        _bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=71)
        response = _completed_response(template)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path, response_path = root / "receipt.json", root / "responses.json"
            receipt_path.write_text('{"x":1,"x":2}', encoding="utf-8")
            response_path.write_text(json.dumps(response), encoding="utf-8")
            duplicate_output = root / "duplicate-report.json"
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = analyzer.main(
                    [
                        str(receipt_path),
                        "--seed",
                        "71",
                        "--responses",
                        str(response_path),
                        "--output",
                        str(duplicate_output),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(duplicate_output.exists())
            self.assertIn("duplicate JSON key", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())

            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            target = root / "target.json"
            target.write_text("keep", encoding="utf-8")
            symlink_output = root / "report-link.json"
            symlink_output.symlink_to(target)
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = analyzer.main(
                    [
                        str(receipt_path),
                        "--seed",
                        "71",
                        "--responses",
                        str(response_path),
                        "--output",
                        str(symlink_output),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")
            self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
