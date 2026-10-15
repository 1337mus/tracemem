from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_qa_contrast_requests
from experiments import prepare_qa_contrast_reviews as qa_reviews
from tests.test_qa_contrast_answer_analysis import _fixture_receipt


class QAContrastReviewPreparationTests(unittest.TestCase):
    def test_bundle_preserves_exact_unique_answers_and_separates_provenance(self) -> None:
        receipt, audit, references = _fixture_receipt()
        payload, _references, _audit = prepare_qa_contrast_requests.prepare_requests()
        from experiments.analyze_qa_contrast_answers import analyze_receipt

        literal_analysis = analyze_receipt(receipt)
        expected_report_bytes = (
            json.dumps(literal_analysis, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

        bundle, key, template = qa_reviews.prepare_reviews(receipt, seed=17)

        self.assertEqual(bundle["version"], 1)
        self.assertEqual(bundle["kind"], "qa-contrast-answer-review")
        self.assertEqual(len(bundle["tasks"]), 46)
        self.assertEqual(len({task["task_id"] for task in bundle["tasks"]}), 46)
        self.assertEqual(len(key["tasks"]), 46)
        self.assertEqual(sum(len(row["cells"]) for row in key["tasks"]), 72)
        self.assertEqual(key["human_label_count"], 0)
        self.assertEqual(key["dev_only"], True)
        self.assertEqual(key["test"], [])
        self.assertEqual(
            key["review_preparer_source_sha256"],
            hashlib.sha256(Path(qa_reviews.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            key.get("literal_analysis_report_sha256"),
            hashlib.sha256(expected_report_bytes).hexdigest(),
        )

        prompt_by_id = {item["prompt_id"]: item for item in payload["items"]}
        output_by_id = {row["prompt_id"]: row["raw_output"] for row in receipt["result"]["outputs"]}
        refs_by_id = {row["question_id"]: row for row in references["questions"]}
        task_rows = {row["task_id"]: row for row in key["tasks"]}
        fixture = json.loads(prepare_qa_contrast_requests._FIXTURE_PATH.read_text(encoding="utf-8"))
        fixture_by_id = {row["item"]["question_id"]: row["item"] for row in fixture["cases"]}
        for task in bundle["tasks"]:
            provenance = task_rows[task["task_id"]]
            prompt_id = provenance["prompt_id"]
            question_ref = refs_by_id[provenance["question_id"]]
            fixture_item = fixture_by_id[provenance["question_id"]]
            self.assertEqual(task["prompt_messages"], prompt_by_id[prompt_id]["messages"])
            self.assertEqual(task["raw_output"], output_by_id[prompt_id])
            self.assertEqual(task["question"], fixture_item["question"])
            self.assertEqual(task["question_date"], fixture_item["question_date"])
            self.assertEqual(
                task["source_sessions"],
                [
                    {
                        "date": date,
                        "messages": [
                            {"role": message["role"], "content": message["content"]}
                            for message in messages
                        ],
                    }
                    for date, messages in zip(
                        fixture_item["haystack_dates"],
                        fixture_item["haystack_sessions"],
                        strict=True,
                    )
                ],
            )
            self.assertEqual(
                provenance["cells"],
                [
                    {"cell_id": row["cell_id"], "mode": row["mode"], "ceiling": row["ceiling"]}
                    for row in audit["cell_mapping"]
                    if row["prompt_id"] == prompt_id
                ],
            )
            self.assertEqual(provenance["authored_reference"], question_ref["answer"])
            self.assertEqual(
                set(task),
                {
                    "task_id",
                    "question",
                    "question_date",
                    "prompt_messages",
                    "source_sessions",
                    "raw_output",
                },
            )
            self.assertTrue(
                all(set(message) == {"role", "content"} for message in task["prompt_messages"])
            )
            self.assertTrue(
                all(
                    set(session) == {"date", "messages"}
                    and all(set(message) == {"role", "content"} for message in session["messages"])
                    for session in task["source_sessions"]
                )
            )

        self.assertEqual(template["kind"], "qa-contrast-answer-review-responses")
        self.assertEqual(template["bundle_sha256"], bundle["bundle_sha256"])
        self.assertEqual(len(template["responses"]), 46)
        for row in template["responses"]:
            self.assertEqual(row["task_id"] in task_rows, True)
            self.assertTrue(
                all(
                    row[field] == ""
                    for field in (
                        "source_answer",
                        "prompt_support",
                        "rationale",
                        "reviewer_id",
                        "reviewer_kind",
                        "timestamp",
                    )
                )
            )

        self.assertEqual(len(audit["cell_mapping"]), 72)

    def test_task_ids_are_content_stable_while_shuffle_uses_seed(self) -> None:
        receipt, _audit, _references = _fixture_receipt()

        bundle_a, key_a, template_a = qa_reviews.prepare_reviews(receipt, seed=17)
        bundle_b, key_b, template_b = qa_reviews.prepare_reviews(receipt, seed=17)
        bundle_c, _key_c, _template_c = qa_reviews.prepare_reviews(receipt, seed=19)

        self.assertEqual(bundle_a, bundle_b)
        self.assertEqual(key_a, key_b)
        self.assertEqual(template_a, template_b)
        self.assertNotEqual(bundle_a["tasks"], bundle_c["tasks"])
        self.assertEqual(
            {row["task_id"] for row in bundle_a["tasks"]},
            {row["task_id"] for row in bundle_c["tasks"]},
        )
        self.assertEqual(key_a["seed"], 17)
        self.assertEqual(key_b["seed"], 17)

    def test_response_round_trip_keeps_axes_separate_and_does_not_mutate_input(self) -> None:
        receipt, _audit, _references = _fixture_receipt()
        bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=23)
        response = copy.deepcopy(template)
        for row in response["responses"]:
            row.update(
                source_answer="uncertain",
                prompt_support="supported",
                rationale="The source supports this response, but the answer is unclear.",
                reviewer_id="reviewer-x",
                reviewer_kind="agent",
                timestamp="2026-10-15T00:00:00Z",
            )
        response["responses"].reverse()
        original = copy.deepcopy(response)

        rows = qa_reviews.validate_responses(receipt, response, seed=23)

        self.assertEqual(response, original)
        self.assertEqual(
            [row["task_id"] for row in rows], [row["task_id"] for row in bundle["tasks"]]
        )
        self.assertEqual({row["source_answer"] for row in rows}, {"uncertain"})
        self.assertEqual({row["prompt_support"] for row in rows}, {"supported"})
        with self.assertRaisesRegex(ValueError, "bundle hash does not match"):
            qa_reviews.validate_responses(receipt, response, seed=24)
        changed_receipt = copy.deepcopy(receipt)
        changed_receipt["result"]["outputs"][0]["raw_output"] += " changed"
        with self.assertRaisesRegex(ValueError, "bundle hash does not match"):
            qa_reviews.validate_responses(changed_receipt, response, seed=23)

    def test_response_validator_rejects_invalid_documents_in_table(self) -> None:
        receipt, _audit, _references = _fixture_receipt()
        bundle, key, template = qa_reviews.prepare_reviews(receipt, seed=29)
        for invalid_seed in (True, 29.0, "29"):
            with (
                self.subTest(seed=invalid_seed),
                self.assertRaisesRegex(TypeError, "seed must be an integer"),
            ):
                qa_reviews.prepare_reviews(receipt, seed=invalid_seed)  # type: ignore[arg-type]
        valid = _completed_response(template)
        task_ids = [task["task_id"] for task in bundle["tasks"]]

        def change(mutator: object) -> dict[str, object]:
            value = copy.deepcopy(valid)
            mutator(value)  # type: ignore[operator]
            return value

        cases = [
            (
                "wrong kind",
                change(lambda value: value.update(kind="claim-review")),
                "unsupported QA review response document kind or version",
            ),
            (
                "wrong hash",
                change(lambda value: value.update(bundle_sha256="0" * 64)),
                "response bundle hash does not match",
            ),
            (
                "boolean version",
                change(lambda value: value.update(version=True)),
                "unsupported QA review response document kind or version",
            ),
            (
                "missing document field",
                change(lambda value: value.pop("kind")),
                "response document has missing or unexpected fields",
            ),
            (
                "extra document field",
                change(lambda value: value.update(extra=True)),
                "response document has missing or unexpected fields",
            ),
            ("blank template", copy.deepcopy(template), "source_answer label is invalid"),
            (
                "unknown task",
                change(lambda value: value["responses"][0].update(task_id="missing")),
                "unknown task ID",
            ),
            (
                "missing task",
                change(lambda value: value["responses"].pop()),
                "missing response task ID",
            ),
            (
                "duplicate task",
                change(
                    lambda value: value["responses"].__setitem__(
                        1, copy.deepcopy(value["responses"][0])
                    )
                ),
                "duplicate task ID",
            ),
            (
                "extra response field",
                change(lambda value: value["responses"][0].update(extra=True)),
                "response row has missing or unexpected fields",
            ),
            (
                "invalid source answer label",
                change(lambda value: value["responses"][0].update(source_answer="supported")),
                "source_answer label is invalid",
            ),
            (
                "invalid prompt support label",
                change(lambda value: value["responses"][0].update(prompt_support="correct")),
                "prompt_support label is invalid",
            ),
            (
                "blank rationale",
                change(lambda value: value["responses"][0].update(rationale="  ")),
                "rationale must be non-blank",
            ),
            (
                "blank reviewer ID",
                change(lambda value: value["responses"][0].update(reviewer_id="")),
                "reviewer_id must be non-blank",
            ),
            (
                "invalid reviewer kind",
                change(lambda value: value["responses"][0].update(reviewer_kind="verified-human")),
                "reviewer_kind must be human or agent",
            ),
            (
                "mixed reviewer ID",
                change(lambda value: value["responses"][1].update(reviewer_id="reviewer-y")),
                "single reviewer ID and kind",
            ),
            (
                "mixed reviewer kind",
                change(lambda value: value["responses"][1].update(reviewer_kind="human")),
                "single reviewer ID and kind",
            ),
            (
                "invalid timestamp",
                change(lambda value: value["responses"][0].update(timestamp="not-a-time")),
                "timestamp must use canonical UTC format",
            ),
        ]
        self.assertEqual(set(task_ids), {row["task_id"] for row in valid["responses"]})
        with patch.object(qa_reviews, "prepare_reviews", return_value=(bundle, key, template)):
            for label, document, error in cases:
                with (
                    self.subTest(case=label),
                    self.assertRaisesRegex((TypeError, ValueError), error),
                ):
                    qa_reviews.validate_responses(receipt, document, seed=29)

        with self.assertRaisesRegex(ValueError, "bundle hash does not match"):
            qa_reviews.validate_responses(receipt, valid, seed=30)

    def test_cli_export_is_exclusive_private_and_strictly_reads_json(self) -> None:
        receipt, _audit, _references = _fixture_receipt()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            output_dir = root / "reviews"
            status = qa_reviews.main(
                ["export", str(receipt_path), "--seed", "31", "--output-dir", str(output_dir)]
            )
            self.assertEqual(status, 0)
            self.assertEqual(stat.S_IMODE(output_dir.stat().st_mode), 0o700)
            for filename in ("bundle.json", "key.json", "response-template.json"):
                self.assertEqual(stat.S_IMODE((output_dir / filename).stat().st_mode), 0o600)

            errors = io.StringIO()
            with redirect_stderr(errors):
                status = qa_reviews.main(
                    ["export", str(receipt_path), "--seed", "31", "--output-dir", str(output_dir)]
                )
            self.assertEqual(status, 2)
            self.assertIn("refusing to overwrite", errors.getvalue())

            errors = io.StringIO()
            with redirect_stderr(errors):
                status = qa_reviews.main(
                    ["export", str(receipt_path), "--seed", "31", "--output-dir", str(receipt_path)]
                )
            self.assertEqual(status, 2)
            self.assertIn("refusing to overwrite", errors.getvalue())

            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text('{"x":1,"x":2}', encoding="utf-8")
            untouched_dir = root / "untouched"
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = qa_reviews.main(
                    [
                        "export",
                        str(duplicate_path),
                        "--seed",
                        "31",
                        "--output-dir",
                        str(untouched_dir),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(untouched_dir.exists())
            self.assertNotIn("Traceback", errors.getvalue())

            symlink = root / "reviews-link"
            symlink.symlink_to(output_dir, target_is_directory=True)
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = qa_reviews.main(
                    ["export", str(receipt_path), "--seed", "31", "--output-dir", str(symlink)]
                )
            self.assertEqual(status, 2)
            self.assertIn("refusing to overwrite", errors.getvalue())

    def test_cli_validation_is_offline_and_reports_only_descriptive_counts(self) -> None:
        receipt, _audit, _references = _fixture_receipt()
        _bundle, _key, template = qa_reviews.prepare_reviews(receipt, seed=37)
        response = _completed_response(template)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            response_path = root / "responses.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            response_path.write_text(json.dumps(response), encoding="utf-8")
            stdout = io.StringIO()
            original_import = builtins.__import__

            def guarded_import(name: str, *args: object, **kwargs: object) -> object:
                if name.split(".", 1)[0] in {"modal", "torch", "transformers", "tokenizers"}:
                    raise AssertionError(f"unexpected heavy import: {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=guarded_import), redirect_stdout(stdout):
                status = qa_reviews.main(
                    [
                        "validate",
                        str(receipt_path),
                        "--seed",
                        "37",
                        "--responses",
                        str(response_path),
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["validated_response_count"], 46)
            self.assertEqual(summary["reviewer_kind_self_attested"], "agent")
            self.assertEqual(summary["source_answer_label_counts"], {"correct": 46})
            self.assertEqual(summary["prompt_support_label_counts"], {"supported": 46})
            self.assertNotIn("accuracy", stdout.getvalue().casefold())


def _completed_response(template: dict[str, object]) -> dict[str, object]:
    response = copy.deepcopy(template)
    for row in response["responses"]:  # type: ignore[union-attr]
        row.update(
            source_answer="correct",
            prompt_support="supported",
            rationale="The answer follows from the fictional source messages.",
            reviewer_id="reviewer-x",
            reviewer_kind="agent",
            timestamp="2026-10-15T00:00:00Z",
        )
    return response


if __name__ == "__main__":
    unittest.main()
