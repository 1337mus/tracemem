from __future__ import annotations

import hashlib
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from experiments import analyze_qa_interpretation_answers as analysis
from experiments import prepare_qa_controls_reviews as reviews
from experiments import prepare_qa_interpretation_controls as controls
from experiments import qa_interpretation_answer_receipt as receipt_contract
from tests.test_qa_interpretation_answer_analysis import _fake_receipt


class QAControlsReviewPreparationTests(unittest.TestCase):
    def test_prepare_reviews_returns_36_distinct_blind_tasks_and_blank_rows(self) -> None:
        bundle, key, blank = reviews.prepare_reviews(_fake_receipt(), seed=17)

        self.assertEqual(bundle["kind"], "qa-interpretation-controls-answer-review")
        self.assertEqual(len(bundle["tasks"]), 36)
        task_ids = [task["task_id"] for task in bundle["tasks"]]
        self.assertEqual(len(set(task_ids)), 36)
        self.assertEqual(key["kind"], "qa-interpretation-controls-answer-review-key")
        self.assertEqual(len(key["task_keys"]), 36)
        self.assertEqual(blank["bundle_sha256"], bundle["bundle_sha256"])
        self.assertEqual(len(blank["responses"]), 36)
        self.assertEqual([row["task_id"] for row in blank["responses"]], task_ids)

    def test_visible_tasks_reconstruct_exact_context_and_keep_labels_in_key(self) -> None:
        fake_receipt = _fake_receipt()
        report = analysis.analyze_receipt(fake_receipt)
        bundle, key, _blank = reviews.prepare_reviews(fake_receipt, seed=17)
        task_keys = {row["task_id"]: row for row in key["task_keys"]}
        prompt_by_id = {
            item["prompt_id"]: item for item in receipt_contract.build_request()["items"]
        }
        output_by_id = {row["prompt_id"]: row for row in report["per_cell"]}
        groups, _counts = controls._parse_fixture(controls._FIXTURE_PATH.read_bytes())
        _payload, references, _audit = controls.prepare_controls()
        cell_by_id = {cell["cell_id"]: cell for cell in references["cells"]}
        source_by_question = {
            question["question_id"]: (group, question)
            for group in groups
            for question in group["questions"]
        }
        self.assertEqual(set(bundle), {"version", "kind", "tasks", "bundle_sha256"})
        cell_ids = set()
        for task in bundle["tasks"]:
            self.assertEqual(
                set(task),
                {"task_id", "question", "prompt_messages", "source_sessions", "raw_output"},
            )
            hidden = task_keys[task["task_id"]]
            group, question = source_by_question[hidden["question_id"]]
            self.assertEqual(task["question"], question["question"])
            self.assertEqual(task["prompt_messages"], prompt_by_id[hidden["prompt_id"]]["messages"])
            self.assertEqual(task["source_sessions"], group["source_sessions"])
            self.assertEqual(task["raw_output"], output_by_id[hidden["prompt_id"]]["raw_output"])
            self.assertEqual(
                set(hidden),
                {
                    "task_id",
                    "cell_id",
                    "scenario_id",
                    "question_id",
                    "variant_id",
                    "prompt_id",
                    "expected_prompt_answer",
                    "full_source_expected_answer",
                },
            )
            cell = cell_by_id[hidden["cell_id"]]
            for field in (
                "scenario_id",
                "question_id",
                "variant_id",
                "prompt_id",
                "expected_prompt_answer",
                "full_source_expected_answer",
            ):
                self.assertEqual(hidden[field], cell[field])
            cell_ids.add(hidden["cell_id"])
        self.assertEqual(len(cell_ids), 36)
        self.assertEqual(
            key["source_pins"]["report_sha256"],
            hashlib.sha256(receipt_contract._canonical_bytes(report)).hexdigest(),
        )
        self.assertEqual(
            key["source_pins"]["receipt_sha256"],
            report["provenance"]["receipt_sha256"],
        )
        self.assertEqual(
            key["source_pins"].get("review_preparer_source_sha256"),
            hashlib.sha256(Path(reviews.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            key["source_pins"].get("fixture_sha256"),
            hashlib.sha256(controls._FIXTURE_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(key["scope"]["human_label_count"], 0)
        self.assertEqual(key["scope"]["test"], [])
        self.assertFalse(key["scope"]["cells_are_independent_trials"])
        self.assertEqual(
            bundle["bundle_sha256"],
            hashlib.sha256(
                receipt_contract._canonical_bytes(
                    {name: bundle[name] for name in ("version", "kind", "tasks")}
                )
            ).hexdigest(),
        )

    def test_cli_writes_private_outputs_exclusively_and_preserves_receipt(self) -> None:
        fake_receipt = _fake_receipt()
        receipt_bytes = json.dumps(fake_receipt, sort_keys=True).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "fake-receipt.json"
            receipt_path.write_bytes(receipt_bytes)
            output_dir = root / "review-packet"
            args = [
                "--receipt",
                str(receipt_path),
                "--output-dir",
                str(output_dir),
                "--seed",
                "17",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(reviews.main(args), 0)
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertEqual(stat.S_IMODE(output_dir.stat().st_mode), 0o700)
            for name in ("bundle.json", "key.json", "response-template.json"):
                self.assertEqual(stat.S_IMODE((output_dir / name).stat().st_mode), 0o600)
            original = {
                name: (output_dir / name).read_bytes()
                for name in ("bundle.json", "key.json", "response-template.json")
            }
            with redirect_stderr(io.StringIO()):
                self.assertEqual(reviews.main(args), 2)
            self.assertEqual(
                {name: (output_dir / name).read_bytes() for name in original}, original
            )
            linked_output = root / "linked-output"
            linked_output.symlink_to(output_dir, target_is_directory=True)
            linked_args = [*args]
            linked_args[linked_args.index(str(output_dir))] = str(linked_output)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(reviews.main(linked_args), 2)
            self.assertEqual(
                {name: (output_dir / name).read_bytes() for name in original}, original
            )
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)

    def test_seed_changes_order_without_changing_task_identity(self) -> None:
        fake_receipt = _fake_receipt()
        first, first_key, first_blank = reviews.prepare_reviews(fake_receipt, seed=17)
        repeat, repeat_key, repeat_blank = reviews.prepare_reviews(fake_receipt, seed=17)
        shuffled, shuffled_key, shuffled_blank = reviews.prepare_reviews(fake_receipt, seed=19)

        self.assertEqual(first, repeat)
        self.assertEqual(first_key, repeat_key)
        self.assertEqual(first_blank, repeat_blank)
        self.assertNotEqual(first["tasks"], shuffled["tasks"])
        first_ids = [task["task_id"] for task in first["tasks"]]
        shuffled_ids = [task["task_id"] for task in shuffled["tasks"]]
        self.assertEqual(set(first_ids), set(shuffled_ids))
        self.assertEqual([row["task_id"] for row in first_blank["responses"]], first_ids)
        self.assertEqual([row["task_id"] for row in shuffled_blank["responses"]], shuffled_ids)
        self.assertEqual(first_key["seed"], 17)
        self.assertEqual(shuffled_key["seed"], 19)
        for invalid_seed in (True, 17.0, "17"):
            with self.subTest(seed=invalid_seed), self.assertRaises(TypeError):
                reviews.prepare_reviews(fake_receipt, seed=invalid_seed)

    def test_mutating_returned_scope_cannot_poison_future_packets(self) -> None:
        fake_receipt = _fake_receipt()
        first_bundle, first_key, _first_blank = reviews.prepare_reviews(fake_receipt, seed=17)
        bundle_before = deepcopy(first_bundle)
        first_key["scope"]["split"] = "test"
        first_key["scope"]["scenario_groups"] = 999
        first_key["scope"]["test"].append("test-only")

        second_bundle, second_key, _second_blank = reviews.prepare_reviews(fake_receipt, seed=17)

        self.assertEqual(second_bundle, bundle_before)
        self.assertEqual(
            second_key["scope"],
            {
                "split": "dev",
                "test": [],
                "human_label_count": 0,
                "scenario_groups": 6,
                "questions": 12,
                "cells": 36,
                "unique_prompts": 36,
                "correlated_groups": 6,
                "cells_are_independent_trials": False,
            },
        )

    def test_altered_missing_and_duplicate_receipts_are_rejected(self) -> None:
        cases = []
        altered = _fake_receipt()
        altered["request_sha256"] = "0" * 64
        cases.append(altered)
        missing = _fake_receipt()
        missing["result"]["outputs"].pop()
        cases.append(missing)
        duplicate = _fake_receipt()
        duplicate["result"]["outputs"][1]["prompt_id"] = duplicate["result"]["outputs"][0][
            "prompt_id"
        ]
        cases.append(duplicate)
        for fake_receipt in cases:
            with self.subTest(receipt=fake_receipt), self.assertRaises(ValueError):
                reviews.prepare_reviews(fake_receipt, seed=17)

    def test_cli_rejects_invalid_receipt_before_creating_output_directory(self) -> None:
        fake_receipt = _fake_receipt()
        fake_receipt["request_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "invalid-receipt.json"
            receipt_path.write_text(json.dumps(fake_receipt), encoding="utf-8")
            output_dir = root / "must-not-exist"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    reviews.main(
                        [
                            "--receipt",
                            str(receipt_path),
                            "--output-dir",
                            str(output_dir),
                            "--seed",
                            "17",
                        ]
                    ),
                    2,
                )
            self.assertFalse(output_dir.exists())
            self.assertEqual(receipt_path.read_text(encoding="utf-8"), json.dumps(fake_receipt))

    def test_cli_retains_partial_export_and_collision_file_on_writer_failure(self) -> None:
        fake_receipt = _fake_receipt()
        receipt_bytes = json.dumps(fake_receipt, sort_keys=True).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "fake-receipt.json"
            receipt_path.write_bytes(receipt_bytes)
            output_dir = root / "partial-review-packet"
            args = [
                "--receipt",
                str(receipt_path),
                "--output-dir",
                str(output_dir),
                "--seed",
                "17",
            ]
            real_writer = reviews._analysis.write_analysis
            calls = 0

            def collide_after_first_file(path: Path, document: dict[str, object]) -> None:
                nonlocal calls
                if calls == 0:
                    calls += 1
                    real_writer(path, document)
                    return
                if calls == 1:
                    calls += 1
                    path.write_text("foreign collision", encoding="utf-8")
                real_writer(path, document)

            with (
                patch.object(
                    reviews._analysis, "write_analysis", side_effect=collide_after_first_file
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(reviews.main(args), 2)
            self.assertEqual(calls, 2)
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
            self.assertTrue(output_dir.exists())
            surviving_bundle = json.loads((output_dir / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(surviving_bundle["kind"], "qa-interpretation-controls-answer-review")
            self.assertEqual(len(surviving_bundle["tasks"]), 36)
            self.assertEqual(
                (output_dir / "key.json").read_text(encoding="utf-8"), "foreign collision"
            )
            self.assertFalse((output_dir / "response-template.json").exists())

    def test_blank_response_documents_choices_outside_annotation_rows(self) -> None:
        _bundle, _key, blank = reviews.prepare_reviews(_fake_receipt(), seed=17)

        self.assertEqual(
            blank["allowed_choices"],
            {
                "source_answer": ["correct", "incorrect", "uncertain"],
                "prompt_support": ["supported", "unsupported", "uncertain"],
                "reviewer_kind": ["human", "agent"],
            },
        )
        for row in blank["responses"]:
            self.assertEqual(
                set(row),
                {
                    "task_id",
                    "source_answer",
                    "prompt_support",
                    "rationale",
                    "reviewer_id",
                    "reviewer_kind",
                    "timestamp",
                },
            )
            self.assertTrue(all(value == "" for field, value in row.items() if field != "task_id"))


if __name__ == "__main__":
    unittest.main()
