from __future__ import annotations

import builtins
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_qa_contrast_requests


class QAContrastAnswerAnalysisTests(unittest.TestCase):
    def test_literal_match_collapses_case_and_whitespace_but_keeps_punctuation(self) -> None:
        from experiments.analyze_qa_contrast_answers import _literal_match

        self.assertTrue(_literal_match("  North\n  Star ", "north star"))
        self.assertFalse(_literal_match("3:00 PM.", "3:00 PM"))

    def test_analysis_scores_literal_unknown_and_maps_reused_outputs_to_all_groups(self) -> None:
        from experiments.analyze_qa_contrast_answers import analyze_receipt

        receipt, audit, references = _fixture_receipt()
        outputs = {row["prompt_id"]: row for row in receipt["result"]["outputs"]}
        answerable = next(row for row in references["questions"] if not row["is_abstention"])
        unknown = next(row for row in references["questions"] if row["is_abstention"])
        answerable_prompt = next(
            row["prompt_id"]
            for row in audit["cell_mapping"]
            if row["question_id"] == answerable["question_id"]
        )
        unknown_prompt = next(
            row["prompt_id"]
            for row in audit["cell_mapping"]
            if row["question_id"] == unknown["question_id"]
        )
        outputs[answerable_prompt]["raw_output"] = " UNKNOWN  "
        outputs[unknown_prompt]["raw_output"] = "not known"

        report = analyze_receipt(receipt)

        self.assertEqual((report["question_count"], report["cell_count"]), (6, 72))
        self.assertEqual(report["unique_prompt_count"], 46)
        self.assertEqual(report["unique_prompt_token_totals"]["input_tokens"], 23533)
        self.assertEqual(report["answer_reuse"]["independent_trials"], False)
        self.assertEqual(report["test"], [])
        self.assertEqual(report["human_label_count"], 0)
        self.assertEqual(len(report["groups"]), 12)
        self.assertEqual({row["n"] for row in report["groups"]}, {6})
        self.assertEqual(sum(group["false_unknown_on_answerable"] for group in report["groups"]), 3)
        self.assertEqual(
            sum(group["non_unknown_outputs_on_unknown"] for group in report["groups"]), 3
        )
        self.assertEqual(
            sum(group["token_totals"]["input_tokens"] for group in report["groups"]), 28944
        )
        for group in report["groups"]:
            self.assertEqual(group["denominators"], {"all": 6, "answerable": 4, "unknown": 2})
        changed_group = next(
            row for row in report["groups"] if row["mode"] == "bm25" and row["ceiling"] == 128
        )
        self.assertEqual(
            changed_group["exact_match"]["all_reference"],
            {"numerator": 4, "denominator": 6, "rate": 4 / 6},
        )
        self.assertEqual(changed_group["exact_match"]["answerable"]["numerator"], 3)
        self.assertEqual(changed_group["exact_match"]["unknown"]["numerator"], 1)
        self.assertEqual(len({row["prompt_id"] for row in report["cells"]}), 46)
        self.assertTrue(all("reference" in row and "raw_output" in row for row in report["cells"]))
        self.assertTrue(all(row["retrieval_diagnostics"] is not None for row in report["cells"]))

    def test_analysis_rejects_bad_wrapper_and_output_provenance(self) -> None:
        from experiments.analyze_qa_contrast_answers import analyze_receipt

        receipt, _audit, _references = _fixture_receipt()
        missing = json.loads(json.dumps(receipt))
        del missing["schema_version"]
        with self.assertRaisesRegex(ValueError, "envelope fields"):
            analyze_receipt(missing)

        wrong_runner = json.loads(json.dumps(receipt))
        wrong_runner["runner_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "envelope provenance"):
            analyze_receipt(wrong_runner)

        wrong_output = json.loads(json.dumps(receipt))
        wrong_output["result"]["outputs"][0]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "remote output identity"):
            analyze_receipt(wrong_output)

        huge_timing = json.loads(json.dumps(receipt))
        huge_timing["client_elapsed_seconds"] = 10**1000
        with self.assertRaisesRegex(ValueError, "timing"):
            analyze_receipt(huge_timing)

        huge_runtime = json.loads(json.dumps(receipt))
        huge_runtime["result"]["runtime"]["elapsed_seconds"] = 10**1000
        with self.assertRaises(ValueError):
            analyze_receipt(huge_runtime)

    def test_analysis_rejects_a_changed_validator_source(self) -> None:
        from experiments import qa_contrast_answer_receipt
        from experiments.analyze_qa_contrast_answers import analyze_receipt

        receipt, _audit, _references = _fixture_receipt()
        with tempfile.TemporaryDirectory() as directory:
            altered_source = Path(directory) / "altered_validator.py"
            altered_source.write_text("# changed validator source\n", encoding="utf-8")
            with patch.object(qa_contrast_answer_receipt, "__file__", str(altered_source)):
                with self.assertRaisesRegex(ValueError, "validator source differs"):
                    analyze_receipt(receipt)

    def test_unknown_nonliteral_outputs_are_not_labeled_unsupported(self) -> None:
        from experiments.analyze_qa_contrast_answers import analyze_receipt

        receipt, audit, references = _fixture_receipt()
        unknown_questions = [row for row in references["questions"] if row["is_abstention"]]
        output_by_prompt = {row["prompt_id"]: row for row in receipt["result"]["outputs"]}
        empty_question, phrase_question = unknown_questions
        empty_prompt = next(
            row["prompt_id"]
            for row in audit["cell_mapping"]
            if row["question_id"] == empty_question["question_id"]
        )
        phrase_prompt = next(
            row["prompt_id"]
            for row in audit["cell_mapping"]
            if row["question_id"] == phrase_question["question_id"]
        )
        output_by_prompt[empty_prompt]["raw_output"] = ""
        output_by_prompt[phrase_prompt]["raw_output"] = "I don't know"

        report = analyze_receipt(receipt)

        self.assertTrue(all("unsupported_answer" not in row for row in report["cells"]))
        self.assertTrue(
            all("unsupported_answers_on_unknown" not in row for row in report["groups"])
        )
        empty_cells = [row for row in report["cells"] if row["prompt_id"] == empty_prompt]
        phrase_cells = [row for row in report["cells"] if row["prompt_id"] == phrase_prompt]
        self.assertTrue(all(row["non_unknown_on_unknown"] and row["empty"] for row in empty_cells))
        self.assertTrue(
            all(row["non_unknown_on_unknown"] and not row["empty"] for row in phrase_cells)
        )
        self.assertEqual(
            sum(row["non_unknown_outputs_on_unknown"] for row in report["groups"]),
            sum(row["prompt_id"] in {empty_prompt, phrase_prompt} for row in audit["cell_mapping"]),
        )
        self.assertEqual(
            sum(row["empty_count"] for row in report["groups"]),
            sum(row["prompt_id"] == empty_prompt for row in audit["cell_mapping"]),
        )

    def test_cli_is_offline_and_writes_exclusive_mode_0600_report(self) -> None:
        from experiments import analyze_qa_contrast_answers as analyzer

        receipt, _audit, _references = _fixture_receipt()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output_path = root / "receipt.json", root / "analysis.json"
            input_path.write_text(json.dumps(receipt), encoding="utf-8")
            original_import = builtins.__import__

            def reject_model_import(name: str, *args: object, **kwargs: object) -> object:
                if name.split(".")[0] in {"modal", "torch", "transformers", "tokenizers"}:
                    raise AssertionError(f"offline analyzer imported {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=reject_model_import):
                status = analyzer.main([str(input_path), "--output", str(output_path)])
            self.assertEqual(status, 0)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["cell_count"], 72)

            printed = io.StringIO()
            with (
                patch("builtins.__import__", side_effect=reject_model_import),
                redirect_stdout(printed),
            ):
                status = analyzer.main([str(input_path)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(printed.getvalue())["unique_prompt_count"], 46)

            output_path.write_text("keep", encoding="utf-8")
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = analyzer.main([str(input_path), "--output", str(output_path)])
            self.assertEqual(status, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep")
            self.assertIn("FileExistsError", errors.getvalue())


def _fixture_receipt() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from experiments.analyze_qa_contrast_answers import _RUNNER_SHA256
    from tests.test_modal_qa_contrast_answer import _valid_result

    payload, references, audit = prepare_qa_contrast_requests.prepare_requests()
    result = _valid_result(payload, _RUNNER_SHA256)
    reference_by_id = {row["question_id"]: row for row in references["questions"]}
    question_by_prompt: dict[str, str] = {}
    for row in audit["cell_mapping"]:
        question_by_prompt.setdefault(row["prompt_id"], row["question_id"])
    for output in result["outputs"]:
        output["raw_output"] = reference_by_id[question_by_prompt[output["prompt_id"]]]["answer"]
    receipt = {
        "schema_version": 1,
        "payload_sha256": "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e",
        "runner_sha256": _RUNNER_SHA256,
        "client_elapsed_seconds": 3.5,
        "result": result,
    }
    return receipt, audit, references


if __name__ == "__main__":
    unittest.main()
