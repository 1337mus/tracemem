"""Focused contract tests for the quote-v2 local evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from training.evaluate_quote_sft import evaluate_report, main
from training.quote_data import render_quote_prompt


def _canonical_hash(value: dict[str, object]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "job_sha256"}
    canonical = json.dumps(
        unsigned, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _case(index: int) -> dict[str, object]:
    message = {
        "session_id": f"session-{index}",
        "message_id": f"message-{index}",
        "role": "user",
        "text": f"I prefer the blue mug for scenario {index}.",
        "recorded_at": "2026-01-02T10:00:00Z",
        "content_kind": "conversation",
    }
    return {
        "case_id": f"DEV-CASE-{index:03d}",
        "split": "dev",
        "scenario_group": f"DEV-SCENARIO-{index:03d}",
        "template_group": f"DEV-TEMPLATE-{index:03d}",
        "entity_ids": [f"DEV-ENTITY-{index:03d}"],
        "tags": ["preference"],
        "messages": [message],
        "expected": [
            {
                "claim": f"The user prefers the blue mug for scenario {index}.",
                "message_id": f"message-{index}",
                "quote": message["text"],
                "valid_from": None,
                "valid_to": None,
            }
        ],
    }


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    cases = [_case(index) for index in range(24)]
    fixture = {
        "version": 2,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": cases,
    }
    fixture_path = root / "dev.json"
    fixture_bytes = json.dumps(fixture, ensure_ascii=False).encode("utf-8")
    fixture_path.write_bytes(fixture_bytes)
    prompts = [{"case_id": case["case_id"], "prompt": render_quote_prompt(case)} for case in cases]
    job: dict[str, object] = {
        "schema_version": 1,
        "request": {"mode": "train", "profile": "quote-v2", "dev_prompts": prompts},
        "source_sha256": {"dev": hashlib.sha256(fixture_bytes).hexdigest()},
    }
    job["job_sha256"] = _canonical_hash(job)
    output_rows = [
        {
            "case_id": prompt["case_id"],
            "prompt_token_ids": [101, 102],
            "completion_token_ids": [201, 202],
            "response_text": '{"claims":[]}',
            "raw_text": '{"claims":[]}',
            "generated_tokens": 2,
            "max_new_tokens": 512,
            "reached_generation_limit": False,
        }
        for prompt in prompts
    ]
    receipt = {
        "mode": "train",
        "job_sha256": job["job_sha256"],
        "result": {
            "status": "completed",
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "generation": {
                "max_new_tokens": 512,
                "do_sample": False,
                "base_outputs": output_rows,
                "tuned_outputs": json.loads(json.dumps(output_rows)),
            },
        },
    }
    job_path, receipt_path = root / "job.json", root / "receipt.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return job_path, receipt_path, fixture_path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_receipt_response(
    receipt_path: Path,
    arm: str,
    index: int,
    response: str,
    *,
    raw_text: str | None = None,
    truncated: bool = False,
    generated_tokens: int = 2,
) -> None:
    receipt = _read_json(receipt_path)
    generation = receipt["result"]["generation"]
    row = generation[f"{arm}_outputs"][index]
    row["response_text"] = response
    row["raw_text"] = response if raw_text is None else raw_text
    row["generated_tokens"] = generated_tokens
    row["completion_token_ids"] = [201] * generated_tokens
    row["reached_generation_limit"] = truncated
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def _source_claim(index: int, text: str) -> dict[str, object]:
    return {
        "claim": text,
        "message_id": f"message-{index}",
        "quote": f"I prefer the blue mug for scenario {index}.",
        "valid_from": None,
        "valid_to": None,
    }


class EvaluateQuoteSftTests(unittest.TestCase):
    def test_rejects_a_job_whose_canonical_hash_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["request"]["profile"] = "other-profile"
            job_path.write_text(json.dumps(job), encoding="utf-8")

            with patch(
                "training.evaluate_quote_sft.render_quote_prompt",
                side_effect=AssertionError("must reject before rendering"),
            ):
                with self.assertRaisesRegex(ValueError, "job hash"):
                    evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

    def test_job_hash_must_bind_the_exact_dev_fixture_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["source_sha256"]["dev"] = "0" * 64
            job["job_sha256"] = _canonical_hash(job)
            job_path.write_text(json.dumps(job), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dev fixture hash"):
                evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

    def test_job_prompts_must_match_the_actual_rendered_dev_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            job = _read_json(job_path)
            job["request"]["dev_prompts"][0]["prompt"] += " altered"
            job["job_sha256"] = _canonical_hash(job)
            job_path.write_text(json.dumps(job), encoding="utf-8")
            receipt = _read_json(receipt_path)
            receipt["job_sha256"] = job["job_sha256"]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rendered dev case"):
                evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

    def test_completion_token_ids_must_match_the_reported_generated_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            receipt = _read_json(receipt_path)
            receipt["result"]["generation"]["base_outputs"][0]["completion_token_ids"] = [201]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "completion token count"):
                evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

    def test_base_and_tuned_prompt_token_ids_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            receipt = _read_json(receipt_path)
            receipt["result"]["generation"]["tuned_outputs"][0]["prompt_token_ids"] = [999]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt token IDs differ"):
                evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

    def test_receipt_must_pin_the_training_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            receipt = _read_json(receipt_path)
            receipt["result"]["revision"] = "unpinned"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "pinned training model"):
                evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

    def test_malformed_and_truncated_responses_are_preserved_without_abstention_credit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            _write_receipt_response(
                receipt_path,
                "base",
                0,
                '{"claims": [',
                raw_text="raw malformed bytes →",
                truncated=True,
                generated_tokens=512,
            )

            report = evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

            row = report["base"]["cases"]["DEV-CASE-000"]
            self.assertEqual(row["response_text"], '{"claims": [')
            self.assertEqual(row["raw_text"], "raw malformed bytes →")
            self.assertTrue(row["parse_failure"])
            self.assertTrue(row["truncated"])
            self.assertFalse(row["valid_empty_target_abstention"])
            self.assertEqual(report["base"]["parse_failures"], 1)
            self.assertEqual(report["base"]["truncated_case_ids"], ["DEV-CASE-000"])

    def test_paraphrase_can_be_cited_without_being_called_semantically_unsupported(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            paraphrase = _source_claim(0, "The user likes the blue mug in scenario zero.")
            _write_receipt_response(receipt_path, "tuned", 0, json.dumps({"claims": [paraphrase]}))

            report = evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

            arm = report["tuned"]
            self.assertEqual(arm["cases"]["DEV-CASE-000"]["citation_valid_claims"], 1)
            self.assertIsNone(arm["cases"]["DEV-CASE-000"]["parsed_claims"][0]["valid_from"])
            self.assertEqual(
                arm["cases"]["DEV-CASE-000"]["reference_string_scores"][
                    "matched_reference_strings"
                ],
                0,
            )
            self.assertIn("not automatically unsupported", report["comparison"]["semantic_support"])
            self.assertNotIn("unsupported_predictions", arm)

    def test_duplicate_predictions_receive_only_one_reference_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            expected = _source_claim(0, "The user prefers the blue mug for scenario 0.")
            _write_receipt_response(
                receipt_path,
                "base",
                0,
                json.dumps({"claims": [expected, expected]}),
            )

            report = evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

            scores = report["base"]["cases"]["DEV-CASE-000"]["reference_string_scores"]
            self.assertEqual(scores["matched_reference_strings"], 1)
            self.assertEqual(scores["predicted_claims"], 2)
            self.assertEqual(scores["precision"], 0.5)
            self.assertEqual(report["base"]["cases"]["DEV-CASE-000"]["citation_valid_claims"], 2)

    def test_date_not_present_in_quote_is_a_citation_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            dated = _source_claim(0, "The user prefers the blue mug for scenario 0.")
            dated["valid_from"] = "2026-01-02"
            _write_receipt_response(receipt_path, "tuned", 0, json.dumps({"claims": [dated]}))

            report = evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

            row = report["tuned"]["cases"]["DEV-CASE-000"]
            self.assertFalse(row["schema_failure"])
            self.assertEqual(row["citation_valid_claims"], 0)
            self.assertEqual(row["citation_invalid_claims"], 1)
            self.assertEqual(report["tuned"]["citation_validation_failure_cases"], 1)

    def test_schema_invalid_response_is_not_treated_as_a_valid_empty_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            _write_receipt_response(
                receipt_path, "base", 0, json.dumps({"claims": [{"claim": "incomplete"}]})
            )

            report = evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

            row = report["base"]["cases"]["DEV-CASE-000"]
            self.assertTrue(row["schema_failure"])
            self.assertEqual(row["raw_claim_count"], 1)
            self.assertEqual(row["schema_valid_claims"], 0)
            self.assertFalse(row["valid_empty_target_abstention"])
            self.assertTrue(row["missed_nonempty_target"])

    def test_valid_json_with_the_wrong_top_level_shape_is_a_schema_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            _write_receipt_response(receipt_path, "base", 0, '["valid JSON"]')

            report = evaluate_report(job_path, receipt_path, fixture_path=fixture_path)

            row = report["base"]["cases"]["DEV-CASE-000"]
            self.assertFalse(row["parse_failure"])
            self.assertTrue(row["schema_failure"])

    def test_cli_writes_a_fresh_private_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_path = _write_inputs(root)
            output_path = root / "report.json"
            args = [
                "--job",
                str(job_path),
                "--receipt",
                str(receipt_path),
                "--fixture",
                str(fixture_path),
                "--output",
                str(output_path),
            ]
            prior_umask = os.umask(0o377)
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(main(args), 0)
            finally:
                os.umask(prior_umask)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            before = output_path.read_bytes()
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(args), 1)
            self.assertEqual(output_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
