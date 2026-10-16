"""Offline contract tests for the quote-verifier scorer."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments import evaluate_quote_verifier
from experiments.quote_verifier_payload import build_preparation, payload_sha256


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _receipt(
    preparation: dict[str, Any],
    verdict_overrides: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    payload = preparation["payload"]
    overrides = verdict_overrides or {}
    outputs: list[dict[str, Any]] = []
    for item in payload["items"]:
        verdicts = overrides.get(
            item["case_id"],
            [
                {
                    "claim_index": index,
                    "support": "supported",
                    "eligible": True,
                    "reason": "The claim is supported and eligible.",
                }
                for index in range(item["candidate_count"])
            ],
        )
        outputs.append(
            {
                "case_id": item["case_id"],
                "item_id": item["item_id"],
                "raw_output": _json_text({"verdicts": verdicts}),
                "input_token_ids": [11, 12, 13],
                "output_token_ids": [41, 2],
                "input_tokens": 3,
                "output_tokens": 2,
                "last_token_id": 2,
                "ended_with_eos": True,
                "hit_max_new_tokens": False,
                "finish_reason": "eos",
            }
        )
    return {
        "schema_version": 1,
        "mode": "quote-verifier-semantic",
        "input_sha256": payload["input_sha256"],
        "generator_receipt_sha256": payload["generator_receipt_sha256"],
        "candidate_sha256": payload["candidate_sha256"],
        "payload_sha256": payload_sha256(payload),
        "items_sha256": payload["items_sha256"],
        "renderer_source_sha256": payload["renderer_source_sha256"],
        "payload_source_sha256": payload["payload_source_sha256"],
        "runner_sha256": payload["runner_source_sha256"],
        "model": copy.deepcopy(payload["model"]),
        "settings": copy.deepcopy(payload["settings"]),
        "generation_eos_token_ids": [2],
        "generation_pad_token_id": 0,
        "case_count": len(outputs),
        "item_count": len(outputs),
        "generated_outputs": outputs,
        "model_load_seconds": 1.25,
        "remote_function_elapsed_seconds": 8.5,
        "gpu_compute_seconds": 4.75,
        "peak_gpu_memory_bytes": 123456,
        "modal_function_call_id": "fc-offline-test",
        "client_call_wall_seconds_including_image_and_startup": 12.0,
    }


def _case(report: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(row for row in report["case_results"] if row["case_id"] == case_id)


class EvaluateQuoteVerifierTests(unittest.TestCase):
    preparation: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.preparation = build_preparation()

    def test_strict_receipt_parser_rejects_duplicate_json_keys(self) -> None:
        for raw in (b'{"payload_sha256":"a","payload_sha256":"b"}', b'{"value":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                evaluate_quote_verifier.load_receipt_bytes(raw)
        with self.assertRaisesRegex(ValueError, "5000000"):
            evaluate_quote_verifier.load_receipt_bytes(b" " * 5_000_001)

    def test_scores_all_original_indices_against_the_frozen_quote_v2_labels(self) -> None:
        preparation = copy.deepcopy(self.preparation)
        report = evaluate_quote_verifier.evaluate(_receipt(preparation), preparation)

        self.assertEqual(report["aggregate"]["semantic"]["total"], 39)
        self.assertEqual(report["aggregate"]["semantic"]["accepted_good"], 36)
        self.assertEqual(report["aggregate"]["semantic"]["accepted_bad"], 3)
        self.assertEqual(report["aggregate"]["semantic"]["rejected_good"], 0)
        self.assertEqual(report["aggregate"]["semantic"]["rejected_bad"], 0)
        self.assertEqual(report["aggregate"]["deterministic_only"]["total"], 39)
        self.assertEqual(report["aggregate"]["deterministic_only"]["accepted_good"], 36)
        self.assertEqual(report["aggregate"]["deterministic_only"]["accepted_bad"], 3)
        self.assertFalse(report["criterion"]["met"])
        self.assertTrue(report["criterion"]["all_ten_batches_valid_untruncated"])
        self.assertEqual(report["receipt_metrics"]["generated_output_tokens"], 20)
        self.assertEqual(report["receipt_metrics"]["gpu_compute_seconds"], 4.75)
        self.assertEqual(report["interpretation"]["human_reviewers"], 0)
        self.assertEqual(
            report["interpretation"]["selected_fact_retention"], "pending separate audit"
        )

        for audit_case in preparation["audit"]["cases"]:
            if audit_case["candidate_indices"]:
                result = _case(report, audit_case["case_id"])
                self.assertEqual(
                    [row["original_index"] for row in result["accepted"]],
                    audit_case["candidate_indices"],
                )
                self.assertEqual(
                    [row["candidate"] for row in result["accepted"]],
                    audit_case["candidates"],
                )
                self.assertEqual(result["source_item_id"], audit_case["source_item_id"])

    def test_rejects_missing_duplicate_output_ids_and_payload_hash_mismatch(self) -> None:
        mutations = ("missing", "duplicate", "hash")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                preparation = copy.deepcopy(self.preparation)
                receipt = _receipt(preparation)
                if mutation == "missing":
                    receipt["generated_outputs"].pop()
                elif mutation == "duplicate":
                    receipt["generated_outputs"][1]["item_id"] = receipt["generated_outputs"][0][
                        "item_id"
                    ]
                    receipt["generated_outputs"][1]["case_id"] = receipt["generated_outputs"][0][
                        "case_id"
                    ]
                else:
                    receipt["payload_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    evaluate_quote_verifier.evaluate(receipt, preparation)

    def test_rejects_forged_source_prompt_even_when_payload_and_receipt_are_rehashed(self) -> None:
        preparation = copy.deepcopy(self.preparation)
        payload = preparation["payload"]
        prompt = payload["items"][0]["prompt"]
        before, candidate_text = prompt.rsplit("\nCandidates: ", 1)
        prefix, message_text = before.rsplit("\nMessages: ", 1)
        messages = json.loads(message_text)
        messages[0]["text"] += " forged source text"
        payload["items"][0]["prompt"] = (
            f"{prefix}\nMessages: {_json_text(messages)}\nCandidates: {candidate_text}"
        )
        payload["items_sha256"] = payload_sha256(payload["items"])
        receipt = _receipt(preparation)
        receipt["items_sha256"] = payload["items_sha256"]
        receipt["payload_sha256"] = payload_sha256(payload)

        with self.assertRaisesRegex(ValueError, "fresh pinned preparation"):
            evaluate_quote_verifier.evaluate(receipt, preparation)

    def test_no_candidates_preserve_generator_gate_and_citation_rejection_details(self) -> None:
        preparation = copy.deepcopy(self.preparation)
        report = evaluate_quote_verifier.evaluate(_receipt(preparation), preparation)
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["no_candidate_cases"], 2)
        self.assertNotIn("no_target_cases", aggregate)
        self.assertEqual(aggregate["gate_status_counts"]["invalid-json"], 1)
        self.assertEqual(aggregate["gate_status_counts"]["no-valid-candidates"], 1)
        self.assertEqual(aggregate["citation_rejection_count"], 1)
        for audit in preparation["audit"]["cases"]:
            if audit["candidates"]:
                continue
            result = _case(report, audit["case_id"])
            self.assertEqual(result["outcome"], "no_candidates")
            for key in ("status", "generator_error", "truncated", "finish_reason"):
                self.assertEqual(result[key], audit[key])

    def test_rejects_negative_input_output_and_eos_token_ids(self) -> None:
        for token_kind in ("input", "output"):
            with self.subTest(token_kind=token_kind):
                preparation = copy.deepcopy(self.preparation)
                receipt = _receipt(preparation)
                row = receipt["generated_outputs"][0]
                row[f"{token_kind}_token_ids"][0] = -1
                report = evaluate_quote_verifier.evaluate(receipt, preparation)
                result = _case(report, row["case_id"])
                self.assertFalse(result["batch_valid"])
                self.assertIn("non-negative", result["batch_failure_reason"])

        preparation = copy.deepcopy(self.preparation)
        receipt = _receipt(preparation)
        receipt["generation_eos_token_ids"] = [-2]
        with self.assertRaisesRegex(ValueError, "EOS token IDs are invalid"):
            evaluate_quote_verifier.evaluate(receipt, preparation)

    def test_invalid_verdict_and_truncated_output_fail_the_whole_batch(self) -> None:
        reasons = {
            "malformed-verdict": "invalid verdict response",
            "fenced-verdict": "invalid verdict response",
            "max-output": "output reached max_new_tokens",
            "token-mismatch": "input token IDs must be non-negative",
            "non-eos": "output did not end with EOS",
        }
        for failure in (
            "malformed-verdict",
            "fenced-verdict",
            "max-output",
            "token-mismatch",
            "non-eos",
        ):
            with self.subTest(failure=failure):
                preparation = copy.deepcopy(self.preparation)
                receipt = _receipt(preparation)
                row = receipt["generated_outputs"][0]
                if failure == "malformed-verdict":
                    row["raw_output"] = '{"verdicts":['
                elif failure == "fenced-verdict":
                    row["raw_output"] = f"```json\n{row['raw_output']}\n```"
                else:
                    if failure == "max-output":
                        row["output_token_ids"] = list(range(1023)) + [2]
                        row["output_tokens"] = 1024
                        row["hit_max_new_tokens"] = True
                    elif failure == "token-mismatch":
                        row["input_tokens"] = 4
                    else:
                        row["output_token_ids"][-1] = 43
                        row["last_token_id"] = 43
                        row["ended_with_eos"] = False
                        row["finish_reason"] = "unknown"
                report = evaluate_quote_verifier.evaluate(receipt, preparation)
                result = _case(report, row["case_id"])
                self.assertFalse(result["batch_valid"])
                self.assertIn(reasons[failure], result["batch_failure_reason"])
                self.assertEqual(result["accepted"], [])
                self.assertEqual(len(result["rejected"]), 3)
                self.assertEqual(report["aggregate"]["semantic"]["valid_batches"], 9)
                self.assertEqual(report["aggregate"]["semantic"]["failed_batches"], 1)

    def test_uncertain_and_ineligible_verdicts_are_rejected_separately_from_no_candidate_cases(
        self,
    ) -> None:
        preparation = copy.deepcopy(self.preparation)
        overrides = {
            "V4D-01": [
                {
                    "claim_index": 0,
                    "support": "uncertain",
                    "eligible": True,
                    "reason": "Support is unclear.",
                },
                {
                    "claim_index": 1,
                    "support": "supported",
                    "eligible": False,
                    "reason": "Not durable.",
                },
                {
                    "claim_index": 2,
                    "support": "unsupported",
                    "eligible": False,
                    "reason": "Not supported.",
                },
            ]
        }
        report = evaluate_quote_verifier.evaluate(_receipt(preparation, overrides), preparation)
        result = _case(report, "V4D-01")

        self.assertEqual(result["outcome"], "valid_abstention")
        self.assertEqual(result["accepted"], [])
        self.assertEqual([row["original_index"] for row in result["rejected"]], [0, 1, 2])
        self.assertEqual(report["aggregate"]["no_candidate_cases"], 2)
        self.assertEqual(report["aggregate"]["valid_abstention_cases"], 1)
        breakdown = report["aggregate"]["semantic"]["support_eligibility"]
        self.assertEqual(breakdown["uncertain"]["eligible"], 1)
        self.assertEqual(breakdown["supported"]["ineligible"], 1)

    def test_all_rejected_receipt_fails_the_predeclared_good_retention_criterion(self) -> None:
        preparation = copy.deepcopy(self.preparation)
        overrides = {
            item["case_id"]: [
                {
                    "claim_index": index,
                    "support": "unsupported",
                    "eligible": True,
                    "reason": "Not supported.",
                }
                for index in range(item["candidate_count"])
            ]
            for item in preparation["payload"]["items"]
        }
        report = evaluate_quote_verifier.evaluate(_receipt(preparation, overrides), preparation)
        semantic = report["aggregate"]["semantic"]

        self.assertEqual(semantic["accepted_good"], 0)
        self.assertEqual(semantic["rejected_good"], 36)
        self.assertEqual(semantic["rejected_bad"], 3)
        self.assertTrue(report["criterion"]["all_three_bad_rejected"])
        self.assertFalse(report["criterion"]["at_least_35_good_retained"])
        self.assertEqual(semantic["valid_batches"], 10)
        self.assertFalse(report["criterion"]["met"])

    def test_cli_refuses_to_overwrite_an_existing_report(self) -> None:
        preparation = copy.deepcopy(self.preparation)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path, output_path = root / "receipt.json", root / "report.json"
            receipt_path.write_text(_json_text(_receipt(preparation)), encoding="utf-8")
            arguments = [
                "evaluate_quote_verifier",
                "--receipt",
                str(receipt_path),
                "--output",
                str(output_path),
            ]
            with patch("sys.argv", arguments):
                self.assertEqual(evaluate_quote_verifier.main(), 0)
            original = output_path.read_bytes()
            with (
                redirect_stderr(StringIO()),
                patch("sys.argv", arguments),
                self.assertRaises(SystemExit) as raised,
            ):
                evaluate_quote_verifier.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
