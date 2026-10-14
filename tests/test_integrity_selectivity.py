"""Fictional-only tests for retrospective integrity selectivity."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_scenario_reviews import _candidate, _fixture, _json_claims, _receipt

from experiments import evaluate_integrity_selectivity as integrity_module
from experiments.evaluate_integrity_selectivity import evaluate_integrity_selectivity
from experiments.prepare_scenario_reviews import prepare_reviews


def _response_document(
    bundle: dict[str, Any], reviewer_id: str, reviewer_kind: str, label: str | Mapping[str, str]
) -> dict[str, Any]:
    return {
        "version": 1,
        "bundle_sha256": bundle["bundle_sha256"],
        "responses": [
            {
                "task_id": task["task_id"],
                "label": label.get(task["task_id"], "unsupported")
                if isinstance(label, Mapping)
                else label,
                "rationale": "Fictional test rationale only.",
                "reviewer_id": reviewer_id,
                "reviewer_kind": reviewer_kind,
                "timestamp": "2026-10-14T19:00:00Z",
            }
            for task in bundle["tasks"]
        ],
    }


class IntegritySelectivityTests(unittest.TestCase):
    def test_quote_repair_can_pass_anchored_filter_without_changing_unsupported_label(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional private claim text")
        shifted = copy.deepcopy(claim)
        shifted["evidence"][0]["char_start"] += 1
        shifted["evidence"][0]["char_end"] += 1
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(shifted)})
        bundle, _, _ = prepare_reviews([receipt], seed=71)
        response = _response_document(bundle, "reviewer-a", "human", "unsupported")

        report = evaluate_integrity_selectivity([receipt], [response], seed=71)

        cell = report["by_model"]["qwen3-4b"][case["split"]]
        raw = cell["filters"]["raw_integrity"]
        anchored = cell["filters"]["anchored_integrity"]
        self.assertEqual((raw["passed_origin_count"], raw["rejected_origin_count"]), (0, 1))
        self.assertEqual(
            (anchored["passed_origin_count"], anchored["rejected_origin_count"]), (1, 0)
        )
        self.assertEqual(raw["reviewers"][0]["rejected_unsupported_origin_count"], 1)
        self.assertIsNone(raw["reviewers"][0]["supported_rate_all_passed_origins"])
        self.assertIsNone(raw["reviewers"][0]["supported_rate_decisive_passed_origins"])
        self.assertEqual(anchored["reviewers"][0]["passed_unsupported_origin_count"], 1)
        self.assertEqual(anchored["reviewers"][0]["supported_rate_all_passed_origins"], 0.0)
        self.assertEqual(anchored["reviewers"][0]["supported_rate_decisive_passed_origins"], 0.0)
        self.assertIsNone(anchored["reviewers"][0]["supported_retention_rate"])
        audit = report["origin_audit"][0]
        self.assertEqual(audit["model_key"], "qwen3-4b")
        self.assertEqual(audit["case_id"], case["case_id"])
        self.assertEqual(audit["claim_index"], 0)
        self.assertFalse(audit["raw_integrity_passes"])
        self.assertTrue(audit["anchored_integrity_passes"])
        self.assertEqual(audit["raw_error_codes"], ["quote_mismatch"])
        self.assertEqual(audit["anchored_error_codes"], [])
        self.assertNotIn("fictional private claim text", json.dumps(report))

    def test_raw_integrity_pass_does_not_upgrade_an_unsupported_review_label(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional unsupported candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=72)
        response = _response_document(bundle, "reviewer-a", "human", "unsupported")

        report = evaluate_integrity_selectivity([receipt], [response], seed=72)

        raw = report["by_model"]["qwen3-4b"][case["split"]]["filters"]["raw_integrity"]
        reviewer = raw["reviewers"][0]
        self.assertEqual(raw["passed_origin_count"], 1)
        self.assertEqual(reviewer["passed_unsupported_origin_count"], 1)
        self.assertEqual(reviewer["supported_rate_all_passed_origins"], 0.0)
        self.assertEqual(reviewer["supported_rate_decisive_passed_origins"], 0.0)
        self.assertIsNone(reviewer["supported_retention_rate"])

    def test_supported_unsupported_and_uncertain_denominators_remain_separate(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claims = [
            _candidate(case["case_id"], "fictional supported-labeled candidate"),
            _candidate(case["case_id"], "fictional unsupported-labeled candidate"),
            _candidate(case["case_id"], "fictional uncertain-labeled candidate"),
        ]
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(*claims)})
        bundle, _, _ = prepare_reviews([receipt], seed=73)
        label_by_object = {
            "fictional supported-labeled candidate": "supported",
            "fictional unsupported-labeled candidate": "unsupported",
            "fictional uncertain-labeled candidate": "uncertain",
        }
        labels = {
            task["task_id"]: label_by_object[task["candidate"]["object"]]
            for task in bundle["tasks"]
        }
        response = _response_document(bundle, "reviewer-a", "human", labels)

        report = evaluate_integrity_selectivity([receipt], [response], seed=73)

        raw = report["by_model"]["qwen3-4b"][case["split"]]["filters"]["raw_integrity"]
        reviewer = raw["reviewers"][0]
        self.assertEqual(raw["passed_origin_count"], 3)
        self.assertEqual(reviewer["passed_supported_origin_count"], 1)
        self.assertEqual(reviewer["passed_unsupported_origin_count"], 1)
        self.assertEqual(reviewer["passed_uncertain_origin_count"], 1)
        self.assertEqual(reviewer["supported_rate_all_passed_origins"], 1 / 3)
        self.assertEqual(reviewer["supported_rate_decisive_passed_origins"], 0.5)
        self.assertEqual(reviewer["supported_retention_rate"], 1.0)

    def test_unrepairable_source_role_conflict_fails_both_integrity_filters(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional role-conflict candidate")
        claim["speaker_role"] = "assistant"
        claim["attribution"] = "assistant_statement"
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=74)
        response = _response_document(bundle, "reviewer-a", "human", "uncertain")

        report = evaluate_integrity_selectivity([receipt], [response], seed=74)

        cell = report["by_model"]["qwen3-4b"][case["split"]]
        for filter_name in ("raw_integrity", "anchored_integrity"):
            result = cell["filters"][filter_name]
            self.assertEqual(
                (result["passed_origin_count"], result["rejected_origin_count"]), (0, 1)
            )
            self.assertEqual(result["reviewers"][0]["rejected_uncertain_origin_count"], 1)
        audit = report["origin_audit"][0]
        self.assertEqual(audit["raw_error_codes"], ["source_role_mismatch"])
        self.assertEqual(audit["anchored_error_codes"], ["source_role_mismatch"])
        self.assertEqual(
            cell["verifier_error_code_counts"],
            {
                "raw": {"source_role_mismatch": 1},
                "anchored": {"source_role_mismatch": 1},
            },
        )

    def test_duplicate_origins_keep_distinct_raw_results_when_they_share_a_task(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        exact = _candidate(case["case_id"], "fictional duplicate candidate")
        shifted = copy.deepcopy(exact)
        shifted["evidence"][0]["char_start"] += 1
        shifted["evidence"][0]["char_end"] += 1
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(exact, shifted)})
        bundle, _, _ = prepare_reviews([receipt], seed=75)
        self.assertEqual(len(bundle["tasks"]), 1)
        response = _response_document(bundle, "reviewer-a", "human", "unsupported")

        report = evaluate_integrity_selectivity([receipt], [response], seed=75)

        cell = report["by_model"]["qwen3-4b"][case["split"]]
        self.assertEqual(cell["schema_valid_origin_count"], 2)
        self.assertEqual(cell["filters"]["raw_integrity"]["passed_origin_count"], 1)
        self.assertEqual(cell["filters"]["raw_integrity"]["rejected_origin_count"], 1)
        self.assertEqual(cell["filters"]["anchored_integrity"]["passed_origin_count"], 2)
        audit = report["origin_audit"]
        self.assertEqual([row["claim_index"] for row in audit], [0, 1])
        self.assertEqual(len({row["task_id"] for row in audit}), 1)

    def test_missing_or_extra_origin_mappings_are_rejected(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional mapping candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, answer_key, provenance = prepare_reviews([receipt], seed=751)

        missing = copy.deepcopy(provenance)
        missing["task_provenance"][0]["origins"].clear()
        extra = copy.deepcopy(provenance)
        unknown_origin = copy.deepcopy(extra["task_provenance"][0]["origins"][0])
        unknown_origin["claim_index"] = 99
        extra["task_provenance"][0]["origins"].append(unknown_origin)

        for name, broken_provenance in (("missing", missing), ("extra", extra)):
            with (
                self.subTest(name=name),
                patch.object(
                    integrity_module,
                    "prepare_reviews",
                    return_value=(bundle, answer_key, broken_provenance),
                ),
                self.assertRaisesRegex(ValueError, "origin"),
            ):
                evaluate_integrity_selectivity([receipt], [], seed=751)

    def test_shared_tasks_stay_separate_by_model_split_and_reviewer(self) -> None:
        cases, _, _ = _fixture()
        dev_case = next(case for case in cases if case["split"] == "dev" and case["expected"])
        test_case = next(case for case in cases if case["split"] == "test" and case["expected"])
        shared = _candidate(dev_case["case_id"], "fictional shared task")
        qwen_test = _candidate(test_case["case_id"], "fictional qwen test task")
        receipts = [
            _receipt(
                "qwen3-4b",
                {
                    dev_case["case_id"]: _json_claims(shared),
                    test_case["case_id"]: _json_claims(qwen_test),
                },
            ),
            _receipt("phi-3.5-mini", {dev_case["case_id"]: _json_claims(shared)}),
        ]
        bundle, _, _ = prepare_reviews(receipts, seed=76)
        responses = [
            _response_document(bundle, "human-a", "human", "supported"),
            _response_document(bundle, "agent-a", "agent", "uncertain"),
        ]

        report = evaluate_integrity_selectivity(receipts, responses, seed=76)

        qwen_dev = report["by_model"]["qwen3-4b"]["dev"]
        qwen_test = report["by_model"]["qwen3-4b"]["test"]
        phi_dev = report["by_model"]["phi-3.5-mini"]["dev"]
        phi_test = report["by_model"]["phi-3.5-mini"]["test"]
        self.assertEqual(qwen_dev["schema_valid_origin_count"], 1)
        self.assertEqual(qwen_test["schema_valid_origin_count"], 1)
        self.assertEqual(phi_dev["schema_valid_origin_count"], 1)
        self.assertEqual(phi_test["schema_valid_origin_count"], 0)
        shared_rows = [
            row for row in report["origin_audit"] if row["case_id"] == dev_case["case_id"]
        ]
        self.assertEqual(len({row["task_id"] for row in shared_rows}), 1)
        reviewer_rows = qwen_dev["filters"]["raw_integrity"]["reviewers"]
        self.assertEqual(
            [(row["reviewer_id"], row["reviewer_kind"]) for row in reviewer_rows],
            [("agent-a", "agent"), ("human-a", "human")],
        )
        self.assertEqual(reviewer_rows[0]["passed_uncertain_origin_count"], 1)
        self.assertEqual(reviewer_rows[1]["passed_supported_origin_count"], 1)

    def test_exclusions_and_valid_empty_outputs_remain_in_accounting(self) -> None:
        cases, _, _ = _fixture()
        selected = [case for case in cases if case["expected"]][:2]
        parse_case, schema_case = selected
        receipt = _receipt(
            "qwen3-4b",
            {
                parse_case["case_id"]: "not JSON",
                schema_case["case_id"]: '{"claims":[{"unexpected":"shape"}]}',
            },
        )

        report = evaluate_integrity_selectivity([receipt], [], seed=77)

        cell = report["accounting"]["by_model"]["qwen3-4b"][parse_case["split"]]
        self.assertEqual(cell["raw_output_count"], 24)
        self.assertEqual(cell["parse_excluded_output_count"], 1)
        self.assertEqual(cell["schema_excluded_output_count"], 1)
        self.assertEqual(cell["schema_valid_empty_output_count"], 22)
        self.assertEqual(len(report["accounting"]["exclusions"]), 2)
        for split in ("dev", "test"):
            model_cell = report["by_model"]["qwen3-4b"][split]
            self.assertEqual(model_cell["schema_valid_origin_count"], 0)
            for filter_row in model_cell["filters"].values():
                self.assertEqual(filter_row["passed_origin_count"], 0)
                self.assertEqual(filter_row["rejected_origin_count"], 0)
                self.assertIsNone(filter_row["coverage"])
                self.assertEqual(filter_row["reviewers"], [])
        self.assertEqual(report["origin_audit"], [])

    def test_no_reviewers_keeps_integrity_counts_and_input_hashes(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional private hash-bound candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        canonical_receipt = json.loads(receipt)
        _, _, expected_provenance = prepare_reviews([receipt], seed=78)

        report = evaluate_integrity_selectivity([receipt], [], seed=78)

        self.assertEqual(
            report["inputs"]["receipts"][0]["sha256"], hashlib.sha256(receipt).hexdigest()
        )
        from tracemem.annotations import _canonical_json

        self.assertEqual(
            report["inputs"]["receipts"][0]["canonical_sha256"],
            hashlib.sha256(_canonical_json(canonical_receipt).encode("utf-8")).hexdigest(),
        )
        self.assertEqual(report["inputs"]["responses"], [])
        self.assertEqual(report["provenance_sha256"], expected_provenance["provenance_sha256"])
        cell = report["by_model"]["qwen3-4b"][case["split"]]
        self.assertEqual(cell["schema_valid_origin_count"], 1)
        for filter_row in cell["filters"].values():
            self.assertEqual(filter_row["passed_origin_count"], 1)
            self.assertEqual(filter_row["reviewers"], [])
        self.assertNotIn("fictional private hash-bound candidate", json.dumps(report))

    def test_response_hashes_are_recorded_without_mutating_or_echoing_documents(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional private report claim")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=79)
        response = _response_document(bundle, "reviewer-a", "human", "supported")
        response["responses"][0]["rationale"] = "fictional private reviewer rationale"
        response_before = copy.deepcopy(response)
        from tracemem.annotations import _canonical_json

        expected_response_hash = hashlib.sha256(
            _canonical_json(response).encode("utf-8")
        ).hexdigest()

        report = evaluate_integrity_selectivity([receipt], [response], seed=79)

        self.assertEqual(response, response_before)
        self.assertEqual(
            report["inputs"]["responses"][0]["canonical_sha256"], expected_response_hash
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("fictional private report claim", serialized)
        self.assertNotIn("fictional private reviewer rationale", serialized)

    def test_late_invalid_receipts_and_response_sets_fail_closed(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional malformed-input candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=80)
        response = _response_document(bundle, "reviewer-a", "human", "unsupported")
        incomplete = copy.deepcopy(response)
        incomplete["responses"].pop()
        wrong_bundle = copy.deepcopy(response)
        wrong_bundle["bundle_sha256"] = "0" * 64

        checks = (
            (
                "duplicate receipts",
                [receipt, receipt],
                [response],
                "duplicate receipt SHA-256 input",
            ),
            (
                "duplicate model",
                [
                    receipt,
                    json.dumps({**json.loads(receipt), "gpu_compute_seconds": 13.5}).encode(
                        "utf-8"
                    ),
                ],
                [response],
                "more than one receipt for model qwen3-4b",
            ),
            (
                "late malformed receipt",
                [receipt, b"not JSON"],
                [response],
                "receipt is not valid UTF-8 JSON",
            ),
            (
                "duplicate reviewer",
                [receipt],
                [response, copy.deepcopy(response)],
                "duplicate reviewer document: reviewer-a",
            ),
            (
                "incomplete response",
                [receipt],
                [incomplete],
                "missing task ID(s):",
            ),
            (
                "wrong bundle",
                [receipt],
                [wrong_bundle],
                "response bundle hash does not match blind bundle",
            ),
            (
                "late malformed response",
                [receipt],
                [response, {"version": 1}],
                "response document has missing or unexpected fields",
            ),
        )
        for name, receipt_inputs, response_inputs, expected_message in checks:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, re.escape(expected_message)),
            ):
                evaluate_integrity_selectivity(receipt_inputs, response_inputs, seed=80)

    def test_cli_creates_exclusive_report_and_preserves_input_files(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "fictional cli candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=81)
        response = _response_document(bundle, "reviewer-a", "agent", "uncertain")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            response_path = root / "response.json"
            output_path = root / "report.json"
            receipt_path.write_bytes(receipt)
            response_bytes = json.dumps(response).encode("utf-8")
            response_path.write_bytes(response_bytes)
            receipt_metadata, response_metadata = receipt_path.stat(), response_path.stat()
            arguments = [
                "--receipts",
                str(receipt_path),
                "--responses",
                str(response_path),
                "--seed",
                "81",
                "--output",
                str(output_path),
            ]

            self.assertEqual(integrity_module.main(arguments), 0)
            written = output_path.read_bytes()
            self.assertEqual(json.loads(written)["schema_version"], 1)
            self.assertEqual(receipt_path.read_bytes(), receipt)
            self.assertEqual(response_path.read_bytes(), response_bytes)
            self.assertEqual(receipt_path.stat().st_mode, receipt_metadata.st_mode)
            self.assertEqual(response_path.stat().st_mode, response_metadata.st_mode)
            self.assertEqual(receipt_path.stat().st_mtime_ns, receipt_metadata.st_mtime_ns)
            self.assertEqual(response_path.stat().st_mtime_ns, response_metadata.st_mtime_ns)

            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(integrity_module.main(arguments), 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())
            self.assertEqual(output_path.read_bytes(), written)

            alias_arguments = [*arguments[:-1], str(receipt_path)]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(integrity_module.main(alias_arguments), 2)
            self.assertIn("aliases an input", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
