"""Tests for per-model accounting of blind scenario reviews."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

from test_scenario_reviews import _candidate, _fixture, _json_claims, _receipt

from experiments import summarize_scenario_reviews
from experiments.prepare_scenario_reviews import prepare_reviews
from tracemem.annotations import _canonical_json


def _response_document(
    bundle: dict[str, Any], reviewer_id: str, reviewer_kind: str, label: str
) -> dict[str, Any]:
    return {
        "version": 1,
        "bundle_sha256": bundle["bundle_sha256"],
        "responses": [
            {
                "task_id": task["task_id"],
                "label": label,
                "rationale": "The cited evidence supports this claim.",
                "reviewer_id": reviewer_id,
                "reviewer_kind": reviewer_kind,
                "timestamp": "2026-10-14T19:00:00Z",
            }
            for task in bundle["tasks"]
        ],
    }


class ScenarioReviewReportTests(unittest.TestCase):
    def test_report_binds_canonical_inputs_and_regenerated_provenance(self) -> None:
        cases, _, dataset_sha256 = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "hash-bound candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, provenance = prepare_reviews([receipt], seed=29)
        response = _response_document(bundle, "reviewer-a", "human", "supported")

        report = summarize_scenario_reviews.summarize_reviews([receipt], [response], seed=29)

        self.assertEqual(report["dataset_sha256"], dataset_sha256)
        self.assertEqual(report["bundle_sha256"], bundle["bundle_sha256"])
        self.assertEqual(
            report["bundle_canonical_sha256"],
            hashlib.sha256(_canonical_json(bundle).encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            report["receipts"][0]["receipt_sha256"],
            hashlib.sha256(receipt).hexdigest(),
        )
        self.assertEqual(
            report["receipts"][0]["canonical_sha256"],
            hashlib.sha256(_canonical_json(json.loads(receipt)).encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            report["responses"][0]["canonical_sha256"],
            hashlib.sha256(_canonical_json(response).encode("utf-8")).hexdigest(),
        )
        self.assertEqual(report["provenance_sha256"], provenance["provenance_sha256"])

    def test_repeated_claim_origins_count_separately_but_share_one_review_task(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "one shared candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim, claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=31)
        response = _response_document(bundle, "reviewer-a", "agent", "supported")

        report = summarize_scenario_reviews.summarize_reviews([receipt], [response], seed=31)

        split = case["split"]
        model_split = report["by_model"]["qwen3-4b"][split]
        reviewer = model_split["reviewers"][0]
        self.assertEqual(model_split["schema_valid_candidate_count"], 2)
        self.assertEqual(model_split["unique_review_task_count"], 1)
        self.assertEqual(reviewer["supported_origin_count"], 2)
        self.assertEqual(reviewer["all_reviewed_origin_count"], 2)
        self.assertEqual(reviewer["supported_rate_all_reviewed_origins"], 1.0)

    def test_shared_task_keeps_model_origins_and_reviewer_kinds_separate(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "shared between models")
        receipts = [
            _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim, claim)}),
            _receipt("phi-3.5-mini", {case["case_id"]: _json_claims(claim)}),
        ]
        bundle, _, _ = prepare_reviews(receipts, seed=7)
        responses = [
            _response_document(bundle, "human-a", "human", "supported"),
            _response_document(bundle, "agent-a", "agent", "uncertain"),
        ]

        report = summarize_scenario_reviews.summarize_reviews(receipts, responses, seed=7)

        split = case["split"]
        qwen = report["by_model"]["qwen3-4b"][split]
        phi = report["by_model"]["phi-3.5-mini"][split]
        self.assertEqual(qwen["unique_review_task_count"], 1)
        self.assertEqual(phi["unique_review_task_count"], 1)
        self.assertEqual(qwen["schema_valid_candidate_count"], 2)
        self.assertEqual(phi["schema_valid_candidate_count"], 1)
        self.assertEqual(
            [(row["reviewer_id"], row["reviewer_kind"]) for row in qwen["reviewers"]],
            [("human-a", "human"), ("agent-a", "agent")],
        )
        self.assertEqual(qwen["reviewers"][0]["supported_origin_count"], 2)
        self.assertEqual(qwen["reviewers"][1]["uncertain_origin_count"], 2)
        self.assertIsNone(qwen["reviewers"][1]["supported_rate_decisive_origins"])
        other_split = "test" if split == "dev" else "dev"
        self.assertIsNone(
            report["by_model"]["phi-3.5-mini"][other_split]["reviewers"][0][
                "supported_rate_all_reviewed_origins"
            ]
        )

    def test_excluded_outputs_are_separate_from_valid_empty_outputs(self) -> None:
        cases, _, _ = _fixture()
        split_cases = [case for case in cases if case["expected"]]
        parse_case, schema_case, candidate_case = split_cases[:3]
        claim = _candidate(candidate_case["case_id"], "valid candidate")
        receipt = _receipt(
            "qwen3-4b",
            {
                parse_case["case_id"]: "not JSON",
                schema_case["case_id"]: '{"claims":[{"unexpected":"shape"}]}',
                candidate_case["case_id"]: _json_claims(claim),
            },
        )

        report = summarize_scenario_reviews.summarize_reviews([receipt], [], seed=3)

        split = parse_case["split"]
        cell = report["by_model"]["qwen3-4b"][split]
        self.assertEqual(cell["raw_output_count"], 24)
        self.assertEqual(cell["parse_excluded_output_count"], 1)
        self.assertEqual(cell["schema_excluded_output_count"], 1)
        self.assertEqual(cell["schema_valid_empty_output_count"], 21)
        self.assertEqual(cell["schema_valid_candidate_count"], 1)

    def test_response_hash_coverage_and_reviewer_uniqueness_are_validated(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "review validation")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=13)
        response = _response_document(bundle, "reviewer-a", "human", "supported")

        bad_hash = copy.deepcopy(response)
        bad_hash["bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle hash"):
            summarize_scenario_reviews.summarize_reviews([receipt], [bad_hash], seed=13)

        incomplete = copy.deepcopy(response)
        incomplete["responses"].clear()
        with self.assertRaisesRegex(ValueError, "missing task ID"):
            summarize_scenario_reviews.summarize_reviews([receipt], [incomplete], seed=13)

        with self.assertRaisesRegex(ValueError, "duplicate reviewer"):
            summarize_scenario_reviews.summarize_reviews([receipt], [response, response], seed=13)

    def test_empty_task_bundle_is_reported_without_inventing_a_reviewer(self) -> None:
        receipt = _receipt("qwen3-4b")

        report = summarize_scenario_reviews.summarize_reviews([receipt], [], seed=17)

        self.assertEqual(report["responses"], [])
        for split in ("dev", "test"):
            cell = report["by_model"]["qwen3-4b"][split]
            self.assertEqual(cell["raw_output_count"], 24)
            self.assertEqual(cell["schema_valid_empty_output_count"], 24)
            self.assertEqual(cell["unique_review_task_count"], 0)
            self.assertEqual(cell["reviewers"], [])

    def test_cli_preflights_inputs_and_creates_report_exclusively(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "cli candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=23)
        response = _response_document(bundle, "reviewer-a", "agent", "supported")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            response_path = root / "response.json"
            output_path = root / "report.json"
            receipt_path.write_bytes(receipt)
            response_bytes = json.dumps(response).encode("utf-8")
            response_path.write_bytes(response_bytes)
            args = [
                "--receipts",
                str(receipt_path),
                "--responses",
                str(response_path),
                "--seed",
                "23",
                "--output",
                str(output_path),
            ]

            self.assertEqual(summarize_scenario_reviews.main(args), 0)
            self.assertEqual(receipt_path.read_bytes(), receipt)
            self.assertEqual(response_path.read_bytes(), response_bytes)
            written = output_path.read_bytes()
            self.assertEqual(json.loads(written)["bundle_sha256"], bundle["bundle_sha256"])

            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(summarize_scenario_reviews.main(args), 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())
            self.assertEqual(output_path.read_bytes(), written)

            alias_args = [*args[:-1], str(receipt_path)]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(summarize_scenario_reviews.main(alias_args), 2)
            self.assertIn("aliases an input", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
