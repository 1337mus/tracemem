"""Fictional-only tests for retrospective scenario-tag breakdowns."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_scenario_reviews import _candidate, _fixture, _json_claims, _receipt

from experiments import summarize_scenario_phenomena as phenomena_module
from experiments.prepare_scenario_reviews import prepare_reviews
from experiments.summarize_scenario_phenomena import main, summarize_phenomena


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


class ScenarioPhenomenaTests(unittest.TestCase):
    def test_summarizes_a_validated_origin_in_its_source_tag_cell(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        tag = case["tags"][0]
        claim = _candidate(case["case_id"], "PRIVATE-CANDIDATE-SENTINEL")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=101)
        response = _response_document(bundle, "reviewer-a", "human", "supported")

        report = summarize_phenomena([receipt], [response], seed=101)

        self.assertEqual(report["schema_version"], 1)
        cell = report["by_model"]["qwen3-4b"][case["split"]]["by_tag"][tag]
        self.assertEqual(cell["schema_valid_origin_count"], 1)
        self.assertEqual(cell["reviewers"][0]["supported_origin_count"], 1)
        self.assertEqual(
            cell["source_case_count"],
            sum(tag in set(row["tags"]) for row in cases if row["split"] == case["split"]),
        )
        self.assertNotIn("PRIVATE-CANDIDATE-SENTINEL", json.dumps(report))

    def test_counts_exclusions_and_empty_outputs_for_every_overlapping_source_tag(self) -> None:
        cases, _, _ = _fixture()
        selected = [case for case in cases if case["split"] == "dev" and case["expected"]][:4]
        parse_case, schema_case, empty_case, origin_case = selected
        claim = _candidate(origin_case["case_id"], "PRIVATE-OVERLAP-SENTINEL")
        receipt = _receipt(
            "qwen3-4b",
            {
                parse_case["case_id"]: "not JSON",
                schema_case["case_id"]: '{"claims":[{"unexpected":"shape"}]}',
                origin_case["case_id"]: _json_claims(claim),
            },
        )
        bundle, _, _ = prepare_reviews([receipt], seed=102)
        response = _response_document(bundle, "reviewer-a", "human", "supported")

        report = summarize_phenomena([receipt], [response], seed=102)

        by_tag = report["by_model"]["qwen3-4b"]["dev"]["by_tag"]
        self.assertEqual(
            set(by_tag), {tag for case in cases if case["split"] == "dev" for tag in case["tags"]}
        )
        self.assertEqual(list(by_tag), sorted(by_tag))
        self.assertEqual(by_tag[parse_case["tags"][0]]["parse_excluded_output_count"], 1)
        self.assertEqual(by_tag[schema_case["tags"][0]]["schema_excluded_output_count"], 1)
        self.assertEqual(by_tag[empty_case["tags"][0]]["schema_valid_empty_output_count"], 1)
        self.assertEqual(by_tag[empty_case["tags"][0]]["schema_valid_origin_count"], 0)
        for tag in origin_case["tags"]:
            self.assertEqual(by_tag[tag]["schema_valid_nonempty_output_count"], 1)
            self.assertEqual(by_tag[tag]["schema_valid_origin_count"], 1)
        self.assertGreater(len(origin_case["tags"]), 1)
        self.assertEqual(by_tag[origin_case["tags"][0]]["source_case_count"], 1)
        self.assertNotIn("PRIVATE-OVERLAP-SENTINEL", json.dumps(report))

    def test_keeps_duplicate_origins_models_splits_and_reviewer_kinds_separate(self) -> None:
        cases, _, _ = _fixture()
        dev_case = next(
            case
            for case in cases
            if case["split"] == "dev" and case["expected"] and len(case["tags"]) > 1
        )
        test_case = next(case for case in cases if case["split"] == "test" and case["expected"])
        shared = _candidate(dev_case["case_id"], "PRIVATE-SHARED-SENTINEL")
        test_claim = _candidate(test_case["case_id"], "PRIVATE-TEST-SENTINEL")
        qwen = _receipt(
            "qwen3-4b",
            {
                dev_case["case_id"]: _json_claims(shared, shared),
                test_case["case_id"]: _json_claims(test_claim),
            },
        )
        phi = _receipt("phi-3.5-mini", {dev_case["case_id"]: _json_claims(shared)})
        bundle, _, _ = prepare_reviews([qwen, phi], seed=103)
        task_by_object = {task["candidate"]["object"]: task["task_id"] for task in bundle["tasks"]}
        human = _response_document(
            bundle,
            "human-a",
            "human",
            {
                task_by_object["PRIVATE-SHARED-SENTINEL"]: "supported",
                task_by_object["PRIVATE-TEST-SENTINEL"]: "unsupported",
            },
        )
        agent = _response_document(bundle, "agent-a", "agent", "uncertain")

        report = summarize_phenomena([qwen, phi], [human, agent], seed=103)

        qwen_dev = report["by_model"]["qwen3-4b"]["dev"]["by_tag"][dev_case["tags"][0]]
        self.assertEqual(qwen_dev["schema_valid_origin_count"], 2)
        self.assertEqual(qwen_dev["unique_review_task_count"], 1)
        self.assertEqual(
            [(row["reviewer_id"], row["reviewer_kind"]) for row in qwen_dev["reviewers"]],
            [("agent-a", "agent"), ("human-a", "human")],
        )
        self.assertEqual(qwen_dev["reviewers"][0]["uncertain_origin_count"], 2)
        self.assertIsNone(qwen_dev["reviewers"][0]["supported_rate_decisive_origins"])
        self.assertEqual(qwen_dev["reviewers"][1]["supported_rate_all_reviewed_origins"], 1.0)
        phi_dev = report["by_model"]["phi-3.5-mini"]["dev"]["by_tag"][dev_case["tags"][0]]
        self.assertEqual(phi_dev["schema_valid_origin_count"], 1)
        qwen_test = report["by_model"]["qwen3-4b"]["test"]["by_tag"][test_case["tags"][0]]
        self.assertEqual(qwen_test["schema_valid_origin_count"], 1)
        self.assertEqual(qwen_test["reviewers"][1]["unsupported_origin_count"], 1)
        phi_test = report["by_model"]["phi-3.5-mini"]["test"]["by_tag"][test_case["tags"][0]]
        self.assertEqual(phi_test["schema_valid_empty_output_count"], 1)
        self.assertEqual(phi_test["schema_valid_origin_count"], 0)
        self.assertIsNone(phi_test["reviewers"][0]["supported_rate_all_reviewed_origins"])
        self.assertIsNone(phi_test["reviewers"][0]["supported_rate_decisive_origins"])
        self.assertNotIn("PRIVATE-SHARED-SENTINEL", json.dumps(report))
        self.assertNotIn("PRIVATE-TEST-SENTINEL", json.dumps(report))

    def test_rejects_invalid_receipt_response_and_contradictory_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "receipt keys"):
            summarize_phenomena([b"{}"], [], seed=104)

        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "PRIVATE-MAPPING-SENTINEL")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, answer_key, provenance = prepare_reviews([receipt], seed=105)
        response = _response_document(bundle, "reviewer-a", "human", "supported")
        bad_response = json.loads(json.dumps(response))
        bad_response["bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle hash"):
            summarize_phenomena([receipt], [bad_response], seed=105)

        corrupted = json.loads(json.dumps(provenance))
        corrupted["task_provenance"][0]["origins"].clear()
        with (
            patch.object(
                phenomena_module,
                "prepare_reviews",
                return_value=(bundle, answer_key, corrupted),
            ),
            self.assertRaisesRegex(ValueError, "task provenance has no origin mapping"),
        ):
            summarize_phenomena([receipt], [response], seed=105)

    def test_is_deterministic_and_group_counts_ignore_receipt_order(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "PRIVATE-ORDER-SENTINEL")
        qwen = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        phi = _receipt("phi-3.5-mini", {case["case_id"]: _json_claims(claim)})
        receipts = [qwen, phi]
        bundle, _, _ = prepare_reviews(receipts, seed=106)
        response = _response_document(bundle, "reviewer-a", "agent", "uncertain")
        original_receipts = list(receipts)
        original_response = deepcopy(response)

        report = summarize_phenomena(receipts, [response], seed=106)
        repeated = summarize_phenomena(receipts, [response], seed=106)
        reversed_receipts = list(reversed(receipts))
        reversed_bundle, _, _ = prepare_reviews(reversed_receipts, seed=106)
        reversed_response = _response_document(reversed_bundle, "reviewer-a", "agent", "uncertain")
        reordered = summarize_phenomena(reversed_receipts, [reversed_response], seed=106)

        self.assertEqual(report, repeated)
        self.assertEqual(report["by_model"], reordered["by_model"])
        self.assertEqual(receipts, original_receipts)
        self.assertEqual(response, original_response)
        self.assertEqual(len(report["inputs"]["receipts"]), 2)
        self.assertTrue(
            all(row["sha256"] and row["canonical_sha256"] for row in report["inputs"]["receipts"])
        )
        self.assertTrue(report["inputs"]["responses"][0]["canonical_sha256"])
        self.assertNotIn("PRIVATE-ORDER-SENTINEL", json.dumps(report))

    def test_cli_saves_private_report_refuses_alias_and_preserves_inputs(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "PRIVATE-CLI-SENTINEL")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=107)
        response = _response_document(bundle, "reviewer-a", "human", "supported")
        response_bytes = json.dumps(response).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path, response_path = root / "receipt.json", root / "response.json"
            output_path = root / "report.json"
            receipt_path.write_bytes(receipt)
            response_path.write_bytes(response_bytes)
            args = [
                "--receipts",
                str(receipt_path),
                "--responses",
                str(response_path),
                "--seed",
                "107",
                "--output",
                str(output_path),
            ]

            self.assertEqual(main(args), 0)
            saved = output_path.read_bytes()
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertNotIn(b"PRIVATE-CLI-SENTINEL", saved)
            self.assertEqual(receipt_path.read_bytes(), receipt)
            self.assertEqual(response_path.read_bytes(), response_bytes)

            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(args), 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())
            self.assertEqual(output_path.read_bytes(), saved)

            alias_args = [*args[:-1], str(receipt_path)]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(alias_args), 2)
            self.assertIn("aliases an input", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), receipt)

            bad_receipt, invalid_output = root / "bad.json", root / "invalid-report.json"
            bad_receipt.write_text("{}", encoding="utf-8")
            invalid_args = [
                "--receipts",
                str(bad_receipt),
                "--seed",
                "107",
                "--output",
                str(invalid_output),
            ]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(invalid_args), 2)
            self.assertIn("receipt keys", stderr.getvalue())
            self.assertFalse(invalid_output.exists())


if __name__ == "__main__":
    unittest.main()
