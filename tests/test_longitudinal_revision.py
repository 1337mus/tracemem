"""Integrity checks for the longitudinal v2 revision and bundle binding."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

from tracemem.annotation_report import build_review_report
from tracemem.annotations import export_review_bundle, validate_responses
from tracemem.challenge import _validate_case
from tracemem.claims import parse_candidates, verify_candidate
from tracemem.schema import EvidenceRecord

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "examples/synthetic"
V1 = SYNTHETIC / "longitudinal-dev-v1.json"
V1_FOILS = SYNTHETIC / "longitudinal-dev-v1-foils.json"
V2 = SYNTHETIC / "longitudinal-dev-v2.json"
V2_FOILS = SYNTHETIC / "longitudinal-dev-v2-foils.json"
V2_UNLABELED = SYNTHETIC / "longitudinal-dev-v2-unlabeled.json"


_RECORD_FIELDS = ("session_id", "message_id", "role", "text", "recorded_at", "content_kind")


def _records(case: dict[str, Any]) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            source="longitudinal-v2",
            part_index=0,
            **{field: row[field] for field in _RECORD_FIELDS},
        )
        for row in case["messages"]
    ]


class LongitudinalRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1_raw, cls.v1_foil_raw, cls.v2_raw = (
            V1.read_bytes(),
            V1_FOILS.read_bytes(),
            V2.read_bytes(),
        )
        cls.v1 = json.loads(cls.v1_raw)
        cls.v1_foils = json.loads(cls.v1_foil_raw)
        cls.v2 = json.loads(cls.v2_raw)
        cls.v2_foils = json.loads(V2_FOILS.read_bytes())
        cls.unlabeled = json.loads(V2_UNLABELED.read_bytes())

    def test_v2_is_only_the_documented_revision_of_v1(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.v1_raw).hexdigest(),
            "d75b2237e739bfd8ff72be25d234f62004b6641a48a8bd9893442d64e8bc8fda",
        )
        self.assertEqual(
            hashlib.sha256(self.v1_foil_raw).hexdigest(),
            "26f65c6c80f8e1abcb4c9d7eaa1e7215dc41d114113f33f781550612d77f91be",
        )
        expected = copy.deepcopy(self.v1)
        case10 = next(case for case in expected["cases"] if case["case_id"] == "long-dev-case-10")
        claim = case10["expected"][1]
        claim["evidence"] = [
            span for span in claim["evidence"] if span["message_id"] != "long-dev-case-10-m06"
        ]
        self.assertEqual(
            [span["message_id"] for span in claim["evidence"]],
            ["long-dev-case-10-m07", "long-dev-case-10-m09"],
        )
        self.assertEqual(self.v2, expected)
        expected_foils = copy.deepcopy(self.v1_foils)
        moved = expected_foils["long-dev-case-07"][0]
        expected_foils["long-dev-case-07"] = []
        self.assertEqual(self.v2_foils, expected_foils)
        self.assertEqual(self.unlabeled, {"long-dev-case-07": [moved]})

    def test_v2_schema_citations_and_revision_counts(self) -> None:
        self.assertEqual(
            set(self.v2), {"version", "license", "annotation_status", "cases", "case_metadata"}
        )
        self.assertEqual(
            (self.v2["version"], self.v2["license"], self.v2["annotation_status"]),
            (1, "MIT", "agent_authored"),
        )
        cases = self.v2["cases"]
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(len(case["messages"]) for case in cases), 146)
        self.assertEqual(
            sum(len({m["session_id"] for m in case["messages"]}) for case in cases), 49
        )
        self.assertTrue(all(case["split"] == "dev" for case in cases))
        self.assertEqual(sum(len(case["expected"]) for case in cases), 13)
        self.assertEqual(sum(len(items) for items in self.v2_foils.values()), 12)
        self.assertEqual(sum(len(items) for items in self.unlabeled.values()), 1)
        self.assertEqual(
            sum(
                len({span["session_id"] for span in claim["evidence"]}) > 1
                for case in cases
                for claim in case["expected"]
            ),
            4,
        )
        by_id = {case["case_id"]: case for case in cases}
        for case in cases:
            _validate_case(case)
        candidates = [(case, claim) for case in cases for claim in case["expected"]]
        candidates += [
            (by_id[case_id], claim)
            for source in (self.v2_foils, self.unlabeled)
            for case_id, items in source.items()
            for claim in items
        ]
        for case, claim in candidates:
            parsed = parse_candidates({"claims": [claim]})[0]
            # Exact source integrity does not establish semantic support.
            self.assertEqual(verify_candidate(parsed, _records(case)).status, "REVIEW")

    def test_export_report_and_old_bundle_binding(self) -> None:
        blind, key = export_review_bundle(
            self.v2["cases"],
            hashlib.sha256(self.v2_raw).hexdigest(),
            self.v2_foils,
            seed=1729,
            case_metadata=self.v2["case_metadata"],
            unlabeled_by_case=self.unlabeled,
        )
        v1_blind, _ = export_review_bundle(
            self.v1["cases"],
            hashlib.sha256(self.v1_raw).hexdigest(),
            self.v1_foils,
            seed=1729,
            case_metadata=self.v1["case_metadata"],
        )
        self.assertEqual(len(blind["tasks"]), 26)
        self.assertEqual(len({task["task_id"] for task in blind["tasks"]}), 26)
        v1_task_ids = {task["task_id"] for task in v1_blind["tasks"]}
        v2_task_ids = {task["task_id"] for task in blind["tasks"]}
        self.assertEqual(len(v1_task_ids), 26)
        self.assertEqual(len(v2_task_ids), 26)
        self.assertTrue(v1_task_ids.isdisjoint(v2_task_ids))
        self.assertEqual(len(key["case_ids"]), 12)
        for membership, count in (("gold", 13), ("foil", 12), ("unlabeled", 1)):
            self.assertEqual(
                sum(row["membership"] == membership for row in key["task_keys"]), count
            )
        self.assertNotEqual(blind["bundle_sha256"], v1_blind["bundle_sha256"])
        self.assertEqual({row["case_id"] for row in key["task_keys"]}, set(key["case_ids"]))
        case07 = [row for row in key["task_keys"] if row["case_id"] == "long-dev-case-07"]
        self.assertEqual(
            {(row["membership"], row["reference_label"]) for row in case07},
            {("gold", "supported"), ("unlabeled", None)},
        )
        # This complete procedural response is test scaffolding, not reviewer evidence.
        v1_test_response = {
            "version": 1,
            "bundle_sha256": v1_blind["bundle_sha256"],
            "responses": [
                {
                    "task_id": task["task_id"],
                    "label": "uncertain",
                    "rationale": "Procedural test-only response; not reviewer evidence.",
                    "reviewer_id": "procedural-test",
                    "reviewer_kind": "agent",
                    "timestamp": "2026-10-14T00:00:00Z",
                }
                for task in v1_blind["tasks"]
            ],
        }
        validated = validate_responses(v1_blind, v1_test_response)
        self.assertEqual(len(validated), 26)
        self.assertEqual({row["task_id"] for row in validated}, v1_task_ids)
        with self.assertRaisesRegex(ValueError, "response bundle hash does not match blind bundle"):
            validate_responses(blind, v1_test_response)
        report = build_review_report(blind, key, [])
        self.assertEqual(report["reviewer_document_count"], 0)
        for kind in ("human", "agent"):
            self.assertEqual(report["by_reviewer_kind"][kind]["reviewers"], 0)
            self.assertEqual(report["by_reviewer_kind"][kind]["responses"], 0)
