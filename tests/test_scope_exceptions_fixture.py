"""Integrity and blinded-export checks for the scoped exception dev slice."""

from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

from tracemem.annotations import export_review_bundle
from tracemem.challenge import _validate_case, load_challenge
from tracemem.claims import parse_candidates, verify_candidate
from tracemem.dataset_audit import audit_cases
from tracemem.scenarios import load_scenarios
from tracemem.schema import EvidenceRecord

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/synthetic/scope-exceptions-dev-v1.json"
FOILS = ROOT / "examples/synthetic/scope-exceptions-dev-v1-foils.json"
LONGITUDINAL = ROOT / "examples/synthetic/longitudinal-dev-v2.json"


def _records(case: dict[str, Any]) -> list[EvidenceRecord]:
    fields = ("session_id", "message_id", "role", "text", "recorded_at", "content_kind")
    return [
        EvidenceRecord(
            source="scope-exceptions-fixture",
            part_index=0,
            **{field: message[field] for field in fields},
        )
        for message in case["messages"]
    ]


class ScopeExceptionsFixtureTests(unittest.TestCase):
    raw: bytes
    payload: dict[str, Any]
    cases: list[dict[str, Any]]
    foils: dict[str, list[dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FIXTURE.read_bytes()
        cls.payload = json.loads(cls.raw)
        cls.cases = cls.payload["cases"]
        cls.foils = json.loads(FOILS.read_text(encoding="utf-8"))

    def test_static_contract_citations_and_case_shape(self) -> None:
        self.assertEqual(
            set(self.payload), {"version", "license", "annotation_status", "cases", "case_metadata"}
        )
        self.assertEqual(self.payload["version"], 1)
        self.assertEqual(self.payload["license"], "MIT")
        self.assertEqual(self.payload["annotation_status"], "agent_authored")
        self.assertEqual(len(self.cases), 6)
        self.assertEqual(sum(len(case["expected"]) for case in self.cases), 10)
        self.assertEqual(sum(map(len, self.foils.values())), 12)
        self.assertEqual(
            set(self.payload["case_metadata"]), {case["case_id"] for case in self.cases}
        )
        self.assertEqual(set(self.foils), {case["case_id"] for case in self.cases})

        for case in self.cases:
            _validate_case(case)
            self.assertTrue(case["case_id"].startswith("scope-dev-"))
            self.assertTrue(case["scenario_id"].startswith("scope-dev-"))
            self.assertEqual(case["split"], "dev")
            self.assertGreaterEqual(len(case["messages"]), 3)
            self.assertLessEqual(len(case["messages"]), 6)
            self.assertGreaterEqual(len({message["session_id"] for message in case["messages"]}), 2)
            timestamps = [message["recorded_at"] for message in case["messages"]]
            parsed = [datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ") for value in timestamps]
            self.assertEqual(timestamps, [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in parsed])
            self.assertTrue(
                all(left < right for left, right in zip(parsed, parsed[1:], strict=False))
            )
            self.assertNotIn(
                "fictional",
                " ".join(message["text"] for message in case["messages"]).casefold(),
            )
            for message in case["messages"]:
                self.assertTrue(message["session_id"].startswith("scope-dev-"))
                self.assertTrue(message["message_id"].startswith("scope-dev-"))
            for target in case["expected"]:
                candidate = parse_candidates({"claims": [target]})[0]
                self.assertEqual(verify_candidate(candidate, _records(case)).status, "REVIEW")

        for case_id, values in self.payload["case_metadata"].items():
            self.assertTrue(case_id.startswith("scope-dev-"))
            self.assertTrue(values["scenario_group"].startswith("scope-dev-"))
            self.assertTrue(values["template_group"].startswith("scope-dev-"))
            self.assertTrue(all(value.startswith("scope-dev-") for value in values["entity_ids"]))

    def test_foils_are_schema_valid_with_exact_citable_source_spans(self) -> None:
        by_id = {case["case_id"]: case for case in self.cases}
        for case_id, candidates in self.foils.items():
            self.assertEqual(len(candidates), 2)
            for raw_candidate in candidates:
                candidate = parse_candidates({"claims": [raw_candidate]})[0]
                self.assertEqual(
                    verify_candidate(candidate, _records(by_id[case_id])).status, "REVIEW"
                )

    def test_identities_and_declared_metadata_are_disjoint_from_prior_84_cases(self) -> None:
        old_scenarios, old_manifest, _ = load_scenarios()
        old_challenge, _ = load_challenge()
        old_longitudinal = json.loads(LONGITUDINAL.read_text(encoding="utf-8"))
        previous = [*old_scenarios, *old_challenge, *old_longitudinal["cases"]]
        self.assertEqual(len(previous), 84)

        old_labels = {
            value for case in previous for value in (case["case_id"], case["scenario_id"])
        }
        new_labels = {
            value for case in self.cases for value in (case["case_id"], case["scenario_id"])
        }
        self.assertFalse(new_labels & old_labels)
        old_pairs = {
            (message["session_id"], message["message_id"])
            for case in previous
            for message in case["messages"]
        }
        new_pairs = {
            (message["session_id"], message["message_id"])
            for case in self.cases
            for message in case["messages"]
        }
        self.assertFalse(new_pairs & old_pairs)

        old_metadata = {**old_manifest["case_metadata"], **old_longitudinal["case_metadata"]}
        old_groups = {
            value
            for row in old_metadata.values()
            for value in (row["scenario_group"], row["template_group"])
        }
        old_entities = {value for row in old_metadata.values() for value in row["entity_ids"]}
        new_metadata = self.payload["case_metadata"]
        new_groups = {
            value
            for row in new_metadata.values()
            for value in (row["scenario_group"], row["template_group"])
        }
        new_entities = {value for row in new_metadata.values() for value in row["entity_ids"]}
        self.assertFalse(new_groups & old_groups)
        self.assertFalse(new_entities & old_entities)

    def test_blind_export_is_deterministic_with_a_separate_dev_only_key(self) -> None:
        dataset_hash = hashlib.sha256(self.raw).hexdigest()
        kwargs = {
            "foils_by_case": self.foils,
            "seed": 1729,
            "case_metadata": self.payload["case_metadata"],
        }
        blind, answer_key = export_review_bundle(self.cases, dataset_hash, **kwargs)
        repeated, repeated_key = export_review_bundle(self.cases, dataset_hash, **kwargs)
        self.assertEqual((blind, answer_key), (repeated, repeated_key))
        self.assertEqual(blind["dataset_sha256"], dataset_hash)
        self.assertEqual(len(blind["tasks"]), 22)
        self.assertEqual(len(answer_key["case_ids"]), 6)
        self.assertEqual(sum(row["membership"] == "gold" for row in answer_key["task_keys"]), 10)
        self.assertEqual(sum(row["membership"] == "foil" for row in answer_key["task_keys"]), 12)
        self.assertEqual({row["split"] for row in answer_key["task_keys"]}, {"dev"})
        self.assertEqual(
            {row["case_id"] for row in answer_key["task_keys"]}, set(answer_key["case_ids"])
        )
        self.assertNotIn("case_metadata", blind)
        self.assertNotIn("task_keys", blind)

    def test_pairwise_lexical_audit_finds_no_exact_duplicate_conversations(self) -> None:
        report = audit_cases(self.cases)
        self.assertEqual(report["case_count"], 6)
        self.assertEqual(report["split_case_counts"]["dev"], 6)
        self.assertEqual(report["exact_duplicate_pairs"], [])


if __name__ == "__main__":
    unittest.main()
