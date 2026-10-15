"""Integrity checks for the separately curated longitudinal development slice."""

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
FIXTURE = ROOT / "examples/synthetic/longitudinal-dev-v1.json"
FOILS = ROOT / "examples/synthetic/longitudinal-dev-v1-foils.json"


def _records(case: dict[str, Any]) -> list[EvidenceRecord]:
    fields = ("session_id", "message_id", "role", "text", "recorded_at", "content_kind")
    return [
        EvidenceRecord(
            source="longitudinal-fixture",
            part_index=0,
            **{field: message[field] for field in fields},
        )
        for message in case["messages"]
    ]


class LongitudinalFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FIXTURE.read_bytes()
        cls.payload = json.loads(cls.raw)
        cls.cases = cls.payload["cases"]
        cls.foils = json.loads(FOILS.read_text(encoding="utf-8"))

    def test_case_contract_citations_and_longitudinal_shape(self) -> None:
        self.assertEqual(
            set(self.payload), {"version", "license", "annotation_status", "cases", "case_metadata"}
        )
        self.assertEqual(self.payload["version"], 1)
        self.assertEqual(self.payload["license"], "MIT")
        self.assertEqual(self.payload["annotation_status"], "agent_authored")
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(sum(len(case["expected"]) for case in self.cases), 13)
        self.assertEqual(sum(map(len, self.foils.values())), 13)
        self.assertEqual(
            set(self.payload["case_metadata"]), {case["case_id"] for case in self.cases}
        )
        no_memory = []
        multi_session_targets = 0
        for case in self.cases:
            _validate_case(case)
            self.assertTrue(case["case_id"].startswith("long-dev-"))
            self.assertTrue(case["scenario_id"].startswith("long-dev-"))
            self.assertEqual(case["split"], "dev")
            self.assertGreaterEqual(len(case["messages"]), 10)
            self.assertLessEqual(len(case["messages"]), 14)
            self.assertGreaterEqual(len({m["session_id"] for m in case["messages"]}), 4)
            timestamps = [m["recorded_at"] for m in case["messages"]]
            times = [datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ") for value in timestamps]
            self.assertEqual(timestamps, [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times])
            self.assertTrue(all(a < b for a, b in zip(times, times[1:], strict=False)))
            self.assertGreater((times[-1] - times[0]).days, 30)
            if case["case_id"] == "long-dev-case-06":
                self.assertGreaterEqual(times[0], datetime(2026, 5, 19))
            if not case["expected"]:
                no_memory.append(case["case_id"])
            else:
                self.assertLessEqual(len(case["expected"]), 3)
            records = _records(case)
            for raw_claim in case["expected"]:
                parsed = parse_candidates({"claims": [raw_claim]})[0]
                self.assertEqual(verify_candidate(parsed, records).status, "REVIEW")
                multi_session_targets += len({span.session_id for span in parsed.evidence}) >= 2
        self.assertEqual(len(no_memory), 2)
        self.assertEqual(sum(bool(case["expected"]) for case in self.cases), 10)
        self.assertEqual(multi_session_targets, 5)

    def test_foils_have_exact_source_citations_and_schema_validity(self) -> None:
        self.assertEqual(set(self.foils), {case["case_id"] for case in self.cases})
        by_id = {case["case_id"]: case for case in self.cases}
        for case_id, candidates in self.foils.items():
            self.assertGreaterEqual(len(candidates), 1)
            self.assertLessEqual(len(candidates), 2)
            records = _records(by_id[case_id])
            for raw_claim in candidates:
                parsed = parse_candidates({"claims": [raw_claim]})[0]
                # Integrity is deterministic; semantic support remains a human-review task.
                self.assertEqual(verify_candidate(parsed, records).status, "REVIEW")

    def test_dev_groups_and_identities_do_not_overlap_existing_fixtures(self) -> None:
        old_scenarios, old_manifest, _ = load_scenarios()
        old_challenge, _ = load_challenge()
        previous = [*old_scenarios, *old_challenge]
        prior_labels = {value for c in previous for value in (c["case_id"], c["scenario_id"])}
        new_labels = {value for c in self.cases for value in (c["case_id"], c["scenario_id"])}
        self.assertFalse(new_labels & prior_labels)
        old_pairs = {(m["session_id"], m["message_id"]) for c in previous for m in c["messages"]}
        new_pairs = {(m["session_id"], m["message_id"]) for c in self.cases for m in c["messages"]}
        self.assertFalse(new_pairs & old_pairs)
        old_meta = old_manifest["case_metadata"]
        old_groups = {
            value
            for row in old_meta.values()
            for value in (row["scenario_group"], row["template_group"])
        }
        old_entities = {entity for row in old_meta.values() for entity in row["entity_ids"]}
        meta = self.payload["case_metadata"]
        new_groups = {
            value
            for row in meta.values()
            for value in (row["scenario_group"], row["template_group"])
        }
        new_entities = {entity for row in meta.values() for entity in row["entity_ids"]}
        self.assertFalse(new_groups & old_groups)
        self.assertFalse(new_entities & old_entities)

    def test_blind_export_binds_dataset_hash_and_retains_every_case(self) -> None:
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
        self.assertEqual(len(blind["tasks"]), 26)
        self.assertEqual(len(answer_key["case_ids"]), 12)
        self.assertEqual({row["membership"] for row in answer_key["task_keys"]}, {"gold", "foil"})
        self.assertEqual(sum(row["membership"] == "gold" for row in answer_key["task_keys"]), 13)
        self.assertEqual(sum(row["membership"] == "foil" for row in answer_key["task_keys"]), 13)
        self.assertEqual(
            {row["case_id"] for row in answer_key["task_keys"]}, set(answer_key["case_ids"])
        )
        changed_hash = hashlib.sha256(b"different longitudinal dataset").hexdigest()
        changed, _ = export_review_bundle(self.cases, changed_hash, **kwargs)
        self.assertNotEqual(
            {task["task_id"] for task in blind["tasks"]},
            {task["task_id"] for task in changed["tasks"]},
        )

    def test_audit_finds_no_exact_duplicate_conversations(self) -> None:
        report = audit_cases(self.cases)
        self.assertEqual(report["case_count"], 12)
        self.assertEqual(report["split_case_counts"]["dev"], 12)
        self.assertEqual(report["no_memory_case_count"], 2)
        self.assertEqual(report["exact_duplicate_pairs"], [])
