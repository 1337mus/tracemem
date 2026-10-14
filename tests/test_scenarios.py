"""Strict fixture and split-integrity tests for the authored scenario set."""

from __future__ import annotations

import hashlib
import unittest
from copy import deepcopy
from importlib.resources import files
from unittest.mock import patch

from tracemem.scenarios import load_scenarios, validate_scenarios


def _payload() -> dict[str, object]:
    cases, manifest, _ = load_scenarios()
    return {
        **manifest,
        "cases": cases,
    }


class ScenarioValidationTests(unittest.TestCase):
    def test_unsupported_version_is_rejected_before_case_validation(self) -> None:
        payload = {
            "version": 2,
            "license": "MIT",
            "annotation_status": "agent_authored",
            "cases": [],
            "case_metadata": {},
        }

        with self.assertRaisesRegex(ValueError, "unsupported scenario version"):
            validate_scenarios(payload)

    def test_loader_returns_manifest_and_raw_file_hash(self) -> None:
        cases, manifest, digest = load_scenarios()
        raw = files("tracemem").joinpath("data", "scenarios_v1.json").read_bytes()

        self.assertEqual(len(cases), 48)
        self.assertEqual(
            set(manifest), {"version", "license", "annotation_status", "case_metadata"}
        )
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["license"], "MIT")
        self.assertEqual(manifest["annotation_status"], "agent_authored")
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            manifest["case_metadata"]["dev-balcony-basil"]["entity_ids"],
            ["dev-balcony-basil:balcony-basil", "dev-balcony-basil:awning"],
        )

    def test_fixture_has_exact_split_message_and_no_memory_counts(self) -> None:
        cases, _ = validate_scenarios(_payload())

        self.assertEqual(
            {split: sum(case["split"] == split for case in cases) for split in ("dev", "test")},
            {"dev": 24, "test": 24},
        )
        self.assertEqual(sum(not case["expected"] for case in cases), 8)
        for case in cases:
            self.assertIn(len(case["messages"]), {3, 4, 5})
            self.assertGreaterEqual(len({message["session_id"] for message in case["messages"]}), 2)

    def test_duplicate_case_and_scenario_ids_are_rejected(self) -> None:
        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        cases = payload["cases"]
        cases[1]["case_id"] = cases[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            validate_scenarios(payload)

        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        cases = payload["cases"]
        cases[1]["scenario_id"] = cases[0]["scenario_id"]
        with self.assertRaisesRegex(ValueError, "duplicate scenario_id"):
            validate_scenarios(payload)

    def test_cross_split_entity_group_is_rejected(self) -> None:
        payload = _payload()
        payload["case_metadata"] = deepcopy(payload["case_metadata"])
        metadata = payload["case_metadata"]
        dev_entity = metadata["dev-balcony-basil"]["entity_ids"][0]
        metadata["test-dinghy-sail"]["entity_ids"][0] = dev_entity

        with self.assertRaisesRegex(ValueError, "entity_id .* crosses scenario splits"):
            validate_scenarios(payload)

    def test_cross_split_scenario_and_template_groups_are_rejected(self) -> None:
        for field in ("scenario_group", "template_group"):
            with self.subTest(field=field):
                payload = _payload()
                payload["case_metadata"] = deepcopy(payload["case_metadata"])
                metadata = payload["case_metadata"]
                metadata["test-dinghy-sail"][field] = metadata["dev-balcony-basil"][field]

                with self.assertRaisesRegex(ValueError, f"{field} .* crosses scenario splits"):
                    validate_scenarios(payload)

    def test_timestamp_order_and_canonical_format_are_required(self) -> None:
        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        first_case = payload["cases"][0]
        first_case["messages"][1]["recorded_at"] = first_case["messages"][0]["recorded_at"]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_scenarios(payload)

        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        payload["cases"][0]["messages"][0]["recorded_at"] = "2026-1-02T09:00:00Z"
        with self.assertRaisesRegex(ValueError, "canonical UTC"):
            validate_scenarios(payload)

    def test_challenge_validator_rejects_bad_gold_span(self) -> None:
        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        claim = payload["cases"][0]["expected"][0]
        claim["evidence"][0]["quote"] = "incorrect quote"

        with self.assertRaisesRegex(ValueError, "expected evidence failed integrity checks"):
            validate_scenarios(payload)

    def test_reported_sources_and_negative_preferences_keep_their_scope(self) -> None:
        cases = {case["case_id"]: case for case in _payload()["cases"]}

        for case_id in (
            "dev-walker-pickup",
            "dev-gallery-hours",
            "dev-quoted-checklist",
            "test-cumin-recipe",
            "test-archive-retention",
            "test-theater-understudy",
        ):
            with self.subTest(case_id=case_id):
                self.assertEqual(cases[case_id]["expected"][0]["predicate"], "REPORTS")

        reading = cases["test-reading-corrections"]["expected"][0]
        self.assertEqual(reading["modality"], "preference")
        self.assertEqual(reading["polarity"], "negative")
        self.assertNotIn("not", reading["object"].casefold())
        self.assertIn("while reading aloud", reading["object"])

        piano = cases["test-piano-reverb"]["expected"][0]
        self.assertEqual(piano["modality"], "preference")
        self.assertEqual(piano["polarity"], "negative")
        self.assertEqual(piano["object"], "reverb on the piano track")
        self.assertEqual(len(cases["dev-brass-latch"]["expected"][0]["evidence"]), 2)
        self.assertEqual(len(cases["dev-quoted-checklist"]["expected"][0]["evidence"]), 2)
        self.assertEqual(len(cases["test-sensor-placement"]["expected"][0]["evidence"]), 2)

        sqlite = cases["dev-sqlite-notes"]["expected"][0]
        self.assertEqual(
            (sqlite["subject"], sqlite["predicate"], sqlite["object"]),
            ("user", "DECIDED", "SQLite for the prototype"),
        )
        robot = cases["dev-robot-test-log"]["expected"][0]
        self.assertIn("this season", robot["object"])
        coffee = cases["dev-coffee-after-lunch"]["expected"][0]
        self.assertEqual((coffee["predicate"], coffee["object"]), ("USES", "coffee after lunch"))
        theater = cases["test-theater-understudy"]["expected"][0]
        self.assertEqual((theater["predicate"], theater["modality"]), ("REPORTS", "tentative"))
        self.assertIn("considering auditioning", theater["object"])
        fundraiser = cases["test-fundraiser-door"]["expected"][0]
        self.assertEqual((fundraiser["predicate"], fundraiser["modality"]), ("PLANS", "intention"))

        route = cases["test-route-timing"]
        self.assertIn("Through August 31, 2025", route["messages"][3]["text"])
        self.assertIn("since September 1, 2025", route["messages"][3]["text"])

    def test_duplicate_json_keys_are_rejected_by_loader(self) -> None:
        for raw in (b'{"version": 1, "version": 1}', b'{"version": NaN}'):
            with self.subTest(raw=raw):
                with patch("tracemem.scenarios.files") as resource_root:
                    resource_root.return_value.joinpath.return_value.read_bytes.return_value = raw
                    with self.assertRaisesRegex(ValueError, "scenario fixture is not valid JSON"):
                        load_scenarios()

    def test_split_counts_and_metadata_keys_are_strict(self) -> None:
        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        payload["cases"][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "scenario split must be dev or test"):
            validate_scenarios(payload)

        payload = _payload()
        payload["cases"] = deepcopy(payload["cases"])
        payload["cases"][0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split counts must be exactly 24 dev and 24 test"):
            validate_scenarios(payload)

        payload = _payload()
        payload["case_metadata"] = deepcopy(payload["case_metadata"])
        payload["case_metadata"]["dev-balcony-basil"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "metadata .* missing or unexpected fields"):
            validate_scenarios(payload)


if __name__ == "__main__":
    unittest.main()
