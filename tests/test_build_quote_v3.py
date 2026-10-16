from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from training.build_quote_v3 import build

ROOT = Path(__file__).resolve().parents[1]
V2_TRAIN = ROOT / "datasets" / "sft-quote-v2" / "train.json"
V3_DIR = ROOT / "datasets" / "sft-quote-v3"


def probe_seeds() -> dict[str, object]:
    return {
        "seed": 23,
        "split": "train",
        "variants": 8,
        "value_banks": {
            "person": [f"Person {i}" for i in range(8)],
            "product": [f"Product {i}" for i in range(8)],
            "object": [f"record {i}" for i in range(8)],
            "place": [f"place {i}" for i in range(8)],
            "date_from": [f"2025-02-{i + 1:02d}" for i in range(8)],
            "date_to": [f"2025-03-{i + 1:02d}" for i in range(8)],
        },
        "families": [
            {
                "family_id": "V3T00-probe",
                "tags": ["chronology", "completed"],
                "messages": [
                    {
                        "ref": "u1",
                        "role": "user",
                        "session": 1,
                        "content_kind": "conversation",
                        "text": "I filed the {object} on {date_from}.",
                    },
                    {
                        "ref": "a1",
                        "role": "assistant",
                        "session": 1,
                        "content_kind": "conversation",
                        "text": "Would you like the record attached to the folder?",
                    },
                ],
                "expected": [
                    {
                        "claim": "The user filed {object} on {date_from}.",
                        "message_ref": "u1",
                        "quote": "I filed the {object} on {date_from}.",
                        "valid_from": "{date_from}",
                        "valid_to": None,
                    }
                ],
            }
        ],
    }


class QuoteV3BuilderTests(unittest.TestCase):
    def test_build_keeps_first_twelve_replay_rows_and_namespaces_merged_ids(self) -> None:
        source = json.loads(V2_TRAIN.read_text(encoding="utf-8"))
        seeds = probe_seeds()
        source_before = copy.deepcopy(source)
        seeds_before = copy.deepcopy(seeds)

        result = build(seeds, source)

        source_variants: Counter[str] = Counter()
        expected_replay_ids: list[str] = []
        for case in source["cases"]:
            family = case["scenario_group"]
            if source_variants[family] < 12:
                expected_replay_ids.append(f"V3T-REPLAY-{case['case_id']}")
            source_variants[family] += 1

        cases = result["cases"]
        self.assertEqual(len(cases), 576 + 8)
        self.assertEqual([case["case_id"] for case in cases[:576]], expected_replay_ids)
        self.assertTrue(all(case["case_id"].startswith("V3T-TRAIN-") for case in cases[576:]))
        for case in cases:
            self.assertTrue(case["case_id"].startswith("V3T-"))
            self.assertTrue(case["scenario_group"].startswith("V3T-"))
            self.assertTrue(case["template_group"].startswith("V3T-"))
            self.assertTrue(all(item.startswith("V3T-") for item in case["entity_ids"]))
            message_ids = {message["message_id"] for message in case["messages"]}
            for message in case["messages"]:
                self.assertTrue(message["session_id"].startswith("V3T-"))
                self.assertTrue(message["message_id"].startswith("V3T-"))
            for claim in case["expected"]:
                self.assertTrue(claim["message_id"].startswith("V3T-"))
                self.assertIn(claim["message_id"], message_ids)

        self.assertEqual(source, source_before)
        self.assertEqual(seeds, seeds_before)

    def test_checked_in_train_matches_768_row_72_family_abstention_contract(self) -> None:
        source = json.loads(V2_TRAIN.read_text(encoding="utf-8"))
        seeds = json.loads((V3_DIR / "target-train-seeds.json").read_text(encoding="utf-8"))
        checked_in = json.loads((V3_DIR / "train.json").read_text(encoding="utf-8"))

        built = build(seeds, source)
        self.assertEqual(checked_in, built)
        cases = built["cases"]
        groups = Counter(case["scenario_group"] for case in cases)
        replay = [case for case in cases if case["case_id"].startswith("V3T-REPLAY-")]
        authored = [case for case in cases if case["case_id"].startswith("V3T-TRAIN-")]

        self.assertEqual(len(cases), 768)
        self.assertEqual(len(groups), 72)
        self.assertEqual(Counter(groups.values()), Counter({12: 48, 8: 24}))
        self.assertEqual(len(replay), 576)
        self.assertEqual(len(authored), 192)
        self.assertEqual(sum(not case["expected"] for case in replay), 108)
        self.assertEqual(sum(not case["expected"] for case in authored), 32)
        self.assertEqual(sum(not case["expected"] for case in cases), 140)

        source_variants: Counter[str] = Counter()
        expected_replay = []
        for case in source["cases"]:
            family = case["scenario_group"]
            if source_variants[family] < 12:
                expected_replay.append(case)
            source_variants[family] += 1
        normalized_replay = []
        for case in replay:
            normalized = copy.deepcopy(case)
            for field in ("case_id", "scenario_group", "template_group"):
                normalized[field] = normalized[field].removeprefix("V3T-REPLAY-")
            normalized["entity_ids"] = [
                value.removeprefix("V3T-REPLAY-") for value in normalized["entity_ids"]
            ]
            for message in normalized["messages"]:
                message["session_id"] = message["session_id"].removeprefix("V3T-REPLAY-")
                message["message_id"] = message["message_id"].removeprefix("V3T-REPLAY-")
            for claim in normalized["expected"]:
                claim["message_id"] = claim["message_id"].removeprefix("V3T-REPLAY-")
            normalized_replay.append(normalized)
        self.assertEqual(normalized_replay, expected_replay)

        authored_families = {case["scenario_group"]: case for case in authored}
        self.assertEqual(len(authored_families), 24)
        self.assertTrue(all(2 <= len(case["messages"]) <= 6 for case in authored))
        self.assertGreaterEqual(
            sum(4 <= len(case["messages"]) <= 6 for case in authored_families.values()),
            4,
        )

        case_ids = [case["case_id"] for case in cases]
        entity_ids = [value for case in cases for value in case["entity_ids"]]
        message_ids = [message["message_id"] for case in cases for message in case["messages"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertEqual(len(entity_ids), len(set(entity_ids)))
        self.assertEqual(len(message_ids), len(set(message_ids)))
        for case in cases:
            self.assertTrue(case["case_id"].startswith("V3T-"))
            self.assertTrue(case["scenario_group"].startswith("V3T-"))
            self.assertTrue(case["template_group"].startswith("V3T-"))
            self.assertTrue(all(item.startswith("V3T-") for item in case["entity_ids"]))
            by_message = {message["message_id"] for message in case["messages"]}
            for message in case["messages"]:
                self.assertTrue(message["session_id"].startswith("V3T-"))
                self.assertTrue(message["message_id"].startswith("V3T-"))
            for claim in case["expected"]:
                self.assertTrue(claim["message_id"].startswith("V3T-"))
                self.assertIn(claim["message_id"], by_message)

    def test_build_is_deterministic_and_rejects_malformed_source_usefully(self) -> None:
        source = json.loads(V2_TRAIN.read_text(encoding="utf-8"))
        seeds = probe_seeds()
        self.assertEqual(build(seeds, source), build(seeds, source))

        with self.assertRaisesRegex(ValueError, "invalid v2 training source"):
            build(seeds, {})

        short_source = copy.deepcopy(source)
        short_source["cases"].pop(0)
        with self.assertRaisesRegex(ValueError, "has 15 variants, expected 16"):
            build(seeds, short_source)

    def test_build_rejects_a_value_bank_with_the_wrong_count(self) -> None:
        source = json.loads(V2_TRAIN.read_text(encoding="utf-8"))
        seeds = probe_seeds()
        seeds["value_banks"]["person"].pop()  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "value bank person must contain exactly 8 values"):
            build(seeds, source)


if __name__ == "__main__":
    unittest.main()
