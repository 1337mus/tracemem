import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tracemem.challenge import _validate_case  # noqa: E402
from training.build_pilot_eval import build_dataset  # noqa: E402

CASE_KEYS = {"case_id", "scenario_id", "split", "tags", "messages", "expected"}
MESSAGE_KEYS = {"session_id", "message_id", "role", "text", "recorded_at", "content_kind"}


class PilotEvalDataTests(unittest.TestCase):
    def test_dev_f01_and_f04_claims_have_direct_first_person_source(self):
        seeds = json.loads((ROOT / "datasets/sft-pilot/eval-seeds.json").read_text())
        cases = build_dataset(seeds, "dev")["cases"]
        source_markers = {
            "EVALDEV-F01": "I confirm my own position:",
            "EVALDEV-F04": "I am describing my own work:",
        }

        for case in cases:
            marker = source_markers.get(case["scenario_id"])
            if marker is None:
                continue
            source = case["messages"][1]
            self.assertEqual(source["role"], "user")
            self.assertIn(marker, source["text"])
            self.assertEqual(len(case["expected"]), 1)
            claim = case["expected"][0]
            self.assertEqual(claim["subject"], "user")
            self.assertEqual(claim["speaker_role"], "user")
            self.assertEqual(claim["attribution"], "direct_user_statement")
            self.assertIn(claim["evidence"][0]["quote"], source["text"])

    def test_test_split_builder_uses_test_ids_and_preserves_citations(self):
        seeds = json.loads((ROOT / "datasets/sft-pilot/eval-seeds.json").read_text())
        test_input = copy.deepcopy(seeds)
        for family in test_input["families"]:
            family["group_id"] = family["group_id"].replace("EVALDEV-", "EVALTEST-")
            family["template_group"] = family["template_group"].replace("EVALDEV-", "EVALTEST-")
            family["entity_ids"] = [
                entity.replace("EVALDEV-", "EVALTEST-") for entity in family["entity_ids"]
            ]

        built = build_dataset(test_input, "test")
        cases = built["cases"]
        metadata = built["case_metadata"]
        self.assertEqual(len(cases), 32)
        self.assertEqual(len(metadata), 32)
        self.assertEqual(sum(not case["expected"] for case in cases), 8)
        self.assertEqual(sum(len(case["expected"]) for case in cases), 40)

        groups = {}
        for case in cases:
            self.assertEqual(case["split"], "test")
            self.assertTrue(case["case_id"].startswith("EVALTEST-"))
            self.assertTrue(case["scenario_id"].startswith("EVALTEST-"))
            groups.setdefault(case["scenario_id"], []).append(case)
            for message in case["messages"]:
                self.assertTrue(message["session_id"].startswith("EVALTEST-"))
                self.assertTrue(message["message_id"].startswith("EVALTEST-"))
            for claim in case["expected"]:
                for span in claim["evidence"]:
                    message = next(
                        row
                        for row in case["messages"]
                        if row["session_id"] == span["session_id"]
                        and row["message_id"] == span["message_id"]
                    )
                    self.assertEqual(
                        message["text"][span["char_start"] : span["char_end"]], span["quote"]
                    )
            info = metadata[case["case_id"]]
            self.assertTrue(info["scenario_group"].startswith("EVALTEST-"))
            self.assertTrue(info["template_group"].startswith("EVALTEST-"))
            self.assertTrue(all(entity.startswith("EVALTEST-") for entity in info["entity_ids"]))
            _validate_case(case)

        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(group) == 4 for group in groups.values()))

    def test_dev_seeds_expand_to_deterministic_valid_four_variant_groups(self):
        seeds = json.loads((ROOT / "datasets/sft-pilot/eval-seeds.json").read_text())
        built = build_dataset(seeds, "dev")
        repeated = build_dataset(seeds, "dev")

        self.assertEqual(json.dumps(built, sort_keys=True), json.dumps(repeated, sort_keys=True))
        self.assertEqual(
            set(built),
            {"version", "license", "annotation_status", "provenance", "cases", "case_metadata"},
        )
        self.assertEqual(built["version"], 1)
        self.assertEqual(built["license"], "MIT")
        self.assertEqual(built["annotation_status"], "agent_authored")
        self.assertEqual(built["provenance"], "newly_authored_fiction")

        cases = built["cases"]
        metadata = built["case_metadata"]
        self.assertEqual(len(cases), 32)
        self.assertEqual(len(metadata), 32)
        self.assertEqual(len({case["case_id"] for case in cases}), 32)
        self.assertEqual(sum(not case["expected"] for case in cases), 8)
        plan_objects = [
            claim["object"]
            for case in cases
            for claim in case["expected"]
            if claim["predicate"] == "PLANS"
        ]
        self.assertEqual(
            plan_objects,
            [
                "try Finch for the next archive pass",
                "use Harbor for the next label round",
                "finish the Linden tiles this season",
                "finish the Moss labels before May",
            ],
        )

        groups = {}
        for case in cases:
            self.assertEqual(set(case), CASE_KEYS)
            self.assertEqual(case["split"], "dev")
            self.assertTrue(case["case_id"].startswith("EVALDEV-"))
            self.assertEqual(len(case["messages"]), 6)
            for message in case["messages"]:
                self.assertEqual(set(message), MESSAGE_KEYS)
                self.assertTrue(message["session_id"].startswith("EVALDEV-"))
                self.assertTrue(message["message_id"].startswith("EVALDEV-"))
            info = metadata[case["case_id"]]
            self.assertEqual(set(info), {"scenario_group", "template_group", "entity_ids"})
            self.assertTrue(info["scenario_group"].startswith("EVALDEV-"))
            self.assertTrue(info["template_group"].startswith("EVALDEV-"))
            self.assertTrue(all(entity.startswith("EVALDEV-") for entity in info["entity_ids"]))
            groups.setdefault(info["scenario_group"], []).append(case)
            for claim in case["expected"]:
                self.assertNotIn(claim["subject"], info["entity_ids"])
                self.assertTrue(
                    claim["subject"] == "user"
                    or any(claim["subject"] in message["text"] for message in case["messages"])
                )
                for span in claim["evidence"]:
                    message = next(
                        row
                        for row in case["messages"]
                        if row["session_id"] == span["session_id"]
                        and row["message_id"] == span["message_id"]
                    )
                    self.assertEqual(
                        message["text"][span["char_start"] : span["char_end"]], span["quote"]
                    )
            _validate_case(case)

        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(group) == 4 for group in groups.values()))
        for group in groups.values():
            self.assertEqual(len({metadata[row["case_id"]]["template_group"] for row in group}), 1)
            self.assertEqual(
                len({tuple(metadata[row["case_id"]]["entity_ids"]) for row in group}), 1
            )


if __name__ == "__main__":
    unittest.main()
