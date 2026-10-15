"""Contract checks for the authored fiction-only SFT pilot training slice."""

from __future__ import annotations

import json
import re
import runpy
import unittest
from pathlib import Path

from tracemem.challenge import _validate_case

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "datasets/sft-pilot/train-seeds.json"
TRAIN = ROOT / "datasets/sft-pilot/train.json"
BUILDER = ROOT / "training/build_pilot_train.py"


class PilotTrainDataTests(unittest.TestCase):
    def test_generated_pilot_contains_512_train_cases(self) -> None:
        self.assertTrue(TRAIN.is_file(), "the generated pilot training fixture must exist")
        payload = json.loads(TRAIN.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload),
            {"version", "license", "annotation_status", "provenance", "cases", "case_metadata"},
        )
        self.assertEqual(len(payload["cases"]), 512)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["license"], "MIT")
        self.assertEqual(payload["annotation_status"], "agent_authored")
        self.assertEqual(payload["provenance"], "newly_authored_fiction")
        seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
        self.assertEqual(len(seeds["seeds"]), 32)
        self.assertEqual(len({seed["family_id"] for seed in seeds["seeds"]}), 32)
        self.assertTrue(all(seed["family_id"].startswith("TRAIN-F") for seed in seeds["seeds"]))
        required_tags = {
            "decision",
            "tentative",
            "negative_fact",
            "intention",
            "completed",
            "exact_date",
            "imprecise_date",
            "third_party_report",
            "tool_observation",
            "causality",
            "location",
            "ownership",
            "multiple_sessions",
            "correction",
            "assistant_decoy",
            "quoted_instruction",
            "unresolved_referent",
            "question",
            "hypothetical",
            "empty_claim",
        }
        self.assertTrue(required_tags <= {tag for seed in seeds["seeds"] for tag in seed["tags"]})
        templates = {
            tuple(re.sub(r"\{[^}]+\}", "{}", message["text"]) for message in seed["messages"])
            for seed in seeds["seeds"]
        }
        self.assertEqual(len(templates), 32, "families must differ in situation, not only names")

        cases = payload["cases"]
        self.assertTrue(all(case["split"] == "train" for case in cases))
        self.assertEqual(len({case["case_id"] for case in cases}), 512)
        self.assertEqual(set(payload["case_metadata"]), {case["case_id"] for case in cases})
        self.assertEqual(len({case["scenario_id"] for case in cases}), 32)
        self.assertEqual(
            len({row["scenario_group"] for row in payload["case_metadata"].values()}), 32
        )
        self.assertEqual(
            len({row["template_group"] for row in payload["case_metadata"].values()}), 32
        )

        scenario_rows: dict[str, list[dict[str, object]]] = {}
        groups: dict[str, set[tuple[str, str]]] = {}
        abstained = 0
        abstained_groups: set[str] = set()
        all_message_ids: set[str] = set()
        all_entity_ids: set[str] = set()
        for case in cases:
            _validate_case(case)
            self.assertTrue(case["case_id"].startswith("TRAIN-CASE-"))
            self.assertTrue(case["scenario_id"].startswith("TRAIN-SCENARIO-"))
            scenario_rows.setdefault(case["scenario_id"], []).append(case)
            metadata = payload["case_metadata"][case["case_id"]]
            self.assertEqual(set(metadata), {"scenario_group", "template_group", "entity_ids"})
            self.assertTrue(metadata["scenario_group"].startswith("TRAIN-SCENARIO-GROUP-"))
            self.assertTrue(metadata["template_group"].startswith("TRAIN-TEMPLATE-GROUP-"))
            groups.setdefault(case["scenario_id"], set()).add(
                (metadata["scenario_group"], metadata["template_group"])
            )
            self.assertTrue(metadata["entity_ids"])
            for entity_id in metadata["entity_ids"]:
                self.assertTrue(entity_id.startswith("TRAIN-ENTITY-"))
                self.assertNotIn(entity_id, all_entity_ids)
                all_entity_ids.add(entity_id)
            for message in case["messages"]:
                self.assertEqual(
                    set(message),
                    {"session_id", "message_id", "role", "text", "recorded_at", "content_kind"},
                )
                self.assertTrue(message["session_id"].startswith("TRAIN-SESSION-"))
                self.assertTrue(message["message_id"].startswith("TRAIN-MSG-"))
                self.assertNotIn(message["message_id"], all_message_ids)
                all_message_ids.add(message["message_id"])
            if not case["expected"]:
                abstained += 1
                abstained_groups.add(metadata["scenario_group"])
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
                    self.assertEqual(message["text"].count(span["quote"]), 1)

        self.assertEqual(abstained, 128)
        self.assertEqual(len(abstained_groups), 8)
        self.assertEqual(set(scenario_rows), {case["scenario_id"] for case in cases})
        self.assertTrue(all(len(rows) == 16 for rows in scenario_rows.values()))
        self.assertTrue(all(len(values) == 1 for values in groups.values()))
        for rows in scenario_rows.values():
            rendered_messages = {
                tuple(message["text"] for message in case["messages"]) for case in rows
            }
            self.assertEqual(len(rendered_messages), 16)
            label_signatures = {
                tuple(
                    (
                        claim["predicate"],
                        claim["speaker_role"],
                        claim["attribution"],
                        claim["modality"],
                        claim["polarity"],
                        claim["valid_from"] is None,
                        claim["valid_to"] is None,
                    )
                    for claim in case["expected"]
                )
                for case in rows
            }
            self.assertEqual(len(label_signatures), 1)
        date_map = dict(
            zip(seeds["value_banks"]["date_text"], seeds["value_banks"]["date_iso"], strict=True)
        )
        for case in cases:
            if "07-exact-event-date" in case["case_id"] or "25-exact-dated-plan" in case["case_id"]:
                text = " ".join(message["text"] for message in case["messages"])
                dates = [date_text for date_text in date_map if date_text in text]
                self.assertEqual(len(dates), 1)
                self.assertEqual(case["expected"][0]["valid_from"], date_map[dates[0]])
                self.assertEqual(case["expected"][0]["valid_to"], date_map[dates[0]])
            elif any(
                family in case["case_id"]
                for family in (
                    "08-imprecise-event-date",
                    "16-correction-preserves-history",
                    "32-preference-update",
                )
            ):
                self.assertTrue(
                    all(
                        claim["valid_from"] is None and claim["valid_to"] is None
                        for claim in case["expected"]
                    )
                )
        unicode_case = next(case for case in cases if "06-completed-action" in case["case_id"])
        unicode_span = unicode_case["expected"][0]["evidence"][0]
        self.assertEqual(unicode_span["char_start"], len("🪷 Note: "))

    def test_builder_regenerates_identical_train_bytes_from_fixed_seed(self) -> None:
        payload = json.loads(SEEDS.read_text(encoding="utf-8"))
        build = runpy.run_path(str(BUILDER))["build"]
        first = json.dumps(build(payload), ensure_ascii=False, indent=2) + "\n"
        second = json.dumps(build(payload), ensure_ascii=False, indent=2) + "\n"
        self.assertEqual(first, second)
        self.assertEqual(first, TRAIN.read_text(encoding="utf-8"))

    def test_reported_cause_preserves_the_maintenance_log_as_source(self) -> None:
        payload = json.loads(TRAIN.read_text(encoding="utf-8"))
        case = next(case for case in payload["cases"] if "30-reported-cause" in case["case_id"])
        claim = case["expected"][0]
        self.assertEqual(claim["predicate"], "REPORTS")
        self.assertEqual(claim["subject"], "the maintenance log")
        self.assertIn("was caused by", claim["object"])
        self.assertEqual(claim["attribution"], "third_party_report")
        self.assertEqual(claim["modality"], "reported")

    def test_cross_session_ownership_resolves_the_instrument_and_cites_both_sessions(self) -> None:
        payload = json.loads(TRAIN.read_text(encoding="utf-8"))
        cases = [case for case in payload["cases"] if "31-cross-session-" in case["case_id"]]
        self.assertEqual(len(cases), 16)
        for case in cases:
            first, second = case["messages"]
            instrument = re.fullmatch(r"I ordered (.+)\.", first["text"])
            self.assertIsNotNone(instrument)
            assert instrument is not None
            self.assertEqual(second["text"], "It belongs to me.")
            self.assertNotEqual(first["session_id"], second["session_id"])

            claim = case["expected"][0]
            self.assertEqual(claim["predicate"], "HAS")
            self.assertEqual(claim["subject"], "user")
            self.assertEqual(claim["object"], instrument.group(1))
            self.assertEqual(
                [span["message_id"] for span in claim["evidence"]],
                [first["message_id"], second["message_id"]],
            )
            self.assertEqual(
                [span["quote"] for span in claim["evidence"]],
                [first["text"], second["text"]],
            )


if __name__ == "__main__":
    unittest.main()
