from __future__ import annotations

import copy
import json
import re
import unittest
from collections import Counter
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path

from training.build_quote_data import build, build_all
from training.quote_data import render_quote_prompt, validate_claims, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
QUOTE_DATA_DIR = ROOT / "datasets" / "sft-quote-v2"


def minimal_dataset() -> dict[str, object]:
    message = {
        "session_id": "session-1",
        "message_id": "message-1",
        "role": "user",
        "text": "I might try the amber notebook.",
        "recorded_at": "2026-01-02T10:00:00Z",
        "content_kind": "conversation",
    }
    case = {
        "case_id": "TRAIN-CASE-001",
        "split": "train",
        "scenario_group": "TRAIN-SCENARIO-001",
        "template_group": "TRAIN-TEMPLATE-001",
        "entity_ids": ["TRAIN-ENTITY-001"],
        "tags": ["tentative"],
        "messages": [message],
        "expected": [],
    }
    return {
        "version": 2,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": [case],
    }


class QuoteDatasetValidationTests(unittest.TestCase):
    def test_validate_dataset_accepts_native_test_split(self) -> None:
        payload = minimal_dataset()
        payload["cases"][0]["split"] = "test"

        cases = validate_dataset(payload, "test")

        self.assertEqual(cases, payload["cases"])

    def test_validate_dataset_accepts_the_authored_v2_case_contract(self) -> None:
        payload = minimal_dataset()
        cases = validate_dataset(payload, "train")
        self.assertEqual(cases, payload["cases"])

        malformed = minimal_dataset()
        malformed["version"] = True
        with self.assertRaisesRegex(ValueError, "version"):
            validate_dataset(malformed, "train")

        with self.assertRaisesRegex(ValueError, "split"):
            validate_dataset(payload, "dev")

        unsupported_quote = minimal_dataset()
        unsupported_quote["cases"][0]["expected"] = [
            {
                "claim": "The user might try a notebook.",
                "message_id": "message-1",
                "quote": "amber page",
                "valid_from": None,
                "valid_to": None,
            }
        ]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_dataset(unsupported_quote, "train")

    def test_dataset_rejects_duplicate_conversations_but_allows_reused_filler(self) -> None:
        payload = minimal_dataset()
        first = payload["cases"][0]
        first["messages"].append(
            {
                "session_id": "session-1",
                "message_id": "assistant-1",
                "role": "assistant",
                "text": "I can help check the next step.",
                "recorded_at": "2026-01-02T10:01:00Z",
                "content_kind": "conversation",
            }
        )
        second = copy.deepcopy(first)
        second["case_id"] = "TRAIN-CASE-002"
        second["scenario_group"] = "TRAIN-SCENARIO-002"
        second["template_group"] = "TRAIN-TEMPLATE-002"
        second["entity_ids"] = ["TRAIN-ENTITY-002"]
        second["messages"][0]["session_id"] = "session-2"
        second["messages"][0]["message_id"] = "message-2"
        second["messages"][0]["text"] = "Would that notebook fit in this drawer?"
        second["messages"][1]["session_id"] = "session-2"
        second["messages"][1]["message_id"] = "assistant-2"
        payload["cases"] = [first, second]
        self.assertEqual(validate_dataset(payload, "train"), [first, second])

        duplicate = copy.deepcopy(second)
        duplicate["case_id"] = "TRAIN-CASE-003"
        duplicate["entity_ids"] = ["TRAIN-ENTITY-003"]
        duplicate["messages"][0]["message_id"] = "message-3"
        duplicate["messages"][1]["message_id"] = "assistant-3"
        payload["cases"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            validate_dataset(payload, "train")

    def test_validate_claims_requires_unique_user_quote_and_enriches_citation(self) -> None:
        messages = [
            {
                "session_id": "session-7",
                "message_id": "message-7",
                "role": "user",
                "text": "Nora might try the amber notebook before 2026-03-02.",
                "recorded_at": "2026-01-02T10:00:00Z",
                "content_kind": "conversation",
            }
        ]
        parsed = {
            "claims": [
                {
                    "claim": "The user might try the amber notebook.",
                    "message_id": "message-7",
                    "quote": "Nora might try the amber notebook before 2026-03-02.",
                    "valid_from": "2026-03-02",
                    "valid_to": None,
                }
            ]
        }
        claims = validate_claims(parsed, messages)
        self.assertEqual(
            claims[0],
            {
                **parsed["claims"][0],
                "session_id": "session-7",
                "char_start": 0,
                "char_end": len(messages[0]["text"]),
                "speaker_role": "user",
            },
        )

        repeated = [dict(messages[0], text="notebook near notebook")]
        duplicate_quote = {
            "claims": [
                {
                    "claim": "The user mentioned a notebook.",
                    "message_id": "message-7",
                    "quote": "notebook",
                    "valid_from": None,
                    "valid_to": None,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_claims(duplicate_quote, repeated)

    def test_render_quote_prompt_is_split_independent_and_omits_labels(self) -> None:
        case = minimal_dataset()["cases"][0]
        prompt = render_quote_prompt(case)
        dev_case = dict(case, split="dev", expected=[{"private": "gold label"}])
        self.assertEqual(prompt, render_quote_prompt(dev_case))
        self.assertIn("do not infer dates from recorded_at", prompt)
        self.assertIn('"message_id":"message-1"', prompt)
        self.assertNotIn("gold label", prompt)

    def test_builder_expands_authored_seed_into_the_v2_public_shape(self) -> None:
        seed = {
            "seed": 23,
            "split": "train",
            "variants": 1,
            "value_banks": {
                "person": ["Nora"],
                "product": ["MapleNote"],
                "object": ["amber notebook"],
                "place": ["room Alder"],
                "date_from": ["2025-04-01"],
                "date_to": ["2025-04-03"],
            },
            "families": [
                {
                    "family_id": "T01-choice",
                    "tags": ["decision"],
                    "messages": [
                        {
                            "ref": "u1",
                            "role": "user",
                            "session": 1,
                            "content_kind": "conversation",
                            "text": "I chose {product} for my notes.",
                        }
                    ],
                    "expected": [
                        {
                            "claim": "The user chose {product} for notes.",
                            "message_ref": "u1",
                            "quote": "I chose {product} for my notes.",
                            "valid_from": None,
                            "valid_to": None,
                        }
                    ],
                }
            ],
        }
        payload = build(seed, "train")
        self.assertEqual(
            set(payload), {"version", "license", "annotation_status", "provenance", "cases"}
        )
        self.assertEqual(
            payload["cases"][0]["expected"][0]["claim"],
            "The user chose MapleNote for notes.",
        )


class QuoteDatasetReleaseAcceptanceTests(unittest.TestCase):
    def test_checked_in_splits_have_expected_counts_and_disjoint_families(self) -> None:
        train = json.loads((QUOTE_DATA_DIR / "train.json").read_text(encoding="utf-8"))
        dev = json.loads((QUOTE_DATA_DIR / "dev.json").read_text(encoding="utf-8"))
        train_cases = validate_dataset(train, "train")
        dev_cases = validate_dataset(dev, "dev")

        self.assertEqual(len(train_cases), 768)
        self.assertEqual(len(dev_cases), 24)

        train_families = {case["scenario_group"] for case in train_cases}
        dev_families = {case["scenario_group"] for case in dev_cases}
        train_templates = {case["template_group"] for case in train_cases}
        dev_templates = {case["template_group"] for case in dev_cases}
        self.assertEqual(len(train_families), 48)
        self.assertEqual(len(dev_families), 24)
        self.assertEqual(len(train_templates), 48)
        self.assertEqual(len(dev_templates), 24)
        self.assertTrue(train_families.isdisjoint(dev_families))
        self.assertTrue(train_templates.isdisjoint(dev_templates))
        train_family_templates = {
            case["scenario_group"]: case["template_group"] for case in train_cases
        }
        dev_family_templates = {
            case["scenario_group"]: case["template_group"] for case in dev_cases
        }
        self.assertEqual(len(set(train_family_templates.values())), 48)
        self.assertEqual(len(set(dev_family_templates.values())), 24)

    def test_empty_target_counts_match_the_authored_abstention_mix(self) -> None:
        train = json.loads((QUOTE_DATA_DIR / "train.json").read_text(encoding="utf-8"))
        dev = json.loads((QUOTE_DATA_DIR / "dev.json").read_text(encoding="utf-8"))
        train_empty = sum(not case["expected"] for case in train["cases"])
        dev_empty = sum(not case["expected"] for case in dev["cases"])

        self.assertEqual(train_empty, 144)
        self.assertEqual(dev_empty, 6)
        self.assertEqual(Fraction(train_empty, len(train["cases"])), Fraction(3, 16))
        self.assertEqual(Fraction(dev_empty, len(dev["cases"])), Fraction(1, 4))
        train_variants = Counter(case["scenario_group"] for case in train["cases"])
        dev_variants = Counter(case["scenario_group"] for case in dev["cases"])
        self.assertEqual(set(train_variants.values()), {16})
        self.assertEqual(set(dev_variants.values()), {1})

    def test_checked_in_expansions_are_byte_identical_to_seed_replay(self) -> None:
        train_seeds = json.loads((QUOTE_DATA_DIR / "train-seeds.json").read_text(encoding="utf-8"))
        dev_seeds = json.loads((QUOTE_DATA_DIR / "dev-seeds.json").read_text(encoding="utf-8"))
        replay = build_all(train_seeds, dev_seeds)

        for split, payload in replay.items():
            expected = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            self.assertEqual((QUOTE_DATA_DIR / f"{split}.json").read_bytes(), expected.encode())

    def test_recorded_at_follows_every_literal_calendar_date(self) -> None:
        date_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
        for split in ("train", "dev"):
            payload = json.loads((QUOTE_DATA_DIR / f"{split}.json").read_text(encoding="utf-8"))
            for case in payload["cases"]:
                for message in case["messages"]:
                    recorded_at = datetime.fromisoformat(
                        message["recorded_at"].replace("Z", "+00:00")
                    )
                    authored_dates = [
                        date.fromisoformat(match.group())
                        for match in date_pattern.finditer(message["text"])
                    ]
                    if authored_dates:
                        self.assertGreater(
                            recorded_at.date(),
                            max(authored_dates),
                            f"{case['case_id']} recorded_at precedes an authored date",
                        )


if __name__ == "__main__":
    unittest.main()
