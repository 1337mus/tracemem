from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from training import verifier_data


def invented_dataset() -> dict[str, Any]:
    messages = [
        {
            "session_id": "synthetic-session-train-a",
            "message_id": "synthetic-message-train-a",
            "role": "user",
            "text": "I might store my fictional sketchbook Birch in room Cedar.",
            "recorded_at": "2026-02-03T10:00:00Z",
            "content_kind": "conversation",
        },
        {
            "session_id": "synthetic-session-train-a",
            "message_id": "synthetic-message-train-b",
            "role": "assistant",
            "text": "That sounds like a useful fictional detail.",
            "recorded_at": "2026-02-03T10:01:00Z",
            "content_kind": "conversation",
        },
    ]
    citation = {
        "message_id": "synthetic-message-train-a",
        "quote": "I might store my fictional sketchbook Birch in room Cedar.",
        "valid_from": None,
        "valid_to": None,
    }
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "scope_policy": "personal-memory-research-v1",
        "cases": [
            {
                "case_id": "synthetic-case-train-a",
                "split": "train",
                "scenario_group": "synthetic-scenario-train-a",
                "template_group": "synthetic-template-train-a",
                "entity_ids": ["synthetic-entity-train-a"],
                "phenomenon": "condition",
                "messages": messages,
                "candidates": [
                    {
                        "candidate_id": "synthetic-candidate-train-a",
                        "claim": (
                            "The user might store the fictional sketchbook Birch in room Cedar."
                        ),
                        **citation,
                        "support": "supported",
                        "eligibility": "eligible",
                        "support_reason": "The claim preserves the user's stated possibility.",
                        "eligibility_reason": "This is a fictional example of a personal fact.",
                    },
                    {
                        "candidate_id": "synthetic-candidate-train-b",
                        "claim": (
                            "The user will store the fictional sketchbook Birch in room Cedar."
                        ),
                        **citation,
                        "support": "uncertain",
                        "eligibility": "uncertain",
                        "support_reason": "The candidate changes the modality of the source.",
                        "eligibility_reason": "The example's scope remains uncertain.",
                    },
                ],
            }
        ],
    }


def case_for_split(
    split: str, suffix: str, *, distinct_conversation: bool = True
) -> dict[str, Any]:
    case: dict[str, Any] = copy.deepcopy(invented_dataset()["cases"][0])
    old_quote = case["messages"][0]["text"]
    new_message_id = f"synthetic-message-{suffix}-1"
    case.update(
        {
            "case_id": f"synthetic-case-{suffix}",
            "split": split,
            "scenario_group": f"synthetic-scenario-{suffix}",
            "template_group": f"synthetic-template-{suffix}",
            "entity_ids": [f"synthetic-entity-{suffix}"],
        }
    )
    for index, message in enumerate(case["messages"]):
        message["session_id"] = f"synthetic-session-{suffix}"
        message["message_id"] = f"synthetic-message-{suffix}-{index + 1}"
    if distinct_conversation:
        new_quote = f"I might keep my fictional atlas {suffix} in room Juniper."
        case["messages"][0]["text"] = new_quote
        case["messages"][1]["text"] = f"That is another invented setting {suffix}."
        for index, candidate in enumerate(case["candidates"]):
            candidate["claim"] = (
                f"The user {'might' if index == 0 else 'will'} keep the fictional atlas "
                f"{suffix} in room Juniper."
            )
            candidate["quote"] = new_quote
    else:
        new_quote = old_quote
    for candidate in case["candidates"]:
        candidate["candidate_id"] = f"synthetic-candidate-{suffix}-{candidate['candidate_id'][-1]}"
        candidate["message_id"] = new_message_id
        if distinct_conversation:
            candidate["quote"] = new_quote
    return case


class VerifierDatasetValidationTests(unittest.TestCase):
    def test_accepts_authored_pair_with_independent_uncertain_labels(self) -> None:
        payload = invented_dataset()

        self.assertEqual(verifier_data.validate_verifier_dataset(payload), payload["cases"])

    def test_rejects_boolean_version_and_human_review_claim(self) -> None:
        for field, value in (("version", True), ("annotation_status", "human_reviewed")):
            with self.subTest(field=field):
                payload = invented_dataset()
                payload[field] = value
                with self.assertRaises(ValueError):
                    verifier_data.validate_verifier_dataset(payload)

    def test_rejects_unexpected_fields_invalid_metadata_and_duplicate_entity_ids(self) -> None:
        invalid_cases = [
            ("root extra field", "root", "extra", "private"),
            ("case extra field", "case", "extra", "private"),
            ("candidate extra field", "candidate", "extra", "private"),
            ("invalid split", "case", "split", "production"),
            ("invalid phenomenon", "case", "phenomenon", "ownership"),
            ("duplicate entities", "case", "entity_ids", ["same", "same"]),
        ]
        for label, location, field, value in invalid_cases:
            with self.subTest(label=label):
                payload = invented_dataset()
                if location == "root":
                    payload[field] = value
                elif location == "case":
                    payload["cases"][0][field] = value
                else:
                    payload["cases"][0]["candidates"][0][field] = value
                with self.assertRaises(ValueError):
                    verifier_data.validate_verifier_dataset(payload)

    def test_rejects_invalid_annotation_labels_and_long_or_empty_reasons(self) -> None:
        invalid_candidates = [
            ("support", "positive"),
            ("eligibility", True),
            ("support_reason", "  "),
            ("support_reason", "r" * 501),
            ("eligibility_reason", "r" * 501),
        ]
        for field, value in invalid_candidates:
            with self.subTest(
                field=field, value_length=len(value) if isinstance(value, str) else 0
            ):
                payload = invented_dataset()
                payload["cases"][0]["candidates"][0][field] = value
                with self.assertRaises(ValueError):
                    verifier_data.validate_verifier_dataset(payload)

    def test_enforces_v3_message_count_bounds(self) -> None:
        payload = invented_dataset()
        payload["cases"][0]["messages"] = []
        with self.assertRaisesRegex(ValueError, "one to twelve messages"):
            verifier_data.validate_verifier_dataset(payload)

        payload = invented_dataset()
        messages = payload["cases"][0]["messages"]
        for index in range(3, 14):
            messages.append(
                {
                    "session_id": "synthetic-session-train-a",
                    "message_id": f"synthetic-extra-message-{index}",
                    "role": "assistant",
                    "text": f"Another invented line {index}.",
                    "recorded_at": f"2026-02-03T10:{index:02d}:00Z",
                    "content_kind": "conversation",
                }
            )
        with self.assertRaisesRegex(ValueError, "one to twelve messages"):
            verifier_data.validate_verifier_dataset(payload)

    def test_case_and_candidate_ids_are_unique_globally(self) -> None:
        payload = invented_dataset()
        second = case_for_split("dev", "dev-a")
        second["case_id"] = payload["cases"][0]["case_id"]
        payload["cases"].append(second)
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            verifier_data.validate_verifier_dataset(payload)

        payload = invented_dataset()
        second = case_for_split("dev", "dev-a")
        second["candidates"][0]["candidate_id"] = payload["cases"][0]["candidates"][0][
            "candidate_id"
        ]
        payload["cases"].append(second)
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            verifier_data.validate_verifier_dataset(payload)

    def test_reuses_long_quote_validation_for_exact_source_citations(self) -> None:
        payload = invented_dataset()
        payload["cases"][0]["messages"][0]["text"] = "fictional and fictional"
        payload["cases"][0]["candidates"][0]["quote"] = "fictional"
        payload["cases"][0]["candidates"][1]["quote"] = "fictional"

        with self.assertRaisesRegex(ValueError, "exactly once"):
            verifier_data.validate_verifier_dataset(payload)

    def test_requires_two_candidates_with_same_citation_and_distinct_claims(self) -> None:
        payload = invented_dataset()
        payload["cases"][0]["candidates"][1]["valid_from"] = "2026-02-03"
        with self.assertRaisesRegex(ValueError, "identical citations"):
            verifier_data.validate_verifier_dataset(payload)

        payload = invented_dataset()
        payload["cases"][0]["candidates"][1]["claim"] = payload["cases"][0]["candidates"][0][
            "claim"
        ]
        with self.assertRaisesRegex(ValueError, "distinct claims"):
            verifier_data.validate_verifier_dataset(payload)

        payload = invented_dataset()
        payload["cases"][0]["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "exactly two candidates"):
            verifier_data.validate_verifier_dataset(payload)

    def test_rejects_split_overlap_for_scenarios_templates_and_entities(self) -> None:
        for field in ("scenario_group", "template_group", "entity_ids"):
            with self.subTest(field=field):
                payload = invented_dataset()
                second = case_for_split("dev", "dev-a")
                second[field] = copy.deepcopy(payload["cases"][0][field])
                payload["cases"].append(second)
                with self.assertRaisesRegex(ValueError, "overlaps splits"):
                    verifier_data.validate_verifier_dataset(payload)

    def test_rejects_duplicate_ids_and_normalized_conversations_across_splits(self) -> None:
        payload = invented_dataset()
        second = case_for_split("dev", "dev-a")
        second["messages"][0]["message_id"] = payload["cases"][0]["messages"][0]["message_id"]
        for candidate in second["candidates"]:
            candidate["message_id"] = payload["cases"][0]["messages"][0]["message_id"]
        payload["cases"].append(second)
        with self.assertRaisesRegex(ValueError, "duplicate message_id"):
            verifier_data.validate_verifier_dataset(payload)

        payload = invented_dataset()
        payload["cases"].append(case_for_split("dev", "dev-a", distinct_conversation=False))
        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            verifier_data.validate_verifier_dataset(payload)

    def test_candidate_prompt_strips_annotations_and_identity_metadata(self) -> None:
        case = invented_dataset()["cases"][0]
        prompt = verifier_data.render_candidate_prompt(case, "synthetic-candidate-train-a")

        self.assertIn("personal-memory-research-v1", prompt)
        self.assertIn("exactly two keys, support and eligibility", prompt)
        self.assertIn("Do not infer claim dates from recorded_at", prompt)
        self.assertIn('"message_id":"synthetic-message-train-a"', prompt)
        self.assertIn("I might store my fictional sketchbook Birch in room Cedar.", prompt)
        for hidden_value in (
            "synthetic-case-train-a",
            "synthetic-scenario-train-a",
            "synthetic-template-train-a",
            "synthetic-entity-train-a",
            "synthetic-candidate-train-a",
            "synthetic-session-train-a",
            "support_reason",
            "eligibility_reason",
            "The claim preserves the user's stated possibility.",
        ):
            with self.subTest(hidden_value=hidden_value):
                self.assertNotIn(hidden_value, prompt)

    def test_training_rows_emit_only_train_prompts_and_two_axis_json_targets(self) -> None:
        payload = invented_dataset()
        payload["cases"][0]["candidates"][0]["eligibility"] = "ineligible"
        payload["cases"][0]["candidates"][1]["support"] = "unsupported"
        payload["cases"][0]["candidates"][1]["eligibility"] = "eligible"
        payload["cases"].append(case_for_split("dev", "dev-a"))

        rows = verifier_data.render_training_rows(payload, "train")

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [json.loads(row["completion"]) for row in rows],
            [
                {"support": "supported", "eligibility": "ineligible"},
                {"support": "unsupported", "eligibility": "eligible"},
            ],
        )
        for row in rows:
            self.assertEqual(set(row), {"prompt", "completion"})
            completion = json.loads(row["completion"])
            self.assertEqual(set(completion), {"support", "eligibility"})
            self.assertIn(completion["support"], {"supported", "unsupported", "uncertain"})
            self.assertIn(completion["eligibility"], {"eligible", "ineligible", "uncertain"})
            self.assertNotIn("synthetic-message-dev-a", row["prompt"])

        with self.assertRaisesRegex(ValueError, "only be rendered for the train split"):
            verifier_data.render_training_rows(payload, "dev")

    def test_candidate_renderer_validates_the_supplied_case(self) -> None:
        case = invented_dataset()["cases"][0]
        case["unexpected_annotation"] = "must not reach model"

        with self.assertRaisesRegex(ValueError, "missing or unexpected fields"):
            verifier_data.render_candidate_prompt(case, "synthetic-candidate-train-a")


if __name__ == "__main__":
    unittest.main()
