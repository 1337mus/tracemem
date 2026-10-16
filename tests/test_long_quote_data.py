from __future__ import annotations

import copy
import unittest
from typing import Any

from training import quote_data


def invented_dataset(message_count: int = 12) -> dict[str, Any]:
    messages = [
        {
            "session_id": f"invented-session-{index:02d}",
            "message_id": f"invented-message-{index:02d}",
            "role": "user",
            "text": f"I invented fictional detail {index} for this test.",
            "recorded_at": f"2026-01-02T10:{index:02d}:00Z",
            "content_kind": "conversation",
        }
        for index in range(1, message_count + 1)
    ]
    return {
        "version": 3,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": [
            {
                "case_id": "INVENTED-CASE-001",
                "split": "train",
                "scenario_group": "INVENTED-SCENARIO-001",
                "template_group": "INVENTED-TEMPLATE-001",
                "entity_ids": ["INVENTED-ENTITY-001"],
                "tags": ["synthetic"],
                "messages": messages,
                "expected": [
                    {
                        "claim": "The user invented fictional detail 12.",
                        "message_id": "invented-message-12",
                        "quote": "I invented fictional detail 12 for this test.",
                        "valid_from": None,
                        "valid_to": None,
                    }
                ],
            }
        ],
    }


class LongQuoteDatasetValidationTests(unittest.TestCase):
    def test_validate_long_dataset_accepts_twelve_invented_v3_messages(self) -> None:
        payload = invented_dataset()
        validator = getattr(quote_data, "validate_long_dataset", None)
        self.assertIsNotNone(validator, "v3 needs its own explicit validator")
        if validator is None:
            return

        self.assertEqual(validator(payload, "train"), payload["cases"])

    def test_v3_accepts_the_existing_one_message_minimum(self) -> None:
        payload = invented_dataset()
        case = payload["cases"][0]
        first_message = case["messages"][0]
        case["messages"] = [first_message]
        case["expected"][0].update(
            {
                "claim": "The user invented fictional detail 1.",
                "message_id": first_message["message_id"],
                "quote": first_message["text"],
            }
        )

        self.assertEqual(quote_data.validate_long_dataset(payload, "train"), payload["cases"])

    def test_v2_validator_keeps_its_six_message_limit(self) -> None:
        payload = invented_dataset(message_count=7)
        payload["version"] = 2

        with self.assertRaises(ValueError) as raised:
            quote_data.validate_dataset(payload, "train")
        self.assertEqual(
            str(raised.exception),
            "case INVENTED-CASE-001 must have one to six messages",
        )

    def test_v3_validator_rejects_thirteen_messages(self) -> None:
        payload = invented_dataset(message_count=13)

        with self.assertRaisesRegex(ValueError, "one to twelve messages"):
            quote_data.validate_long_dataset(payload, "train")

    def test_v3_rejects_an_empty_conversation(self) -> None:
        payload = invented_dataset(message_count=0)

        with self.assertRaisesRegex(ValueError, "one to twelve messages"):
            quote_data.validate_long_dataset(payload, "train")

    def test_versioned_validators_reject_the_other_contract_version(self) -> None:
        payload_v2 = invented_dataset()
        payload_v2["version"] = 2
        with self.assertRaisesRegex(ValueError, "version must be integer 3"):
            quote_data.validate_long_dataset(payload_v2, "train")

        payload_v3 = invented_dataset()
        with self.assertRaisesRegex(ValueError, "version must be integer 2"):
            quote_data.validate_dataset(payload_v3, "train")

    def test_both_validators_reject_boolean_versions(self) -> None:
        for validator, expected_version in (
            (quote_data.validate_dataset, 2),
            (quote_data.validate_long_dataset, 3),
        ):
            with self.subTest(expected_version=expected_version):
                payload = invented_dataset()
                payload["version"] = True

                with self.assertRaisesRegex(
                    ValueError, f"version must be integer {expected_version}"
                ):
                    validator(payload, "train")

    def test_v3_rejects_assistant_tool_and_quoted_document_citations(self) -> None:
        invalid_sources = (
            ("assistant", "conversation"),
            ("tool", "conversation"),
            ("user", "quoted_document"),
        )
        for role, content_kind in invalid_sources:
            with self.subTest(role=role, content_kind=content_kind):
                payload = invented_dataset()
                source = payload["cases"][0]["messages"][11]
                source["role"] = role
                source["content_kind"] = content_kind

                with self.assertRaisesRegex(ValueError, "must be a user conversation message"):
                    quote_data.validate_long_dataset(payload, "train")

    def test_v3_uses_shared_duplicate_conversation_check(self) -> None:
        payload = invented_dataset()
        duplicate = copy.deepcopy(payload["cases"][0])
        duplicate["case_id"] = "INVENTED-CASE-002"
        duplicate["scenario_group"] = "INVENTED-SCENARIO-002"
        duplicate["template_group"] = "INVENTED-TEMPLATE-002"
        duplicate["entity_ids"] = ["INVENTED-ENTITY-002"]
        for index, message in enumerate(duplicate["messages"], start=1):
            message["session_id"] = f"duplicate-session-{index:02d}"
            message["message_id"] = f"duplicate-message-{index:02d}"
        duplicate["expected"][0]["message_id"] = "duplicate-message-12"
        payload["cases"].append(duplicate)

        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            quote_data.validate_long_dataset(payload, "train")

    def test_v3_uses_shared_validity_date_order_check(self) -> None:
        payload = invented_dataset()
        claim = payload["cases"][0]["expected"][0]
        claim["valid_from"] = "2026-01-03"
        claim["valid_to"] = "2026-01-02"

        with self.assertRaisesRegex(ValueError, "valid_from must not be after valid_to"):
            quote_data.validate_long_dataset(payload, "train")

    def test_v3_requires_a_unique_exact_quote(self) -> None:
        payload = invented_dataset()
        message = payload["cases"][0]["messages"][11]
        message["text"] = (
            "I invented fictional detail 12 for this test; fictional detail 12 repeats."
        )
        payload["cases"][0]["expected"][0]["quote"] = "fictional detail 12"

        with self.assertRaisesRegex(ValueError, "quote must occur exactly once"):
            quote_data.validate_long_dataset(payload, "train")

    def test_v3_requires_explicit_dates_to_appear_in_the_quote(self) -> None:
        payload = invented_dataset()
        payload["cases"][0]["expected"][0]["valid_from"] = "2026-01-03"

        with self.assertRaisesRegex(ValueError, "valid_from must appear literally in quote"):
            quote_data.validate_long_dataset(payload, "train")

    def test_v3_requires_timezone_aware_source_dates(self) -> None:
        payload = invented_dataset()
        payload["cases"][0]["messages"][11]["recorded_at"] = "2026-01-02T10:12:00"

        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            quote_data.validate_long_dataset(payload, "train")


if __name__ == "__main__":
    unittest.main()
