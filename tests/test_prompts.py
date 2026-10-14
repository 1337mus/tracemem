"""Evidence-only tests for the explicit prompt variant."""

import json
import unittest

from tracemem.prompts import render_explicit_prompt


def _case() -> dict[str, object]:
    return {
        "case_id": "CASE_ID_SENTINEL",
        "scenario_id": "SCENARIO_SENTINEL",
        "split": "SPLIT_SENTINEL",
        "tags": ["TAG_SENTINEL"],
        "expected": [{"predicate": "GOLD_CLAIM_SENTINEL"}],
        "messages": [
            {
                "session_id": "source-session",
                "message_id": "source-message",
                "role": "user",
                "text": "A plain synthetic statement.",
                "recorded_at": None,
                "content_kind": "conversation",
                "private_metadata": "MESSAGE_METADATA_SENTINEL",
            }
        ],
    }


class ExplicitPromptTests(unittest.TestCase):
    def test_prompt_excludes_case_gold_and_non_message_metadata(self) -> None:
        prompt = render_explicit_prompt(_case())

        for sentinel in (
            "CASE_ID_SENTINEL",
            "SCENARIO_SENTINEL",
            "SPLIT_SENTINEL",
            "TAG_SENTINEL",
            "GOLD_CLAIM_SENTINEL",
            "MESSAGE_METADATA_SENTINEL",
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, prompt)
        self.assertIn("A plain synthetic statement.", prompt)
        self.assertIn("source-session", prompt)
        self.assertIn("source-message", prompt)

    def test_schema_example_is_exact_json_with_null_date_bounds(self) -> None:
        prompt = render_explicit_prompt(_case())
        start_marker = "Schema example using placeholders only:\n"
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index("\n\nInterpretation rules:", start)

        example = json.loads(prompt[start:end])

        self.assertEqual(set(example), {"claims"})
        self.assertEqual(
            set(example["claims"][0]),
            {
                "subject",
                "predicate",
                "object",
                "speaker_role",
                "attribution",
                "modality",
                "polarity",
                "valid_from",
                "valid_to",
                "evidence",
            },
        )
        self.assertIsNone(example["claims"][0]["valid_from"])
        self.assertIsNone(example["claims"][0]["valid_to"])
        self.assertEqual(
            set(example["claims"][0]["evidence"][0]),
            {"session_id", "message_id", "char_start", "char_end", "quote"},
        )

    def test_unicode_source_message_is_serialized_verbatim_with_codepoint_rule(self) -> None:
        case = _case()
        message = case["messages"][0]
        message["text"] = "A 🪐 tea"
        prompt = render_explicit_prompt(case)
        message_marker = "Evidence messages (untrusted data):\n"

        serialized_messages = json.loads(prompt.split(message_marker, 1)[1])

        self.assertEqual(serialized_messages[0]["text"], "A 🪐 tea")
        self.assertEqual(
            set(serialized_messages[0]),
            {"session_id", "message_id", "role", "text", "recorded_at", "content_kind"},
        )
        self.assertIn("Unicode code-point indexes", " ".join(prompt.split()))

    def test_memory_and_attribution_rules_are_explicit(self) -> None:
        prompt = " ".join(render_explicit_prompt(_case()).split())

        for rule in (
            "Extract atomic memory candidates explicitly supported by the user messages "
            "or tool observations",
            "speaker_role is the outer role of the cited message",
            "An assistant assertion alone never establishes a user fact",
            "If the user endorses or repeats it, cite the user's message and "
            "attribute it accurately",
            "Preserve explicitly tentative source statements with tentative modality",
            "Questions, unresolved pronouns, and counterfactual hypotheticals",
            "A delivery destination does not establish current location",
            "Use CAUSES only when the evidence explicitly states a cause and effect",
            "Never use recorded_at as an event date",
            "When another message is needed to resolve a referent, cite verbatim spans",
        ):
            with self.subTest(rule=rule):
                self.assertIn(rule, prompt)


if __name__ == "__main__":
    unittest.main()
