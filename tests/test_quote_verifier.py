from __future__ import annotations

import json
import unittest
from typing import cast

import experiments.quote_verifier as verifier


def user_message(
    text: str = "In the mock exercise, I might keep a cobalt diary.",
) -> dict[str, str]:
    return {
        "session_id": "fiction-session-1",
        "message_id": "fiction-user-1",
        "role": "user",
        "text": text,
        "recorded_at": "2026-10-01T12:00:00Z",
        "content_kind": "conversation",
    }


def valid_candidate(message: dict[str, str] | None = None) -> dict[str, str | None]:
    source = message or user_message()
    return {
        "claim": "The user might keep a cobalt diary in the exercise.",
        "message_id": source["message_id"],
        "quote": source["text"],
        "valid_from": None,
        "valid_to": None,
    }


class RenderVerifierPromptTests(unittest.TestCase):
    def test_prompt_separates_support_from_eligibility_without_salience_gate(self) -> None:
        messages = [user_message()]
        candidates = [valid_candidate(messages[0])]

        prompt = verifier.render_verifier_prompt(messages, candidates)

        self.assertIn("source support and durable personal-memory eligibility separately", prompt)
        self.assertIn("fiction and exercise material", prompt)
        self.assertIn("commands", prompt)
        self.assertIn("conditional", prompt)
        self.assertIn("reported", prompt)
        self.assertIn("one-time", prompt)
        self.assertIn("Judge the claim statement itself", prompt)
        self.assertIn("cannot supply a missing reporter, condition, or uncertainty", prompt)
        self.assertIn("including analysis/request metadata", prompt)
        self.assertIn("ineligible even when accurately quoted", prompt)
        self.assertIn("Do not rank memories by salience", prompt)
        self.assertIn("Do not rewrite or repair", prompt)
        self.assertIn("support must be exactly one of supported, unsupported, or uncertain", prompt)
        self.assertIn("reason must be non-empty and at most 500 characters", prompt)
        self.assertIn('"support":"supported","eligible":true', prompt)
        self.assertNotIn("gold label", prompt.lower())
        self.assertNotIn("reviewer label", prompt.lower())
        encoded_messages = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        encoded_candidates = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
        self.assertTrue(
            prompt.endswith(f"Messages: {encoded_messages}\nCandidates: {encoded_candidates}")
        )
        empty_prompt = verifier.render_verifier_prompt(messages, [])
        self.assertTrue(empty_prompt.endswith(f"Messages: {encoded_messages}\nCandidates: []"))

        quoted_document = dict(messages[0], content_kind="quoted_document")
        with self.assertRaisesRegex(ValueError, "must be a user conversation message"):
            verifier.render_verifier_prompt([quoted_document], candidates)

    def test_prompt_rejects_more_than_64_candidates(self) -> None:
        messages = [user_message()]
        candidates = [valid_candidate(messages[0]) for _ in range(65)]

        with self.assertRaises(ValueError):
            verifier.render_verifier_prompt(messages, candidates)


class ParseVerdictsTests(unittest.TestCase):
    def test_parser_enforces_strict_json_types_and_complete_index_bijection(self) -> None:
        def row(
            index: object = 0,
            *,
            eligible: object = True,
            support: object = "supported",
            reason: str = "evidence",
        ) -> dict[str, object]:
            return {
                "claim_index": index,
                "support": support,
                "eligible": eligible,
                "reason": reason,
            }

        valid = json.dumps({"verdicts": [row(1), row(0, support="uncertain")]})
        parsed = verifier.parse_verdicts(valid, 2)
        self.assertEqual([verdict["claim_index"] for verdict in parsed], [0, 1])
        self.assertEqual(verifier.parse_verdicts('{"verdicts":[]}', 0), [])

        rejected = [
            ('{"verdicts":[],"verdicts":[]}', 0),
            ('{"verdicts":NaN}', 0),
            ('{"verdicts":[]} trailing', 0),
            ('```json\n{"verdicts":[]}\n```', 0),
            (json.dumps({"verdicts": [row(0)]}), 2),
            (json.dumps({"verdicts": [row(2), row(0)]}), 2),
            (json.dumps({"verdicts": [row(0), row(0)]}), 2),
            (json.dumps({"verdicts": [row(True)]}), 1),
            (json.dumps({"verdicts": [row(0, eligible=1)]}), 1),
            (json.dumps({"verdicts": [row(0, support="maybe")]}), 1),
            (json.dumps({"verdicts": [row(0, reason="  ")]}), 1),
            (json.dumps({"verdicts": [row(0, reason="x" * 501)]}), 1),
            (json.dumps({"verdicts": [dict(row(0), extra=True)]}), 1),
            (" " * 100_001, 0),
        ]
        for raw_text, candidate_count in rejected:
            with self.subTest(candidate_count=candidate_count, raw_length=len(raw_text)):
                with self.assertRaises(ValueError):
                    verifier.parse_verdicts(raw_text, candidate_count)

        for invalid_count in (True, -1, 65, "2"):
            with self.subTest(candidate_count=invalid_count):
                with self.assertRaises(ValueError):
                    verifier.parse_verdicts('{"verdicts":[]}', cast(int, invalid_count))


class SelectCandidatesTests(unittest.TestCase):
    def test_selection_keeps_only_supported_eligible_originals_in_input_order(self) -> None:
        candidates = [
            {
                "claim": f"fictional statement {index}",
                "message_id": f"fictional-message-{index}",
                "quote": f"fictional quote {index}",
                "valid_from": None,
                "valid_to": None,
            }
            for index in range(5)
        ]
        verdicts = [
            {"claim_index": 4, "support": "supported", "eligible": True, "reason": "e"},
            {"claim_index": 3, "support": "uncertain", "eligible": True, "reason": "d"},
            {"claim_index": 2, "support": "supported", "eligible": False, "reason": "c"},
            {"claim_index": 1, "support": "unsupported", "eligible": True, "reason": "b"},
            {"claim_index": 0, "support": "supported", "eligible": True, "reason": "a"},
        ]

        selected = verifier.select_supported_candidates(candidates, verdicts)

        self.assertEqual(selected, [candidates[0], candidates[4]])
        self.assertIs(selected[0], candidates[0])
        self.assertIs(selected[1], candidates[4])
        with self.assertRaises(ValueError):
            verifier.select_supported_candidates(candidates, verdicts[:-1])
        with self.assertRaises(ValueError):
            verifier.select_supported_candidates(
                [dict(candidates[0], extra="field")], [verdicts[-1]]
            )
        unsafe = [dict(row) for row in verdicts]
        unsafe[-1]["eligible"] = 1
        with self.assertRaises(ValueError):
            verifier.select_supported_candidates(candidates, unsafe)
        with self.assertRaises(ValueError):
            verifier.select_supported_candidates(candidates * 13, [])


if __name__ == "__main__":
    unittest.main()
