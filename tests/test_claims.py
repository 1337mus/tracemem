import unittest
from dataclasses import FrozenInstanceError

from tracemem.claims import Claim, EvidenceSpan, parse_candidates, verify_candidate
from tracemem.schema import EvidenceRecord


def _valid_payload() -> dict[str, object]:
    return {
        "claims": [
            {
                "subject": "the user",
                "predicate": "PREFERS",
                "object": "quiet workspaces",
                "speaker_role": "user",
                "attribution": "direct_user_statement",
                "modality": "preference",
                "polarity": "positive",
                "valid_from": "2026-10-01",
                "valid_to": None,
                "evidence": [
                    {
                        "session_id": "synthetic-session",
                        "message_id": "synthetic-message",
                        "char_start": 4,
                        "char_end": 23,
                        "quote": "I prefer quiet work",
                    }
                ],
            }
        ]
    }


def _claim_for(
    span: EvidenceSpan,
    *,
    speaker_role: str = "user",
    attribution: str = "direct_user_statement",
) -> Claim:
    return Claim(
        "the user",
        "REPORTS",
        "a synthetic fact",
        speaker_role,
        attribution,
        "asserted",
        "positive",
        None,
        None,
        (span,),
    )


class ClaimParserTests(unittest.TestCase):
    def test_parser_builds_immutable_claim_and_evidence_span(self) -> None:
        parsed = parse_candidates(_valid_payload())

        self.assertEqual(len(parsed), 1)
        claim = parsed[0]
        self.assertIsInstance(claim, Claim)
        self.assertEqual(
            claim.evidence,
            (EvidenceSpan("synthetic-session", "synthetic-message", 4, 23, "I prefer quiet work"),),
        )
        with self.assertRaises(FrozenInstanceError):
            claim.subject = "mutated"

    def test_parser_rejects_unknown_keys_and_boolean_offsets(self) -> None:
        invalid_payloads = []

        claim_extra = _valid_payload()
        claim_extra["claims"][0]["unexpected"] = "unknown fields are not ignored"
        invalid_payloads.append(claim_extra)

        span_extra = _valid_payload()
        span_extra["claims"][0]["evidence"][0]["unexpected"] = "unknown span field"
        invalid_payloads.append(span_extra)

        missing_field = _valid_payload()
        del missing_field["claims"][0]["object"]
        invalid_payloads.append(missing_field)

        boolean_offset = _valid_payload()
        boolean_offset["claims"][0]["evidence"][0]["char_start"] = True
        invalid_payloads.append(boolean_offset)

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    parse_candidates(payload)

    def test_parser_rejects_invalid_labels_dates_and_empty_evidence(self) -> None:
        invalid_payloads = []
        for field, value in (
            ("subject", "  \n"),
            ("predicate", "INFERRED"),
            ("speaker_role", "system"),
            ("modality", "certainish"),
            ("polarity", "neutral"),
            ("valid_from", "2026-1-01"),
        ):
            payload = _valid_payload()
            payload["claims"][0][field] = value
            invalid_payloads.append(payload)

        reversed_dates = _valid_payload()
        reversed_dates["claims"][0]["valid_from"] = "2026-10-02"
        reversed_dates["claims"][0]["valid_to"] = "2026-10-01"
        invalid_payloads.append(reversed_dates)

        empty_evidence = _valid_payload()
        empty_evidence["claims"][0]["evidence"] = []
        invalid_payloads.append(empty_evidence)

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    parse_candidates(payload)

    def test_parser_rejects_excessive_claim_and_text_sizes(self) -> None:
        too_many_claims = {"claims": [_valid_payload()["claims"][0] for _ in range(65)]}
        oversized_subject = _valid_payload()
        oversized_subject["claims"][0]["subject"] = "x" * 2_049
        oversized_quote = _valid_payload()
        oversized_quote["claims"][0]["evidence"][0]["quote"] = "x" * 8_193
        oversized_identifier = _valid_payload()
        oversized_identifier["claims"][0]["evidence"][0]["session_id"] = "x" * 257
        too_many_spans = _valid_payload()
        span = too_many_spans["claims"][0]["evidence"][0]
        too_many_spans["claims"][0]["evidence"] = [span.copy() for _ in range(65)]
        oversized_claim = _valid_payload()
        oversized_claim["claims"][0]["subject"] = "s" * 2_048
        oversized_claim["claims"][0]["object"] = "o" * 2_048
        oversized_claim["claims"][0]["evidence"] = [
            {
                "session_id": "s",
                "message_id": "m",
                "char_start": 0,
                "char_end": 8_192,
                "quote": "x" * 8_192,
            }
            for _ in range(4)
        ]
        large_candidate = _valid_payload()["claims"][0]
        large_candidate["subject"] = "s" * 2_048
        large_candidate["object"] = "o" * 2_048
        large_candidate["evidence"] = [
            {
                "session_id": "s",
                "message_id": "m",
                "char_start": 0,
                "char_end": 8_192,
                "quote": "x" * 8_192,
            }
            for _ in range(2)
        ]
        oversized_payload = {"claims": [large_candidate] * 50}

        for payload in (
            too_many_claims,
            oversized_subject,
            oversized_quote,
            oversized_identifier,
            too_many_spans,
            oversized_claim,
            oversized_payload,
        ):
            with self.subTest(claim_count=len(payload["claims"])):
                with self.assertRaises((TypeError, ValueError)):
                    parse_candidates(payload)

    def test_payload_must_be_a_dictionary_not_encoded_json(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            parse_candidates('{"claims": []}')

    def test_empty_claim_list_is_abstention(self) -> None:
        self.assertEqual(parse_candidates({"claims": []}), ())

    def test_direct_dataclass_construction_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            EvidenceSpan("session", "message", True, 1, "x")
        with self.assertRaises(ValueError):
            Claim(
                "subject",
                "PREFERS",
                "object",
                "user",
                "direct_user_statement",
                "preference",
                "positive",
                None,
                None,
                (),
            )
        with self.assertRaises(ValueError):
            Claim(
                "subject",
                "UNKNOWN_PREDICATE",
                "object",
                "user",
                "direct_user_statement",
                "preference",
                "positive",
                None,
                None,
                (EvidenceSpan("session", "message", 0, 1, "x"),),
            )


class CandidateVerificationTests(unittest.TestCase):
    def test_exact_unicode_span_returns_review_and_revision_bound_citation(self) -> None:
        claim = Claim(
            "the user",
            "REPORTS",
            "a planet",
            "user",
            "direct_user_statement",
            "asserted",
            "positive",
            None,
            None,
            (EvidenceSpan("synthetic-session", "synthetic-message", 1, 2, "🪐"),),
        )
        record = EvidenceRecord(
            "synthetic",
            "synthetic-session",
            "synthetic-message",
            0,
            "user",
            "A🪐B",
        )

        result = verify_candidate(claim, [record])

        self.assertEqual(result.status, "REVIEW")
        self.assertEqual(result.errors, ())
        self.assertEqual(
            result.citations,
            (f"{record.record_id}@{record.revision}:1:2",),
        )

    def test_verifier_rejects_unknown_mismatched_and_out_of_bounds_spans(self) -> None:
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", "exact")
        cases = (
            (EvidenceSpan("missing", "message", 0, 1, "e"), "unknown_reference"),
            (EvidenceSpan("session", "message", 0, 1, "x"), "quote_mismatch"),
            (EvidenceSpan("session", "message", 0, 7, "exact!!"), "span_out_of_bounds"),
        )
        for span, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                result = verify_candidate(_claim_for(span), [record])
                self.assertEqual(result.status, "REJECT")
                self.assertIn(expected_error, result.errors)
                self.assertEqual(result.citations, ())

    def test_verifier_rejects_ambiguous_duplicate_message_spans(self) -> None:
        span = EvidenceSpan("session", "message", 0, 5, "exact")
        first = EvidenceRecord("synthetic", "session", "message", 0, "user", "exact")
        second = EvidenceRecord("synthetic", "session", "message", 1, "user", "exact")

        result = verify_candidate(_claim_for(span), [first, second])

        self.assertEqual(result.status, "REJECT")
        self.assertEqual(result.errors, ("ambiguous_reference",))
        self.assertEqual(result.citations, ())

    def test_verifier_rejects_source_role_and_attribution_conflicts(self) -> None:
        span = EvidenceSpan("session", "message", 0, 5, "exact")
        assistant_record = EvidenceRecord(
            "synthetic", "session", "message", 0, "assistant", "exact"
        )

        source_conflict = verify_candidate(_claim_for(span), [assistant_record])
        self.assertEqual(source_conflict.status, "REJECT")
        self.assertIn("source_role_mismatch", source_conflict.errors)
        self.assertEqual(len(source_conflict.citations), 1)

        attribution_conflict = verify_candidate(
            _claim_for(span, speaker_role="assistant"), [assistant_record]
        )
        self.assertEqual(attribution_conflict.status, "REJECT")
        self.assertIn("attribution_role_conflict", attribution_conflict.errors)


if __name__ == "__main__":
    unittest.main()
