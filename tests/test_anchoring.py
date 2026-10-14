"""Conservative quote anchoring tests."""

import unittest

from tracemem.anchoring import anchor_unique_quotes
from tracemem.claims import Claim, EvidenceSpan, verify_candidate
from tracemem.schema import EvidenceRecord


def _claim_for(span: EvidenceSpan, **overrides: str) -> Claim:
    values = {
        "subject": "the user",
        "predicate": "REPORTS",
        "object": "a synthetic fact",
        "speaker_role": "user",
        "attribution": "direct_user_statement",
        "modality": "asserted",
        "polarity": "positive",
        "valid_from": None,
        "valid_to": None,
        "evidence": (span,),
    }
    values.update(overrides)
    return Claim(**values)  # type: ignore[arg-type]


def _record(text: str, *, part_index: int = 0, role: str = "user") -> EvidenceRecord:
    return EvidenceRecord("synthetic", "session", "message", part_index, role, text)


class QuoteAnchoringTests(unittest.TestCase):
    def test_reanchors_unique_unicode_quote_and_remains_reviewable(self) -> None:
        quote = "🪐 tea"
        record = _record(f"A{quote} B")
        claim = _claim_for(EvidenceSpan("session", "message", 0, len(quote), quote))

        result = anchor_unique_quotes(claim, [record])

        expected_start = record.text.index(quote)
        self.assertEqual(result.claim.evidence[0].char_start, expected_start)
        self.assertEqual(result.claim.evidence[0].char_end, expected_start + len(quote))
        self.assertEqual(len(result.repairs), 1)
        self.assertEqual(result.repairs[0].old_start, 0)
        self.assertEqual(result.repairs[0].old_end, len(quote))
        self.assertEqual(result.repairs[0].new_start, expected_start)
        self.assertEqual(result.repairs[0].new_end, expected_start + len(quote))
        self.assertEqual(result.repairs[0].evidence_index, 0)
        self.assertEqual(result.repairs[0].record_id, record.record_id)
        self.assertEqual(result.repairs[0].revision, record.revision)
        self.assertEqual(result.unresolved, ())
        verification = verify_candidate(result.claim, [record])
        self.assertEqual(verification.status, "REVIEW")
        self.assertEqual(verification.errors, ())

    def test_correct_offsets_are_preserved_when_quote_repeats(self) -> None:
        record = _record("red then red")
        span = EvidenceSpan("session", "message", 0, 3, "red")
        claim = _claim_for(span)

        result = anchor_unique_quotes(claim, [record])

        self.assertIsNot(result.claim, claim)
        self.assertEqual(result.claim.evidence, (span,))
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.unresolved, ())

    def test_overlapping_occurrences_and_distinct_revisions_are_ambiguous(self) -> None:
        overlap_record = _record("aaa")
        overlap_claim = _claim_for(EvidenceSpan("session", "message", 0, 1, "aa"))
        overlap_result = anchor_unique_quotes(overlap_claim, [overlap_record])
        self.assertEqual(overlap_result.repairs, ())
        self.assertEqual(overlap_result.claim.evidence, overlap_claim.evidence)
        self.assertEqual(overlap_result.unresolved, ("evidence[0]:ambiguous_quote_occurrences",))

        first_revision = _record("x apple")
        second_revision = _record("y apple")
        self.assertEqual(first_revision.record_id, second_revision.record_id)
        self.assertNotEqual(first_revision.revision, second_revision.revision)
        revision_claim = _claim_for(EvidenceSpan("session", "message", 0, 5, "apple"))
        revision_result = anchor_unique_quotes(revision_claim, [first_revision, second_revision])
        self.assertEqual(revision_result.repairs, ())
        self.assertEqual(revision_result.claim.evidence, revision_claim.evidence)
        self.assertEqual(revision_result.unresolved, ("evidence[0]:ambiguous_quote_occurrences",))

    def test_duplicate_record_revisions_do_not_create_false_ambiguity(self) -> None:
        record = _record("x apple")
        duplicate = _record("x apple")
        self.assertEqual(record.record_id, duplicate.record_id)
        self.assertEqual(record.revision, duplicate.revision)
        claim = _claim_for(EvidenceSpan("session", "message", 0, 5, "apple"))

        result = anchor_unique_quotes(claim, [record, duplicate])

        self.assertEqual(len(result.repairs), 1)
        self.assertEqual(result.repairs[0].record_id, record.record_id)
        self.assertEqual(result.repairs[0].revision, record.revision)
        self.assertEqual(result.claim.evidence[0].char_start, 2)
        self.assertEqual(result.unresolved, ())

    def test_unknown_references_and_absent_quotes_are_left_unchanged(self) -> None:
        unknown_span = EvidenceSpan("other-session", "message", 0, 5, "apple")
        unknown_claim = _claim_for(unknown_span)
        unknown_result = anchor_unique_quotes(unknown_claim, [_record("x apple")])
        self.assertEqual(unknown_result.claim.evidence, (unknown_span,))
        self.assertEqual(unknown_result.repairs, ())
        self.assertEqual(unknown_result.unresolved, ("evidence[0]:unknown_reference",))

        absent_span = EvidenceSpan("session", "message", 0, 6, "absent")
        absent_claim = _claim_for(absent_span)
        absent_result = anchor_unique_quotes(absent_claim, [_record("source text")])
        self.assertEqual(absent_result.claim.evidence, (absent_span,))
        self.assertEqual(absent_result.repairs, ())
        self.assertEqual(absent_result.unresolved, ("evidence[0]:quote_not_found",))

    def test_repair_does_not_hide_source_role_conflict(self) -> None:
        quote = "🪐 tea"
        record = _record(f"A{quote} B", role="assistant")
        claim = _claim_for(EvidenceSpan("session", "message", 0, len(quote), quote))

        result = anchor_unique_quotes(claim, [record])
        verification = verify_candidate(result.claim, [record])

        self.assertEqual(result.claim.subject, claim.subject)
        self.assertEqual(result.claim.attribution, claim.attribution)
        self.assertEqual(result.claim.speaker_role, claim.speaker_role)
        self.assertEqual(verification.status, "REJECT")
        self.assertIn("source_role_mismatch", verification.errors)


if __name__ == "__main__":
    unittest.main()
