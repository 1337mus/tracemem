"""Tests for exact, revision-bound citation retrieval."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from tracemem import citations
from tracemem.claims import Claim, EvidenceSpan, VerificationResult
from tracemem.ledger import CandidateLedger
from tracemem.schema import EvidenceRecord
from tracemem.storage import EvidenceStore


def _claim_for(record: EvidenceRecord) -> Claim:
    return Claim(
        "the user",
        "PREFERS",
        "tea",
        "user",
        "direct_user_statement",
        "preference",
        "positive",
        None,
        None,
        (
            EvidenceSpan(
                record.session_id,
                record.message_id,
                0,
                len(record.text),
                record.text,
            ),
        ),
    )


class CitationRetrievalTests(unittest.TestCase):
    def test_fetch_returns_exact_unicode_source_slice(self) -> None:
        text = "I prefer café ☕ today."
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", text)
        start = len("I prefer ")
        end = len("I prefer café ☕")
        citation = f"{record.record_id}@{record.revision}:{start}:{end}"

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                store.ingest([record])

                resolved = citations.fetch_citation(store, citation)

        self.assertEqual(resolved.citation, citation)
        self.assertEqual(resolved.record, record)
        self.assertEqual((resolved.char_start, resolved.char_end), (start, end))
        self.assertEqual(resolved.quote, "café ☕")
        with self.assertRaises(FrozenInstanceError):
            resolved.quote = "changed"  # type: ignore[misc]

    def test_old_revision_citation_survives_ingest_and_close_reopen(self) -> None:
        original = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        revised = EvidenceRecord(
            "synthetic", "session", "message", 0, "user", "I prefer tea and coffee."
        )
        self.assertEqual(original.record_id, revised.record_id)
        self.assertNotEqual(original.revision, revised.revision)
        claim = _claim_for(original)

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.sqlite3"
            ledger_path = Path(directory) / "ledger.sqlite3"
            with CandidateLedger(ledger_path) as ledger:
                entry = ledger.append(claim, [original], "extractor-v1", "2026-10-14T10:00:00Z")
                citation = entry.verification.citations[0]
            with EvidenceStore(evidence_path) as store:
                store.ingest([original, revised])

            with EvidenceStore(evidence_path) as reopened_store:
                resolved = citations.fetch_citation(reopened_store, citation)
            with CandidateLedger(ledger_path) as reopened_ledger:
                reopened_entry = reopened_ledger.get(entry.entry_id)

        self.assertEqual(reopened_entry, entry)
        self.assertEqual(resolved.record, original)
        self.assertEqual(resolved.quote, original.text)

    def test_unavailable_revision_raises_without_falling_back_to_head(self) -> None:
        original = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        revised = EvidenceRecord(
            "synthetic", "session", "message", 0, "user", "I prefer tea and coffee."
        )
        citation = f"{original.record_id}@{original.revision}:0:13"

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                store.ingest([revised])

                with self.assertRaises(KeyError):
                    citations.fetch_citation(store, citation)
                self.assertEqual(store.get(revised.record_id, revised.revision), revised)

    def test_malformed_noncanonical_and_out_of_bounds_citations_are_rejected(self) -> None:
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        valid = f"{record.record_id}@{record.revision}:0:5"
        invalid = (
            f"{record.record_id[:-1]}@{record.revision}:0:5",
            f"{record.record_id.upper()}@{record.revision}:0:5",
            f"{record.record_id}@{record.revision.upper()}:0:5",
            f"{record.record_id}@{record.revision}:00:5",
            f"{record.record_id}@{record.revision}:0:05",
            f"{record.record_id}@{record.revision}:-1:5",
            f"{record.record_id}@{record.revision}:2:2",
            f"{record.record_id}@{record.revision}:0:{len(record.text) + 1}",
        )

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                store.ingest([record])
                for citation in invalid:
                    with self.subTest(citation=citation), self.assertRaises(ValueError):
                        citations.fetch_citation(store, citation)

                resolved = citations.fetch_citation(store, valid)

        self.assertEqual(resolved.quote, "I pre")

    def test_entry_evidence_is_resolved_in_claim_span_order(self) -> None:
        first = EvidenceRecord("synthetic", "session", "first", 0, "user", "I prefer tea.")
        second = EvidenceRecord("synthetic", "session", "second", 0, "user", "Tea is soothing.")
        claim = Claim(
            "the user",
            "PREFERS",
            "tea",
            "user",
            "direct_user_statement",
            "preference",
            "positive",
            None,
            None,
            (
                EvidenceSpan("session", "second", 0, len(second.text), second.text),
                EvidenceSpan("session", "first", 0, len(first.text), first.text),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.sqlite3"
            ledger_path = Path(directory) / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([first, second])
                with CandidateLedger(ledger_path) as ledger:
                    entry = ledger.append(
                        claim,
                        [first, second],
                        "extractor-v1",
                        "2026-10-14T10:00:00Z",
                    )
                    records_before = store.records(current_only=False)
                    entries_before = ledger.entries()
                    store_changes_before = store._connection.total_changes
                    ledger_changes_before = ledger._connection.total_changes
                    resolved = citations.fetch_entry_evidence(store, entry)
                    self.assertEqual(store._connection.total_changes, store_changes_before)
                    self.assertEqual(ledger._connection.total_changes, ledger_changes_before)
                    self.assertEqual(store.records(current_only=False), records_before)
                    self.assertEqual(ledger.entries(), entries_before)

        self.assertEqual([item.record.message_id for item in resolved], ["second", "first"])
        self.assertEqual([item.quote for item in resolved], [second.text, first.text])

    def test_reject_candidate_is_refused_without_partial_resolution(self) -> None:
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        claim = _claim_for(record)

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                    entry = ledger.append(claim, [], "extractor-v1", "2026-10-14T10:00:00Z")

                    self.assertEqual(entry.verification.status, "REJECT")
                    with self.assertRaisesRegex(ValueError, "REJECT"):
                        citations.fetch_entry_evidence(store, entry)

    def test_entry_requires_clean_review_and_one_citation_per_span(self) -> None:
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        claim = _claim_for(record)

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                store.ingest([record])
                with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                    entry = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")
                    unclean = replace(
                        entry,
                        verification=VerificationResult(
                            "REVIEW", ("quote_mismatch",), entry.verification.citations
                        ),
                    )
                    missing_citation = replace(
                        entry,
                        verification=VerificationResult("REVIEW", (), ()),
                    )

                    with self.assertRaisesRegex(ValueError, "integrity-clean"):
                        citations.fetch_entry_evidence(store, unclean)
                    with self.assertRaisesRegex(ValueError, "one citation per evidence span"):
                        citations.fetch_entry_evidence(store, missing_citation)

    def test_mismatched_claim_quote_identity_and_role_are_rejected(self) -> None:
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        claim = _claim_for(record)

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                store.ingest([record])
                with CandidateLedger(Path(directory) / "ledger.sqlite3") as ledger:
                    entry = ledger.append(claim, [record], "extractor-v1", "2026-10-14T10:00:00Z")

                    original_span = entry.claim.evidence[0]
                    bad_quote_claim = replace(
                        entry.claim,
                        evidence=(replace(original_span, quote="I prefer oat."),),
                    )
                    wrong_identity_claim = replace(
                        entry.claim,
                        evidence=(replace(original_span, session_id="other-session"),),
                    )
                    wrong_role_claim = replace(
                        entry.claim,
                        speaker_role="assistant",
                        attribution="assistant_statement",
                    )
                    wrong_attribution_claim = replace(
                        entry.claim,
                        attribution="assistant_statement",
                    )
                    for claim_variant in (
                        bad_quote_claim,
                        wrong_identity_claim,
                        wrong_role_claim,
                        wrong_attribution_claim,
                    ):
                        tampered_entry = replace(entry, claim=claim_variant)
                        with self.subTest(claim=claim_variant), self.assertRaises(ValueError):
                            citations.fetch_entry_evidence(store, tampered_entry)

    def test_corrupt_stored_row_identity_is_rejected(self) -> None:
        record = EvidenceRecord("synthetic", "session", "message", 0, "user", "I prefer tea.")
        citation = f"{record.record_id}@{record.revision}:0:{len(record.text)}"

        with tempfile.TemporaryDirectory() as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite3") as store:
                store.ingest([record])
                store._connection.execute("DROP TRIGGER evidence_immutable_update")
                store._connection.execute(
                    "UPDATE evidence SET text = ? WHERE record_id = ? AND revision = ?",
                    ("tampered stored text", record.record_id, record.revision),
                )

                with self.assertRaisesRegex(ValueError, "identity"):
                    citations.fetch_citation(store, citation)


if __name__ == "__main__":
    unittest.main()
