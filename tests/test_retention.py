"""Tests for the pure source-retention impact diagnostic."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tracemem.citations import fetch_citation
from tracemem.claims import Claim, EvidenceSpan, VerificationResult
from tracemem.ledger import CandidateEntry, CandidateLedger
from tracemem.retention import ErasureImpact, plan_source_erasure
from tracemem.schema import EvidenceRecord
from tracemem.storage import EvidenceStore


def _record(
    source: str,
    session_id: str,
    message_id: str,
    text: str,
    *,
    part_index: int = 0,
) -> EvidenceRecord:
    return EvidenceRecord(
        source=source,
        session_id=session_id,
        message_id=message_id,
        part_index=part_index,
        role="user",
        text=text,
        content_kind="conversation",
    )


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _citation(record: EvidenceRecord, start: int = 0, end: int = 1) -> str:
    return f"{record.record_id}@{record.revision}:{start}:{end}"


def _entry(
    label: str,
    *,
    session_id: str = "fictional-session",
    message_id: str = "fictional-message",
    quote: str = "x",
    citations: tuple[str, ...] = (),
    supersedes: tuple[str, ...] = (),
    status: str = "REVIEW",
    errors: tuple[str, ...] = (),
) -> CandidateEntry:
    claim = Claim(
        "fictional subject",
        "PREFERS",
        "fictional object",
        "user",
        "direct_user_statement",
        "preference",
        "positive",
        None,
        None,
        (EvidenceSpan(session_id, message_id, 0, len(quote), quote),),
    )
    return CandidateEntry(
        _id(label),
        claim,
        VerificationResult(status, errors, citations),  # type: ignore[arg-type]
        "fictional-extractor",
        "2026-10-14T10:00:00Z",
        supersedes,
    )


class SourceRetentionTests(unittest.TestCase):
    def test_selects_every_historical_revision_and_part_for_exact_source_session(self) -> None:
        selected_old = _record("provider-a", "session-é", "message-0", "Old café 東京 quote")
        selected_new = _record("provider-a", "session-é", "message-0", "New café 東京 quote")
        selected_part = _record("provider-a", "session-é", "message-0", "Second part", part_index=1)
        different_provider = _record(
            "provider-b", "session-é", "message-0", "Same identity, other provider"
        )
        different_session = _record("provider-a", "session-other", "message-0", "Other session")
        different_message = _record("provider-a", "session-é", "message-1", "Other message")

        impact = plan_source_erasure(
            [
                different_provider,
                selected_new,
                different_session,
                selected_part,
                selected_old,
                different_message,
            ],
            [],
            source="provider-a",
            session_id="session-é",
        )

        self.assertIsInstance(impact, ErasureImpact)
        self.assertEqual(
            impact.selected_record_revisions,
            tuple(
                sorted(
                    {
                        (selected_old.record_id, selected_old.revision),
                        (selected_new.record_id, selected_new.revision),
                        (selected_part.record_id, selected_part.revision),
                        (different_message.record_id, different_message.revision),
                    }
                )
            ),
        )

    def test_exact_citations_and_rejected_quotes_are_direct(self) -> None:
        selected = _record("fictional-provider-a", "fictional-session", "message", "é selected")
        other_provider = _record(
            "fictional-provider-b", "fictional-session", "message", "é elsewhere"
        )
        rejected = _entry(
            "rejected-direct",
            citations=(_citation(selected),),
            status="REJECT",
            errors=("fictional verification note",),
        )
        other_provider_quote = _entry(
            "other-provider-quote",
            citations=(_citation(other_provider),),
            session_id="fictional-session",
            message_id="message",
        )
        unknown_revision_quote = _entry(
            "unknown-revision-quote",
            citations=(f"{selected.record_id}@{_id('not-in-snapshot')}:0:1",),
            session_id="fictional-session",
            message_id="message",
        )
        span_without_citation = _entry(
            "span-without-citation",
            session_id="fictional-session",
            message_id="message",
        )
        unrelated = _entry(
            "unrelated-quote",
            citations=(_citation(other_provider),),
            session_id="another-session",
            message_id="message",
        )

        impact = plan_source_erasure(
            [selected, other_provider],
            [
                unrelated,
                span_without_citation,
                unknown_revision_quote,
                other_provider_quote,
                rejected,
            ],
            source="fictional-provider-a",
            session_id="fictional-session",
        )

        self.assertEqual(impact.direct_candidate_ids, (rejected.entry_id,))
        self.assertEqual(
            impact.ambiguous_candidate_ids,
            tuple(
                sorted(
                    (
                        other_provider_quote.entry_id,
                        span_without_citation.entry_id,
                        unknown_revision_quote.entry_id,
                    )
                )
            ),
        )
        self.assertNotIn(unrelated.entry_id, impact.affected_candidate_ids)
        serialized = repr(impact)
        for private_text in (
            "fictional-provider-a",
            "fictional-session",
            "fictional verification note",
            "fictional subject",
            "fictional object",
            "é selected",
        ):
            self.assertNotIn(private_text, serialized)

    def test_branched_descendants_are_affected_but_ancestors_and_siblings_are_not(self) -> None:
        source = _record("fictional-provider", "fictional-session", "source-message", "x source")
        ancestor = _entry("ancestor")
        direct = _entry(
            "direct-root",
            citations=(_citation(source),),
            supersedes=(ancestor.entry_id,),
        )
        sibling = _entry("unaffected-sibling", supersedes=(ancestor.entry_id,))
        left = _entry("left-child", supersedes=(direct.entry_id,))
        right = _entry("right-child", supersedes=(direct.entry_id,))
        leaf = _entry("shared-leaf", supersedes=(left.entry_id, right.entry_id))
        unrelated = _entry("unrelated-entry")

        impact = plan_source_erasure(
            [source],
            [leaf, sibling, ancestor, right, unrelated, direct, left],
            source="fictional-provider",
            session_id="fictional-session",
        )

        self.assertEqual(impact.direct_candidate_ids, (direct.entry_id,))
        self.assertEqual(
            impact.dependent_candidate_ids,
            tuple(sorted((left.entry_id, right.entry_id, leaf.entry_id))),
        )
        self.assertEqual(
            impact.affected_candidate_ids,
            tuple(sorted((direct.entry_id, left.entry_id, right.entry_id, leaf.entry_id))),
        )
        self.assertNotIn(ancestor.entry_id, impact.affected_candidate_ids)
        self.assertNotIn(sibling.entry_id, impact.affected_candidate_ids)
        self.assertNotIn(unrelated.entry_id, impact.affected_candidate_ids)

    def test_empty_target_and_input_order_do_not_change_results_or_inputs(self) -> None:
        selected = _record("fictional-provider", "fictional-session", "message", "x source")
        direct = _entry("order-direct", citations=(_citation(selected),))
        ambiguous = _entry(
            "order-ambiguous",
            session_id="fictional-session",
            message_id="message",
        )
        child = _entry("order-child", supersedes=(direct.entry_id,))
        records = [selected]
        entries = [child, ambiguous, direct]
        records_before = records.copy()
        entries_before = entries.copy()

        forward = plan_source_erasure(
            records,
            entries,
            source="fictional-provider",
            session_id="fictional-session",
        )
        reverse = plan_source_erasure(
            list(reversed(records)),
            list(reversed(entries)),
            source="fictional-provider",
            session_id="fictional-session",
        )
        empty = plan_source_erasure(
            records,
            entries,
            source="no-matching-provider",
            session_id="no-matching-session",
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(records, records_before)
        self.assertEqual(entries, entries_before)
        self.assertEqual(empty.selected_record_revisions, ())
        self.assertEqual(empty.affected_candidate_ids, ())

    def test_inputs_must_be_sequences_with_exact_record_and_entry_types(self) -> None:
        with self.assertRaises(TypeError):
            plan_source_erasure(
                iter(()),  # type: ignore[arg-type]
                [],
                source="provider",
                session_id="session",
            )
        with self.assertRaises(TypeError):
            plan_source_erasure(
                "records",  # type: ignore[arg-type]
                [],
                source="provider",
                session_id="session",
            )
        with self.assertRaises(TypeError):
            plan_source_erasure(
                [object()],  # type: ignore[list-item]
                [],
                source="provider",
                session_id="session",
            )

        subclass = type("DerivedRecord", (EvidenceRecord,), {})
        derived_record = subclass("provider", "session", "message", 0, "user", "x")
        with self.assertRaises(TypeError):
            plan_source_erasure(
                [derived_record],
                [],
                source="provider",
                session_id="session",
            )

    def test_invalid_selectors_and_duplicate_record_or_entry_keys_fail_sanitized(self) -> None:
        record = _record("private-provider", "private-session", "message", "x text")
        entry = _entry("duplicate-entry")

        for source, session in (("", "private-session"), ("private-provider", "")):
            with self.subTest(selector=(source, session)), self.assertRaises(ValueError) as caught:
                plan_source_erasure([], [], source=source, session_id=session)
            self.assertNotIn("private", str(caught.exception))

        for source, session in (("bad\ud800", "session"), ("provider", "bad\udfff")):
            with self.subTest(selector=(source, session)), self.assertRaises(ValueError):
                plan_source_erasure([], [], source=source, session_id=session)

        with self.assertRaisesRegex(ValueError, "duplicate record/revision"):
            plan_source_erasure(
                [record, record], [], source="private-provider", session_id="private-session"
            )
        with self.assertRaisesRegex(ValueError, "duplicate candidate IDs"):
            plan_source_erasure([], [entry, entry], source="provider", session_id="session")

    def test_invalid_citations_and_lineage_fail_closed_without_echoing_input(self) -> None:
        record = _record("fictional-provider", "fictional-session", "message", "x short")
        malformed = _entry("malformed-citation", citations=("private citation text",))
        out_of_bounds = _entry(
            "out-of-bounds-citation",
            citations=(_citation(record, 0, len(record.text) + 1),),
        )
        oversized_offset = _entry(
            "oversized-offset-citation",
            citations=(f"{record.record_id}@{record.revision}:0:{'9' * 5000}",),
        )
        with self.assertRaises(ValueError) as caught:
            plan_source_erasure(
                [record],
                [malformed],
                source="fictional-provider",
                session_id="fictional-session",
            )
        self.assertNotIn("private citation text", str(caught.exception))
        with self.assertRaisesRegex(ValueError, "invalid citation"):
            plan_source_erasure(
                [record],
                [out_of_bounds],
                source="fictional-provider",
                session_id="fictional-session",
            )
        with self.assertRaisesRegex(ValueError, "invalid citation") as caught:
            plan_source_erasure(
                [record],
                [oversized_offset],
                source="fictional-provider",
                session_id="fictional-session",
            )
        self.assertLess(len(str(caught.exception)), 100)

        dangling = _entry("dangling-lineage", supersedes=("f" * 64,))
        with self.assertRaisesRegex(ValueError, "dangling lineage"):
            plan_source_erasure([], [dangling], source="provider", session_id="session")
        first_id = _id("cycle-first")
        second_id = _id("cycle-second")
        cycle_first = replace(_entry("cycle-first"), supersedes=(second_id,))
        cycle_second = replace(_entry("cycle-second"), supersedes=(first_id,))
        with self.assertRaisesRegex(ValueError, "cyclic lineage"):
            plan_source_erasure(
                [], [cycle_first, cycle_second], source="provider", session_id="session"
            )

    def test_malformed_entry_ids_lineage_and_nested_values_fail_without_a_match(self) -> None:
        valid = _entry("invalid-input-control")
        invalid_id = replace(valid, entry_id="private-candidate-id")
        invalid_supersedes = replace(valid, supersedes=("g" * 64,))
        duplicate_parent = _entry("duplicate-parent")
        duplicate_child = _entry(
            "duplicate-child",
            supersedes=(duplicate_parent.entry_id, duplicate_parent.entry_id),
        )
        self_superseding = _entry("self-superseding")
        self_superseding = replace(self_superseding, supersedes=(self_superseding.entry_id,))
        invalid_lineage_shape = replace(valid, supersedes=[_id("parent")])  # type: ignore[arg-type]

        malformed_claim = object.__new__(Claim)
        for field in (
            "subject",
            "predicate",
            "object",
            "speaker_role",
            "attribution",
            "modality",
            "polarity",
            "valid_from",
            "valid_to",
        ):
            object.__setattr__(malformed_claim, field, getattr(valid.claim, field))
        object.__setattr__(malformed_claim, "evidence", (object(),))
        invalid_span = replace(valid, claim=malformed_claim)

        malformed_verification = object.__new__(VerificationResult)
        object.__setattr__(malformed_verification, "status", "REVIEW")
        object.__setattr__(malformed_verification, "errors", ())
        object.__setattr__(malformed_verification, "citations", [])
        invalid_verification = replace(valid, verification=malformed_verification)
        invalid_verification_type = replace(valid, verification=object())  # type: ignore[arg-type]

        for entries in (
            [invalid_id],
            [invalid_supersedes],
            [duplicate_parent, duplicate_child],
            [self_superseding],
            [invalid_lineage_shape],
            [invalid_span],
            [invalid_verification],
            [invalid_verification_type],
        ):
            with (
                self.subTest(entries=entries),
                self.assertRaises((TypeError, ValueError)) as caught,
            ):
                plan_source_erasure(
                    [],
                    entries,
                    source="no-match-provider",
                    session_id="no-match-session",
                )
            self.assertNotIn("no-match", str(caught.exception))
            self.assertNotIn("private-candidate-id", str(caught.exception))

        for invalid_entries in ("entries", iter(())):
            with self.subTest(entries=type(invalid_entries).__name__), self.assertRaises(TypeError):
                plan_source_erasure(
                    [],
                    invalid_entries,  # type: ignore[arg-type]
                    source="no-match-provider",
                    session_id="no-match-session",
                )
        with self.assertRaises(TypeError):
            plan_source_erasure(
                [],
                [object()],  # type: ignore[list-item]
                source="no-match-provider",
                session_id="no-match-session",
            )

    def test_claims_are_revalidated_and_report_contains_required_limits(self) -> None:
        selected = _record("fictional-provider", "fictional-session", "message", "x source")
        entry = _entry("report-direct", citations=(_citation(selected),))
        malformed_claim = object.__new__(Claim)
        object.__setattr__(malformed_claim, "subject", object())
        object.__setattr__(malformed_claim, "evidence", entry.claim.evidence)
        malformed_entry = replace(entry, claim=malformed_claim)

        with self.assertRaises((TypeError, ValueError)):
            plan_source_erasure(
                [selected],
                [malformed_entry],
                source="fictional-provider",
                session_id="fictional-session",
            )

        impact = plan_source_erasure(
            [selected],
            [entry],
            source="fictional-provider",
            session_id="fictional-session",
        )
        report = " ".join(impact.disclosures).casefold()
        for disclosure in (
            "snapshot dependency diagnostic",
            "deletion is not performed",
            "vector-cache keys correspond to selected record/revision pairs",
            "cache counts and physical deletion are not audited",
            "external exports",
            "backups",
            "unlinked copies",
            "free SQLite pages",
            "in-memory caches",
            "all historical evidence records and candidate ledger entries",
            "semantic dependence is not guaranteed",
            "complete information-flow detection is not guaranteed",
        ):
            self.assertIn(disclosure.casefold(), report)

    def test_real_stores_remain_byte_identical_and_source_citation_reads_still_work(self) -> None:
        original = _record(
            "fictional-provider", "fictional-session", "message", "I prefer fictional tea."
        )
        revised = _record(
            "fictional-provider",
            "fictional-session",
            "message",
            "I prefer fictional tea and toast.",
        )
        claim = Claim(
            "the fictional user",
            "PREFERS",
            "fictional tea",
            "user",
            "direct_user_statement",
            "preference",
            "positive",
            None,
            None,
            (
                EvidenceSpan(
                    original.session_id,
                    original.message_id,
                    0,
                    len(original.text),
                    original.text,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.sqlite3"
            ledger_path = Path(directory) / "ledger.sqlite3"
            with EvidenceStore(evidence_path) as store:
                store.ingest([original, revised])
            with CandidateLedger(ledger_path) as ledger:
                entry = ledger.append(
                    claim,
                    [original],
                    "fictional-extractor",
                    "2026-10-14T10:00:00Z",
                )
            evidence_before = evidence_path.read_bytes()
            ledger_before = ledger_path.read_bytes()

            with EvidenceStore(evidence_path) as store, CandidateLedger(ledger_path) as ledger:
                records = store.records(current_only=False)
                entries = ledger.entries()
                citation = entry.verification.citations[0]
                impact = plan_source_erasure(
                    records,
                    entries,
                    source="fictional-provider",
                    session_id="fictional-session",
                )
                self.assertEqual(store.get(original.record_id, original.revision), original)
                self.assertEqual(fetch_citation(store, citation).record, original)
                self.assertEqual(ledger.get(entry.entry_id), entry)
                self.assertEqual(impact.direct_candidate_ids, (entry.entry_id,))

            self.assertEqual(evidence_path.read_bytes(), evidence_before)
            self.assertEqual(ledger_path.read_bytes(), ledger_before)


if __name__ == "__main__":
    unittest.main()
