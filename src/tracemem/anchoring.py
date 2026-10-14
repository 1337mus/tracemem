"""Conservatively repair unambiguous evidence quote offsets for parsed claims."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .claims import Claim, EvidenceSpan
from .schema import EvidenceRecord


@dataclass(frozen=True, slots=True)
class SpanRepair:
    """One quote-only offset repair bound to its source record revision."""

    evidence_index: int
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    record_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class AnchoringResult:
    """Reanchored claim plus repairs and evidence spans left unresolved."""

    claim: Claim
    repairs: tuple[SpanRepair, ...]
    unresolved: tuple[str, ...]


def anchor_unique_quotes(claim: Claim, records: Sequence[EvidenceRecord]) -> AnchoringResult:
    """Repair only uniquely located exact quotes in their declared source message.

    The input must already be a parsed Claim. This function changes offsets only;
    callers must still run ``verify_candidate`` before using the claim.
    """
    if type(claim) is not Claim:
        raise TypeError("claim must be a parsed Claim")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of EvidenceRecord values")
    if any(type(record) is not EvidenceRecord for record in records):
        raise TypeError("records must contain EvidenceRecord values")

    unique_records: dict[tuple[str, str], EvidenceRecord] = {}
    for record in records:
        unique_records.setdefault((record.record_id, record.revision), record)
    by_reference: dict[tuple[str, str], list[EvidenceRecord]] = {}
    for record in unique_records.values():
        key = (record.session_id, record.message_id)
        by_reference.setdefault(key, []).append(record)

    evidence: list[EvidenceSpan] = []
    repairs: list[SpanRepair] = []
    unresolved: list[str] = []
    for evidence_index, span in enumerate(claim.evidence):
        scoped_records = by_reference.get((span.session_id, span.message_id), [])
        if not scoped_records:
            evidence.append(span)
            unresolved.append(f"evidence[{evidence_index}]:unknown_reference")
            continue

        exact_records = [
            record
            for record in scoped_records
            if span.char_end <= len(record.text)
            and record.text[span.char_start : span.char_end] == span.quote
        ]
        if len(exact_records) == 1:
            evidence.append(span)
            continue

        occurrences: list[tuple[EvidenceRecord, int]] = []
        for record in scoped_records:
            offset = record.text.find(span.quote)
            while offset >= 0:
                occurrences.append((record, offset))
                if len(occurrences) > 1:
                    break
                offset = record.text.find(span.quote, offset + 1)
            if len(occurrences) > 1:
                break

        if len(occurrences) == 1:
            record, new_start = occurrences[0]
            new_end = new_start + len(span.quote)
            evidence.append(replace(span, char_start=new_start, char_end=new_end))
            repairs.append(
                SpanRepair(
                    evidence_index=evidence_index,
                    old_start=span.char_start,
                    old_end=span.char_end,
                    new_start=new_start,
                    new_end=new_end,
                    record_id=record.record_id,
                    revision=record.revision,
                )
            )
            continue

        evidence.append(span)
        reason = "quote_not_found" if not occurrences else "ambiguous_quote_occurrences"
        unresolved.append(f"evidence[{evidence_index}]:{reason}")

    return AnchoringResult(
        replace(claim, evidence=tuple(evidence)), tuple(repairs), tuple(unresolved)
    )


__all__ = ["AnchoringResult", "SpanRepair", "anchor_unique_quotes"]
