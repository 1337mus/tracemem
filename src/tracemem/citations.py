"""Exact revision-bound retrieval for candidate claim citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .claims import Claim, VerificationResult, verify_candidate
from .ledger import CandidateEntry
from .schema import EvidenceRecord
from .storage import EvidenceStore

_CITATION_RE = re.compile(
    r"(?P<record_id>[0-9a-f]{64})@(?P<revision>[0-9a-f]{64}):"
    r"(?P<char_start>0|[1-9][0-9]*):(?P<char_end>0|[1-9][0-9]*)\Z"
)


@dataclass(frozen=True, slots=True)
class ResolvedCitation:
    """One citation resolved to the exact immutable source revision and span."""

    citation: str
    record: EvidenceRecord
    char_start: int
    char_end: int
    quote: str


def fetch_citation(store: EvidenceStore, citation: str) -> ResolvedCitation:
    """Resolve one canonical citation without falling back to another revision."""
    if type(citation) is not str:
        raise ValueError("citation must use the canonical record@revision:start:end format")
    match = _CITATION_RE.fullmatch(citation)
    if match is None:
        raise ValueError("citation must use the canonical record@revision:start:end format")

    record_id = match.group("record_id")
    revision = match.group("revision")
    char_start = int(match.group("char_start"))
    char_end = int(match.group("char_end"))
    if char_end <= char_start:
        raise ValueError("citation span must be non-empty")

    record = store.get(record_id, revision)
    if record.record_id != record_id or record.revision != revision:
        raise ValueError("citation identity does not match the stored evidence record")
    if char_end > len(record.text):
        raise ValueError("citation span is out of bounds")

    return ResolvedCitation(
        citation=citation,
        record=record,
        char_start=char_start,
        char_end=char_end,
        quote=record.text[char_start:char_end],
    )


def fetch_entry_evidence(
    store: EvidenceStore,
    entry: CandidateEntry,
) -> tuple[ResolvedCitation, ...]:
    """Resolve and recheck every citation in a REVIEW entry, in claim order."""
    if type(entry) is not CandidateEntry:
        raise TypeError("entry must be a CandidateEntry")
    if type(entry.verification) is not VerificationResult:
        raise ValueError("candidate verification is malformed")
    if entry.verification.status == "REJECT":
        raise ValueError("REJECT candidates do not have resolvable claim evidence")
    if entry.verification.status != "REVIEW" or entry.verification.errors:
        raise ValueError("candidate must have an integrity-clean REVIEW result")
    if type(entry.claim) is not Claim:
        raise ValueError("candidate claim is malformed")

    spans = entry.claim.evidence
    citations = entry.verification.citations
    if len(citations) != len(spans):
        raise ValueError("candidate must have exactly one citation per evidence span")

    resolved = tuple(fetch_citation(store, citation) for citation in citations)
    for citation, span in zip(resolved, spans, strict=True):
        record = citation.record
        if (
            record.session_id != span.session_id
            or record.message_id != span.message_id
            or citation.char_start != span.char_start
            or citation.char_end != span.char_end
            or citation.quote != span.quote
        ):
            raise ValueError("resolved citation does not match its claim evidence span")

    recomputed = verify_candidate(entry.claim, tuple(item.record for item in resolved))
    if recomputed.status != "REVIEW" or recomputed != entry.verification:
        raise ValueError("candidate verification does not match the resolved source evidence")
    return resolved
