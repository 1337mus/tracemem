"""Pure source-retention impact diagnostics for explicit in-memory snapshots."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from .citations import _CITATION_RE
from .claims import Claim, EvidenceSpan, VerificationResult, _validate_claim
from .ledger import CandidateEntry
from .schema import EvidenceRecord

_RecordKey = tuple[str, str]
_SpanKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ErasureImpact:
    """Snapshot dependencies for a source/session selector, not an erase plan."""

    selected_record_revisions: tuple[_RecordKey, ...]
    direct_candidate_ids: tuple[str, ...]
    ambiguous_candidate_ids: tuple[str, ...]
    dependent_candidate_ids: tuple[str, ...]
    affected_candidate_ids: tuple[str, ...]
    diagnostic_kind: str = "snapshot_dependency_diagnostic"
    deletion_performed: bool = False
    vector_cache_scope: str = (
        "vector-cache keys correspond to selected record/revision pairs; vector-cache counts "
        "and physical deletion are not audited"
    )
    undiscoverable_locations: tuple[str, ...] = (
        "external exports",
        "backups",
        "unlinked copies",
        "free SQLite pages",
        "in-memory caches",
    )
    complete_snapshot_input_requirement: str = (
        "include all historical evidence and ledger entries for a complete local snapshot"
    )
    semantic_dependence_guaranteed: bool = False
    complete_information_flow_detection_guaranteed: bool = False
    disclosures: tuple[str, ...] = (
        "This is a snapshot dependency diagnostic; deletion is not performed.",
        "Vector-cache keys correspond to selected record/revision pairs, but cache counts "
        "and physical deletion are not audited.",
        "External exports, backups, unlinked copies, free SQLite pages, and in-memory caches "
        "cannot be discovered.",
        "A complete local snapshot requires all historical evidence records and candidate "
        "ledger entries.",
        "Semantic dependence is not guaranteed, and complete information-flow detection is "
        "not guaranteed.",
    )


def _utf8_nonempty(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _utf8_text(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _as_sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _validate_selector(value: object) -> str:
    if type(value) is not str:
        raise TypeError("source and session selectors must be nonempty UTF-8 strings")
    if not _utf8_nonempty(value):
        raise ValueError("source and session selectors must be nonempty UTF-8 strings")
    return value


def _validate_record(value: object) -> EvidenceRecord:
    if type(value) is not EvidenceRecord:
        raise TypeError("records must contain EvidenceRecord values")
    record = value
    if not all(
        _utf8_nonempty(getattr(record, field)) for field in ("source", "session_id", "message_id")
    ):
        raise ValueError("records contain an invalid EvidenceRecord")
    if not all(_utf8_text(getattr(record, field)) for field in ("role", "text")):
        raise ValueError("records contain an invalid EvidenceRecord")
    if type(record.part_index) is not int or record.part_index < 0:
        raise ValueError("records contain an invalid EvidenceRecord")
    if (
        type(record.role) is not str
        or type(record.content_kind) is not str
        or record.role not in EvidenceRecord.ROLES
        or record.content_kind not in EvidenceRecord.CONTENT_KINDS
    ):
        raise ValueError("records contain an invalid EvidenceRecord")
    if record.recorded_at is not None and not _utf8_text(record.recorded_at):
        raise ValueError("records contain an invalid EvidenceRecord")
    if record.source_locator is not None and not _utf8_text(record.source_locator):
        raise ValueError("records contain an invalid EvidenceRecord")
    if record.line_number is not None and (
        type(record.line_number) is not int or record.line_number < 1
    ):
        raise ValueError("records contain an invalid EvidenceRecord")
    try:
        _ = (record.record_id, record.revision)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("records contain an invalid EvidenceRecord") from None
    return record


def _canonical_id(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_span(value: object) -> EvidenceSpan:
    if type(value) is not EvidenceSpan:
        raise TypeError("candidate entries contain an invalid evidence span")
    span = value
    if not _utf8_nonempty(span.session_id) or not _utf8_nonempty(span.message_id):
        raise ValueError("candidate entries contain an invalid evidence span")
    if type(span.char_start) is not int or type(span.char_end) is not int:
        raise TypeError("candidate entries contain an invalid evidence span")
    if span.char_start < 0 or span.char_end <= span.char_start or not _utf8_nonempty(span.quote):
        raise ValueError("candidate entries contain an invalid evidence span")
    return span


def _parse_citation(value: object, record_by_key: dict[_RecordKey, EvidenceRecord]) -> _RecordKey:
    if type(value) is not str:
        raise ValueError("candidate verification contains an invalid citation")
    match = _CITATION_RE.fullmatch(value)
    if match is None:
        raise ValueError("candidate verification contains an invalid citation")
    record_id = match.group("record_id")
    revision = match.group("revision")
    try:
        char_start = int(match.group("char_start"))
        char_end = int(match.group("char_end"))
    except ValueError:
        raise ValueError("candidate verification contains an invalid citation") from None
    if char_end <= char_start:
        raise ValueError("candidate verification contains an invalid citation")
    key = (record_id, revision)
    cited_record = record_by_key.get(key)
    if cited_record is not None and char_end > len(cited_record.text):
        raise ValueError("candidate verification contains an invalid citation")
    return key


def _validate_entry(
    value: object, record_by_key: dict[_RecordKey, EvidenceRecord]
) -> tuple[CandidateEntry, tuple[_RecordKey, ...], tuple[_SpanKey, ...]]:
    if type(value) is not CandidateEntry:
        raise TypeError("entries must contain CandidateEntry values")
    entry = value
    if not _canonical_id(entry.entry_id):
        raise ValueError("candidate entries contain an invalid entry ID")
    if type(entry.claim) is not Claim:
        raise TypeError("candidate entries contain an invalid claim")
    try:
        _validate_claim(entry.claim)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("candidate entries contain an invalid claim") from None
    if type(entry.claim.evidence) is not tuple or not entry.claim.evidence:
        raise ValueError("candidate entries contain an invalid claim")
    spans = tuple(_validate_span(span) for span in entry.claim.evidence)
    if type(entry.verification) is not VerificationResult:
        raise TypeError("candidate entries contain an invalid verification result")
    if type(entry.verification.status) is not str or entry.verification.status not in {
        "REVIEW",
        "REJECT",
    }:
        raise ValueError("candidate entries contain an invalid verification result")
    if type(entry.verification.errors) is not tuple or any(
        not _utf8_text(error) for error in entry.verification.errors
    ):
        raise ValueError("candidate entries contain an invalid verification result")
    if type(entry.verification.citations) is not tuple:
        raise TypeError("candidate entries contain an invalid verification result")
    citation_keys = tuple(
        _parse_citation(citation, record_by_key) for citation in entry.verification.citations
    )
    if not _utf8_nonempty(entry.extractor_id) or not _utf8_nonempty(entry.recorded_at):
        raise ValueError("candidate entries contain invalid metadata")
    if type(entry.supersedes) is not tuple:
        raise TypeError("candidate entries contain invalid lineage")
    if any(not _canonical_id(parent_id) for parent_id in entry.supersedes):
        raise ValueError("candidate entries contain invalid lineage")
    if len(set(entry.supersedes)) != len(entry.supersedes):
        raise ValueError("candidate entries contain invalid lineage")
    span_keys = tuple((span.session_id, span.message_id) for span in spans)
    return entry, citation_keys, span_keys


def _validate_lineage(entries_by_id: dict[str, CandidateEntry]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {entry_id: [] for entry_id in entries_by_id}
    parent_count = {entry_id: 0 for entry_id in entries_by_id}
    for entry in entries_by_id.values():
        parent_count[entry.entry_id] = len(entry.supersedes)
        for parent_id in entry.supersedes:
            if parent_id not in entries_by_id:
                raise ValueError("candidate entries contain dangling lineage")
            children[parent_id].append(entry.entry_id)

    pending = deque(entry_id for entry_id, count in parent_count.items() if count == 0)
    visited = 0
    while pending:
        parent_id = pending.popleft()
        visited += 1
        for child_id in children[parent_id]:
            parent_count[child_id] -= 1
            if parent_count[child_id] == 0:
                pending.append(child_id)
    if visited != len(entries_by_id):
        raise ValueError("candidate entries contain cyclic lineage")
    return children


def plan_source_erasure(
    records: Sequence[EvidenceRecord],
    entries: Sequence[CandidateEntry],
    *,
    source: str,
    session_id: str,
) -> ErasureImpact:
    """Return dependency impact for an exact source/session in an explicit snapshot."""
    selected_source = _validate_selector(source)
    selected_session = _validate_selector(session_id)

    raw_records = _as_sequence(records, "records")
    validated_records = tuple(_validate_record(record) for record in raw_records)
    record_by_key: dict[_RecordKey, EvidenceRecord] = {}
    for record in validated_records:
        key = (record.record_id, record.revision)
        if key in record_by_key:
            raise ValueError("records contain duplicate record/revision keys")
        record_by_key[key] = record

    raw_entries = _as_sequence(entries, "entries")
    validated_entries: list[CandidateEntry] = []
    parsed_citations: dict[str, tuple[_RecordKey, ...]] = {}
    parsed_spans: dict[str, tuple[_SpanKey, ...]] = {}
    entries_by_id: dict[str, CandidateEntry] = {}
    for raw_entry in raw_entries:
        entry, citations, spans = _validate_entry(raw_entry, record_by_key)
        if entry.entry_id in entries_by_id:
            raise ValueError("entries contain duplicate candidate IDs")
        entries_by_id[entry.entry_id] = entry
        validated_entries.append(entry)
        parsed_citations[entry.entry_id] = citations
        parsed_spans[entry.entry_id] = spans
    children = _validate_lineage(entries_by_id)

    selected_records = tuple(
        sorted(
            (record.record_id, record.revision)
            for record in validated_records
            if record.source == selected_source and record.session_id == selected_session
        )
    )
    selected_keys = set(selected_records)
    selected_spans = {
        (record.session_id, record.message_id)
        for record in validated_records
        if record.source == selected_source and record.session_id == selected_session
    }

    direct_ids: set[str] = set()
    ambiguous_ids: set[str] = set()
    for entry in validated_entries:
        entry_id = entry.entry_id
        if selected_keys.intersection(parsed_citations[entry_id]):
            direct_ids.add(entry_id)
        elif selected_spans.intersection(parsed_spans[entry_id]):
            ambiguous_ids.add(entry_id)

    descendant_ids: set[str] = set()
    pending = deque(sorted(direct_ids | ambiguous_ids))
    while pending:
        parent_id = pending.popleft()
        for child_id in children[parent_id]:
            if child_id not in descendant_ids:
                descendant_ids.add(child_id)
                pending.append(child_id)

    dependent_ids = descendant_ids - direct_ids - ambiguous_ids
    affected_ids = direct_ids | ambiguous_ids | dependent_ids
    return ErasureImpact(
        selected_record_revisions=selected_records,
        direct_candidate_ids=tuple(sorted(direct_ids)),
        ambiguous_candidate_ids=tuple(sorted(ambiguous_ids)),
        dependent_candidate_ids=tuple(sorted(dependent_ids)),
        affected_candidate_ids=tuple(sorted(affected_ids)),
    )


__all__ = ["ErasureImpact", "plan_source_erasure"]
