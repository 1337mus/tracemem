"""Strict candidate claim parsing and evidence-span verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .schema import EvidenceRecord

MAX_CLAIMS = 64
MAX_EVIDENCE_SPANS = 64
MAX_IDENTIFIER_CHARS = 256
MAX_TERM_CHARS = 2_048
MAX_QUOTE_CHARS = 8_192
MAX_CLAIM_TEXT_CHARS = 32_768
MAX_PAYLOAD_TEXT_CHARS = 1_000_000

PREDICATES = frozenset("PREFERS USES DECIDED PLANS COMPLETED REPORTS LOCATED_IN CAUSES HAS".split())
SPEAKER_ROLES = frozenset({"user", "assistant", "tool", "unknown"})
ATTRIBUTIONS = frozenset(
    "direct_user_statement assistant_statement tool_observation "
    "third_party_report quoted_text unclear".split()
)
MODALITIES = frozenset(
    "asserted tentative preference intention hypothetical question "
    "reported completed historical".split()
)
POLARITIES = frozenset({"positive", "negative"})

_PAYLOAD_KEYS = frozenset({"claims"})
_CLAIM_KEYS = frozenset(
    "subject predicate object speaker_role attribution modality polarity "
    "valid_from valid_to evidence".split()
)
_SPAN_KEYS = frozenset("session_id message_id char_start char_end quote".split())
_ATTRIBUTION_ROLES = {
    "direct_user_statement": "user",
    "assistant_statement": "assistant",
    "tool_observation": "tool",
}


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    session_id: str
    message_id: str
    char_start: int
    char_end: int
    quote: str

    def __post_init__(self) -> None:
        _validate_span(self)


@dataclass(frozen=True, slots=True)
class Claim:
    subject: str
    predicate: str
    object: str
    speaker_role: str
    attribution: str
    modality: str
    polarity: str
    valid_from: str | None
    valid_to: str | None
    evidence: tuple[EvidenceSpan, ...]

    def __post_init__(self) -> None:
        _validate_claim(self)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: Literal["REJECT", "REVIEW"]
    errors: tuple[str, ...]
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in {"REJECT", "REVIEW"}:
            raise ValueError("status must be REJECT or REVIEW")
        for name, values in (("errors", self.errors), ("citations", self.citations)):
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise TypeError(f"{name} must be a tuple of strings")


def parse_candidates(payload: object) -> tuple[Claim, ...]:
    """Parse a candidate payload. The caller is responsible for JSON decoding."""
    _require_keys(payload, _PAYLOAD_KEYS, "payload")
    assert type(payload) is dict
    raw_claims = payload["claims"]
    if type(raw_claims) is not list:
        raise ValueError("claims must be a list")
    if len(raw_claims) > MAX_CLAIMS:
        raise ValueError("too many claims")

    parsed: list[Claim] = []
    total_text_chars = 0
    for raw_claim in raw_claims:
        _require_keys(raw_claim, _CLAIM_KEYS, "claim")
        assert type(raw_claim) is dict
        raw_evidence = raw_claim["evidence"]
        if type(raw_evidence) is not list:
            raise ValueError("claim evidence must be a list")
        if len(raw_evidence) > MAX_EVIDENCE_SPANS:
            raise ValueError("too many evidence spans")
        evidence: list[EvidenceSpan] = []
        for raw_span in raw_evidence:
            _require_keys(raw_span, _SPAN_KEYS, "evidence span")
            assert type(raw_span) is dict
            evidence.append(
                EvidenceSpan(
                    session_id=raw_span["session_id"],
                    message_id=raw_span["message_id"],
                    char_start=raw_span["char_start"],
                    char_end=raw_span["char_end"],
                    quote=raw_span["quote"],
                )
            )
        parsed.append(
            Claim(
                subject=raw_claim["subject"],
                predicate=raw_claim["predicate"],
                object=raw_claim["object"],
                speaker_role=raw_claim["speaker_role"],
                attribution=raw_claim["attribution"],
                modality=raw_claim["modality"],
                polarity=raw_claim["polarity"],
                valid_from=raw_claim["valid_from"],
                valid_to=raw_claim["valid_to"],
                evidence=tuple(evidence),
            )
        )
        total_text_chars += _claim_text_chars(parsed[-1])
        if total_text_chars > MAX_PAYLOAD_TEXT_CHARS:
            raise ValueError("candidate payload text is too large")
    return tuple(parsed)


def verify_candidate(claim: Claim, records: Sequence[EvidenceRecord]) -> VerificationResult:
    """Verify exact source spans and roles, without judging semantic entailment."""
    if type(claim) is not Claim:
        return VerificationResult("REJECT", ("invalid_claim",), ())
    try:
        _validate_claim(claim)
    except (TypeError, ValueError):
        return VerificationResult("REJECT", ("invalid_claim",), ())
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return VerificationResult("REJECT", ("invalid_records",), ())
    if any(not isinstance(record, EvidenceRecord) for record in records):
        return VerificationResult("REJECT", ("invalid_records",), ())

    errors: list[str] = []
    citations: list[str] = []
    required_role = _ATTRIBUTION_ROLES.get(claim.attribution)
    if required_role is not None and claim.speaker_role != required_role:
        errors.append("attribution_role_conflict")

    for span in claim.evidence:
        source_records = [
            record
            for record in records
            if record.session_id == span.session_id and record.message_id == span.message_id
        ]
        if not source_records:
            errors.append("unknown_reference")
            continue

        matches: dict[tuple[str, str], EvidenceRecord] = {}
        in_bounds = False
        for record in source_records:
            if span.char_end > len(record.text):
                continue
            in_bounds = True
            if record.text[span.char_start : span.char_end] == span.quote:
                matches[(record.record_id, record.revision)] = record

        if not matches:
            errors.append("quote_mismatch" if in_bounds else "span_out_of_bounds")
            continue
        if len(matches) > 1:
            errors.append("ambiguous_reference")
            continue

        record = next(iter(matches.values()))
        if record.role != claim.speaker_role:
            errors.append("source_role_mismatch")
        citations.append(f"{record.record_id}@{record.revision}:{span.char_start}:{span.char_end}")

    return VerificationResult("REJECT" if errors else "REVIEW", tuple(errors), tuple(citations))


def _require_keys(value: object, expected: frozenset[str], name: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")


def _require_text(value: object, name: str, max_chars: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    if len(value) > max_chars:
        raise ValueError(f"{name} is too long")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} contains invalid Unicode") from None
    return value


def _require_enum(value: object, name: str, choices: frozenset[str]) -> str:
    text = _require_text(value, name, max(len(choice) for choice in choices))
    if text not in choices:
        raise ValueError(f"{name} has an invalid label")
    return text


def _validate_span(span: EvidenceSpan) -> None:
    _require_text(span.session_id, "session_id", MAX_IDENTIFIER_CHARS)
    _require_text(span.message_id, "message_id", MAX_IDENTIFIER_CHARS)
    if type(span.char_start) is not int or type(span.char_end) is not int:
        raise TypeError("span offsets must be integers")
    if span.char_start < 0 or span.char_end <= span.char_start:
        raise ValueError("span offsets must be non-negative and non-empty")
    _require_text(span.quote, "quote", MAX_QUOTE_CHARS)


def _validate_claim(claim: Claim) -> None:
    _require_text(claim.subject, "subject", MAX_TERM_CHARS)
    _require_enum(claim.predicate, "predicate", PREDICATES)
    _require_text(claim.object, "object", MAX_TERM_CHARS)
    _require_enum(claim.speaker_role, "speaker_role", SPEAKER_ROLES)
    _require_enum(claim.attribution, "attribution", ATTRIBUTIONS)
    _require_enum(claim.modality, "modality", MODALITIES)
    _require_enum(claim.polarity, "polarity", POLARITIES)
    valid_from = _validate_date(claim.valid_from, "valid_from")
    valid_to = _validate_date(claim.valid_to, "valid_to")
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise ValueError("valid_from must not be after valid_to")
    if type(claim.evidence) is not tuple:
        raise TypeError("evidence must be a tuple")
    if not claim.evidence:
        raise ValueError("claim evidence must not be empty")
    if len(claim.evidence) > MAX_EVIDENCE_SPANS:
        raise ValueError("too many evidence spans")
    for span in claim.evidence:
        if type(span) is not EvidenceSpan:
            raise TypeError("evidence must contain EvidenceSpan values")
        _validate_span(span)
    if _claim_text_chars(claim) > MAX_CLAIM_TEXT_CHARS:
        raise ValueError("claim text is too large")


def _validate_date(value: object, name: str) -> date | None:
    if value is None:
        return None
    text = _require_text(value, name, 10)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{name} must be an ISO date or null") from None
    if parsed.isoformat() != text:
        raise ValueError(f"{name} must be an ISO date or null")
    return parsed


def _claim_text_chars(claim: Claim) -> int:
    return (
        len(claim.subject)
        + len(claim.object)
        + sum(
            len(span.session_id) + len(span.message_id) + len(span.quote) for span in claim.evidence
        )
    )


__all__ = ["Claim", "EvidenceSpan", "VerificationResult", "parse_candidates", "verify_candidate"]
