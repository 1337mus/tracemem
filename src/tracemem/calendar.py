"""Provenance-linked Gregorian calendar day arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .claims import EvidenceSpan
from .schema import EvidenceRecord

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True, slots=True)
class CalendarDayEndpoint:
    """A date quote tied to its exact evidence span and source revision."""

    span: EvidenceSpan
    record_id: str
    revision: str
    iso_date: str


@dataclass(frozen=True, slots=True)
class CalendarDayDifference:
    """Signed difference between two validated evidence dates."""

    start: CalendarDayEndpoint
    end: CalendarDayEndpoint
    day_count: int


def calendar_day_difference(
    start_record: EvidenceRecord,
    start_span: EvidenceSpan,
    end_record: EvidenceRecord,
    end_span: EvidenceSpan,
) -> CalendarDayDifference:
    """Return Gregorian end-minus-start days for exact, date-valued evidence spans."""
    start, start_date = _validated_endpoint(start_record, start_span, "start")
    end, end_date = _validated_endpoint(end_record, end_span, "end")
    return CalendarDayDifference(start, end, (end_date - start_date).days)


def _validated_endpoint(
    record: EvidenceRecord, span: EvidenceSpan, name: str
) -> tuple[CalendarDayEndpoint, date]:
    if type(record) is not EvidenceRecord:
        raise TypeError(f"{name}_record must be an EvidenceRecord")
    if type(span) is not EvidenceSpan:
        raise TypeError(f"{name}_span must be an EvidenceSpan")
    if span.session_id != record.session_id or span.message_id != record.message_id:
        raise ValueError(f"{name} span identifiers must match its EvidenceRecord")
    if (
        type(span.char_start) is not int
        or type(span.char_end) is not int
        or span.char_start < 0
        or span.char_end <= span.char_start
        or span.char_end > len(record.text)
    ):
        raise ValueError(f"{name} span is outside record text bounds")
    if record.text[span.char_start : span.char_end] != span.quote:
        raise ValueError(f"{name} span quote must exactly match its source slice")
    if _ISO_DATE.fullmatch(span.quote) is None:
        raise ValueError(f"{name} span quote must be a canonical YYYY-MM-DD calendar date")
    try:
        parsed = date.fromisoformat(span.quote)
    except ValueError:
        raise ValueError(
            f"{name} span quote must be a canonical YYYY-MM-DD calendar date"
        ) from None
    if parsed.isoformat() != span.quote:
        raise ValueError(f"{name} span quote must be a canonical YYYY-MM-DD calendar date")
    endpoint = CalendarDayEndpoint(span, record.record_id, record.revision, parsed.isoformat())
    return endpoint, parsed
