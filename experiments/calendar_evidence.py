"""Deterministic evidence-only calendar arithmetic rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date

from tracemem.calendar import CalendarDayEndpoint, calendar_day_difference
from tracemem.claims import EvidenceSpan
from tracemem.schema import EvidenceRecord

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIMESTAMP_SUFFIX = re.compile(
    r"(?:T| )[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:Z|[+-][0-9]{2}:?[0-9]{2})?"
)
_MAX_OCCURRENCES = 8


def render_calendar_evidence(records: Sequence[EvidenceRecord]) -> str:
    """Render deterministic, provenance-linked arithmetic over literal text dates."""
    occurrences: list[tuple[EvidenceRecord, EvidenceSpan]] = []
    for record in records:
        if type(record) is not EvidenceRecord:
            raise TypeError("records must contain only EvidenceRecord values")
        for match in _DATE.finditer(record.text):
            start, end = match.span()
            if (start > 0 and _is_token_character(record.text[start - 1])) or (
                end < len(record.text) and _is_token_character(record.text[end])
            ):
                continue
            if _TIMESTAMP_SUFFIX.match(record.text, end) is not None:
                continue
            quote = match.group()
            try:
                parsed = date.fromisoformat(quote)
            except ValueError:
                continue
            if parsed.isoformat() != quote:
                continue
            occurrences.append(
                (
                    record,
                    EvidenceSpan(record.session_id, record.message_id, start, end, quote),
                )
            )
            if len(occurrences) > _MAX_OCCURRENCES:
                raise ValueError("calendar evidence supports at most 8 valid date occurrences")

    pairs: list[dict[str, object]] = []
    for left_index, (left_record, left_span) in enumerate(occurrences):
        for right_record, right_span in occurrences[left_index + 1 :]:
            difference = calendar_day_difference(left_record, left_span, right_record, right_span)
            pairs.append(
                {
                    "start": _endpoint_payload(difference.start),
                    "end": _endpoint_payload(difference.end),
                    "day_count": difference.day_count,
                }
            )

    payload = {
        "scope": (
            "Calendar arithmetic only compares literal dates; it does not identify events "
            "or establish completion or attribution."
        ),
        "pairs": pairs,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _endpoint_payload(endpoint: CalendarDayEndpoint) -> dict[str, object]:
    span = endpoint.span
    return {
        "record_id": endpoint.record_id,
        "revision": endpoint.revision,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "quote": span.quote,
    }


def _is_token_character(character: str) -> bool:
    return character.isalnum() or character in "_-"
