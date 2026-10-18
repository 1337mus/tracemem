"""Tests for provenance-linked Gregorian calendar arithmetic."""

import unittest
from dataclasses import FrozenInstanceError

from tracemem.calendar import calendar_day_difference
from tracemem.claims import EvidenceSpan
from tracemem.schema import EvidenceRecord


def _record(
    text: str, *, session_id: str = "calendar-session", recorded_at: str | None = None
) -> EvidenceRecord:
    return EvidenceRecord(
        source="synthetic",
        session_id=session_id,
        message_id=f"message-{text}",
        part_index=0,
        role="user",
        text=text,
        recorded_at=recorded_at,
    )


def _span(record: EvidenceRecord, quote: str) -> EvidenceSpan:
    start = record.text.index(quote)
    return EvidenceSpan(record.session_id, record.message_id, start, start + len(quote), quote)


class CalendarDayDifferenceTests(unittest.TestCase):
    def test_calculates_gregorian_days_across_month_and_leap_day(self) -> None:
        cases = (("2024-02-28", "2024-03-01", 2), ("2024-12-31", "2025-01-01", 1))
        for start_date, end_date, expected in cases:
            with self.subTest(start=start_date, end=end_date):
                start_record = _record(f"I began on {start_date}.")
                end_record = _record(f"I finished on {end_date}.")

                result = calendar_day_difference(
                    start_record,
                    _span(start_record, start_date),
                    end_record,
                    _span(end_record, end_date),
                )

                self.assertEqual(result.day_count, expected)

    def test_allows_same_and_reversed_calendar_days(self) -> None:
        cases = (("2024-03-01", "2024-03-01", 0), ("2024-03-01", "2024-02-28", -2))
        for start_date, end_date, expected in cases:
            with self.subTest(start=start_date, end=end_date):
                start_record = _record(f"Start: {start_date}")
                end_record = _record(f"End: {end_date}")

                result = calendar_day_difference(
                    start_record,
                    _span(start_record, start_date),
                    end_record,
                    _span(end_record, end_date),
                )

                self.assertEqual(result.day_count, expected)

    def test_requires_canonical_date_quotes_and_ignores_recorded_at(self) -> None:
        invalid_quotes = (
            "2024-02-28T12:00:00Z",
            "20240228",
            "2024-02-30",
            "yesterday",
        )
        end_record = _record("End: 2024-03-01")
        valid_end = _span(end_record, "2024-03-01")
        for quote in invalid_quotes:
            with self.subTest(quote=quote):
                record = _record(f"Source says {quote}.", recorded_at="2024-02-28T12:00:00Z")
                with self.assertRaisesRegex(ValueError, "canonical YYYY-MM-DD calendar date"):
                    calendar_day_difference(record, _span(record, quote), end_record, valid_end)

    def test_preserves_provenance_across_sessions_and_records(self) -> None:
        start_record = _record("Opened on 2024-02-28.", session_id="session-before")
        end_record = _record("Returned on 2024-03-01.", session_id="session-after")
        start_span = _span(start_record, "2024-02-28")
        end_span = _span(end_record, "2024-03-01")

        result = calendar_day_difference(start_record, start_span, end_record, end_span)

        self.assertEqual(result.day_count, 2)
        self.assertIs(result.start.span, start_span)
        self.assertIs(result.end.span, end_span)
        self.assertEqual(result.start.span.session_id, "session-before")
        self.assertEqual(result.end.span.session_id, "session-after")
        self.assertNotEqual(result.start.record_id, result.end.record_id)
        self.assertEqual(result.start.revision, start_record.revision)
        self.assertEqual(result.end.revision, end_record.revision)

    def test_rejects_spans_with_wrong_identity_quote_or_bounds(self) -> None:
        record = _record("The event date is 2024-02-28.")
        end_record = _record("The end date is 2024-03-01.")
        valid_start = _span(record, "2024-02-28")
        valid_end = _span(end_record, "2024-03-01")
        start = record.text.index("2024-02-28")
        invalid_spans = (
            (
                EvidenceSpan("other-session", record.message_id, start, start + 10, "2024-02-28"),
                "identifiers must match",
            ),
            (
                EvidenceSpan(record.session_id, "other-message", start, start + 10, "2024-02-28"),
                "identifiers must match",
            ),
            (
                EvidenceSpan(record.session_id, record.message_id, start, start + 10, "2024-02-29"),
                "quote must exactly match",
            ),
            (
                EvidenceSpan(
                    record.session_id, record.message_id, start, len(record.text) + 1, "2024-02-28"
                ),
                "outside record text bounds",
            ),
        )
        for invalid_span, error in invalid_spans:
            with self.subTest(endpoint="start", error=error):
                with self.assertRaisesRegex(ValueError, error):
                    calendar_day_difference(record, invalid_span, end_record, valid_end)

        end_start = end_record.text.index("2024-03-01")
        invalid_end_spans = (
            (
                EvidenceSpan(
                    "other-session", end_record.message_id, end_start, end_start + 10, "2024-03-01"
                ),
                "end span identifiers must match",
            ),
            (
                EvidenceSpan(
                    end_record.session_id,
                    end_record.message_id,
                    end_start,
                    end_start + 10,
                    "2024-02-29",
                ),
                "end span quote must exactly match",
            ),
            (
                EvidenceSpan(
                    end_record.session_id,
                    end_record.message_id,
                    end_start,
                    len(end_record.text) + 1,
                    "2024-03-01",
                ),
                "end span is outside record text bounds",
            ),
        )
        for invalid_span, error in invalid_end_spans:
            with self.subTest(endpoint="end", error=error):
                with self.assertRaisesRegex(ValueError, error):
                    calendar_day_difference(record, valid_start, end_record, invalid_span)

    def test_preserves_endpoint_span_and_record_revision_provenance(self) -> None:
        start_record = _record("🪐 plan starts 2024-02-28.")
        end_record = _record("🪐 plan ends 2024-03-01.")
        start_span = _span(start_record, "2024-02-28")
        end_span = _span(end_record, "2024-03-01")
        self.assertEqual(start_span.char_start, len("🪐 plan starts "))

        result = calendar_day_difference(start_record, start_span, end_record, end_span)

        self.assertIs(result.start.span, start_span)
        self.assertEqual(result.start.record_id, start_record.record_id)
        self.assertEqual(result.start.revision, start_record.revision)
        self.assertEqual(result.start.iso_date, "2024-02-28")
        self.assertIs(result.end.span, end_span)
        self.assertEqual(result.end.record_id, end_record.record_id)
        self.assertEqual(result.end.revision, end_record.revision)
        self.assertEqual(result.end.iso_date, "2024-03-01")
        self.assertFalse(hasattr(result, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            result.__setattr__("day_count", 99)


if __name__ == "__main__":
    unittest.main()
