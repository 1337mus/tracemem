"""Tests for bounded literal-date arithmetic evidence rendering."""

from __future__ import annotations

import json
import unittest

from experiments.calendar_evidence import render_calendar_evidence
from tracemem.schema import EvidenceRecord


def _record(
    text: str,
    message_id: str,
    *,
    session_id: str = "calendar-session",
    recorded_at: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        source="synthetic",
        session_id=session_id,
        message_id=message_id,
        part_index=0,
        role="user",
        text=text,
        recorded_at=recorded_at,
    )


class CalendarEvidenceRenderingTests(unittest.TestCase):
    def test_emits_every_ordered_pair_with_exact_provenance_and_signed_days(self) -> None:
        first = _record(
            "🧪 Secret note at /private/should-not-appear: 2024-03-01 then 2024-02-28",
            "message-first",
            session_id="session-before",
        )
        second = _record("Return date 2024-03-02", "message-second", session_id="session-after")
        records = (first, second)

        payload = json.loads(render_calendar_evidence(records))
        pairs = payload["pairs"]

        self.assertEqual(len(pairs), 3)
        self.assertEqual([pair["day_count"] for pair in pairs], [-2, 1, 3])
        self.assertEqual(
            [(pair["start"]["quote"], pair["end"]["quote"]) for pair in pairs],
            [
                ("2024-03-01", "2024-02-28"),
                ("2024-03-01", "2024-03-02"),
                ("2024-02-28", "2024-03-02"),
            ],
        )
        first_occurrence = pairs[0]["start"]
        second_occurrence = pairs[0]["end"]
        self.assertEqual(first_occurrence["char_start"], first.text.index("2024-03-01"))
        self.assertEqual(first_occurrence["char_end"], first_occurrence["char_start"] + 10)
        self.assertEqual(second_occurrence["char_start"], first.text.index("2024-02-28"))
        self.assertEqual(first_occurrence["record_id"], first.record_id)
        self.assertEqual(first_occurrence["revision"], first.revision)
        self.assertEqual(second_occurrence["record_id"], first.record_id)
        self.assertEqual(pairs[1]["end"]["record_id"], second.record_id)
        self.assertEqual(pairs[1]["end"]["revision"], second.revision)
        rendered = render_calendar_evidence(records)
        self.assertEqual(rendered, render_calendar_evidence(records))
        self.assertNotIn("Secret note", rendered)
        self.assertNotIn("/private/should-not-appear", rendered)
        self.assertNotIn("session-before", rendered)
        self.assertIn("does not identify events", payload["scope"])
        self.assertIn("completion", payload["scope"])
        self.assertIn("attribution", payload["scope"])

    def test_scans_only_standalone_possible_canonical_dates_in_text(self) -> None:
        record = _record(
            "2024-02-30 20240229 x2024-02-29 2024-02-29x "
            "x_2024-02-29 2024-02-29_suffix 2024-02-29-extra "
            "2024-02-29T12:00:00Z 2024-02-29 12:00:00 and 2024-02-29.",
            "message-filtered",
            recorded_at="2030-01-01",
        )

        payload = json.loads(render_calendar_evidence((record,)))

        self.assertEqual(payload["pairs"], [])

    def test_returns_empty_pairs_with_fewer_than_two_valid_occurrences(self) -> None:
        self.assertEqual(json.loads(render_calendar_evidence(()))["pairs"], [])
        one_date = _record("A literal 2024-02-29 date", "message-one-date")
        self.assertEqual(json.loads(render_calendar_evidence((one_date,)))["pairs"], [])

    def test_allows_eight_occurrences_and_fails_closed_above_the_limit(self) -> None:
        eight_dates = _record(
            " ".join(f"2024-01-{day:02d}" for day in range(1, 9)),
            "message-eight-dates",
        )
        payload = json.loads(render_calendar_evidence((eight_dates,)))
        self.assertEqual(len(payload["pairs"]), 28)

        nine_dates = _record(
            " ".join(f"2024-01-{day:02d}" for day in range(1, 10)),
            "message-nine-dates",
        )
        with self.assertRaisesRegex(ValueError, "at most 8"):
            render_calendar_evidence((nine_dates,))


if __name__ == "__main__":
    unittest.main()
