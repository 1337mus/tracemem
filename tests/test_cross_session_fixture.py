"""Integrity checks for the provisional fictional cross-session test fixture."""

from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from experiments import evaluate_longmemeval_retrieval as evaluator
from tracemem.adapters.longmemeval import parse_longmemeval_bytes
from tracemem.citations import fetch_citation
from tracemem.storage import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/synthetic/cross-session-test-v1.json"
METADATA = ROOT / "examples/synthetic/cross-session-test-v1-metadata.json"
PRIOR_FIXTURES = (
    "chatgpt-export.json",
    "longmemeval-format.json",
    "longitudinal-dev-v1.json",
    "longitudinal-dev-v1-foils.json",
    "longitudinal-dev-v2.json",
    "longitudinal-dev-v2-foils.json",
    "longitudinal-dev-v2-unlabeled.json",
    "qa-diagnostic-v1.json",
    "qa-contrast-dev-v1.json",
    "qa-interpretation-controls-dev-v1.json",
)


def _all_strings(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {text for key, item in value.items() for text in ({key} | _all_strings(item))}
    if isinstance(value, list):
        return {text for item in value for text in _all_strings(item)}
    return {value} if isinstance(value, str) else set()


class CrossSessionFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FIXTURE.read_bytes()
        cls.items = json.loads(cls.raw)
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    def test_six_question_shape_dates_roles_and_label_anchors(self) -> None:
        self.assertEqual(len(self.items), 6)
        self.assertEqual(sum(len(s) for item in self.items for s in item["haystack_sessions"]), 72)
        self.assertEqual(len({item["question_id"] for item in self.items}), 6)
        session_ids: set[str] = set()
        answerable = [item for item in self.items if not item["question_id"].endswith("_abs")]
        unknown = [item for item in self.items if item["question_id"].endswith("_abs")]
        self.assertEqual((len(answerable), len(unknown)), (4, 2))

        for item in self.items:
            with self.subTest(question_id=item["question_id"]):
                self.assertEqual(len(item["haystack_sessions"]), 4)
                self.assertEqual(len(item["haystack_session_ids"]), 4)
                self.assertEqual(len(item["haystack_dates"]), 4)
                self.assertEqual(len(set(item["haystack_session_ids"])), 4)
                self.assertFalse(session_ids.intersection(item["haystack_session_ids"]))
                session_ids.update(item["haystack_session_ids"])
                dates = [date.fromisoformat(value) for value in item["haystack_dates"]]
                self.assertEqual(dates, sorted(dates))
                self.assertGreaterEqual((dates[-1] - dates[0]).days, 30)
                self.assertGreater(date.fromisoformat(item["question_date"]), dates[-1])
                self.assertTrue(all(len(session) == 3 for session in item["haystack_sessions"]))
                self.assertTrue(
                    all(
                        turn["role"] in {"user", "assistant"}
                        for session in item["haystack_sessions"]
                        for turn in session
                    )
                )
                supports = [
                    (index, turn_index, turn)
                    for index, session in enumerate(item["haystack_sessions"])
                    for turn_index, turn in enumerate(session, start=1)
                    if turn.get("has_answer") is True
                ]
                support_sessions = [item["haystack_session_ids"][index] for index, _, _ in supports]
                if item in answerable:
                    self.assertGreaterEqual(len(supports), 2)
                    self.assertLessEqual(len(supports), 3)
                    self.assertEqual(
                        item["answer_session_ids"], list(dict.fromkeys(support_sessions))
                    )
                    self.assertTrue(
                        any(
                            abs(left[0] - right[0]) >= 2
                            for i, left in enumerate(supports)
                            for right in supports[i + 1 :]
                        )
                    )
                else:
                    self.assertEqual(item["answer"], "UNKNOWN")
                    self.assertEqual(item["answer_session_ids"], [])
                    self.assertEqual(supports, [])

        questions = parse_longmemeval_bytes(
            self.raw, namespace="cross-session-test-v1", variant="s"
        )
        self.assertEqual(len(questions), 6)
        self.assertEqual(sum(len(question.records) for question in questions), 72)

    def test_metadata_coordinates_and_exact_fixture_isolation(self) -> None:
        self.assertEqual(self.metadata["license"], "MIT")
        self.assertEqual(self.metadata["assigned_split"], "test")
        self.assertEqual(self.metadata["annotation_status"], "agent_authored_provisional")
        rows = self.metadata["cases"]
        self.assertEqual(
            {row["question_id"] for row in rows}, {item["question_id"] for item in self.items}
        )
        for field in ("source_id", "scenario_id", "scenario_group", "template_group"):
            self.assertEqual(len({row[field] for row in rows}), 6)
        entity_ids = [entity for row in rows for entity in row["entity_ids"]]
        self.assertEqual(len(entity_ids), len(set(entity_ids)))

        by_question = {item["question_id"]: item for item in self.items}
        for row in rows:
            item = by_question[row["question_id"]]
            self.assertEqual(row["split"], "test")
            positions = {sid: index for index, sid in enumerate(item["haystack_session_ids"])}
            dependencies = row["support_dependencies"]
            marked = {
                (item["haystack_session_ids"][index], turn_index)
                for index, session in enumerate(item["haystack_sessions"])
                for turn_index, turn in enumerate(session, start=1)
                if turn.get("has_answer") is True
            }
            coordinates = {(dep["session_id"], dep["turn_number"]) for dep in dependencies}
            self.assertEqual(coordinates, marked)
            for dep in dependencies:
                turn = item["haystack_sessions"][positions[dep["session_id"]]][
                    dep["turn_number"] - 1
                ]
                self.assertIs(turn.get("has_answer"), True)
            dependency_positions = [positions[dep["session_id"]] for dep in dependencies]
            if dependency_positions:
                self.assertGreaterEqual(max(dependency_positions) - min(dependency_positions), 2)

        prior_values: set[str] = set()
        for name in PRIOR_FIXTURES:
            previous = json.loads((ROOT / "examples/synthetic" / name).read_text(encoding="utf-8"))
            prior_values.update(text.casefold() for text in _all_strings(previous) if text)
        new_text = {
            text.casefold()
            for item in self.items
            for text in (
                item["question"],
                *([item["answer"]] if item["answer"] != "UNKNOWN" else []),
                *[turn["content"] for session in item["haystack_sessions"] for turn in session],
            )
            if text
        }
        new_ids = {
            text.casefold()
            for row in rows
            for text in (
                row["question_id"],
                row["source_id"],
                row["scenario_id"],
                row["scenario_group"],
                row["template_group"],
                *row["entity_ids"],
            )
        }
        self.assertFalse(new_text & prior_values)
        self.assertFalse(new_ids & prior_values)

    def test_cpu_retrieval_baseline_counts_unknowns_and_citations(self) -> None:
        report = evaluator.evaluate_retrieval_bytes(
            self.raw, namespace="cross-session-test-v1", variant="s", cutoffs=(1, 3, 5)
        )
        self.assertEqual(
            report["counts"],
            {
                "question_count": 6,
                "abstention_count": 2,
                "unlabeled_count": 0,
                "scored_count": 4,
            },
        )
        parsed = parse_longmemeval_bytes(self.raw, namespace="cross-session-test-v1", variant="s")
        by_question = {question.question_id: question for question in parsed}
        self.assertEqual(set(report["questions"][0]), {"question_id", "retrieval"})
        for mode in report["modes"]:
            for aggregate in report["aggregates"][mode]:
                self.assertEqual(aggregate["question_count"], 6)
                self.assertEqual(aggregate["abstention_count"], 2)
                self.assertEqual(aggregate["scored_question_count"], 4)
            for result in report["questions"]:
                question = by_question[result["question_id"]]
                records = {record.record_id: record for record in question.records}
                with TemporaryDirectory() as directory:
                    with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                        store.ingest(question.records)
                        for cell in result["retrieval"][mode]:
                            if question.is_abstention:
                                self.assertEqual(cell["exclusion_reason"], "abstention")
                                self.assertIsNone(cell["label_recall"])
                                self.assertIsNone(cell["all_labeled_turns_found"])
                            for hit in cell["returned"]:
                                record = records[hit["record_id"]]
                                resolved = fetch_citation(store, hit["citation"])
                                self.assertEqual(hit["revision"], record.revision)
                                self.assertEqual(resolved.record, record)
