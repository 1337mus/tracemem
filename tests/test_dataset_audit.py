"""Deterministic contamination/readiness diagnostics for claim datasets."""

from __future__ import annotations

import copy
import json
import math
import unittest
from collections.abc import Callable
from typing import Any

from tracemem.challenge import load_challenge
from tracemem.dataset_audit import audit_cases


def _case(
    case_id: str,
    scenario_id: str,
    split: str,
    texts: tuple[str, ...],
    *,
    roles: tuple[str, ...] | None = None,
    expected: list[object] | None = None,
) -> dict[str, Any]:
    message_roles = roles or tuple("user" for _ in texts)
    return {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "split": split,
        "expected": [] if expected is None else expected,
        "messages": [
            {
                "session_id": f"session-{index}",
                "message_id": f"message-{index}",
                "role": role,
                "text": text,
                "recorded_at": None,
                "content_kind": "conversation",
            }
            for index, (role, text) in enumerate(zip(message_roles, texts, strict=True))
        ],
    }


class DatasetAuditTests(unittest.TestCase):
    def test_exact_duplicates_cross_splits_even_when_scenario_ids_differ(self) -> None:
        cases = [
            _case("z-case", "scenario-z", "test", ("  Same\n text ",)),
            _case("a-case", "scenario-a", "train", ("same text",)),
        ]

        report = audit_cases(cases)

        self.assertEqual(
            report["exact_duplicate_pairs"],
            [{"case_ids": ["a-case", "z-case"], "splits": ["test", "train"]}],
        )
        self.assertNotIn("Same", json.dumps(report))

    def test_unicode_nfkc_casefold_and_whitespace_normalize_exact_duplicates(self) -> None:
        cases = [
            _case("one", "one-scenario", "train", ("Café ﬁle",)),
            _case("two", "two-scenario", "test", ("CAFE\u0301   file",)),
        ]

        report = audit_cases(cases)

        self.assertEqual(len(report["exact_duplicate_pairs"]), 1)
        self.assertEqual(report["exact_duplicate_pairs"][0]["case_ids"], ["one", "two"])
        self.assertIn("NFKC", report["comparison"]["normalization"])

    def test_exact_identity_preserves_role_and_message_boundaries(self) -> None:
        cases = [
            _case("one", "s1", "train", ("same text",)),
            _case("two", "s2", "test", ("same text",), roles=("assistant",)),
            _case("three", "s3", "test", ("same", "text")),
        ]

        report = audit_cases(cases)

        self.assertEqual(report["exact_duplicate_pairs"], [])

    def test_exact_duplicates_within_a_split_are_reported(self) -> None:
        cases = [
            _case("second", "s2", "train", ("repeated phrase",)),
            _case("first", "s1", "train", ("REPEATED   phrase",)),
        ]

        report = audit_cases(cases)

        self.assertEqual(
            report["exact_duplicate_pairs"],
            [{"case_ids": ["first", "second"], "splits": ["train"]}],
        )
        self.assertEqual(report["near_duplicate_pairs"], [])

    def test_near_duplicate_threshold_is_inclusive_and_pair_ids_are_sorted(self) -> None:
        cases = [
            _case("z-case", "z", "test", ("amber birch cedar",)),
            _case("a-case", "a", "train", ("amber birch",)),
        ]

        report = audit_cases(cases, near_duplicate_threshold=2 / 3)

        self.assertEqual(
            report["near_duplicate_pairs"],
            [
                {
                    "case_ids": ["a-case", "z-case"],
                    "splits": ["test", "train"],
                    "jaccard": 2 / 3,
                }
            ],
        )
        self.assertEqual(report["comparison"]["near_duplicate"]["threshold"], 2 / 3)
        self.assertEqual(report["comparison"]["near_duplicate"]["inclusive"], True)
        above_boundary = audit_cases(cases, near_duplicate_threshold=0.67)
        self.assertEqual(above_boundary["near_duplicate_pairs"], [])

    def test_same_scenario_in_multiple_splits_is_reported(self) -> None:
        cases = [
            _case("test-case", "shared-scenario", "test", ("a unique test phrase",)),
            _case("train-case", "shared-scenario", "train", ("another unique phrase",)),
        ]

        report = audit_cases(cases)

        self.assertEqual(
            report["scenarios_across_splits"],
            [
                {
                    "scenario_id": "shared-scenario",
                    "splits": ["test", "train"],
                    "case_ids": ["test-case", "train-case"],
                }
            ],
        )

    def test_declared_metadata_reports_partial_coverage_and_cross_split_groups(self) -> None:
        cases = [
            _case("a", "scenario-a", "dev", ("private alpha wording",)),
            _case("b", "scenario-b", "test", ("unrelated beta wording",)),
            _case("c", "scenario-c", "test", ("other gamma wording",)),
            _case("d", "scenario-d", "diagnostic", ("unlabeled delta wording",)),
        ]
        case_metadata = {
            "c": {
                "scenario_group": "shared scenario",
                "template_group": "shared template",
                "entity_ids": ["entity-shared"],
            },
            "a": {
                "scenario_group": "shared scenario",
                "template_group": "shared template",
                "entity_ids": ["entity-shared", "entity-a"],
            },
            "b": {
                "scenario_group": "shared scenario",
                "template_group": "template-b",
                "entity_ids": ["entity-shared"],
            },
        }

        report = audit_cases(cases, case_metadata=case_metadata)

        declared = report["declared_metadata"]
        self.assertEqual(declared["covered_case_count"], 3)
        self.assertEqual(declared["missing_case_ids"], ["d"])
        self.assertFalse(declared["complete"])
        self.assertEqual(
            declared["covered_case_counts_by_split"],
            {"dev": 1, "diagnostic": 0, "test": 2, "train": 0},
        )
        self.assertEqual(
            declared["overlaps"]["scenario_group"],
            [
                {
                    "value": "shared scenario",
                    "splits": ["dev", "test"],
                    "case_ids": ["a", "b", "c"],
                }
            ],
        )
        self.assertEqual(
            declared["overlaps"]["template_group"],
            [
                {
                    "value": "shared template",
                    "splits": ["dev", "test"],
                    "case_ids": ["a", "c"],
                }
            ],
        )
        self.assertEqual(
            declared["overlaps"]["entity_id"],
            [
                {
                    "value": "entity-shared",
                    "splits": ["dev", "test"],
                    "case_ids": ["a", "b", "c"],
                }
            ],
        )
        serialized = json.dumps(report)
        self.assertNotIn("private alpha wording", serialized)
        self.assertNotIn("unrelated beta wording", serialized)
        self.assertTrue(any("partial metadata" in item.lower() for item in declared["limitations"]))
        self.assertIn("may be sensitive", serialized)

    def test_declared_metadata_rejects_unknown_cases_and_malformed_rows(self) -> None:
        cases = [_case("known", "scenario", "dev", ("text",))]
        valid_row = {
            "scenario_group": "scenario-group",
            "template_group": "template-group",
            "entity_ids": ["entity"],
        }
        with self.assertRaisesRegex(ValueError, "unknown case_id"):
            audit_cases(cases, case_metadata={"unknown": valid_row})

        invalid_metadata: tuple[tuple[str, object], ...] = (
            ("blank case ID", {" ": valid_row}),
            ("missing field", {"known": {"scenario_group": "group"}}),
            (
                "blank group",
                {
                    "known": {
                        "scenario_group": " ",
                        "template_group": "template-group",
                        "entity_ids": ["entity"],
                    }
                },
            ),
            (
                "empty entity IDs",
                {
                    "known": {
                        "scenario_group": "scenario-group",
                        "template_group": "template-group",
                        "entity_ids": [],
                    }
                },
            ),
            (
                "duplicate entity IDs",
                {
                    "known": {
                        "scenario_group": "scenario-group",
                        "template_group": "template-group",
                        "entity_ids": ["entity", "entity"],
                    }
                },
            ),
        )
        for name, metadata in invalid_metadata:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    audit_cases(cases, case_metadata=metadata)  # type: ignore[arg-type]

    def test_empty_declared_metadata_has_zero_coverage_and_is_incomplete(self) -> None:
        cases = [
            _case("b", "scenario-b", "custom", ("second text",)),
            _case("a", "scenario-a", "dev", ("first text",)),
        ]

        report = audit_cases(cases, case_metadata={})

        declared = report["declared_metadata"]
        self.assertEqual(declared["covered_case_count"], 0)
        self.assertEqual(declared["missing_case_ids"], ["a", "b"])
        self.assertFalse(declared["complete"])
        self.assertEqual(
            declared["covered_case_counts_by_split"],
            {"custom": 0, "dev": 0, "diagnostic": 0, "test": 0, "train": 0},
        )
        self.assertEqual(
            declared["overlaps"],
            {"scenario_group": [], "template_group": [], "entity_id": []},
        )

    def test_declared_metadata_is_stable_under_input_order_and_not_mutated(self) -> None:
        cases = [
            _case("z", "scenario-z", "test", ("private z text",)),
            _case("a", "scenario-a", "dev", ("private a text",)),
        ]
        metadata = {
            "z": {
                "scenario_group": "shared",
                "template_group": "shared",
                "entity_ids": ["shared-entity"],
            },
            "a": {
                "scenario_group": "shared",
                "template_group": "shared",
                "entity_ids": ["shared-entity"],
            },
        }
        cases_before = copy.deepcopy(cases)
        metadata_before = copy.deepcopy(metadata)

        report = audit_cases(cases, case_metadata=metadata)
        reverse_report = audit_cases(
            list(reversed(cases)), case_metadata=dict(reversed(list(metadata.items())))
        )

        self.assertEqual(report, reverse_report)
        self.assertEqual(cases, cases_before)
        self.assertEqual(metadata, metadata_before)

    def test_diagnostic_split_is_not_counted_as_held_out(self) -> None:
        cases, _ = load_challenge()

        report = audit_cases(cases)

        self.assertEqual(report["case_count"], 24)
        self.assertEqual(report["split_case_counts"]["diagnostic"], 24)
        self.assertEqual(
            {name: report["split_case_counts"][name] for name in ("train", "dev", "test")},
            {"train": 0, "dev": 0, "test": 0},
        )
        self.assertEqual(report["diagnostic_case_count"], 24)
        self.assertEqual(report["held_out_case_count"], 0)
        self.assertFalse(report["automatic_gate_a_approval"])
        self.assertEqual(
            report["no_memory_case_counts_by_split"]["diagnostic"],
            sum(not case["expected"] for case in cases),
        )

    def test_report_has_no_raw_text_and_input_is_not_mutated(self) -> None:
        cases = [_case("private-case", "scenario", "diagnostic", ("PRIVATE SENTINEL",))]
        before = copy.deepcopy(cases)

        report = audit_cases(cases)

        self.assertEqual(cases, before)
        self.assertNotIn("PRIVATE SENTINEL", json.dumps(report))
        self.assertNotIn("claims", json.dumps(report))
        self.assertIn("legal permission", " ".join(report["limitations"]))
        self.assertIn("annotation quality", " ".join(report["limitations"]))
        self.assertIn("sample adequacy", " ".join(report["limitations"]))
        self.assertIn("O(n²)", " ".join(report["limitations"]))

    def test_case_and_message_fields_are_validated(self) -> None:
        valid = _case("case", "scenario", "train", ("text",))
        invalid_cases: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
            ("blank case ID", lambda item: item.update(case_id=" ")),
            ("blank scenario", lambda item: item.update(scenario_id="")),
            ("blank split", lambda item: item.update(split="\n")),
            ("blank message text", lambda item: item["messages"][0].update(text=" \t")),
            ("blank session ID", lambda item: item["messages"][0].update(session_id="")),
            ("blank message ID", lambda item: item["messages"][0].update(message_id="\n")),
            ("blank role", lambda item: item["messages"][0].update(role="")),
            ("missing expected", lambda item: item.pop("expected")),
        )
        for name, mutate in invalid_cases:
            with self.subTest(name=name):
                case = copy.deepcopy(valid)
                mutate(case)
                with self.assertRaises((TypeError, ValueError)):
                    audit_cases([case])

        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            audit_cases([valid, copy.deepcopy(valid)])

        duplicate_message_id_case = _case(
            "duplicate-message", "scenario", "train", ("first", "second")
        )
        duplicate_message_id_case["messages"][1]["session_id"] = "session-0"
        duplicate_message_id_case["messages"][1]["message_id"] = "message-0"
        with self.assertRaisesRegex(ValueError, "duplicate session/message IDs"):
            audit_cases([duplicate_message_id_case])

    def test_near_duplicate_threshold_must_be_finite_and_between_zero_and_one(self) -> None:
        case = _case("case", "scenario", "train", ("valid text",))
        for threshold in (True, False, 0, -0.1, 1.1, math.nan, math.inf, -math.inf, "0.5", None):
            with self.subTest(threshold=threshold):
                with self.assertRaises((TypeError, ValueError)):
                    audit_cases([case], near_duplicate_threshold=threshold)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
