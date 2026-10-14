"""Tests for the authored synthetic NLI support diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tracemem.support import load_support_challenge, score_support_cases


def _pair(
    scenario_id: str = "scenario-one",
    *,
    split: str = "diagnostic",
    supported_tags: list[str] | None = None,
    unsupported_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"{scenario_id}-positive",
            "scenario_id": scenario_id,
            "split": split,
            "tags": ["shared"] if supported_tags is None else supported_tags,
            "premise": "A fictional record says the sample was prepared.",
            "hypothesis": "The record says the sample was prepared.",
            "supported": True,
        },
        {
            "case_id": f"{scenario_id}-negative",
            "scenario_id": scenario_id,
            "split": split,
            "tags": ["shared"] if unsupported_tags is None else unsupported_tags,
            "premise": "A fictional record says the sample was prepared.",
            "hypothesis": "The record says the sample was discarded.",
            "supported": False,
        },
    ]


def _outputs(
    cases: Sequence[dict[str, Any]],
    positive: list[float] | None = None,
    negative: list[float] | None = None,
) -> dict[str, list[float]]:
    positive_scores = [0.1, 0.8, 0.1] if positive is None else positive
    negative_scores = [0.8, 0.1, 0.1] if negative is None else negative
    return {
        case["case_id"]: (positive_scores if case["supported"] else negative_scores).copy()
        for case in cases
    }


def _load_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with patch("tracemem.support.files") as files:
        files.return_value.joinpath.return_value.read_bytes.return_value = raw
        return load_support_challenge()


class SupportFixtureTests(unittest.TestCase):
    def test_bundled_challenge_has_24_diagnostic_pairs_and_a_raw_file_hash(self) -> None:
        cases, fixture_hash = load_support_challenge()

        self.assertEqual(len(cases), 48)
        self.assertEqual(len({case["scenario_id"] for case in cases}), 24)
        self.assertEqual(len({case["case_id"] for case in cases}), 48)
        self.assertEqual({case["split"] for case in cases}, {"diagnostic"})
        self.assertEqual(len(fixture_hash), 64)
        self.assertEqual(
            fixture_hash,
            hashlib.sha256(
                (
                    Path(__file__).parents[1] / "src/tracemem/data/support_challenge.json"
                ).read_bytes()
            ).hexdigest(),
        )
        by_scenario: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            by_scenario.setdefault(case["scenario_id"], []).append(case)
        for pair in by_scenario.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual({case["supported"] for case in pair}, {True, False})
            self.assertEqual(pair[0]["premise"], pair[1]["premise"])

    def test_loader_rejects_duplicate_json_keys_and_nonstandard_constants(self) -> None:
        for raw, message in (
            (b'{"version":1,"version":1}', "duplicate JSON key"),
            (b'{"version":NaN}', "nonstandard JSON constant"),
        ):
            with self.subTest(message=message), patch("tracemem.support.files") as files:
                files.return_value.joinpath.return_value.read_bytes.return_value = raw
                with self.assertRaisesRegex(ValueError, message):
                    load_support_challenge()

    def test_loader_rejects_duplicate_ids_and_scenario_split_conflicts(self) -> None:
        cases, _ = load_support_challenge()
        with self.subTest(reason="duplicate case ID"):
            changed = copy.deepcopy(cases)
            changed[1]["case_id"] = changed[0]["case_id"]
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                _load_payload({"version": 1, "license": "MIT", "cases": changed})

        with self.subTest(reason="scenario split conflict"):
            changed = copy.deepcopy(cases)
            changed[1]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "scenario appears in multiple splits"):
                _load_payload({"version": 1, "license": "MIT", "cases": changed})


class SupportScoringTests(unittest.TestCase):
    def test_fixed_entailment_thresholds_use_inclusive_decisions_and_all_ids(self) -> None:
        cases = _pair()
        outputs = _outputs(cases, positive=[0.2, 0.5, 0.3], negative=[0.2, 0.5, 0.3])

        report = score_support_cases(cases, outputs)

        self.assertEqual(set(report["thresholds"]), {"0.5", "0.9", "0.99"})
        at_half = report["thresholds"]["0.5"]
        self.assertEqual((at_half["tp"], at_half["fp"], at_half["fn"], at_half["tn"]), (1, 1, 0, 0))
        self.assertEqual(at_half["selected_count"], 2)
        self.assertEqual(at_half["coverage"], 1.0)
        self.assertEqual(at_half["supported_precision"], 0.5)
        self.assertEqual(at_half["supported_recall"], 1.0)
        self.assertEqual(at_half["unsupported_false_positive_rate"], 1.0)
        self.assertEqual(report["case_results"][0]["case_id"], "scenario-one-negative")
        self.assertNotIn("premise", report["case_results"][0])
        self.assertNotIn("hypothesis", report["case_results"][0])

    def test_point_nine_and_point_nine_nine_thresholds_are_inclusive(self) -> None:
        cases = _pair()
        outputs = _outputs(cases, positive=[0.01, 0.99, 0.0], negative=[0.05, 0.9, 0.05])

        report = score_support_cases(cases, outputs)

        self.assertEqual(report["thresholds"]["0.9"]["selected_count"], 2)
        self.assertEqual(report["thresholds"]["0.99"]["selected_count"], 1)

    def test_scenario_bootstrap_is_deterministic_and_reports_defined_draws(self) -> None:
        cases = _pair("scenario-b") + _pair("scenario-a") + _pair("scenario-c")
        outputs = _outputs(cases)

        first = score_support_cases(cases, outputs)
        second = score_support_cases(cases, outputs)

        interval = first["thresholds"]["0.5"]["supported_precision_interval_95"]
        self.assertEqual(interval, second["thresholds"]["0.5"]["supported_precision_interval_95"])
        self.assertEqual(first["thresholds"]["0.5"]["precision_interval_defined_draws"], 1000)
        self.assertEqual(
            first["bootstrap"],
            {
                "unit": "scenario",
                "draws": 1000,
                "seed": 1729,
                "percentile_method": "linear interpolation at position fraction*(n-1)",
                "undefined_precision_draws": (
                    "excluded from intervals; defined draw count reported per threshold"
                ),
            },
        )

    def test_bootstrap_metadata_discloses_interpolation_and_undefined_draw_policy(self) -> None:
        report = score_support_cases(_pair(), _outputs(_pair()))

        self.assertEqual(
            report["bootstrap"]["percentile_method"],
            "linear interpolation at position fraction*(n-1)",
        )
        self.assertEqual(
            report["bootstrap"]["undefined_precision_draws"],
            "excluded from intervals; defined draw count reported per threshold",
        )

    def test_zero_selection_has_no_precision_or_precision_interval(self) -> None:
        cases = _pair()
        report = score_support_cases(
            cases,
            _outputs(cases, positive=[0.7, 0.2, 0.1], negative=[0.6, 0.2, 0.2]),
        )

        metrics = report["thresholds"]["0.5"]
        self.assertEqual(metrics["selected_count"], 0)
        self.assertIsNone(metrics["supported_precision"])
        self.assertIsNone(metrics["supported_precision_interval_95"])
        self.assertEqual(metrics["precision_interval_defined_draws"], 0)

    def test_zero_unsupported_tag_denominator_has_null_false_positive_rate(self) -> None:
        cases = _pair(supported_tags=["positive-only"], unsupported_tags=["other"])

        report = score_support_cases(cases, _outputs(cases))

        positive_only = report["thresholds"]["0.5"]["by_tag"]["positive-only"]
        self.assertIsNone(positive_only["unsupported_false_positive_rate"])

    def test_outputs_must_cover_exact_case_ids_with_valid_probability_triplets(self) -> None:
        cases = _pair()
        valid = _outputs(cases)
        invalid_outputs: list[tuple[str, dict[str, object], str]] = [
            (
                "missing",
                {cases[0]["case_id"]: valid[cases[0]["case_id"]]},
                "missing outputs",
            ),
            ("extra", {**valid, "unexpected": [0.3, 0.3, 0.4]}, "unexpected output IDs"),
            ("length", {**valid, cases[0]["case_id"]: [0.4, 0.6]}, "exactly 3"),
            ("sum", {**valid, cases[0]["case_id"]: [0.4, 0.4, 0.4]}, "sum to 1"),
            (
                "bool",
                {**valid, cases[0]["case_id"]: [False, 0.5, 0.5]},
                "probabilities must be numbers",
            ),
            (
                "nan",
                {**valid, cases[0]["case_id"]: [math.nan, 0.5, 0.5]},
                "finite",
            ),
            (
                "negative",
                {**valid, cases[0]["case_id"]: [-0.1, 0.6, 0.5]},
                "between 0 and 1",
            ),
            (
                "above one",
                {**valid, cases[0]["case_id"]: [0.1, 1.1, -0.2]},
                "between 0 and 1",
            ),
        ]
        for reason, outputs, error in invalid_outputs:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex((TypeError, ValueError), error):
                    score_support_cases(cases, outputs)  # type: ignore[arg-type]

    def test_report_is_diagnostic_not_calibration_or_gate_a_approval(self) -> None:
        cases = _pair()

        report = score_support_cases(cases, _outputs(cases))

        self.assertFalse(report["automatic_accept"])
        text = " ".join(report["limitations"])
        self.assertIn("calibration", text)
        self.assertIn("fictional", text)
        self.assertIn("Gate A", text)
        self.assertIn("not a >99% population guarantee", text)


if __name__ == "__main__":
    unittest.main()
