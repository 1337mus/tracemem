"""Contract tests for strict two-axis verifier-pair scoring."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments import verifier_pairs

_DATASET_PATH = Path(__file__).resolve().parents[1] / "examples/synthetic/verifier-pairs-v1.json"


def _dataset() -> dict[str, object]:
    return json.loads(_DATASET_PATH.read_text(encoding="utf-8"))


def _outputs(dataset: dict[str, object], split: str) -> list[dict[str, object]]:
    cases = dataset["cases"]
    assert isinstance(cases, list)
    return [
        {
            "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
            "raw_output": json.dumps(
                {"support": candidate["support"], "eligibility": candidate["eligibility"]}
            ),
            "finish_reason": "eos",
            "output_tokens": 5,
        }
        for case in cases
        if case["split"] == split
        for candidate in case["candidates"]
    ]


class ParseVerdictTests(unittest.TestCase):
    def test_parses_exact_two_string_axes(self) -> None:
        raw = json.dumps({"support": "supported", "eligibility": "eligible"})

        self.assertEqual(
            verifier_pairs.parse_verdict(raw),
            {"support": "supported", "eligibility": "eligible"},
        )

    def test_rejects_duplicate_json_keys(self) -> None:
        raw = '{"support":"unsupported","support":"supported","eligibility":"eligible"}'

        with self.assertRaisesRegex(ValueError, "duplicate"):
            verifier_pairs.parse_verdict(raw)

    def test_rejects_non_finite_json_constants(self) -> None:
        raw = '{"support":NaN,"eligibility":"eligible"}'

        with self.assertRaisesRegex(ValueError, "non-finite"):
            verifier_pairs.parse_verdict(raw)

    def test_rejects_wrong_keys_types_and_enum_values(self) -> None:
        invalid = (
            '{"support":"supported","eligibility":"eligible","reason":"ok"}',
            '{"support":true,"eligibility":"eligible"}',
            '{"support":"maybe","eligibility":"eligible"}',
            '{"support":"supported","eligibility":false}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                verifier_pairs.parse_verdict(raw)


class ScoreOutputsTests(unittest.TestCase):
    def test_rejects_missing_output_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "output IDs"):
            verifier_pairs.score_outputs(_dataset(), "dev", [])

    def test_scores_test_split_separately(self) -> None:
        dataset = _dataset()

        report = verifier_pairs.score_outputs(dataset, "test", _outputs(dataset, "test"))

        self.assertEqual(report["split"], "test")
        self.assertEqual(report["aggregate"]["candidate_count"], 12)
        self.assertEqual(report["aggregate"]["validity"]["total"], 12)

    def test_length_finished_output_is_invalid_and_never_counts_as_a_catch(self) -> None:
        dataset = _dataset()
        outputs = _outputs(dataset, "dev")
        target = next(row for row in outputs if '"support": "unsupported"' in row["raw_output"])
        target["finish_reason"] = "length"

        report = verifier_pairs.score_outputs(dataset, "dev", outputs)

        validity = report["aggregate"]["validity"]
        self.assertEqual(validity["total"], 12)
        self.assertEqual(validity["valid"], 11)
        self.assertEqual(validity["truncated"], 1)
        unsupported = report["aggregate"]["support"]["by_reference"]["unsupported"]
        self.assertEqual(unsupported["caught"], 5)
        self.assertEqual(unsupported["total"], 6)

    def test_reports_support_confusion_and_decisive_exact_count(self) -> None:
        dataset = _dataset()
        report = verifier_pairs.score_outputs(dataset, "dev", _outputs(dataset, "dev"))
        support = report["aggregate"]["support"]

        self.assertEqual(support["confusion"]["supported"]["supported"], 6)
        self.assertEqual(support["confusion"]["unsupported"]["unsupported"], 6)
        self.assertEqual(support["decisive_exact_correct"], 12)

    def test_rejects_duplicate_and_extra_output_ids(self) -> None:
        dataset = _dataset()
        valid = _outputs(dataset, "dev")
        duplicate = [*valid, dict(valid[0])]
        extra = [dict(row) for row in valid]
        extra[0]["item_id"] = "unknown::candidate"

        for outputs in (duplicate, extra):
            with (
                self.subTest(outputs=outputs[-1]["item_id"]),
                self.assertRaisesRegex(ValueError, "output IDs"),
            ):
                verifier_pairs.score_outputs(dataset, "dev", outputs)

    def test_support_errors_do_not_reduce_independent_eligibility_scores(self) -> None:
        dataset = _dataset()
        outputs = _outputs(dataset, "dev")
        target = next(row for row in outputs if '"eligibility": "ineligible"' in row["raw_output"])
        verdict = json.loads(target["raw_output"])
        verdict["support"] = "unsupported" if verdict["support"] == "supported" else "supported"
        target["raw_output"] = json.dumps(verdict)

        report = verifier_pairs.score_outputs(dataset, "dev", outputs)
        eligibility = report["aggregate"]["eligibility"]
        support = report["aggregate"]["support"]

        self.assertEqual(eligibility["decisive_exact_correct"], 8)
        self.assertEqual(eligibility["uncertain"]["total"], 4)
        self.assertEqual(eligibility["uncertain"]["exact_correct"], 4)
        self.assertEqual(support["decisive_exact_correct"], 11)

    def test_reports_pair_phenomenon_and_explicit_no_filter_control_metrics(self) -> None:
        dataset = _dataset()
        report = verifier_pairs.score_outputs(dataset, "dev", _outputs(dataset, "dev"))
        aggregate = report["aggregate"]
        control = aggregate["no_filter_control"]

        self.assertEqual(aggregate["support_pairs"]["both_support_labels_correct"], 6)
        self.assertEqual(aggregate["support_pairs"]["decisive_total"], 6)
        self.assertEqual(len(aggregate["per_phenomenon"]), 6)
        self.assertTrue(all(item["both_support_labels_correct"] for item in report["pair_results"]))
        self.assertIn("retain all candidates", control["policy"])
        self.assertEqual(control["retained_candidate_count"], 12)
        self.assertEqual(control["support"]["by_reference"]["unsupported"]["caught"], 0)
        self.assertEqual(control["eligibility"]["by_reference"]["ineligible"]["correct"], 0)
        self.assertEqual(aggregate["telemetry"]["output_tokens"], 60)
        self.assertEqual(aggregate["telemetry"]["items_with_output_tokens"], 12)

    def test_malformed_and_unknown_outputs_are_failures_not_catches(self) -> None:
        dataset = _dataset()
        outputs = _outputs(dataset, "dev")
        unsupported = [row for row in outputs if '"support": "unsupported"' in row["raw_output"]]
        unsupported[0]["finish_reason"] = "unknown"
        unsupported[1]["raw_output"] = "not JSON"

        report = verifier_pairs.score_outputs(dataset, "dev", outputs)
        aggregate = report["aggregate"]

        self.assertEqual(aggregate["validity"]["malformed"], 1)
        self.assertEqual(aggregate["validity"]["finish_reasons"]["unknown"], 1)
        self.assertEqual(aggregate["support"]["by_reference"]["unsupported"]["caught"], 4)
        eligibility_invalid = sum(
            row["invalid"] for row in aggregate["eligibility"]["confusion"].values()
        )
        self.assertEqual(eligibility_invalid, 2)

    def test_present_output_token_count_must_be_a_nonnegative_integer(self) -> None:
        dataset = _dataset()
        outputs = _outputs(dataset, "dev")
        outputs[0]["output_tokens"] = None

        with self.assertRaisesRegex(ValueError, "output_tokens"):
            verifier_pairs.score_outputs(dataset, "dev", outputs)
