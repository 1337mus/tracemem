"""Tests for scenario-cluster uncertainty summaries of generated-candidate reviews."""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

from test_scenario_reviews import _candidate, _fixture, _json_claims, _receipt

from experiments import bootstrap_scenario_reviews
from experiments.prepare_scenario_reviews import prepare_reviews
from experiments.summarize_scenario_reviews import summarize_reviews


def _response_document(
    bundle: dict[str, Any], reviewer_id: str, reviewer_kind: str, label: str
) -> dict[str, Any]:
    return {
        "version": 1,
        "bundle_sha256": bundle["bundle_sha256"],
        "responses": [
            {
                "task_id": task["task_id"],
                "label": label,
                "rationale": "The cited evidence supports this claim.",
                "reviewer_id": reviewer_id,
                "reviewer_kind": reviewer_kind,
                "timestamp": "2026-10-14T19:00:00Z",
            }
            for task in bundle["tasks"]
        ],
    }


class ReviewUncertaintyTests(unittest.TestCase):
    def test_percentile_interval_uses_linear_interpolation(self) -> None:
        interval = bootstrap_scenario_reviews._percentile_interval([0.0, 1.0, 2.0, 4.0])
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval[0], 0.075)
        self.assertAlmostEqual(interval[1], 3.85)

    def test_bootstrap_sample_count_must_be_an_integer_between_one_and_one_hundred_thousand(
        self,
    ) -> None:
        for invalid in (0, 100_001, True, 2.5):
            with self.subTest(bootstrap_samples=invalid):
                with self.assertRaisesRegex(ValueError, "bootstrap_samples"):
                    bootstrap_scenario_reviews.bootstrap_reviews(
                        [], [], seed=1, bootstrap_samples=invalid
                    )

    def test_point_estimates_match_accounting_and_all_split_cases_are_sampled(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        other_cases = [
            row for row in cases if row["expected"] and row["case_id"] != case["case_id"]
        ]
        parse_case, schema_case = other_cases[:2]
        claim = _candidate(case["case_id"], "clustered review candidate")
        receipt = _receipt(
            "qwen3-4b",
            {
                case["case_id"]: _json_claims(claim),
                parse_case["case_id"]: "not JSON",
                schema_case["case_id"]: '{"claims":[{"unexpected":"shape"}]}',
            },
        )
        bundle, _, _ = prepare_reviews([receipt], seed=41)
        response = _response_document(bundle, "reviewer-z", "human", "supported")

        report = bootstrap_scenario_reviews.bootstrap_reviews(
            [receipt], [response], seed=41, bootstrap_seed=7, bootstrap_samples=25
        )
        reference = summarize_reviews([receipt], [response], seed=41)
        split = case["split"]
        cell = report["by_model"]["qwen3-4b"][split]["reviewers"][0]
        reference_row = reference["by_model"]["qwen3-4b"][split]["reviewers"][0]
        sampling = cell["scenario_sampling"]

        self.assertEqual(
            cell["metrics"]["supported_rate_all_reviewed_origins"]["estimate"],
            reference_row["supported_rate_all_reviewed_origins"],
        )
        self.assertEqual(sampling["scenarios_per_replicate"], 24)
        self.assertEqual(sampling["available_scenarios_with_reviewed_origins"], 1)
        self.assertEqual(sampling["available_scenarios_without_reviewed_origins"], 23)
        self.assertEqual(
            sampling["sampled_scenario_draws_with_reviewed_origins"]
            + sampling["sampled_scenario_draws_without_reviewed_origins"],
            24 * 25,
        )
        self.assertEqual(report["accounting"]["provenance_sha256"], reference["provenance_sha256"])
        self.assertEqual(
            {row["reason"] for row in report["accounting"]["exclusions"]},
            {"invalid_json", "invalid_candidate_schema"},
        )

    def test_all_origins_in_a_sampled_case_move_together_and_duplicates_count(self) -> None:
        cases, _, _ = _fixture()
        split = next(case["split"] for case in cases if case["expected"])
        active_cases = [case for case in cases if case["expected"] and case["split"] == split][:2]
        case_a, case_b = active_cases
        case_ids = sorted(case["case_id"] for case in cases if case["split"] == split)

        def sampled_case_counts(candidate_seed: int) -> Counter[str]:
            rng = random.Random(candidate_seed)
            return Counter(rng.choice(case_ids) for _ in range(len(case_ids)))

        seed = next(
            candidate_seed
            for candidate_seed in range(1000)
            if all(
                sampled_case_counts(candidate_seed)[case_id] == 1
                for case_id in (case_a["case_id"], case_b["case_id"])
            )
        )
        claim_a = _candidate(case_a["case_id"], "duplicated supported claim")
        claim_b = _candidate(case_b["case_id"], "single unsupported claim")
        receipt = _receipt(
            "qwen3-4b",
            {
                case_a["case_id"]: _json_claims(claim_a, claim_a),
                case_b["case_id"]: _json_claims(claim_b),
            },
        )
        bundle, answer_key, _ = prepare_reviews([receipt], seed=53)
        task_cases = {row["task_id"]: row["case_id"] for row in answer_key["task_keys"]}
        response = {
            "version": 1,
            "bundle_sha256": bundle["bundle_sha256"],
            "responses": [
                {
                    "task_id": task["task_id"],
                    "label": (
                        "supported"
                        if task_cases[task["task_id"]] == case_a["case_id"]
                        else "unsupported"
                    ),
                    "rationale": "This test label is intentionally fictional.",
                    "reviewer_id": "reviewer-cluster",
                    "reviewer_kind": "agent",
                    "timestamp": "2026-10-14T19:00:00Z",
                }
                for task in bundle["tasks"]
            ],
        }

        report = bootstrap_scenario_reviews.bootstrap_reviews(
            [receipt], [response], seed=53, bootstrap_seed=seed, bootstrap_samples=1
        )
        cell = report["by_model"]["qwen3-4b"][split]["reviewers"][0]
        metric = cell["metrics"]["supported_rate_all_reviewed_origins"]

        self.assertEqual(metric["estimate"], 2 / 3)
        self.assertEqual(metric["interval_95"], {"lower": 2 / 3, "upper": 2 / 3})
        self.assertEqual(
            cell["scenario_sampling"]["sampled_scenario_draws_with_reviewed_origins"], 2
        )

        bootstrap_seed = 2025
        samples = 80
        oracle_rng = random.Random(bootstrap_seed)
        oracle_values: list[float] = []
        undefined = 0
        sampled_active_draws = 0
        for _ in range(samples):
            sampled = [oracle_rng.choice(case_ids) for _ in case_ids]
            draw_counts = Counter(sampled)
            count_a = draw_counts[case_a["case_id"]]
            count_b = draw_counts[case_b["case_id"]]
            sampled_active_draws += count_a + count_b
            denominator = 2 * count_a + count_b
            if denominator:
                oracle_values.append(2 * count_a / denominator)
            else:
                undefined += 1
        sorted_values = sorted(oracle_values)

        def oracle_percentile(fraction: float) -> float:
            rank = (len(sorted_values) - 1) * fraction
            lower = int(rank)
            upper = min(lower + 1, len(sorted_values) - 1)
            weight = rank - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

        global_random_state = random.getstate()
        multi_report = bootstrap_scenario_reviews.bootstrap_reviews(
            [receipt], [response], seed=53, bootstrap_seed=bootstrap_seed, bootstrap_samples=samples
        )
        self.assertEqual(random.getstate(), global_random_state)
        multi_cell = multi_report["by_model"]["qwen3-4b"][split]["reviewers"][0]
        multi_metric = multi_cell["metrics"]["supported_rate_all_reviewed_origins"]
        self.assertEqual(
            multi_metric["interval_95"],
            {"lower": oracle_percentile(0.025), "upper": oracle_percentile(0.975)},
        )
        self.assertEqual(multi_metric["valid_replicate_count"], len(oracle_values))
        self.assertEqual(multi_metric["undefined_replicate_count"], undefined)
        self.assertEqual(
            multi_cell["scenario_sampling"]["sampled_scenario_draws_with_reviewed_origins"],
            sampled_active_draws,
        )

    def test_uncertain_only_reviews_keep_decisive_metric_undefined(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "uncertain review candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=59)
        response = _response_document(bundle, "reviewer-uncertain", "agent", "uncertain")

        report = bootstrap_scenario_reviews.bootstrap_reviews(
            [receipt], [response], seed=59, bootstrap_seed=17, bootstrap_samples=30
        )
        metrics = report["by_model"]["qwen3-4b"][case["split"]]["reviewers"][0]["metrics"]
        decisive = metrics["supported_rate_decisive_origins"]
        uncertain = metrics["uncertain_rate_all_reviewed_origins"]

        self.assertIsNone(decisive["estimate"])
        self.assertIsNone(decisive["interval_95"])
        self.assertEqual(decisive["valid_replicate_count"], 0)
        self.assertEqual(decisive["undefined_replicate_count"], 30)
        self.assertEqual(uncertain["estimate"], 1.0)
        self.assertEqual(uncertain["interval_95"], {"lower": 1.0, "upper": 1.0})
        self.assertEqual(
            uncertain["valid_replicate_count"] + uncertain["undefined_replicate_count"], 30
        )

    def test_no_reviewer_documents_produce_empty_reviewer_cells(self) -> None:
        receipt = _receipt("qwen3-4b")

        report = bootstrap_scenario_reviews.bootstrap_reviews(
            [receipt], [], seed=61, bootstrap_seed=3, bootstrap_samples=4
        )

        for split in ("dev", "test"):
            self.assertEqual(report["by_model"]["qwen3-4b"][split]["reviewers"], [])

    def test_seed_parameters_must_be_integers_and_reject_booleans(self) -> None:
        receipt = _receipt("qwen3-4b")
        invalid_arguments = (
            {"seed": True},
            {"seed": 1.5},
            {"bootstrap_seed": False},
            {"bootstrap_seed": 2.5},
        )
        for invalid in invalid_arguments:
            with self.subTest(arguments=invalid):
                seed = invalid.get("seed", 1)
                bootstrap_seed = invalid.get("bootstrap_seed", 1729)
                with self.assertRaisesRegex(ValueError, "integer"):
                    bootstrap_scenario_reviews.bootstrap_reviews(
                        [receipt], [], seed=seed, bootstrap_seed=bootstrap_seed
                    )

    def test_fixed_seed_replays_independently_of_receipt_and_response_order(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "shared across model receipts")
        overrides = {case["case_id"]: _json_claims(claim)}
        receipts = [_receipt("qwen3-4b", overrides), _receipt("phi-3.5-mini", overrides)]
        bundle, _, _ = prepare_reviews(receipts, seed=67)
        responses = [
            _response_document(bundle, "reviewer-human", "human", "supported"),
            _response_document(bundle, "reviewer-agent", "agent", "uncertain"),
        ]

        first = bootstrap_scenario_reviews.bootstrap_reviews(
            receipts, responses, seed=67, bootstrap_seed=101, bootstrap_samples=40
        )
        replay = bootstrap_scenario_reviews.bootstrap_reviews(
            receipts, responses, seed=67, bootstrap_seed=101, bootstrap_samples=40
        )
        reordered = bootstrap_scenario_reviews.bootstrap_reviews(
            list(reversed(receipts)),
            list(reversed(responses)),
            seed=67,
            bootstrap_seed=101,
            bootstrap_samples=40,
        )

        self.assertEqual(first["by_model"], replay["by_model"])
        self.assertEqual(first["by_model"], reordered["by_model"])

    def test_response_hash_coverage_and_duplicate_reviewer_failures_are_preserved(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "validation candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=71)
        response = _response_document(bundle, "reviewer-validation", "human", "supported")

        bad_hash = copy.deepcopy(response)
        bad_hash["bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle hash"):
            bootstrap_scenario_reviews.bootstrap_reviews([receipt], [bad_hash], seed=71)

        incomplete = copy.deepcopy(response)
        incomplete["responses"].clear()
        with self.assertRaisesRegex(ValueError, "missing task ID"):
            bootstrap_scenario_reviews.bootstrap_reviews([receipt], [incomplete], seed=71)

        with self.assertRaisesRegex(ValueError, "duplicate reviewer"):
            bootstrap_scenario_reviews.bootstrap_reviews([receipt], [response, response], seed=71)

    def test_cli_writes_exclusively_and_preserves_inputs_on_errors(self) -> None:
        cases, _, _ = _fixture()
        case = next(case for case in cases if case["expected"])
        claim = _candidate(case["case_id"], "exclusive output candidate")
        receipt = _receipt("qwen3-4b", {case["case_id"]: _json_claims(claim)})
        bundle, _, _ = prepare_reviews([receipt], seed=73)
        response = _response_document(bundle, "reviewer-cli", "agent", "supported")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            response_path = root / "response.json"
            output_path = root / "report.json"
            receipt_path.write_bytes(receipt)
            response_bytes = json.dumps(response).encode("utf-8")
            response_path.write_bytes(response_bytes)
            args = [
                "--receipts",
                str(receipt_path),
                "--responses",
                str(response_path),
                "--seed",
                "73",
                "--bootstrap-seed",
                "5",
                "--bootstrap-samples",
                "8",
                "--output",
                str(output_path),
            ]

            self.assertEqual(bootstrap_scenario_reviews.main(args), 0)
            written = output_path.read_bytes()
            self.assertEqual(json.loads(written)["bootstrap"]["samples"], 8)
            self.assertEqual(receipt_path.read_bytes(), receipt)
            self.assertEqual(response_path.read_bytes(), response_bytes)

            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(bootstrap_scenario_reviews.main(args), 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())
            self.assertEqual(output_path.read_bytes(), written)

            alias_args = [*args[:-1], str(receipt_path)]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(bootstrap_scenario_reviews.main(alias_args), 2)
            self.assertIn("aliases an input", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), receipt)

            invalid_response_path = root / "invalid-response.json"
            invalid_output_path = root / "invalid-report.json"
            invalid_response = copy.deepcopy(response)
            invalid_response["bundle_sha256"] = "0" * 64
            invalid_response_bytes = json.dumps(invalid_response).encode("utf-8")
            invalid_response_path.write_bytes(invalid_response_bytes)
            invalid_args = [
                *args[: args.index("--responses")],
                "--responses",
                str(invalid_response_path),
                *args[args.index("--seed") : -1],
                str(invalid_output_path),
            ]
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(bootstrap_scenario_reviews.main(invalid_args), 2)
            self.assertIn("bundle hash", stderr.getvalue())
            self.assertFalse(invalid_output_path.exists())
            self.assertEqual(receipt_path.read_bytes(), receipt)
            self.assertEqual(invalid_response_path.read_bytes(), invalid_response_bytes)


if __name__ == "__main__":
    unittest.main()
