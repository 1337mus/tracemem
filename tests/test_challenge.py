"""Synthetic claim challenge fixtures and evaluator tests."""

import json
import unittest

from tracemem.challenge import evaluate_outputs, load_challenge, render_prompt


class ChallengeFixtureTests(unittest.TestCase):
    def test_load_challenge_returns_distinct_cases_with_exact_evidence_spans(self) -> None:
        cases, file_hash = load_challenge()

        self.assertGreaterEqual(len(cases), 24)
        self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
        self.assertEqual(len({case["scenario_id"] for case in cases}), len(cases))
        self.assertEqual(len(file_hash), 64)
        for case in cases:
            self.assertEqual(case["split"], "diagnostic")
            messages = {
                (message["session_id"], message["message_id"]): message
                for message in case["messages"]
            }
            for claim in case["expected"]:
                for evidence in claim["evidence"]:
                    message = messages[(evidence["session_id"], evidence["message_id"])]
                    self.assertEqual(
                        message["text"][evidence["char_start"] : evidence["char_end"]],
                        evidence["quote"],
                    )

    def test_evaluator_scores_exact_structured_claims_and_evidence(self) -> None:
        cases, _ = load_challenge()
        case = next(case for case in cases if case["expected"])
        report = evaluate_outputs(
            [case],
            {case["case_id"]: json.dumps({"claims": case["expected"]})},
            bootstrap_samples=20,
            seed=7,
        )

        self.assertEqual(report["exact_match"]["true_positive"], len(case["expected"]))
        self.assertEqual(report["exact_match"]["precision"], 1.0)
        self.assertEqual(report["exact_match"]["recall"], 1.0)
        self.assertEqual(report["evidence_integrity"]["invalid_claims"], 0)
        self.assertEqual(report["parse_failures"], 0)
        self.assertEqual(report["confidence_intervals_95"]["f1"][0], 1.0)

    def test_parse_failure_invalidates_whole_case_and_missing_output_is_rejected(self) -> None:
        cases, _ = load_challenge()
        case = next(case for case in cases if len(case["expected"]) > 1)
        report = evaluate_outputs(
            [case],
            {case["case_id"]: '{"claims": [], trailing text'},
            bootstrap_samples=10,
        )
        self.assertEqual(report["parse_failures"], 1)
        self.assertEqual(report["exact_match"]["false_negative"], len(case["expected"]))
        self.assertEqual(report["case_breakdown"][case["case_id"]]["predicted_claims"], 0)

        with self.assertRaisesRegex(ValueError, "missing outputs"):
            evaluate_outputs([case], {})

        schema_report = evaluate_outputs(
            [case],
            {case["case_id"]: {"claims": [case["expected"][0], {"subject": "partial"}]}},
            bootstrap_samples=10,
        )
        self.assertEqual(schema_report["schema_failures"], 1)
        self.assertEqual(schema_report["case_breakdown"][case["case_id"]]["predicted_claims"], 0)
        self.assertEqual(schema_report["exact_match"]["false_negative"], len(case["expected"]))

    def test_label_errors_and_unknown_abstention_are_reported(self) -> None:
        cases, _ = load_challenge()
        case = next(case for case in cases if case["expected"])
        claim = json.loads(json.dumps(case["expected"][0]))
        claim["modality"] = "hypothetical"
        unknown = next(case for case in cases if not case["expected"])
        report = evaluate_outputs(
            [case, unknown],
            {
                case["case_id"]: {"claims": [claim]},
                unknown["case_id"]: {"claims": []},
            },
            bootstrap_samples=10,
        )

        self.assertEqual(report["label_errors"]["by_field"]["modality"], 1)
        self.assertEqual(report["unknown_case_abstention"]["cases"], 1)
        self.assertEqual(report["unknown_case_abstention"]["abstentions"], 1)

    def test_evaluator_rejects_duplicate_case_ids(self) -> None:
        cases, _ = load_challenge()
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            evaluate_outputs([cases[0], cases[0]], {cases[0]["case_id"]: '{"claims": []}'})

    def test_duplicate_json_keys_and_nonstandard_constants_are_parse_failures(self) -> None:
        cases, _ = load_challenge()
        unknown = next(case for case in cases if not case["expected"])
        report = evaluate_outputs(
            [unknown],
            {unknown["case_id"]: '{"claims": [], "claims": [{"bad": NaN}]}'},
            bootstrap_samples=10,
        )
        self.assertEqual(report["parse_failures"], 1)
        self.assertEqual(report["unknown_case_abstention"]["abstentions"], 0)

        constants_report = evaluate_outputs(
            [unknown],
            {unknown["case_id"]: '{"claims": [], "extra": Infinity}'},
            bootstrap_samples=10,
        )
        self.assertEqual(constants_report["parse_failures"], 1)
        self.assertEqual(constants_report["unknown_case_abstention"]["abstentions"], 0)

    def test_evidence_integrity_and_prompt_boundary(self) -> None:
        cases, _ = load_challenge()
        case = next(case for case in cases if case["expected"])
        claim = json.loads(json.dumps(case["expected"][0]))
        claim["evidence"][0]["quote"] += " fabricated"
        report = evaluate_outputs(
            [case],
            {case["case_id"]: {"claims": [claim]}},
            bootstrap_samples=10,
        )
        self.assertEqual(report["evidence_integrity"]["invalid_claims"], 1)
        self.assertEqual(report["evidence_integrity"]["errors"]["quote_mismatch"], 1)

        prompt = render_prompt(case)
        self.assertIn(case["messages"][0]["message_id"], prompt)
        self.assertIn(f'"role": "{case["messages"][0]["role"]}"', prompt)
        self.assertNotIn('"expected"', prompt)
        self.assertNotIn('"tags"', prompt)


if __name__ == "__main__":
    unittest.main()
