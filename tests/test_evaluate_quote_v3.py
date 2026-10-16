"""Offline contract tests for the provisional quote-v3 evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from training import evaluate_quote_v3
from training.evaluate_sft import _canonical_job_hash
from training.quote_data import render_quote_prompt
from training.train_sft import _QUOTE_V3_PRIOR_ADAPTER, MODEL_ID, MODEL_REVISION

COHORT_SIZES = {"regression": 24, "dev": 24, "test": 48}
ARMS = ("base", "prior", "tuned")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _case(case_id: str, split: str, ordinal: int) -> dict[str, object]:
    message_id = f"message-{case_id}"
    quote = f"I prefer the blue cup for case {case_id}."
    return {
        "case_id": case_id,
        "split": split,
        "scenario_group": f"scenario-{case_id}",
        "template_group": f"template-{case_id}",
        "entity_ids": [f"entity-{case_id}"],
        "tags": ["synthetic", f"group-{ordinal % 4}"],
        "messages": [
            {
                "session_id": f"session-{case_id}",
                "message_id": message_id,
                "role": "user",
                "text": quote,
                "recorded_at": "2026-01-02T03:04:05Z",
                "content_kind": "conversation",
            }
        ],
        "expected": [
            {
                "claim": f"The user prefers the blue cup for case {case_id}.",
                "message_id": message_id,
                "quote": quote,
                "valid_from": None,
                "valid_to": None,
            }
        ],
    }


def _fixture(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": 2,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": cases,
    }


def _write_inputs(
    root: Path,
) -> tuple[Path, Path, dict[str, Path], dict[str, list[dict[str, object]]]]:
    cases_by_cohort: dict[str, list[dict[str, object]]] = {}
    fixture_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {"train": "a" * 64}
    ordered_cohorts: dict[str, list[str]] = {}
    prompts: list[dict[str, str]] = []
    for cohort, count in COHORT_SIZES.items():
        split = "test" if cohort == "test" else "dev"
        cases = [_case(f"{cohort}-{index:02d}", split, index) for index in range(count)]
        cases_by_cohort[cohort] = cases
        fixture_path = root / f"{cohort}.json"
        fixture_bytes = _json_bytes(_fixture(cases))
        fixture_path.write_bytes(fixture_bytes)
        fixture_paths[cohort] = fixture_path
        source_hashes[cohort] = hashlib.sha256(fixture_bytes).hexdigest()
        ordered_cohorts[cohort] = [str(case["case_id"]) for case in cases]
        prompts.extend(
            {
                "case_id": str(case["case_id"]),
                "prompt": render_quote_prompt(case),
            }
            for case in cases
        )

    train_rows = [{"case_id": f"train-{index:04d}", "split": "train"} for index in range(768)]
    request = {
        "mode": "train",
        "profile": "quote-v3",
        "train_rows": train_rows,
        "dev_prompts": prompts,
        "eval_cohorts": ordered_cohorts,
        "prior_adapter": copy.deepcopy(_QUOTE_V3_PRIOR_ADAPTER),
        "remote_run_dir": "/remote/synthetic-run",
    }
    job: dict[str, object] = {
        "schema_version": 1,
        "request": request,
        "source_sha256": source_hashes,
        "module_sha256": {"training/quote_data.py": "b" * 64},
        "runner_sha256": "c" * 64,
        "resource_options": {},
        "dependency_pins": {},
    }
    job["job_sha256"] = _canonical_job_hash(job)

    def output_rows(cohort_cases: list[dict[str, object]], arm: str) -> list[dict[str, object]]:
        outputs: list[dict[str, object]] = []
        for case in cohort_cases:
            response = json.dumps(
                {"claims": case["expected"]}, ensure_ascii=False, separators=(",", ":")
            )
            if arm == "prior" and case["case_id"] == "regression-00":
                response = f"```json\n{response}\n```"
            outputs.append(
                {
                    "case_id": case["case_id"],
                    "prompt_token_ids": [101, 102],
                    "completion_token_ids": [201, 202],
                    "raw_text": response,
                    "response_text": response,
                    "generated_tokens": 2,
                    "max_new_tokens": 512,
                    "reached_generation_limit": False,
                }
            )
        return outputs

    generation: dict[str, object] = {"max_new_tokens": 512, "do_sample": False}
    for arm in ARMS:
        generation[f"{arm}_outputs"] = [
            output
            for cohort in COHORT_SIZES
            for output in output_rows(cases_by_cohort[cohort], arm)
        ]
        generation[f"{arm}_seconds"] = 1.0
    receipt = {
        "job_sha256": job["job_sha256"],
        "runner_sha256": job["runner_sha256"],
        "result": {
            "mode": "train",
            "status": "completed",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "generation": generation,
            "training": {"max_steps": 192, "optimizer_steps": 192, "examples_consumed": 768},
            "comparison": {
                "prior_adapter": copy.deepcopy(request["prior_adapter"]),
                "verified_before": True,
                "verified_after": True,
                "unloaded_without_merge": True,
                "rng_restored": True,
                "no_prior_lora_parameters": True,
            },
        },
    }
    job_path, receipt_path = root / "job.json", root / "receipt.json"
    job_path.write_bytes(_json_bytes(job))
    receipt_path.write_bytes(_json_bytes(receipt))
    return job_path, receipt_path, fixture_paths, cases_by_cohort


class EvaluateQuoteV3Tests(unittest.TestCase):
    def test_rejects_receipt_tampering_of_the_prior_adapter_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result"]["comparison"]["prior_adapter"]["sha256"][
                "adapter_model.safetensors"
            ] = "0" * 64
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "prior adapter comparison"):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_rejects_fixture_bytes_that_do_not_match_the_job_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            fixture = json.loads(fixture_paths["dev"].read_text(encoding="utf-8"))
            fixture_paths["dev"].write_text(
                json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "dev fixture hash does not match fixture bytes"
            ):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_rejects_cohort_ids_mixed_across_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            request = job["request"]
            cohorts = request["eval_cohorts"]
            cohorts["regression"][0], cohorts["dev"][0] = (
                cohorts["dev"][0],
                cohorts["regression"][0],
            )
            prompts_by_id = {row["case_id"]: row for row in request["dev_prompts"]}
            ordered_ids = [case_id for ids in cohorts.values() for case_id in ids]
            request["dev_prompts"] = [prompts_by_id[case_id] for case_id in ordered_ids]
            job["job_sha256"] = _canonical_job_hash(job)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["job_sha256"] = job["job_sha256"]
            job_path.write_bytes(_json_bytes(job))
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "cohort IDs do not match its fixture order"):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_accepts_reordered_json_object_keys_for_eval_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")

            report = evaluate_quote_v3.evaluate_report(
                job_path,
                receipt_path,
                regression_path=fixture_paths["regression"],
                dev_path=fixture_paths["dev"],
                test_path=fixture_paths["test"],
            )

            self.assertEqual(set(report["cohorts"]), set(COHORT_SIZES))

    def test_rejects_unexpected_output_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            generation = receipt["result"]["generation"]
            generation["oracle_outputs"] = copy.deepcopy(generation["base_outputs"])
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "output arms"):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_rejects_missing_case_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result"]["generation"]["base_outputs"].pop()
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "one-to-one"):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_rejects_duplicate_case_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            outputs = receipt["result"]["generation"]["base_outputs"]
            outputs.append(copy.deepcopy(outputs[0]))
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "one-to-one"):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_rejects_cross_arm_prompt_token_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result"]["generation"]["prior_outputs"][0]["prompt_token_ids"] = [999]
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "prompt token IDs differ"):
                evaluate_quote_v3.evaluate_report(
                    job_path,
                    receipt_path,
                    regression_path=fixture_paths["regression"],
                    dev_path=fixture_paths["dev"],
                    test_path=fixture_paths["test"],
                )

    def test_cli_creates_mode_0600_report_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            output_path = root / "report.json"
            args = [
                "--job",
                str(job_path),
                "--receipt",
                str(receipt_path),
                "--regression-fixture",
                str(fixtures["regression"]),
                "--dev-fixture",
                str(fixtures["dev"]),
                "--test-fixture",
                str(fixtures["test"]),
                "--output",
                str(output_path),
            ]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(evaluate_quote_v3.main(args), 0)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            first_contents = output_path.read_bytes()

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(evaluate_quote_v3.main(args), 1)
            self.assertEqual(output_path.read_bytes(), first_contents)

    def test_scores_three_arms_separately_and_limits_fence_diagnostic_to_outer_json_fence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixture_paths, cases = _write_inputs(root)

            self.assertTrue(callable(getattr(evaluate_quote_v3, "evaluate_report", None)))
            report = evaluate_quote_v3.evaluate_report(
                job_path,
                receipt_path,
                regression_path=fixture_paths["regression"],
                dev_path=fixture_paths["dev"],
                test_path=fixture_paths["test"],
            )

            self.assertEqual(report["human_reviewers"], 0)
            self.assertEqual(report["label_provenance"], "agent-authored provisional")
            self.assertEqual(report["semantic_support"], "not yet evaluated")
            self.assertIn("not accuracy", report["reference_match_caveat"])
            self.assertEqual(
                {name: value["case_count"] for name, value in report["cohorts"].items()},
                COHORT_SIZES,
            )
            self.assertEqual(set(report["cohorts"]), set(COHORT_SIZES))

            regression = report["cohorts"]["regression"]
            self.assertEqual(set(regression["arms"]), set(ARMS))
            self.assertEqual(
                regression["arms"]["base"]["strict_raw_metrics"]["reference_string_scores"][
                    "matched_reference_strings"
                ],
                24,
            )
            prior = regression["arms"]["prior"]
            self.assertEqual(prior["strict_raw_metrics"]["parse_failures"], 1)
            self.assertEqual(
                prior["strict_raw_metrics"]["reference_string_scores"]["matched_reference_strings"],
                23,
            )
            diagnostic = prior["fence_diagnostic"]
            self.assertEqual(diagnostic["removed_case_ids"], ["regression-00"])
            self.assertEqual(diagnostic["summary"]["parse_failures"], 0)
            self.assertEqual(
                diagnostic["summary"]["reference_string_scores"]["matched_reference_strings"],
                24,
            )
            diagnostic_case = diagnostic["summary"]["cases"]["regression-00"]
            self.assertEqual(
                diagnostic_case["original_response_text"],
                prior["strict_raw_metrics"]["cases"]["regression-00"]["response_text"],
            )
            self.assertEqual(diagnostic_case["raw_text"], diagnostic_case["original_raw_text"])
            self.assertEqual(diagnostic_case["generated_tokens"], 2)
            self.assertFalse(diagnostic_case["truncated"])
            self.assertEqual(
                report["cohorts"]["dev"]["arms"]["prior"]["strict_raw_metrics"][
                    "reference_string_scores"
                ]["matched_reference_strings"],
                24,
            )
            self.assertEqual(
                report["cohorts"]["test"]["arms"]["tuned"]["strict_raw_metrics"][
                    "reference_string_scores"
                ]["matched_reference_strings"],
                48,
            )


if __name__ == "__main__":
    unittest.main()
