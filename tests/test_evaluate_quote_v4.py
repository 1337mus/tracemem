"""Offline contract tests for the provisional quote-v4 evaluator."""

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
from typing import Any

from tests.test_evaluate_quote_v3 import _write_inputs as _write_v3_inputs
from training import evaluate_quote_v3, evaluate_quote_v4
from training.evaluate_sft import _canonical_job_hash
from training.quote_data import render_quote_prompt
from training.train_sft import _QUOTE_V4_PRIOR_ADAPTER, MODEL_ID, MODEL_REVISION

COHORT_SIZES = {"dev": 12, "test": 24}
ARMS = ("base", "prior", "tuned")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _case(case_id: str, split: str, ordinal: int) -> dict[str, object]:
    message_id = f"message-{case_id}"
    quote = f"I prefer the green mug for case {case_id}."
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
                "recorded_at": "2026-02-03T04:05:06Z",
                "content_kind": "conversation",
            }
        ],
        "expected": [
            {
                "claim": f"The user prefers the green mug for case {case_id}.",
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
        cases = [_case(f"{cohort}-{index:02d}", cohort, index) for index in range(count)]
        cases_by_cohort[cohort] = cases
        fixture_path = root / f"{cohort}.json"
        raw_fixture = _json_bytes(_fixture(cases))
        fixture_path.write_bytes(raw_fixture)
        fixture_paths[cohort] = fixture_path
        source_hashes[cohort] = hashlib.sha256(raw_fixture).hexdigest()
        ordered_cohorts[cohort] = [str(case["case_id"]) for case in cases]
        prompts.extend(
            {"case_id": str(case["case_id"]), "prompt": render_quote_prompt(case)} for case in cases
        )

    train_rows = [{"case_id": f"train-{index:04d}", "split": "train"} for index in range(768)]
    request = {
        "mode": "train",
        "profile": "quote-v4",
        "train_rows": train_rows,
        "dev_prompts": prompts,
        "eval_cohorts": ordered_cohorts,
        "prior_adapter": copy.deepcopy(_QUOTE_V4_PRIOR_ADAPTER),
        "remote_run_dir": "/remote/synthetic-v4-run",
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

    def output_rows(cohort_cases: list[dict[str, object]]) -> list[dict[str, object]]:
        outputs: list[dict[str, object]] = []
        for case in cohort_cases:
            response = json.dumps(
                {"claims": case["expected"]}, ensure_ascii=False, separators=(",", ":")
            )
            outputs.append(
                {
                    "case_id": case["case_id"],
                    "prompt_token_ids": [101, 102],
                    "completion_token_ids": [201, 202],
                    "raw_text": response,
                    "response_text": response,
                    "generated_tokens": 2,
                    "max_new_tokens": 1024,
                    "reached_generation_limit": False,
                }
            )
        return outputs

    generation: dict[str, object] = {"max_new_tokens": 1024, "do_sample": False}
    for arm in ARMS:
        generation[f"{arm}_outputs"] = [
            output for cohort in COHORT_SIZES for output in output_rows(cases_by_cohort[cohort])
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


def _read_job_and_receipt(
    job_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(job_path.read_text(encoding="utf-8")),
        json.loads(receipt_path.read_text(encoding="utf-8")),
    )


def _rewrite_job_and_receipt(
    job_path: Path,
    receipt_path: Path,
    job: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    job["job_sha256"] = _canonical_job_hash(job)
    receipt["job_sha256"] = job["job_sha256"]
    job_path.write_bytes(_json_bytes(job))
    receipt_path.write_bytes(_json_bytes(receipt))


class EvaluateQuoteV4Tests(unittest.TestCase):
    def test_scores_dev_and_test_without_a_regression_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)

            report = evaluate_quote_v4.evaluate_report(
                job_path,
                receipt_path,
                dev_path=fixtures["dev"],
                test_path=fixtures["test"],
            )

            self.assertEqual(list(report["cohorts"]), ["dev", "test"])
            self.assertEqual(
                {name: value["case_count"] for name, value in report["cohorts"].items()},
                COHORT_SIZES,
            )
            self.assertEqual(report["generation"], {"max_new_tokens": 1024, "do_sample": False})
            self.assertEqual(report["human_reviewers"], 0)
            self.assertIn("not accuracy", report["reference_match_caveat"])
            self.assertEqual(set(report["cohorts"]["test"]["arms"]), set(ARMS))

    def test_preserves_the_serialized_v3_report_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_v3_inputs(root)

            report = evaluate_quote_v3.evaluate_report(
                job_path,
                receipt_path,
                regression_path=fixtures["regression"],
                dev_path=fixtures["dev"],
                test_path=fixtures["test"],
            )
            serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)

            self.assertEqual(
                hashlib.sha256((serialized + "\n").encode("utf-8")).hexdigest(),
                "b96c0aa4c629519a83095094b0a542ddfaf215e5da45a98c8cf524579e02ccfd",
            )

    def test_rejects_stale_fixture_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            fixtures["dev"].write_bytes(fixtures["dev"].read_bytes() + b" ")

            with self.assertRaisesRegex(
                ValueError, "dev fixture hash does not match fixture bytes"
            ):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_rejects_cohort_ids_out_of_fixture_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            job, receipt = _read_job_and_receipt(job_path, receipt_path)
            request = job["request"]
            request["eval_cohorts"]["dev"][0], request["eval_cohorts"]["dev"][1] = (
                request["eval_cohorts"]["dev"][1],
                request["eval_cohorts"]["dev"][0],
            )
            _rewrite_job_and_receipt(job_path, receipt_path, job, receipt)

            with self.assertRaisesRegex(
                ValueError, "dev cohort IDs do not match its fixture order"
            ):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_rejects_missing_output_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            _, receipt = _read_job_and_receipt(job_path, receipt_path)
            del receipt["result"]["generation"]["prior_outputs"]
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "output arms"):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_rejects_prompt_mismatch_against_rendered_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            job, receipt = _read_job_and_receipt(job_path, receipt_path)
            job["request"]["dev_prompts"][0]["prompt"] += " Altered."
            _rewrite_job_and_receipt(job_path, receipt_path, job, receipt)

            with self.assertRaisesRegex(
                ValueError, "job prompt does not match rendered case dev-00"
            ):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_rejects_cross_arm_prompt_token_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            _, receipt = _read_job_and_receipt(job_path, receipt_path)
            receipt["result"]["generation"]["prior_outputs"][0]["prompt_token_ids"] = [999]
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "cross-arm prompt token IDs differ"):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_rejects_receipt_generation_cap_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            _, receipt = _read_job_and_receipt(job_path, receipt_path)
            receipt["result"]["generation"]["max_new_tokens"] = 512
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(
                ValueError, "generation settings do not match the quote-v4 contract"
            ):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_rejects_output_cap_mismatch_through_shared_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            _, receipt = _read_job_and_receipt(job_path, receipt_path)
            receipt["result"]["generation"]["base_outputs"][0]["max_new_tokens"] = 512
            receipt_path.write_bytes(_json_bytes(receipt))

            with self.assertRaisesRegex(ValueError, "generation settings differ from the contract"):
                evaluate_quote_v4.evaluate_report(
                    job_path,
                    receipt_path,
                    dev_path=fixtures["dev"],
                    test_path=fixtures["test"],
                )

    def test_retains_truncated_outputs_in_strict_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            _, receipt = _read_job_and_receipt(job_path, receipt_path)
            output = receipt["result"]["generation"]["base_outputs"][-1]
            output.update(
                {
                    "completion_token_ids": list(range(1024)),
                    "raw_text": '{"claims": [',
                    "response_text": '{"claims": [',
                    "generated_tokens": 1024,
                    "reached_generation_limit": True,
                }
            )
            receipt_path.write_bytes(_json_bytes(receipt))

            report = evaluate_quote_v4.evaluate_report(
                job_path,
                receipt_path,
                dev_path=fixtures["dev"],
                test_path=fixtures["test"],
            )
            strict = report["cohorts"]["test"]["arms"]["base"]["strict_raw_metrics"]
            case = strict["cases"]["test-23"]

            self.assertEqual(strict["truncated_case_ids"], ["test-23"])
            self.assertTrue(case["truncated"])
            self.assertTrue(case["parse_failure"])
            self.assertEqual(case["generated_tokens"], 1024)

    def test_cli_requires_all_paths_and_creates_output_exclusively(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as missing_args:
                evaluate_quote_v4.main([])
        self.assertEqual(missing_args.exception.code, 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, fixtures, _ = _write_inputs(root)
            output_path = root / "report.json"
            args = [
                "--job",
                str(job_path),
                "--receipt",
                str(receipt_path),
                "--dev-fixture",
                str(fixtures["dev"]),
                "--test-fixture",
                str(fixtures["test"]),
                "--output",
                str(output_path),
            ]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(evaluate_quote_v4.main(args), 0)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            first_contents = output_path.read_bytes()

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(evaluate_quote_v4.main(args), 1)
            self.assertEqual(output_path.read_bytes(), first_contents)


if __name__ == "__main__":
    unittest.main()
