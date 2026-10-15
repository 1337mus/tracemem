"""Contract tests for local scoring of the prepared SFT pilot receipt."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from tracemem.prompts import render_explicit_prompt
from training.evaluate_sft import evaluate_report, main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "datasets/sft-pilot/eval.json"


def _canonical_hash(value: dict[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("job_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_inputs(root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    fixture_bytes = FIXTURE.read_bytes()
    fixture_payload = json.loads(fixture_bytes)
    dev_cases = [case for case in fixture_payload["cases"] if case["split"] == "dev"][:8]
    dev_prompts = [
        {"case_id": case["case_id"], "prompt": render_explicit_prompt(case)} for case in dev_cases
    ]
    job: dict[str, object] = {
        "schema_version": 1,
        "request": {"mode": "train", "dev_prompts": dev_prompts},
        "source_sha256": {
            "dev": hashlib.sha256(fixture_bytes).hexdigest(),
            "train": "a" * 64,
        },
    }
    job["job_sha256"] = _canonical_hash(job)

    def output_rows(suffix: str, truncated_id: str) -> list[dict[str, object]]:
        return [
            {
                "case_id": prompt["case_id"],
                "prompt_token_ids": [101, 102],
                "response_text": "malformed response"
                if prompt["case_id"] == suffix
                else '{"claims": []}',
                "raw_text": "this field must never be scored",
                "generated_tokens": 1024 if prompt["case_id"] == truncated_id else 2,
                "max_new_tokens": 1024,
                "reached_generation_limit": prompt["case_id"] == truncated_id,
            }
            for prompt in dev_prompts
        ]

    base_truncated = dev_prompts[1]["case_id"]
    tuned_truncated = dev_prompts[2]["case_id"]
    generation = {
        "max_new_tokens": 1024,
        "do_sample": False,
        "base_outputs": output_rows(dev_prompts[0]["case_id"], base_truncated),
        "tuned_outputs": output_rows(dev_prompts[0]["case_id"], tuned_truncated),
    }
    receipt = {
        "job_sha256": job["job_sha256"],
        "result": {"status": "completed", "generation": generation},
    }
    job_path, receipt_path = root / "job.json", root / "receipt.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return job_path, receipt_path, dev_cases


class EvaluateSftTests(unittest.TestCase):
    def test_report_scores_only_job_dev_cases_and_keeps_truncated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, cases = _write_inputs(root)

            report = evaluate_report(job_path, receipt_path)

            self.assertEqual(report["case_count"], 8)
            self.assertEqual(set(report["cases"]), {case["case_id"] for case in cases})
            self.assertEqual(report["base"]["diagnostic"]["parse_failures"], 1)
            self.assertEqual(len(report["base"]["diagnostic"]["case_breakdown"]), 8)
            self.assertEqual(report["base"]["truncated_case_ids"], [cases[1]["case_id"]])
            self.assertEqual(report["tuned"]["truncated_case_ids"], [cases[2]["case_id"]])
            self.assertEqual(
                report["cases"][cases[0]["case_id"]]["base"]["response_text"],
                "malformed response",
            )
            self.assertEqual(report["label_provenance"], "agent-authored provisional")
            self.assertEqual(report["human_reviewers"], 0)
            self.assertFalse(report["held_out"])
            self.assertIn("selective", report["exact_match_caveat"])
            self.assertIn("non-exhaustive", report["exact_match_caveat"])
            self.assertIn("false positive", report["exact_match_caveat"])
            self.assertEqual(
                report["input_sha256"],
                {
                    "job": hashlib.sha256(job_path.read_bytes()).hexdigest(),
                    "receipt": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    "dev_fixture": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                },
            )

    def test_receipt_must_bind_to_the_canonical_job_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["job_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "receipt job hash"):
                evaluate_report(job_path, receipt_path)

    def test_job_dev_source_hash_must_match_the_fixed_fixture_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            job["source_sha256"]["dev"] = "0" * 64
            job["job_sha256"] = _canonical_hash(job)
            receipt["job_sha256"] = job["job_sha256"]
            job_path.write_text(json.dumps(job), encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dev fixture hash"):
                evaluate_report(job_path, receipt_path)

    def test_job_must_include_the_dev_fixture_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            del job["source_sha256"]
            job["job_sha256"] = _canonical_hash(job)
            receipt["job_sha256"] = job["job_sha256"]
            job_path.write_text(json.dumps(job), encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dev fixture hash is missing"):
                evaluate_report(job_path, receipt_path)

    def test_job_prompts_must_match_the_rendered_dev_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            job["request"]["dev_prompts"][0]["prompt"] += " modified"
            job["job_sha256"] = _canonical_hash(job)
            receipt["job_sha256"] = job["job_sha256"]
            job_path.write_text(json.dumps(job), encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rendered dev case"):
                evaluate_report(job_path, receipt_path)

    def test_base_and_tuned_outputs_must_have_matching_prompt_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result"]["generation"]["tuned_outputs"][0]["prompt_token_ids"] = [999]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt token IDs differ"):
                evaluate_report(job_path, receipt_path)

    def test_output_generation_limit_must_match_shared_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result"]["generation"]["tuned_outputs"][0]["max_new_tokens"] = 1025
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "generation settings differ"):
                evaluate_report(job_path, receipt_path)

    def test_truncation_flag_must_match_generated_token_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result"]["generation"]["base_outputs"][0]["reached_generation_limit"] = True
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "truncation metadata"):
                evaluate_report(job_path, receipt_path)

    def test_cli_refuses_to_overwrite_an_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_path, receipt_path, _ = _write_inputs(root)
            output = root / "report.json"
            args = ["--job", str(job_path), "--receipt", str(receipt_path), "--output", str(output)]
            prior_umask = os.umask(0o377)
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(main(args), 0)
            finally:
                os.umask(prior_umask)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            original = output.read_bytes()

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(args), 1)

            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
