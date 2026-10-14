"""Focused contract tests for the frozen synthetic QA receipt analyzer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from experiments.analyze_synthetic_qa import analyze_receipt
from experiments.modal_synthetic_qa import _canonical_bytes, build_payload
from experiments.prepare_synthetic_qa import prepare_synthetic_qa

RUNNER_SHA256 = "a" * 64
TEMPLATE_SHA256 = "b" * 64


def make_receipt(
    answers: dict[str, str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requests, _ = prepare_synthetic_qa()
    payload = build_payload()
    answers = answers or {}
    overrides = overrides or {}
    outputs = []
    for request in requests["requests"]:
        request_id = request["request_id"]
        outputs.append(
            {
                "request_id": request_id,
                "raw_output": answers.get(request_id, "fictional answer"),
                "input_tokens": 7,
                "generated_tokens": 2,
                "stop_reason": "eos",
                "hit_max_new_tokens": False,
                "truncated": False,
                **overrides.get(request_id, {}),
            }
        )
    return {
        "schema_version": 1,
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        "runner_sha256": RUNNER_SHA256,
        "client_elapsed_seconds": 1.25,
        "result": {
            "schema_version": 1,
            "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "config_sha256": hashlib.sha256(_canonical_bytes(payload["model"])).hexdigest(),
            "template_sha256": TEMPLATE_SHA256,
            "package_versions": {
                "torch": "2.7.1+cu126",
                "transformers": "4.51.3",
                "accelerate": "1.6.0",
            },
            "request_count": 72,
            "outputs": outputs,
            "modal_call_id": "call-fictional-123",
        },
    }


def analyze(receipt: dict[str, Any]) -> dict[str, Any]:
    return analyze_receipt(
        receipt,
        approved_runner_sha256=RUNNER_SHA256,
        approved_template_sha256=TEMPLATE_SHA256,
    )


class SyntheticQAAnalysisTests(unittest.TestCase):
    def test_reports_the_18_fixed_cells(self) -> None:
        report = analyze(make_receipt())

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(
            [
                (
                    cell["split"],
                    cell["mode"],
                    cell["byte_ceiling"],
                    cell["n"],
                    cell["answerable_n"],
                    cell["unknown_n"],
                )
                for cell in report["cells"]
            ],
            [
                (split, mode, ceiling, 4, 3, 1)
                for split in ("dev", "test")
                for mode in ("bm25", "vector", "hybrid")
                for ceiling in (0, 512, 2048)
            ],
        )

    def test_reports_literal_agreement_with_all_answerability_denominators(self) -> None:
        requests, references = prepare_synthetic_qa()
        reference_by_id = {
            reference["question_id"]: reference["answer"]
            for split in ("dev", "test")
            for reference in references["questions"][split]
        }
        answers = {
            request["request_id"]: reference_by_id[request["question_id"]]
            for request in requests["requests"]
        }

        report = analyze(make_receipt(answers))

        for cell in report["cells"]:
            self.assertEqual(
                cell["agreement"],
                {
                    "all": {"numerator": 4, "denominator": 4},
                    "answerable": {"numerator": 3, "denominator": 3},
                    "unknown": {"numerator": 1, "denominator": 1},
                },
            )

    def test_normalizes_only_case_and_whitespace(self) -> None:
        requests, references = prepare_synthetic_qa()
        reference_by_id = {
            reference["question_id"]: reference["answer"]
            for split in ("dev", "test")
            for reference in references["questions"][split]
        }
        answers = {
            request["request_id"]: reference_by_id[request["question_id"]]
            for request in requests["requests"]
        }
        cell_requests = [
            request
            for request in requests["requests"]
            if (request["split"], request["mode"], request["byte_ceiling"]) == ("dev", "bm25", 0)
        ]
        first, second = cell_requests[:2]
        answers[first["request_id"]] = " \n" + reference_by_id[first["question_id"]].upper() + "\t "
        answers[second["request_id"]] = "!" + reference_by_id[second["question_id"]]

        report = analyze(make_receipt(answers))
        cell = report["cells"][0]

        self.assertEqual(cell["agreement"]["all"], {"numerator": 3, "denominator": 4})

    def test_zero_context_unknown_rate_includes_answerable_questions(self) -> None:
        requests, references = prepare_synthetic_qa()
        reference_by_id = {
            reference["question_id"]: reference
            for split in ("dev", "test")
            for reference in references["questions"][split]
        }
        answers = {
            request["request_id"]: reference_by_id[request["question_id"]]["answer"]
            for request in requests["requests"]
        }
        answerable = next(
            request
            for request in requests["requests"]
            if (request["split"], request["mode"], request["byte_ceiling"]) == ("dev", "bm25", 0)
            and not reference_by_id[request["question_id"]]["is_abstention"]
        )
        answers[answerable["request_id"]] = "UNKNOWN"

        report = analyze(make_receipt(answers))

        self.assertEqual(
            report["cells"][0]["zero_context_unknown"],
            {"numerator": 2, "denominator": 4},
        )
        self.assertEqual(
            report["cells"][1]["zero_context_unknown"],
            {"numerator": 0, "denominator": 0},
        )

    def test_empty_output_stays_in_agreement_denominator(self) -> None:
        requests, references = prepare_synthetic_qa()
        reference_by_id = {
            reference["question_id"]: reference["answer"]
            for split in ("dev", "test")
            for reference in references["questions"][split]
        }
        answers = {
            request["request_id"]: reference_by_id[request["question_id"]]
            for request in requests["requests"]
        }
        selected = next(
            request
            for request in requests["requests"]
            if (request["split"], request["mode"], request["byte_ceiling"]) == ("dev", "bm25", 512)
        )
        answers[selected["request_id"]] = ""
        report = analyze(
            make_receipt(
                answers,
                {
                    selected["request_id"]: {
                        "generated_tokens": 0,
                        "stop_reason": "other",
                    }
                },
            )
        )
        cell = next(
            cell
            for cell in report["cells"]
            if (cell["split"], cell["mode"], cell["byte_ceiling"]) == ("dev", "bm25", 512)
        )

        self.assertEqual(cell["agreement"]["all"], {"numerator": 3, "denominator": 4})
        self.assertEqual(cell["empty_count"], 1)
        self.assertEqual(cell["finish_counts"], {"eos": 3, "length": 0, "other": 1})

    def test_length_stopped_output_stays_in_denominator_and_is_flagged(self) -> None:
        requests, references = prepare_synthetic_qa()
        reference_by_id = {
            reference["question_id"]: reference["answer"]
            for split in ("dev", "test")
            for reference in references["questions"][split]
        }
        answers = {
            request["request_id"]: reference_by_id[request["question_id"]]
            for request in requests["requests"]
        }
        selected = next(
            request
            for request in requests["requests"]
            if (request["split"], request["mode"], request["byte_ceiling"])
            == ("dev", "vector", 512)
        )
        answers[selected["request_id"]] = "partial output"
        report = analyze(
            make_receipt(
                answers,
                {
                    selected["request_id"]: {
                        "generated_tokens": 128,
                        "stop_reason": "length",
                        "hit_max_new_tokens": True,
                        "truncated": True,
                    }
                },
            )
        )
        cell = next(
            cell
            for cell in report["cells"]
            if (cell["split"], cell["mode"], cell["byte_ceiling"]) == ("dev", "vector", 512)
        )

        self.assertEqual(cell["agreement"]["all"], {"numerator": 3, "denominator": 4})
        self.assertEqual(cell["finish_counts"], {"eos": 3, "length": 1, "other": 0})
        self.assertEqual(cell["hit_max_new_tokens_count"], 1)
        self.assertEqual(cell["truncated_count"], 1)

    def test_reports_actual_token_minimum_maximum_and_sum(self) -> None:
        requests, references = prepare_synthetic_qa()
        reference_by_id = {
            reference["question_id"]: reference["answer"]
            for split in ("dev", "test")
            for reference in references["questions"][split]
        }
        answers = {
            request["request_id"]: reference_by_id[request["question_id"]]
            for request in requests["requests"]
        }
        selected = [
            request
            for request in requests["requests"]
            if (request["split"], request["mode"], request["byte_ceiling"])
            == ("dev", "hybrid", 2048)
        ]
        first, second = selected[:2]
        report = analyze(
            make_receipt(
                answers,
                {
                    first["request_id"]: {
                        "input_tokens": 1,
                        "generated_tokens": 0,
                        "stop_reason": "other",
                    },
                    second["request_id"]: {
                        "input_tokens": 2048,
                        "generated_tokens": 128,
                        "stop_reason": "length",
                        "hit_max_new_tokens": True,
                        "truncated": True,
                    },
                },
            )
        )
        cell = next(
            cell
            for cell in report["cells"]
            if (cell["split"], cell["mode"], cell["byte_ceiling"]) == ("dev", "hybrid", 2048)
        )

        self.assertEqual(cell["input_tokens"], {"min": 1, "max": 2048, "sum": 2063})
        self.assertEqual(cell["generated_tokens"], {"min": 0, "max": 128, "sum": 132})

    def test_preserves_all_per_request_outcomes_without_raw_text(self) -> None:
        report = analyze(make_receipt())

        self.assertEqual(len(report["outcomes"]), 72)
        self.assertEqual(len({outcome["request_id"] for outcome in report["outcomes"]}), 72)
        self.assertNotIn("raw_output", report["outcomes"][0])
        self.assertEqual(
            set(report["outcomes"][0]),
            {
                "request_id",
                "question_id",
                "split",
                "mode",
                "byte_ceiling",
                "is_answerable",
                "literal_agreement",
                "zero_context_unknown",
                "stop_reason",
                "hit_max_new_tokens",
                "truncated",
                "empty",
                "input_tokens",
                "generated_tokens",
            },
        )

    def test_rejects_duplicate_output_request_ids(self) -> None:
        receipt = make_receipt()
        receipt["result"]["outputs"][1]["request_id"] = receipt["result"]["outputs"][0][
            "request_id"
        ]

        with self.assertRaisesRegex(ValueError, "request IDs"):
            analyze(receipt)

    def test_rejects_invalid_or_nonmatching_runner_and_template_approvals(self) -> None:
        receipt = make_receipt()
        for runner, template in (
            ("A" * 64, TEMPLATE_SHA256),
            ("c" * 64, TEMPLATE_SHA256),
            (RUNNER_SHA256, "B" * 64),
            (RUNNER_SHA256, "c" * 64),
        ):
            with self.subTest(runner=runner[:1], template=template[:1]):
                with self.assertRaisesRegex(ValueError, "approval|provenance"):
                    analyze_receipt(
                        receipt,
                        approved_runner_sha256=runner,
                        approved_template_sha256=template,
                    )

    def test_rejects_client_timing_integer_outside_finite_float_range(self) -> None:
        receipt = make_receipt()
        receipt["client_elapsed_seconds"] = 10**400

        with self.assertRaisesRegex(ValueError, "timing"):
            analyze(receipt)

        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            output_path = Path(directory) / "report.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "src"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.analyze_synthetic_qa",
                    "--receipt",
                    str(receipt_path),
                    "--approve-runner-sha256",
                    RUNNER_SHA256,
                    "--approve-template-sha256",
                    TEMPLATE_SHA256,
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(
                completed.stderr.strip(),
                "synthetic QA analysis failed: invalid input or output",
            )

    def test_rejects_unpinned_payload_model_runtime_and_timing_metadata(self) -> None:
        mutations = (
            ("payload_sha256", "c" * 64),
            ("result.payload_sha256", "c" * 64),
            ("result.model_id", "other/model"),
            ("result.revision", "not-the-pinned-revision"),
            ("result.config_sha256", "c" * 64),
            ("result.package_versions.torch", "2.7.0"),
            ("result.package_versions.transformers", "4.51.4"),
            ("result.modal_call_id", "   "),
            ("result.request_count", 71),
            ("result.error", "remote call failed"),
            ("result.status", "error"),
            ("client_elapsed_seconds", float("nan")),
            ("client_elapsed_seconds", -1.0),
            ("client_elapsed_seconds", True),
        )
        for path, value in mutations:
            with self.subTest(path=path, value=repr(value)):
                receipt = make_receipt()
                target = receipt
                fields = path.split(".")
                for field in fields[:-1]:
                    target = target[field]
                target[fields[-1]] = value

                with self.assertRaises(ValueError):
                    analyze(receipt)

    def test_rejects_invalid_output_types_ranges_and_finish_combinations(self) -> None:
        mutations = (
            ({"input_tokens": True},),
            ({"generated_tokens": True},),
            ({"input_tokens": 0},),
            ({"generated_tokens": 129},),
            ({"raw_output": None},),
            ({"stop_reason": "timeout"},),
            ({"hit_max_new_tokens": True},),
            (
                {
                    "generated_tokens": 128,
                    "stop_reason": "length",
                    "hit_max_new_tokens": True,
                    "truncated": False,
                },
            ),
            ({"generated_tokens": 0, "stop_reason": "eos"},),
        )
        for (update,) in mutations:
            with self.subTest(update=update):
                receipt = make_receipt()
                receipt["result"]["outputs"][0].update(update)

                with self.assertRaises(ValueError):
                    analyze(receipt)

    def test_rejects_missing_and_extra_output_request_ids(self) -> None:
        missing = make_receipt()
        missing["result"]["outputs"].pop()
        extra = make_receipt()
        extra_row = dict(extra["result"]["outputs"][0], request_id="not-allowlisted")
        extra["result"]["outputs"].append(extra_row)

        for receipt in (missing, extra):
            with self.subTest(count=len(receipt["result"]["outputs"])):
                with self.assertRaises(ValueError):
                    analyze(receipt)

    def test_shuffling_outputs_preserves_cells_and_fixed_order_outcomes(self) -> None:
        ordered = make_receipt()
        shuffled = make_receipt()
        shuffled["result"]["outputs"].reverse()

        first, second = analyze(ordered), analyze(shuffled)

        self.assertEqual(first["cells"], second["cells"])
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertNotEqual(
            first["provenance"]["source_receipt_sha256"],
            second["provenance"]["source_receipt_sha256"],
        )

    def test_reports_receipt_and_fixed_source_hashes(self) -> None:
        receipt = make_receipt()
        report = analyze(receipt)
        provenance = report["provenance"]
        payload = build_payload()

        self.assertEqual(
            provenance,
            {
                "source_receipt_sha256": hashlib.sha256(_canonical_bytes(receipt)).hexdigest(),
                "request_bundle_sha256": (
                    "85d1334d0401ef2bdc920bf316f066bc3e74e9423ee5bec0aa63f4a559be2084"
                ),
                "reference_bundle_sha256": (
                    "70769ec36ac4c1e6a226d7432ea88241225e35d91d803cdc0a678fe2589226d2"
                ),
                "fixture_sha256": (
                    "a4d94c8e727c71e8c8b723e97c9674b1d0008ac9f35ba675485a7ea796349d78"
                ),
                "payload_sha256": (
                    "80e058c629de8f8a06ca4caa8f1093a8308a299c976b018ab7e8c5c62493b2ac"
                ),
                "runner_sha256": RUNNER_SHA256,
                "template_sha256": TEMPLATE_SHA256,
                "config_sha256": hashlib.sha256(_canonical_bytes(payload["model"])).hexdigest(),
                "analyzer_source_sha256": hashlib.sha256(
                    Path("experiments/analyze_synthetic_qa.py").read_bytes()
                ).hexdigest(),
            },
        )

    def test_discloses_provisional_label_and_inference_limits(self) -> None:
        report = analyze(make_receipt())

        self.assertEqual(report["label"], "provisional agent-only")
        self.assertEqual(report["question_count"], 8)
        self.assertEqual(report["matched_request_count"], 72)
        self.assertEqual(
            set(report["limitations"]),
            {
                "No semantic QA judging; literal reference comparison only.",
                "No matched-token budget; byte ceilings are UTF-8 bytes.",
                "No independent-scenario claim; cells repeat the same eight questions.",
                "References are provisional agent-authored and not independently reviewed.",
                "No pooled dev/test score or model ranking.",
                "Not evidence of benchmark compatibility, calibration, cost per correct answer, "
                "or training readiness.",
            },
        )

    def test_cli_writes_new_mode_0600_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            output_path = Path(directory) / "report.json"
            receipt_path.write_text(json.dumps(make_receipt()), encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "src"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.analyze_synthetic_qa",
                    "--receipt",
                    str(receipt_path),
                    "--approve-runner-sha256",
                    RUNNER_SHA256,
                    "--approve-template-sha256",
                    TEMPLATE_SHA256,
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(output_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["schema_version"], 1
            )

    def test_cli_rejects_duplicate_keys_and_nonfinite_json(self) -> None:
        valid = json.dumps(make_receipt())
        malformed = (
            valid.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1),
            valid.replace('"client_elapsed_seconds": 1.25', '"client_elapsed_seconds": NaN', 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            for index, content in enumerate(malformed):
                output_path = Path(directory) / f"report-{index}.json"
                receipt_path.write_text(content, encoding="utf-8")
                environment = os.environ.copy()
                environment["PYTHONPATH"] = "src"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.analyze_synthetic_qa",
                        "--receipt",
                        str(receipt_path),
                        "--approve-runner-sha256",
                        RUNNER_SHA256,
                        "--approve-template-sha256",
                        TEMPLATE_SHA256,
                        "--output",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    env=environment,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(output_path.exists())
                self.assertEqual(
                    completed.stderr.strip(),
                    "synthetic QA analysis failed: invalid input or output",
                )

    def test_cli_rejects_oversized_input_existing_output_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            output_path = root / "report.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "src"

            def invoke(path: Path) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.analyze_synthetic_qa",
                        "--receipt",
                        str(receipt_path),
                        "--approve-runner-sha256",
                        RUNNER_SHA256,
                        "--approve-template-sha256",
                        TEMPLATE_SHA256,
                        "--output",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    env=environment,
                    check=False,
                )

            receipt_path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
            self.assertEqual(invoke(output_path).returncode, 2)
            self.assertFalse(output_path.exists())

            receipt_path.write_text(json.dumps(make_receipt()), encoding="utf-8")
            output_path.write_text("preserve", encoding="utf-8")
            self.assertEqual(invoke(output_path).returncode, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserve")

            target = root / "target.json"
            target.write_text("preserve target", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            self.assertEqual(invoke(link).returncode, 2)
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "preserve target")
