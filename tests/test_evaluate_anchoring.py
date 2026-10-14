"""Offline extraction anchoring evaluator tests."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments import evaluate_anchoring, modal_baseline
from tracemem.challenge import load_challenge


def _receipt(prompt_version: str = "v1") -> dict[str, Any]:
    payload = modal_baseline.build_payload("extraction", prompt_version=prompt_version)
    cases, _ = load_challenge()
    outputs = [
        {
            "case_id": case["case_id"],
            "raw_output": json.dumps({"claims": case["expected"]}),
            "input_tokens": 100,
            "generated_tokens": 20,
            "hit_max_new_tokens": False,
        }
        for case in cases
    ]
    return {
        "schema_version": 1,
        "mode": "extraction",
        "prompt_version": prompt_version,
        "input_sha256": payload["input_sha256"],
        "payload_sha256": modal_baseline.payload_sha256(payload),
        "model": payload["model"],
        "package_versions": {"torch": "2.7.1", "transformers": "4.51.3"},
        "gpu_compute_seconds": 2.5,
        "model_load_seconds": 1.25,
        "remote_function_elapsed_seconds": 8.0,
        "peak_gpu_memory_bytes": 123_456,
        "case_count": len(outputs),
        "generated_outputs": outputs,
    }


def _encode(receipt: dict[str, Any]) -> bytes:
    return json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")


class EvaluateAnchoringTests(unittest.TestCase):
    def test_prompt_version_is_propagated_and_wrong_variant_is_rejected(self) -> None:
        raw = _encode(_receipt(prompt_version="explicit-v2"))

        report = evaluate_anchoring.evaluate_anchoring(raw, prompt_version="explicit-v2")

        self.assertEqual(report["prompt_version"], "explicit-v2")
        self.assertIn("not an independent evaluation", report["evaluation_scope"])
        with self.assertRaisesRegex(ValueError, "prompt version|payload hash"):
            evaluate_anchoring.evaluate_anchoring(raw, prompt_version="v1")

    def test_corrupt_gold_offset_is_repaired_without_changing_other_claim_fields(self) -> None:
        receipt = _receipt()
        cases, _ = load_challenge()
        expected_by_id = {case["case_id"]: case["expected"] for case in cases}
        row = next(
            output for output in receipt["generated_outputs"] if expected_by_id[output["case_id"]]
        )
        corrupted = json.loads(row["raw_output"])
        raw_claims = deepcopy(corrupted["claims"])
        span = corrupted["claims"][0]["evidence"][0]
        original_start, original_end = span["char_start"], span["char_end"]
        span["char_start"] += 1
        span["char_end"] += 1
        row["raw_output"] = json.dumps(corrupted)
        raw = _encode(receipt)

        report = evaluate_anchoring.evaluate_anchoring(raw)

        self.assertEqual(report["raw_receipt_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertLess(report["raw_result"]["exact_match"]["f1"], 1.0)
        self.assertGreater(report["raw_result"]["evidence_integrity"]["invalid_claims"], 0)
        self.assertEqual(report["anchored_result"]["exact_match"]["f1"], 1.0)
        self.assertEqual(report["anchored_result"]["evidence_integrity"]["invalid_claims"], 0)
        self.assertEqual(row["raw_output"], json.dumps(corrupted))

        diagnostics = report["cases"][row["case_id"]]["claims"][0]
        self.assertEqual(
            sum(len(item["repairs"]) for item in report["cases"][row["case_id"]]["claims"]),
            1,
        )
        self.assertEqual(diagnostics["repairs"][0]["old_start"], original_start + 1)
        self.assertEqual(diagnostics["repairs"][0]["old_end"], original_end + 1)
        self.assertEqual(diagnostics["repairs"][0]["new_start"], original_start)
        self.assertEqual(diagnostics["repairs"][0]["new_end"], original_end)
        anchored = json.loads(report["anchored_outputs"][row["case_id"]])["claims"]
        self.assertEqual(len(anchored), len(raw_claims))
        for outer_claim_index, (raw_claim, anchored_claim) in enumerate(
            zip(raw_claims, anchored, strict=True)
        ):
            self.assertEqual(
                {key: value for key, value in raw_claim.items() if key != "evidence"},
                {key: value for key, value in anchored_claim.items() if key != "evidence"},
            )
            for evidence_index, (raw_span, anchored_span) in enumerate(
                zip(raw_claim["evidence"], anchored_claim["evidence"], strict=True)
            ):
                self.assertEqual(
                    {
                        key: value
                        for key, value in raw_span.items()
                        if key not in {"char_start", "char_end"}
                    },
                    {
                        key: value
                        for key, value in anchored_span.items()
                        if key not in {"char_start", "char_end"}
                    },
                )
                expected_offsets = (
                    (original_start, original_end)
                    if outer_claim_index == 0 and evidence_index == 0
                    else (raw_span["char_start"], raw_span["char_end"])
                )
                self.assertEqual(
                    (anchored_span["char_start"], anchored_span["char_end"]),
                    expected_offsets,
                )

    def test_schema_invalid_output_is_preserved_verbatim(self) -> None:
        receipt = _receipt()
        row = receipt["generated_outputs"][0]
        row["raw_output"] = '{"unexpected":true}'

        report = evaluate_anchoring.evaluate_anchoring(_encode(receipt))

        self.assertEqual(report["anchored_outputs"][row["case_id"]], row["raw_output"])
        self.assertEqual(report["cases"][row["case_id"]]["status"], "schema_invalid")
        self.assertEqual(report["raw_result"]["schema_failures"], 1)
        self.assertEqual(report["anchored_result"]["schema_failures"], 1)

    def test_duplicate_keys_and_nan_are_parse_invalid_and_preserved(self) -> None:
        for raw_output in ('{"claims":[],"claims":[]}', '{"claims":[NaN]}'):
            with self.subTest(raw_output=raw_output):
                receipt = _receipt()
                row = receipt["generated_outputs"][0]
                row["raw_output"] = raw_output

                report = evaluate_anchoring.evaluate_anchoring(_encode(receipt))

                self.assertEqual(report["anchored_outputs"][row["case_id"]], raw_output)
                self.assertEqual(report["cases"][row["case_id"]]["status"], "parse_invalid")
                self.assertEqual(report["raw_result"]["parse_failures"], 1)
                self.assertEqual(report["anchored_result"]["parse_failures"], 1)

    def test_cli_writes_report_without_changing_receipt(self) -> None:
        raw = _encode(_receipt(prompt_version="explicit-v2"))
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "receipt.json"
            output_path = Path(temporary) / "report.json"
            input_path.write_bytes(raw)

            result = evaluate_anchoring.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--prompt-version",
                    "explicit-v2",
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(input_path.read_bytes(), raw)
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["raw_receipt_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(saved_report["prompt_version"], "explicit-v2")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate_anchoring.main(["--input", str(input_path), "--output", str(input_path)])
            self.assertEqual(input_path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
