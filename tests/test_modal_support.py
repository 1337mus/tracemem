"""Contract tests for the bounded independent NLI support runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

from experiments import modal_support
from tracemem.support import load_support_challenge, score_support_cases

_ROOT = Path(__file__).resolve().parents[1]


def _valid_receipt(max_cases: int = 64) -> tuple[dict[str, Any], dict[str, list[float]]]:
    payload = modal_support.build_payload(max_cases=max_cases)
    logits = [0.0, 2.0, -1.0]
    exponents = [math.exp(value) for value in logits]
    probabilities = [value / sum(exponents) for value in exponents]
    outputs = [
        {
            "case_id": item["case_id"],
            "input_tokens": 30,
            "logits": list(logits),
            "probabilities": list(probabilities),
        }
        for item in payload["items"]
    ]
    by_case = {item["case_id"]: list(probabilities) for item in payload["items"]}
    receipt = {
        "schema_version": 1,
        "input_sha256": payload["input_sha256"],
        "payload_sha256": modal_support.payload_sha256(payload),
        "model": deepcopy(payload["model"]),
        "package_versions": {"torch": "2.7.1", "transformers": "4.51.3"},
        "python_version": "3.12.0",
        "torch_cuda_version": "12.6",
        "gpu_name": "Tesla T4",
        "gpu_compute_seconds": 1.25,
        "model_load_seconds": 2.5,
        "remote_function_elapsed_seconds": 4.0,
        "peak_gpu_memory_bytes": 1_000_000,
        "modal_function_call_id": "fc-test",
        "modal_app_id": "ap-test",
        "client_call_wall_seconds_including_image_and_startup": 8.0,
        "runner_sha256": "a" * 64,
        "case_count": len(outputs),
        "outputs": outputs,
    }
    return receipt, by_case


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return json.dumps(receipt, separators=(",", ":")).encode()


class SupportModelConfigTests(unittest.TestCase):
    def test_model_is_pinned_and_license_limit_is_recorded(self) -> None:
        config = json.loads((_ROOT / "configs" / "support-model.json").read_text())

        self.assertEqual(config["model_id"], "cross-encoder/nli-deberta-v3-base")
        self.assertEqual(config["revision"], "6c749ce3425cd33b46d187e45b92bbf96ee12ec7")
        self.assertEqual(
            config["label_mapping"],
            {
                "0": "contradiction",
                "1": "entailment",
                "2": "neutral",
            },
        )
        self.assertEqual(config["license_tag"], "apache-2.0")
        self.assertFalse(config["standalone_license_file_found"])
        self.assertFalse(config["selected"])
        self.assertFalse(config["weights_downloaded_locally"])


class SupportPayloadTests(unittest.TestCase):
    def test_payload_contains_pairs_but_keeps_diagnostic_labels_local(self) -> None:
        payload = modal_support.build_payload()
        cases, fixture_hash = load_support_challenge()

        self.assertEqual(set(payload), {"schema_version", "input_sha256", "model", "items"})
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["input_sha256"], fixture_hash)
        self.assertEqual(payload["model"]["model_id"], "cross-encoder/nli-deberta-v3-base")
        self.assertEqual(len(payload["items"]), len(cases))
        for item, case in zip(payload["items"], cases, strict=True):
            self.assertEqual(set(item), {"case_id", "premise", "hypothesis"})
            self.assertEqual(
                item,
                {
                    "case_id": case["case_id"],
                    "premise": case["premise"],
                    "hypothesis": case["hypothesis"],
                },
            )
        self.assertNotIn("supported", json.dumps(payload))
        self.assertNotIn("diagnostic", json.dumps(payload))
        self.assertNotIn("scenario_id", json.dumps(payload))
        self.assertNotIn("tags", json.dumps(payload))

    def test_pair_count_cannot_exceed_hard_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "64"):
            modal_support.build_payload(max_cases=65)
        for invalid in (1, 3, 47):
            with self.subTest(max_cases=invalid), self.assertRaisesRegex(ValueError, "even prefix"):
                modal_support.build_payload(max_cases=invalid)
        self.assertEqual(len(modal_support.build_payload(max_cases=2)["items"]), 2)

    def test_remote_resource_envelope_is_single_container_and_source_free(self) -> None:
        options = modal_support.modal_function_options()

        self.assertEqual(options["gpu"], "T4")
        self.assertEqual(options["cpu"], (2, 2))
        self.assertEqual(options["memory"], (16_384, 16_384))
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["buffer_containers"], 0)
        self.assertEqual(options["retries"], 0)
        self.assertLessEqual(options["startup_timeout"], 300)
        self.assertLessEqual(options["timeout"], 300)
        self.assertEqual(options["scaledown_window"], 2)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_prepare_is_default_and_launch_requires_exact_payload_hash(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(modal_support.main([]), 0)
        prepared = json.loads(stdout.getvalue())
        self.assertFalse(prepared["remote_started"])
        self.assertEqual(
            prepared["runner_sha256"],
            hashlib.sha256(Path(modal_support.__file__).read_bytes()).hexdigest(),
        )

        with redirect_stderr(stderr):
            code = modal_support.main(["--launch", "--approve-payload-sha256", "0" * 64])
        self.assertEqual(code, 2)
        self.assertIn("exact prepared payload", stderr.getvalue())


class SupportReceiptTests(unittest.TestCase):
    def assert_receipt_rejected(self, receipt: dict[str, Any], message: str) -> None:
        with self.assertRaisesRegex(ValueError, message):
            modal_support.evaluate_receipt(_receipt_bytes(receipt))

    def test_valid_receipt_is_scored_and_retains_raw_outputs_and_provenance(self) -> None:
        receipt, probabilities = _valid_receipt()
        receipt_bytes = _receipt_bytes(receipt)
        cases, _ = load_support_challenge()

        result = modal_support.evaluate_receipt(receipt_bytes)

        self.assertEqual(result["runner_sha256"], "a" * 64)
        self.assertEqual(result["raw_outputs"], receipt["outputs"])
        self.assertEqual(result["score"], score_support_cases(cases, probabilities))
        self.assertEqual(result["payload_sha256"], receipt["payload_sha256"])
        self.assertEqual(result["provenance"]["gpu_name"], "Tesla T4")
        self.assertEqual(result["provenance"]["modal_app_id"], "ap-test")
        self.assertEqual(result["provenance"]["modal_function_call_id"], "fc-test")

    def test_case_coverage_rejects_duplicate_missing_and_extra_ids(self) -> None:
        original, _ = _valid_receipt()
        first_id = original["outputs"][0]["case_id"]

        duplicate = deepcopy(original)
        duplicate["outputs"][-1]["case_id"] = first_id
        self.assert_receipt_rejected(duplicate, "case coverage")

        missing = deepcopy(original)
        missing["outputs"].pop()
        self.assert_receipt_rejected(missing, "output count")

        extra = deepcopy(original)
        extra["outputs"][-1]["case_id"] = "case-not-in-current-fixture"
        self.assert_receipt_rejected(extra, "case coverage")

    def test_receipt_rejects_wrong_fixture_payload_and_model_provenance(self) -> None:
        original, _ = _valid_receipt()
        mutations = {
            "fixture_hash": lambda receipt: receipt.__setitem__("input_sha256", "c" * 64),
            "payload_hash": lambda receipt: receipt.__setitem__("payload_sha256", "d" * 64),
            "model_revision": lambda receipt: receipt["model"].__setitem__("revision", "e" * 40),
            "label_mapping": lambda receipt: receipt["model"]["label_mapping"].__setitem__(
                "1", "contradiction"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(provenance=name):
                receipt = deepcopy(original)
                mutate(receipt)
                expected = (
                    "model provenance"
                    if name.startswith("model") or name == "label_mapping"
                    else "hash"
                )
                self.assert_receipt_rejected(receipt, expected)

    def test_receipt_rejects_nonfinite_logits_and_probabilities(self) -> None:
        original, _ = _valid_receipt()
        for field, value in (("logits", float("nan")), ("probabilities", float("inf"))):
            with self.subTest(field=field):
                receipt = deepcopy(original)
                receipt["outputs"][0][field][0] = value
                with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
                    modal_support.evaluate_receipt(_receipt_bytes(receipt))

    def test_receipt_rejects_token_counts_outside_the_inference_limit(self) -> None:
        original, _ = _valid_receipt()
        for token_count in (0, -1, 513, True):
            with self.subTest(input_tokens=token_count):
                receipt = deepcopy(original)
                receipt["outputs"][0]["input_tokens"] = token_count
                self.assert_receipt_rejected(receipt, "token count")

    def test_receipt_rejects_inconsistent_softmax_probabilities(self) -> None:
        receipt, _ = _valid_receipt()
        receipt["outputs"][0]["probabilities"] = [0.3, 0.3, 0.4]
        self.assert_receipt_rejected(receipt, "softmax")

    def test_evaluation_writes_separate_report_without_changing_raw_receipt(self) -> None:
        receipt, _ = _valid_receipt()
        receipt_bytes = _receipt_bytes(receipt)
        expected = modal_support.evaluate_receipt(receipt_bytes)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "raw-receipt.json"
            output_path = Path(temp_dir) / "evaluation.json"
            input_path.write_bytes(receipt_bytes)

            code = modal_support.main(["--evaluate", str(input_path), "--output", str(output_path)])

            self.assertEqual(code, 0)
            self.assertEqual(input_path.read_bytes(), receipt_bytes)
            self.assertEqual(json.loads(output_path.read_text()), expected)

    def test_evaluation_cannot_overwrite_input_or_omit_output(self) -> None:
        receipt, _ = _valid_receipt()
        receipt_bytes = _receipt_bytes(receipt)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "receipt.json"
            input_path.write_bytes(receipt_bytes)
            before = input_path.read_bytes()
            for args in (
                ["--evaluate", str(input_path), "--output", str(input_path)],
                ["--evaluate", str(input_path)],
            ):
                with self.subTest(args=args), redirect_stderr(StringIO()):
                    self.assertEqual(modal_support.main(args), 2)
                    self.assertEqual(input_path.read_bytes(), before)

    def test_evaluation_rejects_hardlink_to_input_receipt(self) -> None:
        receipt, _ = _valid_receipt()
        receipt_bytes = _receipt_bytes(receipt)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "receipt.json"
            output_link = Path(temp_dir) / "report.json"
            input_path.write_bytes(receipt_bytes)
            os.link(input_path, output_link)

            with redirect_stderr(StringIO()):
                code = modal_support.main(
                    ["--evaluate", str(input_path), "--output", str(output_link)]
                )

            self.assertEqual(code, 2)
            self.assertEqual(input_path.read_bytes(), receipt_bytes)
            self.assertEqual(output_link.read_bytes(), receipt_bytes)


if __name__ == "__main__":
    unittest.main()
