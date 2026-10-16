"""Offline contract tests for the fiction-only transfer evaluator."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

from experiments import evaluate_verifier_transfer, modal_verifier_transfer

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _ROOT / "examples/synthetic/verifier-transfer-v1.json"
_DATASET_SHA256 = "6696c78b35499933310eeabdf626d683d5af122524fb5e50640d85043e7d20e9"


def _outputs(dataset: dict[str, Any], *, retained: int, false_accepts: int) -> list[dict[str, Any]]:
    supported_index = 0
    unsupported_index = 0
    result: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        for candidate in case["candidates"]:
            support = candidate["support"]
            if support == "supported":
                prediction = "supported" if supported_index < retained else "unsupported"
                supported_index += 1
            else:
                prediction = "supported" if unsupported_index < false_accepts else "unsupported"
                unsupported_index += 1
            raw_output = json.dumps(
                {"support": prediction, "eligibility": candidate["eligibility"]},
                separators=(",", ":"),
            )
            result.append(
                {
                    "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
                    "raw_output": raw_output,
                    "finish_reason": "eos",
                    "input_token_ids": [101, 202, 303],
                    "input_tokens": 3,
                    "output_token_ids": [401, 2],
                    "output_tokens": 2,
                    "ended_with_eos": True,
                    "hit_max_new_tokens": False,
                }
            )
    return result


def _valid_inputs() -> tuple[bytes, bytes, bytes]:
    dataset_raw = _DATASET_PATH.read_bytes()
    payload = modal_verifier_transfer.build_payload()
    payload_raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    dataset = json.loads(dataset_raw)
    receipt = {
        "payload_sha256": modal_verifier_transfer.payload_sha256(payload),
        **{
            key: payload[key]
            for key in (
                "schema_version",
                "mode",
                "split",
                "runner_sha256",
                "model",
                "settings",
                "adapter",
            )
        },
        "status": "success",
        "adapter_artifact_sha256": {
            "adapter_config.json": payload["adapter"]["config_sha256"],
            "adapter_model.safetensors": payload["adapter"]["safetensors_sha256"],
        },
        "versions": {
            "modal": "1.6.1",
            "torch": "2.7.1",
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "accelerate": "1.6.0",
            "peft": "0.15.2",
            "bitsandbytes": "0.45.5",
            "safetensors": "0.5.3",
            "python": "3.12.8",
        },
        "tokenizer": {
            "class": "Qwen2TokenizerFast",
            "is_fast": True,
            "chat_template_sha256": "a" * 64,
            "eos_token_id": 2,
            "pad_token_id": 2,
        },
        "timings": {
            "model_load_seconds": 1.0,
            "baseline_seconds": 2.0,
            "tuned_seconds": 3.0,
            "total_seconds": 10.0,
        },
        "memory": {"gpu_name": "NVIDIA A10", "peak_allocated_bytes": 123},
        "baseline_outputs": _outputs(dataset, retained=21, false_accepts=2),
        "tuned_outputs": _outputs(dataset, retained=22, false_accepts=1),
    }
    receipt_raw = json.dumps(receipt, ensure_ascii=False).encode("utf-8")
    return dataset_raw, payload_raw, receipt_raw


class TransferEvaluatorTests(unittest.TestCase):
    def test_matching_frozen_receipt_scores_both_arms_and_prospective_criterion(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        self.assertEqual(hashlib.sha256(dataset_raw).hexdigest(), _DATASET_SHA256)

        report = evaluate_verifier_transfer.evaluate_report(dataset_raw, payload_raw, receipt_raw)

        self.assertEqual(set(report["arms"]), {"baseline", "tuned"})
        self.assertTrue(report["criterion"]["passed"])
        tuned = report["arms"]["tuned"]["score"]["aggregate"]
        self.assertEqual(tuned["support"]["by_reference"]["supported"]["retained"], 22)
        self.assertEqual(tuned["support"]["by_reference"]["unsupported"]["caught"], 23)
        self.assertEqual(tuned["validity"]["valid"], 48)
        self.assertEqual(tuned["eligibility"]["uncertain"]["total"], 6)

    def test_zero_length_input_is_rejected_as_outside_inference_bounds(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        receipt = json.loads(receipt_raw)
        for arm in ("baseline_outputs", "tuned_outputs"):
            receipt[arm][0]["input_token_ids"] = []
            receipt[arm][0]["input_tokens"] = 0

        with self.assertRaisesRegex(ValueError, "input token count"):
            evaluate_verifier_transfer.evaluate_report(
                dataset_raw,
                payload_raw,
                json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
            )

    def test_report_thresholds_do_not_mutate_later_evaluations(self) -> None:
        inputs = _valid_inputs()
        first_report = evaluate_verifier_transfer.evaluate_report(*inputs)
        first_report["criterion"]["thresholds"]["supported_retained_min"] = 0

        second_report = evaluate_verifier_transfer.evaluate_report(*inputs)

        self.assertEqual(second_report["criterion"]["thresholds"]["supported_retained_min"], 22)

    def test_receipt_must_prove_the_pinned_adapter_artifacts_were_loaded(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["adapter_artifact_sha256"]["adapter_model.safetensors"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "adapter artifact SHA-256"):
            evaluate_verifier_transfer.evaluate_report(
                dataset_raw,
                payload_raw,
                json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
            )

    def test_receipt_rejects_runtime_versions_outside_the_frozen_image(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["versions"]["torch"] = "2.7.0"

        with self.assertRaisesRegex(ValueError, "runtime version"):
            evaluate_verifier_transfer.evaluate_report(
                dataset_raw,
                payload_raw,
                json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
            )

    def test_receipt_rejects_invalid_tokenizer_provenance(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["tokenizer"]["chat_template_sha256"] = "invalid"

        with self.assertRaisesRegex(ValueError, "tokenizer provenance"):
            evaluate_verifier_transfer.evaluate_report(
                dataset_raw,
                payload_raw,
                json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
            )

    def test_receipt_rejects_unrelated_gpu_hardware(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["memory"]["gpu_name"] = "NVIDIA H100"

        with self.assertRaisesRegex(ValueError, "hardware metadata"):
            evaluate_verifier_transfer.evaluate_report(
                dataset_raw,
                payload_raw,
                json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
            )

    def test_report_exposes_uncertain_predictions_for_unsupported_claims(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["tuned_outputs"][1]["raw_output"] = json.dumps(
            {"support": "uncertain", "eligibility": "eligible"}, separators=(",", ":")
        )

        report = evaluate_verifier_transfer.evaluate_report(
            dataset_raw,
            payload_raw,
            json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
        )

        self.assertEqual(report["criterion"]["observed"]["tuned"]["unsupported_uncertain"], 1)

    def test_receipt_and_payload_tampering_are_rejected(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        tamper_cases = (
            ("dataset bytes", "SHA-256 is not allowlisted"),
            (
                "prompt with rebound receipt hash",
                "exactly match the current frozen transfer payload",
            ),
            ("missing token metadata", "required on every output row"),
            ("different input token IDs", "input token IDs must match"),
            ("missing candidate", "output IDs must match split candidates exactly one-to-one"),
            ("duplicate candidate", "duplicated in tuned token metadata"),
        )

        for tamper, error in tamper_cases:
            with self.subTest(tamper=tamper):
                case_dataset = dataset_raw
                case_payload = json.loads(payload_raw)
                case_receipt = json.loads(receipt_raw)
                if tamper == "dataset bytes":
                    case_dataset += b"\n"
                elif tamper == "prompt with rebound receipt hash":
                    case_payload["items"][0]["prompt"] += " changed"
                    case_receipt["payload_sha256"] = modal_verifier_transfer.payload_sha256(
                        case_payload
                    )
                elif tamper == "missing token metadata":
                    fields = (
                        "input_token_ids",
                        "input_tokens",
                        "output_token_ids",
                        "output_tokens",
                        "ended_with_eos",
                        "hit_max_new_tokens",
                    )
                    for arm in ("baseline_outputs", "tuned_outputs"):
                        for row in case_receipt[arm]:
                            for field in fields:
                                row.pop(field)
                elif tamper == "different input token IDs":
                    case_receipt["tuned_outputs"][0]["input_token_ids"] = [101, 202, 304]
                elif tamper == "missing candidate":
                    case_receipt["baseline_outputs"].pop()
                    case_receipt["tuned_outputs"].pop()
                else:
                    case_receipt["tuned_outputs"][-1] = dict(case_receipt["tuned_outputs"][0])

                case_payload_raw = (
                    payload_raw
                    if tamper not in {"prompt with rebound receipt hash"}
                    else json.dumps(case_payload, ensure_ascii=False).encode("utf-8")
                )
                with self.assertRaisesRegex(ValueError, error):
                    evaluate_verifier_transfer.evaluate_report(
                        case_dataset,
                        case_payload_raw,
                        json.dumps(case_receipt, ensure_ascii=False).encode("utf-8"),
                    )

    def test_criterion_fails_when_unsupported_accepts_do_not_improve(self) -> None:
        dataset_raw, payload_raw, receipt_raw = _valid_inputs()
        dataset = json.loads(dataset_raw)
        receipt = json.loads(receipt_raw)
        receipt["baseline_outputs"] = _outputs(dataset, retained=21, false_accepts=0)
        receipt["tuned_outputs"] = _outputs(dataset, retained=22, false_accepts=1)

        report = evaluate_verifier_transfer.evaluate_report(
            dataset_raw,
            payload_raw,
            json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
        )

        self.assertFalse(report["criterion"]["passed"])
        self.assertFalse(report["criterion"]["checks"]["fewer_unsupported_accepted"])
        self.assertTrue(report["criterion"]["checks"]["unsupported_rejection"])
