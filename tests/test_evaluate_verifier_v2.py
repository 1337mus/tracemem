"""Contract tests for strict scoring of the frozen verifier-v2 receipt."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments import evaluate_verifier_v2, modal_verifier_v2

_ROOT = Path(__file__).resolve().parents[1]
_V2_DATASET_KEY = "examples/synthetic/verifier-v2-train-dev.json"
_DATASET_PATHS = (
    _ROOT / _V2_DATASET_KEY,
    _ROOT / "examples/synthetic/verifier-transfer-v1.json",
    _ROOT / "examples/synthetic/verifier-pairs-v1.json",
)


def _build_payload() -> dict[str, Any]:
    """Use the real payload builder with fixture-time hashes for in-flight source edits."""
    temporary_pins = {
        key: hashlib.sha256((_ROOT / key).read_bytes()).hexdigest()
        for key in modal_verifier_v2._SOURCE_HASHES
    }
    with patch.dict(modal_verifier_v2._SOURCE_HASHES, temporary_pins):
        return modal_verifier_v2.build_payload()


def _label_rows() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for path in _DATASET_PATHS:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        for case in dataset["cases"]:
            if case["split"] == "test":
                continue
            for candidate in case["candidates"]:
                item_id = f"{case['case_id']}::{candidate['candidate_id']}"
                labels[item_id] = {
                    "support": candidate["support"],
                    "eligibility": candidate["eligibility"],
                }
    return labels


def _generated_row(item_id: str, label: dict[str, str], eos_id: int) -> dict[str, Any]:
    raw_output = json.dumps(label, separators=(",", ":"), ensure_ascii=False)
    input_ids = [151644, 872, 198]
    output_ids = [401, eos_id]
    return {
        "item_id": item_id,
        "raw_output": raw_output,
        "input_token_ids": input_ids,
        "input_tokens": len(input_ids),
        "output_token_ids": output_ids,
        "output_tokens": len(output_ids),
        "ended_with_eos": True,
        "hit_max_new_tokens": False,
        "finish_reason": "eos",
    }


def _outputs_for_cohort(
    payload: dict[str, Any],
    cohort: str,
    labels: dict[str, dict[str, str]],
    eos_id: int,
) -> list[dict[str, Any]]:
    return [
        _generated_row(item["item_id"], labels[item["item_id"]], eos_id)
        for item in payload["items"]
        if item["cohort"] == cohort
    ]


def _with_prediction(row: dict[str, Any], label: dict[str, str], eos_id: int) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    updated["raw_output"] = json.dumps(label, separators=(",", ":"), ensure_ascii=False)
    updated["output_token_ids"] = [401, eos_id]
    updated["output_tokens"] = 2
    updated["ended_with_eos"] = True
    updated["hit_max_new_tokens"] = False
    updated["finish_reason"] = "eos"
    return updated


def _valid_inputs() -> tuple[bytes, bytes, bytes]:
    payload = _build_payload()
    labels = _label_rows()
    reference = payload["prior_reference_outputs"]
    tokenizer = json.loads(
        (_ROOT / "examples/synthetic/verifier-transfer-remote-v1.json").read_text(encoding="utf-8")
    )["tokenizer"]
    eos_id = tokenizer["eos_token_id"]

    transfer_base = {row["item_id"]: copy.deepcopy(row) for row in reference["baseline_outputs"]}
    transfer_prior = {row["item_id"]: copy.deepcopy(row) for row in reference["tuned_outputs"]}
    arms: dict[str, list[dict[str, Any]]] = {"base": [], "prior": [], "continued": []}
    for cohort in ("new_dev", "transfer_regression", "original_paired_dev"):
        if cohort == "transfer_regression":
            arms["base"].extend(transfer_base.values())
            arms["prior"].extend(transfer_prior.values())
            arms["continued"].extend(
                _with_prediction(row, labels[item_id], eos_id)
                for item_id, row in transfer_base.items()
            )
        else:
            rows = _outputs_for_cohort(payload, cohort, labels, eos_id)
            arms["base"].extend(copy.deepcopy(rows))
            arms["prior"].extend(copy.deepcopy(rows))
            arms["continued"].extend(_outputs_for_cohort(payload, cohort, labels, eos_id))

    payload_sha = modal_verifier_v2.payload_sha256(payload)
    receipt = {
        "payload_sha256": payload_sha,
        **{
            key: copy.deepcopy(payload[key])
            for key in (
                "schema_version",
                "mode",
                "split",
                "runner_sha256",
                "model",
                "settings",
                "prior_adapter",
                "counts",
            )
        },
        "status": "success",
        "source_sha256": copy.deepcopy(payload["source_sha256"]),
        "prior_adapter_artifact_sha256": {
            "adapter_config.json": payload["prior_adapter"]["config_sha256"],
            "adapter_model.safetensors": payload["prior_adapter"]["safetensors_sha256"],
        },
        "prior_adapter_artifact_sha256_after_training": {
            "adapter_config.json": payload["prior_adapter"]["config_sha256"],
            "adapter_model.safetensors": payload["prior_adapter"]["safetensors_sha256"],
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
            "python": "3.12.10",
        },
        "tokenizer": tokenizer,
        "timings": {
            "model_load_seconds": 1.0,
            "baseline_seconds": 1.0,
            "prior_seconds": 1.0,
            "training_seconds": 5.0,
            "tuned_reload_seconds": 1.0,
            "tuned_seconds": 1.0,
            "total_seconds": 12.0,
        },
        "memory": {"gpu_name": "NVIDIA A10", "peak_allocated_bytes": 123},
        "training": {
            "row_count": 192,
            "epochs": 2,
            "microsteps": 384,
            "optimizer_steps": 96,
            "learning_rate": 0.00005,
            "microstep_losses": [1.25] * 384,
            "optimizer_step_losses": [1.2] * 96,
            "adapter_path": f"/mnt/tracemem-verifier-v2/runs/{payload_sha}/adapter",
            "trainable_parameter_count": 33_030_144,
            "trainable_parameter_names": [
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
            ],
            "adapter_artifact_sha256": {
                "adapter_config.json": "c" * 64,
                "adapter_model.safetensors": "d" * 64,
            },
            "adapter_artifact_sha256_after_reload": {
                "adapter_config.json": "c" * 64,
                "adapter_model.safetensors": "d" * 64,
            },
            "adapter_mutation": {
                "initial_state_sha256": "a" * 64,
                "final_state_sha256": "b" * 64,
                "changed": True,
            },
        },
        "baseline_outputs": arms["base"],
        "prior_outputs": arms["prior"],
        "tuned_outputs": arms["continued"],
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        json.dumps(payload["prior_reference_outputs"], ensure_ascii=False).encode("utf-8"),
    )


def _evaluate(payload_raw: bytes, receipt_raw: bytes) -> dict[str, Any]:
    temporary_pins = {
        key: hashlib.sha256((_ROOT / key).read_bytes()).hexdigest()
        for key in modal_verifier_v2._SOURCE_HASHES
    }
    with patch.dict(modal_verifier_v2._SOURCE_HASHES, temporary_pins):
        return evaluate_verifier_v2.evaluate_report(payload_raw, receipt_raw)


class VerifierV2EvaluatorTests(unittest.TestCase):
    def test_perfect_continuation_outputs_pass_every_frozen_development_gate(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        report = _evaluate(payload_raw, receipt_raw)

        self.assertEqual(set(report["arms"]), {"base", "prior", "continued"})
        self.assertTrue(report["criterion"]["passed"])
        continued = report["arms"]["continued"]["cohorts"]
        self.assertEqual(continued["new_dev"]["observed"]["supported_retained"], 24)
        self.assertEqual(
            continued["transfer_regression"]["observed"]["supported_eligible_retained"], 19
        )
        self.assertEqual(continued["original_paired_dev"]["observed"]["supported_retained"], 6)
        self.assertTrue(report["criterion"]["checks"]["all_arms_all_outputs_valid_untruncated"])
        sensitivity = continued["transfer_regression"][
            "eligibility_sensitivity_excluding_case_transfer_04"
        ]
        self.assertEqual(sensitivity["candidate_count"], 46)
        self.assertEqual(
            sensitivity["decisive_eligibility_total"] + sensitivity["uncertain_eligibility_total"],
            46,
        )
        self.assertNotIn(
            "eligibility_sensitivity_excluding_case_transfer_04",
            report["criterion"]["checks"],
        )
        self.assertEqual(
            report["candidate"]["adapter_relative_path"],
            f"runs/{report['payload_sha256']}/adapter",
        )
        self.assertEqual(
            report["candidate"]["adapter_artifact_sha256"],
            report["training"]["adapter_artifact_sha256"],
        )

    def test_eligibility_rejection_does_not_hide_an_unsupported_support_false_accept(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        payload = json.loads(payload_raw)
        receipt = json.loads(receipt_raw)
        labels = _label_rows()
        unsupported_eligible = next(
            item["item_id"]
            for item in payload["items"]
            if item["cohort"] == "new_dev"
            and labels[item["item_id"]] == {"support": "unsupported", "eligibility": "eligible"}
        )
        row = next(
            row for row in receipt["tuned_outputs"] if row["item_id"] == unsupported_eligible
        )
        row["raw_output"] = '{"support":"supported","eligibility":"ineligible"}'
        report = _evaluate(
            payload_raw,
            json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

        observed = report["arms"]["continued"]["cohorts"]["new_dev"]["observed"]
        self.assertEqual(observed["unsupported_eligible_accepted"], 0)
        self.assertEqual(observed["unsupported_accepted"], 1)
        self.assertFalse(
            report["criterion"]["checks"]["new_dev_unsupported_false_accepts_no_increase"]
        )
        self.assertFalse(report["criterion"]["passed"])

    def test_observed_counts_include_unsupported_eligible_denominators(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        payload = json.loads(payload_raw)
        labels = _label_rows()
        cohort_counts = {
            cohort: sum(
                item["cohort"] == cohort
                and labels[item["item_id"]] == {"support": "unsupported", "eligibility": "eligible"}
                for item in payload["items"]
            )
            for cohort in ("new_dev", "transfer_regression", "original_paired_dev")
        }

        report = _evaluate(payload_raw, receipt_raw)

        self.assertGreater(sum(cohort_counts.values()), 0)
        for arm in ("base", "prior", "continued"):
            for cohort, expected_total in cohort_counts.items():
                observed = report["arms"][arm]["cohorts"][cohort]["observed"]
                self.assertEqual(observed["unsupported_eligible_total"], expected_total)

    def test_malformed_and_truncated_outputs_are_reported_as_failed_gates(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        malformed = receipt["baseline_outputs"][0]
        malformed["raw_output"] = "not-json"
        truncated = receipt["tuned_outputs"][0]
        truncated["output_token_ids"] = [101] * 128
        truncated["output_tokens"] = 128
        truncated["ended_with_eos"] = False
        truncated["hit_max_new_tokens"] = True
        truncated["finish_reason"] = "length"

        report = _evaluate(
            payload_raw,
            json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

        base_new_dev = report["arms"]["base"]["cohorts"]["new_dev"]["observed"]
        continued_new_dev = report["arms"]["continued"]["cohorts"]["new_dev"]["observed"]
        self.assertEqual(base_new_dev["invalid_count"], 1)
        self.assertEqual(continued_new_dev["truncated_count"], 1)
        self.assertFalse(report["criterion"]["checks"]["all_arms_all_outputs_valid_untruncated"])
        self.assertFalse(report["criterion"]["passed"])

    def test_prior_input_token_ids_must_match_base_and_continued_by_item(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["prior_outputs"][0]["input_token_ids"] = [999, 998, 997]

        with self.assertRaisesRegex(ValueError, "across all three arms"):
            _evaluate(
                payload_raw,
                json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )

    def test_prior_output_token_counts_and_finish_metadata_are_validated(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["prior_outputs"][0]["output_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "token counts are inconsistent"):
            _evaluate(
                payload_raw,
                json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )

    def test_output_that_hits_generation_limit_is_not_counted_as_untruncated(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        eos_id = receipt["tokenizer"]["eos_token_id"]
        row = receipt["tuned_outputs"][0]
        row["output_token_ids"] = [101] * 127 + [eos_id]
        row["output_tokens"] = 128
        row["ended_with_eos"] = True
        row["hit_max_new_tokens"] = True
        row["finish_reason"] = "eos"

        report = _evaluate(
            payload_raw,
            json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

        observed = report["arms"]["continued"]["cohorts"]["new_dev"]["observed"]
        self.assertEqual(observed["valid_count"], 48)
        self.assertEqual(observed["truncated_count"], 0)
        self.assertFalse(observed["valid_untruncated"])
        self.assertFalse(report["criterion"]["checks"]["all_arms_all_outputs_valid_untruncated"])

    def test_optional_adapter_readme_is_validated_but_not_passed_as_a_model_weight(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        readme_hash = "e" * 64
        receipt["training"]["adapter_artifact_sha256"]["README.md"] = readme_hash
        receipt["training"]["adapter_artifact_sha256_after_reload"]["README.md"] = readme_hash

        report = _evaluate(
            payload_raw,
            json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

        self.assertEqual(report["training"]["adapter_artifact_sha256"]["README.md"], readme_hash)
        self.assertEqual(
            set(report["candidate"]["adapter_artifact_sha256"]),
            {"adapter_config.json", "adapter_model.safetensors"},
        )
        self.assertEqual(
            report["candidate"]["adapter_artifact_sha256"],
            {
                key: receipt["training"]["adapter_artifact_sha256"][key]
                for key in ("adapter_config.json", "adapter_model.safetensors")
            },
        )

    def test_payload_must_match_the_current_frozen_builder(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        payload = json.loads(payload_raw)
        payload["items"][0]["prompt"] += " altered"

        with self.assertRaisesRegex(ValueError, "does not exactly match the current frozen"):
            _evaluate(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                receipt_raw,
            )

    def test_payload_source_hash_must_match_the_local_pinned_source_bytes(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        payload = json.loads(payload_raw)
        payload["source_sha256"][_V2_DATASET_KEY] = "0" * 64
        receipt = json.loads(receipt_raw)
        receipt["payload_sha256"] = modal_verifier_v2.payload_sha256(payload)

        with patch.object(modal_verifier_v2, "build_payload", return_value=payload):
            with self.assertRaisesRegex(ValueError, "payload source SHA-256 does not match"):
                evaluate_verifier_v2.evaluate_report(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                    json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                )

    def test_runtime_gpu_name_must_be_a_supported_string(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        receipt["memory"]["gpu_name"] = ["NVIDIA A10"]

        with self.assertRaisesRegex(ValueError, "runtime hardware metadata is invalid"):
            _evaluate(
                payload_raw,
                json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )

    def test_training_schedule_lora_membership_and_prior_immutability_are_guarded(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        mutations: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
            (
                "row count",
                lambda receipt: receipt["training"].update(row_count=191),
                "training counts",
            ),
            ("epochs", lambda receipt: receipt["training"].update(epochs=1), "training counts"),
            (
                "microsteps",
                lambda receipt: receipt["training"].update(microsteps=383),
                "training counts",
            ),
            (
                "optimizer steps",
                lambda receipt: receipt["training"].update(optimizer_steps=95),
                "training counts",
            ),
            (
                "learning rate",
                lambda receipt: receipt["training"].update(learning_rate=0.0001),
                "learning rate",
            ),
            (
                "LoRA parameter count",
                lambda receipt: receipt["training"].update(trainable_parameter_count=512),
                "trainable parameter count",
            ),
            (
                "non-LoRA trainable parameter",
                lambda receipt: receipt["training"].update(
                    trainable_parameter_names=["base_model.model.embed_tokens.weight"]
                ),
                "outside the frozen LoRA targets",
            ),
            (
                "adapter did not mutate",
                lambda receipt: receipt["training"]["adapter_mutation"].update(changed=False),
                "does not prove the new adapter changed",
            ),
            (
                "reload artifact changed",
                lambda receipt: receipt["training"]["adapter_artifact_sha256_after_reload"].update(
                    {"adapter_model.safetensors": "0" * 64}
                ),
                "reloaded adapter artifact SHA-256 values do not match",
            ),
            (
                "prior artifact changed",
                lambda receipt: receipt["prior_adapter_artifact_sha256_after_training"].update(
                    {"adapter_model.safetensors": "0" * 64}
                ),
                "prior adapter artifact hashes changed",
            ),
        )
        for label, mutate, expected_message in mutations:
            with self.subTest(label=label):
                receipt = json.loads(receipt_raw)
                mutate(receipt)
                with self.assertRaisesRegex(ValueError, expected_message):
                    _evaluate(
                        payload_raw,
                        json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode(
                            "utf-8"
                        ),
                    )

    def test_historical_transfer_replay_mismatch_is_reported_as_a_failed_gate(self) -> None:
        payload_raw, receipt_raw, _ = _valid_inputs()
        receipt = json.loads(receipt_raw)
        row = next(
            row
            for row in receipt["baseline_outputs"]
            if row["item_id"].startswith("case_transfer_")
        )
        item_id = row["item_id"]
        row["raw_output"] = '{"support":"uncertain","eligibility":"eligible"}'

        report = _evaluate(
            payload_raw,
            json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

        self.assertFalse(report["criterion"]["checks"]["historical_transfer_replay_matches"])
        self.assertEqual(
            report["criterion"]["historical_transfer_replay_differences"],
            [{"arm": "base", "item_id": item_id}],
        )
        self.assertFalse(report["criterion"]["passed"])


if __name__ == "__main__":
    unittest.main()
