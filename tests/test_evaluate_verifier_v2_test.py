"""Offline contract tests for the frozen verifier-v2 blind-test evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments import evaluate_verifier_v2_test, modal_verifier_v2, modal_verifier_v2_test

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_KEY = "examples/synthetic/verifier-v2-test-v1.json"


def _fixture_dataset() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index in range(24):
        suffix = f"{index:03d}"
        source = f"For fixture {suffix}, the east ramp is closed until noon."
        message_id = f"message-{suffix}-user"
        labels = (
            ("supported", "eligible" if index < 12 else "ineligible"),
            ("unsupported", "eligible" if index < 12 else "uncertain"),
        )
        candidates = [
            {
                "candidate_id": f"candidate-{suffix}-{candidate_index}",
                "claim": claim,
                "message_id": message_id,
                "quote": source,
                "valid_from": None,
                "valid_to": None,
                "support": support,
                "eligibility": eligibility,
                "support_reason": "Fixture annotation.",
                "eligibility_reason": "Fixture policy label.",
            }
            for candidate_index, (claim, support, eligibility) in enumerate(
                (
                    (f"The operator says the ramp closes in fixture {suffix}.", *labels[0]),
                    (f"The ramp is confirmed closed in fixture {suffix}.", *labels[1]),
                )
            )
        ]
        cases.append(
            {
                "case_id": f"case-{suffix}",
                "split": "test",
                "scenario_group": f"scenario-{suffix}",
                "template_group": f"template-{suffix}",
                "entity_ids": [f"entity-{suffix}"],
                "phenomenon": "attribution",
                "messages": [
                    {
                        "session_id": f"session-{suffix}",
                        "message_id": f"message-{suffix}-assistant",
                        "role": "assistant",
                        "text": f"What does the update say in fixture {suffix}?",
                        "recorded_at": "2026-01-01T10:00:00Z",
                        "content_kind": "conversation",
                    },
                    {
                        "session_id": f"session-{suffix}",
                        "message_id": message_id,
                        "role": "user",
                        "text": source,
                        "recorded_at": "2026-01-01T10:01:00Z",
                        "content_kind": "conversation",
                    },
                ],
                "candidates": candidates,
            }
        )
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "scope_policy": "personal-memory-research-v1",
        "cases": cases,
    }


def _payload_and_labels(
    dataset: dict[str, Any], dataset_raw: bytes
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    labels: dict[str, dict[str, str]] = {}
    items: list[dict[str, str]] = []
    for case in dataset["cases"]:
        for candidate in case["candidates"]:
            item_id = f"{case['case_id']}::{candidate['candidate_id']}"
            labels[item_id] = {
                "support": candidate["support"],
                "eligibility": candidate["eligibility"],
            }
            items.append({"item_id": item_id, "prompt": f"Fixture prompt for {item_id}."})

    training_payload_sha = "a" * 64
    training_receipt_sha = "b" * 64
    artifacts = {"adapter_config.json": "c" * 64, "adapter_model.safetensors": "d" * 64}
    candidate_provenance = {
        "training_payload_sha256": training_payload_sha,
        "training_receipt_sha256": training_receipt_sha,
        "adapter_relative_path": f"runs/{training_payload_sha}/adapter",
        "adapter_artifact_sha256": artifacts,
    }
    renderer_hash = hashlib.sha256(modal_verifier_v2_test._RENDERER_PATH.read_bytes()).hexdigest()
    helper_hash = hashlib.sha256(
        modal_verifier_v2_test._REMOTE_HELPER_PATH.read_bytes()
    ).hexdigest()
    runner_hash = hashlib.sha256(Path(modal_verifier_v2_test.__file__).read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "mode": "blind_test_inference",
        "split": "test",
        "source_sha256": {
            _DATASET_KEY: hashlib.sha256(dataset_raw).hexdigest(),
            "training/verifier_data.py": renderer_hash,
            "experiments/verifier_v2_test_remote.py": helper_hash,
        },
        "runner_sha256": runner_hash,
        "model": copy.deepcopy(modal_verifier_v2._MODEL),
        "settings": {
            key: copy.deepcopy(modal_verifier_v2._SETTINGS[key])
            for key in modal_verifier_v2_test._DECODE_SETTING_KEYS
        },
        "prior_adapter": copy.deepcopy(modal_verifier_v2._PRIOR_ADAPTER),
        "continued_adapter": {
            "volume_name": "tracemem-verifier-v2",
            "volume_mount": "/mnt/tracemem-verifier-v2",
            **candidate_provenance,
        },
        "candidate_provenance": candidate_provenance,
        "counts": {"test_cases": 24, "test_candidates": 48},
        "items": items,
        "training_performed": False,
    }
    return payload, labels


def _row(item_id: str, label: dict[str, str], index: int, eos_id: int) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "raw_output": json.dumps(label, separators=(",", ":")),
        "finish_reason": "eos",
        "input_tokens": 4,
        "output_tokens": 2,
        "input_token_ids": [151644, 872, 198, index + 1],
        "output_token_ids": [401, eos_id],
        "ended_with_eos": True,
        "hit_max_new_tokens": False,
    }


def _valid_receipt(payload: dict[str, Any], labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    eos_id = 151645
    rows = [
        _row(item["item_id"], labels[item["item_id"]], index, eos_id)
        for index, item in enumerate(payload["items"])
    ]
    prior = payload["prior_adapter"]
    continued = payload["continued_adapter"]
    return {
        "schema_version": 1,
        "status": "success",
        "mode": "blind_test_inference",
        "split": "test",
        "payload_sha256": _canonical_sha(payload),
        "runner_sha256": payload["runner_sha256"],
        "source_sha256": copy.deepcopy(payload["source_sha256"]),
        "model": copy.deepcopy(payload["model"]),
        "settings": copy.deepcopy(payload["settings"]),
        "prior_adapter": copy.deepcopy(prior),
        "prior_adapter_artifact_sha256": {
            "adapter_config.json": prior["config_sha256"],
            "adapter_model.safetensors": prior["safetensors_sha256"],
        },
        "continued_adapter": copy.deepcopy(continued),
        "continued_adapter_artifact_sha256": copy.deepcopy(continued["adapter_artifact_sha256"]),
        "candidate_provenance": copy.deepcopy(payload["candidate_provenance"]),
        "training_performed": False,
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
        "tokenizer": {
            "class": "Qwen2Tokenizer",
            "is_fast": True,
            "chat_template_sha256": "e" * 64,
            "eos_token_id": eos_id,
            "pad_token_id": 151643,
        },
        "modal_function_call_id": "fc-fixture",
        "base_outputs": copy.deepcopy(rows),
        "prior_outputs": copy.deepcopy(rows),
        "continued_outputs": copy.deepcopy(rows),
        "timings": {
            "model_load_seconds": 1.0,
            "base_seconds": 1.0,
            "prior_seconds": 1.0,
            "continued_seconds": 1.0,
            "total_seconds": 5.0,
        },
        "hardware": {
            "gpu_name": "NVIDIA A10",
            "gpu_capability": [8, 6],
            "peak_allocated_bytes": 123,
        },
        "modal_app_id": "ap-fixture",
        "modal_app_name": "tracemem-verifier-v2-blind-test",
        "client_elapsed_seconds": 6.0,
        "launch_checks": {
            "personal_workspace_verified": True,
            "modal_inventory_checked": True,
            "modal_app_count": 1,
        },
    }


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _valid_inputs() -> Iterator[tuple[bytes, bytes, Path]]:
    dataset = _fixture_dataset()
    dataset_raw = json.dumps(dataset, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload, labels = _payload_and_labels(dataset, dataset_raw)
    receipt = _valid_receipt(payload, labels)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_path = root / "fixture-test.json"
        dataset_path.write_bytes(dataset_raw)
        training_receipt_path = root / "training-receipt.json"
        training_receipt_path.write_bytes(b"synthetic validated training receipt")
        with (
            patch.object(modal_verifier_v2_test, "_DATASET_PATH", dataset_path),
            patch.object(
                modal_verifier_v2_test,
                "_DATASET_SHA256",
                hashlib.sha256(dataset_raw).hexdigest(),
            ),
            patch.object(
                modal_verifier_v2_test,
                "build_payload",
                return_value=copy.deepcopy(payload),
            ),
        ):
            yield (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                json.dumps(receipt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                training_receipt_path,
            )


class VerifierV2BlindTestEvaluatorTests(unittest.TestCase):
    def test_perfect_receipt_passes_the_frozen_blind_test_gates(self) -> None:
        with _valid_inputs() as (payload_raw, receipt_raw, training_receipt_path):
            report = evaluate_verifier_v2_test.evaluate_report(
                payload_raw, receipt_raw, training_receipt_path
            )

        self.assertTrue(report["criterion"]["passed"])
        self.assertTrue(report["criterion"]["checks"]["all_arms_all_outputs_valid_untruncated"])
        self.assertEqual(set(report["arms"]), {"base", "prior", "continued"})
        continued = report["arms"]["continued"]["observed"]
        self.assertEqual(continued["candidate_count"], 48)
        self.assertEqual(continued["supported_eligible_total"], 12)
        self.assertEqual(continued["unsupported_eligible_total"], 12)
        self.assertFalse(report["training_performed"])
        self.assertNotIn("items", report)

    def test_payload_prompts_must_match_the_current_label_free_builder(self) -> None:
        with _valid_inputs() as (payload_raw, receipt_raw, training_receipt_path):
            payload = json.loads(payload_raw)
            payload["items"][0]["prompt"] += " changed"
            with self.assertRaisesRegex(ValueError, "current fixed blind-test payload"):
                evaluate_verifier_v2_test.evaluate_report(
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    receipt_raw,
                    training_receipt_path,
                )

    def test_every_arm_must_return_one_row_for_each_of_the_48_prompts(self) -> None:
        with _valid_inputs() as (payload_raw, receipt_raw, training_receipt_path):
            receipt = json.loads(receipt_raw)
            receipt["base_outputs"].pop()

            with self.assertRaisesRegex(ValueError, "base output IDs or fields"):
                evaluate_verifier_v2_test.evaluate_report(
                    payload_raw,
                    json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
                    training_receipt_path,
                )

    def test_receipt_provenance_training_flag_and_runtime_are_guarded(self) -> None:
        mutations: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
            (
                "training flag",
                lambda receipt: receipt.update(training_performed=True),
                "training_performed",
            ),
            (
                "continued adapter hash",
                lambda receipt: receipt["continued_adapter_artifact_sha256"].update(
                    {"adapter_model.safetensors": "f" * 64}
                ),
                "continued adapter artifact hashes",
            ),
            (
                "prior adapter hash",
                lambda receipt: receipt["prior_adapter_artifact_sha256"].update(
                    {"adapter_model.safetensors": "f" * 64}
                ),
                "prior adapter artifact hashes",
            ),
            (
                "runtime version",
                lambda receipt: receipt["versions"].update(torch="2.7.0"),
                "runtime version",
            ),
            (
                "malformed GPU name",
                lambda receipt: receipt["hardware"].update(gpu_name=["NVIDIA A10"]),
                "hardware metadata",
            ),
        )
        for label, mutate, expected in mutations:
            with (
                self.subTest(label=label),
                _valid_inputs() as (
                    payload_raw,
                    receipt_raw,
                    training_receipt_path,
                ),
            ):
                receipt = json.loads(receipt_raw)
                mutate(receipt)
                with self.assertRaisesRegex(ValueError, expected):
                    evaluate_verifier_v2_test.evaluate_report(
                        payload_raw,
                        json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
                        training_receipt_path,
                    )

    def test_all_three_arms_must_have_exact_prompt_tokens_and_output_metadata(self) -> None:
        with _valid_inputs() as (payload_raw, receipt_raw, training_receipt_path):
            receipt = json.loads(receipt_raw)
            receipt["prior_outputs"][0]["input_token_ids"][-1] += 1

            with self.assertRaisesRegex(ValueError, "input token IDs must match by item_id"):
                evaluate_verifier_v2_test.evaluate_report(
                    payload_raw,
                    json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
                    training_receipt_path,
                )

    def test_continued_arm_cannot_increase_prior_unsupported_false_accepts(self) -> None:
        with _valid_inputs() as (payload_raw, receipt_raw, training_receipt_path):
            receipt = json.loads(receipt_raw)
            row = next(
                item
                for item in receipt["continued_outputs"]
                if item["item_id"].endswith("candidate-000-1")
            )
            row["raw_output"] = '{"support":"supported","eligibility":"eligible"}'
            report = evaluate_verifier_v2_test.evaluate_report(
                payload_raw,
                json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
                training_receipt_path,
            )

        checks = report["criterion"]["checks"]
        self.assertFalse(checks["continued_unsupported_false_accepts_no_increase"])
        self.assertFalse(report["criterion"]["passed"])
        self.assertGreater(
            report["arms"]["continued"]["observed"]["unsupported_eligible_accepted"],
            report["arms"]["prior"]["observed"]["unsupported_eligible_accepted"],
        )


if __name__ == "__main__":
    unittest.main()
