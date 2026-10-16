"""Strict scoring for one frozen verifier-v2 blind-test inference receipt."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from experiments import evaluate_verifier_v2, modal_verifier_v2_test
from experiments.evaluate_verifier_pairs import (
    _output_rows,
    _strict_json,
    _validate_output_metadata,
    _write_exclusive,
)
from experiments.verifier_pairs import score_outputs
from training.verifier_data import validate_verifier_dataset

_ROOT = Path(__file__).resolve().parents[1]
_CRITERIA_PATH = _ROOT / "docs/verifier-v2-criteria.json"
_CRITERIA_SHA256 = "053e9bf5d26846b2d6f9d3d9329bfe8e331329a1ceb8fb1ac888df0ef831f958"
_OUTPUT_FIELDS = {
    "item_id",
    "raw_output",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "input_token_ids",
    "output_token_ids",
    "ended_with_eos",
    "hit_max_new_tokens",
}
_ARM_FIELDS = {
    "base": "base_outputs",
    "prior": "prior_outputs",
    "continued": "continued_outputs",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "mode",
    "split",
    "payload_sha256",
    "runner_sha256",
    "source_sha256",
    "model",
    "settings",
    "prior_adapter",
    "prior_adapter_artifact_sha256",
    "continued_adapter",
    "continued_adapter_artifact_sha256",
    "candidate_provenance",
    "training_performed",
    "versions",
    "tokenizer",
    "modal_function_call_id",
    "base_outputs",
    "prior_outputs",
    "continued_outputs",
    "timings",
    "hardware",
    "modal_app_id",
    "modal_app_name",
    "client_elapsed_seconds",
    "launch_checks",
}
_TIMING_FIELDS = (
    "model_load_seconds",
    "base_seconds",
    "prior_seconds",
    "continued_seconds",
    "total_seconds",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_payload(
    payload_raw: bytes, training_receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = _strict_json(payload_raw, "blind-test payload")
    if type(payload) is not dict:
        raise ValueError("blind-test payload must be an object")
    expected = modal_verifier_v2_test.build_payload(training_receipt_path)
    if type(expected) is not dict or evaluate_verifier_v2._canonical_sha(
        payload
    ) != evaluate_verifier_v2._canonical_sha(expected):
        raise ValueError("payload does not match the current fixed blind-test payload")

    expected_keys = {
        "schema_version",
        "mode",
        "split",
        "source_sha256",
        "runner_sha256",
        "model",
        "settings",
        "prior_adapter",
        "continued_adapter",
        "candidate_provenance",
        "counts",
        "items",
        "training_performed",
    }
    if set(payload) != expected_keys:
        raise ValueError("blind-test payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "blind_test_inference"
        or payload["split"] != "test"
        or type(payload["training_performed"]) is not bool
        or payload["training_performed"] is not False
        or payload["counts"] != {"test_cases": 24, "test_candidates": 48}
    ):
        raise ValueError("blind-test payload mode, counts, or training flag is invalid")

    source_hashes = payload["source_sha256"]
    dataset_key = modal_verifier_v2_test._DATASET_KEY
    renderer_key = "training/verifier_data.py"
    helper_key = "experiments/verifier_v2_test_remote.py"
    if type(source_hashes) is not dict or set(source_hashes) != {
        dataset_key,
        renderer_key,
        helper_key,
    }:
        raise ValueError("blind-test payload source hash map is invalid")
    source_paths = {
        dataset_key: modal_verifier_v2_test._DATASET_PATH,
        renderer_key: modal_verifier_v2_test._RENDERER_PATH,
        helper_key: modal_verifier_v2_test._REMOTE_HELPER_PATH,
    }
    actual_source_hashes = {key: _sha256_file(path) for key, path in source_paths.items()}
    if (
        any(not evaluate_verifier_v2._valid_sha256(value) for value in source_hashes.values())
        or source_hashes != actual_source_hashes
        or source_hashes[dataset_key] != modal_verifier_v2_test._DATASET_SHA256
    ):
        raise ValueError("blind-test payload source SHA-256 values do not match frozen files")

    runner_sha = _sha256_file(Path(modal_verifier_v2_test.__file__))
    if (
        not evaluate_verifier_v2._valid_sha256(payload["runner_sha256"])
        or payload["runner_sha256"] != runner_sha
    ):
        raise ValueError("blind-test payload runner SHA-256 does not match current source")
    items = payload["items"]
    if type(items) is not list or len(items) != 48:
        raise ValueError("blind-test payload must contain exactly 48 prompts")
    item_ids: list[str] = []
    for item in items:
        if (
            type(item) is not dict
            or set(item) != {"item_id", "prompt"}
            or type(item["item_id"]) is not str
            or type(item["prompt"]) is not str
            or not item["prompt"].strip()
        ):
            raise ValueError("blind-test payload prompt rows are invalid")
        item_ids.append(item["item_id"])
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("blind-test payload prompt item IDs are duplicated")

    dataset_raw = modal_verifier_v2_test._DATASET_PATH.read_bytes()
    dataset = _strict_json(dataset_raw, "blind-test dataset")
    cases = validate_verifier_dataset(dataset)
    if (
        len(cases) != 24
        or any(case["split"] != "test" for case in cases)
        or sum(len(case["candidates"]) for case in cases) != 48
    ):
        raise ValueError("blind-test dataset must contain exactly 24 test cases and 48 candidates")
    expected_ids = {
        f"{case['case_id']}::{candidate['candidate_id']}"
        for case in cases
        for candidate in case["candidates"]
    }
    if set(item_ids) != expected_ids:
        raise ValueError("blind-test prompt IDs do not match the frozen dataset candidates")
    payload_sha = evaluate_verifier_v2._canonical_sha(payload)
    return payload, dataset, payload_sha


def _expected_prior_hashes(payload: dict[str, Any]) -> dict[str, str]:
    prior = payload.get("prior_adapter")
    if type(prior) is not dict:
        raise ValueError("blind-test prior adapter metadata is invalid")
    config_sha256 = prior.get("config_sha256")
    model_sha256 = prior.get("safetensors_sha256")
    if (
        type(config_sha256) is not str
        or not evaluate_verifier_v2._valid_sha256(config_sha256)
        or type(model_sha256) is not str
        or not evaluate_verifier_v2._valid_sha256(model_sha256)
    ):
        raise ValueError("blind-test prior adapter artifact hashes are invalid")
    return {
        "adapter_config.json": config_sha256,
        "adapter_model.safetensors": model_sha256,
    }


def _validate_runtime(receipt: dict[str, Any]) -> dict[str, Any]:
    versions = receipt.get("versions")
    expected_version_keys = set(evaluate_verifier_v2._PINNED_RUNTIME_VERSIONS) | {
        "modal",
        "python",
    }
    if (
        type(versions) is not dict
        or set(versions) != expected_version_keys
        or any(type(value) is not str for value in versions.values())
        or any(
            versions.get(package) != expected
            for package, expected in evaluate_verifier_v2._PINNED_RUNTIME_VERSIONS.items()
        )
        or not versions.get("modal")
        or not versions.get("python", "").startswith("3.12.")
    ):
        raise ValueError("blind-test runtime version does not match the frozen image")

    tokenizer = receipt.get("tokenizer")
    if (
        type(tokenizer) is not dict
        or set(tokenizer)
        != {"class", "is_fast", "chat_template_sha256", "eos_token_id", "pad_token_id"}
        or type(tokenizer.get("class")) is not str
        or not tokenizer["class"]
        or type(tokenizer.get("is_fast")) is not bool
        or not evaluate_verifier_v2._valid_sha256(tokenizer.get("chat_template_sha256"))
        or type(tokenizer.get("eos_token_id")) is not int
        or tokenizer["eos_token_id"] < 0
        or type(tokenizer.get("pad_token_id")) is not int
        or tokenizer["pad_token_id"] < 0
    ):
        raise ValueError("blind-test tokenizer provenance is invalid")

    timings = receipt.get("timings")
    if (
        type(timings) is not dict
        or set(timings) != set(_TIMING_FIELDS)
        or any(
            type(timings.get(key)) not in (int, float)
            or not math.isfinite(timings[key])
            or timings[key] < 0
            for key in _TIMING_FIELDS
        )
    ):
        raise ValueError("blind-test runtime timings are invalid")
    stage_total = sum(timings[key] for key in _TIMING_FIELDS if key != "total_seconds")
    if timings["total_seconds"] > 900 or timings["total_seconds"] + 0.01 < stage_total:
        raise ValueError("blind-test total timing is outside its recorded stages or limit")

    hardware = receipt.get("hardware")
    if (
        type(hardware) is not dict
        or set(hardware) != {"gpu_name", "gpu_capability", "peak_allocated_bytes"}
        or type(hardware.get("gpu_name")) is not str
        or hardware["gpu_name"] not in evaluate_verifier_v2._A10_GPU_NAMES
        or type(hardware.get("gpu_capability")) is not list
        or len(hardware["gpu_capability"]) != 2
        or any(type(part) is not int for part in hardware["gpu_capability"])
        or hardware["gpu_capability"] != [8, 6]
        or type(hardware.get("peak_allocated_bytes")) is not int
        or hardware["peak_allocated_bytes"] < 0
    ):
        raise ValueError("blind-test hardware metadata is invalid")
    if (
        type(receipt.get("modal_function_call_id")) is not str
        or not receipt["modal_function_call_id"]
    ):
        raise ValueError("blind-test Modal function call ID is invalid")

    return {"versions": versions, "tokenizer": tokenizer, "timings": timings, "hardware": hardware}


def _validate_launch(receipt: dict[str, Any]) -> dict[str, Any]:
    app_id = receipt.get("modal_app_id")
    if app_id is not None and (type(app_id) is not str or not app_id):
        raise ValueError("blind-test Modal app ID is invalid")
    if receipt.get("modal_app_name") != "tracemem-verifier-v2-blind-test":
        raise ValueError("blind-test Modal app name is invalid")
    elapsed = receipt.get("client_elapsed_seconds")
    if type(elapsed) is int:
        elapsed_seconds = float(elapsed)
    elif type(elapsed) is float:
        elapsed_seconds = elapsed
    else:
        raise ValueError("blind-test client timing is invalid")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("blind-test client timing is invalid")
    checks = receipt.get("launch_checks")
    if (
        type(checks) is not dict
        or set(checks)
        != {"personal_workspace_verified", "modal_inventory_checked", "modal_app_count"}
        or checks.get("personal_workspace_verified") is not True
        or checks.get("modal_inventory_checked") is not True
        or type(checks.get("modal_app_count")) is not int
        or checks["modal_app_count"] < 0
    ):
        raise ValueError("blind-test launch verification metadata is invalid")
    return {
        "modal_app_id": app_id,
        "modal_app_name": receipt["modal_app_name"],
        "launch_checks": checks,
    }


def _validate_receipt(
    receipt_raw: bytes, payload: dict[str, Any], payload_sha: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], str]:
    receipt = _strict_json(receipt_raw, "blind-test receipt")
    if type(receipt) is not dict:
        raise ValueError("blind-test receipt must be an object")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("blind-test receipt keys are invalid")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("status") != "success"
        or receipt.get("mode") != "blind_test_inference"
        or receipt.get("split") != "test"
        or receipt.get("payload_sha256") != payload_sha
        or receipt.get("runner_sha256") != payload["runner_sha256"]
    ):
        raise ValueError("blind-test receipt identity does not match the approved payload")
    if (
        type(receipt.get("training_performed")) is not bool
        or receipt["training_performed"] is not False
    ):
        raise ValueError("blind-test receipt training_performed must be false")
    provenance_fields = (
        "source_sha256",
        "model",
        "settings",
        "prior_adapter",
        "continued_adapter",
        "candidate_provenance",
    )
    for key in provenance_fields:
        if receipt.get(key) != payload.get(key):
            raise ValueError(f"blind-test receipt {key} does not match the payload")

    expected_prior = _expected_prior_hashes(payload)
    if receipt.get("prior_adapter_artifact_sha256") != expected_prior:
        raise ValueError("blind-test prior adapter artifact hashes do not match the payload")
    continued_adapter = payload.get("continued_adapter")
    continued_hashes = (
        continued_adapter.get("adapter_artifact_sha256")
        if type(continued_adapter) is dict
        else None
    )
    if (
        type(continued_hashes) is not dict
        or set(continued_hashes) != {"adapter_config.json", "adapter_model.safetensors"}
        or any(not evaluate_verifier_v2._valid_sha256(value) for value in continued_hashes.values())
        or receipt.get("continued_adapter_artifact_sha256") != continued_hashes
    ):
        raise ValueError("blind-test continued adapter artifact hashes do not match the candidate")

    runtime = _validate_runtime(receipt)
    launch = _validate_launch(receipt)
    item_ids = {item["item_id"] for item in payload["items"]}
    arms = {arm: _output_rows(receipt.get(field), field) for arm, field in _ARM_FIELDS.items()}
    for arm, rows in arms.items():
        ids = [row.get("item_id") for row in rows]
        if (
            len(rows) != 48
            or any(set(row) != _OUTPUT_FIELDS for row in rows)
            or any(type(item_id) is not str for item_id in ids)
            or len(ids) != len(set(ids))
            or set(ids) != item_ids
        ):
            raise ValueError(f"blind-test {arm} output IDs or fields do not match all 48 prompts")
    _validate_output_metadata(
        {"baseline": arms["base"], "prior": arms["prior"], "tuned": arms["continued"]},
        receipt,
        payload["settings"],
    )
    input_ids = {
        arm: {row["item_id"]: row["input_token_ids"] for row in rows} for arm, rows in arms.items()
    }
    if any(input_ids[arm] != input_ids["base"] for arm in ("prior", "continued")):
        raise ValueError("blind-test input token IDs must match by item_id across all arms")
    max_input_tokens = payload["settings"].get("max_input_tokens")
    if (
        type(max_input_tokens) is not int
        or max_input_tokens < 1
        or any(
            not 1 <= row["input_tokens"] <= max_input_tokens
            for rows in arms.values()
            for row in rows
        )
    ):
        raise ValueError("blind-test input token count exceeds the approved prompt limit")
    return arms, {**runtime, **launch}, hashlib.sha256(receipt_raw).hexdigest()


def _load_criteria() -> dict[str, Any]:
    raw = _CRITERIA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _CRITERIA_SHA256:
        raise ValueError("verifier v2 criteria SHA-256 is not the frozen value")
    criteria = _strict_json(raw, "verifier v2 criteria")
    if type(criteria) is not dict or criteria.get("schema_version") != 1:
        raise ValueError("verifier v2 blind-test criteria are invalid")
    blind_test = criteria.get("blind_test")
    cohorts = criteria.get("cohorts")
    if (
        type(blind_test) is not dict
        or blind_test.get("thresholds_same_as") != "new_dev"
        or type(cohorts) is not dict
        or type(cohorts.get("new_dev")) is not dict
    ):
        raise ValueError("verifier v2 blind-test criteria are invalid")
    return cast(dict[str, Any], cohorts["new_dev"])


def _criterion(arms: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    continued = arms["continued"]["observed"]
    prior = arms["prior"]["observed"]
    checks = {
        "all_arms_all_outputs_valid_untruncated": all(
            arm["observed"]["valid_untruncated"] for arm in arms.values()
        ),
        "continued_supported_retained": continued["supported_retained"]
        >= thresholds["supported_retained_min"],
        "continued_unsupported_rejected": continued["unsupported_rejected"]
        >= thresholds["unsupported_rejected_min"],
        "continued_supported_eligible_retained": continued["supported_eligible_retained"]
        >= thresholds["supported_eligible_retained_min"],
        "continued_unsupported_eligible_accepted": continued["unsupported_eligible_accepted"]
        <= thresholds["unsupported_eligible_accepted_max"],
        "continued_decisive_eligibility_exact": continued["decisive_eligibility_exact"]
        >= thresholds["decisive_eligibility_exact_min"],
        "continued_uncertain_eligibility_exact": continued["uncertain_eligibility_exact"]
        >= thresholds["uncertain_eligibility_exact_min"],
        "continued_unsupported_false_accepts_no_increase": continued["unsupported_accepted"]
        <= prior["unsupported_accepted"],
    }
    return {"passed": all(checks.values()), "thresholds": thresholds, "checks": checks}


def evaluate_report(
    payload_raw: bytes, receipt_raw: bytes, training_receipt_path: Path
) -> dict[str, Any]:
    """Validate the exact blind payload and score its one fixed three-arm receipt."""
    if not isinstance(training_receipt_path, Path):
        raise TypeError("training_receipt_path must be a pathlib.Path")
    payload, dataset, payload_sha = _validate_payload(payload_raw, training_receipt_path)
    arms_raw, runtime, receipt_sha = _validate_receipt(receipt_raw, payload, payload_sha)
    thresholds = _load_criteria()
    arms: dict[str, dict[str, Any]] = {}
    for arm, rows in arms_raw.items():
        score = score_outputs(dataset, "test", rows)
        arms[arm] = {
            "receipt_sha256": receipt_sha,
            "score": score,
            "observed": evaluate_verifier_v2._observed(
                score, rows, payload["settings"]["max_new_tokens"]
            ),
        }
    criterion = _criterion(arms, thresholds)
    return {
        "schema_version": 1,
        "mode": payload["mode"],
        "split": payload["split"],
        "payload_sha256": payload_sha,
        "receipt_sha256": receipt_sha,
        "runner_sha256": payload["runner_sha256"],
        "source_sha256": payload["source_sha256"],
        "model": payload["model"],
        "settings": payload["settings"],
        "prior_adapter": payload["prior_adapter"],
        "candidate_provenance": payload["candidate_provenance"],
        "training_performed": False,
        "runtime": runtime,
        "arms": arms,
        "criterion": criterion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(
            args.payload.read_bytes(), args.receipt.read_bytes(), args.training_receipt
        )
        _write_exclusive(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
