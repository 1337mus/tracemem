"""Validate and score frozen-adapter transfer receipts on authored fiction only."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments import modal_verifier_transfer
from experiments.evaluate_verifier_pairs import (
    _output_rows,
    _strict_json,
    _validate_output_metadata,
    _write_exclusive,
)
from experiments.verifier_pairs import score_outputs

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = _ROOT / "examples/synthetic/verifier-transfer-v1.json"
_DATASET_SHA256 = "6696c78b35499933310eeabdf626d683d5af122524fb5e50640d85043e7d20e9"
_REQUIRED_OUTPUT_METADATA = {
    "input_token_ids",
    "input_tokens",
    "output_token_ids",
    "output_tokens",
    "ended_with_eos",
    "hit_max_new_tokens",
}
_PINNED_RUNTIME_VERSIONS = {
    "torch": "2.7.1",
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "accelerate": "1.6.0",
    "peft": "0.15.2",
    "bitsandbytes": "0.45.5",
    "safetensors": "0.5.3",
}
_A10_GPU_NAMES = {"NVIDIA A10", "NVIDIA A10G"}
_PROMISING_CRITERION = {
    "supported_retained_min": 22,
    "unsupported_rejected_min": 23,
    "all48_valid_untruncated": True,
    "strictly_fewer_unsupported_accepted_than_matched_base": True,
    "no_fewer_supported_retained_than_matched_base": True,
}


def _support_observations(score: dict[str, Any]) -> dict[str, int | bool]:
    aggregate = score["aggregate"]
    support = aggregate["support"]
    supported = support["by_reference"]["supported"]
    unsupported = support["by_reference"]["unsupported"]
    validity = aggregate["validity"]
    return {
        "supported_retained": supported["retained"],
        "supported_total": supported["total"],
        "unsupported_rejected": unsupported["caught"],
        "unsupported_total": unsupported["total"],
        "unsupported_accepted": support["confusion"]["unsupported"]["supported"],
        "unsupported_uncertain": support["confusion"]["unsupported"]["uncertain"],
        "valid_untruncated": validity["valid"] == 48 and validity["truncated"] == 0,
        "valid_count": validity["valid"],
        "truncated_count": validity["truncated"],
    }


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_runtime_metadata(receipt: dict[str, Any], payload: dict[str, Any]) -> None:
    expected_artifacts = {
        "adapter_config.json": payload["adapter"]["config_sha256"],
        "adapter_model.safetensors": payload["adapter"]["safetensors_sha256"],
    }
    if receipt.get("adapter_artifact_sha256") != expected_artifacts:
        raise ValueError("receipt adapter artifact SHA-256 values do not match the pinned adapter")

    versions = receipt.get("versions")
    if type(versions) is not dict or any(
        versions.get(package) != expected for package, expected in _PINNED_RUNTIME_VERSIONS.items()
    ):
        raise ValueError("receipt runtime version does not match the frozen image")
    python_version = versions.get("python")
    if type(python_version) is not str or not python_version.startswith("3.12."):
        raise ValueError("receipt runtime version does not match the frozen Python 3.12 image")
    if type(versions.get("modal")) is not str or not versions["modal"]:
        raise ValueError("receipt runtime version is missing Modal provenance")

    tokenizer = receipt.get("tokenizer")
    if (
        type(tokenizer) is not dict
        or type(tokenizer.get("class")) is not str
        or not tokenizer["class"]
        or type(tokenizer.get("is_fast")) is not bool
        or not _valid_sha256(tokenizer.get("chat_template_sha256"))
        or type(tokenizer.get("eos_token_id")) is not int
        or tokenizer["eos_token_id"] < 0
        or type(tokenizer.get("pad_token_id")) is not int
        or tokenizer["pad_token_id"] < 0
    ):
        raise ValueError("receipt tokenizer provenance is invalid")

    timings = receipt.get("timings")
    timing_keys = {
        "model_load_seconds",
        "baseline_seconds",
        "tuned_seconds",
        "total_seconds",
    }
    if type(timings) is not dict or any(
        type(timings.get(key)) not in (int, float)
        or not math.isfinite(timings[key])
        or timings[key] < 0
        for key in timing_keys
    ):
        raise ValueError("receipt runtime timings are invalid")
    if timings["total_seconds"] > 1200:
        raise ValueError("receipt runtime exceeded the 1200 second inference limit")

    memory = receipt.get("memory")
    if (
        type(memory) is not dict
        or type(memory.get("gpu_name")) is not str
        or memory["gpu_name"] not in _A10_GPU_NAMES
        or type(memory.get("peak_allocated_bytes")) is not int
        or memory["peak_allocated_bytes"] < 0
    ):
        raise ValueError("receipt runtime hardware metadata is invalid")


def _criterion(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = _support_observations(arms["baseline"]["score"])
    tuned = _support_observations(arms["tuned"]["score"])
    checks = {
        "supported_retention": tuned["supported_retained"]
        >= _PROMISING_CRITERION["supported_retained_min"],
        "unsupported_rejection": tuned["unsupported_rejected"]
        >= _PROMISING_CRITERION["unsupported_rejected_min"],
        "all48_valid_untruncated": bool(tuned["valid_untruncated"])
        and bool(baseline["valid_untruncated"]),
        "fewer_unsupported_accepted": tuned["unsupported_accepted"]
        < baseline["unsupported_accepted"],
        "no_worse_supported_retention": tuned["supported_retained"]
        >= baseline["supported_retained"],
    }
    return {
        "passed": all(checks.values()),
        "thresholds": dict(_PROMISING_CRITERION),
        "checks": checks,
        "observed": {"baseline": baseline, "tuned": tuned},
        "scope": (
            "Support counts use supported and unsupported references only; uncertain "
            "references remain separate in each full score."
        ),
    }


def evaluate_report(dataset_raw: bytes, payload_raw: bytes, receipt_raw: bytes) -> dict[str, Any]:
    """Validate the exact current transfer payload and score both inference arms."""
    dataset = _strict_json(dataset_raw, "dataset")
    payload = _strict_json(payload_raw, "payload")
    receipt = _strict_json(receipt_raw, "receipt")
    if type(payload) is not dict or type(receipt) is not dict:
        raise ValueError("payload and receipt must be JSON objects")

    dataset_sha = hashlib.sha256(dataset_raw).hexdigest()
    if dataset_sha != _DATASET_SHA256:
        raise ValueError("transfer dataset SHA-256 is not allowlisted")
    expected_payload = modal_verifier_transfer.build_payload()
    if modal_verifier_transfer.payload_sha256(payload) != modal_verifier_transfer.payload_sha256(
        expected_payload
    ):
        raise ValueError("payload does not exactly match the current frozen transfer payload")
    if (
        payload.get("source_sha256", {}).get("examples/synthetic/verifier-transfer-v1.json")
        != dataset_sha
    ):
        raise ValueError("payload dataset source SHA-256 does not match raw dataset bytes")
    if payload.get("mode") != "transfer" or payload.get("split") != "dev":
        raise ValueError("only transfer dev payloads are supported")

    payload_sha = modal_verifier_transfer.payload_sha256(payload)
    if receipt.get("payload_sha256") != payload_sha:
        raise ValueError("receipt payload SHA-256 does not match canonical payload")
    for key in ("schema_version", "mode", "split", "runner_sha256", "model", "settings", "adapter"):
        if receipt.get(key) != payload.get(key):
            raise ValueError(f"receipt {key} does not match payload")
    if receipt.get("status") != "success":
        raise ValueError("receipt status must be success")
    _validate_runtime_metadata(receipt, payload)

    arms = {
        "baseline": _output_rows(receipt.get("baseline_outputs"), "baseline_outputs"),
        "tuned": _output_rows(receipt.get("tuned_outputs"), "tuned_outputs"),
    }
    all_rows = [row for outputs in arms.values() for row in outputs]
    if any(not _REQUIRED_OUTPUT_METADATA <= set(row) for row in all_rows):
        raise ValueError("token IDs, counts, and finish flags are required on every output row")
    _validate_output_metadata(arms, receipt, payload["settings"])
    max_input_tokens = payload["settings"]["max_input_tokens"]
    if any(not 1 <= row["input_tokens"] <= max_input_tokens for row in all_rows):
        raise ValueError("input token count is outside the frozen inference limits")
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    scored_arms = {
        name: {
            "receipt_sha256": receipt_sha,
            "score": score_outputs(dataset, "dev", outputs),
        }
        for name, outputs in arms.items()
    }
    report = {
        "schema_version": 1,
        "mode": "transfer",
        "split": "dev",
        "dataset_sha256": dataset_sha,
        "payload_sha256": payload_sha,
        "receipt_sha256": receipt_sha,
        "runner_sha256": payload["runner_sha256"],
        "model": payload["model"],
        "settings": payload["settings"],
        "adapter": payload["adapter"],
        "runtime": {
            "versions": receipt["versions"],
            "tokenizer": receipt["tokenizer"],
            "timings": receipt["timings"],
            "memory": receipt["memory"],
        },
        "arms": scored_arms,
    }
    report["criterion"] = _criterion(scored_arms)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(
            args.dataset.read_bytes(), args.payload.read_bytes(), args.receipt.read_bytes()
        )
        _write_exclusive(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
