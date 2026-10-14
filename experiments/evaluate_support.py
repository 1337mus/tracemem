"""Validate immutable Modal receipts and score the fictional NLI fixture locally."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from experiments.modal_baseline import payload_sha256
from experiments.modal_support import build_payload
from tracemem.support import load_support_challenge, score_support_cases

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_PAIRS = 64
_MAX_TOKENS = 512
_MAX_RECEIPT_BYTES = 10_000_000
_DIAGNOSTIC_NOTE = (
    "Synthetic diagnostic only; it is not independent human ground truth, calibration, "
    "or an acceptance policy. No claim may be auto-accepted."
)


def output_aliases_input(receipt_path: Path, output_path: Path) -> bool:
    return receipt_path.resolve() == output_path.resolve() or (
        output_path.exists() and output_path.samefile(receipt_path)
    )


def evaluate_receipt(receipt_bytes: bytes, *, max_cases: int = _MAX_PAIRS) -> dict[str, Any]:
    """Validate a raw receipt and score its exact covered fixture cases locally."""
    if (
        not isinstance(receipt_bytes, bytes)
        or not receipt_bytes
        or len(receipt_bytes) > _MAX_RECEIPT_BYTES
    ):
        raise ValueError("receipt is empty or exceeds the size limit")
    try:
        receipt = json.loads(
            receipt_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("receipt is not valid UTF-8 JSON") from None
    if type(receipt) is not dict:
        raise ValueError("receipt must be a JSON object")
    payload = build_payload(max_cases=max_cases)
    expected_keys = {
        "schema_version",
        "input_sha256",
        "payload_sha256",
        "model",
        "package_versions",
        "python_version",
        "torch_cuda_version",
        "gpu_name",
        "gpu_compute_seconds",
        "model_load_seconds",
        "remote_function_elapsed_seconds",
        "peak_gpu_memory_bytes",
        "modal_function_call_id",
        "modal_app_id",
        "client_call_wall_seconds_including_image_and_startup",
        "case_count",
        "outputs",
        "runner_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("receipt keys do not match the support schema")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("receipt schema version is unsupported")
    if receipt["input_sha256"] != payload["input_sha256"] or receipt[
        "payload_sha256"
    ] != payload_sha256(payload):
        raise ValueError("receipt fixture or payload hash does not match")
    if receipt["model"] != payload["model"]:
        raise ValueError("receipt model provenance does not match")
    runner_hash = receipt["runner_sha256"]
    if not isinstance(runner_hash, str) or _SHA256.fullmatch(runner_hash) is None:
        raise ValueError("receipt launch-time runner hash is invalid")
    versions = receipt["package_versions"]
    if (
        type(versions) is not dict
        or not versions
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in versions.items()
        )
    ):
        raise ValueError("receipt package versions are invalid")
    for name in (
        "python_version",
        "torch_cuda_version",
        "gpu_name",
        "modal_function_call_id",
        "modal_app_id",
    ):
        if not isinstance(receipt[name], str) or not receipt[name]:
            raise ValueError(f"receipt {name} is invalid")
    for name in ("gpu_compute_seconds", "model_load_seconds", "remote_function_elapsed_seconds"):
        value = receipt[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"receipt {name} is invalid")
    wall = receipt["client_call_wall_seconds_including_image_and_startup"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(wall)
        or wall < 0
    ):
        raise ValueError("receipt client call wall time is invalid")
    memory = receipt["peak_gpu_memory_bytes"]
    if type(memory) is not int or memory < 0:
        raise ValueError("receipt GPU memory is invalid")
    cases, _ = load_support_challenge()
    expected_ids = [item["case_id"] for item in payload["items"]]
    case_by_id = {case["case_id"]: case for case in cases}
    if type(receipt["case_count"]) is not int or receipt["case_count"] != len(expected_ids):
        raise ValueError("receipt case count does not match the payload")
    outputs = receipt["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(expected_ids):
        raise ValueError("receipt output count does not match the payload")
    seen: set[str] = set()
    probabilities: dict[str, list[float]] = {}
    for item in outputs:
        if type(item) is not dict or set(item) != {
            "case_id",
            "input_tokens",
            "logits",
            "probabilities",
        }:
            raise ValueError("receipt output keys are invalid")
        case_id = item["case_id"]
        if not isinstance(case_id, str) or case_id not in expected_ids or case_id in seen:
            raise ValueError("receipt case coverage is invalid")
        seen.add(case_id)
        token_count = item["input_tokens"]
        if type(token_count) is not int or not 1 <= token_count <= _MAX_TOKENS:
            raise ValueError("receipt token count is invalid or exceeds 512")
        logits, probs = item["logits"], item["probabilities"]
        if (
            not isinstance(logits, list)
            or not isinstance(probs, list)
            or len(logits) != 3
            or len(probs) != 3
        ):
            raise ValueError("receipt class scores are invalid")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (*logits, *probs)
        ):
            raise ValueError("receipt class scores must be finite numbers")
        if any(not 0 <= value <= 1 for value in probs) or not math.isclose(
            sum(probs), 1.0, abs_tol=1e-6
        ):
            raise ValueError("receipt probabilities are invalid")
        peak = max(logits)
        exp_logits = [math.exp(value - peak) for value in logits]
        softmax = [value / sum(exp_logits) for value in exp_logits]
        if any(
            not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
            for actual, expected in zip(probs, softmax, strict=True)
        ):
            raise ValueError("receipt probabilities are inconsistent with softmax logits")
        probabilities[case_id] = [float(value) for value in probs]
    if seen != set(expected_ids):
        raise ValueError("receipt case coverage is incomplete")
    selected_cases = [case_by_id[case_id] for case_id in expected_ids]
    return {
        "schema_version": 1,
        "input_sha256": payload["input_sha256"],
        "payload_sha256": payload_sha256(payload),
        "raw_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "runner_sha256": runner_hash,
        "model": payload["model"],
        "provenance": {
            "package_versions": versions,
            "python_version": receipt["python_version"],
            "torch_cuda_version": receipt["torch_cuda_version"],
            "gpu_name": receipt["gpu_name"],
            "modal_function_call_id": receipt["modal_function_call_id"],
            "modal_app_id": receipt["modal_app_id"],
            "client_call_wall_seconds_including_image_and_startup": wall,
            "gpu_compute_seconds": receipt["gpu_compute_seconds"],
            "model_load_seconds": receipt["model_load_seconds"],
            "remote_function_elapsed_seconds": receipt["remote_function_elapsed_seconds"],
            "peak_gpu_memory_bytes": memory,
        },
        "diagnostic_note": _DIAGNOSTIC_NOTE,
        "raw_outputs": outputs,
        "score": score_support_cases(selected_cases, probabilities),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("receipt contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")
