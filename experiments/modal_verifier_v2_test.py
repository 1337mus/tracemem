"""Frozen source-free inference runner for the verifier v2 blind test."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from types import FunctionType
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_KEY = "examples/synthetic/verifier-v2-test-v1.json"
_DATASET_PATH = _ROOT / _DATASET_KEY
_DATASET_SHA256 = "7f22e5cb39a9dee04438a09fd7bd9b5d9e81b3e86924422e5f8423b48ae606ec"
_RENDERER_PATH = _ROOT / "training/verifier_data.py"
_RENDERER_SHA256 = "ada22823e053042d6d20e533f35410d852855607c8f6d51fd3cab93d8a4234c6"
_REMOTE_HELPER_PATH = _ROOT / "experiments/verifier_v2_test_remote.py"
_REMOTE_HELPER_SHA256 = "7dd8e13defa8cb9163522983e76aa072d8894ae1e1575699749fcd56daf73ab5"
_REMOTE_HELPER_CONTAINER_PATH = "/root/verifier_v2_test_remote.py"
_ADAPTER_VOLUME_NAME = "tracemem-verifier-v2"
_ADAPTER_VOLUME_MOUNT = "/mnt/tracemem-verifier-v2"
_COUNTS = {"test_cases": 24, "test_candidates": 48}
_DECODE_SETTING_KEYS = (
    "do_sample",
    "enable_thinking",
    "max_new_tokens",
    "max_input_tokens",
    "attn_implementation",
    "seed",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def modal_function_options() -> dict[str, Any]:
    """Reuse the fixed source-free A10G envelope with the blind-test timeout."""
    from experiments.modal_verifier_transfer import modal_function_options as transfer_options

    options = transfer_options()
    options["timeout"] = 900
    return options


def _validated_candidate(training_payload: dict[str, Any], receipt_raw: bytes) -> dict[str, Any]:
    from experiments.evaluate_verifier_v2 import evaluate_report
    from experiments.modal_verifier_v2 import payload_sha256 as training_payload_sha256

    training_raw = _canonical_bytes(training_payload)
    report = evaluate_report(training_raw, receipt_raw)
    if type(report) is not dict or type(report.get("candidate")) is not dict:
        raise ValueError("training receipt did not produce a structurally validated candidate")
    candidate = report["candidate"]
    expected_keys = {
        "model",
        "settings",
        "training_payload_sha256",
        "training_receipt_sha256",
        "adapter_relative_path",
        "adapter_artifact_sha256",
    }
    if set(candidate) != expected_keys:
        raise ValueError("validated training candidate has an invalid shape")
    payload_hash = training_payload_sha256(training_payload)
    receipt_hash = hashlib.sha256(receipt_raw).hexdigest()
    if (
        candidate["training_payload_sha256"] != payload_hash
        or candidate["training_receipt_sha256"] != receipt_hash
        or candidate["model"] != training_payload.get("model")
        or candidate["settings"] != training_payload.get("settings")
        or candidate["adapter_relative_path"] != f"runs/{payload_hash}/adapter"
    ):
        raise ValueError("training candidate does not match the exact current receipt")
    artifacts = candidate["adapter_artifact_sha256"]
    if type(artifacts) is not dict or set(artifacts) != {
        "adapter_config.json",
        "adapter_model.safetensors",
    }:
        raise ValueError("validated training adapter artifact map is invalid")
    if any(
        type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in artifacts.values()
    ):
        raise ValueError("validated training adapter artifact hashes are invalid")
    return deepcopy(candidate)


def build_payload(training_receipt_path: Path) -> dict[str, Any]:
    """Build fixed blind-test prompts bound to one validated continued adapter receipt."""
    if not isinstance(training_receipt_path, Path):
        raise TypeError("training_receipt_path must be a pathlib.Path")
    dataset_raw = _DATASET_PATH.read_bytes()
    if hashlib.sha256(dataset_raw).hexdigest() != _DATASET_SHA256:
        raise ValueError("blind-test dataset SHA-256 is not allowlisted")
    renderer_hash = hashlib.sha256(_RENDERER_PATH.read_bytes()).hexdigest()
    if renderer_hash != _RENDERER_SHA256:
        raise ValueError("blind-test prompt renderer SHA-256 is not allowlisted")
    helper_hash = hashlib.sha256(_REMOTE_HELPER_PATH.read_bytes()).hexdigest()
    if helper_hash != _REMOTE_HELPER_SHA256:
        raise ValueError("blind-test remote helper SHA-256 is not allowlisted")
    receipt_raw = training_receipt_path.read_bytes()

    from experiments import modal_verifier_v2
    from training.verifier_data import render_candidate_prompt, validate_verifier_dataset

    training_payload = modal_verifier_v2.build_payload()
    candidate = _validated_candidate(training_payload, receipt_raw)
    dataset = json.loads(dataset_raw)
    cases = validate_verifier_dataset(dataset)
    if len(cases) != _COUNTS["test_cases"] or any(case["split"] != "test" for case in cases):
        raise ValueError("blind-test dataset must contain exactly 24 test cases")
    candidate_count = sum(len(case["candidates"]) for case in cases)
    if candidate_count != _COUNTS["test_candidates"]:
        raise ValueError("blind-test dataset must contain exactly 48 candidates")
    items = [
        {
            "item_id": f"{case['case_id']}::{row['candidate_id']}",
            "prompt": render_candidate_prompt(case, row["candidate_id"]),
        }
        for case in cases
        for row in case["candidates"]
    ]
    settings = {key: deepcopy(candidate["settings"][key]) for key in _DECODE_SETTING_KEYS}
    candidate_provenance = {
        "training_payload_sha256": candidate["training_payload_sha256"],
        "training_receipt_sha256": candidate["training_receipt_sha256"],
        "adapter_relative_path": candidate["adapter_relative_path"],
        "adapter_artifact_sha256": deepcopy(candidate["adapter_artifact_sha256"]),
    }
    return {
        "schema_version": 1,
        "mode": "blind_test_inference",
        "split": "test",
        "source_sha256": {
            _DATASET_KEY: _DATASET_SHA256,
            "training/verifier_data.py": renderer_hash,
            "experiments/verifier_v2_test_remote.py": helper_hash,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": deepcopy(candidate["model"]),
        "settings": settings,
        "prior_adapter": deepcopy(modal_verifier_v2._PRIOR_ADAPTER),
        "continued_adapter": {
            "volume_name": _ADAPTER_VOLUME_NAME,
            "volume_mount": _ADAPTER_VOLUME_MOUNT,
            **candidate_provenance,
        },
        "candidate_provenance": candidate_provenance,
        "counts": deepcopy(_COUNTS),
        "items": items,
        "training_performed": False,
    }


def _make_remote_function(
    prior_volume: Any,
    continued_volume: Any,
    payload: dict[str, Any],
    runner_sha256: str | None = None,
    *,
    helper_path: str | None = None,
) -> FunctionType:
    from experiments.verifier_v2_test_remote import _remote_compute, _verify_adapter_mount

    verify_adapter = FunctionType(
        _verify_adapter_mount.__code__,
        {"__name__": "__main__"},
        name="verify_verifier_v2_test_adapter",
    )
    function = FunctionType(
        _remote_compute.__code__,
        {
            "__name__": "__main__",
            "_EXPECTED_HELPER_SHA256": _REMOTE_HELPER_SHA256,
            "_EXPECTED_PAYLOAD_COMPACT": {
                key: deepcopy(payload[key])
                for key in (
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
                    "training_performed",
                )
            },
            "_EXPECTED_PAYLOAD_SHA256": payload_sha256(payload),
            "_EXPECTED_RUNNER_SHA256": runner_sha256 or payload["runner_sha256"],
            "_REMOTE_HELPER_PATH": helper_path or str(_REMOTE_HELPER_PATH),
            "_verify_adapter_mount": verify_adapter,
        },
        name="remote_verifier_v2_blind_test",
    )
    function.__doc__ = _remote_compute.__doc__
    return function


def _build_modal_function(payload: dict[str, Any]) -> tuple[Any, Any]:
    """Build a source-free one-call Modal app with both adapters mounted read-only."""
    import modal

    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    image = image.add_local_file(
        _REMOTE_HELPER_PATH,
        remote_path=_REMOTE_HELPER_CONTAINER_PATH,
        copy=True,
    )
    app = modal.App("tracemem-verifier-v2-blind-test")
    prior = modal.Volume.from_name(
        payload["prior_adapter"]["volume_name"], create_if_missing=False
    ).with_mount_options(read_only=True)
    continued = modal.Volume.from_name(
        payload["continued_adapter"]["volume_name"], create_if_missing=False
    ).with_mount_options(read_only=True)
    options = modal_function_options()
    options["volumes"] = {
        payload["prior_adapter"]["volume_mount"]: prior,
        payload["continued_adapter"]["volume_mount"]: continued,
    }
    function = app.function(image=image, **options)(
        _make_remote_function(
            prior,
            continued,
            payload,
            payload["runner_sha256"],
            helper_path=_REMOTE_HELPER_CONTAINER_PATH,
        )
    )
    return app, function


def _validate_current_payload(
    payload: dict[str, Any],
    training_receipt_path: Path,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    if type(payload) is not dict:
        raise ValueError("payload differs from the current fixed blind-test payload")
    expected = build_payload(training_receipt_path)
    if _canonical_bytes(payload) != _canonical_bytes(expected):
        raise ValueError("payload differs from the current fixed blind-test payload")
    digest = payload_sha256(payload)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        approved_payload_sha256 != digest
        or approved_runner_sha256 != runner_hash
        or payload.get("runner_sha256") != runner_hash
    ):
        raise ValueError(
            "exact current blind-test payload and runner SHA-256 approvals are required"
        )
    return expected, digest, runner_hash


def run_once(
    payload: dict[str, Any],
    *,
    training_receipt_path: Path,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run the one approved blind-test inference after current-state launch guards."""
    expected, digest, runner_hash = _validate_current_payload(
        payload,
        training_receipt_path,
        approved_payload_sha256,
        approved_runner_sha256,
    )
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    from experiments.sft_launch import (
        _acquire_launch_lock,
        _reserve_output,
        _write_receipt,
        verify_modal_inventory,
        verify_personal_workspace,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fd = _reserve_output(output_path)
    lock_fd: int | None = None
    remote_started = False
    started = __import__("time").perf_counter()
    try:
        lock_fd = _acquire_launch_lock()
        verify_personal_workspace()
        inventory = verify_modal_inventory()
        app, function = _build_modal_function(expected)
        with app.run(detach=False):
            remote_started = True
            result = function.remote(
                {
                    "payload": expected,
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                }
            )
        if (
            type(result) is not dict
            or result.get("status") != "success"
            or result.get("mode") != "blind_test_inference"
            or result.get("split") != "test"
            or result.get("payload_sha256") != digest
            or result.get("runner_sha256") != runner_hash
        ):
            raise TypeError("Modal returned an invalid blind-test result")
        receipt = {
            **result,
            "modal_app_id": getattr(app, "app_id", None),
            "modal_app_name": "tracemem-verifier-v2-blind-test",
            "client_elapsed_seconds": round(__import__("time").perf_counter() - started, 6),
            "launch_checks": {
                "personal_workspace_verified": True,
                "modal_inventory_checked": True,
                "modal_app_count": len(inventory),
            },
        }
        _write_receipt(output_fd, receipt)
        return receipt
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        failure = {
            "schema_version": 1,
            "status": "launch_failed",
            "mode": "blind_test_inference",
            "split": "test",
            "payload_sha256": digest,
            "runner_sha256": runner_hash,
            "remote_started": remote_started,
            "client_elapsed_seconds": round(__import__("time").perf_counter() - started, 6),
            "error": {"type": type(error).__name__, "message": safe_exception_message(error)},
        }
        _write_receipt(output_fd, failure)
        raise
    finally:
        if lock_fd is not None:
            __import__("os").close(lock_fd)
        __import__("os").close(output_fd)
