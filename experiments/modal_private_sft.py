"""Bounded local launcher for one explicitly approved private verifier SFT run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from stat import S_IMODE, S_ISDIR
from types import FunctionType
from typing import Any

from experiments import private_sft
from experiments.private_sft import payload_sha256
from experiments.private_sft_remote import (
    _accumulation_window_size,
    _build_sampling_plan,
    _completion_labels,
    _remote_compute,
    _remote_safe_entry,
    _safe_error_diagnostics,
    _token_ids_sha256,
    _training_metrics,
    _validate_payload_shape,
    _write_private_json_exclusive,
)

_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DIR = _ROOT / ".local"
_PRIVATE_OUTPUT_DIR = _LOCAL_DIR / "private-modal-sft"
_RUNNER_MODULES = (
    "experiments/private_sft.py",
    "experiments/modal_private_sft.py",
    "experiments/private_sft_remote.py",
    "experiments/verifier_v2_test_remote.py",
    "experiments/sft_launch.py",
    "experiments/modal_verifier_pairs.py",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("private SFT data is not canonical JSON") from None


def runner_sha256() -> str:
    """Hash every source file that affects the source-free private SFT call."""
    digest = hashlib.sha256()
    for relative_path in _RUNNER_MODULES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((_ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_approved_payload(
    payload: dict[str, Any], approved_payload_sha256: str, approved_runner_sha256: str
) -> tuple[dict[str, Any], str, str]:
    if type(payload) is not dict:
        raise ValueError("private SFT payload is invalid")
    encoded = _canonical_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    current_runner_hash = runner_sha256()
    if len(encoded) > 4_500_000:
        raise ValueError("private SFT payload exceeds its byte limit")
    if (
        type(approved_payload_sha256) is not str
        or _SHA256.fullmatch(approved_payload_sha256) is None
        or approved_payload_sha256 != digest
    ):
        raise ValueError("exact private SFT payload SHA-256 approval is required")
    if (
        type(approved_runner_sha256) is not str
        or _SHA256.fullmatch(approved_runner_sha256) is None
        or approved_runner_sha256 != current_runner_hash
    ):
        raise ValueError("exact current private SFT runner SHA-256 approval is required")
    output_volume = {
        "name": private_sft._OUTPUT_VOLUME_NAME,
        "mount": private_sft._OUTPUT_VOLUME_MOUNT,
    }
    expected = _validate_payload_shape(
        payload,
        deepcopy(private_sft._PROVENANCE),
        deepcopy(private_sft._MODEL),  # type: ignore[attr-defined]
        deepcopy(private_sft._SETTINGS),
        private_sft._prior_adapter(),
        output_volume,
    )
    return expected, digest, current_runner_hash


def _prepare_private_output_path(output_path: Path) -> Path:
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    absolute_path = output_path if output_path.is_absolute() else _ROOT / output_path
    absolute_path = Path(os.path.abspath(absolute_path)).resolve(strict=False)
    private_dir = _PRIVATE_OUTPUT_DIR.resolve(strict=False)
    if (
        absolute_path.parent != private_dir
        or not absolute_path.name
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", absolute_path.name) is None
    ):
        raise ValueError("private SFT receipt path must be a simple filename under .local")

    try:
        local_stat = _LOCAL_DIR.lstat()
    except FileNotFoundError:
        _LOCAL_DIR.mkdir(mode=0o700)
        local_stat = _LOCAL_DIR.lstat()
    if not S_ISDIR(local_stat.st_mode) or _LOCAL_DIR.resolve(strict=True) != _LOCAL_DIR.absolute():
        raise ValueError("private SFT .local directory is unsafe")
    try:
        _PRIVATE_OUTPUT_DIR.mkdir(mode=0o700, exist_ok=True)
        output_stat = _PRIVATE_OUTPUT_DIR.lstat()
    except OSError:
        raise ValueError("private SFT output directory is unsafe") from None
    if (
        not S_ISDIR(output_stat.st_mode)
        or _PRIVATE_OUTPUT_DIR.resolve(strict=True) != private_dir
        or S_IMODE(output_stat.st_mode) != 0o700
    ):
        raise ValueError("private SFT output directory must be a non-symlink 0700 directory")
    return absolute_path


def modal_function_options() -> dict[str, Any]:
    """Return the fixed source-free single-container A10G envelope."""
    return {
        "gpu": "A10G",
        "cpu": (4, 4),
        "memory": (32_768, 32_768),
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 600,
        "timeout": 1800,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def _make_remote_function(payload: dict[str, Any], runner_hash: str) -> FunctionType:
    """Create a source-free entry with only approved hashes and fixed metadata."""
    from experiments.verifier_v2_test_remote import _verify_adapter_mount

    expected_provenance = deepcopy(private_sft._PROVENANCE)
    expected_model = deepcopy(private_sft._MODEL)  # type: ignore[attr-defined]
    expected_settings = deepcopy(private_sft._SETTINGS)
    expected_prior = private_sft._prior_adapter()
    expected_output = {
        "name": private_sft._OUTPUT_VOLUME_NAME,
        "mount": private_sft._OUTPUT_VOLUME_MOUNT,
    }
    payload_hash = payload_sha256(payload)
    safe_error_diagnostics = FunctionType(
        _safe_error_diagnostics.__code__,
        {"__name__": "__main__"},
        name="private_sft_safe_error_diagnostics",
    )
    build_sampling_plan = FunctionType(
        _build_sampling_plan.__code__,
        {"__name__": "__main__"},
        name="private_sft_build_sampling_plan",
    )
    validate_shape = FunctionType(
        _validate_payload_shape.__code__,
        {"__name__": "__main__", "_BUILD_SAMPLING_PLAN": build_sampling_plan},
        name="validate_private_sft_payload",
    )
    completion_labels = FunctionType(
        _completion_labels.__code__,
        {"__name__": "__main__"},
        name="private_sft_completion_labels",
    )
    accumulation_window = FunctionType(
        _accumulation_window_size.__code__,
        {"__name__": "__main__"},
        name="private_sft_accumulation_window_size",
    )
    training_metrics = FunctionType(
        _training_metrics.__code__,
        {"__name__": "__main__"},
        name="private_sft_training_metrics",
    )
    token_hash = FunctionType(
        _token_ids_sha256.__code__,
        {"__name__": "__main__"},
        name="private_sft_token_ids_sha256",
    )
    write_private_json = FunctionType(
        _write_private_json_exclusive.__code__,
        {"__name__": "__main__"},
        name="write_private_sft_json_exclusive",
    )
    verify_adapter_mount = FunctionType(
        _verify_adapter_mount.__code__,
        {"__name__": "__main__"},
        name="verify_private_sft_prior_mount",
    )
    compute = FunctionType(
        _remote_compute.__code__,
        {
            "__name__": "__main__",
            "_EXPECTED_PAYLOAD_SHA256": payload_hash,
            "_EXPECTED_RUNNER_SHA256": runner_hash,
            "_APPROVED_RUNNER_SHA256": runner_hash,
            "_EXPECTED_PROVENANCE": expected_provenance,
            "_EXPECTED_MODEL": expected_model,
            "_EXPECTED_SETTINGS": expected_settings,
            "_EXPECTED_PRIOR_ADAPTER": expected_prior,
            "_EXPECTED_OUTPUT_VOLUME": expected_output,
            "_SAFE_ERROR_DIAGNOSTICS": safe_error_diagnostics,
            "_VALIDATE_PAYLOAD_SHAPE": validate_shape,
            "_COMPLETION_LABELS": completion_labels,
            "_ACCUMULATION_WINDOW_SIZE": accumulation_window,
            "_TRAINING_METRICS": training_metrics,
            "_TOKEN_IDS_SHA256": token_hash,
            "_VERIFY_ADAPTER_MOUNT": verify_adapter_mount,
            "_WRITE_PRIVATE_JSON": write_private_json,
            "_REMOTE_OUTPUT_VOLUME": _REMOTE_OUTPUT_VOLUME,
        },
        name="private_sft_compute",
    )
    entry = FunctionType(
        _remote_safe_entry.__code__,
        {
            "__name__": "__main__",
            "_REMOTE_COMPUTE": compute,
            "_APPROVED_PAYLOAD_SHA256": payload_hash,
            "_APPROVED_RUNNER_SHA256": runner_hash,
            "_SAFE_ERROR_DIAGNOSTICS": safe_error_diagnostics,
        },
        name="private_sft_entry",
    )
    entry.__doc__ = _remote_safe_entry.__doc__
    return entry


def _build_modal_function(payload: dict[str, Any], runner_hash: str) -> tuple[Any, Any]:
    """Build one source-free call with read-only prior and private output mounts."""
    import modal

    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*_IMAGE_PACKAGES)
        .env(
            {
                "HF_HUB_CACHE": "/mnt/tracemem-public-model-cache/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
    )
    app = modal.App("tracemem-private-sft")
    cache_volume = modal.Volume.from_name(
        "tracemem-public-model-cache-v1", create_if_missing=False
    ).with_mount_options(read_only=True)
    prior_volume = modal.Volume.from_name(
        private_sft._PRIOR_VOLUME_NAME, create_if_missing=False
    ).with_mount_options(read_only=True)
    output_volume = modal.Volume.from_name(private_sft._OUTPUT_VOLUME_NAME, create_if_missing=True)
    options = modal_function_options()
    options["volumes"] = {
        private_sft._PRIOR_VOLUME_MOUNT: prior_volume,
        private_sft._OUTPUT_VOLUME_MOUNT: output_volume,
        "/mnt/tracemem-public-model-cache": cache_volume,
    }
    global _REMOTE_OUTPUT_VOLUME
    _REMOTE_OUTPUT_VOLUME = output_volume
    function = app.function(image=image, **options)(_make_remote_function(payload, runner_hash))
    return app, function


_REMOTE_OUTPUT_VOLUME: Any = None


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run one exact-hash-approved private SFT request, without automatic retries."""
    expected, digest, runner_hash = _validate_approved_payload(
        payload, approved_payload_sha256, approved_runner_sha256
    )
    safe_output_path = _prepare_private_output_path(output_path)
    from experiments.sft_launch import (
        _acquire_launch_lock,
        _require_supported_python,
        _reserve_output,
        _write_receipt,
        verify_modal_inventory,
        verify_personal_workspace,
    )

    lock_fd: int | None = None
    output_fd: int | None = None
    remote_started = False
    stage = "lock"
    inventory_count: int | None = None
    result: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        lock_fd = _acquire_launch_lock()
        output_fd = _reserve_output(safe_output_path)
        stage = "preflight"
        _require_supported_python()
        verify_personal_workspace()
        inventory_count = len(verify_modal_inventory())
        stage = "build"
        app, function = _build_modal_function(expected, runner_hash)
        with app.run(detach=False):
            stage = "remote_call"
            remote_started = True
            result = function.remote(
                {"payload": expected, "payload_sha256": digest, "runner_sha256": runner_hash}
            )
        stage = "validate_result"
        if type(result) is not dict or result.get("status") not in {"success", "remote_failed"}:
            raise TypeError("private SFT worker returned an invalid result")
        if (
            result.get("mode") != "private_sft"
            or result.get("payload_sha256") != digest
            or result.get("runner_sha256") != runner_hash
        ):
            raise TypeError("private SFT worker result does not match the approved request")
        if result.get("status") == "success" and result.get("training_performed") is not True:
            raise TypeError("private SFT worker success omitted training confirmation")
        if result.get("status") == "remote_failed" and result.get("training_performed") is not None:
            raise TypeError("private SFT worker failure has an unsafe training status")
        stage = "write_success_receipt"
        receipt = {
            **result,
            "modal_app_id": getattr(app, "app_id", None),
            "modal_app_name": "tracemem-private-sft",
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_checks": {
                "personal_workspace_verified": True,
                "modal_inventory_checked": True,
                "modal_app_count": inventory_count,
            },
        }
        _write_receipt(output_fd, receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "launch_failed",
            "mode": "private_sft",
            "training_performed": None if remote_started else False,
            "payload_sha256": digest,
            "runner_sha256": runner_hash,
            "remote_started": remote_started,
            "stage": stage,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "error": {"type": type(error).__name__, "message": "private SFT launch failed"},
        }
        if type(result) is dict:
            for arm in ("base_outputs", "prior_outputs", "trained_outputs"):
                if arm in result:
                    failure[arm] = result[arm]
        if output_fd is not None:
            _write_receipt(output_fd, failure)
        return failure
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if output_fd is not None:
            os.close(output_fd)
