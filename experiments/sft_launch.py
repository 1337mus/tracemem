"""One-use, source-free Modal launcher for the fixed technical SFT pilot.

The remote worker commits the training Volume in ``finally`` for ordinary Python
success and failure paths. A hard-kill timeout can bypass ``finally``.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FunctionType
from typing import Any

from experiments import modal_sft

_ROOT = Path(__file__).resolve().parents[1]
_LAUNCH_LOCK = _ROOT / ".local" / "sft-pilot-launch.lock"
_VOLUME_NAME = "tracemem-sft-pilot"
_VOLUME_MOUNT = "/mnt/tracemem-sft-pilot"
_QUOTE_VOLUME_NAME = "tracemem-sft-quote-v2"
_QUOTE_VOLUME_MOUNT = "/mnt/tracemem-sft-quote-v2"
_REMOTE_TRAINING_DIR = Path("/root/training")
_EXPECTED_JOB_METADATA: dict[str, Any] = {}
_EXPECTED_JOB_SHA256 = ""
_EXPECTED_MODULE_SHA256: dict[str, str] = {}
_REMOTE_MODE = ""
_REMOTE_VOLUME: Any = None
_RUN_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_REMOTE_FILES = (
    ("training/train_sft.py", "train_sft.py"),
    ("training/remote_sft.py", "remote_sft.py"),
)
_INVENTORY_KEYS = {
    "app_id",
    "description",
    "state",
    "tasks",
    "created_at",
    "stopped_at",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_job_sha256(job: dict[str, Any]) -> str:
    """Hash a job without its self-referential ``job_sha256`` field."""
    if type(job) is not dict:
        raise TypeError("job must be a dictionary")
    unsigned = dict(job)
    unsigned.pop("job_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _run_id_from_job(job: dict[str, Any]) -> str | None:
    request = job.get("request")
    if type(request) is not dict:
        raise ValueError("job request must be an object")
    mode = request.get("mode")
    profile = request.get("profile", "pilot")
    if type(profile) is not str or profile not in {"pilot", "quote-v2"}:
        raise ValueError("training job has an unsupported profile")
    if mode == "preflight":
        if request.get("remote_run_dir") is not None:
            raise ValueError("preflight jobs must not name a remote run directory")
        return None
    if mode != "train":
        raise ValueError("job mode must be preflight or train")
    remote_run_dir = request.get("remote_run_dir")
    if type(remote_run_dir) is not str:
        raise ValueError("training job must use the fixed remote run path")
    volume_mount = _QUOTE_VOLUME_MOUNT if profile == "quote-v2" else _VOLUME_MOUNT
    prefix = f"{volume_mount}/adapters/"
    if not remote_run_dir.startswith(prefix):
        raise ValueError("training job must use the fixed remote run path")
    run_id = remote_run_dir[len(prefix) :]
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("training job must use a single lowercase run-id slug")
    return run_id


def _validate_approved_job(job: dict[str, Any], approved_job_sha256: str) -> dict[str, Any]:
    if type(job) is not dict:
        raise TypeError("job must be a dictionary")
    if (
        type(approved_job_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", approved_job_sha256) is None
    ):
        raise ValueError("approved job SHA-256 must be 64 lowercase hexadecimal characters")
    actual_hash = canonical_job_sha256(job)
    if job.get("job_sha256") != actual_hash or actual_hash != approved_job_sha256:
        raise ValueError("exact approved job SHA-256 is required")

    run_id = _run_id_from_job(job)
    mode = job["request"]["mode"]
    profile = job["request"].get("profile", "pilot")
    rebuilt = modal_sft.build_payload(mode, run_id=run_id, profile=profile)
    if _canonical_bytes(job) != _canonical_bytes(rebuilt):
        raise ValueError("job differs from the current fixed SFT payload")
    return rebuilt


def validate_modal_inventory(value: str | object) -> list[dict[str, Any]]:
    """Fail closed unless Modal reports no apps or only stopped apps with no tasks."""
    try:
        inventory = json.loads(value) if type(value) is str else value
    except json.JSONDecodeError as error:
        raise ValueError("Modal app inventory is not valid JSON") from error
    if type(inventory) is not list:
        raise ValueError("Modal app inventory must be a JSON array")
    for index, app in enumerate(inventory):
        if type(app) is not dict or set(app) != _INVENTORY_KEYS:
            raise ValueError(f"Modal app inventory row {index} has an invalid shape")
        if (
            type(app["app_id"]) is not str
            or not app["app_id"]
            or type(app["description"]) is not str
            or app["state"] != "stopped"
            or type(app["tasks"]) is not str
            or app["tasks"] != "0"
            or (app["created_at"] is not None and type(app["created_at"]) is not str)
            or (app["stopped_at"] is not None and type(app["stopped_at"]) is not str)
        ):
            raise RuntimeError(f"Modal app inventory row {index} is malformed or not fully stopped")
    return inventory


def verify_personal_workspace() -> None:
    """Run the repository's fresh personal-workspace token check."""
    from experiments.scenario_payload import verify_personal_workspace as verify

    verify()


def verify_modal_inventory() -> list[dict[str, Any]]:
    """Fetch a fresh app inventory with the installed CLI, never PATH's CLI."""
    cli = Path(sys.prefix) / "bin" / "modal"
    if not cli.is_file():
        raise RuntimeError("installed Modal CLI is unavailable for a fresh app inventory")
    try:
        result = subprocess.run(
            [str(cli), "app", "list", "--json"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("fresh Modal app inventory check failed") from None
    if result.returncode != 0:
        raise RuntimeError("fresh Modal app inventory check failed")
    return validate_modal_inventory(result.stdout)


def _require_supported_python() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor != 12:
        raise RuntimeError(
            "Modal SFT launches require local Python 3.12 to match the remote image "
            f"(found {version.major}.{version.minor})"
        )


def _reserve_output(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        if isinstance(error, FileExistsError) or error.errno == errno.ELOOP:
            raise FileExistsError("receipt output path must be new") from None
        raise
    try:
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _acquire_launch_lock() -> int:
    _LAUNCH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(_LAUNCH_LOCK, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError("another SFT launcher is already in progress") from None
    return descriptor


def _write_receipt(descriptor: int, receipt: dict[str, Any]) -> None:
    data = _canonical_bytes(receipt) + b"\n"
    offset = 0
    while offset < len(data):
        offset += os.write(descriptor, data[offset:])
    os.fsync(descriptor)


def _remote_entry(request: dict[str, Any], job_sha256: str) -> dict[str, Any]:
    """Verify the two mounted sources, then import the remote-only SFT entry point."""
    import hashlib
    import importlib.util
    import json
    import sys
    from pathlib import Path

    if type(request) is not dict or job_sha256 != _EXPECTED_JOB_SHA256:
        raise RuntimeError("remote SFT job approval mismatch")
    full_job = {**_EXPECTED_JOB_METADATA, "request": request}
    job_bytes = json.dumps(
        full_job, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(job_bytes).hexdigest() != _EXPECTED_JOB_SHA256:
        raise RuntimeError("remote SFT job contents do not match the approved hash")

    training_dir = Path(_REMOTE_TRAINING_DIR)
    for relative, filename in _REMOTE_FILES:
        path = training_dir / filename
        expected = _EXPECTED_MODULE_SHA256.get(relative)
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"mounted SFT source hash mismatch: {filename}")

    module_name = "_tracemem_sft_train_sft"
    train_path = training_dir / "train_sft.py"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the mounted SFT entry point")
    loader = spec.loader
    train_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = train_module

    def invoke() -> Any:
        loader.exec_module(train_module)
        remote_runner = getattr(train_module, "run_remote_request", None)
        if not callable(remote_runner):
            raise RuntimeError("mounted SFT entry point has no remote request handler")
        return remote_runner(request)

    if _REMOTE_MODE == "train":
        try:
            result = invoke()
        finally:
            _REMOTE_VOLUME.commit()
    else:
        result = invoke()
    if type(result) is not dict:
        raise TypeError("remote SFT entry point returned a non-object result")
    return result


def _make_remote_entry(
    job: dict[str, Any], *, volume: Any, training_dir: Path = _REMOTE_TRAINING_DIR
) -> FunctionType:
    expected = {relative: job["module_sha256"][relative] for relative, _filename in _REMOTE_FILES}
    metadata = {key: value for key, value in job.items() if key not in {"job_sha256", "request"}}
    function = FunctionType(
        _remote_entry.__code__,
        {
            "__name__": "__main__",
            "_EXPECTED_MODULE_SHA256": expected,
            "_EXPECTED_JOB_METADATA": metadata,
            "_EXPECTED_JOB_SHA256": job["job_sha256"],
            "_REMOTE_MODE": job["request"]["mode"],
            "_REMOTE_TRAINING_DIR": str(training_dir),
            "_REMOTE_VOLUME": volume,
            "_REMOTE_FILES": _REMOTE_FILES,
        },
        name="remote_sft_entry",
    )
    function.__doc__ = _remote_entry.__doc__
    return function


def _build_modal_function(job: dict[str, Any]) -> tuple[Any, Any]:
    """Build the fixed two-file, source-free Modal package after launch guards."""
    import modal

    mode = job["request"]["mode"]
    image = modal.Image.debian_slim(python_version="3.12").pip_install(*job["dependency_pins"])
    for relative, filename in _REMOTE_FILES:
        image = image.add_local_file(
            _ROOT / relative,
            remote_path=f"{_REMOTE_TRAINING_DIR}/{filename}",
            copy=True,
        )

    profile = job["request"].get("profile", "pilot")
    if profile == "quote-v2":
        volume_name, volume_mount, app_name = (
            _QUOTE_VOLUME_NAME,
            _QUOTE_VOLUME_MOUNT,
            "tracemem-sft-quote-v2",
        )
    elif profile == "pilot":
        volume_name, volume_mount, app_name = _VOLUME_NAME, _VOLUME_MOUNT, "tracemem-sft-pilot"
    else:
        raise ValueError("job has an unsupported profile")
    volume = (
        modal.Volume.from_name(volume_name, create_if_missing=True) if mode == "train" else None
    )
    options = dict(job["resource_options"])
    options["env"] = {"TRACEMEM_SFT_REMOTE": "1"}
    if volume is not None:
        options["volumes"] = {volume_mount: volume}
    app = modal.App(app_name)
    remote_entry = _make_remote_entry(job, volume=volume)
    function = app.function(image=image, **options)(remote_entry)
    return app, function


def run_once(job: dict[str, Any], *, approved_job_sha256: str, output_path: Path) -> dict[str, Any]:
    """Run one approved job, recording a private result or a redacted failure receipt."""
    _require_supported_python()
    expected_job = _validate_approved_job(job, approved_job_sha256)
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")

    output_fd = _reserve_output(output_path)
    lock_fd: int | None = None
    started = time.perf_counter()
    remote_started = False
    workspace_verified = False
    inventory_checked = False
    inventory_count: int | None = None
    checks_started_at_utc = datetime.now(UTC).isoformat()
    checks_completed_at_utc: str | None = None
    try:
        lock_fd = _acquire_launch_lock()
        verify_personal_workspace()
        workspace_verified = True
        inventory = verify_modal_inventory()
        inventory_checked = True
        inventory_count = len(inventory)
        checks_completed_at_utc = datetime.now(UTC).isoformat()
        app, function = _build_modal_function(expected_job)
        with app.run(detach=False):
            remote_started = True
            result = function.remote(expected_job["request"], approved_job_sha256)
        if type(result) is not dict:
            raise TypeError("remote Modal function returned a non-object result")
        receipt = {
            "schema_version": 1,
            "mode": expected_job["request"]["mode"],
            "job_sha256": approved_job_sha256,
            "runner_sha256": expected_job["runner_sha256"],
            "remote_started": True,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_checks": {
                "started_at_utc": checks_started_at_utc,
                "workspace_verified": workspace_verified,
                "modal_inventory_checked": inventory_checked,
                "modal_app_count": inventory_count,
                "completed_at_utc": checks_completed_at_utc,
            },
            "result": result,
        }
        _write_receipt(output_fd, receipt)
        return receipt
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        failure = {
            "schema_version": 1,
            "mode": expected_job["request"]["mode"],
            "job_sha256": approved_job_sha256,
            "runner_sha256": expected_job["runner_sha256"],
            "remote_started": remote_started,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_checks": {
                "started_at_utc": checks_started_at_utc,
                "workspace_verified": workspace_verified,
                "modal_inventory_checked": inventory_checked,
                "modal_app_count": inventory_count,
                "completed_at_utc": checks_completed_at_utc,
            },
            "error": {
                "type": type(error).__name__,
                "message": safe_exception_message(error),
            },
        }
        _write_receipt(output_fd, failure)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(output_fd)
