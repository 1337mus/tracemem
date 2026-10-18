"""Offline-by-default utility for caching pinned public Qwen3-4B artifacts in Modal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import FunctionType
from typing import Any

_MODEL_ID = "Qwen/Qwen3-4B"
_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
_VOLUME_NAME = "tracemem-public-model-cache-v1"
_VOLUME_MOUNT = "/mnt/tracemem-public-model-cache"
_OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".local" / "public-model-cache"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def build_payload() -> dict[str, Any]:
    """Bind public model identity, allowlist, package pins, and resource envelope."""
    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    return {
        "schema_version": 1,
        "model_id": _MODEL_ID,
        "revision": _REVISION,
        "artifact_files": list(_ARTIFACT_FILES),
        "packages": list(_IMAGE_PACKAGES),
        "cpu": 1,
        "max_containers": 1,
        "retries": 0,
        "timeout": 600,
    }


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _remote_probe(request: dict[str, Any], payload_hash: str, source_hash: str) -> dict[str, Any]:
    """Download only allowlisted public artifacts; never load or evaluate weights."""
    import hashlib as remote_hashlib
    import json as remote_json

    runtime = globals()
    try:
        raw = remote_json.dumps(
            request, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if (
            type(request) is not dict
            or remote_hashlib.sha256(raw).hexdigest() != payload_hash
            or payload_hash != runtime.get("_APPROVED_PAYLOAD_SHA256")
            or source_hash != runtime.get("_APPROVED_SOURCE_SHA256")
            or request.get("model_id") != _MODEL_ID
            or request.get("revision") != _REVISION
            or request.get("artifact_files") != list(_ARTIFACT_FILES)
        ):
            return {"status": "rejected", "error": {"type": "ApprovalMismatch"}}

        import fnmatch
        from pathlib import Path as RemotePath

        import requests  # type: ignore[import-untyped]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        from huggingface_hub.utils import configure_http_backend  # type: ignore[import-not-found]
        from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]

        def no_retry_session() -> Any:
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=0, pool_connections=1, pool_maxsize=1)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            return session

        configure_http_backend(backend_factory=no_retry_session)
        cache_dir = RemotePath(_VOLUME_MOUNT) / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_download(
            repo_id=_MODEL_ID,
            revision=_REVISION,
            token=False,
            cache_dir=str(cache_dir),
            allow_patterns=list(_ARTIFACT_FILES),
            max_workers=1,
            etag_timeout=30,
        )
        root, files = RemotePath(snapshot), []
        for path in sorted(root.iterdir()):
            if not path.is_file() or not any(
                fnmatch.fnmatchcase(path.name, p) for p in _ARTIFACT_FILES
            ):
                continue
            digest = remote_hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
            )
        if not files:
            raise FileNotFoundError("allowlisted public snapshot returned no files")
        runtime["_REMOTE_VOLUME"].commit()
        return {
            "status": "success",
            "model_id": _MODEL_ID,
            "revision": _REVISION,
            "snapshot_path": str(root.relative_to(_VOLUME_MOUNT)),
            "files": files,
            "file_count": len(files),
            "bytes": sum(file["bytes"] for file in files),
            "source_sha256": source_hash,
            "payload_sha256": payload_hash,
        }
    except Exception as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None) or getattr(error, "status_code", None)
        err_no = getattr(error, "errno", None)
        return {
            "status": "failed",
            "model_id": _MODEL_ID,
            "revision": _REVISION,
            "source_sha256": source_hash,
            "payload_sha256": payload_hash,
            "error": {
                "type": type(error).__name__,
                "errno": err_no if type(err_no) is int else None,
                "http_status": status if type(status) is int else None,
            },
        }


def build_modal_function(source_hash: str, payload_hash: str) -> tuple[Any, Any]:
    """Build the one-CPU source-free function without starting an app."""
    import modal

    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    volume = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    remote = FunctionType(
        _remote_probe.__code__,
        {
            "__name__": "__main__",
            "_MODEL_ID": _MODEL_ID,
            "_REVISION": _REVISION,
            "_ARTIFACT_FILES": _ARTIFACT_FILES,
            "_VOLUME_MOUNT": _VOLUME_MOUNT,
            "_APPROVED_SOURCE_SHA256": source_hash,
            "_APPROVED_PAYLOAD_SHA256": payload_hash,
            "_REMOTE_VOLUME": volume,
        },
        name="public_model_cache_probe",
    )
    options: Any = {
        "cpu": 1,
        "memory": 8192,
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 600,
        "timeout": 600,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
        "volumes": {_VOLUME_MOUNT: volume},
    }
    app = modal.App("tracemem-public-model-cache")
    return app, app.function(image=image, **options)(remote)


def _reserve_receipt(output_path: Path | None) -> int:
    local_dir = _OUTPUT_DIR.parent
    if not local_dir.exists():
        local_dir.mkdir(mode=0o700)
    local_info = local_dir.lstat()
    if not stat.S_ISDIR(local_info.st_mode) or stat.S_ISLNK(local_info.st_mode):
        raise ValueError("repository .local must be a real directory")
    _OUTPUT_DIR.mkdir(mode=0o700, exist_ok=True)
    info = _OUTPUT_DIR.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("public cache receipt directory must be a non-symlink 0700 directory")
    path = output_path or (_OUTPUT_DIR / f"receipt-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json")
    if (
        path.parent.resolve() != _OUTPUT_DIR.resolve()
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", path.name) is None
    ):
        raise ValueError("receipt must be a simple filename under .local/public-model-cache")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.fchmod(fd, 0o600)
    return fd


def run_once(
    approved_source_sha256: str, approved_payload_sha256: str, output_path: Path | None = None
) -> dict[str, Any]:
    source, payload = source_sha256(), build_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if not _SHA.fullmatch(approved_source_sha256) or approved_source_sha256 != source:
        raise ValueError("exact current source SHA-256 approval is required")
    if not _SHA.fullmatch(approved_payload_sha256) or approved_payload_sha256 != digest:
        raise ValueError("exact current payload SHA-256 approval is required")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Modal cache launch requires local Python 3.12")
    fd = _reserve_receipt(output_path)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "model_id": _MODEL_ID,
        "revision": _REVISION,
        "source_sha256": source,
        "payload_sha256": digest,
        "remote_started": False,
    }
    try:
        from experiments.sft_launch import verify_modal_inventory, verify_personal_workspace

        verify_personal_workspace()
        inventory = verify_modal_inventory()
        app, function = build_modal_function(source, digest)
        with app.run(detach=False):
            receipt["remote_started"] = True
            result = function.remote(payload, digest, source)
        if type(result) is not dict:
            raise TypeError("remote cache returned a non-object result")
        receipt.update(
            {
                "modal_app_id": getattr(app, "app_id", None),
                "launch_checks": {
                    "personal_workspace_verified": True,
                    "modal_inventory_stopped": True,
                    "modal_app_count": len(inventory),
                },
                "result": result,
            }
        )
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__}
    data, offset = _canonical(receipt) + b"\n", 0
    try:
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch", action="store_true", help="explicitly launch the public-only cache job"
    )
    parser.add_argument("--approved-source-sha256")
    parser.add_argument("--approved-payload-sha256")
    parser.add_argument("--receipt-name", help="new filename inside .local/public-model-cache")
    args = parser.parse_args(argv)
    payload = build_payload()
    source, digest = source_sha256(), hashlib.sha256(_canonical(payload)).hexdigest()
    if not args.launch:
        print(
            json.dumps(
                {
                    "remote_started": False,
                    "source_sha256": source,
                    "payload_sha256": digest,
                    "payload": payload,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if not args.approved_source_sha256 or not args.approved_payload_sha256:
        parser.error("--launch requires both exact source and payload SHA-256 approvals")
    output = _OUTPUT_DIR / args.receipt_name if args.receipt_name else None
    receipt = run_once(args.approved_source_sha256, args.approved_payload_sha256, output)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0 if receipt.get("result", {}).get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
