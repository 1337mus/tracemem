"""Build and run the fixed, source-free Modal tokenizer counter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from experiments.token_catalog import build_catalog

_EXPECTED_CATALOG_SHA256 = "85682edbcdd728103e6540c86106de953f400e217a9341ffbd82420d0429e478"
_EXPECTED_PAYLOAD_SHA256 = "125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88"
_TOKENIZER = {
    "model_id": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "transformers_version": "4.51.3",
    "template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "use_fast": True,
    "trust_remote_code": False,
    "add_generation_prompt": True,
    "enable_thinking": False,
    "truncation": False,
}
_IMAGE_PACKAGES = (
    "transformers==4.51.3",
    "tokenizers==0.21.1",
    "jinja2==3.1.6",
    "huggingface-hub==0.30.2",
    "safetensors==0.5.3",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_payload() -> dict[str, Any]:
    """Regenerate the exact reviewed prompt catalog and tokenizer settings."""
    catalog = build_catalog()
    catalog_sha256 = _sha256(_canonical_bytes(catalog))
    if catalog_sha256 != _EXPECTED_CATALOG_SHA256:
        raise ValueError("current prompt catalog differs from the reviewed catalog")
    items = catalog.get("items")
    if type(items) is not list or len(items) != 28:
        raise ValueError("reviewed tokenizer payload requires exactly 28 prompts")
    payload = {
        "schema_version": 1,
        "mode": "synthetic-token-count-v1",
        "catalog_sha256": catalog_sha256,
        "tokenizer": _TOKENIZER.copy(),
        "items": items,
    }
    if _sha256(_canonical_bytes(payload)) != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("regenerated payload differs from the reviewed payload")
    return payload


def _remote_count(request: dict[str, Any]) -> dict[str, Any]:
    """Source-free Modal body. All imports and constants are local to this function."""
    import hashlib
    import importlib.metadata
    import json
    import platform
    import re
    import time

    def canonical_bytes(value: object) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")

    expected_payload_sha256 = "125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88"
    expected_catalog_sha256 = "85682edbcdd728103e6540c86106de953f400e217a9341ffbd82420d0429e478"
    expected_template_sha256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    expected_tokenizer = {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "transformers_version": "4.51.3",
        "template_sha256": expected_template_sha256,
        "use_fast": True,
        "trust_remote_code": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "truncation": False,
    }
    if type(request) is not dict or set(request) != {
        "payload",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote request keys are invalid")
    payload = request["payload"]
    payload_sha256 = request["payload_sha256"]
    runner_sha256 = request["runner_sha256"]
    if type(payload_sha256) is not str or payload_sha256 != expected_payload_sha256:
        raise ValueError("remote payload hash is not the reviewed payload hash")
    if type(runner_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", runner_sha256) is None:
        raise ValueError("remote runner hash is invalid")
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != payload_sha256:
        raise ValueError("remote payload bytes do not match the reviewed hash")
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "mode",
        "catalog_sha256",
        "tokenizer",
        "items",
    }:
        raise ValueError("remote payload fields are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "synthetic-token-count-v1"
        or payload["catalog_sha256"] != expected_catalog_sha256
        or payload["tokenizer"] != expected_tokenizer
    ):
        raise ValueError("remote payload settings differ from the reviewed configuration")
    items = payload["items"]
    if type(items) is not list or len(items) != 28:
        raise ValueError("remote payload must contain exactly 28 prompt IDs")
    prompt_ids: list[str] = []
    for item in items:
        if type(item) is not dict or set(item) != {"prompt_id", "messages"}:
            raise ValueError("remote prompt fields are invalid")
        prompt_id, messages = item["prompt_id"], item["messages"]
        if (
            type(prompt_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", prompt_id) is None
            or type(messages) is not list
            or len(messages) != 2
            or any(
                type(message) is not dict
                or set(message) != {"role", "content"}
                or type(message["role"]) is not str
                or type(message["content"]) is not str
                for message in messages
            )
            or messages[0]["role"] != "system"
            or messages[1]["role"] != "user"
            or hashlib.sha256(canonical_bytes(messages)).hexdigest() != prompt_id
        ):
            raise ValueError("remote prompt ID or messages are invalid")
        prompt_ids.append(prompt_id)
    if prompt_ids != sorted(set(prompt_ids)):
        raise ValueError("remote prompt IDs must be unique and sorted")
    started = time.perf_counter()
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(
        expected_tokenizer["model_id"],
        revision=expected_tokenizer["revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    is_fast = tokenizer.is_fast
    if is_fast is not True:
        raise ValueError("pinned tokenizer is not a fast tokenizer")
    template = tokenizer.chat_template
    if type(template) is not str or not template:
        raise ValueError("pinned tokenizer has no chat template")
    template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if template_sha256 != expected_template_sha256:
        raise ValueError("pinned tokenizer chat template hash differs from the reviewed template")
    counts = []
    for item in items:
        token_ids = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if type(token_ids) is not list or any(
            type(token_id) is not int or token_id < 0 for token_id in token_ids
        ):
            raise ValueError("tokenizer returned invalid token IDs")
        counts.append({"prompt_id": item["prompt_id"], "prompt_tokens": len(token_ids)})

    package_versions = {
        package: importlib.metadata.version(package)
        for package in ("transformers", "tokenizers", "jinja2")
    }
    expected_versions = {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}
    if package_versions != expected_versions:
        raise ValueError("tokenizer package versions differ from the reviewed pins")
    backend_sha256 = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            backend_source = backend.to_str()
        except Exception:
            backend_source = None
        if type(backend_source) is str:
            backend_sha256 = hashlib.sha256(backend_source.encode("utf-8")).hexdigest()
    template_options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    counter_id = (
        f"Qwen/Qwen3-4B@{expected_tokenizer['revision']}+transformers-4.51.3+template-a55ee1b1"
    )
    return {
        "schema_version": 1,
        "count_receipt": {
            "schema_version": 1,
            "catalog_sha256": expected_catalog_sha256,
            "counter_id": counter_id,
            "counts": counts,
        },
        "tokenizer_provenance": {
            "model_id": expected_tokenizer["model_id"],
            "revision": expected_tokenizer["revision"],
            "template_sha256": template_sha256,
            "use_fast": is_fast,
            "trust_remote_code": False,
            "package_versions": package_versions,
            "python_version": platform.python_version(),
            "template_options": template_options,
            "backend_tokenizer_sha256": backend_sha256,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "payload_sha256": payload_sha256,
            "runner_sha256": runner_sha256,
        },
    }


def modal_function_options() -> dict[str, Any]:
    return {
        "cpu": (1, 2),
        "memory": (2048, 4096),
        "min_containers": 0,
        "max_containers": 1,
        "retries": 0,
        "startup_timeout": 300,
        "timeout": 600,
        "scaledown_window": 2,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def _build_modal_function() -> tuple[Any, Any]:
    """Construct the source-free function only after local launch approvals pass."""
    import types

    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-token-count")
    remote_count = types.FunctionType(_remote_count.__code__, {"__name__": "__main__"})
    remote_count.__doc__ = _remote_count.__doc__
    function = app.function(image=image, **modal_function_options())(remote_count)
    return app, function


def verify_personal_workspace() -> None:
    """Use the existing personal workspace check only after launch approval."""
    from experiments.scenario_payload import verify_personal_workspace as verify

    verify()


def _require_supported_launch_python() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor != 12:
        raise RuntimeError(
            "approved Modal launches require local Python 3.12 "
            f"(found {version.major}.{version.minor})"
        )


def _validate_remote_result(
    result: Any, payload: dict[str, Any], runner_sha256: str
) -> dict[str, Any]:
    if type(result) is not dict or set(result) != {
        "schema_version",
        "count_receipt",
        "tokenizer_provenance",
    }:
        raise ValueError("remote function returned an invalid envelope")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("remote result schema_version must be integer 1")
    receipt = result["count_receipt"]
    if type(receipt) is not dict or set(receipt) != {
        "schema_version",
        "catalog_sha256",
        "counter_id",
        "counts",
    }:
        raise ValueError("remote count receipt shape is invalid")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["catalog_sha256"] != _EXPECTED_CATALOG_SHA256
        or receipt["counter_id"]
        != "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
        "+transformers-4.51.3+template-a55ee1b1"
        or type(receipt["counts"]) is not list
    ):
        raise ValueError("remote count receipt provenance is invalid")
    expected_ids = {item["prompt_id"] for item in payload["items"]}
    observed: dict[str, int] = {}
    for count in receipt["counts"]:
        if type(count) is not dict or set(count) != {"prompt_id", "prompt_tokens"}:
            raise ValueError("remote count row fields are invalid")
        prompt_id, prompt_tokens = count["prompt_id"], count["prompt_tokens"]
        if (
            type(prompt_id) is not str
            or prompt_id not in expected_ids
            or type(prompt_tokens) is not int
            or prompt_tokens < 0
            or prompt_id in observed
        ):
            raise ValueError("remote count row has an unknown or invalid prompt count")
        observed[prompt_id] = prompt_tokens
    if observed.keys() != expected_ids:
        raise ValueError("remote count receipt does not cover every reviewed prompt")

    provenance = result["tokenizer_provenance"]
    if type(provenance) is not dict or set(provenance) != {
        "model_id",
        "revision",
        "template_sha256",
        "use_fast",
        "trust_remote_code",
        "package_versions",
        "python_version",
        "template_options",
        "backend_tokenizer_sha256",
        "elapsed_seconds",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote tokenizer provenance shape is invalid")
    elapsed = provenance["elapsed_seconds"]
    backend_sha256 = provenance["backend_tokenizer_sha256"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("remote elapsed_seconds must be finite and nonnegative")
    if backend_sha256 is not None and (
        type(backend_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", backend_sha256) is None
    ):
        raise ValueError("remote backend tokenizer hash is invalid")
    if (
        provenance["model_id"] != _TOKENIZER["model_id"]
        or provenance["revision"] != _TOKENIZER["revision"]
        or provenance["template_sha256"] != _TOKENIZER["template_sha256"]
        or provenance["use_fast"] is not True
        or provenance["trust_remote_code"] is not False
        or provenance["package_versions"]
        != {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}
        or type(provenance["python_version"]) is not str
        or not provenance["python_version"].startswith("3.12.")
        or provenance["template_options"]
        != {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        or provenance["payload_sha256"] != _EXPECTED_PAYLOAD_SHA256
        or provenance["runner_sha256"] != runner_sha256
    ):
        raise ValueError("remote tokenizer provenance differs from the approved configuration")
    return result


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run one approved count call and preserve one exclusive local receipt."""
    _require_supported_launch_python()
    expected_payload = build_payload()
    if type(payload) is not dict or _canonical_bytes(payload) != _canonical_bytes(expected_payload):
        raise ValueError("caller payload differs from the fixed reviewed payload")
    if approved_payload_sha256 != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("exact reviewed payload approval is required")
    runner_sha256 = _sha256(Path(__file__).read_bytes())
    if approved_runner_sha256 != runner_sha256:
        raise ValueError("exact current runner approval is required")

    started = time.perf_counter()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except FileExistsError:
        raise FileExistsError("receipt output path must be new") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        os.fchmod(output.fileno(), 0o600)
        try:
            verify_personal_workspace()
            app, function = _build_modal_function()
            with app.run(detach=False):
                remote_result = function.remote(
                    {
                        "payload": expected_payload,
                        "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
                        "runner_sha256": runner_sha256,
                    }
                )
            result = _validate_remote_result(remote_result, expected_payload, runner_sha256)
        except Exception as error:
            from experiments.scenario_payload import safe_exception_message

            failure = {
                "schema_version": 1,
                "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
                "runner_sha256": runner_sha256,
                "client_elapsed_seconds": round(time.perf_counter() - started, 6),
                "error": {
                    "type": type(error).__name__,
                    "message": safe_exception_message(error),
                },
            }
            output.write(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            raise
        receipt = {
            "schema_version": 1,
            "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
            "runner_sha256": runner_sha256,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "result": result,
        }
        output.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-token-count")
    parser.add_argument("--launch", action="store_true", help="run one explicitly approved call")
    parser.add_argument("--show-payload", action="store_true")
    parser.add_argument("--approve-payload-sha256")
    parser.add_argument("--approve-runner-sha256")
    parser.add_argument("--output", type=Path, help="new path for the local raw receipt")
    return parser


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload()
        payload_sha256 = _sha256(_canonical_bytes(payload))
        runner_sha256 = _sha256(Path(__file__).read_bytes())
        if not args.launch:
            if args.approve_payload_sha256 or args.approve_runner_sha256 or args.output:
                raise ValueError("approval flags and --output require --launch")
            _write_json(
                payload
                if args.show_payload
                else {
                    "item_count": len(payload["items"]),
                    "payload_sha256": payload_sha256,
                    "remote_started": False,
                    "runner_sha256": runner_sha256,
                }
            )
            return 0
        if args.show_payload:
            raise ValueError("--show-payload is only available during offline preparation")
        if (
            args.approve_payload_sha256 is None
            or args.approve_runner_sha256 is None
            or args.output is None
        ):
            raise ValueError("--launch requires both exact approvals and a new --output path")
        run_once(
            payload,
            approved_payload_sha256=args.approve_payload_sha256,
            approved_runner_sha256=args.approve_runner_sha256,
            output_path=args.output,
        )
        return 0
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        print(f"{type(error).__name__}: {safe_exception_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
