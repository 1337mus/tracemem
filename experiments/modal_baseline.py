"""Prepare or run a bounded, synthetic-only Modal inference baseline.

Preparation is the default. A remote call requires a printed payload hash,
matching explicit approval, and a fresh personal-workspace token check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tracemem.benchmark import load_synthetic_data
from tracemem.remote_vectors import text_sha256

_ROOT = Path(__file__).resolve().parents[1]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_EXTRACTION_MODEL_ID = "Qwen/Qwen3-4B"
_MAX_EXTRACTION_CASES = 24
_MAX_GENERATED_TOKENS = 512
_MAX_PROMPT_TOKENS = 4096
_MAX_TEXT_BYTES = 20_000
_SECRET_FIELD = re.compile(
    r"(?i)\b((?:authorization|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"token|secret|password)\s*[:=]\s*)"
    r"(?:bearer\s+)?[^\s,;]+"
)
_BEARER_SECRET = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/-]+=*")
_MODAL_TOKEN = re.compile(r"\b(?:ak|as)-[A-Za-z0-9_-]{6,}\b")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/:@\s]+:[^/@\s]+@")
_IMAGE_PACKAGES = (
    "torch==2.7.1",
    "transformers==4.51.3",
    "sentence-transformers==4.1.0",
    "accelerate==1.6.0",
)


def _read_config(name: str) -> dict[str, Any]:
    path = _ROOT / "configs" / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("runner configuration has an unsupported shape")
    return value


def _revision(value: object) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ValueError("model revision must be a pinned 40-character commit")
    return value


def _embedding_model() -> dict[str, str | int]:
    config = _read_config("embedding-model.json")
    if config.get("model_id") != _EMBEDDING_MODEL_ID:
        raise ValueError("the configured embedding model is not allowlisted")
    if config.get("license_tag") != "apache-2.0":
        raise ValueError("the embedding model license is not verified")
    if config.get("weights_downloaded_locally") is not False:
        raise ValueError("local embedding weights are prohibited")
    return {
        "model_id": _EMBEDDING_MODEL_ID,
        "revision": _revision(config.get("revision")),
        "dimensions": 384,
    }


def _extraction_model() -> dict[str, str | int]:
    config = _read_config("model-candidates.json")
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate configuration has an unsupported shape")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("id") == _EXTRACTION_MODEL_ID
    ]
    if len(matches) != 1:
        raise ValueError("the extraction model must have exactly one pinned candidate")
    candidate = matches[0]
    if candidate.get("license") != "apache-2.0" or candidate.get("gated") is not False:
        raise ValueError("the extraction model is not an approved public candidate")
    return {
        "model_id": _EXTRACTION_MODEL_ID,
        "revision": _revision(candidate.get("revision")),
        "max_new_tokens": _MAX_GENERATED_TOKENS,
        "max_prompt_tokens": _MAX_PROMPT_TOKENS,
    }


def _payload(
    mode: str, input_hash: str, model: dict[str, str | int], items: list[dict[str, str]]
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "mode": mode,
        "input_sha256": input_hash,
        "model": model,
        "items": items,
    }
    _validate_payload(result)
    return result


def build_payload(
    mode: str,
    *,
    max_cases: int = _MAX_EXTRACTION_CASES,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    """Build only fixture-derived inputs; no user-supplied source paths exist."""
    if mode not in {"embeddings", "extraction"}:
        raise ValueError("mode must be embeddings or extraction")
    if type(prompt_version) is not str or prompt_version not in {"v1", "explicit-v2"}:
        raise ValueError("prompt version must be v1 or explicit-v2")
    if mode == "embeddings" and prompt_version != "v1":
        raise ValueError("explicit-v2 is available only for extraction")
    if isinstance(max_cases, bool) or not isinstance(max_cases, int):
        raise ValueError("max_cases must be an integer")
    if not 1 <= max_cases <= _MAX_EXTRACTION_CASES:
        raise ValueError("extraction supports between 1 and 24 cases")

    if mode == "embeddings":
        records, queries, fixture_hash = load_synthetic_data()
        text_by_hash: dict[str, str] = {}
        for text in [*(record.text for record in records), *(query["query"] for query in queries)]:
            if not isinstance(text, str) or not text:
                raise ValueError("synthetic fixture contains invalid text")
            if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise ValueError("synthetic embedding text exceeds the payload limit")
            digest = text_sha256(text)
            if digest in text_by_hash and text_by_hash[digest] != text:
                raise ValueError("synthetic text hash collision")
            text_by_hash[digest] = text
        items = [
            {"text_sha256": digest, "text": text_by_hash[digest]} for digest in sorted(text_by_hash)
        ]
        return _payload("embeddings", fixture_hash, _embedding_model(), items)

    from tracemem.challenge import load_challenge

    if prompt_version == "v1":
        from tracemem.challenge import render_prompt as render
    else:
        from tracemem.prompts import render_explicit_prompt as render

    cases, fixture_hash = load_challenge()
    if not isinstance(cases, (list, tuple)):
        raise ValueError("challenge fixture has an unsupported shape")
    selected = cases[:max_cases]
    items: list[dict[str, str]] = []
    case_ids: set[str] = set()
    for case in selected:
        if not isinstance(case, dict):
            raise ValueError("challenge fixture contains an unsupported case")
        case_id = case.get("case_id")
        messages = case.get("messages")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("challenge case identifiers must be unique non-empty strings")
        if not isinstance(messages, list):
            raise ValueError("challenge messages have an unsupported shape")
        case_ids.add(case_id)
        safe_messages: list[dict[str, str | None]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("challenge contains an unsupported message")
            role = message.get("role")
            text = message.get("text")
            session_id = message.get("session_id")
            message_id = message.get("message_id")
            recorded_at = message.get("recorded_at")
            content_kind = message.get("content_kind")
            if role not in {"user", "assistant", "tool", "system", "developer", "unknown"}:
                raise ValueError("challenge message role is invalid")
            if not all(isinstance(value, str) and value for value in (session_id, message_id)):
                raise ValueError("challenge message identifiers are invalid")
            if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise ValueError("challenge message text is invalid or too large")
            if recorded_at is not None and not isinstance(recorded_at, str):
                raise ValueError("challenge timestamp is invalid")
            if not isinstance(content_kind, str) or not content_kind:
                raise ValueError("challenge content kind is invalid")
            safe_messages.append(
                {
                    "session_id": session_id,
                    "message_id": message_id,
                    "role": role,
                    "text": text,
                    "recorded_at": recorded_at,
                    "content_kind": content_kind,
                }
            )
        prompt_case: dict[str, Any] = {"case_id": case_id, "messages": safe_messages}
        prompt = render(prompt_case)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("challenge prompt renderer returned invalid text")
        if len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("challenge prompt exceeds the payload limit")
        items.append({"case_id": case_id, "prompt": prompt})
    if not items:
        raise ValueError("challenge fixture contains no cases")
    return _payload("extraction", fixture_hash, _extraction_model(), items)


def _validate_payload(payload: object) -> None:
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "mode",
        "input_sha256",
        "model",
        "items",
    }:
        raise ValueError("payload keys do not match the allowlist")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("payload schema version is unsupported")
    if (
        not isinstance(payload["input_sha256"], str)
        or _SHA256.fullmatch(payload["input_sha256"]) is None
    ):
        raise ValueError("payload input hash is invalid")
    mode = payload["mode"]
    model = payload["model"]
    items = payload["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("payload items must be a non-empty list")
    if mode == "embeddings":
        if model != _embedding_model() or len(items) > 128:
            raise ValueError("embedding model or item count is not allowlisted")
        seen: set[str] = set()
        for item in items:
            if type(item) is not dict or set(item) != {"text_sha256", "text"}:
                raise ValueError("embedding item keys do not match the allowlist")
            digest, text = item["text_sha256"], item["text"]
            if not isinstance(text, str) or not text or len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise ValueError("embedding text is invalid or too large")
            if not isinstance(digest, str) or digest != text_sha256(text):
                raise ValueError("embedding text hash does not match its text")
            if digest in seen:
                raise ValueError("embedding payload contains duplicate text hashes")
            seen.add(digest)
    elif mode == "extraction":
        if model != _extraction_model() or len(items) > _MAX_EXTRACTION_CASES:
            raise ValueError("extraction model or case count is not allowlisted")
        seen_cases: set[str] = set()
        for item in items:
            if type(item) is not dict or set(item) != {"case_id", "prompt"}:
                raise ValueError("extraction item keys do not match the allowlist")
            case_id, prompt = item["case_id"], item["prompt"]
            if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_cases:
                raise ValueError("extraction case identifiers must be unique")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or len(prompt.encode("utf-8")) > 100_000
            ):
                raise ValueError("extraction prompt is invalid or too large")
            seen_cases.add(case_id)
    else:
        raise ValueError("payload mode is not allowlisted")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a canonical JSON payload so the reviewed request can be pinned."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def modal_function_options() -> dict[str, Any]:
    """Return the one-container, source-free resource envelope for SDK 1.6.1."""
    return {
        "gpu": "T4",
        "cpu": (2, 2),
        "memory": (16_384, 16_384),
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 300,
        "timeout": 900,
        "scaledown_window": 2,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def verify_workspace(expected: str, actual: str) -> None:
    if not isinstance(expected, str) or not expected:
        raise RuntimeError("saved Modal workspace is unavailable")
    if not isinstance(actual, str) or actual != expected:
        raise RuntimeError("active Modal workspace does not match the saved workspace")


def _expected_workspace() -> str:
    status_path = _ROOT / ".local" / "modal-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = status.get("expected_workspace") if isinstance(status, dict) else None
    if not isinstance(expected, str) or not expected:
        raise RuntimeError("saved Modal workspace is unavailable")
    return expected


def _active_workspace() -> str:
    cli = Path(sys.prefix) / "bin" / "modal"
    if not cli.is_file():
        raise RuntimeError("Modal CLI is unavailable for a fresh workspace check")
    try:
        result = subprocess.run(
            [str(cli), "token", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("fresh Modal workspace check failed") from None
    if result.returncode != 0:
        raise RuntimeError("fresh Modal workspace check failed")
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    workspaces = re.findall(r"^Workspace:\s*([^\s(]+)", output, flags=re.MULTILINE)
    if len(workspaces) != 1:
        raise RuntimeError("fresh Modal workspace check returned no unique workspace")
    return workspaces[0]


def _safe_exception_message(error: Exception) -> str:
    """Keep useful runtime details while removing common credential forms."""
    try:
        message = str(error)
    except Exception:
        return "exception message unavailable"
    message = re.sub(r"[\x00-\x1f\x7f]+", " ", message)
    message = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", message)
    message = _BEARER_SECRET.sub(r"\1[REDACTED]", message)
    message = _SECRET_FIELD.sub(r"\1[REDACTED]", message)
    message = _MODAL_TOKEN.sub("[REDACTED]", message)
    message = re.sub(r"\s+", " ", message).strip()
    return message[:400] or "no exception message"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-baseline")
    parser.add_argument("--mode", required=True, choices=("embeddings", "extraction"))
    parser.add_argument("--max-cases", type=int, default=_MAX_EXTRACTION_CASES)
    parser.add_argument("--prompt-version", choices=("v1", "explicit-v2"), default="v1")
    parser.add_argument("--launch", action="store_true", help="run one reviewed remote call")
    parser.add_argument("--approve-payload-sha256")
    return parser


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Execute one validated request; all ML imports occur in the remote container."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import math
    import platform
    import re as remote_re
    import time as remote_time

    import modal as modal_sdk

    started = remote_time.perf_counter()
    if type(request) is not dict or set(request) != {"payload", "payload_sha256"}:
        raise ValueError("remote request keys are invalid")
    payload = request["payload"]
    approved_hash = request["payload_sha256"]
    if (
        not isinstance(approved_hash, str)
        or remote_re.fullmatch(r"[0-9a-f]{64}", approved_hash) is None
    ):
        raise ValueError("remote payload hash is invalid")
    payload_bytes = remote_json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if remote_hashlib.sha256(payload_bytes).hexdigest() != approved_hash:
        raise ValueError("remote payload hash does not match approved hash")
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "mode",
        "input_sha256",
        "model",
        "items",
    }:
        raise ValueError("remote payload keys are invalid")
    mode = payload["mode"]
    model_info = payload["model"]
    items = payload["items"]
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("remote payload version is invalid")
    if (
        not isinstance(payload["input_sha256"], str)
        or remote_re.fullmatch(r"[0-9a-f]{64}", payload["input_sha256"]) is None
    ):
        raise ValueError("remote input hash is invalid")
    if not isinstance(items, list) or not items:
        raise ValueError("remote payload items are invalid")

    expected_models = {
        "embeddings": (
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ),
        "extraction": (
            "Qwen/Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
        ),
    }
    if mode not in expected_models or not isinstance(model_info, dict):
        raise ValueError("remote model mode is invalid")
    expected_id, expected_revision = expected_models[mode]
    if model_info.get("model_id") != expected_id or model_info.get("revision") != expected_revision:
        raise ValueError("remote model is not an allowlisted revision")

    if mode == "embeddings":
        if (
            set(model_info) != {"model_id", "revision", "dimensions"}
            or model_info["dimensions"] != 384
        ):
            raise ValueError("remote embedding model settings are invalid")
        if len(items) > 128:
            raise ValueError("remote embedding item count exceeds the limit")
        seen_hashes: set[str] = set()
        for item in items:
            if type(item) is not dict or set(item) != {"text_sha256", "text"}:
                raise ValueError("remote embedding item keys are invalid")
            text, digest = item["text"], item["text_sha256"]
            if not isinstance(text, str) or not text or len(text.encode("utf-8")) > 20_000:
                raise ValueError("remote embedding text is invalid")
            if (
                not isinstance(digest, str)
                or remote_hashlib.sha256(text.encode("utf-8")).hexdigest() != digest
            ):
                raise ValueError("remote embedding text hash is invalid")
            if digest in seen_hashes:
                raise ValueError("remote embedding contains duplicate hashes")
            seen_hashes.add(digest)
    else:
        if set(model_info) != {
            "model_id",
            "revision",
            "max_new_tokens",
            "max_prompt_tokens",
        }:
            raise ValueError("remote extraction model settings are invalid")
        if model_info["max_new_tokens"] != 512 or model_info["max_prompt_tokens"] != 4096:
            raise ValueError("remote extraction token limits are invalid")
        if len(items) > 24:
            raise ValueError("remote extraction case count exceeds the limit")
        seen_cases: set[str] = set()
        for item in items:
            if type(item) is not dict or set(item) != {"case_id", "prompt"}:
                raise ValueError("remote extraction item keys are invalid")
            case_id, prompt = item["case_id"], item["prompt"]
            if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
                raise ValueError("remote extraction case IDs are invalid")
            if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 100_000:
                raise ValueError("remote extraction prompt is invalid")
            seen_cases.add(case_id)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the Modal T4 GPU is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    model_started = remote_time.perf_counter()
    gpu_seconds = 0.0
    results: dict[str, Any]

    if mode == "embeddings":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            expected_id,
            revision=expected_revision,
            trust_remote_code=False,
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16, "use_safetensors": True},
        )
        model_load_seconds = remote_time.perf_counter() - model_started
        texts = [item["text"] for item in items]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        vectors = encoder.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        end_event.record()
        torch.cuda.synchronize()
        gpu_seconds = start_event.elapsed_time(end_event) / 1000
        encoded_vectors: dict[str, list[float]] = {}
        for item, vector in zip(items, vectors, strict=True):
            values = [float(value) for value in vector.tolist()]
            if len(values) != 384 or not all(math.isfinite(value) for value in values):
                raise ValueError("remote embedding returned an invalid vector")
            encoded_vectors[item["text_sha256"]] = values
        results = {
            "vectors": encoded_vectors,
            "vector_count": len(encoded_vectors),
        }
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            expected_id,
            revision=expected_revision,
            trust_remote_code=False,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            expected_id,
            revision=expected_revision,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model.eval()
        model_load_seconds = remote_time.perf_counter() - model_started
        outputs: list[dict[str, Any]] = []
        for item in items:
            messages = [{"role": "user", "content": item["prompt"]}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            input_tokens = int(input_ids.shape[-1])
            if input_tokens > model_info["max_prompt_tokens"]:
                raise ValueError("challenge prompt exceeds the pinned token limit")
            input_ids = input_ids.to(model.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=model_info["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            end_event.record()
            torch.cuda.synchronize()
            gpu_seconds += start_event.elapsed_time(end_event) / 1000
            generated_ids = generated[0, input_tokens:]
            generated_tokens = int(generated_ids.shape[-1])
            outputs.append(
                {
                    "case_id": item["case_id"],
                    "raw_output": tokenizer.decode(generated_ids, skip_special_tokens=True),
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                    "hit_max_new_tokens": generated_tokens >= model_info["max_new_tokens"],
                }
            )
        results = {"generated_outputs": outputs, "case_count": len(outputs)}

    package_versions = dict(
        sorted(
            (
                (name, distribution.version)
                for distribution in metadata.distributions()
                if isinstance(name := distribution.metadata.get("Name"), str) and name
            ),
            key=lambda item: item[0].casefold(),
        )
    )

    torch.cuda.synchronize()
    return {
        "schema_version": 1,
        "mode": mode,
        "input_sha256": payload["input_sha256"],
        "payload_sha256": approved_hash,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
        "model": model_info,
        "package_versions": package_versions,
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_compute_seconds": round(gpu_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "remote_function_elapsed_seconds": round(remote_time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        **results,
    }


def _build_modal_function() -> tuple[Any, Any]:
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-synthetic-baseline")
    function = app.function(image=image, **modal_function_options())(_remote_compute)
    return app, function


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _runner_sha256() -> str:
    """Hash the exact local runner source before its optional remote call."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runner_sha256 = _runner_sha256()
        payload = build_payload(
            args.mode, max_cases=args.max_cases, prompt_version=args.prompt_version
        )
        approved_hash = payload_sha256(payload)
        if not args.launch:
            if args.approve_payload_sha256 is not None:
                raise ValueError("payload approval is only accepted together with --launch")
            _write_json(
                {
                    "mode": payload["mode"],
                    "prompt_version": args.prompt_version,
                    "runner_sha256": runner_sha256,
                    "model": payload["model"],
                    "item_count": len(payload["items"]),
                    "input_sha256": payload["input_sha256"],
                    "payload_sha256": approved_hash,
                    "remote_started": False,
                }
            )
            return 0

        if args.approve_payload_sha256 != approved_hash:
            raise ValueError("--launch requires the exact prepared payload SHA-256")
        expected = _expected_workspace()
        verify_workspace(expected, _active_workspace())
        app, function = _build_modal_function()
        call_started = time.perf_counter()
        app_id: str | None = None
        with app.run(detach=False):
            app_id = app.app_id
            result = function.remote({"payload": payload, "payload_sha256": approved_hash})
        output = dict(result)
        output["modal_app_id"] = app_id
        output["prompt_version"] = args.prompt_version
        output["runner_sha256"] = runner_sha256
        output["client_call_wall_seconds_including_image_and_startup"] = round(
            time.perf_counter() - call_started, 6
        )
        _write_json(output)
        return 0
    except Exception as exc:
        print(
            f"modal-baseline failed ({type(exc).__name__}): {_safe_exception_message(exc)}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
