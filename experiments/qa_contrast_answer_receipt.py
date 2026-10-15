"""Offline contract for the pinned source-only QA contrast answering receipt."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from experiments import prepare_qa_contrast_requests

_EXPECTED_PAYLOAD_SHA256 = "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e"
_EXPECTED_PREPARER_SHA256 = "59270049e744629bdbbda16670160120c3135c6028c933af2a84fa8979462fe7"
_MODEL = {
    "model_id": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "precision": "float16",
    "attn_implementation": "eager",
    "trust_remote_code": False,
    "use_safetensors": True,
    "use_fast": True,
    "enable_thinking": False,
    "do_sample": False,
    "num_beams": 1,
    "seed": 0,
    "max_new_tokens": 128,
    "max_prompt_tokens": 1024,
}
_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
_BACKEND_SHA256 = "a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    return _sha256(_canonical_bytes(payload))


def build_payload() -> dict[str, Any]:
    """Return the exact reviewed request payload and never its local references or audit."""
    payload = prepare_qa_contrast_requests.prepare_requests()[0]
    source_sha256 = _sha256(Path(prepare_qa_contrast_requests.__file__).read_bytes())
    if source_sha256 != _EXPECTED_PREPARER_SHA256:
        raise ValueError("QA contrast request preparer source differs from the reviewed source")
    if (
        type(payload) is not dict
        or payload.get("provenance", {}).get("preparer_source_sha256") != source_sha256
        or payload_sha256(payload) != _EXPECTED_PAYLOAD_SHA256
    ):
        raise ValueError("QA contrast answering payload differs from the reviewed payload")
    return payload


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate the pinned source-only payload before any remote dependency is loaded."""
    if type(payload) is not dict or payload_sha256(payload) != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("payload differs from the reviewed payload hash")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("payload schema version is invalid")
    if payload.get("mode") != "qa-contrast-dev-source-only-requests-v1":
        raise ValueError("payload mode is invalid")
    if payload.get("model") != _MODEL:
        raise ValueError("payload model settings differ from the reviewed configuration")
    tokenizer = payload.get("tokenizer")
    if (
        type(tokenizer) is not dict
        or tokenizer.get("settings", {}).get("template_sha256") != _TEMPLATE_SHA256
    ):
        raise ValueError("payload tokenizer settings are invalid")
    items = payload.get("items")
    if type(items) is not list or len(items) != 46:
        raise ValueError("payload must contain exactly 46 prompts")
    prompt_ids: list[str] = []
    total_tokens = 0
    for item in items:
        if type(item) is not dict or set(item) != {"prompt_id", "messages", "prompt_tokens"}:
            raise ValueError("payload prompt fields are invalid")
        prompt_id, messages, count = item["prompt_id"], item["messages"], item["prompt_tokens"]
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
            or [message["role"] for message in messages] != ["system", "user"]
            or _sha256(_canonical_bytes(messages)) != prompt_id
            or type(count) is not int
            or not 1 <= count <= 1024
        ):
            raise ValueError("payload prompt identity, messages, or token count are invalid")
        prompt_ids.append(prompt_id)
        total_tokens += count
    if prompt_ids != sorted(set(prompt_ids)) or total_tokens != 23533:
        raise ValueError("payload prompt IDs or measured input token total are invalid")
    return payload


def validate_result(result: Any, payload: dict[str, Any], runner_sha256: str) -> dict[str, Any]:
    """Validate a remote answer receipt without importing Modal or model packages."""
    validate_payload(payload)
    if type(result) is not dict or set(result) != {
        "schema_version",
        "payload_sha256",
        "runner_sha256",
        "model",
        "tokenizer",
        "runtime",
        "outputs",
    }:
        raise ValueError("remote result envelope is invalid")
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or result["payload_sha256"] != _EXPECTED_PAYLOAD_SHA256
        or result["runner_sha256"] != runner_sha256
    ):
        raise ValueError("remote result approval provenance is invalid")
    model = result["model"]
    if type(model) is not dict or set(model) != {
        "model_id",
        "revision",
        "generation_protocol_sha256",
    }:
        raise ValueError("remote model provenance fields are invalid")
    expected_protocol_hash = _sha256(_canonical_bytes(payload["model"]))
    if (
        model["model_id"] != _MODEL["model_id"]
        or model["revision"] != _MODEL["revision"]
        or model["generation_protocol_sha256"] != expected_protocol_hash
    ):
        raise ValueError("remote model identity or generation protocol hash is invalid")
    tokenizer = result["tokenizer"]
    tokenizer_fields = {
        "model_id",
        "revision",
        "template_sha256",
        "backend_tokenizer_sha256",
        "use_fast",
        "trust_remote_code",
        "package_versions",
        "python_version",
        "template_options",
        "eos_token_id",
    }
    if type(tokenizer) is not dict or set(tokenizer) != tokenizer_fields:
        raise ValueError("remote tokenizer provenance fields are invalid")
    if (
        tokenizer["model_id"] != _MODEL["model_id"]
        or tokenizer["revision"] != _MODEL["revision"]
        or tokenizer["template_sha256"] != _TEMPLATE_SHA256
        or tokenizer["backend_tokenizer_sha256"] != _BACKEND_SHA256
        or tokenizer["use_fast"] is not True
        or tokenizer["trust_remote_code"] is not False
        or tokenizer["package_versions"]
        != {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}
        or type(tokenizer["python_version"]) is not str
        or not tokenizer["python_version"].startswith("3.12.")
        or tokenizer["template_options"]
        != {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "truncation": False,
        }
        or type(tokenizer["eos_token_id"]) is not int
        or tokenizer["eos_token_id"] < 0
    ):
        raise ValueError("remote tokenizer provenance differs from the reviewed pins")
    runtime = result["runtime"]
    runtime_fields = {
        "python_version",
        "torch_version",
        "accelerate_version",
        "cuda_version",
        "gpu_name",
        "gpu_capability",
        "model_load_seconds",
        "gpu_compute_seconds",
        "elapsed_seconds",
        "gpu_peak_memory_bytes",
        "request_count",
        "modal_call_id",
    }
    if type(runtime) is not dict or set(runtime) != runtime_fields:
        raise ValueError("remote runtime provenance fields are invalid")
    if (
        type(runtime["python_version"]) is not str
        or not runtime["python_version"].startswith("3.12.")
        or runtime["torch_version"] != "2.7.1"
        or runtime["accelerate_version"] != "1.6.0"
        or type(runtime["cuda_version"]) is not str
        or not runtime["cuda_version"]
        or type(runtime["gpu_name"]) is not str
        or runtime["gpu_name"] != "Tesla T4"
        or type(runtime["gpu_capability"]) is not str
        or runtime["gpu_capability"] != "7.5"
        or type(runtime["gpu_peak_memory_bytes"]) is not int
        or runtime["gpu_peak_memory_bytes"] < 0
        or type(runtime["request_count"]) is not int
        or runtime["request_count"] != 46
        or type(runtime["modal_call_id"]) is not str
        or not runtime["modal_call_id"]
        or len(runtime["modal_call_id"]) > 256
    ):
        raise ValueError("remote runtime provenance is invalid")
    for timing in ("model_load_seconds", "gpu_compute_seconds", "elapsed_seconds"):
        value = runtime[timing]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"remote {timing} must be finite and nonnegative")

    items = payload["items"]
    outputs = result["outputs"]
    if type(outputs) is not list or len(outputs) != 46:
        raise ValueError("remote result must contain exactly 46 outputs")
    output_fields = {
        "prompt_id",
        "raw_output",
        "input_tokens",
        "generated_tokens",
        "generated_token_ids",
        "stop_reason",
        "hit_max_new_tokens",
        "truncated",
    }
    for item, output in zip(items, outputs, strict=True):
        if type(output) is not dict or set(output) != output_fields:
            raise ValueError("remote output fields are invalid")
        generated, token_ids = output["generated_tokens"], output["generated_token_ids"]
        if (
            output["prompt_id"] != item["prompt_id"]
            or type(output["raw_output"]) is not str
            or len(output["raw_output"]) > 100_000
            or type(output["input_tokens"]) is not int
            or output["input_tokens"] != item["prompt_tokens"]
            or type(generated) is not int
            or not 0 <= generated <= 128
            or type(token_ids) is not list
            or len(token_ids) != generated
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
            or type(output["hit_max_new_tokens"]) is not bool
            or output["hit_max_new_tokens"] is not (generated == 128)
            or type(output["truncated"]) is not bool
        ):
            raise ValueError("remote output identity, token counts, or text are invalid")
        if output["stop_reason"] == "eos_token":
            if generated == 0 or token_ids[-1] != tokenizer["eos_token_id"] or output["truncated"]:
                raise ValueError("remote EOS stop flags are inconsistent")
        elif output["stop_reason"] == "max_new_tokens":
            if (
                generated != 128
                or not output["truncated"]
                or token_ids[-1] == tokenizer["eos_token_id"]
            ):
                raise ValueError("remote token-cap stop flags are inconsistent")
        else:
            raise ValueError("remote output stop reason is invalid")
    return result
