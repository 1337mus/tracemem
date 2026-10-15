"""Offline request and receipt contract for QA interpretation controls.

This module validates data only. It does not load a model/tokenizer runtime or
claim that a receipt proves tokenization, execution, decoding, or answer truth.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any

from experiments import prepare_qa_interpretation_controls

_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_RECEIPT_PATH = _ROOT / "experiments/qa_contrast_answer_receipt.py"
_EXPECTED_LEGACY_RECEIPT_SHA256 = "63e9bcfa37447f9b6e81b091d4f928bb1ba36668426260c5d99239156a7b0461"
_PREPARER_PATH = _ROOT / "experiments/prepare_qa_interpretation_controls.py"
_EXPECTED_PREPARER_SHA256 = "62c1775ffe6d295e5da39d2743fb7c97d67ccb356c19be1b012f7b01b1ebec5b"
_EXPECTED_PAYLOAD_SHA256 = "36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1"
_EXPECTED_REQUEST_SHA256 = "df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871"
_REQUEST_MODE = "qa-interpretation-controls-answer-request-v1"
_REQUEST_FIELDS = frozenset(
    "schema_version mode prepared_payload_sha256 model tokenizer items".split()
)
_MODEL_BOOL_FIELDS = (
    "trust_remote_code",
    "use_safetensors",
    "use_fast",
    "enable_thinking",
    "do_sample",
)
_TEMPLATE_OPTION_FIELDS = ("tokenize", "add_generation_prompt", "enable_thinking", "truncation")
_MODEL_RECEIPT_FIELDS = {"model_id", "revision", "generation_protocol_sha256"}
_TOKENIZER_RECEIPT_FIELDS = frozenset(
    "model_id revision template_sha256 backend_tokenizer_sha256 use_fast trust_remote_code "
    "package_versions python_version template_options eos_token_id".split()
)
_RUNTIME_RECEIPT_FIELDS = frozenset(
    "python_version torch_version accelerate_version cuda_version gpu_name gpu_capability "
    "model_load_seconds gpu_compute_seconds elapsed_seconds gpu_peak_memory_bytes "
    "request_count modal_call_id".split()
)
_OUTPUT_FIELDS = frozenset(
    "prompt_id raw_output input_tokens input_token_ids generated_tokens generated_token_ids "
    "stop_reason hit_max_new_tokens truncated".split()
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("value is not canonical JSON") from None


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def request_sha256(request: Any) -> str:
    """Return the canonical SHA-256 for a JSON-compatible request object."""
    return _sha256(_canonical_bytes(request))


def _verified_legacy_receipt() -> Any:
    source = _LEGACY_RECEIPT_PATH.read_bytes()
    if _sha256(source) != _EXPECTED_LEGACY_RECEIPT_SHA256:
        raise ValueError("legacy QA contrast answer receipt source differs from its pin")
    return importlib.import_module("experiments.qa_contrast_answer_receipt")


def _prepared_payload() -> dict[str, Any]:
    source_sha256 = _sha256(_PREPARER_PATH.read_bytes())
    if source_sha256 != _EXPECTED_PREPARER_SHA256:
        raise ValueError("QA interpretation controls preparer source differs from its pin")
    payload, _, _ = prepare_qa_interpretation_controls.prepare_controls()
    if type(payload) is not dict or request_sha256(payload) != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("prepared QA interpretation payload differs from its pin")
    return payload


def build_request() -> dict[str, Any]:
    """Build the fixed, source-only 36-prompt request without reference data."""
    legacy = _verified_legacy_receipt()
    payload = _prepared_payload()
    model = copy.deepcopy(legacy._MODEL)
    tokenizer = {
        "template_sha256": legacy._TEMPLATE_SHA256,
        "backend_tokenizer_sha256": legacy._BACKEND_SHA256,
        "package_versions": {
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
            "jinja2": "3.1.6",
        },
        "template_options": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "truncation": False,
        },
        "eos_token_id": 151645,
    }
    request = {
        "schema_version": 1,
        "mode": _REQUEST_MODE,
        "prepared_payload_sha256": _EXPECTED_PAYLOAD_SHA256,
        "model": model,
        "tokenizer": tokenizer,
        "items": [
            {"prompt_id": item["prompt_id"], "messages": copy.deepcopy(item["messages"])}
            for item in payload["items"]
        ],
    }
    if request_sha256(request) != _EXPECTED_REQUEST_SHA256:
        raise ValueError("prepared QA interpretation answer request differs from its pin")
    return request


def validate_request(request: Any) -> dict[str, Any]:
    """Reject caller changes before a future runner loads any model packages."""
    if type(request) is not dict:
        raise ValueError("request must be an object")
    if request_sha256(request) != _EXPECTED_REQUEST_SHA256:
        raise ValueError("request differs from the reviewed request hash")
    expected = build_request()
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("request schema version is invalid")
    if set(request) != _REQUEST_FIELDS:
        raise ValueError("request fields are invalid")
    model = request["model"]
    tokenizer = request["tokenizer"]
    template_options = tokenizer.get("template_options") if type(tokenizer) is dict else None
    if (
        type(model) is not dict
        or any(type(model.get(key)) is not bool for key in _MODEL_BOOL_FIELDS)
        or type(tokenizer) is not dict
        or type(template_options) is not dict
        or any(type(template_options.get(key)) is not bool for key in _TEMPLATE_OPTION_FIELDS)
    ):
        raise ValueError("request pinned settings have invalid types")
    items = request["items"]
    if type(items) is not list or len(items) != 36:
        raise ValueError("request must contain exactly 36 prompt items")
    prompt_ids: list[str] = []
    for item in items:
        if (
            type(item) is not dict
            or set(item) != {"prompt_id", "messages"}
            or type(item["prompt_id"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item["prompt_id"]) is None
            or type(item["messages"]) is not list
            or len(item["messages"]) != 2
            or any(
                type(message) is not dict
                or set(message) != {"role", "content"}
                or type(message["role"]) is not str
                or type(message["content"]) is not str
                for message in item["messages"]
            )
            or [message["role"] for message in item["messages"]] != ["system", "user"]
            or request_sha256(item["messages"]) != item["prompt_id"]
        ):
            raise ValueError("request prompt identity or messages are invalid")
        prompt_ids.append(item["prompt_id"])
    if prompt_ids != sorted(set(prompt_ids)):
        raise ValueError("request prompt IDs must be unique and sorted")
    if request != expected:
        raise ValueError("request fields differ from the reviewed source-only request")
    return request


def _valid_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _nonnegative_finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return value >= 0 and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _is_python_312(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"3\.12\.\d+", value) is not None


def validate_result(result: Any, request: Any, runner_sha256: str) -> dict[str, Any]:
    """Validate a receipt's structure, pins, and recorded IDs without inference.

    Validation cannot prove that token IDs came from the pinned tokenizer, that
    text was decoded from those IDs, that the model ran, or that an answer is
    true. A future GPU producer must record input IDs and counts remotely,
    without truncation, before model loading. Test receipts are fake and make
    no inference claim.
    """
    validate_request(request)
    if not _valid_sha256(runner_sha256):
        raise ValueError("expected runner SHA-256 must be lowercase hexadecimal")
    expected_request_sha256 = request_sha256(request)
    if type(result) is not dict or set(result) != {
        "schema_version",
        "request_sha256",
        "runner_sha256",
        "model",
        "tokenizer",
        "runtime",
        "outputs",
    }:
        raise ValueError("receipt envelope fields are invalid")
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or type(result["request_sha256"]) is not str
        or result["request_sha256"] != expected_request_sha256
        or type(result["runner_sha256"]) is not str
        or result["runner_sha256"] != runner_sha256
    ):
        raise ValueError("receipt request or runner provenance is invalid")

    expected_model = request["model"]
    model = result["model"]
    if type(model) is not dict or set(model) != _MODEL_RECEIPT_FIELDS:
        raise ValueError("receipt model provenance fields are invalid")
    if (
        type(model["model_id"]) is not str
        or model["model_id"] != expected_model["model_id"]
        or type(model["revision"]) is not str
        or model["revision"] != expected_model["revision"]
        or not _valid_sha256(model["generation_protocol_sha256"])
        or model["generation_protocol_sha256"] != request_sha256(expected_model)
    ):
        raise ValueError("receipt model identity or generation protocol hash is invalid")

    tokenizer = result["tokenizer"]
    expected_tokenizer = request["tokenizer"]
    template_options = tokenizer.get("template_options") if type(tokenizer) is dict else None
    if (
        type(tokenizer) is not dict
        or set(tokenizer) != _TOKENIZER_RECEIPT_FIELDS
        or type(tokenizer["model_id"]) is not str
        or tokenizer["model_id"] != expected_model["model_id"]
        or type(tokenizer["revision"]) is not str
        or tokenizer["revision"] != expected_model["revision"]
        or type(tokenizer["template_sha256"]) is not str
        or tokenizer["template_sha256"] != expected_tokenizer["template_sha256"]
        or type(tokenizer["backend_tokenizer_sha256"]) is not str
        or tokenizer["backend_tokenizer_sha256"] != expected_tokenizer["backend_tokenizer_sha256"]
        or tokenizer["use_fast"] is not True
        or tokenizer["trust_remote_code"] is not False
        or tokenizer["package_versions"] != expected_tokenizer["package_versions"]
        or not _is_python_312(tokenizer["python_version"])
        or type(template_options) is not dict
        or set(template_options) != set(expected_tokenizer["template_options"])
        or any(type(value) is not bool for value in template_options.values())
        or template_options != expected_tokenizer["template_options"]
        or type(tokenizer["eos_token_id"]) is not int
        or tokenizer["eos_token_id"] != expected_tokenizer["eos_token_id"]
    ):
        raise ValueError("receipt tokenizer provenance differs from the reviewed pins")

    runtime = result["runtime"]
    if type(runtime) is not dict or set(runtime) != _RUNTIME_RECEIPT_FIELDS:
        raise ValueError("receipt runtime provenance fields are invalid")
    if (
        not _is_python_312(runtime["python_version"])
        or runtime["torch_version"] != "2.7.1"
        or runtime["accelerate_version"] != "1.6.0"
        or type(runtime["cuda_version"]) is not str
        or not runtime["cuda_version"].strip()
        or type(runtime["gpu_name"]) is not str
        or runtime["gpu_name"] != "Tesla T4"
        or type(runtime["gpu_capability"]) is not str
        or runtime["gpu_capability"] != "7.5"
        or type(runtime["gpu_peak_memory_bytes"]) is not int
        or runtime["gpu_peak_memory_bytes"] < 0
        or type(runtime["request_count"]) is not int
        or runtime["request_count"] != 36
        or type(runtime["modal_call_id"]) is not str
        or not runtime["modal_call_id"].strip()
        or len(runtime["modal_call_id"]) > 256
    ):
        raise ValueError("receipt runtime provenance differs from the reviewed pins")
    if any(
        not _nonnegative_finite(runtime[key])
        for key in ("model_load_seconds", "gpu_compute_seconds", "elapsed_seconds")
    ):
        raise ValueError("receipt timings must be finite and nonnegative")

    items = request["items"]
    outputs = result["outputs"]
    if type(outputs) is not list or len(outputs) != 36:
        raise ValueError("receipt must contain exactly 36 outputs")
    total_input_tokens = total_generated_tokens = 0
    for item, output in zip(items, outputs, strict=True):
        if type(output) is not dict or set(output) != _OUTPUT_FIELDS:
            raise ValueError("receipt output fields are invalid")
        input_count = output["input_tokens"]
        input_ids = output["input_token_ids"]
        generated_count = output["generated_tokens"]
        generated_ids = output["generated_token_ids"]
        stop_reason = output["stop_reason"]
        if (
            type(output["prompt_id"]) is not str
            or output["prompt_id"] != item["prompt_id"]
            or type(output["raw_output"]) is not str
            or len(output["raw_output"]) > 65_536
            or type(input_count) is not int
            or not 1 <= input_count <= 1024
            or type(input_ids) is not list
            or len(input_ids) != input_count
            or any(type(token_id) is not int or token_id < 0 for token_id in input_ids)
            or type(generated_count) is not int
            or not 1 <= generated_count <= 128
            or type(generated_ids) is not list
            or len(generated_ids) != generated_count
            or any(type(token_id) is not int or token_id < 0 for token_id in generated_ids)
            or type(output["hit_max_new_tokens"]) is not bool
            or output["hit_max_new_tokens"] is not (generated_count == 128)
            or type(output["truncated"]) is not bool
            or output["truncated"] is not (stop_reason == "max_new_tokens")
        ):
            raise ValueError("receipt output identity, text, or token counts are invalid")
        eos_token_id = expected_tokenizer["eos_token_id"]
        if stop_reason == "eos_token":
            if generated_ids[-1] != eos_token_id or eos_token_id in generated_ids[:-1]:
                raise ValueError("receipt EOS stop must have one final EOS token")
        elif stop_reason == "max_new_tokens":
            if generated_count != 128 or eos_token_id in generated_ids:
                raise ValueError("receipt token-cap stop must have 128 non-EOS tokens")
        else:
            raise ValueError("receipt stop reason is invalid")
        total_input_tokens += input_count
        total_generated_tokens += generated_count
    if total_input_tokens > 36 * 1024 or total_generated_tokens > 36 * 128:
        raise ValueError("receipt aggregate token count exceeds the reviewed limits")
    return result
