"""Build a source-free verifier payload from pinned quote-dev artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from experiments.quote_verifier import render_verifier_prompt
from training.evaluate_quote_sft import _json_object, _strict_schema_claims
from training.quote_data import validate_claims, validate_dataset

_ROOT = Path(__file__).resolve().parents[1]
_DEV_PATH = _ROOT / "datasets" / "sft-quote-v4" / "dev.json"
_GENERATOR_PATH = _ROOT / "examples" / "synthetic" / "quote-baseline-qwen3-4b-remote-v1.json"
_DEV_SHA256 = "b9efbc113e24112786475333868f781baae3af58d43bee7bdc338f229f651cc0"
_GENERATOR_SHA256 = "d47debe18192dae62759ed097017d2a9bcef92c148692b2b9a1e41e3805b2ac3"
_CASE_COUNTS = {
    "V4D-01": 3,
    "V4D-02": 4,
    "V4D-03": 4,
    "V4D-04": 3,
    "V4D-05": 6,
    "V4D-06": 4,
    "V4D-07": 7,
    "V4D-08": 2,
    "V4D-09": 4,
    "V4D-12": 2,
}
_MODEL = {
    "key": "qwen3-8b",
    "model_id": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "license": "apache-2.0",
    "gated": False,
}
_SETTINGS: dict[str, Any] = {
    "precision": "bfloat16",
    "max_new_tokens": 1024,
    "max_prompt_tokens": 4096,
    "do_sample": False,
    "num_beams": 1,
    "attn_implementation": "eager",
    "trust_remote_code": False,
    "use_safetensors": True,
    "enable_thinking": False,
}
_MAX_PROMPT_BYTES = 100_000


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    dev_bytes = _DEV_PATH.read_bytes()
    dev_sha = hashlib.sha256(dev_bytes).hexdigest()
    if dev_sha != _DEV_SHA256:
        raise ValueError("pinned quote-dev fixture SHA-256 does not match")
    cases = validate_dataset(json.loads(dev_bytes), "dev")
    if len(cases) != 12 or len({case["case_id"] for case in cases}) != 12:
        raise ValueError("pinned quote-dev fixture must contain 12 unique cases")

    generator_bytes = _GENERATOR_PATH.read_bytes()
    generator_sha = hashlib.sha256(generator_bytes).hexdigest()
    if generator_sha != _GENERATOR_SHA256:
        raise ValueError("pinned quote generator receipt SHA-256 does not match")
    receipt = json.loads(generator_bytes)
    if (
        type(receipt) is not dict
        or receipt.get("schema_version") != 1
        or receipt.get("mode") != "quote-baseline-prompt-only"
        or receipt.get("input_sha256") != _DEV_SHA256
        or type(receipt.get("generated_outputs")) is not list
    ):
        raise ValueError("pinned quote generator receipt has an invalid shape")
    return cases, receipt, dev_sha, generator_sha


def _quote_v2_rows(receipt: dict[str, Any], case_ids: set[str]) -> dict[str, dict[str, Any]]:
    outputs = receipt["generated_outputs"]
    rows = [row for row in outputs if type(row) is dict and row.get("prompt_version") == "quote-v2"]
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if (
            type(case_id) is not str
            or case_id not in case_ids
            or case_id in by_case
            or row.get("item_id") != f"{case_id}::quote-v2"
        ):
            raise ValueError("quote-v2 generator case IDs must be a unique bijection")
        by_case[case_id] = row
    if set(by_case) != case_ids or len(rows) != 12:
        raise ValueError("quote-v2 generator case IDs must cover all 12 dev cases")
    return by_case


def _select_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    raw_text = row.get("raw_output")
    hit_limit = row.get("hit_max_new_tokens")
    finish_reason = row.get("finish_reason")
    metadata_valid = (
        type(hit_limit) is bool
        and type(row.get("ended_with_eos")) is bool
        and type(finish_reason) is str
        and finish_reason in {"empty", "eos", "length", "unknown"}
    )
    truncated = hit_limit is True or finish_reason == "length"
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "source_item_id": row["item_id"],
        "status": "semantic-ready",
        "pipeline": "semantic-pipeline",
        "candidate_indices": [],
        "candidates": [],
        "rejected": [],
        "truncated": truncated,
        "finish_reason": finish_reason,
        "generator_error": None,
    }
    if not metadata_valid:
        result.update(status="invalid-generator-metadata", pipeline="deterministic-only")
        result["generator_error"] = "generator hit_max_new_tokens or finish metadata is invalid"
        return result
    if type(raw_text) is not str:
        result.update(status="invalid-json", pipeline="deterministic-only")
        result["generator_error"] = "generator raw_output must be text"
        return result
    if truncated:
        result.update(status="truncated", pipeline="deterministic-only")
        result["generator_error"] = "generator output reached max_new_tokens"
        return result
    try:
        parsed = _json_object(raw_text.encode("utf-8"), "generator response")
    except (UnicodeEncodeError, ValueError) as exc:
        result.update(status="invalid-json", pipeline="deterministic-only")
        result["generator_error"] = str(exc)
        return result
    try:
        claims = _strict_schema_claims(parsed)
    except ValueError as exc:
        result.update(status="invalid-schema", pipeline="deterministic-only")
        result["generator_error"] = str(exc)
        return result
    if len(claims) > 64:
        result.update(status="invalid-schema", pipeline="deterministic-only")
        result["generator_error"] = "generator response exceeds 64 candidates"
        return result
    for index, candidate in enumerate(claims):
        try:
            validate_claims({"claims": [candidate]}, case["messages"])
        except ValueError as exc:
            result["rejected"].append({"source_index": index, "reason": str(exc)})
        else:
            result["candidate_indices"].append(index)
            result["candidates"].append(candidate)
    if not result["candidates"]:
        result.update(status="no-valid-candidates", pipeline="deterministic-only")
    return result


def _source_selection_sha256(audited_cases: list[dict[str, Any]]) -> str:
    selected = [
        {
            "case_id": case["case_id"],
            "source_item_id": case["source_item_id"],
            "candidate_indices": case["candidate_indices"],
            "candidates": case["candidates"],
        }
        for case in audited_cases
    ]
    return _canonical_sha256(selected)


def _validate_payload(value: object) -> dict[str, Any]:
    keys = {
        "schema_version",
        "mode",
        "input_sha256",
        "generator_receipt_sha256",
        "candidate_sha256",
        "items_sha256",
        "renderer_source_sha256",
        "payload_source_sha256",
        "runner_source_sha256",
        "model",
        "settings",
        "items",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("quote verifier payload keys are invalid")
    payload = value
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("quote verifier schema version is invalid")
    if payload["mode"] != "semantic-pipeline":
        raise ValueError("quote verifier mode is invalid")
    if (
        payload["input_sha256"] != _DEV_SHA256
        or payload["generator_receipt_sha256"] != _GENERATOR_SHA256
    ):
        raise ValueError("quote verifier input hashes are not allowlisted")
    if payload["renderer_source_sha256"] != _source_sha256(
        _ROOT / "experiments" / "quote_verifier.py"
    ):
        raise ValueError("quote verifier renderer source hash changed")
    if payload["payload_source_sha256"] != _source_sha256(Path(__file__)):
        raise ValueError("quote verifier payload source hash changed")
    runner_path = _ROOT / "experiments" / "modal_quote_verifier.py"
    if payload["runner_source_sha256"] != _source_sha256(runner_path):
        raise ValueError("quote verifier runner source hash changed")
    if payload["model"] != _MODEL or payload["settings"] != _SETTINGS:
        raise ValueError("quote verifier model or settings changed")
    items = payload["items"]
    if type(items) is not list or payload["items_sha256"] != _canonical_sha256(items):
        raise ValueError("quote verifier item hash does not match its prompts")
    if payload["candidate_sha256"] != _EXPECTED_CANDIDATE_SHA256:
        raise ValueError("quote verifier candidate hash is not allowlisted")
    if len(items) != len(_CASE_COUNTS):
        raise ValueError("quote verifier must contain exactly 10 semantic batches")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {
            "case_id",
            "item_id",
            "prompt",
            "candidate_count",
        }:
            raise ValueError(
                "quote verifier items may contain only IDs, prompt, and candidate count"
            )
        case_id = item["case_id"]
        count = item["candidate_count"]
        prompt = item["prompt"]
        if type(case_id) is not str or case_id not in _CASE_COUNTS or case_id in seen:
            raise ValueError("quote verifier case IDs are invalid or duplicated")
        if item["item_id"] != f"{case_id}::quote-verifier-v1":
            raise ValueError("quote verifier item ID is invalid")
        if type(count) is not int or count != _CASE_COUNTS[case_id]:
            raise ValueError("quote verifier candidate count is invalid")
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES
        ):
            raise ValueError("quote verifier prompt is empty or exceeds the byte limit")
        seen.add(case_id)
    if seen != set(_CASE_COUNTS):
        raise ValueError("quote verifier items must cover the approved non-empty cases")
    return payload


def build_preparation() -> dict[str, Any]:
    """Build the semantic payload and local-only selection audit from pinned inputs."""
    cases, receipt, dev_sha, generator_sha = _load_inputs()
    case_ids = {case["case_id"] for case in cases}
    rows = _quote_v2_rows(receipt, case_ids)
    audited_cases = [_select_case(case, rows[case["case_id"]]) for case in cases]
    candidate_sha = _source_selection_sha256(audited_cases)
    if {
        case["case_id"]: len(case["candidates"]) for case in audited_cases if case["candidates"]
    } != _CASE_COUNTS:
        raise ValueError("strict and citation gates no longer match the reviewed candidate set")
    items = []
    for case, selected in zip(cases, audited_cases, strict=True):
        if not selected["candidates"]:
            continue
        case_id = case["case_id"]
        items.append(
            {
                "case_id": case_id,
                "item_id": f"{case_id}::quote-verifier-v1",
                "prompt": render_verifier_prompt(case["messages"], selected["candidates"]),
                "candidate_count": len(selected["candidates"]),
            }
        )
    payload = {
        "schema_version": 1,
        "mode": "semantic-pipeline",
        "input_sha256": dev_sha,
        "generator_receipt_sha256": generator_sha,
        "candidate_sha256": candidate_sha,
        "items_sha256": _canonical_sha256(items),
        "renderer_source_sha256": _source_sha256(_ROOT / "experiments" / "quote_verifier.py"),
        "payload_source_sha256": _source_sha256(Path(__file__)),
        "runner_source_sha256": _source_sha256(_ROOT / "experiments" / "modal_quote_verifier.py"),
        "model": dict(_MODEL),
        "settings": dict(_SETTINGS),
        "items": items,
    }
    _validate_payload(payload)
    audit = {
        "schema_version": 1,
        "mode": "local-selection-audit",
        "input_sha256": dev_sha,
        "generator_receipt_sha256": generator_sha,
        "candidate_sha256": candidate_sha,
        "cases": audited_cases,
    }
    return {"payload": payload, "audit": audit}


def build_payload() -> dict[str, Any]:
    """Return only the outbound payload, without local selection details."""
    return cast(dict[str, Any], build_preparation()["payload"])


def validate_payload(value: object) -> dict[str, Any]:
    """Validate payload hashes, model pins, candidate counts, and item bounds."""
    return _validate_payload(value)


def payload_sha256(value: object) -> str:
    """Hash canonical JSON for an approved quote verifier payload."""
    return _canonical_sha256(value)


_EXPECTED_CANDIDATE_SHA256 = "2e60e4c0b5eadff29b517814f6b1040d3e10e6b453ed796d6015fce5df8f3041"
