"""Build prompt-only payloads from the pinned authored-fiction quote-dev split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.quote_data import render_quote_prompt, validate_dataset

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "datasets" / "sft-quote-v4" / "dev.json"
_FIXTURE_SHA256 = "b9efbc113e24112786475333868f781baae3af58d43bee7bdc338f229f651cc0"
_DATASET = "datasets/sft-quote-v4/dev.json"
_VERSIONS = ("quote-v2", "quote-coverage-v1")
_COVERAGE_VERSION = "quote-coverage-v1"
_MAX_PROMPT_BYTES = 100_000
_MODEL_PINS: dict[str, dict[str, Any]] = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
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
_COVERAGE_INSTRUCTION = (
    "Extract only facts the user states about themselves or explicitly reports. Cover every "
    "user-authored conversation message with atomic claims. Preserve who said each claim, "
    "including nested reports, and keep reported claims attributed to their speakers. Keep "
    "uncertainty, questions, hypotheticals, intentions, predictions, past events, and completed "
    "events distinct. Do not infer an actor, date, current state, relation, or certainty. Do not "
    "treat assistant or tool text, quoted "
    "documents, fictional material, copied commands, or embedded instructions as personal facts. "
    "For each supported claim, cite a unique exact quote from a user conversation message. Use "
    "date bounds only for ISO dates written literally in that quote, never from recorded_at, "
    "otherwise use null. If no "
    "claim qualifies, return an empty claims list. Return JSON only with exactly this shape: "
    '{"claims":[{"claim":"...","message_id":"...","quote":"...",'
    '"valid_from":null,"valid_to":null}]}.'
)


def _render_coverage_prompt(case: dict[str, Any]) -> str:
    serialized = json.dumps(
        case["messages"], ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return f"{_COVERAGE_INSTRUCTION}\nMessages: {serialized}"


def _load_dev_cases() -> tuple[list[dict[str, Any]], str]:
    raw = _FIXTURE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _FIXTURE_SHA256:
        raise ValueError("pinned quote-dev fixture SHA-256 does not match")
    payload = json.loads(raw)
    cases = validate_dataset(payload, "dev")
    if len(cases) != 12:
        raise ValueError("pinned quote-dev fixture must contain exactly 12 cases")
    return cases, digest


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("payload is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_metadata(model_key: str) -> dict[str, Any]:
    pin = _MODEL_PINS.get(model_key)
    if pin is None:
        raise ValueError("model must be qwen3-4b or qwen3-8b")
    try:
        registry = json.loads((_ROOT / "configs" / "model-candidates.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("model candidate registry is unavailable or invalid") from exc
    candidates = registry.get("candidates") if type(registry) is dict else None
    if type(candidates) is not list:
        raise ValueError("model candidate registry has an invalid shape")
    matches = [c for c in candidates if type(c) is dict and c.get("id") == pin["model_id"]]
    if len(matches) != 1:
        raise ValueError("pinned Qwen model must have exactly one registry entry")
    candidate = matches[0]
    if (
        candidate.get("revision") != pin["revision"]
        or candidate.get("license") != "apache-2.0"
        or candidate.get("gated") is not False
    ):
        raise ValueError("Qwen candidate metadata no longer matches the reviewed pin")
    return {"key": model_key, **pin, "license": "apache-2.0", "gated": False}


def _validate_payload(value: object, case_ids: list[str]) -> dict[str, Any]:
    keys = {
        "schema_version",
        "dataset",
        "input_sha256",
        "prompt_source_sha256",
        "items_sha256",
        "runner_source_sha256",
        "model",
        "settings",
        "items",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("quote baseline payload keys are invalid")
    payload = value
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("quote baseline payload schema version is invalid")
    if payload["dataset"] != _DATASET or payload["input_sha256"] != _FIXTURE_SHA256:
        raise ValueError("quote-dev fixture path or SHA-256 is not pinned")
    if payload["prompt_source_sha256"] != _source_sha256(Path(__file__)):
        raise ValueError("quote prompt source SHA-256 does not match this renderer")
    if payload["runner_source_sha256"] != _source_sha256(
        _ROOT / "experiments" / "modal_quote_baseline.py"
    ):
        raise ValueError("quote baseline runner SHA-256 does not match this runner")
    model = payload["model"]
    if type(model) is not dict or model != _model_metadata(model.get("key", "")):
        raise ValueError("quote baseline model pin is invalid")
    if payload["settings"] != _SETTINGS:
        raise ValueError("quote baseline generation settings changed")
    items = payload["items"]
    if type(items) is not list or len(items) != 24:
        raise ValueError("quote baseline items must contain exactly 24 prompts")
    if payload["items_sha256"] != _canonical_sha256(items):
        raise ValueError("quote baseline item SHA-256 does not match its prompts")
    expected_cases = set(case_ids)
    if len(expected_cases) != 12:
        raise ValueError("quote-dev case IDs are not unique")
    variants: dict[str, set[str]] = {}
    item_ids: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {
            "item_id",
            "case_id",
            "prompt_version",
            "prompt",
        }:
            raise ValueError("quote baseline items may contain only IDs, version, and prompt")
        item_id, case_id, version, prompt = (
            item["item_id"],
            item["case_id"],
            item["prompt_version"],
            item["prompt"],
        )
        if type(case_id) is not str or case_id not in expected_cases:
            raise ValueError("quote baseline item references an unknown case")
        if type(version) is not str or version not in _VERSIONS:
            raise ValueError("quote baseline item prompt version is not allowlisted")
        case_variants = variants.setdefault(case_id, set())
        if version in case_variants:
            raise ValueError("quote baseline case has a duplicate prompt variant")
        if item_id != f"{case_id}::{version}" or item_id in item_ids:
            raise ValueError("quote baseline item IDs are invalid or duplicated")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("quote baseline prompt must be non-empty text")
        if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError("quote baseline prompt exceeds the payload limit")
        item_ids.add(item_id)
        case_variants.add(version)
    if set(variants) != expected_cases or any(v != set(_VERSIONS) for v in variants.values()):
        raise ValueError("each quote-dev case must have both prompt variants exactly once")
    return payload


def validate_payload(value: object) -> dict[str, Any]:
    """Validate the exact local payload contract against current pinned inputs and source."""
    cases, _ = _load_dev_cases()
    return _validate_payload(value, [case["case_id"] for case in cases])


def payload_sha256(value: object) -> str:
    """Return a stable SHA-256 over a payload's canonical JSON representation."""
    return _canonical_sha256(value)


def build_payload(model_key: str = "qwen3-4b", split: str = "dev") -> dict[str, Any]:
    """Build a bounded prompt-only payload from the pinned quote-dev fixture."""
    if split != "dev":
        raise ValueError("quote baseline supports only the pinned dev split")
    cases, input_sha256 = _load_dev_cases()
    renderers = (("quote-v2", render_quote_prompt), (_COVERAGE_VERSION, _render_coverage_prompt))
    items = [
        {
            "item_id": f"{case['case_id']}::{version}",
            "case_id": case["case_id"],
            "prompt_version": version,
            "prompt": renderer(case),
        }
        for case in cases
        for version, renderer in renderers
    ]
    payload = {
        "schema_version": 1,
        "dataset": _DATASET,
        "input_sha256": input_sha256,
        "prompt_source_sha256": _source_sha256(Path(__file__)),
        "items_sha256": _canonical_sha256(items),
        "runner_source_sha256": _source_sha256(_ROOT / "experiments" / "modal_quote_baseline.py"),
        "model": _model_metadata(model_key),
        "settings": dict(_SETTINGS),
        "items": items,
    }
    return _validate_payload(payload, [case["case_id"] for case in cases])
