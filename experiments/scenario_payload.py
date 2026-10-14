"""Fixture-only payload construction shared by the scenario runner and evaluator."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tracemem.prompts import render_explicit_prompt
from tracemem.scenarios import load_scenarios

_ROOT = Path(__file__).resolve().parents[1]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MAX_PROMPT_BYTES = 100_000
_MODEL_LICENSES = {
    "Qwen/Qwen3-4B": "apache-2.0",
    "microsoft/Phi-3.5-mini-instruct": "mit",
}
_MODEL_REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "microsoft/Phi-3.5-mini-instruct": "2fe192450127e6a83f7441aef6e3ca586c338b77",
}
_SECRET_FIELD = re.compile(
    r"(?i)\b((?:authorization|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"token|secret|password)\s*[:=]\s*)"
    r"(?:bearer\s+)?[^\s,;]+"
)
_BEARER_SECRET = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/-]+=*")
_MODAL_TOKEN = re.compile(r"\b(?:ak|as)-[A-Za-z0-9_-]{6,}\b")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/:@\s]+:[^/@\s]+@")


def verify_personal_workspace() -> None:
    """Require a fresh Modal token check matching the saved personal workspace."""
    status = _read_json(_ROOT / ".local" / "modal-status.json")
    expected = status.get("expected_workspace")
    if type(expected) is not str or not expected or status.get("profile") != "reflex-personal":
        raise RuntimeError("saved personal Modal workspace is unavailable")
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
        raise RuntimeError("fresh personal Modal workspace check failed") from None
    if result.returncode != 0:
        raise RuntimeError("fresh personal Modal workspace check failed")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    matches = re.findall(r"^Workspace:\s*([^\s(]+)", clean, flags=re.MULTILINE)
    if len(matches) != 1 or matches[0] != expected:
        raise RuntimeError("active Modal workspace does not match the saved personal workspace")


def safe_exception_message(error: Exception) -> str:
    """Keep useful errors while redacting common credential forms."""
    try:
        message = str(error)
    except Exception:
        return "exception message unavailable"
    message = re.sub(r"[\x00-\x1f\x7f]+", " ", message)
    message = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", message)
    message = _BEARER_SECRET.sub(r"\1[REDACTED]", message)
    message = _SECRET_FIELD.sub(r"\1[REDACTED]", message)
    message = _MODAL_TOKEN.sub("[REDACTED]", message)
    return re.sub(r"\s+", " ", message).strip()[:400] or "no exception message"


def _comparison_plan() -> dict[str, Any]:
    plan = _read_json(_ROOT / "configs" / "scenario-comparison-plan.json")
    required = {
        "version",
        "comparison_models",
        "dataset",
        "dataset_sha256",
        "decoding",
        "local_model_inference",
        "max_generated_tokens",
        "max_input_tokens",
        "partitions",
        "prompt_source_sha256",
        "prompt_version",
        "provider_billing_cap",
        "report_partitions_separately",
        "training",
        "automatic_accept",
    }
    if not required.issubset(plan) or type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("scenario comparison plan is incomplete or unsupported")
    if (
        plan["dataset"] != "tracemem/data/scenarios_v1.json"
        or plan["comparison_models"] != ["Qwen/Qwen3-4B", "microsoft/Phi-3.5-mini-instruct"]
        or plan["decoding"] != "greedy"
        or plan["max_generated_tokens"] != 1024
        or plan["max_input_tokens"] != 4096
        or plan["partitions"] != {"dev": 24, "test": 24, "train": 0}
        or plan["prompt_version"] != "explicit-v2"
        or plan["local_model_inference"] is not False
        or plan["training"] is not False
        or plan["provider_billing_cap"] is not False
        or plan["report_partitions_separately"] is not True
        or plan["automatic_accept"] is not False
    ):
        raise ValueError("scenario comparison plan no longer matches the inference contract")
    source_hash = plan["prompt_source_sha256"]
    actual_source_hash = hashlib.sha256(
        (_ROOT / "src" / "tracemem" / "prompts.py").read_bytes()
    ).hexdigest()
    if source_hash != actual_source_hash:
        raise ValueError("frozen explicit-v2 prompt source hash does not match prompts.py")
    dataset_hash = plan["dataset_sha256"]
    if dataset_hash is not None and (
        type(dataset_hash) is not str or _SHA256.fullmatch(dataset_hash) is None
    ):
        raise ValueError("scenario comparison plan dataset hash is invalid")
    return plan


def verify_launch_preconditions(payload: dict[str, Any]) -> None:
    """Require root's dataset SHA freeze before any inference call can start."""
    plan = _comparison_plan()
    if plan["dataset_sha256"] is None:
        raise RuntimeError("scenario dataset SHA-256 must be frozen after root audit before launch")
    if plan["dataset_sha256"] != payload["input_sha256"]:
        raise RuntimeError("scenario dataset SHA-256 does not match the root-audited plan")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read local config file: {path.name}") from exc
    if type(value) is not dict:
        raise ValueError(f"scenario inference config has an invalid shape: {path.name}")
    return value


def _inference_config() -> dict[str, Any]:
    _comparison_plan()
    config = _read_json(_ROOT / "configs" / "scenario-inference.json")
    expected = {
        "schema_version",
        "prompt_version",
        "case_count",
        "max_new_tokens",
        "max_prompt_tokens",
        "precision",
        "do_sample",
        "num_beams",
        "attn_implementation",
        "trust_remote_code",
        "use_safetensors",
        "enable_thinking",
        "models",
    }
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("scenario inference config keys or version are invalid")
    fixed = {
        "prompt_version": "explicit-v2",
        "case_count": 48,
        "max_new_tokens": 1024,
        "max_prompt_tokens": 4096,
        "precision": "float16",
        "do_sample": False,
        "num_beams": 1,
        "attn_implementation": "eager",
        "trust_remote_code": False,
        "use_safetensors": True,
        "enable_thinking": False,
    }
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("scenario inference settings do not match the reviewed contract")
    models = config["models"]
    if models != {
        "qwen3-4b": "Qwen/Qwen3-4B",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    }:
        raise ValueError("scenario inference model aliases are invalid")
    return config


def _model_settings(model_key: str) -> dict[str, Any]:
    config = _inference_config()
    model_id = config["models"].get(model_key)
    if not isinstance(model_id, str):
        raise ValueError("model must be qwen3-4b or phi-3.5-mini")
    candidates_config = _read_json(_ROOT / "configs" / "model-candidates.json")
    candidates = candidates_config.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("model candidate configuration is invalid")
    matches = [
        candidate
        for candidate in candidates
        if type(candidate) is dict and candidate.get("id") == model_id
    ]
    if len(matches) != 1:
        raise ValueError("pinned scenario model must have exactly one model-candidates entry")
    candidate = matches[0]
    revision, license_name = candidate.get("revision"), candidate.get("license")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("scenario model revision is not a pinned commit")
    if revision != _MODEL_REVISIONS[model_id]:
        raise ValueError("scenario model revision differs from the reviewed candidate pin")
    if license_name != _MODEL_LICENSES[model_id] or candidate.get("gated") is not False:
        raise ValueError("scenario model public license or access status is not approved")
    return {
        "key": model_key,
        "model_id": model_id,
        "revision": revision,
        "precision": config["precision"],
        "max_new_tokens": config["max_new_tokens"],
        "max_prompt_tokens": config["max_prompt_tokens"],
        "do_sample": config["do_sample"],
        "num_beams": config["num_beams"],
        "attn_implementation": config["attn_implementation"],
        "trust_remote_code": config["trust_remote_code"],
        "use_safetensors": config["use_safetensors"],
        "enable_thinking": config["enable_thinking"],
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the reviewed payload using stable canonical JSON."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return _canonical_sha256(payload)


def build_payload(model_key: str = "qwen3-4b") -> dict[str, Any]:
    """Build the exact 48-case, explicit-v2 payload without labels or metadata."""
    model = _model_settings(model_key)
    cases, _manifest, fixture_hash = load_scenarios()
    plan = _comparison_plan()
    if plan["dataset_sha256"] is not None and plan["dataset_sha256"] != fixture_hash:
        raise ValueError("scenario fixture SHA-256 does not match the root-audited comparison plan")
    if not isinstance(fixture_hash, str) or _SHA256.fullmatch(fixture_hash) is None:
        raise ValueError("scenario fixture hash is invalid")
    if type(cases) is not list or len(cases) != 48:
        raise ValueError("scenario inference requires exactly 48 current fixture cases")
    splits: dict[str, int] = {"dev": 0, "test": 0}
    seen: set[str] = set()
    items: list[dict[str, str]] = []
    for case in cases:
        if type(case) is not dict:
            raise ValueError("scenario fixture contains an invalid case")
        case_id, split = case.get("case_id"), case.get("split")
        if type(case_id) is not str or not case_id.strip() or case_id in seen:
            raise ValueError("scenario case IDs must be unique non-empty strings")
        if split not in splits:
            raise ValueError("scenario fixture must contain dev and test cases only")
        seen.add(case_id)
        splits[split] += 1
        prompt = render_explicit_prompt(case)
        if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError("scenario prompt exceeds the payload limit")
        items.append({"case_id": case_id, "prompt": prompt})
    if splits != {"dev": 24, "test": 24}:
        raise ValueError("scenario inference requires exactly 24 dev and 24 test cases")
    prompt_hash = _canonical_sha256(items)
    payload = {
        "schema_version": 1,
        "input_sha256": fixture_hash,
        "prompt_source_sha256": plan["prompt_source_sha256"],
        "prompt_sha256": prompt_hash,
        "prompt_version": "explicit-v2",
        "model": model,
        "items": items,
    }
    _validate_payload(payload, model_key=model_key)
    return payload


def _validate_payload(payload: object, *, model_key: str) -> None:
    expected_keys = {
        "schema_version",
        "input_sha256",
        "prompt_source_sha256",
        "prompt_sha256",
        "prompt_version",
        "model",
        "items",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("scenario payload keys do not match the allowlist")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("scenario payload schema version is unsupported")
    if payload["prompt_version"] != "explicit-v2":
        raise ValueError("scenario payload prompt version is unsupported")
    for key in ("input_sha256", "prompt_sha256"):
        if type(payload[key]) is not str or _SHA256.fullmatch(payload[key]) is None:
            raise ValueError(f"scenario payload {key} is invalid")
    if payload["prompt_source_sha256"] != _comparison_plan()["prompt_source_sha256"]:
        raise ValueError("scenario payload prompt source is not the frozen explicit-v2 renderer")
    if payload["model"] != _model_settings(model_key):
        raise ValueError("scenario payload model settings are not allowlisted")
    items = payload["items"]
    if type(items) is not list or len(items) != 48:
        raise ValueError("scenario payload must contain exactly 48 items")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"case_id", "prompt"}:
            raise ValueError("scenario payload items may contain only case_id and prompt")
        case_id, prompt = item["case_id"], item["prompt"]
        if type(case_id) is not str or not case_id.strip() or case_id in seen:
            raise ValueError("scenario payload case IDs must be unique and non-empty")
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES
        ):
            raise ValueError("scenario payload prompt is invalid or too large")
        seen.add(case_id)
    if payload["prompt_sha256"] != _canonical_sha256(items):
        raise ValueError("scenario payload prompt hash does not match its items")


__all__ = [
    "build_payload",
    "payload_sha256",
    "safe_exception_message",
    "verify_personal_workspace",
    "verify_launch_preconditions",
    "load_scenarios",
]
