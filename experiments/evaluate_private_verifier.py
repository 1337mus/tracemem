"""Validate and score a frozen three-arm private verifier receipt offline."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from experiments.verifier_pairs import _summarize, parse_verdict
from training.private_verifier_data import (
    _candidate_exportable,
    render_private_eval_prompts,
    validate_private_verifier_dataset,
)

_ARMS = {
    "base": "base_outputs",
    "prior": "prior_outputs",
    "continued": "continued_outputs",
}
_FINISH_REASONS = {"eos", "length", "unknown", "empty"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ITEM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _canonical_sha(value: object, name: str) -> str:
    try:
        raw = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON data") from exc
    return hashlib.sha256(raw).hexdigest()


def _validate_envelope(
    payload: object, receipt: object
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload_keys = {
        "schema_version",
        "mode",
        "provenance",
        "runner_sha256",
        "model",
        "settings",
        "candidate_manifest",
        "prior_adapter",
        "continued_adapter",
        "items",
        "training_performed",
    }
    if type(payload) is not dict or set(payload) != payload_keys:
        raise ValueError("payload keys do not match the private inference contract")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "private_inference_only"
        or payload["training_performed"] is not False
    ):
        raise ValueError("payload must be schema 1 private inference without training")
    for field in (
        "provenance",
        "model",
        "settings",
        "candidate_manifest",
        "prior_adapter",
        "continued_adapter",
    ):
        if type(payload[field]) is not dict:
            raise ValueError(f"payload {field} must be an object")
    if payload["provenance"].get("source_content_in_payload") != "minimized_prompts_only":
        raise ValueError("payload provenance must declare minimized prompts only")
    runner_sha = payload["runner_sha256"]
    if type(runner_sha) is not str or _SHA256.fullmatch(runner_sha) is None:
        raise ValueError("payload runner_sha256 is invalid")
    payload_sha = _canonical_sha(payload, "payload")

    if type(receipt) is not dict:
        raise ValueError("receipt must be an object")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("status") != "success"
        or receipt.get("mode") != "private_inference_only"
        or receipt.get("training_performed") is not False
    ):
        raise ValueError("receipt is not a successful private inference result")
    if receipt.get("payload_sha256") != payload_sha:
        raise ValueError("receipt payload SHA-256 does not match payload")
    if receipt.get("runner_sha256") != runner_sha:
        raise ValueError("receipt runner SHA-256 does not match payload")
    for field in (
        "provenance",
        "candidate_manifest",
        "model",
        "settings",
        "prior_adapter",
        "continued_adapter",
    ):
        if receipt.get(field) != payload[field]:
            raise ValueError(f"receipt {field} does not match payload")
    expected_artifacts = {
        "prior": {
            "adapter_config.json": payload["prior_adapter"].get("config_sha256"),
            "adapter_model.safetensors": payload["prior_adapter"].get("safetensors_sha256"),
        },
        "continued": payload["continued_adapter"].get("artifact_sha256"),
    }
    artifact_values = [
        expected_artifacts["prior"]["adapter_config.json"],
        expected_artifacts["prior"]["adapter_model.safetensors"],
    ]
    if type(expected_artifacts["continued"]) is not dict:
        raise ValueError("payload continued adapter artifact hashes are invalid")
    artifact_values.extend(expected_artifacts["continued"].values())
    if any(type(value) is not str or _SHA256.fullmatch(value) is None for value in artifact_values):
        raise ValueError("payload adapter artifact hashes are invalid")
    if receipt.get("adapter_artifact_sha256") != expected_artifacts:
        raise ValueError("receipt adapter artifact hashes do not match payload")
    return payload, receipt, payload_sha


def _candidate_refs(cases: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (case, candidate)
        for case in cases
        if case["split"] == "dev" and case["privacy_clearance"]["status"] == "cleared"
        for candidate in case["candidates"]
        if _candidate_exportable(candidate)
    ]


def _outputs_by_id(outputs: object, expected_ids: list[str], arm: str) -> dict[str, dict[str, Any]]:
    if type(outputs) is not list or any(type(row) is not dict for row in outputs):
        raise ValueError(f"receipt {arm} must be a list of output objects")
    ids = [row.get("item_id") for row in outputs]
    if (
        any(type(item_id) is not str for item_id in ids)
        or len(ids) != len(set(ids))
        or set(ids) != set(expected_ids)
    ):
        raise ValueError(f"{arm} output IDs must match payload items exactly once")
    return {row["item_id"]: row for row in outputs}


def _is_accept(row: dict[str, Any]) -> bool:
    return bool(
        row["valid"]
        and row["verdict"]["support"] == "supported"
        and row["verdict"]["eligibility"] == "eligible"
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _score_arm(
    refs: list[tuple[dict[str, Any], dict[str, Any]]],
    item_ids: list[str],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (case, candidate), item_id in zip(refs, item_ids, strict=True):
        output = outputs[item_id]
        raw, reason = output.get("raw_output"), output.get("finish_reason")
        if type(raw) is not str:
            raise ValueError(f"output {item_id} raw_output must be text")
        if type(reason) is not str or reason not in _FINISH_REASONS:
            raise ValueError(f"output {item_id} finish_reason is invalid")
        tokens = output.get("output_tokens")
        if "output_tokens" in output and (type(tokens) is not int or tokens < 0):
            raise ValueError(f"output {item_id} output_tokens is invalid")
        try:
            verdict = parse_verdict(raw)
        except ValueError:
            verdict = None
        valid = verdict is not None and reason == "eos"
        rows.append(
            {
                "item_id": item_id,
                "case_id": case["case_id"],
                "reference": {
                    "support": candidate["support"],
                    "eligibility": candidate["eligibility"],
                },
                "verdict": verdict,
                "valid": valid,
                "malformed": verdict is None,
                "finish_reason": reason,
                "output_tokens": tokens,
            }
        )

    # Private cases may contain any number of candidates; public pair metrics would
    # impose a two-candidate assumption, so keep only their generic axis summaries.
    summary = _summarize(rows, [])
    summary.pop("support_pairs")
    summary["case_count"] = len({row["case_id"] for row in rows})
    useful = [
        row
        for row in rows
        if row["reference"]["support"] == "supported"
        and row["reference"]["eligibility"] == "eligible"
    ]
    false_accepts = [
        row
        for row in rows
        if row["reference"]["support"] == "unsupported"
        and row["reference"]["eligibility"] == "eligible"
    ]
    return {
        "summary": summary,
        "joint_acceptance": {
            "useful_supported_eligible": {
                "total": len(useful),
                "accepted": sum(_is_accept(row) for row in useful),
                "rate": _rate(sum(_is_accept(row) for row in useful), len(useful)),
            },
            "unsupported_eligible_false_accepts": {
                "total": len(false_accepts),
                "accepted": sum(_is_accept(row) for row in false_accepts),
                "rate": _rate(sum(_is_accept(row) for row in false_accepts), len(false_accepts)),
            },
        },
    }


def evaluate_report(
    dataset: object, source_records: object, payload: object, receipt: object
) -> dict[str, Any]:
    """Validate private dev prompts and score base, prior, and continued outputs."""
    payload_obj, receipt_obj, payload_sha = _validate_envelope(payload, receipt)
    cases = validate_private_verifier_dataset(dataset, source_records)
    refs = _candidate_refs(cases)
    expected_prompts = render_private_eval_prompts(dataset, source_records, "dev")
    items = payload_obj["items"]
    if type(items) is not list or not items:
        raise ValueError("payload items must contain exportable dev prompts")
    if len(items) != len(refs) or len(expected_prompts) != len(refs):
        raise ValueError("payload item count does not match exportable rendered dev prompts")

    item_ids: list[str] = []
    for index, (item, expected_prompt) in enumerate(zip(items, expected_prompts, strict=True)):
        if type(item) is not dict or set(item) != {"item_id", "prompt"}:
            raise ValueError("payload items must contain only opaque item_id and prompt")
        item_id, prompt = item["item_id"], item["prompt"]
        if type(item_id) is not str or _ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("payload item_id must be an opaque alphanumeric ID")
        if item_id in item_ids:
            raise ValueError("payload item IDs must be unique")
        if type(prompt) is not str or prompt != expected_prompt:
            raise ValueError(f"payload item {index} differs from rendered dev prompts")
        item_ids.append(item_id)

    arms = {
        name: _score_arm(refs, item_ids, _outputs_by_id(receipt_obj.get(field), item_ids, name))
        for name, field in _ARMS.items()
    }
    sessions = {case["session_id"] for case, _ in refs}
    agent_only = all(
        candidate["label_status"] == "agent_provisional"
        and candidate["annotator"]["reviewer_type"] == "agent"
        and all(review["reviewer_type"] == "agent" for review in candidate["reviews"])
        for _, candidate in refs
    )
    label_basis = (
        "agent-only provisional annotations"
        if agent_only
        else "mixed or independently reviewed annotations"
    )
    session_count = len(sessions)
    session_scope = "one-session" if session_count == 1 else f"{session_count}-session"
    interpretation = f"{session_scope} dev challenge with {label_basis}; not held out."
    return {
        "schema_version": 1,
        "payload_sha256": payload_sha,
        "receipt_sha256": _canonical_sha(receipt_obj, "receipt"),
        "scope": {
            "split": "dev",
            "case_count": len({case["case_id"] for case, _ in refs}),
            "candidate_count": len(refs),
            "session_count": session_count,
            "held_out": False,
            "label_basis": label_basis,
            "interpretation": interpretation,
        },
        "arms": arms,
    }
