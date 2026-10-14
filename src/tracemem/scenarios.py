"""Versioned synthetic scenario data and strict validation."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from importlib.resources import files
from typing import Any

from .challenge import _validate_case

_ROOT_KEYS = frozenset({"version", "license", "annotation_status", "cases", "case_metadata"})
_MANIFEST_KEYS = ("version", "license", "annotation_status", "case_metadata")
_METADATA_KEYS = frozenset({"scenario_group", "template_group", "entity_ids"})
_UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_EXPECTED_SPLIT_COUNTS = {"dev": 24, "test": 24}


def load_scenarios() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Load the packaged scenario fixture and return its cases, manifest and hash."""
    raw = files("tracemem").joinpath("data", "scenarios_v1.json").read_bytes()
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("scenario fixture is not valid JSON") from exc
    cases, manifest = validate_scenarios(payload)
    return cases, manifest, hashlib.sha256(raw).hexdigest()


def validate_scenarios(payload: object) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the complete authored split and return cases plus public manifest."""
    if type(payload) is not dict or set(payload) != _ROOT_KEYS:
        raise ValueError("scenario fixture must contain only the required root fields")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported scenario version")
    if payload["license"] != "MIT":
        raise ValueError("scenario license must be MIT")
    if payload["annotation_status"] != "agent_authored":
        raise ValueError("scenario annotation_status must be agent_authored")

    cases = payload["cases"]
    metadata = payload["case_metadata"]
    if type(cases) is not list or len(cases) != sum(_EXPECTED_SPLIT_COUNTS.values()):
        raise ValueError("scenario cases must contain exactly 48 cases")
    if type(metadata) is not dict:
        raise ValueError("scenario case_metadata must be an object")

    case_ids: set[str] = set()
    scenario_ids: set[str] = set()
    split_counts = {split: 0 for split in _EXPECTED_SPLIT_COUNTS}
    no_memory_count = 0
    for case in cases:
        _validate_case(case)
        case_id = case["case_id"]
        scenario_id = case["scenario_id"]
        split = case["split"]
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        if scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario_id: {scenario_id}")
        case_ids.add(case_id)
        scenario_ids.add(scenario_id)
        if split not in split_counts:
            raise ValueError("scenario split must be dev or test")
        split_counts[split] += 1
        no_memory_count += int(not case["expected"])

        session_ids: set[str] = set()
        previous_time: datetime | None = None
        for message in case["messages"]:
            session_ids.add(message["session_id"])
            timestamp = message["recorded_at"]
            if type(timestamp) is not str or _UTC_TIMESTAMP.fullmatch(timestamp) is None:
                raise ValueError(f"case {case_id} timestamps must be canonical UTC seconds")
            try:
                parsed_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError as exc:
                raise ValueError(f"case {case_id} has an invalid timestamp") from exc
            if previous_time is not None and parsed_time <= previous_time:
                raise ValueError(f"case {case_id} timestamps must be strictly increasing")
            previous_time = parsed_time
        if len(case["messages"]) not in {3, 4, 5}:
            raise ValueError(f"case {case_id} must contain 3 to 5 messages")
        if len(session_ids) < 2:
            raise ValueError(f"case {case_id} must span at least two sessions")

    if split_counts != _EXPECTED_SPLIT_COUNTS:
        raise ValueError("scenario split counts must be exactly 24 dev and 24 test")
    if no_memory_count != 8:
        raise ValueError("scenario fixture must contain exactly 8 no-memory cases")
    if set(metadata) != case_ids:
        raise ValueError("case_metadata keys must exactly match case IDs")

    grouped_splits: dict[str, dict[str, str]] = {
        "scenario_group": {},
        "template_group": {},
        "entity_id": {},
    }
    for case in cases:
        case_id = case["case_id"]
        split = case["split"]
        entry = metadata[case_id]
        if type(entry) is not dict or set(entry) != _METADATA_KEYS:
            raise ValueError(f"metadata for {case_id} has missing or unexpected fields")
        for field in ("scenario_group", "template_group"):
            value = _nonblank(entry[field], f"{field} for {case_id}")
            prior = grouped_splits[field].setdefault(value, split)
            if prior != split:
                raise ValueError(f"{field} {value} crosses scenario splits")
        entity_ids = entry["entity_ids"]
        if (
            type(entity_ids) is not list
            or not entity_ids
            or any(type(value) is not str or not value.strip() for value in entity_ids)
            or len(set(entity_ids)) != len(entity_ids)
        ):
            raise ValueError(f"entity_ids for {case_id} must be unique non-blank strings")
        for entity_id in entity_ids:
            prior = grouped_splits["entity_id"].setdefault(entity_id, split)
            if prior != split:
                raise ValueError(f"entity_id {entity_id} crosses scenario splits")

    manifest = {key: payload[key] for key in _MANIFEST_KEYS}
    return cases, manifest


def _nonblank(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


__all__ = ["load_scenarios", "validate_scenarios"]
