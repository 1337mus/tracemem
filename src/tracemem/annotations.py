"""Blind local review bundles for candidate claim annotations."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from .challenge import _validate_case
from .claims import parse_candidates
from .schema import EvidenceRecord

_BUNDLE_VERSION = 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METADATA_KEYS = {"scenario_group", "template_group", "entity_ids"}
_TASK_KEYS = {"task_id", "messages", "candidate"}
_MESSAGE_KEYS = {
    "session_id",
    "message_id",
    "role",
    "text",
    "recorded_at",
    "content_kind",
}
_RESPONSE_KEYS = {
    "task_id",
    "label",
    "rationale",
    "reviewer_id",
    "reviewer_kind",
    "timestamp",
}


def export_review_bundle(
    cases: Sequence[dict[str, Any]],
    dataset_sha256: str,
    foils_by_case: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    *,
    seed: int,
    case_metadata: Mapping[str, dict[str, Any]] | None = None,
    unlabeled_by_case: Mapping[str, Sequence[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build shuffled blind tasks and a separate answer key.

    Cases without gold, foil, or unlabeled candidates are retained in the key but
    create no review task.
    """
    _require_sha256(dataset_sha256, "dataset_sha256")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("cases must be a non-empty sequence")

    case_ids: set[str] = set()
    checked_cases: list[dict[str, Any]] = []
    for case in cases:
        structural_case = {**case, "expected": []} if type(case) is dict else case
        _validate_case(structural_case)
        parse_candidates({"claims": case["expected"]})
        if case["case_id"] in case_ids:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        case_ids.add(case["case_id"])
        checked_cases.append(case)

    foils = {} if foils_by_case is None else foils_by_case
    if not isinstance(foils, Mapping):
        raise TypeError("foils_by_case must map case IDs to candidate lists")
    if any(type(case_id) is not str for case_id in foils):
        raise TypeError("foil case IDs must be strings")
    unknown_foil_cases = set(foils).difference(case_ids)
    if unknown_foil_cases:
        raise ValueError(f"foils reference unknown case_id: {sorted(unknown_foil_cases)[0]}")
    unlabeled = {} if unlabeled_by_case is None else unlabeled_by_case
    if not isinstance(unlabeled, Mapping):
        raise TypeError("unlabeled_by_case must map case IDs to candidate lists")
    if any(type(case_id) is not str for case_id in unlabeled):
        raise TypeError("unlabeled candidate case IDs must be strings")
    unknown_unlabeled_cases = set(unlabeled).difference(case_ids)
    if unknown_unlabeled_cases:
        raise ValueError(
            f"unlabeled candidates reference unknown case_id: {sorted(unknown_unlabeled_cases)[0]}"
        )
    metadata = {} if case_metadata is None else case_metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("case_metadata must map case IDs to metadata")
    if any(type(case_id) is not str for case_id in metadata):
        raise TypeError("metadata case IDs must be strings")
    if set(metadata).difference(case_ids):
        raise ValueError("case_metadata references an unknown case_id")
    checked_metadata: dict[str, dict[str, Any]] = {}
    for case_id, row in metadata.items():
        _validate_case_metadata(row)
        checked_metadata[case_id] = copy.deepcopy(row)

    tasks: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    seen_task_content: set[str] = set()
    for case in checked_cases:
        case_id = case["case_id"]
        case_foils = foils.get(case_id, [])
        if not isinstance(case_foils, Sequence) or isinstance(case_foils, (str, bytes)):
            raise TypeError("each foil collection must be a sequence")
        case_unlabeled = unlabeled.get(case_id, [])
        if not isinstance(case_unlabeled, Sequence) or isinstance(case_unlabeled, (str, bytes)):
            raise TypeError("each unlabeled candidate collection must be a sequence")
        candidates: list[tuple[str, str | None, dict[str, Any]]] = []
        for membership, reference_label, records in (
            ("gold", "supported", case["expected"]),
            ("foil", "unsupported", case_foils),
            ("unlabeled", None, case_unlabeled),
        ):
            for record in records:
                if type(record) is not dict:
                    raise TypeError("candidate records must be objects")
                parse_candidates({"claims": [record]})
                candidates.append((membership, reference_label, record))
        local_candidates: set[str] = set()
        for membership, reference_label, record in candidates:
            candidate_digest = _sha256(_canonical_json(record))
            if candidate_digest in local_candidates:
                raise ValueError(f"duplicate candidate task in case {case_id}")
            local_candidates.add(candidate_digest)
            messages, candidate = _blind_content(case["messages"], record)
            visible_content = {"messages": messages, "candidate": candidate}
            content_key = _canonical_json(visible_content)
            if content_key in seen_task_content:
                raise ValueError("duplicate identical candidate task")
            seen_task_content.add(content_key)
            task_id = "task_" + _sha256(
                _canonical_json({"dataset_sha256": dataset_sha256, "task": visible_content})
            )
            tasks.append({"task_id": task_id, **visible_content})
            key_rows.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "split": case["split"],
                    "membership": membership,
                    "reference_label": reference_label,
                }
            )

    random.Random(seed).shuffle(tasks)
    task_ids = {task["task_id"] for task in tasks}
    key_rows.sort(key=lambda row: row["task_id"])
    if len(task_ids) != len(tasks):
        raise ValueError("task ID collision")
    body = {
        "version": _BUNDLE_VERSION,
        "dataset_sha256": dataset_sha256,
        "tasks": tasks,
    }
    bundle_sha256 = _sha256(_canonical_json(body))
    blind_bundle = {**body, "bundle_sha256": bundle_sha256}
    answer_key = {
        "version": _BUNDLE_VERSION,
        "dataset_sha256": dataset_sha256,
        "bundle_sha256": bundle_sha256,
        "case_ids": [case["case_id"] for case in checked_cases],
        "task_keys": key_rows,
        "case_metadata": checked_metadata,
    }
    return blind_bundle, answer_key


def _blind_content(
    raw_messages: list[dict[str, Any]], raw_candidate: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session_ids: dict[str, str] = {}
    message_ids: dict[tuple[str, str], str] = {}

    def remap(session_id: str, message_id: str) -> tuple[str, str]:
        if session_id not in session_ids:
            session_ids[session_id] = f"session-{len(session_ids) + 1:03d}"
        identity = (session_id, message_id)
        if identity not in message_ids:
            message_ids[identity] = f"message-{len(message_ids) + 1:03d}"
        return session_ids[session_id], message_ids[identity]

    messages: list[dict[str, Any]] = []
    for row in raw_messages:
        opaque_session, opaque_message = remap(row["session_id"], row["message_id"])
        messages.append(
            {
                "session_id": opaque_session,
                "message_id": opaque_message,
                "role": row["role"],
                "text": row["text"],
                "recorded_at": row["recorded_at"],
                "content_kind": row["content_kind"],
            }
        )
    candidate = copy.deepcopy(raw_candidate)
    for span in candidate["evidence"]:
        span["session_id"], span["message_id"] = remap(span["session_id"], span["message_id"])
    return messages, candidate


def validate_responses(
    blind_bundle: Mapping[str, Any], response_document: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate one reviewer's complete response document against a blind bundle."""
    tasks = _validate_blind_bundle(blind_bundle)
    if type(response_document) is not dict or set(response_document) != {
        "version",
        "bundle_sha256",
        "responses",
    }:
        raise ValueError("response document has missing or unexpected fields")
    if type(response_document["version"]) is not int or response_document["version"] != 1:
        raise ValueError("unsupported response document version")
    response_hash = _require_sha256(response_document["bundle_sha256"], "bundle_sha256")
    if response_hash != blind_bundle["bundle_sha256"]:
        raise ValueError("response bundle hash does not match blind bundle")
    raw_rows = response_document["responses"]
    if type(raw_rows) is not list:
        raise TypeError("responses must be a list")
    task_ids = {task["task_id"] for task in tasks}
    rows: dict[str, dict[str, Any]] = {}
    reviewer_ids: set[str] = set()
    reviewer_kinds: set[str] = set()
    for row in raw_rows:
        if type(row) is not dict or set(row) != _RESPONSE_KEYS:
            raise ValueError("response has missing or unexpected fields")
        task_id = row["task_id"]
        if type(task_id) is not str:
            raise TypeError("response task_id must be text")
        if task_id not in task_ids:
            raise ValueError(f"unknown task ID: {task_id}")
        if task_id in rows:
            raise ValueError(f"duplicate task ID: {task_id}")
        label = row["label"]
        if type(label) is not str or label not in {"supported", "unsupported", "uncertain"}:
            raise ValueError("response label is invalid")
        _require_nonblank_text(row["rationale"], "rationale", 4_000)
        reviewer_id = _require_nonblank_text(row["reviewer_id"], "reviewer_id", 256)
        reviewer_kind = row["reviewer_kind"]
        if type(reviewer_kind) is not str or reviewer_kind not in {"human", "agent"}:
            raise ValueError("reviewer_kind must be human or agent")
        _validate_utc_timestamp(row["timestamp"])
        reviewer_ids.add(reviewer_id)
        reviewer_kinds.add(reviewer_kind)
        rows[task_id] = copy.deepcopy(row)
    if len(reviewer_ids) > 1 or len(reviewer_kinds) > 1:
        raise ValueError("one response document must have a single reviewer ID and kind")
    if set(rows) != task_ids:
        missing = sorted(task_ids.difference(rows))
        raise ValueError(f"missing task ID(s): {', '.join(missing)}")
    return [rows[task["task_id"]] for task in tasks]


def _validate_blind_bundle(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    if type(bundle) is not dict or set(bundle) != {
        "version",
        "dataset_sha256",
        "tasks",
        "bundle_sha256",
    }:
        raise ValueError("blind bundle has missing or unexpected fields")
    if type(bundle["version"]) is not int or bundle["version"] != _BUNDLE_VERSION:
        raise ValueError("unsupported blind bundle version")
    dataset_hash = _require_sha256(bundle["dataset_sha256"], "dataset_sha256")
    expected_hash = _require_sha256(bundle["bundle_sha256"], "bundle_sha256")
    body = {key: bundle[key] for key in ("version", "dataset_sha256", "tasks")}
    if _sha256(_canonical_json(body)) != expected_hash:
        raise ValueError("blind bundle hash mismatch")
    tasks = bundle["tasks"]
    if type(tasks) is not list:
        raise TypeError("blind bundle tasks must be a list")
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for task in tasks:
        if type(task) is not dict or set(task) != _TASK_KEYS:
            raise ValueError("blind task has missing or unexpected fields")
        task_id = task["task_id"]
        if type(task_id) is not str or not task_id.startswith("task_"):
            raise ValueError("blind task ID is invalid")
        if task_id in seen_ids:
            raise ValueError(f"duplicate task ID: {task_id}")
        seen_ids.add(task_id)
        messages = task["messages"]
        if type(messages) is not list or not messages:
            raise TypeError("task messages must be a non-empty list")
        message_pairs: set[tuple[str, str]] = set()
        for message in messages:
            if type(message) is not dict or set(message) != _MESSAGE_KEYS:
                raise ValueError("blind message has missing or unexpected fields")
            session_id = _require_nonblank_text(message["session_id"], "session_id", 256)
            message_id = _require_nonblank_text(message["message_id"], "message_id", 256)
            pair = (session_id, message_id)
            if pair in message_pairs:
                raise ValueError("blind task has duplicate message IDs")
            message_pairs.add(pair)
            try:
                EvidenceRecord(
                    source="review",
                    session_id=session_id,
                    message_id=message_id,
                    part_index=0,
                    role=message["role"],
                    text=message["text"],
                    recorded_at=message["recorded_at"],
                    content_kind=message["content_kind"],
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("blind task has an invalid message") from exc
        candidate = task["candidate"]
        parse_candidates({"claims": [candidate]})
        visible_content = {"messages": messages, "candidate": candidate}
        content_key = _canonical_json(visible_content)
        if content_key in seen_content:
            raise ValueError("duplicate identical candidate task")
        seen_content.add(content_key)
        expected_id = "task_" + _sha256(
            _canonical_json({"dataset_sha256": dataset_hash, "task": visible_content})
        )
        if task_id != expected_id:
            raise ValueError("blind task ID does not match task content")
    return tasks


def _require_nonblank_text(value: object, name: str, max_chars: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > max_chars:
        raise ValueError(f"{name} must be non-blank text of at most {max_chars} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} contains invalid Unicode") from None
    return value


def _validate_utc_timestamp(value: object) -> None:
    if (
        type(value) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None
    ):
        raise ValueError("timestamp must use canonical UTC format YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError("timestamp is not a valid UTC datetime") from None


def _validate_case_metadata(row: object) -> None:
    if type(row) is not dict or set(row) != _METADATA_KEYS:
        raise ValueError("case metadata has missing or unexpected fields")
    for field in ("scenario_group", "template_group"):
        if type(row[field]) is not str or not row[field].strip():
            raise ValueError(f"case metadata {field} must be non-blank text")
    entities = row["entity_ids"]
    if (
        type(entities) is not list
        or not entities
        or any(type(entity) is not str or not entity.strip() for entity in entities)
    ):
        raise ValueError("case metadata entity_ids must be non-empty text values")
    if len(set(entities)) != len(entities):
        raise ValueError("case metadata entity_ids must be unique")


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
