"""Validation and local-only export for private verifier annotations."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from typing import Any

from tracemem.schema import EvidenceRecord
from training.verifier_data import _render_validated_candidate_prompt

_ROOT_KEYS = set("version license annotation_status provenance scope_policy cases".split())
_CASE_KEYS = set("case_id session_id split window_id privacy_clearance messages candidates".split())
_CLEARANCE_KEYS = {"status", "reviewer_id", "reviewer_type", "reviewed_at"}
_MESSAGE_KEYS = set(
    "message_id source_record_id revision session_id text text_sha256 source_locator "
    "line_number role recorded_at content_kind".split()
)
_CANDIDATE_KEYS = set(
    "candidate_id claim message_id source_record_id quote char_start char_end valid_from "
    "valid_to support eligibility support_reason eligibility_reason label_status annotator "
    "reviews".split()
)
_ANNOTATOR_KEYS = {"reviewer_id", "reviewer_type"}
_REVIEW_KEYS = set(
    "reviewer_id reviewer_type status reviewed_at support eligibility support_reason "
    "eligibility_reason".split()
)
_SPLITS = {"train", "dev", "test"}
_SUPPORT = {"supported", "unsupported", "uncertain"}
_ELIGIBILITY = {"eligible", "ineligible", "uncertain"}
_LABEL_STATUSES = {"agent_provisional", "accepted", "pending", "rejected"}
_REVIEWER_TYPES = {"agent", "human"}
_SOURCE_RECORD_KEYS = set(
    "record_id revision session_id message_id text text_sha256 source_locator line_number "
    "role recorded_at content_kind".split()
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_private_verifier_dataset(
    payload: object, source_records: object
) -> list[dict[str, Any]]:
    """Validate private annotations against local source records and return their cases."""
    root = _mapping(payload, _ROOT_KEYS, "dataset root")
    if type(root["version"]) is not int or root["version"] != 1:
        raise ValueError("version must be integer 1")
    for field, expected in (
        ("license", "private-not-for-redistribution"),
        ("annotation_status", "agent_provisional"),
        ("provenance", "private_source"),
        ("scope_policy", "personal-memory-research-v1"),
    ):
        if type(root[field]) is not str or root[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    if type(root["cases"]) is not list or not root["cases"]:
        raise ValueError("cases must be a non-empty list")
    sources = _source_index(source_records)
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    window_ids: set[str] = set()
    candidate_ids: set[str] = set()
    session_splits: dict[str, str] = {}
    long_text_splits: dict[str, str] = {}

    for raw_case in root["cases"]:
        case = _mapping(raw_case, _CASE_KEYS, "case")
        case_id = _text(case["case_id"], "case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        session_id = _text(case["session_id"], "session_id")
        split = _text(case["split"], "split")
        if split not in _SPLITS:
            raise ValueError(f"case {case_id} has invalid split")
        previous_split = session_splits.setdefault(session_id, split)
        if previous_split != split:
            raise ValueError(f"source session {session_id} overlaps splits")
        window_id = _text(case["window_id"], "window_id")
        if window_id in window_ids:
            raise ValueError(f"duplicate window_id: {window_id}")
        window_ids.add(window_id)
        clearance = _validate_clearance(case["privacy_clearance"])
        if type(case["messages"]) is not list or not case["messages"]:
            raise ValueError(f"case {case_id} messages must be non-empty")
        messages: dict[str, dict[str, Any]] = {}
        for raw_message in case["messages"]:
            message = _validate_message(raw_message, session_id, sources)
            normalized = " ".join(unicodedata.normalize("NFKC", message["text"]).casefold().split())
            if len(normalized) >= 160:
                prior_split = long_text_splits.setdefault(normalized, split)
                if prior_split != split:
                    raise ValueError("long message overlap across splits")
            message_id = message["message_id"]
            if message_id in messages:
                raise ValueError(f"case {case_id} has duplicate message_id")
            messages[message_id] = message
        if type(case["candidates"]) is not list or not case["candidates"]:
            raise ValueError(f"case {case_id} candidates must be non-empty")
        for raw_candidate in case["candidates"]:
            candidate = _mapping(raw_candidate, _CANDIDATE_KEYS, "candidate")
            candidate_id = _text(candidate["candidate_id"], "candidate_id")
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            candidate_ids.add(candidate_id)
            _validate_candidate(candidate, messages, sources)
        cases.append({**case, "privacy_clearance": clearance})
    return cases


def render_private_training_rows(payload: object, source_records: object) -> list[dict[str, str]]:
    """Return private supervised rows for cleared, exportable train annotations only."""
    cases = validate_private_verifier_dataset(payload, source_records)
    rows: list[dict[str, str]] = []
    for case in cases:
        if case["split"] != "train" or case["privacy_clearance"]["status"] != "cleared":
            continue
        prompt_case = _prompt_case(case)
        for candidate in case["candidates"]:
            if not _candidate_exportable(candidate):
                continue
            prompt = _render_validated_candidate_prompt(prompt_case, candidate)
            completion = json.dumps(
                {"support": candidate["support"], "eligibility": candidate["eligibility"]},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            rows.append({"prompt": prompt, "completion": completion})
    return rows


def render_private_eval_prompts(payload: object, source_records: object, split: str) -> list[str]:
    """Return label-free prompts for cleared dev or test candidates only."""
    if type(split) is not str or split not in {"dev", "test"}:
        raise ValueError("private evaluation split must be dev or test")
    cases = validate_private_verifier_dataset(payload, source_records)
    prompts: list[str] = []
    for case in cases:
        if case["split"] != split or case["privacy_clearance"]["status"] != "cleared":
            continue
        prompt_case = _prompt_case(case)
        for candidate in case["candidates"]:
            if _candidate_exportable(candidate):
                prompts.append(_render_validated_candidate_prompt(prompt_case, candidate))
    return prompts


def _prompt_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {
                key: message[key]
                for key in ("message_id", "role", "text", "recorded_at", "content_kind")
            }
            for message in case["messages"]
        ]
    }


def _candidate_exportable(candidate: dict[str, Any]) -> bool:
    if candidate["label_status"] not in {"agent_provisional", "accepted"}:
        return False
    return all(
        review["status"] == "accepted"
        and review["support"] == candidate["support"]
        and review["eligibility"] == candidate["eligibility"]
        for review in candidate["reviews"]
    )


def _source_index(value: object) -> dict[str, dict[str, Any]]:
    if type(value) is not dict:
        raise ValueError("source_records must be a local record mapping")
    sources: dict[str, dict[str, Any]] = {}
    for key, raw_record in value.items():
        record = _mapping_required(raw_record, _SOURCE_RECORD_KEYS, "source record")
        record_id = _text(record["record_id"], "source record_id")
        if key != record_id:
            raise ValueError("source record mapping key must match record_id")
        text = _text(record["text"], "source record text")
        digest = _text(record["text_sha256"], "source text_sha256")
        if (
            _SHA256.fullmatch(digest) is None
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != digest
        ):
            raise ValueError("source text hash does not match source record")
        _text(record["revision"], "source revision")
        _text(record["session_id"], "source session_id")
        _text(record["message_id"], "source message_id")
        _text(record["source_locator"], "source_locator")
        if type(record["line_number"]) is not int or record["line_number"] < 1:
            raise ValueError("source line_number must be a positive integer")
        role = _text(record["role"], "source role")
        _validate_datetime(record["recorded_at"], "source recorded_at", allow_none=True)
        content_kind = _text(record["content_kind"], "source content_kind")
        if role not in EvidenceRecord.ROLES or content_kind not in EvidenceRecord.CONTENT_KINDS:
            raise ValueError("source role or content_kind is unsupported")
        if role == "user" and content_kind == "tool_output":
            raise ValueError("user source cannot be tool_output")
        sources[record_id] = record
    return sources


def _validate_message(
    value: object, session_id: str, sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    message = _mapping(value, _MESSAGE_KEYS, "message")
    for field in ("message_id", "source_record_id", "revision", "session_id", "text"):
        _text(message[field], f"message {field}")
    if message["session_id"] != session_id:
        raise ValueError("message session_id must match case session_id")
    record_id = message["source_record_id"]
    source = sources.get(record_id)
    if source is None:
        raise ValueError(f"message references unknown source record: {record_id}")
    for field, source_field in (
        ("message_id", "message_id"),
        ("revision", "revision"),
        ("session_id", "session_id"),
        ("text", "text"),
        ("text_sha256", "text_sha256"),
        ("source_locator", "source_locator"),
        ("line_number", "line_number"),
        ("role", "role"),
        ("recorded_at", "recorded_at"),
        ("content_kind", "content_kind"),
    ):
        if message[field] != source[source_field]:
            raise ValueError(f"message {field} does not match local source record")
    return message


def _validate_candidate(
    candidate: dict[str, Any],
    messages: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    for field in ("claim", "message_id", "source_record_id", "quote"):
        _text(candidate[field], f"candidate {field}")
    message = messages.get(candidate["message_id"])
    source = sources.get(candidate["source_record_id"])
    if (
        message is None
        or source is None
        or message["source_record_id"] != candidate["source_record_id"]
    ):
        raise ValueError("candidate citation must match a message and local source record")
    start, end = candidate["char_start"], candidate["char_end"]
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end > len(source["text"])
    ):
        raise ValueError("candidate offsets must end within source text")
    quote = candidate["quote"]
    if source["text"][start:end] != quote or message["text"][start:end] != quote:
        raise ValueError("candidate quote does not match exact source offsets")
    valid_from = _date(candidate["valid_from"], "valid_from")
    valid_to = _date(candidate["valid_to"], "valid_to")
    if valid_from and valid_to and valid_from > valid_to:
        raise ValueError("valid_from must not be after valid_to")
    if valid_from and valid_from.isoformat() not in quote:
        raise ValueError("valid_from must appear literally in quote")
    if valid_to and valid_to.isoformat() not in quote:
        raise ValueError("valid_to must appear literally in quote")
    support = _text(candidate["support"], "candidate support")
    eligibility = _text(candidate["eligibility"], "candidate eligibility")
    if support not in _SUPPORT or eligibility not in _ELIGIBILITY:
        raise ValueError("candidate labels must use the independent verifier label vocabularies")
    for field in ("support_reason", "eligibility_reason"):
        reason = _text(candidate[field], field)
        if len(reason) > 500:
            raise ValueError(f"{field} must be at most 500 characters")
    if _text(candidate["label_status"], "candidate label_status") not in _LABEL_STATUSES:
        raise ValueError("candidate has invalid label_status")
    _validate_annotator(candidate["annotator"])
    _validate_reviews(candidate["reviews"], candidate)


def _validate_clearance(value: object) -> dict[str, Any]:
    clearance = _mapping(value, _CLEARANCE_KEYS, "privacy_clearance")
    status = _text(clearance["status"], "privacy clearance status")
    if status not in {"cleared", "pending", "rejected"}:
        raise ValueError("privacy clearance status is invalid")
    _text(clearance["reviewer_id"], "privacy reviewer_id")
    if _text(clearance["reviewer_type"], "privacy reviewer_type") not in _REVIEWER_TYPES:
        raise ValueError("privacy reviewer_type must be agent or human")
    if status == "pending":
        if clearance["reviewed_at"] is not None:
            _validate_datetime(clearance["reviewed_at"], "privacy reviewed_at")
    else:
        _validate_datetime(clearance["reviewed_at"], "privacy reviewed_at")
    return clearance


def _validate_annotator(value: object) -> dict[str, Any]:
    annotator = _mapping(value, _ANNOTATOR_KEYS, "annotator")
    _text(annotator["reviewer_id"], "annotator reviewer_id")
    if _text(annotator["reviewer_type"], "annotator reviewer_type") not in _REVIEWER_TYPES:
        raise ValueError("annotator reviewer_type must be agent or human")
    return annotator


def _validate_reviews(value: object, candidate: dict[str, Any]) -> None:
    if type(value) is not list:
        raise ValueError("reviews must be a list")
    reviewer_ids: set[str] = set()
    accepted_independent = False
    accepted_mismatch = False
    for raw_review in value:
        review = _mapping(raw_review, _REVIEW_KEYS, "review")
        reviewer_id = _text(review["reviewer_id"], "reviewer_id")
        if reviewer_id in reviewer_ids:
            raise ValueError("reviewer_id must be unique per candidate")
        reviewer_ids.add(reviewer_id)
        if _text(review["reviewer_type"], "reviewer_type") not in _REVIEWER_TYPES:
            raise ValueError("reviewer_type must be agent or human")
        status = _text(review["status"], "review status")
        if status not in {"accepted", "pending", "rejected"}:
            raise ValueError("review status is invalid")
        if status == "pending":
            if review["reviewed_at"] is not None:
                _validate_datetime(review["reviewed_at"], "reviewed_at")
            if any(
                review[field] is not None
                for field in ("support", "eligibility", "support_reason", "eligibility_reason")
            ):
                raise ValueError("pending reviews cannot carry labels or rationales")
            continue
        _validate_datetime(review["reviewed_at"], "reviewed_at")
        for field, labels in (("support", _SUPPORT), ("eligibility", _ELIGIBILITY)):
            if type(review[field]) is not str or review[field] not in labels:
                raise ValueError(f"completed review requires a valid {field} label")
        for field in ("support_reason", "eligibility_reason"):
            reason = _text(review[field], field)
            if len(reason) > 500:
                raise ValueError(f"{field} must be at most 500 characters")
        if status == "accepted":
            matches = all(review[field] == candidate[field] for field in ("support", "eligibility"))
            if not matches and candidate["label_status"] == "accepted":
                raise ValueError("accepted review labels must match final labels")
            if reviewer_id != candidate["annotator"]["reviewer_id"]:
                accepted_independent = True
                accepted_mismatch = accepted_mismatch or not matches
    if candidate["label_status"] == "accepted":
        if not accepted_independent:
            raise ValueError("accepted labels require an independent reviewer")
        if accepted_mismatch or any(review["status"] != "accepted" for review in value):
            raise ValueError("accepted labels require matching completed reviews")


def _mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has missing or unexpected fields")
    return value


def _mapping_required(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or not keys.issubset(value):
        raise ValueError(f"{label} is missing required fields")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _date(value: object, label: str) -> date | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{label} must be an ISO date or null")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date or null") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must use YYYY-MM-DD")
    return parsed


def _validate_datetime(value: object, label: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if type(value) is not str:
        raise ValueError(f"{label} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
