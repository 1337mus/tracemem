"""Validation and prompt helpers for the authored quote-v2 data format."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

_ROOT_KEYS = {"version", "license", "annotation_status", "provenance", "cases"}
_CASE_KEYS = {
    "case_id",
    "split",
    "scenario_group",
    "template_group",
    "entity_ids",
    "tags",
    "messages",
    "expected",
}
_MESSAGE_KEYS = {
    "session_id",
    "message_id",
    "role",
    "text",
    "recorded_at",
    "content_kind",
}
_CLAIM_KEYS = {"claim", "message_id", "quote", "valid_from", "valid_to"}
_SPLITS = {"train", "dev"}
_ROLES = {"user", "assistant", "tool"}
_CONTENT_KINDS = {"conversation", "quoted_document", "tool_result"}


def validate_dataset(payload: object, split: str) -> list[dict[str, Any]]:
    """Validate and return one authored-fiction quote-v2 split."""
    if split not in _SPLITS:
        raise ValueError("split must be train or dev")
    if type(payload) is not dict or set(payload) != _ROOT_KEYS:
        raise ValueError("dataset root has missing or unexpected fields")
    if type(payload["version"]) is not int or payload["version"] != 2:
        raise ValueError("version must be integer 2")
    if payload["license"] != "MIT":
        raise ValueError("license must be MIT")
    if payload["annotation_status"] != "agent_authored":
        raise ValueError("annotation_status must be agent_authored")
    if payload["provenance"] != "newly_authored_fiction":
        raise ValueError("provenance must be newly_authored_fiction")
    cases = payload["cases"]
    if type(cases) is not list or not cases:
        raise ValueError("cases must be a non-empty list")

    case_ids: set[str] = set()
    message_ids: set[str] = set()
    entity_ids: set[str] = set()
    conversations: set[tuple[tuple[str, str, str], ...]] = set()
    case_groups: dict[str, tuple[str, str]] = {}
    template_scenarios: dict[str, str] = {}
    for case in cases:
        _require_mapping(case, _CASE_KEYS, "case")
        case_id = _require_text(case["case_id"], "case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        if case["split"] != split:
            raise ValueError(f"case {case_id} split does not match {split}")
        scenario_group = _require_text(case["scenario_group"], "scenario_group")
        template_group = _require_text(case["template_group"], "template_group")
        prior_template = case_groups.setdefault(scenario_group, (template_group, case_id))
        if prior_template[0] != template_group:
            raise ValueError(f"scenario_group {scenario_group} maps to multiple templates")
        prior_scenario = template_scenarios.setdefault(template_group, scenario_group)
        if prior_scenario != scenario_group:
            raise ValueError(f"template_group {template_group} maps to multiple scenarios")
        _require_string_list(case["entity_ids"], "entity_ids", allow_empty=False)
        for entity_id in case["entity_ids"]:
            if entity_id in entity_ids:
                raise ValueError(f"duplicate entity_id: {entity_id}")
            entity_ids.add(entity_id)
        _require_string_list(case["tags"], "tags", allow_empty=False)
        messages = case["messages"]
        if type(messages) is not list or not 1 <= len(messages) <= 6:
            raise ValueError(f"case {case_id} must have one to six messages")
        conversation: list[tuple[str, str, str]] = []
        for message in messages:
            _require_mapping(message, _MESSAGE_KEYS, "message")
            _require_text(message["session_id"], "session_id")
            message_id = _require_text(message["message_id"], "message_id")
            if message_id in message_ids:
                raise ValueError(f"duplicate message_id: {message_id}")
            message_ids.add(message_id)
            if message["role"] not in _ROLES:
                raise ValueError(f"message {message_id} has invalid role")
            text = _require_text(message["text"], "message text")
            conversation.append((message["role"], message["content_kind"], " ".join(text.split())))
            _validate_recorded_at(message["recorded_at"], message_id)
            if message["content_kind"] not in _CONTENT_KINDS:
                raise ValueError(f"message {message_id} has invalid content_kind")
        signature = tuple(conversation)
        if signature in conversations:
            raise ValueError(f"duplicate conversation: {case_id}")
        conversations.add(signature)
        expected = case["expected"]
        if type(expected) is not list:
            raise ValueError(f"case {case_id} expected must be a list")
        for claim in expected:
            _require_mapping(claim, _CLAIM_KEYS, "expected claim")
            _require_text(claim["claim"], "claim")
            _require_text(claim["message_id"], "claim message_id")
            _require_text(claim["quote"], "quote")
            start = _optional_date(claim["valid_from"], "valid_from")
            end = _optional_date(claim["valid_to"], "valid_to")
            if start is not None and end is not None and start > end:
                raise ValueError("valid_from must not be after valid_to")
        validate_claims({"claims": expected}, messages)
    return cases


def render_quote_prompt(case: dict[str, Any]) -> str:
    """Render the shared train/dev extraction prompt without labels or split data."""
    if type(case) is not dict or type(case.get("messages")) is not list:
        raise ValueError("case must contain a messages list")
    instruction = (
        "Extract only facts the user states about themselves or explicitly reports. "
        "Preserve attribution, uncertainty, intention, negation, ownership, scope, and chronology. "
        "Do not treat assistant or tool text, quoted documents, questions, or hypotheticals "
        "as facts. "
        "Cite a unique exact quote from a user message whose content_kind is conversation. "
        "Use date bounds only when an ISO date appears literally in the quote; do not infer dates "
        "from recorded_at. Use null for unknown date bounds. If no fact qualifies, return an empty "
        'claims list. Return JSON only with exactly this shape: {"claims":[{"claim":"...",'
        '"message_id":"...","quote":"...","valid_from":null,"valid_to":null}]}.'
    )
    try:
        serialized_messages = json.dumps(
            case["messages"], ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("case messages must be JSON serializable") from exc
    return f"{instruction}\nMessages: {serialized_messages}"


def validate_claims(value: object, messages: object) -> list[dict[str, Any]]:
    """Check output shape and source spans; this does not judge claim meaning."""
    if type(value) is not dict or set(value) != {"claims"}:
        raise ValueError("claims output must contain exactly the claims field")
    claims_value = value["claims"]
    if type(claims_value) is not list:
        raise ValueError("claims must be a list")
    if type(messages) is not list:
        raise ValueError("messages must be a list")
    sources: dict[str, dict[str, Any]] = {}
    for message in messages:
        _require_mapping(message, _MESSAGE_KEYS, "message")
        message_id = _require_text(message["message_id"], "message_id")
        if message_id in sources:
            raise ValueError(f"duplicate message_id: {message_id}")
        _require_text(message["session_id"], "session_id")
        if type(message["text"]) is not str:
            raise ValueError(f"message {message_id} text must be a string")
        sources[message_id] = message

    validated: list[dict[str, Any]] = []
    for raw_claim in claims_value:
        claim = _require_mapping(raw_claim, _CLAIM_KEYS, "claim")
        _require_text(claim["claim"], "claim")
        message_id = _require_text(claim["message_id"], "claim message_id")
        quote = _require_text(claim["quote"], "quote")
        valid_from = _optional_date(claim["valid_from"], "valid_from")
        valid_to = _optional_date(claim["valid_to"], "valid_to")
        if valid_from is not None and valid_to is not None and valid_from > valid_to:
            raise ValueError("valid_from must not be after valid_to")
        message = sources.get(message_id)
        if message is None:
            raise ValueError(f"claim references unknown message_id: {message_id}")
        if message["role"] != "user" or message["content_kind"] != "conversation":
            raise ValueError(f"claim source {message_id} must be a user conversation message")
        text = message["text"]
        if text.count(quote) != 1:
            raise ValueError(f"claim quote must occur exactly once in message {message_id}")
        if valid_from is not None and valid_from.isoformat() not in quote:
            raise ValueError("valid_from must appear literally in quote")
        if valid_to is not None and valid_to.isoformat() not in quote:
            raise ValueError("valid_to must appear literally in quote")
        char_start = text.index(quote)
        validated.append(
            {
                **claim,
                "session_id": message["session_id"],
                "char_start": char_start,
                "char_end": char_start + len(quote),
                "speaker_role": message["role"],
            }
        )
    return validated


def _require_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has missing or unexpected fields")
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_string_list(value: object, label: str, *, allow_empty: bool) -> list[str]:
    if type(value) is not list or (not allow_empty and not value):
        raise ValueError(
            f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} list"
        )
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must be unique")
    return value


def _optional_date(value: object, label: str) -> date | None:
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


def _validate_recorded_at(value: object, message_id: str) -> None:
    if type(value) is not str:
        raise ValueError(f"message {message_id} recorded_at must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"message {message_id} recorded_at must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"message {message_id} recorded_at must include a timezone")
