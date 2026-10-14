"""Explicit, read-only adapter for documented ChatGPT JSON exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from ..schema import EvidenceRecord
from . import AdapterStats

_MAX_EXPORT_BYTES = 64 * 1024 * 1024


class _InvalidJson(ValueError):
    """Private signal for JSON extensions rejected by this adapter."""


def read_chatgpt_export(
    path: str | os.PathLike[str],
    *,
    stats: AdapterStats | None = None,
) -> Iterator[EvidenceRecord]:
    """Yield text from each conversation's selected current-node parent chain."""
    if stats is not None and not isinstance(stats, AdapterStats):
        raise TypeError("stats must be an AdapterStats instance")
    counters = stats if stats is not None else AdapterStats()
    try:
        export_path = Path(path).expanduser()
        locator = hashlib.sha256(str(export_path.resolve(strict=False)).encode("utf-8")).hexdigest()
        with export_path.open("rb") as source_file:
            source_bytes = source_file.read(_MAX_EXPORT_BYTES + 1)
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        raise ValueError("cannot read ChatGPT export") from None
    if len(source_bytes) > _MAX_EXPORT_BYTES:
        raise ValueError("ChatGPT export exceeds 64 MiB limit")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("ChatGPT export is not valid UTF-8") from None
    try:
        conversations: object = json.loads(
            source_text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError):
        raise ValueError("ChatGPT export is not valid JSON") from None
    if not isinstance(conversations, list):
        raise ValueError("ChatGPT export root must be a list")
    seen_conversation_ids: set[str] = set()
    for conversation in conversations:
        if not isinstance(conversation, dict):
            continue
        conversation_id = conversation.get("id")
        if isinstance(conversation_id, str) and conversation_id.strip():
            if conversation_id in seen_conversation_ids:
                raise ValueError("duplicate conversation identifier in ChatGPT export")
            seen_conversation_ids.add(conversation_id)

    for conversation in conversations:
        parsed = _conversation_records(conversation, locator)
        if parsed is None:
            counters.malformed_records += 1
            continue
        records, unsupported_count = parsed
        counters.unsupported_records += unsupported_count
        for record in records:
            counters.role_counts[record.role] = counters.role_counts.get(record.role, 0) + 1
            yield record


__all__ = ["read_chatgpt_export"]


def _conversation_records(
    conversation: object,
    locator: str,
) -> tuple[list[EvidenceRecord], int] | None:
    if not isinstance(conversation, dict):
        return None
    conversation_id = conversation.get("id")
    current_node = conversation.get("current_node")
    mapping = conversation.get("mapping")
    if (
        not isinstance(conversation_id, str)
        or not conversation_id.strip()
        or not _is_utf8_encodable(conversation_id)
        or not isinstance(current_node, str)
        or not current_node.strip()
        or not isinstance(mapping, dict)
    ):
        return None

    chain: list[tuple[str, dict[str, object]]] = []
    seen_nodes: set[str] = set()
    while True:
        if current_node in seen_nodes or current_node not in mapping:
            return None
        seen_nodes.add(current_node)
        node = mapping[current_node]
        if not isinstance(node, dict) or "parent" not in node or "message" not in node:
            return None
        parent = node["parent"]
        if parent is not None and not isinstance(parent, str):
            return None
        message = node["message"]
        if message is not None and not isinstance(message, dict):
            return None
        chain.append((current_node, node))
        if parent is None:
            break
        current_node = parent

    message_ids: set[str] = set()
    for node_id, node in chain:
        message = node["message"]
        if message is None:
            continue
        if not isinstance(message, dict):
            return None
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id.strip():
            message_id = node_id
        if (
            not message_id.strip()
            or not _is_utf8_encodable(message_id)
            or message_id in message_ids
        ):
            return None
        message_ids.add(message_id)

    records: list[EvidenceRecord] = []
    unsupported_count = 0
    for node_id, node in reversed(chain):
        message = node["message"]
        if message is None:
            unsupported_count += 1
            continue
        if not isinstance(message, dict):
            return None
        author = message.get("author")
        content = message.get("content")
        if not isinstance(author, dict) or not isinstance(content, dict):
            return None
        raw_role = author.get("role")
        if not isinstance(raw_role, str):
            return None
        role = raw_role if raw_role in EvidenceRecord.ROLES else "unknown"
        content_type = content.get("content_type")
        if not isinstance(content_type, str):
            return None
        if content_type != "text":
            unsupported_count += 1
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            return None
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id.strip():
            message_id = node_id
        if not message_id.strip():
            return None
        content_kind = {
            "user": "mixed_untrusted",
            "unknown": "mixed_untrusted",
            "tool": "tool_output",
            "system": "instruction",
            "developer": "instruction",
        }.get(role, "conversation")
        for part_index, text in enumerate(parts):
            if not isinstance(text, str):
                unsupported_count += 1
                continue
            if not _is_utf8_encodable(text):
                return None
            records.append(
                EvidenceRecord(
                    source="chatgpt",
                    session_id=conversation_id,
                    message_id=message_id,
                    part_index=part_index,
                    role=role,
                    text=text,
                    recorded_at=_timestamp(message.get("create_time")),
                    source_locator=locator,
                    line_number=None,
                    content_kind=content_kind,
                )
            )
    return records, unsupported_count


def _timestamp(value: object) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        if not math.isfinite(value):
            return None
        return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJson
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _InvalidJson


def _is_utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
