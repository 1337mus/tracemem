"""Read-only adapters for supported conversational transcript formats."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import EvidenceRecord


@dataclass(slots=True)
class AdapterStats:
    """Counts adapter outcomes without retaining transcript contents."""

    malformed_records: int = 0
    unsupported_records: int = 0
    role_counts: dict[str, int] = field(default_factory=dict)

    @property
    def malformed(self) -> int:
        return self.malformed_records

    @property
    def unsupported(self) -> int:
        return self.unsupported_records


def read_transcript(
    path: str | os.PathLike[str],
    provider: str,
    *,
    stats: AdapterStats | None = None,
) -> Iterator[EvidenceRecord]:
    """Yield recognized transcript blocks without changing or normalizing their text."""
    if provider not in {"claude", "codex"}:
        raise ValueError("provider must be 'claude' or 'codex'")
    if stats is not None and not isinstance(stats, AdapterStats):
        raise TypeError("stats must be an AdapterStats instance")

    counters = stats if stats is not None else AdapterStats()
    transcript_path = Path(path).expanduser()
    try:
        locator = hashlib.sha256(
            str(transcript_path.resolve(strict=False)).encode("utf-8")
        ).hexdigest()
    except (OSError, UnicodeError):
        raise ValueError("cannot identify transcript source") from None
    default_session_id = locator
    session_id = default_session_id

    try:
        with transcript_path.open("r", encoding="utf-8") as transcript:
            for line_number, line in enumerate(transcript, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    counters.malformed_records += 1
                    continue
                if not isinstance(event, dict):
                    counters.malformed_records += 1
                    continue

                if provider == "claude":
                    records = _claude_records(
                        event,
                        session_id=default_session_id,
                        locator=locator,
                        line_number=line_number,
                        stats=counters,
                    )
                else:
                    payload = event.get("payload")
                    payload_session = (
                        payload.get("session_id") if isinstance(payload, dict) else None
                    )
                    session_id = _native_string(
                        event.get("session_id"), payload_session, session_id
                    )
                    records, session_id = _codex_records(
                        event,
                        session_id=session_id,
                        locator=locator,
                        line_number=line_number,
                        stats=counters,
                    )
                yield from records
    except OSError:
        raise ValueError("cannot read transcript source") from None
    except UnicodeDecodeError:
        raise ValueError("transcript source is not valid UTF-8") from None


def _claude_records(
    event: dict[str, object],
    *,
    session_id: str,
    locator: str,
    line_number: int,
    stats: AdapterStats,
) -> list[EvidenceRecord]:
    event_type = event.get("type")
    if event_type not in {"user", "assistant", "system", "developer", "tool"}:
        stats.unsupported_records += 1
        return []
    message = event.get("message")
    if not isinstance(message, dict):
        stats.malformed_records += 1
        return []
    content = message.get("content")
    if isinstance(content, str):
        content_blocks: list[object] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content_blocks = content
    else:
        stats.malformed_records += 1
        return []

    role = _role(message.get("role"), str(event_type))
    message_id = _native_string(
        message.get("id"),
        message.get("uuid"),
        event.get("uuid"),
        event.get("messageId"),
        f"line-{line_number}",
    )
    source_session_id = _native_string(
        event.get("sessionId"), event.get("session_id"), message.get("session_id"), session_id
    )
    recorded_at = _optional_string(event.get("timestamp"))
    records: list[EvidenceRecord] = []
    for block_index, block in enumerate(content_blocks):
        if not isinstance(block, dict):
            stats.malformed_records += 1
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            records.append(
                _record(
                    source="claude",
                    session_id=source_session_id,
                    message_id=message_id,
                    part_index=_part_index(block_index),
                    role=role,
                    text=block["text"],
                    recorded_at=recorded_at,
                    locator=locator,
                    line_number=line_number,
                    stats=stats,
                )
            )
        elif block_type == "tool_result":
            output_parts = _text_parts(block.get("content"))
            if not output_parts:
                stats.unsupported_records += 1
            for nested_path, text in output_parts:
                records.append(
                    _record(
                        source="claude",
                        session_id=source_session_id,
                        message_id=message_id,
                        part_index=_part_index(block_index, nested_path),
                        role="tool",
                        text=text,
                        recorded_at=recorded_at,
                        locator=locator,
                        line_number=line_number,
                        stats=stats,
                        content_kind="tool_output",
                    )
                )
        else:
            stats.unsupported_records += 1
    return records


def _codex_records(
    event: dict[str, object],
    *,
    session_id: str,
    locator: str,
    line_number: int,
    stats: AdapterStats,
) -> tuple[list[EvidenceRecord], str]:
    event_type = event.get("type")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        stats.malformed_records += 1
        return [], session_id

    if event_type == "session_meta":
        native_session = _native_string(payload.get("id"), payload.get("session_id"))
        return [], native_session or session_id
    if event_type == "response_item" and payload.get("type") == "message":
        role = _role(payload.get("role"), "unknown")
        content = payload.get("content")
        if isinstance(content, str):
            blocks: list[object] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = content
        else:
            stats.malformed_records += 1
            return [], session_id
        message_id = _native_string(
            payload.get("id"),
            payload.get("message_id"),
            event.get("id"),
            f"line-{line_number}",
        )
        recorded_at = _optional_string(event.get("timestamp"))
        records: list[EvidenceRecord] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                stats.malformed_records += 1
                continue
            block_type = block.get("type")
            if block_type in {"input_text", "output_text", "text"} and isinstance(
                block.get("text"), str
            ):
                records.append(
                    _record(
                        source="codex",
                        session_id=session_id,
                        message_id=message_id,
                        part_index=_part_index(block_index),
                        role=role,
                        text=block["text"],
                        recorded_at=recorded_at,
                        locator=locator,
                        line_number=line_number,
                        stats=stats,
                    )
                )
            else:
                stats.unsupported_records += 1
        return records, session_id
    if event_type == "response_item" and payload.get("type") in {
        "function_call_output",
        "custom_tool_call_output",
    }:
        output = payload.get("output")
        output_text = _output_text(output)
        if output_text is None:
            stats.unsupported_records += 1
            return [], session_id
        message_id = _native_string(
            payload.get("call_id"), payload.get("id"), event.get("id"), f"line-{line_number}"
        )
        record = _record(
            source="codex",
            session_id=session_id,
            message_id=message_id,
            part_index=0,
            role="tool",
            text=output_text,
            recorded_at=_optional_string(event.get("timestamp")),
            locator=locator,
            line_number=line_number,
            stats=stats,
            content_kind="tool_output",
        )
        return [record], session_id
    stats.unsupported_records += 1
    return [], session_id


def _record(
    *,
    source: str,
    session_id: str,
    message_id: str,
    part_index: int,
    role: str,
    text: str,
    recorded_at: str | None,
    locator: str,
    line_number: int,
    stats: AdapterStats,
    content_kind: str | None = None,
) -> EvidenceRecord:
    if content_kind is None:
        content_kind = {
            "user": "mixed_untrusted",
            "system": "instruction",
            "developer": "instruction",
            "unknown": "mixed_untrusted",
        }.get(role, "conversation")
    record = EvidenceRecord(
        source=source,
        session_id=session_id,
        message_id=message_id,
        part_index=part_index,
        role=role,
        text=text,
        recorded_at=recorded_at,
        source_locator=locator,
        line_number=line_number,
        content_kind=content_kind,
    )
    stats.role_counts[role] = stats.role_counts.get(role, 0) + 1
    return record


def _role(value: object, fallback: str) -> str:
    if isinstance(value, str) and value in EvidenceRecord.ROLES:
        if fallback in EvidenceRecord.ROLES and fallback != "unknown" and value != fallback:
            return "unknown"
        return value
    if value is None or value == "":
        return fallback if fallback in EvidenceRecord.ROLES else "unknown"
    return "unknown"


def _native_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _text_parts(value: object, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return [(path, value["text"])]
        return []
    if isinstance(value, list):
        parts: list[tuple[tuple[int, ...], str]] = []
        for index, item in enumerate(value):
            parts.extend(_text_parts(item, path + (index,)))
        return parts
    return []


def _part_index(block_index: int, nested_path: tuple[int, ...] | None = None) -> int:
    nested_index = 0 if nested_path is None else _path_index(nested_path) + 1
    return _pair_indices(block_index, nested_index)


def _path_index(path: tuple[int, ...]) -> int:
    encoded = 0
    for index in reversed(path):
        encoded = _pair_indices(index + 1, encoded + 1)
    return encoded


def _pair_indices(left: int, right: int) -> int:
    total = left + right
    return total * (total + 1) // 2 + right


def _output_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def discover_context_kit(
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    from .context_kit import discover_context_kit as discover

    return discover(config_path)


def __getattr__(name: str) -> object:
    if name == "ContextKitConfigError":
        from .context_kit import ContextKitConfigError

        return ContextKitConfigError
    raise AttributeError(name)


__all__ = [
    "AdapterStats",
    "ContextKitConfigError",
    "discover_context_kit",
    "read_transcript",
]
