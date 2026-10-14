"""Shared immutable evidence model and database identity constants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

EVIDENCE_SCHEMA_VERSION = 3
EVIDENCE_APPLICATION_ID = 0x54524D45  # ASCII "TRME"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One exact transcript block with a stable identity and content revision."""

    source: str
    session_id: str
    message_id: str
    part_index: int
    role: str
    text: str
    recorded_at: str | None = None
    source_locator: str | None = None
    line_number: int | None = None
    content_kind: str = "mixed_untrusted"

    ROLES: ClassVar[frozenset[str]] = frozenset(
        {"user", "assistant", "tool", "system", "developer", "unknown"}
    )
    CONTENT_KINDS: ClassVar[frozenset[str]] = frozenset(
        {"conversation", "tool_output", "instruction", "mixed_untrusted", "derived_note"}
    )

    def __post_init__(self) -> None:
        for name in ("source", "session_id", "message_id", "role", "text"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if not self.source or not self.session_id or not self.message_id:
            raise ValueError("source, session_id, and message_id must be non-empty")
        if (
            isinstance(self.part_index, bool)
            or not isinstance(self.part_index, int)
            or self.part_index < 0
        ):
            raise ValueError("part_index must be a non-negative integer")
        if self.role not in self.ROLES:
            raise ValueError(f"unsupported role: {self.role}")
        if self.content_kind not in self.CONTENT_KINDS:
            raise ValueError(f"unsupported content_kind: {self.content_kind}")
        if self.recorded_at is not None and not isinstance(self.recorded_at, str):
            raise TypeError("recorded_at must be a string or None")
        if self.source_locator is not None and not isinstance(self.source_locator, str):
            raise TypeError("source_locator must be a string or None")
        if self.line_number is not None and (
            isinstance(self.line_number, bool)
            or not isinstance(self.line_number, int)
            or self.line_number < 1
        ):
            raise ValueError("line_number must be a positive integer or None")

    @property
    def record_id(self) -> str:
        identity = [self.source, self.session_id, self.message_id, self.part_index]
        return _digest(identity)

    @property
    def revision(self) -> str:
        evidence = [
            self.source,
            self.session_id,
            self.message_id,
            self.part_index,
            self.role,
            self.text,
            self.recorded_at,
            self.source_locator,
            self.line_number,
            self.content_kind,
        ]
        return _digest(evidence)


def _digest(value: list[object]) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
