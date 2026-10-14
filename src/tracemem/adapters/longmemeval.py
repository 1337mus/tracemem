"""Conservative bytes adapter for documented LongMemEval question records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ..schema import EvidenceRecord

_MAX_INPUT_BYTES = 64 * 1024 * 1024
_QUESTION_FIELDS = frozenset(
    {
        "question_id",
        "question_type",
        "question",
        "answer",
        "question_date",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
        "answer_session_ids",
    }
)
_QUESTION_TYPES = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
    }
)
_TURN_FIELDS = frozenset({"role", "content"})


class _InvalidJson(ValueError):
    """Private signal for JSON extensions rejected by this adapter."""


class _InvalidSchema(ValueError):
    """Private signal for input that does not match the pinned schema."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJson
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _InvalidJson


def _validate_unicode_scalars(value: Any) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("LongMemEval input contains non-scalar Unicode") from None
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())


def _require_string(value: Any, *, nonblank: bool = False) -> str:
    if type(value) is not str or (nonblank and not value.strip()):
        raise _InvalidSchema
    return value


def _require_string_list(value: Any, *, nonblank: bool = False) -> list[str]:
    if type(value) is not list:
        raise _InvalidSchema
    result: list[str] = []
    for item in value:
        result.append(_require_string(item, nonblank=nonblank))
    return result


def _validate_question_items(data: Any) -> list[dict[str, Any]]:
    if type(data) is not list:
        raise _InvalidSchema
    seen_question_ids: set[str] = set()
    for item in data:
        if type(item) is not dict or item.keys() != _QUESTION_FIELDS:
            raise _InvalidSchema

        question_id = _require_string(item["question_id"], nonblank=True)
        if question_id in seen_question_ids:
            raise _InvalidSchema
        seen_question_ids.add(question_id)

        question_type = _require_string(item["question_type"], nonblank=True)
        if question_type not in _QUESTION_TYPES:
            raise _InvalidSchema
        _require_string(item["question"], nonblank=True)
        _require_string(item["answer"])
        _require_string(item["question_date"], nonblank=True)

        session_ids = _require_string_list(item["haystack_session_ids"], nonblank=True)
        dates = _require_string_list(item["haystack_dates"], nonblank=True)
        sessions = item["haystack_sessions"]
        answer_session_ids = _require_string_list(item["answer_session_ids"], nonblank=True)
        if type(sessions) is not list or not (len(session_ids) == len(dates) == len(sessions)):
            raise _InvalidSchema
        if len(set(session_ids)) != len(session_ids):
            raise _InvalidSchema
        if len(set(answer_session_ids)) != len(answer_session_ids):
            raise _InvalidSchema
        if not set(answer_session_ids).issubset(session_ids):
            raise _InvalidSchema

        for session in sessions:
            if type(session) is not list:
                raise _InvalidSchema
            for turn in session:
                if type(turn) is not dict:
                    raise _InvalidSchema
                keys = set(turn)
                if not _TURN_FIELDS.issubset(keys) or not keys.issubset(
                    _TURN_FIELDS | {"has_answer"}
                ):
                    raise _InvalidSchema
                role = _require_string(turn["role"])
                if role not in {"user", "assistant"}:
                    raise _InvalidSchema
                _require_string(turn["content"])
                if "has_answer" in turn and type(turn["has_answer"]) is not bool:
                    raise _InvalidSchema
    return data


@dataclass(frozen=True, slots=True)
class LongMemEvalSession:
    """One upstream session and its source-ordered evidence turns."""

    upstream_session_id: str
    date: str
    records: tuple[EvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class LongMemEvalQuestion:
    """Question metadata kept separate from the evidence transcript records."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    variant: str
    is_abstention: bool
    sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: tuple[str, ...]
    answer_record_ids: tuple[str, ...]

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        """Flatten history records in their original session and turn order."""
        return tuple(record for session in self.sessions for record in session.records)


def parse_longmemeval_bytes(
    raw: bytes, *, namespace: str, variant: str
) -> tuple[LongMemEvalQuestion, ...]:
    """Parse LongMemEval-shaped JSON bytes without reading or writing files."""
    if not isinstance(raw, bytes):
        raise ValueError("LongMemEval input must be bytes")
    if len(raw) > _MAX_INPUT_BYTES:
        raise ValueError("LongMemEval input exceeds 64 MiB limit")
    if type(namespace) is not str or not namespace.strip():
        raise ValueError("namespace must be a nonblank string")
    try:
        namespace.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("namespace must contain scalar Unicode") from None
    if type(variant) is not str or variant not in {"s", "m", "oracle"}:
        raise ValueError("variant must be one of s, m, oracle")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("LongMemEval input is not valid UTF-8") from None
    try:
        data = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError):
        raise ValueError("LongMemEval input is not valid JSON") from None
    try:
        _validate_unicode_scalars(data)
    except ValueError:
        raise ValueError("LongMemEval input contains non-scalar Unicode") from None
    try:
        data = _validate_question_items(data)
    except _InvalidSchema:
        raise ValueError("LongMemEval input does not match the documented schema") from None
    source_identity = json.dumps([namespace, variant], ensure_ascii=False, separators=(",", ":"))
    source = "longmemeval:" + hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
    questions: list[LongMemEvalQuestion] = []

    for item in data:
        sessions: list[LongMemEvalSession] = []
        answer_record_ids: list[str] = []
        for upstream_session_id, date, turns in zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
            strict=True,
        ):
            session_identity = json.dumps(
                [item["question_id"], upstream_session_id],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            records: list[EvidenceRecord] = []
            for turn_index, turn in enumerate(turns):
                record = EvidenceRecord(
                    source=source,
                    session_id=session_identity,
                    message_id=str(turn_index),
                    part_index=0,
                    role=turn["role"],
                    text=turn["content"],
                    recorded_at=None,
                    source_locator=None,
                    line_number=None,
                    content_kind="conversation",
                )
                records.append(record)
                if turn.get("has_answer") is True:
                    answer_record_ids.append(record.record_id)
            sessions.append(
                LongMemEvalSession(
                    upstream_session_id=upstream_session_id,
                    date=date,
                    records=tuple(records),
                )
            )
        question_id = item["question_id"]
        questions.append(
            LongMemEvalQuestion(
                question_id=question_id,
                question_type=item["question_type"],
                question=item["question"],
                answer=item["answer"],
                question_date=item["question_date"],
                variant=variant,
                is_abstention=question_id.endswith("_abs"),
                sessions=tuple(sessions),
                answer_session_ids=tuple(item["answer_session_ids"]),
                answer_record_ids=tuple(answer_record_ids),
            )
        )
    return tuple(questions)


__all__ = ["LongMemEvalQuestion", "LongMemEvalSession", "parse_longmemeval_bytes"]
