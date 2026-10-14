"""Bounded streaming file scanner for fictional LongMemEval-format JSON."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .longmemeval import LongMemEvalQuestion, parse_longmemeval_bytes

_CHUNK_BYTES = 64 * 1024
_MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ITEM_BYTES = 64 * 1024 * 1024 - 2
_MAX_QUESTIONS = 10_000
_MAX_ID_BYTES = 4 * 1024
_MAX_SEEN_ID_BYTES = 1024 * 1024
_MAX_NESTING_DEPTH = 128
_JSON_WHITESPACE = b" \t\r\n"


@dataclass(frozen=True, slots=True)
class LongMemEvalFileScan:
    """Validated source metadata and an optional bounded ordered selection."""

    input_sha256: str
    input_bytes: int
    question_count: int
    selected_questions: tuple[LongMemEvalQuestion, ...]


class _SafeError(ValueError):
    """Static, non-sensitive rejection message for untrusted input."""


class _BoundedReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.input_bytes = 0
        self.buffer = b""
        self.position = 0
        self.eof = False

    def _refill(self) -> bool:
        if self.eof:
            return False
        remaining = _MAX_FILE_BYTES + 1 - self.input_bytes
        if remaining <= 0:
            raise _SafeError("LongMemEval file exceeds 4 GiB limit")
        chunk = self.stream.read(min(_CHUNK_BYTES, remaining))
        if not chunk:
            self.eof = True
            self.buffer = b""
            self.position = 0
            return False
        self.input_bytes += len(chunk)
        if self.input_bytes > _MAX_FILE_BYTES:
            raise _SafeError("LongMemEval file exceeds 4 GiB limit")
        self.digest.update(chunk)
        self.buffer = chunk
        self.position = 0
        return True

    def read_byte(self) -> int | None:
        if self.position >= len(self.buffer) and not self._refill():
            return None
        value = self.buffer[self.position]
        self.position += 1
        return value

    def unread_byte(self) -> None:
        self.position -= 1

    def skip_whitespace(self) -> int | None:
        while True:
            if self.position >= len(self.buffer) and not self._refill():
                return None
            remaining = self.buffer[self.position :]
            stripped = remaining.lstrip(_JSON_WHITESPACE)
            consumed = len(remaining) - len(stripped)
            self.position += consumed
            if stripped:
                return self.read_byte()


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _validate_question_ids(question_ids: tuple[str, ...]) -> frozenset[str]:
    message = (
        "question_ids must be a tuple of at most five distinct nonblank scalar Unicode strings"
    )
    if type(question_ids) is not tuple or len(question_ids) > 5:
        raise _SafeError(message)
    seen: set[str] = set()
    for question_id in question_ids:
        if type(question_id) is not str or not question_id.strip():
            raise _SafeError(message)
        try:
            encoded_id = question_id.encode("utf-8")
        except UnicodeEncodeError:
            raise _SafeError(message) from None
        if len(encoded_id) > _MAX_ID_BYTES:
            raise _SafeError(message)
        if question_id in seen:
            raise _SafeError(message)
        seen.add(question_id)
    return frozenset(seen)


def _read_item(reader: _BoundedReader) -> bytes:
    first = reader.read_byte()
    if first != ord("{"):
        raise _SafeError("LongMemEval input is not valid JSON")

    item = bytearray((first,))
    openings = [ord("{")]
    in_string = False
    escaped = False
    while openings:
        value = reader.read_byte()
        if value is None:
            raise _SafeError("LongMemEval input is not valid JSON")
        item.append(value)
        if len(item) > _MAX_ITEM_BYTES:
            raise _SafeError("LongMemEval question exceeds 64 MiB limit")

        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue

        if value == ord('"'):
            in_string = True
        elif value in (ord("{"), ord("[")):
            openings.append(value)
            if len(openings) > _MAX_NESTING_DEPTH:
                raise _SafeError("LongMemEval input exceeds nesting limit")
        elif value in (ord("}"), ord("]")):
            expected = ord("{") if value == ord("}") else ord("[")
            if not openings or openings[-1] != expected:
                raise _SafeError("LongMemEval input is not valid JSON")
            openings.pop()

    return bytes(item)


def _scan_stream(
    reader: _BoundedReader,
    *,
    namespace: str,
    variant: str,
    requested: frozenset[str],
) -> tuple[int, tuple[LongMemEvalQuestion, ...]]:
    if reader.skip_whitespace() != ord("["):
        raise _SafeError("LongMemEval input is not valid JSON")

    questions_seen: set[str] = set()
    seen_id_bytes = 0
    selected: list[LongMemEvalQuestion] = []
    count = 0
    next_value = reader.skip_whitespace()
    if next_value == ord("]"):
        pass
    else:
        while True:
            if count >= _MAX_QUESTIONS:
                raise _SafeError("LongMemEval input exceeds 10000 question limit")
            if next_value != ord("{"):
                raise _SafeError("LongMemEval input is not valid JSON")
            reader.unread_byte()
            raw_item = _read_item(reader)
            count += 1
            try:
                (question,) = parse_longmemeval_bytes(
                    b"[" + raw_item + b"]", namespace=namespace, variant=variant
                )
            except ValueError as error:
                raise _SafeError(str(error)) from None
            if question.question_id in questions_seen:
                raise _SafeError("LongMemEval input does not match the documented schema")
            question_id_bytes = len(question.question_id.encode("utf-8"))
            if question_id_bytes > _MAX_ID_BYTES:
                raise _SafeError("LongMemEval input exceeds question ID byte limit")
            if seen_id_bytes + question_id_bytes > _MAX_SEEN_ID_BYTES:
                raise _SafeError("LongMemEval input exceeds cumulative question ID budget")
            seen_id_bytes += question_id_bytes
            questions_seen.add(question.question_id)
            if question.question_id in requested:
                selected.append(question)
            del question
            del raw_item

            delimiter = reader.skip_whitespace()
            if delimiter == ord("]"):
                break
            if delimiter != ord(","):
                raise _SafeError("LongMemEval input is not valid JSON")
            next_value = reader.skip_whitespace()
            if next_value != ord("{"):
                raise _SafeError("LongMemEval input is not valid JSON")

    if reader.skip_whitespace() is not None:
        raise _SafeError("LongMemEval input is not valid JSON")
    missing = requested.difference(questions_seen)
    if missing:
        raise _SafeError("requested question IDs were not found")
    return count, tuple(selected)


def scan_longmemeval_file(
    path: str | os.PathLike[str],
    *,
    namespace: str,
    variant: str,
    question_ids: tuple[str, ...] = (),
) -> LongMemEvalFileScan:
    """Validate a complete JSON array using bounded reads and retain at most five items."""
    parse_longmemeval_bytes(b"[]", namespace=namespace, variant=variant)
    requested = _validate_question_ids(question_ids)

    try:
        source_path = Path(path)
        initial_lstat = source_path.lstat()
        if stat.S_ISLNK(initial_lstat.st_mode) or not stat.S_ISREG(initial_lstat.st_mode):
            raise _SafeError("LongMemEval file must be a regular non-symlink file")
        resolved_path = source_path.resolve(strict=True)
        initial_path_stat = resolved_path.stat()
        if not _same_file_state(initial_lstat, initial_path_stat):
            raise _SafeError("LongMemEval file changed while it was being scanned")
        if initial_path_stat.st_size > _MAX_FILE_BYTES:
            raise _SafeError("LongMemEval file exceeds 4 GiB limit")

        flags = os.O_RDONLY
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source_path, flags)
        try:
            descriptor_stat = os.fstat(descriptor)
            if not _same_file_state(initial_path_stat, descriptor_stat):
                raise _SafeError("LongMemEval file changed while it was being scanned")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                reader = _BoundedReader(stream)
                question_count, selected = _scan_stream(
                    reader,
                    namespace=namespace,
                    variant=variant,
                    requested=requested,
                )
                final_descriptor_stat = os.fstat(stream.fileno())
                if not _same_file_state(initial_path_stat, final_descriptor_stat):
                    raise _SafeError("LongMemEval file changed while it was being scanned")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        final_lstat = source_path.lstat()
        final_resolved_path = source_path.resolve(strict=True)
        final_path_stat = final_resolved_path.stat()
        if (
            str(final_resolved_path) != str(resolved_path)
            or not _same_file_state(initial_lstat, final_lstat)
            or not _same_file_state(initial_path_stat, final_path_stat)
        ):
            raise _SafeError("LongMemEval file changed while it was being scanned")
        return LongMemEvalFileScan(
            input_sha256=reader.digest.hexdigest(),
            input_bytes=reader.input_bytes,
            question_count=question_count,
            selected_questions=selected,
        )
    except _SafeError as error:
        raise ValueError(str(error)) from None
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        raise ValueError("LongMemEval file could not be read") from None


__all__ = ["LongMemEvalFileScan", "scan_longmemeval_file"]
