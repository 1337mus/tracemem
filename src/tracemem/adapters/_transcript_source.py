"""Descriptor and path checks for opt-in quiescent transcript reads."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal

_ERROR = "transcript source is not a quiescent regular file"


def _file_key(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


class QuiescentTranscriptSource:
    """Open a regular leaf file and verify descriptor/path state after full reading."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._initial_key: tuple[int, int, int, int, int, int] | None = None
        self._stream: BinaryIO | None = None

    def __enter__(self) -> BinaryIO:
        descriptor = -1
        try:
            path_stat = self.path.lstat()
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValueError(_ERROR)
            flags = os.O_RDONLY | os.O_NONBLOCK
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(self.path, flags)
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode) or _file_key(path_stat) != _file_key(
                descriptor_stat
            ):
                raise ValueError(_ERROR)
            self._initial_key = _file_key(descriptor_stat)
            self._stream = os.fdopen(descriptor, "rb")
            descriptor = -1
            return self._stream
        except ValueError:
            raise
        except OSError:
            raise ValueError(_ERROR) from None
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def verify_unchanged(self) -> None:
        if self._stream is None or self._initial_key is None:
            raise RuntimeError("quiescent transcript source is not open")
        try:
            descriptor_stat = os.fstat(self._stream.fileno())
            path_stat = self.path.lstat()
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or _file_key(descriptor_stat) != self._initial_key
                or _file_key(path_stat) != self._initial_key
            ):
                raise ValueError(_ERROR)
        except ValueError:
            raise
        except OSError:
            raise ValueError(_ERROR) from None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                if exc_type is None:
                    raise ValueError(_ERROR) from None
        return False
