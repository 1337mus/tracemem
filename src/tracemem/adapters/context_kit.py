"""Optional, read-only discovery of Context Kit transcript sources."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


class ContextKitConfigError(RuntimeError):
    """Context Kit configuration could not be discovered safely."""


def discover_context_kit(
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return (provider, path) pairs reported by an installed Context Kit config."""
    try:
        ctxkit_config = importlib.import_module("ctxkit.config")
        ctxkit_transcripts = importlib.import_module("ctxkit.transcripts")
    except ImportError:
        raise ContextKitConfigError(
            "Context Kit is not installed or its configuration module is unavailable"
        ) from None

    load_config = getattr(ctxkit_config, "load_config", None)
    transcript_files = getattr(ctxkit_transcripts, "transcript_files", None)
    if not callable(load_config) or not callable(transcript_files):
        raise ContextKitConfigError(
            "Context Kit config must provide load_config and transcript_files"
        )

    try:
        if config_path is None:
            config = load_config()
        else:
            config = load_config(Path(config_path).expanduser())
        source_pairs = transcript_files(config)
        if source_pairs is None:
            return ()
    except Exception:
        raise ContextKitConfigError("cannot load Context Kit transcript configuration") from None

    try:
        normalized: list[tuple[str, str]] = []
        for source in source_pairs:
            if not isinstance(source, (list, tuple)) or len(source) != 2:
                raise ContextKitConfigError("Context Kit returned a malformed transcript source")
            provider, path = source
            if provider not in {"claude", "codex"} or not isinstance(
                path, (str, bytes, os.PathLike)
            ):
                raise ContextKitConfigError("Context Kit returned an unsupported transcript source")
            normalized.append((provider, os.fsdecode(os.fspath(path))))
        return tuple(normalized)
    except TypeError:
        raise ContextKitConfigError("Context Kit returned malformed transcript sources") from None


__all__ = ["ContextKitConfigError", "discover_context_kit"]
