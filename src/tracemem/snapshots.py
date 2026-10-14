"""Fresh, selected-branch snapshots of explicit ChatGPT JSON exports."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .adapters import AdapterStats
from .adapters.chatgpt import read_chatgpt_export
from .storage import EvidenceStore

_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")


def import_chatgpt_snapshot(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> dict[str, object]:
    """Import one explicit export into a new, exclusively published database."""
    try:
        source_path = _resolve_source(source)
        destination_path = _resolve_destination(source_path, destination)
        stats = AdapterStats()
        with TemporaryDirectory(prefix=".tracemem-chatgpt-", dir=destination_path.parent) as temp:
            temporary_directory = Path(temp)
            os.chmod(temporary_directory, 0o700)
            temporary_database = temporary_directory / "snapshot.sqlite3"
            with EvidenceStore(temporary_database) as store:
                inserted = store.ingest(read_chatgpt_export(source_path, stats=stats))
            if stats.malformed_records:
                raise ValueError("malformed conversation")
            _require_unoccupied_destination(destination_path)
            os.link(temporary_database, destination_path)
        return {
            "inserted_revisions": inserted,
            "malformed_records": stats.malformed_records,
            "unsupported_records": stats.unsupported_records,
            "role_counts": dict(sorted(stats.role_counts.items())),
            "snapshot_policy": "fresh_store_selected_branch",
        }
    except Exception:
        raise ValueError("ChatGPT snapshot import failed; no destination was published") from None


def _resolve_source(source: str | os.PathLike[str]) -> Path:
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("source must be a regular file")
    return source_path


def _resolve_destination(source_path: Path, destination: str | os.PathLike[str]) -> Path:
    supplied = Path(destination).expanduser()
    parent = supplied.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("destination parent must be an existing directory")
    destination_path = parent / supplied.name
    if source_path.parent == parent or source_path.parent in parent.parents:
        raise ValueError("destination must be outside the source directory")
    _require_unoccupied_destination(destination_path)
    return destination_path


def _require_unoccupied_destination(destination: Path) -> None:
    candidates = (
        destination,
        *(destination.with_name(destination.name + suffix) for suffix in _SIDECAR_SUFFIXES),
    )
    if any(os.path.lexists(candidate) for candidate in candidates):
        raise ValueError("destination or SQLite sidecar already exists")


__all__ = ["import_chatgpt_snapshot"]
