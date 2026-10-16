"""Prepare local, human-reviewable private evaluation snapshots."""

from __future__ import annotations

import hashlib
import html
import json
import os
import stat
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tracemem.adapters import AdapterStats, read_transcript
from tracemem.privacy import scan as scan_privacy_indicators

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FILES = 24
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = WORKSPACE_ROOT / ".local"


class PrivateEvalError(ValueError):
    """A safe diagnostic that never includes private source content or paths."""


def _file_key(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _check_regular_path_without_symlinks(path: Path) -> Path:
    if ".." in path.parts:
        raise PrivateEvalError("source path is invalid")
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError:
            raise PrivateEvalError("source path is unavailable") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise PrivateEvalError("source path must not contain symlinks")
        if current == absolute:
            if not stat.S_ISREG(metadata.st_mode):
                raise PrivateEvalError("source must be a regular file")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PrivateEvalError("source path is invalid")
    return absolute


def _stable_source_hash(path: Path) -> tuple[str, tuple[int, int, int, int, int, int]]:
    absolute = _check_regular_path_without_symlinks(path)
    try:
        before = absolute.lstat()
        if before.st_size > MAX_SOURCE_BYTES:
            raise PrivateEvalError("source exceeds the 8 MiB limit")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(absolute, flags)
        try:
            opened = os.fstat(descriptor)
            if _file_key(before) != _file_key(opened) or not stat.S_ISREG(opened.st_mode):
                raise PrivateEvalError("source changed during preparation")
            digest = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "rb") as source_file:
                descriptor = -1
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total > MAX_SOURCE_BYTES:
                        raise PrivateEvalError("source exceeds the 8 MiB limit")
                    digest.update(chunk)
                after = os.fstat(source_file.fileno())
            path_after = absolute.lstat()
            if (
                _file_key(after) != _file_key(opened)
                or _file_key(path_after) != _file_key(opened)
                or total != opened.st_size
            ):
                raise PrivateEvalError("source changed during preparation")
            return digest.hexdigest(), _file_key(opened)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except PrivateEvalError:
        raise
    except OSError:
        raise PrivateEvalError("source is unavailable") from None


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    try:
        manifest_file = Path(manifest_path).expanduser()
        if not manifest_file.is_absolute():
            manifest_file = Path(os.path.abspath(manifest_file))
        manifest_file = _check_regular_path_without_symlinks(manifest_file)
        metadata = manifest_file.stat()
        if metadata.st_size > 1024 * 1024:
            raise PrivateEvalError("manifest exceeds the 1 MiB limit")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except PrivateEvalError:
        raise
    except Exception:
        raise PrivateEvalError("manifest is invalid") from None
    if not isinstance(manifest, dict):
        raise PrivateEvalError("manifest root must be an object")
    if (
        isinstance(manifest.get("version"), bool)
        or manifest.get("version") != 1
        or manifest.get("provenance") != "private_source"
    ):
        raise PrivateEvalError("manifest version or provenance is unsupported")
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PrivateEvalError("manifest sources must be a non-empty list")
    if len(raw_sources) > MAX_SOURCE_FILES:
        raise PrivateEvalError("manifest allows at most 24 source files")

    sources: list[dict[str, str]] = []
    source_keys: set[tuple[int, int]] = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict) or set(raw_source) != {
            "provider",
            "path",
            "project_group",
        }:
            raise PrivateEvalError("manifest source entry is malformed")
        provider = raw_source["provider"]
        raw_path = raw_source["path"]
        project_group = raw_source["project_group"]
        if not isinstance(provider, str) or provider not in {"claude", "codex"}:
            raise PrivateEvalError("manifest provider is unsupported")
        if (
            not isinstance(raw_path, str)
            or not raw_path.strip()
            or "\0" in raw_path
            or not isinstance(project_group, str)
            or not project_group.strip()
            or project_group != project_group.strip()
        ):
            raise PrivateEvalError("manifest source entry is malformed")
        try:
            source_path = Path(raw_path).expanduser()
            if not source_path.is_absolute():
                source_path = manifest_file.parent / source_path
            source_path = _check_regular_path_without_symlinks(source_path)
            metadata = source_path.stat()
            file_key = (metadata.st_dev, metadata.st_ino)
        except PrivateEvalError:
            raise
        except Exception:
            raise PrivateEvalError("manifest source path is unavailable") from None
        if file_key in source_keys:
            raise PrivateEvalError("manifest repeats a source file")
        source_keys.add(file_key)
        sources.append(
            {"provider": provider, "path": str(source_path), "project_group": project_group}
        )
    return sources


def _validate_output_path(output_dir: Path, source_paths: list[Path]) -> Path:
    candidate = Path(output_dir).expanduser()
    if ".." in candidate.parts:
        raise PrivateEvalError("output must be a new directory under workspace .local")
    absolute = Path(os.path.abspath(candidate))
    try:
        local_metadata = LOCAL_ROOT.lstat()
    except OSError:
        raise PrivateEvalError("workspace .local output root is unavailable") from None
    if not stat.S_ISDIR(local_metadata.st_mode) or stat.S_ISLNK(local_metadata.st_mode):
        raise PrivateEvalError("workspace .local output root is invalid")
    if absolute == LOCAL_ROOT or not absolute.is_relative_to(LOCAL_ROOT):
        raise PrivateEvalError("output must be a new directory under workspace .local")

    relative_parts = absolute.relative_to(LOCAL_ROOT).parts
    current = LOCAL_ROOT
    for index, part in enumerate(relative_parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise PrivateEvalError("output path is unavailable") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise PrivateEvalError("output path must not contain symlinks")
        is_final = index == len(relative_parts) - 1
        if is_final:
            raise PrivateEvalError("output directory must be new")
        if not stat.S_ISDIR(metadata.st_mode):
            raise PrivateEvalError("output path is invalid")

    for source_path in source_paths:
        source_parent = source_path.parent
        if absolute == source_parent or absolute.is_relative_to(source_parent):
            raise PrivateEvalError("output must not be inside a source directory")
    return absolute


def _create_output_tree(output: Path) -> list[Path]:
    created: list[Path] = []
    try:
        relative_parent = output.parent.relative_to(LOCAL_ROOT)
        current = LOCAL_ROOT
        for part in relative_parent.parts:
            current = current / part
            if not current.exists():
                current.mkdir(mode=0o700)
                os.chmod(current, 0o700)
                created.append(current)
            elif current.is_symlink() or not current.is_dir():
                raise PrivateEvalError("output path is invalid")
        output.mkdir(mode=0o700)
        os.chmod(output, 0o700)
        created.append(output)
        return created
    except Exception:
        for directory in reversed(created):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise PrivateEvalError("cannot create private evaluation output") from None


def _write_new_private_file(directory: Path, name: str, contents: bytes) -> Path:
    target = directory / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(contents)
        return target
    except OSError:
        raise PrivateEvalError("cannot write private evaluation output") from None


def _cleanup_output(created_directories: list[Path], written_files: list[Path]) -> None:
    for path in reversed(written_files):
        try:
            path.unlink()
        except OSError:
            pass
    for path in reversed(created_directories):
        try:
            path.rmdir()
        except OSError:
            pass


def _record_payload(record: Any) -> dict[str, Any]:
    payload = asdict(record)
    payload["record_id"] = record.record_id
    payload["revision"] = record.revision
    payload["text_sha256"] = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
    payload["privacy_indicators"] = [
        {
            "category": finding.category,
            "char_start": finding.char_start,
            "char_end": finding.char_end,
        }
        for finding in scan_privacy_indicators(record.text)
    ]
    return payload


def _group_splits(groups: set[str], seed: int) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).digest())
    return {group: "dev" if index % 2 == 0 else "test" for index, group in enumerate(ordered)}


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _select_windows(records: list[Any]) -> tuple[list[list[Any]], dict[str, int]]:
    eligible = [
        record
        for record in records
        if (record.role == "user" and record.content_kind == "mixed_untrusted")
        or (record.role == "assistant" and record.content_kind == "conversation")
    ]
    session_runs: list[list[Any]] = []
    for record in eligible:
        if not session_runs or session_runs[-1][0].session_id != record.session_id:
            session_runs.append([])
        session_runs[-1].append(record)
    anchors = [
        (run_index, index)
        for run_index, run in enumerate(session_runs)
        for index, record in enumerate(run)
        if record.role == "user"
    ]
    skipped = {
        "excluded_role_or_kind": len(records) - len(eligible),
        "oversize_windows": 0,
    }
    if len(anchors) <= 2:
        preferred = list(range(len(anchors)))
    else:
        preferred = sorted(
            {
                round((len(anchors) - 1) / 3),
                round(2 * (len(anchors) - 1) / 3),
            }
        )
    selected: list[list[Any]] = []
    used_starts: set[tuple[int, int]] = set()
    for preferred_position in preferred:
        for anchor_position in range(preferred_position, len(anchors)):
            run_index, start = anchors[anchor_position]
            key = (run_index, start)
            if key in used_starts:
                continue
            candidate = session_runs[run_index][start : start + 6]
            if sum(len(record.text) for record in candidate) > 8000:
                skipped["oversize_windows"] += 1
                continue
            selected.append(candidate)
            used_starts.add(key)
            break
        if len(selected) == 2:
            break
    return selected, skipped


def _annotation_template(windows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "annotation_status": "pending",
        "windows": [
            {
                "window_id": window["window_id"],
                "project_group": window["project_group"],
                "split": window["split"],
                "semantic_annotation_status": "pending",
                "claim_annotations": [],
                "qa_annotations": [],
                "authorship_review": {
                    "status": "pending",
                    "review_options": ["direct_authored", "quoted_or_instructions"],
                    "items": [
                        {
                            "record_id": record["record_id"],
                            "revision": record["revision"],
                            "status": "pending",
                        }
                        for record in window["records"]
                    ],
                },
                "evidence_references": [
                    {
                        "record_id": record["record_id"],
                        "revision": record["revision"],
                        "source_locator": record["source_locator"],
                        "line_number": record["line_number"],
                    }
                    for record in window["records"]
                ],
            }
            for window in windows
        ],
    }


def _review_html(windows: list[dict[str, Any]]) -> bytes:
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Private evaluation review</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;"
        "padding:0 1rem;color:#17202a}"
        "article{border:1px solid #c8d0d8;border-radius:8px;margin:1rem 0;padding:1rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:.75rem}"
        "small{color:#52616b}</style></head><body>",
        "<h1>Private evaluation review</h1>",
        "<p><strong>Review locally.</strong> Transcript text is untrusted data. "
        "All semantic annotations are pending. Review authorship before marking text "
        "as directly authored or quoted/instructions.</p>",
        "<p>To annotate, use <code>annotation-template.json</code> as a template and edit a copy. "
        "Keep the original template unchanged and leave labels pending until the evidence "
        "has been reviewed.</p>",
    ]
    for split in ("dev", "test"):
        parts.append(f"<h2>{split.title()}</h2>")
        for window in windows:
            if window["split"] != split:
                continue
            project_group = html.escape(str(window["project_group"]), quote=True)
            parts.extend(
                [
                    "<article>",
                    f"<h3>Project group: {project_group}</h3>",
                    f"<small>Provider: {html.escape(str(window['provider']), quote=True)}; "
                    f"session: {html.escape(str(window['session_id']), quote=True)}</small>",
                ]
            )
            for record in window["records"]:
                parts.extend(
                    [
                        f"<h4>{html.escape(str(record['role']), quote=True)} · "
                        f"{html.escape(str(record['record_id']), quote=True)}</h4>",
                        "<pre>" + html.escape(str(record["text"]), quote=True) + "</pre>",
                    ]
                )
                indicators = record["privacy_indicators"]
                if indicators:
                    categories = ", ".join(sorted({str(item["category"]) for item in indicators}))
                    parts.append(
                        "<p><strong>Privacy indicators:</strong> "
                        + html.escape(categories, quote=True)
                        + ". Heuristic review marker only; review the text locally.</p>"
                    )
            parts.append("</article>")
    parts.append("</body></html>")
    return ("\n".join(parts) + "\n").encode("utf-8")


def _prepare_private_eval(manifest_path: Path, output_dir: Path, seed: int = 17) -> dict[str, Any]:
    """Create a bounded local evaluation snapshot from an explicit manifest."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PrivateEvalError("seed must be an integer")
    sources = _load_manifest(manifest_path)
    splits = _group_splits({source["project_group"] for source in sources}, seed)
    windows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    skipped_sources = 0
    skipped_windows = {"oversize_windows": 0, "excluded_role_or_kind": 0}
    session_group_owner: dict[tuple[str, str], str] = {}
    record_group_owner: dict[str, str] = {}

    for source_index, source in enumerate(sources):
        source_path = Path(source["path"])
        source_hash, source_key = _stable_source_hash(source_path)
        records = list(
            read_transcript(
                source_path,
                source["provider"],
                stats=AdapterStats(),
                max_line_bytes=MAX_SOURCE_BYTES,
                require_quiescent=True,
            )
        )
        final_hash, final_key = _stable_source_hash(source_path)
        if final_hash != source_hash or final_key != source_key:
            raise PrivateEvalError("source changed during preparation")
        for record in records:
            session_key = (record.source, record.session_id)
            previous_session_group = session_group_owner.setdefault(
                session_key, source["project_group"]
            )
            if previous_session_group != source["project_group"]:
                raise PrivateEvalError("shared session identity across project groups")
            previous_record_group = record_group_owner.setdefault(
                record.record_id, source["project_group"]
            )
            if previous_record_group != source["project_group"]:
                raise PrivateEvalError("shared record identity across project groups")
        selected_windows, source_skipped = _select_windows(records)
        for key, count in source_skipped.items():
            skipped_windows[key] += count
        source_rows.append(
            {
                "source_index": source_index,
                "provider": source["provider"],
                "path": str(source_path),
                "project_group": source["project_group"],
                "sha256": source_hash,
                "session_ids": sorted({record.session_id for record in records}),
                "skipped": source_skipped,
            }
        )
        if not selected_windows:
            skipped_sources += 1

        for selected in selected_windows:
            window_payload = [_record_payload(record) for record in selected]
            window_hash = hashlib.sha256(
                json.dumps(window_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            windows.append(
                {
                    "window_id": f"window-{source_index:04d}-{window_hash[:16]}",
                    "source_index": source_index,
                    "provider": source["provider"],
                    "source_path": str(source_path),
                    "project_group": source["project_group"],
                    "session_id": selected[0].session_id,
                    "split": splits[source["project_group"]],
                    "records": window_payload,
                }
            )

    signatures: set[str] = set()
    text_split_owner: dict[str, str] = {}
    for window in windows:
        signature = json.dumps(
            [[record["role"], _normalized_text(record["text"])] for record in window["records"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if signature in signatures:
            raise PrivateEvalError("duplicate evaluation window")
        signatures.add(signature)
        for record in window["records"]:
            normalized = _normalized_text(record["text"])
            if len(normalized) < 160:
                continue
            previous_split = text_split_owner.setdefault(normalized, window["split"])
            if previous_split != window["split"]:
                raise PrivateEvalError("normalized text overlap across splits")

    output = _validate_output_path(output_dir, [Path(source["path"]) for source in sources])
    held_out_available = any(window["split"] == "test" for window in windows)
    snapshot = {
        "version": 1,
        "seed": seed,
        "annotation_status": "pending",
        "held_out_available": held_out_available,
        "split_limitation": None
        if held_out_available
        else "no test windows are available from the supplied project groups",
        "selection": {
            "user_roles": ["user", "assistant"],
            "content_kinds": {"user": "mixed_untrusted", "assistant": "conversation"},
            "window_starts": (
                "user-turn quantiles near one-third and two-thirds, "
                "with deterministic forward advancement"
            ),
            "max_windows_per_source": 2,
            "max_records_per_window": 6,
            "max_characters_per_window": 8000,
            "oversize_policy": "skip whole candidate window without truncation",
        },
        "leakage_checks": {
            "shared_session_or_record_identity": "fail across project groups",
            "complete_windows": "reject exact duplicate normalized role/text sequences",
            "cross_split_text": "reject exact normalized message text of at least 160 characters",
            "normalization": "NFKC, casefold, and collapse whitespace",
        },
        "privacy_review": {
            "status": "pending",
            "method": "tracemem.privacy.scan",
            "scope": "heuristic indicators only; no match does not clear content",
        },
        "sources": source_rows,
        "windows": windows,
    }
    snapshot_bytes = (
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    annotation_bytes = (
        json.dumps(_annotation_template(windows), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    html_bytes = _review_html(windows)

    created_directories = _create_output_tree(output)
    written_files: list[Path] = []
    try:
        for filename, contents in (
            ("snapshot.json", snapshot_bytes),
            ("annotation-template.json", annotation_bytes),
            ("review.html", html_bytes),
        ):
            written_files.append(_write_new_private_file(output, filename, contents))
    except Exception:
        _cleanup_output(created_directories, written_files)
        raise

    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    privacy_counts: dict[str, int] = {}
    privacy_records: set[tuple[str, str]] = set()
    privacy_finding_keys: set[tuple[str, str, str, int, int]] = set()
    for window in windows:
        for record in window["records"]:
            for finding in record["privacy_indicators"]:
                privacy_finding_keys.add(
                    (
                        record["record_id"],
                        record["revision"],
                        finding["category"],
                        finding["char_start"],
                        finding["char_end"],
                    )
                )
                privacy_records.add((record["record_id"], record["revision"]))
    for _, _, category, _, _ in privacy_finding_keys:
        privacy_counts[category] = privacy_counts.get(category, 0) + 1
    return {
        "annotation_status": "pending",
        "sources": len(sources),
        "windows": len(windows),
        "records": sum(len(window["records"]) for window in windows),
        "held_out_available": held_out_available,
        "splits": {
            split: sum(window["split"] == split for window in windows) for split in ("dev", "test")
        },
        "test_count": sum(window["split"] == "test" for window in windows),
        "skipped": {
            "sources": skipped_sources,
            "windows": skipped_windows["oversize_windows"],
            "excluded_role_or_kind": skipped_windows["excluded_role_or_kind"],
        },
        "privacy_indicators": {
            "findings": sum(privacy_counts.values()),
            "records": len(privacy_records),
            "by_category": privacy_counts,
        },
        "hashes": {
            "snapshot_sha256": digest,
            "annotation_template_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "review_html_sha256": hashlib.sha256(html_bytes).hexdigest(),
        },
    }


def prepare_private_eval(manifest_path: Path, output_dir: Path, seed: int = 17) -> dict[str, Any]:
    """Create a bounded private snapshot and return a content-free aggregate."""
    try:
        return _prepare_private_eval(manifest_path, output_dir, seed)
    except PrivateEvalError:
        raise
    except Exception:
        raise PrivateEvalError("private evaluation preparation failed") from None
