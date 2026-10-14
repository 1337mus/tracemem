"""Command-line entry points for local evidence ingestion and retrieval."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from .adapters import AdapterStats, discover_context_kit, read_transcript
from .benchmark import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_TOP_K,
    load_synthetic_data,
    run_benchmark,
)
from .retrieval import (
    Retriever,
    SearchHit,
    SentenceTransformerEmbedder,
    SignedHashEmbedder,
    pack_context,
)
from .snapshots import import_chatgpt_snapshot
from .storage import EvidenceStore

_MODES = ("bm25", "vector", "hybrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracemem")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="search authored synthetic evidence offline")
    demo.add_argument("--query", default="Where is the quarterly planning meeting?")
    demo.add_argument("--mode", choices=_MODES, default="hybrid")
    demo.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    demo.add_argument("--local-model", type=Path)

    benchmark = commands.add_parser("benchmark", help="run the synthetic retrieval benchmark")
    benchmark.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    benchmark.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    benchmark.add_argument("--local-model", type=Path)

    ingest = commands.add_parser("ingest", help="read a supported transcript into a local DB")
    ingest.add_argument("--provider", required=True, choices=("claude", "codex"))
    ingest.add_argument("--input", required=True, type=Path)
    ingest.add_argument("--db", required=True, type=Path)

    import_chatgpt = commands.add_parser(
        "import-chatgpt", help="create a fresh snapshot from an explicit ChatGPT export"
    )
    import_chatgpt.add_argument("--input", required=True, type=Path)
    import_chatgpt.add_argument("--db", required=True, type=Path)

    search = commands.add_parser("search", help="retrieve evidence for an explicit query")
    search.add_argument("--db", required=True, type=Path)
    search.add_argument("--query", required=True)
    search.add_argument("--mode", choices=_MODES, default="hybrid")
    search.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    search.add_argument("--local-model", type=Path)
    search.add_argument("--context-max-bytes", type=int)

    inventory = commands.add_parser("inventory", help="show aggregate counts from a DB")
    inventory.add_argument("--db", required=True, type=Path)

    discover = commands.add_parser("discover", help="count configured Context Kit sources")
    discover.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            return _demo(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "import-chatgpt":
            return _import_chatgpt(args)
        if args.command == "search":
            return _search(args)
        if args.command == "inventory":
            return _inventory(args)
        if args.command == "discover":
            return _discover(args)
        parser.error("unknown command")
    except (OSError, RuntimeError, sqlite3.Error, UnicodeError, ValueError):
        print(
            f"tracemem: {args.command} failed; no transcript contents were printed",
            file=sys.stderr,
        )
        return 2
    return 0


def _make_embedder(model_path: Path | None) -> SignedHashEmbedder | SentenceTransformerEmbedder:
    if model_path is None:
        return SignedHashEmbedder()
    return SentenceTransformerEmbedder(model_path)


def _demo(args: argparse.Namespace) -> int:
    records, _, _ = load_synthetic_data()
    with TemporaryDirectory(prefix="tracemem-demo-") as temporary_directory:
        with EvidenceStore(Path(temporary_directory) / "evidence.sqlite") as store:
            store.ingest(records)
            hits = Retriever(store, _make_embedder(args.local_model)).search(
                args.query, mode=args.mode, k=args.k
            )
    _write_json(
        {
            "query": args.query,
            "mode": args.mode,
            "hits": [_hit_payload(hit) for hit in hits],
        }
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    records, queries, input_hash = load_synthetic_data()
    with TemporaryDirectory(prefix="tracemem-benchmark-") as temporary_directory:
        with EvidenceStore(Path(temporary_directory) / "evidence.sqlite") as store:
            store.ingest(records)
            result = run_benchmark(
                store,
                queries,
                embedder=_make_embedder(args.local_model),
                input_sha256=input_hash,
                k=args.k,
                bootstrap_samples=args.bootstrap_samples,
            )
    _write_json(result)
    return 0


def _ingest(args: argparse.Namespace) -> int:
    source_path = _resolve_source_path(args.input)
    database_path = _resolve_ingest_database_path(source_path, args.db)
    stats = AdapterStats()
    with EvidenceStore(database_path) as store:
        inserted = store.ingest(read_transcript(source_path, args.provider, stats=stats))
    _write_json(
        {
            "inserted_revisions": inserted,
            "malformed_records": stats.malformed_records,
            "unsupported_records": stats.unsupported_records,
            "role_counts": dict(sorted(stats.role_counts.items())),
        }
    )
    return 0


def _import_chatgpt(args: argparse.Namespace) -> int:
    _write_json(import_chatgpt_snapshot(args.input, args.db))
    return 0


def _search(args: argparse.Namespace) -> int:
    database_path = _existing_database_path(args.db)
    if args.context_max_bytes is not None and (
        isinstance(args.context_max_bytes, bool) or args.context_max_bytes < 0
    ):
        raise ValueError("context byte limit must be non-negative")
    with EvidenceStore(database_path) as store:
        hits = Retriever(store, _make_embedder(args.local_model)).search(
            args.query, mode=args.mode, k=args.k
        )
    payload: dict[str, object] = {
        "query": args.query,
        "mode": args.mode,
        "hits": [_hit_payload(hit) for hit in hits],
    }
    if args.context_max_bytes is not None:
        payload["packed_context"] = pack_context(hits, args.context_max_bytes)
        payload["context_budget_unit"] = "UTF-8 bytes including citations; not model tokens"
    _write_json(payload)
    return 0


def _inventory(args: argparse.Namespace) -> int:
    database_path = _existing_database_path(args.db)
    with EvidenceStore(database_path) as store:
        all_records = store.records(current_only=False)
        current_records = store.records(current_only=True)
    _write_json(
        {
            "revisions": len(all_records),
            "current_records": len(current_records),
            "roles": dict(sorted(Counter(record.role for record in all_records).items())),
            "content_kinds": dict(
                sorted(Counter(record.content_kind for record in all_records).items())
            ),
        }
    )
    return 0


def _discover(args: argparse.Namespace) -> int:
    sources = discover_context_kit(args.config)
    source_types: dict[str, Counter[str]] = {}
    for provider, path in sources:
        by_provider = source_types.setdefault(provider, Counter())
        by_provider[Path(path).suffix.lower() or "(no extension)"] += 1
    _write_json(
        {
            "configured_sources": len(sources),
            "source_types_by_provider": {
                provider: dict(sorted(counts.items()))
                for provider, counts in sorted(source_types.items())
            },
        }
    )
    return 0


def _resolve_source_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        raise ValueError("input must be an existing transcript file") from None
    if not resolved.is_file():
        raise ValueError("input must be an existing transcript file")
    return resolved


def _resolve_ingest_database_path(source_path: Path, path: Path) -> Path:
    expanded_path = path.expanduser()
    if expanded_path.is_symlink():
        raise ValueError("database path must not be a symlink")
    try:
        resolved = expanded_path.resolve(strict=False)
    except OSError:
        raise ValueError("database path is invalid") from None
    if (
        resolved == source_path
        or resolved == source_path.parent
        or source_path.parent in resolved.parents
    ):
        raise ValueError("database must be outside the transcript directory")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("database path must be a file")
    return resolved


def _existing_database_path(path: Path) -> Path:
    expanded_path = path.expanduser()
    if expanded_path.is_symlink():
        raise ValueError("database path must not be a symlink")
    try:
        resolved = expanded_path.resolve(strict=True)
    except OSError:
        raise ValueError("database must already exist") from None
    if not resolved.is_file():
        raise ValueError("database must already exist as a file")
    return resolved


def _hit_payload(hit: SearchHit) -> dict[str, object]:
    return {
        "record_id": hit.record.record_id,
        "revision": hit.record.revision,
        "citation": hit.citation,
        "role": hit.record.role,
        "content_kind": hit.record.content_kind,
        "score": hit.score,
        "text": hit.record.text,
    }


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = ["build_parser", "main"]
