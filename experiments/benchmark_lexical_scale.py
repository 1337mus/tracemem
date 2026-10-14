"""Compare bounded BM25 retrieval with the legacy full-head materialization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import sqlite3
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tracemem.retrieval as retrieval_module
import tracemem.storage as storage_module
from tracemem.retrieval import Retriever, SearchHit
from tracemem.schema import EvidenceRecord
from tracemem.storage import EvidenceStore

DEFAULT_CORPUS_SIZE = 10_000
MAX_CORPUS_SIZE = 50_000
DEFAULT_REPEATS = 5
MAX_REPEATS = 20
TOP_K = 10


def _record(index: int, text: str) -> EvidenceRecord:
    return EvidenceRecord(
        source="synthetic-scale",
        session_id="scale-session",
        message_id=f"scale-message-{index:08d}",
        part_index=0,
        role="user",
        text=text,
    )


def _generate_corpus(size: int) -> list[EvidenceRecord]:
    rows: list[EvidenceRecord] = []
    for index in range(size):
        if index == 0:
            rows.append(_record(index, "retired synthetic quartz relay revision"))
        repetitions = index % 7
        text = f"synthetic scale item {index:08d} quartz relay sector {index % 31}" + (
            " quartz" * repetitions
        )
        rows.append(_record(index, text))
    return rows


def _generate_query(corpus_size: int) -> str:
    if corpus_size < 1:
        raise ValueError("corpus size must be positive")
    return "quartz relay"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _corpus_query_sha256(records: list[EvidenceRecord], query: str) -> str:
    digest = hashlib.sha256()
    for record in records:
        row = json.dumps(
            [record.record_id, record.revision, record.text],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    query_bytes = query.encode("utf-8")
    digest.update(len(query_bytes).to_bytes(8, "big"))
    digest.update(query_bytes)
    return digest.hexdigest()


def _ranking_digest(rows: list[tuple[str, str, float]]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256(payload)


def _legacy_search(store: EvidenceStore, query: str, k: int) -> list[SearchHit]:
    records = store.records(current_only=True)
    if not records:
        return []
    record_by_key = {(record.record_id, record.revision): record for record in records}
    lexical_rows = store.lexical_search(query, limit=min(1000, len(records)))
    ranked = [
        (record, score)
        for record, score in lexical_rows
        if score > 0 and (record.record_id, record.revision) in record_by_key
    ][:k]
    return [
        SearchHit(
            record=record,
            score=score,
            citation=f"{record.record_id}@{record.revision}:0:{len(record.text)}",
        )
        for record, score in ranked
    ]


def _ranking_rows(hits: list[SearchHit]) -> list[tuple[str, str, float]]:
    return [(hit.record.record_id, hit.record.revision, hit.score) for hit in hits]


def _measure(
    action: Callable[[], list[SearchHit]],
    repeats: int,
    *,
    expected_rows: list[tuple[str, str, float]],
) -> dict[str, Any]:
    elapsed_ms: list[float] = []
    peak_bytes: list[int] = []
    last_digest = ""
    last_count = 0
    for repeat in range(1, repeats + 1):
        gc.collect()
        started = time.perf_counter_ns()
        hits = action()
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        rows = _ranking_rows(hits)
        if rows != expected_rows:
            raise AssertionError(f"latency repeat {repeat} ranking differs from expected")
        last_digest = _ranking_digest(rows)
        last_count = len(rows)
        del hits, rows

    for repeat in range(1, repeats + 1):
        gc.collect()
        tracemalloc.start()
        try:
            hits = action()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peak_bytes.append(peak)
        rows = _ranking_rows(hits)
        if rows != expected_rows:
            raise AssertionError(f"allocation repeat {repeat} ranking differs from expected")
        last_digest = _ranking_digest(rows)
        last_count = len(rows)
        del hits, rows

    return {
        "latency_ms": {
            "median": statistics.median(elapsed_ms),
            "min": min(elapsed_ms),
            "max": max(elapsed_ms),
        },
        "tracemalloc_peak_python_bytes": {
            "median": statistics.median(peak_bytes),
            "min": min(peak_bytes),
            "max": max(peak_bytes),
        },
        "result_count": last_count,
        "ranking_sha256": last_digest,
    }


def run_scale_benchmark(corpus_size: int, repeats: int) -> dict[str, Any]:
    """Run a deterministic comparison using only a temporary synthetic SQLite store."""
    if (
        isinstance(corpus_size, bool)
        or not isinstance(corpus_size, int)
        or not (TOP_K < corpus_size <= MAX_CORPUS_SIZE)
    ):
        raise ValueError(f"corpus size must be between {TOP_K + 1} and {MAX_CORPUS_SIZE}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(f"repeats must be between 1 and {MAX_REPEATS}")

    records = _generate_corpus(corpus_size)
    query = _generate_query(corpus_size)
    source_path = Path(__file__).resolve()
    retrieval_source_path = Path(retrieval_module.__file__).resolve()
    storage_source_path = Path(storage_module.__file__).resolve()
    generator_source = "".join(
        inspect.getsource(function) for function in (_record, _generate_corpus, _generate_query)
    )
    legacy_function_source = inspect.getsource(_legacy_search)
    corpus_digest = _corpus_query_sha256(records, query)
    setup_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tracemem-lexical-scale-") as temporary_directory:
        database_path = Path(temporary_directory) / "scale.sqlite"
        with EvidenceStore(database_path) as store:
            store.ingest(records)
            setup_seconds = time.perf_counter() - setup_started
            database_bytes = database_path.stat().st_size
            optimized_retriever = Retriever(store)

            def legacy() -> list[SearchHit]:
                return _legacy_search(store, query, TOP_K)

            def optimized() -> list[SearchHit]:
                return optimized_retriever.search(query, mode="bm25", k=TOP_K)

            expected_hits = legacy()
            actual_hits = optimized()
            expected = _ranking_rows(expected_hits)
            actual = _ranking_rows(actual_hits)
            if not expected or expected != actual:
                raise AssertionError("legacy and optimized BM25 rankings differ or are empty")
            legacy_measurements = _measure(legacy, repeats, expected_rows=expected)
            optimized_measurements = _measure(optimized, repeats, expected_rows=expected)
            if legacy_measurements["ranking_sha256"] != optimized_measurements["ranking_sha256"]:
                raise AssertionError("measured BM25 rankings changed between implementations")

    return {
        "benchmark": "synthetic-bm25-scale",
        "version": 1,
        "source_sha256": _sha256(source_path.read_bytes()),
        "retrieval_source_sha256": _sha256(retrieval_source_path.read_bytes()),
        "storage_source_sha256": _sha256(storage_source_path.read_bytes()),
        "legacy_function_source_sha256": _sha256(legacy_function_source.encode("utf-8")),
        "generator_sha256": _sha256(generator_source.encode("utf-8")),
        "corpus_query_sha256": corpus_digest,
        "environment": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "corpus": {
            "requested_size": corpus_size,
            "revision_count": len(records),
            "current_head_count": corpus_size,
            "utf8_text_bytes": sum(len(record.text.encode("utf-8")) for record in records),
            "query_utf8_bytes": len(query.encode("utf-8")),
            "top_k": TOP_K,
            "setup_seconds": setup_seconds,
            "sqlite_file_bytes": database_bytes,
        },
        "repeats": repeats,
        "measurements": {
            "legacy_full_head_materialization": legacy_measurements,
            "optimized_bm25": optimized_measurements,
            "rankings_identical": True,
        },
        "measurement_policy": {
            "latency": "separate pass with tracemalloc disabled",
            "python_allocations": "separate pass with tracemalloc enabled",
            "ranking_equivalence": "checked on every repeat in both passes",
        },
        "measurement_limits": [
            "tracemalloc reports Python allocations, not process RSS or SQLite native memory.",
            "Results describe this measured environment and are not production performance claims.",
            "Corpus records and queries are deterministic synthetic text only.",
            "Legacy timing reconstructs the previous BM25 head materialization and SearchHit "
            "creation; fixed-input validation is omitted.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-size", type=int, default=DEFAULT_CORPUS_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)
    try:
        report = run_scale_benchmark(args.corpus_size, args.repeats)
    except (OSError, RuntimeError, TypeError, ValueError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
