"""Synthetic retrieval diagnostics; metrics do not evaluate generated answers."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sqlite3
import time
from collections.abc import Sequence
from importlib.resources import files
from typing import Any

from .retrieval import Embedder, Retriever, SearchHit, SignedHashEmbedder
from .schema import EVIDENCE_SCHEMA_VERSION, EvidenceRecord
from .storage import EvidenceStore

BENCHMARK_VERSION = 1
RETRIEVAL_INDEX_VERSION = 1
RRF_VERSION = 1
DEFAULT_TOP_K = 5
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 1729


def load_synthetic_data() -> tuple[list[EvidenceRecord], list[dict[str, Any]], str]:
    """Load the authored fixture and return records, query labels, and its input hash."""
    fixture_root = files("tracemem").joinpath("data")
    corpus_bytes = fixture_root.joinpath("corpus.json").read_bytes()
    query_bytes = fixture_root.joinpath("queries.json").read_bytes()
    corpus_data = json.loads(corpus_bytes)
    queries = json.loads(query_bytes)
    if not isinstance(corpus_data, list) or not isinstance(queries, list):
        raise ValueError("synthetic fixture has an unsupported shape")

    records: list[EvidenceRecord] = []
    by_message_id: dict[str, EvidenceRecord] = {}
    for item in corpus_data:
        if not isinstance(item, dict):
            raise ValueError("synthetic fixture has an unsupported record")
        record = EvidenceRecord(
            source="synthetic",
            session_id=item["session_id"],
            message_id=item["message_id"],
            part_index=0,
            role=item["role"],
            text=item["text"],
            recorded_at=item.get("recorded_at"),
            source_locator=hashlib.sha256(
                f"synthetic-session:{item['session_id']}".encode()
            ).hexdigest(),
            content_kind=item.get("content_kind", "conversation"),
        )
        if record.message_id in by_message_id:
            raise ValueError("synthetic fixture contains duplicate message identifiers")
        records.append(record)
        by_message_id[record.message_id] = record

    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("synthetic fixture has an unsupported query")
        relevant = query.get("relevant_message_ids", [])
        sufficient_sets = query.get("sufficient_sets", [])
        if not isinstance(relevant, list) or not isinstance(sufficient_sets, list):
            raise ValueError("synthetic relevance labels have an unsupported shape")
        if any(message_id not in by_message_id for message_id in relevant):
            raise ValueError("synthetic relevance labels reference an unknown message")
        for evidence_set in sufficient_sets:
            if not isinstance(evidence_set, list) or not evidence_set:
                raise ValueError("sufficient evidence labels must be non-empty sets")
            if any(message_id not in by_message_id for message_id in evidence_set):
                raise ValueError("sufficient evidence labels reference an unknown message")
            if any(by_message_id[message_id].role != "user" for message_id in evidence_set):
                raise ValueError("sufficient evidence sets must use direct user evidence")
        query.setdefault("case_tags", [])

    digest = hashlib.sha256()
    digest.update(len(corpus_bytes).to_bytes(8, "big"))
    digest.update(corpus_bytes)
    digest.update(len(query_bytes).to_bytes(8, "big"))
    digest.update(query_bytes)
    return records, queries, digest.hexdigest()


def run_benchmark(
    store: EvidenceStore,
    queries: Sequence[dict[str, Any]],
    *,
    embedder: Embedder | None = None,
    input_sha256: str,
    k: int = DEFAULT_TOP_K,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare retrieval modes against authored message and evidence-set labels."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k must be an integer between 1 and 100")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    selected_embedder = embedder if embedder is not None else SignedHashEmbedder()
    embedder_metadata = dict(selected_embedder.metadata)
    warmup_retriever = Retriever(store, embedder=selected_embedder)
    warmup_started = time.perf_counter_ns()
    for query in queries:
        warmup_retriever.search(query["query"], mode="hybrid", k=k)
    warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
    per_mode: dict[str, Any] = {}

    for mode_index, mode in enumerate(("bm25", "vector", "hybrid")):
        retriever = Retriever(store, embedder=selected_embedder)
        query_rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        for query in queries:
            start = time.perf_counter_ns()
            hits = retriever.search(query["query"], mode=mode, k=k)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            latencies.append(elapsed_ms)
            query_rows.append(_measure_query(query, hits, elapsed_ms, k=k))
        per_mode[mode] = _summarize_mode(
            query_rows,
            latencies,
            bootstrap_samples=bootstrap_samples,
            seed=seed + mode_index,
        )

    answerable_queries = sum(bool(query.get("relevant_message_ids")) for query in queries)
    unknown_queries = len(queries) - answerable_queries
    return {
        "benchmark": "synthetic-retrieval-diagnostic",
        "benchmark_version": BENCHMARK_VERSION,
        "input_sha256": input_sha256,
        "environment": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "index_versions": {
            "evidence_schema": EVIDENCE_SCHEMA_VERSION,
            "retrieval": RETRIEVAL_INDEX_VERSION,
            "rrf": RRF_VERSION,
        },
        "embedder": embedder_metadata,
        "top_k": k,
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
        "latency_cache_state": "warm",
        "warmup_queries": len(queries),
        "warmup_ms": round(warmup_ms, 3),
        "answerable_queries": answerable_queries,
        "unknown_queries": unknown_queries,
        "modes": per_mode,
        "claims": "retrieval diagnostics only; no answer or claim-fidelity evaluation",
    }


def _measure_query(
    query: dict[str, Any], hits: Sequence[SearchHit], elapsed_ms: float, *, k: int
) -> dict[str, Any]:
    relevant = set(query.get("relevant_message_ids", []))
    sufficient_sets = query.get("sufficient_sets", [])
    retrieved = [hit.record.message_id for hit in hits]
    retrieved_set = set(retrieved)
    recall: float | None
    mrr: float | None
    ndcg: float | None
    sufficient: bool | None
    if relevant:
        recall = len(relevant.intersection(retrieved_set)) / len(relevant)
        first_relevant = next(
            (
                index
                for index, message_id in enumerate(retrieved, start=1)
                if message_id in relevant
            ),
            None,
        )
        mrr = 1.0 / first_relevant if first_relevant is not None else 0.0
        dcg = sum(
            1.0 / math.log2(index + 1)
            for index, message_id in enumerate(retrieved, start=1)
            if message_id in relevant
        )
        ideal_length = min(k, len(relevant))
        ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_length + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        sufficient = any(
            set(evidence_set).issubset(retrieved_set) for evidence_set in sufficient_sets
        )
    else:
        recall = None
        mrr = None
        ndcg = None
        sufficient = None

    return {
        "query_id": query["query_id"],
        "case_tags": list(query.get("case_tags", [])),
        "retrieved_message_ids": retrieved,
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "evidence_sufficient": sufficient,
        "abstained": not hits,
        "latency_ms": elapsed_ms,
    }


def _summarize_mode(
    query_rows: list[dict[str, Any]],
    latencies: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    answerable = [row for row in query_rows if row["recall_at_k"] is not None]
    unknown = [row for row in query_rows if row["recall_at_k"] is None]
    recall_values = [float(row["recall_at_k"]) for row in answerable]
    mrr_values = [float(row["mrr"]) for row in answerable]
    ndcg_values = [float(row["ndcg"]) for row in answerable]
    sufficient_values = [float(bool(row["evidence_sufficient"])) for row in answerable]
    abstention_values = [float(bool(row["abstained"])) for row in unknown]
    unknown_abstention_rate = _mean(abstention_values)
    sufficient = [row for row in answerable if row["evidence_sufficient"]]
    unknown_abstentions = sum(bool(row["abstained"]) for row in unknown)
    return {
        "answerable_queries": len(answerable),
        "unknown_queries": len(unknown),
        "recall_at_k": _mean(recall_values),
        "mrr": _mean(mrr_values),
        "ndcg": _mean(ndcg_values),
        "sufficient_evidence_rate": _mean(sufficient_values),
        "sufficient_evidence_queries": len(sufficient),
        "insufficient_evidence_query_ids": [
            row["query_id"] for row in answerable if not row["evidence_sufficient"]
        ],
        "unknown_abstentions": unknown_abstentions,
        "unknown_abstention_rate": unknown_abstention_rate,
        "unknown_false_positive_rate": (
            None if unknown_abstention_rate is None else 1.0 - unknown_abstention_rate
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "confidence_intervals_95": {
            "recall_at_k": _bootstrap_interval(recall_values, bootstrap_samples, seed),
            "mrr": _bootstrap_interval(mrr_values, bootstrap_samples, seed + 1),
            "ndcg": _bootstrap_interval(ndcg_values, bootstrap_samples, seed + 2),
            "sufficient_evidence_rate": _bootstrap_interval(
                sufficient_values, bootstrap_samples, seed + 3
            ),
            "unknown_abstention_rate": _bootstrap_interval(
                abstention_values, bootstrap_samples, seed + 4
            ),
        },
        "case_breakdown": _case_breakdown(query_rows),
        "per_query": query_rows,
    }


def _case_breakdown(query_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tags = sorted({tag for row in query_rows for tag in row.get("case_tags", [])})
    breakdown: dict[str, Any] = {}
    for tag in tags:
        tagged = [row for row in query_rows if tag in row.get("case_tags", [])]
        answerable = [row for row in tagged if row["recall_at_k"] is not None]
        unknown = [row for row in tagged if row["recall_at_k"] is None]
        breakdown[tag] = {
            "queries": len(tagged),
            "recall_at_k": _mean([float(row["recall_at_k"]) for row in answerable]),
            "sufficient_evidence_rate": _mean(
                [float(bool(row["evidence_sufficient"])) for row in answerable]
            ),
            "unknown_abstention_rate": _mean([float(bool(row["abstained"])) for row in unknown]),
        }
    return breakdown


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 3)


def _bootstrap_interval(values: Sequence[float], samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    estimates = sorted(
        sum(generator.choices(values, k=len(values))) / len(values) for _ in range(samples)
    )
    low_index = round((len(estimates) - 1) * 0.025)
    high_index = round((len(estimates) - 1) * 0.975)
    return [round(estimates[low_index], 4), round(estimates[high_index], 4)]


__all__ = ["load_synthetic_data", "run_benchmark"]
