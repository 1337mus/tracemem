"""Deterministic synthetic retrieval evidence under fixed UTF-8 byte budgets."""

from __future__ import annotations

import platform
import sqlite3
from collections.abc import Sequence
from typing import Any

from .benchmark import RETRIEVAL_INDEX_VERSION, RRF_VERSION
from .retrieval import Embedder, Retriever, SignedHashEmbedder, pack_context_with_manifest
from .schema import EVIDENCE_SCHEMA_VERSION
from .storage import EvidenceStore

CONTEXT_BUDGET_BENCHMARK_VERSION = 1
DEFAULT_CONTEXT_BUDGETS_BYTES = (0, 256, 512, 1024, 2048, 4096)
DEFAULT_CONTEXT_TOP_K = 5
DEFAULT_RRF_K = 60
MAX_CONTEXT_BUDGETS = 32
_MODES = ("bm25", "vector", "hybrid")


def run_context_budget_benchmark(
    store: EvidenceStore,
    queries: Sequence[dict[str, Any]],
    *,
    input_sha256: str,
    budgets_bytes: Sequence[int] = DEFAULT_CONTEXT_BUDGETS_BYTES,
    k: int = DEFAULT_CONTEXT_TOP_K,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Measure authored relevant evidence retained by whole-hit context packing."""
    budgets = _validate_budgets(budgets_bytes)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k must be an integer between 1 and 100")

    selected_embedder = embedder if embedder is not None else SignedHashEmbedder()
    embedder_metadata = dict(selected_embedder.metadata)
    answerable_queries = sum(bool(query.get("relevant_message_ids", [])) for query in queries)
    unknown_queries = len(queries) - answerable_queries
    mode_results: dict[str, Any] = {}

    for mode in _MODES:
        retriever = Retriever(store, embedder=selected_embedder, rrf_k=DEFAULT_RRF_K)
        per_query: list[dict[str, Any]] = []
        for query in queries:
            hits = retriever.search(query["query"], mode=mode, k=k)
            retrieved_ids = [hit.record.message_id for hit in hits]
            retrieved_set = set(retrieved_ids)
            relevant = set(query.get("relevant_message_ids", []))
            answerable = bool(relevant)
            row: dict[str, Any] = {
                "query_id": query["query_id"],
                "answerable": answerable,
                "retrieved_message_ids": retrieved_ids,
                "retrieved_citations": [hit.citation for hit in hits],
                "retrieved_relevant_message_recall": (
                    len(relevant.intersection(retrieved_set)) / len(relevant)
                    if answerable
                    else None
                ),
                "budgets": {},
            }
            for budget in budgets:
                packed = pack_context_with_manifest(hits, budget)
                packed_ids = [hit.record.message_id for hit in packed.selected_hits]
                packed_set = set(packed_ids)
                row["budgets"][str(budget)] = {
                    "packed_message_ids": packed_ids,
                    "packed_citations": [hit.citation for hit in packed.selected_hits],
                    "packed_bytes": len(packed.text.encode("utf-8")),
                    "relevant_message_recall": (
                        len(relevant.intersection(packed_set)) / len(relevant)
                        if answerable
                        else None
                    ),
                    "sufficient_evidence_survives": (
                        any(
                            set(evidence_set).issubset(packed_set)
                            for evidence_set in query.get("sufficient_sets", [])
                        )
                        if answerable
                        else None
                    ),
                    "empty_pack": not packed.selected_hits,
                }
            per_query.append(row)

        mode_results[mode] = {
            "per_query": per_query,
            "by_budget": {str(budget): _summarize_budget(per_query, budget) for budget in budgets},
        }

    return {
        "benchmark": "synthetic-context-budget-diagnostic",
        "benchmark_version": CONTEXT_BUDGET_BENCHMARK_VERSION,
        "fixture_sha256": input_sha256,
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
        "budgets_bytes": list(budgets),
        "retrieval_config": {
            "modes": list(_MODES),
            "rrf_k": DEFAULT_RRF_K,
            "vector_embedder": embedder_metadata.get("kind", "unknown"),
        },
        "packing_config": {
            "unit": "UTF-8 bytes",
            "passage_policy": "whole citation and metadata lines; oversized passages are skipped",
            "empty_pack": "mechanical packing outcome, not calibrated semantic abstention",
        },
        "answerable_queries": answerable_queries,
        "unknown_queries": unknown_queries,
        "modes": mode_results,
        "claims": (
            "fictional synthetic retrieval and packing evidence only; no QA accuracy, "
            "token-count, or trained-model quality claim"
        ),
    }


def _validate_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    if isinstance(budgets, (str, bytes)) or not isinstance(budgets, Sequence):
        raise ValueError("budgets must be a non-empty sequence of distinct non-negative integers")
    if not budgets or len(budgets) > MAX_CONTEXT_BUDGETS:
        raise ValueError(f"budgets must contain between 1 and {MAX_CONTEXT_BUDGETS} values")
    if any(
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 0 for budget in budgets
    ):
        raise ValueError("budgets must be non-negative integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be distinct")
    return tuple(budgets)


def _summarize_budget(per_query: Sequence[dict[str, Any]], budget: int) -> dict[str, Any]:
    answerable = [row for row in per_query if row["answerable"]]
    unknown = [row for row in per_query if not row["answerable"]]
    budget_key = str(budget)
    sufficient_count = sum(
        bool(row["budgets"][budget_key]["sufficient_evidence_survives"]) for row in answerable
    )
    return {
        "budget_bytes": budget,
        "answerable_queries": len(answerable),
        "unknown_queries": len(unknown),
        "mean_relevant_message_recall": _mean(
            [float(row["budgets"][budget_key]["relevant_message_recall"]) for row in answerable]
        ),
        "sufficient_evidence_queries": sufficient_count,
        "sufficient_evidence_rate": (sufficient_count / len(answerable) if answerable else None),
        "answerable_empty_packs": sum(
            bool(row["budgets"][budget_key]["empty_pack"]) for row in answerable
        ),
        "unknown_empty_packs": sum(
            bool(row["budgets"][budget_key]["empty_pack"]) for row in unknown
        ),
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = ["run_context_budget_benchmark"]
