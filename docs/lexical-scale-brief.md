# Bounded lexical retrieval

2026-10-14. Local Phase 1 follow-up. Private data and prior receipts remain untouched.

Problem: Retriever.search materializes every current evidence record even for
BM25-only queries, though EvidenceStore.lexical_search already joins current heads
and applies a SQL limit. Avoid loading the whole corpus for the lexical-only path.

Scope: `src/tracemem/retrieval.py`, `tests/test_retrieval.py`, and a small offline
synthetic `experiments/benchmark_lexical_scale.py` only. Preserve vector/hybrid
behavior and rankings, schema, validation order, citations and query limits. For
BM25, call lexical_search with k, retain positive scores, construct identical hits;
do not call records or the embedder. Do not change storage or add dependencies.

Use TDD: first an actual temporary SQLite test, enough records and a newer revision,
assert equal ranking/citations to lexical_search and no call to records/encode.
Run it against the old source to observe the intended failure. Then implement the
small fast path and run retrieval/storage/benchmark tests, Ruff and source mypy.

The scale script must be deterministic, synthetic-only, CPU/stdlib, create only
temporary stores, accept corpus size (bounded maximum) and repeat count, report
median/min/max latency and tracemalloc peak Python bytes, corpus/query-generator
digest, Python/SQLite/environment, and corpus/setup sizes. Compare a retained
legacy algorithm with optimized BM25 on the same store/query; assert nonempty
and identical ranked record/revision IDs and scores. Clear measurements between
runs. Distinguish Python allocations from RSS and measured environment from
production claims. No private sources, inference, downloads, or remote operations.
Root inspects every change and runs the measurement after worker delivery.
