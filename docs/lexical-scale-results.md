# BM25 scale measurements

2026-10-14, local CPU/stdlib only. BM25-only retrieval requests at most `k`
current heads directly from FTS instead of materializing every current
EvidenceRecord. Query validation, SQL ranking, positive-score filtering, exact
revision citations, vector retrieval and hybrid ranking remain intact. The
schema-3 FTS projection uses the indexed `evidence_heads` rowid for FTS
maintenance, including head deletion. Version-2 stores are rejected by a
read-only preflight; no migration is performed. See the [FTS scale brief](fts-scale-brief.md).

The original and schema-3 runs used the same deterministic corpus and query at
each size. The table reports the old and new measurements side by side; setup
and retrieval timings are observations from these runs, not isolated causal
estimates.

| Heads | Setup, before → schema 3 | Legacy median, before → schema 3 | Bounded BM25 median, before → schema 3 | Legacy Python peak, before → schema 3 | Bounded Python peak, before → schema 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4.058199 s → 0.260205 s | 72.770375 ms → 78.502875 ms | 17.656958 ms → 19.308250 ms | 8,708,504 B → 8,708,504 B | 12,670 B → 12,670 B |
| 50,000 | 129.551996 s → 1.575206 s | 442.827459 ms → 461.030917 ms | 150.338708 ms → 149.189541 ms | 42,020,930 B → 42,020,930 B | 12,670 B → 12,670 B |

Each run used five timing repeats and five separate allocation repeats, after
warming both implementations on the same temporary SQLite store. Rankings were
checked on every repeat in both passes. The before/after corpus-query hashes
match exactly: `cd6777a08d821588b27c6a7aa718e2dde2036612e400b9742e2c7991f8668f06`
at 10,000 heads and
`64eb1685d89442710025c334e57688125e166b5877af2887fc22b92fb98b1545` at
50,000. The legacy and optimized ranking hashes also match before and after:
`3e1198691089794b41d23680dc8c494a39cf0b4818c827ee1bb736c223644049` at
10,000 and
`b859cfe99bcf3e77d59eebe5d3d4aacba9f33afc9abb4fd288301936038d4170` at
50,000. Thus the retrieval order, record IDs, revisions and scores matched in
the measured repeats. Source hashes confirm that retrieval, benchmark,
generator, and retained legacy-function sources were unchanged; storage source
changed for schema 3.

Raw receipts: [10,000 heads before](baseline-bm25-scale-10000.json),
[50,000 heads before](baseline-bm25-scale-50000.json),
[10,000 heads with schema 3](baseline-bm25-fts-v3-10000.json), and
[50,000 heads with schema 3](baseline-bm25-fts-v3-50000.json). They record the
synthetic generator, corpus/query, benchmark, retrieval/storage and legacy
function hashes, plus Python 3.12.14, SQLite 3.53.1 and the measured
macOS/arm64 environment. The corpus-query hashes match across before/after
receipts at each size. Reproduce a current measurement with:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.benchmark_lexical_scale --corpus-size 10000 --repeats 5
```

These are deliberately broad-match queries over deterministic synthetic text,
not a production workload or quality benchmark. Latency excludes tracemalloc;
allocation measurements include it in a separate pass. Python allocation peaks
exclude SQLite native memory, process RSS and preexisting corpus objects. The
retained legacy path reconstructs materialization, filtering and SearchHit
creation while omitting fixed-input validation. Both implementations run in a
fixed order, so thermal/cache/order effects are possible. The lower setup times
in the schema-3 receipts do not isolate a single cause and should not be
generalized to other machines, corpora or production workloads. Hybrid and
vector retrieval still materialize all current records. The measurements do
not resolve retention, erasure, unknown-query calibration or semantic-support
safety.
