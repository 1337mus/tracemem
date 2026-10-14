# Phase 1 executed experiments and checks

Run date: 2026-10-13 America/Los_Angeles, 2026-10-14 UTC. Local prototype only.
The measurements below used authored synthetic data. Private-source reads
produced only the aggregate inventory and ignored local receipts.

This is the frozen Phase 1 record. See [Phase 2 synthetic diagnostics](experiments-phase2.md)
for later inference receipts and extraction results.

## Reproduce the baseline

From the workspace root, with Python 3.11+ and SQLite FTS5:

```sh
make demo
make benchmark
PYTHONPATH=src python3 -m tracemem benchmark > /tmp/tracemem-baseline.json
```

The recorded run used `PYTHONPATH=src .venv/bin/python -m tracemem benchmark`,
exit 0. Its full output is [baseline-retrieval.json](baseline-retrieval.json).
The fixture has 40 text records in 14 fictional sessions, 14 answerable queries
and 3 unknown queries. It is newly authored MIT-licensed diagnostic material,
not an audited training dataset or an independent population benchmark.

Input SHA-256, including both exact fixture byte streams and their lengths:
`76862babda5e6e26be1ed14f5265c64e43627a54b1499dba47585605396637f3`.

Configuration: k=5, 384-dimensional signed token hashing, L2 normalization,
cosine similarity, RRF constant 60, evidence schema version 2 and retrieval/RRF
versions 1. Hash
vectors require lexical overlap to suppress pure hash-collision matches;
they are not semantic embeddings. BM25 queries use OR across safely quoted
terms. Both vector and lexical ranks use deterministic identity tie breaks.
[Configuration snapshot](../configs/retrieval-baseline.json) records these defaults;
the implementation currently uses code defaults rather than loading this file.

## Results

| Mode | Mean recall@5 | MRR | nDCG@5 | Sufficient evidence | Warm p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8214 | 0.8595 | 0.7851 | 14/14 | 0.485 / 0.559 |
| Hash vectors | 0.6310 | 0.7917 | 0.6071 | 10/14 | 2.982 / 3.285 |
| Hybrid RRF | 0.7976 | 0.8810 | 0.7444 | 13/14 | 3.632 / 3.980 |

Recall measures retrieval of all labeled relevant messages. Sufficient evidence
requires retrieving at least one complete authored set of direct user passages;
assistant paraphrases can be relevant without satisfying that stricter label.
These are retrieval metrics, not extraction precision or answer correctness.
nDCG's ideal denominator uses min(k, relevant count), even with fewer returned hits.

All three modes retrieved unrelated passages for all three unknown queries:
0/3 abstentions. No calibrated unknown-query rejection policy exists yet.
Hybrid misses the complete evidence set for `q02`, which asks for both archive
bucket and retention. Hash vectors additionally miss `q04`, `q05`, and `q08`.
The fixed toy corpus currently favors BM25 for complete evidence retrieval.
Hybrid's slightly higher MRR does not establish overall superiority.

Bootstrap intervals resample answerable or unknown queries, 1,000 times, base
seed 1729. Modes use seed plus their BM25/vector/hybrid index; each metric uses
the documented offset in `benchmark.py`. Recall@5 95% intervals are BM25
[0.6905, 0.9405], vectors [0.5000, 0.7619], hybrid [0.6667, 0.9286]. These are
unpaired descriptive intervals. Tiny, correlated hand-authored examples and
degenerate bootstrap endpoints cannot justify high-precision or significance claims.

All 17 queries receive one hybrid warmup pass before measured modes, so needed
vector caches are populated consistently. Warmup took 72.286 ms and is reported
separately. Latency includes query encoding, evidence loading, SQLite reads and
ranking; it excludes initial ingest and fixture loading. Modes run sequentially
in a single process. Timing is a small CPU smoke observation, not a scale test.
Vector search loads current records and scans vectors linearly. No whole-private-
corpus latency, index size, throughput, RAM, model token or QA cost is measured.

## Validation actually run

[Phase 1 validation receipt](validation-phase1.json) records observed exits, final source/config
hashes, the installed wheel hash and fixture/result hashes. All work is local
and uncommitted; these hashes identify the validated prototype independently
of a Git commit.

| Command or check | Observed result |
| --- | --- |
| `PYTHONPATH=src:/Users/raj/Projects/ctxkit/src .venv/bin/python -m unittest discover -s tests -v` | exit 0; 32 tests pass, no skips |
| `.venv/bin/ruff check src tests` | exit 0 |
| `.venv/bin/ruff format --check src tests` | exit 0; 15 files formatted |
| `.venv/bin/mypy` | exit 0; strict checking, 10 source files |
| `PYTHONPATH=src:/Users/raj/Projects/ctxkit/src .venv/bin/python -m tracemem discover` | exit 0; real Context Kit discovery works, only aggregate output |
| Offline `setuptools.build_meta.build_wheel("dist")` | exit 0; both fixture JSON files included |
| Fresh temporary venv, offline wheel install with `--no-deps` | exit 0; installed CLI demo and benchmark pass outside checkout |

The isolated installed benchmark matched the source run's input hash, quality
metrics and bootstrap intervals exactly. Latencies are expected to vary.
The build requirement is setuptools 77+ for the SPDX license string; the tested
developer lock pins 80.9.0. The package smoke receipt is ignored at
`.local/package-smoke.json`.

Tests cover source-shaped multipart messages/tool outputs, duplicate-event
suppression, partial JSONL tails, conflicting roles, stable block identities,
Context Kit discovery, transactional rollback, idempotent import, immutable
revisions including SQL REPLACE rejection, historical/current revision selection,
read-only unknown-database rejection, output path/symlink guards, private file
permissions, revision/model-bound vector caches, Unicode citations and context
packing, nDCG short-result behavior, and privacy detector ranges/redaction.
The optional SentenceTransformer loader is tested with a constructor substitute
for local-only settings and content fingerprinting. Real semantic inference had
not run at the Phase 1 boundary; a separate pinned MiniLM retrieval comparison is
recorded in the later Phase 2 report. Context packing budgets UTF-8 bytes including provenance metadata;
it does not measure model tokens.

The accountable final code review traced transcript roles through storage and
retrieval, checked destructive SQL/path cases and citation revisions, and
corrected unstable block positions, structured-tool rendering, role conflicts,
cache/timing/metric behavior and package-data inclusion before these checks.
This is a local engineering review; no external review service was invoked.

Environment: Apple M5 Max, 64 GB unified memory, 18 CPU cores; CPython 3.12.14,
SQLite 3.53.1. Developer dependencies are in `requirements-dev.lock`.
Modal SDK 1.6.1 and its inspection dependencies are separately pinned in
`requirements-modal.lock`. No Torch, model weights, LLM inference, GPU job,
training or private-data upload was used. Coding-agent token spend was not
measured; bounded Luna workers handled implementation and one root reviewer
performed acceptance. Deterministic runtime commands need no LLM tokens.

Final adversarial regression: with schema version 1, a separate SQLite connection
with recursive triggers off could `INSERT OR REPLACE` a historical revision by
explicit rowid. The new test failed with `DatabaseError not raised`. Schema 2
uses `WITHOUT ROWID`, eliminating that alternative uniqueness key; the same
attempt now fails and both original revisions remain. Schema 1 databases are
rejected without modification. History is sorted by record ID/revision rather
than ingestion order. The full 32-test suite passed after this shared-storage
change. A final test-only refinement explicitly sets recursive triggers off;
the focused regression passed again. No old persistent private database required
a migration. The final benchmark was run separately from validation/build work.

## Next experiment

Freeze a larger reviewed synthetic challenge set. Compare a revision-pinned
semantic embedding model against BM25/hash/hybrid, using a bounded synthetic-only
Modal job to honor the owner's compute preference. Then implement the claim
schema and strict span/role validators and compare prompt-only extraction
before considering fine-tuning. An expensive training job needs a concrete
budget and owner approval. See [Modal policy](modal-compute.md).
