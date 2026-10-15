# Frozen-five oracle turn-retrieval diagnostic

Measured 2026-10-15 UTC, locally and without model, tokenizer, network or GPU
calls. Selection was the same first five source-order questions frozen before
the earlier format inspection. No question replacement or retrieval tuning
followed inspection. This is a small engineering diagnostic, not an official
benchmark score, QA evaluation or held-out test.

The pinned oracle object and rights limits are described in
[format compatibility](longmemeval-compatibility.md) and the historical
[rights audit](longmemeval-rights.md). All 500 questions passed file validation;
only five were retrieved and scored. Each selected question had two explicit
`has_answer=true` turn labels. All five were scored; abstention and missing-label
exclusions were both zero. Missing labels remain exclusions in the evaluator.

| Retrieval | Cutoff | Macro explicit-label recall | Questions with all labels / 5 |
| --- | ---: | ---: | ---: |
| BM25 | 1 | 0.50 | 0 |
| BM25 | 3 | 0.60 | 1 |
| BM25 | 5 | 0.80 | 3 |
| Hash vectors | 1 | 0.10 | 0 |
| Hash vectors | 3 | 0.60 | 2 |
| Hash vectors | 5 | 0.80 | 3 |
| Hybrid | 1 | 0.40 | 0 |
| Hybrid | 3 | 0.80 | 3 |
| Hybrid | 5 | 0.80 | 3 |

Every cell returned nonempty results for all five questions. The vector backend
is the existing 384-dimensional `SignedHashEmbedder`: weak lexical features,
not neural semantic embeddings. Hybrid uses reciprocal rank fusion with
`rrf_k=60`. The nine cells reuse five questions and are correlated. Differences
do not establish a winning method or general retrieval quality. Oracle histories
do not represent the distractor-heavy full S/M releases; neither was evaluated.

## Verification

The existing `experiments/evaluate_longmemeval_retrieval.py` ran from outside
the checkout using the previously reviewed installed wheel. All 37 runtime
members matched the frozen source manifest. Reports from selected-file evaluation
and a transient first-five bytes clone agreed on question outcomes, aggregates,
environment, embedding and schema metadata. Source hash, size, mode and mtime
were unchanged before and after execution.

A separate checker recomputed outcomes, denominators, exclusions, prefix
consistency and aggregate scores, and checked question membership, revisions,
duplicate hits and canonical citations. Four fictional controls passed; the
bad-aggregate test failed against a callable no-op checker, and provenance
negative controls failed when their guards were removed. Root independently
reconstructed record IDs, revisions and explicit labels directly from raw
source fields, without adapter calls, then rechecked both reports' citations
and aggregate metrics. Raw IDs, reports and source content remain local only.

The runner reached its final passed receipt in 1.668904 seconds, including
parsing, duplicate evaluation and verification. This is not standalone query
latency. The outer `/usr/bin/time -l` command exited 1 after macOS denied
`sysctl kern.clockrate`; peak RSS is unavailable. Its stdout, stderr and launch
intent were retained, and the benchmark was not rerun to obtain cleaner logs.
See the aggregate-only [validation receipt](validation-oracle-retrieval.json).

No QA accuracy, semantic evidence sufficiency, data-use rights, model selection,
training readiness or Gate A conclusion follows. The next useful evaluation
work is a new development-only fictional fixture where rankings and context
budgets produce different evidence, with the protocol frozen before inference.
Existing test results remain unchanged.
