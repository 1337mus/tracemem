# Bounded turn-retrieval diagnostic

Root architecture, 2026-10-14. Follow `docs/longmemeval-adapter.md`,
`docs/benchmark-contracts.md`, and `docs/longmemeval-rights.md`. No benchmark data,
network, model/GPU calls, dependencies, upstream execution, or original-data writes.

Worker owns ONLY `experiments/evaluate_longmemeval_retrieval.py`,
`tests/test_longmemeval_retrieval.py`, `docs/longmemeval-retrieval.md`, and `.local/`
logs. Do not modify the adapter, existing fixture, CI, receipts, or other docs.
No commits/pushes/subagents. Root reviews the final diff and runs full validation.

Implement `evaluate_retrieval_bytes(raw: bytes, *, namespace: str, variant: str,
cutoffs: tuple[int, ...] = (5, 10, 50)) -> dict[str, Any]` and a small module CLI.
Use the approved bytes adapter. Validate cutoffs before any SQLite creation:
nonempty tuple, unique strict integers (not bool), each 1..100; preserve order.
Reject question strings longer than Retriever's 4096-character limit before any
SQLite creation, with a sanitized ValueError. Never silently truncate queries.

For each question, create its own temporary EvidenceStore, ingest only that
question's records, construct default Retriever, and query each fixed mode
bm25/vector/hybrid once at max(cutoffs). Score each prefix. No supplied embedder
or remote backend: vectors are the existing weak signed-token-hash baseline.
Temporary databases close before their temporary directories are cleaned up.

Measure turn-label retrieval only. For each question/mode/cutoff, report returned
record IDs/revisions/citations, number of explicit answer_record_ids, found label
count, label recall fraction, and all-labeled-turns-found. `_abs` questions always
have null scores and exclusion reason `abstention`; non-abstention questions with
no explicit turn labels have null scores and reason `missing_turn_labels`.
Preserve both categories in all counts. Never infer labels from answer_session_ids.
For scored questions only, aggregate macro label recall and fraction with all
labels found; a zero denominator yields null, not zero or a perfect score.
Also report counts of questions/abstentions/unlabeled/scored and nonempty retrieval
for all questions. Keep model/mode and cutoff cells separate.

Report schema version, exact raw-input SHA256, variant, cutoffs, fixed mode list,
Python/SQLite versions, evidence schema, embedding metadata/fingerprint and RRF k.
No latency fields needed. Do not serialize raw questions, answers, source text,
session dates, or namespace text. Question IDs and evidence identifiers remain
potentially private; docs must warn that real-input reports stay local. A fixture
result measures protocol labels only: the existing fixture has deliberate unrelated
answer/flag sentinels and is not gold QA. No nDCG, session metrics, QA model, or
upstream-equivalence claim in this slice.

CLI: `python -m experiments.evaluate_longmemeval_retrieval --input ...
--namespace ... --variant s --output ...`. Read a regular file with at most
64MiB+1 bytes before calling adapter; do not read a huge release unboundedly.
Use fstat on the opened descriptor for regular-file validation. Source read-only.
Output JSON is serialized before exclusive output creation, mode0600, refuse any
existing path including symlinks. Errors must not dump source/query/answer content.
No new output on invalid inputs; do not claim atomicity against disk-write failures.

Use meaningful TDD: run one new behavior test RED before implementation, then
GREEN. Tests must exercise real SQLite/Retriever, no mocked ranking. Include partial
and complete multi-label retrieval, empty/no-label/abstention denominators, corpus
isolation with positive controls, reference-only edits leaving rankings unchanged,
punctuation queries, Unicode citations, cutoff/query bounds, deterministic replay
apart from recorded environment, input preservation, CLI exclusive-output/symlink
behavior and mode0600. Inline newly authored fictional records are allowed; do not
change existing fixtures or report a real benchmark result. Run focused tests,
Ruff/format and scoped strict mypy; return exact evidence and release ownership.
