# Selected-file retrieval diagnostic

Root architecture after bounded file reader fcfd1da. Implement a narrow connection
between the reviewed file scanner and the existing retrieval diagnostic. No model,
network, benchmark body download, private input or GPU calls. This is a prototype;
no QA/upstream-equivalence/data-rights claim. Keep existing bytes API, report and
CLI without selectors unchanged. Runtime package sources remain unchanged.

Worker owns only `experiments/evaluate_longmemeval_retrieval.py`,
`tests/test_longmemeval_selection.py`, `docs/longmemeval-selection.md`, and logs
`.local/longmemeval-selection-*`. No existing tests/config/other docs/package files,
staging/commit/push. Root owns integration, independent oracle and delivery.
Production delta under200LOC, total production module under500LOC; propose a
partition before exceeding it. Genuine incremental RED/GREEN with a callable stub
for missing API, not missing imports as evidence. Preserve exact RED/GREEN logs.

Add `evaluate_retrieval_file(path, *, namespace, variant, question_ids,
cutoffs=_DEFAULT_CUTOFFS)`. Require a tuple of1..5explicit questionIDs: validate
nonempty tuple and cutoffs before file I/O, then reuse scanner validation and
full-file-before-return behavior. Do not implement another reader/parser. Use
`scan_longmemeval_file` and its selected objects, exact full raw input hash/byte
count and total count. No retrieval/store creation until the entire file and all
selected query lengths (4096characters) validate. At most five selected questions
are evaluated in source order. Unselected overlong queries are schema-valid and
not evaluated; state that limit. Late malformed/schema-invalid items always refuse.

Extract the existing shared scoring/report body into one private helper over the
parsed questions, raw input SHA and existing options. No metric/ranking/policy
change, no duplicate scoring logic. Bytes report stays schema1 with exact prior
fields. Selected-file report is schema2, with same metrics and `input_sha256` for
the COMPLETE input; add `input_bytes` and `selection` containing
`kind: explicit_question_ids`, `source_question_count`, `evaluated_question_count`.
All `counts`, rows, exclusion denominators and aggregates refer only to selected
questions. Do not emit requested selector order, input path, namespace or new text
fields. Existing IDs/citations and hashes are sensitive; docs keep reports local.

CLI adds optional repeatable `--question-id`. If absent, old64MiB whole-input path
and schema1 report remain unchanged. If present, use file API and scanner's4GiB /
per-item / count / metadata / regular-nonsymlink / stability bounds. Invalid input
or selection creates no output. Existing exclusive0600output path is reused only
after report fully serialized. Static ValueError/parser errors omit source paths,
text and selectors. No source writes, persistent corpus, spool, inference or GPU.
Do not widen the bytes parser or its existing CLI default cap.

Tests: compare selected-file rows/metrics to bytes evaluator on exactly selected
fictional items, accounting only for full-file hash/new metadata/schema; reversed
requested order keeps source order; whole-source hash includes unselected bytes;
selected counts/abstention/unlabeled/null denominators; selected/unselected private
sentinels absent from reports; empty/duplicate/missing/>5/non-tuple selectors,
invalid cutoffs and selected overlong query all refuse before store calls; late
invalid unselected item creates no store/output; valid unselected overlong query
can be omitted. End-to-end CLI selective mode with real >64MiB fictional whitespace
suffix, exact hash/bytes and0600/exclusiveoutput/source invariance; default CLI
still refuses that file. FIFO/symlink/errors/nooutput checks as needed. Use real
retriever for meaningful metric/citation comparisons and a fail-if-called store
probe only to prove preflight. Focused existing retrieval plus new tests, Ruff and
established explicit-package-bases strict mypy. Release complete files and hashes;
root reviews before any push. Finish with deliverable or precise blocker, no self-wait.
