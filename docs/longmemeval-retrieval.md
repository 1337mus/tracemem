# LongMemEval turn-retrieval diagnostic

`experiments.evaluate_longmemeval_retrieval` runs the pinned bytes adapter and
the repository's default local retriever against one question at a time. It
reports retrieval of explicit `has_answer: true` turn labels. It does not run a
question-answering model or the upstream benchmark evaluator.

The command accepts a local JSON file up to 64 MiB and writes a new report with
permissions set to `0600`:

```sh
PYTHONPATH=src python -m experiments.evaluate_longmemeval_retrieval \
  --input /path/to/local-input.json \
  --namespace local-diagnostic \
  --variant s \
  --output /path/to/new-report.json
```

The output path must not already exist, including as a symlink. Input must be a
regular file. The input is opened read-only and bounded to 64 MiB plus one byte
to detect oversized files. The input is never modified. The report is serialized
before the output file is exclusively created; a disk error during a later write
can leave a partial output.

Each question gets a fresh temporary `EvidenceStore` containing only that
question's turns. A default `Retriever` performs one search each in `bm25`,
`vector`, and `hybrid` mode at the largest requested cutoff, which defaults to
5, 10, and 50. The vector mode uses the repository's deterministic signed-token
hash embedder, a weak lexical baseline rather than semantic embeddings. The
report records its metadata and fingerprint, the reciprocal-rank-fusion `k`,
cutoffs, SQLite and Python versions, and the evidence schema version.

Turn-label recall is computed only from explicit `answer_record_ids`, which the
adapter derives from `has_answer: true`. Session labels are not expanded into
turn labels. Abstention IDs ending in `_abs` and questions without explicit turn
labels remain in the question and retrieval counts but have null recall and
all-labels-found scores with an exclusion reason. Aggregate recall is the
macro-average over scored questions; when there are no scored questions, the
aggregate scores are null.

Reports contain question IDs and returned record IDs, revisions, and citations.
These identifiers may be private, so reports produced from real inputs must
stay local and must not be committed or uploaded. The report omits question and
answer text, transcript text, session dates, and namespace text. The input hash
still permits matching a report to a specific local file.

The repository's synthetic fixture contains deliberate answer and evidence-flag
sentinels to exercise metadata separation. Those fields are not gold QA targets
or semantic-support annotations. Results from the fixture measure only the
turn-label retrieval protocol. This diagnostic makes no nDCG, session-retrieval,
QA, or upstream-equivalence claim. Real LongMemEval dataset rights and full
release compatibility remain unverified; see the [rights note](longmemeval-rights.md)
and [adapter notes](longmemeval-adapter.md). The S and M releases exceed this
tool's input cap.
