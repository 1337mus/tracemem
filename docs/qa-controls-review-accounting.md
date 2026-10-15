# QA controls response validation and accounting

This offline tool validates completed copies of the controls response template
and reports each reviewer's labels separately. It does not judge answers,
authenticate reviewers, pool consensus, or assign semantic accuracy.

The API is `validate_responses(receipt, response_document, *, seed)` and
`analyze_reviews(receipt, response_documents, *, seed)`. The preparer source is
pinned and the exact bundle is regenerated from the accepted receipt and seed.
Complete the existing response template in place, retaining `allowed_choices`.
Every task needs both labels, a nonblank rationale, reviewer ID, reviewer kind,
and canonical UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).

Validation rejects changed metadata or bundle hashes, unknown or duplicate task
IDs, missing rows, invalid labels, blank or oversized text, invalid timestamps,
and mixed reviewer IDs or kinds within a document. Reordered response rows are
accepted and returned as detached records in bundle order. Duplicate reviewer
IDs across documents are rejected.

Each report retains `uncertain` labels and gives the two label counts and their
3-by-3 cross-tabulation per reviewer and variant. The three variants each have
12 tasks; each reviewer has 36 tasks across six correlated groups. Cell IDs,
question IDs and prompt IDs remain available for mapping audits. Report cells
omit rationales, authored answers and raw model outputs. Canonical response
hashes and source, bundle, receipt, result and literal-report pins identify the
exact inputs. A reordered document can have the same counts and a different
canonical input hash.

`reviewer_kind_self_attested` reports the supplied `human` or `agent` string.
The tool always reports zero independently verified human reviewers. Structural
validation cannot establish who performed the work or whether a label is right.
Reports are provisional dev-only accounting; `test` is empty. A procedural
fixture that declares `human` does not create a human label.

## Offline command

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.analyze_qa_controls_reviews \
  --receipt examples/synthetic/qwen3-4b-qa-controls-answer-receipt.json \
  --responses /path/to/completed-response-template.json \
  --seed 1729 \
  --output .local/qa-controls-review-accounting.json
```

Repeat `--responses` for distinct reviewer IDs. Match the seed used to export
the packet. The output must be a new path; inputs and existing or symlinked
outputs are preserved. Strict JSON parsing rejects duplicate keys and nonfinite
constants. Output files are mode 0600. All inputs are validated before the
report is written. The CLI reports `remote_started: false`.

## Verification scope

Root prepared index-assigned labels before receiving the candidate analyzer.
The audit matched all 72 rows, six variant tables, separate marginal and joint
counts, and canonical provenance across two procedural reviewer streams. It
rejected 27 malformed or blank cases, including the untouched actual 36-task
blank template. These fixtures test plumbing only: no model inference,
semantic annotation or human review occurred. Seven focused tests passed.
See the [validation snapshot](validation-qa-controls-review-accounting.json)
for final source and installed checks, and `.local/active-cycle.json` for exact
delivery and CI. Original receipts and literal metric reports remain unchanged.

The [packet preparation protocol](qa-controls-answer-review.md) describes the
review evidence boundaries and rubric. Collect and audit genuine review work
separately before drawing semantic conclusions. Gate A remains unmet; no model
has been selected or trained.
