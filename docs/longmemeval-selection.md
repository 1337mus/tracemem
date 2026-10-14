# Selected-file retrieval diagnostic

`experiments.evaluate_longmemeval_retrieval.evaluate_retrieval_file` evaluates
one to five explicit question IDs from a local LongMemEval-format file. It first
uses the bounded file scanner to validate the complete source and confirm every
requested ID. It then checks the selected query lengths before it creates any
retrieval stores. The existing per-question retrieval method and metrics are
shared with the bytes API.

The CLI repeats `--question-id` for each selected question:

```sh
PYTHONPATH=src python -m experiments.evaluate_longmemeval_retrieval \
  --input /path/to/local-input.json \
  --namespace local-diagnostic \
  --variant s \
  --output /path/to/new-report.json \
  --question-id fictional-question-1 \
  --question-id fictional-question-2
```

With no `--question-id`, the CLI keeps its original whole-file path, 64 MiB
input cap, and schema 1 report. With selectors, it validates the entire file
under the scanner's regular-file, non-symlink, 4 GiB, per-question, question
count, nesting, and stability limits. The report uses schema 2. It has the
complete source hash and byte count, plus selection kind and source/evaluated
question counts. Question rows, counts, exclusion denominators, and aggregates
cover selected questions only. Requested selector order does not change source
order in the report. The report does not add a separate requested-selector list
or input path; selected question IDs remain in the question rows.

The scanner validates schema for every question. It permits an unselected query
over 4096 characters because that query is not sent to the retriever. A selected
query over 4096 characters is rejected before any retrieval store is created.
A malformed or schema-invalid unselected question still rejects the entire
file. Invalid input or selectors do not create an output file. Reports are
serialized before the existing exclusive `0600` output write.

Reports contain question IDs, record IDs, revisions, and citations, which may
be private. Keep reports from real inputs local. The report omits question,
answer, transcript, and namespace text. This prototype makes no claim about
upstream compatibility, data-use rights, QA accuracy, or benchmark score
reproduction. It performs no model inference or GPU work.
