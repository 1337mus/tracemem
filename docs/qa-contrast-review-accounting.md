# QA contrast review accounting

`experiments/analyze_qa_contrast_reviews.py` produces a deterministic, offline
accounting report for completed QA contrast response documents. It regenerates
the review key from the receipt and seed, validates every document with the
review preparer's validator, and keeps each reviewer in a separate report
section.

```sh
PYTHONPATH=src python3 -m experiments.analyze_qa_contrast_reviews receipt.json \
  --seed 17 \
  --responses reviewer-a.json \
  --responses reviewer-b.json \
  --output review-accounting.json
```

The output path must be new. The CLI writes a mode-0600 JSON file and rejects
existing, symlink, and input-alias paths. Response JSON uses the review
preparer's strict reader, so duplicate object keys and non-finite constants
are errors.

Each reviewer has a 46-task unique-prompt summary and a separate 72-cell
expansion. The cell rows preserve the regenerated task, prompt, question, and
cell IDs plus mode, ceiling, answerability, and both reviewer labels. Cell
counts are correlated when a prompt is reused. Mode/ceiling groups report
explicit all, answerable, and unknown denominators, label counts, and 3x3 joint
counts. The report is descriptive accounting and does not establish semantic
correctness, reviewer consensus, or independent trial estimates.

`reviewer_kind_self_attested` records only the value in the submitted response.
The report always sets `independent_human_reviewers_verified` to zero. It does
not include prompts, model outputs, or reviewer rationales. Provenance records
the seed, regenerated bundle and receipt hashes, the frozen literal-analysis
report hash, source hashes for this tool and the review preparer, and canonical
hashes of the response documents associated with each reviewer.

`authored_reference_human_label_count` describes the frozen fixture's reference
labels. It does not count self-attested human submissions. See the separate
[agent review results](qa-contrast-review-results.md) and [validation snapshot](validation-qa-contrast-review-accounting.json).
