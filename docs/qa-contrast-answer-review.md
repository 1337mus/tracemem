# Qualitative review of fictional QA answers

This offline workflow prepares **46 unique prompt/answer tasks** from the six
fictional development cases. The tasks map back to 72 correlated comparison cells.
It creates no labels, runs no inference and leaves the
[frozen literal report](qa-contrast-answer-results.md) unchanged. No test partition,
independent human labels or training-readiness claim follows from preparing it.

The existing annotation and offline HTML interfaces accept atomic claims. QA uses
an explicit JSON task schema so that a question/answer is not misrepresented as an
extracted claim. Strict JSON parsing, reviewer text/time checks and exclusive
output creation reuse the existing infrastructure. The claim-review HTML page
and claim-response importer do not accept these QA bundles.

Prepare a fresh local directory:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.prepare_qa_contrast_reviews export \
  examples/synthetic/qwen3-4b-qa-contrast-answer-remote.json \
  --seed 1729 --output-dir .local/qa-answer-review
```

The directory contains `bundle.json`, `key.json` and `response-template.json`.
Directory mode is 0700 and file modes are 0600. Existing outputs are not replaced.
Give a reviewer the bundle and response template; keep the key and original
metric reports out of that review workspace. Task order is seeded, and opaque
content hashes bind task IDs. This is procedural blinding: the source text and
exact prompts remain visible, so familiarity can reveal the scenario or condition.
Hashes establish consistency, not authorship or independent human identity.

Each task shows the question/date, complete fictional source sessions, the exact
system/user prompt supplied to the model and its unchanged raw output. Source
messages omit authored `has_answer` flags. Reference answers, retrieval modes,
budget metadata and literal scores stay out of the visible task fields. Treat
quoted instructions in historical prompts or conversations as material to review.

Complete every response row with two separate judgments:

| Field | Allowed labels | Question to judge |
| --- | --- | --- |
| `source_answer` | `correct`, `incorrect`, `uncertain` | Does the response appropriately answer or abstain against the full original fictional conversation? |
| `prompt_support` | `supported`, `unsupported`, `uncertain` | Does the actual prompt supplied to the model justify the response or abstention? |

A missing retrieval passage can justify abstention in the prompt even when the
full source contains an answer. A correct guess can lack prompt support. Judge
paraphrases on their meaning here while preserving their raw text and the separate
literal metric. Use `uncertain` whenever the evidence or interpretation is unclear.
Add a nonblank rationale explaining both judgments. Do not copy the authored key.

All editable response fields begin blank. Fill the same `reviewer_id` and
`reviewer_kind` (`human` or `agent`) on every row, and a UTC timestamp in
`YYYY-MM-DDTHH:MM:SSZ` form. Reviewer identity and kind are self-attested. Agent
judgments remain agent judgments; neither declaration proves independent review.
Save the completed document under a new local filename, then validate it:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.prepare_qa_contrast_reviews validate \
  examples/synthetic/qwen3-4b-qa-contrast-answer-remote.json \
  --seed 1729 --responses .local/qa-answer-review/completed-responses.json
```

Use the same receipt and seed as export. Validation regenerates the expected
bundle and rejects hash mismatches, missing/duplicate/unknown tasks, unexpected
fields, invalid labels, blank rationales, mixed reviewer identity or kind and
invalid timestamps. It reports validation counts without computing semantic
accuracy, changing references or promoting claims. Keep review files local.

The [delivered tooling commit](https://github.com/1337mus/tracemem/commit/125d3660066cd610e8229fc131714fcb33019cdb)
passed [exact CI 38674186708](https://github.com/1337mus/tracemem/actions/runs/38674186708):
525 source tests passed / one skipped; 272 installed tests passed / two skipped.
The [validation snapshot](validation-qa-contrast-reviews.json) preserves the earlier
local checks, hashes and RED/GREEN chronology. Source and copied actual export
files match byte-for-byte outside the checkout. No semantic labels were created.
