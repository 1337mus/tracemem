# QA controls answer review preparation

This is a preparation-only workflow for a future qualitative review of the
fictional QA-controls answers. It creates a key-withheld packet and a blank
response template; it does not create labels, validate responses, score answers,
run inference, select a model, or authorize training.

## Current API and export command

The current source exposes
`prepare_reviews(receipt, *, seed) -> (bundle, key, blank_response)` and the
offline CLI options `--receipt`, `--output-dir`, and `--seed`. Example:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.prepare_qa_controls_reviews \
  --receipt examples/synthetic/qwen3-4b-qa-controls-answer-receipt.json \
  --output-dir .local/qa-controls-answer-review-seed-1729 \
  --seed 1729
```

Choose a fresh, nonexistent output directory. The command writes `bundle.json`,
`key.json`, and `response-template.json`; the directory is mode 0700 and files
are mode 0600. Existing paths are refused. The seed shuffles task order while
content-derived task IDs remain stable. The CLI is offline and must report
`remote_started: false`.

The fixed fictional receipt was exported with seed 1729. Root independently
matched all 36 visible tasks and hidden cell mappings against pre-export
expectations; every annotation field remains blank. Nine focused source and
nine installed-runtime tests passed. Installed execution outside the checkout
reproduced all three files byte-for-byte; all 37 runtime members matched source,
wheel and installation. The full source check passed 584/585 tests (one optional
Context Kit skip), Ruff on 129 files and mypy on 32 runtime files. Actionlint
passed. See the [validation snapshot](validation-qa-controls-answer-review.json);
exact delivery and CI state are recorded separately in `.local/active-cycle.json`.

## What the reviewer sees

The visible bundle has a version, kind, ordered tasks, and bundle hash. Each task
shows its question, exact actual prompt messages, full original fictional source
sessions for the question, and unchanged model output. It omits expected answers,
scores, variant/scenario/question/prompt IDs, and authored rationales. Source
message and session IDs remain visible as evidence references. The separate key
maps task IDs to the expected answers for both axes and includes identifiers and
provenance pins; keep it separate from the bundle.

This withholds the reference key, not the experimental condition. Actual prompt
messages and source context can reveal which condition a task represents; do not
claim condition blinding. Before an initial review, give a reviewer only the
bundle, blank response template, and rubric. Do not provide the key, separate
source fixture files, or prior metric reports. The source text is present inside
the bundle because it is required evidence. Packet preparation does not create
independent ground truth or prove reviewer independence.

## Two separate judgments

Complete both axes for every task; cite relevant message IDs in the rationale.

| Field | Choices | Evidence to judge |
| --- | --- | --- |
| `source_answer` | `correct`, `incorrect`, `uncertain` | Whether the answer or abstention is appropriate against the full original conversation. |
| `prompt_support` | `supported`, `unsupported`, `uncertain` | Whether the actual prompt supplied to the model supports the answer or abstention. |

The axes can differ: a correct answer can be an unsupported guess when the prompt
omits its evidence. Judge `UNKNOWN` separately against each evidence boundary:
the full source determines whether an answer exists, while the actual prompt
determines whether the response was supported. Consider modality, time and
attribution, and note unsupported added claims. Use `uncertain` when the evidence
does not justify a clear judgment.

The blank template has `task_id`, both judgment fields, `rationale`,
`reviewer_id`, `reviewer_kind`, and `timestamp`. Allowed reviewer kinds are
`human` and `agent`; future completed rows should identify the reviewer kind and
provide a timestamp. These fields are self-attested. This preparation slice does
not validate submitted responses, reviewer identity, timestamps, or labels.

## Scope and limits

The frozen fixture contains 36 tasks across six correlated dev groups and 12
questions. The test split is empty and there are zero human labels. A reviewer
packet is not an independent sample or truth source. Keep the separate key
withheld for any review described as reference-key-withheld; agent judgments
remain agent judgments. No automated judge is included. Do not infer semantic
accuracy, calibration, held-out performance, Gate A progress, model selection,
or training readiness from packet creation.
