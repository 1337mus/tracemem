# QA interpretation controls

## Purpose and scope

The earlier fictional QA review exposed two distinct failure patterns: reading a requested destination time as the clock time when the request was sent, and answering `UNKNOWN` when the supplied evidence already contained the answer. The control fixture also includes source omissions so abstention can be compared with answer interpretation.

This is a synthetic, dev-only control design. It does not compare retrieval modes or fixed-token budgets. System and user prompt templates remain unchanged; the fixture varies which source messages reach each prompt. No token counts have been measured, and no model output or model score exists; no GPU or model run has occurred.

## Fixture

The pinned v1 fixture is `examples/synthetic/qa-interpretation-controls-dev-v1.json`, SHA-256 `09698c216d3f8f9751383bef8e984bf4385701b7e199fa7413b70f23fae31d53`. It contains six correlated scenario groups, seven sessions, 22 messages, 12 questions and 36 prompt cells. There are 10 answerable questions and two full-source unknowns. For every question, the support-omitted and empty variants expect `UNKNOWN`, for 24 unknown prompt cells.

The six controls cover destination time versus request-send time, two separately stated times, an assistant proposal versus the user’s later choice, a snack versus its container, a conditional repair without a reported outcome, and user packing commitments versus an assistant suggestion. The v1/v2 candidate files and notes under `.local/` are historical drafts; v2 fixes the post-proposal choice wording. These controls were deliberately derived from prior dev errors and template reuse. They are not independent new held-out scenarios and do not advance Gate A.

The fixture and reference labels are agent-authored and provisional. Root read all source messages and questions, then independently reconstructed every prompt and reference mapping. That is not independent human annotation. There are zero human labels. Model-facing items contain only prompt IDs and messages; authored answers and labels remain in a separate local key. Preparation records fixture, source, payload and key hashes before any model answer is generated.

## Literal baseline and interpretation

The full-source answer reference and expected answer from the supplied passages are kept separate. The deterministic always-`UNKNOWN` baseline is not a model run. The frozen scoring policy uses normalized exact literal comparisons after case-folding and whitespace normalization; `UNKNOWN` must match exactly after that normalization. A literal match does not establish semantic correctness or support.

| Always-UNKNOWN reference baseline | Literal prompt-answer matches |
| --- | ---: |
| Full source | 2/12 |
| Support omitted | 12/12 |
| Empty | 12/12 |
| Total | 26/36 |

Against the full-source answer axis, always-`UNKNOWN` scores 2/12 for each variant and 6/36 overall because only the two questions whose source answer is unknown match. The prompt-axis total is skewed by all omitted and empty cells expecting `UNKNOWN`; it is not a pooled measure of model quality. Literal scoring is separate from any future semantic review.

## Preparation and next step

Reproduce the frozen preparation in a new local directory:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.prepare_qa_interpretation_controls \
  --output-dir .local/qa-interpretation-controls-v1
```

The command writes `payload.json`, `references.json` and `audit.json` using exclusive creation with a mode-0700 directory and mode-0600 files. It prints only hashes and counts. Use a new output path for each replay. Root independently checked all 36 prompts/cells; nine focused tests passed against the installed runtime with model imports blocked, and all three replayed artifacts matched byte for byte. The full check passed 542 tests with one skip, Ruff and mypy. Exact artifact hashes and checks are in the [validation snapshot](validation-qa-interpretation-controls.json); delivery is recorded separately. No inference has occurred.

Before any bounded synthetic-only inference, review the exact source and model-facing payload, then verify the active personal Modal workspace and take a fresh inventory. If authorized to proceed, use one GPU call, keep model weights off the Mac, and do not train. Review generated outputs separately from the frozen literal baseline.

The earlier authoring candidates remain unchanged under `.local/`. The tracked v1 fixture and prepared artifact hashes define this freeze. See the [prior QA review results](qa-contrast-review-results.md) and [review accounting](qa-contrast-review-accounting.md) for the source of the motivating dev observations.
