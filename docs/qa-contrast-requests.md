# Offline token-budget QA requests

This preparation turns the verified [development packing cells](qa-contrast-token-replay.md)
into a frozen source-only answering payload, separate provisional references,
and a local comparison map. It performs no model inference or network calls.

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.prepare_qa_contrast_requests
PYTHONPATH=src:. .venv/bin/python -m experiments.prepare_qa_contrast_requests --output-dir .local/qa-contrast-requests-new
```

The default command prints a summary. The output option requires a new path
and writes `payload.json`, `references.json` and `audit.json` in a directory
with mode `0700`; files use `0600`. Existing files, directories and symlinks
are refused. A write failure can leave partial new output for diagnosis.

The preparer requires the exact canonical replay hash before consuming its
cells. The replay itself requires the frozen fixture, tokenizer receipt,
catalog, template and counting runner. These historical artifacts remain
unchanged. Complete messages are deduplicated by canonical SHA-256 with exact
message/count equality; every original cell retains its mapping.

| Preparation quantity | Value |
| --- | ---: |
| Fictional development questions | 6 |
| Comparison cells | 72 |
| Unique complete prompts | 46 |
| Reused cells | 26 |
| Input tokens over all cells | 28,944 |
| Input tokens over unique prompts | 23,533 |
| Maximum generated tokens over unique prompts | 5,888 |
| Unique input plus generation cap | 29,421 |

These are measured input counts and configured generation bounds, not billing
or latency measurements. Identical prompts deliberately share one future
answer. Their repeated cells are correlated and do not become independent QA
trials. A fixed seed and greedy decoding do not guarantee reproducibility
across different hardware or software.

The diagnostic model is Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, with float16, eager attention,
greedy decoding, one beam, seed zero, thinking disabled, no remote code,
and a 128-token output limit. The maximum complete input is 1,024 tokens;
the actual longest selected prompt has 907. The template and tokenizer pins
match the measured receipt. This freezes an answering diagnostic; it does
not select an extractor or training model.

Payload items contain only prompt identity, source/question messages and
measured input count. Provisional answers, authored evidence labels, scoring
and question/cell metadata stay in the separate reference/audit files.
Ordinary source passages may naturally contain answer words. The local map
preserves question, retrieval mode, token ceiling, selected citations and
scoring provenance so later answers can be compared across all cells.

All six scenarios are agent-authored development examples. The test partition
is explicitly empty. There are zero independent human labels. No QA accuracy,
calibrated abstention, model superiority or training readiness follows from
this preparation; Gate A remains unmet.

A separate remote runner must be implemented and reviewed against the exact
payload and source hashes before any inference. It must verify the personal
Modal workspace, keep one bounded job active, compare measured input counts,
and retain the raw response before teardown or validation.

Root verification compared all 46 complete prompt records, all 72 mappings
(including exact counts, selected citations/records and scoring fields), and
all six references against independently frozen expectations. The complete
model and measured tokenizer configuration match a separate protocol freeze.
Payload provenance distinguishes this request preparer's own source hash from
the historical catalog preparer's hash.

Full checks passed 502 tests (one skip), Ruff on 110 files, and mypy on 32
runtime files. Six focused installed tests passed outside the checkout; all
three JSON artifacts reproduced byte-for-byte, with all 37 runtime members
matching the existing wheel. No local model/tokenizer/Modal import occurred.
See the [validation receipt](validation-qa-contrast-requests.json).

The source-hash regression has a saved actual failing test and passing result.
The initial missing-API failure was observed by the implementer but its raw log
was not saved; no historical RED receipt was reconstructed.

Frozen canonical payload SHA-256:
`80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e`.
