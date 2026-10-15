# QA controls answer review results

These are two-axis labels for one reference-key-withheld agent review and Root's
later unblinded derivative over the frozen fictional answer run. They are
provisional judgments, not independent truth or consensus. No new inference ran.

## Scope and counts

The packet covers 36 tasks: 12 questions across six correlated development
groups. The test split is empty; there are zero human labels and zero
independently verified human reviewers. Labels keep full-source correctness
separate from support in the actual prompt. Counts below use `correct / incorrect
/ uncertain` for the source axis and `supported / unsupported / uncertain` for
the prompt axis.

| Variant | Agent source | Agent prompt | Root source | Root prompt |
| --- | ---: | ---: | ---: | ---: |
| `full_all_message_ids` | 11/1/0 | 12/0/0 | 11/1/0 | 11/1/0 |
| `support_omitted` | 2/10/0 | 12/0/0 | 2/10/0 | 12/0/0 |
| `empty` | 2/10/0 | 12/0/0 | 2/10/0 | 12/0/0 |
| **Total** | **15/21/0** | **36/0/0** | **15/21/0** | **35/1/0** |

Root changed exactly one task's prompt-support label from supported to
unsupported; its source-answer label remained incorrect. The agent rationale
said the full conditional-item prompt was empty. It explicitly supplied a
ceramic wedge if the vent stays open; the question asks which item is needed
under that condition, not whether the outcome occurred. This is an unjustified
abstention against the actual prompt. Root preserved the original labels and
rationale. Root reconstructed all 36 actual contexts and reviewed every agent
rationale; the streams agree on 35 of 36 paired labels, with source-answer
labels matching on all 36. Root had already seen prior reports, so the
derivative is neither independent nor consensus.

## Reading the two axes

In the support-omitted and empty variants, `UNKNOWN` can be supported by the
actual prompt while being incorrect against the full source when that source
contains an answer. Conversely, an answerable question can make `UNKNOWN`
incorrect against the full source. The conditional-item adjudication shows that
`UNKNOWN` can also be unsupported when the actual prompt contains the requested
conditional fact, even without an observed outcome.

The agent and Root each labeled 11/12 full-context answers source-correct and
2/12 in each reduced-context variant correct. Nine full-context paraphrases were
judged correct even though punctuation or added wording failed the frozen
literal matcher (which recorded 6/36 full-source exact matches). These are
reviewer judgments on fictional dev data, not a validated semantic accuracy
estimate. Do not infer generalization, model selection, training readiness, or
Gate A progress.

## Reproduction and provenance

The raw responses and reports remain local under `.local/`; this page records no
raw answer text. Recompute the separate accounting report with the tracked
receipt and a fresh output path:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.analyze_qa_controls_reviews \
  --receipt examples/synthetic/qwen3-4b-qa-controls-answer-receipt.json \
  --responses .local/qa-controls-agent-review-v1.json \
  --responses .local/qa-controls-root-review-v1.json \
  --seed 1729 \
  --output .local/qa-controls-semantic-source-report-replay.json
```

| Local artifact | File-byte SHA-256 |
| --- | --- |
| `.local/qa-controls-agent-review-v1.json` | `372fc6d4e7d4309b9d4025da4d7476ce828d4a56a16725b513b2d25d84fe4bb5` |
| `.local/qa-controls-root-review-v1.json` | `6fb4afbc6416348469cebe9c765787b0c976de55ee770ff62137dff1f81b7e96` |
| `.local/qa-controls-semantic-source-report.json` | `8c6981337ca282e907cfb707fc7c960d14f5f4a767b8a5b5f1e754cb847a4054` |
| `.local/qa-controls-semantic-root-adjudication.json` | `f3b979c73cbe2babcea82c778774a21b435102693b8878d1af7a7239d5f5e171` |

Semantic report canonical hash `40dfd4ba16b0a35b5f70a9c13dad9803e8a9d5f18c7647a9dafcb509ed457dd7`;
bundle hash `0ac1c3bde75773b6dd7f0c521b36635973317a0f9dc71b584af3a02a1242541a`.
Procedure and reading-view receipts remain at
`.local/qa-controls-agent-review-v1-procedure.json` and
`.local/qa-controls-review-reading-view-receipt.json`; the view reconstruction
was exact and excluded the reference key. See [response accounting](qa-controls-review-accounting.md)
and [packet protocol](qa-controls-answer-review.md).

The review and report are dev-only. Installed replay reproduced the 54,259-byte
report byte-for-byte (SHA-256 `8c6981337ca282e907cfb707fc7c960d14f5f4a767b8a5b5f1e754cb847a4054`);
no new answering-model or GPU run occurred. See the [validation snapshot](validation-qa-controls-review-results.json)
for source and installed verification; exact delivery and CI state is in
ignored `.local/active-cycle.json`. Gate A remains unmet.
