# SFT quote-v2 results

Updated: 2026-10-15

The second synthetic-only QLoRA run completed. It improved output formatting,
reference-string matches, and empty-source abstention on a small development
slice. Those results remain diagnostic: strict raw scoring, post-hoc fence
normalization, and agent-only semantic review answer different questions.
No production checkpoint was selected, and Gate A remains unmet.

## Run and data

The run fine-tuned `Qwen/Qwen3-4B` at revision
`1cfa9a7208912126459214e8b04321603b3df60c`. It consumed 768 fictional rows
from 48 scenario families with 16 correlated variants per family. The run used
training seed 17 for one epoch; the scenario expansion seed was 23. It
completed 192 optimizer steps on one A10G container,
with zero retries and a 3,600-second timeout. The 24 development families form
a small adaptive development slice; it is not held out. The private test data
was neither read nor used.

Training took 901.998 seconds, the remote worker ran for 1,169.920 seconds,
and the client call took 1,184.560 seconds. The adapter is 132,187,888 bytes
with SHA-256 `3b30cbf98a685792734d8c16ffd7f31e000707f8f2b72a74e5ca9bc364214093`
in the owner's personal private Modal volume at
`tracemem-sft-quote-v2/adapters/quote-v2-20261015-01/adapter`. No model weights
were downloaded to this Mac. The worker-resource estimate is about `$0.50`,
excluding startup and storage; actual billing was not available. The
recorded post-run inventory at 2026-10-15 23:01 UTC showed both apps stopped
with zero active tasks.

## Strict raw output scoring

The frozen evaluator reads the raw response text without repairing its
presentation. Markdown-fenced JSON made 23 of 24 base responses fail strict
parsing. All tuned responses parsed. Citation integrity checks exact source,
quote, and explicit-date validity; it does not determine whether a claim is
semantically supported.

| Strict raw metric | Base | Tuned |
| --- | ---: | ---: |
| Parse failures | 23 / 24 | 0 / 24 |
| Intact citations / checked claims | 1 / 1 | 19 / 19 |
| Reference-string matches / expected strings | 0 / 19 | 2 / 19 |
| Correct empty-source abstentions / empty-source cases | 0 / 6 | 6 / 6 |

The tuned output recovered two of 19 authored reference strings. This small
sample does not measure recall over all eligible facts.

## Separate post-hoc fence-only analysis

A separate analysis removed only the exact outer JSON code fence from the
existing base outputs. It changed no model response content, did not alter the
frozen receipt or strict results, and did not assess claim meaning.

| After fence-only normalization | Base | Tuned |
| --- | ---: | ---: |
| Parse failures | 0 / 24 | 0 / 24 |
| Intact citations / checked claims | 24 / 26 | 19 / 19 |
| Reference-string matches / expected strings | 0 / 19 | 2 / 19 |
| Correct empty-source abstentions / empty-source cases | 1 / 6 | 6 / 6 |

The tuned values are unchanged from strict scoring. Fence normalization repaired
presentation for the base output but did not establish semantic support. The
analysis script remains local and was not copied into the repository.

## Agent-only semantic review

One fresh-context blinded agent read all 48 readable raw responses and judged
45 generated claims. The blind-packet builder strict-parsed the outputs and
marked the 23 fenced base responses as `unreviewable`, leaving their
`generated_claims` empty. The reviewer still inspected all 48 raw-text values,
including fenced JSON, and reviewed all 45 claims. That parser status describes
packet construction; it does not mean those raw outputs were semantically
unreadable.

| Semantic review outcome | Base | Tuned |
| --- | ---: | ---: |
| Supported claims | 18 | 17 |
| Unsupported claims | 8 | 1 |
| Uncertain claims | 0 | 1 |
| Fully supported nonempty responses | 16 | 16 |
| Correct empty responses / empty-source cases | 1 / 6 | 6 / 6 |

Root then reviewed all source messages, candidate claims, and 45 agent
judgments unblinded and accepted every label. The base unsupported count
includes summaries of questions, assistant content, or tool content that
violate the extraction policy; those summaries are not necessarily false
descriptions of the dialogue. This is an adjudication of an agent review, not
human ground truth or an independent reviewer. Formatting can reveal which arm
produced an output. The tuned output invented unsupported
chronology in D14; the keeper/subject attachment in D21 remains ambiguous and
was labeled uncertain. D01 and D03 showed better cause or attribution handling.

These labels are provisional. Claim counts do not establish recall because
claim segmentation differs and the review did not exhaustively enumerate every
fact in the conversations. There were zero human reviewers, no generalization
claim, and no production checkpoint selection.

## Artifacts and offline replay

The frozen job, remote receipt, blind review inputs, protocol, and adjudication
are preserved as exact fictional JSON copies. Their SHA-256 hashes and local
source paths are recorded in the [aggregate results](sft-quote-v2-results.json)
and [validation record](validation-sft-quote-v2.json). The original receipts
remain unchanged under `.local/sft-quote-v2-20261015/`.

- [Job](../examples/synthetic/sft-quote-v2-job-v1.json) and [remote receipt](../examples/synthetic/sft-quote-v2-remote-v1.json)
- [Blind packet](../examples/synthetic/sft-quote-v2-blind-packet-v1.json), [blind review](../examples/synthetic/sft-quote-v2-blind-review-v1.json), and [curator map](../examples/synthetic/sft-quote-v2-curator-map-v1.json)
- [Semantic protocol](../examples/synthetic/sft-quote-v2-semantic-protocol-v1.json), [root adjudication](../examples/synthetic/sft-quote-v2-root-adjudication-v1.json), and [fence-only analysis](../examples/synthetic/sft-quote-v2-fence-analysis-v1.json)

Recreate the raw development report on CPU. The evaluator reads the frozen
job, receipt, and development fixture; it makes no model or GPU call. Choose a
new output path because the evaluator creates its output exclusively.

```sh
PYTHONPATH=src:. .venv/bin/python -m training.evaluate_quote_sft \
  --job examples/synthetic/sft-quote-v2-job-v1.json \
  --receipt examples/synthetic/sft-quote-v2-remote-v1.json \
  --fixture datasets/sft-quote-v2/dev.json \
  --output .local/replay-v2/quote-v2-dev-report.json
```

## Next research step

Strengthen annotations and make paired chronology and subject-attachment
corrections using distinct scenario families and a separate frozen test. Avoid
assuming that more training steps will address the observed errors.
