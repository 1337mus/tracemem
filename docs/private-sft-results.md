# Private SFT development comparison

Updated 2026-10-17 UTC.

One private continuation run compared the pinned base model, the existing
adapter, and the trained continuation on the same 16 development candidates.
The run used 30 private rows from two sessions plus 48 fictional replay rows.
Development covered 16 candidates in eight windows across two other sessions;
all seven useful references came from one of those sessions.

## Run summary

| Measure | Result |
| --- | ---: |
| Training rows | 78 (30 private, 48 fictional) |
| Microsteps | 78 |
| Optimizer steps | 20 |
| Input tokens | 39,538 |
| Supervised tokens | 894 |
| Trainable parameters | 33,030,144 |
| Training time | 98.017 seconds |
| Remote runtime | 241.949 seconds |
| Client elapsed time | 260.116 seconds |

## Strict output and offline content diagnostic

The original raw receipt and strict-literal report are preserved unchanged.
That report counts 0/16 outputs valid in each arm because decoded response text
includes the terminal EOS token. Before the separate offline diagnostic, the
tokenizer EOS ID, generation finish reason, and final output token were checked:
the ID was 151645, the finish reason was `eos`, and the final output ID was
151645.

The diagnostic removed exactly one terminal EOS token from a copy of each
response before content validation. It left interior content, original output
token IDs and hashes, the receipt, and the strict-literal report unchanged. It
did not rerun inference. The table keeps the strict and content-only validity
findings separate:

| Arm | Strict-literal valid | Offline content-only valid | Supported retained | Unsupported rejected | Useful accepted | Uncertain eligibility correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0/16 | 16/16 | 7/8 | 7/8 | 4/7 | 2/2 |
| Prior adapter | 0/16 | 16/16 | 7/8 | 8/8 | 4/7 | 2/2 |
| Trained continuation | 0/16 | 16/16 | 8/8 | 7/8 | 5/7 | 0/2 |

The frozen development gate required at least 6/7 useful claims accepted, at
least two more than the same-call prior adapter, all eight unsupported claims
rejected, and no loss in supported retention. The trained continuation gained
one useful acceptance over prior and retained all supported claims, but missed
the useful-acceptance target and regressed unsupported rejection from 8/8 to
7/8. The gate failed; no model was selected.

These are provisional agent judgments with zero human reviewers. The candidates
were supplied for development and are correlated; this is not a held-out test
or an estimate of natural error rates. The result does not establish extraction
recall, broader test quality, or downstream QA improvement.

## Separate unrun candidate update

After this run, a blind review and root adjudication revisited four older
writing-task examples. Their support labels stayed at two supported and two
unsupported; all four eligibility labels became uncertain under the existing
prompt. The original run and review records remain unchanged. A separate
candidate now has 34 private rows from three sessions plus the same 48 fictional
replay rows, for 82 rows total. Its development cases, labels, and prompts are
byte-identical to those in this comparison. This candidate has not been
trained, and no additional GPU call was made.

Machine-readable results: [private SFT report](private-sft-results.json).
Checks: [validation record](validation-private-sft.json).
