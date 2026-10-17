# Second private training comparison

Updated 2026-10-17 UTC.

The latest private continuation was compared with the pinned base model and the
same-call prior adapter on 16 supplied development candidates: eight windows
from two sessions. All seven useful references came from one session. Labels
are provisional agent judgments; there were no human reviewers.

Training used 50 private rows from 25 windows across four sessions, plus 192
fictional training replay rows, for 242 rows total. It ran one epoch, took 61
optimizer steps, and trained for 316.333 seconds. The development prompts and
labels were unchanged. All 48 outputs across the three arms were valid and
ended directly in EOS; no EOS corrections were made. For both base and prior
arms, input and output token IDs matched the first private run on all 16/16
candidates.

| Arm | Supported retained | Unsupported rejected | Useful accepted | Uncertain scope correct |
| --- | ---: | ---: | ---: | ---: |
| Base | 7/8 | 7/8 | 4/7 | 2/2 |
| Same-call prior adapter | 7/8 | 8/8 | 4/7 | 2/2 |
| New continuation | 8/8 | 7/8 | 6/7 | 0/2 |

The first private continuation accepted 5/7 useful references, rejected 7/8
unsupported claims, retained 8/8 supported claims, and got 0/2 uncertain-scope
cases correct. In this run, the useful threshold of 6/7 and gain of at least two
over the same-call prior were met. The unsupported threshold of 8/8 and
uncertain-scope threshold of 2/2 failed, so the prospective gate failed and no
product model was selected.

Both the private training data and fictional replay changed, so the result
cannot be attributed to either change alone. Reserved test data remains unused.
This development comparison does not establish extraction quality, natural
error rates, or downstream QA improvement.

Before another training run, target source-backed negation and deferral examples,
clearer scope labels, owner calibration, and a broader separate evaluation. No
additional training run is scheduled automatically.

Machine-readable [aggregate](private-sft-scope-results.json). See the [first
private SFT results](private-sft-results.md) for the preceding run.
