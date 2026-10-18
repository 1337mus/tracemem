# Uncertainty-training results

Updated 2026-10-17 UTC.

This report preserves the earlier scored comparison and its separate training-fit
diagnostic as historical results. The first balanced-support launch failed
during runtime setup before training and consumed no schedule rows. A separate
recovery later completed; see the [balanced-support recovery report](balanced-support-results.md).

The scored candidate documented here was rejected. No model was selected or
promoted. The run continued the Qwen3-4B verifier from synthetic-v2 on 256
rows: 50 unchanged private rows, 24 new fictional ambiguity rows replacing 24
legacy rows, and 182 other fictional rows. Training lasted one epoch, with 256
microsteps and 64 optimizer steps, taking 459.941064 seconds.

The training support-label mix was 125 supported, 124 unsupported, and 7
uncertain, compared with 128 supported, 128 unsupported, and 0 uncertain in the
previous 256-row adapter. Scope labels were unchanged: 146 eligible, 50
ineligible, and 60 uncertain.

All 96 evaluated outputs were valid exact two-key JSON and ended with EOS.
Base and prior input and output token arrays each matched frozen inference on
32/32 cases; candidate input token arrays matched on 32/32 cases.

## Development comparison

The private development set contained 16 candidates from two sessions. All
seven useful references came from one session. The fictional development set
contained 16 candidates across eight scripted scenarios. Counts below compare
the previous 256-row adapter with the uncertainty-training candidate.

| Measure | Private prior | Private candidate | Fictional prior | Fictional candidate |
| --- | ---: | ---: | ---: | ---: |
| Supported claims retained | 8/8 | 8/8 | 7/8 | 7/8 |
| Unsupported claims rejected | 7/8 | 7/8 | 6/6 | 6/6 |
| Useful references accepted | 7/7 | 6/7 | 6/7 | 6/7 |
| Eligibility labels exactly correct | 14/16 | 13/16 | 15/16 | 14/16 |
| Uncertain eligibility cases labeled uncertain | 0/2 | 0/2 | — | — |
| Uncertain support cases labeled uncertain | — | — | 0/2 | 0/2 |
| Unsupported eligible claims automatically accepted | 1/7 | 1/7 | 0/5 | 0/5 |

Both predeclared gates failed. The aggregate machine-readable record is
[uncertainty-training-results.json](uncertainty-training-results.json).

## Training-fit diagnostic

A separate check scored the exact 24 new fictional training examples, including
7 uncertain-support examples. The scorer was frozen before predictions, and the
counts were independently recomputed with the production parser. All 72 outputs
across base, synthetic-v2, and the rejected uncertainty candidate were valid
JSON and ended with EOS.

| Training-fit measure | Base | Synthetic-v2 | Rejected uncertainty candidate |
| --- | ---: | ---: | ---: |
| Support labels exactly matched | 17/24 | 15/24 | 15/24 |
| Uncertain-support cases labeled uncertain | 3/7 | 0/7 | 0/7 |
| Eligibility labels exactly matched | 19/24 | 22/24 | 24/24 |

The candidate did not fit the rare uncertain-support class in this diagnostic.
Insufficient exposure is a hypothesis for that result, not causal evidence. This
training-fit check does not change the candidate's rejected status or provide
efficacy or promotion evidence. The separate [aggregate record](uncertainty-training-results.json)
includes the counts and scope.

## Limits and next step

Labels are provisional agent judgments; zero humans reviewed them. These are
small, adaptive development sets, not held-out evaluation. Prior development
exposure by the candidate author cannot be ruled out. The reserved verifier test
was not used. The results do not support efficacy, product-benefit, causal, or
held-out-quality claims. At the time of this comparison, a balanced-support
sampler and 48 fictional training rows were being prepared. The first launch
failed during runtime setup; a later recovery's results are reported separately
and do not change this historical comparison.
