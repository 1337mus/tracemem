# Matched-exposure training results

## Result

A controlled development run trained on 256 exposures and 64 optimizer steps,
using 212 unique consumed source rows. It included all 52 reviewed fictional
claims once across 26 scenarios. The private exposures used the same 37
identities in the same positions as recovery, with identical support and
eligibility reference labels at every position. The frozen 32 development
prompts and references were unchanged.

The candidate reproduced the recovery output for every one of the 32 trained
predictions. This is a negative development result: the matched exposure
change produced no measured efficacy gain, and the private uncertain-
eligibility target still failed. No model was selected or promoted.

## Exposure and run accounting

All 69 previously consumed protected historical rows remained. Their repeated
exposures fell from 132 to 113: 19 uncertainty repeats were replaced. Another
33 unprotected singleton exposures were replaced. The new run used 143,776
input tokens versus 135,695 for recovery; both runs had 3,087 supervised
tokens. Token totals were not matched.

Training took 486.181436 seconds; the full client run took 638.176827 seconds.
It ran on one A10G with a 1,800-second timeout and zero retries. Actual billed
cost is unavailable.

## Development results

| Cohort | Supported | Unsupported | Useful accepted | Eligibility exact | Uncertain eligibility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Private, 16 claims | 8/8 | 8/8 | 6/7 | 13/16 | 0/2 |
| Fictional, 16 claims | 7/8 | 6/6 | 6/7 | 16/16 | no uncertain-eligibility references |

In the fictional cohort, uncertain-support labels were 2/2. All 96 outputs
were valid JSON and ended with EOS. Base and prior input/output token arrays
matched recovery exactly on 32/32 prompts.

## Interpretation and next step

There were zero human reviewers. These are tiny adaptive development cohorts,
and useful private references are concentrated in one session. The results do
not establish held-out quality, generalization, or downstream question-answer
quality. The reserved verifier test remains sealed.

The next evaluation should use genuinely independent conversations, separately
reviewed labels, and downstream QA. Resolve the disputed memory-scope policy
and change the data or hypothesis before any further training; unchanged small
data does not justify another GPU run.

## Validation context

The full `make check` passed 1,143 tests run (1,142 passed, one optional skip),
before the narrow scorer receipt-parser fix. After that fix, 61 focused tests
passed; Ruff and format checks and configured mypy over 74 files also passed.
The canonical aggregate score receipt succeeded and agrees with the independent
counts. The frozen runner remained pinned to
`181fe0ad6c68e529351e48048e5770da76031aecf7257cb49cb4e84232069ba4`.
The full workspace suite was not rerun after the parser fix.
