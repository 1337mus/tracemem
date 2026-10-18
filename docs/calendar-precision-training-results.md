# Calendar-precision training comparison

## Result

Temporal target scores improved and passed the frozen target gate, but private
useful acceptance regressed from 6/7 to 5/7, failing the private control gate.
No model was selected.

## Training design

The run used 256 exposures and 64 optimizer steps with the same initialization,
seed, settings, and unchanged 32-item development set as the matched baseline.
The pool contained 228 rows; 16 new fictional claims across eight scenarios
were each exposed twice, replacing 32 support- and scope-matched fictional
slots. The schedule consumed 196 unique rows, compared with 212 previously.
All 37 private exposure positions and 177 protected fictional positions were
preserved. The original adaptive temporal prompts were excluded from training.

## Fictional temporal target

The target used the original 32 temporal prompts without calendar JSON. Counts
are candidate versus matched baseline.

| Measure | Baseline | Candidate |
| --- | ---: | ---: |
| Overall support exact | 27/32 | 29/32 |
| Supported claims retained | 14/16 | 14/16 |
| Unsupported claims rejected | 13/16 | 15/16 |
| Unsupported claims accepted | 3/16 | 1/16 |
| Eligibility exact | 32/32 | 32/32 |
| Elapsed-time support exact | 4/8 | 6/8 |
| Elapsed-time supported retained | 3/4 | 3/4 |
| Elapsed-time unsupported rejected | 1/4 | 3/4 |

The candidate met each frozen target threshold. These are adaptive fictional
development examples, not held-out evidence.

Three target predictions changed from the matched baseline: two unsupported
errors changed from incorrect acceptance to correct rejection. One previously
incorrect supported prediction changed from uncertain to unsupported and
remained incorrect.

## Fixed development controls

| Cohort and measure | Matched baseline | Candidate |
| --- | ---: | ---: |
| Private: support retained | 8/8 | 8/8 |
| Private: unsupported rejected | 8/8 | 8/8 |
| Private: useful acceptance | 6/7 | 5/7 |
| Private: scope exact | 13/16 | 13/16 |
| Private: uncertain-scope exact | 0/2 | 1/2 |
| Fictional: supported retained | 7/8 | 7/8 |
| Fictional: unsupported rejected | 6/6 | 6/6 |
| Fictional: useful acceptance | 6/7 | 6/7 |
| Fictional: scope exact | 16/16 | 16/16 |
| Fictional: uncertain-support exact | 2/2 | 2/2 |

The fictional control predictions were unchanged. The private useful-acceptance
regression failed the control gate, so the target improvement does not justify
selection or promotion. Uncertain-scope results are reported separately.

## Run quality and limits

All 96 inference outputs were valid two-key JSON ending with EOS. Base and prior
token replay matched the frozen comparison on all 32 prompts. Training used
139,927 input tokens and 3,087 supervised tokens; training took 341.958821
seconds. The full client run took 508.270063 seconds. Inference took 156.155711
seconds remotely and 173.868349 seconds client-side. Billed cost is unknown.

There were zero human reviewers. The 32-item temporal target was a small,
adaptive fictional development diagnostic; the unchanged 32-item controls
included private and fictional examples. Neither set was held out, so these
results do not establish generalization, causality, or downstream benefit.
Nine development-scorer tests and three payload-builder tests passed; a root
independent raw tally matched the target counts. No full-suite rerun is claimed.
The 28-item public transfer remains blocked pending exact owner-egress approval;
no data was sent. No further GPU run is planned this cycle. Investigate the
support/scope trade-off and freeze a broader, independently labeled source
evaluation before proposing further training.

The reserved verifier dataset was not intentionally opened. An incidental
historical-report exposure included a test-split marker, encoded token arrays,
and two output records; whether those records overlap the reserved dataset is
unresolved. Do not claim that the reserved set was untouched.
