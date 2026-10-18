# Eligibility-transfer results

Updated 2026-10-18 UTC. The candidate met its targeted uncertain-eligibility
criterion: exact eligibility labels on private uncertain cases improved from
0/2 to 2/2, and uncertain-eligibility autoaccepts fell from 1/2 to 0/2. The
overall and original gates failed, so no model was selected or promoted.

| Cohort | Measure | Recovery | Candidate |
| --- | --- | ---: | ---: |
| Private controls | Supported retention | 8/8 | 8/8 |
| Private controls | Unsupported rejection | 8/8 | 8/8 |
| Private controls | Useful acceptance | 6/7 | 5/7 |
| Private controls | Eligibility exact | 13/16 | 13/16 |
| Private controls | Uncertain eligibility exact | 0/2 | 2/2 |
| Private controls | Uncertain-eligibility autoaccepts | 1/2 | 0/2 |
| Fictional dev | Supported retention | 7/8 | 7/8 |
| Fictional dev | Unsupported rejection | 6/6 | 6/6 |
| Fictional dev | Useful acceptance | 6/7 | 6/7 |
| Fictional dev | Eligibility exact | 16/16 | 16/16 |
| Fictional dev | Uncertain-support exact | 2/2 | 1/2 |

Both runs used 64 optimizer steps. The candidate used 256 row exposures and
consumed 212 unique rows, including 41 private rows compared with 37 for the
recovery. It exposed 45 of 52 new claims: 19 two-claim pairs were fully
exposed, 7 were partially exposed, and none received zero exposures. Since
the sampler also changed private-row allocation, this comparison does not
isolate the effect of the new fictional examples.

| Measure | Recovery | Candidate |
| --- | ---: | ---: |
| Input tokens | 135,695 | 147,059 |
| Supervised tokens | 3,087 | 3,094 |
| Training time | 334.441137 s | 503.999687 s |

The candidate client call took 657.561355 seconds; billed cost is unavailable.
All 96 outputs across the three arms were valid JSON and ended with EOS. Input
token IDs matched exactly across arms and between candidate and recovery on all
32 prompts. Base and prior output token IDs matched exactly on all 32 prompts.
Four builder tests and eleven scorer tests passed. No production code changed,
and the full workspace suite was not rerun for this local experiment.

The annotations are provisional agent judgments with zero human reviewers.
The dev set is adaptive, and its uncertain-eligibility pair contains two
claims from one source. These results do not measure held-out performance or
downstream QA efficacy.

The subsequent 24-claim public-dialogue comparison is complete. The continued
candidate matched recovery on supported retention and unsupported rejection,
and kept unsafe autoaccepts at zero, but useful acceptance fell from 10/12 to
9/12 and eligibility exactness fell from 22/24 to 20/24. The frozen gate failed
and no model was selected or promoted. See the [aggregate-only comparison
results](public-dialogue-verifier-results.md) for its scope and limitations;
real-session evaluation, human calibration and downstream QA remain necessary.
