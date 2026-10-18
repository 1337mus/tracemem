# Public-dialogue verifier results

Updated 2026-10-18 UTC. This report records an inference-only comparison on 24
fresh, source-checked claims drawn from 12 public benchmark dialogue excerpts.
It is a development diagnostic, not an official benchmark score or held-out
efficacy result.

| Measure | Base | Recovery | Continued candidate |
| --- | ---: | ---: | ---: |
| Supported retention | 11/12 | 11/12 | 11/12 |
| Unsupported rejection | 9/12 | 11/12 | 11/12 |
| Useful acceptance | 9/12 | 10/12 | 9/12 |
| Eligibility exact | 14/24 | 22/24 | 20/24 |
| Unsafe autoaccepts | 1/12 | 0/12 | 0/12 |

The candidate matched the recovery on supported retention and unsupported
rejection, and did not add unsafe autoaccepts. It accepted one fewer useful
claim and had two fewer exact eligibility labels than recovery. The frozen gate
therefore failed; no model was selected or promoted. The results suggest that
the model retains support judgments while abstaining too often on eligibility.
That pattern is a hypothesis for follow-up, not a causal finding.

All 72 outputs across the three arms were valid JSON and ended with EOS. The
input token arrays matched exactly across arms on all 24 items. These checks
establish comparison consistency, not answer quality.

## Scope and limitations

The references contain 12 supported and 12 unsupported claims, all labeled
eligible. There are no uncertain or ineligible references, so accuracy on those
reference classes cannot be measured here. The sample is fresh development
data, not a held-out test, and its excerpts are not independent owner sessions.
Labels were provisional agent judgments with zero human reviewers. Pretraining
overlap is unknown, and no downstream QA effect was measured.

The review suggests the candidate may associate question or hypothetical
framing, and uncertain intentions, with eligibility uncertainty. This is a
testable explanation, not evidence that training caused the changed decisions.
Broader real-session evaluation and human calibration are still needed.

## Run and verification

The comparison used one A10G call, a 900-second timeout, zero automatic
retries, and one maximum concurrent container. Remote runtime was 111.940487
seconds, and client elapsed time was 124.747842 seconds; billed cost is unavailable.
Shutdown was verified with zero containers remaining.

The 11 builder/Modal SDK tests and 9 scorer tests passed, and the score totals
were independently replayed. The earlier serializer check passed 12 tests.
Hosted CI passed for code head `a303ebbb` in [run 39087892066](https://github.com/1337mus/tracemem/actions/runs/39087892066).
The full workspace suite was not rerun for this local comparison and docs
cycle. No training or promotion occurred.
