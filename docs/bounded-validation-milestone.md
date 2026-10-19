# Bounded validation milestone

**Status: COMPLETED — NO_GO_INSUFFICIENT_EVIDENCE (2026-10-19).** This one-time
pilot closed with no model selected. The automation is PAUSED. All four authorized
GPU calls were used: verifier diagnostic, failed matched QA, optional training,
and successful changed-runner QA. No GPU work is active; no further cycle will
proceed without a new owner request. The original deadline and finite protocol
limits remain: 2026-10-20 21:56 UTC; at most four A10G calls capped at 30 minutes
each, zero retries, and at most one training call capped at 256 exposures and 64
optimizer steps.

## Scope and limits

The evaluation used source-backed benchmark conversations: 100 claims across
20 families and 30 QA questions. It did not add independent personal sessions;
the bounded personal-source routes admitted zero new sources. Claim references
comprised 56 supported, 40 unsupported, and four uncertain labels, including 33
supported-and-eligible candidates. Whole-source independence remains unverified.

The frozen claim diagnostic reached 96.0% support precision versus 87.1% for base,
but useful retention fell from 32/33 to 28/33 (-12.1 percentage points; 95%
family-cluster bootstrap interval [-27.6, 0.0]). The fixed prompt has a broader
scope instruction than the frozen reference policy, limiting interpretation of
eligibility scores. Human-reviewed labels remain zero.

The optional training treatment used 256 exposures, 64 optimizer steps, and 192
unique rows versus 212 in the matched baseline; 32 fictional claims replaced 52
treatment exposures. Private-control useful acceptance fell from 6/7 to 5/7 and
eligibility exactness from 13/16 to 12/16. Fictional uncertain-support exactness
fell from 2/2 to 1/2. Four of ten control checks failed. Training took 483.001
seconds; remote/client elapsed time was 622.884/662.839 seconds; cost is unknown.

Final QA used 30 frozen questions across 20 families, the same answering model
and prompt, and a common 3,840-token input budget with a 256-token answer ceiling.
It evaluated the frozen matched adapter, not the new treatment that failed controls.

| QA condition | Grounded-correct | Unsupported additions |
| --- | ---: | ---: |
| Raw-hybrid retrieval | 14/30 | 4/30 |
| Base-gated memory | 20/30 | 2/30 |
| Trained matched-memory | 18/30 | 2/30 |
| Empty context | 0/30 | 0/30 |

Unsupported additions include false assertions that the supplied evidence lacks
requested information; bare abstentions alone do not count as unsupported facts.

Trained matched-memory gained 13.3 percentage points over raw-hybrid retrieval
observably, with a 95% family-bootstrap interval of [-7.1, 37.9] (10,000
resamples, seed 17); it did not beat base-gated memory. The numerical +10-point
QA and no-extra-unsupported checks passed, but the useful-retention gain and
positive-confidence-bound requirements failed. The final run returned 120/120
EOS-terminated answers without truncation; remote/client elapsed time was
339.189/346.601 seconds and cost is unknown.

The earlier QA call failed remotely after 11.527 client seconds and returned no
answers; the cause remains unknown. The successful changed-runner call followed
a narrow A10 alias-compatibility fix, which does not establish the earlier
failure's cause. Five QA ratings were disputed and agent-judged with arm names
hidden, although context can reveal condition. Root review corrected two initial
bare-abstention rating errors. These ratings are not human review: zero human
labels were reviewed against the formal minimum of 20. Whole-source independence,
natural human authorship, and constituent rights remain unknown. The CPU-only
preflight passed package and tokenizer checks only (8.098 seconds remote, 16.190
seconds client). A restricted-network prelaunch check took 56.805 client seconds
before GPU activity; it was separate from GPU calls and was not a retry.

The result is insufficient validated benefit for further specialist-model
investment under this setup, not evidence of general model inferiority. The
formal GO requirements and original finite limits above remain unchanged.

The exact-span validator checks quote, role, and citation structure; it is not a
semantic support or eligibility classifier. The trained-memory-to-QA bridge,
verifier batch runner, scorer, QA runner, and context assembly were reviewed.
The bridge preserves role, revision, and spans and uses equal token ceilings and
supplied retrieval order; it does not retrieve, rank, or establish semantic
support. Raw-hybrid retrieval is the practical deterministic QA comparison.
Prior aggregate QA results are documented in the [downstream QA baseline](downstream-qa-baseline.md).

## Evidence minimums and decision rule

This completed cycle evaluated a frozen 100-claim set and 30 frozen QA questions across 20 families against the gates below. It did not meet the evidence minimum: zero human-reviewed labels against 20 required, and whole-source independence remains unestablished. For any new, separately authorized cycle, freeze sources, references, policy, splits, and hashes before producing model outputs. The required minimums are 100 claims from 20 new independent source families (target 120), at least 30 supported-and-eligible references and 30 unsupported references, at least 30 QA questions across 10 source families, and at least 20 human-reviewed labels. Claims and questions must be disjoint by whole source and scenario family from prior training, development, and adaptive diagnostics. Human review uses a seeded random sample plus disputed items, with model predictions hidden. Missing any evidence minimum means **NO_GO_INSUFFICIENT_EVIDENCE**, not a model-inferiority finding.

A **GO** requires all of the following:

- Support precision of at least 95% and useful candidate retention of at least 80%, with no more unsupported auto-accepts than the base model.
- At least a 5 percentage-point useful-retention gain over base, and at least a 10-point gain in grounded-correct QA over raw-transcript hybrid retrieval, with no increase in unsupported QA additions.
- Positive session-cluster bootstrap 95% lower bounds for both gains.

Useful retention is conditional on the supplied candidate set; it is not full extraction recall. These are pilot decision thresholds. A GO permits consideration of further research; it does not establish 99% precision or production readiness. See the [aggregate-only prior model results](calendar-precision-training-results.md) for historical context.

## Reporting boundary

Report aggregate counts and session-cluster confidence intervals, limitations, runtime, and cost when known. Keep source bodies, names, paths, claims, identifiers, artifact hashes, and private receipt data out of tracked documentation. This note records the milestone scope and gates; operational state is tracked separately.
