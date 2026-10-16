# Verifier v2 paired train/dev data card

Updated: 2026-10-16

## Composition

Newly authored fictional scenarios under `personal-memory-research-v1`; no test split is included. Every scenario pairs one supported claim with one unsupported contrast claim, with identical exact citations and date fields within the pair. Each conversation has three contextual messages. Train has 96 scenarios / 192 candidates; dev has 24 / 48. Candidate support is balanced in both splits, and candidate order is 50/50 supported-first.

| Split | Condition | Attribution | Actor | Negation | Time | Causality | Eligible / ineligible / uncertain scenarios |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 32 | 16 | 12 | 12 | 12 | 12 | 48 / 24 / 24 |
| Dev | 8 | 4 | 3 | 3 | 3 | 3 | 12 / 6 / 6 |

Eligibility is constant within each pair and independent of support. Eligible cases include useful user plans, observations, completed steps, and attributed third-party reports. Ineligible cases concern characters or rules explicitly inside fictional worlds. Uncertain cases describe the user's writing or exercise activity; that metadata boundary remains unresolved under the literal policy.

The conversations use a repeated authored three-turn scaffold: a topic-setting user turn, a short assistant follow-up, and the cited source turn. The first two turns are constructed context, not naturally collected conversation history. Wording and identifiers vary, but shared scaffolding and logical patterns persist; unique IDs alone do not establish semantic independence.

## Provenance and limits

Labels are provisional and agent-authored. Human reviewers: 0. The accountable agent reviewed the sources and both claims in all 120 train/dev scenarios before freezing the dataset. That review found and corrected actor, attribution, condition and contrast-label errors; it is still an agent review. This curriculum adapts known failure types; it does **not** establish independent generalization. Shared logical structures remain across scenarios, and dev is a development split rather than a blind holdout. Do not describe these labels as human-validated or use them for product-quality claims without independent review and policy adjudication.

Dataset SHA-256: `61e28eca9cb516efae7cc7c88db7b12e0b959505b01b7b540db7791e586c3c75`
