# NLI support challenge data card

Updated 2026-10-14.

## Summary

- **Dataset:** packaged fixture at
  [support_challenge.json](../src/tracemem/data/support_challenge.json), version 1.
- **SHA-256:** `582612fc24e60868c8b0b1f37745188dbf143a41330a5777152b5cce9adc90e9`.
- **Contents:** 48 fictional premise/hypothesis pairs across 24 scenarios. Each scenario
  has one pair labelled supported and one labelled unsupported. All examples use the
  `diagnostic` split.
- **Purpose:** a small engineering diagnostic for a pinned NLI model's entailment
  probabilities on selected support and hard-negative patterns.
- **License:** the fixture declares MIT; it is authored synthetic content shipped with
  this repository. This does not establish rights to the separately evaluated model or
  its training data.

## Creation and labels

The examples and names are fictional and were authored for this project. The fixture
does not contain copied private conversations. Scenario labels and pair labels received
agent review. No independent human annotation, adjudication panel, or external
ground-truth process was used. `supported` is a binary, context-specific label for
whether the hypothesis follows from the premise as written; the fixture does not provide
a neutral class.

Pairs are grouped by scenario because the supported and unsupported hypotheses share a
premise and topic. They must not be treated as 48 independent scenarios. The model's
three-way output is mapped as contradiction=0, entailment=1, neutral=2; the reported
selection score is the entailment probability.

## Coverage

Authored examples target attribution and reported speech, dates and temporal order,
tentative or hypothetical statements, intentions and completion, preference and
uncertainty, negation, ownership and storage, location, coreference, causal claims and
direction, questions and unconfirmed claims, quoted instructions, and document
verification. Coverage is hand-selected, not a prevalence sample. Each tag has very
few examples, and good performance on a tag does not imply general capability.

## Evaluation use and limits

The [support audit](support-audit.md) reports scenario-bootstrap summaries at three
fixed thresholds. Those values describe only this frozen diagnostic fixture. There is
no train/dev/test/held-out partition, no independent support evaluation, no calibration
set, and no population sampling frame. The examples do not represent real users,
conversation styles, languages, domains, or deployment inputs.

Do not use this fixture to train a model, choose a production threshold, claim semantic
verification, or auto-accept memory claims. A future evaluation needs new scenarios,
independent annotators, adjudicated labels, leakage controls, and an explicit rights
review. The NLI output is a diagnostic signal; it does not replace source citations or
human review.
