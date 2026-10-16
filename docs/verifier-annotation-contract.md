# Verifier annotation contract (prospective v0.1)

**Status:** research contract for a future verifier training dataset. This is
not an implemented classifier, a finalized policy, or a human-approved gold
standard. This draft does not change frozen labels or launch a model run.

The separate [paired-data seed and schema](verifier-pairs-data-card.md) now
implement these annotation axes for authored research data. This does not
change the existing runtime verdict format or resolve the scope policy.

The completed [development report](quote-verifier-dev-results.md) records a
failed filter experiment on reused, agent-labeled data. The current
[development protocol](quote-verifier-dev-protocol.md) remains the record of
that frozen experiment. Preserve its labels and reports as they are; do not
silently relabel them under this prospective contract.

## Separate annotation axes

- **Support** asks whether the full cited source entails the standalone claim,
  preserving its actor, reporting source, modality, conditions, negation,
  temporal qualifiers, and causal limits. Values: `supported`, `unsupported`,
  or `uncertain` when the evidence cannot decide.
- **Eligibility** asks whether a supported claim belongs in the task's
  explicitly declared personal-memory scope. Values during curation: `eligible`,
  `ineligible`, or `uncertain` when scope policy is unresolved.
- Annotate support and eligibility independently. A supported statement can be
  ineligible; ineligibility is not evidence that a statement is unsupported.
  Do not collapse uncertainty on either axis into a negative label.
- The current runtime v1 verdict schema remains unchanged: support is
  `supported` / `unsupported` / `uncertain`, while eligibility is boolean.
  It cannot represent uncertain eligibility. Do not automatically migrate,
  reinterpret, or train on uncertain-eligibility labels through that schema.

## Anchors from the completed experiment

| Example | Source and candidate | Contract treatment |
| --- | --- | --- |
| Postcards | User: “The collection I am sorting is mostly postcards, and I need to decide whether to bring everything or just the local views.” Candidate repeats that statement. | Supported: the candidate preserves the exact user statement. Eligibility is a separate scope judgment; the frozen dev label called it supported and eligible. |
| Refund | User: “It says the refund will be reviewed within five business days, so I do not know yet whether the money is coming back.” Candidate: “The refund for the cancelled seats will be reviewed within five business days.” | Unsupported under this contract: it drops the reporting source “It says” and presents reported information as an unattributed fact. |
| Fictional menu | User: “I am comparing two fictional menus for a writing exercise; no ingredient list has been chosen and nobody has said they will cook either version.” Candidate repeats that activity. | Supported as a literal account of the user's exercise. The frozen dev labels excluded exercise metadata from eligibility. Whether writing/exercise metadata belongs in personal-memory scope is policy-sensitive and needs independent human adjudication before new labels are assigned. |

These anchors explain the observed support/eligibility conflation; they do not
settle the task's memory policy. Preserve the old experiment's provisional
counts and interpretation for reproducibility.

## Acceptance requirements for a future dataset

1. Author new fictional scenario families disjoint from the prior 12 dev cases
   and known tests. Treat that prior dev set as spent; do not tune on a sealed
   candidate or use it to select the new contract.
2. Include balanced supported claims and minimally corrupted foils. Every
   candidate must have a valid exact citation. Cover attribution, conditions,
   actor/ownership, negation, time, and causality without changing unrelated
   details in a foil.
3. Record support and eligibility as separate labels with the uncertainty
   values above. Keep original/foil pairs together in one scenario and one
   split; split train/dev/test by scenario family to prevent leakage.
4. Freeze the contract version, dataset split, and acceptance criteria before
   producing or inspecting new model outputs. Keep any sealed test untouched
   until the policy and development criteria are fixed.
5. Keep unresolved eligibility boundaries uncertain and out of decisive scoring
   until adjudicated. Independent human review of disputed boundaries and a
   validation sample is required before calling labels human-validated or making
   Gate A/product-quality claims. A frozen, disclosed agent-only synthetic
   diagnostic may proceed within the owner's standing authorization.

## Gate before another run

Once requirements 1–4 and the development criteria are frozen, an agent-only
synthetic diagnostic may proceed within the owner's standing authorization;
keep unresolved eligibility out of decisive scoring and disclose the status.
Human-validated, Gate A, and product-quality claims require the independent
review in requirement 5. Compare a trained verifier with frozen no-filter and
untuned-verifier controls. Report supported-claim retention, bad-claim catch
rate, failed batches, selected-fact retention, and added time. Keep support
scoring separate from eligibility policy. Do not automatically retry a failed
run or launch a new experiment automatically.

A fifth extractor SFT would change which candidates reach the verifier, so it
would not test whether a verifier learned to preserve source support while
applying a separate memory-scope policy. First resolve the annotation boundary
and test the verifier against fixed controls; consider extractor changes only
as a separately scoped experiment.
