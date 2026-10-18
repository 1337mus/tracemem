# Bounded validation milestone

**Status: IN PROGRESS.** This is one time-bounded pilot to decide whether further specialized-model research is justified. No new model-efficacy result or decision exists; there have been zero new GPU calls, training calls, or source uploads. The deadline is **2026-10-20 21:56 UTC (2:56 p.m. PDT)**.

## Scope and limits

The personal-source route currently has capacity for **zero new admissions**. In the public benchmark route, 24 sessions have been screened and 20 minimized excerpts from 20 source families have been admitted locally; four were withheld for privacy. Root hash and span validation is complete. These conversations are source-backed, but natural human authorship, constituent rights, and pretraining overlap remain unestablished; there is no egress clearance. The source worker confirms that the evidence minimums appear feasible, but final counts are not validated: this is not yet 100 judgments or 30 QA questions, and human-reviewed labels remain at zero. The recommended distinction between durable facts/preferences and one-off requests remains provisional pending the owner decision.

Reuse frozen models first. At most four new A10G GPU calls are allowed, each capped at 30 minutes, with zero retries. At most one call may be optional training, capped at 256 exposures and 64 optimizer steps. Train only if independent development evidence identifies a correctable systematic failure; amend and freeze the protocol and checkpoint before final-test access. Do not tune against final-test results.

The exact-span validator checks quote, role, and citation structure; it is not a semantic support or eligibility classifier. The local trained-memory-to-QA bridge is implemented and root-reviewed with seven focused tests. It preserves exact role, revision, and spans, and uses equal token ceilings and supplied retrieval order; it does not retrieve, rank, or establish semantic support. The opt-in verifier batch runner is root-reviewed, with 31 focused tests passing; the new QA runner is still being built locally. Raw-transcript hybrid retrieval is the practical deterministic QA comparison. Prior aggregate QA results are documented in the [downstream QA baseline](downstream-qa-baseline.md).

## Evidence minimums and decision rule

Freeze sources, references, policy, splits, and hashes before producing model outputs. The evaluation requires at least 100 claims from 20 new independent source families (target 120), at least 30 supported-and-eligible references and 30 unsupported references, at least 30 QA questions across 10 source families, and at least 20 human-reviewed labels. Claims and questions must be disjoint by whole source and scenario family from prior training, development, and adaptive diagnostics. Human review uses a seeded random sample plus disputed items, with model predictions hidden. Missing any evidence minimum means **NO_GO_INSUFFICIENT_EVIDENCE**, not a model-inferiority finding.

A **GO** requires all of the following:

- Support precision of at least 95% and useful candidate retention of at least 80%, with no more unsupported auto-accepts than the base model.
- At least a 5 percentage-point useful-retention gain over base, and at least a 10-point gain in grounded-correct QA over raw-transcript hybrid retrieval, with no increase in unsupported QA additions.
- Positive session-cluster bootstrap 95% lower bounds for both gains.

Useful retention is conditional on the supplied candidate set; it is not full extraction recall. These are pilot decision thresholds. A GO permits consideration of further research; it does not establish 99% precision or production readiness. See the [aggregate-only prior model results](calendar-precision-training-results.md) for historical context.

## Reporting boundary

Report aggregate counts and session-cluster confidence intervals, exact artifact hashes, limitations, runtime, and cost when known. Keep source bodies, names, paths, claims, identifiers, and private receipt data out of tracked documentation. This note records the milestone scope and gates; the current operational state remains in `.local/active-cycle.json`.
