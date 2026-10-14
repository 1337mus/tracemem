# Scenario curation notes

Updated 2026-10-14.

## Annotation corrections

The final curation pass kept expected claims tied to the exact evidence that supports them and narrowed labels where the conversation only reports another person's, a document's, or a quoted source's content. Reported material is represented as `REPORTS`, rather than as a direct user fact. Preference labels preserve preference modality and negative polarity without encoding negation redundantly in the object. Date-bounded changes retain the explicit start and end dates, and tentative consideration remains tentative rather than becoming a definite plan. These choices are reflected in the frozen scenario fixture and its integrity tests.

Unique-quote anchoring in receipt evaluation may repair offsets when a quote has one exact location. That repair does not establish raw citation integrity or semantic support; generated candidates remain unlabeled review inputs.

## Audit interpretation

The readiness report records lexical exact- and near-duplicate checks for the scenario fixture, the legacy diagnostic fixture, and their combined set. Any pair lists produced by these checks are conservative, non-exhaustive review references: lexical overlap can flag candidates, while no match does not establish semantic support, scenario independence, or absence of leakage. The audit does not validate labels, establish legal permission, or grant Gate A approval. See [`scenarios-readiness.json`](scenarios-readiness.json) for method, hashes, counts, and limitations, and [`scenarios-data-card.md`](scenarios-data-card.md) for dataset scope and provenance.

## Status

The fixture remains agent-authored and provisional, with no independent human validation or adjudication. Its approved SHA-256 is `5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214`; this note does not alter fixture content or labels.
