# Scenario extraction dataset data card

Updated 2026-10-14.

## Summary

- **Fixture:** [`src/tracemem/data/scenarios_v1.json`](../src/tracemem/data/scenarios_v1.json), version 1.
- **SHA-256:** `5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214`.
- **Contents:** 48 fictional conversations, with 24 `dev` and 24 `test` cases; 8 cases have empty expected claims, split evenly across the two partitions. Each case has 3–5 messages over at least two sessions. There are 43 authored expected claims.
- **Purpose:** a small diagnostic comparison set for evidence-grounded claim extraction. It is not a training corpus, a production quality measure, or a population sample.
- **License:** the fixture declares MIT and contains newly authored synthetic content. The license does not cover evaluated models or their training data.

## Provenance and partitioning

The messages and labels were authored as fictional examples for this project. They do not reproduce private conversations and do not represent any person's actual conversation history. The generic `user` in a case is a fictional narrator scoped to that case; `user` is not a shared person identifier across cases.

Cases were assigned to `dev` and `test` before inference. There is no training split. The `test` split is reserved for this comparison, but it is not independently human-validated or population-representative. The assignment is not stratified or balanced by domain or conversation length: `dev` has 12 three-message and 12 four-message cases; `test` has 13 three-message, 9 four-message, and 2 five-message cases. The eight no-memory cases are evenly divided, with four in each split.

Each `case_metadata` entry declares scenario and template groups plus case-scoped entity identifiers whose slugs refer to fictional entities named in that case. The loader verifies that declared groups do not cross the partitions. These bookkeeping IDs do not prove that entity surfaces, fictional users, language, or templates are independent across partitions; the lexical audit cannot prove those properties either.

## Labels and coverage

Expected claims use the shared challenge schema for subject, predicate, object, speaker role, attribution, modality, polarity, optional exact-date bounds, and one or more evidence spans. Span offsets count Python Unicode codepoints in the exact message text. The scenario loader reuses the challenge validator to check claim schema and quote integrity, in addition to checking timestamps, message/session counts, IDs, metadata, and split counts.

Coverage targets are non-exhaustive and hand-selected. They include direct and third-party attribution, document reports and quoted instructions, tentative consideration, preferences and negation, temporal changes, corrections, causal direction, ownership, location, coreference, and unresolved ambiguity. Tags are navigation hints, not a balanced taxonomy or exhaustive statement of what a case can test.

## Readiness evidence and limits

The current lexical diagnostics are recorded in [`scenarios-readiness.json`](scenarios-readiness.json). At threshold 0.8, the fresh scenario-only audit found no exact or near-duplicate pairs across `dev` and `test`. A second pass compared the new fixture with the older 24-case diagnostic claim challenge, whose cases remain entirely in `diagnostic`; the combined audit found no exact or near-duplicate pairs. The report contains hashes, counts, and diagnostics, not transcript text.

These are conservative lexical flags, not semantic judgments. An exact match is a review target; an unmatched or unflagged pair is not evidence that a claim is unsupported, that two scenarios are independent, or that no leakage exists. The audit cannot establish legal permission, annotation quality, sample adequacy, or Gate A approval. Its zero-pair result does not demonstrate natural-language, user, entity, or template isolation.

All gold labels are agent-authored and provisional. No independent human annotation or adjudication has validated them. Exact structured scoring can penalize a faithful alternative rendering or a supported claim omitted from the provisional gold. The fixture must not be used to train a model, authorize automated memory writes, select a production threshold, or claim generalization beyond this small authored set.
