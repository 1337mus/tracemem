# Scenario review uncertainty diagnostic

Updated 2026-10-14. Prototype, fictional data only. Root owns final review.

Implement a descriptive scenario-cluster bootstrap over existing generated-candidate
review labels. Current counts are small and correlated within conversations;
claim-wise binomial intervals would obscure that. This does not establish
independent truth, calibration, acceptance precision, Gate A, or a model winner.

Worker owns ONLY new files: experiments/bootstrap_scenario_reviews.py,
tests/test_review_uncertainty.py, docs/review-uncertainty.md. No commits, pushes,
existing-file edits, new GPU/network calls, source/memory changes or model jobs.
Use gpt-6-luna with bounded context. Follow TDD, preserving actual red/green logs
under .local/review-uncertainty-*. Root handles integration, receipts and final checks.

Reuse prepare_reviews, summarize_reviews and validate_responses, existing strict
JSON readers and exclusive output helpers. API bootstrap_reviews(receipt_bytes_sequence,
response_documents, *, seed, bootstrap_seed=1729, bootstrap_samples=2000).
Regenerate/validate provenance and complete response documents. Keep model, split,
reviewer ID/kind separate. Return existing provenance hashes and all exclusions
through a nested accounting report, plus descriptive intervals per reviewer/cell.

Resample whole scenario cases with replacement within each split (all 24 cases,
including zero-candidate/invalid-output cases). Keep all claim origins in a sampled
case together, counting duplicates as origins exactly as existing accounting does.
For each cell measure supported/all-reviewed, supported/decisive and uncertain/all-
reviewed. Point estimates must match existing accounting. Seed uses local Random;
reproducible ordered cells/cases, independent of receipt or response order where
practical. Percentile interval 2.5/97.5%, linear interpolation rank=(n-1)*q.
Null interval when no finite replicate; explicitly report valid/undefined replicate
counts and sampled scenario count (with/without reviewed origins). No denominator
substitution or uncertain->unsupported conversion. No human identity/independence
claims. Sampling bounds 1..100000 with bool rejected; integer seed validation.
No cross-model hypothesis test or paired comparison required.

CLI mirrors summarize_scenario_reviews, with --bootstrap-seed/--bootstrap-samples.
Refuse overwrites/path aliases and preserve inputs on errors. Tests must pin
clustering (all claims in same case move together), percentile arithmetic against
hand-computable samples, fixed-seed replay, case denominators/zero-origin cases,
all-uncertain and no-reviewer cells, origin duplication, existing validation failures
and CLI exclusive output. Use fictional fixtures from existing test helpers.
Return concise changes/test commands, actual failures corrected, limitations.
