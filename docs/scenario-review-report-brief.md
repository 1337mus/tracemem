# Per-model review accounting

2026-10-14. Implement only `experiments/summarize_scenario_reviews.py` and
`tests/test_scenario_review_report.py`. Local synthetic experiment tooling; no
source fixture, runner, production package or historical receipt edits.

Expose `summarize_reviews(receipt_bytes_sequence, response_documents, *, seed)`.
Rebuild the blind bundle/key/provenance using prepare_reviews on actual validated
receipt bytes and the specified seed. This avoids trusting a caller-supplied
model/task mapping. Validate complete reviewer documents with the existing
annotation validators/report; reject duplicate reviewer IDs and any hash/task
mismatch. Empty response collections are legitimate and mean no reviews. Empty
task bundles must be reportable without inventing a reviewer identity.

Return a JSON report binding dataset, bundle, receipt and response canonical
hashes, the seed, and regenerated provenance hash. Count produced claim origins
by model and dev/test separately, then by each reviewer (ID and self-attested
kind); never mix human and agent judgments. One deduplicated blind task may map
to several model origins. Repeated claims within a model count as multiple output
origins but one review task. For every model/split report raw output count,
parse/schema exclusions separately, schema-valid empty outputs, schema-valid
candidate count, unique review-task count and supported/unsupported/uncertain
origin counts per reviewer. Do not count excluded outputs as empty abstentions.
Explicit denominators: supported/all reviewed origins and supported/decisive
origins; use null on zero denominators. Do not claim recall, independent human
truth, precision calibration, semantic acceptance or Gate A from these fractions.
State that citations may have been anchored and semantic judgment and raw citation
integrity are separate.

CLI: --receipts one or more; --responses zero or more; --seed; --output.
Exclusive output creation and input/output alias protection using existing local
annotation CLI helpers; no input modifications, network or inference. Bound report
paths through preflight before reading. Keep implementation about 200 lines;
avoid copying the whole evaluator. TDD against existing _receipt/_candidate helpers
in tests/test_scenario_reviews.py is acceptable. Cover two models sharing one
candidate, repeated origins, two distinct reviewer kinds, uncertain/null rates,
parse/schema failures versus real empty outputs, altered response hash/coverage,
duplicate reviewer rejection, and exclusive output/input preservation.
Run focused tests, Ruff and scoped strict mypy only. Root reviews all source/tests.
