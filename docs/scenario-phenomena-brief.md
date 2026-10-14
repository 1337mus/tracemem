# Descriptive scenario-phenomenon breakdown

Root architecture after selected-file evaluator. Next research slice: group frozen
synthetic generation outcomes and existing support judgments by the scenario's
agent-authored tags. The specification requests phenomenon breakdowns. This is
retrospective descriptive evidence, not semantic error-type attribution, human
precision/calibration, a model winner, a new acceptance rule or prompt tuning.
No labels, scenarios, prompts, references, raw receipts or runtime policy change.
No model/GPU/network calls, actual benchmark or private data input. Keep all models,
dev/test splits and reviewers separate; never pool the test into development.

Worker owns only `experiments/summarize_scenario_phenomena.py`,
`tests/test_scenario_phenomena.py`, `docs/scenario-phenomena.md`, and tasklogs
`.local/scenario-phenomena-*`. No commits/staging/push. Root owns generated report,
independent replay, integration, fullsuite and delivery. No runtime package changes.
Production target <=250LOC; propose partition before exceeding500. Follow applicable
preparation, engineering judgment and genuine incremental callable-stub RED/GREEN.
Compact guard tables; one CLI cross-seam proof, no redundant per-branch boilerplate.

API `summarize_phenomena(receipt_bytes_sequence, response_documents, *, seed)` follows
existing experiment input types/patterns. Reuse existing validated generation,
annotation and origin mapping via `evaluate_integrity_selectivity`, `prepare_reviews`,
`load_scenarios`, `validate_responses` rather than accepting arbitrary report-shaped
input or rebuilding receipt/schema validation. Existing validated integrity report's
origin_audit maps every schema-valid claim origin to model/case/claim_index/task_id.
Prepare provenance supplies all per-case outputs (including parse/schema-invalid
and valid-empty outputs) and bundle for validated task labels. Cross-check dataset,
bundle/provenance/task mapping hashes between reused results. Preserve duplicate
origins; do not deduplicate candidate counts into unique review tasks. Reject extra,
missing or contradictory mapping rather than silently dropping entries.

For EACH model and dev/test split, emit a sorted cell for EVERY distinct source tag
in that split, even if no valid origins. Use distinct case membership (set(tags))
per tag. Every tag cell contains source_case_count; parse_excluded_output_count;
schema_excluded_output_count; schema_valid_empty_output_count;
schema_valid_nonempty_output_count; schema_valid_origin_count; unique_review_task_count;
and separate reviewer rows with reviewer_id/kind, supported/unsupported/uncertain
origin counts, decisive_origin_count, supported_rate_all_reviewed_origins and
supported_rate_decisive_origins (zero denominator =>null). Do not use expected claim
counts or imply recall. Include no synthetic claim/transcript/quote/answer text.
Report source tags/IDs/hashes only as needed. Source tags indicate whole-scenario
membership, NOT which phenomenon caused an unsupported label. Tags overlap, so sums
across tags do not equal total cases/claims; explicitly document overlapping and
small sample sizes. Annotator identity/kind is self-attested, no independent humans.

Report schema1 includes seed/dataset SHA, source receipt/response canonical and raw
hash evidence consistent with existing experiment, mapping hashes, tag definition,
limitations and by_model -> split -> by_tag mappings. Leave underlying reports and
raw files immutable. API deterministic; input order should not silently change model
identity or origin counts. CLI uses established --receipts, --responses, --seed,
--output with exclusive preflight and0600 JSON writing helpers. No overwrite, path
collision or partial output on invalid input; sanitized errors per helper contract.

Meaningful tests using real small fictional receipts derived from existing test
helpers or authored minimal compatible structures: multi-tag overlap with empty,
invalid and multiple-origin cases; each split/model/reviewer kept separate; uncertain
and zero-denominator rows; duplicate origins mapped to one task counted separately;
invalid receipt/response/mapping failures asserted by specific reason; deterministic
report/input immutability and no content sentinels. Prefer reusing real validators
and fixtures over mocks; a narrow mapping-corruption patch may test an otherwise
unreachable defensive check. One CLI save/refusal/privacy proof. Focused tests, Ruff
check/format, explicit-package-bases strict mypy. Release files/hashes and exact
RED/GREEN logs. Root will independently group the frozen48case comparison and
existing agent response files; no new annotations or runtime inference.
Finish with the released deliverable or precise blocker; no self-waiting.
