# Retrospective integrity-versus-support diagnostic

Root architecture after e2ccd54. Bounded offline research slice; reuse immutable
fictional receipts and existing agent responses. No inference, model calls, private
inputs, tuning, training, threshold selection, runtime policy or fixture changes.
This addresses the measured gap between exact citations and semantic support.

Worker owns only `experiments/evaluate_integrity_selectivity.py`,
`tests/test_integrity_selectivity.py`, `docs/integrity-selectivity.md` and its own
`.local/integrity-selectivity-*` logs. Do not edit existing source, tests, fixtures,
receipts or integration/docs files outside that scope. No commit/stage/push.

Implement `evaluate_integrity_selectivity(receipt_bytes_sequence,
response_documents, *, seed) -> dict[str, Any]` and CLI `python -m
experiments.evaluate_integrity_selectivity --receipts ... --responses ... --seed
1729 --output NEW_PATH`. Use the existing exclusive-output helpers; no overwrite.
Require raw immutable receipt bytes. Reuse `prepare_reviews`, `summarize_reviews`,
`validate_responses`, `evaluate_scenarios.evaluate_receipt`, `parse_candidates`,
`_case_records` and `verify_candidate` rather than reproducing their validators.
Validate complete inputs before returning/writing, including duplicate receipt/model,
duplicate reviewer, incomplete/wrong-bundle responses and late malformed inputs.

Regenerate blinded bundle/provenance. Map each original model/case/claim_index to
its existing anchored review task; preserve duplicate origins separately even when
the task deduplicates across candidates/models. A raw claim's review label comes
from its anchored task because anchoring changes offsets only. Explicitly ensure
raw/anchored candidate count/order and all non-offset claim content are unchanged
(e.g. compare to `anchor_unique_quotes(raw, records).claim`). Reject missing/extra
origin mappings; never align by position in a shuffled review task list.

Compare exactly three retrospective filters, no selected winner or tuned threshold:
`all_schema_valid`, `raw_integrity`, `anchored_integrity`. A verifier status of
`REVIEW` passes an integrity filter; this is not semantic acceptance. Verify raw
and anchored claims against their case records separately. Retain raw and anchored
verifier error-code counts and origin audit rows (model/case/claim_index/task_id,
passes and error codes only), omitting transcript/claim text and rationales.

Return schema version, provenance/input hashes, complete existing summary under
`accounting`, explicit static filter definitions/limitations, and `by_model` with
separate dev/test cells. For each filter, count passed/rejected origins, coverage
(passed/schema-valid origins), and separate rows per reviewer ID/kind with passed
and rejected supported/unsupported/uncertain origin counts. Report supported/all
passed, supported/decisive passed, and supported-retention (passed-supported/all
reviewed-supported). Every zero denominator is null. Never convert uncertainty to
unsupported or labels to truth, merge reviewers, count deduplicated tasks as origins,
or treat parse/schema exclusions as empty successful outputs. With no reviewers,
report integrity counts normally and empty reviewer rows; no invented labels.
Exclusions/empty outputs remain explicit in accounting; candidate-denominator
coverage is not whole-output success or recall. State that raw filters are applied
to retrospectively selected schema-valid reviewed origins, labels are self-attested,
human identity/independence is unverified, and no calibration/precision guarantee,
Gate A, prospective test, runtime acceptance policy, or model superiority follows.

Incremental meaningful TDD. Tests should cover a shifted exact quote repaired into
an integrity pass while still labeled unsupported, an unrepairable role conflict,
raw-valid unsupported claims, uncertainty denominators, supported retention and
zero denominators, duplicates sharing one task with distinct raw pass states,
shared tasks across models, separate dev/test/reviewers, parse/schema exclusions
and valid-empty outputs, no reviewers, provenance hashes and immutable inputs,
malformed/duplicate/incomplete/late-invalid inputs and exclusive CLI output.
Reuse fictional test helpers from `test_scenario_reviews` where useful. Run focused
unit tests, Ruff check/format and strict mypy on new experiment (tests may follow
existing typing conventions). Root reviews all final files and independently
recomputes real fictional-receipt counts, then handles reports/integration/full
validation/delivery. Documentation must not invent measurements before root runs.
