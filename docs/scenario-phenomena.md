# Scenario tag breakdown

This report groups existing synthetic generation outputs and validated support
judgments by the source tags authored for each scenario. It is retrospective,
descriptive evidence. It does not assign an error type to an unsupported label,
change a prompt or acceptance rule, or select a model.

## Reading the cells

The schema 1 JSON is nested as `by_model -> split -> by_tag`. It includes one
cell for every distinct source tag in each `dev` and `test` split, even when a
model has no valid origin for that tag. A case belongs to each distinct tag on
its source scenario. The frozen fixture has 45 distinct dev tags across 67
case-tag memberships and 48 distinct test tags across 72 memberships. There
are 24 source cases in each split.

Tags overlap within cases. Counts across tag cells therefore do not add up to
unique cases or origins. A tag describes membership of the whole scenario. It
does not say which phenomenon caused a reviewer label. Cells are small, so
their rates are descriptive and can move sharply with one judgment.

Each cell reports source case membership; parse, schema, and anchoring
exclusions; schema-valid empty and nonempty output counts; validated origin
count; and unique review task count. The anchoring exclusion is separate from
raw schema failures. A repeated candidate can create multiple origins while
remaining one review task, so those counts answer different questions.

Reviewer rows stay separate by reviewer ID and self-attested kind. Supported,
unsupported, and uncertain origin counts are each counted against every
origin. The all-reviewed rate is supported divided by all three labels. The
decisive rate is supported divided by supported plus unsupported. A zero
denominator is `null`. Expected claim counts are not used, and these rates do
not imply recall, precision, calibration, or independent human truth.

The report binds the seed, dataset SHA-256, bundle and provenance hashes,
task-mapping hash, raw and canonical receipt hashes, and canonical response
hashes. It contains no generated claim, transcript, quote, rationale, or answer
text. Reviewer identity and kind are self-attested. No independent human
annotation is established.

## Recreate the summary

The command uses existing validated receipts and response documents. It makes
no inference or network call, does not modify inputs, refuses an existing or
aliased output, and writes a new JSON file with mode `0600`.

```sh
PYTHONPATH=src python -m experiments.summarize_scenario_phenomena \
  --receipts receipt-a.json receipt-b.json \
  --responses review-a.json review-b.json \
  --seed 101 \
  --output scenario-phenomena.json
```

Omit `--responses` when no support judgments are available. Empty response
collections produce empty reviewer lists while preserving output and origin
counts. Keep development and test cells separate when interpreting the report.

## Frozen fictional replay

The [recorded report](scenarios-phenomena.json) uses both frozen model receipts
and existing blind-agent and nonblind-root response files, at seed 1729. Root
independently replayed all 186 tag cells, 372 reviewer cells and 744 rates from
saved provenance and labels. The six focused tests pass outside the checkout
against the unchanged installed runtime; the report reproduces byte for byte.
The [validation receipt](validation-scenario-phenomena.json) binds the checks.

These cells expose the same existing limitations: Phi has no valid development
origins and only one test origin; most outputs are excluded by the strict schema.
Qwen has 41 development and 42 test origins before integrity filtering. Tag cells
cannot make excluded outputs into reviewed claims or establish human quality.
