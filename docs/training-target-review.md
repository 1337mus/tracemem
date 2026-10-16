# Reviewed training-target audit

This audit corrects a small set of quote-training targets before another training run. It does not change the original v2 or v3 seed files, the historical v3 training set, prior evaluation results, or any development/test data. The reviews are provisional agent work: **zero human reviewers**. The reviewed training input was frozen before fresh evaluation authoring began; this document reports no evaluation result.

The original training corpus has 768 rows across 72 families: 48 replay families with 12 selected rows each, plus 24 authored families with 8 rows each. These are correlated variants of 72 templates, not 768 independent examples. The reviewed build keeps that composition and has 140 rows with empty targets. It changes 64 rows in seven families and increases the target-claim count from 680 to 696. The two review files contain 120 selected-fact annotations: 75 for v2 and 45 for v3.

## Corrections

- **T24:** Preserve the user's reported earlier possibility and current choice. The old target incorrectly converted “I said I might try” into an asserted prior consideration.
- **T35:** Record that the user found the item on their shelf and does not know its owner. Finding it there does not establish that it remains there now.
- **V3T07:** Keep both dated plan-choice events in the claim text and clear effective-state bounds. The schema does not define the endpoint convention needed to derive disjoint state intervals from day-resolution decision events.
- **V3T08:** Name the product in the reversal claim so it stands alone, keep event dates in the text, and clear effective-state bounds for the same endpoint reason.
- **V3T09:** Attach the date explicitly to the attempted upload, state that the transfer failed before arrival, and keep the event date in the claim instead of treating it as an ongoing validity bound.
- **V3T16:** Separate the dated, attributed registration report from undated current custody. The registration's ownership attribution remains part of the target.
- **V3T20:** Separate the dated loan event from current possession and the user's continuing ownership statement. The loan date does not date those present facts.

## Limits and remaining risks

The audit found **no confirmed critical fact omitted from an entire original target set**. That does not establish that target issues caused the prior coverage drop, nor that these corrections will improve model behavior. No v4 model has been trained on this corpus, and no quality-improvement claim follows from curation alone.

The adjudications are recorded in `.local/target-quality-20261015/architect-decisions-v2.json`; the independent scope and findings are in `.local/training-target-audit-20261015/summary.md`.

The review gate checks exact source/target spans, hashes, provenance, and selected-fact coverage. These literal checks do not establish semantic entailment or exhaustive recall. Coreference remains a limitation where a stored claim depends on earlier context, including the “old tool” references in T25/T26; splitting a claim also cannot create evidence absent from the source. Composite claims remain a limitation too: V3T14 keeps a custody-until date in prose, while V3T16 and V3T20 were split where their component facts had different time scopes.

## Rebuild to a new directory

The builder first reproduces the historical train file from the original seeds, then applies the reviews to copies and verifies that every non-target field and row order is preserved. It adds a `V4T-` namespace to all IDs, including expected-claim message IDs. It writes only `train.json` and `report.json`; the output directory must not already exist. The report contains hashes for all five inputs, both review-gate reports, row/family/empty-target and claim counts, changed rows/families, and preservation checks.

Run from the repository root, with a fresh output path:

```sh
PYTHONPATH=src .venv/bin/python -m training.build_reviewed_quote_train \
  --v2-source datasets/sft-quote-v2/train-seeds.json \
  --v2-review datasets/sft-target-review/v2-train-review.json \
  --v3-source datasets/sft-quote-v3/target-train-seeds.json \
  --v3-review datasets/sft-target-review/v3-train-review.json \
  --historical-train datasets/sft-quote-v3/train.json \
  --output-dir .local/target-quality-20261015/reviewed-train-v4t-repro
```

The five required input flags are `--v2-source`, `--v2-review`, `--v3-source`, `--v3-review`, and `--historical-train`; `--output-dir` is also required and must name a new directory. The builder refuses an existing directory or symlink. Keep new evaluations separate from existing model results.
