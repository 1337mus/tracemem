# Verifier pairs v1 data card

Updated: 2026-10-16

## Purpose and composition

This is a small agent-authored synthetic diagnostic for checking whether a verifier preserves source support while keeping eligibility as a separate label. It contains 36 fictional scenario families and 72 candidate rows: 36 supported claims and 36 minimally corrupted unsupported claims. Each pair cites the same exact user message.

The six phenomena are attribution, condition, actor or ownership, negation,
time, and causality. Each has four train families, one dev family, and one test
family. Scope counts below are candidate counts; each family contributes two.

| Split | Families | Candidates | Supported / unsupported | Eligible / ineligible / uncertain |
| --- | ---: | ---: | ---: | ---: |
| Train | 24 | 48 | 24 / 24 | 24 / 12 / 12 |
| Dev | 6 | 12 | 6 / 6 | 4 / 4 / 4 |
| Test | 6 | 12 | 6 / 6 | 4 / 4 / 4 |

Pairs stay together by scenario family. Candidate order alternates across cases. The test split is visible to the curator and root, so it is not a blind holdout.

## Labels and provenance

Support and eligibility are independent. Fictional-world statements are ineligible even when supported. Exercise and writing activity describes the user's real activity, but its scope remains uncertain under the stated policy. The split-level eligibility totals are 16 eligible, 10 ineligible, and 10 uncertain families (32, 20, and 20 candidate rows). The support labels do not change with eligibility.

The error categories are informed by prior development findings. Scenarios were newly authored from those categories. No prior synthetic outputs, tests, sealed holdout, or private conversation data were consulted.

Root reviewed all 36 pairs and corrected a writing-activity label, missing
evaluation-scope coverage and one imprecise claim object before freezing the
corpus. This is an unblinded review by another AI agent, not independent human
validation. Matching group identifiers cannot prove semantic independence:
many pairs share simple constructions such as removing a condition or changing
"before" to "after" across splits.
Affirmative negation controls and pairs with two confident causal claims were
added before freezing, so simply rejecting affirmative or causal statements
cannot solve those cases.

## Replay the CPU audit

From the source checkout, choose new output paths:

```sh
PYTHONPATH=src:. .venv/bin/python -m training.audit_verifier_data \
  --dataset examples/synthetic/verifier-pairs-v1.json \
  --output /tmp/verifier-pairs-audit.json \
  --train-output /tmp/verifier-pairs-train.jsonl
```

The CLI refuses existing destinations and validates every split before writing.
It exports 48 training rows. Prompt inputs omit annotations and curator metadata;
targets contain separate support and eligibility strings. Dev and test cannot
be selected for training. These tools run from the checkout and are not part of
the installed `tracemem` package.

The [audit receipt](verifier-pairs-audit.json) pins the source bytes and reports
each split separately. Its keep-everything control retains all six supported
and all six unsupported claims on dev; it rejects zero bad claims. Four dev
candidates have uncertain eligibility and are outside decisive scope scoring.
These counts describe the dataset and a control policy, not model performance.
See the [next experiment](verifier-pairs-next-experiment.md) for the remaining
scorer and matched model comparison.

## Limits

This corpus is one agent's synthetic work. No human reviewer has validated the examples, so it is not a human-approved gold set. It makes no Gate A, product-quality, or generalization claim. The test rows are visible and do not provide blind evaluation. The 48 training rows are only a seed and are insufficient by themselves for product claims. The ten uncertain-eligibility families should remain outside decisive scoring until a human adjudicates the scope policy.
