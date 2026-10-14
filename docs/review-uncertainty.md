# Scenario review uncertainty report

This prototype reports descriptive bootstrap intervals for labels on generated scenario-review
claims. It uses fictional fixtures and small, conversation-correlated samples.

The command regenerates receipt provenance and the blind review bundle, validates complete response
documents, and puts the existing point-estimate report, provenance hashes, and every receipt
exclusion under `accounting`. Bootstrap results are grouped by model, split, and reviewer. The three
reported metrics are supported/all reviewed origins, supported/decisive origins, and
uncertain/all reviewed origins.

Each replicate samples all 24 scenario cases in a split with replacement. A selected case carries
all of its claim origins together; duplicate claim origins remain separate observations. Cases
without reviewed origins stay in the 24-case sampling frame. The report counts available and
sampled scenarios with and without reviewed origins, and reports valid and undefined replicate
counts for each metric. A zero denominator remains undefined; it is never replaced with another
denominator, and an uncertain label is never converted to unsupported. Percentile bounds use linear
interpolation at rank `(n - 1) * q`. The local bootstrap random generator uses the supplied seed.

Example:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.bootstrap_scenario_reviews \
  --receipts examples/synthetic/qwen3-4b-scenarios-remote.json \
    examples/synthetic/phi-3.5-mini-scenarios-remote.json \
  --responses examples/synthetic/reviews/scenario-review-agent-responses.json \
    examples/synthetic/reviews/scenario-review-root-responses.json \
  --seed 1729 --bootstrap-seed 1729 --bootstrap-samples 2000 \
  --output .local/review-uncertainty.json
```

The current fictional fixture output is recorded in
[`scenarios-review-uncertainty.json`](scenarios-review-uncertainty.json). Supported/all-reviewed
origin estimates and 95% intervals are:

| Model | Reviewer ID | Split | Supported / reviewed | 95% interval | Undefined draws |
| --- | --- | --- | ---: | ---: | ---: |
| qwen3-4b | luna-blind-support-20261014 | dev | 26 / 41 | [0.474, 0.794] | 0 / 2000 |
| qwen3-4b | luna-blind-support-20261014 | test | 24 / 42 | [0.388, 0.778] | 0 / 2000 |
| qwen3-4b | root-adjudication-20261014 | dev | 20 / 41 | [0.326, 0.659] | 0 / 2000 |
| qwen3-4b | root-adjudication-20261014 | test | 21 / 42 | [0.306, 0.718] | 0 / 2000 |
| phi-3.5-mini | either reviewer | dev | undefined | undefined | 2000 / 2000 |
| phi-3.5-mini | either reviewer | test | 0 / 1 | [0.000, 0.000] | 746 / 2000 |

The two reviewer IDs are self-attested agent labels; independently verified human review count is
zero. The Phi test interval is degenerate because its one reviewed origin is unsupported, and 746
replicates have no reviewed origin. Neither that interval nor any other estimate is a precision or
population guarantee.

The intervals describe resampling variability in this diagnostic sample. They do not establish
independent truth, calibration, acceptance precision, Gate A, or a model winner. Reviewer identity
and independence are not verified; `reviewer_kind` is self-attested. Zero observed errors can yield
`[1, 1]` intervals, which are not population guarantees.
