# Verifier v2: better eligibility, incomplete support tradeoff

The continued adapter retained more useful memories and rejected more bad
claims, but it also rejected supported claims. It failed the frozen development
criteria. No model was selected for product use.

The comparison uses the same Qwen3-4B base model, the previous adapter, and that
adapter after training on 192 fictional rows for two epochs (96 optimizer steps,
learning rate 0.00005). All three arms received identical prompts and input token
IDs. All 324 development outputs were valid and untruncated. Labels remain
provisional agent judgments with zero human reviewers.

| Development cohort and measure | Base | Prior adapter | Continued adapter |
| --- | --- | --- | --- |
| New: supported claims retained | 24/24 | 24/24 | 22/24 |
| New: unsupported claims rejected | 11/24 | 15/24 | 21/24 |
| New: useful supported memories retained | 9/12 | 2/12 | 11/12 |
| New: unsupported eligible claims accepted | 2/12 | 0/12 | 2/12 |
| New: decisive eligibility labels correct | 12/36 | 9/36 | 36/36 |
| Regression: supported claims retained | 24/24 | 24/24 | 21/24 |
| Regression: unsupported claims rejected | 10/24 | 22/24 | 24/24 |
| Regression: useful supported memories retained | 16/19 | 10/19 | 15/19 |
| Regression: unsupported eligible claims accepted | 3/19 | 1/19 | 0/19 |
| Regression: decisive eligibility labels correct | 27/42 | 25/42 | 36/42 |
| Original paired dev: supported retained | 5/6 | 6/6 | 5/6 |
| Original paired dev: unsupported rejected | 3/6 | 4/6 | 5/6 |

“Useful supported memories retained” requires both support and personal-memory
eligibility. It is not extraction recall: these candidates were supplied to a
verifier. Unsupported catches and valid-memory retention must be considered
together. Uncertain eligibility is reported separately in the machine report.

## Frozen test audit

The sole continued adapter was frozen before the fresh 24-scenario test was
opened. The single audit also failed its frozen criteria. Test outputs cannot
be used to retune this candidate or select another checkpoint.

| Test measure | Base | Prior adapter | Continued adapter |
| --- | --- | --- | --- |
| Supported claims retained | 24/24 | 24/24 | 22/24 |
| Unsupported claims rejected | 9/24 | 18/24 | 23/24 |
| Useful supported memories retained | 12/12 | 5/12 | 10/12 |
| Unsupported eligible claims accepted | 4/12 | 1/12 | 1/12 |
| Decisive eligibility correct | 20/36 | 21/36 | 33/36 |
| Uncertain eligibility correct | 1/12 | 10/12 | 7/12 |

All 144 test outputs were valid and untruncated. The continued arm met the
unsupported-rejection threshold, but missed supported retention, useful-memory
retention, and uncertainty thresholds. This is a small agent-authored synthetic
audit with related logical patterns, not evidence of real-conversation quality.
Worker time was 221.97 seconds on one NVIDIA A10, with no training or retries.
See the [separate test aggregates](verifier-v2-test-results.json).

## What still goes wrong

Root inspected every continued-arm support error on the three development
cohorts. Remaining false accepts include dropping the plumber's reporting
source, treating a fictional courier's report as established story fact,
reversing an only-if relation, and dropping a dry-cell condition. False rejects
include a faithful two-actor canopy repair statement, a faithful only-if
paraphrase, and a statement explicitly declining to assert causality. These
errors motivate diverse positives as well as hard negatives.

The dataset uses a recurring three-turn scaffold and related logical patterns;
new identifiers do not make it a population sample. Regression cohorts are
adaptive and curator-visible. Frozen labels include previously documented
scope/attribution tensions. They are preserved, with the predeclared sensitivity
analysis, rather than changed after seeing results.

## Runtime and reproducibility

The successful call used one NVIDIA A10. Worker time was 1,155.82 seconds,
including 686.91 seconds training; client time was 1,169.33 seconds. Peak tensor
allocation was 5.28 GB. Billing was not measured. Adapter weights remain on
personal Modal and were verified unchanged after reload. The prior adapter was
mounted read-only and its hashes also remained unchanged.

Two earlier attempts remain failed: the first exceeded Modal's serialization
limit before remote execution; the second failed historical inference parity
before any optimizer step. Matching historical model preparation corrected the
runtime difference. The successful call reproduced every historical transfer
output field before training, with no relaxed replay criterion.

A single final adapter was frozen before opening its prospective test. The
predeclared test audit is reported separately; a test pass cannot rescue failed
development criteria. No retuning on that test is permitted.

Reproduce development scoring without a GPU:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.evaluate_verifier_v2 \
  --payload examples/synthetic/verifier-v2-training-payload-v3.json \
  --receipt examples/synthetic/verifier-v2-training-remote-v3.json \
  --output /tmp/tracemem-verifier-v2-dev.json
```

See the [frozen protocol](verifier-v2-protocol.md),
[development report](verifier-v2-dev-results.json), and
[data card](verifier-v2-data-card.md). The [realistic-data work](realistic-data.md)
addresses the next gap. This synthetic experiment used no private conversations.

Reproduce the separate test report:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.evaluate_verifier_v2_test \
  --payload examples/synthetic/verifier-v2-test-payload-v1.json \
  --receipt examples/synthetic/verifier-v2-test-remote-v1.json \
  --training-receipt examples/synthetic/verifier-v2-training-remote-v3.json \
  --output /tmp/tracemem-verifier-v2-test.json
```
