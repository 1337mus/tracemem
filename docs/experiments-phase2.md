# Phase 2 synthetic diagnostics

Updated 2026-10-14. This report covers synthetic-only diagnostics. It is not an
independent benchmark, a model selection decision, a training result, or a novelty
claim. The 24-case challenge is diagnostic, and Gate A is not met.

## Retrieval embedding comparison

The [MiniLM receipt](../examples/synthetic/minilm-remote.json) contains vectors for
40 synthetic records. The [computed report](baseline-minilm-retrieval.json) replays
them against 14 answerable and 3 unknown queries. The exact MiniLM model revision is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

| Mode | Recall@5 | MRR | nDCG@5 | Sufficient evidence | Unknown abstentions | Warm p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8214 | 0.8595 | 0.7851 | 14/14 | 0/3 | 0.496 / 0.545 |
| Signed hash vectors | 0.6310 | 0.7917 | 0.6071 | 10/14 | 0/3 | 2.982 / 3.285 |
| MiniLM vectors | 1.0000 | 0.9167 | 0.9398 | 14/14 | 0/3 | 5.012 / 5.101 |
| MiniLM hybrid RRF | 0.9167 | 0.9524 | 0.8893 | 14/14 | 0/3 | 5.626 / 5.738 |

MiniLM improved retrieval on this fixed fixture, but all three unknown queries still
received results. The sample is too small and hand-authored to calibrate abstention or
establish generalization. These are retrieval scores, not answer or claim-support
scores. The initial Phase 1 record remains in [experiments.md](experiments.md).

## Qwen3-4B V1 extraction

The [raw receipt](../examples/synthetic/qwen3-4b-remote.json) records Qwen/Qwen3-4B at
revision `1cfa9a7208912126459214e8b04321603b3df60c`, with greedy decoding, thinking
disabled, and a 512-token output limit per case. The [offline evaluation](baseline-qwen3-4b-extraction.json)
uses the 24-case diagnostic fixture.

| Check | Result |
| --- | ---: |
| JSON parse failures | 0 |
| Strict schema failures | 4 of 24 responses |
| Parsed claims | 24 |
| Claims with valid evidence integrity | 0 of 24 |
| Evidence errors | 17 quote mismatches, 7 out-of-bounds spans, 1 attribution-role conflict |
| Exact structured F1 | 0.000, with 0 true positives, 24 false positives, and 26 false negatives |
| Correct abstentions | 2 of 4 no-memory cases |
| Truncated outputs | 0 |

Errors can overlap. The exact score checks schema, claim fields, evidence IDs, offsets,
and quotes. F1 of zero does not show that every generated sentence is semantically
false. It combines citation, schema, field, and exact-rendering mismatches. The
[qualitative audit](extraction-audit.md) details the observed modality, date, attribution,
and relation errors.

## Qwen3-4B explicit-v2 extraction

The [raw explicit-v2 receipt](../examples/synthetic/qwen3-4b-explicit-v2-remote.json)
has SHA-256
`e624dcab7ff7bb7c7214e4245ff9659509be7331459b9193f2c76730c8fc9dcd`. The
[offline evaluation](baseline-qwen3-4b-explicit-v2-extraction.json) uses the same 24
diagnostic cases as V1, with payload SHA-256
`ee0d99b5c14e98279a7c598c8d25b5f72baf4a8b137dbb03fb1b03bebd4d47e1`. This is a
prompt-development comparison, not an independent or held-out evaluation.

| Check | Result |
| --- | ---: |
| JSON parse failures | 1 of 24, case 008 truncated at the 512-token output limit |
| Strict schema failures | 1 of 24, case 022 used year-only `valid_from: "2023"` on an abandoned plan |
| Parsed claims | 21 |
| Raw claims with valid evidence integrity | 0 of 21 |
| Raw evidence errors | 16 quote mismatches, 5 out-of-bounds spans, no role conflicts |
| Exact structured F1 | 0.000, with 0 true positives, 21 false positives, and 26 false negatives |
| Correct abstentions | 3 of 4 no-memory cases |

The [explicit-v2 anchoring report](baseline-qwen3-4b-explicit-v2-anchoring.json)
repairs uniquely located exact quotes in all 21 parsed claims, raising evidence
integrity to 21/21 with no role conflicts. The parse failure, schema failure, and exact
F1 remain unchanged. As with V1, repaired offsets do not establish entailment or
semantic claim support. The higher no-memory abstention count is descriptive on the same
small fixture, not evidence of calibrated abstention.

## Unique-quote anchoring replay

The [anchoring report](baseline-qwen3-4b-anchoring.json) tests deterministic repair of
offsets only when a quote occurs exactly once in its declared source message. It leaves
the raw receipt unchanged and does not establish entailment.

After anchoring, 21 of 24 claims pass evidence-integrity checks. Three still have role
conflicts: one attribution-role conflict and two source-role mismatches. The four schema
failures remain, and exact structured F1 remains 0. This is not a model-generated repair,
semantic verification, or a basis for auto-accepting any claim.

## Remote run record

All four Modal calls used authored synthetic data only, and all four apps were verified
stopped with zero active tasks. No private corpus was uploaded. No training ran. The model
weights were fetched remotely; none were downloaded to this Mac.

| Run | GPU compute | Model load | Remote function | Client wall including image/startup | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM embedding | 0.752113 s | 9.295326 s | 13.315121 s | 129.854275 s | 59,023,872 bytes |
| Qwen3-4B V1 extraction | 304.772849 s | 73.579751 s | 382.079294 s | 399.468317 s | 8,822,848,512 bytes |
| Qwen3-4B explicit-v2 extraction | 220.447001 s | 60.141208 s | 283.622759 s | 291.010439 s | 8,822,848,512 bytes |
| NLI support diagnostic | 1.558278 s | 9.804156 s | 11.563326 s | 24.168136 s | 826,340,864 bytes |

The V1 Qwen run used a Tesla T4 and processed 8,560 input tokens and 5,978 generated
tokens without truncation. The explicit-v2 run used the same GPU and processed 23,632
input tokens and 4,231 generated tokens; case 008 was truncated at 512 generated tokens.
All four Modal apps were verified stopped with zero active tasks. At the nominal
allocation rate of $0.00022572 per second, applying the rate to the combined client wall
time of 844.501167 seconds gives an estimated $0.19062. This is not a billing receipt,
actual charge, or provider-enforced spend cap. See [compute notes](modal-compute.md) for
receipt limits and runner controls.

The MiniLM receipt has core package versions but no full environment lock or runner Git
hash. The V1 Qwen receipt has a broader package list, Python and CUDA versions, and GPU
name, but no runner Git hash. The explicit-v2 receipt records runner file SHA-256
`6624bb9281438cdb22173afaea2066667c350cd9fabee98233f1825c5eb3c066`, not a Git commit;
it also records package versions, Python and CUDA versions, and GPU name. These receipts
do not capture an immutable full runtime build.

## Offline replay

From the repository root, with the project dependencies installed, recompute the
retrieval and extraction reports from the saved raw receipts:

```sh
PYTHONPATH=src python3 -m experiments.evaluate_baselines \
  --input examples/synthetic/minilm-remote.json \
  --output /tmp/tracemem-minilm-retrieval.json

PYTHONPATH=src python3 -m experiments.evaluate_baselines \
  --input examples/synthetic/qwen3-4b-remote.json \
  --output /tmp/tracemem-qwen3-4b-extraction.json \
  --prompt-version v1

PYTHONPATH=src python3 -m experiments.evaluate_anchoring \
  --input examples/synthetic/qwen3-4b-remote.json \
  --output /tmp/tracemem-qwen3-4b-anchoring.json \
  --prompt-version v1

PYTHONPATH=src python3 -m experiments.evaluate_baselines \
  --input examples/synthetic/qwen3-4b-explicit-v2-remote.json \
  --output /tmp/tracemem-qwen3-4b-explicit-v2-extraction.json \
  --prompt-version explicit-v2

PYTHONPATH=src python3 -m experiments.evaluate_anchoring \
  --input examples/synthetic/qwen3-4b-explicit-v2-remote.json \
  --output /tmp/tracemem-qwen3-4b-explicit-v2-anchoring.json \
  --prompt-version explicit-v2
```

These commands make no model or network calls. They validate local receipts against
the current fixtures and write reports to `/tmp`.

## Dataset readiness and integrated validation

The bounded [dataset readiness audit](dataset-readiness.json) is complete. It reports 24
diagnostic cases, 24 distinct scenario IDs, zero exact duplicate pairs, and zero assigned
train, dev, test, or held-out cases. No-memory cases: 4, all in the diagnostic split.
Cross-split near-duplicate checks are vacuous because the fixture has no evaluation
partitions. The report does not establish entity or template independence, legal rights,
annotation quality, sample adequacy, or Gate A approval. Reproduce it with:

```sh
make audit-data
# or: PYTHONPATH=src python3 -m experiments.audit_dataset
```

The [three-call baseline validation receipt](validation-phase2-baseline.json) is complete.
It records 100 tests passing with no skips, Ruff check and format passing across 34 files,
strict source mypy passing across 16 files, scoped experiment mypy passing across 19 files,
and an offline wheel build/install smoke test passing. The receipt includes the complete
commands, hashes, environment, and limitations; it validates only the three-call baseline.

Gate A is not met. The next evidence boundary is a fresh, independently audited scenario
dataset with an independent semantic-support evaluation, followed by an extraction run
against another base candidate. Matched-budget downstream QA remains later work. Do not
reuse this 24-case fixture as held-out data or infer semantic support from exact quote
integrity.

## NLI support verifier diagnostic

The fourth source-free call evaluated 48 newly authored premise/hypothesis pairs across
24 fictional scenarios with
[cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
at revision 6c749ce3425cd33b46d187e45b92bbf96ee12ec7. The fixture hash is
582612fc24e60868c8b0b1f37745188dbf143a41330a5777152b5cce9adc90e9; the public raw
receipt hash is
16be7f4c542bd174e49f975bfc3d8baa7ef9995fdd5d8cfcf2e3df03893f3871. The model label
mapping is contradiction=0, entailment=1, neutral=2. The pairs are agent-reviewed
diagnostics, not independently annotated ground truth or held-out data. A distinct
[data card](support-challenge-data-card.md) documents the fixture and its limits.

| Entailment threshold | Supported precision | Supported recall | Coverage | TP / FP / FN / TN |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.8276 (95% interval 0.7273–0.9600) | 1.0000 | 29/48 (0.6042) | 24 / 5 / 0 / 19 |
| 0.90 | 0.8214 (95% interval 0.7188–0.9545) | 0.9583 | 28/48 (0.5833) | 23 / 5 / 1 / 19 |
| 0.99 | 0.8750 (95% interval 0.7667–1.0000) | 0.8750 | 24/48 (0.5000) | 21 / 3 / 3 / 21 |

Intervals are descriptive 1,000-draw scenario-bootstrap percentile intervals, not
population guarantees. The [support audit](support-audit.md) lists every fixture-labelled
false positive and false negative. It shows both high-confidence unsupported selections
and supported examples rejected at higher thresholds, so these scores do not justify an
acceptance threshold or automatic claim verification.

The corrected offline replay reproduced the saved computed report. Its raw receipt
records the launch-time runner SHA; that value is not asserted to match the current
runner, whose current source SHA-256 is
187fd7f4f5d94da1efb5e60f2ea7c2001773549002accac4326c82e09819dc02. The launch-time
runner snapshot SHA-256 is
d2ad3ffe144e500a4c00574cb5f41aaeb4339723e7cb5b15ca4969c06e60aaab and is preserved
hash-verified under ignored .local/runner-snapshots/. Twenty-five focused tests passed.
Final integrated checks passed 136 tests with no skips, Ruff check/format across 41 files,
strict source mypy across 18 files, and scoped evaluator mypy across 22 files. The
fresh wheel smoke passed from an isolated install without PYTHONPATH/PYTHONHOME; its
48-case support replay matched the computed report exactly. See docs/validation.json for
the durable validation record. The completed
[three-call baseline validation receipt](validation-phase2-baseline.json) remains
unchanged and does not cover this NLI slice.
