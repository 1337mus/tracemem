# Independent support diagnostic

2026-10-14. The validated three-call baseline is preserved in
`validation-phase2-baseline.json`. Both Qwen prompt variants still confuse
modality, entities, attribution and dates after offset repair. A source-free NLI
diagnostic has now been measured against authored synthetic hard negatives. The final
integrated suite and fresh installed-wheel smoke passed; no model-based acceptance policy
is proposed.

Create 48 fictional premise/hypothesis pairs in 24 new scenarios, each containing
one supported and one unsupported hypothesis. All are diagnostic. Include
attribution, time, modality, negation, causality, ownership/location, referents,
and instructions embedded in evidence. Preserve statements about what someone
reported without treating them as established physical facts. The root agent
reviews every annotation; this is engineering review, not independent human
ground truth. No private examples, training data or held-out claim is implied.

Use `cross-encoder/nli-deberta-v3-base` at revision
`6c749ce3425cd33b46d187e45b92bbf96ee12ec7`. The public card declares Apache-2.0;
the snapshot has no standalone license file. Config labels are contradiction=0,
entailment=1, neutral=2. Pin and verify this mapping. Use the existing image
packages, FP32 GPU inference, batch 8, and a 512-token limit without silent
truncation. Weights stay remote. The new runner must not modify the frozen
Qwen runner or old receipts.

Only premise, hypothesis, IDs and provenance hashes enter the remote payload.
Gold labels and scenario metadata stay local. Require payload-hash approval
and a fresh personal-workspace check. One T4, CPU 2, RAM 16 GiB, maximum one
container, no warm pool/retries/detach/schedule/source mounts. Startup and
execution timeouts are each at most 300 seconds. One additional synthetic call
fits the existing inexpensive inference scope and $2 operational target; that
target is not an enforced provider billing cap. No training or private uploads.

Preserve untouched raw logits/probabilities and the launch-time runner hash.
Validate receipt schemas, finite numbers, exact case coverage and probability
consistency. Report support precision/recall, unsupported false positives,
coverage at fixed diagnostic thresholds 0.5/0.9/0.99, and descriptive bootstrap
intervals resampling whole scenarios. Zero selected claims means undefined
precision. Scores are uncalibrated diagnostics; no claim becomes ACCEPT.

Acceptance evidence includes the audited fixture, reviewed source-free runner, actual
remote receipt, corrected offline replay, and stopped app. Final integrated validation
passed 136 tests with no skips, Ruff check/format across 41 files, strict source mypy
across 18 files, and scoped evaluator mypy across 22 files. A fresh wheel with SHA-256
deb3caccc5c429ec55e8eac5eef94f6ab77babf358ae63174b40102dd3df8287 installed without
PYTHONPATH or PYTHONHOME; module and console demos passed. The support replay and report
matched exactly in that install. See docs/validation.json for the durable validation
record. A positive NLI result would justify a larger independent support study, not
deployment or Gate A approval. A negative result is useful evidence too.

## Recorded run

The fixture contains 48 pairs across 24 scenarios, evenly split between supported and
unsupported labels. The
[fixture](../src/tracemem/data/support_challenge.json) SHA-256 is
582612fc24e60868c8b0b1f37745188dbf143a41330a5777152b5cce9adc90e9; its labels were
agent-reviewed and are not independent human truth. The distinct
[data card](support-challenge-data-card.md) documents authorship, scope and limitations.
The pinned model is cross-encoder/nli-deberta-v3-base,
revision 6c749ce3425cd33b46d187e45b92bbf96ee12ec7. Labels map 0 to contradiction,
1 to entailment and 2 to neutral.

| Entailment threshold | Supported precision | Supported recall | Coverage | TP / FP / FN / TN |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.8276 (95% interval 0.7273–0.9600) | 1.0000 | 29/48 (0.6042) | 24 / 5 / 0 / 19 |
| 0.90 | 0.8214 (95% interval 0.7188–0.9545) | 0.9583 | 28/48 (0.5833) | 23 / 5 / 1 / 19 |
| 0.99 | 0.8750 (95% interval 0.7667–1.0000) | 0.8750 | 24/48 (0.5000) | 21 / 3 / 3 / 21 |

The intervals are descriptive scenario-bootstrap percentiles over the 24 grouped
scenarios. The [support audit](support-audit.md) lists every false positive and false
negative. It documents high-confidence false positives as well as supported pairs missed
at stricter thresholds; none of these thresholds is calibrated, and no claim may be
auto-accepted.

The saved public raw receipt
[nli-deberta-remote.json](../examples/synthetic/nli-deberta-remote.json) has SHA-256
16be7f4c542bd174e49f975bfc3d8baa7ef9995fdd5d8cfcf2e3df03893f3871. The corrected
offline replay reproduced [baseline-nli-support.json](baseline-nli-support.json).
Remote timing was 1.558278 GPU seconds, 9.804156 seconds model load, 11.563326 seconds
remote function elapsed, and 24.168136 seconds client wall including image and startup;
peak GPU memory was 826,340,864 bytes on a Tesla T4. All four Modal apps are reported
stopped. The raw receipt's runner digest is the launch-time SHA
d2ad3ffe144e500a4c00574cb5f41aaeb4339723e7cb5b15ca4969c06e60aaab; its hash-verified
snapshot is preserved under ignored .local/runner-snapshots/. The output-path hard-link
alias guard is now in the runner. The current experiments/modal_support.py SHA-256 is
187fd7f4f5d94da1efb5e60f2ea7c2001773549002accac4326c82e09819dc02. Twenty-five focused
tests, strict source/evaluator type checks, and Ruff passed. Final integrated checks
passed 136 tests with no skips, Ruff check/format across 41 files, strict source mypy
across 18 files, and scoped evaluator mypy across 22 files. The fresh wheel smoke used
an isolated install at /private/tmp/tracemem-wheel-current-037cbxak with no
PYTHONPATH/PYTHONHOME; the 48-case support replay exactly matched this report. The
existing three-call baseline validation receipt remains unchanged and does not validate
this slice.
