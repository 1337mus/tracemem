# Documentation index

Paths in this index are relative to this `docs/` directory.

## Start here

- [Project status and design](project-status.md): current system boundary,
  architecture, measurements, and open research gates.
- [Roadmap](roadmap.md): phase status and acceptance boundaries.
- [Handoff](handoff.md): current work checkpoint and what to verify before resuming.
- [Handoff archive (2026-10-15 v2)](handoff-history-2026-10-15-v2.md): byte-identical pre-compaction project history.
- [Implementation brief](implementation-brief.md): initial scope and design notes.
- [Phase 1 validation](validation-phase1.json) and [latest finalized validation](validation.json): machine-readable check records.
- [Phase 2 baseline validation](validation-phase2-baseline.json): completed checks
  for the frozen retrieval and extraction baseline.

## Source adapters and ingestion

- [Transcript bounds](transcript-bounds.md): bounded Claude and Codex JSONL reads,
  cap overrides, and rollback behavior.
- [Transcript consistency](transcript-consistency.md): optional detection of
  ordinary source changes during a full import.
- [ChatGPT export reader](chatgpt-export.md) and its [design brief](chatgpt-export-brief.md):
  explicit JSON parsing and supported source shape.
- [ChatGPT snapshot import](chatgpt-snapshot.md) and its [brief](chatgpt-snapshot-brief.md):
  fresh-database behavior and refusal cases.
- [Context Kit inventory](context-kit-inventory.md): source discovery, configuration,
  and integration boundaries.
- [CI coverage](ci.md) and the [CI brief](ci-brief.md): checks and release packaging.
- [Installed API coverage](installed-new-apis-brief.md): installed-package API checks.

## Evidence storage and retrieval

- [Exact citation retrieval](citation-resolution-brief.md): resolve a historical
  source revision after later imports and database reopen.
- [Local claim ledger](ledger-brief.md): proposal history, evidence retention,
  and current-fact boundaries.
- [Fixed context budgets](context-budget.md) and the [design brief](context-budget-brief.md):
  passage retention under UTF-8 byte ceilings.
- [Token-budget packing](token-budget-packing.md): whole-passage selection and
  caller-supplied full-prompt token counts.
- [Token-count catalog](token-count-catalog.md), [results](token-count-results.md),
  and [runner](modal-token-count.md): offline prompt catalog and bounded measurement.
- [BM25 scale results](lexical-scale-results.md) and the [scale brief](lexical-scale-brief.md):
  indexed FTS behavior and synthetic corpus measurements.
- [FTS scale brief](fts-scale-brief.md): storage and indexing design notes.
- [Source-retention impact](source-retention.md) and its [brief](source-retention-brief.md):
  dependency impact for historical evidence and proposal lineage.
- [Persisted-store retention report](retention-report.md) and its [brief](retention-report-brief.md):
  bounded copied-byte analysis of eligible local databases.
- [Integrity versus support](integrity-selectivity.md) and its [brief](integrity-selectivity-brief.md):
  retrospective citation-filter replay against separate review labels.

## Experiments and model research

- [Training-target audit](training-target-review.md): seven source-preserving
  corrections, literal review checks and a reproducible 768-row corpus.
- [Fourth experiment protocol](sft-quote-v4-protocol.md): fresh evaluation,
  matched base/prior/tuned arms and predeclared success criteria; preparation only.

- [Third extraction experiment protocol](sft-quote-v3-protocol.md): fixed-budget
  chronology and subject corrections, three matched arms and a separate frozen test.

- [Third SFT run: quote-v3](sft-quote-v3-results.md), [aggregate evaluator report](sft-quote-v3-results.json),
  and [validation record](validation-sft-quote-v3.json): mixed provisional agent/root
  review, required-fact coverage decline, paired accounting, and exact synthetic
  job, receipt, and review artifacts.
- [Second SFT run: quote-v2](sft-quote-v2-results.md), [aggregate results](sft-quote-v2-results.json),
  and [validation record](validation-sft-quote-v2.json): strict raw scoring,
  separate fence-only analysis, agent-only semantic review, and exact fictional
  job, receipt, and audit copies.
- [First training pilot](sft-pilot.md): fictional data and the completed bounded
  QLoRA run. [Results](sft-pilot-results.md) include matched base/tuned
  development metrics and an offline replay record.

- [Experiments](experiments.md): commands, measurements, and failure cases.
- [Phase 2 diagnostics](experiments-phase2.md): retrieval and extraction diagnostic receipts.
- [Model candidates](model-candidates.md): pinned retrieval, extraction, and verifier comparisons.
- [Multi-session comparison](experiments-scenarios.md): frozen fictional scenario comparison.
- [Scenario data card](scenarios-data-card.md) and [annotation workflow](annotation-workflow.md):
  fixture scope and provisional labels.
- [Longer development conversations](longitudinal-data-card.md): new fictional
  cases with distant corrections and separate semantic foils; agent labels only.
- [Scoped defaults and exceptions](scope-exceptions-data-card.md), [replay report](scope-exceptions-report.json),
  [semantic audit](scope-exceptions-semantic-audit.json), and
  [validation record](validation-scope-exceptions.json): six fictional dev cases
  for branch scope, attribution, dated exceptions, canceled proposals, and prerequisites.
- [Longitudinal revision v2](longitudinal-revision-v2.md): the same fictional
  development slice with one unresolved candidate kept unlabeled; see its
  [validation record](validation-longitudinal-v2.json).
- [Longitudinal v2 review slice](longitudinal-v2-review.md) and its
  [validation record](validation-longitudinal-v2-review.json): targeted agent review,
  preserved artifacts, source/installed replay parity, and limits.
- [Cross-session test data card](cross-session-test-data-card.md),
  [retrieval replay](cross-session-test-retrieval.json), and
  [validation record](validation-cross-session-test.json): six newly authored
  fictional cases and a CPU turn-retrieval diagnostic; labels remain provisional
  and do not establish held-out quality.
- [Cross-session MiniLM comparison](cross-session-minilm.md), [replay report](cross-session-minilm-retrieval.json),
  and [validation record](validation-cross-session-minilm.json): remote MiniLM
  vectors compared with BM25, hash vectors, and hybrid retrieval on the same
  small fictional slice; no held-out or QA quality claim.
- [Dataset readiness report](dataset-readiness.json): assigned split and duplicate counts.
- [Dataset isolation audit](dataset-isolation.md), [report](dataset-isolation-report.json),
  and [validation record](validation-dataset-isolation.json): cross-split lexical
  and declared-identifier diagnostics.
- [Scenario curation notes](scenario-curation-notes.md) and [scenario dataset brief](scenario-dataset-brief.md):
  fixture construction records.
- [Modal compute](modal-compute.md): bounded synthetic inference runs and resource notes.
- [Related work](related-work.md): adjacent projects and references.

## Review and QA diagnostics

- [Offline review page](offline-review.md) and its [brief](offline-review-brief.md):
  inspect blind candidates beside source evidence and export responses.
- [Review uncertainty](review-uncertainty.md) and its [brief](review-uncertainty-brief.md):
  scenario-cluster intervals and undefined resamples.
- [Agent support review](scenarios-reviewed-support.json),
  [root adjudication](scenarios-root-adjudication.json), and
  [review uncertainty data](scenarios-review-uncertainty.json): separate reviewer artifacts.
- [Scenario tag breakdown](scenario-phenomena.md) and its [brief](scenario-phenomena-brief.md):
  source-tag cells without causal attribution.
- [Extraction audit](extraction-audit.md) and [abstention diagnostic](abstention-diagnostic.md):
  focused output and abstention checks.
- [Independent support diagnostic](verification-brief.md), its [error audit](support-audit.md),
  and [challenge data card](support-challenge-data-card.md).
- [Synthetic QA preparation](synthetic-qa.md), [analysis](synthetic-qa-analysis.md),
  [answer protocol](synthetic-qa-answer-protocol.md), and [Qwen results](synthetic-qa-qwen3-results.md).
- [Synthetic QA analysis brief](synthetic-qa-analysis-brief.md),
  [preparation brief](synthetic-qa-brief.md), [Modal brief](synthetic-qa-modal-brief.md),
  and [Modal run](synthetic-qa-modal.md).
- [QA contrast preparation](qa-contrast-preparation.md), [source-only requests](qa-contrast-requests.md),
  [token counter](qa-contrast-token-counter.md), and [token replay](qa-contrast-token-replay.md).
- [QA contrast answers](qa-contrast-answer-results.md),
  [review results](qa-contrast-review-results.md), and [review accounting](qa-contrast-review-accounting.md).
- [QA controls](qa-interpretation-controls.md), [answer contract](qa-controls-answer-contract.md),
  [answer runner](qa-controls-answer-runner.md), and [answer run results](qa-controls-answer-run-results.md).
- [QA controls answer review](qa-controls-answer-review.md),
  [review results](qa-controls-review-results.md), and [review accounting](qa-controls-review-accounting.md).
- [QA and review briefs](synthetic-qa-brief.md), [contrast preparation brief](qa-contrast-preparation.md),
  and [phase 2 prompt brief](phase2-prompt-brief.md).
- [Fictional review artifacts](../examples/synthetic/reviews/README.md) reproduce
  the agent-support report.
- [Longitudinal v2 review artifacts](../examples/synthetic/reviews/longitudinal-v2/manifest.json):
  copied probe responses, report, and unblinded root audit with byte hashes.

## Benchmarks and dataset formats

- [Benchmark contracts](benchmark-contracts.md) and
  [machine-readable contracts](benchmark-contracts.json): evaluator semantics,
  source hashes, and rights questions.
- [LongMemEval compatibility probe](longmemeval-compatibility.md): local oracle-format parsing.
- [LongMemEval format adapter](longmemeval-adapter.md) and its [brief](longmemeval-adapter-brief.md):
  fixture parsing, label separation, and historical citations.
- [LongMemEval metadata and rights audit](longmemeval-rights.md): dataset metadata
  and unresolved provenance questions.
- [LongMemEval retrieval](longmemeval-retrieval.md) and its [brief](longmemeval-retrieval-brief.md):
  per-question turn-retrieval diagnostics.
- [Bounded LongMemEval file scan](longmemeval-file.md) and its [brief](longmemeval-file-brief.md):
  incremental validation for larger local-format files.
- [Selected-file retrieval](longmemeval-selection.md) and its [brief](longmemeval-selection-brief.md):
  explicit question selection and score denominators.
- [Frozen-five oracle results](longmemeval-oracle-retrieval-results.md):
  locally verified turn-label recall, without QA scoring.

## Data governance and validation records

- [Data governance](data-governance.md): private-source handling, synthetic data,
  rights evidence, and release gates.
- [LongMemEval rights metadata](longmemeval-rights.json): pinned metadata audit record.
- [Current review validation](validation-qa-controls-review-results.json),
  [answer contract validation](validation-qa-controls-answer-contract.json), and
  [answer runner validation](validation-qa-controls-answer-runner.json).
- [Token-budget packing validation](validation-token-budget-packing.json),
  [source-retention validation](validation-source-retention.json), and
  [support-ledger validation](validation-support-ledger.json).
- [Transcript-bound validation](validation-transcript-bounds.json) and
  [transcript-consistency validation](validation-transcript-consistency.json).
