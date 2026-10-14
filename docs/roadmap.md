# Execution state

Scope: local TraceMem prototype. Phase 0/1 are implemented. Phase 2 has
synthetic-only diagnostics and a completed frozen comparison, but its
acceptance boundary is not met.

| Phase | State | Deliverable and remaining acceptance boundary |
| --- | --- | --- |
| 0 | Complete | Verify Context Kit inventory, original-source schema, and aggregate data-quality observations. |
| 1 | Complete | Read-only provider adapters, append-only evidence revisions, BM25/hash/hybrid retrieval, exact-revision citations, synthetic demo, and retrieval diagnostics. New stores use schema-3 indexed FTS head-rowid maintenance; schema-2 stores are rejected read-only and are not migrated. Synthetic scale receipts show the same rankings before and after the storage change; see [BM25 scale results](lexical-scale-results.md). The explicit ChatGPT reader and fresh-store snapshot CLI are verified only against fictional fixtures; actual-export compatibility remains unverified. Full private-corpus ingestion is out of scope. |
| 2 | In progress | Historical 24-case claim-challenge and prompt diagnostics remain development evidence only. A separate frozen 48-case multi-session comparison has 24 dev/24 test cases and 43 provisional agent-authored targets; Qwen and Phi runs are complete. The 43-target reference review had no negative controls; a separate simple 16-task dev-derived control check had 8 positives and 8 negatives, with one agent agreeing 16/16 and zero human reviewers. Generated-candidate review is complete for 84 schema-valid claims, with a separate nonblind root adjudication. Gate A remains unmet. See the [scenario comparison](experiments-scenarios.md), [data card](scenarios-data-card.md), [annotation workflow](annotation-workflow.md), and [model report](model-candidates.md). |
| 3 | Not started | Consider model selection and reproducible SFT only after rights, leakage, independent validation, and owner-approval gates pass. No training is authorized now. |
| 4 | In progress (early engineering) | A local append-only proposal ledger records claims and proposal lineage. Citation fetching resolves exact evidence revisions when callers retain the EvidenceStore separately; the ledger stores claim quotes, not full source revisions. Annotation exports and per-model/split/reviewer accounting support blind review but do not promote labels or establish semantic acceptance. Supersession records proposal lineage, not current-fact truth. Current-fact resolution and source erasure remain unimplemented. See the [ledger brief](ledger-brief.md), [annotation workflow](annotation-workflow.md), and [scenario comparison](experiments-scenarios.md). |
| 5 | Fictional diagnostic complete; offline packing delivered privately; prompt-count catalog delivered privately | The eight-case fixture has a completed 72-answer Qwen3-4B diagnostic and a root-replayed literal analysis. Every cell scores 1/4 overall and 0/3 on answerable questions; the result is provisional and cannot compare retrieval modes or context ceilings because each question received identical nonzero context across modes and ceilings. Generic whole-line token-budget packing now uses a caller-supplied full-prompt counter and has passed root review and offline validation. Exact reviewed packing source is signed in commit `717ff8c986238cbbf5b133238df9ad17729bc8bb`, delivered privately in `d7bb4f8bec846295ecb2eb8a96e0e55aff20dc79`; [hosted CI](https://github.com/1337mus/tracemem/actions/runs/38650751759) passed for that packing delivery. A separate offline synthetic prompt-count catalog is root-reviewed in signed commit `2f959a8347fd4ccbb97c4db100c1d1b587746650`; root's full suite, independent oracle, and installed-wheel replay passed. The catalog source and validation docs were delivered privately at `403cc47edc54ae462466cec0522842dd6aceda01`; [catalog CI](https://github.com/1337mus/tracemem/actions/runs/38652529551) passed with 460 source tests (one skip) and 208 installed tests (two skips). Replay uses invented caller-supplied counts only; no tokenizer-attested measurement or matched-token QA has run. See the [split results](synthetic-qa-qwen3-results.md), [analysis](synthetic-qa-analysis.md), [packing contract](token-budget-packing.md), [packing validation](validation-token-budget-packing.json), [catalog API](token-count-catalog.md), and [catalog validation](validation-token-count-catalog.json). |
| 6 | Not started | Prepare an owner-reviewed release of code, data, and weights only after leakage, rights, privacy, and memorization review. |

## Evaluation state

The earlier 24-case claim challenge is a single `diagnostic` partition, with no
assigned train/dev/test/held-out cases. Its Qwen V1 and explicit-v2 runs share
the same cases and are prompt-development diagnostics, not independent
evaluations. The measured NLI slice is another diagnostic; its agent-reviewed
labels are not independent human truth, and observed false positives do not
support a verifier policy. See the [Phase 2 report](experiments-phase2.md),
[support audit](support-audit.md), and [dataset readiness report](dataset-readiness.json).

The newer frozen scenario comparison is a different fixture and evaluation:
48 authored fictional cases, 24 each in dev and test, eight no-memory cases,
and 43 provisional expected claims. Root audited the source cases and targets;
a separate blind agent review agreed with all 43 authored targets. That
reference-only review had no negative controls and no human reviewers. A
different 16-task dev-derived control check included eight positives and eight
simple negatives; one blind agent agreed with all 16. This shows discrimination
on those controls only, not semantic calibration, independent truth or held-out
performance. Qwen and Phi have both completed their frozen runs, and all six
synthetic Modal apps stopped with zero tasks. No model has been selected and no
training has run. Details and limitations are in the [data card](scenarios-data-card.md)
and [comparison report](experiments-scenarios.md).

Read-only citation checks verify exact quote and source-revision identity;
annotation decisions separately ask whether a conversation supports a claim.
Review accounting preserves reviewer kind, model, split, decisive denominators,
and exclusions for parse/schema failures. These are distinct integrity and
measurement workflows. Neither a valid citation, an agreed authored reference,
nor an exact structured score establishes semantic acceptance. An agent review
covered all 84 schema-valid candidates with original totals of 50 supported, 31
unsupported and 3 uncertain. A separate nonblind root adjudication changed nine
labels (two to unsupported, seven to uncertain), yielding 41 supported, 33
unsupported and 10 uncertain. The root labels are not independent, human,
consensus or automatic acceptance. Parse/schema exclusions remain explicit. See
the [per-model/split accounting](scenarios-reviewed-support.json) and
[adjudication receipt](scenarios-root-adjudication.json).

The retrospective [integrity-versus-support diagnostic](integrity-selectivity.md)
compares the existing raw and quote-anchored citation checks against each reviewer
separately. It preserves duplicate claim origins, uncertainty and output exclusions.
An integrity pass remains a proposal for review; it does not establish semantic
support, and the agent judgments do not establish calibration.

The bounded dataset readiness audit found no exact or near-duplicate lexical
flags across the assigned partitions, but that result cannot establish legal
rights, entity separation, template independence, annotation quality, or Gate A.
The lexical method's limitations are recorded in the [readiness report](scenarios-readiness.json).

Do not equate source-text revisions with changes in real-world truth. Current
assertions, temporal validity, semantic support, unknown-query calibration, and
downstream answer quality remain separate problems. The retrieval and extraction
diagnostics do not establish training need, model superiority, publication
readiness, or novelty.
