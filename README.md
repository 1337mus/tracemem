# TraceMem

A local research prototype for conversational memory with original-source evidence.

TraceMem reads Claude Code and Codex transcripts, preserves source roles and exact
text revisions in SQLite, and retrieves cited passages with BM25, local vectors,
or reciprocal rank fusion. An explicit [ChatGPT JSON reader](docs/chatgpt-export.md)
also supports a conservative selected-branch fixture contract. The runnable demo
contains fictional synthetic data.

**Status:** Phase 0/1 is runnable. Phase 2 has completed synthetic-only Qwen3-4B
and Phi-3.5-mini diagnostics on 48 fictional multi-session cases, with 24 dev and
24 test cases. Most Phi outputs failed the strict schema; Qwen still produced
attribution, temporal and condition errors. No model is selected. The authored
targets are provisional and non-exhaustive; agent support reviews and root
adjudication are reported separately. Gate A is not met. No training, independent
human semantic evaluation, downstream QA result, release or novelty claim exists.

The prototype also includes an append-only candidate ledger, exact historical
citation resolution and blind review tooling. Callers must retain matching
EvidenceStore revisions separately; the ledger stores claim quotes. It does not
resolve current real-world truth or automatically accept semantic claims.
New stores use evidence schema 3 and indexed FTS maintenance; older stores are
rejected read-only, with no automatic migration. Default vectors use deterministic
token hashing; pinned MiniLM is a separate measured retrieval diagnostic.
See [current validation](docs/validation.json) for the last finalized checkpoint
and [handoff](docs/handoff.md) for live work.

## Run offline

Python 3.11+ with SQLite FTS5 is enough. No runtime packages or API keys are required.

```sh
make demo
make benchmark
make test
```

Try the fictional ChatGPT export with a fresh destination:

```sh
mkdir -p .local/chatgpt-demo
PYTHONPATH=src python3 -m tracemem import-chatgpt --input examples/synthetic/chatgpt-export.json --db .local/chatgpt-demo/evidence.sqlite3
PYTHONPATH=src python3 -m tracemem search --db .local/chatgpt-demo/evidence.sqlite3 --query "Cedar Café" --mode bm25
```

The snapshot importer preserves the source and refuses existing outputs. Use a
new database path for another import. See [snapshot behavior](docs/chatgpt-snapshot.md)
and [CI coverage](docs/ci.md). Actual ChatGPT-export compatibility is unverified.

Optional developer validation:

```sh
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.lock
make check
```

## Read the evidence

- [Context Kit inventory](docs/context-kit-inventory.md): verified source architecture,
  schemas, aggregate coverage and integration limits.
- [Experiments](docs/experiments.md): commands, measurements and failure cases.
- [Phase 2 diagnostics](docs/experiments-phase2.md): pinned MiniLM retrieval and Qwen3-4B
  V1 and explicit-v2 extraction receipts, anchoring replays, and remaining gates. Both
  prompt runs use the same diagnostic cases and are not independent evaluations.
- [Dataset readiness report](docs/dataset-readiness.json): exact duplicate and assigned
  split counts for the current challenge. It does not establish held-out quality or Gate A.
- [Independent support diagnostic](docs/verification-brief.md): measured synthetic NLI
  verifier slice, its [error audit](docs/support-audit.md), and the distinct
  [support-challenge data card](docs/support-challenge-data-card.md).
- [Three-call baseline validation](docs/validation-phase2-baseline.json): completed checks
  for the frozen retrieval/extraction baseline only, not the NLI verifier slice.
- [Data governance](docs/data-governance.md): private-source handling and release gates.
- [Pinned benchmark contracts](docs/benchmark-contracts.md): exact LongMemEval and
  LoCoMo evaluator semantics, source hashes, and unresolved dataset rights. No benchmark
  conversations were downloaded or evaluated.
- [LongMemEval format adapter](docs/longmemeval-adapter.md): strict fixture-only parsing,
  separate question stores, reference-label separation, and historical Unicode citations.
  The [dataset metadata audit](docs/longmemeval-rights.md) records the MIT card declaration
  and release sizes; full S/M files exceed the current parser's 64 MiB cap.
- [Turn-retrieval diagnostic](docs/longmemeval-retrieval.md): a local CLI with per-question
  stores, explicit missing-label exclusions, and separate BM25/hash/hybrid recall cells.
  Its fictional protocol checks do not measure benchmark or QA performance.
- [Source-retention impact](docs/source-retention.md): a pure dependency diagnostic
  for historical evidence, candidate quotes, ambiguous spans and correction lineage.
  It opens no stores and performs no deletion.
- [Persisted-store retention report](docs/retention-report.md): bounded copied-byte
  analysis of quiescent rollback-mode databases, with integrity checks and a new
  private report. WAL databases and journal sidecars are refused.
- [Model candidates](docs/model-candidates.md): pinned extraction/retrieval/verifier
  diagnostics and the completed Qwen/Phi scenario comparison. No model is selected.
- [Multi-session comparison](docs/experiments-scenarios.md): frozen explicit-v2
  comparison on 24 dev/24 test fictional cases, with [data card](docs/scenarios-data-card.md)
  and [annotation workflow](docs/annotation-workflow.md). Labels remain provisional.
- [Modal compute](docs/modal-compute.md): six completed synthetic-only inference calls,
  stopped apps, and bounded-resource notes. Weights and inference stay remote.
- [BM25 scale measurements](docs/lexical-scale-results.md): matching corpus/rankings
  at 10k/50k heads, bounded Python allocations and indexed FTS maintenance.
- [Agent support review](docs/scenarios-reviewed-support.json): separate model/split/
  reviewer counts, explicit exclusions and [root adjudication](docs/scenarios-root-adjudication.json).
  [Fictional review artifacts](examples/synthetic/reviews/README.md) reproduce the report.
- [Review uncertainty](docs/review-uncertainty.md): scenario-cluster intervals with
  explicit undefined resamples and separate reviewers. Descriptive agent labels only.
- [Offline review page](docs/offline-review.md): inspect blind candidates beside
  their sources, save drafts, and export complete responses for validation.
  It does not create independent human labels or satisfy Gate A.
- [Fixed context budgets](docs/context-budget.md): measured evidence retention
  after packing identical rankings under UTF-8 byte ceilings. This is a retrieval
  diagnostic; answer accuracy and tokenizer counts are not measured.
- [Local claim ledger](docs/ledger-brief.md): implemented early Phase 4 engineering and
  its evidence-retention and current-fact boundaries.
- [Exact citation retrieval](docs/citation-resolution-brief.md): resolve a cited
  historical source revision after later imports and store reopen.
- [Roadmap](docs/roadmap.md) and [handoff](docs/handoff.md): implemented boundaries
  and the next smallest experiment.

Private data belongs in `.local/`, never in tracked fixtures. Existing Context Kit,
provider logs and memory behavior remain unchanged. The original transcripts remain
authoritative; derived indexes can be rebuilt. Source revisions do not resolve
semantic contradictions or establish current real-world facts.

Code and authored synthetic examples use the [MIT license](LICENSE).
