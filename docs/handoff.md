# TraceMem continuation

Updated 2026-10-14. Verify by: 2026-10-15. Read `.local/active-cycle.json`
first, then verify live git/workers/CI. This is a prototype checkpoint, not overall
completion. Existing `continue-tracemem-prototype` heartbeat continues this chat;
do not create a duplicate. History remains in validation documents and git.

## Authority and execution

Workspace `/Users/raj/Projects/tracemem`; original specification
`/Users/raj/Downloads/TRACEMEM_AGENT_BRIEF.md` was read entirely. Reviewed code,
docs and fictional fixtures may be signed and pushed to PRIVATE personal
`1337mus/tracemem`. Assert PRIVATE immediately before every push. No publication,
private-data upload, destructive change or expensive training without separate
approval. Context Kit `/Users/raj/Projects/ctxkit`, original conversations and
`~/.llm-context` remain read-only. Keep private text/claims/vectors/checkpoints in
ignored mode-0700 `.local`, mode-0600 files. No persistent private corpus imported.
No model weights, Torch, Transformers or inference on this Mac. Six historical
fictional Modal calls completed and apps stopped; do not relaunch them. No GPU
needed for current slice. Computer Use is unused; browser rendering unverified.

Preparation/TDD/model routing/handoff workflow applies. Root architects, adjudicates,
reviews every diff and independently verifies; bounded Luna/medium workers implement
from file-level briefs. Do not duplicate active workers. Full affected suite once
at final boundary; focused checks for narrow review fixes, then CI full rerun.
Prototype scope requires no Linear/PR. Preserve historical raw receipts and prompts.

Signed commits use:
`git -c gpg.program=/Users/raj/Projects/tracemem/.local/gpg-noninteractive.sh commit -S`.
Local signature verification succeeds when allowed to read the GPG trust database.
GitHub reports `bad_email`; do not change identity/signing settings or rewrite history.
Git/network writes require the tool's sandbox escalation. Never hide failing exit
codes behind pipes or later commands.

## Latest delivered source and validation

Signed source `1f1a54083c8e140d12e2d668881cde239dd5d608` pushed PRIVATE.
Hosted CI [38642368196](https://github.com/1337mus/tracemem/actions/runs/38642368196)
passed: 397 tests, one expected Context Kit skip; Ruff 90 files; source mypy 31;
145 installed tests with the same expected skip; frozen budget and HTML smoke.
Exact log hash and signature facts: `docs/github-validation.json`.

Latest slice adds optional `read_transcript(..., require_quiescent=True)` and
`ingest --require-quiescent`. Regular nonsymlink leaf only; nonblocking descriptor
opening; compare path/descriptor dev, inode, size, mtime, ctime and mode before
read and after natural EOF. Ordinary change or replacement refuses import and
EvidenceStore rolls back logical revisions/heads. Default streaming behavior is
preserved. Early close releases descriptor without checking unconsumed remainder.
Not an atomic snapshot or adversarial metadata-restoration defense. New empty
schema may remain after refused import. See `docs/transcript-consistency.md`.

Local full boundary: 396 tests, zero skips, 10.600s; Ruff 90 files, source mypy31,
scoped mypy47, actionlint and diff check. Root then found a self-loop symlink error
gap. Genuine RED/GREEN fix added one test, preserved default RuntimeError, and
passed25 focused tests with one optional Context Kit skip, Ruff and affected mypy.
Final hosted full suite397 passed. Root final17-check mutation/rollback oracle plus
self-loop/default assertion passed. Worker EOF mutation tests followed implementation;
do not claim per-behavior preimplementation TDD for the entire slice.

Approved manifest `.local/package-source-approved-transcript-consistency.json`
binds36 files: adapter init and CLI changed, `_transcript_source.py` added,
33prior runtime files unchanged. Offline wheel SHA256
`3d8263b7b97fa4f9ac28b204cb285beea4ac2dedf451bb4dcfaf9216409df637`.
145copied installed tests passed with one expected Context Kit skip outside checkout,
PYTHONPATH/HOME unset. Frozen budget report and byte-identical84-task HTML reproduce.
Validation `docs/validation-transcript-consistency.json`; wheel receipt
`.local/package-smoke-fts-v3-chatgpt-20261014T214119419628Z.json`; installed receipt
`.local/transcript-consistency-installed-receipt.json`. Inspect/approve runtime bytes
before any new wheel; use `.local/package-smoke-fts-v3-chatgpt.py` with approved manifest.

## Existing prototype and evidence boundaries

Phase0/1 runnable. Context Kit clean at `20096bcb1095883262c77a37bbbb458732d9cf3a`
(last rechecked this cycle); no conversation/vector DB or ChatGPT importer. Reuse
config/discovery, read original transcripts separately, no write-capable ctx commands.
`docs/context-kit-inventory.md` records schemas and aggregate coverage.

Provider-neutral immutable evidence revisions, mutable last-imported heads,
schema3 SQLite FTS5 BM25, signed-hash vectors/RRF, optional pinned remote MiniLM
vectors, exact historical citations and UTF8-byte context packing are implemented.
Default package has no dependencies. Schema2 is rejected read-only, no migration.
Claude/Codex JSONL defaults8MiB per raw LF/CRLF line, override up to64MiB; CR-only
separators unsupported. No total-memory guarantee. `docs/transcript-bounds.md`.
ChatGPT selected-parent-chain reader/fresh snapshot CLI is fictional-fixture verified;
actual-export compatibility unverified. `docs/chatgpt-export.md`, `docs/chatgpt-snapshot.md`.

Proposal ledger records REVIEW/REJECT claims, exact quotes and lineage. No automatic
semantic acceptance or current-fact truth. Pure source-retention planner handles
historical revisions, citation links, ambiguous spans and complete descendants.
Persisted-store report reads bounded64MiB quiescent rollback-mode DB copies into
memory; rejects WAL/journal sidecars, validates integrity, writes a new private report.
No deletion, complete ownership proof, cross-store snapshot or external-copy audit.
`docs/source-retention.md`, `docs/retention-report.md`.

Frozen48fictional scenarios:24dev/24test,43 provisional nonexhaustive targets,
eight no-memory, zero train/human labels. GateA unmet; no model selected/trained.
Dataset/prompt/payload/runner hashes in active-cycle; historical launches in its
history_archive. Never tune/select/salvage on frozen test outputs. Qwen/Phi diagnostic
and six Modal calls are completed. Qwen dev1schemafailure/41claims,test2/42; Phi dev
one parse+23schemafailures,test23schemafailures+one unsupported end-date claim.
84valid candidates: blind agent50supported/31unsupported/3uncertain; separate nonblind
root41/33/10 after nine overrides. Neither is human truth or consensus. NLI showed
confident false positives; no verifier policy. `docs/experiments-scenarios.md`,
`docs/scenarios-reviewed-support.json`, `docs/scenarios-root-adjudication.json`.

Integrity-versus-support replay, overlapping scenario-tag breakdown and scenario-cluster
bootstrap preserve model/split/reviewer distinctions, uncertainty and exclusions.
Tags describe whole scenarios, not error causes; bootstrap is descriptive, not model
selection. Simple16-control agent agreement is not calibration.
`docs/integrity-selectivity.md`, `docs/scenario-phenomena.md`, `docs/review-uncertainty.md`.

Fixed context-budget diagnostic:14answerable/3unknown,3modes,6UTF8-byte ceilings.
At2048bytes authored sufficiency BM25=14/14,hash10/14,hybrid13/14; not QA or tokenizer
counts. Unknowns receive nonempty packs512bytes+. `docs/context-budget.md`.
Offline84-task blind reviewer has NodeVM/roundtrip/Unicode checks; browser unverified.
FinalHTML SHA256 `07521f971f5554e5476d15a1299eb75cf0b5b20fbc6854c79b6053a1aecc474e`.
`docs/offline-review.md`.

LongMemEval bytes adapter, bounded whole-file scanner and selected-question turn
retrieval diagnostic are fictional-format verified, not real benchmark/QA results.
Explicit turnlabels only; abstention/missing labels excluded separately. No benchmark
bodies downloaded. HF card declares MIT at pinned commit; constituent rights unresolved.
LoCoMo NC restrictions and exact evaluator differences remain documented. Graphiti
source/license contracts audited, no upstream execution. `docs/longmemeval-file.md`,
`docs/longmemeval-selection.md`, `docs/benchmark-contracts.md`, `docs/related-work.md`.

## Synthetic QA checkpoint and next action

New source-only QA preparation is implemented and root reviewed. Eight new authored
fictional cases have four dev/four test questions, six answerable and two unknowns.
72request cells cover three retrieval modes and byte ceilings0/512/2048; zero-byte
controls are empty. References stay separate and bind the canonical request hash.
Root independently replayed72contexts/citation lists/byte lengths and eight reference
rows, then changed answer/label sentinels and confirmed every answering message
remained identical. Provisional answers were source-reviewed by root, not humans.
No inference/scoring or tokenizer-matched-budget claim. See `docs/synthetic-qa.md`.

Full local boundary407tests,zero skips,10.607s; Ruff92files,source mypy31,scoped48,
actionlint/diffcheck pass. Ten copied installed protocol tests pass outside checkout,
PYTHONPATH/HOME unset, and both canonical bundles reproduce exactly. All36approved
runtime files unchanged; previous wheel reused. `docs/validation-synthetic-qa.json`.
Worker chronology is an honest transcription, not raw RED outputs. Initial callable,
source-identity and invalid-split corrections had RED/GREEN; broader tests followed
initial implementation. Do not claim per-behavior preimplementation TDD.

Canonical requests SHA256
`85d1334d0401ef2bdc920bf316f066bc3e74e9423ee5bec0aa63f4a559be2084`;
references `70769ec36ac4c1e6a226d7432ea88241225e35d91d803cdc0a678fe2589226d2`.
CLI created two new0600files; raw paths/hashes in
`.local/synthetic-qa-command-receipt.json`; root/installed receipts in
`.local/synthetic-qa-root-receipt.json` and `.local/synthetic-qa-installed-receipt.json`.
Signed source/private push/hostedCI are pending at this documentation checkpoint.
No workers or GPU calls active. Original sources and frozen extraction data unchanged.

Next: `docs/synthetic-qa-modal-brief.md` pins a bounded source-free Qwen3-4B runner
for the exact reviewed72messages. Start with one bounded Luna implementer; no launch
during implementation. Root reviews source/payload/SDK serialization, verifies live
personal Modal workspace and absent active jobs, saves launch intent and exact hashes,
then may run one bounded synthetic-only call under existing authorization. Keep
weights/inference remote. No private data or references in remote payload. No retries,
training or public publication. Inspect exact source and live state before acting.

Do not use `examples/synthetic/longmemeval-format.json` as QA truth: its deliberately
mismatched answer/turnlabel tests parsing separation. New synthetic QA references
remain provisional. A later answering run must report actual tokens and preserve
byte-budget limitations; it cannot establish real benchmark or matched-token QA.

Remaining substantive gates: independent human labels/GateA, calibrated semantic
verification, temporal truth, owner-authorized erasure, real benchmark rights and
compatibility, matched-budget downstream QA, and eventual owner-reviewed release.
Continue useful bounded work; this checkpoint does not mark the project complete.
