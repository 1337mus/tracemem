# Phase 0/1 implementation contract

Date: 2026-10-13. Local prototype only. No hosted models, publishing or training.

Use Python 3.11+ and SQLite FTS5. The stdlib demo must work offline. Reuse Context Kit's
configuration/source discovery optionally, never its lossy usage-event parser as evidence.
No Context Kit writes. Default demonstrations contain only authored synthetic data.

## Shared API

`EvidenceRecord` is a frozen dataclass with `source`, `session_id`, `message_id`, `part_index`,
`role`, `text`, `recorded_at` (optional), `source_locator` (opaque hash), `line_number`,
`content_kind` and computed `record_id`, `revision`. ID is stable across text edits; revision
hash binds all evidentiary fields. Offsets are Unicode codepoints into one unmodified text block.
Roles: user, assistant, tool, system, developer, unknown. Kinds: conversation, tool_output,
instruction, mixed_untrusted, derived_note. User history is untrusted, never executed.

`read_transcript(path, provider)` yields records; provider is claude/codex. Ignore non-message
events (including duplicate Codex event_msg text); preserve recognized tool outputs separately.
Use native IDs when present, a line/block fallback otherwise. Count malformed/unsupported
records in `AdapterStats`; do not print input data in error messages.
`discover_context_kit(config_path=None)` uses ctxkit.config.load_config and transcript_files
when installed. Return configured sources, fail clearly if dependency/config absent. Explicit
file ingestion works without Context Kit installed. No imports with source-state mutations.

`EvidenceStore(path)` is a context manager. `ingest(records)` returns inserted revision count;
all-or-nothing transaction, idempotent repeated ingest, immutable evidence rows, latest revision
view only for retrieval. `records(current_only=True)` returns a list; `get(record_id, revision)`
returns a record or raises KeyError. `lexical_search(query, limit=10)` returns list[(record, score)]
using parameterized FTS5 BM25; tokenize/quote search terms, never arbitrary model SQL.
Evidence schema version 2 and application_id checked before touching an existing DB.
Evidence uses WITHOUT ROWID to block explicit rowid replacement; historical reads are
ordered by record_id,revision, not ingestion order. New DB
0600 and parent directory 0700 when created; never chmod existing unrelated parent. Reject
symlink DB, and avoid source directories as output. No automatically destructive erase in v1.

`retrieval.py`: embedder protocol `metadata: dict[str,str|int]` and `encode(text)->list[float]`.
Default signed token hashing (fixed sha256, dimensionality, normalization, version metadata)
is explicitly a weak lexical vector baseline, not a pretrained semantic model. Persist derived
vectors separately via store APIs (worker owns these additional APIs); rebuild on fingerprint
change. Optional SentenceTransformer accepts an explicit local model directory and local-only
loading, never downloads. `Retriever(store, embedder=None).search(query, mode='hybrid', k=5)`
returns `SearchHit(record, score, citation)`. Modes bm25/vector/hybrid, reciprocal rank fusion,
exclude zero vectors/no-overlap defaults, stable ties, validate bounds, always use exact source
revision in citations. Citation format record_id@revision:0:len. Context packing includes citation
overhead in a conservative UTF-8 byte budget (explicitly not model-token counts).

## Verification and evidence

Tests exercise live writer shapes: Claude multipart user/assistant/tool_result; Codex
session_meta/response_item messages/function_call_output; native IDs and fallback, partial
JSONL tail, Unicode spans, duplicate ingest, edited sources, rollback, existing unknown DB,
BM25 + vector + RRF, offline synthetic end-to-end citations, source file hash unchanged.
Use unittest. Follow RED then GREEN, grouped guard tables where useful. Avoid test bloat.
Full suite once at final boundary, focused tests during development. Record red/green commands.

No extraction quality, entailment, novelty or >99% precision claims in Phase 1. Synthetic
retrieval benchmark is diagnostic and must include lexical failures, temporal revisions,
assistant decoys and unknown questions. Compare modes with recall@k, MRR, nDCG, sufficient
multi-evidence sets, query latency p50/p95, input hash and environment. No QA model in v1.
