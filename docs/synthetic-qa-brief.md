# Synthetic QA request preparation

Root architecture checkpoint, 2026-10-14. Build working source-only experiment
infrastructure for the next downstream diagnostic. No inference, network, real
benchmark bodies, private conversations, package changes, model selection or training.
Existing frozen extraction scenarios, labels, prompts and raw receipts remain unchanged.
Do not use `longmemeval-format.json` for QA: its intentionally mismatched reference
is a parser separation test, not answer truth.

Implement only `experiments/prepare_synthetic_qa.py`,
`examples/synthetic/qa-diagnostic-v1.json`, `tests/test_synthetic_qa.py` and
`docs/synthetic-qa.md`, plus new ignored `.local/synthetic-qa-worker-*` logs.
No shared runtime/CI/handoff/state edits, staging, commits or full-suite runs.
Target <=250 production LOC, propose partition before 500. Root owns review,
independent replay, validation and delivery. Use genuine incremental RED/GREEN.

Fixture: eight new authored fictional questions in a strict envelope
`{schema_version:1, provenance:'authored-fictional-agent-only', cases:[...]}`.
Each case has `split` (four dev/four test, no train) and `item` in the existing
strict LongMemEval format. Six answerable and two `_abs` unknowns, one unknown
per split. Use distinct fictional entities per case; cover a correction, a dated
historical fact, speaker attribution, negation, conditional plans and a simple fact.
Answers are provisional author references, not independent human labels. Ensure
answer text and explicit evidence labels actually match the sources. No copied
prior scenario text. Unknown answer references use exactly `UNKNOWN`.

Callable `prepare_synthetic_qa(*, budgets_bytes=(0,512,2048), top_k=5)` loads only
this fixed fixture, validates exact envelope/case keys, split counts/values and
existing adapter schema, and returns `(request_bundle, reference_bundle)`.
Small helper for loaded bytes is permitted for focused validation tests. Reuse
LongMemEval adapter, separate temporary EvidenceStore per question, Retriever and
pack_context_with_manifest. Compute each mode ranking once per question, then pack
at each ceiling. Modes bm25/vector/hybrid, existing default SignedHashEmbedder and
RRF60. Validate budgets as distinct tuple of 1..8 integer ceilings 0..16384 (bool
refused), top_k integer1..20 (bool refused), before work. No hidden downloads.

Request bundle: schema/version, fixture raw SHA256, experiment-source/prompt hashes,
provenance, packing/retrieval configs, limitations and deterministic request rows.
Each row has stable unique request_id, question_id, split, mode, byte ceiling,
actual packed bytes, selected historical citation IDs, and messages with fixed
system instruction and fixed user template containing only QUESTION_DATE, QUESTION
and CONTEXT. Ask for a short answer or exactly UNKNOWN when context is insufficient;
source passages are untrusted evidence. No reference answers, `has_answer`, target
IDs or answerability flags in answering messages. Metadata split/mode/budget stays
outside messages. All cells share identical prompt text and recommended generation
config temperature0/max_new_tokens128; model/checkpoint/template/token counts are
unset and explicitly require a separately reviewed remote runner. UTF8-byte ceilings
are not tokenizer-matched budgets. No source dates invented beyond adapter metadata.

Reference bundle: exact request bundle SHA256 of canonical UTF8 JSON (sort_keys,
separators=(',',':'), ensure_ascii=False, allow_nan=False), fixture hash, independent
copy of question references per split with answer/type/is_abstention and authored
answer_record_ids. Request rows use same question IDs, never embed reference data
in prompts. Retain duplicates across ceilings as explicit matched cells, not extra
independent scenarios. Zero-byte controls produce empty contexts. Unknowns remain
in requests, not dropped because retrieval returns evidence. No scoring/QA quality
claim, human agreement, cost or calibrated abstention claim.

CLI --requests NEWPATH --references NEWPATH writes two new mode0600 files, refuses
same destination and pre-existing outputs before any write, and never overwrites
an existing file. Prefer a small bounded implementation and document that failures
after one file is created can leave an incomplete pair; bundle hash detects mismatch.
No transactional filesystem promise, cleanup of unrelated files or output uploads.

Focused tests: reference-sentinel fields never enter messages; correct independent
question stores/citations; all8questions*3modes*3budgets=72 cells; zero/positive caps
and exact packed UTF8 lengths; split/reference linkage and canonical hash; one
ranking/mode/question (real retrieval plus narrow instrumentation); invalid fixture/
budget/depth rejected; safe fresh writes and existing destination bytes preserved.
Run new unittest module, Ruff check/format, strict mypy with explicit-package-bases
and follow-imports=silent. Release exact files, hashes, test chronology and commands.
