# Conservative LongMemEval format adapter

Updated 2026-10-14. Prototype only; root owns architecture/review. Read the pinned
README schema lines75-88 in .local/benchmark-contracts/LongMemEval/README.md and
its source receipt/docs/benchmark-contracts.md. Do not fetch benchmark data or
execute upstream code. Dataset-specific rights and real-data compatibility remain
unverified. This is fixture-based engineering toward raw-history retrieval baselines.

Worker owns ONLY src/tracemem/adapters/longmemeval.py,
tests/test_longmemeval_adapter.py, examples/synthetic/longmemeval-format.json,
docs/longmemeval-adapter.md, plus .local logs. No other edits, commits, pushes,
network/GPU calls, dependencies, original-data writes or real corpus reads.
Use strict TDD; return ownership and exact focused evidence, root runs full checks.

Implement parse_longmemeval_bytes(raw: bytes, *, namespace: str, variant: str)
returning tuple of frozen, slotted dataclasses. Require explicit namespace and
variant in {s,m,oracle}; stable namespace comes from caller, not whole-file hash.
Input max64MiB (document conservative fixture contract, not full dataset support).
Strict UTF8/JSON: reject duplicate keys, NaN/Infinity, recursion/huge-integer parse
errors and lone surrogates with sanitized ValueError messages. All-or-nothing
parse; no lazy partial output and no file writes. No file/path wrapper or CLI yet.

Expected top-level list of exact question objects with README's nine fields:
question_id,question_type,question,answer,question_date,haystack_session_ids,
haystack_dates,haystack_sessions,answer_session_ids. Reject extra/missing fields,
wrong types and duplicate question IDs. Six documented question_type labels only.
Question metadata strings; question_id/type/question/date nonblank, answer may
be empty. Preserve all strings exactly; validate scalar Unicode without trimming.
Haystack IDs/dates/sessions are parallel lists with equal lengths; reject duplicate
session IDs within a question. Sessions are lists of turns with exact role/content
and optional has_answer. Only user/assistant roles, content string (empty allowed),
has_answer strictly bool if present. Answer session IDs unique and contained in
that question's history. Preserve original session order, including oracle.

Dataclasses: LongMemEvalSession(upstream_session_id,date,records tuple);
LongMemEvalQuestion(question_id,question_type,question,answer,question_date,
variant,is_abstention,sessions tuple,answer_session_ids tuple,answer_record_ids tuple).
Expose a read-only records property flattening session records in source order.
is_abstention = question_id.endswith('_abs'); no-answer labels are not semantic
truth. Do not infer evidence labels, timestamps, current facts or missing records.

Use existing EvidenceRecord. source = 'longmemeval:' + SHA256(canonical JSON
[namespace,variant]); session_id = canonical JSON[question_id,upstream_session_id]
(no delimiter collisions), message_id = decimal0-basedturnindex,part_index0,
role preserved,text exact,content_kind conversation,source_locator/line_number None.
recorded_at = None: the README supplies session dates, not message timestamps.
Keep raw session date in dataclass only; downstream use must carry this context.
answer_record_ids consists only of IDs of has_answer=True turns, in source order.
Do not require those flags to exactly cover answer_session_ids, nor infer flags.
Never put question, expected answer, has_answer, question type, question date or
answer IDs into EvidenceRecord text or revisions. Changing only reference labels
or question/answer text must leave existing evidence identities/revisions unchanged.
Scope evidence IDs by question, namespace and variant to avoid accidental merging.
Retriever has no corpus filter: evaluation must use a separate EvidenceStore per
question and index only question.records. Test that a second question's distinct
source text is absent from the first question's actual SQLite store. Unique IDs
alone do not prevent cross-question retrieval contamination.

Write a tiny authored MIT fictional fixture: two questions, one answerable with
Unicode/newline source text and flagged turn, one '_abs' with no evidence labels.
Reuse an upstream session ID across questions with different content to exercise
namespacing. No copied benchmark content. Tests exercise actual EvidenceStore
and Retriever: exact Unicode historical citation resolution, no answer/reference
text indexed, input bytes preserved, stable revision under label/answer changes,
unique IDs across items/variants/namespaces, source order, empty session/history,
strict schema/error cases and no flags inferred. Reuse existing citation API.

Docs must state fixture-only contract,64MiB cap,metadata separation,session-date
context gap and unverified real-data rights/compatibility. No evaluation, recall,
QA accuracy, upstream metric equivalence or Gate A claim follows from this adapter.
