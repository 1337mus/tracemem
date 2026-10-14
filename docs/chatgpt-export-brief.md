# Explicit local ChatGPT export adapter

2026-10-14. Follow-on Phase 1 engineering. No private exports have been located
or read. This implements a documented, conservative fixture contract, not a
claim of complete compatibility with every current or historical export.

Own only `src/tracemem/adapters/chatgpt.py`, `tests/test_chatgpt_adapter.py`,
and `docs/chatgpt-export.md`. Do not change JSONL adapters, schemas, CLI,
Context Kit discovery, existing tests or fixtures. No auto-discovery, network,
models, source writes, migrations, training or private-data ingestion.

Expose `read_chatgpt_export(path, *, stats=None)` returning an iterator of
EvidenceRecord, imported explicitly from tracemem.adapters.chatgpt. Reuse the
existing AdapterStats type. Input is UTF-8 JSON, a list of conversation objects
with nonblank string `id`, `current_node` and `mapping`. Each node has `parent`
(null or mapping key), and null `message` or a message object. Follow only the
current_node parent chain; validate the complete chain before emitting any of
that conversation. Reject malformed chains (missing parent, missing node,
cycles, invalid pointer types) without falling back to arbitrary branches.
Preserve root-to-leaf order. Do not infer a branch from timestamps or children.
Invalid conversations count malformed and emit no records. Duplicate nonblank
conversation IDs within one export invalidate the export with a sanitized
ValueError before any yield, so unrelated branches cannot share identities.

Use source='chatgpt', session_id=conversation.id, message_id=nonblank message.id
or node key, and original parts-list index as part_index. Detect repeated
message IDs on a selected chain and reject that conversation before yielding.
This prevents contradictory nodes from collapsing to one identity. Only
content_type='text' with parts list is supported; emit strings verbatim,
including Unicode/newlines/empty text, skip nonstrings with unsupported count.
Do not serialize structured blocks, attachments, tool calls or images into text.
Skip unsupported content types and null messages with unsupported count;
malformed message/author/content shapes increment malformed. Supported author
roles use EvidenceRecord.ROLES; unrecognized role is unknown. All user/unknown
text is mixed_untrusted; assistant conversation; tool tool_output;
system/developer instruction. Text is untrusted and never executed.

Message create_time is optional finite numeric Unix seconds (exclude bool);
convert UTC to ISO8601 Z, preserving fractional seconds when present. Missing
or invalid/out-of-range time stays None; do not substitute conversation time or
derive fact validity. A path-derived SHA256 source_locator hides the path;
line_number=None because JSON export offsets are not JSONL source lines.
Do not normalize, concatenate or strip message text. Unmodified exact strings
remain citeable by Python codepoint offsets and immutable evidence revisions.

Bound whole-file input to 64 MiB: open read-only binary, read(limit+1), reject
oversize before JSON decode; no unbounded read. Reject duplicate JSON keys,
NaN/Infinity, malformed JSON/UTF8 and non-list root with sanitized exceptions
that never contain source text/paths/IDs. Do not introduce dependencies.

Use TDD one behavior at a time; first RED should call the explicit reader on
a minimal selected branch and fail on absent behavior via getattr, not import.
Then test a real branched mapping (excluded sibling never emitted), reversed
mapping insertion order, multi-conversation IDs, null root, malformed/cyclic
chain, duplicate message IDs, exact Unicode text, original part-index stability,
timestamp handling, unsupported blocks/roles and bounded-read rejection. Verify
source bytes/hash/mtime unchanged and no neighboring output/sidecars. Include
an actual temporary EvidenceStore ingest/reopen/citation after edited export
that preserves record_id but advances revision. Never use real exports in tests.

Run focused new tests + existing adapter/citation tests, Ruff and source mypy.
No full-suite or wheel rebuild; root owns integration after final diff review.
Document Python API example, selected-branch semantics, memory cap, trust/format
limits and no actual-export validation. Return exact RED/GREEN and limitations.
